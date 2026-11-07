# flip-rate-analysis

Direct non-eager generation flip-rate runner for the MedQA prompt set captured by `tests/final-hidden-mse`.

The runner uses an existing full-FP32 LayerCast generation artifact as the prompt/reference source, then runs the same unique precision families as the final-hidden MSE test with ordinary `LLM.generate()`:

- LayerCast FP32 and TF32
- Original FP16 and BF16, both forced through `TRITON_ATTN`
- Significand k03-k23
- HEAL base, plus, and full; HEAL-Full uses
  `HEAL_FP16_GEMM_COMPENSATION=1`, using the `x1 * 32 + x2 / 32` FP16 split.
  Runner-level `--heal-skip-qkv-quant-layers` materializes HEAL++ by selecting
  HEAL+ layers that fall back to HEAL attention/QKV handling. Runner-level
  `--heal-skip-fp16-comp-layers` materializes HEAL-FULL+ by selecting
  HEAL-Full layers that fall back to HEAL+ MLP handling; it also exports the
  patch runtime alias `HEAL_SKIP_INT8_COMP_LAYERS`.

It does not enable final-hidden tensor tracing and does not force eager execution by default. Each mode generates up to 4096 tokens per sample, records the actual generated token IDs, decodes each MedQA answer as `A`, `B`, `C`, `D`, or `F`, and writes answer-flip metrics against the reference artifact answer. `F` means parsing failed or generation hit the max-token limit.

MCR-Bench reference artifacts use `--answer-parser mcr-bench`, which reads the `target_kind` metadata written by `tests/final-hidden-mse/run.py --dataset-kind mcr-bench`.


Run the full experiment:

```bash
python tests/flip-rate-analysis/run.py --all-models --reference-generation-dir output/final-hidden-mse/<reference_run>/generation_artifacts --run-id flip_rate_$(date -u +%Y%m%dT%H%M%SZ)
```

Outputs:

```text
output/flip-rate-analysis/<run_id>/<model_slug>/<mode>/generated_tokens.jsonl
output/flip-rate-analysis/<run_id>/<model_slug>/<mode>/answers.jsonl
tests/flip-rate-analysis/results/<run_id>/<model_slug>/metrics.csv
tests/flip-rate-analysis/results/<run_id>/<model_slug>/metrics.json
```

Before running, `vllm/` must be clean and no incremental-build patch should be active. The script uses `scripts/vllm_incremental_build.sh prepare-version` and `revert-version`, matching the final-hidden MSE workflow.


## HEAL Policy Reuse

`heal_plus_plus_generate` and `heal_full_plus_generate` are the policy-applied aliases for HEAL++ and HEAL-FULL+. Pass `--heal-policy-json tests/error-contribution/policies/<run>.json` to load per-model skip-layer lists exported from error-contribution tuning. The baseline `heal_plus_generate` and `heal_full_generate` modes remain non-downgraded unless explicit manual skip-layer flags are supplied.
