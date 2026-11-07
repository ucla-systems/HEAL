# final-hidden-mse

Self-contained MedQA final-hidden MSE runner. It switches vLLM
implementations only through `scripts/vllm_incremental_build.sh
prepare-version` / `revert-version`.

Default bundled dataset:

```text
tests/error-contribution/subsets/medqa_calibration32_seed20260528.jsonl
```

For a full MedQA-200 run, pass the desired 200-question JSONL with
`--subset-file`.

For MCR-Bench, use the wrapper in `tests/mcr-bench/`; it calls this runner
with `--dataset-kind mcr-bench` and the same model/runtime defaults.

Typical one-model run:

```bash
MODEL=qwen3-8b
python tests/final-hidden-mse/run.py \
  --model "${MODEL}" \
  --run-id "${MODEL}_final_hidden_mse_$(date -u +%Y%m%dT%H%M%SZ)"
```

Change only `MODEL` to run a different supported alias, for example
`qwen3-14b`, `qwen3-32b`, `llama-3.1-8b-instruct`, or
`deepseek-r1-distill-qwen-14b`.

Generation and replay can also be split across machines. Run generation on the
first machine:

```bash
MODEL=qwen3-8b
RUN_ID="${MODEL}_generation_$(date -u +%Y%m%dT%H%M%SZ)"
python tests/final-hidden-mse/run.py \
  --model "${MODEL}" \
  --run-id "${RUN_ID}" \
  --stage generate
```

Download this single generation artifact parent directory:

```text
output/final-hidden-mse/<run_id>/generation_artifacts/
```

It is intentionally separate from replay outputs. Each model lives under a
model slug inside that parent directory, and each model directory contains
`generated_tokens.jsonl`, `replay_manifest.jsonl`, `replay_positions.csv`,
`worker_result.json`, and `hidden_states/`. Then replay from that parent
directory on another machine:

```bash
MODEL=qwen3-8b
python tests/final-hidden-mse/run.py \
  --model "${MODEL}" \
  --run-id "${MODEL}_replay_$(date -u +%Y%m%dT%H%M%SZ)" \
  --stage replay \
  --generation-dir /path/to/generation_artifacts
```

For multiple models, pass the same parent directory to both stages:

```bash
RUN_ID="multi_generation_$(date -u +%Y%m%dT%H%M%SZ)"
python tests/final-hidden-mse/run.py \
  --models qwen3-8b qwen3-14b \
  --run-id "${RUN_ID}" \
  --stage generate

python tests/final-hidden-mse/run.py \
  --models qwen3-8b qwen3-14b \
  --run-id "multi_replay_$(date -u +%Y%m%dT%H%M%SZ)" \
  --stage replay \
  --generation-dir "output/final-hidden-mse/${RUN_ID}/generation_artifacts"
```

`--generation-dir` is optional for normal full runs; by default generation
artifacts go under `output/final-hidden-mse/<run_id>/generation_artifacts/`,
while replay artifacts go under `output/final-hidden-mse/<run_id>/<model_slug>/`.
With `--stage replay`, `--generation-dir` points to an existing downloaded
generation artifact parent.

The runner first materializes prompt ids from the chat template, then uses
`VLLM_LAYERCAST` full-FP32 LayerCast to generate actual MedQA completions with
`TokensPrompt(prompt_token_ids=...)`. This keeps the generation prompt token
stream identical to replay, including models such as Llama whose string
tokenizer path can add an extra BOS token. Patched vLLM records per-row
request/token metadata with each `forward.norm.hidden` trace chunk, so
generation hidden rows are split by exact `(request_id, absolute_token_pos)`
instead of inferred scheduler order. The default generation budget is
`max_tokens=4096`, and replay also uses at most the first 4096 generated tokens
per sample unless `--replay-generated-token-limit 0` is passed. Replay manifests
are additionally trimmed for exact self-circulating suffixes: KMP finds a
repeated suffix period, keeps two periods, and drops the extra repeated
tokens/hidden rows. The original finish reason and full token count remain in
the manifest for auditing. Every prefill mode also replays
`TokensPrompt(prompt_token_ids=...)`, captures `forward.norm.hidden`, and
compares against the initial FP32 generation hidden reference with the same
output metrics.

Generation submits the selected MedQA requests together by default
(`--generation-batch-size 200`) and lets vLLM choose the active work.
Generation and replay use separate GPU-memory-utilization defaults. Generation
defaults to `--generation-gpu-memory-utilization 0.93`; the alias
`--gpu-memory-utilization` spelling is an alias for that generation setting.
Replay/prefill defaults to `--replay-gpu-memory-utilization 0.86`, also
available as `--prefill-gpu-memory-utilization`.

The full prefill matrix covers:

- `layercast_fp32_prefill_ref`, `layercast_fp32_prefill`, and
  `layercast_tf32_prefill` from `VLLM_LAYERCAST`
- `original_fp16_prefill` and `original_bf16_prefill` from `VLLM_ORIGINAL`
- `significand_k03_prefill` through `significand_k23_prefill` from
  `VLLM_SIGNIFICAND_BITS`
- `heal_base_prefill`, `heal_plus_prefill`, and `heal_full_prefill` from the
  unified `VLLM_HEAL` patch with `HEAL_MODE=base|plus|full`; HEAL-Full uses
  `HEAL_FP16_GEMM_COMPENSATION=1`, using the `x1 * 32 + x2 / 32` FP16 split.
  Runner-level `--heal-skip-qkv-quant-layers` materializes HEAL++ by selecting
  HEAL+ layers that fall back to HEAL attention/QKV handling. Runner-level
  `--heal-skip-fp16-comp-layers` materializes HEAL-FULL+ by selecting
  HEAL-Full layers that fall back to HEAL+ MLP handling; it also exports the
  patch runtime alias `HEAL_SKIP_INT8_COMP_LAYERS`.

Hidden artifacts are written to:

```text
output/final-hidden-mse/<run_id>/generation_artifacts/<model_slug>/
output/final-hidden-mse/<run_id>/<model_slug>/
```

Token MSE data is written to:

```text
tests/final-hidden-mse/data/<run_id>/<model_slug>/<mode>/
```

Metrics and the Qwen3-8B gate report are written to:

```text
tests/final-hidden-mse/results/<run_id>/<model_slug>/
```

`data/`, `results/`, `run_logs/`, and `__pycache__/` are generated locally and
ignored by `tests/final-hidden-mse/.gitignore`.

Smoke test:

```bash
python tests/final-hidden-mse/run.py \
  --model qwen3-8b \
  --limit 2 \
  --max-new-tokens 8 \
  --prefill-batch-size 2
```


## HEAL Policy Reuse

`heal_plus_plus_prefill` and `heal_full_plus_prefill` are the policy-applied aliases for HEAL++ and HEAL-FULL+. Pass `--heal-policy-json tests/error-contribution/policies/<run>.json` to load per-model skip-layer lists exported from error-contribution tuning. The baseline `heal_plus_prefill` and `heal_full_prefill` modes remain non-downgraded unless explicit manual skip-layer flags are supplied.
