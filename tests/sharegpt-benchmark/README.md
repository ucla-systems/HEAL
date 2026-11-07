# ShareGPT Serving Benchmark

This runner executes vLLM's official serving benchmark path for ShareGPT:
it applies one canonical vLLM patch at a time, starts `vllm serve`, runs
`vllm bench serve`, and collects the resulting metrics.

Default setup:

- models: Qwen3-8B, Qwen3-14B, Qwen3-32B, Llama-3.1-8B-Instruct,
  DeepSeek-R1-Distill-Qwen-14B
- dataset: `output/sharegpt_bench/data/sharegpt_vicuna_subset4096.json`
- prompts: 1024
- `--max-model-len 8192`
- `--max-num-batched-tokens 8192`
- `--max-num-seqs 256`
- request rate: `inf`
- cases: Original FP16/BF16, LayerCast FP32/TF32, HEAL base/plus/full
- all cases, including Original FP16/BF16, pass `--attention-backend TRITON_ATTN`
- all HEAL-Full cases use `HEAL_FP16_GEMM_COMPENSATION=1` for the FP16
  compensation GEMM path, decomposing `x` as `x1 * 32 + x2 / 32`.
- runner-level `--heal-skip-qkv-quant-layers` materializes HEAL++ by selecting
  HEAL+ layers that fall back to HEAL attention/QKV handling. Runner-level
  `--heal-skip-fp16-comp-layers` materializes HEAL-FULL+ by selecting
  HEAL-Full layers that fall back to HEAL+ MLP handling; it also exports the
  patch runtime alias `HEAL_SKIP_INT8_COMP_LAYERS`.

Run the full experiment:

```bash
python tests/sharegpt-benchmark/run.py \
  --all-models \
  --run-id sharegpt_$(date -u +%Y%m%dT%H%M%SZ)
```

Smoke/dry-run without starting vLLM:

```bash
python tests/sharegpt-benchmark/run.py \
  --model qwen3-8b \
  --cases original_bf16 layercast_tf32 heal_base \
  --dry-run
```

Raw serving outputs go under `output/sharegpt-benchmark/<run_id>/...`.
Aggregated JSON/CSV results go under
`tests/sharegpt-benchmark/results/<run_id>/`. The summary includes
`memory_usage_gib`, defined as vLLM model-loading memory plus CUDA graph memory
(`model_loading_memory_gib + graph_capture_memory_gib`), matching the table
column `GPU Memory (GiB)`.


## HEAL Policy Reuse

`heal_plus_plus` and `heal_full_plus` are the policy-applied ShareGPT cases for HEAL++ and HEAL-FULL+. Pass `--heal-policy-json tests/error-contribution/policies/<run>.json` to load per-model skip-layer lists. The baseline `heal_plus` and `heal_full` cases remain non-downgraded unless explicit manual skip-layer flags are supplied. In paper tables, the compact `HEAL-Full` row may be populated from the tuned `heal_full_plus` case when reporting HEAL-FULL+. The LaTeX memory column is `GPU Memory (GiB)`.
