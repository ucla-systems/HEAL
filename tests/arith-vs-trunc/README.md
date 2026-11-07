# arith-vs-trunc

Operator/kernel-level truncation analysis for MedQA using the active
`VLLM_DUMP_TENSORS` patch. It does not modify or require any new vLLM patches.

## Scope

The runner covers the summary figure families plus the kernels that matter for
self-attention:

- GEMM/Linear: Q/K/V projection, O projection, MLP gate/up projection, MLP down
  projection, and optional `lm_head`; the BF16 path uses the HEAL cuBLASLt
  FP32-output GEMM op during analysis.
- Attention: full QKV causal attention using dumped full replay context for K/V.
  QK and PV are replayed with Triton `tl.dot(..., input_precision="tf32")` rather
  than dense `torch.matmul`.
- RoPE: Q and K rotary outputs.
- RMSNorm: decoder input/post-attention norms, Q/K norms when present, and final
  norm.
- SiLU-and-mul: MLP activation.

For each supported operator/kernel and selected generated-token row, the main
reported gap is the lower-precision-input FP32 truncation MSE. For dumped
operator outputs, the reference is the FP32 LayerCast dump. For computed
Attention QK/PV kernel outputs, the reference is the same kernel recomputed in
FP32 on GPU from the dumped full-context FP32 inputs.

The primary CSV column for the lower-precision-input FP32 truncation gap is
`lower_precision_fp32_mse_mean`; additional BF16/trunc delta columns are
written for the interval plots.

## Default Run

The default uses the first 8 rows from the error-contribution MedQA calibration
set, generates exactly 1024 tokens with FP32 LayerCast by using `ignore_eos`, then full-prefill replays
the exact prompt+generated token trajectory while dumping full replay rows for
each document. Analysis still only scores the selected generated-token rows, but
full rows are needed for Attention QK/PV context. Replay dumps are flushed into
one safetensors segment per document so the trace buffer never holds the full
run in memory.

```bash
python3.12 tests/arith-vs-trunc/run.py \
  --model qwen3-8b \
  --limit 8 \
  --max-new-tokens 1024
```

Before running, `vllm/` must be clean. The runner applies
`VLLM_DUMP_TENSORS` with `scripts/vllm_incremental_build.sh prepare-version`
and reverts it at the end. Use `--dry-run` to write the manifest only.

## Outputs

Raw artifacts go under:

```text
output/arith-vs-trunc/<run_id>/<model>/
```

Curated summaries go under:

```text
tests/arith-vs-trunc/results/<run_id>/<model>/
```

Important files:

- `generated_tokens.jsonl`: FP32 LayerCast generated trajectories.
- `trace_segments/segment_*.safetensors`: per-document dumped FP32 LayerCast
  operator outputs for full replay rows.
- `trace_segments.jsonl`: segment paths and row ranges.
- `trace_rows.jsonl`: row order metadata for the dumped generated-token rows.
- `arith_vs_trunc_summary.csv/json`: per-operator/kernel truncation gaps.
- `arith_vs_trunc_points.csv`: optional per-token rows when `--write-points` is
  passed.
- `arith_vs_trunc_mse.{png,pdf}` plus residual, BF16/trunc delta, and
  Shapley family interval outputs: `arith_vs_trunc_family_intervals.*`,
  `arith_vs_trunc_bf16_trunc_delta_family_intervals.*`, and
  `arith_vs_trunc_shapley_family_intervals.*`.
