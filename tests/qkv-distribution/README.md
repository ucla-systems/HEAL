# qkv-distribution

BF16 MedQA Q/K/V dynamic-range distribution experiment.

This runner applies `VLLM_QKV_DISTRIBUTION`, enables eager execution, runs the
selected MedQA subset for each selected model, and records the per-token
`max(abs(x)) / std(abs(x), correction=0)` value for the Q, K, and V tensors
passed directly into `self.attn(q, k, v)`. Q/K are captured after Q/K norm and
RoPE where those operations exist; V is the corresponding projection-split V.
The hook locations mirror the attention-feeding Q/K/V positions used by
`VLLM_DUMP_TENSORS`, but this patch stores only scalar dynamic-range statistics
instead of full tensors.

Default setup:

- models: Qwen3-8B, Qwen3-14B, Qwen3-32B, Llama-3.1-8B-Instruct,
  DeepSeek-R1-Distill-Qwen-14B
- dataset: `tests/error-contribution/subsets/medqa_calibration32_seed20260528.jsonl`
  by default; pass `--subset-file` for MedQA-200
- dtype: BF16
- execution: `enforce_eager=True`, CUDA graph disabled, V1 multiprocessing disabled
- default max generated tokens per document: 4096

Run the full experiment:

```bash
python tests/qkv-distribution/run.py \
  --all-models \
  --run-id qkv_distribution_$(date -u +%Y%m%dT%H%M%SZ)
```

Smoke plan without applying vLLM patches or launching GPU code:

```bash
python tests/qkv-distribution/run.py --model qwen3-8b --doc-limit 1 --dry-run
```

Rebuild plots from an existing run:

```bash
python tests/qkv-distribution/run.py --plot-only --run-id <run_id>
```

Raw tensors and generation summaries are written under:

```text
output/qkv-distribution/<run_id>/<model>/
```

Plots, per-model summaries, and combined CSVs are written under:

```text
tests/qkv-distribution/results/<run_id>/
```

`--plot-only --data-root <dir>` accepts any completed qkv-distribution output
root where each model directory contains `qkv_absmax_over_absstd.pt`.
