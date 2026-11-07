# MedQA Figure Aggregation

This directory contains the lightweight plotting entrypoint for the MedQA final figure pair:

- `mantissa_bits_vs_flip_rate`
- `error_vs_flip_rate_by_model_errorbars`

It does not run vLLM. It joins completed `tests/final-hidden-mse` replay metrics with completed `tests/flip-rate-analysis` generation metrics by `(model, setting)` and writes plot data plus PNG/PDF figures under `figures/<GPU_SLUG>/fig-error-vs-flip-and-mantissa-vs-flip/plots/`.

Example:

```bash
python3.12 tests/medqa-figures/run.py \
  --gpu-slug NVIDIA_H100_80GB_HBM3 \
  --final-hidden-run-id "$MEDQA_MSE_RUN" \
  --flip-run-id "$MEDQA_FLIP_RUN"
```

Both `--final-hidden-run-id` and `--flip-run-id` may be passed multiple times; later rows override earlier rows for the same `(model, setting)` pair. Use this when combining separate baseline, HEAL-policy, or significand-sweep runs.

Native BF16/FP16 rows are included only when both joined source rows record `attention_backend=TRITON_ATTN`. This drops native rows whose backend was not pinned.
