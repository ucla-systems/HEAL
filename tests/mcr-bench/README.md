# MCR-Bench Tests

This directory adapts the active final-hidden-MSE and flip-rate-analysis tests
to the local MCR-Bench datasets under `MCR-Bench/`.

The wrapper does not duplicate model or vLLM execution logic. It calls:

- `tests/final-hidden-mse/run.py` with `--dataset-kind mcr-bench`
- `tests/flip-rate-analysis/run.py` with `--answer-parser mcr-bench`

That keeps model selection, batching, HEAL/HEAL++ knobs, and output formats
aligned with the MedQA tests. MCR-Bench defaults to a 4096-token generation
budget and the reported configurations: `original_bf16`, `original_fp16`,
`layercast_tf32`, `layercast_fp32`, `heal_base`, `heal_plus`,
`heal_plus_plus`, and tuned `heal_full_plus` reported as HEAL-Full. Untuned
HEAL-Full is still available through explicit mode lists.

## Datasets

- `MCR-Bench/combined_mcr_bench.jsonl` (default `--domains all`)
- `MCR-Bench/med_data/combined_med_data.jsonl`
- `MCR-Bench/legal_data/combined_legal_data.jsonl`
- `MCR-Bench/financial_data/combined_financial_data.jsonl`

Dry run:

```bash
python3.12 tests/mcr-bench/run.py --model qwen3-8b --dry-run
```

Standard ops entrypoint:

```bash
bash tests/ops/05_mcr_bench.sh
```

## Output Locations

The default wrapper writes generation artifacts to the final-hidden runner
location:

```text
output/final-hidden-mse/<run_id>_<domain>_final_hidden_mse/generation_artifacts/
```

The standard `tests/ops/05_mcr_bench.sh` entrypoint instead uses a shared
portable directory:

```text
output/mcr-bench/<run_id>/generation_artifacts/
```

The flip-rate stage then uses that directory as its FP32 LayerCast reference.
For multi-model runs, each model is stored under the shared generation directory
using the same model-label subdirectories as `tests/final-hidden-mse`.

Table export is optional. When requested,
`tests/mcr-bench/build_mse_flip_table.py` writes generated reporting files such
as:

- `tables/data/mse-flip-rate/final-hidden-mse-mcr-bench-<domain>-<gpu>.csv`
- `tables/data/mse-flip-rate/answer-flip-rate-mcr-bench-<domain>-<gpu>.csv`
- `tables/data/execution-time/mcr-bench-<domain>-<gpu>.csv`
- `tables/mse-flip-rate.tex`
- `tables/execution-time.tex`

`tables/` is an output destination, not required input data.

## HEAL Policy Reuse

This wrapper forwards `--heal-policy-json` to both underlying runners. Use
`heal_plus_plus_*` and `heal_full_plus_*` modes for HEAL++ and HEAL-Full+; keep
`heal_plus_*` and `heal_full_*` for the non-downgraded baselines.
