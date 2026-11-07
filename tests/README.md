# Active Test Runners

This directory contains the active experiment runners. It is a runbook for
launching tests and understanding where their outputs go; generated figures and
tables are not required input data.

## Model Set

Unless a task says otherwise, the reported five-model set is:

- `qwen3-8b`
- `qwen3-14b`
- `qwen3-32b`
- `llama-3.1-8b-instruct`
- `deepseek-r1-distill-qwen-14b`

## Output Locations

Runners create artifacts in these locations:

- `output/`: heavy hidden states, generation artifacts, replay tensors, and
  portable run artifacts.
- `tests/<runner>/results/`: compact CSV/JSON summaries.
- `tests/<runner>/run_logs/`: run logs.
- `figures/<gpu_slug>/...`: generated plots from figure builders.
- `tables/...`: generated CSV/JSON/LaTeX table exports from table builders.

`figures/` and `tables/` are reporting destinations. They can be regenerated
from completed run artifacts and do not need to exist before the core tests run.

## Common Workflow

Long runs should normally use the shell wrappers under `tests/ops/`. The
expanded Python commands in each test directory are useful for dry runs,
debugging, or targeted reruns.

The precision implementation is selected by applying one vLLM patch before a
run:

```bash
scripts/vllm_incremental_build.sh prepare-version VLLM_HEAL
scripts/vllm_incremental_build.sh build

# run a test

scripts/vllm_incremental_build.sh revert-version
```

All reported native FP16/BF16, LayerCast, HEAL, and replay paths should record
Triton attention provenance. Native rows without that provenance should be
rerun before they are used in reports.

## Runners

### Final-Hidden MSE

Runner: `tests/final-hidden-mse/run.py`

This runner generates FP32 LayerCast reference answers and final hidden states,
then replays the exact prompt plus generated-token sequence under other
precision modes. Generated reference artifacts usually live under:

```text
output/final-hidden-mse/<run_id>/generation_artifacts/
```

### Flip-Rate Analysis

Runner: `tests/flip-rate-analysis/run.py`

This runner generates answers under selected precision modes and compares them
against reference answers. It shares reference-generation artifacts with the
final-hidden runner when `--reference-generation-dir` is passed.

### MCR-Bench

Runner: `tests/mcr-bench/run.py`

Dataset:

- `MCR-Bench/combined_mcr_bench.jsonl` for `--domains all`
- `MCR-Bench/med_data/combined_med_data.jsonl`
- `MCR-Bench/legal_data/combined_legal_data.jsonl`
- `MCR-Bench/financial_data/combined_financial_data.jsonl`

Dry run:

```bash
python3.12 tests/mcr-bench/run.py --model qwen3-8b --dry-run
```

Standard ops wrapper:

```bash
bash tests/ops/05_mcr_bench.sh
```

Table export is optional. When requested,
`tests/mcr-bench/build_mse_flip_table.py` writes generated CSV/JSON/LaTeX files
under `tables/`.

### ShareGPT Serving

Runner: `tests/sharegpt-benchmark/run.py`

This wraps the official vLLM serving benchmark path. Raw artifacts go under
`output/sharegpt-benchmark/<run_id>/`; compact summaries go under
`tests/sharegpt-benchmark/results/<run_id>/`. ShareGPT table exports, when
requested, are generated under `tables/`.

### Error Contribution

Runner: `tests/error-contribution/run.py`

This replays sampled MedQA generated-token positions from final-hidden
generation artifacts and attributes residual/KV contribution to final-hidden
MSE. HEAL++ and HEAL-Full+ policies are tuned with
`tests/error-contribution/tune_heal_variants.py` and exported as small JSON
policy files.

### QKV Distribution

Runner: `tests/qkv-distribution/run.py`

This runs BF16 eager MedQA generation with CUDA graph disabled and records
dynamic-range statistics for Q, K, and V tensors. Plot builders may write under
`figures/<gpu_slug>/qkv-distribution/`.

### MedQA Figures

Runner: `tests/medqa-figures/run.py`

This consumes completed final-hidden and flip-rate artifacts and writes plots
under `figures/<gpu_slug>/...`.

## Ops Scripts

`tests/ops/` contains reproducible command wrappers for longer experiment
flows. These scripts may create `output/`, `figures/`, and `tables/` artifacts
as needed. Treat those directories as generated results, not as source inputs.
