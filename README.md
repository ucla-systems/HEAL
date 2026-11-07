# Precision Experiments Workspace

> **Preliminary version.** This repository is a preliminary release of the code
> and data used in the HEAL paper. We will clean up and update it in mid-November
> 2026.

This repository contains the active code and local benchmark data used for
precision-mode experiments around vLLM inference. The public tree is meant to be
small: keep source code, patch definitions, and reproducible benchmark inputs in
git; generate heavy outputs locally.

## Repository Layout

- `MCR-Bench/`: local MCR-Bench JSONL data. The root file contains all
  questions, and each domain subdirectory keeps one combined JSONL file.
- `tests/`: active experiment runners for final-hidden MSE, flip-rate analysis,
  MCR-Bench, ShareGPT serving, QKV distribution, and error contribution.
- `patches/`: standalone vLLM implementation diffs. Each patch is applied to a
  clean `vllm/` checkout before a run.
- `scripts/`: setup, patch, and incremental-build helpers.
- `vllm/`: the editable vLLM checkout or submodule used by the experiments.
- `memory/AGENTS.md`: short maintainer notes.
- `memory/ENVS.md`: NVIDIA/AMD environment setup instructions.

Generated tables and generated figures are not required to run the active
tests.

## Generated Artifacts

The following directories are output locations. They can be absent from a fresh
checkout and recreated by the runners or plotting/table scripts:

- `output/`: heavy generation artifacts, hidden-state dumps, tensors, and
  replay outputs.
- `tests/*/results/` and `tests/*/run_logs/`: compact summaries and logs owned
  by each runner.
- `figures/`: generated plots.
- `tables/`: generated CSV/JSON/LaTeX table exports.

Do not treat `figures/` or `tables/` as benchmark input data. They are useful
for reporting, but the active runners should not need preexisting files there.

## Setup

First prepare the machine environment using `memory/ENVS.md`. Once Python,
CUDA or ROCm, and the basic build dependencies are ready, initialize the vLLM
checkout and run the build helper:

```bash
git submodule update --init --recursive
scripts/vllm_incremental_build.sh setup --confirm-setup-or-configure
scripts/vllm_incremental_build.sh configure --confirm-setup-or-configure
scripts/vllm_incremental_build.sh build
```

For build-helper details, see `scripts/README.md`.

## Patch Workflow

Experiment versions live under `patches/`. Apply one version to a clean vLLM
tree, build it, run the experiment, then revert:

```bash
scripts/vllm_incremental_build.sh prepare-version VLLM_HEAL
scripts/vllm_incremental_build.sh build

# run one or more experiments

scripts/vllm_incremental_build.sh revert-version
```

Useful commands:

```bash
scripts/vllm_incremental_build.sh list-versions
scripts/vllm_incremental_build.sh status
```

## MCR-Bench

The active MCR-Bench data files (4,500 questions, 1,500 per domain) are:

- `MCR-Bench/combined_mcr_bench.jsonl`
- `MCR-Bench/med_data/combined_med_data.jsonl`
- `MCR-Bench/legal_data/combined_legal_data.jsonl`
- `MCR-Bench/financial_data/combined_financial_data.jsonl`

The runner is `tests/mcr-bench/run.py`. A dry run shows the commands without
launching model execution:

```bash
python3.12 tests/mcr-bench/run.py --model qwen3-8b --dry-run
```

Domain-specific runs can be selected with `--domains medical legal finance`;
the default domain is `all`.

## Active Tests

The main entrypoints are documented in `tests/README.md`. In short:

- `tests/final-hidden-mse/run.py`: final-hidden reference generation and replay
  MSE.
- `tests/flip-rate-analysis/run.py`: answer flip-rate generation and parsing.
- `tests/mcr-bench/run.py`: MCR-Bench wrapper around the two runners above.
- `tests/sharegpt-benchmark/run.py`: ShareGPT serving benchmark wrapper.
- `tests/qkv-distribution/run.py`: BF16 eager Q/K/V dynamic-range experiment.
- `tests/error-contribution/run.py`: residual/KV attribution experiment.

Most long experiment commands are wrapped in `tests/ops/`.

## License

The code in this repository is released under the Apache License 2.0 (see
`LICENSE`). The MCR-Bench data in `MCR-Bench/` is derived from existing public
datasets; each subset remains subject to the license of its original source
dataset.
