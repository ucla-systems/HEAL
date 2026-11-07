# Scripts

## vLLM Patch And Incremental Build

Use this after the accelerator environment and editable vLLM install are ready.
It currently reflects the NVIDIA CUDA workflow, but the scripts directory is
intentionally platform-neutral so the AMD implementation can replace or adapt
the internals later without changing the top-level layout. The helper follows
vLLM's CMake incremental workflow so CUDA/C++ edits rebuild only the affected
native components.

**Warning:** this is not a full rebuild workflow. Do not remove
`vllm/cmake-build-release/`, run `uv pip install -v -e . --no-build-isolation`,
or otherwise force a full source rebuild unless explicitly requested. Normal
iteration should repeat only the incremental `build` step.

`setup`, `configure`, and `all` are guarded because they can mutate the editable
install or regenerate CMake build configuration. Confirm with the user first,
then pass `--confirm-setup-or-configure`. The unguarded low-level command is
only `build`.

```bash
# One-time editable install using precompiled wheels. This is not a full rebuild.
scripts/vllm_incremental_build.sh setup --confirm-setup-or-configure

# One-time CMake preset generation and configure.
scripts/vllm_incremental_build.sh configure --confirm-setup-or-configure

# Repeat after CUDA/C++ edits.
scripts/vllm_incremental_build.sh build
```

Useful knobs:

```bash
JOBS=32 scripts/vllm_incremental_build.sh build
TARGET=install PRESET=release scripts/vllm_incremental_build.sh build
VLLM_DIR=/path/to/vllm scripts/vllm_incremental_build.sh configure --confirm-setup-or-configure
```

The build artifacts stay in `vllm/cmake-build-release/`, and the install target
copies rebuilt shared libraries back into the editable `vllm/` package tree.

## Patch Workflow

Canonical implementation diffs live in `patches/`. Every experiment version,
including `VLLM_ORIGINAL`, is saved as `patches/VERSION.patch`; the unpatched
vLLM tree is only the clean base state used before applying one version.

Before an experiment run, apply the version to a clean `vllm/` tree and build:

```bash
scripts/vllm_incremental_build.sh prepare-version VLLM_ORIGINAL
```

Run the experiment while that patch is active, then revert it:

```bash
scripts/vllm_incremental_build.sh revert-version
```

Useful patch commands:

```bash
scripts/vllm_incremental_build.sh list-versions
scripts/vllm_incremental_build.sh status
scripts/vllm_incremental_build.sh apply-version VLLM_LAYERCAST
scripts/vllm_incremental_build.sh save-version VLLM_LAYERCAST
```

`apply-version` and `prepare-version` refuse to run unless `vllm/` has no code
changes. If current edits are the intended implementation, save them first with
`save-version VERSION`, then return `vllm/` to the clean base before applying a
patch for a run.

## Generated Outputs

Scripts may help create experiment artifacts, plots, or tables, but those files
belong under generated-output locations such as `output/`, `figures/`, and
`tables/`. The `scripts/` directory itself should stay source-only.
