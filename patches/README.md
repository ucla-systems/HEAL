# vLLM Patch Registry

This directory is the active registry for the canonical vLLM implementation
versions. Save implementation diffs here with:

```bash
scripts/vllm_incremental_build.sh save-version VERSION
```

Every canonical version is a standalone diff against the checked-out vLLM
submodule commit recorded by this repository. The unpatched vLLM tree is the
clean base state used before applying a version.

- `VLLM_ORIGINAL.patch`
- `VLLM_LAYERCAST.patch`
- `VLLM_SIGNIFICAND_BITS.patch`
- `VLLM_HEAL.patch`
- `VLLM_DUMP_TENSORS.patch`
- `VLLM_QKV_DISTRIBUTION.patch`
- `VLLM_ERROR_CONTRIBUTION.patch`
- `VLLM_ERROR_CONTRIBUTION_HEAL.patch`
- `VLLM_SUMMARIZED.patch`

This README is the public patch contract. `memory/AGENTS.md` may contain
maintainer notes, but active behavior should be inferred from this file, the
patch files themselves, and the runner code under `tests/`.

`VLLM_ORIGINAL.patch` is original vLLM behavior plus strict FP32 attention
precision control and an env-gated final-hidden trace hook. It is needed
because vLLM's
`VLLM_FLOAT32_MATMUL_PRECISION` / `torch.set_float32_matmul_precision()` path
does not by itself control Triton unified-attention `tl.dot`; the patch passes
`input_precision="ieee"` when `VLLM_FLOAT32_MATMUL_PRECISION=highest`.

Original, LayerCast, Significand, and HEAL patches should record only
`forward.norm.hidden` when hidden dumping is enabled. Detailed logits, per-layer,
Q/K/V, attention, MLP, and module I/O traces belong to
`VLLM_DUMP_TENSORS.patch`. `VLLM_LAYERCAST.patch` should only carry
LayerCast runtime gates; `VLLM_LAYERCAST.patch` and
`VLLM_SIGNIFICAND_BITS.patch` should not carry generic tensor-dump,
final-logits recording, or operator-oracle capture paths. Significand may still
round `forward.logits` as a boundary.

Helper module names inside patches should match their role: `tensor_trace.py`
for env-gated tensor dumps, `error_contribution_trace.py` for sampled
residual/KV contribution tracing, `layercast_runtime.py` for LayerCast gates,
`significand_runtime.py` for k-bit rounding, and `heal_runtime.py` /
`heal_blaslt_gemm.cu` for HEAL runtime/GEMM helpers.

Precision dtype is part of the contract. `VLLM_ORIGINAL.patch` follows the
requested/base vLLM dtype. `VLLM_DUMP_TENSORS.patch` is the standalone analyzer
for BF16, FP16, TF32, strict-FP32, and optional LayerCast-FP32 tensor dumps. It
must carry the same strict-FP32 Triton attention control as ORIGINAL, and it
uses `VLLM_LAYERCAST_FORCE_FP32=1` only when LayerCast-FP32 dumps are requested.
`VLLM_LAYERCAST.patch` and `VLLM_SIGNIFICAND_BITS.patch` support only the
32-bit precision path at experiment entrypoints: `dtype=auto` normalizes to
`float32`, and explicit FP16/BF16 dtype values are rejected. For LayerCast, TF32
vs full-FP32 is controlled by precision settings, not by setting a 16-bit model
dtype. Significand Bits is fixed to the full-FP32 LayerCast workflow.

Rotary cos/sin cache dtype is also part of the contract. LayerCast,
Significand, HEAL, and `VLLM_DUMP_TENSORS.patch` LayerCast-FP32 runs must
compute and store rotary cos/sin cache in FP32. HEAL's 16-bit query-path casts
must be temporary and must not downcast the persistent `cos_sin_cache` buffer.

`VLLM_HEAL.patch` is the unified HEAL implementation. Runtime selection
uses `HEAL_MODE=base|plus|full` (experiment runners default to `full`). HEAL
runtime/code names should use `heal` terminology.
For the unified patch, optional skip lists are
`HEAL_SKIP_QKV_QUANT_LAYERS` for `HEAL_MODE=plus` and
`HEAL_SKIP_INT8_COMP_LAYERS` for `HEAL_MODE=full`. In current runners that
second list means "skip FP16 compensation for these MLP layers"; runner-level
`--heal-skip-fp16-comp-layers` aliases are allowed, but they export that
runtime env as well. HEAL-Full uses `HEAL_FP16_GEMM_COMPENSATION=1`,
which decomposes `x` as `x1 * 32 + x2 / 32` and accumulates the two
FP16-input/FP32-output GEMMs. The INT8 compensation code remains only as an
explicit ablation path. The native
`torch.ops._C.blaslt_fp32_output_gemm` op is the normal FP32-output GEMM path;
the Triton FP32-output GEMM fallback is disabled unless
`HEAL_ALLOW_TRITON_FP32_GEMM=1` is set explicitly. HEAL RMSNorm uses
FP32-input/FP32-output CUDA kernels that load the resident FP16/BF16 norm weight
and cast it to FP32 inside the kernel, rather than using PyTorch
`forward_static()` on the CUDA hot path.

`VLLM_SIGNIFICAND_BITS.patch` applies significand rounding at major rounding
boundaries, not inside every primitive/kernel op. `VLLM_SIGNIFICAND_BITS` must
be an integer in `[3, 23]`; invalid values fail fast instead of silently
disabling rounding. Boundary subsets for `VLLM_SIGNIFICAND_BOUNDARY_SET` use
names such as `linear_outputs`,
`embedding_outputs`, `rmsnorm_outputs`, `activation_outputs`,
`operator_outputs`, `attention_qk`, `attention_qkv`, `attention_v`,
`attention_outputs`, and `attention_boundaries`; `all` is the default. Unknown
boundary-set names are rejected. `all` means all known canonical boundaries,
not an arbitrary stage-name wildcard. `operator_outputs` covers the linear,
embedding, RMSNorm, activation, rotary Q/K, and final-logits boundaries.
`attention_qkv` covers split Q/K/V projection outputs only; `attention_qk`
covers split Q/K projection outputs plus rotary Q/K outputs;
`attention_boundaries` covers split Q/K/V, rotary Q/K, and attention output
boundaries.
Q/K/V boundary stage names are split projection outputs:
`layer_*.self_attn.qkv_proj.output.q/k/v`. Optional channel-skip masks use
`VLLM_SIGNIFICAND_SKIP_CHANNEL_FILE`,
`VLLM_SIGNIFICAND_SKIP_CHANNEL_PERCENT`, and
`VLLM_SIGNIFICAND_SKIP_CHANNEL_METRIC=absmax|mean_square|square_sum`;
`square_sum` is the stored-stat equivalent ranking for `mean_square`.
Unknown stage names in the outlier file
warn and are ignored; with skip-channel percent > 0, every rounded boundary
must have a matching outlier mask with the same channel width or the run fails.
Boundary names are emitted in the model forward files, for example
`layer_0.self_attn.qkv_proj.output.q`,
`layer_0.self_attn.q_norm.output`,
`layer_0.self_attn.rotary_emb.output.q`,
`layer_0.input_layernorm.output.hidden`,
`layer_0.mlp.silu_and_mul.output`, and `forward.norm.hidden`; avoid generic
operator-only keys because multiple same-kind operators can exist in one layer.
These locations and names must match the shared stages defined by
`VLLM_DUMP_TENSORS.patch`; DUMP_TENSORS is the source of truth for tensor-dump
names and Significand rounds the corresponding tensors.
The `lm_head.linear.logits` boundary is the raw lm_head linear output before
logits-processor gather, padding removal, soft-cap, scale, or sampling logic.
Because lm_head execution happens inside `LogitsProcessor`, model files attach
the model-scoped stage name to `lm_head` and `LogitsProcessor` applies the hook
immediately after the native `lm_head.quant_method.apply(...)`; do not
reimplement logits gathering, padding removal, soft-cap, or scale in model
files. `VLLM_SIGNIFICAND_BITS.patch` also rounds `forward.logits` after
logits-processor gather, padding removal, soft-cap, and scale, but does not
record final logits.

Use `VLLM_DUMP_TENSORS_MODE=qkv` to dump the existing output-stage subset that
feeds attention as Q/K/V. Q/K use the post-RoPE output stages
`layer_*.self_attn.rotary_emb.output.q/k`; V uses the projection output stage
`layer_*.self_attn.qkv_proj.output.v`.

`VLLM_QKV_DISTRIBUTION.patch` is a separate scalar analyzer for
`tests/qkv-distribution`. It records only `max(abs(x)) / std(abs(x),
correction=0)` for the Q, K, and V tensors passed directly into
`self.attn(q, k, v)`: Q/K after Q/K norm and RoPE where present, and the
matching projection-split V. It is enabled with
`VLLM_QKV_DISTRIBUTION_FILE=/path/to/qkv_absmax_over_absstd.pt`, and the active
runner uses BF16 plus eager mode rather than dumping full tensors.

`VLLM_ERROR_CONTRIBUTION.patch` is the analyzer for
`tests/error-contribution`. It carries the LayerCast runtime gates plus
minimal residual/KV hooks for sampled MedQA replay rows. FP32 LayerCast
reference batches save sampled rows for
`layer_*.input_layernorm.output.residual`,
`layer_*.post_attention_layernorm.output.residual`, `attention.k_full.*`,
`attention.v_full.*`, `forward.norm.input`, and `forward.norm.hidden`.
BF16/FP16 candidate batches read those FP32 reference tensors and write only
compact per-row MSE payloads; they must not dump full candidate tensors.
Recursive contribution figures are computed from these residual-boundary MSE
scalars; WLS is used only to estimate the K/V direct fraction.
`VLLM_ERROR_CONTRIBUTION_HEAL.patch` is the HEAL base/plus/full variant of the
same residual/KV analyzer.

`VLLM_DUMP_TENSORS.patch` has mutually exclusive `everything`, `qkv`, and
`outlier` modes via `VLLM_DUMP_TENSORS_MODE` (or
`TENSOR_TRACE_DUMP_TENSORS_MODE`). Select exactly one mode per run. The
trace runner can pass `--layercast-fp32` to collect these dumps from the
LayerCast FP32 path; otherwise it follows the requested/base vLLM dtype and
float32 precision settings.
`everything` mode defines the canonical shared tensor-dump boundary locations
and stage names, emitted at model-call sites rather than generic operator
internals, with no duplicate coarse/module aliases. Decoder-layer names use
`layer_{i}.{op_name}.{tensor_name}` such as
`layer_0.self_attn.qkv_proj.output.q`,
`layer_0.self_attn.q_norm.output`,
`layer_0.self_attn.k_norm.output`,
`layer_0.self_attn.rotary_emb.output.q`,
`layer_0.self_attn.rotary_emb.output.k`,
`layer_0.self_attn.attention.output`, and
`layer_0.mlp.down_proj.output`. Top-level names use `forward.*`, including
`forward.embed_tokens.output`, `forward.norm.hidden`, and `forward.logits`.
`lm_head.linear.logits` is recorded at the raw lm_head linear output before
logits-processor gather, padding removal, soft-cap, scale, or sampling logic.
`forward.logits` is recorded after logits-processor gather, padding removal,
soft-cap, and scale, before sampling logic. Both are captured through the
native `LogitsProcessor` path.
Use the trace runner's `--dump-token-indices` option, or call
`set_tensor_trace_token_index_filter([...])` from external orchestration code,
to filter dumps within each captured batch to zero-based token rows from each
tensor's first dimension. For example, `[0, 5]` records the first and sixth
token rows in the current forward batch, regardless of their token-id values.
Omit the option / pass `None` for full-batch dumps; pass an empty list or empty
string to record no token rows. The filter uses `index_select(0, indices)` on
each captured tensor, ignores indices outside that tensor's first-dimension
length, and records no rows for that tensor if every requested index is out of
range. Negative indices are rejected; duplicate indices are collapsed while
preserving first occurrence order. The filter applies to raw
`everything`/`qkv` dumps and to `outlier` statistics.
`VLLM_DUMP_TENSORS_TOKEN_INDICES`, `TENSOR_TRACE_TOKEN_INDICES`, and
`VLLM_TENSOR_TRACE_TOKEN_INDICES` env vars are debugging fallbacks for
direct low-level tracer use, but should not be the primary experiment
interface; the trace runner clears the filter unless `--dump-token-indices` is
passed.
The `outlier` mode writes a `.pt` payload containing per-stage channel
`absmax`, `square_sum`, `mean_square`, `count`, and descending channel-index
rankings for Significand skip-channel runs. If the tensor output path has
another suffix, the outlier payload is written next to it with an added `.pt`
suffix.

Theoretical `VLLM_DUMP_TENSORS` storage sizing, assuming TP=1/PP=1, no stage
regex filter, one captured row per generated/prompt token, and the current
float32 CPU clone/safetensors path:

- Raw safetensors disk size and runtime CPU tensor-buffer size are the same
  first-order number because every captured floating tensor is kept as a
  CPU-side float32 clone until `flush_tensor_trace_tensors()`.
- The formulas below use `n` for captured token rows. Metadata/header overhead
  is small relative to real runs; reserve about 1.2x the listed raw CPU buffer
  size for Python metadata and serialization overhead.
- `everything` includes `lm_head.linear.logits` and `forward.logits`.

| Model | `qkv` disk and raw CPU buffer | `qkv` CPU RAM budget | `everything` disk and raw CPU buffer | `everything` CPU RAM budget |
| --- | ---: | ---: | ---: | ---: |
| Qwen3-8B | `0.000824 * n GiB` (`0.824 GiB / 1k`) | `0.989 GiB / 1k` | `0.01213 * n GiB` (`12.13 GiB / 1k`) | `14.56 GiB / 1k` |
| Qwen3-14B | `0.00107 * n GiB` (`1.07 GiB / 1k`) | `1.28 GiB / 1k` | `0.01717 * n GiB` (`17.17 GiB / 1k`) | `20.61 GiB / 1k` |
| Qwen3-32B | `0.00244 * n GiB` (`2.44 GiB / 1k`) | `2.93 GiB / 1k` | `0.03557 * n GiB` (`35.57 GiB / 1k`) | `42.69 GiB / 1k` |
| Llama-3.1-8B-Instruct | `0.000732 * n GiB` (`0.732 GiB / 1k`) | `0.879 GiB / 1k` | `0.01086 * n GiB` (`10.86 GiB / 1k`) | `13.03 GiB / 1k` |
| DeepSeek-R1-Distill-Qwen-14B | `0.00128 * n GiB` (`1.28 GiB / 1k`) | `1.54 GiB / 1k` | `0.01736 * n GiB` (`17.36 GiB / 1k`) | `20.83 GiB / 1k` |

With TP/PP enabled, per-rank files change because Q/K/V, MLP, and
`lm_head.linear.logits` are sharded while hidden-state boundaries and
`forward.logits` can remain replicated or rank-specific depending on the
pipeline/tensor-parallel path. Recompute sizing from the actual captured
per-rank shapes instead of dividing the TP=1 table mechanically.

`VLLM_SUMMARIZED.patch` is the integrated runtime-selection patch for the
current Original, LayerCast, DUMP_TENSORS, and Significand Bits feature set.
With no extra runtime env it follows the requested/base vLLM dtype and the
ORIGINAL strict-FP32 attention controls with final-hidden stage
`forward.norm.hidden`. `VLLM_LAYERCAST_FORCE_FP32=1` enables the compact-weight
LayerCast FP32 path. `VLLM_DUMP_TENSORS_MODE` enables the same
`everything|qkv|outlier` tensor-dump modes and canonical names as
`VLLM_DUMP_TENSORS.patch`, including `lm_head.linear.logits`,
`forward.logits`, and decoder-layer stages such as
`layer_0.self_attn.rotary_emb.output.q/k`. `VLLM_SIGNIFICAND_BITS` enables
Significand rounding at those same canonical DUMP_TENSORS boundary stages and
forces the full-FP32 LayerCast workflow. SUMMARIZED also supports the same
token-row filter as DUMP_TENSORS through `--dump-token-indices`,
`set_tensor_trace_token_index_filter([...])`, and the env fallback interfaces.

Recommended run loop:

```bash
scripts/vllm_incremental_build.sh prepare-version VLLM_ORIGINAL
# run experiment while the patch is active
scripts/vllm_incremental_build.sh revert-version
```

Patch application requires a clean `vllm/` worktree. If there are code edits
that must be preserved, save them into this directory first, then return
`vllm/` to the clean base before applying a version.
