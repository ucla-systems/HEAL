# Agent Notes

This is the primary durable maintainer note for future Codex sessions in this
workspace. Keep it short and stable. Environment setup instructions live in
`memory/ENVS.md`; keep transient task planning in the conversation.

## Workspace Shape

- Active public roots should stay small: `memory/`, `vllm/`, `scripts/`,
  `patches/`, `MCR-Bench/`, `tests/final-hidden-mse/`,
  `tests/flip-rate-analysis/`, `tests/mcr-bench/`,
  `tests/qkv-distribution/`, `tests/error-contribution/`, and
  `tests/sharegpt-benchmark/`.
- `vllm/` is the remaining submodule and may be dirty. Never reset or clean it
  unless the user explicitly asks.
- Public checkouts should rely only on files in the active roots. Do not add
  dependencies on local-only directories.
- Generated reporting/output directories such as `output/`, `figures/`, and
  `tables/` may be absent in a fresh checkout. Runners and builders can create
  them when needed; do not treat them as source inputs.
- `tests/final-hidden-mse/` is active source for the MedQA final-hidden MSE
  replay test. Its `data/`, `results/`, `run_logs/`, and `__pycache__/`
  contents are generated artifacts, not durable memory or source code.
- `tests/sharegpt-benchmark/` is active source for the official ShareGPT
  serving benchmark wrapper. Its `results/`, `run_logs/`, and `__pycache__/`
  contents are generated artifacts, not durable memory or source code.
- `tests/flip-rate-analysis/` is active source for non-eager MedQA
  generation flip-rate runs against final-hidden generation artifacts. Its
  `results/`, `run_logs/`, and `__pycache__/` contents are generated artifacts.
- `MCR-Bench/` holds active local MCR-Bench JSONL datasets. Keep only
  `combined_mcr_bench.jsonl` plus one `combined_*_data.jsonl` file per domain;
  raw source subsets and combine scripts are not needed by active
  runners. `tests/mcr-bench/` is a thin wrapper that calls final-hidden-mse with
  `--dataset-kind mcr-bench` and flip-rate-analysis with
  `--answer-parser mcr-bench`, preserving the same model/runtime defaults as
  the MedQA tests.
- `tests/qkv-distribution/` is active source for the BF16 eager MedQA Q/K/V
  dynamic-range distribution experiment using `VLLM_QKV_DISTRIBUTION`. Its
  `results/`, `run_logs/`, and `__pycache__/` contents are generated artifacts.
- `tests/error-contribution/` is active source for the residual/KV
  attribution experiment using `VLLM_ERROR_CONTRIBUTION`. It samples MedQA
  generation replay rows from final-hidden-mse artifacts; FP32 LayerCast saves
  sampled reference residual/K/V/final-norm-input/final-hidden rows, while BF16/FP16 write only
  streaming per-row MSE payloads and aggregated CSVs. Its `results/`,
  `run_logs/`, and `__pycache__/` contents are generated artifacts.
- `figures/` and `tables/` are generated reporting destinations. Table
  builders may export CSV/JSON/LaTeX under `tables/`, and plot builders may
  export figures under `figures/`; neither directory is required input data.
- `scripts/` is platform-neutral. The current incremental build helper reflects
  the NVIDIA CUDA workflow, but AMD can later adapt the same top-level script
  layout instead of using a separate `scripts/amd/` tree.
- `patches/` is the active registry for the canonical vLLM implementation
  versions. Every canonical version, including `VLLM_ORIGINAL`, should be saved
  as `patches/VERSION.patch`. The unpatched vLLM tree is only the clean base
  state used before applying a version.
- The active patch files currently present are `VLLM_ORIGINAL`,
  `VLLM_LAYERCAST`, `VLLM_SIGNIFICAND_BITS`, `VLLM_HEAL`,
  `VLLM_DUMP_TENSORS`, `VLLM_QKV_DISTRIBUTION`,
  `VLLM_ERROR_CONTRIBUTION`, `VLLM_ERROR_CONTRIBUTION_HEAL`,
  and `VLLM_SUMMARIZED`. Some helper scripts may still list
  `VLLM_HEAL_BASE`, `VLLM_HEAL_PLUS`, and `VLLM_HEAL_FULL`, but those
  separate patch files are not present in the active registry; use unified
  `VLLM_HEAL` with `HEAL_MODE=base|plus|full` for current active tests.
- Helper module names inside patches should describe their current role:
  `tensor_trace.py` for env-gated tensor dumps,
  `error_contribution_trace.py` for sampled residual/KV contribution tracing,
  `layercast_runtime.py` for LayerCast gates, `significand_runtime.py` for
  k-bit rounding, and `heal_runtime.py` / `heal_blaslt_gemm.cu` for HEAL
  runtime/GEMM helpers.

## Current Experiment Contract

Current experiments compare 5 models across selected datasets using the
canonical implementation versions below.

Models:

- Qwen3-8B
- Qwen3-14B
- Qwen3-32B
- Llama-3.1-8B-Instruct
- DeepSeek-R1-Distill-Qwen-14B

Current final-hidden test:

- `tests/final-hidden-mse/run.py` is the active self-contained MedQA
  final-hidden MSE replay runner. It switches vLLM versions only through
  `scripts/vllm_incremental_build.sh prepare-version` and `revert-version`.
  MCR-Bench reuses this runner through `tests/mcr-bench/run.py`, which
  passes `--dataset-kind mcr-bench` and writes task/target metadata into
  generation artifacts for downstream answer parsing.
- The runner first materializes chat-template prompt token IDs, then uses
  `VLLM_LAYERCAST` full-FP32 generation as
  `reference_layercast_fp32_generate`, saves actual generated token IDs and the
  generation-time `forward.norm.hidden` rows, then prefill-replays those exact
  prompt+generated token IDs with `TokensPrompt(prompt_token_ids=...)`.
- Default generation and replay budgets are 4096 generated tokens per sample.
  Replay manifests also trim exact self-circulating repeated suffixes by KMP,
  keeping two periods and recording dropped-token audit fields.
- Generation uses token prompts by default and submits the selected MedQA requests
  together by default (`--generation-batch-size 200`) while letting vLLM choose
  the active work. Qwen3-32B is capped to effective generation batch size 16 to
  leave headroom for vLLM profiling. GPU memory utilization defaults are split
  by stage: generation uses `--generation-gpu-memory-utilization 0.93`
  (`--gpu-memory-utilization` is an alias), while replay/prefill uses
  `--replay-gpu-memory-utilization 0.86` by default.
- The prefill matrix is LayerCast FP32 ref/FP32/TF32, Original FP16/BF16,
  Significand k03-k23, and unified HEAL `HEAL_MODE=base|plus|full`.
  Every reported precision mode, including Original FP16/BF16, forces
  `attention_backend=TRITON_ATTN`; native rows without this provenance should not be reported.
- Current Qwen3-32B replay safety overrides: Significand modes use replay GPU
  memory utilization 0.91; `heal_full_prefill` uses batch size 1 and token
  budget 4344. All HEAL-Full modes use `HEAL_FP16_GEMM_COMPENSATION=1`,
  decomposing `x` as `x1 * 32 + x2 / 32` for FP16-input/FP32-output GEMMs.
- Generation and replay can be decoupled with `--stage generate|replay|all`.
  Generation artifacts are intentionally separate from replay outputs and go
  under `output/final-hidden-mse/<run_id>/generation_artifacts/<model_slug>/`,
  with `generated_tokens.jsonl`, `replay_manifest.jsonl`,
  `replay_positions.csv`, `worker_result.json`, and `hidden_states/` in each
  model directory. Download the `generation_artifacts/` parent to reuse one or
  many models on another machine with `--stage replay --generation-dir ...`.
- Replay hidden tensors go under
  `output/final-hidden-mse/<run_id>/<model_slug>/<mode>/`. Token MSE, token
  top-k/top-1 flip data, metrics, worker configs, and logs are generated
  artifacts under `tests/final-hidden-mse/data`, `results`, and `run_logs`;
  keep detailed numbers there rather than in memory.
- The Qwen3-8B strict H100 row-count/baseline gate is disabled by default
  unless `--strict-qwen-gate` is passed, because current generated
  token counts can intentionally differ from baseline artifacts.

General rules:

- Every patch must be a standalone diff from the README-defined base vLLM
  develop/SHA. Do not stack patches.
- Current code-review verified patches are only `VLLM_ORIGINAL` and
  `VLLM_LAYERCAST`. Treat `VLLM_SIGNIFICAND_BITS`, `VLLM_DUMP_TENSORS`,
  `VLLM_QKV_DISTRIBUTION`, `VLLM_ERROR_CONTRIBUTION`,
  `VLLM_SUMMARIZED`, and unified `VLLM_HEAL` as pending fresh review before
  they are considered finalized, because their code changed after earlier
  review passes.
- Before an experiment run, `vllm/` must be clean. Apply one patch with
  `scripts/vllm_incremental_build.sh prepare-version VERSION`, run the
  experiment while that patch is active, then run
  `scripts/vllm_incremental_build.sh revert-version`.
- Always use that `prepare-version` plus `revert-version` workflow to set up
  and tear down an experiment environment. Do not manually apply/reverse patches
  for experiment runs.
- If vLLM code edits need to be preserved before applying another version, save
  them with `scripts/vllm_incremental_build.sh save-version VERSION` into
  `patches/`.
- Current patch names, composition rules, and runtime settings in this file are
  the source of truth.
- `VLLM_ORIGINAL` is not a no-op: it carries the strict FP32 Triton attention
  precision control needed for valid FP32 baselines, plus an env-gated
  `forward.norm.hidden` trace hook. When trace output is not enabled, it is the
  performance baseline. When MedQA hidden enables trace output, dump only the
  final hidden tensor and run with CUDA graph disabled.
- For LayerCast, Significand, and HEAL versions, final-hidden dumping may also
  be a runtime setting inside the same implementation. When disabled, the
  version is used for performance. When enabled, dump only the final hidden
  tensor, meaning the model output after the final decoder layer/norm and before
  logits processing, and run with CUDA graph disabled.
- Runtime metadata must record the base SHA, version name, model, dataset,
  `dump_final_hidden`, and any runtime selection such as significand bits or
  skipped layers.
- Experiment runner defaults are task-specific: ShareGPT uses only performance
  versions, MedQA hidden uses final-hidden-capable versions, and trace-dump
  uses only analyzer versions. Do not run analyzer patches as performance
  baselines.
- After implementation changes, validate with the relevant active `tests/`
  runners and patch apply checks. Choose final-hidden, ShareGPT serving, QKV
  distribution, error-contribution, or flip-rate runs according to the changed
  patch surface.
- After a patch is finalized, run the standard finalized-patch validation
  suite. MedQA small-scale final-hidden/MSE validation applies to every
  finalized patch. Performance validation applies to finalized patches except
  `VLLM_SIGNIFICAND_BITS`, which is excluded from performance runs.
- Use the finalized `VLLM_LAYERCAST` full-FP32 path as the default MedQA
  final-hidden/MSE reference. In `tests/final-hidden-mse`, that means the
  actual LayerCast FP32 generation hidden captured by
  `reference_layercast_fp32_generate`, not a separate synthetic prefill dump.
  A `VLLM_ORIGINAL` full-FP32 reference is acceptable only when explicitly
  noted or when the two references have already been verified equivalent for
  the run.
- When outputs change, rebuild summaries from the active `tests/*/results`
  artifacts and make sure MSE rows match the same complete hidden-dump batch as
  their metadata.
- ShareGPT TPOT must come from the official serving benchmark path
  (`tests/sharegpt-benchmark/run_sharegpt_serving.py` / `vllm bench serve` semantics), not from
  offline `LLM.generate()` elapsed time divided by generated tokens. Offline
  ShareGPT smoke can still validate build/runtime throughput, but should not be
  reported as official TPOT.
- `tests/qkv-distribution/run.py` applies `VLLM_QKV_DISTRIBUTION`, runs BF16
  MedQA generation in eager mode with CUDA graph disabled, and records
  `max(abs(x)) / std(abs(x), correction=0)` for the Q, K, and V tensors
  passed directly into `self.attn(q, k, v)`. Q/K are after Q/K norm and RoPE
  where present, while V is the matching projection-split V. This is distinct
  from `VLLM_DUMP_TENSORS_MODE=qkv`, which dumps full tensors instead of scalar
  distribution stats.
- `tests/error-contribution/run.py` applies `VLLM_ERROR_CONTRIBUTION` and
  replays sampled MedQA generated-token positions from final-hidden-mse
  generation artifacts with a default 512-token generation cap and 10%
  sampling probability. Reference mode is FP32 LayerCast and stores only the
  sampled residual anchors, K/V, final norm input, and final hidden. Candidate BF16/FP16 modes
  compare against the reference inside the hook and save only per-row MSE
  payloads for residual/KV WLS aggregation. Recursive contribution figures are
  derived from these residual-boundary MSE scalars. HEAL+ attribution, when needed, uses
  the additional `VLLM_ERROR_CONTRIBUTION_HEAL` patch with the same trace
  stages and should be run separately with `--skip-reference --candidate-modes
  heal_plus` against an existing FP32 reference dump.
- `tests/sharegpt-benchmark/run.py` wraps that official serving path. Defaults
  are 5 active models, 1024 ShareGPT prompts,
  `--max-model-len 8192`, `--max-num-batched-tokens 8192`,
  `--max-num-seqs 256`, request rate `inf`, and 7 base cases: Original
  FP16/BF16, LayerCast FP32/TF32, and unified HEAL `base|plus|full`.
  All cases pass Triton attention; rerun Original FP16/BF16 rows that did
  not record `attention_backend=TRITON_ATTN`.
  HEAL++ and HEAL-FULL+ are materialized by rerunning `heal_plus`/`heal_full`
  with tuned skip-layer lists rather than separate case names. It
  intentionally excludes Significand Bits performance runs. Raw artifacts go
  under `output/sharegpt-benchmark/<run_id>/`; aggregated JSON/CSV summaries
  go under `tests/sharegpt-benchmark/results/<run_id>/`.
- HEAL-Full ShareGPT and flip-rate-analysis use
  `HEAL_FP16_GEMM_COMPENSATION=1` for all models. INT8 HEAL-Full
  ShareGPT rows/artifacts are not part of the current table convention; rerun
  missing models before filling their HEAL-Full table cells. Runner-level
  `--heal-skip-qkv-quant-layers` is preferred for HEAL++ downgrades;
  `--heal-skip-fp16-comp-layers` is preferred for HEAL-FULL+ downgrades. The
  current patch runtime also reads the `HEAL_SKIP_INT8_COMP_LAYERS`
  env name for FP16-compensation skip layers.
- HEAL++/HEAL-FULL+ layer selection is tuned with a 1.5x median final-hidden-MSE
  budget on the fixed MedQA validation8 set. For each `k`, evaluate both
  `prefix` downgrades (`0..k-1`) and calibration-low downgrades; accept `k` if
  either order is within budget, and use the passing order with lower median MSE
  as the exported layer list. Search starts at `k=4`, doubles while feasible,
  then binary-searches the bracket; if `k=4` fails, search below 4.
- Memory comparisons should use vLLM log-derived values such as
  `model_loading_memory_gib` and `available_kv_cache_memory_gib`, not
  `nvidia-smi` peak process memory. `nvidia-smi` peak is only an auxiliary
  occupancy/debug field.
- When reporting ShareGPT table memory GiB for performance comparisons, use
  `memory_usage_gib = model_loading_memory_gib + graph_capture_memory_gib`
  when graph capture is present, and label the table column
  `Mem. (GiB)`.
- Output directory model names should follow the slug convention used by the
  active `tests/` runners; keep run naming, summary rebuilding, MSE output
  names, and validation aligned with that slug.
- Tensor dumping uses chunked safetensors keys/metadata and should not combine
  repeated captures with `torch.stack` or `torch.cat` during tracing.
- Trace/dump runs should disable V1 multiprocessing and clear trace buffers
  after engine initialization, before measured generation, so warmup/profile
  dummy passes are not mixed into experiment outputs.
- Do not set `TENSOR_TRACE_TENSOR_FILE`/`VLLM_TENSOR_TRACE_TENSOR_FILE`
  before `LLM(...)` finishes initialization. Enable trace output only after
  engine init, then clear trace buffers, otherwise warmup/profile passes can be
  captured and large analyzer modes can OOM or produce polluted dumps.
- The final-hidden stage is `forward.norm.hidden`, recorded at the model output
  after the final decoder layer/norm and before logits processing. Do not
  record it from `compute_logits()`, which may only see sampled decode tokens.
- Original, LayerCast, Significand, and HEAL patches also patch
  `vllm/v1/worker/gpu_model_runner.py` to attach per-row request id, token
  position, token id, and padding metadata to each `forward.norm.hidden` trace
  chunk. Generation splitting must use that metadata when present and only fall
  back to scheduler inference for artifacts that lack it.
- Original, LayerCast, Significand, and HEAL patches may record
  `forward.norm.hidden` only.
  Detailed logits, per-layer, Q/K/V, attention, MLP, and module I/O traces
  belong in the dedicated analyzer patches.
  `VLLM_LAYERCAST` should only carry LayerCast runtime gates and no generic
  tensor-dump, final-logits recording, or operator-oracle capture paths.
  `VLLM_SIGNIFICAND_BITS` should not carry generic tensor-dump, final-logits
  recording, or operator-oracle capture paths; it may still round
  `forward.logits` as a Significand boundary.
- `VLLM_SUMMARIZED` is the explicit integrated exception: it combines the
  current Original, LayerCast, DUMP_TENSORS, and Significand Bits feature set in
  one runtime-selected patch, but still excludes HEAL. This does not make the
  non-verified component patches code-review finalized.
- Runtime env parsing that affects compiled forward paths, such as HEAL skip
  layer lists, must be precomputed during module init or weight loading. The
  compiled/cudagraph forward should read cached booleans, not call string/regex
  parsers.
- HEAL versions are FP16-path implementations for resident weights, GEMM
  inputs, and attention inputs. The main residual stream and selected op
  outputs may be promoted back to FP32 via helpers such as `main_stream()`;
  only attention and GEMM input-side execution should be described as the FP16
  path. RMSNorm in the HEAL FP32 stream uses HEAL-specific CUDA kernels with
  FP32 input/output while loading the resident FP16/BF16 norm weight and
  casting it to FP32 inside the kernel; do not route the CUDA hot path through
  PyTorch `forward_static()`. Experiment scripts normalize `dtype=auto` to
  `float16` for HEAL and reject explicit non-FP16 dtype values rather than
  silently running an unintended BF16/FP32 path.
- Precision dtype contract: `VLLM_ORIGINAL` follows the requested/base vLLM
  dtype. `VLLM_DUMP_TENSORS` is the standalone analyzer for the same BF16,
  FP16, TF32, and strict-FP32 baseline paths, and may also enable the LayerCast
  FP32 execution path with `VLLM_LAYERCAST_FORCE_FP32=1` when tensor dumps are
  needed under LayerCast. LayerCast and Significand are 32-bit precision-path
  implementations: experiment `dtype=auto` normalizes to `float32`, and
  explicit FP16/BF16 dtypes are rejected. For LayerCast, TF32 vs full-FP32 is
  controlled by precision settings, not by changing the model dtype to 16-bit.
  Significand Bits is fixed to the full-FP32 LayerCast workflow: force
  LayerCast, torch matmul precision `highest`, TF32 disabled, and Triton
  attention `tl.dot(input_precision="ieee")`. HEAL remains the FP16-path family
  and normalizes `dtype=auto` to `float16`.
  `VLLM_SUMMARIZED` follows the requested/base dtype by default, enables
  LayerCast only through `VLLM_LAYERCAST_FORCE_FP32` or Significand mode, and
  forces full-FP32 LayerCast when `VLLM_SIGNIFICAND_BITS` is set.
- Rotary cos/sin cache dtype contract: LayerCast, Significand, HEAL, and
  `VLLM_DUMP_TENSORS` LayerCast-FP32 runs must compute and store rotary cos/sin
  cache in FP32. HEAL's 16-bit query-path casts must be temporary; do not
  downcast the persistent `cos_sin_cache` buffer.

## Implementation Versions

1. `VLLM_ORIGINAL`
   - Base vLLM behavior for BF16, FP16, TF32, and FP32 baselines.
   - Same patch is used for performance and original final-hidden dumps. The
     final-hidden hook is env-gated; performance runs must not enable trace
     output.
   - A clean no-patch FP32 run can still accidentally use TF32 inside Triton
     attention because `torch.set_float32_matmul_precision()` does not by
     itself control Triton `tl.dot` kernels. The `VLLM_ORIGINAL.patch` reads
     `VLLM_FLOAT32_MATMUL_PRECISION` and forces unified-attention QK/PV
     `tl.dot` to `input_precision="ieee"` when it is `highest`.

2. `VLLM_LAYERCAST`
   - Compact resident weights with FP32/TF32 LayerCast execution.
   - Purpose: save weight/model memory while running the computation in
     32-bit.
   - Only the 32-bit precision path is supported at experiment entrypoints:
     `dtype=auto` normalizes to `float32`, and explicit FP16/BF16 dtype values
     are rejected. TF32 vs full-FP32 LayerCast is a precision-setting choice,
     not a 16-bit dtype choice.
   - Full FP32 is not just `torch.set_float32_matmul_precision("highest")`;
     attention `tl.dot` must use full FP32 semantics, e.g. `precision="ieee"`.
   - The FP32 attention path currently needs Triton attention.
   - KV cache storage is intentionally FP32 in LayerCast FP32 runs. K/V are
     loaded and cast to Q dtype before `tl.dot` to guarantee FP32 attention
     computation; lower KV token capacity than BF16 is expected.
   - Model-loading memory should stay close to BF16 because weights remain
     compact. If model-loading memory grows toward true FP32 weight memory, the
     implementation is probably materializing resident FP32 weights.
   - Optional final-hidden dump only; no HEAL logic.

3. `VLLM_SIGNIFICAND_BITS`
   - Based on the FP32 LayerCast execution path.
   - Only the 32-bit precision path is supported at experiment entrypoints:
     `dtype=auto` normalizes to `float32`, and explicit FP16/BF16 dtype values
     are rejected. This version is fixed to the full-FP32 LayerCast workflow:
     force LayerCast, torch matmul precision `highest`, TF32 disabled, and
     Triton attention dot precision `ieee`.
   - Applies k-bit significand rounding at major rounding boundaries, not inside
     every primitive/kernel op. Current boundaries cover Linear outputs,
     embedding outputs, RMSNorm outputs, SiLU-and-mul outputs, split Q/K/V
     projection outputs, Q/K norm outputs where present, rotary Q/K outputs,
     `attention.output`, and final logits.
   - `rmsnorm_outputs` means every tensor returned at the RMSNorm boundary is
     rounded, including fused add RMSNorm tuple outputs such as updated hidden
     and residual tensors.
   - Operator boundary names must be emitted at model-call sites in
     `llama.py`/`qwen2.py`/`qwen3.py`, not from generic operator internals.
     These boundary locations and names must match the shared stages defined by
     `VLLM_DUMP_TENSORS`; DUMP_TENSORS is the source of truth for tensor-dump
     names and Significand rounds the corresponding tensors.
     Use canonical model-forward names such as
     `layer_0.self_attn.qkv_proj.output.q`,
     `layer_0.self_attn.q_norm.output`,
     `layer_0.self_attn.rotary_emb.output.q`,
     `layer_0.input_layernorm.output.residual`,
     `layer_0.mlp.silu_and_mul.output`, and `forward.norm.hidden`. The
     `lm_head.linear.logits` boundary is the raw lm_head linear output before
     logits-processor gather, padding removal, soft-cap, scale, or sampling
     logic. Because lm_head execution happens inside `LogitsProcessor`, model
     files attach that stage name to `lm_head` and
     `LogitsProcessor` applies the hook immediately after the native
     `lm_head.quant_method.apply(...)`; do not reimplement logits gathering,
     padding removal, soft-cap, or scale in model files. `VLLM_SIGNIFICAND_BITS`
     uses this boundary for rounding only. It also rounds `forward.logits`
     after logits-processor gather, padding removal, soft-cap, and scale, before
     sampling logic, but it must not record final logits. `forward.logits` is a
     DUMP_TENSORS/SUMMARIZED dump stage at the same location. Do not record
     generic operator-only keys such as
     `op.rms_norm.output` or `op.activation.silu_and_mul.output`.
   - Boundary subsets use canonical names:
     `linear_outputs`, `embedding_outputs`, `rmsnorm_outputs`,
     `activation_outputs`, `operator_outputs`, `attention_qk`,
     `attention_qkv`, `attention_v`, `attention_outputs`, and
     `attention_boundaries`. `all` remains the default for the main sweep, but
     it means all known canonical boundaries, not an arbitrary stage-name
     wildcard.
     `operator_outputs` covers the linear, embedding, RMSNorm, activation,
     rotary Q/K, and final-logits boundaries.
     `attention_qkv` covers split Q/K/V projection outputs only;
     `attention_qk` covers split Q/K projection outputs plus rotary Q/K outputs;
     `attention_boundaries` covers split Q/K/V, rotary Q/K, and attention
     output boundaries.
   - Runtime envs use the `VLLM_SIGNIFICAND_*` prefix, e.g.
     `VLLM_SIGNIFICAND_BITS` and `VLLM_SIGNIFICAND_BOUNDARY_SET`.
     `VLLM_SIGNIFICAND_BITS` must be an integer in `[3, 23]`; invalid values
     or unknown boundary-set names must fail fast instead of silently disabling
     rounding.
   - Optional channel-skip protection uses
     `VLLM_SIGNIFICAND_SKIP_CHANNEL_FILE`,
     `VLLM_SIGNIFICAND_SKIP_CHANNEL_PERCENT`, and optional
     `VLLM_SIGNIFICAND_SKIP_CHANNEL_METRIC=absmax|mean_square|square_sum`
     (`square_sum` is the stored-stat equivalent ranking for `mean_square`).
     These masks must be selected
     at startup/load time from `VLLM_DUMP_TENSORS_MODE=outlier` channel
     rankings. Unknown stage names in the outlier file should warn and be
     ignored; when skip-channel is enabled with percent > 0, every boundary
     that is actually rounded must have a matching outlier mask with the same
     channel width, otherwise fail fast.
   - Optional final-hidden dump only; no HEAL logic.

4. `VLLM_HEAL`
   - Unified HEAL implementation patch. Current final-hidden work uses this
     patch for base, plus, and full behavior. Compatibility names
     `VLLM_HEAL_BASE`, `VLLM_HEAL_PLUS`, and `VLLM_HEAL_FULL` are not separate
     active patch files unless they are restored under `patches/`.
   - Runtime selection uses `HEAL_MODE=base|plus|full` (`full` is the
     experiment-runner default for `VLLM_HEAL`). Do not use the
     `HEAP_V1_*` env names in this patch implementation.
   - `HEAL_MODE=base` is the FP16 input path with FP32 main stream/op outputs,
     GEMM FP32-output kernel, BF16 GEMM-weight scaling, and native FP16
     attention.
   - `HEAL_MODE=plus` adds INT16 Q/K/V attention protection with dual-FP16 dot
     kernels and preserves the HEAL-Plus FP32 attention-output
     buffer. Optional Q/K/V quantization skip layers use
     `HEAL_SKIP_QKV_QUANT_LAYERS`.
   - `HEAL_MODE=full` adds GEMM residual compensation/split. It keeps all
     layers using Q/K/V quantization and uses the FP16 compensation path when
     `HEAL_FP16_GEMM_COMPENSATION=1` is set by the runner. That path decomposes
     `x` as `x1 * 32 + x2 / 32` before accumulating the two FP16-input,
     FP32-output GEMMs. The INT8 path remains
     in the patch for explicit debug/ablation. Runner-level
     `--heal-skip-fp16-comp-layers` is the preferred HEAL-FULL+ spelling, but
     the current patch runtime also reads the
     `HEAL_SKIP_INT8_COMP_LAYERS` env name to disable compensation by layer.
   - The native `torch.ops._C.blaslt_fp32_output_gemm` op is the normal
     FP32-output GEMM path. The Triton FP32-output GEMM fallback may remain in
     code for cross-platform/debug builds, but it must fail closed unless
     `HEAL_ALLOW_TRITON_FP32_GEMM=1` is set explicitly.
   - RMSNorm uses HEAL-specific FP32-stream CUDA ops for both plain and fused
     residual-add RMSNorm. The activations/residuals are FP32 in and FP32 out;
     the norm weight may stay FP16/BF16 in memory and is cast after loading in
     the kernel.
   - Variable/function/cache names should use `heal` terminology such as
     `_heal_*`, `configure_heal_attention`, and `heal_attention_inputs`, not
     `heap_v1` or `hv1` names.
   - Optional final-hidden dump only. Metadata should record `heal_mode`, any
     active HEAL skip lists, and any non-default dynamic-INT8/stats envs.

5. `VLLM_QKV_DISTRIBUTION`
   - Standalone scalar analyzer for the QKV distribution figure. It adds an
     env-gated `qkv_distribution_trace.py` helper and records Q/K/V
     `max(abs(x)) / std(abs(x), correction=0)` on the Q, K, and V
     tensors passed directly into `self.attn(q, k, v)` in Llama, Qwen2, and
     Qwen3 model paths.
   - Enabled by `VLLM_QKV_DISTRIBUTION_FILE=/path/to/qkv_absmax_over_absstd.pt`.
     The active runner uses BF16, `enforce_eager=True`, V1 multiprocessing off,
     and CUDA graph disabled over the selected MedQA subset.
   - Do not use this patch for full tensor dumps or attribution; use
     `VLLM_DUMP_TENSORS` for full tensors and the recursive truncation
     contribution code for error attribution.

6. `VLLM_DUMP_TENSORS`
   - Standalone analyzer for selective tensor dumps. It carries the ORIGINAL
     strict FP32 attention precision control and supports requested BF16, FP16,
     TF32, and strict-FP32 baseline runs. It can also enable LayerCast FP32
     with `VLLM_LAYERCAST_FORCE_FP32=1` / trace-runner `--layercast-fp32` when
     dumps must be collected from the LayerCast execution path. Do not combine
     it with Significand or HEAL.
   - For strict FP32 attention, unified-attention QK/PV `tl.dot` must use
     `input_precision="ieee"` when `VLLM_FLOAT32_MATMUL_PRECISION=highest`;
     otherwise the TF32 trace path can use `input_precision="tf32"`.
   - In LayerCast FP32 mode, resident weights stay compact, linear/embedding
     and RMSNorm execution uses FP32 temporaries, and rotary cos/sin cache is
     stored in FP32 without mutating it down to query dtype.
   - Runs with CUDA graph disabled.
   - Provides one runtime-selected mode in one patch:
     `VLLM_DUMP_TENSORS_MODE=everything|qkv|outlier` (the
     `TENSOR_TRACE_DUMP_TENSORS_MODE` alias also works). Select exactly one
     mode per run.
   - `qkv` mode dumps the existing output-stage subset that feeds attention as
     Q/K/V. Q/K use the post-RoPE output stages
     `layer_*.self_attn.rotary_emb.output.q/k`; V uses the projection output
     stage `layer_*.self_attn.qkv_proj.output.v`.
   - `everything` mode defines the canonical shared boundary locations and
     stage names without duplicate coarse/module aliases. Layer stages use
     `layer_{i}.{op_name}.{tensor_name}` names, for example
     `layer_0.self_attn.qkv_proj.output.q`,
     `layer_0.self_attn.q_norm.output`,
     `layer_0.self_attn.k_norm.output`,
     `layer_0.self_attn.rotary_emb.output.q`,
     `layer_0.self_attn.rotary_emb.output.k`,
     `layer_0.self_attn.attention.output`,
     `layer_0.self_attn.o_proj.output`,
     `layer_0.input_layernorm.output.residual`,
     `layer_0.post_attention_layernorm.output.residual`, and
     `layer_0.mlp.down_proj.output`. Top-level stages use `forward.*`, e.g.
     `forward.embed_tokens.output`, `forward.norm.hidden`, and
     `forward.logits`. The raw lm_head linear output is
     `lm_head.linear.logits`.
   - `outlier` mode does not dump raw tensors. It records per-stage channel
     `absmax`, `square_sum`, `mean_square`, `count`, and descending index
     rankings into a `.pt` payload consumable by Significand skip-channel runs.
     If the tensor output path has another suffix, the outlier payload is
     written next to it with an added `.pt` suffix.
   - The trace runner's `--dump-token-indices` option, or external code
     calling `set_tensor_trace_token_index_filter([...])`, optionally filters
     each traced batch to zero-based token rows from each tensor's first
     dimension. For example, `[0, 5]` records the first and sixth token rows in
     the current forward batch, regardless of token-id values. Omitted / `None`
     means full batch; an empty list/string means no token rows. The filter
     uses `index_select(0, indices)` on each captured tensor, ignores indices
     outside that tensor's first-dimension length, and records no rows for that
     tensor if every requested index is out of range. Negative indices are
     rejected; duplicate indices are collapsed while preserving first
     occurrence order. The filter applies to raw `everything`/`qkv` dumps and
     `outlier` statistics. The
     `VLLM_DUMP_TENSORS_TOKEN_INDICES`, `TENSOR_TRACE_TOKEN_INDICES`, and
     `VLLM_TENSOR_TRACE_TOKEN_INDICES` env vars are debugging fallback interfaces for
     direct low-level tracer use, not the primary experiment interface; the
     trace runner clears the filter unless `--dump-token-indices` is passed.

7. `VLLM_SUMMARIZED`
   - Integrated patch for the current Original, LayerCast, DUMP_TENSORS, and
     Significand Bits functionality. It is selected once as
     `VLLM_SUMMARIZED.patch`; runtime envs choose the behavior.
   - Default/no feature env follows the requested/base vLLM dtype and carries
     ORIGINAL strict-FP32 attention precision controls plus `forward.norm.hidden`
     tracing.
   - `VLLM_LAYERCAST_FORCE_FP32=1` enables the compact-weight LayerCast FP32
     path, including FP32 rotary cos/sin cache handling.
   - `VLLM_DUMP_TENSORS_MODE=everything|qkv|outlier` enables the DUMP_TENSORS
     analyzer modes and canonical stage names, including
     `layer_{i}.self_attn.qkv_proj.output.q/k/v`,
     `layer_{i}.self_attn.rotary_emb.output.q/k`, the qkv-mode output-stage
     subset, `forward.norm.hidden`, `lm_head.linear.logits`, and
     `forward.logits`.
   - The DUMP_TENSORS token-row filter is also available here through the same
     trace-runner `--dump-token-indices`, external
     `set_tensor_trace_token_index_filter([...])`, and env fallback interfaces.
   - `VLLM_SIGNIFICAND_BITS` enables k-bit rounding on those same canonical
     DUMP_TENSORS boundary stages and forces the full-FP32 LayerCast workflow.
     Rounding is integrated in the shared tensor-boundary hook before optional
     tensor dump/outlier recording, so dump outputs reflect the active runtime
     mode.
   - Does not include HEAL runtime/GEMM logic.

## Verified Sanity Checks

These results have been verified and are reliable implementation checks. Treat
large deviations as signs that the patch, runtime flags, or measurement setup
may be wrong.

Performance on ShareGPT V3, 1024 prompts, max token budget 8192:

- FP32 performance should be very poor.
- FP32 LayerCast model-loading memory should stay close to BF16 and should not
  grow much toward true FP32 weight memory. KV cache is intentionally FP32, so
  lower KV capacity / available KV cache compared with BF16 is expected. Use
  vLLM log-derived model/KV/graph memory fields for this check, not raw
  `nvidia-smi` process occupancy.
- TF32 runtime/cost should be roughly 2x BF16.
- HEAL base mode overhead should be within 30%; on NVIDIA, efficient cuBLASLt
  paths usually keep it within 10%.
- HEAL plus mode overhead should be within 50%; on NVIDIA, efficient cuBLASLt
  paths usually keep it within 25%.
- HEAL full mode overhead should be within 150%.
- `VLLM_HEAL` with `HEAL_MODE=base|plus|full` should match the corresponding
  HEAL behavior for ShareGPT performance and HEAL hidden-error band.

Final-hidden MSE on a limited MedQA subset:

- HEAL full mode and FP32 should have the clearly lowest error.
- HEAL plus mode should be next.
- HEAL base mode and TF32 should be roughly similar.
- FP16 and BF16 should be worse than those groups.

Significand-bit sweep expectations:

- k=7 should be in the BF16 neighborhood.
- k=10 should be in the FP16 neighborhood, but worse than TF32.
- k>=16 should be close to FP32.

OOM guardrail:

- Be careful with `torch.stack` and similar tensor aggregation. It can easily
  explode CPU memory or GPU memory during hidden/error collection. Prefer
  batched/chunked processing when tensor volume is large.

## Working Rules

- Keep active planning in the conversation or in experiment code, not in extra
  memory files.
- If the experiment contract changes, replace outdated sections here instead of
  appending a diary.
- Keep detailed metrics in reports or CSV/JSON artifacts, not here.
