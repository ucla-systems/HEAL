#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"

: "${MCR_TASK:=all}"
: "${MCR_DOMAINS:=all}"
: "${MCR_STAGE:=}"
: "${MCR_GENERATION_DIR:=output/mcr-bench/${MCR_RUN}/generation_artifacts}"
: "${MCR_REFERENCE_GENERATION_DIR:=$MCR_GENERATION_DIR}"
: "${MCR_BUILD_TABLE:=1}"
: "${MCR_DRY_RUN:=0}"
: "${MCR_FINAL_HIDDEN_MODES:=original_bf16_prefill original_fp16_prefill layercast_tf32_prefill layercast_fp32_prefill heal_base_prefill heal_plus_prefill heal_plus_plus_prefill heal_full_plus_prefill}"
: "${MCR_FLIP_MODES:=original_bf16_generate original_fp16_generate layercast_tf32_generate layercast_fp32_generate heal_base_generate heal_plus_generate heal_plus_plus_generate heal_full_plus_generate}"
: "${MCR_FINAL_SOURCE_RUN_ID:=$MCR_RUN}"
: "${MCR_FLIP_SOURCE_RUN_IDS:=$MCR_RUN}"
: "${MCR_GENERATION_BATCH_SIZE:=32}"
: "${MCR_GENERATION_BATCH_MODE:=chunked}"
: "${MCR_GENERATION_GPU_MEMORY_UTILIZATION:=0.93}"

read -r -a MODEL_SELECTOR_ARRAY <<< "$MODEL_SELECTOR"
read -r -a MCR_DOMAINS_ARRAY <<< "$MCR_DOMAINS"
read -r -a MCR_FINAL_HIDDEN_MODES_ARRAY <<< "$MCR_FINAL_HIDDEN_MODES"
read -r -a MCR_FLIP_MODES_ARRAY <<< "$MCR_FLIP_MODES"
read -r -a MCR_FLIP_SOURCE_RUN_IDS_ARRAY <<< "$MCR_FLIP_SOURCE_RUN_IDS"

MCR_CMD=(
  python3.12 tests/mcr-bench/run.py
  "${MODEL_SELECTOR_ARRAY[@]}"
  --run-id "$MCR_RUN"
  --domains "${MCR_DOMAINS_ARRAY[@]}"
  --task "$MCR_TASK"
  --generation-dir "$MCR_GENERATION_DIR"
  --reference-generation-dir "$MCR_REFERENCE_GENERATION_DIR"
  --limit "$MCR_LIMIT"
  --max-new-tokens "$MCR_MAX_NEW_TOKENS"
  --generation-batch-size "$MCR_GENERATION_BATCH_SIZE"
  --generation-batch-mode "$MCR_GENERATION_BATCH_MODE"
  --generation-gpu-memory-utilization "$MCR_GENERATION_GPU_MEMORY_UTILIZATION"
  --final-hidden-modes "${MCR_FINAL_HIDDEN_MODES_ARRAY[@]}"
  --flip-modes "${MCR_FLIP_MODES_ARRAY[@]}"
  --heal-policy-json "$HEAL_POLICY"
  --resume
)
if [[ -n "$MCR_STAGE" ]]; then
  MCR_CMD+=(--stage "$MCR_STAGE")
fi
if [[ "$MCR_DRY_RUN" == "1" ]]; then
  MCR_CMD+=(--dry-run)
fi
run_cmd "${MCR_CMD[@]}"

if [[ "$MCR_BUILD_TABLE" == "1" && "$MCR_DRY_RUN" != "1" && "$MCR_TASK" == "all" ]]; then
  BUILD_CMD=(
    python3.12 tests/mcr-bench/build_mse_flip_table.py
    --run-id "$MCR_RUN"
    --final-source-run-id "$MCR_FINAL_SOURCE_RUN_ID"
    --domains "${MCR_DOMAINS_ARRAY[@]}"
    --gpu "$GPU_LABEL"
  )
  for run_id in "${MCR_FLIP_SOURCE_RUN_IDS_ARRAY[@]}"; do
    BUILD_CMD+=(--flip-source-run-id "$run_id")
  done
  run_cmd "${BUILD_CMD[@]}"
fi
