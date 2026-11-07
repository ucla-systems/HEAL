#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"

: "${MCR_REFERENCE_RUN:=$MCR_RUN}"
: "${MCR_DOMAINS:=all}"
: "${MCR_ROUND1_RUN:=${MCR_REFERENCE_RUN}_generate_round1_chunked}"
: "${MCR_ROUND2_RUN:=${MCR_REFERENCE_RUN}_generate_round2_full_batch}"
: "${MCR_ROUND3_RUN:=${MCR_REFERENCE_RUN}_generate_round3_full_batch_reverse}"
: "${MCR_WRITE_TABLE:=1}"
: "${MCR_FINAL_SOURCE_RUN_ID:=$MCR_REFERENCE_RUN}"
: "${MCR_FLIP_SOURCE_RUN_IDS:=$MCR_ROUND1_RUN $MCR_ROUND2_RUN $MCR_ROUND3_RUN}"

export MCR_RUN="$MCR_ROUND3_RUN"
export MCR_REFERENCE_GENERATION_DIR="${MCR_REFERENCE_GENERATION_DIR:-output/mcr-bench/${MCR_REFERENCE_RUN}/generation_artifacts}"
export MCR_TASK=flip-rate-analysis
export MCR_BUILD_TABLE=0
export MCR_GENERATION_BATCH_MODE=full-reverse

bash tests/ops/05_mcr_bench.sh

if [[ "$MCR_WRITE_TABLE" == "1" ]]; then
  read -r -a MCR_DOMAINS_ARRAY <<< "$MCR_DOMAINS"
  read -r -a MCR_FLIP_SOURCE_RUN_IDS_ARRAY <<< "$MCR_FLIP_SOURCE_RUN_IDS"
  BUILD_CMD=(
    python3.12 tests/mcr-bench/build_mse_flip_table.py
    --run-id "$MCR_REFERENCE_RUN"
    --final-source-run-id "$MCR_FINAL_SOURCE_RUN_ID"
    --domains "${MCR_DOMAINS_ARRAY[@]}"
    --gpu "$GPU_LABEL"
  )
  for run_id in "${MCR_FLIP_SOURCE_RUN_IDS_ARRAY[@]}"; do
    BUILD_CMD+=(--flip-source-run-id "$run_id")
  done
  run_cmd "${BUILD_CMD[@]}"
fi
