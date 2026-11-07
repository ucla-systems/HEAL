#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$ROOT"

: "${RUN_STAMP:=$(date -u +%Y%m%dT%H%M%SZ)}"
: "${GPU_SLUG:=NVIDIA_H100_80GB_HBM3}"
: "${GPU_LABEL:=H100}"
: "${MODEL_SELECTOR:=--all-models}"
: "${MEDQA_LIMIT:=200}"
: "${MEDQA_MAX_NEW_TOKENS:=4096}"
: "${MEDQA_GENERATION_BATCH_SIZE:=32}"
: "${CALIB_LIMIT:=32}"
: "${VALIDATION_LIMIT:=8}"
: "${MCR_LIMIT:=200}"
: "${MCR_MAX_NEW_TOKENS:=4096}"
: "${OPS_DRY_RUN:=0}"

: "${REF_RUN:=medqa_ref_${RUN_STAMP}}"
: "${REF_GEN_DIR:=output/final-hidden-mse/${REF_RUN}/generation_artifacts}"
: "${EC_RUN:=error_contribution_calibration32_${RUN_STAMP}}"
: "${TUNE_RUN:=heal_variant_tuning_pluspp1p2_fullplus1p5_${RUN_STAMP}}"
: "${VALIDATION_RUN_PREFIX:=${TUNE_RUN}_validation}"
: "${HEAL_POLICY:=tests/error-contribution/policies/${TUNE_RUN}.json}"
: "${MEDQA_MSE_RUN:=medqa_final_hidden_mse_${RUN_STAMP}}"
: "${MEDQA_FLIP_RUN:=medqa_flip_rate_${RUN_STAMP}}"
: "${MEDQA_NATIVE_MSE_RUN:=medqa_native_triton_final_hidden_mse_${RUN_STAMP}}"
: "${MEDQA_NATIVE_FLIP_RUN:=medqa_native_triton_flip_rate_${RUN_STAMP}}"
: "${SHAREGPT_RUN:=sharegpt_${RUN_STAMP}}"
: "${SHAREGPT_NATIVE_RUN:=sharegpt_native_triton_${RUN_STAMP}}"
: "${MCR_RUN:=mcr_bench_${RUN_STAMP}}"

MANTISSA_PREFILL_MODES="${MANTISSA_PREFILL_MODES:-$(printf 'significand_k%02d_prefill ' $(seq 3 23))}"
MANTISSA_GENERATE_MODES="${MANTISSA_GENERATE_MODES:-$(printf 'significand_k%02d_generate ' $(seq 3 23))}"
MEDQA_OTHER_GENERATE_MODES="${MEDQA_OTHER_GENERATE_MODES:-layercast_fp32_generate layercast_tf32_generate original_fp16_generate original_bf16_generate heal_base_generate heal_plus_generate heal_plus_plus_generate heal_full_generate heal_full_plus_generate}"

run_cmd() {
  printf '\n+ '
  printf '%q ' "$@"
  printf '\n'
  if [[ "$OPS_DRY_RUN" == "1" ]]; then
    printf '[ops dry-run] command not executed\n'
    return 0
  fi
  "$@"
}
