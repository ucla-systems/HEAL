#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$ROOT"
source tests/ops/common.sh

: "${ARITH_VS_TRUNC_RUN:=arith_vs_trunc_${RUN_STAMP}}"
: "${ARITH_VS_TRUNC_MODEL_SELECTOR:=${MODEL_SELECTOR}}"
: "${ARITH_VS_TRUNC_SUBSET_FILE:=tests/error-contribution/subsets/medqa_calibration32_seed20260528.jsonl}"
: "${ARITH_VS_TRUNC_LIMIT:=8}"
: "${ARITH_VS_TRUNC_MAX_NEW_TOKENS:=1024}"
: "${ARITH_VS_TRUNC_STAGE:=all}"
: "${ARITH_VS_TRUNC_GENERATION_BATCH_SIZE:=8}"
: "${ARITH_VS_TRUNC_MAX_MODEL_LEN:=4096}"
: "${ARITH_VS_TRUNC_MAX_NUM_BATCHED_TOKENS:=4096}"
: "${ARITH_VS_TRUNC_GPU_MEMORY_UTILIZATION:=0.86}"
: "${ARITH_VS_TRUNC_ANALYSIS_DEVICE:=auto}"
: "${ARITH_VS_TRUNC_ANALYSIS_CHUNK_ROWS:=128}"
: "${ARITH_VS_TRUNC_EXTRA_ARGS:=}"

read -r -a ARITH_MODEL_ARGS <<< "${ARITH_VS_TRUNC_MODEL_SELECTOR}"
read -r -a ARITH_EXTRA_ARGS <<< "${ARITH_VS_TRUNC_EXTRA_ARGS}"

run_cmd python3.12 tests/arith-vs-trunc/run.py \
  "${ARITH_MODEL_ARGS[@]}" \
  --run-id "${ARITH_VS_TRUNC_RUN}" \
  --subset-file "${ARITH_VS_TRUNC_SUBSET_FILE}" \
  --limit "${ARITH_VS_TRUNC_LIMIT}" \
  --max-new-tokens "${ARITH_VS_TRUNC_MAX_NEW_TOKENS}" \
  --analyze-last-generated-tokens "${ARITH_VS_TRUNC_MAX_NEW_TOKENS}" \
  --stage "${ARITH_VS_TRUNC_STAGE}" \
  --generation-batch-size "${ARITH_VS_TRUNC_GENERATION_BATCH_SIZE}" \
  --max-model-len "${ARITH_VS_TRUNC_MAX_MODEL_LEN}" \
  --max-num-batched-tokens "${ARITH_VS_TRUNC_MAX_NUM_BATCHED_TOKENS}" \
  --gpu-memory-utilization "${ARITH_VS_TRUNC_GPU_MEMORY_UTILIZATION}" \
  --analysis-device "${ARITH_VS_TRUNC_ANALYSIS_DEVICE}" \
  --analysis-chunk-rows "${ARITH_VS_TRUNC_ANALYSIS_CHUNK_ROWS}" \
  "${ARITH_EXTRA_ARGS[@]}"
