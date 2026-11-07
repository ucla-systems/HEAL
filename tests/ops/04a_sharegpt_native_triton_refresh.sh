#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"

run_cmd python3.12 tests/sharegpt-benchmark/run.py \
  $MODEL_SELECTOR \
  --run-id "$SHAREGPT_NATIVE_RUN" \
  --cases original_bf16 original_fp16 \
  --resume \
  --keep-going

run_cmd python3.12 tables/build_sharegpt_tables.py \
  --run-id "$SHAREGPT_NATIVE_RUN" \
  --gpu "$GPU_LABEL"
