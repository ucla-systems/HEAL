#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"

run_cmd python3.12 tests/sharegpt-benchmark/run.py \
  $MODEL_SELECTOR \
  --run-id "$SHAREGPT_RUN" \
  --cases \
    original_bf16 original_fp16 layercast_tf32 layercast_fp32 \
    heal_base heal_plus heal_plus_plus heal_full heal_full_plus \
  --heal-policy-json "$HEAL_POLICY" \
  --resume \
  --keep-going

run_cmd python3.12 tables/build_sharegpt_tables.py \
  --run-id "$SHAREGPT_RUN" \
  --gpu "$GPU_LABEL"
