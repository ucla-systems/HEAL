#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"

run_cmd python3.12 tests/sharegpt-benchmark/run.py \
  $MODEL_SELECTOR \
  --run-id "$SHAREGPT_RUN" \
  --cases heal_plus_plus heal_full_plus \
  --heal-policy-json "$HEAL_POLICY" \
  --resume \
  --keep-going

run_cmd python3.12 tables/build_sharegpt_tables.py \
  --run-id "$SHAREGPT_RUN" \
  --gpu "$GPU_LABEL"
