#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"

run_cmd python3.12 tests/final-hidden-mse/run.py \
  $MODEL_SELECTOR \
  --stage generate \
  --run-id "$REF_RUN" \
  --generation-dir "$REF_GEN_DIR" \
  --limit "$MEDQA_LIMIT" \
  --max-new-tokens "$MEDQA_MAX_NEW_TOKENS" \
  --no-generation-enforce-eager \
  --resume \
  --skip-gate
