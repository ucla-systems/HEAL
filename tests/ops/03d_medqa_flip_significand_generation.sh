#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"

run_cmd python3.12 tests/flip-rate-analysis/run.py \
  $MODEL_SELECTOR \
  --run-id "$MEDQA_FLIP_RUN" \
  --reference-generation-dir "$REF_GEN_DIR" \
  --limit "$MEDQA_LIMIT" \
  --max-new-tokens "$MEDQA_MAX_NEW_TOKENS" \
  --generation-batch-size "$MEDQA_GENERATION_BATCH_SIZE" \
  --modes \
    $MANTISSA_GENERATE_MODES \
  --resume
