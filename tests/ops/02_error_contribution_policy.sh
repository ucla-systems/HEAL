#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"

: "${HEAL_PLUS_PLUS_ALLOWED_MSE_RATIO:=1.2}"
: "${HEAL_FULL_PLUS_ALLOWED_MSE_RATIO:=1.5}"

run_cmd python3.12 tests/error-contribution/run.py \
  $MODEL_SELECTOR \
  --run-id "$EC_RUN" \
  --reference-generation-dir "$REF_GEN_DIR" \
  --subset-file tests/error-contribution/subsets/medqa_calibration32_seed20260528.jsonl \
  --limit "$CALIB_LIMIT" \
  --full-replay-attribution \
  --max-replay-tokens-per-doc 1024 \
  --prefill-batch-size 10 \
  --prefill-token-budget 10240 \
  --candidate-modes bf16 fp16 \
  --cleanup-trace-batches

run_cmd python3.12 tests/error-contribution/run.py \
  $MODEL_SELECTOR \
  --run-id "$EC_RUN" \
  --reference-generation-dir "$REF_GEN_DIR" \
  --subset-file tests/error-contribution/subsets/medqa_calibration32_seed20260528.jsonl \
  --limit "$CALIB_LIMIT" \
  --full-replay-attribution \
  --max-replay-tokens-per-doc 1024 \
  --prefill-batch-size 10 \
  --prefill-token-budget 10240 \
  --candidate-modes heal_base heal_plus heal_full \
  --skip-reference \
  --cleanup-trace-batches

run_cmd python3.12 tests/error-contribution/tune_heal_variants.py \
  $MODEL_SELECTOR \
  --contribution-run-id "$EC_RUN" \
  --output-run-id "$TUNE_RUN" \
  --validation-run-prefix "$VALIDATION_RUN_PREFIX" \
  --reference-generation-dir "$REF_GEN_DIR" \
  --validation-subset-file tests/error-contribution/subsets/medqa_validation8_seed20260528.jsonl \
  --validation-limit "$VALIDATION_LIMIT" \
  --max-replay-tokens-per-doc 1024 \
  --heal-plus-plus-allowed-mse-ratio "$HEAL_PLUS_PLUS_ALLOWED_MSE_RATIO" \
  --heal-full-plus-allowed-mse-ratio "$HEAL_FULL_PLUS_ALLOWED_MSE_RATIO"

run_cmd python3.12 tests/error-contribution/export_heal_policy.py \
  --tuning-run-id "$TUNE_RUN" \
  --output "$HEAL_POLICY"
