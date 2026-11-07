#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"

run_cmd python3.12 tests/final-hidden-mse/run.py \
  $MODEL_SELECTOR \
  --stage replay \
  --run-id "$MEDQA_NATIVE_MSE_RUN" \
  --generation-dir "$REF_GEN_DIR" \
  --limit "$MEDQA_LIMIT" \
  --replay-generated-token-limit "$MEDQA_MAX_NEW_TOKENS" \
  --prefill-modes original_fp16_prefill original_bf16_prefill \
  --resume \
  --skip-gate

run_cmd python3.12 tests/flip-rate-analysis/run.py \
  $MODEL_SELECTOR \
  --run-id "$MEDQA_NATIVE_FLIP_RUN" \
  --reference-generation-dir "$REF_GEN_DIR" \
  --limit "$MEDQA_LIMIT" \
  --max-new-tokens "$MEDQA_MAX_NEW_TOKENS" \
  --modes original_fp16_generate original_bf16_generate \
  --resume

PLOT_ARGS=(--gpu-slug "$GPU_SLUG")
if [[ -d "tests/final-hidden-mse/results/$MEDQA_MSE_RUN" ]]; then
  PLOT_ARGS+=(--final-hidden-run-id "$MEDQA_MSE_RUN")
fi
PLOT_ARGS+=(--final-hidden-run-id "$MEDQA_NATIVE_MSE_RUN")
if [[ -d "tests/flip-rate-analysis/results/$MEDQA_FLIP_RUN" ]]; then
  PLOT_ARGS+=(--flip-run-id "$MEDQA_FLIP_RUN")
fi
PLOT_ARGS+=(--flip-run-id "$MEDQA_NATIVE_FLIP_RUN")
run_cmd python3.12 tests/medqa-figures/run.py "${PLOT_ARGS[@]}"
