#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"

run_cmd python3.12 tests/final-hidden-mse/run.py \
  $MODEL_SELECTOR \
  --stage replay \
  --run-id "$MEDQA_MSE_RUN" \
  --generation-dir "$REF_GEN_DIR" \
  --limit "$MEDQA_LIMIT" \
  --replay-generated-token-limit "$MEDQA_MAX_NEW_TOKENS" \
  --prefill-modes \
    layercast_fp32_prefill_ref layercast_fp32_prefill layercast_tf32_prefill \
    original_fp16_prefill original_bf16_prefill \
    $MANTISSA_PREFILL_MODES \
    heal_base_prefill heal_plus_prefill heal_plus_plus_prefill heal_full_prefill heal_full_plus_prefill \
  --heal-policy-json "$HEAL_POLICY" \
  --resume \
  --skip-gate

# Flip-rate generation is split into 03c (non-significand cases) and 03d (significand sweep).

PLOT_ARGS=(--gpu-slug "$GPU_SLUG")
PLOT_ARGS+=(--final-hidden-run-id "$MEDQA_MSE_RUN")
if [[ -d "tests/final-hidden-mse/results/$MEDQA_NATIVE_MSE_RUN" ]]; then
  PLOT_ARGS+=(--final-hidden-run-id "$MEDQA_NATIVE_MSE_RUN")
fi
PLOT_ARGS+=(--flip-run-id "$MEDQA_FLIP_RUN")
if [[ -d "tests/flip-rate-analysis/results/$MEDQA_NATIVE_FLIP_RUN" ]]; then
  PLOT_ARGS+=(--flip-run-id "$MEDQA_NATIVE_FLIP_RUN")
fi
run_cmd python3.12 tests/medqa-figures/run.py "${PLOT_ARGS[@]}"
