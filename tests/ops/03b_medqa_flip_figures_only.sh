#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"

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
