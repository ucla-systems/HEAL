#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"

: "${MCR_REFERENCE_RUN:=$MCR_RUN}"
: "${MCR_ROUND1_RUN:=${MCR_REFERENCE_RUN}_generate_round1_chunked}"

export MCR_RUN="$MCR_ROUND1_RUN"
export MCR_REFERENCE_GENERATION_DIR="${MCR_REFERENCE_GENERATION_DIR:-output/mcr-bench/${MCR_REFERENCE_RUN}/generation_artifacts}"
export MCR_TASK=flip-rate-analysis
export MCR_BUILD_TABLE=0
export MCR_GENERATION_BATCH_MODE=chunked
export MCR_GENERATION_BATCH_SIZE="${MCR_GENERATION_BATCH_SIZE:-32}"

bash tests/ops/05_mcr_bench.sh
