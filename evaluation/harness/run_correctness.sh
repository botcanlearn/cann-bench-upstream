#!/usr/bin/env bash
# Correctness-only evaluation stage (precision check, no timing).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

echo "[cann-bench] Running correctness evaluation"
python3 "${EVALUATE_SCRIPT}" \
    --submission "${SUBMISSION_DIR}" \
    ${MODE_ARGS} \
    --device-id "${DEVICE_ID}" \
    --json-output "${OUTPUT_DIR}/evaluation_results.json" \
    --skip-performance
