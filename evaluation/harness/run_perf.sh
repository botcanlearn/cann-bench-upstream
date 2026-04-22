#!/usr/bin/env bash
# Full evaluation stage: correctness + performance (speedup vs baseline).
# Set BENCH_MEASURE_BASELINES=1 to remeasure each case's golden on NPU and
# use that as the speedup baseline (for calibrating cases.yaml values).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

EXTRA_ARGS=""
if [ "${BENCH_MEASURE_BASELINES:-0}" = "1" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --measure-baselines"
fi

echo "[cann-bench] Running full evaluation (correctness + performance)"
python3 "${EVALUATE_SCRIPT}" \
    --submission "${SUBMISSION_DIR}" \
    ${MODE_ARGS} \
    --device-id "${DEVICE_ID}" \
    --json-output "${OUTPUT_DIR}/evaluation_results.json" \
    ${EXTRA_ARGS}
