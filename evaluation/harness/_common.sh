#!/usr/bin/env bash
# Shared setup for run_correctness.sh and run_perf.sh. Sourced by both —
# sets SUBMISSION_DIR/OUTPUT_DIR/DEVICE_ID/EVALUATE_SCRIPT/MODE_ARGS and
# exports ASCEND_CUSTOM_OPP_PATH when a vendors/customize dir is present.

BUNDLE_DIR="${BENCH_BUNDLE_DIR}"
SUBMISSION_DIR="${BENCH_SUBMISSION_DIR}"
OUTPUT_DIR="${BENCH_OUTPUT_DIR}"
DEVICE_ID="${BENCH_DEVICE_ID:-3}"
OPERATOR_DIR="${BUNDLE_DIR}/data"

for CUSTOM_OPP in "${SUBMISSION_DIR}/vendors/customize" "${BUNDLE_DIR}/vendors/customize"; do
    if [ -d "${CUSTOM_OPP}" ]; then
        export ASCEND_CUSTOM_OPP_PATH="${CUSTOM_OPP}"
        echo "[cann-bench] ASCEND_CUSTOM_OPP_PATH=${CUSTOM_OPP}"
        [ -f "${CUSTOM_OPP}/bin/set_env.bash" ] && source "${CUSTOM_OPP}/bin/set_env.bash"
        break
    fi
done

EVALUATE_SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/evaluate.py"

# Single-op bundle (data/proto.yaml) vs multi-op batch (data/levelN/<op>/).
if [ -f "${OPERATOR_DIR}/proto.yaml" ]; then
    MODE_ARGS="--operator-dir ${OPERATOR_DIR}"
else
    MODE_ARGS="--bench-root ${OPERATOR_DIR}"
fi
