#!/usr/bin/env bash
# Build step for CANN Kernel Bench submissions.
#
# The new submission format ships a pre-built `cann_bench` wheel directly
# (under dist/*.whl). If the wheel is already present we skip the build. If
# it's missing but the submission includes a build.sh, we invoke it.
set -euo pipefail

SUBMISSION_DIR="${BENCH_SUBMISSION_DIR:-${BENCH_WORK_DIR}/input/submission}"

echo "[cann-bench] Build step"
echo "[cann-bench] SUBMISSION_DIR=${SUBMISSION_DIR}"

# ── Detect CANN (some submissions need it to build) ─────────────────────────
if [ -z "${ASCEND_HOME_PATH:-}" ]; then
    for candidate in /usr/local/Ascend/cann-8.5.0 /usr/local/Ascend/cann-8.0.0 /usr/local/Ascend/ascend-toolkit/latest; do
        [ -d "$candidate" ] && { export ASCEND_HOME_PATH="$candidate"; break; }
    done
fi
if [ -n "${ASCEND_HOME_PATH:-}" ]; then
    export ASCEND_AICPU_PATH="${ASCEND_HOME_PATH}"
    export ASCEND_OPP_PATH="${ASCEND_HOME_PATH}/opp"
    echo "[cann-bench] ASCEND_HOME_PATH=${ASCEND_HOME_PATH}"
fi

# ── 1. Wheel already present → nothing to build ────────────────────────────
WHEEL_COUNT=$(find "${SUBMISSION_DIR}" -name "*.whl" | wc -l)
if [ "${WHEEL_COUNT}" -gt 0 ]; then
    echo "[cann-bench] Pre-built wheel found in submission, skipping build"
    find "${SUBMISSION_DIR}" -name "*.whl" | head -5
    exit 0
fi

# ── 2. No wheel → invoke user build.sh ─────────────────────────────────────
USER_BUILD="${SUBMISSION_DIR}/build.sh"
if [ -f "${USER_BUILD}" ]; then
    echo "[cann-bench] Running user build.sh"
    cd "${SUBMISSION_DIR}"
    bash "${USER_BUILD}"
    WHEEL_COUNT=$(find "${SUBMISSION_DIR}" -name "*.whl" | wc -l)
    if [ "${WHEEL_COUNT}" -eq 0 ]; then
        echo "ERROR: user build.sh did not produce any .whl" >&2
        exit 1
    fi
    echo "[cann-bench] Build produced:"
    find "${SUBMISSION_DIR}" -name "*.whl"
    exit 0
fi

echo "ERROR: No wheel and no build.sh in ${SUBMISSION_DIR}" >&2
exit 1
