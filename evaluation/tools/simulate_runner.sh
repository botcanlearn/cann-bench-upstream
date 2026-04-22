#!/usr/bin/env bash
# Simulate the cann-bench runner's job lifecycle for local testing.
#
# Mirrors what runner/main.py does without the control-plane HTTP round-trip:
# stages prepare → compile → correctness → performance, reading harness
# paths from benchmark.yaml and exporting the BENCH_* env vars that
# harness/{build,run_correctness,run_perf}.sh expect.
#
# Usage:
#   simulate_runner.sh <submission_wheel> <bundle_dir> <result_subdir>
#
# Example:
#   ./tools/simulate_runner.sh \
#       submission_examples/direct_launch_simple_example/dist/*.whl \
#       /tmp/test_bundle \
#       direct_launch_simple
#
# <bundle_dir> layout:
#   benchmark.yaml         (with harness.{build,correctness,performance} paths)
#   harness/*.sh           (build/run_correctness/run_perf + _common.sh)
#   evaluate.py            (+ evaluation/core alongside for imports)
#   data/                  (proto.yaml + cases.yaml + golden.py for single op)
#     OR data/levelN/<op>/ (multi-op batch layout)
#
# Results (evaluation_results.json, per-stage logs, result.json, summary.md)
# are copied into evaluation/result_examples/<result_subdir>/.

set -euo pipefail

SUB_WHEEL="$1"
BUNDLE_DIR="$2"
OUT_SUBDIR="$3"
DEVICE_ID="${BENCH_DEVICE_ID:-3}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RESULT_ROOT="${EVAL_ROOT}/result_examples"

WS=/tmp/runner_sim_job_$$
trap 'rm -rf "$WS"' EXIT
mkdir -p "$WS/input/submission" "$WS/input/benchmark" "$WS/work" "$WS/output/logs" "$WS/output/artifacts"

# Stage: prepare — normally the runner downloads submission.zip + benchmark.tar.gz
# from the control plane. Here we copy local files into the expected layout.
cp "$SUB_WHEEL" "$WS/input/submission/"
cp -r "$BUNDLE_DIR"/* "$WS/input/benchmark/"

export BENCH_JOB_ID="sim-job-$(date +%s)-$$"
export BENCH_SUBMISSION_ID="sim-submission"
export BENCH_RUN_ID="$BENCH_JOB_ID"
export BENCH_DEVICE_ID="$DEVICE_ID"
export BENCH_INPUT_DIR="$WS/input"
export BENCH_BENCHMARK_DIR="$WS/input/benchmark"
export BENCH_SUBMISSION_DIR="$WS/input/submission"
export BENCH_OUTPUT_DIR="$WS/output"
export BENCH_WORK_DIR="$WS/work"
export BENCH_BUNDLE_DIR="$WS/input/benchmark"

read_harness() {
    python3 -c "import yaml; print(yaml.safe_load(open('$WS/input/benchmark/benchmark.yaml'))['harness']['$1'])"
}
BUILD_SCRIPT=$(read_harness build)
CORR_SCRIPT=$(read_harness correctness)
PERF_SCRIPT=$(read_harness performance)

echo "=== [sim-runner] stage=prepare OK (ws=$WS) ==="

run_stage() {
    # Returns duration (seconds) on stdout. All progress logging goes to
    # stderr so it does not contaminate the captured return value.
    local name="$1" rel="$2"
    echo "=== [sim-runner] stage=$name ===" >&2
    local t0 dt
    t0=$(date +%s)
    if ! bash "$WS/input/benchmark/$rel" > "$WS/output/logs/$name.log" 2>&1; then
        dt=$(( $(date +%s) - t0 ))
        echo "[sim-runner] $name FAILED after ${dt}s" >&2
        tail -20 "$WS/output/logs/$name.log" >&2
        exit 1
    fi
    dt=$(( $(date +%s) - t0 ))
    echo "[sim-runner] $name OK (${dt}s)" >&2
    printf '%s' "$dt"
}

dt_compile=$(run_stage compile "$BUILD_SCRIPT")
dt_corr=$(run_stage correctness "$CORR_SCRIPT")
dt_perf=$(run_stage performance "$PERF_SCRIPT")

# Copy out results (replace, don't nest-into, any existing logs/)
RESULT_DEST="$RESULT_ROOT/$OUT_SUBDIR"
mkdir -p "$RESULT_DEST"
cp "$WS/output/evaluation_results.json" "$RESULT_DEST/evaluation_results.json"
rm -rf "$RESULT_DEST/logs"
cp -r "$WS/output/logs" "$RESULT_DEST/logs"

python3 - <<PY
import json
r = {
    "job_id": "$BENCH_JOB_ID",
    "device_id": $DEVICE_ID,
    "final_status": "SUCCEEDED",
    "stages": {
        "prepare":     {"status": "passed"},
        "compile":     {"status": "passed", "duration_sec": $dt_compile, "log": "logs/compile.log"},
        "correctness": {"status": "passed", "duration_sec": $dt_corr,    "log": "logs/correctness.log"},
        "performance": {"status": "passed", "duration_sec": $dt_perf,    "log": "logs/performance.log"},
        "archive":     {"status": "passed"},
    },
}
with open("$RESULT_DEST/result.json", "w") as f:
    json.dump(r, f, indent=2)
PY

python3 "$SCRIPT_DIR/summarize.py" \
    --evaluation-results "$RESULT_DEST/evaluation_results.json" \
    --stages "$RESULT_DEST/result.json" \
    --output "$RESULT_DEST/summary.md"

echo ""
echo "=== [sim-runner] DONE — results at $RESULT_DEST ==="
