#!/bin/bash
# cann-bench-eval entrypoint: `docker run <image> [源码目录] [选项]` == an evaluation.
# Everything after the optional source dir is forwarded verbatim to scripts/run_evaluation.sh.
set -euo pipefail

CANN_BENCH_DIR="${CANN_BENCH_DIR:-/opt/cann-bench}"
SUBMISSION_DIR="${SUBMISSION_DIR:-/submission}"
REPORTS_DIR="${REPORTS_DIR:-/reports}"
WORK_SRC=/work/src

# The base image's ENTRYPOINT sourced /etc/cann-env.sh; this image replaces that ENTRYPOINT, so redo
# it. That script is the base's single source of truth for the Ascend environment: set_env.sh (whose
# ASCEND_OPP_PATH / ASCEND_AICPU_PATH side effects the eval and the .run-form custom-op install depend
# on), the venv on PATH, and the <arch>-linux/lib64 that set_env.sh itself omits -- with the image's
# own architecture already baked in. Re-deriving any of it here is what left an aarch64 lib64 path
# hardcoded in an image that also ships for x86_64. Sourcing is idempotent, so repeating what BASH_ENV
# already did costs nothing.
source /etc/cann-env.sh

usage() {
    cat <<EOF
cann-bench 评测镜像 -- docker run 即评测

  docker run --rm <npu-flags> \\
      -v /host/submission:/submission:ro \\
      -v /host/reports:/reports \\
      ${IMAGE_NAME:-cann-bench-eval} [源码目录] [选项]

[源码目录]  容器内路径; 省略则用 ${SUBMISSION_DIR} (即上面挂进来的提交)。
[选项]      原样透传给 scripts/run_evaluation.sh -- --operator / --case-id /
            --task-dir / --no-perf / --device-id / --warmup / --repeat / ...
            (完整列表: docker run <image> --help)

本层自己消费的参数:
  --in-place      直接在源码目录里编译, 不先复制到 ${WORK_SRC}
                  (默认复制, 所以 /submission 可以 :ro 挂载且不被写脏)

逃生口 (作为第一个参数):
  bash | sh | python3 | python     直接执行, 不跑评测
  --self-test                      镜像自检 (torch_npu 见卡 / harness 可导入 / 算子可枚举)
  --help                           本用法 + run_evaluation.sh 的完整选项

报告 (含 profiler 产物 prof_data/) 写到 ${REPORTS_DIR}; 编译日志和 wheel 收在 ${REPORTS_DIR}/build/。
EOF
}

case "${1:-}" in
    bash|sh|python3|python|uv|pip)
        exec "$@"
        ;;
    --self-test)
        shift
        exec python3 "${CANN_BENCH_DIR}/docker-self-test.py" "$@"
        ;;
    -h|--help)
        usage
        echo
        echo "================ scripts/run_evaluation.sh --help ================"
        exec bash "${CANN_BENCH_DIR}/scripts/run_evaluation.sh" --help
        ;;
    "")
        # Zero args is only an error when there is also nothing mounted to evaluate; with a
        # submission present it means "evaluate it with all defaults".
        if [[ ! -d "${SUBMISSION_DIR}" ]] || [[ -z "$(ls -A "${SUBMISSION_DIR}" 2>/dev/null)" ]]; then
            usage
            exit 1
        fi
        ;;
esac

IN_PLACE=0
ARGS=()
for a in "$@"; do
    if [[ "$a" == "--in-place" ]]; then IN_PLACE=1; else ARGS+=("$a"); fi
done
set -- ${ARGS[@]+"${ARGS[@]}"}

# Echo the value of an explicit --source-dir, empty if the caller gave none. Per-argument scan, not
# a substring match on the joined "$*": joined-string matching reads as if it depended on what
# FOLLOWS the flag, which is easy to misread. `--source-dir=X` is recognised for symmetry even
# though run_evaluation.sh's own parser rejects that form (it only handles the space-separated one).
explicit_source_dir() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --source-dir)   printf '%s' "${2:-}"; return ;;
            --source-dir=*) printf '%s' "${1#*=}"; return ;;
        esac
        shift
    done
}
EXPLICIT_SRC="$(explicit_source_dir "$@")"

# run_evaluation.sh's own contract: the source dir is the FIRST positional, i.e. $1 not starting
# with '-'. Scanning further would be wrong -- option VALUES ("--operator Exp") also lack a dash.
SOURCE_DIR=""
if [[ $# -gt 0 && "$1" != -* ]]; then
    SOURCE_DIR="$1"; shift
elif [[ -z "${EXPLICIT_SRC}" ]] \
     && [[ -d "${SUBMISSION_DIR}" ]] && [[ -n "$(ls -A "${SUBMISSION_DIR}" 2>/dev/null)" ]]; then
    SOURCE_DIR="${SUBMISSION_DIR}"
fi
# An explicit --source-dir is passed through untouched: the caller named a container path on
# purpose, so honour it (and skip the copy below) rather than second-guessing the mount.

# kernel_eval builds the submission IN the source dir -- PackageManager._clean_build_artifacts()
# rmtree's <src>/{build,dist,*.egg-info} and _run_build() writes <src>/_compile.log, with build.sh
# running there as cwd. Copying first is what lets /submission be a read-only mount and keeps the
# host tree pristine across repeated runs.
if [[ -n "${SOURCE_DIR}" && "${IN_PLACE}" -eq 0 ]]; then
    rm -rf "${WORK_SRC}"; mkdir -p "${WORK_SRC}"
    cp -a "${SOURCE_DIR}/." "${WORK_SRC}/"
    echo "[entrypoint] 源码 ${SOURCE_DIR} -> ${WORK_SRC} (编译在副本上进行; --in-place 可关闭)"
    SOURCE_DIR="${WORK_SRC}"
fi

mkdir -p "${REPORTS_DIR}"

# Where the build actually happened: the copy (or the in-place dir) when we resolved a source dir,
# otherwise whatever the caller named via --source-dir. Without this second case an explicit
# --source-dir run collected nothing, silently losing _compile.log.
COLLECT_DIR="${SOURCE_DIR:-${EXPLICIT_SRC}}"

finalize() {
    # Salvage the build trace even on failure -- a compile error is the most common outcome and
    # _compile.log is the only place the bisheng/g++ diagnostics land.
    if [[ -n "${COLLECT_DIR}" ]]; then
        mkdir -p "${REPORTS_DIR}/build"
        cp -a "${COLLECT_DIR}/_compile.log" "${REPORTS_DIR}/build/" 2>/dev/null || true
        cp -a "${COLLECT_DIR}/dist"         "${REPORTS_DIR}/build/" 2>/dev/null || true
    fi
    # We run as root -- a .run-form submission installs custom ops into /usr/local/Ascend -- so every
    # report would otherwise land root-owned on the host. Hand them to whoever owns the mount point.
    chown -R "$(stat -c '%u:%g' "${REPORTS_DIR}")" "${REPORTS_DIR}" 2>/dev/null || true
}
trap finalize EXIT

# --reports-dir is mandatory here: without it run_evaluation.sh writes ${PROJECT_ROOT}/reports,
# which inside this image is the read-only-by-intent frozen harness layer.
EVAL_ARGS=(--reports-dir "${REPORTS_DIR}")
[[ -n "${SOURCE_DIR}" ]] && EVAL_ARGS+=("${SOURCE_DIR}")
bash "${CANN_BENCH_DIR}/scripts/run_evaluation.sh" "${EVAL_ARGS[@]}" "$@"
