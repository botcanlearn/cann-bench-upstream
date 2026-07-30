#!/usr/bin/env bash
# Host-side launcher for cann-bench-eval. Run on an NPU host.
#
#   bash docker/eval/run.sh self-test                                  # 镜像自检
#   bash docker/eval/run.sh /host/ai_ops --operator Exp --no-perf      # 评测一个提交
#   bash docker/eval/run.sh -- --task-dir tasks/level1 --no-perf       # 无提交 (golden 自评)
#   bash docker/eval/run.sh shell                                      # 交互 shell, NPU 已绑入
#
# 第一个参数若是 host 上存在的目录, 就当提交源码目录, 只读挂到 /submission; 其余参数原样透传。
# 报告落到 ${REPORTS:-$PWD/reports}。
#
# 与 docker/base/run.sh 相同: 本镜像是 toolkit-only 血统, host driver runtime libs 必须在这里
# 放上 LD_LIBRARY_PATH (toolkit lib64 由 entrypoint 负责)。

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

VERSION="$(cat "${REPO_ROOT}/VERSION")"
NPU_ARCH="${NPU_ARCH:-ascend910b}"
OPS_MODE="${OPS_MODE:-none}"
IMAGE="${IMAGE:-cann-bench-eval:${VERSION}-${NPU_ARCH}-ops${OPS_MODE}}"
REPORTS="${REPORTS:-${PWD}/reports}"

DRV=/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64

NPU_FLAGS=(
    --privileged
    --ipc=host
    --device /dev/davinci_manager
    --device /dev/devmm_svm
    --device /dev/hisi_hdc
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro
    -v /usr/local/dcmi:/usr/local/dcmi:ro
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro
    -v /etc/ascend_install.info:/etc/ascend_install.info:ro
    -e LD_LIBRARY_PATH="${DRV}"
)
# Unset => the eval's multi-card mode auto-detects every card, which is the normal full-run posture.
[[ -n "${ASCEND_RT_VISIBLE_DEVICES:-}" ]] && NPU_FLAGS+=(-e ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES}")

case "${1:-}" in
    shell)
        exec docker run --rm -it "${NPU_FLAGS[@]}" -v "${REPORTS}:/reports" "$IMAGE" bash
        ;;
    self-test)
        exec docker run --rm "${NPU_FLAGS[@]}" "$IMAGE" --self-test
        ;;
    -h|--help|"")
        docker run --rm "$IMAGE" --help || true
        echo
        echo "本 launcher: $0 [self-test|shell|<host 源码目录>|--] [选项...]"
        exit 0
        ;;
esac

MOUNTS=(-v "${REPORTS}:/reports")
if [[ -d "$1" ]]; then
    SRC="$(cd "$1" && pwd)"; shift
    MOUNTS+=(-v "${SRC}:/submission:ro")
    echo "==> 提交: ${SRC} -> /submission (ro)"
elif [[ "$1" == "--" ]]; then
    shift   # explicit "no submission, options only"
fi

mkdir -p "${REPORTS}"
echo "==> 报告: ${REPORTS} -> /reports"
echo "==> 镜像: ${IMAGE}"
set -x
exec docker run --rm "${NPU_FLAGS[@]}" "${MOUNTS[@]}" "$IMAGE" "$@"
