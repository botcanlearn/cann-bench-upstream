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
ARCH="${ARCH:-$(uname -m)}"
case "${ARCH}" in arm64|aarch64) ARCH=aarch64 ;; amd64|x86_64) ARCH=x86_64 ;; esac
NPU_ARCH="${NPU_ARCH:-ascend910b}"
# Mirrors build.sh's per-SoC default (950 cannot run OPS_MODE=none) so the derived tag matches.
case "${NPU_ARCH}" in ascend950) OPS_MODE="${OPS_MODE:-refonly}" ;; *) OPS_MODE="${OPS_MODE:-none}" ;; esac
IMAGE="${IMAGE:-cann-bench-eval:${VERSION}-${NPU_ARCH}-${ARCH}-ops${OPS_MODE}}"
REPORTS="${REPORTS:-${PWD}/reports}"

DRV=/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64

NPU_FLAGS=(
    --privileged
    --ipc=host
    --device /dev/davinci_manager
    --device /dev/devmm_svm
    --device /dev/hisi_hdc
    -e LD_LIBRARY_PATH="${DRV}"
)
# Only the driver tree is universal. dcmi / npu-smi / ascend_install.info sit wherever the host's
# driver install put them, and docker CREATES a missing bind-mount source as a root-owned empty
# DIRECTORY on the host -- littering a shared box and shadowing the in-container path. Mount each only
# if it exists AND is the right type: a5 already carries an empty /usr/local/bin/npu-smi directory left
# by some earlier unconditional mount, and passing that through would put a directory on PATH where an
# executable belongs. npu-smi matters because some submissions' build.sh shells out to it for the SoC.
maybe_mount() {   # $1 = required type (d|f), $2 = host path mounted at the same path in-container
    case "$1" in
        d) [[ -d "$2" ]] || return 0 ;;
        f) [[ -f "$2" ]] || return 0 ;;
    esac
    NPU_FLAGS+=(-v "$2:$2:ro")
}
maybe_mount d /usr/local/Ascend/driver
maybe_mount d /usr/local/dcmi
maybe_mount f /usr/local/bin/npu-smi
maybe_mount f /etc/ascend_install.info
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
