#!/usr/bin/env bash
# Build the cann-bench-eval image. Run from anywhere; the build context is forced to the repo root
# because the Dockerfile COPYs src/ tasks/ scripts/ (see .dockerignore for what is kept out).
#
#   bash docker/eval/build.sh                          # OPS_MODE=none (default), 910b
#   OPS_MODE=refonly bash docker/eval/build.sh         # ops installed, kernel binaries stripped
#   NPU_ARCH=ascend910_93 bash docker/eval/build.sh    # A3
#   TRITON_ASCEND_VERSION=3.2.1 bash docker/eval/build.sh
#
# CN mirrors (see README): MIRROR=cn bash docker/eval/build.sh

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

BASE_IMAGE="${BASE_IMAGE:-cann-toolkit-base:9.0.1-py3.13}"
OPS_MODE="${OPS_MODE:-none}"
NPU_ARCH="${NPU_ARCH:-ascend910b}"
CANN_VERSION="${CANN_VERSION:-9.0.1}"
TRITON_ASCEND_VERSION="${TRITON_ASCEND_VERSION:-}"

VERSION="$(cat VERSION)"
# Tag carries everything that changes what a score means: benchmark version, ops posture, SoC.
IMAGE="${IMAGE:-cann-bench-eval:${VERSION}-${NPU_ARCH}-ops${OPS_MODE}}"

ARGS=(
    --build-arg "BASE_IMAGE=${BASE_IMAGE}"
    --build-arg "OPS_MODE=${OPS_MODE}"
    --build-arg "NPU_ARCH=${NPU_ARCH}"
    --build-arg "CANN_VERSION=${CANN_VERSION}"
    --build-arg "TRITON_ASCEND_VERSION=${TRITON_ASCEND_VERSION}"
)

# One switch for the whole in-region mirror set, rather than five build-args at every call site.
if [[ "${MIRROR:-}" == "cn" ]]; then
    ARGS+=(
        --build-arg "PYPI_MIRROR=https://mirrors.huaweicloud.com/repository/pypi"
        --build-arg "TORCH_MIRROR=https://mirror.nju.edu.cn/pytorch/whl/cpu"
        --build-arg "UV_PYTHON_INSTALL_MIRROR=https://mirror.nju.edu.cn/github-release/astral-sh/python-build-standalone"
    )
fi

if ! docker image inspect "${BASE_IMAGE}" >/dev/null 2>&1; then
    echo "==> base image ${BASE_IMAGE} not found; build it first: cd docker/base && docker build -t ${BASE_IMAGE} ." >&2
    exit 1
fi

echo "==> building ${IMAGE}  (base=${BASE_IMAGE} ops=${OPS_MODE} soc=${NPU_ARCH})"
set -x
docker build --network=host -f docker/eval/Dockerfile -t "${IMAGE}" "${ARGS[@]}" "$@" .
set +x
echo "==> built ${IMAGE}"
echo "==> smoke:  IMAGE=${IMAGE} bash docker/eval/run.sh self-test"
