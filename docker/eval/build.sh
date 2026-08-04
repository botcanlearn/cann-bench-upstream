#!/usr/bin/env bash
# Build the cann-bench-eval image. Run from anywhere; the build context is forced to the repo root
# because the Dockerfile COPYs src/ tasks/ scripts/ (see .dockerignore for what is kept out).
#
#   bash docker/eval/build.sh                          # this host's arch, 910b, OPS_MODE=none
#   NPU_ARCH=ascend950 bash docker/eval/build.sh       # 950PR -- OPS_MODE defaults to refonly (see below)
#   NPU_ARCH=ascend910_93 bash docker/eval/build.sh    # A3
#   OPS_MODE=full bash docker/eval/build.sh            # cheatable; for re-collecting aclnn baselines
#   TRITON_ASCEND_VERSION=3.2.1 bash docker/eval/build.sh
#
# CN mirrors (see README): MIRROR=cn bash docker/eval/build.sh

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# ARCH is the BUILD HOST's -- this is a native build, not a cross-compile. Overriding it to something
# other than `uname -m` produces an image that cannot run here. Normalise the aliases: CANN's .run
# names and the toolkit's <arch>-linux dirs use aarch64/x86_64, while uname says arm64 on macOS and
# amd64 is the common docker spelling -- an unnormalised value silently builds a 404 download URL.
ARCH="${ARCH:-$(uname -m)}"
case "${ARCH}" in
    arm64|aarch64) ARCH=aarch64 ;;
    amd64|x86_64)  ARCH=x86_64 ;;
    *) echo "unsupported ARCH=${ARCH} (expected aarch64 | x86_64)" >&2; exit 1 ;;
esac
NPU_ARCH="${NPU_ARCH:-ascend910b}"
CANN_VERSION="${CANN_VERSION:-9.0.1}"
BASE_IMAGE="${BASE_IMAGE:-cann-toolkit-base:${CANN_VERSION}-py3.13}"
TRITON_ASCEND_VERSION="${TRITON_ASCEND_VERSION:-}"

# The ops .run spells the SoC differently from the compiler flag (ascend910b -> 910b, ascend950 -> 950).
case "${NPU_ARCH}" in
    ascend910b)   OPS_PKG="${OPS_PKG:-910b}"   ; DEFAULT_OPS_MODE=none ;;
    ascend910_93) OPS_PKG="${OPS_PKG:-910_93}" ; DEFAULT_OPS_MODE=none ;;
    # 950 cannot run OPS_MODE=none -- even a bare .npu() copy needs aclnnInplaceCopy there (ERR01007).
    # refonly still blocks builtins, so the anti-cheat posture is preserved. See docker/eval/README.md.
    ascend950)    OPS_PKG="${OPS_PKG:-950}"    ; DEFAULT_OPS_MODE=refonly ;;
    *) echo "unknown NPU_ARCH=${NPU_ARCH} (expected ascend910b | ascend910_93 | ascend950)" >&2; exit 1 ;;
esac
OPS_MODE="${OPS_MODE:-${DEFAULT_OPS_MODE}}"

VERSION="$(cat VERSION)"
# Tag carries everything that changes what a score means: benchmark version, SoC, CPU arch, ops posture.
IMAGE="${IMAGE:-cann-bench-eval:${VERSION}-${NPU_ARCH}-${ARCH}-ops${OPS_MODE}}"

# No ARCH / CANN_VERSION build-arg: the eval layer inherits CANN_ARCH and CANN_VERSION from the base
# image. Both are used here only for the tag, the base-image name, and the consistency check below.
ARGS=(
    --build-arg "BASE_IMAGE=${BASE_IMAGE}"
    --build-arg "OPS_MODE=${OPS_MODE}"
    --build-arg "OPS_PKG=${OPS_PKG}"
    --build-arg "NPU_ARCH=${NPU_ARCH}"
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
    echo "==> base image ${BASE_IMAGE} not found. Build it first (same ARCH, same host):" >&2
    echo "    cd docker/base && docker build --build-arg ARCH=${ARCH} -t ${BASE_IMAGE} ." >&2
    exit 1
fi

# The base owns the architecture and the CANN release; fail loudly here rather than let the tag claim
# one thing while the image is another. (The Dockerfile re-checks the arch against `uname -m` inside
# the build, and refuses a base that publishes neither variable.)
BASE_ENV="$(docker image inspect "${BASE_IMAGE}" --format '{{range .Config.Env}}{{println .}}{{end}}')"
BASE_ARCH="$(sed -n 's/^CANN_ARCH=//p' <<<"${BASE_ENV}")"
BASE_CANN="$(sed -n 's/^CANN_VERSION=//p' <<<"${BASE_ENV}")"
if [[ -z "${BASE_ARCH}" || -z "${BASE_CANN}" ]]; then
    MISSING=""
    [[ -z "${BASE_ARCH}" ]] && MISSING="CANN_ARCH"
    [[ -z "${BASE_CANN}" ]] && MISSING="${MISSING:+${MISSING} }CANN_VERSION"
    echo "==> ${BASE_IMAGE} publishes no ${MISSING} -- it predates the cross-arch change; rebuild docker/base." >&2
    exit 1
fi
if [[ "${BASE_ARCH}" != "${ARCH}" ]]; then
    echo "==> ${BASE_IMAGE} is ${BASE_ARCH} but this host is ${ARCH}; rebuild the base here." >&2
    exit 1
fi

echo "==> building ${IMAGE}  (base=${BASE_IMAGE} cann=${BASE_CANN} arch=${ARCH} soc=${NPU_ARCH} ops=${OPS_MODE}/${OPS_PKG})"
set -x
docker build --network=host -f docker/eval/Dockerfile -t "${IMAGE}" "${ARGS[@]}" "$@" .
set +x
echo "==> built ${IMAGE}"
echo "==> smoke:  IMAGE=${IMAGE} bash docker/eval/run.sh self-test"
