#!/bin/bash
# Build the cann-bench-st CI image on an NPU host (run from the repo root). Pairs with
# tests/st/run_st_docker.sh, which runs the ST using this image. The image is toolchain-only
# (CANN base + torch/torch_npu + pytest) with no repo COPY — so it's cache-friendly and only
# needs rebuilding when the deps in tests/st/docker/Dockerfile change, not when code changes.
#
#   bash tests/st/build_st_docker_image.sh
#
# Overridable via env (kept in sync with tests/st/run_st_docker.sh's NPU_IMAGE):
#   NPU_IMAGE        image tag to build      (default cann-bench-st:latest)
#   PYPI_INDEX_URL   pip/uv mirror override  (default: Huawei Cloud, see Dockerfile)
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
NPU_IMAGE="${NPU_IMAGE:-cann-bench-st:latest}"

cd "$REPO"
build_args=()
[ -n "${PYPI_INDEX_URL:-}" ] && build_args+=(--build-arg "PYPI_INDEX_URL=$PYPI_INDEX_URL")

docker build -t "$NPU_IMAGE" "${build_args[@]}" -f tests/st/docker/Dockerfile .
echo "built: $NPU_IMAGE"
docker images --format '{{.Repository}}:{{.Tag}}  {{.Size}}  {{.CreatedSince}}' | grep -F "${NPU_IMAGE%%:*}" || true
