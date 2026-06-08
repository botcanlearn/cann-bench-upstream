#!/bin/bash
# Run the golden-candidate NPU integration ST inside the CI image on an NPU host, wrapping
# the long `docker run` (NPU device + host driver mounts) into a one-liner. Delegates to
# tests/st/run_st.sh inside the container — that script is the single source of truth for the
# pytest invocation + artifact handling. Run this ON the NPU server, from the cann-bench repo
# root (which gets mounted as /workspace):
#
#   bash tests/st/run_st_docker.sh             # default: quick Cummin smoke
#   bash tests/st/run_st_docker.sh -m level1   # one level (any pytest args pass through)
#   bash tests/st/run_st_docker.sh -k Gelu     # one op
#   bash tests/st/run_st_docker.sh --full      # full 53-op matrix
#
# Artifacts land in tests/st/_artifacts/ (matrix_junit.xml + eval_*.json|.md, flat) — ST-owned
# and gitignored, on the bind-mounted /workspace, so they persist after the transient container exits.
#
# Build the image first (once):  bash tests/st/build_st_docker_image.sh
#
# Overridable via env:
#   NPU_IMAGE  image to run                        (default cann-bench-st:latest)
#   DEVICE     ASCEND_RT_VISIBLE_DEVICES, free card (default 0)
#   WORKSPACE  host dir mounted to /workspace      (default: repo root)
#   ST_OUT     artifact dir (forwarded to run_st.sh, default tests/st/_artifacts)
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
NPU_IMAGE="${NPU_IMAGE:-cann-bench-st:latest}"
DEVICE="${DEVICE:-0}"
WORKSPACE="${WORKSPACE:-$REPO}"

exec docker run --rm --privileged --ipc=host \
  --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc \
  -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64:ro \
  -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info:ro \
  -v /usr/local/dcmi:/usr/local/dcmi:ro \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
  -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
  -e ASCEND_RT_VISIBLE_DEVICES="$DEVICE" -e ST_OUT="${ST_OUT:-tests/st/_artifacts}" \
  -v "$WORKSPACE":/workspace -w /workspace \
  "$NPU_IMAGE" \
  bash -c 'exec bash tests/st/run_st.sh "$@"' bash "$@"
