#!/bin/bash
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# ----------------------------------------------------------------------------------------------------------
# Build script for cann_bench_utils
#
# Usage:
#   bash build.sh [--soc=<soc>] [--clean]
#
# Environment:
#   PYTHON:    Python interpreter to build against (must match the runtime that
#              loads _C.abi3.so). Defaults to python3.
#   NPU_ARCH:  Target NPU architecture (ascend910b, ascend910_93, ascend950)
#              Default: auto-detect from acl.get_soc_name()

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Caller may override the Python interpreter (must match the runtime that loads _C.abi3.so).
# Defaults to python3.  Pass PYTHON=/path/to/python to avoid torch/Python ABI mismatch.
PYTHON="${PYTHON:-python3}"

CLEAN=0
NPU_ARCH=""

for arg in "$@"; do
  case "$arg" in
    --soc=*)   NPU_ARCH="${arg#*=}";;
    --clean)   CLEAN=1;;
    *) echo "Unknown arg: $arg"; exit 1;;
  esac
done

# Auto-detect NPU_ARCH if not specified
if [[ -z "$NPU_ARCH" ]]; then
  soc_name="$("$PYTHON" -c 'import acl; print(acl.get_soc_name() or "")' 2>/dev/null || true)"
  if [[ -n "$soc_name" ]]; then
    case "$soc_name" in
      Ascend910B*)   NPU_ARCH="ascend910b";;
      Ascend910_93*) NPU_ARCH="ascend910_93";;
      Ascend910_95*|Ascend950*) NPU_ARCH="ascend950";;
      *) echo "[ERROR] Unknown SoC: $soc_name"; exit 1;;
    esac
    echo "[INFO] Auto-detected NPU_ARCH=${NPU_ARCH} from ${soc_name}"
  else
    echo "[WARN] Cannot auto-detect, using default: ascend910b"
    NPU_ARCH="ascend910b"
  fi
fi

export NPU_ARCH

if [[ $CLEAN -eq 1 ]]; then
  echo "Cleaning..."
  "$PYTHON" setup.py clean
  rm -rf build dist *.egg-info cann_bench_utils/*.so
fi

"$PYTHON" -c "import wheel" 2>/dev/null || "$PYTHON" -m pip install wheel

echo "Building cann_bench_utils (NPU_ARCH=${NPU_ARCH}, PYTHON=$PYTHON)..."
"$PYTHON" setup.py bdist_wheel

echo "============================================"
echo "Build complete!"
echo "Install with: pip install dist/*.whl"
echo "============================================"
