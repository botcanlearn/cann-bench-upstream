#!/bin/bash
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# ----------------------------------------------------------------------------------------------------------
# Build Python wheel package

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DIST_DIR="${PROJECT_DIR}/dist"

echo "=== Building Python wheel package ==="

# Clean previous wheel builds
rm -rf "${PROJECT_DIR}/build" "${PROJECT_DIR}/*.egg-info"

# Ensure dist directory exists
mkdir -p "${DIST_DIR}"

# Build wheel using setup.py
cd "${PROJECT_DIR}"
python3 setup.py bdist_wheel --dist-dir "${DIST_DIR}"

# Rename wheel to standard format
WHEEL_FILE=$(find "${DIST_DIR}" -name "*.whl" -type f | head -1)
if [[ -z "${WHEEL_FILE}" ]]; then
    echo "ERROR: No wheel package found"
    exit 1
fi

# Extract version from setup.py
VERSION=$(python3 -c "import setup; print(setup.VERSION)" 2>/dev/null || echo "1.0.0")
WHEEL_FILENAME="cann_bench-${VERSION}-cp38-abi3-linux_aarch64.whl"

# Rename if needed
if [[ "$(basename "${WHEEL_FILE}")" != "${WHEEL_FILENAME}" ]]; then
    mv "${WHEEL_FILE}" "${DIST_DIR}/${WHEEL_FILENAME}"
fi

echo "=== Wheel package built successfully ==="
echo "Output: ${DIST_DIR}/${WHEEL_FILENAME}"