#!/bin/bash
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# ----------------------------------------------------------------------------------------------------------
# Build both run package and wheel package

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="${SCRIPT_DIR}"

# Parse arguments
SOC_VERSION="ascend910b"
while [[ $# -gt 0 ]]; do
    case $1 in
        --soc=*)
            SOC_VERSION="${1#*=}"
            shift
            ;;
        *)
            shift
            ;;
    esac
done

echo "=== Building cann_bench packages ==="
echo "SOC: ${SOC_VERSION}"

# Clean dist directory
DIST_DIR="${PROJECT_DIR}/dist"
rm -rf "${DIST_DIR}"
mkdir -p "${DIST_DIR}"

# Build run package
bash "${PROJECT_DIR}/scripts/build_run.sh" --soc=${SOC_VERSION}

# Build wheel package
bash "${PROJECT_DIR}/scripts/build_wheel.sh"

echo ""
echo "=== Build complete ==="
echo "Output directory: ${DIST_DIR}"
ls -la "${DIST_DIR}"