#!/bin/bash
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# ----------------------------------------------------------------------------------------------------------
# Build, install and test

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}"

echo "=== Installing dependencies ==="
pip install -r requirements.txt

echo "=== Building wheel package ==="
bash build.sh --install

echo "=== Running tests ==="
pytest tests/ -v

echo "=== Build and test completed ==="