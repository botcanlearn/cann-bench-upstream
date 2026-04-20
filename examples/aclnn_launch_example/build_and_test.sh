
#!/bin/bash
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# ----------------------------------------------------------------------------------------------------------
# Build, install and test

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}"

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

echo "=== Building packages ==="
bash build.sh --soc=${SOC_VERSION}

echo "=== Installing run package (custom ops binary) ==="
RUN_PACKAGE=$(ls dist/*.run 2>/dev/null | head -1)
if [[ -n "${RUN_PACKAGE}" ]]; then
    # Install to CANN opp directory
    CANN_OPP_PATH="${ASCEND_HOME_PATH}/opp"
    if [[ -z "${CANN_OPP_PATH}" ]]; then
        echo "ERROR: ASCEND_HOME_PATH not set"
        exit 1
    fi
    "${RUN_PACKAGE}" --install-path="${CANN_OPP_PATH}" --quiet
    echo "Run package installed to: ${CANN_OPP_PATH}/vendors/custom_ops"
else
    echo "ERROR: No run package found in dist/"
    exit 1
fi

echo "=== Installing wheel package ==="
# Force reinstall only cann_bench, keep dependencies
pip install --force-reinstall --no-deps dist/cann_bench*.whl

echo "=== Setting custom ops environment ==="
source "${ASCEND_HOME_PATH}/opp/vendors/custom_ops/bin/set_env.bash"
echo "ASCEND_CUSTOM_OPP_PATH: ${ASCEND_CUSTOM_OPP_PATH}"

echo "=== Running tests ==="
pytest tests/ -v

echo "=== Build and test completed ==="