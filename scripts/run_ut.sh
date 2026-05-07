#!/bin/bash
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
#
# 单元测试运行脚本
#
# 用法:
#   ./run_ut.sh                         # 运行全部单元测试
#   ./run_ut.sh -v                      # 详细模式
#   ./run_ut.sh -k "config"             # 按关键字筛选
#   ./run_ut.sh -f test_config.py       # 指定文件
#   ./run_ut.sh -f test_config.py -t TestConfig::test_default_config  # 指定方法
#   ./run_ut.sh -s                      # 不捕获输出（调试用）
#   ./run_ut.sh -x                      # 首次失败即停止
#   ./run_ut.sh -q                      # 静默模式（只显示结果）
#   ./run_ut.sh --pdb                   # 失败时进入调试器

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${SCRIPT_DIR}/.."
SRC_DIR="${PROJECT_DIR}/src"
TEST_DIR="${PROJECT_DIR}/tests/unit"

# 默认选项
VERBOSE=""
KEYWORD=""
TARGET_FILE=""
TARGET_TEST=""
CAPTURE=""
FAIL_FAST=""
QUIET=""
PDB=""
JOBS=""
EXTRA_ARGS=()

show_help() {
    cat <<EOF
单元测试运行脚本

用法: $0 [选项]

选项:
  -v, --verbose       详细模式，显示每个测试名称和结果
  -q, --quiet         静默模式，只显示最终统计
  -k, --keyword <kw>  按关键字筛选测试（pytest -k）
  -f, --file <name>   指定测试文件（如 test_config.py，自动定位到 tests/unit/）
  -t, --test <spec>   指定测试方法（如 TestConfig::test_default_config）
  -x, --fail-fast     首次失败即停止
  -s, --no-capture    不捕获 stdout/stderr（调试用）
  -j, --jobs <n>      并行执行（pytest-xdist）
  --pdb               失败时进入 pdb 调试器
  -h, --help          显示帮助

示例:
  $0                                  # 全部单元测试
  $0 -v                               # 详细模式
  $0 -k "config"                      # 运行名称含 config 的测试
  $0 -f test_config.py                # 运行指定测试文件
  $0 -f test_config.py -t TestConfig::test_default_config  # 指定测试方法
  $0 -x -v                            # 失败即停 + 详细输出
  $0 -s --pdb                         # 无捕获 + 失败调试

EOF
    exit 0
}

# 解析参数
while [[ $# -gt 0 ]]; do
    case "$1" in
        -v|--verbose)   VERBOSE="-v" ;;
        -q|--quiet)     QUIET="-q" ;;
        -k|--keyword)   shift; KEYWORD="$1" ;;
        -f|--file)      shift; TARGET_FILE="$1" ;;
        -t|--test)      shift; TARGET_TEST="$1" ;;
        -x|--fail-fast) FAIL_FAST="-x" ;;
        -s|--no-capture) CAPTURE="-s" ;;
        -j|--jobs)      shift; JOBS="-n $1" ;;
        --pdb)          PDB="--pdb" ;;
        -h|--help)      show_help ;;
        *)              EXTRA_ARGS+=("$1") ;;
    esac
    shift
done

# 构建 pytest 命令
PYTEST_ARGS=()

# 测试目标
if [ -n "$TARGET_FILE" ]; then
    if [ -f "${TEST_DIR}/${TARGET_FILE}" ]; then
        TARGET_PATH="${TEST_DIR}/${TARGET_FILE}"
    elif [ -f "${TARGET_FILE}" ]; then
        TARGET_PATH="${TARGET_FILE}"
    else
        echo "[ERROR] 找不到测试文件: ${TARGET_FILE}"
        echo "  已搜索: ${TEST_DIR}/${TARGET_FILE}"
        exit 1
    fi

    if [ -n "$TARGET_TEST" ]; then
        TARGET_PATH="${TARGET_PATH}::${TARGET_TEST}"
    fi
    PYTEST_ARGS+=("${TARGET_PATH}")
else
    PYTEST_ARGS+=("${TEST_DIR}")
fi

# 选项
[ -n "$VERBOSE" ]   && PYTEST_ARGS+=("$VERBOSE")
[ -n "$QUIET" ]     && PYTEST_ARGS+=("$QUIET")
[ -n "$FAIL_FAST" ] && PYTEST_ARGS+=("$FAIL_FAST")
[ -n "$CAPTURE" ]   && PYTEST_ARGS+=("$CAPTURE")
[ -n "$KEYWORD" ]   && PYTEST_ARGS+=("-k" "$KEYWORD")
[ -n "$JOBS" ]      && PYTEST_ARGS+=($JOBS)
[ -n "$PDB" ]       && PYTEST_ARGS+=("$PDB")
[ ${#EXTRA_ARGS[@]} -gt 0 ] && PYTEST_ARGS+=("${EXTRA_ARGS[@]}")

# 打印即将执行的命令
echo "=========================================="
echo "  单元测试"
echo "=========================================="
echo "  PYTHONPATH=${SRC_DIR}"
echo "  pytest ${PYTEST_ARGS[*]}"
echo "=========================================="
echo ""

# 执行
PYTHONPATH="${SRC_DIR}" python -m pytest "${PYTEST_ARGS[@]}"
