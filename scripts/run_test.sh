#!/bin/bash
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
#
# 测试运行脚本
#
# 用法:
#   ./run_test.sh                       # 运行全部测试（ut + e2e）
#   ./run_test.sh ut                    # 只运行单元测试
#   ./run_test.sh e2e                   # 只运行端到端测试
#   ./run_test.sh ut -v                 # 单元测试 + 详细模式
#   ./run_test.sh ut -k "config"        # 单元测试 + 按关键字筛选
#   ./run_test.sh ut -f test_config.py  # 单元测试 + 指定文件
#   ./run_test.sh e2e -v                # 端到端测试 + 详细模式

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${SCRIPT_DIR}/.."
SRC_DIR="${PROJECT_DIR}/src"
TESTS_DIR="${PROJECT_DIR}/tests"

# 默认：执行全部
TARGETS=("ut" "e2e")
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
测试运行脚本

用法: $0 [ut|e2e] [选项]

测试类型:
  ut                  只运行单元测试（tests/ut/）
  e2e                 只运行端到端测试（tests/e2e/）
  不指定              运行全部测试（ut + e2e）

选项:
  -v, --verbose       详细模式，显示每个测试名称和结果
  -q, --quiet         静默模式，只显示最终统计
  -k, --keyword <kw>  按关键字筛选测试（pytest -k）
  -f, --file <name>   指定测试文件（如 test_config.py，自动定位到对应 tests 子目录）
  -t, --test <spec>   指定测试方法（如 TestConfig::test_default_config）
  -x, --fail-fast     首次失败即停止
  -s, --no-capture    不捕获 stdout/stderr（调试用）
  -j, --jobs <n>      并行执行（pytest-xdist）
  --pdb               失败时进入 pdb 调试器
  -h, --help          显示帮助

示例:
  $0                                  # 全部测试（ut + e2e）
  $0 ut                               # 只跑单元测试
  $0 e2e                              # 只跑端到端测试
  $0 ut -v                            # 单元测试 + 详细模式
  $0 ut -k "config"                   # 单元测试 + 按关键字筛选
  $0 ut -f test_config.py             # 单元测试 + 指定文件
  $0 ut -f test_config.py -t TestConfig::test_default_config  # 指定测试方法
  $0 e2e -v                           # 端到端测试 + 详细模式
  $0 -x -v                            # 全部测试 + 失败即停 + 详细输出

EOF
    exit 0
}

# 第一个非选项参数可能是测试类型
if [[ $# -gt 0 ]] && [[ "$1" == "ut" || "$1" == "e2e" ]]; then
    TARGETS=("$1")
    shift
fi

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

# 构建 pytest 公共选项
COMMON_ARGS=()
[ -n "$VERBOSE" ]   && COMMON_ARGS+=("$VERBOSE")
[ -n "$QUIET" ]     && COMMON_ARGS+=("$QUIET")
[ -n "$FAIL_FAST" ] && COMMON_ARGS+=("$FAIL_FAST")
[ -n "$CAPTURE" ]   && COMMON_ARGS+=("$CAPTURE")
[ -n "$KEYWORD" ]   && COMMON_ARGS+=("-k" "$KEYWORD")
[ -n "$JOBS" ]      && COMMON_ARGS+=($JOBS)
[ -n "$PDB" ]       && COMMON_ARGS+=("$PDB")
[ ${#EXTRA_ARGS[@]} -gt 0 ] && COMMON_ARGS+=("${EXTRA_ARGS[@]}")

# 运行指定测试类型
run_target() {
    local target="$1"
    local dir="${TESTS_DIR}/${target}"
    local label="单元测试"
    if [[ "$target" == "e2e" ]]; then
        label="端到端测试"
    fi

    local PYTEST_ARGS=("${COMMON_ARGS[@]}")

    if [ -n "$TARGET_FILE" ]; then
        if [ -f "${dir}/${TARGET_FILE}" ]; then
            TARGET_PATH="${dir}/${TARGET_FILE}"
        elif [ -f "${TARGET_FILE}" ]; then
            TARGET_PATH="${TARGET_FILE}"
        else
            echo "[ERROR] 找不到测试文件: ${TARGET_FILE}"
            echo "  已搜索: ${dir}/${TARGET_FILE}"
            return 1
        fi

        if [ -n "$TARGET_TEST" ]; then
            TARGET_PATH="${TARGET_PATH}::${TARGET_TEST}"
        fi
        PYTEST_ARGS+=("${TARGET_PATH}")
    else
        PYTEST_ARGS+=("${dir}")
    fi

    echo "=========================================="
    echo "  ${label}"
    echo "=========================================="
    echo "  PYTHONPATH=${SRC_DIR}"
    echo "  pytest ${PYTEST_ARGS[*]}"
    echo "=========================================="
    echo ""

    PYTHONPATH="${SRC_DIR}" python -m pytest "${PYTEST_ARGS[@]}"
}

for target in "${TARGETS[@]}"; do
    run_target "$target"
done