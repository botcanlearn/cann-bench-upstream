#!/bin/bash
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You can not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
KERNELBENCH_DIR="${PROJECT_ROOT}/bench_lab/stanford_bench/KernelBench"

KERNELBENCH_REPO_URL="https://github.com/ScalingIntelligence/KernelBench.git"
KERNELBENCH_COMMIT="21fbe5a642898cd60b8f60c7aefb43d475e11f33"

function check_git() {
  if ! command -v git &> /dev/null; then
    echo "错误: 未找到 git，请先安装 git"
    exit 1
  fi
}

function download_stanfordbench() {
  check_git

  if [ -d "${KERNELBENCH_DIR}" ]; then
    echo "StanfordBench 数据已存在: ${KERNELBENCH_DIR}，跳过下载。如需重新下载，请先删除该目录。"
    return 0
  fi

  echo "正在克隆 StanfordBench (KernelBench) 到 ${KERNELBENCH_DIR}..."
  git clone "${KERNELBENCH_REPO_URL}" "${KERNELBENCH_DIR}"

  if [ -n "${KERNELBENCH_COMMIT}" ]; then
    echo "切换 StanfordBench 到 commit ${KERNELBENCH_COMMIT}..."
    git -C "${KERNELBENCH_DIR}" checkout "${KERNELBENCH_COMMIT}"
  fi

  echo "StanfordBench 下载完成！"
  echo "  路径: ${KERNELBENCH_DIR}"
  echo "  注意: 数据目录名为 KernelBench（GitHub 仓库原名），评测时使用 --bench-name stanford"
}

download_stanfordbench

echo ""
echo "========== 下载完成 =========="
echo "  StanfordBench: ${KERNELBENCH_DIR}"
