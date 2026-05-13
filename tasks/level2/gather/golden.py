#!/usr/bin/python3
# coding=utf-8

# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

import torch

"""
Gather算子Torch Golden参考实现（PyTorch torch.gather 语义）

沿 dim 维按 index 逐元素提取，输出 shape 与 index.shape 一致。
公式: output[i_0,...,i_{n-1}] = x[..., index[i_0,...,i_{n-1}], ...]
      （第 dim 维替换为 index 给出的下标，其余维度保留索引位置）
"""
def gather(
    x: torch.Tensor, index: torch.Tensor, dim: int = 0
) -> torch.Tensor:
    """
    沿 dim 维按 index 逐元素提取（torch.gather 语义）。

    Args:
        x: 输入张量（数据源）
        index: 索引张量，需与 x 维度数相同；除 dim 维外，shape 各维不大于 x
        dim: gather 维度索引，默认 0

    Returns:
        输出张量，shape 与 index 完全一致，dtype 与 x 一致
    """

    # 不做 .long()：PyTorch 2.1+ 的 torch.gather 已接受任意整型 idx（int32/int64 均可），
    # 在 NPU 上 .long() 会触发冗余 Cast (int32→int64) + 后端再 Cast 回 int32。
    y = torch.gather(x, dim, index)
    return y
