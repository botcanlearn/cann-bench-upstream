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
Engram算子Torch Golden参考实现

记忆增强的注意力机制中的记忆编码与检索融合操作
公式: y = x + alpha * softmax(x @ memory^T / sqrt(d)) @ memory
"""
def engram(
    x: torch.Tensor, memory: torch.Tensor, alpha: float = 1.0, scale: float = -1.0
) -> torch.Tensor:
    """
    Engram 记忆增强注意力算子

    公式: y = x + alpha * softmax(x @ memory^T / sqrt(d)) @ memory

    Args:
        x: 输入特征张量，shape [B, S, D]
        memory: 记忆库张量，shape [B, M, D]
        alpha: 记忆增强系数
        scale: 缩放因子，<=0 表示自动使用 1/sqrt(D)

    Returns:
        y: 记忆增强后的输出张量，shape [B, S, D]
    """
    d = x.shape[-1]
    if scale <= 0:
        scale = 1.0 / (d ** 0.5)
    scores = torch.matmul(x, memory.transpose(-2, -1)) * scale
    attn = torch.nn.functional.softmax(scores, dim=-1)
    mem_out = torch.matmul(attn, memory)
    return x + alpha * mem_out
