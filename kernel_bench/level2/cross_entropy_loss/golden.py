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
CrossEntropyLoss 算子 Torch Golden 参考实现

计算交叉熵损失，用于分类任务

公式:
    L = -log(exp(x[target]) / sum(exp(x)))
    或带 weight: L = -weight[target] * log(exp(x[target]) / sum(exp(x)))

参考 PyTorch API: torch.nn.CrossEntropyLoss
    https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html

Parameters:
    - input: (N, C) 或 (N, C, H, W) 等 - logits 张量（未经 softmax）
    - target: (N,) 硬标签 或 (N, C) 软标签（概率分布）
    - weight: (C,) 各类别的权重（可选）
    - ignore_index: int, 默认 -100 - 忽略的标签索引
    - reduction: 'none' | 'mean' | 'sum', 默认 'mean' - 损失聚合方式
"""


def cross_entropy_loss(
    x: torch.Tensor,
    target: torch.Tensor,
    reduction: str = 'mean',
    ignore_index: int = -100
) -> torch.Tensor:
    """
    计算交叉熵损失

    Args:
        x: 输入 logits 张量，shape (N, C) 或 (N, C, d1, d2, ...)
           N = batch size, C = 类别数（channel_first 约定）
           注意：输入应为 logits（未经 softmax），内部会自动应用 log_softmax
        target: 目标标签
               - 硬标签：shape (N,) 或 (N, d1, d2, ...)，值为类别索引
               - 软标签：shape (N, C)，值为概率分布
        reduction: 损失聚合方式
                  'none': 返回每个样本的损失，shape (N,)
                  'mean': 返回 batch 平均损失
                  'sum': 返回 batch 总损失
        ignore_index: 忽略的标签索引
                      当 target 为硬标签且值为 ignore_index 时，该样本不计入损失

    Returns:
        损失值：如果 reduction='none'，返回 shape (N,) 的张量
               否则返回标量张量

    Examples:
        >>> N, C = 16, 10  # 16个样本，10个类别
        >>> x = torch.randn(N, C)
        >>> target = torch.randint(0, C, (N,))
        >>> loss = cross_entropy_loss(x, target)
    """
    # 直接调用 PyTorch 标准 CrossEntropyLoss 实现
    # torch.nn.functional.cross_entropy 内部会自动应用 log_softmax
    loss = torch.nn.functional.cross_entropy(
        input=x,
        target=target,
        reduction=reduction,
        ignore_index=ignore_index
    )

    return loss