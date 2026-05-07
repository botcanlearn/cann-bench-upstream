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
Scatter算子Torch Golden参考实现

将updates按索引indices更新到data中
公式: y[index[i][j][k]] = src[i][j][k] (if dim == 0)
"""
def scatter(
    data: torch.Tensor, dim: int, indices: torch.Tensor, updates: torch.Tensor, reduce: str = None
) -> torch.Tensor:
    """
    将updates按索引indices更新到data中

    公式: y[index[i][j][k]] = src[i][j][k] (if dim == 0)

    Args:
        data: 输入数据张量
        dim: 沿哪个维度scatter
        indices: 索引张量
        updates: 更新值张量
        reduce: 聚合方式 (None/update, add, multiply, amin, amax)

    Returns:
        输出张量，scatter结果
    """

    y = data.clone()
    idx = indices.long()
    if reduce is None or reduce == 'update':
        y.scatter_(dim, idx, updates)
    elif reduce == 'add':
        y.scatter_add_(dim, idx, updates)
    elif reduce == 'multiply':
        y.scatter_reduce_(dim, idx, updates, reduce="prod", include_self=True)
    elif reduce == 'amin':
        y.scatter_reduce_(dim, idx, updates, reduce="amin", include_self=True)
    elif reduce == 'amax':
        y.scatter_reduce_(dim, idx, updates, reduce="amax", include_self=True)
    return y
