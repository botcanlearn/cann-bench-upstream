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
Cummin算子Torch Golden参考实现

计算输入张量中的累积最小值
公式: y[i] = min(x[0], x[1], ..., x[i]) 沿指定轴
"""
def cummin(
    x: torch.Tensor, dim: int
) -> torch.Tensor:
    """
    计算输入张量中的累积最小值

    公式: y[i] = min(x[0], x[1], ..., x[i]) 沿指定轴

    Args:
        x: 输入张量
        dim: 计算累积最小值的轴

    Returns:
        输出张量，累积最小值
    """

    y = torch.cummin(x, dim=dim)[0]
    return y
