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
SwiGlu算子Torch Golden参考实现

采用Swish作为激活函数的GLU变体，输入在第-1维拆分成x0和x1两部分
公式: y = swish(x0) * x1 = x0 * sigmoid(beta * x0) * x1
"""
def swi_glu(
    x: torch.Tensor, scalarValue: float
) -> torch.Tensor:
    """
    采用Swish作为激活函数的GLU变体，输入在第-1维拆分成x0和x1两部分

    公式: y = swish(x0) * x1 = x0 * sigmoid(beta * x0) * x1

    Args:
        x: 输入张量，会在-1维拆分成x0和x1
        scalarValue: Swish激活函数的beta参数

    Returns:
        输出张量，形状为输入shape除以2
    """

    # 在最后一维拆分为两部分
    last_dim_size = x.shape[-1]

    # 对于奇数维度，只取前偶数个元素进行拆分，确保两部分大小一致
    if last_dim_size % 2 != 0:
        # 取前 floor(n/2)*2 个元素
        usable_size = (last_dim_size // 2) * 2
        x = x[..., :usable_size]

    x0, x1 = x.chunk(2, dim=-1)
    swish = x0 * torch.sigmoid(scalarValue * x0)
    y = swish * x1
    return y
