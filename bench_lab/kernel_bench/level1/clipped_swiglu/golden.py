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
from typing import Optional


def clipped_swiglu(x: torch.Tensor, group_index: Optional[torch.Tensor] = None,
                    dim: int = -1, alpha: float = 1.702,
                    limit: float = 7.0, bias: float = 1.0,
                    interleaved: bool = True):
    """
    带截断的 Swish 门控线性单元激活函数。

    Args:
        x: 输入张量，数据类型为 float16 或 bfloat16
        group_index: 可选，MoE 分组索引，1 维 INT64 张量，None 表示不使用分组
        dim: 合轴及切分的维度序号，默认 -1
        alpha: SwiGLU 参数，默认 1.702
        limit: 门限值，默认 7.0
        bias: 偏差参数，默认 1.0
        interleaved: true=奇偶切分，false=前后切分，默认 true

    Returns:
        y: 输出张量，dim 对应维度为 x 的一半，数据类型与 x 一致
    """
    dim = dim if dim >= 0 else dim + x.dim()
    dtype = x.dtype
    if x.dtype in [torch.bfloat16, torch.float16]:
        x = x.to(torch.float32)
    shape = list(x.shape)
    if x.ndim > 1:
        if dim != 0:
            dim1 = int(torch.prod(torch.tensor(shape[:dim])).item())
            x = x.reshape(dim1, int(torch.prod(torch.tensor(shape[dim:])).item())).clone()
        else:
            x = x.reshape(1, int(torch.prod(torch.tensor(shape[dim:])).item())).clone()
    else:
        x = x.reshape(1, shape[0]).clone()
    group = x.shape[0]
    if group_index is not None:
        group = min(int(torch.sum(group_index).item()), x.shape[0])
    x_tensor = x[:group]
    remain_tensor = torch.zeros_like(x[group:, :x.shape[1] // 2])
    if interleaved:
        x_glu = x_tensor[..., ::2]
        x_linear = x_tensor[..., 1::2]
    else:
        out = torch.chunk(x_tensor, 2, dim=-1)
        x_glu = out[0]
        x_linear = out[1]
    x_glu = x_glu.clamp(min=None, max=limit)
    x_linear = x_linear.clamp(min=-limit, max=limit)
    sigmoid_part = torch.sigmoid(alpha * x_glu)
    result = x_glu * sigmoid_part * (x_linear + bias)
    result = torch.cat((result, remain_tensor), dim=0)
    res_shape = list(shape)
    res_shape[dim] = res_shape[dim] // 2
    if result.numel() != 0:
        result = result.reshape(res_shape)
    return result.to(dtype)

