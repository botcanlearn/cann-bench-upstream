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
from typing import Optional, List

"""
QuantMatmul 算子 Torch Golden 参考实现

计算公式：
    无bias:     out = x1 @ x2 * scale + offset
    int32 bias: out = (x1 @ x2 + bias) * scale + offset
    浮点bias:   out = x1 @ x2 * scale + bias (无offset时)
    pertoken:   out = (x1 @ x2 * scale + offset) * pertoken_scale
"""


def quant_matmul(
    x1: torch.Tensor,
    x2: torch.Tensor,
    scale: torch.Tensor,
    offset: Optional[torch.Tensor] = None,
    pertoken_scale: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    output_dtype: Optional[str] = None,
    group_sizes: Optional[List[int]] = None,
) -> torch.Tensor:
    """
    量化矩阵乘法

    Args:
        x1: [..., m, k] int8/int32 左矩阵
        x2: [..., k, n] int8/int32 右矩阵
        scale: [t] 或 2D，反量化 scale
        offset: [t] 或 2D，反量化偏移（scale为2D时必选）
        pertoken_scale: [m] per-token scale
        bias: [n] 或 [batch, 1, n] 偏置，int32走pre-scale，浮点走post-scale
        output_dtype: 输出类型 "int8"/"float16"/"bfloat16"/"int32"，默认int8
        group_sizes: 分组量化粒度 [group_m, group_n, group_k]

    Returns:
        out: [..., m, n]
    """
    # 矩阵乘（int8/int32 用 float32 等效计算）
    mm = torch.matmul(x1.float(), x2.float())

    # int32 bias 在反量化前累加 (pre-scale)
    if bias is not None and bias.dtype == torch.int32:
        mm = mm + bias.float()

    # 反量化 scale
    y = mm * scale.float()

    # offset
    if offset is not None:
        y = y + offset.float()

    # pertoken_scale 沿 m 维广播
    if pertoken_scale is not None:
        y = y * pertoken_scale.float().unsqueeze(-1)

    # 浮点 bias 在反量化后相加（仅无 offset 时）
    if bias is not None and bias.dtype != torch.int32 and offset is None:
        y = y + bias.float()

    # 输出 dtype
    if output_dtype is None or output_dtype == "int8":
        out_dtype = torch.int8
    elif output_dtype == "float16":
        out_dtype = torch.float16
    elif output_dtype == "bfloat16":
        out_dtype = torch.bfloat16
    elif output_dtype == "int32":
        out_dtype = torch.int32
    else:
        raise ValueError(f"unsupported output_dtype: {output_dtype}")

    return y.to(out_dtype)