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
DynamicQuant算子Torch Golden参考实现

对输入张量进行per-token对称动态量化
公式: scaleOut = row_max(abs(x)) / dtypeMax, yOut = round(x / scaleOut)
"""
def dynamic_quant(
    x: torch.Tensor, axis: int = -1, dst_type: int = 0
) -> torch.Tensor:
    """
    对输入张量进行per-token对称动态量化

    公式: scaleOut = row_max(abs(x)) / dtypeMax, yOut = round(x / scaleOut)

    Args:
        x: 输入张量
        axis: 计算scale和zero_point的维度，默认为最后一个维度
        dst_type: 目标数据类型

    Returns:
        量化后的张量
    """
    # FP16/BF16 输入需要升到 FP32 计算以保证精度
    # FP32/FP64 输入保持原样计算
    input_dtype = x.dtype
    if input_dtype in (torch.float16, torch.bfloat16):
        compute_dtype = torch.float32
    else:
        compute_dtype = input_dtype

    x_compute = x.to(compute_dtype)
    scale_out = torch.max(torch.abs(x_compute), dim=axis, keepdim=True)[0] / 127.0
    y = torch.round(x_compute / scale_out)
    return y
