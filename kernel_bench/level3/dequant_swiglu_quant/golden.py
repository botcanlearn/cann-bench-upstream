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
DequantSwigluQuant算子Torch Golden参考实现

反量化、SwiGLU和量化的融合
公式: y = quantize(SwiGLU(dequantize(x)))
"""
def dequant_swiglu_quant(
    x: torch.Tensor, activate_left: bool = False, quant_mode: str = 'static', dst_type: int = 0
) -> torch.Tensor:
    """
    反量化、SwiGLU和量化的融合
    
    公式: y = quantize(SwiGLU(dequantize(x)))
    
    Args:
        x: 输入张量
        activate_left: 是否激活左侧
        quant_mode: 量化模式'
        dst_type: 目标数据类型 (0:DT_INT8)
    
    Returns:
        输出张量
    """

    def swiglu(x, activate_left=False):
        # 截断到偶数维以支持非对齐输入
        last_dim = x.shape[-1] - (x.shape[-1] % 2)
        x = x[..., :last_dim]
        half = x.shape[-1] // 2
        if activate_left:
            x_left = x[..., :half]
            x_right = x[..., half:]
            return x_left * torch.nn.functional.silu(x_right)
        else:
            x_left = x[..., :half]
            x_right = x[..., half:]
            return torch.nn.functional.silu(x_left) * x_right
    
    if x.dtype == torch.int32:
        scale = 0.1
        x_float = x.float() * scale
    else:
        x_float = x
    
    result = swiglu(x_float, activate_left)
    
    # INT8 量化
    max_val = result.abs().max()
    if max_val == 0:
        # 处理全零情况，避免 scale=inf
        y = torch.zeros_like(result, dtype=torch.int8)
    else:
        scale = (127.0 / max_val).to(torch.float32)
        y = torch.clamp((result.float() * scale.item()).round(), -128, 127).to(torch.int8)

    return y
