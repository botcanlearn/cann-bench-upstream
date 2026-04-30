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
from typing import List, Optional

"""
ResizeBilinear 算子 Torch Golden 参考实现

使用双线性插值调整图像大小
公式：y = resize_bilinear(x, size)
"""
def resize_bilinear(
    x: torch.Tensor,
    output_size: Optional[List[int]] = None,
    align_corners: bool = False,
    scale_factor: Optional[List[float]] = None
) -> torch.Tensor:
    """
    使用双线性插值调整图像大小

    Args:
        x: 输入张量，形状为 (N, C, H, W)
        output_size: 输出尺寸 [output_height, output_width]
        align_corners: 是否对齐角点
        scale_factor: 缩放因子 [scale_height, scale_width]，与 output_size 互斥

    Returns:
        输出张量，调整大小后的结果
    """
    # 根据输入维度自动选择插值模式
    # 3D -> 1D linear, 4D -> 2D bilinear, 5D -> 3D trilinear
    dim = x.dim()
    if dim == 3:
        mode = 'linear'
    elif dim == 4:
        mode = 'bilinear'
    elif dim == 5:
        mode = 'trilinear'
    else:
        raise ValueError(f"ResizeBilinear requires 3D, 4D, or 5D input, got {dim}D")

    # NPU 对 fp16/bf16 使用 fp32 进行内部计算
    # Golden 也使用 fp32 计算以保持精度一致
    orig_dtype = x.dtype
    if orig_dtype in [torch.float16, torch.bfloat16]:
        x = x.float()

    # 使用 PyTorch 的 interpolate 实现
    y = torch.nn.functional.interpolate(
        x,
        size=output_size,
        scale_factor=scale_factor[0] if scale_factor and len(scale_factor) == 1 else scale_factor,
        mode=mode,
        align_corners=align_corners if dim >= 4 else None
    )

    # 转回原始 dtype
    return y.to(orig_dtype)
