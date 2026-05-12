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
DepthwiseConv2D算子Torch Golden参考实现

二维深度卷积运算
公式: y = bias + weight * x
"""
def depthwise_conv_2d(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, kernelSize: list, stride: list, padding: list, dilation: list, groups: int
) -> torch.Tensor:
    """
    二维深度卷积运算
    
    公式: y = bias + weight * x
    
    Args:
        x: 输入特征图
        weight: 卷积核
        bias: 偏置
        kernelSize: 卷积核大小
        stride: 步长
        padding: 填充
        dilation: 膨胀率
        groups: 分组数
    
    Returns:
        输出特征图
    """

    stride_val = (stride[0], stride[1])
    padding_val = (padding[0], padding[1])
    dilation_val = (dilation[0], dilation[1])
    
    y = torch.nn.functional.conv2d(x, weight, bias, stride=stride_val, padding=padding_val, dilation=dilation_val, groups=groups)
    return y
