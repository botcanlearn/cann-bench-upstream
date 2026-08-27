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
import torch.nn.functional as F


def fused_conv_sigmoid(
    x: torch.Tensor, filter: torch.Tensor, bias: torch.Tensor,
    strides: list, pads: list, dilations: list = None
) -> torch.Tensor:
    """Golden reference: conv2d + sigmoid fusion.

    Args:
        x: input tensor [N, C_in, H, W]
        filter: weight tensor [C_out, C_in, K_h, K_w]
        bias: bias tensor [C_out]
        strides: [stride_h, stride_w]
        pads: [pad_top, pad_bottom, pad_left, pad_right]
        dilations: [dilation_h, dilation_w]

    Returns:
        y: sigmoid(conv2d(x, filter, bias))
    """
    stride = tuple(strides)
    if dilations is None:
        dilations = [1, 1]
    dilation = tuple(dilations)

    if pads[0] == pads[1] and pads[2] == pads[3]:
        padding = (pads[0], pads[2])
    else:
        x = F.pad(x, (pads[2], pads[3], pads[0], pads[1]))
        padding = 0

    conv_out = F.conv2d(x, filter, bias, stride=stride, padding=padding, dilation=dilation)
    return torch.sigmoid(conv_out)
