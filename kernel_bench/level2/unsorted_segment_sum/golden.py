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
UnsortedSegmentSum算子Torch Golden参考实现

沿segment_ids指定的段对数据进行求和
公式: y[i] = sum(data[j]) where segment_ids[j] == i
"""
def unsorted_segment_sum(
    data: torch.Tensor, segment_ids: torch.Tensor, num_segments: int
) -> torch.Tensor:
    """
    沿segment_ids指定的段对数据进行求和
    
    公式: y[i] = sum(data[j]) where segment_ids[j] == i
    
    Args:
        data: 输入数据张量
        segment_ids: 段ID张量
        num_segments: 段数量
    
    Returns:
        输出张量，段求和结果
    """

    y = torch.zeros(num_segments, *data.shape[1:], dtype=data.dtype, device=data.device)
    for i in range(num_segments):
        mask = (segment_ids == i)
        y[i] = data[mask].sum(dim=0)
    return y
