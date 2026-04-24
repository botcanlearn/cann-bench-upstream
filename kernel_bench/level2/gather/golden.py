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
Gather算子Torch Golden参考实现

从输入Tensor的指定维度按index提取元素
公式: y[i][m][n] = x[index[i]][m][n]
"""
def gather(
    x: torch.Tensor, index: torch.Tensor, batch_dims: int = 0
) -> torch.Tensor:
    """
    从输入Tensor的指定维度按index提取元素
    
    公式: y[i][m][n] = x[index[i]][m][n]
    
    Args:
        x: 输入张量
        index: 索引张量
        batch_dims: batch维度数
    
    Returns:
        输出张量，gather结果
    """

    y = torch.gather(x, batch_dims, index.long())
    return y
