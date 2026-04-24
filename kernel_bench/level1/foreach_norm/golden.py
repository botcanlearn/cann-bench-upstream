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
from typing import List

"""
ForeachNorm 算子 Torch Golden 参考实现

对输入张量列表的每个张量进行范数运算
公式：y = (sum |x_i|^p)^(1/p)
"""
def foreach_norm(
    x: List[torch.Tensor], scalar: float
) -> List[torch.Tensor]:
    """
    对输入张量列表的每个张量进行范数运算

    公式：y = (sum |x_i|^p)^(1/p)

    Args:
        x: 输入张量列表 (TensorList)
        scalar: 范数阶数

    Returns:
        输出张量列表，每个张量的范数结果
    """

    y = [torch.norm(tensor, p=scalar) for tensor in x]
    return y
