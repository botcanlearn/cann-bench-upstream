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

import torch.nn as nn

class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        scale = torch.tensor(1.0 / q.shape[-1], dtype=torch.float16, device=q.device).sqrt()

        # QK^T / sqrt(d)，得到 (batch, heads, seq, seq) 注意力分数矩阵
        acc = torch.einsum("bhsd,bhkd->bhsk", q, k) * scale

        s = q.shape[2]
        causal_mask = torch.tril(torch.ones(s, s, dtype=torch.bool, device=q.device))

        # 将上三角位置（不可见位置）填充为 -inf，softmax 后该位置概率为 0
        acc = acc.masked_fill(~causal_mask, float("-inf"))
        acc = acc.softmax(dim=-1)

        # 注意力加权求和：acc @ V
        o = torch.einsum("bhsk,bhkd->bhsd", acc, v)
        return o


def flash_attention_causal(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Torch golden implementation aligned to the tile2asc reference Model.forward."""
    model = Model()
    return model(q, k, v)
