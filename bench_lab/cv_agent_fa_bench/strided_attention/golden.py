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

import torch.nn.functional as F

import math

class Model(nn.Module):
    def __init__(self, stride):
        super().__init__()
        self.stride = stride

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        batch_size, n_heads, seq_len, d_k = q.shape

        mask = torch.zeros(seq_len, seq_len, device=q.device)
        for i in range(seq_len):
            mask[i, i] = 1
            for j in range(i % self.stride, seq_len, self.stride):
                mask[i, j] = 1
        mask = mask.unsqueeze(0).unsqueeze(0)

        scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) / math.sqrt(d_k)
        scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, v.float())

        return output.to(torch.float16)


def strided_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, stride: int = 4) -> torch.Tensor:
    """Torch golden implementation aligned to the tile2asc reference Model.forward."""
    model = Model(stride)
    return model(q, k, v)
