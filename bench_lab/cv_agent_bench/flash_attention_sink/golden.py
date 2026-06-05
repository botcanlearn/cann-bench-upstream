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

class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        sink_k: torch.Tensor,
        sink_v: torch.Tensor,
    ):
        # sink_k/sink_v: [b, h, sink_size, d]
        scale = (1.0 / q.shape[-1]) ** 0.5
        sink_score = torch.einsum("bhsd,bhkd->bhsk", q, sink_k) * scale
        local_score = torch.einsum("bhsd,bhkd->bhsk", q, k) * scale
        # concat sink tokens before local tokens along key dimension
        acc = torch.cat([sink_score, local_score], dim=-1)
        acc = acc.softmax(dim=-1)
        sink_size = sink_k.shape[2]
        o = torch.einsum("bhsk,bhkd->bhsd", acc[..., :sink_size], sink_v) + \
            torch.einsum("bhsk,bhkd->bhsd", acc[..., sink_size:], v)
        return o


def flash_attention_sink(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, sink_k: torch.Tensor, sink_v: torch.Tensor) -> torch.Tensor:
    """Torch golden implementation aligned to the tile2asc reference Model.forward."""
    model = Model()
    return model(q, k, v, sink_k, sink_v)
