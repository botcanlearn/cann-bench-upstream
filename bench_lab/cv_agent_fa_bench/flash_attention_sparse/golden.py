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

import math

class Model(nn.Module):
    """
    Sparse Flash Attention

    Input Q:       (batch, seqlen_q, nheads_q, headdim), bfloat16
    Input KV:      (batch, seqlen_kv, nheads_kv, headdim), bfloat16
    Input indices: (batch, seqlen_q, nheads_kv, topk), int32  — selected KV token positions
    Output:        (batch, seqlen_q, nheads_q, headdim), bfloat16

    nheads_q must be a multiple of nheads_kv
    """

    def __init__(self, nheads_q, nheads_kv, headdim, topk):
        super().__init__()
        assert nheads_q % nheads_kv == 0, "nheads_q must be a multiple of nheads_kv"
        self.nheads_q = nheads_q
        self.nheads_kv = nheads_kv
        self.headdim = headdim
        self.topk = topk
        self.scale = 1.0 / math.sqrt(headdim)
        self.groups = nheads_q // nheads_kv

    def forward(self, q, kv, indices):
        batch, seqlen_q, nheads_q, _ = q.shape
        _, seqlen_kv, nheads_kv, _ = kv.shape

        outputs = []
        for b in range(batch):
            out_seq = []
            for s in range(seqlen_q):
                out_heads = []
                for h in range(nheads_q):
                    kv_head = h // self.groups
                    # indices[b, s, kv_head, :]: (topk,) — token positions
                    token_pos = indices[b, s, kv_head, :].clamp(0, seqlen_kv - 1).long()

                    # Gather selected KV: (topk, headdim)
                    sel_kv = kv[b, :, kv_head].index_select(0, token_pos).float()

                    # Q: (1, headdim)
                    q_pos = q[b, s, h, :].unsqueeze(0).float()

                    # Scores: (1, topk)
                    scores = torch.mm(q_pos, sel_kv.t()) * self.scale
                    attn = torch.softmax(scores, dim=-1)

                    # Output: (headdim,)
                    out_h = torch.mm(attn, sel_kv).squeeze(0)
                    out_heads.append(out_h)

                out_seq.append(torch.stack(out_heads, dim=0))  # (nheads_q, headdim)

            out_b = torch.stack(out_seq, dim=0)  # (seqlen_q, nheads_q, headdim)
            outputs.append(out_b)

        return torch.stack(outputs, dim=0).to(q.dtype)


def flash_attention_sparse(q: torch.Tensor, kv: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Torch golden implementation aligned to the tile2asc reference Model.forward."""
    model = Model(q.shape[2], kv.shape[2], q.shape[-1], indices.shape[-1])
    return model(q, kv, indices)


def get_input(q: torch.Tensor, kv: torch.Tensor, indices: torch.Tensor, **kwargs):
    batch, seqlen_q = q.shape[:2]
    seqlen_kv, nheads_kv = kv.shape[1], kv.shape[2]
    topk = indices.shape[-1]
    base = torch.arange(topk, device=kv.device, dtype=torch.int64) % seqlen_kv
    q_offsets = torch.arange(seqlen_q, device=kv.device, dtype=torch.int64).view(1, seqlen_q, 1, 1)
    h_offsets = torch.arange(nheads_kv, device=kv.device, dtype=torch.int64).view(1, 1, nheads_kv, 1)
    legal_indices = (base.view(1, 1, 1, topk) + q_offsets + h_offsets) % seqlen_kv
    legal_indices = legal_indices.expand(batch, -1, -1, -1).contiguous()
    return q, kv, legal_indices.to(dtype=indices.dtype, device=indices.device)
