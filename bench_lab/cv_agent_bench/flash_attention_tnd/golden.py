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

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                actual_q_len: torch.Tensor, actual_kv_len: torch.Tensor):
        """
        TND 格式的 flash attention。
        q/kv: (T, H, D) — 所有 batch 拼接后的总长度
        actual_q_len / actual_kv_len: (B,) — 每个 batch 的长度前缀和，最后一个元素即总长度
        """
        T_q, H, D = q.shape
        T_kv = k.shape[0]
        B = actual_q_len.shape[0]
        scale = (1.0 / D) ** 0.5

        o = torch.zeros_like(q)

        prev_q = 0
        prev_kv = 0
        for i in range(B):
            cur_q = actual_q_len[i].item()
            cur_kv = actual_kv_len[i].item()
            q_i = q[prev_q:cur_q]          # (Sq, H, D)
            k_i = k[prev_kv:cur_kv]        # (Sk, H, D)
            v_i = v[prev_kv:cur_kv]        # (Sk, H, D)

            # (H, Sq, Sk)
            acc = torch.einsum("shd,thd->hst", q_i, k_i) * scale
            acc = acc.softmax(dim=-1)
            # (H, Sq, D)
            out_i = torch.einsum("hst,thd->hsd", acc, v_i)
            # 写回 (Sq, H, D)
            o[prev_q:cur_q] = out_i.transpose(0, 1)

            prev_q = cur_q
            prev_kv = cur_kv

        return o


def flash_attention_tnd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, actual_q_len: torch.Tensor, actual_kv_len: torch.Tensor) -> torch.Tensor:
    """Torch golden implementation aligned to the tile2asc reference Model.forward."""
    model = Model()
    return model(q, k, v, actual_q_len, actual_kv_len)


def get_input(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
              actual_q_len: torch.Tensor, actual_kv_len: torch.Tensor,
              q_len_prefix=None, kv_len_prefix=None, **kwargs):
    q_prefix = q_len_prefix if q_len_prefix is not None else [q.shape[0]]
    kv_prefix = kv_len_prefix if kv_len_prefix is not None else [k.shape[0]]
    legal_q_len = torch.tensor(q_prefix, dtype=actual_q_len.dtype, device=actual_q_len.device)
    legal_kv_len = torch.tensor(kv_prefix, dtype=actual_kv_len.dtype, device=actual_kv_len.device)
    return q, k, v, legal_q_len, legal_kv_len
