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
from typing import Tuple

"""
MobaAttention 算子 Torch Golden 参考实现

MoBA（Mixture of Block Attention）块稀疏注意力：把 KV 序列按 block_size 切块，
每个 query 用"query 与块内 key 均值的点积"做门控，只对选中的 top_k 个块计算注意力。

公式（每个 (b, n) 独立；nb = ceil(S / block_size)，cur = floor(t / block_size)）:
    k̄_j = mean(key[块 j])                                  # 块代表键，末块按实际长度取均值
    s_{t,j} = q_t · k̄_j          (j < cur，不乘 scaleValue)  # 门控分数
    s_{t,cur} = +inf（当前块必选，计入 top_k 名额）; s_{t,j} = -inf (j > cur，不可选)
    B_t = top_k(s_{t,·})          # 并列时取块索引更小者（稳定排序保证）
    P_t = {p ∈ ∪_{j∈B_t} 块 j 且 p ≤ t}                     # 当前块内因果，过去块全可见
    y_t = softmax_{p∈P_t}(scaleValue · q_t · k_p) v_p
    block_indices[t] = sorted(B_t) 升序，不足 top_k 用 -1 填充

plain golden 内部升 fp32 计算（bench 语义），y 转回输入 dtype；
moba_attention_oracle 跟随输入精度（golden_precision=fp64_cpu 下即为 fp64 真值）。
block_indices 为 int32 结构输出，plain 与 oracle 的选择语义一致（同一稳定排序规则）。
"""


def _moba_attention_core(query, key, value, block_size, top_k, scaleValue, compute_dtype):
    """核心计算：以 compute_dtype 精度做块门控 top-k 选择 + 块稀疏注意力。

    逐 (b, n) 循环并对 query 分块，控制 [Q, S] 中间量的峰值内存。
    """
    B, S, N, D = query.shape
    device = query.device
    nb = (S + block_size - 1) // block_size

    q_all = query.permute(0, 2, 1, 3).to(compute_dtype)   # [B, N, S, D]
    k_all = key.permute(0, 2, 1, 3).to(compute_dtype)     # [B, N, S, D]
    v_all = value.permute(0, 2, 1, 3).to(compute_dtype)   # [B, N, S, D]

    pos = torch.arange(S, device=device)                  # [S]
    blk_of_pos = pos // block_size                        # [S] 每个位置所属块
    blk_ids = torch.arange(nb, device=device)             # [nb]
    # 每块实际长度（末块可残缺）: len_j = min((j+1)·bs, S) - j·bs
    blk_len = torch.clamp((blk_ids + 1) * block_size, max=S) - blk_ids * block_size

    y = torch.empty(B, S, N, D, dtype=compute_dtype, device=device)
    block_indices = torch.empty(B, S, N, top_k, dtype=torch.int32, device=device)

    q_chunk = 2048                                        # query 分块大小（仅控内存，不影响结果）
    arange_k = torch.arange(top_k, device=device)         # [top_k]
    k_take = min(top_k, nb)

    for b in range(B):
        for n in range(N):
            q = q_all[b, n]                               # [S, D]
            k = k_all[b, n]                               # [S, D]
            v = v_all[b, n]                               # [S, D]

            # 块代表键 k̄_j = 块内 key 均值（零填充到 nb·bs 后分块求和，除以实际长度）
            k_pad = torch.zeros(nb * block_size, D, dtype=compute_dtype, device=device)
            k_pad[:S] = k
            k_mean = k_pad.reshape(nb, block_size, D).sum(dim=1) / blk_len.unsqueeze(-1)  # [nb, D]

            for qs in range(0, S, q_chunk):
                qe = min(qs + q_chunk, S)
                qt = q[qs:qe]                             # [Q, D]
                t_pos = pos[qs:qe]                        # [Q]
                cur = blk_of_pos[qs:qe]                   # [Q] 当前块索引

                # === 门控分数（不乘 scaleValue）===
                gate = qt @ k_mean.T                      # [Q, nb]
                gate = gate.masked_fill(blk_ids.unsqueeze(0) > cur.unsqueeze(-1), float('-inf'))
                gate = gate.masked_fill(blk_ids.unsqueeze(0) == cur.unsqueeze(-1), float('inf'))

                # === top_k 选择：降序稳定排序（基序为块索引升序 → 并列取小索引）===
                order = torch.sort(gate, dim=-1, descending=True, stable=True).indices  # [Q, nb]
                sel = order[:, :k_take]                   # [Q, k_take]
                if k_take < top_k:                        # top_k > nb: 全选后补哨兵
                    pad = torch.full((qe - qs, top_k - k_take), nb, dtype=sel.dtype, device=device)
                    sel = torch.cat([sel, pad], dim=-1)   # [Q, top_k]
                # 可选块数 = cur + 1（当前块 + 过去块），超出部分置哨兵 nb
                n_valid = torch.clamp(cur + 1, max=top_k)                     # [Q]
                sel = sel.masked_fill(arange_k.unsqueeze(0) >= n_valid.unsqueeze(-1), nb)

                # === block_indices 输出：升序排列，哨兵（排在尾部）替换为 -1 ===
                sel_sorted = sel.sort(dim=-1).values                          # [Q, top_k]
                block_indices[b, qs:qe, n] = torch.where(
                    sel_sorted == nb, torch.full_like(sel_sorted, -1), sel_sorted
                ).to(torch.int32)

                # === 块级选中掩码 → 位置级掩码 ===
                sel_mask = torch.zeros(qe - qs, nb + 1, dtype=torch.bool, device=device)
                sel_mask.scatter_(1, sel.long(), True)
                sel_mask = sel_mask[:, :nb]                                   # [Q, nb] 哨兵列丢弃
                pos_mask = sel_mask[:, blk_of_pos]                            # [Q, S] 位置所属块被选中
                pos_mask &= pos.unsqueeze(0) <= t_pos.unsqueeze(-1)           # 当前块内因果（p ≤ t）

                # === 块稀疏注意力（p = t 恒可见，softmax 每行至少一个有效位置）===
                scores = (qt @ k.T) * scaleValue                              # [Q, S]
                scores.masked_fill_(~pos_mask, float('-inf'))
                y[b, qs:qe, n] = torch.softmax(scores, dim=-1) @ v

    return y, block_indices


def moba_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    block_size: int,
    top_k: int,
    scaleValue: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    MoBA 块稀疏注意力 golden reference（plain golden = bench：内部 fp32 计算）

    Args:
        query: [B, S, N, D] 查询张量（BSND，MHA，N 头共享同一 top-k 语义、各头独立选择）
        key: [B, S, N, D] 键张量
        value: [B, S, N, D] 值张量
        block_size: KV 块大小（正整数，S 无需整除，末块按实际长度处理）
        top_k: 每个 query 选择的块数（≥ 1，当前块必选并占一个名额）
        scaleValue: 注意力缩放因子（仅作用于注意力分数，门控分数不乘）

    Returns:
        y: [B, S, N, D] 注意力输出，dtype 与 query 一致
        block_indices: [B, S, N, top_k] int32，选中块索引升序，不足 top_k 用 -1 填充
    """
    original_dtype = query.dtype
    y, block_indices = _moba_attention_core(
        query, key, value, block_size, top_k, scaleValue, torch.float32)
    return y.to(original_dtype), block_indices


def moba_attention_oracle(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    block_size: int,
    top_k: int,
    scaleValue: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Oracle (g)：dtype-agnostic，计算精度跟随输入（fp64_cpu 下为 fp64 真值），不硬编码 .float()。"""
    return _moba_attention_core(query, key, value, block_size, top_k, scaleValue, query.dtype)
