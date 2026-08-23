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
CompressedWeightGemv 算子 Torch Golden 参考实现

自定义压缩位流（2:4 结构化稀疏 + int4 分组量化）的在线解码融合 GEMV：
逻辑权重 W [N, K] 中每连续 4 个 K 位置恰有 2 个非零。压缩格式（格式即规格，见 desc.md §2）：
- packed_w [N, K/4] int32，仅低 8 位有效：低 nibble（bit 0-3）= 块内第一个非零的 int4 码，
  高 nibble（bit 4-7）= 第二个非零的 int4 码；int4 码 c ∈ [0, 15] 解码为 c - 8 ∈ [-8, 7]
  （偏移二进制码表，注意码 0 → 值 -8，全零 packed_w 不等于全零权重）
- sparse_sel [N, K/4] int32 ∈ [0, 5]，编码块内 4 选 2 的非零位置组合（升序对）：
  0→(0,1) 1→(0,2) 2→(0,3) 3→(1,2) 4→(1,3) 5→(2,3)
- scales [N, K/group_size]，分组量化 scale；块 j 的第 pos 个位置（绝对列 4j+pos）的
  scale 组索引 = (4j+pos) // group_size（group_size 为 4 的倍数，故一个 4 块不跨组）

重建：W[n, 4j+pos_a] = (低 nibble - 8) * scales[n, (4j+pos_a)//group_size]，
      W[n, 4j+pos_b] = (高 nibble - 8) * scales[n, (4j+pos_b)//group_size]，其余位置为 0。
输出：y = x @ W^T + bias，y [B, N]，dtype 与 x 一致。

plain golden 内部升 fp32 计算（bench 语义：解码、matmul 累加均 fp32），输出转回 x 的 dtype；
compressed_weight_gemv_oracle 跟随输入精度（golden_precision=fp64_cpu 下即为 fp64 真值）。
"""

# 4 选 2 位置组合表：sel -> (pos_a, pos_b)，升序对
_POS_A = (0, 0, 0, 1, 1, 2)
_POS_B = (1, 2, 3, 2, 3, 3)


def _compressed_weight_gemv_core(x, packed_w, sparse_sel, scales, bias, group_size, compute_dtype):
    """核心计算：整数解码（查表 + 移位）重建 W，再以 compute_dtype 做 GEMV。"""
    N, KQ = packed_w.shape          # KQ = K/4
    K = KQ * 4
    device = x.device

    x_f = x.to(compute_dtype)                      # [B, K]
    scales_f = scales.to(compute_dtype)            # [N, K/group_size]
    bias_f = bias.to(compute_dtype)                # [N]

    packed = packed_w.long()                       # 仅低 8 位有效
    lo = (packed & 0xF) - 8                        # [N, K/4] 第一个非零的 int4 值 ∈ [-8, 7]
    hi = ((packed >> 4) & 0xF) - 8                 # [N, K/4] 第二个非零的 int4 值 ∈ [-8, 7]

    sel = sparse_sel.long()                        # [N, K/4] ∈ [0, 5]
    pos_a = torch.tensor(_POS_A, dtype=torch.long, device=device)[sel]   # [N, K/4]
    pos_b = torch.tensor(_POS_B, dtype=torch.long, device=device)[sel]   # [N, K/4]
    base = torch.arange(KQ, device=device, dtype=torch.long).unsqueeze(0) * 4   # [1, K/4]
    col_a = base + pos_a                           # [N, K/4] 绝对列 4j+pos_a
    col_b = base + pos_b                           # [N, K/4] 绝对列 4j+pos_b

    # scale 组索引 = 绝对列 // group_size（4 | group_size，两个非零同组，仍按定义逐列取）
    sc_a = torch.gather(scales_f, 1, col_a // group_size)   # [N, K/4]
    sc_b = torch.gather(scales_f, 1, col_b // group_size)   # [N, K/4]

    w = torch.zeros(N, K, dtype=compute_dtype, device=device)
    w.scatter_(1, col_a, lo.to(compute_dtype) * sc_a)
    w.scatter_(1, col_b, hi.to(compute_dtype) * sc_b)

    y = x_f @ w.t() + bias_f                       # [B, N]
    return y


def compressed_weight_gemv(
    x: torch.Tensor,
    packed_w: torch.Tensor,
    sparse_sel: torch.Tensor,
    scales: torch.Tensor,
    bias: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """
    压缩权重在线解码融合 GEMV golden reference（plain golden = bench：内部 fp32 计算）

    Args:
        x: [B, K] 激活，bfloat16/float16/float32，K % 4 == 0 且 K % group_size == 0
        packed_w: [N, K/4] int32，取值 [0, 255]：低/高 nibble = 块内两个非零的 int4 码（码 c → 值 c-8）
        sparse_sel: [N, K/4] int32，取值 [0, 5]：4 选 2 非零位置组合
            （0→(0,1) 1→(0,2) 2→(0,3) 3→(1,2) 4→(1,3) 5→(2,3)）
        scales: [N, K/group_size] float16/float32，分组量化 scale
        bias: [N]，dtype 与 x 一致
        group_size: 分组大小（4 的正倍数且整除 K，评测取 {32, 64, 128}）

    Returns:
        y: [B, N] = x @ W^T + bias，dtype 与 x 一致
    """
    y = _compressed_weight_gemv_core(
        x, packed_w, sparse_sel, scales, bias, group_size, torch.float32)
    return y.to(x.dtype)


def compressed_weight_gemv_oracle(
    x: torch.Tensor,
    packed_w: torch.Tensor,
    sparse_sel: torch.Tensor,
    scales: torch.Tensor,
    bias: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """Oracle (g)：dtype-agnostic，计算精度跟随输入（fp64_cpu 下为 fp64 真值），不硬编码 .float()。"""
    return _compressed_weight_gemv_core(
        x, packed_w, sparse_sel, scales, bias, group_size, x.dtype)
