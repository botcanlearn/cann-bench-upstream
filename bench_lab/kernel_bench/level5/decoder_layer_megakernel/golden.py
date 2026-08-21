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
DecoderLayerMegakernel 算子 Torch Golden 参考实现

单 token 解码整层融合（megakernel，Mirage MPK / Hazy Research megakernel 形态）：
标准 pre-norm decoder 层的一步解码（S_q = 1），每个 batch 样本 b 独立：
    1. h  = RMSNorm(x, γ1, ε)                                （沿 H，rms = sqrt(mean(x²) + ε)）
    2. q  = h @ wq → [B,1,Nq,D]；k_new = h @ wk、v_new = h @ wv → [B,1,Nkv,D]
    3. q、k_new 施加 RoPE（半维旋转）：x1,x2 = chunk(x,2,-1)；rot = cat(-x2,x1)；
       out = x·cos + rot·sin，cos/sin [B, D] 已按各样本当前位置索引好
    4. cache 追加：k_cache_out[b, :, cache_len[b], :] = k_new[b]（v 同）；
       其余槽位逐位保持输入值
    5. GQA attention（因果，有效长度 L_b = cache_len[b]+1）：query 头 n 的 KV 头
       g = n // (Nq/Nkv)，score = q·k / sqrt(D)，softmax 后加权 v，合并头 @ wo
    6. x2 = x + attn_proj（残差）
    7. h2 = RMSNorm(x2, γ2, ε)；mlp = (SiLU(h2@w_gate) ⊙ (h2@w_up)) @ w_down；y = x2 + mlp
输入约定：cache_len [B] int32 ∈ [1, Smax-1]（由 value_range 保证，写入槽位恒合法）。
plain golden 内部升 fp32 计算（bench 语义），输出转回输入 dtype；
decoder_layer_megakernel_oracle 跟随输入精度（golden_precision=fp64_cpu 下即为 fp64 真值）。
"""


def _decoder_layer_megakernel_core(x, gamma1, wq, wk, wv, wo, gamma2, w_gate, w_up, w_down,
                                   k_cache, v_cache, cache_len, rope_cos, rope_sin,
                                   epsilon, compute_dtype):
    """核心计算：以 compute_dtype 精度执行整层解码，返回 (y, k_cache_out, v_cache_out)。"""
    Bsz, _, H = x.shape
    Nkv, Smax, D = k_cache.shape[1], k_cache.shape[2], k_cache.shape[3]
    Nq = wq.shape[1] // D

    x_f = x.to(compute_dtype)
    g1_f = gamma1.to(compute_dtype)
    g2_f = gamma2.to(compute_dtype)
    wq_f = wq.to(compute_dtype)
    wk_f = wk.to(compute_dtype)
    wv_f = wv.to(compute_dtype)
    wo_f = wo.to(compute_dtype)
    wg_f = w_gate.to(compute_dtype)
    wu_f = w_up.to(compute_dtype)
    wd_f = w_down.to(compute_dtype)
    cos_f = rope_cos.to(compute_dtype)
    sin_f = rope_sin.to(compute_dtype)
    kc = k_cache.to(compute_dtype).clone()                          # [B, Nkv, Smax, D]
    vc = v_cache.to(compute_dtype).clone()

    # 1. RMSNorm（沿 H，eps 加在均方内）
    h = x_f / torch.sqrt((x_f * x_f).mean(dim=-1, keepdim=True) + epsilon) * g1_f

    # 2. QKV 投影
    q = torch.matmul(h, wq_f).reshape(Bsz, 1, Nq, D)                # [B, 1, Nq, D]
    k_new = torch.matmul(h, wk_f).reshape(Bsz, 1, Nkv, D)           # [B, 1, Nkv, D]
    v_new = torch.matmul(h, wv_f).reshape(Bsz, 1, Nkv, D)           # [B, 1, Nkv, D]

    # 3. q / k_new 施加 RoPE（半维旋转，cos/sin 已按各样本当前位置索引好）
    def _rope(t):
        t1, t2 = t.chunk(2, dim=-1)
        rot = torch.cat([-t2, t1], dim=-1)
        return t * cos_f[:, None, None, :] + rot * sin_f[:, None, None, :]

    q = _rope(q)
    k_new = _rope(k_new)

    # 4 + 5. cache 追加 + 变长 GQA attention（逐 b：各样本有效长度不同）
    grp = Nq // Nkv
    scale = 1.0 / float(D) ** 0.5
    attn = torch.empty(Bsz, 1, Nq * D, dtype=compute_dtype, device=x.device)
    for b in range(Bsz):
        pos = int(cache_len[b])
        kc[b, :, pos, :] = k_new[b, 0]                              # 写入槽位 cache_len[b]
        vc[b, :, pos, :] = v_new[b, 0]
        lb = pos + 1                                                # 有效长度 L_b
        k_act = kc[b, :, :lb, :]                                    # [Nkv, L_b, D]
        v_act = vc[b, :, :lb, :]
        q_b = q[b, 0].reshape(Nkv, grp, D)                          # 头 n 的 KV 头 g = n // grp
        scores = torch.matmul(q_b, k_act.transpose(-1, -2)) * scale  # [Nkv, grp, L_b]
        probs = torch.softmax(scores, dim=-1)
        attn[b, 0] = torch.matmul(probs, v_act).reshape(Nq * D)     # [Nkv, grp, D] → [Nq*D]
    attn_proj = torch.matmul(attn, wo_f)                            # [B, 1, H]

    # 6. 残差
    x2 = x_f + attn_proj

    # 7. RMSNorm + SwiGLU MLP + 残差
    h2 = x2 / torch.sqrt((x2 * x2).mean(dim=-1, keepdim=True) + epsilon) * g2_f
    gate = torch.matmul(h2, wg_f)
    mlp = torch.matmul(gate * torch.sigmoid(gate) * torch.matmul(h2, wu_f), wd_f)
    y = x2 + mlp
    return y, kc, vc


def decoder_layer_megakernel(
    x: torch.Tensor,
    gamma1: torch.Tensor,
    wq: torch.Tensor,
    wk: torch.Tensor,
    wv: torch.Tensor,
    wo: torch.Tensor,
    gamma2: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cache_len: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    epsilon: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    DecoderLayerMegakernel golden reference（plain golden = bench：内部 fp32 计算）

    Args:
        x: [B, 1, H] 本步输入 hidden states
        gamma1: [H] attention 前 RMSNorm 的 γ
        wq: [H, Nq*D] query 投影权重
        wk: [H, Nkv*D] key 投影权重
        wv: [H, Nkv*D] value 投影权重
        wo: [Nq*D, H] attention 输出投影权重
        gamma2: [H] MLP 前 RMSNorm 的 γ
        w_gate: [H, F] SwiGLU gate 投影权重
        w_up: [H, F] SwiGLU up 投影权重
        w_down: [F, H] SwiGLU down 投影权重
        k_cache: [B, Nkv, Smax, D] key cache（本步写入槽位 cache_len[b]，其余槽位逐位保持）
        v_cache: [B, Nkv, Smax, D] value cache（同上）
        cache_len: [B] int32，各样本已有的有效 cache 长度 ∈ [1, Smax-1]（由 value_range 保证）
        rope_cos: [B, D] RoPE 余弦（已按各样本当前位置索引好）
        rope_sin: [B, D] RoPE 正弦（已按各样本当前位置索引好）
        epsilon: 两处 RMSNorm 的 epsilon，默认 1e-6

    Returns:
        y: [B, 1, H] 本层输出 hidden states，dtype 与 x 一致
        k_cache_out: [B, Nkv, Smax, D] 追加后的 key cache，shape/dtype 与 k_cache 一致
        v_cache_out: [B, Nkv, Smax, D] 追加后的 value cache，shape/dtype 与 v_cache 一致
    """
    y, kc, vc = _decoder_layer_megakernel_core(
        x, gamma1, wq, wk, wv, wo, gamma2, w_gate, w_up, w_down,
        k_cache, v_cache, cache_len, rope_cos, rope_sin, epsilon, torch.float32)
    return y.to(x.dtype), kc.to(k_cache.dtype), vc.to(v_cache.dtype)


def decoder_layer_megakernel_oracle(
    x: torch.Tensor,
    gamma1: torch.Tensor,
    wq: torch.Tensor,
    wk: torch.Tensor,
    wv: torch.Tensor,
    wo: torch.Tensor,
    gamma2: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cache_len: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    epsilon: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Oracle (g)：dtype-agnostic，计算精度跟随输入（fp64_cpu 下为 fp64 真值），不硬编码 .float()。"""
    return _decoder_layer_megakernel_core(
        x, gamma1, wq, wk, wv, wo, gamma2, w_gate, w_up, w_down,
        k_cache, v_cache, cache_len, rope_cos, rope_sin, epsilon, x.dtype)
