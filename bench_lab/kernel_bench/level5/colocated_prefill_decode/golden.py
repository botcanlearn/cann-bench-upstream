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
ColocatedPrefillDecode 算子 Torch Golden 参考实现

双模型同 die 共驻（colocated serving）的一次融合调用，同时执行两个相互独立的负载：
负载 A 为 prefill 模型的一个完整 pre-norm decoder 层（因果自注意力，GEMM 密集），
负载 B 为 decode 模型的单 token 解码层（KV cache 追加 + 变长 GQA attention，
带宽/向量密集，语义与 DecoderLayerMegakernel 一致）。

独立性契约：y_a 仅由 A 侧输入（x_a、*_a 权重、rope_cos_a/rope_sin_a、epsilon）决定；
y_b / k_cache_out / v_cache_out 仅由 B 侧输入（x_b、*_b 权重、k_cache/v_cache/cache_len、
rope_cos_b/rope_sin_b、epsilon）决定。共驻不得引入任何串扰。

负载 A（x_a [B_a, S_a, H_a]，因果自注意力，每个 b 独立）：
    1. h = RMSNorm(x_a, γ1_a, ε)                    （沿 H_a，rms = sqrt(mean(x²) + ε)）
    2. q = h @ wq_a → [B_a, S_a, Nq_a, D_a]；k = h @ wk_a、v = h @ wv_a → [B_a, S_a, Nkv_a, D_a]
    3. q、k 施加 RoPE（半维旋转）：x1,x2 = chunk(x,2,-1)；rot = cat(-x2,x1)；
       out = x·cos + rot·sin，cos/sin [B_a, S_a, D_a] 已逐位置索引好
    4. 因果 GQA attention：query 头 n 的 KV 头 g = n // (Nq_a/Nkv_a)，
       score = q·k / sqrt(D_a)，位置 i 只看 j ≤ i，softmax 后加权 v
    5. 合并头 @ wo_a；x2 = x_a + attn_proj（残差）
    6. h2 = RMSNorm(x2, γ2_a, ε)；mlp = (SiLU(h2@w_gate_a) ⊙ (h2@w_up_a)) @ w_down_a；
       y_a = x2 + mlp
负载 B（x_b [B_b, 1, H_b]，单 token + KV cache 追加，每个 b 独立）：
    1. h = RMSNorm(x_b, γ1_b, ε)；q/k_new/v_new 投影；q、k_new 施加 RoPE
       （cos/sin [B_b, D_b] 已按各样本当前位置索引好）
    2. cache 追加：k_cache_out[b, :, cache_len[b], :] = k_new[b]（v 同）；
       其余槽位逐位保持输入值
    3. 变长 GQA attention（有效长度 L_b = cache_len[b]+1）→ wo_b → 残差
    4. RMSNorm(γ2_b) → SwiGLU → 残差 → y_b
输入约定：cache_len [B_b] int32 ∈ [1, Smax-1]（由 value_range 保证，写入槽位恒合法）；
D_a、D_b 为偶数（RoPE 半维旋转）。
plain golden 内部升 fp32 计算（bench 语义），输出转回输入 dtype；
colocated_prefill_decode_oracle 跟随输入精度（golden_precision=fp64_cpu 下即为 fp64 真值）。
"""


def _colocated_prefill_decode_core(x_a, gamma1_a, wq_a, wk_a, wv_a, wo_a, gamma2_a,
                                   w_gate_a, w_up_a, w_down_a,
                                   x_b, gamma1_b, wq_b, wk_b, wv_b, wo_b, gamma2_b,
                                   w_gate_b, w_up_b, w_down_b,
                                   k_cache, v_cache, cache_len,
                                   rope_cos_a, rope_sin_a, rope_cos_b, rope_sin_b,
                                   epsilon, compute_dtype):
    """核心计算：以 compute_dtype 精度独立执行负载 A 与负载 B，返回 (y_a, y_b, k_cache_out, v_cache_out)。"""

    def _rms_norm(t, gamma):
        return t / torch.sqrt((t * t).mean(dim=-1, keepdim=True) + epsilon) * gamma

    def _rope(t, cos, sin):
        t1, t2 = t.chunk(2, dim=-1)
        rot = torch.cat([-t2, t1], dim=-1)
        return t * cos + rot * sin

    # ===== 负载 A：prefill 层（因果自注意力，GQA，仅消费 A 侧输入）=====
    b_a, s_a, _ = x_a.shape
    d_a = rope_cos_a.shape[-1]
    nq_a = wq_a.shape[1] // d_a
    nkv_a = wk_a.shape[1] // d_a

    xa = x_a.to(compute_dtype)
    g1a = gamma1_a.to(compute_dtype)
    g2a = gamma2_a.to(compute_dtype)
    cos_a = rope_cos_a.to(compute_dtype)[:, :, None, :]              # [B_a, S_a, 1, D_a]
    sin_a = rope_sin_a.to(compute_dtype)[:, :, None, :]

    # 1. RMSNorm（沿 H_a，eps 加在均方内）
    h = _rms_norm(xa, g1a)

    # 2 + 3. QKV 投影 + q/k 施加 RoPE（逐位置 cos/sin，广播到全部头）
    q = _rope(torch.matmul(h, wq_a.to(compute_dtype)).reshape(b_a, s_a, nq_a, d_a), cos_a, sin_a)
    k = _rope(torch.matmul(h, wk_a.to(compute_dtype)).reshape(b_a, s_a, nkv_a, d_a), cos_a, sin_a)
    v = torch.matmul(h, wv_a.to(compute_dtype)).reshape(b_a, s_a, nkv_a, d_a)

    # 4. 因果 GQA attention（逐 b：控制峰值内存，样本间独立）
    grp_a = nq_a // nkv_a
    scale_a = 1.0 / float(d_a) ** 0.5
    causal = torch.triu(torch.ones(s_a, s_a, dtype=torch.bool, device=x_a.device), diagonal=1)
    attn_a = torch.empty(b_a, s_a, nq_a * d_a, dtype=compute_dtype, device=x_a.device)
    for b in range(b_a):
        qh = q[b].reshape(s_a, nkv_a, grp_a, d_a).permute(1, 2, 0, 3)     # 头 n 的 KV 头 g = n // grp
        kh = k[b].permute(1, 0, 2)                                        # [Nkv_a, S_a, D_a]
        vh = v[b].permute(1, 0, 2)
        scores = torch.matmul(qh, kh.unsqueeze(1).transpose(-1, -2)) * scale_a   # [Nkv_a, grp, S_a, S_a]
        scores = scores.masked_fill(causal, float('-inf'))                # 位置 i 只看 j ≤ i
        probs = torch.softmax(scores, dim=-1)
        ctx = torch.matmul(probs, vh.unsqueeze(1))                        # [Nkv_a, grp, S_a, D_a]
        attn_a[b] = ctx.permute(2, 0, 1, 3).reshape(s_a, nq_a * d_a)

    # 5. 输出投影 + 残差
    x2 = xa + torch.matmul(attn_a, wo_a.to(compute_dtype))

    # 6. RMSNorm + SwiGLU MLP + 残差
    h2 = _rms_norm(x2, g2a)
    gate = torch.matmul(h2, w_gate_a.to(compute_dtype))
    y_a = x2 + torch.matmul(gate * torch.sigmoid(gate) * torch.matmul(h2, w_up_a.to(compute_dtype)),
                            w_down_a.to(compute_dtype))

    # ===== 负载 B：单 token decode 层（KV cache 追加 + 变长 GQA，仅消费 B 侧输入）=====
    b_b = x_b.shape[0]
    nkv_b, _, d_b = k_cache.shape[1], k_cache.shape[2], k_cache.shape[3]
    nq_b = wq_b.shape[1] // d_b

    xb = x_b.to(compute_dtype)
    g1b = gamma1_b.to(compute_dtype)
    g2b = gamma2_b.to(compute_dtype)
    cos_b = rope_cos_b.to(compute_dtype)[:, None, None, :]               # [B_b, 1, 1, D_b]
    sin_b = rope_sin_b.to(compute_dtype)[:, None, None, :]
    kc = k_cache.to(compute_dtype).clone()                               # [B_b, Nkv_b, Smax, D_b]
    vc = v_cache.to(compute_dtype).clone()

    # 1. RMSNorm + QKV 投影 + q/k_new 施加 RoPE
    hb = _rms_norm(xb, g1b)
    qb = _rope(torch.matmul(hb, wq_b.to(compute_dtype)).reshape(b_b, 1, nq_b, d_b), cos_b, sin_b)
    k_new = _rope(torch.matmul(hb, wk_b.to(compute_dtype)).reshape(b_b, 1, nkv_b, d_b), cos_b, sin_b)
    v_new = torch.matmul(hb, wv_b.to(compute_dtype)).reshape(b_b, 1, nkv_b, d_b)

    # 2 + 3. cache 追加 + 变长 GQA attention（逐 b：各样本有效长度不同）
    grp_b = nq_b // nkv_b
    scale_b = 1.0 / float(d_b) ** 0.5
    attn_b = torch.empty(b_b, 1, nq_b * d_b, dtype=compute_dtype, device=x_b.device)
    for b in range(b_b):
        pos = int(cache_len[b])
        kc[b, :, pos, :] = k_new[b, 0]                                   # 写入槽位 cache_len[b]
        vc[b, :, pos, :] = v_new[b, 0]
        lb = pos + 1                                                     # 有效长度 L_b
        k_act = kc[b, :, :lb, :]                                         # [Nkv_b, L_b, D_b]
        v_act = vc[b, :, :lb, :]
        q_b = qb[b, 0].reshape(nkv_b, grp_b, d_b)                        # 头 n 的 KV 头 g = n // grp
        scores = torch.matmul(q_b, k_act.transpose(-1, -2)) * scale_b    # [Nkv_b, grp, L_b]
        probs = torch.softmax(scores, dim=-1)
        attn_b[b, 0] = torch.matmul(probs, v_act).reshape(nq_b * d_b)
    x2b = xb + torch.matmul(attn_b, wo_b.to(compute_dtype))

    # 4. RMSNorm + SwiGLU MLP + 残差
    h2b = _rms_norm(x2b, g2b)
    gate_b = torch.matmul(h2b, w_gate_b.to(compute_dtype))
    y_b = x2b + torch.matmul(gate_b * torch.sigmoid(gate_b) * torch.matmul(h2b, w_up_b.to(compute_dtype)),
                             w_down_b.to(compute_dtype))
    return y_a, y_b, kc, vc


def colocated_prefill_decode(
    x_a: torch.Tensor,
    gamma1_a: torch.Tensor,
    wq_a: torch.Tensor,
    wk_a: torch.Tensor,
    wv_a: torch.Tensor,
    wo_a: torch.Tensor,
    gamma2_a: torch.Tensor,
    w_gate_a: torch.Tensor,
    w_up_a: torch.Tensor,
    w_down_a: torch.Tensor,
    x_b: torch.Tensor,
    gamma1_b: torch.Tensor,
    wq_b: torch.Tensor,
    wk_b: torch.Tensor,
    wv_b: torch.Tensor,
    wo_b: torch.Tensor,
    gamma2_b: torch.Tensor,
    w_gate_b: torch.Tensor,
    w_up_b: torch.Tensor,
    w_down_b: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cache_len: torch.Tensor,
    rope_cos_a: torch.Tensor,
    rope_sin_a: torch.Tensor,
    rope_cos_b: torch.Tensor,
    rope_sin_b: torch.Tensor,
    epsilon: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    ColocatedPrefillDecode golden reference（plain golden = bench：内部 fp32 计算）

    Args:
        x_a: [B_a, S_a, H_a] 负载 A（prefill 层）输入 hidden states
        gamma1_a: [H_a] A 侧 attention 前 RMSNorm 的 γ
        wq_a: [H_a, Nq_a*D_a] A 侧 query 投影权重（列按头拼接，头 n 占列 n*D_a..(n+1)*D_a-1）
        wk_a: [H_a, Nkv_a*D_a] A 侧 key 投影权重（列按 KV 头拼接，Nq_a % Nkv_a == 0）
        wv_a: [H_a, Nkv_a*D_a] A 侧 value 投影权重
        wo_a: [Nq_a*D_a, H_a] A 侧 attention 输出投影权重
        gamma2_a: [H_a] A 侧 MLP 前 RMSNorm 的 γ
        w_gate_a: [H_a, F_a] A 侧 SwiGLU gate 投影权重
        w_up_a: [H_a, F_a] A 侧 SwiGLU up 投影权重
        w_down_a: [F_a, H_a] A 侧 SwiGLU down 投影权重
        x_b: [B_b, 1, H_b] 负载 B（decode 层）输入 hidden states（单 token，S_q = 1）
        gamma1_b: [H_b] B 侧 attention 前 RMSNorm 的 γ
        wq_b: [H_b, Nq_b*D_b] B 侧 query 投影权重
        wk_b: [H_b, Nkv_b*D_b] B 侧 key 投影权重（Nq_b % Nkv_b == 0）
        wv_b: [H_b, Nkv_b*D_b] B 侧 value 投影权重
        wo_b: [Nq_b*D_b, H_b] B 侧 attention 输出投影权重
        gamma2_b: [H_b] B 侧 MLP 前 RMSNorm 的 γ
        w_gate_b: [H_b, F_b] B 侧 SwiGLU gate 投影权重
        w_up_b: [H_b, F_b] B 侧 SwiGLU up 投影权重
        w_down_b: [F_b, H_b] B 侧 SwiGLU down 投影权重
        k_cache: [B_b, Nkv_b, Smax, D_b] key cache（本步写入槽位 cache_len[b]，其余槽位逐位保持）
        v_cache: [B_b, Nkv_b, Smax, D_b] value cache（同上）
        cache_len: [B_b] int32，各样本已有的有效 cache 长度 ∈ [1, Smax-1]（由 value_range 保证）
        rope_cos_a: [B_a, S_a, D_a] A 侧 RoPE 余弦（已逐位置索引好）
        rope_sin_a: [B_a, S_a, D_a] A 侧 RoPE 正弦（已逐位置索引好）
        rope_cos_b: [B_b, D_b] B 侧 RoPE 余弦（已按各样本当前位置索引好）
        rope_sin_b: [B_b, D_b] B 侧 RoPE 正弦（已按各样本当前位置索引好）
        epsilon: 四处 RMSNorm 的 epsilon（加在均方内），默认 1e-6

    Returns:
        y_a: [B_a, S_a, H_a] 负载 A 输出，dtype 与 x_a 一致，仅由 A 侧输入决定
        y_b: [B_b, 1, H_b] 负载 B 输出，dtype 与 x_b 一致，仅由 B 侧输入决定
        k_cache_out: [B_b, Nkv_b, Smax, D_b] 追加后的 key cache，shape/dtype 与 k_cache 一致
        v_cache_out: [B_b, Nkv_b, Smax, D_b] 追加后的 value cache，shape/dtype 与 v_cache 一致
    """
    y_a, y_b, kc, vc = _colocated_prefill_decode_core(
        x_a, gamma1_a, wq_a, wk_a, wv_a, wo_a, gamma2_a, w_gate_a, w_up_a, w_down_a,
        x_b, gamma1_b, wq_b, wk_b, wv_b, wo_b, gamma2_b, w_gate_b, w_up_b, w_down_b,
        k_cache, v_cache, cache_len, rope_cos_a, rope_sin_a, rope_cos_b, rope_sin_b,
        epsilon, torch.float32)
    return y_a.to(x_a.dtype), y_b.to(x_b.dtype), kc.to(k_cache.dtype), vc.to(v_cache.dtype)


def colocated_prefill_decode_oracle(
    x_a: torch.Tensor,
    gamma1_a: torch.Tensor,
    wq_a: torch.Tensor,
    wk_a: torch.Tensor,
    wv_a: torch.Tensor,
    wo_a: torch.Tensor,
    gamma2_a: torch.Tensor,
    w_gate_a: torch.Tensor,
    w_up_a: torch.Tensor,
    w_down_a: torch.Tensor,
    x_b: torch.Tensor,
    gamma1_b: torch.Tensor,
    wq_b: torch.Tensor,
    wk_b: torch.Tensor,
    wv_b: torch.Tensor,
    wo_b: torch.Tensor,
    gamma2_b: torch.Tensor,
    w_gate_b: torch.Tensor,
    w_up_b: torch.Tensor,
    w_down_b: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cache_len: torch.Tensor,
    rope_cos_a: torch.Tensor,
    rope_sin_a: torch.Tensor,
    rope_cos_b: torch.Tensor,
    rope_sin_b: torch.Tensor,
    epsilon: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Oracle (g)：dtype-agnostic，计算精度跟随输入（fp64_cpu 下为 fp64 真值），不硬编码 .float()。"""
    return _colocated_prefill_decode_core(
        x_a, gamma1_a, wq_a, wk_a, wv_a, wo_a, gamma2_a, w_gate_a, w_up_a, w_down_a,
        x_b, gamma1_b, wq_b, wk_b, wv_b, wo_b, gamma2_b, w_gate_b, w_up_b, w_down_b,
        k_cache, v_cache, cache_len, rope_cos_a, rope_sin_a, rope_cos_b, rope_sin_b,
        epsilon, x_a.dtype)
