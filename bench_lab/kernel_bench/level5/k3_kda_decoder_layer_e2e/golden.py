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

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

"""
K3KdaDecoderLayerE2E：Kimi K3 KDA 型 decoder 层「整层端到端」融合
（注意力子层 + LatentMoE 子层，decode / 推测解码验证负载，多 die 协同）。

语义来源（逐行对齐源码，非论文默写）:
- moonshotai/Kimi-K3 modeling_kimi_linear.py:
  KimiDecoderLayer._forward_attn_residual（AttnRes 双分支混合、快照、载体 prefix_sum 语义）
  KimiMoEGate（sigmoid 打分 + e_score_correction_bias 选 top-k，权重取自未校正分数，
    renormalize 除以 sum+1e-20，×routed_scaling_factor；分组路由分支保留）
  KimiSparseMoeBlock（latent 降维 -> 专家 -> latent 加权求和 -> RMSNorm -> 升维；
    共享专家在 hidden 空间与路由输出相加）
  SituAndMul（gate/up 拼接输入：β·tanh(g/β)·sigmoid(g) × lβ·tanh(u/lβ)，内部 fp32）
- moonshotai/Kimi-K3 modeling_kimi_linear.py KimiDeltaAttention（注意力层结构权威：
  f_a/f_b 低秩衰减门、全秩输出门 g_proj、全部投影无 bias）
- fla-org/flash-linear-attention（递归核与门归一化语义）:
  fla/ops/kda/fused_recurrent.py、fla/modules/fused_norm_gate.py

整层流程（全局语义，与卡数/并行方式无关）:
 A1 x_mix = AttnResMix(x, R; 注意力分支参数)；快照层 R_out = concat(R, x)，载体置空
 A2 h = RMSNorm(x_mix)；qkv = SiLU(CausalConv1dUpdate(h·[Wq|Wk|Wv], conv_state))
 A3 g_log = ℓ·sigmoid(exp(A_log)·((h·Wf1)Wf2 + dt_bias))（ℓ=null 时 -exp(A_log)·softplus(·)）
    # 衰减门低秩两级 f_a/f_b、输出门全秩单级 g_proj（use_full_rank_gate=True），均无 bias
    β = sigmoid(h·Wb)；q̂,k̂ = L2Norm(q,k)（eps=1e-6 加在平方和上），q̂ ×= scale
 A4 S ⊙= exp(g_log)；u = β⊙(v − Sᵀk̂)；S += k̂ uᵀ；o = Sᵀq̂
 A5 o = RMSNorm_head(o)⊙sigmoid(h·Wg)；attn = o·Wo
 A6 x2 = x + attn（快照层 x2 = attn）
 M1 m_mix = AttnResMix(x2, R_out; MLP 分支参数)；h2 = RMSNorm(m_mix)
 M2 路由：logits = h2·Wrᵀ；s = sigmoid(logits)；idx = topk(s + bias, k)
    w = s[idx]；renormalize 时 w /= (Σw + 1e-20)；w ×= routed_scaling_factor
 M3 z = h2·W_lat_down；每 token 对其 k 个专家：
    e_out = SituAndMul(z·W_gu[e])·W_dn[e]；y_lat = Σ w·e_out（latent 空间加权求和）
 M4 y_lat = RMSNorm(y_lat)；y_r = y_lat·W_lat_up
 M5 shared = SituAndMul(h2·W_sh_gu)·W_sh_dn；moe = y_r + shared
 M6 y = x2 + moe

padding 约定：x [B, Tmax, H] 配 seq_lens，无效位 y=0，
状态只随有效 token 推进。plain golden 内部 fp32（bench 语义），输出转回输入 dtype；
oracle 跟随输入精度。SituAndMul 与路由在建模源码中即为 fp32 计算，plain 与其逐位对齐。
"""


# ============================ 注意力子层原语 ============================

def _rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight


def _attn_res_mix(x2d, blocks3d, proj_w, norm_w, eps):
    v = torch.cat([blocks3d, x2d.unsqueeze(1)], dim=1)
    v_hat = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + eps)
    scores = (v_hat * (norm_w * proj_w)).sum(-1)
    probs = scores.softmax(-1).unsqueeze(1)
    return torch.matmul(probs, v).squeeze(1)


def _causal_conv1d_update(seq_cd, hist_cd, w_conv):
    W = w_conv.shape[-1]
    full = torch.cat([hist_cd, seq_cd], dim=1)
    win = full.unfold(dimension=1, size=W, step=1)
    y = F.silu((win * w_conv.unsqueeze(1)).sum(-1))
    return y, full[:, full.shape[1] - (W - 1):]


def _kda_recurrence(q_hat, k_hat, v, g_log, beta, state):
    L = q_hat.shape[0]
    o = torch.empty_like(v)
    for t in range(L):
        state = state * torch.exp(g_log[t]).unsqueeze(-1)
        r = (state * k_hat[t].unsqueeze(-1)).sum(dim=-2)
        u = beta[t].unsqueeze(-1) * (v[t] - r)
        state = state + k_hat[t].unsqueeze(-1) * u.unsqueeze(-2)
        o[t] = (state * q_hat[t].unsqueeze(-1)).sum(dim=-2)
    return o, state


# ============================ MoE 子层原语 ============================

def _situ_and_mul(gate_up, beta, linear_beta):
    """SituAndMul：输入为 cat(gate, up)，对齐 modeling SituAndMul.forward。"""
    d = gate_up.shape[-1] // 2
    gate, up = gate_up[..., :d], gate_up[..., d:]
    a = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
    if linear_beta is not None:
        up = linear_beta * torch.tanh(up / linear_beta)
    return a * up


def _validate_routing(num_experts, top_k, num_expert_group, topk_group):
    """路由参数硬约束（违反时组分 top2 越界或 top-k 溢出到被 mask 组）。"""
    if num_expert_group < 1 or num_experts % num_expert_group != 0:
        raise ValueError(
            f"num_expert_group={num_expert_group} 必须为正且整除 E={num_experts}")
    group = num_experts // num_expert_group
    if group < 2:
        raise ValueError(
            f"每组专家数 E/num_expert_group={group} 必须 >= 2（组分为组内 top2 之和）")
    if not 1 <= topk_group <= num_expert_group:
        raise ValueError(
            f"topk_group={topk_group} 必须在 [1, num_expert_group={num_expert_group}] 内")
    if not 1 <= top_k <= topk_group * group:
        raise ValueError(
            f"top_k={top_k} 必须在 [1, topk_group*(E/num_expert_group)="
            f"{topk_group * group}] 内，否则 top-k 会选入被 mask 组的专家")


def _moe_gate(h2d, router_w, router_bias, top_k, num_expert_group, topk_group,
              renormalize, scaling):
    """对齐 KimiMoEGate.forward（推理分支）。h2d [N,H] -> (idx [N,k], w [N,k])。"""
    _validate_routing(router_w.shape[0], top_k, num_expert_group, topk_group)
    logits = h2d @ router_w.t()
    scores = torch.sigmoid(logits)                                   # 打分
    choice = scores + router_bias                                    # 选择用校正分
    if num_expert_group > 1 and num_expert_group > topk_group:
        N, E = choice.shape
        gs = choice.view(N, num_expert_group, -1).topk(2, dim=-1)[0].sum(-1)
        gidx = torch.topk(gs, k=topk_group, dim=-1, sorted=False)[1]
        gmask = torch.zeros_like(gs).scatter_(1, gidx, 1)
        smask = gmask.unsqueeze(-1).expand(N, num_expert_group,
                                           E // num_expert_group).reshape(N, E)
        choice = choice.masked_fill(~smask.bool(), float("-inf"))
    idx = torch.topk(choice, k=top_k, dim=-1, sorted=False)[1]
    w = scores.gather(1, idx)                                        # 权重取未校正分数
    if top_k > 1 and renormalize:
        w = w / (w.sum(dim=-1, keepdim=True) + 1e-20)
    return idx, w * scaling


def _moe_sublayer(h2d, router_w, router_bias, w_latent_down, w_latent_up,
                  latent_norm_w, w_expert_gate_up, w_expert_down,
                  w_shared_gate_up, w_shared_down,
                  top_k, num_expert_group, topk_group, renormalize, scaling,
                  situ_beta, situ_linear_beta, eps):
    """对齐 KimiSparseMoeBlock.forward + moe_infer（逐专家聚集，latent 空间加权求和）。"""
    N = h2d.shape[0]
    E = router_w.shape[0]
    idx, w = _moe_gate(h2d, router_w, router_bias, top_k,
                       num_expert_group, topk_group, renormalize, scaling)
    z = h2d @ w_latent_down                                          # [N, L]
    y = torch.zeros(N, z.shape[-1], dtype=z.dtype, device=z.device)
    for e in range(E):
        tok, slot = torch.where(idx == e)
        if tok.numel() == 0:
            continue
        gu = z[tok] @ w_expert_gate_up[e]
        eo = _situ_and_mul(gu, situ_beta, situ_linear_beta) @ w_expert_down[e]
        y.index_add_(0, tok, eo * w[tok, slot].unsqueeze(-1))
    y = _rmsnorm(y, latent_norm_w, eps)                              # latent_moe_use_norm
    y = y @ w_latent_up
    shared = _situ_and_mul(h2d @ w_shared_gate_up,
                           situ_beta, situ_linear_beta) @ w_shared_down
    return y + shared


# ============================ 整层核心 ============================

def _k3_layer_core(
    x, seq_lens, block_residual, conv_state, recurrent_state,
    w_q, w_k, w_v, w_conv, w_f1, w_f2, w_b, w_g, A_log, dt_bias,
    rms_in_w, o_norm_w, w_o, attnres_attn_norm_w, attnres_attn_proj_w,
    attnres_mlp_norm_w, attnres_mlp_proj_w, post_attn_norm_w,
    router_w, router_bias, w_latent_down, w_latent_up, latent_norm_w,
    w_expert_gate_up, w_expert_down, w_shared_gate_up, w_shared_down,
    world_size, num_heads, num_v_heads, head_dim, head_v_dim, conv_size,
    gate_lower_bound, allow_neg_eigval, append_snapshot, rms_eps, scale,
    num_experts, top_k, num_expert_group, topk_group,
    moe_renormalize, routed_scaling_factor, situ_beta, situ_linear_beta,
    compute_dtype,
):
    del world_size, conv_size, num_experts  # 执行属性 / 由形状推出
    B, Tmax, Hs = x.shape
    Hq, HV, Kd, Vd = num_heads, num_v_heads, head_dim, head_v_dim
    G = HV // Hq
    NB = block_residual.shape[2]
    if scale is None:
        scale = Kd ** -0.5
    cd = compute_dtype
    xf = x.to(cd)

    # ---- A1 注意力分支 AttnRes 混合与快照 ----
    if NB > 0:
        mixed = _attn_res_mix(
            xf.reshape(B * Tmax, Hs),
            block_residual.to(cd).reshape(B * Tmax, NB, Hs),
            attnres_attn_proj_w.to(cd), attnres_attn_norm_w.to(cd), rms_eps,
        ).reshape(B, Tmax, Hs)
    else:
        mixed = xf
    if append_snapshot:
        block_residual_out = torch.cat([block_residual, x.unsqueeze(2)], dim=2)
    else:
        block_residual_out = block_residual.clone()

    # ---- A2 输入归一化、投影、短卷积 ----
    h = _rmsnorm(mixed, rms_in_w.to(cd), rms_eps)
    qkv_pre = h @ torch.cat([w_q.to(cd), w_k.to(cd), w_v.to(cd)], dim=1)
    conv_out = torch.zeros_like(qkv_pre)
    conv_state_out = conv_state.to(cd).clone()
    wc = w_conv.to(cd)
    for b in range(B):
        L = int(seq_lens[b])
        y_b, hist_b = _causal_conv1d_update(
            qkv_pre[b, :L].transpose(0, 1), conv_state[b].to(cd), wc)
        conv_out[b, :L] = y_b.transpose(0, 1)
        conv_state_out[b] = hist_b
    dq = Hq * Kd
    q = conv_out[..., :dq].view(B, Tmax, Hq, Kd)
    k = conv_out[..., dq:2 * dq].view(B, Tmax, Hq, Kd)
    v = conv_out[..., 2 * dq:].view(B, Tmax, HV, Vd)

    # ---- A3 门 ----
    f = ((h @ w_f1.to(cd)) @ w_f2.to(cd) + dt_bias.to(cd)).view(B, Tmax, HV, Kd)
    a = torch.exp(A_log.to(cd)).view(1, 1, HV, 1)
    if gate_lower_bound is not None:
        g_log = gate_lower_bound * torch.sigmoid(a * f)
    else:
        g_log = -a * F.softplus(f)
    beta = torch.sigmoid(h @ w_b.to(cd)).view(B, Tmax, HV)
    if allow_neg_eigval:
        beta = beta * 2.0

    # ---- A4 递归 ----
    q = q.repeat_interleave(G, dim=2)
    k = k.repeat_interleave(G, dim=2)
    q_hat = q / torch.sqrt(q.pow(2).sum(-1, keepdim=True) + 1e-6) * scale
    k_hat = k / torch.sqrt(k.pow(2).sum(-1, keepdim=True) + 1e-6)
    o = torch.zeros(B, Tmax, HV, Vd, dtype=cd, device=x.device)
    recurrent_state_out = recurrent_state.to(cd).clone()
    for b in range(B):
        L = int(seq_lens[b])
        o[b, :L], recurrent_state_out[b] = _kda_recurrence(
            q_hat[b, :L], k_hat[b, :L], v[b, :L], g_log[b, :L], beta[b, :L],
            recurrent_state_out[b])

    # ---- A5 输出门归一化与投影（输出门全秩无 bias） ----
    g_out = (h @ w_g.to(cd)).view(B, Tmax, HV, Vd)
    o = _rmsnorm(o, o_norm_w.to(cd), rms_eps) * torch.sigmoid(g_out)
    attn = o.reshape(B, Tmax, HV * Vd) @ w_o.to(cd)

    # ---- A6 载体更新 ----
    x2 = attn if append_snapshot else xf + attn

    # ---- M1 MLP 分支 AttnRes 混合（用更新后的快照）与 post-attn 归一化 ----
    NB2 = block_residual_out.shape[2]
    m_mix = _attn_res_mix(
        x2.reshape(B * Tmax, Hs),
        block_residual_out.to(cd).reshape(B * Tmax, NB2, Hs),
        attnres_mlp_proj_w.to(cd), attnres_mlp_norm_w.to(cd), rms_eps,
    ).reshape(B, Tmax, Hs)
    h2 = _rmsnorm(m_mix, post_attn_norm_w.to(cd), rms_eps)

    # ---- M2~M5 MoE ----
    moe = _moe_sublayer(
        h2.reshape(B * Tmax, Hs),
        router_w.to(cd), router_bias.to(cd),
        w_latent_down.to(cd), w_latent_up.to(cd), latent_norm_w.to(cd),
        w_expert_gate_up.to(cd), w_expert_down.to(cd),
        w_shared_gate_up.to(cd), w_shared_down.to(cd),
        top_k, num_expert_group, topk_group, moe_renormalize,
        routed_scaling_factor, situ_beta, situ_linear_beta, rms_eps,
    ).reshape(B, Tmax, Hs)

    # ---- M6 载体更新与 padding 置零 ----
    y = x2 + moe
    y = y * (torch.arange(Tmax, device=x.device)[None, :]
             < seq_lens.to(torch.int64)[:, None]).unsqueeze(-1).to(cd)
    return y, conv_state_out, recurrent_state_out, block_residual_out


_ARG_NAMES = [
    "x", "seq_lens", "block_residual", "conv_state", "recurrent_state",
    "w_q", "w_k", "w_v", "w_conv", "w_f1", "w_f2", "w_b", "w_g",
    "A_log", "dt_bias", "rms_in_w", "o_norm_w", "w_o",
    "attnres_attn_norm_w", "attnres_attn_proj_w",
    "attnres_mlp_norm_w", "attnres_mlp_proj_w", "post_attn_norm_w",
    "router_w", "router_bias", "w_latent_down", "w_latent_up", "latent_norm_w",
    "w_expert_gate_up", "w_expert_down", "w_shared_gate_up", "w_shared_down",
]


def k3_kda_decoder_layer_e2e(
    x: torch.Tensor, seq_lens: torch.Tensor, block_residual: torch.Tensor,
    conv_state: torch.Tensor, recurrent_state: torch.Tensor,
    w_q: torch.Tensor, w_k: torch.Tensor, w_v: torch.Tensor,
    w_conv: torch.Tensor, w_f1: torch.Tensor, w_f2: torch.Tensor,
    w_b: torch.Tensor, w_g: torch.Tensor,
    A_log: torch.Tensor, dt_bias: torch.Tensor,
    rms_in_w: torch.Tensor, o_norm_w: torch.Tensor, w_o: torch.Tensor,
    attnres_attn_norm_w: torch.Tensor, attnres_attn_proj_w: torch.Tensor,
    attnres_mlp_norm_w: torch.Tensor, attnres_mlp_proj_w: torch.Tensor,
    post_attn_norm_w: torch.Tensor,
    router_w: torch.Tensor, router_bias: torch.Tensor,
    w_latent_down: torch.Tensor, w_latent_up: torch.Tensor,
    latent_norm_w: torch.Tensor,
    w_expert_gate_up: torch.Tensor, w_expert_down: torch.Tensor,
    w_shared_gate_up: torch.Tensor, w_shared_down: torch.Tensor,
    world_size: int = 8,
    num_heads: int = 96, num_v_heads: int = 96,
    head_dim: int = 128, head_v_dim: int = 128, conv_size: int = 4,
    gate_lower_bound: Optional[float] = -5.0,
    allow_neg_eigval: bool = False, append_snapshot: bool = False,
    rms_eps: float = 1e-5, scale: Optional[float] = None,
    num_experts: int = 896, top_k: int = 16,
    num_expert_group: int = 1, topk_group: int = 1,
    moe_renormalize: bool = True, routed_scaling_factor: float = 1.0,
    situ_beta: float = 4.0, situ_linear_beta: Optional[float] = 25.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    K3 KDA 型 decoder 层整层 golden（plain golden = bench：内部 fp32，状态 fp32）。

    注意力子层参数（衰减门 w_f1/w_f2 低秩、输出门 w_g 全秩，
    AttnRes 权重按分支拆成 attn/mlp 两套）。
    MoE 侧：
        router_w [E, H]，router_bias [E] fp32（e_score_correction_bias）
        w_latent_down [H, L]，w_latent_up [L, H]，latent_norm_w [L]（L=routed_expert_hidden_size）
        w_expert_gate_up [E, L, 2I]（前 I 列 gate，后 I 列 up），w_expert_down [E, I, L]
        w_shared_gate_up [H, 2Is]，w_shared_down [Is, H]（Is = num_shared_experts × I）
    返回 (y, conv_state_out, recurrent_state_out, block_residual_out)，
    y 为经过两个子层与两次载体残差后的隐状态，padding 位置为 0。
    """
    args = locals()
    y, cs, rs, br = _k3_layer_core(
        *[args[n] for n in _ARG_NAMES],
        world_size, num_heads, num_v_heads, head_dim, head_v_dim, conv_size,
        gate_lower_bound, allow_neg_eigval, append_snapshot, rms_eps, scale,
        num_experts, top_k, num_expert_group, topk_group,
        moe_renormalize, routed_scaling_factor, situ_beta, situ_linear_beta,
        torch.float32,
    )
    return (y.to(x.dtype), cs.to(conv_state.dtype),
            rs.to(recurrent_state.dtype), br)


def k3_kda_decoder_layer_e2e_oracle(
    x, seq_lens, block_residual, conv_state, recurrent_state,
    w_q, w_k, w_v, w_conv, w_f1, w_f2, w_b, w_g, A_log, dt_bias,
    rms_in_w, o_norm_w, w_o, attnres_attn_norm_w, attnres_attn_proj_w,
    attnres_mlp_norm_w, attnres_mlp_proj_w, post_attn_norm_w,
    router_w, router_bias, w_latent_down, w_latent_up, latent_norm_w,
    w_expert_gate_up, w_expert_down, w_shared_gate_up, w_shared_down,
    world_size: int = 8,
    num_heads: int = 96, num_v_heads: int = 96,
    head_dim: int = 128, head_v_dim: int = 128, conv_size: int = 4,
    gate_lower_bound: Optional[float] = -5.0,
    allow_neg_eigval: bool = False, append_snapshot: bool = False,
    rms_eps: float = 1e-5, scale: Optional[float] = None,
    num_experts: int = 896, top_k: int = 16,
    num_expert_group: int = 1, topk_group: int = 1,
    moe_renormalize: bool = True, routed_scaling_factor: float = 1.0,
    situ_beta: float = 4.0, situ_linear_beta: Optional[float] = 25.0,
):
    """Oracle (g)：dtype-agnostic，计算精度跟随输入（fp64_cpu 下为 fp64 真值）。"""
    args = locals()
    return _k3_layer_core(
        *[args[n] for n in _ARG_NAMES],
        world_size, num_heads, num_v_heads, head_dim, head_v_dim, conv_size,
        gate_lower_bound, allow_neg_eigval, append_snapshot, rms_eps, scale,
        num_experts, top_k, num_expert_group, topk_group,
        moe_renormalize, routed_scaling_factor, situ_beta, situ_linear_beta,
        x.dtype,
    )
