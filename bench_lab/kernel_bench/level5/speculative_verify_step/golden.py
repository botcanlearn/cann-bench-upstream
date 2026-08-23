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
SpeculativeVerifyStep 算子 Torch Golden 参考实现

投机解码验证步（speculative decoding verify step）：draft 模型已产出 T 个候选 token，
本算子对照 target 模型分布做标准拒绝采样验证（Leviathan & Matias 2023 / Chen et al. 2023），
输出接受长度与最终 token 序列。两个输出均为 int32，零容差精确比对。

确定性设计（见 desc.md §2）：
1. p_target / p_draft 为非负未归一权重，算子内部沿 V 归一化：P = w / S。
   所有沿 V 的求和/前缀和定义为：按索引升序、以 ≥ fp64 精度累加，每个前缀正确舍入回
   工作精度 fp32（torch CPU cumsum 对 fp32 的确切语义）。评测值域下该结果与"fp32 权重
   精确和的正确舍入"一致，对累加顺序不敏感（可并行归约复现，见 desc.md §4）。
   S 取前缀和末元素，P = w / S 为单次 fp32 除法（权重下界 > 0，无除零）。
2. 接受判定写成单次乘法 + 比较：accept_noise[b,i] * P_d[i][tok] < P_t[i][tok]
   （不做除法；单次 IEEE 精确舍入的乘法与比较，跨实现逐位确定）。
3. 拒绝位置 i 的修正采样与全接受时的 bonus 采样均为逆 CDF：
   c = cumsum(r)（语义同上），total = c[V-1]，
   取最小的 j 使 c[j] > resample_noise[b] * total（乘法形式，同样无除法）。
4. 残差 r = max(P_t[i] − P_d[i], 0)；若 r 恰为全零则回退 r = P_t[i]
   （完备性定义：归一化分布下残差全零蕴含 P_t == P_d，此时验证必接受，
   该分支仅在极端浮点边界下可触达）。

plain golden 判定路径固定 fp32（bench 语义）。两个输出均为 int32 结构输出，算子语义
在 fp32 判定粒度下**定义**（fp32 舍入是规格的一部分，不是精度捷径），因此
speculative_verify_step_oracle 复用同一 fp32 判定核心：golden_precision=fp64_cpu 下
evaluator 喂入的 fp64 输入（由 fp32 无损升精度而来）先舍回 fp32，逐位还原原始输入，
oracle 与 plain 逐位一致。若改为在 fp64 下判定，接受判定不受影响（实测零翻转、最小
相对裕度 3.6e-5 ≫ 双路径舍入差），但逆 CDF 的桶边界宽度 ~1/V，对判定精度提升敏感
（实测 505 次运行出现 5 次索引偏移），故规格明确判定操作数为 fp32 舍入值（见 desc.md §4）。
"""


def _speculative_verify_step_core(draft_tokens, p_target, p_draft, accept_noise,
                                  resample_noise, compute_dtype):
    """核心计算：以 compute_dtype 精度执行归一化 / 接受判定 / 逆 CDF 重采样。"""
    B, T = draft_tokens.shape
    V = p_target.shape[-1]
    device = draft_tokens.device

    w_t = p_target.to(compute_dtype)          # [B, T+1, V] 非负未归一权重
    w_d = p_draft.to(compute_dtype)           # [B, T, V]
    noise_a = accept_noise.to(compute_dtype)  # [B, T]
    noise_r = resample_noise.to(compute_dtype)  # [B]

    accept_len = torch.empty(B, dtype=torch.int32, device=device)
    output_tokens = torch.full((B, T + 1), -1, dtype=torch.int32, device=device)
    row_idx = torch.arange(T, device=device)

    for b in range(B):
        # === 1. 归一化（分母 = 沿 V 的前缀和末元素；cumsum 以 ≥ fp64 精度累加后舍回工作精度）===
        s_t = torch.cumsum(w_t[b], dim=-1)[:, -1:]   # [T+1, 1]
        s_d = torch.cumsum(w_d[b], dim=-1)[:, -1:]   # [T, 1]
        p_t = w_t[b] / s_t                           # [T+1, V] 归一化目标分布
        p_d = w_d[b] / s_d                           # [T, V]   归一化 draft 分布

        # === 2. 顺序验证：accept_noise * P_d[i][tok] < P_t[i][tok]（单次乘法 + 比较）===
        toks = draft_tokens[b].long()                # [T]
        pt_tok = p_t[row_idx, toks]                  # [T] P_t[i][tok_i]（i = 0..T-1）
        pd_tok = p_d[row_idx, toks]                  # [T]
        accept = noise_a[b] * pd_tok < pt_tok        # [T] bool
        rejected = torch.nonzero(~accept)
        n_acc = int(rejected[0].item()) if rejected.numel() > 0 else T

        # === 3./4. 拒绝 → 残差分布逆 CDF 重采样；全接受 → bonus 从 P_t[T] 采样 ===
        if n_acc < T:
            row = torch.clamp(p_t[n_acc] - p_d[n_acc], min=0)   # 残差 r = max(P_t - P_d, 0)
            cum = torch.cumsum(row, dim=-1)                     # 前缀和（≥ fp64 累加，舍回工作精度）
            total = cum[-1]
            if total == 0:                                      # 残差恰为全零 → 回退 r = P_t[i]
                row = p_t[n_acc]
                cum = torch.cumsum(row, dim=-1)
                total = cum[-1]
        else:
            row = p_t[T]                                        # bonus token 分布
            cum = torch.cumsum(row, dim=-1)
            total = cum[-1]
        # 逆 CDF：最小的 j 使 cum[j] > resample_noise * total（乘法形式，无除法）。
        # resample_noise ≤ 0.9999 保证阈值 < total = cum[V-1]，j 必存在。
        thr = noise_r[b] * total
        sel = int(torch.searchsorted(cum, thr, right=True).item())

        # === 5. 输出组装 ===
        accept_len[b] = n_acc
        output_tokens[b, :n_acc] = draft_tokens[b, :n_acc]
        output_tokens[b, n_acc] = sel
    return accept_len, output_tokens


def speculative_verify_step(
    draft_tokens: torch.Tensor,
    p_target: torch.Tensor,
    p_draft: torch.Tensor,
    accept_noise: torch.Tensor,
    resample_noise: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    投机解码验证步 golden reference（plain golden = bench：判定路径固定 fp32）

    Args:
        draft_tokens: [B, T] int32，draft 模型产出的候选 token，取值 [0, V-1]
        p_target: [B, T+1, V] float32，target 模型的非负未归一权重（算子内部沿 V 归一化），
                  行 0..T-1 用于验证，行 T 用于全接受时的 bonus 采样
        p_draft: [B, T, V] float32，draft 模型的非负未归一权重（算子内部沿 V 归一化）
        accept_noise: [B, T] float32，接受判定随机数，取值 (0, 1)（评测范围 [0.0001, 0.9999]）
        resample_noise: [B] float32，重采样/bonus 采样随机数，取值 (0, 1)（同上）

    Returns:
        accept_len: [B] int32，接受的 draft token 个数（0..T）
        output_tokens: [B, T+1] int32，前 accept_len 个接受 token + 1 个修正/bonus token，
                       其余位置填 -1
    """
    return _speculative_verify_step_core(
        draft_tokens, p_target, p_draft, accept_noise, resample_noise, torch.float32)


def speculative_verify_step_oracle(
    draft_tokens: torch.Tensor,
    p_target: torch.Tensor,
    p_draft: torch.Tensor,
    accept_noise: torch.Tensor,
    resample_noise: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Oracle (g)：int32 结构输出，算子语义在 fp32 判定粒度下定义（fp32 舍入是规格的一部分，
    见 desc.md §2/§4），oracle 复用同一 fp32 判定核心。fp64_cpu 下 evaluator 喂入的 fp64
    输入由 fp32 无损升精度而来，核心内舍回 fp32 即逐位还原原始输入，oracle ≡ plain。
    （若按输入 dtype 提升判定精度，逆 CDF 桶边界（宽 ~1/V）会以 ~1% 概率偏离规格定义的
    索引，破坏 int 零容差评测，见 desc.md §4 实测。）"""
    return _speculative_verify_step_core(
        draft_tokens, p_target, p_draft, accept_noise, resample_noise, torch.float32)
