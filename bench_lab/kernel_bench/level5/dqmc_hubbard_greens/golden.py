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

from typing import Tuple

import torch

"""
DqmcHubbardGreens 算子 Torch Golden 参考实现

行列式量子蒙特卡洛（DQMC/BSS 算法）的等时 Green 函数 + 行列式符号与对数行列式。

数学定义（详见 desc.md §2）:
    s_l = 2*aux_field[l] - 1 ∈ {-1,+1}^N
    K <- (K + K^T)/2（对称化，算子语义）
    E = expm(-dtau*K)（K 对称 => E 对称正定）
    B_l = diag(exp(lam*s_l)) @ E
    A = I + B_Ltau @ ... @ B_1
    greens = A^{-1}, sign = sgn(det A), logdet = log|det A|

稳定化（QR-UDT，对数域承载动态范围，desc §2 数值验证过的具体形式）:
    维护 P_partial = U @ diag(exp(logd)) @ T，U 正交、T 单位上三角。
    关键恒等式: QR(M @ diag(D)) 的 Q 与 QR(M) 的 Q 相同（列缩放不改变各列方向，
    正交化只依赖列方向与列顺序），且 R 恰为 R~ @ diag(D)。因此 QR 始终作用于
    O(1) 范数矩阵 M = B_group @ U，动态范围只存在于 logd 向量与 T 更新的比值
    因子 exp(logd_k - logd_j) 中:
        Q, R~ = QR(B_group @ U);  sigma = sign(diag R~)（0 取 +1）
        U' = Q diag(sigma);  logd'_j = logd_j + log|R~_jj|
        T' = [ sigma_j R~_jk / |R~_jj| * exp(logd_k - logd_j) ]_{j<=k} @ T （单位上三角）
    最终解（Loh-Gubernatis 大小尺度分离，log 域全程无上溢）:
        Db^{-1} = exp(-max(logd, 0)) <= 1,  Ds = exp(min(logd, 0)) <= 1
        A = I + U D T = U Db (Db^{-1} U^T T^{-1} + Ds) T
        M_inner = Db^{-1} 逐行缩放 (U^T T^{-1}) + diag(Ds)   （元素均 O(1)）
        greens = T^{-1} M_inner^{-1} (Db^{-1} 逐行缩放 U^T)
        logdet = sum_j max(logd_j, 0) + log|det M_inner|      （det T = 1）
        sign   = sgn(det U) * sgn(det M_inner)                （det Db > 0）

plain golden = bench：fp32 稳定化链（重正交组固定 8，即正确 fp32 kernel 的数据通路），
greens/logdet 输出 fp32，sign int32（零容差）；dqmc_hubbard_greens_oracle 浮点计算精度
跟随 kinetic（golden_precision=fp64_cpu 下即为 fp64 真值，fp64 下重正交组大小不影响
结果），fp32 输入下与 plain 逐位一致，不硬编码精度。
stab_interval 为被测 kernel 重正交节奏的上限（kernel 可更频繁重正交），不影响数学
结果；golden 按 desc §2 的工作精度可行性规则自适应选组（上限 8）。
"""


def _dqmc_udt_chain(aux_field, kinetic, lam, dtau, compute_dtype, group):
    """UDT 链：返回 (U, logd, T)，使 B_Ltau...B_1 = U @ diag(exp(logd)) @ T。"""
    n = kinetic.shape[0]
    ltau = aux_field.shape[0]
    dev = kinetic.device

    k_sym = kinetic.to(compute_dtype)
    k_sym = 0.5 * (k_sym + k_sym.transpose(0, 1))            # 对称化（算子语义）
    e_mat = torch.matrix_exp(-dtau * k_sym)                  # 对称正定
    s = aux_field.to(compute_dtype) * 2.0 - 1.0              # {0,1} -> {-1,+1}
    v_diag = torch.exp(lam * s)                              # [Ltau, N] 逐 slice 对角因子

    one = torch.tensor(1.0, dtype=compute_dtype, device=dev)
    u = torch.eye(n, dtype=compute_dtype, device=dev)
    logd = torch.zeros(n, dtype=compute_dtype, device=dev)
    t = torch.eye(n, dtype=compute_dtype, device=dev)

    steps = 0
    for l in range(ltau):
        u = v_diag[l].unsqueeze(1) * (e_mat @ u)             # B_l @ (U 工作矩阵)
        steps += 1
        if steps == group or l == ltau - 1:
            q, r = torch.linalg.qr(u)
            diag = r.diagonal()
            sigma = torch.where(diag < 0, -one, one)
            absd = diag.abs()
            u = q * sigma.unsqueeze(0)                       # 列符号修正 => diag(R') > 0
            rn = (sigma.unsqueeze(1) * r) / absd.unsqueeze(1)
            w = torch.exp(logd.unsqueeze(0) - logd.unsqueeze(1))   # w[j,k]=exp(logd_k-logd_j)
            factor = torch.triu(rn * w)
            factor.diagonal().fill_(1.0)                     # 单位上三角
            t = factor @ t
            logd = logd + torch.log(absd)
            steps = 0
    return u, logd, t


def _dqmc_stable_outputs(u, logd, t):
    """由 UDT 分解稳定计算 (greens, sign, logdet)，全程无大数上溢。"""
    n = u.shape[0]
    eye = torch.eye(n, dtype=u.dtype, device=u.device)
    t_inv = torch.linalg.solve_triangular(t, eye, upper=True, unitriangular=True)
    db_inv = torch.exp(-torch.clamp(logd, min=0.0))          # ≤ 1
    ds = torch.exp(torch.clamp(logd, max=0.0))               # ≤ 1
    m_inner = db_inv.unsqueeze(1) * (u.transpose(0, 1) @ t_inv) + torch.diag(ds)

    sgn_inner, logabs_inner = torch.linalg.slogdet(m_inner)
    sgn_u, _ = torch.linalg.slogdet(u)                       # 正交阵，det = ±1
    logdet = torch.clamp(logd, min=0.0).sum() + logabs_inner
    sign = sgn_u * sgn_inner
    greens = t_inv @ torch.linalg.solve(m_inner, db_inv.unsqueeze(1) * u.transpose(0, 1))
    return greens, sign, logdet


def _dqmc_hubbard_greens_core(aux_field, kinetic, lam, dtau, compute_dtype, group=1):
    """核心计算：QR-UDT 稳定化链 + 稳定输出。group 为重正交组大小。"""
    u, logd, t = _dqmc_udt_chain(aux_field, kinetic, lam, dtau, compute_dtype, group)
    return _dqmc_stable_outputs(u, logd, t)


def _stable_group_size(kinetic, lam, dtau, cap=8, budget=10.0):
    """按工作精度可行性规则选重正交组大小（desc §2）：
    组内对数动态范围 group * (dtau*max|λ(K_sym)| + |lam|) ≤ budget（≈ ln(1/eps_fp32) 留裕量），
    并以 cap 为上限。golden 参考实现固定 cap=8。"""
    k_sym = 0.5 * (kinetic + kinetic.transpose(0, 1))
    lam_max = float(torch.linalg.eigvalsh(k_sym.float()).abs().max())
    per_slice = abs(dtau) * lam_max + abs(lam)
    return max(1, min(cap, int(budget / max(per_slice, 1e-6))))


def dqmc_hubbard_greens(
    aux_field: torch.Tensor,
    kinetic: torch.Tensor,
    lam: float,
    dtau: float,
    stab_interval: int = 8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """plain golden = bench：fp32 稳定化链（重正交组固定 8，即正确 fp32 kernel 的数据通路），
    greens/logdet 输出 fp32，sign int32。

    stab_interval 为被测 kernel 重正交节奏的上限（kernel 可更频繁重正交），不影响数学
    结果（数学值 = fp64 真值，见 oracle）；golden 参考实现按 §2 的工作精度可行性规则
    自适应选组（上限 8），不使用该 attr。
    """
    group = _stable_group_size(kinetic, float(lam), float(dtau))
    greens, sign, logdet = _dqmc_hubbard_greens_core(
        aux_field, kinetic, float(lam), float(dtau), torch.float32, group=group)
    return (greens.to(torch.float32),
            sign.to(torch.int32).reshape(1),
            logdet.to(torch.float32).reshape(1))


def dqmc_hubbard_greens_oracle(
    aux_field: torch.Tensor,
    kinetic: torch.Tensor,
    lam: float,
    dtau: float,
    stab_interval: int = 8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Oracle (g)：dtype-agnostic，浮点计算精度跟随 kinetic（golden_precision=fp64_cpu 下即
    fp64 真值；fp64 下重正交组大小不影响结果，实测组 1/4/8 偏差 < 3.3e-12）；
    sign 恒为 int32（类别量）。fp32 输入下与 plain golden 逐位一致。"""
    group = _stable_group_size(kinetic, float(lam), float(dtau))
    greens, sign, logdet = _dqmc_hubbard_greens_core(
        aux_field, kinetic, float(lam), float(dtau), kinetic.dtype, group=group)
    return (greens.to(kinetic.dtype),
            sign.to(torch.int32).reshape(1),
            logdet.to(kinetic.dtype).reshape(1))
