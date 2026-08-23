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
CcsdDoublesContraction 算子 Torch Golden 参考实现

耦合簇 CCSD/CCD 迭代中 T2 残差的主缩并项（自旋轨道反对称化形式的代表性子集）:
粒子-粒子 ladder（O(o²v⁴)）+ ring（O(o³v³)）+ 单粒子 Fock 项（O(o²v³)+O(o³v²)）。
o = 占据轨道数, v = 虚轨道数。

投影语义（第一步，保证任意随机输入合法）: 五个输入先各自投影到其物理对称类
    A[i,j,a,b] = (t2[i,j,a,b] − t2[j,i,a,b] − t2[i,j,b,a] + t2[j,i,b,a]) / 4      （t2 反对称化）
    W[a,b,c,d] = (eri_vvvv[a,b,c,d] − eri_vvvv[b,a,c,d] − eri_vvvv[a,b,d,c]
                  + eri_vvvv[b,a,d,c]) / 4                                        （⟨ab||cd⟩ 反对称化）
    X[i,a,j,b] = (eri_ovov[i,a,j,b] + eri_ovov[j,b,i,a]) / 2                      （ovov 对称化）
    Fv = (fock_vv + fock_vvᵀ) / 2,   Fo = (fock_oo + fock_ooᵀ) / 2                （Fock 对称化）
投影均为幂等算子；对已满足对称性的物理输入是恒等变换。

缩并定义（P(ij) f(i,j) = f(i,j) − f(j,i)，P(ab) 同）:
    r2[i,j,a,b] = Σ_{cd} W[a,b,c,d] A[i,j,c,d]                    （ladder）
                + P(ij)P(ab) Σ_{kc} X[k,c,j,b] A[i,k,a,c]         （ring, 展开 4 项）
                + P(ab)      Σ_c   Fv[b,c] A[i,j,a,c]             （粒子 Fock, 展开 2 项）
                − P(ij)      Σ_k   Fo[j,k] A[i,k,a,b]             （空穴 Fock, 展开 2 项）
投影语义下输出精确满足反对称契约 r2[i,j,a,b] = −r2[j,i,a,b] = −r2[i,j,b,a]
（对任意随机输入数学成立）。

plain golden 内部升 fp32 计算（bench 语义），输出转回输入 dtype；
ccsd_doubles_contraction_oracle 跟随输入精度（golden_precision=fp64_cpu 下即为 fp64 真值）。
"""


def _ccsd_doubles_contraction_core(t2, eri_vvvv, eri_ovov, fock_vv, fock_oo, compute_dtype):
    """核心计算：投影 + 9 项缩并，以 compute_dtype 精度执行，返回 r2 [o, o, v, v]。"""
    a = t2.to(compute_dtype)
    w = eri_vvvv.to(compute_dtype)
    x = eri_ovov.to(compute_dtype)
    fv = fock_vv.to(compute_dtype)
    fo = fock_oo.to(compute_dtype)

    # === 第一步：对称类投影（幂等） ===
    a = (a - a.permute(1, 0, 2, 3) - a.permute(0, 1, 3, 2) + a.permute(1, 0, 3, 2)) / 4
    w = (w - w.permute(1, 0, 2, 3) - w.permute(0, 1, 3, 2) + w.permute(1, 0, 3, 2)) / 4
    x = (x + x.permute(2, 3, 0, 1)) / 2
    fv = (fv + fv.transpose(0, 1)) / 2
    fo = (fo + fo.transpose(0, 1)) / 2

    # === ladder: Σ_{cd} W[a,b,c,d] A[i,j,c,d] ===
    r2 = torch.einsum('abcd,ijcd->ijab', w, a)

    # === ring: P(ij)P(ab) Σ_{kc} X[k,c,j,b] A[i,k,a,c] ===
    # T[i,j,a,b] = Σ_{kc} X[k,c,j,b] A[i,k,a,c]；
    # 其余 3 项是 T 的输出下标重排（与逐项独立 einsum 逐位一致）:
    #   −T[j,i,a,b] = −Σ X[k,c,i,b] A[j,k,a,c]
    #   −T[i,j,b,a] = −Σ X[k,c,j,a] A[i,k,b,c]
    #   +T[j,i,b,a] = +Σ X[k,c,i,a] A[j,k,b,c]
    ring = torch.einsum('kcjb,ikac->ijab', x, a)
    r2 = r2 + ring - ring.permute(1, 0, 2, 3) - ring.permute(0, 1, 3, 2) + ring.permute(1, 0, 3, 2)

    # === 粒子 Fock: P(ab) Σ_c Fv[b,c] A[i,j,a,c] ===
    pf = torch.einsum('bc,ijac->ijab', fv, a)
    r2 = r2 + pf - pf.permute(0, 1, 3, 2)

    # === 空穴 Fock: −P(ij) Σ_k Fo[j,k] A[i,k,a,b] ===
    hf = torch.einsum('jk,ikab->ijab', fo, a)
    r2 = r2 - hf + hf.permute(1, 0, 2, 3)

    return r2.contiguous()


def ccsd_doubles_contraction(
    t2: torch.Tensor,
    eri_vvvv: torch.Tensor,
    eri_ovov: torch.Tensor,
    fock_vv: torch.Tensor,
    fock_oo: torch.Tensor,
) -> torch.Tensor:
    """
    CCSD T2 残差主缩并 golden reference（plain golden = bench：fp32 数据通路）

    Args:
        t2: [o, o, v, v] 双激发振幅，算子先做反对称化投影 A
        eri_vvvv: [v, v, v, v] 虚-虚双电子积分 ⟨ab||cd⟩，算子先做反对称化投影 W
        eri_ovov: [o, v, o, v] 占据-虚双电子积分，算子先做 (ia)↔(jb) 对称化投影 X
        fock_vv: [v, v] 虚-虚 Fock 块，算子先对称化 Fv = (F+Fᵀ)/2
        fock_oo: [o, o] 占据-占据 Fock 块，算子先对称化 Fo = (F+Fᵀ)/2

    Returns:
        r2: [o, o, v, v] T2 残差主缩并项，dtype 与 t2 一致；
            满足 r2[i,j,a,b] = −r2[j,i,a,b] = −r2[i,j,b,a]
    """
    r2 = _ccsd_doubles_contraction_core(
        t2, eri_vvvv, eri_ovov, fock_vv, fock_oo, torch.float32)
    return r2.to(t2.dtype)


def ccsd_doubles_contraction_oracle(
    t2: torch.Tensor,
    eri_vvvv: torch.Tensor,
    eri_ovov: torch.Tensor,
    fock_vv: torch.Tensor,
    fock_oo: torch.Tensor,
) -> torch.Tensor:
    """Oracle (g)：dtype-agnostic，计算精度跟随输入（fp64_cpu 下为 fp64 真值），不硬编码 .float()。"""
    return _ccsd_doubles_contraction_core(
        t2, eri_vvvv, eri_ovov, fock_vv, fock_oo, t2.dtype)
