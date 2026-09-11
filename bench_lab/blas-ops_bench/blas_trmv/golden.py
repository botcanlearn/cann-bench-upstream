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
Trmv 算子 Torch Golden 参考实现

计算三角矩阵-向量乘 x := op(A) * x，x 原地覆写。

语义与 ops-blas/blas/trmv（arch22）实现对齐：
- A 为列主序 n×lda 缓冲区：A_bl(i,j) 位于 buf[i + j*lda]
  （kernel OP_N 按 gm_A + row + k*lda 取值，OP_T 转置读取，见 strmv_kernel.cpp；
    对应 torch 行主序 [n, lda] 张量 T：A_bl = T.t()[:n,:n]）
- trans: N -> op(A)=A；T -> op(A)=A^T；C -> op(A)=A^H（实数下与 T 等价）
- diag: NON_UNIT 对角元从 A 读取；UNIT 对角元固定为 1（对角元为乘法因子，
  无除法，存储对角值即使为 0 也不影响合法性）
- incx != 0（ctrmv 仅支持 >0），向量元素 i 位于 x[i*incx]；
  incx < 0 为 BLAS 反向存储约定，等价于从缓冲区末尾按 |incx| 反向取元素
- 输出原地覆写：仅按 incx 步长访问的位置被写为结果，其余位置保持输入值
- n == 0 时空操作直接返回
"""


def _norm_uplo(v):
    m = {"LOWER": "LOWER", "UPPER": "UPPER", 121: "UPPER", 122: "LOWER"}
    if v not in m:
        raise ValueError(f"uplo 必须为 UPPER/LOWER(121/122)，got {v!r}")
    return m[v]


def _norm_trans(v):
    m = {"N": "N", "T": "T", "C": "C", 111: "N", 112: "T", 113: "C"}
    if v not in m:
        raise ValueError(f"trans 必须为 N/T/C(111/112/113)，got {v!r}")
    return m[v]


def _norm_diag(v):
    m = {"NON_UNIT": "NON_UNIT", "UNIT": "UNIT", 131: "NON_UNIT", 132: "UNIT"}
    if v not in m:
        raise ValueError(f"diag 必须为 NON_UNIT/UNIT(131/132)，got {v!r}")
    return m[v]


def trmv(
    A: torch.Tensor,
    x: torch.Tensor,
    uplo="LOWER",
    trans="N",
    diag="NON_UNIT",
    incx: int = 1,
    lda=None,
    **kwargs,
) -> torch.Tensor:
    """计算 op(A) * vec，返回覆写后的 x 缓冲区。

    Args:
        A: n×lda 三角矩阵（列主序），NON_UNIT 时无对角非零约束（乘法因子）
        x: 输入向量缓冲区，长度 (n-1)*|incx|+1，按 incx 步长访问，输出原地覆写
        uplo: UPPER/LOWER，指定 A 的三角部分
        trans: N/T/C，指定 op(A)
        diag: NON_UNIT/UNIT
        incx: x 的存储增量，非零
        lda: A 的前导维（信息性参数，实际以 A.shape[1] 为准）
    """
    incx = int(incx)
    if incx == 0:
        raise ValueError("incx 不能为 0")

    n = int(A.shape[0])
    if lda is not None and int(lda) != int(A.shape[1]):
        raise ValueError(f"lda({lda}) 与 A.shape[1]({A.shape[1]}) 不一致")
    if n == 0:
        return x.clone()

    uplo = _norm_uplo(uplo)
    trans = _norm_trans(trans)
    diag = _norm_diag(diag)

    # 列主序解读: A_bl = T.t()[:n,:n]（T 为 torch 行主序 [n, lda] 张量）
    M = A.t()[:n, :n]
    # 只保留存储三角, 清掉另一侧垃圾（uplo 描述 A_bl 的存储三角）
    M = M.triu() if uplo == "UPPER" else M.tril()
    # op(A): T -> A^T; C -> A^H（实数 conj 为恒等）; 转置后三角翻转
    if trans == "T":
        M = M.t()
    elif trans == "C":
        M = M.t().conj()
    is_upper = (uplo == "UPPER") ^ (trans in ("T", "C"))

    # UNIT 对角: 对角元视为 1（乘法因子, 直接改写对角; 转置不改变对角位置）
    if diag == "UNIT":
        idx = torch.arange(n)
        M[idx, idx] = 1

    # 按 incx 步长取出输入向量（torch 切片不支持负步长, flip 等价实现）
    if incx > 0:
        vec = x[::incx]
    else:
        vec = x.flip(0)[::(-incx)]
    if vec.numel() != n:
        raise ValueError(f"x 缓冲区长度 {x.numel()} 与 (n-1)*|incx|+1 不符")

    # 升精度计算（FP32/CF64 以 FP64/CF128 作参考）
    compute_dtype = torch.complex128 if A.is_complex() else torch.float64
    y = torch.matmul(M.to(compute_dtype), vec.to(compute_dtype))

    # 原地覆写语义：仅 strided 位置写入结果，其余位置保持输入值
    out = x.clone()
    y = y.to(x.dtype)
    if incx > 0:
        out[::incx] = y
    else:
        out = out.flip(0)
        out[::(-incx)] = y
        out = out.flip(0)
    return out
