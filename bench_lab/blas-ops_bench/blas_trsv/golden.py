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
Trsv 算子 Torch Golden 参考实现

求解三角线性系统 op(A) * x = b，x 原地覆写 b。

语义与 ops-blas/blas/trsv（arch22）实现对齐：
- A 为行主序 n×lda 矩阵，仅 uplo 指定的三角被访问
  （kernel OP_N 按 A[i][j] 取值，OP_T 按 A[j][i] 取值，见 strsv_kernel.cpp CopyIn）
- trans: N -> op(A)=A；T -> op(A)=A^T；C -> op(A)=A^H（实数下与 T 等价）
- diag: NON_UNIT 对角元从 A 读取；UNIT 对角元固定为 1
- incx != 0，向量元素 i 位于 x[i*incx]（host 侧 gather/scatter，见 strsv_host.cpp）；
  incx < 0 为 BLAS 反向存储约定，等价于从缓冲区末尾按 |incx| 反向取元素
- 输出原地覆写：仅按 incx 步长访问的位置被写为解，其余位置保持输入值
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


def trsv(
    A: torch.Tensor,
    x: torch.Tensor,
    uplo="LOWER",
    trans="N",
    diag="NON_UNIT",
    incx: int = 1,
    lda=None,
    **kwargs,
) -> torch.Tensor:
    """求解 op(A) * sol = b，返回覆写后的 x 缓冲区。

    Args:
        A: n×lda 三角矩阵（行主序），NON_UNIT 时 uplo 三角对角元非零
        x: 右端向量 b 的缓冲区，长度 (n-1)*|incx|+1，按 incx 步长访问
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

    # op(A)：N -> A；T -> A^T；C -> A^H（实数 conj 为恒等）
    # uplo 描述 A 的存储三角；先清掉另一侧垃圾再转置（转置后三角自然翻转）
    M = A[:, :n]
    M = M.triu() if uplo == "UPPER" else M.tril()
    if trans == "T":
        M = M.t()
    elif trans == "C":
        M = M.t().conj()
    # 转置/共轭转置使三角翻转
    is_upper = (uplo == "UPPER") ^ (trans in ("T", "C"))

    # 按 incx 步长取出右端向量
    # incx<0 为 BLAS 反向存储约定: vec[i] = x[(n-1-i)*|incx|]
    # （torch 切片不支持负步长, 用 flip(0)+正步长等价实现）
    if incx > 0:
        vec = x[::incx]
    else:
        vec = x.flip(0)[::(-incx)]
    if vec.numel() != n:
        raise ValueError(f"x 缓冲区长度 {x.numel()} 与 (n-1)*|incx|+1={vec.numel()} 不符")

    # 升精度计算（FP32/CF64 求解以 FP64/CF128 作参考，降低 golden 自身误差）
    compute_dtype = torch.complex128 if A.is_complex() else torch.float64
    sol = torch.linalg.solve_triangular(
        M.to(compute_dtype),
        vec.to(compute_dtype).unsqueeze(-1),
        upper=is_upper,
        unitriangular=(diag == "UNIT"),
        left=True,
    ).squeeze(-1)

    # 原地覆写语义：仅 strided 位置写入解，其余位置保持输入值
    out = x.clone()
    sol = sol.to(x.dtype)
    if incx > 0:
        out[::incx] = sol
    else:
        out = out.flip(0)
        out[::(-incx)] = sol
        out = out.flip(0)
    return out
