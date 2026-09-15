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
Spmv 算子 Torch Golden 参考实现

稀疏矩阵-稠密向量乘（SpMV, Sparse Matrix - Dense Vector Multiplication，CSR 格式）
公式: y_out = alpha * op(A) * x + beta * y
      非转置 op(A)=A:   y_out[i] = alpha * sum_{j in row(i)} A[i,j] * x[j] + beta * y[i]
      转置   op(A)=A^T: y_out[j] = alpha * sum_{i: A[i,j]!=0} A[i,j] * x[i] + beta * y[j]

输出 dtype 与输入 y 一致（FP16/BF16 输入按 computeType=ACL_FLOAT 以 FP32 乘累加后写回）。
"""


def _expand_rows(row_ptr: torch.Tensor, nnz: int) -> torch.Tensor:
    """求每个非零元（按 CSR 线性位置 0..nnz-1）所在的行号。

    row_ptr[1:] 为各行终点，非零元 k 属于第 r 行当且仅当 row_ptr[r] <= k < row_ptr[r+1]，
    即 r = #{ends <= k} = searchsorted(ends, k, right=True)。空行（终点相等）自然被跳过。
    """
    return torch.searchsorted(row_ptr[1:].contiguous(), torch.arange(nnz), right=True)


def spmv(
    csr_row_ptr: torch.Tensor,
    csr_col_ind: torch.Tensor,
    csr_val: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 0.0,
    transpose: bool = False,
) -> torch.Tensor:
    """SpMV 同精度参考（bench, b）——忠实复现硬件累加语义

    aclsparseSpMV 的 computeType 固定 ACL_FLOAT：FP32 输入直接 FP32 乘累加；FP16/BF16
    输入先 Cast 到 FP32 再做乘法，ReduceSum 固定 float 精度，alpha/beta 以 FP32 参与运算，
    结果写回 y 的 dtype。本函数把 csr_val/x/y 升到 FP32 后做 FP32 乘累加（index_add 的
    FP32 累加对应核内 ReduceSum/AtomicAdd 的 FP32 累加器），最后 cast 回 y.dtype，与硬件
    路径的精度约定一致；对 FP32 输入 .to(float32) 为恒等变换。

    公式: y_out = alpha * op(A) * x + beta * y

    Args:
        csr_row_ptr: CSR 行偏移数组，int32，shape [M+1]，单调不减
        csr_col_ind: CSR 列索引数组，int32，shape [nnzA]，行内严格升序且唯一
        csr_val: CSR 非零元值，float16/float32/bfloat16，shape [nnzA]
        x: 输入稠密向量，float16/float32/bfloat16（非转置 [N]，转置 [M]）
        y: 输入稠密向量，float16/float32/bfloat16（非转置 [M]，转置 [N]）
        alpha: 标量系数，乘在 op(A)*x 上
        beta: 标量系数，乘在输入 y 上
        transpose: false=op(A)=A，true=op(A)=Aᵀ

    Returns:
        结果向量 y_out，dtype 与 y 一致，shape 与 y 相同
    """
    row_ptr = csr_row_ptr.to(torch.int64)
    col_idx = csr_col_ind.to(torch.int64)
    vals = csr_val.to(torch.float32)
    nnz = int(csr_val.shape[0])
    acc = torch.zeros(int(y.shape[0]), dtype=torch.float32)
    if transpose:
        # 按行号 gather x、按列号累加：y_out[j] += A[i,j] * x[i]
        prod = vals * x.to(torch.float32)[_expand_rows(row_ptr, nnz)]
        acc.index_add_(0, col_idx, prod)
    else:
        # 按列号 gather x、按行号累加：y_out[i] += A[i,j] * x[j]
        prod = vals * x.to(torch.float32)[col_idx]
        acc.index_add_(0, _expand_rows(row_ptr, nnz), prod)
    out = alpha * acc + beta * y.to(torch.float32)
    return out.to(y.dtype)


def spmv_oracle(
    csr_row_ptr: torch.Tensor,
    csr_col_ind: torch.Tensor,
    csr_val: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 0.0,
    transpose: bool = False,
) -> torch.Tensor:
    """Oracle: SpMV 的数学真值 (g)。dtype-agnostic——不硬编码 .float()（=float32，
    会把 fp64 下采成 fp32），整条 gather/mul/index_add 跟随输入精度；在
    golden_precision=fp64_cpu 下即真 fp64 oracle。

    plain ``spmv`` 的 .to(torch.float32) 是硬件忠实语义（computeType=ACL_FLOAT：FP16/BF16
    操作数升 FP32、FP32 累加、写回 y dtype），本身就是标准同精度参考 (b)——evaluator 的
    bench 路径（``get_bench_function(rel_path) or golden_func``）缺 _bench 时回退
    plain golden 即正确，故本算子默认让 plain golden 兼任 bench，只补 oracle。唯一病理是
    plain 的 .to(float32) 也把 fp64 oracle 封成 fp32，导致 |b−oracle|=0、小值域/相消严格
    分支误杀；修正 oracle 为真 fp64 后，plain-golden bench 的 FP32 累加在相消行（正负
    抵消、行和接近 0）相对 fp64 有非零误差，严格分支按 |b−g| 放松。

    公式: y_out = alpha * op(A) * x + beta * y

    Args:
        csr_row_ptr: CSR 行偏移数组，int32，shape [M+1]，单调不减
        csr_col_ind: CSR 列索引数组，int32，shape [nnzA]，行内严格升序且唯一
        csr_val: CSR 非零元值，float16/float32/bfloat16，shape [nnzA]
        x: 输入稠密向量，float16/float32/bfloat16（非转置 [N]，转置 [M]）
        y: 输入稠密向量，float16/float32/bfloat16（非转置 [M]，转置 [N]）
        alpha: 标量系数，乘在 op(A)*x 上
        beta: 标量系数，乘在输入 y 上
        transpose: false=op(A)=A，true=op(A)=Aᵀ

    Returns:
        结果向量（跟随输入精度，fp64_cpu 下为 fp64），shape 与 y 相同
    """
    row_ptr = csr_row_ptr.to(torch.int64)
    col_idx = csr_col_ind.to(torch.int64)
    nnz = int(csr_val.shape[0])
    acc = torch.zeros(int(y.shape[0]), dtype=csr_val.dtype)
    if transpose:
        prod = csr_val * x[_expand_rows(row_ptr, nnz)]
        acc.index_add_(0, col_idx, prod)
    else:
        prod = csr_val * x[col_idx]
        acc.index_add_(0, _expand_rows(row_ptr, nnz), prod)
    return alpha * acc + beta * y


def get_input(
    csr_row_ptr: torch.Tensor,
    csr_col_ind: torch.Tensor,
    csr_val: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    transpose: bool = False,
    **kwargs,
) -> list:
    """重建 CSR 结构（row_ptr 单调不减、行内列索引严格升序且唯一），使 SpMV 良定义。

    cases.yaml 只能用单区间 value_range 表达索引，通用生成器按 randint 独立采样——
    row_ptr 乱序且不单调、col_ind 行内重复/乱序几乎必然。而 CSR 的良定义性要求
    row_ptr 单调且 csr_row_ptr[M]=nnzA、行内列索引严格升序唯一（canonical CSR，
    kernel 的行分解/二分均依赖此结构），任何真实单算子都无法复现乱序结构下的行为。

    规整策略（保持 nnzA、M、N 与用例声明一致，csr_val/x/y 原样保留）：
    1. nnzA 个非零元用固定种子均匀随机落入 M 行（多项式分布，允许空行），
       cumsum 得 row_ptr——行分布随机、贴近真实稀疏矩阵；
    2. 行内列索引：每个非零元采样 u∈[0,1)，行内按 u 排序后量化到 [0, N-maxc)
       （maxc=最大行长），再加行内秩——量化值行内非降，加严格递增的秩后必严格
       递增，且落在 [0, N-1] 内、行内无重复，分布近似均匀；
    3. 固定种子保证跨 eval 运行可复现（精度二次验证换新值时结构不变）。

    kernel_eval 用输入名 + attrs 作为关键字调用本函数，并用返回值（按 golden 签名的
    Tensor 顺序）同时替换 golden 与候选的输入，故比较公平。

    Returns:
        [csr_row_ptr, csr_col_ind, csr_val, x, y]，顺序与 spmv 签名的 Tensor 参数一致。
    """
    nnz = int(csr_val.shape[0])
    if transpose:
        m_rows, n_cols = int(x.shape[0]), int(y.shape[0])
    else:
        m_rows, n_cols = int(y.shape[0]), int(x.shape[0])
    g = torch.Generator().manual_seed(0)  # 固定种子：跨 eval 运行必须可复现
    if nnz == 0:
        row_ptr = torch.zeros(m_rows + 1, dtype=torch.int32)
        col_ind = torch.empty(0, dtype=torch.int32)
        return [row_ptr, col_ind, csr_val, x, y]

    # 1) 行分布：nnzA 个非零元均匀随机落入 M 行（多项式分布，允许空行）
    row_ids = torch.randint(0, m_rows, (nnz,), generator=g)
    counts = torch.bincount(row_ids, minlength=m_rows)
    maxc = int(counts.max())
    assert maxc <= n_cols, (
        f"最大行长 {maxc} 超过列数 {n_cols}，行内唯一列索引不存在；"
        "请调整用例的 M/N/nnzA（nnzA/M 应显著小于 N）")

    # 2) 行内严格升序唯一列索引：按 (row, u) 双稳定排序后量化 + 行内秩
    u = torch.rand(nnz, generator=g)
    order = torch.argsort(u, stable=True)              # 次关键字 u 升序
    order = order[torch.argsort(row_ids[order], stable=True)]  # 主关键字行号稳定排序
    u_o, rows_o = u[order], row_ids[order]
    starts = counts.cumsum(0) - counts                 # 每行起始位置（排他前缀和）
    rank = torch.arange(nnz) - starts[rows_o]          # 行内秩 0..c_r-1
    col_ind = ((u_o * (n_cols - maxc)).to(torch.int64) + rank).to(torch.int32)

    # 3) row_ptr = [0, cumsum(counts)]，单调不减，末位 = nnzA
    row_ptr = torch.cat([torch.zeros(1, dtype=torch.int64), counts.cumsum(0)]).to(torch.int32)
    return [row_ptr, col_ind, csr_val, x, y]
