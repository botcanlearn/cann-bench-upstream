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
Spvv 算子 Torch Golden 参考实现

稀疏向量-稠密向量点积（SPVV, Sparse vector - dense Vector dot Product）
公式: result = sum_{i=0}^{nnz-1} values[i] * y[indices[i]]

输出恒为 shape [1] 的 float32 标量（与输入 dtype 无关）。
"""


def spvv(
    indices: torch.Tensor,
    values: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """SPVV 同精度参考（bench, b）——忠实复现硬件累加语义

    aclsparseSpvv 的 computeType 固定 ACL_FLOAT：FP32 输入直接做 FP32 乘累加；
    FP16 输入先 Cast 到 FP32 再做乘法与累加（FP32 乘积），输出恒为 FP32。
    本函数把 values/y 升到 FP32 后做 FP32 乘累加、FP32 输出，与两条硬件路径
    的精度约定一致；对 FP32 输入 .to(float32) 为恒等变换。

    公式: result = sum_{i=0}^{nnz-1} values[i] * y[indices[i]]

    Args:
        indices: 稀疏向量非零元索引，int32，严格升序且唯一，取值 [0, yLen-1]
        values: 稀疏向量非零元值，float16/float32，shape [nnz]
        y: 稠密向量，float16/float32，shape [yLen]

    Returns:
        点积结果，float32，shape [1]
    """
    idx = indices.to(torch.int64)
    prod = values.to(torch.float32) * y.to(torch.float32)[idx]
    return torch.sum(prod, dim=0, keepdim=True)


def spvv_oracle(
    indices: torch.Tensor,
    values: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """Oracle: SPVV 的数学真值 (g)。dtype-agnostic——不硬编码 .float()（=float32，
    会把 fp64 下采成 fp32），整条 gather/mul/sum 跟随输入精度；在 golden_precision=fp64_cpu
    下即真 fp64 oracle。

    plain ``spvv`` 的 .to(torch.float32) 是硬件忠实语义（computeType=ACL_FLOAT：FP16
    操作数升 FP32、FP32 累加），本身就是标准同精度参考 (b)——evaluator 的 bench 路径
    （``get_bench_function(rel_path) or golden_func``）缺 _bench 时回退 plain golden 即
    正确，故本算子默认让 plain golden 兼任 bench，只补 oracle。唯一病理是 plain 的
    .to(float32) 也把 fp64 oracle 封成 fp32，导致 |b−oracle|=0、小值域/相消严格分支
    误杀；修正 oracle 为真 fp64 后，plain-golden bench 的 FP32 累加在相消（正负抵消、
    结果接近 0）时相对 fp64 有非零误差，严格分支按 |b−g| 放松。

    公式: result = sum_{i=0}^{nnz-1} values[i] * y[indices[i]]

    Args:
        indices: 稀疏向量非零元索引，int32，严格升序且唯一，取值 [0, yLen-1]
        values: 稀疏向量非零元值，float16/float32，shape [nnz]
        y: 稠密向量，float16/float32，shape [yLen]

    Returns:
        点积结果（跟随输入精度，fp64_cpu 下为 fp64），shape [1]
    """
    idx = indices.to(torch.int64)
    prod = values * y[idx]
    return torch.sum(prod, dim=0, keepdim=True)


def get_input(
    indices: torch.Tensor,
    values: torch.Tensor,
    y: torch.Tensor,
    **kwargs,
) -> list:
    """重建 indices 为 [0, yLen) 内严格升序且唯一的随机子集，使 SPVV 良定义。

    cases.yaml 只能用单区间 value_range 表达索引，通用生成器按 randint(0, yLen-1)
    独立采样——重复与乱序几乎必然。而 aclsparseSpvv 语义要求 indices 严格升序且唯一
    （kernel 依赖有序性做二分查找 segment 边界），任何真实单算子都无法复现无序/重复
    索引下的结果。

    这里按用例声明的 nnz（= indices.shape[0]）用固定种子从 [0, yLen) 随机置换取前
    min(nnz, yLen) 项、升序排列——精确保持 nnz（去重式规整会随机缩水，密集用例
    nnz=yLen 时尤甚），且跨 eval 运行可复现。values/y 原样返回。

    kernel_eval 用输入名 + attrs 作为关键字调用本函数，并用返回值（按 golden 签名的
    Tensor 顺序）同时替换 golden 与候选的输入，故比较公平。

    Returns:
        [indices, values, y]，顺序与 spvv 签名的 Tensor 参数一致。
    """
    nnz = int(indices.shape[0])
    y_len = int(y.shape[0])
    n = min(nnz, y_len)
    g = torch.Generator().manual_seed(0)  # 固定种子：跨 eval 运行必须可复现
    picked = torch.randperm(y_len, generator=g)[:n]
    new_indices = torch.sort(picked).values.to(torch.int32)
    return [new_indices, values, y]
