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


def weight_quant_batch_matmul_v2(
    x: torch.Tensor,
    weight: torch.Tensor,
    antiquantScale: torch.Tensor,
    antiquantOffset: torch.Tensor,
    bias: torch.Tensor,
    transpose_x: bool = False,
    transpose_weight: bool = False,
    antiquant_group_size: int = 0,
    output_quant: bool = False,
    y_dtype: str = "float32",
) -> torch.Tensor:
    """Torch golden for weight_quant_batch_matmul_v2 antiquant matmul path.

    同精度参考 (bench b)：int8 权重反量化到输入精度 T=x.dtype（与硬件 A16W8 反量化
    精度一致，保留 int8→fp16/bf16 的舍入），fp32 累加器做 matmul，输出为输出精度。
    fp64 数学真值见 ``weight_quant_batch_matmul_v2_oracle``；拆分约定见
    docs/guide/contributing.md §2.4。
    """
    if output_quant:
        raise ValueError("This benchmark fixes output_quant=False")
    if transpose_x:
        x = x.transpose(-2, -1)
    if transpose_weight:
        weight = weight.transpose(-2, -1)
    m, k = x.shape
    k2, n = weight.shape
    if k != k2 or bias.shape != (n,):
        raise ValueError("shape mismatch")
    T = x.dtype
    w = weight.to(T)
    if antiquant_group_size == 0:
        if antiquantScale.shape != (n,) or antiquantOffset.shape != (n,):
            raise ValueError("per-channel antiquant expects [N] scale/offset")
        w_dq = (w + antiquantOffset.reshape(1, n).to(T)) * antiquantScale.reshape(1, n).to(T)
    else:
        group_num = (k + antiquant_group_size - 1) // antiquant_group_size
        if antiquantScale.shape != (group_num, n) or antiquantOffset.shape != (group_num, n):
            raise ValueError("per-group antiquant expects [ceil(K/group),N] scale/offset")
        chunks = []
        for g, start in enumerate(range(0, k, antiquant_group_size)):
            end = min(start + antiquant_group_size, k)
            chunks.append((w[start:end, :] + antiquantOffset[g:g + 1, :].to(T)) * antiquantScale[g:g + 1, :].to(T))
        w_dq = torch.cat(chunks, dim=0)
    # fp32 累加器（tensor-core 约定）：T 操作数升 fp32 相乘累加，保留已有的 T 舍入
    return x.to(torch.float32) @ w_dq.to(torch.float32) + bias.to(torch.float32).reshape(1, n)


def weight_quant_batch_matmul_v2_oracle(
    x: torch.Tensor,
    weight: torch.Tensor,
    antiquantScale: torch.Tensor,
    antiquantOffset: torch.Tensor,
    bias: torch.Tensor,
    transpose_x: bool = False,
    transpose_weight: bool = False,
    antiquant_group_size: int = 0,
    output_quant: bool = False,
    y_dtype: str = "float32",
) -> torch.Tensor:
    """A16W8 antiquant 的数学真值 (g)，见 docs/guide/contributing.md §2.4。

    与 plain golden 同结构，但反量化 (weight + offset) * scale 与 matmul 全程跟随输入
    精度、不硬编码 .float()/.double() —— 在 golden_precision=fp64_cpu 下 x 升为 fp64，
    整条在 fp64 计算，是精确反量化的 fp64 真值上界（不再被下采成 fp32），使
    |bench − oracle| 不再恒为 0。输出 dtype 跟随 x.dtype。
    """
    if output_quant:
        raise ValueError("This benchmark fixes output_quant=False")
    if transpose_x:
        x = x.transpose(-2, -1)
    if transpose_weight:
        weight = weight.transpose(-2, -1)
    m, k = x.shape
    k2, n = weight.shape
    if k != k2 or bias.shape != (n,):
        raise ValueError("shape mismatch")
    cdt = x.dtype
    w = weight.to(cdt)
    if antiquant_group_size == 0:
        if antiquantScale.shape != (n,) or antiquantOffset.shape != (n,):
            raise ValueError("per-channel antiquant expects [N] scale/offset")
        w_dq = (w + antiquantOffset.reshape(1, n).to(cdt)) * antiquantScale.reshape(1, n).to(cdt)
    else:
        group_num = (k + antiquant_group_size - 1) // antiquant_group_size
        if antiquantScale.shape != (group_num, n) or antiquantOffset.shape != (group_num, n):
            raise ValueError("per-group antiquant expects [ceil(K/group),N] scale/offset")
        chunks = []
        for g, start in enumerate(range(0, k, antiquant_group_size)):
            end = min(start + antiquant_group_size, k)
            chunks.append((w[start:end, :] + antiquantOffset[g:g + 1, :].to(cdt)) * antiquantScale[g:g + 1, :].to(cdt))
        w_dq = torch.cat(chunks, dim=0)
    y = torch.matmul(x, w_dq) + bias.to(cdt).reshape(1, n)
    return y.to(x.dtype)
