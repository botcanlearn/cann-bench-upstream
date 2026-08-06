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
    """Torch golden for weight_quant_batch_matmul_v2 antiquant matmul path."""
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

    # Compute dtype = x dtype (kernel requires antiquantScale/offset & y dtype == x dtype:
    # docs/aclnnWeightQuantBatchMatmulV2.md scale/offset/y "与x一致").
    compute_dtype = x.dtype if x.dtype in (torch.float16, torch.bfloat16) else torch.float16
    # anti_quant.h AntiQuantCalType: xType=half -> compute in fp16; bf16 -> compute in fp32.
    # (op_kernel/weight_quant_batch_matmul_v2_common.h:84-91,100)
    antiquant_dtype = torch.float16 if compute_dtype == torch.float16 else torch.float32

    # WeightCopyInAndCast: int8 weight -> half (common.h:714); bf16 also -> fp32 (:722).
    w = weight.to(antiquant_dtype)
    # scale/offset participate in antiquant at compute dtype (loaded as xType in common.h:343-344;
    # bf16 upcast to fp32 in BroadCastAntiquantParams common.h:630/664). Round to x dtype first.
    scale = antiquantScale.to(compute_dtype).to(antiquant_dtype)
    offset = antiquantOffset.to(compute_dtype).to(antiquant_dtype)

    if antiquant_group_size == 0:
        if scale.shape != (n,) or offset.shape != (n,):
            raise ValueError("per-channel antiquant expects [N] scale/offset")
        # anti_quant.h AntiQuant: Adds(offset) then Muls(scale) => (w + offset) * scale in compute dtype.
        w_dq = (w + offset.reshape(1, n)) * scale.reshape(1, n)
    else:
        group_num = (k + antiquant_group_size - 1) // antiquant_group_size
        if scale.shape != (group_num, n) or offset.shape != (group_num, n):
            raise ValueError("per-group antiquant expects [ceil(K/group),N] scale/offset")
        chunks = []
        for g, start in enumerate(range(0, k, antiquant_group_size)):
            end = min(start + antiquant_group_size, k)
            chunks.append((w[start:end, :] + offset[g:g + 1, :]) * scale[g:g + 1, :])
        w_dq = torch.cat(chunks, dim=0)

    # w_dq is materialized in x dtype before the cube: fp16 kept (common.h:886);
    # bf16 cast back with CAST_RINT round-to-nearest (common.h:883). This narrow-dtype
    # rounding is the precision the all-fp32 golden missed.
    w_dq = w_dq.to(compute_dtype)

    # bias GM dtype tracks the kernel's biasType template param, not always fp32: DTYPE_BIAS
    # macro in weight_quant_batch_matmul_v2.cpp (:49-54) sets biasType = DTYPE_X (fp16) when
    # x is fp16, else float32; op_host/weight_quant_batch_matmul_v2_def.cpp pairs the bias
    # dtype array 1:1 with the x dtype array the same way (fp16 x -> fp16 bias, bf16 x ->
    # float32 bias, never bf16 bias). biasType is also what selects antiQuantCalType
    # (common.h:100), so it equals antiquant_dtype here. Round bias through that dtype before
    # promoting to fp32 for the cube accumulator, matching scale/offset's existing
    # compute_dtype/antiquant_dtype round-trip -- otherwise fp16-x cases silently keep a
    # higher-precision bias than the real fp16 bias buffer holds.
    bias_cast = bias.to(antiquant_dtype).to(torch.float32)
    # Cube matmul accumulates in fp32; bias added in fp32 (MatmulImpl fp32 accum + SetBias, custom.h:74,305).
    y = x.to(torch.float32) @ w_dq.to(torch.float32) + bias_cast.reshape(1, n)

    # y dtype == x dtype (docs: y "与x一致"); round to x dtype, then present as requested y_dtype.
    y = y.to(compute_dtype)
    out_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}.get(y_dtype, torch.float32)
    return y.to(out_dtype)


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
