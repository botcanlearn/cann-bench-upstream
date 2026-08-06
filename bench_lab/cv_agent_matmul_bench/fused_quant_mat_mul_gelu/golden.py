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
from torch.nn.functional import gelu as torch_gelu


def fused_quant_mat_mul(
    x1: torch.Tensor,
    x2: torch.Tensor,
    x1Scale: torch.Tensor,
    x2Scale: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """FusedQuantMatMul 的 INT8、per-channel、gelu_erf 路径 Golden。

    Args:
        x1: shape 为 [M, K]、dtype 为 torch.int8 的 Tensor。
        x2: shape 为 [K, N]、dtype 为 torch.int8 的 Tensor。
        x1Scale: shape 为 [M]、dtype 为 torch.float32 的 per-token scale。
        x2Scale: shape 为 [N]、dtype 为 torch.bfloat16 的
            per-channel scale。
        bias: shape 为 [N]、dtype 为 torch.bfloat16 的 Tensor。

    Returns:
        shape 为 [M, N]、dtype 为 torch.bfloat16 的 Tensor。

    计算公式:
        qbmmout = (x1 @ x2) * x2Scale * x1Scale + bias
        out = gelu_erf(qbmmout)
    """
    # PyTorch 的 INT8 矩阵乘会返回 INT8，无法表达量化矩阵乘的
    # INT32 累加语义，因此这里将两个输入转换为 INT32。
    qbmmout = torch.matmul(x1.to(torch.int32), x2.to(torch.int32))

    # 将累加结果转换为 FP32，以 FP32 执行后续缩放、bias 相加和
    # gelu_erf。若直接计算 INT32 * BF16，PyTorch 会产生 BF16
    # 中间结果并发生过早舍入。
    qbmmout = qbmmout.to(torch.float32) * x2Scale

    # x1Scale 已经是 FP32；x2Scale 和 bias 虽然是 BF16，但与 FP32
    # 中间结果运算时会按 PyTorch 类型提升规则参与 FP32 计算。
    qbmmout = qbmmout * x1Scale.unsqueeze(-1)
    qbmmout = qbmmout + bias

    out = torch_gelu(qbmmout, approximate="none")

    # aclnnFusedQuantMatmul 这条已选路径的接口输出 dtype 为 BF16。
    return out.to(torch.bfloat16)
