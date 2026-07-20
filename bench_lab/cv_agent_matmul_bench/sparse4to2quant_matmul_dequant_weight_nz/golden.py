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


# CANN dtype 枚举 -> torch dtype（仅覆盖本 benchmark 路径涉及的输出类型）
_DTYPE_ENUM = {
    0: torch.float32,
    1: torch.float16,
    27: torch.bfloat16,
}


def sparse4to2quant_matmul_dequant(
    x: torch.Tensor,
    weight: torch.Tensor,
    xScale: torch.Tensor,
    sparseWeightScale: torch.Tensor,
    bias: torch.Tensor = None,
    dtype: int = 27,
    with_bias: bool = True,
):
    """Torch golden for aclnnSparse4to2QuantMatmulWeightNz.

    Notes:
      - ``weight`` is the 4:2-sparsified DENSE representation (every 4 consecutive
        elements have exactly 2 zeros). NPU compresses it via
        ``aclnnTransSparse4to2Para`` into ``sparseWeight`` + ``index``; the golden
        uses the dense form directly because zero elements contribute nothing to
        the matmul (mathematically equivalent).
      - This benchmark fixes the BF16 output + FP32 per-token/per-channel scale +
        optional BF16 bias path; ``dtype`` 驱动输出 dtype（27=BF16）。
    """
    out_dtype = _DTYPE_ENUM.get(int(dtype), torch.bfloat16)
    if x.dim() != 2:
        raise ValueError(f"x expects 2D [M, K], got {list(x.shape)}")
    if weight.dim() != 2:
        raise ValueError(f"weight expects 2D [N, K], got {list(weight.shape)}")

    m, k = x.shape
    n, wk = weight.shape
    if wk != k:
        raise ValueError(f"x.K ({k}) must match weight.K ({wk})")
    if k > 65535:
        raise ValueError(f"K ({k}) exceeds 65535")
    # K and N do NOT need to be aligned: NPU pads K via CeilAlign(K, 8) and pads N
    # via FRACTAL_NZ ceil(N/16). Golden uses dense weight; padding bytes on the
    # NPU side are zero-filled and do not pollute the logical [M, N] output.
    # (Non-aligned cases are valid; current cases.yaml only exercises aligned shapes.)
    if xScale.numel() != m:
        raise ValueError(f"xScale length ({xScale.numel()}) must match M ({m})")
    if sparseWeightScale.numel() != n:
        raise ValueError(f"sparseWeightScale length ({sparseWeightScale.numel()}) must match N ({n})")
    if with_bias:
        if bias is None:
            raise ValueError("with_bias=True but bias tensor is None")
        if bias.numel() != n:
            raise ValueError(f"bias length ({bias.numel()}) must match N ({n})")

    # Verify 4:2 sparsity pattern (every 4 consecutive elements have exactly 2 zeros).
    # Reshape weight to [N, K/4, 4] and count zeros per group.
    weight_view = weight.reshape(n, k // 4, 4)
    zeros_per_group = (weight_view == 0).sum(dim=-1)
    if not bool((zeros_per_group == 2).all()):
        raise ValueError(
            "weight does not satisfy 4:2 sparsity pattern "
            "(every 4 consecutive elements must have exactly 2 zeros). "
            "Check data preparation step."
        )

    out_fp32 = torch.matmul(x.to(torch.float32), weight.t().to(torch.float32))   # [M, N]
    out_fp32 = out_fp32 * xScale.to(torch.float32).reshape(-1, 1)
    out_fp32 = out_fp32 * sparseWeightScale.to(torch.float32).reshape(1, -1)
    if with_bias:
        out_fp32 = out_fp32 + bias.to(torch.float32).reshape(1, -1)
    return out_fp32.to(out_dtype)
