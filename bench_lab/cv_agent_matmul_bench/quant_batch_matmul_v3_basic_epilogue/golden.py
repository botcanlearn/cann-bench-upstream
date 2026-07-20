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


def quant_batch_matmul_v3(
    x1: torch.Tensor,
    x2: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor,
    perTokenScale: torch.Tensor = None,
    variant: str = "bf16_basic",
    y_dtype: str = "bfloat16",
) -> torch.Tensor:
    """Torch golden for selected quant_batch_matmul_v3 C->V paths."""
    if variant not in ("bf16_basic", "pertoken_basic"):
        raise ValueError(f"Unsupported quant_batch_matmul_v3 variant: {variant}")
    if x1.dim() != 2 or x2.dim() != 2:
        raise ValueError(f"quant_batch_matmul_v3 expects 2D x1/x2, got {list(x1.shape)} and {list(x2.shape)}")
    m, k = x1.shape
    k2, n = x2.shape
    if k != k2:
        raise ValueError(f"x1 K ({k}) must match x2 K ({k2})")
    if scale.numel() != n:
        raise ValueError(f"scale length ({scale.numel()}) must match N ({n})")
    if bias is None:
        raise ValueError("This benchmark fixes the floating-point bias path and requires bias")
    if bias.numel() != n:
        raise ValueError(f"bias length ({bias.numel()}) must match N ({n})")
    if variant == "bf16_basic" and perTokenScale is not None:
        raise ValueError("bf16_basic does not use perTokenScale in this benchmark")
    if variant == "pertoken_basic":
        if perTokenScale is None:
            raise ValueError("pertoken_basic requires perTokenScale")
        if perTokenScale.numel() != m:
            raise ValueError(f"perTokenScale length ({perTokenScale.numel()}) must match M ({m})")

    out = torch.matmul(x1.to(torch.float32), x2.to(torch.float32))
    out = out * scale.to(torch.float32).reshape(1, n)
    if variant == "pertoken_basic":
        out = out * perTokenScale.to(torch.float32).reshape(m, 1)
    out = out + bias.to(torch.float32).reshape(1, n)
    return _cast_output(out, y_dtype)


def _cast_output(out: torch.Tensor, y_dtype: str) -> torch.Tensor:
    name = str(y_dtype).split(".")[-1].lower()
    if name in ("bf16", "bfloat16"):
        return out.to(torch.bfloat16)
    if name in ("fp16", "float16", "half"):
        return out.to(torch.float16)
    if name in ("fp32", "float32", "float"):
        return out.to(torch.float32)
    raise ValueError(f"Unsupported y_dtype: {y_dtype}")
