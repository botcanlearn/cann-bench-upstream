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

import math
import torch
import torch.nn.functional as F

"""
EngramGateFusion Torch Golden 参考实现

DeepSeek Engram 模块 (arXiv:2601.07372) 的 7 步融合算子：
  1-2. 双路 RMSNorm
  3.   缩放点积门控（FP32 累加）
  4.   sqrt + sigmoid 非线性门控
  5.   门控广播乘法
  6.   ShortConv（RMSNorm → 深度可分离扩张 Conv1d → SiLU），支持 Decode 状态缓存
  7.   残差相加

返回 (output, conv_state_out)。
"""


def engram_gate_fusion(
    keys: torch.Tensor,
    hidden_states: torch.Tensor,
    value: torch.Tensor,
    norm1_weight: torch.Tensor,
    norm2_weight: torch.Tensor,
    conv_norm_weight: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_state: torch.Tensor = None,
    hc_mult: int = 4,
    hidden_size: int = 1024,
    kernel_size: int = 4,
    dilation: int = 3,
    norm_eps: float = 1e-5,
):
    """EngramGateFusion: dual RMSNorm gate + broadcast multiply + ShortConv + residual.

    Args:
        keys:             [B, L, HC, D], bfloat16
        hidden_states:    [B, L, HC, D], bfloat16
        value:            [B, L, D],     bfloat16
        norm1_weight:     [HC, D], float32
        norm2_weight:     [HC, D], float32
        conv_norm_weight: [HC, D], float32
        conv_weight:      [HC*D, 1, K], float32
        conv_state:       [B, HC*D, (K-1)*dilation], bfloat16 or None

    Returns:
        output: [B, L, HC, D]
        conv_state_out: [B, HC*D, (K-1)*dilation]
    """
    B, L, HC, D = keys.shape
    state_len = (kernel_size - 1) * dilation

    assert HC == hc_mult and D == hidden_size
    assert hidden_states.shape == (B, L, HC, D)
    assert value.shape == (B, L, D)
    assert norm1_weight.shape == (HC, D)
    assert norm2_weight.shape == (HC, D)
    assert conv_norm_weight.shape == (HC, D)
    assert conv_weight.shape == (HC * D, 1, kernel_size)
    if conv_state is not None:
        assert conv_state.shape == (B, HC * D, state_len)

    def rms_norm(x, w, eps):
        # x: [B,L,HC,D], w: [HC,D]
        w = w.view(1, 1, HC, D)
        rms = x.float().pow(2).mean(dim=-1, keepdim=True).add(eps).sqrt()
        return (x.float() / rms * w.float()).to(x.dtype)

    # Step 1 & 2: dual RMSNorm (vectorized on HC)
    normed_keys = rms_norm(keys, norm1_weight, norm_eps)
    normed_qs = rms_norm(hidden_states, norm2_weight, norm_eps)

    # Step 3: scaled dot-product gate (FP32 accumulation)
    raw_gate = (normed_keys.float() * normed_qs.float()).sum(dim=-1) / math.sqrt(D)  # [B,L,HC]

    # Step 4: nonlinear + sigmoid gate (FP32), then cast back
    safe_abs = raw_gate.abs().clamp_min(1e-6)
    gate = torch.sigmoid(safe_abs.sqrt() * raw_gate.sign()).unsqueeze(-1)  # [B,L,HC,1]
    gate = gate.to(value.dtype)

    # Step 5: broadcast gating
    value_gated = gate * value.unsqueeze(2)  # [B,L,HC,D]

    # Step 6: ShortConv
    normed_vg = rms_norm(value_gated, conv_norm_weight, norm_eps)
    x = normed_vg.permute(0, 2, 3, 1).reshape(B, HC * D, L)

    if conv_state is None:
        x_cat = F.pad(x, (state_len, 0))
    else:
        x_cat = torch.cat([conv_state.to(x.dtype), x], dim=-1)

    conv_state_out = (x_cat[:, :, -state_len:].contiguous()
                      if state_len > 0 else x_cat[:, :, :0])

    y = F.conv1d(
        x_cat,
        conv_weight.to(x_cat.dtype),
        dilation=dilation,
        groups=HC * D,
    )
    y = F.silu(y)
    conv_out = y.reshape(B, HC, D, L).permute(0, 3, 1, 2)  # [B,L,HC,D]

    # Step 7: residual
    output = value_gated + conv_out
    return output, conv_state_out
