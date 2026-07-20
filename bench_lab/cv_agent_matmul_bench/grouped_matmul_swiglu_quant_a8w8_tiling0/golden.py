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


def get_input(
    x: torch.Tensor,
    weight: torch.Tensor,
    weightScale: torch.Tensor,
    xScale: torch.Tensor,
    groupList: torch.Tensor,
    **attrs,
) -> list:
    """从 attrs.group_list_values 重建确定性的 groupList 张量。

    cases.yaml 将 cumsum 分组边界放在 group_list_values 属性里（golden 读取它），
    但被测 kernel 只看 groupList 张量。若不由 get_input 重建，groupList 会被
    value_range 随机生成（可能为负、非单调），导致 kernel 与 golden 分组不一致。
    返回值同时替换 golden 与 AI 算子的输入，确保对比公平。
    """
    gl = attrs.get("group_list_values")
    if gl is not None:
        groupList = torch.tensor(list(gl), dtype=torch.int64, device=x.device)
    return [x, weight, weightScale, xScale, groupList]


def grouped_matmul_swiglu_quant(
    x: torch.Tensor,
    weight: torch.Tensor,
    weightScale: torch.Tensor,
    xScale: torch.Tensor,
    groupList: torch.Tensor,
    variant: str = "A8W8_tiling_key_0",
    group_list_values=None,
    tiling_key: int = 0,
):
    """Torch golden for grouped_matmul_swiglu_quant A8W8 tiling_key=0."""
    if variant != "A8W8_tiling_key_0" or tiling_key != 0:
        raise ValueError("This benchmark fixes grouped_matmul_swiglu_quant A8W8 tiling_key=0")
    if x.dim() != 2:
        raise ValueError(f"x expects 2D [M,K], got {list(x.shape)}")
    if weight.dim() != 3:
        raise ValueError(f"weight expects 3D [E,K,N], got {list(weight.shape)}")

    m, k = x.shape
    expert_num, wk, n = weight.shape
    if wk != k:
        raise ValueError(f"weight K ({wk}) must match x K ({k})")
    if n % 2 != 0:
        raise ValueError("weight last dimension N must be even for SwiGLU split")
    if weightScale.shape != (expert_num, n):
        raise ValueError(f"weightScale expects shape [{expert_num}, {n}], got {list(weightScale.shape)}")
    if xScale.numel() != m:
        raise ValueError(f"xScale length ({xScale.numel()}) must match M ({m})")

    hidden = n // 2
    groups = _cumsum_group_list(groupList, m, expert_num, group_list_values)
    y = torch.zeros(m, hidden, dtype=torch.int8, device=x.device)
    y_scale = torch.zeros(m, dtype=torch.float32, device=x.device)

    start = 0
    for expert_id, end in enumerate(groups):
        if end == start:
            continue
        x_i = x[start:end].to(torch.float32)
        w_i = weight[expert_id].to(torch.float32)
        c = torch.matmul(x_i, w_i)
        c = c * xScale[start:end].to(torch.float32).reshape(-1, 1)
        c = c * weightScale[expert_id].to(torch.float32).reshape(1, n)

        act, gate = c.chunk(2, dim=-1)
        s = (act * torch.sigmoid(act)) * gate

        abs_max = torch.abs(s).amax(dim=-1)
        scale = abs_max / 127.0
        safe_scale = torch.where(scale > 0, scale, torch.ones_like(scale))
        normalized = s / safe_scale.reshape(-1, 1)
        normalized = torch.where(scale.reshape(-1, 1) > 0, normalized, torch.zeros_like(s))
        q = torch.round(normalized).clamp(-128, 127).to(torch.int8)
        y[start:end] = q
        y_scale[start:end] = scale
        start = end

    return y, y_scale


def _cumsum_group_list(group_list, total_m: int, group_num: int, group_list_values):
    if group_list_values is not None:
        values = torch.tensor(group_list_values, dtype=torch.int64)
    else:
        values = group_list.to(torch.int64).flatten()
    if values.numel() != group_num:
        raise ValueError(f"groupList length ({values.numel()}) must match expert count ({group_num})")
    if bool(torch.any(values < 0)):
        raise ValueError("groupList values must be non-negative")
    if bool(torch.any(values[1:] < values[:-1])):
        raise ValueError("groupList must be non-decreasing cumsum")
    if int(values[-1]) != total_m:
        raise ValueError(f"groupList last value ({int(values[-1])}) must equal M ({total_m})")
    return [int(v) for v in values.tolist()]
