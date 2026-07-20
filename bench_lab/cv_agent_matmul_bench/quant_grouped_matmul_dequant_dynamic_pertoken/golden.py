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
    quantized_weight: torch.Tensor,
    weightScale: torch.Tensor,
    groupList: torch.Tensor,
    bias: torch.Tensor,
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
    return [x, quantized_weight, weightScale, groupList, bias]


def _groups(groupList: torch.Tensor, groupListType: int):
    values = [int(v) for v in groupList.detach().cpu().tolist()]
    if groupListType == 0:
        starts = [0] + values[:-1]
        ends = values
    elif groupListType == 1:
        starts, ends, cur = [], [], 0
        for count in values:
            starts.append(cur)
            cur += count
            ends.append(cur)
    else:
        raise ValueError("groupListType must be 0 or 1")
    return list(zip(starts, ends))


def quant_grouped_matmul_dequant(
    x: torch.Tensor,
    quantized_weight: torch.Tensor,
    weightScale: torch.Tensor,
    groupList: torch.Tensor,
    bias: torch.Tensor,
    quant_mode: str = "pertoken",
    xScaleOptional=None,
    transposeWeight: bool = False,
    groupListType: int = 0,
    group_list_values=None,
) -> torch.Tensor:
    """Torch golden for quant_grouped_matmul_dequant dynamic per-token path."""
    if quant_mode != "pertoken" or xScaleOptional is not None:
        raise ValueError("This benchmark fixes dynamic per-token path with xScaleOptional=None")
    if group_list_values is not None:
        groupList = torch.tensor(group_list_values, dtype=torch.int64, device=x.device)
    if transposeWeight:
        quantized_weight = quantized_weight.transpose(-2, -1)
    groups = _groups(groupList, groupListType)
    g, k, n = quantized_weight.shape
    if len(groups) != g or x.shape[1] != k:
        raise ValueError("shape mismatch")
    eps = torch.finfo(torch.float32).tiny
    x_scale = x.to(torch.float32).abs().amax(dim=1).clamp_min(eps) / 127.0
    xq = torch.round(x.to(torch.float32) / x_scale.reshape(-1, 1)).clamp(-127, 127)
    out = torch.zeros(x.shape[0], n, dtype=torch.float32, device=x.device)
    for idx, (start, end) in enumerate(groups):
        if end <= start:
            continue
        mm = xq[start:end, :] @ quantized_weight[idx].to(torch.float32)
        out[start:end, :] = mm * weightScale[idx].to(torch.float32).reshape(1, n) * x_scale[start:end].reshape(-1, 1)
        out[start:end, :] = out[start:end, :] + bias[idx].to(torch.float32).reshape(1, n)
    return out
