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
    scale: torch.Tensor,
    groupList: torch.Tensor,
    perTokenScale: torch.Tensor,
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
    return [x, weight, scale, groupList, perTokenScale]


def grouped_matmul(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    groupList: torch.Tensor,
    perTokenScale: torch.Tensor,
    variant: str = "A8W8O16_pertoken_CV",
    group_list_values=None,
    y_dtype: str = "bfloat16",
    split_item: int = 3,
    group_type: int = 0,
    group_list_type: int = 0,
) -> torch.Tensor:
    """Torch golden for selected grouped_matmul A8W8O16 per-token C->V path."""
    if variant != "A8W8O16_pertoken_CV":
        raise ValueError(f"Unsupported grouped_matmul variant: {variant}")
    if split_item != 3 or group_type != 0 or group_list_type != 0:
        raise ValueError("This benchmark fixes split_item=3, group_type=0, group_list_type=0")
    if x.dim() != 2:
        raise ValueError(f"x expects 2D [M,K], got {list(x.shape)}")
    if weight.dim() != 3:
        raise ValueError(f"weight expects 3D [E,K,N], got {list(weight.shape)}")

    m, k = x.shape
    expert_num, wk, n = weight.shape
    if wk != k:
        raise ValueError(f"weight K ({wk}) must match x K ({k})")
    if scale.shape != (expert_num, n):
        raise ValueError(f"scale expects shape [{expert_num}, {n}], got {list(scale.shape)}")
    if perTokenScale.numel() != m:
        raise ValueError(f"perTokenScale length ({perTokenScale.numel()}) must match M ({m})")

    groups = _cumsum_group_list(groupList, m, expert_num, group_list_values)
    out = torch.zeros(m, n, dtype=torch.float32, device=x.device)

    start = 0
    for expert_id, end in enumerate(groups):
        if end == start:
            continue
        xi = x[start:end].to(torch.float32)
        wi = weight[expert_id].to(torch.float32)
        yi = torch.matmul(xi, wi)
        yi = yi * scale[expert_id].to(torch.float32).reshape(1, n)
        yi = yi * perTokenScale[start:end].to(torch.float32).reshape(-1, 1)
        out[start:end] = yi
        start = end

    return _cast_output(out, y_dtype)


def _cumsum_group_list(groupList, total_m: int, group_num: int, group_list_values):
    if group_list_values is not None:
        values = torch.tensor(group_list_values, dtype=torch.int64)
    else:
        values = groupList.to(torch.int64).flatten()
    if values.numel() != group_num:
        raise ValueError(f"groupList length ({values.numel()}) must match expert count ({group_num})")
    if bool(torch.any(values < 0)):
        raise ValueError("groupList values must be non-negative")
    if bool(torch.any(values[1:] < values[:-1])):
        raise ValueError("groupList must be non-decreasing cumsum")
    if int(values[-1]) != total_m:
        raise ValueError(f"groupList last value ({int(values[-1])}) must equal M ({total_m})")
    return [int(v) for v in values.tolist()]


def _cast_output(out: torch.Tensor, y_dtype: str) -> torch.Tensor:
    name = str(y_dtype).split(".")[-1].lower()
    if name in ("bf16", "bfloat16"):
        return out.to(torch.bfloat16)
    if name in ("fp16", "float16", "half"):
        return out.to(torch.float16)
    raise ValueError(f"Unsupported y_dtype: {y_dtype}")
