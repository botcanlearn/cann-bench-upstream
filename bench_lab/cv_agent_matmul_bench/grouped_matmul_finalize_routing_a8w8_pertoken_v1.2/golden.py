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
    w: torch.Tensor,
    scale: torch.Tensor,
    perTokenScale: torch.Tensor,
    groupList: torch.Tensor,
    logit: torch.Tensor,
    rowIndex: torch.Tensor,
    sharedInput: torch.Tensor,
    **attrs,
) -> list:
    """从 attrs.group_list_values / row_index_values 重建 groupList、rowIndex 张量。

    cases.yaml 将确定性的 cumsum 分组边界放在 group_list_values 属性、行索引放在
    row_index_values 属性（golden 读取它们），但被测 kernel 只看 groupList / rowIndex
    张量。若不由 get_input 重建，这两个张量会被 value_range 随机生成（可能为负、
    非单调），导致 kernel 与 golden 分组/累加目标不一致。返回值同时替换 golden 与
    AI 算子的输入，确保对比公平。
    """
    gl = attrs.get("group_list_values")
    if gl is not None:
        groupList = torch.tensor(list(gl), dtype=torch.uint64, device=x.device)
    ri = attrs.get("row_index_values")
    if ri is not None:
        rowIndex = torch.tensor(list(ri), dtype=torch.uint64, device=x.device)
    return [x, w, scale, perTokenScale, groupList, logit, rowIndex, sharedInput]


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


def grouped_matmul_finalize_routing(
    x: torch.Tensor,
    w: torch.Tensor,
    scale: torch.Tensor,
    perTokenScale: torch.Tensor,
    groupList: torch.Tensor,
    logit: torch.Tensor,
    rowIndex: torch.Tensor,
    sharedInput: torch.Tensor,
    group_list_values=None,
    row_index_values=None,
    groupListType: int = 0,
    output_bs: int = 0,
    shared_input_weight: float = 1.0,
    shared_input_offset: int = 0,
    deterministic: bool = False,
) -> torch.Tensor:
    """Torch golden for grouped_matmul_finalize_routing selected path.

    契约 dtype（与 proto.yaml / cases 对齐，v1.2 适配）：
      x/w: int8; scale/perTokenScale/logit: float32;
      groupList/rowIndex: uint64; sharedInput: bfloat16; out: float32。
    注入参数 group_list_values/row_index_values 重建为 uint64 以贴合契约；
    内部经 _groups()/.item() 取 python int 使用，sharedInput 经 .to(float32) 反量化，dtype 安全。
    """
    if group_list_values is not None:
        groupList = torch.tensor(group_list_values, dtype=torch.uint64, device=x.device)
    if row_index_values is not None:
        rowIndex = torch.tensor(row_index_values, dtype=torch.uint64, device=x.device)
    groups = _groups(groupList, groupListType)
    e, k, n = w.shape
    if len(groups) != e or x.shape[1] != k or scale.shape != (e, n):
        raise ValueError("shape mismatch")
    tmp = torch.zeros(x.shape[0], n, dtype=torch.float32, device=x.device)
    for idx, (start, end) in enumerate(groups):
        if end <= start:
            continue
        mm = x[start:end].to(torch.float32) @ w[idx].to(torch.float32)
        tmp[start:end] = mm * scale[idx].to(torch.float32).reshape(1, n) * perTokenScale[start:end].to(torch.float32).reshape(-1, 1)
    tmp = tmp * logit.to(torch.float32).reshape(-1, 1)
    if output_bs <= 0:
        output_bs = sharedInput.shape[0]
    out = sharedInput.to(torch.float32).clone() * float(shared_input_weight)
    if out.shape != (output_bs, n):
        raise ValueError("sharedInput must be [output_bs,N]")
    for token in range(tmp.shape[0]):
        dst = int(rowIndex[token].item()) + int(shared_input_offset)
        if 0 <= dst < output_bs:
            out[dst] = out[dst] + tmp[token]
    return out
