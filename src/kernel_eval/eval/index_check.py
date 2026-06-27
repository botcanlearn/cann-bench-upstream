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

"""索引类输出（TopK / ArgSort / ArgMax 等）的"指向值"校验。

背景（issue #40）：TopK 的 golden 用 torch.topk，相等元素的索引顺序非确定，
NPU 候选算子对并列元素可能选不同下标，导致索引输出**逐元素**比对必然不过。
此前的做法是把索引输出标 `compare: false` 直接跳过——但这等于放弃验证索引，
留下"返回正确数值却乱填索引也能过"的防作弊缺口。

本模块提供 tie 顺序无关的索引校验：对在 proto.yaml 中声明了 `index_gather` 的
输出，校验"候选返回的索引确实指向候选返回的值"——
    x.gather(dim, idx_candidate) == values_candidate
配合框架已有的"值输出 vs golden"比对（候选值 ≈ golden 值），即可在不依赖并列
顺序的前提下完整验证索引正确性：
- 值不对   → 值输出比对（关系 golden）判失败；
- 索引乱填 → 本校验（gather 自洽）判失败；
- 仅并列顺序不同 → 两者都通过（正确接受）。

proto.yaml 声明示例（TopK 的 idx 输出）::

    outputs:
      - name: y            # 值输出（正常参与比对）
        ...
      - name: idx
        dtype: [int64]
        compare: false                     # 不做逐元素比对
        index_gather:
          input: x                         # 从哪个输入张量按索引取值
          dim_attr: dim                    # 取 dim 的 attr 名（值见 case attrs）
          value_output: y                  # 与哪个值输出做自洽校验
"""

from typing import Any, Dict, List, Optional, Tuple

import torch


def _as_list(outputs: Any) -> List[Any]:
    if isinstance(outputs, (list, tuple)):
        return list(outputs)
    return [outputs]


def validate_index_output(
    x: torch.Tensor,
    dim: int,
    idx_candidate: torch.Tensor,
    values_candidate: torch.Tensor,
) -> Tuple[bool, str]:
    """校验"候选索引指向候选值"：x.gather(dim, idx) 须逐元素等于 values。

    与并列元素顺序无关：只要索引指向的元素值正确即通过。
    返回 (ok, msg)。
    """
    if not isinstance(x, torch.Tensor) or not isinstance(idx_candidate, torch.Tensor) \
            or not isinstance(values_candidate, torch.Tensor):
        return False, "index_gather: 输入/索引/值必须均为 Tensor"

    x_c = x.detach().cpu()
    idx_c = idx_candidate.detach().cpu().to(torch.int64)
    val_c = values_candidate.detach().cpu()

    if idx_c.dim() != x_c.dim():
        return False, (f"index_gather: 索引维度 {idx_c.dim()} 与输入维度 {x_c.dim()} 不一致，"
                       f"无法 gather")
    dim_n = dim if dim >= 0 else x_c.dim() + dim
    if not (0 <= dim_n < x_c.dim()):
        return False, f"index_gather: dim={dim} 越界（输入 {x_c.dim()} 维）"

    if idx_c.shape != val_c.shape:
        return False, (f"index_gather: 索引输出形状 {tuple(idx_c.shape)} 与值输出形状 "
                       f"{tuple(val_c.shape)} 不一致")

    # gather 要求除 dim 外各维 idx.size(d) <= x.size(d)
    for d in range(x_c.dim()):
        if d != dim_n and idx_c.size(d) > x_c.size(d):
            return False, (f"index_gather: 第 {d} 维索引尺寸 {idx_c.size(d)} 超过输入 "
                           f"{x_c.size(d)}")

    dim_size = x_c.size(dim_n)
    if idx_c.numel() > 0:
        lo = int(idx_c.min().item())
        hi = int(idx_c.max().item())
        if lo < 0 or hi >= dim_size:
            return False, f"index_gather: 索引越界 [{lo},{hi}]，合法区间 [0,{dim_size})"

    gathered = torch.gather(x_c, dim_n, idx_c).to(val_c.dtype)
    # 索引取出的元素必须与候选自报的值严格一致（topk 值即被选元素本身，应逐位相等）。
    # 注意：不能直接用 torch.equal —— 它对 NaN 返回不等（NaN != NaN），会把
    # value_range=[nan,nan] 这类全 NaN 用例（如 top_k case 15）即便索引完全正确也误判失败。
    # 故按位置做 NaN-aware 比较：两侧同为 NaN 视为相等，否则按值相等。
    if gathered.is_floating_point() or val_c.is_floating_point():
        both_nan = torch.isnan(gathered) & torch.isnan(val_c)
        eq = (gathered == val_c) | both_nan
    else:
        eq = gathered == val_c
    if not bool(eq.all()):
        mism = int((~eq).sum().item())
        return False, (f"index_gather: 候选索引指向的元素与其值输出不一致（{mism} 处不符），"
                       f"索引无效")
    return True, ""


def validate_index_gather_outputs(
    op_info: Any,
    params: Dict[str, Any],
    case_attrs: Dict[str, Any],
    ai_outputs: Any,
) -> Tuple[bool, str]:
    """对算子中所有声明了 index_gather 的输出执行索引校验。

    无此类声明时立即返回 (True, "")，对其他算子零影响。

    Args:
        op_info: 算子规格（需有 .outputs，每个 output 可能带 index_gather）
        params:  按参数名索引的调用入参（含输入张量，如 params['x']）
        case_attrs: 本用例的 attrs（含 dim 等）
        ai_outputs: 候选算子的输出（单个或列表）
    """
    outputs = getattr(op_info, "outputs", None) or []
    name_to_idx = {o.name: i for i, o in enumerate(outputs)}
    ai = _as_list(ai_outputs)

    for i, out in enumerate(outputs):
        spec: Optional[Dict[str, Any]] = getattr(out, "index_gather", None)
        if not spec:
            continue

        src_name = spec.get("input")
        dim_attr = spec.get("dim_attr", "dim")
        val_name = spec.get("value_output")

        x = params.get(src_name) if isinstance(params, dict) else None
        if not isinstance(x, torch.Tensor):
            return False, f"index_gather[{out.name}]: 找不到源输入张量 {src_name!r}"

        if i >= len(ai) or not isinstance(ai[i], torch.Tensor):
            return False, f"index_gather[{out.name}]: 缺少索引输出 #{i}"
        idx = ai[i]

        if val_name is None or val_name not in name_to_idx:
            return False, f"index_gather[{out.name}]: value_output {val_name!r} 未在输出中声明"
        vi = name_to_idx[val_name]
        if vi >= len(ai) or not isinstance(ai[vi], torch.Tensor):
            return False, f"index_gather[{out.name}]: 缺少配对值输出 {val_name!r} (#{vi})"
        vals = ai[vi]

        # dim 是索引校验的必要信息：缺失不静默兜底为 -1（会在错误维度上 gather、
        # 给出错误的通过/失败），直接判失败暴露配置问题。
        if not isinstance(case_attrs, dict) or dim_attr not in case_attrs:
            return False, (f"index_gather[{out.name}]: 缺少维度属性 {dim_attr!r}"
                           f"（index_gather 声明的 dim_attr 未在 case attrs 中找到）")
        dim = case_attrs[dim_attr]
        ok, msg = validate_index_output(x, int(dim), idx, vals)
        if not ok:
            return False, msg

    return True, ""
