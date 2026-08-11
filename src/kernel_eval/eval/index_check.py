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

**纯索引算子（ArgMax / ArgMin）**：这类算子只有索引一个输出，没有可配对的值输出，
上面的 `value_output` 形式用不了。改用 `value_reduce` 声明参考值由输入沿 dim 归约
得到——索引只要指向"一个"最大/最小元素即合法，与并列顺序无关::

    outputs:
      - name: indices
        dtype: [int64]
        compare: false
        index_gather:
          input: input
          dim_attr: dim
          value_reduce: amax               # 参考值 = input.amax(dim, keepdim=True)
          keepdim_attr: keepdim            # 可选：候选索引是否保留归约维（缺省 false）

校验式为 ``input.gather(dim, idx) == input.amax(dim, keepdim=True)``，即"索引指向的
元素确实是该切片的最大值"。这是 argmax 除并列顺序外的完整定义：值不对 → gather 出
的元素小于最大值判失败；索引乱填 → 同上；仅并列顺序不同 → 通过（正确接受）。
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


_REDUCE_FUNCS = {"amax": torch.amax, "amin": torch.amin}


def validate_index_reduce_output(
    x: torch.Tensor,
    dim: int,
    idx_candidate: torch.Tensor,
    reduce: str,
    keepdim: bool = False,
) -> Tuple[bool, str]:
    """校验"候选索引指向该切片的最大/最小元素"，用于 ArgMax / ArgMin 这类纯索引算子。

    与 validate_index_output 的区别：没有可配对的值输出，参考值改由输入沿 dim 归约
    得到（x.amax / x.amin）。校验式 x.gather(dim, idx) == x.amax(dim, keepdim=True)，
    与并列元素顺序无关。

    Args:
        x: 源输入张量
        dim: 归约维
        idx_candidate: 候选返回的索引输出
        reduce: 'amax'（argmax）或 'amin'（argmin）
        keepdim: 候选索引是否保留了归约维；False 时先补回该维再 gather

    返回 (ok, msg)。
    """
    if reduce not in _REDUCE_FUNCS:
        return False, f"index_gather: value_reduce={reduce!r} 不支持（可选 amax / amin）"
    if not isinstance(x, torch.Tensor) or not isinstance(idx_candidate, torch.Tensor):
        return False, "index_gather: 输入/索引必须均为 Tensor"

    x_c = x.detach().cpu()
    idx_c = idx_candidate.detach().cpu().to(torch.int64)

    dim_n = dim if dim >= 0 else x_c.dim() + dim
    if not (0 <= dim_n < x_c.dim()):
        return False, f"index_gather: dim={dim} 越界（输入 {x_c.dim()} 维）"

    # keepdim=False 时索引比输入少一维，补回归约维才能 gather
    if not keepdim:
        if idx_c.dim() != x_c.dim() - 1:
            return False, (f"index_gather: keepdim=False 下索引应为 {x_c.dim() - 1} 维，"
                           f"实际 {idx_c.dim()} 维")
        idx_c = idx_c.unsqueeze(dim_n)
    elif idx_c.dim() != x_c.dim():
        return False, (f"index_gather: keepdim=True 下索引应为 {x_c.dim()} 维，"
                       f"实际 {idx_c.dim()} 维")

    ref = _REDUCE_FUNCS[reduce](x_c, dim=dim_n, keepdim=True)
    # 形状必须与归约结果完全一致：value_output 形式靠"索引 shape == 值 shape"锁定
    # 形状，这里没有值输出，改用归约结果兜住，避免候选返回残缺/多余的索引也能过。
    if idx_c.shape != ref.shape:
        return False, (f"index_gather: 索引输出形状 {tuple(idx_candidate.shape)} 与 "
                       f"{reduce} 归约结果形状 "
                       f"{tuple(ref.shape if keepdim else ref.squeeze(dim_n).shape)} 不一致")

    dim_size = x_c.size(dim_n)
    if idx_c.numel() > 0:
        lo = int(idx_c.min().item())
        hi = int(idx_c.max().item())
        if lo < 0 or hi >= dim_size:
            return False, f"index_gather: 索引越界 [{lo},{hi}]，合法区间 [0,{dim_size})"

    gathered = torch.gather(x_c, dim_n, idx_c)
    # NaN-aware：torch 的 amax 对含 NaN 的切片返回 NaN，argmax 也返回该 NaN 的下标，
    # 两侧同为 NaN 应视为一致（否则 value_range=[nan,nan] 的用例即便索引正确也误判）。
    if gathered.is_floating_point():
        eq = (gathered == ref) | (torch.isnan(gathered) & torch.isnan(ref))
    else:
        eq = gathered == ref
    if not bool(eq.all()):
        mism = int((~eq).sum().item())
        return False, (f"index_gather: 候选索引指向的元素不是该切片的 {reduce} "
                       f"（{mism} 处不符），索引无效")
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
        reduce = spec.get("value_reduce")

        x = params.get(src_name) if isinstance(params, dict) else None
        if not isinstance(x, torch.Tensor):
            return False, f"index_gather[{out.name}]: 找不到源输入张量 {src_name!r}"

        if i >= len(ai) or not isinstance(ai[i], torch.Tensor):
            return False, f"index_gather[{out.name}]: 缺少索引输出 #{i}"
        idx = ai[i]

        if val_name is None and reduce is None:
            return False, (f"index_gather[{out.name}]: 须声明 value_output（配对值输出）"
                           f"或 value_reduce（由输入归约得到参考值）之一")
        if val_name is not None and reduce is not None:
            return False, (f"index_gather[{out.name}]: value_output 与 value_reduce "
                           f"互斥，不能同时声明")

        # dim 是索引校验的必要信息：缺失不静默兜底为 -1（会在错误维度上 gather、
        # 给出错误的通过/失败），直接判失败暴露配置问题。
        if not isinstance(case_attrs, dict) or dim_attr not in case_attrs:
            return False, (f"index_gather[{out.name}]: 缺少维度属性 {dim_attr!r}"
                           f"（index_gather 声明的 dim_attr 未在 case attrs 中找到）")
        dim = case_attrs[dim_attr]

        if reduce is not None:
            # 纯索引算子（ArgMax/ArgMin）：无配对值输出，参考值由输入沿 dim 归约得到。
            # keepdim 与 dim 不同——它在 schema 里有明确默认值 false，case 不写即为
            # 默认值，故此处缺省不判失败。
            keepdim_attr = spec.get("keepdim_attr")
            keepdim = bool(case_attrs.get(keepdim_attr, False)) if keepdim_attr else False
            ok, msg = validate_index_reduce_output(x, int(dim), idx, reduce, keepdim)
        else:
            if val_name not in name_to_idx:
                return False, f"index_gather[{out.name}]: value_output {val_name!r} 未在输出中声明"
            vi = name_to_idx[val_name]
            if vi >= len(ai) or not isinstance(ai[vi], torch.Tensor):
                return False, f"index_gather[{out.name}]: 缺少配对值输出 {val_name!r} (#{vi})"
            ok, msg = validate_index_output(x, int(dim), idx, ai[vi])
        if not ok:
            return False, msg

    return True, ""
