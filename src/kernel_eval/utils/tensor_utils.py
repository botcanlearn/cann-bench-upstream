#!/usr/bin/python3
# coding=utf-8

# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""
张量处理工具

职责：
1. 张量精度转换（FP64）
2. 张量设备迁移（CPU/NPU）
3. 批量张量处理
4. 候选输出的内存布局契约校验

统一实现，消除 op_runner.py, accuracy_eval.py, evaluator.py 中的重复代码。
"""

from typing import List, Any, Optional, Union

import torch


def non_contiguous_reason(tensor: torch.Tensor) -> Optional[str]:
    """候选输出内存布局非连续时返回失败原因, 连续则返回 None。

    Why: 只比 shape + 数值的话, 算子可以用 view / transpose / expand 这类纯元数据
    操作包装输入直接返回 -- 数值与 golden 一致, 但没有真正算过。要求候选输出是
    contiguous 才能堵掉这条路。

    调用方须在张量落 CPU **之前**调用: Tensor.to(device) 只对 non-overlapping-
    and-dense 的张量保留 stride, 切片这类稀疏步长会在跨设备拷贝时被悄悄归一化成
    连续布局, 检查放在 .cpu() 之后就漏了。

    Args:
        tensor: 候选 (被测算子) 输出张量

    Returns:
        非连续时返回带 "内存布局非连续" 标记的原因字符串, 连续时返回 None。
        该标记同时登记在 base/result.py 的 STRUCTURAL_FAILURE_MARKERS 中,
        使其被归类为结构性失败而非精度不达标。
    """
    if tensor.is_contiguous():
        return None
    return (
        f"输出内存布局非连续 (non-contiguous): shape={tuple(tensor.shape)}, "
        f"stride={tuple(tensor.stride())}, 算子须返回 contiguous 张量, "
        f"不能用 view / transpose / expand 等元数据操作包装输入返回"
    )


def tensor_to_fp64_cpu(tensor: torch.Tensor) -> torch.Tensor:
    """将张量迁移到 CPU 并转换为 FP64 精度

    只对浮点 tensor 做 upcast，整型/bool 保持原 dtype。
    否则像 GCD（int16/int32 输入）、CrossEntropyLoss（int64 target）这类
    算子的 golden 会在 CPU 上因为 dtype 错误报错。

    Args:
        tensor: 输入张量

    Returns:
        CPU FP64 张量（浮点类型）或原 dtype 张量（整型/bool）
    """
    t = tensor.cpu()
    return t.double() if t.is_floating_point() else t


def tensors_to_cpu(tensors: List[Any]) -> List[Any]:
    """将张量列表迁移到 CPU

    处理三种情况：
    1. 单个 Tensor -> Tensor.cpu()
    2. Tensor 列表/元组 -> 每个 Tensor.cpu()
    3. 其他类型 -> 保持原样

    Args:
        tensors: 输入张量列表（可能包含嵌套结构）

    Returns:
        CPU 张量列表（保持原结构）
    """
    result = []
    for item in tensors:
        if isinstance(item, torch.Tensor):
            result.append(item.cpu())
        elif isinstance(item, (list, tuple)):
            result.append([
                sub.cpu() if isinstance(sub, torch.Tensor) else sub
                for sub in item
            ])
        else:
            result.append(item)
    return result


def tensors_to_fp64_cpu(tensors: List[Any]) -> List[Any]:
    """将张量列表迁移到 CPU 并转换为 FP64 精度

    用于 Golden 参考计算，确保精度高于 NPU 原生 dtype。

    Args:
        tensors: 输入张量列表（可能包含嵌套结构）

    Returns:
        CPU FP64 张量列表（浮点类型），整型/bool 保持原 dtype
    """
    result = []
    for item in tensors:
        if isinstance(item, torch.Tensor):
            result.append(tensor_to_fp64_cpu(item))
        elif isinstance(item, (list, tuple)):
            result.append([
                tensor_to_fp64_cpu(sub) if isinstance(sub, torch.Tensor) else sub
                for sub in item
            ])
        else:
            result.append(item)
    return result


def tensors_to_device(tensors: List[Any], device: str) -> List[Any]:
    """将张量列表迁移到指定设备

    Args:
        tensors: 输入张量列表（可能包含嵌套结构）
        device: 目标设备（如 'cpu', 'npu:0'）

    Returns:
        目标设备上的张量列表（保持原结构）
    """
    result = []
    for item in tensors:
        if isinstance(item, torch.Tensor):
            result.append(item.to(device))
        elif isinstance(item, (list, tuple)):
            result.append([sub.to(device) if isinstance(sub, torch.Tensor) else sub for sub in item])
        else:
            result.append(item)
    return result