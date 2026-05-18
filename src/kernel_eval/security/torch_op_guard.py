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

"""Torch op call guard for AI operator execution.

防作弊：检测 AI 算子在执行时是否直接调用了 PyTorch 的内置数学 API
（torch.matmul / torch.nn.functional.conv* / torch.nn.functional.linear /
softmax / attention 等）。AI 算子应当通过编译好的 AscendC kernel 完成计算，
直接调用 torch 内置算子相当于把工作甩给 PyTorch / torch_npu 内核——评测无效。

使用方式：

    from .torch_op_guard import TorchOpGuard

    with TorchOpGuard(forbidden=BUILTIN_COMPUTE_OPS, mode="warn") as g:
        outputs = ai_op_func(**params)
    if g.forbidden_calls:
        # g.forbidden_calls: List[str]
        ...

mode='warn'：检测到禁止 API 时打印 [WARN]，不阻断执行（默认，方便排查）
mode='block'：检测到禁止 API 时抛 RuntimeError（生产 / 防作弊评测使用）
"""
from __future__ import annotations

from typing import List, Optional, Set


# 计算密集型 builtin ops 的默认禁用集合——AI kernel 不应当直接调它们。
# 元数据操作（reshape / view / transpose / contiguous / to / 张量创建）
# 不在此列，AI kernel 的 Python wrapper 经常需要它们做参数预处理。
BUILTIN_COMPUTE_OPS: Set[str] = {
    "torch.matmul",
    "torch.mm",
    "torch.bmm",
    "torch.einsum",
    "torch.nn.functional.linear",
    "torch.nn.functional.conv1d",
    "torch.nn.functional.conv2d",
    "torch.nn.functional.conv3d",
    "torch.nn.functional.conv_transpose1d",
    "torch.nn.functional.conv_transpose2d",
    "torch.nn.functional.conv_transpose3d",
    "torch.nn.functional.softmax",
    "torch.nn.functional.log_softmax",
    "torch.nn.functional.scaled_dot_product_attention",
    "torch.nn.functional.silu",
    "torch.nn.functional.gelu",
    "torch.nn.functional.relu",
    "torch.nn.functional.layer_norm",
    "torch.nn.functional.rms_norm",
    "torch.nn.functional.batch_norm",
}


def _qualified_name(func) -> str:
    """Best-effort 'torch.xxx.fn' 名提取，给一个稳定的字符串用来匹配 forbidden set。"""
    mod = getattr(func, "__module__", "") or ""
    name = getattr(func, "__name__", "") or repr(func)
    if mod.startswith("torch._C._TensorBase"):
        return f"torch.Tensor.{name}"
    return f"{mod}.{name}" if mod else name


class TorchOpGuard:
    """ContextManager：在 with 块内监听 torch.* 调用。

    用 ``torch.overrides.TorchFunctionMode``（PyTorch ≥1.11）。如果当前
    PyTorch 不支持，构造时打印 [WARN] 并降级为无操作 context。
    """

    def __init__(self, forbidden: Optional[Set[str]] = None, mode: str = "warn"):
        self.forbidden = forbidden if forbidden is not None else BUILTIN_COMPUTE_OPS
        if mode not in ("warn", "block", "off"):
            raise ValueError(f"mode must be one of warn/block/off, got {mode!r}")
        self.mode = mode
        self.forbidden_calls: List[str] = []
        self._inner = None
        self._available = mode != "off"

    def __enter__(self):
        if not self._available:
            return self
        try:
            import torch.overrides as _ov
        except ImportError:
            print("[WARN] TorchOpGuard: torch.overrides 不可用（PyTorch <1.11），跳过守卫。", flush=True)
            self._available = False
            return self

        outer = self

        class _Mode(_ov.TorchFunctionMode):
            def __torch_function__(self, func, types, args=(), kwargs=None):
                kwargs = kwargs or {}
                fq = _qualified_name(func)
                if fq in outer.forbidden:
                    outer.forbidden_calls.append(fq)
                    if outer.mode == "block":
                        raise RuntimeError(
                            f"[SECURITY] AI 算子调用了被禁用的 PyTorch 内置 API: {fq}。"
                            f"AscendC kernel 应通过编译好的算子完成计算，不应直接调用 torch.matmul 等。"
                        )
                return func(*args, **kwargs)

        self._inner = _Mode()
        self._inner.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._inner is not None:
            self._inner.__exit__(exc_type, exc_val, exc_tb)
            if self.mode == "warn" and self.forbidden_calls:
                uniq = sorted(set(self.forbidden_calls))
                print(
                    f"[WARN] TorchOpGuard: AI 算子调用了 {len(self.forbidden_calls)} 次"
                    f" 禁用 API（去重 {len(uniq)} 种）: {', '.join(uniq[:5])}"
                    f"{' ...' if len(uniq) > 5 else ''}",
                    flush=True,
                )
        return False
