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
精度判断器基类

CorrectnessChecker: 精度判断器抽象基类

Why: 为精度判断提供统一抽象
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import torch

if TYPE_CHECKING:
    from .result import AccuracyResult


class CorrectnessChecker(ABC):
    """精度判断器抽象基类

    设计原则:
    - 多输出为默认（单输出是 len=1 的特化）
    - 判断逻辑与二次验证分离
    - 通过注册机制支持多种判断标准
    """

    @abstractmethod
    def get_name(self) -> str:
        """返回判断器名称"""
        pass

    def get_description(self) -> str:
        """返回判断器描述"""
        return ""

    @abstractmethod
    def check(
        self,
        ai_outputs: Union[torch.Tensor, List[torch.Tensor], Tuple],
        golden_outputs: Union[torch.Tensor, List[torch.Tensor], Tuple],
        dtype: str,
        threshold: float,
        native_outputs: Optional[Union[torch.Tensor, List[torch.Tensor], Tuple]] = None,
        ignore_indices: Optional[List[int]] = None,
        custom_thresholds: Optional[Dict[str, float]] = None,
    ) -> "AccuracyResult":
        """精度判断（多输出）"""
        pass

    # ==================== 辅助方法 ====================

    def _normalize_outputs(
        self,
        outputs: Union[torch.Tensor, List[torch.Tensor], Tuple],
    ) -> List[torch.Tensor]:
        """规范化输出为列表形式"""
        if isinstance(outputs, torch.Tensor):
            return [outputs]
        elif isinstance(outputs, (list, tuple)):
            return list(outputs)
        else:
            raise TypeError(f"outputs must be Tensor or List/Tuple of Tensor, got {type(outputs)}")

    def _ensure_cpu(self, tensor: torch.Tensor) -> torch.Tensor:
        """确保 tensor 在 CPU 上"""
        if isinstance(tensor, torch.Tensor):
            return tensor.cpu() if tensor.device.type != "cpu" else tensor
        return tensor

    def _check_output_count(
        self,
        ai_outputs: List[torch.Tensor],
        golden_outputs: List[torch.Tensor],
        native_outputs: Optional[List[torch.Tensor]] = None,
    ) -> Optional[str]:
        """检查输出数量是否匹配"""
        if len(ai_outputs) != len(golden_outputs):
            return f"输出数量不匹配: ai={len(ai_outputs)}, golden={len(golden_outputs)}"

        if native_outputs is not None and len(native_outputs) != len(golden_outputs):
            return f"同精度输出数量不匹配: native={len(native_outputs)}, golden={len(golden_outputs)}"

        return None