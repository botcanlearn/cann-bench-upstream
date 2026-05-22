#!/usr/bin/python3
# coding=utf-8

# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You can not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""
Allclose 精度判断器

包含：
- AllcloseOutputResult: 单输出精度结果
- AllcloseChecker: 精度判断器实现

基于 torch.allclose 的简化精度判断，适用于快速验证场景。

Why: 提供轻量级精度判断器，用于简单场景或调试
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import torch

from ..base.checker import CorrectnessChecker
from ..base.result import OutputResult, AccuracyResult
from ..registry.checker_registry import register_correctness_checker


# === OutputResult 子类 ===

@dataclass
class AllcloseOutputResult(OutputResult):
    """Allclose 单输出判断结果

    实现 OutputResult 抽象基类，包含简化指标。
    """
    index: int
    passed: bool = True
    dtype: str = ""
    threshold: float = 0.0          # rtol
    error_msg: str = ""
    # 扩展字段
    atol: float = 0.0               # 绝对容差
    mismatch_count: int = 0         # torch.allclose 不提供详细信息
    total_count: int = 0            # 总数量

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'index': self.index,
            'passed': self.passed,
            'dtype': self.dtype,
            'threshold': self.threshold,
            'error_msg': self.error_msg,
            'atol': self.atol,
            'mismatch_count': self.mismatch_count,
            'total_count': self.total_count,
        }

    def format_summary(self) -> str:
        """格式化摘要"""
        dtype_str = f"{self.dtype}[{self.index}]"
        if self.passed:
            return f"{dtype_str}: ✅ (allclose passed, rtol={self.threshold})"
        else:
            return f"{dtype_str}: ❌ (allclose failed, rtol={self.threshold})"


# === Checker 实现 ===

@register_correctness_checker("allclose")
class AllcloseChecker(CorrectnessChecker):
    """基于 torch.allclose 的精度判断器

    简化的精度判断，适用于:
    - 快速验证场景
    - 调试阶段
    - 不需要详细误差指标的场景
    """

    def get_name(self) -> str:
        return "allclose"

    def get_description(self) -> str:
        return "基于torch.allclose的简化精度判断器，用于快速验证"

    def check(
        self,
        ai_outputs: Union[torch.Tensor, List[torch.Tensor], tuple],
        golden_outputs: Union[torch.Tensor, List[torch.Tensor], tuple],
        dtype: str,
        threshold: float,
        native_outputs: Optional[Union[torch.Tensor, List[torch.Tensor], tuple]] = None,
        ignore_indices: Optional[List[int]] = None,
        custom_thresholds: Optional[Dict[str, float]] = None,
    ) -> AccuracyResult:
        """精度判断（多输出）

        Args:
            ai_outputs: AI算子输出
            golden_outputs: Golden参考输出
            dtype: 数据类型字符串
            threshold: 精度阈值（用作相对容差 rtol）
            native_outputs: 忽略（allclose不使用同精度对照）
            ignore_indices: 需要忽略对比的输出索引列表
            custom_thresholds: 忽略（使用传入的threshold）

        Returns:
            AccuracyResult: 统一格式的精度结果
        """
        ai_list = self._normalize_outputs(ai_outputs)
        golden_list = self._normalize_outputs(golden_outputs)

        error_msg = self._check_output_count(ai_list, golden_list)
        if error_msg:
            return AccuracyResult(
                passed=False,
                threshold=threshold,
                error_msg=error_msg,
            )

        all_passed = True
        output_results: List[AllcloseOutputResult] = []

        for i, (ai_tensor, golden_tensor) in enumerate(zip(ai_list, golden_list)):
            if ignore_indices and i in ignore_indices:
                output_results.append(AllcloseOutputResult(
                    index=i,
                    passed=True,
                    error_msg="(跳过对比)",
                ))
                continue

            ai_cpu = self._ensure_cpu(ai_tensor)
            golden_cpu = self._ensure_cpu(golden_tensor)

            out_dtype = str(ai_cpu.dtype).replace('torch.', '')
            out_threshold = threshold

            if ai_cpu.shape != golden_cpu.shape:
                output_results.append(AllcloseOutputResult(
                    index=i,
                    passed=False,
                    dtype=out_dtype,
                    threshold=out_threshold,
                    error_msg=f"形状不匹配: ai={ai_cpu.shape}, golden={golden_cpu.shape}",
                ))
                all_passed = False
                continue

            try:
                passed = torch.allclose(ai_cpu, golden_cpu, rtol=threshold, atol=0.0, equal_nan=True)
                output_results.append(AllcloseOutputResult(
                    index=i,
                    passed=passed,
                    dtype=out_dtype,
                    threshold=out_threshold,
                    total_count=ai_cpu.numel(),
                ))
                if not passed:
                    all_passed = False
            except Exception as e:
                output_results.append(AllcloseOutputResult(
                    index=i,
                    passed=False,
                    dtype=out_dtype,
                    threshold=out_threshold,
                    error_msg=str(e),
                ))
                all_passed = False

        return AccuracyResult(
            passed=all_passed,
            threshold=threshold,
            output_results=output_results,
        )