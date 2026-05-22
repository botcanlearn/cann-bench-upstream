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
CANN 默认精度判断器

包含：
- CannOutputResult: 单输出精度结果
- CannDefaultChecker: 精度判断器实现

采用生态算子开源精度标准（MERE/MARE）:
- MERE (平均相对误差) = avg(|actual - golden| / (|golden| + 1e-7))
- MARE (最大相对误差) = max(|actual - golden| / (|golden| + 1e-7))

通过条件: MERE < threshold AND MARE < 10*threshold

特殊场景处理:
- 小值域: 当 |golden| < small_value_threshold 时，采用 ErrorCount 比值标准
- 相消处理: 当 output ≈ 0 且 golden 在精度边界附近时，采用 CPU 同精度对照标准

Why: 提供 cann-bench 默认的精度判断实现
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import torch

from ..base.checker import CorrectnessChecker
from ..base.result import OutputResult, AccuracyResult
from ..utils.compare import compare_tensors


# === OutputResult 子类 ===

@dataclass
class CannOutputResult(OutputResult):
    """CANN 单输出判断结果

    实现 OutputResult 抽象基类，包含完整的精度指标。
    """
    index: int
    passed: bool = True
    dtype: str = ""
    threshold: float = 0.0
    error_msg: str = ""
    # 扩展字段（CANN 特有）
    name: str = ""                  # 输出名称（可选）
    dtype_category: str = ""        # 'float' 或 'int'
    # 浮点类型指标
    mere: float = 0.0               # 平均相对误差
    mare: float = 0.0               # 最大相对误差
    max_diff: float = 0.0
    mean_diff: float = 0.0
    # 整数类型指标
    mismatch_count: int = 0
    total_count: int = 0
    max_abs_diff: int = 0
    # 小值域/相消指标
    small_value_error_count: int = 0
    small_value_cpu_error_count: int = 0
    small_value_total_count: int = 0
    cancel_error_count: int = 0
    cancel_cpu_error_count: int = 0
    cancel_total_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'index': self.index,
            'passed': self.passed,
            'dtype': self.dtype,
            'threshold': self.threshold,
            'error_msg': self.error_msg,
            'name': self.name,
            'dtype_category': self.dtype_category,
            'mere': self.mere,
            'mare': self.mare,
            'max_diff': self.max_diff,
            'mean_diff': self.mean_diff,
            'mismatch_count': self.mismatch_count,
            'total_count': self.total_count,
            'max_abs_diff': self.max_abs_diff,
            'small_value_error_count': self.small_value_error_count,
            'small_value_cpu_error_count': self.small_value_cpu_error_count,
            'small_value_total_count': self.small_value_total_count,
            'cancel_error_count': self.cancel_error_count,
            'cancel_cpu_error_count': self.cancel_cpu_error_count,
            'cancel_total_count': self.cancel_total_count,
        }

    def format_summary(self) -> str:
        """格式化摘要"""
        dtype_str = f"{self.dtype}[{self.name or self.index}]"

        if self.dtype_category == 'int':
            if self.passed:
                return f"{dtype_str}: ✅ (exact match)"
            else:
                ratio = self.mismatch_count / max(self.total_count, 1)
                return f"{dtype_str}: ❌ mismatch={self.mismatch_count}/{self.total_count} ({ratio:.2%}), max_diff={self.max_abs_diff}"
        else:  # float
            if self.passed:
                return f"{dtype_str}: ✅ MERE={self.mere:.6f}, MARE={self.mare:.6f}"
            else:
                mare_threshold = 10 * self.threshold
                return f"{dtype_str}: ❌ MERE={self.mere:.6f}, MARE={self.mare:.6f} (threshold={self.threshold:.6f}, mare_threshold={mare_threshold:.6f})"


def _convert_single_output(sr: Any) -> CannOutputResult:
    """将 precision.py 的 SingleOutputResult 转换为 CannOutputResult"""
    return CannOutputResult(
        index=sr.index,
        passed=sr.passed,
        dtype=sr.dtype,
        threshold=sr.threshold,
        error_msg=sr.error_msg or "",
        name=sr.name if hasattr(sr, 'name') else "",
        dtype_category=sr.dtype_category if hasattr(sr, 'dtype_category') else "",
        mere=sr.mere,
        mare=sr.mare,
        max_diff=sr.max_diff,
        mean_diff=sr.mean_diff,
        mismatch_count=sr.mismatch_count,
        total_count=sr.total_count,
        max_abs_diff=int(sr.max_diff) if hasattr(sr, 'dtype_category') and sr.dtype_category == 'int' else 0,
        small_value_error_count=getattr(sr, 'small_value_error_count', 0),
        small_value_cpu_error_count=getattr(sr, 'small_value_cpu_error_count', 0),
        small_value_total_count=getattr(sr, 'small_value_total_count', 0),
        cancel_error_count=getattr(sr, 'cancel_error_count', 0),
        cancel_cpu_error_count=getattr(sr, 'cancel_cpu_error_count', 0),
        cancel_total_count=getattr(sr, 'cancel_total_count', 0),
    )


# === Checker 实现 ===

# 注册由 benches/cann.py 负责
class CannDefaultChecker(CorrectnessChecker):
    """CANN 默认精度判断器

    封装 compare_tensors，提供完整的精度判断能力:
    - MERE/MARE 标准
    - 小值域处理
    - 相消处理
    - 多输出支持
    """

    def get_name(self) -> str:
        return "cann_default"

    def get_description(self) -> str:
        return "CANN默认精度判断器，采用MERE/MARE标准 + 小值域 + 相消处理"

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
            ai_outputs: AI算子输出（单或多输出）
            golden_outputs: Golden参考输出（FP64精度）
            dtype: 数据类型字符串
            threshold: 精度阈值
            native_outputs: 同精度参考输出（用于小值域比较）
            ignore_indices: 需要忽略对比的输出索引列表
            custom_thresholds: 自定义精度阈值表

        Returns:
            AccuracyResult: 统一格式的精度结果
        """
        compare_result = compare_tensors(
            output=ai_outputs,
            golden=golden_outputs,
            dtype=dtype,
            threshold=threshold,
            native_output=native_outputs,
            ignore_output_indices=ignore_indices,
            custom_thresholds=custom_thresholds,
        )

        # 转换 output_results
        output_results = [_convert_single_output(sr) for sr in compare_result.output_results]

        # 聚合指标存入 metadata
        metadata = {
            'mere': compare_result.mere,
            'mare': compare_result.mare,
            'max_diff': compare_result.max_diff,
            'mean_diff': compare_result.mean_diff,
            'mismatch_count': compare_result.mismatch_count,
            'total_count': compare_result.total_count,
            'mismatch_ratio': compare_result.mismatch_ratio,
            'small_value_error_count': compare_result.small_value_error_count,
            'small_value_cpu_error_count': compare_result.small_value_cpu_error_count,
            'small_value_total_count': compare_result.small_value_total_count,
            'cancel_error_count': compare_result.cancel_error_count,
            'cancel_cpu_error_count': compare_result.cancel_cpu_error_count,
            'cancel_total_count': compare_result.cancel_total_count,
        }

        return AccuracyResult(
            passed=compare_result.passed,
            threshold=compare_result.threshold,
            error_msg=compare_result.error_msg,
            output_results=output_results,
            metadata=metadata,
        )