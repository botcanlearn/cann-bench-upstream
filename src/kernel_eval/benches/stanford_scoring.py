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
StanfordBench 评分方案

使用加速比作为得分指标：
- score = baseline / elapsed
- 无 SOL-Score（不考虑理论硬件下界）
"""

from typing import Any, List, Optional

from ..base.scoring import ScoringScheme, CaseScoreInfo
from ..base.result import PerfResult


class StanfordScoringScheme(ScoringScheme):
    """StanfordBench 评分方案

    使用加速比作为得分指标：
    - score = baseline / elapsed
    - 无 SOL-Score（不考虑理论硬件下界）
    """

    def get_scheme_name(self) -> str:
        return "stanford"

    def get_scheme_description(self) -> str:
        return "StanfordBench 评分方案：加速比 = baseline / elapsed"

    def prepare_baseline(self, case_spec: Any) -> float:
        """从用例定义获取基线时间"""
        if hasattr(case_spec, 'baseline_perf_us'):
            return float(case_spec.baseline_perf_us)
        if hasattr(case_spec, 'metadata'):
            return float(case_spec.metadata.get('baseline_perf_us', 0.0))
        if isinstance(case_spec, dict):
            return float(case_spec.get('baseline_perf_us', 0.0))
        return 0.0

    def calculate_case_score(self, perf_result: PerfResult, baseline_us: float) -> Optional[float]:
        """计算加速比"""
        elapsed_us = perf_result.elapsed_us
        if baseline_us <= 0 or elapsed_us <= 0:
            return None
        return baseline_us / elapsed_us

    def aggregate_operator_scores(
        self,
        case_scores: List[CaseScoreInfo],
        compile_passed: bool = True,
        total_cases: int = None
    ) -> float:
        """聚合算子得分（使用平均加速比）"""
        passed_cases = [s for s in case_scores if s.passed]
        if not passed_cases:
            return 0.0

        speedups = [s.score for s in passed_cases if s.score is not None]
        if not speedups:
            return 0.0

        avg_speedup = sum(speedups) / len(speedups)
        normalized_score = min(avg_speedup * 10, 100)
        return round(normalized_score, 4)