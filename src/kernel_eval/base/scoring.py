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
评分方案基类

ScoringScheme: 评分方案抽象基类
CaseScoreInfo: 用例级得分信息

Why: 为不同评测体系提供统一的评分接口
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .result import PerfResult


@dataclass
class CaseScoreInfo:
    """用例级得分信息"""
    operator: str = ""
    rel_path: str = ""
    passed: bool = False
    elapsed_us: float = 0.0
    baseline_us: float = 0.0
    score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'operator': self.operator,
            'rel_path': self.rel_path,
            'passed': self.passed,
            'elapsed_us': self.elapsed_us,
            'baseline_us': self.baseline_us,
            'score': self.score,
            'metadata': self.metadata,
        }


class ScoringScheme(ABC):
    """评分方案抽象基类

    定义评分方案的核心接口。
    """

    @abstractmethod
    def prepare_baseline(self, case_spec: Any) -> float:
        """准备基线数据

        Args:
            case_spec: 用例规格

        Returns:
            baseline_us: 基线时间（微秒）
        """
        pass

    @abstractmethod
    def calculate_case_score(self, perf_result: PerfResult, baseline_us: float) -> Optional[float]:
        """计算单个用例的得分

        Args:
            perf_result: 性能评测结果
            baseline_us: 基线时间

        Returns:
            得分（None 表示无法评分）
        """
        pass

    @abstractmethod
    def aggregate_operator_scores(self, case_scores: List[CaseScoreInfo]) -> float:
        """聚合算子的综合得分

        Args:
            case_scores: 各用例得分列表

        Returns:
            算子综合得分
        """
        pass

    @abstractmethod
    def get_scheme_name(self) -> str:
        """获取评分方案名称"""
        pass

    @abstractmethod
    def get_scheme_description(self) -> str:
        """获取评分方案描述"""
        pass