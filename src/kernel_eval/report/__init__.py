#!/usr/bin/python3
# coding=utf-8

# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You can not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""
报告层模块（通用组件）

职责：
1. 评测报告生成（JSON + Markdown）
2. Summary生成（几何平均加速比）

架构：
- 基类从 base/ 导入
- CANN 特化评分从 benches.cann 导入

使用方式:
    # 通用组件
    from kernel_eval.report import ReportGenerator, EvaluationSummary

    # CANN 评分方案
    from kernel_eval.benches.cann import CannScoringScheme, per_case_sol_score
"""

from .report_generator import ReportGenerator, EvalResult
from .summary_generator import (
    EvaluationSummary, OperatorSummary,
    calculate_geometric_mean, generate_summary, render_summary_markdown, save_summary,
)

# 评分方案抽象（从 base 导入）
from ..base.scoring import ScoringScheme, CaseScoreInfo

# 评分方案注册表（从 registry 导入）
from ..registry.scoring_registry import ScoringSchemeRegistry, get_scoring_scheme

__all__ = [
    # 报告生成
    "ReportGenerator", "EvalResult",
    # Summary
    "EvaluationSummary", "OperatorSummary",
    "calculate_geometric_mean", "generate_summary", "render_summary_markdown", "save_summary",
    # 评分方案抽象
    "ScoringScheme", "CaseScoreInfo",
    "ScoringSchemeRegistry", "get_scoring_scheme",
]