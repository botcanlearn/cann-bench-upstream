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
基类层模块

职责：
- 提供所有评测体系的抽象基类
- 定义统一的接口契约
- 无任何特化类依赖

目录设计：
- base/ 目录包含所有基类定义
- 特化类在 benches/ 目录中实现
"""

# === 枚举 ===
from .enums import (
    DifficultyLevel,
    BackendType,
    SourceType,
    GoldenReference,
    EvaluationMode,
)

# === 数据模型基类（统一）===
from .models import (
    AttrSpec,
    TaskSpec,
    CaseSpec,
    InputSpec,
    OutputSpec,
    SolutionSpec,
)

# === 加载器基类（统一）===
from .loaders import (
    TaskLoader,
    CaseLoader,
    OperatorDirMixin,
    GoldenLoaderBase,
)

# === Operator Matcher 基类 ===
from .matcher import OperatorMatcherBase

# === Checker 基类 ===
from .checker import CorrectnessChecker

# === 评测结果基类（统一）===
from .result import (
    AccuracyResult,
    OutputResult,
    PerfResult,
    compute_speedup,
)

# === Scoring 基类 ===
from .scoring import ScoringScheme, CaseScoreInfo

__all__ = [
    # 枚举
    "DifficultyLevel",
    "BackendType",
    "SourceType",
    "GoldenReference",
    "EvaluationMode",
    # 数据模型基类
    "AttrSpec",
    "TaskSpec",
    "CaseSpec",
    "InputSpec",
    "OutputSpec",
    "SolutionSpec",
    # 加载器基类
    "TaskLoader",
    "CaseLoader",
    "OperatorDirMixin",
    "GoldenLoaderBase",
    # Operator Matcher 基类
    "OperatorMatcherBase",
    # Checker 基类
    "CorrectnessChecker",
    # 评测结果基类
    "AccuracyResult",
    "OutputResult",
    "PerfResult",
    "compute_speedup",
    # Scoring 基类
    "ScoringScheme",
    "CaseScoreInfo",
]