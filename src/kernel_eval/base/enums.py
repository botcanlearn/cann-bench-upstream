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
通用枚举定义

Why: 为 TaskSpec/CaseSpec/SolutionSpec 提供统一的枚举类型
"""

from enum import Enum


class InvalidDifficulty(ValueError):
    """proto.yaml 声明的 difficulty 不在 DifficultyLevel 之内"""


class DifficultyLevel(str, Enum):
    """难度级别"""
    L1 = "L1"
    L2 = "L2"
    L3 = "L3"
    L4 = "L4"
    L5 = "L5"

    @classmethod
    def parse(cls, raw) -> "DifficultyLevel":
        """proto.yaml 的 difficulty 取值 -> 枚举成员

        Why: 此处曾是一条与本枚举平行的 if/elif 梯子, 未知取值静默回落 L1.
        于是新增级别时忘了同步梯子, 整级算子会无声无息地串成 L1 -- 级别筛选
        和计分跟着一起错. 派生 + 报错让这种遗漏在第一次加载时就暴露.
        """
        try:
            return cls(str(raw).upper())
        except ValueError:
            declared = ", ".join(level.value for level in cls)
            raise InvalidDifficulty(
                f"未知的 difficulty {raw!r}, DifficultyLevel 声明的取值为: {declared}"
            ) from None


class BackendType(str, Enum):
    """Backend 类型"""
    TORCH = "torch"
    TORCH_NPU = "torch_npu"
    TORCH_CUDA = "torch_cuda"
    TORCH_COMPILE = "torch_compile"
    ASCENDC = "ascendc"
    AICPU = "aicpu"
    TRITON = "triton"
    CUDA = "cuda"
    HIP = "hip"
    PALLAS = "pallas"
    SYCL = "sycl"


class SourceType(str, Enum):
    """源码类型"""
    FILE = "file"
    CODE = "code"
    MODULE = "module"
    GENERATED = "generated"


class GoldenReference(str, Enum):
    """Golden 参考来源"""
    FILE = "file"
    SELF = "self"
    FP64_CPU = "fp64_cpu"
    NONE = "none"


class EvaluationMode(str, Enum):
    """评测模式"""
    ACCURACY = "accuracy"
    PERFORMANCE = "performance"
    FULL = "full"