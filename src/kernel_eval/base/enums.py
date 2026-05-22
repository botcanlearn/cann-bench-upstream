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


class DifficultyLevel(str, Enum):
    """难度级别"""
    L1 = "L1"
    L2 = "L2"
    L3 = "L3"
    L4 = "L4"


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