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
性能指标策略注册表

PerfMetricStrategyRegistry: PerfMetricStrategy 注册表

Why: 支持多评测体系扩展，统一管理性能指标策略
"""

from typing import Dict, List, Optional

from ..base.perf_strategy import PerfMetricStrategy, ProfFileLocations
from .base import BaseRegistry


class PerfMetricStrategyRegistry(BaseRegistry[PerfMetricStrategy]):
    """性能指标策略注册表"""

    _items: Dict[str, PerfMetricStrategy] = {}


def get_perf_metric_strategy(name: str = None) -> PerfMetricStrategy:
    """获取性能指标策略实例

    Args:
        name: 策略名称，默认为 "kernel_details"

    Returns:
        PerfMetricStrategy 实例

    Raises:
        ValueError: 策略未注册
    """
    if name is None:
        name = "kernel_details"
    strategy = PerfMetricStrategyRegistry.get(name)
    if strategy is None:
        registered = PerfMetricStrategyRegistry.list_all()
        raise ValueError(
            f"PerfMetricStrategy '{name}' 未注册，已注册: {registered}"
        )
    return strategy