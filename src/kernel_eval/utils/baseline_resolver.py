#!/usr/bin/python3
# coding=utf-8

# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""
Baseline解析模块

职责：
1. 解析cases.yaml中的baseline_perf_us字段
2. 支持单硬件baseline（scalar形式）
3. 支持多硬件baseline（dict形式）
4. 提供baseline测量和校准功能

cases.yaml格式：
    # 单硬件baseline（默认910b2）
    baseline_perf_us: 40.2

    # 多硬件baseline
    baseline_perf_us:
      910b2: 40.2
      910b1: 45.1
      910a: 50.0
      950: 30.0

参考evaluation/evaluate.py中的resolve_baseline_us函数

硬件配置：
    DEFAULT_HARDWARE 从环境变量 `CANN_BENCH_HARDWARE` 读取，未设置时 fallback 到 "910b2"。
    非默认硬件 + scalar baseline 组合下旧实现会 silent 返回 0.0（F170）；现改为同时
    log WARNING（每个 hardware first-seen 一次），让调用方能观察到 silent failure，
    而 API contract 保持向后兼容（仍返回 float）。
"""

import logging
import math
import os
from typing import Any, Dict, Optional, Set, Union
from dataclasses import dataclass


# 默认硬件——支持环境变量覆盖以避免硬编码 (F170)。
# 未设置时 fallback 到 "910b2"，与历史行为一致。
DEFAULT_HARDWARE: str = os.environ.get("CANN_BENCH_HARDWARE", "910b2")


# 已 warn 过的 hardware 集合，避免 per-case 重复 spam
_WARNED_HARDWARES: Set[str] = set()

_logger = logging.getLogger(__name__)


@dataclass
class BaselineInfo:
    """Baseline信息"""
    yaml_us: float       # YAML配置的baseline
    measured_us: float   # 实测的baseline
    used_us: float       # 实际使用的baseline
    source: str          # baseline来源: "yaml" / "measured"


def has_baseline_for(case_raw: Dict[str, Any], hardware: str = DEFAULT_HARDWARE) -> bool:
    """
    判断指定硬件是否有可用 baseline。

    `resolve_baseline_us` 在缺失情形返回 0.0 维持向后兼容，但 0.0 也是合法值，
    无法仅从返回值反推。需要区分"缺失"和"=0"时使用此函数（如评分系统决定是否
    走 sol-based fallback）。

    Args:
        case_raw: 用例原始数据
        hardware: 目标硬件名称

    Returns:
        True 表示该硬件有 baseline，False 表示需要走 fallback。
    """
    bp = case_raw.get("baseline_perf_us")
    if bp is None or bp == "None":
        return False
    if isinstance(bp, dict):
        v = bp.get(hardware)
        return v is not None and v != "None"
    # scalar 仅匹配默认硬件
    return hardware == DEFAULT_HARDWARE


def resolve_baseline_us(
    case_raw: Dict[str, Any],
    hardware: str = DEFAULT_HARDWARE
) -> float:
    """
    解析baseline_perf_us字段

    Args:
        case_raw: 用例原始数据（从cases.yaml解析）
        hardware: 目标硬件名称

    Returns:
        baseline时间（微秒），无baseline返回0.0（向后兼容；区分"缺失"和"=0"
        请用 has_baseline_for）

    F170: 非默认硬件 + scalar baseline 组合会返回 0.0，且 once-per-hardware
    log WARNING，提示评测在该硬件下走的是 fallback 路径。
    """
    bp = case_raw.get("baseline_perf_us", 0)

    if bp is None or bp == "None":
        return 0.0

    # Dict形式：多硬件baseline
    if isinstance(bp, dict):
        v = bp.get(hardware)
        if v is None or v == "None":
            return 0.0
        try:
            return float(v)
        except (TypeError, ValueError):
            return 0.0

    # Scalar形式：仅对默认硬件有效
    if hardware != DEFAULT_HARDWARE:
        # F170: 不再 silent return 0.0；log 一次让调用方/运维察觉。
        # 同一 hardware 只 warn 一次，避免 per-case spam。
        if hardware not in _WARNED_HARDWARES:
            _WARNED_HARDWARES.add(hardware)
            _logger.warning(
                "baseline_resolver: hardware=%r != DEFAULT_HARDWARE=%r, "
                "scalar baseline 仅对默认硬件有效，将返回 0.0。请将 cases.yaml 的 "
                "baseline_perf_us 改为 dict 形式（如 {%s: <us>, %s: <us>}），"
                "或设置环境变量 CANN_BENCH_HARDWARE=%s。",
                hardware, DEFAULT_HARDWARE, DEFAULT_HARDWARE, hardware, hardware,
            )
        return 0.0

    try:
        return float(bp)
    except (TypeError, ValueError):
        return 0.0


def resolve_baseline_info(
    case_raw: Dict[str, Any],
    hardware: str = DEFAULT_HARDWARE,
    measured_us: Optional[float] = None,
    prefer_measured: bool = False
) -> BaselineInfo:
    """
    解析完整baseline信息

    Args:
        case_raw: 用例原始数据
        hardware: 目标硬件
        measured_us: 实测baseline（可选）
        prefer_measured: 是否优先使用实测值

    Returns:
        BaselineInfo: 完整baseline信息
    """
    yaml_us = resolve_baseline_us(case_raw, hardware)
    measured_us = measured_us or 0.0

    # 确定使用哪个baseline
    if prefer_measured:
        used_us = measured_us or yaml_us
        source = "measured" if measured_us > 0 else "yaml"
    else:
        used_us = yaml_us or measured_us
        source = "yaml" if yaml_us > 0 else "measured"

    return BaselineInfo(
        yaml_us=yaml_us,
        measured_us=measured_us,
        used_us=used_us,
        source=source
    )


def calculate_speedup(
    baseline_us: float,
    custom_us: float
) -> Optional[float]:
    """
    计算加速比

    Args:
        baseline_us: baseline时间
        custom_us: 自定义算子时间

    Returns:
        加速比，无效输入返回None
    """
    if baseline_us <= 0 or custom_us <= 0:
        return None
    return baseline_us / custom_us


def geometric_mean_speedup(speedups: list) -> float:
    """
    计算几何平均加速比

    Args:
        speedups: 加速比列表

    Returns:
        几何平均值
    """
    if not speedups:
        return 0.0
    # 过滤无效值
    valid_speedups = [s for s in speedups if s > 0]
    if not valid_speedups:
        return 0.0
    # 几何平均 = exp(mean(log(x)))
    return math.exp(sum(math.log(max(s, 1e-9)) for s in valid_speedups) / len(valid_speedups))


class BaselineResolver:
    """Baseline解析器"""

    def __init__(self, hardware: str = DEFAULT_HARDWARE):
        self.hardware = hardware
        self.measured_baselines: Dict[str, float] = {}  # (op_name, case_id) -> us

    def resolve(self, case_raw: Dict[str, Any], op_name: str, case_id: int) -> BaselineInfo:
        """
        解析用例baseline

        Args:
            case_raw: 用例原始数据
            op_name: 算子名称
            case_id: 用例ID

        Returns:
            BaselineInfo
        """
        key = f"{op_name}_{case_id}"
        measured_us = self.measured_baselines.get(key)
        return resolve_baseline_info(case_raw, self.hardware, measured_us)

    def record_measured(self, op_name: str, case_id: int, measured_us: float) -> None:
        """
        记录实测baseline

        Args:
            op_name: 算子名称
            case_id: 用例ID
            measured_us: 实测时间
        """
        key = f"{op_name}_{case_id}"
        self.measured_baselines[key] = measured_us

    def get_measured(self, op_name: str, case_id: int) -> Optional[float]:
        """
        获取实测baseline

        Args:
            op_name: 算子名称
            case_id: 用例ID

        Returns:
            实测时间，无记录返回None
        """
        key = f"{op_name}_{case_id}"
        return self.measured_baselines.get(key)