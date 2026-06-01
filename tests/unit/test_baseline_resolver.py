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

baseline 解析模块单元测试

测试对象：kernel_eval.utils.baseline_resolver
核心功能：
1. resolve_baseline_us - baseline 时间解析
2. resolve_baseline_info - baseline 信息解析
3. calculate_speedup - 加速比计算
4. geometric_mean_speedup - 几何平均加速比
5. BaselineResolver 类
"""

import pytest
import math

from kernel_eval.utils.baseline_resolver import (
    resolve_baseline_us,
    resolve_baseline_info,
    calculate_speedup,
    geometric_mean_speedup,
    BaselineResolver,
    BaselineInfo,
    DEFAULT_HARDWARE,
)


class TestResolveBaselineUs:
    """resolve_baseline_us 函数测试"""

    def test_scalar_baseline_default_hardware(self):
        """单硬件 baseline（默认硬件）"""
        case_raw = {"baseline_perf_us": 40.2}
        result = resolve_baseline_us(case_raw, DEFAULT_HARDWARE)
        assert result == 40.2

    def test_scalar_baseline_non_default_hardware(self):
        """单硬件 baseline（非默认硬件）"""
        case_raw = {"baseline_perf_us": 40.2}
        result = resolve_baseline_us(case_raw, "910a")
        # 非默认硬件无法使用 scalar baseline
        assert result == 0.0

    def test_dict_baseline_specific_hardware(self):
        """多硬件 baseline（指定硬件）"""
        case_raw = {
            "baseline_perf_us": {
                "910b2": 40.2,
                "910b1": 45.1,
                "910a": 50.0,
                "950": 30.0,
            }
        }
        assert resolve_baseline_us(case_raw, "910b2") == 40.2
        assert resolve_baseline_us(case_raw, "910b1") == 45.1
        assert resolve_baseline_us(case_raw, "910a") == 50.0
        assert resolve_baseline_us(case_raw, "950") == 30.0

    def test_dict_baseline_missing_hardware(self):
        """多硬件 baseline（缺少指定硬件）"""
        case_raw = {
            "baseline_perf_us": {
                "910b2": 40.2,
            }
        }
        result = resolve_baseline_us(case_raw, "unknown_hardware")
        assert result == 0.0

    def test_no_baseline(self):
        """无 baseline"""
        case_raw = {}
        result = resolve_baseline_us(case_raw, DEFAULT_HARDWARE)
        assert result == 0.0

    def test_none_baseline(self):
        """baseline 为 None"""
        case_raw = {"baseline_perf_us": None}
        result = resolve_baseline_us(case_raw, DEFAULT_HARDWARE)
        assert result == 0.0

    def test_string_none_baseline(self):
        """baseline 为字符串 'None'"""
        case_raw = {"baseline_perf_us": "None"}
        result = resolve_baseline_us(case_raw, DEFAULT_HARDWARE)
        assert result == 0.0

    def test_dict_baseline_none_value(self):
        """多硬件 baseline 中某硬件值为 None"""
        case_raw = {
            "baseline_perf_us": {
                "910b2": None,
                "910b1": 45.1,
            }
        }
        assert resolve_baseline_us(case_raw, "910b2") == 0.0
        assert resolve_baseline_us(case_raw, "910b1") == 45.1

    def test_invalid_baseline_value(self):
        """无效 baseline 值"""
        case_raw = {"baseline_perf_us": "invalid"}
        result = resolve_baseline_us(case_raw, DEFAULT_HARDWARE)
        assert result == 0.0

    def test_zero_baseline(self):
        """零值 baseline"""
        case_raw = {"baseline_perf_us": 0}
        result = resolve_baseline_us(case_raw, DEFAULT_HARDWARE)
        assert result == 0.0

    def test_negative_baseline(self):
        """负值 baseline"""
        case_raw = {"baseline_perf_us": -10.0}
        result = resolve_baseline_us(case_raw, DEFAULT_HARDWARE)
        assert result == -10.0  # 函数不验证正负，由使用者判断


class TestResolveBaselineInfo:
    """resolve_baseline_info 函数测试"""

    def test_yaml_only(self):
        """仅 YAML baseline"""
        case_raw = {"baseline_perf_us": 40.2}
        info = resolve_baseline_info(case_raw, DEFAULT_HARDWARE)
        assert info.yaml_us == 40.2
        assert info.measured_us == 0.0
        assert info.used_us == 40.2
        assert info.source == "yaml"

    def test_measured_only(self):
        """仅实测 baseline"""
        case_raw = {}
        info = resolve_baseline_info(case_raw, DEFAULT_HARDWARE, measured_us=38.5)
        assert info.yaml_us == 0.0
        assert info.measured_us == 38.5
        assert info.used_us == 38.5
        assert info.source == "measured"

    def test_both_present_prefer_yaml(self):
        """两者都存在，优先 YAML"""
        case_raw = {"baseline_perf_us": 40.2}
        info = resolve_baseline_info(case_raw, DEFAULT_HARDWARE, measured_us=38.5)
        assert info.yaml_us == 40.2
        assert info.measured_us == 38.5
        assert info.used_us == 40.2
        assert info.source == "yaml"

    def test_both_present_prefer_measured(self):
        """两者都存在，优先实测"""
        case_raw = {"baseline_perf_us": 40.2}
        info = resolve_baseline_info(
            case_raw, DEFAULT_HARDWARE, measured_us=38.5, prefer_measured=True
        )
        assert info.yaml_us == 40.2
        assert info.measured_us == 38.5
        assert info.used_us == 38.5
        assert info.source == "measured"

    def test_yaml_zero_measured_present(self):
        """YAML 为零，实测存在"""
        case_raw = {}
        info = resolve_baseline_info(case_raw, DEFAULT_HARDWARE, measured_us=38.5)
        assert info.used_us == 38.5
        assert info.source == "measured"


class TestCalculateSpeedup:
    """calculate_speedup 函数测试"""

    def test_normal_speedup(self):
        """正常加速比"""
        result = calculate_speedup(40.0, 20.0)
        assert result == 2.0

    def test_slowdown(self):
        """减速（加速比 < 1）"""
        result = calculate_speedup(40.0, 80.0)
        assert result == 0.5

    def test_zero_baseline(self):
        """零 baseline"""
        result = calculate_speedup(0.0, 20.0)
        assert result is None

    def test_zero_custom(self):
        """零自定义时间"""
        result = calculate_speedup(40.0, 0.0)
        assert result is None

    def test_negative_values(self):
        """负值"""
        result = calculate_speedup(-40.0, 20.0)
        assert result is None
        result = calculate_speedup(40.0, -20.0)
        assert result is None

    def test_both_zero(self):
        """两者都为零"""
        result = calculate_speedup(0.0, 0.0)
        assert result is None


class TestGeometricMeanSpeedup:
    """geometric_mean_speedup 函数测试"""

    def test_normal_geometric_mean(self):
        """正常几何平均"""
        speedups = [2.0, 4.0, 8.0]
        # 几何平均 = (2 * 4 * 8)^(1/3) = 64^(1/3) = 4
        result = geometric_mean_speedup(speedups)
        assert result == pytest.approx(4.0)

    def test_single_value(self):
        """单一值"""
        speedups = [3.0]
        result = geometric_mean_speedup(speedups)
        # log(3.0) 计算可能有微小浮点误差
        assert result == pytest.approx(3.0)

    def test_empty_list(self):
        """空列表"""
        result = geometric_mean_speedup([])
        assert result == 0.0

    def test_contains_zero(self):
        """包含零值"""
        speedups = [2.0, 0.0, 4.0]
        # 零值被过滤掉，只计算 [2.0, 4.0] 的几何平均
        result = geometric_mean_speedup(speedups)
        expected = math.sqrt(2.0 * 4.0)
        assert result == pytest.approx(expected)

    def test_all_zeros(self):
        """全为零值"""
        speedups = [0.0, 0.0, 0.0]
        result = geometric_mean_speedup(speedups)
        assert result == 0.0

    def test_contains_negative(self):
        """包含负值"""
        speedups = [2.0, -1.0, 4.0]
        # 负值被过滤掉（< 0）
        result = geometric_mean_speedup(speedups)
        expected = math.sqrt(2.0 * 4.0)
        assert result == pytest.approx(expected)

    def test_very_small_values(self):
        """极小值"""
        speedups = [1e-9, 1e-8]
        # 极小值会被 clamp 到 1e-9
        result = geometric_mean_speedup(speedups)
        assert result > 0  # 应有正结果


class TestBaselineResolver:
    """BaselineResolver 类测试"""

    def test_initialization(self):
        """初始化"""
        resolver = BaselineResolver(hardware="910b2")
        assert resolver.hardware == "910b2"
        assert resolver.measured_baselines == {}

    def test_resolve_without_measured(self):
        """无实测值时解析"""
        resolver = BaselineResolver()
        case_raw = {"baseline_perf_us": 40.2}
        info = resolver.resolve(case_raw, "Exp", 0)
        assert info.yaml_us == 40.2
        assert info.measured_us == 0.0

    def test_resolve_with_measured(self):
        """有实测值时解析"""
        resolver = BaselineResolver()
        resolver.record_measured("Exp", 0, 38.5)
        case_raw = {"baseline_perf_us": 40.2}
        info = resolver.resolve(case_raw, "Exp", 0)
        assert info.measured_us == 38.5

    def test_record_and_get_measured(self):
        """记录和获取实测值"""
        resolver = BaselineResolver()
        resolver.record_measured("Add", 1, 25.0)
        resolver.record_measured("Mul", 2, 30.0)

        assert resolver.get_measured("Add", 1) == 25.0
        assert resolver.get_measured("Mul", 2) == 30.0
        assert resolver.get_measured("Exp", 0) is None

    def test_key_format(self):
        """键格式测试"""
        resolver = BaselineResolver()
        resolver.record_measured("Softmax", 10, 100.0)
        # 键格式为 "{op_name}_{case_id}"
        assert resolver.measured_baselines.get("Softmax_10") == 100.0

    def test_different_hardware(self):
        """不同硬件"""
        resolver_910a = BaselineResolver(hardware="910a")
        resolver_910b2 = BaselineResolver(hardware="910b2")
        resolver_950 = BaselineResolver(hardware="950")

        case_raw = {
            "baseline_perf_us": {
                "910a": 50.0,
                "910b2": 40.0,
                "950": 30.0,
            }
        }

        info_a = resolver_910a.resolve(case_raw, "Exp", 0)
        info_b = resolver_910b2.resolve(case_raw, "Exp", 0)
        info_c = resolver_950.resolve(case_raw, "Exp", 0)

        assert info_a.yaml_us == 50.0
        assert info_b.yaml_us == 40.0
        assert info_c.yaml_us == 30.0


class TestBaselineInfo:
    """BaselineInfo 数据类测试"""

    def test_dataclass_creation(self):
        """数据类创建"""
        info = BaselineInfo(
            yaml_us=40.0,
            measured_us=38.0,
            used_us=40.0,
            source="yaml",
        )
        assert info.yaml_us == 40.0
        assert info.measured_us == 38.0
        assert info.used_us == 40.0
        assert info.source == "yaml"

    def test_dataclass_equality(self):
        """数据类相等性"""
        info1 = BaselineInfo(yaml_us=40.0, measured_us=38.0, used_us=40.0, source="yaml")
        info2 = BaselineInfo(yaml_us=40.0, measured_us=38.0, used_us=40.0, source="yaml")
        assert info1 == info2