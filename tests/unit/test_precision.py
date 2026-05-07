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
精度验证模块单元测试

测试对象：kernel_eval.utils.precision
核心功能：
1. get_threshold - 精度阈值获取
2. get_small_value_threshold - 小值域阈值
3. get_small_value_error - 小值域误差阈值
4. compare_tensors - 张量对比（核心函数）
5. CompareResult 数据类
"""

import pytest
import torch

from kernel_eval.utils.precision import (
    PRECISION_THRESHOLDS,
    SMALL_VALUE_THRESHOLDS,
    SMALL_VALUE_ERROR_THRESHOLDS,
    get_threshold,
    get_small_value_threshold,
    get_small_value_error,
    get_cancel_boundary,
    get_cancel_zero_threshold,
    CompareResult,
    compare_tensors,
)


class TestGetThreshold:
    """get_threshold 函数测试"""

    def test_float16_threshold(self):
        """float16 精度阈值"""
        assert get_threshold("float16") == pytest.approx(2**-10)

    def test_float32_threshold(self):
        """float32 精度阈值"""
        assert get_threshold("float32") == pytest.approx(2**-13)

    def test_bfloat16_threshold(self):
        """bfloat16 精度阈值"""
        assert get_threshold("bfloat16") == pytest.approx(2**-7)

    def test_int_type_threshold(self):
        """整数类型阈值（完全相等）"""
        assert get_threshold("int8") == 0
        assert get_threshold("int32") == 0
        assert get_threshold("int64") == 0

    def test_unknown_dtype(self):
        """未知类型使用默认阈值"""
        result = get_threshold("unknown_dtype")
        # 默认使用 float32 阈值
        assert result == pytest.approx(2**-13)

    def test_case_insensitive(self):
        """大小写不敏感"""
        assert get_threshold("FLOAT16") == pytest.approx(2**-10)
        assert get_threshold("Float32") == pytest.approx(2**-13)


class TestGetSmallValueThreshold:
    """get_small_value_threshold 函数测试"""

    def test_float16_small_value(self):
        """float16 小值域阈值"""
        assert get_small_value_threshold("float16") == pytest.approx(2**-11)

    def test_float32_small_value(self):
        """float32 小值域阈值"""
        assert get_small_value_threshold("float32") == pytest.approx(2**-14)

    def test_unknown_dtype_default(self):
        """未知类型使用默认"""
        result = get_small_value_threshold("unknown")
        assert result == pytest.approx(2**-14)


class TestGetSmallValueError:
    """get_small_value_error 函数测试"""

    def test_float16_small_error(self):
        """float16 小值域误差阈值"""
        assert get_small_value_error("float16") == pytest.approx(2**-16)

    def test_float32_small_error(self):
        """float32 小值域误差阈值"""
        assert get_small_value_error("float32") == pytest.approx(2**-30)


class TestGetCancelBoundary:
    """get_cancel_boundary 函数测试"""

    def test_float32_cancel_boundary(self):
        """float32 相消边界"""
        assert get_cancel_boundary("float32") == pytest.approx(2**-8)

    def test_float16_cancel_boundary(self):
        """float16 相消边界"""
        assert get_cancel_boundary("float16") == pytest.approx(2**-5)

    def test_bfloat16_cancel_boundary(self):
        """bfloat16 相消边界"""
        assert get_cancel_boundary("bfloat16") == pytest.approx(2**-3)


class TestGetCancelZeroThreshold:
    """get_cancel_zero_threshold 函数测试"""

    def test_float32_cancel_zero(self):
        """float32 相消零值阈值"""
        assert get_cancel_zero_threshold("float32") == pytest.approx(2**-8)


class TestCompareResult:
    """CompareResult 数据类测试"""

    def test_creation(self):
        """创建结果"""
        result = CompareResult(
            passed=True,
            dtype="float32",
            threshold=2**-13,
            mere=1e-5,
            mare=2e-5,
        )
        assert result.passed is True
        assert result.dtype == "float32"
        assert result.mere == 1e-5
        assert result.mare == 2e-5

    def test_to_dict(self):
        """序列化为字典"""
        result = CompareResult(
            passed=True,
            dtype="float32",
            threshold=0.0001,
            mere=1e-5,
            mare=2e-5,
            mismatch_count=0,
            total_count=100,
        )
        d = result.to_dict()
        assert d["passed"] is True
        assert d["dtype"] == "float32"
        assert d["mere"] == 1e-5
        assert d["mismatch_count"] == 0

    def test_failed_result(self):
        """失败结果"""
        result = CompareResult(
            passed=False,
            dtype="float32",
            threshold=0.0001,
            mere=0.01,
            mare=0.05,
            error_msg="MARE exceeded threshold",
        )
        assert result.passed is False
        assert result.error_msg == "MARE exceeded threshold"

    def test_default_values(self):
        """默认值"""
        result = CompareResult(passed=True, dtype="float32", threshold=0.0001)
        assert result.mere == 0.0
        assert result.mare == 0.0
        assert result.mismatch_count == 0


class TestCompareTensors:
    """compare_tensors 函数测试"""

    def test_identical_tensors_pass(self):
        """完全相同的张量"""
        golden = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        output = golden.clone()
        result = compare_tensors(output, golden, "float32")
        assert result.passed is True
        assert result.mere == pytest.approx(0.0, abs=1e-10)
        assert result.mare == pytest.approx(0.0, abs=1e-10)

    def test_small_difference_pass(self):
        """小误差通过"""
        golden = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        output = golden + 1e-6  # 极小误差
        result = compare_tensors(output, golden, "float32")
        # float32 阈值约 2^-13 ≈ 0.000122
        # 1e-6 / 1.0 ≈ 1e-6 << 阈值，应通过
        assert result.passed is True

    def test_large_difference_fail(self):
        """大误差失败"""
        golden = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        output = golden + 0.1  # 10% 误差
        result = compare_tensors(output, golden, "float32")
        # MERE ≈ 0.1，远超阈值
        assert result.passed is False

    def test_int_tensors_exact_match(self):
        """整数张量精确匹配"""
        golden = torch.tensor([1, 2, 3], dtype=torch.int64)
        output = golden.clone()
        result = compare_tensors(output, golden, "int32")
        assert result.passed is True
        assert result.mere == 0.0

    def test_int_tensors_mismatch(self):
        """整数张量不匹配"""
        golden = torch.tensor([1, 2, 3], dtype=torch.int64)
        output = torch.tensor([1, 2, 4], dtype=torch.int64)
        result = compare_tensors(output, golden, "int32")
        # int 类型阈值 = 0，任何差异都失败
        assert result.passed is False

    def test_multiple_outputs(self):
        """多输出对比"""
        golden = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]
        output = [golden[0].clone(), golden[1].clone()]
        result = compare_tensors(output, golden, "float32")
        assert result.passed is True

    def test_output_count_mismatch(self):
        """输出数量不匹配"""
        golden = [torch.tensor([1.0]), torch.tensor([2.0])]
        output = [torch.tensor([1.0])]
        result = compare_tensors(output, golden, "float32")
        assert result.passed is False
        assert result.error_msg is not None

    def test_shape_mismatch(self):
        """形状不匹配"""
        golden = torch.tensor([1.0, 2.0, 3.0])
        output = torch.tensor([1.0, 2.0])
        # 形状不匹配会导致错误
        result = compare_tensors(output, golden, "float32")
        assert result.passed is False

    def test_float16_threshold(self):
        """float16 阈值测试"""
        golden = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        output = golden + 0.0005  # 约 0.05% 误差
        result = compare_tensors(output, golden, "float16")
        # float16 阈值约 2^-10 ≈ 0.001
        # 0.0005 / 1.0 ≈ 0.0005 < 阈值，应通过
        assert result.passed is True

    def test_empty_tensor(self):
        """空张量"""
        golden = torch.tensor([], dtype=torch.float64)
        output = torch.tensor([], dtype=torch.float64)
        result = compare_tensors(output, golden, "float32")
        # 空张量：元素级比较无差异，视为通过
        assert result.passed is True


class TestPrecisionThresholds:
    """精度阈值表测试"""

    def test_threshold_table_complete(self):
        """阈值表完整性"""
        expected_types = [
            "float16", "float32", "float64", "bfloat16",
            "int8", "int16", "int32", "int64",
            "uint8", "uint16", "uint32", "uint64",
        ]
        for dtype in expected_types:
            assert dtype in PRECISION_THRESHOLDS, f"Missing threshold for {dtype}"
            threshold = PRECISION_THRESHOLDS[dtype]
            assert isinstance(threshold, (int, float))

    def test_float_thresholds_positive(self):
        """浮点阈值为正数"""
        for dtype in ["float16", "float32", "bfloat16"]:
            assert dtype in PRECISION_THRESHOLDS, f"Missing threshold for {dtype}"
            assert PRECISION_THRESHOLDS[dtype] > 0

    def test_int_thresholds_zero(self):
        """整数阈值为零"""
        for dtype in ["int8", "int16", "int32", "int64"]:
            assert dtype in PRECISION_THRESHOLDS, f"Missing threshold for {dtype}"
            assert PRECISION_THRESHOLDS[dtype] == 0