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
精度阈值模块单元测试

测试对象：kernel_eval.utils.thresholds
"""

import pytest

from kernel_eval.utils.thresholds import (
    PRECISION_THRESHOLDS,
    SMALL_VALUE_THRESHOLDS,
    SMALL_VALUE_ERROR_THRESHOLDS,
    CANCEL_BOUNDARY_THRESHOLDS,
    CANCEL_ZERO_THRESHOLDS,
    get_threshold,
    get_small_value_threshold,
    get_small_value_error,
    get_cancel_boundary,
    get_cancel_zero_threshold,
)


class TestGetThreshold:
    """get_threshold 函数测试"""

    def test_float16_threshold(self):
        assert get_threshold("float16") == pytest.approx(2**-10)

    def test_float32_threshold(self):
        assert get_threshold("float32") == pytest.approx(2**-13)

    def test_bfloat16_threshold(self):
        assert get_threshold("bfloat16") == pytest.approx(2**-7)

    def test_int_type_threshold(self):
        assert get_threshold("int8") == 0
        assert get_threshold("int32") == 0
        assert get_threshold("int64") == 0

    def test_unknown_dtype(self):
        result = get_threshold("unknown_dtype")
        assert result == pytest.approx(2**-13)

    def test_case_insensitive(self):
        assert get_threshold("FLOAT16") == pytest.approx(2**-10)
        assert get_threshold("Float32") == pytest.approx(2**-13)


class TestGetSmallValueThreshold:
    """get_small_value_threshold 函数测试"""

    def test_float16_small_value(self):
        assert get_small_value_threshold("float16") == pytest.approx(2**-11)

    def test_float32_small_value(self):
        assert get_small_value_threshold("float32") == pytest.approx(2**-14)

    def test_unknown_dtype_default(self):
        result = get_small_value_threshold("unknown")
        assert result == pytest.approx(2**-14)


class TestGetSmallValueError:
    """get_small_value_error 函数测试"""

    def test_float16_small_error(self):
        assert get_small_value_error("float16") == pytest.approx(2**-16)

    def test_float32_small_error(self):
        assert get_small_value_error("float32") == pytest.approx(2**-30)


class TestGetCancelBoundary:
    """get_cancel_boundary 函数测试"""

    def test_float32_cancel_boundary(self):
        assert get_cancel_boundary("float32") == pytest.approx(2**-8)

    def test_float16_cancel_boundary(self):
        assert get_cancel_boundary("float16") == pytest.approx(2**-5)

    def test_bfloat16_cancel_boundary(self):
        assert get_cancel_boundary("bfloat16") == pytest.approx(2**-3)


class TestGetCancelZeroThreshold:
    """get_cancel_zero_threshold 函数测试"""

    def test_float32_cancel_zero(self):
        assert get_cancel_zero_threshold("float32") == pytest.approx(2**-8)


class TestPrecisionThresholds:
    """精度阈值表测试"""

    def test_threshold_table_complete(self):
        expected_types = [
            "float16", "float32", "float64", "bfloat16",
            "int8", "int16", "int32", "int64",
            "uint8", "uint16", "uint32", "uint64",
        ]
        for dtype in expected_types:
            assert dtype in PRECISION_THRESHOLDS, f"Missing threshold for {dtype}"
            assert isinstance(PRECISION_THRESHOLDS[dtype], (int, float))

    def test_float_thresholds_positive(self):
        for dtype in ["float16", "float32", "bfloat16"]:
            assert PRECISION_THRESHOLDS[dtype] > 0

    def test_int_thresholds_zero(self):
        for dtype in ["int8", "int16", "int32", "int64"]:
            assert PRECISION_THRESHOLDS[dtype] == 0

    def test_small_value_table_has_float_keys(self):
        for dtype in ["float16", "float32", "bfloat16", "float8_e4m3fn", "float8_e5m2"]:
            assert dtype in SMALL_VALUE_THRESHOLDS, f"Missing {dtype} in SMALL_VALUE_THRESHOLDS"

    def test_small_value_error_table_has_float_keys(self):
        for dtype in ["float16", "float32", "bfloat16", "float8_e4m3fn", "float8_e5m2"]:
            assert dtype in SMALL_VALUE_ERROR_THRESHOLDS, f"Missing {dtype} in SMALL_VALUE_ERROR_THRESHOLDS"

    def test_cancel_boundary_table_has_float_keys(self):
        for dtype in ["float16", "float32", "bfloat16", "float8_e4m3fn", "float8_e5m2"]:
            assert dtype in CANCEL_BOUNDARY_THRESHOLDS, f"Missing {dtype} in CANCEL_BOUNDARY_THRESHOLDS"

    def test_cancel_zero_table_has_float_keys(self):
        for dtype in ["float16", "float32", "bfloat16", "float8_e4m3fn", "float8_e5m2"]:
            assert dtype in CANCEL_ZERO_THRESHOLDS, f"Missing {dtype} in CANCEL_ZERO_THRESHOLDS"
