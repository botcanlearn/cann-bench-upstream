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
dtype 映射模块单元测试

测试对象：kernel_eval.utils.dtype_mapper
核心功能：
1. str_to_torch_dtype - 字符串转 torch.dtype
2. torch_dtype_to_str - torch.dtype 转字符串
3. is_float_dtype / is_int_dtype - 类型判断
4. get_dtype_size / get_supported_dtypes - 类型信息
"""

import pytest
import torch

from kernel_eval.utils.dtype_mapper import (
    str_to_torch_dtype,
    torch_dtype_to_str,
    is_float_dtype,
    is_int_dtype,
    get_dtype_size,
    get_supported_dtypes,
)


class TestStrToTorchDtype:
    """str_to_torch_dtype 函数测试"""

    def test_basic_float_types(self):
        """基础浮点类型"""
        assert str_to_torch_dtype("float16") == torch.float16
        assert str_to_torch_dtype("float32") == torch.float32
        assert str_to_torch_dtype("float64") == torch.float64

    def test_basic_int_types(self):
        """基础整数类型"""
        assert str_to_torch_dtype("int8") == torch.int8
        assert str_to_torch_dtype("int16") == torch.int16
        assert str_to_torch_dtype("int32") == torch.int32
        assert str_to_torch_dtype("int64") == torch.int64

    def test_uint_types(self):
        """无符号整数类型"""
        assert str_to_torch_dtype("uint8") == torch.uint8

    def test_case_insensitive(self):
        """大小写不敏感"""
        assert str_to_torch_dtype("FLOAT16") == torch.float16
        assert str_to_torch_dtype("Float32") == torch.float32
        assert str_to_torch_dtype("INT32") == torch.int32

    def test_invalid_dtype(self):
        """无效数据类型"""
        with pytest.raises(ValueError):
            str_to_torch_dtype("invalid_type")
        with pytest.raises(ValueError):
            str_to_torch_dtype("float128")


class TestTorchDtypeToStr:
    """torch_dtype_to_str 函数测试"""

    def test_float_to_str(self):
        """浮点类型转字符串"""
        assert torch_dtype_to_str(torch.float16) == "float16"
        assert torch_dtype_to_str(torch.float32) == "float32"
        assert torch_dtype_to_str(torch.float64) == "float64"

    def test_int_to_str(self):
        """整数类型转字符串"""
        assert torch_dtype_to_str(torch.int8) == "int8"
        assert torch_dtype_to_str(torch.int16) == "int16"
        assert torch_dtype_to_str(torch.int32) == "int32"
        assert torch_dtype_to_str(torch.int64) == "int64"

    def test_uint_to_str(self):
        """无符号整数转字符串"""
        assert torch_dtype_to_str(torch.uint8) == "uint8"

    def test_invalid_dtype(self):
        """无效 torch.dtype"""
        with pytest.raises(ValueError):
            torch_dtype_to_str("not_a_dtype")


class TestIsFloatDtype:
    """is_float_dtype 函数测试"""

    def test_float_types(self):
        """浮点类型判断"""
        assert is_float_dtype("float16") is True
        assert is_float_dtype("float32") is True
        assert is_float_dtype("float64") is True

    def test_int_types_not_float(self):
        """整数类型不是浮点"""
        assert is_float_dtype("int8") is False
        assert is_float_dtype("int32") is False
        assert is_float_dtype("int64") is False

    def test_case_insensitive(self):
        """大小写不敏感"""
        assert is_float_dtype("FLOAT16") is True
        assert is_float_dtype("Float32") is True


class TestIsIntDtype:
    """is_int_dtype 函数测试"""

    def test_int_types(self):
        """整数类型判断"""
        assert is_int_dtype("int8") is True
        assert is_int_dtype("int16") is True
        assert is_int_dtype("int32") is True
        assert is_int_dtype("int64") is True
        assert is_int_dtype("uint8") is True

    def test_float_types_not_int(self):
        """浮点类型不是整数"""
        assert is_int_dtype("float16") is False
        assert is_int_dtype("float32") is False
        assert is_int_dtype("float64") is False

    def test_case_insensitive(self):
        """大小写不敏感"""
        assert is_int_dtype("INT32") is True
        assert is_int_dtype("Int64") is True


class TestGetDtypeSize:
    """get_dtype_size 函数测试"""

    def test_float_sizes(self):
        """浮点类型大小"""
        assert get_dtype_size("float16") == 2  # 2 bytes
        assert get_dtype_size("float32") == 4  # 4 bytes
        assert get_dtype_size("float64") == 8  # 8 bytes

    def test_int_sizes(self):
        """整数类型大小"""
        assert get_dtype_size("int8") == 1
        assert get_dtype_size("int16") == 2
        assert get_dtype_size("int32") == 4
        assert get_dtype_size("int64") == 8
        assert get_dtype_size("uint8") == 1


class TestGetSupportedDtypes:
    """get_supported_dtypes 函数测试"""

    def test_returns_list(self):
        """返回列表"""
        dtypes = get_supported_dtypes()
        assert isinstance(dtypes, list)

    def test_contains_basic_types(self):
        """包含基础类型"""
        dtypes = get_supported_dtypes()
        assert "float16" in dtypes
        assert "float32" in dtypes
        assert "int32" in dtypes
        assert "int64" in dtypes

    def test_list_not_empty(self):
        """列表非空"""
        dtypes = get_supported_dtypes()
        assert len(dtypes) > 0


class TestDtypeConversionRoundtrip:
    """dtype 转换往返测试"""

    def test_str_dtype_roundtrip(self):
        """字符串 → torch.dtype → 字符串"""
        test_types = ["float16", "float32", "float64", "int8", "int32", "int64"]
        for dtype_str in test_types:
            dtype = str_to_torch_dtype(dtype_str)
            result = torch_dtype_to_str(dtype)
            assert result == dtype_str

    def test_dtype_str_roundtrip(self):
        """torch.dtype → 字符串 → torch.dtype"""
        test_dtypes = [torch.float16, torch.float32, torch.int32, torch.int64]
        for dtype in test_dtypes:
            dtype_str = torch_dtype_to_str(dtype)
            result = str_to_torch_dtype(dtype_str)
            assert result == dtype