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
张量处理工具单元测试

测试对象：kernel_eval.utils.tensor_utils
核心功能：
1. tensor_to_fp64_cpu - 单张量 FP64 CPU 转换
2. tensors_to_cpu - 批量张量 CPU 迁移
3. tensors_to_fp64_cpu - 批量张量 FP64 CPU 转换
4. tensors_to_device - 批量张量设备迁移
"""

import pytest
import torch

from kernel_eval.utils.tensor_utils import (
    tensor_to_fp64_cpu,
    tensors_to_cpu,
    tensors_to_fp64_cpu,
    tensors_to_device,
)


class TestTensorToFp64Cpu:
    """tensor_to_fp64_cpu 函数测试"""

    def test_float_tensor_conversion(self):
        """浮点张量转换"""
        t = torch.randn(3, 4, dtype=torch.float16)
        result = tensor_to_fp64_cpu(t)
        assert result.dtype == torch.float64
        assert result.device.type == "cpu"

    def test_float32_to_fp64(self):
        """float32 转 fp64"""
        t = torch.randn(2, 3, dtype=torch.float32)
        result = tensor_to_fp64_cpu(t)
        assert result.dtype == torch.float64

    def test_int_tensor_no_conversion(self):
        """整数张量不转换精度"""
        t = torch.randint(0, 100, (3, 4), dtype=torch.int32)
        result = tensor_to_fp64_cpu(t)
        assert result.dtype == torch.int32  # 保持原 dtype
        assert result.device.type == "cpu"

    def test_int64_no_conversion(self):
        """int64 不转换精度"""
        t = torch.randint(0, 100, (3, 4), dtype=torch.int64)
        result = tensor_to_fp64_cpu(t)
        assert result.dtype == torch.int64

    def test_int8_no_conversion(self):
        """int8 不转换精度"""
        t = torch.randint(0, 100, (3, 4), dtype=torch.int8)
        result = tensor_to_fp64_cpu(t)
        assert result.dtype == torch.int8

    def test_values_preserved(self):
        """值保持"""
        t = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        result = tensor_to_fp64_cpu(t)
        assert torch.allclose(result, t.double())

    def test_already_cpu_tensor(self):
        """已在 CPU 的张量"""
        t = torch.randn(3, 4, dtype=torch.float16, device="cpu")
        result = tensor_to_fp64_cpu(t)
        assert result.device.type == "cpu"
        assert result.dtype == torch.float64

    def test_empty_tensor(self):
        """空张量"""
        t = torch.empty(0, dtype=torch.float16)
        result = tensor_to_fp64_cpu(t)
        assert result.dtype == torch.float64
        assert result.shape == (0,)


class TestTensorsToCpu:
    """tensors_to_cpu 函数测试"""

    def test_single_tensor(self):
        """单个张量"""
        t = torch.randn(2, 3)
        result = tensors_to_cpu([t])
        assert len(result) == 1
        assert result[0].device.type == "cpu"

    def test_multiple_tensors(self):
        """多个张量"""
        tensors = [torch.randn(2, 3), torch.randn(4, 5)]
        result = tensors_to_cpu(tensors)
        assert len(result) == 2
        assert all(t.device.type == "cpu" for t in result)

    def test_nested_tensor_list(self):
        """嵌套张量列表"""
        tensors = [
            torch.randn(2, 3),
            [torch.randn(1, 2), torch.randn(3, 4)],
            torch.randn(5, 6),
        ]
        result = tensors_to_cpu(tensors)
        assert result[0].device.type == "cpu"
        assert isinstance(result[1], list)
        assert result[1][0].device.type == "cpu"
        assert result[2].device.type == "cpu"

    def test_nested_tensor_tuple(self):
        """嵌套张量元组"""
        tensors = [
            torch.randn(2, 3),
            (torch.randn(1, 2), torch.randn(3, 4)),
        ]
        result = tensors_to_cpu(tensors)
        assert isinstance(result[1], list)  # 元组转为列表
        assert result[1][0].device.type == "cpu"

    def test_mixed_types(self):
        """混合类型"""
        tensors = [
            torch.randn(2, 3),
            42,  # 整数
            "string",  # 字符串
            None,  # None
        ]
        result = tensors_to_cpu(tensors)
        assert result[0].device.type == "cpu"
        assert result[1] == 42
        assert result[2] == "string"
        assert result[3] is None

    def test_empty_list(self):
        """空列表"""
        result = tensors_to_cpu([])
        assert result == []

    def test_dtype_preserved(self):
        """dtype 保持"""
        tensors = [
            torch.randn(2, 3, dtype=torch.float16),
            torch.randint(0, 10, (2, 3), dtype=torch.int32),
        ]
        result = tensors_to_cpu(tensors)
        assert result[0].dtype == torch.float16
        assert result[1].dtype == torch.int32


class TestTensorsToFp64Cpu:
    """tensors_to_fp64_cpu 函数测试"""

    def test_float_tensors_conversion(self):
        """浮点张量批量转换"""
        tensors = [
            torch.randn(2, 3, dtype=torch.float16),
            torch.randn(4, 5, dtype=torch.float32),
        ]
        result = tensors_to_fp64_cpu(tensors)
        assert all(t.dtype == torch.float64 for t in result)
        assert all(t.device.type == "cpu" for t in result)

    def test_int_tensors_no_conversion(self):
        """整数张量不转换"""
        tensors = [
            torch.randint(0, 10, (2, 3), dtype=torch.int32),
            torch.randint(0, 10, (4, 5), dtype=torch.int64),
        ]
        result = tensors_to_fp64_cpu(tensors)
        assert result[0].dtype == torch.int32
        assert result[1].dtype == torch.int64

    def test_mixed_tensors(self):
        """混合类型张量"""
        tensors = [
            torch.randn(2, 3, dtype=torch.float16),  # float
            torch.randint(0, 10, (2, 3), dtype=torch.int32),  # int
        ]
        result = tensors_to_fp64_cpu(tensors)
        assert result[0].dtype == torch.float64
        assert result[1].dtype == torch.int32

    def test_nested_tensors(self):
        """嵌套张量"""
        tensors = [
            torch.randn(2, 3, dtype=torch.float16),
            [torch.randn(1, 2, dtype=torch.float32), torch.randint(0, 10, (3, 4), dtype=torch.int32)],
        ]
        result = tensors_to_fp64_cpu(tensors)
        assert result[0].dtype == torch.float64
        assert result[1][0].dtype == torch.float64
        assert result[1][1].dtype == torch.int32

    def test_values_preserved(self):
        """值保持"""
        tensors = [torch.tensor([1.0, 2.0], dtype=torch.float32)]
        result = tensors_to_fp64_cpu(tensors)
        assert torch.allclose(result[0], torch.tensor([1.0, 2.0], dtype=torch.float64))


class TestTensorsToDevice:
    """tensors_to_device 函数测试"""

    def test_to_cpu(self):
        """迁移到 CPU"""
        tensors = [torch.randn(2, 3)]
        result = tensors_to_device(tensors, "cpu")
        assert result[0].device.type == "cpu"

    def test_nested_to_cpu(self):
        """嵌套张量迁移到 CPU"""
        tensors = [
            torch.randn(2, 3),
            [torch.randn(1, 2), torch.randn(3, 4)],
        ]
        result = tensors_to_device(tensors, "cpu")
        assert result[0].device.type == "cpu"
        assert result[1][0].device.type == "cpu"

    def test_dtype_preserved_on_device_change(self):
        """设备迁移时 dtype 保持"""
        tensors = [torch.randn(2, 3, dtype=torch.float16)]
        result = tensors_to_device(tensors, "cpu")
        assert result[0].dtype == torch.float16

    def test_empty_list_device(self):
        """空列表设备迁移"""
        result = tensors_to_device([], "cpu")
        assert result == []

    def test_mixed_types_device(self):
        """混合类型设备迁移"""
        tensors = [
            torch.randn(2, 3),
            42,
            "string",
        ]
        result = tensors_to_device(tensors, "cpu")
        assert result[0].device.type == "cpu"
        assert result[1] == 42
        assert result[2] == "string"


class TestTensorUtilsIntegration:
    """张量工具集成测试"""

    def test_chain_cpu_fp64(self):
        """链式操作：先 CPU 再 FP64"""
        tensors = [torch.randn(2, 3, dtype=torch.float16)]
        # 先迁移到 CPU
        cpu_tensors = tensors_to_cpu(tensors)
        # 再转 FP64
        fp64_tensors = tensors_to_fp64_cpu(cpu_tensors)
        assert fp64_tensors[0].dtype == torch.float64

    def test_fp64_cpu_is_cpu(self):
        """FP64 CPU 结果已在 CPU"""
        tensors = [torch.randn(2, 3, dtype=torch.float16)]
        result = tensors_to_fp64_cpu(tensors)
        assert result[0].device.type == "cpu"

    def test_real_world_scenario(self):
        """真实场景：输入预处理"""
        # 模拟算子输入：float tensor + int tensor + scalar
        inputs = [
            torch.randn(3, 4, dtype=torch.float16),
            torch.randint(0, 10, (3, 4), dtype=torch.int32),
            1.0,  # scalar
        ]
        # Golden 计算前需要转 FP64
        fp64_inputs = tensors_to_fp64_cpu(inputs)
        assert fp64_inputs[0].dtype == torch.float64
        assert fp64_inputs[1].dtype == torch.int32
        assert fp64_inputs[2] == 1.0