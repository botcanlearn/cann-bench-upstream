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
输入池管理模块单元测试

测试对象：kernel_eval.eval.input_pool
核心功能：
1. InputPool 类 - 输入池管理
2. InputPoolConfig 配置
3. create_input_pool 便捷函数
"""

import pytest
import torch

from kernel_eval.eval.input_pool import (
    InputPool,
    InputPoolConfig,
    create_input_pool,
)


class TestInputPoolConfig:
    """InputPoolConfig 数据类测试"""

    def test_default_config(self):
        """默认配置"""
        config = InputPoolConfig()
        assert config.max_pool_size == 8
        assert config.max_memory_mb == 512

    def test_custom_config(self):
        """自定义配置"""
        config = InputPoolConfig(max_pool_size=16, max_memory_mb=1024)
        assert config.max_pool_size == 16
        assert config.max_memory_mb == 1024


class TestInputPool:
    """InputPool 类测试"""

    def test_basic_creation(self):
        """基本创建"""
        inputs = [torch.randn(2, 3)]
        pool = InputPool(inputs, pool_size=4)
        assert pool.size() == 4

    def test_get_next_returns_clone(self):
        """get_next 返回 clone"""
        inputs = [torch.randn(2, 3)]
        pool = InputPool(inputs, pool_size=2)

        item1 = pool.get_next()
        item2 = pool.get_next()

        # 应为不同的 clone
        assert item1[0].data_ptr() != inputs[0].data_ptr()
        assert item2[0].data_ptr() != inputs[0].data_ptr()

    def test_rotation(self):
        """轮换使用"""
        inputs = [torch.randn(2, 3)]
        pool = InputPool(inputs, pool_size=2)

        # 获取多次，应轮换
        items = [pool.get_next() for _ in range(6)]
        # 2 个 clone，获取 6 次，应循环使用
        # 第 0 次和第 2 次应使用同一 clone
        assert items[0][0].data_ptr() == items[2][0].data_ptr()
        # 第 1 次和第 3 次应使用同一 clone
        assert items[1][0].data_ptr() == items[3][0].data_ptr()

    def test_nested_inputs(self):
        """嵌套输入"""
        inputs = [
            torch.randn(2, 3),
            [torch.randn(1, 2), torch.randn(3, 4)],
        ]
        pool = InputPool(inputs, pool_size=2)
        item = pool.get_next()
        assert isinstance(item[1], list)
        assert len(item[1]) == 2

    def test_memory_limit(self):
        """内存限制"""
        # 创建大张量
        large_tensor = torch.randn(1000, 1000)  # ~4MB
        inputs = [large_tensor]

        # 设置较小内存限制
        config = InputPoolConfig(max_memory_mb=10)  # 10MB 限制
        pool = InputPool(inputs, pool_size=8, config=config)

        # 内存限制应减少实际池大小
        # 4MB * 8 = 32MB > 10MB，实际池大小应减少
        assert pool.size() < 8

    def test_max_pool_size_limit(self):
        """池大小限制"""
        inputs = [torch.randn(2, 3)]
        config = InputPoolConfig(max_pool_size=4)
        pool = InputPool(inputs, pool_size=10, config=config)
        # 请求 10 但限制为 4
        assert pool.size() == 4

    def test_empty_inputs_min_pool_size(self):
        """空 inputs + pool_size=0: pool 最小 clamp 为 1，get_next 返回空列表"""
        pool = InputPool([], pool_size=0)
        assert pool.size() == 1
        result = pool.get_next()
        # 空 inputs 产生空的克隆列表
        assert result == []

    def test_clear_pool_raises_on_get_next(self):
        """clear 后 pool 为空，get_next 抛 RuntimeError"""
        inputs = [torch.randn(2, 3)]
        pool = InputPool(inputs, pool_size=2)
        assert pool.size() == 2
        pool.clear()
        with pytest.raises(RuntimeError):
            pool.get_next()

    def test_clear_pool(self):
        """清空池"""
        inputs = [torch.randn(2, 3)]
        pool = InputPool(inputs, pool_size=2)
        assert pool.size() == 2
        pool.clear()
        assert pool.size() == 0
        assert pool.idx == 0

    def test_len_method(self):
        """len 方法"""
        inputs = [torch.randn(2, 3)]
        pool = InputPool(inputs, pool_size=4)
        assert len(pool) == 4

    def test_single_pool_size(self):
        """单个池大小"""
        inputs = [torch.randn(2, 3)]
        pool = InputPool(inputs, pool_size=1)
        assert pool.size() == 1
        item1 = pool.get_next()
        item2 = pool.get_next()
        # 单个 clone，每次返回相同
        assert item1[0].data_ptr() == item2[0].data_ptr()

    def test_preserves_dtype(self):
        """保持 dtype"""
        inputs = [
            torch.randn(2, 3, dtype=torch.float16),
            torch.randint(0, 10, (2, 3), dtype=torch.int32),
        ]
        pool = InputPool(inputs, pool_size=2)
        item = pool.get_next()
        assert item[0].dtype == torch.float16
        assert item[1].dtype == torch.int32


class TestCreateInputPool:
    """create_input_pool 函数测试"""

    def test_basic_creation(self):
        """基本创建"""
        inputs = [torch.randn(2, 3)]
        pool = create_input_pool(inputs, warmup=3, repeat=5)
        # pool_size = warmup + repeat = 8
        assert pool.size() == 8

    def test_with_memory_limit(self):
        """带内存限制"""
        inputs = [torch.randn(100, 100)]  # ~40KB
        pool = create_input_pool(inputs, warmup=5, repeat=10, max_memory_mb=1)
        # 40KB * 15 = 600KB < 1MB，应正常创建 15 个
        # 但 max_pool_size 默认为 warmup+repeat
        assert pool.size() <= 15

    def test_warmup_repeat_zero(self):
        """预热和重复为零"""
        inputs = [torch.randn(2, 3)]
        pool = create_input_pool(inputs, warmup=0, repeat=0)
        # pool_size = 0，至少创建 1 个
        assert pool.size() >= 1

    def test_returns_input_pool(self):
        """返回 InputPool 实例"""
        inputs = [torch.randn(2, 3)]
        pool = create_input_pool(inputs, warmup=2, repeat=3)
        assert isinstance(pool, InputPool)


class TestInputPoolMemoryEstimation:
    """输入池内存估算测试"""

    def test_estimate_tensor_memory(self):
        """估算张量内存"""
        # float32: 4 bytes per element
        tensor = torch.randn(1000, 1000, dtype=torch.float32)  # 4MB
        inputs = [tensor]
        pool = InputPool(inputs, pool_size=1)

        # 内存估算应正确
        # 池大小 1 * 4MB < 512MB，应正常创建
        assert pool.size() == 1

    def test_estimate_nested_memory(self):
        """估算嵌套张量内存"""
        inputs = [
            torch.randn(500, 500),  # 1MB
            [torch.randn(500, 500), torch.randn(500, 500)],  # 2MB
        ]
        pool = InputPool(inputs, pool_size=1)
        # 总 ~3MB
        assert pool.size() == 1

    def test_mixed_types_memory(self):
        """混合类型内存估算"""
        inputs = [
            torch.randn(1000, 1000),  # float32 tensor
            42,  # scalar (不计入内存)
            "string",  # string (不计入内存)
        ]
        pool = InputPool(inputs, pool_size=1)
        # 只有 tensor 计入内存
        assert pool.size() == 1


class TestInputPoolIntegration:
    """输入池集成测试"""

    def test_performance_measurement_scenario(self):
        """性能测量场景"""
        inputs = [torch.randn(10, 10)]
        pool = create_input_pool(inputs, warmup=3, repeat=5)

        # 模拟性能测量循环
        for i in range(8):  # warmup + repeat
            cloned_inputs = pool.get_next()
            # 模拟算子执行
            output = cloned_inputs[0] * 2
        # 循环完成后池应仍在工作
        assert pool.size() > 0

    def test_different_tensors_in_pool(self):
        """池中不同张量"""
        inputs = [
            torch.randn(2, 3),
            torch.randn(4, 5),
        ]
        pool = InputPool(inputs, pool_size=2)
        item = pool.get_next()
        # 应 clone 所有输入张量
        assert len(item) == 2
        assert item[0].shape == (2, 3)
        assert item[1].shape == (4, 5)

    def test_pool_rotation_correctness(self):
        """池轮换正确性"""
        inputs = [torch.randn(2, 3)]
        pool = InputPool(inputs, pool_size=3)

        # 获取 9 次，应循环 3 次
        ptrs = []
        for _ in range(9):
            item = pool.get_next()
            ptrs.append(item[0].data_ptr())

        # 第 0, 3, 6 次应相同
        assert ptrs[0] == ptrs[3] == ptrs[6]
        # 第 1, 4, 7 次应相同
        assert ptrs[1] == ptrs[4] == ptrs[7]
        # 第 2, 5, 8 次应相同
        assert ptrs[2] == ptrs[5] == ptrs[8]