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

"""value_range 标量区间广播语义单元测试

测试对象：kernel_eval.data.data_generator.DataGenerator._normalize_value_ranges
回归目标：cases.yaml 里 `value_range: [-1, 1]` 是"所有输入同一区间"的简写。原实现写作
`[value_ranges] + [None] * (num_inputs - 1)`（注释标的是"单输入算子"），多输入算子除
第 1 个输入外全部拿不到声明区间，静默落到 _parse_range 的**非负**默认值
（float [0,1] / int [0,100]）。
"""

import pytest
import torch

from kernel_eval.data.data_generator import DataGenerator


@pytest.fixture
def gen():
    return DataGenerator()


class TestNormalizeValueRanges:
    def test_scalar_range_broadcasts_to_all_inputs(self, gen):
        assert gen._normalize_value_ranges([-1, 1], 3) == [[-1, 1], [-1, 1], [-1, 1]]

    def test_scalar_range_single_input_unchanged(self, gen):
        assert gen._normalize_value_ranges([-1, 1], 1) == [[-1, 1]]

    def test_per_input_ranges_preserved(self, gen):
        vr = [[-1, 1], [0, 5], [2, 3]]
        assert gen._normalize_value_ranges(vr, 3) == vr

    def test_per_input_short_list_pads_with_last(self, gen):
        """逐输入形式给少了仍沿用原有的"补最后一个"行为，本次不改"""
        assert gen._normalize_value_ranges([[-1, 1], [0, 5]], 4) == \
            [[-1, 1], [0, 5], [0, 5], [0, 5]]

    def test_empty_range_yields_none_per_input(self, gen):
        assert gen._normalize_value_ranges([], 3) == [None, None, None]

    def test_special_value_scalar_range_broadcasts(self, gen):
        vr = [float('-inf'), float('inf')]
        out = gen._normalize_value_ranges(vr, 2)
        assert out == [vr, vr]


class TestGeneratedDataRespectsDeclaration:
    def test_all_inputs_get_declared_symmetric_range(self, gen):
        """核心回归：多输入算子的后续输入不能再落到非负默认值"""
        tensors = gen.generate_input_tensors_from_case(
            input_shapes=[[256, 64], [256, 64], [256, 64]],
            dtypes=['float32', 'float32', 'float32'],
            value_ranges=[-1, 1],
            seed=0,
        )
        assert len(tensors) == 3
        for i, t in enumerate(tensors):
            assert t.min().item() < -0.9, f"输入 {i} 未取到负值：声明区间未生效"
            assert t.max().item() > 0.9
            assert t.min().item() >= -1.0 and t.max().item() <= 1.0

    def test_narrow_range_applies_to_every_input(self, gen):
        tensors = gen.generate_input_tensors_from_case(
            input_shapes=[[128, 32], [128, 32]],
            dtypes=['float32', 'float32'],
            value_ranges=[-0.1, 0.1],
            seed=1,
        )
        for t in tensors:
            assert t.abs().max().item() <= 0.1

    def test_constant_range_zeroes_every_input(self, gen):
        tensors = gen.generate_input_tensors_from_case(
            input_shapes=[[16, 8], [16, 8], [16, 8]],
            dtypes=['float32', 'float32', 'float32'],
            value_ranges=[0, 0],
            seed=2,
        )
        for t in tensors:
            assert bool((t == 0).all())

    def test_per_input_form_still_differentiates(self, gen):
        """逐输入形式不受影响：各输入仍取各自区间"""
        tensors = gen.generate_input_tensors_from_case(
            input_shapes=[[512], [512]],
            dtypes=['float32', 'float32'],
            value_ranges=[[-1, 1], [10, 20]],
            seed=3,
        )
        assert tensors[0].min().item() < 0
        assert tensors[1].min().item() >= 10 and tensors[1].max().item() <= 20

    def test_int_inputs_broadcast_too(self, gen):
        tensors = gen.generate_input_tensors_from_case(
            input_shapes=[[512], [512]],
            dtypes=['int32', 'int32'],
            value_ranges=[-5, 5],
            seed=4,
        )
        for t in tensors:
            assert int(t.min()) >= -5 and int(t.max()) <= 5
            assert int(t.min()) < 0, "整数输入同样应取到负值"

    def test_none_shape_placeholder_unaffected(self, gen):
        tensors = gen.generate_input_tensors_from_case(
            input_shapes=[[64, 8], None, [64, 8]],
            dtypes=['float32', None, 'float32'],
            value_ranges=[-2, 2],
            seed=5,
        )
        assert tensors[1] is None
        for t in (tensors[0], tensors[2]):
            assert t.min().item() < -1.5 and t.max().item() <= 2.0

    def test_determinism_preserved(self, gen):
        kw = dict(input_shapes=[[128], [128], [128]],
                  dtypes=['float32'] * 3, value_ranges=[-1, 1], seed=7)
        a = gen.generate_input_tensors_from_case(**kw)
        b = gen.generate_input_tensors_from_case(**kw)
        for x, y in zip(a, b):
            assert torch.equal(x, y)

    def test_inputs_are_not_all_identical(self, gen):
        """广播的是区间不是数据：各输入仍是独立采样"""
        a, b = gen.generate_input_tensors_from_case(
            input_shapes=[[1024], [1024]],
            dtypes=['float32', 'float32'],
            value_ranges=[-1, 1],
            seed=8,
        )
        assert not torch.equal(a, b)
