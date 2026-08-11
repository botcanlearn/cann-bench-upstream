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
bool dtype 输入生成单元测试

测试对象：kernel_eval.data.data_generator.DataGenerator + kernel_eval.utils.dtype_mapper
回归目标：bool 既不属于 is_float_dtype 也不属于 is_int_dtype，修复前会落到 zeros 兜底
分支，导致所有 bool 输入被静默生成为全 False、value_range 完全失效（MoeGatingTopKSoftmax
的 finished 恒为 False，"已完成 token" 分支从未被覆盖）。
"""

import pytest
import torch

from kernel_eval.data.data_generator import DataGenerator
from kernel_eval.utils.dtype_mapper import is_bool_dtype, is_float_dtype, is_int_dtype


def _gen(shape, value_range, seed=42):
    gen = torch.Generator()
    gen.manual_seed(seed)
    return DataGenerator().generate_input_tensor(shape, 'bool', value_range, generator=gen)


class TestIsBoolDtype:
    """is_bool_dtype 函数测试"""

    def test_bool_recognized(self):
        assert is_bool_dtype('bool')
        assert is_bool_dtype('BOOL')

    def test_bool_excluded_from_float_and_int(self):
        """bool 必须与 float/int 分支互斥，否则会被 randint 以非法 dtype 调用"""
        assert not is_float_dtype('bool')
        assert not is_int_dtype('bool')

    def test_non_bool_rejected(self):
        for dtype in ('float32', 'int32', 'uint8', 'bfloat16'):
            assert not is_bool_dtype(dtype)


class TestGenerateBoolTensor:
    """bool 张量生成测试"""

    def test_dtype_is_bool(self):
        assert _gen([4, 8], [0, 1]).dtype == torch.bool

    def test_range_0_1_produces_both_values(self):
        """[0, 1] 必须产出 True 和 False —— 这是修复前失效的核心行为"""
        t = _gen([64, 64], [0, 1])
        assert t.any(), "全 False：value_range 未生效"
        assert not t.all(), "全 True：value_range 未生效"
        assert 0.4 < t.float().mean().item() < 0.6

    def test_range_0_0_all_false(self):
        assert not _gen([16, 16], [0, 0]).any()

    def test_range_1_1_all_true(self):
        assert _gen([16, 16], [1, 1]).all()

    def test_default_range_is_random(self):
        """未指定 value_range 时 _parse_range 返回 (0, 100)，夹取后应等价于 [0, 1]"""
        t = _gen([64, 64], None)
        assert t.any() and not t.all()

    def test_out_of_range_values_clamped(self):
        """越界 value_range 夹到 {0, 1}"""
        upper = _gen([64, 64], [0, 255])          # hi 夹到 1 -> 随机
        assert upper.any() and not upper.all()
        assert not _gen([8, 8], [-5, 0]).any()    # lo 夹到 0，hi=0 -> 全 False

    def test_reversed_range_normalized(self):
        """min > max 时归一化而非返回空/报错"""
        t = _gen([64, 64], [1, 0])
        assert t.any() and not t.all()

    def test_deterministic_with_same_seed(self):
        assert torch.equal(_gen([256], [0, 1], seed=7), _gen([256], [0, 1], seed=7))

    def test_different_seed_differs(self):
        assert not torch.equal(_gen([256], [0, 1], seed=7), _gen([256], [0, 1], seed=8))


class TestGenerateFromCaseWithBool:
    """多输入用例中的 bool 输入（对应 MoeGatingTopKSoftmax 的 x + finished）"""

    def test_bool_input_in_mixed_case(self):
        tensors = DataGenerator().generate_input_tensors_from_case(
            input_shapes=[[128, 16], [128]],
            dtypes=['float16', 'bool'],
            value_ranges=[[-1, 1], [0, 1]],
            seed=123,
        )
        x, finished = tensors
        assert x.dtype == torch.float16
        assert finished.dtype == torch.bool
        assert finished.any() and not finished.all(), "finished 恒为 False，bool 分支未生效"
