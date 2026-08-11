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

"""Scatter get_input 单元测试

测试对象：tasks/level2/scatter/golden.py::get_input
背景：torch.scatter_ 在 index 沿 dim 有重复项时结果 nondeterministic（同一目标位置
被多次写入，留下哪个取决于实现的写入顺序）。通用生成器按 randint 独立采样索引，
重复几乎必然，导致 reduce=None 用例的期望输出根本不唯一。get_input 把索引重建为
沿 dim 互异；带可交换 reduce 的用例原样放行。
"""

import importlib.util

import pytest
import torch

from kernel_eval.config import get_project_root


@pytest.fixture(scope="module")
def get_input():
    path = get_project_root() / "tasks" / "level2" / "scatter" / "golden.py"
    spec = importlib.util.spec_from_file_location("_scatter_golden", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.get_input


def _dup_count(idx: torch.Tensor, dim: int) -> int:
    """沿 dim 的重复元素个数"""
    dim_n = dim if dim >= 0 else idx.dim() + dim
    flat = idx.movedim(dim_n, -1).reshape(-1, idx.shape[dim_n]).long()
    srt, _ = flat.sort(dim=-1)
    return int((srt[:, 1:] == srt[:, :-1]).sum())


def _both_write_orders(data, idx, upd, dim):
    """两种同样符合 scatter 语义的实现：正序写（后写赢）与逆序写（先写赢）"""
    dim_n = dim if dim >= 0 else idx.dim() + dim
    i64 = idx.long()
    a = data.clone()
    a.scatter_(dim_n, i64, upd)
    rev = torch.arange(i64.shape[dim_n] - 1, -1, -1)
    b = data.clone()
    b.scatter_(dim_n, i64.index_select(dim_n, rev), upd.index_select(dim_n, rev))
    return a, b


def _case(shape_data, shape_idx, D, seed=0):
    g = torch.Generator().manual_seed(seed)
    data = torch.rand(shape_data, generator=g)
    idx = torch.randint(0, D, shape_idx, generator=g, dtype=torch.int32)
    upd = torch.rand(shape_idx, generator=g)
    return data, idx, upd


class TestUniqueness:
    def test_duplicates_removed_dim0(self, get_input):
        data, idx, upd = _case((64, 32), (16, 32), 64)
        assert _dup_count(idx, 0) > 0, "构造失败：原始索引本应有重复"
        _, new_idx, _ = get_input(data, idx, upd, dim=0, reduce=None)
        assert _dup_count(new_idx, 0) == 0

    def test_duplicates_removed_last_dim(self, get_input):
        data, idx, upd = _case((32, 64), (32, 16), 64)
        _, new_idx, _ = get_input(data, idx, upd, dim=-1, reduce=None)
        assert _dup_count(new_idx, -1) == 0

    def test_duplicates_removed_middle_dim(self, get_input):
        data, idx, upd = _case((4, 32, 8), (4, 12, 8), 32)
        _, new_idx, _ = get_input(data, idx, upd, dim=1, reduce=None)
        assert _dup_count(new_idx, 1) == 0

    def test_full_length_is_a_permutation(self, get_input):
        """n == D 时结果应为 [0, D) 的完整置换"""
        data, idx, upd = _case((16, 8), (16, 8), 16)
        _, new_idx, _ = get_input(data, idx, upd, dim=0, reduce=None)
        for col in range(8):
            assert sorted(new_idx[:, col].tolist()) == list(range(16))


class TestDeterminism:
    def test_result_independent_of_write_order(self, get_input):
        """核心目标：两种合规实现必须给出同一结果"""
        data, idx, upd = _case((64, 32), (16, 32), 64)
        a, b = _both_write_orders(data, idx, upd, 0)
        assert not torch.equal(a, b), "构造失败：原始索引本应产生实现相关的结果"
        d2, i2, u2 = get_input(data, idx, upd, dim=0, reduce=None)
        a2, b2 = _both_write_orders(d2, i2, u2, 0)
        assert torch.equal(a2, b2)

    def test_reproducible_across_calls(self, get_input):
        """固定种子：跨 eval 运行必须可复现"""
        data, idx, upd = _case((64, 32), (16, 32), 64)
        _, i1, _ = get_input(data, idx, upd, dim=0, reduce=None)
        _, i2, _ = get_input(data, idx, upd, dim=0, reduce=None)
        assert torch.equal(i1, i2)


class TestContractPreserved:
    def test_indices_in_range(self, get_input):
        data, idx, upd = _case((64, 32), (16, 32), 64)
        _, new_idx, _ = get_input(data, idx, upd, dim=0, reduce=None)
        assert int(new_idx.min()) >= 0
        assert int(new_idx.max()) < data.shape[0]

    def test_shape_and_dtype_unchanged(self, get_input):
        data, idx, upd = _case((64, 32), (16, 32), 64)
        _, new_idx, _ = get_input(data, idx, upd, dim=0, reduce=None)
        assert new_idx.shape == idx.shape
        assert new_idx.dtype == idx.dtype

    def test_data_and_updates_untouched(self, get_input):
        data, idx, upd = _case((64, 32), (16, 32), 64)
        d2, _, u2 = get_input(data, idx, upd, dim=0, reduce=None)
        assert d2 is data and u2 is upd

    def test_returns_three_tensors_in_signature_order(self, get_input):
        data, idx, upd = _case((64, 32), (16, 32), 64)
        out = get_input(data, idx, upd, dim=0, reduce=None)
        assert len(out) == 3
        assert out[0].shape == data.shape and out[2].shape == upd.shape

    def test_extra_kwargs_tolerated(self, get_input):
        """evaluator 会额外塞入 skip2_exist 等 kwargs"""
        data, idx, upd = _case((64, 32), (16, 32), 64)
        get_input(data, idx, upd, dim=0, reduce=None, skip2_exist=True)


class TestCommutativeReduceUntouched:
    @pytest.mark.parametrize("reduce", ["add", "multiply", "amin", "amax"])
    def test_reduce_cases_passthrough(self, get_input, reduce):
        """可交换归约下重复索引良定义，不应改变其数据分布"""
        data, idx, upd = _case((64, 32), (16, 32), 64)
        _, new_idx, _ = get_input(data, idx, upd, dim=0, reduce=reduce)
        assert new_idx is idx

    def test_update_string_is_treated_as_none(self, get_input):
        data, idx, upd = _case((64, 32), (16, 32), 64)
        _, new_idx, _ = get_input(data, idx, upd, dim=0, reduce='update')
        assert _dup_count(new_idx, 0) == 0


class TestDegenerateShapes:
    def test_single_index_along_dim(self, get_input):
        """n == 1：本就不可能重复，原样返回"""
        data, idx, upd = _case((8, 4), (1, 4), 8)
        _, new_idx, _ = get_input(data, idx, upd, dim=0, reduce=None)
        assert new_idx is idx

    def test_n_greater_than_d_passthrough(self, get_input):
        """抽屉原理：互异索引不存在时保持原样而非报错"""
        data, idx, upd = _case((4, 8), (16, 8), 4)
        _, new_idx, _ = get_input(data, idx, upd, dim=0, reduce=None)
        assert new_idx is idx
