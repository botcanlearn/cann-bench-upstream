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

"""RoPE 类算子 cos/sin 三角约束的 get_input 测试

测试对象：
1. tasks/level2/apply_rotary_pos_emb/golden.py::get_input
2. tasks/level4/mla_prolog/golden.py::get_input

RoPE 的 cos / sin 是同一组位置角 theta 的余弦与正弦，逐元素满足 cos^2 + sin^2 = 1，
对应保范数的平面旋转。单区间 value_range 只能声明"两个张量各自落在 [-1,1]"，通用生成器
把它们当成互相独立的随机张量，(cos, sin) 落在正方形而非单位圆上。
"""

import importlib.util

import pytest
import torch

from kernel_eval.config import get_project_root


def _load(op_rel):
    path = get_project_root() / "tasks" / op_rel / "golden.py"
    spec = importlib.util.spec_from_file_location(f"_golden_{op_rel.replace('/', '_')}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def rope():
    return _load("level2/apply_rotary_pos_emb")


@pytest.fixture(scope="module")
def mla_prolog():
    return _load("level4/mla_prolog")


def _norm2(cos, sin):
    return (cos.float() ** 2 + sin.float() ** 2)


# ---------------------------------------------------------------------------
# ApplyRotaryPosEmb
# ---------------------------------------------------------------------------


def _rope_inputs(dtype=torch.float32, s=64, d=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    q = ((torch.rand(2, s, 4, d * 2, generator=g) * 2 - 1)).to(dtype)
    k = ((torch.rand(2, s, 4, d * 2, generator=g) * 2 - 1)).to(dtype)
    cos = ((torch.rand(s, d, generator=g) * 2 - 1)).to(dtype)
    sin = ((torch.rand(s, d, generator=g) * 2 - 1)).to(dtype)
    return q, k, cos, sin


class TestRopeIdentity:
    def test_original_data_violates_identity(self, rope):
        _, _, cos, sin = _rope_inputs()
        n = _norm2(cos, sin)
        assert n.min() < 0.5 and n.max() > 1.5, "构造失败：原始数据本应散布在 [0,2]"

    def test_identity_holds_after_rebuild(self, rope):
        q, k, cos, sin = _rope_inputs()
        out = rope.get_input(q, k, cos, sin)
        n = _norm2(out[2], out[3])
        assert torch.allclose(n, torch.ones_like(n), atol=1e-6)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
    def test_identity_within_dtype_precision(self, rope, dtype):
        q, k, cos, sin = _rope_inputs(dtype=dtype)
        out = rope.get_input(q, k, cos, sin)
        n = _norm2(out[2], out[3])
        assert (n - 1).abs().max() < 0.02

    def test_values_stay_in_unit_interval(self, rope):
        q, k, cos, sin = _rope_inputs()
        out = rope.get_input(q, k, cos, sin)
        for t in (out[2], out[3]):
            assert t.min() >= -1.0 and t.max() <= 1.0

    def test_covers_all_four_quadrants(self, rope):
        """随机角覆盖整个单位圆，而非某一象限"""
        q, k, cos, sin = _rope_inputs()
        c, s = rope.get_input(q, k, cos, sin)[2:]
        assert (c > 0).any() and (c < 0).any()
        assert (s > 0).any() and (s < 0).any()

    def test_shape_dtype_preserved(self, rope):
        q, k, cos, sin = _rope_inputs(dtype=torch.float16)
        out = rope.get_input(q, k, cos, sin)
        assert out[2].shape == cos.shape and out[2].dtype == cos.dtype
        assert out[3].shape == sin.shape and out[3].dtype == sin.dtype

    def test_query_key_untouched(self, rope):
        q, k, cos, sin = _rope_inputs()
        out = rope.get_input(q, k, cos, sin)
        assert out[0] is q and out[1] is k

    def test_reproducible(self, rope):
        q, k, cos, sin = _rope_inputs()
        a = rope.get_input(q, k, cos, sin)
        b = rope.get_input(q, k, cos, sin)
        assert torch.equal(a[2], b[2]) and torch.equal(a[3], b[3])

    def test_returns_four_tensors(self, rope):
        q, k, cos, sin = _rope_inputs()
        assert len(rope.get_input(q, k, cos, sin)) == 4

    def test_extra_kwargs_tolerated(self, rope):
        q, k, cos, sin = _rope_inputs()
        rope.get_input(q, k, cos, sin, layout=1, rotaryMode='interleaved', skip2_exist=True)

    def test_golden_runs_after_rebuild(self, rope):
        q, k, cos, sin = _rope_inputs()
        out = rope.get_input(q, k, cos, sin)
        qo, ko = rope.apply_rotary_pos_emb(out[0], out[1], out[2], out[3])
        assert qo.shape == q.shape and ko.shape == k.shape

    def test_rotation_is_norm_preserving(self, rope):
        """重建后 RoPE 恢复成保范数旋转——这正是三角约束的意义"""
        q, k, cos, sin = _rope_inputs(dtype=torch.float32)
        out = rope.get_input(q, k, cos, sin)
        qo, _ = rope.apply_rotary_pos_emb(out[0], out[1], out[2], out[3])
        n_in = out[0].float().pow(2).sum(-1)
        n_out = qo.float().pow(2).sum(-1)
        assert torch.allclose(n_in, n_out, rtol=1e-4, atol=1e-4)


class TestRopeSpecialValuePassthrough:
    def test_nan_case_passed_through(self, rope):
        """c14 value_range=[nan,nan]：NaN 传播覆盖不应被重建抹掉"""
        q, k, cos, sin = _rope_inputs()
        cos = torch.full_like(cos, float('nan'))
        sin = torch.full_like(sin, float('nan'))
        out = rope.get_input(q, k, cos, sin)
        assert out[2] is cos and out[3] is sin

    def test_inf_case_passed_through(self, rope):
        q, k, cos, sin = _rope_inputs()
        cos[0, 0] = float('inf')
        out = rope.get_input(q, k, cos, sin)
        assert out[2] is cos and out[3] is sin

    def test_all_zero_case_passed_through(self, rope):
        """c15 value_range=[0,0]：零值边界覆盖不应被重建抹掉"""
        q, k, cos, sin = _rope_inputs()
        cos = torch.zeros_like(cos)
        sin = torch.zeros_like(sin)
        out = rope.get_input(q, k, cos, sin)
        assert out[2] is cos and out[3] is sin

    def test_mismatched_shape_passed_through(self, rope):
        q, k, cos, sin = _rope_inputs()
        out = rope.get_input(q, k, cos, sin[:, :8])
        assert out[2] is cos


# ---------------------------------------------------------------------------
# MlaProlog
# ---------------------------------------------------------------------------


def _mla_rope(b=2, s=1, dr=16, dtype=torch.float32, seed=0):
    g = torch.Generator().manual_seed(seed)
    sin = ((torch.rand(b, s, dr, generator=g) * 2 - 1)).to(dtype)
    cos = ((torch.rand(b, s, dr, generator=g) * 2 - 1)).to(dtype)
    return sin, cos


def _mla_args(sin, cos):
    """其余 7 个输入用占位张量即可——get_input 对它们只做透传"""
    t = torch.zeros(1)
    return (t, t, t, t, t, t, t, sin, cos)


class TestMlaPrologRopeIdentity:
    def test_identity_holds_after_rebuild(self, mla_prolog):
        sin, cos = _mla_rope()
        out = mla_prolog.get_input(*_mla_args(sin, cos))
        n = _norm2(out[8], out[7])
        assert torch.allclose(n, torch.ones_like(n), atol=1e-6)

    def test_front_and_back_half_share_the_same_angle(self, mla_prolog):
        """全宽 Dr 的前后半必须是同一角度的复制

        本算子的 apply_rope 直接逐元素乘 cos/sin，**没有** level2 那样的内部 repeat，
        配对约定只能由输入自身满足。
        """
        sin, cos = _mla_rope(dr=16)
        out = mla_prolog.get_input(*_mla_args(sin, cos))
        h = 16 // 2
        assert torch.equal(out[7][..., :h], out[7][..., h:])
        assert torch.equal(out[8][..., :h], out[8][..., h:])

    def test_apply_rope_is_norm_preserving(self, mla_prolog):
        """重建后 apply_rope 必须是保范数旋转——这是三角约束的真正意义

        回归：只校验逐元素 cos^2+sin^2=1 是不够的。若在全宽上逐元素独立取角，恒等式
        处处成立，配对 (x[j], x[j+h]) 的模长比却在 0.023~1.996 之间乱跳，整体不是旋转。
        """
        dr = 64
        sin, cos = _mla_rope(b=2, s=4, dr=dr, dtype=torch.float32)
        out = mla_prolog.get_input(*_mla_args(sin, cos))
        g = torch.Generator().manual_seed(1)
        x = torch.randn(2, 4, dr, generator=g)
        y = mla_prolog.apply_rope(x, out[8], out[7])
        h = dr // 2
        n_in = x[..., :h] ** 2 + x[..., h:] ** 2
        n_out = y[..., :h] ** 2 + y[..., h:] ** 2
        assert torch.allclose(n_out, n_in, rtol=1e-5, atol=1e-5)

    def test_odd_dr_passed_through(self, mla_prolog):
        """apply_rope 的 chunk(2) 要求 Dr 为偶数；奇数时配对约定不成立，原样放行"""
        sin, cos = _mla_rope(dr=15)
        out = mla_prolog.get_input(*_mla_args(sin, cos))
        assert out[7] is sin and out[8] is cos

    def test_sin_cos_positions_in_signature_order(self, mla_prolog):
        """签名顺序是 (..., rope_sin, rope_cos)，返回值不能颠倒"""
        sin, cos = _mla_rope()
        out = mla_prolog.get_input(*_mla_args(sin, cos))
        assert len(out) == 9
        assert out[7].shape == sin.shape and out[8].shape == cos.shape

    def test_shape_dtype_preserved(self, mla_prolog):
        sin, cos = _mla_rope(dtype=torch.bfloat16)
        out = mla_prolog.get_input(*_mla_args(sin, cos))
        assert out[7].dtype == torch.bfloat16 and out[8].dtype == torch.bfloat16

    def test_other_inputs_untouched(self, mla_prolog):
        sin, cos = _mla_rope()
        args = _mla_args(sin, cos)
        out = mla_prolog.get_input(*args)
        for i in range(7):
            assert out[i] is args[i]

    def test_all_zero_case_passed_through(self, mla_prolog):
        """c20 value_range=[0,0]"""
        sin, cos = _mla_rope()
        sin, cos = torch.zeros_like(sin), torch.zeros_like(cos)
        out = mla_prolog.get_input(*_mla_args(sin, cos))
        assert out[7] is sin and out[8] is cos

    def test_nan_case_passed_through(self, mla_prolog):
        sin, cos = _mla_rope()
        sin = torch.full_like(sin, float('nan'))
        out = mla_prolog.get_input(*_mla_args(sin, cos))
        assert out[7] is sin and out[8] is cos

    def test_reproducible(self, mla_prolog):
        sin, cos = _mla_rope()
        a = mla_prolog.get_input(*_mla_args(sin, cos))
        b = mla_prolog.get_input(*_mla_args(sin, cos))
        assert torch.equal(a[7], b[7]) and torch.equal(a[8], b[8])

    def test_extra_kwargs_tolerated(self, mla_prolog):
        sin, cos = _mla_rope()
        mla_prolog.get_input(*_mla_args(sin, cos), n_heads=8, skip2_exist=True)
