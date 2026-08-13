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

"""输入数据分布选择（--input-dist）单元测试

测试对象：kernel_eval.data.data_generator.DataGenerator(input_dist=...) 及其 CLI 接线。
默认 input_dist="uniform" 必须与历史行为**逐位一致**——这是本特性的硬约束。
"""

import pytest
import torch

from kernel_eval.config import Config
from kernel_eval.data.data_generator import (
    DataGenerator,
    INPUT_DISTRIBUTIONS,
    normalize_input_dist,
)


def _g(seed=42):
    g = torch.Generator()
    g.manual_seed(seed)
    return g


class TestDistNameNormalization:
    @pytest.mark.parametrize("alias", ["uniform", "u", "UNIFORM", " uniform "])
    def test_uniform_aliases(self, alias):
        assert normalize_input_dist(alias) == "uniform"

    @pytest.mark.parametrize("alias", ["normal", "norm", "gaussian", "n", "NORM", " norm "])
    def test_normal_aliases(self, alias):
        assert normalize_input_dist(alias) == "normal"

    def test_none_defaults_to_uniform(self):
        assert normalize_input_dist(None) == "uniform"

    @pytest.mark.parametrize("bad", ["gauss", "", "abc", 123, "uniform2"])
    def test_unknown_rejected(self, bad):
        with pytest.raises(ValueError, match="不支持的输入分布"):
            normalize_input_dist(bad)

    def test_registry_shape(self):
        """别名表本身自洽：规范名必须出现在自己的别名列表里"""
        for canon, aliases in INPUT_DISTRIBUTIONS.items():
            assert canon in aliases


class TestDefaultUnchanged:
    def test_config_default(self):
        assert Config().input_dist == "uniform"

    def test_generator_default(self):
        assert DataGenerator().input_dist == "uniform"

    def test_explicit_uniform_bit_identical(self):
        a = DataGenerator().generate_input_tensor([4096], 'float32', [-1, 1], generator=_g())
        b = DataGenerator(input_dist='uniform').generate_input_tensor(
            [4096], 'float32', [-1, 1], generator=_g())
        assert torch.equal(a, b)

    def test_uniform_consumes_same_rng_stream(self):
        """随机数消费顺序也须一致，否则同一 case 内后续张量会整体错位"""
        ref = None
        for gen in (DataGenerator(), DataGenerator(input_dist='u')):
            g = _g()
            got = (gen.generate_input_tensor([1000], 'float32', [-1, 1], generator=g),
                   gen.generate_input_tensor([1000], 'float32', [-2, 2], generator=g))
            if ref is None:
                ref = got
            else:
                assert torch.equal(ref[0], got[0]) and torch.equal(ref[1], got[1])


class TestNormalShape:
    def test_is_bell_shaped(self):
        """正态：中间桶远高于两端桶；均匀则各桶相等"""
        t = DataGenerator(input_dist='normal').generate_input_tensor(
            [500, 500], 'float32', [-1, 1], generator=_g()).float()
        h = torch.histc(t, bins=10, min=-1, max=1) / t.numel()
        assert h[4] > 4 * h[0] and h[5] > 4 * h[9]

    def test_uniform_is_flat(self):
        t = DataGenerator(input_dist='uniform').generate_input_tensor(
            [500, 500], 'float32', [-1, 1], generator=_g()).float()
        h = torch.histc(t, bins=10, min=-1, max=1) / t.numel()
        assert (h - 0.1).abs().max() < 0.01

    def test_mu_and_sigma_from_value_range(self):
        """mu=(min+max)/2, sigma=(max-min)/6"""
        t = DataGenerator(input_dist='normal').generate_input_tensor(
            [200000], 'float32', [5.0, 7.0], generator=_g())
        assert abs(t.mean().item() - 6.0) < 0.01
        assert abs(t.std().item() - 2.0 / 6.0) < 0.01

    def test_std_smaller_than_uniform(self):
        kw = dict(shape=[200, 200], dtype='float32', value_range=[-1, 1])
        u = DataGenerator(input_dist='uniform').generate_input_tensor(generator=_g(), **kw)
        n = DataGenerator(input_dist='normal').generate_input_tensor(generator=_g(), **kw)
        assert n.std().item() < u.std().item()


class TestBoundsHonored:
    def test_never_exceeds_declared_range(self):
        lo, hi = -1.0, 1.0
        t = DataGenerator(input_dist='normal').generate_input_tensor(
            [300, 300], 'float32', [lo, hi], generator=_g())
        assert t.min().item() >= lo and t.max().item() <= hi

    def test_boundary_pileup_near_theoretical(self):
        """±3σ 截断 → 约 0.27% 的样本堆在边界上"""
        lo, hi = -1.0, 1.0
        t = DataGenerator(input_dist='normal').generate_input_tensor(
            [200000], 'float32', [lo, hi], generator=_g()).float()
        share = float(((t <= lo + 1e-6) | (t >= hi - 1e-6)).float().mean())
        assert 0.001 < share < 0.01

    @pytest.mark.parametrize("dtype", ['float16', 'bfloat16', 'float32'])
    def test_dtype_and_shape_preserved(self, dtype):
        t = DataGenerator(input_dist='normal').generate_input_tensor(
            [64, 64], dtype, [-1, 1], generator=_g())
        assert t.shape == torch.Size([64, 64])
        assert torch.isfinite(t.float()).all()

    def test_huge_range_does_not_overflow(self):
        """range 接近 dtype 上限时不得溢出为 inf"""
        t = DataGenerator(input_dist='normal').generate_input_tensor(
            [10000], 'float16', [-65504, 65504], generator=_g())
        assert torch.isfinite(t.float()).all()


class TestUnaffectedPaths:
    def test_int_dtype_unaffected(self):
        """整数输入取值分布常带语义约束（索引需覆盖整个维度），不应被改变"""
        kw = dict(shape=[10000], dtype='int32', value_range=[0, 1023])
        a = DataGenerator(input_dist='uniform').generate_input_tensor(generator=_g(), **kw)
        b = DataGenerator(input_dist='normal').generate_input_tensor(generator=_g(), **kw)
        assert torch.equal(a, b)

    def test_bool_dtype_unaffected(self):
        """bool 走 _gen_bool，与浮点分布选项无关"""
        kw = dict(shape=[10000], dtype='bool', value_range=[0, 1])
        a = DataGenerator(input_dist='uniform').generate_input_tensor(generator=_g(), **kw)
        b = DataGenerator(input_dist='normal').generate_input_tensor(generator=_g(), **kw)
        assert a.dtype == torch.bool
        assert torch.equal(a, b)

    @pytest.mark.parametrize("vr", [
        [float('-inf'), float('inf')],
        [float('nan'), float('nan')],
        [3.0, 3.0],
    ])
    def test_special_and_constant_ranges_unaffected(self, vr):
        """inf / nan 压力用例与常数区间不走分布分支"""
        a = DataGenerator(input_dist='uniform').generate_input_tensor(
            [1000], 'float32', vr, generator=_g())
        b = DataGenerator(input_dist='normal').generate_input_tensor(
            [1000], 'float32', vr, generator=_g())
        assert bool(((a == b) | (torch.isnan(a) & torch.isnan(b))).all())


class TestDeterminism:
    def test_same_seed_reproducible(self):
        kw = dict(shape=[2000], dtype='float32', value_range=[-1, 1])
        a = DataGenerator(input_dist='normal').generate_input_tensor(generator=_g(7), **kw)
        b = DataGenerator(input_dist='normal').generate_input_tensor(generator=_g(7), **kw)
        assert torch.equal(a, b)

    def test_different_seed_differs(self):
        kw = dict(shape=[2000], dtype='float32', value_range=[-1, 1])
        a = DataGenerator(input_dist='normal').generate_input_tensor(generator=_g(7), **kw)
        b = DataGenerator(input_dist='normal').generate_input_tensor(generator=_g(8), **kw)
        assert not torch.equal(a, b)


class TestMultiInputCase:
    def test_every_float_input_uses_the_distribution(self):
        tensors = DataGenerator(input_dist='norm').generate_input_tensors_from_case(
            input_shapes=[[300, 300], [300, 300]],
            dtypes=['float32', 'float32'],
            value_ranges=[[-1, 1], [-1, 1]],
            seed=0,
        )
        for t in tensors:
            h = torch.histc(t.float(), bins=10, min=-1, max=1) / t.numel()
            assert h[4] > 4 * h[0]

    def test_int_input_stays_uniform_in_range(self):
        """整数输入的**分布**不变（仍均匀覆盖整个区间）

        注意具体取值会变：randn 与 rand 的随机数消耗不同，共享同一 generator 的
        后续张量随之平移。分布性质与取值边界不受影响——索引仍均匀覆盖整个维度，
        这正是"仅浮点"这条边界要保住的东西。
        """
        kw = dict(input_shapes=[[20000], [20000]], dtypes=['float32', 'int32'],
                  value_ranges=[[-1, 1], [0, 255]], seed=0)
        u = DataGenerator(input_dist='uniform').generate_input_tensors_from_case(**kw)
        n = DataGenerator(input_dist='normal').generate_input_tensors_from_case(**kw)
        assert not torch.equal(u[0], n[0]), "float 输入应受影响"

        for t in (u[1], n[1]):
            assert int(t.min()) == 0 and int(t.max()) == 255
            counts = torch.bincount(t.long() // 32, minlength=8).float() / t.numel()
            assert counts.min() > 0.10 and counts.max() < 0.15

    def test_int_only_case_bit_identical(self):
        """全整数输入的用例：没有浮点张量搅动随机数流，取值逐位不变"""
        kw = dict(input_shapes=[[5000], [5000]], dtypes=['int32', 'int64'],
                  value_ranges=[[0, 255], [0, 1023]], seed=0)
        u = DataGenerator(input_dist='uniform').generate_input_tensors_from_case(**kw)
        n = DataGenerator(input_dist='normal').generate_input_tensors_from_case(**kw)
        for x, y in zip(u, n):
            assert torch.equal(x, y)


class TestCliWiring:
    """CLI -> Config -> DataGenerator，以及 eval-child 子进程透传"""

    def _parse(self, argv):
        from kernel_eval.cli import create_parser
        return create_parser().parse_args(argv)

    def test_eval_default(self):
        assert self._parse(['eval', '--source-dir', 'x']).input_dist == 'uniform'

    @pytest.mark.parametrize("val", ['normal', 'norm', 'gaussian', 'u'])
    def test_eval_accepts_aliases(self, val):
        assert self._parse(['eval', '--source-dir', 'x', '--input-dist', val]).input_dist == val

    def test_eval_rejects_unknown(self):
        with pytest.raises(SystemExit):
            self._parse(['eval', '--source-dir', 'x', '--input-dist', 'gauss'])

    def test_eval_child_has_the_flag(self):
        """子进程入口必须也认这个参数，否则多卡并行下会静默退回默认分布"""
        base = ['eval-child', '--device-id', '0', '--cases-file', 'c.json', '--output', 'o.json']
        assert self._parse(base).input_dist == 'uniform'
        assert self._parse(base + ['--input-dist', 'norm']).input_dist == 'norm'

    def test_child_cmd_forwards_non_default_only(self):
        import inspect
        from kernel_eval.eval.process_pool import ProcessPoolCoordinator
        src = inspect.getsource(ProcessPoolCoordinator._build_child_cmd)
        assert '--input-dist' in src
        assert 'input_dist != "uniform"' in src

    def test_evaluator_passes_config_to_generator(self):
        import inspect
        from kernel_eval.eval.evaluator import Evaluator
        src = inspect.getsource(Evaluator.__init__)
        assert 'input_dist' in src

    def test_staged_eval_has_the_flag(self):
        """staged_eval 是独立入口（scripts/run_evaluation.sh --staged 走它），
        parser 与 _make_config 都必须接上，否则标准三阶段评测里该特性不可达。
        """
        import kernel_eval.staged_eval as se
        p = se.create_parser()
        assert p.parse_args([]).input_dist == 'uniform'
        assert p.parse_args(['--input-dist', 'norm']).input_dist == 'norm'

    def test_staged_eval_make_config_sets_input_dist(self):
        import inspect
        import kernel_eval.staged_eval as se
        src = inspect.getsource(se)
        assert 'cfg.input_dist' in src, "_make_config 未把 args.input_dist 写入 Config"

    def test_shell_wrapper_exposes_the_flag(self):
        """run_evaluation.sh 是文档里的主入口，需能透传给 staged_eval"""
        from kernel_eval.config import get_project_root
        sh = (get_project_root() / 'scripts' / 'run_evaluation.sh').read_text()
        assert '--input-dist)' in sh, "shell 未解析 --input-dist"
        assert 'INPUT_DIST=' in sh, "shell 未初始化 INPUT_DIST"
        assert '--input-dist ${INPUT_DIST}' in sh, "shell 未透传 --input-dist"

    def test_choices_derived_from_registry(self):
        """argparse choices 必须由 INPUT_DISTRIBUTIONS 派生，避免两处清单漂移"""
        from kernel_eval.data.data_generator import INPUT_DIST_CHOICES, INPUT_DISTRIBUTIONS
        expected = sorted({a for al in INPUT_DISTRIBUTIONS.values() for a in al})
        assert INPUT_DIST_CHOICES == expected
        # 两个 parser 都必须接受 registry 里的每一个别名
        import kernel_eval.staged_eval as se
        from kernel_eval.cli import create_parser
        for alias in INPUT_DIST_CHOICES:
            assert create_parser().parse_args(
                ['eval', '--source-dir', 'x', '--input-dist', alias]).input_dist == alias
            assert se.create_parser().parse_args(['--input-dist', alias]).input_dist == alias
