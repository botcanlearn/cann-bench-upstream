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
bench_lab/kernel_bench/level5/batched_svd 结构化精度校验回归测试

背景（MR #273 检视意见）：SVD 若仅逐元素比较 u/s/v 且阈值宽松，会接受数学上无效的分解
（如 U_bad = 1.04·U 时 S/V 正确、逐元素误差 4% 仍可通过，但 U_bad^T U_bad = 1.0816 I）。
本测试用仓库真实 checker（compare_tensors）验证 golden.get_input / get_output 方案：
1. get_input 产出 cond=10、谱间隔恒定的良条件矩阵
2. fp32 参考实现（LAPACK）在 proto 声明阈值下通过
3. 缩放 / 非正交 / 未规范化符号 / U-V 列错配 / S 乱序 五类结构性错误均被拒
"""

import importlib.util

import pytest
import torch
import yaml

from kernel_eval.config import get_project_root
from kernel_eval.utils.compare import compare_tensors


_OP_DIR = get_project_root() / "bench_lab" / "kernel_bench" / "level5" / "batched_svd"

pytestmark = pytest.mark.skipif(not (_OP_DIR / "golden.py").is_file(),
                                reason="bench_lab/kernel_bench/level5/batched_svd 不在本 checkout 中")


@pytest.fixture(scope="module")
def golden():
    spec = importlib.util.spec_from_file_location("batched_svd_golden", _OP_DIR / "golden.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def fp32_threshold():
    proto = yaml.safe_load((_OP_DIR / "proto.yaml").read_text(encoding="utf-8"))
    return float(proto["operator"]["precision_thresholds"]["float32"])


def _make(golden, B, M, N, seed=0):
    torch.manual_seed(seed)
    a = torch.rand(B, M, N, dtype=torch.float32) * 2 - 1
    (a_c,) = golden.get_input(a)
    return a_c


def _check(golden, a_c, u, s, v, thr):
    """用仓库 checker 比对候选 (u,s,v) 与 fp64 oracle，返回 CompareResult。"""
    gold = golden.get_output(*golden.batched_svd_oracle(a_c.double()))
    native = golden.get_output(*golden.batched_svd(a_c))
    cand = golden.get_output(u, s, v)
    return compare_tensors(cand, gold, dtype="float32", threshold=thr, native_output=native)


class TestGetInputConditioning:

    @pytest.mark.parametrize("B,M,N", [(4, 32, 16), (2, 128, 64), (3, 64, 64), (2, 8, 1)])
    def test_prescribed_spectrum_and_shape(self, golden, B, M, N):
        a_c = _make(golden, B, M, N)
        assert a_c.shape == (B, M, N) and a_c.dtype == torch.float32
        s = torch.linalg.svdvals(a_c.double())                       # [B, N]
        ratio = s / s[:, :1]
        expect = 1.0 - 0.9 * torch.arange(N, dtype=torch.float64) / max(N - 1, 1)
        if N == 1:
            expect = torch.ones(1, dtype=torch.float64)
        assert torch.allclose(ratio, expect.expand_as(ratio), atol=1e-5)
        if N > 1:
            assert torch.allclose(s[:, 0] / s[:, -1], torch.full((B,), 10.0, dtype=torch.float64), rtol=1e-4)

    def test_scale_follows_value_range(self, golden):
        torch.manual_seed(1)
        a_small = (torch.rand(2, 16, 8) * 2 - 1) * 0.01
        a_big = (torch.rand(2, 16, 8) * 2 - 1) * 100.0
        s_small = torch.linalg.svdvals(golden.get_input(a_small)[0].double())[:, 0]
        s_big = torch.linalg.svdvals(golden.get_input(a_big)[0].double())[:, 0]
        assert (s_small < 0.02).all() and (s_big > 50).all()


class TestStructuralAcceptance:

    @pytest.mark.parametrize("B,M,N", [(4, 32, 16), (2, 256, 64), (2, 512, 128), (4, 64, 64)])
    def test_fp32_reference_passes(self, golden, fp32_threshold, B, M, N):
        a_c = _make(golden, B, M, N)
        u, s, v = golden.batched_svd(a_c)
        r = _check(golden, a_c, u, s, v, fp32_threshold)
        assert r.passed, r.error_msg

    def test_get_output_shapes(self, golden):
        a_c = _make(golden, 2, 32, 16)
        recon, s, gram = golden.get_output(*golden.batched_svd(a_c))
        assert recon.shape == (2, 32, 16) and s.shape == (2, 16) and gram.shape == (2, 32, 16)
        # 良好分解：重构 ≈ A，Gram+1 ≈ I+1
        assert torch.allclose(recon, a_c, atol=1e-4)
        eye1 = torch.eye(16).repeat(2, 2, 1) + 1.0
        assert torch.allclose(gram, eye1, atol=1e-4)


class TestStructuralRejection:
    """每类结构性错误都必须被仓库 checker 拒绝（proto 声明阈值下）。"""

    @pytest.fixture(scope="class")
    def decomposed(self, golden):
        a_c = _make(golden, 3, 64, 32, seed=7)
        u, s, v = golden.batched_svd(a_c)
        return a_c, u, s, v

    def test_scaled_u_rejected(self, golden, fp32_threshold, decomposed):
        a_c, u, s, v = decomposed
        assert not _check(golden, a_c, u * 1.04, s, v, fp32_threshold).passed

    @pytest.mark.parametrize("eps", [0.01, 0.003])
    def test_non_orthogonal_u_rejected(self, golden, fp32_threshold, decomposed, eps):
        a_c, u, s, v = decomposed
        torch.manual_seed(3)
        assert not _check(golden, a_c, u + eps * torch.randn_like(u), s, v, fp32_threshold).passed

    def test_unnormalized_sign_rejected(self, golden, fp32_threshold, decomposed):
        """u_j 与 v_j 同时取反仍是合法分解（重构不变），但违反符号规范化约定 → 拒。"""
        a_c, u, s, v = decomposed
        assert not _check(golden, a_c, -u, s, -v, fp32_threshold).passed

    def test_mismatched_uv_pairing_rejected(self, golden, fp32_threshold, decomposed):
        a_c, u, s, v = decomposed
        torch.manual_seed(5)
        perm = torch.randperm(v.shape[-1])
        assert not _check(golden, a_c, u, s, v[:, :, perm], fp32_threshold).passed

    def test_unsorted_singular_values_rejected(self, golden, fp32_threshold, decomposed):
        a_c, u, s, v = decomposed
        assert not _check(golden, a_c, u.flip(-1), s.flip(-1), v.flip(-1), fp32_threshold).passed
