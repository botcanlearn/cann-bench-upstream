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

"""_MaskedStats 单元测试

_MaskedStats 替代 `values[mask]` 之后取 count/mean/max 的写法，用分块归约避开那份
与输入等长的 gather。这里对着朴素实现逐个场景比对，重点是分块边界。
"""

import pytest
import torch

from kernel_eval.utils.compare import _MASKED_CHUNK, _MaskedStats


def _reference(values, mask):
    """朴素实现：物化 gather 再归约，作为对照基准"""
    selected = values.reshape(-1)[mask.reshape(-1)]
    if selected.numel() == 0:
        return 0, 0.0, 0.0
    return selected.numel(), float(selected.mean()), float(selected.max())


def _assert_matches(values, mask, chunk, *, rel=1e-12):
    stats = _MaskedStats(values, mask, chunk=chunk)
    count, mean, maximum = _reference(values, mask)
    assert len(stats) == count
    # max 是逐块取 max 再取 max，与整段 max 逐位相同
    assert stats.max() == maximum
    # mean 的求和顺序变了，fp64 下允许几个 ULP
    assert stats.mean() == pytest.approx(mean, rel=rel, abs=1e-300)


class TestMaskedStatsAgainstNaive:
    """与朴素 values[mask] 实现逐场景比对"""

    @pytest.mark.parametrize("chunk", [1, 2, 7, 64, 1000])
    def test_chunk_sizes_do_not_change_results(self, chunk):
        torch.manual_seed(7)
        values = torch.rand(1000, dtype=torch.float64) * 10
        mask = torch.rand(1000) > 0.5
        _assert_matches(values, mask, chunk)

    @pytest.mark.parametrize("n", [0, 1, 15, 16, 17, 31, 32, 33])
    def test_tail_and_boundary_lengths(self, n):
        """块长取 16，覆盖不足一块 / 整除 / 有余数三种收尾"""
        torch.manual_seed(n + 1)
        values = torch.rand(n, dtype=torch.float64)
        mask = torch.rand(n) > 0.4 if n else torch.zeros(0, dtype=torch.bool)
        _assert_matches(values, mask, 16)

    def test_all_true_mask(self):
        values = torch.arange(100, dtype=torch.float64)
        _assert_matches(values, torch.ones(100, dtype=torch.bool), 16)

    def test_all_false_mask_is_empty(self):
        values = torch.arange(100, dtype=torch.float64)
        stats = _MaskedStats(values, torch.zeros(100, dtype=torch.bool), chunk=16)
        assert len(stats) == 0
        # 空集合的 mean/max 只在调用方已经用 len() 守卫过时才会被读到，
        # 这里定义成 0.0 而不是抛异常/NaN，避免守卫漏掉时静默产生 NaN
        assert stats.mean() == 0.0
        assert stats.max() == 0.0

    def test_whole_chunks_selected_nothing(self):
        """整块被 mask 掉时该块要被跳过，不能污染 count/max"""
        values = torch.arange(64, dtype=torch.float64)
        mask = torch.zeros(64, dtype=torch.bool)
        mask[48:] = True                      # 前三块全空，只有第四块有值
        _assert_matches(values, mask, 16)
        stats = _MaskedStats(values, mask, chunk=16)
        assert len(stats) == 16
        assert stats.max() == 63.0

    def test_negative_values_max_is_not_zero(self):
        """全负值时 max 必须是最大的负数，不能被 0.0 的空值兜底顶掉"""
        values = -torch.arange(1, 65, dtype=torch.float64)
        stats = _MaskedStats(values, torch.ones(64, dtype=torch.bool), chunk=16)
        assert stats.max() == -1.0

    def test_multidimensional_input(self):
        """调用点传进来的是与输出同形的张量，不保证一维"""
        torch.manual_seed(11)
        values = torch.rand(8, 16, 4, dtype=torch.float64)
        mask = torch.rand(8, 16, 4) > 0.3
        _assert_matches(values, mask, 20)

    def test_non_contiguous_input(self):
        torch.manual_seed(13)
        base = torch.rand(200, dtype=torch.float64)
        values = base[::2]
        mask = torch.rand(100) > 0.5
        _assert_matches(values, mask, 16)

    def test_inf_values_propagate_to_max(self):
        values = torch.tensor([1.0, float("inf"), 2.0, 3.0], dtype=torch.float64)
        stats = _MaskedStats(values, torch.ones(4, dtype=torch.bool), chunk=2)
        assert stats.max() == float("inf")


class TestMaskedStatsSummationOrder:
    """mean 的求和顺序变化只应停留在 ULP 量级"""

    def test_mean_matches_torch_to_a_few_ulp(self):
        torch.manual_seed(17)
        n = 3 * _MASKED_CHUNK // 4          # 跨块但不整除
        values = torch.rand(n, dtype=torch.float64) + 1.0
        mask = torch.ones(n, dtype=torch.bool)

        stats = _MaskedStats(values, mask)
        reference = float(values.mean())
        deviation = abs(stats.mean() - reference) / abs(reference)
        # fp64 eps = 2.2e-16；实测语料上最坏 6.5e-16，这里留一位数量级余量
        assert deviation < 1e-14, f"mean 偏差 {deviation:.3e} 超出 ULP 量级"

    def test_default_chunk_is_used_when_not_given(self):
        values = torch.ones(10, dtype=torch.float64)
        assert _MaskedStats(values, torch.ones(10, dtype=torch.bool)).count == 10


class TestMaskedStatsDoesNotMaterialiseGather:
    """峰值不随输入长度线性增长——这是整个类存在的理由"""

    def test_peak_allocation_is_bounded_by_chunk(self):
        # 拦 aten 层的分配，只看 CPU 端**新** storage 的最大单块大小。
        # reshape / 切片返回的是视图，共享输入的 storage，按 data_ptr 排除掉，
        # 否则会把输入自己的大小当成一次分配。
        from torch.utils._python_dispatch import TorchDispatchMode
        from torch.utils._pytree import tree_flatten

        class _Watch(TorchDispatchMode):
            def __init__(self, ignore_ptrs):
                self.ignore = ignore_ptrs
                self.largest = 0

            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                out = func(*args, **(kwargs or {}))
                for obj in tree_flatten(out)[0]:
                    if not isinstance(obj, torch.Tensor) or obj.device.type != "cpu":
                        continue
                    storage = obj.untyped_storage()
                    if storage.data_ptr() in self.ignore:
                        continue
                    self.largest = max(self.largest, storage.nbytes())
                return out

        n = 1 << 20
        chunk = 1 << 12
        values = torch.rand(n, dtype=torch.float64)
        mask = torch.ones(n, dtype=torch.bool)

        watch = _Watch({values.untyped_storage().data_ptr(),
                        mask.untyped_storage().data_ptr()})
        with watch:
            _MaskedStats(values, mask, chunk=chunk)

        # 单块 gather 至多 chunk * 8 字节；朴素实现会一次分配 n * 8 = 8MB
        assert watch.largest <= chunk * 8, (
            f"最大单次分配 {watch.largest} 字节，超过一个 chunk（{chunk * 8} 字节）"
        )
        assert watch.largest < n * 8
