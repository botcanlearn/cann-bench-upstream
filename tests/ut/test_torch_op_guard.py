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
TorchOpGuard 单元测试

覆盖：
- 高层 API 命中（torch.matmul / functional.linear 等）—— sanity
- aten 变体 / @ overload / Tensor 方法 命中 —— **绕过漏洞回归测试**
- pause() context manager 暂停守卫 —— **freq-boost false positive 回归测试**
- mode='warn' / 'block' / 'off' 三种模式行为
"""

import pytest
import torch

from kernel_eval.security.torch_op_guard import (
    BUILTIN_COMPUTE_OPS,
    TorchOpGuard,
)


# ============================================================================
# Sanity: 高层 API 应当被命中
# ============================================================================
class TestHighLevelApiCaught:
    """高层 torch.* / nn.functional.* API 应当被命中（既有功能不能回归）"""

    def test_torch_matmul_caught(self):
        g = TorchOpGuard(mode="warn")
        a, b = torch.rand(8, 8), torch.rand(8, 8)
        with g:
            _ = torch.matmul(a, b)
        assert len(g.forbidden_calls) == 1
        assert "matmul" in g.forbidden_calls[0]

    def test_torch_mm_caught(self):
        g = TorchOpGuard(mode="warn")
        a, b = torch.rand(8, 8), torch.rand(8, 8)
        with g:
            _ = torch.mm(a, b)
        assert len(g.forbidden_calls) == 1

    def test_torch_bmm_caught(self):
        g = TorchOpGuard(mode="warn")
        a, b = torch.rand(2, 8, 8), torch.rand(2, 8, 8)
        with g:
            _ = torch.bmm(a, b)
        assert len(g.forbidden_calls) == 1

    def test_nn_functional_linear_caught(self):
        g = TorchOpGuard(mode="warn")
        x = torch.rand(4, 8)
        w = torch.rand(16, 8)
        with g:
            _ = torch.nn.functional.linear(x, w)
        assert len(g.forbidden_calls) >= 1

    def test_nn_functional_softmax_caught(self):
        g = TorchOpGuard(mode="warn")
        x = torch.rand(4, 8)
        with g:
            _ = torch.nn.functional.softmax(x, dim=-1)
        assert len(g.forbidden_calls) >= 1


# ============================================================================
# Sanity: 非禁用 API 不应被命中
# ============================================================================
class TestNonForbiddenNotCaught:
    """元数据 / 普通运算（add / view / to 等）不在禁用集合，不应触发计数"""

    def test_add_not_caught(self):
        g = TorchOpGuard(mode="warn")
        a, b = torch.rand(8, 8), torch.rand(8, 8)
        with g:
            _ = torch.add(a, b)
        assert g.forbidden_calls == []

    def test_view_not_caught(self):
        g = TorchOpGuard(mode="warn")
        a = torch.rand(16)
        with g:
            _ = a.view(4, 4)
        assert g.forbidden_calls == []


# ============================================================================
# 绕过漏洞回归测试：aten 变体 / @ / Tensor 方法
# ============================================================================
class TestBypassDetection:
    """
    用户可能尝试通过 aten 层 / operator overload / Tensor method 调 matmul
    来绕过守卫。这些调用 TorchFunctionMode 实际能拦截，但 _qualified_name 必须
    把它们归一化到禁用集合里的高层名字才能命中。

    在修复前这些测试 FAIL（current _qualified_name 不做 normalization）。
    """

    def test_aten_matmul_caught(self):
        """torch.ops.aten.matmul 绕过尝试 — 必须被识别为 matmul"""
        g = TorchOpGuard(mode="warn")
        a, b = torch.rand(8, 8), torch.rand(8, 8)
        with g:
            _ = torch.ops.aten.matmul(a, b)
        assert len(g.forbidden_calls) >= 1, (
            "aten.matmul 应被识别为禁用 matmul；当前实现 forbidden_calls="
            f"{g.forbidden_calls}"
        )

    def test_aten_mm_caught(self):
        """torch.ops.aten.mm 绕过尝试"""
        g = TorchOpGuard(mode="warn")
        a, b = torch.rand(8, 8), torch.rand(8, 8)
        with g:
            _ = torch.ops.aten.mm(a, b)
        assert len(g.forbidden_calls) >= 1

    def test_aten_mm_default_overload_caught(self):
        """torch.ops.aten.mm.default 完整 overload 路径"""
        g = TorchOpGuard(mode="warn")
        a, b = torch.rand(8, 8), torch.rand(8, 8)
        with g:
            _ = torch.ops.aten.mm.default(a, b)
        assert len(g.forbidden_calls) >= 1

    def test_at_operator_caught(self):
        """a @ b operator overload"""
        g = TorchOpGuard(mode="warn")
        a, b = torch.rand(8, 8), torch.rand(8, 8)
        with g:
            _ = a @ b
        assert len(g.forbidden_calls) >= 1, (
            "a @ b 应被识别为 matmul；当前 forbidden_calls=" f"{g.forbidden_calls}"
        )

    def test_tensor_matmul_method_caught(self):
        """a.matmul(b) Tensor method"""
        g = TorchOpGuard(mode="warn")
        a, b = torch.rand(8, 8), torch.rand(8, 8)
        with g:
            _ = a.matmul(b)
        assert len(g.forbidden_calls) >= 1

    def test_block_mode_raises_on_aten(self):
        """block 模式下 aten 变体也必须抛 RuntimeError"""
        a, b = torch.rand(8, 8), torch.rand(8, 8)
        with pytest.raises(RuntimeError, match="SECURITY"):
            with TorchOpGuard(mode="block"):
                _ = torch.ops.aten.matmul(a, b)


# ============================================================================
# false positive 回归测试：pause() 暂停守卫
# ============================================================================
class TestPauseContextManager:
    """
    harness 内部预热阶段 (perf_eval._boost_freq_and_clear_cache) 会调用
    torch.matmul 提升 NPU 频率。这些调用不是 candidate kernel 真实计算，
    必须通过 TorchOpGuard.pause() 显式排除，否则会产生 false positive
    （见 PR #102/#107 上下游 user feedback）。
    """

    def test_pause_suppresses_matmul(self):
        """pause() 期间的 matmul 调用不计数"""
        g = TorchOpGuard(mode="warn")
        a, b = torch.rand(8, 8), torch.rand(8, 8)
        with g:
            with TorchOpGuard.pause():
                _ = torch.matmul(a, b)
        assert g.forbidden_calls == [], (
            f"pause() 期间 matmul 不应被计数，got {g.forbidden_calls}"
        )

    def test_pause_then_resume(self):
        """pause() 退出后守卫恢复，后续 matmul 仍被计数"""
        g = TorchOpGuard(mode="warn")
        a, b = torch.rand(8, 8), torch.rand(8, 8)
        with g:
            with TorchOpGuard.pause():
                _ = torch.matmul(a, b)   # 不计数
            _ = torch.matmul(a, b)        # 计数
        assert len(g.forbidden_calls) == 1, (
            f"pause 退出后应恢复守卫，期望 1 次计数，got {g.forbidden_calls}"
        )

    def test_pause_block_mode_doesnt_raise(self):
        """pause() 期间即使是 block 模式也不应抛异常"""
        a, b = torch.rand(8, 8), torch.rand(8, 8)
        with TorchOpGuard(mode="block"):
            with TorchOpGuard.pause():
                # 即使 mode='block'，pause 期间也不应该抛
                _ = torch.matmul(a, b)

    def test_pause_nested(self):
        """嵌套 pause() — 内层退出后外层仍为 paused 状态"""
        g = TorchOpGuard(mode="warn")
        a, b = torch.rand(8, 8), torch.rand(8, 8)
        with g:
            with TorchOpGuard.pause():
                with TorchOpGuard.pause():
                    _ = torch.matmul(a, b)
                # 内层退出 — 外层仍 paused
                _ = torch.matmul(a, b)
            # 全部退出 — 守卫恢复
            _ = torch.matmul(a, b)
        assert len(g.forbidden_calls) == 1

    def test_pause_outside_guard_is_noop(self):
        """没在守卫内时调 pause() 不应崩"""
        a, b = torch.rand(8, 8), torch.rand(8, 8)
        with TorchOpGuard.pause():
            _ = torch.matmul(a, b)  # 没人监听，无害

    def test_pause_aten_also_suppressed(self):
        """pause() 也应该屏蔽 aten 变体（防止 warmup 用 aten 写法时 normalize 重新命中）"""
        g = TorchOpGuard(mode="warn")
        a, b = torch.rand(8, 8), torch.rand(8, 8)
        with g:
            with TorchOpGuard.pause():
                _ = torch.ops.aten.matmul(a, b)
                _ = a @ b
                _ = a.matmul(b)
        assert g.forbidden_calls == []


# ============================================================================
# 模式行为
# ============================================================================
class TestModes:
    """warn / block / off 三种模式"""

    def test_warn_mode_does_not_raise(self):
        g = TorchOpGuard(mode="warn")
        a, b = torch.rand(8, 8), torch.rand(8, 8)
        with g:
            _ = torch.matmul(a, b)
        # 无异常即可

    def test_block_mode_raises(self):
        a, b = torch.rand(8, 8), torch.rand(8, 8)
        with pytest.raises(RuntimeError, match="SECURITY"):
            with TorchOpGuard(mode="block"):
                _ = torch.matmul(a, b)

    def test_off_mode_noop(self):
        g = TorchOpGuard(mode="off")
        a, b = torch.rand(8, 8), torch.rand(8, 8)
        with g:
            _ = torch.matmul(a, b)
        # off 模式不监听，forbidden_calls 应为空
        assert g.forbidden_calls == []

    def test_invalid_mode_rejected(self):
        with pytest.raises(ValueError, match="mode must be one of"):
            TorchOpGuard(mode="invalid")


# ============================================================================
# 集成测试：模拟 perf_eval._boost_freq_and_clear_cache 的调用模式
# ============================================================================
class TestPerfEvalBoostIntegration:
    """
    复现 false positive 场景：
        op_runner 入 TorchOpGuard → perf_eval 在内部做 _boost_freq_and_clear_cache
        (调 torch.matmul) → 错误地把 freq-boost 算到 candidate op 头上。

    本测试模拟这条链路在 CPU 上：
      with guard:
          # 模拟 candidate op 之前的 freq boost（需要 pause）
          freq_boost_warmup()
          # 模拟 candidate op 真实计算（mish 等 elementwise，不应触发）
          torch.nn.functional.silu(x)   # silu 在 forbidden set 里，但作为 candidate 真实计算，正确做法是改 guard 屏蔽内部 warmup，而 candidate 本身的 silu 应留计数
    """

    def test_freq_boost_does_not_trigger_guard(self):
        """模拟：warmup matmul 包在 pause() 里，guard 不应报告任何禁用 API"""
        g = TorchOpGuard(mode="warn")
        mm1 = torch.rand(64, 64)
        mm2 = torch.rand(64, 64)
        reduce_input = torch.rand(16, 16, 16)

        def _simulated_freq_boost():
            """复刻 perf_eval._boost_freq_and_clear_cache 的算子序列"""
            with TorchOpGuard.pause():
                _ = torch.matmul(mm1, mm2)
                _ = torch.max(reduce_input)

        with g:
            _simulated_freq_boost()
            # 之后假设 candidate kernel 实际跑了 mish（fast vector op，不在 forbidden set）
            x = torch.rand(64, 64)
            _ = torch.nn.functional.mish(x)

        # forbidden_calls 应为空：freq boost 被 pause 屏蔽；mish 不在 forbidden set
        assert g.forbidden_calls == [], (
            f"freq boost (paused) + mish (非 forbidden) 不应触发计数，"
            f"got {g.forbidden_calls}"
        )

    def test_freq_boost_then_candidate_violation_caught(self):
        """正向场景：freq boost 静音，candidate 真的调了 matmul 仍被抓"""
        g = TorchOpGuard(mode="warn")
        mm1 = torch.rand(64, 64)
        mm2 = torch.rand(64, 64)

        with g:
            with TorchOpGuard.pause():
                _ = torch.matmul(mm1, mm2)  # warmup — 屏蔽
            # 模拟 candidate 作弊调 matmul — 应被捕获
            x, y = torch.rand(8, 8), torch.rand(8, 8)
            _ = torch.matmul(x, y)

        assert len(g.forbidden_calls) == 1, (
            f"candidate 调 matmul 必须被抓，got {g.forbidden_calls}"
        )
