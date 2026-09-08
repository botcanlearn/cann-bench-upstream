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
host 内存优化单元测试（evaluator 侧）

覆盖：
1. RSS 读取 / 阈值解析（KERNEL_EVAL_RSS_TRIM_MB 覆盖、<=0 禁用、非法回退）
2. _cleanup_memory 的 RSS 触发 / 不触发 / 降级路径
3. evaluate_case 精度对比后即时释放大对象的端到端回归
   （stub 掉 loader/runner 等协作者，验证结果正确且 run result 输出已释放）
"""

from types import SimpleNamespace

import pytest
import torch

import kernel_eval.eval.evaluator as evaluator_mod
from kernel_eval.base.result import FAILURE_TYPE_COMPILE_RUNTIME_ERROR
from kernel_eval.eval.accuracy_eval import AccuracyEvaluator
from kernel_eval.eval.evaluator import (
    Evaluator,
    _malloc_trim,
    _read_rss_mb,
    _rss_trim_threshold_mb,
)
from kernel_eval.eval.op_runner import OpRunResult

_RSS_ENV = "KERNEL_EVAL_RSS_TRIM_MB"
_DEFAULT_MB = float(48 * 1024)


class TestReadRss:
    """_read_rss_mb：读取 /proc/self/status 的 VmRSS"""

    def test_real_proc_returns_positive_mb(self):
        rss = _read_rss_mb()
        assert rss is not None
        assert rss > 0

    def test_open_failure_returns_none(self, monkeypatch):
        def _boom(*args, **kwargs):
            raise OSError("no /proc")

        monkeypatch.setattr("builtins.open", _boom)
        assert _read_rss_mb() is None

    def test_malformed_content_returns_none(self, monkeypatch):
        import io

        monkeypatch.setattr("builtins.open",
                            lambda *a, **k: io.StringIO("Name:\tproc\nBadLine\n"))
        assert _read_rss_mb() is None


class TestRssTrimThreshold:
    """_rss_trim_threshold_mb：阈值解析与禁用语义"""

    def test_default_48_gib_when_unset(self, monkeypatch):
        monkeypatch.delenv(_RSS_ENV, raising=False)
        assert _rss_trim_threshold_mb() == _DEFAULT_MB

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv(_RSS_ENV, "65536")
        assert _rss_trim_threshold_mb() == 65536.0

    @pytest.mark.parametrize("value", ["0", "-1", "-0.5"])
    def test_non_positive_disables(self, monkeypatch, value):
        monkeypatch.setenv(_RSS_ENV, value)
        assert _rss_trim_threshold_mb() is None

    def test_invalid_value_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv(_RSS_ENV, "not-a-number")
        assert _rss_trim_threshold_mb() == _DEFAULT_MB


class TestCleanupMemoryRssTrim:
    """_cleanup_memory：RSS 超阈值时追加 malloc_trim，失败静默降级"""

    @staticmethod
    def _call_cleanup():
        # _cleanup_memory 不读取任何实例状态，绕过重量级 __init__ 直接调用
        Evaluator._cleanup_memory(Evaluator.__new__(Evaluator))

    def test_trim_triggered_above_threshold(self, monkeypatch):
        monkeypatch.delenv(_RSS_ENV, raising=False)
        monkeypatch.setattr(evaluator_mod, "_read_rss_mb", lambda: 60 * 1024.0)
        trim_calls = []
        monkeypatch.setattr(evaluator_mod, "_malloc_trim",
                            lambda: trim_calls.append(1) or True)
        self._call_cleanup()
        assert trim_calls == [1]

    def test_trim_skipped_below_threshold(self, monkeypatch):
        monkeypatch.delenv(_RSS_ENV, raising=False)
        monkeypatch.setattr(evaluator_mod, "_read_rss_mb", lambda: 1024.0)
        trim_calls = []
        monkeypatch.setattr(evaluator_mod, "_malloc_trim",
                            lambda: trim_calls.append(1) or True)
        self._call_cleanup()
        assert trim_calls == []

    def test_trim_skipped_when_disabled_by_env(self, monkeypatch):
        monkeypatch.setenv(_RSS_ENV, "0")
        monkeypatch.setattr(evaluator_mod, "_read_rss_mb", lambda: 1024 * 1024.0)
        trim_calls = []
        monkeypatch.setattr(evaluator_mod, "_malloc_trim",
                            lambda: trim_calls.append(1) or True)
        self._call_cleanup()
        assert trim_calls == []

    def test_rss_read_failure_degrades_silently(self, monkeypatch):
        monkeypatch.delenv(_RSS_ENV, raising=False)
        monkeypatch.setattr(evaluator_mod, "_read_rss_mb", lambda: None)
        trim_calls = []
        monkeypatch.setattr(evaluator_mod, "_malloc_trim",
                            lambda: trim_calls.append(1) or True)
        self._call_cleanup()  # 不抛异常
        assert trim_calls == []

    def test_malloc_trim_real_call_succeeds_on_glibc(self):
        # 本环境为 glibc Linux：真实调用应成功；非 glibc 平台则静默 False
        assert _malloc_trim() is True

    def test_malloc_trim_failure_degrades_silently(self, monkeypatch):
        import ctypes

        def _boom(*args, **kwargs):
            raise OSError("no libc")

        monkeypatch.setattr(ctypes, "CDLL", _boom)
        assert _malloc_trim() is False


class _StubCase:
    rel_path = "level1/stub_op"
    operator = "StubOp"
    case_id = 1
    case_num = 1
    attrs = {}
    input_shapes = [[8]]
    dtypes = ["float32"]
    value_ranges = None
    baseline_perf_us = 100.0
    t_hw_us = 10.0

    def get_case_id_str(self):
        return "level1/stub_op_1"


class _StubGoldenLoader:
    @staticmethod
    def _golden(x):
        return x * 2

    def get_golden_function(self, rel_path):
        return self._golden

    def get_input_function(self, rel_path):
        return None

    def get_oracle_function(self, rel_path):
        return None

    def get_bench_function(self, rel_path):
        return None

    def get_output_function(self, rel_path):
        return None


class _StubDataGenerator:
    def generate_input_tensors_from_case(self, input_shapes, dtypes, value_ranges, seed):
        return [torch.ones(*input_shapes[0], dtype=torch.float32)]


class _StubParamBuilder:
    def build_call_params(self, func, case, tensors):
        return {"x": tensors[0]}


class _StubOpRunner:
    """run() 真实调用传入函数产生输出；run_ai_op 可注入输出或失败"""

    def __init__(self, ai_output=None, ai_success=True):
        self._ai_output = ai_output
        self._ai_success = ai_success

    def run(self, func, params, case_id_str, inputs, to_device=False, enable_profiler=False):
        return OpRunResult(success=True, outputs=[func(**params)], elapsed_us=1.0)

    def run_ai_op(self, ai_op_func, params, case_id_str, input_tensors, enable_perf=False):
        if not self._ai_success:
            return OpRunResult(success=False, error="stub ai crash", elapsed_us=0)
        out = self._ai_output if self._ai_output is not None else ai_op_func(**params)
        return OpRunResult(success=True, outputs=[out], elapsed_us=2.0)


class _StubOperatorMatcher:
    def load_ai_operator(self, operator):
        return lambda x: x * 2


class _StubOperatorLoader:
    def get_operator(self, rel_path):
        return SimpleNamespace(
            inputs=[SimpleNamespace(name="x")],
            outputs=[SimpleNamespace(
                name="y", dtype=["float32"], compare=True, index_gather=None)],
            precision_thresholds=None,
        )


def _ensure_correctness_checker():
    """按需补注册 relative_error 判断器。

    全量运行 tests/ut 时 test_checkers.py 会 clear_checker_registry() 且不恢复；
    此处对齐 benches/cann.py 的幂等注册模式，保证本文件用例与执行顺序无关。
    """
    from kernel_eval.checkers.relative_error_checker import RelativeErrorChecker
    from kernel_eval.registry.checker_registry import CheckerRegistry
    if 'relative_error' not in CheckerRegistry.get_all():
        CheckerRegistry.register('relative_error', RelativeErrorChecker())


def _build_stub_evaluator(op_runner):
    """绕过重量级 __init__，仅装配 evaluate_case 实际使用的协作者"""
    _ensure_correctness_checker()
    evaluator = Evaluator.__new__(Evaluator)
    evaluator.config = SimpleNamespace(
        eval_seed=0,
        precision_thresholds={},
        enable_accuracy_retry=False,
    )
    evaluator.bench_config = SimpleNamespace(
        golden_precision="fp64_cpu",
        dtype_tolerance_map=None,
    )
    evaluator.incremental_output_path = None
    evaluator._pypto_pro_jit_isolation = False
    evaluator.perf_evaluator = None
    evaluator.golden_loader = _StubGoldenLoader()
    evaluator.data_generator = _StubDataGenerator()
    evaluator.operator_loader = _StubOperatorLoader()
    evaluator.param_builder = _StubParamBuilder()
    evaluator.op_runner = op_runner
    evaluator.operator_matcher = _StubOperatorMatcher()
    evaluator.accuracy_evaluator = AccuracyEvaluator()
    return evaluator


class TestEvaluateCaseRelease:
    """精度对比完成后立即释放 fp64 golden 输入副本 / golden/native/AI 输出的回归

    早期释放在函数外部不可直接观测，这里验证的是端到端行为不被破坏
    （无 NameError / use-after-release），且两个 run result 的 outputs 均已清空。
    """

    def test_pass_path_releases_outputs(self):
        evaluator = _build_stub_evaluator(_StubOpRunner())
        result = evaluator.evaluate_case(_StubCase())

        assert result.success is True
        assert result.accuracy_result is not None
        assert result.golden_run_result is not None
        assert result.golden_run_result.outputs is None
        assert result.ai_run_result is not None
        assert result.ai_run_result.outputs is None

    def test_accuracy_fail_path_releases_outputs(self):
        wrong_output = torch.full((8,), 100.0, dtype=torch.float32)
        evaluator = _build_stub_evaluator(_StubOpRunner(ai_output=wrong_output))
        result = evaluator.evaluate_case(_StubCase())

        assert result.success is False
        assert result.error_msg.startswith("精度不达标")
        assert result.golden_run_result.outputs is None
        assert result.ai_run_result.outputs is None

    def test_ai_run_failure_releases_golden_outputs(self):
        evaluator = _build_stub_evaluator(_StubOpRunner(ai_success=False))
        result = evaluator.evaluate_case(_StubCase())

        assert result.success is False
        assert result.failure_type == FAILURE_TYPE_COMPILE_RUNTIME_ERROR
        assert "AI算子执行失败" in result.error_msg
        assert result.golden_run_result.outputs is None
        assert result.ai_run_result.outputs is None
