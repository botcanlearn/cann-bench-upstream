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
PerfEvaluator 单元测试

测试对象：kernel_eval.eval.perf_eval.PerfEvaluator
核心功能：
1. run_profiled 临时目录清理（finally 块）
2. profiler 异常后资源不泄漏
3. perf_metric_strategy 从 Config 自取
4. profiler 采集失败后的自动重试机制
"""

from unittest.mock import patch

from kernel_eval.config import Config
from kernel_eval.eval.perf_eval import PerfEvaluator
from kernel_eval.base.perf_strategy import ProfFileLocations


class TestProfileOperatorTempDirCleanup:
    """测试 run_profiled 的临时目录清理"""

    def test_temp_dir_cleaned_on_profiler_exception(self):
        """profiler 抛异常后 finally 块仍执行清理，异常信息保留在 metadata"""
        # PerfEvaluator 从 Config.perf_metric_strategy_override 获取策略
        config = Config(enable_profiler=True, perf_metric_strategy_override="kernel_details")
        evaluator = PerfEvaluator(config, archive_prof=False, freq_boost=False)

        def dummy_func():
            pass

        with patch.object(evaluator, '_profile', side_effect=RuntimeError("crash")), \
             patch.object(evaluator, '_parse_case_id', return_value=('L1/Add', '0001')), \
             patch('shutil.rmtree') as mock_rmtree:
            outputs, result = evaluator.run_profiled(
                "L1/Add/0001", dummy_func, warmup=1, repeat=2,
            )
            # profiler 异常信息保留在 metadata 中，error_msg 由策略报告
            assert result.metadata["profile_exception"] == "crash"
            # finally 块应执行了清理
            assert mock_rmtree.call_count >= 1

    def test_temp_dir_cleaned_even_when_csv_walk_throws(self):
        """CSV 遍历抛异常后 finally 块仍执行清理"""
        config = Config(enable_profiler=True, perf_metric_strategy_override="kernel_details")
        evaluator = PerfEvaluator(config, archive_prof=False, freq_boost=False)

        def dummy_func():
            pass

        with patch.object(evaluator, '_profile', return_value=None), \
             patch.object(evaluator, '_parse_case_id', return_value=('L1/Add', '0001')), \
             patch('os.walk', side_effect=OSError("walk failed")), \
             patch('shutil.rmtree') as mock_rmtree:
            try:
                evaluator.run_profiled("L1/Add/0001", dummy_func, warmup=1, repeat=2)
            except OSError:
                pass
            assert mock_rmtree.call_count >= 1

    def test_archive_mode_does_not_cleanup(self):
        """archive_prof=True 时 finally 块不应清理目录"""
        config = Config(enable_profiler=True, perf_metric_strategy_override="kernel_details")
        evaluator = PerfEvaluator(config, archive_prof=True, freq_boost=False)

        def dummy_func():
            pass

        with patch.object(evaluator, '_profile', return_value=None), \
             patch.object(evaluator, '_parse_case_id', return_value=('L1/Add', '0001')), \
             patch('os.makedirs', return_value=None), \
             patch('os.listdir', return_value=[]), \
             patch('shutil.rmtree') as mock_rmtree:
            evaluator.run_profiled("L1/Add/0001", dummy_func, warmup=1, repeat=2)
            mock_rmtree.assert_not_called()


class TestMeasureSimple:
    """测试 _measure_simple 方法（CPU 计时路径）"""

    def test_measure_simple_basic(self):
        """enable_profiler=False 时走简单计时路径"""
        config = Config(enable_profiler=False)
        evaluator = PerfEvaluator(config)

        def add(a, b):
            return a + b

        outputs, result = evaluator.run_profiled(
            "L1/Add/0001", add, 1.0, 2.0,
            warmup=2, repeat=3,
        )

        assert result.elapsed_us >= 0
        assert result.error_msg is None


def test_run_profiled_uses_trace_view_strategy():
    """Config.perf_metric_strategy_override="trace_view" 时使用 trace_view 口径"""
    # PerfEvaluator 从 Config.perf_metric_strategy_override 自取策略
    config = Config(enable_profiler=True, perf_metric_strategy_override="trace_view")
    evaluator = PerfEvaluator(config, archive_prof=False, freq_boost=False)
    trace_strategy = evaluator.perf_metric_strategy  # 从 registry 取到的 TraceViewStrategy 实例

    def dummy_func():
        return "ok"

    def run_stub(fn, prof_dir, warmup, repeat):
        fn()

    # Mock TraceViewStrategy.parse to simulate successful trace_view parsing
    with patch.object(evaluator, "_profile", side_effect=run_stub), \
         patch.object(evaluator, "_parse_case_id", return_value=("L1/Add", "0001")), \
         patch.object(trace_strategy, "parse") as mock_parse:
        # Simulate TraceViewStrategy.parse filling the result
        def fake_parse(prof_files, result):
            result.elapsed_us = 12.35
            result.op_times = {"trace_view": {
                "aicore_e2e": 12.35,
                "aicpukernel_gap": 1.23,
                "aicore_e2e_jitter": 0.04,
            }}
            result.metadata["perf_source"] = "trace_view"
            result.metadata["elapsed_us_source"] = "trace_view.aicore_e2e"
            return result
        mock_parse.side_effect = fake_parse

        outputs, result = evaluator.run_profiled(
            "L1/Add/0001", dummy_func, warmup=1, repeat=2,
        )

    assert outputs == "ok"
    assert result.elapsed_us == 12.35
    assert result.op_times == {
        "trace_view": {
            "aicore_e2e": 12.35,
            "aicpukernel_gap": 1.23,
            "aicore_e2e_jitter": 0.04,
        }
    }
    assert result.metadata["perf_source"] == "trace_view"
    assert result.metadata["elapsed_us_source"] == "trace_view.aicore_e2e"
    mock_parse.assert_called_once()


class TestProfilerStepSynchronization:
    """测试 profiler step 边界同步"""

    def test_profile_step_synchronizes_before_step(self):
        """候选算子返回后应先等待 NPU stream，再推进 profiler step"""
        calls = []

        class FakeDeviceManager:
            def synchronize(self):
                calls.append("sync")

        class FakeProfiler:
            def step(self):
                calls.append("step")

        config = Config(enable_profiler=True, perf_metric_strategy_override="kernel_details")
        evaluator = PerfEvaluator(config, device_manager=FakeDeviceManager(), archive_prof=False, freq_boost=False)

        def fn():
            calls.append("fn")

        exc = evaluator._run_profile_step(fn, FakeProfiler())

        assert exc is None
        assert calls == ["fn", "sync", "step"]

    def test_profile_step_advances_step_when_sync_fails(self):
        """同步暴露异步执行错误时仍推进 step，让 profiler 上下文干净退出"""
        calls = []

        class FakeDeviceManager:
            def synchronize(self):
                calls.append("sync")
                raise RuntimeError("device failed")

        class FakeProfiler:
            def step(self):
                calls.append("step")

        config = Config(enable_profiler=True, perf_metric_strategy_override="kernel_details")
        evaluator = PerfEvaluator(config, device_manager=FakeDeviceManager(), archive_prof=False, freq_boost=False)

        def fn():
            calls.append("fn")

        exc = evaluator._run_profile_step(fn, FakeProfiler())

        assert isinstance(exc, RuntimeError)
        assert str(exc) == "device failed"
        assert calls == ["fn", "sync", "step"]


class TestProfilerRetry:
    """Test profiler collection retry mechanism.

    When kernel_details.csv is missing or elapsed_us <= 0 but the op executed
    successfully (Host side dispatched the kernel — last_outputs is not None),
    the profiler collection should be automatically retried instead of
    directly scoring 0.
    """

    def _make_evaluator(self, max_retries=2, retry_delay=0.0):
        config = Config(
            enable_profiler=True,
            perf_metric_strategy_override="kernel_details",
            profiler_max_retries=max_retries,
            profiler_retry_delay=retry_delay,
        )
        return PerfEvaluator(config, archive_prof=False, freq_boost=False)

    @staticmethod
    def _run_stub(fn, prof_dir, warmup, repeat):
        fn()

    def test_retry_succeeds_on_second_attempt(self):
        """First attempt elapsed_us=0, retry produces valid data."""
        evaluator = self._make_evaluator(max_retries=2, retry_delay=0.0)
        strategy = evaluator.perf_metric_strategy

        parse_calls = [0]

        def fake_parse(prof_files, result):
            parse_calls[0] += 1
            if parse_calls[0] == 1:
                result.elapsed_us = 0.0
                result.error_msg = "no csv"
            else:
                result.elapsed_us = 42.0
                result.error_msg = None
            return result

        with patch.object(evaluator, '_profile', side_effect=self._run_stub), \
             patch.object(evaluator, '_parse_case_id',
                          return_value=('L1/Add', '0001')), \
             patch.object(evaluator, '_locate_prof_files',
                          return_value=ProfFileLocations()), \
             patch.object(strategy, 'parse', side_effect=fake_parse), \
             patch('time.sleep'):
            outputs, result = evaluator.run_profiled(
                "L1/Add_0001", lambda: "ok", warmup=1, repeat=2,
            )

        assert result.elapsed_us == 42.0
        assert result.metadata.get("profiler_retries_used") == 1
        assert parse_calls[0] == 2

    def test_retry_when_profile_raises_but_op_ran(self):
        """_profile raises but last_outputs set (op ran in pre-flight) → retry."""
        evaluator = self._make_evaluator(max_retries=2, retry_delay=0.0)
        strategy = evaluator.perf_metric_strategy

        profile_calls = [0]

        def profile_stub(fn, prof_dir, warmup, repeat):
            fn()
            profile_calls[0] += 1
            if profile_calls[0] == 1:
                raise RuntimeError("profiler crashed")

        def fake_parse(prof_files, result):
            result.elapsed_us = 77.0
            return result

        with patch.object(evaluator, '_profile', side_effect=profile_stub), \
             patch.object(evaluator, '_parse_case_id',
                          return_value=('L1/Add', '0001')), \
             patch.object(evaluator, '_locate_prof_files',
                          return_value=ProfFileLocations()), \
             patch.object(strategy, 'parse', side_effect=fake_parse), \
             patch('time.sleep'):
            outputs, result = evaluator.run_profiled(
                "L1/Add_0001", lambda: "ok", warmup=1, repeat=2,
            )

        assert result.elapsed_us == 77.0
        assert result.metadata.get("profiler_retries_used") == 1
        assert profile_calls[0] == 2

    def test_no_retry_when_op_fails(self):
        """When _profile raises and last_outputs is None, no retry."""
        evaluator = self._make_evaluator(max_retries=2, retry_delay=0.0)

        with patch.object(evaluator, '_profile',
                          side_effect=RuntimeError("crash")) as mock_profile, \
             patch.object(evaluator, '_parse_case_id',
                          return_value=('L1/Add', '0001')), \
             patch('time.sleep'), \
             patch('shutil.rmtree'):
            outputs, result = evaluator.run_profiled(
                "L1/Add_0001", lambda: "ok", warmup=1, repeat=2,
            )

        assert result.elapsed_us == 0.0
        assert result.metadata.get("profiler_retries_used") is None
        assert result.metadata.get("profiler_retries_exhausted") is None
        assert mock_profile.call_count == 1

    def test_no_retry_when_first_attempt_succeeds(self):
        """When first attempt produces valid elapsed_us, no retry."""
        evaluator = self._make_evaluator(max_retries=2, retry_delay=0.0)
        strategy = evaluator.perf_metric_strategy

        def fake_parse(prof_files, result):
            result.elapsed_us = 99.0
            return result

        with patch.object(evaluator, '_profile', side_effect=self._run_stub) as mock_profile, \
             patch.object(evaluator, '_parse_case_id',
                          return_value=('L1/Add', '0001')), \
             patch.object(evaluator, '_locate_prof_files',
                          return_value=ProfFileLocations()), \
             patch.object(strategy, 'parse', side_effect=fake_parse), \
             patch('time.sleep'):
            outputs, result = evaluator.run_profiled(
                "L1/Add_0001", lambda: "ok", warmup=1, repeat=2,
            )

        assert result.elapsed_us == 99.0
        assert result.metadata.get("profiler_retries_used") is None
        assert mock_profile.call_count == 1

    def test_retries_exhausted(self):
        """All attempts fail → profiler_retries_exhausted=True."""
        evaluator = self._make_evaluator(max_retries=2, retry_delay=0.0)
        strategy = evaluator.perf_metric_strategy

        def fake_parse(prof_files, result):
            result.elapsed_us = 0.0
            result.error_msg = "no csv"
            return result

        with patch.object(evaluator, '_profile', side_effect=self._run_stub) as mock_profile, \
             patch.object(evaluator, '_parse_case_id',
                          return_value=('L1/Add', '0001')), \
             patch.object(evaluator, '_locate_prof_files',
                          return_value=ProfFileLocations()), \
             patch.object(strategy, 'parse', side_effect=fake_parse), \
             patch('time.sleep'), \
             patch('shutil.rmtree'):
            outputs, result = evaluator.run_profiled(
                "L1/Add_0001", lambda: "ok", warmup=1, repeat=2,
            )

        assert result.elapsed_us == 0.0
        assert result.metadata.get("profiler_retries_exhausted") is True
        assert len(result.metadata.get("profiler_retry_reasons", [])) == 2
        assert mock_profile.call_count == 3

    def test_no_retry_when_disabled(self):
        """profiler_max_retries=0 means no retry."""
        evaluator = self._make_evaluator(max_retries=0, retry_delay=0.0)
        strategy = evaluator.perf_metric_strategy

        def fake_parse(prof_files, result):
            result.elapsed_us = 0.0
            result.error_msg = "no csv"
            return result

        with patch.object(evaluator, '_profile', side_effect=self._run_stub) as mock_profile, \
             patch.object(evaluator, '_parse_case_id',
                          return_value=('L1/Add', '0001')), \
             patch.object(evaluator, '_locate_prof_files',
                          return_value=ProfFileLocations()), \
             patch.object(strategy, 'parse', side_effect=fake_parse), \
             patch('time.sleep'), \
             patch('shutil.rmtree'):
            outputs, result = evaluator.run_profiled(
                "L1/Add_0001", lambda: "ok", warmup=1, repeat=2,
            )

        assert result.elapsed_us == 0.0
        assert result.metadata.get("profiler_retries_exhausted") is None
        assert mock_profile.call_count == 1

    def test_prof_dir_cleaned_between_retries(self):
        """_clean_prof_dir_contents called before each retry attempt."""
        evaluator = self._make_evaluator(max_retries=2, retry_delay=0.0)
        strategy = evaluator.perf_metric_strategy

        parse_calls = [0]

        def fake_parse(prof_files, result):
            parse_calls[0] += 1
            if parse_calls[0] <= 2:
                result.elapsed_us = 0.0
                result.error_msg = "no csv"
            else:
                result.elapsed_us = 50.0
                result.error_msg = None
            return result

        with patch.object(evaluator, '_profile', side_effect=self._run_stub), \
             patch.object(evaluator, '_parse_case_id',
                          return_value=('L1/Add', '0001')), \
             patch.object(evaluator, '_locate_prof_files',
                          return_value=ProfFileLocations()), \
             patch.object(strategy, 'parse', side_effect=fake_parse), \
             patch.object(evaluator, '_clean_prof_dir_contents') as mock_clean, \
             patch('time.sleep'):
            outputs, result = evaluator.run_profiled(
                "L1/Add_0001", lambda: "ok", warmup=1, repeat=2,
            )

        assert result.elapsed_us == 50.0
        assert mock_clean.call_count == 2
