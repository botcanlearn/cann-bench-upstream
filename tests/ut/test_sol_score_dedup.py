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
验证 SOL-Score 计算正确性

核心目标：确保 cann_scoring.py 中的 per_case_sol_score 和权重常量工作正常。
（scoring.py 已合并到 cann_scoring.py，不再需要验证跨模块引用）
"""

import pytest


def _make_perf_result(elapsed_us, t_hw_us=0.0):
    """构造 PerfResult 用于测试 CannScoringScheme"""
    from kernel_eval.base.result import PerfResult
    return PerfResult(
        elapsed_us=elapsed_us,
        metadata={'t_hw_us': t_hw_us},
    )


class TestCannScoringSchemeCalculation:
    """CannScoringScheme.calculate_case_score 结果验证"""

    def test_normal_case_matches(self):
        """正常情况：CannScoringScheme 与 per_case_sol_score 结果一致"""
        from kernel_eval.benches.cann_scoring import per_case_sol_score, CannScoringScheme

        scheme = CannScoringScheme()
        perf = _make_perf_result(elapsed_us=50, t_hw_us=50)
        result = scheme.calculate_case_score(perf, baseline_us=100)
        expected = per_case_sol_score(t_baseline=100, t_cand=50, t_hw=50)

        assert result == pytest.approx(expected)

    def test_cand_below_hw_matches(self):
        """突破硬件下界：T_cand < T_HW，score > 1.0"""
        from kernel_eval.benches.cann_scoring import per_case_sol_score, CannScoringScheme

        scheme = CannScoringScheme()
        perf = _make_perf_result(elapsed_us=25, t_hw_us=50)
        result = scheme.calculate_case_score(perf, baseline_us=100)
        expected = per_case_sol_score(t_baseline=100, t_cand=25, t_hw=50)

        assert result == pytest.approx(expected)
        assert result > 1.0

    def test_baseline_missing_both_use_fallback(self):
        """baseline==0 时双方都应该使用 fallback"""
        from kernel_eval.benches.cann_scoring import per_case_sol_score, CannScoringScheme

        scheme = CannScoringScheme()
        perf = _make_perf_result(elapsed_us=100, t_hw_us=50)
        result = scheme.calculate_case_score(perf, baseline_us=0)
        expected = per_case_sol_score(t_baseline=0, t_cand=100, t_hw=50)

        assert result == pytest.approx(expected)

    def test_invalid_inputs_both_return_none(self):
        """无效输入双方都返回 None"""
        from kernel_eval.benches.cann_scoring import per_case_sol_score, CannScoringScheme

        scheme = CannScoringScheme()
        perf = _make_perf_result(elapsed_us=0, t_hw_us=50)
        result = scheme.calculate_case_score(perf, baseline_us=100)
        expected = per_case_sol_score(t_baseline=100, t_cand=0, t_hw=50)

        assert result is None
        assert expected is None

    def test_t_hw_zero_both_return_none(self):
        """t_hw==0 时双方都返回 None"""
        from kernel_eval.benches.cann_scoring import per_case_sol_score, CannScoringScheme

        scheme = CannScoringScheme()
        perf = _make_perf_result(elapsed_us=80, t_hw_us=0)
        result = scheme.calculate_case_score(perf, baseline_us=100)
        expected = per_case_sol_score(t_baseline=100, t_cand=80, t_hw=0)

        assert result is None
        assert expected is None


class TestWeightConstantsDefined:
    """权重常量应该在 cann_scoring.py 中定义"""

    def test_weights_defined_in_cann_scoring(self):
        """权重常量定义在 cann_scoring.py 中"""
        from kernel_eval.benches.cann_scoring import (
            WEIGHT_COMPILATION,
            WEIGHT_FUNCTION,
            WEIGHT_PERFORMANCE,
        )
        # 验证权重值
        assert WEIGHT_COMPILATION == 0.2
        assert WEIGHT_FUNCTION == 0.3
        assert WEIGHT_PERFORMANCE == 0.5
        # 验证权重之和为 1
        assert WEIGHT_COMPILATION + WEIGHT_FUNCTION + WEIGHT_PERFORMANCE == 1.0
