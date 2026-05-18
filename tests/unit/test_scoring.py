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
Scoring 模块单元测试

测试对象：kernel_eval.report.scoring
覆盖：Eq.3 (per_case_sol_score)、Eq.4 (aggregate_eq4 / ScoringCalculator)、
Eq.5 (calculate_overall_score / calculate_level_score)。
"""

import pytest

from kernel_eval.eval.results import EvalCaseResult, EvalOperatorResult
from kernel_eval.eval.perf_eval import PerfResult
from kernel_eval.report.scoring import (
    WEIGHT_COMPILATION,
    WEIGHT_FUNCTION,
    WEIGHT_PERFORMANCE,
    ScoreInfo,
    ScoringCalculator,
    aggregate_eq4,
    per_case_sol_score,
)


class TestPerCaseSolScore:
    """Eq.3: score_i = (T_base - T_HW) / ((T_cand - T_HW) + (T_base - T_HW))"""

    def test_score_at_baseline_equals_half(self):
        # T_cand == T_baseline ⇒ score == 0.5
        assert per_case_sol_score(100, 100, 50) == pytest.approx(0.5)

    def test_score_at_hw_equals_one(self):
        # T_cand == T_HW ⇒ score == 1.0
        assert per_case_sol_score(100, 50, 50) == pytest.approx(1.0)

    def test_score_can_exceed_one_when_cand_below_hw(self):
        # T_cand < T_HW（突破硬件下界）允许 score > 1.0
        score = per_case_sol_score(t_baseline=100, t_cand=25, t_hw=50)
        assert score > 1.0

    def test_zero_anchor_returns_none(self):
        assert per_case_sol_score(0, 80, 50) is None
        assert per_case_sol_score(100, 0, 50) is None
        assert per_case_sol_score(100, 80, 0) is None

    def test_negative_anchor_returns_none(self):
        assert per_case_sol_score(-1, 80, 50) is None
        assert per_case_sol_score(100, -1, 50) is None
        assert per_case_sol_score(100, 80, -1) is None

    def test_denom_zero_returns_none(self):
        # (t_cand - t_hw) + (t_base - t_hw) == 0
        assert per_case_sol_score(t_baseline=50, t_cand=50, t_hw=50) is None

    def test_baseline_below_hw_emits_warning(self, capsys):
        # 应警告但仍返回数值（按用户设计）
        score = per_case_sol_score(t_baseline=40, t_cand=80, t_hw=50)
        assert score is not None
        captured = capsys.readouterr()
        assert "T_baseline" in captured.out
        assert "< T_HW" in captured.out


class TestAggregateEq4:
    """Eq.4 聚合（dict-input 与 EvalOperatorResult 共用同一路径）"""

    def test_all_pass_full_score_when_cand_at_hw(self):
        # 所有用例 score_i=1.0 ⇒ total_score=100
        agg = aggregate_eq4(
            compile_passed=True,
            total_cases=3,
            case_scores=[(True, 1.0), (True, 1.0), (True, 1.0)],
        )
        assert agg["compilation_score"] == pytest.approx(20.0)
        assert agg["function_score"] == pytest.approx(30.0)
        assert agg["performance_score"] == pytest.approx(50.0)
        assert agg["total_score"] == pytest.approx(100.0)

    def test_compile_fail_zeros_everything(self):
        # 编译失败：function/performance 都置零
        agg = aggregate_eq4(
            compile_passed=False,
            total_cases=3,
            case_scores=[(True, 1.0), (True, 1.0), (True, 1.0)],
        )
        assert agg["compilation_score"] == 0.0
        assert agg["function_score"] == 0.0
        assert agg["performance_score"] == 0.0
        assert agg["total_score"] == 0.0
        assert agg["per_case_scores"] == [None, None, None]

    def test_none_score_counted_as_zero_for_performance(self):
        # score_i=None（缺锚点）：功能通过但性能项按 0 计入
        agg = aggregate_eq4(
            compile_passed=True,
            total_cases=1,
            case_scores=[(True, None)],
        )
        assert agg["compilation_score"] == pytest.approx(20.0)
        assert agg["function_score"] == pytest.approx(30.0)
        assert agg["performance_score"] == 0.0
        assert agg["total_score"] == pytest.approx(50.0)

    def test_partial_pass_uses_total_cases_as_denominator(self):
        # 5 个用例只过 2 个，分母用 5 而不是 2
        agg = aggregate_eq4(
            compile_passed=True,
            total_cases=5,
            case_scores=[(True, 1.0), (True, 1.0), (False, None),
                         (False, None), (False, None)],
        )
        assert agg["function_score"] == pytest.approx(2 * WEIGHT_FUNCTION / 5 * 100)
        assert agg["performance_score"] == pytest.approx(2 * WEIGHT_PERFORMANCE / 5 * 100)

    def test_cand_below_hw_inflates_score_above_100(self):
        # T_cand < T_HW 时 score_i > 1，总分可超过 100
        agg = aggregate_eq4(
            compile_passed=True,
            total_cases=1,
            case_scores=[(True, 2.0)],
        )
        assert agg["total_score"] > 100.0


class TestScoringCalculator:
    """ScoringCalculator.calculate_operator_score 与 dict 路径一致性"""

    @staticmethod
    def _make_case(success, baseline_us, t_hw, elapsed_us):
        perf = PerfResult(case_id="c", elapsed_us=elapsed_us) if elapsed_us else None
        return EvalCaseResult(
            case_id="c", rel_path="level1/exp", operator="Exp",
            case_num=0, success=success, perf_result=perf,
            baseline_perf_us=baseline_us, t_hw_us=t_hw,
        )

    def test_all_pass(self):
        cases = [
            self._make_case(True, 100, 50, 50),   # score=1.0
            self._make_case(True, 100, 50, 75),   # score=0.667
        ]
        op = EvalOperatorResult(
            rel_path="level1/exp", operator="Exp",
            total_cases=2, passed_cases=2, failed_cases=0, skipped_cases=0,
            results=cases, pass_rate=1.0, avg_speedup=1.5,
        )
        info = ScoringCalculator().calculate_operator_score(op)
        assert info.compile_passed is True
        assert info.compilation_score == pytest.approx(20.0)
        assert info.function_score == pytest.approx(30.0)
        # perf_score_sum = 1.0 + 2/3 = 5/3; (5/3 * 0.5 / 2) * 100 = 41.67
        assert info.performance_score == pytest.approx(5 / 3 * WEIGHT_PERFORMANCE / 2 * 100)

    def test_total_cases_uses_max_of_declared_and_actual(self):
        # 声明 total_cases=10，但 results 只有 2 条——分母用 10
        cases = [self._make_case(True, 100, 50, 50)] * 2
        op = EvalOperatorResult(
            rel_path="level1/exp", operator="Exp",
            total_cases=10, passed_cases=2, failed_cases=0, skipped_cases=8,
            results=cases, pass_rate=0.2, avg_speedup=1.0,
        )
        info = ScoringCalculator().calculate_operator_score(op)
        assert info.total_cases == 10
        assert info.function_score == pytest.approx(2 * WEIGHT_FUNCTION / 10 * 100)

    def test_total_cases_floors_at_one_for_empty_results(self):
        op = EvalOperatorResult(
            rel_path="level1/exp", operator="Exp",
            total_cases=0, passed_cases=0, failed_cases=0, skipped_cases=0,
            results=[], pass_rate=0.0, avg_speedup=0.0,
        )
        info = ScoringCalculator().calculate_operator_score(op)
        assert info.total_cases == 1  # max(0, 0, 1)
        assert info.function_score == 0.0

    def test_compile_failed_zeroes_function_and_perf(self):
        cases = [self._make_case(True, 100, 50, 50)]
        op = EvalOperatorResult(
            rel_path="level1/exp", operator="Exp",
            total_cases=1, passed_cases=1, failed_cases=0, skipped_cases=0,
            results=cases, pass_rate=1.0, avg_speedup=1.0,
            compilation_error="compile failed",
        )
        info = ScoringCalculator().calculate_operator_score(op)
        assert info.compile_passed is False
        assert info.compilation_score == 0.0
        assert info.function_score == 0.0
        assert info.performance_score == 0.0


class TestLevelAndOverallScores:
    """Eq.5: overall = Σ EachOperatorScore; level_score(level) 过滤 rel_path."""

    @staticmethod
    def _make_info(rel_path, total_score):
        return ScoreInfo(operator="op", rel_path=rel_path, total_score=total_score)

    def test_overall_score_sums_all(self):
        infos = [
            self._make_info("level1/a", 10),
            self._make_info("level2/b", 20),
            self._make_info("level3/c", 30),
        ]
        assert ScoringCalculator().calculate_overall_score(infos) == 60

    def test_level_score_filters_by_rel_path(self):
        infos = [
            self._make_info("level1/a", 10),
            self._make_info("level1/b", 15),
            self._make_info("level2/c", 20),
        ]
        sc = ScoringCalculator()
        assert sc.calculate_level_score(infos, "level1") == 25
        assert sc.calculate_level_score(infos, "level2") == 20
        assert sc.calculate_level_score(infos, "level3") == 0  # 无匹配

    def test_overall_equals_sum_of_levels(self):
        infos = [
            self._make_info("level1/a", 10),
            self._make_info("level1/b", 15),
            self._make_info("level2/c", 20),
            self._make_info("level3/d", 5),
        ]
        sc = ScoringCalculator()
        total = sum(sc.calculate_level_score(infos, lv) for lv in sc.list_levels(infos))
        assert total == sc.calculate_overall_score(infos)

    def test_list_levels_preserves_first_occurrence_order(self):
        infos = [
            self._make_info("level3/a", 1),
            self._make_info("level1/b", 1),
            self._make_info("level3/c", 1),
            self._make_info("level2/d", 1),
        ]
        assert ScoringCalculator().list_levels(infos) == ["level3", "level1", "level2"]
