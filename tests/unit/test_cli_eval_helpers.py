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
验证 P0-4: cli.py cmd_eval_process 中两段重复的 case 评测循环抽取为共享 helper

目标：rel_paths 模式与 cases_file 模式共享 _evaluate_cases_batch 和 _build_op_result_dict，
     消除 ~90 行重复代码。
"""

from unittest.mock import MagicMock, patch
import pytest

from kernel_eval.benches import CannCaseSpec
from kernel_eval.eval.results import EvalCaseResult, EvalOperatorResult, summarize_case_results


def _make_case(case_id="test_1", rel_path="level1/test", operator="Test", case_num=1,
               baseline_perf_us=100.0, t_hw_us=50.0):
    """构造最小 CannCaseSpec fixture"""
    return CannCaseSpec(
        case_id=case_id,
        rel_path=rel_path,
        operator=operator,
        case_num=case_num,
        input_shapes=[[2, 3]],
        dtypes=["float32"],
        baseline_perf_us=baseline_perf_us,
        t_hw_us=t_hw_us,
    )


def _make_eval_case_result(case_id="test_1", rel_path="level1/test", operator="Test",
                            case_num=1, success=True, elapsed_us=80.0, baseline_perf_us=100.0):
    """构造一个成功的 EvalCaseResult"""
    from kernel_eval.base.result import PerfResult
    from kernel_eval.eval.accuracy_eval import AccuracyResult

    perf = PerfResult(elapsed_us=elapsed_us, metadata={'t_hw_us': 50.0})
    acc = AccuracyResult(passed=True, threshold=0.001, metadata={'mere': 0.001, 'mare': 0.001})
    return EvalCaseResult(
        case_id=case_id,
        rel_path=rel_path,
        operator=operator,
        case_num=case_num,
        success=success,
        accuracy_result=acc,
        perf_result=perf,
        baseline_perf_us=baseline_perf_us,
        t_hw_us=50.0,
    )


class TestEvaluateCasesBatch:
    """_evaluate_cases_batch 应遍历 cases 逐条评测并返回结果列表"""

    def test_returns_case_results_for_all_cases(self):
        """两个用例都应被评测并返回"""
        from kernel_eval.cli import _evaluate_cases_batch

        cases = [_make_case("t_1", case_num=1), _make_case("t_2", case_num=2)]
        mock_evaluator = MagicMock()
        mock_evaluator.evaluate_case.return_value = _make_eval_case_result("t_1", case_num=1)
        mock_golden = MagicMock()
        mock_golden.get_golden_function.return_value = lambda *a, **kw: None

        results = _evaluate_cases_batch(mock_evaluator, cases, mock_golden, process_id="99")

        assert len(results) == 2
        assert mock_evaluator.evaluate_case.call_count == 2

    def test_passes_golden_func_to_evaluator(self):
        """golden_loader.get_golden_function 的结果应传给 evaluator"""
        from kernel_eval.cli import _evaluate_cases_batch

        cases = [_make_case("t_1")]
        mock_evaluator = MagicMock()
        mock_evaluator.evaluate_case.return_value = _make_eval_case_result("t_1")
        mock_golden = MagicMock()
        fake_func = lambda: "golden"
        mock_golden.get_golden_function.return_value = fake_func

        _evaluate_cases_batch(mock_evaluator, cases, mock_golden, process_id="0")

        call_args = mock_evaluator.evaluate_case.call_args
        assert call_args[0][0] == cases[0]
        assert call_args[0][1] is fake_func

    def test_empty_cases_returns_empty_list(self):
        """空用例列表应返回空结果"""
        from kernel_eval.cli import _evaluate_cases_batch

        mock_evaluator = MagicMock()
        mock_golden = MagicMock()
        results = _evaluate_cases_batch(mock_evaluator, [], mock_golden, process_id="0")
        assert results == []


class TestBuildOpResultDict:
    """_build_op_result_dict 应根据 case_results 构建算子级结果字典"""

    def test_all_passed(self):
        """全部通过时应正确统计"""
        from kernel_eval.cli import _build_op_result_dict

        case_results = [
            _make_eval_case_result("t_1", case_num=1, success=True),
            _make_eval_case_result("t_2", case_num=2, success=True),
        ]
        d = _build_op_result_dict("level1/test", "Test", case_results)

        assert d['operator'] == 'Test'
        assert d['rel_path'] == 'level1/test'
        assert d['total_cases'] == 2
        assert d['passed_cases'] == 2
        assert d['failed_cases'] == 0
        assert d['pass_rate'] == 1.0

    def test_mixed_results(self):
        """混合通过/失败时应正确统计"""
        from kernel_eval.cli import _build_op_result_dict

        case_results = [
            _make_eval_case_result("t_1", case_num=1, success=True),
            _make_eval_case_result("t_2", case_num=2, success=False),
            _make_eval_case_result("t_3", case_num=3, success=True),
        ]
        d = _build_op_result_dict("level1/test", "Test", case_results)

        assert d['passed_cases'] == 2
        assert d['failed_cases'] == 1
        assert d['pass_rate'] == pytest.approx(2.0 / 3.0)

    def test_empty_results(self):
        """空结果列表应返回全零"""
        from kernel_eval.cli import _build_op_result_dict

        d = _build_op_result_dict("level1/x", "X", [])

        assert d['total_cases'] == 0
        assert d['passed_cases'] == 0
        assert d['pass_rate'] == 0.0

    def test_output_is_json_serializable_dict(self):
        """返回值应可直接 json.dumps（to_dict 兼容）"""
        import json
        from kernel_eval.cli import _build_op_result_dict

        case_results = [_make_eval_case_result("t_1")]
        d = _build_op_result_dict("level1/test", "Test", case_results)

        json.dumps(d)  # 不应抛出异常
