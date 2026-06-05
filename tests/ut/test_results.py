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
评测结果数据类单元测试

测试对象：kernel_eval.eval.results
核心功能：
1. EvalCaseResult - 单用例结果
2. EvalOperatorResult - 算子结果
3. EvalSessionResult - 会话结果
4. to_dict 序列化
"""

import pytest

from kernel_eval.eval.results import (
    EvalCaseResult,
    EvalOperatorResult,
    EvalSessionResult,
)


class TestEvalCaseResult:
    """EvalCaseResult 数据类测试"""

    def test_basic_creation(self):
        """基本创建"""
        result = EvalCaseResult(
            case_id="case_001",
            rel_path="level1/Exp",
            operator="Exp",
            case_num=1,
            success=True,
        )
        assert result.case_id == "case_001"
        assert result.rel_path == "level1/Exp"
        assert result.operator == "Exp"
        assert result.case_num == 1
        assert result.success is True

    def test_with_error_msg(self):
        """带错误消息"""
        result = EvalCaseResult(
            case_id="case_002",
            rel_path="level1/Add",
            operator="Add",
            case_num=2,
            success=False,
            error_msg="Runtime error: shape mismatch",
        )
        assert result.success is False
        assert result.error_msg == "Runtime error: shape mismatch"

    def test_with_baseline_perf(self):
        """带 baseline 性能"""
        result = EvalCaseResult(
            case_id="case_003",
            rel_path="level1/Mul",
            operator="Mul",
            case_num=3,
            success=True,
            baseline_perf_us=100.0,
        )
        assert result.baseline_perf_us == 100.0

    def test_get_speedup_no_perf_result(self):
        """无性能结果时加速比为 0"""
        result = EvalCaseResult(
            case_id="case_001",
            rel_path="level1/Exp",
            operator="Exp",
            case_num=1,
            success=True,
            baseline_perf_us=100.0,
        )
        assert result.get_speedup() == 0.0

    def test_get_speedup_zero_baseline(self):
        """baseline 为 0 时加速比为 0"""
        # 需要模拟 perf_result
        from kernel_eval.eval.perf_eval import PerfResult
        perf = PerfResult(elapsed_us=50.0, op_times={})
        result = EvalCaseResult(
            case_id="case_001",
            rel_path="level1/Exp",
            operator="Exp",
            case_num=1,
            success=True,
            perf_result=perf,
            baseline_perf_us=0.0,
        )
        assert result.get_speedup() == 0.0

    def test_get_speedup_normal(self):
        """正常加速比计算"""
        from kernel_eval.eval.perf_eval import PerfResult
        perf = PerfResult(elapsed_us=50.0, metadata={'baseline_us': 100.0})
        result = EvalCaseResult(
            case_id="case_001",
            rel_path="level1/Exp",
            operator="Exp",
            case_num=1,
            success=True,
            perf_result=perf,
            baseline_perf_us=100.0,
        )
        # speedup = baseline / elapsed = 100 / 50 = 2.0
        assert result.get_speedup() == 2.0

    def test_to_dict_basic(self):
        """基本序列化"""
        result = EvalCaseResult(
            case_id="case_001",
            rel_path="level1/Exp",
            operator="Exp",
            case_num=1,
            success=True,
            baseline_perf_us=100.0,
        )
        d = result.to_dict()
        assert d["case_id"] == "case_001"
        assert d["rel_path"] == "level1/Exp"
        assert d["operator"] == "Exp"
        assert d["success"] is True
        assert d["baseline_perf_us"] == 100.0
        assert d["accuracy"] is None
        assert d["perf"] is None

    def test_to_dict_with_error(self):
        """带错误消息序列化"""
        result = EvalCaseResult(
            case_id="case_002",
            rel_path="level1/Add",
            operator="Add",
            case_num=2,
            success=False,
            error_msg="Test error",
        )
        d = result.to_dict()
        assert d["success"] is False
        assert d["error_msg"] == "Test error"


class TestEvalOperatorResult:
    """EvalOperatorResult 数据类测试"""

    def test_basic_creation(self):
        """基本创建"""
        result = EvalOperatorResult(
            operator="Exp",
            rel_path="level1/Exp",
            total_cases=10,
            passed_cases=8,
            failed_cases=2,
            skipped_cases=0,
            results=[],
            pass_rate=0.8,
            avg_speedup=2.0,
        )
        assert result.operator == "Exp"
        assert result.rel_path == "level1/Exp"
        assert result.total_cases == 10
        assert result.passed_cases == 8
        assert result.failed_cases == 2
        assert result.pass_rate == 0.8

    def test_with_case_results(self):
        """带用例结果"""
        case_results = [
            EvalCaseResult(
                case_id="case_001",
                rel_path="level1/Exp",
                operator="Exp",
                case_num=1,
                success=True,
            ),
            EvalCaseResult(
                case_id="case_002",
                rel_path="level1/Exp",
                operator="Exp",
                case_num=2,
                success=False,
                error_msg="Error",
            ),
        ]
        result = EvalOperatorResult(
            operator="Exp",
            rel_path="level1/Exp",
            total_cases=2,
            passed_cases=1,
            failed_cases=1,
            skipped_cases=0,
            results=case_results,
            pass_rate=0.5,
            avg_speedup=0.0,
        )
        assert len(result.results) == 2

    def test_with_compilation_error(self):
        """带编译错误"""
        result = EvalOperatorResult(
            operator="CustomOp",
            rel_path="level2/CustomOp",
            total_cases=0,
            passed_cases=0,
            failed_cases=0,
            skipped_cases=0,
            results=[],
            pass_rate=0.0,
            avg_speedup=0.0,
            compilation_error="Compile failed: syntax error",
        )
        assert result.compilation_error == "Compile failed: syntax error"

    def test_with_subprocess_failure(self):
        """带子进程失败"""
        result = EvalOperatorResult(
            operator="CustomOp",
            rel_path="level2/CustomOp",
            total_cases=0,
            passed_cases=0,
            failed_cases=0,
            skipped_cases=0,
            results=[],
            pass_rate=0.0,
            avg_speedup=0.0,
            subprocess_failure_reason="Timeout after 60s",
        )
        assert result.subprocess_failure_reason == "Timeout after 60s"

    def test_to_dict(self):
        """序列化"""
        result = EvalOperatorResult(
            operator="Exp",
            rel_path="level1/Exp",
            total_cases=10,
            passed_cases=8,
            failed_cases=2,
            skipped_cases=0,
            results=[],
            pass_rate=0.8,
            avg_speedup=2.0,
        )
        d = result.to_dict()
        assert d["operator"] == "Exp"
        assert d["total_cases"] == 10
        assert d["pass_rate"] == 0.8
        assert "compilation_error" not in d  # 无编译错误时不包含

    def test_to_dict_with_errors(self):
        """带错误序列化"""
        result = EvalOperatorResult(
            operator="CustomOp",
            rel_path="level2/CustomOp",
            total_cases=0,
            passed_cases=0,
            failed_cases=0,
            skipped_cases=0,
            results=[],
            pass_rate=0.0,
            avg_speedup=0.0,
            compilation_error="Error",
            subprocess_failure_reason="Timeout",
        )
        d = result.to_dict()
        assert "compilation_error" in d
        assert "subprocess_failure_reason" in d


class TestEvalSessionResult:
    """EvalSessionResult 数据类测试"""

    def test_empty_session(self):
        """空会话"""
        result = EvalSessionResult(operators=[])
        assert len(result.operators) == 0
        assert result.package_info is None

    def test_with_operators(self):
        """带算子结果"""
        op_results = [
            EvalOperatorResult(
                operator="Exp",
                rel_path="level1/Exp",
                total_cases=5,
                passed_cases=5,
                failed_cases=0,
                skipped_cases=0,
                results=[],
                pass_rate=1.0,
                avg_speedup=2.0,
            ),
            EvalOperatorResult(
                operator="Add",
                rel_path="level1/Add",
                total_cases=3,
                passed_cases=2,
                failed_cases=1,
                skipped_cases=0,
                results=[],
                pass_rate=0.67,
                avg_speedup=1.5,
            ),
        ]
        result = EvalSessionResult(operators=op_results)
        assert len(result.operators) == 2

    def test_to_dict(self):
        """序列化"""
        result = EvalSessionResult(operators=[])
        d = result.to_dict()
        assert d["operators"] == []
        assert d["package_info"] is None


class TestResultIntegration:
    """结果集成测试"""

    def test_full_result_chain(self):
        """完整结果链"""
        # 创建 case 结果
        case_results = [
            EvalCaseResult(
                case_id=f"case_{i}",
                rel_path="level1/Exp",
                operator="Exp",
                case_num=i,
                success=True,
                baseline_perf_us=100.0,
            )
            for i in range(3)
        ]

        # 创建 operator 结果
        op_result = EvalOperatorResult(
            operator="Exp",
            rel_path="level1/Exp",
            total_cases=3,
            passed_cases=3,
            failed_cases=0,
            skipped_cases=0,
            results=case_results,
            pass_rate=1.0,
            avg_speedup=2.0,
        )

        # 创建 session 结果
        session_result = EvalSessionResult(operators=[op_result])

        assert len(session_result.operators) == 1
        assert session_result.operators[0].operator == "Exp"

    def test_serialization_chain(self):
        """序列化链"""
        case = EvalCaseResult(
            case_id="case_1",
            rel_path="level1/Exp",
            operator="Exp",
            case_num=1,
            success=True,
        )
        op = EvalOperatorResult(
            operator="Exp",
            rel_path="level1/Exp",
            total_cases=1,
            passed_cases=1,
            failed_cases=0,
            skipped_cases=0,
            results=[case],
            pass_rate=1.0,
            avg_speedup=0.0,
        )
        session = EvalSessionResult(operators=[op])

        session_dict = session.to_dict()
        assert session_dict["operators"][0]["operator"] == "Exp"
        assert session_dict["operators"][0]["results"][0]["case_id"] == "case_1"