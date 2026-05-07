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
失败结果合成器

职责：
1. 为编译失败的算子合成 FAIL 结果
2. 为安全检查失败的算子合成 FAIL 结果
3. 为子进程异常的算子合成 FAIL 结果

这样失败算子仍然出现在 session 结果里，报告可见原因。
"""

from typing import Dict, List, Optional, Any

from .results import EvalCaseResult, EvalOperatorResult
from ..data.case_loader import CaseLoader
from ..data.operator_loader import OperatorInfo


class FailureSynthesizer:
    """失败结果合成器"""

    def __init__(self, case_loader: CaseLoader):
        self.case_loader = case_loader

    def synthesize_compile_failure(
        self,
        op_info: OperatorInfo,
        error_excerpt: str,
        case_filter: Optional[Dict] = None,
        filter_func: Optional[callable] = None,
    ) -> EvalOperatorResult:
        """为编译失败的算子生成一条 all-FAIL 的 EvalOperatorResult

        Args:
            op_info: 算子信息
            error_excerpt: 错误摘要
            case_filter: 用例筛选条件
            filter_func: 筛选函数（可选）

        Returns:
            EvalOperatorResult: 全 FAIL 的算子结果
        """
        try:
            cases = self.case_loader.scan_by_operator(op_info.name)
            if case_filter and filter_func:
                cases = filter_func(cases, case_filter)
        except Exception:
            cases = []

        # 取错误摘要的第一行做 case-level detail
        first_line = (error_excerpt.strip().splitlines() or ["(no detail)"])[0]
        reason_short = f"compile failed: {first_line[:180]}"

        case_results: List[EvalCaseResult] = []
        for c in cases:
            case_results.append(EvalCaseResult(
                case_id=str(getattr(c, "case_id", 0)),
                rel_path=op_info.rel_path,
                operator=op_info.name,
                case_num=int(getattr(c, "case_id", 0)),
                success=False,
                error_msg=reason_short,
            ))

        return EvalOperatorResult(
            rel_path=op_info.rel_path,
            operator=op_info.name,
            total_cases=len(case_results),
            passed_cases=0,
            failed_cases=len(case_results),
            skipped_cases=0,
            results=case_results,
            pass_rate=0.0,
            avg_speedup=0.0,
            compilation_error=error_excerpt,
        )

    def synthesize_security_failure(
        self,
        op_info: OperatorInfo,
        security_error: str,
        case_filter: Optional[Dict] = None,
        filter_func: Optional[callable] = None,
    ) -> EvalOperatorResult:
        """为安全检查失败的算子生成一条 all-FAIL 的 EvalOperatorResult

        Args:
            op_info: 算子信息
            security_error: 安全错误信息
            case_filter: 用例筛选条件
            filter_func: 筛选函数（可选）

        Returns:
            EvalOperatorResult: 全 FAIL 的算子结果
        """
        try:
            cases = self.case_loader.scan_by_operator(op_info.name)
            if case_filter and filter_func:
                cases = filter_func(cases, case_filter)
        except Exception:
            cases = []

        first_line = (security_error.strip().splitlines() or ["(no detail)"])[0]
        reason_short = f"security check failed: {first_line[:180]}"

        case_results: List[EvalCaseResult] = []
        for c in cases:
            case_results.append(EvalCaseResult(
                case_id=str(getattr(c, "case_id", 0)),
                rel_path=op_info.rel_path,
                operator=op_info.name,
                case_num=int(getattr(c, "case_id", 0)),
                success=False,
                error_msg=reason_short,
            ))

        return EvalOperatorResult(
            rel_path=op_info.rel_path,
            operator=op_info.name,
            total_cases=len(case_results),
            passed_cases=0,
            failed_cases=len(case_results),
            skipped_cases=0,
            results=case_results,
            pass_rate=0.0,
            avg_speedup=0.0,
            subprocess_failure_reason=security_error,
        )

    def synthesize_subprocess_failure(
        self,
        operator_name: str,
        rel_path: str = "",
        reason: str = "",
        case_filter: Optional[Dict] = None,
        filter_func: Optional[callable] = None,
    ) -> EvalOperatorResult:
        """子进程超时 / 崩溃时合成 all-FAIL 的 EvalOperatorResult

        Args:
            operator_name: 算子名称
            rel_path: 相对路径
            reason: 失败原因
            case_filter: 用例筛选条件
            filter_func: 筛选函数（可选）

        Returns:
            EvalOperatorResult: 全 FAIL 的算子结果
        """
        try:
            cases = self.case_loader.scan_by_operator(operator_name)
            if case_filter and filter_func:
                cases = filter_func(cases, case_filter)
        except Exception:
            cases = []

        short = f"subprocess failed: {reason}"
        case_results: List[EvalCaseResult] = []
        for c in cases:
            case_results.append(EvalCaseResult(
                case_id=str(getattr(c, "case_id", 0)),
                rel_path=rel_path,
                operator=operator_name,
                case_num=int(getattr(c, "case_id", 0)),
                success=False,
                error_msg=short,
            ))

        return EvalOperatorResult(
            rel_path=rel_path,
            operator=operator_name,
            total_cases=len(case_results),
            passed_cases=0,
            failed_cases=len(case_results),
            skipped_cases=0,
            results=case_results,
            pass_rate=0.0,
            avg_speedup=0.0,
            subprocess_failure_reason=reason,
        )