#!/usr/bin/python3
# coding=utf-8

# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""
评测结果数据类

职责：
1. 定义评测结果数据结构
2. 提供 to_dict 序列化方法

从 evaluator.py 拆分出来，避免循环导入。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

from .op_runner import OpRunResult
from .accuracy_eval import AccuracyResult
from .perf_eval import PerfResult
from ..data.package_manager import PackageInfo


@dataclass
class EvalCaseResult:
    """单用例评测结果"""
    case_id: str
    rel_path: str           # 替代 level，使用相对路径
    operator: str
    case_num: int
    success: bool
    accuracy_result: Optional[AccuracyResult] = None
    perf_result: Optional[PerfResult] = None
    golden_run_result: Optional[OpRunResult] = None
    ai_run_result: Optional[OpRunResult] = None
    error_msg: Optional[str] = None
    baseline_perf_us: float = 0.0

    def get_speedup(self) -> float:
        """计算加速比"""
        if self.perf_result and self.baseline_perf_us > 0:
            return self.baseline_perf_us / self.perf_result.elapsed_us if self.perf_result.elapsed_us > 0 else 0.0
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'case_id': self.case_id,
            'rel_path': self.rel_path,
            'operator': self.operator,
            'case_num': self.case_num,
            'success': self.success,
            'accuracy': self.accuracy_result.to_dict() if self.accuracy_result else None,
            'perf': {
                'elapsed_us': self.perf_result.elapsed_us if self.perf_result else 0,
                'speedup': self.get_speedup(),
                'op_times': self.perf_result.op_times if self.perf_result else {},
            } if self.perf_result else None,
            'golden_elapsed_us': self.golden_run_result.elapsed_us if self.golden_run_result else 0,
            'ai_elapsed_us': self.ai_run_result.elapsed_us if self.ai_run_result else 0,
            'error_msg': self.error_msg,
            'baseline_perf_us': self.baseline_perf_us,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EvalCaseResult':
        """从字典反序列化"""
        perf_data = data.get('perf')
        perf_result = None
        if perf_data:
            perf_result = PerfResult(
                case_id=data.get('case_id', ''),
                elapsed_us=perf_data.get('elapsed_us', 0),
                op_times=perf_data.get('op_times', {}),
            )

        accuracy_data = data.get('accuracy')
        accuracy_result = None
        if accuracy_data:
            accuracy_result = AccuracyResult(
                passed=accuracy_data.get('passed', True),
                dtype=accuracy_data.get('dtype', 'float32'),
                threshold=accuracy_data.get('threshold', 0.001),
                mare=accuracy_data.get('mare', 0.0),
                mere=accuracy_data.get('mere', 0.0),
                max_diff=accuracy_data.get('max_diff', 0.0),
                mean_diff=accuracy_data.get('mean_diff', 0.0),
            )

        return cls(
            case_id=data.get('case_id', ''),
            rel_path=data.get('rel_path', ''),
            operator=data.get('operator', ''),
            case_num=data.get('case_num', 0),
            success=data.get('success', False),
            accuracy_result=accuracy_result,
            perf_result=perf_result,
            error_msg=data.get('error_msg'),
            baseline_perf_us=data.get('baseline_perf_us', 0.0),
        )


@dataclass
class EvalOperatorResult:
    """算子评测结果"""
    rel_path: str           # 替代 level，使用相对路径
    operator: str
    total_cases: int
    passed_cases: int
    failed_cases: int
    skipped_cases: int
    results: List[EvalCaseResult]
    pass_rate: float
    avg_speedup: float
    # 当算子跑不起来时附带的诊断信息
    compilation_error: Optional[str] = None
    subprocess_failure_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {
            'rel_path': self.rel_path,
            'operator': self.operator,
            'total_cases': self.total_cases,
            'passed_cases': self.passed_cases,
            'failed_cases': self.failed_cases,
            'skipped_cases': self.skipped_cases,
            'pass_rate': self.pass_rate,
            'avg_speedup': self.avg_speedup,
            'results': [r.to_dict() for r in self.results],
        }
        if self.compilation_error:
            d['compilation_error'] = self.compilation_error
        if self.subprocess_failure_reason:
            d['subprocess_failure_reason'] = self.subprocess_failure_reason
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EvalOperatorResult':
        """从字典反序列化"""
        results_data = data.get('results', [])
        results = [EvalCaseResult.from_dict(r) for r in results_data]

        return cls(
            rel_path=data.get('rel_path', ''),
            operator=data.get('operator', ''),
            total_cases=data.get('total_cases', 0),
            passed_cases=data.get('passed_cases', 0),
            failed_cases=data.get('failed_cases', 0),
            skipped_cases=data.get('skipped_cases', 0),
            results=results,
            pass_rate=data.get('pass_rate', 0.0),
            avg_speedup=data.get('avg_speedup', 0.0),
            compilation_error=data.get('compilation_error'),
            subprocess_failure_reason=data.get('subprocess_failure_reason'),
        )


@dataclass
class EvalSessionResult:
    """评测会话结果"""
    operators: List[EvalOperatorResult]
    package_info: Optional[PackageInfo] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'operators': [op.to_dict() for op in self.operators],
            'package_info': {
                'source_dir': self.package_info.source_dir if self.package_info else '',
                'whl_path': self.package_info.whl_path if self.package_info else '',
                'run_path': self.package_info.run_path if self.package_info else '',
            } if self.package_info else None,
        }