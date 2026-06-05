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
回归:perf_result 缺失(--no-perf / 非 profiler 路径,score_i=None)时,
aggregate_eq4 仅把该 case 的 **perf 分按 0 计入**,而 function/compilation/total
仍照常计算。钉死"墙钟弃用后,缺 perf 不破坏总分"的不变量。
"""

from kernel_eval.report.scoring import (
    aggregate_eq4,
    WEIGHT_COMPILATION,
    WEIGHT_FUNCTION,
    WEIGHT_PERFORMANCE,
)


def test_missing_perf_zeros_only_perf_keeps_function_and_total():
    # 两个精度通过的 case:一个有 perf 分 0.8,一个无 perf(None)
    agg = aggregate_eq4(
        compile_passed=True,
        total_cases=2,
        case_scores=[(True, 0.8), (True, None)],
    )
    # 准确性:两个都过 → wf * 2/2
    assert agg["function_score"] == WEIGHT_FUNCTION * (2 / 2) * 100
    # 性能:None 计 0 → 仅 0.8 计入,分母仍是 total_cases=2(不是 1)
    assert abs(agg["performance_score"] - WEIGHT_PERFORMANCE * (0.8 / 2) * 100) < 1e-9
    # 编译分照常
    assert agg["compilation_score"] == WEIGHT_COMPILATION * 100
    # 总分 = 三项和,可计算
    assert abs(
        agg["total_score"]
        - (agg["compilation_score"] + agg["function_score"] + agg["performance_score"])
    ) < 1e-9
    # 缺 perf 的 case 在 per_case_scores 里记为 None(不污染为 0/NaN)
    assert agg["per_case_scores"] == [0.8, None]


def test_all_missing_perf_still_scores_function_and_total():
    # 全部精度通过但都无 perf:perf 分整体 0,function/compilation/total 仍算
    agg = aggregate_eq4(
        compile_passed=True,
        total_cases=2,
        case_scores=[(True, None), (True, None)],
    )
    assert agg["performance_score"] == 0.0
    assert agg["function_score"] == WEIGHT_FUNCTION * (2 / 2) * 100
    assert agg["compilation_score"] == WEIGHT_COMPILATION * 100
    assert agg["total_score"] == agg["compilation_score"] + agg["function_score"]


def test_missing_perf_does_not_penalize_a_failed_case_differently():
    # 一个通过(无 perf)+ 一个未通过:function 只按通过数,perf 全 0,total 仍算
    agg = aggregate_eq4(
        compile_passed=True,
        total_cases=2,
        case_scores=[(True, None), (False, None)],
    )
    assert agg["function_score"] == WEIGHT_FUNCTION * (1 / 2) * 100
    assert agg["performance_score"] == 0.0
    assert agg["per_case_scores"] == [None, None]
