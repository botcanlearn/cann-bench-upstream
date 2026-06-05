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
回归测试：KernelDetailsStrategy 的性能口径必须计入 direct-launch / 自定义
AscendC kernel（如 *_custom），不能只统计 aclnn 辅助 kernel。

历史 bug：parse_trace_view_kernels 只识别 `aclnn*AiCore_` 命名，当一个 case
同时包含小的 ACLNN 辅助 kernel（如 Fill 2us）和大的自定义 kernel（如
cummin_custom 33429us）时，trace_view 口径只统计到 Fill，elapsed_us 偏小、
speedup 虚高。修复：以 kernel_details.csv（完整 kernel 列表）为权威源。
"""

import csv

import pytest

from kernel_eval.base.perf_strategy import (
    ProfFileLocations,
    KernelDetailsStrategy,
    parse_csv_kernels,
)
from kernel_eval.base.result import PerfResult


# 复现 job_6e7e9bd422b2 的 level2/cummin_13：一个 Fill 辅助 kernel + 一个
# 占绝大多数耗时的 cummin_custom + 若干其它真实 kernel。
_KERNELS = [
    # (Name, Type, Duration(us), Input Shapes)
    ("aclnnInplaceFillScalar_FillAiCore_Fill", "Fill", "2", ""),
    ("ReduceMax", "ReduceMax", "158", "1,1024"),  # 非 warmup 形状 → 必须保留
    ("ConcatD", "ConcatD", "5", ""),
    ("cummin_custom", "cummin_custom", "33429", ""),  # direct-launch 自定义 kernel
]
_EXPECTED_TOTAL = 2 + 158 + 5 + 33429  # 33594


def _write_kernel_details_csv(path, n_steps=3):
    """写一个跨 n_steps 个 ProfilerStep、每步含上述 kernel 的 CSV。"""
    fieldnames = ["Step Id", "Name", "Type", "Duration(us)", "Input Shapes"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for step in range(n_steps):
            for name, op_type, dur, shapes in _KERNELS:
                writer.writerow({
                    "Step Id": str(step),
                    "Name": name,
                    "Type": op_type,
                    "Duration(us)": dur,
                    "Input Shapes": shapes,
                })


class TestParseCsvKernelsIncludesCustom:
    """parse_csv_kernels 必须计入自定义 kernel。"""

    def test_custom_kernel_present_and_total_correct(self, tmp_path):
        csv_path = tmp_path / "kernel_details.csv"
        _write_kernel_details_csv(csv_path)

        data = parse_csv_kernels(str(csv_path))

        assert "cummin_custom" in data["device_kernels"], \
            "自定义 kernel cummin_custom 被漏掉了"
        assert data["device_kernels"]["cummin_custom"] == pytest.approx(33429)
        # 总耗时应包含所有真实 kernel，而不是只有 Fill
        assert data["total_kernel_us"] == pytest.approx(_EXPECTED_TOTAL)
        # 绝不能退化成只有 Fill 的 2us
        assert data["total_kernel_us"] != pytest.approx(2.0)


class TestKernelDetailsStrategyPrefersCsv:
    """KernelDetailsStrategy 应以 CSV 为权威源，elapsed 计入自定义 kernel。"""

    def test_elapsed_includes_custom_kernel(self, tmp_path):
        csv_path = tmp_path / "kernel_details.csv"
        _write_kernel_details_csv(csv_path)

        prof_files = ProfFileLocations(
            csv_path=str(csv_path),
            trace_view_path=None,  # 仅 CSV
        )
        result = KernelDetailsStrategy().parse(prof_files, PerfResult())

        assert result.elapsed_us == pytest.approx(_EXPECTED_TOTAL)
        assert result.elapsed_us != pytest.approx(2.02), \
            "elapsed 退化成了 Fill 的耗时（历史 bug）"
        assert "cummin_custom" in result.op_times["device_kernels"]
        assert result.metadata["data_source"] == "kernel_details_csv"

    def test_speedup_back_to_sane_magnitude(self, tmp_path):
        """以真实 baseline 反推，speedup 应回落到合理量级（~22x，而非 ~37 万）。"""
        csv_path = tmp_path / "kernel_details.csv"
        _write_kernel_details_csv(csv_path)

        result = KernelDetailsStrategy().parse(
            ProfFileLocations(csv_path=str(csv_path)), PerfResult()
        )

        baseline_us = 749479.0  # job 中记录的基线
        speedup = baseline_us / result.elapsed_us
        assert speedup < 100, f"speedup 仍然虚高: {speedup:.1f}x"
        assert speedup == pytest.approx(22.4, abs=1.0)
