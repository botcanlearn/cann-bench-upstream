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
StanfordBench 精度判断器

使用 torch.allclose(atol/rtol) 对比，而非 MERE/MARE。
"""

import torch
from typing import Dict, List, Optional, Union

from ..base.checker import CorrectnessChecker
from ..base.result import AccuracyResult
from ..utils.compare import SingleOutputResult


class StanfordChecker(CorrectnessChecker):
    """StanfordBench 精度判断器

    使用 torch.allclose 进行对比，默认 atol=rtol=0.01。

    与 CannBench 的 MERE/MARE + 小值域 + 相消处理不同，
    StanfordBench 使用更简单直接的 allclose 对比。
    """

    def get_name(self) -> str:
        return "stanford_default"

    def get_description(self) -> str:
        return "StanfordBench 精度判断器（torch.allclose）"

    def check(
        self,
        ai_outputs: Union[torch.Tensor, List[torch.Tensor]],
        golden_outputs: Union[torch.Tensor, List[torch.Tensor]],
        dtype: str,
        threshold: float = 0.01,
        native_outputs=None,
        ignore_indices: Optional[List[int]] = None,
        custom_thresholds: Optional[Dict[str, float]] = None,
    ) -> AccuracyResult:
        """使用 allclose 对比

        Args:
            ai_outputs: AI 算子输出
            golden_outputs: Golden 参考输出
            dtype: 数据类型
            threshold: 默认精度阈值（当 custom_thresholds 未指定时使用）
            native_outputs: 同精度参考输出（StanfordBench native_npu 模式下直接复用 golden）
            ignore_indices: 需要忽略对比的输出索引
            custom_thresholds: 自定义阈值 {'atol': x, 'rtol': y}

        Returns:
            AccuracyResult: 精度对比结果
        """
        ai_list = self._normalize_outputs(ai_outputs)
        golden_list = self._normalize_outputs(golden_outputs)

        # 检查输出数量是否匹配
        if len(ai_list) != len(golden_list):
            return AccuracyResult(
                passed=False,
                error_msg=f"输出数量不匹配: ai={len(ai_list)}, golden={len(golden_list)}"
            )

        # 获取阈值
        atol = custom_thresholds.get('atol', threshold) if custom_thresholds else threshold
        rtol = custom_thresholds.get('rtol', threshold) if custom_thresholds else threshold

        results: List[SingleOutputResult] = []
        all_passed = True

        for i, (ai, golden) in enumerate(zip(ai_list, golden_list)):
            # 检查是否需要忽略
            if ignore_indices and i in ignore_indices:
                results.append(SingleOutputResult(
                    index=i,
                    name=f"output_{i}",
                    passed=True,
                    error_msg="(跳过对比)",
                ))
                continue

            # 确保 tensor 在 CPU 上进行对比
            ai_cpu = self._ensure_cpu(ai)
            golden_cpu = self._ensure_cpu(golden)

            # 判断 dtype 类别
            dtype_category = 'float' if ai_cpu.dtype in (torch.float32, torch.float16, torch.float64, torch.bfloat16) else 'int'

            # 整数类型：直接比较是否相等
            if dtype_category == 'int':
                passed = torch.equal(ai_cpu, golden_cpu)
                max_diff = float((ai_cpu - golden_cpu).abs().max().item()) if ai_cpu.numel() > 0 else 0.0
                mean_diff = 0.0  # 整数类型不计算 mean_diff
                mismatch_count = int((ai_cpu != golden_cpu).sum().item())
                total_count = ai_cpu.numel()
            else:
                # 浮点类型：使用 allclose 对比
                passed = torch.allclose(ai_cpu, golden_cpu, rtol=rtol, atol=atol)

                # 计算差异统计
                diff = torch.abs(ai_cpu - golden_cpu)
                max_diff = float(diff.max().item()) if diff.numel() > 0 else 0.0
                mean_diff = float(diff.mean().item()) if diff.numel() > 0 else 0.0

                # 计算不匹配数量
                mismatch_count = int((diff > atol + rtol * torch.abs(golden_cpu)).sum().item())
                total_count = ai_cpu.numel()

            results.append(SingleOutputResult(
                index=i,
                name=f"output_{i}",
                passed=passed,
                threshold=threshold,
                max_diff=max_diff,
                mean_diff=mean_diff,
                mismatch_count=mismatch_count,
                total_count=total_count,
                dtype_category=dtype_category,
            ))

            if not passed:
                all_passed = False

        return AccuracyResult(
            passed=all_passed,
            output_results=results,
            threshold=threshold,
            error_msg=None if all_passed else "精度不达标（allclose 失败）",
        )