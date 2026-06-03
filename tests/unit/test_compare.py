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
张量对比模块单元测试

测试对象：kernel_eval.utils.compare
"""

import pytest
import torch

from kernel_eval.utils.compare import (
    SingleOutputResult,
    CompareResult,
    compare_tensors,
)


class TestSingleOutputResult:
    """SingleOutputResult 数据类测试"""

    def test_creation_float(self):
        r = SingleOutputResult(index=0, dtype="float32", passed=True,
                               metadata={"dtype_category": "float", "threshold": 2**-13,
                                         "mere": 0.001, "mare": 0.005})
        assert r.index == 0
        assert r.dtype == "float32"
        assert r.passed is True

    def test_creation_int(self):
        r = SingleOutputResult(index=1, dtype="int64", passed=True,
                               mismatch_count=0, total_count=100,
                               metadata={"dtype_category": "int", "threshold": 0})
        assert r.metadata["dtype_category"] == "int"
        assert r.mismatch_count == 0

    def test_to_dict(self):
        r = SingleOutputResult(index=0, dtype="float32", passed=True,
                               metadata={"threshold": 0.001, "mere": 1e-5, "mare": 2e-5})
        d = r.to_dict()
        assert d["index"] == 0
        assert d["passed"] is True
        assert d["mere"] == 1e-5

    def test_format_summary_float_pass(self):
        r = SingleOutputResult(index=0, dtype="float32", passed=True,
                               metadata={"dtype_category": "float", "threshold": 2**-13,
                                         "mere": 1e-5, "mare": 2e-5})
        s = r.format_summary()
        assert "✅" in s

    def test_format_summary_float_fail(self):
        r = SingleOutputResult(index=0, dtype="float32", passed=False,
                               error_msg="MARE exceeded threshold",
                               metadata={"dtype_category": "float", "threshold": 2**-13,
                                         "mere": 0.01, "mare": 0.05})
        s = r.format_summary()
        assert "❌" in s

    def test_format_summary_int_pass(self):
        r = SingleOutputResult(index=0, dtype="int64", passed=True,
                               mismatch_count=0, total_count=100,
                               metadata={"dtype_category": "int", "threshold": 0})
        s = r.format_summary()
        assert "✅" in s

    def test_format_summary_int_fail(self):
        r = SingleOutputResult(index=0, dtype="int64", passed=False,
                               mismatch_count=5, total_count=100,
                               metadata={"dtype_category": "int", "threshold": 0})
        s = r.format_summary()
        assert "❌" in s


class TestCompareResult:
    """CompareResult 数据类测试"""

    def test_creation(self):
        result = CompareResult(
            passed=True, dtype="float32", threshold=2**-13,
            mere=1e-5, mare=2e-5,
        )
        assert result.passed is True
        assert result.dtype == "float32"

    def test_to_dict(self):
        result = CompareResult(
            passed=True, dtype="float32", threshold=0.0001,
            mere=1e-5, mare=2e-5, mismatch_count=0, total_count=100,
        )
        d = result.to_dict()
        assert d["passed"] is True
        assert d["mismatch_count"] == 0
        assert "output_results" in d

    def test_failed_result(self):
        result = CompareResult(
            passed=False, dtype="float32", threshold=0.0001,
            mere=0.01, mare=0.05, error_msg="MARE exceeded threshold",
        )
        assert result.passed is False
        assert result.error_msg == "MARE exceeded threshold"

    def test_accuracy_type_check_preserves_diagnostic_context(self):
        from kernel_eval.eval.accuracy_eval import AccuracyEvaluator

        evaluator = AccuracyEvaluator()
        result = evaluator.evaluate(
            ai_output=None,
            golden_output=torch.ones(1),
            dtype="float32",
            diagnostic_context="raw call failed: unexpected keyword x",
        )

        assert result.passed is False
        assert "输出类型不支持: NoneType" in result.error_msg
        assert "raw call failed: unexpected keyword x" in result.error_msg

    def test_default_values(self):
        result = CompareResult(passed=True, dtype="float32", threshold=0.0001)
        assert result.mere == 0.0
        assert result.mare == 0.0

    def test_format_all_outputs(self):
        sr = SingleOutputResult(index=0, dtype="float32", passed=True,
                                metadata={"dtype_category": "float", "threshold": 2**-13,
                                          "mere": 1e-5, "mare": 2e-5})
        result = CompareResult(passed=True, dtype="float32", threshold=2**-13,
                               output_results=[sr])
        s = result.format_all_outputs()
        assert "float32" in s


class TestCompareTensors:
    """compare_tensors 函数测试"""

    def test_identical_tensors_pass(self):
        golden = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        output = golden.clone()
        result = compare_tensors(output, golden, "float32")
        assert result.passed is True
        assert result.mere == pytest.approx(0.0, abs=1e-10)

    def test_small_difference_pass(self):
        golden = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        output = golden + 1e-6
        result = compare_tensors(output, golden, "float32")
        assert result.passed is True

    def test_large_difference_fail(self):
        golden = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        output = golden + 0.1
        result = compare_tensors(output, golden, "float32")
        assert result.passed is False

    def test_int_tensors_exact_match(self):
        golden = torch.tensor([1, 2, 3], dtype=torch.int64)
        output = golden.clone()
        result = compare_tensors(output, golden, "int32")
        assert result.passed is True

    def test_int_tensors_mismatch(self):
        golden = torch.tensor([1, 2, 3], dtype=torch.int64)
        output = torch.tensor([1, 2, 4], dtype=torch.int64)
        result = compare_tensors(output, golden, "int32")
        assert result.passed is False

    def test_multiple_outputs(self):
        golden = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]
        output = [golden[0].clone(), golden[1].clone()]
        result = compare_tensors(output, golden, "float32")
        assert result.passed is True

    def test_output_count_mismatch(self):
        golden = [torch.tensor([1.0]), torch.tensor([2.0])]
        output = [torch.tensor([1.0])]
        result = compare_tensors(output, golden, "float32")
        assert result.passed is False

    def test_shape_mismatch(self):
        golden = torch.tensor([1.0, 2.0, 3.0])
        output = torch.tensor([1.0, 2.0])
        result = compare_tensors(output, golden, "float32")
        assert result.passed is False

    def test_float16_threshold(self):
        golden = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        output = golden + 0.0005
        result = compare_tensors(output, golden, "float16")
        assert result.passed is True

    def test_empty_tensor(self):
        golden = torch.tensor([], dtype=torch.float64)
        output = torch.tensor([], dtype=torch.float64)
        result = compare_tensors(output, golden, "float32")
        assert result.passed is True


class TestSmallValueFallback:
    """小值域兜底判定修复测试

    Bug 场景：当 native_output=None 时，CPU 参考是 perfect truncation（cpu_diff=0），
    导致 small_value_cpu_error_count=0。旧逻辑 ratio = NPU_errors / max(0,1) = NPU_errors，
    即使 NPU 在小值域有巨大误差（如 golden=1e-6, output=65504）也会通过兜底。

    修复：当 cpu_error_count=0 时，要求 NPU 也必须无错误才算通过。
    """

    def test_small_value_huge_error_fails_when_cpu_clean(self):
        """golden 很小 + output 极大 → 应失败（修复前会误判通过）"""
        # 构造场景：约一半 golden 值很小（小值域区域）
        n = 1000
        golden = torch.zeros(n, dtype=torch.float64)
        small_count = 500
        golden[:small_count] = 1e-6   # 小值域区域
        golden[small_count:] = 1.0    # 正常区域

        # output 基本匹配，但在一个小值域位置有巨大偏差
        output = golden.half()
        output[0] = 65504.0  # fp16 max，误差 = 65504 / 1e-6 = 6.55e10

        # native_output=None → CPU 参考是 golden_truncated（完美截断，cpu_diff=0）
        result = compare_tensors(output, golden, "float16")

        # 修复后应失败
        assert result.passed is False
        assert result.small_value_error_count > 0
        assert result.small_value_cpu_error_count == 0
        assert result.mismatch_count > 0

    def test_small_value_passes_when_both_npu_and_cpu_have_errors(self):
        """NPU 和 CPU 都有小值域误差（ratio ≤ 2）→ 应通过"""
        # 构造 native_output 使得 CPU 也有小值域误差
        n = 100
        golden = torch.zeros(n, dtype=torch.float64)
        golden[:50] = 1e-6   # 小值域
        golden[50:] = 1.0    # 正常

        output = golden.half()
        # NPU 在小值域有 2 个错误
        output[0] = 0.001   # 误差 > small_value_error
        output[1] = 0.002

        # native_output 也有小值域误差（模拟 CPU 精度截断也会犯错）
        native = golden.half()
        native[0] = 0.0015  # CPU 也有误差
        native[1] = 0.0025

        result = compare_tensors(output, golden, "float16", native_output=native)

        # NPU=2, CPU=2, ratio=1 ≤ 2 → 通过
        assert result.passed is True

    def test_small_value_fails_when_ratio_exceeds_2(self):
        """NPU 误差远多于 CPU（ratio > 2）→ 应失败"""
        n = 100
        golden = torch.zeros(n, dtype=torch.float64)
        golden[:50] = 1e-6
        golden[50:] = 1.0

        output = golden.half()
        # NPU 在小值域有 10 个错误
        for i in range(10):
            output[i] = 0.001 + i * 0.001

        # native_output 只有 2 个错误
        native = golden.half()
        native[0] = 0.001
        native[1] = 0.002

        result = compare_tensors(output, golden, "float16", native_output=native)

        # NPU=10, CPU=2, ratio=5 > 2 → 失败
        assert result.passed is False

    def test_small_value_no_errors_passes(self):
        """小值域区域无任何误差 → 应通过"""
        n = 100
        golden = torch.zeros(n, dtype=torch.float64)
        golden[:50] = 1e-6
        golden[50:] = 1.0

        output = golden.half()  # 完美截断

        result = compare_tensors(output, golden, "float16")

        assert result.passed is True
        assert result.small_value_error_count == 0


class TestCancelFallback:
    """相消区域兜底判定修复测试（与小值域相同逻辑）"""

    def test_cancel_huge_error_fails_when_cpu_clean(self):
        """相消区域巨大误差 + CPU 无错误 → 应失败"""
        # 构造相消场景：output ≈ 0，golden 在 cancel_boundary 附近
        n = 100
        golden = torch.full((n,), 0.01, dtype=torch.float64)  # 在 cancel boundary 附近
        output = torch.zeros(n, dtype=torch.float16)
        # 一个位置 output 不接近 0（触发 cancel error）
        output[0] = 0.5

        result = compare_tensors(output, golden, "float16")

        # 如果有 cancel mismatch 且 CPU 无错误，应失败
        if result.cancel_error_count > 0:
            assert result.passed is False


class TestBitExactFloat:
    """threshold=0 触发的浮点 bit-exact 路径"""

    BIT_EXACT = {"float16": 0, "bfloat16": 0, "float32": 0}

    def test_identical_fp32_passes(self):
        a = torch.tensor([1.0, -1.0, 0.0, float("inf"), -float("inf")], dtype=torch.float32)
        result = compare_tensors(a.clone(), a, "float32", custom_thresholds=self.BIT_EXACT)
        assert result.passed is True
        assert result.mismatch_count == 0

    def test_signed_zero_divergence_fails_fp32(self):
        out = torch.tensor([+0.0, 1.0, 2.0], dtype=torch.float32)
        gold = torch.tensor([-0.0, 1.0, 2.0], dtype=torch.float32)
        result = compare_tensors(out, gold, "float32", custom_thresholds=self.BIT_EXACT)
        assert result.passed is False
        assert result.mismatch_count == 1

    def test_signed_inf_divergence_fails(self):
        out = torch.tensor([float("inf"), 1.0], dtype=torch.float32)
        gold = torch.tensor([-float("inf"), 1.0], dtype=torch.float32)
        result = compare_tensors(out, gold, "float32", custom_thresholds=self.BIT_EXACT)
        assert result.passed is False

    def test_one_ulp_off_fails(self):
        a = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        b = a.clone()
        b[1] = torch.nextafter(b[1], torch.tensor(100.0))
        result = compare_tensors(a, b, "float32", custom_thresholds=self.BIT_EXACT)
        assert result.passed is False

    def test_fp64_golden_against_target_dtype_output(self):
        for target_dtype, dtype_str in [
            (torch.bfloat16, "bfloat16"),
            (torch.float16, "float16"),
            (torch.float32, "float32"),
        ]:
            x = torch.tensor([1.0, 2.0, 3.0, 1.0, 2.0], dtype=target_dtype)
            x_fp64 = x.double()
            golden_fp64, _ = torch.unique(x_fp64, return_inverse=True)
            output, _ = torch.unique(x, return_inverse=True)
            result = compare_tensors(output, golden_fp64, dtype_str,
                                     custom_thresholds=self.BIT_EXACT)
            assert result.passed, f"{target_dtype} round-trip via fp64 should pass bit-exact"
