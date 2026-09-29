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

from pathlib import Path

import torch

from kernel_eval.benches.cann_loader import CannTaskLoader
from kernel_eval.base.result import FAILURE_TYPE_COMPILE_RUNTIME_ERROR, get_accuracy_failure_type
from kernel_eval.eval.accuracy_eval import AccuracyEvaluator
from kernel_eval.eval.evaluator import Evaluator


def test_candidate_output_must_match_proto_dtype_contract():
    tasks_root = Path(__file__).resolve().parents[2] / "tasks"
    nms = CannTaskLoader(tasks_root=str(tasks_root)).get_task("level3/nms")
    expected_dtypes = Evaluator._get_output_dtype_contracts(nms)
    output_names = [output.name for output in nms.outputs]

    golden = torch.tensor([0, 3], dtype=torch.int64)
    wrong_dtype = torch.tensor([0, 3], dtype=torch.int32)
    evaluator = AccuracyEvaluator()

    # 数值完全相同的跨整数 dtype 输出在未应用 proto 契约的旧路径上会通过。
    legacy_result = evaluator.evaluate(wrong_dtype, golden, dtype="int32")
    assert legacy_result.passed

    mismatch = evaluator.evaluate(
        wrong_dtype,
        golden,
        dtype="int32",
        expected_output_dtypes=expected_dtypes,
        output_names=output_names,
    )
    assert not mismatch.passed
    assert "keep_indices" in mismatch.error_msg
    assert "int64" in mismatch.error_msg
    assert "int32" in mismatch.error_msg
    assert get_accuracy_failure_type(mismatch) == FAILURE_TYPE_COMPILE_RUNTIME_ERROR

    transformed_output_mismatch = evaluator.evaluate(
        golden.clone(),
        golden,
        dtype="int64",
        expected_output_dtypes=expected_dtypes,
        output_names=output_names,
        output_dtype_source=wrong_dtype,
    )
    assert not transformed_output_mismatch.passed

    matching = evaluator.evaluate(
        golden.clone(),
        golden,
        dtype="int64",
        expected_output_dtypes=expected_dtypes,
        output_names=output_names,
    )
    assert matching.passed

    multi_dtype = evaluator.evaluate(
        wrong_dtype,
        golden,
        dtype="int32",
        expected_output_dtypes=[["int32", "int64"]],
        output_names=output_names,
    )
    assert multi_dtype.passed


def test_multi_output_contracts_check_each_declared_dtype():
    evaluator = AccuracyEvaluator()
    wrong_dtype_outputs = (
        torch.tensor([1.0], dtype=torch.float32),
        torch.tensor([0, 3], dtype=torch.int32),
    )
    golden_outputs = (
        torch.tensor([1.0], dtype=torch.float32),
        torch.tensor([0, 3], dtype=torch.int64),
    )

    mismatch = evaluator.evaluate(
        wrong_dtype_outputs,
        golden_outputs,
        dtype="int32",
        expected_output_dtypes=[[], ["int64"]],
        output_names=["score", "keep_indices"],
    )
    assert not mismatch.passed
    assert "keep_indices" in mismatch.error_msg

    unconstrained = evaluator.evaluate(
        wrong_dtype_outputs[1],
        golden_outputs[1],
        dtype="int32",
        expected_output_dtypes=[[]],
    )
    assert unconstrained.passed
