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
验证 P0-3: FailureSynthesizer 三个方法统一为一个内部实现

目标：synthesize_compile_failure / synthesize_security_failure / synthesize_subprocess_failure
     三个方法仅 error 前缀不同，应统一为单一内部实现。
"""

import pytest

from kernel_eval.benches import CannTaskSpec
from kernel_eval.eval.results import EvalOperatorResult


# 构造最小 CannTaskSpec fixture
@pytest.fixture
def op_info():
    return CannTaskSpec(
        task_id="level1/exp",
        name="Exp",
        rel_path="level1/exp",
        dir_name="exp",
    )


class TestFailureSynthesizerUnified:
    """三个公有方法应产生一致的输出结构，仅在 error 前缀和诊断字段上有差异"""

    def test_compile_failure_has_correct_error_prefix(self, op_info):
        """compile failure 应带 'compile failed:' 前缀"""
        from kernel_eval.eval.failure_synthesizer import FailureSynthesizer
        from kernel_eval.benches import CannCaseLoader
        from kernel_eval.config import get_project_root

        loader = CannCaseLoader(str(get_project_root() / "tasks"))
        fs = FailureSynthesizer(loader)
        result = fs.synthesize_compile_failure(op_info, "gcc error: undefined reference")

        assert isinstance(result, EvalOperatorResult)
        assert result.compilation_error is not None
        assert result.passed_cases == 0
        assert result.failed_cases >= 0
        # 诊断字段：compile 用 compilation_error
        assert "gcc error" in result.compilation_error
        for case in result.results:
            assert case.success is False
            assert "compile failed:" in case.error_msg

    def test_security_failure_has_correct_error_prefix(self, op_info):
        """security failure 应带 'security check failed:' 前缀"""
        from kernel_eval.eval.failure_synthesizer import FailureSynthesizer
        from kernel_eval.benches import CannCaseLoader
        from kernel_eval.config import get_project_root

        loader = CannCaseLoader(str(get_project_root() / "tasks"))
        fs = FailureSynthesizer(loader)
        result = fs.synthesize_security_failure(op_info, "timing api patched")

        assert isinstance(result, EvalOperatorResult)
        # 诊断字段：security 用 subprocess_failure_reason
        assert result.subprocess_failure_reason is not None
        assert "timing api" in result.subprocess_failure_reason
        for case in result.results:
            assert case.success is False
            assert "security check failed:" in case.error_msg

    def test_subprocess_failure_has_correct_error_prefix(self, op_info):
        """subprocess failure 应带 'subprocess failed:' 前缀"""
        from kernel_eval.eval.failure_synthesizer import FailureSynthesizer
        from kernel_eval.benches import CannCaseLoader
        from kernel_eval.config import get_project_root

        loader = CannCaseLoader(str(get_project_root() / "tasks"))
        fs = FailureSynthesizer(loader)
        result = fs.synthesize_subprocess_failure("Exp", "level1/exp", "timeout after 300s")

        assert isinstance(result, EvalOperatorResult)
        assert result.subprocess_failure_reason is not None
        assert "timeout" in result.subprocess_failure_reason
        for case in result.results:
            assert case.success is False
            assert "subprocess failed:" in case.error_msg

    def test_all_three_produce_same_result_structure(self, op_info):
        """三者应有一致的结构：total = failed = len(results), passed = skipped = 0"""
        from kernel_eval.eval.failure_synthesizer import FailureSynthesizer
        from kernel_eval.benches import CannCaseLoader
        from kernel_eval.config import get_project_root

        loader = CannCaseLoader(str(get_project_root() / "tasks"))
        fs = FailureSynthesizer(loader)

        r1 = fs.synthesize_compile_failure(op_info, "err")
        r2 = fs.synthesize_security_failure(op_info, "err")
        r3 = fs.synthesize_subprocess_failure("Exp", "level1/exp", "err")

        for r in [r1, r2, r3]:
            assert r.passed_cases == 0
            assert r.skipped_cases == 0
            assert r.pass_rate == 0.0
            assert r.avg_speedup == 0.0
            assert r.failed_cases == len(r.results)

    def test_empty_cases_on_nonexistent_operator(self, op_info):
        """不存在的算子应返回 total_cases=0 的合法结果"""
        from kernel_eval.eval.failure_synthesizer import FailureSynthesizer
        from kernel_eval.benches import CannCaseLoader
        from kernel_eval.config import get_project_root

        loader = CannCaseLoader(str(get_project_root() / "tasks"))
        fs = FailureSynthesizer(loader)

        unknown = CannTaskSpec(task_id="nonexistent", name="NoSuchOp", rel_path="x")
        result = fs.synthesize_compile_failure(unknown, "err")

        assert result.total_cases == 0
        assert result.results == []
        assert result.pass_rate == 0.0
