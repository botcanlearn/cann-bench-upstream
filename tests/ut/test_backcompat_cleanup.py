#!/usr/bin/python3
# coding=utf-8

# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You can not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""
验证架构分离：base/ 纯基类，benches/cann/ 纯特化

清理目标：
- OperatorLoader → CannTaskLoader（移除别名）
- CaseLoader 别名 → 移除 shadowing，保留 ABC CaseLoader
- CaseLoaderCompat → 删除（无调用方）
- LoaderRegistry.list_benches → 删除（无调用方）
- data/ 不再导出 CANN 特化类（应从 benches.cann 导入）
"""

import pytest


class TestOperatorLoaderAliasRemoved:
    """OperatorLoader 别名应被移除"""

    def test_no_operator_loader_in_cann_loader_module(self):
        """cann_loader 模块不应有 OperatorLoader"""
        import kernel_eval.benches.cann_loader as mod
        assert not hasattr(mod, 'OperatorLoader')

    def test_no_operator_loader_in_data_init(self):
        """data/__init__.py 不应导出 OperatorLoader"""
        import kernel_eval.data as mod
        assert not hasattr(mod, 'OperatorLoader')

    def test_cann_task_loader_directly_importable(self):
        """CannTaskLoader 应可直接从 benches.cann 导入"""
        from kernel_eval.benches import CannTaskLoader
        assert CannTaskLoader is not None

    def test_operator_loader_not_in_all(self):
        """OperatorLoader 不应在 __all__ 中"""
        import kernel_eval.data as mod
        all_list = getattr(mod, '__all__', [])
        assert 'OperatorLoader' not in all_list


class TestCaseLoaderCompatRemoved:
    """CaseLoaderCompat 死代码应被移除"""

    def test_no_case_loader_compat_in_module(self):
        """cann_loader 模块不应有 CaseLoaderCompat"""
        import kernel_eval.benches.cann_loader as mod
        assert not hasattr(mod, 'CaseLoaderCompat')


class TestCaseLoaderNoLongerShadowed:
    """data/__init__.py 中 CaseLoader 应指向 ABC 而非 CannCaseLoader"""

    def test_case_loader_is_abc_not_cann(self):
        """data.__init__.CaseLoader 应为 loader_base.CaseLoader ABC"""
        from kernel_eval.data import CaseLoader
        from kernel_eval.base.loaders import CaseLoader as ABCCaseLoader
        assert CaseLoader is ABCCaseLoader

    def test_cann_case_loader_importable_directly(self):
        """CannCaseLoader 应可直接从 benches.cann 导入"""
        from kernel_eval.benches import CannCaseLoader
        assert CannCaseLoader is not None

    def test_cann_case_loader_not_in_data_all(self):
        """data/__init__.py 不应导出 CannCaseLoader"""
        import kernel_eval.data as mod
        all_list = getattr(mod, '__all__', [])
        assert 'CannCaseLoader' not in all_list

    def test_case_loader_not_cann_case_loader(self):
        """CaseLoader 不应是 CannCaseLoader"""
        from kernel_eval.data import CaseLoader
        from kernel_eval.benches import CannCaseLoader
        assert CaseLoader is not CannCaseLoader


class TestLoaderRegistryListBenchesRemoved:
    """LoaderRegistry.list_benches() 死代码应被移除"""

    def test_list_benches_not_in_loader_registry(self):
        """LoaderRegistry 不应有 list_benches 方法"""
        from kernel_eval.registry.loader_registry import LoaderRegistry
        assert not hasattr(LoaderRegistry, 'list_benches')


class TestPerCaseSolScoreAliasRemoved:
    """_per_case_sol_score 无意义别名应被移除"""

    def test_alias_not_in_module(self):
        """cann_scoring 模块不应有 _per_case_sol_score"""
        import kernel_eval.benches.cann_scoring as mod
        assert not hasattr(mod, '_per_case_sol_score')


class TestEvaluateWithRetryRemoved:
    """evaluate_with_retry 死代码应被移除"""

    def test_method_not_in_accuracy_evaluator(self):
        """AccuracyEvaluator 不应有 evaluate_with_retry"""
        from kernel_eval.eval.accuracy_eval import AccuracyEvaluator
        assert not hasattr(AccuracyEvaluator, 'evaluate_with_retry')