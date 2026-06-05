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
P1-7: 验证 OperatorDirMixin 消除 CannCaseLoader / CannTaskLoader 重复代码

目标：
- loader_base.py 新增 OperatorDirMixin，提供 REQUIRED_FILES / _is_operator_dir / _find_operator_dirs
- CannCaseLoader 和 CannTaskLoader 继承 mixin，移除各自重复实现
- path_resolver.is_operator_directory 使用共享常量
"""

import pytest
from pathlib import Path

from kernel_eval.base.loaders import OperatorDirMixin


class TestOperatorDirMixin:
    """OperatorDirMixin 基类功能测试"""

    def test_required_files_constant(self):
        assert OperatorDirMixin.REQUIRED_FILES == ['proto.yaml', 'cases.yaml', 'golden.py']

    def test_is_operator_dir_all_present(self, tmp_path):
        for f in ['proto.yaml', 'cases.yaml', 'golden.py']:
            (tmp_path / f).touch()
        mixin = OperatorDirMixin()
        assert mixin._is_operator_dir(tmp_path) is True

    def test_is_operator_dir_missing_file(self, tmp_path):
        (tmp_path / 'proto.yaml').touch()
        mixin = OperatorDirMixin()
        assert mixin._is_operator_dir(tmp_path) is False

    def test_is_operator_dir_empty(self, tmp_path):
        mixin = OperatorDirMixin()
        assert mixin._is_operator_dir(tmp_path) is False

    def test_find_operator_dirs(self, tmp_path):
        op_dir1 = tmp_path / "level1" / "Add"
        op_dir2 = tmp_path / "level2" / "Scatter"
        op_dir1.mkdir(parents=True)
        op_dir2.mkdir(parents=True)
        for f in ['proto.yaml', 'cases.yaml', 'golden.py']:
            (op_dir1 / f).touch()
            (op_dir2 / f).touch()
        # 非算子目录（只有 proto.yaml）
        non_op = tmp_path / "level1" / "not_an_op"
        non_op.mkdir(parents=True)
        (non_op / 'proto.yaml').touch()

        mixin = OperatorDirMixin()
        mixin.bench_root = tmp_path
        dirs = mixin._find_operator_dirs()
        assert sorted(dirs) == sorted([op_dir1, op_dir2])

    def test_find_operator_dirs_empty(self, tmp_path):
        mixin = OperatorDirMixin()
        mixin.bench_root = tmp_path
        assert mixin._find_operator_dirs() == []


class TestCannLoadersInheritMixin:
    """CannCaseLoader 和 CannTaskLoader 应继承 OperatorDirMixin"""

    def test_case_loader_inherits_mixin(self):
        from kernel_eval.benches.cann_loader import CannCaseLoader
        assert issubclass(CannCaseLoader, OperatorDirMixin)

    def test_task_loader_inherits_mixin(self):
        from kernel_eval.benches.cann_loader import CannTaskLoader
        assert issubclass(CannTaskLoader, OperatorDirMixin)

    def test_case_loader_no_own_required_files(self):
        from kernel_eval.benches.cann_loader import CannCaseLoader
        assert 'REQUIRED_FILES' not in CannCaseLoader.__dict__

    def test_task_loader_no_own_required_files(self):
        from kernel_eval.benches.cann_loader import CannTaskLoader
        assert 'REQUIRED_FILES' not in CannTaskLoader.__dict__

    def test_case_loader_no_own_is_operator_dir(self):
        from kernel_eval.benches.cann_loader import CannCaseLoader
        assert '_is_operator_dir' not in CannCaseLoader.__dict__

    def test_task_loader_no_own_is_operator_dir(self):
        from kernel_eval.benches.cann_loader import CannTaskLoader
        assert '_is_operator_dir' not in CannTaskLoader.__dict__

    def test_case_loader_no_own_find_operator_dirs(self):
        from kernel_eval.benches.cann_loader import CannCaseLoader
        assert '_find_operator_dirs' not in CannCaseLoader.__dict__

    def test_task_loader_no_own_find_operator_dirs(self):
        from kernel_eval.benches.cann_loader import CannTaskLoader
        assert '_find_operator_dirs' not in CannTaskLoader.__dict__


class TestPathResolverUsesSharedConstant:
    """path_resolver.is_operator_directory 应使用共享 REQUIRED_FILES"""

    def test_uses_shared_files_list(self):
        from kernel_eval.utils.path_resolver import is_operator_directory
        from kernel_eval.base.loaders import OperatorDirMixin
        import inspect
        source = inspect.getsource(is_operator_directory)
        # 不应硬编码文件列表，应引用共享常量
        assert "OperatorDirMixin.REQUIRED_FILES" in source or \
               "REQUIRED_FILES" in source.replace("OperatorDirMixin.REQUIRED_FILES", "REQUIRED_FILES")
