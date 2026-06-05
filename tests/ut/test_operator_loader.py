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
OperatorLoader/CannTaskLoader 单元测试

测试对象：kernel_eval.data.CannTaskLoader (OperatorLoader 别名)
核心功能：
1. get_operator(rel_path) / get_task(task_id) 解析 proto.yaml
2. get_operator_by_name(operator) 按名称查找
3. list_operators() / list_tasks() 列出所有算子
4. proto.yaml 解析容错
5. 异常路径的 warning 日志
"""

import logging
import tempfile
import pytest
from pathlib import Path

from kernel_eval.benches import CannTaskLoader


class TestOperatorLoaderGetOperator:
    """测试 get_operator 方法的异常容错"""

    def test_corrupt_proto_yaml_raises_on_get_operator(self, caplog):
        """损坏的 proto.yaml 在 get_operator 时应抛出异常并记录 warning"""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            level_dir = root / "level1" / "badop"
            level_dir.mkdir(parents=True)
            # 需要 golden.py 和 cases.yaml 才能被 _is_operator_dir 识别
            (level_dir / "golden.py").write_text("def badop(*args):\n    return args\n", encoding="utf-8")
            (level_dir / "cases.yaml").write_text("cases: []\n", encoding="utf-8")
            (level_dir / "proto.yaml").write_text(":\n\t- broken: yaml: [[[\n", encoding="utf-8")

            loader = CannTaskLoader(bench_root=str(root))
            # get_operator 对损坏的 YAML 会抛出 yaml.YAMLError
            with pytest.raises(Exception):
                loader.get_operator("level1/badop")

    def test_unicode_decode_error_in_proto_yaml(self, caplog):
        """非 UTF-8 编码的 proto.yaml 应抛出异常"""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            level_dir = root / "level1" / "binop"
            level_dir.mkdir(parents=True)
            (level_dir / "golden.py").write_text("def binop(*args):\n    return args\n", encoding="utf-8")
            (level_dir / "cases.yaml").write_text("cases: []\n", encoding="utf-8")
            with open(level_dir / "proto.yaml", "wb") as f:
                f.write(b'\xff\xfe\x00\x00\xff\xff')

            loader = CannTaskLoader(bench_root=str(root))
            with pytest.raises(UnicodeDecodeError):
                loader.get_operator("level1/binop")

    def test_missing_directory_raises_file_not_found(self):
        """不存在的算子目录应抛出 FileNotFoundError"""
        with tempfile.TemporaryDirectory() as td:
            loader = CannTaskLoader(bench_root=str(td))
            with pytest.raises(FileNotFoundError):
                loader.get_operator("level1/NonExistentOp")

    def test_valid_proto_yaml_no_warning(self, caplog):
        """正常的 proto.yaml 不应产生 warning，且正确解析"""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            level_dir = root / "level1" / "add"
            level_dir.mkdir(parents=True)
            (level_dir / "golden.py").write_text("def add(*args):\n    return args\n", encoding="utf-8")
            (level_dir / "cases.yaml").write_text("cases: []\n", encoding="utf-8")
            (level_dir / "proto.yaml").write_text(
                "operator:\n  name: Add\n  schema: add(Tensor a, Tensor b) -> Tensor\n",
                encoding="utf-8"
            )

            with caplog.at_level(logging.WARNING, logger="kernel_eval.data.operator_loader"):
                loader = CannTaskLoader(bench_root=str(root))
                op_info = loader.get_operator("level1/add")

            assert op_info.name == "Add"
            assert op_info.rel_path == "level1/add"
            assert op_info.dir_name == "add"
            assert len(caplog.records) == 0


class TestOperatorLoaderListOperators:
    """测试 list_operators 方法的异常容错"""

    def test_corrupt_proto_skipped_with_warning(self, caplog):
        """list_operators 中损坏的 proto.yaml 应被跳过并记录 warning"""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            level_dir = root / "level1" / "badop"
            level_dir.mkdir(parents=True)
            (level_dir / "golden.py").write_text("def badop(*args):\n    return args\n", encoding="utf-8")
            (level_dir / "cases.yaml").write_text("cases: []\n", encoding="utf-8")
            (level_dir / "proto.yaml").write_text("\tbad: : [[[\n", encoding="utf-8")

            with caplog.at_level(logging.WARNING, logger="kernel_eval.data.operator_loader"):
                loader = CannTaskLoader(bench_root=str(root))
                operators = loader.list_operators()

            # 损坏的算子被跳过
            assert len(operators) == 0
            assert len(caplog.records) >= 1

    def test_mixed_valid_and_invalid_operators(self, caplog):
        """混合有效和无效算子时，有效的正常加载，无效的跳过并记录 warning"""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            # 有效的算子
            valid_dir = root / "level1" / "add"
            valid_dir.mkdir(parents=True)
            (valid_dir / "golden.py").write_text("def add(*args):\n    return args\n", encoding="utf-8")
            (valid_dir / "cases.yaml").write_text("cases: []\n", encoding="utf-8")
            (valid_dir / "proto.yaml").write_text(
                "operator:\n  name: Add\n  schema: add(Tensor a, Tensor b) -> Tensor\n",
                encoding="utf-8"
            )
            # 损坏的算子
            corrupt_dir = root / "level1" / "badop"
            corrupt_dir.mkdir(parents=True)
            (corrupt_dir / "golden.py").write_text("def badop(*args):\n    return args\n", encoding="utf-8")
            (corrupt_dir / "cases.yaml").write_text("cases: []\n", encoding="utf-8")
            (corrupt_dir / "proto.yaml").write_text("]\n:\nbad\n", encoding="utf-8")

            with caplog.at_level(logging.WARNING, logger="kernel_eval.data.operator_loader"):
                loader = CannTaskLoader(bench_root=str(root))
                operators = loader.list_operators()

            # 有效算子被加载，损坏的被跳过
            assert len(operators) == 1
            assert operators[0].name == "Add"
            assert len(caplog.records) >= 1


class TestOperatorLoaderGetByOperatorName:
    """测试 get_operator_by_name 方法"""

    def test_find_by_operator_name(self):
        """按算子名称查找"""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            level_dir = root / "level2" / "my_op"
            level_dir.mkdir(parents=True)
            (level_dir / "golden.py").write_text("def my_op(*args):\n    return args\n", encoding="utf-8")
            (level_dir / "cases.yaml").write_text("cases: []\n", encoding="utf-8")
            (level_dir / "proto.yaml").write_text(
                "operator:\n  name: MyOp\n  schema: my_op(Tensor a) -> Tensor\n",
                encoding="utf-8"
            )

            loader = CannTaskLoader(bench_root=str(root))
            op_info = loader.get_operator_by_name("MyOp")
            assert op_info is not None
            assert op_info.name == "MyOp"
            assert op_info.rel_path == "level2/my_op"

    def test_not_found_returns_none(self):
        """不存在的算子应返回 None"""
        with tempfile.TemporaryDirectory() as td:
            loader = CannTaskLoader(bench_root=str(td))
            assert loader.get_operator_by_name("NonExistent") is None


class TestOperatorLoaderGetStatistics:
    """测试 get_statistics 方法"""

    def test_statistics_with_operators(self):
        """有算子时返回正确统计"""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for op_name, dir_name in [("Add", "add"), ("Mul", "mul")]:
                op_dir = root / "level1" / dir_name
                op_dir.mkdir(parents=True)
                (op_dir / "golden.py").write_text(f"def {dir_name}(*args):\n    return args\n", encoding="utf-8")
                (op_dir / "cases.yaml").write_text("cases: []\n", encoding="utf-8")
                (op_dir / "proto.yaml").write_text(
                    f"operator:\n  name: {op_name}\n  schema: {dir_name}(Tensor a, Tensor b) -> Tensor\n",
                    encoding="utf-8"
                )

            loader = CannTaskLoader(bench_root=str(root))
            stats = loader.get_statistics()
            assert stats['total'] == 2
            assert len(stats['operators']) == 2
            assert len(stats['rel_paths']) == 2
