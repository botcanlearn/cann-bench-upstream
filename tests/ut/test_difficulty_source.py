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

"""难度级别的单一真源

`DifficultyLevel` 是级别的唯一声明处. 历史上另有两处与它平行的字面量梯子:
proto.yaml 字符串 -> 枚举 (cann_loader), 枚举 -> 序号 (TaskSpec.get_level_id),
两者都带静默兜底, 于是新增一个级别会让算子无声无息地落到 L1.

本文件钉住三件事:
1. 未知 / 缺失的 difficulty 必须报错, 不再回落 L1;
2. 这类错误必须穿透 `list_tasks` 的容错层, 否则算子只是从列表里消失;
3. 目录 `levelN` 与 proto 声明的 `LN` 必须一致.
"""

from pathlib import Path
from typing import List

import pytest

from kernel_eval.base.enums import DifficultyLevel
from kernel_eval.benches.cann_loader import CannTaskLoader
from kernel_eval.config import get_project_root


_PROTO = """\
operator:
  name: DummyOp
{difficulty_line}\
  inputs:
  - name: x
    dtype: [float32]
    shape: "[N]"
  outputs:
  - name: y
    dtype: [float32]
    shape: "[N]"
  schema: dummy_op(Tensor x) -> Tensor y
"""


def _write_op(root: Path, level: str, difficulty) -> str:
    """写一个最小算子目录; difficulty=None 表示整个字段缺失."""
    line = "" if difficulty is None else f"  difficulty: {difficulty}\n"
    op_dir = root / level / "dummy_op"
    op_dir.mkdir(parents=True)
    (op_dir / "proto.yaml").write_text(_PROTO.format(difficulty_line=line), encoding="utf-8")
    (op_dir / "cases.yaml").write_text("cases: []\n", encoding="utf-8")
    (op_dir / "golden.py").write_text("def dummy_op(x):\n    return x\n", encoding="utf-8")
    return f"{level}/dummy_op"


def _level_dir_mismatches(root: Path) -> List[str]:
    """目录 levelN 与 proto 声明的难度不符的算子."""
    mismatches = []
    for spec in CannTaskLoader(str(root)).list_tasks():
        dir_level = spec.rel_path.split("/", 1)[0]
        if dir_level != f"level{spec.get_level_id()}":
            mismatches.append(f"{spec.rel_path}: 目录 {dir_level} != 声明 {spec.difficulty.value}")
    return mismatches


class TestDifficultyIsRejectedWhenNotDeclared:

    @pytest.mark.parametrize("difficulty,expected", [
        ("L1", DifficultyLevel.L1),
        ("L5", DifficultyLevel.L5),
        ("l5", DifficultyLevel.L5),
    ])
    def test_declared_values_still_parse(self, tmp_path, difficulty, expected):
        rel = _write_op(tmp_path, "level1", difficulty)
        assert CannTaskLoader(str(tmp_path)).get_task(rel).difficulty is expected

    def test_unknown_difficulty_raises_instead_of_falling_back(self, tmp_path):
        rel = _write_op(tmp_path, "level1", "L9")
        with pytest.raises(ValueError, match="L9"):
            CannTaskLoader(str(tmp_path)).get_task(rel)

    def test_missing_difficulty_raises_instead_of_defaulting(self, tmp_path):
        rel = _write_op(tmp_path, "level1", None)
        with pytest.raises(ValueError, match="difficulty"):
            CannTaskLoader(str(tmp_path)).get_task(rel)


class TestInvalidDifficultySurvivesTheLoaderTolerance:
    """`list_tasks` 对坏 proto 是 warn-and-skip 的; 难度错误必须是例外.

    否则修复只是把 "被当成 L1" 换成 "从列表里消失", 一样静默.
    """

    def test_list_tasks_propagates_invalid_difficulty(self, tmp_path):
        _write_op(tmp_path, "level1", "L9")
        with pytest.raises(ValueError, match="L9"):
            CannTaskLoader(str(tmp_path)).list_tasks()

    def test_get_operator_by_name_propagates_invalid_difficulty(self, tmp_path):
        _write_op(tmp_path, "level1", "L9")
        with pytest.raises(ValueError, match="L9"):
            CannTaskLoader(str(tmp_path)).get_operator_by_name("DummyOp")

    def test_list_tasks_still_tolerates_an_unparsable_proto(self, tmp_path):
        op_dir = tmp_path / "level1" / "broken_op"
        op_dir.mkdir(parents=True)
        (op_dir / "proto.yaml").write_text("operator: [unclosed\n", encoding="utf-8")
        (op_dir / "golden.py").write_text("", encoding="utf-8")

        assert CannTaskLoader(str(tmp_path)).list_tasks() == []


class TestLevelIdCoversEveryDeclaredLevel:
    """加一个枚举成员不应该还要记得去改第二张表."""

    def test_every_declared_level_maps_to_its_own_id(self, tmp_path):
        ids = {}
        for level in DifficultyLevel:
            root = tmp_path / level.value
            rel = _write_op(root, "level1", level.value)
            ids[level] = CannTaskLoader(str(root)).get_task(rel).get_level_id()

        assert sorted(ids.values()) == list(range(1, len(DifficultyLevel) + 1))
        assert all(ids[level] == int(level.value[1:]) for level in DifficultyLevel)


class TestOperatorDirMatchesDeclaredDifficulty:

    def test_checker_flags_a_misfiled_operator(self, tmp_path):
        _write_op(tmp_path, "level4", "L5")
        assert _level_dir_mismatches(tmp_path) == [
            "level4/dummy_op: 目录 level4 != 声明 L5"
        ]

    @pytest.mark.parametrize("root_name", ["tasks", "bench_lab/kernel_bench"])
    def test_repository_trees_are_consistent(self, root_name):
        root = get_project_root() / root_name
        if not root.is_dir():
            pytest.skip(f"{root_name} 不在本 checkout 中")
        assert _level_dir_mismatches(root) == []
