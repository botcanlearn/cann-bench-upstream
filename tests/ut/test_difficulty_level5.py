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
L5 难度等级回归测试

覆盖：
1. DifficultyLevel 枚举包含 L5
2. CannTaskLoader 解析 proto.yaml 的 difficulty: L5（不回落为 L1）
3. TaskSpec.get_level_id() 对 L5 返回 5
4. bench_lab/kernel_bench 新增的 L4/L5 算子：proto 难度与目录层级一致，
   且 golden.py 提供签名一致的 <op>_oracle 钩子（dtype-agnostic 真值）
5. tests/st 的 golden-npu-mock 收集器已纳入 level5
"""

import ast
import inspect
import tempfile
import textwrap
from pathlib import Path

import pytest

from kernel_eval.base.enums import DifficultyLevel
from kernel_eval.benches.cann_loader import CannTaskLoader, GoldenLoader
from kernel_eval.config import get_project_root
from scripts.utils.build_golden_wheel import _available_level_choices, scan_golden_operators


_MINIMAL_PROTO = """\
operator:
  name: DummyOp
  category: FusedComposite
  difficulty: {difficulty}
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

_BENCH_LAB_NEW_OPS = {
    "level4": ["selective_scan", "linear_cross_entropy", "bev_pool", "tridiagonal_solve"],
    "level5": ["flash_attention_backward", "mamba2_ssd", "fft_conv", "batched_svd"],
}


def _calls_hardcoded_cast(fn) -> bool:
    """fn 源码中是否存在 `<expr>.float()` / `<expr>.double()` 调用（AST 判定）。"""
    tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                and node.func.attr in ("float", "double") and not node.args:
            return True
    return False


def _write_op(root: Path, level: str, difficulty: str) -> str:
    op_dir = root / level / "dummy_op"
    op_dir.mkdir(parents=True)
    (op_dir / "proto.yaml").write_text(_MINIMAL_PROTO.format(difficulty=difficulty), encoding="utf-8")
    (op_dir / "cases.yaml").write_text("cases: []\n", encoding="utf-8")
    (op_dir / "golden.py").write_text("def dummy_op(x):\n    return x\n", encoding="utf-8")
    return f"{level}/dummy_op"


class TestDifficultyLevelEnum:

    def test_l5_member_exists(self):
        assert DifficultyLevel.L5.value == "L5"
        assert DifficultyLevel("L5") is DifficultyLevel.L5

    def test_levels_are_ordered_l1_to_l5(self):
        assert [m.value for m in DifficultyLevel] == ["L1", "L2", "L3", "L4", "L5"]


class TestLoaderParsesL5:

    @pytest.mark.parametrize("difficulty,expected,level_id", [
        ("L4", DifficultyLevel.L4, 4),
        ("L5", DifficultyLevel.L5, 5),
        ("l5", DifficultyLevel.L5, 5),
    ])
    def test_difficulty_and_level_id(self, difficulty, expected, level_id):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            rel = _write_op(root, "level5" if difficulty.lower() == "l5" else "level4", difficulty)
            spec = CannTaskLoader(str(root)).get_task(rel)
            assert spec is not None
            assert spec.difficulty is expected
            assert spec.get_level_id() == level_id

    def test_unknown_difficulty_still_falls_back_to_l1(self):
        """未知字符串保持既有回落语义，避免误把 L5 支持写成宽松解析。"""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            rel = _write_op(root, "level1", "L9")
            spec = CannTaskLoader(str(root)).get_task(rel)
            assert spec.difficulty is DifficultyLevel.L1
            assert spec.get_level_id() == 1


class TestBenchLabNewOps:
    """bench_lab/kernel_bench 新增 8 个算子的难度声明与 oracle 钩子。"""

    _root = get_project_root() / "bench_lab" / "kernel_bench"

    @pytest.mark.parametrize("level,op", [(l, o) for l, ops in _BENCH_LAB_NEW_OPS.items() for o in ops])
    def test_difficulty_matches_level_dir(self, level, op):
        if not (self._root / level / op / "proto.yaml").is_file():
            pytest.skip(f"{level}/{op} 不在本 checkout 中")
        spec = CannTaskLoader(str(self._root)).get_task(f"{level}/{op}")
        assert spec.difficulty.value == level.replace("level", "L")
        assert spec.get_level_id() == int(level[-1])

    @pytest.mark.parametrize("level,op", [(l, o) for l, ops in _BENCH_LAB_NEW_OPS.items() for o in ops])
    def test_oracle_hook_present_with_matching_signature(self, level, op):
        if not (self._root / level / op / "golden.py").is_file():
            pytest.skip(f"{level}/{op} 不在本 checkout 中")
        loader = GoldenLoader(bench_root=str(self._root))
        rel = f"{level}/{op}"
        golden = loader.get_golden_function(rel)
        oracle = loader.get_oracle_function(rel, required=True)
        assert oracle is not None
        assert oracle.__name__ == f"{op}_oracle"
        assert inspect.signature(oracle) == inspect.signature(golden)
        # oracle 及其共享核心不得硬编码 .float()/.double() 调用（dtype-agnostic 约定；按 AST 判定，忽略注释/docstring）
        module = inspect.getmodule(oracle)
        for fn_name in (oracle.__name__, f"_{op}_core"):
            fn = getattr(module, fn_name, None)
            if fn is None:
                continue
            assert not _calls_hardcoded_cast(fn), f"{fn_name} 中存在硬编码 .float()/.double()"


class TestStCollectorIncludesLevel5:

    def test_golden_npu_mock_levels_include_level5(self):
        st_file = get_project_root() / "tests" / "st" / "test_golden_npu_mock.py"
        text = st_file.read_text(encoding="utf-8")
        assert '"level5"' in text.split("def _all_ops", 1)[0], \
            "tests/st/test_golden_npu_mock.py 的 _LEVELS 应包含 level5，否则 pytest -m level5 为空集合"


class TestGoldenWheelLevelFilter:

    def test_available_levels_are_derived_from_task_root(self, tmp_path):
        for level in (4, 5, 10):
            (tmp_path / f"level{level}").mkdir()
        (tmp_path / "level_future").mkdir()
        (tmp_path / "level6").write_text("not a directory", encoding="utf-8")

        assert _available_level_choices(tmp_path) == (4, 5, 10)

    @pytest.mark.parametrize("selected_level,expected_level", [(4, "level4"), (5, "level5")])
    def test_level_filter_selects_only_requested_level(self, selected_level, expected_level):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_op(root, "level4", "L4")
            _write_op(root, "level5", "L5")

            operators = scan_golden_operators(root, level_filter=selected_level)

            assert [op["rel_path"] for op in operators] == [f"{expected_level}/dummy_op"]

    def test_level_filter_does_not_match_a_prefix_of_another_level(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_op(root, "level1", "L1")
            _write_op(root, "level10", "L1")

            operators = scan_golden_operators(root, level_filter=1)

            assert [op["rel_path"] for op in operators] == ["level1/dummy_op"]


class TestKernelEvalLevelFilter:

    def test_cpu_eval_does_not_fail_open_when_level_has_no_operator(self, monkeypatch, tmp_path):
        pytest.importorskip("torch")
        import kernel_eval.cli as cli

        args = cli.create_parser().parse_args([
            "eval", "--device", "cpu", "--task-dir", str(tmp_path), "--level", "5",
        ])
        monkeypatch.setattr(cli, "get_project_root", lambda: tmp_path)
        monkeypatch.setattr(cli, "resolve_task_dir", lambda task_dir, project_root: (str(tmp_path), ""))
        monkeypatch.setattr(cli, "_operator_names_for_level", lambda *unused: [])
        class DummyConfig:
            reports_dir = tmp_path

        monkeypatch.setattr(cli, "_create_config_from_args", lambda *unused: DummyConfig())

        class DummyReportGenerator:
            def __init__(self, **unused):
                pass

        monkeypatch.setattr(cli, "ReportGenerator", DummyReportGenerator)
        called = False

        def fail_if_called(*unused, **unused_kwargs):
            nonlocal called
            called = True

        monkeypatch.setattr("kernel_eval.simulation.simulate", fail_if_called)

        assert cli.cmd_eval(args) == 0
        assert called is False

    def test_cli_accepts_level_five_for_all_selection_commands(self):
        pytest.importorskip("torch")
        from kernel_eval.cli import create_parser

        parser = create_parser()
        for command in ("eval", "list", "info"):
            argv = [command, "--level", "5"]
            if command == "info":
                argv.extend(["--operator", "Mamba2Ssd"])
            assert parser.parse_args(argv).level == 5

    def test_list_command_filters_operators_by_level(self, tmp_path, capsys):
        pytest.importorskip("torch")
        from kernel_eval.cli import cmd_list, create_parser

        _write_op(tmp_path, "level4", "L4")
        _write_op(tmp_path, "level5", "L5")
        parser = create_parser()
        args = parser.parse_args(["list", "--task-dir", str(tmp_path), "--level", "5"])

        assert cmd_list(args) == 0
        output = capsys.readouterr().out
        assert "共 1 个算子" in output
        assert "L5" in output
        assert "L4" not in output

    @pytest.mark.parametrize(
        "rel_path,selected,expected",
        [
            ("level5/mamba2_ssd", 5, True),
            ("level4/exp", 5, False),
            ("level50/future", 5, False),
        ],
    )
    def test_level_filter_matches_a_complete_level_component(self, rel_path, selected, expected):
        pytest.importorskip("torch")
        from kernel_eval.cli import _matches_level

        assert _matches_level(rel_path, selected) is expected
