#!/usr/bin/python3
# coding=utf-8

# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""get_output 后处理钩子单元测试

验证：
1. GoldenLoader.get_output_function 正确加载/返回 None
2. StanfordGoldenLoader.get_output_function 恒返回 None
3. Evaluator._apply_get_output 正确变换输出
4. 无 get_output 的算子不受影响
"""

import tempfile
from pathlib import Path

import pytest
import torch

from kernel_eval.benches.cann_loader import GoldenLoader
from kernel_eval.benches.stanford_loader import StanfordGoldenLoader
from kernel_eval.eval.evaluator import Evaluator


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

class _Out:
    def __init__(self, name):
        self.name = name


class _Op:
    def __init__(self, outputs):
        self.outputs = outputs


def _make_op_dir(root, name, golden_body, schema=None):
    """在 root 下创建算子目录（proto.yaml + golden.py + cases.yaml）"""
    d = root / "level1" / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "golden.py").write_text(golden_body, encoding="utf-8")
    if schema is None:
        schema = f"{name}(Tensor a) -> Tensor"
    (d / "proto.yaml").write_text(
        f"operator:\n  name: {name.capitalize()}\n  schema: {schema}\n",
        encoding="utf-8",
    )
    (d / "cases.yaml").write_text("cases: []\n", encoding="utf-8")
    return d


# ---------------------------------------------------------------------------
# GoldenLoader.get_output_function
# ---------------------------------------------------------------------------

class TestGetOutputFunctionLoader:
    """测试 GoldenLoader.get_output_function"""

    def test_returns_none_when_no_get_output(self):
        """golden.py 未定义 get_output 时返回 None"""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _make_op_dir(root, "add", "def add(a):\n    return a\n")
            loader = GoldenLoader(bench_root=str(root))
            assert loader.get_output_function("level1/add") is None

    def test_loads_get_output_when_defined(self):
        """golden.py 定义了 get_output 时返回可调用对象"""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            golden_body = (
                "def add(a):\n"
                "    return a\n"
                "\n"
                "def get_output(y, **kwargs):\n"
                "    return [y]\n"
            )
            _make_op_dir(root, "add", golden_body)
            loader = GoldenLoader(bench_root=str(root))
            func = loader.get_output_function("level1/add")
            assert callable(func)

    def test_get_input_still_works_alongside_get_output(self):
        """get_input 和 get_output 可共存"""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            golden_body = (
                "def add(a):\n"
                "    return a\n"
                "\n"
                "def get_input(a, **kwargs):\n"
                "    return [a]\n"
                "\n"
                "def get_output(y, **kwargs):\n"
                "    return [y]\n"
            )
            _make_op_dir(root, "add", golden_body)
            loader = GoldenLoader(bench_root=str(root))
            assert loader.get_input_function("level1/add") is not None
            assert loader.get_output_function("level1/add") is not None


# ---------------------------------------------------------------------------
# StanfordGoldenLoader.get_output_function
# ---------------------------------------------------------------------------

class TestStanfordGetOutput:
    """StanfordBench 不支持 get_output，恒返回 None"""

    def test_always_returns_none(self):
        loader = StanfordGoldenLoader(bench_root="/tmp/nonexistent_stanford")
        assert loader.get_output_function("any_task") is None


# ---------------------------------------------------------------------------
# Evaluator._apply_get_output
# ---------------------------------------------------------------------------

class TestApplyGetOutput:
    """测试 _apply_get_output 静态方法"""

    def test_single_output_transform(self):
        """单输出：截断到 [0, 1]"""
        def get_output(y, **kwargs):
            return [torch.clamp(y, 0.0, 1.0)]

        op = _Op([_Out("y")])
        raw = torch.tensor([-0.5, 0.5, 1.5])
        result = Evaluator._apply_get_output(get_output, [raw], op, {})
        assert isinstance(result, list)
        assert len(result) == 1
        assert torch.equal(result[0], torch.tensor([0.0, 0.5, 1.0]))

    def test_multi_output_transform(self):
        """多输出：只变换第一个，第二个原样返回"""
        def get_output(y, idx, **kwargs):
            return [torch.clamp(y, 0.0, 1.0), idx]

        op = _Op([_Out("y"), _Out("idx")])
        y = torch.tensor([-1.0, 2.0])
        idx = torch.tensor([0, 1])
        result = Evaluator._apply_get_output(get_output, [y, idx], op, {})
        assert len(result) == 2
        assert torch.equal(result[0], torch.tensor([0.0, 1.0]))
        assert torch.equal(result[1], idx)

    def test_attrs_passed_as_kwargs(self):
        """attrs 作为 kwargs 传入"""
        def get_output(y, dim=None, **kwargs):
            assert dim == 1
            return [y]

        op = _Op([_Out("y")])
        Evaluator._apply_get_output(get_output, [torch.zeros(3)], op, {"dim": 1})

    def test_returns_tuple_normalized_to_list(self):
        """get_output 返回 tuple 时规整为 list"""
        def get_output(y, **kwargs):
            return (y,)

        op = _Op([_Out("y")])
        result = Evaluator._apply_get_output(get_output, [torch.zeros(3)], op, {})
        assert isinstance(result, list)

    def test_returns_single_tensor_wrapped(self):
        """get_output 返回单个 tensor 时包装为 list"""
        def get_output(y, **kwargs):
            return y

        op = _Op([_Out("y")])
        result = Evaluator._apply_get_output(get_output, [torch.zeros(3)], op, {})
        assert isinstance(result, list)
        assert len(result) == 1

    def test_none_outputs_returns_none(self):
        """outputs 为 None 时返回 None"""
        def get_output(**kwargs):
            pytest.fail("should not be called")

        op = _Op([_Out("y")])
        assert Evaluator._apply_get_output(get_output, None, op, {}) is None

    def test_no_op_info_works(self):
        """op_info 为 None 时不崩溃（attrs 仍传入）"""
        def get_output(**kwargs):
            return [kwargs["dim"]]

        result = Evaluator._apply_get_output(
            get_output, [torch.zeros(3)], None, {"dim": 5})
        assert result == [5]

    def test_output_names_mapped_correctly(self):
        """输出按 proto.yaml 声明的名称映射"""
        received = {}

        def get_output(values, indices, **kwargs):
            received["values"] = values is not None
            received["indices"] = indices is not None
            return [values, indices]

        op = _Op([_Out("values"), _Out("indices")])
        v = torch.tensor([1.0, 2.0])
        i = torch.tensor([0, 1])
        Evaluator._apply_get_output(get_output, [v, i], op, {})
        assert received["values"] is True
        assert received["indices"] is True

    def test_fewer_outputs_than_declared(self):
        """实际输出数少于声明数时，缺失位置传 None"""
        def get_output(y, idx, **kwargs):
            assert idx is None
            return [y]

        op = _Op([_Out("y"), _Out("idx")])
        Evaluator._apply_get_output(
            get_output, [torch.zeros(3)], op, {})

    def test_attrs_not_overriding_output_names(self):
        """与输出同名的 attr 不覆盖输出张量"""
        def get_output(y, **kwargs):
            return [y]

        op = _Op([_Out("y")])
        y_tensor = torch.tensor([1.0, 2.0])
        result = Evaluator._apply_get_output(
            get_output, [y_tensor], op, {"y": 999})
        assert torch.equal(result[0], y_tensor)


# ---------------------------------------------------------------------------
# 集成测试：GoldenLoader + _apply_get_output 端到端
# ---------------------------------------------------------------------------

class TestGetOutputIntegration:
    """端到端：从 golden.py 加载 get_output → 变换 → 对比"""

    def test_clamp_end_to_end(self):
        """截断场景：golden.py 定义 get_output 对输出做 clamp，
        框架加载后对 golden/AI 两路输出统一变换，再送入 compare_tensors 对比"""
        from kernel_eval.utils.compare import compare_tensors

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            golden_body = (
                "import torch\n"
                "\n"
                "def clamp_op(x):\n"
                "    return torch.clamp(x, -1.0, 1.0)\n"
                "\n"
                "def get_output(y, **kwargs):\n"
                "    return [torch.clamp(y, -0.5, 0.5)]\n"
            )
            _make_op_dir(root, "clamp_op", golden_body,
                         schema="clamp_op(Tensor x) -> Tensor")

            loader = GoldenLoader(bench_root=str(root))
            golden_func = loader.get_golden_function("level1/clamp_op")
            get_output_func = loader.get_output_function("level1/clamp_op")
            assert get_output_func is not None

            x = torch.randn(100)
            golden_out = golden_func(x)
            ai_out = golden_func(x + 1e-6)

            op_info = _Op([_Out("y")])
            golden_transformed = Evaluator._apply_get_output(
                get_output_func, [golden_out], op_info, {})
            ai_transformed = Evaluator._apply_get_output(
                get_output_func, [ai_out], op_info, {})

            assert golden_transformed[0].max().item() <= 0.5
            assert golden_transformed[0].min().item() >= -0.5
            assert ai_transformed[0].max().item() <= 0.5
            assert ai_transformed[0].min().item() >= -0.5

            result = compare_tensors(
                ai_transformed, golden_transformed, dtype='float32')
            assert result.passed

    def test_no_get_output_no_interference(self):
        """无 get_output 的算子：流程与无钩子时完全一致"""
        from kernel_eval.utils.compare import compare_tensors

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _make_op_dir(root, "add", "def add(a):\n    return a\n")

            loader = GoldenLoader(bench_root=str(root))
            assert loader.get_output_function("level1/add") is None

            golden_out = torch.randn(50)
            ai_out = golden_out + 1e-8

            result = compare_tensors([ai_out], [golden_out], dtype='float32')
            assert result.passed

    def test_sort_non_deterministic_output(self):
        """非确定性排序场景：get_output 排序后对比，消除顺序差异"""
        from kernel_eval.utils.compare import compare_tensors

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            golden_body = (
                "import torch\n"
                "\n"
                "def unique_op(x):\n"
                "    return torch.unique(x)\n"
                "\n"
                "def get_output(y, **kwargs):\n"
                "    y_sorted, _ = torch.sort(y)\n"
                "    return [y_sorted]\n"
            )
            _make_op_dir(root, "unique_op", golden_body,
                         schema="unique_op(Tensor x) -> Tensor")

            loader = GoldenLoader(bench_root=str(root))
            golden_func = loader.get_golden_function("level1/unique_op")
            get_output_func = loader.get_output_function("level1/unique_op")

            x = torch.tensor([3, 1, 4, 1, 5, 9, 2, 6])
            golden_out = golden_func(x)
            ai_out = golden_out.flip(0)
            assert not torch.equal(golden_out, ai_out)

            op_info = _Op([_Out("y")])
            golden_t = Evaluator._apply_get_output(
                get_output_func, [golden_out], op_info, {})
            ai_t = Evaluator._apply_get_output(
                get_output_func, [ai_out], op_info, {})

            assert torch.equal(golden_t[0], ai_t[0])

            result = compare_tensors(ai_t, golden_t, dtype='int64')
            assert result.passed
