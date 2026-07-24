"""Structural coverage for the submit-ready Triton Ascend CANN example."""

from __future__ import annotations

import importlib
import math
import os
import sys
import types
from pathlib import Path

import pytest
import torch

from kernel_eval.eval.op_runner import OpRunner


REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_DIR = REPO_ROOT / "examples" / "triton_ascend_cann_example"


class _FakeTritonKernel:
    def __init__(self, fn, launches):
        self.fn = fn
        self.launches = launches

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.launches.append((grid, args, kwargs))

        return launch


class _FakeDeviceManager:
    def to_device_batch(self, tensors):
        return tensors

    def get_device(self):
        return "npu:0"

    def synchronize(self):
        return None


def _clear_example_modules(monkeypatch):
    for module_name in list(sys.modules):
        if module_name == "cann_bench" or module_name.startswith("cann_bench."):
            monkeypatch.delitem(sys.modules, module_name, raising=False)


def _import_fake_example(monkeypatch):
    launches = []
    triton_module = types.ModuleType("triton")
    triton_language_module = types.ModuleType("triton.language")
    triton_extra_module = types.ModuleType("triton.language.extra")
    triton_cann_module = types.ModuleType("triton.language.extra.cann")
    triton_libdevice_module = types.ModuleType("triton.language.extra.cann.libdevice")
    triton_language_module.constexpr = object()
    triton_module.language = triton_language_module
    triton_module.jit = lambda fn: _FakeTritonKernel(fn, launches)
    triton_module.cdiv = lambda value, divisor: (value + divisor - 1) // divisor
    triton_module.next_power_of_2 = lambda value: 1 << (value - 1).bit_length()
    triton_cann_module.libdevice = triton_libdevice_module

    monkeypatch.setitem(sys.modules, "triton", triton_module)
    monkeypatch.setitem(sys.modules, "triton.language", triton_language_module)
    monkeypatch.setitem(sys.modules, "triton.language.extra", triton_extra_module)
    monkeypatch.setitem(sys.modules, "triton.language.extra.cann", triton_cann_module)
    monkeypatch.setitem(
        sys.modules,
        "triton.language.extra.cann.libdevice",
        triton_libdevice_module,
    )
    monkeypatch.setitem(sys.modules, "torch_npu", types.ModuleType("torch_npu"))
    monkeypatch.syspath_prepend(str(EXAMPLE_DIR))
    _clear_example_modules(monkeypatch)

    return importlib.import_module("cann_bench"), launches


def test_triton_ascend_example_has_submit_ready_layout():
    assert EXAMPLE_DIR.joinpath("build.sh").is_file()
    assert EXAMPLE_DIR.joinpath("setup.py").is_file()
    assert EXAMPLE_DIR.joinpath("cann_bench", "__init__.py").is_file()
    operator_names = ["exp", "masked_scale", "mish", "sigmoid", "swi_glu"]
    for operator_name in operator_names:
        kernel_path = EXAMPLE_DIR.joinpath("cann_bench", f"{operator_name}.py")
        assert kernel_path.is_file()
        assert "@triton.jit" in kernel_path.read_text(encoding="utf-8")

    build_script = EXAMPLE_DIR.joinpath("build.sh").read_text(encoding="utf-8")
    setup_source = EXAMPLE_DIR.joinpath("setup.py").read_text(encoding="utf-8")

    assert "bdist_wheel" in build_script
    assert 'name="cann_bench"' in setup_source


def test_docker_flavor_runs_a_real_triton_ascend_smoke():
    dockerfile = REPO_ROOT.joinpath("docker", "Dockerfile").read_text(encoding="utf-8")
    smoke = REPO_ROOT.joinpath("docker", "test_env.py").read_text(encoding="utf-8")

    assert "ARG TRITON_ASCEND_VERSION=" in dockerfile
    assert "TRITON_CACHE_DIR=/tmp/cann-bench-triton-cache" in dockerfile
    assert "triton-ascend==${TRITON_ASCEND_VERSION}" in dockerfile
    assert "@triton.jit" in smoke
    assert 'target.backend == "npu"' in smoke
    assert "Triton-Ascend JIT/vector add" in smoke


def test_op_runner_executes_exported_triton_callable(monkeypatch):
    example, launches = _import_fake_example(monkeypatch)
    runner = OpRunner(_FakeDeviceManager())
    x = torch.randn(11)

    result = runner.run_ai_op(
        example.exp,
        {"x": x, "base": 2.0, "scale": 1.5, "shift": -0.5},
        "exp_1",
        [x],
        enable_perf=False,
    )

    assert result.success is True
    assert result.outputs.shape == x.shape
    assert len(launches) == 1

    grid, args, kwargs = launches[0]
    assert grid == (1,)
    assert args[0] is x
    assert args[2] == x.numel()
    assert args[3:6] == (1.5, -0.5, math.log(2.0))
    assert kwargs == {"HAS_BASE": True, "BLOCK_SIZE": 4096}


def test_new_triton_wrappers_launch_with_expected_shapes(monkeypatch):
    example, launches = _import_fake_example(monkeypatch)

    x = torch.randn(2, 6, 4)
    assert example.sigmoid(x).shape == x.shape
    assert example.mish(x).shape == x.shape

    mask = torch.ones_like(x, dtype=torch.int8)
    assert example.masked_scale(x, mask, scale=0.5).shape == x.shape

    swi_glu_output = example.swi_glu(x, dim=1)
    assert swi_glu_output.shape == (2, 3, 4)

    assert len(launches) == 4
    _, sigmoid_args, sigmoid_kwargs = launches[0]
    assert sigmoid_args[2] == x.numel()
    assert sigmoid_kwargs == {"BLOCK_SIZE": 4096}

    _, masked_args, masked_kwargs = launches[2]
    assert masked_args[1] is mask
    assert masked_args[3:5] == (x.numel(), 0.5)
    assert masked_kwargs == {"BLOCK_SIZE": 4096}

    _, swi_glu_args, swi_glu_kwargs = launches[3]
    assert swi_glu_args[2] == 12
    assert swi_glu_kwargs == {"BLOCK_SIZE": 16}
    assert launches[3][0] == (1, 2)


def test_new_triton_wrappers_reject_invalid_shapes(monkeypatch):
    example, _ = _import_fake_example(monkeypatch)

    with pytest.raises(ValueError, match="same shape"):
        example.masked_scale(torch.empty(2), torch.empty(3))
    with pytest.raises(ValueError, match="even size"):
        example.swi_glu(torch.empty(2, 3), dim=1)
    with pytest.raises(IndexError, match="dimension out of range"):
        example.swi_glu(torch.empty(2, 4), dim=2)


@pytest.mark.skipif(
    os.environ.get("CANN_BENCH_RUN_TRITON_NPU") != "1",
    reason="set CANN_BENCH_RUN_TRITON_NPU=1 in a Triton-Ascend NPU environment",
)
def test_triton_ascend_exp_runs_on_npu(monkeypatch):
    monkeypatch.syspath_prepend(str(EXAMPLE_DIR))
    _clear_example_modules(monkeypatch)

    exp = importlib.import_module("cann_bench").exp
    x = torch.tensor([-1.0, 0.0, 1.0, 2.0], dtype=torch.float32, device="npu:0")
    actual = exp(x, base=2.0, scale=1.5, shift=-0.5)
    expected = torch.exp((x * 1.5 - 0.5) * math.log(2.0))
    torch.npu.synchronize()

    assert actual.device.type == "npu"
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.skipif(
    os.environ.get("CANN_BENCH_RUN_TRITON_NPU") != "1",
    reason="set CANN_BENCH_RUN_TRITON_NPU=1 in a Triton-Ascend NPU environment",
)
@pytest.mark.parametrize("operator_name", ["sigmoid", "mish", "masked_scale", "swi_glu"])
def test_new_triton_operators_run_on_npu(monkeypatch, operator_name):
    monkeypatch.syspath_prepend(str(EXAMPLE_DIR))
    _clear_example_modules(monkeypatch)
    example = importlib.import_module("cann_bench")

    if operator_name == "sigmoid":
        x = torch.tensor([-100.0, -1.0, 0.0, 1.0, 100.0], device="npu:0")
        actual = example.sigmoid(x)
        expected = torch.sigmoid(x)
    elif operator_name == "mish":
        x = torch.tensor(
            [float("-inf"), -20.0, -1.0, 0.0, 20.0, float("inf"), float("nan")],
            device="npu:0",
        )
        actual = example.mish(x)
        expected = torch.nn.functional.mish(x)
    elif operator_name == "masked_scale":
        x = torch.tensor([-3.0, -1.0, 0.0, 2.0], device="npu:0")
        mask = torch.tensor([0, 1, 2, 127], dtype=torch.int8, device="npu:0")
        actual = example.masked_scale(x, mask, scale=-0.5)
        expected = (x * mask * -0.5).to(x.dtype)
    else:
        x = torch.arange(24, dtype=torch.float32, device="npu:0").reshape(2, 4, 3)
        actual = example.swi_glu(x, dim=1)
        x0, x1 = x.chunk(2, dim=1)
        expected = torch.nn.functional.silu(x0) * x1

    torch.npu.synchronize()
    assert actual.device.type == "npu"
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6, equal_nan=True)
