#!/usr/bin/python3
# coding=utf-8

import subprocess
import sys
import types

from src.kernel_eval.data.package_manager import PackageManager


def test_install_run_package_uses_supported_makeself_flags(monkeypatch, tmp_path):
    """The generated .run installer supports --quiet and --install-path."""
    run_file = tmp_path / "custom.run"
    run_file.write_text("#!/bin/sh\nexit 0\n")
    opp_path = tmp_path / "opp"
    opp_path.mkdir()

    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setenv("ASCEND_OPP_PATH", str(opp_path))
    monkeypatch.setattr(subprocess, "run", fake_run)

    PackageManager().install_run_package(str(run_file))

    assert calls
    cmd = calls[0]
    assert "--quiet" in cmd
    assert f"--install-path={opp_path}" in cmd
    assert "--force" not in cmd


def test_install_whl_force_reinstalls_same_version_without_dependencies(monkeypatch, tmp_path):
    """A new submission must replace an installed wheel with the same version."""
    wheel = tmp_path / "cann_bench-1.0.0-py3-none-any.whl"
    wheel.write_bytes(b"placeholder")
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert PackageManager().install_whl_package(str(wheel)) is True

    install_cmd = calls[0]
    assert install_cmd[1:4] == ["-m", "pip", "install"]
    assert "--force-reinstall" in install_cmd
    assert "--no-deps" in install_cmd
    assert str(wheel) in install_cmd


def test_scan_interfaces_reloads_cann_bench_submodules(monkeypatch, tmp_path):
    """A replacement submission must not reuse child modules from the old wheel."""
    package_dir = tmp_path / "cann_bench"
    package_dir.mkdir()
    package_dir.joinpath("__init__.py").write_text(
        "from .fresh_op import fresh_op\n", encoding="utf-8"
    )
    package_dir.joinpath("fresh_op.py").write_text(
        "def fresh_op(x):\n    return x\n", encoding="utf-8"
    )

    stale_package = types.ModuleType("cann_bench")
    stale_child = types.ModuleType("cann_bench.fresh_op")
    stale_package.fresh_op = lambda x: "stale"
    monkeypatch.setitem(sys.modules, "cann_bench", stale_package)
    monkeypatch.setitem(sys.modules, "cann_bench.fresh_op", stale_child)
    monkeypatch.syspath_prepend(str(tmp_path))

    interfaces = PackageManager().scan_interfaces()
    fresh_op = next(item.callable for item in interfaces if item.name == "fresh_op")

    assert fresh_op("fresh") == "fresh"
    assert sys.modules["cann_bench.fresh_op"] is not stale_child
