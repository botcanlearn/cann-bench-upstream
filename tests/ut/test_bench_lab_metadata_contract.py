#!/usr/bin/python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
# See the repository LICENSE for the full text.

"""Check directory-keyed metadata without requiring torch or an NPU."""

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SUITES = ("kernel_bench", "omni-ops_bench")


def _metadata_operators():
    for suite in SUITES:
        bench_root = ROOT / "bench_lab" / suite
        for path in sorted((bench_root / "metadata").glob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            for key, value in data.items():
                if key.startswith("_"):
                    continue
                entries = value.items() if key.startswith("level") else [(key, value)]
                for name, cases in entries:
                    relative = f"{key}/{name}" if key.startswith("level") else name
                    yield pytest.param(
                        bench_root, path.stem, relative, cases,
                        id=f"{suite}/{path.stem}/{relative}",
                    )


@pytest.fixture
def baseline_store_class(monkeypatch):
    # Import the real standard-library-only modules in an isolated package.
    # Importing kernel_eval.utils normally also imports torch-dependent utilities.
    package_name = "_bench_metadata_contract"
    package = types.ModuleType(package_name)
    package.__path__ = [str(ROOT / "src/kernel_eval/utils")]
    monkeypatch.setitem(sys.modules, package_name, package)
    for name in ("baseline_resolver", "baseline_store"):
        full_name = f"{package_name}.{name}"
        spec = importlib.util.spec_from_file_location(
            full_name, ROOT / "src/kernel_eval/utils" / f"{name}.py",
        )
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, full_name, module)
        spec.loader.exec_module(module)
    return module.BaselineStore


@pytest.mark.parametrize("bench_root,hardware,relative,cases", list(_metadata_operators()))
def test_metadata_keys_resolve_to_delivery_and_performance(
    baseline_store_class, bench_root, hardware, relative, cases,
):
    folder = bench_root / relative
    assert (folder / "proto.yaml").is_file(), f"Stale metadata path: {folder}"
    assert cases, f"Empty metadata for {relative}"
    # Start at the delivery directory to cover the runner's upward search.
    store = baseline_store_class(folder, project_root=ROOT, hardware=hardware)
    store.load()
    for case_id, expected in cases.items():
        assert store._resolve_entry(relative, int(case_id)) == expected
        for field, getter in (
            ("baseline_perf_us", store.get_perf), ("t_hw_us", store.get_t_hw),
        ):
            value = expected.get(field)
            if isinstance(value, dict):
                value = value.get(hardware)
            # Null/zero means unmeasured; do not invent measurements or disable
            # the production loader's intentional default-platform fallback.
            if value is not None and value > 0:
                assert getter(relative, int(case_id)) == pytest.approx(value)
        baseline = expected.get("baseline_perf_us")
        if isinstance(baseline, dict):
            baseline = baseline.get(hardware)
        if baseline is not None:
            assert store.has_baseline(relative, int(case_id))
