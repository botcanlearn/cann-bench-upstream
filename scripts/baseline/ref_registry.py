"""Reference implementation registry — auto-loads refs/level{1,2,3,4}.py.

Each `refs/levelN.py` (or any other module dropped into refs/) must export a
top-level dict named `REGISTRY` mapping operator paths ("levelN/op_dir") to
callables of the form `ref(inputs, attrs) -> outputs`. inputs already lives
on the target NPU device.

When no NPU equivalent exists for an op, simply omit the entry — the harness
will skip method 3 ("npu_ref") for that op.
"""
from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
from typing import Callable, Dict, Optional


_REG: Optional[Dict[str, Callable]] = None


def _load() -> Dict[str, Callable]:
    here = Path(__file__).resolve().parent / "refs"
    out: Dict[str, Callable] = {}
    if not here.is_dir():
        return out
    for p in sorted(here.glob("*.py")):
        if p.name == "__init__.py":
            continue
        spec = importlib.util.spec_from_file_location(f"_3way_refs_{p.stem}", p)
        try:
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
        except Exception as e:
            # subagents may write partial files; print but don't abort
            print(f"[ref_registry] WARN: failed to load {p.name}: {e}")
            continue
        reg = getattr(mod, "REGISTRY", None)
        if isinstance(reg, dict):
            out.update(reg)
    return out


def _ensure() -> Dict[str, Callable]:
    global _REG
    if _REG is None:
        _REG = _load()
    return _REG


def get_ref(op_path: str) -> Optional[Callable]:
    return _ensure().get(op_path)


def all_keys():
    return list(_ensure().keys())
