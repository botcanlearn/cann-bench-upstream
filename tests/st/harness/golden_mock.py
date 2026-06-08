"""Build the runtime golden candidate (the "golden mock" part).

A future baseline_mock.py will sit alongside this, building a torch_npu baseline candidate
the same way (private). Both produce a `cann_bench` package consumed by eval_run.run_eval_cli.
"""
from __future__ import annotations

import ast
import shutil
from pathlib import Path

from .eval_run import TASKS, LEVELS


def _assert_golden_defines(golden_py: Path, op_id: str) -> None:
    """Guard the implicit contract: golden.py 必须有顶层 `def <op_id>`(目录名 == golden 函数名)。

    re-export 是 `from ._goldens.<lvl>__<op_id> import <op_id>`,若 golden.py 没有该顶层函数,
    错误会以一条难定位的 ImportError 出现在运行时生成的 cann_bench/__init__.py 里。这里提前
    用 AST 校验(只认**顶层** def/async def —— 嵌套函数 import 不到),给出清晰的报错。
    """
    top_defs = {
        n.name for n in ast.parse(golden_py.read_text(encoding="utf-8")).body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert op_id in top_defs, (
        f"{golden_py}: 未定义顶层 `def {op_id}()`。golden mock 约定目录名必须 == golden 函数名"
        f"(re-export `from ._goldens.<lvl>__{op_id} import {op_id}`)。现有顶层函数: {sorted(top_defs)}"
    )


def build_golden_candidate(dest, tasks_dir=TASKS) -> Path:
    """Generate a `cann_bench` package under `dest` re-exporting every task's golden fn.

    Each tasks/<lvl>/<op>/golden.py defines a top-level `def <op>` of pure device-agnostic
    torch — copy it in verbatim and re-export under its op id (== the schema function name
    kernel_eval looks up). Built fresh at test time, not committed (always in sync with tasks/).
    Returns `dest` — pass it to kernel_eval as --source-dir.
    """
    dest = Path(dest)
    pkg = dest / "cann_bench"
    goldens = pkg / "_goldens"
    goldens.mkdir(parents=True, exist_ok=True)
    (goldens / "__init__.py").write_text("", encoding="utf-8")

    reexports, names = [], []
    for lvl in LEVELS:
        base = Path(tasks_dir) / lvl
        for op_dir in sorted(p for p in base.iterdir() if p.is_dir()):
            op_id = op_dir.name
            golden_py = op_dir / "golden.py"
            _assert_golden_defines(golden_py, op_id)
            shutil.copyfile(golden_py, goldens / f"{lvl}__{op_id}.py")
            reexports.append(f"from ._goldens.{lvl}__{op_id} import {op_id} as {op_id}\n")
            names.append(op_id)

    body = ["# AUTO-GENERATED golden candidate (runtime build) -- not committed\n"]
    body += reexports
    body.append("\n__all__ = [\n")
    body += [f'    "{n}",\n' for n in sorted(names)]
    body.append("]\n")
    (pkg / "__init__.py").write_text("".join(body), encoding="utf-8")
    return dest
