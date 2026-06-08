"""Fixtures + default selection for the golden-candidate NPU ST (tests/st/).

conftest puts tests/st on sys.path so `import harness` resolves (st is NOT a package).
Default run = only the Cummin smoke; widen with `--full` / `-k` / `-m`. @pytest.mark.npu
auto-skips off-NPU, so `pytest tests/st` is safe on a dev box.
"""
from __future__ import annotations

import sys
from pathlib import Path

# tests/st on sys.path so `import harness` resolves (st is not a package; no root pytest config).
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pytest

from harness import (
    TASKS, has_npu, load_known_issues, hang_cases, build_trimmed_tasktree,
)

KNOWN_ISSUES_PATH = Path(__file__).resolve().parent / "known_issues.yaml"

requires_npu = pytest.mark.skipif(
    not has_npu(), reason="needs Ascend NPU (run on the NPU server)"
)


def pytest_addoption(parser):
    parser.addoption(
        "--full", action="store_true", default=False,
        help="run the full operator matrix (default: only the Cummin smoke). 注意:single-run "
             "口径下每 op 在 kernel_eval 的子进程隔离里只有 240s(硬编码)超时,重算子(SFA/MLA/"
             "Moe*/NMS/ROIAlign 等)可能超时失败 —— 全量跑需要先让 kernel_eval 的 per-op 子进程超时可配。",
    )


def pytest_configure(config):
    for line in (
        "npu: 需要 Ascend NPU 才能跑的集成测试",
        "level1: L1 算子", "level2: L2 算子", "level3: L3 算子", "level4: L4 算子",
    ):
        config.addinivalue_line("markers", line)


@pytest.fixture(scope="session")
def known_issues():
    return load_known_issues(KNOWN_ISSUES_PATH)


@pytest.fixture(scope="session")
def trimmed_tasks(request, tmp_path_factory, known_issues):
    """<repo>/tasks 的副本,**只保留本次选中的算子**(-k/-m/默认 Cummin)+ 删掉 skip-hang 用例。
    这是 single-run 集成口径的输入:cli 一次跑完这棵子树 → 单一报告。纯文件操作,不需 NPU。"""
    dst = tmp_path_factory.mktemp("trimmed") / "tasks"
    keep_ops = getattr(request.config, "_st_selected_ops", None)
    return build_trimmed_tasktree(TASKS, dst, hang_cases(known_issues), keep_ops=keep_ops)


def _selected_ops(items) -> set[str]:
    """从收集后的 test item 取选中的算子名(parametrize 的 `operator` 形参)。"""
    ops = set()
    for it in items:
        cs = getattr(it, "callspec", None)
        if cs and "operator" in cs.params:
            ops.add(cs.params["operator"])
    return ops


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config, items):
    """默认只跑 Cummin 冒烟(除非 --full / -k / -m);非 NPU 机器 auto-skip @pytest.mark.npu;
    并把最终选中的算子集合记到 config._st_selected_ops,供 trimmed_tasks 裁剪 task 树。

    **trylast 关键**:-k/-m 的反选发生在 pytest 内置的 pytest_collection_modifyitems 里;
    本钩子必须在其**之后**运行,items[] 才是过滤后的最终集,_st_selected_ops 才不会误抓全部 53 个。
    """
    if (not config.getoption("--full")
            and not config.option.keyword and not config.option.markexpr):
        keep = [it for it in items if "Cummin" in it.name]
        drop = [it for it in items if it not in keep]
        if drop:
            config.hook.pytest_deselected(items=drop)
            items[:] = keep
    # items[] 此刻已是最终选中集 → 锁定要喂给 single-run 的算子子集。
    config._st_selected_ops = _selected_ops(items)
    if not has_npu():
        skip = pytest.mark.skip(reason="needs Ascend NPU (run on the NPU server)")
        for it in items:
            if "npu" in it.keywords:
                it.add_marker(skip)
