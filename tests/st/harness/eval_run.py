"""Invoke in-repo kernel_eval against a runtime candidate (the "调用 kernel_eval" part).

In-repo (cann-bench/tests/st): kernel_eval lives at <repo>/src, tasks at <repo>/tasks —
no vendored submodule, no baseline_mock. The candidate (golden_mock today, baseline_mock
later) is passed to the cli as --source-dir + exposed via PYTHONPATH.

**集成口径(single-run)**:不按 op 拆成 N 次 cli 调用,而是给 cli 一个**已按 -k/-m 修剪到
选中算子**的 --task-dir,**一次** `kernel_eval.cli eval`(不带 --operator)跑完整个子集 →
**单一报告**。让 cli 自己 discover+schedule,与真实 benchmark(scripts/run_evaluation.sh)
同一条编排路径。所有算子在独立子进程评测(OOM 保护、超时保护、进程隔离)。
--skip-install 进程内候选(PYTHONPATH 暴露 cann_bench),--task-dir/--reports-dir 被 respect。
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

ST_DIR = Path(__file__).resolve().parent.parent     # cann-bench/tests/st
REPO_ROOT = ST_DIR.parent.parent                     # cann-bench (holds src/, tasks/, scripts/)
KERNEL_EVAL_SRC = REPO_ROOT / "src"
TASKS = REPO_ROOT / "tasks"
RUN_EVALUATION_SH = REPO_ROOT / "scripts" / "run_evaluation.sh"
LEVELS = ("level1", "level2", "level3", "level4")


def has_npu() -> bool:
    """True iff torch + torch_npu importable and an NPU is visible."""
    try:
        import torch  # noqa: F401
        import torch_npu  # noqa: F401
        return bool(torch.npu.is_available())
    except Exception:
        return False


def _can_import_cann_bench_utils() -> tuple[bool, str]:
    """Check importability in a subprocess.

    A stale _C.abi3.so built against a different Python/torch ABI causes a fatal
    Aborted (not ImportError) on import — uncatchable and kills the process.
    Running in a subprocess isolates the crash so we can recover by rebuilding.

    Returns (ok, stderr) so callers can surface the actual error on final failure.
    """
    ret = subprocess.run(
        [sys.executable, "-c",
         "import torch, torch_npu; "
         "print(f'torch={torch.__version__} torch_npu={torch_npu.__version__}'); "
         "print(f'torch_path={torch.__file__}'); "
         "print(f'torch_npu_path={torch_npu.__file__}'); "
         "from cann_bench_utils import cann_bench_warmup, cann_bench_cache_clean; "
         "print('import OK')"],
        capture_output=True, text=True, timeout=30,
    )
    combined = (ret.stdout + "\n" + ret.stderr) if ret.stdout else ret.stderr
    return ret.returncode == 0, combined


def ensure_cann_bench_utils() -> None:
    """Ensure cann_bench_utils (V3 Anti-Cheat C++ extension) is importable.

    ST 绕过 run_evaluation.sh(后者有 ensure_cann_bench_utils)直调 kernel_eval.cli,
    需在此确保编译的 cann_bench_utils 已安装 —— perf_eval._boost_freq_and_clear_cache
    硬导入 cann_bench_warmup/cann_bench_cache_clean,未装时 import 失败会先于 fn()
    杀死整 case(精度+性能全失)。逻辑对齐 scripts/run_evaluation.sh:ensure_cann_bench_utils。
    """
    if _can_import_cann_bench_utils()[0]:
        return

    utils_dir = REPO_ROOT / "src" / "cann_bench_utils"
    if not utils_dir.is_dir():
        raise FileNotFoundError(
            f"cann_bench_utils 源码目录不存在: {utils_dir} — "
            "V3 Anti-Cheat 需要 cann_bench_utils,请检查代码库完整性"
        )

    # Uninstall any stale build (wrong-Python ABI → import Aborted, not ImportError).
    subprocess.run(
        [sys.executable, "-m", "pip", "uninstall", "-y", "cann_bench_utils"],
        capture_output=True, text=True,
    )

    # Physical purge: some CI images pre-install cann_bench_utils as RECORD-less
    # (egg / setup.py install). pip uninstall can't evict those ("No files were
    # found to uninstall"), leaving a stale .so that shadows the new wheel.
    # Remove all cann_bench_utils* from site-packages, but NOT from the source
    # tree (src/cann_bench_utils).
    import site
    for sp_dir in site.getsitepackages() + [site.getusersitepackages()]:
        sp = Path(sp_dir)
        if not sp.is_dir():
            continue
        for entry in sp.glob("cann_bench_utils*"):
            if entry.is_dir():
                shutil.rmtree(entry, ignore_errors=True)
            else:
                entry.unlink(missing_ok=True)
            print(f"[ST] purged stale {entry}")

    print("[ST] cann_bench_utils 未安装或 ABI 不匹配,开始自动编译安装...")
    build_env = dict(os.environ)
    build_env["PYTHON"] = sys.executable
    # Diagnostics: print torch versions that the BUILD subprocess will see
    build_diag = subprocess.run(
        [sys.executable, "-c",
         "import torch, torch_npu; "
         "print(f'BUILD torch={torch.__version__} path={torch.__file__}'); "
         "print(f'BUILD torch_npu={torch_npu.__version__} path={torch_npu.__file__}')"],
        capture_output=True, text=True, env=build_env,
    )
    print(f"[ST] {build_diag.stdout.strip()}")
    ret = subprocess.run(
        ["bash", "build.sh", "--clean"], cwd=str(utils_dir),
        capture_output=True, text=True, env=build_env,
    )
    if ret.returncode != 0:
        raise RuntimeError(
            f"cann_bench_utils 编译失败:\n"
            f"--- stdout (last 1500) ---\n{ret.stdout[-1500:]}\n"
            f"--- stderr (last 1500) ---\n{ret.stderr[-1500:]}"
        )
    print(f"[ST] build.sh stdout (first 500): {ret.stdout[:500]}")

    wheels = sorted(
        (utils_dir / "dist").glob("cann_bench_utils-*.whl"),
        key=lambda p: p.stat().st_mtime, reverse=True,
    )
    if not wheels:
        raise RuntimeError("cann_bench_utils 编译完成但未找到 wheel 包")
    wheel = wheels[0]

    ret = subprocess.run(
        [sys.executable, "-m", "pip", "install", str(wheel),
         "--force-reinstall", "--no-deps"],
        capture_output=True, text=True,
    )
    if ret.returncode != 0:
        raise RuntimeError(
            f"cann_bench_utils 安装失败:\n{ret.stderr[-2000:]}"
        )

    # 清除 sys.modules 中可能残留的 namespace package 缓存(PYTHONPATH=src 会暴露
    # src/cann_bench_utils/ 项目目录为 namespace package,首次 import 失败后残留在
    # sys.modules,即使 pip install 后 invalidate_caches 也不会清除 sys.modules 条目)。
    import importlib
    for key in [k for k in sys.modules if k == "cann_bench_utils" or k.startswith("cann_bench_utils.")]:
        del sys.modules[key]
    importlib.invalidate_caches()
    if not _can_import_cann_bench_utils()[0]:
        _, err = _can_import_cann_bench_utils()
        raise RuntimeError(
            f"cann_bench_utils 安装验证失败: import 仍异常\n"
            f"--- subprocess output (first 3000 chars) ---\n{err[:3000]}"
        )

    print(f"[ST] cann_bench_utils 安装成功 ({wheel.name})")


def kernel_eval_env(candidate_dir) -> dict:
    """Env for the cli subprocess: put in-repo kernel_eval (src) + the candidate package
    dir (exposing `cann_bench`) on PYTHONPATH.

    候选用 PYTHONPATH 暴露而非 `pip install -e` —— NPU 服务器的 immutable 容器里 uv ephemeral
    环境无需 setuptools/wheel/pip。cli 以 --skip-install 进程内评测,`import cann_bench` 经此解析。
    guard 模式由 build_eval_cmd 显式传 `--torch-op-guard-mode warn`(见该函数注释);
    Config 默认是 block,不显式降级会把 golden 合法的 torch builtin 当作弊误杀。
    """
    env = dict(os.environ)
    extra = os.pathsep.join([str(KERNEL_EVAL_SRC), str(candidate_dir)])
    prev = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = extra + (os.pathsep + prev if prev else "")
    return env


def build_eval_cmd(*, source_dir, task_dir, reports_dir, operator=None, case_id=None) -> list[str]:
    """kernel_eval.cli eval 命令。operator=None → 不过滤,跑遍 --task-dir 里所有算子(集成口径)。
    所有算子在独立子进程评测(OOM 保护、超时保护、进程隔离)。"""
    cmd = [
        sys.executable, "-m", "kernel_eval.cli", "eval",
        "--bench-name", "cann", "--device", "npu",
        "--source-dir", str(source_dir), "--skip-install",
        "--task-dir", str(task_dir), "--reports-dir", str(reports_dir),
        # golden 是参考实现,合法使用 torch 内置算子(matmul 系 golden 如 WeightQuantBatchMatmul
        # 必然调 torch.matmul)。TorchOpGuard 的 Config 默认是 block,会把 golden 当"调内置算子作弊"
        # 直接 raise [SECURITY] → 该算子全 case 失败。golden 候选应只告警不 gate,故显式降为 warn,
        # 与上方 kernel_eval_env 注释及 e2e 冒烟(--torch-op-guard-mode warn)保持一致。
        "--torch-op-guard-mode", "warn",
    ]
    if operator is not None:
        cmd += ["--operator", str(operator)]
    if case_id is not None:
        cmd += ["--case-id", str(case_id)]
    return cmd


def run_eval_cli(*, source_dir, task_dir, reports_dir, operator=None, case_id=None,
                 timeout=14400):
    """Run ONE kernel_eval.cli eval over --task-dir; returns CompletedProcess (capture_output).
    operator=None 跑遍整棵(已修剪的)task 树 → 单一报告。候选包经 PYTHONPATH 暴露(=source_dir)。
    timeout 默认放宽到 4h:single-run 覆盖整个选中子集(逐 op 子进程串行)。"""
    return subprocess.run(
        build_eval_cmd(source_dir=source_dir, task_dir=task_dir, reports_dir=reports_dir,
                       operator=operator, case_id=case_id),
        env=kernel_eval_env(source_dir), capture_output=True, text=True, timeout=timeout,
    )
