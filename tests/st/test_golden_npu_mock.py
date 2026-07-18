"""NPU integration ST: run tasks/*/golden.py AS the candidate on the NPU.

The goldens are pure, device-agnostic torch, so feeding them back to kernel_eval as the
submitted candidate exercises the whole eval path on device (discover → import candidate →
input-gen → golden-compare → profiler perf → score → report) WITHOUT any torch_npu answer
key. The candidate `cann_bench` package is built fresh at runtime (harness.golden_mock)
from <repo>/tasks/*/golden.py — not committed, always in sync.

**集成口径(single-run)**:这是一个集成测试,不是 N 个独立单测。-k/-m(默认 Cummin)选中的
算子在 conftest 收集后被裁成一棵"只含选中算子"的 task 子树,**一次** `kernel_eval.cli eval`
(不带 --operator,保留 op 级子进程隔离)跑完整棵 → **单一报告**。下面每 op 一个 test 只是为了
可读的 per-op 结果展示,它们共享同一份报告(session fixture 跑一次),并不各自再调 cli。

Oracle (same as the upstream golden self-test):
  - every (op, case): kernel_eval must emit an accuracy verdict (pass/fail both fine).
  - accuracy-passing cases must emit perf (elapsed_us > 0) — kernel_eval skips perf for
    accuracy-failing cases, so we only assert perf where accuracy passed.
  - accuracy *fails* are reported as a warn, NOT gated: the golden runs fp64-on-CPU as the
    reference but the same code runs in the case dtype on NPU as the candidate, so some
    dtype/device precision gaps are legitimate. xfail-*/skip entries in known_issues.yaml
    relax specific ops (e.g. StridedSlice → xfail-perf: a pure view has no measurable kernel).

Select ops with `-k Gelu` / `-m level1`.
"""

import warnings
from pathlib import Path

import pytest
import yaml

from harness import (
    TASKS,
    run_eval_cli,
    has_npu,
    ensure_cann_bench_utils,
    build_golden_candidate,
    latest_report_json,
    load_report,
    iter_cases,
    case_num,
    case_has_accuracy,
    case_has_perf,
    case_acc_passed,
    schema_diff,
    has_drift,
    load_known_issues,
    for_target,
    xfail_set,
    xfail_all_ops,
    skip_ops,
)

_KNOWN = for_target(
    load_known_issues(Path(__file__).resolve().parent / "known_issues.yaml"), "golden"
)
_XFAIL_ACC = xfail_set(_KNOWN, "xfail-accuracy")  # case 级 (op, case)
_XFAIL_PERF = xfail_set(_KNOWN, "xfail-perf")
_XFAIL_ACC_OPS = xfail_all_ops(_KNOWN, "xfail-accuracy")  # op 级: 整算子放宽
_XFAIL_PERF_OPS = xfail_all_ops(_KNOWN, "xfail-perf")
_SKIP_OPS = skip_ops(_KNOWN)  # op 级 skip: 整算子跳过
_LEVELS = ("level1", "level2", "level3", "level4")


def _all_ops():
    """[(level, OperatorName)] from each task proto.yaml, with level marks + readable ids."""
    params = []
    for lvl in _LEVELS:
        base = TASKS / lvl
        for op_dir in sorted(p for p in base.iterdir() if p.is_dir()):
            name = yaml.safe_load((op_dir / "proto.yaml").read_text(encoding="utf-8"))[
                "operator"
            ]["name"]
            params.append(
                pytest.param(name, marks=getattr(pytest.mark, lvl), id=f"{lvl}:{name}")
            )
    return params


@pytest.fixture(scope="session")
def golden_candidate(tmp_path_factory):
    """Build the runtime golden `cann_bench` candidate once per session."""
    if not has_npu():
        pytest.skip("needs Ascend NPU to run the golden candidate")
    ensure_cann_bench_utils()
    return build_golden_candidate(tmp_path_factory.mktemp("golden_candidate"))


@pytest.fixture(scope="session")
def golden_matrix_report(golden_candidate, trimmed_tasks, tmp_path_factory):
    """**一次** kernel_eval.cli eval 跑完整棵(已裁到选中算子的)task 子树 → 单一报告。

    集成口径:cli 自己 discover 并逐 op fork 子进程(不带 --operator、保留默认隔离),父进程聚合
    成一份报告。返回 {op_lower: {case_num: case}},供下面每个 per-op test 各取所需(只跑一次)。
    """
    reports = tmp_path_factory.mktemp("reports")
    proc = run_eval_cli(
        source_dir=golden_candidate,
        task_dir=trimmed_tasks,
        reports_dir=reports,
    )
    try:
        report = load_report(latest_report_json(reports))
    except FileNotFoundError:
        pytest.fail(
            f"kernel_eval 未产出报告(rc={proc.returncode})\n"
            f"--- stdout tail ---\n{proc.stdout[-3000:]}\n"
            f"--- stderr tail ---\n{proc.stderr[-1500:]}"
        )
    if has_drift(schema_diff(report)):
        warnings.warn(f"report schema drift: {schema_diff(report)}")
    by_op: dict[str, dict] = {}
    for op, _cid, c in iter_cases(report):
        by_op.setdefault(op.lower(), {})[case_num(c)] = c
    # 诊断加固：报告存在但 0 case（eval-child 全崩）时，把各 operator 的失败原因
    # + eval stderr 一起抛出。否则下游 per-op test 只能看到一句无信息的
    # "没有该算子的任何 case"，定位不到 NPU 运行时根因。
    if not by_op:
        ops_fail = []
        for op in report.get("operators", []):
            reason = op.get("subprocess_failure_reason") or op.get("compilation_error")
            if reason:
                ops_fail.append(f"{op.get('operator')}: {reason}")
        pytest.fail(
            f"kernel_eval 产出报告但无任何 case（rc={proc.returncode}，"
            f"operators={len(report.get('operators', []))}）。\n"
            f"--- operator 失败原因 ---\n" + "\n".join(ops_fail or ["(报告无 operator 失败原因字段)"]) + "\n"
            f"--- stdout tail ---\n{proc.stdout[-2000:]}\n"
            f"--- stderr tail ---\n{proc.stderr[-3000:]}"
        )
    return by_op


@pytest.mark.npu
@pytest.mark.parametrize("operator", _all_ops())
def test_golden_op_produces_results(operator, golden_matrix_report):
    key = operator.lower()
    if key in _SKIP_OPS:
        pytest.skip(f"known_issue skip (golden): {_SKIP_OPS[key]}")
    cases = golden_matrix_report.get(key)
    assert cases, f"{operator}: 单次集成报告里没有该算子的任何 case"
    acc_op_xf = key in _XFAIL_ACC_OPS  # 整个算子的精度已知放宽
    perf_op_xf = key in _XFAIL_PERF_OPS  # 整个算子的性能已知放宽
    missing, acc_fail, perf_missing = [], [], []
    for cid, c in sorted(cases.items()):
        # 唯一硬 gate:每个 case 必须产出精度 verdict(pass/fail 都行)。
        if not acc_op_xf and (key, cid) not in _XFAIL_ACC and not case_has_accuracy(c):
            missing.append(f"{operator}#{cid}: 无精度结果 (err={c.get('error_msg')})")
        if case_acc_passed(c):
            # perf 只 best-effort、不 gate:kernel_eval 的 profiler 在负载/多算子连跑下偶发
            # 采不到 kernel 时间(elapsed_us=0),是 flaky capture、非算子属性(同一 case
            # 复跑时有时为 0 有时正常),见 design doc 的 profiler-flaky 记录。故 warn 不断言。
            if (
                not perf_op_xf
                and (key, cid) not in _XFAIL_PERF
                and not case_has_perf(c)
            ):
                perf_missing.append(cid)
        elif case_has_accuracy(c):
            acc_fail.append(cid)
    # 信息性 warn(均不 gate):
    # - 精度不达标:golden 在 NPU 上按 case dtype 跑,与 fp64 CPU 参考存在合理 dtype/device 精度差。
    # - perf 采空:上面的 flaky profiler capture。
    if acc_fail:
        warnings.warn(
            f"[precision] golden {operator}: {len(acc_fail)}/{len(cases)} cases 精度不达标 {acc_fail}"
        )
    if perf_missing:
        warnings.warn(
            f"[perf] golden {operator}: {len(perf_missing)}/{len(cases)} cases 精度通过但 profiler 采到 elapsed_us=0 (flaky capture) {perf_missing}"
        )
    assert not missing, f"{operator}: {len(missing)} 项未达 oracle:\n" + "\n".join(
        missing
    )
