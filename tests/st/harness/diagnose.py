"""Post-mortem digest of an ST run, printed to stdout (`python -m harness.diagnose $ST_OUT`).

CI keeps the job's stdout and nothing else — no _artifacts download, no container to re-enter.
So anything needed to locate a failure must be *printed*, and printed within a budget: a 4h
--full run's eval_cli.log is tens of MB, and pytest's own failure block already burns 40 lines
repeating one synthesized error. This module turns $ST_OUT into a bounded, greppable digest:

  report  → per-op failure reasons + per-case failures COLLAPSED by identical cause
            (20 cases, one reason → one line + a case range, not 20 lines)
  log     → every Python traceback block, error-ish lines, and the tail

Every line is prefixed `[ST-DIAG]`, so `grep '\\[ST-DIAG\\]'` over the CI log reconstructs the
whole post-mortem out of the interleaved pipeline noise.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

from .eval_run import EVAL_LOG_NAME

P = "[ST-DIAG]"

# Lines worth surfacing from the eval log even when no traceback was printed: CANN/ACL runtime
# errors (EZ9999/EI0002/ERR01005...), OOM/segv death notices, and the scheduler's own [ERROR].
ERROR_LINE_RE = re.compile(
    r"Traceback|\[ERROR\]|\bE[A-Z]\d{4}\b|\bERR\d{5}\b|Error:|Exception|RuntimeError"
    r"|Segmentation fault|core dumped|Aborted|Killed|Out of memory|OOM|CANNOT|FAILED",
    re.IGNORECASE,
)
MAX_TRACEBACKS = 6
MAX_ERROR_LINES = 60
DEFAULT_TAIL = 120


def _out(line: str = "") -> None:
    print(f"{P} {line}" if line else P, flush=True)


def _section(title: str) -> None:
    print(flush=True)
    _out(f"===== {title} =====")


# ── case-number ranges ────────────────────────────────────────────────────────
def compact_ranges(nums) -> str:
    """[1,2,3,5,9,10] -> '1-3,5,9-10'. Keeps a 20-case collapse readable in one line."""
    ns = sorted(set(nums))
    if not ns:
        return "-"
    out, start, prev = [], ns[0], ns[0]
    for n in ns[1:] + [None]:
        if n is not None and n == prev + 1:
            prev = n
            continue
        out.append(str(start) if start == prev else f"{start}-{prev}")
        start = prev = n
    return ",".join(out)


def _case_num(case: dict):
    """Trailing int of a composite case_id ('level2/cummin_3' -> 3); None if unparsable."""
    cid = case.get("case_id")
    if isinstance(cid, int):
        return cid
    try:
        return int(str(cid).rsplit("_", 1)[-1])
    except (ValueError, AttributeError):
        return None


def _acc_error(case: dict) -> str:
    acc = case.get("accuracy")
    return str(acc.get("error_msg") or "") if isinstance(acc, dict) else ""


# ── report ────────────────────────────────────────────────────────────────────
def digest_report(report: dict) -> None:
    s = report.get("summary") or {}
    _out(f"device={report.get('device')} eval_code={report.get('eval_code')} "
         f"ops={s.get('total_operators', report.get('total_operators'))} "
         f"cases={s.get('total_cases', report.get('total_cases'))} "
         f"passed={s.get('passed_cases', report.get('passed_cases'))} "
         f"failed={s.get('failed_cases', report.get('failed_cases'))} "
         f"score={s.get('overall_score', report.get('overall_score'))}")
    for op in report.get("operators") or []:
        _out(f"op {op.get('operator')} ({op.get('rel_path')}): "
             f"{op.get('passed_cases')}/{op.get('total_cases')} passed score={op.get('score')}")
        # op-level reasons: set when the whole op died before producing cases. Never printed
        # by the per-op assertion, which only sees case['error_msg'].
        for field in ("subprocess_failure_reason", "compilation_error",
                      "score_error", "score_error_code", "compile_runtime_fail_cases"):
            if op.get(field):
                _out(f"    {field}: {op[field]}")
        # Collapse failing cases by identical cause — N cases with one root cause is one fact.
        buckets: dict[tuple, list] = {}
        for case in op.get("cases") or []:
            if case.get("accuracy") is not None and (case.get("elapsed_us") or 0) > 0:
                continue  # fully healthy case: no verdict gap, no perf gap
            key = (case.get("status"), case.get("failure_type"),
                   str(case.get("error_msg") or ""), _acc_error(case),
                   case.get("accuracy") is None, (case.get("elapsed_us") or 0) > 0)
            buckets.setdefault(key, []).append(_case_num(case) or 0)
        for (status, ftype, err, acc_err, no_acc, has_perf), nums in sorted(
                buckets.items(), key=lambda kv: -len(kv[1])):
            _out(f"    x{len(nums)} cases [{compact_ranges(nums)}] status={status} "
                 f"failure_type={ftype} accuracy={'MISSING' if no_acc else 'present'} "
                 f"perf={'ok' if has_perf else 'MISSING'}")
            if err:
                _out(f"        error_msg: {err}")
            if acc_err:
                _out(f"        accuracy.error_msg: {acc_err[:500]}")


# ── eval_cli.log ──────────────────────────────────────────────────────────────
def traceback_blocks(lines) -> list[list[str]]:
    """Python traceback blocks: 'Traceback (most recent call last):' through the first
    following non-indented line (the `SomeError: msg` terminator), inclusive."""
    blocks, cur = [], None
    for line in lines:
        if line.startswith("Traceback (most recent call last)"):
            if cur:
                blocks.append(cur)
            cur = [line]
        elif cur is not None:
            cur.append(line)
            if line.strip() and not line[0].isspace():
                blocks.append(cur)
                cur = None
    if cur:
        blocks.append(cur)
    return blocks


def digest_log(text: str, tail: int = DEFAULT_TAIL, full: bool = False) -> None:
    lines = text.splitlines()
    _out(f"{len(lines)} lines, {len(text)} bytes")
    if full:
        for line in lines:
            _out(line)
        return

    blocks = traceback_blocks(lines)
    _out(f"tracebacks: {len(blocks)}"
         + (f" (showing last {MAX_TRACEBACKS})" if len(blocks) > MAX_TRACEBACKS else ""))
    for block in blocks[-MAX_TRACEBACKS:]:
        _out()
        for line in block:
            _out(f"  | {line}")

    hits = [(i, l) for i, l in enumerate(lines, 1)
            if ERROR_LINE_RE.search(l) and not l.startswith("Traceback")]
    _out()
    _out(f"error-ish lines: {len(hits)}"
         + (f" (showing last {MAX_ERROR_LINES})" if len(hits) > MAX_ERROR_LINES else ""))
    for i, line in hits[-MAX_ERROR_LINES:]:
        _out(f"  {i}: {line}")

    _out()
    _out(f"tail {min(tail, len(lines))} lines (ST_DIAG_FULL=1 to dump the whole log):")
    for line in lines[-tail:]:
        _out(f"  | {line}")


# ── entry ─────────────────────────────────────────────────────────────────────
def main(argv=None) -> int:
    argv = list(argv if argv is not None else sys.argv[1:])
    out_dir = Path(argv[0] if argv else os.environ.get("ST_OUT", "tests/st/_artifacts"))
    tail = int(os.environ.get("ST_DIAG_TAIL", DEFAULT_TAIL))
    full = os.environ.get("ST_DIAG_FULL", "") not in ("", "0")

    _section(f"artifacts in {out_dir}")
    if not out_dir.is_dir():
        _out(f"MISSING: {out_dir} does not exist — the run died before producing anything")
        return 0
    for f in sorted(out_dir.rglob("*")):
        if f.is_file():
            _out(f"{f.stat().st_size:>12} B  {f.relative_to(out_dir)}")

    for jf in sorted(out_dir.glob("*eval_*.json")):
        _section(f"report {jf.name}")
        try:
            digest_report(json.loads(jf.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError) as e:
            _out(f"unreadable: {e}")

    log = out_dir / EVAL_LOG_NAME
    _section(f"kernel_eval cli log ({EVAL_LOG_NAME})")
    if log.is_file():
        digest_log(log.read_text(encoding="utf-8", errors="replace"), tail=tail, full=full)
    else:
        _out("MISSING — the cli was never launched (fixture failed earlier: candidate build, "
             "cann_bench_utils compile, task-tree trim). See pytest's 'Captured stdout setup'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
