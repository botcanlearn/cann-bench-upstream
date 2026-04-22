#!/usr/bin/env python3
"""Render a human-friendly summary.md from evaluation_results.json (+ optional
stages result.json). Used by simulate_runner.sh and suitable for any post-run
consumer that wants a readable artifact alongside the raw JSON."""
import argparse
import json
from pathlib import Path


def _fmt_us(v):
    return f"{v:.2f}" if isinstance(v, (int, float)) and v > 0 else "—"


def _fmt_speedup(v):
    return f"{v:.3f}x" if isinstance(v, (int, float)) and v > 0 else "—"


def _fmt_duration(sec):
    if sec is None:
        return "—"
    if sec < 60:
        return f"{sec}s"
    return f"{sec // 60}m{sec % 60:02d}s"


def render(eval_results: dict, stages: dict | None) -> str:
    out: list[str] = []
    out.append("# Evaluation summary\n")

    if stages:
        out.append(f"- **Job ID**: `{stages.get('job_id', '—')}`")
        out.append(f"- **Device**: NPU {stages.get('device_id', '—')}")
    out.append(f"- **Hardware**: {eval_results.get('hardware', '—')}")
    if stages:
        out.append(f"- **Final status**: {stages.get('final_status', '—')}")
    total = eval_results.get("total_cases", 0)
    passed = eval_results.get("total_passed", 0)
    n_ops = eval_results.get("total_operators", 0)
    geo = eval_results.get("overall_geometric_mean_speedup", 0.0)
    out.append(f"- **Operators**: {n_ops}")
    out.append(f"- **Cases passed**: {passed}/{total}")
    out.append(f"- **Overall geomean speedup**: {_fmt_speedup(geo)}")
    out.append("")

    if stages and stages.get("stages"):
        out.append("## Stages\n")
        out.append("| Stage | Status | Duration |")
        out.append("| --- | --- | --- |")
        for name, info in stages["stages"].items():
            status = info.get("status", "—")
            dur = _fmt_duration(info.get("duration_sec"))
            out.append(f"| {name} | {status} | {dur} |")
        out.append("")

    operators = eval_results.get("operators", [])
    if operators:
        out.append("## Operators\n")
        out.append("| Operator | Passed | Geomean speedup |")
        out.append("| --- | --- | --- |")
        for op in operators:
            out.append(
                f"| {op['operator']} | "
                f"{op['passed_cases']}/{op['total_cases']} | "
                f"{_fmt_speedup(op.get('geometric_mean_speedup', 0))} |"
            )
        out.append("")

    # Include both yaml + measured baselines when evaluate was run with
    # --measure-baselines. Detect by checking if any result has a
    # `baseline_measured_us` populated.
    show_both = any(
        r.get("baseline_measured_us") not in (None, 0)
        for op in operators for r in op.get("results", [])
    )

    for op in operators:
        header = (
            f"### {op['operator']} "
            f"({op['passed_cases']}/{op['total_cases']} passed, "
            f"{_fmt_speedup(op.get('geometric_mean_speedup', 0))})\n"
        )
        out.append(header)
        if show_both:
            out.append("| # | Case | Status | Speedup | Baseline yaml µs | Baseline measured µs | Custom µs | Detail |")
            out.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        else:
            out.append("| # | Case | Status | Speedup | Baseline µs | Custom µs | Detail |")
            out.append("| --- | --- | --- | --- | --- | --- | --- |")
        for r in op.get("results", []):
            detail = (r.get("detail") or "").replace("|", "\\|")
            if show_both:
                out.append(
                    f"| {r['case_id']} | "
                    f"`{r['case_name']}` | "
                    f"{r['status']} | "
                    f"{_fmt_speedup(r.get('speedup'))} | "
                    f"{_fmt_us(r.get('baseline_yaml_us'))} | "
                    f"{_fmt_us(r.get('baseline_measured_us'))} | "
                    f"{_fmt_us(r.get('custom_time_us'))} | "
                    f"{detail} |"
                )
            else:
                out.append(
                    f"| {r['case_id']} | "
                    f"`{r['case_name']}` | "
                    f"{r['status']} | "
                    f"{_fmt_speedup(r.get('speedup'))} | "
                    f"{_fmt_us(r.get('baseline_perf_us'))} | "
                    f"{_fmt_us(r.get('custom_time_us'))} | "
                    f"{detail} |"
                )
        out.append("")

    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description="Render summary.md from evaluation_results.json")
    ap.add_argument("--evaluation-results", required=True, type=Path)
    ap.add_argument("--stages", type=Path, default=None,
                    help="Optional result.json with per-stage timings")
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()

    eval_results = json.loads(args.evaluation_results.read_text())
    stages = json.loads(args.stages.read_text()) if args.stages and args.stages.exists() else None

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(eval_results, stages))
    print(f"[cann-bench] Summary written to {args.output}")


if __name__ == "__main__":
    main()
