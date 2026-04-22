#!/usr/bin/env python3
"""
Standardized evaluation script for CANN Kernel Bench.

Takes a user-submitted `cann_bench` wheel (either `aclnn_launch` or
`direct_launch_simple` style — both expose the same `cann_bench.<op>` API)
and evaluates it against one or more benchmarks from kernel_bench/.

Precision checking uses MERE/MARE from core/precision_checker.py.

Usage:
    # Single operator
    python3 evaluate.py \\
        --submission <path-to-submission-dir-or-wheel> \\
        --operator-dir <path-to-kernel_bench/levelN/opname> \\
        [--device-id <n>] [--json-output <path>] [--skip-performance]

    # Batch: evaluate every operator under a bench root
    python3 evaluate.py \\
        --submission <path-to-submission> \\
        --bench-root <path-to-kernel_bench> [--levels 0,1] \\
        [--device-id <n>] [--json-output <path>] [--skip-performance]
"""

import os
import sys
import json
import math
import yaml
import argparse
import importlib
import importlib.util
import inspect
import logging
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Setup ASCEND env before torch_npu import
# ---------------------------------------------------------------------------
if "ASCEND_CUSTOM_OPP_PATH" not in os.environ:
    for p in [
        "/usr/local/Ascend/cann-8.5.0/opp/vendors/customize",
        "/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize",
    ]:
        if os.path.isdir(p):
            os.environ["ASCEND_CUSTOM_OPP_PATH"] = p
            break

# Make `evaluation.core` importable in both layouts:
#   repo:   <repo>/evaluation/evaluate.py          → grandparent has `evaluation/`
#   bundle: <bundle>/evaluate.py                   → parent has `evaluation/`
_eval_file_dir = os.path.dirname(os.path.abspath(__file__))
for candidate in (_eval_file_dir, os.path.dirname(_eval_file_dir)):
    if os.path.isdir(os.path.join(candidate, "evaluation")) and candidate not in sys.path:
        sys.path.insert(0, candidate)

from evaluation.core.precision_checker import PrecisionChecker
from evaluation.core.data_generator import DataGenerator
from evaluation.core.case_loader import CaseInfo
from evaluation.core.param_builder import ParamBuilder
from evaluation.core.profiler_manager import measure_kernel_us

# ---------------------------------------------------------------------------
# Anti-tampering: snapshot critical timing-API identities BEFORE the
# submission runs any code, then verify they haven't been monkey-patched.
# Defeats Attack #1 (override Event.elapsed_time to lie).
# ---------------------------------------------------------------------------

_CRITICAL_API_IDS: Dict[str, int] = {}

def snapshot_timing_apis() -> None:
    """Take id() of every timing-critical callable before the submission
    module gets a chance to patch them. Must run after `import torch_npu`
    and before `install_submission_wheel`.

    Covers both the legacy event-based timing path (kept as defense in
    depth) and the torch_npu.profiler path now used by measure_perf.
    """
    import torch
    import torch_npu  # noqa
    _CRITICAL_API_IDS.update({
        # Legacy event-based timing (no longer on the hot path, but kept
        # so wheels that monkey-patch them still get caught at startup).
        "torch.npu.Event.elapsed_time": id(torch.npu.Event.elapsed_time),
        "torch.npu.Event.record":       id(torch.npu.Event.record),
        "torch.npu.synchronize":        id(torch.npu.synchronize),
        # Current profiler-based timing path — every measure_perf call
        # goes through these.
        "torch_npu.profiler.profile":                   id(torch_npu.profiler.profile),
        "torch_npu.profiler.schedule":                  id(torch_npu.profiler.schedule),
        "torch_npu.profiler.tensorboard_trace_handler": id(torch_npu.profiler.tensorboard_trace_handler),
        "torch_npu.profiler._ExperimentalConfig":       id(torch_npu.profiler._ExperimentalConfig),
    })

def verify_timing_apis() -> None:
    """Abort the run if any timing-critical callable has been replaced."""
    import torch
    import torch_npu  # noqa — needed by the eval() below
    env = {"torch": torch, "torch_npu": torch_npu}
    changed = []
    for name, orig in _CRITICAL_API_IDS.items():
        obj = eval(name, env)  # resolves torch.npu.*, torch_npu.profiler.*, etc.
        if id(obj) != orig:
            changed.append(name)
    if changed:
        raise RuntimeError(
            "[cann-bench] SECURITY: submission tampered with timing APIs: "
            f"{changed}. Aborting evaluation — results cannot be trusted. "
            "(Disable this check with ALLOW_TIMING_TAMPERING=1 only for debugging.)"
        )


# ---------------------------------------------------------------------------
# Submission loading: install the cann_bench wheel
# ---------------------------------------------------------------------------

def install_submission_wheel(submission_path: Path) -> None:
    """Install the cann_bench wheel from the submission.

    Accepts either:
      - a .whl path
      - a directory containing a dist/*.whl (or any nested *.whl)
    """
    if submission_path.is_file() and submission_path.suffix == ".whl":
        wheel = submission_path
    else:
        whls = list(submission_path.glob("**/*.whl"))
        if not whls:
            raise FileNotFoundError(f"No .whl found under {submission_path}")
        if len(whls) > 1:
            # Prefer dist/ over build/
            whls_dist = [w for w in whls if "dist" in w.parts]
            wheel = whls_dist[0] if whls_dist else whls[0]
        else:
            wheel = whls[0]

    # Uninstall any prior cann_bench variants (aclnn ships as `cann_bench`,
    # direct_launch_simple ships as `cann_bench_ops`, both install the
    # `cann_bench` module — stale files from a previous install would shadow
    # the new wheel's extensions).
    for pkg in ("cann_bench", "cann_bench_ops"):
        subprocess.run([sys.executable, "-m", "pip", "uninstall", pkg, "-y", "-q"],
                       capture_output=True, text=True)

    log.info(f"[cann-bench] Installing submission wheel: {wheel.name}")
    rc = subprocess.run(
        [sys.executable, "-m", "pip", "install", str(wheel),
         "--force-reinstall", "--no-deps", "-q"],
        capture_output=True, text=True,
    )
    if rc.returncode != 0:
        raise RuntimeError(f"Failed to install wheel: {rc.stderr[:400]}")

    # Force reimport of cann_bench if previously cached
    for mod_name in list(sys.modules.keys()):
        if mod_name == "cann_bench" or mod_name.startswith("cann_bench."):
            del sys.modules[mod_name]


def resolve_kernel_fn(cann_bench_mod, op_name: str, schema: str = ""):
    """Look up the user's kernel function in the cann_bench module.

    Tries (in order):
      1. Function name from proto.yaml schema (e.g. "add" from "add(Tensor...)")
      2. snake_case of op_name (e.g. "Add" -> "add")
      3. lowercase op_name
    """
    import re
    tries = []
    if schema:
        m = re.match(r"^\s*(\w+)\s*\(", schema)
        if m:
            tries.append(m.group(1))
    snake = re.sub(r"(?<!^)(?=[A-Z])", "_", op_name).lower()
    tries.extend([snake, op_name.lower(), op_name])
    for name in tries:
        if hasattr(cann_bench_mod, name):
            return getattr(cann_bench_mod, name), name
    available = [n for n in dir(cann_bench_mod) if not n.startswith("_") and callable(getattr(cann_bench_mod, n, None))]
    raise AttributeError(
        f"Kernel for operator '{op_name}' not found in cann_bench. "
        f"Tried: {tries}. Available: {available}"
    )


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


DEFAULT_HARDWARE = "910b2"

def resolve_baseline_us(case_raw: dict, hardware: str) -> float:
    """Pick baseline_perf_us for the selected hardware.

    cases.yaml supports two forms:
      - scalar: `baseline_perf_us: 40.2` → treated as the default hardware (910b2)
      - dict:   `baseline_perf_us: {910b2: 40.2, 910b1: 45.1, ...}`
    Returns 0 if no baseline exists for the requested hardware.
    """
    bp = case_raw.get("baseline_perf_us", 0)
    if bp is None or bp == "None":
        return 0.0
    if isinstance(bp, dict):
        v = bp.get(hardware)
        if v is None or v == "None":
            return 0.0
        try:
            return float(v)
        except (TypeError, ValueError):
            return 0.0
    # Scalar form — only valid when running against the default hardware.
    if hardware != DEFAULT_HARDWARE:
        return 0.0
    try:
        return float(bp)
    except (TypeError, ValueError):
        return 0.0


def load_golden_function(golden_path: Path):
    """Import golden.py and return the golden function."""
    spec = importlib.util.spec_from_file_location("_golden_" + golden_path.parent.name, golden_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for name in dir(mod):
        if name.startswith("_"):
            continue
        obj = getattr(mod, name)
        if callable(obj) and not isinstance(obj, type):
            return obj
    raise RuntimeError(f"No callable function found in {golden_path}")


def discover_operators(bench_root: Path, levels: Optional[List[int]] = None) -> List[Path]:
    """Find all operator dirs containing proto.yaml + golden.py + cases.yaml."""
    level_dirs = []
    if levels is None:
        for p in sorted(bench_root.iterdir()):
            if p.is_dir() and p.name.startswith("level"):
                level_dirs.append(p)
    else:
        for lv in levels:
            p = bench_root / f"level{lv}"
            if p.is_dir():
                level_dirs.append(p)

    op_dirs = []
    for ld in level_dirs:
        for op in sorted(ld.iterdir()):
            if not op.is_dir():
                continue
            if (op / "proto.yaml").exists() and (op / "golden.py").exists() and (op / "cases.yaml").exists():
                op_dirs.append(op)
    return op_dirs


# ---------------------------------------------------------------------------
# Performance measurement
# ---------------------------------------------------------------------------

def measure_perf(fn, device, warmup=3, trials=5) -> float:
    """Return NPU kernel-only time (µs) of a zero-arg callable.

    Uses torch_npu.profiler to isolate on-device kernel duration so
    submissions that use different dispatch layers (aclnn vs. direct launch)
    are compared on kernel time alone, not on dispatch overhead. A MatMul +
    ReduceMax pair before each step boosts NPU frequency and clears L2, then
    those kernels are filtered out of the trace. Defaults of warmup=3 /
    trials=5 match the cann-bench original.
    """
    result = measure_kernel_us(fn, warmup=warmup, repeat=trials)
    if result.error:
        raise RuntimeError(f"profiler measurement failed: {result.error}")
    return result.elapsed_us


# ---------------------------------------------------------------------------
# Per-operator evaluation
# ---------------------------------------------------------------------------

def evaluate_operator(
    operator_dir: Path,
    cann_bench_mod,
    device: str,
    skip_performance: bool,
    data_gen: DataGenerator,
    precision_checker: PrecisionChecker,
    param_builder: ParamBuilder,
    golden_baselines: Dict[Tuple[str, int], float],
    hardware: str = DEFAULT_HARDWARE,
    prefer_measured_baselines: bool = False,
) -> Dict[str, Any]:
    """Evaluate a single operator against its benchmark cases."""
    import torch
    proto = load_yaml(operator_dir / "proto.yaml")["operator"]
    op_name = proto["name"]
    schema = proto.get("schema", "")
    cases_data = load_yaml(operator_dir / "cases.yaml").get("cases", [])
    golden_fn = load_golden_function(operator_dir / "golden.py")

    # Caller filters out operators not present in the submission; if we get
    # here without one, that's a programming error, not an expected state.
    custom_fn, resolved_name = resolve_kernel_fn(cann_bench_mod, op_name, schema)

    log.info(f"Evaluating {op_name} ({len(cases_data)} cases) → cann_bench.{resolved_name}")

    results = []
    passed_count = 0
    speedups = []

    for case_raw in cases_data:
        case_id = case_raw["case_id"]
        case_note = case_raw.get("note", f"case_{case_id}")

        input_shapes = case_raw.get("input_shape", [])
        if isinstance(input_shapes, list) and input_shapes and not isinstance(input_shapes[0], list):
            input_shapes = [input_shapes]
        dtypes = case_raw.get("dtype", [])
        if isinstance(dtypes, str):
            dtypes = [dtypes]

        case_info = CaseInfo(
            level=0, operator=op_name, case_id=case_id,
            input_shapes=input_shapes, dtypes=dtypes,
            attrs=case_raw.get("attrs", {}) or {},
            value_ranges=case_raw.get("value_range", []) or [],
            note=case_note, yaml_path=str(operator_dir / "cases.yaml"),
        )
        dt_str = dtypes[0] if dtypes else "float32"

        try:
            input_tensors = data_gen.generate_input_tensors_from_case(
                case_info.input_shapes, case_info.dtypes, case_info.value_ranges)

            # Golden on CPU fp64 — we want the precision reference to exceed
            # the NPU's native dtype, so over/underflow behaviour of the
            # declared dtype doesn't also corrupt the reference. The
            # precision checker casts back to fp32 for MERE/MARE, so the
            # thresholds stay keyed to the NPU dtype.
            fp64_tensors = [t.cpu().double() if isinstance(t, torch.Tensor) else
                           [sub.cpu().double() for sub in t]
                           for t in input_tensors]
            golden_params = param_builder.build_call_params(golden_fn, case_info, fp64_tensors)
            with torch.no_grad():
                golden_out = golden_fn(**golden_params)

            # Custom on NPU (positional args — cann_bench.op(*tensors))
            npu_tensors = [t.to(device) if isinstance(t, torch.Tensor) else
                          [sub.to(device) for sub in t]
                          for t in input_tensors]
            flat_npu = [t if isinstance(t, torch.Tensor) else t[0] for t in npu_tensors]
            with torch.no_grad():
                custom_out = custom_fn(*flat_npu)

            if isinstance(golden_out, tuple):
                golden_out = golden_out[0]
            if isinstance(custom_out, tuple):
                custom_out = custom_out[0]

            # DEFENSE: strict type check — reject FakeTensor / lazy wrappers
            # that might pass eq-style correctness but never materialise compute.
            if type(custom_out) is not torch.Tensor:
                raise RuntimeError(
                    f"custom op must return torch.Tensor, got {type(custom_out).__name__} "
                    "(possible lazy-evaluation / subclass-spoofing attack)")

            prec = precision_checker.check(golden_out.cpu(), custom_out.cpu(), dt_str)
            ok = prec.passed

            # DEFENSE: second correctness trial with FRESH inputs. A submission
            # that serves a cached first result or flips a 'computed-once' flag
            # after the first call will produce garbage on trial #2.
            if ok:
                fresh_inputs = data_gen.generate_input_tensors_from_case(
                    case_info.input_shapes, case_info.dtypes, case_info.value_ranges,
                )
                # Perturb one input so content differs from trial #1 even if
                # the DataGenerator seed happened to repeat.
                for t in fresh_inputs:
                    if isinstance(t, torch.Tensor) and t.is_floating_point():
                        t.add_(0.01)
                        break
                fresh_fp64 = [t.cpu().double() if isinstance(t, torch.Tensor) else
                              [sub.cpu().double() for sub in t] for t in fresh_inputs]
                fresh_npu = [t.to(device) if isinstance(t, torch.Tensor) else
                            [sub.to(device) for sub in t] for t in fresh_inputs]
                fresh_flat = [t if isinstance(t, torch.Tensor) else t[0] for t in fresh_npu]
                with torch.no_grad():
                    g2 = golden_fn(**param_builder.build_call_params(golden_fn, case_info, fresh_fp64))
                    c2 = custom_fn(*fresh_flat)
                if isinstance(g2, tuple): g2 = g2[0]
                if isinstance(c2, tuple): c2 = c2[0]
                if type(c2) is not torch.Tensor:
                    raise RuntimeError("trial #2: custom op returned non-Tensor")
                prec2 = precision_checker.check(g2.cpu(), c2.cpu(), dt_str)
                if not prec2.passed:
                    ok = False
                    prec = prec2  # surface trial-2 failure in the result detail

            # Performance — baseline sources: the value recorded in
            # cases.yaml, and/or a fresh NPU measurement of golden from
            # pre_measure_baselines. Which takes priority is controlled by
            # prefer_measured_baselines; the other is still surfaced in the
            # output so users can compare (e.g. to calibrate yaml against a
            # new timing mechanism).
            yaml_baseline = resolve_baseline_us(case_raw, hardware)
            measured_baseline = golden_baselines.get((op_name, case_id), 0)
            if prefer_measured_baselines:
                baseline_us = measured_baseline or yaml_baseline
            else:
                baseline_us = yaml_baseline or measured_baseline
            custom_us = None
            speedup = None
            if ok and not skip_performance:
                # DEFENSE: pre-allocate a pool of cloned input sets so each
                # timing iteration passes a different `data_ptr` to the
                # kernel. Cycling through the pool has O(1) per-iter cost
                # (no memcpy in the hot loop). An attacker caching outputs
                # by data_ptr would need an entry per pool slot; when the
                # pool fills the timing window, every iteration is a cache
                # miss so the cache attack degenerates to the real kernel.
                # Profiler-based timing: fewer iterations needed than event-
                # based, since each step's kernel time is measured directly
                # from the trace rather than inferred from elapsed wall time.
                WARMUP, TRIALS = 3, 5
                MAX_CLONE_POOL_MB = 512
                per_set_bytes = sum(
                    t.element_size() * t.numel() for t in flat_npu
                    if isinstance(t, torch.Tensor)
                ) or 1
                pool_cap = max(1, (MAX_CLONE_POOL_MB * 1024 * 1024) // per_set_bytes)
                pool_size = min(WARMUP + TRIALS, pool_cap)
                clone_pool = [[t.clone() for t in flat_npu] for _ in range(pool_size)]
                it = [0]
                def _pooled_call(pool=clone_pool, ctr=it, fn=custom_fn):
                    idx = ctr[0] % len(pool)
                    ctr[0] += 1
                    return fn(*pool[idx])
                custom_us = measure_perf(_pooled_call, device,
                                         warmup=WARMUP, trials=TRIALS)
                del clone_pool  # free pool memory immediately after timing
                if baseline_us > 0 and custom_us > 0:
                    speedup = baseline_us / custom_us
                    speedups.append(speedup)

            if ok:
                passed_count += 1

            results.append({
                "case_id": case_id, "case_name": case_note,
                "status": "PASS" if ok else "FAIL",
                "detail": prec.detail,
                "mere": prec.mere, "mare": prec.mare,
                "speedup": speedup,
                "baseline_perf_us": baseline_us,
                "baseline_yaml_us": yaml_baseline or None,
                "baseline_measured_us": measured_baseline or None,
                "custom_time_us": custom_us,
            })

            spd_str = f" {speedup:.3f}x" if speedup else ""
            log.info(f"  [{case_id:>2}] {'PASS' if ok else 'FAIL':4s}{spd_str}  {prec.detail}  {case_note}")

        except Exception as e:
            results.append({
                "case_id": case_id, "case_name": case_note,
                "status": "FAIL", "detail": str(e)[:200],
                "mere": 0, "mare": 0,
                "speedup": None, "baseline_perf_us": 0, "custom_time_us": None,
            })
            log.info(f"  [{case_id:>2}] FAIL  {case_note}: {e}")

    geo_mean = 0.0
    if speedups:
        geo_mean = math.exp(sum(math.log(max(s, 1e-9)) for s in speedups) / len(speedups))

    log.info(f"  → {passed_count}/{len(cases_data)} passed, geo_mean={geo_mean:.3f}x")

    return {
        "operator": op_name,
        "hardware": hardware,
        "total_cases": len(cases_data),
        "passed_cases": passed_count,
        "geometric_mean_speedup": round(geo_mean, 6),
        "results": results,
    }


# ---------------------------------------------------------------------------
# Baseline pre-measurement (before custom kernel contaminates dispatch)
# ---------------------------------------------------------------------------

def pre_measure_baselines(operator_dirs: List[Path], device: str,
                          data_gen: DataGenerator, param_builder: ParamBuilder,
                          hardware: str = DEFAULT_HARDWARE,
                          measure_all: bool = False,
                          ) -> Dict[Tuple[str, int], float]:
    """Measure golden on NPU BEFORE the custom kernel registers a dispatch.
    Keyed by (op_name, case_id).

    Default: only measure cases whose cases.yaml has no usable
    ``baseline_perf_us`` for the target hardware. Set ``measure_all=True`` to
    re-measure every case — useful for calibrating yaml baselines against a
    new timing mechanism (e.g. after switching from wall-time events to
    profiler-based kernel timing).

    We can only legitimately pre-measure when the NPU we're running on
    matches the target hardware (otherwise we'd be measuring on HW A but
    claiming it's a HW B baseline). For non-default hardware we skip —
    speedup for cases without explicit baseline will just be blank.
    """
    import torch
    baselines: Dict[Tuple[str, int], float] = {}
    if hardware != DEFAULT_HARDWARE:
        log.info(f"  skipping pre-measurement — hardware={hardware} != this runner's {DEFAULT_HARDWARE}")
        return baselines
    for op_dir in operator_dirs:
        try:
            proto = load_yaml(op_dir / "proto.yaml")["operator"]
            op_name = proto["name"]
            cases = load_yaml(op_dir / "cases.yaml").get("cases", [])
            needs = cases if measure_all else [c for c in cases if resolve_baseline_us(c, hardware) <= 0]
            if not needs:
                continue
            golden_fn = load_golden_function(op_dir / "golden.py")
            log.info(f"  pre-measuring golden for {op_name} ({len(needs)} cases)")
            for c in needs:
                cid = c["case_id"]
                input_shapes = c.get("input_shape", [])
                if isinstance(input_shapes, list) and input_shapes and not isinstance(input_shapes[0], list):
                    input_shapes = [input_shapes]
                dtypes = c.get("dtype", [])
                if isinstance(dtypes, str):
                    dtypes = [dtypes]
                case_info = CaseInfo(
                    level=0, operator=op_name, case_id=cid,
                    input_shapes=input_shapes, dtypes=dtypes,
                    attrs=c.get("attrs", {}) or {},
                    value_ranges=c.get("value_range", []) or [],
                    note=c.get("note", ""), yaml_path=str(op_dir / "cases.yaml"),
                )
                try:
                    tensors = data_gen.generate_input_tensors_from_case(
                        case_info.input_shapes, case_info.dtypes, case_info.value_ranges)
                    npu_tensors = [t.to(device) if isinstance(t, torch.Tensor) else
                                  [sub.to(device) for sub in t] for t in tensors]
                    golden_params = param_builder.build_call_params(golden_fn, case_info, npu_tensors)
                    baselines[(op_name, cid)] = measure_perf(
                        lambda: golden_fn(**golden_params), device)
                except Exception as e:
                    log.warning(f"    [{cid}] baseline measurement failed: {e}")
        except Exception as e:
            log.warning(f"  baseline pre-measure skipped {op_dir.name}: {e}")
    return baselines


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="CANN Kernel Bench evaluation")
    parser.add_argument("--submission", required=True,
                        help="Path to submission wheel (.whl) or directory containing one")
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--operator-dir",
                   help="Single operator dir (contains proto.yaml/cases.yaml/golden.py)")
    g.add_argument("--bench-root",
                   help="kernel_bench root dir — evaluate all ops under it (batch mode)")
    parser.add_argument("--levels", default=None,
                        help="Comma-separated level numbers to include (batch mode only, default: all)")
    parser.add_argument("--device-id", type=int, default=3, help="NPU device ID")
    parser.add_argument("--hardware", default=DEFAULT_HARDWARE,
                        help=f"Hardware to compare against for baseline_perf_us "
                             f"(default: {DEFAULT_HARDWARE}). "
                             f"Scalar baseline_perf_us in cases.yaml is treated as {DEFAULT_HARDWARE}; "
                             f"use dict form for multi-hardware baselines.")
    parser.add_argument("--json-output", help="Output JSON path")
    parser.add_argument("--skip-performance", action="store_true", help="Skip performance measurement")
    parser.add_argument("--measure-baselines", action="store_true",
                        help="Re-measure every case's golden on NPU and use that as baseline "
                             "(instead of the cases.yaml value). Useful for calibrating yaml "
                             "baselines against the profiler-based timing. Each case's result "
                             "also records baseline_yaml_us + baseline_measured_us so the two "
                             "can be compared. Only meaningful when this runner's hardware "
                             f"matches --hardware (default {DEFAULT_HARDWARE}).")
    args = parser.parse_args()

    import torch
    import torch_npu  # noqa
    device = f"npu:{args.device_id}"
    torch.npu.set_device(device)

    # Snapshot timing-API identities NOW, before any submission code runs.
    # We verify after wheel install that nothing was monkey-patched.
    snapshot_timing_apis()

    # Decide operator list
    if args.operator_dir:
        operator_dirs = [Path(args.operator_dir).resolve()]
    else:
        bench_root = Path(args.bench_root).resolve()
        levels = None
        if args.levels:
            levels = [int(x.strip()) for x in args.levels.split(",") if x.strip()]
        operator_dirs = discover_operators(bench_root, levels)
        if not operator_dirs:
            log.error(f"No operators found under {bench_root}")
            sys.exit(1)
        log.info(f"[cann-bench] Discovered {len(operator_dirs)} operator(s): "
                 f"{[d.name for d in operator_dirs]}")

    # Setup modules
    data_gen = DataGenerator(seed=42)
    precision_checker = PrecisionChecker()
    param_builder = ParamBuilder()

    # Pre-measure baselines BEFORE installing custom kernel. By default we
    # only measure cases missing a yaml baseline; --measure-baselines
    # re-measures every case so users can calibrate yaml values against the
    # profiler-based timing.
    if args.skip_performance:
        golden_baselines: Dict[Tuple[str, int], float] = {}
    else:
        scope = "all cases" if args.measure_baselines else "cases missing a yaml baseline"
        log.info(f"[cann-bench] Pre-measuring golden baselines for {scope} (before custom dispatch)")
        golden_baselines = pre_measure_baselines(
            operator_dirs, device, data_gen, param_builder,
            hardware=args.hardware, measure_all=args.measure_baselines,
        )

    # Install submission wheel
    install_submission_wheel(Path(args.submission).resolve())
    import cann_bench  # noqa — triggers any monkey-patches in __init__.py
    log.info(f"[cann-bench] Loaded cann_bench from {cann_bench.__file__}")

    # Timing-API integrity check: import above may have patched timing funcs
    if os.environ.get("ALLOW_TIMING_TAMPERING") != "1":
        verify_timing_apis()
        log.info("[cann-bench] Timing-API integrity verified")

    # Filter the bundle's operator list to those the submission actually
    # provides. Operators missing a kernel in cann_bench are dropped
    # silently — the submission defines the scope.
    evaluable = []
    for op_dir in operator_dirs:
        proto = load_yaml(op_dir / "proto.yaml")["operator"]
        op_name = proto["name"]
        schema = proto.get("schema", "")
        try:
            resolve_kernel_fn(cann_bench, op_name, schema)
        except AttributeError as e:
            log.info(f"[cann-bench] submission does not provide {op_name} — dropped ({e})")
            continue
        evaluable.append(op_dir)
    if not evaluable:
        log.error("[cann-bench] submission provides no operators matching the benchmark bundle")
        sys.exit(2)

    # Evaluate each operator
    all_results = []
    log.info(f"[cann-bench] Comparing against baselines for hardware: {args.hardware}")
    for op_dir in evaluable:
        result = evaluate_operator(
            op_dir, cann_bench, device, args.skip_performance,
            data_gen, precision_checker, param_builder, golden_baselines,
            hardware=args.hardware,
            prefer_measured_baselines=args.measure_baselines,
        )
        all_results.append(result)

    total_cases = sum(r["total_cases"] for r in all_results)
    total_passed = sum(r["passed_cases"] for r in all_results)
    all_speedups = [r["geometric_mean_speedup"] for r in all_results
                    if r["geometric_mean_speedup"] > 0]
    overall_geo = 0.0
    if all_speedups:
        overall_geo = math.exp(sum(math.log(max(s, 1e-9)) for s in all_speedups) / len(all_speedups))

    summary = {
        "hardware": args.hardware,
        "total_operators": len(all_results),
        "total_cases": total_cases,
        "total_passed": total_passed,
        "overall_geometric_mean_speedup": round(overall_geo, 6),
        "operators": all_results,
    }

    log.info(f"\n[cann-bench] OVERALL: {total_passed}/{total_cases} cases passed across "
             f"{len(all_results)} operators, geo_mean={overall_geo:.3f}x")

    if args.json_output:
        Path(args.json_output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_output, "w") as f:
            json.dump(summary, f, indent=2)
        log.info(f"[cann-bench] Results written to {args.json_output}")

    return summary


if __name__ == "__main__":
    main()
