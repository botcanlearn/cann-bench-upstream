#!/usr/bin/env python3
"""Shared HAP (Hardware-Anchored Performance, t_hw) model for Ascend 910B2, 950PR and 910C.

t_hw = max(t_cube, t_vector, t_hbm_read, t_write)   [microseconds]

t_hw is a roofline LOWER BOUND on kernel wall time: every engine at peak, perfect
overlap, zero launch overhead -- physically unreachable, useful as a floor.

Model rules baked in (identical in form across all platforms; only the
hardware constants differ):
  * Read path  : input_bytes / HBM_BW                       (cold L2 -> HBM)
  * Write path : L2-fill-first piecewise on L2 capacity     (L2 bus, then HBM spill)
  * Cube       : flops / cube_peak_for_dtype
  * Vector     : element-op-count / vector_rate_for(op_kind, dtype)   (SIMD path)
  * Reduction  : FLAT, charged as N/peak (NO log2 factor) -- log2 is the tree
                 depth (latency), not a throughput cost.
  * Scan/prefix: N * ceil(log2(N_scan)) element-ops (the genuine log2 case).
  * Casts are real vector components; FixP epilogue (writeback cast/scale/bias) is free.
  * Cube & Vector overlap (max); same-unit components are serial (sum).
  * Unique bytes only (no reload factors). L2-capacity check is on OUTPUT bytes only.

Output conventions applied at finalize():
  * t_hw_us          = max(raw_t_hw_us, 1.0)           (1us clip)
  * baseline_cap_us  = max(10.0 * t_hw_us, 10.0)
    This is the CAP on the published baseline, not the published value itself:
    per the _metadata block of tasks/metadata/910b2.json, published
    baseline_perf_us = min(measured, baseline_cap_us); where no measurement
    exists, the cap itself is published.

PUBLIC RELEASE NOTE -- hardware constants are NOT included.
This directory publishes the *logic* of the t_hw computation. The per-platform
hardware constant table (HBM read bandwidth, L2 bus bandwidth and capacity,
cube peak FLOP/s per dtype, vector rates per dtype, SFU/div rate factors) is
loaded from an external `hap_platform_constants.json` that is not distributed.
Obtain the values for your device from the official hardware documentation:

    https://www.hiascend.com/document

then fill in `hap_platform_constants.template.json` (schema below) and save it
as `hap_platform_constants.json` next to this file. Without it every script
still imports and reads cleanly, but raises a clear error on first computation.
"""
import json
import os
from functools import reduce

HERE = os.path.dirname(os.path.abspath(__file__))

# This directory ships as <repo>/docs/hap_thw_model/, so the repository root --
# the directory holding tasks/ (open cases + metadata) and, on the closed side,
# inner/tasks/ -- is two levels up. Every compute_<op>.py resolves its data
# paths from this single constant.
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))

# ---------------------------------------------------------------------------
# Hardware facts -- loaded from an external file, not distributed publicly.
# Schema, per platform key (all three of "910b2" / "950pr" / "910c" required):
#   hbm        : HBM read bandwidth, bytes/s
#   l2         : L2 write-fill (bus) bandwidth, bytes/s
#   l2cap      : L2 capacity, bytes
#   cube       : {dtype: peak FLOP/s}          -- dtypes the cube unit supports
#   vec_basic  : {dtype: basic element-op/s}   -- per-die aggregate SIMD rate;
#                MUST include "float32" (it is the fallback rate for dtypes
#                without an entry of their own, see vec_rate())
#   cast_rate  : cast element-op/s
#   sfu_factor : SFU (exp/log/rsqrt/sqrt) rate as a fraction of basic
#   div_factor : divide rate as a fraction of basic
# See https://www.hiascend.com/document for the hardware specifications.
# ---------------------------------------------------------------------------

_CONST_FILE = os.path.join(HERE, "hap_platform_constants.json")
_CONST_HELP = (
    "fill in hap_platform_constants.template.json from your device's official "
    "hardware documentation (https://www.hiascend.com/document) and save it as "
    "hap_platform_constants.json next to hap_lib.py")

_PLATFORMS = ("910b2", "950pr", "910c")
_SCALAR_KEYS = ("hbm", "l2", "l2cap", "cast_rate", "sfu_factor", "div_factor")
_TABLE_KEYS = ("cube", "vec_basic")


class _MissingConstants(dict):
    """Import-safe placeholder that raises a helpful error on first use."""

    def _fail(self):
        raise RuntimeError(
            "hap_platform_constants.json not found. This public copy ships the "
            "t_hw logic without the per-platform hardware constants; " + _CONST_HELP + ".")

    def __getitem__(self, key):
        self._fail()

    def get(self, key, default=None):
        self._fail()

    def __contains__(self, key):
        self._fail()


def _validate_platform_constants(plat_table):
    """Reject partial/misshapen constant files with a readable error instead of
    letting a half-filled template surface as KeyError/TypeError mid-compute."""
    problems = []
    for plat in _PLATFORMS:
        block = plat_table.get(plat)
        if not isinstance(block, dict):
            problems.append(f"missing platform block {plat!r}")
            continue
        for k in _SCALAR_KEYS:
            if not isinstance(block.get(k), (int, float)):
                problems.append(f"{plat}.{k} must be a number")
        for k in _TABLE_KEYS:
            tbl = block.get(k)
            if not isinstance(tbl, dict) or not tbl or \
                    not all(isinstance(v, (int, float)) for v in tbl.values()):
                problems.append(f"{plat}.{k} must be a non-empty dtype->rate table")
        if isinstance(block.get("vec_basic"), dict) and \
                not isinstance(block["vec_basic"].get("float32"), (int, float)):
            problems.append(f"{plat}.vec_basic must include 'float32' (fallback rate)")
    if problems:
        raise RuntimeError(
            "hap_platform_constants.json is incomplete: "
            + "; ".join(problems) + ". " + _CONST_HELP + ".")


def _load_platform_constants():
    if os.path.exists(_CONST_FILE):
        with open(_CONST_FILE) as f:
            table = json.load(f)
        _validate_platform_constants(table)
        return table
    return _MissingConstants()


PLAT = _load_platform_constants()

DTYPE_BYTES = {
    "float16": 2, "bfloat16": 2, "float32": 4, "float64": 8,
    "int8": 1, "uint8": 1, "int16": 2, "int32": 4, "int64": 8,
    "bool": 1,
    "float8_e4m3fn": 1, "float8_e5m2": 1, "hif8": 1,
    "mxfp4": 0.5,
}


def prod(xs):
    return reduce(lambda a, b: a * b, xs, 1)


def dbytes(dt):
    return DTYPE_BYTES[dt]


def inner_cases_available(innerdir):
    """The closed (inner) case sets under inner/tasks/ are not distributed
    publicly. Scripts call this after open-case validation and exit cleanly
    when the closed cases are absent."""
    return os.path.isdir(innerdir)


# ---------------------------------------------------------------------------
# Per-component times (return microseconds)
# ---------------------------------------------------------------------------
def t_read_us(input_bytes, plat):
    return input_bytes / PLAT[plat]["hbm"] * 1e6


def t_write_us(output_bytes, plat):
    p = PLAT[plat]
    cap = p["l2cap"]
    if output_bytes <= cap:
        return output_bytes / p["l2"] * 1e6
    return (cap / p["l2"] + (output_bytes - cap) / p["hbm"]) * 1e6


def t_cube_us(flops, dtype, plat):
    """flops = total multiply-add FLOPs (i.e. 2*M*N*K for a matmul)."""
    if flops <= 0:
        return 0.0
    peak = PLAT[plat]["cube"].get(dtype)
    if peak is None:
        raise ValueError(f"cube dtype {dtype!r} unsupported on {plat}")
    return flops / peak * 1e6


def cube_dtype_hf32(dtype, plat):
    """Map a float32 cube operand to the HF32/TF32 peak on platforms that run it there.

    msprof reports `HF32 Eligible = YES` for convolution kernels on platforms whose
    cube unit executes fp32 inputs on a reduced-precision HF32/TF32 path; charging
    the (much slower) true-fp32 peak there produces a "floor" that real measured
    kernels then break. Only use this for ops whose captured profile confirms HF32
    eligibility. Platforms without a "tf32" cube entry fall through to the caller's
    dtype unchanged.
    """
    if dtype == "float32" and "tf32" in PLAT[plat]["cube"]:
        return "tf32"
    return dtype


def vec_rate(dtype, plat, kind="basic"):
    p = PLAT[plat]
    basic = p["vec_basic"].get(dtype, p["vec_basic"]["float32"])
    if kind == "basic":
        return basic
    if kind == "cast":
        return p["cast_rate"]
    if kind == "sfu":
        return basic * p["sfu_factor"]
    if kind == "div":
        return basic * p["div_factor"]
    raise ValueError(kind)


def t_vec_us(n_ops, dtype, plat, kind="basic"):
    """n_ops = number of element-wise operations of this kind (already summed
    across the elements they touch). For a reduction over N elements charge
    n_ops=N (flat). For a scan over N_scan charge n_ops = N*ceil(log2(N_scan))."""
    if n_ops <= 0:
        return 0.0
    return n_ops / vec_rate(dtype, plat, kind) * 1e6


# ---------------------------------------------------------------------------
# Aggregation + finalize
# ---------------------------------------------------------------------------
def aggregate(t_cube=0.0, t_vector=0.0, t_read=0.0, t_write=0.0):
    """Bottleneck = the largest component. Exact ties resolve by component
    order (cube, vector, hbm_read, write) -- max() returns the first maximal
    key. In particular a degenerate case with zero work in every component is
    labeled "cube" here; finalize() clips such cases to the 1us floor, so the
    label is a fixed convention, not a physical statement (a no-compute op may
    relabel it, see compute_strided_slice.py)."""
    comps = {"cube": t_cube, "vector": t_vector, "hbm_read": t_read, "write": t_write}
    bottleneck = max(comps, key=comps.get)
    return comps[bottleneck], bottleneck, comps


def finalize(raw_t_hw_us):
    """Apply the output conventions to a raw roofline time.

    Returns (t_hw_us, baseline_cap_us):
      t_hw_us         = max(raw, 1.0)             -- 1us clip
      baseline_cap_us = max(10*t_hw_us, 10.0)     -- CAP on the published baseline

    baseline_cap_us is deliberately named as a cap: the published
    baseline_perf_us equals min(measured, baseline_cap_us) per the _metadata
    block of tasks/metadata/910b2.json. Only where no measurement exists is the
    cap itself published. t_hw alone cannot reproduce measured baselines.
    """
    t_hw = max(raw_t_hw_us, 1.0)
    baseline_cap = max(10.0 * t_hw, 10.0)
    return round(t_hw, 4), round(baseline_cap, 2)


def confidence_950pr(bottleneck, output_bytes):
    # Unconfirmed only if write-bound with L2-resident output (the L2 bus
    # bandwidth constant is unconfirmed on this platform).
    if bottleneck == "write" and output_bytes <= PLAT["950pr"]["l2cap"]:
        return "Unconfirmed"
    return "Confirmed"


def confidence_910c(bottleneck, output_bytes):
    # Same rule as 950pr: Unconfirmed only if write-bound with L2-resident
    # output (the 910c L2 bus bandwidth constant is likewise unconfirmed; the
    # HBM constant is derived-from-measurement and treated as the working
    # ceiling -> read/compute/HBM-spill-write results are all Confirmed).
    if bottleneck == "write" and output_bytes <= PLAT["910c"]["l2cap"]:
        return "Unconfirmed"
    return "Confirmed"


def compute_both(vec_ops=None, input_bytes=0.0, output_bytes=0.0,
                 cube_dtype=None, cube_flops=0.0, cube_hf32=False):
    """Convenience: compute (t_hw, bottleneck) for all platforms given
    platform-independent work descriptors.

    vec_ops : list of (n_ops, dtype, kind) tuples summed onto t_vector.
    cube_hf32 : set True for ops whose msprof capture shows `HF32 Eligible = YES`, so a
        float32 cube operand is costed at the platform's HF32/TF32 peak where one
        exists. See cube_dtype_hf32().
    """
    vec_ops = vec_ops or []
    res = {}
    for plat in _PLATFORMS:
        cdt = cube_dtype_hf32(cube_dtype, plat) if (cube_hf32 and cube_dtype) else cube_dtype
        tc = t_cube_us(cube_flops, cdt, plat) if cube_flops and cdt else 0.0
        tv = sum(t_vec_us(n, dt, plat, k) for (n, dt, k) in vec_ops)
        tr = t_read_us(input_bytes, plat)
        tw = t_write_us(output_bytes, plat)
        raw, bn, comps = aggregate(tc, tv, tr, tw)
        t_hw, cap = finalize(raw)
        entry = {"t_hw_us": t_hw, "baseline_cap_us": cap, "bottleneck": bn}
        if plat == "950pr":
            entry["confidence"] = confidence_950pr(bn, output_bytes)
        elif plat == "910c":
            entry["confidence"] = confidence_910c(bn, output_bytes)
        res[plat] = entry
    return res
