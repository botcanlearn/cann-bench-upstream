#!/usr/bin/env python3
"""
npu_dynamic_mx_quant 算子性能测试脚本

基于 cases.csv 中的 shape 配置，支持 warmup + trials 模式运行。
运行方式:
    # 运行单个 case (例如 case 1)
    python3 test_baseline_perf.py 1

    # 运行所有 cases
    python3 test_baseline_perf.py all

    # 用 msprof 采集性能
    msprof --application="python3 test_baseline_perf.py 1" --output=./prof_out
"""

import ast
import os
import sys
import csv
import json
import time
import random
import numpy as np
import torch
import torch_npu

# =============================================================================
# 参数配置
# =============================================================================
DEVICE_ID = int(os.environ.get("DEVICE_ID", "0"))
NUM_WARMUP = int(os.environ.get("NUM_WARMUP", "10"))
NUM_TRIALS = int(os.environ.get("NUM_TRIALS", "50"))
SEED = 42

# cases.csv 路径（相对于脚本所在目录）
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CASES_CSV = os.path.join(SCRIPT_DIR, "cases.csv")


# =============================================================================
# dtype 映射
# =============================================================================
DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

# dst_type 到 torch_npu dtype 的映射
DST_TYPE_NPU_MAP = {
    35: torch.float8_e5m2,
    36: torch.float8_e4m3fn,
    40: torch_npu.float4_e2m1fn_x2,
    41: torch_npu.float4_e1m2fn_x2,
}


ROUND_MODES = {"rint", "floor", "round"}


def _is_int(x):
    return isinstance(x, int) and not isinstance(x, bool)


def _is_number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _literal(field, raw):
    try:
        return ast.literal_eval(raw)
    except (ValueError, SyntaxError) as e:
        raise ValueError(f"字段 {field} 不是合法字面量: {raw!r} ({e})")


def _parse_input_shape(raw):
    shapes = _literal("input_shape", raw)
    if not isinstance(shapes, list) or len(shapes) != 1:
        raise ValueError(f"input_shape 必须形如 [[M,N,...]]，实际: {raw!r}")
    shape = shapes[0]
    if not isinstance(shape, list) or not shape:
        raise ValueError(f"input_shape[0] 必须是非空列表，实际: {shape!r}")
    if not all(_is_int(d) and d > 0 for d in shape):
        raise ValueError(f"input_shape 维度必须为正整数，实际: {shape!r}")
    return shapes


def _parse_dtype(raw):
    dtypes = _literal("dtype", raw)
    if not isinstance(dtypes, list) or not dtypes:
        raise ValueError(f"dtype 必须是非空列表，实际: {raw!r}")
    for d in dtypes:
        if not isinstance(d, str) or d not in DTYPE_MAP:
            raise ValueError(f"dtype 元素必须是 {sorted(DTYPE_MAP)} 之一，实际: {d!r}")
    return dtypes


def _parse_attrs(raw):
    attrs = _literal("attrs", raw)
    if not isinstance(attrs, dict):
        raise ValueError(f"attrs 必须是 dict，实际: {raw!r}")
    for key in ("axis", "round_mode", "dst_type", "blocksize", "scale_alg", "dst_type_max"):
        if key not in attrs:
            raise ValueError(f"attrs 缺少必填字段 {key}")
    if not _is_int(attrs["axis"]):
        raise ValueError(f"attrs[axis] 必须为 int，实际: {attrs['axis']!r}")
    if not isinstance(attrs["round_mode"], str) or attrs["round_mode"] not in ROUND_MODES:
        raise ValueError(f"round_mode 必须是 {sorted(ROUND_MODES)} 之一，实际: {attrs['round_mode']!r}")
    if not _is_int(attrs["dst_type"]) or attrs["dst_type"] not in DST_TYPE_NPU_MAP:
        raise ValueError(f"dst_type 必须是 {sorted(DST_TYPE_NPU_MAP)} 之一，实际: {attrs['dst_type']!r}")
    if not _is_int(attrs["blocksize"]) or attrs["blocksize"] <= 0:
        raise ValueError(f"blocksize 必须为正整数，实际: {attrs['blocksize']!r}")
    if not _is_int(attrs["scale_alg"]) or attrs["scale_alg"] < 0:
        raise ValueError(f"scale_alg 必须为非负整数，实际: {attrs['scale_alg']!r}")
    if not _is_number(attrs["dst_type_max"]):
        raise ValueError(f"dst_type_max 必须为数值，实际: {attrs['dst_type_max']!r}")
    return attrs


def _parse_value_range(raw):
    vr = _literal("value_range", raw)
    if not isinstance(vr, list) or len(vr) != 2:
        raise ValueError(f"value_range 必须形如 [vmin, vmax]，实际: {raw!r}")
    if not all(_is_number(v) for v in vr):
        raise ValueError(f"value_range 元素必须为数值，实际: {vr!r}")
    return vr


def load_cases(csv_path):
    """从 cases.csv 加载测试用例。"""
    cases = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            case_id = int(row["case_id"])
            shapes = _parse_input_shape(row["input_shape"])
            dtypes = _parse_dtype(row["dtype"])
            attrs = _parse_attrs(row["attrs"])
            value_range = _parse_value_range(row["value_range"])
            cases[case_id] = {
                "input_shape": shapes[0],
                "dtype": DTYPE_MAP[dtypes[0]],
                "axis": attrs["axis"],
                "round_mode": attrs["round_mode"],
                "dst_type": attrs["dst_type"],
                "blocksize": attrs["blocksize"],
                "scale_alg": attrs["scale_alg"],
                "dst_type_max": float(attrs["dst_type_max"]),
                "value_range": value_range,
                "note": row.get("note", ""),
            }
    return cases


def build_input(case_cfg, device):
    """根据 case 配置构造输入张量。"""
    shape = case_cfg["input_shape"]
    dtype = case_cfg["dtype"]
    vmin, vmax = case_cfg["value_range"]

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    if vmin == 0 and vmax == 0:
        x = torch.zeros(shape, dtype=dtype)
    else:
        x = torch.rand(shape, dtype=torch.float32) * (vmax - vmin) + vmin
        x = x.to(dtype)

    return x.to(device)


def run_single_case(case_id, case_cfg, device):
    """运行单个 case 的 warmup + trials。"""
    x = build_input(case_cfg, device)
    npu_dtype = DST_TYPE_NPU_MAP[case_cfg["dst_type"]]

    print(f"\n{'='*60}")
    print(f"[CASE {case_id}] shape={case_cfg['input_shape']}, "
          f"dtype={case_cfg['dtype']}, dst_type={case_cfg['dst_type']}")
    print(f"           axis={case_cfg['axis']}, round_mode={case_cfg['round_mode']}, "
          f"blocksize={case_cfg['blocksize']}, scale_alg={case_cfg['scale_alg']}, "
          f"dst_type_max={case_cfg['dst_type_max']}")
    print(f"           note: {case_cfg['note']}")
    print(f"{'='*60}")

    def _run_op():
        return torch_npu.npu_dynamic_mx_quant(
            x,
            axis=case_cfg["axis"],
            round_mode=case_cfg["round_mode"],
            dst_type=npu_dtype,
            block_size=case_cfg["blocksize"],
            scale_alg=case_cfg["scale_alg"],
            dst_type_max=case_cfg["dst_type_max"],
        )

    # 预执行一次，确认算子可运行
    try:
        with torch.no_grad():
            _ = _run_op()
        torch.npu.synchronize()
    except Exception as e:
        print(f"[ERROR] Case {case_id} 首次执行失败: {e}")
        return False

    # Case boundary marker (visible in profiling)
    marker = torch.add(torch.zeros(1, device=device), float(case_id))
    torch.npu.synchronize()
    print(f"[MARKER] Begin case {case_id} marker_value={marker.item():.1f}")

    # Warmup
    print(f"[INFO] Warmup {NUM_WARMUP} iterations...")
    for _ in range(NUM_WARMUP):
        with torch.no_grad():
            _ = _run_op()
        torch.npu.synchronize()
    print(f"[INFO] Warmup finished.")

    # Trials
    print(f"[INFO] Running {NUM_TRIALS} trials...")
    timings = []
    for _ in range(NUM_TRIALS):
        torch.npu.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = _run_op()
        torch.npu.synchronize()
        t1 = time.perf_counter()
        timings.append((t1 - t0) * 1e6)  # us

    avg_us = sum(timings) / len(timings)
    min_us = min(timings)
    max_us = max(timings)
    p50_us = sorted(timings)[len(timings) // 2]

    print(f"[RESULT] Case {case_id}: avg={avg_us:.2f} us, "
          f"min={min_us:.2f} us, max={max_us:.2f} us, p50={p50_us:.2f} us")

    # Case end marker (visible in profiling)
    marker_end = torch.add(torch.zeros(1, device=device), float(case_id) + 0.5)
    torch.npu.synchronize()
    print(f"[MARKER] End case {case_id} marker_value={marker_end.item():.1f}")

    return True


def main():
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <case_id | all>")
        print(f"       case_id: 1~20 的整数")
        print(f"       all: 运行全部 cases")
        sys.exit(1)

    arg = sys.argv[1]

    # 加载 cases
    if not os.path.exists(CASES_CSV):
        print(f"[ERROR] cases.csv 不存在: {CASES_CSV}")
        sys.exit(1)
    cases = load_cases(CASES_CSV)

    # 设置设备
    torch.npu.set_device(DEVICE_ID)
    device = f"npu:{DEVICE_ID}"
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Warmup: {NUM_WARMUP}, Trials: {NUM_TRIALS}")

    if arg.lower() == "all":
        case_ids = sorted(cases.keys())
    else:
        try:
            case_ids = [int(arg)]
        except ValueError:
            print(f"[ERROR] 无效的 case_id: {arg}")
            sys.exit(1)

    success_count = 0
    for cid in case_ids:
        if cid not in cases:
            print(f"[WARN] Case {cid} 不存在，跳过。")
            continue
        ok = run_single_case(cid, cases[cid], device)
        if ok:
            success_count += 1

    print(f"\n{'='*60}")
    print(f"[SUMMARY] 成功: {success_count} / {len(case_ids)}")
    print(f"[INFO] 如需 msprof 采集，请执行:")
    print(f'    msprof --application="python3 {__file__} {arg}" --output=./prof_out')
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
