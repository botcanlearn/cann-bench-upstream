#!/usr/bin/env python3
# coding=utf-8
"""
ScatterPaKvCache 算子性能采集脚本

基于 cann-bench 性能采集规则:
1. 使用 torch_npu.profiler 硬件级计时（非墙钟）
2. 测量前执行 MatMul 升频 + ReduceMax 清 L2 cache
3. profiler schedule: warmup=3, active=5
4. 从 kernel_details.csv 解析，按 Step 分组 → 每 kernel 跨步取中位数 → 累加
5. 自动过滤升频/清cache 的 warmup kernel

运行方式:
    python3 test_baseline_perf.py 1          # 运行 case 1
    python3 test_baseline_perf.py all         # 运行全部 cases
    python3 test_baseline_perf.py all --csv   # 输出 CSV 格式结果
"""

import os
import sys
import csv
import json
import shutil
import tempfile
import logging
import random
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import torch

_logger = logging.getLogger(__name__)

# =============================================================================
# 参数配置
# =============================================================================
DEVICE_ID = int(os.environ.get("DEVICE_ID", "0"))
PROFILER_WARMUP = 3       # profiler schedule warmup steps
PROFILER_REPEAT = 5       # profiler schedule active steps
SEED = 42

# cases.csv 路径（按需修改）
CASES_CSV = os.path.join(os.path.dirname(__file__), "cases.csv")

# 升频矩阵尺寸（与 cann-bench PerfEvaluator 一致）
FREQ_BOOST_DIM = 10240
CACHE_CLEAR_SHAPE = (96, 1024, 1024)

# warmup kernel 的 Input Shapes（用于从 CSV 中过滤）
WARMUP_KERNEL_SHAPES = {
    "10240,10240;10240,10240",   # MatMul 升频
    "96,1024,1024;3",            # ReduceMax 清 cache
}


# =============================================================================
# dtype 映射
# =============================================================================
DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "int8": torch.int8,
    "int32": torch.int32,
    "int64": torch.int64,
}


# =============================================================================
# 用例加载
# =============================================================================
def load_cases(csv_path: str) -> Dict[int, dict]:
    """从 cases.csv 加载测试用例。"""
    cases = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            case_id = int(row["case_id"])
            shapes = json.loads(row["input_shape"])
            dtypes = json.loads(row["dtype"])
            attrs = json.loads(row["attrs"])
            value_range = json.loads(row["value_range"])
            cases[case_id] = {
                "shape_key": shapes[0],          # [numTokens, numHeads, kHeadSize]
                "shape_key_cache": shapes[1],    # [numBlocks, NH*kHS/lastDim, blockSize, lastDim]
                "shape_slot_mapping": shapes[2], # [numTokens]
                "shape_value": shapes[3],        # [numTokens, numHeads, vHeadSize]
                "shape_value_cache": shapes[4],  # [numBlocks, NH*vHS/lastDim, blockSize, lastDim]
                "dtype_key": DTYPE_MAP[dtypes[0]],
                "dtype_key_cache": DTYPE_MAP[dtypes[1]],
                "dtype_slot_mapping": DTYPE_MAP[dtypes[2]],
                "dtype_value": DTYPE_MAP[dtypes[3]],
                "dtype_value_cache": DTYPE_MAP[dtypes[4]],
                "cache_mode": attrs.get("cache_mode", "PA_NZ"),
                "value_range": value_range,
                "note": row.get("note", ""),
            }
    return cases


def build_inputs(case_cfg: dict, device: str):
    """根据 case 配置构造输入张量。"""
    # 固定随机种子保证可复现
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    def _make_tensor(shape, dtype, value_range):
        low, high = value_range
        if dtype in (torch.int8, torch.int32, torch.int64):
            # 整数类型：在范围内随机生成
            t = torch.randint(low, high + 1, shape, dtype=torch.int64)
            return t.to(dtype)
        else:
            # 浮点类型：均匀分布
            t = torch.rand(shape, dtype=torch.float32) * (high - low) + low
            return t.to(dtype)

    key = _make_tensor(case_cfg["shape_key"], case_cfg["dtype_key"], case_cfg["value_range"][0])
    key_cache = _make_tensor(case_cfg["shape_key_cache"], case_cfg["dtype_key_cache"], case_cfg["value_range"][1])
    slot_mapping = _make_tensor(case_cfg["shape_slot_mapping"], case_cfg["dtype_slot_mapping"], case_cfg["value_range"][2])
    value = _make_tensor(case_cfg["shape_value"], case_cfg["dtype_value"], case_cfg["value_range"][3])
    value_cache = _make_tensor(case_cfg["shape_value_cache"], case_cfg["dtype_value_cache"], case_cfg["value_range"][4])

    return key.to(device), key_cache.to(device), slot_mapping.to(device), value.to(device), value_cache.to(device)


# =============================================================================
# NPU 升频 + L2 cache 清空（与 cann-bench PerfEvaluator 一致）
# =============================================================================
def boost_freq_and_clear_cache(device: str):
    """测量窗口前执行一次: MatMul 升频 + ReduceMax 清 L2 cache。"""
    mm1 = torch.rand((FREQ_BOOST_DIM, FREQ_BOOST_DIM), dtype=torch.float16).to(device)
    mm2 = torch.rand((FREQ_BOOST_DIM, FREQ_BOOST_DIM), dtype=torch.float16).to(device)
    reduce_input = torch.rand(CACHE_CLEAR_SHAPE, dtype=torch.float16).to(device)

    torch.matmul(mm1, mm2)
    torch.npu.synchronize()
    torch.max(reduce_input)
    torch.npu.synchronize()

    del mm1, mm2, reduce_input


def clear_cache(device: str):
    """每个 active step 前清空 L2 cache。"""
    reduce_input = torch.rand(CACHE_CLEAR_SHAPE, dtype=torch.float16).to(device)
    torch.max(reduce_input)
    torch.npu.synchronize()
    del reduce_input


# =============================================================================
# Profiler 采集
# =============================================================================
def run_profiled(fn, prof_dir: str, device: str,
                 warmup: int = PROFILER_WARMUP,
                 repeat: int = PROFILER_REPEAT) -> str:
    """使用 torch_npu.profiler 采集性能数据。

    Args:
        fn: 待测算子的 callable
        prof_dir: profiler 输出目录
        device: NPU 设备字符串
        warmup: profiler warmup steps
        repeat: profiler active steps

    Returns:
        prof_dir: profiler 输出目录路径
    """
    import torch_npu

    # 抑制 profiler 日志
    os.environ['ASCEND_SLOG_PRINT_TO_STDOUT'] = '0'
    os.environ['ASCEND_GLOBAL_LOG_LEVEL'] = '3'
    for name in ['', 'torch', 'torch_npu', 'torch_npu.profiler', 'ascend', 'profiler']:
        lg = logging.getLogger(name)
        lg.setLevel(logging.ERROR)
        lg.handlers = []
        lg.addHandler(logging.NullHandler())

    experimental_config = torch_npu.profiler._ExperimentalConfig(
        export_type=[torch_npu.profiler.ExportType.Text],
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone,
    )

    # 预检: 确认算子可运行
    fn()

    # 升频 + 清 cache
    boost_freq_and_clear_cache(device)

    # 重定向 stdout/stderr 抑制 profiler 输出
    saved_stdout_fd = os.dup(1)
    saved_stderr_fd = os.dup(2)
    sink_file = tempfile.NamedTemporaryFile(
        mode='w+', prefix='perf_profiler_', suffix='.log', delete=False
    )
    sink_fd = sink_file.fileno()

    try:
        os.dup2(sink_fd, 1)
        os.dup2(sink_fd, 2)

        with torch_npu.profiler.profile(
            activities=[
                torch_npu.profiler.ProfilerActivity.CPU,
                torch_npu.profiler.ProfilerActivity.NPU,
            ],
            schedule=torch_npu.profiler.schedule(
                wait=0, warmup=warmup, active=repeat, repeat=1
            ),
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(prof_dir),
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
            experimental_config=experimental_config,
        ) as prof:
            for i in range(warmup + repeat):
                if i >= warmup:
                    clear_cache(device)
                fn()
                prof.step()

        # 等待 profiler 异步解析完成
        try:
            from torch_npu.profiler.analysis.prof_common_func._multi_process_pool import MultiProcessPool
            pool = MultiProcessPool()
            pool.close_pool(wait=True)
        except Exception:
            pass

    finally:
        os.dup2(saved_stdout_fd, 1)
        os.dup2(saved_stderr_fd, 2)
        os.close(saved_stdout_fd)
        os.close(saved_stderr_fd)
        sink_file.close()
        try:
            os.unlink(sink_file.name)
        except OSError:
            pass

    return prof_dir


# =============================================================================
# kernel_details.csv 解析（与 cann-bench KernelDetailsStrategy 一致）
# =============================================================================
def locate_kernel_details_csv(prof_dir: str) -> Optional[str]:
    """三层搜索 kernel_details.csv。"""
    # 1) 直接
    direct = os.path.join(prof_dir, "kernel_details.csv")
    if os.path.isfile(direct):
        return direct

    # 2) 一层子目录
    try:
        for entry in os.listdir(prof_dir):
            candidate = os.path.join(prof_dir, entry, "kernel_details.csv")
            if os.path.isfile(candidate):
                return candidate
    except OSError:
        pass

    # 3) 递归搜索
    for root, dirs, files in os.walk(prof_dir):
        if "kernel_details.csv" in files:
            return os.path.join(root, "kernel_details.csv")

    return None


def _median(values: List[float]) -> float:
    """计算中位数。"""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    if n % 2 == 1:
        return sorted_vals[n // 2]
    return (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2


def parse_kernel_details(csv_path: str) -> Dict[str, Any]:
    """解析 kernel_details.csv，返回性能指标。

    流程（与 cann-bench PerfMetricStrategy 一致）:
    1. 读取 Step Id / Duration(us) / Type / Input Shapes / Name
    2. 过滤 warmup kernel（按 Input Shapes 精确匹配）
    3. 按 Step Id 分组 → 每步内同名 kernel Duration 求和
    4. 每 kernel 跨步取中位数
    5. 累加得 total_kernel_us

    Returns:
        {
            "total_kernel_us": float,          # 所有 kernel 中位数之和
            "device_kernels": {name: median},  # 每个 kernel 的中位数耗时
            "step_count": int,                 # 有效 step 数量
        }
    """
    step_kernel_times: Dict[str, Dict[str, List[float]]] = {}

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            step_id = row.get("Step Id", "").strip()
            duration_str = row.get("Duration(us)", "").strip()
            input_shapes = row.get("Input Shapes", "").strip()
            name = row.get("Name", "").strip()

            if not step_id or not duration_str:
                continue

            try:
                duration = float(duration_str)
            except ValueError:
                continue

            if duration <= 0:
                continue

            # 过滤升频/清cache 的 warmup kernel
            if input_shapes in WARMUP_KERNEL_SHAPES:
                continue

            if step_id not in step_kernel_times:
                step_kernel_times[step_id] = {}
            if name not in step_kernel_times[step_id]:
                step_kernel_times[step_id][name] = []
            step_kernel_times[step_id][name].append(duration)

    # 跨步聚合: 每步内同名 kernel 求和
    all_kernel_times: Dict[str, List[float]] = {}
    for step_id, kernels in step_kernel_times.items():
        for name, durations in kernels.items():
            step_sum = sum(durations)
            if name not in all_kernel_times:
                all_kernel_times[name] = []
            all_kernel_times[name].append(step_sum)

    # 每 kernel 取中位数
    device_kernels = {}
    for name, times in all_kernel_times.items():
        device_kernels[name] = round(_median(times), 2)

    total_kernel_us = round(sum(device_kernels.values()), 2)

    return {
        "total_kernel_us": total_kernel_us,
        "device_kernels": device_kernels,
        "step_count": len(step_kernel_times),
    }


# =============================================================================
# 单 case 评测
# =============================================================================
def run_single_case(case_id: int, case_cfg: dict, device: str) -> Optional[Dict[str, Any]]:
    """运行单个 case 的性能评测。

    Returns:
        {
            "case_id": int,
            "key_shape": list,
            "key_cache_shape": list,
            "slot_mapping_shape": list,
            "value_shape": list,
            "value_cache_shape": list,
            "dtype": str,
            "cache_mode": str,
            "elapsed_us": float,       # 核心指标: kernel 中位数耗时之和
            "device_kernels": dict,     # 每 kernel 中位数
            "step_count": int,
            "note": str,
        }
        或 None（评测失败时）
    """
    import torch_npu

    key, key_cache, slot_mapping, value, value_cache = build_inputs(case_cfg, device)

    print(f"\n{'='*60}")
    print(f"[CASE {case_id}] key={case_cfg['shape_key']}, key_cache={case_cfg['shape_key_cache']}, "
          f"slot_mapping={case_cfg['shape_slot_mapping']}")
    print(f"           value={case_cfg['shape_value']}, value_cache={case_cfg['shape_value_cache']}")
    print(f"           dtype={case_cfg['dtype_key']}, cache_mode={case_cfg['cache_mode']}")
    print(f"           note: {case_cfg['note']}")
    print(f"{'='*60}")

    def _run_op():
        torch_npu.npu_scatter_pa_kv_cache(
            key, value, key_cache, value_cache, slot_mapping,
            cache_mode=case_cfg["cache_mode"],
        )

    # 创建临时 profiling 目录
    prof_dir = tempfile.mkdtemp(prefix=f"perf_case{case_id}_")

    try:
        # profiler 采集
        run_profiled(_run_op, prof_dir, device)

        # 定位并解析 kernel_details.csv
        csv_path = locate_kernel_details_csv(prof_dir)
        if csv_path is None:
            print(f"[ERROR] Case {case_id}: kernel_details.csv 未找到")
            return None

        metrics = parse_kernel_details(csv_path)
        elapsed_us = metrics["total_kernel_us"]

        print(f"[RESULT] Case {case_id}: elapsed_us={elapsed_us:.2f} us "
              f"(kernel median sum, {metrics['step_count']} steps)")

        # 打印 top-5 kernel 明细
        sorted_kernels = sorted(metrics["device_kernels"].items(),
                                key=lambda x: x[1], reverse=True)
        for i, (name, us) in enumerate(sorted_kernels[:5]):
            pct = us / elapsed_us * 100 if elapsed_us > 0 else 0
            print(f"         [{i+1}] {name}: {us:.2f} us ({pct:.1f}%)")

        return {
            "case_id": case_id,
            "key_shape": case_cfg["shape_key"],
            "key_cache_shape": case_cfg["shape_key_cache"],
            "slot_mapping_shape": case_cfg["shape_slot_mapping"],
            "value_shape": case_cfg["shape_value"],
            "value_cache_shape": case_cfg["shape_value_cache"],
            "dtype": str(case_cfg["dtype_key"]),
            "cache_mode": case_cfg["cache_mode"],
            "elapsed_us": elapsed_us,
            "device_kernels": metrics["device_kernels"],
            "step_count": metrics["step_count"],
            "note": case_cfg.get("note", ""),
        }

    except Exception as e:
        print(f"[ERROR] Case {case_id} 评测失败: {e}")
        return None

    finally:
        shutil.rmtree(prof_dir, ignore_errors=True)


# =============================================================================
# 主函数
# =============================================================================
def main():
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <case_id | all> [--csv]")
        print(f"       case_id: cases.csv 中的 case 编号")
        print(f"       all: 运行全部 cases")
        print(f"       --csv: 输出 CSV 格式结果到 stdout")
        sys.exit(1)

    arg = sys.argv[1]
    output_csv = "--csv" in sys.argv

    if not os.path.exists(CASES_CSV):
        print(f"[ERROR] cases.csv 不存在: {CASES_CSV}")
        sys.exit(1)
    cases = load_cases(CASES_CSV)

    torch.npu.set_device(DEVICE_ID)
    device = f"npu:{DEVICE_ID}"
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Profiler: warmup={PROFILER_WARMUP}, repeat={PROFILER_REPEAT}")
    print(f"[INFO] 采集方式: torch_npu.profiler → kernel_details.csv → 中位数统计")

    if arg.lower() == "all":
        case_ids = sorted(cases.keys())
    else:
        try:
            case_ids = [int(arg)]
        except ValueError:
            print(f"[ERROR] 无效的 case_id: {arg}")
            sys.exit(1)

    results = []
    for cid in case_ids:
        if cid not in cases:
            print(f"[WARN] Case {cid} 不存在，跳过。")
            continue
        result = run_single_case(cid, cases[cid], device)
        if result:
            results.append(result)

    # 汇总
    print(f"\n{'='*60}")
    print(f"[SUMMARY] 成功: {len(results)} / {len(case_ids)}")
    print(f"{'='*60}")

    if results:
        print(f"\n{'case_id':>8} {'dtype':>10} {'elapsed_us':>12} {'note'}")
        print("-" * 72)
        for r in results:
            print(f"{r['case_id']:>8} {r['dtype']:>10} {r['elapsed_us']:>12.2f} {r['note']}")

    if output_csv and results:
        print("\n--- CSV OUTPUT ---")
        print("case_id,key_shape,key_cache_shape,slot_mapping_shape,value_shape,value_cache_shape,dtype,cache_mode,elapsed_us,step_count,note")
        for r in results:
            print(f"{r['case_id']},"
                  f"{r['key_shape']},{r['key_cache_shape']},{r['slot_mapping_shape']},"
                  f"{r['value_shape']},{r['value_cache_shape']},"
                  f"{r['dtype']},{r['cache_mode']},{r['elapsed_us']:.2f},"
                  f"{r['step_count']},{r['note']}")

    # 保存 JSON 结果
    if results:
        output_json = os.path.join(os.path.dirname(__file__), "perf_results.json")
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        print(f"\n[INFO] 结果已保存到: {output_json}")


if __name__ == "__main__":
    main()
