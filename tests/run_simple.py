#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""
测试执行脚本

职责：
1. --cpu 模式：在 CPU 上执行 golden，验证算子定义正确性
2. --npu 模式：在 NPU 上执行模拟评测（golden 伪装成 AI 算子），验证 NPU 实现正确性并采集性能

使用方式:
    python run_simple.py --cpu --operator Sigmoid          # CPU 简单验证
    python run_simple.py --npu --operator Sigmoid          # NPU 模拟评测
    python run_simple.py --npu --device-id 1 --level 1    # 指定设备，筛选级别
"""

import sys
import argparse
import json
from pathlib import Path
from typing import Callable, Any
from datetime import datetime

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.kernel_eval.data.case_loader import CaseLoader, CaseInfo
from src.kernel_eval.eval.op_runner import OpRunner
from src.kernel_eval.eval.evaluator import Evaluator, EvalOperatorResult
from src.kernel_eval.utils.device_manager import DeviceManager, DeviceConfig
from src.kernel_eval.data.golden_loader import GoldenLoader
from src.kernel_eval.utils.param_builder import ParamBuilder
from src.kernel_eval.eval.perf_eval import PerfEvaluator
from src.kernel_eval.config import get_config

# 测试专用模块
from core.result_recorder import ResultRecorder


def GoldenAsAiAdapter(level: int, operator: str, loader: GoldenLoader = None):
    """将 Golden 函数伪装成 AI 算子接口

    返回一个伪装成 torch 函数的包装器，以避免 torch_npu profiler ERROR 日志

    Args:
        level: 算子级别
        operator: 算子名称
        loader: GoldenLoader 实例

    Returns:
        包装后的函数，__module__ 伪装为 'torch'
    """
    loader = loader or GoldenLoader()
    golden_func = loader.get_golden_function(level, operator)

    def wrapper(*args, **kwargs):
        return golden_func(*args, **kwargs)

    # 伪装属性，让 torch_npu profiler 认为是 torch 原生函数
    wrapper.__name__ = operator.lower()
    wrapper.__module__ = 'torch'
    wrapper.__qualname__ = operator.lower()
    wrapper._is_wrapper = True
    wrapper._original = golden_func

    return wrapper


def parse_args():
    parser = argparse.ArgumentParser(description='测试执行脚本')

    # 设备选项（互斥）
    parser.add_argument('--cpu', action='store_true', help='CPU 简单验证模式')
    parser.add_argument('--npu', action='store_true', help='NPU 模拟评测模式（走标准流程）')
    parser.add_argument('--device-id', type=int, default=0, help='NPU 设备 ID (默认: 0)')

    # 用例筛选
    parser.add_argument('--level', type=int, default=None, choices=[1, 2, 3, 4])
    parser.add_argument('--operator', type=str, default=None)
    parser.add_argument('--case-id', type=int, default=None)

    # 性能配置（仅 NPU 模式）
    parser.add_argument('--warmup', type=int, default=3, help='预热次数')
    parser.add_argument('--repeat', type=int, default=5, help='采集次数')

    # 输出选项
    parser.add_argument('--output', type=str, default='reports/test_results.json')
    parser.add_argument('--export-baseline', type=str, default=None,
                        help='导出性能基线到 JSON（仅 NPU 模式有效）')
    parser.add_argument('--no-perf', action='store_true', help='关闭性能采集，仅做精度验证')
    parser.add_argument('-v', '--verbose', action='store_true')

    return parser.parse_args()


def filter_cases(cases: list, level=None, operator=None, case_id=None) -> list:
    result = cases
    if level:
        result = [c for c in result if c.level == level]
    if operator:
        result = [c for c in result if c.operator.lower() == operator.lower()]
    if case_id:
        result = [c for c in result if c.case_id == case_id]
    return result


def run_cpu_mode(args) -> dict:
    """CPU 模式：简单验证 golden 可执行"""
    print("\n" + "=" * 60)
    print("CPU 验证模式")
    print("=" * 60)

    # 1. 加载用例
    bench_root = project_root / "kernel_bench"
    loader = CaseLoader(str(bench_root))
    all_cases = loader.scan_all_cases()
    cases = filter_cases(all_cases, args.level, args.operator, args.case_id)

    if not cases:
        print("[WARN] 无匹配用例")
        return {"total": 0, "passed": 0, "failed": 0}

    print(f"[INFO] 用例数: {len(cases)}")

    # 2. 初始化
    device_mgr = DeviceManager(DeviceConfig(type="cpu", device_id=0, auto_fallback=True))
    runner = OpRunner(device_mgr, None)  # CPU 模式不启用 profiler
    importer = GoldenLoader(str(bench_root))
    recorder = ResultRecorder(args.output)

    passed = failed = skipped = 0

    for i, case in enumerate(cases):
        case_id_str = case.get_case_id_str()
        print(f"\n[{i+1}/{len(cases)}] {case_id_str}")

        status = _run_single_cpu(case, runner, importer, recorder)
        if status == "passed":
            passed += 1
        elif status == "skipped":
            skipped += 1
        else:
            failed += 1

    recorder.save()
    recorder.print_summary()

    return {"total": len(cases), "passed": passed, "failed": failed, "skipped": skipped}


def _run_single_cpu(case: CaseInfo, runner: OpRunner, importer: GoldenLoader,
                    recorder: ResultRecorder) -> str:
    """执行单个 CPU 验证"""
    case_id_str = case.get_case_id_str()

    # 1. 获取 golden
    try:
        golden_func = importer.get_golden_function(case.level, case.operator)
    except Exception as e:
        recorder.record_skip(case, f"Golden加载失败: {e}")
        print(f"[SKIP] {case_id_str}: {e}")
        return "skipped"

    # 2. 生成输入
    from src.kernel_eval.data.data_generator import DataGenerator
    generator = DataGenerator()
    try:
        input_tensors = generator.generate_input_tensors_from_case(
            input_shapes=case.input_shapes,
            dtypes=case.dtypes,
            value_ranges=case.value_ranges
        )
    except Exception as e:
        recorder.record_skip(case, f"生成输入失败: {e}")
        print(f"[SKIP] {case_id_str}: 生成输入失败 - {e}")
        return "skipped"

    # 3. 预处理
    try:
        get_input_func = importer.get_input_function(case.level, case.operator)
        if get_input_func:
            builder = ParamBuilder(importer)
            params = builder.build_call_params(golden_func, case, input_tensors)
            input_tensors = get_input_func(**params)
            if isinstance(input_tensors, tuple):
                input_tensors = list(input_tensors)
    except Exception as e:
        recorder.record_skip(case, f"预处理失败: {e}")
        print(f"[SKIP] {case_id_str}: 预处理失败 - {e}")
        return "skipped"

    # 4. 构建参数并执行
    builder = ParamBuilder(importer)
    try:
        params = builder.build_call_params(golden_func, case, input_tensors)
    except Exception as e:
        recorder.record_skip(case, f"构建参数失败: {e}")
        print(f"[SKIP] {case_id_str}: 构建参数失败 - {e}")
        return "skipped"

    # CPU 执行
    result = runner.run_golden(golden_func, params, case_id_str, input_tensors)
    recorder.record(case, result)

    if result.success:
        print(f"[PASS] {case_id_str} - {result.elapsed_us / 1000:.2f}ms")
        return "passed"
    else:
        print(f"[FAIL] {case_id_str}: {result.error}")
        return "failed"


def run_npu_mode(args) -> EvalOperatorResult:
    """NPU 模式：走标准评测流程（golden 伪装成 AI 算子）"""
    print("\n" + "=" * 60)
    print("NPU 模拟评测模式（Golden 伪装）")
    print("=" * 60)
    print(f"[CONFIG] 设备: NPU:{args.device_id}")
    print(f"[CONFIG] Warmup/Repeat: {args.warmup}/{args.repeat}")
    if args.no_perf:
        print("[CONFIG] 性能采集: 关闭（仅精度验证）")

    # 1. 配置
    config = get_config()
    config.device_type = "npu"
    config.device_id = args.device_id
    config.enable_profiler = not args.no_perf  # --no-perf 关闭 profiler
    config.warmup = args.warmup
    config.repeat = args.repeat
    config.reports_dir = str(Path(args.output).parent)

    # 2. 加载用例
    loader = CaseLoader(config.kernel_bench_root)
    all_cases = loader.scan_all_cases()
    cases = filter_cases(all_cases, args.level, args.operator, args.case_id)

    if not cases:
        print("[WARN] 无匹配用例")
        return None

    # 3. 确定算子列表
    operators = set((c.level, c.operator) for c in cases)
    print(f"[INFO] 算子数: {len(operators)}, 用例数: {len(cases)}")

    # 4. 执行评测
    evaluator = Evaluator(config)
    all_results = []
    perf_data = {}  # 收集性能数据用于导出基线

    for level, operator in sorted(operators):
        print(f"\n{'=' * 60}")
        print(f"评测算子: {operator} (L{level})")
        print(f"{'=' * 60}")

        # 创建伪装 AI 算子
        golden_as_ai = GoldenAsAiAdapter(level, operator, GoldenLoader(config.kernel_bench_root))

        # 执行标准评测
        result = evaluator.evaluate_operator(
            operator=operator,
            level=level,
            ai_op_func=golden_as_ai,
            case_filter={'case_id': args.case_id} if args.case_id else None,
        )

        all_results.append(result)

        # 收集性能数据
        for r in result.results:
            if r.success and r.perf_result:
                perf_data[r.case_id] = {
                    "level": r.level,
                    "operator": r.operator,
                    "case_id": r.case_num,
                    "elapsed_us": r.perf_result.elapsed_us,
                    "timestamp": datetime.now().isoformat()
                }

        # 打印结果
        print(f"\n[结果] 算子: {operator}")
        print(f"  总用例: {result.total_cases}, 通过: {result.passed_cases}, "
              f"失败: {result.failed_cases}, 跳过: {result.skipped_cases}")
        print(f"  通过率: {result.pass_rate * 100:.1f}%")

        if result.pass_rate < 1.0 and args.verbose:
            for r in result.results:
                if not r.success:
                    print(f"    - {r.case_id}: {r.error_msg or '精度不匹配'}")

    # 5. 汇总
    total_passed = sum(r.passed_cases for r in all_results)
    total_cases = sum(r.total_cases for r in all_results)
    overall_rate = total_passed / total_cases if total_cases > 0 else 0

    print(f"\n{'=' * 60}")
    print("汇总")
    print(f"{'=' * 60}")
    print(f"总用例: {total_cases}")
    print(f"通过: {total_passed}")
    print(f"失败: {total_cases - total_passed}")
    print(f"整体通过率: {overall_rate * 100:.1f}%")

    if overall_rate == 1.0:
        print("\n[✓] Golden NPU 实现与 CPU 参考一致，验证通过！")
    else:
        print("\n[✗] 存在失败的用例，Golden NPU 实现可能有问题。")

    # 6. 导出基线（如果指定）
    if args.export_baseline and perf_data:
        export_baseline(args.export_baseline, perf_data, total_cases)

    # 7. 保存详细结果
    _save_results(args.output, all_results, total_cases, total_passed)

    evaluator.shutdown()

    return all_results[0] if all_results else None


def export_baseline(output_path: str, perf_data: dict, total_cases: int):
    """导出性能基线"""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, 'w', encoding='utf-8') as f:
        json.dump({
            "summary": {
                "total_cases": total_cases,
                "collected": len(perf_data),
                "timestamp": datetime.now().isoformat()
            },
            "baselines": perf_data
        }, f, indent=2, ensure_ascii=False)

    print(f"\n[INFO] 性能基线已导出到: {output}")


def _save_results(output_path: str, results: list, total: int, passed: int):
    """保存详细结果"""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    from dataclasses import asdict
    with open(output, 'w', encoding='utf-8') as f:
        json.dump({
            "summary": {
                "total_cases": total,
                "passed_cases": passed,
                "failed_cases": total - passed,
                "pass_rate": passed / total if total > 0 else 0,
                "timestamp": datetime.now().isoformat()
            },
            "operators": [r.to_dict() for r in results]
        }, f, indent=2, ensure_ascii=False)

    print(f"[INFO] 详细结果已保存到: {output}")


def main():
    args = parse_args()

    if args.cpu:
        result = run_cpu_mode(args)
        if result.get('failed', 0) > 0:
            sys.exit(1)
    elif args.npu:
        result = run_npu_mode(args)
        if result is None or result.pass_rate < 1.0:
            sys.exit(1)
    else:
        print("[ERROR] 请指定 --cpu 或 --npu 模式")
        print("  --cpu: CPU 简单验证")
        print("  --npu: NPU 模拟评测（标准流程）")
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
