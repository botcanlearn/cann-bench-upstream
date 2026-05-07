#!/usr/bin/python3
# coding=utf-8

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
进程池协调器

基于多卡 × 多进程并行方案实现：
1. 每个进程独立运行，拥有独立的 torch_npu.profiler 单例
2. 进程内单线程执行，避免 profiler 竞争
3. 任务分配策略：rel_path_parallel / case_parallel
4. 无进程间通信，通过文件传递结果

配置示例：
    processes_per_card = 2  # 每卡 2 个进程
    card_count = 2          # 2 张卡
    total_processes = 4     # 总共 4 个独立进程池
"""

import json
import os
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Any

import torch

from .results import EvalOperatorResult, EvalCaseResult
from ..config import Config, get_config, get_project_root
from ..data.case_loader import CaseInfo


@dataclass
class ProcessConfig:
    """进程池配置"""
    processes_per_card: int = 2      # 每卡进程数
    timeout_per_process: int = 300   # 单进程超时（秒）
    enable_profiler: bool = True     # 是否启用 profiler


class ProcessWorker:
    """单进程工作单元

    独立子进程，拥有独立的：
    - MultiProcessPool 单例（torch_npu.profiler）
    - device_id
    - 单线程执行器

    通过文件传递结果，无进程间通信。
    """

    def __init__(
        self,
        process_id: int,
        card_id: int,
        base_config: Config,
        process_config: ProcessConfig,
    ):
        self.process_id = process_id
        self.card_id = card_id
        self.base_config = base_config
        self.process_config = process_config

        self._process: Optional[subprocess.Popen] = None
        self._output_file: Optional[str] = None
        self._cases_file: Optional[str] = None
        self._started = False

    def start(
        self,
        rel_paths: List[str] = None,
        cases: List[CaseInfo] = None,
    ):
        """启动子进程

        Args:
            rel_paths: 算子相对路径列表（rel_path_parallel 模式）
            cases: 用例列表（case_parallel 模式）
        """
        # 创建输出文件
        fd, self._output_file = tempfile.mkstemp(
            suffix=".json",
            prefix=f"proc{self.process_id}_",
        )
        os.close(fd)

        # 构建命令
        kernel_eval_root = str(get_project_root() / "src")
        cmd = [
            sys.executable, "-m", "kernel_eval.cli", "eval-process",
            "--process-id", str(self.process_id),
            "--card-id", str(self.card_id),
            "--output", self._output_file,
            "--warmup", str(self.base_config.warmup),
            "--repeat", str(self.base_config.repeat),
        ]

        # 添加 profiler 配置
        if self.process_config.enable_profiler:
            cmd.append("--enable-profiler")

        # 添加任务数据
        if rel_paths:
            # rel_path_parallel 模式：传递 rel_path 列表
            cmd.extend(["--rel-paths", ",".join(rel_paths)])

        elif cases:
            # case_parallel 模式：传递 case 数据文件
            fd, self._cases_file = tempfile.mkstemp(suffix=".json", prefix="cases_")
            os.close(fd)
            case_data = self._serialize_cases(cases)
            with open(self._cases_file, 'w') as f:
                json.dump(case_data, f)
            cmd.extend(["--cases-file", self._cases_file])

        # 设置环境变量
        env = os.environ.copy()
        # PYTHONPATH 需要追加，不能覆盖（保留父进程的 TBE 等模块路径）
        existing_pythonpath = env.get("PYTHONPATH", "")
        if existing_pythonpath:
            env["PYTHONPATH"] = f"{kernel_eval_root}:{existing_pythonpath}"
        else:
            env["PYTHONPATH"] = kernel_eval_root
        env["KERNEL_BENCH_ROOT"] = self.base_config.kernel_bench_root

        # 强制无缓冲输出（stdout 是管道而非 TTY 时 Python 默认块缓冲）
        env["PYTHONUNBUFFERED"] = "1"

        # 继承关键的 CANN/Ascend 环境变量（确保子进程能正确访问 NPU）
        cann_env_vars = [
            "ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME", "ASCEND_OPP_PATH",
            "ASCEND_AICPU_PATH", "ASCEND_VISIBLE_DEVICES",
            "TBE_IMPL_PATH", "ASCEND_SLOG_PRINT_TO_STDOUT", "ASCEND_GLOBAL_LOG_LEVEL",
        ]
        for var in cann_env_vars:
            if var in os.environ:
                env[var] = os.environ[var]

        # 启动子进程
        self._process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,  # 行缓冲，实时输出
        )
        self._started = True

    def _serialize_cases(self, cases: List[CaseInfo]) -> List[Dict]:
        """序列化用例数据"""
        return [
            {
                "rel_path": c.rel_path,
                "operator": c.operator,
                "case_id": c.case_id,
                "input_shapes": c.input_shapes,
                "dtypes": c.dtypes,
                "value_ranges": c.value_ranges,
                "attrs": getattr(c, 'attrs', {}),
                "note": getattr(c, 'note', ''),
                "yaml_path": getattr(c, 'yaml_path', ''),
                "baseline_perf_us": c.baseline_perf_us,
            }
            for c in cases
        ]

    def wait(self, timeout: int = None) -> List[Dict]:
        """等待子进程完成并返回结果

        Args:
            timeout: 超时时间（秒）

        Returns:
            结果数据列表（字典格式）
        """
        if not self._started or self._process is None:
            return []

        timeout = timeout or self.process_config.timeout_per_process

        try:
            # 实时读取输出并打印
            stdout_lines = []
            while True:
                line = self._process.stdout.readline()
                if not line:
                    if self._process.poll() is not None:
                        break
                    time.sleep(0.1)
                    continue
                stdout_lines.append(line)
                # 打印子进程输出（进度信息）
                print(line.rstrip())

            self._process.wait(timeout=timeout)

            # 读取结果文件
            if self._output_file and os.path.exists(self._output_file):
                with open(self._output_file, 'r') as f:
                    data = json.load(f)
                return data.get("results", [])

        except subprocess.TimeoutExpired:
            self._process.kill()
            print(f"[WARN] Process {self.process_id} 超时，已终止")
            return []
        finally:
            self._cleanup()

        return []

    def _cleanup(self):
        """清理临时文件"""
        if self._output_file and os.path.exists(self._output_file):
            try:
                os.unlink(self._output_file)
            except OSError:
                pass
        if self._cases_file and os.path.exists(self._cases_file):
            try:
                os.unlink(self._cases_file)
            except OSError:
                pass
        self._output_file = None
        self._cases_file = None

    def is_alive(self) -> bool:
        """检查进程是否仍在运行"""
        return self._process is not None and self._process.poll() is None

    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            'process_id': self.process_id,
            'card_id': self.card_id,
            'started': self._started,
            'alive': self.is_alive(),
        }


class ProcessPoolCoordinator:
    """进程池协调器

    管理 card_count × processes_per_card 个独立进程，
    分配任务并汇总结果。

    支持单卡和多卡模式：
    - 单卡模式：指定 device_id，所有进程绑定到该卡
    - 多卡模式：不指定 device_id，自动检测并轮询分配
    """

    def __init__(
        self,
        base_config: Config = None,
        process_config: ProcessConfig = None,
        device_id: int = None,  # 指定单卡时，所有进程绑定到该卡
    ):
        self.base_config = base_config or get_config()
        self.process_config = process_config or ProcessConfig()
        self.device_id = device_id

        # 单卡模式：所有进程绑定到指定卡
        if device_id is not None:
            self.card_count = 1
        else:
            # 多卡模式：自动检测
            self.card_count = self._detect_cards()

        self.total_processes = self.card_count * self.process_config.processes_per_card
        self.workers: List[ProcessWorker] = []

    def _detect_cards(self) -> int:
        """检测可用 NPU 卡数"""
        if self.base_config.device_type != "npu":
            return 0
        try:
            import torch_npu
            if not torch.npu.is_available():
                return 0
            return torch.npu.device_count()
        except ImportError:
            return 0

    def _create_workers(self) -> List[ProcessWorker]:
        """创建进程工作单元"""
        workers = []
        for i in range(self.total_processes):
            if self.device_id is not None:
                # 单卡模式：所有进程绑定到指定卡
                card_id = self.device_id
            else:
                # 多卡模式：轮询分配到各卡
                card_id = i // self.process_config.processes_per_card
            worker = ProcessWorker(
                process_id=i,
                card_id=card_id,
                base_config=self.base_config,
                process_config=self.process_config,
            )
            workers.append(worker)
        return workers

    def distribute_rel_paths(
        self,
        rel_paths: List[str],
    ) -> Dict[int, List[str]]:
        """分配 rel_paths 到进程池

        Args:
            rel_paths: 算子相对路径列表

        Returns:
            {process_id: [rel_paths]} 分配映射
        """
        distribution = defaultdict(list)
        for i, rel_path in enumerate(rel_paths):
            process_id = i % self.total_processes
            distribution[process_id].append(rel_path)
        return dict(distribution)

    def distribute_cases(
        self,
        cases: List[CaseInfo],
    ) -> Dict[int, List[CaseInfo]]:
        """分配 cases 到进程池

        Args:
            cases: 用例列表

        Returns:
            {process_id: [cases]} 分配映射
        """
        distribution = defaultdict(list)
        for i, case in enumerate(cases):
            process_id = i % self.total_processes
            distribution[process_id].append(case)
        return dict(distribution)

    def evaluate_operators(
        self,
        rel_paths: List[str],
        progress_callback: callable = None,
    ) -> List[EvalOperatorResult]:
        """并行评测多个算子

        自动选择并行策略：
        - 多算子：按 rel_path 分配到进程（rel_path_parallel）
        - 单算子：按用例分配到进程（case_parallel）

        Args:
            rel_paths: 算子相对路径列表
            progress_callback: 进度回调（未实现）

        Returns:
            算子评测结果列表
        """
        if self.card_count == 0:
            print("[WARN] 无可用 NPU 卡")
            return []

        print(f"[INFO] 使用 {self.total_processes} 个进程池并行评测")
        print(f"[INFO] 配置: {self.card_count} 卡 × {self.process_config.processes_per_card} 进程/卡")

        # 根据 rel_path 数量选择并行策略
        if len(rel_paths) == 1:
            # 单算子：使用 case_parallel 模式
            rel_path = rel_paths[0]
            print(f"[INFO] 单算子模式，按用例分配到 {self.total_processes} 个进程")
            result = self.evaluate_cases_parallel(rel_path)
            return [result] if result else []
        else:
            # 多算子：使用 rel_path_parallel 模式
            return self._evaluate_multiple_rel_paths(rel_paths)

    def _evaluate_multiple_rel_paths(
        self,
        rel_paths: List[str],
    ) -> List[EvalOperatorResult]:
        """多算子并行评测（rel_path_parallel 模式）"""
        # 分配任务
        distribution = self.distribute_rel_paths(rel_paths)

        print(f"[INFO] 任务分配 (rel_path_parallel):")
        for proc_id, paths in distribution.items():
            card_id = proc_id // self.process_config.processes_per_card
            print(f"  Process {proc_id} (Card {card_id}): {len(paths)} 算子 - {paths}")

        # 创建并启动进程
        self.workers = self._create_workers()
        for worker in self.workers:
            proc_id = worker.process_id
            if proc_id in distribution:
                worker.start(rel_paths=distribution[proc_id])

        # 并行等待所有进程完成
        all_results = []
        started_workers = [w for w in self.workers if w._started]

        with ThreadPoolExecutor(max_workers=len(started_workers)) as executor:
            futures = {executor.submit(worker.wait): worker for worker in started_workers}
            for future in as_completed(futures):
                results_data = future.result()
                for data in results_data:
                    result = EvalOperatorResult.from_dict(data)
                    all_results.append(result)

        return all_results

    def evaluate_cases_parallel(
        self,
        rel_path: str,
    ) -> Optional[EvalOperatorResult]:
        """单算子多进程并行评测（case_parallel 模式）

        将单个算子的用例分配到多个进程并行执行。

        Args:
            rel_path: 算子相对路径

        Returns:
            算子评测结果（合并所有进程结果）
        """
        # 加载用例
        from ..data.case_loader import CaseLoader
        loader = CaseLoader(self.base_config.kernel_bench_root)
        cases = loader.scan_by_rel_path(rel_path)

        if not cases:
            print(f"[WARN] 算子 {rel_path} 无用例")
            return EvalOperatorResult(
                rel_path=rel_path,
                operator=Path(rel_path).name,  # fallback to dir name
                total_cases=0,
                passed_cases=0,
                failed_cases=0,
                skipped_cases=0,
                results=[],
                pass_rate=0.0,
                avg_speedup=0.0,
            )

        operator_name = cases[0].operator
        print(f"[INFO] 算子 {rel_path} ({operator_name}), 用例数: {len(cases)}")

        # 分配用例到进程
        distribution = self.distribute_cases(cases)

        print(f"[INFO] 任务分配 (case_parallel):")
        for proc_id, proc_cases in distribution.items():
            card_id = proc_id // self.process_config.processes_per_card
            print(f"  Process {proc_id} (Card {card_id}): {len(proc_cases)} 用例")

        # 创建并启动进程
        self.workers = self._create_workers()
        for worker in self.workers:
            proc_id = worker.process_id
            if proc_id in distribution and distribution[proc_id]:
                worker.start(cases=distribution[proc_id])

        # 等待并收集结果（case_parallel 模式）
        all_case_results = []
        started_workers = [w for w in self.workers if w._started]

        with ThreadPoolExecutor(max_workers=len(started_workers)) as executor:
            futures = {executor.submit(worker.wait): worker for worker in started_workers}
            for future in as_completed(futures):
                results_data = future.result()
                for data in results_data:
                    # data 是 EvalOperatorResult.to_dict()，需要提取其中的 results
                    if 'results' in data:
                        for case_data in data['results']:
                            result = EvalCaseResult.from_dict(case_data)
                            all_case_results.append(result)
                    else:
                        # 兼容：如果直接是 case result
                        result = EvalCaseResult.from_dict(data)
                        all_case_results.append(result)

        # 合并统计
        passed = sum(1 for r in all_case_results if r.success)
        failed = sum(1 for r in all_case_results if not r.success and r.accuracy_result is not None)
        skipped = sum(1 for r in all_case_results if not r.success and r.accuracy_result is None)
        speedups = [r.get_speedup() for r in all_case_results if r.success and r.get_speedup() > 0]
        avg_speedup = sum(speedups) / len(speedups) if speedups else 0.0

        return EvalOperatorResult(
            rel_path=rel_path,
            operator=operator_name,
            total_cases=len(cases),
            passed_cases=passed,
            failed_cases=failed,
            skipped_cases=skipped,
            results=all_case_results,
            pass_rate=passed / len(cases) if len(cases) > 0 else 0.0,
            avg_speedup=avg_speedup,
        )

    def evaluate_cases(
        self,
        cases: List[CaseInfo],
        rel_path: str,
        progress_callback: callable = None,
    ) -> EvalOperatorResult:
        """并行评测单个算子的多个用例

        Args:
            cases: 用例列表
            rel_path: 算子相对路径
            progress_callback: 进度回调（未实现）

        Returns:
            算子评测结果（合并所有进程结果）
        """
        operator_name = cases[0].operator if cases else Path(rel_path).name

        if self.card_count == 0:
            print("[WARN] 无可用 NPU 卡")
            return EvalOperatorResult(
                rel_path=rel_path,
                operator=operator_name,
                total_cases=len(cases),
                passed_cases=0,
                failed_cases=len(cases),
                skipped_cases=0,
                results=[],
                pass_rate=0.0,
                avg_speedup=0.0,
            )

        print(f"[INFO] 使用 {self.total_processes} 个进程池并行评测")
        print(f"[INFO] 配置: {self.card_count} 卡 × {self.process_config.processes_per_card} 进程/卡")

        # 分配任务
        distribution = self.distribute_cases(cases)

        print(f"[INFO] 任务分配 (case_parallel):")
        for proc_id, proc_cases in distribution.items():
            card_id = proc_id // self.process_config.processes_per_card
            print(f"  Process {proc_id} (Card {card_id}): {len(proc_cases)} 用例")

        # 创建并启动进程
        self.workers = self._create_workers()
        for worker in self.workers:
            proc_id = worker.process_id
            if proc_id in distribution and distribution[proc_id]:
                worker.start(cases=distribution[proc_id])

        # 等待并收集结果（case_parallel 模式）
        all_case_results = []
        for worker in self.workers:
            if worker._started:
                results_data = worker.wait()
                for data in results_data:
                    # data 是 EvalOperatorResult.to_dict()，需要提取其中的 results
                    if 'results' in data:
                        for case_data in data['results']:
                            result = EvalCaseResult.from_dict(case_data)
                            all_case_results.append(result)
                    else:
                        # 兼容：如果直接是 case result
                        result = EvalCaseResult.from_dict(data)
                        all_case_results.append(result)

        # 合并统计
        passed = sum(1 for r in all_case_results if r.success)
        failed = sum(1 for r in all_case_results if not r.success and r.accuracy_result is not None)
        skipped = sum(1 for r in all_case_results if not r.success and r.accuracy_result is None)
        speedups = [r.get_speedup() for r in all_case_results if r.success and r.get_speedup() > 0]
        avg_speedup = sum(speedups) / len(speedups) if speedups else 0.0

        return EvalOperatorResult(
            rel_path=rel_path,
            operator=operator_name,
            total_cases=len(cases),
            passed_cases=passed,
            failed_cases=failed,
            skipped_cases=skipped,
            results=all_case_results,
            pass_rate=passed / len(cases) if len(cases) > 0 else 0.0,
            avg_speedup=avg_speedup,
        )

    def shutdown(self):
        """关闭所有进程"""
        for worker in self.workers:
            if worker.is_alive():
                worker._process.kill()
        self.workers = []

    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            'device_id': self.device_id,
            'card_count': self.card_count,
            'processes_per_card': self.process_config.processes_per_card,
            'total_processes': self.total_processes,
            'workers': [w.get_stats() for w in self.workers],
        }