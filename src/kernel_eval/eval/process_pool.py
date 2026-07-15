#!/usr/bin/python3
# coding=utf-8

# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""
进程池协调器 — 统一 TaskUnit 调度

核心设计：
1. 任务单元 = (算子, 用例组, device_id)，统一调度粒度
2. 不管单算子还是多算子，按 (算子×用例组) 均分到各卡
3. 子进程通过 eval-child 独立子命令执行（纯执行者，不做调度/编译/fork）
4. 主进程按算子维度聚合 case 结果 → EvalOperatorResult

配置示例：
    processes_per_card = 1  # 每卡并发进程数（profiler 开启时强制为 1）
    card_count = 8          # 8 张 NPU 卡
    timeout_per_operator = 300  # 单算子超时（秒）
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
from typing import Dict, List, Optional

import torch

from .results import EvalOperatorResult, EvalCaseResult, summarize_case_results
from .subprocess_utils import (
    _CANN_ENV_VARS,
    _write_oom_score_adj,
    _is_oom_killed,
    _synthesize_failure_cases,
    _try_recover_partial_results,
)
from ..config import Config, get_config, get_project_root
from ..base.models import CaseSpec


# ---------------------------------------------------------------------------
# 数据类
# ---------------------------------------------------------------------------

@dataclass
class ProcessConfig:
    """进程池配置"""
    processes_per_card: int = 2      # 每卡最大并发进程数
    timeout_per_operator: int = 300  # 单算子超时（秒）
    enable_profiler: bool = True     # 是否启用 profiler
    max_retries: int = 1             # 最大重试次数
    retry_on_timeout: bool = True    # 超时是否重试
    retry_on_oom: bool = True        # OOM 是否重试
    retry_on_failure: bool = True    # 子进程异常是否重试
    exclude_repeatedly_failed_cases: bool = True  # 排除多次失败的case


@dataclass
class TaskUnit:
    """统一任务单元 = (算子, 用例组, device_id)"""
    operator: str               # 算子名称
    rel_path: str               # 算子相对路径
    cases: List[CaseSpec]       # 该进程需要跑的用例列表
    device_id: int              # 分配的 NPU 卡 ID
    retry_count: int = 0        # 重试次数
    excluded_devices: set = None  # 排除的设备ID集合
    parent_task_id: str = None  # 父任务ID（用于追踪）

    def __post_init__(self):
        if self.excluded_devices is None:
            self.excluded_devices = set()


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def split_into_chunks(items: list, n: int) -> list:
    """将列表分成 n 个尽量均匀的块"""
    if n <= 0:
        return [items]
    k, m = divmod(len(items), n)
    return [items[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]


def build_task_units(
    cases_by_operator: Dict[str, List[CaseSpec]],
    card_count: int,
) -> List[TaskUnit]:
    """将算子×用例拆分为 TaskUnit，均分到各卡。

    每个算子的用例按卡数分组，形成 TaskUnit 列表。
    多算子场景：算子A的用例分到卡0-7，算子B的用例也分到卡0-7 → 自然负载均衡。
    单算子场景：用例分到卡0-7 → 单算子多卡并行。
    单卡场景：只有一个 chunk → 退化串行。
    """
    if card_count <= 0:
        return []

    task_units: List[TaskUnit] = []
    card_ids = list(range(card_count))

    for operator_name, cases in cases_by_operator.items():
        chunks = split_into_chunks(cases, card_count)
        for i, chunk in enumerate(chunks):
            if chunk:
                task_units.append(TaskUnit(
                    operator=operator_name,
                    rel_path=chunk[0].rel_path,
                    cases=chunk,
                    device_id=card_ids[i % len(card_ids)],
                ))

    return task_units


def aggregate_by_operator(
    all_case_results: List[EvalCaseResult],
) -> List[EvalOperatorResult]:
    """按算子名聚合 case 结果 → EvalOperatorResult"""
    grouped: Dict[str, List[EvalCaseResult]] = defaultdict(list)
    for cr in all_case_results:
        grouped[cr.operator].append(cr)

    results: List[EvalOperatorResult] = []
    for op_name, case_results in grouped.items():
        summary = summarize_case_results(case_results)
        rel_path = case_results[0].rel_path if case_results else ""
        results.append(EvalOperatorResult(
            rel_path=rel_path,
            operator=op_name,
            total_cases=len(case_results),
            passed_cases=summary.passed,
            failed_cases=summary.failed,
            skipped_cases=summary.skipped,
            results=case_results,
            pass_rate=summary.pass_rate,
            avg_speedup=summary.avg_speedup,
        ))

    return results


# ---------------------------------------------------------------------------
# ProcessPoolCoordinator
# ---------------------------------------------------------------------------

class ProcessPoolCoordinator:
    """进程池协调器

    管理 card_count × processes_per_card 个并发槽位，
    按 TaskUnit 分配任务并汇总结果。

    支持单卡和多卡模式：
    - 单卡模式：指定 device_id，card_count=1
    - 多卡模式：不指定 device_id，自动检测
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
            # 多卡模式：自动检测，过滤掉不健康的卡
            self.card_count = self._detect_cards()

        # torch_npu.profiler 使用 ACL 设备级 profiling 硬件资源，
        # 同一 NPU 卡上多进程并发 profile 会竞争该资源。
        # profiling 开启时每卡仅 1 进程，保证 profiler 独占硬件资源。
        if self.process_config.enable_profiler:
            self.process_config.processes_per_card = 1

        self.total_processes = self.card_count * self.process_config.processes_per_card
        # 记录活跃子进程，用于 shutdown 时清理
        self._active_processes: List[subprocess.Popen] = []

    def _detect_cards(self) -> int:
        """检测可用 NPU 卡数，并提供详细的诊断信息

        通过 npu-smi info 检测每张卡的健康状态，过滤掉 Alarm/异常卡。
        """
        if self.base_config.device_type != "npu":
            return 0

        try:
            import torch_npu
        except ImportError as e:
            print("[ERROR] 无法导入 torch_npu 模块")
            print(f"  原因: {e}")
            self._check_npu_smi()
            return 0

        if not torch.npu.is_available():
            print("[ERROR] torch.npu.is_available() 返回 False")
            self._check_npu_smi()
            return 0

        card_count = torch.npu.device_count()
        if card_count == 0:
            print("[ERROR] torch.npu.device_count() 返回 0")
            self._check_npu_smi()
            return 0

        # 检测卡健康状态，过滤掉 Alarm 卡
        healthy_cards = self._filter_healthy_cards(card_count)

        print(f"[INFO] 检测到 {card_count} 张 NPU 卡, {len(healthy_cards)} 张健康")
        for i in healthy_cards:
            try:
                name = torch.npu.get_device_name(i) if hasattr(torch.npu, 'get_device_name') else 'unknown'
                print(f"  NPU:{i} - {name}")
            except Exception:
                print(f"  NPU:{i}")

        if not healthy_cards:
            print("[ERROR] 没有健康的 NPU 卡可用")
            return 0

        # 更新 card_count 为健康卡数
        return len(healthy_cards)

    # npu-smi 的 health 列位置随驱动/CANN 版本变化（有时 id+name 合并、有时 health
    # 不在 parts[3]）。fail-open：仅当某行**明确**出现坏状态词且能解析出 npu_id 时才跳过；
    # 解析不出或列错位时一律保留，避免把健康卡误杀导致整 eval 无可用卡（见 ST 0-case 故障）。
    _NPU_BAD_HEALTH = ("ALARM", "CRITICAL", "WARNING", "ABNORMAL", "FAULT", "ERROR")

    def _filter_healthy_cards(self, card_count: int) -> List[int]:
        """通过 npu-smi info 检测卡健康状态，返回健康卡的 ID 列表（fail-open）。

        列布局不固定，故扫描整行的 `|` 分隔 cell：任一 cell 命中坏状态词才视为异常行，
        再从行内取首个可解析为 int 的 cell 作为 npu_id。无法对齐/解析时保留该卡。
        """
        healthy = list(range(card_count))
        try:
            result = subprocess.run(
                ["npu-smi", "info"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode != 0 or not result.stdout.strip():
                return healthy

            for line in result.stdout.strip().split('\n'):
                if '|' not in line:
                    continue
                parts = [p.strip() for p in line.split('|')]
                # 行内任一 cell 是坏状态词才考虑跳过；否则保留（fail-open）
                if not any(p.upper() in self._NPU_BAD_HEALTH for p in parts):
                    continue
                # npu-smi 数据行：首个 cell 的首 token 是 NPU ID
                # （可能单独 "0" 或与 name 合并 "1 Ascend910"）
                first_cell = parts[1] if len(parts) > 1 else ""
                id_tokens = first_cell.split()
                if not id_tokens:
                    continue
                try:
                    npu_id = int(id_tokens[0])
                except ValueError:
                    continue
                if npu_id < card_count and npu_id in healthy:
                    print(f"[WARN] NPU:{npu_id} 状态异常，跳过该卡")
                    healthy.remove(npu_id)
        except FileNotFoundError:
            pass
        except Exception:
            pass
        return healthy

    def _check_npu_smi(self):
        """调用 npu-smi info 检查硬件状态"""
        try:
            result = subprocess.run(
                ["npu-smi", "info"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0 and result.stdout.strip():
                print("\n[诊断] npu-smi info 输出:")
                lines = result.stdout.strip().split('\n')
                for line in lines[:20]:
                    print(f"  {line}")
                if len(lines) > 20:
                    print(f"  ... (共 {len(lines)} 行，已截断)")
        except FileNotFoundError:
            print("\n[诊断] 未找到 npu-smi 命令")
        except subprocess.TimeoutExpired:
            print("\n[诊断] npu-smi info 执行超时")
        except Exception as e:
            print(f"\n[诊断] npu-smi info 执行异常: {e}")

    def _build_env(self) -> Dict[str, str]:
        """构建子进程环境变量"""
        kernel_eval_root = str(get_project_root() / "src")
        env = os.environ.copy()
        existing_pythonpath = env.get("PYTHONPATH", "")
        if kernel_eval_root not in existing_pythonpath:
            env["PYTHONPATH"] = f"{kernel_eval_root}:{existing_pythonpath}" if existing_pythonpath else kernel_eval_root
        env["PYTHONUNBUFFERED"] = "1"
        env["TASKS_ROOT"] = self.base_config.tasks_root
        env["ASCEND_SLOG_PRINT_TO_STDOUT"] = "0"
        env["ASCEND_GLOBAL_LOG_LEVEL"] = "3"

        for var in _CANN_ENV_VARS:
            if var in os.environ:
                env[var] = os.environ[var]

        return env

    def _build_env_for_task(self, base_env: Dict[str, str], task: TaskUnit) -> Dict[str, str]:
        """构建子进程的环境变量

        F745: 为每个子进程设置独立的 ASCEND_RT_VISIBLE_DEVICES，
        让每个进程只能看到分配给它的那张卡，从而彻底隔离多卡之间的资源竞争。

        关键语义：task.device_id 是**相对父进程可见集的逻辑索引**（0..card_count-1），
        因为 card_count 来自 torch.npu.device_count()，已受父进程可见性约束。
        ASCEND_RT_VISIBLE_DEVICES 接受的也是父进程可见空间内的相对索引，而非全局
        物理卡号——因此这里直接写 task.device_id 即可，无需做物理号映射。

        （曾误改为"逻辑→物理号映射"，在父进程用 ASCEND_VISIBLE_DEVICES=12,13 收窄时
        会写出 ASCEND_RT_VISIBLE_DEVICES=12 导致子进程 device_count=0、set_device 失败。）
        """
        env = base_env.copy()

        # 让每个子进程只看到分配给它的那张卡（相对父进程可见集的索引）
        env['ASCEND_RT_VISIBLE_DEVICES'] = str(task.device_id)

        return env

    def _select_device_for_retry(
        self,
        original_device: int,
        excluded_devices: set,
        healthy_cards: List[int]
    ) -> int:
        """为重试任务选择设备，优先选择未出问题的卡

        Args:
            original_device: 原始失败的设备ID
            excluded_devices: 已排除的设备ID集合
            healthy_cards: 健康的卡ID列表

        Returns:
            选择的设备ID
        """
        # 优先选择从未失败过的卡
        available = [d for d in healthy_cards if d not in excluded_devices]
        if available:
            # 轮询选择（简单的负载均衡）
            return available[0]

        # 所有卡都有问题，重用原设备（可能是偶发问题）
        return original_device

    def _build_child_cmd(self, task: TaskUnit, cases_file: str, output_file: str) -> List[str]:
        """构建 eval-child 子进程命令"""
        cmd = [sys.executable, "-u", "-m", "kernel_eval.cli", "eval-child",
               "--bench-name", self.base_config.bench_name,
               "--device-id", str(task.device_id),
               "--cases-file", cases_file,
               "--output", output_file,
               "--warmup", str(self.base_config.warmup),
               "--repeat", str(self.base_config.repeat),
               ]

        reports_dir = getattr(self.base_config, "reports_dir", "") or ""
        if reports_dir:
            cmd += ["--reports-dir", str(reports_dir)]

        # task-dir 透传
        tasks_root = getattr(self.base_config, "tasks_root", "")
        if tasks_root:
            cmd += ["--task-dir", str(tasks_root)]

        # source-dir 透传（Stanford bench 等需要在子进程中加载 ai_op.py）
        source_dir = getattr(self.base_config, "source_dir", "") or ""
        if source_dir:
            cmd += ["--source-dir", str(source_dir)]

        # profiler 配置
        if not self.process_config.enable_profiler:
            cmd.append("--no-perf")
        profiler_level = getattr(self.base_config, "profiler_level", None)
        if profiler_level:
            cmd += ["--profiler-level", str(profiler_level)]
        if not getattr(self.base_config, "perf_freq_boost", True):
            cmd.append("--no-freq-boost")

        # torch op guard 模式
        torch_op_guard_mode = getattr(self.base_config, "torch_op_guard_mode", None)
        if torch_op_guard_mode:
            cmd += ["--torch-op-guard-mode", str(torch_op_guard_mode)]

        # eval seed
        eval_seed = getattr(self.base_config, "eval_seed", None)
        if eval_seed is not None:
            cmd += ["--eval-seed", str(eval_seed)]

        return cmd

    def evaluate_task_units(self, task_units: List[TaskUnit]) -> List[EvalCaseResult]:
        """按 TaskUnit 并行评测

        每个 TaskUnit 启动一个 eval-child 子进程，
        通过 ThreadPoolExecutor 实现多卡并行和动态负载均衡。
        支持失败任务自动重试机制。
        """
        if self.card_count == 0:
            if os.environ.get("ALLOW_NO_NPU_CARDS") == "1":
                print("[WARN] 无可用 NPU 卡 (ALLOW_NO_NPU_CARDS=1)", flush=True)
                return []
            raise RuntimeError(
                "[ERROR] 无可用 NPU 卡 (card_count=0)。多卡评测需要至少 1 张 NPU。"
                "如确需在无 NPU 环境跑空评测做 dry-run，设置 ALLOW_NO_NPU_CARDS=1。"
            )

        if not task_units:
            return []

        max_workers = self.card_count * self.process_config.processes_per_card
        total_cases = sum(len(t.cases) for t in task_units)

        print(f"[INFO] 配置: {self.card_count} 卡 × {self.process_config.processes_per_card} 并发/卡")
        print(f"[INFO] TaskUnit 数: {len(task_units)}, 用例数: {total_cases}")
        print(f"[INFO] 单算子超时: {self.process_config.timeout_per_operator}s")
        print(f"[INFO] 最大并发: {max_workers}")
        if self.process_config.max_retries > 0:
            print(f"[INFO] 重试策略: 最大重试 {self.process_config.max_retries} 次 "
                  f"(timeout={self.process_config.retry_on_timeout}, "
                  f"oom={self.process_config.retry_on_oom}, "
                  f"failure={self.process_config.retry_on_failure})")

        base_env = self._build_env()
        all_case_results: List[EvalCaseResult] = []
        completed_count = 0
        retry_queue: List[TaskUnit] = []  # 重试队列
        healthy_cards = list(range(self.card_count))
        case_failure_count: Dict[str, int] = defaultdict(int)  # 记录每个case的失败次数

        def _run_task(idx_and_task):
            """在线程中运行一个 TaskUnit，返回 (task, case_results, should_retry, failure_type)"""
            idx, task = idx_and_task
            # 写 cases JSON 文件
            fd, cases_file = tempfile.mkstemp(suffix=".json", prefix="cases_")
            os.close(fd)
            try:
                Path(cases_file).write_text(json.dumps([c.to_dict() for c in task.cases], ensure_ascii=False))

                # 写 output 文件占位
                fd, output_file = tempfile.mkstemp(suffix=".json", prefix="cannbench_")
                os.close(fd)

                cmd = self._build_child_cmd(task, cases_file, output_file)
                env = self._build_env_for_task(base_env, task)
                timeout = len(task.cases) * self.process_config.timeout_per_operator

                proc = subprocess.Popen(cmd, start_new_session=True, env=env)
                self._active_processes.append(proc)
                oom_ok = _write_oom_score_adj(proc.pid, 1000)
                # 父进程外部写是双保险，子进程自设（cmd_eval_child）才是主路径
                if not oom_ok:
                    print(f"[WARN] 子进程 PID={proc.pid} oom_score_adj 设置失败"
                          f"（OOM Kill 时主进程也可能被杀）", flush=True)

                try:
                    rc = proc.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    # 从活跃列表移除已完成的子进程
                    try:
                        self._active_processes.remove(proc)
                    except ValueError:
                        pass

                    print(f"[WARN] TaskUnit {task.operator}@Card{task.device_id} 超时 ({timeout}s)")
                    proc.terminate()
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
                    # 超时：尝试恢复部分结果，失败则合成全量 timeout 失败
                    partial = _try_recover_partial_results(output_file)
                    if partial:
                        completed_ids = {r.case_id for r in partial}
                        remaining = [c for c in task.cases if c.get_case_id_str() not in completed_ids]
                        oom_rest = _synthesize_failure_cases(remaining, "timeout",
                            f"子进程超时 ({timeout}s) 被 SIGTERM/SIGKILL")
                        print(f"[INFO] {task.operator}: 超时后恢复 {len(partial)} 个已完成用例，"
                              f"合成 {len(oom_rest)} 个超时失败用例")
                        # 判断是否应该重试
                        should_retry = (
                            self.process_config.retry_on_timeout and
                            task.retry_count < self.process_config.max_retries and
                            len(remaining) > 0
                        )
                        return (task, partial + oom_rest, should_retry, "timeout", remaining)
                    should_retry = (
                        self.process_config.retry_on_timeout and
                        task.retry_count < self.process_config.max_retries
                    )
                    return (task, _synthesize_failure_cases(task.cases, "timeout",
                        f"子进程超时 ({timeout}s) 被 SIGTERM/SIGKILL"), should_retry, "timeout", task.cases)

                # 从活跃列表移除已完成的子进程，避免内存累积
                try:
                    self._active_processes.remove(proc)
                except ValueError:
                    pass

                if rc != 0:
                    if _is_oom_killed(proc, rc):
                        # OOM Kill：尝试恢复部分结果 + 合成剩余用例的 oom_killed 失败
                        partial = _try_recover_partial_results(output_file)
                        if partial:
                            completed_ids = {r.case_id for r in partial}
                            remaining = [c for c in task.cases if c.get_case_id_str() not in completed_ids]
                            oom_rest = _synthesize_failure_cases(remaining, "oom_killed",
                                "子进程被 OOM Killer 杀死 (SIGKILL/-9)，内存不足")
                            print(f"[WARN] {task.operator}@Card{task.device_id}: OOM Kill (rc={rc})")
                            print(f"[INFO] {task.operator}: OOM Kill 后恢复 {len(partial)} 个已完成用例，"
                                  f"合成 {len(oom_rest)} 个 OOM 失败用例")
                            # 判断是否应该重试
                            should_retry = (
                                self.process_config.retry_on_oom and
                                task.retry_count < self.process_config.max_retries and
                                len(remaining) > 0
                            )
                            return (task, partial + oom_rest, should_retry, "oom_killed", remaining)
                        print(f"[WARN] {task.operator}@Card{task.device_id}: OOM Kill (rc={rc})，无部分结果可恢复")
                        should_retry = (
                            self.process_config.retry_on_oom and
                            task.retry_count < self.process_config.max_retries
                        )
                        return (task, _synthesize_failure_cases(task.cases, "oom_killed",
                            "子进程被 OOM Killer 杀死 (SIGKILL/-9)，内存不足"), should_retry, "oom_killed", task.cases)
                    print(f"[WARN] {task.operator}@Card{task.device_id}: 子进程异常退出 rc={rc}")
                    should_retry = (
                        self.process_config.retry_on_failure and
                        task.retry_count < self.process_config.max_retries
                    )
                    return (task, _synthesize_failure_cases(task.cases, "subprocess_failure",
                        f"子进程异常退出 rc={rc}"), should_retry, "subprocess_failure", task.cases)

                # 正常退出：读取完整结果
                # 从活跃列表移除已完成的子进程
                try:
                    self._active_processes.remove(proc)
                except ValueError:
                    pass

                try:
                    data = json.loads(Path(output_file).read_text())
                except (json.JSONDecodeError, OSError) as e:
                    print(f"[WARN] TaskUnit {task.operator}@Card{task.device_id} 结果解析失败: {e}")
                    should_retry = (
                        self.process_config.retry_on_failure and
                        task.retry_count < self.process_config.max_retries
                    )
                    return (task, _synthesize_failure_cases(task.cases, "subprocess_failure",
                        f"子进程结果 JSON 解析失败: {e}"), should_retry, "subprocess_failure", task.cases)

                case_results = [EvalCaseResult.from_dict(r) for r in data.get("case_results", [])]
                return (task, case_results, False, None, [])  # 成功，不需要重试

            finally:
                try:
                    os.unlink(cases_file)
                except OSError:
                    pass
                try:
                    os.unlink(output_file)
                except OSError:
                    pass

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            indexed_tasks = list(enumerate(task_units))
            futures = {
                executor.submit(_run_task, item): item
                for item in indexed_tasks
            }

            for future in as_completed(futures):
                item = futures[future]
                idx, task = item
                try:
                    task_info, case_results, should_retry, failure_type, failed_cases = future.result()
                    completed_count += 1
                    all_case_results.extend(case_results)
                    passed = sum(1 for r in case_results if r.success)
                    status = "✅" if passed > 0 else "❌"
                    print(f"[INFO] [{completed_count}/{len(task_units)}] Card {task.device_id}: "
                          f"{task.operator} {status} ({passed}/{len(task.cases)})")

                    # 处理重试逻辑
                    if should_retry and failed_cases:
                        # 更新失败计数
                        for case in failed_cases:
                            case_id = case.get_case_id_str()
                            case_failure_count[case_id] += 1

                        # 过滤掉多次失败的case（如果启用了排除策略）
                        if self.process_config.exclude_repeatedly_failed_cases:
                            repeatedly_failed = []
                            cases_to_retry = []
                            for case in failed_cases:
                                case_id = case.get_case_id_str()
                                if case_failure_count[case_id] >= 2:
                                    # 已经失败2次，不再重试
                                    repeatedly_failed.append(case)
                                else:
                                    cases_to_retry.append(case)

                            # 将多次失败的case记录到结果中（标记为最终失败）
                            if repeatedly_failed:
                                for case in repeatedly_failed:
                                    case_id = case.get_case_id_str()
                                    print(f"[WARN] 用例 {case_id} 已连续失败 {case_failure_count[case_id]} 次，"
                                          f"不再重试，标记为最终失败")
                        else:
                            cases_to_retry = failed_cases

                        # 只有还有需要重试的case时才创建重试任务
                        if cases_to_retry:
                            # 创建重试任务
                            new_device = self._select_device_for_retry(
                                task.device_id,
                                task.excluded_devices,
                                healthy_cards
                            )
                            retry_task = TaskUnit(
                                operator=task.operator,
                                rel_path=task.rel_path,
                                cases=cases_to_retry,
                                device_id=new_device,
                                retry_count=task.retry_count + 1,
                                excluded_devices=task.excluded_devices | {task.device_id},
                                parent_task_id=f"{task.operator}@Card{task.device_id}"
                            )
                            retry_queue.append(retry_task)
                            excluded_count = len(failed_cases) - len(cases_to_retry)
                            excluded_msg = f", 排除 {excluded_count} 个多次失败的用例" if excluded_count > 0 else ""
                            print(f"[INFO] 重试任务已加入队列: {task.operator} "
                                  f"(retry {retry_task.retry_count}/{self.process_config.max_retries}, "
                                  f"{len(cases_to_retry)} 个用例{excluded_msg}, "
                                  f"Card{task.device_id}→Card{new_device}, "
                                  f"原因: {failure_type})")

                    # 定期清理已退出的子进程引用，避免内存累积
                    self._cleanup_completed_processes()

                    # 每完成 3 个任务执行一次 GC，回收主进程临时对象
                    if completed_count % 3 == 0:
                        import gc
                        gc.collect()
                        avail_mb = self._get_available_memory_mb()
                        if avail_mb > 0 and avail_mb < 2048:
                            print(f"[WARN] 可用内存低: {avail_mb:.0f} MB，"
                                  f"活跃子进程: {len(self._active_processes)}", flush=True)

                except Exception as e:
                    completed_count += 1
                    print(f"[WARN] [{completed_count}/{len(task_units)}] Card {task.device_id}: "
                          f"{task.operator} 异常: {e}")

        print(f"[INFO] 初次调度完成: {completed_count}/{len(task_units)} 个 TaskUnit")

        # 处理重试队列
        if retry_queue:
            print(f"[INFO] 开始处理重试队列: {len(retry_queue)} 个任务")
            retry_results = self._process_retry_queue(retry_queue, base_env, max_workers)
            all_case_results.extend(retry_results)

        print(f"[INFO] 调度完成: 总共 {len(task_units)} + {len(retry_queue)} 个 TaskUnit")

        # F744: 移除父进程的统一等待（无效）
        # 原因：父进程等待时，子进程已经退出，profiler 解析进程已被杀死
        # 正确的方案：在子进程退出前等待（cli.py 中的 20 秒等待）

        return all_case_results

    def _process_retry_queue(
        self,
        retry_queue: List[TaskUnit],
        base_env: Dict[str, str],
        max_workers: int
    ) -> List[EvalCaseResult]:
        """处理重试队列中的任务

        Args:
            retry_queue: 需要重试的TaskUnit列表
            base_env: 基础环境变量
            max_workers: 最大并发数

        Returns:
            重试任务的用例结果列表
        """
        retry_results: List[EvalCaseResult] = []
        completed_count = 0

        def _run_retry_task(idx_and_task):
            """运行重试任务（不再生成新的重试）"""
            idx, task = idx_and_task
            fd, cases_file = tempfile.mkstemp(suffix=".json", prefix="retry_cases_")
            os.close(fd)
            try:
                Path(cases_file).write_text(json.dumps([c.to_dict() for c in task.cases], ensure_ascii=False))

                fd, output_file = tempfile.mkstemp(suffix=".json", prefix="retry_cannbench_")
                os.close(fd)

                cmd = self._build_child_cmd(task, cases_file, output_file)
                env = self._build_env_for_task(base_env, task)
                timeout = len(task.cases) * self.process_config.timeout_per_operator

                proc = subprocess.Popen(cmd, start_new_session=True, env=env)
                self._active_processes.append(proc)
                _write_oom_score_adj(proc.pid, 1000)

                try:
                    rc = proc.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    try:
                        self._active_processes.remove(proc)
                    except ValueError:
                        pass
                    print(f"[WARN] 重试任务 {task.operator}@Card{task.device_id} 仍然超时")
                    proc.terminate()
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
                    partial = _try_recover_partial_results(output_file)
                    if partial:
                        completed_ids = {r.case_id for r in partial}
                        remaining = [c for c in task.cases if c.get_case_id_str() not in completed_ids]
                        failed = _synthesize_failure_cases(remaining, "timeout",
                            f"重试后仍超时 ({timeout}s)")
                        return (task, partial + failed)
                    return (task, _synthesize_failure_cases(task.cases, "timeout",
                        f"重试后仍超时 ({timeout}s)"))

                try:
                    self._active_processes.remove(proc)
                except ValueError:
                    pass

                if rc != 0:
                    if _is_oom_killed(proc, rc):
                        partial = _try_recover_partial_results(output_file)
                        if partial:
                            completed_ids = {r.case_id for r in partial}
                            remaining = [c for c in task.cases if c.get_case_id_str() not in completed_ids]
                            failed = _synthesize_failure_cases(remaining, "oom_killed",
                                "重试后仍 OOM Kill")
                            return (task, partial + failed)
                        return (task, _synthesize_failure_cases(task.cases, "oom_killed",
                            "重试后仍 OOM Kill"))
                    return (task, _synthesize_failure_cases(task.cases, "subprocess_failure",
                        f"重试后仍异常退出 rc={rc}"))

                try:
                    data = json.loads(Path(output_file).read_text())
                    case_results = [EvalCaseResult.from_dict(r) for r in data.get("case_results", [])]
                    return (task, case_results)
                except (json.JSONDecodeError, OSError) as e:
                    return (task, _synthesize_failure_cases(task.cases, "subprocess_failure",
                        f"重试后结果解析失败: {e}"))

            finally:
                try:
                    os.unlink(cases_file)
                except OSError:
                    pass
                try:
                    os.unlink(output_file)
                except OSError:
                    pass

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            indexed_tasks = list(enumerate(retry_queue))
            futures = {
                executor.submit(_run_retry_task, item): item
                for item in indexed_tasks
            }

            for future in as_completed(futures):
                item = futures[future]
                idx, task = item
                try:
                    task_info, case_results = future.result()
                    completed_count += 1
                    retry_results.extend(case_results)
                    passed = sum(1 for r in case_results if r.success)
                    status = "✅" if passed > 0 else "❌"
                    print(f"[INFO] [重试 {completed_count}/{len(retry_queue)}] Card {task.device_id}: "
                          f"{task.operator} {status} ({passed}/{len(task.cases)}) "
                          f"(retry {task.retry_count})")

                    self._cleanup_completed_processes()

                except Exception as e:
                    completed_count += 1
                    print(f"[WARN] [重试 {completed_count}/{len(retry_queue)}] Card {task.device_id}: "
                          f"{task.operator} 异常: {e}")

        return retry_results

    def _cleanup_completed_processes(self):
        """清理已完成的子进程引用，避免内存累积。

        `_active_processes` 列表持有所有子进程的 Popen 对象引用。
        已退出的进程对象虽然轻量但仍占用内存，定期清理可防止长时间运行时内存泄漏。
        """
        completed = [p for p in self._active_processes if p.poll() is not None]
        for p in completed:
            try:
                self._active_processes.remove(p)
            except ValueError:
                pass
        if completed:
            import gc
            gc.collect()

    @staticmethod
    def _get_available_memory_mb() -> float:
        """获取系统可用内存（MB）。"""
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        kb = int(line.split()[1])
                        return kb / 1024.0
        except (OSError, ValueError, IndexError):
            pass
        return 0.0

    def shutdown(self):
        """关闭所有活跃子进程

        SIGTERM 先，10s 宽限后 SIGKILL。
        """
        grace_sec = 5
        for proc in self._active_processes:
            if proc.poll() is None:
                try:
                    proc.terminate()
                except Exception:
                    pass

        if self._active_processes:
            deadline = time.time() + grace_sec
            for proc in self._active_processes:
                remaining = max(deadline - time.time(), 0)
                if remaining <= 0 or proc.poll() is not None:
                    continue
                try:
                    proc.wait(timeout=remaining)
                except subprocess.TimeoutExpired:
                    pass

        for proc in self._active_processes:
            if proc.poll() is None:
                try:
                    proc.kill()
                except Exception:
                    pass
        self._active_processes = []

    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            'device_id': self.device_id,
            'card_count': self.card_count,
            'processes_per_card': self.process_config.processes_per_card,
            'total_processes': self.total_processes,
        }
