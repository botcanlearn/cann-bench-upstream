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
评测配置采集模块

采集评测运行时的元信息和环境参数，供报告生成使用。

数据来源：
- metadata:   framework / date / benchmark 由框架推断；agent_skill / base_model 由用户配置
- environment: NPU/CPU/CANN/PyTorch/Python/OS 运行时自动获取

Usage:
    from kernel_eval.report.setup_info import collect_setup_info

    setup = collect_setup_info(config)
    # setup == { "metadata": {...}, "environment": {...} }
"""

from __future__ import annotations

import os
import sys
import platform
from datetime import datetime
from typing import Dict, Optional

from ..config import Config
from .._version import FRAMEWORK_VERSION, TASKS_VERSION


# ---------------------------------------------------------------------------
# 环境信息采集 (不依赖 torch_npu 也能返回部分信息)
# ---------------------------------------------------------------------------

def _get_npu_info() -> Optional[str]:
    """获取 NPU 设备名称和数量"""
    try:
        import torch_npu  # noqa: F401
        import torch

        count = torch.npu.device_count()
        if count <= 0:
            return None
        name = torch.npu.get_device_name(0)
        return f"{name} × {count}"
    except Exception:
        return None


def _get_cpu_arch() -> str:
    """获取 CPU 架构"""
    return platform.machine() or "unknown"


def _get_cann_version() -> Optional[str]:
    """获取 CANN 版本 (优先从 version.info / version.cfg 读取)"""
    # 1. 从 ASCEND_TOOLKIT_HOME 环境变量读取
    ascend_home = os.environ.get("ASCEND_TOOLKIT_HOME", "")
    if ascend_home:
        for sub in ["compiler/version.info", "version.info", "toolkit/version.cfg"]:
            cfg = os.path.join(ascend_home, sub)
            try:
                with open(cfg) as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("Version="):
                            return line.split("=", 1)[1].strip()
                        if "=" in line and not line.startswith("#"):
                            return line.split("=", 1)[1].strip()
            except Exception:
                continue
    # 2. 尝试标准路径
    for p in ["/usr/local/Ascend/ascend-toolkit/version.cfg", "/usr/local/Ascend/latest/version.cfg"]:
        try:
            with open(p) as f:
                for line in f:
                    if "=" in line and not line.startswith("#"):
                        return line.split("=", 1)[1].strip()
        except Exception:
            continue
    # 3. 尝试从路径推断 (/home/developer/Ascend/cann-<version>/)
    if ascend_home:
        import re
        m = re.search(r'cann-([\d.]+(?:-[\w.]+)?)', ascend_home)
        if m:
            return m.group(1)
    return None


def _get_driver_version() -> Optional[str]:
    """获取 Driver 版本 (跟随 CANN 版本)"""
    cann = _get_cann_version()
    return f"cann-{cann}" if cann else None


def _get_torch_npu_version() -> Optional[str]:
    """获取 PyTorch NPU 插件版本 (来自 torch_npu)"""
    try:
        import torch_npu
        return getattr(torch_npu, '__version__', None)
    except Exception:
        return None


def _get_pytorch_version() -> str:
    """获取 PyTorch 版本"""
    try:
        import torch
        return torch.__version__
    except Exception:
        return "unknown"


def _get_python_version() -> str:
    """获取 Python 版本"""
    return sys.version.split()[0] if sys.version else "unknown"


def _get_os_info() -> str:
    """获取操作系统信息"""
    try:
        return platform.platform() or f"{platform.system()} {platform.release()}"
    except Exception:
        return "unknown"


def _in_container() -> bool:
    """是否跑在容器里。/.dockerenv 覆盖 docker，cgroup 里的 docker/containerd/kubepods
    覆盖 k8s（CI 的评测 pod 就是这一支）。两者都读不到就当作裸机。"""
    if os.path.exists("/.dockerenv"):
        return True
    try:
        with open("/proc/1/cgroup") as f:
            blob = f.read()
        return any(k in blob for k in ("docker", "containerd", "kubepods"))
    except Exception:
        return False


def _detect_docker() -> Optional[str]:
    """报告里的 "docker" 字段：镜像标识 + 其 CANN 版本。

    Why not 硬编码: 这里原本写死 "cake-ci / CANN 9.0.0"。镜像早已换代、CANN 也升到
    9.1.0 之后，那行字仍原样出现在每一份报告里 —— 同一份报告的 environment.cann 显示
    9.1.0、docker 显示 9.0.0，自相矛盾，且读报告的人无从知道哪个可信。

    取值顺序：
      1. CANN_BENCH_IMAGE —— 镜像自己注入的标识（docker/eval 会写成 <name>:<tag>），
         最准确，因为只有构建镜像的人知道它叫什么。
      2. 容器内但没注入 —— 回退成 "container"，至少不谎报具体镜像。
      3. 不在容器里 —— None（报告里显示为空），而不是编一个。
    CANN 版本一律取实际探测到的那个，与 environment.cann 同源，不会再对不上。
    """
    image = (os.environ.get("CANN_BENCH_IMAGE") or "").strip()
    if not image:
        if not _in_container():
            return None
        image = "container"
    cann = _get_cann_version()
    return f"{image} / CANN {cann}" if cann else image


# ---------------------------------------------------------------------------
# 评测集名称推断
# ---------------------------------------------------------------------------

def _resolve_benchmark_name(tasks_root: str) -> str:
    """从 tasks_root 路径推断评测集显示名称"""
    return "CANN-Bench tasks"


# ---------------------------------------------------------------------------
# 主采集函数
# ---------------------------------------------------------------------------

def collect_setup_info(config: Optional[Config] = None) -> Dict:
    """采集评测配置信息

    Args:
        config: 评测配置对象，为 None 时使用默认值

    Returns:
        {
            "metadata": {
                "framework": str,
                "date": str,          # ISO 格式时间戳
                "agent_skill": str,   # 用户配置，可为空
                "base_model": str,    # 用户配置，可为空
                "benchmark": str,     # 评测集名称
                "license": str,
            },
            "environment": {
                "npu": str | None,
                "cpu": str,
                "cann": str | None,
                "driver": str | None,
                "pytorch": str,
                "torchvision": str | None,
                "python": str,
                "os": str,
                "docker": str | None,
            },
        }
    """
    if config is None:
        from ..config import get_config
        config = get_config()

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # torchvision 版本
    torchvision_ver = None
    try:
        import torchvision
        torchvision_ver = torchvision.__version__
    except Exception:
        pass

    return {
        "metadata": {
            "framework": f"CANN-Bench V{FRAMEWORK_VERSION}",
            "tasks_version": TASKS_VERSION,
            "date": now,
            "agent_skill": getattr(config, 'agent_skill', '') or '',
            "base_model": getattr(config, 'base_model', '') or '',
            "harness": getattr(config, 'harness', '') or '',
            "benchmark": _resolve_benchmark_name(config.tasks_root),
            "license": "CANN Open Software License v2.0",
        },
        "environment": {
            "npu": _get_npu_info(),
            "cpu": _get_cpu_arch(),
            "cann": _get_cann_version(),
            "driver": _get_driver_version(),
            "pytorch": _get_pytorch_version(),
            "pytorch_npu": _get_torch_npu_version(),
            "torchvision": torchvision_ver,
            "python": _get_python_version(),
            "os": _get_os_info(),
            "docker": _detect_docker(),
        },
    }
