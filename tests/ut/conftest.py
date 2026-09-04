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
pytest 配置与共享 fixtures
"""

import sys
from pathlib import Path

import pytest

# 添加 src 目录到 Python 路径
src_path = Path(__file__).parent.parent.parent / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))


@pytest.fixture(scope="session", autouse=True)
def _init_npu_backend():
    """NPU 可用时初始化 torch.npu 后端。

    cann_bench_utils 的自定义算子（cann_bench_copy 等）只注册了 NPU
    （PrivateUse1）实现，依赖它的 UT 需要在 NPU 后端运行。
    无 NPU 环境下静默跳过初始化，相关用例由各自文件内的 skipif 显式跳过。
    """
    try:
        import torch
        import torch_npu  # noqa: F401
        if torch.npu.is_available():
            torch.npu.set_device("npu:0")
    except Exception:
        pass


@pytest.fixture(autouse=True)
def _save_restore_global_config():
    """自动保存/恢复全局配置，防止测试间状态泄漏"""
    from kernel_eval.config import _global_config as gc, get_config
    # 触达原值
    saved = gc
    try:
        yield
    finally:
        # 直接恢复模块级变量，避免 set_config 的副作用
        import kernel_eval.config as _mod
        _mod._global_config = saved
