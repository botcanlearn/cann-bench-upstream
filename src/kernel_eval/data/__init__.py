#!/usr/bin/python3
# coding=utf-8

# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You can not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""
数据层模块（通用工具）

职责：
1. 导出加载器基类
2. 数据生成器（根据 shape/dtype 生成输入张量）
3. 包管理器（源码扫描、编译、安装、接口扫描）

架构：
- 基类从 base.loaders 导入
- CANN 加载器从 benches.cann 导入
- 通用工具：DataGenerator, PackageManager

使用方式:
    # 基类
    from kernel_eval.data import TaskLoader, CaseLoader, GoldenLoaderBase

    # CANN 加载器
    from kernel_eval.benches.cann import CannTaskLoader, CannCaseLoader, GoldenLoader
"""

# === 基类（从 base/ 导入）===
from ..base.loaders import TaskLoader, CaseLoader, GoldenLoaderBase

# === 通用工具 ===
from .data_generator import DataGenerator
from .package_manager import PackageManager, PackageInfo, InterfaceInfo

__all__ = [
    # 基类
    "TaskLoader",
    "CaseLoader",
    "GoldenLoaderBase",
    # 通用工具
    "DataGenerator",
    "PackageManager",
    "PackageInfo",
    "InterfaceInfo",
]