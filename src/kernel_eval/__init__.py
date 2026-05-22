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
kernel_eval 评测工程包

用于验证AI生成的Ascend C算子代码（通过whl包传递）
涵盖精度验证和性能评测两个核心维度

主要模块：
- data: 数据层（任务/用例加载器、Golden加载、数据生成）
- eval: 评测层（精度评测、性能评测、算子执行器）
- report: 报告层（JSON+Markdown报告、评分计算）
- utils: 工具层（设备管理、类型映射、精度验证）

架构：
- TaskLoader/CaseLoader: 加载器抽象基类
- CannTaskLoader/CannCaseLoader: CANN特化加载器
- LoaderRegistry: 加载器注册机制（支持多评测体系接入）

使用方法：
    import kernel_eval

    # 加载任务信息（推荐方式）
    from kernel_eval.benches.cann import CannTaskLoader, CannCaseLoader
    task_loader = CannTaskLoader("tasks")
    task_spec = task_loader.get_task("level1/Exp")

    # 或使用注册机制
    from kernel_eval.data import get_task_loader, get_case_loader
    task_loader = get_task_loader("cann", bench_root="tasks")
    case_loader = get_case_loader("cann", bench_root="tasks")

    # 向后兼容方式
    from kernel_eval.benches.cann import CannTaskLoader, CannCaseLoader
    loader = CannTaskLoader("tasks")
    op_info = loader.get_operator("level1/Exp")

    # 执行评测
    from kernel_eval.eval.evaluator import Evaluator
    evaluator = Evaluator()
    results = evaluator.evaluate_operator(operator="Exp", rel_path="level1/Exp")
"""

__version__ = "0.1.0"

from .config import Config, get_config

__all__ = ["Config", "get_config", "__version__"]