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
加载器注册表

LoaderRegistry: TaskLoader 和 CaseLoader 注册表

Why: 支持多评测体系扩展，统一管理加载器实例
"""

from typing import Dict, Optional, Type

from ..base.loaders import TaskLoader, CaseLoader
from .base import BaseRegistry


class LoaderRegistry(BaseRegistry[Type[TaskLoader]]):
    """加载器注册表

    管理各评测体系的 TaskLoader 和 CaseLoader。
    """

    _task_loaders: Dict[str, Type[TaskLoader]] = {}
    _case_loaders: Dict[str, Type[CaseLoader]] = {}
    DEFAULT_EVAL_SYSTEM = "cann"

    @classmethod
    def register_task_loader(cls, eval_system: str, loader_cls: Type[TaskLoader]) -> None:
        """注册任务加载器"""
        cls._task_loaders[eval_system] = loader_cls

    @classmethod
    def register_case_loader(cls, eval_system: str, loader_cls: Type[CaseLoader]) -> None:
        """注册用例加载器"""
        cls._case_loaders[eval_system] = loader_cls

    @classmethod
    def get_task_loader_cls(cls, eval_system: str = None) -> Optional[Type[TaskLoader]]:
        """获取任务加载器类"""
        if eval_system is None:
            eval_system = cls.DEFAULT_EVAL_SYSTEM
        return cls._task_loaders.get(eval_system)

    @classmethod
    def get_case_loader_cls(cls, eval_system: str = None) -> Optional[Type[CaseLoader]]:
        """获取用例加载器类"""
        if eval_system is None:
            eval_system = cls.DEFAULT_EVAL_SYSTEM
        return cls._case_loaders.get(eval_system)

    @classmethod
    def list_eval_systems(cls) -> list:
        """列出已注册的评测体系"""
        task_systems = set(cls._task_loaders.keys())
        case_systems = set(cls._case_loaders.keys())
        return list(task_systems & case_systems)

    @classmethod
    def is_registered(cls, eval_system: str) -> bool:
        """检查评测体系是否已注册"""
        return eval_system in cls._task_loaders and eval_system in cls._case_loaders

    @classmethod
    def clear(cls) -> None:
        """清空注册表"""
        cls._task_loaders.clear()
        cls._case_loaders.clear()


def get_task_loader(eval_system: str = "cann", **kwargs) -> TaskLoader:
    """获取任务加载器实例"""
    loader_cls = LoaderRegistry.get_task_loader_cls(eval_system)
    if loader_cls is None:
        raise ValueError(f"评测体系 '{eval_system}' 未注册")
    return loader_cls(**kwargs)


def get_case_loader(eval_system: str = "cann", **kwargs) -> CaseLoader:
    """获取用例加载器实例"""
    loader_cls = LoaderRegistry.get_case_loader_cls(eval_system)
    if loader_cls is None:
        raise ValueError(f"评测体系 '{eval_system}' 未注册")
    return loader_cls(**kwargs)