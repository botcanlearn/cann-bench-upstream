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
加载器基类（统一）

包含：
- OperatorDirMixin: 算子目录检测 mixin
- TaskLoader: 任务加载器抽象基类
- CaseLoader: 用例加载器抽象基类
- GoldenLoaderBase: Golden 参考实现加载器抽象基类

Why: 为所有评测体系提供统一的加载器接口
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .models import TaskSpec, CaseSpec


# === 算子目录检测 mixin ===

class OperatorDirMixin:
    """算子目录检测 mixin

    提供统一的算子目录判定逻辑。
    """

    REQUIRED_FILES = ['proto.yaml', 'cases.yaml', 'golden.py']

    def _is_operator_dir(self, dir_path: Path) -> bool:
        """检查目录是否为有效的算子目录"""
        for required_file in self.REQUIRED_FILES:
            if not (dir_path / required_file).exists():
                return False
        return True

    def _find_operator_dirs(self) -> List[Path]:
        """递归查找所有算子目录"""
        operator_dirs = []
        for proto_path in self.bench_root.rglob("proto.yaml"):
            op_dir = proto_path.parent
            if self._is_operator_dir(op_dir):
                operator_dirs.append(op_dir)
        return sorted(operator_dirs)


# === 任务加载器基类 ===

class TaskLoader(ABC):
    """任务加载器基类

    定义任务加载的核心接口。
    """

    @abstractmethod
    def list_tasks(self) -> List[TaskSpec]:
        """列出所有任务"""
        pass

    @abstractmethod
    def get_task(self, task_id: str) -> Optional[TaskSpec]:
        """获取指定任务"""
        pass

    def get_task_by_name(self, name: str) -> Optional[TaskSpec]:
        """按名称获取任务（默认实现）"""
        for task in self.list_tasks():
            if task.name.lower() == name.lower():
                return task
        return None

    @abstractmethod
    def get_statistics(self) -> Dict[str, Any]:
        """获取任务统计信息"""
        pass


# === 用例加载器基类 ===

class CaseLoader(ABC):
    """用例加载器基类

    定义用例加载的核心接口。
    """

    @abstractmethod
    def scan_all(self) -> List[CaseSpec]:
        """扫描所有用例"""
        pass

    @abstractmethod
    def scan_by_task(self, task_name: str) -> List[CaseSpec]:
        """扫描指定任务的用例"""
        pass

    @abstractmethod
    def get_statistics(self) -> Dict[str, Any]:
        """获取用例统计信息"""
        pass


# === Golden 加载器基类 ===

class GoldenLoaderBase(ABC):
    """Golden 函数加载器抽象基类

    定义 Golden 参考实现加载的核心接口。
    """

    @abstractmethod
    def get_golden_function(self, task_id: str) -> Callable:
        """获取 golden 参考实现函数

        Args:
            task_id: 任务标识

        Returns:
            Callable: golden 函数，可直接调用

        Raises:
            ImportError: Golden 模块不存在
            AttributeError: Golden 函数不存在
        """
        pass

    @abstractmethod
    def get_input_function(self, task_id: str) -> Optional[Callable]:
        """获取输入生成函数（可选）

        Args:
            task_id: 任务标识

        Returns:
            Optional[Callable]: get_inputs 函数，或 None
        """
        pass

    def get_operator_dir(self, task_id: str) -> str:
        """获取任务目录路径（可选实现）

        Args:
            task_id: 任务标识

        Returns:
            str: 任务目录路径
        """
        return ""