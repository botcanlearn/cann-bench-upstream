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
精度判断器注册表

CheckerRegistry: CorrectnessChecker 注册表

Why: 支持多评测体系扩展，统一管理精度判断器
"""

from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..base.checker import CorrectnessChecker


_CHECKER_REGISTRY: Dict[str, "CorrectnessChecker"] = {}


def register_correctness_checker(name: str):
    """判断器注册装饰器"""
    def decorator(cls):
        if name in _CHECKER_REGISTRY:
            raise ValueError(f"CorrectnessChecker '{name}' already registered")
        instance = cls()
        _CHECKER_REGISTRY[name] = instance
        return cls
    return decorator


def get_correctness_checker(name: str) -> Optional["CorrectnessChecker"]:
    """获取已注册的判断器"""
    return _CHECKER_REGISTRY.get(name)


def list_correctness_checkers() -> List[str]:
    """列出所有已注册的判断器名称"""
    return list(_CHECKER_REGISTRY.keys())


def is_checker_registered(name: str) -> bool:
    """检查判断器是否已注册"""
    return name in _CHECKER_REGISTRY


def clear_checker_registry() -> None:
    """清空注册表（用于测试）"""
    _CHECKER_REGISTRY.clear()


def get_registry_size() -> int:
    """获取注册数量"""
    return len(_CHECKER_REGISTRY)


def get_checker_info(name: str) -> Optional[Dict[str, str]]:
    """获取判断器信息"""
    checker = get_correctness_checker(name)
    if checker:
        return {
            "name": checker.get_name(),
            "description": checker.get_description() if hasattr(checker, 'get_description') else "",
        }
    return None


class CheckerRegistry:
    """判断器注册表管理类"""

    @staticmethod
    def register(name: str, checker: "CorrectnessChecker") -> None:
        """手动注册判断器"""
        if name in _CHECKER_REGISTRY:
            raise ValueError(f"CorrectnessChecker '{name}' already registered")
        _CHECKER_REGISTRY[name] = checker

    @staticmethod
    def unregister(name: str) -> bool:
        """取消注册"""
        if name in _CHECKER_REGISTRY:
            del _CHECKER_REGISTRY[name]
            return True
        return False

    @staticmethod
    def get_all() -> Dict[str, "CorrectnessChecker"]:
        """获取所有已注册的判断器"""
        return dict(_CHECKER_REGISTRY)

    @staticmethod
    def list_all() -> List[str]:
        """列出所有已注册名称"""
        return list_correctness_checkers()

    @staticmethod
    def is_registered(name: str) -> bool:
        """检查是否已注册"""
        return is_checker_registered(name)

    @staticmethod
    def clear() -> None:
        """清空注册表"""
        clear_checker_registry()