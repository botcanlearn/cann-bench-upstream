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
AI 算子匹配器注册表

OperatorMatcherRegistry: OperatorMatcher 注册表

Why: 支持多评测体系扩展，统一管理 AI 算子匹配器
"""

from typing import Any, Dict, List, Optional, Type

from ..base.matcher import OperatorMatcherBase


class OperatorMatcherRegistry:
    """AI 算子匹配器注册表"""

    _matchers: Dict[str, Type[OperatorMatcherBase]] = {}
    DEFAULT_EVAL_SYSTEM = "cann"

    @classmethod
    def register(cls, name: str, matcher_cls: Type[OperatorMatcherBase]) -> None:
        """注册 AI 算子匹配器"""
        if name in cls._matchers:
            raise ValueError(f"Operator matcher '{name}' 已注册")
        cls._matchers[name] = matcher_cls

    @classmethod
    def get(cls, name: str = None, operator_loader: Any = None) -> OperatorMatcherBase:
        """获取 AI 算子匹配器实例"""
        if name is None:
            name = cls.DEFAULT_EVAL_SYSTEM

        matcher_cls = cls._matchers.get(name)
        if matcher_cls is None:
            registered = cls.list_matchers()
            raise ValueError(f"Operator matcher '{name}' 未注册，已注册: {registered}")

        if operator_loader is None:
            raise ValueError(f"operator_loader 参数未提供")

        return matcher_cls(operator_loader)

    @classmethod
    def get_cls(cls, name: str = None) -> Optional[Type[OperatorMatcherBase]]:
        """获取匹配器类（不实例化）"""
        if name is None:
            name = cls.DEFAULT_EVAL_SYSTEM
        return cls._matchers.get(name)

    @classmethod
    def list_matchers(cls) -> List[str]:
        """列出已注册的 AI 算子匹配器"""
        return list(cls._matchers.keys())

    @classmethod
    def list_all(cls) -> List[str]:
        """列出所有已注册名称"""
        return cls.list_matchers()

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """检查匹配器是否已注册"""
        return name in cls._matchers

    @classmethod
    def clear(cls) -> None:
        """清空注册表"""
        cls._matchers.clear()


def get_operator_matcher(eval_system: str = "cann", operator_loader: Any = None) -> OperatorMatcherBase:
    """获取 AI 算子匹配器实例"""
    return OperatorMatcherRegistry.get(eval_system, operator_loader)