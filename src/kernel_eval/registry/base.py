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
注册表泛型基类

BaseRegistry: 为各模块提供统一的 register / get / list_all 接口

Why: 减少各 Registry 的样板代码
"""

from typing import Dict, Generic, List, Optional, TypeVar

T = TypeVar('T')


class BaseRegistry(Generic[T]):
    """注册表泛型基类"""

    _items: Dict[str, T] = {}

    @classmethod
    def register(cls, name: str, item: T) -> None:
        """注册一个条目"""
        if name in cls._items:
            raise ValueError(f"'{name}' 已注册")
        cls._items[name] = item

    @classmethod
    def get(cls, name: str) -> Optional[T]:
        """按名称获取条目"""
        return cls._items.get(name)

    @classmethod
    def list_all(cls) -> List[str]:
        """列出所有已注册名称"""
        return list(cls._items.keys())

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """检查名称是否已注册"""
        return name in cls._items

    @classmethod
    def clear(cls) -> None:
        """清空注册表"""
        cls._items.clear()