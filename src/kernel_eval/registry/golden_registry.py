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
Golden 加载器注册表

GoldenLoaderRegistry: GoldenLoader 注册表

Why: 支持多评测体系扩展，统一管理 Golden 加载器
"""

from typing import Dict, List, Optional, Type

from ..base.loaders import GoldenLoaderBase
from .base import BaseRegistry


class GoldenLoaderRegistry(BaseRegistry[Type[GoldenLoaderBase]]):
    """Golden 加载器注册表"""

    _items: Dict[str, Type[GoldenLoaderBase]] = {}
    DEFAULT_EVAL_SYSTEM = "cann"

    @classmethod
    def get(cls, name: str = None, **kwargs) -> GoldenLoaderBase:
        """获取 Golden 加载器实例"""
        if name is None:
            name = cls.DEFAULT_EVAL_SYSTEM

        loader_cls = cls._items.get(name)
        if loader_cls is None:
            registered = cls.list_loaders()
            raise ValueError(f"Golden loader '{name}' 未注册，已注册: {registered}")

        return loader_cls(**kwargs)

    @classmethod
    def get_cls(cls, name: str = None) -> Optional[Type[GoldenLoaderBase]]:
        """获取 Golden 加载器类（不实例化）"""
        if name is None:
            name = cls.DEFAULT_EVAL_SYSTEM
        return cls._items.get(name)

    @classmethod
    def list_loaders(cls) -> List[str]:
        """列出已注册的 Golden 加载器"""
        return cls.list_all()


def get_golden_loader(eval_system: str = "cann", **kwargs) -> GoldenLoaderBase:
    """获取 Golden 加载器实例"""
    return GoldenLoaderRegistry.get(eval_system, **kwargs)