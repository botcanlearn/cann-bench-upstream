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
评分方案注册表

ScoringSchemeRegistry: ScoringScheme 注册表

Why: 支持多评测体系扩展，统一管理评分方案
"""

from typing import Dict, List, Optional

from ..base.scoring import ScoringScheme
from .base import BaseRegistry


class ScoringSchemeRegistry(BaseRegistry[ScoringScheme]):
    """评分方案注册表"""

    _items: Dict[str, ScoringScheme] = {}

    @classmethod
    def list_schemes(cls) -> List[str]:
        """列出已注册的评分方案"""
        return cls.list_all()

    @classmethod
    def get_default(cls) -> Optional[ScoringScheme]:
        """获取默认评分方案"""
        return cls._items.get('cann')


def get_scoring_scheme(name: str = None) -> ScoringScheme:
    """获取评分方案实例"""
    if name is None:
        name = 'cann'
    scheme = ScoringSchemeRegistry.get(name)
    if scheme is None:
        raise ValueError(f"评分方案 '{name}' 未注册")
    return scheme