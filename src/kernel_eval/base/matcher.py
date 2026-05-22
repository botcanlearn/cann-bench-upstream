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
AI 算子匹配器基类

OperatorMatcherBase: AI 算子函数加载器抽象基类

Why: 为不同评测体系提供统一的 AI 算子加载接口
"""

from abc import ABC, abstractmethod
from typing import Callable, Optional


class OperatorMatcherBase(ABC):
    """AI 算子匹配器抽象基类

    定义 AI 算子函数加载的核心接口。
    """

    @abstractmethod
    def load_ai_operator(self, operator_name: str) -> Callable:
        """加载 AI 生成的算子函数

        Args:
            operator_name: 算子名称

        Returns:
            Callable: AI 算子函数

        Raises:
            AttributeError: 无法找到算子
        """
        pass

    @abstractmethod
    def find_operator_info(self, operator_name: str) -> Optional[object]:
        """查找算子定义信息

        Args:
            operator_name: 算子名称

        Returns:
            TaskSpec 或 None
        """
        pass

    @abstractmethod
    def clear_cache(self) -> None:
        """清空算子缓存"""
        pass

    def find_operator_info_by_snake(self, snake_name: str) -> Optional[object]:
        """通过 snake_case 名称反查算子信息（可选实现）

        Args:
            snake_name: snake_case 形式的算子名

        Returns:
            TaskSpec 或 None
        """
        # 默认实现：转回 PascalCase 尝试查找
        parts = snake_name.split('_')
        camel_name = ''.join(p.capitalize() for p in parts)
        return self.find_operator_info(camel_name)