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
Golden函数动态导入器

职责：
1. 根据rel_path定位golden模块
2. 动态导入golden函数
"""

import importlib
import importlib.util
import logging
import re
from pathlib import Path
from typing import Callable, Dict, List, Optional

import yaml

from ..config import get_project_root
from ..utils.naming import camel_to_snake

logger = logging.getLogger(__name__)


class GoldenLoader:
    """Golden函数动态导入器"""

    def __init__(self, bench_root: str = None):
        if bench_root:
            self.bench_root = Path(bench_root)
        else:
            self.bench_root = get_project_root() / "kernel_bench"
        self._func_cache: Dict[str, str] = {}
        self._module_cache: Dict[str, object] = {}

    def _load_module(self, rel_path: str):
        """导入 golden 模块（带缓存，避免每个 case 重复磁盘 I/O）"""
        if rel_path in self._module_cache:
            return self._module_cache[rel_path]

        module_path = self.bench_root / rel_path / "golden.py"
        if not module_path.exists():
            raise ImportError(f"Golden模块不存在: {module_path}")

        # 使用 rel_path 转换为模块名
        # 例如 "level2/scatter" -> "kernel_bench.level2.scatter"
        module_name = f"kernel_bench.{rel_path.replace('/', '.')}"
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        self._module_cache[rel_path] = module
        return module

    def get_golden_function(self, rel_path: str) -> Callable:
        """获取golden函数

        Args:
            rel_path: 相对路径，如 "level2/scatter"
        """
        module = self._load_module(rel_path)

        # 从 proto.yaml 获取函数名
        func_name = self._get_function_name(rel_path)
        if not hasattr(module, func_name):
            # 尝试算子名小写
            func_name = self._get_operator_name(rel_path).lower()
            if not hasattr(module, func_name):
                raise AttributeError(
                    f"模块 kernel_bench.{rel_path.replace('/', '.')} 中找不到函数: {func_name}")

        return getattr(module, func_name)

    def _get_operator_name(self, rel_path: str) -> str:
        """从 proto.yaml 获取算子名称"""
        proto_path = self.bench_root / rel_path / "proto.yaml"
        if proto_path.exists():
            try:
                with open(proto_path, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
                if data and 'operator' in data:
                    return data['operator'].get('name', '')
            except Exception as e:
                logger.warning("Failed to parse proto.yaml at %s: %s", proto_path, e)
        return Path(rel_path).name  # fallback to dir name

    def _get_function_name(self, rel_path: str) -> str:
        """从proto.yaml获取函数名"""
        if rel_path in self._func_cache:
            return self._func_cache[rel_path]

        proto_path = self.bench_root / rel_path / "proto.yaml"
        if proto_path.exists():
            try:
                with open(proto_path, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
                if data and 'operator' in data:
                    schema = data['operator'].get('schema', '')
                    match = re.match(r'^(\w+)\s*\(', schema.strip())
                    if match:
                        self._func_cache[rel_path] = match.group(1)
                        return match.group(1)
            except Exception as e:
                logger.warning("Failed to parse proto.yaml at %s: %s", proto_path, e)

        # fallback to operator name lower
        return self._get_operator_name(rel_path).lower()

    def get_operator_dir(self, rel_path: str) -> Path:
        """获取算子目录路径"""
        return self.bench_root / rel_path

    def get_input_function(self, rel_path: str) -> Optional[Callable]:
        """获取 get_input 函数（可选）

        检查 golden.py 是否实现了 get_input() 函数。
        如果存在则返回该函数，否则返回 None。
        """
        module = self._load_module(rel_path)

        if hasattr(module, 'get_input'):
            return getattr(module, 'get_input')

        return None

    def get_golden_by_operator_name(self, operator: str) -> Callable:
        """按算子名称查找golden函数（遍历查找）"""
        # 递归查找 proto.yaml
        for proto_path in self.bench_root.rglob("proto.yaml"):
            try:
                with open(proto_path, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
                if data and 'operator' in data:
                    op_name = data['operator'].get('name', '')
                    if op_name.lower() == operator.lower():
                        op_dir = proto_path.parent
                        rel_path = str(op_dir.relative_to(self.bench_root))
                        return self.get_golden_function(rel_path)
            except Exception as e:
                logger.warning("Failed to parse proto.yaml at %s: %s", proto_path, e)
        raise ImportError(f"未找到算子 {operator} 的golden函数")