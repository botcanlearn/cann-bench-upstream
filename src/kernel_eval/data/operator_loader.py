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
算子定义加载器

职责：
1. 解析 proto.yaml 文件
2. 提供算子schema、attrs、inputs、outputs信息
3. 支持任意目录结构，递归扫描proto.yaml
"""

import logging
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml

from ..config import get_project_root
from ..utils.naming import camel_to_snake

logger = logging.getLogger(__name__)


@dataclass
class AttrInfo:
    """算子属性信息"""
    name: str
    type: str
    default: Any = None
    description: str = ""


@dataclass
class TensorInfo:
    """张量信息"""
    name: str
    description: str = ""
    dtype: List[str] = field(default_factory=list)
    compare: bool = True  # 是否参与精度对比，默认True


@dataclass
class OperatorInfo:
    """算子定义信息"""
    name: str
    rel_path: str       # 相对路径，如 "level2/scatter"
    category: str = ""
    difficulty: str = ""  # 保留难度标识（如"L2"），仅作信息展示
    formula: str = ""
    description: str = ""
    shape_support: str = ""
    note: str = ""
    precision_thresholds: Dict[str, float] = field(default_factory=dict)  # 自定义精度阈值
    attrs: List[AttrInfo] = field(default_factory=list)
    inputs: List[TensorInfo] = field(default_factory=list)
    outputs: List[TensorInfo] = field(default_factory=list)
    schema: str = ""
    dir_name: str = ""  # 实际目录名

    def get_function_name(self) -> str:
        """从schema解析函数名"""
        if self.schema:
            match = re.match(r'^(\w+)\s*\(', self.schema.strip())
            if match:
                return match.group(1)
        return self.name.lower()


class OperatorLoader:
    """算子定义加载器"""

    # 算子目录必须包含的文件
    REQUIRED_FILES = ['proto.yaml', 'cases.yaml', 'golden.py']

    def __init__(self, bench_root: str = None):
        if bench_root:
            self.bench_root = Path(bench_root)
        else:
            self.bench_root = get_project_root() / "tasks"

        self._cache: Dict[str, OperatorInfo] = {}

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

    def get_operator(self, rel_path: str) -> OperatorInfo:
        """获取算子定义信息

        Args:
            rel_path: 相对路径，如 "level2/scatter"
        """
        if rel_path in self._cache:
            return self._cache[rel_path]

        op_dir = self.bench_root / rel_path
        proto_path = op_dir / "proto.yaml"

        if not proto_path.exists():
            raise FileNotFoundError(f"proto.yaml不存在: {proto_path}")

        with open(proto_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        if not data or 'operator' not in data:
            raise ValueError(f"proto.yaml格式错误: {proto_path}")

        op_data = data['operator']
        op_info = self._parse_operator(op_data, rel_path, op_dir.name)
        self._cache[rel_path] = op_info
        return op_info

    def get_operator_by_name(self, operator: str) -> Optional[OperatorInfo]:
        """按算子名称获取算子定义（遍历查找）"""
        for op_dir in self._find_operator_dirs():
            proto_path = op_dir / "proto.yaml"
            try:
                with open(proto_path, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
                if data and 'operator' in data:
                    op_name = data['operator'].get('name', '')
                    if op_name.lower() == operator.lower():
                        rel_path = str(op_dir.relative_to(self.bench_root))
                        return self.get_operator(rel_path)
            except Exception as e:
                logger.warning("Failed to parse proto.yaml at %s: %s", proto_path, e)
        return None

    def _parse_operator(self, data: Dict, rel_path: str, dir_name: str) -> OperatorInfo:
        """解析算子定义"""
        # 解析attrs
        attrs = []
        for attr_data in data.get('attrs', []) or []:
            attrs.append(AttrInfo(
                name=attr_data.get('name', ''),
                type=attr_data.get('type', ''),
                default=attr_data.get('default'),
                description=attr_data.get('description', '')
            ))

        # 解析inputs
        inputs = []
        for input_data in data.get('inputs', []) or []:
            dtype_list = input_data.get('dtype', [])
            if isinstance(dtype_list, str):
                dtype_list = [dtype_list]
            inputs.append(TensorInfo(
                name=input_data.get('name', ''),
                description=input_data.get('description', ''),
                dtype=dtype_list
            ))

        # 解析outputs
        outputs = []
        for output_data in data.get('outputs', []) or []:
            dtype_list = output_data.get('dtype', [])
            if isinstance(dtype_list, str):
                dtype_list = [dtype_list]
            outputs.append(TensorInfo(
                name=output_data.get('name', ''),
                description=output_data.get('description', ''),
                dtype=dtype_list,
                compare=output_data.get('compare', True)
            ))

        # 解析自定义精度阈值
        precision_thresholds = data.get('precision_thresholds', {}) or {}

        return OperatorInfo(
            name=data.get('name', ''),
            rel_path=rel_path,
            category=data.get('category', ''),
            difficulty=data.get('difficulty', ''),
            formula=data.get('formula', ''),
            description=data.get('description', ''),
            shape_support=data.get('shape_support', ''),
            note=data.get('note', ''),
            precision_thresholds=precision_thresholds,
            attrs=attrs,
            inputs=inputs,
            outputs=outputs,
            schema=data.get('schema', ''),
            dir_name=dir_name
        )

    def list_operators(self) -> List[OperatorInfo]:
        """列出所有算子"""
        operators = []
        for op_dir in self._find_operator_dirs():
            proto_path = op_dir / "proto.yaml"
            try:
                rel_path = str(op_dir.relative_to(self.bench_root))
                op_info = self.get_operator(rel_path)
                operators.append(op_info)
            except Exception as e:
                logger.warning("Failed to load operator from %s: %s", proto_path, e)

        return operators

    def get_statistics(self) -> Dict[str, Any]:
        """获取算子统计"""
        operators = self.list_operators()
        categories = {}
        for op in operators:
            cat = op.category or 'Unknown'
            categories[cat] = categories.get(cat, 0) + 1
        return {
            'total': len(operators),
            'operators': [op.name for op in operators],
            'rel_paths': [op.rel_path for op in operators],
            'categories': categories
        }