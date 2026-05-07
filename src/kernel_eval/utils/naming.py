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
命名转换工具

职责：
1. PascalCase → snake_case 转换
2. 生成多个 snake_case 候选形式（用于模糊匹配）

统一实现，消除 golden_loader.py, operator_loader.py, evaluator.py 中的重复定义。
"""

import re
from typing import List


def camel_to_snake(name: str) -> str:
    """将 PascalCase 名称转换为 snake_case

    处理规则：
    1. 数字+大写边界: 3D -> _3_D（避免重复下划线）
    2. 字母+数字边界: er3D -> er_3D
    3. 大写字母+小写字母边界: HelloWorld -> Hello_World

    Examples:
        Mish -> mish
        GridSampler3D -> grid_sampler_3d
        ROIAlign -> roi_align
        AdaptiveAvgPool3D -> adaptive_avg_pool3_d
    """
    # 处理字母+数字的边界: er3D -> er_3D
    s = re.sub(r'([a-zA-Z])([0-9])', r'\1_\2', name)
    # 处理数字+大写的组合: 3D -> _3_D
    s = re.sub(r'([0-9])([A-Z])', r'\1_\2', s)
    # 处理小写字母+大写字母组合
    s = re.sub(r'(.)([A-Z][a-z]+)', r'\1_\2', s)
    # 处理其他大写字母边界
    return re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', s).lower()


def snake_case_candidates(name: str) -> List[str]:
    """生成多个 plausible snake_case 形式用于模糊匹配

    cann_bench 模块命名不一致，有些使用 pool3_d（数字粘在字母后），
    有些使用 sampler_3d（数字前有下划线），acronym 如 ROIAlign/NMS 也需要特殊处理。

    Examples:
        GridSampler3D -> ['grid_sampler3_d', 'grid_sampler_3d', 'grid_sampler_3_d']
        ROIAlign -> ['roi_align', 'roi_align']
        NMS -> ['nms', 'nms']
        Mish -> ['mish', 'mish']

    Returns:
        List[str]: 按优先级排列的候选形式（去重）
    """
    candidates: List[str] = []

    # V1: naive - 每个大写字母前插入下划线
    # Mish -> mish, AdaptiveAvgPool3D -> adaptive_avg_pool3_d
    v1 = re.sub(r"([A-Z])", r"_\1", name).lower().lstrip("_")
    v1 = re.sub(r"_{2,}", "_", v1)
    candidates.append(v1)

    # V2: acronym-aware - 保持连续大写字母在一起
    # ROIAlign -> roi_align, NMS -> nms
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
    s = re.sub(r"([a-z\d])([A-Z])", r"\1_\2", s)
    v2 = s.lower()
    if v2 not in candidates:
        candidates.append(v2)

    # V3: V2 + 小写字母和数字之间插入下划线
    # grid_sampler3_d -> grid_sampler_3_d
    v3 = re.sub(r"([a-z])(\d)", r"\1_\2", s).lower()
    if v3 not in candidates:
        candidates.append(v3)

    # V4: V3 但数字和字母不分开
    # grid_sampler_3_d -> grid_sampler3d
    v4 = re.sub(r"(\d)_([a-z])", r"\1\2", v3)
    if v4 not in candidates:
        candidates.append(v4)

    return candidates