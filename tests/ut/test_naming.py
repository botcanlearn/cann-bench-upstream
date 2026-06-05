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
命名转换模块单元测试

测试对象：kernel_eval.utils.naming
核心功能：
1. camel_to_snake - PascalCase → snake_case
2. snake_case_candidates - 多候选形式生成
"""

import pytest
from kernel_eval.utils.naming import camel_to_snake, snake_case_candidates


class TestCamelToSnake:
    """camel_to_snake 函数测试"""

    def test_simple_name(self):
        """简单名称转换"""
        assert camel_to_snake("Exp") == "exp"
        assert camel_to_snake("Add") == "add"
        assert camel_to_snake("Mul") == "mul"

    def test_multi_word_name(self):
        """多单词名称转换"""
        assert camel_to_snake("AdaptiveAvgPool") == "adaptive_avg_pool"
        assert camel_to_snake("BatchNorm") == "batch_norm"
        assert camel_to_snake("Softmax") == "softmax"

    def test_with_digit_suffix(self):
        """带数字后缀的名称"""
        assert camel_to_snake("AddN") == "add_n"
        assert camel_to_snake("TopK") == "top_k"

    def test_with_digit_in_middle(self):
        """数字在中间的名称"""
        # GridSampler3D -> grid_sampler_3_d (数字和字母分开)
        assert camel_to_snake("GridSampler3D") == "grid_sampler_3_d"
        # Pool3D -> pool_3_d
        assert camel_to_snake("Pool3D") == "pool_3_d"

    def test_acronym_handling(self):
        """缩写词处理"""
        # ROIAlign -> roi_align (ROI 被视为一个整体)
        assert camel_to_snake("ROIAlign") == "roi_align"
        # NMS -> nms
        assert camel_to_snake("NMS") == "nms"

    def test_complex_cases(self):
        """复杂边界情况"""
        # Mish -> mish
        assert camel_to_snake("Mish") == "mish"
        # SiLU -> si_lu
        assert camel_to_snake("SiLU") == "si_lu"
        # GELU -> gelu (连续大写字母整体转换)
        assert camel_to_snake("GELU") == "gelu"

    def test_empty_string(self):
        """空字符串"""
        assert camel_to_snake("") == ""

    def test_already_snake_case(self):
        """已经是 snake_case 的名称"""
        assert camel_to_snake("add") == "add"
        assert camel_to_snake("batch_norm") == "batch_norm"

    def test_single_letter(self):
        """单字母"""
        assert camel_to_snake("A") == "a"
        assert camel_to_snake("X") == "x"


class TestSnakeCaseCandidates:
    """snake_case_candidates 函数测试"""

    def test_simple_name_candidates(self):
        """简单名称的候选形式"""
        candidates = snake_case_candidates("Add")
        assert "add" in candidates
        assert len(candidates) >= 1

    def test_multi_word_candidates(self):
        """多单词名称的候选形式"""
        candidates = snake_case_candidates("AdaptiveAvgPool")
        # 应包含 adaptive_avg_pool
        assert "adaptive_avg_pool" in candidates

    def test_digit_handling_candidates(self):
        """带数字名称的候选形式"""
        candidates = snake_case_candidates("GridSampler3D")
        # 应包含多种形式
        assert "grid_sampler_3_d" in candidates
        # 可能也包含其他形式如 grid_sampler3d
        assert len(candidates) >= 2

    def test_acronym_candidates(self):
        """缩写词的候选形式"""
        candidates = snake_case_candidates("ROIAlign")
        assert "roi_align" in candidates

    def test_returns_unique_candidates(self):
        """返回的候选形式应唯一"""
        candidates = snake_case_candidates("NMS")
        # 检查无重复
        assert len(candidates) == len(set(candidates))

    def test_order_by_priority(self):
        """候选形式按优先级排列"""
        candidates = snake_case_candidates("GridSampler3D")
        # 第一个候选为 v1 naive 形式
        assert candidates[0] == "grid_sampler3_d"

    def test_empty_string_candidates(self):
        """空字符串的候选形式"""
        candidates = snake_case_candidates("")
        assert candidates == [""]


class TestNamingIntegration:
    """命名转换集成测试"""

    def test_consistency_between_functions(self):
        """camel_to_snake 和 snake_case_candidates 的一致性"""
        # 对于简单名称，camel_to_snake 结果应包含在 candidates 中
        test_names = ["Add", "Exp", "Mul", "Softmax"]
        for name in test_names:
            single_result = camel_to_snake(name)
            candidates = snake_case_candidates(name)
            assert single_result in candidates

    def test_real_operator_names(self):
        """真实算子名称测试"""
        # 来自 CANN-Bench 的真实算子名
        real_names = [
            "Exp", "Sqrt", "Add", "Mul", "Div",
            "Softmax", "BatchNorm", "LayerNorm",
            "AdaptiveAvgPool", "AdaptiveAvgPool3D",
            "GridSampler", "GridSampler3D",
            "ROIAlign", "NMS",
            "Mish", "GELU", "SiLU",
            "TopK", "AddN",
        ]
        for name in real_names:
            result = camel_to_snake(name)
            # 结果应为小写
            assert result == result.lower()
            # 结果应包含下划线（多单词情况）或为单单词
            candidates = snake_case_candidates(name)
            assert len(candidates) >= 1