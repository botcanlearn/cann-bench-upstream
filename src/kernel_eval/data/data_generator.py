#!/usr/bin/python3
# coding=utf-8

# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
6# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""
测试数据生成器

职责：
1. 根据shape和dtype生成输入张量
2. 根据value_range填充数据
3. 支持确定性种子（通过 torch.Generator）确保评测可复现
"""

from typing import Any, List, Optional, Union

import torch

from ..utils.dtype_mapper import str_to_torch_dtype, is_float_dtype, is_int_dtype


class DataGenerator:
    """测试数据生成器"""

    def __init__(self, seed: int = None):
        if seed is not None:
            torch.manual_seed(seed)

    def generate_input_tensor(self, shape: List[int], dtype: str, value_range: Any = None,
                              generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """生成输入张量

        Args:
            shape: 张量形状
            dtype: 数据类型字符串
            value_range: 值范围 [min, max] 或 [[min1, max1], ...]
            generator: torch.Generator 用于确定性随机数生成（可选）
        """
        # 支持 shape 为 None 或包含 None 的情况，表示不使用该参数
        if shape is None:
            return None
        if isinstance(shape, list) and None in shape:
            return None

        torch_dtype = str_to_torch_dtype(dtype)
        min_val, max_val = self._parse_range(value_range, dtype)

        if is_float_dtype(dtype):
            # 特殊值（NaN/inf）必须优先处理，避免 'nan' == 'nan' 走全量填充路径
            if isinstance(min_val, str) or isinstance(max_val, str):
                return self._gen_float(shape, torch_dtype, min_val, max_val, generator=generator)
            if self._is_special_float(min_val) or self._is_special_float(max_val):
                return self._gen_float(shape, torch_dtype, min_val, max_val, generator=generator)
            if min_val == max_val:
                return torch.full(shape, float(min_val), dtype=torch_dtype)
            return self._gen_float(shape, torch_dtype, float(min_val), float(max_val), generator=generator)
        elif is_int_dtype(dtype):
            if min_val == max_val:
                return torch.full(shape, int(min_val), dtype=torch_dtype)
            # torch.randint 支持 generator 参数（generator=None 时等同默认 RNG），
            # 直接透传即可保证确定性，无需 rand 缩放
            return torch.randint(
                int(min_val), int(max_val) + 1, shape,
                generator=generator, dtype=torch_dtype,
            )
        else:
            return torch.zeros(shape, dtype=torch_dtype)

    def generate_input_tensors_from_case(self, input_shapes: List, dtypes: List, value_ranges: List,
                                         seed: Optional[int] = None) -> List:
        """根据用例信息生成输入数据

        假设输入已由 CaseLoader 规范化：
        - input_shapes: 嵌套列表 [[shape1], [shape2], ...]
        - dtypes: 列表 [dtype1, dtype2, ...]
        - value_ranges: 列表 [[min1, max1], [min2, max2], ...]

        Args:
            input_shapes: 输入形状列表
            dtypes: 数据类型列表
            value_ranges: 值范围列表
            seed: 确定性种子（可选）。设置后使用 torch.Generator 确保可复现。
                  None 表示使用全局随机状态（不保证可复现）。
        """
        input_tensors = []
        num_inputs = self._count_inputs(input_shapes)

        # 创建 local Generator 用于确定性种子
        gen = None
        if seed is not None:
            gen = torch.Generator()
            gen.manual_seed(seed)

        # 扩展dtypes
        if not isinstance(dtypes, list) or (dtypes and isinstance(dtypes[0], str) and len(dtypes) < num_inputs):
            if isinstance(dtypes, str):
                dtypes = [dtypes]
            elif not isinstance(dtypes, list):
                dtypes = ['float32']
        if isinstance(dtypes, list) and len(dtypes) < num_inputs:
            dtypes = dtypes + [dtypes[-1]] * (num_inputs - len(dtypes)) if dtypes else ['float32'] * num_inputs

        # 规范化 value_ranges: 区分单输入 [min, max] 和多输入 [[min1, max1], [min2, max2], ...]
        value_ranges = self._normalize_value_ranges(value_ranges, num_inputs)

        self._generate_tensors(input_shapes, dtypes, value_ranges, input_tensors, generator=gen)

        return input_tensors

    def _normalize_value_ranges(self, value_ranges: List, num_inputs: int) -> List:
        """规范化 value_ranges 为每个输入一个范围"""
        if not value_ranges:
            return [None] * num_inputs

        # 判断是单输入 [min, max] 还是多输入 [[min1, max1], ...]
        first_item = value_ranges[0]
        is_single_range = not isinstance(first_item, list)

        if is_single_range:
            # 单输入算子: value_range 就是 [min, max]
            return [value_ranges] + [None] * (num_inputs - 1)
        else:
            # 多输入算子: value_ranges 是 [[min1, max1], [min2, max2], ...]
            if len(value_ranges) < num_inputs:
                value_ranges = value_ranges + [value_ranges[-1]] * (num_inputs - len(value_ranges))
            return value_ranges

    def _parse_range(self, value_range: Any, dtype: str) -> tuple:
        """解析值范围"""
        if value_range is None:
            return (0.0, 1.0) if is_float_dtype(dtype) else (0, 100)

        if isinstance(value_range, list):
            if len(value_range) >= 2:
                return (value_range[0], value_range[1])
            elif len(value_range) == 1:
                return (value_range[0], value_range[0])

        return (value_range, value_range)

    @staticmethod
    def _is_special_float(val: Any) -> bool:
        """检测 float NaN/inf 特殊值"""
        import math
        return isinstance(val, float) and (math.isnan(val) or math.isinf(val))

    def _is_tensor_list(self, shape_item: Any) -> bool:
        """判断是否为张量列表"""
        return (isinstance(shape_item, list) and shape_item and
                isinstance(shape_item[0], list) and shape_item[0] and
                isinstance(shape_item[0][0], int))

    def _gen_float(self, shape: List[int], dtype: torch.dtype, min_val: float, max_val: float,
                   generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """生成浮点张量

        Args:
            shape: 张量形状
            dtype: torch 数据类型
            min_val: 最小值
            max_val: 最大值
            generator: torch.Generator 用于确定性随机数生成（可选）
        """
        # 处理特殊值（字符串形式或 float NaN/inf）
        if isinstance(min_val, str) or isinstance(max_val, str):
            return self._gen_special(shape, dtype, min_val, max_val, generator=generator)
        if self._is_special_float(min_val) or self._is_special_float(max_val):
            return self._gen_special(shape, dtype, min_val, max_val, generator=generator)

        # 使用 torch.finfo 获取 dtype 的精确范围
        finfo = torch.finfo(dtype)
        dmin = finfo.min
        dmax = finfo.max

        # 裁剪到 dtype 可表示范围
        min_val = max(min_val, dmin)
        max_val = min(max_val, dmax)

        # 统一使用 torch.rand + 缩放 路径，支持 generator 参数
        # 无论 range_val 是否超过 dmax，都通过 float64 中间计算避免溢出
        range_val = max_val - min_val
        rand_f64 = torch.rand(shape, dtype=torch.float64, generator=generator)
        tensor_f64 = rand_f64 * range_val + min_val
        # clamp 确保值严格在 dtype 范围内，避免转换溢出
        tensor_f64 = torch.clamp(tensor_f64, dmin, dmax)
        return tensor_f64.to(dtype)

    def _gen_special(self, shape: List[int], dtype: torch.dtype, min_val: Any, max_val: Any,
                     generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """生成包含特殊值的张量

        Args:
            shape: 张量形状
            dtype: torch 数据类型
            min_val: 最小值（可能是特殊值字符串或 float NaN/inf）
            max_val: 最大值（可能是特殊值字符串或 float NaN/inf）
            generator: torch.Generator 用于确定性随机数生成（可选）
        """
        import math

        # 转换特殊值字符串
        def to_float(v):
            if v == 'inf': return float('inf')
            if v == '-inf': return float('-inf')
            if v == 'nan': return float('nan')
            return float(v)

        min_f = to_float(min_val) if isinstance(min_val, str) else min_val
        max_f = to_float(max_val) if isinstance(max_val, str) else max_val

        # 处理 NaN 范围：混合 NaN 与正常随机值，避免全 NaN 触发 kernel 边界 bug
        if math.isnan(min_f) and math.isnan(max_f):
            if generator is not None:
                # 确定性路径：使用 torch.rand + generator 生成基础值
                rand_base = torch.rand(shape, dtype=torch.float32, generator=generator)
                tensor = (rand_base * 2 - 1).to(dtype)  # 简化的正态近似
            else:
                tensor = torch.randn(shape, dtype=torch.float32).to(dtype)
            # 使用确定性 seed 生成 NaN mask
            if generator is not None:
                nan_rand = torch.rand(shape, dtype=torch.float32, generator=generator)
                nan_mask = nan_rand < 0.5
            else:
                nan_mask = torch.rand(shape, dtype=torch.float32) < 0.5
            tensor[nan_mask] = float('nan')
            return tensor

        if generator is not None:
            rand_base = torch.rand(shape, dtype=torch.float32, generator=generator)
            tensor = (rand_base * 2 - 1).to(dtype)
        else:
            tensor = torch.randn(shape, dtype=torch.float32).to(dtype)

        # 在边界填充特殊值
        flat = tensor.flatten()
        n = max(1, len(flat) // 20)
        if min_f == float('-inf') or max_f == float('inf'):
            flat[:n] = float('-inf')
            flat[-n:] = float('inf')

        return tensor

    # ---- 嵌套结构支持（GroupedMatmul 等算子） ----

    def _count_inputs(self, shapes: Any) -> int:
        """统计输入张量总数（顶层列表长度）"""
        if isinstance(sizes := shapes, list):
            return len(sizes)
        return 1

    def _generate_tensors(self, shapes, dtypes, value_ranges, output: list,
                          generator: Optional[torch.Generator] = None):
        """递归生成张量，支持嵌套结构

        处理三种情况：
        1. 单个形状 [N, C, H, W] -> 单个 tensor
        2. TensorList [[shape1], [shape2], ...] 其中每个 shape_i 是 [dims...] -> 保持为 list
        3. 扁平形状列表 [[N,C,H,W], [N,C,H,W], ...] -> 每个元素一个 tensor

        Args:
            shapes: 形状列表
            dtypes: 数据类型列表
            value_ranges: 值范围列表
            output: 输出张量列表（追加）
            generator: torch.Generator 用于确定性随机数生成（可选）
        """
        if not isinstance(shapes, list):
            return

        # 单个形状 [N, C, H, W] —— 直接生成单个张量
        if shapes and isinstance(shapes[0], int):
            dtype_item = dtypes if isinstance(dtypes, str) else (
                dtypes[0] if isinstance(dtypes, list) and dtypes else 'float32'
            )
            vr_item = value_ranges if (
                isinstance(value_ranges, list) and len(value_ranges) == 2
                and not isinstance(value_ranges[0], list)
            ) else None
            output.append(self.generate_input_tensor(shapes, dtype_item, vr_item, generator=generator))
            return

        # 判断每个元素是否为 TensorList（嵌套）还是单个 shape
        for i, s in enumerate(shapes):
            d = dtypes[i] if isinstance(dtypes, list) and i < len(dtypes) else dtypes
            vr = value_ranges[i] if isinstance(value_ranges, list) and i < len(value_ranges) else value_ranges

            if s is None:
                # None 形状占位符
                output.append(None)
            elif isinstance(s, list) and s and isinstance(s[0], int):
                # s 是单个 shape [N, C, H, W] -> 单个 tensor
                dtype_item = d if isinstance(d, str) else (d[0] if isinstance(d, list) and d else 'float32')
                vr_item = vr if isinstance(vr, list) and len(vr) == 2 and not isinstance(vr[0], list) else None
                output.append(self.generate_input_tensor(s, dtype_item, vr_item, generator=generator))
            elif isinstance(s, list) and s and isinstance(s[0], list) and s[0] and isinstance(s[0][0], int):
                # s 是 TensorList [[shape1], [shape2], ...] 其中 shape_i 是 int list
                # 递归生成每个元素，保持为 list 结构
                sub_list = []
                sub_dtypes = d if isinstance(d, list) else d
                sub_vr = vr if isinstance(vr, list) and not (len(vr) == 2 and isinstance(vr[0], (int, float))) else None
                for j, sub_shape in enumerate(s):
                    sub_d = sub_dtypes[j] if isinstance(sub_dtypes, list) and j < len(sub_dtypes) else sub_dtypes
                    if sub_vr and isinstance(sub_vr, list) and j < len(sub_vr):
                        item_vr = sub_vr[j]
                    else:
                        item_vr = sub_vr
                    sub_list.append(self.generate_input_tensor(sub_shape, sub_d, item_vr, generator=generator))
                output.append(sub_list)  # 保持为 TensorList
            elif isinstance(s, list) and s and isinstance(s[0], list) and s[0] and isinstance(s[0][0], list):
                # 三层嵌套：每个元素本身是 TensorList -> 递归处理
                sub_list = []
                for j, sub_item in enumerate(s):
                    sub_d = d[j] if isinstance(d, list) and j < len(d) else d
                    sub_vr_item = vr[j] if isinstance(vr, list) and j < len(vr) else vr
                    self._generate_tensors(sub_item, sub_d, sub_vr_item, sub_list, generator=generator)
                output.append(sub_list)