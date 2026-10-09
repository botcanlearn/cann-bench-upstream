#!/usr/bin/python3
# coding=utf-8

# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details about use and distribution.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See the License for the specific language governing permissions and limitations under the License.
# ----------------------------------------------------------------------------------------------------------

import torch

from kernel_eval.data.data_generator import DataGenerator


def _generator(seed=7):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def test_positive_finite_to_positive_infinity_range_stays_within_bounds():
    values = DataGenerator().generate_input_tensor([4096], "float32", [0.1, float("inf")], generator=_generator())

    finite = values[torch.isfinite(values)]
    assert finite.numel() > 0
    assert float(finite.min()) >= 0.1
    assert float(finite.max()) <= 1.0
    assert not torch.isneginf(values).any()
    assert torch.isposinf(values).any()


def test_negative_infinity_to_finite_range_stays_within_bounds():
    values = DataGenerator().generate_input_tensor([4096], "float32", [float("-inf"), -0.1], generator=_generator())

    finite = values[torch.isfinite(values)]
    assert finite.numel() > 0
    assert float(finite.min()) >= -1.0
    assert float(finite.max()) <= -0.1
    assert torch.isneginf(values).any()
    assert not torch.isposinf(values).any()


def test_symmetric_infinity_range_keeps_both_boundary_values():
    values = DataGenerator().generate_input_tensor(
        [4096], "float32", [float("-inf"), float("inf")], generator=_generator()
    )

    finite = values[torch.isfinite(values)]
    assert finite.numel() > 0
    assert float(finite.min()) >= -1.0
    assert float(finite.max()) <= 1.0
    assert torch.isneginf(values).any()
    assert torch.isposinf(values).any()


def test_constant_infinity_ranges_remain_constant():
    negative = DataGenerator().generate_input_tensor(
        [64], "float32", [float("-inf"), float("-inf")], generator=_generator()
    )
    positive = DataGenerator().generate_input_tensor(
        [64], "float32", [float("inf"), float("inf")], generator=_generator()
    )

    assert torch.isneginf(negative).all()
    assert torch.isposinf(positive).all()
