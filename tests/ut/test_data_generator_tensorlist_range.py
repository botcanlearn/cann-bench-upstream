#!/usr/bin/env python3
# coding=utf-8

# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Regression for declared ranges on supported TensorList inputs."""

import torch

from kernel_eval.data.data_generator import DataGenerator


def test_tensorlist_members_use_declared_input_range():
    shapes = [[[4096], [4096]]]
    actual = DataGenerator().generate_input_tensors_from_case(
        input_shapes=shapes,
        dtypes=["float32"],
        value_ranges=[[-1.0, 1.0]],
        seed=12345,
    )
    explicit_per_member_range = DataGenerator().generate_input_tensors_from_case(
        input_shapes=shapes,
        dtypes=[["float32", "float32"]],
        value_ranges=[[[-1.0, 1.0], [-1.0, 1.0]]],
        seed=12345,
    )

    assert len(actual) == 1
    assert len(actual[0]) == 2
    for got, expected in zip(actual[0], explicit_per_member_range[0]):
        assert torch.equal(got, expected)

    values = torch.cat([tensor.flatten() for tensor in actual[0]])
    assert values.min().item() >= -1.0
    assert values.max().item() <= 1.0
    assert (values < 0).any().item()
    assert (values > 0).any().item()
