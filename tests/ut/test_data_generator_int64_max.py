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

import torch

from kernel_eval.data.data_generator import DataGenerator

INT64_MIN = -(1 << 63)
INT64_MAX = (1 << 63) - 1


def _generate(value_range, size=64, seed=17):
    return DataGenerator().generate_input_tensors_from_case(
        input_shapes=[[size]],
        dtypes=["int64"],
        value_ranges=[value_range],
        seed=seed,
    )[0]


def test_positive_int64_range_including_max_is_generated_and_repeatable():
    values = _generate([0, INT64_MAX])
    again = _generate([0, INT64_MAX])

    assert values.dtype == torch.int64
    assert torch.equal(values, again)
    assert int(values.min()) >= 0
    assert int(values.max()) <= INT64_MAX


def test_int64_crossing_zero_and_full_domain_ranges_are_supported():
    for value_range in ([-(1 << 62), INT64_MAX], [INT64_MIN, INT64_MAX]):
        values = _generate(value_range)
        again = _generate(value_range)

        assert values.dtype == torch.int64
        assert torch.equal(values, again)
        assert int(values.min()) >= value_range[0]
        assert int(values.max()) <= value_range[1]
        assert bool((values < 0).any())
        assert bool((values >= 0).any())


def test_int64_near_max_and_constant_max_controls():
    near_max = _generate([INT64_MAX - 2, INT64_MAX], size=4096)
    assert set(near_max.tolist()) == {INT64_MAX - 2, INT64_MAX - 1, INT64_MAX}

    constant_max = _generate([INT64_MAX, INT64_MAX])
    assert torch.equal(constant_max, torch.full((64,), INT64_MAX, dtype=torch.int64))

    below_max = _generate([0, INT64_MAX - 1])
    assert int(below_max.min()) >= 0
    assert int(below_max.max()) <= INT64_MAX - 1
