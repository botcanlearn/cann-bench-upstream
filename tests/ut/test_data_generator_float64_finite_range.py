#!/usr/bin/python3
# coding=utf-8

# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details and conditions of use.
# ----------------------------------------------------------------------------------------------------------

import torch

from kernel_eval.data.data_generator import DataGenerator


def _generate(bounds, *, input_dist="uniform", seed=17):
    return DataGenerator(input_dist=input_dist).generate_input_tensors_from_case(
        input_shapes=[[512]],
        dtypes=["float64"],
        value_ranges=[bounds],
        seed=seed,
    )[0]


def test_uniform_float64_generation_handles_finite_range_with_overflowing_span():
    bounds = [-1.0e308, 1.0e308]
    values = _generate(bounds)
    repeated = _generate(bounds)

    assert values.dtype == torch.float64
    assert torch.isfinite(values).all()
    assert torch.equal(values, repeated)
    assert bool((values >= bounds[0]).all())
    assert bool((values <= bounds[1]).all())
    assert float(values.min()) < -8.0e307
    assert float(values.max()) > 8.0e307


def test_normal_float64_generation_handles_finite_range_with_overflowing_span():
    bounds = [-1.0e308, 1.0e308]
    values = _generate(bounds, input_dist="normal")

    assert torch.isfinite(values).all()
    assert bool((values >= bounds[0]).all())
    assert bool((values <= bounds[1]).all())
    assert bool((values < 0).any())
    assert bool((values > 0).any())
    assert torch.unique(values).numel() > 1


def test_float64_narrow_and_one_sided_ranges_remain_valid_controls():
    for bounds in ([-1.0e100, 1.0e100], [-1.0e308, 0.0], [0.0, 1.0e308]):
        values = _generate(bounds)
        assert torch.isfinite(values).all()
        assert bool((values >= bounds[0]).all())
        assert bool((values <= bounds[1]).all())
        assert float(values.min()) < float(values.max())
