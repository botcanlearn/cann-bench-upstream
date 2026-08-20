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

"""CrossEntropyLoss Golden target-dtype contract tests."""

import pytest
import torch
import torch.nn.functional as F

from tasks.level2.cross_entropy_loss.golden import cross_entropy_loss


@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
def test_int32_hard_labels_match_int64_reference(reduction):
    logits = torch.tensor(
        [[1.5, -0.5, 0.25], [-1.0, 2.0, 0.5], [0.0, 0.5, 1.0]],
        dtype=torch.float64,
    )
    target = torch.tensor([0, -100, 2], dtype=torch.int32)

    actual = cross_entropy_loss(
        logits,
        target,
        reduction=reduction,
        ignore_index=-100,
    )
    expected = F.cross_entropy(
        logits,
        target.long(),
        reduction=reduction,
        ignore_index=-100,
    )

    torch.testing.assert_close(actual, expected)
    assert target.dtype == torch.int32


def test_float_soft_labels_remain_on_probability_target_path():
    logits = torch.tensor(
        [[1.5, -0.5, 0.25], [-1.0, 2.0, 0.5]],
        dtype=torch.float64,
    )
    target = torch.tensor(
        [[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]],
        dtype=torch.float64,
    )

    actual = cross_entropy_loss(logits, target)
    expected = F.cross_entropy(logits, target)

    torch.testing.assert_close(actual, expected)
    assert target.dtype == torch.float64


def test_out_of_range_int32_hard_label_is_not_silently_rewritten():
    logits = torch.randn(2, 3, dtype=torch.float64)
    target = torch.tensor([0, 3], dtype=torch.int32)

    with pytest.raises(IndexError, match="out of bounds"):
        cross_entropy_loss(logits, target)
