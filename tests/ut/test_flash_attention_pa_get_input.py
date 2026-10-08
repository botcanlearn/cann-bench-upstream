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

import importlib.util
from pathlib import Path

import torch


_GOLDEN_PATH = (
    Path(__file__).resolve().parents[2]
    / "bench_lab/cv_agent_fa_bench/flash_attention_pa/golden.py"
)
_SPEC = importlib.util.spec_from_file_location("flash_attention_pa_golden", _GOLDEN_PATH)
_GOLDEN = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_GOLDEN)


def test_get_input_preserves_pages_beyond_query_length():
    q = torch.zeros((1, 1, 1, 1), dtype=torch.float16)
    cache = torch.tensor([1, 3, 5, 7], dtype=torch.float16).reshape(2, 2, 1, 1)
    block_table = torch.tensor([[0, 1]], dtype=torch.int32)

    q2, k2, v2, generated_table = _GOLDEN.get_input(q, cache, cache, block_table)

    assert generated_table.shape == block_table.shape
    expected = _GOLDEN.flash_attention_pa(q, cache, cache, block_table)
    actual = _GOLDEN.flash_attention_pa(q2, k2, v2, generated_table)
    torch.testing.assert_close(actual, expected)
