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

"""索引类输出 tie 顺序无关校验（issue #40：TopK golden 索引非确定）。"""

import torch

from kernel_eval.eval.index_check import validate_index_output, validate_index_gather_outputs


def _topk_alt_tiebreak(x, k, dim, largest=True):
    """模拟 NPU 候选：对并列元素用与 torch.topk 不同的 tie-break（翻转后 topk 再映射）。"""
    n = x.size(dim)
    v, i_rev = torch.topk(torch.flip(x, dims=[dim]), k=k, dim=dim, largest=largest)
    return v, n - 1 - i_rev


class _Out:
    def __init__(self, name, index_gather=None):
        self.name = name
        self.index_gather = index_gather


class _Op:
    def __init__(self, outputs):
        self.outputs = outputs


def _ties_input(seed=0):
    torch.manual_seed(seed)
    # 值域极小 → 必然大量并列，制造 torch.topk 与候选索引顺序不一致
    return torch.randint(-3, 4, (4, 64)).to(torch.float32)


def test_correct_candidate_with_different_tie_order_passes():
    x = _ties_input()
    v_g, i_g = torch.topk(x, k=10, dim=-1, largest=True)
    v_c, i_c = _topk_alt_tiebreak(x, 10, -1, largest=True)
    # 值相同、索引顺序不同（正是 issue #40 的场景）
    assert torch.equal(v_g, v_c)
    assert not torch.equal(i_g, i_c)
    ok, msg = validate_index_output(x, -1, i_c, v_c)
    assert ok, msg


def test_garbage_indices_fail():
    x = _ties_input()
    v_c, i_c = _topk_alt_tiebreak(x, 10, -1)
    i_bad = torch.zeros_like(i_c)  # 全指向第 0 个元素
    ok, msg = validate_index_output(x, -1, i_bad, v_c)
    assert not ok and "不一致" in msg


def test_out_of_range_indices_fail():
    x = _ties_input()
    v_c, i_c = _topk_alt_tiebreak(x, 10, -1)
    i_oob = i_c.clone()
    i_oob[0, 0] = 9999
    ok, msg = validate_index_output(x, -1, i_oob, v_c)
    assert not ok and "越界" in msg


def test_tie_equivalent_indices_pass():
    """golden 索引 + 候选值（同值但位置不同）应通过——并列等价。"""
    x = _ties_input()
    v_g, i_g = torch.topk(x, k=10, dim=-1, largest=True)
    v_c, _ = _topk_alt_tiebreak(x, 10, -1)
    ok, msg = validate_index_output(x, -1, i_g, v_c)
    assert ok, msg


def test_dim0_direction():
    torch.manual_seed(1)
    x = torch.randint(-2, 3, (32, 16)).float()
    v_c, i_c = _topk_alt_tiebreak(x, 8, 0)
    ok, msg = validate_index_output(x, 0, i_c, v_c)
    assert ok, msg


def test_shape_mismatch_fails():
    x = _ties_input()
    v_c, i_c = _topk_alt_tiebreak(x, 10, -1)
    ok, msg = validate_index_output(x, -1, i_c[:, :5], v_c)
    assert not ok and "形状" in msg


def test_declarative_path_passes_and_rejects():
    x = _ties_input()
    v_c, i_c = _topk_alt_tiebreak(x, 10, -1)
    op = _Op([_Out("y"), _Out("idx", {"input": "x", "dim_attr": "dim", "value_output": "y"})])
    params = {"x": x, "k": 10}
    attrs = {"dim": -1, "k": 10, "largest": True}
    ok, _ = validate_index_gather_outputs(op, params, attrs, [v_c, i_c])
    assert ok
    ok, msg = validate_index_gather_outputs(op, params, attrs, [v_c, torch.zeros_like(i_c)])
    assert not ok and "不一致" in msg


def test_all_nan_input_correct_indices_pass():
    """全 NaN 输入（如 top_k case 15, value_range=[nan,nan]）：索引正确应通过。

    回归：旧实现用 torch.equal 比对 gather 结果与值输出，NaN != NaN 使正确候选
    被误判失败。NaN-aware 比较应放行——两侧同为 NaN 视为相等。
    """
    x = torch.full((4, 64), float("nan"), dtype=torch.float32)
    k, dim = 10, -1
    # 候选：合法 distinct 索引（任取前 k 个），值即 gather 出的 NaN
    i_c = torch.arange(k).unsqueeze(0).expand(4, k).contiguous()
    v_c = torch.gather(x, dim, i_c)  # 全 NaN
    ok, msg = validate_index_output(x, dim, i_c, v_c)
    assert ok, msg


def test_nan_mixed_indices_still_validated():
    """部分 NaN 部分正常值时，错误索引仍应被拒（NaN-aware 不放过真实不符）。"""
    x = torch.tensor([[float("nan"), 1.0, 2.0, 3.0, float("nan"), 5.0]], dtype=torch.float32)
    dim = -1
    # 正确索引：指向 5.0, 3.0, 2.0
    i_good = torch.tensor([[5, 3, 2]])
    v = torch.gather(x, dim, i_good)
    ok, msg = validate_index_output(x, dim, i_good, v)
    assert ok, msg
    # 错误索引：把第一项指向 NaN 位置(0)，但值仍声称 5.0 → 不符，应判失败
    i_bad = torch.tensor([[0, 3, 2]])
    ok, msg = validate_index_output(x, dim, i_bad, v)
    assert not ok and "不一致" in msg


def test_operator_without_declaration_is_noop():
    """未声明 index_gather 的算子立即通过，对其他算子零影响。"""
    op = _Op([_Out("a"), _Out("b")])
    ok, msg = validate_index_gather_outputs(op, {}, {}, [torch.randn(3), torch.randn(3)])
    assert ok and msg == ""


def test_missing_dim_attr_fails_loudly():
    """声明了 index_gather 但 case attrs 缺 dim_attr 时应判失败（不静默兜底 dim=-1）。"""
    x = _ties_input()
    v_c, i_c = _topk_alt_tiebreak(x, 10, -1)
    op = _Op([_Out("y"), _Out("idx", {"input": "x", "dim_attr": "dim", "value_output": "y"})])
    # attrs 不含 'dim'
    ok, msg = validate_index_gather_outputs(op, {"x": x}, {"k": 10}, [v_c, i_c])
    assert not ok and "dim" in msg
    # attrs 为 None 同样判失败
    ok, msg = validate_index_gather_outputs(op, {"x": x}, None, [v_c, i_c])
    assert not ok and "dim" in msg


def test_missing_source_input_fails():
    x = _ties_input()
    v_c, i_c = _topk_alt_tiebreak(x, 10, -1)
    op = _Op([_Out("y"), _Out("idx", {"input": "NOPE", "dim_attr": "dim", "value_output": "y"})])
    ok, msg = validate_index_gather_outputs(op, {"x": x}, {"dim": -1}, [v_c, i_c])
    assert not ok and "源输入" in msg
