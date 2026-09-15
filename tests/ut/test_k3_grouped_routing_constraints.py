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
bench_lab/kernel_bench/level5/k3_kda_decoder_layer_e2e 分组路由参数约束回归测试

背景（PR #312 检视意见）：golden 的分组路由对每组取 top2 之和选组，若不校验参数，
E/num_expert_group=1 时组内 topk(2) 直接越界报错；top_k 超出选中组容量
（topk_group × 每组专家数）时 top-k 会静默选入被 mask 组的专家。本测试验证：
1. 三类非法参数组合被 golden 以 ValueError 明确拒绝（而非运行期错误或静默错误语义）
2. 合法分组路由下 top-k 只落在按「组内 top2 之和」选出的前 topk_group 组内
3. num_expert_group=1（K3 配置，无分组）路径不受校验影响
"""

import importlib.util

import pytest
import torch

from kernel_eval.config import get_project_root


_OP_DIR = (get_project_root() / "bench_lab" / "kernel_bench" / "level5"
           / "k3_kda_decoder_layer_e2e")

pytestmark = pytest.mark.skipif(
    not (_OP_DIR / "golden.py").is_file(),
    reason="bench_lab/kernel_bench/level5/k3_kda_decoder_layer_e2e 不在本 checkout 中")


def _load_golden():
    spec = importlib.util.spec_from_file_location("k3_golden", _OP_DIR / "golden.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def golden():
    return _load_golden()


def _gate_args(golden, n_tok=6, h=32, num_experts=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    h2d = torch.randn(n_tok, h, generator=g)
    router_w = torch.randn(num_experts, h, generator=g) * 0.1
    router_bias = torch.randn(num_experts, generator=g) * 0.1
    return h2d, router_w, router_bias


@pytest.mark.parametrize(
    "num_experts,top_k,num_expert_group,topk_group,pattern",
    [
        # 每组仅 1 个专家：组内 top2 越界（评审复现 1）
        (4, 2, 4, 1, "每组专家数"),
        # top_k 超出选中组容量：会选入被 mask 组（评审复现 2）
        (4, 3, 2, 1, "top_k"),
        # topk_group 超出组数
        (8, 2, 2, 3, "topk_group"),
        # num_expert_group 不整除 E
        (6, 2, 4, 2, "num_expert_group"),
    ],
)
def test_invalid_routing_params_rejected(golden, num_experts, top_k,
                                         num_expert_group, topk_group, pattern):
    h2d, router_w, router_bias = _gate_args(golden, num_experts=num_experts)
    with pytest.raises(ValueError, match=pattern):
        golden._moe_gate(h2d, router_w, router_bias, top_k,
                         num_expert_group, topk_group, True, 1.0)


def test_grouped_topk_stays_inside_selected_groups(golden):
    """合法分组参数下，选中专家所属的组必须恰为「组内 top2 之和」最大的前 topk_group 组。"""
    num_experts, num_expert_group, topk_group, top_k = 16, 4, 2, 4
    group = num_experts // num_expert_group
    for seed in range(8):
        h2d, router_w, router_bias = _gate_args(
            golden, n_tok=5, num_experts=num_experts, seed=seed)
        idx, w = golden._moe_gate(h2d, router_w, router_bias, top_k,
                                  num_expert_group, topk_group, True, 1.0)
        # 独立计算每 token 的合法组集合
        choice = torch.sigmoid(h2d @ router_w.t()) + router_bias
        group_scores = choice.view(-1, num_expert_group, group).topk(
            2, dim=-1)[0].sum(-1)
        allowed = torch.topk(group_scores, k=topk_group, dim=-1)[1]  # [N, topk_group]
        got_groups = idx // group                                     # [N, top_k]
        for t in range(h2d.shape[0]):
            ok = torch.isin(got_groups[t], allowed[t]).all()
            assert bool(ok), (
                f"seed={seed} token={t}: 选中组 {sorted(got_groups[t].tolist())} "
                f"超出合法组 {sorted(allowed[t].tolist())}")
        assert bool((w > 0).all()), "路由权重必须来自未 mask 的 sigmoid 分数（恒正）"


def test_ungrouped_path_unaffected(golden):
    """num_expert_group=1（K3 配置）：校验放行，且行为等价于纯 top-k。"""
    h2d, router_w, router_bias = _gate_args(golden, num_experts=8)
    idx, w = golden._moe_gate(h2d, router_w, router_bias, 3, 1, 1, True, 1.0)
    choice = torch.sigmoid(h2d @ router_w.t()) + router_bias
    expect = torch.topk(choice, k=3, dim=-1, sorted=False)[1]
    assert torch.equal(idx.sort(dim=-1)[0], expect.sort(dim=-1)[0])
    assert w.shape == (h2d.shape[0], 3)
