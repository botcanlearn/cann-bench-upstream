#!/usr/bin/python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
# See the repository LICENSE for the full text.

"""Check renamed bench_lab contracts without importing torch or NPU libraries."""

import __future__
import ast
import csv
import re
from collections import Counter
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
DELIVERIES = (
    "kernel_bench/level2/manifold_constrained_hyper_connection_post",
    "kernel_bench/level3/moe_init_routing_v3",
    "omni-ops_bench/esa_select_topk",
    "omni-ops_bench/fused_infer_attention_sink",
    "omni-ops_bench/kv_quant_sparse_flash_attention",
    "omni-ops_bench/sparse_flash_attention_gqa",
    "omni-ops_bench/sparse_flash_attention_pioneer",
)


@pytest.mark.parametrize("relative", DELIVERIES)
def test_renamed_delivery_contract(relative):
    folder = ROOT / "bench_lab" / relative
    name = folder.name
    assert not folder.with_name("ai_infra_" + name).exists()
    proto = yaml.safe_load((folder / "proto.yaml").read_text(encoding="utf-8"))["operator"]
    assert proto["name"] == name
    assert re.match(r"(\w+)\s*\(", proto["schema"]).group(1) == name
    tree = ast.parse((folder / "golden.py").read_text(encoding="utf-8-sig"))
    assert name in {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    cases = yaml.safe_load((folder / "cases.yaml").read_text(encoding="utf-8"))["cases"]
    with (folder / "cases.csv").open(encoding="utf-8-sig", newline="") as stream:
        csv_cases = list(csv.DictReader(stream))
    assert cases and len(cases) == len(csv_cases)
    assert {row["operator"] for row in cases} == {name}
    assert {row["operator"] for row in csv_cases} == {name}
    assert (folder / "desc.md").read_text(encoding="utf-8").startswith(f"# {name}")


@pytest.mark.parametrize("relative,external_name", [
    (DELIVERIES[0], "npu_ai_infra_manifold_constrained_hyper_connection_post"),
    (DELIVERIES[1], "npu_ai_infra_moe_init_routing_v2"),
])
def test_external_baseline_api_is_not_renamed(relative, external_name):
    source = (ROOT / "bench_lab" / relative / "test_baseline_perf.py").read_text(encoding="utf-8")
    calls = [node.func for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Call)]
    assert any(isinstance(func, ast.Attribute) and func.attr == external_name for func in calls)


@pytest.mark.parametrize("relative", DELIVERIES)
def test_golden_top_level_functions_are_unique(relative):
    source = (ROOT / "bench_lab" / relative / "golden.py").read_text(encoding="utf-8-sig")
    tree = ast.parse(source)
    counts = Counter(node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)))
    assert not {name: count for name, count in counts.items() if count > 1}


@pytest.mark.parametrize("extra", [
    {},
    {"blk_size": 32, "init_blk_num": 1, "local_blk_num": 2, "topk": 3, "compress_blk_size": 8},
    {
        "actual_seq_q_len_optional": [2, 4],
        "actual_seq_k_len_optional": [8, 16],
        "actual_cmp_seq_k_len_optional": [1, 2],
        "input_layout": "TND",
        "chunk_size": 2,
        "chunk_index": 1,
        "unused_option": True,
    },
])
def test_esa_entry_forwards_once_to_private_implementation(extra):
    """Execute the real wrapper with a spy implementation, without torch/NPU."""
    path = ROOT / "bench_lab" / DELIVERIES[2] / "golden.py"
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    entries = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "esa_select_topk"]
    assert len(entries) == 1
    implementations = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_esa_select_topk_impl"
    ]
    assert len(implementations) == 1
    calls = []
    result = object()

    def spy(*args, **kwargs):
        calls.append((args, kwargs))
        return result

    namespace = {"_esa_select_topk_impl": spy}
    wrapper = ast.Module(body=entries, type_ignores=[])
    exec(compile(wrapper, str(path), "exec", flags=__future__.annotations.compiler_flag), namespace)
    query, key = object(), object()
    assert namespace["esa_select_topk"](query, key, **extra) is result
    expected = {
        "blk_size": 64, "init_blk_num": 2, "local_blk_num": 4, "topk": 4,
        "input_layout": "TND", "compress_blk_size": 16,
        "actual_seq_q_len_optional": None, "actual_seq_k_len_optional": None,
        "actual_cmp_seq_k_len_optional": None,
    }
    expected.update({name: value for name, value in extra.items() if name in expected})
    assert calls == [((query, key), expected)]
