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
基准性能数据迁移脚本

将 baseline_perf_us 和 t_hw_us 从 cases.yaml/cases.csv 中抽取到 data/ 下的 JSON 文件，
按评测集分类存储。

功能：
1. extract: 提取 baseline 数据，生成 JSON 文件（不修改源文件）
2. strip:   从 cases.yaml/cases.csv 中移除 baseline/t_hw 字段
3. verify:  验证迁移后数据完整性

JSON 格式 (level → op_dir → case_id 三级嵌套，与 stanford_baseline.json 一致):
{
  "_metadata": { ... },
  "level1": {
    "exp": {
      "1": { "baseline_perf_us": 13.78, "t_hw_us": 1.09 },
      "2": { "baseline_perf_us": 29.06, "t_hw_us": 8.74 }
    }
  }
}
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import yaml

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# === 辅助函数 ===

def _is_nullish(value) -> bool:
    """判断值是否为 null / None / 空字符串 / 0.0（表示无数据）"""
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in ("none", "null", "nan", "")
    # float/int 0.0/0 视为有效值（某些算子确实有 0.0 的 baseline）
    return False


def _normalize_perf_value(value):
    """归一化性能值：保持原样（float / dict）或返回 None（表示无数据）

    支持：
    - float/int → float
    - dict（多硬件）→ dict（过滤掉 null 子值）
    - None/"None" → None（无数据）
    """
    if value is None:
        return None
    if isinstance(value, str):
        if value.strip().lower() in ("none", "null", "nan", ""):
            return None
        try:
            return float(value)
        except ValueError:
            return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dict):
        # 多硬件格式：过滤掉 null 子值
        result = {}
        for hw, v in value.items():
            if _is_nullish(v):
                continue
            if isinstance(v, (int, float)):
                result[hw] = float(v)
            elif isinstance(v, str):
                try:
                    result[hw] = float(v)
                except ValueError:
                    continue
            else:
                result[hw] = v
        return result if result else None
    return None


def _extract_from_cases_yaml(yaml_path: Path) -> list[tuple[int, dict]]:
    """从单个 cases.yaml 提取 baseline 数据

    Returns:
        [(case_id, perf_dict)] 列表，perf_dict 含 baseline_perf_us/t_hw_us（或省略）
    """
    with yaml_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not data or "cases" not in data:
        return []

    results = []
    for case in data["cases"]:
        case_id = case.get("case_id")
        if case_id is None:
            continue

        bp = _normalize_perf_value(case.get("baseline_perf_us"))
        thw = _normalize_perf_value(case.get("t_hw_us"))

        # 两者都无数据时跳过
        if bp is None and thw is None:
            results.append((case_id, {}))
            continue

        perf_dict = {}
        if bp is not None:
            perf_dict["baseline_perf_us"] = bp
        if thw is not None:
            perf_dict["t_hw_us"] = thw

        results.append((case_id, perf_dict))

    return results


def _resolve_rel_path_structure(op_dir: Path, bench_root: Path) -> tuple[str, str]:
    """解析算子目录的层级结构

    Returns:
        (level_key, op_key)
        - tasks/level1/exp → ("level1", "exp")
        - bench_lab/cv_agent_bench/flash_attention → ("", "flash_attention")（无 level）
        - bench_lab/kernel_bench/level3/roi_pooling → ("level3", "roi_pooling")
    """
    try:
        rel = str(op_dir.relative_to(bench_root))
    except ValueError:
        return ("", op_dir.name)

    parts = rel.split("/")

    # 检查第一部分是否为 level 目录
    if len(parts) >= 2 and parts[0].startswith("level"):
        return (parts[0], parts[1])
    elif len(parts) == 1:
        return ("", parts[0])
    else:
        # 多级但无 level（如 bench_lab 子目录）
        return ("", rel)


# === 提取逻辑 ===

BENCH_CONFIGS = {
    "tasks": {
        "bench_root": PROJECT_ROOT / "tasks",
        "output": PROJECT_ROOT / "tasks" / "metadata" / "910b2.json",
        "description": "CANN 主评测集 baseline 性能数据",
    },
    "cv_agent_bench": {
        "bench_root": PROJECT_ROOT / "bench_lab" / "cv_agent_bench",
        "output": PROJECT_ROOT / "bench_lab" / "cv_agent_bench" / "metadata" / "910b2.json",
        "description": "CV Agent Bench baseline 性能数据",
    },
    "kernel_bench": {
        "bench_root": PROJECT_ROOT / "bench_lab" / "kernel_bench",
        "output": PROJECT_ROOT / "bench_lab" / "kernel_bench" / "metadata" / "910b2.json",
        "description": "Kernel Bench baseline 性能数据",
    },
    "pypto_cann_bench": {
        "bench_root": PROJECT_ROOT / "bench_lab" / "pypto_cann_bench",
        "output": PROJECT_ROOT / "bench_lab" / "pypto_cann_bench" / "metadata" / "910b2.json",
        "description": "PYPTO CANN Bench baseline 性能数据",
    },
}


def extract_bench(bench_name: str) -> dict:
    """提取单个评测集的 baseline 数据"""
    config = BENCH_CONFIGS[bench_name]
    bench_root = config["bench_root"]

    if not bench_root.exists():
        print(f"[WARN] bench_root 不存在: {bench_root}")
        return {}

    data = {
        "_metadata": {
            "description": config["description"],
            "hardware": "910b2",
            "generated_at": datetime.now().isoformat(),
            "source": f"从 {bench_root.relative_to(PROJECT_ROOT)}/cases.yaml 迁移",
        }
    }

    total_cases = 0
    total_with_perf = 0

    # 扫描所有 cases.yaml
    for cases_yaml in bench_root.rglob("cases.yaml"):
        op_dir = cases_yaml.parent
        level_key, op_key = _resolve_rel_path_structure(op_dir, bench_root)

        extracted = _extract_from_cases_yaml(cases_yaml)
        total_cases += len(extracted)

        # 确定数据层级
        if level_key:
            level_data = data.setdefault(level_key, {})
            op_data = level_data.setdefault(op_key, {})
        else:
            op_data = data.setdefault(op_key, {})

        for case_id, perf_dict in extracted:
            if perf_dict:  # 有值才写入
                op_data[str(case_id)] = perf_dict
                total_with_perf += 1

    print(f"[INFO] {bench_name}: 总用例 {total_cases}, 有 baseline 数据 {total_with_perf}")

    # 移除空的 level/op 层
    _clean_empty_levels(data)

    return data


def _clean_empty_levels(data: dict):
    """移除空的 level/op 层（无实际数据的中间节点）"""
    keys_to_remove = []
    for key, value in data.items():
        if key == "_metadata":
            continue
        if isinstance(value, dict):
            # 递归清理
            inner_keys = []
            for inner_key, inner_value in value.items():
                if isinstance(inner_value, dict) and not inner_value:
                    inner_keys.append(inner_key)
            for k in inner_keys:
                del value[k]
            if not value:
                keys_to_remove.append(key)
    for k in keys_to_remove:
        del data[k]


def write_json(data: dict, output_path: Path):
    """写入 JSON 文件"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"[INFO] 已写入: {output_path}")
    print(f"[INFO] 文件大小: {output_path.stat().st_size / 1024:.1f} KB")


def do_extract():
    """执行提取：扫描所有评测集，生成 JSON 文件"""
    print("=" * 60)
    print("步骤 1：提取 baseline 数据 → JSON")
    print("=" * 60)

    for bench_name in BENCH_CONFIGS:
        data = extract_bench(bench_name)
        if data:
            write_json(data, BENCH_CONFIGS[bench_name]["output"])

    print("\n[完成] 所有 baseline JSON 已生成")


# === Strip 逻辑 ===

def strip_yaml_file(yaml_path: Path) -> bool:
    """从 cases.yaml 中移除 baseline_perf_us 和 t_hw_us 字段

    使用 ruamel.yaml 保留注释和格式，同时确保 None/null 值
    在输出中以 "null" 文本表示（而非空值 "- "）。
    """
    from ruamel.yaml import YAML

    ryaml = YAML()
    ryaml.preserve_quotes = True
    ryaml.width = 4096  # 避免长行被折行

    with yaml_path.open("r", encoding="utf-8") as f:
        data = ryaml.load(f)

    if not data or "cases" not in data:
        return False

    modified = False
    for case in data["cases"]:
        if "baseline_perf_us" in case:
            del case["baseline_perf_us"]
            modified = True
        if "t_hw_us" in case:
            del case["t_hw_us"]
            modified = True

    if modified:
        with yaml_path.open("w", encoding="utf-8") as f:
            ryaml.dump(data, f)

        # ruamel.yaml 会把 None 序列项写成 "- "（空值），而非 "- null"。
        # 这会导致 value_range 等列表中的 null 丢失可读性。
        # 将所有 "  - "（行尾空）还原为 "  - null"。
        content = yaml_path.read_text(encoding="utf-8")
        content = content.replace("  - \n", "  - null\n")
        yaml_path.write_text(content, encoding="utf-8")

    return modified


def strip_csv_file(csv_path: Path) -> bool:
    """从 cases.csv 中移除 baseline_perf_us 和 t_hw_us 列"""
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    if not fieldnames:
        return False

    # 检查是否包含这两个列
    cols_to_remove = ["baseline_perf_us", "t_hw_us"]
    new_fieldnames = [h for h in fieldnames if h not in cols_to_remove]

    if len(new_fieldnames) == len(fieldnames):
        return False  # 无需修改

    # 写回
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=new_fieldnames)
        writer.writeheader()
        for row in rows:
            for col in cols_to_remove:
                row.pop(col, None)
            writer.writerow(row)

    return True


def do_strip():
    """执行 strip：从所有 cases.yaml/cases.csv 中移除 baseline 字段"""
    print("=" * 60)
    print("步骤 3：从 cases.yaml/cases.csv 中移除 baseline/t_hw 字段")
    print("=" * 60)

    yaml_modified = 0
    csv_modified = 0

    for bench_name, config in BENCH_CONFIGS.items():
        bench_root = config["bench_root"]
        if not bench_root.exists():
            continue

        # Strip YAML
        for cases_yaml in bench_root.rglob("cases.yaml"):
            if strip_yaml_file(cases_yaml):
                yaml_modified += 1
                print(f"  [STRIP YAML] {cases_yaml.relative_to(PROJECT_ROOT)}")

        # Strip CSV
        for cases_csv in bench_root.rglob("cases.csv"):
            if strip_csv_file(cases_csv):
                csv_modified += 1
                print(f"  [STRIP CSV] {cases_csv.relative_to(PROJECT_ROOT)}")

    print(f"\n[完成] YAML 修改 {yaml_modified} 个, CSV 修改 {csv_modified} 个")


# === 验证逻辑 ===

def do_verify():
    """验证迁移后数据完整性"""
    print("=" * 60)
    print("步骤 4：验证迁移后数据完整性")
    print("=" * 60)

    all_ok = True

    for bench_name, config in BENCH_CONFIGS.items():
        bench_root = config["bench_root"]
        json_path = config["output"]

        if not bench_root.exists():
            print(f"[SKIP] {bench_name}: bench_root 不存在")
            continue

        if not json_path.exists():
            print(f"[ERROR] {bench_name}: JSON 文件不存在: {json_path}")
            all_ok = False
            continue

        # 加载 JSON
        with json_path.open("r", encoding="utf-8") as f:
            json_data = json.load(f)

        # 统计 JSON 中的有值条目
        json_count = 0
        for key, value in json_data.items():
            if key == "_metadata":
                continue
            if isinstance(value, dict):
                for inner_key, inner_value in value.items():
                    if isinstance(inner_value, dict):
                        for case_key, case_data in inner_value.items():
                            if isinstance(case_data, dict):
                                if "baseline_perf_us" in case_data or "t_hw_us" in case_data:
                                    json_count += 1

        # 统计 YAML 中的有值条目（应已移除）
        yaml_remaining = 0
        for cases_yaml in bench_root.rglob("cases.yaml"):
            with cases_yaml.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if not data or "cases" not in data:
                continue
            for case in data["cases"]:
                if "baseline_perf_us" in case or "t_hw_us" in case:
                    yaml_remaining += 1

        # 统计 CSV 中是否还有这两列
        csv_remaining = 0
        for cases_csv in bench_root.rglob("cases.csv"):
            with cases_csv.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                if reader.fieldnames and "baseline_perf_us" in reader.fieldnames:
                    csv_remaining += 1

        status = "✓" if yaml_remaining == 0 and csv_remaining == 0 else "✗"
        print(f"  {status} {bench_name}: JSON 有值={json_count}, YAML残留={yaml_remaining}, CSV残留={csv_remaining}")

        if yaml_remaining > 0 or csv_remaining > 0:
            all_ok = False

    if all_ok:
        print("\n[✓] 验证通过：所有 baseline 数据已成功迁移到 JSON，YAML/CSV 中无残留")
    else:
        print("\n[✗] 验证失败：部分 baseline 数据未迁移或有残留")

    return all_ok


# === 预验证（extract 后、strip 前） ===

def do_pre_verify():
    """提取后预验证：JSON 数据与 YAML 原始数据一致性"""
    print("=" * 60)
    print("预验证：JSON 数据与 YAML 原始数据一致性")
    print("=" * 60)

    all_ok = True

    for bench_name, config in BENCH_CONFIGS.items():
        bench_root = config["bench_root"]
        json_path = config["output"]

        if not bench_root.exists() or not json_path.exists():
            continue

        # 加载 JSON
        with json_path.open("r", encoding="utf-8") as f:
            json_data = json.load(f)

        mismatches = []

        for cases_yaml in bench_root.rglob("cases.yaml"):
            op_dir = cases_yaml.parent
            level_key, op_key = _resolve_rel_path_structure(op_dir, bench_root)

            with cases_yaml.open("r", encoding="utf-8") as f:
                yaml_data = yaml.safe_load(f)

            if not yaml_data or "cases" not in yaml_data:
                continue

            # 获取 JSON 中对应的层级
            if level_key:
                json_level = json_data.get(level_key, {})
                json_op = json_level.get(op_key, {})
            else:
                json_op = json_data.get(op_key, {})

            for case in yaml_data["cases"]:
                case_id = case.get("case_id")
                if case_id is None:
                    continue

                bp_yaml = _normalize_perf_value(case.get("baseline_perf_us"))
                thw_yaml = _normalize_perf_value(case.get("t_hw_us"))

                json_entry = json_op.get(str(case_id), {})

                bp_json = json_entry.get("baseline_perf_us")
                thw_json = json_entry.get("t_hw_us")

                # 比对
                if bp_yaml is not None and bp_json is not None:
                    if isinstance(bp_yaml, float) and isinstance(bp_json, float):
                        if abs(bp_yaml - bp_json) > 0.001:
                            mismatches.append(
                                f"{level_key}/{op_key}/{case_id}: "
                                f"baseline YAML={bp_yaml} vs JSON={bp_json}"
                            )
                    elif isinstance(bp_yaml, dict) and isinstance(bp_json, dict):
                        if bp_yaml != bp_json:
                            mismatches.append(
                                f"{level_key}/{op_key}/{case_id}: "
                                f"baseline dict YAML={bp_yaml} vs JSON={bp_json}"
                            )

                if thw_yaml is not None and thw_json is not None:
                    if isinstance(thw_yaml, float) and isinstance(thw_json, float):
                        if abs(thw_yaml - thw_json) > 0.001:
                            mismatches.append(
                                f"{level_key}/{op_key}/{case_id}: "
                                f"t_hw YAML={thw_yaml} vs JSON={thw_json}"
                            )

        status = "✓" if not mismatches else "✗"
        print(f"  {status} {bench_name}: mismatches={len(mismatches)}")
        if mismatches:
            for m in mismatches[:5]:
                print(f"    {m}")
            if len(mismatches) > 5:
                print(f"    ... 还有 {len(mismatches) - 5} 个不匹配")
            all_ok = False

    if all_ok:
        print("\n[✓] 预验证通过：JSON 数据与 YAML 原始数据一致")
    else:
        print("\n[✗] 预验证失败：JSON 数据与 YAML 原始数据不一致")

    return all_ok


# === 主入口 ===

def main():
    parser = argparse.ArgumentParser(description="基准性能数据迁移脚本")
    parser.add_argument("action", choices=["extract", "strip", "verify", "pre-verify", "all"],
                        help="操作: extract=提取→JSON, strip=移除YAML/CSV字段, "
                             "verify=验证完整性, pre-verify=提取后预验证, all=完整迁移流程")
    args = parser.parse_args()

    if args.action == "extract":
        do_extract()
    elif args.action == "pre-verify":
        do_extract()
        do_pre_verify()
    elif args.action == "strip":
        do_strip()
    elif args.action == "verify":
        do_verify()
    elif args.action == "all":
        do_extract()
        if not do_pre_verify():
            print("\n[中断] 预验证失败，请检查后再继续")
            sys.exit(1)
        do_strip()
        if not do_verify():
            print("\n[中断] 最终验证失败")
            sys.exit(1)
        print("\n[✓] 迁移完成！")


if __name__ == "__main__":
    main()