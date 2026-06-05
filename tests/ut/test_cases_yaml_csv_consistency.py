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
cases.yaml / cases.csv 一致性检查

扫描 tasks/ 和 bench_lab/ 目录，验证每个算子目录中 cases.yaml 与 cases.csv 的内容一致性。
检查项：
1. 文件配对：cases.yaml 和 cases.csv 必须同时存在
2. case 数量一致
3. case_id 集合完全相同
4. 列名覆盖一致
5. 按 case_id 对齐后逐字段值一致（含 nan/inf/None 归一化）
"""

import csv
import json
import math
from pathlib import Path

import pytest
import yaml

from kernel_eval.config import get_project_root


# === 辅助函数 ===


def _scan_dirs() -> list[Path]:
    """扫描 tasks/ 和 bench_lab/ 下含 cases.yaml 或 cases.csv 的所有目录"""
    root = get_project_root()
    dirs: list[Path] = []
    for bench_dir in [root / "tasks", root / "bench_lab"]:
        if not bench_dir.is_dir():
            continue
        for yaml_path in bench_dir.rglob("cases.yaml"):
            dirs.append(yaml_path.parent)
        for csv_path in bench_dir.rglob("cases.csv"):
            parent = csv_path.parent
            if parent not in dirs:
                dirs.append(parent)
    return sorted(dirs)


def _rel_dir(op_dir: Path) -> str:
    """将绝对路径转为相对于项目根的短路径"""
    root = get_project_root()
    try:
        return str(op_dir.relative_to(root))
    except ValueError:
        return op_dir.name


def _normalize_special_strings(obj: object) -> object:
    """递归将 list/dict 中的特殊浮点字符串 ('nan'/'inf'/'-inf') 转为 float

    JSON 解析后 nan/inf 会变成字符串而非 float，需二次归一化以匹配 YAML 的 float('nan')/float('inf')。

    注意：'none' 字符串不做转换——在 JSON 中 null 才是 None 的表示，'none' 是普通字符串（如
    GELU 的 approximate='none'）。同理 'true'/'false' 也不转换，JSON 中布尔用 true/false
    字面量而非字符串。
    """
    if isinstance(obj, list):
        return [_normalize_special_strings(item) for item in obj]
    if isinstance(obj, dict):
        return {k: _normalize_special_strings(v) for k, v in obj.items()}
    if isinstance(obj, str):
        low = obj.lower()
        if low == "nan":
            return float("nan")
        if low in ("inf", "+inf", "infinity", ".inf", "+.inf"):
            return float("inf")
        if low in ("-inf", "-infinity", "-.inf"):
            return float("-inf")
        return obj
    return obj


def _normalize_yaml_val(val: object) -> object:
    """归一化 YAML 值中的特殊字符串，使其与 CSV 归一化结果对齐

    YAML 1.1 (safe_load) 中 .inf/.nan 是 float，但裸 inf/-inf/nan 是 str。
    此函数将字符串形式的特殊浮点值也转为 float，与 CSV 侧 _normalize_special_strings 对齐。
    对于 baseline_perf_us/t_hw_us 中字符串 "None"，也归一化为 Python None。
    """
    if isinstance(val, list):
        return [_normalize_yaml_val(item) for item in val]
    if isinstance(val, dict):
        return {k: _normalize_yaml_val(v) for k, v in val.items()}
    if isinstance(val, str):
        low = val.lower()
        # 裸 inf/-inf/nan 字符串 → float（匹配 CSV 侧归一化）
        if low == "nan":
            return float("nan")
        if low in ("inf", "+inf", "infinity"):
            return float("inf")
        if low in ("-inf", "-infinity"):
            return float("-inf")
        # 'none' 字符串不转换——在 attrs 等字段中 'none' 是合法字符串值
        # （如 GELU 的 approximate='none'）；baseline_perf_us/t_hw_us 中的
        # 'None' 通过 _coerce_perf_us 单独处理
        return val
    return val


def _coerce_perf_us(value: object) -> object:
    """将 baseline_perf_us / t_hw_us 归一化为统一表示

    YAML 侧：Python None / str("None") / int / float → None 或 float
    CSV 侧：空字符串 / str("None") / 数值字符串 → None 或 float

    两侧 None / "None" / 空字符串 均视为「无数据」，归一化到 Python None。
    """
    if value is None:
        return None
    if isinstance(value, str):
        low = value.strip().lower()
        if low in ("none", "null", "nan", ""):
            return None
        try:
            return float(value)
        except ValueError:
            return None
    if isinstance(value, (int, float)):
        return float(value)
    return value


def _parse_csv_field(raw: str) -> object:
    """将 CSV 字符串字段反序列化为 Python 对象

    策略：JSON → 单引号修正 JSON → int → float → 特殊值 → 原始字符串
    JSON 解析后再归一化嵌套的特殊字符串（nan/inf）。
    """
    if raw == "":
        return None
    # JSON（双引号格式）
    try:
        parsed = json.loads(raw)
        return _normalize_special_strings(parsed)
    except (json.JSONDecodeError, ValueError):
        pass
    # Python 风格单引号 → 替换为双引号后 JSON（同时替换 True/False/None 为 JSON 字面量）
    try:
        fixed = raw.replace("'", '"').replace("True", "true").replace("False", "false").replace("None", "null")
        parsed = json.loads(fixed)
        return _normalize_special_strings(parsed)
    except (json.JSONDecodeError, ValueError):
        pass
    # int
    try:
        return int(raw)
    except ValueError:
        pass
    # float（含 nan / inf）
    try:
        return float(raw)
    except ValueError:
        pass
    # 特殊浮点值字符串
    low = raw.lower()
    if low == "nan":
        return float("nan")
    if low in ("inf", "+inf", "infinity"):
        return float("inf")
    if low in ("-inf", "-infinity"):
        return float("-inf")
    # 特殊布尔字符串（'none' 不转换——在 JSON 中 null 才是 None，'none' 是普通字符串）
    if low in ("true", "false"):
        return low == "true"
    # 原始字符串兜底
    return raw


def _deep_eq(a: object, b: object, *, float_tol: float = 1e-9) -> bool:
    """递归比较两个值，处理 nan/inf/None/float tolerances/嵌套结构"""
    # None 比较
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False

    # 类型不同时允许 int ↔ float 互通，以及数值 ↔ 数值字符串互通
    if type(a) is not type(b):
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            af, bf = float(a), float(b)
            if math.isnan(af) and math.isnan(bf):
                return True
            if math.isinf(af) and math.isinf(bf):
                return af == bf
            return abs(af - bf) <= float_tol
        # 数值 ↔ 数值字符串（如 YAML float 1e-08 vs CSV str "1e-08"）
        if isinstance(a, (int, float)) and isinstance(b, str):
            try:
                return abs(float(a) - float(b)) <= float_tol
            except ValueError:
                return False
        if isinstance(a, str) and isinstance(b, (int, float)):
            try:
                return abs(float(a) - float(b)) <= float_tol
            except ValueError:
                return False
        return False

    # float 比较
    if isinstance(a, float):
        if math.isnan(a):
            return isinstance(b, float) and math.isnan(b)
        if math.isinf(a):
            return isinstance(b, float) and math.isinf(b) and a == b
        if isinstance(b, float):
            if math.isnan(b) or math.isinf(b):
                return False
            return abs(a - b) <= float_tol
        return False

    # dict 比较
    if isinstance(a, dict):
        if not isinstance(b, dict):
            return False
        if set(a.keys()) != set(b.keys()):
            return False
        return all(_deep_eq(a[k], b[k]) for k in a)

    # list 比较
    if isinstance(a, list):
        if not isinstance(b, list):
            return False
        if len(a) != len(b):
            return False
        return all(_deep_eq(ai, bi) for ai, bi in zip(a, b))

    # str 比较
    if isinstance(a, str):
        # 尝试数值比较：CSV 中科学记数法如 "1e-08" 是字符串但 YAML 中是 float
        if isinstance(b, (int, float)):
            try:
                return abs(float(a) - float(b)) <= float_tol
            except ValueError:
                return False
        return a == b

    # bool / int / 其他
    return a == b


def _yaml_keys(yaml_cases: list[dict]) -> set[str]:
    """收集 YAML 所有 case 中出现过的 key 集合"""
    keys: set[str] = set()
    for c in yaml_cases:
        keys.update(c.keys())
    return keys


def _load_yaml(op_dir: Path) -> list[dict]:
    """加载 cases.yaml，返回 case 列表"""
    yaml_path = op_dir / "cases.yaml"
    with yaml_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not data or "cases" not in data:
        return []
    return data["cases"]


def _load_csv(op_dir: Path) -> list[dict[str, str]]:
    """加载 cases.csv，返回行列表（值全为字符串）"""
    csv_path = op_dir / "cases.csv"
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _yaml_cases_by_id(yaml_cases: list[dict]) -> dict[int, dict]:
    """将 YAML case 列表转为 {case_id: case_dict}"""
    result: dict[int, dict] = {}
    for c in yaml_cases:
        cid = c.get("case_id")
        if isinstance(cid, int):
            result[cid] = c
    return result


def _csv_cases_by_id(csv_rows: list[dict[str, str]]) -> dict[int, dict[str, str]]:
    """将 CSV 行列表转为 {case_id: row_dict}"""
    result: dict[int, dict[str, str]] = {}
    for r in csv_rows:
        cid_raw = r.get("case_id", "")
        try:
            cid = int(cid_raw)
        except ValueError:
            continue
        result[cid] = r
    return result


# === 目录列表（module 级参数化） ===

_ALL_DIRS = _scan_dirs()


# === 测试类 ===


@pytest.mark.parametrize("op_dir", _ALL_DIRS, ids=[_rel_dir(d) for d in _ALL_DIRS])
class TestCasesFilePairExistence:
    """每个含 cases 文件的目录必须同时拥有 cases.yaml 和 cases.csv"""

    def test_yaml_and_csv_coexist(self, op_dir: Path):
        """cases.yaml 和 cases.csv 必须同时存在"""
        has_yaml = (op_dir / "cases.yaml").is_file()
        has_csv = (op_dir / "cases.csv").is_file()
        rel = _rel_dir(op_dir)
        if has_yaml and not has_csv:
            pytest.fail(f"{rel}: missing cases.csv (only cases.yaml exists)")
        if has_csv and not has_yaml:
            pytest.fail(f"{rel}: missing cases.yaml (only cases.csv exists)")


# 只对同时存在两个文件的目录做一致性检查
_PAIRED_DIRS = [d for d in _ALL_DIRS
                if (d / "cases.yaml").is_file() and (d / "cases.csv").is_file()]


@pytest.mark.parametrize("op_dir", _PAIRED_DIRS, ids=[_rel_dir(d) for d in _PAIRED_DIRS])
class TestCasesYamlCsvConsistency:
    """cases.yaml / cases.csv 内容一致性检查"""

    def test_case_count(self, op_dir: Path):
        """case 数量一致"""
        yaml_cases = _load_yaml(op_dir)
        csv_rows = _load_csv(op_dir)
        rel = _rel_dir(op_dir)
        assert len(yaml_cases) == len(csv_rows), \
            f"{rel}: YAML has {len(yaml_cases)} cases but CSV has {len(csv_rows)} cases"

    def test_case_id_set(self, op_dir: Path):
        """case_id 集合完全相同"""
        yaml_cases = _load_yaml(op_dir)
        csv_rows = _load_csv(op_dir)
        rel = _rel_dir(op_dir)
        yaml_ids = set(_yaml_cases_by_id(yaml_cases).keys())
        csv_ids = set(_csv_cases_by_id(csv_rows).keys())
        missing_in_csv = yaml_ids - csv_ids
        missing_in_yaml = csv_ids - yaml_ids
        msg_parts: list[str] = []
        if missing_in_csv:
            msg_parts.append(f"case_ids in YAML but not in CSV: {sorted(missing_in_csv)}")
        if missing_in_yaml:
            msg_parts.append(f"case_ids in CSV but not in YAML: {sorted(missing_in_yaml)}")
        assert not msg_parts, f"{rel}: {'; '.join(msg_parts)}"

    def test_column_coverage(self, op_dir: Path):
        """YAML key 集合与 CSV 列名集合一致"""
        yaml_cases = _load_yaml(op_dir)
        csv_rows = _load_csv(op_dir)
        rel = _rel_dir(op_dir)

        yaml_keys_set = _yaml_keys(yaml_cases)
        csv_cols = set(csv_rows[0].keys()) if csv_rows else set()

        missing_in_csv = yaml_keys_set - csv_cols
        missing_in_yaml = csv_cols - yaml_keys_set

        msg_parts: list[str] = []
        if missing_in_csv:
            msg_parts.append(f"YAML keys missing from CSV columns: {sorted(missing_in_csv)}")
        if missing_in_yaml:
            msg_parts.append(f"CSV columns missing from YAML keys: {sorted(missing_in_yaml)}")
        assert not msg_parts, f"{rel}: {'; '.join(msg_parts)}"

    def test_field_values(self, op_dir: Path):
        """按 case_id 对齐后逐字段值一致"""
        yaml_cases = _load_yaml(op_dir)
        csv_rows = _load_csv(op_dir)
        rel = _rel_dir(op_dir)

        yaml_by_id = _yaml_cases_by_id(yaml_cases)
        csv_by_id = _csv_cases_by_id(csv_rows)

        mismatches: list[str] = []
        for cid in sorted(yaml_by_id.keys()):
            if cid not in csv_by_id:
                # 已在 test_case_id_set 中检查，此处跳过
                continue
            yc = yaml_by_id[cid]
            cc = csv_by_id[cid]

            for field in yc:
                if field not in cc:
                    # 列覆盖已在 test_column_coverage 中检查
                    continue

                yaml_val = yc[field]
                csv_raw = cc[field]

                # 归一化 YAML 值（处理裸 inf/nan/None 字符串等）
                yaml_norm = _normalize_yaml_val(yaml_val)

                # 归一化 CSV 字段后比较
                # 简单字段（operator / note）直接字符串比较
                # 复杂字段需先反序列化
                if field in ("operator", "note"):
                    # 纯字符串字段
                    csv_val = csv_raw
                elif field == "case_id":
                    # int 字段
                    try:
                        csv_val = int(csv_raw)
                    except ValueError:
                        csv_val = csv_raw
                elif field in ("baseline_perf_us", "t_hw_us"):
                    # 性能数值字段：两侧归一化后再比较
                    # None / "None" / 空字符串 → Python None；数值 → float
                    yaml_norm = _coerce_perf_us(yaml_norm)
                    csv_val = _coerce_perf_us(csv_raw)
                else:
                    # 复杂字段：input_shape / dtype / attrs / value_range 等
                    csv_val = _parse_csv_field(csv_raw)

                if not _deep_eq(yaml_norm, csv_val):
                    mismatches.append(
                        f"case_id={cid}, field={field}: "
                        f"YAML={_repr_val(yaml_norm)} vs CSV={_repr_val(csv_val)}"
                    )

        assert not mismatches, f"{rel}:\n" + "\n".join(mismatches)


def _repr_val(v: object) -> str:
    """安全 repr：nan/inf/None 显示友好字符串"""
    if v is None:
        return "None"
    if isinstance(v, float):
        if math.isnan(v):
            return "nan"
        if math.isinf(v):
            return "inf" if v > 0 else "-inf"
        return repr(v)
    if isinstance(v, list):
        if len(v) > 10:
            return f"[{', '.join(_repr_val(x) for x in v[:5])}, ...] (len={len(v)})"
        return f"[{', '.join(_repr_val(x) for x in v)}]"
    if isinstance(v, dict):
        if len(v) > 5:
            items = list(v.items())[:3]
            return f"{{{', '.join(f'{k}: {_repr_val(val)}' for k, val in items)}, ...}}"
        return f"{{{', '.join(f'{k}: {_repr_val(val)}' for k, val in v.items())}}}"
    return repr(v)