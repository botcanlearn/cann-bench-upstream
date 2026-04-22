#!/usr/bin/env python3
"""
Register a cann-bench operator as a BenchSite benchmark.

Creates a benchmark bundle (.tar.gz) containing:
  - benchmark.yaml (metadata)
  - harness/ (standardized build/eval scripts)
  - data/ (proto.yaml + cases.yaml + golden.py from cann-bench)
  - evaluate.py (standardized evaluation script)

Usage:
    python3 register_benchmark.py <cann_bench_operator_dir> [--upload]

Example:
    python3 register_benchmark.py kernel_bench/level1/exp --upload
"""

import os
import sys
import re
import yaml
import shutil
import tarfile
import argparse
import requests
from pathlib import Path

EVAL_ROOT = Path(__file__).resolve().parent.parent  # evaluation/
HARNESS_DIR = EVAL_ROOT / "harness"
EVALUATE_SCRIPT = EVAL_ROOT / "evaluate.py"
CORE_DIR = EVAL_ROOT / "core"

LEVEL_MAP = {
    "L1": "level_1",
    "L2": "level_2",
    "L3": "level_3",
    "L4": "level_4",
}

CATEGORY_MAP = {
    "Elementwise": "elementwise",
    "Broadcast": "broadcast",
    "Reduction": "reduction",
    "Contraction": "contraction",
    "FusedComposite": "fused_composite",
    "SpecialNumeric": "special_numeric",
    "Spatial": "spatial",
    "Indexing": "indexing",
    "Sorting": "sorting",
    "Reshape": "reshape",
    "Pooling": "pooling",
    "Normalization": "normalization",
    "Attention": "attention",
    "Recurrent": "recurrent",
    "QuantizedCompute": "quantized_compute",
    "MoE": "moe",
}


def camel_to_kebab(name: str) -> str:
    """Convert CamelCase to kebab-case slug."""
    s = re.sub(r"(?<!^)(?=[A-Z])", "-", name)
    return s.lower().replace("_", "-")


def _detect_level_from_path(op_dir: Path) -> str:
    """Detect benchmark level from directory path (e.g., kernel_bench/level1/exp → L1)."""
    parts = op_dir.resolve().parts
    for part in parts:
        if part.startswith("level"):
            try:
                return f"L{part.replace('level', '')}"
            except ValueError:
                pass
    return ""


def build_bundle(op_dir: Path, output_dir: Path, level_override: str = "") -> Path:
    """Build a benchmark bundle from a cann-bench operator directory."""
    proto_path = op_dir / "proto.yaml"
    if not proto_path.exists():
        raise FileNotFoundError(f"No proto.yaml in {op_dir}")

    with open(proto_path) as f:
        proto = yaml.safe_load(f)["operator"]

    op_name = proto["name"]
    category = proto.get("category", "Elementwise")
    description = proto.get("description", "")

    # Level: prefer override > directory detection > proto.yaml difficulty
    if level_override:
        difficulty = level_override.upper()
    else:
        detected = _detect_level_from_path(op_dir)
        difficulty = detected if detected else proto.get("difficulty", "L1")

    slug = camel_to_kebab(op_name)
    version = "1.0"
    level = LEVEL_MAP.get(difficulty, difficulty.lower())

    # Determine operator_type from inputs
    n_inputs = len(proto.get("inputs", []))
    if n_inputs <= 1:
        op_type = "unary"
    elif n_inputs == 2:
        op_type = "binary"
    else:
        op_type = "multi_input"

    # Create bundle directory
    bundle_name = f"{op_name}"
    bundle_root = output_dir / bundle_name
    if bundle_root.exists():
        shutil.rmtree(bundle_root)
    bundle_root.mkdir(parents=True)

    # Build tags: level tag + category tag
    level_tag = difficulty.lower()
    if level_tag.startswith("l") and len(level_tag) == 2:
        level_tag = "lv" + level_tag[1]
    cat_tag = CATEGORY_MAP.get(category, category.lower())
    tags = [level_tag, cat_tag]

    # 1. benchmark.yaml
    bm_yaml = {
        "benchmark_slug": slug,
        "name": op_name,
        "version": version,
        "level": level,
        "category": cat_tag,
        "operator_type": op_type,
        "difficulty": difficulty.lower(),
        "description": description,
        "tags": tags,
        "harness": {
            "build": "./harness/build.sh",
            "correctness": "./harness/run_correctness.sh",
            "performance": "./harness/run_perf.sh",
        },
        "submission_contract": {
            "layout": "custom_op_project",
            "required_prefixes": ["op_host/", "op_kernel/"],
        },
    }
    with open(bundle_root / "benchmark.yaml", "w") as f:
        yaml.dump(bm_yaml, f, default_flow_style=False, allow_unicode=True)

    # 2. Copy harness scripts (includes _common.sh sourced by run_*.sh)
    harness_dest = bundle_root / "harness"
    harness_dest.mkdir()
    for src in sorted(HARNESS_DIR.glob("*.sh")):
        dst = harness_dest / src.name
        shutil.copy2(src, dst)
        dst.chmod(0o755)

    # 3. Copy cann-bench data
    data_dest = bundle_root / "data"
    data_dest.mkdir()
    for fn in ["proto.yaml", "cases.yaml", "golden.py"]:
        src = op_dir / fn
        if src.exists():
            shutil.copy2(src, data_dest / fn)

    # 4. Copy evaluate.py + the evaluation/core package it imports from.
    # The bundle is self-contained: the layout inside the tarball is
    #   <bundle>/evaluate.py
    #   <bundle>/evaluation/__init__.py
    #   <bundle>/evaluation/core/...
    # so `from evaluation.core.<x>` works when the runner runs evaluate.py
    # with the bundle dir on sys.path (evaluate.py handles that itself).
    if EVALUATE_SCRIPT.exists():
        shutil.copy2(EVALUATE_SCRIPT, bundle_root / "evaluate.py")
    pkg_dest = bundle_root / "evaluation"
    pkg_dest.mkdir()
    (pkg_dest / "__init__.py").write_text('"""Bundled evaluation package."""\n')
    shutil.copytree(CORE_DIR, pkg_dest / "core",
                    ignore=shutil.ignore_patterns("__pycache__"))

    # 5. Create tar.gz
    tgz_path = output_dir / f"{slug}-{version}.tar.gz"
    with tarfile.open(tgz_path, "w:gz") as tf:
        tf.add(str(bundle_root), arcname=bundle_name)

    print(f"Bundle created: {tgz_path}")
    print(f"  Slug: {slug}, Name: {op_name}, Level: {level}")
    return tgz_path


def upload_bundle(tgz_path: Path, api_base: str, token: str, force: bool = False):
    """Upload the bundle to BenchSite API."""
    url = f"{api_base}/api/benchmarks/upload"
    with open(tgz_path, "rb") as f:
        files = {"file": (tgz_path.name, f, "application/gzip")}
        data = {}
        if force:
            data["force"] = "true"
        resp = requests.post(url, files=files, data=data,
                             headers={"Authorization": f"Bearer {token}"})
    if resp.status_code != 200:
        print(f"  Upload failed: {resp.status_code} {resp.text}")
        return False
    result = resp.json()
    if result.get("conflict"):
        print(f"  Conflict: {result.get('name')} already exists. Use --force to overwrite.")
        return False
    verb = "Overwritten" if result.get("overwritten") else "Registered"
    print(f"  {verb}: {result.get('name')} ({result.get('benchmark_slug')} v{result.get('version')})")
    return True


def main():
    parser = argparse.ArgumentParser(description="Register cann-bench operator as BenchSite benchmark")
    parser.add_argument("operator_dir", help="Path to cann-bench operator (e.g., kernel_bench/level1/exp)")
    parser.add_argument("--output-dir", default="/tmp/benchmarks", help="Output directory for bundles")
    parser.add_argument("--upload", action="store_true", help="Upload to BenchSite API")
    parser.add_argument("--api-base", default="http://localhost:8000", help="BenchSite API base URL")
    parser.add_argument("--token", help="Admin auth token")
    parser.add_argument("--force", action="store_true", help="Overwrite existing benchmark")
    parser.add_argument("--level", default="", help="Override level (e.g., L1, L2). Auto-detected from directory path if not set.")
    args = parser.parse_args()

    op_dir = Path(args.operator_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tgz_path = build_bundle(op_dir, output_dir, level_override=args.level)

    if args.upload:
        if not args.token:
            print("Error: --token required for upload")
            sys.exit(1)
        upload_bundle(tgz_path, args.api_base, args.token, force=args.force)


if __name__ == "__main__":
    main()
