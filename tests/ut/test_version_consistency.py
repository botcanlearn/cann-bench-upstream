#!/usr/bin/python3
# coding=utf-8

"""
版本一致性单元测试

校验 VERSION / tasks/VERSION 与代码引用是否一致。
每次 pytest 运行时自动校验，无需手动执行脚本。
"""

import os
import re
import subprocess

import pytest

# 项目根目录
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_DIR = os.path.join(ROOT, "src")
KERNEL_EVAL_DIR = os.path.join(SRC_DIR, "kernel_eval")


# ---------------------------------------------------------------------------
# VERSION 文件存在性
# ---------------------------------------------------------------------------

def test_version_file_exists():
    """项目根 VERSION 文件必须存在"""
    assert os.path.isfile(os.path.join(ROOT, "VERSION")), \
        f"VERSION 文件不存在: {os.path.join(ROOT, 'VERSION')}"


def test_tasks_version_file_exists():
    """tasks/metadata/VERSION 文件必须存在"""
    assert os.path.isfile(os.path.join(ROOT, "tasks", "metadata", "VERSION")), \
        f"tasks/metadata/VERSION 文件不存在: {os.path.join(ROOT, 'tasks', 'metadata', 'VERSION')}"


# ---------------------------------------------------------------------------
# VERSION 文件格式
# ---------------------------------------------------------------------------

def test_version_file_format():
    """VERSION 文件格式必须为 X.Y.Z（首行）"""
    content = open(os.path.join(ROOT, "VERSION")).read().strip()
    version_line = content.split("\n")[0].strip()
    assert re.match(r"^\d+\.\d+\.\d+", version_line), \
        f"VERSION 格式不合法: '{version_line}'，应为 X.Y.Z 格式"


def test_tasks_version_file_format():
    """tasks/metadata/VERSION 文件格式必须为 X.Y.Z（首行）"""
    content = open(os.path.join(ROOT, "tasks", "metadata", "VERSION")).read().strip()
    version_line = content.split("\n")[0].strip()
    assert re.match(r"^\d+\.\d+\.\d+", version_line), \
        f"tasks/metadata/VERSION 格式不合法: '{version_line}'，应为 X.Y.Z 格式"


# ---------------------------------------------------------------------------
# 运行时版本与 VERSION 文件一致性
# ---------------------------------------------------------------------------

def test_kernel_eval_version_matches_version_file():
    """kernel_eval.__version__ 必须与 VERSION 文件一致"""
    import kernel_eval
    version_file = open(os.path.join(ROOT, "VERSION")).read().strip().split("\n")[0]
    assert kernel_eval.__version__ == version_file, \
        f"kernel_eval.__version__='{kernel_eval.__version__}' != VERSION文件='{version_file}'"


def test_tasks_version_matches_tasks_version_file():
    """kernel_eval.TASKS_VERSION 必须与 tasks/metadata/VERSION 文件一致"""
    import kernel_eval
    tasks_version_file = open(os.path.join(ROOT, "tasks", "metadata", "VERSION")).read().strip().split("\n")[0]
    assert kernel_eval.TASKS_VERSION == tasks_version_file, \
        f"kernel_eval.TASKS_VERSION='{kernel_eval.TASKS_VERSION}' != tasks/metadata/VERSION文件='{tasks_version_file}'"


# ---------------------------------------------------------------------------
# 无残留硬编码版本字符串
# ---------------------------------------------------------------------------

def test_no_hardcoded_version_in_kernel_eval():
    """src/kernel_eval/ 中无残留硬编码版本字符串（排除 _version.py）

    搜索以下模式：
    - __version__ = "X.Y.Z"  （应由 _version.py 动态读取）
    - V0.X.Y 或 V1.X.Y  （框架版本应动态注入）
    - CANN-Bench V0.X.Y  （应动态读取）
    - VERSION = "X.Y"  （应从 _version.py 引用）

    排除 _version.py、__pycache__、.pyc 文件。
    """
    patterns = [
        r'__version__\s*=\s*"\d+\.\d+\.\d+"',
        r'V0\.\d+\.\d+',
        r'V1\.\d+\.\d+',
        r'CANN-Bench\s+V\d+\.\d+\.\d+',
        r'VERSION\s*=\s*"\d+\.\d+"',
    ]

    violations = []
    for root_dir, dirs, files in os.walk(KERNEL_EVAL_DIR):
        # 排除 __pycache__
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for fname in files:
            if not fname.endswith(".py"):
                continue
            # 排除 _version.py（版本真相源，允许包含版本相关代码）
            if fname == "_version.py":
                continue
            filepath = os.path.join(root_dir, fname)
            try:
                content = open(filepath).read()
            except Exception:
                continue
            for pattern in patterns:
                matches = re.findall(pattern, content)
                if matches:
                    for m in matches:
                        violations.append(f"{filepath}: 硬编码版本 '{m}'")

    assert not violations, \
        "发现硬编码版本字符串:\n" + "\n".join(violations)


# ---------------------------------------------------------------------------
# Changelog 最新版本与 VERSION 文件一致性
# ---------------------------------------------------------------------------

def test_changelog_latest_version_matches_version_file():
    """docs/changelog.md 中的最新版本号必须与 VERSION 文件一致"""
    version_file = open(os.path.join(ROOT, "VERSION")).read().strip().split("\n")[0]

    changelog_path = os.path.join(ROOT, "docs", "changelog.md")
    if not os.path.isfile(changelog_path):
        pytest.skip("changelog.md 不存在")

    changelog = open(changelog_path).read()
    # 搜索 ## V0.3.0 或 ## V0.1.0 格式的版本标题
    version_headers = re.findall(r"^## V(\d+\.\d+\.\d+)", changelog, re.MULTILINE)

    if not version_headers:
        pytest.skip("changelog.md 中无版本标题")

    latest_changelog_version = version_headers[0]
    assert latest_changelog_version == version_file, \
        f"changelog 最新版本='V{latest_changelog_version}' != VERSION文件='{version_file}'"