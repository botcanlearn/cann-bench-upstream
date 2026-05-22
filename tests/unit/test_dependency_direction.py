#!/usr/bin/python3
# coding=utf-8

"""
依赖方向验证测试

验证重构后的依赖关系是否符合设计：
- Registry 不应直接 import 特化类
- eval / data / report 等通用模块不应直接 import 特化类
- benches 应 import base 模块并继承基类
"""

import ast
import sys
from pathlib import Path
from typing import List, Set

import pytest


# 特化类列表（CANN 专属）
CANN_SPECIFIC_CLASSES = {
    'CannTaskLoader', 'CannCaseLoader', 'GoldenLoader',
    'CannTaskSpec', 'CannCaseSpec', 'CannSolutionSpec',
    'CannInputSpec', 'CannOutputSpec',
    'OperatorMatcher', 'CannDefaultChecker', 'CannOutputResult',
    'CannScoringScheme', 'ScoringCalculator', 'OperatorScoreInfo',
}

# 特化文件列表（允许依赖特化类的文件）
ALLOWED_SPECIFIC_FILES = {
    'benches/__init__.py',
    'benches/cann.py',
    'benches/cann_spec.py',
    'benches/cann_loader.py',
    'benches/cann_matcher.py',
    'benches/cann_checker.py',
    'benches/cann_scoring.py',
    'benches/cann_solution.py',
    'cli.py',
}

# Registry 文件列表（不允许依赖特化类）
REGISTRY_FILES = {
    'registry/loader_registry.py',
    'registry/golden_registry.py',
    'registry/matcher_registry.py',
    'registry/checker_registry.py',
    'registry/scoring_registry.py',
    'registry/bench_registry.py',
}

# 通用模块文件列表（不允许直接依赖特化类）
GENERIC_FILES = {
    'eval/evaluator.py',
    'eval/perf_eval.py',
    'eval/process_pool.py',
    'eval/failure_synthesizer.py',
    'eval/accuracy_eval.py',
    'eval/allclose_checker.py',
    'eval/op_runner.py',
    'eval/input_pool.py',
    'eval/results.py',
    'eval/subprocess_runner.py',
    'data/data_generator.py',
    'data/package_manager.py',
    'report/report_generator.py',
    'report/scoring.py',
    'report/summary_generator.py',
}


def get_benches_imports(filepath: Path) -> Set[str]:
    """提取文件中从 benches 模块 import 的 CANN 特化名称"""
    try:
        with open(filepath) as f:
            tree = ast.parse(f.read())

        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ''
                # 只检查从 benches 模块导入的情况
                if 'benches' in module:
                    for alias in node.names:
                        if alias.name in CANN_SPECIFIC_CLASSES:
                            names.add(alias.name)
        return names
    except Exception as e:
        print(f"[ERROR] 解析 {filepath}: {e}")
        return set()


def get_local_cann_imports(filepath: Path) -> Set[str]:
    """提取文件中通过 ``from . import X`` 导入的 CANN 特化名称（可能已失效）"""
    try:
        with open(filepath) as f:
            tree = ast.parse(f.read())

        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ''
                # ``from . import X`` 或 ``from .module import X`` —— module 为 None 或相对当前包
                if module == '' or module is None or (module.startswith('.') and 'benches' not in module):
                    for alias in node.names:
                        if alias.name in CANN_SPECIFIC_CLASSES:
                            names.add(alias.name)
        return names
    except Exception as e:
        print(f"[ERROR] 解析 {filepath}: {e}")
        return set()


def test_registry_no_cann_import():
    """Registry 不应从 benches 模块 import CANN 特化类"""
    src_path = Path(__file__).parent.parent.parent / "src" / "kernel_eval"

    violations = []
    for file in REGISTRY_FILES:
        filepath = src_path / file
        if filepath.exists():
            found = get_benches_imports(filepath)
            if found:
                violations.append(f"{file}: {sorted(found)}")

    assert not violations, f"Registry 文件从 benches 导入特化类: {violations}"


def test_generic_no_cann_import():
    """通用模块（eval / data / report）不应从 benches 导入 CANN 特化类"""
    src_path = Path(__file__).parent.parent.parent / "src" / "kernel_eval"

    violations = []
    local_violations = []
    for file in GENERIC_FILES:
        filepath = src_path / file
        if filepath.exists():
            found = get_benches_imports(filepath)
            if found:
                violations.append(f"{file}: from benches import {sorted(found)}")
            local = get_local_cann_imports(filepath)
            if local:
                local_violations.append(f"{file}: local import of {sorted(local)} (可能已失效)")

    assert not violations, f"通用模块从 benches 导入特化类: {violations}"
    assert not local_violations, f"通用模块从本地导入 CANN 特化名称: {local_violations}"


def test_benches_inherit_base():
    """benches/ 中的特化文件应引用 base 模块的基类"""
    src_path = Path(__file__).parent.parent.parent / "src" / "kernel_eval"
    benches_dir = src_path / "benches"

    if not benches_dir.exists():
        pytest.skip("benches/ 目录尚未创建")

    expected_base = {
        'TaskLoader', 'CaseLoader', 'GoldenLoaderBase',
        'OperatorMatcherBase', 'CorrectnessChecker', 'ScoringScheme',
        'TaskSpec', 'CaseSpec', 'SolutionSpec', 'InputSpec', 'OutputSpec',
        'AttrSpec',
    }

    missing = []
    for py_file in sorted(benches_dir.glob("*.py")):
        if py_file.name == "__init__.py":
            continue
        with open(py_file) as f:
            content = f.read()
        found = [name for name in expected_base if name in content]
        if not found:
            missing.append(py_file.name)

    assert not missing, f"以下 benches 文件未引用任何 base 基类: {missing}"


def test_no_reverse_dependency():
    """base / registry / utils / __init__ 不应从 benches 导入任何名称"""
    src_path = Path(__file__).parent.parent.parent / "src" / "kernel_eval"

    files_to_check = (
        list(REGISTRY_FILES) +
        ['base/__init__.py', 'base/loaders.py', 'base/models.py',
         'base/checker.py', 'base/enums.py', 'base/matcher.py',
         'base/result.py', 'base/scoring.py',
         'data/__init__.py', 'eval/__init__.py', 'report/__init__.py',
         'registry/__init__.py',
         'utils/__init__.py']
    )

    violations = []
    for file in files_to_check:
        filepath = src_path / file
        if filepath.exists():
            found = get_benches_imports(filepath)
            if found:
                violations.append(f"{file}: {sorted(found)}")

    assert not violations, f"反向依赖（base/registry/init 从 benches 导入）: {violations}"


def run_all_tests():
    """运行所有依赖方向测试"""
    print("=" * 60)
    print("依赖方向验证测试")
    print("=" * 60)

    results = [
        test_registry_no_cann_import(),
        test_generic_no_cann_import(),
        test_benches_inherit_base(),
        test_no_reverse_dependency(),
    ]

    passed = sum(results)
    total = len(results)

    print("\n" + "=" * 60)
    print(f"结果: {passed}/{total} 测试通过")
    print("=" * 60)

    return all(results)


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)