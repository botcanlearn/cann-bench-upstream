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

"""
Golden函数打包器

职责：
1. 收集所有golden函数
2. 生成Python包结构
3. 注册到torch.ops.cann_bench
4. 打包为whl
"""

import re
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import yaml


@dataclass
class OperatorInfo:
    """算子信息"""
    level: int
    name: str
    dir_name: str
    schema: str
    golden_path: Path
    func_name: str


class GoldenPackager:
    """Golden函数打包器"""

    def __init__(self, bench_root: str, output_dir: str = None):
        self.bench_root = Path(bench_root)
        if not self.bench_root.exists():
            raise ValueError(f"kernel_bench目录不存在: {bench_root}")

        self.output_dir = Path(output_dir) if output_dir else Path.cwd()
        self.package_name = "cann_bench_golden"
        self.version = "0.1.0"

    def scan_operators(self) -> List[OperatorInfo]:
        """扫描所有算子的golden函数"""
        operators = []
        for level in [1, 2, 3, 4]:
            level_dir = self.bench_root / f"level{level}"
            if not level_dir.exists():
                continue

            for op_dir in level_dir.iterdir():
                if op_dir.is_dir() and not op_dir.name.startswith('.'):
                    golden_path = op_dir / "golden.py"
                    proto_path = op_dir / "proto.yaml"

                    if golden_path.exists():
                        schema, func_name = self._extract_schema(proto_path, golden_path)
                        # 从proto.yaml获取算子名
                        op_name = self._extract_operator_name(proto_path) or op_dir.name
                        operators.append(OperatorInfo(
                            level=level,
                            name=op_name,
                            dir_name=op_dir.name,
                            schema=schema,
                            golden_path=golden_path,
                            func_name=func_name
                        ))
        return operators

    def _extract_operator_name(self, proto_path: Path) -> Optional[str]:
        """从proto.yaml提取算子名"""
        if not proto_path.exists():
            return None
        try:
            with open(proto_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            if data and 'operator' in data:
                return data['operator'].get('name')
        except Exception:
            pass
        return None

    def _extract_schema(self, proto_path: Path, golden_path: Path) -> Tuple[str, str]:
        """提取schema和函数名"""
        schema = ""
        func_name = ""

        # 从proto.yaml提取schema
        if proto_path.exists():
            try:
                with open(proto_path, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
                if data and 'operator' in data:
                    schema = data['operator'].get('schema', '')
                    # 从schema提取函数名: mish(Tensor x) -> Tensor y -> mish
                    match = re.match(r'^(\w+)\s*\(', schema.strip())
                    if match:
                        func_name = match.group(1)
            except Exception:
                pass

        # 如果没有schema，从golden.py提取函数名
        if not func_name:
            try:
                content = golden_path.read_text(encoding='utf-8')
                matches = re.findall(r'def\s+(\w+)\s*\(', content)
                # 找第一个非辅助函数
                for m in matches:
                    if not m.startswith('_') and m not in ('get_input', 'to_float'):
                        func_name = m
                        break
            except Exception:
                pass

        return schema, func_name

    def generate_package(self, operators: List[OperatorInfo]) -> Path:
        """生成Python包结构"""
        temp_dir = Path(tempfile.mkdtemp(prefix="golden_packager_"))

        # 创建包目录结构
        pkg_dir = temp_dir / self.package_name
        pkg_dir.mkdir(parents=True)

        # 生成 __init__.py (torch.library注册)
        init_content = self._generate_init_py(operators)
        (pkg_dir / "__init__.py").write_text(init_content, encoding='utf-8')

        # 生成 golden_impls.py (复制所有golden函数代码)
        impls_content = self._generate_golden_impls_py(operators)
        (pkg_dir / "golden_impls.py").write_text(impls_content, encoding='utf-8')

        # 生成 setup.py
        setup_content = self._generate_setup_py()
        (temp_dir / "setup.py").write_text(setup_content, encoding='utf-8')

        # 生成 pyproject.toml
        pyproject_content = self._generate_pyproject_toml()
        (temp_dir / "pyproject.toml").write_text(pyproject_content, encoding='utf-8')

        return temp_dir

    def _fix_schema_for_torch_library(self, schema: str) -> str:
        """修复schema使其兼容torch.library

        torch.library支持的类型有限: Tensor, SymInt, int, float, bool, Scalar, str
        不支持的类型包括: TensorList, Tensor[], list[int], int[]等数组类型
        默认参数必须在非默认参数之后
        """
        import re

        # 检查是否包含不支持的类型
        # torch.library支持的类型：Tensor, Tensor[], int[], float[], float, int, bool, str, Scalar, SymInt等
        # 注意：float[] 和 int[] 作为参数类型是支持的，但作为返回类型可能不支持
        # 不支持的类型：
        # - TensorList (应使用 Tensor[])
        # - list[int], List[int] (应使用 int[])
        # - list[float], List[float] (应使用 float[])
        unsupported_patterns = [
            'TensorList',      # 使用Tensor[]代替
            'list[int]',       # 使用int[]代替
            'list[float]',     # 使用float[]代替
            'List[int]',       # 使用int[]代替
            'List[float]',     # 使用float[]代替
        ]

        for pattern in unsupported_patterns:
            if pattern in schema:
                return ''  # 无法注册，返回空字符串表示跳过

        # 检查返回类型是否为 float[] (不支持)
        # schema格式: ... -> ReturnType
        return_match = re.search(r'->\s*(.+)$', schema)
        if return_match:
            return_type = return_match.group(1).strip()
            if 'float[]' in return_type and 'float[]?' not in return_type:
                return ''  # float[] 作为返回类型不支持

        # 先移除带 null 默认值的数组参数（必须在转换 null->None 之前）
        # int[] 参数如果是可选且默认null，移除（int[]不支持null默认值）
        schema = re.sub(r',\s*int\[\]\s+\w+=null', '', schema)
        schema = re.sub(r',\s*int\[\]\?\?\s+\w+=null', '', schema)

        # 移除函数签名开头位置的 int[] 类型 null 参数
        schema = re.sub(r'int\[\]\s+\w+=null,\s*', '', schema)

        # 修复 bool 默认值: false/true -> False/True
        schema = schema.replace('=false', '=False').replace('=true', '=True')
        schema = schema.replace('false)', 'False)').replace('true)', 'True)')

        # 修复 null 默认值: null -> None
        # 支持 float[]? scale=None 这种格式
        schema = schema.replace('=null', '=None').replace('null)', 'None)')

        # 检查默认参数顺序：默认参数必须在非默认参数之后
        # 模式：检测是否出现 "=None" 或 "=False" 或 "=True" 或 "=数字" 后面跟着非默认参数
        # 简化检查：提取参数部分，检查是否在某个默认参数后有非默认参数
        params_match = re.match(r'\w+\(([^)]+)\)', schema)
        if params_match:
            params_str = params_match.group(1)
            params = [p.strip() for p in params_str.split(',')]
            seen_default = False
            for param in params:
                has_default = '=' in param
                if seen_default and not has_default:
                    # 非默认参数出现在默认参数之后，跳过
                    return ''
                if has_default:
                    seen_default = True

        return schema

    def _generate_init_py(self, operators: List[OperatorInfo]) -> str:
        """生成 __init__.py，包含torch.library注册"""
        lines = [
            "#!/usr/bin/python3",
            "# coding=utf-8",
            "",
            "import torch",
            "from .golden_impls import *",
            "",
            "# 注册所有算子到 torch.ops.cann_bench",
            "# 使用 DEF 模式定义 schema，IMPL 模式实现函数",
            "_LIB_DEF = torch.library.Library('cann_bench', 'DEF')",
            "_LIB_IMPL = torch.library.Library('cann_bench', 'IMPL')",
            "",
        ]

        for op in operators:
            if op.schema:
                # 从schema提取函数名 (schema格式: func_name(args) -> outputs)
                import re
                schema_func_name_match = re.match(r'^(\w+)\s*\(', op.schema.strip())
                schema_func_name = schema_func_name_match.group(1) if schema_func_name_match else op.name.lower()

                # 修复 schema 使其兼容 torch.library
                schema_fixed = self._fix_schema_for_torch_library(op.schema)
                if schema_fixed:  # 只有能修复的才注册
                    lines.append(f"# {op.name} (L{op.level})")
                    lines.append(f"_LIB_DEF.define('{schema_fixed}')")
                    # impl使用schema中的函数名（已修复后的）
                    fixed_func_name_match = re.match(r'^(\w+)\s*\(', schema_fixed.strip())
                    fixed_func_name = fixed_func_name_match.group(1) if fixed_func_name_match else schema_func_name
                    lines.append(f"_LIB_IMPL.impl('{fixed_func_name}', {op.func_name})")
                    lines.append("")
                else:
                    # 不支持的schema类型，仅导入函数，不注册
                    lines.append(f"# {op.name} (L{op.level}) - schema unsupported, imported as function")
                    lines.append("")

        lines.extend([
            "__all__ = ['get_golden_function']",
            "",
            "",
            "def get_golden_function(level: int, operator: str):",
            "    \"\"\"获取golden函数\"",
            "    ",
            "    Args:",
            "        level: 难度级别",
            "        operator: 算子名称",
            "    ",
            "    Returns:",
            "        golden函数",
            "    \"\"\"",
            "    func_name = operator.lower()",
            "    if hasattr(torch.ops.cann_bench, func_name):",
            "        return getattr(torch.ops.cann_bench, func_name)",
            "    raise AttributeError(f'cann_bench中没有算子: {operator}')",
        ])

        return '\n'.join(lines)

    def _generate_golden_impls_py(self, operators: List[OperatorInfo]) -> str:
        """生成 golden_impls.py，包含所有golden函数"""
        lines = [
            "#!/usr/bin/python3",
            "# coding=utf-8",
            "",
            "\"\"\"Golden函数实现集合\"\"\"",
            "",
            "import torch",
            "",
        ]

        # 复制每个golden.py的内容（去掉import重复）
        for op in operators:
            lines.append(f"# ---- {op.name} (L{op.level}) ----")
            lines.append("")

            try:
                content = op.golden_path.read_text(encoding='utf-8')
                # 移除头部注释和import torch
                code_lines = content.split('\n')
                skip_header = True
                for line in code_lines:
                    if skip_header:
                        if line.startswith('import') or line.strip() == '' or line.startswith('#'):
                            if line.startswith('import torch'):
                                skip_header = False
                            continue
                        else:
                            skip_header = False
                    lines.append(line)
            except Exception as e:
                lines.append(f"# ERROR: {e}")

            lines.append("")
            lines.append("")

        return '\n'.join(lines)

    def _generate_setup_py(self) -> str:
        """生成 setup.py"""
        return f'''#!/usr/bin/python3
from setuptools import setup, find_packages

setup(
    name="{self.package_name}",
    version="{self.version}",
    packages=find_packages(),
    # 不声明 torch 依赖，避免 pip 安装冲突版本
    # 前置检测已确保环境有 torch
    install_requires=[],
    python_requires=">=3.8",
)
'''

    def _generate_pyproject_toml(self) -> str:
        """生成 pyproject.toml"""
        return f'''[build-system]
requires = ["setuptools>=45", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "{self.package_name}"
version = "{self.version}"
description = "CANN-Bench golden function implementations"
requires-python = ">=3.8"
# 不声明 torch 依赖，避免 pip 安装冲突版本
dependencies = []

[tool.setuptools]
packages = ["{self.package_name}"]
'''

    def build_wheel(self, temp_dir: Path) -> Path:
        """构建whl包"""
        import subprocess

        # 使用绝对路径
        output_dir_abs = self.output_dir.resolve()
        output_dir_abs.mkdir(parents=True, exist_ok=True)

        result = subprocess.run(
            ["pip", "wheel", ".", "--no-deps", "-w", str(output_dir_abs)],
            cwd=str(temp_dir),
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            print(f"[ERROR] pip wheel stderr: {result.stderr}")
            raise RuntimeError(f"构建whl失败: {result.stderr}")

        # 查找生成的whl文件
        for f in output_dir_abs.iterdir():
            if f.suffix == '.whl' and f.name.startswith(self.package_name):
                return f

        raise RuntimeError("未找到生成的whl文件")

    def _check_torch_installed(self) -> Tuple[bool, str]:
        """检测环境是否已安装 torch（包括用户目录和系统目录）

        Returns:
            (是否已安装, 版本信息)
        """
        try:
            import torch
            version = torch.__version__
            # 检查是否有 torch_npu 且版本匹配
            try:
                import torch_npu
                npu_version = torch_npu.__version__ if hasattr(torch_npu, '__version__') else "unknown"
                # 简单版本匹配检查（主要版本号）
                torch_major = version.split('.')[0]
                npu_major = npu_version.split('.')[0] if npu_version != "unknown" else torch_major
                if torch_major != npu_major:
                    return (False, f"torch {version} 与 torch_npu {npu_version} 版本不匹配")
            except ImportError:
                pass  # 没有 torch_npu 也允许，可能是纯 CPU 环境
            return (True, f"torch {version}")
        except ImportError:
            return (False, "torch 未安装")

    def package(self, clean_up: bool = True) -> Path:
        """执行打包流程"""
        # 前置检测：确保环境已安装 torch
        torch_installed, torch_info = self._check_torch_installed()
        if not torch_installed:
            raise RuntimeError(
                f"无法打包: {torch_info}\n"
                "请先安装 torch: pip install torch\n"
                "注意: 本工具不声明 torch 依赖，避免安装冲突版本"
            )
        print(f"[INFO] 环境检测: {torch_info}")

        print(f"[INFO] 扫描算子...")
        operators = self.scan_operators()
        print(f"[INFO] 发现 {len(operators)} 个算子")

        print(f"[INFO] 生成包结构...")
        temp_dir = self.generate_package(operators)

        print(f"[INFO] 构建whl...")
        whl_path = self.build_wheel(temp_dir)
        print(f"[INFO] 输出: {whl_path}")

        if clean_up:
            shutil.rmtree(temp_dir, ignore_errors=True)

        return whl_path


def main():
    """CLI入口"""
    import argparse

    parser = argparse.ArgumentParser(description="Golden函数打包器")
    parser.add_argument("--bench-root", required=True, help="kernel_bench目录路径")
    parser.add_argument("--output-dir", default=".", help="whl输出目录")

    args = parser.parse_args()

    packager = GoldenPackager(args.bench_root, args.output_dir)
    whl_path = packager.package()
    print(f"[SUCCESS] 打包完成: {whl_path}")


if __name__ == "__main__":
    main()