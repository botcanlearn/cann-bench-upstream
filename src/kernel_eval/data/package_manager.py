"""
包管理模块

职责：
1. 扫描源码目录结构
2. 检查/编译whl包和run包
3. 安装whl包和run包
4. 扫描cann_bench模块提供的算子接口
5. 匹配kernel_bench中的算子定义
"""

import os
import subprocess
import sys
import importlib
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass

from ..config import get_config


@dataclass
class PackageInfo:
    """包信息"""
    source_dir: str
    whl_path: Optional[str] = None
    run_path: Optional[str] = None
    has_build_sh: bool = False
    has_dist: bool = False


@dataclass
class InterfaceInfo:
    """接口信息"""
    name: str
    callable: Any
    signature: str = ""

    def __repr__(self):
        return f"InterfaceInfo(name={self.name}, signature={self.signature})"


class PackageManager:
    """包管理器"""

    def __init__(self):
        self.config = get_config()
        self._interface_cache: Dict[str, InterfaceInfo] = {}

    def scan_source_dir(self, source_dir: str) -> PackageInfo:
        """扫描源码目录结构

        检查目录中是否存在：
        - build.sh 编译脚本
        - dist/ 目录（包含whl和run包）
        - cann_bench/ Python包目录
        """
        source_path = Path(source_dir)
        if not source_path.exists():
            raise FileNotFoundError(f"源码目录不存在: {source_dir}")

        package_info = PackageInfo(source_dir=str(source_path))

        # 检查 build.sh
        build_sh = source_path / "build.sh"
        package_info.has_build_sh = build_sh.exists()

        # 检查 dist 目录
        dist_dir = source_path / "dist"
        package_info.has_dist = dist_dir.exists()

        if package_info.has_dist:
            # 扫描whl包
            whl_files = list(dist_dir.glob("cann_bench*.whl"))
            if whl_files:
                package_info.whl_path = str(whl_files[0])

            # 扫描run包
            run_files = list(dist_dir.glob("cann_bench*.run"))
            if run_files:
                package_info.run_path = str(run_files[0])

        return package_info

    def check_dist_packages(self, source_dir: str) -> Tuple[Optional[str], Optional[str]]:
        """检查dist目录的whl包和run包

        返回: (whl_path, run_path)
        """
        dist_dir = Path(source_dir) / "dist"
        if not dist_dir.exists():
            return None, None

        # 查找whl包
        whl_path = None
        whl_files = list(dist_dir.glob("cann_bench*.whl"))
        if whl_files:
            whl_path = str(whl_files[0])

        # 查找run包
        run_path = None
        run_files = list(dist_dir.glob("cann_bench*.run"))
        if run_files:
            run_path = str(run_files[0])

        return whl_path, run_path

    def build_packages(self, source_dir: str) -> PackageInfo:
        """执行build.sh编译生成包

        执行build.sh后，扫描dist目录获取生成的包路径
        """
        source_path = Path(source_dir)
        build_sh = source_path / "build.sh"

        if not build_sh.exists():
            raise FileNotFoundError(f"build.sh不存在: {build_sh}")

        print(f"[INFO] 执行编译: bash build.sh")

        # 执行build.sh
        try:
            result = subprocess.run(
                ["bash", "build.sh"],
                cwd=str(source_path),
                capture_output=True,
                text=True,
                timeout=300  # 5分钟超时
            )
            if result.returncode != 0:
                print(f"[ERROR] 编译失败:")
                print(result.stderr)
                raise RuntimeError(f"build.sh执行失败: {result.stderr}")

            print(f"[INFO] 编译成功")
            if result.stdout:
                print(result.stdout[:500])  # 只打印前500字符

        except subprocess.TimeoutExpired:
            raise RuntimeError("build.sh执行超时（超过5分钟）")
        except Exception as e:
            raise RuntimeError(f"build.sh执行异常: {e}")

        # 扫描编译结果
        package_info = self.scan_source_dir(source_dir)
        if not package_info.whl_path:
            raise RuntimeError("编译后未找到whl包")

        return package_info

    def install_run_package(self, run_path: str) -> bool:
        """安装run包（NPU内核包）

        run包安装方式取决于包的实际格式，常见方式：
        - 直接执行: ./xxx.run --install
        """
        run_file = Path(run_path)
        if not run_file.exists():
            raise FileNotFoundError(f"run包不存在: {run_path}")

        print(f"[INFO] 安装run包: {run_file.name}")

        try:
            # 设置可执行权限
            os.chmod(run_path, 0o755)

            # 执行安装
            result = subprocess.run(
                [run_path, "--install"],
                cwd=str(run_file.parent),
                capture_output=True,
                text=True,
                timeout=120  # 2分钟超时
            )

            if result.returncode != 0:
                print(f"[WARN] run包安装可能失败: {result.stderr}")
                # run包安装可能不需要特定参数，继续执行
                return True

            print(f"[INFO] run包安装成功")
            return True

        except subprocess.TimeoutExpired:
            print(f"[WARN] run包安装超时")
            return False
        except Exception as e:
            print(f"[WARN] run包安装异常: {e}")
            return False

    def install_whl_package(self, whl_path: str) -> bool:
        """安装whl包（Python包）

        安装策略：先卸载旧版本，再安装新版本（不使用force-reinstall，避免重装依赖）
        """
        whl_file = Path(whl_path)
        if not whl_file.exists():
            raise FileNotFoundError(f"whl包不存在: {whl_path}")

        print(f"[INFO] 安装whl包: {whl_file.name}")

        try:
            # 先卸载旧版本（如果存在）
            print(f"[INFO] 卸载旧版本 cann_bench")
            subprocess.run(
                [sys.executable, "-m", "pip", "uninstall", "cann_bench", "-y"],
                capture_output=True,
                timeout=30
            )

            # 安装新版本
            print(f"[INFO] 安装: pip install {whl_path}")
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", whl_path],
                capture_output=True,
                text=True,
                timeout=60
            )

            if result.returncode != 0:
                print(f"[ERROR] whl包安装失败: {result.stderr}")
                raise RuntimeError(f"pip install 失败: {result.stderr}")

            print(f"[INFO] whl包安装成功")
            return True

        except subprocess.TimeoutExpired:
            raise RuntimeError("whl包安装超时")
        except Exception as e:
            raise RuntimeError(f"whl包安装异常: {e}")

    def install_packages(self, package_info: PackageInfo) -> bool:
        """安装所有包（run包 + whl包）

        安装顺序：先安装run包，再安装whl包
        """
        # 安装run包（如果存在）
        if package_info.run_path:
            if not self.install_run_package(package_info.run_path):
                print(f"[WARN] run包安装失败，继续安装whl包")

        # 安装whl包（必须成功）
        if package_info.whl_path:
            return self.install_whl_package(package_info.whl_path)

        return False

    def scan_interfaces(self) -> List[InterfaceInfo]:
        """扫描cann_bench模块接口

        扫描已安装的cann_bench模块，获取所有算子接口
        """
        interfaces = []

        try:
            # 重新导入模块（确保使用最新安装的版本）
            if 'cann_bench' in sys.modules:
                del sys.modules['cann_bench']

            import cann_bench

            # 扫描模块属性
            for name in dir(cann_bench):
                if name.startswith('_'):
                    continue

                attr = getattr(cann_bench, name)
                if callable(attr) and not isinstance(attr, type):
                    # 获取函数签名
                    try:
                        from inspect import signature
                        sig = str(signature(attr))
                    except Exception:
                        sig = ""

                    interfaces.append(InterfaceInfo(
                        name=name,
                        callable=attr,
                        signature=f"{name}{sig}"
                    ))

            # 扫描 torch.ops.cann_bench
            try:
                import torch
                if hasattr(torch.ops, 'cann_bench'):
                    for name in dir(torch.ops.cann_bench):
                        if name.startswith('_'):
                            continue

                        op = getattr(torch.ops.cann_bench, name)
                        if callable(op):
                            interfaces.append(InterfaceInfo(
                                name=name,
                                callable=op,
                                signature=f"torch.ops.cann_bench.{name}"
                            ))
            except ImportError:
                pass

        except ImportError as e:
            raise ImportError(f"无法导入 cann_bench 模块: {e}")

        return interfaces

    def print_interfaces(self, interfaces: List[InterfaceInfo]) -> None:
        """打印接口信息

        格式化显示扫描到的接口列表
        """
        print("")
        print("=" * 60)
        print("扫描到的 cann_bench 接口:")
        print("=" * 60)

        if not interfaces:
            print("  未找到任何接口")
            print("=" * 60)
            return

        for i, iface in enumerate(interfaces, 1):
            print(f"  {i}. {iface.signature}")

        print("")
        print(f"共 {len(interfaces)} 个算子接口")
        print("=" * 60)
        print("")

    def match_operators(self, interfaces: List[InterfaceInfo]) -> List[str]:
        """匹配kernel_bench中的算子

        根据接口名称，匹配kernel_bench中的算子定义
        返回匹配到的算子名称列表
        """
        from .operator_loader import OperatorLoader

        operator_loader = OperatorLoader(self.config.kernel_bench_root)
        all_operators = operator_loader.list_operators()  # 获取所有算子

        matched = []
        interface_names = [iface.name.lower() for iface in interfaces]

        for op_info in all_operators:
            # 尝试多种匹配方式
            op_name_lower = op_info.name.lower()
            op_func_name = op_info.get_function_name().lower()

            if op_name_lower in interface_names or op_func_name in interface_names:
                matched.append(op_info.name)

        return matched

    def prepare_from_source(
        self,
        source_dir: str,
        verbose: bool = False
    ) -> Tuple[List[str], PackageInfo]:
        """从源码目录准备评测环境

        流程：
        1. 扫描源码目录
        2. 检查/编译包
        3. 安装包
        4. 扫描接口
        5. 返回算子列表

        返回: (算子列表, 包信息)
        """
        print(f"[INFO] 扫描源码目录: {source_dir}")

        # Step 1: 扫描源码目录
        package_info = self.scan_source_dir(source_dir)

        if verbose:
            print(f"[INFO] 目录结构:")
            print(f"       - build.sh: {'存在' if package_info.has_build_sh else '不存在'}")
            print(f"       - dist目录: {'存在' if package_info.has_dist else '不存在'}")
            if package_info.whl_path:
                print(f"       - whl包: {Path(package_info.whl_path).name}")
            if package_info.run_path:
                print(f"       - run包: {Path(package_info.run_path).name}")

        # Step 2: 检查/编译包
        if package_info.whl_path:
            print(f"[INFO] 使用现有whl包: {Path(package_info.whl_path).name}")
        else:
            # 需要编译
            if not package_info.has_build_sh:
                raise RuntimeError(f"源码目录无build.sh且无whl包，无法编译")

            print(f"[INFO] 无现有whl包，执行编译...")
            package_info = self.build_packages(source_dir)

        # Step 3: 安装包
        if not self.install_packages(package_info):
            raise RuntimeError("包安装失败")

        # Step 4: 扫描接口
        interfaces = self.scan_interfaces()
        self.print_interfaces(interfaces)

        if not interfaces:
            raise RuntimeError("未扫描到任何cann_bench接口")

        # Step 5: 匹配算子
        matched_operators = self.match_operators(interfaces)

        if not matched_operators:
            print(f"[WARN] 未匹配到kernel_bench中的算子")
            return [], package_info

        print(f"[INFO] 匹配到 {len(matched_operators)} 个kernel_bench算子:")
        for op_name in matched_operators:
            print(f"       - {op_name}")

        return matched_operators, package_info

    def prepare_skip_build(self) -> List[str]:
        """跳过编译安装，直接扫描已安装的cann_bench

        返回: 算子列表
        """
        print(f"[INFO] 跳过编译安装，扫描已安装的cann_bench")

        # 扫描接口
        interfaces = self.scan_interfaces()
        self.print_interfaces(interfaces)

        if not interfaces:
            raise RuntimeError("未扫描到任何cann_bench接口，请先安装cann_bench包")

        # 匹配算子
        matched_operators = self.match_operators(interfaces)

        if not matched_operators:
            print(f"[WARN] 未匹配到kernel_bench中的算子")
            return []

        print(f"[INFO] 匹配到 {len(matched_operators)} 个kernel_bench算子:")
        for op_name in matched_operators:
            print(f"       - {op_name}")

        return matched_operators