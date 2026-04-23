"""
数据层模块

职责：
1. 算子定义加载（proto.yaml解析）
2. 测试用例加载（cases.yaml解析）
3. Golden函数加载（动态导入）
4. 数据生成（根据shape/dtype生成输入张量）
5. 包管理（源码扫描、编译、安装、接口扫描）
"""

from .operator_loader import OperatorLoader, OperatorInfo
from .case_loader import CaseLoader, CaseInfo
from .golden_loader import GoldenLoader
from .data_generator import DataGenerator
from .package_manager import PackageManager, PackageInfo, InterfaceInfo

__all__ = [
    "OperatorLoader", "OperatorInfo",
    "CaseLoader", "CaseInfo",
    "GoldenLoader",
    "DataGenerator",
    "PackageManager", "PackageInfo", "InterfaceInfo",
]