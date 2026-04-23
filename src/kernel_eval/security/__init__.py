"""
安全防护层

职责：
1. Timing API防护（防止monkey-patch攻击）
2. 返回值类型检查（防止懒求值/子类伪装攻击）
3. 二次验证支持

使用方法：
    from kernel_eval.security import APIGuard, check_output_type

    # Timing API防护
    guard = APIGuard()
    guard.snapshot()  # 安装wheel前快照
    install_wheel(path)
    guard.verify()    # 验证完整性
    guard.restore()   # 程序退出前恢复
"""

from .api_guard import APIGuard, snapshot_timing_apis, verify_timing_apis, restore_timing_apis
from .type_checker import check_output_type, check_tensor_validity

__all__ = [
    "APIGuard",
    "snapshot_timing_apis",
    "verify_timing_apis",
    "restore_timing_apis",
    "check_output_type",
    "check_tensor_validity",
]