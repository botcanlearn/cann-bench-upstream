"""
工具层模块

职责：
1. 设备管理（CPU/NPU切换、张量迁移）
2. 数据类型映射（字符串与torch.dtype转换）
3. 参数构建（根据函数签名构建调用参数）
4. 精度验证（MERE/MARE标准）
5. Baseline解析（多硬件支持）
"""

from .device_manager import DeviceManager, DeviceConfig
from .dtype_mapper import str_to_torch_dtype, torch_dtype_to_str, is_float_dtype, is_int_dtype
from .param_builder import ParamBuilder
from .precision import compare_tensors, CompareResult, PRECISION_THRESHOLDS
from .baseline_resolver import (
    BaselineResolver, BaselineInfo,
    resolve_baseline_us, resolve_baseline_info,
    calculate_speedup, geometric_mean_speedup,
)

__all__ = [
    "DeviceManager", "DeviceConfig",
    "str_to_torch_dtype", "torch_dtype_to_str", "is_float_dtype", "is_int_dtype",
    "ParamBuilder",
    "compare_tensors", "CompareResult", "PRECISION_THRESHOLDS",
    "BaselineResolver", "BaselineInfo",
    "resolve_baseline_us", "resolve_baseline_info",
    "calculate_speedup", "geometric_mean_speedup",
]