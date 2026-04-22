"""
精度校验器

职责：
1. 使用 MERE/MARE 检验浮点输出精度
2. 使用精确匹配检验整型输出
3. 处理 NaN/Inf 等特殊值
"""

import torch
from dataclasses import dataclass
from typing import Optional


# MERE threshold per dtype (CANN community standard)
# Pass criteria: MERE < threshold AND MARE < 10 * threshold
MERE_THRESHOLDS = {
    "float16":  2 ** (-10),   # ~0.000977
    "bfloat16": 2 ** (-7),    # ~0.00781
    "float32":  2 ** (-13),   # ~0.000122
    "float64":  2 ** (-13),
}


@dataclass
class PrecisionResult:
    """精度校验结果"""
    passed: bool
    detail: str
    mere: float = 0.0
    mare: float = 0.0
    threshold: float = 0.0
    mare_limit: float = 0.0
    special_values_ok: bool = True


class PrecisionChecker:
    """精度校验器

    浮点类型使用 MERE (Mean Element-wise Relative Error) 和
    MARE (Max Absolute Relative Error) 作为精度指标。
    整型使用精确匹配。
    """

    def __init__(self, thresholds: Optional[dict] = None):
        self.thresholds = thresholds or MERE_THRESHOLDS

    def check(self, golden: torch.Tensor, actual: torch.Tensor, dtype_str: str) -> PrecisionResult:
        """比较 golden 输出和 custom kernel 输出

        Args:
            golden: 参考输出 (通常为 CPU fp32)
            actual: 自定义算子输出 (NPU, 原生 dtype)
            dtype_str: 数据类型字符串 (e.g. "float16", "int32")

        Returns:
            PrecisionResult 包含通过/失败状态及详细指标
        """
        # Integer types: exact match
        if dtype_str in ("int8", "int16", "int32", "int64", "uint8", "bool"):
            return self._check_exact(golden, actual)

        # Float types: MERE/MARE
        return self._check_float(golden, actual, dtype_str)

    def _check_exact(self, golden: torch.Tensor, actual: torch.Tensor) -> PrecisionResult:
        """整型精确匹配"""
        passed = torch.equal(golden, actual)
        if passed:
            return PrecisionResult(passed=True, detail="exact match")
        mismatch = (golden != actual).sum().item()
        return PrecisionResult(passed=False, detail=f"exact mismatch: {mismatch} elements differ")

    def _check_float(self, golden: torch.Tensor, actual: torch.Tensor, dtype_str: str) -> PrecisionResult:
        """浮点 MERE/MARE 校验"""
        go = golden.flatten().float()
        co = actual.flatten().float()
        eps = 1e-38  # avoid division by zero

        # Check special values (NaN/Inf)
        nan_match = torch.isnan(go) == torch.isnan(co)
        inf_match = (go == float('inf')) == (co == float('inf'))
        ninf_match = (go == float('-inf')) == (co == float('-inf'))
        special_ok = nan_match.all().item() and inf_match.all().item() and ninf_match.all().item()

        if not special_ok:
            return PrecisionResult(
                passed=False, detail="NaN/Inf mismatch",
                special_values_ok=False,
            )

        # Mask out non-finite values for relative error computation
        valid = torch.isfinite(go) & torch.isfinite(co)
        if valid.sum() == 0:
            return PrecisionResult(passed=True, detail="all special values matched")

        relative_errors = (go - co).abs() / (go.abs() + eps)
        re_valid = relative_errors[valid]
        mere = re_valid.mean().item()
        mare = re_valid.max().item()

        threshold = self.thresholds.get(dtype_str, 2 ** (-13))
        mare_limit = 10 * threshold

        passed = mere < threshold and mare < mare_limit
        if passed:
            detail = f"MERE={mere:.2e} MARE={mare:.2e}"
        else:
            detail = (f"MERE={mere:.2e}(limit {threshold:.2e}) "
                      f"MARE={mare:.2e}(limit {mare_limit:.2e})")

        return PrecisionResult(
            passed=passed, detail=detail,
            mere=mere, mare=mare,
            threshold=threshold, mare_limit=mare_limit,
        )
