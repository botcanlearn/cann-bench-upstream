"""CANN Bench Simple - 不注册torch.ops的简化版本"""
__version__ = "1.0.0"

import torch

try:
    from . import _C
except ImportError:
    raise ImportError("Cannot import _C. Run: pip install .")

def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return _C.add(x, y)

def sqrt(x: torch.Tensor) -> torch.Tensor:
    return _C.sqrt(x)