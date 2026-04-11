import torch
from typing import List

"""
ForeachNorm 算子 Torch Golden 参考实现

对输入张量列表的每个张量进行范数运算
公式：y = (sum |x_i|^p)^(1/p)
"""
def foreach_norm(
    x: List[torch.Tensor], scalar: float
) -> List[torch.Tensor]:
    """
    对输入张量列表的每个张量进行范数运算

    公式：y = (sum |x_i|^p)^(1/p)

    Args:
        x: 输入张量列表 (TensorList)
        scalar: 范数阶数

    Returns:
        输出张量列表，每个张量的范数结果
    """

    y = [torch.norm(tensor, p=scalar) for tensor in x]
    return y
