import torch
from typing import List

"""
Concat 算子 Torch Golden 参考实现

沿指定维度将多个输入 Tensor 进行拼接
公式：y = Concat(x[0], ..., x[N-1], axis)
"""
def concat(
    x: List[torch.Tensor], concat_dim: int
) -> torch.Tensor:
    """
    沿指定维度将多个输入 Tensor 进行拼接

    公式：y = Concat(x[0], ..., x[N-1], axis)

    Args:
        x: 输入张量列表 (TensorList)
        concat_dim: 拼接维度，指定沿哪个维度拼接输入张量 (取值范围：-ndim ~ ndim-1)

    Returns:
        输出张量，拼接结果
    """

    y = torch.cat(x, dim=concat_dim)
    return y
