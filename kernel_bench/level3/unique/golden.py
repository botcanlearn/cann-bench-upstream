import torch

"""
Unique 算子 Torch Golden 参考实现

去除张量中的重复元素
公式：y, inverse = unique(x, return_inverse)
"""
def unique(
    x: torch.Tensor,
    return_inverse: bool = False
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    去除张量中的重复元素

    公式：y, inverse = unique(x, return_inverse)

    Args:
        x: 输入张量
        return_inverse: 是否返回逆索引，用于重建原始张量

    Returns:
        y: 唯一值张量
        inverse: 逆索引，满足 x = y[inverse] (当 return_inverse=True 时)
    """

    if return_inverse:
        y, inverse = torch.unique(x, return_inverse=True)
        return y, inverse
    else:
        y = torch.unique(x, return_inverse=False)
        return y, None
