import torch

"""
Sort算子Torch Golden参考实现

对输入张量进行排序
公式: y = sort(x, dim, descending)
"""
def sort(
    x: torch.Tensor, dim: int, stable: bool = False, descending: bool = False
) -> torch.Tensor:
    """
    对输入张量进行排序
    
    公式: y = sort(x, dim, descending)
    
    Args:
        x: 输入张量
        stable: 是否稳定排序 (相等元素的原始相对顺序保持不变)
        dim: 排序维度 (取值范围: -ndim ~ ndim-1)
        descending: 是否降序排序
    
    Returns:
        输出张量，排序结果
    """

    y = torch.sort(x, dim=dim, descending=descending)[0]
    return y
