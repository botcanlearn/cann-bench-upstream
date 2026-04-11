import torch

"""
TopK算子Torch Golden参考实现

返回k个最大或最小的元素及其索引
公式: y, idx = topk(x, k, dim)
"""
def top_k(
    x: torch.Tensor, k: int, dim: int, largest: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    返回k个最大或最小的元素及其索引
    
    公式: y, idx = topk(x, k, dim)
    
    Args:
        x: 输入张量
        k: 返回的topk数量 (取值范围: 1 <= k <= dim_size)
        dim: 排序维度 (取值范围: -ndim ~ ndim-1)
        largest: 是否返回最大值 (false时返回最小值)
    
    Returns:
        y, idx
    """

    values, indices = torch.topk(x, k=k, dim=dim, largest=largest)
    return values, indices
