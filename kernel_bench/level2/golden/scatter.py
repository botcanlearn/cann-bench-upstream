import torch

"""
Scatter算子Torch Golden参考实现

将updates按索引indices更新到data中
公式: y[i] = updates[j] where indices[j] == i
"""
def scatter(
    data: torch.Tensor, indices: torch.Tensor, updates: torch.Tensor, dim: int, reduce: str = None
) -> torch.Tensor:
    """
    将updates按索引indices更新到data中

    公式: y[i] = updates[j] where indices[j] == i

    Args:
        data: 输入数据张量
        indices: 索引张量
        updates: 更新值张量
        dim: 沿哪个维度scatter
        reduce: 聚合方式

    Returns:
        输出张量，scatter结果
    """

    y = data.clone()
    if reduce is None or reduce == 'update':
        y.scatter_(dim, indices.long(), updates)
    elif reduce == 'add':
        y.scatter_add_(dim, indices.long(), updates)
    return y
