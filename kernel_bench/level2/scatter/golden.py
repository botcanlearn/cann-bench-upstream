import torch

"""
Scatter算子Torch Golden参考实现

将updates按索引indices更新到data中
公式: y[index[i][j][k]] = src[i][j][k] (if dim == 0)
"""
def scatter(
    data: torch.Tensor, indices: torch.Tensor, updates: torch.Tensor, dim: int, reduce: str = None
) -> torch.Tensor:
    """
    将updates按索引indices更新到data中

    公式: y[index[i][j][k]] = src[i][j][k] (if dim == 0)

    Args:
        data: 输入数据张量
        indices: 索引张量
        updates: 更新值张量
        dim: 沿哪个维度scatter
        reduce: 聚合方式 (None/update, add, multiply, amin, amax)

    Returns:
        输出张量，scatter结果
    """

    y = data.clone()
    idx = indices.long()
    if reduce is None or reduce == 'update':
        y.scatter_(dim, idx, updates)
    elif reduce == 'add':
        y.scatter_add_(dim, idx, updates)
    elif reduce == 'multiply':
        y.scatter_reduce_(dim, idx, updates, reduce="prod", include_self=True)
    elif reduce == 'amin':
        y.scatter_reduce_(dim, idx, updates, reduce="amin", include_self=True)
    elif reduce == 'amax':
        y.scatter_reduce_(dim, idx, updates, reduce="amax", include_self=True)
    return y
