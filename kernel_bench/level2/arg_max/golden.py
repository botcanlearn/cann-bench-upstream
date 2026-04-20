import torch

"""
ArgMax 算子 Torch Golden 参考实现

返回张量在指定维度上的最大值的索引
公式: y = argmax(x, axis=dimension)
"""
def arg_max(
    x: torch.Tensor, dimension: int
) -> torch.Tensor:
    """
    返回张量在指定维度上的最大值的索引

    公式: y = argmax(x, axis=dimension)

    Args:
        x: 输入张量
        dimension: 计算 argmax 的维度

    Returns:
        输出张量，最大值的索引（int64）
    """

    y = torch.argmax(x, dim=dimension)
    return y
