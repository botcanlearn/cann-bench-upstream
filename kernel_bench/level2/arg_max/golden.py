import torch

"""
ArgMax 算子 Torch Golden 参考实现

返回张量在指定维度上的最大值的索引
公式: y = argmax(x, axis=dim)
"""
def arg_max(
    x: torch.Tensor, dim: int, keepdims: bool = False
) -> torch.Tensor:
    """
    返回张量在指定维度上的最大值的索引

    公式: y = argmax(x, axis=dim)

    Args:
        x: 输入张量
        dim: 计算 argmax 的维度
        keepdims: 是否保持维度

    Returns:
        输出张量，最大值的索引
    """

    y = torch.argmax(x, dim=dim, keepdim=keepdims)
    return y
