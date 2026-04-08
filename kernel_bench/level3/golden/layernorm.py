import torch
from typing import List, Optional, Tuple

"""
LayerNorm 算子 Torch Golden 参考实现

对指定层进行归一化计算
公式：y = (x - mean) / sqrt(variance + eps) * gamma + beta
"""
def layer_norm(
    x: torch.Tensor,
    normalized_shape: List[int],
    gamma: Optional[torch.Tensor] = None,
    beta: Optional[torch.Tensor] = None,
    epsilon: float = 1e-5
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    对指定层进行归一化计算

    公式：y = (x - mean) / sqrt(variance + eps) * gamma + beta

    Args:
        x: 输入张量
        normalized_shape: 归一化的维度形状，指定最后 D 个维度进行归一化
        gamma: 缩放参数，形状与 normalized_shape 一致（可选）
        beta: 偏移参数，形状与 normalized_shape 一致（可选）
        epsilon: 防止除零的小值

    Returns:
        y: 归一化后的输出张量
        mean: 均值
        variance: 方差
    """
    # 计算归一化的维度
    ndim = x.dim()
    norm_ndim = len(normalized_shape)
    begin_norm_axis = ndim - norm_ndim

    # 计算均值和方差
    dims = tuple(range(begin_norm_axis, ndim))
    mean = x.mean(dim=dims, keepdim=True)
    variance = x.var(dim=dims, keepdim=True, unbiased=False)

    # LayerNorm 计算
    y = (x - mean) / torch.sqrt(variance + epsilon)

    # 应用 gamma 和 beta
    if gamma is not None:
        y = y * gamma
    if beta is not None:
        y = y + beta

    # 压缩均值和方差到原始维度
    mean = mean.squeeze()
    variance = variance.squeeze()

    return y, mean, variance
