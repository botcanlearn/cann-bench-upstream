import torch

"""
Mish算子Torch Golden参考实现

自正则化的非单调神经网络激活函数
公式: y = x * tanh(softplus(x))
"""
def mish(
    x: torch.Tensor
) -> torch.Tensor:
    """
    自正则化的非单调神经网络激活函数

    公式: y = x * tanh(softplus(x))

    Args:
        x: 输入张量

    Returns:
        输出张量，Mish激活结果
    """
    return torch.nn.functional.mish(x)
