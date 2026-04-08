import torch

"""
RMSNorm算子Torch Golden参考实现

大模型常用归一化操作
公式: y = x * gamma / sqrt(mean(x^2) + epsilon)
"""
def rms_norm(
    x: torch.Tensor, gamma: torch.Tensor, epsilon: float = 1e-6
) -> torch.Tensor:
    """
    大模型常用归一化操作
    
    公式: y = x * gamma / sqrt(mean(x^2) + epsilon)
    
    Args:
        x: 输入张量
        gamma: 缩放参数
        epsilon: 防止除零的小值
    
    Returns:
        归一化后的输出张量
    """

    rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + epsilon)
    y = (x / rms) * gamma
    return y
