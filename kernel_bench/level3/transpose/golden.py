import torch

"""
Transpose算子Torch Golden参考实现

对tensor的任意维度进行调换
公式: y[i0,...,in-1] = x[i_perm[0],...,i_perm[n-1]]
"""
def transpose(
    x: torch.Tensor, perm: list
) -> torch.Tensor:
    """
    对tensor的任意维度进行调换
    
    公式: y[i0,...,in-1] = x[i_perm[0],...,i_perm[n-1]]
    
    Args:
        x: 输入张量
        perm: 维度置换顺序
    
    Returns:
        输出张量，转置后的结果
    """

    y = torch.permute(x, perm)
    return y
