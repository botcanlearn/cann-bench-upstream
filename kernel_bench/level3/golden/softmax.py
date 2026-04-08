import torch

"""
Softmax算子Torch Golden参考实现

对输入张量在指定维度上计算Softmax值
公式: y[i] = exp(x[i]) / sum(exp(x[j])) 沿指定轴
"""
def softmax(
    x: torch.Tensor, dim: int = -1
) -> torch.Tensor:
    """
    对输入张量在指定维度上计算Softmax值
    
    公式: y[i] = exp(x[i]) / sum(exp(x[j])) 沿指定轴
    
    Args:
        x: 输入张量
        dim: 计算softmax的维度 (取值范围: -ndim ~ ndim-1, 负值表示从后往前数)
    
    Returns:
        输出张量，与输入shape相同
    """

    y = torch.nn.functional.softmax(x, dim=dim)
    return y
