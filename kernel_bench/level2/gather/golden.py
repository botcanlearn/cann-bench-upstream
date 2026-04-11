import torch

"""
Gather算子Torch Golden参考实现

从输入Tensor的指定维度按index提取元素
公式: y[i][m][n] = x[index[i]][m][n]
"""
def gather(
    x: torch.Tensor, index: torch.Tensor, batch_dims: int = 0
) -> torch.Tensor:
    """
    从输入Tensor的指定维度按index提取元素
    
    公式: y[i][m][n] = x[index[i]][m][n]
    
    Args:
        x: 输入张量
        index: 索引张量
        batch_dims: batch维度数
    
    Returns:
        输出张量，gather结果
    """

    y = torch.gather(x, batch_dims, index.long())
    return y
