import torch

"""
MatmulAllReduce算子Torch Golden参考实现

矩阵乘法与AllReduce通信的融合
公式: y = AllReduce(Matmul(x1, x2))
"""
def matmul_all_reduce(
    x1: torch.Tensor, x2: torch.Tensor, group: str, reduce_op: str = 'sum', is_trans_b: bool = False
) -> torch.Tensor:
    """
    矩阵乘法与AllReduce通信的融合
    
    公式: y = AllReduce(Matmul(x1, x2))
    
    Args:
        x1: 第1个输入矩阵
        x2: 第2个输入矩阵
        group: 通信组
        reduce_op: 归约操作
        is_trans_b: 是否转置x2
    
    Returns:
        输出张量
    """

    x2_adj = x2.transpose(-2, -1) if is_trans_b else x2
    y = torch.matmul(x1, x2_adj)
    return y
