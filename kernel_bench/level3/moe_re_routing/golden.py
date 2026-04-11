import torch

"""
MoeReRouting算子Torch Golden参考实现

将token按照专家顺序重新排列
公式: y = moe_rerouting(x, expert_token_num_per_rank)
"""
def moe_re_routing(
    x: torch.Tensor, expert_token_num_per_rank: torch.Tensor, expert_token_num_type: int = 1
) -> torch.Tensor:
    """
    将token按照专家顺序重新排列
    
    公式: y = moe_rerouting(x, expert_token_num_per_rank)
    
    Args:
        x: 输入token张量
        expert_token_num_per_rank: 每个rank的token数量
        expert_token_num_type: 专家token数量类型
    
    Returns:
        输出张量，重新排列的token
    """

    expert_token_num = expert_token_num_per_rank.tolist()
    if isinstance(expert_token_num[0], list):
        expert_token_num = [item for sublist in expert_token_num for item in sublist]
    
    indices = []
    for num_tokens in expert_token_num:
        num = int(num_tokens)
        indices.extend(range(len(indices), len(indices) + num))
    
    if len(indices) > x.shape[0]:
        indices = indices[:x.shape[0]]
    elif len(indices) < x.shape[0]:
        indices.extend(range(len(indices), x.shape[0]))
    
    y = x[indices]
    return y
