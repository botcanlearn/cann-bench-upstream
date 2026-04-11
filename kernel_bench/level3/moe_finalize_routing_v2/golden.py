import torch

"""
MoeFinalizeRoutingV2 算子 Torch Golden 参考实现

合并 MoE FFN 的输出结果，使用路由权重对专家输出进行加权求和
公式：output = sum(routing_weights * expert_outputs)
"""
def moe_finalize_routing_v2(
    expert_outputs: torch.Tensor,
    routing_weights: torch.Tensor,
    sorted_token_indices: torch.Tensor = None,
    num_tokens: int = None,
    renormalize: bool = False
) -> torch.Tensor:
    """
    合并 MoE FFN 的输出结果

    对每个 token，使用路由权重对其分配的专家输出进行加权求和

    Args:
        expert_outputs: 专家输出张量，形状为 (num_tokens * topk, hidden_dim)
                       或 (num_tokens, topk, hidden_dim)
        routing_weights: 路由权重，形状为 (num_tokens, topk)，经过 softmax 归一化
        sorted_token_indices: 可选，排序后的 token 索引，用于恢复原始顺序
        num_tokens: 可选，token 数量
        renormalize: 是否重新归一化权重

    Returns:
        输出张量，形状为 (num_tokens, hidden_dim)
    """

    # 处理输入形状
    if expert_outputs.dim() == 2:
        # 形状：(num_tokens * topk, hidden_dim) -> (num_tokens, topk, hidden_dim)
        if num_tokens is None:
            num_tokens = expert_outputs.shape[0] // routing_weights.shape[1]
        topk = routing_weights.shape[1]
        hidden_dim = expert_outputs.shape[1]
        expert_outputs = expert_outputs.view(num_tokens, topk, hidden_dim)
    elif expert_outputs.dim() == 3:
        # 形状：(num_tokens, topk, hidden_dim)
        num_tokens = expert_outputs.shape[0]

    # 可选：重新归一化权重
    if renormalize:
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)

    # 加权求和：output[i] = sum_k(routing_weights[i, k] * expert_outputs[i, k])
    # 扩展权重形状以匹配专家输出：(num_tokens, topk) -> (num_tokens, topk, 1)
    output = (expert_outputs * routing_weights.unsqueeze(-1)).sum(dim=1)

    # 可选：恢复原始顺序
    if sorted_token_indices is not None:
        output = output[torch.argsort(sorted_token_indices)]

    return output
