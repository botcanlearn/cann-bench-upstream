import torch

"""
ApplyRotaryPosEmb 算子 Torch Golden 参考实现

对 query 和 key 执行旋转位置编码 (RoPE) 计算
公式:
    rotate_half(x) = concat(-x[head_dim/2:], x[:head_dim/2])
    y = (x * cos) + (rotate_half(x) * sin)

参考:
    - RoFormer: https://arxiv.org/abs/2104.09864
    - LLaMA: https://github.com/meta-llama/llama
    - HuggingFace transformers: https://huggingface.co/docs/transformers/internal/rope_utils
"""
def apply_rotary_pos_emb(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    layout: int = 0,
    rotaryMode: str = 'half'
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    对 query 和 key 执行旋转位置编码 (RoPE) 计算

    Args:
        query: 查询张量，shape 为 (B, S, N, D) 或 (B, N, S, D)
        key: 键张量，shape 同 query
        cos: 余弦位置编码，shape 为 (S, D/2) 或 (B, S, D/2)
        sin: 正弦位置编码，shape 同 cos
        layout: 输入布局 (0: [B,S,N,D], 1: [B,N,S,D])
        rotaryMode: 旋转模式 ("half": 连续半分式，"interleaved": 交错式)

    Returns:
        query_out: 旋转后的查询张量
        key_out: 旋转后的键张量

    Examples:
        >>> B, S, N, D = 2, 4, 8, 128
        >>> query = torch.randn(B, S, N, D)
        >>> key = torch.randn(B, S, N, D)
        >>> cos = torch.randn(S, D // 2)
        >>> sin = torch.randn(S, D // 2)
        >>> q_out, k_out = apply_rotary_pos_emb(query, key, cos, sin)
    """

    def rotate_half(x: torch.Tensor, mode: str) -> torch.Tensor:
        """
        旋转输入张量的一半维度

        Args:
            x: 输入张量
            mode: 旋转模式

        Returns:
            旋转后的张量
        """
        if mode == 'interleaved':
            # GPT-J 风格的交错式旋转
            x1 = x[..., ::2]       # 取偶数索引
            x2 = x[..., 1::2]      # 取奇数索引
            rotated = torch.stack([-x2, x1], dim=-1).flatten(-2)
        else:
            # LLaMA/Meta 风格的连续半分式旋转
            half_dim = x.shape[-1] // 2
            x1 = x[..., :half_dim]
            x2 = x[..., half_dim:]
            rotated = torch.cat([-x2, x1], dim=-1)
        return rotated

    def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, mode: str) -> torch.Tensor:
        """
        对单个张量应用 RoPE

        Args:
            x: 输入张量
            cos: 余弦编码
            sin: 正弦编码
            mode: 旋转模式

        Returns:
            旋转后的张量
        """
        # 调整 cos/sin 的 shape 以匹配输入
        # cos/sin: (S, D/2) 或 (B, S, D/2)
        # 需要扩展到 (B, S, N, D) 或 (B, N, S, D)

        if cos.dim() == 2:
            # cos: (S, D/2) -> 需要扩展到 (B, S, 1, D)
            cos = cos.unsqueeze(0).unsqueeze(2)  # (1, S, 1, D/2)
            sin = sin.unsqueeze(0).unsqueeze(2)
        elif cos.dim() == 3:
            # cos: (B, S, D/2) -> 需要扩展到 (B, S, 1, D)
            cos = cos.unsqueeze(2)  # (B, S, 1, D/2)
            sin = sin.unsqueeze(2)

        # 如果 layout=1 (B,N,S,D)，需要调整
        if layout == 1:
            cos = cos.transpose(1, 2)  # (B, 1, S, D/2)
            sin = sin.transpose(1, 2)

        # 重复 cos/sin 到完整的 head_dim
        cos = cos.repeat(1, 1, 1, 2) if cos.dim() == 4 else cos.repeat_interleave(2, dim=-1)
        sin = sin.repeat(1, 1, 1, 2) if sin.dim() == 4 else sin.repeat_interleave(2, dim=-1)

        # 应用 RoPE 公式
        x_rotate = rotate_half(x, mode)
        return (x * cos) + (x_rotate * sin)

    # 对 query 和 key 分别应用 RoPE
    query_out = apply_rotary(query, cos, sin, rotaryMode)
    key_out = apply_rotary(key, cos, sin, rotaryMode)

    return query_out, key_out
