import torch

"""
CTCLoss 算子 Torch Golden 参考实现

计算连续时间序列与目标序列之间的 CTC (Connectionist Temporal Classification) 损失

公式: L = -log P(z|x) where z is the target sequence

参考 PyTorch API: torch.nn.CTCLoss
    https://pytorch.org/docs/stable/generated/torch.nn.CTCLoss.html

Parameters:
    - log_probs: (T, N, C) 或 (N, T, C) - 对数概率，T=时间步，N=batch，C=类别数（含blank）
    - targets: (N, S) 或 flattened (sum(target_lengths)) - 目标序列
    - input_lengths: (N,) - 每个样本的有效输入长度
    - target_lengths: (N,) - 每个样本的目标序列长度
    - blank: int, 默认 0 - blank 标签索引
    - reduction: 'none' | 'mean' | 'sum', 默认 'mean' - 损失聚合方式
    - zero_infinity: bool, 默认 False - 是否将无穷损失置零
"""


def ctc_loss(
    log_probs: torch.Tensor,
    targets: torch.Tensor,
    input_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
    blank: int = 0,
    reduction: str = 'mean',
    zero_infinity: bool = False
) -> torch.Tensor:
    """
    计算 CTC (Connectionist Temporal Classification) 损失

    Args:
        log_probs: 对数概率张量，shape (T, N, C)
                   T = 时间步数，N = batch size，C = 类别数（包含 blank）
                   要求 log_probs 已经是 log 概率（通常由 log_softmax 得到）
        targets: 目标序列张量，flattened shape (sum(target_lengths),)
                 值域 [0, C-1]（不含 blank 或包含 blank 取决于实现）
        input_lengths: 每个样本的有效输入长度，shape (N,)
                       值域 [1, T]，表示每个样本实际使用的输入帧数
        target_lengths: 每个样本的目标序列长度，shape (N,)
                        值域 [0, S]，表示每个样本的实际目标长度
        blank: blank 标签的索引，默认 0
               在 log_probs 的最后一维中，blank 位置的索引
        reduction: 损失聚合方式，可选 'none' | 'mean' | 'sum'
                   'none': 返回每个样本的损失，shape (N,)
                   'mean': 返回 batch 平均损失
                   'sum': 返回 batch 总损失
        zero_infinity: 是否将无穷大的损失置零，默认 False
                       当目标序列过长导致无法对齐时，损失可能为 inf

    Returns:
        损失值：如果 reduction='none'，返回 shape (N,) 的张量
               否则返回标量张量
    """
    # CTC Loss 在 CPU 上不支持 float16/bfloat16，需要转换到 float32
    original_dtype = log_probs.dtype
    if log_probs.dtype in [torch.float16, torch.bfloat16]:
        log_probs = log_probs.float()

    # 直接调用 PyTorch 标准 CTCLoss 实现
    loss = torch.nn.functional.ctc_loss(
        log_probs=log_probs,
        targets=targets,
        input_lengths=input_lengths,
        target_lengths=target_lengths,
        blank=blank,
        reduction=reduction,
        zero_infinity=zero_infinity
    )

    # 如果输入是half类型，输出也转回half
    if original_dtype in [torch.float16, torch.bfloat16]:
        loss = loss.to(original_dtype)

    return loss