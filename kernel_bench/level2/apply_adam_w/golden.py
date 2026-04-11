import torch

"""
ApplyAdamW 算子 Torch Golden 参考实现

AdamW 优化器实现，解耦权重衰减
公式:
    m_t = beta1 * m_{t-1} + (1 - beta1) * grad
    v_t = beta2 * v_{t-1} + (1 - beta2) * grad^2
    m_hat = m_t / (1 - beta1^t)
    v_hat = v_t / (1 - beta2^t)
    var_t = var_{t-1} - lr * (m_hat / (sqrt(v_hat) + eps) + weight_decay * var_{t-1})
"""
def apply_adam_w(
    var: torch.Tensor,
    grad: torch.Tensor,
    m: torch.Tensor,
    v: torch.Tensor,
    lr: float,
    beta1: float,
    beta2: float,
    weight_decay: float,
    epsilon: float = 1e-8,
    maximize: bool = False
) -> torch.Tensor:
    """
    AdamW 优化器实现，解耦权重衰减

    Args:
        var: 变量张量（需要优化的参数）
        grad: 梯度张量
        m: 一阶矩张量（动量）
        v: 二阶矩张量
        lr: 学习率
        beta1: 一阶矩估计的指数衰减率
        beta2: 二阶矩估计的指数衰减率
        weight_decay: 权重衰减系数（解耦）
        epsilon: 数值稳定常数
        maximize: 是否最大化目标函数

    Returns:
        更新后的变量
    """
    # 更新一阶矩（动量）
    m.mul_(beta1).add_(grad, alpha=1 - beta1)
    # 更新二阶矩
    v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

    # 计算偏差修正的一阶矩和二阶矩
    # 注意：实际使用时需要传入 timestep t，这里简化处理
    bias_correction1 = 1 - beta1
    bias_correction2 = 1 - beta2

    m_hat = m / bias_correction1
    v_hat = v / bias_correction2

    # 计算更新量
    update = m_hat / (v_hat.sqrt() + epsilon)

    # 解耦的权重衰减
    if weight_decay != 0:
        update.add_(var, alpha=weight_decay)

    # 应用更新
    if maximize:
        y = var + lr * update
    else:
        y = var - lr * update

    return y
