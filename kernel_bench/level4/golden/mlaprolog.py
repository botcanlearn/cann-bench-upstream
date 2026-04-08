import torch

"""
MlaProlog算子Torch Golden参考实现

Multi-Head Latent Attention前处理
公式: y = Query/Key计算 + RmsNorm + ROPE编码
"""
def mla_prolog(
    token_x: torch.Tensor, wk_cq: torch.Tensor, wk_ckv: torch.Tensor, rmsnormEpsilonCq: float, rmsnormEpsilonCkv: float
) -> torch.Tensor:
    """
    Multi-Head Latent Attention前处理
    
    公式: y = Query/Key计算 + RmsNorm + ROPE编码
    
    Args:
        token_x: 输入token张量
        wk_cq: CQ权重张量
        wk_ckv: CKV权重张量
        rmsnormEpsilonCq: CQ的RMSNorm epsilon
        rmsnormEpsilonCkv: CKV的RMSNorm epsilon
    
    Returns:
        输出张量
    """

    cq = torch.matmul(token_x, wk_cq.transpose(-2, -1))
    variance_cq = cq.pow(2).mean(-1, keepdim=True)
    rms_cq = torch.sqrt(variance_cq + rmsnormEpsilonCq)
    cq_norm = cq / rms_cq
    
    ckv = torch.matmul(token_x, wk_ckv.transpose(-2, -1))
    variance_ckv = ckv.pow(2).mean(-1, keepdim=True)
    rms_ckv = torch.sqrt(variance_ckv + rmsnormEpsilonCkv)
    ckv_norm = ckv / rms_ckv
    
    y = torch.cat([cq_norm, ckv_norm], dim=-1)
    return y
