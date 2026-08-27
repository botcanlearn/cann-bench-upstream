# FusedRMSNormPosQKVQKNorm 算子

## 算子简介

将 RMSNorm、位置编码加法、QKV线性投影、Q/K归一化融合为单个算子，减少中间Tensor的内存开销。

## 算子定义

### 计算流程

```
h = rmsnorm(x, gamma, eps) + pos_emb
qkv = h @ Wqkv
q, k, v = split(qkv, 3, dim=-1)
q = rmsnorm(q, gamma_q, eps)
k = rmsnorm(k, gamma_k, eps)
output = (q, k, v)
```

其中 `rmsnorm(x, g, eps) = x / sqrt(mean(x^2) + eps) * g`

## 接口规范

```python
def fused_rmsnorm_pos_qkv_qknorm(x, gamma, pos_emb, Wqkv, gamma_q, gamma_k,
                                   num_heads=None, eps=1e-6):
    """
    Args:
        x: Tensor[B, S, D] - 输入隐状态
        gamma: Tensor[D] - 第一个RMSNorm权重
        pos_emb: Tensor[1, S, D] - 位置编码
        Wqkv: Tensor[D, 3*H] - QKV投影矩阵
        gamma_q: Tensor[H] - Q归一化权重
        gamma_k: Tensor[H] - K归一化权重
        num_heads: int - 注意力头数
        eps: float - RMSNorm epsilon

    Returns:
        q: Tensor[B, S, H] - 归一化后的Q
        k: Tensor[B, S, H] - 归一化后的K
        v: Tensor[B, S, H] - V投影
    """
```

### 支持的数据类型

| 输入/输出 | dtype |
|-----------|-------|
| float16 | 支持 |
| bfloat16 | 支持 |

### 支持的参数范围

- B: 1~8
- S: 64~2048
- D: 128~1024
- H = D (head_dim * num_heads)
- num_heads: 2~16

## 精度要求

| dtype | Threshold | 通过条件 |
|-------|-----------|---------|
| float16 | 2^-10 ≈ 0.00098 | MERE < T 且 MARE < 10T |
| bfloat16 | 2^-7 ≈ 0.0078 | 同上 |

## 使用示例

```python
import cann_bench

x = torch.randn(1, 128, 256, dtype=torch.float16, device='npu')
gamma = torch.ones(256, dtype=torch.float16, device='npu')
pos_emb = torch.randn(1, 128, 256, dtype=torch.float16, device='npu') * 0.1
Wqkv = torch.randn(256, 768, dtype=torch.float16, device='npu') * 0.05
gamma_q = torch.ones(256, dtype=torch.float16, device='npu')
gamma_k = torch.ones(256, dtype=torch.float16, device='npu')
q, k, v = cann_bench.fused_rmsnorm_pos_qkv_qknorm(
    x, gamma, pos_emb, Wqkv, gamma_q, gamma_k, num_heads=4, eps=1e-6
)
```

## 标准Golden代码

全程 fp32 计算，只在最终 return 时转回原 dtype。

```python
import torch
from typing import Tuple

def rmsnorm(x, gamma, eps=1e-6):
    """RMS Normalization (keeps fp32, no intermediate dtype drop)."""
    rms = torch.sqrt(torch.mean(x.float() ** 2, dim=-1, keepdim=True) + eps)
    return x.float() / rms * gamma.float()

def fused_rmsnorm_pos_qkv_qknorm(
    x: torch.Tensor, gamma: torch.Tensor, pos_emb: torch.Tensor,
    Wqkv: torch.Tensor, gamma_q: torch.Tensor, gamma_k: torch.Tensor,
    num_heads: int = 4, eps: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    original_dtype = x.dtype
    B, S, D = x.shape
    H = Wqkv.shape[1] // 3

    h = rmsnorm(x, gamma, eps) + pos_emb.float()
    qkv = torch.matmul(h, Wqkv.float())
    q, k, v = qkv.split(H, dim=-1)
    q = rmsnorm(q, gamma_q, eps)
    k = rmsnorm(k, gamma_k, eps)
    return q.to(original_dtype), k.to(original_dtype), v.to(original_dtype)
```
