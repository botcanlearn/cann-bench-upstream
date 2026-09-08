# MultiQueryAttentionDecode 算子 API 描述

## 1. 算子简介

`MultiQueryAttentionDecode` decode 阶段 Multi-Query Attention，单路 K/V 共享到多路 query heads，用于补充热门大模型 decode 路径中的 attention 变体覆盖。

**算子特征**：

- 难度等级：L4（FusedComposite）
- q: [B,Sq,Hq,D], k/v: [B,Skv,D]
- 输出为 attention 上下文张量
- 参考实现使用 float32 中间计算，再 cast 回输入 dtype

## 2. 算子定义

```text
output = softmax(q @ k^T / sqrt(D) + causal_mask) @ v, where k/v use one shared KV head
```

## 3. 接口规范

```python
cann_bench.multi_query_attention_decode(Tensor q, Tensor k, Tensor v, bool causal=True) -> Tensor output
```

### 输入参数

| 参数 | Shape | dtype | 描述 |
|---|---|---|---|
| `q` | `[B, Sq, Hq, D]` | float16 / bfloat16 / float32 | query 张量 |
| `k` | `[B, Skv, D]` | float16 / bfloat16 / float32 | 共享 key cache 张量 |
| `v` | `[B, Skv, D]` | float16 / bfloat16 / float32 | 共享 value cache 张量 |
| `causal` | 标量 | bool | 是否应用右对齐 causal mask |

### 输出

| 参数 | Shape | dtype | 描述 |
|---|---|---|---|
| `output` | `[B, Sq, Hq, D]` | 与 `q` 相同 | attention 输出 |

### 数据类型

| 输入 dtype | 输出 dtype |
|---|---|
| float16 | float16 |
| bfloat16 | bfloat16 |
| float32 | float32 |

### 规则与约束

- 输入 batch、head 和 head_dim 维度需满足 `proto.yaml` 中 shape 约束。
- `Sq` 和 `Skv` 均需大于 0。
- attention score 使用最后一维 `D` 的 `D ** -0.5` 缩放。
- 输出 shape 与 query 对齐。

## 4. 精度要求

采用[生态算子精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md)进行验证。

**误差指标**：MERE / MARE。

**通过标准**：当平均相对误差 MERE < Threshold，最大相对误差 MARE < 10 * Threshold 时判定为通过。

## 5. 标准 Golden 代码

```python
#!/usr/bin/python3
# coding=utf-8

import torch


def _causal_mask(sq: int, skv: int, device) -> torch.Tensor:
    q_pos = torch.arange(sq, device=device).unsqueeze(1)
    k_pos = torch.arange(skv, device=device).unsqueeze(0)
    return k_pos > (q_pos + skv - sq)


def multi_query_attention_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = True,
) -> torch.Tensor:
    if q.dim() != 4 or k.dim() != 3 or v.dim() != 3:
        raise ValueError("q must be [B,Sq,Hq,D], k/v must be [B,Skv,D]")
    if k.shape != v.shape:
        raise ValueError("k and v must have the same shape")
    batch, seqlen_q, _, headdim = q.shape
    if k.shape[0] != batch or k.shape[2] != headdim:
        raise ValueError("k/v batch and head dimension must match q")

    out_dtype = q.dtype
    q_f = q.float().transpose(1, 2)
    k_f = k.float().unsqueeze(1)
    v_f = v.float().unsqueeze(1)
    scores = torch.matmul(q_f, k_f.transpose(-2, -1)) * (headdim ** -0.5)
    if causal:
        mask = _causal_mask(seqlen_q, k.shape[1], scores.device)
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn, v_f).transpose(1, 2).contiguous()
    return out.to(out_dtype)
```
