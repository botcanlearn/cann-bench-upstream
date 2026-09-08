# SlidingWindowAttentionDecode 算子 API 描述

## 1. 算子简介

`SlidingWindowAttentionDecode` decode 阶段滑动窗口 attention，仅保留最近 window_size 个 KV token，用于补充热门大模型 decode 路径中的 attention 变体覆盖。

**算子特征**：

- 难度等级：L4（FusedComposite）
- q: [B,Sq,H,D], k/v: [B,Skv,H,D]
- 输出为 attention 上下文张量
- 参考实现使用 float32 中间计算，再 cast 回输入 dtype

## 2. 算子定义

```text
output = softmax(q @ k^T / sqrt(D) + sliding_window_causal_mask) @ v
```

## 3. 接口规范

```python
cann_bench.sliding_window_attention_decode(Tensor q, Tensor k, Tensor v, int window_size=128, bool causal=True) -> Tensor output
```

### 输入参数

| 参数 | Shape | dtype | 描述 |
|---|---|---|---|
| `q` | `[B, Sq, H, D]` | float16 / bfloat16 / float32 | query 张量 |
| `k` | `[B, Skv, H, D]` | float16 / bfloat16 / float32 | key cache 张量 |
| `v` | `[B, Skv, H, D]` | float16 / bfloat16 / float32 | value cache 张量 |
| `window_size` | 标量 | int | 每个 query 可见的最近 KV token 数量 |
| `causal` | 标量 | bool | 是否禁止看未来 token |

### 输出

| 参数 | Shape | dtype | 描述 |
|---|---|---|---|
| `output` | `[B, Sq, H, D]` | 与 `q` 相同 | attention 输出 |

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


def _window_mask(sq: int, skv: int, window_size: int, causal: bool, device) -> torch.Tensor:
    q_pos = torch.arange(sq, device=device).unsqueeze(1) + skv - sq
    k_pos = torch.arange(skv, device=device).unsqueeze(0)
    too_old = k_pos <= (q_pos - window_size)
    if causal:
        too_new = k_pos > q_pos
        return too_old | too_new
    return too_old


def sliding_window_attention_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    window_size: int = 128,
    causal: bool = True,
) -> torch.Tensor:
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError("q/k/v must be [B,S,H,D]")
    if k.shape != v.shape:
        raise ValueError("k and v must have the same shape")
    batch, seqlen_q, heads, headdim = q.shape
    if k.shape[0] != batch or k.shape[2:] != (heads, headdim):
        raise ValueError("k/v batch/head dimensions must match q")

    out_dtype = q.dtype
    q_f = q.float().transpose(1, 2)
    k_f = k.float().transpose(1, 2)
    v_f = v.float().transpose(1, 2)
    scores = torch.matmul(q_f, k_f.transpose(-2, -1)) * (headdim ** -0.5)
    mask = _window_mask(seqlen_q, k.shape[1], window_size, causal, scores.device)
    scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn, v_f).transpose(1, 2).contiguous()
    return out.to(out_dtype)
```
