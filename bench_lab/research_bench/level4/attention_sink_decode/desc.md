# AttentionSinkDecode 算子 API 描述

## 1. 算子简介

`AttentionSinkDecode` decode 阶段带 attention sink token 的 scaled dot-product attention，用于补充热门大模型 decode 路径中的 attention 变体覆盖。

**算子特征**：

- 难度等级：L4（FusedComposite）
- q: [B,Sq,H,D], k/v: [B,Skv,H,D], sink_k/sink_v: [B,Sink,H,D]
- 输出为 attention 上下文张量
- 参考实现使用 float32 中间计算，再 cast 回输入 dtype

## 2. 算子定义

```text
output = softmax(q @ concat(sink_k, k)^T / sqrt(D) + mask) @ concat(sink_v, v)
```

## 3. 接口规范

```python
cann_bench.attention_sink_decode(Tensor q, Tensor k, Tensor v, Tensor sink_k, Tensor sink_v, bool causal=True) -> Tensor output
```

### 输入参数

| 参数 | Shape | dtype | 描述 |
|---|---|---|---|
| `q` | `[B, Sq, H, D]` | float16 / bfloat16 / float32 | query 张量 |
| `k` | `[B, Skv, H, D]` | float16 / bfloat16 / float32 | 普通 key cache 张量 |
| `v` | `[B, Skv, H, D]` | float16 / bfloat16 / float32 | 普通 value cache 张量 |
| `sink_k` | `[B, Sink, H, D]` | float16 / bfloat16 / float32 | attention sink key 张量 |
| `sink_v` | `[B, Sink, H, D]` | float16 / bfloat16 / float32 | attention sink value 张量 |
| `causal` | 标量 | bool | 普通 KV 部分是否应用右对齐 causal mask，sink token 始终可见 |

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


def _sink_causal_mask(sq: int, skv: int, sink: int, device) -> torch.Tensor:
    q_pos = torch.arange(sq, device=device).unsqueeze(1)
    ordinary_pos = torch.arange(skv, device=device).unsqueeze(0)
    ordinary_mask = ordinary_pos > (q_pos + skv - sq)
    sink_mask = torch.zeros((sq, sink), dtype=torch.bool, device=device)
    return torch.cat((sink_mask, ordinary_mask), dim=1)


def attention_sink_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sink_k: torch.Tensor,
    sink_v: torch.Tensor,
    causal: bool = True,
) -> torch.Tensor:
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4 or sink_k.dim() != 4 or sink_v.dim() != 4:
        raise ValueError("all inputs must be 4D tensors")
    if k.shape != v.shape or sink_k.shape != sink_v.shape:
        raise ValueError("k/v and sink_k/sink_v must have matching shapes")
    batch, seqlen_q, heads, headdim = q.shape
    if k.shape[0] != batch or sink_k.shape[0] != batch or k.shape[2:] != (heads, headdim) or sink_k.shape[2:] != (heads, headdim):
        raise ValueError("batch/head dimensions must match")

    out_dtype = q.dtype
    kv_k = torch.cat((sink_k, k), dim=1)
    kv_v = torch.cat((sink_v, v), dim=1)
    q_f = q.float().transpose(1, 2)
    k_f = kv_k.float().transpose(1, 2)
    v_f = kv_v.float().transpose(1, 2)
    scores = torch.matmul(q_f, k_f.transpose(-2, -1)) * (headdim ** -0.5)
    if causal:
        mask = _sink_causal_mask(seqlen_q, k.shape[1], sink_k.shape[1], scores.device)
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn, v_f).transpose(1, 2).contiguous()
    return out.to(out_dtype)
```
