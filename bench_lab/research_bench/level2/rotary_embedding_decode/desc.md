# RotaryEmbeddingDecode 算子 API 描述

## 1. 算子简介

`RotaryEmbeddingDecode` 面向 LLM decode / prefill 场景，对 Q/K 张量按照 `position_ids` 应用 RoPE 旋转位置编码。该任务重点覆盖 partial rotary dim、非连续 position id 以及 interleaved / non-interleaved 两种布局。

**算子特征**：

- 难度等级：L2（FusedComposite）
- 输入为 `q`、`k`、`cos`、`sin`、`position_ids`
- 输出为 `(q_out, k_out)`
- 支持只旋转前 `rotary_dim` 维，剩余 head dim 原样保留

## 2. 算子定义

设 `rotary_dim = R`，`half = R / 2`。

non-interleaved 布局：

```text
x1 = x[..., :half]
x2 = x[..., half:R]
out[..., :half] = x1 * cos - x2 * sin
out[..., half:R] = x2 * cos + x1 * sin
out[..., R:] = x[..., R:]
```

interleaved 布局：

```text
even = x[..., :R:2]
odd  = x[..., 1:R:2]
out_even = even * cos - odd * sin
out_odd  = odd  * cos + even * sin
```

`cos` / `sin` 通过 `position_ids` 按 position 维 gather 后广播到 head 维。

## 3. 接口规范

```python
cann_bench.rotary_embedding_decode(
    Tensor q,
    Tensor k,
    Tensor cos,
    Tensor sin,
    Tensor position_ids,
    int rotary_dim=-1,
    bool interleaved=False,
) -> (Tensor q_out, Tensor k_out)
```

### 输入参数

| 参数 | Shape | dtype | 描述 |
|---|---|---|---|
| `q` | `[B, S, Hq, D]` | float16 / bfloat16 / float32 | Query 张量 |
| `k` | `[B, S, Hk, D]` | float16 / bfloat16 / float32 | Key 张量 |
| `cos` | `[MaxPos, R/2]` | float16 / bfloat16 / float32 | cos 表 |
| `sin` | `[MaxPos, R/2]` | float16 / bfloat16 / float32 | sin 表 |
| `position_ids` | `[B, S]` | int32 / int64 | 每个 token 的位置 |
| `rotary_dim` | 标量 | int | 旋转维度；`-1` 表示 `D` |
| `interleaved` | 标量 | bool | 是否使用偶/奇交错布局 |

### 输出

| 参数 | Shape | dtype | 描述 |
|---|---|---|---|
| `q_out` | 与 `q` 相同 | 与 `q` 相同 | 应用 RoPE 后的 Query |
| `k_out` | 与 `k` 相同 | 与 `k` 相同 | 应用 RoPE 后的 Key |

### 规则与约束

- `q` 和 `k` 必须为 4D，且最后一维 `D` 一致。
- `rotary_dim=-1` 时取 `D`；否则必须满足 `0 < rotary_dim <= D` 且为偶数。
- `cos.shape == sin.shape == [MaxPos, rotary_dim / 2]`。
- `position_ids` 必须满足 `0 <= position_ids < MaxPos`。
- `position_ids.shape == [B, S]`，其中 `B/S` 与 `q`、`k` 前两维一致。
- 首版不覆盖动态生成 cos/sin 表，只评测已有 cos/sin 表 gather + rotate。

### 数据类型

| q/k dtype | cos/sin dtype | 输出 dtype |
|---|---|---|
| float16 | float16 / float32 | float16 |
| bfloat16 | bfloat16 / float32 | bfloat16 |
| float32 | float32 | float32 |

## 4. 精度要求

采用[生态算子精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md)进行验证。

**误差指标**：

1. 平均相对误差（MERE）：采样点中相对误差平均值

   $$
   \text{MERE} = \text{avg}(\frac{\text{abs}(actual - golden)}{\text{abs}(golden)+\text{1e-7}})
   $$

2. 最大相对误差（MARE）：采样点中相对误差最大值

   $$
   \text{MARE} = \max(\frac{\text{abs}(actual - golden)}{\text{abs}(golden)+\text{1e-7}})
   $$

**通过标准**：

| 数据类型 | FLOAT16 | BFLOAT16 | FLOAT32 | HiFLOAT32 | FLOAT8 E4M3 | FLOAT8 E5M2 |
|----------|---------|----------|---------|-----------|-------------|-------------|
| **通过阈值(Threshold)** | 2^-10 | 2^-7 | 2^-13 | 2^-11 | 2^-3 | 2^-2 |

当平均相对误差 MERE < Threshold，最大相对误差 MARE < 10 * Threshold 时判定为通过。

## 5. 标准 Golden 代码

```python
#!/usr/bin/python3
# coding=utf-8

import torch


def _rotate(
    x: torch.Tensor,
    cos_pos: torch.Tensor,
    sin_pos: torch.Tensor,
    rotary_dim: int,
    interleaved: bool,
) -> torch.Tensor:
    out_dtype = x.dtype
    x_f = x.float()
    cos_f = cos_pos.float()
    sin_f = sin_pos.float()

    while cos_f.dim() < x_f.dim():
        cos_f = cos_f.unsqueeze(-2)
        sin_f = sin_f.unsqueeze(-2)

    x_rot = x_f[..., :rotary_dim]
    x_pass = x_f[..., rotary_dim:]
    half = rotary_dim // 2

    if interleaved:
        x_even = x_rot[..., 0::2]
        x_odd = x_rot[..., 1::2]
        y_even = x_even * cos_f - x_odd * sin_f
        y_odd = x_odd * cos_f + x_even * sin_f
        y_rot = torch.empty_like(x_rot)
        y_rot[..., 0::2] = y_even
        y_rot[..., 1::2] = y_odd
    else:
        x1 = x_rot[..., :half]
        x2 = x_rot[..., half:]
        y1 = x1 * cos_f - x2 * sin_f
        y2 = x2 * cos_f + x1 * sin_f
        y_rot = torch.cat([y1, y2], dim=-1)

    y = torch.cat([y_rot, x_pass], dim=-1)
    return y.to(out_dtype)


def rotary_embedding_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
    rotary_dim: int = -1,
    interleaved: bool = False,
):
    """Apply rotary embedding to q/k according to position_ids."""
    if q.dim() != 4 or k.dim() != 4:
        raise ValueError("q and k must be 4D [B,S,H,D]")
    if q.shape[0] != k.shape[0] or q.shape[1] != k.shape[1] or q.shape[-1] != k.shape[-1]:
        raise ValueError("q and k must have same B, S and D")
    if position_ids.shape != q.shape[:2]:
        raise ValueError(f"position_ids shape must be {q.shape[:2]}, got {tuple(position_ids.shape)}")

    head_dim = q.shape[-1]
    if rotary_dim == -1:
        rotary_dim = head_dim
    if rotary_dim <= 0 or rotary_dim > head_dim or rotary_dim % 2 != 0:
        raise ValueError(f"rotary_dim must be positive, even and <= head_dim, got {rotary_dim}")
    if cos.shape != sin.shape or cos.shape[-1] != rotary_dim // 2:
        raise ValueError("cos/sin must have shape [MaxPos, rotary_dim/2]")

    pos = position_ids.to(torch.long)
    if pos.numel() and (pos.min() < 0 or pos.max() >= cos.shape[0]):
        raise ValueError("position_ids out of cos/sin table range")

    cos_pos = cos.index_select(0, pos.reshape(-1)).reshape(*pos.shape, rotary_dim // 2)
    sin_pos = sin.index_select(0, pos.reshape(-1)).reshape(*pos.shape, rotary_dim // 2)

    q_out = _rotate(q, cos_pos, sin_pos, rotary_dim, interleaved)
    k_out = _rotate(k, cos_pos, sin_pos, rotary_dim, interleaved)
    return q_out, k_out
```
