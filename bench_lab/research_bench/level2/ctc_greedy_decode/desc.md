# CtcGreedyDecode 算子 API 描述

## 1. 算子简介

`CtcGreedyDecode` 面向 OCR / ASR 的 CTC 后处理，对时序 logits 做 argmax，去除 blank，并可合并连续重复 token，输出 padded token 序列和有效长度。

**主要应用场景**：
- OCR 文本识别后处理
- CTC ASR greedy decode
- 变长 batch 的 token collapse

**算子特征**：
- 难度等级：L2（Reduction）
- 输入 logits 为 `[B, T, V]`
- 输出 token 为 int32 padded 矩阵
- 输出 lengths 为每个 batch 的有效 token 数

## 2. 算子定义

### 数学公式

```text
ids = argmax(logits, dim=-1)
tokens = remove_blank_and_optional_merge_repeated(ids, input_lengths)
```

对于每个 batch，仅处理 `input_lengths[b]` 范围内的时间步。

## 3. 接口规范

### 算子原型

```python
cann_bench.ctc_greedy_decode(
    Tensor logits,
    Tensor input_lengths,
    int blank_id=0,
    bool merge_repeated=True,
    int pad_id=-1,
) -> (Tensor tokens, Tensor lengths)
```

### 输入参数说明

| 参数 | 类型 | Shape | dtype | 描述 |
|------|------|-------|-------|------|
| `logits` | Tensor | `[B, T, V]` | float16 / bfloat16 / float32 | CTC logits |
| `input_lengths` | Tensor | `[B]` | int32 / int64 | 每个 batch 的有效时间步 |
| `blank_id` | int | - | - | blank token id |
| `merge_repeated` | bool | - | - | 是否合并连续重复 token |
| `pad_id` | int | - | - | 输出 token padding 值 |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| `tokens` | `[B, max_decoded_len]` | int32 | padded decode token |
| `lengths` | `[B]` | int32 | 每个 batch 的有效输出长度 |

### 规则与约束

- `logits` 必须为 3D。
- `input_lengths.shape == [B]`。
- `blank_id` 必须在 `[0, V)`。
- `input_lengths` 会裁剪到 `[0, T]` 范围。
- 若某个 batch 没有有效 token，则对应 `lengths[b]=0`，`tokens[b]` 全部为 `pad_id`。

### 支持范围

| 维度 / 参数 | 范围 | 备注 |
|---|---|---|
| `B` | 1 ~ 16 | 变长 batch |
| `T` | 8 ~ 256 | 时序长度 |
| `V` | 8 ~ 512 | vocab 大小 |
| `blank_id` | 0 或 V 范围内其他值 | 首版公开 case 覆盖 |
| `merge_repeated` | {False, True} | 两者均覆盖 |

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

当平均相对误差 MERE < Threshold，最大相对误差 MARE < 10 * Threshold 时判定为通过。整数输出 `tokens` 和 `lengths` 要求与 golden 完全一致。

## 5. 标准 Golden 代码

```python
#!/usr/bin/python3
# coding=utf-8

import torch


def ctc_greedy_decode(
    logits: torch.Tensor,
    input_lengths: torch.Tensor,
    blank_id: int = 0,
    merge_repeated: bool = True,
    pad_id: int = -1,
):
    if logits.dim() != 3:
        raise ValueError("logits must be [B, T, V]")
    if input_lengths.dim() != 1 or input_lengths.shape[0] != logits.shape[0]:
        raise ValueError("input_lengths must be [B]")
    batch, steps, vocab = logits.shape
    if blank_id < 0 or blank_id >= vocab:
        raise ValueError("blank_id must be in [0, V)")

    pred = torch.argmax(logits.float(), dim=-1).cpu()
    lengths_in = torch.clamp(input_lengths.to(torch.long).cpu(), min=0, max=steps)
    sequences = []
    max_out = 0
    for b in range(batch):
        seq = []
        prev = None
        for token in pred[b, :int(lengths_in[b].item())].tolist():
            if token == blank_id:
                prev = token
                continue
            if merge_repeated and prev == token:
                prev = token
                continue
            seq.append(int(token))
            prev = token
        sequences.append(seq)
        max_out = max(max_out, len(seq))

    max_out = max(max_out, 1)
    tokens = torch.full((batch, max_out), int(pad_id), dtype=torch.int32, device=logits.device)
    lengths = torch.empty((batch,), dtype=torch.int32, device=logits.device)
    for b, seq in enumerate(sequences):
        lengths[b] = len(seq)
        if seq:
            tokens[b, :len(seq)] = torch.tensor(seq, dtype=torch.int32, device=logits.device)
    return tokens, lengths
```
