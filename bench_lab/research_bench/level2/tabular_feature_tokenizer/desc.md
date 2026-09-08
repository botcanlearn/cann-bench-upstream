# TabularFeatureTokenizer 算子 API 描述

## 1. 算子简介

`TabularFeatureTokenizer` 面向 TabFM / tabular foundation model 输入层，将数值特征和类别特征统一转换为 token 序列。

**算子特征**：

- 难度等级：L2（FusedComposite）
- 数值特征通过 feature-wise 权重投影为 token
- 类别特征通过共享 embedding table gather
- 输出 `[B, F_num + F_cat, D]`

## 2. 算子定义

```text
num_tokens[b, i, :] = x_num[b, i] * num_weight[i, :] + bias[i, :]
cat_tokens[b, j, :] = cat_table[x_cat[b, j]]
tokens = concat(num_tokens, cat_tokens, dim=1)
```

`bias` 为 None 时跳过。

## 3. 接口规范

```python
cann_bench.tabular_feature_tokenizer(
    Tensor x_num,
    Tensor x_cat,
    Tensor num_weight,
    Tensor cat_table,
    Tensor? bias=None,
    int unknown_id=0,
) -> Tensor tokens
```

### 输入参数

| 参数 | Shape | dtype | 描述 |
|---|---|---|---|
| `x_num` | `[B, F_num]` | float16 / bfloat16 / float32 | 数值特征 |
| `x_cat` | `[B, F_cat]` | int32 / int64 | 类别 id |
| `num_weight` | `[F_num, D]` | 与 `x_num` 相同 | 数值特征投影权重 |
| `cat_table` | `[Cardinality, D]` | 与 `x_num` 相同 | 类别 embedding table |
| `bias` | `[F_num, D]` 或 None | 与 `x_num` 相同 | 数值特征 bias |
| `unknown_id` | 标量 | int | 非法类别 id 的替换值 |

### 输出

| 参数 | Shape | dtype |
|---|---|---|
| `tokens` | `[B, F_num + F_cat, D]` | 与 `x_num` 相同 |

### 规则与约束

- `x_cat` 中 id 若不在 `[0, Cardinality)`，会被替换为 `unknown_id`。
- `unknown_id` 必须在 `[0, Cardinality)`。
- 首版使用单个共享 `cat_table`，不覆盖每个类别特征独立 table 的形式。

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


def tabular_feature_tokenizer(
    x_num: torch.Tensor,
    x_cat: torch.Tensor,
    num_weight: torch.Tensor,
    cat_table: torch.Tensor,
    bias: torch.Tensor = None,
    unknown_id: int = 0,
) -> torch.Tensor:
    if x_num.dim() != 2 or x_cat.dim() != 2:
        raise ValueError("x_num and x_cat must be 2D")
    if num_weight.dim() != 2 or cat_table.dim() != 2:
        raise ValueError("num_weight and cat_table must be 2D")
    if x_num.shape[1] != num_weight.shape[0]:
        raise ValueError("num_weight first dim must match number of numerical features")
    if not (0 <= unknown_id < cat_table.shape[0]):
        raise ValueError("unknown_id out of cat_table range")

    out_dtype = x_num.dtype
    num_tokens = x_num.float().unsqueeze(-1) * num_weight.float().unsqueeze(0)
    if bias is not None:
        num_tokens = num_tokens + bias.float().unsqueeze(0)

    ids = x_cat.to(torch.long)
    valid = (ids >= 0) & (ids < cat_table.shape[0])
    ids = torch.where(valid, ids, torch.full_like(ids, int(unknown_id)))
    cat_tokens = cat_table.float().index_select(0, ids.reshape(-1))
    cat_tokens = cat_tokens.reshape(*ids.shape, cat_table.shape[1])

    return torch.cat([num_tokens, cat_tokens], dim=1).to(out_dtype)
```
