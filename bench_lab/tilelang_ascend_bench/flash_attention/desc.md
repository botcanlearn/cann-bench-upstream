# FlashAttention 算子 API 描述

## 1. 算子简介

标准 Flash Attention（缩放点积注意力）算子，对已分头的 Q/K/V 执行 `softmax(Q @ K^T * scale) @ V`，是 Transformer 架构的核心组件。

**主要应用场景**：
- Transformer 编码器/解码器中的自注意力与交叉注意力
- 大语言模型 prefill / decode 阶段的注意力计算
- 视觉 Transformer 中的注意力模块

**算子特征**：
- 难度等级：L4（FusedComposite）
- 多输入（query, key, value）单输出
- 输入为已分头的张量 [B, S, N, D]
- 支持可配置缩放因子和因果掩码

## 2. 算子定义

### 数学公式

$$
y = \text{softmax}\left(Q \times K^T \times \text{scaleValue}\right) \times V
$$

其中 scaleValue <= 0 时自动使用 $1/\sqrt{D}$。

## 3. 接口规范

```python
cann_bench.flash_attention(Tensor query, Tensor key, Tensor value, float scaleValue=-1.0, bool is_causal=False) -> Tensor y
```

### 输入参数

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| query | Tensor | 必选 | 查询张量 [B, S, N, D] |
| key | Tensor | 必选 | 键张量 [B, S_kv, N, D] |
| value | Tensor | 必选 | 值张量 [B, S_kv, N, D] |
| scaleValue | float | -1.0 | 缩放因子，<=0 时自动 1/sqrt(D) |
| is_causal | bool | False | 因果掩码 |

### 输出

| 参数 | Shape | dtype |
|------|-------|-------|
| y | [B, S, N, D] | 与输入一致 |

### 数据类型

| 输入 dtype | 输出 dtype |
|-----------|-----------|
| float16 | float16 |
| bfloat16 | bfloat16 |

## 4. 精度要求

采用生态算子精度标准：MERE < Threshold 且 MARE < 10×Threshold。

| 数据类型 | Threshold |
|----------|-----------|
| FLOAT16 | 2^-10 |
| BFLOAT16 | 2^-7 |
