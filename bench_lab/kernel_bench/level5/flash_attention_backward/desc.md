# FlashAttentionBackward 算子 API 描述

## 1. 算子简介

FlashAttention 反向传播算子。给定前向输入 Q/K/V 与上游梯度 dO（即 `dy`），按 FlashAttention-2 的反向公式计算 Q/K/V 三者的梯度 dQ/dK/dV。遵循 FlashAttention 的 recompute 设计：前向的注意力概率矩阵 P、输出 O 与 logsumexp（LSE）均为派生量，**不作为算子输入**，由算子内部重算——这一接口设计保证评测框架按 value_range 独立随机生成的任意输入都是合法自洽的（不存在"输入之间必须满足某种数学关系"的隐式契约）。

**主要应用场景**：
- LLM 预训练 / 全参微调 / LoRA-SFT 中 attention 层的反向传播
- 长序列训练（4k~8k 及以上）中以重算换显存的 memory-efficient backward
- 因果（decoder）与双向（encoder）两类 attention 架构的梯度计算

**算子特征**：
- 难度等级：L5（FusedComposite）
- 四输入（query, key, value, dy）三输出（dq, dk, dv），BNSD 布局
- 融合前向重算（QK^T、softmax、PV）与反向 5 次矩阵乘、Delta 修正项计算

**为何是 L5**（约为 L4 FusedComposite 的 2 倍难度）：
- **前向重算 + LSE 管理**：kernel 需在反向过程中重算 softmax。高效实现需先做一遍前向分块扫描保存每行 logsumexp（LSE）与 Delta = rowsum(dO ⊙ O)，反向阶段用 `P = exp(S - LSE)` 免归一化地重建概率矩阵——涉及两遍扫描的数据流设计，而 L4 融合算子均为单遍前向数据流
- **5 次链式矩阵乘**：S = QK^T、dV = P^T dO、dP = dO V^T、dQ = dS·K、dK = dS^T·Q，远多于 L4 attention 类算子的 2 次，CUBE 流水编排与 L1/UB 驻留策略复杂度显著上升
- **causal 分块跳过**：因果掩码下按块跳过全屏蔽区域、对角块内做三角掩码，反向的 dS 同样需要掩码一致性
- **跨块累加**：外层循环沿 KV 分块时 dQ 被多个 KV 块累加（沿 Q 分块时则是 dK/dV 被累加），需要原子加或二遍扫描（split + reduce）方案解决写冲突——这是前向 FlashAttention 完全没有的难点

## 2. 算子定义

### 数学公式

设 $S, P, O$ 为前向中间量（内部重算，非输入）：

**前向重算**：

$$
S = Q K^T \cdot \text{scaleValue} \;(+\, \text{causal mask}), \quad
P = \text{softmax}(S), \quad
O = P V
$$

**反向传播**（FA2 手推公式，对 $O = \text{softmax}(QK^T \cdot s)\,V$ 求导）：

$$
dV = P^T \, dO
$$

$$
dP = dO \, V^T, \quad
\Delta_i = \text{rowsum}(dO \odot O)
$$

$$
dS = P \odot (dP - \Delta), \quad
dQ = dS \cdot K \cdot \text{scaleValue}, \quad
dK = dS^T \cdot Q \cdot \text{scaleValue}
$$

其中 $\Delta_i$ 是 softmax 归一化项的梯度修正（softmax 的 Jacobian 为 $\text{diag}(p) - p p^T$，$\Delta_i = p_i^T dp_i$）。

### 计算子步骤

1. **前向重算 scores**：$S = Q K^T \cdot \text{scaleValue}$，shape `[B, N, S, S]`
2. **因果掩码（可选）**：`is_causal=True` 时屏蔽 $j > i$ 的位置（下三角可见，对角线上每行至少自身可见，softmax 恒有效）
3. **softmax 重算**：$P = \text{softmax}(S)$（高效实现保存前向 LSE，用 $P = \exp(S - \text{LSE})$ 重建）
4. **前向输出重算**：$O = P V$（仅用于计算 $\Delta$；高效实现可在前向预处理阶段计算 $\Delta$ 后丢弃 O）
5. **dV**：$dV = P^T dO$
6. **dP 与 Delta**：$dP = dO\, V^T$，$\Delta = \text{rowsum}(dO \odot O)$
7. **dS**：$dS = P \odot (dP - \Delta)$（causal 时 $dS$ 在掩码位置自然为 0，因 $P$ 为 0）
8. **dQ / dK**：$dQ = dS \cdot K \cdot \text{scaleValue}$，$dK = dS^T \cdot Q \cdot \text{scaleValue}$

### 与前向 FlashAttention 的关系

| 项目 | FlashAttention（前向） | FlashAttentionBackward |
|------|----------------------|------------------------|
| 矩阵乘次数 | 2（QK^T、PV） | 5（QK^T、P^T dO、dO V^T、dS·K、dS^T·Q） |
| softmax | 在线 softmax（增量归一化） | 由 LSE 重建（免归一化） |
| 扫描遍数 | 单遍 | 两遍（预处理算 Delta/LSE + 反向主循环） |
| 跨块写冲突 | 无 | dQ（或 dK/dV）跨块累加，需原子加或二遍扫描 |

## 3. 接口规范

### 算子原型

```python
flash_attention_backward(Tensor query, Tensor key, Tensor value, Tensor dy, float scaleValue, bool is_causal=False) -> (Tensor dq, Tensor dk, Tensor dv)
```

### 输入参数说明

| 参数 | 类型 | 必需 | dtype | shape | 描述 |
|------|------|------|-------|-------|------|
| query | Tensor | 是 | float32/float16/bfloat16 | [B, N, S, D] | 前向查询张量 Q |
| key | Tensor | 是 | float32/float16/bfloat16 | [B, N, S, D] | 前向键张量 K |
| value | Tensor | 是 | float32/float16/bfloat16 | [B, N, S, D] | 前向值张量 V |
| dy | Tensor | 是 | float32/float16/bfloat16 | [B, N, S, D] | 上游梯度 dO（对前向输出 O 的梯度） |
| scaleValue | float | 是 | - | 标量 | 缩放因子，通常为 1/sqrt(D) |
| is_causal | bool | 否 | - | - | 是否启用因果掩码（下三角，j ≤ i 可见），默认 False |

### 输出

| 名称 | dtype | shape | 描述 |
|------|-------|-------|------|
| dq | 与 query 一致 | [B, N, S, D] | query 的梯度 |
| dk | 与 query 一致 | [B, N, S, D] | key 的梯度 |
| dv | 与 query 一致 | [B, N, S, D] | value 的梯度 |

### 数据类型

| query/key/value/dy dtype | 输出 dtype | 内部计算 |
|--------------------------|-----------|----------|
| bfloat16 | bfloat16 | fp32（矩阵乘 fp32 累加，softmax/Delta fp32） |
| float16 | float16 | fp32 |
| float32 | float32 | fp32 |

### 规则与约束

- query、key、value、dy 四者 dtype 与 shape 必须完全一致（本算子为训练场景 MHA 自注意力反向，S_q == S_kv，不含 GQA 分组）
- scaleValue 通常设置为 $1/\sqrt{D}$，必须为正浮点数
- `is_causal=True` 时掩码为标准下三角（对角线对齐）：位置 (i, j) 在 j > i 时屏蔽；每行至少 j = i 可见，不存在全屏蔽行
- 前向的 P / O / LSE 均为派生量，一律不作为输入，由算子（和 Golden）内部重算；任意随机 Q/K/V/dy 输入均合法

### 支持范围

| 维度 / 参数 | 范围 | 备注 |
|---|---|---|
| `B`（batch） | 1 ~ 64 | cases.csv 实测 1 ~ 8 |
| `N`（注意力头数） | 1 ~ 64 | cases.csv 实测 8 ~ 32 |
| `S`（序列长度） | 128 ~ 8192 | cases.csv 实测 1024 ~ 8192，训练典型 2k / 4k / 8k |
| `D`（head dim） | {64, 128} | cases.csv 实测 64 / 128 |
| `scaleValue` | 任意正浮点 | cases.csv 实测 0.08838（D=128）/ 0.125（D=64），约 `1/sqrt(D)` |
| `is_causal` | {True, False} | cases.csv 实测两种均覆盖 |
| dtype | float32 / float16 / bfloat16 | cases.csv 实测三种均覆盖 |
| 输入数值范围 | [-1, 1] 典型 | cases.csv 实测 [-1, 1]（19 case）和 [0, 0]（zero-input 1 case） |

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

**通过标准**（本算子放宽阈值）：

反向链路包含 5 次链式矩阵乘、softmax 重算和 $(dP - \Delta)$ 相消项，低精度输入下误差沿链路逐级放大，相对生态标准适度放宽：

| 数据类型 | FLOAT32 | FLOAT16 | BFLOAT16 |
|----------|---------|---------|----------|
| **通过阈值(Threshold)** | 0.001 | 0.01 | 0.01 |

当平均相对误差 MERE < Threshold，最大相对误差 MARE < 10 * Threshold 时判定为通过。梯度输出存在零穿越（元素级相消），小值域与相消场景由评测框架的兜底标准处理。

## 5. 标准 Golden 代码

```python
import torch
from typing import Tuple


def _flash_attention_backward_core(query, key, value, dy, scaleValue, is_causal, compute_dtype):
    """核心计算：以 compute_dtype 精度逐 (b, n) 重算前向并求 FA2 反向。"""
    B, N, S, D = query.shape
    q = query.reshape(B * N, S, D).to(compute_dtype)
    k = key.reshape(B * N, S, D).to(compute_dtype)
    v = value.reshape(B * N, S, D).to(compute_dtype)
    do = dy.reshape(B * N, S, D).to(compute_dtype)

    causal_mask = None
    if is_causal:
        # 下三角可见：屏蔽 j > i 的位置（对角线上每行至少 j=i 可见，softmax 恒有效）
        causal_mask = torch.triu(
            torch.ones(S, S, dtype=torch.bool, device=query.device), diagonal=1)

    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)

    # 按 (b, n) 逐头计算，控制 [S, S] 中间矩阵的峰值内存
    for i in range(B * N):
        qi, ki, vi, doi = q[i], k[i], v[i], do[i]
        # === 前向重算 ===
        scores = torch.matmul(qi, ki.transpose(-2, -1)) * scaleValue   # [S, S]
        if causal_mask is not None:
            scores = scores.masked_fill(causal_mask, float('-inf'))
        p = torch.softmax(scores, dim=-1)                              # [S, S]
        o = torch.matmul(p, vi)                                        # [S, D]
        # === FA2 反向公式 ===
        dv[i] = torch.matmul(p.transpose(-2, -1), doi)                 # dV = P^T @ dO
        dp = torch.matmul(doi, vi.transpose(-2, -1))                   # dP = dO @ V^T
        delta = (doi * o).sum(dim=-1, keepdim=True)                    # Delta_i = rowsum(dO ⊙ O)
        ds = p * (dp - delta)                                          # dS = P ⊙ (dP - Delta)
        dq[i] = torch.matmul(ds, ki) * scaleValue                      # dQ = dS @ K * scale
        dk[i] = torch.matmul(ds.transpose(-2, -1), qi) * scaleValue    # dK = dS^T @ Q * scale

    return dq.reshape(B, N, S, D), dk.reshape(B, N, S, D), dv.reshape(B, N, S, D)


def flash_attention_backward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    dy: torch.Tensor,
    scaleValue: float,
    is_causal: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    FlashAttention 反向传播 golden reference（plain golden = bench：内部 fp32 计算）

    Args:
        query: [B, N, S, D] 前向查询张量 Q
        key: [B, N, S, D] 前向键张量 K
        value: [B, N, S, D] 前向值张量 V
        dy: [B, N, S, D] 上游梯度 dO（对前向输出 O 的梯度）
        scaleValue: 缩放因子，通常为 1/sqrt(D)
        is_causal: 是否启用因果掩码（下三角，j <= i 可见），默认 False

    Returns:
        dq [B, N, S, D], dk [B, N, S, D], dv [B, N, S, D] — dtype 与输入一致
    """
    original_dtype = query.dtype
    dq, dk, dv = _flash_attention_backward_core(query, key, value, dy, scaleValue, is_causal, torch.float32)
    return dq.to(original_dtype), dk.to(original_dtype), dv.to(original_dtype)


def flash_attention_backward_oracle(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    dy: torch.Tensor,
    scaleValue: float,
    is_causal: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Oracle (g)：dtype-agnostic，计算精度跟随输入（fp64_cpu 下为 fp64 真值），不硬编码 .float()。"""
    return _flash_attention_backward_core(query, key, value, dy, scaleValue, is_causal, query.dtype)
```

## 6. 额外信息

### 算子调用示例

```python
import torch

B, N, S, D = 4, 32, 2048, 128

query = torch.randn(B, N, S, D, dtype=torch.bfloat16, device="npu")
key = torch.randn(B, N, S, D, dtype=torch.bfloat16, device="npu")
value = torch.randn(B, N, S, D, dtype=torch.bfloat16, device="npu")
dy = torch.randn(B, N, S, D, dtype=torch.bfloat16, device="npu")

dq, dk, dv = flash_attention_backward(query, key, value, dy,
                                      scaleValue=1.0 / (D ** 0.5),
                                      is_causal=True)
# dq.shape == dk.shape == dv.shape == [B, N, S, D]
```

### 与 torch.autograd 的等价性

上述手推公式与 PyTorch autograd 对 `softmax(Q K^T · scale) @ V` 自动求导数学等价，可用以下方式交叉验证（fp32）：

```python
q = torch.randn(1, 4, 128, 64, requires_grad=True)
k = torch.randn(1, 4, 128, 64, requires_grad=True)
v = torch.randn(1, 4, 128, 64, requires_grad=True)
dy = torch.randn(1, 4, 128, 64)

o = torch.softmax(q @ k.transpose(-2, -1) * 0.125, dim=-1) @ v
o.backward(dy)
# q.grad / k.grad / v.grad 应与 flash_attention_backward(q, k, v, dy, 0.125) 一致
```
