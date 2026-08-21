# MobaAttention 算子 API 描述

## 1. 算子简介

MoBA（Mixture of Block Attention）块稀疏注意力算子，来源于 Moonshot AI 的长上下文注意力工作（arXiv:2502.13189），把 MoE 的路由思想应用到注意力的 KV 块上：将 KV 序列按 `block_size` 切块，每个 query 以「query 与块内 key 均值的点积」作为门控分数，仅对选中的 `top_k` 个块计算因果注意力。长上下文 LLM 的 prefill 与训练瓶颈在注意力的 O(S²) 计算量；MoBA 把每个 query 的注意力工作集从 S 缩小到约 `top_k · block_size` 个位置，且选哪些块由数据在运行时决定。

**主要应用场景**：
- 长上下文 LLM（百万 token 级）的 prefill 与训练加速，Kimi 系列模型的块稀疏注意力层
- 稠密/稀疏混合架构中的可切换注意力（MoBA 与 dense attention 参数兼容、可互换）
- flash-linear-attention（FLA）生态 2026-04 加入的 FlashMoBA kernel 所对应的算子语义

**算子特征**：
- 难度等级：L5（FusedComposite）
- 三输入（query/key/value，BSND，MHA）+ 三属性（block_size、top_k、scaleValue），双输出（y + block_indices）
- 融合块均值池化、带精确并列规则的门控 top-k 块选择、块稀疏因果 softmax 注意力
- 稀疏结构数据依赖：每个 (b, t, n) 的可见 KV 集合由输入数据在运行时决定
- block_indices 为 int32 结构输出，逐元素精确比对（零容差）

**为何是 L5**：
- **性能墙，而非数学墙**：本文档给出了完整数学定义与所需恒等式，朴素 dense 实现（先算满 [S, S] 分数、再按选择结果置 −∞）数学正确，但计算量与 dense 因果注意力相同，长序列下完全没有体现块稀疏收益，只能锚定 baseline 水平。要显著超越 baseline，必须真正跳过未选中的块——而每个 query 行的 KV 工作集由门控在运行时决定，静态切分不再适用；可选块数 cur+1 随位置增长、top_k 截断带来的负载不均衡，都要求由数据依赖结构决定调度。
- **选择语义零容差**：block_indices 是 int32 结构输出，逐元素精确比对。top-k 的并列规则（分数相同取块索引更小者）、当前块强制入选并计入名额、可选块不足时的 −1 填充，都必须与本规格完全一致；不稳定排序等含糊实现会直接失败。
- **两级数值路径交织**：门控走「均值池化 + 点积 + top-k」，注意力走「缩放点积 + softmax + 加权和」，两者对同一份 key 以不同归约方式消费；当前块内因果、过去块全可见的混合掩码形态使标准 FlashAttention 模板不能照搬（§2 给出可精确合并的分块 softmax 恒等式）。
- **隐藏测试集**：评测含未公开用例（末块残缺、top_k ≥ nb、block_size=1、素数 S、D=96 等边界），实现必须以本规格为准，而非以可见用例为准。

## 2. 算子定义

### 记号与切块

位置 0..S−1 切成 nb = ⌈S / block_size⌉ 个块，块 j 覆盖位置区间 [j·bs, min((j+1)·bs, S))，末块可残缺（长度 S − (nb−1)·bs）。记 block(p) = ⌊p / bs⌋ 为位置 p 所在块，cur = block(t) 为 query 位置 t 的当前块。每个 (b, n) 完全独立（MHA，无 KV 组共享），下式省略 b、n 下标。

### 数学公式

**(1) 块代表键**（块内 key 沿位置取均值，末块按实际长度）：

$$
\bar{k}_j = \frac{1}{|\text{块 } j|} \sum_{p \in \text{块 } j} k_p
$$

**(2) 门控分数**（不乘 scaleValue）：

$$
s_{t,j} = \begin{cases} q_t \cdot \bar{k}_j & j < \text{cur} \\ +\infty & j = \text{cur（当前块必选，计入 top\_k 名额）} \\ -\infty & j > \text{cur（不可选）} \end{cases}
$$

**(3) 块选择**：

$$
B_t = \operatorname{top\_k}\left(s_{t,\cdot}\right)
$$

可选块数 cur+1 < top_k 时全选（|B_t| = cur+1）；分数并列时取块索引更小者。等价的确定性描述：按字典序 (−s_{t,j}, j) 升序排序，取前 min(top_k, cur+1) 个 j。

**(4) 键位置集合与注意力**：

$$
P_t = \{\, p : \operatorname{block}(p) \in B_t \ \wedge\ p \le t \,\}
$$

（当前块内因果，过去块内全可见；p = t 恒属于 P_t，softmax 每行至少一个有效位置）

$$
y_t = \sum_{p \in P_t} \operatorname{softmax}_{p \in P_t}\left(\text{scaleValue} \cdot q_t \cdot k_p\right) \, v_p
$$

**(5) 结构输出**：

$$
\text{block\_indices}[b, t, n, :] = \operatorname{sorted}(B_t) \text{ 升序，不足 top\_k 尾部用 } -1 \text{ 填充（int32）}
$$

### 数学参考：分块 softmax 的合并恒等式

选中块集合上的 softmax 加权和可以按块独立计算局部统计量后**精确**合并（数学恒等，与对 P_t 一次性 softmax 逐位一致）。令 s_p = scaleValue · q_t·k_p，块 j 的局部统计量（p 取块 j 中 ≤ t 的位置）：

$$
m_j = \max_p s_p, \qquad l_j = \sum_p e^{s_p - m_j}, \qquad o_j = \sum_p e^{s_p - m_j} \, v_p
$$

一次性合并：

$$
m = \max_j m_j, \qquad l = \sum_j l_j e^{m_j - m}, \qquad y_t = \frac{\sum_j o_j e^{m_j - m}}{l}
$$

两两在线合并 $(m_1, l_1, o_1) \oplus (m_2, l_2, o_2)$：

$$
m = \max(m_1, m_2), \quad l = l_1 e^{m_1 - m} + l_2 e^{m_2 - m}, \quad o = o_1 e^{m_1 - m} + o_2 e^{m_2 - m}
$$

合并满足交换律与结合律，任意合并顺序结果一致。以上恒等式已数值验证（fp64 下与直接 softmax 最大偏差 < 1e−12）；任何数值等价的计算顺序均可接受。

## 3. 接口规范

### 算子原型

```python
moba_attention(Tensor query, Tensor key, Tensor value, int block_size, int top_k, float scaleValue) -> (Tensor y, Tensor block_indices)
```

### 输入参数说明

| 参数 | 类型 | 必需 | dtype | shape | 描述 |
|------|------|------|-------|-------|------|
| query | Tensor | 是 | bfloat16/float16/float32 | [B, S, N, D] | 查询张量（BSND，各头独立做块选择） |
| key | Tensor | 是 | bfloat16/float16/float32 | [B, S, N, D] | 键张量，同时用于块代表键（块内均值）与注意力 |
| value | Tensor | 是 | bfloat16/float16/float32 | [B, S, N, D] | 值张量 |
| block_size | int | 是 | - | 标量 | KV 块大小（≥ 1），S 无需整除，末块按实际长度处理 |
| top_k | int | 是 | - | 标量 | 每个 query 选择的块数（≥ 1），当前块必选且计入名额 |
| scaleValue | float | 是 | - | 标量 | 注意力缩放因子，通常 1/sqrt(D)；仅作用于注意力分数，门控分数不乘 |

### 输出

| 名称 | dtype | shape | 描述 |
|------|-------|-------|------|
| y | 与 query 一致 | [B, S, N, D] | 块稀疏注意力输出 |
| block_indices | int32 | [B, S, N, top_k] | 选中块索引升序，不足 top_k 用 −1 填充（结构输出，精确比对） |

### 数据类型

| query/key/value dtype | y dtype | block_indices dtype | 内部计算 |
|----------------------|---------|---------------------|----------|
| bfloat16 | bfloat16 | int32 | fp32（门控与注意力归约均需 ≥ fp32 精度） |
| float16 | float16 | int32 | fp32 |
| float32 | float32 | int32 | fp32 |

### 规则与约束

- query/key/value 的 dtype 与 shape 必须一致
- `block_size ≥ 1`，S 无需被 block_size 整除（末块按实际长度参与均值与注意力）；`block_size ≥ S` 时 nb = 1，退化为标准因果注意力
- `top_k ≥ 1`；`top_k ≥ nb` 时全部过去块与当前块均被选中，同样退化为标准因果注意力；`top_k = 1` 时仅当前块（块对角因果）
- 门控分数**不乘** scaleValue，注意力分数乘 scaleValue
- 当前块必选、计入 top_k 名额；j > cur 的块不可选
- 并列规则：门控分数相同的块取索引更小者（确定性选择）
- block_indices 中有效索引升序在前、−1 填充在后
- 门控归约（均值、点积）与注意力归约（softmax、PV 加权和）需以 fp32 或更高精度进行——这是低精度输入下 y 精度达标与 block_indices 精确复现的必要条件（见 §4）

### 支持范围

| 维度 / 参数 | 范围 | 备注 |
|---|---|---|
| `B`（batch） | 1 ~ 8 | cases.csv 实测 1 ~ 4 |
| `S`（序列长度） | 1 ~ 32768 | cases.csv 实测 512 ~ 16384，含素数 / 非对齐 |
| `N`（头数） | 1 ~ 64 | cases.csv 实测 1 ~ 32 |
| `D`（head dim） | {32, 64, 96, 128, 256} | cases.csv 实测以 128 为主（MoBA 论文配置） |
| `block_size` | 1 ~ 2048 | cases.csv 实测 1 ~ 2048，论文默认 512 |
| `top_k` | 1 ~ 16 | 论文默认 3（含当前块） |
| dtype | bfloat16 / float16 / float32 | cases.csv 实测三种均覆盖 |
| `query/key/value` 取值 | [-1, 1] | 常规随机范围；边界场景 [-8, 8] / [-0.001, 0.001] |
| `scaleValue` | 常用 1/sqrt(D) | 边界场景 0.5 / 1.0 等非常规值 |

## 4. 精度要求

采用[生态算子精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md)进行验证。

**输出 y（浮点）**——误差指标：

1. 平均相对误差（MERE）：采样点中相对误差平均值

   $$
   \text{MERE} = \text{avg}(\frac{\text{abs}(actual - golden)}{\text{abs}(golden)+\text{1e-7}})
   $$

2. 最大相对误差（MARE）：采样点中相对误差最大值

   $$
   \text{MARE} = \max(\frac{\text{abs}(actual - golden)}{\text{abs}(golden)+\text{1e-7}})
   $$

**通过标准**（采用生态默认阈值，未放宽）：

| 数据类型 | FLOAT32 | FLOAT16 | BFLOAT16 |
|----------|---------|---------|----------|
| **通过阈值(Threshold)** | 2^-13 | 2^-10 | 2^-7 |

当 MERE < Threshold 且 MARE < 10 * Threshold 时判定为通过。实测依据（仓库 checker，公开 case 1 规模 B1S8192N16D128-bs512-tk3，plain golden 的 fp32 计算路径 vs fp64 真值）：bfloat16 MERE 1.7e-6、MARE 7.4e-2 < 10×2^-7；float16 MERE 1.4e-7、MARE 9.8e-4 < 10×2^-10；float32 MERE 3.1e-7、MARE 1.5e-5 < 10×2^-13——三种 dtype 默认阈值均可达标，未放宽；达标前提是归约以 fp32 或更高精度进行。小值域与相消场景由评测框架兜底标准处理。

**输出 block_indices（int32 结构输出）**——零容差：阈值 0，逐元素完全相等，一处不等即失败。可行性依据：bf16 输入下以 fp32 与 fp64 分别计算门控，在 52 万 query 行（含 S=8192/16384 最大公开规模）上选择零翻转；实现只要以 ≥ fp32 精度计算门控并遵守 §2 的并列规则，即可精确复现。

## 5. 标准 Golden 代码

```python
import torch
from typing import Tuple


def _moba_attention_core(query, key, value, block_size, top_k, scaleValue, compute_dtype):
    """核心计算：以 compute_dtype 精度做块门控 top-k 选择 + 块稀疏注意力。

    逐 (b, n) 循环并对 query 分块，控制 [Q, S] 中间量的峰值内存。
    """
    B, S, N, D = query.shape
    device = query.device
    nb = (S + block_size - 1) // block_size

    q_all = query.permute(0, 2, 1, 3).to(compute_dtype)   # [B, N, S, D]
    k_all = key.permute(0, 2, 1, 3).to(compute_dtype)     # [B, N, S, D]
    v_all = value.permute(0, 2, 1, 3).to(compute_dtype)   # [B, N, S, D]

    pos = torch.arange(S, device=device)                  # [S]
    blk_of_pos = pos // block_size                        # [S] 每个位置所属块
    blk_ids = torch.arange(nb, device=device)             # [nb]
    # 每块实际长度（末块可残缺）: len_j = min((j+1)·bs, S) - j·bs
    blk_len = torch.clamp((blk_ids + 1) * block_size, max=S) - blk_ids * block_size

    y = torch.empty(B, S, N, D, dtype=compute_dtype, device=device)
    block_indices = torch.empty(B, S, N, top_k, dtype=torch.int32, device=device)

    q_chunk = 2048                                        # query 分块大小（仅控内存，不影响结果）
    arange_k = torch.arange(top_k, device=device)         # [top_k]
    k_take = min(top_k, nb)

    for b in range(B):
        for n in range(N):
            q = q_all[b, n]                               # [S, D]
            k = k_all[b, n]                               # [S, D]
            v = v_all[b, n]                               # [S, D]

            # 块代表键 k̄_j = 块内 key 均值（零填充到 nb·bs 后分块求和，除以实际长度）
            k_pad = torch.zeros(nb * block_size, D, dtype=compute_dtype, device=device)
            k_pad[:S] = k
            k_mean = k_pad.reshape(nb, block_size, D).sum(dim=1) / blk_len.unsqueeze(-1)  # [nb, D]

            for qs in range(0, S, q_chunk):
                qe = min(qs + q_chunk, S)
                qt = q[qs:qe]                             # [Q, D]
                t_pos = pos[qs:qe]                        # [Q]
                cur = blk_of_pos[qs:qe]                   # [Q] 当前块索引

                # === 门控分数（不乘 scaleValue）===
                gate = qt @ k_mean.T                      # [Q, nb]
                gate = gate.masked_fill(blk_ids.unsqueeze(0) > cur.unsqueeze(-1), float('-inf'))
                gate = gate.masked_fill(blk_ids.unsqueeze(0) == cur.unsqueeze(-1), float('inf'))

                # === top_k 选择：降序稳定排序（基序为块索引升序 → 并列取小索引）===
                order = torch.sort(gate, dim=-1, descending=True, stable=True).indices  # [Q, nb]
                sel = order[:, :k_take]                   # [Q, k_take]
                if k_take < top_k:                        # top_k > nb: 全选后补哨兵
                    pad = torch.full((qe - qs, top_k - k_take), nb, dtype=sel.dtype, device=device)
                    sel = torch.cat([sel, pad], dim=-1)   # [Q, top_k]
                # 可选块数 = cur + 1（当前块 + 过去块），超出部分置哨兵 nb
                n_valid = torch.clamp(cur + 1, max=top_k)                     # [Q]
                sel = sel.masked_fill(arange_k.unsqueeze(0) >= n_valid.unsqueeze(-1), nb)

                # === block_indices 输出：升序排列，哨兵（排在尾部）替换为 -1 ===
                sel_sorted = sel.sort(dim=-1).values                          # [Q, top_k]
                block_indices[b, qs:qe, n] = torch.where(
                    sel_sorted == nb, torch.full_like(sel_sorted, -1), sel_sorted
                ).to(torch.int32)

                # === 块级选中掩码 → 位置级掩码 ===
                sel_mask = torch.zeros(qe - qs, nb + 1, dtype=torch.bool, device=device)
                sel_mask.scatter_(1, sel.long(), True)
                sel_mask = sel_mask[:, :nb]                                   # [Q, nb] 哨兵列丢弃
                pos_mask = sel_mask[:, blk_of_pos]                            # [Q, S] 位置所属块被选中
                pos_mask &= pos.unsqueeze(0) <= t_pos.unsqueeze(-1)           # 当前块内因果（p ≤ t）

                # === 块稀疏注意力（p = t 恒可见，softmax 每行至少一个有效位置）===
                scores = (qt @ k.T) * scaleValue                              # [Q, S]
                scores.masked_fill_(~pos_mask, float('-inf'))
                y[b, qs:qe, n] = torch.softmax(scores, dim=-1) @ v

    return y, block_indices


def moba_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    block_size: int,
    top_k: int,
    scaleValue: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    MoBA 块稀疏注意力 golden reference（plain golden = bench：内部 fp32 计算）

    Args:
        query: [B, S, N, D] 查询张量（BSND，MHA，N 头共享同一 top-k 语义、各头独立选择）
        key: [B, S, N, D] 键张量
        value: [B, S, N, D] 值张量
        block_size: KV 块大小（正整数，S 无需整除，末块按实际长度处理）
        top_k: 每个 query 选择的块数（≥ 1，当前块必选并占一个名额）
        scaleValue: 注意力缩放因子（仅作用于注意力分数，门控分数不乘）

    Returns:
        y: [B, S, N, D] 注意力输出，dtype 与 query 一致
        block_indices: [B, S, N, top_k] int32，选中块索引升序，不足 top_k 用 -1 填充
    """
    original_dtype = query.dtype
    y, block_indices = _moba_attention_core(
        query, key, value, block_size, top_k, scaleValue, torch.float32)
    return y.to(original_dtype), block_indices


def moba_attention_oracle(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    block_size: int,
    top_k: int,
    scaleValue: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Oracle (g)：dtype-agnostic，计算精度跟随输入（fp64_cpu 下为 fp64 真值），不硬编码 .float()。"""
    return _moba_attention_core(query, key, value, block_size, top_k, scaleValue, query.dtype)
```

## 6. 额外信息

### 算子调用示例

```python
import torch

B, S, N, D = 1, 8192, 16, 128
block_size, top_k = 512, 3
scaleValue = 1.0 / (D ** 0.5)

query = torch.randn(B, S, N, D, dtype=torch.bfloat16, device="npu")
key = torch.randn(B, S, N, D, dtype=torch.bfloat16, device="npu")
value = torch.randn(B, S, N, D, dtype=torch.bfloat16, device="npu")

y, block_indices = moba_attention(query, key, value, block_size, top_k, scaleValue)
# y.shape: [B, S, N, D]；block_indices.shape: [B, S, N, top_k]，int32
```

### 参考文献

- Lu, E. et al. (2025). "MoBA: Mixture of Block Attention for Long-Context LLMs". arXiv:2502.13189（Moonshot AI，算法来源）
- flash-linear-attention（FLA）项目 FlashMoBA kernel（2026-04 加入，本算子对应的开源 GPU 实现）
- Dao, T. et al. (2022). "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness". NeurIPS 2022（§2 分块 softmax 合并恒等式的出处之一）
