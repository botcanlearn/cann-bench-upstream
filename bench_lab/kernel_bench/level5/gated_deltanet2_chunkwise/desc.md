# GatedDeltaNet2Chunkwise 算子 API 描述

## 1. 算子简介

Gated DeltaNet-2（GDN-2）线性注意力层的前向计算。GDN-2 把 Gated DeltaNet 的标量写强度 β 解耦为两个通道级门：作用于键通道（Dk 维）的 erase 门 $b_t$ 与作用于值通道（Dv 维）的 write 门 $w_t$，并继承 KDA 的通道级衰减 $\alpha_t = \exp(g_t)$。该层以 delta-rule（快权重误差修正）方式维护一个 $D_k \times D_v$ 的矩阵状态：每个 token 先按通道衰减旧状态，再从状态中"擦除"当前键方向上的旧关联、"写入"新的键值关联，最后用查询读出。

这类层是新一代混合架构 LLM（Qwen3-Next / Qwen3.5、Kimi Linear 谱系）中替代大部分 softmax attention 的核心算子：训练与 prefill 的复杂度从 $O(L^2)$ 降到 $O(L)$，长上下文场景下该层的前向吞吐直接决定整网 prefill 速度，是这一代模型的关键性能瓶颈。

**主要应用场景**：
- 混合线性注意力 LLM（Qwen3-Next / Qwen3.5、Kimi Linear 等 GDN/KDA 谱系）的训练与 prefill 推理
- 长上下文（32k ~ 1M token）场景的线性复杂度序列建模
- 需要携带跨段状态（final_state 作为下一段初始状态由框架侧衔接）的流式/分段推理

**算子特征**：
- 难度等级：L5（FusedComposite）
- 六输入（q, k, v, erase_gate, write_gate, log_decay）双输出（y, final_state）
- 融合 L2 归一化、通道级衰减、delta-rule 擦除/写入与状态读出
- erase_gate / write_gate ∈ (0, 1)、log_decay ≤ 0（由评测 value_range 保证），配合 k̂ 的 L2 归一化使状态更新为收缩映射，任意随机输入下递归数值稳定

**为何是 L5**（约为 L4 FusedComposite 的 2 倍难度）：
- **正确 ≠ 快**：按 §2 的逐 token 递归直译即可得到完全正确的结果并通过全部精度用例，但计算沿 L 完全串行、无法利用 matmul 单元，性能只能锚定朴素 baseline（预期得分 ~0.5 档）；要逼近硬件下界，必须改用 §2 给出的分块并行形式——块内把 delta-rule 反馈整理成一个下三角线性系统、其余计算全部 matmul 化，块间只串行传递 $[H, D_k, D_v]$ 状态（串行长度从 $L$ 降到 $L/\text{chunk\_size}$）——并把块内 matmul 链与块间依赖排成流水。所需数学恒等式已在 §2 完整给出并经数值验证，难点在工程映射而非推导
- **实现规格而非过可见测试**：隐藏评测集覆盖公开用例之外的规格维度（末块残缺、chunk_size=1 与 ≥ L、$D_k \neq D_v$、极端门值/衰减、全素数维度、H=1 等），逐 token 定义中每一处约定（eps 位置、门的作用通道、衰减作用于 Dk 行）都会被单独检验
- **精度约束严**：递归状态与 delta 反馈路径必须以 fp32 保存/累加（实测中间量整体降 bf16 时误差放大三个数量级、必不达标，见 §4），低精度输入下需要全程管理精度与布局转换
- **项多且不规则**：双通道级门、通道级衰减、两处 L2 归一化、状态擦除/写入反馈、两个输出（序列输出 + 末态），任何一项的顺序或广播维度写错都会被隐藏用例捕获

## 2. 算子定义

### 数学定义（逐 token 递归）

每个 (batch, head) 独立，状态 $S_t \in \mathbb{R}^{D_k \times D_v}$，$S_0 = 0$。对 $t = 1, \dots, L$：

$$
\hat{q}_t = \frac{q_t}{\lVert q_t \rVert_2 + 10^{-6}}, \qquad
\hat{k}_t = \frac{k_t}{\lVert k_t \rVert_2 + 10^{-6}}
\quad \text{（沿 } D_k \text{ 维 L2 归一化，eps 加在范数上）}
$$

$$
\bar{S}_t = \mathrm{Diag}(\alpha_t)\, S_{t-1}, \qquad \alpha_t = \exp(g_t) \in (0, 1)
\quad \text{（按 } D_k \text{ 行逐通道衰减）}
$$

$$
e_t = b_t \odot \hat{k}_t, \qquad
r_t = \bar{S}_t^{\mathsf T} e_t \in \mathbb{R}^{D_v}, \qquad
z_t = w_t \odot v_t
$$

$$
S_t = \bar{S}_t + \hat{k}_t\, (z_t - r_t)^{\mathsf T}, \qquad
o_t = S_t^{\mathsf T} \hat{q}_t \in \mathbb{R}^{D_v}
$$

其中 $b_t \in (0,1)^{D_k}$ 为 erase 门、$w_t \in (0,1)^{D_v}$ 为 write 门、$g_t \le 0$ 为通道级对数衰减（三者均为算子输入）。输出 $y = [o_1, \dots, o_L]$，`final_state` $= S_L$。

**本算子的精确约定**：
- L2 归一化的 eps 加在范数上（$x / (\lVert x \rVert_2 + 10^{-6})$），不是加在平方和内
- $\hat{k}$ 的归一化保证 $\hat{k}_t^{\mathsf T} e_t = \hat{k}_t^{\mathsf T} (b_t \odot \hat{k}_t) \le 1$，状态更新为收缩映射
- 衰减 $\mathrm{Diag}(\alpha_t)$ 作用于状态的 $D_k$ 行（键通道），不作用于 $D_v$ 列
- `chunk_size` 仅约束 kernel 的分块实现粒度，**不影响数学结果**（任意 chunk_size 下输出一致）

### 分块并行形式（数学参考）

以下恒等式与上面的逐 token 递归**数学等价**（已数值验证：fp64 下与逐 token 递归的最大偏差 < 1e-15，覆盖 chunk 大小 1 / 8 / 16 / 37 / 64 与末块残缺），供实现分块 kernel 时直接使用。

记 $u_t = z_t - r_t$，则状态更新是线性递归 $S_t = \mathrm{Diag}(\alpha_t) S_{t-1} + \hat{k}_t u_t^{\mathsf T}$，只是系数 $u_t$ 依赖 $S_{t-1}$。把序列切成长度 $C$ 的块（末块按实际长度）。块内用局部下标 $i = 1, \dots, C$，进入态记 $S_{\mathrm{in}}$（首块为 0），并定义块内对数衰减前缀和与累计衰减：

$$
\mathrm{cs}_i = \sum_{m \le i} g_m \in \mathbb{R}^{D_k}, \qquad \Gamma_i = \exp(\mathrm{cs}_i)
$$

**(1) 块内下三角系统**（先解出全部 $u_i$）：把 $S_{i-1}$ 沿块内展开代入 $r_i = S_{i-1}^{\mathsf T} (\alpha_i \odot e_i)$，得

$$
u_i = z_i - S_{\mathrm{in}}^{\mathsf T} (\Gamma_i \odot e_i) - \sum_{j < i} M_{ij}\, u_j, \qquad
M_{ij} = \sum_{n=1}^{D_k} e_i[n]\, \hat{k}_j[n]\, \exp\!\big(\mathrm{cs}_i[n] - \mathrm{cs}_j[n]\big)
$$

$M \in \mathbb{R}^{C \times C}$ 严格下三角。矩阵形式 $(I + M)\, U = Z - (E \odot \Gamma)\, S_{\mathrm{in}}$（$U, Z \in \mathbb{R}^{C \times D_v}$、$E \odot \Gamma \in \mathbb{R}^{C \times D_k}$ 为按行堆叠），可用前向替换或下三角求逆求解。

**(2) 块内输出**：

$$
o_i = S_{\mathrm{in}}^{\mathsf T} (\Gamma_i \odot \hat{q}_i) + \sum_{j \le i} G_{ij}\, u_j, \qquad
G_{ij} = \sum_{n=1}^{D_k} \hat{q}_i[n]\, \hat{k}_j[n]\, \exp\!\big(\mathrm{cs}_i[n] - \mathrm{cs}_j[n]\big)
$$

$G \in \mathbb{R}^{C \times C}$ 为含对角的下三角。

**(3) 块间状态传递**：

$$
S_{\mathrm{out}} = \mathrm{Diag}(\Gamma_C)\, S_{\mathrm{in}} + \sum_{j=1}^{C} \mathrm{Diag}\!\big(\Gamma_C / \Gamma_j\big)\, \hat{k}_j\, u_j^{\mathsf T},
\qquad \Gamma_C / \Gamma_j = \exp(\mathrm{cs}_C - \mathrm{cs}_j)
$$

**数值安全性**：上式中所有跨步衰减都以比值 $\exp(\mathrm{cs}_i - \mathrm{cs}_j)$（$i \ge j$）出现，每个元素 $\le 1$，不含 $\Gamma^{-1}$ 一类可能上溢的量。

## 3. 接口规范

### 算子原型

```python
gated_deltanet2_chunkwise(Tensor q, Tensor k, Tensor v, Tensor erase_gate, Tensor write_gate, Tensor log_decay, int chunk_size=64) -> (Tensor y, Tensor final_state)
```

### 输入参数说明

| 参数 | 类型 | 必需 | dtype | shape | 描述 |
|------|------|------|-------|-------|------|
| q | Tensor | 是 | float32/float16/bfloat16 | [B, L, H, Dk] | 查询张量，算子内部沿 Dk 做 L2 归一化 |
| k | Tensor | 是 | float32/float16/bfloat16 | [B, L, H, Dk] | 键张量，算子内部沿 Dk 做 L2 归一化 |
| v | Tensor | 是 | float32/float16/bfloat16 | [B, L, H, Dv] | 值张量 |
| erase_gate | Tensor | 是 | float32/float16/bfloat16 | [B, L, H, Dk] | 通道级 erase 门 b_t ∈ (0, 1)（评测取值范围 [0.05, 0.95]） |
| write_gate | Tensor | 是 | float32/float16/bfloat16 | [B, L, H, Dv] | 通道级 write 门 w_t ∈ (0, 1)（评测取值范围 [0.05, 0.95]） |
| log_decay | Tensor | 是 | float32/float16/bfloat16 | [B, L, H, Dk] | 通道级对数衰减 g_t ≤ 0（评测取值范围 [-0.5, -0.001]） |
| chunk_size | int | 否 | - | 标量 | 分块大小，默认 64；仅约束 kernel 分块实现，不影响数学结果 |

### 输出

| 名称 | dtype | shape | 描述 |
|------|-------|-------|------|
| y | 与 q 一致 | [B, L, H, Dv] | 输出序列 |
| final_state | 与 q 一致 | [B, H, Dk, Dv] | 序列末尾的递归状态 S_L |

### 数据类型

| q/k/v/erase_gate/write_gate/log_decay dtype | 输出 dtype | 内部计算 |
|---------------------------------------------|-----------|----------|
| bfloat16 | bfloat16 | fp32（状态与 delta 反馈路径必须 fp32 保存/累加） |
| float16 | float16 | fp32 |
| float32 | float32 | fp32 |

### 规则与约束

- 六个 Tensor 输入 dtype 必须一致
- 维度一致性：q/k/erase_gate/log_decay 的 shape 完全一致（[B, L, H, Dk]）；v/write_gate 的 shape 完全一致（[B, L, H, Dv]）；两组共享 B, L, H
- Dk 与 Dv 允许不同（评测覆盖 Dk=Dv、Dk<Dv、Dk>Dv）
- erase_gate/write_gate ∈ (0, 1)、log_decay ≤ 0 由 cases 的 value_range 保证；算子不对违反此约定的输入负责
- `chunk_size` 为正整数，无需整除 L（末块按实际长度处理），chunk_size ≥ L 时退化为单块，chunk_size=1 时退化为逐 token
- 递归状态 S 与 delta 反馈路径（r_t 的计算与 u_t 的回代）以 fp32 保存/累加（低精度输入场景下这是精度达标的必要条件，见 §4）
- 输出须为 contiguous 张量

### 支持范围

| 维度 / 参数 | 范围 | 备注 |
|---|---|---|
| `B`（batch） | 1 ~ 32 | cases.csv 实测 1 ~ 8 |
| `L`（序列长度） | 16 ~ 16384 | cases.csv 实测 512 ~ 8192 |
| `H`（头数） | 1 ~ 128 | cases.csv 实测 16 / 32（Qwen3-Next / Qwen3.5 量级） |
| `Dk`（键/查询头维） | 8 ~ 256 | cases.csv 实测固定 128 |
| `Dv`（值头维） | 8 ~ 256 | cases.csv 实测固定 128（隐藏用例含 Dk ≠ Dv） |
| `chunk_size` | ≥ 1 | cases.csv 实测 32 / 64 / 128，默认 64 |
| dtype | bfloat16 / float16 / float32 | cases.csv 实测三种均覆盖 |
| `q` / `k` / `v` 取值 | [-1, 1] | 常规随机范围 |
| `erase_gate` / `write_gate` 取值 | [0.05, 0.95] | 恒在 (0, 1) 内 |
| `log_decay` 取值 | [-0.5, -0.001] | 恒负，α = exp(g) ∈ (0.61, 0.999) |

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

长递归链（L 最长 8192）误差随序列累积，且分块实现与朴素逐 token 递归的求和顺序不同。阈值经评测框架 checker 实测确定（fp32 数据通路的误差下限：bf16 MERE≈1e-6 / MARE≈5.7e-2，fp16 MERE≈1.3e-7 / MARE≈7.0e-3，fp32 MERE≈1.0e-6）：

| 数据类型 | FLOAT32 | FLOAT16 | BFLOAT16 |
|----------|---------|---------|----------|
| **通过阈值(Threshold)** | 0.001 | 0.005 | 0.02 |

当平均相对误差 MERE < Threshold，最大相对误差 MARE < 10 * Threshold 时判定为通过。输出为 Dk 项内积（存在相消），小值域与相消场景由评测框架的兜底标准处理。**达标前提**：递归状态 S 与 delta 反馈路径以 fp32 保存/累加——实测把状态/反馈中间量整体降为 bf16 时 MERE 达 3e-2、且小值域兜底判定必失败。

## 5. 标准 Golden 代码

```python
import torch
from typing import Tuple


def _gated_deltanet2_chunkwise_core(q, k, v, erase_gate, write_gate, log_decay, compute_dtype):
    """核心计算：以 compute_dtype 精度执行朴素逐 token 递归，返回 (y, final_state)。"""
    Bsz, L, H, Dk = q.shape
    Dv = v.shape[-1]

    q_f = q.to(compute_dtype)
    k_f = k.to(compute_dtype)
    v_f = v.to(compute_dtype)
    b_f = erase_gate.to(compute_dtype)
    w_f = write_gate.to(compute_dtype)
    g_f = log_decay.to(compute_dtype)

    # 沿 Dk 做 L2 归一化（eps 加在范数上）
    q_hat = q_f / (q_f.norm(dim=-1, keepdim=True) + 1e-6)          # [B, L, H, Dk]
    k_hat = k_f / (k_f.norm(dim=-1, keepdim=True) + 1e-6)          # [B, L, H, Dk]
    alpha = torch.exp(g_f)                                          # [B, L, H, Dk] 通道级衰减 ∈ (0, 1]
    e_all = b_f * k_hat                                             # [B, L, H, Dk] erase 向量 e_t = b_t ⊙ k̂_t
    z_all = w_f * v_f                                               # [B, L, H, Dv] write 向量 z_t = w_t ⊙ v_t

    S = torch.zeros(Bsz, H, Dk, Dv, dtype=compute_dtype, device=q.device)
    y = torch.empty(Bsz, L, H, Dv, dtype=compute_dtype, device=q.device)
    for t in range(L):
        # S̄_t = Diag(α_t) S_{t-1}：按 Dk 行逐通道衰减
        S = alpha[:, t].unsqueeze(-1) * S                                          # [B, H, Dk, Dv]
        # r_t = S̄_tᵀ e_t（沿 Dk 内积）
        r = (S * e_all[:, t].unsqueeze(-1)).sum(dim=-2)                            # [B, H, Dv]
        # S_t = S̄_t + k̂_t (z_t − r_t)ᵀ（外积 [Dk] × [Dv]）
        S = S + k_hat[:, t].unsqueeze(-1) * (z_all[:, t] - r).unsqueeze(-2)        # [B, H, Dk, Dv]
        # o_t = S_tᵀ q̂_t（沿 Dk 内积）
        y[:, t] = (S * q_hat[:, t].unsqueeze(-1)).sum(dim=-2)                      # [B, H, Dv]
    return y, S


def gated_deltanet2_chunkwise(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    erase_gate: torch.Tensor,
    write_gate: torch.Tensor,
    log_decay: torch.Tensor,
    chunk_size: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Gated DeltaNet-2 golden reference（朴素逐 token 递归；plain golden = bench：fp32 状态）

    Args:
        q: [B, L, H, Dk] 查询，算子内部沿 Dk 做 L2 归一化
        k: [B, L, H, Dk] 键，算子内部沿 Dk 做 L2 归一化
        v: [B, L, H, Dv] 值
        erase_gate: [B, L, H, Dk] 通道级 erase 门 b_t ∈ (0, 1)（评测取值范围 [0.05, 0.95]）
        write_gate: [B, L, H, Dv] 通道级 write 门 w_t ∈ (0, 1)（评测取值范围 [0.05, 0.95]）
        log_decay: [B, L, H, Dk] 通道级对数衰减 g_t ≤ 0（评测取值范围 [-0.5, -0.001]），α_t = exp(g_t)
        chunk_size: 分块大小，仅约束 kernel 分块实现，Golden 的朴素递归不使用

    Returns:
        y: [B, L, H, Dv] 输出序列，dtype 与 q 一致
        final_state: [B, H, Dk, Dv] 序列末尾的状态 S_L，dtype 与 q 一致
    """
    y, final_state = _gated_deltanet2_chunkwise_core(
        q, k, v, erase_gate, write_gate, log_decay, torch.float32)
    return y.to(q.dtype), final_state.to(q.dtype)


def gated_deltanet2_chunkwise_oracle(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    erase_gate: torch.Tensor,
    write_gate: torch.Tensor,
    log_decay: torch.Tensor,
    chunk_size: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Oracle (g)：dtype-agnostic，计算精度跟随输入（fp64_cpu 下为 fp64 真值），不硬编码 .float()。"""
    return _gated_deltanet2_chunkwise_core(q, k, v, erase_gate, write_gate, log_decay, q.dtype)
```

## 6. 额外信息

### 算子调用示例

```python
import torch

B, L, H, Dk, Dv = 4, 1024, 16, 128, 128

q = torch.randn(B, L, H, Dk, dtype=torch.bfloat16, device="npu")
k = torch.randn(B, L, H, Dk, dtype=torch.bfloat16, device="npu")
v = torch.randn(B, L, H, Dv, dtype=torch.bfloat16, device="npu")
erase_gate = torch.empty(B, L, H, Dk, dtype=torch.bfloat16, device="npu").uniform_(0.05, 0.95)
write_gate = torch.empty(B, L, H, Dv, dtype=torch.bfloat16, device="npu").uniform_(0.05, 0.95)
log_decay = torch.empty(B, L, H, Dk, dtype=torch.bfloat16, device="npu").uniform_(-0.5, -0.001)

y, final_state = gated_deltanet2_chunkwise(q, k, v, erase_gate, write_gate, log_decay, chunk_size=64)
# y.shape: [B, L, H, Dv]，final_state.shape: [B, H, Dk, Dv]
```

### 退化关系

当 $b_t = w_t = \beta \cdot \mathbf{1}$（标量门）时，本算子退化为 KDA 形式的 gated delta rule：
$S_t = (I - \beta \hat{k}_t \hat{k}_t^{\mathsf T})\, \mathrm{Diag}(\alpha_t)\, S_{t-1} + \beta \hat{k}_t v_t^{\mathsf T}$（已数值验证一致）。
再取 $\alpha_t$ 为标量时退化为 Gated DeltaNet（GDN-1）。

### 参考文献

- Hatamizadeh, A., Choi, J., Kautz, J. (2026). "Gated DeltaNet-2". arXiv:2605.22791（本算子来源：通道级 erase/write 双门 + 通道级衰减）
- Yang, S., Kautz, J., Hatamizadeh, A. (2025). "Gated Delta Networks: Improving Mamba2 with Delta Rule". ICLR 2025, arXiv:2412.06464
- Yang, S. et al. (2024). "Parallelizing Linear Transformers with the Delta Rule over Sequence Length". NeurIPS 2024, arXiv:2406.06484（delta-rule 的分块并行形式来源）
