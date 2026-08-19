# Mamba2Ssd 算子 API 描述

## 1. 算子简介

Mamba2 分块状态空间对偶（State Space Duality, SSD）算子，对应 Mamba2 架构核心层的前向计算：以逐 token 的时间步长 dt 对连续状态空间方程离散化，做选择性状态递归（selective state space recurrence），叠加组共享的 B/C 投影（类比 attention 的 K/Q，G 个 KV 组、组内共享）与逐头残差跳连 D。

数学上该算子等价于一个带指数衰减掩码的"线性 attention"（对偶性即 SSD 名称的由来）：直接按 token 维朴素递归即得正确结果，但计算完全串行；高效实现必须采用 SSD 分块（chunkwise）算法——把序列切成 chunk_size 大小的块，块内用 segsum 衰减掩码将计算 matmul 化（走 CUBE），块间只递归传递 [H, P, N] 状态（串行长度从 L 降到 L/chunk_size）。

**主要应用场景**：
- Mamba2 / 混合架构（Jamba、Zamba、Nemotron-H 等）LLM 的训练与 prefill 推理
- 长序列建模（线性复杂度 O(L)，对比 attention 的 O(L²)）
- 语音、基因组等超长序列任务的状态空间骨干网络

**算子特征**：
- 难度等级：L5（FusedComposite）
- 六输入（x, dt, A, B_mat, C_mat, D_skip）单输出（y）
- 融合离散化（exp(dt·A)）、状态外积累加、组广播、状态读出与残差跳连
- dt 恒正（[0.001, 0.1]）、A 恒负（[-8, -0.5]），保证衰减系数 a_t = exp(dt·A) ∈ (0, 1)，任意随机输入下递归数值稳定

**为何是 L5**（约为 L4 FusedComposite 的 2 倍难度）：
- **分块算法重构**：L4 递归类算子（LSTM/GRU）可按时间步直译；本算子朴素递归完全串行、无法打满 CUBE，高效实现必须重写为 SSD 分块形式——块内构造 segsum 下三角衰减掩码 L_mat = exp(segsum(dt·A)) 后以 (C B^T ⊙ L_mat) X 的 matmul 链计算、块间以 h_next = decay·h + 块内状态汇聚 递归传递，是"数学等价但计算结构完全不同"的算法级重构，为 SelectiveScan（L4）的超集（多出组共享 B/C、外积状态 [P, N] 与分块对偶结构）
- **巨大状态张量的驻留**：递归状态 h ∈ [B, H, P, N]（P=64, N=128 时每头 8K 元素 fp32），块间串行依赖下的 UB 驻留与流水编排远难于 LSTM 的 [B, hidden] 向量状态
- **CUBE/VEC 深度交织**：块内三段 matmul（CB^T、掩码加权、状态读出）与 VEC 的 exp/segsum/广播交错，且状态必须 fp32 累加、输入输出 bf16，精度与布局转换贯穿全程
- **数值细节**：segsum 需以差分形式计算避免上溢/下溢，衰减跨度大（dt·A ∈ [-0.8, -5e-4]），块边界处的衰减连乘容易引入误差

## 2. 算子定义

### 数学公式

逐 token 递归（$h_0 = 0$，状态 $h \in \mathbb{R}^{B \times H \times P \times N}$）：

$$
a_t = \exp(dt_t \cdot A), \quad
h_t = a_t \cdot h_{t-1} + dt_t \cdot (B_t \otimes x_t), \quad
y_t = h_t \cdot C_t + D_{skip} \odot x_t
$$

逐元素展开（$b$ 为 batch，$h$ 为头，$g = \lfloor h / (H/G) \rfloor$ 为该头所属 KV 组）：

$$
h_t[b, h, p, n] = e^{dt_t[b,h] \cdot A[h]} \cdot h_{t-1}[b, h, p, n] + dt_t[b,h] \cdot B_t[b, g, n] \cdot x_t[b, h, p]
$$

$$
y_t[b, h, p] = \sum_{n=0}^{N-1} h_t[b, h, p, n] \cdot C_t[b, g, n] + D_{skip}[h] \cdot x_t[b, h, p]
$$

### 计算子步骤

1. **离散化**：$a_t = \exp(dt_t \cdot A)$，逐头标量衰减系数，$A < 0$、$dt > 0$ 保证 $a_t \in (0, 1)$
2. **组广播**：B_mat / C_mat 由 `[B, L, G, N]` 沿头维广播到 `[B, L, H, N]`（组内 H/G 个头共享）
3. **状态递归**：$h_t = a_t \cdot h_{t-1} + dt_t \cdot (B_t \otimes x_t)$，其中 $B_t \otimes x_t$ 为外积 `[P] × [N] → [P, N]`
4. **状态读出**：$y_t = h_t \cdot C_t$（沿 N 维内积）
5. **残差跳连**：$y_t \mathrel{+}= D_{skip} \odot x_t$

### SSD 分块对偶形式（kernel 预期实现，数学等价）

将序列切成长度 Q = chunk_size 的块，记块内对数衰减前缀和 $\text{cs}_t = \sum_{r \le t} dt_r A$：

- **块内（对角块，matmul 化）**：$Y^{diag} = \left[ (C X^T)^{(块内)} \odot L \right]$ 变形，其中 $L_{ts} = \exp(\text{cs}_t - \text{cs}_s) \cdot \mathbb{1}[t \ge s]$ 为 segsum 下三角衰减掩码
- **块尾状态汇聚**：$h^{blk} = \sum_{s \in 块} \exp(\text{cs}_{end} - \text{cs}_s) \cdot dt_s (B_s \otimes x_s)$
- **块间递归**：$h_{c} = \exp(\text{cs}^{(c)}_{end}) \cdot h_{c-1} + h^{blk}_c$，串行长度 L/Q
- **跨块读出**：$Y^{cross}_t = \exp(\text{cs}_t) \cdot (h_{c-1} \cdot C_t)$，$y = Y^{diag} + Y^{cross} + D_{skip} \odot x$

`chunk_size` 仅约束 kernel 的分块粒度，不影响数学结果；Golden 采用朴素递归。

## 3. 接口规范

### 算子原型

```python
mamba2_ssd(Tensor x, Tensor dt, Tensor A, Tensor B_mat, Tensor C_mat, Tensor D_skip, int chunk_size=256) -> Tensor y
```

### 输入参数说明

| 参数 | 类型 | 必需 | dtype | shape | 描述 |
|------|------|------|-------|-------|------|
| x | Tensor | 是 | float32/bfloat16 | [B, L, H, P] | 输入序列，H 个头、每头 headdim P |
| dt | Tensor | 是 | float32/bfloat16 | [B, L, H] | 逐 token 逐头时间步长 Δt，恒正（评测取值范围 [0.001, 0.1]） |
| A | Tensor | 是 | float32/bfloat16 | [H] | 逐头状态转移标量（对数衰减率），恒负（评测取值范围 [-8, -0.5]） |
| B_mat | Tensor | 是 | float32/bfloat16 | [B, L, G, N] | 输入投影系数（SSM 的 B），G 个 KV 组、组内 H/G 个头共享 |
| C_mat | Tensor | 是 | float32/bfloat16 | [B, L, G, N] | 状态读出系数（SSM 的 C），G 个 KV 组、组内 H/G 个头共享 |
| D_skip | Tensor | 是 | float32/bfloat16 | [H] | 逐头残差跳连系数 D |
| chunk_size | int | 否 | - | 标量 | SSD 分块大小，默认 256；仅约束 kernel 分块实现，不影响数学结果 |

### 输出

| 名称 | dtype | shape | 描述 |
|------|-------|-------|------|
| y | 与 x 一致 | [B, L, H, P] | 输出序列 |

### 数据类型

| x/dt/A/B_mat/C_mat/D_skip dtype | 输出 dtype | 内部计算 |
|--------------------------------|-----------|----------|
| bfloat16 | bfloat16 | fp32（状态 h 必须 fp32 保存/累加） |
| float32 | float32 | fp32 |

### 规则与约束

- 六个 Tensor 输入 dtype 必须一致
- `H % G == 0`（组共享约束），`G == H` 时退化为逐头独立 B/C，`G == 1` 时全部头共享
- dt 恒正、A 恒负（由 cases 的 value_range 保证），使 $a_t = \exp(dt \cdot A) \in (0, 1)$，递归无发散风险；算子不对违反此约定的输入负责
- 递归状态 h 以 fp32 精度保存与累加（bf16 输入场景下这是精度达标的必要条件）
- `chunk_size` 为正整数，无需整除 L（尾块按实际长度处理）
- 维度一致性：`x.shape[0:2] == dt.shape[0:2] == B_mat.shape[0:2] == C_mat.shape[0:2]`，`x.shape[2] == dt.shape[2] == A.shape[0] == D_skip.shape[0] == H`，`B_mat.shape == C_mat.shape`

### 支持范围

| 维度 / 参数 | 范围 | 备注 |
|---|---|---|
| `B`（batch） | 1 ~ 32 | cases.csv 实测 1 ~ 8 |
| `L`（序列长度） | 128 ~ 16384 | cases.csv 实测 1024 ~ 8192 |
| `H`（头数） | 8 ~ 128 | cases.csv 实测 24（Mamba2-130M）/ 64（1.3B）/ 80（2.7B） |
| `P`（headdim） | {32, 64, 128} | cases.csv 实测固定 64（Mamba2 全系默认） |
| `N`（状态维 d_state） | {64, 128} | cases.csv 实测 64 / 128 |
| `G`（KV 组数） | 1 ~ H，H % G == 0 | cases.csv 实测 1 / 8（GVA 分组） |
| `chunk_size` | {64, 128, 256} | cases.csv 实测三种均覆盖，默认 256 |
| dtype | bfloat16 / float32 | cases.csv 实测两种均覆盖 |
| `x` / `B_mat` / `C_mat` / `D_skip` 取值 | [-1, 1] | 常规随机范围 |
| `dt` 取值 | [0.001, 0.1] | 恒正，对应真实模型 softplus(dt_bias) 后的典型范围 |
| `A` 取值 | [-8, -0.5] | 恒负，对应真实模型 -exp(A_log) 的典型范围 |

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

长时间递归链（L 最长 8192）误差随序列累积，且 SSD 分块实现与 Golden 朴素递归的求和顺序不同，相对生态标准适度放宽：

| 数据类型 | FLOAT32 | BFLOAT16 |
|----------|---------|----------|
| **通过阈值(Threshold)** | 0.001 | 0.02 |

当平均相对误差 MERE < Threshold，最大相对误差 MARE < 10 * Threshold 时判定为通过。输出 y 为 N 项内积（存在相消），小值域与相消场景由评测框架的兜底标准处理。达标前提：kernel 的递归状态 h 以 fp32 保存与累加。

## 5. 标准 Golden 代码

```python
import torch


def _mamba2_ssd_core(x, dt, A, B_mat, C_mat, D_skip, compute_dtype):
    """核心计算：以 compute_dtype 精度执行朴素逐步递归。"""
    Bsz, L, H, P = x.shape
    G = B_mat.shape[2]
    N = B_mat.shape[3]
    rep = H // G  # 组内共享：每组广播到 H/G 个头

    x_f = x.to(compute_dtype)
    dt_f = dt.to(compute_dtype)
    A_f = A.to(compute_dtype)
    D_f = D_skip.to(compute_dtype)
    # B/C 按组广播到 H 个头: [B, L, G, N] -> [B, L, H, N]
    B_h = B_mat.to(compute_dtype).repeat_interleave(rep, dim=2)
    C_h = C_mat.to(compute_dtype).repeat_interleave(rep, dim=2)

    # 循环外预计算逐步系数
    a = torch.exp(dt_f * A_f.view(1, 1, H))            # [B, L, H] 衰减系数
    dtx = dt_f.unsqueeze(-1) * x_f                     # [B, L, H, P] dt_t ⊙ x_t

    h = torch.zeros(Bsz, H, P, N, dtype=compute_dtype, device=x.device)
    y = torch.empty(Bsz, L, H, P, dtype=compute_dtype, device=x.device)
    for t in range(L):
        # h_t = a_t · h_{t-1} + dt_t · (B_t ⊗ x_t)，外积 [B,H,P,1] × [B,H,1,N]
        h = a[:, t].unsqueeze(-1).unsqueeze(-1) * h \
            + dtx[:, t].unsqueeze(-1) * B_h[:, t].unsqueeze(2)
        # y_t = h_t · C_t（沿 N 维内积）+ D_skip ⊙ x_t
        y[:, t] = (h * C_h[:, t].unsqueeze(2)).sum(dim=-1) \
            + D_f.view(1, H, 1) * x_f[:, t]
    return y


def mamba2_ssd(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B_mat: torch.Tensor,
    C_mat: torch.Tensor,
    D_skip: torch.Tensor,
    chunk_size: int = 256,
) -> torch.Tensor:
    """
    Mamba2 SSD golden reference（朴素递归；plain golden = bench：fp32 状态）

    Args:
        x: [B, L, H, P] 输入序列
        dt: [B, L, H] 逐 token 逐头时间步长 Δt（恒正）
        A: [H] 逐头状态转移标量（恒负）
        B_mat: [B, L, G, N] 输入投影系数，组内共享
        C_mat: [B, L, G, N] 状态读出系数，组内共享
        D_skip: [H] 逐头残差跳连系数
        chunk_size: SSD 分块大小，仅约束 kernel 分块实现，Golden 的朴素递归不使用

    Returns:
        y: [B, L, H, P] 输出序列，dtype 与 x 一致
    """
    y = _mamba2_ssd_core(x, dt, A, B_mat, C_mat, D_skip, torch.float32)
    return y.to(x.dtype)


def mamba2_ssd_oracle(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B_mat: torch.Tensor,
    C_mat: torch.Tensor,
    D_skip: torch.Tensor,
    chunk_size: int = 256,
) -> torch.Tensor:
    """Oracle (g)：dtype-agnostic，计算精度跟随输入（fp64_cpu 下为 fp64 真值），不硬编码 .float()。"""
    return _mamba2_ssd_core(x, dt, A, B_mat, C_mat, D_skip, x.dtype)
```

## 6. 额外信息

### 算子调用示例

```python
import torch

B, L, H, P, N, G = 4, 2048, 24, 64, 128, 1

x = torch.randn(B, L, H, P, dtype=torch.bfloat16, device="npu")
dt = torch.empty(B, L, H, dtype=torch.bfloat16, device="npu").uniform_(0.001, 0.1)
A = torch.empty(H, dtype=torch.bfloat16, device="npu").uniform_(-8.0, -0.5)
B_mat = torch.randn(B, L, G, N, dtype=torch.bfloat16, device="npu")
C_mat = torch.randn(B, L, G, N, dtype=torch.bfloat16, device="npu")
D_skip = torch.randn(H, dtype=torch.bfloat16, device="npu")

y = mamba2_ssd(x, dt, A, B_mat, C_mat, D_skip, chunk_size=256)
# y.shape: [B, L, H, P]
```

### 参考文献

- Dao, T. & Gu, A. (2024). "Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality". ICML 2024（Mamba2 / SSD 算法来源）
- Gu, A. & Dao, T. (2023). "Mamba: Linear-Time Sequence Modeling with Selective State Spaces". arXiv:2312.00752
