# BatchedSvd 算子 API 描述

## 1. 算子简介

批量薄奇异值分解（economy SVD）算子：对一批 M × N（M ≥ N）矩阵同时计算 $A = U \,\text{diag}(S)\, V^T$，输出列正交的左奇异向量 U、降序非负的奇异值 S 与右奇异向量 V。SVD 的奇异向量存在符号不唯一性，本算子定义了确定性的**符号规范化约定**（见第 2 节），使不同实现（LAPACK 分治 / 单边 Jacobi）的输出可逐元素比对。

**主要应用场景**：
- 大规模 MIMO 通信中批量信道矩阵的预编码 / 波束赋形（每个子载波一个小矩阵）
- 阵列信号处理（MUSIC / ESPRIT 子空间分解、雷达快拍协方差分析）
- 数据白化（ZCA/PCA whitening）与批量最小二乘
- LoRA 权重增量 ΔW 的低秩分解与秩截断

**算子特征**：
- 难度等级：L5（NumericalStable）
- 单输入（a）三输出（u, s, v）
- 迭代数值算法：被测 kernel 预期采用**单边 Jacobi 旋转**迭代至收敛（收敛判据：off-diagonal 范数 / 列间内积低于阈值），迭代次数数据依赖
- 输出经符号规范化，任意 [-1, 1] 连续随机输入下结果确定且可比对

**为何是 L5**（约为 L4 FusedComposite 的 2 倍难度）：
- **数据依赖的迭代次数**：Jacobi 扫描直至收敛，sweep 次数取决于输入矩阵的谱分布，kernel 必须实现"迭代—判敛—提前退出"的动态控制流，而 L1~L4 算子的计算量在编译期完全确定
- **收敛控制流上硬件**：收敛判据需要全局归约（off-diagonal 范数）后广播决策，在多核 AI Core 上意味着核间同步或标量循环控制，是最难映射到 CUBE/VEC 流水的一类结构
- **旋转对配对调度**：单边 Jacobi 每个 sweep 要遍历 N(N-1)/2 个列对；并行实现需 round-robin 配对调度（每轮 N/2 个互不相交的列对并行旋转），配对表本身就是一个置换网络设计问题
- **数值细节密集**：旋转角计算（τ = (β - α)/2γ 的稳定公式）、极小 off-diagonal 的提前跳过、奇异值排序与符号规范化的后处理，全程 fp32 下保证正交性不退化
- 反观 L4 融合算子只是"多个确定性子步骤的流水拼接"，本算子的难度在算法结构本身

## 2. 算子定义

### 数学公式

$$
A = U \cdot \text{diag}(S) \cdot V^T, \quad
U^T U = I_N, \quad V^T V = I_N, \quad
S_1 \ge S_2 \ge \cdots \ge S_N \ge 0
$$

- U ∈ [B, M, N]：左奇异向量（列正交）
- S ∈ [B, N]：奇异值，降序非负
- V ∈ [B, N, N]：右奇异向量，**列向量形式**（即 $A v_j = s_j u_j$，注意与 LAPACK 返回 $V^T$ 的习惯不同）

### 符号规范化约定（必须严格遵循）

SVD 中 $(u_j, v_j) \to (-u_j, -v_j)$ 不改变 $A$，直接输出将无法逐元素比对。本算子约定：

1. 对每列 $u_j$（U 的第 j 列，长度 M），取 $|u_j|$ **最大分量**所在位置 $i^* = \arg\max_i |u_j[i]|$；
2. 若 $u_j[i^*] < 0$，则 $u_j$ 与对应 $v_j$（V 的第 j 列）**同时取反**；$u_j[i^*] \ge 0$ 则保持不变。

选 $|u_j|$ 最大分量（而非固定首分量）判定符号，避免在数值零附近判定翻转，保证不同实现规范化后结果一致。Golden 用 `argmax + gather` 定位（fp32 计算）。

### 计算子步骤（以 kernel 预期的单边 Jacobi 为例）

1. **初始化**：$W \leftarrow A$（工作矩阵 [B, M, N]），$V \leftarrow I_N$
2. **Jacobi sweep**：遍历所有列对 $(p, q)$，计算 $\alpha = \|w_p\|^2$、$\beta = \|w_q\|^2$、$\gamma = w_p^T w_q$；若 $|\gamma| > \varepsilon \sqrt{\alpha \beta}$，求 Givens 旋转角并同时右乘到 W 与 V，使两列正交
3. **收敛判断**：所有列对的归一化内积（off-diagonal 范数）低于阈值则停止，否则回到步骤 2（迭代次数数据依赖，典型 5~10 个 sweep）
4. **提取结果**：$s_j = \|w_j\|$，$u_j = w_j / s_j$；按 $s$ 降序同步重排 $u_j$ / $v_j$
5. **符号规范化**：按上述约定翻转 $(u_j, v_j)$ 符号

## 3. 接口规范

### 算子原型

```python
batched_svd(Tensor a) -> (Tensor u, Tensor s, Tensor v)
```

### 输入参数说明

| 参数 | 类型 | 必需 | dtype | shape | 描述 |
|------|------|------|-------|-------|------|
| a | Tensor | 是 | float32 | [B, M, N] | 待分解的批量矩阵，M ≥ N（由 cases 保证） |

### 输出

| 名称 | dtype | shape | 描述 |
|------|-------|-------|------|
| u | float32 | [B, M, N] | 左奇异向量（列正交，经符号规范化） |
| s | float32 | [B, N] | 奇异值，降序非负 |
| v | float32 | [B, N, N] | 右奇异向量（列向量形式，A = U diag(S) V^T，经符号规范化） |

### 数据类型

| a dtype | 输出 dtype | 内部计算 |
|---------|-----------|----------|
| float32 | float32 | fp32（正交性维持要求全程 fp32 及以上） |

### 规则与约束

- M ≥ N（薄 SVD 约定，由 cases 保证；M = N 时退化为方阵 SVD）
- 奇异值必须降序排列，u/v 列顺序与 s 严格对应
- 输出必须已按第 2 节约定完成符号规范化，否则无法通过逐元素比对
- 输入为 [-1, 1] 连续随机矩阵时奇异值几乎必然良分离（简并概率为零测集），奇异向量唯一（至符号）；算子不对人为构造的精确简并谱（如重复奇异值）的 u/v 列序与列内混叠负责
- 重建残差 $\|U \,\text{diag}(S)\, V^T - A\|_F / \|A\|_F$ 应达到 fp32 常规水平（~1e-6）

### 支持范围

| 维度 / 参数 | 范围 | 备注 |
|---|---|---|
| `B`（batch） | 1 ~ 256 | cases.csv 实测 4 ~ 128 |
| `M`（行数） | 16 ~ 512 | cases.csv 实测 32 ~ 512 |
| `N`（列数） | 8 ~ 128 | cases.csv 实测 16 ~ 128；恒有 M ≥ N，以 M ≥ 2N 为主，含少量 M = N 方阵 |
| dtype | float32 | 唯一支持 |
| 输入数值范围 | [-1, 1] | 连续随机矩阵，保证奇异值良分离 |

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

**通过标准**（本算子对结构化校验量比对，阈值收紧至 5e-4，附说明）：

| 数据类型 | FLOAT32 |
|----------|---------|
| **通过阈值(Threshold)** | 0.0005 |

当平均相对误差 MERE < Threshold，最大相对误差 MARE < 10 * Threshold 时判定为通过。

**比对对象说明**（golden.py 的 `get_input` / `get_output` 钩子，框架对 golden 与候选同等作用）：

- **输入条件化 `get_input`**：随机张量 `a` 被重建为良条件矩阵 $A = Q_u \,\text{diag}(\sigma)\, Q_v^T$，其中 $Q_u$、$Q_v$ 取自 `a` 的 QR 正交因子，$\sigma_j = c_b \cdot (1 - 0.9\,j/(N-1))$（线性 1.0 → 0.1，$c_b = \max|a_b|$ 保留 value_range 量级）。$\text{cond}(A)=10$、相邻奇异值间隔恒定，消除简并 / 接近 0 导致的 U、V 数值不适定。
- **结构化校验量 `get_output`**：精度比对不直接作用于裸 `u`/`v`，而是对三路输出统一变换为
  1. $U\,\text{diag}(S)\,V^T$（形状 [B, M, N]）—— 等价于校验重构残差 $\|A - U\,\text{diag}(S)\,V^T\|$；
  2. $S$（[B, N]）—— golden 降序非负，逐元素比对即校验排序与非负；
  3. $\text{cat}(U^T(U \odot \text{sgn}),\, V^T V) + 1$（[B, 2N, N]）—— 校验 $U^T U \approx I$、$V^T V \approx I$；$\text{sgn}$ 为每列 $|u_j|$ 最大分量的符号，未按约定规范化的列使对角元变为 $-1$ 而被拒；整体 +1 平移使 golden 无接近 0 的元素，正交性偏差按绝对量计入相对误差。

  该组校验量对合法 SVD 的自由度不敏感，对结构性错误敏感。例：候选 $U_{bad} = 1.04\,U$（S、V 正确）→ 重构 $\approx 1.04A$、Gram 对角 $\approx 2.0816$ vs $2$ → 4% 误差 ≫ 阈值，被拒；1% / 0.3% 的非正交扰动、U-V 列错配、S 未排序、符号未规范化均被拒（见 `tests/ut/test_batched_svd_structural_check.py`）。
- **阈值依据**：fp32 参考实现（LAPACK）对上述校验量实测 MERE≈1e-6、MARE≤1e-3（重构中落在小值域边界附近的元素）；取 5e-4（MARE 阈值 5e-3）较默认 2^-13 留出约 4 倍余量以容纳迭代式 fp32 kernel（Jacobi 收敛路径与 LAPACK 不同），同时保证上述结构性错误全部被拒。

## 5. 标准 Golden 代码

```python
import torch
from typing import Tuple


def _batched_svd_core(a, compute_dtype):
    """核心计算：以 compute_dtype 精度执行薄 SVD + 符号规范化。"""
    a_f = a.to(compute_dtype)
    # torch.linalg.svd 返回 (U, S, Vh)，Vh = V^T；S 已按降序排列
    u, s, vh = torch.linalg.svd(a_f, full_matrices=False)   # [B,M,N], [B,N], [B,N,N]
    v = vh.transpose(-2, -1)                                # [B, N, N]，列向量形式

    # === 符号规范化（gather 定位）===
    # 对每列 u_j 找 |u_j| 最大分量的位置: idx [B, N]
    idx = u.abs().argmax(dim=-2)                            # 沿 M 维 argmax
    # gather 取出该分量的带符号值: pivot [B, N]
    pivot = u.gather(-2, idx.unsqueeze(-2)).squeeze(-2)
    # 若该分量为负，则该列 u_j 与对应 v_j 同时取反（pivot == 0 时不翻转）
    sign = torch.where(pivot < 0,
                       torch.tensor(-1.0, dtype=compute_dtype, device=a.device),
                       torch.tensor(1.0, dtype=compute_dtype, device=a.device))  # [B, N]
    u = u * sign.unsqueeze(-2)                              # 按列翻转
    v = v * sign.unsqueeze(-2)
    return u, s, v


def batched_svd(a: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    批量薄 SVD golden reference（含符号规范化；plain golden = bench：fp32 计算）

    Args:
        a: [B, M, N] 待分解的批量矩阵，M ≥ N（由 cases 保证），float32

    Returns:
        u: [B, M, N] 左奇异向量（列正交，经符号规范化）
        s: [B, N] 奇异值，降序非负
        v: [B, N, N] 右奇异向量（列向量形式，A = U diag(S) V^T，经符号规范化）
    """
    original_dtype = a.dtype
    u, s, v = _batched_svd_core(a, torch.float32)
    return u.to(original_dtype), s.to(original_dtype), v.to(original_dtype)


def batched_svd_oracle(a: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Oracle (g)：dtype-agnostic，计算精度跟随输入（fp64_cpu 下为 fp64 真值），不硬编码 .float()。"""
    return _batched_svd_core(a, a.dtype)


def _pivot_sign(u: torch.Tensor) -> torch.Tensor:
    """每列 |u_j| 最大分量的符号 [B, N]（符号规范化后恒为 +1；该分量幅值 ≥ 1/sqrt(M)，不为 0）。"""
    idx = u.abs().argmax(dim=-2)                                     # [B, N]
    pivot = u.gather(-2, idx.unsqueeze(-2)).squeeze(-2)              # [B, N]
    return torch.where(pivot < 0, -torch.ones_like(pivot), torch.ones_like(pivot))


def get_input(a: torch.Tensor, **kwargs) -> list:
    """把随机输入重建为良条件矩阵：A = Q_u · diag(σ) · Q_v^T（同时替换 golden 与候选的输入）。

    通用生成器给出的连续随机矩阵在 M≈N 时最小奇异值趋近 0、相邻奇异值可能几乎简并，
    U/V 对应列数值不适定，逐元素比对无意义。这里以 a 的 QR 正交因子作为 Q_u [B,M,N]、
    Q_v [B,N,N]，并施加预设的良分离谱 σ_j = c_b · (1 − 0.9·j/(N−1))（线性 1.0 → 0.1，
    c_b = max|a_b| 保留 value_range 的量级语义），使 cond(A) = 10、相邻谱间隔恒定。

    Returns:
        [a_conditioned]，dtype/shape 与 a 一致。
    """
    Bsz, M, N = a.shape
    a64 = a.to(torch.float64)
    q_u, _ = torch.linalg.qr(a64)                                        # [B, M, N] 列正交
    q_v, _ = torch.linalg.qr(a64[:, :N, :].transpose(-2, -1))            # [B, N, N] 列正交
    if N > 1:
        sigma = 1.0 - 0.9 * torch.arange(N, dtype=torch.float64, device=a.device) / (N - 1)
    else:
        sigma = torch.ones(1, dtype=torch.float64, device=a.device)
    scale = a64.abs().amax(dim=(-2, -1), keepdim=True).clamp_min(1e-3)   # [B, 1, 1]
    a_cond = (q_u * sigma.view(1, 1, N)) @ q_v.transpose(-2, -1) * scale # [B, M, N]
    return [a_cond.to(a.dtype)]


def get_output(u: torch.Tensor, s: torch.Tensor, v: torch.Tensor, **kwargs) -> list:
    """把 (u, s, v) 变换为对合法 SVD 自由度不敏感、但对结构性错误敏感的校验量。

    返回三个张量（对 golden / 候选 / 同精度参考统一变换后逐元素比对）：
      1. recon = U · diag(S) · V^T            [B, M, N]  —— 重构残差 ||A − U diag(S) V^T||
      2. s                                    [B, N]     —— 奇异值（golden 降序非负，逐元素比对即校验排序/非负）
      3. gram = cat(U^T (U ⊙ sgn), V^T V) + 1 [B, 2N, N] —— 正交性 U^T U ≈ I、V^T V ≈ I，
         其中 sgn = 每列 |u_j| 最大分量的符号：候选若未按约定规范化符号，对应对角元为 −1 → 不匹配；
         整体 +1 平移使 golden 无接近 0 的元素，正交性偏差按绝对量计入相对误差。
    例：候选 U_bad = 1.04·U（S、V 正确）→ recon ≈ 1.04·A、gram 对角 ≈ 2.0816 vs 2 → 被拒。
    """
    recon = (u * s.unsqueeze(-2)) @ v.transpose(-2, -1)                  # [B, M, N]
    u_signed = u * _pivot_sign(u).unsqueeze(-2)                          # 未规范化的列 → 取反
    gram_u = u.transpose(-2, -1) @ u_signed                              # [B, N, N]
    gram_v = v.transpose(-2, -1) @ v                                     # [B, N, N]
    gram = torch.cat([gram_u, gram_v], dim=-2) + 1.0                     # [B, 2N, N]
    return [recon, s, gram]
```

## 6. 额外信息

### 算子调用示例

```python
import torch

B, M, N = 16, 256, 64

a = torch.empty(B, M, N, dtype=torch.float32, device="npu").uniform_(-1.0, 1.0)

u, s, v = batched_svd(a)
# u.shape: [B, M, N], s.shape: [B, N], v.shape: [B, N, N]

# 重建验证
recon = u @ torch.diag_embed(s) @ v.transpose(-2, -1)
assert torch.allclose(recon, a, atol=1e-4)
```

### 参考文献

- Demmel, J. & Veselić, K. (1992). "Jacobi's Method is More Accurate than QR". SIAM J. Matrix Anal. Appl. 13(4)（单边 Jacobi 的精度优势）
- Golub, G. & Van Loan, C. "Matrix Computations" (4th ed.), §8.6 Jacobi Methods（旋转对调度与收敛性）
