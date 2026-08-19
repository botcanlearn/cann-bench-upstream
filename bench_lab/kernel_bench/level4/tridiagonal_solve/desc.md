# TridiagonalSolve 算子 API 描述

## 1. 算子简介

批量三对角线性方程组求解算子。三对角方程组 $T x = b$ 是科学计算中出现频率最高的结构化线性系统之一：CFD 中 ADI（交替方向隐式）格式每个方向、每条网格线各产生一个三对角系统；三次样条插值的系数求解、热传导/扩散方程的隐式时间推进（Crank-Nicolson）、海洋与大气模式的垂直混合隐式求解等，都归结为大批量、中等规模的三对角系统。

本算子对 B 个相互独立的三对角矩阵（各由三条对角线 dl/d/du 定义）与各自的 K 个右端项批量求解。输入约定严格对角占优（d 的 value_range [4, 6]，dl/du 的 value_range [-1, 1]，即 |d| ≥ 4 > |dl| + |du| ≤ 2），保证消元过程无需选主元即数值稳定——这也是任意随机输入下解的存在唯一性与算法稳定性的保证。

**主要应用场景**：
- CFD：ADI / 近似因子化格式的隐式扫掠（大 B、中等 N）
- 三次样条插值 / 曲线拟合的系数方程组（中小 N、大批量）
- 抛物型 PDE 隐式时间推进（Crank-Nicolson、热传导）
- 气象 / 海洋模式垂直列隐式求解（列数即 batch）

**算子特征**：
- 难度等级：L4（NumericalStable）
- 四输入单输出；Golden 用 Thomas 算法（前向消元 + 回代），复杂度 O(B·N·K)
- 消元存在 N 方向的真实数据依赖（c'_i 依赖 c'_{i-1}），kernel 侧预期用 **PCR（Parallel Cyclic Reduction）/ CR（Cyclic Reduction）** 并行消元，或"分块 Thomas + 界面缩减系统"两级求解，打破串行瓶颈——这是本算子的主要难度来源
- batch 维 B 与右端项维 K 天然并行；单系统内并行必须依赖 PCR/CR
- 严格对角占优保证 Thomas 与 PCR/CR 均无需选主元、误差有界；内部计算全程 fp32

## 2. 算子定义

### 数学公式

第 b 个系统的三对角矩阵 $T_b \in \mathbb{R}^{N \times N}$ 由三条对角线定义：

$$
T_b = \begin{pmatrix}
d_{b,0} & du_{b,0} & & & \\
dl_{b,1} & d_{b,1} & du_{b,1} & & \\
 & dl_{b,2} & d_{b,2} & \ddots & \\
 & & \ddots & \ddots & du_{b,N-2} \\
 & & & dl_{b,N-1} & d_{b,N-1}
\end{pmatrix}
$$

求解 $T_b \, x_b = b_{rhs,b}$（对全部 $b \in [0, B)$、右端项 $k \in [0, K)$），即逐行满足：

$$
dl[b,i] \cdot x[b,i-1,k] + d[b,i] \cdot x[b,i,k] + du[b,i] \cdot x[b,i+1,k] = b_{rhs}[b,i,k]
$$

按约定 dl[:, 0] 与 du[:, N-1]（矩阵外的元素）被忽略，不参与计算。

### Thomas 算法（Golden 参考）

**前向消元**（$i = 1 \dots N-1$，在 B、K 维向量化）：

$$
c'_0 = \frac{du_0}{d_0}, \quad d'_0 = \frac{b_0}{d_0}; \qquad
c'_i = \frac{du_i}{d_i - dl_i \, c'_{i-1}}, \quad
d'_i = \frac{b_i - dl_i \, d'_{i-1}}{d_i - dl_i \, c'_{i-1}}
$$

**回代**（$i = N-2 \dots 0$）：

$$
x_{N-1} = d'_{N-1}, \qquad x_i = d'_i - c'_i \, x_{i+1}
$$

### 数值稳定性分析

- 严格对角占优（$|d_i| \ge 4 > |dl_i| + |du_i| \le 2$）时可归纳证明 $|c'_i| < 1$，消元分母 $|d_i - dl_i c'_{i-1}| \ge |d_i| - |dl_i| > 3$ 恒远离 0，Thomas 算法无需选主元、前向误差有界
- 同样条件下 CR/PCR 每一步消元后的缩减系统仍保持对角占优，并行消元同样稳定
- 解的幅值有界：$\|x\|_\infty \le \|b\|_\infty / (|d| - |dl| - |du|)_{\min}$，随机输入下无溢出风险

### 并行化特点（kernel 侧预期）

- **CR（Cyclic Reduction）**：每轮用相邻奇偶行消元，将系统规模减半，log2(N) 轮后回代展开，总工作量 O(N)，但每轮有同步
- **PCR（Parallel Cyclic Reduction）**：每轮所有行同时消元，log2(N) 轮后每行独立可解，工作量 O(N log N)，无回代、并行度满
- 实用方案常为两级混合：单核内分块 Thomas，块界面组成的缩减三对角系统用 PCR 求解
- B×K 个独立"右端项列"是最外层并行维度；消元系数（只依赖 dl/d/du）可在 K 个右端项间复用

## 3. 接口规范

### 算子原型

```python
tridiagonal_solve(Tensor dl, Tensor d, Tensor du, Tensor b_rhs) -> Tensor x
```

### 输入参数说明

| 参数 | 类型 | 必需 | dtype | shape | 描述 |
|------|------|------|-------|-------|------|
| dl | Tensor | 是 | float32/float16 | [B, N] | 下对角线，首元素 dl[:, 0] 按约定忽略 |
| d | Tensor | 是 | 与 dl 一致 | [B, N] | 主对角线，**严格对角占优**（评测 value_range [4, 6]） |
| du | Tensor | 是 | 与 dl 一致 | [B, N] | 上对角线，末元素 du[:, N-1] 按约定忽略 |
| b_rhs | Tensor | 是 | 与 dl 一致 | [B, N, K] | 右端项，K 个右端向量共享同一系数矩阵 |

### 输出

| 参数 | dtype | shape | 描述 |
|------|-------|-------|------|
| x | 与输入一致 | [B, N, K] | 解向量，满足 T @ x = b_rhs |

### 数据类型

| dl/d/du/b_rhs dtype | 输出 dtype | 内部计算 |
|---------------------|-----------|---------|
| float32 | float32 | fp32 |
| float16 | float16 | fp32 |

### 规则与约束

- 四个输入的 dtype 必须一致；dl、d、du 的 shape 必须完全一致（[B, N]），b_rhs 的前两维与之相同
- 系数必须严格对角占优：|d[b,i]| > |dl[b,i]| + |du[b,i]|（评测框架按 value_range 独立随机生成输入，d ∈ [4, 6]、dl/du ∈ [-1, 1] 保证该条件对任意随机输入成立）
- dl[:, 0] 与 du[:, N-1] 为矩阵外元素，按约定忽略；评测数据中它们仍按 value_range 随机填充，实现不得读取其值参与计算
- 消元与回代全程 fp32；低精度输入升精度计算后输出转回原 dtype
- kernel 实现的消元顺序（Thomas / CR / PCR / 分块混合）不限，但结果须与 Golden 的 Thomas 算法在精度阈值内一致

### 支持范围

| 维度 / 参数 | 支持值 | 备注 |
|---|---|---|
| `B`（batch，独立系统数） | 1 ~ 64 | cases.csv 实测 1 / 2 / 4 / 8 / 16 / 32 / 64 |
| `N`（系统规模） | 128 ~ 8192 | cases.csv 实测 128 / 256 / 512 / 1024 / 2048 / 4096 / 8192 |
| `K`（右端项个数） | 1 ~ 32 | cases.csv 实测 1 / 4 / 8 / 16 / 32 |
| `d` 取值 | [4, 6] | 严格对角占优主对角 |
| `dl` / `du` 取值 | [-1, 1] | 次对角 |
| `b_rhs` 取值 | [-1, 1] | |
| dtype | float32 / float16 | cases.csv 两种均覆盖 |

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
import torch


def _tridiagonal_solve_core(dl, d, du, b_rhs, compute_dtype):
    """核心计算：以 compute_dtype 精度执行 Thomas 前向消元 + 回代。"""
    dl_f = dl.to(compute_dtype)
    d_f = d.to(compute_dtype)
    du_f = du.to(compute_dtype)
    b_f = b_rhs.to(compute_dtype)
    Bsz, N, K = b_f.shape

    # 前向消元: c_prime [B, N], d_prime [B, N, K]
    c_prime = torch.zeros(Bsz, N, dtype=compute_dtype, device=b_rhs.device)
    d_prime = torch.zeros(Bsz, N, K, dtype=compute_dtype, device=b_rhs.device)
    c_prime[:, 0] = du_f[:, 0] / d_f[:, 0]
    d_prime[:, 0] = b_f[:, 0] / d_f[:, 0].unsqueeze(-1)
    for i in range(1, N):
        denom = d_f[:, i] - dl_f[:, i] * c_prime[:, i - 1]                     # [B]
        c_prime[:, i] = du_f[:, i] / denom                                     # 末行 c' 不参与回代
        d_prime[:, i] = (b_f[:, i] - dl_f[:, i].unsqueeze(-1) * d_prime[:, i - 1]) / denom.unsqueeze(-1)

    # 回代
    x = torch.zeros_like(d_prime)
    x[:, N - 1] = d_prime[:, N - 1]
    for i in range(N - 2, -1, -1):
        x[:, i] = d_prime[:, i] - c_prime[:, i].unsqueeze(-1) * x[:, i + 1]
    return x


def tridiagonal_solve(
    dl: torch.Tensor,
    d: torch.Tensor,
    du: torch.Tensor,
    b_rhs: torch.Tensor,
) -> torch.Tensor:
    """
    批量三对角线性方程组求解 (Thomas 算法；plain golden = bench：fp32 计算)

    Args:
        dl: [B, N] 下对角线, 首元素 dl[:, 0] 忽略, float32/float16
        d: [B, N] 主对角线 (严格对角占优), dtype 与 dl 一致
        du: [B, N] 上对角线, 末元素 du[:, N-1] 忽略, dtype 与 dl 一致
        b_rhs: [B, N, K] 右端项 (K 个右端向量共享同一系数矩阵), dtype 与 dl 一致

    Returns:
        x: [B, N, K] 解向量, 满足 T @ x = b_rhs, dtype 与输入一致
    """
    x = _tridiagonal_solve_core(dl, d, du, b_rhs, torch.float32)
    return x.to(b_rhs.dtype)


def tridiagonal_solve_oracle(
    dl: torch.Tensor,
    d: torch.Tensor,
    du: torch.Tensor,
    b_rhs: torch.Tensor,
) -> torch.Tensor:
    """Oracle (g)：dtype-agnostic，计算精度跟随输入（fp64_cpu 下为 fp64 真值），不硬编码 .float()。"""
    return _tridiagonal_solve_core(dl, d, du, b_rhs, b_rhs.dtype)
```

## 6. 额外信息

### 算子调用示例

```python
import torch

B, N, K = 8, 1024, 8  # ADI 扫掠场景

dl = torch.empty(B, N, dtype=torch.float32, device="npu").uniform_(-1.0, 1.0)
d = torch.empty(B, N, dtype=torch.float32, device="npu").uniform_(4.0, 6.0)
du = torch.empty(B, N, dtype=torch.float32, device="npu").uniform_(-1.0, 1.0)
b_rhs = torch.randn(B, N, K, dtype=torch.float32, device="npu")

x = tridiagonal_solve(dl, d, du, b_rhs)
# x.shape: [B, N, K]; 验证: T @ x ≈ b_rhs
```
