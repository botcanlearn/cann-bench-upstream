# DqmcHubbardGreens 算子 API 描述

## 1. 算子简介

行列式量子蒙特卡洛（Determinant Quantum Monte Carlo，DQMC，又称 BSS 算法）单步测量的核心 kernel：给定 Hubbard 模型经 Hubbard-Stratonovich 变换后的一个辅助场构型 $\{s_{l,i}\}$ 与动能矩阵 $K$，计算该构型的**等时 Green 函数** $G = A^{-1}$、**行列式符号** $\mathrm{sgn}(\det A)$ 与**对数行列式** $\log|\det A|$，其中 $A = I + B_{L_\tau} B_{L_\tau-1} \cdots B_1$，$B_l = \mathrm{diag}(e^{\lambda s_l})\, e^{-\Delta\tau K}$。Green 函数驱动全部物理观测量与局部更新接受率；$\log|\det|$ 与符号驱动构型权重。这三个量在每个 Monte Carlo 扫描中被计算 $O(N \cdot L_\tau)$ 次，是 DQMC 模拟（关联电子体系、冷原子晶格、量子临界现象数值研究）的绝对性能瓶颈。

**符号问题背景**（为何 sign 是零容差输出）：DQMC 的构型权重 $w = \det A_\uparrow \det A_\downarrow$ 在一般填充/相互作用下**不保正**。物理量按 $\langle O \rangle = \langle O\,\mathrm{sgn}\rangle_{|w|} / \langle \mathrm{sgn}\rangle_{|w|}$ 重加权，而平均符号随反演温度与体积**指数衰减**：$\langle\mathrm{sgn}\rangle \sim e^{-\beta N \Delta f}$（Loh et al. 1990；Troyer & Wiese 2005 证明其一般解由 NP-hard 问题决定）。当 $\langle\mathrm{sgn}\rangle$ 本身小到 $10^{-3}$ 量级时，任何单构型符号的计算错误都会直接摧毁重加权统计——**单构型的 sign 必须逐个精确**，这是符号问题落到 kernel 层的面目。本算子将 sign 定义为 int32 零容差输出：一步丢符号、输出翻号即失败。

**主要应用场景**：
- Hubbard 模型及其变体的有限温 DQMC 模拟（ALF、QUEST、SmoQyDQMC 等程序包的内核）
- 冷原子光晶格量子模拟的数值对照
- 辅助场量子蒙特卡洛（AFQMC）中等时 Green 函数与权重的稳定计算

**算子特征**：
- 难度等级：L5（NumericalStable）；**建议定级 L6**（见下）
- 双输入（aux_field, kinetic）+ 三 attr（lam, dtau, stab_interval），三输出（greens, sign, logdet）
- 朴素连乘的奇异值动态范围 $\sim e^{\pm L_\tau(\Delta\tau\,\lVert K\rVert + \lambda)}$，fp32 上限 $e^{88.7}$、fp64 上限 $e^{709.8}$，长链下**朴素实现连 fp64 都不准、fp32 直接溢出**（§2 给出量级估算），必须实现 QR-UDT 稳定化链
- sign 零容差（int32），greens/logdet 浮点阈值（§4）

**建议定级 L6**（仓库难度枚举现行最高为 L5，本任务按 L5 收录；按下述评估，其复杂度超出 L5——至少四根 L5 难度轴耦合于一个算子，且存在错误全局传播、不可分段验证的长链——建议仓库增设 L6 档后调升）：

| # | 难度轴 | 对应 L5 先例 | 本算子中的形态 |
|---|--------|--------------|----------------|
| 1 | 迭代正交化链 | batched_svd（单边 Jacobi 迭代正交化） | $L_\tau/\text{stab}$ 次 $N \times N$ QR 重正交串成一条链，每次 QR 的 $Q$ 列方向、$R$ 对角符号、三角因子传递全部进入后继计算 |
| 2 | 指数动态范围的序列积 | 无同量级 L5 先例（数值墙超出 L5 全部先例） | 奇异值跨度达 $e^{\pm 370}$：任何把 $D$ 以线性尺度存进 fp32 的路径必然上溢/下溢，必须 log 域携带尺度 + 大小尺度分离（Loh-Gubernatis 分层） |
| 3 | 块间严格串行 + 块内 matmul | gated_deltanet2_chunkwise（块间状态传递串行、块内 matmul 化） | $L_\tau$ 个时间片严格串行（每片依赖前片的 $U/\log d/T$），组内是 matmul 富集段（$B_l$ 连乘、$Q$ 更新、三角回代），调度上是"串行链 × 矩阵核"的流水问题 |
| 4 | 离散全局精确输出叠加浮点长链 | ntt_prime_field / ising_gibbs_philox（零容差先例，但其运算本身是整数/位精确的） | sign 是 int32 零容差输出，却要从**浮点** QR 链中逐环节收集符号（$R$ 对角符号 × $\det U$ × $\det M_{\text{inner}}$ 符号），任何一处符号约定错误都使输出恰好翻号——错误不衰减、不平均，全局传播 |
| 5（加重） | 矩阵指数 | 无 L5 先例 | $e^{-\Delta\tau K}$ 需按 §2 的对称本征分解语义实现 |

**为何复合后不可分解**：三个输出全部落在链的末端，链上没有任何可观测中间量；QR 链的每一环同时承担"正交基传递（轴1）× 尺度分离（轴2）× 串行依赖（轴3）× 符号记账（轴4）"，一环之差不产生局部可见误差，而是让末端的 $G$、sign、logdet 同时错——尤其 sign 是二值输出，错了就是 100% 错，无法靠阈值宽容，也无法把链切开分段验证（中间的 $U D T$ 分解不唯一，不同实现的中间量本就不同，只有末端三个量是良定义的比对对象）。

**性能墙**：数学上必须走稳定化链，因此"朴素"与"正解"的分野在链的组织方式：每片都重正交（组大小 1）的实现正确但 QR 占比过高、matmul 单元大量空转；正解是把 `stab_interval` 个 $B_l$ 的作用合并成组内连乘 matmul 段（算强高、可流水），组间才做一次 QR 重正交与三角因子更新，并把 $E = e^{-\Delta\tau K}$ 的本征分解一次算好复用全链。组大小受数值约束——组内连乘不重正交的对数动态范围 $\text{group}\cdot(\Delta\tau\,\lambda_{\max}(K)+\lambda)$ 必须留在工作精度的有效位内（fp32 为 $\ln(1/\varepsilon) \approx 16$，留裕量取 $\lesssim 10$），超出则组内小尺度信息在正交化前就被大尺度的舍入噪声淹没、不可恢复。`stab_interval` 是重正交节奏的**上限**（kernel 可以更频繁重正交），**不改变数学结果**（fp64 下组大小 1/4/8 的输出偏差 < 3e-12，已数值验证）。

## 2. 算子定义

### 输入译码与对称化（算子语义）

$$
s_{l} = 2 \cdot \text{aux\_field}[l] - 1 \in \{-1, +1\}^N, \qquad
K \leftarrow \tfrac{1}{2}(K + K^{\mathsf T})
$$

对称化是算子第一步（任意随机 kinetic 输入合法）。

### 矩阵指数与传播子

$$
E = \exp(-\Delta\tau K) = \sum_{k=0}^{\infty} \frac{(-\Delta\tau K)^k}{k!}
$$

$K$ 对称 ⇒ 等价于本征分解形式 $E = V\,\mathrm{diag}(e^{-\Delta\tau \lambda_i})\,V^{\mathsf T}$（$K = V \Lambda V^{\mathsf T}$），$E$ 对称正定。golden 用 `torch.matrix_exp`；两种路径数学等价，任一实现均可。

$$
B_l = \mathrm{diag}(e^{\lambda s_l})\, E, \qquad
A = I + B_{L_\tau} B_{L_\tau - 1} \cdots B_1
$$

**输出**：$\text{greens} = A^{-1}$，$\text{sign} = \mathrm{sgn}(\det A) \in \{-1, +1\}$，$\text{logdet} = \log|\det A|$。（$\det B_l = e^{\lambda \sum_i s_{l,i}} e^{-\Delta\tau\,\mathrm{tr}K} > 0$，但 $\det A$ 可为负——这正是符号问题。）

### 为什么朴素连乘必然失败（量级估算）

$B$ 链的奇异值增长率由 $K$ 的谱半径与 $\lambda$ 决定：乘满 $L_\tau$ 片后奇异值跨度约 $e^{\pm L_\tau (\Delta\tau\, \lambda_{\max}(K) + \lambda)}$。随机对称 $K \in [-1,1]^{N\times N}$ 的 $\lambda_{\max} \approx 2\sqrt{N/3}$（半圆律）：

| 规模 | 每片对数增长 $\Delta\tau\lambda_{\max}+\lambda$ | $L_\tau$ 片总跨度 | 后果 |
|------|------|------|------|
| $N=256$, $\Delta\tau=0.0625$, $\lambda=0.3$ | $\approx 1.46$ | $L_\tau = 64$: $e^{\pm 93}$ | **fp32 朴素连乘上溢**（上限 $e^{88.7}$）；实测 $L_\tau=192$ 链 fp32 直乘产出 inf/nan |
| $N=100$, $\Delta\tau=0.125$, $\lambda=0.5$ | $\approx 1.94$ | $L_\tau = 192$: $e^{\pm 372}$ | fp64 直乘虽不上溢（上限 $e^{709.8}$），但小尺度早已淹没在大尺度的舍入噪声里：实测 $\log|\det|$ 偏差 $O(10^4)$、sign 随机翻号 |
| $N=24$, $\Delta\tau=0.125$, $\lambda=0.4$ | $\approx 1.0$ | $L_\tau = 96$: $e^{\pm 92}$ | 已用 mpmath 高精度参考证实：fp64 朴素 greens 偏差 $O(1)$、logdet 偏差 $O(10^2)$、**sign 输出错误**，而稳定化链偏差 < 1.2e-13 |

### QR-UDT 稳定化（本算子验证过的具体形式）

维护部分积的分解 $B_l \cdots B_1 = U\,\mathrm{diag}(e^{\log d})\,T$：$U$ 正交、$\log d \in \mathbb{R}^N$（对数域尺度）、$T$ 单位上三角。

**关键恒等式（QR 对列缩放的等变性）**：设 $M$ 满秩、$D$ 正对角，则 $M\,\mathrm{diag}(D)$ 与 $M$ 的（Householder/Gram-Schmidt）QR 分解有相同的 $Q$（至多差列符号），且 $R_{MD} = R_M\,\mathrm{diag}(D)$——因为正交化只依赖各列的方向与顺序，列缩放不改变方向。于是**尺度因子 $D$ 永远不需要以线性尺度进入任何矩阵**：QR 始终作用于 $O(1)$ 范数的工作矩阵，动态范围只存在于向量 $\log d$ 和 $T$ 更新的比值因子中。

**链式更新**（组大小 $g$ = 实现自选的重正交间隔，须满足 §1 的工作精度可行性约束且 $g \le$ `stab_interval`；golden 参考实现取 $g = \min(8,\, \lfloor 10/(\Delta\tau\lambda_{\max}+\lambda) \rfloor)$，数学结果与组大小无关）：

1. 组内连乘：$M = B_{l+g-1} \cdots B_l\, U$（每步 $M \leftarrow \mathrm{diag}(e^{\lambda s})\,(E\,M)$，行缩放 + matmul）。
2. QR：$Q, \tilde R = \mathrm{QR}(M)$；$\sigma_j = \mathrm{sgn}(\tilde R_{jj})$（0 取 +1）。
3. 更新：

$$
U' = Q\,\mathrm{diag}(\sigma), \qquad
\log d'_j = \log d_j + \log|\tilde R_{jj}|, \qquad
T' = F\,T, \quad
F_{jk} = \frac{\sigma_j \tilde R_{jk}}{|\tilde R_{jj}|}\, e^{\log d_k - \log d_j} \ (j \le k)
$$

$F$ 单位上三角（$F_{jj} = 1$），故 $T$ 全程单位上三角、$\det T = 1$。（$e^{\log d_k - \log d_j}$ 在 $j < k$ 时因 $\log d$ 沿链自然降序而 $\le O(1)$；实测全部 case 空间内 $\max|T| < 10^3$、$\mathrm{cond}(T) < 10^9$。）

**最终稳定求解（Loh-Gubernatis 大小尺度分离，全程无上溢）**：

$$
D_b^{-1} = e^{-\max(\log d,\, 0)} \le 1, \qquad D_s = e^{\min(\log d,\, 0)} \le 1
$$

利用 $U U^{\mathsf T} = I$：

$$
A = I + U D T = U\, D_b \underbrace{\left( D_b^{-1} U^{\mathsf T} T^{-1} + D_s \right)}_{M_{\text{inner}}\text{（元素均 } O(1)\text{）}}\, T
$$

$$
\boxed{\;\text{greens} = T^{-1}\, M_{\text{inner}}^{-1}\, D_b^{-1} U^{\mathsf T}\;}
\qquad
\boxed{\;\text{logdet} = \sum_j \max(\log d_j, 0) + \log|\det M_{\text{inner}}|\;}
$$

$$
\boxed{\;\text{sign} = \mathrm{sgn}(\det U) \cdot \mathrm{sgn}(\det M_{\text{inner}})\;}
$$

**符号记账（每一处符号来源）**：(i) 每次 QR 的 $R$ 对角符号 $\sigma$ 折入 $U$（保持 $\log d$ 记录的是正尺度）；(ii) $T$ 的行归一化因子是 $|\tilde R_{jj}|$，其符号已被 (i) 取走，$T$ 单位上三角 ⇒ $\det T = 1$，对符号无贡献；(iii) $\det D_b > 0$；(iv) 剩余两个符号源是正交阵 $U$（$\det = \pm 1$，条件数 1，可安全求出）与 $O(1)$ 良态的小矩阵 $M_{\text{inner}}$（slogdet）。$T^{-1}$ 用单位上三角回代，无除法主元问题。

**公式验证结果**（全部先于本文档写定完成）：
- 小动态范围（$N \le 36$, $L_\tau \le 16$）：与朴素 fp64 直乘对照，greens 偏差 < 1.8e-15、logdet < 7.2e-15、sign 全部一致、$G \cdot A - I$ 残差 < 1.3e-14
- 中动态范围：logdet 与 `torch.linalg.slogdet(I + 直乘链)` 一致到 < 1.5e-14（相对）
- 大动态范围（$N=24$, $L_\tau=96$，跨度 $e^{\pm 92}$）：与 mpmath（130 位十进制）参考对照，稳定式 greens 偏差 1.2e-13、logdet 1.1e-13、sign 一致；同点朴素 fp64 greens 偏差 0.84、logdet 偏差 274、sign 错误
- 极端规模（$N=100$, $L_\tau=192$, $\lambda=0.5$, $\Delta\tau=0.125$）：无上溢自洽恒等式 $G A = G + T^{-1} M_{\text{inner}}^{-1} D_s\, T = I$ 残差 1.7e-12；组大小 1/4/8 输出互差 < 3.3e-12
- fp32 稳定实现（按上式可行性规则选组）：全部校准 case 上 sign 与 fp64 一致（含 $\sigma_{\min}(M_{\text{inner}}) > 10^{-5}$ 余量断言），greens/logdet 误差见 §4

## 3. 接口规范

### 算子原型

```python
dqmc_hubbard_greens(Tensor aux_field, Tensor kinetic, float lam, float dtau, int stab_interval=8) -> (Tensor greens, Tensor sign, Tensor logdet)
```

### 输入参数说明

| 参数 | 类型 | 必需 | dtype | shape | 描述 |
|------|------|------|-------|-------|------|
| aux_field | Tensor | 是 | int8 | [Ltau, N] | 辅助场构型，0 → s=−1、1 → s=+1 |
| kinetic | Tensor | 是 | float32 | [N, N] | 动能矩阵，算子内部对称化（评测取值范围 [-1, 1]） |
| lam | float | 是 | - | 标量 | Hubbard-Stratonovich 耦合 λ（评测取值范围 [0.1, 0.5]） |
| dtau | float | 是 | - | 标量 | 虚时间步长 Δτ（评测取值范围 [0.05, 0.125]） |
| stab_interval | int | 否 | - | 标量 | 重正交间隔上限，默认 8；kernel 可更频繁重正交（须满足 §1/§2 的工作精度可行性约束），不影响数学结果 |

### 输出

| 名称 | dtype | shape | 描述 |
|------|-------|-------|------|
| greens | float32 | [N, N] | 等时 Green 函数 $A^{-1}$ |
| sign | int32 | [1] | $\mathrm{sgn}(\det A) \in \{-1, +1\}$（零容差） |
| logdet | float32 | [1] | $\log|\det A|$ |

### 数据类型

| aux_field | kinetic | greens/logdet | sign | 内部计算 |
|-----------|---------|---------------|------|----------|
| int8 | float32 | float32 | int32 | 稳定化链（尺度必须 log 域携带；线性域中间量须保持在工作精度可表示范围内，见 §2） |

### 规则与约束

- $K$ 的对称化是算子语义的一部分：非对称输入先做 $(K+K^{\mathsf T})/2$
- `stab_interval` 为正整数，是重正交节奏的上限；任意节奏下三个输出的数学值一致（golden 参考实现按 §2 可行性规则自适应选组，不使用该 attr）
- sign 只能取 $\pm 1$（$A$ 满秩由 case 生成时校准保证），int32 精确比对
- 尺度向量禁止以线性尺度存入工作精度（fp32 上限 $e^{88.7}$，case 空间内 $\log d$ 跨度可达 $e^{\pm 370}$）
- 输出须为 contiguous 张量

### 支持范围

| 维度 / 参数 | 范围 | 备注 |
|---|---|---|
| `N`（格点数） | 36 ~ 400 | cases.csv 实测 36 ~ 256（16×16 格 = 256）；隐藏用例至 400（20×20 格） |
| `Ltau`（虚时间片数） | 16 ~ 192 | cases.csv 实测 16 ~ 192 |
| `lam` | 0.1 ~ 0.5 | attr |
| `dtau` | 0.05 ~ 0.125 | attr |
| `stab_interval` | 1 ~ 32 | attr，默认 8 |
| `aux_field` 取值 | [0, 1] | int8 |
| `kinetic` 取值 | [-1, 1] | 内部对称化 |

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

**通过标准**（浮点输出放宽阈值 + 整数输出零容差）：

| 输出 | 数据类型 | 阈值 |
|------|----------|------|
| greens / logdet | FLOAT32 | 0.02 |
| sign | INT32 | **0（零容差，逐元素完全相等）** |

浮点输出：MERE < 0.05 且 MARE < 0.5 判定通过。greens 是矩阵逆：fp32 实现的绝对误差沿全矩阵均布，而 $|G|$ 元素随机过零，过零点附近的相对误差尖峰由评测框架的小值域/相消兜底标准处理（native 参考 = plain golden，即 fp32 稳定化链本身；native 与被测同为 fp32 路径时兜底比较才公平）。阈值经评测框架 checker 实测定值：plain（fp32 稳定化链，可行性规则分组）vs fp64 oracle 在全部 100 case × 6 draw 上双跑校准，checker 全量判 passed，实测 greens 正常值域 MERE ≤ 2.6e-3、max|ΔG| ≤ 2.1e-3、logdet MERE ≤ 4.3e-4；0.05 对 MERE 留 ≥19x 裕量，与同为"误差全矩阵均布"类的 batched_svd 先例（float32 0.05）一致。四个上界同时取满的角点（N≥100 且 Ltau≥128 且 λ=0.5 且 Δτ=0.125，实测 fp32 误差进入百分位）不在 case 空间内。

**sign 零容差的可行性余量**：cases 生成时逐 case（6 draw）断言：(i) plain(fp32)、独立路径 fp32 实现（$E$ 经对称本征分解）与 fp64 三方 sign 一致；(ii) $\sigma_{\min}(M_{\text{inner}}) > 10^{-5}$（实测全部 draw ≥ 5.1e-4，fp32 扰动量级 ~1e-6，距奇异点留 ≥500x 余量）。正确实现稳定化链的 kernel 不会因浮点抖动翻符号；丢失任何一处符号记账（§2）则必然翻号。

## 5. 标准 Golden 代码

```python
from typing import Tuple

import torch


def _dqmc_udt_chain(aux_field, kinetic, lam, dtau, compute_dtype, group):
    """UDT 链：返回 (U, logd, T)，使 B_Ltau...B_1 = U @ diag(exp(logd)) @ T。"""
    n = kinetic.shape[0]
    ltau = aux_field.shape[0]
    dev = kinetic.device

    k_sym = kinetic.to(compute_dtype)
    k_sym = 0.5 * (k_sym + k_sym.transpose(0, 1))            # 对称化（算子语义）
    e_mat = torch.matrix_exp(-dtau * k_sym)                  # 对称正定
    s = aux_field.to(compute_dtype) * 2.0 - 1.0              # {0,1} -> {-1,+1}
    v_diag = torch.exp(lam * s)                              # [Ltau, N] 逐 slice 对角因子

    one = torch.tensor(1.0, dtype=compute_dtype, device=dev)
    u = torch.eye(n, dtype=compute_dtype, device=dev)
    logd = torch.zeros(n, dtype=compute_dtype, device=dev)
    t = torch.eye(n, dtype=compute_dtype, device=dev)

    steps = 0
    for l in range(ltau):
        u = v_diag[l].unsqueeze(1) * (e_mat @ u)             # B_l @ (U 工作矩阵)
        steps += 1
        if steps == group or l == ltau - 1:
            q, r = torch.linalg.qr(u)
            diag = r.diagonal()
            sigma = torch.where(diag < 0, -one, one)
            absd = diag.abs()
            u = q * sigma.unsqueeze(0)                       # 列符号修正 => diag(R') > 0
            rn = (sigma.unsqueeze(1) * r) / absd.unsqueeze(1)
            w = torch.exp(logd.unsqueeze(0) - logd.unsqueeze(1))   # w[j,k]=exp(logd_k-logd_j)
            factor = torch.triu(rn * w)
            factor.diagonal().fill_(1.0)                     # 单位上三角
            t = factor @ t
            logd = logd + torch.log(absd)
            steps = 0
    return u, logd, t


def _dqmc_stable_outputs(u, logd, t):
    """由 UDT 分解稳定计算 (greens, sign, logdet)，全程无大数上溢。"""
    n = u.shape[0]
    eye = torch.eye(n, dtype=u.dtype, device=u.device)
    t_inv = torch.linalg.solve_triangular(t, eye, upper=True, unitriangular=True)
    db_inv = torch.exp(-torch.clamp(logd, min=0.0))          # ≤ 1
    ds = torch.exp(torch.clamp(logd, max=0.0))               # ≤ 1
    m_inner = db_inv.unsqueeze(1) * (u.transpose(0, 1) @ t_inv) + torch.diag(ds)

    sgn_inner, logabs_inner = torch.linalg.slogdet(m_inner)
    sgn_u, _ = torch.linalg.slogdet(u)                       # 正交阵，det = ±1
    logdet = torch.clamp(logd, min=0.0).sum() + logabs_inner
    sign = sgn_u * sgn_inner
    greens = t_inv @ torch.linalg.solve(m_inner, db_inv.unsqueeze(1) * u.transpose(0, 1))
    return greens, sign, logdet


def _dqmc_hubbard_greens_core(aux_field, kinetic, lam, dtau, compute_dtype, group=1):
    """核心计算：QR-UDT 稳定化链 + 稳定输出。group 为重正交组大小。"""
    u, logd, t = _dqmc_udt_chain(aux_field, kinetic, lam, dtau, compute_dtype, group)
    return _dqmc_stable_outputs(u, logd, t)


def _stable_group_size(kinetic, lam, dtau, cap=8, budget=10.0):
    """按工作精度可行性规则选重正交组大小（desc §2）：
    组内对数动态范围 group * (dtau*max|λ(K_sym)| + |lam|) ≤ budget（≈ ln(1/eps_fp32) 留裕量），
    并以 cap 为上限。golden 参考实现固定 cap=8。"""
    k_sym = 0.5 * (kinetic + kinetic.transpose(0, 1))
    lam_max = float(torch.linalg.eigvalsh(k_sym.float()).abs().max())
    per_slice = abs(dtau) * lam_max + abs(lam)
    return max(1, min(cap, int(budget / max(per_slice, 1e-6))))


def dqmc_hubbard_greens(
    aux_field: torch.Tensor,
    kinetic: torch.Tensor,
    lam: float,
    dtau: float,
    stab_interval: int = 8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """plain golden = bench：fp32 稳定化链（重正交组固定 8，即正确 fp32 kernel 的数据通路），
    greens/logdet 输出 fp32，sign int32。

    stab_interval 为被测 kernel 重正交节奏的上限（kernel 可更频繁重正交），不影响数学
    结果（数学值 = fp64 真值，见 oracle）；golden 参考实现按 §2 的工作精度可行性规则
    自适应选组（上限 8），不使用该 attr。
    """
    group = _stable_group_size(kinetic, float(lam), float(dtau))
    greens, sign, logdet = _dqmc_hubbard_greens_core(
        aux_field, kinetic, float(lam), float(dtau), torch.float32, group=group)
    return (greens.to(torch.float32),
            sign.to(torch.int32).reshape(1),
            logdet.to(torch.float32).reshape(1))


def dqmc_hubbard_greens_oracle(
    aux_field: torch.Tensor,
    kinetic: torch.Tensor,
    lam: float,
    dtau: float,
    stab_interval: int = 8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Oracle (g)：dtype-agnostic，浮点计算精度跟随 kinetic（golden_precision=fp64_cpu 下即
    fp64 真值；fp64 下重正交组大小不影响结果，实测组 1/4/8 偏差 < 3.3e-12）；
    sign 恒为 int32（类别量）。fp32 输入下与 plain golden 逐位一致。"""
    group = _stable_group_size(kinetic, float(lam), float(dtau))
    greens, sign, logdet = _dqmc_hubbard_greens_core(
        aux_field, kinetic, float(lam), float(dtau), kinetic.dtype, group=group)
    return (greens.to(kinetic.dtype),
            sign.to(torch.int32).reshape(1),
            logdet.to(kinetic.dtype).reshape(1))
```

## 6. 额外信息

### 算子调用示例

```python
import torch

N, Ltau = 144, 96          # 12x12 格，beta = Ltau * dtau = 9.6

aux_field = torch.randint(0, 2, (Ltau, N), dtype=torch.int8, device="npu")
kinetic = torch.rand(N, N, dtype=torch.float32, device="npu") * 2 - 1

greens, sign, logdet = dqmc_hubbard_greens(aux_field, kinetic,
                                           lam=0.4, dtau=0.1, stab_interval=8)
# greens.shape: [N, N]，sign.shape: [1]（±1），logdet.shape: [1]
```

### 可用于自检的性质（均已数值验证）

- **自洽性**：$G \cdot A = I$（大动态范围下用无上溢形式 $G A = G + T^{-1} M_{\text{inner}}^{-1} D_s T$ 验证）
- **译码对称性**：aux 全 0 配 $\lambda$ 与 aux 全 1 配 $-\lambda$ 逐位同结果
- **stab 不变性**：重正交组大小（1/4/8/…）不改变输出
- **中动态范围对照**：logdet 与 `torch.linalg.slogdet(I + 直乘链)` 一致
- **行列式恒等**：$\text{logdet} = \sum_j \max(\log d_j, 0) + \log|\det M_{\text{inner}}|$ 与直接 slogdet 在可表示范围内一致

### 参考文献

- Blankenbecler, R., Scalapino, D. J., Sugar, R. L. (1981). "Monte Carlo calculations of coupled boson-fermion systems". Phys. Rev. D 24, 2278（BSS 算法）
- Loh, E. Y., Gubernatis, J. E. et al. (1990). "Sign problem in the numerical simulation of many-electron systems". Phys. Rev. B 41, 9301（符号问题 + 大小尺度分离稳定化）
- Troyer, M., Wiese, U.-J. (2005). "Computational complexity and fundamental limitations to fermionic quantum Monte Carlo simulations". Phys. Rev. Lett. 94, 170201（符号问题 NP-hard）
- Assaad, F. F., Evertz, H. G. (2008). "World-line and Determinantal Quantum Monte Carlo Methods". Lecture Notes in Physics 739（UDT/QR 稳定化链的标准表述）
- Bai, Z., Chen, C., Scalettar, R., Yamazaki, I. (2009). "Numerical methods for quantum Monte Carlo simulations of the Hubbard model"（稳定化线性代数分析）
