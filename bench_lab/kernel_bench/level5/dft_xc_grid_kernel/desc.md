# DftXcGridKernel 算子 API 描述

## 1. 算子简介

DFT（密度泛函理论）数值积分核：格点密度评估 → PBE 交换-相关泛函逐点求值 → XC 能量与 $V_{xc}$ 矩阵组装。这是 Gaussian / PySCF / GPU4PySCF 等量子化学软件中 `numint` 模块的热点——SCF 每步迭代都要在数万个积分格点上评估电子密度 $\rho$ 及其梯度、逐点求值交换-相关泛函（每点约 20 次 pow/ln/exp/div 级超越函数运算）、再把泛函导数组装回基函数矩阵。三个阶段的算力属性截然不同：密度评估与 $V_{xc}$ 组装是 $[G, N_b] \times [N_b, N_b]$ 级 GEMM，泛函求值是纯逐元素超越函数链，且三段之间存在数据依赖的串联。

本算子取无自旋极化（closed-shell）的 PBE 泛函（GGA 家族的事实标准），语义规格完全自足：全部公式与常数在 §2 给全，$v_\rho / v_\sigma$ 定义为泛函的精确偏导，并附带两个可逐 case 验证的强契约（$V_{xc}$ 对称性与变分一致性 $\partial E_{xc}/\partial \texttt{dm}_{\mu\nu} = V_{xc,\mu\nu}$）。

**主要应用场景**：
- KS-DFT SCF 迭代的 XC 能量与势矩阵构造（Gaussian/PySCF/GPU4PySCF numint 热点）
- 单算子内"超越函数逐点链 + GEMM 组装"的数据依赖混载负载（本任务集首个此机制轴的算子；与 colocated_prefill_decode 的双负载共驻呼应但机制不同——这里的混载是同一数据流内的串联，不是两个独立负载的并置）
- 大格点批量下的分块流水与中间量驻留优化

**算子特征**：
- 难度等级：L5（FusedComposite）
- 四输入双输出（标量能量 + 对称矩阵），无属性参数
- 数值防护（$\rho/\sigma$ 下限与 $t^2$ 上限）是规格的一部分，任意随机输入合法
- $v_\rho$、$v_\sigma$ 为泛函对 $\rho$、$\sigma$ 的**精确偏导**（golden 用 torch.autograd 实现；实现方可用解析式、自动微分或任何等价手段）
- **能量精度契约锚定化学精度**：case 空间定标到真实分子 $|E_{xc}|$ 量级，阈值保证逐 case 能量绝对误差 ≤ 1 kcal/mol（见 §4）；$V_{xc}$ 输出经常数平移规范化（见 §2 第六步）

**为何是 L5**（约为 L4 FusedComposite 的 2 倍难度）：
- **正确 ≠ 快**：按 §2 逐阶段直译（密度 GEMM → 逐点泛函 → 组装 GEMM，各阶段全场张量往返）即可得到完全正确的结果并通过全部精度用例，但性能只能锚定朴素 baseline（预期得分 ~0.5 档）。逼近硬件下界要求按格点分块，把"密度评估 → 泛函链 → 组装"排成分块流水：泛函链是纯 VEC 超越函数吞吐，密度/组装是 CUBE GEMM，两类单元的占用天然互补，分块流水使其重叠；$v_\rho/v_\sigma$ 的求值方式（解析式展开 vs 数值手段）与中间量（$\rho, \nabla\rho, \sigma, v_\rho, v_\sigma$ 共 $O(G)$ 级 6 条流）的驻留与复用也是设计空间。**混载编排与导数求值方案的选择是被测内容**，desc 不提供
- **不规则精确项**：PBE/PW92 泛函链有 12 个规格常数、两层嵌套超越函数（$\ln$ 内含 $A(ε_c(\rho))$ 反馈）与三处数值防护，每个常数、每处防护位置、$t^2$ 与 $s^2$ 两个易混的约化梯度定义都会被隐藏用例单独检验（附标准 $r_s$ 点的文献值锚，见 §2）
- **契约可判**：变分一致性 $\partial E_{xc}/\partial \texttt{dm} = V_{xc}$ 把能量路径与势矩阵路径锁死为自洽整体（fp64 中心差分验证至 3.7e-11），$v_\rho/v_\sigma$ 的任何近似误差、组装公式的任何缺项（如漏因子 2、漏对称项）都会破坏该契约
- **实现规格而非过可见测试**：隐藏评测集覆盖公开用例之外的规格维度（dm 全 0 / 常数秩 1 / ±10 大值域、ao_grad 全 0 的 LDA 退化、素数 G、非对齐 Nb、G≈Nb 与 G≫Nb、防护区活跃的极端值域等）

## 2. 算子定义

单位制：Hartree 原子单位。$G$ = 格点数，$N_b$ = 基函数数；$\mu,\nu \in [0,N_b)$，$g \in [0,G)$。

### 第一步：密度矩阵对称化 + 格点密度评估

$$
D = \tfrac{1}{2}\big(\texttt{dm} + \texttt{dm}^{\mathsf T}\big) \quad \text{（幂等投影；物理密度矩阵是其不动点）}
$$

$$
\rho_g = \sum_{\mu\nu} \varphi_{g\mu} D_{\mu\nu} \varphi_{g\nu}, \qquad
(\nabla\rho)^{x}_g = 2\sum_{\mu\nu} (\partial_x\varphi)_{g\mu} D_{\mu\nu} \varphi_{g\nu} \ \ (x=1,2,3), \qquad
\sigma_g = |\nabla\rho_g|^2
$$

其中 $\varphi = \texttt{ao\_values}$，$\partial\varphi = \texttt{ao\_grad}$。注意 $\nabla\rho$ 的因子 2 来自 $D$ 的对称性（$\nabla(\varphi D \varphi) = 2\,\partial\varphi\, D\, \varphi$）；对称化不可省略，否则 $\nabla\rho$ 对非对称随机 `dm` 即错。

### 第二步：数值防护（规格的一部分，保证任意随机输入合法）

$$
\rho \leftarrow \max(\rho,\ 10^{-8}), \qquad
\sigma \leftarrow \max(\sigma,\ 10^{-20}), \qquad
t^2 \leftarrow \min(t^2,\ 10^{8})
$$

三处防护都在"可微语义"之内：$v_\rho/v_\sigma$ 是防护后泛函的精确偏导，截断点处偏导为 0。取值依据：
- $\rho$ 下限 $10^{-8}$：保证 fp32 通路上 $s^2 = \sigma/(4c\rho^{8/3})$ 一类除式的反向传播（分母平方 $\ge 6.9\times10^{-41}$）与 $A^2t^4$（$\le 2.8\times10^{35}$）均不上溢/下溢产生 inf——下限取 $10^{-10}$ 时分母平方 $\sim10^{-49}$ 在 fp32 下溢为 0，$v_\rho$ 出 inf（随机低密度输入实测可复现）
- $t^2$ 上限的依据：$H$ 在大 $t$ 极限精确饱和于 $-\varepsilon_c$（截断处相对偏差 $\le 1/(A\,t^2) \le 10^{-5}$，实测 1.0e-14）

### 第三步：PBE 交换

$$
e_x^{\text{unif}}(\rho) = -\tfrac{3}{4}\left(\tfrac{3}{\pi}\right)^{1/3} \rho^{4/3}, \qquad
s^2 = \frac{\sigma}{4\,(3\pi^2)^{2/3}\, \rho^{8/3}}
$$

$$
F_x = 1 + \kappa - \frac{\kappa}{1 + \mu s^2/\kappa}, \qquad
\kappa = 0.804, \quad \mu = 0.2195149727645171
$$

$$
e_{x} = e_x^{\text{unif}} \cdot F_x \quad \text{（交换能量密度，每单位体积；已含 } \rho^{4/3} \text{ 因子）}
$$

### 第四步：PW92 均匀电子气关联 + PBE 梯度修正

PW92（无自旋极化，$\zeta = 0$）：

$$
r_s = \left(\frac{3}{4\pi\rho}\right)^{1/3}, \qquad
\varepsilon_c^{\text{unif}}(r_s) = -2A(1+\alpha_1 r_s)\,
\ln\!\left[1 + \frac{1}{2A(\beta_1 r_s^{1/2} + \beta_2 r_s + \beta_3 r_s^{3/2} + \beta_4 r_s^2)}\right]
$$

$$
A = 0.0310907,\ \ \alpha_1 = 0.21370,\ \ \beta_1 = 7.5957,\ \ \beta_2 = 3.5876,\ \ \beta_3 = 1.6382,\ \ \beta_4 = 0.49294
$$

标准 $r_s$ 点的数值锚（本公式在 fp64 下的精确值，实现可用于自检，Hartree）：

| $r_s$ | 0.5 | 1.0 | 2.0 | 5.0 |
|---|---|---|---|---|
| $\varepsilon_c^{\text{unif}}$ | −0.0766187 | −0.0597737 | −0.0447595 | −0.0282162 |

PBE 梯度修正（无自旋极化 $\phi = 1$）：

$$
t^2 = \frac{\sigma}{4\,k_s^2\, \rho^2}, \qquad
k_s = \sqrt{\frac{4 k_F}{\pi}}, \qquad
k_F = (3\pi^2\rho)^{1/3}
\qquad\Big(\text{即 } t^2 = \frac{\pi\,\sigma}{16\,(3\pi^2\rho)^{1/3}\rho^2}\Big)
$$

$$
A_H = \frac{\beta/\gamma}{\exp(-\varepsilon_c^{\text{unif}}/\gamma) - 1}, \qquad
H = \gamma \ln\!\left[1 + \frac{\beta}{\gamma} t^2 \frac{1 + A_H t^2}{1 + A_H t^2 + A_H^2 t^4}\right]
$$

$$
\gamma = \frac{1-\ln 2}{\pi^2}, \qquad \beta = 0.06672455060314922
$$

$$
e_{c} = \rho\,(\varepsilon_c^{\text{unif}} + H) \quad \text{（关联能量密度，每单位体积；含 } \rho \text{ 因子）}
$$

**易混警示**：$s^2$（交换用，分母 $\rho^{8/3}$）与 $t^2$（关联用，分母 $(3\pi^2\rho)^{1/3}\rho^2 = k_F\rho^2$）是两个不同的约化梯度，不可混用。

### 第五步：能量与 $V_{xc}$ 组装

$$
\texttt{exc\_energy} = \sum_g w_g\, \big(e_{x,g} + e_{c,g}\big) \quad \in \mathbb{R}^{1}
$$

量纲说明：$e_x$ 已含 $\rho^{4/3}$、$e_c$ 已含 $\rho$，加和时**不再**乘 $\rho$。

$$
v_{\rho,g} = \frac{\partial e_{xc}}{\partial \rho}\Big|_g, \qquad
v_{\sigma,g} = \frac{\partial e_{xc}}{\partial \sigma}\Big|_g
\quad \text{（防护后泛函 } e_{xc} = e_x + e_c \text{ 的精确偏导）}
$$

golden 用 torch.autograd 对 $\rho/\sigma$ 叶子张量求导得到精确偏导；实现方可用解析式、自动微分或任何数学等价手段。LDA 部分的解析锚（用于自检）：$\partial e_x^{\text{unif}}/\partial\rho = -(3/\pi)^{1/3}\rho^{1/3}$。

$$
V_{xc,\mu\nu} = \sum_g w_g \Big[ v_{\rho,g}\, \varphi_{g\mu}\varphi_{g\nu}
+ 2 v_{\sigma,g} \big( (\nabla\rho_g \!\cdot\! (\nabla\varphi)_{g\mu})\, \varphi_{g\nu}
+ \varphi_{g\mu}\, (\nabla\rho_g \!\cdot\! (\nabla\varphi)_{g\nu}) \big) \Big]
$$

### 第六步：输出规范化（评测口径，规格的一部分）

$$
\texttt{vxc\_shifted} = V_{xc} + C, \qquad C = 10.0
$$

物理 $V_{xc}$ 的元素在零点两侧密集分布（非对角元围绕 0），逐元素相对误差判据在零点邻域没有意义。与 batched_svd 的符号规范化同型，本算子把一个常数平移写进输出语义：平移后非对角元聚在 $+C$ 附近、对角元（case 空间定标后 $\approx C-5$）远离零点，比较变为 $|err|/(|V_{xc}+C|) \approx |err|/C$ 的绝对误差口径，能量阈值得以按化学精度收紧（§4）。$C=10.0$ 取物理 $V_{xc}$ 元素量级（定标后对角元 $\sim -5$、非对角元 $\sim \pm 0.3$）之上、又不吞没判别力的档位。使用方以 $\texttt{vxc\_shifted} - C$ 还原物理 $V_{xc}$；平移使还原值的绝对精度下限为 $C\cdot\varepsilon_{fp32} \approx 1.2\times10^{-6}$ Ha，远低于化学显著性。

**契约**（对任意随机输入数学成立，隐藏用例逐 case 检验的锚）：
- 对称性：$\texttt{vxc\_shifted} = \texttt{vxc\_shifted}^{\mathsf T}$（fp64 验证至 2.5e-15 相对偏差）
- 变分一致性：$\partial\, \texttt{exc\_energy} / \partial\, \texttt{dm}_{\mu\nu} = \texttt{vxc\_shifted}_{\mu\nu} - C$（对原始非对称 `dm` 的逐元素偏导；fp64 中心差分验证至 3.7e-11）——能量路径与势矩阵路径的整体自洽约束
- 退化关系：`ao_grad`=0 $\Rightarrow$ $F_x = 1$、$H = 0$，退化为 LSDA（LDA 交换 + PW92）；`dm`=0 $\Rightarrow$ $\rho$ 全场走防护下限，$\texttt{exc}$ 有限且 $\texttt{vxc\_shifted} \equiv C$（精确）；`dm` 为常数矩阵 $\Rightarrow$ 秩 1 密度

## 3. 接口规范

### 算子原型

```python
dft_xc_grid_kernel(Tensor ao_values, Tensor ao_grad, Tensor dm, Tensor grid_weights) -> (Tensor exc_energy, Tensor vxc_shifted)
```

### 输入参数说明

| 参数 | 类型 | 必需 | dtype | shape | 描述 |
|------|------|------|-------|-------|------|
| ao_values | Tensor | 是 | float32 | [G, Nb] | 基函数在格点上的值 φ_gμ |
| ao_grad | Tensor | 是 | float32 | [3, G, Nb] | 基函数梯度三分量 |
| dm | Tensor | 是 | float32 | [Nb, Nb] | 密度矩阵，先对称化 D=(dm+dmᵀ)/2；评测值域逐 case 定标使 \|Exc\| 为真实分子量级 |
| grid_weights | Tensor | 是 | float32 | [G] | 积分权重，恒正；评测取值 [1e-6, min(1e-2, 16/G)]（Σw ~ 常数的物理口径） |

### 输出

| 名称 | dtype | shape | 描述 |
|------|-------|-------|------|
| exc_energy | float32 | [1] | XC 能量 Σ w·e_xc（Hartree；化学契约 sub-kcal/mol，见 §4） |
| vxc_shifted | float32 | [Nb, Nb] | 平移后 XC 势矩阵 V_xc + 10.0（对称；∂exc_energy/∂dm = vxc_shifted − 10.0） |

### 数据类型

| 输入 dtype | 输出 dtype | 内部计算 |
|-----------|-----------|----------|
| float32 | float32 | fp32（超越函数链逐点求值；低密度分支建议用 log1p/expm1 类高精度原语，见 §4） |

### 规则与约束

- 四个 Tensor 输入 dtype 必须一致（float32）
- 维度一致性：ao_values [G,Nb] 与 ao_grad [3,G,Nb] 共享 G、Nb；dm 为 [Nb,Nb] 方阵；grid_weights 为 [G]
- 数值防护（ρ 下限 1e-8、σ 下限 1e-20、t² 上限 1e8）与输出平移 +10.0 是语义的一部分，必须实现；防护在可微语义之内（截断点偏导为 0）
- grid_weights 恒正由 cases 的 value_range 保证；ao/dm 任意随机值合法（防护兜底）
- 显存约束：case 空间保证全部输入+输出按 fp64 计合计 ≤ 2 GB
- 两个输出均须为 contiguous 张量

### 支持范围

| 维度 / 参数 | 范围 | 备注 |
|---|---|---|
| `G`（格点数） | 500 ~ 65536 | cases.csv 实测 500 ~ 65536，主档 2000 ~ 50000 |
| `Nb`（基函数数） | 8 ~ 256 | cases.csv 实测 8 ~ 256 |
| dtype | float32 | 单一 dtype |
| `ao_values` 取值 | [-2, 2] | 常规评测取 [-1, 1] |
| `ao_grad` 取值 | [-10, 10] | 常规评测取 [-1, 1]；全 0 为 LDA 退化用例 |
| `dm` 取值 | [-1e4, 1e4] | 常规评测逐 case 定标（\|Exc\| 目标 min(0.32·Nb, 30) Ha）；±10/±1e4 为值域鲁棒性用例 |
| `grid_weights` 取值 | [1e-6, 1e-2] | 恒正；常规评测上界取 min(1e-2, 16/G) |

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

**通过标准**（本算子按化学精度锚定阈值）：

| 数据类型 | FLOAT32 |
|----------|---------|
| **通过阈值(Threshold)** | 3e-5 |

当平均相对误差 MERE < Threshold，最大相对误差 MARE < 10 * Threshold 时判定为通过。

**化学精度契约**（阈值依据）：

1 kcal/mol = 1.5936e-3 Ha。case 空间的 `dm`/`grid_weights` 值域逐 case 定标到真实分子量级：$|E_{xc}|$ 目标 $\min(0.32 N_b, 30)$ Ha（对照真实体系：H₂O/def2-SVP $\approx -9.2$ Ha、苯/def2-SVP $\approx -37$ Ha），实测非鲁棒性档 $|E_{xc}| \in [2.7, 52]$ Ha（含 draw 间波动）。相对阈值 3e-5 给出的逐 case 能量绝对误差上界：

| 档位 | 典型 \|Exc\| (Ha) | 3e-5 × \|Exc\| (kcal/mol) |
|---|---|---|
| 小分子（H₂O/NH₃/CH₄ 档，Nb ≤ 48） | 3 ~ 15 | 0.06 ~ 0.28 |
| 中体系（乙醇/甘氨酸/苯档，Nb 72~119） | 20 ~ 36 | 0.38 ~ 0.68 |
| 大基组/大体系（Nb 128~256） | 25 ~ 52 | 0.47 ~ 0.98 |

即全部分子档 case 的能量误差契约 **sub-kcal/mol**（最大 0.98 kcal/mol）。参考实现（fp32 数据通路）实测能量相对误差 $\le 1.4\times10^{-7}$，换算 $\le 0.004$ kcal/mol，距契约上界有 200x 以上裕量。值域鲁棒性档（dm 全 0/常数/±10/±1e4、权重常数、ao_grad 全 0/±10 等）的 $|E_{xc}|$ 不代表分子体系，不在 kcal 锚内，由同一相对判据（尺度无关）覆盖。

**阈值可行性与 $V_{xc}$ 的处理**：物理 $V_{xc}$ 元素过零密集，直接用相对判据在 3e-5 档必然误杀——输出规范化 $V_{xc}+10$（§2 第六步）把逐元素判据变为绝对误差口径。用独立 fp32 实现（pow 走 exp/log、$\ln(1+x)$/$\exp(x)-1$ 走朴素式、归约顺序翻转、偏导建在替代图上）vs fp64 oracle 经评测框架 checker 实测：23 组 case 配置全部通过；plain golden 对全部 100 case × 2~8 seed 扫描 0 失败，worst MERE = 3.6e-7（裕量 83x）、worst MARE = 3.6e-5（MARE 判据 3e-4，裕量 8.4x）。**数值建议**：低密度分支的 $\ln(1+1/\text{den})$ 与 $\exp(-\varepsilon_c/\gamma)-1$ 用 log1p/expm1 类高精度原语求值（golden 即如此实现）。判别力（实测，3e-5）：漏 dm 对称化、$\nabla\rho$ 漏因子 2、漏 $H$ 修正、PW92 常数用错、$e_c$ 多乘 $\rho$、漏输出平移、$t^2$ 误用 $s^2$ 在定标档与值域档均被拒；$V_{xc}$ 梯度项漏因子 2 在定标档低于阈值（2.2e-5）但被值域档（dm±10 的用例）以 MERE=9.2e-5、MARE=1.8e-3 明确拒绝——实现必须通过全部用例，判别力由 case 空间共同承担。

## 5. 标准 Golden 代码

```python
import math
from typing import Tuple

import torch



# === PBE 交换常数 ===
_PBE_KAPPA = 0.804
_PBE_MU = 0.2195149727645171

# === PBE 关联常数 ===
_PBE_BETA = 0.06672455060314922
_PBE_GAMMA = (1.0 - math.log(2.0)) / (math.pi ** 2)

# === PW92 无自旋极化 ε_c(r_s) 常数 ===
_PW92_A = 0.0310907
_PW92_ALPHA1 = 0.21370
_PW92_BETA1 = 7.5957
_PW92_BETA2 = 3.5876
_PW92_BETA3 = 1.6382
_PW92_BETA4 = 0.49294

# === 数值防护下限/上限（规格的一部分） ===
_RHO_FLOOR = 1e-8
_SIGMA_FLOOR = 1e-20
_T2_CEIL = 1e8

# === 输出规范化常数（规格的一部分，见 desc §2）===
# vxc_shifted = vxc + _VXC_SHIFT；使用方减去该常数还原物理 V_xc
_VXC_SHIFT = 10.0


def _xc_energy_density(rho_raw, sigma_raw):
    """逐点 XC 能量密度 e_xc(ρ, σ) [G]（含数值防护，可微，dtype 跟随输入）。"""
    rho = torch.clamp(rho_raw, min=_RHO_FLOOR)
    sigma = torch.clamp(sigma_raw, min=_SIGMA_FLOOR)

    # --- PBE 交换 ---
    # e_x^unif = −(3/4)(3/π)^{1/3} ρ^{4/3}
    ex_unif = -0.75 * (3.0 / math.pi) ** (1.0 / 3.0) * torch.pow(rho, 4.0 / 3.0)
    # s² = σ / (4 (3π²)^{2/3} ρ^{8/3})
    s2 = sigma / (4.0 * (3.0 * math.pi ** 2) ** (2.0 / 3.0) * torch.pow(rho, 8.0 / 3.0))
    # F_x = 1 + κ − κ/(1 + μ s²/κ)
    fx = 1.0 + _PBE_KAPPA - _PBE_KAPPA / (1.0 + _PBE_MU * s2 / _PBE_KAPPA)
    ex_density = ex_unif * fx

    # --- PW92 均匀电子气关联 ε_c^unif(r_s) ---
    # r_s = (3/(4πρ))^{1/3}
    rs = torch.pow(3.0 / (4.0 * math.pi) / rho, 1.0 / 3.0)
    rs_sqrt = torch.sqrt(rs)
    den = 2.0 * _PW92_A * (_PW92_BETA1 * rs_sqrt + _PW92_BETA2 * rs
                           + _PW92_BETA3 * rs * rs_sqrt + _PW92_BETA4 * rs * rs)
    eps_c = -2.0 * _PW92_A * (1.0 + _PW92_ALPHA1 * rs) * torch.log1p(1.0 / den)

    # --- PBE 梯度修正 H(ρ, σ, ε_c) ---
    # t² = σ / (4 k_s² ρ²)，k_s = √(4 k_F/π)，k_F = (3π²ρ)^{1/3}（无自旋极化 φ=1）
    kf = torch.pow(3.0 * math.pi ** 2 * rho, 1.0 / 3.0)
    t2 = sigma * math.pi / (16.0 * kf * rho * rho)
    t2 = torch.clamp(t2, max=_T2_CEIL)
    # A = (β/γ) / [exp(−ε_c/γ) − 1]
    a_h = (_PBE_BETA / _PBE_GAMMA) / torch.expm1(-eps_c / _PBE_GAMMA)
    at2 = a_h * t2
    # H = γ ln[1 + (β/γ) t² (1 + A t²)/(1 + A t² + A² t⁴)]
    h = _PBE_GAMMA * torch.log1p(
        (_PBE_BETA / _PBE_GAMMA) * t2 * (1.0 + at2) / (1.0 + at2 + at2 * at2))
    ec_density = rho * (eps_c + h)

    return ex_density + ec_density


def _dft_xc_grid_kernel_core(ao_values, ao_grad, dm, grid_weights, compute_dtype):
    """核心计算：以 compute_dtype 精度执行，返回 (exc_energy [1], vxc_matrix [Nb, Nb])。"""
    phi = ao_values.to(compute_dtype)
    dphi = ao_grad.to(compute_dtype)
    d = dm.to(compute_dtype)
    w = grid_weights.to(compute_dtype)

    # === 1. 密度矩阵对称化投影（幂等）+ 格点密度/梯度 ===
    d = (d + d.transpose(0, 1)) / 2
    phid = phi @ d                                             # [G, Nb] = (φ D)
    rho = (phi * phid).sum(dim=-1)                             # [G]
    grad_rho = 2.0 * (dphi * phid.unsqueeze(0)).sum(dim=-1)    # [3, G]
    sigma = (grad_rho * grad_rho).sum(dim=0)                   # [G]

    # === 2/3. 逐点泛函求值 + 精确偏导 v_ρ = ∂e/∂ρ、v_σ = ∂e/∂σ（autograd） ===
    rho_leaf = rho.detach().requires_grad_(True)
    sigma_leaf = sigma.detach().requires_grad_(True)
    e_xc = _xc_energy_density(rho_leaf, sigma_leaf)            # [G]
    vrho, vsigma = torch.autograd.grad(e_xc.sum(), (rho_leaf, sigma_leaf))

    # === 4. 能量积分 ===
    exc_energy = (w * e_xc.detach()).sum().reshape(1)          # [1]

    # === 5. V_xc 矩阵组装（GEMM） ===
    wvr = w * vrho                                             # [G]
    wvs = w * vsigma                                           # [G]
    m = (grad_rho.unsqueeze(-1) * dphi).sum(dim=0)             # [G, Nb] = ∇ρ·∇φ
    vxc = phi.transpose(0, 1) @ (wvr.unsqueeze(-1) * phi)
    vxc = vxc + 2.0 * (m.transpose(0, 1) @ (wvs.unsqueeze(-1) * phi)
                       + phi.transpose(0, 1) @ (wvs.unsqueeze(-1) * m))

    # === 6. 输出规范化：常数平移（评测口径，规格的一部分，见 desc §2）===
    vxc_shifted = vxc + _VXC_SHIFT
    return exc_energy.contiguous(), vxc_shifted.contiguous()


def dft_xc_grid_kernel(
    ao_values: torch.Tensor,
    ao_grad: torch.Tensor,
    dm: torch.Tensor,
    grid_weights: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    DFT XC 数值积分 golden reference（plain golden = bench：fp32 数据通路）

    Args:
        ao_values: [G, Nb] 基函数在格点上的值 φ_gμ
        ao_grad: [3, G, Nb] 基函数梯度三分量 (∂φ/∂x, ∂φ/∂y, ∂φ/∂z)
        dm: [Nb, Nb] 密度矩阵，算子先对称化 D = (dm + dmᵀ)/2
        grid_weights: [G] 积分权重（正，评测取值范围 [1e-6, 1e-2]）

    Returns:
        exc_energy: [1] XC 能量 Σ_g w_g e_xc,g，dtype 与 ao_values 一致
        vxc_shifted: [Nb, Nb] 平移后 XC 势矩阵 V_xc + 10.0（对称；
                     ∂exc_energy/∂dm_μν = vxc_shifted_μν − 10.0），dtype 与 ao_values 一致
    """
    exc_energy, vxc_shifted = _dft_xc_grid_kernel_core(
        ao_values, ao_grad, dm, grid_weights, torch.float32)
    return exc_energy.to(ao_values.dtype), vxc_shifted.to(ao_values.dtype)


def dft_xc_grid_kernel_oracle(
    ao_values: torch.Tensor,
    ao_grad: torch.Tensor,
    dm: torch.Tensor,
    grid_weights: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Oracle (g)：dtype-agnostic，计算精度跟随输入（fp64_cpu 下为 fp64 真值），不硬编码 .float()。"""
    return _dft_xc_grid_kernel_core(
        ao_values, ao_grad, dm, grid_weights, ao_values.dtype)
```

## 6. 额外信息

### 算子调用示例

```python
import torch

G, Nb = 24000, 114   # 苯 / def2-SVP 量级

ao_values = torch.empty(G, Nb, dtype=torch.float32, device="npu").uniform_(-1, 1)
ao_grad = torch.empty(3, G, Nb, dtype=torch.float32, device="npu").uniform_(-1, 1)
dm = torch.empty(Nb, Nb, dtype=torch.float32, device="npu").uniform_(-0.03, 0.03)
grid_weights = torch.empty(G, dtype=torch.float32, device="npu").uniform_(1e-6, 6.7e-4)

exc_energy, vxc_shifted = dft_xc_grid_kernel(ao_values, ao_grad, dm, grid_weights)
# exc_energy.shape: [1]；vxc_shifted.shape: [Nb, Nb]，vxc_shifted == vxc_shifted.T
vxc_physical = vxc_shifted - 10.0   # 还原物理 V_xc
```

### 真实分子量级对照（cases.csv 标注口径）

| 体系 / 基组 | Nb | 典型格点 G |
|---|---|---|
| H₂O / def2-SVP | 24 | ~34000 |
| H₂O / def2-TZVP | 43 | ~34000 |
| 乙醇 / def2-SVP | 72 | ~46000 |
| 甘氨酸 / def2-SVP | 95 | ~50000 |
| 苯 / def2-SVP | 114 | ~50000（实际 ~130k，评测截取） |
| 苯 / def2-TZVP | 222 | ~50000 |
| 萘 / def2-SVP | 180 | ~50000 |

### 公式验证（golden 交付前已执行）

- 均匀电子气极限（ao_grad=0）：exc 与 $V_{xc}$ 与独立手写 LDA-x + PW92（解析 $v_\rho$）全流水线对照，相对偏差 ≤ 2.1e-15
- PW92 $\varepsilon_c$ 在 $r_s$ = 0.5/1/2/5 与独立标量实现对照 ≤ 6.1e-16（§2 表中数值锚）
- autograd $v_\rho$ 与 LDA 解析式 $-(3/\pi)^{1/3}\rho^{1/3}$ 对照 ≤ 6.6e-16
- 独立逐点标量实现（G=5, Nb=3，纯 Python float）：能量 1.1e-16、组装（平移域）逐位、FD 偏导 4.4e-08
- $\texttt{vxc\_shifted}$ 对称性 ≤ 2.5e-15；变分一致性 $\partial E/\partial \texttt{dm}$ 中心差分 ≤ 3.7e-11（含防护区活跃的一般随机输入）
- $t^2$ 截断处 $H$ 饱和于 $-\varepsilon_c$，偏差 1.0e-14；ρ 下限 1e-8 的 fp32 反向安全性经 16 seed 低密度 case 验证（1e-10 下实测出 inf）

### 参考文献

- Perdew, J. P., Burke, K., Ernzerhof, M. (1996). "Generalized Gradient Approximation Made Simple". Phys. Rev. Lett. 77, 3865（PBE 泛函与 $t$、$A$、$H$ 的标准定义）
- Perdew, J. P., Wang, Y. (1992). "Accurate and simple analytic representation of the electron-gas correlation energy". Phys. Rev. B 45, 13244（PW92 常数表）
- Sun, Q. et al. (2020). "Recent developments in the PySCF program package". J. Chem. Phys. 153, 024109（numint 格点积分流水线，本算子的负载来源）
