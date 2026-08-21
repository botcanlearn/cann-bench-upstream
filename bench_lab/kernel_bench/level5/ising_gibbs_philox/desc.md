# IsingGibbsPhilox 算子 API 描述

## 1. 算子简介

周期边界 2D 晶格上的棋盘（checkerboard）Gibbs 更新算子：**bit-exact 随机模拟——计数器 RNG + 并行更新顺序即规格，输出零容差**。每个格点的随机数不来自任何有状态的随机数流，而是由计数器型 RNG Philox4x32-10 按 (sweep, color, batch, site) 四元组即时算出——随机性是输入 (seed, counter) 的确定性函数，因此整个模拟的结果逐 bit 唯一确定，评测按 int8 零容差精确比对。

棋盘分解是并行 MCMC 的标准做法：把晶格按 (i+j) 奇偶二染色，同色格子的四邻居全是异色，因此一个颜色相内的全部格子可以**以任意顺序（含全并行）更新而结果不变**——"并行顺序即规格"。计数器 RNG（Random123 谱系，SC 2011 最佳论文）则是 GPU/加速器上大规模并行随机模拟的标准原语：无状态、可跳读、每个格点独立生成，是 cuRAND / JAX 等框架并行 RNG 的基础。两者组合出一个语义完全确定、却必须在片上高效生成随机数的 L5 算子。

`accept_table` 是任意 [2, 5] 的 32 位无符号阈值表：行 = 当前 spin，列 = 邻居和索引。它把"是否翻转"定义成一次整数比较，**任意随机表都合法**——该表定义一个确定性元胞自动机，物理 Ising 模型的 Boltzmann 接受概率表（对 exp(−ΔE/T) 的定点化）只是特例，评测不要求物理性。

**主要应用场景**：
- 统计物理 / 自旋系统的大规模并行 Monte Carlo 模拟（Ising / Potts / spin glass）
- 概率计算与组合优化（模拟退火、Boltzmann 机采样）的加速器原语
- 并行随机数生成（counter-based RNG）与可复现随机模拟的正确性基准

**算子特征**：
- 难度等级：L5（FusedComposite）
- 双张量输入（spins int8、accept_table int64）+ 三整数属性（num_sweeps、seed_hi、seed_lo），单输出 spins_out int8
- 全整数计算（Philox + 比较 + 翻转），无浮点，输出零容差
- 双色相交替的顺序依赖：白格必须看到本 sweep 更新后的黑格

**为何是 L5**：
- **规格自足，不考知识**：Philox4x32-10 的全部常数、轮函数、key 调度与计数器映射在 §2 给全并附已验证的测试向量，实现不需要查任何资料；难点在把一个"逐格点定义"的随机模拟映射成高吞吐的片上计算
- **正确 ≠ 快（性能墙）**：按 §2 直译（每个颜色相一次全晶格遍历、随机数逐点算或先物化成随机场）完全正确，但性能只能锚定朴素 baseline（预期得分 ~0.5 档）。物化全部随机数需要 B·H·W·num_sweeps × 4 字节（支持范围上限下约 128 MiB）的额外 GM 流量，把 RNG 从计算问题变成访存问题；高效实现必须**按计数器在片上即时生成随机数**（Philox 每格点 10 轮整数乘/异或，纯寄存器计算、零访存），并处理双色相交替带来的核间同步（每个颜色相是一次全局同步点，2·num_sweeps 个相）与棋盘格的间隔访存模式
- **零容差下的位级推理**：32×32 位乘积达 2^64 量级，超出带符号 64 位范围；mulhi/mullo 的实现路径（拆分、进位）必须逐 bit 正确——错一个进位就是逐 bit 失败，且大规模随机输入下必然触发
- **实现规格而非过可见测试**：隐藏评测集覆盖最小晶格 H=W=2（周期边界下四邻居退化为两个格点各计两次）、H≠W、全 0 / 全 1 表与自旋、seed=(0,0)、非 2 幂偶数维等公开集之外的规格维度

## 2. 算子定义

### 晶格与棋盘分解

自旋场 $\mathrm{spins} \in \{0, 1\}^{B \times H \times W}$（0 代表自旋 $\sigma = -1$，1 代表 $\sigma = +1$，即 $\sigma = 2 \cdot \mathrm{spin} - 1$）。晶格按周期边界（torus）取邻居；**H、W 均为偶数**（奇数维在周期边界下会破坏棋盘二染色，见"规则与约束"）。格点 (i, j) 的颜色 = (i + j) mod 2（0 黑、1 白）。batch 维 B 内各晶格相互独立。

### 一个 sweep 的定义

一个 sweep 依次执行两个颜色相：**先**更新全部黑格（c = 0），**再**更新全部白格（c = 1）。同色格子的四邻居全是异色，因此一个颜色相内的更新相互独立、以任意顺序执行结果相同；但白格相读取的是**本 sweep 黑格相更新完成后**的自旋场。整个算子执行 num_sweeps 个 sweep（sweep 编号 s = 0..num_sweeps−1）。

### 单个格点的更新规则

对 sweep s、颜色相 c、格点 (b, i, j)（满足 (i+j) mod 2 = c）：

1. 四邻居自旋和（周期边界）：

$$
\mathrm{nsum} = \sum_{(i',j') \in \mathcal{N}(i,j)} \big(2 \cdot \mathrm{spin}[b,i',j'] - 1\big), \qquad
\mathcal{N} = \{(i{\pm}1 \bmod H,\, j),\ (i,\, j{\pm}1 \bmod W)\}
$$

nsum ∈ {−4, −2, 0, 2, 4}，索引 $k = (\mathrm{nsum} + 4) / 2 \in \{0, 1, 2, 3, 4\}$。

2. 生成 32 位随机数（x0 为 Philox 输出的第一个字）：

$$
r = \mathrm{Philox4x32\text{-}10}\big(\mathrm{key} = (\mathrm{seed\_hi}, \mathrm{seed\_lo}),\ \mathrm{counter} = (s,\ c,\ b,\ i \cdot W + j)\big).x_0
$$

3. 翻转判定（无符号 32 位整数比较）：

$$
\text{若 } r < \mathrm{accept\_table}[\mathrm{spin}[b,i,j],\ k] \text{，则 } \mathrm{spin}[b,i,j] \leftarrow 1 - \mathrm{spin}[b,i,j]
$$

### Philox4x32-10 完整规范

状态为 4 个 32 位无符号计数器字 $(c_0, c_1, c_2, c_3)$ 与 2 个 32 位 key 字 $(k_0, k_1)$。本算子的映射（**按此顺序**）：

$$
(c_0, c_1, c_2, c_3) = (s,\ c,\ b,\ i \cdot W + j), \qquad (k_0, k_1) = (\mathrm{seed\_hi},\ \mathrm{seed\_lo})
$$

常数：乘数 $M_0 = \mathtt{0xD2511F53}$、$M_1 = \mathtt{0xCD9E8D57}$；key 增量 $W_0 = \mathtt{0x9E3779B9}$、$W_1 = \mathtt{0xBB67AE85}$。

对 32 位无符号数定义 $\mathrm{mullo}(a, x) = (a \cdot x) \bmod 2^{32}$，$\mathrm{mulhi}(a, x) = \lfloor (a \cdot x) / 2^{32} \rfloor$（精确的 64 位乘积的低/高 32 位）。

**共 10 轮**，每轮依次执行（所有运算 mod $2^{32}$，⊕ 为按位异或）：

$$
(c_0, c_1, c_2, c_3) \leftarrow \big(\mathrm{mulhi}(M_1, c_2) \oplus c_1 \oplus k_0,\ \ \mathrm{mullo}(M_1, c_2),\ \ \mathrm{mulhi}(M_0, c_0) \oplus c_3 \oplus k_1,\ \ \mathrm{mullo}(M_0, c_0)\big)
$$

$$
k_0 \leftarrow (k_0 + W_0) \bmod 2^{32}, \qquad k_1 \leftarrow (k_1 + W_1) \bmod 2^{32}
$$

（轮内先用当前 key 完成置换，再增量 key；第 10 轮后的 key 增量对输出无影响。）输出 $x_0$ = 10 轮后的 $c_0$。

**测试向量**（(c0,c1,c2,c3; k0,k1) → 10 轮后的 (x0,x1,x2,x3)，与 Random123 参考实现一致，均已数值验证）：

| counter | key | 输出 |
|---------|-----|------|
| (00000000, 00000000, 00000000, 00000000) | (00000000, 00000000) | (6627e8d5, e169c58d, bc57ac4c, 9b00dbd8) |
| (ffffffff, ffffffff, ffffffff, ffffffff) | (ffffffff, ffffffff) | (408f276d, 41c83b0e, a20bc7c6, 6d5451fd) |
| (243f6a88, 85a308d3, 13198a2e, 03707344) | (a4093822, 299f31d0) | (d16cfe09, 94fdcceb, 5001e420, 24126ea1) |

**实现提示（位级正确性）**：32×32 位乘积最大 $(2^{32}-1)^2 \approx 1.8 \times 10^{19}$，超出带符号 64 位上界 $2^{63}-1 \approx 9.2 \times 10^{18}$；若中间算术使用带符号 64 位，需按 16 位（或更细）拆分乘法（如 $a \cdot x = a \cdot x_{hi} \cdot 2^{16} + a \cdot x_{lo}$，全部中间量 < $2^{49}$），或使用无符号 64 位。任何逐 bit 等价的路径均可接受。

### accept_table 语义

$\mathrm{accept\_table} \in [0, 2^{32}-1]^{2 \times 5}$（int64 承载）：行下标 = 当前 spin ∈ {0, 1}，列下标 = 邻居和索引 k ∈ {0..4}。$r < \mathrm{table}[\mathrm{spin}, k]$ 时翻转，因此表值 0 表示该状态永不翻转、表值 $2^{32}-1$ 表示以概率 $(2^{32}-1)/2^{32}$ 翻转。表是任意的：**评测不要求表具有物理意义**（物理 Boltzmann 表——如 heat-bath 规则 $\mathrm{table}[\mathrm{spin}, k] = \lfloor 2^{32} \cdot p_{\mathrm{flip}} \rfloor$——只是特例）。

## 3. 接口规范

### 算子原型

```python
ising_gibbs_philox(Tensor spins, Tensor accept_table, int num_sweeps, int seed_hi, int seed_lo) -> Tensor spins_out
```

### 输入参数说明

| 参数 | 类型 | 必需 | dtype | shape | 描述 |
|------|------|------|-------|-------|------|
| spins | Tensor | 是 | int8 | [B, H, W] | 初始自旋场，取值 {0, 1}；H、W 均为偶数 |
| accept_table | Tensor | 是 | int64 | [2, 5] | 32 位无符号阈值表，取值 [0, 2^32−1]，任意随机表合法 |
| num_sweeps | int | 是 | - | 标量 | sweep 次数（一个 sweep = 黑格相 + 白格相），评测取值 1 ~ 64 |
| seed_hi | int | 是 | - | 标量 | Philox key 字 k0，按无符号 32 位解释 |
| seed_lo | int | 是 | - | 标量 | Philox key 字 k1，按无符号 32 位解释 |

### 输出

| 名称 | dtype | shape | 描述 |
|------|-------|-------|------|
| spins_out | int8 | [B, H, W] | num_sweeps 个 sweep 后的自旋场，取值 {0, 1} |

### 数据类型

| spins dtype | accept_table dtype | spins_out dtype | 内部计算 |
|-------------|--------------------|-----------------|----------|
| int8 | int64 | int8 | 全整数（golden 为 int64 中间量 + 16 位拆分乘法；kernel 任何逐 bit 等价路径均可） |

### 规则与约束

- **H、W 均为偶数**（≥ 2）：周期边界下奇数维会使 (i+j) 奇偶染色在环绕处相邻同色，破坏棋盘二染色——该约束由 cases 保证，算子不对奇数维输入负责
- spins 取值 {0, 1}、accept_table 取值 [0, 2^32−1]（由 value_range 保证）
- 颜色相顺序固定：每个 sweep 内先黑（c=0）后白（c=1）；白格相必须读取本 sweep 黑格相更新后的自旋场
- 同一颜色相内更新顺序任意（含全并行），结果唯一
- 计数器字均在 32 位无符号范围内：s < num_sweeps ≤ 64，c ∈ {0,1}，b < B，i·W+j < H·W
- H = W = 2 等退化晶格下邻居按周期边界正常取（上下邻居为同一格点、各计一次，左右同理）
- 结果零容差：与数学定义逐 bit 一致（见 §4）；输出须为 contiguous 张量

### 支持范围

| 维度 / 参数 | 范围 | 备注 |
|---|---|---|
| `B`（batch） | 1 ~ 64 | cases.csv 实测 1 ~ 64 |
| `H`、`W`（晶格维度） | 2 ~ 512（偶数） | cases.csv 实测 2 ~ 512，含 H ≠ W 与非 2 幂偶数 |
| `num_sweeps` | 1 ~ 64 | cases.csv 实测 1 ~ 64 |
| `B·H·W·num_sweeps` | ≤ 2^25 | 控制单 case 计算量 |
| `seed_hi` / `seed_lo` | [0, 2^32−1] | 含 (0, 0) |
| `spins` 取值 | {0, 1} | 特殊值 case 用 [0,0] / [1,1] |
| `accept_table` 取值 | [0, 2^32−1] | 特殊值 case 用 [0,0]（永不翻转）/ [2^32−1, 2^32−1]（几乎必翻） |

## 4. 精度要求

**零容差**：输出为 int8，精度阈值为 0——评测框架对整数输出走精确比对路径（torch.equal 逐元素完全相等），**一 bit 错即失败**。不适用 MERE/MARE 浮点误差指标，无小值域/相消兜底。

golden 为全整数精确实现（int64 中间量 + 16 位拆分乘法，全程无浮点），plain golden 与 oracle 结果完全一致；评测的 fp64 golden 路径不改变整数输入（框架仅对浮点输入升精度），因此参考结果与本规格的数学定义逐 bit 相同。

## 5. 标准 Golden 代码

```python
import torch

_M0 = 0xD2511F53
_M1 = 0xCD9E8D57
_W0 = 0x9E3779B9
_W1 = 0xBB67AE85
_MASK32 = 0xFFFFFFFF


def _mulhilo32(m, x):
    """32×32 位乘积的 (hi, lo) 32 位字。m 为 python int 常数，x 为 int64 张量（值 < 2^32）。

    直接乘会超 int64（(2^32−1)^2 > 2^63−1），按 16 位拆分：
    m·x = m·x_hi·2^16 + m·x_lo，全部中间量 < 2^49。
    """
    x_hi = x >> 16
    x_lo = x & 0xFFFF
    p_hi = m * x_hi                      # < 2^48
    p_lo = m * x_lo                      # < 2^48
    t = p_lo + ((p_hi & 0xFFFF) << 16)   # < 2^49
    lo = t & _MASK32
    hi = (p_hi >> 16) + (t >> 32)        # < 2^32
    return hi, lo


def _philox4x32_10_x0(c0, c1, c2, c3, k0, k1):
    """Philox4x32-10，返回输出第一个字 x0。c0..c3 为 int64 张量（可广播），k0/k1 为 python int。"""
    for _ in range(10):
        hi0, lo0 = _mulhilo32(_M0, c0)
        hi1, lo1 = _mulhilo32(_M1, c2)
        c0 = hi1 ^ c1 ^ k0
        c1 = lo1
        c2 = hi0 ^ c3 ^ k1
        c3 = lo0
        k0 = (k0 + _W0) & _MASK32
        k1 = (k1 + _W1) & _MASK32
    return c0


def _ising_gibbs_philox_core(spins, accept_table, num_sweeps, seed_hi, seed_lo):
    """核心计算：全整数棋盘 Gibbs 更新，返回 int8 spins_out。"""
    Bsz, H, W = spins.shape
    if H % 2 != 0 or W % 2 != 0:
        raise ValueError(f"H and W must be even for checkerboard 2-coloring, got {H}x{W}")
    dev = spins.device
    seed_hi = int(seed_hi) & _MASK32
    seed_lo = int(seed_lo) & _MASK32

    s_flat = spins.to(torch.int64).reshape(Bsz, H * W)      # [B, HW]，值 ∈ {0, 1}
    table = accept_table.to(torch.int64).reshape(10)        # 展平 [2, 5] → spin*5 + k

    ii = torch.arange(H, dtype=torch.int64, device=dev).view(H, 1).expand(H, W)
    jj = torch.arange(W, dtype=torch.int64, device=dev).view(1, W).expand(H, W)
    site = (ii * W + jj).reshape(H * W)                     # 计数器字 c3 = i·W + j
    # 周期边界四邻居的扁平下标 [HW, 4]（上、下、左、右）
    nb = torch.stack([
        ((ii - 1) % H) * W + jj,
        ((ii + 1) % H) * W + jj,
        ii * W + (jj - 1) % W,
        ii * W + (jj + 1) % W,
    ], dim=-1).reshape(H * W, 4)

    color = ((ii + jj) % 2).reshape(H * W)
    b_idx = torch.arange(Bsz, dtype=torch.int64, device=dev).view(Bsz, 1)

    per_color = []
    for c in (0, 1):
        sites_c = torch.nonzero(color == c, as_tuple=False).reshape(-1)   # [HW/2]
        per_color.append((sites_c, nb[sites_c], site[sites_c]))

    for s in range(int(num_sweeps)):
        for c in (0, 1):
            sites_c, nb_c, ctr3 = per_color[c]
            sigma = 2 * s_flat - 1                                        # [B, HW]
            nsum = sigma[:, nb_c].sum(dim=-1)                             # [B, HW/2]
            k = (nsum + 4) >> 1                                           # ∈ {0..4}
            cur = s_flat[:, sites_c]                                      # [B, HW/2]
            thr = table[cur * 5 + k]                                      # [B, HW/2]
            r = _philox4x32_10_x0(
                torch.tensor(s, dtype=torch.int64, device=dev),
                torch.tensor(c, dtype=torch.int64, device=dev),
                b_idx, ctr3.unsqueeze(0), seed_hi, seed_lo)               # [B, HW/2]
            s_flat[:, sites_c] = torch.where(r < thr, 1 - cur, cur)
    return s_flat.reshape(Bsz, H, W).to(torch.int8)


def ising_gibbs_philox(
    spins: torch.Tensor,
    accept_table: torch.Tensor,
    num_sweeps: int,
    seed_hi: int,
    seed_lo: int,
) -> torch.Tensor:
    """
    棋盘 Gibbs + Philox4x32-10 golden reference（全整数精确运算，输出零容差）

    Args:
        spins: [B, H, W] int8，取值 {0, 1}（0 代表自旋 −1，1 代表 +1）；H、W 均为偶数
        accept_table: [2, 5] int64 阈值表，取值 [0, 2^32−1]，行 = 当前 spin，列 = 邻居和
            索引 k = (nsum+4)/2；任意随机表均合法（定义一个确定性元胞自动机）
        num_sweeps: sweep 次数（一个 sweep = 黑格相 + 白格相）
        seed_hi: Philox key 高 32 位字（按无符号 32 位解释）
        seed_lo: Philox key 低 32 位字（按无符号 32 位解释）

    Returns:
        spins_out: [B, H, W] int8，num_sweeps 个 sweep 后的自旋场（零容差精确比对）
    """
    return _ising_gibbs_philox_core(spins, accept_table, num_sweeps, seed_hi, seed_lo)


def ising_gibbs_philox_oracle(
    spins: torch.Tensor,
    accept_table: torch.Tensor,
    num_sweeps: int,
    seed_hi: int,
    seed_lo: int,
) -> torch.Tensor:
    """Oracle (g)：整数域精确运算，与 plain golden 完全一致，直接复用核心。"""
    return _ising_gibbs_philox_core(spins, accept_table, num_sweeps, seed_hi, seed_lo)
```

## 6. 额外信息

### 算子调用示例

```python
import torch

B, H, W = 4, 256, 256

spins = torch.randint(0, 2, (B, H, W), dtype=torch.int8, device="npu")
accept_table = torch.randint(0, 1 << 32, (2, 5), dtype=torch.int64, device="npu")

spins_out = ising_gibbs_philox(spins, accept_table, num_sweeps=16,
                               seed_hi=0x12345678, seed_lo=0x9ABCDEF0)
# spins_out.shape: [B, H, W] int8，取值 {0, 1}
```

### 可用于自检的性质（均已数值验证）

- accept_table 全 0 → 任何输入下永不翻转，spins_out == spins
- accept_table 全 2^32−1 → 每次更新以概率 (2^32−1)/2^32 翻转（随机输入下一个相内几乎全翻）
- 同一颜色相内更新顺序任意：逐点串行（任意顺序、含立即写入）与全并行结果逐 bit 相同
- 相同 (spins, table, num_sweeps, seeds) 两次调用结果逐 bit 相同（无状态、可复现）
- §2 的三条 Philox 测试向量

### 参考文献

- Salmon, J. K., Moraes, M. A., Dror, R. O., Shaw, D. E. (2011). "Parallel Random Numbers: As Easy as 1, 2, 3". SC 2011（Philox / Random123 计数器 RNG 的来源，本算子 §2 规范与其参考实现逐 bit 一致）
- Preis, T., Virnau, P., Paul, W., Schneider, J. J. (2009). "GPU accelerated Monte Carlo simulation of the 2D and 3D Ising model". Journal of Computational Physics 228(12)（棋盘分解并行 Ising 模拟的经典方案）
- Geman, S. & Geman, D. (1984). "Stochastic Relaxation, Gibbs Distributions, and the Bayesian Restoration of Images". IEEE TPAMI 6(6)（Gibbs 采样的来源）
- cuRAND / JAX 的并行 RNG 均采用 Philox 谱系（计数器 RNG 作为加速器随机模拟标准原语的背景）
