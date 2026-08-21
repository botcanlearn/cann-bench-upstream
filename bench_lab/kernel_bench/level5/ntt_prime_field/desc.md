# NttPrimeField 算子 API 描述

## 1. 算子简介

素域 Z_p 上的批量数论变换（Number Theoretic Transform，NTT）算子：以模 p 整数运算在有限域上执行离散傅里叶变换。NTT 是零知识证明（zk-STARK 的 FRI 承诺、zk-SNARK 的多项式运算）、后量子格密码（多项式环乘法）与全同态加密的核心计算原语——这些负载正在与 LLM 基础设施汇合（可验证推理、隐私推理），UniNTT（ASPLOS 2025）与 CGO 2025 的 GPU 多字模算术代码生成工作都以它为中心目标。

与浮点算子本质不同的两个约束：

1. **整数域零容差**：结果由模运算唯一确定，输出逐 bit 精确比对（int32 阈值 0），一 bit 错即失败，不存在浮点误差的容忍空间。
2. **无原生指令，必须换表示**：AI 加速器的向量/矩阵单元围绕浮点设计，没有模乘/模加指令；p ≈ 2^31 时乘积达 62 bit，任何浮点近似路径的尾数都不够。素域运算必须以现有整数乘/加/移位原语组合实现，通常需要把操作数换到便于约减的表示（§2 数学参考给出 Montgomery 域的完整定义）。

**主要应用场景**：
- zk-STARK / zk-SNARK 证明系统的多项式承诺与求值（BabyBear 域 2013265921 为 RISC Zero、Plonky3 等 zkVM 的默认域）
- 后量子格密码与同态加密中的负循环/循环卷积（多项式乘法 = NTT → 逐点乘 → 逆 NTT）
- 竞赛与通用大整数/多项式库的精确卷积（998244353 等经典 NTT 素数）

**算子特征**：
- 难度等级：L5（FusedComposite）
- 单输入 x [B, N] int32 + 两属性（modulus、inverse），单输出 y [B, N] int32
- 全程模 p 整数运算，输出零容差；前向与逆变换共用同一蝶形结构
- N 为 2 的幂且 N | (p−1)，五种内置 NTT 友好素数（均 < 2^31）

**为何是 L5**：
- **性能墙，而非数学墙**：本文档给出完整数学定义与全部所需恒等式。按定义式逐点求和是 O(N²)，N = 2^20 时慢四个数量级；即便走 O(N log N) 蝶形，若每次模乘都用整数除法取模，吞吐也差一个数量级。高效实现需要蝶形数据流 + 高效模约减（§2 给出 Montgomery 约减的数学定义；Barrett 等任何逐 bit 等价的方法均可接受）。蝶形各阶段的访存跨度按 2 的幂指数变化（相邻 → N/2 间隔），大 N 单批与小 N 大批两种形态的多核切分与数据重排是主要工程难点。
- **零容差下的范围推理**：中间值何时约减、何处会溢出（p² < 2^62、和式逼近 2^63）必须严格推理正确——错一个边界就是逐 bit 失败，且大规模随机输入下必然触发。
- **隐藏测试集**：评测含未公开用例（全部五种 modulus、退化 N = 1/2/4、全 0 / 全 p−1 极值、N = 2^22 大规模、大 batch 小 N 等），实现必须以本规格为准，而非以可见用例为准。

## 2. 算子定义

### 数学公式

设素数 p，g 为 p 的原根，N 为 2 的幂且 N | (p−1)。主 N 次单位根：

$$
\omega = g^{(p-1)/N} \bmod p
$$

**前向变换**（inverse = False）：

$$
y_j = \sum_{i=0}^{N-1} x_i \cdot \omega^{ij} \bmod p, \qquad 0 \le j < N
$$

**逆变换**（inverse = True，N^{-1} 为 N 的模逆）：

$$
y_j = N^{-1} \cdot \sum_{i=0}^{N-1} x_i \cdot \omega^{-ij} \bmod p
$$

输入输出均为**自然顺序**（不要求位逆序）。N = 1 时 y = x（前向与逆变换均如此）。

### 原根表（算子契约的一部分）

ω 由下表唯一确定，输出必须与该 ω 的变换逐 bit 一致：

| modulus p | 分解 | 原根 g | 最大 N |
|-----------|------|--------|--------|
| 2013265921（BabyBear，默认） | 15·2^27 + 1 | 31 | 2^27 |
| 998244353 | 119·2^23 + 1 | 3 | 2^23 |
| 167772161 | 5·2^25 + 1 | 3 | 2^25 |
| 469762049 | 7·2^26 + 1 | 3 | 2^26 |
| 754974721 | 45·2^24 + 1 | 11 | 2^24 |

### 可用于自检的性质

- 逆变换与前向互逆：inverse(forward(x)) = x
- 全 1 输入的前向变换 = N·δ_0（y_0 = N mod p，其余为 0）；逆变换 = δ_0
- 对模加法线性：NTT(a·x + b·z) = a·NTT(x) + b·NTT(z)（mod p）

### 数学参考（以下恒等式均已数值验证；任何逐 bit 等价的实现路径均可接受）

**(a) radix-2 Cooley–Tukey 分解**（mod p 恒等式）。记 E、O 为偶/奇下标子序列的 N/2 点 NTT（单位根 ω²），则

$$
X_j = E_j + \omega^j O_j \bmod p, \qquad X_{j+N/2} = E_j - \omega^j O_j \bmod p \qquad (0 \le j < N/2)
$$

其中用到 $\omega^{N/2} \equiv -1 \pmod p$。迭代形式：先做位逆序置换（rev(i) = i 的 log2(N) 位二进制翻转），再做 log2(N) 个阶段；阶段 s（块大小 m = 2^s，块内下标 j ∈ [0, m/2)，$\omega_m = \omega^{N/m}$）的蝶形为

$$
u' = (u + \omega_m^j \cdot w) \bmod p, \qquad w' = (u - \omega_m^j \cdot w) \bmod p
$$

全部阶段完成后输出即自然顺序。

**(b) Montgomery 约减**（REDC，R = 2^32；已对全部五种 p 数值验证）。预计算 $p' = -p^{-1} \bmod R$。对 $0 \le T < R \cdot p$：

$$
m = (T \cdot p') \bmod R, \qquad t = \frac{T + m \cdot p}{R} \ (\text{该除法整除}), \qquad t \ge p \text{ 时 } t \leftarrow t - p
$$

则 $t = T \cdot R^{-1} \bmod p$ 且 $0 \le t < p$。范围：条件减前 $t < 2p$；当 T 为两个 [0, p) 值之积时 $T + m \cdot p < p^2 + R \cdot p < 2^{64}$（超出带符号 64 位范围，按无符号 64 位或拆分处理）。域映射：$\bar{a} = a \cdot R \bmod p$（进域可用 $\text{REDC}(a \cdot (R^2 \bmod p))$，$R^2 \bmod p$ 预计算）；域内乘法闭合 $\text{REDC}(\bar{a} \cdot \bar{b}) = (a \cdot b) \cdot R \bmod p$；出域 $\text{REDC}(\bar{a}) = a$。

**(c) 模逆（费马小定理）**：$\omega^{-1} = \omega^{p-2} \bmod p$，$N^{-1} = N^{p-2} \bmod p$。

## 3. 接口规范

### 算子原型

```python
ntt_prime_field(Tensor x, int modulus=2013265921, bool inverse=False) -> Tensor y
```

### 输入参数说明

| 参数 | 类型 | 必需 | dtype | shape | 描述 |
|------|------|------|-------|-------|------|
| x | Tensor | 是 | int32 | [B, N] | 批量输入序列，取值 [0, modulus−1]；N 为 2 的幂且 N \| (modulus−1) |
| modulus | int | 否（默认 2013265921） | - | 标量 | NTT 友好素数 p，取值限 §2 原根表中的五种 |
| inverse | bool | 否（默认 False） | - | 标量 | True 时执行逆变换（结果乘 N^{-1} mod p） |

### 输出

| 名称 | dtype | shape | 描述 |
|------|-------|-------|------|
| y | int32 | [B, N] | 变换结果，取值 [0, modulus−1]，自然顺序 |

### 数据类型

| x dtype | y dtype | 内部计算 |
|---------|---------|----------|
| int32 | int32 | 整数模运算（golden 为 int64 + 每次乘法后取模；kernel 任何逐 bit 等价路径均可） |

### 规则与约束

- N 为 2 的幂（含 N = 1）且 N | (modulus−1)；由 cases 保证，算子不对违反此约定的输入负责
- x 取值范围 [0, modulus−1]（由 cases 的 value_range 保证）
- modulus 限 §2 原根表的五种素数；ω 由原根表唯一确定
- 输出取值范围 [0, modulus−1]，自然顺序（不要求位逆序）
- 批维 B 内各行相互独立
- 结果零容差：与数学定义逐 bit 一致（见 §4）

### 支持范围

| 维度 / 参数 | 范围 | 备注 |
|---|---|---|
| `B`（batch） | 1 ~ 4096 | cases.csv 实测 1 ~ 4096 |
| `N`（变换长度） | 1 ~ 2^22（2 的幂） | cases.csv 实测 1 ~ 2^22 |
| `B · N` | ≤ 2^24 | 控制单 case 内存 |
| `modulus` | 五种（§2 原根表） | 公开集以 BabyBear 为主 |
| `inverse` | false / true | 两种均覆盖 |
| `x` 取值 | [0, modulus−1] | 上界随 modulus 变化；特殊值 case 用 [0,0] / [1,1] 等子区间 |

## 4. 精度要求

**零容差**：输出为 int32，精度阈值为 0——评测框架对整数输出走精确比对路径（torch.equal 逐元素完全相等），**一 bit 错即失败**。不适用 MERE/MARE 浮点误差指标，无小值域/相消兜底。

golden 为整数精确实现（int64 中间精度 + 每次乘法后取模，全程无浮点），plain golden 与 oracle 结果完全一致；评测的 fp64 golden 路径不改变整数输入（框架仅对浮点输入升精度），因此参考结果与本规格的数学定义逐 bit 相同。

## 5. 标准 Golden 代码

```python
import torch

# 内置素数 → 原根表（均满足 p < 2^31，p-1 含大 2-幂因子）
_PRIMITIVE_ROOTS = {
    2013265921: 31,   # BabyBear: 15 · 2^27 + 1，zk-STARK 常用域
    998244353: 3,     # 119 · 2^23 + 1，竞赛/多项式乘法常用
    167772161: 3,     # 5 · 2^25 + 1
    469762049: 3,     # 7 · 2^26 + 1
    754974721: 11,    # 45 · 2^24 + 1
}


def _ntt_prime_field_core(x, modulus, inverse):
    """核心计算：向量化迭代 radix-2 Cooley–Tukey（DIT），int64 模运算。"""
    p = int(modulus)
    if p not in _PRIMITIVE_ROOTS:
        raise ValueError(f"unsupported modulus {p}, expected one of {sorted(_PRIMITIVE_ROOTS)}")
    Bsz, N = x.shape
    if N & (N - 1) != 0:
        raise ValueError(f"N must be a power of two, got {N}")
    if (p - 1) % N != 0:
        raise ValueError(f"N={N} does not divide p-1={p - 1}")

    logn = N.bit_length() - 1
    v = x.to(torch.int64)

    # 主 N 次单位根 ω = g^((p-1)/N)；逆变换用 ω^{-1} = ω^{p-2}（费马小定理）
    omega = pow(_PRIMITIVE_ROOTS[p], (p - 1) // N, p)
    if inverse:
        omega = pow(omega, p - 2, p)

    if N > 1:
        idx = torch.arange(N, dtype=torch.int64, device=x.device)

        # 位逆序置换（DIT 输入序），log2(N) 次向量化位操作
        rev = torch.zeros_like(idx)
        for i in range(logn):
            rev = (rev << 1) | ((idx >> i) & 1)
        v = v[:, rev]

        # 全局 twiddle 表 tw[k] = ω^k (0 ≤ k < N/2)，向量化模幂（对指数位循环）
        e = idx[: N // 2]
        tw = torch.ones(N // 2, dtype=torch.int64, device=x.device)
        base = omega % p
        for i in range(logn):
            tw = torch.where((e >> i) & 1 == 1, (tw * base) % p, tw)
            base = (base * base) % p

        # log2(N) 个蝶形阶段：阶段 s 的块大小 m = 2^s，块内 twiddle 为 (ω^{N/m})^k = tw[k·N/m]
        for s in range(1, logn + 1):
            m = 1 << s
            half = m >> 1
            w = tw[:: N // m][:half]                        # [half]
            blocks = v.reshape(Bsz, N // m, m)
            u = blocks[:, :, :half]                         # [B, N/m, half]
            t = (blocks[:, :, half:] * w) % p               # 乘积 < p^2 < 2^62，int64 无溢出
            v = torch.cat([(u + t) % p, (u - t) % p], dim=-1).reshape(Bsz, N)

    if inverse:
        n_inv = pow(N, p - 2, p)                            # N^{-1} mod p
        v = (v * n_inv) % p

    return v.to(torch.int32)


def ntt_prime_field(
    x: torch.Tensor,
    modulus: int = 2013265921,
    inverse: bool = False,
) -> torch.Tensor:
    """
    素域 NTT golden reference（整数精确运算，零容差）

    Args:
        x: [B, N] int32 输入，取值 [0, modulus-1]；N 为 2 的幂且 N | (modulus-1)
        modulus: NTT 友好素数 p（内置原根表支持 2013265921 / 998244353 /
            167772161 / 469762049 / 754974721，均 < 2^31），默认 BabyBear
        inverse: False 为前向变换，True 为逆变换（结果乘 N^{-1} mod p）

    Returns:
        y: [B, N] int32，取值 [0, modulus-1]，自然顺序
    """
    return _ntt_prime_field_core(x, modulus, inverse)


def ntt_prime_field_oracle(
    x: torch.Tensor,
    modulus: int = 2013265921,
    inverse: bool = False,
) -> torch.Tensor:
    """Oracle (g)：整数域精确运算，与 plain golden 完全一致，直接复用核心。"""
    return _ntt_prime_field_core(x, modulus, inverse)
```

## 6. 额外信息

### 算子调用示例

```python
import torch

B, N = 4, 1 << 16
p = 2013265921  # BabyBear

x = torch.randint(0, p, (B, N), dtype=torch.int32, device="npu")

# 多项式乘法的典型用法：前向 NTT → 逐点模乘 → 逆 NTT
xf = ntt_prime_field(x, modulus=p, inverse=False)
# ... 逐点模乘 ...
xr = ntt_prime_field(xf, modulus=p, inverse=True)
# torch.equal(xr, x) == True（互逆，零误差）
```

### 参考文献

- Cooley, J. W. & Tukey, J. W. (1965). "An Algorithm for the Machine Calculation of Complex Fourier Series". Mathematics of Computation 19(90)（蝶形分解的来源）
- Montgomery, P. L. (1985). "Modular Multiplication Without Trial Division". Mathematics of Computation 44(170)（§2 数学参考 (b) 的出处）
- UniNTT（ASPLOS 2025）：NTT 在多类硬件上的统一加速研究（本算子的体系结构背景）
- CGO 2025：GPU 多字模算术的代码生成（模算术在浮点向量硬件上的表示变换背景）
- BabyBear 域：RISC Zero / Plonky3 等 zk-STARK 系统的默认素域
