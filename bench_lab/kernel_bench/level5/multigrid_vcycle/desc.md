# MultigridVcycle 算子 API 描述

## 1. 算子简介

3D Poisson 方程 $-\Delta u = f$（零 Dirichlet 边界）的**一次几何多重网格 V-cycle**。几何多重网格是椭圆型方程渐近最优（$O(N_{\text{总点数}})$）的求解器：光滑误差分量由细网格上的红黑 Gauss–Seidel 平滑消去，低频误差分量限制（restriction）到粗网格上递归求解后再延拓（prolongation）回来修正。一次 V-cycle 把残差压缩一个固定倍数（教科书值 < 0.3，见 §6 实测表），外层迭代若干次即收敛到离散解。

Poisson 求解出现在几乎所有不可压缩流体模拟（压力投影步，每个时间步解一次）、等离子体/半导体器件的静电场、自重力天体模拟、扩散型隐式时间步进中；V-cycle 是这些负载的单步核心 kernel，其吞吐直接决定整个模拟的时间步速率。本算子把一次完整 V-cycle（含全部层级与层间转移）定义为单个融合 kernel。

**主要应用场景**：
- 不可压缩 Navier–Stokes 求解器的压力 Poisson 投影步（CFD、图形学流体）
- 等离子体 PIC / 半导体器件模拟的静电 Poisson 场
- 自重力 N 体/流体天体物理模拟的引力势求解
- 扩散/热传导隐式格式的每步线性求解

**算子特征**：
- 难度等级：L5（NumericalStable），机制轴 = 多分辨率递归，且**红黑序即规格**
- 双输入（u0, f）+ 四 attr（num_levels, pre_smooth, post_smooth, h），单输出
- 融合红黑 Gauss–Seidel 平滑、残差、full-weighting 限制、三线性延拓与跨层递归
- 平滑的并行语义由棋盘二染色精确定义（同 ising_gibbs_philox 的棋盘先例）：同色点互不相邻 ⇒ 同色内并行顺序无关，**红先黑后的两遍扫描就是规格本身**，不存在"串行 GS 与并行实现结果不同"的歧义

**为何是 L5**（约为 L4 NumericalStable 的 2 倍难度）：
- **正确 ≠ 快**：按 §2 公式逐层直译（每个平滑 sweep 读写全场、每次层间转移读写全场）即可得到完全正确的结果并通过全部精度用例，但性能只能锚定朴素 baseline（预期得分 ~0.5 档）。性能墙在多分辨率结构本身：相邻层网格体积差 8 倍，细层（129³ 量级）是带宽受限的 stencil 扫描、粗层（5³~17³）小到可整块驻留片上而完全延迟受限，朴素实现让每一层、每一个 sweep 都独立往返 HBM——细层往返被 (pre+post) 个 sweep 与残差/转移各自放大，粗层则被启动开销支配。墙 = 红黑两遍扫的访存交错与融合（红/黑半场更新的交错布局）、平滑-残差-限制的同遍融合、粗层整链片上常驻（此处仅点名领域算法概念，实现方案不限）
- **实现规格而非过可见测试**：隐藏评测集覆盖公开用例之外的规格维度（levels 与 N 的全部合法组合、pre/post 非对称、h 极值、f 全 0 / u0 全 0 / 双零不动点、大幅值、素数 batch 等），逐点定义中每一处约定（红黑的全局索引奇偶、限制核对齐到偶索引、最粗层 20 次平滑、num_levels = 粗化次数、边界置 0 时机）都会被单独检验
- **递归结构不规则**：每层网格尺寸 $(N-1)/2^\ell + 1$ 逐层减半、h 逐层翻倍，5 个层级 × 4 种网格操作的组合中任何一层的对齐/权重/顺序写错，误差都会经延拓污染全场并被隐藏用例捕获
- **数值语义严格**：红黑序是两遍有依赖的扫描（黑点读到更新后的红点），实测把红黑写反 MERE = 3.8、换成 Jacobi MERE = 15（阈值 0.002，判别相隔 3 个数量级）——平滑器的精确语义无法靠阈值宽容混过

## 2. 算子定义

网格：$[B, N, N, N]$，$N = 2^k + 1$（含边界点的内点语义）。$u$ 的最外层在所有操作后保持 0；置零是算子语义的一部分：进入 V-cycle 前先将 u0 的最外层置 0（任意随机输入合法），$f$ 的边界值不参与计算。每个 batch 独立。

### 离散算子

$$
(A u)[i,j,k] = \frac{6\,u[i,j,k] - \sum_{\text{六邻居}} u}{h^2} \qquad \text{（仅内点 } 1 \le i,j,k \le N-2\text{）}
$$

### 平滑：红黑 Gauss–Seidel（红黑序即规格）

红点 = 全局 0-based 索引 $(i+j+k)$ 为**偶**的内点，黑点 = 奇。一个 sweep：

1. 先更新**全部红点**：$u[i,j,k] \leftarrow \big( h^2 f[i,j,k] + \sum_{\text{六邻居}} u \big) / 6$
2. 再更新**全部黑点**（读到的红邻居是第 1 步更新后的值）

六邻居与中心点异色，故同色内所有点的更新互不依赖——**同色内并行顺序无关，红先黑后的两遍扫描就是本算子的精确规格**（并行顺序即规格，参照 ising_gibbs_philox 的棋盘先例）。

### 残差

$$
r = f - A u \quad \text{（内点）}, \qquad r[\text{边界}] = 0
$$

### 限制（fine → coarse，full-weighting 27 点核）

粗网格 $N_c = (N-1)/2 + 1$，粗点 $(I,J,K)$ 对应细点 $(2I, 2J, 2K)$（偶索引对齐），粗层间距 $h_c = 2h$：

$$
r_c[I,J,K] = \frac{1}{64} \sum_{a,b,c \in \{-1,0,1\}} w_a w_b w_c\; r[2I\!+\!a,\, 2J\!+\!b,\, 2K\!+\!c],
\qquad (w_{-1}, w_0, w_{+1}) = (1, 2, 1)
$$

即 $\frac{1}{64}[1,2,1] \otimes [1,2,1] \otimes [1,2,1]$（权重和为 1）：中心 $8/64$、面邻 $4/64$、棱邻 $2/64$、角邻 $1/64$。只对粗内点计算，粗边界为 0。

### 延拓（coarse → fine，三线性插值）

细点按三个索引的奇偶分 8 种情形（$I = \lfloor i/2 \rfloor$ 等）：

| 细点奇偶 (i,j,k) | 插值 |
|---|---|
| (偶,偶,偶) | $e_c[I,J,K]$（直接注入） |
| (奇,偶,偶) | $\tfrac{1}{2}(e_c[I,J,K] + e_c[I\!+\!1,J,K])$ |
| (偶,奇,偶) | $\tfrac{1}{2}(e_c[I,J,K] + e_c[I,J\!+\!1,K])$ |
| (偶,偶,奇) | $\tfrac{1}{2}(e_c[I,J,K] + e_c[I,J,K\!+\!1])$ |
| (奇,奇,偶) | $\tfrac{1}{4}\big(e_c[I,J,K] + e_c[I,J\!+\!1,K] + e_c[I\!+\!1,J,K] + e_c[I\!+\!1,J\!+\!1,K]\big)$ |
| (奇,偶,奇) | $\tfrac{1}{4}\big(e_c[I,J,K] + e_c[I,J,K\!+\!1] + e_c[I\!+\!1,J,K] + e_c[I\!+\!1,J,K\!+\!1]\big)$ |
| (偶,奇,奇) | $\tfrac{1}{4}\big(e_c[I,J,K] + e_c[I,J,K\!+\!1] + e_c[I,J\!+\!1,K] + e_c[I,J\!+\!1,K\!+\!1]\big)$ |
| (奇,奇,奇) | $\tfrac{1}{8}\big($ 8 个粗角点之和 $\big)$ |

粗边界为 0 ⇒ 细边界自动保持 0。**伴随关系（已数值验证）**：$\langle R\,r,\, e \rangle = \tfrac{1}{8}\langle r,\, P\,e \rangle$（ℓ2 内积，fp64 实测相对偏差 2.4e-15）；配网格加权内积 $\langle x, y\rangle_h = h^3 \sum x y$（$h_c^3 = 8 h^3$）则 $R$ 与 $P$ 严格互为伴随——这是多重网格收敛理论（Galerkin 性质）成立的前提。

### V-cycle 递归

```
vcycle(u, f, h, levels_left):
    若 levels_left == 0:                          # 最粗层
        对 (u, f, h) 做 20 次红黑平滑，返回 u      # 以 20 次平滑代替直接解
    u ← 红黑平滑 pre_smooth 次
    r ← f − A u
    r_c ← R(r)                                    # 限制到粗网格
    e_c ← vcycle(0, r_c, 2h, levels_left − 1)     # 误差方程，初值 0
    u ← u + P(e_c)                                # 延拓修正
    u ← 红黑平滑 post_smooth 次
    返回 u
```

顶层调用 `vcycle(u0, f, h, num_levels)`。**num_levels = 粗化（限制）次数**，网格层数 = num_levels + 1；case 保证最粗层 ≥ 5³（$(N-1)/2^{\text{num\_levels}} \ge 4$），故合法组合为 N=33 → levels ≤ 3、N=65 → levels ≤ 4、N=129 → levels ≤ 5。

**公式验证结果**（全部先于本文档写定完成，fp64）：
- 独立逐点标量实现（9³ 1 粗化 1+1、17³ 2 粗化 2+2）与向量化 golden **逐位一致**
- V-cycle 不动点 == 稠密直接解（N=17，随机 f，30 次迭代后最大偏差 1.7e-18）
- 解析验证：$u^\* = \sin(\pi x)\sin(\pi y)\sin(\pi z)$、$f = 3\pi^2 u^\*$、$h = 1/(N-1)$，迭代收敛后 $\max|u - u^\*| = 8.0\text{e-}4 = O(h^2)$ 离散化误差（N=33），残差降至 3e-11
- R/P 伴随关系至舍入（见上）
- 收敛率实测见 §6：深层级配置 < 0.3（教科书值，Briggs et al. 2000），全部配置残差单调下降

## 3. 接口规范

### 算子原型

```python
multigrid_vcycle(Tensor u0, Tensor f, int num_levels, int pre_smooth, int post_smooth, float h) -> Tensor u_out
```

### 输入参数说明

| 参数 | 类型 | 必需 | dtype | shape | 描述 |
|------|------|------|-------|-------|------|
| u0 | Tensor | 是 | float32 | [B, N, N, N] | 初始猜测，最外层由算子置 0（N = 2^k + 1） |
| f | Tensor | 是 | float32 | [B, N, N, N] | 右端项，边界值不参与计算（评测取值范围 [-1, 1]） |
| num_levels | int | 是 | - | 标量 | 粗化次数（评测取值 2 ~ 5；case 保证最粗层 ≥ 5³） |
| pre_smooth | int | 是 | - | 标量 | 每层延拓前的红黑 sweep 数（评测取值 1 ~ 4） |
| post_smooth | int | 是 | - | 标量 | 每层延拓修正后的红黑 sweep 数（评测取值 1 ~ 4） |
| h | float | 是 | - | 标量 | 最细层网格间距（评测取值范围 [0.01, 1.0]） |

### 输出

| 名称 | dtype | shape | 描述 |
|------|-------|-------|------|
| u_out | float32 | [B, N, N, N] | 一次 V-cycle 后的解（最外层为 0） |

### 数据类型

| u0/f dtype | 输出 dtype | 内部计算 |
|---|---|---|
| float32 | float32 | fp32 |

### 规则与约束

- 两个 Tensor 输入 dtype 必须一致（float32），shape 完全一致
- $N = 2^k + 1$（内点语义，含边界点；由 cases 保证）
- num_levels 满足 $(N-1)/2^{\text{num\_levels}} \ge 4$（最粗层 ≥ 5³，由 cases 保证）
- u0 最外层置 0 是算子语义（第一步）；输出最外层为 0
- 红黑序是规格：红 = 全局 0-based 索引 $(i+j+k)$ 偶，先红后黑；最粗层固定 20 次红黑平滑
- h 极小时残差中间量放大 $1/h^2$（h=0.01 时 ~1e4 倍），fp32 值域内安全（已实测）
- 输出须为 contiguous 张量

### 支持范围

| 维度 / 参数 | 范围 | 备注 |
|---|---|---|
| `B`（batch） | 1 ~ 8 | cases.csv 实测 1 ~ 8 |
| `N`（每维网格点数） | 33 / 65 / 129 | $2^k+1$，cases.csv 实测三档均覆盖 |
| `num_levels` | 2 ~ 5 | 与 N 的合法组合见 §2 |
| `pre_smooth` / `post_smooth` | 1 ~ 4 | 含非对称组合 |
| `h` | [0.01, 1.0] | attr |
| `u0` / `f` 取值 | [-1, 1] | 常规随机范围；隐藏用例含全 0、[-100, 100] 等 |
| dtype | float32 | 全部用例 |

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

V-cycle 链 = $(pre+post) \times (\text{num\_levels}+1)$ 次红黑平滑 + 最粗层 20 次平滑 + 层间转移，fp32 误差沿红黑两遍扫与递归累积；解在过零点附近存在相对误差尖峰（由评测框架小值域/相消兜底标准处理，native 参考 = plain golden）。阈值经评测框架 checker 实测确定：独立 fp32 路径（求和顺序不同的实现）vs fp64 oracle 在 12 组配置（N/levels/smooth/h/值域全覆盖）× 3~6 draw 上校准，checker 全量判 passed，实测 MERE ≤ 5.7e-6、MARE 尖峰 ≤ 1.8e-2：

| 数据类型 | FLOAT32 |
|----------|---------|
| **通过阈值(Threshold)** | 0.002 |

当平均相对误差 MERE < Threshold，最大相对误差 MARE < 10 * Threshold 时判定为通过。MERE 裕量 ≥ 350x；判别力实测：红黑写反（黑先红后）MERE = 3.8、Jacobi 平滑 MERE = 15、injection 限制 MERE = 83，与阈值相隔 3 个数量级以上——阈值放宽不损失判别力。

## 5. 标准 Golden 代码

```python
import torch

# 27 点 full-weighting 权重（[1,2,1] 张量积 / 64），以偏移 (a,b,c) ∈ {-1,0,1}³ 索引
_FW_W1 = (1.0, 2.0, 1.0)


def _redblack_masks(n, device):
    """内点红/黑掩码（红 = 全局 0-based 索引 i+j+k 偶），shape [n-2, n-2, n-2]。"""
    idx = torch.arange(1, n - 1, device=device)
    par = (idx.view(-1, 1, 1) + idx.view(1, -1, 1) + idx.view(1, 1, -1)) % 2
    red = par == 0
    return red, ~red


def _smooth_redblack(u, f, h2, sweeps, masks):
    """红黑 Gauss–Seidel：每个 sweep 先全部红点再全部黑点（就地更新 u 内点）。"""
    red, black = masks
    for _ in range(sweeps):
        for mask in (red, black):
            nb = (u[:, :-2, 1:-1, 1:-1] + u[:, 2:, 1:-1, 1:-1]
                  + u[:, 1:-1, :-2, 1:-1] + u[:, 1:-1, 2:, 1:-1]
                  + u[:, 1:-1, 1:-1, :-2] + u[:, 1:-1, 1:-1, 2:])
            upd = (h2 * f[:, 1:-1, 1:-1, 1:-1] + nb) / 6.0
            inner = u[:, 1:-1, 1:-1, 1:-1]
            u[:, 1:-1, 1:-1, 1:-1] = torch.where(mask, upd, inner)
    return u


def _residual(u, f, h2):
    """r = f − A u（内点），边界置 0。"""
    r = torch.zeros_like(u)
    nb = (u[:, :-2, 1:-1, 1:-1] + u[:, 2:, 1:-1, 1:-1]
          + u[:, 1:-1, :-2, 1:-1] + u[:, 1:-1, 2:, 1:-1]
          + u[:, 1:-1, 1:-1, :-2] + u[:, 1:-1, 1:-1, 2:])
    au = (6.0 * u[:, 1:-1, 1:-1, 1:-1] - nb) / h2
    r[:, 1:-1, 1:-1, 1:-1] = f[:, 1:-1, 1:-1, 1:-1] - au
    return r


def _restrict_fw(r):
    """full-weighting 限制：粗内点 (I,J,K) ← (1/64) Σ w_a w_b w_c * r[2I+a, 2J+b, 2K+c]。"""
    n = r.shape[-1]
    nc = (n - 1) // 2 + 1
    rc = torch.zeros(r.shape[0], nc, nc, nc, dtype=r.dtype, device=r.device)
    acc = torch.zeros(r.shape[0], nc - 2, nc - 2, nc - 2, dtype=r.dtype, device=r.device)
    # 粗内点 I=1..nc-2 对应细点 2I=2..n-3；偏移窗口切片 [2+a : n-3+a+1 : 2]
    for a in (-1, 0, 1):
        for b in (-1, 0, 1):
            for c in (-1, 0, 1):
                w = _FW_W1[a + 1] * _FW_W1[b + 1] * _FW_W1[c + 1]
                acc = acc + w * r[:, 2 + a: n - 2 + a: 2,
                                  2 + b: n - 2 + b: 2,
                                  2 + c: n - 2 + c: 2]
    rc[:, 1:-1, 1:-1, 1:-1] = acc / 64.0
    return rc


def _prolong_trilinear(ec, n):
    """三线性延拓：粗 [B,nc,nc,nc] → 细 [B,n,n,n]（粗边界为 0 ⇒ 细边界为 0）。"""
    ef = torch.zeros(ec.shape[0], n, n, n, dtype=ec.dtype, device=ec.device)
    # 三偶：直接注入
    ef[:, ::2, ::2, ::2] = ec
    # 单奇（3 种）：沿奇方向两点平均
    ef[:, 1::2, ::2, ::2] = 0.5 * (ec[:, :-1, :, :] + ec[:, 1:, :, :])
    ef[:, ::2, 1::2, ::2] = 0.5 * (ec[:, :, :-1, :] + ec[:, :, 1:, :])
    ef[:, ::2, ::2, 1::2] = 0.5 * (ec[:, :, :, :-1] + ec[:, :, :, 1:])
    # 双奇（3 种）：面上四点平均
    ef[:, 1::2, 1::2, ::2] = 0.25 * (ec[:, :-1, :-1, :] + ec[:, :-1, 1:, :]
                                     + ec[:, 1:, :-1, :] + ec[:, 1:, 1:, :])
    ef[:, 1::2, ::2, 1::2] = 0.25 * (ec[:, :-1, :, :-1] + ec[:, :-1, :, 1:]
                                     + ec[:, 1:, :, :-1] + ec[:, 1:, :, 1:])
    ef[:, ::2, 1::2, 1::2] = 0.25 * (ec[:, :, :-1, :-1] + ec[:, :, :-1, 1:]
                                     + ec[:, :, 1:, :-1] + ec[:, :, 1:, 1:])
    # 三奇：体上八点平均
    ef[:, 1::2, 1::2, 1::2] = 0.125 * (
        ec[:, :-1, :-1, :-1] + ec[:, :-1, :-1, 1:] + ec[:, :-1, 1:, :-1] + ec[:, :-1, 1:, 1:]
        + ec[:, 1:, :-1, :-1] + ec[:, 1:, :-1, 1:] + ec[:, 1:, 1:, :-1] + ec[:, 1:, 1:, 1:])
    return ef


_COARSEST_SWEEPS = 20


def _vcycle(u, f, h, coarsenings_left, pre, post, mask_cache):
    """递归 V-cycle（u 就地更新并返回）。coarsenings_left = 0 表示已在最粗层。"""
    n = u.shape[-1]
    h2 = h * h
    if n not in mask_cache:
        mask_cache[n] = _redblack_masks(n, u.device)
    masks = mask_cache[n]

    if coarsenings_left == 0:
        return _smooth_redblack(u, f, h2, _COARSEST_SWEEPS, masks)

    u = _smooth_redblack(u, f, h2, pre, masks)
    r = _residual(u, f, h2)
    rc = _restrict_fw(r)
    ec = torch.zeros_like(rc)
    ec = _vcycle(ec, rc, 2.0 * h, coarsenings_left - 1, pre, post, mask_cache)
    u = u + _prolong_trilinear(ec, n)
    u = _smooth_redblack(u, f, h2, post, masks)
    return u


def _multigrid_vcycle_core(u0, f, num_levels, pre_smooth, post_smooth, h, compute_dtype):
    """核心计算：以 compute_dtype 精度执行一次 V-cycle，返回 u_out。"""
    u = u0.to(compute_dtype).clone()
    # 边界置 0 是算子语义的一部分（任意随机 u0 输入合法）
    u[:, 0, :, :] = 0
    u[:, -1, :, :] = 0
    u[:, :, 0, :] = 0
    u[:, :, -1, :] = 0
    u[:, :, :, 0] = 0
    u[:, :, :, -1] = 0
    f_c = f.to(compute_dtype)
    u = _vcycle(u, f_c, float(h), int(num_levels),
                int(pre_smooth), int(post_smooth), {})
    return u.contiguous()


def multigrid_vcycle(
    u0: torch.Tensor,
    f: torch.Tensor,
    num_levels: int,
    pre_smooth: int,
    post_smooth: int,
    h: float,
) -> torch.Tensor:
    """
    3D Poisson 几何多重网格 V-cycle golden reference（plain golden = bench：fp32 计算）

    Args:
        u0: [B, N, N, N] 初始猜测（N = 2^k + 1；算子先将最外层置 0）
        f: [B, N, N, N] 右端项 −Δu = f（f 的边界值不参与计算）
        num_levels: 粗化次数（评测取值 2 ~ 5；case 保证最粗层 ≥ 5³）
        pre_smooth: 前平滑红黑 sweep 数（评测取值 1 ~ 4）
        post_smooth: 后平滑红黑 sweep 数（评测取值 1 ~ 4）
        h: 最细层网格间距（评测取值范围 [0.01, 1.0]；粗一层 h 翻倍）

    Returns:
        u_out: [B, N, N, N] 一次 V-cycle 后的解，dtype 与 u0 一致，最外层为 0
    """
    u_out = _multigrid_vcycle_core(
        u0, f, num_levels, pre_smooth, post_smooth, h, torch.float32)
    return u_out.to(u0.dtype)


def multigrid_vcycle_oracle(
    u0: torch.Tensor,
    f: torch.Tensor,
    num_levels: int,
    pre_smooth: int,
    post_smooth: int,
    h: float,
) -> torch.Tensor:
    """Oracle (g)：dtype-agnostic，计算精度跟随输入（fp64_cpu 下为 fp64 真值），不硬编码 .float()。"""
    return _multigrid_vcycle_core(
        u0, f, num_levels, pre_smooth, post_smooth, h, u0.dtype)
```

## 6. 额外信息

### 算子调用示例

```python
import torch

B, N = 2, 65

u0 = torch.rand(B, N, N, N, dtype=torch.float32, device="npu") * 2 - 1
f = torch.rand(B, N, N, N, dtype=torch.float32, device="npu") * 2 - 1

u_out = multigrid_vcycle(u0, f, num_levels=4, pre_smooth=2, post_smooth=2, h=1.0 / 64)
# u_out.shape: [B, N, N, N]，最外层为 0；层级 65³ → 33³ → 17³ → 9³ → 5³
```

### 收敛率实测（fp64，随机 f、零初值、外层迭代 8 次的渐近残差压缩率）

一次 V-cycle 是单步 kernel，收敛率不影响单次输出的正确性；下表供实现自检（对随机 f 迭代本算子，残差 2-范数应按表中倍率单调下降）：

| N | levels（最粗层） | pre+post | 实测收缩率/cycle | 说明 |
|---|---|---|---|---|
| 33 | 3（5³） | 2+2 | 0.098 | 教科书值 < 0.3（Briggs et al. 2000） |
| 65 | 4（5³） | 2+2 | 0.101 | 同上 |
| 129 | 5（5³） | 2+2 | 0.101 | 同上 |
| 33 | 3（5³） | 1+1 | 0.203 | 同上 |
| 33 | 3（5³） | 4+4 | 0.056 | 平滑越多收缩越快 |
| 65 | 2（17³） | 2+2 | 0.450 | 浅层级：最粗层 20 次平滑非精确解，收缩退化 |
| 129 | 2（33³） | 2+2 | 0.799 | 同上（仍单调下降） |

深层级（最粗层 ≤ 9³）配置全部 < 0.3；浅层级配置收缩率退化是"最粗层用 20 次平滑代替直接解"的规格后果，所有配置残差均单调下降（已数值验证）。

### 可用于自检的性质（均已数值验证）

- **不动点**：u0 = 0、f = 0 时输出恒 0；对随机 f 迭代至不动点后与稠密直接解一致（N=17 实测偏差 1.7e-18）
- **解析收敛**：$f = 3\pi^2 \sin(\pi x)\sin(\pi y)\sin(\pi z)$、$h = 1/(N-1)$ 时迭代收敛到 $u^\*$，误差为 $O(h^2)$ 离散化误差
- **伴随关系**：$\langle R r, e \rangle = \tfrac{1}{8}\langle r, P e \rangle$ 至舍入
- **f = 0 衰减**：任意 u0 下迭代解范数按收缩率衰减到 0
- **边界恒 0**：输出最外层精确为 0（不是近似 0）
- **线性性**：输出对 (u0, f) 联合线性（可用叠加原理抽查）

### 参考文献

- Brandt, A. (1977). "Multi-level adaptive solutions to boundary-value problems". Mathematics of Computation 31(138)（几何多重网格来源）
- Briggs, W. L., Henson, V. E., McCormick, S. F. (2000). "A Multigrid Tutorial, 2nd ed." SIAM（V-cycle、full-weighting/三线性转移对、收敛率教科书值）
- Trottenberg, U., Oosterlee, C. W., Schüller, A. (2001). "Multigrid". Academic Press（红黑 Gauss–Seidel 平滑分析）
- Adams, M., Brezina, M., Hu, J., Tuminaro, R. (2003). "Parallel multigrid smoothing: polynomial versus Gauss–Seidel". J. Comput. Phys. 188（红黑序并行平滑语义）
