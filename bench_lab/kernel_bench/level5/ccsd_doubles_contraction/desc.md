# CcsdDoublesContraction 算子 API 描述

## 1. 算子简介

耦合簇 CCSD/CCD 迭代中 T2 残差的主缩并项。耦合簇理论是量子化学的"金标准"方法，其每步迭代的算力主体是双激发振幅 $t_2$ 与双电子积分张量的高维稠密缩并：粒子-粒子 ladder 项（$O(o^2v^4)$）与 ring 项（$O(o^3v^3)$），再加上单粒子 Fock 项（$O(o^2v^3) + O(o^3v^2)$）。本算子取自旋轨道反对称化形式（CCD 残差的代表性子集），$o$ 为占据轨道数、$v$ 为虚轨道数，实际计算中 $v \gg o$，ladder 项的 $v^4$ 积分张量是显存与算力的双重主项。

这一缩并模式是 NWChem / GAMESS / MPQC 等量子化学软件包在超算上的核心负载（多次入围与获得 Gordon Bell 奖的负载类型）：单步迭代对 $v=128$ 档已是 $10^{11}$ 级浮点操作，且 8 个展开缩并项各自要求不同的张量布局，是稠密张量缩并引擎（TCE）优化的经典对象。

**主要应用场景**：
- 耦合簇（CCD/CCSD/CCSD(T)）能量迭代的残差构造（量子化学从头算）
- 稠密张量缩并引擎（TCE 类）与多项共享布局规划的负载测试
- 高维张量缩并的 matmul 化映射（本任务集首个以"布局墙"为机制轴的算子）

**算子特征**：
- 难度等级：L5（FusedComposite）
- 五输入单输出，无属性参数；全部输入先做对称类投影（幂等），任意随机输入合法
- 1 个 ladder + 4 个 ring 展开 + 4 个 Fock 展开共 9 个缩并项，缩并轴各不共面
- 输出满足反对称契约 $r_2[i,j,a,b] = -r_2[j,i,a,b] = -r_2[i,j,b,a]$（投影语义下对任意输入数学成立，可逐 case 验证）
- fp32 精度契约按化学精度锚定：$r_2$ 相对误差 ≤ 5e-4，经关联能敏感度传播即 sub-kcal/mol（见 §4）

**为何是 L5**（约为 L4 FusedComposite 的 2 倍难度）：
- **正确 ≠ 快**：按 §2 逐项直译 9 个 einsum 即可得到完全正确的结果并通过全部精度用例，但每项独立选择缩并布局——朴素实现对共享的 $A$ 与四个积分张量逐项重排/转置、$o^2v^2$ 级中间张量逐项往返 HBM，性能只能锚定朴素 baseline（预期得分 ~0.5 档）。逼近硬件下界要求把 9 项映射为共享驻留布局下的 matmul 簇：ladder 是 $[o^2, v^2] \times [v^2, v^2]$ 的大矩阵乘（$v=128$ 时积分张量单个 1 GB 量级、算强比高，纯矩阵乘算力饱和测试）；4 个 ring 项在 $[(i,a),(k,c)] \times [(k,c),(j,b)]$ 布局下共享同一次重排；投影、P 算子展开与 Fock 项可融合进主缩并的写出阶段。**布局与融合方案的选择是被测内容**，desc 不提供
- **实现规格而非过可见测试**：隐藏评测集覆盖公开用例之外的规格维度（$o=1$、$o=v$、$o>v$、素数 $o/v$、单边值域、各输入置零的退化分支等），投影的每一处约定（哪两对下标、除以 4 还是 2）、P 算子的符号与下标映射都会被单独检验
- **对称性契约可判**：投影语义使输出的双重反对称性对任意随机输入精确成立，实现中任何一处投影/符号/下标错误都会破坏该契约并被隐藏用例捕获
- **精度约束**：$v^4$ 缩并单元素归约长度达 $v^2 = 16384$，fp32 累加下相消位置误差被放大；低精度（bf16/fp16）输入下需要管理中间精度（见 §4）

## 2. 算子定义

### 记号

$o$ = 占据自旋轨道数，$v$ = 虚自旋轨道数。下标 $i,j,k \in [0,o)$ 为占据轨道，$a,b,c,d \in [0,v)$ 为虚轨道。

### 第一步：对称类投影（规格的一部分）

五个输入先各自投影到其物理对称类。投影均为**幂等**算子（对已满足对称性的物理输入是恒等变换），并保证任意随机输入合法：

$$
A[i,j,a,b] = \tfrac{1}{4}\big(t_2[i,j,a,b] - t_2[j,i,a,b] - t_2[i,j,b,a] + t_2[j,i,b,a]\big)
$$

$$
W[a,b,c,d] = \tfrac{1}{4}\big(V[a,b,c,d] - V[b,a,c,d] - V[a,b,d,c] + V[b,a,d,c]\big),
\quad V = \texttt{eri\_vvvv}
$$

$$
X[i,a,j,b] = \tfrac{1}{2}\big(\texttt{eri\_ovov}[i,a,j,b] + \texttt{eri\_ovov}[j,b,i,a]\big)
$$

$$
F_v = \tfrac{1}{2}\big(\texttt{fock\_vv} + \texttt{fock\_vv}^{\mathsf T}\big), \qquad
F_o = \tfrac{1}{2}\big(\texttt{fock\_oo} + \texttt{fock\_oo}^{\mathsf T}\big)
$$

物理背景：反对称化双电子积分 $\langle ab\Vert cd\rangle$ 与双激发振幅天然满足这些对称性；投影把"任意随机输入"映射回物理子空间。

### 第二步：缩并（四项和，P 算子全展开共 9 个缩并项）

置换算子定义：$P(ij)\,f(i,j) = f(i,j) - f(j,i)$，$P(ab)$ 同。

$$
r_2[i,j,a,b] = \underbrace{\sum_{cd} W[a,b,c,d]\, A[i,j,c,d]}_{\text{ladder, } O(o^2v^4)}
+ P(ij)P(ab) \underbrace{\sum_{kc} X[k,c,j,b]\, A[i,k,a,c]}_{\text{ring, } O(o^3v^3)}
+ P(ab) \underbrace{\sum_{c} F_v[b,c]\, A[i,j,a,c]}_{\text{粒子 Fock, } O(o^2v^3)}
- P(ij) \underbrace{\sum_{k} F_o[j,k]\, A[i,k,a,b]}_{\text{空穴 Fock, } O(o^3v^2)}
$$

P 算子展开后的 9 个缩并项（einsum 记号，输出下标均为 $[i,j,a,b]$）：

| # | 项 | einsum |
|---|----|--------|
| 1 | ladder | $+\sum_{cd} W[a,b,c,d]\,A[i,j,c,d]$ |
| 2 | ring 直接项 | $+\sum_{kc} X[k,c,j,b]\,A[i,k,a,c]$ |
| 3 | ring $i \leftrightarrow j$ | $-\sum_{kc} X[k,c,i,b]\,A[j,k,a,c]$ |
| 4 | ring $a \leftrightarrow b$ | $-\sum_{kc} X[k,c,j,a]\,A[i,k,b,c]$ |
| 5 | ring 双交换 | $+\sum_{kc} X[k,c,i,a]\,A[j,k,b,c]$ |
| 6 | 粒子 Fock 直接项 | $+\sum_{c} F_v[b,c]\,A[i,j,a,c]$ |
| 7 | 粒子 Fock $a \leftrightarrow b$ | $-\sum_{c} F_v[a,c]\,A[i,j,b,c]$ |
| 8 | 空穴 Fock 直接项 | $-\sum_{k} F_o[j,k]\,A[i,k,a,b]$ |
| 9 | 空穴 Fock $i \leftrightarrow j$ | $+\sum_{k} F_o[i,k]\,A[j,k,a,b]$ |

**本算子的精确约定**：
- 投影是规格的一部分：实现必须对五个输入全部先投影（对随机输入，漏掉任何一个投影都会得到完全错误的结果；对物理输入投影是恒等变换，不改变语义）
- 反对称契约：投影语义下 $r_2[i,j,a,b] = -r_2[j,i,a,b] = -r_2[i,j,b,a]$ 对**任意随机输入**数学成立（fp64 下已数值验证至 $4.3\times10^{-16}$ 相对偏差），由此 $i=j$ 或 $a=b$ 处 $r_2$ 精确为 0
- 退化关系（隐藏用例的锚点）：$o=1 \Rightarrow r_2 \equiv 0$（$A$ 投影恒零）；全常数输入 $\Rightarrow r_2 \equiv 0$；`eri_vvvv`=`eri_ovov`=0 $\Rightarrow$ 纯 Fock 项；`fock_vv`=`fock_oo`=0 $\Rightarrow$ 纯双电子项；`t2`=0 $\Rightarrow r_2 \equiv 0$

### 复杂度与访存特征（性能墙所在，机制说明）

9 项共享 $A$ 与四个积分张量，但缩并轴各不共面：ladder 缩 $(c,d)$ 对 $(i,j)$/$(a,b)$ 成块；4 个 ring 项都缩 $(k,c)$ 但输出下标映射互不相同（$X$ 的第 3/4 下标分别接到输出的 $j/b$、$i/b$、$j/a$、$i/a$）；Fock 项缩单轴。朴素逐项实现每项独立重排布局并写出 $o^2v^2$ 中间张量；高效实现需要为共享张量选择驻留布局、让多项复用同一次重排，并把投影/P 展开/加和融合进主缩并——这是本算子的"布局墙"，也是任务集中首个以高维稠密张量缩并布局为机制轴的任务。

## 3. 接口规范

### 算子原型

```python
ccsd_doubles_contraction(Tensor t2, Tensor eri_vvvv, Tensor eri_ovov, Tensor fock_vv, Tensor fock_oo) -> Tensor r2
```

### 输入参数说明

| 参数 | 类型 | 必需 | dtype | shape | 描述 |
|------|------|------|-------|-------|------|
| t2 | Tensor | 是 | float32/float16/bfloat16 | [o, o, v, v] | 双激发振幅，先做 (ij)/(ab) 反对称化投影 |
| eri_vvvv | Tensor | 是 | float32/float16/bfloat16 | [v, v, v, v] | 虚-虚双电子积分 ⟨ab‖cd⟩，先做 (ab)/(cd) 反对称化投影 |
| eri_ovov | Tensor | 是 | float32/float16/bfloat16 | [o, v, o, v] | 占据-虚双电子积分，先做 (ia)↔(jb) 对称化投影 |
| fock_vv | Tensor | 是 | float32/float16/bfloat16 | [v, v] | 虚-虚 Fock 块，先对称化 (F+Fᵀ)/2 |
| fock_oo | Tensor | 是 | float32/float16/bfloat16 | [o, o] | 占据-占据 Fock 块，先对称化 (F+Fᵀ)/2 |

### 输出

| 名称 | dtype | shape | 描述 |
|------|-------|-------|------|
| r2 | 与 t2 一致 | [o, o, v, v] | T2 残差主缩并项，满足双重反对称契约 |

### 数据类型

| 输入 dtype | 输出 dtype | 内部计算 |
|-----------|-----------|----------|
| float32 | float32 | fp32 |
| float16 | float16 | fp32（投影与缩并累加以 fp32 进行） |
| bfloat16 | bfloat16 | fp32 |

### 规则与约束

- 五个 Tensor 输入 dtype 必须一致
- 维度一致性：t2 的 shape 为 [o,o,v,v]；eri_vvvv 为 [v,v,v,v]；eri_ovov 为 [o,v,o,v]；fock_vv 为 [v,v]；fock_oo 为 [o,o]；o、v 由 t2 推断
- 投影与 9 项缩并的累加以 fp32（或更高）进行；低精度输入场景下这是精度达标的必要条件
- 显存约束：case 空间保证全部输入+输出按 fp64 计合计 ≤ 2.5 GB（fp64 oracle 上限；主项为 eri_vvvv 的 v⁴，v=128 时 fp32 1 GB / fp64 2 GB）
- 输出须为 contiguous 张量

### 支持范围

| 维度 / 参数 | 范围 | 备注 |
|---|---|---|
| `o`（占据轨道数） | 1 ~ 32 | cases.csv 实测 1 ~ 32；o=1 时输出恒 0 |
| `v`（虚轨道数） | 16 ~ 128 | cases.csv 实测 16 ~ 128，v=128 时 eri_vvvv 为 GB 级 |
| dtype | float32 / float16 / bfloat16 | cases.csv 实测三种均覆盖 |
| `t2` 取值 | [-1, 1] | 逐 case 定标使 r2 元素 std ≈ 0.3（物理振幅量级 ±0.02 ~ ±0.2） |
| `eri_vvvv` / `eri_ovov` 取值 | [-1e3, 1e3] | 常规评测取 [-1, 1]（t2 值域陷阱 case 中按 r2 std 定标） |
| `fock_vv` / `fock_oo` 取值 | [-20, 20] | 常规评测取 [-2, 2] |

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

**通过标准**（fp32 按化学精度锚定）：

| 数据类型 | FLOAT32 | FLOAT16 | BFLOAT16 |
|----------|---------|---------|----------|
| **通过阈值(Threshold)** | 5e-4 | 0.005 | 0.02 |

当平均相对误差 MERE < Threshold，最大相对误差 MARE < 10 * Threshold 时判定为通过。

**化学精度契约**（fp32 阈值依据）：

$r_2$ 是残差张量而非能量，化学口径经敏感度传播转译。关联能

$$
E_{\text{corr}} = \tfrac{1}{4}\sum_{ijab} \langle ij\Vert ab\rangle\, t_2[i,j,a,b]
$$

CC 迭代收敛到 $r_2 = 0$ 的不动点，$r_2$ 的相对误差 $\delta$ 通过不动点方程传播为收敛振幅、进而 $E_{\text{corr}}$ 的同量级相对误差：$|\Delta E| \approx \delta \cdot |E_{\text{corr}}|$。cc-pVDZ 档小分子的典型 $|E_{\text{corr}}| \sim 0.2 - 2$ Ha（1 kcal/mol = 1.5936e-3 Ha），sub-kcal/mol 要求 $\delta \le 8\times10^{-4}$。取 $\delta = 5\times10^{-4}$：$|\Delta E| \le 5\times10^{-4} \times 2\ \text{Ha} = 0.63$ kcal/mol，即振幅迭代收敛到 **sub-kcal 关联能**所需的残差精度。参考实现（fp32 数据通路）实测 MERE $\le 1.1\times10^{-6}$，裕量 450x。

**阈值可行性**：cases 的 t2（或陷阱本体所在的 eri）值域逐 case 定标，使 $r_2$ 元素 std $\approx 0.3$（物理残差量级，且与 t2 振幅随基组增大而变小的真实趋势一致）。该定标把反对称结构零元（$i=j$ / $a=b$ 处 golden 精确为 0）与过零相消带的 fp32 误差尾部压入 MARE 判据（$10\times$ 阈值 = 5e-3）安全区：更紧的阈值档（1e-4 ~ 4e-4）经 16-seed 实测存在相消计数比的 Poisson 误杀（plain golden 与 fp64 oracle 共享求和顺序、在相消元异常相关，使兜底判定的同精度参考计数偏低），5e-4 + 定标后 16-seed 全配置稳定通过。bf16/fp16 为低精度数据通路变体，不承诺化学精度，沿用同类先例阈值。**达标前提**：投影与缩并累加以 fp32 保存/累加。判别力（实测，5e-4）：漏投影、投影归一化错、P 展开缺项、符号错等实现错误的 MERE 在 1.3 ~ 7.9，与阈值相隔 3 个数量级以上。

## 5. 标准 Golden 代码

```python
import torch


def _ccsd_doubles_contraction_core(t2, eri_vvvv, eri_ovov, fock_vv, fock_oo, compute_dtype):
    """核心计算：投影 + 9 项缩并，以 compute_dtype 精度执行，返回 r2 [o, o, v, v]。"""
    a = t2.to(compute_dtype)
    w = eri_vvvv.to(compute_dtype)
    x = eri_ovov.to(compute_dtype)
    fv = fock_vv.to(compute_dtype)
    fo = fock_oo.to(compute_dtype)

    # === 第一步：对称类投影（幂等） ===
    a = (a - a.permute(1, 0, 2, 3) - a.permute(0, 1, 3, 2) + a.permute(1, 0, 3, 2)) / 4
    w = (w - w.permute(1, 0, 2, 3) - w.permute(0, 1, 3, 2) + w.permute(1, 0, 3, 2)) / 4
    x = (x + x.permute(2, 3, 0, 1)) / 2
    fv = (fv + fv.transpose(0, 1)) / 2
    fo = (fo + fo.transpose(0, 1)) / 2

    # === ladder: Σ_{cd} W[a,b,c,d] A[i,j,c,d] ===
    r2 = torch.einsum('abcd,ijcd->ijab', w, a)

    # === ring: P(ij)P(ab) Σ_{kc} X[k,c,j,b] A[i,k,a,c] ===
    # T[i,j,a,b] = Σ_{kc} X[k,c,j,b] A[i,k,a,c]；
    # 其余 3 项是 T 的输出下标重排（与逐项独立 einsum 逐位一致）:
    #   −T[j,i,a,b] = −Σ X[k,c,i,b] A[j,k,a,c]
    #   −T[i,j,b,a] = −Σ X[k,c,j,a] A[i,k,b,c]
    #   +T[j,i,b,a] = +Σ X[k,c,i,a] A[j,k,b,c]
    ring = torch.einsum('kcjb,ikac->ijab', x, a)
    r2 = r2 + ring - ring.permute(1, 0, 2, 3) - ring.permute(0, 1, 3, 2) + ring.permute(1, 0, 3, 2)

    # === 粒子 Fock: P(ab) Σ_c Fv[b,c] A[i,j,a,c] ===
    pf = torch.einsum('bc,ijac->ijab', fv, a)
    r2 = r2 + pf - pf.permute(0, 1, 3, 2)

    # === 空穴 Fock: −P(ij) Σ_k Fo[j,k] A[i,k,a,b] ===
    hf = torch.einsum('jk,ikab->ijab', fo, a)
    r2 = r2 - hf + hf.permute(1, 0, 2, 3)

    return r2.contiguous()


def ccsd_doubles_contraction(
    t2: torch.Tensor,
    eri_vvvv: torch.Tensor,
    eri_ovov: torch.Tensor,
    fock_vv: torch.Tensor,
    fock_oo: torch.Tensor,
) -> torch.Tensor:
    """
    CCSD T2 残差主缩并 golden reference（plain golden = bench：fp32 数据通路）

    Args:
        t2: [o, o, v, v] 双激发振幅，算子先做反对称化投影 A
        eri_vvvv: [v, v, v, v] 虚-虚双电子积分 ⟨ab||cd⟩，算子先做反对称化投影 W
        eri_ovov: [o, v, o, v] 占据-虚双电子积分，算子先做 (ia)↔(jb) 对称化投影 X
        fock_vv: [v, v] 虚-虚 Fock 块，算子先对称化 Fv = (F+Fᵀ)/2
        fock_oo: [o, o] 占据-占据 Fock 块，算子先对称化 Fo = (F+Fᵀ)/2

    Returns:
        r2: [o, o, v, v] T2 残差主缩并项，dtype 与 t2 一致；
            满足 r2[i,j,a,b] = −r2[j,i,a,b] = −r2[i,j,b,a]
    """
    r2 = _ccsd_doubles_contraction_core(
        t2, eri_vvvv, eri_ovov, fock_vv, fock_oo, torch.float32)
    return r2.to(t2.dtype)


def ccsd_doubles_contraction_oracle(
    t2: torch.Tensor,
    eri_vvvv: torch.Tensor,
    eri_ovov: torch.Tensor,
    fock_vv: torch.Tensor,
    fock_oo: torch.Tensor,
) -> torch.Tensor:
    """Oracle (g)：dtype-agnostic，计算精度跟随输入（fp64_cpu 下为 fp64 真值），不硬编码 .float()。"""
    return _ccsd_doubles_contraction_core(
        t2, eri_vvvv, eri_ovov, fock_vv, fock_oo, t2.dtype)
```

## 6. 额外信息

### 算子调用示例

```python
import torch

o, v = 16, 76   # (H2O)4 / cc-pVDZ 冻芯量级

t2 = torch.empty(o, o, v, v, dtype=torch.float32, device="npu").uniform_(-0.05, 0.05)
eri_vvvv = torch.empty(v, v, v, v, dtype=torch.float32, device="npu").uniform_(-1, 1)
eri_ovov = torch.empty(o, v, o, v, dtype=torch.float32, device="npu").uniform_(-1, 1)
fock_vv = torch.empty(v, v, dtype=torch.float32, device="npu").uniform_(-2, 2)
fock_oo = torch.empty(o, o, dtype=torch.float32, device="npu").uniform_(-2, 2)

r2 = ccsd_doubles_contraction(t2, eri_vvvv, eri_ovov, fock_vv, fock_oo)
# r2.shape: [o, o, v, v]；r2[i,j,a,b] == -r2[j,i,a,b] == -r2[i,j,b,a]
```

### 真实分子量级对照（cases.csv 标注口径，空间轨道计数）

| 体系 / 基组 | o | v |
|---|---|---|
| (H₂O)₂ / cc-pVDZ 冻芯 | 8 | 38 |
| (H₂O)₃ / cc-pVDZ 冻芯 | 12 | 57 |
| (H₂O)₄ / cc-pVDZ 冻芯 | 16 | 76 |
| 苯 / cc-pVDZ 全电子 | 21 | 93 |
| 苯 / cc-pVDZ 冻芯 | 15 | 93 |
| (H₂O)₂ / cc-pVTZ 冻芯档 | 8 | 106~128 |

### 公式验证（golden 交付前已执行）

- 与独立五重循环标量实现（纯 Python float 累加，o=3/v=4 与 o=4/v=6）fp64 对照：最大相对偏差 5.9e-16
- 反对称契约在 5 组随机形状上验证：最大相对偏差 4.3e-16
- 退化锚点全部逐位成立（o=1 / 常数输入 / t2=0 → 精确 0；eri=0 与 fock=0 分支与逐项独立 einsum 实现一致）
- 投影幂等性：oracle(P(x)) 与 oracle(x) 相对偏差 2.8e-16

### 参考文献

- Bartlett, R. J., Musiał, M. (2007). "Coupled-cluster theory in quantum chemistry". Rev. Mod. Phys. 79, 291（CCD/CCSD 残差方程的谱系与记号）
- Shavitt, I., Bartlett, R. J. (2009). "Many-Body Methods in Chemistry and Physics: MBPT and Coupled-Cluster Theory". Cambridge University Press（自旋轨道反对称化形式、P 置换算子约定）
- Hirata, S. (2003). "Tensor Contraction Engine". J. Phys. Chem. A 107, 9887（高维稠密张量缩并的布局/公共子式优化，本算子性能墙的经典处理）
