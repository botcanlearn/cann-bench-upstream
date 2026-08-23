# StatevectorAdjointGradient 算子 API 描述

## 1. 算子简介

量子态矢电路模拟 + 伴随法梯度（adjoint differentiation）。算子输入一个 $n$ qubit 的复数态矢（实部/虚部分离存储）、一段以整数编码的量子门序列（程序即数据）、旋转门参数向量与一个 Pauli 张量积可观测量；输出可观测量期望值 $\langle\psi_G|O|\psi_G\rangle$ 及其对全部参数的梯度。梯度用伴随法（adjoint method）计算：前向作用完全部门后，**逆序**逐门作用 $U_g^\dagger$ 回退态矢，同时维护第二个态矢 $|\phi\rangle$，每个含参门贡献一项 $2\,\mathrm{Re}\langle\phi|\partial U_g|\psi\rangle$。这是 PennyLane-lightning 与 cuQuantum 中变分量子算法（VQE/QAOA/量子机器学习）梯度计算的核心 kernel：相比 parameter-shift（每参数两次全电路模拟，$O(T \cdot G \cdot 2^n)$），伴随法只需一次前向 + 一次逆序回放（$O(G \cdot 2^n)$），是经典模拟侧训练变分电路的标准算法。

**主要应用场景**：
- 变分量子算法（VQE、QAOA、量子核方法）的经典模拟训练循环
- 量子电路可微编程框架（PennyLane-lightning、cuQuantum ExpectationCalculation/adjoint）的后端算子
- 量子体系结构研究中的大规模态矢模拟（n = 20~40 是单机态矢模拟的主战场）

**算子特征**：
- 难度等级：L5（FusedComposite）；**建议定级 L6**（见下）
- 四输入（state, circuit, thetas, pauli_obs）双输出（expval, grads）
- 复数全程以 [..., 2] 实部/虚部分离表示，纯实数算术
- 电路以整数取模译码（任意随机 int32 输入都是合法程序）；qa == qb 的两比特门定义为恒等门，任意随机电路合法
- 输入态由算子归一化（$\psi \leftarrow \psi/(\lVert\psi\rVert+10^{-12})$），任意随机态合法

**建议定级 L6**（仓库难度枚举现行最高为 L5，本任务按 L5 收录；按下述评估，其复杂度超出 L5——至少四根 L5 难度轴耦合于一个算子，且存在错误全局传播、不可分段验证的长链——建议仓库增设 L6 档后调升）：

| # | 难度轴 | 对应 L5 先例 | 本算子中的形态 |
|---|--------|--------------|----------------|
| 1 | 复数算术手工实虚分离 | fft_conv（复数蝶形运算） | NPU 无复数类型，全部门都是 2×2/受控复矩阵作用，每次配对更新是 4 次实数乘加的手工复数乘法，实/虚部的符号约定错一处即全错 |
| 2 | 蝶形跨步访存 | fft_conv（bit-reverse/蝶形数据流） | 对 qubit $q$ 作用 = 以跨步 $2^q$ 配对整个态矢：低位 qubit 配对相邻元素（向量化友好），高位 qubit 配对相距半个态矢的元素（跨越任何片上缓冲），同一电路内两种极端交替出现 |
| 3 | 程序即数据 | tensor_program_vm（kernel 即解释器） | 门序列是运行时输入，kernel 必须译码-分派；访存模式、算强、可否融合全部数据依赖，无法编译期特化 |
| 4 | 反演长链（伴随回放） | 无 L5 先例，本算子独有 | 逆序链要求每个 $U_g^\dagger$ 与前向的 $U_g$ **精确互逆**：一个门的实现偏差不是局部误差，而是从该门开始污染其后全部 $2 \times (G-g)$ 次全态矢变换与全部梯度分量；且中间态不可存（保存 $G$ 份 $2^n$ 态矢需要 $G\times$ 显存，物理不可行），只能靠逆运算重建 |

**为何复合后不可分解**：输出只有末端的两个量（expval 标量与 grads 向量），整条 前向 $G$ 门 → Pauli 作用 → 逆序 $G$ 门双态矢回放 的 $3G$ 遍全态矢链上没有任何可观测的中间结果；四根轴不是四个可独立调通的阶段，而是同一个内层循环的四个耦合属性——每一次配对更新同时是（复数乘法 × 某个跨步 × 某条译码指令 × 前向或逆向的某一环）。任何一处错误都全局传播到两个输出，无法用分段单测定位。

**性能墙**：朴素实现按定义逐门扫描全态矢，共 $3G$ 遍全局读写（前向 $G$ + 逆向 $\psi$、$\phi$ 各 $G$），访存完全受限（每遍算强 ~10 FLOP/元素）。逼近硬件下界要求：(1) **门融合**——相邻的同 qubit/相邻 qubit 门在寄存器/片上缓冲内连乘后一次作用（lightning/cuQuantum 的核心优化），把全局遍数从 $3G$ 压到 $3G/\text{fusion factor}$；(2) **按 qubit 局部性重排块内计算**——低位 qubit 的配对在块内完成，高位 qubit 的配对通过块间通信/二次划分处理；(3) 逆序回放与梯度内积（归约）的流水重叠。这些优化全部依赖运行时译码结果，静态调度无从下手。

## 2. 算子定义

### 数据表示

复数态矢 $\psi \in \mathbb{C}^{2^n}$ 存储为 `state [2^n, 2]`（`[..., 0]` 实部、`[..., 1]` 虚部）。基态编号约定：$|b_{n-1} \cdots b_1 b_0\rangle$ 的下标为 $i = \sum_q b_q 2^q$，即 **qubit $q$ 的跨步为 $2^q$**。$n$ 由 `state.shape[0] = 2^n` 推出，`pauli_obs.shape[0]` 必须等于 $n$（由 cases 保证）。

对第 $q$ 个 qubit 作用一个 $2\times2$ 矩阵 $M$，等价于把态矢 view 成 $(2^{n-q-1}, 2, 2^q)$ 后对中间维做变换：记 $a = V[:, 0], b = V[:, 1]$（各 $[2^{n-q-1}, 2^q]$ 复数），则

$$
\begin{pmatrix} a' \\ b' \end{pmatrix} = \begin{pmatrix} m_{00} & m_{01} \\ m_{10} & m_{11} \end{pmatrix} \begin{pmatrix} a \\ b \end{pmatrix}
$$

复数乘法以实虚分离展开（4 实数乘 + 2 实数加）：$(x_r + i x_i)(w_r + i w_i) = (x_r w_r - x_i w_i) + i(x_r w_i + x_i w_r)$。

### 归一化（第一步，算子语义）

$$
\psi_0 = \frac{\text{state}}{\lVert \text{state} \rVert_2 + 10^{-12}}, \qquad \lVert \text{state} \rVert_2 = \sqrt{\textstyle\sum_i (\text{re}_i^2 + \text{im}_i^2)}
$$

eps 加在范数上。归一化保证任意随机输入态合法，且全 I 可观测量下 expval = 1（自检性质）。

### 电路译码（程序即数据）

`circuit [G, 4] int32` 每行 $[c_0, c_1, c_2, c_3]$ 译码（任意非负整数合法）：

$$
\text{gate} = c_0 \bmod 5, \quad qa = c_1 \bmod n, \quad qb = c_2 \bmod n, \quad tid = c_3 \bmod T
$$

多条指令可共享同一 $tid$（该参数的梯度按链式法则**累加**）。

### 门集（矩阵全给出，$h = \theta/2$）

| gate | 门 | 矩阵 | 实虚分离形式 |
|------|----|------|--------------|
| 0 | $H(qa)$ | $\frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}$ | $a' = (a+b)/\sqrt2$，$b' = (a-b)/\sqrt2$（实/虚部各自如此） |
| 1 | $RZ(\theta)(qa)$ | $\begin{pmatrix} e^{-ih} & 0 \\ 0 & e^{ih} \end{pmatrix}$ | $a' = (\cos h - i\sin h)\,a$，$b' = (\cos h + i\sin h)\,b$ |
| 2 | $RX(\theta)(qa)$ | $\begin{pmatrix} \cos h & -i\sin h \\ -i\sin h & \cos h \end{pmatrix}$ | $a' = \cos h\, a - i \sin h\, b$，$b' = -i\sin h\, a + \cos h\, b$ |
| 3 | $\mathrm{CNOT}(qa \to qb)$ | 控制位 $qa$ 为 1 时对 $qb$ 作用 $X$ | 控制比特 = 1 的半空间内，目标比特 0/1 两片振幅互换（实虚整体交换） |
| 4 | $\mathrm{CZ}(qa, qb)$ | $\mathrm{diag}(1,1,1,-1)$ | 两比特均为 1 的振幅乘 $-1$（$qa/qb$ 对称） |

**约定**：gate 3/4 在 $qa = qb$ 时定义为**恒等门**（不作用），保证任意随机电路合法。$\theta$ 一律取 $\theta_{tid}$。

两比特门的下标语义：记 $hi = \max(qa,qb), lo = \min(qa,qb)$，view 成 $(2^{n-1-hi},\, 2,\, 2^{hi-1-lo},\, 2,\, 2^{lo})$ 后第 2 维是 qubit $hi$ 的比特、第 4 维是 qubit $lo$ 的比特，按上表在对应半空间内变换。

### Pauli 张量积可观测量

$O = \bigotimes_{q} P_q$（$P_q$ 按 `pauli_obs[q] mod 4` 取 $I/X/Y/Z$），逐 qubit 独立作用（相互对易）：

| 编码 | $P$ | 作用（跨步 $2^q$ 配对 $(a, b)$） |
|------|-----|------|
| 0 | $I$ | 不变 |
| 1 | $X$ | $(a, b) \to (b, a)$ |
| 2 | $Y = \begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix}$ | $(a, b) \to (-i b,\; i a)$，实虚分离：$a' = (b_{im},\, -b_{re})$，$b' = (-a_{im},\, a_{re})$ |
| 3 | $Z$ | $(a, b) \to (a, -b)$ |

### 伴随法完整流程

1. **前向**：$|\psi\rangle = U_{G-1} \cdots U_1 U_0 |\psi_0\rangle$（按 $g = 0, \dots, G-1$ 顺序作用）。
2. $|\phi\rangle = O|\psi\rangle$；$\text{expval} = \mathrm{Re}\langle\phi|\psi\rangle = \sum_i (\phi_{re,i}\psi_{re,i} + \phi_{im,i}\psi_{im,i})$（$O$ 厄米，结果为实数）。
3. **逆序回放** $g = G-1, \dots, 0$：
   - $|\psi\rangle \leftarrow U_g^\dagger |\psi\rangle$（此后 $|\psi\rangle$ 是门 $g$ **之前**的态）；
   - 若 $U_g$ 含参（RZ/RX）：$\text{grads}[tid_g] \mathrel{+}= 2\,\mathrm{Re}\langle\phi|\,\partial U_g/\partial\theta\,|\psi\rangle$；
   - $|\phi\rangle \leftarrow U_g^\dagger |\phi\rangle$。

其中 $U^\dagger$：H/CNOT/CZ 自伴（$U^\dagger = U$）；$RZ(\theta)^\dagger = RZ(-\theta)$、$RX(\theta)^\dagger = RX(-\theta)$。导数矩阵（$h = \theta/2$）：

$$
\frac{\partial RZ}{\partial \theta} = \frac{1}{2}\begin{pmatrix} -\sin h - i\cos h & 0 \\ 0 & -\sin h + i\cos h \end{pmatrix}, \qquad
\frac{\partial RX}{\partial \theta} = \frac{1}{2}\begin{pmatrix} -\sin h & -i\cos h \\ -i\cos h & -\sin h \end{pmatrix}
$$

以上全部公式已数值验证：与独立稠密矩阵实现（complex128 直接构造 $2^n \times 2^n$ 门矩阵，$n \le 8$）的 expval/grads 偏差 < 1e-15；grads 与中心差分数值梯度（fp64，eps=1e-5）全部参数绝对偏差 < 1e-9、非平凡分量相对偏差 < 3e-10；性质 $H^2 = I$、$\mathrm{CNOT}^2 = I$、RZ-only 电路对全 Z 可观测量梯度为 0、共享 tid 梯度可加性均逐位成立。

## 3. 接口规范

### 算子原型

```python
statevector_adjoint_gradient(Tensor state, Tensor circuit, Tensor thetas, Tensor pauli_obs) -> (Tensor expval, Tensor grads)
```

### 输入参数说明

| 参数 | 类型 | 必需 | dtype | shape | 描述 |
|------|------|------|-------|-------|------|
| state | Tensor | 是 | float32 | [2^n, 2] | 输入态（实部/虚部分离），算子内部归一化；行数必须为 2 的幂 |
| circuit | Tensor | 是 | int32 | [G, 4] | 电路指令，逐行 mod 译码，任意非负整数合法（评测取值范围 [0, 1073741823]） |
| thetas | Tensor | 是 | float32 | [T] | 旋转门参数（评测取值范围 [-3.14159, 3.14159]），可被多条指令共享 |
| pauli_obs | Tensor | 是 | int32 | [n] | 可观测量逐 qubit 编码 0 I/1 X/2 Y/3 Z（评测取值范围 [0, 3]），长度 = n |

### 输出

| 名称 | dtype | shape | 描述 |
|------|-------|-------|------|
| expval | float32 | [1] | 期望值 $\langle\psi_G|O|\psi_G\rangle$ |
| grads | float32 | [T] | $\partial\,\text{expval}/\partial\,\theta_t$，无参数贡献的分量为 0 |

### 数据类型

| state/thetas dtype | 输出 dtype | 内部计算 |
|--------------------|-----------|----------|
| float32 | float32 | fp32（唯一支持组合，见 §4） |

### 规则与约束

- `state.shape[0]` 必须为 2 的幂；$n = \log_2(\text{state.shape}[0])$；`pauli_obs.shape[0] == n`（由 cases 保证）
- 译码取模保证任意 int32 非负输入合法：越界 qubit/tid 不存在
- $qa = qb$ 的 CNOT/CZ 是恒等门（含 $c_1 \bmod n = c_2 \bmod n$ 的所有情形）
- grads 中从未被任何含参门引用的 $tid$ 分量为 0；仅含 H/CNOT/CZ 的电路 grads 恒为全 0
- 中间态精度：态矢与梯度累加全程 fp32；不得以 fp16/bf16 保存态矢（见 §4）
- 输出须为 contiguous 张量

### 支持范围

| 维度 / 参数 | 范围 | 备注 |
|---|---|---|
| `n`（qubit 数） | 8 ~ 22 | cases.csv 实测 10 ~ 20（state 最大 8 MB）；隐藏用例至 22 |
| `G`（门数） | 1 ~ 256 | cases.csv 实测 1 ~ 256 |
| `T`（参数数） | 1 ~ 64 | cases.csv 实测 1 ~ 64 |
| `state` 取值 | [-1, 1] | 算子内部归一化 |
| `circuit` 取值 | [0, 1073741823] | mod 译码，窄值域可构造特定门型电路 |
| `thetas` 取值 | [-3.14159, 3.14159] | 一个周期 |
| `pauli_obs` 取值 | [0, 3] | 全 I / 全 Z 为特殊用例 |
| dtype | float32（state/thetas）+ int32（circuit/pauli_obs） | 唯一组合 |

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

| 数据类型 | FLOAT32 |
|----------|---------|
| **通过阈值(Threshold)** | 0.0001 |

当平均相对误差 MERE < Threshold，最大相对误差 MARE < 10 * Threshold 时判定为通过。阈值经评测框架 checker 实测确定：plain(fp32) vs fp64 oracle 全部 100 case 双跑校准（含最大规模 n=22/G=256/T=64、深参数链 G=256 全 RZ/RX、全门共享单参数 T=1 与全部陷阱用例），checker 全量判 passed，实测 MERE ≤ 1.26e-5、MARE ≤ 3.94e-4；0.0001 留 ≥7x（MERE）/ ≥2.5x（MARE 判据 1e-3）裕量，覆盖独立 fp32 实现（不同归约顺序、门融合重排）的合理差异；小值域/相消尖峰由评测框架兜底标准处理（native 参考 = plain golden，同为 fp32 路径，兜底比较公平）。

**为何仅支持 fp32**：$G$ 级酉变换链的误差按 $O(\sqrt{G}\,\varepsilon)$ 积累且逆序回放再走一遍（等效 $3G$ 链）；fp16/bf16 的 $\varepsilon$（~5e-4 / 4e-3）会使末端态矢与梯度误差进入百分比量级，梯度符号都不可靠，复数模拟界（cuQuantum、lightning）的最低精度实践即为 fp32（态矢与累加均 fp32）。评测不提供 fp16/bf16 用例。

## 5. 标准 Golden 代码

```python
import math
from typing import Tuple

import torch


def _apply_1q(psi, n, q, m):
    """对第 q 个 qubit 应用 2x2 复矩阵 m（python 复数元组 ((m00,m01),(m10,m11))）。

    psi: [2^n, 2]（最后一维 = 实部/虚部），返回新张量。
    视图 (2^(n-q-1), 2, 2^q, 2)：dim1 为 qubit q 的比特（跨步 2^q）。
    """
    v = psi.reshape(1 << (n - 1 - q), 2, 1 << q, 2)
    a_re, a_im = v[:, 0, :, 0], v[:, 0, :, 1]
    b_re, b_im = v[:, 1, :, 0], v[:, 1, :, 1]
    (m00, m01), (m10, m11) = m
    out = torch.empty_like(v)
    out[:, 0, :, 0] = m00.real * a_re - m00.imag * a_im + m01.real * b_re - m01.imag * b_im
    out[:, 0, :, 1] = m00.real * a_im + m00.imag * a_re + m01.real * b_im + m01.imag * b_re
    out[:, 1, :, 0] = m10.real * a_re - m10.imag * a_im + m11.real * b_re - m11.imag * b_im
    out[:, 1, :, 1] = m10.real * a_im + m10.imag * a_re + m11.real * b_im + m11.imag * b_re
    return out.reshape(psi.shape)


def _apply_cnot(psi, n, qa, qb):
    """CNOT：qa 为控制位，qb 为目标位（qa != qb 由调用方保证）。"""
    hi, lo = max(qa, qb), min(qa, qb)
    v = psi.reshape(1 << (n - 1 - hi), 2, 1 << (hi - 1 - lo), 2, 1 << lo, 2).clone()
    if qa == hi:                              # 控制位在 dim1，目标位在 dim3
        tmp = v[:, 1, :, 0].clone()
        v[:, 1, :, 0] = v[:, 1, :, 1]
        v[:, 1, :, 1] = tmp
    else:                                     # 控制位在 dim3，目标位在 dim1
        tmp = v[:, 0, :, 1].clone()
        v[:, 0, :, 1] = v[:, 1, :, 1]
        v[:, 1, :, 1] = tmp
    return v.reshape(psi.shape)


def _apply_cz(psi, n, qa, qb):
    """CZ：两比特均为 1 的振幅乘 -1（qa != qb 由调用方保证，qa/qb 对称）。"""
    hi, lo = max(qa, qb), min(qa, qb)
    v = psi.reshape(1 << (n - 1 - hi), 2, 1 << (hi - 1 - lo), 2, 1 << lo, 2).clone()
    v[:, 1, :, 1] = -v[:, 1, :, 1]
    return v.reshape(psi.shape)


_INV_SQRT2 = 1.0 / math.sqrt(2.0)
_H_MAT = ((complex(_INV_SQRT2, 0.0), complex(_INV_SQRT2, 0.0)),
          (complex(_INV_SQRT2, 0.0), complex(-_INV_SQRT2, 0.0)))


def _rz_mat(theta):
    h = 0.5 * theta
    return ((complex(math.cos(h), -math.sin(h)), complex(0.0, 0.0)),
            (complex(0.0, 0.0), complex(math.cos(h), math.sin(h))))


def _rx_mat(theta):
    h = 0.5 * theta
    return ((complex(math.cos(h), 0.0), complex(0.0, -math.sin(h))),
            (complex(0.0, -math.sin(h)), complex(math.cos(h), 0.0)))


def _drz_mat(theta):
    """dRZ/dtheta = (-i/2) diag(e^{-i theta/2}, -e^{i theta/2})。"""
    h = 0.5 * theta
    return ((complex(-0.5 * math.sin(h), -0.5 * math.cos(h)), complex(0.0, 0.0)),
            (complex(0.0, 0.0), complex(-0.5 * math.sin(h), 0.5 * math.cos(h))))


def _drx_mat(theta):
    """dRX/dtheta = 0.5 * [[-sin(h), -i cos(h)], [-i cos(h), -sin(h)]]，h = theta/2。"""
    h = 0.5 * theta
    return ((complex(-0.5 * math.sin(h), 0.0), complex(0.0, -0.5 * math.cos(h))),
            (complex(0.0, -0.5 * math.cos(h)), complex(-0.5 * math.sin(h), 0.0)))


def _apply_gate(psi, n, gate, qa, qb, theta):
    """按译码结果应用一个门；qa == qb 的两比特门为恒等门。"""
    if gate == 0:
        return _apply_1q(psi, n, qa, _H_MAT)
    if gate == 1:
        return _apply_1q(psi, n, qa, _rz_mat(theta))
    if gate == 2:
        return _apply_1q(psi, n, qa, _rx_mat(theta))
    if qa == qb:
        return psi
    if gate == 3:
        return _apply_cnot(psi, n, qa, qb)
    return _apply_cz(psi, n, qa, qb)


def _apply_gate_dagger(psi, n, gate, qa, qb, theta):
    """应用门的厄米共轭 U^dagger：H/CNOT/CZ 自伴，RZ/RX 取 -theta。"""
    if gate in (1, 2):
        return _apply_gate(psi, n, gate, qa, qb, -theta)
    return _apply_gate(psi, n, gate, qa, qb, theta)


def _apply_pauli(psi, n, pauli):
    """应用张量积可观测量 O = ⊗_q P_q（0 I、1 X、2 Y、3 Z），返回新张量。"""
    phi = psi
    for q in range(n):
        p = pauli[q]
        if p == 0:
            continue
        v = phi.reshape(1 << (n - 1 - q), 2, 1 << q, 2)
        out = torch.empty_like(v)
        if p == 1:                            # X: (a, b) -> (b, a)
            out[:, 0] = v[:, 1]
            out[:, 1] = v[:, 0]
        elif p == 2:                          # Y: (a, b) -> (-i*b, i*a)
            out[:, 0, :, 0] = v[:, 1, :, 1]
            out[:, 0, :, 1] = -v[:, 1, :, 0]
            out[:, 1, :, 0] = -v[:, 0, :, 1]
            out[:, 1, :, 1] = v[:, 0, :, 0]
        else:                                 # Z: (a, b) -> (a, -b)
            out[:, 0] = v[:, 0]
            out[:, 1] = -v[:, 1]
        phi = out.reshape(phi.shape)
    return phi


def _re_inner(x, y):
    """Re<x|y> = sum(x_re*y_re + x_im*y_im)。"""
    return (x[:, 0] * y[:, 0] + x[:, 1] * y[:, 1]).sum()


def _statevector_adjoint_gradient_core(state, circuit, thetas, pauli_obs, compute_dtype):
    """核心计算：归一化 -> 前向电路 -> 期望值 -> 伴随法逆序梯度。

    返回 (expval [1], grads [T])，精度为 compute_dtype。
    """
    dim = state.shape[0]
    n = dim.bit_length() - 1                  # dim = 2^n
    num_theta = thetas.shape[0]
    num_gates = circuit.shape[0]

    c = circuit.to(torch.int64)
    gate_v = (c[:, 0] % 5).tolist()
    qa_v = (c[:, 1] % n).tolist()
    qb_v = (c[:, 2] % n).tolist()
    tid_v = (c[:, 3] % num_theta).tolist()
    theta_v = [float(t) for t in thetas.to(compute_dtype).tolist()]
    pauli_v = (pauli_obs.to(torch.int64) % 4).tolist()

    # 1) 归一化（算子语义：任意随机输入态合法）
    psi = state.to(compute_dtype)
    norm = torch.sqrt((psi * psi).sum())
    psi = psi / (norm + 1e-12)

    # 2) 前向作用全部门
    for g in range(num_gates):
        psi = _apply_gate(psi, n, gate_v[g], qa_v[g], qb_v[g], theta_v[tid_v[g]])

    # 3) |phi> = O |psi>，expval = Re<phi|psi>
    phi = _apply_pauli(psi, n, pauli_v)
    expval = _re_inner(phi, psi).reshape(1)

    # 4) 伴随法逆序回放
    grads = torch.zeros(num_theta, dtype=compute_dtype, device=state.device)
    for g in range(num_gates - 1, -1, -1):
        gate, qa, qb, tid = gate_v[g], qa_v[g], qb_v[g], tid_v[g]
        theta = theta_v[tid]
        psi = _apply_gate_dagger(psi, n, gate, qa, qb, theta)   # 回到门 g 之前的态
        if gate == 1:
            dpsi = _apply_1q(psi, n, qa, _drz_mat(theta))
            grads[tid] = grads[tid] + 2.0 * _re_inner(phi, dpsi)
        elif gate == 2:
            dpsi = _apply_1q(psi, n, qa, _drx_mat(theta))
            grads[tid] = grads[tid] + 2.0 * _re_inner(phi, dpsi)
        phi = _apply_gate_dagger(phi, n, gate, qa, qb, theta)
    return expval, grads


def statevector_adjoint_gradient(
    state: torch.Tensor,
    circuit: torch.Tensor,
    thetas: torch.Tensor,
    pauli_obs: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    量子态矢电路模拟 + 伴随法梯度 golden reference（plain golden = bench：fp32 计算）

    Args:
        state: [2^n, 2] float32 输入态（实部/虚部分离，算子内部归一化），n 由 shape 推出
        circuit: [G, 4] int32 电路指令，逐行取模译码（任意非负整数合法，
            评测取值范围 [0, 1073741823]）
        thetas: [T] float32 旋转门参数（评测取值范围 [-3.14159, 3.14159]），
            多条指令可共享同一参数（梯度累加）
        pauli_obs: [n] int32 张量积可观测量（0 I、1 X、2 Y、3 Z，评测取值范围 [0, 3]）

    Returns:
        expval: [1] float32 期望值 <psi_G|O|psi_G>
        grads: [T] float32 伴随法梯度 d expval / d thetas
    """
    expval, grads = _statevector_adjoint_gradient_core(
        state, circuit, thetas, pauli_obs, torch.float32)
    return expval.to(state.dtype), grads.to(state.dtype)


def statevector_adjoint_gradient_oracle(
    state: torch.Tensor,
    circuit: torch.Tensor,
    thetas: torch.Tensor,
    pauli_obs: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Oracle (g)：dtype-agnostic，计算精度跟随 state（fp64_cpu 下为 fp64 真值），不硬编码 .float()。"""
    return _statevector_adjoint_gradient_core(
        state, circuit, thetas, pauli_obs, state.dtype)
```

## 6. 额外信息

### 算子调用示例

```python
import torch

n, G, T = 16, 128, 32

state = torch.rand(1 << n, 2, dtype=torch.float32, device="npu") * 2 - 1
circuit = torch.randint(0, 1073741824, (G, 4), dtype=torch.int32, device="npu")
thetas = torch.empty(T, dtype=torch.float32, device="npu").uniform_(-3.14159, 3.14159)
pauli_obs = torch.randint(0, 4, (n,), dtype=torch.int32, device="npu")

expval, grads = statevector_adjoint_gradient(state, circuit, thetas, pauli_obs)
# expval.shape: [1]，grads.shape: [T]
```

### 可用于自检的性质（均已数值验证）

- **全 I 可观测量**：expval = 1（归一化的直接后果），grads = 0
- **酉性**：任意电路作用后 $\lVert\psi\rVert = 1$；$H^2 = \mathrm{CNOT}^2 = \mathrm{CZ}^2 = I$
- **无参数电路**（仅 H/CNOT/CZ）：grads 恒为全 0
- **对易性**：RZ-only 电路 + 全 Z 可观测量时全部对角矩阵对易，expval 与 $\theta$ 无关，grads = 0
- **共享 tid 可加性**：两门共享 $tid$ 的梯度 = 拆成两个独立 $tid$（同值）后两分量之和
- **梯度对照**：grads 与中心差分 $(E(\theta+\varepsilon)-E(\theta-\varepsilon))/2\varepsilon$ 一致

### 参考文献

- Jones, T., Gacon, J. (2020). "Efficient calculation of gradients in classical simulations of variational quantum algorithms". arXiv:2009.02823（伴随法梯度算法来源）
- Bergholm, V. et al. (2018). "PennyLane: Automatic differentiation of hybrid quantum-classical computations". arXiv:1811.04968（lightning.qubit adjoint 后端）
- NVIDIA cuQuantum SDK: cuStateVec adjoint expectation gradients（工程实现先例）
