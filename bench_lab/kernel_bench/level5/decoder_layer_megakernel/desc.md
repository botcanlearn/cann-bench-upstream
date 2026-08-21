# DecoderLayerMegakernel 算子 API 描述

## 1. 算子简介

单 token 解码整层融合算子（megakernel / superkernel 形态，Mirage MPK 与 Hazy Research megakernel 的算子化）：把标准 pre-norm decoder 层的一步解码（$S_q = 1$）——RMSNorm → QKV 投影 → RoPE → KV cache 追加 → 变长 GQA attention → 输出投影 → 残差 → RMSNorm → SwiGLU MLP → 残差——在**一个 kernel** 内完成。每个 batch 样本 $b$ 携带各自的历史长度 `cache_len[b]`，本步把新 token 的 key/value 写入槽位 `cache_len[b]` 并对前 `cache_len[b]+1` 个位置做因果 attention。

单 token 解码是 LLM 推理的延迟主导路径：每生成一个 token 都要把整层权重与 KV cache 从主存读一遍，计算强度极低（矩阵-向量乘），性能由带宽与 kernel 启动/同步开销决定。megakernel 路线（Mirage MPK、Hazy Research low-latency Llama）证明：把整层（乃至整网）融进单个持久 kernel、以细粒度任务图重叠各阶段的访存与计算，能把解码延迟压到接近硬件下界。

**主要应用场景**：
- LLaMA / Qwen 谱系 decoder-only LLM 的单 token 解码（latency-bound 在线推理）
- 低延迟 agent / 交互式服务（B 小、每 token 延迟敏感）的整层融合推理引擎
- megakernel 编译/运行时系统（Mirage MPK、Hazy Research megakernel）在 NPU 上的算子级对应物

**算子特征**：
- 难度等级：L5（FusedComposite）
- 十五输入（hidden、2×γ、7 个权重矩阵、2 个 cache、cache_len、RoPE cos/sin）三输出（y + 两个更新后的 cache）
- 融合 2×RMSNorm、7 次 matmul、RoPE、KV cache 原位追加、变长 GQA softmax attention、SwiGLU、2 处残差，共 9 类异构阶段
- GQA 头共享（Nkv=Nq 退化 MHA、Nkv=1 退化 MQA），逐样本变长 attention（cache_len 数据依赖）
- k_cache_out / v_cache_out 除写入槽位外**逐位等于输入**（可逐元素比对）

**为何是 L5**（约为 L4 FusedComposite 的 2 倍难度）：
- **长程工程一致性**：数学全为标准件（本文档 §2 给全，无任何新恒等式需要推导），难度在把 9 类异构阶段放进同一个 kernel——RMSNorm 的规约、7 个形状各异的 matmul、RoPE 逐元素旋转、cache 原位写、逐样本变长的 softmax attention、SwiGLU——共享片上驻留与流水编排：任何一处 tiling / 布局 / 同步的改动都会连锁破坏其余阶段的驻留假设，正确性与性能必须同时在整条链上成立
- **性能墙，而非数学墙**：$S_q = 1$ 解码是带宽受限的矩阵-向量工作负载，把 §2 直译成逐算子的朴素实现（约 11 次 kernel launch：2×RMSNorm、7 次 matmul、RoPE、cache 写、attention、SwiGLU、残差）完全正确、能通过全部精度用例，但每次 launch 的启动/同步开销与中间量往返主存使其只能锚定 baseline（预期得分 ~0.5 档）；要逼近 t_hw，必须单 kernel 化——权重/K V cache 的搬运与上一阶段的计算重叠、中间量全程驻留片上、阶段间以细粒度依赖而非全局同步衔接
- **变长负载不均**：各样本的 attention 长度 $L_b = \text{cache\_len}[b]+1$ 在运行时才知道且可以彼此悬殊（评测含全同与偏斜分布），静态均匀切分必然让部分核心空转，调度必须按数据依赖的负载划分
- **实现规格而非过可见测试**：隐藏评测集覆盖公开用例之外的规格维度（MHA/MQA、D=64/256、非 2 幂与素数 H/F/Smax、cache_len 边界 1 与 Smax−1、γ/RoPE 极值、特殊值输入等），逐条约定（eps 位置、头映射、RoPE 半维旋转、未写槽位逐位保持）都会被单独检验

## 2. 算子定义

### 记号

$B$ 为 batch，$H$ 为 hidden 维，$N_q$ / $N_{kv}$ 为 query / KV 头数（$N_q \bmod N_{kv} = 0$，组大小 $\text{grp} = N_q / N_{kv}$），$D$ 为头维（偶数），$F$ 为 MLP 中间维，$S_{max}$ 为 cache 容量。query 头 $n$（$0 \le n < N_q$）共享 KV 头

$$
g(n) = \lfloor n / \text{grp} \rfloor .
$$

每个样本 $b$ 独立计算。以下向量均为行向量，$x_b \in \mathbb{R}^{H}$ 为样本 $b$ 的输入 hidden（$x$ 的 $[b, 0, :]$）。

### 数学定义

**(1) attention 前 RMSNorm**（沿 $H$，eps 加在均方内）：

$$
h_b = \frac{x_b}{\sqrt{\tfrac{1}{H}\sum_{i=1}^{H} x_{b,i}^2 + \varepsilon}} \odot \gamma_1
$$

**(2) QKV 投影**（权重列按头拼接：头 $n$ 占 `wq` 的列 $nD .. (n{+}1)D{-}1$，KV 头同理）：

$$
q_b = h_b\, W_q \in \mathbb{R}^{N_q D}, \qquad
k^{new}_b = h_b\, W_k \in \mathbb{R}^{N_{kv} D}, \qquad
v^{new}_b = h_b\, W_v \in \mathbb{R}^{N_{kv} D}
$$

按头切分记 $q_{b,n}, k^{new}_{b,g}, v^{new}_{b,g} \in \mathbb{R}^{D}$。

**(3) RoPE**（半维旋转，作用于 $q$ 与 $k^{new}$ 的每个头，不作用于 $v^{new}$）。对向量 $t \in \mathbb{R}^{D}$，记 $t = [t^{(1)}, t^{(2)}]$（前后各 $D/2$ 维），旋转向量 $\mathrm{rot}(t) = [-t^{(2)}, t^{(1)}]$：

$$
\mathrm{RoPE}(t) = t \odot \cos_b + \mathrm{rot}(t) \odot \sin_b
$$

其中 $\cos_b, \sin_b \in \mathbb{R}^{D}$ 为输入 `rope_cos[b]` / `rope_sin[b]`（已按样本 $b$ 的当前位置索引好，同一样本的所有头共用）。

**(4) KV cache 追加**（写入槽位 $p_b = \text{cache\_len}[b]$）：

$$
\text{k\_cache\_out}[b, g, t, :] =
\begin{cases}
\mathrm{RoPE}(k^{new}_{b,g}) & t = p_b \\
\text{k\_cache}[b, g, t, :] & t \neq p_b
\end{cases}
\qquad (\text{v 同，写入 } v^{new}_{b,g}\text{，不做 RoPE})
$$

**未写槽位（$t \neq p_b$，含 $t > p_b$ 的尾部）逐位保持输入值**，保证输出 cache 可与输入逐元素比对。

**(5) 变长 GQA attention**（有效长度 $L_b = p_b + 1$，缩放 $D^{-1/2}$）：对每个 query 头 $n$，

$$
s_{n,t} = \frac{\mathrm{RoPE}(q_{b,n}) \cdot \text{k\_cache\_out}[b, g(n), t, :]}{\sqrt{D}}, \quad t = 0, \dots, L_b - 1
$$

$$
p_{n,t} = \frac{e^{s_{n,t}}}{\sum_{t'=0}^{L_b-1} e^{s_{n,t'}}}, \qquad
o_{b,n} = \sum_{t=0}^{L_b-1} p_{n,t}\; \text{v\_cache\_out}[b, g(n), t, :] \in \mathbb{R}^{D}
$$

合并各头并做输出投影：

$$
\text{attn}_b = [\,o_{b,0}, o_{b,1}, \dots, o_{b,N_q-1}\,]\; W_o \in \mathbb{R}^{H}
$$

**(6) attention 残差**：

$$
x^{(2)}_b = x_b + \text{attn}_b
$$

**(7) MLP 前 RMSNorm + SwiGLU + 残差**（$\mathrm{SiLU}(z) = z \cdot \sigma(z)$，$\sigma$ 为 sigmoid）：

$$
h^{(2)}_b = \frac{x^{(2)}_b}{\sqrt{\tfrac{1}{H}\sum_i (x^{(2)}_{b,i})^2 + \varepsilon}} \odot \gamma_2, \qquad
y_b = x^{(2)}_b + \Big( \mathrm{SiLU}\big(h^{(2)}_b W_{gate}\big) \odot \big(h^{(2)}_b W_{up}\big) \Big) W_{down}
$$

**本算子的精确约定**：
- 两处 RMSNorm 的 eps 均加在**均方内**（$x / \sqrt{\mathrm{mean}(x^2) + \varepsilon}$），不是加在 rms 上
- RoPE 只作用于 $q$ 与 $k^{new}$；`rope_cos` / `rope_sin` 是普通输入张量，**不要求**满足 $\cos^2 + \sin^2 = 1$（评测含任意取值）
- 头映射固定为 $g(n) = \lfloor n / (N_q / N_{kv}) \rfloor$；$N_{kv} = N_q$ 退化为 MHA，$N_{kv} = 1$ 退化为 MQA
- attention 只读位置 $0 .. L_b - 1$；$t > p_b$ 的槽位既不参与计算也不被修改
- softmax 分母至少含 1 项（$L_b \ge 2$，由 $\text{cache\_len} \ge 1$ 保证），无空行
- `epsilon` 为编译期标量属性，默认 $10^{-6}$

## 3. 接口规范

### 算子原型

```python
decoder_layer_megakernel(Tensor x, Tensor gamma1, Tensor wq, Tensor wk, Tensor wv, Tensor wo, Tensor gamma2, Tensor w_gate, Tensor w_up, Tensor w_down, Tensor k_cache, Tensor v_cache, Tensor cache_len, Tensor rope_cos, Tensor rope_sin, float epsilon=1e-6) -> (Tensor y, Tensor k_cache_out, Tensor v_cache_out)
```

### 输入参数说明

| 参数 | 类型 | 必需 | dtype | shape | 描述 |
|------|------|------|-------|-------|------|
| x | Tensor | 是 | float32/float16/bfloat16 | [B, 1, H] | 本步输入 hidden states（单 token 解码） |
| gamma1 | Tensor | 是 | 与 x 一致 | [H] | attention 前 RMSNorm 的 γ |
| wq | Tensor | 是 | 与 x 一致 | [H, Nq*D] | query 投影权重，列按头拼接 |
| wk | Tensor | 是 | 与 x 一致 | [H, Nkv*D] | key 投影权重，列按 KV 头拼接 |
| wv | Tensor | 是 | 与 x 一致 | [H, Nkv*D] | value 投影权重，列按 KV 头拼接 |
| wo | Tensor | 是 | 与 x 一致 | [Nq*D, H] | attention 输出投影权重 |
| gamma2 | Tensor | 是 | 与 x 一致 | [H] | MLP 前 RMSNorm 的 γ |
| w_gate | Tensor | 是 | 与 x 一致 | [H, F] | SwiGLU gate 投影权重 |
| w_up | Tensor | 是 | 与 x 一致 | [H, F] | SwiGLU up 投影权重 |
| w_down | Tensor | 是 | 与 x 一致 | [F, H] | SwiGLU down 投影权重 |
| k_cache | Tensor | 是 | 与 x 一致 | [B, Nkv, Smax, D] | key cache，有效位置 0..cache_len[b]-1 |
| v_cache | Tensor | 是 | 与 x 一致 | [B, Nkv, Smax, D] | value cache（同上） |
| cache_len | Tensor | 是 | int32 | [B] | 各样本已有的有效 cache 长度 ∈ [1, Smax-1]（由评测 value_range 保证） |
| rope_cos | Tensor | 是 | 与 x 一致 | [B, D] | RoPE 余弦（已按各样本当前位置索引好） |
| rope_sin | Tensor | 是 | 与 x 一致 | [B, D] | RoPE 正弦（同上） |
| epsilon | float | 否 | - | 标量 | 两处 RMSNorm 的 epsilon，默认 1e-6 |

### 输出

| 名称 | dtype | shape | 描述 |
|------|-------|-------|------|
| y | 与 x 一致 | [B, 1, H] | 本层输出 hidden states |
| k_cache_out | 与 k_cache 一致 | [B, Nkv, Smax, D] | 追加后的 key cache，除写入槽位外逐位等于输入 |
| v_cache_out | 与 v_cache 一致 | [B, Nkv, Smax, D] | 追加后的 value cache，除写入槽位外逐位等于输入 |

### 数据类型

| x / 权重 / cache / rope dtype | cache_len dtype | 输出 dtype | 内部计算 |
|-------------------------------|-----------------|-----------|----------|
| bfloat16 | int32 | bfloat16 | fp32 |
| float16 | int32 | float16 | fp32 |
| float32 | int32 | float32 | fp32 |

### 规则与约束

- 除 cache_len（恒 int32）外，全部 Tensor 输入 dtype 必须一致
- 维度一致性：wq/wk/wv 共享行数 H；wq 列数 = Nq*D、wk/wv 列数 = Nkv*D；wo 为 [Nq*D, H]；w_gate/w_up 为 [H, F]、w_down 为 [F, H]；k_cache 与 v_cache shape 完全一致
- $N_q \bmod N_{kv} = 0$（GQA；$N_{kv} = N_q$ 即 MHA、$N_{kv} = 1$ 即 MQA）；$D$ 为偶数（RoPE 半维旋转）
- cache_len ∈ [1, Smax-1] 由 cases 的 value_range 保证（写入槽位恒合法、softmax 无空行）；算子不对违反此约定的输入负责
- rope_cos/rope_sin 为任意取值的普通输入（不要求 cos²+sin²=1）
- 输出须为 contiguous 张量；k_cache_out/v_cache_out 未写槽位逐位等于输入
- 中间计算以 fp32 累加（低精度输入场景下这是精度达标的前提，见 §4）

### 支持范围

| 维度 / 参数 | 范围 | 备注 |
|---|---|---|
| `B`（batch） | 1 ~ 64 | cases.csv 实测 1 ~ 64 |
| `H`（hidden 维） | 8 ~ 8192 | cases.csv 实测 8 ~ 4096（LLaMA/Qwen 量级 2048/4096），含非 2 幂与素数 |
| `Nq`（query 头数） | 1 ~ 64 | cases.csv 实测 1 ~ 32 |
| `Nkv`（KV 头数） | 1 ~ Nq | cases.csv 实测 1 ~ 32（MQA/GQA/MHA 均覆盖），Nq % Nkv == 0 |
| `D`（头维，偶数） | 8 ~ 256 | cases.csv 实测 8 ~ 256，含非 2 幂 |
| `F`（MLP 中间维） | 16 ~ 32768 | cases.csv 实测 16 ~ 14336，含素数 |
| `Smax`（cache 容量） | 2 ~ 65536 | cases.csv 实测 2 ~ 16384，含素数与非 2 幂 |
| `cache_len` 取值 | [1, Smax-1] | 逐样本随机；边界（全 1、全 Smax-1、全同、偏斜）均覆盖 |
| dtype | bfloat16 / float16 / float32 | cases.csv 实测三种均覆盖 |
| `x` / cache 取值 | [-1, 1] | 常规随机范围（特殊值用例除外） |
| `gamma1` / `gamma2` 取值 | [0.5, 1.5] | 极值用例含 [0.001, 0.01] 与 [50, 100] |
| 权重取值 | [-0.02, 0.02] | 约 1/√H 量级，保证 matmul 输出 O(1) |
| `rope_cos` / `rope_sin` 取值 | [-1, 1] | 任意值合法（含 [-2, 2] 与全 0 用例） |
| `epsilon` | > 0 | cases.csv 实测 1e-6 / 1e-5 / 1e-3 |

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

阈值经评测框架 checker 实测确定：plain golden（fp32 数据通路）对 fp64 真值的误差下限，在公开 case 规模（H=4096/F=14336/Smax=4096 与 H=2048/F=5504/Smax=2048）下为 bf16 y.MERE≈1e-6~3e-6 / y.MARE≈4.7e-3~7.6e-3，fp16 MERE≈9e-7~7e-6 / MARE≈2.5e-3~4.2e-3，fp32 MERE≈3e-7~5e-6 / MARE≈2.5e-5~6.0e-4。MARE 尖峰位于两处残差与 softmax 加权和的相消位置：fp16/fp32 的实测 MARE 下限距默认阈值的 MARE 上限仅约 2 倍，换一种求和顺序的正确实现没有余量，故放宽为下表（MARE 上限为实测下限的 12~16 倍），与同类先例 flash_attention_backward（bf16 0.01 / fp32 0.001）一致：

| 数据类型 | FLOAT32 | FLOAT16 | BFLOAT16 |
|----------|---------|---------|----------|
| **通过阈值(Threshold)** | 0.001 | 0.005 | 0.01 |

当平均相对误差 MERE < Threshold，最大相对误差 MARE < 10 * Threshold 时判定为通过。小值域与相消场景由评测框架的兜底标准处理。

**结构性要求**：
- k_cache_out / v_cache_out 的未写槽位（$t \neq \text{cache\_len}[b]$）**应逐位等于输入**（实测 plain golden 该两输出 MERE ≤ 2.3e-9；未写槽位被改动会被 MARE 上限直接判失败，已负测试验证）。含 NaN 的特殊值用例按评测框架标准以 NaN 位置比对（NaN 位置须一致，payload 位不要求）
- **达标前提**：中间计算以 fp32 累加。golden 的 fp32 数据通路在写入槽位上先以 fp32 值参与 attention、最终按输出 dtype 舍入存储；kernel 若读回已舍入的槽位值，其差异在上表阈值内

## 5. 标准 Golden 代码

```python
import torch
from typing import Tuple


def _decoder_layer_megakernel_core(x, gamma1, wq, wk, wv, wo, gamma2, w_gate, w_up, w_down,
                                   k_cache, v_cache, cache_len, rope_cos, rope_sin,
                                   epsilon, compute_dtype):
    """核心计算：以 compute_dtype 精度执行整层解码，返回 (y, k_cache_out, v_cache_out)。"""
    Bsz, _, H = x.shape
    Nkv, Smax, D = k_cache.shape[1], k_cache.shape[2], k_cache.shape[3]
    Nq = wq.shape[1] // D

    x_f = x.to(compute_dtype)
    g1_f = gamma1.to(compute_dtype)
    g2_f = gamma2.to(compute_dtype)
    wq_f = wq.to(compute_dtype)
    wk_f = wk.to(compute_dtype)
    wv_f = wv.to(compute_dtype)
    wo_f = wo.to(compute_dtype)
    wg_f = w_gate.to(compute_dtype)
    wu_f = w_up.to(compute_dtype)
    wd_f = w_down.to(compute_dtype)
    cos_f = rope_cos.to(compute_dtype)
    sin_f = rope_sin.to(compute_dtype)
    kc = k_cache.to(compute_dtype).clone()                          # [B, Nkv, Smax, D]
    vc = v_cache.to(compute_dtype).clone()

    # 1. RMSNorm（沿 H，eps 加在均方内）
    h = x_f / torch.sqrt((x_f * x_f).mean(dim=-1, keepdim=True) + epsilon) * g1_f

    # 2. QKV 投影
    q = torch.matmul(h, wq_f).reshape(Bsz, 1, Nq, D)                # [B, 1, Nq, D]
    k_new = torch.matmul(h, wk_f).reshape(Bsz, 1, Nkv, D)           # [B, 1, Nkv, D]
    v_new = torch.matmul(h, wv_f).reshape(Bsz, 1, Nkv, D)           # [B, 1, Nkv, D]

    # 3. q / k_new 施加 RoPE（半维旋转，cos/sin 已按各样本当前位置索引好）
    def _rope(t):
        t1, t2 = t.chunk(2, dim=-1)
        rot = torch.cat([-t2, t1], dim=-1)
        return t * cos_f[:, None, None, :] + rot * sin_f[:, None, None, :]

    q = _rope(q)
    k_new = _rope(k_new)

    # 4 + 5. cache 追加 + 变长 GQA attention（逐 b：各样本有效长度不同）
    grp = Nq // Nkv
    scale = 1.0 / float(D) ** 0.5
    attn = torch.empty(Bsz, 1, Nq * D, dtype=compute_dtype, device=x.device)
    for b in range(Bsz):
        pos = int(cache_len[b])
        kc[b, :, pos, :] = k_new[b, 0]                              # 写入槽位 cache_len[b]
        vc[b, :, pos, :] = v_new[b, 0]
        lb = pos + 1                                                # 有效长度 L_b
        k_act = kc[b, :, :lb, :]                                    # [Nkv, L_b, D]
        v_act = vc[b, :, :lb, :]
        q_b = q[b, 0].reshape(Nkv, grp, D)                          # 头 n 的 KV 头 g = n // grp
        scores = torch.matmul(q_b, k_act.transpose(-1, -2)) * scale  # [Nkv, grp, L_b]
        probs = torch.softmax(scores, dim=-1)
        attn[b, 0] = torch.matmul(probs, v_act).reshape(Nq * D)     # [Nkv, grp, D] → [Nq*D]
    attn_proj = torch.matmul(attn, wo_f)                            # [B, 1, H]

    # 6. 残差
    x2 = x_f + attn_proj

    # 7. RMSNorm + SwiGLU MLP + 残差
    h2 = x2 / torch.sqrt((x2 * x2).mean(dim=-1, keepdim=True) + epsilon) * g2_f
    gate = torch.matmul(h2, wg_f)
    mlp = torch.matmul(gate * torch.sigmoid(gate) * torch.matmul(h2, wu_f), wd_f)
    y = x2 + mlp
    return y, kc, vc


def decoder_layer_megakernel(
    x: torch.Tensor,
    gamma1: torch.Tensor,
    wq: torch.Tensor,
    wk: torch.Tensor,
    wv: torch.Tensor,
    wo: torch.Tensor,
    gamma2: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cache_len: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    epsilon: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    DecoderLayerMegakernel golden reference（plain golden = bench：内部 fp32 计算）

    Args:
        x: [B, 1, H] 本步输入 hidden states
        gamma1: [H] attention 前 RMSNorm 的 γ
        wq: [H, Nq*D] query 投影权重
        wk: [H, Nkv*D] key 投影权重
        wv: [H, Nkv*D] value 投影权重
        wo: [Nq*D, H] attention 输出投影权重
        gamma2: [H] MLP 前 RMSNorm 的 γ
        w_gate: [H, F] SwiGLU gate 投影权重
        w_up: [H, F] SwiGLU up 投影权重
        w_down: [F, H] SwiGLU down 投影权重
        k_cache: [B, Nkv, Smax, D] key cache（本步写入槽位 cache_len[b]，其余槽位逐位保持）
        v_cache: [B, Nkv, Smax, D] value cache（同上）
        cache_len: [B] int32，各样本已有的有效 cache 长度 ∈ [1, Smax-1]（由 value_range 保证）
        rope_cos: [B, D] RoPE 余弦（已按各样本当前位置索引好）
        rope_sin: [B, D] RoPE 正弦（已按各样本当前位置索引好）
        epsilon: 两处 RMSNorm 的 epsilon，默认 1e-6

    Returns:
        y: [B, 1, H] 本层输出 hidden states，dtype 与 x 一致
        k_cache_out: [B, Nkv, Smax, D] 追加后的 key cache，shape/dtype 与 k_cache 一致
        v_cache_out: [B, Nkv, Smax, D] 追加后的 value cache，shape/dtype 与 v_cache 一致
    """
    y, kc, vc = _decoder_layer_megakernel_core(
        x, gamma1, wq, wk, wv, wo, gamma2, w_gate, w_up, w_down,
        k_cache, v_cache, cache_len, rope_cos, rope_sin, epsilon, torch.float32)
    return y.to(x.dtype), kc.to(k_cache.dtype), vc.to(v_cache.dtype)


def decoder_layer_megakernel_oracle(
    x: torch.Tensor,
    gamma1: torch.Tensor,
    wq: torch.Tensor,
    wk: torch.Tensor,
    wv: torch.Tensor,
    wo: torch.Tensor,
    gamma2: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cache_len: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    epsilon: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Oracle (g)：dtype-agnostic，计算精度跟随输入（fp64_cpu 下为 fp64 真值），不硬编码 .float()。"""
    return _decoder_layer_megakernel_core(
        x, gamma1, wq, wk, wv, wo, gamma2, w_gate, w_up, w_down,
        k_cache, v_cache, cache_len, rope_cos, rope_sin, epsilon, x.dtype)
```

## 6. 额外信息

### 算子调用示例

```python
import torch

B, H, Nq, Nkv, D, F, Smax = 4, 4096, 32, 8, 128, 14336, 2048
dt = torch.bfloat16

x = torch.randn(B, 1, H, dtype=dt, device="npu")
gamma1 = torch.ones(H, dtype=dt, device="npu")
gamma2 = torch.ones(H, dtype=dt, device="npu")
wq = torch.randn(H, Nq * D, dtype=dt, device="npu") * 0.02
wk = torch.randn(H, Nkv * D, dtype=dt, device="npu") * 0.02
wv = torch.randn(H, Nkv * D, dtype=dt, device="npu") * 0.02
wo = torch.randn(Nq * D, H, dtype=dt, device="npu") * 0.02
w_gate = torch.randn(H, F, dtype=dt, device="npu") * 0.02
w_up = torch.randn(H, F, dtype=dt, device="npu") * 0.02
w_down = torch.randn(F, H, dtype=dt, device="npu") * 0.02
k_cache = torch.randn(B, Nkv, Smax, D, dtype=dt, device="npu")
v_cache = torch.randn(B, Nkv, Smax, D, dtype=dt, device="npu")
cache_len = torch.randint(1, Smax, (B,), dtype=torch.int32, device="npu")
rope_cos = torch.randn(B, D, dtype=dt, device="npu").clamp(-1, 1)
rope_sin = torch.randn(B, D, dtype=dt, device="npu").clamp(-1, 1)

y, k_cache_out, v_cache_out = decoder_layer_megakernel(
    x, gamma1, wq, wk, wv, wo, gamma2, w_gate, w_up, w_down,
    k_cache, v_cache, cache_len, rope_cos, rope_sin, epsilon=1e-6)
# y.shape: [B, 1, H]；k_cache_out/v_cache_out: [B, Nkv, Smax, D]
```

### 退化与组合关系

- $N_{kv} = N_q$ 时 attention 退化为 MHA；$N_{kv} = 1$ 时退化为 MQA
- `rope_cos` 全 1、`rope_sin` 全 0 时 RoPE 退化为恒等映射（评测含使该退化不成立的任意 cos/sin 用例）
- 本算子即 LLaMA/Qwen 谱系 decoder 层的一步解码；沿层堆叠 L 次并配合 embedding / lm_head，即构成完整 decoder-only LLM 的单 token 前向（Mirage MPK 与 Hazy Research megakernel 融合的正是这一计算）

### 参考文献

- Mirage Team (2025). "Mirage Persistent Kernel: Compiling LLMs into a MegaKernel for Low-Latency Inference". arXiv:2512.22219（megakernel 形态出处：整网编译进单个持久 kernel、细粒度任务图重叠各阶段）
- Hazy Research (2025). "Look Ma, No Bubbles! Designing a Low-Latency Megakernel for Llama-1B". Stanford Hazy Research blog（megakernel 形态出处：整层/整网单 kernel 化消除 launch 气泡）
- Ainslie, J. et al. (2023). "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints". EMNLP 2023, arXiv:2305.13245（GQA 头共享）
- Su, J. et al. (2021). "RoFormer: Enhanced Transformer with Rotary Position Embedding". arXiv:2104.09864（RoPE）
- Shazeer, N. (2020). "GLU Variants Improve Transformer". arXiv:2002.05202（SwiGLU）
- Zhang, B., Sennrich, R. (2019). "Root Mean Square Layer Normalization". NeurIPS 2019（RMSNorm）
