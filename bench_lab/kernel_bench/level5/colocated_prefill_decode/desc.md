# ColocatedPrefillDecode 算子 API 描述

## 1. 算子简介

双模型同 die 共驻（colocated serving）融合算子：一次调用同时执行两个**相互独立**的负载——负载 A 是 prefill 模型的一个完整 pre-norm decoder 层（因果自注意力 + SwiGLU MLP，GEMM 密集），负载 B 是 decode 模型的单 token 解码层（KV cache 追加 + 变长 GQA attention，带宽/向量密集）。这一形态对应 LLM 推理系统中把 prefill 请求与 decode 请求混布到同一设备的主流做法（chunked prefill / hybrid batching / prefill-decode 融合 kernel）。

Ascend AI Core 的 CUBE 核（AIC）与 VECTOR 核（AIV）物理分离、各自拥有独立指令流（910B 上 AIC:AIV = 1:2）。本算子的两个负载资源画像恰好互补：负载 A 的耗时由矩阵乘吞吐（CUBE 峰值）决定，负载 B 的耗时由 KV cache 的搬运量（HBM 带宽）与向量计算决定。朴素地先算完 A 再算 B，两段时间相加；两个负载同时驻留、相互填充对方空闲的资源，总时间才可能逼近两段中较长的一段。本算子考核的正是这种**核间调度与资源共驻**能力。

现实原型：POD-Attention（prefill+decode 融合 attention kernel）、NanoFlow（设备内异构资源重叠调度）、Sarathi-Serve（chunked prefill 混布）；反面参照是 DistServe / Splitwise 一类把 prefill 与 decode 分离到不同设备的部署形态——分离部署回避了本算子考核的共驻问题，代价是跨设备搬运 KV cache。

**主要应用场景**：
- prefill/decode 混布推理服务（chunked prefill、hybrid batching）中，同一 die 上并发执行两类异构负载
- 双模型共驻场景（例如主模型 decode 与投机草稿模型 prefill、在线服务与后台摘要任务共卡）
- 推理引擎的设备内资源重叠调度（NanoFlow 形态）的算子级考核

**算子特征**：
- 难度等级：L5（FusedComposite）
- 27 输入（A 侧 10 + B 侧 13 + 双侧 RoPE 4）四输出（y_a, y_b, k_cache_out, v_cache_out）
- 两个负载均为标准 decoder 层结构件：RMSNorm、QKV 投影、RoPE、GQA attention、输出投影、残差、SwiGLU MLP
- **独立性契约**：y_a 仅由 A 侧输入决定；y_b / k_cache_out / v_cache_out 仅由 B 侧输入决定，共驻不得引入任何串扰

**为何是 L5**（约为 L4 FusedComposite 的 2 倍难度，性能墙而非数学墙）：
- **正确性人人可过**：两个负载全部由标准件组成，§2 给出全部公式、规格自足，逐条直译即可通过全部精度用例；正确性不是区分点，**区分度全部在性能分**
- **性能墙**：朴素串行实现（先 A 后 B）耗时 ≈ t_A + t_B，只能锚定 baseline 水平（预期得分 ~0.5 档）；负载 A 计算密集、负载 B 带宽密集，资源画像互补，理想的共驻调度能把总时间压向 max(t_A, t_B)，这正是本算子 t_hw 的标定口径（见 §6）——不做真正的负载重叠，就摸不到性能墙
- **调度必须适应负载比**：评测用例覆盖从 t_A ≈ t_B（公开用例多数按此配平）到十倍级失衡（隐藏用例）的完整谱系；任何按固定比例静态划分核资源的方案，在失衡用例上会退化到与串行相当——一侧的核在另一侧完成后闲置
- **实现规格而非过可见测试**：隐藏评测集覆盖公开用例之外的规格维度（极端失衡、B_b=1、cache_len 全同与参差、A 侧 MHA 与 B 侧 MQA、素数 S_a、非对齐 F_a、H_a ≠ H_b 等），逐条约定（eps 位置、RoPE 半维旋转、GQA 头映射、cache 未写槽位逐位保持、独立性契约）都会被单独检验

## 2. 算子定义

### 记号与独立性契约

负载 A：x_a ∈ R^{B_a×S_a×H_a}，GQA 头数 Nq_a / Nkv_a（Nq_a % Nkv_a == 0），头维 D_a（偶数），MLP 中间维 F_a。
负载 B：x_b ∈ R^{B_b×1×H_b}，头数 Nq_b / Nkv_b（Nq_b % Nkv_b == 0），头维 D_b（偶数），MLP 中间维 F_b，cache 容量 Smax。
两个负载不共享任何输入张量、任何中间量：

$$
y_a = f_A(x_a, \gamma_{1a}, W_{*a}, \cos_a, \sin_a, \varepsilon), \qquad
(y_b, K_{\mathrm{out}}, V_{\mathrm{out}}) = f_B(x_b, \gamma_{*b}, W_{*b}, K, V, \mathrm{len}, \cos_b, \sin_b, \varepsilon)
$$

扰动任一 A 侧输入不得改变 y_b / k_cache_out / v_cache_out 的任何一个比特，反之亦然（评测含此项检验）。

### 公共构件

**RMSNorm**（eps 加在均方内，沿最后一维 H）：

$$
\mathrm{RMSNorm}(x, \gamma, \varepsilon) = \frac{x}{\sqrt{\mathrm{mean}(x^2) + \varepsilon}} \odot \gamma
$$

**RoPE 半维旋转**（对头维 D 的向量 t，cos/sin ∈ R^D 已按位置索引好）：

$$
t_1, t_2 = \mathrm{chunk}(t, 2, -1), \qquad
\mathrm{rot} = \mathrm{cat}(-t_2, t_1), \qquad
\mathrm{RoPE}(t) = t \odot \cos + \mathrm{rot} \odot \sin
$$

**GQA 头映射**：query 头 n（0 起）使用 KV 头 g = ⌊n / (Nq/Nkv)⌋。

**SwiGLU MLP**：mlp(h) = (SiLU(h @ w_gate) ⊙ (h @ w_up)) @ w_down，SiLU(z) = z·σ(z)。

### 负载 A（prefill 层，因果自注意力）

对每个样本 b（样本间独立）：

$$
h = \mathrm{RMSNorm}(x_a, \gamma_{1a}, \varepsilon)
$$

$$
q = \mathrm{reshape}(h W_{qa}) \in \mathbb{R}^{S_a \times N_{qa} \times D_a}, \quad
k = \mathrm{reshape}(h W_{ka}),\ v = \mathrm{reshape}(h W_{va}) \in \mathbb{R}^{S_a \times N_{kva} \times D_a}
$$

（reshape 按头切列：头 n 取列 n·D_a .. (n+1)·D_a−1。）q、k 逐位置施加 RoPE（位置 (b, s) 使用 rope_cos_a[b, s]、rope_sin_a[b, s]，广播到该位置的全部头；v 不加）。因果 attention，query 位置 i 只看 j ≤ i：

$$
\mathrm{score}^{(n)}_{ij} = \frac{q^{(n)}_i \cdot k^{(g)}_j}{\sqrt{D_a}} \ (j \le i), \qquad
\mathrm{attn}^{(n)}_i = \sum_{j \le i} \mathrm{softmax}_j\big(\mathrm{score}^{(n)}_{i\cdot}\big)\, v^{(g)}_j,
\qquad g = \lfloor n / (N_{qa}/N_{kva}) \rfloor
$$

合并头（按头序拼接回 [S_a, Nq_a·D_a]）后：

$$
x_2 = x_a + \mathrm{attn}\, W_{oa}, \qquad
h_2 = \mathrm{RMSNorm}(x_2, \gamma_{2a}, \varepsilon), \qquad
y_a = x_2 + \mathrm{mlp}(h_2)
$$

### 负载 B（decode 层，单 token + KV cache 追加）

对每个样本 b（样本间独立，有效 cache 长度 len_b = cache_len[b]）：

$$
h = \mathrm{RMSNorm}(x_b, \gamma_{1b}, \varepsilon), \qquad
q = \mathrm{RoPE}(\mathrm{reshape}(h W_{qb})), \quad
k_{\mathrm{new}} = \mathrm{RoPE}(\mathrm{reshape}(h W_{kb})), \quad
v_{\mathrm{new}} = \mathrm{reshape}(h W_{vb})
$$

（cos/sin 取 rope_cos_b[b]、rope_sin_b[b]，已按该样本当前位置索引好。）**cache 追加**：

$$
K_{\mathrm{out}}[b, :, \mathrm{len}_b, :] = k_{\mathrm{new}}[b], \qquad
V_{\mathrm{out}}[b, :, \mathrm{len}_b, :] = v_{\mathrm{new}}[b]
$$

其余槽位（含 len_b 之后的未使用区域）**逐位保持输入值**。变长 GQA attention 对追加后的前 L_b = len_b + 1 个位置计算（scale = D_b^{-1/2}，头映射同上），合并头后：

$$
x_2 = x_b + \mathrm{attn}\, W_{ob}, \qquad
h_2 = \mathrm{RMSNorm}(x_2, \gamma_{2b}, \varepsilon), \qquad
y_b = x_2 + \mathrm{mlp}(h_2)
$$

### 本算子的精确约定

- RMSNorm 的 eps 加在均方内（x / sqrt(mean(x²) + ε) · γ），不是加在 rms 上
- RoPE 为半维旋转（如上公式），要求 D_a、D_b 为偶数；A 侧 cos/sin 逐位置（[B_a, S_a, D_a]）、B 侧 cos/sin 逐样本（[B_b, D_b]），均已索引好、算子内不再按位置查表
- GQA 头映射 g = ⌊n / (Nq/Nkv)⌋；Nkv=Nq 退化为 MHA，Nkv=1 退化为 MQA（两侧独立支持）
- A 侧因果掩码：位置 i 严格只看 j ≤ i（含自身）；B 侧有效长度 L_b = cache_len[b] + 1，逐样本变长
- cache 写入槽位恒为 cache_len[b]（∈ [1, Smax−1] 由评测 value_range 保证，恒合法）；除写入槽位外 k_cache_out / v_cache_out 与输入**逐位相等**（含 dtype 精度不变，评测按输出精度阈值直接检验）
- 独立性契约（见上）：跨负载的任何数据流都是错误实现
- attention 的 softmax 沿键位置维计算；每行至少含一个有效位置（A 侧对角线、B 侧新写入槽位），不存在全掩码行

## 3. 接口规范

### 算子原型

```python
colocated_prefill_decode(Tensor x_a, Tensor gamma1_a, Tensor wq_a, Tensor wk_a, Tensor wv_a, Tensor wo_a, Tensor gamma2_a, Tensor w_gate_a, Tensor w_up_a, Tensor w_down_a, Tensor x_b, Tensor gamma1_b, Tensor wq_b, Tensor wk_b, Tensor wv_b, Tensor wo_b, Tensor gamma2_b, Tensor w_gate_b, Tensor w_up_b, Tensor w_down_b, Tensor k_cache, Tensor v_cache, Tensor cache_len, Tensor rope_cos_a, Tensor rope_sin_a, Tensor rope_cos_b, Tensor rope_sin_b, float epsilon=1e-6) -> (Tensor y_a, Tensor y_b, Tensor k_cache_out, Tensor v_cache_out)
```

### 输入参数说明

| 参数 | 类型 | 必需 | dtype | shape | 描述 |
|------|------|------|-------|-------|------|
| x_a | Tensor | 是 | float32/float16/bfloat16 | [B_a, S_a, H_a] | 负载 A（prefill 层）输入 hidden states，评测取值范围 [-1, 1] |
| gamma1_a | Tensor | 是 | float32/float16/bfloat16 | [H_a] | A 侧 attention 前 RMSNorm 的 γ1，评测取值范围 [0.5, 1.5] |
| wq_a | Tensor | 是 | float32/float16/bfloat16 | [H_a, Nq_a*D_a] | A 侧 query 投影权重，列按头拼接，评测取值范围 [-0.05, 0.05] |
| wk_a | Tensor | 是 | float32/float16/bfloat16 | [H_a, Nkv_a*D_a] | A 侧 key 投影权重，列按 KV 头拼接 |
| wv_a | Tensor | 是 | float32/float16/bfloat16 | [H_a, Nkv_a*D_a] | A 侧 value 投影权重 |
| wo_a | Tensor | 是 | float32/float16/bfloat16 | [Nq_a*D_a, H_a] | A 侧 attention 输出投影权重 |
| gamma2_a | Tensor | 是 | float32/float16/bfloat16 | [H_a] | A 侧 MLP 前 RMSNorm 的 γ2 |
| w_gate_a | Tensor | 是 | float32/float16/bfloat16 | [H_a, F_a] | A 侧 SwiGLU gate 投影权重 |
| w_up_a | Tensor | 是 | float32/float16/bfloat16 | [H_a, F_a] | A 侧 SwiGLU up 投影权重 |
| w_down_a | Tensor | 是 | float32/float16/bfloat16 | [F_a, H_a] | A 侧 SwiGLU down 投影权重 |
| x_b | Tensor | 是 | float32/float16/bfloat16 | [B_b, 1, H_b] | 负载 B（decode 层）输入 hidden states（单 token） |
| gamma1_b | Tensor | 是 | float32/float16/bfloat16 | [H_b] | B 侧 attention 前 RMSNorm 的 γ1 |
| wq_b | Tensor | 是 | float32/float16/bfloat16 | [H_b, Nq_b*D_b] | B 侧 query 投影权重 |
| wk_b | Tensor | 是 | float32/float16/bfloat16 | [H_b, Nkv_b*D_b] | B 侧 key 投影权重 |
| wv_b | Tensor | 是 | float32/float16/bfloat16 | [H_b, Nkv_b*D_b] | B 侧 value 投影权重 |
| wo_b | Tensor | 是 | float32/float16/bfloat16 | [Nq_b*D_b, H_b] | B 侧 attention 输出投影权重 |
| gamma2_b | Tensor | 是 | float32/float16/bfloat16 | [H_b] | B 侧 MLP 前 RMSNorm 的 γ2 |
| w_gate_b | Tensor | 是 | float32/float16/bfloat16 | [H_b, F_b] | B 侧 SwiGLU gate 投影权重 |
| w_up_b | Tensor | 是 | float32/float16/bfloat16 | [H_b, F_b] | B 侧 SwiGLU up 投影权重 |
| w_down_b | Tensor | 是 | float32/float16/bfloat16 | [F_b, H_b] | B 侧 SwiGLU down 投影权重 |
| k_cache | Tensor | 是 | float32/float16/bfloat16 | [B_b, Nkv_b, Smax, D_b] | key cache，本步写入槽位 cache_len[b]，其余槽位逐位保持 |
| v_cache | Tensor | 是 | float32/float16/bfloat16 | [B_b, Nkv_b, Smax, D_b] | value cache（追加语义同 k_cache） |
| cache_len | Tensor | 是 | int32 | [B_b] | 各样本已有的有效 cache 长度 ∈ [1, Smax-1]（由评测 value_range 保证） |
| rope_cos_a | Tensor | 是 | float32/float16/bfloat16 | [B_a, S_a, D_a] | A 侧 RoPE 余弦（已逐位置索引好） |
| rope_sin_a | Tensor | 是 | float32/float16/bfloat16 | [B_a, S_a, D_a] | A 侧 RoPE 正弦（已逐位置索引好） |
| rope_cos_b | Tensor | 是 | float32/float16/bfloat16 | [B_b, D_b] | B 侧 RoPE 余弦（已按各样本当前位置索引好） |
| rope_sin_b | Tensor | 是 | float32/float16/bfloat16 | [B_b, D_b] | B 侧 RoPE 正弦（已按各样本当前位置索引好） |
| epsilon | float | 否 | - | 标量 | 四处 RMSNorm 的 epsilon（加在均方内），默认 1e-6 |

### 输出

| 名称 | dtype | shape | 描述 |
|------|-------|-------|------|
| y_a | 与 x_a 一致 | [B_a, S_a, H_a] | 负载 A 输出，仅由 A 侧输入决定 |
| y_b | 与 x_b 一致 | [B_b, 1, H_b] | 负载 B 输出，仅由 B 侧输入决定 |
| k_cache_out | 与 k_cache 一致 | [B_b, Nkv_b, Smax, D_b] | 追加后的 key cache，除写入槽位外逐位等于输入 |
| v_cache_out | 与 v_cache 一致 | [B_b, Nkv_b, Smax, D_b] | 追加后的 value cache，除写入槽位外逐位等于输入 |

### 数据类型

| 浮点输入 dtype（A/B 两侧一致） | cache_len | 输出 dtype | 内部计算 |
|-------------------------------|-----------|-----------|----------|
| bfloat16 | int32 | bfloat16 | fp32 |
| float16 | int32 | float16 | fp32 |
| float32 | int32 | float32 | fp32 |

### 规则与约束

- 26 个浮点输入 dtype 必须一致（A/B 同 dtype）；cache_len 恒为 int32
- 维度一致性：Nq_a·D_a 由 wq_a 列数给出、Nkv_a·D_a 由 wk_a/wv_a 列数给出（D_a 取自 rope_cos_a 末维）；B 侧同理（D_b、Nkv_b、Smax 取自 k_cache）；Nq % Nkv == 0 两侧分别成立
- D_a、D_b 为偶数（RoPE 半维旋转）；H_a 与 H_b、D_a 与 D_b、F_a 与 F_b 允许不同，Nq·D 不要求等于 H
- cache_len ∈ [1, Smax-1] 由评测 value_range 保证；算子不对违反此约定的输入负责
- k_cache_out / v_cache_out 除写入槽位外必须逐位保持输入值（不得整体重量化/改写）
- 独立性契约：跨负载不得有任何数据依赖
- 输出须为 contiguous 张量

### 支持范围

| 维度 / 参数 | 范围 | 备注 |
|---|---|---|
| `B_a` | 1 ~ 8 | cases.csv 实测 1 ~ 4 |
| `S_a` | 8 ~ 8192 | cases.csv 实测 512 ~ 2048（隐藏用例含 4096 与素数） |
| `H_a` / `H_b` | 32 ~ 4096 | cases.csv 实测 1024 / 2048 |
| `Nq_a` / `Nq_b` | 2 ~ 32 | cases.csv 实测 8 / 16 |
| `Nkv_a` / `Nkv_b` | 1 ~ Nq | cases.csv 实测 1 ~ 8（MQA/GQA/MHA 均覆盖） |
| `D_a` / `D_b` | 8 ~ 128（偶数） | cases.csv 实测 64 / 128 |
| `F_a` / `F_b` | 64 ~ 14336 | cases.csv 实测 2752 / 5504（≈2.7×H），隐藏用例含非对齐值 |
| `B_b` | 1 ~ 64 | cases.csv 实测 8 ~ 64 |
| `Smax` | 8 ~ 8192 | cases.csv 实测 2048 ~ 8192 |
| `cache_len` 取值 | [1, Smax-1] | 隐藏用例含全同与窄带分布 |
| dtype | bfloat16 / float16 / float32 | cases.csv 实测三种均覆盖 |
| `x_a` / `x_b` / cache / rope 取值 | [-1, 1] | 常规随机范围 |
| `gamma*` 取值 | [0.5, 1.5] | 正缩放 |
| 投影权重取值 | [-0.05, 0.05] | ≈ 1/√H 初始化量级，保持各级激活 O(1) |

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

阈值经评测框架 checker 实测确定（plain golden 的 fp32 数据通路对 fp64 oracle 的误差下限，公开 case 量级：A 侧 H_a=1024/2048、S_a=512/1024，B 侧 H_b=1024/2048、Smax=1024/2048、B_b=8/16）：bf16 MERE≈5.6e-7~1.3e-6 / MARE≈7.8e-3，fp16 MERE≈8.6e-7~2.8e-6 / MARE≈9.8e-4~3.0e-3，fp32 MERE≈1.3e-6~2.5e-6 / MARE≈1.1e-4~1.2e-3。MARE 尖峰位于两处残差与 softmax 加权和的相消位置，小值域与相消场景由评测框架的兜底标准处理。为换一种求和顺序的正确实现留出余量，取：

| 数据类型 | FLOAT32 | FLOAT16 | BFLOAT16 |
|----------|---------|---------|----------|
| **通过阈值(Threshold)** | 0.001 | 0.005 | 0.01 |

当平均相对误差 MERE < Threshold，最大相对误差 MARE < 10 * Threshold 时判定为通过（与同类先例 decoder_layer_megakernel 一致）。四个输出（y_a、y_b、k_cache_out、v_cache_out）逐一按上述标准检验：k_cache_out / v_cache_out 除写入槽位外逐位等于输入，未写槽位被改动会被 MARE 上限直接判失败；y_a 与 y_b 独立判定，任一负载出错即整体失败。内部计算须以 fp32（或更高）精度累加，低精度输入场景下直接用 bf16/fp16 累加长归约（S_a 维 softmax 加权和、F 维 down 投影）无法达标。

## 5. 标准 Golden 代码

```python
import torch
from typing import Tuple


def _colocated_prefill_decode_core(x_a, gamma1_a, wq_a, wk_a, wv_a, wo_a, gamma2_a,
                                   w_gate_a, w_up_a, w_down_a,
                                   x_b, gamma1_b, wq_b, wk_b, wv_b, wo_b, gamma2_b,
                                   w_gate_b, w_up_b, w_down_b,
                                   k_cache, v_cache, cache_len,
                                   rope_cos_a, rope_sin_a, rope_cos_b, rope_sin_b,
                                   epsilon, compute_dtype):
    """核心计算：以 compute_dtype 精度独立执行负载 A 与负载 B，返回 (y_a, y_b, k_cache_out, v_cache_out)。"""

    def _rms_norm(t, gamma):
        return t / torch.sqrt((t * t).mean(dim=-1, keepdim=True) + epsilon) * gamma

    def _rope(t, cos, sin):
        t1, t2 = t.chunk(2, dim=-1)
        rot = torch.cat([-t2, t1], dim=-1)
        return t * cos + rot * sin

    # ===== 负载 A：prefill 层（因果自注意力，GQA，仅消费 A 侧输入）=====
    b_a, s_a, _ = x_a.shape
    d_a = rope_cos_a.shape[-1]
    nq_a = wq_a.shape[1] // d_a
    nkv_a = wk_a.shape[1] // d_a

    xa = x_a.to(compute_dtype)
    g1a = gamma1_a.to(compute_dtype)
    g2a = gamma2_a.to(compute_dtype)
    cos_a = rope_cos_a.to(compute_dtype)[:, :, None, :]              # [B_a, S_a, 1, D_a]
    sin_a = rope_sin_a.to(compute_dtype)[:, :, None, :]

    # 1. RMSNorm（沿 H_a，eps 加在均方内）
    h = _rms_norm(xa, g1a)

    # 2 + 3. QKV 投影 + q/k 施加 RoPE（逐位置 cos/sin，广播到全部头）
    q = _rope(torch.matmul(h, wq_a.to(compute_dtype)).reshape(b_a, s_a, nq_a, d_a), cos_a, sin_a)
    k = _rope(torch.matmul(h, wk_a.to(compute_dtype)).reshape(b_a, s_a, nkv_a, d_a), cos_a, sin_a)
    v = torch.matmul(h, wv_a.to(compute_dtype)).reshape(b_a, s_a, nkv_a, d_a)

    # 4. 因果 GQA attention（逐 b：控制峰值内存，样本间独立）
    grp_a = nq_a // nkv_a
    scale_a = 1.0 / float(d_a) ** 0.5
    causal = torch.triu(torch.ones(s_a, s_a, dtype=torch.bool, device=x_a.device), diagonal=1)
    attn_a = torch.empty(b_a, s_a, nq_a * d_a, dtype=compute_dtype, device=x_a.device)
    for b in range(b_a):
        qh = q[b].reshape(s_a, nkv_a, grp_a, d_a).permute(1, 2, 0, 3)     # 头 n 的 KV 头 g = n // grp
        kh = k[b].permute(1, 0, 2)                                        # [Nkv_a, S_a, D_a]
        vh = v[b].permute(1, 0, 2)
        scores = torch.matmul(qh, kh.unsqueeze(1).transpose(-1, -2)) * scale_a   # [Nkv_a, grp, S_a, S_a]
        scores = scores.masked_fill(causal, float('-inf'))                # 位置 i 只看 j ≤ i
        probs = torch.softmax(scores, dim=-1)
        ctx = torch.matmul(probs, vh.unsqueeze(1))                        # [Nkv_a, grp, S_a, D_a]
        attn_a[b] = ctx.permute(2, 0, 1, 3).reshape(s_a, nq_a * d_a)

    # 5. 输出投影 + 残差
    x2 = xa + torch.matmul(attn_a, wo_a.to(compute_dtype))

    # 6. RMSNorm + SwiGLU MLP + 残差
    h2 = _rms_norm(x2, g2a)
    gate = torch.matmul(h2, w_gate_a.to(compute_dtype))
    y_a = x2 + torch.matmul(gate * torch.sigmoid(gate) * torch.matmul(h2, w_up_a.to(compute_dtype)),
                            w_down_a.to(compute_dtype))

    # ===== 负载 B：单 token decode 层（KV cache 追加 + 变长 GQA，仅消费 B 侧输入）=====
    b_b = x_b.shape[0]
    nkv_b, _, d_b = k_cache.shape[1], k_cache.shape[2], k_cache.shape[3]
    nq_b = wq_b.shape[1] // d_b

    xb = x_b.to(compute_dtype)
    g1b = gamma1_b.to(compute_dtype)
    g2b = gamma2_b.to(compute_dtype)
    cos_b = rope_cos_b.to(compute_dtype)[:, None, None, :]               # [B_b, 1, 1, D_b]
    sin_b = rope_sin_b.to(compute_dtype)[:, None, None, :]
    kc = k_cache.to(compute_dtype).clone()                               # [B_b, Nkv_b, Smax, D_b]
    vc = v_cache.to(compute_dtype).clone()

    # 1. RMSNorm + QKV 投影 + q/k_new 施加 RoPE
    hb = _rms_norm(xb, g1b)
    qb = _rope(torch.matmul(hb, wq_b.to(compute_dtype)).reshape(b_b, 1, nq_b, d_b), cos_b, sin_b)
    k_new = _rope(torch.matmul(hb, wk_b.to(compute_dtype)).reshape(b_b, 1, nkv_b, d_b), cos_b, sin_b)
    v_new = torch.matmul(hb, wv_b.to(compute_dtype)).reshape(b_b, 1, nkv_b, d_b)

    # 2 + 3. cache 追加 + 变长 GQA attention（逐 b：各样本有效长度不同）
    grp_b = nq_b // nkv_b
    scale_b = 1.0 / float(d_b) ** 0.5
    attn_b = torch.empty(b_b, 1, nq_b * d_b, dtype=compute_dtype, device=x_b.device)
    for b in range(b_b):
        pos = int(cache_len[b])
        kc[b, :, pos, :] = k_new[b, 0]                                   # 写入槽位 cache_len[b]
        vc[b, :, pos, :] = v_new[b, 0]
        lb = pos + 1                                                     # 有效长度 L_b
        k_act = kc[b, :, :lb, :]                                         # [Nkv_b, L_b, D_b]
        v_act = vc[b, :, :lb, :]
        q_b = qb[b, 0].reshape(nkv_b, grp_b, d_b)                        # 头 n 的 KV 头 g = n // grp
        scores = torch.matmul(q_b, k_act.transpose(-1, -2)) * scale_b    # [Nkv_b, grp, L_b]
        probs = torch.softmax(scores, dim=-1)
        attn_b[b, 0] = torch.matmul(probs, v_act).reshape(nq_b * d_b)
    x2b = xb + torch.matmul(attn_b, wo_b.to(compute_dtype))

    # 4. RMSNorm + SwiGLU MLP + 残差
    h2b = _rms_norm(x2b, g2b)
    gate_b = torch.matmul(h2b, w_gate_b.to(compute_dtype))
    y_b = x2b + torch.matmul(gate_b * torch.sigmoid(gate_b) * torch.matmul(h2b, w_up_b.to(compute_dtype)),
                             w_down_b.to(compute_dtype))
    return y_a, y_b, kc, vc


def colocated_prefill_decode(
    x_a: torch.Tensor,
    gamma1_a: torch.Tensor,
    wq_a: torch.Tensor,
    wk_a: torch.Tensor,
    wv_a: torch.Tensor,
    wo_a: torch.Tensor,
    gamma2_a: torch.Tensor,
    w_gate_a: torch.Tensor,
    w_up_a: torch.Tensor,
    w_down_a: torch.Tensor,
    x_b: torch.Tensor,
    gamma1_b: torch.Tensor,
    wq_b: torch.Tensor,
    wk_b: torch.Tensor,
    wv_b: torch.Tensor,
    wo_b: torch.Tensor,
    gamma2_b: torch.Tensor,
    w_gate_b: torch.Tensor,
    w_up_b: torch.Tensor,
    w_down_b: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cache_len: torch.Tensor,
    rope_cos_a: torch.Tensor,
    rope_sin_a: torch.Tensor,
    rope_cos_b: torch.Tensor,
    rope_sin_b: torch.Tensor,
    epsilon: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    ColocatedPrefillDecode golden reference（plain golden = bench：内部 fp32 计算）

    Args:
        x_a: [B_a, S_a, H_a] 负载 A（prefill 层）输入 hidden states
        gamma1_a: [H_a] A 侧 attention 前 RMSNorm 的 γ
        wq_a: [H_a, Nq_a*D_a] A 侧 query 投影权重（列按头拼接，头 n 占列 n*D_a..(n+1)*D_a-1）
        wk_a: [H_a, Nkv_a*D_a] A 侧 key 投影权重（列按 KV 头拼接，Nq_a % Nkv_a == 0）
        wv_a: [H_a, Nkv_a*D_a] A 侧 value 投影权重
        wo_a: [Nq_a*D_a, H_a] A 侧 attention 输出投影权重
        gamma2_a: [H_a] A 侧 MLP 前 RMSNorm 的 γ
        w_gate_a: [H_a, F_a] A 侧 SwiGLU gate 投影权重
        w_up_a: [H_a, F_a] A 侧 SwiGLU up 投影权重
        w_down_a: [F_a, H_a] A 侧 SwiGLU down 投影权重
        x_b: [B_b, 1, H_b] 负载 B（decode 层）输入 hidden states（单 token，S_q = 1）
        gamma1_b: [H_b] B 侧 attention 前 RMSNorm 的 γ
        wq_b: [H_b, Nq_b*D_b] B 侧 query 投影权重
        wk_b: [H_b, Nkv_b*D_b] B 侧 key 投影权重（Nq_b % Nkv_b == 0）
        wv_b: [H_b, Nkv_b*D_b] B 侧 value 投影权重
        wo_b: [Nq_b*D_b, H_b] B 侧 attention 输出投影权重
        gamma2_b: [H_b] B 侧 MLP 前 RMSNorm 的 γ
        w_gate_b: [H_b, F_b] B 侧 SwiGLU gate 投影权重
        w_up_b: [H_b, F_b] B 侧 SwiGLU up 投影权重
        w_down_b: [F_b, H_b] B 侧 SwiGLU down 投影权重
        k_cache: [B_b, Nkv_b, Smax, D_b] key cache（本步写入槽位 cache_len[b]，其余槽位逐位保持）
        v_cache: [B_b, Nkv_b, Smax, D_b] value cache（同上）
        cache_len: [B_b] int32，各样本已有的有效 cache 长度 ∈ [1, Smax-1]（由 value_range 保证）
        rope_cos_a: [B_a, S_a, D_a] A 侧 RoPE 余弦（已逐位置索引好）
        rope_sin_a: [B_a, S_a, D_a] A 侧 RoPE 正弦（已逐位置索引好）
        rope_cos_b: [B_b, D_b] B 侧 RoPE 余弦（已按各样本当前位置索引好）
        rope_sin_b: [B_b, D_b] B 侧 RoPE 正弦（已按各样本当前位置索引好）
        epsilon: 四处 RMSNorm 的 epsilon（加在均方内），默认 1e-6

    Returns:
        y_a: [B_a, S_a, H_a] 负载 A 输出，dtype 与 x_a 一致，仅由 A 侧输入决定
        y_b: [B_b, 1, H_b] 负载 B 输出，dtype 与 x_b 一致，仅由 B 侧输入决定
        k_cache_out: [B_b, Nkv_b, Smax, D_b] 追加后的 key cache，shape/dtype 与 k_cache 一致
        v_cache_out: [B_b, Nkv_b, Smax, D_b] 追加后的 value cache，shape/dtype 与 v_cache 一致
    """
    y_a, y_b, kc, vc = _colocated_prefill_decode_core(
        x_a, gamma1_a, wq_a, wk_a, wv_a, wo_a, gamma2_a, w_gate_a, w_up_a, w_down_a,
        x_b, gamma1_b, wq_b, wk_b, wv_b, wo_b, gamma2_b, w_gate_b, w_up_b, w_down_b,
        k_cache, v_cache, cache_len, rope_cos_a, rope_sin_a, rope_cos_b, rope_sin_b,
        epsilon, torch.float32)
    return y_a.to(x_a.dtype), y_b.to(x_b.dtype), kc.to(k_cache.dtype), vc.to(v_cache.dtype)


def colocated_prefill_decode_oracle(
    x_a: torch.Tensor,
    gamma1_a: torch.Tensor,
    wq_a: torch.Tensor,
    wk_a: torch.Tensor,
    wv_a: torch.Tensor,
    wo_a: torch.Tensor,
    gamma2_a: torch.Tensor,
    w_gate_a: torch.Tensor,
    w_up_a: torch.Tensor,
    w_down_a: torch.Tensor,
    x_b: torch.Tensor,
    gamma1_b: torch.Tensor,
    wq_b: torch.Tensor,
    wk_b: torch.Tensor,
    wv_b: torch.Tensor,
    wo_b: torch.Tensor,
    gamma2_b: torch.Tensor,
    w_gate_b: torch.Tensor,
    w_up_b: torch.Tensor,
    w_down_b: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cache_len: torch.Tensor,
    rope_cos_a: torch.Tensor,
    rope_sin_a: torch.Tensor,
    rope_cos_b: torch.Tensor,
    rope_sin_b: torch.Tensor,
    epsilon: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Oracle (g)：dtype-agnostic，计算精度跟随输入（fp64_cpu 下为 fp64 真值），不硬编码 .float()。"""
    return _colocated_prefill_decode_core(
        x_a, gamma1_a, wq_a, wk_a, wv_a, wo_a, gamma2_a, w_gate_a, w_up_a, w_down_a,
        x_b, gamma1_b, wq_b, wk_b, wv_b, wo_b, gamma2_b, w_gate_b, w_up_b, w_down_b,
        k_cache, v_cache, cache_len, rope_cos_a, rope_sin_a, rope_cos_b, rope_sin_b,
        epsilon, x_a.dtype)
```

## 6. 额外信息

### 算子调用示例

```python
import torch

# 负载 A：prefill 层（2 × 1024 token）
B_a, S_a, H_a, Nq_a, Nkv_a, D_a, F_a = 2, 1024, 1024, 8, 2, 128, 2752
# 负载 B：decode 层（32 路并发，cache 容量 4096）
B_b, H_b, Nq_b, Nkv_b, D_b, F_b, Smax = 32, 1024, 8, 2, 128, 2752, 4096

dt = torch.bfloat16
x_a = torch.rand(B_a, S_a, H_a, dtype=dt) * 2 - 1
gamma1_a = torch.rand(H_a, dtype=dt) + 0.5
wq_a = (torch.rand(H_a, Nq_a * D_a, dtype=dt) - 0.5) * 0.1
wk_a = (torch.rand(H_a, Nkv_a * D_a, dtype=dt) - 0.5) * 0.1
wv_a = (torch.rand(H_a, Nkv_a * D_a, dtype=dt) - 0.5) * 0.1
wo_a = (torch.rand(Nq_a * D_a, H_a, dtype=dt) - 0.5) * 0.1
gamma2_a = torch.rand(H_a, dtype=dt) + 0.5
w_gate_a = (torch.rand(H_a, F_a, dtype=dt) - 0.5) * 0.1
w_up_a = (torch.rand(H_a, F_a, dtype=dt) - 0.5) * 0.1
w_down_a = (torch.rand(F_a, H_a, dtype=dt) - 0.5) * 0.1
x_b = torch.rand(B_b, 1, H_b, dtype=dt) * 2 - 1
gamma1_b = torch.rand(H_b, dtype=dt) + 0.5
wq_b = (torch.rand(H_b, Nq_b * D_b, dtype=dt) - 0.5) * 0.1
wk_b = (torch.rand(H_b, Nkv_b * D_b, dtype=dt) - 0.5) * 0.1
wv_b = (torch.rand(H_b, Nkv_b * D_b, dtype=dt) - 0.5) * 0.1
wo_b = (torch.rand(Nq_b * D_b, H_b, dtype=dt) - 0.5) * 0.1
gamma2_b = torch.rand(H_b, dtype=dt) + 0.5
w_gate_b = (torch.rand(H_b, F_b, dtype=dt) - 0.5) * 0.1
w_up_b = (torch.rand(H_b, F_b, dtype=dt) - 0.5) * 0.1
w_down_b = (torch.rand(F_b, H_b, dtype=dt) - 0.5) * 0.1
k_cache = torch.rand(B_b, Nkv_b, Smax, D_b, dtype=dt) * 2 - 1
v_cache = torch.rand(B_b, Nkv_b, Smax, D_b, dtype=dt) * 2 - 1
cache_len = torch.randint(1, Smax, (B_b,), dtype=torch.int32)
rope_cos_a = torch.rand(B_a, S_a, D_a, dtype=dt) * 2 - 1
rope_sin_a = torch.rand(B_a, S_a, D_a, dtype=dt) * 2 - 1
rope_cos_b = torch.rand(B_b, D_b, dtype=dt) * 2 - 1
rope_sin_b = torch.rand(B_b, D_b, dtype=dt) * 2 - 1

y_a, y_b, k_cache_out, v_cache_out = colocated_prefill_decode(
    x_a, gamma1_a, wq_a, wk_a, wv_a, wo_a, gamma2_a, w_gate_a, w_up_a, w_down_a,
    x_b, gamma1_b, wq_b, wk_b, wv_b, wo_b, gamma2_b, w_gate_b, w_up_b, w_down_b,
    k_cache, v_cache, cache_len, rope_cos_a, rope_sin_a, rope_cos_b, rope_sin_b)
# y_a: [B_a, S_a, H_a]，y_b: [B_b, 1, H_b]，cache_out: [B_b, Nkv_b, Smax, D_b]
```

### 负载配平与 t_hw 标定

**配平原则**：共驻的重叠收益在 t_A ≈ t_B 时最大（理想调度可把总时间压至串行的一半）。公开用例多数按「负载 A 的总 FLOPs / 负载 B 的总搬运字节数」贴近机器强度（CUBE 峰值 ÷ HBM 带宽）粗配平；隐藏用例覆盖从配平到十倍级失衡的完整谱系。

**t_hw 标定备注（硬件标定人员必读）**：本算子的 t_hw 必须按合并 roofline 标定——

$$
t_{\mathrm{hw}} = \max\left(\frac{\text{A+B 总 FLOPs}}{\text{CUBE 峰值}},\ \frac{\text{A+B 总搬运字节}}{\text{HBM 带宽}}\right)
$$

**不得按两段独立 roofline 之和（t_A + t_B）标定**：按和标定等于把朴素串行实现的耗时当作硬件下界，共驻重叠的收益空间被抹掉，性能墙失效（配平 case 下两种口径相差近一倍）。

### 现实原型与反面参照

- **POD-Attention**（ASPLOS 2025）：把 prefill 与 decode 的 attention 融合进同一 kernel，按两类负载的资源画像分配片上资源，单卡混布吞吐提升 ~20%
- **NanoFlow**（2024）：设备内把计算密集、带宽密集、网络密集的操作重叠调度，论证了单设备内并行是吞吐上界的主要缺口
- **Sarathi-Serve**（OSDI 2024）：chunked prefill 把 prefill 切块与 decode 混批，是本算子「双负载同驻一卡」的系统背景
- **反面参照**：DistServe（OSDI 2024）/ Splitwise（ISCA 2024）把 prefill 与 decode 分离到不同设备，回避共驻调度问题，代价是 KV cache 跨设备搬运与设备利用率损失——本算子考核的正是分离部署所回避的那部分能力

### 参考文献

- Kamath, A. K. et al. (2025). "POD-Attention: Unlocking Full Prefill-Decode Overlap for Faster LLM Inference". ASPLOS 2025, arXiv:2410.18038
- Zhu, K. et al. (2024). "NanoFlow: Towards Optimal Large Language Model Serving Throughput". arXiv:2408.12757
- Agrawal, A. et al. (2024). "Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve". OSDI 2024, arXiv:2403.02310
- Zhong, Y. et al. (2024). "DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving". OSDI 2024, arXiv:2401.09670（反面参照）
- Patel, P. et al. (2024). "Splitwise: Efficient Generative LLM Inference Using Phase Splitting". ISCA 2024, arXiv:2311.18677（反面参照）
