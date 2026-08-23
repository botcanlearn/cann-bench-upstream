# CompressedWeightGemv 算子 API 描述

## 1. 算子简介

自定义压缩位流的**在线解码融合 GEMV**。逻辑权重 $W \in \mathbb{R}^{N \times K}$ 以"2:4 结构化稀疏 + int4 分组量化"的自定义格式压缩存放：每连续 4 个 K 位置恰有 2 个非零，非零值为 int4 码（4 bit）乘分组 scale，非零位置由 4 选 2 组合编码（≤ 3 bit）给出。算子在 GEMV 数据通路内在线解码，计算 $y = x W^{\mathsf T} + b$，权重全程只以压缩形式读取一遍。

这是 decode-bound LLM 推理的前沿形态：decode 阶段 GEMV 的时延完全由权重字节数决定，"2:4 稀疏 × int4 量化"把每 4 个逻辑权重位置的**有效负载压到 16 bit**（两个 int4 码 + 3 bit 位置组合 + 摊薄的分组 scale），约为 fp16 稠密（64 bit）的 1/6、fp32 稠密的 1/12。本评测以 int32 为码流载体（有效位在低 8 位 / 低 3 位），压缩流实际字节数为 fp32 稠密的 1/2；高效 kernel 在片上仅保留有效字节即可进一步逼近 1/6 的信息密度下界。

**格式即规格**：该压缩格式不是任何现成指令或库的格式，desc 的码表与位布局就是全部定义（§2 给出完整、自足、已数值验证的规格），实现的正确性只取决于是否逐条遵守规格——这正是隐藏用例检验的对象。

**主要应用场景**：
- 稀疏 + 量化联合压缩 LLM 的 decode 推理（单 token / 小批量 GEMV，权重带宽是唯一瓶颈）
- 端侧 / 显存受限部署：权重驻留为压缩形式，不允许解压成稠密副本
- 投机解码、MoE 等对 decode 时延敏感的 serving 场景

**算子特征**：
- 难度等级：L5（FusedComposite）
- 五输入（激活 + 2 个 int32 码流 + scale + bias）单输出
- 融合 nibble 提取、码表解码、位置组合展开、分组反量化与 GEMV 累加
- 解码是精确整数运算，无数值脆弱性；矩阵累加为标准 fp32 数值路径

**为何是 L5**（约为 L4 FusedComposite 的 2 倍难度）：
- **正确 ≠ 快**：按 §2 直译——先把 packed_w / sparse_sel 解压成稠密 $W$，再调一次 GEMV——结果完全正确，但要写出 + 读回一个 fp32 稠密 $W$（4 字节/位置），总 HBM 流量 ≥ 5 倍于压缩流本身（读压缩 2 字节/位置 + 写稠密 4 + 读稠密 4），在 decode-bound 场景把压缩的带宽优势全部丢掉，性能锚定朴素 baseline（预期得分 ~0.5 档）。要逼近带宽下界，解码必须**融合在 GEMV 数据通路内**：压缩流只读一遍，nibble/查表/scale 展开在寄存器/片上完成，直接进入乘加
- **格式即规格，实现规格而非过可见测试**：码表（码 0 → 值 −8！全零 packed_w 不等于全零权重）、4 选 2 组合表、scale 组索引按绝对列计算、int32 载体仅低位有效——每一条都有对应的隐藏陷阱用例（全 0 / 全 255 / 全 136（解码恰为全零）/ sel 全 0 全 5 / 单 scale 组 / K 不整除 4·group_size 等）
- **不规则访存**：非零位置由数据决定（sparse_sel 驱动的 4 选 2 散射），激活侧的 gather 模式逐块变化，与规则稠密 GEMV 模板不同构
- **独特轴**：位流解码 + 结构化稀疏 + 分组量化 + GEMV 融合的组合在本评测集内独一无二

## 2. 算子定义

### 压缩格式（完整规格，已数值验证）

记 $K_4 = K/4$（要求 $4 \mid K$）。对每行 $n \in [0, N)$、每个 4 块 $j \in [0, K_4)$（覆盖绝对列 $4j..4j{+}3$）：

**(1) 非零位置**：`sparse_sel[n, j]` ∈ [0, 5] 按下表给出块内两个非零位置 $(pos_a, pos_b)$（升序对，4 选 2 全部组合）：

| sparse_sel | 0 | 1 | 2 | 3 | 4 | 5 |
|------------|-----|-----|-----|-----|-----|-----|
| (pos_a, pos_b) | (0,1) | (0,2) | (0,3) | (1,2) | (1,3) | (2,3) |

**(2) 非零值**：`packed_w[n, j]` ∈ [0, 255]（int32 载体，**仅低 8 位有效**）：

- 低 nibble $c_a = \text{packed\_w}[n,j] \mathbin{\&} \text{0xF}$：第一个非零（$pos_a$ 处）的 int4 码
- 高 nibble $c_b = (\text{packed\_w}[n,j] \gg 4) \mathbin{\&} \text{0xF}$：第二个非零（$pos_b$ 处）的 int4 码
- 码 → 值（偏移二进制）：$\mathrm{val}(c) = c - 8 \in [-8, 7]$

| 码 c | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
|------|----|----|----|----|----|----|----|----|---|---|----|----|----|----|----|----|
| 值 | −8 | −7 | −6 | −5 | −4 | −3 | −2 | −1 | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |

注意两个陷阱：**码 0 解码为 −8**（packed_w 全 0 时 $W$ 的非零位置全为 $-8 \cdot scale$，不是全零）；**码 8 解码为 0**（packed_w 全 136 = 0x88 时 $W$ 恰为全零，$y = bias$）。

**(3) 分组反量化**：`scales[n, G]`，$G = K/\text{group\_size}$（要求 $\text{group\_size} \mid K$ 且 $4 \mid \text{group\_size}$，故任一 4 块不跨组）。绝对列 $k$ 的 scale 组索引为 $\lfloor k / \text{group\_size} \rfloor$。

**(4) 权重重建**（数学定义；高效实现不必显式物化 $W$）：

$$
W[n,\ 4j + pos_a] = (c_a - 8) \cdot \text{scales}[n,\ \lfloor (4j + pos_a)/\text{group\_size} \rfloor]
$$

$$
W[n,\ 4j + pos_b] = (c_b - 8) \cdot \text{scales}[n,\ \lfloor (4j + pos_b)/\text{group\_size} \rfloor]
$$

$$
W[n, k] = 0 \quad (k \notin \{4j+pos_a,\ 4j+pos_b\})
$$

任意满足值域的随机 `packed_w` / `sparse_sel` 都是合法码流（无跨输入约束），这使评测可以独立随机生成各输入。

### GEMV 计算

$$
y = x\, W^{\mathsf T} + bias, \qquad y \in \mathbb{R}^{B \times N}
$$

内部计算精度：解码出的权重值、激活与累加均以 fp32 进行（bfloat16/float16 输入升 fp32），bias 以 fp32 相加，输出舍回 x 的 dtype。K 维累加顺序不作规定（标准浮点 GEMV 数值路径，精度按 §4 阈值判定）。

## 3. 接口规范

### 算子原型

```python
compressed_weight_gemv(Tensor x, Tensor packed_w, Tensor sparse_sel, Tensor scales, Tensor bias, int group_size) -> Tensor y
```

### 输入参数说明

| 参数 | 类型 | 必需 | dtype | shape | 描述 |
|------|------|------|-------|-------|------|
| x | Tensor | 是 | bfloat16/float16/float32 | [B, K] | 激活；K % 4 == 0 且 K % group_size == 0 |
| packed_w | Tensor | 是 | int32 | [N, K/4] | 压缩权重码流，取值 [0, 255]（仅低 8 位有效，见 §2 码表） |
| sparse_sel | Tensor | 是 | int32 | [N, K/4] | 4 选 2 位置组合编码，取值 [0, 5]（见 §2 组合表） |
| scales | Tensor | 是 | float16/float32 | [N, K/group_size] | 分组量化 scale；评测取值 [0.001, 0.1] |
| bias | Tensor | 是 | 与 x 一致 | [N] | 输出偏置 |
| group_size | int | 是 | - | 标量 | 分组大小：4 的正倍数且整除 K，评测取值 {32, 64, 128} |

### 输出

| 名称 | dtype | shape | 描述 |
|------|-------|-------|------|
| y | 与 x 一致 | [B, N] | y = x @ W^T + bias |

### 数据类型

| x / bias dtype | scales dtype | 输出 dtype | 内部计算 |
|----------------|--------------|-----------|----------|
| bfloat16 | float16 / float32 | bfloat16 | fp32（解码值、乘加、bias 均 fp32） |
| float16 | float16 / float32 | float16 | fp32 |
| float32 | float16 / float32 | float32 | fp32 |

### 规则与约束

- `bias` 的 dtype 与 `x` 一致；`scales` 为 float16 或 float32（与 x 独立，评测含交叉组合）
- 形状一致性：`packed_w` 与 `sparse_sel` 同形 [N, K/4]；`scales` 第二维必须等于 K/group_size（逐 case 校验）
- `packed_w` 取值 [0, 255]、`sparse_sel` 取值 [0, 5]（由 value_range 保证）；超出低 8 位 / [0,5] 的输入行为未定义
- K % 4 == 0 且 K % group_size == 0；**不要求** K % (4·group_size) == 0（隐藏用例专门覆盖不整除场景）
- 解码必须精确实现 §2 码表与组合表（整数运算，无容差）；GEMV 累加须以 ≥ fp32 精度进行
- 输出须为 contiguous 张量
- 显存约束：单 case 输入张量合计 ≤ 2 GB（全部用例逐 case 校验通过，最大 case 约 258 MB）

### 支持范围

| 维度 / 参数 | 范围 | 备注 |
|---|---|---|
| `B`（batch） | 1 ~ 32 | GEMV 场景 B 小；cases.csv 实测 1 ~ 16 |
| `N`（输出维） | 1 ~ 32768 | cases.csv 实测 4096 ~ 14336（LLM proj/FFN 尺寸），隐藏用例含素数 N |
| `K`（归约维） | 32 ~ 16384 | cases.csv 实测 4096 ~ 14336；须整除 4 与 group_size |
| `group_size` | {32, 64, 128} | 隐藏用例含 group_size == K（单组）与 K % (4·g) != 0 |
| `x` 取值 | [-1, 1] | 隐藏用例含 [-8, 8]（fp16）、[-100, 100]（fp32）、全零 |
| `packed_w` / `sparse_sel` 取值 | [0, 255] / [0, 5] | 隐藏用例含全 0 / 全 255 / 全 136 / 常数 sel |
| `scales` 取值 | [0.001, 0.1] | 隐藏用例含全 0（y == bias）与常数端点 |
| `bias` 取值 | [-1, 1] | 隐藏用例含 [-100, 100] 与全零 |

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

**通过标准**（采用生态默认阈值，未放宽）：

| 数据类型 | FLOAT32 | FLOAT16 | BFLOAT16 |
|----------|---------|---------|----------|
| **通过阈值(Threshold)** | 2^-13 | 2^-10 | 2^-7 |

当 MERE < Threshold 且 MARE < 10 × Threshold 时判定为通过；小值域与相消位置由评测框架兜底标准处理。

**阈值可行性实测**（仓库 checker `kernel_eval.utils.compare.compare_tensors`，native = plain golden）：plain（fp32 数据通路）vs fp64 oracle，覆盖 LLM 尺寸（N 4096~14336、K 4096~11008、g 32/64/128）、陷阱值域（bias [-100,100]、fp16 x [-8,8]）与 x/scales dtype 交叉组合，多种子共 39 组配置全部通过——bf16 MERE≈1.1e-6 / MARE≈8.0e-3，fp16 MERE≈9.5e-7 / MARE≈1.1e-3，fp32 MERE≈1.7e-6 / MARE≈1.1e-3（fp32 MARE 尖峰位于 K 维内积相消处，由兜底标准处理）。解码部分为精确整数运算，误差全部来自 GEMV 浮点累加；以 ≥ fp32 精度累加即可稳定达标，无需放宽阈值。

## 5. 标准 Golden 代码

```python
import torch

# 4 选 2 位置组合表：sel -> (pos_a, pos_b)，升序对
_POS_A = (0, 0, 0, 1, 1, 2)
_POS_B = (1, 2, 3, 2, 3, 3)


def _compressed_weight_gemv_core(x, packed_w, sparse_sel, scales, bias, group_size, compute_dtype):
    """核心计算：整数解码（查表 + 移位）重建 W，再以 compute_dtype 做 GEMV。"""
    N, KQ = packed_w.shape          # KQ = K/4
    K = KQ * 4
    device = x.device

    x_f = x.to(compute_dtype)                      # [B, K]
    scales_f = scales.to(compute_dtype)            # [N, K/group_size]
    bias_f = bias.to(compute_dtype)                # [N]

    packed = packed_w.long()                       # 仅低 8 位有效
    lo = (packed & 0xF) - 8                        # [N, K/4] 第一个非零的 int4 值 ∈ [-8, 7]
    hi = ((packed >> 4) & 0xF) - 8                 # [N, K/4] 第二个非零的 int4 值 ∈ [-8, 7]

    sel = sparse_sel.long()                        # [N, K/4] ∈ [0, 5]
    pos_a = torch.tensor(_POS_A, dtype=torch.long, device=device)[sel]   # [N, K/4]
    pos_b = torch.tensor(_POS_B, dtype=torch.long, device=device)[sel]   # [N, K/4]
    base = torch.arange(KQ, device=device, dtype=torch.long).unsqueeze(0) * 4   # [1, K/4]
    col_a = base + pos_a                           # [N, K/4] 绝对列 4j+pos_a
    col_b = base + pos_b                           # [N, K/4] 绝对列 4j+pos_b

    # scale 组索引 = 绝对列 // group_size（4 | group_size，两个非零同组，仍按定义逐列取）
    sc_a = torch.gather(scales_f, 1, col_a // group_size)   # [N, K/4]
    sc_b = torch.gather(scales_f, 1, col_b // group_size)   # [N, K/4]

    w = torch.zeros(N, K, dtype=compute_dtype, device=device)
    w.scatter_(1, col_a, lo.to(compute_dtype) * sc_a)
    w.scatter_(1, col_b, hi.to(compute_dtype) * sc_b)

    y = x_f @ w.t() + bias_f                       # [B, N]
    return y


def compressed_weight_gemv(
    x: torch.Tensor,
    packed_w: torch.Tensor,
    sparse_sel: torch.Tensor,
    scales: torch.Tensor,
    bias: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """
    压缩权重在线解码融合 GEMV golden reference（plain golden = bench：内部 fp32 计算）

    Args:
        x: [B, K] 激活，bfloat16/float16/float32，K % 4 == 0 且 K % group_size == 0
        packed_w: [N, K/4] int32，取值 [0, 255]：低/高 nibble = 块内两个非零的 int4 码（码 c → 值 c-8）
        sparse_sel: [N, K/4] int32，取值 [0, 5]：4 选 2 非零位置组合
            （0→(0,1) 1→(0,2) 2→(0,3) 3→(1,2) 4→(1,3) 5→(2,3)）
        scales: [N, K/group_size] float16/float32，分组量化 scale
        bias: [N]，dtype 与 x 一致
        group_size: 分组大小（4 的正倍数且整除 K，评测取 {32, 64, 128}）

    Returns:
        y: [B, N] = x @ W^T + bias，dtype 与 x 一致
    """
    y = _compressed_weight_gemv_core(
        x, packed_w, sparse_sel, scales, bias, group_size, torch.float32)
    return y.to(x.dtype)


def compressed_weight_gemv_oracle(
    x: torch.Tensor,
    packed_w: torch.Tensor,
    sparse_sel: torch.Tensor,
    scales: torch.Tensor,
    bias: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """Oracle (g)：dtype-agnostic，计算精度跟随输入（fp64_cpu 下即为 fp64 真值），不硬编码 .float()。"""
    return _compressed_weight_gemv_core(
        x, packed_w, sparse_sel, scales, bias, group_size, x.dtype)
```

## 6. 额外信息

### 算子调用示例

```python
import torch

B, N, K, group_size = 1, 11008, 4096, 128

x = torch.randn(B, K, dtype=torch.bfloat16, device="npu")
packed_w = torch.randint(0, 256, (N, K // 4), dtype=torch.int32, device="npu")
sparse_sel = torch.randint(0, 6, (N, K // 4), dtype=torch.int32, device="npu")
scales = torch.empty(N, K // group_size, dtype=torch.float16, device="npu").uniform_(0.001, 0.1)
bias = torch.randn(N, dtype=torch.bfloat16, device="npu")

y = compressed_weight_gemv(x, packed_w, sparse_sel, scales, bias, group_size)
# y.shape: [B, N]
```

### 性质（可用于实现自检，均已数值验证）

- `packed_w` 全 0 → 每个非零位置的值为 $-8 \cdot scale$（**不是**全零权重）；全 136（0x88）→ $W$ 恰为全零，$y == bias$；全 255（0xFF）→ 两个非零均为 $+7 \cdot scale$
- `scales` 全 0 或 `x` 全 0 → $y == bias$（逐位精确）
- 交换 `packed_w` 的高低 nibble ⟺ 交换块内两个非零位置上的解码值（sparse_sel 不变）
- 独立逐元素解码器（python 循环查码表重建 $W$）+ fp64 matmul 与 golden oracle 的最大相对偏差 < 1e-12（实测为 0）

### 有效负载与载体

每 4 个逻辑权重位置的有效信息：8 bit（两个 int4 码）+ 3 bit（4 选 2 组合，以 [0,5] 整数存放）+ 分组 scale 摊薄（16/group_size ~ 每 4 位置 0.5~2 bit），合计约 12~13 bit，约为 fp16 稠密 64 bit 的 1/5~1/6。评测载体为 int32（低位有效），压缩流实际字节数 = 2 字节/位置（packed_w 1 + sparse_sel 1，按有效字节计），为 fp32 稠密的 1/2；朴素"解压成稠密 fp32 再 GEMV"的 HBM 流量 ≥ 5 倍于融合实现（读压缩 + 写稠密 + 读稠密）。

### 参考文献

- Mishra, A. et al. (2021). "Accelerating Sparse Deep Neural Networks". arXiv:2104.08378（2:4 结构化稀疏格式的来源）
- Frantar, E., Alistarh, D. (2023). "SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot". ICML 2023, arXiv:2301.00774（LLM 2:4 稀疏化）
- Frantar, E. et al. (2023). "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers". ICLR 2023, arXiv:2210.17323（int4 分组量化与 group_size 约定的来源）
- Lin, J. et al. (2024). "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration". MLSys 2024（decode-bound 权重带宽瓶颈的工程背景）
