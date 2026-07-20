# FlatQuant (pertoken INT4-logical) 算子 API 描述

> 本文档自洽：合并原 `desc.md` 与算子生成 `prompt.md`，作为 `flat_quant_pertoken_int4`
> benchmark reference 套件的唯一说明文件。算子语义（§1–§4）、实现契约（§5 强制 C→V
> 融合路径）、精度要求（§6）、验收约束（§7）具有同等约束力；AscendC 实现与调试前必须先
> 读取并遵守。本套件刷新为「少量 small + 大量 LLM-shape」用例（见 §9），把 token 维 `K`
> 与方阵变换维 `M/N` 在 `golden.py` 允许范围内放大到 LLM 规模（最大 5120）。

## 1. 算子简介

**算子特征**：
- 难度等级：L3（Contraction）

`flat_quant` 对齐源码目录 `ops-nn/quant/flat_quant`。该算子先对 `[K, M, N]` 输入做
**Kronecker 形式的左右小矩阵乘**（`kroneckerP1 @ x @ kroneckerP2`），再执行 **per-token
动态量化**到逻辑 INT4。本 benchmark 固定 **per-token INT4 逻辑输出路径**：

- **C 侧（Cube / AIC）** 完成 `kroneckerP1 @ x @ kroneckerP2` 两段 matmul；
- **V 侧（Vector / AIV）** 执行 ReduceMax(absmax)、scale 计算、归一化、round、clip、INT8 写回；

属于 **C→V kernel flow**（Cube 产出中间结果交给 Vector 做量化 epilogue）。

PyTorch golden 使用 `int8` 承载**逻辑 INT4** 值（值域 `[-7, 7]`），**不做 bit-pack**，
也不覆盖 FLOAT4_E2M1 pergroup 路径或真实 INT4 pack 布局。

## 2. 算子定义

```text
# x: [K, M, N], kroneckerP1: [M, M], kroneckerP2: [N, N]
x1[k]         = kroneckerP1 @ x[k]                       # [M, N]，左乘
x2[k]         = x1[k] @ kroneckerP2                      # [M, N]，右乘
quantScale[k] = max(abs(x2[k])) / (7 / clipRatio)       # 标量，per-token
out[k]        = round(x2[k] / quantScale[k]).clamp(-7, 7)
```

等价 einsum 形式（与 `golden.py` 一致，matmul 在 float32 累加）：

```python
tmp         = einsum('ab,kbn->kan', kroneckerP1, x)         # [K, M, N]
transformed = einsum('kmn,nc->kmc', tmp, kroneckerP2)       # [K, M, N]
max_abs     = transformed.abs().amax(dim=(1, 2), keepdim=True)  # [K, 1, 1]
denom       = 7.0 / clipRatio
scale       = max_abs / denom                              # [K, 1, 1] float32
normalized  = where(scale > 0, transformed / scale, 0)
out         = normalized.round().clamp(-7, 7).to(int8)     # [K, M, N]
quantScale  = scale.reshape(K).to(float32)                 # [K]
```

**量化语义关键点（精度强约束，须与 golden 逐位对齐）：**

- absmax 在每个 token（K 维 slice）上对全部 `M*N` 个元素求绝对值最大；归约范围是
  `dim=(1, 2)`，即**每 token 一个标量 scale**，不是 per-row / per-channel。
- `denom = 7 / clipRatio`；`clipRatio` 越小，`denom` 越大，scale 越小，量化越激进。
- `scale > 0` 才做除法；`scale == 0`（全零 token）时输出 0，`quantScale` 亦为 0（见 §6 全零边界）。
- `round` 为四舍五入到偶（torch 默认）后 `clamp` 到 `[-7, 7]`，再转 `int8`。
- matmul 在 **float32** 累加（golden 把 fp16/bf16 输入 `.to(float32)` 后再 einsum）。

## 3. 接口规范

```python
flat_quant(x, kroneckerP1, kroneckerP2,
           clipRatio=1.0, quant_mode="pertoken", out_dtype="int4_logical")
    -> (out, quantScale)
```

| 参数 | 输入/输出 | dtype | shape | 说明 |
|------|-----------|-------|-------|------|
| `x` | 输入 | `FLOAT16 / BFLOAT16` | `[K, M, N]` | 原始输入张量 |
| `kroneckerP1` | 输入 | 同 `x` | `[M, M]` | **左乘方阵**（行列均为 M） |
| `kroneckerP2` | 输入 | 同 `x` | `[N, N]` | **右乘方阵**（行列均为 N） |
| `out` | 输出 | `INT8` | `[K, M, N]` | 逻辑 INT4 量化值，值域 `[-7, 7]` |
| `quantScale` | 输出 | `FLOAT32` | `[K]` | per-token scale |

> **dtype 提醒**：`out_dtype="int4_logical"` 是**输出**逻辑 dtype（用 `int8` 承载），**不是
> 输入 dtype**。三个输入张量 `x / kroneckerP1 / kroneckerP2` 全部是 `float16` 或 `bfloat16`，
> 且三者 dtype 必须一致。`quantScale` 始终是 `float32`。

## 4. 约束说明

### 4.1 形状约束（`golden.py` 实际校验）

- `x` 维度必须为 3，形状 `[K, M, N]`（`x.dim() != 3` → 报错）。
- `kroneckerP1.shape == (M, M)`，`kroneckerP2.shape == (N, N)` —— **两个变换矩阵必须是方阵，
  且分别绑定 `x` 的 M、N 维**；这是 golden 唯一强校验的 shape 关系（违反即 `ValueError`）。
- 三个张量 dtype 一致，均为 `float16` 或 `bfloat16`。

### 4.2 原始语义量级（参考，非 golden 强校验）

原 reference 出于「让 CPU golden 在 agent 调试阶段不至于过慢」的考量，建议把规模控制在：

- `1 <= K <= 16`，`4 <= M <= 32`，`16 <= N <= 64`。

> 这些上界**只是建议量级**，由原 `desc.md` / `prompt.md` 文字给出，**`golden.py` 本身并不强制**
> （它只要求 `x` 三维 + P1/P2 为对应方阵）。本刷新套件为产出 LLM-shape 用例，在 golden 允许的
> 范围内把 `K / M / N` 放大到最大 5120（见 §9 与文末「刷新说明」）；放大时严格保持 P1=`[M,M]`、
> P2=`[N,N]` 的方阵关系，不触发 golden 的 shape 校验。

### 4.3 固定参数

- `quant_mode = "pertoken"`：仅支持 per-token 路径（其他值 → `ValueError`）。
- `out_dtype = "int4_logical"`：用 `int8` 承载逻辑 INT4 值，**不做 bit-pack**（其他值 → `ValueError`）。
- `clipRatio`：浮点属性（默认 `1.0`），决定量化分母 `denom = 7 / clipRatio`；本套件用例覆盖
  `1.0` 与 `0.75` 两档。

### 4.4 输出

| 名称 | shape | dtype | 说明 |
|------|-------|-------|------|
| `out` | `[K, M, N]` | `int8` | 承载逻辑 INT4，值域 `[-7, 7]` |
| `quantScale` | `[K]` | `float32` | per-token scale |

`golden.py` 返回 **TUPLE `(out, quantScale)`**；下游模型与验证必须按二元组消费。

## 5. 实现约束与参考设计（C→V 融合路径）

> 本节分两层。**§5.1 硬约束**是正确性与合法性底线，必须遵守；**§5.2 参考设计与已知反模式**是非强制指导——给出一条已验证可行的实现路径与避坑提示，**鼓励在守住 §5.1 的前提下自行设计更优方案；当生成算子性能不足时，应主动超越本参考路径，而非照抄。**

### 5.1 硬约束（正确性 + 反作弊，必须遵守）

- **真融合，禁退化**：必须生成真正的 Cube + Vector 融合 AscendC kernel；禁止退化为纯 AIV（在 Vector 侧用逐元素循环模拟矩阵乘）、纯 CPU、torch（`model_new_tilelang.py` / `model_new_ascendc.py` 中**禁止用 torch 算子做任何实际计算**）、aclnn 高层组合算子、Python fallback。
- **两段 matmul 必须落 Cube**：Kronecker 左右乘 `kroneckerP1[M,M] @ x[k][M,N] -> tmp[k]`、`tmp[k] @ kroneckerP2[N,N] -> transformed[k]` 两段 matmul 必须由 AIC/Cube 用 AscendC Cube / MatMul / MMAD 原语完成（matmul 在 float32 累加，与 §2 / golden 一致）；**禁止在 AIV 侧用逐元素循环模拟矩阵乘**。
- **per-token 量化必须片上向量化**：absmax（按 K 维对 `transformed[k]` 全部 `M*N` 元素求绝对值最大，即 `dim=(1,2)` 归约，每 token 一个标量）、`scale = absmax / (7/clipRatio)`、`scale>0` 才做除法（否则输出 0、`quantScale=0`）、`round`→`clamp(-7,7)`→`int8` 写回 `out[k]`、`float32 quantScale[k]` 写回，必须在片上 AIV/Vector 完成；**禁止**下沉到 torch / host / CPU / aclnn / Python 计算输出，**禁止**用 AIC 标量循环替代。
- **量化语义与 golden 一致**：absmax 归约范围、`denom = 7/clipRatio`、`scale>0` guard、`round`（四舍五入到偶）后 `clamp(-7,7)` 转 `int8` 必须与 §2 / golden 逐位对齐；不得为消性能或在容差内改算子数学语义。
- **输出契约**：返回 **TUPLE `(out:int8[K,M,N], quantScale:float32[K])`**；`out_dtype="int4_logical"` 是**输出**逻辑 dtype（用 `int8` 承载逻辑 INT4，值域 `[-7,7]`，不做 bit-pack），**不是输入 dtype**；必须正确支持 `float16` 与 `bfloat16` 输入（三输入同型），`quantScale` 输出为 `float32`；P1=`[M,M]`、P2=`[N,N]` 方阵与 §4.1 shape / dtype 断言一致。
- **跨核同步正确性**：AIC→AIV 交接必须正确同步、保证跨核数据可见，不得出现数据竞争（否则结果错）；K 维并行下 AIC 与 AIV 需做 token 级同步，中间缓冲须有明确生命周期。
- kernel 的 `__global__` 核函数名和 Host `_do` 入口名必须包含 `custom`（如 `flat_quant_custom` / `flat_quant_custom_<dtype>` 等）；不得生成不含 `custom` 的 profiling kernel 名；AscendC 热路径禁止标量逐元素循环写法（少量边界 / 控制元数据除外），须使用块级 / 向量化原语。
- 精度遵循 §6 / [`../PRECISION_SPEC.md`](../PRECISION_SPEC.md) 与 `proto.yaml` 的 `precision` 节点（`out` 用 `int8_three_tier`，`bit_exact_ratio` 收紧到 `0.995`；`quantScale` 用 `input_dtype_inherited` 并保留全零边界兜底）。

### 5.2 参考设计与已知反模式（指导，非强制，鼓励超越）

> 以下是一条**已验证可行**的 C→V 实现路径与若干提示，作为起点与避坑参考，**不是必须照抄的配方**。在守住 §5.1 的前提下，鼓励 agent 探索更优实现；**性能不足时优先在此处突破。**

- **参考数据流（C→V）**：

  ```text
  AIC (Cube)：第一段 matmul  kroneckerP1[M,M] @ x[k][M,N]      -> tmp[k][M,N]
  AIC (Cube)：第二段 matmul  tmp[k][M,N]      @ kroneckerP2[N,N] -> transformed[k][M,N]
     --- C→V 交接：AIC 把 transformed 写入 workspace / 中间缓冲 ---
  AIV (Vector)：按 K 维对 transformed[k] 全部 M*N 元素求 absmax
  AIV (Vector)：scale = absmax / (7 / clipRatio)
  AIV (Vector)：normalized = transformed / scale（scale>0 时），否则 0
  AIV (Vector)：round -> clamp(-7, 7) -> int8 写回 out[k]
  AIV (Vector)：float32 quantScale[k] 写回
  ```

  AIC 把两段 matmul 后的 `transformed` 写入 workspace / GM ring slot，AIV 读取后完成 per-token 量化 epilogue。这是一种直接可行的切分；agent 可自行探索更优方案（如 AIC 内部连续两次 MMAD、中间 `tmp` 在 L1/L0 流转不落 workspace、把量化融进 matmul epilogue、不同 tile/buffer 策略等）。
- **参考中间张量 / workspace 布局**：`tmp` 与 `transformed` 的存放位置与复用策略由 agent 按性能选择；中间 `tmp` 可在 L1/L0/workspace 中流转，也允许由 AIC 内部连续两次 MMAD 直接产出 `transformed`。
- **参考同步原语**：AIC 写 GM 后用 `PipeBarrier<PIPE_ALL>` 排空再 `CrossCoreSetFlag` 通知 AIV；ring slot 维护明确 C2V/V2C 生命周期。具体同步原语与 buffer 布局由 agent 按性能选择，只要满足 §5.1 的跨核同步正确性即可。
- **参考 tiling**：AIC + AIV 混合执行（如 1C2V），按 shape 自适应选核与切分——大 K 走 token 并行、大 M/N 走方阵分块；避免过小 tile 导致核利用率过低（本套件含 K 高达 5120 的 many-token 用例与 M/N 高达 5120 的大方阵变换用例）。
- **已知反模式（建议避开）**：epilogue 逐行用标量读 per-token scale、且每个向量 op 后插同步 fence 的串行写法，在大 shape 下被 fence 串行主导；优先**多行整块** `[M,N]` 向量处理。
- 建议在设计文档与 `trace.md` 记录：AIC/AIV 分工（两段 matmul 如何在 cube 侧串联）、workspace 布局（`tmp` 与 `transformed` 存放 / 复用）、同步方式（K 维并行下的 token 级同步），便于性能复盘。

## 6. 精度要求

本算子精度判定遵循 [`../PRECISION_SPEC.md`](../PRECISION_SPEC.md)。通过条件与阈值参数定义在
同目录 `proto.yaml` 的 `precision` 节点，以下仅说明本算子特定的取舍。

### 6.1 算子特定说明

- **`out` 阈值归属**：规则 `int8_three_tier`，承载**逻辑 INT4** 值（值域 `[-7, 7]`）。
  **`bit_exact_ratio` 从默认 `0.99` 收紧到 `0.995`**：INT4 值域里 `±1` 相对宽松度约 14%
  （INT8 仅 0.8%），99% bit-exact + 1% 允许 ±1 在 INT4 下相对宽松度过大，故收紧；
  `tolerance_abs_diff=1` / `fatal_abs_diff=2` 保持默认（`±1` 仍由 fp16 matmul vs fp32 golden
  的 round 边界不可避免）。
- **`quantScale` 阈值归属**：规则 `input_dtype_inherited`，其数值由 FP16/BF16 Kronecker matmul
  和 ReduceMax 推导，精度上限受输入 dtype 制约（`bfloat16: 2^-7`、`float16: 2^-10`）。
- **`quantScale` 全零边界**：全零 token 行 ReduceMax=0 → `quantScale=0`；保留显式
  `small_value_handling`（`threshold=1e-6`、`absolute_tolerance=1e-5`）兜底全零 / 极小尺度，
  避免自适应默认在 RMS 尺度 `s≈0` 时退化过严。
- **后续可调**：若发现某些 case 实际 `bit_exact_ratio` 难以稳定达到 `99.5%`（例如 K 较小的
  Kronecker 路径），可在 SPEC 加 INT4 专项规则或回调到 `0.99`。

## 7. 强制验收约束

1. **正确性是硬门**：所有 `cases.yaml` / `cases.csv` 中的用例**精度必须全部达标**；不得只通过
   `basic_case` 或部分 `general_case` 后停止。
2. **性能是软门**：所有用例的**计算速度必须优于 torch 小算子拼接实现**（duration-only 口径）；
   如任一用例性能未优于该基线，必须继续优化 AscendC tiling、workspace 流水或 vector 量化路径，
   直到性能达标，或在 `trace.md` 中记录明确阻塞原因（软门不阻塞交付，但必须标注「性能未达标」）。
3. **不得为消性能或在容差内的问题改算子数学语义**：乘加顺序、absmax 归约范围、`denom = 7/clipRatio`、
   `clamp(-7,7)`、round 行为必须与 golden 一致。
4. 遇环境差异（profiling 失败 / runtime 缺失等）按「停下报告」约定处理，不得伪造成 PASS。

## 8. 标准 Golden 代码

`golden.py` 使用 `einsum` 完成左右乘（fp32 累加），再按 K 维每个 slice 做 ReduceMax 和逻辑
INT4 量化，返回 `(out:int8[K,M,N], quantScale:float32[K])` 二元组。**禁止修改 golden 数学语义。**

### 测试资料对应关系

- `docs/aclnnFlatQuant.md`：Kronecker 左右乘、per-token / per-group 量化公式。
- `op_kernel/flat_quant_cube.h` 与 `op_kernel/flat_quant_vec.h`：C/V 分段路径参考。

## 9. 本 benchmark case 设计（刷新说明）

`cases.yaml` / `cases.csv` 当前包含 **20 个正向 case**，采用「**少量 small + 大量 LLM-shape**」
布局，与 `golden.py` 的 shape 契约严格一致（`x:[K,M,N]`、`P1:[M,M]`、`P2:[N,N]` 方阵）：

- **6 个 SMALL（smoke / edge，maxdim < 512）**：贴近原语义量级（`K=1..16`、`M=4..32`、`N=16..64`），
  覆盖最小 M/N、方阵、最大 K、FP16/BF16 与 `clipRatio ∈ {1.0, 0.75}`，用于冒烟与边界。
- **14 个 LARGE（LLM-shape，maxdim ≥ 512，最大 5120）**：
  - **K-heavy（many-token）**：`K ∈ {512, 1024, 2048, 4096, 5120, 3072}`，M/N 保持小（per-token
    量化的典型「大量 token」形状，cost 随 K 线性增长）。
  - **N-heavy**：`N ∈ {512, 1024, 2048}`，放大右乘方阵 `P2[N,N]`。
  - **M-heavy**：`M ∈ {512, 1024, 2048}`，放大左乘方阵 `P1[M,M]`。
  - **balanced square / max square**：`512×512` 与 `5120×5120` 大方阵变换。

所有 case：`dtype` 为三元同型列表（`float16` 或 `bfloat16`）；`attrs` 结构恒为
`{clipRatio, quant_mode=pertoken, out_dtype=int4_logical}`，`clipRatio` 在 `1.0 / 0.75` 间变化；
`value_range = [-1, 1]`（沿用原 op）；`baseline_perf_us = 0.0`、`t_hw_us = 0.0`（占位，待真机回填）。

> **放大边界与 caveat**：dims 仅放大到 golden 数学允许处。`M`、`N` 放大意味着 `P1`、`P2` 是
> `M×M`、`N×N` 方阵且与 `x` 的对应维严格相等（golden 强校验），故 M、N 的放大必然伴随
> 二次增长的方阵与中间张量，CPU golden 在 5120×5120 单 token 下约 ~0.9s（仍可跑过 validator）。
> 原 reference 文字建议的 `K≤16 / M≤32 / N≤64` 仅为调试期 CPU 速度考量，非 golden 强约束，本套件
> 为 LLM-shape 目标在 golden 允许范围内予以超出，并保持方阵关系不破坏 shape 校验。

## 标准 Golden 代码

```python
#!/usr/bin/python3
# coding=utf-8

# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

import torch


def flat_quant(
    x: torch.Tensor,
    kroneckerP1: torch.Tensor,
    kroneckerP2: torch.Tensor,
    clipRatio: float = 1.0,
    quant_mode: str = "pertoken",
    out_dtype: str = "int4_logical",
):
    """Torch golden for flat_quant per-token logical INT4 path."""
    if quant_mode != "pertoken" or out_dtype != "int4_logical":
        raise ValueError("This benchmark fixes pertoken logical INT4 output")
    if x.dim() != 3:
        raise ValueError("x must be [K,M,N]")
    k, m, n = x.shape
    if kroneckerP1.shape != (m, m) or kroneckerP2.shape != (n, n):
        raise ValueError("kronecker matrices must be [M,M] and [N,N]")
    tmp = torch.einsum('ab,kbn->kan', kroneckerP1.to(torch.float32), x.to(torch.float32))
    transformed = torch.einsum('kmn,nc->kmc', tmp, kroneckerP2.to(torch.float32))
    max_abs = transformed.abs().amax(dim=(1, 2), keepdim=True)
    denom = 7.0 / float(clipRatio)
    scale = max_abs / denom
    normalized = torch.where(scale > 0, transformed / scale, torch.zeros_like(transformed))
    out = torch.round(normalized).clamp(-7, 7).to(torch.int8)
    return out, scale.reshape(k).to(torch.float32)
```
