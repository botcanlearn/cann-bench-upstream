# WeightQuantBatchMatmulV2 算子 API 描述

## 1. 算子简介

**算子特征**：
- 难度等级：L3（Contraction）

`weight_quant_batch_matmul_v2` 对齐源码目录 `ops-nn/matmul/weight_quant_batch_matmul_v2`。本 benchmark 选取**权重反量化 + 浮点 matmul + bias** 路径：低 bit 权重（`int8`）在 **V（Vector / AIV）侧**按 `antiquantScale / antiquantOffset` 还原为浮点权重，再与高精度激活 `x`（`float16 / bfloat16`）在 **C（Cube / AIC）侧**执行浮点 matmul，最后加浮点 `bias`，属于 **V->C kernel flow**。目录名中的 `_antiquant` 用于明确**不覆盖**输出再量化分支（`output_quant=False` 固定）。

## 2. 算子定义

```text
weight_dq = (weight + antiquantOffset) * antiquantScale     # int8 -> float32 反量化
y         = x @ weight_dq + bias
```

- `antiquant_group_size == 0`：per-channel scale，`antiquantScale / antiquantOffset` shape 为 `[N]`，沿 K 维广播同一组系数。
- `antiquant_group_size > 0`：K 维 per-group scale，shape 为 `[ceil(K/G), N]`，第 `g` 个 group 覆盖 `weight[g*G : min((g+1)*G, K), :]`。

## 3. 接口规范

```python
weight_quant_batch_matmul_v2(x, weight, antiquantScale, antiquantOffset, bias, transpose_x=false, transpose_weight=false, antiquant_group_size=0, output_quant=false, y_dtype="float32") -> y
```

| 参数 | 输入/输出 | dtype | shape | 说明 |
|------|-----------|-------|-------|------|
| `x` | 输入 | `FLOAT16 / BFLOAT16` | `[M, K]` | 左矩阵（高精度激活） |
| `weight` | 输入 | `INT8` | `[K, N]` | 伪量化权重 |
| `antiquantScale` | 输入 | `FLOAT32` | `[N]` 或 `[ceil(K/G), N]` | 权重反量化 scale |
| `antiquantOffset` | 输入 | `FLOAT32` | 同 `antiquantScale` | 权重反量化 offset |
| `bias` | 输入 | `FLOAT32` | `[N]` | 浮点 bias |
| `y` | 输出 | `FLOAT32` | `[M, N]` | matmul 结果 |

固定参数：`transpose_x = False`、`transpose_weight = False`、`output_quant = False`、`y_dtype = "float32"`。

## 4. 约束说明

- 固定 `transpose_x = False`、`transpose_weight = False`。
- 覆盖 **per-channel**（`antiquant_group_size == 0`）和 **per-group along K**（`antiquant_group_size > 0`，本 benchmark 取 `32 / 64 / 128`）两类反量化路径；不覆盖输出量化分支（`output_quant=False` 固定）。
- INT4 / FRACTAL_NZ 打包细节不进入 benchmark 数学定义。
- `bias`、`antiquantScale`、`antiquantOffset` 均为 `FLOAT32`；`y` 输出为 `FLOAT32`。

## 5. 实现约束与参考设计

> 本节分两层。**§5.1 硬约束**是正确性与合法性底线，必须遵守；**§5.2 参考设计思想与已知反模式**是非强制指导——只给出已验证设计思路和避坑提示，**不提供代码模板，也不要求照抄既有实现。** 通用 CV 反作弊约束亦见 §6，与本节互为补充。

### 5.1 硬约束（正确性 + 反作弊，必须遵守）

- **目标语义与乘加顺序**（与 §2 / golden 完全一致）：

  ```text
  w     = Cast<float>(weight)                       # [K, N] int8 -> float
  # per-channel (antiquant_group_size == 0):
  w_dq  = (w + antiquantOffset[None, :]) * antiquantScale[None, :]      # 沿 K 广播 [N]
  # per-group along K (antiquant_group_size == G > 0):
  for g, [start:end] in groups_of_K(G):
      w_dq[start:end, :] = (w[start:end, :] + antiquantOffset[g, :]) * antiquantScale[g, :]
  y     = x.to(float32) @ w_dq + bias[None, :]      # [M, N] float32
  ```

  **乘加顺序必须保持为** `(weight_fp + offset) * scale`，再 `x @ w_dq`，最后 `+ bias`：不得把 `offset * scale` 预合、不得把 `scale` 折进 `x`、不得改变浮点累加/舍入顺序，否则与 golden 产生精度残差。`bias` 累加必须保持 `float32` 精度。
- **反量化语义双路径**：必须正确实现 per-channel 与 per-group 两类 scale/offset 语义——`antiquant_group_size == 0` 时系数 shape 为 `[N]`、沿 K 维广播同一组；`antiquant_group_size > 0` 时系数 shape 为 `[ceil(K/G), N]`、第 `g` 个 group（`g = k // G`）覆盖对应 K 行。两者均须与 golden 一致；validator 会对 shape 与 `antiquant_group_size` 不匹配直接判错。
- **固定语义边界**：`output_quant = False`（不覆盖输出再量化）、`y_dtype = float32`、`transpose_x = False`、`transpose_weight = False`；dtype 固定 `x ∈ {float16, bfloat16}`、`weight = int8`、`antiquantScale / antiquantOffset / bias = float32`，且 `x` 的 `float16` 与 `bfloat16` 两种输入都必须正确支持。
- **真融合，禁退化**：必须生成真正的 Cube + Vector 融合 AscendC kernel；**禁止**退化为纯 AIV / 纯 AIC / 纯 CPU / torch 计算 / aclnn 高层组合算子 / Python fallback。
- **权重反量化必须片上向量化在 AIV**：`cast int8 → float`、`+offset`、`*scale` 的反量化必须在片上 AIV/Vector 完成；**禁止**在 AIC/Cube 侧用元素级反量化模拟，**禁止**下沉到 torch / host / CPU / aclnn / Python。
- **matmul 必须落 Cube/AIC**：`x @ w_dq` 浮点矩阵乘（`M×K @ K×N → M×N`）必须由 AIC/Cube 用 AscendC Cube / MatMul / MMAD 原语完成；**禁止**在 AIV 侧用逐元素循环模拟矩阵乘。
- **`bias` 累加**：可在 AIC 收尾或 AIV 写回阶段完成，但必须保持 `float32` 精度累加并直接写回 GM；**禁止**用 torch 或 host 端计算输出。
- **跨核同步正确性**：必须正确处理 **AIV → AIC**（V→C）同步，保证 AIV 写入的 `w_dq` 对 AIC 可见、不出现数据竞争（否则结果错）；**不得用局部 barrier 冒充跨 AIC/AIV 的可见性同步**。
- kernel `__global__` 名与 host `_do` 入口名必须含 `custom`；AscendC 热路径禁止标量逐元素 `GetValue/SetValue` 循环（少量边界 / 控制元数据除外），必须使用块级 / 向量化原语。
- 精度遵循 §7 / [`benchmark/PRECISION_SPEC.md`](benchmark/PRECISION_SPEC.md) 与 `proto.yaml` 的 `precision` 节点。

### 5.2 参考设计思想与已知反模式（指导，非强制，鼓励超越）

> 以下只给出已验证历史任务沉淀出的**设计思想**，不提供代码、文件结构或可复制实现。agent 应把它当作架构约束与性能复盘线索，而不是照抄模板；若当前 case 分布不同，可在守住 §5.1 的前提下重新设计。

- **整体分工思想**：把算子拆成 AIV producer、AIC consumer、AIV epilogue 三段。AIV 只负责权重反量化与必要的写回后处理；AIC 只负责主矩阵乘累加；bias 可以放在 AIC 收尾或 AIV epilogue，但必须只在 K 全部累加完成后按输出 tile 加一次。
- **流式 workspace 思想**：不要把完整 `[K, N]` 反量化权重一次性物化。按 K 分块和 N 分片流式生成当前 tile 的 `w_dq`，AIV 写入 task-local workspace 后立刻交给 AIC 消费；workspace slot 的生命周期应清楚表达为“V 写当前 K tile → C 读当前 K tile → 进入下一 K tile”。
- **tiling 思想**：历史可行路径偏向“小/中 M 整块、N 按列分片、K 循环流式”的组织方式。每个 AIC/AIV 协作单元处理一个连续 N 分片，M 做 16 对齐后的整块或少量整块，K 按 group 或固定块长推进。这样可以减少跨 N 写冲突，并让 bias、scale、offset 都以连续 N 向量加载。
- **per-channel / per-group 统一思想**：per-channel 时所有 K tile 使用同一组 `[N]` 系数；per-group 时让 K tile 尽量与 `antiquant_group_size` 对齐，使每个 K tile 只使用一行 group 系数。若 shape 导致尾块不完整，仍按 valid K / valid N 做边界处理，不能改变 group 归属。
- **AIV 并行思想**：同一个 N 分片内，AIV 可以按 K 行或子块分摊反量化工作，避免多个 vector lane 写同一连续缓存区域造成冲突。反量化后的数据应以适合 Cube 读取的连续布局落到 workspace，而不是后续再做大规模重排。
- **AIC 累加思想**：AIC 对每个 K tile 消费 `x` tile 与已反量化权重 tile，跨 K tile 保持同一个输出 tile 的累加状态。只有第一个 K tile 初始化累加器，后续 K tile 继续累加，最后统一写回输出。
- **精度保持思想**：若 Cube 路径不能直接以足够高精度消费反量化权重，可考虑把反量化结果拆成主值与残差两路分别参与 Cube 累加，以逼近 float32 反量化精度；是否采用由 correctness 结果决定，不应牺牲 §5.1 的 golden 语义。
- **同步思想**：本算子是典型 V→C handoff，局部 pipe/barrier 只能保证本核流水排空，不能替代 AIV/AIC 之间的跨核可见性同步。每个 K tile 必须有清晰的“V 已写好、C 可读取”信号；若 bias 放到 AIV epilogue，还需要有“C 已完成输出 tile”信号。
- **尾块与 padding 思想**：M / N / K 为硬件友好对齐而 padding 时，计算只允许在 padding 区域补零或丢弃尾部，不得让 padding 数据参与真实输出。所有 GM 读写都要围绕 valid M / valid N / valid K 做边界保护。
- **已知反模式（建议避开）**：完整物化大 `[K, N]` 权重再计算、AIC 侧逐元素反量化、AIV 侧模拟 matmul、每个输出元素标量 epilogue、只用局部 barrier 当跨核同步、per-group 时让一个 K tile 跨多个 group 却不拆分系数，都会导致正确性或性能风险。
- 设计文档与 `trace.md` 应记录：AIC/AIV 分工、N 分片与 K tile 策略、workspace 生命周期、跨核同步点、per-channel/per-group 系数加载方式、bias 放置位置、tail/padding 处理方式，以及若使用残差双路累加时的精度验证依据。

## 6. 强制 CV 算子约束（无退化）

1. 必须生成**真正的 Cube + Vector 融合 AscendC kernel**，**禁止退化**为：纯 AIV / 纯 AIC / 纯 CPU / torch / aclnn 高层组合算子 / Python fallback。
2. **权重反量化必须由 AIV/Vector 侧完成**：GM 读 `int8 weight` 分块 → `cast int8 → float` → 按 `antiquant_group_size` 广播 `offset / scale` → `w_dq = (weight_fp + offset) * scale` 写 workspace；**禁止**在 AIC/Cube 侧用元素级反量化模拟。
3. **`x @ w_dq` 的矩阵乘必须由 AIC/Cube 侧完成**（Cube / MatMul / MMAD 原语）；**禁止**在 AIV 侧用逐元素循环模拟矩阵乘。
4. **`bias` 累加**可在 AIC 收尾或 AIV 写回阶段，但必须 `float32` 精度累加并直接写回 GM；**禁止**用 torch 或 host 端计算输出。
5. AscendC kernel 必须采用 **V→C 数据流**：AIV 将 `w_dq` 分块写 workspace，AIC 读取该中间结果与 `x` 完成 matmul，并正确处理 AIV/AIC 同步；逐 group 流水时必须说明 group 与 cube tile 对齐方式。
6. kernel tiling / launch 必须体现 **AIC + AIV 混合执行**；设计文档与 `trace.md` 必须说明：AIC/AIV 分工、workspace 布局（`w_dq` 中间缓冲 shape/dtype/分块）、同步方式（V→C flag / barrier）、per-channel vs per-group 系数加载策略。
7. `model_new_tilelang.py` 与 `model_new_ascendc.py` 中**禁止使用 torch 算子做任何实际计算**；kernel `__global__` 名与 host `_do` 入口名必须含 `custom`。

## 7. 精度要求

本算子精度判定遵循 [`benchmark/PRECISION_SPEC.md`](benchmark/PRECISION_SPEC.md)。通过条件与阈值参数定义在同目录 `proto.yaml` 的 `precision` 节点，以下仅说明本算子特定的取舍。

### 7.1 算子特定说明

- **`y` 阈值归属**：规则 `input_dtype_inherited`，更贴近 NPU 上 FP16 / BF16 输入参与 matmul 的误差上限。
  - `bfloat16` 输入：阈值 `2^-7`。
  - `float16` 输入：阈值 `2^-10`。
- **乘加顺序**：精度强约束见 §5.1，`((weight_fp + offset) * scale)` → `x @ w_dq` → `+ bias`（以 golden 为准），不得预合 offset / scale，`bias` 必须 float32 累加。

## 8. 强制验收约束

1. **正确性硬门**：所有 `cases.yaml` / `cases.csv` 中的用例**精度必须全部达标**；不得只通过 `basic_case` 或部分 `general_case` 后停止。validator 会对每个 case 跑 golden，**反量化 scale/offset shape 与 `antiquant_group_size` 不匹配会被直接判错**（`gs==0` 必须 `[N]`；`gs>0` 必须 `[ceil(K/G), N]`）。
2. **性能软门**：所有用例的计算速度必须优于 torch 小算子拼接基线（duration-only 口径）；任一用例性能未达标，必须继续优化 AscendC tiling、workspace 流水或 vector 反量化路径，直到达标或在 `trace.md` 记录明确阻塞原因。性能为软门、不阻塞交付，但正确性始终是硬门。
3. 全量验证失败时调用 **debugger 子流程**继续修复，不要停下来问用户；TIMEOUT / 卡死先触发 timeout watchdog。

## 9. 标准 Golden 代码

`golden.py` 先根据 `antiquant_group_size` 广播 `antiquantScale / antiquantOffset` 生成 `weight_dq`（`gs==0` per-channel 广播 `[N]`；`gs>0` 逐 group 拼接 `[ceil(K/G), N]`），反量化到输入精度 `T=x.dtype`（与硬件 A16W8 反量化精度一致，保留 int8→fp16/bf16 的舍入），再在 fp32 累加器上 `x.to(float32) @ weight_dq.to(float32) + bias`。golden 内部对 `antiquantScale / antiquantOffset` 的 shape 做强校验：per-channel 期望 `[N]`，per-group 期望 `[ceil(K/G), N]`，否则抛错。

为消除小值域/相消退化（见 contributing.md §2.4），另提供 `_oracle`：与 plain 同结构，但反量化与 matmul 全程跟随输入精度、不硬编码 `.float()`，在 `golden_precision=fp64_cpu` 下整条在 fp64 计算，是精确反量化的 fp64 真值上界，使 `|bench − oracle|` 不再恒为 0。

## 10. 额外信息

### 测试资料对应关系

- `docs/aclnnWeightQuantBatchMatmulV2.md`：反量化公式、per-channel / per-group shape 约束。
- `op_kernel/anti_quant.h`：权重反量化辅助逻辑。

### 本 benchmark case 设计

`cases.yaml` / `cases.csv` 当前包含 **20 个正向 case，1:1 对应**，覆盖 BF16 / FP16 输入、per-channel（`gs=0`）与 per-group（`gs=32 / 64 / 128`）两类路径、不同 `M / K / N` 与 tail M：

- **少量小用例（6 个，smoke / edge）**：最大单维 < 512；覆盖 `gs=0`（per-channel）与 `gs=32 / 64 / 128`（per-group）、`M=1` decode 边界、tail K（如 `K=96 / 192`）。
- **大量 LLM 用例（14 个）**：真实 LLM 规模 `M / K / N`，单维**放大至最大 5120、不超过 5120**；覆盖 prefill 大 M、decode `M=1`、非对齐 M（`333 / 1000 / 640`）与非对齐 K（`4608`）、`gs=0` 与 `gs=64 / 128` 两路、per-group scale shape `[ceil(K/G), N]`（如 `K=512, gs=128 → [4, N]`；`K=5120, gs=128 → [40, N]`）。

所有 case 固定 `value_range = [-2, 2]`、`baseline_perf_us = 0.0`、`t_hw_us = 0.0`。

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


def weight_quant_batch_matmul_v2(
    x: torch.Tensor,
    weight: torch.Tensor,
    antiquantScale: torch.Tensor,
    antiquantOffset: torch.Tensor,
    bias: torch.Tensor,
    transpose_x: bool = False,
    transpose_weight: bool = False,
    antiquant_group_size: int = 0,
    output_quant: bool = False,
    y_dtype: str = "float32",
) -> torch.Tensor:
    """Torch golden for weight_quant_batch_matmul_v2 antiquant matmul path.

    同精度参考 (bench b)：int8 权重反量化到输入精度 T=x.dtype（与硬件 A16W8 反量化
    精度一致，保留 int8→fp16/bf16 的舍入），fp32 累加器做 matmul，输出为输出精度。
    fp64 数学真值见 ``weight_quant_batch_matmul_v2_oracle``；拆分约定见
    docs/guide/contributing.md §2.4。
    """
    if output_quant:
        raise ValueError("This benchmark fixes output_quant=False")
    if transpose_x:
        x = x.transpose(-2, -1)
    if transpose_weight:
        weight = weight.transpose(-2, -1)
    m, k = x.shape
    k2, n = weight.shape
    if k != k2 or bias.shape != (n,):
        raise ValueError("shape mismatch")
    T = x.dtype
    w = weight.to(T)
    if antiquant_group_size == 0:
        if antiquantScale.shape != (n,) or antiquantOffset.shape != (n,):
            raise ValueError("per-channel antiquant expects [N] scale/offset")
        w_dq = (w + antiquantOffset.reshape(1, n).to(T)) * antiquantScale.reshape(1, n).to(T)
    else:
        group_num = (k + antiquant_group_size - 1) // antiquant_group_size
        if antiquantScale.shape != (group_num, n) or antiquantOffset.shape != (group_num, n):
            raise ValueError("per-group antiquant expects [ceil(K/group),N] scale/offset")
        chunks = []
        for g, start in enumerate(range(0, k, antiquant_group_size)):
            end = min(start + antiquant_group_size, k)
            chunks.append((w[start:end, :] + antiquantOffset[g:g + 1, :].to(T)) * antiquantScale[g:g + 1, :].to(T))
        w_dq = torch.cat(chunks, dim=0)
    # fp32 累加器（tensor-core 约定）：T 操作数升 fp32 相乘累加，保留已有的 T 舍入
    return x.to(torch.float32) @ w_dq.to(torch.float32) + bias.to(torch.float32).reshape(1, n)


def weight_quant_batch_matmul_v2_oracle(
    x: torch.Tensor,
    weight: torch.Tensor,
    antiquantScale: torch.Tensor,
    antiquantOffset: torch.Tensor,
    bias: torch.Tensor,
    transpose_x: bool = False,
    transpose_weight: bool = False,
    antiquant_group_size: int = 0,
    output_quant: bool = False,
    y_dtype: str = "float32",
) -> torch.Tensor:
    """A16W8 antiquant 的数学真值 (g)，见 docs/guide/contributing.md §2.4。

    与 plain golden 同结构，但反量化 (weight + offset) * scale 与 matmul 全程跟随输入
    精度、不硬编码 .float()/.double() —— 在 golden_precision=fp64_cpu 下 x 升为 fp64，
    整条在 fp64 计算，是精确反量化的 fp64 真值上界（不再被下采成 fp32），使
    |bench − oracle| 不再恒为 0。输出 dtype 跟随 x.dtype。
    """
    if output_quant:
        raise ValueError("This benchmark fixes output_quant=False")
    if transpose_x:
        x = x.transpose(-2, -1)
    if transpose_weight:
        weight = weight.transpose(-2, -1)
    m, k = x.shape
    k2, n = weight.shape
    if k != k2 or bias.shape != (n,):
        raise ValueError("shape mismatch")
    cdt = x.dtype
    w = weight.to(cdt)
    if antiquant_group_size == 0:
        if antiquantScale.shape != (n,) or antiquantOffset.shape != (n,):
            raise ValueError("per-channel antiquant expects [N] scale/offset")
        w_dq = (w + antiquantOffset.reshape(1, n).to(cdt)) * antiquantScale.reshape(1, n).to(cdt)
    else:
        group_num = (k + antiquant_group_size - 1) // antiquant_group_size
        if antiquantScale.shape != (group_num, n) or antiquantOffset.shape != (group_num, n):
            raise ValueError("per-group antiquant expects [ceil(K/group),N] scale/offset")
        chunks = []
        for g, start in enumerate(range(0, k, antiquant_group_size)):
            end = min(start + antiquant_group_size, k)
            chunks.append((w[start:end, :] + antiquantOffset[g:g + 1, :].to(cdt)) * antiquantScale[g:g + 1, :].to(cdt))
        w_dq = torch.cat(chunks, dim=0)
    y = torch.matmul(x, w_dq) + bias.to(cdt).reshape(1, n)
    return y.to(x.dtype)
```
