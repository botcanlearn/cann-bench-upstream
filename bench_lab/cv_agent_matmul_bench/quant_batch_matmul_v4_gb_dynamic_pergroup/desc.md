# QuantBatchMatmulV4 (G-B dynamic per-group) 算子 API 描述

## 1. 算子简介

**算子特征**：
- 难度等级：L3（Contraction）

`quant_batch_matmul_v4` 对齐源码目录 `ops-nn/matmul/quant_batch_matmul_v4`。该算子有 K-C、K-G、G-B、B-B、MX 等多种量化模式，本 benchmark 固定 **G-B per-group 动态量化**代表路径（`variant=GB_DYNAMIC_PERGROUP`）：V 侧沿 K 维按 group 动态求 `x1/x2` 的 scale，C 侧执行 INT8 量化 block matmul，V 侧反量化累加并加 `float32` bias，属于 **V→C→V kernel flow**。

目录名中的 `_gb_dynamic_pergroup` 用于明确本 benchmark 只覆盖该 kernel path；K-C / K-G / B-B / MX 量化和 FP8 细节不纳入本目录。

## 2. 算子定义

按 K group（`gs_k = group_size[2] = 128`）分块，对左/右矩阵分别做动态对称 INT8 量化，反量化 block matmul 后沿 K group 累加，最后加浮点 bias：

```text
for each K group j (K 沿 gs_k=128 切块，最后一组按 min(start+gs_k, K) 截断):
    s1[m, j] = amax(abs(x1[m, Kj])) / 127          # per-row(per-M) absmax，[M,1]
    s2[j, n] = amax(abs(x2[Kj, n])) / 127          # per-col(per-N) absmax，[1,N]
    a_q = round(x1[:, Kj] / s1).clamp(-127, 127)   # int8，[M, g]
    b_q = round(x2[Kj, :] / s2).clamp(-127, 127)   # int8，[g, N]
    out += (a_q @ b_q) * s1 * s2                    # 反量化累加，[M,N] float32
out += bias                                         # per-channel(per-N)，float32
```

固定 `group_size = [1, 128, 128]`。

## 3. 接口规范

```python
quant_batch_matmul_v4(
    x1, x2, bias,
    variant="GB_DYNAMIC_PERGROUP",
    transpose_x1=False, transpose_x2=False,
    group_size=(1, 128, 128),
) -> out
```

| 参数 | 输入/输出 | dtype | shape | 说明 |
|------|-----------|-------|-------|------|
| `x1` | 输入 | `FLOAT16/BFLOAT16` | `[M, K]` | 动态量化左矩阵 |
| `x2` | 输入 | 与 `x1` 相同 | `[K, N]` | 动态量化右矩阵 |
| `bias` | 输入 | `FLOAT32` | `[N]` | 浮点 bias |
| `out` | 输出 | `FLOAT32` | `[M, N]` | 反量化累加输出 |

`x1` 与 `x2` 的 `K` 必须相等，`bias` 长度必须等于 `N`，否则 golden 直接 `raise ValueError("shape mismatch")`。

### 3.1 attrs 结构

| attr | type | 取值 | 说明 |
|------|------|------|------|
| `variant` | str | `GB_DYNAMIC_PERGROUP`（固定） | 非该值时 golden `raise ValueError` |
| `transpose_x1` | bool | `false`（固定） | — |
| `transpose_x2` | bool | `false`（固定） | — |
| `group_size` | list[int] | `[1, 128, 128]`（固定） | `group_size[2]=128` 为 K 方向 group size `gs_k` |

## 4. 约束说明

### 4.1 语义与固定参数

- 固定 `variant = "GB_DYNAMIC_PERGROUP"`，本 benchmark 只覆盖该 kernel path。
- 固定 `transpose_x1 = False`、`transpose_x2 = False`。
- 固定 `group_size = (1, 128, 128)`，沿 `K` 维以 `gs_k = group_size[2] = 128` 分块；当 `K` 不能整除时最后一组按 `min(start + gs_k, K)` 截断（本 benchmark 所有 case 的 `K` 均为 128 的整数倍，K group 数 `= K / 128`，无尾块，但 kernel 仍须正确处理尾块语义）。
- `qmax = 127.0`，对称 INT8 量化范围 `[-127, 127]`。
- `eps = torch.finfo(float32).tiny`，用于 absmax 下限保护（`clamp_min(eps)`），避免除零。
- 反量化与累加均在 `float32` 中执行；`out` 输出固定为 `float32`。

### 4.2 scale（s1/s2）的归属——关键正确性约束

- **`s1`、`s2` 是 kernel 内部按 K group 动态求得的量化产物，不作为外部输入也不作为外部输出。** 因此本算子的外部张量只有 3 个：`x1[M,K]`、`x2[K,N]`、`bias[N]`；测试用例 `input_shape` 永远是这三者，**没有独立 scale 张量**。
- per-group 量化沿 **K 维**进行：`x1` 逐行（per-M）求 group 内 absmax 得 `s1`（逻辑形如 `[M, ceil(K/128)]`），`x2` 逐列（per-N）求 group 内 absmax 得 `s2`（逻辑形如 `[ceil(K/128), N]`）。这些 scale 在 kernel 内随 K group 循环即用即弃，**不出现在 proto.yaml 的 inputs / outputs 中**，也不出现在 case 的 `input_shape` 里。

### 4.3 K group 与本 benchmark case 设计

- K group size 固定 `gs_k = 128`；每个 case 的 K group 数 = `K / 128`（本 benchmark 所有 K 均整除 128）。
- `cases.yaml` 当前包含 **20 个正向 case**：**6 个 small（smoke/edge）+ 14 个 LLM 规模**。
  - small：`M∈{1,3,4,7,8,16}`、`K∈{128,256,384}`、`N∈{64,128,192,256}`，覆盖单 K group、多 K group、单行（M=1）、奇 M、tail N 等边界。
  - LLM：方阵扫描 `512²→1024²→2048²→4096²→5120²`，以及非对称投影 shape（deep-K / wide-N / long-M），单维上限严格 `≤ 5120`，**任一维都不会超过 5120**。K group 数从 1 一直覆盖到 40。
  - 全部 case `dtype` 列保持 `[x_dtype, x_dtype, float32]`（`x_dtype ∈ {bfloat16, float16}`，bf16/fp16 交替），`attrs` 结构与原始一致（`variant/transpose_x1/transpose_x2/group_size`），`value_range=[-2, 2]`，`baseline_perf_us=0.0`、`t_hw_us=0.0`。

## 5. 实现约束与参考设计

> 本节分两层。**§5.1 硬约束**是正确性与合法性底线，必须遵守；**§5.2 参考设计与已知反模式**是非强制指导——给出一条已验证可行的实现路径与避坑提示，**鼓励在守住 §5.1 的前提下自行设计更优方案；当生成算子性能不足时，应主动超越本参考路径，而非照抄。**

### 5.1 硬约束（正确性 + 反作弊，必须遵守）

**目标语义与乘加顺序**

```text
gs_k = group_size[2]                 # 固定 128
out  = zeros(M, N, float32)
for start in range(0, K, gs_k):
    end   = min(start + gs_k, K)
    a_blk = x1[:, start:end].to(float32)                              # [M, g]
    b_blk = x2[start:end, :].to(float32)                              # [g, N]
    s1 = a_blk.abs().amax(dim=1, keepdim=True).clamp_min(eps) / 127   # [M, 1]
    s2 = b_blk.abs().amax(dim=0, keepdim=True).clamp_min(eps) / 127   # [1, N]
    a_q = round(a_blk / s1).clamp(-127, 127)                          # int8 [M, g]
    b_q = round(b_blk / s2).clamp(-127, 127)                          # int8 [g, N]
    out += (a_q @ b_q) * s1 * s2                                      # [M, N] float32
out += bias.to(float32).reshape(1, N)
```

- **G-B 每 K-group 动态量化语义**：沿 K 维以 `gs_k = group_size[2] = 128` 分块；`x1` 逐行（per-M）求 group 内 absmax 得 `s1[M,1]`、`x2` 逐列（per-N）求 group 内 absmax 得 `s2[1,N]`；`clamp_min(eps)`（`eps = torch.finfo(float32).tiny`，下限保护避免除零）后 `/127`；`round` + `clamp(-127,127)`（`qmax=127.0`，对称 INT8 范围 `[-127,127]`）量化为 INT8；反量化与累加均在 `float32` 中执行。语义与 §2 / golden 一致。
- **固定参数**：`group_size = (1, 128, 128)`；`variant = "GB_DYNAMIC_PERGROUP"`；`transpose_x1 = False`、`transpose_x2 = False`。
- **K 尾块语义**：`K` 不整除 `gs_k` 时最后一组按 `min(start + gs_k, K)` 截断（本 benchmark 所有 case 的 `K` 均为 128 整数倍，无尾块，但 kernel 仍须正确处理尾块语义）。
- **乘加顺序必须与 golden 一致**：每个 K group 内 `(int32_partial) * s1 * s2` 后再沿 K group 累加，最后加 `float32` bias。不得把 `s1 * s2` 或 `scale * bias` 预先合成后再乘/加，会改变浮点舍入顺序，与 golden 产生精度残差。
- **dtype 与 shape 断言**：正确支持 `float16` 与 `bfloat16` 输入，`out` 固定 `float32`；`x1`/`x2` 的 `K` 必须相等、`bias` 长度等于 `N`（否则 golden `raise ValueError("shape mismatch")`）。`s1`/`s2` 为 kernel 内部按 K group 动态求得的量化产物，不作外部输入/输出（外部张量仅 `x1`/`x2`/`bias` 三个，见 §4.2）。
- **真融合，禁退化**：必须生成**真正的 Cube + Vector 融合 AscendC kernel**，禁止退化为以下任一形态：纯 AIV、纯 AIC、纯 CPU、torch、aclnn 高层组合算子、Python fallback。
- **matmul 必须落 Cube**：INT8 块矩阵乘 `a_q @ b_q`（每个 K group 内 `[M,128] × [128,N]`）**必须由 AIC/Cube 侧完成**，使用 AscendC Cube / MatMul / MMAD 原语；**禁止在 AIV 侧用逐元素循环模拟矩阵乘**。
- **动态量化与反量化累加必须片上向量化**：沿 K 维按 `gs_k=128` 分块；`x1` 逐行 absmax、`x2` 逐列 absmax；`clamp_min(eps)` 与 `/127` 求 `s1`、`s2`；`round` + `clamp(-127,127)` 量化为 INT8；反量化 `int32_partial * s1 * s2` 并沿 K group 累加到 float32；最终加 `float32` bias 并写回 `out`——这些动态量化与 dequant-accumulate **必须在片上 AIV/Vector 完成**，**禁止使用 torch / host / CPU / aclnn / Python 计算输出**，**禁止用 AIC 标量循环**替代向量化量化。
- **跨核同步正确性**：AIV→AIC（量化结果就绪）与 AIC→AIV（partial block 就绪）必须正确双向同步、保证跨核数据可见，不得出现读未写 / 写未消费即覆盖等数据竞争（否则结果错）；不得用局部 `PipeBarrier` 冒充跨 AIC/AIV 的可见性同步。
- kernel `__global__` 核函数名与 Host `_do` 入口名必须包含 `custom`（如 `<op_name>_custom` / `<op_name>_custom_<dtype>` 及对应 `_do`）；AscendC 热路径禁止标量逐元素写法（标量 `GetValue/SetValue` 循环），必须使用 `T.copy`、`T.tile.*`、矩阵/向量原语等块级或向量化操作（少量边界 / 控制元数据除外）。
- 精度遵循 §6 / [`../PRECISION_SPEC.md`](../PRECISION_SPEC.md) 与 `proto.yaml` 的 `precision` 节点。

### 5.2 参考设计与已知反模式（指导，非强制，鼓励超越）

> 以下是一条**已验证可行**的 V→C→V 实现路径与若干提示，作为起点与避坑参考，**不是必须照抄的配方**。在守住 §5.1 的前提下，鼓励 agent 探索更优实现；**性能不足时优先在此处突破。**

- **参考三阶段编排（V→C→V）**：这是一种直接可行的切分，agent 可自行探索更优方案（如不同 tile/buffer 策略、把部分反量化融进 matmul epilogue、调整 AIC/AIV 分工等）。
  - **AIV 第一阶段（动态量化）**：沿 K 维以 `gs_k=128` 分块；对 `x1` 逐行求 absmax、对 `x2` 逐列求 absmax；`clamp_min(eps)` 后 `/127` 得 `s1`、`s2`；`round` + `clamp(-127,127)` 将 `x1/x2` 量化为 INT8；将 INT8 `a_q`/`b_q` 与 `s1`/`s2` 写入 workspace。
  - **AIC 第二阶段（INT8 block matmul）**：从 workspace 读取 INT8 `a_q`、`b_q`，对每个 K group 执行 `[M,128] × [128,N]` 的 INT8 block matmul，输出 INT32 partial block 写回 workspace。
  - **AIV 第三阶段（反量化累加）**：读取 INT32 partial block，乘以对应 group 的 `s1 * s2` 做反量化，沿 K group 累加到 `float32` out buffer；最后加 `float32` bias 写回 `out`。
- **参考 workspace 布局**：暂存 INT8 `a_q`/`b_q`、`s1`/`s2`、INT32 partial。具体 buffer 布局与 slot 复用策略由 agent 按性能选择，只要满足 §5.1 的同步正确性即可。
- **参考同步原语 / 生命周期**：AIC 每次 `CrossCoreSetFlag` 前用 `PipeBarrier` 排空对应 pipe 的 GM 写；workspace slot（INT8 操作数 / INT32 partial）维护明确的 V2C / C2V 生命周期；多 AIV lane 的同步按 collective 语义，不能只让 lane0 推进全局进度。具体同步原语与 slot 生命周期选择由 agent 按性能决定，只要满足 §5.1 的同步正确性即可。
- **参考 tiling**：kernel tiling/launch 体现 **AIC + AIV 混合执行**；按 shape 自适应选 tile，避免过小 tile 导致 cube 利用率过低。
- **已知反模式（建议避开）**：epilogue 逐行用标量 `GetValue` 读 per-group `s1` / per-col `s2`、且每个向量 op 后 `PipeBarrier<PIPE_V>` 的串行写法，在大 shape 下被 fence 串行主导；优先**多行整块** `[validM, N]` 向量处理。
- 建议在设计文档与 `trace.md` 记录：AIC/AIV 三阶段分工（量化 / INT8 matmul / 反量化累加）、workspace 布局（INT8 `a_q`/`b_q`、`s1`/`s2`、INT32 partial）、K group 循环与 AIV↔AIC 同步方式、`float16`/`bfloat16` 分支路径，便于性能复盘。

### 5.3 实现流程要求（阶段零到阶段五）

1. 按工作流定义的**阶段零到阶段五**完整执行。
2. 生成 TileLang 设计与实现：`design/block_level/`、`design/tile_level/`、`model_new_tilelang.py`；若 TileLang 环境可用，应通过对应脚本验证。
3. 生成 AscendC kernel 与 Python 封装：`kernel/`、`model_new_ascendc.py`，并通过 AscendC 验证脚本。
4. 每次执行验证脚本后，必须将 `PASS / FAIL / TIMEOUT` 事件写入 `knowledge_inbox/`。
5. AscendC 基础用例通过后，备份 `kernel/` 与 `model_new_ascendc.py` 到 `archive_kernel/`。
6. 全量验证失败时调用 **debugger 子流程**继续修复，不要停下来问用户。**终止条件**（满足任一即停止 debugger 循环）：① 连续 5 轮迭代精度与性能均无改进；② debugger 子流程累计耗时超过 30 分钟。触发终止时必须在 `trace.md` 写明阻塞原因、最后一次失败用例与最近一次代码改动 diff 摘要。
7. 最终必须生成 `trace.md`、`knowledge_curation_done.md`、`performance_result.md`。

### 5.4 设计检查清单

> 本清单是 §5.1 硬约束与 §5.2 参考设计的自检汇总，供实现前快速核对；带 **AIC/Cube**、**AIV/Vector**、乘加顺序、dtype、`custom` 命名、禁退化等项为 §5.1 硬约束，其余为 §5.2 参考路径的落地提示。

- [ ] INT8 block matmul（每 K group `[M,128]×[128,N]`）由 **AIC/Cube** 完成，未在 AIV 侧模拟矩阵乘？
- [ ] 动态量化（per-row/per-col absmax + `clamp_min(eps)` + `/127` + `round` + `clamp(-127,127)`）与反量化累加由 **AIV/Vector** 完成？
- [ ] 数据流为 **V→C→V**，workspace 暂存 INT8 `a_q`/`b_q`、`s1`/`s2`、INT32 partial？
- [ ] 乘加顺序为 `(int32_partial * s1 * s2)` 沿 K group 累加后再加 `float32` bias（与 golden 一致，未预合 scale/bias）？
- [ ] `float16` 与 `bfloat16` 输入均正确，`out` 固定 `float32`？
- [ ] AIV↔AIC 双向同步，每个 `CrossCoreSetFlag` 前 `PipeBarrier`，workspace slot 有明确 V2C/C2V 生命周期？
- [ ] 未退化为纯 AIV / 纯 AIC / 纯 CPU / torch / aclnn 高层组合 / Python fallback？
- [ ] kernel 名含 `custom`，TileLang/AscendC 无 torch 计算、无标量逐元素写法？

## 6. 精度要求

本算子精度判定遵循 [`../PRECISION_SPEC.md`](../PRECISION_SPEC.md)。通过条件与阈值参数定义在同目录 `proto.yaml` 的 `precision` 节点，以下仅说明本算子特定的取舍。

### 6.1 算子特定说明

- **`out` 阈值归属**：规则 `input_dtype_inherited`。该路径输出 `float32`，但数值由 FP16/BF16 输入动态量化 round-trip 后生成，因此阈值随输入 dtype 选取：`proto.yaml` 中 `bfloat16: 2^-7`、`float16: 2^-10`。
- **乘加顺序**：精度强约束见 §5.1，`(int32_partial * s1 * s2)` 沿 K group 累加后再加 `bias`（以 golden 为准），不得预合 `s1*s2` 或 `scale*bias`。

## 7. 强制验收约束

1. 所有 `cases.yaml` / `cases.csv` 中的用例**精度必须全部达标**；不得只通过 `basic_case` 或部分 `general_case` 后停止。
2. 所有用例的**计算速度必须优于 torch 小算子拼接实现**；如任一用例性能未优于该基线，必须继续优化 AscendC tiling、workspace 流水或 vector 量化/反量化路径，直到性能达标或在 `trace.md` 中记录明确阻塞原因。
3. 正确性始终是硬门（全部 case 精度达标）；性能为软门，仅在未达标时触发优化迭代，预算用尽仍不达标时接受当前正确 kernel 并在 `trace.md` 标注「性能未达标」。

## 8. 标准 Golden 代码

`golden.py` 按 `groupSizeK = group_size[2] = 128` 沿 K 维分块动态量化 `x1/x2`（per-row/per-col absmax → `clamp_min(eps)` → `/127` → `round` → `clamp(-127,127)`），反量化 block matmul 后沿 K group 累加，最后加 `float32` bias。`golden.py` 为唯一精度基准，**不得修改其数学语义**。

## 9. 额外信息

### 9.1 测试资料对应关系

- `docs/aclnnQuantMatmulV5.md`：V4/V5 量化模式、G-B 公式和 `groupSize` 约束。
- `op_kernel/quant_batch_matmul_v4_pergroup.h`：per-group kernel 路径。
- `benchmark/cann` 中的动态 G-B 参考实现：用于交叉核对公式。

### 9.2 本 benchmark case 设计摘要

`cases.yaml` / `cases.csv` 共 20 个正向 case，1:1 对应：6 个 small（smoke/edge）+ 14 个 LLM 规模；覆盖 FP16/BF16、`K=128..5120`（均为 128 倍数，K group 数 1..40）、`N=64..5120`、`M=1..5120`，单/多 K group 与单/多 N tile；所有维度 `≤ 5120`。`s1`/`s2` 为内部量化产物，不进入 `input_shape`。

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


def quant_batch_matmul_v4(
    x1: torch.Tensor,
    x2: torch.Tensor,
    bias: torch.Tensor,
    variant: str = "GB_DYNAMIC_PERGROUP",
    transpose_x1: bool = False,
    transpose_x2: bool = False,
    group_size=(1, 128, 128),
) -> torch.Tensor:
    """Torch golden for QuantBatchMatmulV4 G-B dynamic per-group path."""
    if variant != "GB_DYNAMIC_PERGROUP":
        raise ValueError("This benchmark fixes variant=GB_DYNAMIC_PERGROUP")
    a = x1.t() if transpose_x1 else x1
    b = x2.t() if transpose_x2 else x2
    a = a.to(torch.float32)
    b = b.to(torch.float32)
    m, k = a.shape
    k2, n = b.shape
    if k != k2 or bias.shape != (n,):
        raise ValueError("shape mismatch")
    gs_k = int(group_size[2])
    eps = torch.finfo(torch.float32).tiny
    qmax = 127.0
    out = torch.zeros(m, n, dtype=torch.float32, device=x1.device)
    for start in range(0, k, gs_k):
        end = min(start + gs_k, k)
        a_blk = a[:, start:end]
        b_blk = b[start:end, :]
        s1 = a_blk.abs().amax(dim=1, keepdim=True).clamp_min(eps) / qmax
        s2 = b_blk.abs().amax(dim=0, keepdim=True).clamp_min(eps) / qmax
        a_q = torch.round(a_blk / s1).clamp(-qmax, qmax)
        b_q = torch.round(b_blk / s2).clamp(-qmax, qmax)
        out = out + (a_q @ b_q) * s1 * s2
    return out + bias.to(torch.float32).reshape(1, n)
```
