# Sparse4to2QuantMatmul 算子 API 描述（sparse4to2quant_matmul_dequant_weight_nz）

## 1. 算子简介

**算子特征**：
- 难度等级：L3（Contraction）

`sparse4to2quant_matmul_dequant` 对齐源码目录 `ops-nn/matmul/sparse4to2quant_matmul`，对应 aclnn 接口 `aclnnSparse4to2QuantMatmulWeightNz`。该算子完成 **4:2 稀疏量化的矩阵乘**：C（Cube）侧使用稀疏化的 INT8 `weight`（每连续 4 个元素恰好保留 2 个非零）+ 索引 `index` 重建做 INT8 matmul，V（Vector）侧执行 `xScale / sparseWeightScale` 反量化与可选 BF16 bias，属于 **C->V kernel flow**。

NPU 实际执行时：

1. host 端调用 `aclnnTransSparse4to2Para` 把稠密 `weight` 压缩为 `sparseWeight`（只保留非零元素，FRACTAL_NZ 布局）+ `index`（UINT8 4D 索引）。
2. device 端 C 段用 `sparseWeight + index` 重建做 INT8 matmul（Cube 内 INT32 累加），V 段做 dequant + 可选 bias，cast 回 BF16。

数学上与“稠密 weight 直接做 matmul”等价（零元素相乘贡献为 0），因此 `golden.py` 直接用稠密 weight 计算，与 NPU 端 `sparseWeight + index` 重建后的 INT8 matmul 结果一致。

产品支持情况：

| 产品 | 是否支持 |
|------|----------|
| Atlas A3 训练系列产品 / Atlas A3 推理系列产品 | 支持 |
| Atlas A2 训练系列产品 / Atlas A2 推理系列产品 | 支持 |

## 2. 算子定义

设 `x` 的形状为 `[M, K]`，稠密 `weight` 的形状为 `[N, K]`，且满足 **4:2 稀疏 pattern**（每连续 4 元素恰好 2 个为 0）。计算公式：

```text
out = (x.int32 @ weight.int32.T) * xScale[:, None] * sparseWeightScale[None, :] + bias
```

等价的逐步语义（**乘加顺序以 golden 为准**）：

```python
out_fp32 = torch.matmul(x.float(), weight.t().float())          # [M, N], int32 等价 fp32 累加（bit-exact）
out_fp32 = out_fp32 * xScale.float().reshape(-1, 1)             # per-token dequant（按 M 行）
out_fp32 = out_fp32 * sparseWeightScale.float().reshape(1, -1)  # per-channel dequant（按 N 列）
if with_bias and bias is not None:
    out_fp32 = out_fp32 + bias.float().reshape(1, -1)           # 可选 BF16 bias（cast 到 FP32 再加）
out = out_fp32.to(torch.bfloat16)                              # cast 到 BF16 写回
```

**乘加顺序必须保持为** `((matmul * xScale) * sparseWeightScale) + bias`：不得把 `xScale * sparseWeightScale` 或 `scale * bias` 预先合成后再乘 / 加，否则改变浮点舍入顺序，与 golden 产生精度残差。

## 3. 接口规范

### aclnn 两段式接口

```cpp
aclnnStatus aclnnSparse4to2QuantMatmulWeightNzGetWorkspaceSize(
  const aclTensor *x,
  const aclTensor *sparseWeight,        // host 端 aclnnTransSparse4to2Para 压缩得到（FRACTAL_NZ）
  const aclTensor *index,               // UINT8 4D 索引
  const aclTensor *xScale,
  const aclTensor *sparseWeightScale,
  const aclTensor *biasOptional,
  aclTensor       *out,
  uint64_t        *workspaceSize,
  aclOpExecutor   **executor)

aclnnStatus aclnnSparse4to2QuantMatmulWeightNz(
  void *workspace, uint64_t workspaceSize,
  aclOpExecutor *executor, aclrtStream stream)
```

### benchmark 抽象接口（driver 内部完成 trans 预处理）

```python
sparse4to2quant_matmul_dequant(
    x, weight, xScale, sparseWeightScale, bias=None, dtype=27, with_bias=True
) -> out
```

> benchmark 接口接受**已 4:2 稀疏化的稠密 `weight`**（`[N, K]`，50% 元素为 0）；driver 内部调用 `aclnnTransSparse4to2Para` 转换为 NPU 需要的 `sparseWeight`（FRACTAL_NZ）+ `index` 后再下发。

### 参数说明

| 参数 | 输入/输出 | dtype | format | shape | 说明 |
|------|-----------|-------|--------|-------|------|
| `x` | 输入 | `INT8` | ND | `[M, K]` | 量化激活矩阵 |
| `weight` | 输入 | `INT8` | ND | `[N, K]` | 已 4:2 稀疏化的稠密表示（50% 为 0）；driver 内部转 `sparseWeight`（FRACTAL_NZ）+ `index` |
| `xScale` | 输入 | `FLOAT32` | ND | `[M]` | per-token 反量化 scale |
| `sparseWeightScale` | 输入 | `FLOAT32` | ND | `[N]` | per-channel 反量化 scale |
| `bias`（可选） | 输入 | `BFLOAT16` | ND | `[N]` | 可选 bias，`with_bias=true` 时启用 |
| `out` | 输出 | `BFLOAT16` | ND | `[M, N]` | 反量化矩阵乘输出 |

### 属性（attrs）

| 名称 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `dtype` | int | `27` | op_def 声明的 REQUIRED Int 属性；tiling 当前写死 `out == BF16`，该 attr 实际未生效；benchmark 固定 `dtype = 27`（`ge::DT_BF16` 枚举值）以兼容。 |
| `with_bias` | bool | `true` | 是否启用 BF16 bias；本套件**同时覆盖 `with_bias=true` 与 `with_bias=false`**。 |

## 4. 约束说明

Atlas A2 / A3 产品约束（tiling 硬校验项不可放宽）：

- **`K ≤ 65535`**（docs 约束；tiling 代码未直接校验，依赖上游 `aclnnTransSparse4to2Para`）。`golden.py` 显式校验 `K > 65535` 时抛错。
- **`K` 不要求整除 8**：NPU 端通过 `CeilAlign(K, SPARSE_ATOMIC_SIZE=8)` 内部补齐；tiling 校验 `ceil(K/8)*8 == 2 * sparseWeight.K_half`。**但 `K` 必须为 4 的倍数**才能干净切 4:2 分组（golden 把 K reshape 为 `[N, K/4, 4]` 统计每组零元素个数；`K % 4 != 0` 直接无法构造）——这是 case 设计的**硬性隐含约束**。
- **`N` 不要求整除 16**：`sparseWeight` 为 FRACTAL_NZ，StorageShape `[ceil(N/16), ceil(K_half/32), 16, 32]`，NPU 内部按 16 ceil padding；输出 `out` 仍为逻辑 `[M, N]`，padding 字节零填充不污染输出。
- **`weight` 必须严格满足 4:2 稀疏 pattern**（每连续 4 元素恰好 2 个为 0），由 benchmark 数据准备阶段自动生成。若 weight 不满足该 pattern，NPU（基于压缩 `sparseWeight`）与 golden（基于稠密 weight）结果将不一致，**不应归因于精度问题**。
- 长度约束：`xScale` 长度 `== M`；`sparseWeightScale` 长度 `== N`；`with_bias=true` 时 `bias` 长度 `== N`。
- dtype 严格固定（tiling 硬校验）：`x` / `weight`（`sparseWeight`）= `INT8`；`xScale` / `sparseWeightScale` = `FLOAT32`；`bias`（可选） / `out` = `BFLOAT16`。
- 全部输入支持 `IgnoreContiguous`（非连续 tensor），但本 benchmark 只测连续 tensor。
- 本 benchmark 固定走 `aclnnSparse4to2QuantMatmulWeightNz` 路径，不覆盖未来可能扩展的其他输出 dtype。

## 5. 实现约束与参考设计

> 本节分两层。**§5.1 硬约束**是正确性与合法性底线，必须遵守；**§5.2 参考设计与已知反模式**是非强制指导——给出一条已验证可行的实现路径与避坑提示，**鼓励在守住 §5.1 的前提下自行设计更优方案；当生成算子性能不足时，应主动超越本参考路径，而非照抄。** 乘加顺序见 §2，精度归属见 §6。

### 5.1 硬约束（正确性 + 反作弊，必须遵守）

- **真融合，禁退化**：必须生成**真正的 Cube + Vector 融合 AscendC kernel**；**禁止退化**为纯 AIV（在 Vector 侧用逐元素循环模拟 INT8 矩阵乘）、纯 CPU、torch 计算（`model_new_tilelang.py` / `model_new_ascendc.py` 中禁止用 torch 算子做任何实际计算）、aclnn 高层组合算子、Python fallback。
- **INT8 sparse matmul 必须落 Cube**：稀疏矩阵乘 `x @ weight.T`（NPU 端等价于 `sparseWeight + index` 重建后的 INT8 matmul）必须由 AIC / Cube 用 AscendC Cube / MatMul / MMAD 原语完成，在 Cube 内以 **INT32 累加**（bit-exact）；**禁止在 AIV 侧用逐元素循环模拟矩阵乘**。
- **4:2 稀疏 weight 语义与 NZ 布局**：`weight` 必须严格满足 **4:2 稀疏 pattern**（每连续 4 元素恰好 2 个为 0，`K` 为 4 的倍数），NPU 端以 `sparseWeight`（FRACTAL_NZ，StorageShape `[ceil(N/16), ceil(K_half/32), 16, 32]`，仅保留非零元素）+ `index`（UINT8 4D 索引）形式存在，由 host 端 `aclnnTransSparse4to2Para` 从稠密 4:2 weight 压缩得到；Cube 侧必须按该稀疏格式（直接吃 NZ + sparse 格式）驱动 MMAD 或调用支持 4:2 sparse 的 cube matmul 接口，**禁止在 host 侧解压 / 重建权重**。weight 不满足该 pattern 时 NPU 与 golden 结果不一致，**不应归因于精度问题**。
- **反量化必须片上向量化（AIV）**：INT32→FP32、per-token（`xScale`）与 per-channel（`sparseWeightScale`）反量化、可选 BF16 bias、cast 到 BF16，必须在片上 AIV / Vector 完成；**禁止**下沉到 torch / host / CPU / aclnn / Python 计算输出，**禁止**把 scale / bias / cast 搬到 host。`xScale` / `sparseWeightScale` 为 `FLOAT32` 读入，`out` 为 `BFLOAT16` 写回；V 段只做 element-wise dequant + bias，**无激活函数、无跨核 atomic 累加**。
- **乘加顺序与语义**：`((matmul * xScale) * sparseWeightScale) + bias` 后 cast 到 BF16，与 §2 / golden 一致（详见 §2 / §4）；不得把 `xScale * sparseWeightScale` 或 `scale * bias` 预先合成后再乘 / 加。
- **shape / dtype 断言**：`K ≤ 65535`、`K` 为 4 的倍数（4:2 分组前提）；dtype 严格固定——`x` / `weight`（`sparseWeight`）= `INT8`，`xScale` / `sparseWeightScale` = `FLOAT32`，`bias`（可选）/ `out` = `BFLOAT16`；`xScale` 长度 `== M`、`sparseWeightScale` 长度 `== N`、`with_bias=true` 时 `bias` 长度 `== N`（详见 §4）。
- **跨核同步正确性**：AIC→AIV 交接必须正确同步、保证跨核数据可见，不得出现数据竞争（否则结果错）；不得用局部 `PipeBarrier` 冒充跨 AIC/AIV 的可见性同步。
- **反作弊命名 / 写法**：AscendC kernel 的 `__global__` 核函数名与 Host `_do` 入口名**必须包含 `custom`**（如 `<op>_custom` / `<op>_custom_<dtype>` / `..._do`），不得生成不含 `custom` 的 profiling kernel 名；热路径禁止标量 `GetValue/SetValue` 逐元素写法（少量边界 / 控制元数据除外），必须使用 `T.copy`、`T.tile.*`、矩阵 / 向量原语等块级或向量化操作。
- 精度遵循 §6 / [`benchmark/PRECISION_SPEC.md`](../PRECISION_SPEC.md) 与 `proto.yaml` 的 `precision` 节点。

### 5.2 参考设计与已知反模式（指导，非强制，鼓励超越）

> 以下是一条**已验证可行**的 C→V 实现路径与若干提示，作为起点与避坑参考，**不是必须照抄的配方**。在守住 §5.1 的前提下，鼓励 agent 探索更优实现；**性能不足时优先在此处突破。**

- **参考数据流（C→V）**：AIC 将分块 INT32 matmul 结果写入 workspace / GM ring slot，AIV 读取后完成反量化 + 可选 bias + cast 并 BF16 写回。一条可行的逐 tile V 段语义：

  ```text
  v = Cast<float>(acc_int32)        # 读取 cube 写入 workspace 的 INT32 中间结果，cast 到 FP32
  v = v * xScale[m]                 # per-token，按 M 行（尽量整块广播，避免逐行标量 GetValue）
  v = v * sparseWeightScale[n0:n1]  # per-channel，按 N 列整列向量乘
  v = v + bias[n0:n1]               # 可选：BF16 bias cast 到 FP32 再加（with_bias=true 时）
  out = Cast<bf16>(v)               # cast 到 BF16 写回 out
  ```

  这是一种直接可行的切分；agent 可自行探索更优方案（如把反量化融进 matmul epilogue、不同 tile/buffer 策略、1C2V 分工等）。
- **参考同步 / workspace**：一种可行编排——

  ```text
  for each tile:
    AIC 产出 int32 acc tile 写 workspace / GM ring slot
    AIC PipeBarrier<PIPE_ALL>（排空 GM 写）后 CrossCoreSetFlag 通知 AIV
    AIV CrossCoreWaitFlag -> 消费 tile -> 反量化 + 可选 bias + cast -> 写 out
    AIV 通知 AIC：slot 可复用
  ```

  其中每个 `CrossCoreSetFlag` 前 `PipeBarrier<PIPE_ALL>` 排空 GM 写、多 AIV lane 按 collective 语义推进、ring slot 维护明确 C2V/V2C 生命周期。具体同步原语（`PipeBarrier<PIPE_ALL>` / `CrossCoreSetFlag` / collective）与 buffer 布局由 agent 按性能选择，只要满足 §5.1 的同步正确性即可。workspace 可用：AIC/AIV tile 交接的 INT32 accumulator ring slot、ping-pong / queue style tile buffer、scale/bias 的 UB staging。
- **参考 padding / tiling**：`K` / `N` 非对齐时在 cube 侧按 `CeilAlign(K, 8)` / `ceil(N/16)` padding 且 padding 字节零填充（逻辑输出仍 `[M, N]`）；tiling 体现 **AIC + AIV 混合执行（1C2V）**，按 shape（含大 M，本套件 LLM case 覆盖 M 至 5120）与硬件资源自适应选 `baseM/baseN/baseK`、used AIC/AIV 核数、ring slot 数，避免小 tile 造成 cube 利用率过低。
- **已知反模式（建议避开）**：epilogue 逐行用标量 `GetValue` 读 per-token scale、且每个向量 op 后 `PipeBarrier<PIPE_V>` 的串行写法，在大 shape 下被 fence 串行主导；优先**多行整块** `[validM, N]` 向量处理。
- 建议在设计文档与 `trace.md` 记录：AIC/AIV 分工、workspace 布局（含 INT32 中间结果分块策略）、同步方式、4:2 稀疏 weight / index 的加载与重建路径，便于性能复盘。

## 6. 精度要求

本算子精度判定遵循 [`benchmark/PRECISION_SPEC.md`](../PRECISION_SPEC.md)。通过条件与阈值参数定义在同目录 `proto.yaml` 的 `precision` 节点，以下仅说明本算子特定的取舍。

### 6.1 算子特定说明

- **`out` 阈值归属**：规则 `output_dtype`，固定 BFLOAT16 → 阈值 `2^-7`。INT8 matmul 在 Cube 内 INT32 累加（bit-exact）+ V 段 FP32 dequant + cast 到 BF16，最终精度上限即输出 dtype，无中间精度损失隐患。
- **4:2 稀疏决定性**：`weight` 在数据准备阶段必须按 4:2 pattern 生成（每连续 4 元素恰好 2 个为 0）。NPU 端 `aclnnTransSparse4to2Para` 的压缩逻辑期望该 pattern；若 weight 不满足，NPU 与 golden 结果不一致**不应归因于精度问题**。
- **无 V 段非线性 / 无 atomic**：本算子 V 段只做 element-wise dequant + bias，无激活函数、无跨核累加，精度归属直接由 output dtype 决定。

## 7. 标准 Golden 代码

`golden.py` 使用 PyTorch FP32 完成 INT8 matmul 与反量化，最后按 `dtype` 枚举（`27=BF16`）cast 输出，核心逻辑：

```python
out_dtype = _DTYPE_ENUM.get(int(dtype), torch.bfloat16)   # 27 -> bfloat16
out_fp32 = torch.matmul(x.float(), weight.t().float())   # [M, N]
out_fp32 = out_fp32 * xScale.float().reshape(-1, 1)
out_fp32 = out_fp32 * sparseWeightScale.float().reshape(1, -1)
if with_bias and bias is not None:
    out_fp32 = out_fp32 + bias.float().reshape(1, -1)
out = out_fp32.to(out_dtype)
```

由于 `weight` 已是 4:2 稀疏化的稠密表示（50% 为 0），零元素参与 matmul 不贡献结果，故 golden 直接用稠密 weight 计算等价于 NPU 用压缩后的 `sparseWeight + index` 重建计算。golden 还会显式校验：`x`/`weight` 为 2D、`x.K == weight.K`、`K ≤ 65535`、`xScale` 长度 `== M`、`sparseWeightScale` 长度 `== N`、`with_bias` 时 `bias` 长度 `== N`，并验证 weight 满足 4:2 稀疏 pattern。

## 8. 强制验收约束

1. 所有 `cases.yaml` / `cases.csv` 中的用例**精度必须全部达标**；不得只通过 `basic_case` 或部分 `general_case` 后停止。正确性是**硬门**。
2. 所有用例的**计算速度必须优于 torch 小算子拼接实现**（duration-only 口径）；如任一用例性能未优于该基线，必须继续优化 AscendC tiling、workspace 流水或 vector dequant 路径，直到性能达标或在 `trace.md` 中记录明确阻塞原因。性能是**软门**（不阻塞交付，但需在 trace 标注未达标 case）。
3. 数据准备阶段必须按 **4:2 稀疏 pattern** 生成 `weight`（每连续 4 元素恰好 2 个为 0），且 **`K` 维必须是 4 的倍数**才能干净切 4:2 分组，否则 NPU 与 golden 结果不一致。
4. 每次执行验证脚本后，必须将 `PASS / FAIL / TIMEOUT` 事件写入 `current_task/knowledge_inbox/`；全量验证失败时调用 debugger 子流程继续修复，不停下来问用户（终止条件：连续 5 轮无改进，或 debugger 累计耗时超 30 分钟；触发终止须在 `trace.md` 写明阻塞原因、最后失败用例与最近代码改动 diff 摘要）。

## 9. 额外信息

### 测试资料对应关系

- `docs/aclnnSparse4to2QuantMatmulWeightNz.md`：aclnn 接口规约与端到端调用示例
- `op_kernel/sparse4to2quant_matmul.h`：C 段 sparse matmul + V 段 dequant 实现
- `examples/`：`M=64, K=512, N=128` 端到端样例

### 本 benchmark case 设计（本套件，刷新为 “少量小 + 多 LLM” 形态）

`cases.yaml` 当前包含 **23 个正向 case**，与原套件用例数对齐，按 “少量小规模 smoke/edge + 多数 LLM 尺度” 重排（所有 case 的 `dtype` 列与 `attrs` 结构与原套件一致，`value_range` 保持 `[-8, 8]`，`baseline_perf_us = t_hw_us = 0.0`）：

- **小规模 smoke / edge（9 个，case 1-9）**：`M ∈ {1,3,4,8,12,16,24,33}` 含 M=1 退化与非 2 的幂 tailM；`K ∈ {64,100,128,132,256}`（含 `K=132` 8 非对齐、`K=100` 非对齐，均为 4 的倍数）；`N ∈ {32,48,64,128}`（含 `N=48` 由 16 非对齐 ceil 而来、`N` 非 2 的幂）。覆盖最小 shape、单 token、tailM、小 K、K/N 非对齐边界。
- **LLM 尺度（14 个，case 10-23）**：`M / N / K` 放大至 **上界 5120（绝不超过）**：含 `K` 触顶 `5120`（case 18）、`M` 触顶 `5120`（case 19）、`N` 触顶 `5120`（case 20）、`4096×4096` 近方阵（case 16）；并含 ragged / 非对齐 LLM shape（如 `1152/2304/1536/3072/1500` 等非 16 对齐 N、`1000/1536/3500` 等非对齐 M）验证 cube 侧 `CeilAlign(K,8)` 与 `ceil(N/16)` padding。
- `with_bias=true`（13 个）与 `with_bias=false`（10 个）大致平衡，两类均覆盖小规模与 LLM 尺度。
- weight 数据由 validator / driver 按 4:2 pattern 自动注入（每连续 4 元素恰好 2 个为 0）；**case 设计只需保证 SHAPE 正确**，尤其 `K % 4 == 0`、`K ≤ 65535`、`M/N/K ≤ 5120`。

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


# CANN dtype 枚举 -> torch dtype（仅覆盖本 benchmark 路径涉及的输出类型）
_DTYPE_ENUM = {
    0: torch.float32,
    1: torch.float16,
    27: torch.bfloat16,
}


def sparse4to2quant_matmul_dequant(
    x: torch.Tensor,
    weight: torch.Tensor,
    xScale: torch.Tensor,
    sparseWeightScale: torch.Tensor,
    bias: torch.Tensor = None,
    dtype: int = 27,
    with_bias: bool = True,
):
    """Torch golden for aclnnSparse4to2QuantMatmulWeightNz.

    Notes:
      - ``weight`` is the 4:2-sparsified DENSE representation (every 4 consecutive
        elements have exactly 2 zeros). NPU compresses it via
        ``aclnnTransSparse4to2Para`` into ``sparseWeight`` + ``index``; the golden
        uses the dense form directly because zero elements contribute nothing to
        the matmul (mathematically equivalent).
      - This benchmark fixes the BF16 output + FP32 per-token/per-channel scale +
        optional BF16 bias path; ``dtype`` 驱动输出 dtype（27=BF16）。
    """
    out_dtype = _DTYPE_ENUM.get(int(dtype), torch.bfloat16)
    if x.dim() != 2:
        raise ValueError(f"x expects 2D [M, K], got {list(x.shape)}")
    if weight.dim() != 2:
        raise ValueError(f"weight expects 2D [N, K], got {list(weight.shape)}")

    m, k = x.shape
    n, wk = weight.shape
    if wk != k:
        raise ValueError(f"x.K ({k}) must match weight.K ({wk})")
    if k > 65535:
        raise ValueError(f"K ({k}) exceeds 65535")
    # K and N do NOT need to be aligned: NPU pads K via CeilAlign(K, 8) and pads N
    # via FRACTAL_NZ ceil(N/16). Golden uses dense weight; padding bytes on the
    # NPU side are zero-filled and do not pollute the logical [M, N] output.
    # (Non-aligned cases are valid; current cases.yaml only exercises aligned shapes.)
    if xScale.numel() != m:
        raise ValueError(f"xScale length ({xScale.numel()}) must match M ({m})")
    if sparseWeightScale.numel() != n:
        raise ValueError(f"sparseWeightScale length ({sparseWeightScale.numel()}) must match N ({n})")
    if with_bias:
        if bias is None:
            raise ValueError("with_bias=True but bias tensor is None")
        if bias.numel() != n:
            raise ValueError(f"bias length ({bias.numel()}) must match N ({n})")

    # Verify 4:2 sparsity pattern (every 4 consecutive elements have exactly 2 zeros).
    # Reshape weight to [N, K/4, 4] and count zeros per group.
    weight_view = weight.reshape(n, k // 4, 4)
    zeros_per_group = (weight_view == 0).sum(dim=-1)
    if not bool((zeros_per_group == 2).all()):
        raise ValueError(
            "weight does not satisfy 4:2 sparsity pattern "
            "(every 4 consecutive elements must have exactly 2 zeros). "
            "Check data preparation step."
        )

    out_fp32 = torch.matmul(x.to(torch.float32), weight.t().to(torch.float32))   # [M, N]
    out_fp32 = out_fp32 * xScale.to(torch.float32).reshape(-1, 1)
    out_fp32 = out_fp32 * sparseWeightScale.to(torch.float32).reshape(1, -1)
    if with_bias:
        out_fp32 = out_fp32 + bias.to(torch.float32).reshape(1, -1)
    return out_fp32.to(out_dtype)
```
