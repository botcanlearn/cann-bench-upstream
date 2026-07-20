# TransposeQuantBatchMatMul 算子 API 描述

## 1. 算子简介

**算子特征**：
- 难度等级：L3（Contraction）

`transpose_quant_batch_mat_mul` 选取 K-C 量化路径：先按 `permX1/permX2` 调整 batch matmul 输入视图，C 侧完成 INT8 batch matmul，V 侧执行 per-token/per-channel scale、bias 和可选 `permY`，属于 C->V kernel flow。

## 2. 算子定义

```text
A = permute(x1, permX1)
B = permute(x2, permX2)
Y = (A @ B) * x1Scale[..., :, None] * x2Scale[..., None, :] + bias[..., None, :]
out = permute(Y, permY)
```

## 3. 接口规范

```python
transpose_quant_batch_mat_mul(x1, x2, x1Scale, x2Scale, bias, permX1, permX2, permY, groupSize=0, batchSplitFactor=1, y_dtype="float32") -> out
```

| 参数 | 输入/输出 | dtype | shape | 说明 |
|------|-----------|-------|-------|------|
| `x1` | 输入 | `INT8` | `[B,M,K]` 或 `[B,K,M]` | 左矩阵，按 `permX1` 解释 |
| `x2` | 输入 | `INT8` | `[B,K,N]` 或 `[B,N,K]` | 右矩阵，按 `permX2` 解释 |
| `x1Scale` | 输入 | `FLOAT32` | `[B,M]` | per-token scale |
| `x2Scale` | 输入 | `FLOAT32` | `[B,N]` | per-channel scale |
| `bias` | 输入 | `FLOAT32` | `[B,N]` | 浮点 bias |
| `out` | 输出 | `FLOAT32` | `[B,M,N]` 或 `[B,N,M]` | 输出矩阵 |

## 4. 约束说明

- 固定 K-C 路径，`K=512`、`N=128`，`groupSize=0`。
- 覆盖 `permX1/permX2/permY` 的常见 `[0,1,2]` 与 `[0,2,1]` 组合。
- 目录名中的 `_kc` 用于明确本 benchmark 固定 K-C 量化路径；MX 量化和 FP8 细节不纳入本目录。

## 5. 实现约束与结构设计（强制 C→V 融合路径）

> 本节为算子的硬性实现契约，与 §2 计算语义同等约束力；AscendC 实现与调试前必须先读取并遵守。通用 CV 反作弊约束（禁 torch 计算、kernel 名含 `custom`、禁退化为 aclnn 高层组合 / 纯 AIV / 纯 CPU / Python fallback）统一见 `SKILL.md` 全局约束，此处不复述。

### 5.1 目标语义与乘加顺序

```text
A = permute(x1, permX1)            # [B, M, K] int8
B = permute(x2, permX2)            # [B, K, N] int8
Y = (A @ B)                        # [B, M, N] int32 -> float32
Y = Y * x1Scale[:, :, None]        # per-token (per-M 行)
Y = Y * x2Scale[:, None, :]        # per-channel (per-N 列)
Y = Y + bias[:, None, :]           # per-channel
out = permute(Y, permY)            # [B, M, N] 或 [B, N, M], float32
```

固定参数：`K = 512`、`N = 128`、`groupSize = 0`、`batchSplitFactor = 1`、`y_dtype = float32`。`permX1/permX2/permY` 仅覆盖 `[0,1,2]` 与 `[0,2,1]`。

**乘加顺序必须保持为** `((matmul * x1Scale) * x2Scale) + bias`（与 golden 一致）：不得把 `x1Scale * x2Scale` 或 `scale * bias` 预先合成后再乘/加，会改变浮点舍入顺序，与 golden 产生精度残差。

### 5.2 硬件 / 高阶 API 前提

本算子的结构选择**由真机卡能力约束**，已确认：

```text
支持   : 高阶 Matmul API（AscendC::MatmulImpl，含 aTrans/bTrans 转置）
不支持 : L0C -> VECIN UB 直通（IsSupportL0CToUB）—— int32 中间结果必须落 GM
不支持 : MicroAPI __VEC_SCOPE__ / RegTensor 寄存器级反量化（VF dequant）
```

由此两条硬约束：

- **不得依赖 L0C->UB 直通**（`cT = MatmulType<VECIN, ...>`）**与 MicroAPI 寄存器级反量化**的实现路径——本卡不支持，int32 中间结果必须落 GM。
- 主路径必须是**框架 GM 输出模式 + 标准 vector 反量化**（见 5.4 / 5.5）。

### 5.3 可借鉴 / 不可照搬的技术

**可借鉴**：

- `MatmulType<GM, ND, int8, bTrans>` + `SetTensorB(B, bTrans)`：转置交给 cube 的 nd2nz/LoadData 在载入时完成，**B 原样喂源 GM，向量侧不物化转置**。
- 蛇形 tile 调度（奇数行 `nIndex` 反向）提升 A/B 在 L2 的复用。
- 自适应 tiling（`baseM/baseN/baseK/usedCoreNum` 由 host 按 shape 选）。

**不可照搬（本卡不支持，必须改）**：

- `cT = MatmulType<VECIN, ...>`（L0C->UB） → 改为 `cT = MatmulType<GM, ND, int32>`。
- MicroAPI 寄存器级反量化 → 改为标准 vector op 反量化。

### 5.4 禁止的主路径

反例结构**禁止作为主性能路径**：

```text
AIV NormalizeInput 预处理：对转置 x2 做每 batch 512 次 strided 1-byte gather 物化 [K,N]
  -> 两阶段 launch（phase0 normalize / phase1 compute）
  -> 手写 MMAD + baseM=32 写死 tiling
  -> epilogue 逐行 4 op，每 op 后 PipeBarrier<PIPE_V>，per-token 用标量 GetValue 逐行读
  -> permY=021 时 y_tmp(GM) 暂存 + 逐列 gather 第二次转置
```

历史证据：转置物化是每 batch 512 次字节粒度 gather，转置 x2 case 实测 `~3.1-3.9 ms`（`0.03-0.08x`）；手写 MMAD 的 `bTrans` 改造全部精度崩（L1->L0B 转置朝向调不对，已回滚）；`baseM=32` 大 M 下海量小 tile、cube 占核极差、B 反复重载；L0C->GM->UB int32 往返 + permY 二次 gather + epilogue 逐行 barrier，大 shape 下被 GM 流量与 fence 主导。

**禁止**把「向量侧转置物化（NormalizeInput）」「手写 MMAD」「`baseM=32` 固定 tiling」作为主路径。

### 5.5 推荐主路径：框架 GM 输出 + 标准 vector epilogue

**Cube (AIC)：高阶 MatmulImpl，GM 输出**

```text
aT = MatmulType<GM, ND, int8,  aTrans>   aTrans = (permX1 == [0,2,1])
bT = MatmulType<GM, ND, int8,  bTrans>   bTrans = (permX2 == [0,2,1])
cT = MatmulType<GM, ND, int32>           # GM 输出，绕开不支持的 L0C->UB
mm.SetTensorA(x1, aTrans); mm.SetTensorB(x2, bTrans);
mm.Iterate(); mm.GetTensorC(<int32 GM tile>);
```

- **转置完全交给框架**：permX1->aTrans、permX2->bTrans，cube 直接吃源 GM 布局。**`NormalizeInput`、两阶段 launch 整体删除。**
- tiling 由框架按 shape（尤其大 M 256-2048）选 `baseM/baseN/baseK` 并占满 cube 核，**不得硬编码 `baseM=32`**。

**C->V 交接（GM ring，本卡强制）**

- cube 把 int32 tile 写入 GM ring slot；`CrossCoreSetFlag/WaitFlag` 握手；两个 AIV lane 按 collective 参与。
- 仅做 tile 级交接，**不得物化完整 `contrib[B,M,N]` int32/float**。

**Vector (AIV) epilogue：标准 op（本卡强制，无 MicroAPI）**

对每个 tile，按语义顺序：

```text
v = Cast<float>(acc_int32)
v = v * x1Scale[t]          # per-token，按行；尽量整块广播，避免逐行标量 GetValue
v = v * x2Scale[n0:n1]      # per-channel，整列向量乘
v = v + bias[n0:n1]         # per-channel
```

- 优先**多行整块**处理 `[validM, N]` 而非逐行，减少 `PipeBarrier<PIPE_V>` 串行。1C2V：两个 AIV 各分 `baseM/2` 行。

**permY 输出（消除第二次转置）**

- `permY=[0,1,2]`：`DataCopyPad` 行连续写 `out[B,M,N]`（`dstStride=N`）。
- `permY=[0,2,1]`：用 `DataCopyPad` 的 **dstStride 按列步长直写** `out[B,N,M]`。**禁止** `y_tmp(GM)` 暂存 + 逐列 gather 的二次转置。

### 5.6 同步要求

```text
for each tile:
  AIC 产出 int32 acc tile 写 GM ring slot
  AIC PipeBarrier<PIPE_ALL>（排空 GM 写）后 CrossCoreSetFlag 通知 AIV
  AIV CrossCoreWaitFlag -> 消费 tile -> 反量化 + 写 out
  AIV 通知 AIC：slot 可复用
```

- 每个 `CrossCoreSetFlag` 前必须 `PipeBarrier<PIPE_ALL>` 保证 GM 写可见（铁律）。
- 多 AIV lane 的同步按 collective 语义，不能只让 lane0 推进全局进度。
- ring slot 复用必须有明确 C2V/V2C 生命周期。
- 不要用局部 `PipeBarrier` 代替跨 AIC/AIV 的可见性同步。

### 5.7 Tiling 与 workspace 设计

**Host tiling**：不要固定 `baseM=32 / baseN=64`，应根据 shape（含大 M 256-2048）和硬件资源选择 `baseM/baseN/baseK`、used AIC/AIV 核数、ring slot 数、N tile 数、M/N tail；优先复用框架与参考实现的 tiling 思想（蛇形、tail split），避免小 tile 造成核利用率过低。

**Workspace 允许**：AIC/AIV tile 交接的 int32 accumulator ring slot；ping-pong / queue style tile buffer；scale/bias 的 UB staging。

**Workspace 禁止作为主路径**：向量侧物化的 `x1n/x2n`（转置交框架，cube 直接吃源 GM）；完整 `contrib[B,M,N]` int32/float 中间结果；permY 的 `y_tmp` 全量副本 + 二次 gather。

### 5.8 验证与 Trace 记录要求

- **阶段三 correctness**：`bash scripts/evaluate_ascendc.sh current_task basic` / `... all`，记录 `basic` / `all` 结果与失败 case 的 shape/误差/mismatch，`22/22` 精度必须保持。
- **阶段四 performance**：按 benchmark 指定的性能门脚本，**duration-only 口径**，不得只看 profiler wait-time。重点：转置 x2 大 case（`permX2=[0,2,1]` 且 M=1024/1536/2048）；锚点 M=15（极小尾块）/257/1000（非 16 对齐，守 padding）；大 case 必须显著提速，仅 scaler/prefetch 微调而大 case 仍数百微秒不算有效优化。
- **`current_task/trace.md` 必须记录**：是否读取本节；是否用高阶 `MatmulImpl` GM 输出 + `aTrans/bTrans`（转置交框架）；是否删除 `NormalizeInput` + 两阶段 launch；tiling 是否自适应（非 `baseM=32`）、是否占满 cube 核；`permY=[0,2,1]` 是否 dstStride 直写；性能门结果与未达标 case；失败时定位结构 / 同步 / 精度 / tiling。

### 5.9 设计检查清单（实现前必须满足）

- [ ] cube 用高阶 `MatmulImpl`，permX1->aTrans / permX2->bTrans，转置**不在向量侧物化**？
- [ ] `cT = MatmulType<GM, ND, int32>` GM 输出（不依赖 L0C->UB）？
- [ ] 反量化用**标准 vector op**（不依赖 MicroAPI）？
- [ ] `NormalizeInput` + 两阶段 launch 已删除？
- [ ] tiling 自适应大 M、占满 cube 核（**非 `baseM=32`**）？
- [ ] epilogue 顺序 `((mm * x1Scale) * x2Scale) + bias`（与 golden 一致），未预合 scale/bias？
- [ ] `permY=[0,2,1]` 用 dstStride 直写，无 `y_tmp` 二次 gather？
- [ ] C->V GM ring 有明确 C2V/V2C 生命周期，两 AIV collective？
- [ ] 每个 `CrossCoreSetFlag` 前 `PipeBarrier<PIPE_ALL>`？
- [ ] 性能优化是**结构性改动**（换框架 + tiling + 直写），不是 scaler-only patch？

## 6. 精度要求

本算子精度判定遵循 [`benchmark/PRECISION_SPEC.md`](benchmark/PRECISION_SPEC.md)。通过条件与阈值参数定义在同目录 `proto.yaml` 的 `precision` 节点,以下仅说明本算子特定的取舍。

### 6.1 算子特定说明

- **`out` 阈值归属**:规则 `output_dtype`。输出为 float32 golden，重点验证 transpose 视图与 scale/bias 广播。
- **乘加顺序**:精度强约束见 §5.1，`((matmul * x1Scale) * x2Scale) + bias`（以 golden 为准），不得预合 scale/bias。

## 7. 标准 Golden 代码

`golden.py` 使用 `permute` 构造逻辑输入，再执行 batch matmul、scale/bias 和输出置换。

## 8. 额外信息
`cases.yaml` 当前包含 20 个正向 case，覆盖 batch、不同 `M`、三类 perm 组合和输出转置。

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


def transpose_quant_batch_mat_mul(
    x1: torch.Tensor,
    x2: torch.Tensor,
    x1Scale: torch.Tensor,
    x2Scale: torch.Tensor,
    bias: torch.Tensor,
    permX1=(0, 1, 2),
    permX2=(0, 1, 2),
    permY=(0, 1, 2),
    groupSize: int = 0,
    batchSplitFactor: int = 1,
    y_dtype: str = "float32",
) -> torch.Tensor:
    """Torch golden for transpose_quant_batch_mat_mul K-C path."""
    a = x1.permute(*permX1).to(torch.float32)
    b = x2.permute(*permX2).to(torch.float32)
    if a.dim() != 3 or b.dim() != 3:
        raise ValueError("This benchmark fixes 3D batched inputs")
    batch, m, k = a.shape
    batch2, k2, n = b.shape
    if batch != batch2 or k != k2:
        raise ValueError("shape mismatch after permute")
    y = torch.matmul(a, b)
    y = y * x1Scale.to(torch.float32).reshape(batch, m, 1) * x2Scale.to(torch.float32).reshape(batch, 1, n)
    y = y + bias.to(torch.float32).reshape(batch, 1, n)
    return y.permute(*permY)
```
