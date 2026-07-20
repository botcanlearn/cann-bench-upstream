# FusedQuantMatMul 算子 API 描述

## 1. 算子简介

**算子特征**：
- 难度等级：L3（FusedComposite）

`fused_quant_mat_mul` 对齐源码目录 `ops-nn/matmul/fused_quant_mat_mul`。本 benchmark 选取 QuantBatchMatmul + GELU 融合路径：C 侧完成 INT8 matmul（累加到 INT32），V 侧执行 scale 反量化、浮点 bias 和 GELU 非线性激活，属于 **C->V kernel flow**。固定 `transpose_x1=false`、`transpose_x2=false`、`y_dtype=float32`，仅在 `fused_op_type ∈ {gelu_erf, gelu_tanh}` 两个分支间切换。

## 2. 算子定义

```text
qbmm = (x1 @ x2)                                  # [M,K] int8 @ [K,N] int8 -> [M,N] int32 -> float32
qbmm = qbmm * x1Scale[:, None] * x2Scale[None, :] # per-token * per-channel 反量化
qbmm = qbmm + bias[None, :]                       # 浮点 bias，先反量化后加 bias
out  = gelu_erf(qbmm)  or  gelu_tanh(qbmm)        # [M,N] float32
```

其中两种 GELU 分支：

```text
gelu_erf(x)  = 0.5 * x * (1 + erf(x / sqrt(2)))
gelu_tanh(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
```

## 3. 接口规范

```python
fused_quant_mat_mul(x1, x2, x1Scale, x2Scale, bias, fused_op_type="gelu_erf", transpose_x1=false, transpose_x2=false, y_dtype="float32") -> out
```

| 参数 | 输入/输出 | dtype | shape | 说明 |
|------|-----------|-------|-------|------|
| `x1` | 输入 | `INT8` | `[M,K]` | 左矩阵 |
| `x2` | 输入 | `INT8` | `[K,N]` | 右矩阵 |
| `x1Scale` | 输入 | `FLOAT32` | `[M]` | per-token scale |
| `x2Scale` | 输入 | `FLOAT32` | `[N]` | per-channel scale |
| `bias` | 输入 | `FLOAT32` | `[N]` | 浮点 bias，沿 M 维广播 |
| `out` | 输出 | `FLOAT32` | `[M,N]` | GELU 后结果 |

固定 attrs：`fused_op_type ∈ {"gelu_erf", "gelu_tanh"}`（由用例 `attrs.fused_op_type` 决定）、`transpose_x1=false`、`transpose_x2=false`、`y_dtype=float32`。

## 4. 约束说明

### 4.1 形状与语义约束

- `x1` 的 K 维与 `x2` 的 K 维必须一致（`golden.py` 中 `K mismatch` 断言）。
- `x1Scale` 长度等于 M（per-token scale）。
- `x2Scale` 长度等于 N（per-channel scale）。
- `bias` 长度等于 N，沿 M 维广播。
- 固定覆盖 `gelu_erf` 与 `gelu_tanh` 两种 `fusedOpType`。
- 固定浮点 bias 路径，公式是 `matmul * scale + bias` 后再 GELU，**不混入** INT32 bias 的 `(matmul + bias) * scale` 语义。
- **乘加顺序必须保持** `((matmul * x1Scale) * x2Scale) + bias`（与 golden 一致）：不得把 `x1Scale * x2Scale` 或 `scale * bias` 预先合成后再乘/加，会改变浮点舍入顺序，与 golden 产生精度残差。
- `transpose_x1 = false`、`transpose_x2 = false`（固定）。

### 4.2 实现约束与参考设计

> 本节分两层。**§4.2.1 硬约束**是正确性与合法性底线，必须遵守；**§4.2.2 参考设计与已知反模式**是非强制指导——给出一条已验证可行的实现路径与避坑提示，**鼓励在守住 §4.2.1 的前提下自行设计更优方案；当生成算子性能不足时，应主动超越本参考路径，而非照抄。**

#### 4.2.1 硬约束（正确性 + 反作弊，必须遵守）

- **真融合，禁退化**：必须生成真正的 Cube + Vector 融合 AscendC kernel；禁止退化为纯 AIV（在 Vector 侧用逐元素循环模拟 INT8 矩阵乘）、纯 CPU、torch 计算、aclnn 高层组合算子、Python fallback。
- **matmul 必须落 Cube**：INT8 矩阵乘 `x1 @ x2`（`M×K @ K×N → M×N`，累加到 INT32）必须由 AIC/Cube 用 AscendC Cube / MatMul / MMAD 原语完成；**禁止在 AIV 侧用逐元素循环模拟矩阵乘**。
- **量化后处理必须片上向量化**：INT32→FP32、per-token（`x1Scale`）与 per-channel（`x2Scale`）反量化、浮点 bias、GELU（`gelu_erf` / `gelu_tanh` 两分支）必须在片上 AIV/Vector 完成；**禁止**下沉到 torch / host 计算输出，**禁止**把 scale / bias / GELU 搬到 host。
- **乘加顺序与语义**：`((matmul * x1Scale) * x2Scale) + bias` 后再过 GELU，与 §2 / golden 一致（详见 §4.1）；不得预合 scale/bias。
- **跨核同步正确性**：AIC→AIV 交接必须正确同步、保证跨核数据可见，不得出现数据竞争（否则结果错）；不得用局部 `PipeBarrier` 冒充跨核可见性同步。
- kernel `__global__` 与 Host `_do` 入口名必须含 `custom`；AscendC 热路径禁止标量 `GetValue/SetValue` 循环（少量边界 / 控制元数据除外）。
- 精度遵循 §5 / [`../PRECISION_SPEC.md`](../PRECISION_SPEC.md) 与 `proto.yaml` 的 `precision` 节点。

#### 4.2.2 参考设计与已知反模式（指导，非强制，鼓励超越）

> 以下是一条**已验证可行**的 C→V 实现路径与若干提示，作为起点与避坑参考，**不是必须照抄的配方**。在守住 §4.2.1 的前提下，鼓励 agent 探索更优实现；**性能不足时优先在此处突破。**

- **参考数据流（C→V）**：AIC 把 INT8 matmul 的 INT32 tile 写入 workspace / GM ring slot，AIV 读取后完成反量化 + bias + GELU 并 float32 写回。这是一种直接可行的切分；agent 可自行探索更优方案（如把反量化融进 matmul epilogue、不同 tile/buffer 策略、1C2V 分工等）。
- **参考同步 / workspace**：AIC 每次 `CrossCoreSetFlag` 前用 `PipeBarrier<PIPE_ALL>` 排空 GM 写；ring slot 维护明确 C2V/V2C 生命周期。具体同步原语与 buffer 布局由 agent 按性能选择，只要满足 §4.2.1 的同步正确性即可。
- **参考 tiling**：AIC + AIV 混合执行（1C2V），按 shape 自适应选 tile；避免过小 tile 导致 cube 利用率过低。
- **已知反模式（建议避开）**：epilogue 逐行用标量 `GetValue` 读 per-token scale、且每个向量 op 后 `PipeBarrier<PIPE_V>` 的串行写法，在大 shape 下被 fence 串行主导；优先**多行整块** `[validM, N]` 向量处理。
- 建议在设计文档与 `trace.md` 记录：AIC/AIV 分工、workspace 布局、同步方式、`fused_op_type` 分支路径，便于性能复盘。

### 4.3 用例覆盖范围

- 本 benchmark `cases.yaml` 共 **20 个正向 case**：6 个 small（smoke / edge）+ 14 个 LLM 大 GEMM。
- small case 覆盖：tiny M（M=1）、非 16 对齐的 M/N/K 尾块、`gelu_erf` 与 `gelu_tanh` 两个分支。
- LLM case 覆盖：真实 LLM GEMM 形状，`M/K/N` 取自 `{1024, 1536, 2048, 2560, 3072, 4096, 5120}`，**不超过 5120**；含非 16 对齐大 M（1000 / 2049），两个 GELU 分支均覆盖。
- `value_range = [-8, 8]`（int8 输入与 scale 的取值范围）。

## 5. 精度要求

本算子精度判定遵循 [`../PRECISION_SPEC.md`](../PRECISION_SPEC.md)。通过条件与阈值参数定义在同目录 `proto.yaml` 的 `precision` 节点，以下仅说明本算子特定的取舍。

### 5.1 算子特定说明

- **`out` 阈值归属**：规则 `intermediate_dtype_inherited`，`intermediate_dtype: float16`。虽然 `out` 自身 dtype 为 FLOAT32 golden，但 V 段 GELU 是**非线性**：NPU V 段实际计算 GELU 用 FP16 中间精度（标准 vector unit 路径），GELU 在 `[-2, 2]` 范围内的近似误差非平凡，直接按 FLOAT32 阈值 `2^-13` 卡会假阴性。改按 FP16 阈值 `2^-10` 与实际精度上限一致。
- **若实测 MARE 远小于 `2^-10`**：说明 NPU V 段可能用 FP32 中间精度；可把 `intermediate_dtype` 改回 `float32`（阈值收紧到 `2^-13`）。

## 6. 标准 Golden 代码

`golden.py` 实现 `gelu_erf` 与 `gelu_tanh`，输入 scale 先广播到 `[M,N]`，乘加顺序为 `((x1@x2) * x1Scale * x2Scale) + bias`，再过 GELU。**禁止修改 `golden.py` 中定义的算子语义，禁止在实现中用 torch 算子做实际计算。**

## 7. 强制验收约束

1. 所有 `cases.yaml` / `cases.csv` 中的用例**精度必须全部达标**；不得只通过 `basic_case` 或部分 `general_case` 后停止。正确性（全量精度达标）是**硬门**。
2. 所有用例的**计算速度必须优于 torch 小算子拼接实现**（per-case 加速比软门）；如任一用例性能未优于该基线，必须继续优化 AscendC tiling、workspace 流水或 vector 量化路径，直到性能达标或在 `trace.md` 中记录明确阻塞原因。性能是**软门**：预算用尽仍不达标时接受当前正确 kernel，并在 trace 标注「性能未达标」。

## 8. 额外信息

### 测试资料对应关系

- `docs/aclnnFusedQuantMatmul.md`：scale、bias 与 `fusedOpType` 公式。
- `op_kernel/fused_quant_mat_mul_swiglu.h` 和 kernel 入口：融合后处理路径参考。

### 本 benchmark case 设计

`cases.yaml` 当前包含 20 个正向 case：6 个 small（smoke/edge，含 tiny M、非 16 对齐尾块、两 GELU 分支）+ 14 个 LLM 大 GEMM（`M/K/N ≤ 5120`，覆盖 MLP up/down、QKV/attn proj 等 LLM 形状，两 GELU 分支），覆盖两种 GELU、不同 `M/K/N`、小 N 与中等宽输出，以及对齐与非对齐边界。

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


def _gelu_erf(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * x * (1.0 + torch.erf(x / 1.4142135623730951))


def _gelu_tanh(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * x * (1.0 + torch.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)))


def fused_quant_mat_mul(
    x1: torch.Tensor,
    x2: torch.Tensor,
    x1Scale: torch.Tensor,
    x2Scale: torch.Tensor,
    bias: torch.Tensor,
    fused_op_type: str = "gelu_erf",
    transpose_x1: bool = False,
    transpose_x2: bool = False,
    y_dtype: str = "float32",
) -> torch.Tensor:
    """Torch golden for fused_quant_mat_mul GELU path."""
    if transpose_x1:
        x1 = x1.transpose(-2, -1)
    if transpose_x2:
        x2 = x2.transpose(-2, -1)
    m, k = x1.shape
    k2, n = x2.shape
    if k != k2:
        raise ValueError("K mismatch")
    qbmm = (x1.to(torch.float32) @ x2.to(torch.float32))
    qbmm = qbmm * x1Scale.to(torch.float32).reshape(m, 1) * x2Scale.to(torch.float32).reshape(1, n)
    qbmm = qbmm + bias.to(torch.float32).reshape(1, n)
    if fused_op_type == "gelu_erf":
        return _gelu_erf(qbmm)
    if fused_op_type == "gelu_tanh":
        return _gelu_tanh(qbmm)
    raise ValueError("fused_op_type must be gelu_erf or gelu_tanh")
```
