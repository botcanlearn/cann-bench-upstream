# QuantMatmulReduceSum (weight NZ) 算子 API 描述

## 1. 算子简介

**算子特征**：
- 难度等级：L3（Contraction）

`quant_matmul_reduce_sum` 完成一组 batch int8 矩阵乘，经过 scale 反量化后在 batch 维求和。本 benchmark 对齐源码目录 `ops-nn/matmul/quant_matmul_reduce_sum` 中的 `aclnnQuantMatmulReduceSumWeightNz` 语义：`x1` 为 ND int8 激活，`x2` 为 NZ int8 权重，AIC 完成分块 matmul，AIV 做 `x2Scale/x1Scale` 反量化并通过 batch 维累加写回。

该路径是 **C->V 算子**：cube 侧输出 int32 matmul 中间结果，vector 侧完成反量化与 reduce-sum，最终 cast 到 bfloat16。

## 2. 算子定义

设 `x1` 的形状为 `[B, M, K]`，`x2` 的逻辑原始形状为 `[B, K, N]`，物理 NZ 形状为 `[B, ceil(N/32), ceil(K/16), 16, 32]`。

$$
out = \sum_{b=0}^{B-1} (x1_b @ x2_b) \odot x1Scale_b[:, None] \odot x2Scale[None, :]
$$

其中 `x1Scale` 的形状为 `[B, M]`，`x2Scale` 的形状为 `[N]`，输出 `out` 的形状为 `[M, N]`。`x2Scale.numel() == N` 是 N 的唯一来源（golden 用它确定 N，并据此校验 `x2` 的 `N1 == ceil(N/32)`）。

## 3. 接口规范

benchmark 抽象接口：

```python
quant_matmul_reduce_sum(
    x1, x2, x1Scale, x2Scale,
    dims=[0], keep_dims=False, x2_format="NZ"
) -> out
```

参数说明：

| 参数 | 输入/输出 | dtype | shape | 说明 |
|------|-----------|-------|-------|------|
| `x1` | 输入 | `INT8` | `[B, M, K]` | batch 量化激活，ND 布局 |
| `x2` | 输入 | `INT8` | `[B, ceil(N/32), ceil(K/16), 16, 32]` | NZ 权重，逻辑 shape 为 `[B, K, N]`，`k0=16, n0=32` |
| `x1Scale` | 输入 | `FLOAT32` | `[B, M]` | per-token 激活 scale |
| `x2Scale` | 输入 | `BFLOAT16` | `[N]` | per-channel 权重 scale（决定 N） |
| `dims` | 输入 | `int[]` | `[0]` | 本 benchmark 固定 `[0]`，仅 batch 维求和 |
| `keep_dims` | 输入 | `bool` | 标量 | 本 benchmark 固定 `False` |
| `x2_format` | 输入 | `str` | `"NZ"` | 本 benchmark 固定 `NZ` |
| `out` | 输出 | `BFLOAT16` | `[M, N]` | batch 维归约后的反量化 matmul |

## 4. 约束说明

- `dims` 固定为 `[0]`，只覆盖 batch 维求和；`tuple(dims) != (0,)` 时 golden 直接抛错。
- `keep_dims` 固定为 `False`；为 `True` 时 golden 直接抛错。
- `x2_format` 固定为 `NZ`，不覆盖 ND 权重路径；非 `NZ` 时 golden 直接抛错。
- `x1` 必须是 3D `[B, M, K]`；`x2` 必须是 5D `[B, N1, K1, 16, 32]`，且 `k0 == 16`、`n0 == 32`。
- `x2` NZ 物理形状必须满足 `k1 == ceil(K / 16)`、`n1 == ceil(N / 32)`，尾部 padding 只在还原逻辑 ND 时裁掉。**写 case 时 K1/N1 算错会让 golden 直接 raise（validator 会捕获 `x2 K1 must equal ceil(K/16)` / `x2 N1 must equal ceil(N/32)`）。**
- `x1Scale.shape` 必须等于 `[B, M]`；`x2` 的 batch 维必须与 `x1` 一致。
- 预留参数 `yScale/x1Offset/x2Offset/yOffset/bias` 不在本 benchmark 范围内。

固定参数：`dims = [0]`、`keep_dims = False`、`x2_format = "NZ"`。

## 5. 实现约束与参考设计

> 本节分两层。**§5.1 硬约束**是正确性与合法性底线，必须遵守；**§5.2 参考设计与已知反模式**是非强制指导——给出一条已验证可行的实现路径与避坑提示，**鼓励在守住 §5.1 的前提下自行设计更优方案；当生成算子性能不足时，应主动超越本参考路径，而非照抄。**

### 5.1 硬约束（正确性 + 反作弊，必须遵守）

#### 目标语义与乘加顺序

```text
# 1. 把 NZ 权重还原为逻辑 ND [B, K, N]（仅用于语义对照，kernel 不得在 host 还原）
x2_nd = x2.permute(0, 2, 3, 1, 4).contiguous().reshape(B, K1 * 16, N1 * 32)
x2_nd = x2_nd[:, :K, :N]

# 2. int8 matmul，以 float32 中间累加
mm = matmul(x1.float(), x2_nd.float())          # [B, M, N], int32 -> float32

# 3. 反量化
mm = mm * x1Scale[:, :, None]                   # per-token（per-M 行）
mm = mm * x2Scale[None, None, :]                # per-channel（per-N 列，bf16 -> fp32）

# 4. batch 维归约并 cast 到 bfloat16
out = mm.sum(dim=0).to(bfloat16)                # [M, N]
```

- **乘加顺序须保持 `((matmul * x1Scale) * x2Scale)` 再 `sum_b`**（与 §2 / golden 一致）：先 per-token、再 per-channel，最后 batch 维 reduce-sum。不得把 `x1Scale * x2Scale` 预合成后再乘，会改变浮点舍入顺序，与 golden 产生精度残差。
- int8 matmul 以 int32 累加后 cast 到 float32 做反量化中间精度；batch 维归约前保持 fp32 语义，最终 cast 到 bfloat16 写回 `out`。
- **形状/语义断言**（与 §2 / §4 / golden 一致，不得违反）：`x1` 为 3D `[B, M, K]`；`x2` 为 5D NZ `[B, ceil(N/32), ceil(K/16), 16, 32]`（`k0=16`、`n0=32`）；`x2Scale` 长度等于 N（N 的唯一来源）；`x1Scale.shape == [B, M]`；batch 维一致；输出 `out` 形状为 batch reduce 后的 `[M, N]`。

#### 反作弊（真 C+V 融合，禁退化）

1. 必须生成**真正的 Cube + Vector 融合 AscendC kernel**，**禁止退化为**：纯 AIV / 纯 CPU / torch / aclnn 高层组合算子 / Python fallback。
2. batch 内 int8 矩阵乘 `x1[b] @ x2[b]` 必须由 **AIC/Cube 侧完成**，使用 AscendC Cube / MatMul / MMAD 原语**直接消费 NZ 权重**；**禁止在 AIV 侧用逐元素循环模拟矩阵乘**，也**禁止在 host 端把 NZ 还原成 ND 再拼 matmul**。
3. 反量化与 batch 维归约必须由 **AIV/Vector 侧完成**（片上向量化），包括：int32 中间结果 cast 到 float32、`x1Scale[b, :, None]` 广播乘（per-token）、`x2Scale[None, :]`（bfloat16 → float32）广播乘（per-channel）、batch 维 reduce-sum、最终 cast 到 bfloat16 写回 `out`。**禁止使用 torch / host / CPU / aclnn / Python 计算输出，禁止把 scale / 归约搬到 host。**
4. 自定义 kernel 的 `__global__` 核函数名与 Host `_do` 入口名必须包含 `custom`（如 `<op_name>_custom` / `<op_name>_custom_do`）；不得生成不含 `custom` 的 profiling kernel 名。
5. 热路径禁止标量逐元素 `GetValue/SetValue` 写法（少量边界 / 控制元数据除外），必须使用块级或向量化操作。

#### NZ 布局正确性与尾部 padding

- `x2` 物理布局为 `[B, N1, K1, 16, 32]`，其中 `K1 = ceil(K/16)`、`N1 = ceil(N/32)`、`k0 = 16`、`n0 = 32`；NZ 权重应**直接喂 Cube**消费（禁止在 host 端把 NZ 还原成 ND，见上条 2）。
- 必须正确处理尾部 K/N padding：逻辑维度只取 `[:K, :N]`，padding lane 不得污染反量化结果（尾部行/列在 V 段按有效 M/N 裁剪）。

#### 跨核同步正确性

- AIC→AIV 交接必须正确同步、保证跨核数据可见，不得出现数据竞争（否则结果错）；batch 维归约的累加策略必须保证正确性。

#### 精度

- 精度遵循 §6 / [`../PRECISION_SPEC.md`](../PRECISION_SPEC.md) 与 `proto.yaml` 的 `precision` 节点。

#### 验收

1. 所有 `cases.yaml` / `cases.csv` 中的用例**精度必须全部达标**；不得只通过 `basic_case` 或部分 `general_case` 后停止。正确性（全量精度达标）是**硬门**。
2. 所有用例的**计算速度必须优于 torch 小算子拼接基线**；若任一用例性能未达标，必须继续优化（见 §5.2 参考突破点），直到性能达标或在 `trace.md` 中记录明确阻塞原因。性能是**软门**：预算用尽仍不达标时接受当前正确 kernel，并在 `trace.md` 标注「性能未达标」。
3. 全量验证失败时调用 debugger 子流程继续修复，不得停下来问用户；TIMEOUT / 卡死 / 长时间无输出须先触发超时定位流程再改 kernel。

### 5.2 参考设计与已知反模式（指导，非强制，鼓励超越）

> 以下是一条**已验证可行**的 C→V 实现路径与若干提示，作为起点与避坑参考，**不是必须照抄的配方**。在守住 §5.1 的前提下，鼓励 agent 探索更优实现；**性能不足时优先在此处突破。**

- **参考数据流（C→V）**：AIC 将每个 batch 的 int32 matmul 结果写入 workspace 或等价中间缓冲，AIV 读取中间结果完成反量化与 batch 维累加并 float32→bfloat16 写回。这是一种直接可行的切分；agent 可自行探索更优方案（如把反量化融进 matmul epilogue、不同 tile/buffer 策略、1C2V 分工等）。
- **参考 AIC / AIV 分工**：AIC（Cube）执行 batch 内 int8 matmul（NZ 权重直接喂 Cube），输出 int32 中间结果到 workspace / GM ring slot；AIV（Vector）执行 `x2Scale` 反量化（bf16→fp32）、`x1Scale` 广播乘、batch 维 reduce-sum，最终 cast 到 bfloat16 写回 `out`。
- **参考同步 / workspace**：C→V 交接走 workspace / GM ring，每次 `CrossCoreSetFlag` 前用 `PipeBarrier`（如 `PipeBarrier<PIPE_ALL>`）排空 GM 写以保证可见性；多 AIV lane 按 collective 语义参与而非只让 lane0 推进全局进度；ring slot 维护明确 C2V/V2C 生命周期；batch 维归约可用 fp32 累加 + atomic add 或 workspace 累加。具体同步原语、累加策略与 buffer 布局由 agent 按性能选择，只要满足 §5.1 的同步正确性即可。
- **参考 tiling**：kernel tiling/launch 体现 AIC + AIV 混合执行；按 shape 自适应选 tile，避免过小 tile 导致 cube 利用率过低。
- **已知反模式（建议避开）**：epilogue 逐行用标量 `GetValue` 读 per-token scale、且每个向量 op 后串行 `PipeBarrier` 的写法，在大 shape 下被 fence 串行主导；优先**多行整块**向量处理。
- 建议在设计文档与 `trace.md` 记录：AIC/AIV 分工、workspace 布局（含 int32 中间矩阵与可能的 fp32 累加缓冲）、同步方式（含 batch 维归约的 atomic 或 barrier 策略），便于性能复盘。

## 6. 精度要求

本算子精度判定遵循 [`../PRECISION_SPEC.md`](../PRECISION_SPEC.md)。通过条件与阈值参数定义在同目录 `proto.yaml` 的 `precision` 节点，以下仅说明本算子特定的取舍。

### 6.1 算子特定说明

- **`out` 阈值归属**：规则 `output_dtype`，固定 BFLOAT16 → 阈值 2^-7。
- **大 batch 累加风险**：V 段 batch 维 reduce 若使用 fp32 中间再 cast 到 bf16，本阈值合理；若 NPU 实现采用 bf16 累加 + 大 B(≥16)，MARE 可能逼近阈值。**实测若超阈值**，需把 `proto.yaml.precision.outputs[out].threshold_rule` 改为 `intermediate_dtype_inherited` 并补 `intermediate_dtype: bfloat16`（阈值仍 2^-7 但语义更准），或考虑放宽到 2^-6。

## 7. 标准 Golden 代码

`golden.py` 先将 NZ 权重还原为逻辑 ND `[B, K, N]`，再执行 matmul、scale 和 batch 维求和：

```python
x2_nd = x2.permute(0, 2, 3, 1, 4).contiguous().reshape(B, K1 * 16, N1 * 32)
x2_nd = x2_nd[:, :K, :N]
mm = torch.matmul(x1.float(), x2_nd.float())
mm = mm * x1Scale.float().reshape(B, M, 1)
mm = mm * x2Scale.float().reshape(1, 1, N)
out = mm.sum(dim=0).to(torch.bfloat16)
```

## 8. 额外信息

### 测试资料对应关系

- `docs/aclnnQuantMatmulReduceSumWeightNz.md`：描述 `x1/x2/x1Scale/x2Scale`、NZ 形状和 batch 维求和公式。
- `op_kernel/quant_matmul_reduce_sum_mixcore.h`：vector 侧完成 `x2Scale` 反量化、`x1Scale` 广播乘法和 atomic 累加。
- `tests/ut/op_kernel/test_quant_matmul_reduce_sum.cpp`：包含 `B=8, M=32, K=64, N=32` 的 kernel 基础样例（对应本套件 case 3）。

### 本 benchmark case 设计

`cases.yaml` 当前包含 20 个正向 case，遵循「少量小 shape + 多数 LLM shape」标准，全部固定 `dims=[0]`、`keep_dims=false`、`x2_format=NZ`：

- **6 个小 shape（smoke/edge）**：tiny M、小 B、K/N 非 16/32 整除（exercise ceil padding）、tail M（M=17/33）、单 batch；改编自原始小用例与 kernel UT 样例。
- **14 个 LLM shape**：M 放大到 ≤5120、K ≤5120、N ≤5120、B 小（1–8）。覆盖方阵（512²/1024²/4096²/5120²）、大 M（4096/5120）、大 K（4096/5120）、大 N（5120）、非对齐 K（760）与非对齐 N（2300）以及 MoE/FFN 风格的 realistic 维度。
- 所有 case 的 `x2` 物理 shape 均按 `[B, ceil(N/32), ceil(K/16), 16, 32]` 写出，`x2Scale` 长度等于 N。

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


def quant_matmul_reduce_sum(
    x1: torch.Tensor,
    x2: torch.Tensor,
    x1Scale: torch.Tensor,
    x2Scale: torch.Tensor,
    dims=(0,),
    keep_dims: bool = False,
    x2_format: str = "NZ",
) -> torch.Tensor:
    """Torch golden for quant_matmul_reduce_sum with x2 in NZ layout."""
    if tuple(dims) != (0,):
        raise ValueError("This benchmark fixes dims=[0]")
    if keep_dims:
        raise ValueError("This benchmark fixes keep_dims=False")
    if str(x2_format).upper() != "NZ":
        raise ValueError("This benchmark fixes x2_format=NZ")
    if x1.dim() != 3:
        raise ValueError(f"x1 expects 3D [B,M,K], got {list(x1.shape)}")
    if x2.dim() != 5:
        raise ValueError(f"x2 expects 5D NZ [B,N1,K1,16,32], got {list(x2.shape)}")

    b, m, k = x1.shape
    n = x2Scale.numel()
    if x1Scale.shape != (b, m):
        raise ValueError(f"x1Scale expects shape [{b}, {m}], got {list(x1Scale.shape)}")

    x2_nd = _nz_weight_to_nd(x2, b, k, n)
    mm = torch.matmul(x1.to(torch.float32), x2_nd.to(torch.float32))
    mm = mm * x1Scale.to(torch.float32).reshape(b, m, 1)
    mm = mm * x2Scale.to(torch.float32).reshape(1, 1, n)
    out = mm.sum(dim=0)
    return out.to(torch.bfloat16)


def _nz_weight_to_nd(x2: torch.Tensor, batch: int, k: int, n: int) -> torch.Tensor:
    b, n1, k1, k0, n0 = x2.shape
    if b != batch:
        raise ValueError(f"x2 batch ({b}) must match x1 batch ({batch})")
    if k0 != 16 or n0 != 32:
        raise ValueError(f"NZ x2 expects k0=16,n0=32, got k0={k0}, n0={n0}")
    if k1 != (k + 15) // 16:
        raise ValueError(f"x2 K1 ({k1}) must equal ceil(K/16) for K={k}")
    if n1 != (n + 31) // 32:
        raise ValueError(f"x2 N1 ({n1}) must equal ceil(N/32) for N={n}")
    nd = x2.permute(0, 2, 3, 1, 4).contiguous().reshape(b, k1 * k0, n1 * n0)
    return nd[:, :k, :n]
```
