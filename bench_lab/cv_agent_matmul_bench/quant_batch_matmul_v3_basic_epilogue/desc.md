# QuantBatchMatmulV3 (basic epilogue) 算子 API 描述

## 1. 算子简介

**算子特征**：
- 难度等级：L3（Contraction）

`quant_batch_matmul_v3` 完成量化矩阵乘后的反量化与浮点输出。本 benchmark 对齐源码目录 `ops-nn/matmul/quant_batch_matmul_v3`，选取 `quant_batch_matmul_v3_bf16_basic.h` 与 `quant_batch_matmul_v3_pertoken_basic.h` 两条基础 **C->V** kernel 路径：AIC 完成 int8 矩阵乘写 workspace，AIV 等待 C2V 同步后执行 per-channel scale、可选 per-token scale、浮点 bias 和写回。

本 benchmark 只覆盖**浮点 bias 路径**，不覆盖 INT32 bias 路径。原因是两者数学顺序不同：浮点 bias 为 `matmul * scale + bias`，INT32 bias 为 `(matmul + bias) * scale`，二者**不可互换**。

`quant_batch_matmul_v3_basic_epilogue` 是本 benchmark 对应的 CV (cube + vector) 融合算子生成需求：从 `benchmark/quant_batch_matmul_v3_basic_epilogue` 的参考资料（`desc.md` / `proto.yaml` / `golden.py` / `cases.yaml` / `cases.csv`）出发，端到端产出真正的 Cube+Vector 融合 AscendC kernel。本文档为该需求的自洽说明，已折合原始 `desc.md` 与生成任务 `prompt.md` 的全部算子语义、实现约束与验收约束。

## 2. 算子定义

设 `x1` 的形状为 `[M, K]`，`x2` 的形状为 `[K, N]`。

基础浮点 bias 路径（`bf16_basic`，4 输入）：

$$
Y = (x1_{\text{int8}} @ x2_{\text{int8}}) \odot scale + bias
$$

per-token 路径（`pertoken_basic`，5 输入）：

$$
Y = (x1_{\text{int8}} @ x2_{\text{int8}}) \odot scale \odot perTokenScale[:, None] + bias
$$

其中 `scale` 与 `bias` 均按 N 维广播。`perTokenScale` 仅在 `pertoken_basic` 路径使用，按 M 维广播。

```text
out = (x1.int8 @ x2.int8) -> int32 -> float32   # [M, N]
out = out * scale[None, :]                       # per-channel dequant
if variant == "pertoken_basic":
    out = out * perTokenScale[:, None]           # per-token dequant
out = out + bias[None, :]                         # floating-point bias
y   = cast(out, y_dtype)                          # BF16 / FP16 / FP32
```

**乘加顺序必须保持为** `matmul * scale (* perTokenScale) + bias`（与 golden 一致）：不得把 `scale * perTokenScale` 或 `scale * bias` 预先合成后再乘/加，会改变浮点舍入顺序并与 golden 产生精度残差。

## 3. 接口规范

benchmark 抽象接口：

```python
quant_batch_matmul_v3(
    x1, x2, scale, bias=None, perTokenScale=None,
    variant="bf16_basic", y_dtype="bfloat16"
) -> y
```

参数说明：

| 参数 | 输入/输出 | dtype | shape | 说明 |
|------|-----------|-------|-------|------|
| `x1` | 输入 | `INT8` | `[M, K]` | 量化激活矩阵 |
| `x2` | 输入 | `INT8` | `[K, N]` | 量化权重矩阵 |
| `scale` | 输入 | `BFLOAT16`、`FLOAT32` | `[N]` | per-channel 反量化 scale，按 N 维广播 |
| `bias` | 输入 | `BFLOAT16`、`FLOAT32` | `[N]` | 浮点 bias，按 N 维广播；本 benchmark 固定使用浮点 bias 路径 |
| `perTokenScale` | 输入（可选） | `FLOAT32` | `[M]` | per-token scale，仅 `pertoken_basic` 使用，按 M 维广播 |
| `y` | 输出 | `BFLOAT16`、`FLOAT16`、`FLOAT32` | `[M, N]` | 反量化后的矩阵乘输出 |

固定参数（attrs）：

| 名称 | dtype | 取值 | 默认 | 说明 |
|------|-------|------|------|------|
| `variant` | str | `bf16_basic` / `pertoken_basic` | `bf16_basic` | 选择 kernel 分支 |
| `y_dtype` | str | `bfloat16` / `float16` / `float32` | `bfloat16` | golden 输出 dtype |

schema：`quant_batch_matmul_v3(Tensor x1, Tensor x2, Tensor scale, Tensor bias, Tensor? perTokenScale, str variant="bf16_basic", str y_dtype="bfloat16") -> Tensor y`。

### 输入个数与可选输入

- `bf16_basic` 输入为 `x1 / x2 / scale / bias` **四个**张量（不传 `perTokenScale`）。
- `pertoken_basic` 输入为 `x1 / x2 / scale / bias / perTokenScale` **五个**张量。
- `proto.yaml` 声明的是**最大输入集（5 个）**，其中 `perTokenScale` 标记为 `optional: true`；某个 case 省略尾部可选输入（即 4 输入的 `bf16_basic`）是合法的，校验器允许（其 `opt_input` 备注属预期行为）。
- **注意**：在 golden.py 中可选的尾部输入是 `perTokenScale`，**`bias` 始终必传**（本 benchmark 固定浮点 bias 路径，`bias=None` 直接报错）。两条分支在用例集中都必须保留，不得收敛为单一分支。

## 4. 约束说明

### 4.1 语义/形状约束

- 本 benchmark 固定 **2D ND 输入**，不覆盖 batch、转置、int4、fp8、per-tensor scale、offset 和融合激活路径。
- `x1.shape[1] == x2.shape[0] == K`。
- `scale.shape == bias.shape == [N]`。
- `pertoken_basic` 必须满足 `perTokenScale.shape == [M]`；`bf16_basic` **不得**传入 `perTokenScale`。
- `bias` 不可为 `None`（本 benchmark 固定浮点 bias 路径，不覆盖 INT32 bias）。
- `variant ∈ {"bf16_basic", "pertoken_basic"}`，默认 `"bf16_basic"`。
- `y_dtype ∈ {"bfloat16", "float16", "float32"}`，默认 `"bfloat16"`。

### 4.2 实现约束与参考设计

> 本节分两层。**§4.2.1 硬约束**是正确性与合法性底线，必须遵守；**§4.2.2 参考设计与已知反模式**是非强制指导——给出一条已验证可行的实现路径与避坑提示，**鼓励在守住 §4.2.1 的前提下自行设计更优方案；当生成算子性能不足时，应主动超越本参考路径，而非照抄。**

#### 4.2.1 硬约束（正确性 + 反作弊，必须遵守）

- **真融合，禁退化**：必须生成**真正的 Cube + Vector 融合 AscendC kernel**；**禁止退化为**纯 AIV、纯 CPU、torch（`model_new_tilelang.py` 与 `model_new_ascendc.py` 中禁止使用 torch 算子做任何实际计算）、aclnn 高层组合算子、Python fallback。
- **matmul 必须落 Cube**：int8 矩阵乘 `x1 @ x2 → int32`（`M×K @ K×N → M×N`，累加到 int32）必须由 **AIC/Cube 侧完成**，使用 AscendC Cube / MatMul / MMAD 原语；**禁止在 AIV 侧用逐元素循环模拟 int8 矩阵乘**，亦**不得**将 int8 提前 cast 到 fp16/fp32 后用 vector matmul 代替。
- **量化后处理必须片上向量化**：从中间缓冲读取 int32 matmul 结果并 cast 到 float32、per-channel scale 反量化（`scale` 为 bf16 时先 cast 到 fp32）、`pertoken_basic` 分支按 M 维广播乘以 `perTokenScale`、浮点 bias 叠加（`bias` 为 bf16 时先 cast 到 fp32）、cast 到目标 `y_dtype`（BF16 / FP16 / FP32）并写回 GM，**必须在片上 AIV/Vector 完成**；**禁止使用 torch 或 host 端计算输出**，**禁止**把 scale / bias 搬到 host。
- **两分支语义必须保留**：必须正确处理 `bf16_basic`（4 输入，无 `perTokenScale`）与 `pertoken_basic`（5 输入，含 `perTokenScale`）两条分支，且正确支持 `scale` / `bias` 的 `bfloat16` 与 `float32` 两种 dtype 与 `y` 的 BF16 / FP16 / FP32 三种输出 dtype；两条分支不得收敛为单一分支。
- **乘加顺序与语义**：`matmul * scale (* perTokenScale) + bias`，与 §2 / golden 一致（详见 §4.1 与 §2）；不得预合 `scale * perTokenScale` 或 `scale * bias`，亦不得混入 INT32 bias 的 `(matmul + bias) * scale` 语义。
- **跨核同步正确性**：AIC→AIV 交接必须正确同步、保证跨核数据可见，不得出现数据竞争（否则结果错）；不得用局部 `PipeBarrier` 冒充跨核可见性同步。
- kernel `__global__` 核函数名与 Host `_do` 入口名必须包含 `custom`（如 `<op_name>_custom` / `<op_name>_custom_do`）；AscendC 热路径禁止标量循环（少量边界 / 控制元数据除外）。
- 精度遵循 §5 / [`../PRECISION_SPEC.md`](../PRECISION_SPEC.md) 与 `proto.yaml` 的 `precision` 节点。

#### 4.2.2 参考设计与已知反模式（指导，非强制，鼓励超越）

> 以下是一条**已验证可行**的 C→V 实现路径与若干提示，作为起点与避坑参考，**不是必须照抄的配方**。在守住 §4.2.1 的前提下，鼓励 agent 探索更优实现；**性能不足时优先在此处突破。**

- **参考数据流（C→V）**：AIC（Cube）执行 `int8 × int8 → int32` 矩阵乘，将分块 int32 结果写入 workspace（或等价中间缓冲）；AIV（Vector）等待 C2V 同步后读取中间结果，完成 per-channel scale 反量化、可选 per-token scale 反量化、浮点 bias 叠加，以及向 `y_dtype` 的 cast 与写回 GM。这是一种直接可行的切分；agent 可自行探索更优方案（如把反量化融进 matmul epilogue、不同 tile/buffer 策略等）。
- **参考同步 / workspace**：AIC/AIV 同步可用 C2V flag、cross-core barrier 等，每个 `CrossCoreSetFlag` 前用 `PipeBarrier` 排空 GM 写以保证可见性；workspace 布局（int32 中间结果的 shape、stride、对齐方式）明确维护。具体同步原语与 buffer 布局由 agent 按性能选择，只要满足 §4.2.1 的同步正确性即可。
- **参考 tiling**：kernel tiling / launch 体现 **AIC + AIV 混合执行**，按 shape 自适应选 tile；避免过小 tile 导致 cube 利用率过低。
- **参考分支复用**：`bf16_basic` 与 `pertoken_basic` 两条分支在 kernel 内可复用主体路径、仅在 per-token 乘法处分支；具体复用与锁定路径由 agent 自行权衡。
- **已知反模式（建议避开）**：epilogue 逐行用标量读 per-token scale、且每个向量 op 后 `PipeBarrier<PIPE_V>` 的串行写法，在大 shape 下被 fence 串行主导；优先**多行整块**向量处理。
- 建议在设计文档与 `trace.md` 记录：AIC/AIV 分工与分核策略、workspace 布局、C2V 同步方式、`bf16_basic` 与 `pertoken_basic` 两条分支的 kernel 内复用与分支处理，便于性能复盘。

### 4.3 强制验收约束

1. 所有 `cases.yaml` / `cases.csv` 中的用例**精度必须全部达标**；不得只通过 `basic_case` 或部分 `general_case` 后停止。
2. 所有用例的**计算速度必须优于 torch 小算子拼接实现**；如任一用例性能未优于该基线，必须继续优化 AscendC tiling、workspace 流水或 vector 反量化路径，直到性能达标，或在 `trace.md` 中记录明确阻塞原因。
3. 性能门为软门、正确性为硬门：性能耗时统计须来自 msprof（duration-only 口径），不得使用 host wall-clock 自写计时。

## 5. 精度要求

本算子精度判定遵循 [`../PRECISION_SPEC.md`](../PRECISION_SPEC.md)。通过条件与阈值参数定义在同目录 `proto.yaml` 的 `precision` 节点，以下仅说明本算子特定的取舍。

### 5.1 算子特定说明

- **`y` 阈值归属**：规则 `output_dtype`。int8 matmul 输出 int32 → 乘 fp32 scale（+ 可选 perTokenScale）+ fp32 bias → cast 到目标 `y_dtype`；最终精度上限即输出自身 dtype，无中间精度损失隐患。
- **多 dtype 输出**：`y_dtype` 可变（BF16 / FP16 / FP32），评测脚本按当前 case 实际输出 dtype 查 SPEC §3 阈值表（BF16→2^-7、FP16→2^-10、FP32→2^-13）。
- **乘加顺序**：精度强约束见 §2，`matmul * scale (* perTokenScale) + bias`（以 golden 为准），不得预合 scale / perTokenScale / bias。
- **不覆盖 INT32 bias 路径**：INT32 bias 公式为 `(matmul + bias) * scale`，与本 benchmark 浮点 bias 路径 `matmul * scale + bias` 数学顺序不同；若后续扩展需独立 case 与独立精度规约。

## 6. 标准 Golden 代码

`golden.py` 使用 PyTorch 描述本 benchmark 的 selected C->V 路径，完整实现见同目录 `golden.py`。核心逻辑如下：

```python
out = torch.matmul(x1.to(torch.float32), x2.to(torch.float32))
out = out * scale.to(torch.float32).reshape(1, -1)
if variant == "pertoken_basic":
    out = out * perTokenScale.to(torch.float32).reshape(-1, 1)
out = out + bias.to(torch.float32).reshape(1, -1)
# 末尾按 y_dtype cast 到 BF16 / FP16 / FP32
```

golden 强制的形状校验（实现与用例必须遵守）：`x1/x2` 必须 2D；`x1.K == x2.K`；`scale.numel() == N`；`bias` 必传且 `bias.numel() == N`；`bf16_basic` 不得带 `perTokenScale`；`pertoken_basic` 必须带 `perTokenScale` 且 `perTokenScale.numel() == M`。

## 7. 额外信息

### 7.1 测试资料对应关系

- `docs/aclnnQuantMatmulV3.md`：说明浮点 bias 路径公式为 `out = x1@x2 * scale + bias`。
- `op_kernel/quant_batch_matmul_v3_bf16_basic.h`：基础 C->V 反量化路径。
- `op_kernel/quant_batch_matmul_v3_pertoken_basic.h`：带 per-token scale 的 C->V 反量化路径。

### 7.2 本 benchmark case 设计

`cases.yaml` / `cases.csv` 当前包含 **20 个正向 case**，与 1:1 对应，采用「少量小用例 + 大量 LLM 用例」结构：

- **小用例（6 个，case 1–6，smoke/edge）**：tiny M、单 token（M=1）、tail/非对齐维度（M=7/17、N=32/96/160），覆盖 `bf16_basic`（4 输入）与 `pertoken_basic`（5 输入）两条分支、BF16/FP16/FP32 三种输出、`BFLOAT16`/`FLOAT32` 及混合 scale/bias dtype。最大维度 < 512。
- **LLM 用例（14 个，case 7–20）**：真实大 shape，M/K/N 取自 `{1024, 1536, 2048, 2560, 3072, 4096, 5120}`，**上限 5120 且绝不超过**；覆盖典型 LLM 投影（qkv proj、MLP up/down）、方阵、非方阵、混合 scale/bias dtype。其中 case 7–12 为 `bf16_basic`（4 输入），case 13–20 为 `pertoken_basic`（5 输入）。

两条分支（4 输入 `bf16_basic` 与 5 输入 `pertoken_basic`）在小用例和 LLM 用例两层均有保留。所有 case 的 `value_range` 固定 `[-2, 2]`，`baseline_perf_us = 0.0`、`t_hw_us = 0.0`（性能字段留待真机回填）。

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


def quant_batch_matmul_v3(
    x1: torch.Tensor,
    x2: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor,
    perTokenScale: torch.Tensor = None,
    variant: str = "bf16_basic",
    y_dtype: str = "bfloat16",
) -> torch.Tensor:
    """Torch golden for selected quant_batch_matmul_v3 C->V paths."""
    if variant not in ("bf16_basic", "pertoken_basic"):
        raise ValueError(f"Unsupported quant_batch_matmul_v3 variant: {variant}")
    if x1.dim() != 2 or x2.dim() != 2:
        raise ValueError(f"quant_batch_matmul_v3 expects 2D x1/x2, got {list(x1.shape)} and {list(x2.shape)}")
    m, k = x1.shape
    k2, n = x2.shape
    if k != k2:
        raise ValueError(f"x1 K ({k}) must match x2 K ({k2})")
    if scale.numel() != n:
        raise ValueError(f"scale length ({scale.numel()}) must match N ({n})")
    if bias is None:
        raise ValueError("This benchmark fixes the floating-point bias path and requires bias")
    if bias.numel() != n:
        raise ValueError(f"bias length ({bias.numel()}) must match N ({n})")
    if variant == "bf16_basic" and perTokenScale is not None:
        raise ValueError("bf16_basic does not use perTokenScale in this benchmark")
    if variant == "pertoken_basic":
        if perTokenScale is None:
            raise ValueError("pertoken_basic requires perTokenScale")
        if perTokenScale.numel() != m:
            raise ValueError(f"perTokenScale length ({perTokenScale.numel()}) must match M ({m})")

    out = torch.matmul(x1.to(torch.float32), x2.to(torch.float32))
    out = out * scale.to(torch.float32).reshape(1, n)
    if variant == "pertoken_basic":
        out = out * perTokenScale.to(torch.float32).reshape(m, 1)
    out = out + bias.to(torch.float32).reshape(1, n)
    return _cast_output(out, y_dtype)


def _cast_output(out: torch.Tensor, y_dtype: str) -> torch.Tensor:
    name = str(y_dtype).split(".")[-1].lower()
    if name in ("bf16", "bfloat16"):
        return out.to(torch.bfloat16)
    if name in ("fp16", "float16", "half"):
        return out.to(torch.float16)
    if name in ("fp32", "float32", "float"):
        return out.to(torch.float32)
    raise ValueError(f"Unsupported y_dtype: {y_dtype}")
```
