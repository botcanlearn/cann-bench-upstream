# QuantBatchMatmulV3 算子 API 描述

## 1. 算子简介

`quant_batch_matmul_v3` 完成量化矩阵乘后的反量化与浮点输出。本 benchmark 对齐源码目录 `ops-nn/matmul/quant_batch_matmul_v3`，选取 `quant_batch_matmul_v3_bf16_basic.h` 与 `quant_batch_matmul_v3_pertoken_basic.h` 两条基础 C->V kernel 路径：AIC 完成 int8 矩阵乘写 workspace，AIV 等待 C2V 同步后执行 scale、可选 per-token scale、浮点 bias 和写回。

本 benchmark 只覆盖浮点 bias 路径，不覆盖 INT32 bias 路径。原因是两者数学顺序不同：浮点 bias 为 `matmul * scale + bias`，INT32 bias 为 `(matmul + bias) * scale`。

## 2. 算子定义

设 `x1` 的形状为 `[M, K]`，`x2` 的形状为 `[K, N]`。

基础浮点 bias 路径：

$$
Y = (x1_{\text{int8}} @ x2_{\text{int8}}) \odot scale + bias
$$

per-token 路径：

$$
Y = (x1_{\text{int8}} @ x2_{\text{int8}}) \odot scale \odot perTokenScale[:, None] + bias
$$

其中 `scale` 与 `bias` 均按 N 维广播。`perTokenScale` 仅在 `pertoken_basic` 路径使用，按 M 维广播。

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
| `scale` | 输入 | `BFLOAT16`、`FLOAT32` | `[N]` | per-channel 反量化 scale |
| `bias` | 输入 | `BFLOAT16`、`FLOAT32` | `[N]` | 浮点 bias；本 benchmark 固定使用浮点 bias 路径 |
| `perTokenScale` | 输入 | `FLOAT32` | `[M]` | per-token scale，仅 `pertoken_basic` 使用 |
| `y` | 输出 | `BFLOAT16`、`FLOAT16`、`FLOAT32` | `[M, N]` | 反量化后的矩阵乘输出 |

## 4. 约束说明

- 本 benchmark 固定 2D ND 输入，不覆盖 batch、转置、int4、fp8、per-tensor scale、offset 和融合激活路径。
- `variant` 仅支持 `bf16_basic` 与 `pertoken_basic`。
- `bf16_basic` 输入为 `x1/x2/scale/bias` 四个张量。
- `pertoken_basic` 输入为 `x1/x2/scale/bias/perTokenScale` 五个张量。
- `scale.shape == bias.shape == [N]`。
- `pertoken_basic` 必须满足 `perTokenScale.shape == [M]`。

## 5. 精度要求

本算子精度判定遵循 [`benchmark_spec.md` §4.4](../../../docs/spec/benchmark_spec.md)。通过条件与阈值参数定义在同目录 `proto.yaml` 的 `precision` 节点,以下仅说明本算子特定的取舍。

### 5.1 算子特定说明

- **`y` 阈值归属**:规则 `output_dtype`。int8 matmul 输出 int32 → 乘 fp32 scale(+ optional perTokenScale) + fp32 bias → cast 到目标 `y_dtype`;最终精度上限即输出自身 dtype,无中间精度损失隐患。
- **多 dtype 输出**:`y_dtype` 可变(BF16/FP16/FP32),评测脚本按当前 case 实际输出 dtype 查 SPEC §3 阈值表(BF16→2^-7、FP16→2^-10、FP32→2^-13)。
- **不覆盖 INT32 bias 路径**:INT32 bias 公式为 `(matmul + bias) * scale`,与本 benchmark 浮点 bias 路径 `matmul * scale + bias` 数学顺序不同;若后续扩展需独立 case 与独立精度规约。

## 6. 标准 Golden 代码

`golden.py` 使用 PyTorch 描述本 benchmark 的 selected C->V 路径，完整实现见同目录 `golden.py`。核心逻辑如下：

```python
out = torch.matmul(x1.to(torch.float32), x2.to(torch.float32))
out = out * scale.to(torch.float32).reshape(1, -1)
if variant == "pertoken_basic":
    out = out * perTokenScale.to(torch.float32).reshape(-1, 1)
out = out + bias.to(torch.float32).reshape(1, -1)
```

## 7. 额外信息

### 测试资料对应关系

- `docs/aclnnQuantMatmulV3.md`：说明浮点 bias 路径公式为 `out = x1@x2 * scale + bias`。
- `op_kernel/quant_batch_matmul_v3_bf16_basic.h`：基础 C->V 反量化路径。
- `op_kernel/quant_batch_matmul_v3_pertoken_basic.h`：带 per-token scale 的 C->V 反量化路径。

### 本 benchmark case 设计

`cases.yaml` 当前包含 20 个正向 case，覆盖 `bf16_basic/pertoken_basic`、`bfloat16/float16/float32` 输出、`BFLOAT16/FLOAT32` scale 与 bias、tail M、单 token、中等 K/N 和较大 N。所有 case 控制在 CPU golden 可快速运行的规模内。

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
