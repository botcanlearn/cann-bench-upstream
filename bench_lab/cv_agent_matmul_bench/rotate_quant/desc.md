# RotateQuant 算子 API 描述

## 1. 算子简介

`rotate_quant` 对输入张量 `x` 进行旋转变换，再执行对称动态量化。本 benchmark 选取 `float16/bfloat16 -> int8 + float32 scale` 路径，作为 C->V 算子基准：AIC 执行旋转矩阵乘，AIV 执行逐行动态量化并写回输出；V2C 同步只用于 workspace/ping-pong 流控。

产品支持情况：

| 产品 | 是否支持 |
|------|----------|
| Ascend 950PR/Ascend 950DT | 不支持 |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | 支持 |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | 支持 |
| Atlas 200I/500 A2 推理产品 | 不支持 |
| Atlas 推理系列产品 | 不支持 |
| Atlas 训练系列产品 | 不支持 |

## 2. 算子定义

设 `x` 的形状为 `[m, n]`，`rotation` 的形状为 `[k, k]`。旋转变换为：

$$
Y = (x.\text{reshape}(*, k) @ \text{rotation}).\text{reshape}(m, n)
$$

对称动态量化为逐行量化：

$$
s_i = \frac{\max_{j \in [0,\ n-1]} |Y_{i,j}|}{C_{\text{MAX}}}, \qquad
y_{i,j} = \frac{Y_{i,j}}{s_i}
$$

`C_MAX` 在 int8 场景取 127，quint4x2 场景取 7。本 benchmark 固定 int8 路径，CPU golden 在归一化后执行 round 和 `[-127, 127]` 裁剪。当前 benchmark 固定 `alpha=0.0`，即不做 clamp。

## 3. 接口规范

### aclnn 两段式接口

第一段接口获取 workspace 大小和执行器：

```cpp
aclnnStatus aclnnRotateQuantGetWorkspaceSize(
  const aclTensor   *x,
  const aclTensor   *rotation,
  float              alpha,
  aclTensor         *yOut,
  aclTensor         *scaleOut,
  uint64_t          *workspaceSize,
  aclOpExecutor    **executor)
```

第二段接口执行算子：

```cpp
aclnnStatus aclnnRotateQuant(
  void            *workspace,
  uint64_t         workspaceSize,
  aclOpExecutor   *executor,
  aclrtStream      stream)
```

### 参数说明

| 参数 | 输入/输出 | dtype | format | shape | 说明 |
|------|-----------|-------|--------|-------|----------|
| `x` | 输入 | `BFLOAT16`、`FLOAT16` | `ND` | 2D | 待旋转量化的输入张量，支持非连续 Tensor，不支持空 Tensor |
| `rotation` | 输入 | `BFLOAT16`、`FLOAT16` | `ND` | 2D | 旋转矩阵，支持非连续 Tensor，不支持空 Tensor |
| `alpha` | 输入 | `float` | - | 标量 | 实际接口的 clamp 缩放系数；本 benchmark 固定为 `0.0`，表示不做 clamp |
| `yOut` | 输出 | `INT8`、`INT32`、`FLOAT4_E2M1` | `ND` | 2D | 量化后的输出张量，需预先分配 |
| `scaleOut` | 输出 | `FLOAT32`、`FLOAT8_E8M0` | `ND` | 1D | 动态量化计算出的缩放系数，需预先分配 |

benchmark 抽象接口：

```python
rotate_quant(x, rotation, alpha=0.0, y_dtype="int8") -> (y, scale)
```

## 4. 约束说明

Atlas A2/A3 产品约束如下：

- `x` 的 shape 为 `(M, N)`，`rotation` 的 shape 为 `(K, K)`，且 `rotation` 必须是方阵。
- `N` 必须是 `K` 的整数倍，且 `N` 必须可以整除 8。
- `x` 和 `rotation` 的数据类型必须相同。
- `scaleOut` 的 shape 必须是 `(M)`。
- `N` 的范围为 `[128, 16000]`，`K` 的范围为 `[16, 1024]`。
- 当 `yOut` 为 `INT8` 时，shape 为 `(M, N)`；当 `yOut` 为 `INT32` 时，shape 为 `(M, N // 8)`。

本 benchmark 仅覆盖 `yOut=INT8`、`scaleOut=FLOAT32`、`x/rotation=BFLOAT16/FLOAT16`，并固定 `alpha=0.0`。

## 5. 精度要求

本算子精度判定遵循 [`../PRECISION_SPEC.md`](../PRECISION_SPEC.md)。通过条件与阈值参数定义在同目录 `proto.yaml` 的 `precision` 节点,以下仅说明本算子特定的取舍。

### 5.1 算子特定说明

- **`scale` 阈值归属**:规则 `input_dtype_inherited`。`scale` 自身 dtype 为 FLOAT32,但其数值由 NPU 上 BF16/FP16 rotation matmul 推导,精度上限受输入 dtype 制约;直接按 FLOAT32 阈值 `2^-13` 会假阴性。具体阈值见 `proto.yaml.precision.outputs[scale].threshold_per_input_dtype`(BF16→2^-7、FP16→2^-10)。
- **`y` 阈值归属**:规则 `int8_three_tier`,采用默认参数(fatal=2 / tolerance=1 / bit_exact_ratio=0.99)。
- **`scale = 0` 边界**(全零输入或某行 amax=0):由 SPEC §5 小值特殊处理覆盖,actual 为 fp32 噪声 `~1e-5` 时改走绝对误差判定,无算子专属逻辑。

## 6. 标准 Golden 代码

`golden.py` 使用 PyTorch 描述本 benchmark 的 int8 路径，完整实现见同目录 `golden.py`。核心逻辑如下：

```python
y_rot = torch.matmul(
    x.to(torch.float32).reshape(m, n // k, k),
    rotation.to(torch.float32),
).reshape(m, n)

c_max = 127.0
max_abs = torch.abs(y_rot).amax(dim=-1, keepdim=True)
scale = max_abs / c_max
normalized = torch.where(scale > 0, y_rot / scale, torch.zeros_like(y_rot))
y = torch.round(normalized).clamp(-c_max, c_max).to(torch.int8)
```

## 7. 额外信息

### 测试资料对应关系

- `tests/ut/op_host/test_rotate_quant_tiling.cpp`：覆盖 BF16/FP16 输入、INT8 输出、`K=16/64/256/1024`、`N=128/256/1024` 等 tiling 场景，并包含 `N` 不能被 `K` 整除的失败用例。
- `tests/ut/op_kernel/test_rotate_quant.cpp`：包含 BF16+INT8 和 FP16+INT8 两条 kernel UT 路径。

### 本 benchmark case 设计

`cases.yaml` 当前包含 20 个正向 case，覆盖 `bfloat16/float16` 输入、`K=16/32/64/128/256/512/800/1000/1024`、`N=128` 到 `16000`，均满足 `N % K == 0` 与 `N % 8 == 0`。其中 `M=1`、非 2 的幂 `M`、`N=K`、`N=2K`、多 block、`N=16000` 上界等场景用于覆盖 tiling 边界；`M=4, N=256, K=64` 对齐小规模 BF16/INT8 kernel UT 路径。大 `N/K` case 控制在较小 `M`，避免 CPU golden 在 agent 调试阶段过慢。

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


def rotate_quant(
    x: torch.Tensor,
    rotation: torch.Tensor,
    alpha: float = 0.0,
    y_dtype: str = "int8",
):
    """Torch golden for the selected rotate_quant int8 path."""
    if str(y_dtype).lower() != "int8":
        raise ValueError("This benchmark fixes rotate_quant y_dtype=int8")
    if alpha != 0.0:
        raise ValueError("This benchmark fixes rotate_quant alpha=0.0")
    if x.dim() != 2:
        raise ValueError(f"rotate_quant expects x to be 2D, got shape {list(x.shape)}")
    if rotation.dim() != 2 or rotation.shape[0] != rotation.shape[1]:
        raise ValueError(f"rotation must be square, got shape {list(rotation.shape)}")

    m, n = x.shape
    k = rotation.shape[0]
    if n % k != 0:
        raise ValueError(f"N ({n}) must be divisible by K ({k})")
    if n % 8 != 0:
        raise ValueError(f"N ({n}) must be divisible by 8")
    if n < 128 or n > 16000:
        raise ValueError(f"N ({n}) must be in [128, 16000]")
    if k < 16 or k > 1024:
        raise ValueError(f"K ({k}) must be in [16, 1024]")

    # AIC computes the rotation matmul with the cube unit's internal fp32
    # accumulator, but MatmulImpl's C tensor is typed DTYPE_X (see
    # op_kernel/rotate_quant.cpp: `using cType = MatmulType<..., DTYPE_X>`),
    # so the rotated result is truncated to x.dtype (fp16/bf16) when written
    # to the workspace GM buffer. The AIV quant stage then reads it back and
    # up-casts to fp32 with RoundMode::CAST_NONE (op_kernel/rotate_quant.h
    # `CopyInVector`). Skipping this fp16/bf16 round-trip overstates the
    # precision of the rotated activations before quantization.
    x_fp32 = x.to(torch.float32)
    rot_fp32 = rotation.to(torch.float32)
    y_rot_fp32 = torch.matmul(x_fp32.reshape(m, n // k, k), rot_fp32).reshape(m, n)
    y_rot = y_rot_fp32.to(x.dtype).to(torch.float32)

    # Per-row symmetric dynamic quant, matching op_kernel/rotate_quant.h AIV stage:
    #   scaleTmp      = amax_j |Y[i, j]|                       (ComputeReduceMax)
    #   quantScaleTmp = 127.0 / scaleTmp                        (Div constScale/scaleTmp)
    #   normalized    = Y * quantScaleTmp                       (Mul, broadcast)
    #   y             = round-to-nearest-even(normalized)       (Cast CAST_RINT)
    #   scaleOut      = scaleTmp * (1/127)                      (Mul constInvScale)
    # The reciprocal-multiply form (not Y / (amax/127)) and the 1/127 constant
    # are reproduced so fp32 rounding matches the kernel rather than an
    # algebraically-equal but numerically-different division.
    c_max = 127.0
    inv_c_max = float(1.0) / c_max
    max_abs = torch.abs(y_rot).amax(dim=-1, keepdim=True)
    quant_scale = torch.where(max_abs > 0, c_max / max_abs, torch.zeros_like(max_abs))
    normalized = y_rot * quant_scale
    # CAST_RINT is round-half-to-even; the symmetric scale keeps values within
    # [-127, 127] by construction, so the clamp is a defensive no-op.
    y = torch.round(normalized).clamp(-c_max, c_max).to(torch.int8)
    scale = (max_abs * inv_c_max).reshape(m).to(torch.float32)
    return y, scale
```
