# Fp4DequantMatmul 算子 API 描述

## 1. 算子简介

`Fp4DequantMatmul` 面向通用 FP4 E2M1 低比特权重推理，完成 4-bit 权重解包、按 block scale 反量化和矩阵乘。本接口使用 `uint8` 承载打包编码，不声明原生 FP4 Tensor dtype。

**主要应用场景**：
- FP4 E2M1 LLM linear 层推理
- 低比特权重压缩后的在线反量化
- dequant + matmul 融合优化

**算子特征**：
- 难度等级：L3（Contraction）
- 首版固定 FP4 E2M1 codebook
- 一个 `uint8` 存两个 4-bit code，低 4 位在前
- `scales` 为 `[N, ceil(K / block_size)]`

## 2. 算子定义

### 数学公式

```text
W = dequant_fp4_e2m1(w_packed, scales, block_size)
y = x @ W.T
```

其中 `x.shape[-1] = K`，`W.shape = [N, K]`。首版 FP4 E2M1 codebook 为：

```text
[0, 0.5, 1, 1.5, 2, 3, 4, 6, -0, -0.5, -1, -1.5, -2, -3, -4, -6]
```

## 3. 接口规范

### 算子原型

```python
cann_bench.fp4_dequant_matmul(
    Tensor x,
    Tensor w_packed,
    Tensor scales,
    int out_features,
    int in_features,
    int block_size=16,
) -> Tensor y
```

### 输入参数说明

| 参数 | 类型 | Shape | dtype | 描述 |
|------|------|-------|-------|------|
| `x` | Tensor | `[..., K]` | float16 / bfloat16 / float32 | activation 输入 |
| `w_packed` | Tensor | `[ceil(N*K/2)]` | uint8 | 打包 FP4 code |
| `scales` | Tensor | `[N, ceil(K/block_size)]` | float16 / bfloat16 / float32 | block scale |
| `out_features` | int | - | - | 输出特征 `N` |
| `in_features` | int | - | - | 输入特征 `K` |
| `block_size` | int | - | - | scale block size |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| `y` | `[..., N]` | 与 `x` 相同 | matmul 输出 |

### 数据类型

| x dtype | w_packed dtype | scales dtype | 输出 dtype |
|---------|----------------|--------------|-----------|
| float16 | uint8 | float16 / bfloat16 / float32 | float16 |
| bfloat16 | uint8 | float16 / bfloat16 / float32 | bfloat16 |
| float32 | uint8 | float16 / bfloat16 / float32 | float32 |

### 规则与约束

- `x.shape[-1] == in_features`。
- `w_packed` 至少包含 `ceil(out_features * in_features / 2)` 个 uint8。
- `scales.shape == [out_features, ceil(in_features / block_size)]`。
- `block_size > 0`。
- 首版只覆盖固定 FP4 E2M1 codebook，不覆盖 NVFP4 的分层缩放约定、CANN MXFP4 的 E8M0 scale 或 FP8。
- Ascend 910 通过 `uint8` 解包后计算实现兼容路径；Ascend 950 可在保持相同接口语义的前提下使用 FP4 能力优化。该接口不表示 Ascend 910 原生支持 FP4 dtype。

### 支持范围

| 维度 / 参数 | 范围 | 备注 |
|---|---|---|
| `M` | 1 ~ 128 | `x` 可为 1D/2D/3D，末维为 K |
| `K` | 16 ~ 512 | 覆盖 block 对齐和非对齐 |
| `N` | 16 ~ 512 | 输出特征 |
| `block_size` | 16 / 32 | 首版公开 case 覆盖 |

## 4. 精度要求

采用[生态算子精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md)进行验证。

**误差指标**：

1. 平均相对误差（MERE）：采样点中相对误差平均值

   $$
   \text{MERE} = \text{avg}(\frac{\text{abs}(actual - golden)}{\text{abs}(golden)+\text{1e-7}})
   $$

2. 最大相对误差（MARE）：采样点中相对误差最大值

   $$
   \text{MARE} = \max(\frac{\text{abs}(actual - golden)}{\text{abs}(golden)+\text{1e-7}})
   $$

**通过标准**：

| 数据类型 | FLOAT16 | BFLOAT16 | FLOAT32 | HiFLOAT32 | FLOAT8 E4M3 | FLOAT8 E5M2 |
|----------|---------|----------|---------|-----------|-------------|-------------|
| **通过阈值(Threshold)** | 2^-10 | 2^-7 | 2^-13 | 2^-11 | 2^-3 | 2^-2 |

当平均相对误差 MERE < Threshold，最大相对误差 MARE < 10 * Threshold 时判定为通过。

## 5. 标准 Golden 代码

```python
#!/usr/bin/python3
# coding=utf-8

import torch


_FP4_E2M1_TABLE = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)


def _unpack_nibbles(packed: torch.Tensor, n_values: int) -> torch.Tensor:
    values = packed.to(torch.uint8).reshape(-1)
    low = values & 0x0F
    high = (values >> 4) & 0x0F
    codes = torch.stack((low, high), dim=1).reshape(-1)
    return codes[:n_values].to(torch.long)


def fp4_dequant_matmul(
    x: torch.Tensor,
    w_packed: torch.Tensor,
    scales: torch.Tensor,
    out_features: int,
    in_features: int,
    block_size: int = 16,
) -> torch.Tensor:
    if x.shape[-1] != in_features:
        raise ValueError("x.shape[-1] must equal in_features")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if scales.dim() != 2 or scales.shape[0] != out_features:
        raise ValueError("scales must be [out_features, ceil(in_features / block_size)]")

    total = out_features * in_features
    codes = _unpack_nibbles(w_packed, total)
    table = _FP4_E2M1_TABLE.to(device=x.device)
    weight = table.index_select(0, codes.to(x.device)).reshape(out_features, in_features)

    scale_blocks = scales.float()
    expanded_scales = scale_blocks.repeat_interleave(block_size, dim=1)[:, :in_features]
    weight = weight * expanded_scales.to(weight.device)

    out = torch.matmul(x.float(), weight.t())
    return out.to(x.dtype)
```
