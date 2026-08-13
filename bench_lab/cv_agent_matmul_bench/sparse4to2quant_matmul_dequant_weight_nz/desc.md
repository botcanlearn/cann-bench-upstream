# Sparse4to2QuantMatmul 算子 API 描述

## 1. 算子简介

`sparse4to2quant_matmul_dequant` 对齐源码目录 `ops-nn/matmul/sparse4to2quant_matmul`,对应 aclnn 接口 `aclnnSparse4to2QuantMatmulWeightNz`。该算子完成 4:2 稀疏量化的矩阵乘:C 侧使用稀疏化的 INT8 weight(每连续 4 个元素恰好保留 2 个)+ 索引 `index` 重建做 matmul,V 侧执行 `xScale / sparseWeightScale` 反量化与可选 BF16 bias,属于 **C->V kernel flow**。

产品支持情况:

| 产品 | 是否支持 |
|------|----------|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | 支持 |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | 支持 |

## 2. 算子定义

设 `x` 的形状为 `[M, K]`,稠密 `weight` 的形状为 `[N, K]`,且满足 4:2 稀疏 pattern(每连续 4 元素恰好 2 个为 0)。计算公式:

$$
out = (x \cdot weight^T) \odot xScale[:, None] \odot sparseWeightScale[None, :] + bias
$$

NPU 实际执行时:
1. host 端调用 `aclnnTransSparse4to2Para` 把稠密 `weight` 压缩为 `sparseWeight`(只保留非零元素,FRACTAL_NZ 布局)+ `index`(UINT8 4D 索引)
2. device 端 C 段用 `sparseWeight + index` 重建做 INT8 matmul,V 段做 dequant + bias

数学上与"稠密 weight 直接做 matmul"等价(因为零元素相乘为零),因此 golden 直接用稠密 weight 计算。

## 3. 接口规范

### aclnn 两段式接口

```cpp
aclnnStatus aclnnSparse4to2QuantMatmulWeightNzGetWorkspaceSize(
  const aclTensor *x,
  const aclTensor *sparseWeight,
  const aclTensor *index,
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

### benchmark 抽象接口(driver 内部完成 trans 预处理)

```python
sparse4to2quant_matmul_dequant(
    x, weight, xScale, sparseWeightScale, bias=None, with_bias=True
) -> out
```

### 参数说明

| 参数 | 输入/输出 | dtype | format | shape | 说明 |
|------|-----------|-------|--------|-------|------|
| `x` | 输入 | `INT8` | ND | `[M, K]` | 量化激活矩阵 |
| `weight` | 输入 | `INT8` | ND | `[N, K]` | benchmark 接口接受**已 4:2 稀疏化的稠密表示**(50% 为 0);benchmark driver 内部调用 `aclnnTransSparse4to2Para` 转换为 NPU 需要的 `sparseWeight`(FRACTAL_NZ) + `index` |
| `xScale` | 输入 | `FLOAT32` | ND | `[M]` | per-token 反量化 scale |
| `sparseWeightScale` | 输入 | `FLOAT32` | ND | `[N]` | per-channel 反量化 scale |
| `bias` | 输入 | `BFLOAT16` | ND | `[N]` | 可选 bias,`with_bias=true` 时启用 |
| `out` | 输出 | `BFLOAT16` | ND | `[M, N]` | 反量化矩阵乘输出 |

## 4. 约束说明

Atlas A2/A3 产品约束:

- `K` 不超过 65535(来自 docs 约束;tiling 代码未直接校验,依赖 `aclnnTransSparse4to2Para` 上游)
- `xScale` 与 `sparseWeightScale` 都不能为 nullptr
- `weight` 必须满足 4:2 稀疏 pattern(每连续 4 元素恰好 2 个为 0);本 benchmark 数据准备阶段自动生成符合该约束的随机 weight
- `K` **不要求整除 8**:NPU 端通过 `CeilAlign(K, SPARSE_ATOMIC_SIZE=8)` 内部补齐;tiling 校验 `ceil(K/8)*8 == 2 * sparseWeight.K_half`
- `N` **不要求整除 16**:`sparseWeight` 是 FRACTAL_NZ,StorageShape `[ceil(N/16), ceil(K_half/32), 16, 32]`,NPU 内部按 16 ceil padding;输出 `out` shape 仍为逻辑 `[M, N]`,padding 不污染输出
- dtype 约束(tiling 硬校验,不可放宽):
  - `x` / `sparseWeight`:`INT8`
  - `xScale` / `sparseWeightScale`:`FLOAT32`
  - `bias`(可选):`BFLOAT16`
  - `out`:`BFLOAT16`
- op_def 注册了 `dtype` Int 属性(REQUIRED),但 tiling 当前写死 `out == BF16`,该 attr 实际未生效;benchmark 固定 `dtype = 27`(`ge::DT_BF16` 枚举值)以兼容
- 全部输入支持 `IgnoreContiguous`(非连续 tensor),但本 benchmark 只测连续 tensor
- 本 benchmark 固定走 `aclnnSparse4to2QuantMatmulWeightNz` 路径,不覆盖未来可能扩展的其他输出 dtype

## 5. 精度要求

本算子精度判定遵循 [`benchmark_spec.md` §4.4](../../../docs/spec/benchmark_spec.md)。通过条件与阈值参数定义在同目录 `proto.yaml` 的 `precision` 节点,以下仅说明本算子特定的取舍。

### 5.1 算子特定说明

- **`out` 阈值归属**:规则 `output_dtype`,固定 BFLOAT16 → 阈值 `2^-7`。INT8 matmul 在 Cube 内 int32 累加(bit-exact)+ V 段 FP32 dequant + cast 到 BF16,最终精度上限即输出 dtype,无中间精度损失隐患。
- **4:2 稀疏决定性**:`weight` 在数据准备阶段必须按 4:2 pattern 生成(每连续 4 元素恰好 2 个为 0)。NPU 端 `aclnnTransSparse4to2Para` 的压缩逻辑期望该 pattern;若 weight 不满足,NPU 实际计算结果与 golden(基于稠密 weight)将不一致,不应归因于精度问题。
- **无 V 段非线性 / 无 atomic**:本算子 V 段只做 element-wise dequant + bias,无激活函数,无跨核累加,精度归属直接由 output dtype 决定。

## 6. 标准 Golden 代码

`golden.py` 使用 PyTorch FP32 完成 INT8 matmul 与反量化,最后 cast 到 BF16。核心逻辑:

```python
out_fp32 = torch.matmul(x.float(), weight.t().float())   # [M, N]
out_fp32 = out_fp32 * xScale.float().reshape(-1, 1)
out_fp32 = out_fp32 * sparseWeightScale.float().reshape(1, -1)
if with_bias and bias is not None:
    out_fp32 = out_fp32 + bias.float().reshape(1, -1)
out = out_fp32.to(torch.bfloat16)
```

由于 `weight` 已是 4:2 稀疏化的稠密表示(50% 为 0),零元素参与 matmul 不贡献结果,故 golden 直接用稠密 weight 计算等价于 NPU 用压缩后的 `sparseWeight + index` 重建计算。

## 7. 额外信息

### 测试资料对应关系

- `docs/aclnnSparse4to2QuantMatmulWeightNz.md`:aclnn 接口规约与端到端调用示例
- `op_kernel/sparse4to2quant_matmul.h`:C 段 sparse matmul + V 段 dequant 实现
- `examples/`:M=64, K=512, N=128 端到端样例

### 本 benchmark case 设计

`cases.yaml` 当前包含 20 个正向 case:

- **case 1-18:对齐 shape 主体**
  - `M ∈ {1, 2, 3, 4, 5, 7, 8, 16, 24, 32, 33, 40, 64, 96, 128, 256}`(含 M=1 退化与非 2 的幂 tailM)
  - `K ∈ {64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65528}`(均为 8 倍数,含 K 上界附近)
  - `N ∈ {32, 64, 96, 128, 256, 512, 768}`(均为 16 倍数)
- **case 19-20:非对齐 shape 边界**(验证 NPU 内部 K 维 `CeilAlign(K, 8)` 与 N 维 FRACTAL_NZ `ceil(N/16)` padding 是否正确)
  - case 19:`K=132`(8 非对齐,4 整除以满足 4:2 pattern)+ `N=64`(对齐)
  - case 20:`K=128`(对齐)+ `N=33`(16 非对齐,ceil 到 48)
- `with_bias` 启用 / 不启用 case 数大致平衡
- weight 数据准备脚本按 4:2 pattern 随机置零(每连续 4 元素随机选 2 个置 0);**K 维必须是 4 的倍数才能干净切 4:2 分组**,这是 case 设计的隐含约束

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


def sparse4to2quant_matmul_dequant(
    x: torch.Tensor,
    weight: torch.Tensor,
    xScale: torch.Tensor,
    sparseWeightScale: torch.Tensor,
    bias: torch.Tensor = None,
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
        optional BF16 bias path.
    """
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

    # Cube accumulates INT8 x INT8 in INT32 bit-exactly (op_kernel/sparse4to2quant_matmul.h:
    # mmOutGm_ is int32_t, CMatmulType=MatmulType<..., int32_t>, srcLocal=AllocTensor<int32_t>).
    # An FP32 matmul rounds partial sums once they exceed 2^24, so accumulate in int32 first.
    out_i32 = torch.matmul(x.to(torch.int32), weight.t().to(torch.int32))   # [M, N] exact int32
    out_fp32 = out_i32.to(torch.float32)
    # Dequant ORDER matches kernel: AscendDequant applies sparseWeightScale (per-channel) FIRST,
    # then PertokenCalculate applies xScale (per-token). FP32 mul is non-associative so order is
    # precision-relevant. (op_kernel/sparse4to2quant_matmul.h BasicDequantCompute + docs formula:
    # out = x@sparseWeight * sparseWeightScale * xScale + bias)
    out_fp32 = out_fp32 * sparseWeightScale.to(torch.float32).reshape(1, -1)
    out_fp32 = out_fp32 * xScale.to(torch.float32).reshape(-1, 1)
    if with_bias:
        # Kernel casts bf16 bias -> fp32 (Cast CAST_NONE, lossless) then adds in fp32 (CalBiasAdd).
        out_fp32 = out_fp32 + bias.to(torch.float32).reshape(1, -1)
    # Final fp32 -> bf16 via round-to-nearest-even (kernel Cast RoundMode::CAST_RINT == torch default).
    return out_fp32.to(torch.bfloat16)
```
