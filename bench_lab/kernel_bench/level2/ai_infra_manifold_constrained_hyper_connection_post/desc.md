# AiInfraManifoldConstrainedHyperConnectionPost 算子 API 描述

## 1. 算子简介

Manifold Constrained Hyper Connection 后处理算子。将 h_post 加权的 h_out 与 h_res 加权的 x 合并得到最终输出。该算子是超连接（Hyper Connection）结构的关键组成部分，用于流形约束下的多流信息融合。

**主要应用场景**：
- DeepSeek 等 MoE 模型的超连接（Hyper Connection）后处理
- 多流注意力输出的约束融合
- 流形约束下的残差连接与输出加权的统一计算

**算子特征**：
- 难度等级：L2（Attention）
- 4 输入，1 输出，无属性参数
- 支持 ND 格式输入
- x 和 h_out 支持 FLOAT16 / BFLOAT16，h_res 和 h_post 固定 FLOAT32

## 2. 算子定义

### 数学公式

输入 x、h_res、h_out、h_post，计算输出：

$$
y = h\_post.unsqueeze(-1) \cdot h\_out.unsqueeze(-2) + \sum_{n} h\_res.unsqueeze(-1) \cdot x.unsqueeze(-2)
$$

具体来说，给定 x 为 [..., S, n, D]，h_res 为 [..., S, n, n]，h_out 为 [..., S, D]，h_post 为 [..., S, n]：

$$
output_{...,S,n,D} = h\_post_{...,S,n,1} \cdot h\_out_{...,S,1,D} + \sum_{k} h\_res_{...,S,n,k} \cdot x_{...,S,k,D}
$$

## 3. 接口规范

### 算子原型

```python
cann_bench.ai_infra_manifold_constrained_hyper_connection_post(Tensor x, Tensor h_res, Tensor h_out, Tensor h_post) -> Tensor output
```

### 输入参数说明

| 参数名 | 输入/输出 | 描述 | 数据类型 | 数据格式 |
|--------|----------|------|---------|---------|
| x | 输入 | 输入张量, [..., S, n, D] | FLOAT16 / BFLOAT16 | ND |
| h_res | 输入 | 残差权重, [..., S, n, n] | FLOAT32 | ND |
| h_out | 输入 | 输出权重, [..., S, D] | FLOAT16 / BFLOAT16 | ND |
| h_post | 输入 | 后处理权重, [..., S, n] | FLOAT32 | ND |

### 输出

| 参数名 | 输入/输出 | 描述 | 数据类型 | 数据格式 |
|--------|----------|------|---------|---------|
| output | 输出 | 输出张量, [..., S, n, D] | 与 x 一致 | ND |

### 规则与约束

- x 和 h_out 的 dtype 必须一致（同为 FLOAT16 或同为 BFLOAT16）
- h_res 和 h_post 固定为 FLOAT32
- 所有输入的前缀维度必须一致
- x 的倒数第二维 = h_post 的最后一维 = h_res 的倒数第二维 = h_res 的最后一维 = n
- x 的最后一维 = h_out 的最后一维 = D
- n 的取值范围为 4~8
- D 支持 384, 2560, 3264, 4368, 5120, 6544

### 支持范围

| 维度 / 参数 | 范围 | 备注 |
|---|---|---|
| x dtype | float16 / bfloat16 | 与 h_out 一致 |
| h_res dtype | float32 | 固定 float32 |
| h_out dtype | float16 / bfloat16 | 与 x 一致 |
| h_post dtype | float32 | 固定 float32 |
| n | 4 ~ 8 | 超连接流数 |
| D | 384 ~ 6544 | head_dim 维度 |
| S | 1 ~ 2048 | 序列维度 |
| 前缀 batch 维度 | 1 ~ 64 | batch 维度 |

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
import torch
from typing import Optional
import random
import numpy as np


def get_input(x: torch.Tensor, h_res: torch.Tensor,
              h_out: torch.Tensor, h_post: torch.Tensor, **kwargs):
    """
    预处理输入，确保满足算子的语义约束：
    - h_res: 使用 Sinkhorn 算法生成双随机矩阵（行和=列和=1）
    - h_post: 应用 sigmoid 归一化到 (0, 1) 范围
    """
    # h_res → doubly stochastic matrix via Sinkhorn
    h_res = _generate_dsmat_sinkhorn(h_res)

    # h_post → sigmoid normalization
    h_post = torch.sigmoid(h_post)

    return (x, h_res, h_out, h_post)


def _generate_dsmat_sinkhorn(orig_tsr: torch.Tensor,
                              max_iter: int = 100,
                              tol: float = 1e-8):
    """使用 Sinkhorn 算法生成双随机矩阵（行和=列和=1）"""
    shape_list = list(orig_tsr.shape)
    orig_device = orig_tsr.device
    orig_dtype = orig_tsr.dtype

    # 创建随机正矩阵
    random_tensor = torch.rand(shape_list, dtype=torch.float32, device=orig_device) + 1e-10

    # Sinkhorn 迭代归一化
    for _ in range(max_iter):
        # 行归一化
        row_sum = random_tensor.sum(dim=-1, keepdim=True)
        random_tensor = random_tensor / (row_sum + 1e-10)

        # 列归一化
        col_sum = random_tensor.sum(dim=-2, keepdim=True)
        random_tensor = random_tensor / (col_sum + 1e-10)

        # 检查收敛
        row_check = torch.abs(random_tensor.sum(dim=-1) - 1).max()
        col_check = torch.abs(random_tensor.sum(dim=-2) - 1).max()
        if row_check < tol and col_check < tol:
            break

    return random_tensor.to(orig_dtype).to(orig_device)


def ai_infra_manifold_constrained_hyper_connection_post(
        x: torch.Tensor, h_res: torch.Tensor,
        h_out: torch.Tensor, h_post: torch.Tensor):
    """
    Manifold Constrained Hyper Connection 后处理算子。

    公式:
        y = h_post.unsqueeze(-1) * h_out.unsqueeze(-2)
            + sum(h_res.unsqueeze(-1) * x.unsqueeze(-2), dim=-3)

    参数:
        x: [..., S, n, D] float16/bfloat16
        h_res: [..., S, n, n] float32 — 双随机矩阵
        h_out: [..., S, D] float16/bfloat16, dtype 与 x 一致
        h_post: [..., S, n] float32 — 已 sigmoid 归一化

    返回:
        output: [..., S, n, D], dtype 与 x 一致
    """
    orig_dtype = x.dtype

    x_fp32 = x.float()
    h_out_fp32 = h_out.float()
    h_res_fp32 = h_res.float()
    h_post_fp32 = h_post.float()

    y = (h_post_fp32.unsqueeze(-1) * h_out_fp32.unsqueeze(-2) +
         torch.sum(h_res_fp32.unsqueeze(-1) * x_fp32.unsqueeze(-2), dim=-3))

    return y.to(orig_dtype)
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

# 示例: batch=1, S=1024, n=4, D=5120
x = torch.randn(1, 1024, 4, 5120, dtype=torch.float16, device="npu")
h_res = torch.randn(1, 1024, 4, 4, dtype=torch.float32, device="npu")
h_out = torch.randn(1, 1024, 5120, dtype=torch.float16, device="npu")
h_post = torch.randn(1, 1024, 4, dtype=torch.float32, device="npu")

output = cann_bench.ai_infra_manifold_constrained_hyper_connection_post(
    x, h_res, h_out, h_post
)
```
