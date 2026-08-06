# FlatQuant per-token INT4 算子描述

## 1. 算子简介

FlatQuant 是面向大语言模型量化的融合算子，通过对三维输入依次执行右侧和左侧小矩阵乘法来平坦化输入特征，再按 token 计算缩放因子并将结果量化为 INT4。

**主要应用场景**：

- 大语言模型激活值的 per-token INT4 量化
- decode / batch-decode 阶段的低比特激活转换
- prefill 与 MoE token 场景中的融合旋转和量化
- 为后续低比特矩阵乘或低比特存储准备输入

**算子特征**：

- 难度等级：L3（Contraction + Reduction + Quantization）
- 输入：BFLOAT16，三维 `[K,M,N]`
- 计算：两次小矩阵乘法、per-token ReduceMax 和 INT4 量化
- 输出：物理 INT4 `out` 和 FP32 per-token `quantScale`
- 数据流：CUBE 矩阵乘计算后衔接 VECTOR 归约与量化

本 benchmark 固定 Atlas A2/A3、BF16 输入、per-token INT4 和
`tiling key=1 / MM_BASE_MODE` 路径，不覆盖 per-group FLOAT4_E2M1
以及 `MM_DOUBLE_MODE`、`MM_SPLIT_MODE`、`MM_HIGH_MODE`、`MM_ONE_MODE`。

对应的官方实现路径为：

```text
aclnnFlatQuant
  -> FlatQuant
  -> TILING_KEY_IS(1)
  -> FlatQuantCube<bfloat16_t, MM_BASE_MODE>
     + FlatQuantVec<bfloat16_t, MM_BASE_MODE>
```

## 2. 算子定义

### 数学公式

输入 `x` 先右乘 `kroneckerP2`：

$$
x' = x \mathbin{@} kroneckerP2
$$

再由 `kroneckerP1` 左乘：

$$
x'' = kroneckerP1 \mathbin{@} x'
$$

对每个 `x''[k,:,:]` 独立计算最大绝对值：

$$
maxAbs[k] = \max\left(\left|x''[k,:,:]\right|\right)
$$

计算 per-token 量化因子：

$$
quantScale[k] = \frac{maxAbs[k]}{7 / clipRatio}
$$

根据量化因子对矩阵乘结果进行归一化：

$$
normalized[k,:,:] = \frac{x''[k,:,:]}{quantScale[k]}
$$

最后按照 AscendC `CAST_RINT` 规则舍入，并饱和到 signed INT4 范围：

$$
out[k,:,:]
=
\operatorname{sat}_{[-8,7]}
\left(
\operatorname{rint}(normalized[k,:,:])
\right)
$$

### 步骤说明

1. **右矩阵乘**：`x1 = x @ kroneckerP2`，输出 shape 保持 `[K,M,N]`。
2. **左矩阵乘**：`x2 = kroneckerP1 @ x1`，输出 shape 保持 `[K,M,N]`。
3. **per-token 归约**：保留 `K` 维，对每个 `[M,N]` 切片求最大绝对值。
4. **scale 计算**：`quantScale = maxAbs / (7 / clipRatio)`，shape 为 `[K]`。
5. **归一化**：将 `[K]` reshape 为 `[K,1,1]`，计算 `x2 / quantScale`。
6. **全零处理**：当某个 token 的 `maxAbs=0` 时，该 token 的输出为 0。
7. **INT4 转换**：模拟 AscendC `Cast(..., RoundMode::CAST_RINT, ...)` 的最近偶数舍入，并按 signed INT4 范围 `[-8,7]` 饱和。

Golden 使用 INT8 Tensor 承载上述逻辑 INT4 数值。**强制约束：AscendC 算子的真实输出必须使用物理 INT4，不得以 INT8、FP16、BF16 或 FP32 输出替代；INT8 仅允许作为 Golden 和精度比较器的逻辑承载类型。**

## 3. 接口规范

### 算子原型

```python
cann_bench.flat_quant(
    Tensor x,
    Tensor kroneckerP1,
    Tensor kroneckerP2,
    float clipRatio,
) -> tuple[Tensor out, Tensor quantScale]
```

### 输入参数

| 参数 | 类型 | Shape | dtype | 描述 |
|---|---|---|---|---|
| `x` | Tensor（必选） | `[K,M,N]` | bfloat16 | 原始输入，`K` 表示 token 数 |
| `kroneckerP1` | Tensor（必选） | `[M,M]` | bfloat16 | 左乘小矩阵，dtype 必须与 `x` 相同 |
| `kroneckerP2` | Tensor（必选） | `[N,N]` | bfloat16 | 右乘小矩阵，dtype 必须与 `x` 相同 |
| `clipRatio` | float（必选） | 标量 | aclnn Host double 语义 | 裁剪比例，取值范围 `(0,1]` |

### 输出

| 参数 | Shape | dtype | 描述 |
|---|---|---|---|
| `out` | `[K,M,N]` | INT4 | 真实算子输出，每个元素为 signed INT4 |
| `quantScale` | `[K]` | FLOAT32 | 每个 token 一个量化因子 |

受 PyTorch reference 表达和精度比较方式限制，Golden 将 `out` 表示为 `torch.int8`，但其数值严格限制在 `[-8,7]`。该 INT8 Tensor 只是逻辑比较载体，不改变真实算子的 INT4 输出契约。

### 数据类型组合

| `x` | `kroneckerP1` | `kroneckerP2` | `out` | `quantScale` | 量化模式 |
|---|---|---|---|---|---|
| bfloat16 | bfloat16 | bfloat16 | INT4 | FLOAT32 | per-token INT4 |

### 规则与约束

- `x` 必须是非空三维 Tensor，shape 为 `[K,M,N]`。
- `kroneckerP1.shape == [M,M]`。
- `kroneckerP2.shape == [N,N]`。
- 三个输入 Tensor 的 dtype 必须完全相同。
- `1 <= K <= 262144`。
- `1 <= M <= 256`，`1 <= N <= 256`。
- `out=INT4` 时，`N` 必须是偶数。
- **AscendC 算子的 `out` 必须使用物理 INT4；不得将 Golden 的逻辑 INT8 承载方式作为算子输出实现。**
- `0 < clipRatio <= 1`。
- 输入和输出格式为 ND。
- INT4 路径的 `quantScale` 必须为 FLOAT32 `[K]`。

### 支持范围

| 维度 / 参数 | cases 覆盖 | 备注 |
|---|---:|---|
| `K` | 1～2048 | decode、prefill、MoE tail、长序列 |
| `M` | 2～128 | 不包含会进入 `MM_ONE_MODE` 的 `M=1` |
| `N` | 8～128 | 全部为偶数；包含 `N=10` 非 8/16 对齐尾块 |
| `M*N` | 16～16384 | 包含 650、1072 以及常见 LLM 隐藏维 |
| `clipRatio` | 0.5～1.0 | 覆盖无裁剪和不同裁剪强度 |
| 输入 dtype | BFLOAT16 | benchmark 固定 BF16 |

令 `mAlign=ceil16(M)`、`nAlign=ceil16(N)`。20 个 case 均满足：

- `M != 1`，不会进入 tiling key 5 / `MM_ONE_MODE`；
- `K>1` 时 `mAlign>64`，不会进入 tiling key 2 / `MM_DOUBLE_MODE`；
- `mAlign<=128` 且 `nAlign<=128`，不会进入 tiling key 3 / `MM_SPLIT_MODE`；
- `mAlign^2+nAlign^2+4*mAlign*nAlign` 最大为 98304，小于 key 4 /
  `MM_HIGH_MODE` 的 262144 阈值。

因此全部 case 最终落入 tiling key 1。Vector 侧的 `splitRow` 也始终不小于
`M`，统一执行 `FlatQuantVec::Quant`，不会切换到 `SplitQuant`。

## 4. 精度要求

采用 [CANN 生态算子精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md)

### 浮点输出 `quantScale`

对 `quantScale` 统计平均相对误差 MERE 和最大相对误差 MARE：

$$
\operatorname{MERE}
=
\operatorname{avg}
\left(
\frac{|actual-golden|}{|golden|+10^{-7}}
\right)
$$

$$
\operatorname{MARE}
=
\max
\left(
\frac{|actual-golden|}{|golden|+10^{-7}}
\right)
$$

`quantScale` 的接口 dtype 为 FLOAT32，但精度来源受 BF16 输入与矩阵乘路径限制，因此采用 `input_dtype_inherited`：

| 输出 | 继承 dtype | Threshold | 通过条件 |
|---|---|---:|---|
| `quantScale` | BFLOAT16 | `2^-7` | `MERE < 2^-7` 且 `MARE < 10 * 2^-7` |

### 整数输出 `out`

逻辑 INT4 输出使用 `int8_three_tier`，不使用相对误差：

| 指标 | 通过条件 |
|---|---|
| `abs(actual-golden) >= 2` 的元素数 | 必须为 0 |
| `abs(actual-golden) <= 1` 的元素比例 | 必须为 100% |
| `actual == golden` 的元素比例 | 必须不低于 99.5% |

## 5. 标准 Golden 代码

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


def flat_quant(x, p1, p2, clipRatio):
    """执行 BF16 per-token INT4 FlatQuant。

    计算语义：
        x1 = x @ p2
        x2 = p1 @ x1
        quantScale[k] = max(abs(x2[k, :, :])) / (7 / clipRatio)
        out = round(x2 / quantScale)，并裁剪到有符号 INT4 范围 [-8, 7]

    输入：
        x:
            shape 为 [K, M, N]、dtype 为 torch.bfloat16 的非空 Tensor。
            K <= 262144，M <= 256，N <= 256；INT4 输出要求 N 为偶数。
        p1:
            shape 为 [M, M]、dtype 为 torch.bfloat16 的非空 Tensor。
        p2:
            shape 为 [N, N]、dtype 为 torch.bfloat16 的非空 Tensor。
        clipRatio:
            范围为 (0, 1] 的浮点数；None 在 Golden 中按 1.0 处理。

    输出：
        out:
            逻辑 shape 为 [K, M, N]，数值范围为 [-8, 7]。
            算子真实输出必须使用有符号 INT4。当前 PyTorch Golden 使用
            torch.int8 Tensor 承载逻辑 INT4 数值；真实 INT4 的物理存储
            和解包由算子运行及比较环节处理。
        quantScale:
            shape 为 [K]、dtype 为 torch.float32 的 per-token 量化因子。

    典型 case（x、p1、p2 均为 torch.bfloat16）：
        - Smoke：x=[1, 2, 8]，p1=[2, 2]，p2=[8, 8]，clipRatio=1.0。
        - decode：x=[1, 64, 64]，p1=[64, 64]，p2=[64, 64]，
          clipRatio=1.0。
        - MoE/tail：x=[257, 86, 128]，p1=[86, 86]，
          p2=[128, 128]，clipRatio=0.5。

    完整测试集合见同目录 cases.csv。
    """
    if (clipRatio is None):
        clipRatio = 1.0

    # 输入 x 先右乘 p2，再由 p1 左乘。
    x1 = torch.matmul(x, p2)
    x2 = torch.matmul(p1, x1)

    # per-token 语义：每个 x2[k, :, :] 独立计算最大绝对值。
    x2_flat = x2.flatten(-2, -1)
    qscale = torch.abs(x2_flat).max(dim=-1, keepdim=True)[0].to(torch.float32)
    ratio = torch.ones_like(qscale) * 7 / clipRatio
    quantScale = torch.flatten(qscale / ratio)

    # 公式：out = x2 / quantScale。
    # reshape 仅用于将 [K] 的 quantScale 广播到 [K, M, N]。
    scale = quantScale.reshape(x2.shape[0], 1, 1)

    # 内核在 max_abs 为 0 时输出 0，避免产生 0 / 0。
    normalized = torch.where(
        scale > 0,
        x2 / scale,
        torch.zeros_like(x2),
    )

    # AscendC 内核使用 Cast(..., RoundMode::CAST_RINT, ...)：舍入到最近整数；
    # 当数值恰好位于两个整数的中点时取偶数（ties-to-even）。
    # 有符号 INT4 的表示范围为 [-8, 7]。
    # Golden 使用 INT8 承载逻辑 INT4；真实算子输出必须使用物理 INT4。
    out = torch.round(normalized).clamp(-8, 7).to(torch.int8)

    return out, quantScale
```

上述代码描述 benchmark Golden 的精度策略，不是对 AscendC kernel 中间 dtype 的逐指令复刻：

- Golden 保持 BF16 输入，不在两次 `torch.matmul` 前主动升精度；
- `qscale` 在 ReduceMax 后转换为 FP32，因此 `ratio`、除法和最终 `quantScale` 均为 FP32；
- `out` 按 AscendC `CAST_RINT` 语义舍入，再模拟 signed INT4 的饱和范围；
- Golden 使用 INT8 承载逻辑 INT4，精度比较前需要将真实 INT4 输出解包为同 shape 的逻辑整数。

如需进一步贴近已有 AscendC 实现，可以参考其 BF16 kernel 的主要中间路径。CUBE 使用 FP32 累加，第一次右矩阵乘结果转换为 BF16，第二次左矩阵乘结果转换为 FP16 并写入 workspace；VECTOR 从 FP16 结果计算绝对值和最大值，将最大值转换为 FP32 后生成 FP32 `quantScale`，再执行 FP32 缩放、`CAST_RINT` 舍入和物理 INT4 写回。该路径仅用于辅助精度分析和实现设计，不要求 Golden 或生成算子逐指令复现；本 benchmark 以数学语义和输入输出契约为准。

## 6. 额外信息

### Golden 调用示例

```python
import torch

from golden import flat_quant

# LLM prefill 示例：K=128，M*N=112*128=14336，命中 MM_BASE_MODE
x = torch.randn((128, 112, 128), dtype=torch.bfloat16)
p1 = torch.randn((112, 112), dtype=torch.bfloat16)
p2 = torch.randn((128, 128), dtype=torch.bfloat16)

out, quant_scale = flat_quant(x, p1, p2, 0.95)

# benchmark reference 的逻辑输出
assert out.shape == (128, 112, 128)
assert quant_scale.shape == (128,)
assert out.dtype == torch.int8
assert quant_scale.dtype == torch.float32
```

上例中的 INT8 是 Golden/比较器使用的逻辑载体；AscendC 算子的 `out` 接口仍须声明并写出 INT4。

### Case 设计

`cases.yaml` 与 `cases.csv` 一一对应，共包含 20 个 BF16 正向 case：

- 3 个基础与非对齐 tail case；
- 7 个 `K=1` decode case；
- 10 个 `K>1` prefill、MoE token tail 或长序列 case；
- 隐藏维 `M*N` 覆盖 `16/650/1072/4096/5120/7168/8192/11008/12288/14336/16384`；
- `K` 覆盖 `1/2/3/4/8/9/16/32/64/128/256/257/512`；
- 所有 `N` 都是偶数，满足 INT4 路径约束，并包含 `N=10` 尾块；
- 全部 case 固定命中 `TILING_KEY_IS(1)` 的 `MM_BASE_MODE` 模板；
- 三个输入的值域统一为 `[-1,1]`，覆盖正负值并控制矩阵乘结果的数值规模。

### 实现对齐依据

- **aclnnFlatQuant**：两段式 aclnn 接口
- **flat_quant_tiling.cpp**：`MM_BASE_MODE` 的 tiling key 选择条件
- **flat_quant.cpp**：A2/A3 `TILING_KEY_IS(1)` 的 CUBE/VECTOR 分发
- **flat_quant_cube.h**：`MM_BASE_MODE` 两次小矩阵乘法的 CUBE 路径
- **flat_quant_vec.h**：`MM_BASE_MODE` per-token ReduceMax、scale 和 INT4 量化路径
- **executor_aclnnFlatQuant.py**：`quantScale` 精度对比参考

### 参考资料

- [CANN 9.0 aclnnFlatQuant 文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/API/aolapi/context/ops-nn/aclnnFlatQuant.md)
- [cann-bench QuantMatmul desc.md 模板](https://gitcode.com/LAIM321/cann-bench/blob/master/tasks/level3/quant_matmul/desc.md)
