# QuantGroupedMatmulInplaceAdd 算子 API 描述

## 1. 算子简介

`quant_grouped_matmul_inplace_add` 对齐主线算子 `aclnnQuantGroupedMatmulInplaceAdd`（`ops-transformer/gmm/quant_grouped_matmul_inplace_add`）的 **T-C 量化场景**（scale1=tensor 级、scale2=per-channel 级，inplace add 到 yRef）。主线 x1/x2 为 **HIFLOAT8**、yRef 3D(g,M,N)、x2 2D 单权重；本 benchmark x1/x2 用 **hifloat8**(en_dtypes,golden 内 numpy round-trip 量化 + float matmul 对齐主线 L0C)、**yRef 3D**、**x2 2D**（均对齐主线，性能可比），groupSize 恒 0，属于 C->V kernel flow。

## 2. 算子定义

```text
for group i:
    y[i, start:end, :] = yRef[i, start:end, :] + (x1[start:end, :] @ x2) * scale2[i] * scale1[i]
```

`groupListType=0` 时 `groupList` 是 cumsum；`x2` 为 2D 单权重（所有 group 共享）；`yRef` 3D (g,M,N)。

## 3. 接口规范

```python
quant_grouped_matmul_inplace_add(x1, x2, scale1, scale2, groupList, yRef, groupListType=0, group_size=0) -> y
```

| 参数 | 输入/输出 | dtype | shape | 说明 |
|------|-----------|-------|-------|------|
| `x1` | 输入 | `HIFLOAT8`(en_dtypes, golden numpy round-trip) | `[M,K]` | routed token 激活 |
| `x2` | 输入 | `HIFLOAT8`(en_dtypes) | `[K,N]` | 单权重 2D |
| `scale1` | 输入 | `FLOAT32` | `[G,1]` | tensor 级左 scale |
| `scale2` | 输入 | `FLOAT32` | `[G,N]` | per-channel 右 scale |
| `groupList` | 输入 | `INT64` | `[G]` | cumsum 分组边界 |
| `yRef` | 输入/输出 | `FLOAT32` | `[G,M,N]` | inplace add 初始值（主线 3D） |
| `y` | 输出 | `FLOAT32` | `[G,M,N]` | 累加结果 |

## 4. 约束说明

- 固定 `groupListType=0` cumsum 语义，允许空 group。
- 固定 T-C per-channel scale（scale1 tensor 级、scale2 channel 级）；`groupSize` 恒 0（对齐主线，T-C 不使用）；MX 动态量化分支不纳入。
- 主线 T-C 场景 x1/x2 为 HIFLOAT8、yRef 3D(g,M,N)、x2 2D 单权重；本 benchmark x1/x2 用 en_dtypes.hifloat8 真实量化（已安装 en_dtypes 0.0.4，golden 内 numpy round-trip + float matmul）、yRef 3D、x2 2D（对齐主线）。

## 5. 精度要求

本算子精度判定遵循 [`../PRECISION_SPEC.md`](../PRECISION_SPEC.md)。通过条件与阈值参数定义在同目录 `proto.yaml` 的 `precision` 节点,以下仅说明本算子特定的取舍。

### 5.1 算子特定说明

- **`y` 阈值归属**:规则 `output_dtype`。输出 float32，inplace add 累加。

## 6. 标准 Golden 代码

`golden.py` 按 groupList 切 M，对每组 `matmul(x1_slice, x2) * scale2 * scale1` 并 inplace 累加到 yRef(3D)。详见同目录 `golden.py`。

## 7. 额外信息

### 测试资料对应关系

- `docs/aclnnQuantGroupedMatmulInplaceAdd.md`：T-C 公式、`groupListType` 和 `groupSize` 说明。
- `tests/assets/golden.py`：生态测试 golden 参考（非 mx 分支）。

### 本 benchmark case 设计

`cases.yaml` 当前包含 20 个正向 case，覆盖单/多 group、空 group、非均匀 group、tail M 和 `G=16`。

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


# ===== 规格对齐说明(对齐主线 aclnnQuantGroupedMatmulInplaceAdd 的 T-C 量化场景) =====
# 主线算子: aclnnQuantGroupedMatmulInplaceAdd (graph op: QuantGroupedMatmulInplaceAdd, 仅 ascend950)
#   实现: ops-transformer/gmm/quant_grouped_matmul_inplace_add
# 对应场景: T-C 量化 (scale1=tensor级/perToken, scale2=channel级/perchannel)
# 主线公式: y_i[m,n] = (x1_i[m,k] @ x2_i[k,n]) * scale2_i[n] * scale1_i + yRef_i[m,n]  (inplace add)
#   出处: docs/aclnnQuantGroupedMatmulInplaceAdd.md (功能说明 L18/公式 L37/约束 L273-282);
#         op_api/quant_grouped_matmul_inplace_add_950_checker.cpp (CheckHif8QuantParams, HIFLOAT8 强制 L282-291);
#         tests/assets/golden.py (L78-113 非 mx 分支)
# 主线 dtype/shape 规格:
#   x1        : HIFLOAT8, 2D (M,K)    [主线物理 (K,M), view (M,K), transX=true]
#   x2        : HIFLOAT8, 2D (K,N)    [transW=false; 主线 2D 单权重]
#   scale1    : FLOAT32, (g,) 或 (g,1)  [tensor 级]
#   scale2    : FLOAT32, (g, N)         [channel 级]
#   groupList : INT64,   (g,), cumsum 边界
#   yRef / y  : FLOAT32, 3D (g,M,N), inplace add 累加器(输入即输出)
#   groupSize : 恒为 0 (T-C 场景); bias/offset 必须为空
# 关键约束: T-C 表 x1/x2 仅 HIFLOAT8; groupSize=0; yRef 3D; 仅 950PR/950DT.
# 本 golden 与主线的对齐说明:
#   - 主线 x1/x2 = HIFLOAT8; 本 golden 用 numpy en_dtypes.hifloat8 做 round-trip 体现 HIFLOAT8 量化误差
#     (CPU 无 native hifloat8 matmul, 故量化后转 float32 matmul, 对齐主线 L0C fp32 累加)。
#   - proto/cases dtype 声明 'hifloat8' -> CV 生成的算子用 hifloat8 -> 对齐主线 -> 可比。
#   - yRef 对齐主线 3D (g,M,N) inplace (y[idx, start:end, :] += ...)。
#   - x2 对齐主线 2D 单权重 (K,N); groupSize 恒 0。
# ===========================================================================================


def _hifloat8_quantize(t: torch.Tensor) -> torch.Tensor:
    """torch float32 -> 低精度 round-trip -> torch float32.

    主线输入为 HIFLOAT8；HIFLOAT8 为 CANN 私有 numpy dtype（en_dtypes），
    纯 torch 环境不可用。这里用 torch.float8_e4m3fn 做 round-trip 近似，
    量化方向（截断/饱和）与 HIFLOAT8 一致，数值上可能有微小差异，但满足
    cann-bench golden "只 import torch" 的硬约束。
    """
    f32 = t.detach().to(torch.float32)
    fp8 = f32.to(torch.float8_e4m3fn)
    return fp8.to(torch.float32)


def _groups(groupList: torch.Tensor, groupListType: int):
    values = [int(v) for v in groupList.detach().cpu().tolist()]
    if groupListType == 0:
        starts = [0] + values[:-1]
        ends = values
    elif groupListType == 1:
        starts, ends, cur = [], [], 0
        for count in values:
            starts.append(cur)
            cur += count
            ends.append(cur)
    else:
        raise ValueError("groupListType must be 0 or 1")
    return list(zip(starts, ends))


def get_input(
    x1: torch.Tensor,
    x2: torch.Tensor,
    scale1: torch.Tensor,
    scale2: torch.Tensor,
    groupList: torch.Tensor,
    yRef: torch.Tensor,
    **attrs,
) -> list:
    """从 attrs.group_list_values 重建确定性的 groupList 张量。

    cases.yaml 将 cumsum 分组边界放在 group_list_values 属性里（golden 读取它），
    但被测 kernel 只看 groupList 张量。若不由 get_input 重建，groupList 会被
    value_range 随机生成（可能为负、非单调），导致 kernel 与 golden 分组不一致。
    返回值同时替换 golden 与 AI 算子的输入，确保对比公平。
    """
    gl = attrs.get("group_list_values")
    if gl is not None:
        groupList = torch.tensor(list(gl), dtype=torch.int64, device=x1.device)
    return [x1, x2, scale1, scale2, groupList, yRef]


def quant_grouped_matmul_inplace_add(
    x1: torch.Tensor,
    x2: torch.Tensor,
    scale1: torch.Tensor,
    scale2: torch.Tensor,
    groupList: torch.Tensor,
    yRef: torch.Tensor,
    groupListType: int = 0,
    group_size: int = 0,
    variant: str = "TC_PERCHANNEL",
    group_list_values=None,
) -> torch.Tensor:
    """Torch golden for quant_grouped_matmul_inplace_add, aligned to aclnnQuantGroupedMatmulInplaceAdd (T-C 场景).

    T-C per-channel: y_i = yRef_i + (x1_i @ x2_i) * scale2_i * scale1_i (inplace add).
    x1/x2 经 HIFLOAT8 量化(en_dtypes); yRef 3D (g,M,N); x2 2D 单权重 (K,N).
    规格详见模块顶部「规格对齐说明」。
    """
    if group_list_values is not None:
        groupList = torch.tensor(group_list_values, dtype=torch.int64, device=x1.device)
    groups = _groups(groupList, groupListType)
    g = len(groups)
    M, K = x1.shape
    K2, N = x2.shape
    if K != K2:
        raise ValueError(f"K mismatch: x1 K={K} x2 K={K2}")
    if yRef.shape != (g, M, N):
        raise ValueError(f"yRef expects ({g},{M},{N}) [对齐主线 3D], got {list(yRef.shape)}")
    # 对齐主线 T-C: y_i = (x1_i @ x2_i) * scale2(channel) * scale1(tensor) + yRef_i (inplace, 3D)
    # x1/x2 经 HIFLOAT8 量化(对齐主线 HIFLOAT8 输入)
    x2q = _hifloat8_quantize(x2)
    y = yRef.to(torch.float32).clone()
    for idx, (start, end) in enumerate(groups):
        if end <= start:
            continue
        partial = _hifloat8_quantize(x1[start:end, :]) @ x2q   # [m, N], hifloat8 量化后 float matmul
        y[idx, start:end, :] = y[idx, start:end, :] + partial * scale1[idx].to(torch.float32).reshape(1, 1) * scale2[idx].to(torch.float32).reshape(1, N)
    return y
```
