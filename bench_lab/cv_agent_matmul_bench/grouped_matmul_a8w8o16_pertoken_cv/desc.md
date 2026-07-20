# GroupedMatmul (A8W8O16 per-token, C→V) 算子 API 描述

## 1. 算子简介

**算子特征**：
- 难度等级：L3（Contraction）

`grouped_matmul` 按 `groupList` 将 routed token 分配给不同 expert，并对每个 expert 执行独立的矩阵乘（MoE grouped matmul）。本 benchmark 对齐源码目录 `ops-transformer/gmm/grouped_matmul`，固定选取 `A8W8O16_pertoken_CV` 路径：`x` 与 `weight` 为 int8，AIC（Cube）执行 grouped int8 matmul 得到 int32 中间结果，AIV（Vector）完成 per-channel scale × per-token scale 反量化，并按 `y_dtype` 写回 bfloat16/float16。

该路径在 `grouped_matmul.cpp` 的 `QUANT_A8W8O16` 分支中进入 `GMM_CV_SPLIT_IMP`，属于 **C→V kernel flow**（cube 产出 int32、vector 反量化）。本 benchmark **不覆盖** A8W8O8/O32 纯 cube 路径，也**不覆盖** A4W4/A8W4/weight-nz/anti-quant 路径。

## 2. 算子定义

设 `x` 的形状为 `[M, K]`，`weight` 的形状为 `[E, K, N]`，`scale` 的形状为 `[E, N]`，`perTokenScale` 的形状为 `[M]`。`groupList` 使用 cumsum（累积和）语义，本 benchmark 固定 `groupList[-1] == M`。

```text
start = 0
for expert i in [0, E):
    end = groupList[i]
    if end == start:          # 空 expert：无 token，跳过
        continue
    Xi = x[start:end, :]                                          # [m_i, K] int8
    Yi = (Xi.int32 @ weight[i].int32)                            # [m_i, N] int8*int8 -> int32
    Yi = Yi * scale[i][None, :]                                  # per-channel (per-N 列)
    Yi = Yi * perTokenScale[start:end, None]                     # per-token (per-M 行)
    y[start:end, :] = Yi
    start = end
y = y.to(y_dtype)             # bfloat16 或 float16
```

允许相邻两个 `groupList` 值相等，此时对应 expert 没有 token，跳过其 matmul 与写回，且不污染输出。

## 3. 接口规范

benchmark 抽象接口（与 `golden.py` 一致）：

```python
grouped_matmul(
    x, weight, scale, groupList, perTokenScale,
    variant="A8W8O16_pertoken_CV",
    group_list_values=None,
    y_dtype="bfloat16",
    split_item=3, group_type=0, group_list_type=0,
) -> y
```

参数说明：

| 参数 | 输入/输出 | dtype | shape | 说明 |
|------|-----------|-------|-------|------|
| `x` | 输入 | `INT8` | `[M, K]` | routed token 激活 |
| `weight` | 输入 | `INT8` | `[E, K, N]` | expert 权重，本 benchmark 固定不转置 |
| `scale` | 输入 | `FLOAT32`、`BFLOAT16` | `[E, N]` | expert/channel 反量化 scale |
| `groupList` | 输入 | `INT64` | `[E]` | cumsum token 边界 |
| `perTokenScale` | 输入 | `FLOAT32` | `[M]` | per-token activation scale |
| `y` | 输出 | `BFLOAT16`、`FLOAT16` | `[M, N]` | 反量化 grouped matmul 输出 |

> 注：`group_list_values`（见 §4 attrs）是 golden 与数据准备共同使用的确定性 cumsum 列表；当其提供时 golden 以它为准，`groupList` 张量本身只承载形状。

## 4. 约束说明

### 4.1 固定路径与固定参数

- `variant` 固定为 `"A8W8O16_pertoken_CV"`，golden 对其它 variant 直接报错。
- `split_item = 3`、`group_type = 0`、`group_list_type = 0`（golden 对其它取值直接报错）。
- `y_dtype` 由 case 指定，取值为 `"bfloat16"` 或 `"float16"`。
- `weight` 固定为 `[E, K, N]` 布局，**不覆盖**转置权重。
- `scale` 固定为 per-channel `[E, N]`，**不覆盖** per-tensor scale。
- 本 benchmark 固定**无 bias、无 offset、无激活输入输出、无动态输出量化**。

### 4.2 形状/语义约束（构造数据与实现都必须满足）

- `x` 为 2D `[M, K]`；`weight` 为 3D `[E, K, N]`，且 `weight` 的 `K` 必须等于 `x` 的 `K`。
- `scale` 形状必须为 `[E, N]`；`perTokenScale` 元素个数必须等于 `M`。
- `group_list_values` 必须是长度为 `E` 的**非负、单调非递减 cumsum 序列**，且最后一个值 `group_list_values[-1] == M`（它是各 expert token 计数的前缀和）。
- 允许相邻 `group_list_values` 值相等（空 expert），此时该 expert 区间无 token，跳过 matmul 与写回；评测只统计已有 token 行。

## 5. 实现约束与参考设计

> 本节分两层。**§5.1 硬约束**是正确性与合法性底线，必须遵守；**§5.2 参考设计与已知反模式**是非强制指导——给出一条已验证可行的实现路径与避坑提示，**鼓励在守住 §5.1 的前提下自行设计更优方案；当生成算子性能不足时，应主动超越本参考路径，而非照抄。**

### 5.1 硬约束（正确性 + 反作弊，必须遵守）

**目标语义与乘加顺序**

```text
for each non-empty expert i (group_list_values[i] > group_list_values[i-1]):
    acc_int32 = x[start:end] (int8) @ weight[i] (int8)     # AIC/Cube，int8 输入 -> int32 累加
    v = Cast<float32>(acc_int32)                            # AIV/Vector，int32 -> float32
    v = v * scale[i][None, :]                               # per-channel (per-N 列；scale 若 bf16 先转 fp32)
    v = v * perTokenScale[start:end][:, None]               # per-token (per-M 行)
    y[start:end, :] = Cast<y_dtype>(v)                      # bfloat16 或 float16，写回 GM
```

- **乘加顺序必须与 golden 一致**：`((matmul) * scale[i]) * perTokenScale`。不得把 `scale[i] * perTokenScale` 预合成后再乘，会改变浮点舍入顺序，与 golden 产生精度残差。
- **dtype / 累加精度**：matmul 为 int8 输入、int32 累加；反量化前 `int32 -> float32`，`scale` 为 bfloat16 时先转 float32；按 `y_dtype` 转换为 bfloat16/float16 写回 GM。必须正确支持 `scale` 的 `float32` 与 `bfloat16` 两种 dtype，以及 `y` 的 `bfloat16` 与 `float16` 两种 dtype。
- **shape 断言**：`x` 为 2D `[M, K]`、`weight` 为 3D `[E, K, N]` 且 `K` 一致；`scale` 为 `[E, N]`（per-channel `scale[E,N]` 广播到列），`perTokenScale` 为 `[M]`（per-token 广播到行）；`group_list_values` 为长度 `E` 的非负、单调非递减 cumsum 序列，末值 `group_list_values[-1] == M`（见 §4.2）。
- **算子语义不变量**：`group_list` cumsum 末值必须等于 `M`；空 expert（`groupList[i] == groupList[i-1]`）必须跳过——不发起 matmul，也不污染输出；`scale[E,N]` 按列广播、`perTokenScale[M]` 按行广播的语义不得改变。

**真融合，禁退化（反作弊）**

- 必须生成**真正的 Cube + Vector 融合 AscendC kernel**，**禁止**退化为以下任意一种：纯 AIC、纯 AIV、纯 CPU、torch、aclnn 高层组合算子（包括直接调用 `aclnnGroupedMatmulV*`）、Python fallback。
- **matmul 必须落 Cube/AIC**：每个非空 expert 的 `x[start:end] @ weight[expert_id]`（`int8 [m_i, K] × int8 [K, N] -> int32 [m_i, N]`）必须由 **AIC/Cube** 用 AscendC Cube / MatMul / MMAD 原语完成；**禁止**在 AIV 侧用逐元素循环模拟矩阵乘，也**禁止**把 int8 提升为 float 后用 vector 累加替代 cube；**禁止** AIC 标量循环模拟矩阵乘。
- **反量化 / per-channel / per-token scale 必须片上向量化**：`int32 -> float32` 反量化、per-channel `scale[expert_id]` 乘法、per-token `perTokenScale[start:end]` 乘法、转换为 bfloat16/float16 并写回 GM，必须在片上 **AIV/Vector** 完成；**禁止**下沉到 torch / host / CPU / aclnn 高层 / Python fallback 计算输出。
- **kernel/host 命名含 `custom`**：AscendC 自定义 kernel 的 `__global__` 核函数名和 Host `_do` 入口名必须包含 `custom`（如 `<op_name>_custom` / `<op_name>_custom_<dtype>`）；不得生成不含 `custom` 的 profiling kernel 名。
- **热路径禁标量循环**：**禁止**在 `model_new_tilelang.py` 与 `model_new_ascendc.py` 中使用 torch 算子做任何实际计算；热路径**禁止**标量逐元素 `GetValue` / `SetValue` 写法（少量边界 / 控制元数据除外），必须使用块级 / 向量化原语（`T.copy`、`T.tile.*`、矩阵/向量原语等）。

**跨核同步正确性**

- AIC→AIV 的 int32 中间结果交接必须**正确同步**、保证跨核数据**可见**、**无数据竞争**（否则结果错）；不得用局部 `PipeBarrier` 冒充跨 AIC/AIV 的可见性同步。
- 按 expert 切换区间时，必须正确处理 M 方向 tail（`m_i` 非 16 对齐）与空 expert 跳过的边界，保证正确性。

**精度**

- 精度遵循 §6 / [`../PRECISION_SPEC.md`](../PRECISION_SPEC.md) 与 `proto.yaml` 的 `precision` 节点。

### 5.2 参考设计与已知反模式（指导，非强制，鼓励超越）

> 以下是一条**已验证可行**的 C→V 实现路径与若干提示，作为起点与避坑参考，**不是必须照抄的配方**。在守住 §5.1 的前提下，鼓励 agent 探索更优实现；**性能不足时优先在此处突破。**

- **参考数据流（C→V）**：AIC（Cube）按 `groupList` 切分 token 区间，对每个非空 expert 把 `int8 [m_i, K] × int8 [K, N] -> int32 [m_i, N]` 的分块 matmul 结果写入 workspace / GM ring 中间缓冲；AIV（Vector）从 workspace 读取 int32 分块结果，完成反量化 + per-channel scale + per-token scale + dtype cast 并写回 GM `y[start:end, :]`。这是一种直接可行的切分；agent 可自行探索更优方案（如把反量化融进 matmul epilogue、不同 tile/buffer 策略、1c2v 分工等）。
- **参考 workspace 布局**：每 expert 的 int32 中间结果在 workspace / GM ring slot 中摆放，ring slot 复用维护明确的 C2V / V2C 生命周期。具体 workspace 布局由 agent 按性能选择，只要满足 §5.1 的同步正确性即可。
- **参考同步原语**：AIC↔AIV 的 int32 中间结果交接走 workspace / GM ring slot，可用 TQue（或 ping-pong / queue style tile buffer）管理生产-消费生命周期；AIC 每次 `CrossCoreSetFlag` 前用 `PipeBarrier<PIPE_ALL>` 排空 GM 写以保证中间结果对 AIV 可见，多 AIV lane 按 collective 语义同步（不要只让 lane0 推进全局进度）。具体同步原语（`CrossCoreSetFlag` / `PipeBarrier` / ring-slot 选择）由 agent 按性能选择，只要满足 §5.1 的同步正确性即可。
- **参考 tiling**：kernel tiling / launch 体现 **AIC + AIV 混合执行**（1c2v：1 个 cube 配 2 个 vector lane），按 expert / 按 M / 按 N 的分块策略与 TQue 切分按 shape 自适应选择；面向 LLM 大 shape（M 最高 5120，K/N 最高 5120/4096，E 取 2/4/8/16）应自适应选择 tile 粒度，避免小 tile 造成核利用率过低。
- **已知反模式（建议避开）**：epilogue 逐行用标量 `GetValue` 读 per-token scale、且每个向量 op 后用局部 `PipeBarrier` 串行的写法，在大 shape 下被 fence 串行主导；优先**多行整块** `[m_i, N]` 向量处理。
- 建议在设计文档与 `trace.md` 记录：AIC/AIV 分工、按 expert / 按 M / 按 N 的分块策略、workspace 布局（含每 expert int32 中间结果摆放）、空 expert 跳过逻辑、同步方式，便于性能复盘。

## 6. 精度要求

本算子精度判定遵循 [`../PRECISION_SPEC.md`](../PRECISION_SPEC.md)。通过条件与阈值参数定义在同目录 `proto.yaml` 的 `precision` 节点，以下仅说明本算子特定的取舍。

### 6.1 算子特定说明

- **`y` 阈值归属**：规则 `output_dtype`。`y_dtype` 可变（BF16 / FP16），评测脚本按当前 case 实际输出 dtype 查 SPEC §3 阈值表。
- **乘加顺序**：精度强约束见 §5.1，`((matmul) * scale) * perTokenScale`（以 golden 为准），不得预合 scale。
- **空 expert**：`group_list_values` 允许相邻相等（空 expert），此时该 expert 没有 token 区间，golden 与 actual 都不会额外生成输出行；评测只统计已有 token 行。

## 7. 验收约束

1. 所有 `cases.yaml` / `cases.csv` 中的用例**精度必须全部达标**；不得只通过 `basic_case` 或部分 `general_case` 后停止。
2. 所有用例的**计算速度必须优于 torch 小算子拼接实现**；如任一用例性能未优于该基线，必须继续优化 AscendC tiling、workspace 流水或 vector 反量化路径，直到性能达标或在 `trace.md` 中记录明确阻塞原因。

## 8. 标准 Golden 代码

`golden.py` 使用 PyTorch float32 完成 int8 matmul 与反量化，最后按 `y_dtype` 转换：

```python
for expert_id, end in enumerate(group_list_values):
    if end == start:
        continue
    xi = x[start:end].float()
    wi = weight[expert_id].float()
    yi = torch.matmul(xi, wi)
    yi = yi * scale[expert_id].float().reshape(1, -1)
    yi = yi * perTokenScale[start:end].float().reshape(-1, 1)
    out[start:end] = yi
    start = end
y = out.to(y_dtype)   # bfloat16 或 float16
```

## 9. 额外信息

### 9.1 测试资料对应关系

- `docs/aclnnGroupedMatmulV3.md`：描述 groupType、splitItem、groupList 和 grouped matmul 基础约束。
- `op_kernel/grouped_matmul.cpp`：`QUANT_A8W8O16` 分支调用 `GMM_CV_SPLIT_IMP`。
- `op_kernel/grouped_matmul.h`：读取 cumsum `groupList` 并按 expert 切分 token。

### 9.2 本 benchmark case 设计

`cases.yaml` 当前包含 20 个正向 case，遵循「少量小 shape + 大量 LLM shape」标准：

- **6 个 small case**（smoke / edge）：单 expert / 少量 expert、空 first expert、空 middle expert（相邻 group 值相等）、many-experts-small（E16）、tail M，复用并适配原始小用例。
- **14 个 LLM case**（MoE-prefill）：M 最高 5120、K/N 最高 5120/4096，E 取 2/4/8/16，使用 ragged / 非 16 对齐且求和等于 M 的 `group_list_values`，realistic 维度（768/1024/1152/1536/2048/2304/3072/4096/5120 等）。
- 覆盖 `bfloat16` / `float16` 输出与 `float32` / `bfloat16` scale 两种 dtype 组合；attrs 结构与原始用例保持一致；`value_range` 沿用算子原始 `[-2, 2]`，`baseline_perf_us = 0.0`、`t_hw_us = 0.0`。

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


def get_input(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    groupList: torch.Tensor,
    perTokenScale: torch.Tensor,
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
        groupList = torch.tensor(list(gl), dtype=torch.int64, device=x.device)
    return [x, weight, scale, groupList, perTokenScale]


def grouped_matmul(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    groupList: torch.Tensor,
    perTokenScale: torch.Tensor,
    variant: str = "A8W8O16_pertoken_CV",
    group_list_values=None,
    y_dtype: str = "bfloat16",
    split_item: int = 3,
    group_type: int = 0,
    group_list_type: int = 0,
) -> torch.Tensor:
    """Torch golden for selected grouped_matmul A8W8O16 per-token C->V path."""
    if variant != "A8W8O16_pertoken_CV":
        raise ValueError(f"Unsupported grouped_matmul variant: {variant}")
    if split_item != 3 or group_type != 0 or group_list_type != 0:
        raise ValueError("This benchmark fixes split_item=3, group_type=0, group_list_type=0")
    if x.dim() != 2:
        raise ValueError(f"x expects 2D [M,K], got {list(x.shape)}")
    if weight.dim() != 3:
        raise ValueError(f"weight expects 3D [E,K,N], got {list(weight.shape)}")

    m, k = x.shape
    expert_num, wk, n = weight.shape
    if wk != k:
        raise ValueError(f"weight K ({wk}) must match x K ({k})")
    if scale.shape != (expert_num, n):
        raise ValueError(f"scale expects shape [{expert_num}, {n}], got {list(scale.shape)}")
    if perTokenScale.numel() != m:
        raise ValueError(f"perTokenScale length ({perTokenScale.numel()}) must match M ({m})")

    groups = _cumsum_group_list(groupList, m, expert_num, group_list_values)
    out = torch.zeros(m, n, dtype=torch.float32, device=x.device)

    start = 0
    for expert_id, end in enumerate(groups):
        if end == start:
            continue
        xi = x[start:end].to(torch.float32)
        wi = weight[expert_id].to(torch.float32)
        yi = torch.matmul(xi, wi)
        yi = yi * scale[expert_id].to(torch.float32).reshape(1, n)
        yi = yi * perTokenScale[start:end].to(torch.float32).reshape(-1, 1)
        out[start:end] = yi
        start = end

    return _cast_output(out, y_dtype)


def _cumsum_group_list(groupList, total_m: int, group_num: int, group_list_values):
    if group_list_values is not None:
        values = torch.tensor(group_list_values, dtype=torch.int64)
    else:
        values = groupList.to(torch.int64).flatten()
    if values.numel() != group_num:
        raise ValueError(f"groupList length ({values.numel()}) must match expert count ({group_num})")
    if bool(torch.any(values < 0)):
        raise ValueError("groupList values must be non-negative")
    if bool(torch.any(values[1:] < values[:-1])):
        raise ValueError("groupList must be non-decreasing cumsum")
    if int(values[-1]) != total_m:
        raise ValueError(f"groupList last value ({int(values[-1])}) must equal M ({total_m})")
    return [int(v) for v in values.tolist()]


def _cast_output(out: torch.Tensor, y_dtype: str) -> torch.Tensor:
    name = str(y_dtype).split(".")[-1].lower()
    if name in ("bf16", "bfloat16"):
        return out.to(torch.bfloat16)
    if name in ("fp16", "float16", "half"):
        return out.to(torch.float16)
    raise ValueError(f"Unsupported y_dtype: {y_dtype}")
```
