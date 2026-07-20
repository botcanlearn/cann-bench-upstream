# QuantGroupedMatmulInplaceAdd 算子 API 描述

## 1. 算子简介

**算子特征**：
- 难度等级：L3（Contraction）

`quant_grouped_matmul_inplace_add` 对齐源码目录 `ops-transformer/gmm/quant_grouped_matmul_inplace_add`。本 benchmark 固定 **T-C per-channel** 代表路径：C（Cube）侧按 `groupList` 执行 grouped INT8 matmul，V（Vector）侧做 scale 反量化并就地累加到 `yRef`，属于 **C->V kernel flow**。面向 MoE-prefill（routed token 激活 × 各 expert 权重）的 grouped quantized matmul + inplace-add 累加器场景。

本目录后缀 `_tc_perchannel` 用于明确本 benchmark 固定 **T-C per-channel scale** 路径：`scale1` 为 group 级标量左 scale（per-group / per-token-group），`scale2` 为 per-channel 右 scale；**MX 动态量化分支（dynamic per-group quant）不纳入本目录**。

## 2. 算子定义

```text
for group i in [0, G):                # groups 由 groupList(cumsum) 切分
    start, end = group_i 的 token 区间
    if end <= start: continue         # 空 group 跳过
    partial = x1[start:end, :] @ x2[i]                         # [m_i, N], INT8·INT8 -> INT32
    y[start:end, :] = yRef[start:end, :]
                      + partial * scale1[i] * scale2[i]        # 反量化 + inplace add
```

更精确的 reference（与 `golden.py` 完全一致，**乘加顺序不可改**）：

```python
y = yRef.to(float32).clone()
for i, (start, end) in enumerate(groups):     # groups 由 groupList cumsum 切分
    if end <= start:
        continue
    partial = x1[start:end, :].to(float32) @ x2[i].to(float32)   # [m_i, N]
    y[start:end, :] = y[start:end, :] + partial * scale1[i].reshape(1, 1) * scale2[i].reshape(1, N)
return y
```

`groupListType=0` 时 `groupList` 是 **cumsum（累加边界）**：第 `i` 组 token 区间为 `[groupList[i-1], groupList[i])`，约定 `groupList[-1] = 0`（即首组起点为 0）。

## 3. 接口规范

```python
quant_grouped_matmul_inplace_add(x1, x2, scale1, scale2, groupList, yRef, groupListType=0, group_size=[1,0,0], variant="TC_PERCHANNEL") -> y
```

| 参数 | 输入/输出 | dtype | shape | 说明 |
|------|-----------|-------|-------|------|
| `x1` | 输入 | `INT8` | `[M, K]` | routed token 激活 |
| `x2` | 输入 | `INT8` | `[G, K, N]` | group（expert）权重 |
| `scale1` | 输入 | `FLOAT32` | `[G, 1]` | group 级左 scale（标量，广播到 `[m_i, N]`） |
| `scale2` | 输入 | `FLOAT32` | `[G, N]` | per-channel 右 scale（`[1, N]` 广播） |
| `groupList` | 输入 | `INT64` | `[G]` | cumsum 分组边界 |
| `yRef` | 输入/输出 | `FLOAT32` | `[M, N]` | inplace add 初始值（累加器） |
| `y` | 输出 | `FLOAT32` | `[M, N]` | 累加结果 |

维度关系（硬约束，与 `golden.py` 的 shape 校验一致）：

- `x1.shape[1] == x2.shape[1] == K`；
- `x2.shape[0] == len(groupList) == G`（即 expert/group 数 `E == G`）；
- `scale1.shape == [G, 1]`，`scale2.shape == [G, N]`；
- `yRef.shape == (M, N) == (x1.shape[0], x2.shape[2])`，且 `y` 与 `yRef` **同形同 dtype**；
- golden 内显式断言 `len(groups) == g and x1.shape[1] == k and yRef.shape == (x1.shape[0], n)`，任一不满足直接抛 `shape mismatch`。

## 4. 约束说明

### 4.1 固定参数

- `groupListType = 0`（cumsum 语义；`group_list_values` 作为整型 cumsum 列表传入并覆盖 `groupList` 张量取值）。
- `group_size = (1, 0, 0)`。
- `variant = "TC_PERCHANNEL"`（固定 T-C per-channel scale 路径，不涉及 MX 动态量化分支）。
- 输出 `y` dtype 固定为 `float32`。

### 4.2 分组与边界约束

- `group_list_values` 必须是 **长度为 G 的累加（cumulative）非递减 int 列表**，且**最后一个值严格等于 M**；第 `i` 组区间为 `[group_list_values[i-1], group_list_values[i])`（`i=0` 时起点为 0）。
- **允许空 group**：当某组 `start == end`（相邻 cumsum 值相等，含首组为 0、中间为 0、连续相等）时，跳过该组的 cube/vector 计算，不写回对应行。
- `len(group_list_values) == G == x2.shape[0]`；若 `groupList` 末值不等于 M 或 `yRef` 形状不等于 `[M, N]`，validator 会以 “groupList last value must equal M” / yRef-shape 类错误判 FAIL。

### 4.3 数值与范围

- `value_range = [-8, 8]`：`x1`/`x2` 为 INT8，取整后落在 `[-8, 8]`；`scale1`/`scale2`/`yRef` 为 float32，落在 `[-8, 8]`。
- INT8 × INT8 的 partial 必须以 **INT32 累加器** 表达，再 cast FP32 做反量化，避免中间溢出/丢精。

## 5. 实现约束与参考设计

> 本节分两层。**§5.1 硬约束**是正确性与合法性底线，必须遵守；**§5.2 参考设计与已知反模式**是非强制指导——给出一条已验证可行的实现路径与避坑提示，**鼓励在守住 §5.1 的前提下自行设计更优方案；当生成算子性能不足时，应主动超越本参考路径，而非照抄。** 本节折叠原始 `prompt.md` 的算子语义、实现要求、强制 CV 约束与验收约束，使本文档自洽，不再依赖外部 prompt。

### 5.1 硬约束（正确性 + 反作弊，必须遵守）

- **真融合，禁退化**：必须生成**真正的 Cube + Vector 融合 AscendC kernel**；**禁止退化为**纯 AIV（在 Vector 侧用逐元素循环模拟矩阵乘）、纯 CPU、torch 计算、aclnn 高层组合算子、Python fallback。
- **grouped matmul 必须落 Cube**：每个非空 group 的 INT8 grouped matmul `x1[start:end, :] @ x2[i]` 必须由 **AIC/Cube 侧完成**，使用 AscendC Cube / MatMul / MMAD 原语；**累加器 dtype 必须为 INT32**；**禁止在 AIV 侧用逐元素循环/标量模拟矩阵乘**。INT8×INT8 的 partial 以 INT32 累加器表达，再 cast FP32 反量化，避免中间溢出/丢精（见 §4.3）。
- **分组与 group_list 语义**：按 `groupList`（`groupListType=0`，cumsum 累加边界）切 token，第 `i` 组区间为 `[groupList[i-1], groupList[i])`（首组起点 0），**cumsum 末值必须 == M**；**允许空 group**：当某组 `end <= start`（相邻 cumsum 值相等）时，cube 与 vector 两侧都必须正确跳过，不写回对应行。
- **per-channel 反量化 + inplace-add 必须片上向量化**：INT32 partial → FP32 cast、乘 `scale1[i]`（group 级标量，`reshape(1,1)` 广播全 `[m_i, N]`）、乘 `scale2[i]`（per-channel，`reshape(1,N)` 按列广播）、加 `yRef[start:end, :]`、float32 写回 `y[start:end, :]`，必须在片上 **AIV/Vector** 完成；**禁止**下沉到 torch / host / CPU / aclnn / Python 计算输出，**禁止**把 scale / dequant / add 搬到 host。`yRef` / `y` 按 `[M, N]` 的 float32 行布局正确读写，保持 inplace-add 累加器语义（`yRef` shape == 输出 `[M, N]`）。
- **乘加顺序与语义（精度强约束）**：乘加顺序必须保持为 `partial * scale1[i] * scale2[i]`，再 `+ yRef[start:end, :]`（与 §2 / golden 一致）；**不得**把 `scale1[i] * scale2[i]` 预合后再乘，**不得**把 `scale * yRef` 预合，否则改变浮点舍入顺序、与 golden 产生精度残差。
- **dtype 与 shape 断言**：dtypes 与 §3 接口表一致（`x1`/`x2` INT8、`scale1`/`scale2`/`yRef`/`y` FLOAT32、`groupList` INT64）；满足 §3 的维度关系与 golden 内 shape 断言（`len(groups)==G`、`x1.shape[1]==K`、`yRef.shape==(M, N)`），任一不满足直接 FAIL。
- **跨核同步正确性**：AIC→AIV 交接、以及 **group 间依赖** 必须正确同步、保证跨核数据可见，不得出现数据竞争（否则结果错）；不得用局部 `PipeBarrier` 冒充跨核可见性同步。
- kernel `__global__` 核函数名与 Host `_do` 入口名必须含 `custom`（如 `<op_name>_custom` / `<op_name>_custom_do` / `<op_name>_custom_<dtype>`），不得生成不含 `custom` 的 profiling kernel 名；AscendC 热路径**禁止标量逐元素**写法（`GetValue/SetValue` 循环，少量边界 / 控制元数据除外），必须使用块级 / 向量化原语。
- 精度遵循 §6 / [`../PRECISION_SPEC.md`](../PRECISION_SPEC.md) 与 `proto.yaml` 的 `precision` 节点。

### 5.2 参考设计与已知反模式（指导，非强制，鼓励超越）

> 以下是一条**已验证可行**的 C→V 实现路径与若干提示，作为起点与避坑参考，**不是必须照抄的配方**。在守住 §5.1 的前提下，鼓励 agent 探索更优实现；**性能不足时优先在此处突破。**

- **参考数据流（C→V）**：AIC 按 `groupList` 切 token，对每个非空 group 执行 INT8 grouped matmul，累加为 INT32 partial 并写入 workspace / GM 中间缓冲（INT32 partial ring/slot）；AIV 读取 INT32 partial 后完成 `Cast<float32>` → `* scale1[i]` → `* scale2[i]` → `+ yRef[start:end, :]` → float32 写回 `y[start:end, :]`。这是一种直接可行的切分；agent 可自行探索更优方案（如把反量化融进 matmul epilogue、不同 tile/buffer 策略、group 调度/分工等）。
- **参考 workspace / 同步**：AIC 把每个 group 的 INT32 matmul 结果写入 workspace（或等价中间缓冲），AIV 读取后完成反量化 + add；具体同步原语与 buffer 布局由 agent 按性能选择，只要满足 §5.1 的跨核同步正确性与 group 间依赖即可。
- **参考 tiling**：kernel tiling/launch 体现 **AIC + AIV 混合执行**，按 `groupList` 切 group 的调度策略与空 group 处理；按 shape 自适应选 tile，避免过小 tile 导致 cube 利用率过低。
- **已知反模式（建议避开）**：epilogue 逐行用标量 `GetValue` 读 `scale1` / `scale2`、且每个向量 op 后串行 `PipeBarrier` 的写法，在大 shape 下被 fence 串行主导；优先**多行整块** `[m_i, N]` 向量处理。
- 建议在设计文档与 `trace.md` 记录：AIC/AIV 分工、按 `groupList` 切 group 的调度与空 group 处理、workspace 布局（INT32 partial 中间缓冲）、同步方式，便于性能复盘。

## 6. 精度要求

本算子精度判定遵循 [`../PRECISION_SPEC.md`](../PRECISION_SPEC.md)。通过条件与阈值参数定义在同目录 `proto.yaml` 的 `precision` 节点，以下仅说明本算子特定的取舍。

### 6.1 算子特定说明

- **`y` 阈值归属**：规则 `output_dtype`。输出为 float32 golden，重点验证 grouped matmul 后的 scale 反量化与 inplace add。
- **乘加顺序**：精度强约束见 §5.1，`partial * scale1[i] * scale2[i] + yRef`（以 golden 为准），不得预合。

## 7. 标准 Golden 代码

`golden.py` 按 `groupList`（cumsum，`groupListType=0`）切 token，对每个非空 group 执行 INT8 matmul（float32 累加表达 INT32 语义）、`* scale1 * scale2` 反量化并累加到 `yRef.clone()`，空 group 跳过。`y` 与 `yRef` 同形同 dtype，语义上对 `yRef` 做 inplace add（reference 中通过 `clone` 表达）。

## 8. 强制验收约束

1. 所有 `cases.yaml` / `cases.csv` 中的用例**精度必须全部达标**；不得只通过 `basic_case` 或部分 `general_case` 后停止。**正确性是硬门。**
2. 所有用例的**计算速度必须优于 torch 小算子拼接实现**；如任一用例性能未优于该基线，必须继续优化 AscendC tiling、workspace 流水或 vector 反量化/加法路径，直到性能达标或在 `trace.md` 中记录明确阻塞原因。**性能是软门**（不阻塞交付，但需在 trace 标注未达标 case 与原因）。
3. 性能耗时口径：来自 msprof `op_summary` 的 **duration-only**（`sum(base Task Duration) / asc custom Task Duration`），不得用 host wall-clock / `time.perf_counter` 自写计时。

## 9. 额外信息

### 9.1 测试资料对应关系

- `docs/aclnnQuantGroupedMatmulInplaceAdd.md`：T-C 公式、`groupListType` 和 `groupSize` 说明。
- `tests/assets/golden.py`：生态测试 golden 参考。

### 9.2 本 benchmark case 设计

`cases.yaml` / `cases.csv` 当前包含 **20 个正向 case**，1:1 对应，覆盖：

- **小规模（~6，smoke/edge）**：单 expert、双 expert 均分、单 token + 首组空、空中间 group + tail M、`E=16` many-experts 小 shape、多空 group 交错。
- **LLM（~14，MoE-prefill）**：`M ≤ 5120`、`K/N ≤ 5120`、`E ∈ {2, 4, 8, 16}`，含对齐 ramp、`K/N=5120` max-dim 角、ragged 非对齐 group list、含首/中空 expert 的大 shape、奇数 tail M、单 expert dense、small-M-big-KN 角。

所有 case 的 `group_list_values` 为长度 `E` 的 cumsum 非递减整型列表且末值 == M；`baseline_perf_us = t_hw_us = 0.0`，由评测阶段填充。

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


def quant_grouped_matmul_inplace_add(
    x1: torch.Tensor,
    x2: torch.Tensor,
    scale1: torch.Tensor,
    scale2: torch.Tensor,
    groupList: torch.Tensor,
    yRef: torch.Tensor,
    groupListType: int = 0,
    group_size=(1, 0, 0),
    variant: str = "TC_PERCHANNEL",
    group_list_values=None,
) -> torch.Tensor:
    """Torch golden for quant_grouped_matmul_inplace_add T-C per-channel path."""
    if group_list_values is not None:
        groupList = torch.tensor(group_list_values, dtype=torch.int64, device=x1.device)
    groups = _groups(groupList, groupListType)
    g, k, n = x2.shape
    if len(groups) != g or x1.shape[1] != k or yRef.shape != (x1.shape[0], n):
        raise ValueError("shape mismatch")
    y = yRef.to(torch.float32).clone()
    for idx, (start, end) in enumerate(groups):
        if end <= start:
            continue
        partial = x1[start:end, :].to(torch.float32) @ x2[idx].to(torch.float32)
        y[start:end, :] = y[start:end, :] + partial * scale1[idx].to(torch.float32).reshape(1, 1) * scale2[idx].to(torch.float32).reshape(1, n)
    return y
```
