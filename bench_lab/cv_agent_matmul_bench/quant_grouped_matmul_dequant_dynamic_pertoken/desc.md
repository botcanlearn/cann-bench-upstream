# QuantGroupedMatmulDequant 算子 API 描述（动态 per-token 路径）

> 本文为 `quant_grouped_matmul_dequant_dynamic_pertoken` 的自洽参考文档，融合原 `desc.md` 的算子语义与原 `prompt.md` 的 V→C→V 实现契约、强制 CV 无退化约束与验收约束。AscendC 生成与调试前必须先读取并遵守本文全部条款。

## 1. 算子简介

**算子特征**：
- 难度等级：L3（Contraction）

`quant_grouped_matmul_dequant` 对齐源码目录 `ops-transformer/gmm/quant_grouped_matmul_dequant`。本 benchmark **固定动态 per-token 量化代表路径**，属于典型的 **V→C→V（AIV→AIC→AIV）grouped MoE 融合算子**：

1. **V 侧（前）动态 per-token 量化**：对浮点激活 `x` 按 token（行）动态计算 `xScale` 并量化到 INT8（概念上的 `xq`）。
2. **C 侧 grouped INT8 matmul**：按 `groupList` 切分 token，对每个非空 group 执行 `xq[start:end] @ quantized_weight[i]`，INT8 输入、INT32/FLOAT32 累加。
3. **V 侧（后）反量化 + bias**：按 per-channel `weightScale[i]` 与 per-token 内部 `xScale` 反量化，加 `bias[i]`，FLOAT32 写回 `out`。

这是一个 **grouped / MoE 算子**：`x` 的 M 行被 `groupList`（cumsum 形式）切成 E 个专家段，每段与各自的权重相乘。

## 2. 算子定义

```text
eps    = finfo(float32).tiny
xScale = row_max(abs(x)) / 127                         # [M]，逐行 absmax，含 clamp_min(eps)
xq     = round(x / xScale[:, None]).clamp(-127, 127)   # [M, K]，概念上为 int8
for group i, (start, end) in groups_from_cumsum(groupList):
    if end <= start:        # 空 group 必须跳过，不写输出
        continue
    mm  = xq[start:end] @ quantized_weight[i]           # [end-start, N]，int8 -> int32/float32
    out[start:end] = mm * weightScale[i].reshape(1, N) \
                        * xScale[start:end].reshape(-1, 1) \
                        + bias[i].reshape(1, N)
```

`groups_from_cumsum`（`groupListType=0`）：`starts = [0] + groupList[:-1]`，`ends = groupList`。即 `groupList` 为**累积（cumsum）边界**，相邻值相等表示空 group。

## 3. 接口规范

```python
quant_grouped_matmul_dequant(
    x, quantized_weight, weightScale, groupList, bias,
    quant_mode="pertoken", transposeWeight=false, groupListType=0,
) -> out
```

| 参数 | 输入/输出 | dtype | shape | 说明 |
|------|-----------|-------|-------|------|
| `x` | 输入 | `FLOAT16` / `BFLOAT16` | `[M, K]` | 未量化激活 |
| `quantized_weight` | 输入 | `INT8` | `[G, K, N]` | per-group（per-expert）量化权重 |
| `weightScale` | 输入 | `FLOAT32` | `[G, N]` | per-channel 权重 scale |
| `groupList` | 输入 | `INT64` | `[G]` | cumsum 分组边界（`groupListType=0`） |
| `bias` | 输入 | `FLOAT32` | `[G, N]` | per-group bias |
| `out` | 输出 | `FLOAT32` | `[M, N]` | 反量化输出 |

其中 `G == E`（专家数）。`input_shape` 顺序固定为
`[x[M,K], quantized_weight[E,K,N], weightScale[E,N], groupList[E], bias[E,N]]`。

## 4. 约束说明（语义 + 固定参数）

### 4.1 固定参数（本 benchmark 不可变）

- `quant_mode = "pertoken"`（动态 per-token 量化）。
- `xScaleOptional = None`（**xScale 由 kernel 内部动态计算**，不从外部传入）。
- `transposeWeight = False`。
- `groupListType = 0`（`groupList` 为 cumsum 形式，**最后一个元素必须等于 `M`**）。

### 4.2 grouped / groupList 语义

- `group_list_values` 是**累积非递减 int 列表**，长度为 `E`，是每专家 token 数的 cumsum，**末值严格等于 `M`**。
  - 校验器 FAIL `groupList last value must equal M` 即 cumsum 未到达 `M`，需修正。
- **允许空 group**：相邻 cumsum 值相等（含前导 `0`、中间重复、末尾重复）。空 group 必须**跳过 cube 计算与 vector 写回**，不写 `out`。
- 形状自洽：`x.shape[1] == K`，`quantized_weight.shape == [E, K, N]`，`len(groupList) == E`，`weightScale.shape == bias.shape == [E, N]`。

### 4.3 不纳入本目录的路径

- `smoothScale` 路径。
- `weightScale=INT64 + xScale=FP16 + pertensor` 静态量化路径（即外部传入 `xScaleOptional` 的静态 per-tensor 路径）。

## 5. 实现约束与参考设计

> 本节分两层。**§5.1 硬约束**是正确性与合法性底线，必须遵守；**§5.2 参考设计与已知反模式**是非强制指导——给出一条已验证可行的实现路径与避坑提示，**鼓励在守住 §5.1 的前提下自行设计更优方案；当生成算子性能不足时，应主动超越本参考路径，而非照抄。**

### 5.1 硬约束（正确性 + 反作弊，必须遵守）

- **golden 数学语义**：实现必须等价于 §2 计算语义——动态 per-token 量化（逐行 absmax 含 `clamp_min(eps)`、`eps = finfo(float32).tiny`、`/127`、`round`、`clamp(-127, 127)`）；grouped INT8 matmul 按 `groupList`（`groupListType=0` cumsum，末值严格等于 `M`）切分，**空 group（相邻 cumsum 相等）必须跳过 cube 计算与 vector 写回，不写 `out`**；dequant + bias。
- **乘加顺序与语义**：保持 golden 顺序 `(mm * weightScale[i]) * x_scale[row] + bias[i]`，**不得**把 `weightScale * x_scale` 或 `scale * bias` 预先合成后再乘/加，以免改变浮点舍入顺序、与 golden 产生精度残差。
- **dtype / shape 自洽**：正确支持 `float16` 与 `bfloat16` 输入；`x.shape[1] == K`，`quantized_weight.shape == [E, K, N]`，`len(groupList) == E`，`weightScale.shape == bias.shape == [E, N]`，`out` 为 FLOAT32 `[M, N]`。
- **真融合，禁退化**：必须生成**真正的 Cube + Vector 融合 AscendC kernel（V→C→V）**；**禁止**退化为纯 AIV、纯 CPU、torch（任何一步最终值）、aclnn 高层组合算子、Python fallback。
- **grouped INT8 matmul 必须落 Cube**：每个非空 group 的 `xq[start:end] @ quantized_weight[i]`（int8 in，int32/float32 累加）必须由 AIC/Cube 用 AscendC Cube / MatMul / MMAD 原语完成；**禁止在 AIV 侧用逐元素循环模拟矩阵乘**，**禁止用 AIC 标量循环模拟**。
- **动态量化与 dequant 必须片上向量化**：逐行 absmax / `x_scale` / round-clamp 到 int8、读 cube 中间结果、`mm * weightScale[i] * x_scale[row]` 反量化、加 bias、FLOAT32 写回，必须在片上 AIV/Vector 完成；**禁止**下沉到 torch / host / CPU / aclnn / Python 计算上述任何一步的最终值。
- **跨核同步正确性**：AIV↔AIC 交接（含 group 间依赖）必须正确同步、保证跨核数据可见，不得出现数据竞争（否则结果错）；不得用局部 `PipeBarrier` 冒充跨核可见性同步。
- kernel `__global__` 核函数名与 Host `_do` 入口名必须含 `custom`；kernel tiling/launch 必须体现 **AIC + AIV 混合执行**；AscendC 热路径禁止标量 `GetValue/SetValue` 循环（少量边界 / 控制元数据除外）。
- 精度遵循 §6 / [`../PRECISION_SPEC.md`](../PRECISION_SPEC.md) 与 `proto.yaml` 的 `precision` 节点。

### 5.2 参考设计与已知反模式（指导，非强制，鼓励超越）

> 以下是一条**已验证可行**的 V→C→V 实现路径与若干提示，作为起点与避坑参考，**不是必须照抄的配方**。在守住 §5.1 的前提下，鼓励 agent 探索更优实现；**性能不足时优先在此处突破。**

- **参考数据流（V→C→V 三段 AIC/AIV 分工）**：经 workspace 串接的一种直接可行切分：
  1. **第一段 AIV（量化）**：完成 per-token 动态量化，输出 `xq`（int8）与 `x_scale` 到 workspace。
  2. **C 侧 AIC（grouped matmul）**：按 `groupList` 切分，对每个非空 group 执行 INT8 grouped matmul，int32/float32 中间结果写入 workspace。分组遍历可在 host tiling 或 kernel 内根据 `groupList` 派发。
  3. **第二段 AIV（反量化 + bias）**：读取 matmul 中间结果，应用 `mm * weightScale[i] * x_scale[row]` 并加 `bias[i]`，FLOAT32 写回 `out`。
  agent 可自行探索更优方案（如把量化/反量化融进 matmul 前后处理、不同 tile/buffer 策略、1C2V 分工等）。
- **参考同步 / workspace 布局**：workspace 含 `xq`、`x_scale`、grouped matmul 中间结果三类张量；AIC 每次跨核置位前用 `PipeBarrier<PIPE_ALL>` 排空 GM 写，ring slot 维护明确 C2V/V2C 生命周期。具体同步原语与 buffer 布局由 agent 按性能选择，只要满足 §5.1 的同步正确性即可。
- **参考 tiling**：AIC + AIV 混合执行（1C2V），按 shape 与 group 切分自适应选 tile；避免过小 tile 导致 cube 利用率过低，注意 ragged / 空 group 下的负载均衡。
- **已知反模式（建议避开）**：epilogue 逐行用标量 `GetValue` 读 per-token `x_scale`、且每个向量 op 后 `PipeBarrier<PIPE_V>` 的串行写法，在大 shape 下被 fence 串行主导；优先**多行整块** `[validM, N]` 向量处理。
- 建议在设计文档与 `trace.md` 记录：V→C→V 三段 AIC/AIV 分工、workspace 布局（`xq` / `x_scale` / grouped matmul 中间结果）、按 `groupList` 的 group 切分与空 group 处理、AIV↔AIC 同步方式（含 group 间依赖），便于性能复盘。

## 6. 精度要求

本算子精度判定遵循 `../PRECISION_SPEC.md`。通过条件与阈值参数定义在同目录 `proto.yaml` 的 `precision` 节点，以下仅说明本算子特定取舍。

### 6.1 算子特定说明

- **`out` 阈值归属**：规则 `input_dtype_inherited`。输入为 FP16/BF16，输出反量化误差受动态量化 round-trip 影响。
- 阈值（`proto.yaml`）：`bfloat16 → 2^-7`，`float16 → 2^-10`。

## 7. 强制验收约束

1. **正确性是硬门**：`cases.yaml` / `cases.csv` 中**所有用例精度必须全部达标**；不得只通过 `basic_case` 或部分 `general_case` 后停止。
2. **性能为软门**：所有用例计算速度应优于 torch 小算子拼接基线；逐 case 加速比由 msprof（`op_summary_*.csv`）的 duration-only 口径裁决，不得使用 host wall-clock。未达标时继续优化 AscendC tiling、workspace 流水或 vector 量化/反量化路径，或在 `trace.md` 记录明确阻塞原因。
3. **空 group** 必须正确处理（跳过 cube 计算与 vector 写回）。

## 8. 标准 Golden 代码

`golden.py` 内部计算 per-token `xScale`，再对每个 group 执行 INT8 matmul 与反量化（`(mm * weightScale * xScale) + bias`），空 group 跳过。`golden.py` 为精度基准，**禁止修改其数学语义**。

## 9. 额外信息

### 9.1 测试资料对应关系

- `docs/aclnnQuantGroupedMatmulDequant.md`：动态 per-token 量化、`xScaleOptional` 为空时的公式。
- `op_kernel/quant_grouped_matmul_dequant_normal.h`：普通 grouped dequant 路径。

### 9.2 本 benchmark case 设计

`cases.yaml` 包含 **20 个正向 case**，采用「少量 small + 大量 LLM」结构，覆盖动态 per-token 量化、xScaleOptional=None 路径、grouped/MoE 切分与空 group：

- **6 个 small（smoke / edge）**：单专家、双专家均分、tail-M（非 2 幂 M）、空首专家（前导 0 group）、多专家小 shape（E16）、多空/重复专家（M==G 边界、单 token 尾段）。
- **14 个 LLM（MoE-prefill）**：`M ≤ 5120`，`K/N ≤ 5120`，`E ∈ {2, 4, 8, 16}`，含 ragged（非 16 对齐）group list、LLM 规模下的空/单 token 专家、宽 K/N（4096/5120）maxDim 锚点、长 M（5120）maxM 锚点。

所有 case 固定 `groupListType=0 / quant_mode=pertoken / transposeWeight=false / xScaleOptional=null`，`x` dtype 在 `float16` / `bfloat16` 间交替，`value_range=[-2, 2]`，`group_list_values` 末值严格等于各自的 `M`。

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
    quantized_weight: torch.Tensor,
    weightScale: torch.Tensor,
    groupList: torch.Tensor,
    bias: torch.Tensor,
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
    return [x, quantized_weight, weightScale, groupList, bias]


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


def quant_grouped_matmul_dequant(
    x: torch.Tensor,
    quantized_weight: torch.Tensor,
    weightScale: torch.Tensor,
    groupList: torch.Tensor,
    bias: torch.Tensor,
    quant_mode: str = "pertoken",
    xScaleOptional=None,
    transposeWeight: bool = False,
    groupListType: int = 0,
    group_list_values=None,
) -> torch.Tensor:
    """Torch golden for quant_grouped_matmul_dequant dynamic per-token path."""
    if quant_mode != "pertoken" or xScaleOptional is not None:
        raise ValueError("This benchmark fixes dynamic per-token path with xScaleOptional=None")
    if group_list_values is not None:
        groupList = torch.tensor(group_list_values, dtype=torch.int64, device=x.device)
    if transposeWeight:
        quantized_weight = quantized_weight.transpose(-2, -1)
    groups = _groups(groupList, groupListType)
    g, k, n = quantized_weight.shape
    if len(groups) != g or x.shape[1] != k:
        raise ValueError("shape mismatch")
    eps = torch.finfo(torch.float32).tiny
    x_scale = x.to(torch.float32).abs().amax(dim=1).clamp_min(eps) / 127.0
    xq = torch.round(x.to(torch.float32) / x_scale.reshape(-1, 1)).clamp(-127, 127)
    out = torch.zeros(x.shape[0], n, dtype=torch.float32, device=x.device)
    for idx, (start, end) in enumerate(groups):
        if end <= start:
            continue
        mm = xq[start:end, :] @ quantized_weight[idx].to(torch.float32)
        out[start:end, :] = mm * weightScale[idx].to(torch.float32).reshape(1, n) * x_scale[start:end].reshape(-1, 1)
        out[start:end, :] = out[start:end, :] + bias[idx].to(torch.float32).reshape(1, n)
    return out
```
