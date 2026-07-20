# QuantBatchMatmulInplaceAdd (MX dynamic) 算子 API 描述

> 本文档自洽（self-contained）：把原 `desc.md` 与 `prompt.md` 的算子语义、实现契约、强制 CV 约束、验收约束合并为单一文档，作为本 benchmark 套件唯一权威说明。AscendC 实现与调试前必须先读取并遵守 §4 / §5。

## 1. 算子简介

**算子特征**：
- 难度等级：L3（Contraction）

`quant_batch_matmul_inplace_add` 对齐源码目录 `ops-nn/matmul/quant_batch_matmul_inplace_add`。本 benchmark 固定 **MX 动态量化 inplace-add 代表路径**：输入为浮点张量（FP16/BF16），V（Vector/AIV）侧按 K 维 block 动态求 scale 并做量化/反量化，C（Cube/AIC）侧完成 block matmul，最后把分块结果累加到 `yRef`，整体属于 **V→C→V kernel flow**。目录名中的 `_mx_dynamic` 用于明确本 benchmark 只覆盖 MX dynamic path；HiFloat8 T-T、其他 MX/CMCT 变体不纳入本目录。

属性 / 计算流（一句话总览）：

```text
y = yRef + sum_blocks( dequant(dynamic_quant(x1_block)) @ dequant(dynamic_quant(x2_block)) )
```

## 2. 算子定义

固定 `variant="MX_DYNAMIC"`、`transposeX1=True`、`transposeX2=False`、`groupSize=32`。因此 `x1` 物理形状为 `[K, M]`、逻辑视图为 `[M, K]`；`x2` 形状为 `[K, N]`；`yRef`/`y` 形状为 `[M, N]`。K 维按 `groupSize=32` 分块，**最后一块允许 `< 32` 的尾块**；`scale` 不作为外部输入，由算子内部按 K block 动态计算。

逐 K-block 语义（与 `golden.py` 完全一致）：

```python
a = x1.t().to(float32)           # [M, K]   (transposeX1=True)
b = x2.to(float32)               # [K, N]   (transposeX2=False)
out = yRef.to(float32).clone()   # [M, N]   (inplace-add 初值；输出与其 shape/dtype 一致)
eps = finfo(float32).tiny
qmax = 127.0
for start in range(0, K, 32):                                         # groupSize=32
    end = min(start + 32, K)                                          # 尾块 < 32 合法
    a_blk = a[:, start:end]                                           # [M, gK]
    b_blk = b[start:end, :]                                           # [gK, N]
    s1 = a_blk.abs().amax(dim=1, keepdim=True).clamp_min(eps) / qmax  # [M, 1]  per-row(token)
    s2 = b_blk.abs().amax(dim=0, keepdim=True).clamp_min(eps) / qmax  # [1, N]  per-col(channel)
    a_dq = (a_blk / s1).round().clamp(-qmax, qmax) * s1               # 反量化后的 a block
    b_dq = (b_blk / s2).round().clamp(-qmax, qmax) * s2               # 反量化后的 b block
    out = out + a_dq @ b_dq                                           # block matmul + inplace-add
```

要点：
- `s1` 形状 `[M, 1]`（对 `x1` 在 K-block 内**按行/dim=K** 求 absmax）；`s2` 形状 `[1, N]`（对 `x2` 在 K-block 内**按列/dim=K** 求 absmax）。两者都做 `clamp_min(eps)` 下限保护并除以 `qmax=127`。
- 每个 block 完成 `round → clip(-127,127) → 乘回 scale` 得到反量化 `a_dq`/`b_dq`，再 `a_dq @ b_dq` 累加。
- `yRef` 既是输入也是 inplace-add 初值；**`yRef.shape` 必须等于 matmul 输出 `(M, N)`，否则 golden 抛 `shape mismatch`**。
- 累加与输出严格保持 `float32`。

## 3. 接口规范

```python
quant_batch_matmul_inplace_add(x1, x2, yRef, variant="MX_DYNAMIC", transposeX1=True, transposeX2=False, groupSize=32) -> y
```

| 参数 | 输入/输出 | dtype | shape | 说明 |
|------|-----------|-------|-------|------|
| `x1` | 输入 | `FLOAT16/BFLOAT16` | `[K, M]` | 左矩阵，`transposeX1=True` 转置后逻辑 `[M, K]` 参与计算 |
| `x2` | 输入 | `FLOAT16/BFLOAT16` | `[K, N]` | 右矩阵，`transposeX2=False` 原样参与 |
| `yRef` | 输入/输出 | `FLOAT32` | `[M, N]` | inplace-add 初始值，必须等于 matmul 输出形状 |
| `y` | 输出 | `FLOAT32` | `[M, N]` | 累加后的结果（与 `yRef` 同 shape/dtype） |

固定参数（attrs，结构与每个 case 完全一致）：

| attr | type | 固定值 | 含义 |
|------|------|--------|------|
| `variant` | str | `MX_DYNAMIC` | 只支持 MX 动态量化路径；其他值 golden 抛错 |
| `transposeX1` | bool | `true` | `x1` 物理 `[K,M]`、逻辑 `[M,K]` |
| `transposeX2` | bool | `false` | `x2` 形状 `[K,N]` |
| `groupSize` | int | `32` | K 维量化分块大小，尾块 `<32` 合法 |

### 3.1 输入张量与 shape 关系（实现/用例设计必须遵守）

本算子**只有 3 个输入张量**（`scale` 为内部动态量化产物，不是输入）。每个 case 的 `input_shape` 固定为：

```text
input_shape = [ [K, M],   # x1   (physical, transposeX1=True)
                [K, N],   # x2   (physical, transposeX2=False)
                [M, N] ]  # yRef (== matmul output (M, N))
dtype       = [ x_dtype, x_dtype, "float32" ]   # x1/x2 同 dtype (FP16 或 BF16)，yRef 恒 float32
```

派生关系（校验器会跑 golden 并据此核对，写错 yRef 或 scale-block 形状会被当场抓出）：
- `x1[0] == x2[0] == K`（共享 contraction 维）；`x1[1] == M`；`x2[1] == N`。
- `yRef == [M, N] == [x1[1], x2[1]]`（inplace-add accumulator，必须等于输出 `(M, N)`）。
- 内部动态量化 scale 形状由 **MX block size = `groupSize` = 32** 与矩阵维联合决定：K 维按 32 切成 `ceil(K/32)` 个 block，逐 block 求 `s1:[M,1]` / `s2:[1,N]`（即每个 K-block 一组 per-row / per-col scale；scale 不出现在 `input_shape` 里）。

## 4. 实现约束与参考设计（MX 动态量化 + block matmul + inplace-add）

> 本节分两层。**§4.1 硬约束**是正确性与合法性底线，必须遵守；**§4.2 参考设计与已知反模式**是非强制指导——给出一条已验证可行的实现路径与避坑提示，**鼓励在守住 §4.1 的前提下自行设计更优方案；当生成算子性能不足时，应主动超越本参考路径，而非照抄。**

### 4.1 硬约束（正确性 + 反作弊，必须遵守）

> 以下与 §2 计算语义同等约束力，关乎正确性与合法性，**一个都不能省**。

- **数学语义与 MX 动态量化语义**：与 §2 / `golden.py` 完全一致——逐 K-block（`groupSize=32`，尾块 `<32` 合法）求 per-K-block absmax：对 `x1` 在 block 内**按行（dim=K）** 得 `s1=[M,1]`、对 `x2` 在 block 内**按列（dim=K）** 得 `s2=[1,N]`，两者均 `clamp_min(eps)`（`eps=finfo(float32).tiny`）后 `/qmax`（`qmax=127`）；每 block 做 `round → clip(-127,127) → 乘回 scale` 得反量化 `a_dq`/`b_dq`。
- **乘加顺序**：**按 block 累加 `a_dq @ b_dq` 后再 `+yRef`**，与 golden 一致；不得把 scale 预合、不得改变跨 block 的累加顺序（会改变浮点舍入，产生精度残差）。`yRef` 既是输入也是 inplace-add 初值。
- **dtype / 精度**：正确支持 `float16` 与 `bfloat16` 输入；**block 累加与最终输出严格保持 `float32`**。
- **shape 断言**：`yRef.shape == [M, N] == [x1[1], x2[1]]`，否则 golden 抛 `shape mismatch`（详见 §2 / §3.1）。
- **真融合，禁退化**：必须生成真正的 Cube + Vector 融合 AscendC kernel；禁止退化为纯 AIV、纯 CPU、torch（`model_new_tilelang.py` / `model_new_ascendc.py` 中不得用 torch 算子做任何实际计算）、aclnn 高层组合算子、Python fallback。
- **block matmul 必须落 Cube**：`a_dq @ b_dq` 的 block matmul 必须由 AIC/Cube 用 AscendC Cube / MatMul / MMAD 原语（或本工作区历史 cube matmul 模板）完成；**禁止在 AIV 侧用逐元素循环模拟矩阵乘**。
- **MX 量化与 inplace-add 必须片上向量化（AIV）**：per-block absmax / scale / `round` / `clip(-127,127)` / dequant，以及 inplace-add（每 block `out += a_dq@b_dq` 与最终 `+yRef`）必须在片上 AIV/Vector 完成；**禁止**下沉到 torch / host / CPU / aclnn / Python 计算输出，**禁止在 AIC 侧用标量循环模拟 absmax / round / clip**。
- **跨核同步正确性**：AIV↔AIC 交接必须正确同步、保证跨核数据可见，K-block 循环依赖必须正确处理，不得出现数据竞争（否则结果错）；不得用局部 `PipeBarrier` 冒充跨核可见性同步。
- kernel `__global__` 与 Host `_do` 入口名必须含 `custom`（如 `<op_name>_custom` / `<op_name>_custom_<dtype>`）；AscendC 热路径禁止标量逐元素循环（少量边界 / 控制元数据除外）。
- 精度遵循 §6 / `benchmark/PRECISION_SPEC.md` 与 `proto.yaml` 的 `precision` 节点。

### 4.2 参考设计与已知反模式（指导，非强制，鼓励超越）

> 以下是一条**已验证可行**的 V→C→V 实现路径与若干提示，作为起点与避坑参考，**不是必须照抄的配方**。在守住 §4.1 的前提下，鼓励 agent 探索更优实现；**性能不足时优先在此处突破。**

- **参考数据流（V→C→V）**：一种可行切分为三段编排——
  1. **AIV 第一阶段（量化/反量化）**：对每个 K block 计算 `x1`/`x2` 的逐行/逐列 absmax 与 scale，完成 `round/clip(-127,127)/dequant`，把反量化后的中间张量写入 **workspace**。
  2. **AIC 阶段（block matmul）**：对每个 K block 在 workspace 上执行 `a_dq @ b_dq` 的 block matmul，把分块结果累加到 **`float32` 累加 workspace**。
  3. **AIV 第二阶段（inplace-add 收尾）**：将 `float32` 累加 workspace 与 `yRef` 相加，输出最终 `float32` 张量 `y`。

  这只是一种直接可行的切分；agent 可自行探索更优方案（如把 dequant 融进 matmul epilogue、不同 tile/buffer 与 1C2V 分工等）。
- **参考 workspace 布局**：为反量化中间张量与 `float32` 累加张量各划独立 workspace 区。具体布局由 agent 按性能选择，只要满足 §4.1 的累加/输出 `float32` 与同步正确性即可。
- **参考同步原语**：AIV→AIC、AIC→AIV 用跨核 flag（如 `CrossCoreSetFlag` / 对应 `Wait`）交接，写后排空 GM 再置 flag；K-block 循环依赖按需串接。具体同步原语选择由 agent 决定，只要满足 §4.1 的跨核可见性即可。
- **参考 tiling**：kernel tiling/launch 体现 AIC + AIV 混合执行（如 1C2V），按 shape 自适应选 tile；避免过小 tile 导致 cube 利用率过低。
- **已知反模式（建议避开）**：热路径上逐行用标量读 per-row/per-col scale、且每个向量 op 后 `PipeBarrier` 的串行写法，在大 shape 下被 fence 串行主导；优先**整块**向量处理。
- 建议在设计文档与 `trace.md` 记录：AIC/AIV 分工与 K-block 流水、workspace 布局（反量化中间张量、`float32` 累加张量）、同步方式（AIV→AIC、AIC→AIV）与 K-block 循环依赖处理，便于性能复盘。

## 5. 强制 CV 无退化约束（反作弊）

1. 必须生成**真正的 Cube + Vector 融合 AscendC kernel**，**禁止退化为**：
   - 纯 AIV
   - 纯 CPU
   - torch（`model_new_tilelang.py` / `model_new_ascendc.py` 中禁止使用 torch 算子做任何实际计算）
   - aclnn 高层组合算子
   - Python fallback
2. AscendC 自定义 kernel 的 `__global__` 核函数名和 Host `_do` 入口名**必须包含 `custom`**（如 `<op_name>_custom` / `<op_name>_custom_<dtype>`）；不得生成不含 `custom` 的 profiling kernel 名。
3. 禁止标量逐元素写法：必须使用 `T.copy`、`T.tile.*`、矩阵/向量原语等块级或向量化操作（量化/反量化走 Vector 原语，matmul 走 Cube 原语）。
4. MX 动态量化、block matmul、inplace-add 三段分工（见 §4.1）不得互相替代或塌缩到单一引擎。

## 6. 精度要求

本算子精度判定遵循 Ops 精度规范（`benchmark/PRECISION_SPEC.md`）。通过条件与阈值参数定义在同目录 `proto.yaml` 的 `precision` 节点，以下仅说明本算子特定的取舍。

### 6.1 算子特定说明

- **`y` 阈值归属**：规则 `input_dtype_inherited`。输出 dtype 为 `float32`，但真实精度上限由 **FP16/BF16 输入**经动态量化 round-trip 和 inplace-add 决定，不得按 FP32 阈值判定。
- **`threshold_per_input_dtype`**（见 `proto.yaml`）：`bfloat16: 2^-7`、`float16: 2^-10`；多输入取最严格阈值，`mare_multiplier` 默认 `10`。
- **误差来源**：每个 K-block 的 `round/clip` 量化边界 + 跨 block 的 `float32` 累加顺序；`yRef` 非零累加会抬高输出量级，相对误差（MERE/MARE）按非小值元素统计，近 0 元素走自适应绝对容差（`small_value_threshold = T·s`，`absolute_tolerance = m·T·s`，`s = sqrt(mean(golden^2))`）。
- **边界**：单行 `M=1`、小 N、非 2 的幂维度、K 尾块 `<32`、`yRef` 非零初值均在用例覆盖范围内；无需在 `proto.yaml` 覆盖小值兜底，走规范自适应默认即可。

## 7. 标准 Golden 代码

`golden.py` 按 K 维 32 分块，对 `x1`（转置为 `[M,K]`）、`x2`（`[K,N]`）动态量化、反量化、做 block matmul，并把每块结果累加到 `yRef.to(float32).clone()`，返回 `float32` 的 `y`。**禁止修改 golden 的数学语义**；精度规则只用于验收误差，不得作为修改算子数学语义的理由。

## 8. 验收约束

1. **正确性是硬门**：所有 `cases.yaml` / `cases.csv` 中的用例**精度必须全部达标**；不得只通过 `basic_case` 或部分 `general_case` 后停止。
2. **性能门（软门）**：所有用例的计算速度必须优于 torch 小算子拼接（eager / torch_npu）基线；性能口径采用 msprof **duration-only**（`sum(base Task Duration) / asc custom Task Duration`），不得用 host wall-clock。任一用例未优于基线时，必须继续优化 AscendC tiling、workspace 流水或 vector 量化路径，直到达标或在 `trace.md` 记录明确阻塞原因。性能是软门、不阻塞交付，但正确性始终是硬门。
3. 每次执行验证脚本后，必须把 `PASS / FAIL / TIMEOUT` 事件写入 `current_task/knowledge_inbox/`；全量验证失败时调用 debugger 子流程继续修复（连续 5 轮无改进或累计超 30 分钟则终止并在 `trace.md` 记录阻塞原因、最后失败用例与最近一次 diff 摘要）。

## 9. 额外信息

### 9.1 测试资料对应关系
- `docs/aclnnQuantBatchMatmulInplaceAdd.md`：MX 量化、`groupSize` 和 inplace add 公式。
- `op_kernel/arch35/qbmmia_mx_basic_api_cmct.h`：MX basic API 路径。
- `examples/arch35/test_aclnn_quant_batch_matmul_inplace_add_mxfp8.cpp`：MXFP8 样例。

### 9.2 本 benchmark case 设计
`cases.yaml`/`cases.csv` 共 **20 个正向 case**，1:1 对应，覆盖 FP16/BF16 交替、`yRef` 非零累加、非 2 的幂维度与 K 尾块：
- **少量小用例（6 个，smoke/edge）**：维度均 `<512`，含最小 `M4K64N32`、方块 tile、`M1` 单行、`M2` 极小 M、非 2 的幂 `K96/N48` 触发 K 尾块、小 M 宽 N。
- **大量 LLM 用例（14 个）**：把 matmul 维度（M/K/N）放大到**真实 LLM 规模但绝不超过 5120**，覆盖 decode 小 M（16/32/64/128）到 prefill 大 M（512/1024/2048/3072/4096），hidden/FFN 维取 `1024/1536/2048/2304/2560/3072/4096/4608/5120`，含注意力投影、FFN up/down、方块 GEMM 与非 2 的幂维度；最大维度恰为 5120。
- 所有 case：`value_range=[-2, 2]`，`baseline_perf_us=0.0`，`t_hw_us=0.0`。

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


def quant_batch_matmul_inplace_add(
    x1: torch.Tensor,
    x2: torch.Tensor,
    yRef: torch.Tensor,
    variant: str = "MX_DYNAMIC",
    transposeX1: bool = True,
    transposeX2: bool = False,
    groupSize: int = 32,
) -> torch.Tensor:
    """Torch golden for quant_batch_matmul_inplace_add MX dynamic path."""
    if variant != "MX_DYNAMIC":
        raise ValueError("This benchmark fixes variant=MX_DYNAMIC")
    a = x1.t() if transposeX1 else x1
    b = x2.t() if transposeX2 else x2
    a = a.to(torch.float32)
    b = b.to(torch.float32)
    m, k = a.shape
    k2, n = b.shape
    if k != k2 or yRef.shape != (m, n):
        raise ValueError("shape mismatch")
    eps = torch.finfo(torch.float32).tiny
    qmax = 127.0
    out = yRef.to(torch.float32).clone()
    for start in range(0, k, int(groupSize)):
        end = min(start + int(groupSize), k)
        a_blk = a[:, start:end]
        b_blk = b[start:end, :]
        s1 = a_blk.abs().amax(dim=1, keepdim=True).clamp_min(eps) / qmax
        s2 = b_blk.abs().amax(dim=0, keepdim=True).clamp_min(eps) / qmax
        a_dq = torch.round(a_blk / s1).clamp(-qmax, qmax) * s1
        b_dq = torch.round(b_blk / s2).clamp(-qmax, qmax) * s2
        out = out + a_dq @ b_dq
    return out
```
