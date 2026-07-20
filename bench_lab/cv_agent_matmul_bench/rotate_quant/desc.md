# RotateQuant 算子 API 描述

## 1. 算子简介

**算子特征**：
- 难度等级：L2（Transform）

`rotate_quant` 对输入张量 `x` 进行旋转变换，再执行对称动态量化。本 benchmark 选取 `float16/bfloat16 -> int8 + float32 scale` 路径，作为 **C->V 融合算子基准**：AIC（Cube）执行分块旋转矩阵乘并将中间结果写入 workspace，AIV（Vector）读取中间结果执行逐行动态量化（逐行 absmax -> scale -> 归一化 -> round -> clip -> int8 写回 + float32 scale 写回）；V2C / C2V 同步只用于 workspace / ping-pong 流控。该算子是一条典型的 **1C2V**（1 个 Cube core 配 2 个 Vector core）混合执行路径。

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

设 `x` 的形状为 `[m, n]`，`rotation` 的形状为 `[k, k]`（方阵）。旋转变换为按 `k` 分块的 batched matmul：

$$
Y = (x.\text{reshape}(m,\ n/k,\ k) @ \text{rotation}).\text{reshape}(m, n)
$$

对称动态量化为逐行（per-row / per-token）量化：

$$
s_i = \frac{\max_{j \in [0,\ n-1]} |Y_{i,j}|}{C_{\text{MAX}}}, \qquad
y_{i,j} = \mathrm{round}\!\left(\frac{Y_{i,j}}{s_i}\right),\ \text{clip}\ [-C_{\text{MAX}},\ C_{\text{MAX}}]
$$

`C_MAX` 在 int8 场景取 127，quint4x2 场景取 7。本 benchmark 固定 int8 路径（`C_MAX = 127`），CPU golden 在归一化后执行 `round` 和 `[-127, 127]` 裁剪。当某行 `amax = 0`（如全零输入行）时 `scale = 0`，golden 用 `torch.where(scale > 0, Y/scale, 0)` 把该行量化结果置零，避免除零。

### 2.1 计算步骤（与 golden 对齐，乘加 / 归一化顺序为精度强约束）

```python
Y = matmul(x.reshape(M, N // K, K), rotation).reshape(M, N)   # rotation matmul，AIC 完成
scale = Y.abs().amax(dim=1) / 127        # 逐行 absmax / 127，shape [M]，dtype float32
y = (Y / scale[:, None]).round().clamp(-127, 127).to(int8)    # 逐行动态量化，AIV 完成
```

- matmul 必须按 `float32` 累加：golden 先把 `x`、`rotation` 升到 `float32`，再做 `reshape(M, N//K, K) @ rotation`，回 reshape 到 `[M, N]`。
- `amax` 在量化前的 `float32` 旋转结果 `Y` 上沿最后一维（整行 N 个元素）求取；不得在分块边界处提前 round/clip。
- 输出 `scale` 为 `float32`，shape 严格为 `[M]`（一维），即每行一个标量。

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
| `x` | 输入 | `BFLOAT16`、`FLOAT16` | `ND` | 2D `[M, N]` | 待旋转量化的输入张量，支持非连续 Tensor，不支持空 Tensor |
| `rotation` | 输入 | `BFLOAT16`、`FLOAT16` | `ND` | 2D `[K, K]` | 旋转矩阵（方阵），dtype 必须与 `x` 相同，支持非连续 Tensor，不支持空 Tensor |
| `alpha` | 输入 | `float` | - | 标量 | 实际接口的 clamp 缩放系数；本 benchmark 固定为 `0.0`，表示不做 alpha 软裁剪（int8 饱和 clamp 仍执行） |
| `yOut` | 输出 | `INT8`、`INT32`、`FLOAT4_E2M1` | `ND` | 2D | 量化后的输出张量，需预先分配 |
| `scaleOut` | 输出 | `FLOAT32`、`FLOAT8_E8M0` | `ND` | 1D `[M]` | 动态量化计算出的逐行缩放系数，需预先分配 |

benchmark 抽象接口（golden 与本套件用例使用此签名）：

```python
rotate_quant(x, rotation, alpha=0.0, y_dtype="int8") -> (y, scale)
```

### 固定参数

- `alpha = 0.0`：不做 alpha 软裁剪（int8 饱和 clamp 仍执行）。golden 在 `alpha != 0.0` 时直接抛错。
- `y_dtype = "int8"`：固定 int8 输出路径。golden 在 `y_dtype != "int8"` 时直接抛错。

## 4. 约束说明

Atlas A2 / A3 产品约束，以及 golden.py 中强制校验的 shape 约束如下：

- `x` 的 shape 为 `(M, N)`（必须 2D），`rotation` 的 shape 为 `(K, K)`，且 `rotation` 必须是方阵（`rotation.shape[0] == rotation.shape[1]`）。
- `N` 必须是 `K` 的整数倍（`N % K == 0`），且 `N` 必须可以整除 8（`N % 8 == 0`）。
- `x` 和 `rotation` 的数据类型必须相同。
- `scaleOut` 的 shape 必须是 `(M)`（一维，float32）。
- `N` 的范围为 `[128, 16000]`，`K` 的范围为 `[16, 1024]`。
- `K <= N`（由 `N % K == 0` 与 `N >= 128 >= K_min` 共同保证；本套件所有 case 均满足 `K <= N`）。
- 当 `yOut` 为 `INT8` 时，shape 为 `(M, N)`；当 `yOut` 为 `INT32` 时，shape 为 `(M, N // 8)`。

本 benchmark 仅覆盖 `yOut=INT8`、`scaleOut=FLOAT32`、`x/rotation=BFLOAT16/FLOAT16`，并固定 `alpha=0.0`。

> **输入 / 输出张量形状关系（实现与用例必须严格遵守 golden 的断言）**
> - 输入 `x`: `[M, N]`；输入 `rotation`: `[K, K]`。
> - 旋转 matmul 把 `x` 视作 `[M, N/K, K]`，与 `[K, K]` 的 `rotation` 做 batched matmul，得 `[M, N/K, K]`，再 reshape 回 `[M, N]`；`N/K` 即「block 数」。
> - 输出 `y`: `[M, N]` int8；输出 `scale`: `[M]` float32（**这是一个 tuple 多输出：int8 量化输出 + 动态逐行 scale**）。

## 5. 实现约束与参考设计

> 本节分两层。**§5.1 硬约束**是正确性与合法性底线，必须遵守；**§5.2 参考设计与已知反模式**是非强制指导——给出一条已验证可行的实现路径与避坑提示，**鼓励在守住 §5.1 的前提下自行设计更优方案；当生成算子性能不足时，应主动超越本参考路径，而非照抄。**

### 5.1 硬约束（正确性 + 反作弊，必须遵守）

> 本层与 §2 计算语义同等约束力；AscendC 实现与调试前必须先读取并遵守。**禁止改变 golden 定义的算子数学语义**（包括 float32 matmul 累加、逐行 absmax、`/127` scale、round、`[-127,127]` clip、scale=0 行置零）。

目标语义与执行顺序（与 golden 一致）：

```text
Y      = matmul(x.reshape(M, N//K, K), rotation).reshape(M, N)   # float32 累加，AIC/Cube
amax   = abs(Y).amax(dim=-1)        # 逐行（整行 N 元素）最大绝对值，AIV/Vector
scale  = amax / 127                 # [M] float32，AIV
y      = round(Y / scale[:,None]).clamp(-127,127).to(int8)        # [M,N] int8，AIV；scale==0 行置零
```

- **真融合，禁退化**：必须生成**真正的 Cube + Vector 融合 AscendC kernel**，**禁止**退化为以下任一形态：纯 AIV（在向量侧用逐元素标量循环模拟矩阵乘）、纯 CPU、torch（在 `model_new_tilelang.py` / `model_new_ascendc.py` 中用 torch 算子做任何实际计算）、aclnn 高层组合算子（用现成高阶融合 API 拼装替代自定义 kernel）、Python fallback。
- **旋转 matmul 必须落 Cube**：`x.reshape(M, N//K, K) @ rotation` 的分块旋转矩阵乘必须在 **AIC/Cube** 侧用 AscendC Cube / Matmul / MMAD 原语完成，并按 **float32 累加**（与 golden 数学语义一致）；**禁止在 AIV/Vector 侧用逐元素标量循环模拟矩阵乘。**
- **动态量化必须片上向量化**：逐行 absmax、scale 计算、归一化、round、clip、int8 写回、float32 scale 写回，全部必须在片上 **AIV/Vector** 管线完成；**禁止**下沉到 torch / host / CPU / aclnn / Python 计算输出，**禁止**用 AIC 标量循环替代。
- **归一化 / 量化顺序必须与 golden 一致**：先在 float32 旋转结果 `Y` 上取整行 absmax，再 `/127` 得 scale，再 `Y/scale` 归一化后 round+clip；不得在 block 边界提前 round，也不得把 `amax` 在分块结果上分段近似求取。
- **`scale==0` 行置零**：当整行 `amax=0`（如全零输入行）时 `scale=0`，该行量化结果按 golden 置零，不得除零。
- **tuple 多输出与 dtype**：输出为 `(y, scale)` 二元 tuple——`y` 为 `[M,N]` int8，`scale` 为 `[M]` float32（一维）；必须同时正确支持 `float16` 与 `bfloat16` 两种输入 dtype，`scale` 输出固定为 `float32`。
- **shape 断言**：实现 / 用例必须满足 golden 强制断言（`x.dim()==2`、`rotation` 为方阵、`N % K == 0`、`N % 8 == 0`、`128 <= N <= 16000`、`16 <= K <= 1024`），否则直接抛错。
- **跨核同步正确性**：AIC→AIV 交接必须正确同步、保证跨核 GM 写对消费侧可见，不得出现数据竞争（否则结果错）；每个跨核 `SetFlag` 前必须保证对应 GM 写已对消费侧可见（排空写后再置位，铁律）；不得用局部 `PipeBarrier` 冒充跨 AIC/AIV 的可见性同步。
- kernel `__global__` 核函数名与 Host `_do` 入口名必须包含 `custom`（如 `rotate_quant_custom` / `rotate_quant_custom_do` 或 `rotate_quant_custom_<dtype>` / `..._do`）；不得生成不含 `custom` 的 profiling kernel 名。AscendC 热路径禁止标量 `GetValue/SetValue` 循环（少量边界 / 控制元数据除外）。
- 精度遵循 §6 / [`../PRECISION_SPEC.md`](../PRECISION_SPEC.md) 与 `proto.yaml` 的 `precision` 节点。不得为了消性能 / 容差内问题而修改算子数学语义。

### 5.2 参考设计与已知反模式（指导，非强制，鼓励超越）

> 以下是一条**已验证可行**的 C→V 实现路径与若干提示，作为起点与避坑参考，**不是必须照抄的配方**。在守住 §5.1 的前提下，鼓励 agent 探索更优实现；**性能不足时优先在此处突破。**

- **参考数据流（C→V，AIC + AIV 混合执行）**：

  ```text
  for each (M-tile, N) 工作块:
    AIC: 取 x tile + rotation，做分块旋转 matmul，得到 float32 旋转结果 Y_tile
    AIC: 把 Y_tile 写入 workspace / 中间缓冲（GM ring slot 或等价 ping-pong buffer）
    AIC: 通知 AIV（V2C / C2V 同步握手，保证 GM 写对 AIV 可见）
    AIV: 读取 Y_tile -> 逐行 absmax -> scale=amax/127 -> 归一化 -> round -> clip -> int8 写回 y
    AIV: 写回该 tile 覆盖行的 float32 scale
    AIV: 通知 AIC，workspace slot 可复用
  ```

  这是一种直接可行的切分：AIC 将分块 matmul 结果写入 workspace 或等价中间缓冲，AIV 读取中间结果完成量化。agent 可自行探索更优方案（如把量化融进 matmul epilogue、不同 tile/buffer 策略等）。注意逐行 absmax 需要整行 N 个元素，参考路径下应保证「同一行的全部 N 元素」对 AIV 可见后再做量化（按行 / 按 M-tile 交接，不要在不足整行时提前量化）。
- **参考同步 / workspace**：

  ```text
  for each tile:
    AIC 产出 float32 旋转结果 tile 写 workspace / GM ring slot
    AIC 排空 GM 写（PipeBarrier）后，CrossCoreSetFlag / V2C 通知 AIV
    AIV CrossCoreWaitFlag -> 消费 tile -> 逐行量化 -> 写 y / scale
    AIV 通知 AIC：slot 可复用（C2V 生命周期闭环）
  ```

  ring slot / workspace 维护明确的 C2V / V2C 生命周期；多 AIV lane 按 collective 语义同步。具体同步原语与 buffer 布局由 agent 按性能选择，只要满足 §5.1 的同步正确性即可。**Workspace 可用于**：AIC/AIV tile 交接的 float32 旋转结果 ring slot；ping-pong / queue style tile buffer；rotation / scale 的 UB staging。
- **参考 tiling**：Host tiling 体现 AIC + AIV 混合执行（**1C2V**：两个 AIV lane 各按 collective 分担一部分 M 行，不只让单 lane 推进全局进度）；按 shape（含大 M、大 N、`N/K` block 数）自适应选 M-tile / N 切分、used AIC/AIV 核数、ring slot 数，避免小 tile 造成核利用率过低；优先复用框架与历史参考实现的 tiling 思想。
- **已知反模式（建议避开）**：在 block 边界提前 round / 分段近似 amax；epilogue 逐元素标量写法、每个向量 op 后局部 `PipeBarrier` 的串行写法（大 shape 下被 fence 串行主导）；优先**整块向量处理**并使用 `T.copy`、`T.tile.*`、矩阵 / 向量原语等块级 / 向量化操作。
- 建议在设计文档与 `trace.md` 记录：AIC/AIV 分工、workspace 布局、同步方式，便于性能复盘。

设计检查清单（参考，便于自查是否守住 §5.1）：

- [ ] 旋转 matmul `x.reshape(M,N//K,K) @ rotation` 在 **AIC/Cube** 完成（用 Matmul/MMAD 原语，非向量侧逐元素模拟）？
- [ ] matmul 按 **float32 累加**，与 golden 数学语义一致？
- [ ] 动态量化（逐行 absmax / scale=amax/127 / 归一化 / round / clip / int8 写回 / float32 scale 写回）全部在 **AIV/Vector** 完成？
- [ ] 归一化 / round / clip 顺序与 golden 一致，未在 block 边界提前 round，未分段近似 amax？
- [ ] `scale==0`（整行 amax=0）行按 golden 置零，未除零？
- [ ] C→V 数据流：AIC 写 workspace、AIV 读 workspace，AIC/AIV 同步正确、slot 生命周期闭环？
- [ ] 1C2V：两个 AIV lane 按 collective 分担 M 行，未只让单 lane 推进？
- [ ] kernel 名含 `custom`，无 torch / aclnn 高层组合 / 纯 AIV / 纯 CPU / Python fallback 退化？
- [ ] 同时正确支持 `float16` 与 `bfloat16`，`scale` 输出为 `float32`、shape `[M]`？

## 6. 精度要求

本算子精度判定遵循 [`../PRECISION_SPEC.md`](../PRECISION_SPEC.md)。通过条件与阈值参数定义在同目录 `proto.yaml` 的 `precision` 节点，以下仅说明本算子特定的取舍。

### 6.1 算子特定说明

- **`scale` 阈值归属**：规则 `input_dtype_inherited`。`scale` 自身 dtype 为 FLOAT32，但其数值由 NPU 上 BF16/FP16 rotation matmul 推导，精度上限受输入 dtype 制约；直接按 FLOAT32 阈值 `2^-13` 会假阴性。具体阈值见 `proto.yaml.precision.outputs[scale].threshold_per_input_dtype`（BF16 -> `2^-7`、FP16 -> `2^-10`）。
- **`y` 阈值归属**：规则 `int8_three_tier`，采用默认参数（fatal=2 / tolerance=1 / bit_exact_ratio=0.99）。
- **`scale = 0` 边界**（全零输入或某行 amax=0）：由 SPEC §5 小值特殊处理覆盖，actual 为 fp32 噪声 `~1e-5` 时改走绝对误差判定，无算子专属逻辑。

## 7. 标准 Golden 代码

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
return y, scale.reshape(m).to(torch.float32)
```

golden 在入口处强制以下断言（实现 / 用例必须满足，否则直接抛错）：`y_dtype == "int8"`、`alpha == 0.0`、`x.dim() == 2`、`rotation` 为方阵、`N % K == 0`、`N % 8 == 0`、`128 <= N <= 16000`、`16 <= K <= 1024`。

## 8. 强制验收约束

1. 所有 `cases.yaml` / `cases.csv` 中的用例**精度必须全部达标**；不得只通过 basic / 部分 general 用例后停止。
2. 所有用例的**计算速度必须优于 torch 小算子拼接实现**（matmul + abs/amax + 除法 + round/clip + cast 的 eager 拼接基线）；如任一用例性能未优于该基线，必须继续优化 AscendC tiling、workspace 流水或 vector 量化路径，直到性能达标或在 `trace.md` 中记录明确阻塞原因。
3. 性能耗时口径以 benchmark 性能门脚本为准（duration-only），不得使用 host wall-clock 自写计时；正确性是硬门，性能是软门（NEEDS_OPTIMIZE 时触发优化迭代）。

## 9. 额外信息

### 9.1 测试资料对应关系

- `tests/ut/op_host/test_rotate_quant_tiling.cpp`：覆盖 BF16/FP16 输入、INT8 输出、`K=16/64/256/1024`、`N=128/256/1024` 等 tiling 场景，并包含 `N` 不能被 `K` 整除的失败用例。
- `tests/ut/op_kernel/test_rotate_quant.cpp`：包含 BF16+INT8 和 FP16+INT8 两条 kernel UT 路径。

### 9.2 本 benchmark case 设计（LLM-shape 刷新版）

`cases.yaml` / `cases.csv` 当前包含 **20 个正向 case**，按「少量小规模 + 大量 LLM 规模」组织，1:1 对应，均满足 `N % K == 0`、`N % 8 == 0`、`128 <= N <= 16000`、`16 <= K <= 1024`、`K <= N`，且**所有维度（M / N / K）均不超过 5120**：

- **6 个 SMALL（smoke / edge，maxdim < 512）**：`case 1~6`。覆盖 `K=16/32/64/128` 最小档、`N=128/256/384`、`M=1`（单行）、`M=3/5`（奇数 / 非对齐尾行）、`N=K*8`（多 block）、kernelUT-like（`M4xN256xK64`）等 tiling 边界。
- **14 个 LLM（realistic large，maxdim >= 512）**：`case 7~20`。采用真实大 shape（`M/N/K` 取 512/640/768/1024/1280/1536/2048/2560/3072/3840/4096/5120 等典型值），覆盖：
  - `N = K`（block=1，`case 9/15`）、`N = 2K`（block=2，`case 13`）、多 block（block=4/5/6/8，`case 7/8/10/12/16/17/18/19`）。
  - `maxdim = 5120` 上界场景：N 维上界（`case 11/16`）、M 维上界（`case 11/15`）。
  - 非对齐 / 尾 M 场景：`M=1000`（`case 18`）、`M=513`（奇数尾块，`case 20`）。
  - `bfloat16` 与 `float16` 交替，保持与原始套件相同的 dtype 列表与 attrs 结构。

设计取舍：维度上限收敛到 5120（原始套件的 `N=16000`、`N=12288` 等超大维度按本任务要求缩放到 `<= 5120`），既保留 LLM 真实大 shape 的 tiling 压力，又便于 CPU golden 在 agent 调试阶段保持可控耗时；`value_range` 沿用算子原始 `[-1, 1]`；`baseline_perf_us` / `t_hw_us` 统一置 `0.0`。

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

    x_fp32 = x.to(torch.float32)
    rot_fp32 = rotation.to(torch.float32)
    y_rot = torch.matmul(x_fp32.reshape(m, n // k, k), rot_fp32).reshape(m, n)

    c_max = 127.0
    max_abs = torch.abs(y_rot).amax(dim=-1, keepdim=True)
    scale = max_abs / c_max
    normalized = torch.where(scale > 0, y_rot / scale, torch.zeros_like(y_rot))
    y = torch.round(normalized).clamp(-c_max, c_max).to(torch.int8)
    return y, scale.reshape(m).to(torch.float32)
```
