# AllGatherMatmulV2 算子 API 描述

## 1. 算子简介

AllGatherMatmulV2 是一个MC2通算融合算子，用于将 AllGather 集合通信与量化矩阵乘融合执行。该算子是 `aclnnAllGatherMatmul` 的功能扩展，在支持 `x1`、`x2` 为 FLOAT16 / BFLOAT16 的基础上，新增了对 INT8 / INT4 等低精度量化数据类型的支持。

**主要应用场景**：
- 大模型张量并行（TP）场景中的低精度线性层通信计算融合
- 需要将 AllGather 通信与量化 MatMul 计算流水并行的推理或训练任务
- GPT、LLaMA 等模型中 TP 维度的全收集后量化矩阵乘场景

**算子特征**：
- 难度等级：L4（FusedComposite）
- 支持 FLOAT16、BFLOAT16、INT8、INT4 输入（当前 benchmark cases 使用 INT8）
- `x1` 仅支持二维 ND 输入，不支持转置
- `x2` 支持二维 ND 输入，支持转置 / 不转置场景
- 支持 `x1Scale`、`x2Scale` 反量化参数
- 需要 HCCL 通信域，benchmark 中每个 case 会启动一个 rank 对应的进程

## 2. 算子定义

### 数学公式

**情形 1**：当 `x1`、`x2` 数据类型为 FLOAT16 / BFLOAT16 时，对 `x1` 执行 AllGather 后再与 `x2` 做 MatMul 计算：

$$
output = AllGather(x1) \times x2 + bias
$$

$$
gatherOut = AllGather(x1)
$$

**情形 2**：当 `x1`、`x2` 数据类型为 INT8 / INT4 的 pertoken / perchannel 量化场景时，对 `x1` 执行 AllGather 后做量化 MatMul，再进行反量化：

$$
output = (x1Scale \times x2Scale) \times (AllGather(x1) \times x2 + bias)
$$

$$
gatherOut = AllGather(x1)
$$

其中：
- `x1` 是当前 rank 的本地量化输入，shape 为 `[M_local, K]`
- `AllGather(x1)` 在第 0 维拼接所有 rank 的 `x1`，shape 为 `[M_local * rank_size, K]`
- `x2` 为量化权值。不转置时 shape 为 `[K, N]`；转置时 shape 为 `[N, K]`
- `x1Scale` 为 `x1` 的反量化参数，pertoken 场景下 shape 为 `[M_local]`，运算时 broadcast 为 `[M_total, 1]`
- `x2Scale` 为 `x2` 的反量化参数，perchannel 场景下 shape 为 `[N]`，运算时 broadcast 为 `[1, N]`
- `bias` 为可选一维偏置，shape 为 `[N]`；当前 benchmark cases 均不使用 bias
- `output` 的数据类型由 `out_dtype` 指定

### 计算步骤

1. 在 HCCL 通信域内对各 rank 的 `x1` 执行 AllGather，得到 `gatherOut`。
2. 在 HCCL 通信域内对各 rank 的 `x1Scale` 执行 AllGather（量化场景）。
3. 根据 `is_trans_b` 判断是否对 `x2` 做矩阵乘意义上的转置。
4. 执行矩阵乘 `AllGather(x1) @ x2`。
5. 当 `bias` 非空时，对矩阵乘结果加上 bias。
6. 量化场景下，乘以反量化尺度 `x1Scale * x2Scale`。
7. 当 `gather_output=True` 时，同时返回 AllGather 后的 `gatherOut`。

## 3. 接口规范

### 算子原型

```python
cann_bench.all_gather_matmul_v2(
    x1: Tensor,
    x2: Tensor,
    hcomm_info: str,
    world_size: int,
    bias: Optional[Tensor] = None,
    x1_scale: Optional[Tensor] = None,
    x2_scale: Optional[Tensor] = None,
    gather_output: bool = True,
    is_trans_b: bool = False,
    out_dtype: str = "float16",
    x1_dtype: str = "int8",
    x2_dtype: str = "int8",
    x1_scale_dtype: str = "float32",
    x2_scale_dtype: str = "float32",
    block_size: int = 128,
    group_sizes: List[int] = [],
) -> Tuple[Tensor, Optional[Tensor]]
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| x1 | Tensor | 必选 | 当前 rank 的本地左矩阵，shape 为 `[M_local, K]`。当前版本仅支持二维 ND 输入，不支持转置 |
| x2 | Tensor | 必选 | 右矩阵。不转置时 shape 为 `[K, N]`；转置时 shape 为 `[N, K]`。支持二维 ND 输入，支持转置 / 不转置场景 |
| hcomm_info | str | 必选 | HCCL 通信域名称，由通信库接口获取；benchmark runner 会在每个 rank 初始化后传入 |
| world_size | int | 必选 | 通信域内 rank 数 |
| bias | Tensor? | None | 可选偏置，shape 为 `[N]`。当前 benchmark cases 不使用 |
| x1_scale | Tensor? | None | `x1` 的反量化参数。pertoken 场景下 shape 为 `[M_local]`，数据类型为 float32 |
| x2_scale | Tensor? | None | `x2` 的反量化参数。perchannel 场景下 shape 为 `[N]`，数据类型为 float32 |
| gather_output | bool | True | 是否返回 AllGather 后的 `gatherOut` |
| is_trans_b | bool | False | 是否按转置形式使用 `x2` |
| out_dtype | str | `"float16"` | 输出 dtype，取值为 `float16` 或 `bfloat16` |
| x1_dtype | str | `"int8"` | `x1` 的逻辑量化 dtype，当前固定为 `int8` |
| x2_dtype | str | `"int8"` | `x2` 的逻辑量化 dtype，当前固定为 `int8` |
| x1_scale_dtype | str | `"float32"` | `x1_scale` 的 dtype，当前固定为 `float32` |
| x2_scale_dtype | str | `"float32"` | `x2_scale` 的 dtype，当前固定为 `float32` |
| block_size | int | 128 | 量化矩阵乘的 block size，当前 cases 使用默认值 128 |
| group_sizes | List[int] | `[]` | 量化矩阵乘的 group sizes，当前 cases 使用空列表 |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| y | `[M_local * world_size, N]` | float16 / bfloat16 | AllGather + 量化 MatMul 的输出 |
| gather_out | `[M_local * world_size, K]` 或 None | 与 `x1` 相同 | AllGather 后的中间结果；`gather_output=False` 时可不返回 |

### 数据类型

| x1 dtype | x2 dtype | x1_scale dtype | x2_scale dtype | 输出 dtype |
|----------|----------|----------------|----------------|------------|
| int8 | int8 | float32 | float32 | float16 |
| int8 | int8 | float32 | float32 | bfloat16 |

> 当前 benchmark cases 仅覆盖 INT8 输入 + FLOAT16 / BFLOAT16 输出场景。

### 规则与约束

- `x1` 和 `x2` 必须是二维 Tensor，且 dtype 保持一致。
- `x1` 仅支持不转置场景，`x2` 支持转置 / 不转置场景。
- `x1` 的 shape 为 `[M_local, K]`；`x2` 的 K 轴必须与 `x1` 的 K 轴匹配。
- `x1_scale` 在 pertoken 场景下 shape 为 `[M_local]`，`x2_scale` 在 perchannel 场景下 shape 为 `[N]`。
- `gather_output=True` 时输出 `gather_out`；`gather_output=False` 时只要求主输出 `y` 正确。
- `hcomm_info` 必须来自当前 HCCL 通信域，所有 rank 必须使用同一个通信域。
- 当前 benchmark case 使用 `world_size=8`。
- Atlas A2 / A3 支持 2、4、8 卡。

### 支持范围

| 维度 / 参数 | 范围 | 备注 |
|---|---|---|
| `M_local` | 64 ~ 2048 | cases.csv 实测范围 |
| `M_total` | `M_local * world_size` | cases.csv 实测 512 ~ 16384 |
| `K` | 4096 ~ 12288 | 典型 Transformer 模型隐藏层维度 |
| `N` | 1024 ~ 4096 | cases.csv 实测范围 |
| `world_size` | 8 | 当前 benchmark case 固定为 8 卡 |
| `out_dtype` | float16 / bfloat16 | 输出 dtype |
| `gather_output` | True / False | cases.csv 覆盖两种场景 |
| dtype | int8 (x1/x2), float32 (scale) | x1、x2 与 scale dtype 保持一致 |

## 4. 精度要求

本任务使用 cann-bench 的张量比较逻辑进行验证，并在 `proto.yaml` 中为 float16 / bfloat16 设置精度阈值 `0.005`。

**误差指标**：

1. 平均相对误差（MERE）：

   $$
   \text{MERE} = \text{avg}\left(\frac{\text{abs}(actual - golden)}{\text{abs}(golden)+1e-7}\right)
   $$

2. 最大相对误差（MARE）：

   $$
   \text{MARE} = \max\left(\frac{\text{abs}(actual - golden)}{\text{abs}(golden)+1e-7}\right)
   $$

当候选算子输出与 Golden 输出均满足阈值要求时，判定精度通过。

## 5. 标准 Golden 代码

```python
#!/usr/bin/python3
# coding=utf-8

from typing import Any, Dict, Optional, Tuple, Union

import torch


def all_gather_matmul_v2(
    x1: torch.Tensor,
    x2: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    x1_scale: Optional[torch.Tensor] = None,
    x2_scale: Optional[torch.Tensor] = None,
    hcomm_info: str = "",
    world_size: int = 1,
    gather_output: bool = True,
    is_trans_b: bool = False,
    out_dtype: str = "float16",
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Single-process reference for metadata and smoke checks."""
    del hcomm_info
    x2_eff = x2.t() if is_trans_b else x2
    gathered = torch.cat([x1] * int(world_size), dim=0).float()
    out = torch.matmul(gathered, x2_eff.float())
    if x1_scale is not None:
        scale = torch.cat([x1_scale] * int(world_size), dim=0).float().reshape(-1, 1)
        out = out * scale
    if x2_scale is not None:
        out = out * x2_scale.float().reshape(1, -1)
    if bias is not None:
        out = out + bias.float()
    out = out.to(_torch_dtype(out_dtype))
    if gather_output:
        return out, gathered.to(x1.dtype)
    return out


def mc2_make_rank_inputs(ctx: Dict[str, Any], case_payload: Dict[str, Any]) -> Dict[str, Any]:
    device = ctx["device"]
    shapes = case_payload["input_shapes"]
    dtypes = case_payload["dtypes"]
    ranges = case_payload["value_ranges"]
    attrs = case_payload["attrs"]
    seed = int(attrs.get("seed", 1))
    weight_same = int(attrs.get("weight_same", 1))
    rank = int(ctx["rank"])

    x1 = _make_tensor(shapes[0], dtypes[0], ranges[0], seed, device)
    x2_seed = seed + rank * 2 if weight_same else seed
    x2 = _make_tensor(shapes[1], dtypes[1], ranges[1], x2_seed, device)

    bias = None
    if bool(attrs.get("is_bias", False)) and len(shapes) > 2 and shapes[2] is not None:
        bias = _make_tensor(shapes[2], dtypes[2], ranges[2], rank * 3, device)

    x1_scale_index = 3 if bool(attrs.get("is_bias", False)) else 2
    x2_scale_index = x1_scale_index + 1
    x1_scale = _make_tensor(
        shapes[x1_scale_index],
        dtypes[x1_scale_index],
        ranges[x1_scale_index],
        rank * 11,
        device,
    )
    x2_scale = _make_tensor(
        shapes[x2_scale_index],
        dtypes[x2_scale_index],
        ranges[x2_scale_index],
        rank * 11,
        device,
    )
    return {"x1": x1, "x2": x2, "bias": bias, "x1_scale": x1_scale, "x2_scale": x2_scale}


def mc2_call_candidate(candidate, ctx: Dict[str, Any], inputs: Dict[str, Any], attrs: Dict[str, Any]):
    return candidate(
        inputs["x1"],
        inputs["x2"],
        ctx["hcomm_info"],
        int(ctx["world_size"]),
        bias=inputs.get("bias"),
        x1_scale=inputs.get("x1_scale"),
        x2_scale=inputs.get("x2_scale"),
        gather_output=bool(attrs.get("gather_output", True)),
        is_trans_b=bool(attrs.get("is_trans_b", False)),
        out_dtype=str(attrs.get("out_dtype", "float16")),
        x1_dtype=str(attrs.get("x1_dtype", "int8")),
        x2_dtype=str(attrs.get("x2_dtype", "int8")),
        x1_scale_dtype=str(attrs.get("x1_scale_dtype", "float32")),
        x2_scale_dtype=str(attrs.get("x2_scale_dtype", "float32")),
        block_size=int(attrs.get("block_size", 128)),
        group_sizes=list(attrs.get("group_sizes", [])),
    )


def mc2_distributed_golden(ctx: Dict[str, Any], inputs: Dict[str, Any], attrs: Dict[str, Any]):
    """Distributed golden following mc2_test's hccl_mm reference path."""
    torch_npu = ctx["torch_npu"]
    dist = ctx["dist"]
    world_size = int(ctx["world_size"])
    gather_shape = [inputs["x1"].shape[0] * world_size] + list(inputs["x1"].shape[1:])
    gathered = torch.empty(gather_shape, dtype=inputs["x1"].dtype, device=inputs["x1"].device)
    dist._all_gather_base(gathered, inputs["x1"])
    gathered_x1_scale = inputs.get("x1_scale")
    if gathered_x1_scale is not None:
        scale_shape = [gathered_x1_scale.shape[0] * world_size] + list(gathered_x1_scale.shape[1:])
        gathered_scale = torch.empty(scale_shape, dtype=gathered_x1_scale.dtype, device=gathered_x1_scale.device)
        dist._all_gather_base(gathered_scale, gathered_x1_scale)
        gathered_x1_scale = gathered_scale
    x2_eff = inputs["x2"].t() if bool(attrs.get("is_trans_b", False)) else inputs["x2"]
    out_dtype = _torch_dtype(str(attrs.get("out_dtype", "float16")))
    if str(attrs.get("x1_dtype", "int8")).lower() in ("fp16", "float16", "bf16", "bfloat16"):
        output = torch.matmul(gathered, x2_eff)
        if inputs.get("bias") is not None:
            output = output + inputs["bias"]
    else:
        output = torch_npu.npu_quant_matmul(
            gathered,
            x2_eff,
            inputs.get("x2_scale"),
            offset=None,
            pertoken_scale=gathered_x1_scale,
            bias=inputs.get("bias"),
            output_dtype=out_dtype,
            group_sizes=list(attrs.get("group_sizes", [])),
        )
    if bool(attrs.get("gather_output", True)):
        return output, gathered
    return output


def _make_tensor(shape, dtype_name: str, value_range, seed: int, device):
    torch.manual_seed(int(seed))
    dtype = _torch_dtype(dtype_name)
    lo, hi = value_range if value_range is not None else [0, 1]
    if dtype.is_floating_point:
        tensor = torch.empty(shape, dtype=torch.float32).uniform_(float(lo), float(hi)).to(dtype)
    else:
        tensor = torch.randint(int(lo), int(hi) + 1, shape, dtype=dtype)
    return tensor.to(device)


def _torch_dtype(dtype_name: str):
    dtype = str(dtype_name).lower()
    mapping = {
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
        "float32": torch.float32,
        "int8": torch.int8,
        "int32": torch.int32,
        "int64": torch.int64,
    }
    if dtype not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    return mapping[dtype]
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

world_size = 8
M_local, K, N = 256, 4096, 1024

x1 = torch.randint(-8, 9, (M_local, K), dtype=torch.int8, device="npu")
x2 = torch.randint(-8, 9, (K, N), dtype=torch.int8, device="npu")
x1_scale = torch.empty(M_local, 1, dtype=torch.float32, device="npu").uniform_(0.001, 0.02)
x2_scale = torch.empty(1, N, dtype=torch.float32, device="npu").uniform_(0.001, 0.02)
hcomm_info = "<hccl-comm-name>"

y, gather_out = cann_bench.all_gather_matmul_v2(
    x1,
    x2,
    hcomm_info,
    world_size,
    x1_scale=x1_scale,
    x2_scale=x2_scale,
    gather_output=True,
    is_trans_b=False,
    out_dtype="float16",
)

# y.shape: [M_local * world_size, N]
# gather_out.shape: [M_local * world_size, K]
```

### benchmark 说明

该任务不是单进程算子用例。case 中设置了 `attrs.mc2_distributed: true`，cann-bench 会为每个 case 启动 `world_size` 个 rank 进程，初始化 HCCL 通信域后再调用候选实现。候选源码包需要暴露 `cann_bench.all_gather_matmul_v2` 或 `torch.ops.cann_bench.all_gather_matmul_v2`。当前 cases 目标平台为 Atlas A2 / A3。
