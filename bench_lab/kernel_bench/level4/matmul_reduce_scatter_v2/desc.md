# MatmulReduceScatterV2 算子 API 描述

## 1. 算子简介

MatmulReduceScatterV2 是一个 MC2 通算融合量化算子，用于将量化 MatMul 计算和 ReduceScatter 集合通信融合执行。该算子是 `aclnnMatmulReduceScatter` 的功能扩展，在支持 `x1`、`x2` 为 FLOAT16 / BFLOAT16 的基础上，新增了对 INT8 等低精度量化数据类型的支持。

**主要应用场景**：
- 大模型张量并行（TP）场景中的量化矩阵乘后 ReduceScatter 通信
- 需要将量化 MatMul 计算与集合通信流水并行的推理或训练任务
- GPT、LLaMA 等模型中 TP 维度的 reduce-scatter 场景

**算子特征**：
- 难度等级：L4（FusedComposite）
- 支持 FLOAT16、BFLOAT16、INT8 输入（当前 benchmark cases 使用 INT8）
- `x1` 仅支持二维 ND 输入，不支持转置
- `x2` 支持二维 ND 输入，支持转置 / 不转置场景
- 支持 `x1Scale`、`x2Scale` 反量化参数
- `reduce_op` 当前仅支持 `"sum"`
- 需要 HCCL 通信域，benchmark 中每个 case 会启动一个 rank 对应的进程

## 2. 算子定义

### 数学公式

**情形 1**：当 `x1`、`x2` 数据类型为 FLOAT16 / BFLOAT16 时，对 `x1`、`x2` 做 MatMul 计算后，进行 ReduceScatter 通信：

$$
output = ReduceScatter(x1 \times x2 + bias_{optional})
$$

**情形 2**：当 `x1`、`x2` 数据类型为 INT8 的 pertoken / perchannel 量化场景时，对 `x1`、`x2` 做 MatMul 计算和反量化后，再进行 ReduceScatter 通信：

$$
output = ReduceScatter((x1Scale \times x2Scale) \times (x1 \times x2 + bias_{optional}))
$$

其中：
- `x1` 为矩阵乘左矩阵，shape 为 `[M, K]`
- `x2` 为矩阵乘右矩阵。不转置时 shape 为 `[K, N]`；转置时 shape 为 `[N, K]`
- `x1Scale` 为 `x1` 的反量化参数，pertoken 场景下 shape 为 `[M]`，运算时 broadcast 为 `[M, 1]`
- `x2Scale` 为 `x2` 的反量化参数，perchannel 场景下 shape 为 `[N]`，运算时 broadcast 为 `[1, N]`
- `bias` 为可选一维偏置，shape 为 `[N]`；当前 benchmark cases 均不使用 bias
- `ReduceScatter` 对各 rank 的矩阵乘结果做 sum 规约，并沿第 0 维切分输出，shape 为 `[M / rank_size, N]`

### 计算步骤

1. 根据 `is_trans_b` 判断是否对 `x2` 做矩阵乘意义上的转置。
2. 执行矩阵乘 `x1 @ x2`。
3. 当 `bias` 非空时，对矩阵乘结果加上 bias。
4. 量化场景下，乘以反量化尺度 `x1Scale * x2Scale`。
5. 在 HCCL 通信域内对各 rank 的结果执行 ReduceScatter（`reduce_op="sum"`）。
6. 返回当前 rank 对应的规约切片。

## 3. 接口规范

### 算子原型

```python
cann_bench.matmul_reduce_scatter_v2(
    x1: Tensor,
    x2: Tensor,
    hcomm_info: str,
    world_size: int,
    reduce_op: str = "sum",
    bias: Optional[Tensor] = None,
    x1_scale: Optional[Tensor] = None,
    x2_scale: Optional[Tensor] = None,
    is_trans_b: bool = False,
    out_dtype: str = "float16",
    x1_dtype: str = "int8",
    x2_dtype: str = "int8",
    x1_scale_dtype: str = "float32",
    x2_scale_dtype: str = "float32",
    block_size: int = 128,
    group_sizes: List[int] = [],
) -> Tensor
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| x1 | Tensor | 必选 | 矩阵乘左矩阵，shape 为 `[M, K]`。当前版本仅支持二维 ND 输入，不支持转置 |
| x2 | Tensor | 必选 | 矩阵乘右矩阵。不转置时 shape 为 `[K, N]`；转置时 shape 为 `[N, K]`。支持二维 ND 输入，支持转置 / 不转置场景 |
| hcomm_info | str | 必选 | HCCL 通信域名称，由通信库接口获取；benchmark runner 会在每个 rank 初始化后传入 |
| world_size | int | 必选 | 通信域内 rank 数 |
| reduce_op | str | `"sum"` | ReduceScatter 规约类型，当前仅支持 `"sum"` |
| bias | Tensor? | None | 可选偏置，shape 为 `[N]`。当前 benchmark cases 不使用 |
| x1_scale | Tensor? | None | `x1` 的反量化参数。pertoken 场景下 shape 为 `[M]`，数据类型为 float32 |
| x2_scale | Tensor? | None | `x2` 的反量化参数。perchannel 场景下 shape 为 `[N]`，数据类型为 float32 |
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
| y | `[M / world_size, N]` | float16 / bfloat16 | ReduceScatter 后当前 rank 的输出切片 |

### 数据类型

| x1 dtype | x2 dtype | x1_scale dtype | x2_scale dtype | 输出 dtype |
|----------|----------|----------------|----------------|------------|
| int8 | int8 | float32 | float32 | float16 |
| int8 | int8 | float32 | float32 | bfloat16 |

> 当前 benchmark cases 仅覆盖 INT8 输入 + FLOAT16 / BFLOAT16 输出场景。

### 规则与约束

- `x1` 和 `x2` 必须是二维 Tensor，且 dtype 保持一致。
- `x1` 仅支持不转置场景，`x2` 支持转置 / 不转置场景。
- `x1` 的 shape 为 `[M, K]`，其中 `M` 必须能被 `world_size` 整除。
- `x2` 的 K 轴必须与 `x1` 的 K 轴匹配。
- `x1_scale` 在 pertoken 场景下 shape 为 `[M]`，`x2_scale` 在 perchannel 场景下 shape 为 `[N]`。
- `reduce_op` 当前仅支持 `"sum"`。
- `hcomm_info` 必须来自当前 HCCL 通信域，所有 rank 必须使用同一个通信域。
- 当前 benchmark case 使用 `world_size=8`。
- Atlas A2 / A3 支持 2、4、8 卡。

### 支持范围

| 维度 / 参数 | 范围 | 备注 |
|---|---|---|
| `M` | 512 ~ 8192 | cases.csv 实测范围，必须能被 `world_size` 整除 |
| `K` | 4096 ~ 12288 | 典型 Transformer 模型隐藏层维度 |
| `N` | 1024 ~ 4096 | cases.csv 实测范围 |
| `world_size` | 8 | 当前 benchmark case 固定为 8 卡 |
| `reduce_op` | `"sum"` | 当前版本唯一支持值 |
| `out_dtype` | float16 / bfloat16 | 输出 dtype |
| dtype | int8 (x1/x2), float32 (scale) | x1、x2 与 scale dtype 保持一致 |

## 4. 精度要求

本任务使用 cann-bench 的张量比较逻辑进行验证，并在 `proto.yaml` 中为 float16 / bfloat16 设置精度阈值 `0.01`。

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

from typing import Any, Dict, Optional

import torch


def matmul_reduce_scatter_v2(
    x1: torch.Tensor,
    x2: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    x1_scale: Optional[torch.Tensor] = None,
    x2_scale: Optional[torch.Tensor] = None,
    hcomm_info: str = "",
    world_size: int = 1,
    reduce_op: str = "sum",
    is_trans_b: bool = False,
    out_dtype: str = "float16",
) -> torch.Tensor:
    """Single-process reference for metadata and smoke checks."""
    del hcomm_info
    if reduce_op != "sum":
        raise ValueError(f"Unsupported reduce_op: {reduce_op}")
    x2_eff = x2.t() if is_trans_b else x2
    out = torch.matmul(x1.float(), x2_eff.float())
    if bias is not None:
        out = out + bias.float()
    if x1_scale is not None or x2_scale is not None:
        out = _apply_dequant_scale(out, x1_scale, x2_scale)
    out = out * int(world_size)
    chunk = out.shape[0] // int(world_size)
    return out[:chunk].to(_torch_dtype(out_dtype))


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
    x2_seed = seed if weight_same else seed + rank * 2
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
        seed * 11,
        device,
    )
    x2_scale = _make_tensor(
        shapes[x2_scale_index],
        dtypes[x2_scale_index],
        ranges[x2_scale_index],
        seed * 13,
        device,
    )
    return {"x1": x1, "x2": x2, "bias": bias, "x1_scale": x1_scale, "x2_scale": x2_scale}


def mc2_call_candidate(candidate, ctx: Dict[str, Any], inputs: Dict[str, Any], attrs: Dict[str, Any]):
    return candidate(
        inputs["x1"],
        inputs["x2"],
        ctx["hcomm_info"],
        int(ctx["world_size"]),
        reduce_op=str(attrs.get("reduce_op", "sum")),
        bias=inputs.get("bias"),
        x1_scale=inputs.get("x1_scale"),
        x2_scale=inputs.get("x2_scale"),
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
    """Distributed CPU-semantic golden following mc2_test's get_cpu path.

    Compute matmul and dequant in float32 for precision, then cast to the
    operator's output dtype before all_reduce so the communication precision
    matches the fused MC2 kernel (which communicates in fp16/bf16).
    """
    dist = ctx["dist"]
    x2_eff = inputs["x2"].t() if bool(attrs.get("is_trans_b", False)) else inputs["x2"]
    mm_out = torch.matmul(inputs["x1"].float(), x2_eff.float())
    if inputs.get("bias") is not None:
        mm_out = mm_out + inputs["bias"].float()
    if _is_quant_case(attrs):
        mm_out = _apply_dequant_scale(mm_out, inputs.get("x1_scale"), inputs.get("x2_scale"))
    # Cast to output dtype before all_reduce to match candidate's communication precision
    out_dtype = _torch_dtype(str(attrs.get("out_dtype", "float16")))
    mm_out = mm_out.to(out_dtype)
    dist.all_reduce(mm_out, op=dist.ReduceOp.SUM)
    chunk = mm_out.shape[0] // int(ctx["world_size"])
    return mm_out.narrow(0, int(ctx["rank"]) * chunk, chunk)


def _is_quant_case(attrs: Dict[str, Any]) -> bool:
    x1_dtype = str(attrs.get("x1_dtype", "int8")).lower()
    x2_dtype = str(attrs.get("x2_dtype", "int8")).lower()
    non_quant = {"fp16", "float16", "bf16", "bfloat16"}
    return x1_dtype not in non_quant or x2_dtype not in non_quant


def _apply_dequant_scale(
    out: torch.Tensor,
    x1_scale: Optional[torch.Tensor],
    x2_scale: Optional[torch.Tensor],
) -> torch.Tensor:
    """Apply MC2 INT8 per-token/per-channel dequant scale.

    mc2_test truncates the float32 scale product with ``0xffffe000`` before it
    multiplies the matmul result.  Keep that behavior here, while leaving final
    dtype rounding to cann-bench's normal compare path.
    """
    if x1_scale is None and x2_scale is None:
        return out

    rows, cols = int(out.shape[-2]), int(out.shape[-1])
    device = out.device
    x1 = _scale_tensor_or_ones(x1_scale, rows, device)
    x2 = _scale_tensor_or_ones(x2_scale, cols, device)

    if x1.numel() == rows and x2.numel() == cols:
        result = out.clone()
        row_scale = x1.float().reshape(rows, 1)
        col_scale = x2.float().reshape(1, cols)
        for start in range(0, rows, 512):
            end = min(start + 512, rows)
            scale = _truncate_hw_scale(row_scale[start:end] * col_scale)
            result[start:end] = result[start:end] * scale
        return result

    scale = _broadcast_dequant_scale(x1, x2, rows, cols)
    return out * _truncate_hw_scale(scale)


def _scale_tensor_or_ones(scale: Optional[torch.Tensor], size: int, device: torch.device) -> torch.Tensor:
    if scale is None:
        return torch.ones(size, dtype=torch.float32, device=device)
    return scale.float().to(device)


def _broadcast_dequant_scale(
    x1_scale: torch.Tensor,
    x2_scale: torch.Tensor,
    rows: int,
    cols: int,
) -> torch.Tensor:
    x1 = x1_scale.float()
    x2 = x2_scale.float()
    if x1.numel() == 1 and x2.numel() == 1:
        return x1.reshape(1) * x2.reshape(1)
    if x1.numel() == rows and x2.numel() == 1:
        return x1.reshape(rows, 1) * x2.reshape(1)
    if x1.numel() == 1 and x2.numel() == cols:
        return x1.reshape(1) * x2.reshape(1, cols)
    if list(x1.shape) == [rows, 1] and x2.numel() == cols:
        return x1 * x2.reshape(1, cols)
    if x1.numel() == rows and list(x2.shape) == [1, cols]:
        return x1.reshape(rows, 1) * x2
    if list(x1.shape) == [rows, cols] and x2.numel() == 1:
        return x1 * x2.reshape(1)
    raise ValueError(
        "Unsupported dequant scale shapes: "
        f"x1_scale={tuple(x1_scale.shape)}, x2_scale={tuple(x2_scale.shape)}, output=({rows}, {cols})"
    )


def _truncate_hw_scale(scale: torch.Tensor) -> torch.Tensor:
    bits = scale.float().view(torch.int32)
    bits &= torch.tensor(0xFFFFE000, dtype=torch.int32, device=scale.device)
    return bits.view(torch.float32)


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
M, K, N = 1024, 4096, 1024

x1 = torch.randint(-8, 9, (M, K), dtype=torch.int8, device="npu")
x2 = torch.randint(-8, 9, (K, N), dtype=torch.int8, device="npu")
x1_scale = torch.empty(M, 1, dtype=torch.float32, device="npu").uniform_(0.001, 0.02)
x2_scale = torch.empty(1, N, dtype=torch.float32, device="npu").uniform_(0.001, 0.02)
hcomm_info = "<hccl-comm-name>"

y = cann_bench.matmul_reduce_scatter_v2(
    x1,
    x2,
    hcomm_info,
    world_size,
    reduce_op="sum",
    x1_scale=x1_scale,
    x2_scale=x2_scale,
    is_trans_b=False,
    out_dtype="float16",
)

# y.shape: [M / world_size, N]
```

### benchmark 说明

该任务不是单进程算子用例。case 中设置了 `attrs.mc2_distributed: true`，cann-bench 会为每个 case 启动 `world_size` 个 rank 进程，初始化 HCCL 通信域后再调用候选实现。候选源码包需要暴露 `cann_bench.matmul_reduce_scatter_v2` 或 `torch.ops.cann_bench.matmul_reduce_scatter_v2`。当前 cases 目标平台为 Atlas A2 / A3。
