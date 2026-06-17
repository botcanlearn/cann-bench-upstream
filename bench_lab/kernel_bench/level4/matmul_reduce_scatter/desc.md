# MatmulReduceScatter 算子 API 描述

## 1. 算子简介

MatmulReduceScatter 是一个 MC2 通算融合算子，用于将 MatMul 计算和 ReduceScatter 集合通信融合执行。算子先计算本 rank 的矩阵乘结果，再在通信域内执行 ReduceScatter，将规约后的结果按第 0 维切分到各 rank。

**主要应用场景**：
- 大模型张量并行场景中的矩阵乘后 ReduceScatter 通信
- 需要将 MatMul 计算与集合通信流水并行的推理或训练任务
- GPT、LLaMA、BLOOM 等模型中 TP 维度的 reduce-scatter 场景

**算子特征**：
- 难度等级：L4（FusedComposite）
- 支持 FLOAT16、BFLOAT16 输入输出
- `x1` 仅支持二维 ND 输入，不支持转置
- `x2` 支持二维 ND 输入，支持转置 / 不转置场景
- `reduce_op` 当前仅支持 `"sum"`
- 需要 HCCL 通信域，benchmark 中每个 case 会启动一个 rank 对应的进程

## 2. 算子定义

### 数学公式

$$
mm = x1 \times x2 + bias
$$

$$
y = ReduceScatter(mm, op=sum)
$$

其中：
- `x1` 为矩阵乘左矩阵，shape 为 `[M, K]`
- `x2` 为矩阵乘右矩阵，未转置时 shape 为 `[K, N]`，转置场景下输入 shape 为 `[N, K]`
- `bias` 为可选一维偏置，shape 为 `[N]`
- `ReduceScatter` 对各 rank 的 `mm` 做 sum 规约，并沿第 0 维切分输出，shape 为 `[M / rank_size, N]`

### 计算步骤

1. 根据 `is_trans_b` 判断是否对 `x2` 做矩阵乘意义上的转置。
2. 执行矩阵乘 `x1 @ x2`。
3. 当 `bias` 非空时，对矩阵乘结果加上 bias。
4. 在 HCCL 通信域内对各 rank 的矩阵乘结果执行 ReduceScatter。
5. 返回当前 rank 对应的规约切片。

## 3. 接口规范

### 算子原型

```python
cann_bench.matmul_reduce_scatter(
    x1: Tensor,
    x2: Tensor,
    hcomm_info: str,
    world_size: int,
    reduce_op: str = "sum",
    bias: Optional[Tensor] = None,
    is_trans_b: bool = False,
) -> Tensor
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| x1 | Tensor | 必选 | 矩阵乘左矩阵，shape 为 `[M, K]`，dtype 为 float16 / bfloat16 |
| x2 | Tensor | 必选 | 矩阵乘右矩阵。`is_trans_b=False` 时 shape 为 `[K, N]`；`is_trans_b=True` 时 shape 为 `[N, K]` |
| hcomm_info | str | 必选 | HCCL 通信域名称，由通信库接口获取；benchmark runner 会在每个 rank 初始化后传入 |
| world_size | int | 必选 | 通信域内 rank 数 |
| reduce_op | str | `"sum"` | ReduceScatter 规约类型，当前仅支持 `"sum"` |
| bias | Tensor? | None | 可选偏置，shape 为 `[N]` |
| is_trans_b | bool | False | 是否按转置形式使用 `x2` |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| y | `[M / world_size, N]` | 与 `x1` 相同 | ReduceScatter 后当前 rank 的输出切片 |

### 数据类型

| x1 dtype | x2 dtype | bias dtype | 输出 dtype |
|----------|----------|------------|------------|
| float16 | float16 | float16 / None | float16 |
| bfloat16 | bfloat16 | bfloat16 / None | bfloat16 |

### 规则与约束

- `x1` 和 `x2` 必须是二维 ND Tensor，且 dtype 一致。
- `x1` 的 shape 为 `[M, K]`，其中 `M` 必须能被 `world_size` 整除。
- `x2` 的 K 轴必须与 `x1` 的 K 轴匹配。
- `x2` 支持转置 / 不转置场景，仅支持两根轴转置情况下的非连续 Tensor。
- `x1` 只支持不转置场景。
- `reduce_op` 当前仅支持 `"sum"`。
- `hcomm_info` 必须来自当前 HCCL 通信域，所有 rank 必须使用同一个通信域。
- `commTurn` 语义固定为 0，`streamMode` 语义固定为 stop-on-failure。
- 当前 benchmark case 使用 `world_size=8`，来源于 `mc2_test/excel/aclnnMatmulReduceScatter.xlsx` 的 `level0` sheet；未使用 `david_excel` 中的 910D 用例。

### 支持范围

| 维度 / 参数 | 范围 | 备注 |
|---|---|---|
| `M` | 2048 ~ 16384 | cases.csv 实测范围，必须能被 `world_size` 整除 |
| `K` | 512 ~ 5120 | cases.csv 中包含 GPT / LLaMA / BLOOM / Z1 典型形状 |
| `N` | 4096 ~ 12288 | cases.csv 实测范围 |
| `world_size` | 8 | 当前 benchmark case 固定为 8 卡 |
| `reduce_op` | `"sum"` | 当前版本唯一支持值 |
| `is_trans_b` | True | 当前 cases.csv 覆盖转置右矩阵场景 |
| dtype | float16 / bfloat16 | x1、x2、bias 与输出 dtype 保持一致 |

产品与组网约束参考 aclnnMatmulReduceScatter 文档：Atlas A2 支持 2、4、8 卡 HCCS all mesh；Atlas A3 支持 2、4、8、16、32 卡 HCCS double ring；Ascend 950PR/950DT 支持 2、4、8、16、32、64 卡 HCCS all mesh。Atlas A2 / A3 上默认非确定性实现，Ascend 950PR/950DT 上默认为确定性实现。

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
import torch
import torch.distributed as dist


def matmul_reduce_scatter(
    x1: torch.Tensor,
    x2: torch.Tensor,
    hcomm_info: str,
    world_size: int,
    reduce_op: str = "sum",
    bias: torch.Tensor = None,
    is_trans_b: bool = False,
) -> torch.Tensor:
    """
    MatmulReduceScatter Golden 参考实现。

    benchmark runner 会负责初始化 HCCL 通信域并传入 hcomm_info；
    Golden 侧使用 torch.distributed 的 ReduceScatter 构造参考结果。
    """
    if reduce_op != "sum":
        raise ValueError("matmul_reduce_scatter only supports reduce_op='sum'")

    weight = x2.transpose(0, 1) if is_trans_b else x2
    mm = torch.matmul(x1.float(), weight.float())
    if bias is not None:
        mm = mm + bias.float()
    mm = mm.to(dtype=x1.dtype).contiguous()

    output = torch.empty(
        mm.shape[0] // world_size,
        mm.shape[1],
        dtype=mm.dtype,
        device=mm.device,
    )
    dist.reduce_scatter_tensor(output, mm, op=dist.ReduceOp.SUM)
    return output
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

world_size = 8
M, K, N = 16384, 640, 5120

x1 = torch.randn(M, K, dtype=torch.float16, device="npu")
x2 = torch.randn(N, K, dtype=torch.float16, device="npu")
hcomm_info = "<hccl-comm-name>"

y = cann_bench.matmul_reduce_scatter(
    x1,
    x2,
    hcomm_info,
    world_size,
    reduce_op="sum",
    bias=None,
    is_trans_b=True,
)

# y.shape: [M / world_size, N]
```

### benchmark 说明

该任务不是单进程算子用例。case 中设置了 `attrs.mc2_distributed: true`，cann-bench 会为每个 case 启动 `world_size` 个 rank 进程，初始化 HCCL 通信域后再调用候选实现。候选源码包需要暴露 `cann_bench.matmul_reduce_scatter` 或 `torch.ops.cann_bench.matmul_reduce_scatter`。
