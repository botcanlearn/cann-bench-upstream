# AllGatherMatmul 算子 API 描述

## 1. 算子简介

AllGatherMatmul 是一个 MC2 通算融合算子，用于将 AllGather 集合通信和 MatMul 计算融合执行。算子先在通信域内收集各 rank 的 `x1`，再与 `x2` 做矩阵乘，并可选输出 AllGather 后的中间结果。

**主要应用场景**：
- 大模型张量并行场景中的线性层通信计算融合
- 需要将 AllGather 通信与矩阵乘法流水并行的推理或训练任务
- GPT、LLaMA、BLOOM 等模型中 TP 维度的全收集后矩阵乘场景

**算子特征**：
- 难度等级：L4（FusedComposite）
- 支持 FLOAT16、BFLOAT16 输入输出
- `x1` 仅支持二维 ND 输入，不支持转置
- `x2` 支持二维 ND 输入，支持转置 / 不转置场景
- 支持可选 `bias`，当前 benchmark case 均为空 bias
- 需要 HCCL 通信域，benchmark 中每个 case 会启动一个 rank 对应的进程

## 2. 算子定义

### 数学公式

$$
gather\_out = AllGather(x1)
$$

$$
y = gather\_out \times x2 + bias
$$

其中：
- `x1` 是当前 rank 的本地输入，shape 为 `[M_local, K]`
- `AllGather(x1)` 在第 0 维拼接所有 rank 的 `x1`，shape 为 `[M_local * rank_size, K]`
- `x2` 为右矩阵，未转置时 shape 为 `[K, N]`，转置场景下输入 shape 为 `[N, K]`
- `bias` 为可选一维偏置，shape 为 `[N]`

### 计算步骤

1. 在 HCCL 通信域内对各 rank 的 `x1` 执行 AllGather。
2. 根据 `is_trans_b` 判断是否对 `x2` 做矩阵乘意义上的转置。
3. 执行矩阵乘 `AllGather(x1) @ x2`。
4. 当 `bias` 非空时，对输出结果加上 bias。
5. 当 `gather_output=True` 时，同时返回 AllGather 后的 `gather_out`。

## 3. 接口规范

### 算子原型

```python
cann_bench.all_gather_matmul(
    x1: Tensor,
    x2: Tensor,
    hcomm_info: str,
    world_size: int,
    bias: Optional[Tensor] = None,
    gather_output: bool = True,
    is_trans_b: bool = False,
) -> Tuple[Tensor, Optional[Tensor]]
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| x1 | Tensor | 必选 | 当前 rank 的本地左矩阵，shape 为 `[M_local, K]`，dtype 为 float16 / bfloat16 |
| x2 | Tensor | 必选 | 右矩阵。`is_trans_b=False` 时 shape 为 `[K, N]`；`is_trans_b=True` 时 shape 为 `[N, K]` |
| hcomm_info | str | 必选 | HCCL 通信域名称，由通信库接口获取；benchmark runner 会在每个 rank 初始化后传入 |
| world_size | int | 必选 | 通信域内 rank 数 |
| bias | Tensor? | None | 可选偏置，shape 为 `[N]` |
| gather_output | bool | True | 是否返回 AllGather 后的 `gather_out` |
| is_trans_b | bool | False | 是否按转置形式使用 `x2` |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| y | `[M_local * world_size, N]` | 与 `x1` 相同 | AllGather + MatMul 的输出 |
| gather_out | `[M_local * world_size, K]` 或 None | 与 `x1` 相同 | AllGather 后的中间结果；`gather_output=False` 时可不返回 |

### 数据类型

| x1 dtype | x2 dtype | bias dtype | 输出 dtype |
|----------|----------|------------|------------|
| float16 | float16 | float16 / None | float16 |
| bfloat16 | bfloat16 | bfloat16 / None | bfloat16 |

### 规则与约束

- `x1` 和 `x2` 必须是二维 ND Tensor，且 dtype 一致。
- `x1` 的 shape 为 `[M_local, K]`；`x2` 的 K 轴必须与 `x1` 的 K 轴匹配。
- `x2` 支持转置 / 不转置场景，仅支持两根轴转置情况下的非连续 Tensor。
- `x1` 只支持不转置场景。
- `gather_output=True` 时输出 `gather_out`；`gather_output=False` 时只要求主输出 `y` 正确。
- `hcomm_info` 必须来自当前 HCCL 通信域，所有 rank 必须使用同一个通信域。
- `gatherIndex` 语义固定为 0，即 gather 目标为 `x1`；`commTurn` 语义固定为 0。
- 当前 benchmark case 使用 `world_size=8`，来源于 `mc2_test/excel/aclnnAllGatherMatmul.xlsx` 的 `level0` sheet；未使用 `david_excel` 中的 910D 用例。

### 支持范围

| 维度 / 参数 | 范围 | 备注 |
|---|---|---|
| `M_local` | 512 ~ 2048 | cases.csv 实测范围 |
| `M_total` | `M_local * world_size` | cases.csv 实测 4096 ~ 16384 |
| `K` | 4096 ~ 12288 | 需满足 MatMul K 轴匹配 |
| `N` | 640 ~ 15744 | cases.csv 实测范围 |
| `world_size` | 8 | 当前 benchmark case 固定为 8 卡 |
| `is_trans_b` | True / False | cases.csv 覆盖两种场景 |
| `gather_output` | True / False | cases.csv 覆盖两种场景 |
| dtype | float16 / bfloat16 | x1、x2、bias 与输出 dtype 保持一致 |

产品与组网约束参考 aclnnAllGatherMatmul 文档：Atlas A2 支持 2、4、8 卡 HCCS all mesh；Atlas A3 支持 2、4、8、16、32 卡 HCCS double ring；Ascend 950PR/950DT 支持 2、4、8、16、32、64 卡 HCCS all mesh。

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


def all_gather_matmul(
    x1: torch.Tensor,
    x2: torch.Tensor,
    hcomm_info: str,
    world_size: int,
    bias: torch.Tensor = None,
    gather_output: bool = True,
    is_trans_b: bool = False,
):
    """
    AllGatherMatmul Golden 参考实现。

    benchmark runner 会负责初始化 HCCL 通信域并传入 hcomm_info；
    Golden 侧使用 torch.distributed 的 AllGather 构造参考结果。
    """
    gathered = [torch.empty_like(x1) for _ in range(world_size)]
    dist.all_gather(gathered, x1)
    gather_out = torch.cat(gathered, dim=0)

    weight = x2.transpose(0, 1) if is_trans_b else x2
    y = torch.matmul(gather_out.float(), weight.float())
    if bias is not None:
        y = y + bias.float()
    y = y.to(dtype=x1.dtype)

    if gather_output:
        return y, gather_out
    return y, None
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

world_size = 8
M_local, K, N = 2048, 5120, 1920

x1 = torch.randn(M_local, K, dtype=torch.float16, device="npu")
x2 = torch.randn(N, K, dtype=torch.float16, device="npu")
hcomm_info = "<hccl-comm-name>"

y, gather_out = cann_bench.all_gather_matmul(
    x1,
    x2,
    hcomm_info,
    world_size,
    bias=None,
    gather_output=True,
    is_trans_b=True,
)

# y.shape: [M_local * world_size, N]
# gather_out.shape: [M_local * world_size, K]
```

### benchmark 说明

该任务不是单进程算子用例。case 中设置了 `attrs.mc2_distributed: true`，cann-bench 会为每个 case 启动 `world_size` 个 rank 进程，初始化 HCCL 通信域后再调用候选实现。候选源码包需要暴露 `cann_bench.all_gather_matmul` 或 `torch.ops.cann_bench.all_gather_matmul`。
