# GroupedMatMulAlltoAllv 算子 API 描述

## 1. 算子简介

GroupedMatMulAlltoAllv 是一个 MC2 通算融合算子，用于将 Grouped MatMul（GMM）、Unpermute 和 AllToAllv 集合通信融合执行。算子先对本地专家执行分组矩阵乘，再按 token 路由目标重新排列（unpermute），最后通过 AllToAllv 将结果发送到对应 rank。同时支持可选的 shared-expert MatMul 并行计算。

**主要应用场景**：
- 大模型 MoE（Mixture-of-Experts）场景中的 EP（Expert Parallelism）通信计算融合
- 需要将 GMM 计算、token 重排与 AllToAllv 通信流水并行的推理或训练任务
- GPT、LLaMA、DeepSeek 等模型中专家并行域的通算融合场景

**算子特征**：
- 难度等级：L4（FusedComposite）
- 支持 FLOAT16、BFLOAT16 输入输出
- `gmm_x` 为二维 ND 输入，按本地专家分组
- `gmm_weight` 支持转置 / 不转置场景
- 支持可选 `mm_x` / `mm_weight` 的 shared-expert MatMul
- 需要 HCCL 通信域，benchmark 中每个 case 会启动一个 rank 对应的进程

## 2. 算子定义

### 数学公式

$$
gmm\_out_i = \text{GroupedMatMul}(gmm\_x_i, gmm\_weight_i), \quad i = 0 \dots e-1
$$

$$
unpermute\_out = \text{Unpermute}([gmm\_out_0; \dots; gmm\_out_{e-1}])
$$

$$
y = \text{AllToAllv}(unpermute\_out)
$$

$$
mm\_y = mm\_x \times mm\_weight \quad \text{(optional)}
$$

其中：
- `gmm_x` 是当前 rank 的本地输入，shape 为 `[A, H1]`，按本地 `e` 个专家分组
- `gmm_weight` 为专家权重，不转置时 shape 为 `[e, H1, N1]`，转置时为 `[e, N1, H1]`
- `send_counts` / `recv_counts` 描述每个 (rank, expert) 对的 token 分布，长度均为 `world_size * e`
- `mm_x` / `mm_weight` 为可选 shared-expert 输入，shape 分别为 `[BS, H2]` 和 `[H2, N2]`（或转置 `[N2, H2]`）

### 计算步骤

1. 根据 `trans_gmm_weight` 判断是否对每个专家权重做转置。
2. 按 `group_list` 将 `gmm_x` 分成 `e` 组，分别与对应专家权重执行 MatMul。
3. 拼接 GMM 结果，按 token 路由目标执行 Unpermute 重排。
4. 通过 AllToAllv 将重排后的数据发送到对应 rank。
5. 当 `mm_out=True` 时，并行执行 shared-expert MatMul。

## 3. 接口规范

### 算子原型

```python
cann_bench.grouped_mat_mul_allto_allv(
    gmm_x: Tensor,
    gmm_weight: Tensor,
    hcomm_info: str,
    world_size: int,
    send_counts: List[int],
    recv_counts: List[int],
    mm_x: Optional[Tensor] = None,
    mm_weight: Optional[Tensor] = None,
    trans_gmm_weight: bool = False,
    trans_mm_weight: bool = False,
) -> Tuple[Tensor, Optional[Tensor]]
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| gmm_x | Tensor | 必选 | 当前 rank 的本地 GMM 输入，shape 为 `[A, H1]`，dtype 为 float16 / bfloat16 |
| gmm_weight | Tensor | 必选 | 专家权重。`trans_gmm_weight=False` 时 shape 为 `[e, H1, N1]`；`trans_gmm_weight=True` 时 shape 为 `[e, N1, H1]` |
| hcomm_info | str | 必选 | HCCL 通信域名称，由通信库接口获取；benchmark runner 会在每个 rank 初始化后传入 |
| world_size | int | 必选 | 通信域内 rank 数 |
| send_counts | List[int] | 必选 | 发送计数，长度 `world_size * e`，描述当前 rank 发送给每个 (目标rank, 本地expert) 的 token 数 |
| recv_counts | List[int] | 必选 | 接收计数，长度 `world_size * e`，描述当前 rank 接收的每个 global expert 的 token 数 |
| mm_x | Tensor? | None | 可选 shared-expert 输入，shape 为 `[BS, H2]` |
| mm_weight | Tensor? | None | 可选 shared-expert 权重，`trans_mm_weight=False` 时 shape 为 `[H2, N2]`；转置时为 `[N2, H2]` |
| trans_gmm_weight | bool | False | 是否按转置形式使用 `gmm_weight` |
| trans_mm_weight | bool | False | 是否按转置形式使用 `mm_weight` |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| y | `[BSK, N1]` | 与 `gmm_x` 相同 | AllToAllv 后的最终输出 |
| mm_y | `[BS, N2]` 或 None | 与 `mm_x` 相同 | 可选 shared-expert MatMul 输出；`mm_out=False` 时为 None |

### 数据类型

| gmm_x dtype | gmm_weight dtype | mm_x dtype | 输出 dtype |
|-------------|------------------|------------|------------|
| float16 | float16 | float16 / None | float16 |
| bfloat16 | bfloat16 | bfloat16 / None | bfloat16 |

### 规则与约束

- `gmm_x` 和 `gmm_weight` 必须是二维 / 三维 ND Tensor，且 dtype 一致。
- `send_counts` 和 `recv_counts` 长度必须等于 `world_size * e`。
- `send_counts` 之和必须等于 `gmm_x.shape[0]`。
- `gmm_weight` 的第 0 维必须等于 `e`（每个卡上的专家数）。
- `trans_gmm_weight` / `trans_mm_weight` 仅支持两根轴转置情况下的非连续 Tensor。
- `hcomm_info` 必须来自当前 HCCL 通信域，所有 rank 必须使用同一个通信域。
- 当前 benchmark case 使用 `world_size=8`，来源于 `mc2_test/excel/aclnnGroupedMatMulAlltoAllv.xlsx` 的 `level0` sheet。

### 支持范围

| 维度 / 参数 | 范围 | 备注 |
|---|---|---|
| `A` | 动态生成 | 由 token 分布算法根据 seed 和原始 shape 生成，约为原始 shape[0] 的 `e / world_size` 比例 |
| `H1` | 2048 ~ 7168 | cases.csv 实测范围 |
| `N1` | 2048 ~ 7168 | cases.csv 实测范围 |
| `e` | 4 | 当前 benchmark case 固定为每卡 4 个专家 |
| `world_size` | 8 | 当前 benchmark case 固定为 8 卡 |
| `trans_gmm_weight` | True / False | cases.csv 覆盖两种场景 |
| `trans_mm_weight` | True / False | cases.csv 覆盖两种场景 |
| `mm_out` | True / False | cases.csv 覆盖有无 shared-expert 场景 |
| dtype | float16 / bfloat16 | 输入与输出 dtype 保持一致 |

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


def grouped_mat_mul_allto_allv(
    gmm_x: torch.Tensor,
    gmm_weight: torch.Tensor,
    hcomm_info: str,
    world_size: int,
    send_counts,
    recv_counts,
    mm_x=None,
    mm_weight=None,
    trans_gmm_weight: bool = False,
    trans_mm_weight: bool = False,
):
    """
    GroupedMatMulAlltoAllv 简化 Golden 参考实现。

    benchmark runner 会负责初始化 HCCL 通信域、生成 token 分布并传入 hcomm_info；
    Golden 侧使用 torch.distributed 的 all_to_all_single 构造参考结果。
    """
    e = gmm_weight.shape[0]
    device = gmm_x.device

    # 1. grouped matmul
    gmm_x_f = gmm_x.float()
    B_list = list(torch.unbind(gmm_weight.float(), dim=0))
    group_list = [sum(send_counts[i * e:(i + 1) * e]) for i in range(world_size)]
    A_groups = torch.split(gmm_x_f, group_list, dim=0)
    results = []
    for a, b in zip(A_groups, B_list):
        if trans_gmm_weight:
            b = b.t()
        results.append(torch.matmul(a, b))
    gmm_out = torch.cat(results, dim=0).to(gmm_x.dtype)

    # 2. unpermute (简化示意：实际需按 expTokenNums 重排)
    # 3. all_to_all_single
    output_splits = [sum(recv_counts[i * e:(i + 1) * e]) for i in range(world_size)]
    output = torch.empty((sum(output_splits), gmm_out.shape[1]), dtype=gmm_out.dtype, device=device)
    dist.all_to_all_single(output, input=gmm_out, output_split_sizes=output_splits, input_split_sizes=group_list)

    # 4. optional mm
    mm_y = None
    if mm_x is not None and mm_weight is not None:
        w = mm_weight.t() if trans_mm_weight else mm_weight
        mm_y = torch.matmul(mm_x.float(), w.float()).to(mm_x.dtype)

    return output, mm_y
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

world_size = 8
e = 4
A, H1, N1 = 2048, 7168, 4096

gmm_x = torch.randn(A, H1, dtype=torch.bfloat16, device="npu")
gmm_weight = torch.randn(e, H1, N1, dtype=torch.bfloat16, device="npu")
send_counts = [64] * (e * world_size)
recv_counts = [64] * (e * world_size)
hcomm_info = "<hccl-comm-name>"

y, mm_y = cann_bench.grouped_mat_mul_allto_allv(
    gmm_x,
    gmm_weight,
    hcomm_info,
    world_size,
    send_counts,
    recv_counts,
    mm_x=None,
    mm_weight=None,
    trans_gmm_weight=False,
    trans_mm_weight=False,
)

# y.shape: [sum(recv_counts) / e * world_size 分配后的结果, N1]
# mm_y: None
```

### benchmark 说明

该任务不是单进程算子用例。case 中设置了 `attrs.mc2_distributed: true`，cann-bench 会为每个 case 启动 `world_size` 个 rank 进程，初始化 HCCL 通信域后再调用候选实现。候选源码包需要暴露 `cann_bench.grouped_mat_mul_allto_allv` 或 `torch.ops.cann_bench.grouped_mat_mul_allto_allv`。

由于 token 分布依赖于随机 seed，cann-bench 的 runner 会根据 case 的 `seed`、`world_size` 和 `e` 动态生成 `send_counts` / `recv_counts`，并据此调整 `gmm_x` 的实际行数。精度对比时，Golden 与 Candidate 使用同一套 token 分布数据。
