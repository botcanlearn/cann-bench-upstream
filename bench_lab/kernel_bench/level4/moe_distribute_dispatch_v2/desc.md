# MoeDistributeDispatchV2 算子 API 描述

## 1. 算子简介

MoeDistributeDispatchV2 是一个 MC2 通算融合算子，用于 MoE（Mixture-of-Experts）场景中的 token dispatch 阶段。算子对本地 token 数据执行量化（可选），在 EP（Expert Parallelism）域执行 AllToAllV 通信；当存在 TP（Tensor Parallelism）域时，额外执行 AllGatherV 通信。输出扩展后的 token 特征、动态量化参数以及供 `MoeDistributeCombineV2` 使用的辅助信息。

**主要应用场景**：
- 大模型 MoE 推理/训练中的 token dispatch 阶段
- 需要将量化、token 扩展与 AllToAllV / AllGatherV 通信融合执行的场景
- 与 `MoeDistributeCombineV2` 配套使用的 EP+TP 分布式 MoE 流水线

**算子特征**：
- 难度等级：L4（FusedComposite）
- 支持 FLOAT16、BFLOAT16 输入输出（非量化场景）
- `x` 为二维 ND 输入，`expert_ids` 为二维 INT32 输入
- 支持可选的 `x_active_mask`、`expert_scales`、`elastic_info` 等输入
- 需要 HCCL 通信域，benchmark 中每个 case 会启动一个 rank 对应的进程
- 必须与 `MoeDistributeCombineV2` 或 `MoeDistributeCombineAddRmsNorm` 配套使用

## 2. 算子定义

### 数学公式

非量化场景（`quant_mode = 0`）：

$$
allToAllXOut = AllToAllV(x)
$$

$$
expandXOut =
\begin{cases}
AllToAllV(x), & \text{无 TP 通信域} \\
AllGatherV(allToAllXOut), & \text{有 TP 通信域}
\end{cases}
$$

动态量化场景（`quant_mode = 2`）：

$$
xFp32 = CastToFp32(x) \times scales \\
dynamicScales = dstTypeMax / Max(Abs(xFp32)) \\
quantOut = CastToInt8(xFp32 \times dynamicScales) \\
allToAllXOut = AllToAllV(quantOut) \\
expandXOut =
\begin{cases}
AllToAllV(quantOut), & \text{无 TP} \\
AllGatherV(allToAllXOut), & \text{有 TP}
\end{cases}
$$

其中：
- `x` 是当前 rank 的本地输入，shape 为 `[BS, H]`
- `expert_ids` 是每个 token 的 topK 专家索引，shape 为 `[BS, K]`
- `A` 表示本卡需要分发的最大 token 数量，由 `bs`、`ep_world_size`、`shared_expert_num`、`shared_expert_rank_num`、`moe_expert_num`、`k` 等共同决定
- `tp_world_size = 0` 或 `1` 时表示无 TP 域通信

### 计算步骤

1. 根据 `quant_mode` 判断是否对 `x` 执行量化。
2. 在 EP 通信域内执行 AllToAllV，将 token 按 expert 路由目标发送到对应 rank。
3. 当 `tp_world_size > 1` 时，在 TP 通信域内执行 AllGatherV。
4. 输出扩展后的 token 特征 `expand_x`、动态量化参数 `dynamic_scales`、以及辅助信息 `assist_info_for_combine`、`expert_token_nums`、`ep_recv_counts`、`tp_recv_counts`、`expand_scales`。

## 3. 接口规范

### 算子原型

```python
cann_bench.moe_distribute_dispatch_v2(
    x: Tensor,
    expert_ids: Tensor,
    hcomm_info: str,
    world_size: int,
    moe_expert_num: int,
    tp_world_size: int = 0,
    shared_expert_num: int = 1,
    shared_expert_rank_num: int = 0,
    quant_mode: int = 0,
    activeMask_Dim: int = 0,
    is_combine_x: int = 0,
    comm_mode: int = 0,
    is_elastic: int = 0,
    zero_expert_num: int = 0,
    copy_expert_num: int = 0,
    const_expert_num: int = 0,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| x | Tensor | 必选 | 当前 rank 的本地 token 输入，shape 为 `[BS, H]`，dtype 为 float16 / bfloat16 |
| expert_ids | Tensor | 必选 | 每个 token 的 topK 专家索引，shape 为 `[BS, K]`，dtype 为 int32 |
| hcomm_info | str | 必选 | HCCL EP 通信域名称，由通信库接口获取；benchmark runner 会在每个 rank 初始化后传入 |
| world_size | int | 必选 | EP 通信域内 rank 数（`ep_world_size`） |
| moe_expert_num | int | 必选 | MoE 专家总数量 |
| tp_world_size | int | 0 | TP 通信域大小。0 或 1 表示无 TP 域通信 |
| shared_expert_num | int | 1 | 共享专家数量 |
| shared_expert_rank_num | int | 0 | 共享专家占用 rank 数量 |
| quant_mode | int | 0 | 量化模式：0 表示非量化，2 表示 pertoken 动态量化 |
| activeMask_Dim | int | 0 | `x_active_mask` 维度：0 表示不传，1 表示 1D `[BS,]`，2 表示 2D `[BS, K]` |
| is_combine_x | int | 0 | 是否传入 `expert_scales`（combine 阶段使用的权重） |
| comm_mode | int | 0 | 通信算法模式：0 表示默认（空字符串），2 表示 hierarchy |
| is_elastic | int | 0 | 是否传入 `elastic_info` |
| zero_expert_num | int | 0 | zero expert 数量 |
| copy_expert_num | int | 0 | copy expert 数量 |
| const_expert_num | int | 0 | const expert 数量 |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| expand_x | `[max(tp_world_size,1)*A, H]` | 与 `x` 相同 | 扩展后的 token 特征 |
| dynamic_scales | `[A]` 或 `[A, ceil(H/128)]` | float32 | 动态量化缩放参数（quant_mode=0 时 shape 可能为 `[A]`） |
| assist_info_for_combine | `[A*128, ]` | int32 | 供 CombineV2 使用的辅助信息 |
| expert_token_nums | `[localExpertNum, ]` | int64 | 每个本地专家收到的 token 数量 |
| ep_recv_counts | `[ep_world_size*max(tp_world_size,1)*localExpertNum, ]` | int32 | 从 EP 域各 rank 接收的 token 数 |
| tp_recv_counts | `[tp_world_size, ]` | int32 | 从 TP 域各 rank 接收的 token 数（无 TP 时可能为空） |
| expand_scales | `[A, ]` | float32 | 扩展后的专家权重（is_combine_x=1 时有效） |

### 数据类型

| x dtype | expert_ids dtype | 输出 dtype |
|---------|------------------|------------|
| float16 | int32 | float16 |
| bfloat16 | int32 | bfloat16 |

### 规则与约束

- `x` 必须是二维 ND Tensor，dtype 支持 float16 / bfloat16。
- `expert_ids` 必须是二维 ND Tensor，dtype 为 int32，shape 第 1 维 `K` 需满足 `0 < K <= moe_expert_num`。
- `ep_rank_id` 取值范围为 `[0, ep_world_size)`。
- `shared_expert_rank_num` 需满足：`shared_expert_rank_num % shared_expert_num == 0`（当 `shared_expert_rank_num > 0` 时）。
- `moe_expert_num % (ep_world_size - shared_expert_rank_num) == 0`。
- 当前 benchmark case 均使用 `world_size=8`（来源于 ops-transformer UT `Test_dispatch_combine_16die_V2`）。
- `hcomm_info` 必须来自当前 HCCL 通信域，所有 rank 必须使用同一个 EP 通信域。

### 支持范围

| 维度 / 参数 | 范围 | 备注 |
|---|---|---|
| `BS` | 1 ~ 512 | cases.csv 实测范围 |
| `H` | 1024 ~ 8192 | cases.csv 实测范围，需为 32 的整数倍 |
| `K` | 1 ~ 16 | cases.csv 实测范围，需满足 `K <= moe_expert_num` |
| `world_size` | 8 | 当前 benchmark case 固定为 8 卡 |
| `tp_world_size` | 0 / 1 / 2 | 0/1 表示无 TP，2 表示有 TP 通信 |
| `moe_expert_num` | 2 ~ 512 | cases.csv 实测范围 |
| `shared_expert_num` | 0 ~ 3 | 0 表示无共享专家 |
| `shared_expert_rank_num` | 0 ~ 6 | 需满足 `shared_expert_rank_num % shared_expert_num == 0` |
| `quant_mode` | 0 | 当前 cases.csv 仅覆盖非量化场景 |
| `activeMask_Dim` | 0 / 1 / 2 | 0=None, 1=1D, 2=2D |
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


def moe_distribute_dispatch_v2(
    x: torch.Tensor,
    expert_ids: torch.Tensor,
    hcomm_info: str,
    world_size: int,
    moe_expert_num: int,
    tp_world_size: int = 0,
    shared_expert_num: int = 1,
    shared_expert_rank_num: int = 0,
    quant_mode: int = 0,
    activeMask_Dim: int = 0,
    is_combine_x: int = 0,
    comm_mode: int = 0,
    is_elastic: int = 0,
    zero_expert_num: int = 0,
    copy_expert_num: int = 0,
    const_expert_num: int = 0,
):
    """
    MoeDistributeDispatchV2 简化 Golden 参考实现。

    benchmark runner 会负责初始化 HCCL 通信域、生成 expert 路由并传入 hcomm_info；
    Golden 侧使用 torch.distributed 的 all_to_all_single / all_gather 构造参考结果。
    """
    # TODO: Implement full multi-rank golden reference with token routing,
    #       AllToAllV, optional AllGatherV, and auxiliary output generation.
    raise NotImplementedError(
        "MoeDistributeDispatchV2 golden reference is not yet implemented. "
        "The real benchmark path uses MC2DistributedEvaluator with HCCL."
    )
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

world_size = 8
bs, h, k = 32, 7168, 8
moe_expert_num = 256

x = torch.randn(bs, h, dtype=torch.bfloat16, device="npu")
expert_ids = torch.randint(0, moe_expert_num, (bs, k), dtype=torch.int32, device="npu")
hcomm_info = "<hccl-comm-name>"

expand_x, dynamic_scales, assist_info, expert_token_nums, ep_recv_counts, tp_recv_counts, expand_scales = (
    cann_bench.moe_distribute_dispatch_v2(
        x,
        expert_ids,
        hcomm_info,
        world_size,
        moe_expert_num,
        tp_world_size=0,
        quant_mode=0,
    )
)

# expand_x.shape: [A, H]
```

### benchmark 说明

该任务不是单进程算子用例。case 中设置了 `attrs.mc2_distributed: true`，cann-bench 会为每个 case 启动 `world_size` 个 rank 进程，初始化 HCCL 通信域后再调用候选实现。候选源码包需要暴露 `cann_bench.moe_distribute_dispatch_v2` 或 `torch.ops.cann_bench.moe_distribute_dispatch_v2`。

**TODO**：
- [ ] 在 `mc2_distributed_runner.py` 中新增 `_golden_moe_distribute_dispatch_v2` 多进程 Golden 实现
- [ ] 在 `mc2_distributed_runner.py` 中新增 `_call_moe_distribute_dispatch_v2` 候选调用封装
- [ ] 在 `ops-transformer/torch_extension/cann_bench/__init__.py` 中注册 `moe_distribute_dispatch_v2` torch adapter
- [ ] 补充 `golden.py` 单进程 CPU 参考实现（用于元数据检查和冒烟测试）
