# AdvanceStepV2 算子 API 描述

## 1. 算子简介

vLLM是一个高性能的LLM推理和服务框架，专注于优化大规模语言模型的推理效率。它的核心特点包括PageAttention和高效内存管理。advance_step算子的主要作用是推进推理步骤，即在每个生成步骤中更新模型的状态并生成新的inputTokens、inputPositions、seqLens和slotMapping，为vLLM的推理提升效率。

**主要应用场景**：
- 深度学习自定义算子
- 模型训练与推理
- 特定领域计算加速

**算子特征**：
- 难度等级：L2（IndexGather）
- 8 输入，4 个 in-place 输出，3 个属性参数
- 支持 ND 格式输入
- in-place 输出：`input_tokens`、`input_positions`、`seq_lens`、`slot_mapping`

## 2. 算子定义

### 数学公式

设：
- `num_seqs = B`：序列数量
- `spec_token` 的列数为 `S`，则每个请求处理的 token 数为：

$$
token\_each\_req = 1 + S
$$

**Step 1: 更新 input_positions**

$$
input\_positions \mathrel{+}= \text{repeat\_interleave}(accepted\_num, token\_each\_req) + 1
$$

**Step 2: 更新 seq_lens**

$$
seq\_lens = input\_positions + 1
$$

**Step 3: 从 sampled_token_ids 中取出每个序列的最后一个有效 token**

$$
index = \arg\min(\text{concat}(sampled\_token\_ids, -1_{(B,1)}), \text{dim}=1) - 1
$$

$$
last\_tokens = sampled\_token\_ids[arange(B), index]
$$

**Step 4: 更新 input_tokens**

- 当 `token_each_req == 1` 时：

$$
input\_tokens[:B] = last\_tokens
$$

- 当 `token_each_req > 1` 时，将 `input_tokens` 按 `[B, token_each_req]` 视图展开：

$$
input\_tokens\_2d[:B, 0] = last\_tokens
$$

$$
input\_tokens\_2d[:B, 1:] = spec\_token
$$

**Step 5: 计算 slot_mapping**

$$
req\_indices = \text{repeat\_interleave}(arange(B), token\_each\_req)
$$

$$
max\_num\_blocks\_per\_req = block\_tables.shape[1]
$$

$$
block\_tables\_indices = req\_indices \times max\_num\_blocks\_per\_req + \left\lfloor \frac{input\_positions}{block\_size} \right\rfloor
$$

$$
block\_numbers = block\_tables.flatten()[block\_tables\_indices]
$$

$$
block\_offset = input\_positions \bmod block\_size
$$

$$
slot\_mapping = block\_numbers \times block\_size + block\_offset
$$


## 3. 接口规范

### 算子原型

```python
cann_bench.advance_step_v2(Tensor input_tokens, Tensor sampled_token_ids, Tensor input_positions, Tensor seq_lens, Tensor slot_mapping, Tensor block_tables, Tensor spec_token, Tensor accepted_num, int64 num_seqs, int64 num_queries, int64 block_size) -> (Tensor input_tokens, Tensor input_positions, Tensor seq_lens, Tensor slot_mapping)
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| input_tokens | Tensor | 必选 | 输入张量 `input_tokens` |
| sampled_token_ids | Tensor | 必选 | 输入张量 `sampled_token_ids` |
| input_positions | Tensor | 必选 | 输入张量 `input_positions` |
| seq_lens | Tensor | 必选 | 输入张量 `seq_lens` |
| slot_mapping | Tensor | 必选 | 输入张量 `slot_mapping` |
| block_tables | Tensor | 必选 | 输入张量 `block_tables` |
| spec_token | Tensor | 必选 | 输入张量 `spec_token` |
| accepted_num | Tensor | 必选 | 输入张量 `accepted_num` |
| num_seqs | int64 | 必选 | 属性 `num_seqs` |
| num_queries | int64 | 必选 | 属性 `num_queries` |
| block_size | int64 | 必选 | 属性 `block_size` |


### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| input_tokens | 与输入 `input_tokens` 相同 | int64 | 更新后的 input_tokens（in-place） |
| input_positions | 与输入 `input_positions` 相同 | int64 | 更新后的 input_positions（in-place） |
| seq_lens | 与输入 `seq_lens` 相同 | int64 | 更新后的 seq_lens（in-place） |
| slot_mapping | 与输入 `slot_mapping` 相同 | int64 | 更新后的 slot_mapping（in-place） |### 数据类型

| 输入 dtype | 输出 dtype |
|-----------|-----------|
| int64 | int64 |

### 规则与约束

- **inputTokens**（输入/输出，INT64，ND）：公式中的输入/输出inputTokens。
- **sampledTokenIds**（输入，INT64，ND）：公式中的输入sampledTokenIds。
- **inputPositions**（输入/输出，INT64，ND）：公式中的输入/输出inputPositions。
- **seqLens**（输入/输出，INT64，ND）：公式中的输入/输出seqLens。
- **slotMapping**（输入/输出，INT64，ND）：公式中的输入/输出slotMapping。
- **blockTables**（输入，INT，ND）：公式中的输入blockTables。
- **numSeqs**（INT）：记录输入的seq数量，大小与seqLens的长度一致。取值范围是大于0的正整数。numSeqs的值大于输入numQueries的值。
- **numQueries**（INT）：记录输入的Query的数量，大小与sampledTokenIds第一维的长度一致。取值范围是大于0的正整数。
- **blockSize**（INT64）：每个block的大小。取值范围是大于0的正整数。

### 支持范围

输入 tensor 各维度与参数的支持范围：

| 维度 / 参数 | 范围 | 备注 |
|---|---|---|
| `ndim`（输入维度数） | 1 ~ 2 | cases 实测范围 |
| `dim_0`（第0维大小） | 1 ~ 1136125 | cases 实测范围 |
| `dim_1`（第1维大小） | 1 ~ 10000 | cases 实测范围 |
| `dtype` | int64 | cases 实测覆盖 |
| `num_seqs` | 1 ~ 9664 | cases 实测范围 |
| `num_queries` | 1 ~ 9664 | cases 实测范围 |
| `block_size` | 8 ~ 16 | cases 实测范围 |

## 4. 精度要求

采用[生态算子精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md)进行验证。

**误差指标**：

1. 平均相对误差（MERE）：采样点中相对误差平均值

   $$
   \text{MERE} = \text{avg}(\frac{\text{abs}(actual - golden)}{\text{abs}(golden)+\text{1e-7}})
   $$

2. 最大相对误差（MARE）：采样点中相对误差最大值

   $$
   \text{MARE} = \max(\frac{\text{abs}(actual - golden)}{\text{abs}(golden)+\text{1e-7}})
   $$

**通过标准**：

| 数据类型 | FLOAT16 | BFLOAT16 | FLOAT32 | HiFLOAT32 | FLOAT8 E4M3 | FLOAT8 E5M2 |
|----------|---------|----------|---------|-----------|-------------|-------------|
| **通过阈值(Threshold)** | 2^-10 | 2^-7 | 2^-13 | 2^-11 | 2^-3 | 2^-2 |

当平均相对误差 MERE < Threshold，最大相对误差 MARE < 10 * Threshold 时判定为通过。

## 5. 标准 Golden 代码

```python
import torch

def advance_step_v2(num_seqs: int, num_queries: int, block_size: int,
                    input_tokens: torch.Tensor, sampled_token_ids: torch.Tensor,
                    input_positions: torch.Tensor, seq_lens: torch.Tensor,
                    slot_mapping: torch.Tensor, block_tables: torch.Tensor,
                    spec_token: torch.Tensor, accepted_num: torch.Tensor):
    token_each_reqs = 1 + len(spec_token[0])
    input_positions += torch.repeat_interleave(accepted_num, token_each_reqs) + 1
    seq_lens.copy_((input_positions + 1).to(seq_lens.dtype))
    index = torch.argmin(
        torch.cat([
            sampled_token_ids,
            torch.full((num_seqs, 1), -1, device=sampled_token_ids.device)
        ], dim=1),
        dim=1
    ) - 1
    last_tokens = sampled_token_ids[torch.arange(num_seqs), index]
    if token_each_reqs == 1:
        input_tokens[:num_seqs] = last_tokens.to(dtype=input_tokens.dtype)
    else:
        input_tokens_2d = input_tokens.view(-1, token_each_reqs)
        input_tokens_2d[:num_seqs, 0] = last_tokens
        input_tokens_2d[:num_seqs, 1:] = spec_token
    req_indices = torch.repeat_interleave(
        torch.arange(num_seqs),
        token_each_reqs,
        dim=0
    )
    max_num_blocks_per_req = block_tables.shape[1]
    block_tables_indices = (
        req_indices * max_num_blocks_per_req +
        input_positions // block_size
    )
    block_numbers = block_tables.flatten()[block_tables_indices]
    block_offset = input_positions % block_size
    slot_mapping.copy_(block_numbers * block_size + block_offset)
    return input_tokens, input_positions, seq_lens, slot_mapping
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

input_tokens = torch.randn(16956, dtype=torch.int64, device="npu")
sampled_token_ids = torch.randn(314, 54, dtype=torch.int64, device="npu")
input_positions = torch.randn(16956, dtype=torch.int64, device="npu")
seq_lens = torch.randn(16956, dtype=torch.int64, device="npu")
slot_mapping = torch.randn(16956, dtype=torch.int64, device="npu")
block_tables = torch.randn(314, 10000, dtype=torch.int64, device="npu")
spec_token = torch.randn(314, 53, dtype=torch.int64, device="npu")
accepted_num = torch.randn(314, dtype=torch.int64, device="npu")
(input_tokens, input_positions, seq_lens, slot_mapping) = cann_bench.advance_step_v2(input_tokens, sampled_token_ids, input_positions, seq_lens, slot_mapping, block_tables, spec_token, accepted_num, num_seqs=314, num_queries=314, block_size=8)
```
