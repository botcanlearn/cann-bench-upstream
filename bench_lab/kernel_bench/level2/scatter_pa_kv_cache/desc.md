# ScatterPaKvCache 算子 API 描述

## 1. 算子简介

更新 KvCache 中指定位置的 key 和 value。该算子将推理过程中生成的 key/value token 按照 slotMapping 索引写入到分块（Paged）KV Cache 中。当前测试集覆盖 PA_NZ (FRACTAL_NZ) 格式。

**产品支持情况**：

| 产品 | 是否支持 |
|------|:--------:|
| Ascend 950PR/Ascend 950DT | √ |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | × |

**算子特征**：
- 难度等级：L2（Attention）
- 5 输入（key, key_cache, slot_mapping, value, value_cache），2 输出（key_cache, value_cache）
- 1 个属性参数（cache_mode）
- in-place 更新 cache 张量

## 2. 算子定义

### 功能说明

PA_NZ 场景下，对于每个 token $i$，根据 slotMapping 索引将 key[i] 和 value[i] 写入 FRACTAL_NZ 格式的 cache 对应位置。

```
key:       [batch * seq_len, num_head, k_head_size]
value:     [batch * seq_len, num_head, v_head_size]
keyCache:  [num_blocks, num_head * k_head_size // last_dim_k, block_size, last_dim_k]
valueCache:[num_blocks, num_head * v_head_size // last_dim_v, block_size, last_dim_v]
slotMapping:[batch * seq_len]
cacheMode: "PA_NZ"

last_dim_k = 32 / sizeof(dtypeKey)
last_dim_v = 32 / sizeof(dtypeValue)
(k_head_size * sizeof(dtypeKey)) % 32 == 0
(v_head_size * sizeof(dtypeValue)) % 32 == 0
```

key 和 value 的 dtype 必须一致。

### 数学公式

$$
\text{block\_index} = \lfloor \text{slot\_mapping}[i] / \text{block\_size} \rfloor
$$

$$
\text{block\_offset} = \text{slot\_mapping}[i] \mod \text{block\_size}
$$

将 key[i] reshape 为一维后按 lastDim 切片写入：

$$
\text{key\_cache}[\text{block\_index}][k][\text{block\_offset}][:] = \text{key\_flat}[k \cdot \text{lastDim} : (k+1) \cdot \text{lastDim}]
$$

其中 lastDim 为 NZ 格式最后一维大小（float16/bfloat16 为 16，int8 为 32）。slot_mapping[i] < 0 时跳过该 token 不写入。

## 3. 接口规范

### 算子原型

```python
cann_bench.scatter_pa_kv_cache(Tensor key, Tensor key_cache, Tensor slot_mapping, Tensor value, Tensor value_cache, string cache_mode="PA_NZ") -> (Tensor key_cache, Tensor value_cache)
```

### 输入参数说明

| 参数名 | 输入/输出 | 描述 | 使用说明 | 数据类型 | 数据格式 | 维度(shape) | 非连续Tensor |
|--------|----------|------|---------|---------|---------|------------|-------------|
| key | 输入 | 待更新的key值，当前step多个token的key | 3D Tensor [batch*seq_len, num_head, k_head_size] | FLOAT16/BFLOAT16/INT8 | ND | 3 | 不支持 |
| key_cache | 输入/输出 | 需要更新的key cache | 4D Tensor [numBlocks, NH*HS/lastDim, blockSize, lastDim]，FRACTAL_NZ排布 | 与key一致 | ND | 4 | 不支持 |
| slot_mapping | 输入 | 每个token在cache中的存储偏移 | 1D Tensor [batch*seq_len]，值范围 [0, numBlocks*blockSize-1] | INT32/INT64 | ND | 1 | 不支持 |
| value | 输入 | 待更新的value值，当前step多个token的value | 3D Tensor [batch*seq_len, num_head, v_head_size] | 与key一致 | ND | 3 | 不支持 |
| value_cache | 输入/输出 | 需要更新的value cache | 4D Tensor [numBlocks, NH*VHS/lastDim, blockSize, lastDim]，FRACTAL_NZ排布 | 与key一致 | ND | 4 | 不支持 |
| cache_mode | 属性 | keyCacheRef和valueCacheRef的内存排布格式 | 固定为"PA_NZ" | STRING | - | - | - |

### 输出

| 参数名 | 输入/输出 | 描述 | 使用说明 | 数据类型 | 数据格式 | 维度(shape) | 非连续Tensor |
|--------|----------|------|---------|---------|---------|------------|-------------|
| key_cache | 输出 | 更新后的key cache张量 | shape 和 dtype 与输入 key_cache 一致 | 与key一致 | ND | 4 | 不支持 |
| value_cache | 输出 | 更新后的value cache张量 | shape 和 dtype 与输入 value_cache 一致 | 与key一致 | ND | 4 | 不支持 |

### 数据类型

PA_NZ 场景 Atlas A2/A3 支持的数据类型：

| key dtype | key_cache dtype | slot_mapping dtype | value dtype | value_cache dtype |
|-----------|----------------|-------------------|-------------|-------------------|
| float16 | float16 | int32/int64 | float16 | float16 |
| bfloat16 | bfloat16 | int32/int64 | bfloat16 | bfloat16 |
| int8 | int8 | int32/int64 | int8 | int8 |

### 规则与约束

- key、value、keyCacheRef、valueCacheRef 的数据类型必须一致
- slotMapping 的取值范围 [0, num_blocks * block_size - 1]，且 slotMapping 内的元素值保证不重复，重复时不保证正确性
- key 和 value 的前两维 shape 必须相同
- keyCacheRef 和 valueCacheRef 的倒数第二维（blockSize）必须小于 UINT16_MAX
- k_head_size * sizeof(dtype) 和 v_head_size * sizeof(dtype) 必须 32 字节对齐
- last_dim_k = 32 / sizeof(dtypeKey)，last_dim_v = 32 / sizeof(dtypeValue)

### 支持范围

| 维度 / 参数 | 范围 | 备注 |
|---|---|---|
| numTokens（batch * seq_len） | 1 ~ 32708 | key/value/slot_mapping 的第 0 维 |
| numHeads | 1 ~ 257 | 注意力头数 |
| k_head_size | 16 ~ 288 | key 每头维度，需 32 字节对齐 |
| v_head_size | 16 ~ 288 | value 每头维度，需 32 字节对齐 |
| numBlocks | 1 ~ 13271 | cache block 数 |
| blockSize | 2 ~ 257 | 每 block 的 token 容量，需 < UINT16_MAX |
| key/value dtype | float16, bfloat16, int8 | Atlas A2/A3 |
| slot_mapping dtype | int32, int64 | 索引类型 |

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

| 数据类型 | FLOAT16 | BFLOAT16 | INT8 |
|----------|---------|----------|------|
| **通过阈值(Threshold)** | 2^-10 | 2^-7 | 精确匹配 |

由于 ScatterPaKvCache 是纯数据搬运操作（无计算），理论上应精确匹配（zero tolerance）。

## 5. 标准 Golden 代码

```python
import copy
import torch

def scatter_pa_kv_cache(
    key: torch.Tensor,
    key_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    value: torch.Tensor,
    value_cache: torch.Tensor,
    cache_mode: str = "PA_NZ",
) -> tuple:
    key_cache_golden = copy.deepcopy(key_cache)
    value_cache_golden = copy.deepcopy(value_cache)

    block_size = key_cache.shape[2]
    lastDim_k = key_cache.shape[3]
    lastDim_v = value_cache.shape[3]
    num_head = key.shape[1]
    k_head_size = key.shape[2]
    v_head_size = value.shape[2]

    for i, slot in enumerate(slot_mapping):
        if slot < 0:
            continue
        block_index = slot // block_size
        block_offset = slot % block_size

        token_key = key[i].reshape(num_head * k_head_size)
        for k in range(num_head * k_head_size // lastDim_k):
            key_cache_golden[block_index][k][block_offset][:] = token_key[k * lastDim_k: k * lastDim_k + lastDim_k]

        token_value = value[i].reshape(num_head * v_head_size)
        for v in range(num_head * v_head_size // lastDim_v):
            value_cache_golden[block_index][v][block_offset][:] = token_value[v * lastDim_v: v * lastDim_v + lastDim_v]

    return key_cache_golden, value_cache_golden
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

key = torch.randn(16, 8, 128, dtype=torch.bfloat16, device="npu")
key_cache = torch.randn(17, 64, 8, 16, dtype=torch.bfloat16, device="npu")
slot_mapping = torch.arange(16, dtype=torch.int32, device="npu")
value = torch.randn(16, 8, 128, dtype=torch.bfloat16, device="npu")
value_cache = torch.randn(17, 64, 8, 16, dtype=torch.bfloat16, device="npu")

key_cache_out, value_cache_out = cann_bench.scatter_pa_kv_cache(
    key, key_cache, slot_mapping, value, value_cache,
    cache_mode="PA_NZ"
)
```
