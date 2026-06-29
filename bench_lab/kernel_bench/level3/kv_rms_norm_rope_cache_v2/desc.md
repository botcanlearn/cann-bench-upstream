# kv_rms_norm_rope_cache_v2 算子 API 描述

## 1. 算子简介

对输入张量(kv)的尾轴，拆分出左半边用于rms_norm计算，右半边用于rope计算，再将计算结果分别scatter到两块cache中。

**主要应用场景**：
- 深度网络归一化层
- 大模型训练稳定化
- Batch/Layer/RMS 归一化

**算子特征**：
- 难度等级：L3（Normalization）
- 12 输入，4 输出，3 个属性参数
- 可选属性：epsilon, cache_mode, is_output_kv

## 2. 算子定义

### 数学公式

**interleaveRope**

$$
x=kv[...,Dv:]
$$

$$
x1=x[...,::2]
$$

$$
x2=x[...,1::2]
$$

$$
x\_part1=torch.cat((x1,x2),dim=-1)
$$

$$
x\_part2=torch.cat((-x2,x1),dim=-1)
$$

$$
y=x\_part1*cos+x\_part2*sin
$$

**rmsNorm**

$$
x=kv[...,:Dv]
$$

$$
square\_x=x*x
$$

$$
mean\_square\_x=square\_x.mean(dim=-1,keepdim=True)
$$

$$
rms=torch.sqrt(mean\_square\_x+epsilon)
$$

$$
y=(x/rms)*gamma
$$

## 3. 接口规范

### 算子原型

```python
cann_bench.kv_rms_norm_rope_cache_v2(Tensor kv, Tensor gamma, Tensor cos, Tensor sin, Tensor index, Tensor k_cache, Tensor ckv_cache, Tensor k_rope_scale, Tensor c_kv_scale, Tensor k_rope_offset, Tensor c_kv_offset, Tensor v, float epsilon=1e-5, string cache_mode="Norm", bool is_output_kv=false) -> (Tensor k_cache, Tensor ckv_cache, Tensor k_rope, Tensor c_kv)
```

### 输入参数说明

| 参数名 | 输入/输出 | 描述 | 使用说明 | 数据类型 | 数据格式 | 维度(shape) | 非连续Tensor |
|--------|----------|------|---------|---------|---------|------------|-------------|
| kv | 输入 | 公式中用于切分出 RMSNorm 数据 Dv 和 RoPE 数据 Dk 的输入数据 | shape 仅支持 4 维 [Bkv, N, Skv, D]，D = Dk + Dk；不支持空 Tensor | FLOAT16、BFLOAT16 | ND | 4 | 支持 |
| gamma | 输入 | 公式中用于 RMSNorm 计算的权重 | 与 kv 数据类型一致，shape 为 1 维 [Dv]；不支持空 Tensor | FLOAT16、BFLOAT16 | ND | 1 | 支持 |
| cos | 输入 | 公式中用于 RoPE 计算的余弦变换数据 | 与 kv 数据类型一致，shape 为 4 维 [Bkv, 1, Skv, Dk] 或 [Bkv, 1, 1, Dk]；不支持空 Tensor | FLOAT16、BFLOAT16 | ND | 4 | 支持 |
| sin | 输入 | 公式中用于 RoPE 计算的正弦变换数据 | 与 kv 数据类型一致，shape 与 cos 保持一致；不支持空 Tensor | FLOAT16、BFLOAT16 | ND | 4 | 支持 |
| index | 输入 | 用于指定写入 cache 的具体索引位置，-1 表示跳过更新 | Norm 模式下 shape 为 2 维 [Bkv, Skv]；PA_BNSD、PA_NZ 下为 1 维 [Bkv * Skv]；PA_BLK_BNSD、PA_BLK_NZ 下为 1 维 [Bkv * ceil_div(Skv, BlockSize)] | INT64 | ND | 1-2 | 支持 |
| k_cache | 输入/输出 | 提前申请的 K cache，输入输出同地址复用 | 非量化场景 dtype 与 kv 一致；量化场景 dtype 为 INT8；PA 系列 shape 为 [BlockNum, BlockSize, N, Dk]，Norm 场景 shape 为 [Bcache, N, Scache, Dk]；不支持空 Tensor | FLOAT16、BFLOAT16、INT8 | ND | 4 | 支持 |
| ckv_cache | 输入/输出 | 提前申请的 V cache，输入输出同地址复用 | 非量化场景 dtype 与 kv 一致；量化场景 dtype 为 INT8；PA 系列 shape 为 [BlockNum, BlockSize, N, Dv]，Norm 场景 shape 为 [Bcache, N, Scache, Dv]；不支持空 Tensor | FLOAT16、BFLOAT16、INT8 | ND | 4 | 支持 |
| k_rope_scale | 输入 | K 量化缩放，k_cache 为量化类型时必填 | shape 为 2 维 [N, Dk]、1 维 [Dk] 或 [1]；不支持空 Tensor | FLOAT32 | ND | 1-2 | 支持 |
| c_kv_scale | 输入 | V 量化缩放，ckv_cache 为量化类型时必填 | shape 为 2 维 [N, Dv]、1 维 [Dv] 或 [1]；不支持空 Tensor | FLOAT32 | ND | 1-2 | 支持 |
| k_rope_offset | 输入 | K 量化偏移，非对称量化时必填 | shape 为 2 维 [N, Dk]、1 维 [Dk] 或 [1]；不支持空 Tensor | FLOAT32 | ND | 1-2 | 支持 |
| c_kv_offset | 输入 | V 量化偏移，非对称量化时必填 | shape 为 2 维 [N, Dv]、1 维 [Dv] 或 [1]；不支持空 Tensor | FLOAT32 | ND | 1-2 | 支持 |
| v | 输入 | 可选的 V 输入；为 None 时 method_mode=0（kv 拼接），否则 method_mode=1（kv 分离） | shape 为 4 维 [Bkv, N, Skv, Dv]；不支持空 Tensor | FLOAT16、BFLOAT16 | ND | 4 | 支持 |
| epsilon | 属性 | RMSNorm 计算防止除 0 | 建议设为 1e-5 | DOUBLE | - | - | - |
| cache_mode | 属性 | cache 格式的选择标记 | 类型有 Norm、PA、PA_BNSD、PA_NZ、PA_BLK_BNSD、PA_BLK_NZ，建议设为 Norm | CHAR* | - | - | - |
| is_output_kv | 属性 | k_rope 和 c_kv 输出控制标记 | 为 true 时输出 k_rope 和 c_kv，建议设为 false | BOOL | - | - | - |

### 输出

| 参数名 | 输入/输出 | 描述 | 使用说明 | 数据类型 | 数据格式 | 维度(shape) | 非连续Tensor |
|--------|----------|------|---------|---------|---------|------------|-------------|
| k_cache | 输入/输出 | 更新后的 K cache | 与输入 k_cache 同 shape、同 dtype | FLOAT16、BFLOAT16、INT8 | ND | 4 | 支持 |
| ckv_cache | 输入/输出 | 更新后的 V cache | 与输入 ckv_cache 同 shape、同 dtype | FLOAT16、BFLOAT16、INT8 | ND | 4 | 支持 |
| k_rope | 输出 | 由 is_output_kv 控制 | is_output_kv 为 true 时输出，shape 为 [Bkv, N, Skv, Dk]；否则为同 shape 全 0 | FLOAT16、BFLOAT16 | ND | 4 | 支持 |
| c_kv | 输出 | 由 is_output_kv 控制 | is_output_kv 为 true 时输出，shape 为 [Bkv, N, Skv, Dv]；否则为同 shape 全 0 | FLOAT16、BFLOAT16 | ND | 4 | 支持 |


### 数据类型

| kv dtype | gamma dtype | cos/sin dtype | index dtype | k_cache/ckv_cache dtype | k_rope/c_kv dtype |
|---------|------------|--------------|------------|------------------------|------------------|
| FLOAT16 | FLOAT16 | FLOAT16 | INT64 | FLOAT16 / BFLOAT16 / INT8 | FLOAT16 |
| BFLOAT16 | BFLOAT16 | BFLOAT16 | INT64 | FLOAT16 / BFLOAT16 / INT8 | BFLOAT16 |

### 规则与约束

- 参数说明里 shape 格式说明：
    - Bkv 为输入 kv 的 batch size，Skv 为输入 kv 的 sequence length，大小由用户输入场景决定，无明确限制。
    - N 为输入 kv 的 head number。此算子与 DeepSeekV3 网络结构强相关，仅支持 N=1 的场景，不存在 N 非 1 的场景。
    - D 为输入 kv 的 head dim。rms_norm 计算所需数据 Dv 和 RoPE 计算所需数据 Dk 由输入 kv 的 D 切分而来。故 Dk、Dv 大小需满足 Dk + Dv = D。同时，Dk 需满足 rope 规则。根据 rope 规则，Dk 为偶数。若 cache_mode 为 NZ 场景（cache_mode 为 PA_NZ、PA_BLK_NZ），Dk、Dv 需 32B 对齐。
    - 若 cache_mode 为 PA 场景（cache_mode 为 PA、PA_BNSD、PA_NZ、PA_BLK_BNSD、PA_BLK_NZ），BlockSize 需 32B 对齐。
    - 关于上述 32B 对齐的情形，对齐值由 cache 的数据类型决定。以 BlockSize 为例，若 cache 的数据类型为 INT8，则需 BlockSize % 32 = 0；若 cache 的数据类型为 FLOAT16 或 BFLOAT16，则需 BlockSize % 16 = 0；若 k_cache 与 ckv_cache 的 dtype 不一致，BlockSize 需同时满足 BlockSize % 32 = 0 和 BlockSize % 16 = 0。
    - Bcache 为输入 cache 的 batch size，Scache 为输入 cache 的 sequence length，大小由用户输入场景决定，无明确限制。
    - BlockNum 为写入 cache 的内存块数，大小由用户输入场景决定，无明确限制。
- index 相关约束：
    - 当 cache_mode 为 Norm 时，shape 为 2 维 [Bkv, Skv]，要求 index 的 value 值范围为 [-1, Scache)。不同的 Bkv 下，value 数值可以重复。
    - 当 cache_mode 为 PA_BNSD、PA_NZ 时，shape 为 1 维 [Bkv * Skv]，要求 index 的 value 值范围为 [-1, BlockNum * BlockSize)。value 数值不能重复。
    - 当 cache_mode 为 PA_BLK_BNSD、PA_BLK_NZ 时，shape 为 1 维 [Bkv * ceil_div(Skv, BlockSize)]，要求 index 的 value 的数值范围为 [-1, BlockNum * BlockSize)。value / BlockSize 的值不能重复。
- 量化场景的相关约束：
    - 量化场景支持的情况 1：k_cache 的数据类型为 FLOAT16 或 BFLOAT16，ckv_cache 的数据类型为 INT8。
    - 量化场景支持的情况 2：ckv_cache 的数据类型为 FLOAT16 或 BFLOAT16，k_cache 的数据类型为 INT8。
    - 量化场景支持的情况 3：k_cache 与 ckv_cache 的数据类型一致，为 INT8。

### 支持范围

输入 tensor 各维度与参数的支持范围：

| 维度 / 参数 | 范围 | 备注 |
|---|---|---|
| `Bkv`（输入 kv 的 batch size） | 4 ~ 32 | 输入 kv 的第 0 维 |
| `N`（输入 kv 的 head number） | 1 ~ 8 | 输入 kv 的第 1 维，建议 N=1 |
| `Skv`（输入 kv 的 sequence length） | 1 ~ 512 | 输入 kv 的第 2 维 |
| `D`（输入 kv 的 head dim） | 192 | 输入 kv 的第 3 维，D = Dk + Dv |
| `Dk`（RoPE 维度） | 64 | 从 D 中拆分出的 RoPE 计算维度，需为偶数 |
| `Dv`（RMSNorm 维度） | 128 | 从 D 中拆分出的 RMSNorm 计算维度，D - Dk |
| `BlockNum`（PA 系列 cache 块数） | 69 ~ 268 | PA 系列 cache 的第 0 维 |
| `BlockSize`（PA 系列 cache 块大小） | 8 ~ 32 | PA 系列 cache 的第 1 维，需满足对齐约束 |

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
from einops import rearrange

def kv_rms_norm_rope_cache_v2(
    kv: torch.Tensor,
    gamma: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    index: torch.Tensor,
    k_cache: torch.Tensor,
    ckv_cache: torch.Tensor,
    k_rope_scale: torch.Tensor = None,
    c_kv_scale: torch.Tensor = None,
    k_rope_offset: torch.Tensor = None,
    c_kv_offset: torch.Tensor = None,
    v: torch.Tensor = None,
    epsilon: float = 1e-5,
    cache_mode: str = "Norm",
    is_output_kv: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    v_cache = ckv_cache
    k_scale = k_rope_scale
    v_scale = c_kv_scale
    k_offset = k_rope_offset
    v_offset = c_kv_offset
    tensor_v = v
    eps = epsilon

    ori_dtype = kv.dtype
    ori_k_cache_dtype = k_cache.dtype
    ori_v_cache_dtype = v_cache.dtype
    kv_dtype = kv.dtype

    tensor_v = tensor_v.to(torch.float32)

    kv = kv.to(torch.float32)
    gamma = gamma.to(torch.float32)
    cos = cos.to(torch.float32)
    sin = sin.to(torch.float32)

    kv_shape = kv.shape
    Bkv = kv_shape[0]
    Nkv = kv_shape[1]
    Skv = kv_shape[2]
    Dkv = kv_shape[3]

    v_dim = tensor_v.shape[3]
    k_dim = Dkv

    # b n s d -> b s n d
    kv = rearrange(kv, 'b n s d -> b s n d')
    cos = rearrange(cos, 'b n s d -> b s n d')
    sin = rearrange(sin, 'b n s d -> b s n d')

    if cache_mode == "Norm" and k_scale is None and v_scale is None:
        is_output_kv = False

    rms_in = kv
    v_in = tensor_v
    v_in = rearrange(v_in, 'b n s d -> b s n d')

    # RMS Norm
    v = rms_in / torch.sqrt(torch.mean(rms_in ** 2, dim=-1, keepdim=True) + eps)
    v = v * gamma

    # RoPE
    rope_dim = cos.shape[-1]
    rope_in = v[..., :rope_dim]
    k = rope_in.view(Bkv, Skv, Nkv, rope_dim // 2, 2) \
                .transpose(-1, -2) \
                .reshape(Bkv, Skv, Nkv, rope_dim)
    k1 = k[..., : k.shape[-1] // 2]
    k2 = k[..., k.shape[-1] // 2 :]
    rotate_half_k = torch.cat((-k2, k1), dim=-1)
    k_embed = (k * cos) + (rotate_half_k * sin)

    kv_out = torch.cat([k_embed, v[..., rope_dim:]], dim=-1)
    k_embed_out = rearrange(kv_out, 'b s n d -> b n s d').to(kv_dtype)

    if k_scale is not None:
        kv_out = kv_out * k_scale
    if k_offset is not None:
        kv_out = kv_out + k_offset
    if k_scale is not None:
        kv_out = torch.round(kv_out).clamp(-128, 127)
    k_embed = kv_out

    # tensor v
    v_out = rearrange(v_in, 'b s n d -> b n s d').to(kv_dtype)
    if v_scale is not None:
        v_in = v_in * v_scale
    if v_offset is not None:
        v_in = v_in + v_offset
    if v_scale is not None:
        v_in = torch.round(v_in).clamp(-128, 127)
    v = v_in

    if cache_mode == "PA_BNSD" or cache_mode == "PA":
        k_cache_shape = k_cache.shape
        v_cache_shape = v_cache.shape
        # b n s d -> (b s) n d
        k_cache = rearrange(k_cache, 'b s n d -> (b s)  n d')
        v_cache = rearrange(v_cache, 'b s n d -> (b s)  n d')
        k_embed = rearrange(k_embed, 'b s n d -> (b s)  n d')
        v = rearrange(v, 'b s n d -> (b s) n d')
        for batch in range(len(index)):
            if index[batch] == -1:
                continue
            k_cache[index[batch], :, :] = k_embed[batch, :, :].to(k_cache.dtype)
            v_cache[index[batch], :, :] = v[batch, :, :].to(v_cache.dtype)
        k_cache = rearrange(k_cache, '(b s) n d -> b s n d', b=k_cache_shape[0])
        v_cache = rearrange(v_cache, '(b s) n d -> b s n d', b=v_cache_shape[0])

    elif cache_mode == "PA_NZ":
        k_cache_shape = k_cache.shape
        v_cache_shape = v_cache.shape
        bn = k_cache_shape[0]
        block_size = k_cache_shape[1]
        dk = k_cache_shape[-1]
        dv = v_cache_shape[-1]
        dk0 = 32 if k_cache.dtype == torch.int8 else 16
        dv0 = 32 if v_cache.dtype == torch.int8 else 16
        dk1 = dk // dk0
        dv1 = dv // dv0
        num_head = k_cache_shape[2]
        k_cache = k_cache.reshape(bn, num_head, dk1, block_size, dk0)
        v_cache = v_cache.reshape(bn, num_head, dv1, block_size, dv0)
        k_embed = rearrange(k_embed, 'b s n d -> (b s)  n d')
        v = rearrange(v, 'b s n d -> (b s) n d')
        for batch in range(len(index)):
            index_value = index[batch]
            if index_value < 0:
                continue
            bn_id = index_value // block_size
            block_offset = index_value % block_size
            for i in range(dk1):
                k_cache[bn_id, :, i, block_offset, :] = \
                    k_embed[batch, :, i * dk0:(i + 1) * dk0].to(k_cache.dtype)
            for i in range(dv1):
                v_cache[bn_id, :, i, block_offset, :] = \
                    v[batch, :, i * dv0:(i + 1) * dv0].to(v_cache.dtype)
        k_cache = k_cache.reshape(k_cache_shape)
        v_cache = v_cache.reshape(v_cache_shape)

    elif cache_mode == "PA_BLK_BNSD":
        k_cache_shape = k_cache.shape
        v_cache_shape = v_cache.shape
        block_size = k_cache_shape[1]
        block_num = k_cache_shape[0]
        ceil_div_s = (Skv + block_size - 1) // block_size
        for batch in range(Bkv):
            for seq_id in range(ceil_div_s):
                seq_start = seq_id * block_size
                seq_end = Skv if seq_id == (ceil_div_s - 1) else (seq_id + 1) * block_size
                copy_len = seq_end - seq_start
                index_value = index[batch * ceil_div_s + seq_id]
                cache_b = index_value // block_size
                if index_value == -1:
                    continue
                k_cache[cache_b, :copy_len, :, :] = \
                    k_embed[batch, seq_start:seq_end, :, :].to(k_cache.dtype)
                v_cache[cache_b, :copy_len, :, :] = \
                    v[batch, seq_start:seq_end, :, :].to(v_cache.dtype)

    elif cache_mode == "PA_BLK_NZ":
        k_cache_shape = k_cache.shape
        v_cache_shape = v_cache.shape
        bn = k_cache_shape[0]
        block_size = k_cache_shape[1]
        dk = k_cache_shape[-1]
        dv = v_cache_shape[-1]
        dk0 = 32 if k_cache.dtype == torch.int8 else 16
        dv0 = 32 if v_cache.dtype == torch.int8 else 16
        dk1 = dk // dk0
        dv1 = dv // dv0
        num_head = k_cache_shape[2]
        k_cache = k_cache.reshape(bn, num_head, dk1, block_size, dk0)
        v_cache = v_cache.reshape(bn, num_head, dv1, block_size, dv0)
        ceil_div_s = (Skv + block_size - 1) // block_size
        for batch in range(Bkv):
            for seq_id in range(ceil_div_s):
                seq_start = seq_id * block_size
                seq_end = Skv if seq_id == (ceil_div_s - 1) else (seq_id + 1) * block_size
                copy_len = seq_end - seq_start
                index_value = index[batch * ceil_div_s + seq_id]
                cache_b = index_value // block_size
                if index_value == -1:
                    continue
                for n_idx in range(num_head):
                    for i in range(dk1):
                        k_cache[cache_b, n_idx, i, :copy_len, :] = \
                            k_embed[batch, seq_start:seq_end, n_idx, i * dk0:(i + 1) * dk0].to(k_cache.dtype)
                    for i in range(dv1):
                        v_cache[cache_b, n_idx, i, :copy_len, :] = \
                            v[batch, seq_start:seq_end, n_idx, i * dv0:(i + 1) * dv0].to(v_cache.dtype)
        k_cache = k_cache.reshape(k_cache_shape)
        v_cache = v_cache.reshape(v_cache_shape)

    else:
        v_cache = rearrange(v_cache, 'b n s d -> b s n d')
        k_cache = rearrange(k_cache, 'b n s d -> b s n d')
        for batch in range(index.shape[0]):
            for sdx in range(index.shape[1]):
                if index[batch][sdx] == -1:
                    continue
                v_cache[batch, index[batch][sdx], :, :] = v[batch, sdx, :, :].to(v_cache.dtype)
                k_cache[batch, index[batch][sdx], :, :] = k_embed[batch, sdx, :, :].to(k_cache.dtype)
        v_cache = rearrange(v_cache, 'b s n d -> b n s d')
        k_cache = rearrange(k_cache, 'b s n d -> b n s d')

    if is_output_kv:
        output_data = (
            k_cache.to(ori_k_cache_dtype),
            v_cache.to(ori_v_cache_dtype),
            k_embed_out.to(ori_dtype),
            v_out.to(ori_dtype),
        )
    else:
        output_data = (
            k_cache.to(ori_k_cache_dtype),
            v_cache.to(ori_v_cache_dtype),
            torch.zeros_like(k_embed_out).to(ori_dtype),
            torch.zeros_like(v_out).to(ori_dtype),
        )
    return output_data
```

## 6. 额外信息

### 算子调用示例

以下示例展示了如何在 `cache_mode="Norm"` 与 `cache_mode="PA"` 两种典型场景下调用 `cann_bench.kv_rms_norm_rope_cache_v2`。

```python
import torch
import torch_npu
import cann_bench

device = "npu:0"

Bkv, Nkv, Skv, Dkv = 8, 1, 256, 192
Dk = 64
Dv = Dkv - Dk

kv = torch.rand(Bkv, Nkv, Skv, Dkv, dtype=torch.float16, device=device)
v = torch.rand(Bkv, Nkv, Skv, Dv, dtype=torch.float16, device=device)
gamma = torch.rand(Dkv, dtype=torch.float16, device=device) * 0.5 + 0.5
cos = torch.rand(Bkv, Nkv, Skv, Dk, dtype=torch.float16, device=device) * 2 - 1
sin = torch.rand(Bkv, Nkv, Skv, Dk, dtype=torch.float16, device=device) * 2 - 1

BlockNum, BlockSize = 69, 32
index = torch.randint(-1, BlockNum * BlockSize, (Bkv * Skv,), dtype=torch.int64, device=device)

k_cache = torch.zeros(BlockNum, BlockSize, Nkv, Dkv, dtype=torch.float16, device=device)
ckv_cache = torch.zeros(BlockNum, BlockSize, Nkv, Dv, dtype=torch.float16, device=device)

k_cache, ckv_cache, k_rope, c_kv = cann_bench.kv_rms_norm_rope_cache_v2(
    kv, gamma, cos, sin, index, k_cache, ckv_cache,
    k_rope_scale=None, c_kv_scale=None,
    k_rope_offset=None, c_kv_offset=None,
    v=v,
    epsilon=1e-5,
    cache_mode="PA",
    is_output_kv=False,
)
```
