# TopKTopPSampleV2 算子 API 描述

## 1. 算子简介

根据输入词频 logits、topK/topP/minP 采样参数、随机采样权重分布 q，进行 topK-topP-minP 采样计算。当输入 `is_need_sample_result` 为 false 时，输出每个 batch 的最大词频索引 `logits_select_idx`，以及 topK-topP-minP 采样后的词频分布 `logits_top_kp_select`；当输入 `is_need_sample_result` 为 true 时，输出 topK-topP-minP 采样后的中间计算结果 `logits_idx` 和 `logits_sort_masked`，其中 `logits_sort_masked` 为词频 logits 经过 topK-topP-minP 采样计算后的中间结果，`logits_idx` 为 `logits_sort_masked` 在 logits 中对应的索引。

采样模式由 `q` 是否存在决定：
- `q` 存在时走 `qSample`：按 `selected_probs / (|q| + eps)` 取 argmax
- `q` 不存在时走 `argmax`：直接对筛选后的概率取 argmax

**主要应用场景**：
- 推荐系统候选排序
- 分类结果 Top-K 筛选
- 搜索排序加速

**算子特征**：
- 难度等级：L3（SortSelect）
- 5 输入，4 输出，6 个属性参数
- 支持 ND 格式输入
- 可选属性：eps, is_need_logits, top_k_guess, ks_max, input_is_logits, is_need_sample_result

## 2. 算子定义

### 数学公式

$$
topKValue[b] = {Max(topK[b])}_{s=1}^{\left \lceil \frac{S}{v} \right \rceil }\left \{ topKValue[b]\left \{s-1 \right \}  \cup \left \{ logits[b][v] \ge topKMin[b][s-1] \right \} \right \}\\
  Card(topKValue[b])=topK[b]
$$

$$
topKMin[b][s] = Min(topKValue[b]\left \{  s \right \})
$$

$$
v = 8 * \text{ks\_max}
$$

## 3. 接口规范

### 算子原型

```python
cann_bench.top_k_top_p_sample_v2(
    Tensor logits,
    Tensor top_k,
    Tensor top_p,
    Tensor q,
    Tensor min_ps,
    float eps=1e-8,
    bool is_need_logits=False,
    int64 top_k_guess=32,
    int64 ks_max=1024,
    bool input_is_logits=True,
    bool is_need_sample_result=False
) -> (Tensor logits_select_idx, Tensor logits_top_kp_select, Tensor logits_idx, Tensor logits_sort_masked)
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| logits | Tensor | 必选 | 输入张量 `logits`，shape `[B, V]` |
| top_k | Tensor | 必选 | 每个 batch 采样的 k 值，INT32 |
| top_p | Tensor | 必选 | 每个 batch 采样的 p 值 |
| q | Tensor | 必选 | 随机采样权重分布，shape 与 `logits` 一致；可为空 Tensor 表示不使用 qSample |
| min_ps | Tensor | 必选 | 每个 batch 采样的 minP 值 |
| eps | float | 1e-8 | 在权重采样中防止除零 |
| is_need_logits | bool | False | 控制 `logits_top_kp_select` 的输出条件 |
| top_k_guess | int64 | 32 | 每个 batch 在尝试 topP 部分遍历采样 logits 时的候选 logits 大小 |
| ks_max | int64 | 1024 | 每个 batch 在 topK 采样时最大 topK 值，必须为正整数 |
| input_is_logits | bool | True | 表示输入的 logits 是否未进行归一化 |
| is_need_sample_result | bool | False | 表示是否输出中间计算结果 `logits_idx` 和 `logits_sort_masked` |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| logits_select_idx | `[B]` | int64 | 经过 topK-topP-sample 计算后，每个 batch 中选中词频在输入 logits 中的位置索引 |
| logits_top_kp_select | `[B, V]` | float32 | 经过 topK-topP 计算后，输入 logits 中剩余未被过滤的 logits |
| logits_idx | `[B, V]` | int64 | 经过 topK-topP-minP 计算后，每个 batch 的中间采样结果在输入 logits 中的位置索引 |
| logits_sort_masked | `[B, V]` | float32 | 经过 topK-topP-minP 计算后，每个 batch 的中间采样结果 |

### 数据类型

| 输入 dtype | 输出 dtype |
|-----------|-----------|
| float16/bfloat16/float32 | int64 |
| float16/bfloat16/float32 | float32 |

### 规则与约束

- **logits**（输入，FLOAT16、BFLOAT16、FLOAT32，ND）：待采样的输入词频，shape `[B, V]`。
- **top_k**（输入，INT32）：表示每个 batch 采样的 k 值。
- **top_p**（输入，FLOAT16、BFLOAT16、FLOAT32）：表示每个 batch 采样的 p 值。
- **q**（输入，FLOAT32，ND）：表示 topK-topP 采样输出的指数采样矩阵，维度与尺寸需要与 `logits` 保持一致；传入空 Tensor 时跳过 qSample。
- **min_ps**（输入，FLOAT16、BFLOAT16、FLOAT32）：表示每个 batch 采样的 minP 值。
- **eps**（输入，FLOAT32）：在权重采样中防止除零，建议设置为 1e-8。
- **is_need_logits**（输入，BOOL）：控制 `logits_top_kp_select` 的输出条件，建议设置为 False。
- **top_k_guess**（输入，INT64）：表示每个 batch 在尝试 topP 部分遍历采样 logits 时的候选 logits 大小，对应公式中的 `GuessK`。
- **ks_max**（输入，INT64）：表示每个 batch 在 topK 采样时最大 topK 值，必须为正整数。
- **input_is_logits**（输入，BOOL）：表示输入的 logits 是否未进行归一化，默认为 True。
- **is_need_sample_result**（输入，BOOL）：表示是否输出中间计算结果，默认为 False。
- **logits_select_idx**（输出，INT64，ND）：表示经过 topK-topP-sample 计算流程后，每个 batch 中词频最大元素在输入 logits 中的位置索引。
- **logits_top_kp_select**（输出，FLOAT32，ND）：表示经过 topK-topP 计算流程后，输入 logits 中剩余未被过滤的 logits。
- **logits_idx**（输出，INT64，ND）：表示经过 topK-topP-minP 计算流程后，每个 batch 的中间采样结果在输入 logits 中的位置索引。
- **logits_sort_masked**（输出，FLOAT32，ND）：表示经过 topK-topP-minP 计算流程后，每个 batch 的中间采样结果。

### 采样/路由确定性约束

为保证 golden 与 NPU 输出在相等 key 场景下可逐元素比对一致，本算子对排序与采样行为有如下确定性要求：

1. **排序稳定性**：所有 `topK` 与 `topP` 筛选步骤必须使用**稳定排序**（stable sort）。对于值相等的元素，必须保持其在原始 `logits` 中的相对顺序。
2. **取首个最大索引**：在 `argmax` 或 `qSample` 取最值时，若多个位置得分相同，必须返回**第一个（索引最小）**最大位置。
3. **采样模式**：
   - `q` 存在时走 `qSample`：`selected_probs / (|q| + eps)` 后取 argmax；
   - `q` 不存在时走 `argmax`：直接对筛选后的概率取 argmax。
   - 本算子不通过 `torch.multinomial` 进行随机采样，所有随机性由外部输入 `q` 张量承载。
4. **NPU kernel 对齐要求**：NPU 侧实现必须遵循上述稳定排序 + 取首个最大索引的约定，否则在存在相等 key 的用例中会出现 golden 与 NPU 结果发散、逐元素比对失败的问题。

### 支持范围

输入 tensor 各维度与参数的支持范围：

| 维度 / 参数 | 范围 | 备注 |
|---|---|---|
| `ndim`（输入维度数） | 1 ~ 2 | cases 实测范围 |
| `dim_0`（第0维大小） | 1 ~ 140 | cases 实测范围 |
| `dim_1`（第1维大小） | 32 ~ 153600 | cases 实测范围 |
| `dtype` | bfloat16, float16, float32 | cases 实测覆盖 |
| `eps` | 1e-08 | cases 实测值 |
| `is_need_logits` | False ~ True | cases 实测范围 |
| `top_k_guess` | 12 ~ 32 | cases 实测范围 |
| `ks_max` | 1024 | cases 实测值 |
| `input_is_logits` | False ~ True | cases 实测范围 |
| `is_need_sample_result` | False ~ False | cases 实测范围 |

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
import numpy as np


FLT_NEG_INF = float('-inf')
USE_FAST_PROBS = True
ALL_P_MAX = 1.0


NP_TO_TORCH_DTYPE = {
    np.float32: torch.float32,
    np.float64: torch.float64,
    np.int32: torch.int32,
    np.int64: torch.int64,
    np.uint8: torch.uint8,
    np.bool_: torch.bool,
}


def onlySoftmax(x, dim=-1):
    if dim < 0:
        dim = x.dim() + dim

    max_vals = torch.max(x, dim=dim, keepdim=True)[0]
    shifted = x - max_vals
    exp_vals = torch.exp(shifted)
    softmax_output = exp_vals / torch.sum(exp_vals, dim=dim, keepdim=True)
    return softmax_output


def _to_device_tensor(t, device):
    """将 numpy/torch 输入转换到目标设备。"""
    if t is None:
        return None
    if isinstance(t, torch.Tensor):
        return t.to(device)
    # numpy 或标量
    return torch.as_tensor(t).to(device)


def top_k_top_p_sample_v2(
    logits: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    q: torch.Tensor,
    min_ps: torch.Tensor,
    eps: float = 1e-8,
    is_need_logits: bool = False,
    top_k_guess: int = 32,
    ks_max: int = 1024,
    input_is_logits: bool = True,
    is_need_sample_result: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    TopKTopPSampleV2 Golden 参考实现。

    对齐真实算子接口（无 post_sample 属性）：
    - q 存在时走 qSample
    - q 不存在时走 argmax
    返回 (logits_select_idx, logits_top_kp_select, logits_idx, logits_sort_masked)。
    """
    device = logits.device
    logits = _to_device_tensor(logits, device)
    topK = _to_device_tensor(top_k, device)
    topP = _to_device_tensor(top_p, device)
    q = _to_device_tensor(q, device) if q is not None else None
    min_ps = _to_device_tensor(min_ps, device) if min_ps is not None else None

    batch_size, vocab_size = logits.shape

    # 计算实际 k_max（与 kernel 对齐）
    k_max_aligned = (ks_max * 4 + 32 - 1) // 32 * 32 // 4
    k_max = min(k_max_aligned, 1024)

    # 初始化结果张量
    rs_index = torch.zeros(batch_size, dtype=torch.long, device=device)
    logits_idx = torch.zeros((batch_size, vocab_size), dtype=torch.long, device=device)
    logits_sort_masked = torch.zeros((batch_size, vocab_size), dtype=torch.float32, device=device)

    # 根据是否需要 logits 初始化 rs_value
    if is_need_logits:
        if input_is_logits:
            rs_value = torch.ones((batch_size, vocab_size), dtype=torch.float32, device=device) * FLT_NEG_INF
        else:
            rs_value = torch.zeros((batch_size, vocab_size), dtype=torch.float32, device=device)
    else:
        rs_value = torch.empty(0, dtype=torch.float32, device=device)

    # compute golden
    for i in range(batch_size):
        original_logits = logits[i].float()

        k_val = topK[i].item()
        top_ks_max = min(k_max, vocab_size)
        use_top_k = (1 <= k_val <= top_ks_max)

        p = topP[i].item()
        use_top_p = p < ALL_P_MAX

        # 降序排序
        topk_logits, topk_indices = torch.sort(original_logits, dim=-1, descending=True, stable=True)

        # topK
        if use_top_k:
            k_val = min(k_val, vocab_size)
            topk_logits = topk_logits[:k_val]
            topk_indices = topk_indices[:k_val]

        # 归一化
        if input_is_logits:
            topk_probs = onlySoftmax(topk_logits, dim=-1)
        else:
            topk_probs = topk_logits

        # topP
        if use_top_p:
            sorted_probs, sorted_probs_indices = torch.sort(topk_probs, dim=-1, descending=True, stable=True)
            if p > 0:
                probs_sum = sorted_probs.cumsum(dim=-1)
                top_p_mask = (probs_sum - sorted_probs) >= p
            else:
                top_p_mask = torch.ones(sorted_probs.numel(), dtype=torch.bool)
                top_p_mask[0] = False

            top_p_sel = ~top_p_mask
            selected_probs_indices = sorted_probs_indices[top_p_sel]

            if USE_FAST_PROBS:
                selected_indices = topk_indices[selected_probs_indices]
                selected_logits = sorted_probs[top_p_sel]
            else:
                selected_indices = topk_indices[selected_probs_indices]
                selected_logits = topk_logits[selected_probs_indices]

            false_count = (top_p_sel > 0).sum().item()
        else:
            selected_indices = topk_indices
            selected_logits = topk_probs
            false_count = topk_probs.numel()
            top_p_sel = torch.ones(false_count, dtype=torch.bool)

        if p <= 0 and input_is_logits:
            selected_logits[0] = 1

        # minP
        if min_ps is not None:
            min_p = min_ps[i].item()
        else:
            min_p = -1

        if not use_top_k and not use_top_p and min_p < 1:
            selected_indices = torch.arange(len(original_logits), device=device)
            if input_is_logits:
                selected_logits = onlySoftmax(original_logits, dim=-1)
            else:
                selected_logits = original_logits

        if min_p <= 0:
            min_p_sel = torch.ones(false_count, dtype=torch.bool)
        elif min_p < 1:
            min_p_thd = torch.max(selected_logits) * min_p
            sel_prob_mask = selected_logits >= min_p_thd
            min_p_sel = sel_prob_mask
        else:
            min_p_sel = torch.zeros(false_count, dtype=torch.bool)
            min_p_sel[0] = True

        selected_logits = selected_logits[min_p_sel]
        selected_indices = selected_indices[min_p_sel]
        false_count = selected_logits.numel()

        if USE_FAST_PROBS:
            selected_probs = selected_logits
        else:
            if input_is_logits:
                selected_probs = onlySoftmax(selected_logits, dim=-1)
            else:
                selected_probs = selected_logits

        # 采样逻辑（真实算子语义）：q 存在用 qSample，否则 argmax
        if q is not None:
            q_i = q[i, :false_count]
            q_sample = selected_probs / (q_i.abs() + eps)
            probs_index = q_sample.argmax(dim=0).view(-1)
        else:
            probs_index = selected_probs.argmax(dim=0).view(-1)

        golden_index = selected_indices[probs_index].squeeze(0)
        rs_index[i] = golden_index

        if is_need_logits:
            rs_value[i, selected_indices] = original_logits[selected_indices]

    # 与自定义 kernel 的输出形状对齐：
    if not is_need_logits:
        rs_value = torch.zeros((batch_size, vocab_size), dtype=torch.float32, device=device)

    if not is_need_sample_result:
        logits_idx = torch.zeros((batch_size, vocab_size), dtype=torch.int64, device=device)
        logits_sort_masked = torch.zeros((batch_size, vocab_size), dtype=torch.float32, device=device)

    return rs_index, rs_value, logits_idx, logits_sort_masked

```

## 6. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

logits = torch.randn(1, 32000, dtype=torch.float16, device="npu")
top_k = torch.randint(1, 50, (1,), dtype=torch.int32, device="npu")
top_p = torch.tensor([0.9], dtype=torch.float16, device="npu")
q = torch.randn(1, 32000, dtype=torch.float32, device="npu")
min_ps = torch.tensor([0.1], dtype=torch.float16, device="npu")
logits_select_idx, logits_top_kp_select, logits_idx, logits_sort_masked = cann_bench.top_k_top_p_sample_v2(
    logits, top_k, top_p, q, min_ps,
    eps=1e-8, is_need_logits=True, top_k_guess=32, ks_max=1024,
    input_is_logits=True, is_need_sample_result=False
)
```
