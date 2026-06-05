# FlashAttentionMask 算子 API 描述

## 1. 算子简介

显式 mask 版本的 attention，输入 mask 控制每个 query 位置可见的 key 位置。

**主要应用场景**：

- 用户自定义 attention mask
- padding/causal/block mask 验证
- 需要外部传入可见性矩阵的 attention

**算子特征**：

- mask shape 与 scores 广播或一致
- mask 为 false 的位置填充 -inf
- 内部以 float32 累积后转回 float16
- 难度等级：L4（FusedComposite）

## 2. 算子定义

### 数学公式

```text
Y = softmax((Q @ K^T / sqrt(D)) masked by the supplied boolean mask) @ V
```

### Golden 语义来源

- 参考实现：`/root/asc/cv_agent_all/cv_agent/tile2asc/flash_attention_mask/model.py`
- staging golden 只保留 schema 同名函数以及 `Model.forward` 所需 helper，不保留 tile2asc 的 case 构造接口。


## 3. 接口规范

### 算子原型

```python
flash_attention_mask(Tensor q, Tensor k, Tensor v, Tensor mask) -> Tensor output
```

### 输入参数说明

| 参数 | 类型 | 支持 dtype | Shape | 描述 |
|---|---|---|---|---|
| `q` | Tensor | `float16` | q [B, H, Sq, D], k/v [B, H, Skv, D], mask [B, H, Sq, Skv]. | q input tensor |
| `k` | Tensor | `float16` | q [B, H, Sq, D], k/v [B, H, Skv, D], mask [B, H, Sq, Skv]. | k input tensor |
| `v` | Tensor | `float16` | q [B, H, Sq, D], k/v [B, H, Skv, D], mask [B, H, Sq, Skv]. | v input tensor |
| `mask` | Tensor | `uint8` | q [B, H, Sq, D], k/v [B, H, Skv, D], mask [B, H, Sq, Skv]. | mask input tensor |

### 属性

| 参数 | 类型 | 默认值 | 描述 |
|---|---|---|---|
| - | - | - | 无正式算子属性；shape/dtype 仅由 case 输入张量决定 |

### 输出

| 参数 | Shape | dtype | 描述 |
|---|---|---|---|
| `output` | 由参考 forward 和输入 shape 决定 | 跟随参考实现输出 | golden 输出张量 |

### 规则与约束

- q shape 为 [B, H, Sq, D]，k/v shape 为 [B, H, Skv, D]。
- mask shape 为 [B, H, Sq, Skv]，dtype 为 uint8/bool，非零表示可见。
- mask_type 仅用于 staging case 构造，不是正式 schema attr；forward 只消费 mask tensor。

### 支持范围

| 维度 / 参数 | staging cases 覆盖 |
|---|---|
| `batch_size` | 1 ~ 4 |
| `d_k` | 32 ~ 128 |
| `dtype` | {float16} |
| `mask_type` | {band, block, causal, full} |
| `n_heads` | 4 ~ 32 |
| `seq_len_kv` | 32 ~ 1024 |
| `seq_len_q` | 1 ~ 1024 |
| `input dtype tuple` | float16/float16/float16/uint8 |

## 4. 精度要求

采用生态算子精度标准进行验证，主要使用平均相对误差（MERE）和最大相对误差（MARE）。

```text
MERE = avg(abs(actual - golden) / (abs(golden) + 1e-7))
MARE = max(abs(actual - golden) / (abs(golden) + 1e-7))
```

当 MERE < dtype 对应 Threshold 且 MARE < 10 * Threshold 时判定为通过。对 sparse/page/mask/TND 等结构化输入，比较前必须保证索引、page table、mask 或 prefix length 语义合法。

## 5. 标准 Golden 代码

完整 PyTorch 参考实现位于同目录 `golden.py`。正式入口为 schema 同名函数：

```python
def flash_attention_mask(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
```

该函数内部仅按输入 tensor shape 和正式 attrs 初始化参考 `Model`，然后调用 upstream `Model.forward`。

## 6. 额外信息

- `cases.yaml` / `cases.csv`：20 条 staging case，attrs 使用 YAML 原生类型。
- `baseline_perf_us` 和 `t_hw_us` 当前保持 `null`，需后续硬件补测。
- 本任务仍处于 `bench_lab/cv_agent_bench` staging 目录，正式并入 `tasks/level4` 前应再次跑性能与 parser 兼容性检查。
