# BigbirdAttention 算子 API 描述

## 1. 算子简介

BigBird 风格稀疏 attention，组合局部窗口、首尾全局 token 和随机可见 token。

**主要应用场景**：

- 长序列 BigBird attention
- local + global + random 稀疏模式
- 稀疏 mask kernel 验证

**算子特征**：

- 显式输入 mask
- window_size 控制局部范围
- num_random_blocks 控制随机可见 token 数
- 难度等级：L4（FusedComposite）

## 2. 算子定义

### 数学公式

```text
Y = softmax((Q @ K^T / sqrt(D)) masked by local + global + random BigBird mask) @ V
```

### Golden 语义来源

- 参考实现：`/root/asc/cv_agent_all/cv_agent/tile2asc/bigbird_attention/model.py`
- staging golden 只保留 schema 同名函数以及 `Model.forward` 所需 helper，不保留 tile2asc 的 case 构造接口。


## 3. 接口规范

### 算子原型

```python
bigbird_attention(Tensor q, Tensor k, Tensor v, Tensor mask, int window_size=32, int num_random_blocks=3) -> Tensor output
```

### 输入参数说明

| 参数 | 类型 | 支持 dtype | Shape | 描述 |
|---|---|---|---|---|
| `q` | Tensor | `float32` | q/k/v [B, H, S, D], mask [S, S] broadcast over batch and heads. | q input tensor |
| `k` | Tensor | `float32` | q/k/v [B, H, S, D], mask [S, S] broadcast over batch and heads. | k input tensor |
| `v` | Tensor | `float32` | q/k/v [B, H, S, D], mask [S, S] broadcast over batch and heads. | v input tensor |
| `mask` | Tensor | `uint8` | q/k/v [B, H, S, D], mask [S, S] broadcast over batch and heads. | mask input tensor |

### 属性

| 参数 | 类型 | 默认值 | 描述 |
|---|---|---|---|
| `window_size` | int | `32` | window_size operator attribute |
| `num_random_blocks` | int | `3` | num_random_blocks operator attribute |

### 输出

| 参数 | Shape | dtype | 描述 |
|---|---|---|---|
| `output` | 由参考 forward 和输入 shape 决定 | 跟随参考实现输出 | golden 输出张量 |

### 规则与约束

- q/k/v shape 为 [B, H, S, D]，mask shape 为 [S, S]，在 scores [B, H, S, S] 上广播。
- d_model = H * D，且必须能被 H 整除。
- mask 语义包含 local window、首尾 global token 和 torch.randperm 生成的 random token。
- staging cases 使用 float32 输入以避免 upstream CPU float16 masked_fill(-1e9) 溢出；forward 语义未改。

### 支持范围

| 维度 / 参数 | staging cases 覆盖 |
|---|---|
| `batch_size` | 1 ~ 32 |
| `d_model` | 512 ~ 4096 |
| `dtype` | {float32} |
| `n_heads` | 4 ~ 16 |
| `num_random_blocks` | 2 ~ 4 |
| `seq_len` | 128 ~ 1024 |
| `window_size` | 16 ~ 64 |
| `input dtype tuple` | float32/float32/float32/uint8 |

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
def bigbird_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor, window_size: int = 32, num_random_blocks: int = 3) -> torch.Tensor:
```

该函数内部仅按输入 tensor shape 和正式 attrs 初始化参考 `Model`，然后调用 upstream `Model.forward`。

## 6. 额外信息

- `cases.yaml` / `cases.csv`：20 条 staging case，attrs 使用 YAML 原生类型。
- `baseline_perf_us` 和 `t_hw_us` 当前保持 `null`，需后续硬件补测。
- 本任务仍处于 `bench_lab/cv_agent_bench` staging 目录，正式并入 `tasks/level4` 前应再次跑性能与 parser 兼容性检查。
