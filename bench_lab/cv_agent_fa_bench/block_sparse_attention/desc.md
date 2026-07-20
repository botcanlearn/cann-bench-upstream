# BlockSparseAttention 算子 API 描述

## 1. 算子简介

块内 attention，将序列按 block_size 切分，只在每个块内部计算 scaled dot-product attention。

**主要应用场景**：

- 局部块稀疏 attention
- 长序列分块计算
- block-level attention kernel 验证

**算子特征**：

- seq_len 会按 block_size 对齐
- 每个 block 独立 softmax
- 输出再 reshape 回原序列布局
- 难度等级：L4（FusedComposite）

## 2. 算子定义

### 数学公式

```text
Y blocks are computed as softmax(Q_block @ K_block^T / sqrt(D)) @ V_block for each block.
```

### Golden 语义来源

- 参考实现：`/root/asc/cv_agent_all/cv_agent/tile2asc/block_sparse_attention/model.py`
- staging golden 只保留 schema 同名函数以及 `Model.forward` 所需 helper，不保留 tile2asc 的 case 构造接口。


## 3. 接口规范

### 算子原型

```python
block_sparse_attention(Tensor q, Tensor k, Tensor v, int block_size=32) -> Tensor output
```

### 输入参数说明

| 参数 | 类型 | 支持 dtype | Shape | 描述 |
|---|---|---|---|---|
| `q` | Tensor | `float16` | q/k/v [B, H, S, D], S must be divisible by block_size. | q input tensor |
| `k` | Tensor | `float16` | q/k/v [B, H, S, D], S must be divisible by block_size. | k input tensor |
| `v` | Tensor | `float16` | q/k/v [B, H, S, D], S must be divisible by block_size. | v input tensor |

### 属性

| 参数 | 类型 | 默认值 | 描述 |
|---|---|---|---|
| `block_size` | int | `32` | block_size operator attribute |

### 输出

| 参数 | Shape | dtype | 描述 |
|---|---|---|---|
| `output` | 由参考 forward 和输入 shape 决定 | 跟随参考实现输出 | golden 输出张量 |

### 规则与约束

- q/k/v shape 必须一致，均为 [B, H, S, D]。
- S 必须能被 block_size 整除；forward 会按 block_size reshape 为 block attention。
- 每个 block 内独立执行 scaled dot-product attention，block 之间不互相可见。

### 支持范围

| 维度 / 参数 | staging cases 覆盖 |
|---|---|
| `batch_size` | 1 ~ 16 |
| `block_size` | 32 ~ 128 |
| `d_k` | 64 ~ 512 |
| `dtype` | {float16} |
| `n_heads` | 4 ~ 16 |
| `seq_len` | 64 ~ 1024 |
| `input dtype tuple` | float16/float16/float16 |

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
def block_sparse_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, block_size: int = 32) -> torch.Tensor:
```

该函数内部仅按输入 tensor shape 和正式 attrs 初始化参考 `Model`，然后调用 upstream `Model.forward`。

## 6. 额外信息

- `cases.yaml` / `cases.csv`：20 条 staging case，attrs 使用 YAML 原生类型。
- `baseline_perf_us` 和 `t_hw_us` 当前保持 `null`，需后续硬件补测。
- 本任务仍处于 `bench_lab/cv_agent_bench` staging 目录，正式并入 `tasks/level4` 前应再次跑性能与 parser 兼容性检查。
