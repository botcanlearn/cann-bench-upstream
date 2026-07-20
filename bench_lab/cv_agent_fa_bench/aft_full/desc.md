# AftFull 算子 API 描述

## 1. 算子简介

Attention Free Transformer 的 full 变体，使用 position bias、key 指数权重和 query sigmoid 门控完成 token mixing。

**主要应用场景**：

- AFT 结构验证
- 不显式构造 QK attention matrix 的 token mixing
- 位置偏置参与的序列聚合

**算子特征**：

- 输入 q/k/v shape 为 [B,N,D]
- position_bias shape 为 [N,N]
- 输出 shape 为 [B,N,D]
- 难度等级：L4（FusedComposite）

## 2. 算子定义

### 数学公式

```text
Y = sigmoid(Q) * sum(exp(K + position_bias) * V) / sum(exp(K + position_bias))
```

### Golden 语义来源

- 参考实现：`/root/asc/cv_agent_all/cv_agent/tile2asc/aft_full/model.py`
- staging golden 只保留 schema 同名函数以及 `Model.forward` 所需 helper，不保留 tile2asc 的 case 构造接口。


## 3. 接口规范

### 算子原型

```python
aft_full(Tensor q, Tensor k, Tensor v) -> Tensor output
```

### 输入参数说明

| 参数 | 类型 | 支持 dtype | Shape | 描述 |
|---|---|---|---|---|
| `q` | Tensor | `float32` | q/k/v [B, S, D]. | q input tensor |
| `k` | Tensor | `float32` | q/k/v [B, S, D]. | k input tensor |
| `v` | Tensor | `float32` | q/k/v [B, S, D]. | v input tensor |

### 属性

| 参数 | 类型 | 默认值 | 描述 |
|---|---|---|---|
| - | - | - | 无正式算子属性；shape/dtype 仅由 case 输入张量决定 |

### 输出

| 参数 | Shape | dtype | 描述 |
|---|---|---|---|
| `output` | 由参考 forward 和输入 shape 决定 | 跟随参考实现输出 | golden 输出张量 |

### 规则与约束

- q/k/v shape 必须一致，均为 [B, S, D]。
- position_bias 大小由输入序列长度 S 初始化；staging golden 保持 upstream nn.Parameter(torch.ones) 语义。
- 输出 dtype 跟随 PyTorch 参考实现，staging cases 使用 float32。

### 支持范围

| 维度 / 参数 | staging cases 覆盖 |
|---|---|
| `batch_size` | 1 ~ 32 |
| `d_model` | 128 ~ 512 |
| `seq_len` | 16 ~ 256 |
| `input dtype tuple` | float32/float32/float32 |

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
def aft_full(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
```

该函数内部仅按输入 tensor shape 和正式 attrs 初始化参考 `Model`，然后调用 upstream `Model.forward`。

## 6. 额外信息

- `cases.yaml` / `cases.csv`：20 条 staging case，attrs 使用 YAML 原生类型。
- `baseline_perf_us` 和 `t_hw_us` 当前保持 `null`，需后续硬件补测。
- 本任务仍处于 `bench_lab/cv_agent_bench` staging 目录，正式并入 `tasks/level4` 前应再次跑性能与 parser 兼容性检查。
