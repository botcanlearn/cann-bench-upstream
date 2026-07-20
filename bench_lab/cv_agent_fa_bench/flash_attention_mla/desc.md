# FlashAttentionMla 算子 API 描述

## 1. 算子简介

MLA 风格 attention，将 nope 与 rope 两部分 Q/K 拼接后计算注意力，V 可以使用独立输出维度。

**主要应用场景**：

- DeepSeek MLA 类结构验证
- RoPE 与非 RoPE head_dim 拆分场景
- KV head 少于 query head 的 latent attention

**算子特征**：

- q_nope/q_rope 与 k_nope/k_rope 分开输入
- K/V 按 kv_heads repeat 到 n_heads
- 输出最后一维为 d_v
- 难度等级：L4（FusedComposite）

## 2. 算子定义

### 数学公式

```text
Concatenate nope/rope score components, repeat KV heads as needed, then compute scaled dot-product attention.
```

### Golden 语义来源

- 参考实现：`/root/asc/cv_agent_all/cv_agent/tile2asc/flash_attention_mla/model.py`
- staging golden 只保留 schema 同名函数以及 `Model.forward` 所需 helper，不保留 tile2asc 的 case 构造接口。


## 3. 接口规范

### 算子原型

```python
flash_attention_mla(Tensor q_nope, Tensor q_rope, Tensor k_nope, Tensor k_rope, Tensor v) -> Tensor output
```

### 输入参数说明

| 参数 | 类型 | 支持 dtype | Shape | 描述 |
|---|---|---|---|---|
| `q_nope` | Tensor | `bfloat16, float16` | q_nope/q_rope [B, Hq, Sq, D*], k_nope/k_rope/v [B, Hkv, Skv, D*], Hq % Hkv == 0. | q_nope input tensor |
| `q_rope` | Tensor | `bfloat16, float16` | q_nope/q_rope [B, Hq, Sq, D*], k_nope/k_rope/v [B, Hkv, Skv, D*], Hq % Hkv == 0. | q_rope input tensor |
| `k_nope` | Tensor | `bfloat16, float16` | q_nope/q_rope [B, Hq, Sq, D*], k_nope/k_rope/v [B, Hkv, Skv, D*], Hq % Hkv == 0. | k_nope input tensor |
| `k_rope` | Tensor | `bfloat16, float16` | q_nope/q_rope [B, Hq, Sq, D*], k_nope/k_rope/v [B, Hkv, Skv, D*], Hq % Hkv == 0. | k_rope input tensor |
| `v` | Tensor | `bfloat16, float16` | q_nope/q_rope [B, Hq, Sq, D*], k_nope/k_rope/v [B, Hkv, Skv, D*], Hq % Hkv == 0. | v input tensor |

### 属性

| 参数 | 类型 | 默认值 | 描述 |
|---|---|---|---|
| - | - | - | 无正式算子属性；shape/dtype 仅由 case 输入张量决定 |

### 输出

| 参数 | Shape | dtype | 描述 |
|---|---|---|---|
| `output` | 由参考 forward 和输入 shape 决定 | 跟随参考实现输出 | golden 输出张量 |

### 规则与约束

- q_nope/q_rope shape 为 [B, Hq, Sq, D*]，k_nope/k_rope/v shape 为 [B, Hkv, Skv, D*]。
- Hq 必须能被 Hkv 整除；KV heads 会 repeat 到 query head 数。
- q_nope/k_nope 的 d_nope 一致，q_rope/k_rope 的 d_rope 一致。

### 支持范围

| 维度 / 参数 | staging cases 覆盖 |
|---|---|
| `batch_size` | 1 ~ 8 |
| `d_nope` | 128 ~ 256 |
| `d_rope` | 64 ~ 128 |
| `d_v` | 128 ~ 256 |
| `dtype` | {bfloat16, float16} |
| `kv_heads` | 1 ~ 8 |
| `n_heads` | 8 ~ 32 |
| `seq_len_kv` | 2 ~ 4096 |
| `seq_len_q` | 1 ~ 4096 |
| `input dtype tuple` | bfloat16/bfloat16/bfloat16/bfloat16/bfloat16 ; float16/float16/float16/float16/float16 |

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
def flash_attention_mla(q_nope: torch.Tensor, q_rope: torch.Tensor, k_nope: torch.Tensor, k_rope: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
```

该函数内部仅按输入 tensor shape 和正式 attrs 初始化参考 `Model`，然后调用 upstream `Model.forward`。

## 6. 额外信息

- `cases.yaml` / `cases.csv`：20 条 staging case，attrs 使用 YAML 原生类型。
- `baseline_perf_us` 和 `t_hw_us` 当前保持 `null`，需后续硬件补测。
- 本任务仍处于 `bench_lab/cv_agent_bench` staging 目录，正式并入 `tasks/level4` 前应再次跑性能与 parser 兼容性检查。
