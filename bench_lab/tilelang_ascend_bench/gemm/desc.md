# Gemm (基础版) TileLang 评测

## 1. 任务范围

本任务是 TileLang Gemm 基础版算子的性能评测 benchmark。

覆盖范围：

- `case_id`: 1-20
- 输入 dtype: float16, bfloat16
- 输入 rank: 2D
- 输出 shape: [M, N]

## 2. 算子定义

接口：

```python
cann_bench.gemm(Tensor A, Tensor B) -> Tensor C
```

数学语义：

```text
C[i, j] = Σ_k A[i, k] * B[k, j]
```

参数说明：

| 参数 | 类型 | 描述 |
|------|------|------|
| A | Tensor | 左矩阵 [M, K]，float16 或 bfloat16 |
| B | Tensor | 右矩阵 [K, N]，float16 或 bfloat16 |

## 3. 实现说明

基于 TileLang Expert&Developr 混合模式：
- 分块参数：block_M=128, block_N=256, K_L1=64
- 计算指令：`T.gemm_v0`
- 同步方式：`pass_configs` 启用 auto_sync

## 4. 精度要求

与 `torch.matmul` 的参考实现对比，rtol=1e-2, atol=1e-2。

## 5. Golden 代码

```python
import torch

def gemm(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return torch.matmul(A, B)
```
