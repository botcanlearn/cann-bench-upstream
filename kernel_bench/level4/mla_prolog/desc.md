# MlaProlog 算子 API 描述

## 1. 算子简介

Multi-Head Latent Attention 前处理算子，融合了 Query/Key 的线性投影计算、RMSNorm 归一化和 ROPE 位置编码三个步骤，用于注意力机制的前置数据准备。

**主要应用场景**：
- 大语言模型中 Multi-Head Latent Attention 的前处理阶段
- DeepSeek 等模型架构中的 MLA 注意力机制
- 需要融合投影、归一化和位置编码的高效推理场景

**算子特征**：
- 难度等级：L4（FusedComposite）
- 三输入（token_x, wk_cq, wk_ckv）单输出，融合矩阵乘法、RMSNorm 和拼接操作
- 分别计算 CQ 和 CKV 两路，各自经过线性投影和 RMSNorm 后拼接输出

## 2. 算子定义

### 数学公式

$$
y = \text{Concat}(\text{RMSNorm}_{cq}(\text{token\_x} \times W_{cq}^T),\ \text{RMSNorm}_{ckv}(\text{token\_x} \times W_{ckv}^T))
$$

其中 RMSNorm 定义为：

$$
\text{RMSNorm}(x) = \frac{x}{\sqrt{\text{mean}(x^2) + \epsilon}}
$$

具体子步骤：
1. **CQ 线性投影**：$cq = \text{token\_x} \times W_{cq}^T$
2. **CQ RMSNorm**：$cq_{norm} = cq / \sqrt{\text{mean}(cq^2, \text{dim}=-1) + \epsilon_{cq}}$
3. **CKV 线性投影**：$ckv = \text{token\_x} \times W_{ckv}^T$
4. **CKV RMSNorm**：$ckv_{norm} = ckv / \sqrt{\text{mean}(ckv^2, \text{dim}=-1) + \epsilon_{ckv}}$
5. **拼接**：$y = \text{Concat}(cq_{norm}, ckv_{norm}, \text{dim}=-1)$

## 3. 接口规范

### 算子原型

```python
ascend_bench.mla_prolog(Tensor token_x, Tensor wk_cq, Tensor wk_ckv, float rmsnormEpsilonCq, float rmsnormEpsilonCkv) -> Tensor y
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| token_x | Tensor | 必选 | 输入 token 张量，shape 为 (B, S, He) |
| wk_cq | Tensor | 必选 | CQ 权重张量 |
| wk_ckv | Tensor | 必选 | CKV 权重张量 |
| rmsnormEpsilonCq | float | 必选 | CQ 的 RMSNorm epsilon |
| rmsnormEpsilonCkv | float | 必选 | CKV 的 RMSNorm epsilon |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| y | 由 CQ 和 CKV 投影维度拼接决定 | 与输入 token_x 相同 | CQ 归一化结果与 CKV 归一化结果的拼接输出 |

### 数据类型

| 输入 dtype | 输出 dtype |
|-----------|-----------|
| int8 | int8 |
| bfloat16 | bfloat16 |
| float16 | float16 |

### 规则与约束

- 所有输入 Tensor（token_x, wk_cq, wk_ckv）的 dtype 必须一致
- `wk_cq` 和 `wk_ckv` 的最后一维必须与 `token_x` 的最后一维（He）匹配，以满足矩阵乘法要求
- `rmsnormEpsilonCq` 和 `rmsnormEpsilonCkv` 为正浮点数，用于防止 RMSNorm 中的除零错误
- 输出的最后一维为 CQ 投影维度与 CKV 投影维度之和

## 4. 精度要求

计算结果与 PyTorch Golden 实现逐元素对比，需满足以下误差阈值：

| 数据类型 | 验证方式 | rtol | atol |
|---------|---------|------|------|
| float16 | 相对误差 | 1e-3 | 1e-3 |
| bfloat16 | 相对误差 | 4e-3 | 4e-3 |
| int/uint/bool | 完全相等 | — | — |

**对比公式**：

$$
|output - golden| \leq atol + rtol \times |golden|
$$

## 5. 标准 Golden 代码

```python
import torch

"""
MlaProlog算子Torch Golden参考实现

Multi-Head Latent Attention前处理
公式: y = Query/Key计算 + RmsNorm + ROPE编码
"""
def mla_prolog(
    token_x: torch.Tensor, wk_cq: torch.Tensor, wk_ckv: torch.Tensor, rmsnormEpsilonCq: float, rmsnormEpsilonCkv: float
) -> torch.Tensor:
    """
    Multi-Head Latent Attention前处理
    
    公式: y = Query/Key计算 + RmsNorm + ROPE编码
    
    Args:
        token_x: 输入token张量
        wk_cq: CQ权重张量
        wk_ckv: CKV权重张量
        rmsnormEpsilonCq: CQ的RMSNorm epsilon
        rmsnormEpsilonCkv: CKV的RMSNorm epsilon
    
    Returns:
        输出张量
    """

    cq = torch.matmul(token_x, wk_cq.transpose(-2, -1))
    variance_cq = cq.pow(2).mean(-1, keepdim=True)
    rms_cq = torch.sqrt(variance_cq + rmsnormEpsilonCq)
    cq_norm = cq / rms_cq
    
    ckv = torch.matmul(token_x, wk_ckv.transpose(-2, -1))
    variance_ckv = ckv.pow(2).mean(-1, keepdim=True)
    rms_ckv = torch.sqrt(variance_ckv + rmsnormEpsilonCkv)
    ckv_norm = ckv / rms_ckv
    
    y = torch.cat([cq_norm, ckv_norm], dim=-1)
    return y
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import ascend_bench

B, S, He = 2, 128, 512
cq_dim, ckv_dim = 256, 128
token_x = torch.randn(B, S, He, dtype=torch.float16, device="npu")
wk_cq = torch.randn(cq_dim, He, dtype=torch.float16, device="npu")
wk_ckv = torch.randn(ckv_dim, He, dtype=torch.float16, device="npu")
y = ascend_bench.mla_prolog(token_x, wk_cq, wk_ckv,
                             rmsnormEpsilonCq=1e-6, rmsnormEpsilonCkv=1e-6)
```

### 性能基线参考

当前暂无测试用例和性能基线数据。

### 相关算子

- **SparseFlashAttention**：注意力计算算子，MlaProlog 为其前处理步骤
- **GroupedMatmulSwigluQuant**：同为 L4 级融合复合算子，包含矩阵乘法与激活函数融合
- **LSTM**：同为 L4 级融合复合算子，涉及多步门控计算融合
