# ApplyAdamW 算子 API 描述

## 1. 算子简介

AdamW 优化器实现，解耦权重衰减。

**主要应用场景**：
- 深度学习模型参数优化，尤其是 Transformer 类模型的训练
- 解耦权重衰减正则化，避免 L2 正则化与自适应学习率的耦合问题
- 大规模预训练模型（如 GPT、BERT）的优化器实现

**算子特征**：
- 难度等级：L2（FusedComposite）
- 四输入（var、grad、m、v）单输出（y），逐元素运算，输出 shape 与输入一致
- 支持 1-8 维张量

## 2. 算子定义

### 数学公式

$$
m_t = \beta_1 \cdot m_{t-1} + (1 - \beta_1) \cdot grad
$$

$$
v_t = \beta_2 \cdot v_{t-1} + (1 - \beta_2) \cdot grad^2
$$

$$
\hat{m} = \frac{m_t}{1 - \beta_1^t}
$$

$$
\hat{v} = \frac{v_t}{1 - \beta_2^t}
$$

$$
var_t = var_{t-1} - lr \cdot \left( \frac{\hat{m}}{\sqrt{\hat{v}} + \epsilon} + weight\_decay \cdot var_{t-1} \right)
$$

其中：
- $m_t$ 为一阶矩估计（动量）
- $v_t$ 为二阶矩估计
- $\hat{m}$、$\hat{v}$ 为偏差修正后的矩估计
- 权重衰减（weight_decay）以解耦方式直接作用于参数

## 3. 接口规范

### 算子原型

```python
cann_bench.apply_adam_w(Tensor var, Tensor grad, Tensor m, Tensor v, float lr, float beta1, float beta2, float weight_decay, float epsilon=1e-8, bool maximize=false) -> Tensor y
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| var | Tensor | 必选 | 变量张量（需要优化的参数） |
| grad | Tensor | 必选 | 梯度张量 |
| m | Tensor | 必选 | 一阶矩张量（动量） |
| v | Tensor | 必选 | 二阶矩张量 |
| lr | float | 必选 | 学习率 |
| beta1 | float | 必选 | 一阶矩估计的指数衰减率 (默认 0.9) |
| beta2 | float | 必选 | 二阶矩估计的指数衰减率 (默认 0.999) |
| weight_decay | float | 必选 | 权重衰减系数（解耦） |
| epsilon | float | 1e-8 | 数值稳定常数 |
| maximize | bool | false | 是否最大化目标函数（默认最小化） |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| y | 与输入 var 相同 | 与输入 var 相同 | 更新后的变量 |

### 数据类型

| 输入 dtype | 输出 dtype |
|-----------|-----------|
| float32 | float32 |
| float16 | float16 |
| bfloat16 | bfloat16 |

### 规则与约束

- var、grad、m、v 四个张量的 shape 和 dtype 必须完全一致
- 支持 1-8 维张量
- 输出 shape 与输入 var 的 shape 完全一致，输出 dtype 与输入一致
- `beta1`、`beta2` 取值范围通常为 [0, 1)
- `epsilon` 用于防止除零，通常取极小正数
- 当 `maximize=true` 时，更新方向取反（用于最大化目标函数）

## 4. 精度要求

计算结果与 PyTorch Golden 实现逐元素对比，需满足以下误差阈值：

| 数据类型 | 验证方式 | rtol | atol |
|---------|---------|------|------|
| float16 | 相对误差 | 1e-3 | 1e-3 |
| float32 | 相对误差 | 1e-4 | 1e-4 |
| bfloat16 | 相对误差 | 4e-3 | 4e-3 |

**对比公式**：

$$
|output - golden| \leq atol + rtol \times |golden|
$$

## 5. 标准 Golden 代码

```python
import torch

"""
ApplyAdamW 算子 Torch Golden 参考实现

AdamW 优化器实现，解耦权重衰减
公式:
    m_t = beta1 * m_{t-1} + (1 - beta1) * grad
    v_t = beta2 * v_{t-1} + (1 - beta2) * grad^2
    m_hat = m_t / (1 - beta1^t)
    v_hat = v_t / (1 - beta2^t)
    var_t = var_{t-1} - lr * (m_hat / (sqrt(v_hat) + eps) + weight_decay * var_{t-1})
"""
def apply_adam_w(
    var: torch.Tensor,
    grad: torch.Tensor,
    m: torch.Tensor,
    v: torch.Tensor,
    lr: float,
    beta1: float,
    beta2: float,
    weight_decay: float,
    epsilon: float = 1e-8,
    maximize: bool = False
) -> torch.Tensor:
    """
    AdamW 优化器实现，解耦权重衰减

    Args:
        var: 变量张量（需要优化的参数）
        grad: 梯度张量
        m: 一阶矩张量（动量）
        v: 二阶矩张量
        lr: 学习率
        beta1: 一阶矩估计的指数衰减率
        beta2: 二阶矩估计的指数衰减率
        weight_decay: 权重衰减系数（解耦）
        epsilon: 数值稳定常数
        maximize: 是否最大化目标函数

    Returns:
        更新后的变量
    """
    # 更新一阶矩（动量）
    m.mul_(beta1).add_(grad, alpha=1 - beta1)
    # 更新二阶矩
    v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

    # 计算偏差修正的一阶矩和二阶矩
    # 注意：实际使用时需要传入 timestep t，这里简化处理
    bias_correction1 = 1 - beta1
    bias_correction2 = 1 - beta2

    m_hat = m / bias_correction1
    v_hat = v / bias_correction2

    # 计算更新量
    update = m_hat / (v_hat.sqrt() + epsilon)

    # 解耦的权重衰减
    if weight_decay != 0:
        update.add_(var, alpha=weight_decay)

    # 应用更新
    if maximize:
        y = var + lr * update
    else:
        y = var - lr * update

    return y
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

var = torch.randn(1024, 1024, dtype=torch.float32, device="npu")
grad = torch.randn(1024, 1024, dtype=torch.float32, device="npu")
m = torch.zeros(1024, 1024, dtype=torch.float32, device="npu")
v = torch.zeros(1024, 1024, dtype=torch.float32, device="npu")

y = cann_bench.apply_adam_w(var, grad, m, v, lr=0.001, beta1=0.9, beta2=0.999, weight_decay=0.01)
y = cann_bench.apply_adam_w(var, grad, m, v, lr=0.001, beta1=0.9, beta2=0.999, weight_decay=0.0, epsilon=1e-8, maximize=True)
```

### 性能基线参考

基于 cases.yaml 中 20 个测试用例，NPU 上的基准 kernel 执行时间范围为 53~676 微秒。

### 相关算子

- **DynamicQuant**：动态量化算子，同为 FusedComposite 类别的多步融合计算
- **ApplyRotaryPosEmb**：旋转位置编码，同为多输入融合算子
- **CrossEntropyLoss**：交叉熵损失，涉及 softmax 和对数计算的复合算子
