# AdaptiveAvgPool3D 算子 API 描述

## 1. 算子简介

完成输入张量的3D自适应平均池化计算。

**主要应用场景**：
- 3D 视频特征的空间和时间维度自适应降采样
- 点云和体素数据的空间压缩
- 全局平均池化（output_size=1）用于分类网络的特征聚合
- 不同分辨率输入统一到固定尺寸输出

**算子特征**：
- 难度等级：L3（Reduction）
- 单输入单输出，输入为 [N, C, D, H, W] 5维张量，输出空间维度由 output_size 决定

## 2. 算子定义

### 数学公式

$$
y = \text{adaptive\_avg\_pool3d}(x, \text{output\_size})
$$

自适应平均池化根据目标输出尺寸自动计算每个输出位置对应的池化窗口大小和步长，对窗口内元素取平均值。对于每个输出位置 $(d, h, w)$，其对应的输入区域由 output_size 和输入尺寸共同决定。

## 3. 接口规范

### 算子原型

```python
cann_bench.adaptive_avg_pool3_d(Tensor x, list[int] output_size) -> Tensor y
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| x | Tensor | 必选 | 输入张量，shape 为 [N, C, D, H, W] 的5维张量 |
| output_size | list[int] | 必选 | 输出尺寸，格式为 [output_d, output_h, output_w] |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| y | [N, C, output_size_d, output_size_h, output_size_w] | 与输入 x 相同 | 输出张量，池化结果 |

### 数据类型

| 输入 dtype | 输出 dtype |
|-----------|-----------|
| float32 | float32 |
| float16 | float16 |
| bfloat16 | bfloat16 |

### 规则与约束

- 输入必须为5维张量，shape 格式为 [N, C, D, H, W]
- output_size 指定输出的空间维度大小
- 输出 dtype 与输入 dtype 一致
- 输出的 N 和 C 维度与输入保持一致，仅空间维度 (D, H, W) 发生变化

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
AdaptiveAvgPool3D算子Torch Golden参考实现

完成输入张量的3D自适应平均池化计算
公式: y = adaptive_avg_pool3d(x, output_size)
"""
def adaptive_avg_pool3_d(
    x: torch.Tensor, output_size: tuple[int, int, int]
) -> torch.Tensor:
    """
    完成输入张量的3D自适应平均池化计算

    公式: y = adaptive_avg_pool3d(x, output_size)

    Args:
        x: 输入张量，shape 为 [N, C, D, H, W]
        output_size: 输出尺寸，格式为 (output_d, output_h, output_w)

    Returns:
        输出张量，池化结果
    """

    y = torch.nn.functional.adaptive_avg_pool3d(x, output_size)
    return y
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

x = torch.randn(2, 32, 16, 64, 64, dtype=torch.float16, device="npu")
y = cann_bench.adaptive_avg_pool3_d(x, [8, 8, 8])  # 自适应池化到 8x8x8

x = torch.randn(2, 64, 32, 128, 128, dtype=torch.float32, device="npu")
y = cann_bench.adaptive_avg_pool3_d(x, [1, 1, 1])  # 全局平均池化
```

### 性能基线参考

基于 cases.yaml 中 20 个测试用例，所有用例的 baseline_perf_us 均为 None，基线性能尚未测量。测试用例覆盖 float16、float32、bfloat16 三种数据类型，输入尺寸从 [1, 128, 16, 32, 32] 到 [8, 8, 8, 32, 512]，输出尺寸从 [1, 1, 1]（全局池化）到 [16, 16, 2]，包含非对齐质数 shape、零值输入和特殊值范围等边界场景。

### 相关算子

- **TopK**：同为 L3 级别的归约类算子，沿指定维度选取最大 K 个值
- **RoiPooling**：区域池化操作，对感兴趣区域进行空间池化
- **RoiAlign**：双线性插值的区域对齐池化，常用于目标检测
