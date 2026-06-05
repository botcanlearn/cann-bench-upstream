# CANN-Bench算子测试报告

## 1. 概述
本报告覆盖CANN-Bench算子测试Level 1至Level 4四个测试级别的测试结果。本次测试验证了基础算子、中级算子、复杂算子和高级算子的功能正确性，测试范围包括激活函数、归一化算子、数学运算算子、图像处理算子、优化器算子、RNN算子、注意力机制算子等多种类型的算子测试用例。

## 2. 版本测试信息

**硬件和版本要求**

- 产品型号：CPU设备（默认）
- 操作系统：Linux 5.10.0-182.0.0.95.r2220_156.hce2.aarch64
- Python版本：3.12.9
- PyTorch版本：2.9.0+cpu
- torchvision版本：0.24.1
- 测试设备：CPU
- 测试Repo源：cann-bench

## 3. 测试结论

本版本测试，共计执行1056个测试用例，发现0个问题。整体质量良好，满足出口质量标准，建议发布。

- Level 1测试：160个用例，成功率100.00%
- Level 2测试：320个用例，成功率100.00%
- Level 3测试：416个用例，成功率100.00%
- Level 4测试：160个用例，成功率100.00%

## 4. 特性质量评估

|序号|特性|测试结论|功能|精度|性能|可靠性|兼容性|
|---|---|---|---|---|---|---|---|
|1|Level 1基础算子测试|通过|Pass|Pass|Pass|Pass|Pass|
|2|Level 2中级算子测试|通过|Pass|Pass|Pass|Pass|Pass|
|3|Level 3复杂算子测试|通过|Pass|Pass|Pass|Pass|Pass|
|4|Level 4高级算子测试|通过|Pass|Pass|Pass|Pass|Pass|

### 4.1 Level 1算子测试详情

Level 1测试涵盖8类算子，每类20个用例，共计160个用例：

|算子类型|用例数|通过数|通过率|
|---|---|---|---|
|Exp|20|20|100%|
|ForeachAddcdivScalar|20|20|100%|
|ForeachNorm|20|20|100%|
|Gelu|20|20|100%|
|MaskedScale|20|20|100%|
|Mish|20|20|100%|
|Sigmoid|20|20|100%|
|SwiGlu|20|20|100%|

执行耗时：~67s，平均每用例：~0.42s

### 4.2 Level 2算子测试详情

Level 2测试涵盖16类算子，每类20个用例，共计320个用例：

|算子类型|用例数|通过数|通过率|
|---|---|---|---|
|ApplyAdamW|20|20|100%|
|ApplyRotaryPosEmb|20|20|100%|
|ArgMax|20|20|100%|
|CrossEntropyLoss|20|20|100%|
|Cummin|20|20|100%|
|DynamicQuant|20|20|100%|
|Gather|20|20|100%|
|Gcd|20|20|100%|
|GridSampler3D|20|20|100%|
|GroupNorm|20|20|100%|
|Maximum|20|20|100%|
|ResizeBilinear|20|20|100%|
|RmsNorm|20|20|100%|
|Scatter|20|20|100%|
|Softmax|20|20|100%|
|UnsortedSegmentSum|20|20|100%|

执行耗时：~93s，平均每用例：~0.29s

### 4.3 Level 3算子测试详情

Level 3测试涵盖21类算子，共计416个用例：

|算子类型|用例数|通过数|通过率|
|---|---|---|---|
|AdaptiveAvgPool3D|20|20|100%|
|AddRmsNormDynamicQuant|20|20|100%|
|Conv2D|20|20|100%|
|Conv3DBackpropFilter|20|20|100%|
|DepthwiseConv2D|20|20|100%|
|DequantSwigluQuant|16|16|100%|
|Dilation2D|20|20|100%|
|Engram|20|20|100%|
|GroupedMatmul|20|20|100%|
|MhcSinkhorn|20|20|100%|
|MoeFinalizeRouting|20|20|100%|
|MoeGatingTopKSoftmax|20|20|100%|
|MoeReRouting|20|20|100%|
|NMS|20|20|100%|
|QuantMatmul|20|20|100%|
|ROIAlign|20|20|100%|
|StridedSlice|20|20|100%|
|TopK|20|20|100%|
|Transpose|20|20|100%|
|Unique|20|20|100%|
|WeightQuantBatchMatmul|20|20|100%|

执行耗时：~107s，平均每用例：~0.26s

### 4.4 Level 4算子测试详情

Level 4测试涵盖8类算子，每类20个用例，共计160个用例：

|算子类型|用例数|通过数|通过率|
|---|---|---|---|
|GQA|20|20|100%|
|GRU|20|20|100%|
|GroupedMatmulSwigluQuant|20|20|100%|
|LSTM|20|20|100%|
|MHA|20|20|100%|
|MLA|20|20|100%|
|MlaProlog|20|20|100%|
|SparseFlashAttention|20|20|100%|

执行耗时：~321s，平均每用例：~2.01s

## 5. DFX专项质量评估

### 5.1 安全测试
本次测试为功能验证测试，未涉及安全测试专项。

### 5.2 可靠性测试
|序号|可靠性特性|测试结论|遗留风险|
|---|---|---|---|
|1|算子功能稳定性|Pass|暂无|
|2|算子精度稳定性|Pass|暂无|

### 5.3 性能测试

|场景|算子类型|特性|性能指标|测试环境|测试结果|遗留风险|
|---|---|---|---|---|---|---|
|Level 1|Exp|数学运算|平均9.03ms|CPU|Pass||
|Level 1|ForeachAddcdivScalar|数学运算|平均12.23ms|CPU|Pass||
|Level 1|ForeachNorm|归一化|平均4.86ms|CPU|Pass||
|Level 1|Gelu|激活函数|平均8.54ms|CPU|Pass||
|Level 1|MaskedScale|激活函数|平均6.60ms|CPU|Pass||
|Level 1|Mish|激活函数|平均12.86ms|CPU|Pass||
|Level 1|Sigmoid|激活函数|平均3.02ms|CPU|Pass||
|Level 1|SwiGlu|激活函数|平均9.18ms|CPU|Pass||
|Level 2|ApplyAdamW|优化器|平均31.22ms|CPU|Pass||
|Level 2|ApplyRotaryPosEmb|位置编码|平均47.31ms|CPU|Pass||
|Level 2|ArgMax|数学运算|平均4.93ms|CPU|Pass||
|Level 2|CrossEntropyLoss|损失函数|平均6.37ms|CPU|Pass||
|Level 2|Cummin|数学运算|平均245.28ms|CPU|Pass||
|Level 2|DynamicQuant|量化|平均14.76ms|CPU|Pass||
|Level 2|Gather|数据操作|平均8.91ms|CPU|Pass||
|Level 2|Gcd|数学运算|平均15.94ms|CPU|Pass||
|Level 2|GridSampler3D|图像处理|平均516.90ms|CPU|Pass||
|Level 2|GroupNorm|归一化|平均1.26ms|CPU|Pass||
|Level 2|Maximum|数学运算|平均2.46ms|CPU|Pass||
|Level 2|ResizeBilinear|图像处理|平均4.40ms|CPU|Pass||
|Level 2|RmsNorm|归一化|平均6.89ms|CPU|Pass||
|Level 2|Scatter|数据操作|平均6.27ms|CPU|Pass||
|Level 2|Softmax|归一化|平均10.34ms|CPU|Pass||
|Level 2|UnsortedSegmentSum|数据操作|平均5.16ms|CPU|Pass||
|Level 3|AdaptiveAvgPool3D|池化|平均22.61ms|CPU|Pass||
|Level 3|AddRmsNormDynamicQuant|融合量化|平均78.83ms|CPU|Pass||
|Level 3|Conv2D|卷积|平均269.24ms|CPU|Pass||
|Level 3|Conv3DBackpropFilter|卷积反向|平均2337.43ms|CPU|Pass||
|Level 3|DepthwiseConv2D|深度卷积|平均305.21ms|CPU|Pass||
|Level 3|DequantSwigluQuant|量化融合|平均82.92ms|CPU|Pass||
|Level 3|Dilation2D|图像处理|平均51.71ms|CPU|Pass||
|Level 3|Engram|激活函数|平均153.79ms|CPU|Pass||
|Level 3|GroupedMatmul|矩阵运算|平均54.05ms|CPU|Pass||
|Level 3|MhcSinkhorn|注意力机制|平均30.80ms|CPU|Pass||
|Level 3|MoeFinalizeRouting|MoE路由|平均801.38ms|CPU|Pass||
|Level 3|MoeGatingTopKSoftmax|MoE门控|平均2.89ms|CPU|Pass||
|Level 3|MoeReRouting|MoE路由|平均250.55ms|CPU|Pass||
|Level 3|NMS|目标检测|平均880.66ms|CPU|Pass||
|Level 3|QuantMatmul|量化矩阵|平均97.55ms|CPU|Pass||
|Level 3|ROIAlign|目标检测|平均96.15ms|CPU|Pass||
|Level 3|StridedSlice|数据操作|平均0.06ms|CPU|Pass||
|Level 3|TopK|数学运算|平均9.90ms|CPU|Pass||
|Level 3|Transpose|数据操作|平均0.04ms|CPU|Pass||
|Level 3|Unique|数据操作|平均1145.11ms|CPU|Pass||
|Level 3|WeightQuantBatchMatmul|量化矩阵|平均17.57ms|CPU|Pass||
|Level 4|GQA|注意力机制|平均1348.00ms|CPU|Pass||
|Level 4|GRU|RNN|平均36.61ms|CPU|Pass||
|Level 4|GroupedMatmulSwigluQuant|融合算子|平均622.97ms|CPU|Pass||
|Level 4|LSTM|RNN|平均46.08ms|CPU|Pass||
|Level 4|MHA|注意力机制|平均418.16ms|CPU|Pass||
|Level 4|MLA|注意力机制|平均5231.42ms|CPU|Pass||
|Level 4|MlaProlog|MLA前处理|平均600.69ms|CPU|Pass||
|Level 4|SparseFlashAttention|稀疏注意力|平均782.30ms|CPU|Pass||

### 5.4 兼容性测试
兼容性评估：通过

|序号|兼容性场景|验证结果|遗留风险|
|---|---|---|---|
|1|CPU设备兼容|Pass||
|2|算子接口兼容|Pass||
|3|torchvision版本兼容(ROIAlign)|Pass|torchvision 0.24.1与torch 2.9.0兼容|
|4|TensorList格式兼容(GRU/LSTM)|Pass|对标PyTorch标准接口|

## 6. 测试执行评估

### 6.1 测试覆盖

|测试活动|测试结论|用例数|用例覆盖率|用例通过率|
|---|---|---|---|---|
|Level 1算子测试|Pass|160|100%|100%|
|Level 2算子测试|Pass|320|100%|100%|
|Level 3算子测试|Pass|416|100%|100%|
|Level 4算子测试|Pass|160|100%|100%|
|特性测试|Pass|1056|100%|100%|
|继承特性测试|Pass|1056|100%|100%|

## 7. 遗留问题和关键风险
本次测试未发现遗留问题。

### 7.1 遗留问题统计

||问题总数|严重|主要|次要|不重要|已取消|
|---|---|---|---|---|---|---|
|数目|0|0|0|0|0|0|
|百分比|0%|0%|0%|0%|0%|0%|

### 7.2 遗留问题列表

|问题单(issue链接)|问题描述|问题级别|问题影响和规避措施|当前状态|
|---|---|---|---|---|
|无|无|无|无|无|

## 8. 附件

测试结果已保存到：reports/test_results.json