# 算子代码生成评测方案V1.0.0

## 1. 方案概述

本评测方案建立了一套AI生成Ascend C算子代码评测体系，用于量化评估AI生成的算子代码质量，涵盖编译正确性、功能正确性、性能优化性三个核心维度，支撑模型选型、训练效果评估，推动AI能力在算子领域的持续演进。

### 1.1 版本演进

| 版本 | 主要变更 |
|------|----------|
| V1.0.0 | 初版，建立基础评测框架|
| V0.2.0 | 引入Pass@k评测、算子分类体系、三大维度评测 |
| V0.3.0 | 完善算子复杂度定义、规范用例输入输出 |
| V0.4.0 | 增加评测报告结构、优化评测流程 |
| V0.5.0 | 规范算子交付件要求 |
| V0.6.0 | 调整算子，明确第一版55个算子|
---

### 1.2 演进
Bench会持续更新版本

## 2. 评测体系架构

### 2.1 三层评测框架

- **数据层**：评测任务集（算子规格描述、算子Golden实现、测试样例、 内部泛化验证集）
- **评测层**：三大维度评测（编译、功能精度、性能）
- **应用层**：评测报告、CI流水线工程、问题和改进、CANN评测结果网站等

### 2.2 核心评测指标

- **编译正确性**: Pass@k指标
- **功能正确性**: 精度用例通过数量 
- **性能优化性**: 相比基准性能的加速比(当前)；相比理论性能的比例(规划)
```
综合评分
├── 编译正确性 (权重分 Wc=2)
│   └── Pass@1 (编译通过率)
├── 功能正确性 (权重分 Wf=3)
│   └── 用例通过数 (通过精度用例的数量) 
└── 性能优化性 (权重分 Wp=5)
    └── 加速比 (验证性能/测试基准性能)

计算方式：
编译得分：Pass@1 x Wc
功能得分：Pass@1 x Wf
性能得分：SpeedUp x Wp
综合评分 = 编译通过用例数 * 编译得分 + 功能通过用例数 × (功能得分 + 性能得分)
```


### 2.3 评测流程

1. AI算子生成（基于算子原型定义要求）
  - 输入：基于operator-info.yaml中的算子原型定义要求
  - 输出：算子工程源码
2. 本地评测
  - 编译评测：
    - 输入：AI生成的算子代码
    - 输出：Pass@1指标（建议生成5个候选，计算Pass@1）
  - 功能精度评测：
    - 输入：基于cases.yaml中算子测试用例和提供的Golden函数进行功能精度用例测试
    - 输出：算子用例通过率
  - 性能评测：
    - 输入：基于cases.yaml中算子基准性能，使用msprof工具获取**通过精度用例**的算子性能
    - 输出：算子性能/算子性能相比基准性能的比例（SpeedUp）
3. CANN官方评测：
  - 输入：按照指定格式提供算子源码包（要求提供模型信息/每个算子任务的Prompt信息/算子工程源码）；接口定义要求和proto.yaml中schema定义一致
  - 输出：CANN官方CI工程输出评测报告（基于内部用例集, 每个算子100泛化用例）

## 3. 数据层

数据层主要包含了一系列的评测任务集合，当前评测主要针对端到端算子生成任务，发布的评测任务集以各种算子开发任务为主，涉及不同开发难度的算子。
数据层给出的信息，都可以作为Prompt的输入信息给到模型。

### 3.1 算子分类和难度等级定义

**算子分类**

**算子难度等级**

算子根据计算流、计算模式不同，可以大致分为以下几个难度等级：

| 等级 | 特征描述 | AI生成难度 |代表算子 |
|------|----------|------------|------------|
| L1 | 单输入单输出、Elewise操作、无特殊优化 | 简单 |Add、Exp、MaskedScale |
| L2 | 多输入、轻量级Broadcast、需Tiling但策略固定 | 中等 |Gather、ApplyAdamW、Gelu |
| L3 | 多维度归约、多Tiling策略可选 | 较难 | TopK、AvgPool、Matmul、Conv2D、BatchMatMul |
| L4 | 矩阵运算、多算子融合、复杂数据流、需要极致性能调优 | 困难 |FlashAttentionScore、LSTM|

不同level在总分计量中会有不同的分值权重。

### 3.2 评测算子清单

结合Ascend C算子开发特点，以当前CANN仓中算子作为基础评测任务，并加入部分[KernelBench](https://github.com/ScalingIntelligence/KernelBench)评测任务作为扩展用例。

**算子定义文件：** operator-info.yaml
```yaml
# operator-info.yaml
- name: Exp
    category: Elementwise
    difficulty: L1
    formula: "y = e^((x * scale + shift) * ln(base))"
    description: "计算输入张量的指数函数，支持自定义底数、缩放和偏移"
    shape_support: "输入任意维度，输出与输入相同shape"
    attrs:
      - name: base
        type: float
        default: -1.0
        description: "指数底数，-1.0表示使用自然底数e，正值表示自定义底数"
      - name: scale
        type: float
        default: 1.0
        description: "输入缩放因子"
      - name: shift
        type: float
        default: 0.0
        description: "输入偏移量"
    note: "当base=-1时，公式简化为 y = e^(x * scale + shift)"
    inputs:
      - name: x
        description: 输入张量
        dtype: ["float16", "float32", "bfloat16"]
    outputs:
      - name: y
        description: 指数计算结果
        dtype: ["float16", "float32", "bfloat16"]
    schema: exp(Tensor x, float base, float scale, float shift) -> Tensor y
```
> 算子自定义TorchAPI接口，需要与schema一致, lib空间统一为`ascend_bench`
```
import ascend_bench
y = ascend_bench.exp(x, -1.0, 1.0, 0.0)
```

**L1算子**
- Elewise：Exp、MaskedScale
- 激活函数：Gelu、Sigmoid、SwiGLU、Mish
- Foreach类：ForeachNorm、ForeachAddcdivScalar

**L2算子**
- 优化器算子：ApplyAdamW
- Broadcast：Maximum、Gcd
- 量化算子：DynamicQuant
- 损失函数算子：CrossEntropyLoss
- 索引操作：Gather、Scatter、UnsortedSegmentSum(仅Int)
- 插值类：ResizeBilinearV2、GridSampler3D
- Reduce：ArgMax、Cummin
- 正则化：Softmax、RMSNorm、GroupNorm
- Transform类：ApplyRotaryPosEmb

**L3算子**
- 池化算子：AdaptiveAvgPool3D
- 张量变换：Transpose、StridedSlice
- 排序类：TopK、Unique
- Hash类：EmbeddingHashLookupOrInsert
- 图像处理：Dilation2D
- 目标检测：NMSWithMask、ROIAlign、ROIPooling
- MoE类：MoeReRouting、MoeFinalizeRoutingV2、MoeGatingTopKSoftmax
- 矩阵运算：GroupedMatmul
- MM量化：QuantBatchMatmul、WeightQuantBatchMatmul
- 卷积：Conv2D、DepthwiseConv2D
- 卷积反向：Conv3DBackpropFilter
- VV融合：AddRmsNormDynamicQuant、DequantSwigluQuant、MhcSinkhorn、Engram
 

**L4算子**
- Transformer类：MHA、GQA、MLA、SparseFlashAttention、MlaProlog
- RNN类：LSTM、GRU
- 量化融合：GroupedMatmulSwigluQuant


- 不同难度等级的算子会有不同的分值

### 3.2 算子评测用例

该评测体系有两种用例：
- 开放用例：随算子评测标准一起发布，由算子任务集中算子典型场景Shape和Attr属性组合（20+）；
- 官方评测用例：不随算子评测标准发布，按照指定工程形式提供自定义算子源码文件，由CANN官方CI工程完成评测。

**算子评测用例设计原则**
- 用例生成：输入输出（Shape维度/数据类型）、属性泛化、取值范围泛化、特殊值，常见网络Shape/网络Shape泛化

**算子用例定义文件：** cases.csv
```
operator,case_id,input_shape,dtype,attrs,value_range,baseline_perf_us,note
Exp,1,"[[1024, 1024]]",['float16'],"{'base': -1.0, 'scale': 1.0, 'shift': 0.0}","[-1, 1]",21.05,float16-1M-对齐-对称小值域-base=-1
Exp,2,"[[2048, 2048]]",['float32'],"{'base': -1.0, 'scale': 1.5, 'shift': 0.0}","[-2, 2]",18.27,float32-4M-对齐-对称小值域-scale=1.5
```

### 3.3 Golden脚本

根据proto.yaml中算子的定义，提供相应算子的Golden脚本

实现方式：基于pytorch官方API
```python
def exp(
    x: torch.Tensor,
    base: float = -1.0,
    scale: float = 1.0,
    shift: float = 0.0
) -> torch.Tensor:
    """
    计算输入张量的指数函数（核心Golden计算逻辑）

    公式: y = base^(x * scale + shift)，当base=-1时，y = e^(x * scale + shift)

    Args:
        x: 输入张量
        base: 底数，默认-1.0表示使用e
        scale: 缩放因子，默认1.0
        shift: 偏移量，默认0.0

    Returns:
        输出张量 y
    """
    temp = x * scale + shift
    if base == -1.0:
        y = torch.exp(temp)
    else:
        y = torch.exp(temp * torch.log(torch.tensor(base, dtype=x.dtype, device=x.device)))
    return y
```
---

## 4. 评测层

### 4.1 三大评测维度

| 维度 | 权重 | 评测重点 | 评测工具 | 评分范围 |
|------|------|----------|----------|----------|
| 编译正确性 | Wc=2 | 编译通过 | cmake、gtest | [0, 100] |
| 功能正确性 | Wf=3 | 通过测试用例Golden对比 | cmake、gtest | [0, 100] |
| 性能优化性 | Wp=5 | 相比基准时间的比例| msprof、cannsim | [0, 100] |

### 4.3 Pass@k评测

业界标准的代码生成评测方法，生成k个候选代码中至少有1个通过所有测试用例的概率。

**简化计算**
```
Pass@k = 1 - C(n-c, k) / C(n, k)
```
- **n**: 生成的候选代码总数
- **c**: 通过测试的候选代码数量
- **k**: 选择的最优候选数量（通常取1、5、10）

**简化计算**
```
Pass@1 = c / n
```
即：单次生成通过率 = 通过数 / 总生成数

> 官方评测中，由于只要求提供一份源码，因此Pass@1只有可能是0或者1两种取值

#### 4.4 精度标准

当前采用单精度标准，等新精度标准社区完善后，采用新精度标准

| 数据类型 | 验证方式 | 误差阈值 |
|---------|---------|---------|
| float16 | 相对误差 | rtol=1e-03, atol=1e-03 |
| float32 | 相对误差 | rtol=1e-04, atol=1e-04 |
| bfloat16 | 相对误差 | rtol=4e-03, atol=4e-03 |
| int32/int64/int16/int8 | 完全相等 | - |
| uint32/uint64/uint16/uint8 | 完全相等 | - |
| bool | 完全相等 | - |

**对比公式**:
```
|output - golden| ≤ (atol + rtol × |golden|)
```

>精度新标准：参考[算子精度标准](https://gitcode.com/cann/opbase/tree/master/docs/zh/ops_precision_standard)

### 4.5 性能评测

性能评测流程
```
功能通过 → 预热执行 → 正式性能测试 → 计算统计结果 → 与基准对比 → 计算加速比
```

**性能采集方式1**
预热3次后，执行100次取中位数，使用msprof工具，先执行3次预热消除缓存影响，再执行100次正式测试，取中位数作为最终执行时间，排除异常值影响
```
msprof op --warm-up=3 --launch-count=100 --output=./msprof_output ./your_op arg1 arg2
```
**性能采集方式2**
基于torch_npu.profiler的性能采集方案，通过解析chrome trace JSON获取NPU内核执行时间。

```python
import torch_npu

# 配置 experimental_config
experimental_config = torch_npu.profiler._ExperimentalConfig(
    export_type=[torch_npu.profiler.ExportType.Text],
    profiler_level=torch_npu.profiler.ProfilerLevel.Level0,
    aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone,
)

# 使用 schedule 机制：warmup预热 + repeat采集
with torch_npu.profiler.profile(
    activities=[
        torch_npu.profiler.ProfilerActivity.CPU,
        torch_npu.profiler.ProfilerActivity.NPU,
    ],
    schedule=torch_npu.profiler.schedule(
        wait=0, warmup=3, active=5, repeat=1
    ),
    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(prof_dir),
    record_shapes=False,
    profile_memory=False,
    with_stack=False,
    experimental_config=experimental_config,
) as prof:
    # 执行 warmup + active 次循环
    for _ in range(warmup + repeat):
        outputs = func(*args, **kwargs)
        prof.step()
```

**性能数据解析**
解析生成的trace_view.json文件，通过cat字段区分Host/Device阶段，获取NPU内核执行时间：
```python
# 解析 chrome trace JSON
events = data.get('traceEvents', [])
for event in events:
    if event.get('ph') != 'X':  # 只处理完整事件
        continue
    dur = event.get('dur', 0)
    name = event.get('name', '')

    # 通过 cat 字段判断：有 cat = Host端，无 cat = Device端
    if 'cat' in event:
        host_ops[name] = host_ops.get(name, 0) + dur
    else:
        device_kernels[name] = device_kernels.get(name, 0) + dur
        total_kernel_us += dur

# 对 repeat 次采集结果取平均
kernel_time_us = total_kernel_us / repeat
```

**采集参数配置**
| 参数 | 默认值 | 说明 |
|------|--------|------|
| warmup | 3 | 预热次数，消除缓存影响 |
| repeat | 5 | 正式采集次数 |
| ProfilerLevel | Level0 | 采集详细程度 |
| export_type | Text | 输出格式 |

**特点**
- 基于Python接口，无需外部命令行工具
- 支持异步解析，不影响后续测试执行
- 自动归档profiling数据到 `test/reports/prof_data/{level}/{op_name}/{caseid}/`
- 可获取Host端和Device端各算子的详细耗时

## 5 应用层
评测报告、评测工程、问题和改进、CANN评测结果网站等
### 5.1 评测报告
**评测报告核心要素**
- 评测集版本号：每个版本号对应明确的算子任务清单和开放和未开放的用例集合，用例验收标准以及性能基线
- 评测代号：自定义提交评测任务的组织代号
- 基础模型：GLM5/Opus等
- Agent/Skill: CANNBot等
- 综合得分：计算综合得分
- 子项得分：评估各个子项维度的得分（编译、功能精度、性能）
- 各算子任务得分情况

### 5.2 评测工程
- 提供自动化编译、评测、输出报告的能力
```
┌────────────────────────────────────────────────────────────────┐
│                        CANN Evaluator                          │
│                     (主调度器 - asyncio)                        │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    Task State Manager                    │  │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐     │  │
│  │  │  PENDING │ │ RUNNING  │ │COMPLETED │ │ FAILED   │     │  │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘     │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                │
│  ┌────────────────┐  ┌────────────────────┐  ┌────────────────┐│
│  │  AI Worker     │  │  Validation Worker │  │  Result Worker ││
│  │  (每个算子独立) │  │      Pool          │  │   (可选)       ││
│  │                │  │                    │  │                ││
│  │ ┌───┐ ┌───┐    │  │ ┌───┐ ┌───┐ ┌───┐  │  │                ││
│  │ │W1 │ │W2 │... │  │ │W1 │ │W2 │ │W3 │  │  │                ││
│  │ └───┘ └───┘    │  │ └───┘ └───┘ └───┘  │  │                ││
│  └────────────────┘  └────────────────────┘  └────────────────┘│
│         │                    │                                 │
│         ▼                    ▼                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                      asyncio.Queue                       │  │
│  │  ┌──────────────────┐  ┌──────────────────┐              │  │
│  │  │   Code Queue     │  │ Validate Queue   │              │  │
│  │  │ (AI→Compile)     │  │ (Compile→Acc→Perf)│             │  │
│  │  └──────────────────┘  └──────────────────┘              │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │  Test Case  │  │   AI Model  │  │    Report Generator     │ │
│  │   Loader    │  │   Interface │  │                         │ │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘ │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```
