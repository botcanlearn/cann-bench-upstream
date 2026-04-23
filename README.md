# cann-bench: CANN 算子生成评测基准

评测AI模型生成Ascend C算子代码的能力，涵盖编译正确性、功能精度、性能优化三大维度，支撑模型选型、训练效果评估，推动AI能力在CANN领域的持续演进。

## 👋 Task Description

评测AI模型生成Ascend C算子代码的能力，按算子复杂度分为4个等级：

- **Level 1**: 基础算子 (Element-wise, Activation)
  单输入单输出、Elewise操作、无特殊优化，如 Add、Exp、Gelu、Sigmoid、Mish
- **Level 2**: 中级算子 (Normalization, Reduction, Gather/Scatter)
  多输入、轻量级Broadcast、需Tiling但策略固定，如 Gather、ApplyAdamW、Softmax
- **Level 3**: 高级算子 (Conv, Pooling, MoE)
  多维度归约、多Tiling策略可选，如 TopK、Conv2D、Matmul、NMS
- **Level 4**: 复杂算子 (Attention, RNN)
  矩阵运算、多算子融合、复杂数据流，如 FlashAttention、LSTM、GRU

## ⚖️ Evaluation

### 三层评测框架

- **数据层**：评测任务集（算子规格描述、Golden实现、测试用例、泛化验证集）
- **评测层**：评测维度（编译正确性、功能精度、性能优化性）
- **应用层**：评测报告、CI流水线工程、问题改进、评测结果网站

### 核心评测指标

| 维度 | 指标 | 权重 | 说明 |
|------|------|------|------|
| 编译正确性 | Pass/Fail | Wc=2 | 是否编译通过（官方提交单份代码，二值判定） |
| 功能正确性 | 精度用例通过数 | Wf=3 | 通过精度用例的数量 |
| 性能优化性 | 加速比 (SpeedUp) | Wp=5 | 验证性能/测试基准性能 |

单算子综合评分 = 编译通过得分 + 功能通过用例数 × (功能得分 + 性能得分)
  - 编译通过得分 = compile_pass × Wc    # 单算子一次，整份提交的编译结果
  - 功能得分     = Wf                   # 每个功能通过的用例
  - 性能得分     = SpeedUp × Wp         # 每个功能通过的用例（按该用例实测）

Level-N 得分 = 该 level 内所有算子综合评分之和
benchmark 总分 = 所有算子综合评分之和（= Level1 + Level2 + Level3 + Level4）

## 🔍 Directory Structure

```
cann-bench/
├── kernel_bench/           # 算子生成评测任务
│   ├── level1/             # 基础算子
│   ├── ...                 # 中级算子
│   └── level4/             # 复杂算子
├── bench_lab/              # 实验室级测试用例(后续版本会规划进主评测集)
│   └── kernel_bench/       # 实验室级算子评测任务
├── examples/               # 示例代码工程
│   ├── aclnn_launch_example/        # ACLNN 算子工程样例
│   └── direct_launch_example/       # 直接算子工程样例
├── docs/                   # 设计文档
├── scripts/                # 测试脚本
│   ├── run_test.sh         # 统一测试运行脚本
│   └── run_evaluation.py   # 评测运行脚本
├── src/                    # 源代码
│   └── kernel_eval/        # 算子评测模块
├── test/                   # 测试代码
├── requirements.txt        # Python 依赖
├── LICENSE                 # 许可证文件
└── README.md               # 项目说明文档
```

## 🔧 Setup

### 环境要求

- Python 3.8+
- PyTorch 2.3+
- NumPy 1.21+

### 安装依赖

```bash
pip install -r requirements.txt
```

## 🚀 Quick Start

### 运行测试

```bash
# 运行所有测试（默认 CPU）
./scripts/run_test.sh

# 使用 NPU 设备测试
./scripts/run_test.sh --npu

# 运行指定 level 测试
./scripts/run_test.sh --level 1
./scripts/run_test.sh --level 2
./scripts/run_test.sh --level 3
./scripts/run_test.sh --level 4

# 运行指定算子测试
./scripts/run_test.sh --operator gelu
./scripts/run_test.sh --operator softmax

# 运行指定用例
./scripts/run_test.sh --operator gelu --case-id 1

# 详细输出
./scripts/run_test.sh --cpu --level 1 -v

# 启用性能采集（NPU 测试）
./scripts/run_test.sh --npu --prof

# 查看帮助
./scripts/run_test.sh --help
```

### 测试结果

测试结果默认保存在 `test/reports/` 目录：

```bash
# 查看测试报告
cat test/reports/test_results.json

# 指定自定义输出路径
./scripts/run_test.sh --output my_results.json

# 查看性能 trace（NPU 测试 + --prof）
ls test/reports/traces/
```

## 📋 Test Case Structure

每个算子目录下包含以下文件：

| 文件 | 说明 |
|------|------|
| `cases.csv` | 测试用例 CSV 格式 |
| `golden.py` | PyTorch 参考实现，用于结果验证 |
| `proto.yaml` | 算子原型定义 |
| `desc.md` | 算子详细说明文档 |

### 待评测算子工程样例

项目提供了多种算子开发示例：

1. **ACNN 算子启动示例**：[examples/aclnn_launch_example/](examples/aclnn_launch_example/)
2. **直接算子启动示例**：[examples/direct_launch_example/](examples/direct_launch_example/)

这些示例演示如何使用 Ascend C 和 PyTorch Extension 开发自定义 NPU 算子。

使用自定义算子：

```python
import torch
import torch_npu
import cann_bench.kernel_bench

x = torch.randn(10, 32, dtype=torch.float32).npu()
y = torch.randn(10, 32, dtype=torch.float32).npu()
result = cann_bench.kernel_bench.add(x, y)
```

## ➕ Add New Operator

在 `bench_lab/{problems}/` 目录下创建算子文件夹
1. 创建 `proto.yaml` 定义算子原型
2. 创建 `golden.py` 实现 PyTorch 参考代码
3. 创建 `desc.md` 编写算子说明文档
4. 创建 `cases.yaml/cases.csv` 编写测试用例


## 🛣️ Roadmap

- 工程平台构建
  - [ ] 完成剩余 Level3/Level4 算子核对验证, 发布第一版算子评测集合
  - [ ] 建立持续评测 CI 流水线
  - [ ] 发布评测结果网站

- 评测集构建
  - [ ] 增加更多算子类型覆盖
  - [ ] 根据领域场景分类，算子特征等，构建出更多独立榜单集合，覆盖不同评测场景的需求

- 评测标准构建
  - [ ] 评测精度标准，精度衡量方法构建
  - [ ] 评测性能基线，理论性能评估
  - [ ] 评分算法优化（例如算子复杂度、用例难度），科学评价生成能力
  - [ ] 算子分级/分类方法


## 🪪 License

CANN Open Software License Agreement Version 2.0