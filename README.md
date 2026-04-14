# cann-bench

CANN 评测仓库，用于量化评估在CNNN领域的AI生成任务的能力，涵盖算子Kernel生成等多场景评测，支撑模型选型、训练效果评估，推动AI能力在CANN领域的持续演进。

## 目录结构

```
cann-bench/
├── kernel_bench/         # 算子生成评测任务，按算子任务复杂度分级
│   ├── level1/           # 基础算子（Element-wise, Activation）
│   ├── level2/           # 中级算子（Normalization, Reduction, Gather/Scatter）
│   ├── level3/           # 高级算子（Conv, Pooling, MoE）
│   └── level4/           # 复杂算子（Attention, RNN）
├── bench_lab/            # 实验室级测试用例
├── examples/             # 示例代码
│   └── fast_kernel_launch_example/  # 快速算子开发示例
├── requirements.txt      # Python 依赖
└── README.md
```

## 快速开始

### 环境要求

- Python 3.8+
- PyTorch 2.3+
- NumPy 1.21+

### 安装依赖

```bash
pip install -r requirements.txt
```

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

#### 测试结果

测试结果保存在 `test/reports/` 目录：

```bash
# 查看测试报告
cat test/reports/test_results.json

# 查看性能 trace（NPU 测试 + --prof）
ls test/reports/traces/
```

## 测试用例结构

每个算子目录下包含以下文件：

| 文件 | 说明 |
|------|------|
| `cases.yaml` | 测试用例配置，定义输入参数和预期输出 |
| `cases.csv` | 测试用例 CSV 格式 |
| `golden.py` | PyTorch 参考实现，用于结果验证 |
| `proto.yaml` | 算子原型定义 |
| `desc.md` | 算子详细说明文档 |

## 待评测算子工程样例

### 快速算子开发示例

参见 [examples/fast_kernel_launch_example/](examples/fast_kernel_launch_example/) 目录。

该示例演示如何使用 Ascend C 和 PyTorch Extension 开发自定义 NPU 算子：

```bash
cd examples/fast_kernel_launch_example
pip install -r requirements.txt
pip install dist/*.whl --force-reinstall --no-deps
```

使用自定义算子：

```python
import torch
import torch_npu
import cann_bench.kernel_bench

x = torch.randn(10, 32, dtype=torch.float32).npu()
y = torch.randn(10, 32, dtype=torch.float32).npu()
result = cann_bench.kernel_bench.add(x, y)
```

## 添加新算子

1. 在 `bench_lab/kernel_bench/level{N}/` 目录下创建算子文件夹
2. 创建 `cases.csv` 定义测试用例
3. 创建 `golden.py` 实现 PyTorch 参考代码
4. 创建 `proto.yaml` 定义算子原型
5. 创建 `desc.md` 编写算子说明文档


## 许可证

