# cann-bench

CANN 评测仓库，用于量化评估 CANN 领域下 AI 生成任务的能力，涵盖算子 kernel 生成等多场景评测，支撑模型选型、训练效果评估，推动 AI 能力在 CANN 领域的持续演进。

## 目录结构

```
cann-bench/
├── kernel_bench/               # 算子生成评测任务，按算子任务复杂度分级
│   ├── level1/                 # 基础算子（Element-wise, Activation）
│   ├── level2/                 # 中级算子（Normalization, Reduction, Gather/Scatter）
│   ├── level3/                 # 高级算子（Conv, Pooling, MoE）
│   └── level4/                 # 复杂算子（Attention, RNN）
├── bench_lab/                  # 实验室级测试用例
├── evaluation/                 # 标准化评测流水线（harness + evaluate.py + 工具）
│   ├── evaluate.py             # 主调度器
│   ├── harness/                # runner 调用的各阶段脚本
│   ├── core/                   # evaluate.py 依赖的 Python 内部模块
│   ├── tools/                  # 开发者/管理员工具：打包、模拟 runner、渲染摘要
│   ├── submission_examples/    # 三份参考提交样例
│   │   ├── aclnn_launch_example/         # aclnn 风格
│   │   ├── direct_launch_example/        # torch.library 直通 AscendC
│   │   └── direct_launch_simple_example/ # pybind 直通 AscendC
│   └── result_examples/        # 上述提交在 NPU 上的样例输出
├── docs/                       # 设计与评测相关文档
├── requirements.txt            # Python 依赖
└── README.md
```

## 快速开始

### 环境要求

- Python 3.8+
- PyTorch 2.3+，`torch_npu` 与对应 CANN（推荐 8.5.0）
- NumPy 1.21+

### 安装依赖

```bash
pip install -r requirements.txt
```

### 提交一个算子进行评测

本仓库的标准化评测流水线位于 `evaluation/`。详细说明见 [evaluation/README.md](evaluation/README.md)。三份参考提交放在 `evaluation/submission_examples/` 下，任选其一为基础复制并替换 kernel 代码即可：

| 示例 | 风格 | 入口 |
| --- | --- | --- |
| `aclnn_launch_example` | 走 ACLNN 框架，算子注册为 opapi | `torch.ops.cann_bench.<op>` |
| `direct_launch_example` | 通过 `torch.library` 绑定，AscendC kernel 直接由 plugin 启动 | `torch.ops.cann_bench.<op>` |
| `direct_launch_simple_example` | pybind11 + 简化 tiling，最薄的一层 | `cann_bench.<op>` |

本地构建并在 NPU 上模拟一次 job：

```bash
# 1. 构建某个示例提交的 wheel
bash evaluation/submission_examples/direct_launch_simple_example/build.sh

# 2. 在本地 NPU 上跑完 prepare→compile→correctness→performance 四个阶段
bash evaluation/tools/simulate_runner.sh \
    evaluation/submission_examples/direct_launch_simple_example/dist/*.whl \
    /path/to/benchmark_bundle \
    my_run
```

评测结束后，`evaluation/result_examples/my_run/summary.md` 汇总各算子通过率、加速比与每条 case 的 baseline / custom 耗时。

## 测试用例结构

每个算子目录下包含以下文件：

| 文件 | 说明 |
|------|------|
| `cases.yaml` | 测试用例配置，定义输入参数、期望精度阈值以及 `baseline_perf_us` |
| `cases.csv` | 测试用例的 CSV 形式（可选） |
| `golden.py` | PyTorch 参考实现，用于精度比对 |
| `proto.yaml` | 算子原型与 schema |
| `desc.md` | 算子详细说明 |

## 添加新算子

1. 在 `kernel_bench/level{N}/<op_name>/` 目录下创建算子文件夹
2. 创建 `cases.yaml` / `cases.csv` 定义测试用例
3. 创建 `golden.py`，用 PyTorch 表达参考计算
4. 创建 `proto.yaml` 声明算子 schema
5. 创建 `desc.md` 写算子说明
6. （可选）通过 `evaluation/tools/register_benchmark.py` 打包成 bundle：

```bash
python3 evaluation/tools/register_benchmark.py kernel_bench/level1/<op>
```

## 许可证

见 [LICENSE](LICENSE)。
