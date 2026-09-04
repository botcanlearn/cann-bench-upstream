# StanfordBench Example

StanfordBench（KernelBench）格式的 AI 算子示例集，演示如何用 Triton-Ascend 编写算子并接入 CANN Bench 评测。

## 目录结构

```
stanfordbench_example/
├── ReLU/                # ReLU 算子示例（Triton 实现）
│   ├── ai_op.py         # AI 算子实现（定义 ModelNew）
│   └── README.md        # 算子级说明（接口要求、评测流程详解）
└── README.md            # 本文档
```

每个算子一个子目录，子目录名即算子名；新增算子时按同样结构添加 `<算子名>/ai_op.py` 即可。

## 前置条件

1. 安装 cann-bench 核心依赖：
```bash
pip install -r requirements.txt
```

2. 安装 Triton-Ascend（示例算子使用 Triton 编写，wheel 由专用源提供）：
```bash
pip install -r requirements-triton.txt
```

3. 下载 StanfordBench 任务数据集（评测 golden 来源）：
```bash
bash bench_lab/stanford_bench/download.sh
```

4. 已配置 Ascend NPU 环境（torch_npu 可用，`source` CANN 环境变量）。

## 运行评测

`--source-dir` 指向**算子子目录**（包含 `ai_op.py` 的目录），而非本示例根目录：

```bash
# 单卡评测 ReLU 算子
./scripts/run_evaluation.sh \
  --bench-name stanford \
  --task-dir bench_lab/stanford_bench/KernelBench/KernelBench \
  --operator ReLU \
  --source-dir examples/stanfordbench_example/ReLU \
  --device-id 0

# 多卡并行评测
./scripts/run_evaluation.sh \
  --bench-name stanford \
  --task-dir bench_lab/stanford_bench/KernelBench/KernelBench \
  --operator ReLU \
  --source-dir examples/stanfordbench_example/ReLU
```

## ai_op.py 接口要求

- 定义 `ModelNew` 类（继承 `torch.nn.Module`），其 `__init__` 与 `forward` 签名必须与对应 StanfordBench 任务的 `Model` 一致；
- 算子在 NPU 上执行（Triton-Ascend 或 torch_npu）；
- 文件名固定为 `ai_op.py`，否则评测框架无法发现。

详见 `ReLU/README.md`。

## 预期输出

评测通过时输出类似：

```
[Process 0] [1/1] level1/19_ReLU_1: ✅ (3.08μs) MARE=0.000000, max_diff=0.000000

============================================================
评测结果摘要
============================================================
评测算子数: 1
总用例数: 1
通过用例数: 1
失败用例数: 0
通过率: 100.00%
平均加速比: 2.50x
============================================================
```

同时生成 `reports/eval_*.json`、`reports/eval_*.md` 报告及 `reports/prof_data/` Profiler 数据。
