# StanfordBench AI 算子测试示例

本目录包含 StanfordBench 格式的 AI 算子示例，演示如何使用 Triton 编写算子并进行评测。

## 目录结构

```
examples/stanfordbench/
├── ReLU/
│   ├── ai_op.py        # AI 算子实现（Triton）
│   └── README.md       # 本文档
└── ...                  # 其他算子示例
```

## 测试方式

```bash
# 单卡测试
./scripts/run_evaluation.sh --bench-name stanford --task-dir bench_lab/stanford_bench/KernelBench/KernelBench --operator ReLU --device-id 0 --source-dir examples/stanfordbench_example/ReLU

# 多卡并行测试
./scripts/run_evaluation.sh --bench-name stanford --task-dir bench_lab/stanford_bench/KernelBench/KernelBench --operator ReLU --source-dir examples/stanfordbench_example/ReLU
```

## AI 算子文件要求

AI 算子文件 (`ai_op.py`) 需要满足以下要求：

1. **包含 `Model` 类**: 继承 `torch.nn.Module`，实现 `forward` 方法
2. **匹配 StanfordBench 接口**: forward 的输入输出与 StanfordBench golden 的签名一致
3. **支持 NPU**: 使用 Triton 或 torch_npu 实现，确保能在 NPU 上执行

## 评测流程

```
┌─────────────────────────────────────────────────────────────┐
│  1. 加载 StanfordBench Golden (bench_lab/stanford_bench/KernelBench)      │
│     - get_inputs(): 生成测试输入                             │
│     - Model.forward(): PyTorch 参考实现                      │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│  2. 执行 Golden (NPU)                                        │
│     - 输入 tensors → NPU                                     │
│     - Golden.forward() → golden_output                       │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│  3. 执行 AI 算子 (NPU)                                       │
│     - 加载 ai_op.py                                          │
│     - AI Model.forward() → ai_output                         │
│     - Profiler 采集性能                                      │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│  4. 精度对比                                                 │
│     - ai_output vs golden_output                            │
│     - atol/rtol=0.01 (StanfordBench 默认)                    │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│  5. 性能对比                                                 │
│     - AI 算子执行时间                                        │
│     - Golden 执行时间                                        │
│     - 加速比 = Golden时间 / AI时间                           │
└─────────────────────────────────────────────────────────────┘
```

## 输出结果

评测完成后会生成：

- `reports/eval_*.json`: JSON 格式详细报告
- `reports/eval_*.md`: Markdown 格式报告
- `reports/prof_data/`: Profiler 性能数据

## 示例输出

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