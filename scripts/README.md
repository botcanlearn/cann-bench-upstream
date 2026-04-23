# Scripts

本目录包含运行测试和评测的脚本。

## 目录结构

```
scripts/
├── run_evaluation.sh      # Kernel Bench 评测脚本
├── run_test.sh           # 测试运行脚本
└── README.md             # 本文档
```

---

## run_evaluation.sh

Kernel Bench 评测脚本，支持多种操作模式。

### 用法

```bash
./scripts/run_evaluation.sh [选项]
```

### 命令行选项

| 选项 | 说明 |
|------|------|
| `-a, --action <action>` | 操作类型: `eval`(评测), `list`(列表), `info`(详情), `config`(配置)，默认: `eval` |
| `--source-dir <dir>` | AI生成的算子源码目录（自动扫描编译安装） |
| `-l, --level <level>` | 算子难度级别 (1/2/3/4) |
| `-o, --operator <name>` | 算子名称 (如 Exp, Softmax) |
| `-c, --case-id <id>` | 用例编号 |
| `-v, --verbose` | 详细输出 |
| `-h, --help` | 显示帮助信息 |

### 使用示例

#### 查看帮助

```bash
./scripts/run_evaluation.sh --help
```

#### 列出算子

```bash
# 列出 L1 级所有算子
./scripts/run_evaluation.sh -a list -l 1

# 列出包含指定名称的算子
./scripts/run_evaluation.sh -a list -o matmul
```

#### 查看算子详情

```bash
./scripts/run_evaluation.sh -a info -o Exp
```

#### 从源码目录评测

```bash
# 自动扫描、编译、安装 AI 生成的算子源码并评测
./scripts/run_evaluation.sh --source-dir /path/to/ai_ops
```

#### 执行评测

```bash
# 评测所有算子
./scripts/run_evaluation.sh

# 评测指定级别
./scripts/run_evaluation.sh -l 1

# 评测指定算子
./scripts/run_evaluation.sh -o Exp

# 评测单个用例
./scripts/run_evaluation.sh -l 1 -o Exp -c 1

# 详细输出
./scripts/run_evaluation.sh -l 1 -o Exp -v
```

#### 查看配置

```bash
./scripts/run_evaluation.sh -a config
```

### 输出

评测报告保存在 `reports/` 目录，格式为 Markdown。

---

## run_test.sh

统一的测试运行脚本，用于验证算子功能正确性。

```bash
# 运行所有测试
./scripts/run_test.sh

# 按级别筛选
./scripts/run_test.sh --level 1

# 按算子筛选
./scripts/run_test.sh --operator gelu

# 指定设备
./scripts/run_test.sh --cpu    # CPU 测试（默认）
./scripts/run_test.sh --npu    # NPU 测试

# 查看帮助
./scripts/run_test.sh --help
```

测试结果保存在 `tests/reports/` 目录，格式为 JSON。