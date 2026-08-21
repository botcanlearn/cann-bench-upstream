---
name: op-difficulty-classify
description: 判定算子开发难度等级（L1-L4）。输入算子 task 文件夹（含 proto.yaml、golden.py、cases.yaml），输出该算子的 difficulty level。触发词：算子难度判定、difficulty level、算子分级。
---

# 算子难度等级判定

根据 cann-bench/tasks/ 的分级标准，判定一个算子 task 属于 L1/L2/L3/L4 中的哪个等级。

---

## 1. 判定流程

```
读取 task 目录下的 proto.yaml + golden.py
  ↓
提取特征（计算模式、输入数量、golden 行数、有无 matmul/conv、有无循环等）
  ↓
对照分级标准逐项匹配
  ↓
输出 difficulty level
```

---

## 2. 分级标准

### Level 1 — 基础元素运算

| 判定条件 | 说明 |
|----------|------|
| 计算模式 | 逐元素映射，O(N) 无跨元素依赖 |
| golden.py | ≤65 行，1~5 步操作，通常直接调单个 PyTorch API |
| 输入 tensor | 1~3 个，shape 相同（无异构 shape） |
| 输出 | 单 tensor（或 TensorList 但每个元素独立） |
| matmul/conv | 无 |
| 显式循环 | 无（列表推导遍历 TensorList 不算） |
| reduction | 无或仅全局范数 |
| attrs | 0~3 个简单标量 |
| 典型 category | Elementwise, MaskPredicate |
| 代表算子 | sigmoid, exp, gelu, mish, swi_glu |

### Level 2 — 中等复杂度

| 判定条件 | 说明 |
|----------|------|
| 计算模式 | 有 reduction / 索引操作 / 归一化，跨元素通信但模式固定 |
| golden.py | 31~159 行 |
| 输入 tensor | 1~4 个，shape 常异构（需 broadcast / unsqueeze 对齐） |
| 输出 | 1~2 tensor |
| matmul/conv | 无 |
| 条件分支 | 多数算子有（dtype 分支、mode 分支） |
| reduction | 有（softmax/norm/mean/argmax/cummin 等） |
| 整数 dtype | 开始出现 int8/int32/int64 |
| attrs | 0~7 个，含 bool/string/ListInt |
| 典型 category | Normalization, FusedComposite, ScatterUpdate, IndexGather, Broadcast |
| 代表算子 | softmax, rms_norm, group_norm, gather, scatter, apply_rotary_pos_emb |

### Level 3 — 计算密集 / 复杂融合

| 判定条件 | 说明 |
|----------|------|
| 计算模式 | 有 matmul/conv（计算密集型），或多步复杂融合，或动态输出 shape |
| golden.py | 39~313 行 |
| 输入 tensor | 1~8 个（含 optional） |
| 输出 | 1~4 tensor（43% 多输出） |
| matmul/conv | 有 |
| 显式循环 | 部分有（分组 matmul、迭代归一化、贪心选择） |
| 混合精度 | 普遍（int8+fp16 同时出现） |
| attrs | 0~9 个 |
| 动态输出 | 有（NMS keep 数不定、Unique 去重后长度不定） |
| 典型 category | Contraction, LayoutTransform, VVFusion, SortSelect |
| 代表算子 | conv_2d, grouped_matmul, nms, roi_align, moe_finalize_routing |

### Level 4 — 端到端模型子图

| 判定条件 | 说明 |
|----------|------|
| 计算模式 | 完整模型子图，多次 matmul + 序列依赖 |
| golden.py | 72~169 行 |
| 输入 tensor | 3~9 个 |
| 输出 | 1~3 tensor |
| matmul | 全部有，且多次（attention pattern: Q@K^T → softmax → @V） |
| 序列依赖 | RNN 类有时序循环 |
| category | 全部为 FusedComposite |
| 特征 | attention / RNN / 多阶段 matmul+量化流水线 |
| 代表算子 | mha, gqa, mla, lstm, gru, sparse_flash_attention |

---

## 3. 判定规则（按优先级）

按以下顺序判断，**命中第一个即停止**：

1. **golden.py 中有多次 matmul 且构成 attention/RNN pattern** → **L4**
2. **golden.py 中有 matmul 或 conv 调用** → **L3**
3. **有 reduction 操作（mean/sum/softmax/norm/argmax）或 scatter/gather 索引操作或异构 shape 输入需要 broadcast** → **L2**
4. **其余（逐元素映射、简单激活函数）** → **L1**

**辅助判断**（用于边界 case）：
- 输入 tensor ≥5 个 → 至少 L3
- golden.py ≥150 行 → 至少 L3
- 有显式 for/while 循环（非列表推导） → 至少 L3
- 输出为动态 shape → 至少 L3
- category 为 Contraction → L3
- category 为 FusedComposite 且含 matmul → L3 或 L4

---

## 4. 使用方式

给定 task 目录路径，执行以下检查：

```
1. 读取 proto.yaml → 获取 category、inputs/outputs 数量、attrs
2. 读取 golden.py → 统计行数、检查是否有 matmul/conv/reduction/循环
3. 读取 cases.yaml → 检查 dtype 覆盖、input_shape 维度
4. 对照上述标准输出 level
```

---

## 5. 参考资料

- [references/difficulty-standard.html](references/difficulty-standard.html) — 完整分级标准网页（含 53 个算子的详细分类数据）
- 数据来源：https://gitcode.com/cann/cann-bench 仓库 `tasks/` 目录（level1~level4 共 53 个算子），经 4 轮 subagent 交叉核实
