# HAP `t_hw` 计算逻辑（公开版）

本目录公开 cann-bench 中 **`t_hw`（Hardware-Anchored Performance，硬件锚定理论下限耗时）** 的计算方法：共享模型库 `hap_lib.py`，加上 **8 个代表性算子**各一个 `compute_<op>.py`。目的是让读者**逐行读懂 `t_hw` 是如何分解与计算的**——8 个算子合起来覆盖模型的全部机制（四个瓶颈分量、L2 分段写、flat 归约、SFU/除法/cast 规则、HF32 路径、因果三角计数、纯数据搬运）。

## 模型

```
t_hw = max(t_cube, t_vector, t_hbm_read, t_write)   [微秒]
```

`t_hw` 是 roofline 理论**下界**：所有 engine 满峰值、完美 overlap、无 launch 开销，物理上不可达，用作性能地板。

- **读**：`input_bytes / HBM_BW`（kernel 启动时 L2 为空，输入全部来自 HBM；只计 unique bytes，无 reload 因子）
- **写**：L2-fill-first 分段（先按 L2 总线带宽填满 L2 容量，溢出部分走 HBM；容量判定只看 output bytes）
- **cube**：`FLOPs / cube_peak_for_dtype`（matmul 记 `2*M*N*K`）
- **vector**：`元素操作数 / vector_rate(op_kind, dtype)`；**reduction 按 flat 计**（`N/peak`，**无 log2**——log2 是树深/延迟，不是吞吐代价）；**scan/prefix 才有 log2**（`N*ceil(log2(N_scan))`）
- cast 是真实 vector 分量；FixP 写回旁路（dtype 转换/scale/bias）免费
- cube 与 vector 并行（取 max）；同一 engine 内的分量串行（求和）
- 收尾约定：`t_hw_us = max(raw, 1.0)`（1µs clip）；`baseline_cap_us = max(10*t_hw_us, 10.0)` 是发布基线的**封顶值**——发布的 `baseline_perf_us = min(实测, baseline_cap_us)`（见 `tasks/metadata/910b2.json` 的 `_metadata` 块），仅在无实测处直接取封顶值。脚本输出的是封顶值，单凭 `t_hw` 不能复现实测基线。

## 代表算子

| 算子 | level | 展示的机制 |
|---|---|---|
| swi_glu | level1 | 逐元素 + SFU(sigmoid) + fp16/bf16 cast 规则；输入两半读、输出减半写 |
| apply_adam_w | level2 | 多张量读（4 输入）、fp32 计算 + 5 cast、read-bound |
| arg_max | level2 | flat 归约（无 log2）、int64 索引小输出写 |
| softmax | level2 | fp32 计算链（max-reduce/sub/exp/sum-reduce/div）+ 双 cast，vector-bound |
| strided_slice | level3 | 纯数据搬运：fetched-bytes 规则、L2-fill-first 写、无计算算子的瓶颈标注 |
| conv_2d | level3 | cube FLOPs 计数、fp32 卷积的 HF32/TF32 路径（`cube_hf32`）、FixP 免费 bias |
| mha | level4 | 双 matmul cube + fp32 softmax vector、因果三角精确计数 |
| mla | level4 | 分裂 QK 头维的双 matmul、GQA/共享 latent KV 的 unique-bytes 读、显式 cast |

每个 `compute_<op>.py` 的 docstring 给出该算子的完整分量分解（读/写字节数、cube FLOPs、vector 逐操作计数及依据）。

## 分解的依据与校验

每个脚本内置**开源用例复算**：对仓库开源 1–20 用例计算 `t_hw`，与 `tasks/metadata/910b2.json` 已发布值对比（门限：每条 ≤2% 相对误差或 ≤0.05µs 绝对误差），直接运行即可复现。

分解中的计数分两类，脚本里明确区分：**由算子语义直接推导**的（绝大多数：字节数、FLOPs、逐操作 vector 计数），开源复算对它们是独立复核；以及**少数按开源用例标定的经验约定**（如 mha 的 softmax 不计 cast 而 mla 计 cast），这类在脚本中以 "empirical convention validated on the open cases" 字样标出，开源复算对它们是一致性检查而非独立验证。

## 硬件常量不随本目录发布

`t_hw` 的数值依赖各平台的硬件常量表（HBM 读带宽、L2 总线带宽与容量、各 dtype 的 cube 峰值、vector 速率、SFU/除法折减系数）。**本公开版不包含这些数值**：`hap_lib.py` 从外部文件 `hap_platform_constants.json`（不随发布提供）加载常量表，加载时校验完整性并对缺项给出明确报错。硬件规格请查阅官方文档：

> https://www.hiascend.com/document

## 运行方式（三种状态）

1. **不提供常量文件**（默认）：所有脚本可正常导入、可读，首次计算时给出指向上述文档链接的明确报错。
2. **提供常量文件**：按 `hap_platform_constants.template.json` 填写三个平台块（`vec_basic` 必须含 `float32` 回退键），置于本目录。此时每个脚本可直接运行开源 1–20 用例复算（读取仓库 `tasks/`，以本目录上两级为仓库根）。
3. **闭源用例**：`inner/tasks/` 下的闭源 21–100 用例不随仓库公开；脚本在开源复算后检测到其缺失会打印说明并正常退出。计算结果覆盖 910b2/950pr/910c 三平台（公开 metadata 现有 910b2 与 950pr；三平台用同一套分解，只换常量）。

## 目录

单一平铺目录，无子目录：

```
hap_thw_model/
├── README.md
├── hap_lib.py                             # 模型库：分量公式 + 汇总/收尾（常量外置）
├── hap_platform_constants.template.json   # 常量 schema 模板（数值自行填写）
└── compute_<op>.py × 8                    # 代表算子的分量分解（见上表）
```
