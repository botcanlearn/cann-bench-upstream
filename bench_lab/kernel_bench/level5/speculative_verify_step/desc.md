# SpeculativeVerifyStep 算子 API 描述

## 1. 算子简介

投机解码（speculative decoding）验证步。draft（草稿）模型已自回归产出 $T$ 个候选 token，target（目标）模型对这 $T$ 个位置及其后一个位置并行给出词表分布。本算子执行标准拒绝采样验证（Leviathan & Matias 2023 / Chen et al. 2023）：顺序判定每个候选 token 是否被接受，在首个拒绝位置从残差分布重采样一个修正 token；若全部接受则从 target 的第 $T+1$ 个分布采样一个 bonus token。两个输出（接受长度与最终 token 序列）均为 int32，**零容差精确比对**。

采样所需的随机数（`accept_noise` / `resample_noise`）由框架作为输入张量传入，算子本身是**确定性函数**——这是所有生产级投机解码验证 kernel（vLLM `rejection_sampler`、EAGLE/Medusa 的 verify 阶段、TensorRT-LLM）为保证可复现所采用的标准做法。

**主要应用场景**：
- LLM 推理加速：投机解码（draft-then-verify）每步生成的核心算子，其时延直接决定 decode 吞吐
- EAGLE / Medusa / Lookahead 等多头/树形草稿方案退化为链式验证时的 verify kernel
- 大批量 serving 场景（B 路请求并行验证）

**算子特征**：
- 难度等级：L5（FusedComposite）
- 五输入（1 个 int32 token 序列 + 4 个 float32 权重/噪声）双输出（均 int32）
- 数据依赖控制流：接受判定顺序执行、首个拒绝位置决定后续读哪一行做重采样
- 融合归一化、逐 token gather 判定、残差构造与逆 CDF 采样

**为何是 L5**（约为 L4 FusedComposite 的 2 倍难度）：
- **正确 ≠ 快**：按 §2 逐 batch、逐 token、逐元素直译即可得到完全正确的结果并通过全部用例，但那样对 $[B, T{+}1, V]$ / $[B, T, V]$ 两块数据要做归一化、gather、残差、前缀和等多趟独立遍历（朴素实现 ~5 遍 HBM 往返），性能只能锚定朴素 baseline（预期得分 ~0.5 档）。要逼近带宽下界，必须把整个计算组织成对权重张量的**一次流式扫描**：归一化分母、被验证 token 的 gather、残差前缀和与逆 CDF 阈值判定在同一遍数据通过中完成，且拒绝位置产生的行选择（数据依赖控制流）要在 device 侧解决，不能回读 host
- **整数输出零容差**：accept_len / output_tokens 逐元素精确比对，一处不等即失败。§2 把判定链的每一步都定义为单次 IEEE 精确舍入运算或语义固定的前缀和，实现必须逐条遵守（例如把"乘法比较"写成除法、把归一化分母换成低精度 fp32 顺序累加，都可能在隐藏用例上翻转判定）
- **实现规格而非过可见测试**：隐藏评测集覆盖公开用例之外的规格维度——$T{=}1$、必全接受 / 高拒绝率值域、残差整体极小、常数权重（归一化后两分布相等，考内部归一化是否存在）、素数词表、$V{=}2$ 极小词表、noise 贴边等，验证/重采样两条路径的每处约定都会被单独检验
- **独特轴**：数据依赖控制流 + 拒绝采样语义 + 整数零容差的组合在本评测集内独一无二，不存在可套用的现成注意力/矩阵乘模板

## 2. 算子定义

### 数值语义约定（先于公式，实现必须遵守）

- 工作精度为 fp32。**沿 V 的求和与前缀和**统一定义为：按索引升序、以不低于 fp64 的精度累加，每个前缀正确舍入回 fp32（这正是 torch CPU `cumsum` 对 fp32 输入的语义）。在评测值域下该结果与"fp32 加数的精确和的正确舍入"一致，因此**对归约顺序不敏感**：树形/分块并行归约只要以 ≥ fp64 精度（或等效的补偿求和）累加，即可逐位复现
- 除法、乘法、减法、`max` 均为单次 IEEE-754 fp32 精确舍入运算
- 比较为精确比较，无容差

### 数学定义（对每个 batch $b$ 独立）

记 $V$ 为词表大小（由 `p_target` 末维决定），$w_t = \text{p\_target}[b] \in \mathbb{R}^{(T+1) \times V}$，$w_d = \text{p\_draft}[b] \in \mathbb{R}^{T \times V}$，均为**非负未归一权重**（算子内部归一化，任意满足值域的随机输入都是合法分布）。

**第 1 步——归一化**：

$$
S_t[i] = \sum_{v=0}^{V-1} w_t[i, v], \qquad
P_t[i] = w_t[i] / S_t[i] \quad (i = 0, \dots, T)
$$

$$
S_d[i] = \sum_{v=0}^{V-1} w_d[i, v], \qquad
P_d[i] = w_d[i] / S_d[i] \quad (i = 0, \dots, T-1)
$$

求和按上文约定（升序、≥ fp64 累加、舍回 fp32，取前缀和末元素）；除法为逐元素单次 fp32 除法。评测值域保证 $w > 0$，无除零。

**第 2 步——顺序验证**（$i = 0, \dots, T-1$）：令 $tok_i = \text{draft\_tokens}[b, i]$，

$$
\text{接受} \iff \text{accept\_noise}[b, i] \cdot P_d[i][tok_i] < P_t[i][tok_i]
$$

判定写成**单次乘法 + 精确比较**（不写成 $u < P_t/P_d$ 的除法形式：避免除法、单次舍入跨实现逐位确定）。接受则继续 $i{+}1$；拒绝则停止，记拒绝位置为 $i^\*$。该式即标准拒绝采样 $u < \min(1, P_t/P_d)$：当 $P_t \ge P_d$ 时 $u \cdot P_d < P_t$ 对任意 $u < 1$ 成立。

**第 3 步——拒绝时的残差重采样**（若存在拒绝位置 $i^\*$）：

$$
r = \max(P_t[i^\*] - P_d[i^\*],\ 0) \quad (\text{逐元素 fp32 减法与 max})
$$

若 $r$ **恰为全零**，回退取 $r = P_t[i^\*]$。（完备性定义：两个归一化分布若满足 $P_t \le P_d$ 逐点成立且和均为 1，则必逐点相等，而逐点相等时第 2 步必接受，故该分支在精确算术下不可达，仅在极端浮点舍入下可能触达；定义之以保证任意输入下行为完备。评测随机输入不会触发。）

**第 4 步——全接受时的 bonus 采样**：$r = P_t[T]$（target 对第 $T+1$ 个位置的分布）。

**逆 CDF 采样**（第 3/4 步共用，使用同一个 $\text{resample\_noise}[b]$）：

$$
c[j] = \text{前缀和}(r)[j] \ (\text{语义同上}), \qquad total = c[V-1]
$$

$$
sel = \min\{\, j : c[j] > \text{resample\_noise}[b] \cdot total \,\}
$$

阈值为**单次 fp32 乘法**（不把 $r$ 归一化再比较：避免除法）。因 $\text{resample\_noise} \le 0.9999 < 1$ 且 fp32 乘法舍入误差相对量级 $2^{-24}$，恒有 $\text{resample\_noise} \cdot total < total = c[V-1]$，故 $sel$ 必存在且 $sel \le V-1$。$c$ 由非负数累加而来单调不减，"最小的 $j$"即二分/线性扫描的第一个越阈位置。

**第 5 步——输出组装**：记接受个数 $n_b$（$0 \le n_b \le T$；全接受时 $n_b = T$）：

- $\text{accept\_len}[b] = n_b$
- $\text{output\_tokens}[b, 0..n_b-1] = \text{draft\_tokens}[b, 0..n_b-1]$（接受前缀）
- $\text{output\_tokens}[b, n_b] = sel$（拒绝时为修正 token，全接受时为 bonus token）
- $\text{output\_tokens}[b, n_b+1..T] = -1$（填充）

每个 batch 恰好输出 $n_b + 1$ 个有效 token——这是投机解码"每步至少产出一个 token"的标准保证。

## 3. 接口规范

### 算子原型

```python
speculative_verify_step(Tensor draft_tokens, Tensor p_target, Tensor p_draft, Tensor accept_noise, Tensor resample_noise) -> (Tensor accept_len, Tensor output_tokens)
```

### 输入参数说明

| 参数 | 类型 | 必需 | dtype | shape | 描述 |
|------|------|------|-------|-------|------|
| draft_tokens | Tensor | 是 | int32 | [B, T] | draft 模型产出的候选 token，取值 [0, V−1] |
| p_target | Tensor | 是 | float32 | [B, T+1, V] | target 模型非负未归一权重；行 0..T−1 验证用，行 T 为 bonus 分布；评测取值 [0.0001, 1.0] |
| p_draft | Tensor | 是 | float32 | [B, T, V] | draft 模型非负未归一权重；评测取值 [0.0001, 1.0] |
| accept_noise | Tensor | 是 | float32 | [B, T] | 接受判定随机数 ∈ (0, 1)（评测取值 [0.0001, 0.9999]） |
| resample_noise | Tensor | 是 | float32 | [B] | 重采样/bonus 采样随机数 ∈ (0, 1)（评测取值 [0.0001, 0.9999]） |

### 输出

| 名称 | dtype | shape | 描述 |
|------|-------|-------|------|
| accept_len | int32 | [B] | 接受的 draft token 个数（0..T），零容差精确比对 |
| output_tokens | int32 | [B, T+1] | 接受前缀 + 1 个修正/bonus token，其余 −1，零容差精确比对 |

### 数据类型

| draft_tokens | p_target / p_draft / accept_noise / resample_noise | 输出 | 内部计算 |
|--------------|------------------------------------------------------|------|----------|
| int32 | float32 | int32 | fp32 判定路径 + ≥ fp64 精度的沿 V 累加（见 §2 数值语义约定） |

### 规则与约束

- $V \ge 2$、$1 \le T \le 16$、$1 \le B \le 32$；`draft_tokens` 取值必须在 $[0, V-1]$ 内（由 cases 的 value_range 保证，逐 case 上界为该 case 的 $V-1$）
- `p_target` / `p_draft` 为非负未归一权重且每行至少一个正元素（评测值域 [0.0001, 1.0] 保证严格为正）；算子内部归一化，不要求输入已归一
- `accept_noise` / `resample_noise` ∈ (0, 1)；算子不对违反上述取值约定的输入负责
- 沿 V 的求和/前缀和必须实现 §2 的舍入语义（≥ fp64 累加、逐前缀舍回 fp32）；接受判定与逆 CDF 阈值必须写成乘法比较形式
- 输出须为 contiguous 张量；`output_tokens` 填充值恒为 −1
- 显存约束：单 case 输入张量合计 $B(2T+1)V \cdot 4$ 字节 ≤ 2 GB（全部用例逐 case 校验通过，最大 case 约 642 MB）

### 支持范围

| 维度 / 参数 | 范围 | 备注 |
|---|---|---|
| `B`（batch） | 1 ~ 32 | cases.csv 实测 2 ~ 32 |
| `T`（草稿长度） | 1 ~ 16 | cases.csv 实测 4 / 8 / 16（隐藏用例含 T=1 及素数 T） |
| `V`（词表） | 2 ~ 152064 | cases.csv 实测 32000（LLaMA2）/ 128256（LLaMA3）/ 152064（Qwen2.5）；隐藏用例含素数 V、V=2 等 |
| `p_target` / `p_draft` 取值 | [0.0001, 1.0] | 非负未归一权重（隐藏用例含常数值域、窄值域、贴边值域） |
| `accept_noise` / `resample_noise` 取值 | [0.0001, 0.9999] | 恒在 (0, 1) 内（隐藏用例含贴边值域） |

## 4. 精度要求

两个输出均为 **int32 结构输出，阈值 0**：逐元素完全相等，一处不等即失败（评测框架 int 类型默认即零容差，本算子显式声明）。

**零容差的可行性依据**（设计 + 实测，全部 100 个 case 配置 × 多种子，共 42630 个接受判定行、4935 次逆 CDF 采样）：

1. 判定链中每一步都是精确定义的运算：归一化分母与逆 CDF 前缀和采用"升序 ≥ fp64 累加、逐前缀舍回 fp32"的固定语义（等价于 fp32 加数精确和的正确舍入，对归约顺序不敏感）；接受判定与采样阈值均为单次 fp32 乘法 + 精确比较；除法为单次 fp32 除法。遵守 §2 语义的任何实现与 golden 逐位一致
2. 接受判定对精度提升不敏感：实测 fp32 判定路径 vs fp64 判定路径在全部 42630 个判定行上**零翻转**，最小相对裕度 3.6e-5，比双路径舍入差（~$2^{-24}$）高两个数量级以上——评测 value_range 使接受判定边界事件测度为零
3. 逆 CDF 的索引选择桶宽仅 ~$1/V$，对判定精度**提升**是敏感的（实测把判定路径提升到 fp64 时 505 次运行出现 5 次索引偏移，符合 ~$V \cdot 2^{-24}$ 的边界命中概率量级）。因此**fp32 舍入粒度是算子语义的一部分**（§2 数值语义约定），oracle 与 plain golden 复用同一 fp32 判定核心；实现只要逐条遵守 §2 语义即与参考逐位一致，与评测精度模式无关
4. 陷阱用例的确定性由构造保证：如"必全接受"用例（p_target ∈ [0.9, 1.0]、p_draft ∈ [0.0001, 0.01]、accept_noise ≤ 0.4）在值域层面保证判定裕度 ≥ 10% 相对量级，实测多种子 accept_len ≡ T

**评测方式**：golden_precision=fp64_cpu 下 evaluator 喂给 oracle 的 fp64 输入由 fp32 无损升精度而来，oracle 核心内舍回 fp32 即逐位还原原始输入，故参考输出与 plain golden 逐位一致（实测零差异）；kernel 输出与参考逐元素比对，int32 阈值 0，无 MERE/MARE 概念。

## 5. 标准 Golden 代码

```python
import torch
from typing import Tuple


def _speculative_verify_step_core(draft_tokens, p_target, p_draft, accept_noise,
                                  resample_noise, compute_dtype):
    """核心计算：以 compute_dtype 精度执行归一化 / 接受判定 / 逆 CDF 重采样。"""
    B, T = draft_tokens.shape
    V = p_target.shape[-1]
    device = draft_tokens.device

    w_t = p_target.to(compute_dtype)          # [B, T+1, V] 非负未归一权重
    w_d = p_draft.to(compute_dtype)           # [B, T, V]
    noise_a = accept_noise.to(compute_dtype)  # [B, T]
    noise_r = resample_noise.to(compute_dtype)  # [B]

    accept_len = torch.empty(B, dtype=torch.int32, device=device)
    output_tokens = torch.full((B, T + 1), -1, dtype=torch.int32, device=device)
    row_idx = torch.arange(T, device=device)

    for b in range(B):
        # === 1. 归一化（分母 = 沿 V 的前缀和末元素；cumsum 以 ≥ fp64 精度累加后舍回工作精度）===
        s_t = torch.cumsum(w_t[b], dim=-1)[:, -1:]   # [T+1, 1]
        s_d = torch.cumsum(w_d[b], dim=-1)[:, -1:]   # [T, 1]
        p_t = w_t[b] / s_t                           # [T+1, V] 归一化目标分布
        p_d = w_d[b] / s_d                           # [T, V]   归一化 draft 分布

        # === 2. 顺序验证：accept_noise * P_d[i][tok] < P_t[i][tok]（单次乘法 + 比较）===
        toks = draft_tokens[b].long()                # [T]
        pt_tok = p_t[row_idx, toks]                  # [T] P_t[i][tok_i]（i = 0..T-1）
        pd_tok = p_d[row_idx, toks]                  # [T]
        accept = noise_a[b] * pd_tok < pt_tok        # [T] bool
        rejected = torch.nonzero(~accept)
        n_acc = int(rejected[0].item()) if rejected.numel() > 0 else T

        # === 3./4. 拒绝 → 残差分布逆 CDF 重采样；全接受 → bonus 从 P_t[T] 采样 ===
        if n_acc < T:
            row = torch.clamp(p_t[n_acc] - p_d[n_acc], min=0)   # 残差 r = max(P_t - P_d, 0)
            cum = torch.cumsum(row, dim=-1)                     # 前缀和（≥ fp64 累加，舍回工作精度）
            total = cum[-1]
            if total == 0:                                      # 残差恰为全零 → 回退 r = P_t[i]
                row = p_t[n_acc]
                cum = torch.cumsum(row, dim=-1)
                total = cum[-1]
        else:
            row = p_t[T]                                        # bonus token 分布
            cum = torch.cumsum(row, dim=-1)
            total = cum[-1]
        # 逆 CDF：最小的 j 使 cum[j] > resample_noise * total（乘法形式，无除法）。
        # resample_noise ≤ 0.9999 保证阈值 < total = cum[V-1]，j 必存在。
        thr = noise_r[b] * total
        sel = int(torch.searchsorted(cum, thr, right=True).item())

        # === 5. 输出组装 ===
        accept_len[b] = n_acc
        output_tokens[b, :n_acc] = draft_tokens[b, :n_acc]
        output_tokens[b, n_acc] = sel
    return accept_len, output_tokens


def speculative_verify_step(
    draft_tokens: torch.Tensor,
    p_target: torch.Tensor,
    p_draft: torch.Tensor,
    accept_noise: torch.Tensor,
    resample_noise: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    投机解码验证步 golden reference（plain golden = bench：判定路径固定 fp32）

    Args:
        draft_tokens: [B, T] int32，draft 模型产出的候选 token，取值 [0, V-1]
        p_target: [B, T+1, V] float32，target 模型的非负未归一权重（算子内部沿 V 归一化），
                  行 0..T-1 用于验证，行 T 用于全接受时的 bonus 采样
        p_draft: [B, T, V] float32，draft 模型的非负未归一权重（算子内部沿 V 归一化）
        accept_noise: [B, T] float32，接受判定随机数，取值 (0, 1)（评测范围 [0.0001, 0.9999]）
        resample_noise: [B] float32，重采样/bonus 采样随机数，取值 (0, 1)（同上）

    Returns:
        accept_len: [B] int32，接受的 draft token 个数（0..T）
        output_tokens: [B, T+1] int32，前 accept_len 个接受 token + 1 个修正/bonus token，
                       其余位置填 -1
    """
    return _speculative_verify_step_core(
        draft_tokens, p_target, p_draft, accept_noise, resample_noise, torch.float32)


def speculative_verify_step_oracle(
    draft_tokens: torch.Tensor,
    p_target: torch.Tensor,
    p_draft: torch.Tensor,
    accept_noise: torch.Tensor,
    resample_noise: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Oracle (g)：int32 结构输出，算子语义在 fp32 判定粒度下定义（fp32 舍入是规格的一部分，
    见 desc.md §2/§4），oracle 复用同一 fp32 判定核心。fp64_cpu 下 evaluator 喂入的 fp64
    输入由 fp32 无损升精度而来，核心内舍回 fp32 即逐位还原原始输入，oracle ≡ plain。
    （若按输入 dtype 提升判定精度，逆 CDF 桶边界（宽 ~1/V）会以 ~1% 概率偏离规格定义的
    索引，破坏 int 零容差评测，见 desc.md §4 实测。）"""
    return _speculative_verify_step_core(
        draft_tokens, p_target, p_draft, accept_noise, resample_noise, torch.float32)
```

## 6. 额外信息

### 算子调用示例

```python
import torch

B, T, V = 8, 4, 32000

draft_tokens = torch.randint(0, V, (B, T), dtype=torch.int32, device="npu")
p_target = torch.empty(B, T + 1, V, device="npu").uniform_(0.0001, 1.0)
p_draft = torch.empty(B, T, V, device="npu").uniform_(0.0001, 1.0)
accept_noise = torch.empty(B, T, device="npu").uniform_(0.0001, 0.9999)
resample_noise = torch.empty(B, device="npu").uniform_(0.0001, 0.9999)

accept_len, output_tokens = speculative_verify_step(
    draft_tokens, p_target, p_draft, accept_noise, resample_noise)
# accept_len.shape: [B]; output_tokens.shape: [B, T+1]
```

### 与标准投机解码的对应关系

- 接受判定 $u \cdot P_d[tok] < P_t[tok]$ 等价于经典形式 $u < \min(1, P_t[tok]/P_d[tok])$（$u \in (0,1)$）
- 残差分布 $\mathrm{norm}(\max(P_t - P_d, 0))$ 即 Leviathan & Matias (2023) 定理 1 的修正分布；按逆 CDF 用 $\text{resample\_noise} \cdot total$ 采样等价于从归一化残差分布采样
- 全接受时从 $P_t[T]$ 采样 bonus token 即"验证 $T$ 个、至多产出 $T+1$ 个"的标准做法
- 该验证规则保证输出 token 序列的联合分布与 target 模型自回归采样完全一致（无损加速）

### 性质（可用于实现自检）

- `p_target` 与 `p_draft` 为同一张量（或归一化后逐位相等，如两侧均为常数权重）时，任意 `accept_noise` ∈ (0,1) 下必全接受，`accept_len ≡ T`
- `resample_noise` → 0⁺ 时 `sel` 趋向 CDF 头部（第一个 $r > 0$ 的索引）；→ 1⁻ 时趋向尾部
- 大 B 统计下，接受率随 $P_t[tok]/P_d[tok]$ 比值单调上升

### 参考文献

- Leviathan, Y., Kalman, M., Matias, Y. (2023). "Fast Inference from Transformers via Speculative Decoding". ICML 2023, arXiv:2211.17192（拒绝采样验证与残差分布的来源）
- Chen, C., Borgeaud, S., Irving, G., Lespiau, J.-B., Sifre, L., Jumper, J. (2023). "Accelerating Large Language Model Decoding with Speculative Sampling". arXiv:2302.01318
- Li, Y. et al. (2024). "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty". ICML 2024, arXiv:2401.15077（verify kernel 工程形态的现实原型）
