# K3KdaDecoderLayerE2E 算子 API 描述

## 1. 算子简介

Kimi K3 的 KDA 型 decoder 层（93 层中的 69 层）由注意力子层与 LatentMoE 子层构成。
本任务把**整层端到端**融合为一个多 die 协同的 megakernel：从注意力分支的 AttnRes
混合起，到 MoE 残差写回载体止，覆盖两个子层与两次载体更新。两个子层的完整数学
在 §2 给出：注意力子层逐行对齐 `modeling_kimi_linear.py` 的 KimiDeltaAttention 与
fla 的 fused_recurrent_kda kernel；MoE 子层逐行对齐 KimiMoEGate、KimiSparseMoeBlock
与 SituAndMul。

**提交形态（硬约束）**：整层为**单次 kernel 提交**——host→device 一次下发、
device→host 一次返回，layer 完成即返回。层内全部计算与全部 die 间通信必须在这
一个 kernel 内由 device 侧发起并完成，host 不参与层内任何调度、同步或通信。
详见 §2「并行语义与正确性契约」。

**难度定级**：按仓库现行难度枚举（最高 L5）收录为 L5；**建议定级 L6**——至少四根
L5 难度轴耦合于同一 kernel：① kernel 内多 die 集合通信编排（TP/SP 集合通信与 EP
dispatch/combine 全部 device 侧发起，任务集无先例）；② 数据依赖的专家路由与权重流
（路由结果在运行时决定访存与通信形态；K3 满配下两份路由专家权重（逻辑 bf16）
合计约 55 GiB，现网 MXFP4 部署单 die 驻留约 2.2 GB，容量管理本身即一面墙）；③ fp32 状态 delta-rule 递归长链（错误沿
token 链与两个子层全局传播）；④ 整层融合跨度（32 输入 / 19 属性 / 4 输出、conv 与
递归状态原位更新语义、两次 AttnRes 混合首尾耦合）。输出仅在层尾可观测，层内没有
任何可校验断面，不可分段验证。待仓库增设 L6 档后调升。

多卡语义：`world_size ∈ {4, 8, 16}` 为任务属性。张量并行、序列并行、专家并行的
分片方式与集合通信算法属于实现方自由度（自带分布声明或自行优化），评测只按
「全局输入 → 全局输出与状态更新」判定。**主形状**为 world_size=8、B=8、Tmax=8
（q_len=8）；公开 case 采用 K3 缩放配置（见 §6），K3 满配为设计目标点。

现网基线（POD 128K、TP16×DP2、EP32，每 die 每层，稳态实测；基线为 host 编排的
算子链，该形态在本任务中仅作性能对标，**不是**合法提交形态）：

| 口径 | 注意力子层 | MoE 子层 | 整层 |
|---|---|---|---|
| kernel 时间 | 137.3 us | 225.2 us | 362.5 us |
| 含间隙跨度（对标目标） | 209.1 us | 265.8 us | 474.9 us |
| 理论下界（泳道/流量法） | | | pod 口径约 200~240 us；8 die 口径约 235~280 us |

## 2. 算子定义

### 记号

- `B` 请求数，`Tmax` 每请求 token 槽位数，`seq_lens[b]` 有效 token 数（varlen 右 padding）
- `H` 隐维；`Hq`/`HV` q,k / v 头数（GVA：`HV = G·Hq`）；`Kd`/`Vd` q,k / v 头维
- `Dc = 2·Hq·Kd + HV·Vd` 卷积通道数（通道顺序 q | k | v）；`W` 卷积宽度
- `NB` 本层可见的 AttnRes 快照数；衰减门低秩瓶颈维恒为 `Kd`（f_a 输出维）
- `S ∈ R^{HV×Kd×Vd}` 递归状态（fp32）；`ℓ = gate_lower_bound`
- MoE 侧：`E` 路由专家数，`k` 每 token 激活数，`L` latent 维
  （routed_expert_hidden_size），`I` 专家中间维（moe_intermediate_size），
  `Is = 2I` 共享专家中间维（num_shared_experts=2），`β_s / lβ` SituAndMul 参数
- 所有归一化的 `ε = rms_eps`；`RMSNorm(u; γ) = u · rsqrt(mean(u²) + ε) · γ`

### 数学定义（逐 token，全局语义）

**A1 AttnRes 混合与快照**（对齐 `modeling_kimi_linear._apply_attn_res`）。对每个
token，令 `V = [r_1, …, r_NB, x] ∈ R^{(NB+1)×H}`（快照与当前载体拼接）：

```
V̂_i   = V_i · rsqrt(mean(V_i²) + ε)                    （逐向量 RMS 归一化）
s_i   = Σ_H V̂_i ⊙ (γ_res ⊙ w_res)                      （本分支 γ_res=attnres_attn_norm_w，w_res=attnres_attn_proj_w）
p     = softmax(s)
x_mix = Σ_i p_i · V_i                                   （加权的是未归一化的 V）
```

`NB = 0` 时 `x_mix = x`。快照：`append_snapshot = True`（对应
`layer_idx % attn_res_block_size == 0`）时 `block_residual_out = concat(block_residual, x)`
（追加**未混合**的 x），且本层残差载体置空；否则 `block_residual_out = block_residual`
逐位透传。

**A2~A5 KDA 注意力**（对齐 `fla/layers/kda.py` 与 `fused_recurrent_kda` kernel 内公式）：

```
h        = RMSNorm(x_mix; rms_in_w)
[q|k|v]  = SiLU(CausalConv1dUpdate(h·[Wq|Wk|Wv], conv_state))    # 逐通道宽 W 因果卷积，带状态
f        = (h·Wf1)·Wf2 + dt_bias                                 # 衰减门低秩两级（f_a/f_b，无 bias），[HV, Kd]
g        = ℓ · sigmoid(exp(A_log) ⊙ f)          （ℓ ≠ null，K3 取 ℓ = -5.0）
         = -exp(A_log) ⊙ softplus(f)            （ℓ = null 分支）
β        = sigmoid(h·Wb)                        （allow_neg_eigval 时 ×2）
q̂        = q / sqrt(Σ_Kd q² + 1e-6) · scale     （scale 默认 1/√Kd）
k̂        = k / sqrt(Σ_Kd k² + 1e-6)
S_t      = S_{t-1} ⊙ exp(g_t)                   （沿 Kd 通道逐维衰减）
u_t      = β_t ⊙ (v_t − S_tᵀ k̂_t)               （erase 后写入）
S_t      = S_t + k̂_t u_tᵀ
o_t      = S_tᵀ q̂_t
o        = RMSNorm_head(o; o_norm_w) ⊙ sigmoid(h·Wg)   # 输出门全秩无 bias（use_full_rank_gate=True）；先归一化后乘门
attn     = o · Wo
```

GVA 时 q̂、k̂ 沿头维 `repeat_interleave` 到 HV。递归对每个 (b, h) 独立，初值取
`recurrent_state` 输入，末值写 `recurrent_state_out`。卷积状态为最近 `W-1` 个
pre-conv 输入（旧在前），只随有效 token 推进。

**A6 载体更新**（对齐 `_forward_attn_residual` 的 prefix_sum 语义）：
`x2 = x + attn`；快照层 `x2 = attn`（载体由快照承接）。残差基底是**未混合**的 x。

**LatentMoE 子层（M1~M6）**，对每个 token：

```
M1  m  = AttnResMix(x2, R_out)                      # 公式同 A1；分支参数换为
    h2 = RMSNorm(m; post_attn_norm_w, ε)            # attnres_mlp_norm_w / attnres_mlp_proj_w；
                                                    # R_out 为本层更新后的快照序列（快照层为 NB+1 份）
M2  logits = h2 · Wrᵀ                               # 打分在 fp32
    s      = sigmoid(logits)
    idx    = topk(s + bias_e, k)                    # 选择用校正分（noaux_tc），不排序
    w      = s[idx]                                 # 加权用未校正的原始分数
    （num_expert_group > topk_group 时先按组内 top2 之和选组，组外置 -inf 再 topk）
    moe_renormalize 时 w = w / (Σ_k w + 1e-20)；w = w · routed_scaling_factor
M3  z = h2 · W_lat_down                             # [L]
    对 idx 中每个专家 e：
      gu    = z · W_gu[e]                           # [2I]，前 I 为 gate，后 I 为 up
      act   = β_s·tanh(gate/β_s)·sigmoid(gate) ⊙ (lβ·tanh(up/lβ))     # SituAndMul
      e_out = act · W_dn[e]                         # [L]
    y_lat = Σ_k w · e_out                           # latent 空间加权求和
M4  y_lat = RMSNorm(y_lat; latent_norm_w, ε)        # latent_moe_use_norm=true
    y_r   = y_lat · W_lat_up                        # [H]
M5  shared = SituAndMul(h2 · W_sh_gu) · W_sh_dn     # 共享专家在 hidden 空间
    moe    = y_r + shared
M6  y = x2 + moe                                    # 载体残差；padding 位置 y = 0
```

### 并行语义与正确性契约

- **提交形态（硬约束）**：整层为**单次 kernel 提交**——host 只做一次下发（输入在层前
  就位，输出在层后取回），层内全部计算与全部 die 间通信（TP/SP 集合通信、EP
  dispatch/combine、状态交换）必须在这一个 kernel 内由 device 侧发起并完成。
  不允许拆成 host 编排的 kernel 序列，不允许以 host 侧 HCCL/aclnn 调用作为层内
  步骤，不允许层中 device↔host 往返，不允许多 stream 提交。kernel 内部的多核
  （AIC/AIV/SDMA）与核间流水编排属实现自由度。性能计时口径即该单次提交的
  device 端到端时间（in → out）。
- `world_size ∈ {4, 8, 16}`。评测 harness 在一个 `world_size` 大小的通信组上同步
  启动每 die 一份该 kernel。
- 默认输入放置为全量复制（每 die 可见全部逻辑张量）；实现方可随 kernel 附带一份
  分布声明（manifest），说明各输入希望的预分片方式与各输出/状态由哪个 die 的哪个
  分片提供，harness 按声明散布输入、收集输出后与全局 golden 对账。
- `recurrent_state` 的物理排布（如 v-first `[Vd, Kd]`）可由实现声明，逻辑对账
  一律按 `[B, HV, Kd, Vd]`。
- 现网参照（仅性能对标）：EP 组横跨全 pod（32 die，每 die 28 专家）、
  dispatch/combine 为融合通信 kernel、共享专家用独立流与路由路径重叠——此为
  host 编排的算子链形态，不满足本任务的提交形态约束。

## 3. 接口规范

### 算子原型

见 proto.yaml 的 schema（32 输入、19 属性、4 输出）。

### 输入参数说明

| 参数 | 形状 | dtype | 说明 |
|---|---|---|---|
| x | [B, Tmax, H] | bf16/fp16/fp32 | 载体隐状态（prefix_sum），varlen 右 padding |
| seq_lens | [B] | int32 | 有效 token 数，1..Tmax |
| block_residual | [B, Tmax, NB, H] | 同 x | AttnRes 快照 |
| conv_state | [B, Dc, W-1] | 同 x | 卷积历史，原位更新语义 |
| recurrent_state | [B, HV, Kd, Vd] | fp32 | 递归状态，原位更新语义 |
| w_q / w_k | [H, Hq·Kd] | 同 x | 投影权重，FRACTAL_NZ 交付 |
| w_v | [H, HV·Vd] | 同 x | NZ |
| w_conv | [Dc, W] | 同 x | 无 bias，通道顺序 q\|k\|v |
| w_f1 / w_f2 | [H, Kd] / [Kd, HV·Kd] | 同 x | 低秩衰减门 f_a/f_b，无 bias；f_a 为 ND（跟随现网），f_b 为 NZ |
| w_b | [H, HV] | 同 x | beta 投影，NZ |
| w_g | [H, HV·Vd] | 同 x | 全秩输出门（use_full_rank_gate=True），无 bias，NZ |
| A_log / dt_bias | [HV] / [HV·Kd] | fp32 | 门参数 |
| rms_in_w / o_norm_w | [H] / [Vd] | 同 x | 归一化权重 |
| w_o | [HV·Vd, H] | 同 x | 输出投影，NZ |
| attnres_attn_norm_w / attnres_attn_proj_w | [H] | 同 x | AttnRes 注意力分支参数 |
| attnres_mlp_norm_w / attnres_mlp_proj_w | [H] | 同 x | AttnRes MLP 分支参数 |
| post_attn_norm_w | [H] | 同 x | post-attention RMSNorm |
| router_w | [E, H] | 同 x | 路由打分权重，打分 fp32 |
| router_bias | [E] | fp32 | 选择校正偏置，仅参与选择不参与加权 |
| w_latent_down / w_latent_up | [H,L] / [L,H] | 同 x | latent 投影（NZ，现网 bf16） |
| latent_norm_w | [L] | 同 x | latent RMSNorm |
| w_expert_gate_up | [E, L, 2I] | 同 x | 专家 gate\|up（前 I 列 gate） |
| w_expert_down | [E, I, L] | 同 x | 专家 down |
| w_shared_gate_up / w_shared_down | [H,2Is] / [Is,H] | 同 x | 共享专家（现网 bf16） |

### 输出

| 输出 | 形状 | dtype | 说明 |
|---|---|---|---|
| y | [B, Tmax, H] | 同 x | 整层输出（含 MoE 残差），padding 位置为 0 |
| conv_state_out | [B, Dc, W-1] | 同 conv_state | 只随有效 token 推进 |
| recurrent_state_out | [B, HV, Kd, Vd] | fp32 | |
| block_residual_out | [B, Tmax, NB(+1), H] | 同 x | 非快照层逐位等于输入（负测试项） |

### 规则与约束

1. 递归状态与 delta 反馈 fp32；路由打分、SituAndMul、latent 加权求和按 golden
   的 fp32 语义（建模源码即 fp32 计算）。
2. 权重排布跟随现网：衰减门一级 w_f1 为 ND、其余投影（含全秩输出门 w_g、latent 投影）NZ。
   实现不得假设可以离线改排布，如需其他排布须在 kernel 内自行转换并计入耗时。
3. **专家权重量化**：现网路由专家权重为 MXFP4（group=32 对称量化），激活经
   DynamicMxQuant；共享专家、latent 投影、注意力权重不量化。本任务按逻辑浮点
   交付专家权重、golden 为精确数学；实现方可在 kernel 内自行量化，但须满足
   精度阈值。以 MXFP4 张量为输入的量化保真变体是后续任务扩展项。
4. 专家总数 E 不随 die 数缩减（K3 为 896）；EP 分片属实现自由度。
5. top-k 并列打破遵循 torch.topk；专家枚举顺序不影响结果（置换不变性已验证）。
6. 非快照层 block_residual_out、未触及状态槽位逐位透传（负测试项）。
7. `HV % Hq == 0`。**路由参数硬约束**（golden 显式校验，违反抛 ValueError）：
   `E % num_expert_group == 0` 且 `E / num_expert_group >= 2`（组分为组内
   top2 之和）；`1 <= topk_group <= num_expert_group`；
   `1 <= top_k <= topk_group · (E / num_expert_group)`（否则 top-k 会选入
   被 mask 组的专家）。回归测试见
   `tests/ut/test_k3_grouped_routing_constraints.py`。
8. 门与归一化的取值范围（cases 的 value_range）是稳定性条件的一部分：
   `A_log ∈ [0, ln16]`、`dt_bias ∈ [-6.9, -2.25]`、归一化权重取 1 附近，
   任意随机输入下递归稳定。
9. **提交形态硬约束见 §2**：单 kernel、单 stream、层内无 host 参与；违反即为
   无效提交。

### 支持范围

- `B ∈ [1, 64]`，`Tmax ∈ [1, 16]`（decode/推测解码域；主口径 B=world_size, Tmax=8）
- `H ∈ [1024, 8192]`，`E ∈ [32, 896]`，`k ∈ [4, 16]`，`L ∈ [512, 3584]`，`I ∈ [384, 3072]`
- 公开 case 主配置（K3 缩放，case 1~6）：`H=3584, Hq=HV=48, Kd=Vd=128, W=4,
  E=112, k=16, L=1024, I=768, β_s=4.0, lβ=25.0, ℓ=-5.0, ε=1e-5`
- Kimi K3 满配（设计目标点，不进入公开 case，理由见 §6）：`H=7168, Hq=HV=96,
  Kd=Vd=128, W=4, E=896, k=16, L=3584, I=3072`，其余参数同上

## 4. 精度要求

| dtype | 阈值 |
|---|---|
| float32 | 0.001 |
| float16 | 0.005 |
| bfloat16 | 0.02 |

标定口径：err=|a−b|/max(|b|,1e-3)，MERE≤阈值、MARE≤10×阈值，fp64_cpu oracle 为
真值，输出转回用例 dtype 后比对。**全部 20 个公开 case 已在 CPU 侧实测通过**
（多 seed，0 fail；各 dtype 的 MARE 下限恰为一个量化步，余量 19x~51x，逐档
数值在 proto 注释），无待定项。负测试（透传槽位篡改、padding 位篡改）与换和序
正确实现（专家置换、逐 token 朴素路径）的余量已验证。

## 5. 标准 Golden 代码

见同目录 `golden.py`：`k3_kda_decoder_layer_e2e`（plain，内部 fp32）与
`k3_kda_decoder_layer_e2e_oracle`（跟随输入精度）。已验证：MoE 路径
（w_expert_down/w_shared_down/w_latent_up）置零时退化为纯注意力子层且与其独立
golden 逐位相等；整层与独立逐 token 朴素实现 fp64 对拍至 1e-15 量级；专家置换
不变；链式状态一致（分段调用状态串联 == 单次调用）。

## 6. 额外信息

### 主用例与 K3 满配设计目标点

公开主用例（case 1，K3 缩放配置）：world_size=8、B=8 请求 × Tmax=8（q_len=8），
bf16，NB=8，`H=3584, Hq=HV=48, E=112, L=1024, I=768`——保持 K3 的层结构、
B/T/ws 形态与全部机制（AttnRes、快照、GVA、分组路由可开、双份专家权重流），
尺寸缩放到默认评测链路（fp64_cpu oracle 的 CPU 宿主）可普遍执行并可标定。

K3 满配（`H=7168, Hq=HV=96, E=896, L=3584, I=3072`）为设计目标点，**不进入
公开 case**：该规格单份 case 原始输入约 56 GiB（bf16，其中 w_expert_gate_up +
w_expert_down 两份专家权重合计 55.1 GiB），默认 fp64_cpu oracle 在原输入存活时
构造 fp64 副本，仅进入 oracle 前峰值即约 4~5 倍原始字节（数百 GiB），不满足
「公开 case 须在默认评测链路普遍可执行」。满配下 B=8×T=8×ws=8 使每 die 激活
工作量与 32-die pod 基线恒等（64 token × 每 die 12 头 = 128 token × 每 die
6 头），权重流量为 pod 每 die 的 2 倍（TP8 分片）；该规格的性能对标经硬件侧
baseline/t_hw metadata 锚定（比较方法见仓库 README：比值口径 + 逐算子锚定 +
距下界比值），正确性校验如需满配复核，属大内存专用集合的后续扩展项。

### 性能参照（现网 profiling，每 die 每层）

整层基线 474.9 us（含间隙；host 编排算子链口径）；理想化后 cube/权重流泳道为
最长泳道：pod 口径下界约 200~240 us，8 die 口径约 235~280 us（聚簇路由前提）。

### 退化与组合关系

- MoE 权重（w_expert_down / w_shared_down / w_latent_up）置零时退化为纯 KDA
  注意力子层（已作为等价测试固化）；KDA 注意力子层单独成任务是后续扩展项。
- `E = k` 且 renormalize 时退化为稠密 latent FFN 的加权平均；
  `num_expert_group = 1` 即 K3 配置（无分组限制路由）。
- MLA 型层（93 层中的 24 层）不在本任务范围。

### 参考文献

- Kimi Team, "Kimi K3: Open Frontier Intelligence", arXiv:2607.24653
- Kimi Team, "Kimi Linear: An Expressive, Efficient Attention Architecture", arXiv:2510.26692
- moonshotai/Kimi-K3（HuggingFace）：config.json、modeling_kimi_linear.py（KimiMoEGate /
  KimiSparseMoeBlock / SituAndMul / KimiDeltaAttention / KimiDecoderLayer._forward_attn_residual）
- fla-org/flash-linear-attention：fla/layers/kda.py、fla/ops/kda/fused_recurrent.py、
  fla/ops/kda/naive.py、fla/modules/fused_norm_gate.py
- cann-bench bench_lab/kernel_bench/level5：gated_deltanet2_chunkwise（同族递归）、
  decoder_layer_megakernel（整层融合先例）、colocated_prefill_decode（多负载编排先例）
