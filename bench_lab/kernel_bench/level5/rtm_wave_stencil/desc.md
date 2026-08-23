# RtmWaveStencil 算子 API 描述

## 1. 算子简介

地震逆时偏移（Reverse Time Migration，RTM）正/反向波场传播的核心 kernel：二维声波方程

$$
\frac{\partial^2 p}{\partial t^2} = v^2 \nabla^2 p
$$

的显式时间步进——空间用 8 阶中心差分离散拉普拉斯算子，时间用 2 阶 leapfrog 格式，叠加简化 Cerjan 型海绵吸收边界（逐点衰减因子 $g \le 1$）。RTM 对每个炮点做一次正向传播与一次逆时传播并互相关成像，波场传播占整个偏移作业 90% 以上的算力；同一 kernel 也是全波形反演（FWI）梯度计算的内核。工业规模的偏移作业对单炮波场推进的吞吐极其敏感，这类高阶 stencil 时间步进是 HPC 领域带宽受限计算的标杆负载。

**主要应用场景**：
- 石油/天然气勘探的叠前深度偏移（RTM 成像），正向与逆时传播共用本 kernel
- 全波形反演（FWI）中正演模拟与伴随波场计算
- 地震灾害模拟、超声成像等波动方程显式时间推进负载

**算子特征**：
- 难度等级：L5（FusedComposite），任务集中首个高阶 stencil 时间步进机制轴算子
- 五输入（p_prev, p_curr, velocity, damp_x, damp_z）+ 三 attr（num_steps, dt, dx），双输出（最终两个时间片）
- 每步融合 8 阶叉形 stencil（17 点）、速度场逐点缩放、leapfrog 时间更新与海绵衰减
- 数值有界性由 case 生成保证 CFL 约束 $v_{\max} \cdot dt / dx \le 0.4$（本格式 2D 稳定上限 $\approx 0.5546$，见 §2），任意满足 CFL 的随机输入下 64 步内场值有界（已数值验证）

**为何是 L5**（约为 L4 FusedComposite 的 2 倍难度）：
- **正确 ≠ 快**：按 §2 公式逐步直译（每步读全场、算 stencil、写全场）即可得到完全正确的结果并通过全部精度用例，但每步 4 个全场张量往返 HBM（读 p_prev/p_curr/velocity、写 p_next），每点约 25 次浮点操作对约 16 字节访存，num_steps（最长 64）次全场往返使总访存放大 64 倍——性能只能锚定朴素 baseline（预期得分 ~0.5 档）。性能墙在时间维：多个时间步共享片上驻留数据（时间维分块，temporal blocking）才能把访存从"每步全场"压到"每块一进一出"，而 8 阶 stencil 宽度 4 的 halo 使相邻分块间需要跨核交换波场边缘、且逐步扩大依赖锥，时间分块深度与 halo 交换开销的权衡是本算子的核心工程问题（此处仅点名领域算法概念，实现方案不限）
- **实现规格而非过可见测试**：隐藏评测集覆盖公开用例之外的规格维度（Nz ≠ Nx 极端长宽比、全素数维、num_steps 奇数（双缓冲滚动的奇偶陷阱）、v 常数/窄带值域、damp 单方向为 0、CFL 顶界、波场大/小幅值等），逐点定义中每一处约定（halo 置 0 时机、g 的作用位置、系数除以 dx² 的位置、输出是"最终两个时间片"）都会被单独检验
- **项多且不规则**：17 点叉形 stencil 系数 5 个、速度场逐点平方缩放、1D×1D 广播的海绵因子、宽 4 的固定零边界、双时间片滚动输出，任何一处顺序或广播维度写错都会被隐藏用例捕获
- **误差随步数累积**：64 步时间链上 fp32 舍入误差逐步放大（实测见 §4），中间量精度与求和顺序的管理直接决定能否达标

## 2. 算子定义

### 空间离散（8 阶中心差分）

一维二阶导的 8 阶中心差分系数（标准 Taylor 展开系数，来源：Fornberg 1988 通用差分系数算法的 8 阶特例）：

| $c_0$ | $c_1$ | $c_2$ | $c_3$ | $c_4$ |
|---|---|---|---|---|
| $-205/72$ | $8/5$ | $-1/5$ | $8/315$ | $-1/560$ |

**系数已数值验证**：对解析函数 $\sin(kx)$ 施加该 stencil，与 $-k^2\sin(kx)$ 对照，网格加密时误差按 8 阶收敛（fp64 实测收敛阶 7.66 / 7.91 / 7.98，2D 拉普拉斯乘积函数实测 7.92 / 7.98）。

二维拉普拉斯（z/x 同间距 $dx$）：

$$
\mathrm{lap}[i,j] = \frac{1}{dx^2}\left( 2c_0\, p[i,j] + \sum_{k=1}^{4} c_k \big( (p[i\!-\!k,j] + p[i\!+\!k,j]) + (p[i,j\!-\!k] + p[i,j\!+\!k]) \big) \right)
$$

### 时间步进与海绵衰减

$$
p_{\text{next}} = \big( 2\, p_{\text{curr}} - p_{\text{prev}} + (v \cdot dt)^2 \cdot \mathrm{lap} \big) \cdot g,
\qquad
g[i,j] = \frac{1}{1 + \text{damp\_z}[i] + \text{damp\_x}[j]}
$$

$g$ 是简化的 Cerjan 型海绵衰减（Cerjan et al. 1985 的指数衰减在小衰减系数下的一阶有理近似）：damp 剖面通常只在近边界处非零，但**任意随机剖面均为合法输入**（$g \le 1$ 恒成立，只会增强衰减）。

### 边界与滚动

- **固定零边界**：$p$ 的最外 4 圈（= stencil 半宽的 halo）恒为 0。置零是算子语义的一部分：第一步前先把 p_prev/p_curr 的最外 4 圈置 0（任意随机输入合法），此后每步更新完成后把 p_next 的最外 4 圈置 0。halo 恒为 0 保证内点 stencil 从不读到未定义值
- **滚动**：$p_{\text{prev}} \leftarrow p_{\text{curr}}$，$p_{\text{curr}} \leftarrow p_{\text{next}}$；重复 num_steps 次
- **输出**：最终两个时间片 (p_out_prev, p_out_curr)，即 $t + \text{num\_steps} - 1$ 与 $t + \text{num\_steps}$ 时刻的波场（leapfrog 推进需要两个时间片，输出两片使框架侧可无缝续推）

### CFL 稳定性约束（case 生成保证）

本格式的 von Neumann 稳定性上限由 stencil 的 2D 谱半径决定：一维符号在 Nyquist 波数处取 $|c_0 - 2c_1 + 2c_2 - 2c_3 + 2c_4| = 6.5016$，2D 系数 $13.0032$，稳定条件 $(v\,dt/dx)^2 \le 4/13.0032$，即

$$
\text{CFL} = \frac{v_{\max}\, dt}{dx} \le 0.5546
$$

评测 case 全部满足更保守的 $v_{\max} \cdot dt / dx \le 0.4$（$v_{\max}$ 取该 case velocity 取值范围上界），留出裕量使任意随机速度场/初始波场下 64 步内场值有界（fp64 实测：damp=0、CFL=0.4、随机 v，64 步幅值 ≤ 11 倍初始幅值，无增长发散）。算子不对违反 CFL 的输入负责。

### 性能墙（为什么朴素实现只能锚定 baseline）

逐步直译的访存模式：每步读 p_prev/p_curr/velocity 三个全场、写 p_next 一个全场（约 16 B/点/步），每点约 25 次浮点操作，算强 ~1.5 flop/B——典型带宽受限；num_steps 步 = num_steps 次全场 HBM 往返，数据完全无复用。墙在两处：**时间维分块**（temporal blocking——让片上驻留的波场块连续推进多个时间步，把全场往返摊薄）与 **8 阶宽 halo 的跨核交换**（半宽 4 的依赖锥随时间分块深度逐步扩大，分块深度与冗余计算/交换量的权衡）。这里只点名领域算法概念，具体映射方式不限。

## 3. 接口规范

### 算子原型

```python
rtm_wave_stencil(Tensor p_prev, Tensor p_curr, Tensor velocity, Tensor damp_x, Tensor damp_z, int num_steps, float dt, float dx) -> (Tensor p_out_prev, Tensor p_out_curr)
```

### 输入参数说明

| 参数 | 类型 | 必需 | dtype | shape | 描述 |
|------|------|------|-------|-------|------|
| p_prev | Tensor | 是 | float32 | [B, Nz, Nx] | 时间片 t-1 的波场，最外 4 圈由算子置 0 |
| p_curr | Tensor | 是 | float32 | [B, Nz, Nx] | 时间片 t 的波场，最外 4 圈由算子置 0 |
| velocity | Tensor | 是 | float32 | [B, Nz, Nx] | 介质声速场，单位 m/s（评测取值范围 [1500, 4500]） |
| damp_x | Tensor | 是 | float32 | [Nx] | 海绵层 x 方向衰减剖面（评测取值范围 [0, 0.05]） |
| damp_z | Tensor | 是 | float32 | [Nz] | 海绵层 z 方向衰减剖面（评测取值范围 [0, 0.05]） |
| num_steps | int | 是 | - | 标量 | 时间步数（评测取值 1 ~ 64） |
| dt | float | 是 | - | 标量 | 时间步长，单位 s（评测取值范围 [2e-4, 8e-4]） |
| dx | float | 是 | - | 标量 | 网格间距，单位 m，z/x 同间距（评测取值范围 [5, 20]） |

### 输出

| 名称 | dtype | shape | 描述 |
|------|-------|-------|------|
| p_out_prev | float32 | [B, Nz, Nx] | 最终时间片 t+num_steps-1（最外 4 圈为 0） |
| p_out_curr | float32 | [B, Nz, Nx] | 最终时间片 t+num_steps（最外 4 圈为 0） |

### 数据类型

| p_prev/p_curr/velocity/damp_x/damp_z dtype | 输出 dtype | 内部计算 |
|---|---|---|
| float32 | float32 | fp32 |

### 规则与约束

- 五个 Tensor 输入 dtype 必须一致（float32）
- p_prev/p_curr/velocity 的 shape 完全一致（[B, Nz, Nx]）；damp_x 长度 = Nx，damp_z 长度 = Nz
- 最外 4 圈置 0 是算子语义：第一步前对两个输入时间片置 0，每步更新后对 p_next 置 0
- CFL 约束 $v_{\max} \cdot dt / dx \le 0.4$ 由 case 生成保证（$v_{\max}$ = velocity 取值范围上界）；算子不对违反 CFL 的输入负责
- num_steps ≥ 1；num_steps 为奇数时两个输出时间片的滚动奇偶与偶数步不同（双缓冲实现须正确处理）
- 输出须为 contiguous 张量

### 支持范围

| 维度 / 参数 | 范围 | 备注 |
|---|---|---|
| `B`（batch，炮点数） | 1 ~ 8 | cases.csv 实测 1 ~ 8 |
| `Nz` × `Nx`（网格） | 64 ~ 1024（每维） | cases.csv 实测 64² ~ 1024²，含非方形与素数维 |
| `num_steps` | 1 ~ 64 | cases.csv 实测 1 / 4 / 16 / 64 |
| `dt` | [2e-4, 8e-4] s | attr |
| `dx` | [5, 20] m | attr |
| `p_prev` / `p_curr` 取值 | [-1, 1] | 常规随机范围；隐藏用例含 [-10, 10] 与小幅值 |
| `velocity` 取值 | [1500, 4500] m/s | 物理量程（水速 ~1500 至盐丘/基岩 ~4500）；隐藏用例含常数与窄带值域 |
| `damp_x` / `damp_z` 取值 | [0, 0.05] | 任意随机剖面合法 |
| dtype | float32 | 全部用例 |

## 4. 精度要求

采用[生态算子精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md)进行验证。

**误差指标**：

1. 平均相对误差（MERE）：采样点中相对误差平均值

   $$
   \text{MERE} = \text{avg}(\frac{\text{abs}(actual - golden)}{\text{abs}(golden)+\text{1e-7}})
   $$

2. 最大相对误差（MARE）：采样点中相对误差最大值

   $$
   \text{MARE} = \max(\frac{\text{abs}(actual - golden)}{\text{abs}(golden)+\text{1e-7}})
   $$

**通过标准**（本算子放宽阈值）：

时间步进链（num_steps 最长 64）误差随步数累积，且波场是振荡场、场值连续过零——绝对误差全场均布而相对误差在过零点附近出现尖峰（同 batched_svd / dqmc_hubbard_greens 一类，由评测框架的小值域/相消兜底标准处理，native 参考 = plain golden）。阈值经评测框架 checker 实测确定：独立 fp32 路径（求和顺序不同的 stencil 实现）vs fp64 oracle 在 14 组配置 × 3~6 draw 上校准，checker 全量判 passed，实测正常值域 MERE ≤ 3.2e-5、正常值域 MARE 尖峰 ≤ 4e-2：

| 数据类型 | FLOAT32 |
|----------|---------|
| **通过阈值(Threshold)** | 0.02 |

当平均相对误差 MERE < Threshold，最大相对误差 MARE < 10 * Threshold 时判定为通过。MERE 裕量 ≥ 600x；判别力实测：2 阶 stencil 错误实现 MERE = 2.9、漏乘海绵因子 MERE = 20，与阈值相隔 2 个数量级以上——阈值放宽不损失判别力。case 空间约束：波场取值上限 ±10（±100 时过零点相对误差尖峰随 draw 波动过大，该角点不在 case 空间内）。

## 5. 标准 Golden 代码

```python
import torch
from typing import Tuple

# 8 阶中心差分二阶导系数（标准 Taylor 展开系数，见 desc §2 表格；已数值验证 8 阶收敛）
_FD8_C0 = -205.0 / 72.0
_FD8_CK = (8.0 / 5.0, -1.0 / 5.0, 8.0 / 315.0, -1.0 / 560.0)
_HALO = 4


def _zero_halo(p):
    """将最外 4 圈（stencil halo）置 0（就地修改并返回）。"""
    p[..., :_HALO, :] = 0
    p[..., -_HALO:, :] = 0
    p[..., :, :_HALO] = 0
    p[..., :, -_HALO:] = 0
    return p


def _rtm_wave_stencil_core(p_prev, p_curr, velocity, damp_x, damp_z,
                           num_steps, dt, dx, compute_dtype):
    """核心计算：以 compute_dtype 精度执行 num_steps 步 8 阶 stencil 时间步进。"""
    pp = _zero_halo(p_prev.to(compute_dtype).clone())      # [B, Nz, Nx]
    pc = _zero_halo(p_curr.to(compute_dtype).clone())      # [B, Nz, Nx]
    v = velocity.to(compute_dtype)
    dz = damp_z.to(compute_dtype)                          # [Nz]
    dxp = damp_x.to(compute_dtype)                         # [Nx]

    # 海绵衰减因子 g[i,j] = 1 / (1 + damp_z[i] + damp_x[j])，广播到 [Nz, Nx]
    g = 1.0 / (1.0 + dz.unsqueeze(-1) + dxp.unsqueeze(0))
    v_dt_sq = (v * dt) ** 2                                # [B, Nz, Nx]
    dx2 = dx * dx

    for _ in range(num_steps):
        # 8 阶叉形 stencil：lap = 2*c0*p + Σ_k c_k*((p[i-k]+p[i+k]) + (p[j-k]+p[j+k]))
        # halo 恒为 0，roll 的环回值全部落在 halo 内且每步后置 0，不污染内点
        lap = (2.0 * _FD8_C0) * pc
        for k, ck in enumerate(_FD8_CK, start=1):
            lap = lap + ck * (
                (torch.roll(pc, k, dims=-2) + torch.roll(pc, -k, dims=-2))
                + (torch.roll(pc, k, dims=-1) + torch.roll(pc, -k, dims=-1))
            )
        pn = (2.0 * pc - pp + (v_dt_sq * lap) / dx2) * g
        _zero_halo(pn)
        pp, pc = pc, pn
    return pp.contiguous(), pc.contiguous()


def rtm_wave_stencil(
    p_prev: torch.Tensor,
    p_curr: torch.Tensor,
    velocity: torch.Tensor,
    damp_x: torch.Tensor,
    damp_z: torch.Tensor,
    num_steps: int,
    dt: float,
    dx: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    RTM 8 阶声波 stencil 时间步进 golden reference（plain golden = bench：fp32 计算）

    Args:
        p_prev: [B, Nz, Nx] 时间片 t-1 的波场（算子先将最外 4 圈置 0）
        p_curr: [B, Nz, Nx] 时间片 t 的波场（算子先将最外 4 圈置 0）
        velocity: [B, Nz, Nx] 介质声速场（评测取值范围 [1500, 4500] m/s）
        damp_x: [Nx] 海绵层 x 方向衰减剖面（评测取值范围 [0, 0.05]，任意随机剖面合法）
        damp_z: [Nz] 海绵层 z 方向衰减剖面（评测取值范围 [0, 0.05]，任意随机剖面合法）
        num_steps: 时间步数（评测取值 1 ~ 64）
        dt: 时间步长，单位 s（评测取值范围 [2e-4, 8e-4]；case 保证 v_max*dt/dx ≤ 0.4）
        dx: 空间网格间距，单位 m（评测取值范围 [5, 20]，z/x 同间距）

    Returns:
        p_out_prev: [B, Nz, Nx] 最终时间片 t+num_steps-1，dtype 与 p_prev 一致
        p_out_curr: [B, Nz, Nx] 最终时间片 t+num_steps，dtype 与 p_prev 一致
    """
    p_out_prev, p_out_curr = _rtm_wave_stencil_core(
        p_prev, p_curr, velocity, damp_x, damp_z,
        int(num_steps), float(dt), float(dx), torch.float32)
    return p_out_prev.to(p_prev.dtype), p_out_curr.to(p_prev.dtype)


def rtm_wave_stencil_oracle(
    p_prev: torch.Tensor,
    p_curr: torch.Tensor,
    velocity: torch.Tensor,
    damp_x: torch.Tensor,
    damp_z: torch.Tensor,
    num_steps: int,
    dt: float,
    dx: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Oracle (g)：dtype-agnostic，计算精度跟随输入（fp64_cpu 下为 fp64 真值），不硬编码 .float()。"""
    return _rtm_wave_stencil_core(
        p_prev, p_curr, velocity, damp_x, damp_z,
        int(num_steps), float(dt), float(dx), p_prev.dtype)
```

## 6. 额外信息

### 算子调用示例

```python
import torch

B, Nz, Nx = 2, 512, 512

p_prev = torch.rand(B, Nz, Nx, dtype=torch.float32, device="npu") * 2 - 1
p_curr = torch.rand(B, Nz, Nx, dtype=torch.float32, device="npu") * 2 - 1
velocity = torch.rand(B, Nz, Nx, dtype=torch.float32, device="npu") * 3000 + 1500
damp_x = torch.rand(Nx, dtype=torch.float32, device="npu") * 0.05
damp_z = torch.rand(Nz, dtype=torch.float32, device="npu") * 0.05

p_out_prev, p_out_curr = rtm_wave_stencil(p_prev, p_curr, velocity, damp_x, damp_z,
                                          num_steps=16, dt=4e-4, dx=10.0)
# 输出 shape 均为 [B, Nz, Nx]，CFL = 4500 * 4e-4 / 10 = 0.18 ≤ 0.4
```

### 可用于自检的性质（均已数值验证）

- **FD 收敛阶**：stencil 对 $\sin(kx)$ 的二阶导误差按 8 阶收敛（实测 7.66 → 7.98）
- **全格式收敛**：均匀介质驻波解析解 $\sin(\pi x/L)\sin(\pi z/L)\cos(\omega t)$（$\omega = v\pi\sqrt{2}/L$）下，固定 CFL 比例细化时内区误差按时间 2 阶收敛（实测阶 2.15 / 2.08；空间 8 阶误差远低于时间项）
- **能量有界**：damp=0、CFL=0.4 顶界、随机速度场，64 步幅值有界（实测 ≤ 11 倍初始幅值）
- **衰减单调**：damp 取上界时任意时刻幅值小于 damp=0 的同输入结果
- **单步退化**：num_steps=1 时 p_out_prev = 置零 halo 后的 p_curr，p_out_curr = 单步手算结果
- **零输入不动点**：p_prev = p_curr = 0 时输出恒 0

### 参考文献

- Baysal, E., Kosloff, D. D., Sherwood, J. W. C. (1983). "Reverse time migration". Geophysics 48(11)（RTM 方法来源）
- Cerjan, C., Kosloff, D., Kosloff, R., Reshef, M. (1985). "A nonreflecting boundary condition for discrete acoustic and elastic wave equations". Geophysics 50(4)（海绵吸收边界）
- Fornberg, B. (1988). "Generation of finite difference formulas on arbitrarily spaced grids". Mathematics of Computation 51(184)（高阶差分系数）
- Virieux, J., Operto, S. (2009). "An overview of full-waveform inversion in exploration geophysics". Geophysics 74(6)（FWI 场景）
- Micikevicius, P. (2009). "3D finite difference computation on GPUs using CUDA". GPGPU-2（高阶 stencil 时间步进的访存墙与 halo 分析）
