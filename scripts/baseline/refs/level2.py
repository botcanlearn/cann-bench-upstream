"""level2 reference implementations on NPU."""
import torch


def apply_adam_w_ref(inputs, attrs):
    """torch_npu.npu_apply_adam_w fused NPU 路径 (CANN 9.0.0 实测签名)。

    Bug fix (refs-bug-2, 2026-05-15): 原版是纯 PyTorch composite (Mul/Sub/Add/Sqrt/Div × N),
    与 NPU production 路径 (单 fused kernel) 不一致 —— 写 cases.yaml 的 baseline_kernels
    拓扑会反映这串 element-wise，误导后续版本对比。改 fused API 后必须 rebaseline 全 20 case。

    实测签名 (P3 NPU probe, CANN 9.0.0 / torch_npu 2.9.0):
      npu_apply_adam_w(
        beta1_power: Scalar, beta2_power: Scalar, lr: Scalar, weight_decay: Scalar,
        beta1: Scalar, beta2: Scalar, epsilon: Scalar, grad: Tensor,
        max_grad_norm: Tensor?, amsgrad: bool?, maximize: bool?,
        *, out: Tensor[]  # [var, m, v] 作 out 缓冲，原地更新
      ) -> (Tensor, Tensor, Tensor)

    R1 risk realized: npu_apply_adam_w_v2 在 CANN 9.0.0 不存在 —— 直接用 v1。
    baseline-perf 测单次 step，step=1 时 beta1_power=beta1, beta2_power=beta2。
    """
    import torch_npu
    var, grad, m, v = inputs[0], inputs[1], inputs[2], inputs[3]

    lr = float(attrs.get("lr", 1e-3))
    beta1 = float(attrs.get("beta1", 0.9))
    beta2 = float(attrs.get("beta2", 0.999))
    weight_decay = float(attrs.get("weight_decay", 0.0))
    epsilon = float(attrs.get("epsilon", 1e-8))
    maximize = bool(attrs.get("maximize", False))
    beta1_power = beta1  # step=1
    beta2_power = beta2

    return torch_npu.npu_apply_adam_w(
        beta1_power, beta2_power, lr, weight_decay,
        beta1, beta2, epsilon, grad,
        None,     # max_grad_norm: 未启用
        False,    # amsgrad: cases.yaml 不暴露
        maximize,
        out=[var, m, v],
    )


def apply_rotary_pos_emb_ref(inputs, attrs):
    """torch_npu.npu_apply_rotary_pos_emb wrap 到 golden.py signature。

    golden.py:apply_rotary_pos_emb(query, key, cos, sin, layout=0, rotaryMode='half')
    接受 cos/sin 半 D (S, D/2) 或 (B, S, D/2)，layout int (0=BSND, 1=BNSD)，
    rotaryMode 'half' | 'interleaved'。

    torch_npu Python 入口 (`npu_apply_rotary_pos_emb`) 比 golden.py 严格：
      - cos/sin.ndim == q.ndim 且 dim 0/1/2 必须与 q 完全相等 (无 broadcast)
      - cos/sin 最后一维 = D (full head_dim), 不接受半 D
      - rotary_mode: 'half'|'interleave'|'quarter'; 不接受 cases.yaml 的 'interleaved'
      - head_dim 必须 ∈ {64, 128} (NPU 硬限制 ApplyRotaryPosEmb tiling)
      - 不接 layout 参数 (默认 BSND); layout=1 BNSD 时要 transpose cos/sin

    本 ref 在 Python 层做 wrapper.cpp::NormalizeRotaryMode + Reshape + Expand 等价工作:
      1. 'interleaved' → 'interleave'
      2. (S, D/2) / (B, S, D/2) → (B, S, 1, D)  (cat 复制半 D, expand 到 B & 1)
      3. layout=1 → transpose(1, 2) 得 (B, 1, S, D), expand N 到 q.shape[1]
    这会让 baseline_kernels 拓扑含 Concat + Expand kernels (与 ACLNN aten 直入口的
    纯 ApplyRotaryPosEmb 拓扑不同, 是 Python 入口的代价)。

    head_dim ∉ {64, 128} 的 case 由 NPU 硬限制拒绝, baseline_perf_us 保留 null。
    """
    import torch_npu
    q, k, cos, sin = inputs[0], inputs[1], inputs[2], inputs[3]
    layout = int(attrs.get("layout", 0))
    mode = attrs.get("rotaryMode", "half")
    if mode == "interleaved":
        mode = "interleave"

    if mode == "interleave":
        cos_full = cos.repeat_interleave(2, dim=-1)
        sin_full = sin.repeat_interleave(2, dim=-1)
    else:
        cos_full = torch.cat([cos, cos], dim=-1)
        sin_full = torch.cat([sin, sin], dim=-1)

    B = q.shape[0]
    if cos_full.dim() == 2:
        cos_4d = cos_full.unsqueeze(0).unsqueeze(2)   # (1, S, 1, D)
        sin_4d = sin_full.unsqueeze(0).unsqueeze(2)
    elif cos_full.dim() == 3:
        cos_4d = cos_full.unsqueeze(2)                # (B, S, 1, D)
        sin_4d = sin_full.unsqueeze(2)
    else:
        cos_4d, sin_4d = cos_full, sin_full

    # BSND-form cos/sin: ACLNN 要求 dim0==B, dim2==1 (broadcast over N)
    cos_4d = cos_4d.expand(B, -1, 1, -1).contiguous()
    sin_4d = sin_4d.expand(B, -1, 1, -1).contiguous()

    if layout == 1:
        # torch_npu.npu_apply_rotary_pos_emb 不接 layout 参数，假定 BSND。
        # BNSD 输入需先 transpose q/k 到 BSND, 调完再 transpose 回。
        q_in = q.transpose(1, 2).contiguous()
        k_in = k.transpose(1, 2).contiguous()
        q_out, k_out = torch_npu.npu_apply_rotary_pos_emb(q_in, k_in, cos_4d, sin_4d, rotary_mode=mode)
        return q_out.transpose(1, 2).contiguous(), k_out.transpose(1, 2).contiguous()

    return torch_npu.npu_apply_rotary_pos_emb(q, k, cos_4d, sin_4d, rotary_mode=mode)


def arg_max_ref(inputs, attrs):
    """torch.argmax(input, dim, keepdim).

    Bug fix (refs-bug-1, 2026-05-15): 原版 attrs.get("dimension", -1) 永远落
    default —— cases.yaml 的 attr key 是 `dim` (proto schema 也是)。这导致
    case 4/6/8 (dim=0) 测的根本不是配置的维度。
    """
    return torch.argmax(
        inputs[0],
        dim=attrs.get("dim", -1),
        keepdim=attrs.get("keepdim", False),
    )


def cross_entropy_loss_ref(inputs, attrs):
    """torch.nn.functional.cross_entropy 完整透传。

    Bug fix (refs-bug-3, 2026-05-15): 原版未透传 weight / label_smoothing —— 当前
    cases.yaml 未触发但未来 case 会被静默忽略。
    """
    import torch
    weight = attrs.get("weight")  # 可为 None
    # weight 在 cases.yaml 中是 list / None；若是 list 需转 tensor —— 当前所有 case 都是 None
    if isinstance(weight, list):
        weight = torch.tensor(weight, dtype=inputs[0].dtype, device=inputs[0].device)
    return torch.nn.functional.cross_entropy(
        inputs[0], inputs[1],
        weight=weight,
        reduction=attrs.get("reduction", "mean"),
        ignore_index=attrs.get("ignore_index", -100),
        label_smoothing=attrs.get("label_smoothing", 0.0),
    )


def cummin_ref(inputs, attrs):
    v, idx = torch.cummin(inputs[0], dim=attrs.get("dim", 0))
    return v, idx


def dynamic_quant_ref(inputs, attrs):
    """torch_npu.npu_dynamic_quant: per-token min-max int8 quant."""
    import torch_npu
    return torch_npu.npu_dynamic_quant(inputs[0])


def gather_ref(inputs, attrs):
    """PyTorch torch.gather 语义（对齐 golden + cases.yaml 数据形态）。

    cases.yaml 中 idx 与 x 同维度数（PyTorch torch.gather 的输入约束），
    输出 shape == idx.shape，按 dim 维逐元素索引。

    NPU 路径：torch.gather 通过 aten dispatcher 落到 CANN 原生 GatherElementsV2，
    实测 kernel 序列：Transpose × N + Cast(int64→int32) + GatherElementsV2 + Transpose。

    注：torch_npu.npu_gather_sparse_index 是 TF tf.gather 风格
    (output = idx.shape + x.shape[1:])，与本算子语义不符，故不使用。
    """
    x, idx = inputs[0], inputs[1]
    # 兼容旧 batch_dims 字段名（PyTorch 化前的 cases.yaml）
    dim = attrs.get("dim", attrs.get("batch_dims", 0))
    # 不做 .long()：PyTorch 2.1+ 的 torch.gather 已接受任意整型 idx；
    # 在 NPU 上 .long() 会触发 int32→int64→int32 两次冗余 Cast。
    return torch.gather(x, dim, idx)


def gcd_ref(inputs, attrs):
    return torch.gcd(inputs[0], inputs[1])


def grid_sampler_3d_ref(inputs, attrs):
    return torch.nn.functional.grid_sample(
        inputs[0], inputs[1],
        mode=attrs.get("interpolation_mode", "bilinear"),
        padding_mode=attrs.get("padding_mode", "zeros"),
        align_corners=bool(attrs.get("align_corners", False)),
    )


def group_norm_ref(inputs, attrs):
    return torch.nn.functional.group_norm(
        inputs[0], num_groups=attrs.get("num_groups"),
        weight=inputs[1], bias=inputs[2], eps=attrs.get("epsilon", 1e-5),
    )


def maximum_ref(inputs, attrs):
    return torch.maximum(inputs[0], inputs[1])


def resize_bilinear_ref(inputs, attrs):
    sf = attrs.get("scale_factor", None)
    out_size = attrs.get("output_size", None)
    return torch.nn.functional.interpolate(
        inputs[0], size=out_size, scale_factor=sf,
        mode="bilinear", align_corners=bool(attrs.get("align_corners", False)),
    )


def rms_norm_ref(inputs, attrs):
    """torch_npu.npu_rms_norm(x, gamma, epsilon=...) returns (rstd, y) — order varies."""
    import torch_npu
    x, gamma = inputs[0], inputs[1]
    epsilon = attrs.get("epsilon", 1e-6)
    res = torch_npu.npu_rms_norm(x, gamma, epsilon=epsilon)
    # npu_rms_norm returns (y, rstd); we just want y
    if isinstance(res, (tuple, list)):
        return res[0]
    return res


def scatter_ref(inputs, attrs):
    """torch.scatter (out-of-place via clone).

    Mirrors tasks/level2/scatter/golden.py reduce-mode mapping:
      - None / 'update' → Tensor.scatter_
      - 'add'           → Tensor.scatter_add_
      - 'multiply'      → Tensor.scatter_reduce_(reduce='prod', include_self=True)
      - 'amin' / 'amax' → Tensor.scatter_reduce_(reduce=..., include_self=True)
    (PyTorch's `Tensor.scatter_(reduce=...)` only supports 'add'/'multiply' and
    is being deprecated; for amin/amax must go through scatter_reduce_.)
    """
    data, idx, upd = inputs[0], inputs[1], inputs[2]
    dim = attrs.get("dim", 0)
    reduce = attrs.get("reduce")
    out = data.clone()
    idx = idx.long()
    if reduce in (None, "", "none", "update"):
        return out.scatter_(dim, idx, upd)
    if reduce == "add":
        return out.scatter_add_(dim, idx, upd)
    if reduce == "multiply":
        return out.scatter_reduce_(dim, idx, upd, reduce="prod", include_self=True)
    if reduce in ("amin", "amax"):
        return out.scatter_reduce_(dim, idx, upd, reduce=reduce, include_self=True)
    raise ValueError(f"unknown reduce mode: {reduce!r}")


def softmax_ref(inputs, attrs):
    return torch.nn.functional.softmax(inputs[0], dim=attrs.get("dim", -1))


def unsorted_segment_sum_ref(inputs, attrs):
    """No torch_npu fused API. Composite via torch.zeros + index_add_.

    `num_segments` is required (proto.yaml), so read directly. A `.get` with a
    `segment_ids.max().item()` fallback would always evaluate the default and
    add a stray ReduceMax + device sync to every timed call.
    """
    data, segment_ids = inputs[0], inputs[1]
    num_segments = int(attrs["num_segments"])
    out_shape = (num_segments,) + tuple(data.shape[segment_ids.dim():])
    out = torch.zeros(out_shape, dtype=data.dtype, device=data.device)
    out.index_add_(0, segment_ids.flatten().long(), data.reshape(-1, *data.shape[segment_ids.dim():]))
    return out


REGISTRY = {
    "level2/apply_adam_w": apply_adam_w_ref,
    "level2/apply_rotary_pos_emb": apply_rotary_pos_emb_ref,
    "level2/arg_max": arg_max_ref,
    "level2/cross_entropy_loss": cross_entropy_loss_ref,
    "level2/cummin": cummin_ref,
    "level2/dynamic_quant": dynamic_quant_ref,
    "level2/gather": gather_ref,
    "level2/gcd": gcd_ref,
    "level2/grid_sampler_3d": grid_sampler_3d_ref,
    "level2/group_norm": group_norm_ref,
    "level2/maximum": maximum_ref,
    "level2/resize_bilinear": resize_bilinear_ref,
    "level2/rms_norm": rms_norm_ref,
    "level2/scatter": scatter_ref,
    "level2/softmax": softmax_ref,
    "level2/unsorted_segment_sum": unsorted_segment_sum_ref,
}
