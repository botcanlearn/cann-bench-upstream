#!/usr/bin/python3
# coding=utf-8

from typing import Any, Dict, Optional, Tuple, Union

import torch


def all_gather_matmul_v2(
    x1: torch.Tensor,
    x2: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    x1_scale: Optional[torch.Tensor] = None,
    x2_scale: Optional[torch.Tensor] = None,
    hcomm_info: str = "",
    world_size: int = 1,
    gather_output: bool = True,
    is_trans_b: bool = False,
    out_dtype: str = "float16",
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Single-process reference for metadata and smoke checks."""
    del hcomm_info
    x2_eff = x2.t() if is_trans_b else x2
    gathered = torch.cat([x1] * int(world_size), dim=0).float()
    out = torch.matmul(gathered, x2_eff.float())
    if x1_scale is not None:
        scale = torch.cat([x1_scale] * int(world_size), dim=0).float().reshape(-1, 1)
        out = out * scale
    if x2_scale is not None:
        out = out * x2_scale.float().reshape(1, -1)
    if bias is not None:
        out = out + bias.float()
    out = out.to(_torch_dtype(out_dtype))
    if gather_output:
        return out, gathered.to(x1.dtype)
    return out


def mc2_make_rank_inputs(ctx: Dict[str, Any], case_payload: Dict[str, Any]) -> Dict[str, Any]:
    device = ctx["device"]
    shapes = case_payload["input_shapes"]
    dtypes = case_payload["dtypes"]
    ranges = case_payload["value_ranges"]
    attrs = case_payload["attrs"]
    seed = int(attrs.get("seed", 1))
    weight_same = int(attrs.get("weight_same", 1))
    rank = int(ctx["rank"])

    x1 = _make_tensor(shapes[0], dtypes[0], ranges[0], seed, device)
    x2_seed = seed + rank * 2 if weight_same else seed
    x2 = _make_tensor(shapes[1], dtypes[1], ranges[1], x2_seed, device)

    bias = None
    if bool(attrs.get("is_bias", False)) and len(shapes) > 2 and shapes[2] is not None:
        bias = _make_tensor(shapes[2], dtypes[2], ranges[2], rank * 3, device)

    x1_scale_index = 3 if bool(attrs.get("is_bias", False)) else 2
    x2_scale_index = x1_scale_index + 1
    x1_scale = _make_tensor(
        shapes[x1_scale_index],
        dtypes[x1_scale_index],
        ranges[x1_scale_index],
        rank * 11,
        device,
    )
    x2_scale = _make_tensor(
        shapes[x2_scale_index],
        dtypes[x2_scale_index],
        ranges[x2_scale_index],
        rank * 11,
        device,
    )
    return {"x1": x1, "x2": x2, "bias": bias, "x1_scale": x1_scale, "x2_scale": x2_scale}


def mc2_call_candidate(candidate, ctx: Dict[str, Any], inputs: Dict[str, Any], attrs: Dict[str, Any]):
    return candidate(
        inputs["x1"],
        inputs["x2"],
        ctx["hcomm_info"],
        int(ctx["world_size"]),
        bias=inputs.get("bias"),
        x1_scale=inputs.get("x1_scale"),
        x2_scale=inputs.get("x2_scale"),
        gather_output=bool(attrs.get("gather_output", True)),
        is_trans_b=bool(attrs.get("is_trans_b", False)),
        out_dtype=str(attrs.get("out_dtype", "float16")),
        x1_dtype=str(attrs.get("x1_dtype", "int8")),
        x2_dtype=str(attrs.get("x2_dtype", "int8")),
        x1_scale_dtype=str(attrs.get("x1_scale_dtype", "float32")),
        x2_scale_dtype=str(attrs.get("x2_scale_dtype", "float32")),
        block_size=int(attrs.get("block_size", 128)),
        group_sizes=list(attrs.get("group_sizes", [])),
    )


def mc2_distributed_golden(ctx: Dict[str, Any], inputs: Dict[str, Any], attrs: Dict[str, Any]):
    """Distributed golden following mc2_test's hccl_mm reference path."""
    torch_npu = ctx["torch_npu"]
    dist = ctx["dist"]
    world_size = int(ctx["world_size"])
    gather_shape = [inputs["x1"].shape[0] * world_size] + list(inputs["x1"].shape[1:])
    gathered = torch.empty(gather_shape, dtype=inputs["x1"].dtype, device=inputs["x1"].device)
    dist._all_gather_base(gathered, inputs["x1"])
    gathered_x1_scale = inputs.get("x1_scale")
    if gathered_x1_scale is not None:
        scale_shape = [gathered_x1_scale.shape[0] * world_size] + list(gathered_x1_scale.shape[1:])
        gathered_scale = torch.empty(scale_shape, dtype=gathered_x1_scale.dtype, device=gathered_x1_scale.device)
        dist._all_gather_base(gathered_scale, gathered_x1_scale)
        gathered_x1_scale = gathered_scale
    x2_eff = inputs["x2"].t() if bool(attrs.get("is_trans_b", False)) else inputs["x2"]
    out_dtype = _torch_dtype(str(attrs.get("out_dtype", "float16")))
    if str(attrs.get("x1_dtype", "int8")).lower() in ("fp16", "float16", "bf16", "bfloat16"):
        output = torch.matmul(gathered, x2_eff)
        if inputs.get("bias") is not None:
            output = output + inputs["bias"]
    else:
        output = torch_npu.npu_quant_matmul(
            gathered,
            x2_eff,
            inputs.get("x2_scale"),
            offset=None,
            pertoken_scale=gathered_x1_scale,
            bias=inputs.get("bias"),
            output_dtype=out_dtype,
            group_sizes=list(attrs.get("group_sizes", [])),
        )
    if bool(attrs.get("gather_output", True)):
        return output, gathered
    return output


def _make_tensor(shape, dtype_name: str, value_range, seed: int, device):
    torch.manual_seed(int(seed))
    dtype = _torch_dtype(dtype_name)
    lo, hi = value_range if value_range is not None else [0, 1]
    if dtype.is_floating_point:
        tensor = torch.empty(shape, dtype=torch.float32).uniform_(float(lo), float(hi)).to(dtype)
    else:
        tensor = torch.randint(int(lo), int(hi) + 1, shape, dtype=dtype)
    return tensor.to(device)


def _torch_dtype(dtype_name: str):
    dtype = str(dtype_name).lower()
    mapping = {
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
        "float32": torch.float32,
        "int8": torch.int8,
        "int32": torch.int32,
        "int64": torch.int64,
    }
    if dtype not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    return mapping[dtype]
