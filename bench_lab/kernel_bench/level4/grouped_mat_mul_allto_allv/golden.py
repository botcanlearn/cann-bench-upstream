#!/usr/bin/python3
# coding=utf-8

from typing import Any, Dict, List

import torch


def grouped_mat_mul_allto_allv(
    gmm_x,
    gmm_weight,
    hcomm_info: str = "",
    world_size: int = 1,
    send_counts: List[int] = None,
    recv_counts: List[int] = None,
    mm_x=None,
    mm_weight=None,
    trans_gmm_weight: bool = False,
    trans_mm_weight: bool = False,
):
    """Single-process smoke golden.

    Distributed cases use ``mc2_distributed_golden`` because the real semantics
    include all-to-all communication.
    """
    del hcomm_info, world_size, send_counts, recv_counts
    weight = gmm_weight.transpose(1, 2) if trans_gmm_weight else gmm_weight
    y = torch.matmul(gmm_x.float(), weight[0].float())
    mm_y = None
    if mm_x is not None and mm_weight is not None:
        w = mm_weight.t() if trans_mm_weight else mm_weight
        mm_y = torch.matmul(mm_x.float(), w.float())
    return y, mm_y


def mc2_make_rank_inputs(ctx: Dict[str, Any], case_payload: Dict[str, Any]) -> Dict[str, Any]:
    rank = int(ctx["rank"])
    device = ctx["device"]
    attrs = case_payload["attrs"]
    shapes = case_payload["input_shapes"]
    dtypes = case_payload["dtypes"]
    ranges = case_payload["value_ranges"]
    world_size = int(ctx["world_size"])
    exp_per_card = int(attrs.get("e", shapes[1][0]))
    seed = int(attrs.get("seed", 1))

    rank_rows = _generate_positive_integers(world_size, int(shapes[0][0]), seed)
    exp_token_nums, _ = _generate_gmm_alltoallv_matrix(rank_rows, exp_per_card, seed=seed)
    send_counts = _get_send_counts(exp_token_nums, rank, exp_per_card, world_size)
    recv_counts = [int(x) for x in exp_token_nums[rank]]
    input_splits = _gen_output_splits(exp_token_nums, rank, exp_per_card, world_size)
    output_splits = _gen_input_splits(exp_token_nums, rank, exp_per_card, world_size)
    group_list = _get_group_list(exp_token_nums, rank, exp_per_card, world_size)

    gmm_x_shape = [sum(send_counts), int(shapes[0][1])]
    gmm_x = _make_tensor(gmm_x_shape, dtypes[0], ranges[0], seed, device)
    gmm_weight = _make_tensor(shapes[1], dtypes[1], ranges[1], seed, device)

    mm_out = bool(attrs.get("mm_out", False))
    mm_x = _make_tensor(shapes[4], dtypes[4], ranges[4], seed, device) if mm_out else None
    mm_weight = _make_tensor(shapes[5], dtypes[5], ranges[5], seed, device) if mm_out else None

    return {
        "gmm_x": gmm_x,
        "gmm_weight": gmm_weight,
        "send_counts": send_counts,
        "recv_counts": recv_counts,
        "input_splits": input_splits,
        "output_splits": output_splits,
        "group_list": group_list,
        "exp_token_nums": exp_token_nums,
        "mm_x": mm_x,
        "mm_weight": mm_weight,
    }


def mc2_call_candidate(candidate, ctx: Dict[str, Any], inputs: Dict[str, Any], attrs: Dict[str, Any]):
    return candidate(
        inputs["gmm_x"],
        inputs["gmm_weight"],
        ctx["hcomm_info"],
        int(ctx["world_size"]),
        inputs["send_counts"],
        inputs["recv_counts"],
        mm_x=inputs.get("mm_x"),
        mm_weight=inputs.get("mm_weight"),
        trans_gmm_weight=bool(attrs.get("trans_gmm_weight", False)),
        trans_mm_weight=bool(attrs.get("trans_mm_weight", False)),
    )


def mc2_distributed_golden(ctx: Dict[str, Any], inputs: Dict[str, Any], attrs: Dict[str, Any]):
    dist = ctx["dist"]
    gmm_x = inputs["gmm_x"]
    gmm_weight = inputs["gmm_weight"]
    if bool(attrs.get("trans_gmm_weight", False)):
        gmm_weight = gmm_weight.transpose(1, 2)

    gmm_out = _grouped_matmul(gmm_x.float(), gmm_weight.float(), inputs["group_list"])
    unpermuted = _unpermute(
        gmm_out,
        int(ctx["world_size"]),
        int(attrs.get("e", gmm_weight.shape[0])),
        inputs["exp_token_nums"],
        int(ctx["rank"]),
        gmm_out.device,
    )
    y = torch.empty(
        (sum(inputs["output_splits"]), gmm_out.shape[-1]),
        dtype=torch.float32,
        device=gmm_out.device,
    )
    dist.all_to_all_single(
        y,
        input=unpermuted,
        output_split_sizes=inputs["output_splits"],
        input_split_sizes=inputs["input_splits"],
    )

    mm_y = None
    mm_x = inputs.get("mm_x")
    mm_weight = inputs.get("mm_weight")
    if mm_x is not None and mm_weight is not None:
        if bool(attrs.get("trans_mm_weight", False)):
            mm_weight = mm_weight.t()
        mm_y = torch.matmul(mm_x.float(), mm_weight.float())
    return y, mm_y


def _make_tensor(shape, dtype_name: str, value_range, seed: int, device):
    torch.manual_seed(int(seed))
    dtype_map = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "int32": torch.int32,
        "int64": torch.int64,
    }
    dtype = dtype_map.get(str(dtype_name).lower(), torch.float32)
    lo, hi = value_range if value_range is not None else [0, 1]
    if dtype.is_floating_point:
        tensor = torch.empty(shape, dtype=torch.float32).uniform_(float(lo), float(hi)).to(dtype)
    else:
        tensor = torch.randint(int(lo), int(hi) + 1, shape, dtype=dtype)
    return tensor.to(device)


def _grouped_matmul(gmm_x, gmm_weight, group_list):
    parts = torch.split(gmm_x, [int(x) for x in group_list], dim=0)
    outputs = []
    for idx, part in enumerate(parts):
        if part.numel() == 0:
            continue
        outputs.append(torch.matmul(part, gmm_weight[idx]))
    return torch.cat(outputs, dim=0) if outputs else gmm_x.new_empty((0, gmm_weight.shape[-1]))


def _unpermute(tokens, world_size: int, exp_per_card: int, exp_token_nums, rank: int, device):
    counts = torch.zeros(world_size, exp_per_card, dtype=torch.int64, device="cpu")
    for src in range(world_size):
        counts[src] = torch.tensor(
            exp_token_nums[src][rank * exp_per_card:(rank + 1) * exp_per_card],
            dtype=torch.int64,
        )
    by_expert = counts.t()
    offsets = [0] + torch.cumsum(by_expert.sum(dim=1), dim=0).long().tolist()[:-1]
    cumsums = torch.cumsum(by_expert, dim=1)
    selected = []
    for src in range(world_size):
        for expert in range(exp_per_card):
            start = 0 if src == 0 else int(cumsums[expert][src - 1])
            end = int(cumsums[expert][src])
            indices = torch.arange(
                start + offsets[expert],
                end + offsets[expert],
                dtype=torch.long,
                device=device,
            )
            selected.append(tokens.index_select(dim=0, index=indices))
    return torch.cat(selected, dim=0) if selected else tokens


def _get_send_counts(exp_token_nums, rank: int, exp_per_card: int, world_size: int) -> List[int]:
    counts = []
    for src in range(world_size):
        counts.extend(int(x) for x in exp_token_nums[src][rank * exp_per_card:(rank + 1) * exp_per_card])
    return counts


def _gen_input_splits(exp_token_nums, rank: int, exp_per_card: int, world_size: int) -> List[int]:
    row = exp_token_nums[rank]
    return [sum(int(x) for x in row[i * exp_per_card:(i + 1) * exp_per_card]) for i in range(world_size)]


def _gen_output_splits(exp_token_nums, rank: int, exp_per_card: int, world_size: int) -> List[int]:
    return [
        sum(int(x) for x in exp_token_nums[i][rank * exp_per_card:(rank + 1) * exp_per_card])
        for i in range(world_size)
    ]


def _get_group_list(exp_token_nums, rank: int, exp_per_card: int, world_size: int) -> List[int]:
    counts = torch.zeros(world_size, exp_per_card, dtype=torch.int64, device="cpu")
    for src in range(world_size):
        counts[src] = torch.tensor(
            exp_token_nums[src][rank * exp_per_card:(rank + 1) * exp_per_card],
            dtype=torch.int64,
        )
    return counts.sum(dim=0).long().tolist()


def _generate_positive_integers(n: int, sum_total: int, seed: int):
    if sum_total % n != 0:
        raise ValueError(f"sum_total={sum_total} must be divisible by n={n}")
    quotient = sum_total // n
    if quotient < n:
        raise ValueError(f"sum_total must satisfy sum_total >= n^2, got {sum_total}")
    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)
    remain = quotient - n
    if remain == 0:
        return torch.full((n,), n, dtype=torch.int64)
    probs = torch.ones(n, dtype=torch.float32) / n
    samples = torch.multinomial(probs, num_samples=remain, replacement=True, generator=rng)
    counts = torch.bincount(samples, minlength=n).long()
    return (counts + 1) * n


def _generate_gmm_alltoallv_matrix(a_array, exp_per_card: int, recv_count_zero_rank_list=None, seed: int = 1):
    del recv_count_zero_rank_list
    world_size = len(a_array)
    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)
    k_values = [int(a) // world_size for a in a_array]
    blocks = []
    special = any(k < exp_per_card or int(a) % world_size != 0 for a, k in zip(a_array, k_values))
    probs = torch.ones(exp_per_card, dtype=torch.float32) / exp_per_card
    for k in k_values:
        block = torch.zeros(exp_per_card, world_size, dtype=torch.int64)
        for col in range(world_size):
            if special:
                n_samples = max(k, 0)
                if n_samples == 0:
                    block[:, col] = 0
                else:
                    samples = torch.multinomial(probs, num_samples=n_samples, replacement=True, generator=rng)
                    block[:, col] = torch.bincount(samples, minlength=exp_per_card).long()
            elif k == 0:
                block[:, col] = 1
            else:
                n_samples = k - exp_per_card
                samples = torch.multinomial(probs, num_samples=n_samples, replacement=True, generator=rng)
                block[:, col] = torch.bincount(samples, minlength=exp_per_card).long() + 1
        blocks.append(block)
    matrix = torch.cat(blocks, dim=0)
    return matrix.t().tolist(), matrix.t().to(torch.int64)
