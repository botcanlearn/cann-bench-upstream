#!/usr/bin/python3
# coding=utf-8

import torch


def tabular_feature_tokenizer(
    x_num: torch.Tensor,
    x_cat: torch.Tensor,
    num_weight: torch.Tensor,
    cat_table: torch.Tensor,
    bias: torch.Tensor = None,
    unknown_id: int = 0,
) -> torch.Tensor:
    if x_num.dim() != 2 or x_cat.dim() != 2:
        raise ValueError("x_num and x_cat must be 2D")
    if num_weight.dim() != 2 or cat_table.dim() != 2:
        raise ValueError("num_weight and cat_table must be 2D")
    if x_num.shape[1] != num_weight.shape[0]:
        raise ValueError("num_weight first dim must match number of numerical features")
    if not (0 <= unknown_id < cat_table.shape[0]):
        raise ValueError("unknown_id out of cat_table range")

    out_dtype = x_num.dtype
    num_tokens = x_num.float().unsqueeze(-1) * num_weight.float().unsqueeze(0)
    if bias is not None:
        num_tokens = num_tokens + bias.float().unsqueeze(0)

    ids = x_cat.to(torch.long)
    valid = (ids >= 0) & (ids < cat_table.shape[0])
    ids = torch.where(valid, ids, torch.full_like(ids, int(unknown_id)))
    cat_tokens = cat_table.float().index_select(0, ids.reshape(-1))
    cat_tokens = cat_tokens.reshape(*ids.shape, cat_table.shape[1])

    return torch.cat([num_tokens, cat_tokens], dim=1).to(out_dtype)
