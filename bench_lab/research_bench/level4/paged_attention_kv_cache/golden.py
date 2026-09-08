#!/usr/bin/python3
# coding=utf-8

import torch


def _reconstruct_cache(cache: torch.Tensor, cache_seqlen: int, page_table_row: torch.Tensor) -> torch.Tensor:
    page_block_size = cache.shape[1]
    num_blocks = cache.shape[0]
    table_width = page_table_row.numel()
    max_len = table_width * page_block_size
    seq_len = max(1, min(int(cache_seqlen), max_len))
    blocks_needed = (seq_len + page_block_size - 1) // page_block_size

    parts = []
    for block_idx in range(blocks_needed):
        physical_block = int(page_table_row[block_idx].item()) % num_blocks
        start = block_idx * page_block_size
        remaining = seq_len - start
        take = min(page_block_size, remaining)
        parts.append(cache[physical_block, :take])
    return torch.cat(parts, dim=0)


def paged_attention_kv_cache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cache_seqlens: torch.Tensor,
    page_table: torch.Tensor,
    causal: bool = True,
) -> torch.Tensor:
    if q.dim() != 4 or k_cache.dim() != 4 or v_cache.dim() != 4:
        raise ValueError("q, k_cache and v_cache must be 4D tensors")
    if k_cache.shape != v_cache.shape:
        raise ValueError("k_cache and v_cache must have the same shape")
    if cache_seqlens.dim() != 1 or page_table.dim() != 2:
        raise ValueError("cache_seqlens must be [B] and page_table must be [B, MaxBlocks]")

    batch, seqlen_q, nheads_q, headdim = q.shape
    if cache_seqlens.shape[0] != batch or page_table.shape[0] != batch:
        raise ValueError("cache_seqlens/page_table batch dimension must match q")
    nheads_kv = k_cache.shape[2]
    if k_cache.shape[-1] != headdim:
        raise ValueError("q and cache head dimensions must match")
    if nheads_q % nheads_kv != 0:
        raise ValueError("Hq must be divisible by Hkv")

    out_dtype = q.dtype
    scale = headdim ** -0.5
    group = nheads_q // nheads_kv
    outputs = []

    for b in range(batch):
        seq_len = int(cache_seqlens[b].item())
        if causal and seq_len < seqlen_q:
            seq_len = seqlen_q
        k = _reconstruct_cache(k_cache, seq_len, page_table[b])
        v = _reconstruct_cache(v_cache, seq_len, page_table[b])
        seq_len = k.shape[0]

        k = k.transpose(0, 1).unsqueeze(0).float()
        v = v.transpose(0, 1).unsqueeze(0).float()
        if group > 1:
            k = k.repeat_interleave(group, dim=1)
            v = v.repeat_interleave(group, dim=1)

        q_b = q[b:b + 1].transpose(1, 2).float()
        scores = torch.matmul(q_b, k.transpose(-2, -1)) * scale

        if causal:
            row_idx = torch.arange(seqlen_q, device=scores.device).unsqueeze(1)
            col_idx = torch.arange(seq_len, device=scores.device).unsqueeze(0)
            mask = col_idx > (row_idx + seq_len - seqlen_q)
            scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        out_b = torch.matmul(attn, v).transpose(1, 2).contiguous()
        outputs.append(out_b.to(out_dtype))

    return torch.cat(outputs, dim=0)
