#!/usr/bin/python3
# coding=utf-8

import torch


def ctc_greedy_decode(
    logits: torch.Tensor,
    input_lengths: torch.Tensor,
    blank_id: int = 0,
    merge_repeated: bool = True,
    pad_id: int = -1,
):
    if logits.dim() != 3:
        raise ValueError("logits must be [B, T, V]")
    if input_lengths.dim() != 1 or input_lengths.shape[0] != logits.shape[0]:
        raise ValueError("input_lengths must be [B]")
    batch, steps, vocab = logits.shape
    if blank_id < 0 or blank_id >= vocab:
        raise ValueError("blank_id must be in [0, V)")

    pred = torch.argmax(logits.float(), dim=-1).cpu()
    lengths_in = torch.clamp(input_lengths.to(torch.long).cpu(), min=0, max=steps)
    sequences = []
    max_out = 0
    for b in range(batch):
        seq = []
        prev = None
        for token in pred[b, :int(lengths_in[b].item())].tolist():
            if token == blank_id:
                prev = token
                continue
            if merge_repeated and prev == token:
                prev = token
                continue
            seq.append(int(token))
            prev = token
        sequences.append(seq)
        max_out = max(max_out, len(seq))

    max_out = max(max_out, 1)
    tokens = torch.full((batch, max_out), int(pad_id), dtype=torch.int32, device=logits.device)
    lengths = torch.empty((batch,), dtype=torch.int32, device=logits.device)
    for b, seq in enumerate(sequences):
        lengths[b] = len(seq)
        if seq:
            tokens[b, :len(seq)] = torch.tensor(seq, dtype=torch.int32, device=logits.device)
    return tokens, lengths
