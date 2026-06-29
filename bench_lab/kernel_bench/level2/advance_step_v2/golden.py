#!/usr/bin/python3
# coding=utf-8
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
# Adapted from vllm-project/vllm/vllm/worker/gpu_model_runner.py
#
import torch

def advance_step_v2(num_seqs: int, num_queries: int, block_size: int,
                    input_tokens: torch.Tensor, sampled_token_ids: torch.Tensor,
                    input_positions: torch.Tensor, seq_lens: torch.Tensor,
                    slot_mapping: torch.Tensor, block_tables: torch.Tensor,
                    spec_token: torch.Tensor, accepted_num: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """AdvanceStepV2 golden implementation for vLLM inference step update.

    Args:
        num_seqs: Number of sequences.
        num_queries: Number of queries.
        block_size: Block size for page attention.
        input_tokens: Input token tensor.
        sampled_token_ids: Sampled token ids tensor.
        input_positions: Input positions tensor.
        seq_lens: Sequence lengths tensor.
        slot_mapping: Slot mapping tensor.
        block_tables: Block tables tensor.
        spec_token: Speculative token tensor.
        accepted_num: Accepted number tensor.

    Returns:
        Updated (input_tokens, input_positions, seq_lens, slot_mapping).
    """
    # Clone in-place tensors to avoid mutating the caller's inputs.
    # The evaluator may share input tensors between golden and AI runs,
    # so in-place updates here would otherwise pollute the AI inputs.
    input_tokens = input_tokens.clone()
    input_positions = input_positions.clone()
    seq_lens = seq_lens.clone()
    slot_mapping = slot_mapping.clone()

    token_each_reqs = 1 + len(spec_token[0])
    input_positions += torch.repeat_interleave(accepted_num, token_each_reqs) + 1
    seq_lens.copy_((input_positions + 1).to(seq_lens.dtype))
    index = torch.argmin(
        torch.cat([
            sampled_token_ids,
            torch.full((num_seqs, 1), -1, device=sampled_token_ids.device)
        ], dim=1),
        dim=1
    ) - 1
    last_tokens = sampled_token_ids[torch.arange(num_seqs), index]
    if token_each_reqs == 1:
        input_tokens[:num_seqs] = last_tokens.to(dtype=input_tokens.dtype)
    else:
        input_tokens_2d = input_tokens.view(-1, token_each_reqs)
        input_tokens_2d[:num_seqs, 0] = last_tokens
        input_tokens_2d[:num_seqs, 1:] = spec_token
    req_indices = torch.repeat_interleave(
        torch.arange(num_seqs),
        token_each_reqs,
        dim=0
    )
    max_num_blocks_per_req = block_tables.shape[1]
    block_tables_indices = (
        req_indices * max_num_blocks_per_req +
        input_positions // block_size
    )
    block_numbers = block_tables.flatten()[block_tables_indices]
    block_offset = input_positions % block_size
    slot_mapping.copy_(block_numbers * block_size + block_offset)
    return input_tokens, input_positions, seq_lens, slot_mapping
