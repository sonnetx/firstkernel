"""Baseline candidate: stock vLLM, unmodified.

Every optimization is measured against this. It calls vLLM's own
fused_experts, so the baseline is exactly what a user of the released engine
gets today, not a reimplementation that might be accidentally slower.
"""

import torch
from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts


def moe(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    return fused_experts(hidden_states, w1, w2, topk_weights, topk_ids)
