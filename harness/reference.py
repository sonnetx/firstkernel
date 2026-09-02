"""Independent MoE reference. Never imported by candidate code.

Deliberately naive and written from the mathematical definition rather than
adapted from vLLM's implementation, so that a candidate and the reference
cannot share a bug. Accumulates in fp32 and loops experts in Python: slow, but
the benchmark only calls it once per shape to build expected output.
"""

import torch


def moe_reference(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    """SwiGLU mixture-of-experts forward pass.

    hidden_states: (M, K)
    w1: (E, 2 * I, K) fused gate and up projections
    w2: (E, K, I) down projection
    topk_weights: (M, top_k) router weights, already normalized
    topk_ids: (M, top_k) expert indices
    returns: (M, K)
    """
    M, K = hidden_states.shape
    E = w1.shape[0]
    inter = w1.shape[1] // 2

    x = hidden_states.float()
    out = torch.zeros(M, K, dtype=torch.float32, device=hidden_states.device)

    for e in range(E):
        # Which (token, slot) pairs routed to this expert.
        hit = topk_ids == e
        if not hit.any():
            continue
        tok_idx, slot_idx = hit.nonzero(as_tuple=True)

        xe = x[tok_idx]                                    # (T, K)
        gate_up = xe @ w1[e].float().t()                   # (T, 2I)
        gate, up = gate_up[:, :inter], gate_up[:, inter:]
        act = torch.nn.functional.silu(gate) * up          # (T, I)
        ye = act @ w2[e].float().t()                       # (T, K)

        weight = topk_weights[tok_idx, slot_idx].float().unsqueeze(1)
        out.index_add_(0, tok_idx, ye * weight)

    return out.to(hidden_states.dtype)
