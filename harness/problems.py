"""Build MoE benchmark inputs. Owned by the harness, not by candidates.

Inputs are generated from a seed the candidate never sees, and the router
assignment is drawn from a realistic distribution rather than uniform noise:
real routers are imbalanced, so a kernel tuned against a perfectly balanced
assignment would be tuned against the wrong problem.
"""

import json
import os

import torch


def load_problem_set(path: str):
    with open(path) as f:
        return json.load(f)


def default_problem_set(path: str = "results/shapes.json"):
    if os.path.exists(path):
        return load_problem_set(path)
    raise FileNotFoundError(
        f"{path} not found; run src/capture_shapes.py on a GPU node first"
    )


def make_inputs(
    M: int,
    K: int,
    inter: int,
    E: int,
    top_k: int,
    seed: int,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    imbalance: float = 1.0,
):
    """Return (hidden_states, w1, w2, topk_weights, topk_ids).

    imbalance controls how skewed expert routing is. 0 is uniform; larger
    values concentrate tokens on fewer experts, which is what real routers do
    and what makes the padded-block path in the kernel matter.
    """
    g = torch.Generator(device=device).manual_seed(seed)

    hidden = torch.randn(M, K, generator=g, device=device, dtype=torch.float32)
    hidden = (hidden * 0.05).to(dtype)

    # Weight scale ~ 1/sqrt(fan_in), so activations stay in a realistic range
    # and correctness tolerances mean something.
    w1 = (
        torch.randn(E, 2 * inter, K, generator=g, device=device, dtype=torch.float32)
        * (K ** -0.5)
    ).to(dtype)
    w2 = (
        torch.randn(E, K, inter, generator=g, device=device, dtype=torch.float32)
        * (inter ** -0.5)
    ).to(dtype)

    # Router logits with a skewed prior over experts.
    prior = torch.randn(E, generator=g, device=device, dtype=torch.float32) * imbalance
    logits = torch.randn(M, E, generator=g, device=device, dtype=torch.float32) + prior
    topk_weights, topk_ids = torch.topk(logits, top_k, dim=-1)
    topk_weights = torch.softmax(topk_weights, dim=-1).to(dtype)
    topk_ids = topk_ids.to(torch.int32)

    return hidden, w1, w2, topk_weights, topk_ids
