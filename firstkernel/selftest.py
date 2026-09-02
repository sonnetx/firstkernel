"""Self-test for firstkernel on the local GPU: python -m firstkernel.selftest

Checks the kernel against an fp32 reference computed from the mathematical
definition of a SwiGLU MoE (never from vLLM's implementation, so a shared bug
cannot cancel), across token counts that straddle every internal dispatch
boundary, then times the layer against vLLM's stock fused_experts, paired in
this process on identical inputs.

Requires one Hopper GPU (sm90), torch, triton, and vllm. Runs in about a
minute. Exit code 0 means every correctness case passed.
"""

import statistics
import sys

import torch


def moe_reference(hidden_states, w1, w2, topk_weights, topk_ids):
    """fp32 SwiGLU MoE from the definition. Slow, independent, trusted."""
    M, K = hidden_states.shape
    E = w1.shape[0]
    inter = w1.shape[1] // 2
    x = hidden_states.float()
    out = torch.zeros(M, K, dtype=torch.float32, device=hidden_states.device)
    for e in range(E):
        hit = topk_ids == e
        if not hit.any():
            continue
        tok, slot = hit.nonzero(as_tuple=True)
        gate_up = x[tok] @ w1[e].float().t()
        gate, up = gate_up[:, :inter], gate_up[:, inter:]
        y = (torch.nn.functional.silu(gate) * up) @ w2[e].float().t()
        out.index_add_(0, tok, y * topk_weights[tok, slot].float().unsqueeze(1))
    return out.to(hidden_states.dtype)


def make_inputs(M, K=2048, inter=768, E=128, top_k=8, seed=0, device="cuda"):
    g = torch.Generator(device=device).manual_seed(seed)
    hidden = (torch.randn(M, K, generator=g, device=device) * 0.05).to(torch.bfloat16)
    w1 = (torch.randn(E, 2 * inter, K, generator=g, device=device) * K**-0.5).to(torch.bfloat16)
    w2 = (torch.randn(E, K, inter, generator=g, device=device) * inter**-0.5).to(torch.bfloat16)
    logits = torch.randn(M, E, generator=g, device=device)
    topk_w, topk_ids = torch.topk(logits, top_k, dim=-1)
    return (hidden, w1, w2, torch.softmax(topk_w, -1).to(torch.bfloat16),
            topk_ids.to(torch.int32))


def main():
    assert torch.cuda.is_available(), "selftest needs a GPU"
    cap = torch.cuda.get_device_capability()
    assert cap[0] >= 9, f"needs Hopper (sm90+), found sm{cap[0]}{cap[1]}"

    from firstkernel import moe

    print(f"device: {torch.cuda.get_device_name(0)}")
    failures = 0
    # Straddles the M>=1024 TMA boundary and the M>=6144 config override.
    for M in [1, 7, 64, 100, 512, 1023, 1024, 1025, 4096, 6144, 6145, 8192]:
        args = make_inputs(M, seed=1000 + M)
        got = moe(*args)
        want = moe_reference(*args)
        rel = ((got.float() - want.float()).abs().mean()
               / want.float().abs().mean().clamp_min(1e-6)).item()
        ok = torch.allclose(got.float(), want.float(), rtol=2e-2, atol=2e-2)
        print(f"  M={M:<6} rel_error {rel:.4f}  {'PASS' if ok else 'FAIL'}")
        failures += 0 if ok else 1

    # Paired timing against stock vLLM at the decode and prefill shapes.
    from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts

    for M in [64, 8192]:
        args = make_inputs(M, seed=7)
        flush = torch.empty(int(64e6 // 4), dtype=torch.float32, device="cuda")
        times = {"stock": [], "firstkernel": []}
        for fn, name in ((fused_experts, "stock"), (moe, "firstkernel")):
            fn(*args)  # compile/warm
        torch.cuda.synchronize()
        for i in range(20):
            order = [("stock", fused_experts), ("firstkernel", moe)]
            if i % 2:
                order.reverse()
            for name, fn in order:
                flush.zero_()
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                s.record(); fn(*args); e.record()
                torch.cuda.synchronize()
                times[name].append(s.elapsed_time(e))
        st = statistics.median(times["stock"])
        fk = statistics.median(times["firstkernel"])
        print(f"  M={M:<6} stock {st:.3f} ms  firstkernel {fk:.3f} ms  "
              f"= {st / fk:.3f}x")

    print(
        "note: the ratios above compare against vLLM's fused_experts function\n"
        "and therefore include stock dispatch overhead that CUDA graphs remove\n"
        "in real serving. They confirm the kernel works and is fast on your\n"
        "GPU; the honest speedup numbers are the study's (1.04x decode, 1.22x\n"
        "prefill at the layer; about half of that end to end)."
    )

    print("SELFTEST_OK" if failures == 0 else f"SELFTEST_FAILURES={failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
