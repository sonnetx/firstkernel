# firstkernel

A drop-in faster fused-MoE layer for vLLM on H100. The layer runs in two
kernel launches instead of five: the gate/up GEMM applies SwiGLU in its fp32
epilogue and writes only the activated intermediate, and the down GEMM folds
the router weight and top-k reduction into relaxed bf16 atomic accumulation.
At 1,024 tokens and above, both kernels load expert weights through Hopper TMA
descriptors.

Measured on Qwen3-30B-A3B (bf16, one H100 80GB) against stock vLLM 0.28.0,
paired in one process on identical inputs: 1.037x at the 64-token decode
shape, 1.216x at the 16,384-token prefill shape, monotone in between. In the
full engine with CUDA graphs, prefill throughput improved 11.6%, mixed 2.7%,
and decode about 2%. The measurement methodology behind these numbers, which
matters more than usual, is documented in the repository's study.

## Use it

```bash
pip install -e .          # from the repository root
python -m firstkernel.selftest   # verify on your GPU first (about a minute)
```

As a library call, on bf16 tensors shaped like vLLM's MoE layer:

```python
from firstkernel import moe
out = moe(hidden_states, w1, w2, topk_weights, topk_ids)
# hidden_states (M, K) bf16, w1 (E, 2I, K), w2 (E, K, I),
# topk_weights/topk_ids (M, top_k)
```

Inside vLLM, patch before the engine is built and keep the engine in-process
so the patch reaches the worker:

```python
import os
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
import vllm, firstkernel
firstkernel.install()
llm = vllm.LLM(model="Qwen/Qwen3-30B-A3B", ...)
```

`install()` intercepts vLLM's `TritonExperts.apply` for the unquantized bf16
SwiGLU case with no LoRA and no expert map, and falls through to stock for
everything else, so it is safe to install unconditionally.

## Requirements and limitations

- Hopper (sm90). The TMA path needs it, and nothing below sm90 was tested.
- bf16 unquantized MoE with SwiGLU. Quantized paths fall through to stock.
- vLLM 0.28.x for `install()`. The bare `moe()` call needs vLLM only for its
  `moe_align_block_size` op.
- Output is not bitwise deterministic run to run. Atomic accumulation reorders
  additions; the measured spread is a few bf16 ulps, roughly 700x below the
  correctness tolerance, and greedy decode outputs matched stock token for
  token over 2,048 steps. Do not use it where bitwise reproducibility is a
  requirement (e.g. vLLM's batch-invariant mode).
- The bundled launch-config table is tuned for E=128, N=768 on H100 80GB
  (Qwen3-30B-A3B). Other expert shapes run on the heuristic fallback.

## Hack it

The kernel is one file, `firstkernel/kernel.py`, about 650 lines of Triton
with the launch logic beside it. The three mechanisms are independent and
separately deletable: `fused_gateup_silu_kernel` (activation fusion),
`moe_down_gemm_atomic_kernel` (reduction fusion), and the `*_tma` variants of
each (TMA weight loads), dispatched in `moe()` by token count. Launch configs
come from a JSON table; point `FK_MOE_CONFIG` at your own table to re-tune for
a different expert shape or GPU, and the repository's `src/tune_moe.py` shows
a two-stage search that avoids selecting its own measurement noise (rank
cheaply, then re-measure finalists before choosing, or your table will
overstate itself).

For any change, run `python -m firstkernel.selftest` first and last. It
checks against an fp32 reference implementation of the MoE definition across
every internal dispatch boundary, and it exists because this project's most
expensive bugs were kernels that were fast, plausible, and wrong. The
repository's tournament journals record which optimization ideas already
failed and why, which may save you the ten submissions they cost us.
