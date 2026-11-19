# firstkernel

A faster fused mixture-of-experts layer for vLLM on H100, produced by coding
agents under a frozen evaluation protocol and verified adversarially. The
layer runs in two kernel launches instead of five: SwiGLU fuses into the
gate/up GEMM's fp32 epilogue, the router-weighted top-k reduction folds into
the down GEMM through relaxed bf16 atomics, and both GEMMs load expert weights
through Hopper TMA descriptors at 1,024 tokens and above.

Measured on Qwen3-30B-A3B (bf16, one H100 80GB) against stock vLLM 0.28.0 on
held-out shapes with entropy-drawn seeds: **1.037x at the 64-token decode
shape, 1.216x at the 16,384-token prefill shape**. Integrated into the engine
with CUDA graphs on: **prefill throughput +11.6%, mixed +2.7%, decode ~+2%**,
with outputs token-identical to stock across 2,048 greedy decode steps.
Roughly half of a kernel-level win survives the engine. The short public
writeup is <https://sonnetx.github.io/firstkernel/>, and the study in
[docs/article.md](docs/article.md) is about why the other half of most
published kernel numbers was never there.

## Quick start

```bash
pip install -e .
python -m firstkernel.selftest    # verify on your GPU, about a minute
```

Use inside vLLM (0.28.x), patching before the engine is built:

```python
import os
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
import vllm, firstkernel
firstkernel.install()
llm = vllm.LLM(model="Qwen/Qwen3-30B-A3B")
```

Or call the layer directly on vLLM-shaped tensors:

```python
from firstkernel import moe
out = moe(hidden_states, w1, w2, topk_weights, topk_ids)
```

Requirements and limitations (the honest list): Hopper sm90 only, bf16
unquantized SwiGLU MoE only (everything else falls through to stock),
non-bitwise-deterministic output from the atomic reduction (measured at a few
bf16 ulps, ~700x under tolerance), and the bundled launch-config table is
tuned for E=128, N=768. Details in
[firstkernel/README.md](firstkernel/README.md).

## Hack it

The kernel is one file, [firstkernel/kernel.py](firstkernel/kernel.py), and
the three mechanisms are independent and separately deletable. The repo ships
the measurement machinery that kept this project honest, because a kernel
change without trustworthy measurement is worse than no change:

- `harness/bench.py` benchmarks any candidate file exposing `moe(...)`, paired
  in-process against a reference, gated by correctness against an fp32
  implementation of the MoE definition, input sensitivity, aliasing, and an
  HBM-bandwidth impossibility floor. Read
  [harness/README.md](harness/README.md) before trusting any number,
  including ours.
- `kernels/standalone.py` is the overhead-matched reference to compare
  against; `kernels/baseline.py` (stock vLLM path) includes ~52us of dispatch
  overhead per call that flatters candidates.
- `src/tune_moe.py` re-tunes the launch-config table for a different expert
  shape or GPU with a two-stage search that re-measures finalists, because a
  one-stage search selects its own noise (measured: it overstated itself at
  15 of 15 shapes). Point `FK_MOE_CONFIG` at your table.
- [PROTOCOL.md](PROTOCOL.md) is the frozen evaluation protocol the results
  were produced under, and [results/MEASUREMENT.md](results/MEASUREMENT.md)
  is the retraction that taught us to write it.

A typical hacking loop:

```bash
cp firstkernel/kernel.py my_kernel.py        # edit freely
printf 'from my_kernel import moe\n' > kernels/mine.py
python3 harness/bench.py --candidate kernels/mine.py \
    --compare kernels/standalone.py --set dev
python -m firstkernel.selftest               # before and after
```

The full research record, including the four agent tournament journals with
every failed idea and why it failed, lives in the companion research
repository referenced by the study.

## Provenance

The kernel was produced by a four-branch agent tournament plus a composition
agent (six agent runs, ~1.1M output tokens, ~9 H100-hours), survived 41
adversarial verification checks, and its byte-identical research artifact is
frozen in the companion repository. This package changes only how the config
table is located.

Apache-2.0.
