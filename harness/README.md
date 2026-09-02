# The harness contract

The harness is the reward signal for the phase-3 agent loop. Its design
assumption is that a capable agent optimizing against a benchmark will find
whatever the benchmark actually rewards, which is not always a faster kernel.
Sakana's AI CUDA Engineer reported 50-120x speedups that turned out to be
exploits of its evaluation framework, so the checks here come first and the
timing comes second.

## What an agent may and may not touch

Agents edit files under `kernels/`. Agents never edit anything under
`harness/`, and never see the seed a run will be graded with. A candidate is a
module exposing one function:

```python
def moe(hidden_states, w1, w2, topk_weights, topk_ids) -> torch.Tensor
```

`hidden_states` is `(M, K)`, `w1` is `(E, 2I, K)` fused gate and up
projections, `w2` is `(E, K, I)`, and the router tensors are `(M, top_k)`. The
return is `(M, K)` in the input dtype.

## Checks that gate every timing number

No timing is reported for a candidate that fails any of these.

**Numerical agreement** against `reference.py`, which computes the MoE from its
mathematical definition in fp32, looping experts in Python. It is written from
the definition rather than adapted from vLLM so that a candidate and the
reference cannot share a bug.

**Input sensitivity.** The candidate runs a second time on a different input.
Byte-identical output means it is not reading its inputs, which catches a
cached or precomputed result.

**No aliasing.** Returning the input buffer is rejected.

**Shape and dtype** are checked before anything else, since a wrong-shaped
output makes the numerical comparison meaningless.

**A hardware lower bound.** Expert weights with tokens routed to them have to
cross HBM at least once, so `bytes / peak_bandwidth` is a floor no honest
kernel can beat. A candidate below it is reported as impossible rather than
fast. This is the check that catches a kernel which has quietly stopped doing
the work.

**Disjoint test sets.** PROTOCOL.md defines development, validation, and
final-test sets separated by token count, not just by seed. An earlier version
held out every third token count, which is predictable to anyone reading the
harness and therefore not a private holdout. Agents iterate on the development
set only; selection happens on validation numbers agents never see; the final
set runs once with routing seeds drawn at run time.

## Measurement

Timing uses CUDA events around each call, a median over 30 reps after 10
warmup calls, and a 64 MB buffer zeroed between reps to evict the expert
weights from L2. Without that flush the second rep onward reads weights from
cache and the measurement stops resembling serving.

The cluster job tries to lock GPU clocks, and on Sherlock it cannot: an
unprivileged user is refused. Every measurement therefore carries the power
and thermal governor's noise, so the harness reports an interquartile range
next to each median. A speedup smaller than that spread is not a result, and
the phase-3 loop should treat it as noise rather than progress.

Results are weighted by how often each token count actually ran in serving, so
the summary tracks production time rather than treating a rare shape as equal
to a common one.

## Benchmark units

The benchmark unit is a token count, meaning one MoE layer call. A layer call
launches the kernel twice at different sizes: the fused gate/up projection at
the token count with `K = hidden_size`, and the down projection at token count
times `top_k` with `K = moe_intermediate`. Only the gate/up launches carry the
token count. An earlier version of this harness grouped launches by raw size
and so benchmarked a synthetic MoE layer of width 768 that the model never
runs, which passed every correctness check because the inputs were internally
consistent. Correct-looking numbers for the wrong problem are the failure mode
this harness exists to prevent, so the derivation is spelled out in
`build_shape_list`.

Alongside the three token counts observed in serving (64, 8192, 16384), the
harness sweeps powers of two from 1 to 16384. Swept-only counts are checked
for correctness and reported, but carry zero production weight, so they build
a speedup curve over batch size without distorting the headline number.

## Why the shapes come from a hook rather than the traces

Phase-1 traces cannot supply decode shapes. Decode steps replay inside CUDA
graphs, so no CPU-side annotation is emitted per step and only the eager
prefill chunks appear. `src/capture_shapes.py` instead wraps vLLM's Triton
launch wrapper and runs the engine eager, which records the real problem size
of every launch along with the config vLLM's own heuristic chose. Eager
changes timing, not shapes.
