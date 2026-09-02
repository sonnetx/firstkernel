# Frozen evaluation protocol

Fixed before the kernel tournament starts. Changes after this point require a
logged amendment at the bottom of this file, with the reason.

## Reference

Every kernel variant is compared paired (one process, interleaved reps,
identical inputs) against `kernels/standalone.py` running the tuned config
table from phase 3a. Beating the untuned heuristic proves nothing, so the
question is always whether a kernel change adds credible gain beyond config
tuning. Variants may choose their own launch configs (a kernel change can
demand different tiling), and the diff shows it.

## Test sets

Three disjoint roles. Token counts, not just seeds, separate them, because a
predictable every-third-shape holdout is visible to anyone reading the
harness.

- **Development** (agents iterate here): M in {1, 4, 16, 64, 256, 1024, 4096},
  routing seeds 11, 137, 263. Agents see these numbers freely.
- **Validation** (selection happens here, run only by the operator): M in
  {2, 8, 32, 128, 512, 2048, 8192, 16384}, routing seeds 1009, 2003, 3001.
  Agents never receive validation numbers during their run.
- **Final test** (run once, after the winner is chosen and integrated): M in
  {24, 48, 96, 192, 384, 768, 1536, 3072, 6144, 12288} plus the production
  shapes {64, 8192, 16384}, with routing seeds drawn from system entropy at
  final-run time and recorded in the result. Nothing is tuned after this run.

Production weighting uses the observed serving distribution (6096/48/192 layer
calls at M = 64/8192/16384). Sets without production shapes report unweighted
medians.

## Tournament budget

- Four strategy branches, one agent each, all forking `kernels/standalone.py`.
- Per branch: at most 10 benchmark submissions to the evaluation worker, at
  most one session. The worker evaluates on the development set only.
- No human code contributions to any branch. Operator intervention is allowed
  only for infrastructure failures (queue death, ssh loss) and is logged in
  the branch record.
- Every branch is reported in the article, including failures and
  zero-improvement outcomes. Attempt counts are part of the result.
- Per branch record: the observable record only, meaning transcripts, tool
  calls, token usage, elapsed time, and effort settings. No claims about
  hidden reasoning.

## Selection rule

The tournament winner is the variant with the highest production-weighted
paired speedup on the **validation** set, subject to all of:

1. Every harness gate passes on every validation shape (correctness vs the
   fp32 reference, input sensitivity, no aliasing, above the HBM floor).
2. No UNSTABLE flag on any production shape.
3. Credibility: production-weighted validation speedup at least 1.02, and
   every one of the three validation routing draws individually above 1.00.
   Anything smaller is reported as no credible kernel-level gain, which is a
   legitimate published outcome of the tournament.

## Pre-registered end-to-end evaluation (phase 5)

Model Qwen3-30B-A3B bf16, one H100 80GB, stock vLLM 0.28.0 defaults (CUDA
graphs on), gpu_memory_utilization 0.92, max_model_len 8192, temperature 0.8,
ignore_eos.

Workloads:
- decode-heavy: 64 concurrent requests, 128-token prompts, 256 output tokens
- prefill-heavy: 16 concurrent requests, 4096-token prompts, 1 output token
- mixed: 32 concurrent requests, 512-token prompts, 128 output tokens

Each workload runs 3 times per allocation across 3 separate SLURM allocations.
Report median and min-to-max range of throughput, TTFT, and ITL.

Success threshold, fixed now: median decode-throughput gain of at least 2%
with the minimum across allocations above 0%, and no median TTFT or ITL
regression worse than 1%. Results are published whether or not the threshold
is met.

An integration smoke test (the phase-3a tuned config transplanted into stock
vLLM's own path, measured end to end under CUDA graphs) runs before the
tournament, so integration-versus-microbenchmark surprises are known first.

## Amendments

- 2026-09-02, before the final test and full e2e ran: TTFT and ITL are
  reported as batch-level proxies from the offline engine rather than
  per-request serving percentiles. The prefill workload's round latency (16
  concurrent 4096-token prefills to first token) stands in for TTFT, and the
  decode workload's per-step time stands in for ITL. Reason: per-request
  percentiles require the OpenAI serving stack, whose in-process kernel
  patching is fragile enough to risk measuring a broken integration; the
  offline engine runs the same engine core under the same CUDA graphs. The
  success thresholds apply to these proxies unchanged.
