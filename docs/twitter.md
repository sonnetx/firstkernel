# The Kernel Got 22% Faster. vLLM Got 12% Faster.

We used coding agents to optimize the mixture-of-experts kernel that dominates Qwen3-30B-A3B inference in vLLM. The best prefill result was a 1.216× layer-level speedup and a 1.116× improvement after integration into the engine.

Getting there was mostly a measurement problem.

## Choosing the kernel

We profiled Qwen3-30B-A3B in bf16 on an H100. vLLM's Triton MoE kernel accounted for 66.5% of decode GPU time and 47.9% of prefill GPU time. It was both important and editable.

The first improvement did not require a new kernel. vLLM has hundreds of tuned configuration tables for its fused MoE implementation, but none covered this model's shape: 128 experts with an expert width of 768. We ran vLLM's configuration search and used the resulting table as the baseline. That contributed a 1.015× production-weighted improvement, which we kept separate from subsequent kernel work.

Four agents then investigated different parts of the layer under a frozen protocol. Each received one strategy and at most ten benchmark submissions. Development, validation, and final token counts were separated before the agents started.

One agent fused SwiGLU into the gate/up GEMM. Another fused router weighting and the top-k reduction into the down GEMM using relaxed bf16 atomics. A third used Hopper's Tensor Memory Accelerator for expert-weight loads. The fourth worked on small decode shapes and mostly found that the tuned kernel was already difficult to improve: it streamed expert weights at about 75% of peak HBM bandwidth.

A fifth agent combined the compatible changes. The resulting MoE layer uses two GPU launches instead of five. On the held-out final shapes, it improved by 1.037× at the 64-token decode shape, 1.196× at 8,192 tokens, and 1.216× at 16,384 tokens.

The atomic reduction is not bitwise deterministic. Across repeated runs, the variation was a few bf16 ULPs and roughly 700 times below our numerical tolerance. Stock and modified vLLM still generated identical tokens across a 2,048-token greedy decode.

## Most of the work was rejecting bad measurements

Before the final result, the pipeline produced several speedups that looked credible and were wrong.

A noisy neighbor on the shared cluster created a reproducible-looking 2× latency cliff that disappeared on another allocation. Our configuration tuner selected the fastest of 120 noisy measurements and consequently favored lucky samples; one apparent winner was slower when remeasured. An early benchmark used an internally consistent MoE shape that the model never executes.

The most useful failure came from the editable benchmark scaffold. An unchanged copy of vLLM's implementation ran 1.13× faster there than through the stock path. The absolute difference was approximately 52 microseconds at every token count from 1 to 16,384. That pattern identified fixed dispatch and workspace overhead, not faster GPU computation. CUDA graph replay also amortizes much of this overhead in the engine.

We changed the evaluation rather than trying to explain these numbers away. Baseline and candidate ran in the same process, interleaved on identical inputs. Measurements covered multiple routing seeds, evicted caches between repetitions, and were rejected when unstable. The final shapes were evaluated once after candidate selection.

These controls removed larger gains than the optimization eventually produced.

## What survived integration

We installed the composed implementation behind vLLM's existing `TritonExperts.apply` interface, with the stock path retained for unsupported configurations. CUDA graphs remained enabled.

Across three independent H100 allocations, fixed-batch `LLM.generate()` measurements showed:

| Workload | Stock | Composed | Gain |
|---|---:|---:|---:|
| Decode | 4,547.7 tok/s | 4,636.7 tok/s | 1.020× |
| Prefill | 14.3 tok/s | 16.0 tok/s | 1.116× |
| Mixed | 3,473.2 tok/s | 3,565.7 tok/s | 1.027× |

Every workload, allocation, and timed round improved. Prefill batch latency fell from 1.117 seconds to 1.001 seconds. Decode step time fell from 14.07 to 13.80 milliseconds.

The decode result sits directly on our preregistered 2% threshold. Pooling all rounds gives a 1.96% improvement; taking the median allocation-level gain gives 2.45%. Both are valid readings of wording we should have made more precise, so we report both.

Roughly half of the relative layer-level gain remained after integration:

| Boundary | Decode | Prefill |
|---|---:|---:|
| MoE layer | 1.037× | 1.216× |
| Fixed-batch vLLM engine | 1.020× | 1.116× |

That is not an online-serving result. The engine benchmark uses fixed batches and does not measure asynchronous arrivals, queueing, or per-request tail latency. A separate online campaign covering fixed and variable request lengths, low and high arrival rates, single-stream decode, and long-context prefill is currently queued on Sherlock.

The project used six agent runs, roughly 1.1 million output tokens, 34 candidate evaluations, and about nine H100-hours. The agents found useful kernel changes. The harder part was building an evaluation that could distinguish those changes from noise, selection bias, and a flattering benchmark boundary.

Code: https://github.com/sonnetx/firstkernel
Writeup: https://sonnetx.github.io/firstkernel/

## Post

A fused MoE layer for vLLM on H100, and a bench for the next one.

Install it, or replace kernel.py. The harness grades you against an fp32 reference. On Qwen3-30B, prefill was 1.22× at the layer and 1.12× in the engine.

https://sonnetx.github.io/firstkernel/
