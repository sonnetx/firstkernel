# From Microbenchmark to vLLM Engine Throughput: Agent-Guided Optimization of an MoE Kernel

GPU kernel optimizations are usually developed in a microbenchmark environment, while the numbers people actually experience are engine-level. Leaderboards like Artificial Analysis and OpenRouter rank providers by delivered throughput and latency, and no isolated kernel time appears anywhere in them, so a kernel speedup only matters to the extent it survives into those engine numbers. We profile Qwen3-30B-A3B serving on an NVIDIA H100, identify vLLM's Triton fused mixture-of-experts (MoE) kernel as the dominant editable GPU workload, and use coding agents to investigate four optimization strategies. Our final implementation cost roughly 1.1 million agent output tokens and nine H100-hours to produce, and combines activation fusion, router-reduction fusion, and Tensor Memory Accelerator (TMA) weight loads. On a held-out layer benchmark, it improves decode performance by 1.037× and prefill performance by up to 1.216×. After integration into vLLM with CUDA graphs enabled, these gains become 1.020× for decode, 1.116× for prefill, and 1.027× for a mixed workload. 

## 1. Experimental objective

Kernel optimization is commonly evaluated with isolated operators and synthetic input shapes, which fail to capture important aspects such as framework dispatch, memory management, CUDA graph execution, or the other model components that share end-to-end latency. 

Our objective was to optimize a kernel on the complete optimization path. Experiments used Qwen3-30B-A3B in bf16, vLLM 0.28.0, and a single NVIDIA H100 80GB. 


We profiled separate prefill-heavy and decode-heavy serving workloads and ranked CUDA kernels by aggregate GPU time. vLLM's Triton `fused_moe_kernel`, which evaluates the expert projections in each MoE layer, accounted for 47.9% of prefill GPU time and 66.5% of decode GPU time. The next-largest categories were vendor-provided FlashAttention and cuBLAS kernels. The MoE kernel was therefore both performance-relevant and directly modifiable.

## 2. Baseline and evaluation protocol

### 2.1 Configuration baseline

vLLM selects `fused_moe_kernel` launch parameters from tables indexed by model dimensions and hardware. No H100 table entry covered this model's configuration of 128 experts and expert width 768, so the stock implementation used a generic heuristic.

Before modifying the kernel, we ran the existing configuration search. The resulting table improved production-weighted MoE layer time by 1.0154×. All subsequent kernel candidates were compared with this tuned reference. This prevents launch-parameter search from being attributed to a change in kernel design.

### 2.2 Benchmark unit and correctness gates

The benchmark evaluates a complete MoE layer: gate/up projection, SwiGLU activation, down projection, and router-weighted top-k reduction. This boundary includes the intermediate memory traffic that fusion is intended to remove.

Every candidate was evaluated against an independent fp32 implementation of the mathematical MoE definition. Additional checks covered output shape and dtype, randomized inputs and routing assignments, input sensitivity, storage aliasing, and a physical lower bound derived from mandatory HBM traffic. Timing results were withheld for candidates that failed any gate.

Baseline and candidate executions were paired within one process and interleaved on identical inputs. Measurements used CUDA events, warmup iterations, three fixed routing draws, and cache eviction between repetitions. Runs were marked unstable when the interquartile range exceeded 15% of the median or when variation across routing draws exceeded the predefined limit.

### 2.3 Scaffold calibration

An unchanged copy of the vLLM implementation measured 1.13× faster in the editable scaffold than through the stock path. The absolute difference remained approximately 52 microseconds across token counts from 1 to 16,384, indicating removed fixed overhead rather than faster GPU computation. The scaffold bypassed portions of vLLM's dispatch and workspace logic.

This calibration established two comparison boundaries. Kernel candidates were compared with the unchanged implementation inside the same scaffold. Application claims were made only after reintegration into vLLM. The 1.13× scaffold difference was excluded from reported kernel gains.

### 2.4 Agent protocol

Development, validation, and final token-count sets were fixed before optimization. Agents received development measurements but not validation results. The final set was evaluated once after candidate selection and adversarial verification, using routing seeds drawn within the evaluation job.

Four optimization branches received the same initial implementation, distinct strategy briefs, and a budget of ten benchmark submissions each. Human-written kernel changes were excluded during the tournament, and all branch outcomes were retained.

## 3. Kernel optimization

The four branches addressed different components of the MoE computation.

| Branch | Optimization hypothesis | Weighted validation speedup |
|---|---|---:|
| Activation fusion | Evaluate SwiGLU in the gate/up GEMM epilogue | 1.0587× |
| Reduction fusion | Accumulate router-weighted expert outputs in the down GEMM | 1.0479× |
| Decode specialization | Improve the small-M memory-access path | 1.0176× |
| TMA weight loading | Load expert-weight tiles through Hopper TMA | **1.1422×** |

The validation score was determined by the production shapes present in the validation set. Because the 64-token decode shape belonged to the development set, the weighted selection criterion favored prefill. This limitation was preserved rather than revised after observing the results; decode was evaluated again in the final set and in the preregistered engine workload.

The decode branch found limited remaining headroom. After configuration tuning, the kernel streamed expert weights at approximately 75% of peak HBM bandwidth. Most scheduling changes were neutral or detrimental, and unpredicated loads produced only a small improvement.

### 3.1 Composed implementation

A fifth agent combined the three compatible mechanisms: activation fusion, reduction fusion, and TMA weight loading. Their dataflow differs as follows:

```text
Stock path
  gate/up GEMM → SwiGLU → down GEMM → top-k reduction → output

Composed path
  gate/up GEMM + SwiGLU → down GEMM + top-k reduction → output
            TMA loads              TMA loads + bf16 atomics
```

The first kernel computes the gate and up projections together, evaluates SwiGLU in fp32 registers, and writes only the activated `(M × top_k, 768)` intermediate. This removes the wider `(M, top_k, 1536)` intermediate and the standalone activation launch.

The second kernel multiplies the down-projection result by the router weight and accumulates directly into the final `(M, 2048)` output using relaxed bf16 atomic additions. This removes the `(M, top_k, 2048)` intermediate and the separate top-k reduction. At token counts of 1,024 or greater, both kernels load expert-weight tiles through TMA descriptors. The production prefill path also uses a 256-column down-projection tile selected during branch development.

At 4,096 tokens, the individual branch results predicted a 1.128× speedup under an independence assumption; the composed implementation measured 1.131×. On validation shapes, it achieved 1.223× and 1.232× at 8,192 and 16,384 tokens, respectively. These were validation measurements rather than final results.

## 4. Verification and integration

The selected implementation underwent 41 adversarial checks covering token counts around dispatch boundaries, routing patterns designed to maximize atomic contention, extreme activation magnitudes, and repeated execution. No additional correctness failure was observed within the supported input contract.

Reduction fusion introduces bitwise non-determinism because atomic additions may be applied in different orders. Across twenty repeated executions, output variation remained within a few bf16 units in the last place and was approximately 700 times smaller than the numerical tolerance. The implementation is therefore tolerance-correct under the tested conditions, but not bitwise deterministic.

The composed path was installed at `TritonExperts.apply` for the unquantized bf16/SwiGLU case without LoRA or expert remapping. It uses vLLM's output and workspace buffers so that allocation behavior remains compatible with CUDA graph capture. Unsupported configurations fall through to the original implementation. This is a runtime integration into stock vLLM 0.28.0; it has not yet been merged upstream.

At model level, stock and modified vLLM produced identical tokens across a 2,048-token greedy decode. Log-probability differences were consistent with the kernel's measured run-to-run variation.

## 5. Results

### 5.1 One-time final layer evaluation

The final layer benchmark included ten token counts absent from development and validation, the three production sizes, and entropy-drawn routing seeds 81,374, 71,607, and 34,693. All thirteen cases passed the correctness and stability gates.

| Tokens | Speedup | Tokens | Speedup |
|---:|---:|---:|---:|
| 24 | 1.053× | 768 | 1.050× |
| 48 | 1.038× | 1,536 | 1.082× |
| **64 (decode)** | **1.037×** | 3,072 | 1.117× |
| 96 | 1.046× | 6,144 | 1.156× |
| 192 | 1.058× | **8,192 (prefill)** | **1.196×** |
| 384 | 1.039× | 12,288 | 1.219× |
| — | — | **16,384 (prefill)** | **1.216×** |

The production-shape results were 1.037× for decode and 1.196× to 1.216× for prefill. The larger prefill gains are consistent with the design: fusion removes intermediate traffic whose contribution grows with problem size, while decode remains dominated by expert-weight streaming.

### 5.2 Fixed-batch vLLM engine evaluation

The engine evaluation used `vllm.LLM.generate()` with three fixed batches, three independent allocations, and three timed rounds per allocation. CUDA graphs were enabled. These measurements include model execution and the in-process scheduler, but not an HTTP server, asynchronous arrivals, or per-request latency distributions.

| Workload | Stock | Composed | Median gain | Per-allocation gains |
|---|---:|---:|---:|---:|
| Decode | 4,547.7 tok/s | 4,636.7 tok/s | 1.0196× | 1.0284×, 1.0192×, 1.0245× |
| Prefill | 14.3 tok/s | 16.0 tok/s | **1.1159×** | 1.1179×, 1.1115×, 1.1169× |
| Mixed | 3,473.2 tok/s | 3,565.7 tok/s | 1.0266× | 1.0273×, 1.0252×, 1.0297× |

The prefill latency proxy decreased from 1.117 seconds to 1.001 seconds, corresponding to a 1.116× speedup, or a 10.4% reduction in elapsed time. Decode step time decreased from 14.07 to 13.80 milliseconds. No workload, round, or allocation regressed.

Decode improved in all nine allocation-round comparisons. The preregistered threshold admitted two aggregation interpretations: the pooled median gain was 1.96%, while the median allocation-level gain was 2.45%. Both are reported; the result is summarized as approximately 2%.

### 5.3 Microbenchmark-to-engine translation

| Measurement boundary | Decode | Prefill |
|---|---:|---:|
| Final MoE layer benchmark | 1.037× | 1.216× |
| Integrated fixed-batch vLLM engine | 1.020× | 1.116× |
| Ratio of relative gains | approximately 53% | approximately 54% |

Approximately half of the relative layer-level improvement remained at the engine level. This ratio is descriptive rather than a modeled attribution: the layer and engine benchmarks have different boundaries, and the full engine includes attention, scheduling, sampling, graph replay, and other work unaffected by the optimization.

## 6. Discussion

The layer and fixed-batch engine measurements provide complementary evidence. The layer benchmark isolates the optimized computation, supports rapid experimental iteration, and identifies the shapes for which each mechanism is effective. The engine benchmark measures the value after framework and model costs are restored. It does not establish online-serving throughput or tail latency; those require an asynchronous client and controlled request arrivals.

The results are also strongly workload-dependent. Fusion and TMA loading are effective for large prefill calls, where intermediate traffic and weight-transfer scheduling represent material fractions of execution time. The 64-token decode path is already close to its memory-bandwidth limit, so the same mechanisms yield a substantially smaller gain.

The agent tournament was useful primarily as a method for parallel hypothesis evaluation. Distinct strategy briefs produced separable mechanisms that could be validated independently and later composed. The study does not establish that agents outperform expert kernel engineers, because no human-control condition was evaluated.

Several limitations constrain generalization. The experiments cover one model, one H100, one vLLM release, bf16 inputs, and an unquantized MoE path. The atomic reduction is not bitwise deterministic. The validation-set composition favored prefill, the decode engine gain is close to the measurement threshold, and the current integration is a runtime patch rather than an upstream vLLM implementation. The evaluation uses fixed batches rather than an online request process.

## 7. Future work

Two follow-up campaigns are in preparation, and we state them here to mark the territory rather than as open questions.

The first tests transfer to online serving. The fixed-batch engine evaluation in Section 5.2 measures throughput under saturated batches, and a separate pre-registered campaign measures the same stock-versus-composed comparison under an asynchronous request process with controlled arrivals, reporting goodput and per-request latency distributions.

The second applies the same protocol to collective communication. Every measurement in this study is single-GPU because the model fits on one device, so serving involves no communication. Under expert parallelism the dominant cost becomes the all-to-all dispatch and combine surrounding exactly the GEMMs optimized here, and tensor parallelism adds the small-message allreduce and GEMM-communication overlap paths. Agent-based optimization has not touched these kernels, and the measurement hazards this study catalogs grow worse there. Collective microbenchmarks report bus bandwidth on warmed rings while production value depends on achieved overlap, ranks share no clock so timing requires cross-rank discipline, a collective finishes when its slowest rank finishes, and correctness adds delivery completeness and ordering to numerical tolerance. A draft protocol for this campaign accompanies the repository, extending the frozen-protocol design of this study with rank-aware timing rules, a wire-bandwidth impossibility bound, and overlap accounting that requires wins to appear in step time rather than in hidden-latency microbenchmarks.

We also plan to convert the runtime integration and tuned configuration table into an upstream vLLM contribution, with the non-determinism of the atomic reduction disclosed in the submission.

## 8. Conclusion

Starting from a production profile, agent-guided optimization produced a composed MoE implementation that removes two intermediate operations and uses Hopper TMA for expert-weight movement. The final layer benchmark measured up to a 1.216× prefill speedup and a 1.037× decode speedup. Fixed-batch vLLM execution retained approximately half of each relative gain, reaching 1.116× for prefill and approximately 1.020× for decode without a measured regression.

Six agent runs consumed approximately 1.1 million output tokens. The evaluation worker executed 34 candidate runs, and the full effort consumed approximately nine H100-hours. The remaining engineering step is to convert the runtime integration and tuned configuration into an upstream vLLM contribution. The repository contains the benchmark harness, experiment records, final measurements, and SLURM entry points used in the evaluation.
