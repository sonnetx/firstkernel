# A finding that was not real, and what it changed

The phase-2 baseline reported a non-monotonic latency curve. Sixteen tokens
measured 0.630 ms and 32 tokens measured 0.719 ms, against 0.357 ms for 64
tokens, and [BASELINE.md](BASELINE.md) attributed the gap to vLLM's untuned
fallback config. That attribution was wrong.

Two later runs measured the same default config at the same token counts:

| tokens | phase-2 baseline | tuner, same default config | tuned-candidate run |
|---|---|---|---|
| 16 | 0.630 ms | 0.280 ms | 0.263 ms |
| 32 | 0.719 ms | 0.325 ms | 0.294 ms |

The cliff did not reproduce. The first suspect was the router, since the
harness drew a fresh expert prior from a run-time seed and the number of
distinct experts a draw hits sets how many padded blocks the kernel launches.

## Routing draw matters, but not that much

Timing one config across eight routing draws per token count, 30 reps each:

| tokens | median | range across draws | ratio | distinct experts hit |
|---|---|---|---|---|
| 4 | 0.196 ms | 0.192 to 0.202 | 1.06x | 21 to 25 of 128 |
| 8 | 0.238 ms | 0.223 to 0.259 | 1.16x | 29 to 41 |
| 16 | 0.277 ms | 0.254 to 0.289 | 1.14x | 40 to 52 |
| 32 | 0.312 ms | 0.295 to 0.343 | 1.16x | 53 to 69 |
| 64 | 0.357 ms | 0.342 to 0.393 | 1.15x | 66 to 84 |
| 128 | 0.409 ms | 0.387 to 0.446 | 1.15x | 75 to 96 |
| 512 | 0.497 ms | 0.473 to 0.528 | 1.12x | 94 to 111 |

Routing luck alone moves a measurement by up to 1.16x, which is larger than
most of the speedups this project is chasing, so it has to be controlled. But
it does not explain the original outlier: 0.630 ms sits far outside the 0.254
to 0.289 ms range that 16 tokens spans across draws.

## What actually happened

The measurement was interfered with, most likely by another tenant on a shared
node, with GPU clocks unlocked because Sherlock refuses clock control to
unprivileged users. The evidence was already in the phase-2 output and was
reported without being acted on: the interquartile range at 32 tokens was
0.306 ms against a 0.719 ms median, meaning the reps disagreed with each other
by more than 40%. A number that unstable is not a measurement.

## Changes

Timing now runs across three fixed routing draws and takes the median of their
medians, so results are comparable across runs and no longer depend on which
draw a run happened to get. Correctness still runs on a seed chosen at run
time, so a candidate still cannot special-case the tensors it is graded on.

Any measurement whose interquartile range exceeds 15% of its median, or whose
slowest routing draw exceeds its fastest by more than 1.35x, is now labelled
UNSTABLE in the output and in the results JSON. Unstable numbers are reported
as unreliable rather than as findings.

The lesson generalizes past this project. The harness was built to stop a
kernel from cheating its way to a fast number, and the first false result it
produced came from the measurement environment instead, with the warning sign
sitting in plain view in the output. An agent loop optimizing against a signal
this noisy would have spent its budget chasing a phantom 2x at 16 tokens.
