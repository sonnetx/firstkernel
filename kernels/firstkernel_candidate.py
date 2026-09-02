"""Harness candidate shim for the packaged kernel.

The harness loads candidates by file path; this file exposes the installed
package's kernel so you can benchmark it like any candidate:

    python3 harness/bench.py --candidate kernels/firstkernel_candidate.py \
        --compare kernels/standalone.py --set dev

Compare against kernels/standalone.py (the overhead-matched reference running
the tuned config table) to measure the kernel change itself. Comparing against
kernels/baseline.py (vLLM's stock dispatch path) additionally includes ~52us
of per-call framework overhead that CUDA graphs remove in real serving, so
that ratio flatters the kernel; see results/MEASUREMENT.md and the study.
"""

from firstkernel import moe  # noqa: F401
