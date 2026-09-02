"""firstkernel: a faster fused-MoE layer for vLLM on H100.

Two entry points:

    from firstkernel import moe
    out = moe(hidden_states, w1, w2, topk_weights, topk_ids)

runs the layer directly (bf16, SwiGLU, Hopper), and

    import firstkernel
    firstkernel.install()

patches the running vLLM (0.28.x) so its Triton MoE backend uses this kernel
for the unquantized bf16 SwiGLU case, falling through to stock everywhere
else. Call install() before the engine is built, and force the engine
in-process (VLLM_ENABLE_V1_MULTIPROCESSING=0) so the patch reaches the worker.

Verify on your own GPU with:  python -m firstkernel.selftest
"""

from firstkernel.integration import install
from firstkernel.kernel import moe

__version__ = "0.1.0"
__all__ = ["moe", "install", "__version__"]
