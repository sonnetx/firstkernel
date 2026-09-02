"""Install the firstkernel MoE kernel inside stock vLLM's MoE path.

Patches TritonExperts.apply: for the exact case this project optimized
(unquantized bf16, SILU, no LoRA, no expert map, router weight on output),
the layer runs the composed kernel's two launches; every other case falls
through to stock. The patch writes into vLLM's own output buffer and takes its
intermediate from vLLM's workspace2 so allocation behavior under CUDA-graph
capture matches stock. workspace13 is not used because vLLM may alias the
output tensor into it, and the atomic down GEMM must not read its input from
memory it is writing.

Import and call install() before the engine is built, with
VLLM_ENABLE_V1_MULTIPROCESSING=0 so the patch reaches the worker.
"""

import torch

from firstkernel import kernel as _kernel


def install() -> None:
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonExperts

    comp = _kernel
    orig_apply = TritonExperts.apply
    stats = {"composed": 0, "fallback": 0}

    def apply(
        self,
        output,
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        activation,
        global_num_experts,
        expert_map,
        a1q_scale,
        a2_scale,
        workspace13,
        workspace2,
        expert_tokens_meta,
        apply_router_weight_on_input,
    ):
        qc = self.quant_config
        eligible = (
            activation == MoEActivation.SILU
            and expert_map is None
            and not apply_router_weight_on_input
            and hidden_states.dtype == torch.bfloat16
            and getattr(self, "_lora_context", None) is None
            and self.w1_scale is None
            and self.w2_scale is None
            and self.w1_bias is None
            and self.w2_bias is None
            and not (
                qc.use_fp8_w8a8
                or qc.use_int8_w8a8
                or qc.use_int8_w8a16
                or qc.use_int4_w4a16
            )
        )
        if not eligible:
            stats["fallback"] += 1
            return orig_apply(
                self, output, hidden_states, w1, w2, topk_weights, topk_ids,
                activation, global_num_experts, expert_map, a1q_scale,
                a2_scale, workspace13, workspace2, expert_tokens_meta,
                apply_router_weight_on_input,
            )

        stats["composed"] += 1
        M, K = hidden_states.shape
        E = w1.size(0)
        N = w1.size(1)
        top_k = topk_ids.size(1)
        inter = N // 2

        config, key = comp.default_config(M, E)
        down_config = config
        if M >= comp._DOWN_TMA_OVERRIDE_MIN_M:
            down_config = dict(comp._DOWN_TMA_OVERRIDE)
        use_tma_down = (
            M >= comp._TMA_MIN_M and inter % down_config["BLOCK_SIZE_K"] == 0
        )
        fused_cfg = comp.fused_config(config, key)
        use_tma_fused = (
            M >= comp._TMA_MIN_M
            and K % fused_cfg["BLOCK_SIZE_K"] == 0
            and inter % fused_cfg["BLOCK_SIZE_N"] == 0
        )

        sorted_token_ids, expert_ids, num_tokens_post_padded = (
            comp.moe_align_block_size(topk_ids, down_config["BLOCK_SIZE_M"], E)
        )

        cache2 = workspace2.reshape(-1)[: M * top_k * inter].view(M * top_k, inter)

        comp.launch_fused(
            hidden_states, w1, cache2, sorted_token_ids, expert_ids,
            num_tokens_post_padded, top_k, fused_cfg, use_tma=use_tma_fused,
        )
        output.zero_()
        comp.launch_down_atomic(
            cache2, w2, output, topk_weights, sorted_token_ids, expert_ids,
            num_tokens_post_padded, top_k, down_config, use_tma=use_tma_down,
        )

    TritonExperts.apply = apply
    apply._fk_stats = stats
    print("firstkernel: kernel installed in TritonExperts.apply",
          flush=True)
