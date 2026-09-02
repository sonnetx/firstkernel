"""Base candidate with the Triton source under this file's control.

kernels/baseline.py and kernels/tuned.py both call vLLM's fused_experts, so
neither can change the kernel itself. This one owns the Triton source: it is
the starting point every kernel-level variant is forked from, and the thing
agents in phase 3 edit.

The kernel is vLLM's fused_moe_kernel with the branches this model never takes
removed. Qwen3-30B-A3B runs unquantized bf16, so the fp8, int8, int4, bias,
LoRA, swap-AB and tensor-descriptor paths are all dead code here. Removing them
is not an optimization; it exists so a variant's diff shows the change being
made rather than being buried in branches that never execute. Being no faster
than vLLM is the expected result for this file, and the harness says so.

The surrounding layer (block alignment, activation, weighted reduction) reuses
vLLM's own CUDA ops, so a variant is compared on the Triton GEMM alone.
"""

import json
import os

import torch
import triton
import triton.language as tl
from vllm import _custom_ops as ops
from vllm.model_executor.layers.fused_moe.activation import (
    MoEActivation,
    apply_moe_activation,
)
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)


@triton.jit
def moe_gemm_kernel(
    a_ptr, b_ptr, c_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    N, K, EM, num_valid_tokens,
    stride_am, stride_ak,
    stride_be, stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
):
    """One block of C = A @ B[expert] for the tokens routed to that expert.

    A is (num_tokens, K) activations. B is (E, N, K) stacked expert weights.
    C is (num_tokens * top_k, N). sorted_token_ids groups token slots by the
    expert they routed to, padded so each expert's run divides BLOCK_SIZE_M.
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return

    offs = tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_token_id = pid_m * BLOCK_SIZE_M + offs
    # int64 throughout: stride * offset overflows int32 at large token counts.
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id).to(tl.int64)
    token_mask = offs_token < num_valid_tokens

    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    if off_experts == -1:
        # Padding block for an expert with no tokens on this rank.
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
        c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
        tl.store(c_ptrs, tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=compute_type),
                 mask=c_mask)
        return

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    # Each token slot reads the same source row for all of its top_k copies.
    a_ptrs = a_ptr + (offs_token[:, None] // top_k * stride_am
                      + offs_k[None, :] * stride_ak)
    b_ptrs = (b_ptr + off_experts * stride_be
              + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn))

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_remaining = K - k * BLOCK_SIZE_K
        a = tl.load(a_ptrs,
                    mask=token_mask[:, None] & (offs_k[None, :] < k_remaining),
                    other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < k_remaining, other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if MUL_ROUTED_WEIGHT:
        # In fp32, before the cast: the router weight is small and casting
        # first loses precision that shows up in the layer output.
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)
        accumulator *= moe_weight[:, None]

    accumulator = accumulator.to(compute_type)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


_CONFIG_PATH = os.environ.get(
    "FK_MOE_CONFIG",
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "results",
        "E=128,N=768,device_name=NVIDIA_H100_80GB_HBM3.json",
    ),
)


def _load_config_table():
    try:
        with open(_CONFIG_PATH) as f:
            return {int(k): v for k, v in json.load(f).items()}
    except FileNotFoundError:
        return None


_CONFIG_TABLE = _load_config_table()


def default_config(M: int, E: int) -> dict:
    """The phase-3a tuned table (vLLM lookup semantics: nearest token count).

    PROTOCOL.md makes this the reference config: a variant compared against
    this file must add gain beyond config tuning, not rediscover it. The
    inlined vLLM heuristic below is only the fallback when the table is
    missing.
    """
    if _CONFIG_TABLE:
        nearest = min(_CONFIG_TABLE, key=lambda t: abs(t - M))
        cfg = dict(_CONFIG_TABLE[nearest])
        cfg.pop("SPLIT_K", None)
        return cfg
    if M <= 32:
        block_m = 16
    elif M <= 96:
        block_m = 32
    elif M <= 512:
        block_m = 64
    else:
        block_m = 128
    return {
        "BLOCK_SIZE_M": block_m,
        "BLOCK_SIZE_N": 64 if M <= 64 else 128,
        "BLOCK_SIZE_K": 128 if M <= 64 else 64,
        "GROUP_SIZE_M": 16 if M // max(E, 1) > 128 else 1,
        "num_warps": 4 if M <= 128 else 8,
        "num_stages": 4 if M <= 32 else 3,
    }


def launch(A, B, C, topk_weights, sorted_token_ids, expert_ids,
           num_tokens_post_padded, mul_routed_weight, top_k, config):
    EM = sorted_token_ids.size(0)
    if A.size(0) < config["BLOCK_SIZE_M"]:
        # Fewer tokens than a block: every expert can hold at most one token,
        # so blocks past that are all padding and need not be launched.
        EM = min(EM, A.size(0) * top_k * config["BLOCK_SIZE_M"])
    N = B.size(1)

    grid = lambda META: (  # noqa: E731
        triton.cdiv(EM, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    moe_gemm_kernel[grid](
        A, B, C,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        # num_valid_tokens is the count of real (token, slot) pairs; slots past
        # it in sorted_token_ids are alignment padding. Both GEMMs see the same
        # count: the first has M rows with top_k slots each, the second has
        # M * top_k rows with one slot each.
        N, A.size(1), EM, A.size(0) * top_k,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(2), B.stride(1),
        C.stride(1), C.stride(2),
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=top_k,
        compute_type=tl.bfloat16,
        **config,
    )


def moe(hidden_states, w1, w2, topk_weights, topk_ids):
    M, K = hidden_states.shape
    E = w1.size(0)
    N = w1.size(1)            # 2 * moe_intermediate
    top_k = topk_ids.size(1)
    inter = N // 2

    config = default_config(M, E)
    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids, config["BLOCK_SIZE_M"], E
    )

    cache1 = torch.empty((M, top_k, N), device=hidden_states.device,
                         dtype=hidden_states.dtype)
    cache2 = torch.empty((M * top_k, inter), device=hidden_states.device,
                         dtype=hidden_states.dtype)
    cache3 = torch.empty((M, top_k, K), device=hidden_states.device,
                         dtype=hidden_states.dtype)

    launch(hidden_states, w1, cache1, topk_weights, sorted_token_ids, expert_ids,
           num_tokens_post_padded, False, top_k, config)

    # vLLM's own activation entry point, so a variant is compared on the
    # Triton GEMM rather than on a different SwiGLU implementation.
    apply_moe_activation(MoEActivation.SILU, cache2, cache1.view(-1, N))

    launch(cache2, w2, cache3, topk_weights, sorted_token_ids, expert_ids,
           num_tokens_post_padded, True, 1, config)

    out = torch.empty((M, K), device=hidden_states.device,
                      dtype=hidden_states.dtype)
    ops.moe_sum(cache3, out)
    return out
