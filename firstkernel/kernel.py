"""firstkernel.kernel: the composed MoE kernel, packaged.

Provenance: byte-derived from kernels/variants/composed.py at the commit that
passed final testing; only the config-table loading differs. The research
record (journals, verification report, measurements) lives in the repository
root.

Operator-commissioned composition of the validated tournament mechanisms,
which touch disjoint parts of the layer:

1. fuse-act (validated 1.025-1.069x at every validation shape): the gate/up
   GEMM computes matching gate and up column tiles in one program, applies
   silu(gate) * up in fp32 in registers, and writes only the activated
   (M*top_k, 768) intermediate. The (M, top_k, 1536) round trip and the
   activation launch are gone. Carried verbatim, including the per-key
   FUSED_OVERRIDES launch configs.

2. prefill-wave (validated 1.128x/1.144x at 8192/16384): expert weight tiles
   load through one host-built 3D (E, N, K) tensor descriptor per launch (TMA
   bulk copies into swizzled shared memory, indexed by expert id per block)
   at M >= 1024, with the down GEMM's config overridden to 128x256x64/G16/s4
   at M >= 6144 (the branch's sub-04 probe showed the table's 128-wide tile
   loses ~7% under TMA).

3. fuse-reduce (validated 1.047x at the production prefill shapes): the down
   GEMM multiplies by the router weight in fp32, casts to bf16, and
   accumulates straight into the zero-initialized (M, 2048) output with
   relaxed atomic adds, so the (M, top_k, 2048) intermediate and the moe_sum
   launch disappear (the output memset takes moe_sum's slot).

The layer is two kernel launches at every M: the fused gate/up GEMM writes
the activated (M*top_k, 768) intermediate, and the atomic down GEMM reduces
into the output. At M >= 1024 both switch from pointer-path weight loads to
the TMA descriptor path (the fused kernel reads gate and up tiles from one
(E, 1536, 2048) descriptor, the gate tile at column pid_n*BN and the up tile
768 columns further along). The pieces were added one per submission and
measured against the previous step; see tournament/composed/JOURNAL.md.
"""

import json
import os

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)


@triton.jit
def fused_gateup_silu_kernel(
    a_ptr, b_ptr, c_ptr,
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
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
):
    """C = silu(A @ Bg[expert]) * (A @ Bu[expert]) for tokens routed there.

    A is (num_tokens, K) activations. B is (E, 2N, K) stacked expert weights
    whose first N output columns are the gate projection and last N are the
    up projection. C is (num_tokens * top_k, N) activated intermediate.
    N here is the intermediate width (768), half of B's output dimension.
    Verbatim from the fuse-act branch winner.
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
    bg_ptrs = (b_ptr + off_experts * stride_be
               + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn))
    # Up-projection columns sit N further along B's output dimension.
    bu_ptrs = bg_ptrs + N * stride_bn

    acc_gate = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    acc_up = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_remaining = K - k * BLOCK_SIZE_K
        a = tl.load(a_ptrs,
                    mask=token_mask[:, None] & (offs_k[None, :] < k_remaining),
                    other=0.0)
        bg = tl.load(bg_ptrs, mask=offs_k[:, None] < k_remaining, other=0.0)
        bu = tl.load(bu_ptrs, mask=offs_k[:, None] < k_remaining, other=0.0)
        acc_gate += tl.dot(a, bg)
        acc_up += tl.dot(a, bu)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        bg_ptrs += BLOCK_SIZE_K * stride_bk
        bu_ptrs += BLOCK_SIZE_K * stride_bk

    # silu in fp32 before the product and the bf16 downcast; casting earlier
    # loses precision that shows up against the fp32 reference.
    acc = acc_gate * tl.sigmoid(acc_gate) * acc_up
    acc = acc.to(compute_type)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


@triton.jit
def fused_gateup_silu_kernel_tma(
    a_ptr, b_desc, c_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    N, K, EM, num_valid_tokens,
    stride_am, stride_ak,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
):
    """Fused gate/up kernel with B tiles via a host-built descriptor.

    b_desc is a 3D (E, 2N, K) tensor descriptor over w1 with block
    (1, BLOCK_SIZE_N, BLOCK_SIZE_K); the gate tile sits at column offset
    pid_n * BLOCK_SIZE_N and the up tile N further along. The host guarantees
    K % BLOCK_SIZE_K == 0 and N % BLOCK_SIZE_N == 0 on this path, so only the
    token-validity mask applies to A and tiles never straddle the gate/up
    boundary.
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
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id).to(tl.int64)
    token_mask = offs_token < num_valid_tokens

    # int32: descriptor indices must be 32-bit, and E*2N*K fits easily.
    off_experts = tl.load(expert_ids_ptr + pid_m)
    if off_experts == -1:
        # Padding block for an expert with no tokens on this rank.
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
        c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
        tl.store(c_ptrs, tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=compute_type),
                 mask=c_mask)
        return

    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_token[:, None] // top_k * stride_am
                      + offs_k[None, :] * stride_ak)

    acc_gate = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    acc_up = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0)
        bg = tl.reshape(
            b_desc.load([off_experts, pid_n * BLOCK_SIZE_N, k * BLOCK_SIZE_K]),
            (BLOCK_SIZE_N, BLOCK_SIZE_K),
        )
        bu = tl.reshape(
            b_desc.load([off_experts, pid_n * BLOCK_SIZE_N + N, k * BLOCK_SIZE_K]),
            (BLOCK_SIZE_N, BLOCK_SIZE_K),
        )
        acc_gate += tl.dot(a, bg.T)
        acc_up += tl.dot(a, bu.T)
        a_ptrs += BLOCK_SIZE_K * stride_ak

    # silu in fp32 before the product and the bf16 downcast; casting earlier
    # loses precision that shows up against the fp32 reference.
    acc = acc_gate * tl.sigmoid(acc_gate) * acc_up
    acc = acc.to(compute_type)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


@triton.jit
def moe_down_gemm_atomic_kernel(
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
    OUT_TOPK: tl.constexpr,
    compute_type: tl.constexpr,
):
    """Down GEMM with the router-weighted top_k reduction fused in.

    Pointer-path weight loads; used below the TMA gate. Verbatim from the
    fuse-reduce branch winner. A is (num_tokens * top_k, K) activated
    intermediates. C is the final (num_tokens, N) layer output, pre-zeroed by
    the caller: each block multiplies its tile by the slot's router weight in
    fp32 and atomically adds it into row (slot // OUT_TOPK).
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
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id).to(tl.int64)
    token_mask = offs_token < num_valid_tokens

    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    if off_experts == -1:
        # Padding block: the output is pre-zeroed, nothing to contribute.
        return

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_token[:, None] * stride_am + offs_k[None, :] * stride_ak)
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

    # Router weight in fp32, before the cast, exactly as the baseline epilogue.
    moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)
    accumulator *= moe_weight[:, None]
    accumulator = accumulator.to(compute_type)

    offs_out_m = offs_token // OUT_TOPK
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_out_m[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.atomic_add(c_ptrs, accumulator, mask=c_mask, sem="relaxed")


@triton.jit
def moe_down_gemm_atomic_tma(
    a_ptr, b_desc, c_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    N, K, EM, num_valid_tokens,
    stride_am, stride_ak,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    OUT_TOPK: tl.constexpr,
    compute_type: tl.constexpr,
):
    """Down GEMM with TMA B loads and the top_k reduction fused in.

    A is (num_tokens * top_k, K) activated intermediates, one row per
    (token, slot) pair. b_desc is the 3D (E, N, K) weight descriptor. C is
    the final (num_tokens, N) layer output, pre-zeroed by the caller: each
    block multiplies its tile by the slot's router weight in fp32, casts to
    bf16, and atomically adds it into row (slot // OUT_TOPK). A token's top_k
    slots always route to distinct experts, so no two rows within one block
    collide; the atomics only serialize the at-most-OUT_TOPK contributions
    per output element arriving from different expert runs. fuse-reduce's
    validated epilogue on prefill-wave's validated load path.
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
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id).to(tl.int64)
    token_mask = offs_token < num_valid_tokens

    # int32: descriptor indices must be 32-bit.
    off_experts = tl.load(expert_ids_ptr + pid_m)
    if off_experts == -1:
        # Padding block: the output is pre-zeroed, nothing to contribute.
        return

    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_token[:, None] * stride_am
                      + offs_k[None, :] * stride_ak)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0)
        b = tl.reshape(
            b_desc.load([off_experts, pid_n * BLOCK_SIZE_N, k * BLOCK_SIZE_K]),
            (BLOCK_SIZE_N, BLOCK_SIZE_K),
        )
        accumulator += tl.dot(a, b.T)
        a_ptrs += BLOCK_SIZE_K * stride_ak

    # Router weight in fp32, before the cast, exactly as the baseline epilogue.
    moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)
    accumulator *= moe_weight[:, None]
    accumulator = accumulator.to(compute_type)

    offs_out_m = offs_token // OUT_TOPK
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_out_m[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.atomic_add(c_ptrs, accumulator, mask=c_mask, sem="relaxed")


_CONFIG_NAME = "E=128,N=768,device_name=NVIDIA_H100_80GB_HBM3.json"
_HERE = os.path.dirname(os.path.abspath(__file__))
# The tuned launch-config table ships with the package. FK_MOE_CONFIG
# overrides it, which is how you point the kernel at a table you re-tuned
# for a different expert shape or GPU (see the README).
_CONFIG_CANDIDATES = [
    os.environ.get("FK_MOE_CONFIG"),
    os.path.join(_HERE, "configs", _CONFIG_NAME),
]


def _load_config_table():
    for path in _CONFIG_CANDIDATES:
        if not path:
            continue
        try:
            with open(path) as f:
                return {int(k): v for k, v in json.load(f).items()}
        except (FileNotFoundError, NotADirectoryError):
            continue
    return None


_CONFIG_TABLE = _load_config_table()


def default_config(M: int, E: int):
    """The phase-3a tuned table (vLLM lookup semantics: nearest token count).

    Returns (config, nearest_table_key); the key indexes FUSED_OVERRIDES.
    """
    if _CONFIG_TABLE:
        nearest = min(_CONFIG_TABLE, key=lambda t: abs(t - M))
        cfg = dict(_CONFIG_TABLE[nearest])
        cfg.pop("SPLIT_K", None)
        return cfg, nearest
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
    }, None


# Per-table-key launch configs for the fused kernel, carried verbatim from the
# fuse-act branch winner (measured over its submissions 01-07). BLOCK_SIZE_M
# must stay equal to the down GEMM's because both share the block-aligned
# token sort.
FUSED_OVERRIDES = {
    # key 64: BN=32/S5/W4 measured best on the decode shape (1.036, 1.040).
    64: {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 128,
         "GROUP_SIZE_M": 16, "num_warps": 4, "num_stages": 5},
    # key 1024: G1/S4 measured best (1.034; G16 1.030, S3 0.994).
    1024: {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64,
           "GROUP_SIZE_M": 1, "num_warps": 8, "num_stages": 4},
    # key 4096: G1/S4, reproduced 1.034 twice.
    4096: {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64,
           "GROUP_SIZE_M": 1, "num_warps": 8, "num_stages": 4},
    # Keys below are never exercised by the dev set; set by extrapolation from
    # the nearest measured winner so the derived-cap heuristic's untested
    # fallbacks never run.
    # key 2: mirror key 1's fused config, which measured 1.13x at M=1.
    2: {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 5},
    # key 512: mirror key 256's measured fused config.
    512: {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64,
          "GROUP_SIZE_M": 1, "num_warps": 8, "num_stages": 4},
    # keys 2048/8192/16384: mirror the key-4096 winner.
    2048: {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64,
           "GROUP_SIZE_M": 1, "num_warps": 8, "num_stages": 4},
    8192: {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64,
           "GROUP_SIZE_M": 1, "num_warps": 8, "num_stages": 4},
    16384: {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 1, "num_warps": 8, "num_stages": 4},
}


def fused_config(config: dict, key) -> dict:
    """Derive the fused kernel's config from the tuned one.

    The fused kernel keeps two fp32 accumulators and two B tiles in flight,
    doubling register and shared-memory pressure per program, so BLOCK_SIZE_N
    is capped at 128 and num_stages reduced until the software-pipelined
    shared-memory footprint fits comfortably. BLOCK_SIZE_M must match the
    down GEMM's because both share the block-aligned token sort.
    """
    if key is not None and key in FUSED_OVERRIDES:
        return dict(FUSED_OVERRIDES[key])
    cfg = dict(config)
    cfg["BLOCK_SIZE_N"] = min(cfg["BLOCK_SIZE_N"], 128)
    bm, bn, bk = cfg["BLOCK_SIZE_M"], cfg["BLOCK_SIZE_N"], cfg["BLOCK_SIZE_K"]
    stage_bytes = 2 * (bm * bk + 2 * bk * bn)
    while cfg["num_stages"] > 2 and stage_bytes * cfg["num_stages"] > 200 * 1024:
        cfg["num_stages"] -= 1
    return cfg


# Token count at and above which the down GEMM loads weights through the
# host-built TMA descriptor. Below it the launch is byte-identical to
# fuse-act's; prefill-wave measured TMA at 0.99x at M=256, so the gate stays
# at 1024.
_TMA_MIN_M = 1024

# prefill-wave's M >= 6144 launch config: its sub-04 probe measured TMA plus
# the table's 128x128x64/G1/s3 entry at 0.961x on dev 4096 while TMA plus the
# 256-wide tile below scored 1.028x against the identical reference, so the
# production prefill shapes take the 256-wide tile for the down GEMM instead
# of inheriting the table's 128-wide one. BLOCK_SIZE_M stays 128, matching
# the table keys it replaces and the fused kernel's override.
_DOWN_TMA_OVERRIDE_MIN_M = 6144
_DOWN_TMA_OVERRIDE = {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256,
                      "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 16,
                      "num_warps": 8, "num_stages": 4}


def launch_fused(A, B, C, sorted_token_ids, expert_ids,
                 num_tokens_post_padded, top_k, config, use_tma=False):
    """First launch: gate/up GEMM with silu-and-mul epilogue into C."""
    EM = sorted_token_ids.size(0)
    if A.size(0) < config["BLOCK_SIZE_M"]:
        EM = min(EM, A.size(0) * top_k * config["BLOCK_SIZE_M"])
    inter = B.size(1) // 2

    grid = lambda META: (  # noqa: E731
        triton.cdiv(EM, META["BLOCK_SIZE_M"])
        * triton.cdiv(inter, META["BLOCK_SIZE_N"]),
    )
    if not use_tma:
        fused_gateup_silu_kernel[grid](
            A, B, C,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            inter, A.size(1), EM, A.size(0) * top_k,
            A.stride(0), A.stride(1),
            B.stride(0), B.stride(2), B.stride(1),
            C.stride(0), C.stride(1),
            top_k=top_k,
            compute_type=tl.bfloat16,
            **config,
        )
        return

    # Host-built 3D descriptor over the full (E, 2*inter, K) stacked gate/up
    # weights; the kernel reads the gate tile at column pid_n*BN and the up
    # tile inter columns further along from the same descriptor.
    b_desc = TensorDescriptor.from_tensor(
        B, [1, config["BLOCK_SIZE_N"], config["BLOCK_SIZE_K"]]
    )
    fused_gateup_silu_kernel_tma[grid](
        A, b_desc, C,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        inter, A.size(1), EM, A.size(0) * top_k,
        A.stride(0), A.stride(1),
        C.stride(0), C.stride(1),
        top_k=top_k,
        compute_type=tl.bfloat16,
        **config,
    )


def launch_down_atomic(A, B, C, topk_weights, sorted_token_ids,
                       expert_ids, num_tokens_post_padded, out_topk,
                       config, use_tma=False):
    """Down GEMM into the pre-zeroed (M, K) output via atomic accumulation.

    A is (M * top_k, inter); C is the final (M, K) output. num_valid_tokens
    is A.size(0): the same M * top_k real slots as the plain second launch.
    Weight loads go through the pointer path or the host-built descriptor.
    """
    EM = sorted_token_ids.size(0)
    if A.size(0) < config["BLOCK_SIZE_M"]:
        EM = min(EM, A.size(0) * config["BLOCK_SIZE_M"])
    N = B.size(1)

    grid = lambda META: (  # noqa: E731
        triton.cdiv(EM, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    if not use_tma:
        moe_down_gemm_atomic_kernel[grid](
            A, B, C,
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            N, A.size(1), EM, A.size(0),
            A.stride(0), A.stride(1),
            B.stride(0), B.stride(2), B.stride(1),
            C.stride(0), C.stride(1),
            OUT_TOPK=out_topk,
            compute_type=tl.bfloat16,
            **config,
        )
        return

    b_desc = TensorDescriptor.from_tensor(
        B, [1, config["BLOCK_SIZE_N"], config["BLOCK_SIZE_K"]]
    )
    moe_down_gemm_atomic_tma[grid](
        A, b_desc, C,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        N, A.size(1), EM, A.size(0),
        A.stride(0), A.stride(1),
        C.stride(0), C.stride(1),
        OUT_TOPK=out_topk,
        compute_type=tl.bfloat16,
        **config,
    )


def moe(hidden_states, w1, w2, topk_weights, topk_ids):
    M, K = hidden_states.shape
    E = w1.size(0)
    N = w1.size(1)            # 2 * moe_intermediate
    top_k = topk_ids.size(1)
    inter = N // 2

    config, key = default_config(M, E)

    # Down GEMM config: the tuned table entry, except at production prefill
    # sizes where prefill-wave's measured override applies under TMA.
    down_config = config
    if M >= _DOWN_TMA_OVERRIDE_MIN_M:
        down_config = dict(_DOWN_TMA_OVERRIDE)
    use_tma_down = (
        M >= _TMA_MIN_M
        and inter % down_config["BLOCK_SIZE_K"] == 0
    )
    fused_cfg = fused_config(config, key)
    # The fused TMA path drops all K masks and offsets the up tile by inter
    # columns, so it requires K % BK == 0 and inter % BN == 0 (a gate tile
    # must never straddle into the up columns).
    use_tma_fused = (
        M >= _TMA_MIN_M
        and K % fused_cfg["BLOCK_SIZE_K"] == 0
        and inter % fused_cfg["BLOCK_SIZE_N"] == 0
    )

    # Both kernels share the block-aligned token sort, so their BLOCK_SIZE_M
    # values are equal by construction (table, FUSED_OVERRIDES, and the TMA
    # override all agree per key).
    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids, down_config["BLOCK_SIZE_M"], E
    )

    cache2 = torch.empty((M * top_k, inter), device=hidden_states.device,
                         dtype=hidden_states.dtype)

    launch_fused(hidden_states, w1, cache2, sorted_token_ids, expert_ids,
                 num_tokens_post_padded, top_k, fused_cfg,
                 use_tma=use_tma_fused)

    # fuse-reduce's validated epilogue: the down GEMM accumulates straight
    # into the zeroed output with relaxed bf16 atomics; the memset replaces
    # the moe_sum launch and the (M, top_k, K) round trip disappears.
    out = torch.zeros((M, K), device=hidden_states.device,
                      dtype=hidden_states.dtype)
    launch_down_atomic(cache2, w2, out, topk_weights, sorted_token_ids,
                       expert_ids, num_tokens_post_padded, top_k,
                       down_config, use_tma=use_tma_down)
    return out
