"""Search the Triton config space for the MoE kernel, per token count.

vLLM picks a config for the fused MoE kernel by loading a JSON file keyed by
expert count and expert width, and falls back to a generic heuristic when no
file matches. No file ships for E=128, N=768 (Qwen3-30B-A3B on H100), which is
why the baseline latency curve is non-monotonic.

This produces the missing file, in vLLM's exact format, so the result is
directly comparable to what vLLM's own tuning script would generate and can be
upstreamed as-is.

The search runs in two stages, because a single-stage search selects noise.
Ranking every config with a cheap measurement and keeping the fastest picks
whichever config got the most favourable draw, so its reported speedup does not
survive re-measurement. Measured directly: a one-stage search overstated its
own speedup at all fifteen token counts it tuned, by up to 0.11x, and one of
its picks was slower than the config it replaced.

Stage one ranks the sampled configs cheaply. Stage two re-measures only the
survivors using the harness's own timing function across its fixed routing
draws, and selection happens on stage-two numbers. vLLM's own choice is carried
through both stages so the comparison is always like for like.
"""

import argparse
import itertools
import json
import os
import random
import statistics
import sys
import time

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

import torch  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "harness"))
from bench import TIMING_SEEDS, time_candidate  # noqa: E402
from problems import make_inputs  # noqa: E402

SEARCH_SPACE = {
    "BLOCK_SIZE_M": [16, 32, 64, 128],
    "BLOCK_SIZE_N": [32, 64, 128, 256],
    "BLOCK_SIZE_K": [64, 128, 256],
    "GROUP_SIZE_M": [1, 16, 32],
    "num_warps": [4, 8],
    "num_stages": [2, 3, 4, 5],
}

# Hopper shared memory per block, bytes. Configs that would exceed it fail to
# compile, so they are filtered before costing a JIT compilation.
SMEM_LIMIT = 227 * 1024


def smem_bytes(cfg, itemsize=2):
    """Bytes of shared memory the pipelined K-loop needs for A and B tiles."""
    a = cfg["BLOCK_SIZE_M"] * cfg["BLOCK_SIZE_K"]
    b = cfg["BLOCK_SIZE_K"] * cfg["BLOCK_SIZE_N"]
    return cfg["num_stages"] * (a + b) * itemsize


def enumerate_configs():
    keys = list(SEARCH_SPACE)
    out = []
    for values in itertools.product(*(SEARCH_SPACE[k] for k in keys)):
        cfg = dict(zip(keys, values))
        cfg["SPLIT_K"] = 1
        if smem_bytes(cfg) <= SMEM_LIMIT:
            out.append(cfg)
    return out


def install_config_override():
    """Force every MoE launch to use a config this module chooses.

    Returns a setter. vLLM binds try_get_optimal_moe_config into several
    module namespaces by from-import, so every binding is replaced.
    """
    import vllm.model_executor.layers.fused_moe.fused_moe as fm
    import vllm.model_executor.layers.fused_moe.experts.triton_moe  # noqa: F401

    original = fm.try_get_optimal_moe_config
    state = {"config": None}

    def override(*args, **kwargs):
        if state["config"] is None:
            return original(*args, **kwargs)
        return dict(state["config"])

    NAME = "try_get_optimal_moe_config"
    patched = 0
    for mod_name, mod in list(sys.modules.items()):
        if mod is None or not mod_name.startswith("vllm"):
            continue
        if getattr(mod, NAME, None) is original:
            setattr(mod, NAME, override)
            patched += 1
    assert patched, f"could not find {NAME} bound in any vllm module"
    print(f"config override installed in {patched} module(s)", flush=True)

    def set_config(cfg):
        state["config"] = cfg

    return set_config, original


def time_config(fn, args, reps, warmup, flush):
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        flush.zero_()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn(*args)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shapes", default="results/shapes.json")
    parser.add_argument("--out", required=True)
    parser.add_argument("--budget", type=int, default=120,
                        help="configs sampled per token count")
    parser.add_argument("--reps", type=int, default=9,
                        help="reps during stage-one ranking")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--finalists", type=int, default=12,
                        help="configs carried into stage-two validation")
    parser.add_argument("--val-reps", type=int, default=30,
                        help="reps per routing draw during validation")
    parser.add_argument("--val-warmup", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tokens", type=str, default=None,
                        help="comma-separated token counts; default is the "
                             "benchmark sweep")
    args = parser.parse_args()

    assert torch.cuda.is_available(), "tuning requires a GPU"

    from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts

    spec = json.load(open(args.shapes))
    model = spec["model"]
    K = model["hidden_size"]
    inter = model["moe_intermediate_size"]
    E = model["num_experts"]
    top_k = model["top_k"]
    dtype = getattr(torch, model["dtype"])

    if args.tokens:
        token_counts = [int(t) for t in args.tokens.split(",")]
    else:
        token_counts = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048,
                        4096, 8192, 16384]

    set_config, original_chooser = install_config_override()
    all_configs = enumerate_configs()
    print(f"{len(all_configs)} configs pass the shared-memory filter", flush=True)

    rng = random.Random(args.seed)
    flush = torch.empty(int(64e6 // 4), dtype=torch.float32, device="cuda")

    def moe(*a):
        return fused_experts(*a)

    tuned = {}
    for M in token_counts:
        inputs = make_inputs(M, K, inter, E, top_k, seed=1234 + M, dtype=dtype)

        # vLLM's own choice for this shape, always evaluated so the search
        # cannot report a win that is really a measurement difference.
        set_config(None)
        w1_shape = tuple(inputs[1].shape)
        w2_shape = tuple(inputs[2].shape)
        default_cfg = original_chooser(w1_shape, w2_shape, top_k, None, M, None)
        try:
            default_ms = time_config(moe, inputs, args.reps, args.warmup, flush)
        except Exception as exc:  # noqa: BLE001
            print(f"M={M}: default config failed: {exc}", flush=True)
            continue

        sample = rng.sample(all_configs, min(args.budget, len(all_configs)))
        t0 = time.perf_counter()

        # Stage one: cheap ranking. These numbers choose who advances, never
        # who wins.
        ranked = []
        for cfg in sample:
            set_config(cfg)
            try:
                ms = time_config(moe, inputs, args.reps, args.warmup, flush)
            except Exception:  # noqa: BLE001
                # Out of resources, unsupported tile, or a compile failure.
                continue
            ranked.append((ms, cfg))
        ranked.sort(key=lambda r: r[0])
        finalists = [cfg for _, cfg in ranked[: args.finalists]]
        stage1_best = ranked[0][0] if ranked else float("inf")

        # Stage two: re-measure survivors and the incumbent with the harness's
        # own timing, across its fixed routing draws.
        def validate(cfg):
            set_config(cfg)
            per_draw = []
            for ts in TIMING_SEEDS:
                targs = make_inputs(M, K, inter, E, top_k, seed=ts, dtype=dtype)
                ms, _ = time_candidate(moe, targs, reps=args.val_reps,
                                       warmup=args.val_warmup)
                per_draw.append(ms)
            return statistics.median(per_draw)

        default_val = validate(None)
        best_cfg, best_val = None, default_val
        for cfg in finalists:
            try:
                v = validate(cfg)
            except Exception:  # noqa: BLE001
                continue
            if v < best_val:
                best_cfg, best_val = dict(cfg), v
        set_config(None)

        # Keeping vLLM's choice is a legitimate outcome: it means the search
        # found nothing that survives re-measurement.
        kept_default = best_cfg is None
        if kept_default:
            best_cfg = dict(default_cfg)

        speedup = default_val / best_val
        optimistic = default_ms / stage1_best if stage1_best else float("nan")
        tuned[M] = {
            "config": best_cfg,
            "ms": best_val,
            "default_ms": default_val,
            "speedup": speedup,
            "stage1_speedup": optimistic,
            "kept_default": kept_default,
            "configs_ran": len(ranked),
            "finalists": len(finalists),
        }
        print(
            f"M={M:<6} default {default_val:7.3f} ms -> best {best_val:7.3f} ms "
            f"({speedup:4.2f}x, stage-1 claimed {optimistic:4.2f}x)  "
            f"{len(ranked)}/{len(sample)} ran, {len(finalists)} validated, "
            f"{time.perf_counter() - t0:5.0f}s"
            + ("  [kept vLLM default]" if kept_default else ""),
            flush=True,
        )
        print(f"        best: {best_cfg}", flush=True)

    # vLLM's format: one object per token count, keyed by the count as a string.
    config_file = {
        str(M): {k: v for k, v in r["config"].items()}
        for M, r in sorted(tuned.items())
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(config_file, f, indent=4)
        f.write("\n")

    detail = args.out.replace(".json", "_search.json")
    with open(detail, "w") as f:
        json.dump(
            {
                "device": torch.cuda.get_device_name(0),
                "budget": args.budget,
                "seed": args.seed,
                "results": {str(m): r for m, r in sorted(tuned.items())},
            },
            f,
            indent=1,
        )

    gains = [r["speedup"] for r in tuned.values()]
    if gains:
        curse = [r["stage1_speedup"] - r["speedup"] for r in tuned.values()]
        kept = sum(1 for r in tuned.values() if r["kept_default"])
        print(f"\nwrote {args.out}")
        print(f"validated speedup over vLLM's heuristic: "
              f"min {min(gains):.2f}x, median {statistics.median(gains):.2f}x, "
              f"max {max(gains):.2f}x")
        print(f"stage-one optimism: median {statistics.median(curse):+.3f}x, "
              f"max {max(curse):+.3f}x")
        print(f"kept vLLM's own config at {kept} of {len(tuned)} token counts")


if __name__ == "__main__":
    main()
