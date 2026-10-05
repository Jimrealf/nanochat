"""
Decode throughput: the SAP block head against plain next-token decoding, same model.

Both loops live in nanochat/engine.py and are symmetric on purpose: same prefill, same KV
cache, no tool-use state machine, and the same unembedding work per generated token (one
lm_head row per token for next-token decoding, T rows per block for the head). What differs
is the number of sequential trunk passes: one per token against one per T tokens. Reported
at batch 1, 16 and 128, because a fixed-length block keeps every row of a batch in step,
which is exactly where speculative decoding loses its speedup.

Two regimes are timed. Eager is what nanochat's Engine does today, and it is host-bound: per
kernel-launch overhead, not weights or FLOPs, sets the speed (LEARNINGS: dense decode is ~1000x
off the bandwidth floor). CUDA graphs capture one decode step and replay it, which is how a
real server runs; there the comparison is the work per generated token. Both loops get the
same treatment, and the graph steps use the KV cache's graph-safe path (positions read on the
device, no .item()).

    python -m scripts.sap_decode_bench --checkpoint-dir out/s00_sap/d8/SAP_p1_discrete_T4_s1
    python -m scripts.sap_decode_bench --smoke        # random tiny model, checks the code path
"""

from __future__ import annotations

import argparse
import json
import time

import torch

from nanochat.block_head import pick
from nanochat.engine import _kv_cache_for, generate_ar_kv, generate_block_kv


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def time_loop(fn, device, warmup, repeats):
    for _ in range(warmup):
        fn()
    _sync(device)
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        _sync(device)
        best = min(best, time.perf_counter() - t0)
    return best


def _capture(step, warmup):
    """Warm the step up on a side stream (allocator and autotuning), then capture it once."""
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(warmup):
            step()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()
    return graph


def _sample_token(logits, temperature):
    if temperature <= 0:
        return logits.argmax(-1)
    probs = torch.softmax(logits.float() / temperature, dim=-1)
    return pick(probs, torch.rand(probs.shape[0], device=probs.device))


@torch.inference_mode()
def graph_ar_seconds(model, prompts, n_tokens, temperature, warmup=3):
    """Next-token decoding, one captured step per token. Returns seconds for n_tokens steps."""
    dev = model.get_device()
    dtype = torch.bfloat16 if dev.type == "cuda" else torch.float32
    B, L = prompts.shape
    kv = _kv_cache_for(model, B, L + n_tokens + warmup + 4, dev, dtype)
    kv.graph_safe = True
    logits = model.forward(prompts.to(dev), kv_cache=kv)[:, -1, :]
    ids = _sample_token(logits, temperature)[:, None].contiguous()

    def step():
        lg = model.forward(ids, kv_cache=kv)[:, -1, :]
        ids.copy_(_sample_token(lg, temperature)[:, None])

    graph = _capture(step, warmup)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_tokens):
        graph.replay()
    torch.cuda.synchronize()
    return time.perf_counter() - t0


@torch.inference_mode()
def graph_block_seconds(model, prompts, n_tokens, temperature, warmup=3):
    """Block decoding, one captured step per T tokens. Returns seconds for the n_tokens."""
    head = model.sap_head
    T, W = head.T, head.W
    dev = model.get_device()
    dtype = torch.bfloat16 if dev.type == "cuda" else torch.float32
    B, L = prompts.shape
    assert W == 0 or L >= W, f"prompt ({L}) shorter than the head's window ({W})"
    n_blocks = -(-n_tokens // T)
    kv = _kv_cache_for(model, B, L + (n_blocks + warmup + 2) * T + 4, dev, dtype)
    kv.graph_safe = True
    if head.mode in ("depth_local", "depth_tree", "depth_roll"):
        first = head.depth_layer_ids[0]
        ckv = model.sap_depth_copy_cache(B, kv.k_cache.size(2), dev, kv.k_cache.dtype)
        x, st = model.forward(prompts.to(dev), kv_cache=kv, skip_logits=True, sap_capture=True)
        model.sap_depth_extend_copy_cache(ckv, prompts.to(dev), st, kv)
        state = model.sap_depth_state(st).clone()
        last = x[:, -1].clone()
        blk = model._sap_depth_sample(kv, state, model._sap_readout(last), temperature=temperature,
                                      copy_kv=ckv, top_first=last).contiguous()

        def step():
            xs, sts = model.forward(blk, kv_cache=kv, skip_logits=True, sap_capture=True)
            model.sap_depth_extend_copy_cache(ckv, blk, sts, kv)
            state.copy_(model.sap_depth_state(sts))
            last.copy_(xs[:, -1])
            blk.copy_(model._sap_depth_sample(kv, state, model._sap_readout(last), temperature=temperature,
                                              copy_kv=ckv, top_first=last))

        graph = _capture(step, warmup)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_blocks):
            graph.replay()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) * n_tokens / (n_blocks * T)
    x = model.forward(prompts.to(dev), kv_cache=kv, skip_logits=True)
    hist = x[:, -W:].clone() if W > 0 else None
    valid = torch.ones(B, W, dtype=torch.bool, device=dev) if W > 0 else None
    kw = dict(embed=model.transformer.wte, embed_table=model.transformer.wte.weight,
              temperature=temperature)
    blk = head.sample(x[:, -1], hist, valid, model._sap_readout, **kw).contiguous()

    def step():
        xs = model.forward(blk, kv_cache=kv, skip_logits=True)
        if W > 0:
            hist.copy_(torch.cat([hist, xs], dim=1)[:, -W:])
        blk.copy_(head.sample(xs[:, -1], hist, valid, model._sap_readout, **kw))

    graph = _capture(step, warmup)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_blocks):
        graph.replay()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * n_tokens / (n_blocks * T)


def smoke_model(device, mode="p1_discrete", nce_props=0):
    from nanochat.gpt import GPT, GPTConfig
    layers = 3 if mode in ("sir_conf", "sir_pyramid", "sir_tree", "sir_full") else 2
    cfg = GPTConfig(sequence_len=512, vocab_size=1024, n_layer=2, n_head=2, n_kv_head=2,
                    n_embd=128, window_pattern="L", sap_block_T=4, sap_block_mode=mode,
                    sap_head_layers=layers, sap_nce_props=nce_props)
    with torch.device("meta"):
        model = GPT(cfg)
    model.to_empty(device=device)
    model.init_weights()
    return model


@torch.inference_mode()
def graph_lane_seconds(model, prompts, L, S, lane_token, temperature, warmup=3):
    """Lane decoding, one captured lockstep step (L tokens) per replay. Returns seconds for the
    S steps that write L*S tokens; the prompt pass and its first token are excluded, as for AR."""
    from nanochat.lanes import _choose, check_lane_windows, lane_step
    dev = model.get_device()
    dtype = torch.bfloat16 if dev.type == "cuda" else torch.float32
    B, P = prompts.shape
    n_rows = P + max(S, warmup + 1) * L                 # warmup reuses these slots: the cache is rewound
    check_lane_windows(model, n_rows)
    kv = _kv_cache_for(model, B, n_rows + 8, dev, dtype)
    kv.graph_safe = True
    logits = model.forward(prompts.to(dev), kv_cache=kv)[:, -1]
    base = kv.cache_seqlens.clone()
    ids = torch.full((B, L), lane_token, dtype=torch.long, device=dev)
    ids[:, 0] = _choose(logits, temperature, None)
    ids0 = ids.clone()
    lane0 = torch.arange(L, device=dev)[None] * S

    def step():
        s = (kv.cache_seqlens.to(torch.long) - P) // L                  # this step's index, per row
        lg = lane_step(model, kv, ids, P + lane0 + s[:, None])
        ids.copy_(_choose(lg, temperature, None))

    graph = _capture(step, warmup)
    kv.cache_seqlens.copy_(base)
    ids.copy_(ids0)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(S):
        graph.replay()
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def lane_bench(model, args, device, lane_token):
    """Lane decoding against next-token decoding of the same model, both under CUDA graphs at
    graph temperature, for L*S generated tokens per row."""
    L = args.lanes
    S = max(1, (args.gen_tokens - 1) // L)
    n = L * S
    g = torch.Generator().manual_seed(0)
    results = {"mode": "lanes", "lanes": L, "lane_len": S, "gen_tokens": n, "prompt_len": args.prompt_len,
               "device": str(device), "graph_temperature": args.graph_temperature, "rows": []}
    for B in args.batch_sizes:
        prompts = torch.randint(0, model.config.vocab_size, (B, args.prompt_len), generator=g)
        if device.type != "cuda" or args.no_graphs:
            print("lane timing needs CUDA graphs; skipped")
            break
        ta = min(graph_ar_seconds(model, prompts, n, args.graph_temperature) for _ in range(args.repeats))
        tl = min(graph_lane_seconds(model, prompts, L, S, lane_token, args.graph_temperature)
                 for _ in range(args.repeats))
        row = {"batch": B, "graph_ar_tok_per_s": B * n / ta, "graph_lane_tok_per_s": B * n / tl,
               "graph_speedup": ta / tl}
        results["rows"].append(row)
        print(f"batch {B:4d} | {n} tokens | next-token {row['graph_ar_tok_per_s']:10,.0f} tok/s | "
              f"{L} lanes {row['graph_lane_tok_per_s']:10,.0f} tok/s | {row['graph_speedup']:.2f}x", flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"wrote {args.out}")


@torch.inference_mode()
def graph_wb_seconds(model, prompts, steps, mask_token, temperature, warmup=2):
    """S11 two-stream decoding (window bisection or bridged lanes), one captured CUDA graph per
    step (step shapes differ), replayed in order. Returns seconds for every step after the prompt;
    the prompt's prefill is excluded, as for next-token decoding."""
    from nanochat.wbisect import WBDecoder
    dec = WBDecoder(model, prompts.size(0), steps, mask_token, temperature)
    dec.prefill(prompts)
    base = dec.kv.cache_seqlens.clone()
    graphs, pool = [], None
    for s in range(len(dec.plan)):
        snap, out_snap = dec.kv.cache_seqlens.clone(), dec.out.clone()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(warmup):
                dec.kv.cache_seqlens.copy_(snap)
                dec.step(s)
        torch.cuda.current_stream().wait_stream(side)
        dec.kv.cache_seqlens.copy_(snap)
        dec.out.copy_(out_snap)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=pool):
            dec.step(s)
        pool = graph.pool()
        graphs.append(graph)
        dec.kv.cache_seqlens.copy_(snap)                  # capture records without running: run it once
        dec.out.copy_(out_snap)
        graph.replay()
    dec.kv.cache_seqlens.copy_(base)
    dec.out[:, dec.P:] = 0
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for graph in graphs:
        graph.replay()
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def wb_bench(model, args, device, mask_token):
    """Two-stream decoding of the whole block against next-token decoding of the same number of
    tokens with the same model shape, both under CUDA graphs (greedy at graph temperature 0)."""
    from nanochat.wbisect import bridged_lanes_steps, lane_order_steps, seeded_lanes_steps, wb_steps
    N = model.config.sequence_len if not args.smoke else 64
    P = args.wb_prefix if not args.smoke else 16
    if args.wb_order == "lanes":
        steps = lane_order_steps(N, P, args.wb_lanes)
    elif args.wb_order == "seeded":
        steps = seeded_lanes_steps(N, P, args.wb_lanes)
    elif args.wb_lanes:
        steps = bridged_lanes_steps(N, P, args.wb_lanes, args.wb_window)
    else:
        steps = wb_steps(N, P, args.wb_window)
    n = N - P
    S = int(steps.max()) + 1
    g = torch.Generator().manual_seed(0)
    mode = {"lanes": "two_stream_lanes", "seeded": "seeded_lanes"}.get(args.wb_order) or \
        ("bridged_lanes" if args.wb_lanes else "window_bisection")
    results = {"mode": mode, "wb_window": args.wb_window,
               "wb_lanes": args.wb_lanes, "prefix": P, "gen_tokens": n, "steps": S, "device": str(device),
               "graph_temperature": args.graph_temperature, "rows": []}
    print(f"{results['mode']} n={args.wb_window} L={args.wb_lanes}: {S} steps for {n} tokens after a {P}-token prompt")
    for B in args.batch_sizes:
        prompts = torch.randint(0, model.config.vocab_size, (B, P), generator=g)
        if device.type != "cuda" or args.no_graphs:
            print("two-stream timing needs CUDA graphs; skipped")
            break
        ta = min(graph_ar_seconds(model, prompts, n, args.graph_temperature) for _ in range(args.repeats))
        tw = min(graph_wb_seconds(model, prompts, steps, mask_token, args.graph_temperature)
                 for _ in range(args.repeats))
        row = {"batch": B, "graph_ar_tok_per_s": B * n / ta, "graph_wb_tok_per_s": B * n / tw,
               "graph_speedup": ta / tw, "ar_seconds": ta, "wb_seconds": tw}
        results["rows"].append(row)
        print(f"batch {B:4d} | {n} tokens | next-token {row['graph_ar_tok_per_s']:10,.0f} tok/s | "
              f"{S}-step two-stream {row['graph_wb_tok_per_s']:10,.0f} tok/s | {row['graph_speedup']:.2f}x", flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"wrote {args.out}")


@torch.inference_mode()
def graph_onepass_seconds(model, prompts, G, temperature, warmup=2, repeats=3):
    """S13 Q3: one pass that computes all G rows of a block on top of the prompt's cache and draws
    a token at every row (a one-pass generator's cost: G rows of trunk and unembedding work).
    One captured graph; the cache is rewound before each replay. Returns the best seconds."""
    dev = model.get_device()
    dtype = torch.bfloat16 if dev.type == "cuda" else torch.float32
    B, P = prompts.shape
    kv = _kv_cache_for(model, B, P + G + 8, dev, dtype)
    kv.graph_safe = True
    model.forward(prompts.to(dev), kv_cache=kv)
    base = kv.cache_seqlens.clone()
    ids = torch.randint(0, model.config.vocab_size, (B, G), device=dev)

    def step():                                         # rewinds the cache first, inside the graph
        kv.cache_seqlens.copy_(base)
        lg = model.forward(ids, kv_cache=kv)
        ids.copy_(_sample_token(lg.reshape(B * G, -1), temperature).view(B, G))

    graph = _capture(step, warmup)
    best = float("inf")
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        graph.replay()
        torch.cuda.synchronize()
        best = min(best, time.perf_counter() - t0)
    return best


def roofline(args, device):
    """S13 Q3: is one pass worth it? Times, on a random-weight model of the given depth (nanochat
    shape: width 64 x depth, 128-dim heads), next-token decoding, lockstep lanes at R rounds
    (L = G / R lanes) and a single G-row pass, all under CUDA graphs, for G generated tokens after
    a P-token prompt. Weights do not change the timing, only the shape and the schedule do."""
    from nanochat.gpt import GPT, GPTConfig
    D = args.roofline
    d = 64 * D
    cfg = GPTConfig(sequence_len=2048, vocab_size=32768, n_layer=D, n_head=max(1, d // 128),
                    n_kv_head=max(1, d // 128), n_embd=d, window_pattern="L")
    with torch.device("meta"):
        model = GPT(cfg)
    model.to_empty(device=device)
    model.init_weights()
    model.eval()
    G = args.gen_tokens
    rounds = [int(r) for r in args.rounds.split(",")]
    g = torch.Generator().manual_seed(0)
    results = {"mode": "roofline", "depth": D, "n_embd": d, "gen_tokens": G, "prompt_len": args.prompt_len,
               "rounds": rounds, "rows": []}
    for B in args.batch_sizes:
        prompts = torch.randint(0, cfg.vocab_size, (B, args.prompt_len), generator=g)
        row = {"batch": B}
        try:
            ta = min(graph_ar_seconds(model, prompts, G, args.graph_temperature) for _ in range(args.repeats))
            row["ar_seconds"] = ta
            for R in rounds:
                L = G // R
                tl = min(graph_lane_seconds(model, prompts, L, R, 0, args.graph_temperature)
                         for _ in range(args.repeats))
                row[f"lanes_R{R}_seconds"] = tl * G / (L * R)
            row["onepass_seconds"] = graph_onepass_seconds(model, prompts, G, args.graph_temperature,
                                                           repeats=args.repeats)
        except torch.cuda.OutOfMemoryError:
            row["oom"] = True
            torch.cuda.empty_cache()
        results["rows"].append(row)
        if "onepass_seconds" in row:
            one = row["onepass_seconds"]
            print(f"d{D} batch {B:3d} | {G} tokens | next-token {row['ar_seconds'] * 1e3:8.1f} ms | " +
                  " | ".join(f"{R} rounds {row[f'lanes_R{R}_seconds'] * 1e3:7.2f} ms "
                             f"({row['ar_seconds'] / row[f'lanes_R{R}_seconds']:6.1f}x)" for R in rounds) +
                  f" | one pass {one * 1e3:7.2f} ms ({row['ar_seconds'] / one:6.1f}x; "
                  f"= {one / (row[f'lanes_R{rounds[0]}_seconds'] / rounds[0]):4.1f} rounds of the {rounds[0]}-round schedule)",
                  flush=True)
        else:
            print(f"d{D} batch {B}: out of memory", flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"wrote {args.out}")


def ar_only(model, args, device):
    """Next-token decoding speed alone (eager and CUDA graphs), for the dense depth frontier."""
    g = torch.Generator().manual_seed(0)
    results = {"block_T": 0, "mode": "dense", "n_layer": model.config.n_layer, "n_embd": model.config.n_embd,
               "gen_tokens": args.gen_tokens, "prompt_len": args.prompt_len, "device": str(device),
               "graph_temperature": args.graph_temperature, "rows": []}
    for B in args.batch_sizes:
        prompts = torch.randint(0, model.config.vocab_size, (B, args.prompt_len), generator=g)
        t_ar = time_loop(lambda: generate_ar_kv(model, prompts, args.gen_tokens, temperature=args.temperature),
                         device, args.warmup, args.repeats)
        row = {"batch": B, "ar_tok_per_s": B * args.gen_tokens / t_ar}
        if device.type == "cuda" and not args.no_graphs:
            tg = min(graph_ar_seconds(model, prompts, args.gen_tokens, args.graph_temperature)
                     for _ in range(args.repeats))
            row["graph_ar_tok_per_s"] = B * args.gen_tokens / tg
        results["rows"].append(row)
        print(f"batch {B:4d} | next-token eager {row['ar_tok_per_s']:10,.0f} tok/s | graphs "
              f"{row.get('graph_ar_tok_per_s', float('nan')):10,.0f} tok/s", flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"wrote {args.out}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint-dir", type=str, default=None)
    p.add_argument("--step", type=int, default=None, help="default: the last saved step")
    p.add_argument("--tokenizer-dir", type=str, default=None)
    p.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 16, 128])
    p.add_argument("--prompt-len", type=int, default=64)
    p.add_argument("--gen-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--graph-temperature", type=float, default=0.0,
                   help="temperature inside CUDA graphs; default greedy avoids graph-unsafe RNG offsets")
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--override-mode", type=str, default="", help="override block head mode (e.g. local_jacobi)")
    p.add_argument("--jacobi-sweeps", type=int, default=None, help="override jacobi sweeps")
    p.add_argument("--nce-props", type=int, default=None,
                   help="override the self-contrastive head's decode proposals (they cost compute at large batch)")
    p.add_argument("--no-graphs", action="store_true", help="eager loops only")
    p.add_argument("--ar-only", action="store_true",
                   help="time next-token decoding only (a dense checkpoint has no block head; implied then)")
    p.add_argument("--lanes", type=int, default=0,
                   help="time lane decoding with this many lockstep lanes against next-token decoding")
    p.add_argument("--wb-window", type=int, default=0,
                   help="S11: time two-stream decoding (window bisection, or bridged lanes with --wb-lanes)")
    p.add_argument("--wb-lanes", type=int, default=0)
    p.add_argument("--wb-order", type=str, default="bisect", choices=["bisect", "lanes", "seeded"])
    p.add_argument("--wb-prefix", type=int, default=128)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--roofline", type=int, default=0,
                   help="S13 Q3: time next-token, R-round lanes and one pass on a random model of this depth")
    p.add_argument("--rounds", type=str, default="8,16,32,64", help="S13 Q3: round counts for --roofline")
    args = p.parse_args()
    device = torch.device(args.device)
    if args.roofline:
        return roofline(args, device)

    if args.smoke:
        model = smoke_model(device, mode=args.override_mode or "p1_discrete", nce_props=args.nce_props or 0)
        args.batch_sizes, args.gen_tokens, args.repeats = [1, 4], 32, 1
    else:
        from nanochat.checkpoint_manager import build_model, find_last_step
        step = args.step if args.step is not None else find_last_step(args.checkpoint_dir)
        model, _tok, _meta = build_model(args.checkpoint_dir, step, device, phase="eval",
                                         tokenizer_dir=args.tokenizer_dir)
    model.eval()
    if args.wb_window > 0 or args.wb_order != "bisect":
        from nanochat.lanes import LANE_TOKEN
        tok = 15 if args.smoke else _tok.encode_special(LANE_TOKEN)
        return wb_bench(model, args, device, tok)
    if args.lanes > 0:
        from nanochat.lanes import LANE_TOKEN
        tok = 15 if args.smoke else _tok.encode_special(LANE_TOKEN)
        return lane_bench(model, args, device, tok)
    if model.sap_head is None or args.ar_only:
        return ar_only(model, args, device)
    if args.override_mode:
        model.sap_head.mode = args.override_mode
    if args.jacobi_sweeps is not None:
        model.sap_head.jacobi_sweeps = args.jacobi_sweeps
    if args.nce_props is not None:
        assert getattr(model.sap_head, "nce_props", 0) > 0 or args.nce_props == 0, "no NCE scorer in this head"
        model.sap_head.nce_props = args.nce_props
    T = model.sap_head.T

    g = torch.Generator().manual_seed(0)
    results = {"block_T": T, "mode": model.sap_head.mode, "gen_tokens": args.gen_tokens,
               "prompt_len": args.prompt_len, "device": str(device),
               "eager_temperature": args.temperature,
               "graph_temperature": args.graph_temperature,
               "nce_props": getattr(model.sap_head, "nce_props", 0), "rows": []}
    use_graphs = device.type == "cuda" and not args.no_graphs
    for B in args.batch_sizes:
        prompts = torch.randint(0, model.config.vocab_size, (B, args.prompt_len), generator=g)
        row = {"batch": B}
        print(f"batch {B:4d}: starting timing (eager next-token)...", flush=True)
        try:
            t_ar = time_loop(lambda: generate_ar_kv(model, prompts, args.gen_tokens,
                                                    temperature=args.temperature),
                             device, args.warmup, args.repeats)
            print(f"batch {B:4d}: eager next-token done ({B * args.gen_tokens / t_ar:,.0f} tok/s), timing eager block T={T}...", flush=True)
            t_blk = time_loop(lambda: generate_block_kv(model, prompts, args.gen_tokens,
                                                        temperature=args.temperature),
                              device, args.warmup, args.repeats)
            print(f"batch {B:4d}: eager block done ({B * args.gen_tokens / t_blk:,.0f} tok/s)", flush=True)
            row.update(ar_tok_per_s=B * args.gen_tokens / t_ar,
                       block_tok_per_s=B * args.gen_tokens / t_blk, speedup=t_ar / t_blk)
            if use_graphs:
                print(f"batch {B:4d}: capturing and replaying CUDA graphs...", flush=True)
                try:
                    tg_ar = min(graph_ar_seconds(model, prompts, args.gen_tokens,
                                                 args.graph_temperature)
                                for _ in range(args.repeats))
                    tg_blk = min(graph_block_seconds(model, prompts, args.gen_tokens,
                                                     args.graph_temperature)
                                 for _ in range(args.repeats))
                    row.update(graph_ar_tok_per_s=B * args.gen_tokens / tg_ar,
                               graph_block_tok_per_s=B * args.gen_tokens / tg_blk,
                               graph_speedup=tg_ar / tg_blk)
                except Exception as e:  # capture can fail on a kernel that syncs; keep eager
                    row["graph_error"] = repr(e)[:300]
        except torch.cuda.OutOfMemoryError:
            print(f"batch {B}: out of memory, skipped")
            torch.cuda.empty_cache()
            continue
        results["rows"].append(row)
        line = (f"batch {B:4d} | eager: next-token {row['ar_tok_per_s']:10,.0f} | "
                f"block T={T} {row['block_tok_per_s']:10,.0f} tok/s | {row['speedup']:.2f}x")
        if "graph_speedup" in row:
            line += (f" || CUDA graphs: next-token {row['graph_ar_tok_per_s']:10,.0f} | "
                     f"block {row['graph_block_tok_per_s']:10,.0f} tok/s | {row['graph_speedup']:.2f}x")
        elif "graph_error" in row:
            line += f" || CUDA graphs failed: {row['graph_error'][:120]}"
        print(line, flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
