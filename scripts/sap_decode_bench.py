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


def smoke_model(device, mode="p1_discrete"):
    from nanochat.gpt import GPT, GPTConfig
    cfg = GPTConfig(sequence_len=512, vocab_size=1024, n_layer=2, n_head=2, n_kv_head=2,
                    n_embd=128, window_pattern="L", sap_block_T=4, sap_block_mode=mode)
    with torch.device("meta"):
        model = GPT(cfg)
    model.to_empty(device=device)
    model.init_weights()
    return model


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint-dir", type=str, default=None)
    p.add_argument("--step", type=int, default=None, help="default: the last saved step")
    p.add_argument("--tokenizer-dir", type=str, default=None)
    p.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 16, 128])
    p.add_argument("--prompt-len", type=int, default=64)
    p.add_argument("--gen-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--override-mode", type=str, default="", help="override block head mode (e.g. local_jacobi)")
    p.add_argument("--jacobi-sweeps", type=int, default=None, help="override jacobi sweeps")
    p.add_argument("--no-graphs", action="store_true", help="eager loops only")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    device = torch.device(args.device)

    if args.smoke:
        model = smoke_model(device, mode=args.override_mode or "p1_discrete")
        args.batch_sizes, args.gen_tokens, args.repeats = [1, 4], 32, 1
    else:
        from nanochat.checkpoint_manager import build_model, find_last_step
        step = args.step if args.step is not None else find_last_step(args.checkpoint_dir)
        model, _tok, _meta = build_model(args.checkpoint_dir, step, device, phase="eval",
                                         tokenizer_dir=args.tokenizer_dir)
    model.eval()
    assert model.sap_head is not None, "this checkpoint has no block head (sap_block_T=0)"
    if args.override_mode:
        model.sap_head.mode = args.override_mode
    if args.jacobi_sweeps is not None:
        model.sap_head.jacobi_sweeps = args.jacobi_sweeps
    T = model.sap_head.T

    g = torch.Generator().manual_seed(0)
    results = {"block_T": T, "mode": model.sap_head.mode, "gen_tokens": args.gen_tokens,
               "prompt_len": args.prompt_len, "device": str(device), "rows": []}
    use_graphs = device.type == "cuda" and not args.no_graphs
    for B in args.batch_sizes:
        prompts = torch.randint(0, model.config.vocab_size, (B, args.prompt_len), generator=g)
        row = {"batch": B}
        try:
            t_ar = time_loop(lambda: generate_ar_kv(model, prompts, args.gen_tokens,
                                                    temperature=args.temperature),
                             device, args.warmup, args.repeats)
            t_blk = time_loop(lambda: generate_block_kv(model, prompts, args.gen_tokens,
                                                        temperature=args.temperature),
                              device, args.warmup, args.repeats)
            row.update(ar_tok_per_s=B * args.gen_tokens / t_ar,
                       block_tok_per_s=B * args.gen_tokens / t_blk, speedup=t_ar / t_blk)
            if use_graphs:
                try:
                    tg_ar = min(graph_ar_seconds(model, prompts, args.gen_tokens, args.temperature)
                                for _ in range(args.repeats))
                    tg_blk = min(graph_block_seconds(model, prompts, args.gen_tokens, args.temperature)
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
