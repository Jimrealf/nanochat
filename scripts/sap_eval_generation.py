"""
Generation quality of the SAP block head against the same model's next-token decoding.

For N validation prefixes, the model continues each prefix twice: with its own next-token
head (one token per pass) and with its block head (T tokens per pass, no verification).
Both continuations are scored by a reference model (the dense arm of the same sweep) and
written out as pairs for an LLM judge.

Which number to trust. Reference perplexity rewards "typical" text, so a sampler that
repeats itself can score well (K-Forcing reports exactly this). The pairs file is therefore
the primary output: judge it pairwise, block versus next-token, and report the win rate.
Repetition (distinct 3-grams) is reported next to the perplexity so a degenerate sampler
is visible here too.

    python -m scripts.sap_eval_generation --checkpoint-dir out/s00_sap/d8/SAP_p1_discrete_T4_s1 \
        --reference-dir out/s00_sap/d8/B1_dense_s1 --out gen_p1.jsonl
    python -m scripts.sap_eval_generation --smoke
"""

from __future__ import annotations

import argparse
import json
import math

import torch
import torch.nn.functional as F

from nanochat.engine import generate_ar_kv, generate_block_kv


@torch.no_grad()
def reference_nll(ref, prefixes, conts):
    """Mean per-token NLL of each continuation given its prefix, under the reference model."""
    seq = torch.cat([prefixes, conts], dim=1)
    logits = ref(seq[:, :-1])
    L = prefixes.size(1)
    lp = torch.log_softmax(logits[:, L - 1:].float(), dim=-1)
    nll = -lp.gather(-1, conts[..., None]).squeeze(-1)
    return nll.mean(dim=1)


def distinct3(row):
    grams = [tuple(row[i:i + 3]) for i in range(len(row) - 2)]
    return len(set(grams)) / max(1, len(grams))


def unigram_entropy(row):
    """Entropy (nats) of the token histogram of one sample, the diversity measure diffusion-LM
    papers report beside generative perplexity (low values flag repetition)."""
    c = torch.bincount(torch.tensor(row)).double()
    p = c[c > 0] / c.sum()
    return float(-(p * p.log()).sum())


def load(checkpoint_dir, step, device, tokenizer_dir=None):
    from nanochat.checkpoint_manager import build_model, find_last_step
    step = step if step is not None else find_last_step(checkpoint_dir)
    model, tok, _ = build_model(checkpoint_dir, step, device, phase="eval", tokenizer_dir=tokenizer_dir)
    return model.eval(), tok


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint-dir", type=str, default=None, help="model with a block head")
    p.add_argument("--reference-dir", type=str, default=None, help="dense model used as the scorer")
    p.add_argument("--step", type=int, default=None)
    p.add_argument("--tokenizer-dir", type=str, default=None)
    p.add_argument("--data-dir", type=str, default=None)
    p.add_argument("--n-prefixes", type=int, default=1024)
    p.add_argument("--prefix-len", type=int, default=64)
    p.add_argument("--gen-tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", type=str, default="sap_generation.jsonl")
    p.add_argument("--override-mode", type=str, default="", help="override block head mode (e.g. local_jacobi)")
    p.add_argument("--jacobi-sweeps", type=int, default=None, help="override jacobi sweeps")
    p.add_argument("--smoke", action="store_true", help="random tiny models, random prefixes")
    p.add_argument("--lanes", type=int, default=0,
                   help="score lane decoding (nanochat/lanes.py) with this many lanes instead of a block head; "
                        "the generated length is 1 + lanes*floor((gen_tokens-1)/lanes)")
    p.add_argument("--ar-dir", type=str, default=None,
                   help="lanes: the model whose next-token samples are the comparison (default: the lane model)")
    p.add_argument("--wb-window", type=int, default=0,
                   help="S11: score two-stream decoding (window bisection, or bridged lanes with --wb-lanes); "
                        "the row is prefix-len + gen-tokens and must match the schedule the model trained on")
    p.add_argument("--wb-lanes", type=int, default=0)
    p.add_argument("--ar-temperature", type=float, default=None, help="next-token side only (default --temperature)")
    p.add_argument("--ar-top-p", type=float, default=None, help="next-token side: nucleus sampling")
    p.add_argument("--ar-rep-penalty", type=float, default=None,
                   help="next-token side: CTRL repetition penalty (e.g. 1.2) over prompt and output tokens")
    p.add_argument("--real", action="store_true",
                   help="also score each prompt's true continuation (rows are read at prefix + gen tokens): the "
                        "anchor that separates a parallel sampler's incoherence from next-token repetition")
    args = p.parse_args()
    device = torch.device(args.device)

    if args.smoke:
        from scripts.sap_decode_bench import smoke_model
        model = ref = smoke_model(device, mode=args.override_mode or "p1_discrete").eval()
        tok = None
        prefixes = torch.randint(0, model.config.vocab_size, (8, args.prefix_len))
        args.gen_tokens, args.batch = 16, 4
    else:
        model, tok = load(args.checkpoint_dir, args.step, device, args.tokenizer_dir)
        ref, _ = (load(args.reference_dir, None, device, args.tokenizer_dir)
                  if args.reference_dir else (model, None))
        ar_model = load(args.ar_dir, None, device, args.tokenizer_dir)[0] if args.ar_dir else model
        if args.override_mode and model.sap_head is not None:
            model.sap_head.mode = args.override_mode
        if args.jacobi_sweeps is not None and model.sap_head is not None:
            model.sap_head.jacobi_sweeps = args.jacobi_sweeps
        from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
        row_len = args.prefix_len + (args.gen_tokens if args.real else 0)
        loader = tokenizing_distributed_data_loader_bos_bestfit(
            tok, args.batch, row_len, split="val", device="cpu", data_dir=args.data_dir)
        rows = []
        while sum(r.size(0) for r in rows) < args.n_prefixes:
            x, y = next(loader)
            rows.append(torch.cat([x, y[:, -1:]], 1) if args.real else x)
        full_rows = torch.cat(rows)[:args.n_prefixes]
        prefixes = full_rows[:, :args.prefix_len]
        real = full_rows[:, args.prefix_len:args.prefix_len + args.gen_tokens] if args.real else None
    if args.smoke:
        ar_model = model
        real = torch.randint(0, model.config.vocab_size, (prefixes.size(0), args.gen_tokens)) if args.real else None
    if args.lanes > 0:
        from nanochat.lanes import LANE_TOKEN, generate_lanes
        L, S = args.lanes, max(1, (args.gen_tokens - 1) // args.lanes)
        args.gen_tokens = 1 + L * S
        lane_token = 15 if tok is None else tok.encode_special(LANE_TOKEN)

        def gen_block(pre, i):
            g = torch.Generator(device=device)
            g.manual_seed(i)
            return generate_lanes(model, pre, L, S, lane_token, temperature=args.temperature, generator=g)
    elif args.wb_window > 0:
        from nanochat.lanes import LANE_TOKEN
        from nanochat.wbisect import bridged_lanes_steps, wb_generate, wb_steps
        P, N = prefixes.size(1), prefixes.size(1) + args.gen_tokens
        steps = (bridged_lanes_steps(N, P, args.wb_lanes, args.wb_window) if args.wb_lanes
                 else wb_steps(N, P, args.wb_window))
        mask_token = 15 if tok is None else tok.encode_special(LANE_TOKEN)

        def gen_block(pre, i):
            g = torch.Generator(device=device)
            g.manual_seed(i)
            return wb_generate(model, pre, steps, mask_token, temperature=args.temperature, generator=g)[:, P:]
    else:
        assert model.sap_head is not None, "the checkpoint has no block head (sap_block_T=0)"

        def gen_block(pre, i):
            return generate_block_kv(model, pre, args.gen_tokens, temperature=args.temperature, seed=i)

    out = open(args.out, "w")
    agg = {"ar_nll": 0.0, "block_nll": 0.0, "ar_distinct3": 0.0, "block_distinct3": 0.0, "n": 0,
           "real_nll": 0.0, "real_distinct3": 0.0, "ar_ent": 0.0, "block_ent": 0.0, "real_ent": 0.0}
    for i in range(0, prefixes.size(0), args.batch):
        pre = prefixes[i:i + args.batch].to(device)
        ar_t = args.temperature if args.ar_temperature is None else args.ar_temperature
        ar = generate_ar_kv(ar_model, pre, args.gen_tokens, temperature=ar_t, seed=i, top_p=args.ar_top_p,
                            rep_penalty=args.ar_rep_penalty)
        blk = gen_block(pre, i)
        ar_nll = reference_nll(ref, pre, ar)
        blk_nll = reference_nll(ref, pre, blk)
        if real is not None:
            rl = real[i:i + args.batch].to(device)[:, :blk.size(1)]
            rl_nll = reference_nll(ref, pre, rl)
            for j in range(pre.size(0)):
                agg["real_nll"] += rl_nll[j].item()
                agg["real_distinct3"] += distinct3(rl[j].tolist())
                agg["real_ent"] += unigram_entropy(rl[j].tolist())
        for j in range(pre.size(0)):
            rec = {"prefix_ids": pre[j].tolist(), "ar_ids": ar[j].tolist(), "block_ids": blk[j].tolist(),
                   "ar_ref_nll": ar_nll[j].item(), "block_ref_nll": blk_nll[j].item(),
                   "ar_distinct3": distinct3(ar[j].tolist()), "block_distinct3": distinct3(blk[j].tolist())}
            if tok is not None:
                rec.update(prefix=tok.decode(rec["prefix_ids"]), ar=tok.decode(rec["ar_ids"]),
                           block=tok.decode(rec["block_ids"]))
            out.write(json.dumps(rec) + "\n")
            agg["ar_nll"] += rec["ar_ref_nll"]
            agg["block_nll"] += rec["block_ref_nll"]
            agg["ar_distinct3"] += rec["ar_distinct3"]
            agg["block_distinct3"] += rec["block_distinct3"]
            agg["ar_ent"] += unigram_entropy(rec["ar_ids"])
            agg["block_ent"] += unigram_entropy(rec["block_ids"])
            agg["n"] += 1
    out.close()
    n = max(agg["n"], 1)
    what = (f"{args.lanes} lanes" if args.lanes > 0 else
            f"two-stream {'bridged lanes L=' + str(args.wb_lanes) if args.wb_lanes else 'window bisection'} "
            f"n={args.wb_window}, {int(steps.max()) + 1} steps" if args.wb_window > 0 else
            f"block T={model.sap_head.T} ({model.sap_head.mode})")
    print(f"{agg['n']} prefixes, {args.gen_tokens} tokens each, {what}")
    print(f"  reference ppl  next-token {math.exp(agg['ar_nll'] / n):8.2f} | block {math.exp(agg['block_nll'] / n):8.2f}")
    print(f"  distinct 3-grams  next-token {agg['ar_distinct3'] / n:.3f} | block {agg['block_distinct3'] / n:.3f}")
    print(f"  unigram entropy  next-token {agg['ar_ent'] / n:.3f} | block {agg['block_ent'] / n:.3f}")
    if real is not None:
        print(f"  real continuation  reference ppl {math.exp(agg['real_nll'] / n):8.2f} | distinct 3-grams "
              f"{agg['real_distinct3'] / n:.3f} | unigram entropy {agg['real_ent'] / n:.3f}")
    print(f"  pairs for the LLM judge: {args.out}")


if __name__ == "__main__":
    main()
