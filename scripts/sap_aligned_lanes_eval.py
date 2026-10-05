"""
S12 S-1 gate: sentence-aligned lanes against plain lanes and dense, on the same validation rows.

Block bpb for each model on its own layout, after a fixed prefix of P tokens:
  dense    causal, targets r[P..N]
  ln:L     plain lanes (S08), the same targets
  la:L:W   sentence-aligned lanes (nanochat/lanes.py::aligned_lanes_rows): every target after
           the prefix, the pad (lane-end) decisions included in the nats, over the text bytes
An aligned row holds less text than a plain one (the padding), so it is reported with its text
tokens per decode step (placed / S) and the share of lanes cut at a sentence end or document start.
The pre-registered comparison (s12_sap_brainstorm.md): aligned lanes against plain lanes at the
same text tokens per step, by interpolating the plain-lane tax in log2(L).

    python -m scripts.sap_aligned_lanes_eval --tokenizer-dir tokenizer_sap --data-dir data \\
        --model dense=out/.../S11dense_x1_s1 --model ln:64=out/.../S11ln64x1_s1 \\
        --model la:64:15=out/.../S11la64w15x1_s1
"""
from __future__ import annotations

import argparse
import json
import math

import torch

from nanochat.checkpoint_manager import build_model, find_last_step
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from nanochat.lanes import LANE_TOKEN, PAD_TOKEN, aligned_lanes_rows, lane_inputs, lane_mask, sentence_end_table
from nanochat.tokenizer import get_token_bytes


@torch.no_grad()
def block_score(model, xs, ys, P, token_bytes, batch, mask=None, count_zero_byte=False):
    """Nats over every scored target at positions >= P - 1 and the bytes of those targets. Zero-byte
    targets (pads) add nats only when count_zero_byte."""
    nats, nbytes = 0.0, 0
    kw = {} if mask is None else {"lane_mask": mask}
    for i in range(0, xs.size(0), batch):
        x, y = xs[i:i + batch], ys[i:i + batch]
        loss = model(x, y, loss_reduction="none", **kw).view(y.shape).float()[:, P - 1:]
        yt = y[:, P - 1:]
        valid = yt >= 0
        b = torch.where(valid, token_bytes[yt.clamp_min(0)], torch.zeros_like(yt))
        keep = valid if count_zero_byte else (b > 0)
        nats += float((loss * keep).sum())
        nbytes += int(b.sum())
    return nats, nbytes


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", action="append", required=True, help="dense=DIR, ln:L=DIR or la:L:W=DIR")
    p.add_argument("--tokenizer-dir", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--prefix", type=int, default=128)
    p.add_argument("--rows", type=int, default=256)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    specs = [m.split("=", 1) for m in args.model]
    first, tok, _ = build_model(specs[0][1], find_last_step(specs[0][1]), dev, phase="eval",
                                tokenizer_dir=args.tokenizer_dir)
    N = first.config.sequence_len
    loader = tokenizing_distributed_data_loader_bos_bestfit(tok, args.batch, N, split="val", data_dir=args.data_dir)
    xs, ys = [], []
    while sum(x.size(0) for x in xs) < args.rows:
        x, y = next(loader)
        xs.append(x.to(dev))
        ys.append(y.to(dev))
    X, Y = torch.cat(xs)[:args.rows], torch.cat(ys)[:args.rows]
    token_bytes = get_token_bytes(device=dev, tokenizer_dir=args.tokenizer_dir)
    lane_tok, pad_tok = tok.encode_special(LANE_TOKEN), tok.encode_special(PAD_TOKEN)
    ends = sentence_end_table(tok, first.config.vocab_size)
    P = args.prefix
    out = {}
    for k, (kind, ckdir) in enumerate(specs):
        model = first if k == 0 else build_model(ckdir, find_last_step(ckdir), dev, phase="eval",
                                                 tokenizer_dir=args.tokenizer_dir)[0]
        model.eval()
        parts = kind.split(":")
        if parts[0] == "dense":
            nats, nb = block_score(model, X, Y, P, token_bytes, args.batch)
            res = {"bpb": nats / (nb * math.log(2)), "text_tokens_per_step": 1.0}
        elif parts[0] == "ln":                                  # ln:L or ln:L:LAG (S-3 lag oracle)
            L = int(parts[1])
            lag = int(parts[2]) if len(parts) > 2 else 0
            from nanochat.lanes import lagged_lane_mask
            mask = lagged_lane_mask(N, P, L, lag, dev) if lag else lane_mask(N, P, L, dev)
            nats, nb = block_score(model, lane_inputs(X, P, L, lane_tok), Y, P, token_bytes, args.batch, mask=mask)
            res = {"bpb": nats / (nb * math.log(2)), "lanes": L, "lag": lag, "text_tokens_per_step": float(L)}
        elif parts[0] == "la":
            L, W = int(parts[1]), int(parts[2])
            xa, ya, placed, aligned = aligned_lanes_rows(X, Y, P, L, lane_tok, pad_tok, ends, tok.get_bos_token_id(), W)
            nats, nb = block_score(model, xa, ya, P, token_bytes, args.batch, mask=lane_mask(N, P, L, dev),
                                   count_zero_byte=True)
            nats_text, _ = block_score(model, xa, ya, P, token_bytes, args.batch, mask=lane_mask(N, P, L, dev))
            S = (N - P) // L
            res = {"bpb": nats / (nb * math.log(2)), "bpb_without_pad_decisions": nats_text / (nb * math.log(2)),
                   "lanes": L, "window": W, "text_tokens_per_step": float(placed.float().mean()) / S,
                   "text_share": float(placed.float().mean()) / (N - P),
                   "lanes_cut_at_boundary": float(aligned.float().mean()) / L}
        else:
            raise ValueError(kind)
        out[kind] = res
        if k > 0:
            del model
    ref = out[specs[0][0]]["bpb"]
    print(f"block bpb after a {P}-token prefix over {len(X)} rows (ratio to {specs[0][0]}):")
    for kind, r in out.items():
        extra = ""
        if kind.startswith("la"):
            extra = (f" | text tokens per step {r['text_tokens_per_step']:.1f} ({r['text_share']:.0%} of slots), "
                     f"{r['lanes_cut_at_boundary']:.0%} of lanes cut at a boundary, without pad decisions "
                     f"{r['bpb_without_pad_decisions']:.4f}")
        elif kind.startswith("ln"):
            extra = f" | text tokens per step {r['text_tokens_per_step']:.0f}"
        print(f"  {kind:12s} {r['bpb']:.4f} ({r['bpb'] / ref:.4f}){extra}", flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
