"""
Where lane samples lose reference likelihood: a per-position profile of sap_eval_generation's
lane output (prefix, next-token continuation, lane continuation per row), scored again by the
reference model with the per-token NLL kept.

For L lanes of S tokens (the continuation is x_P then lane 0's S tokens, lane 1's, ...) it
reports, for the lane samples and for the next-token samples over the same position ranges:
  - the mean reference NLL per lane range (lane 0 has the prompt as its immediate context,
    lane j > 0 starts j*S tokens past it);
  - the mean over the first `--head` tokens of each lane j > 0 (the junction) against the rest;
  - distinct 3-grams per lane range, since repetitive text is cheap for a reference model and
    long next-token samples from small models repeat.

    python -m scripts.sap_lane_gen_profile --gen-jsonl gen_S07_lanes4_s1.jsonl --lanes 4 \\
        --reference-dir out/.../S07_dense_L_s2 --tokenizer-dir tokenizer
"""
from __future__ import annotations

import argparse
import json
import math

import torch

from nanochat.checkpoint_manager import build_model, find_last_step


@torch.no_grad()
def token_nll(ref, prefix, cont):
    """Per-token reference NLL of cont given prefix (B, n)."""
    seq = torch.cat([prefix, cont], dim=1)
    logits = ref(seq[:, :-1]).float()
    lp = torch.log_softmax(logits, -1).gather(-1, seq[:, 1:, None]).squeeze(-1)
    return -lp[:, prefix.size(1) - 1:]


def distinct3(ids):
    tri = list(zip(ids, ids[1:], ids[2:]))
    return len(set(tri)) / max(1, len(tri))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gen-jsonl", required=True)
    p.add_argument("--reference-dir", required=True)
    p.add_argument("--tokenizer-dir", required=True)
    p.add_argument("--lanes", type=int, required=True)
    p.add_argument("--head", type=int, default=16, help="tokens counted as a lane's junction")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ref, _, _ = build_model(args.reference_dir, find_last_step(args.reference_dir), device, phase="eval",
                            tokenizer_dir=args.tokenizer_dir)
    ref.eval()
    rows = [json.loads(line) for line in open(args.gen_jsonl)]
    n = len(rows[0]["block_ids"])
    L = args.lanes
    S = (n - 1) // L
    ranges = {f"lane{j}": (1 + j * S, 1 + (j + 1) * S) for j in range(L)}
    sums = {k: {"ar": 0.0, "lanes": 0.0, "n": 0} for k in list(ranges) + ["junction", "interior", "all"]}
    d3 = {k: {"ar": 0.0, "lanes": 0.0} for k in ranges}
    for i in range(0, len(rows), args.batch):
        chunk = rows[i:i + args.batch]
        pre = torch.tensor([r["prefix_ids"] for r in chunk], device=device)
        nll = {k: token_nll(ref, pre, torch.tensor([r[f"{k2}_ids"] for r in chunk], device=device)).cpu()
               for k, k2 in (("ar", "ar"), ("lanes", "block"))}
        for name, (a, b) in ranges.items():
            for k in ("ar", "lanes"):
                sums[name][k] += nll[k][:, a:b].sum().item()
            sums[name]["n"] += nll["ar"][:, a:b].numel()
            for r in chunk:
                d3[name]["ar"] += distinct3(r["ar_ids"][a:b])
                d3[name]["lanes"] += distinct3(r["block_ids"][a:b])
        head = torch.zeros(n, dtype=torch.bool)
        for j in range(1, L):
            head[1 + j * S: 1 + j * S + args.head] = True
        for name, m in (("junction", head), ("interior", ~head), ("all", torch.ones(n, dtype=torch.bool))):
            for k in ("ar", "lanes"):
                sums[name][k] += nll[k][:, m].sum().item()
            sums[name]["n"] += nll["ar"][:, m].numel()
    res = {"rows": len(rows), "lanes": L, "lane_len": S, "head": args.head, "ranges": {}}
    print(f"{len(rows)} rows, {n} tokens each, {L} lanes of {S}; reference PPL (next-token | lanes) and distinct-3")
    for name in list(ranges) + ["junction", "interior", "all"]:
        s = sums[name]
        if s["n"] == 0:
            continue
        ar, ln = math.exp(s["ar"] / s["n"]), math.exp(s["lanes"] / s["n"])
        line = {"ppl_ar": ar, "ppl_lanes": ln, "tokens": s["n"]}
        extra = ""
        if name in d3:
            line.update(d3_ar=d3[name]["ar"] / len(rows), d3_lanes=d3[name]["lanes"] / len(rows))
            extra = f" | distinct-3 {line['d3_ar']:.3f} | {line['d3_lanes']:.3f}"
        res["ranges"][name] = line
        print(f"  {name:9s} ppl {ar:8.2f} | {ln:8.2f} ({ln / ar - 1:+.1%}){extra}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
