"""
Re-score saved generation samples under another reference model, without regenerating them.

`scripts/sap_eval_generation.py` writes one JSON record per prompt (prefix_ids, ar_ids,
block_ids). A small reference model judges local fluency but is weak on global coherence, which
is where a parallel order would fail, so this script scores the same samples under a stronger
model. It also re-derives each prompt's true continuation with the eval's own deterministic
loader (same batch, same row length) and checks that the prompts match, so the real-text anchor
is scored by the same model.

    python -m scripts.sap_rescore_samples --reference-dir out/.../base --tokenizer-dir tokenizer_sap \\
        --data-dir data --jsonl gen_a.jsonl gen_b.jsonl --out rescore.json
"""
from __future__ import annotations

import argparse
import json
import math

import torch

from nanochat.checkpoint_manager import build_model, find_last_step
from scripts.sap_eval_generation import reference_nll, unigram_entropy


def real_rows(tok, data_dir, n, batch, prefix_len, gen_tokens):
    """The eval's --real rows: prompt and true continuation, read exactly as sap_eval_generation does."""
    from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
    loader = tokenizing_distributed_data_loader_bos_bestfit(tok, batch, prefix_len + gen_tokens, split="val",
                                                            device="cpu", data_dir=data_dir)
    rows = []
    while sum(r.size(0) for r in rows) < n:
        x, y = next(loader)
        rows.append(torch.cat([x, y[:, -1:]], 1))
    full = torch.cat(rows)[:n]
    return full[:, :prefix_len], full[:, prefix_len:prefix_len + gen_tokens]


@torch.no_grad()
def mean_nll(ref, prefixes, conts, batch, device):
    tot, n = 0.0, 0
    for i in range(0, prefixes.size(0), batch):
        nll = reference_nll(ref, prefixes[i:i + batch].to(device), conts[i:i + batch].to(device))
        tot += nll.sum().item()
        n += nll.numel()
    return tot / n


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--reference-dir", required=True)
    p.add_argument("--step", type=int, default=None)
    p.add_argument("--tokenizer-dir", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--jsonl", nargs="+", required=True)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--loader-batch", type=int, default=16, help="the eval's --batch (fixes which rows were read)")
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    step = args.step if args.step is not None else find_last_step(args.reference_dir)
    ref, tok, _ = build_model(args.reference_dir, step, device, phase="eval", tokenizer_dir=args.tokenizer_dir)
    ref.eval()
    out, real_cache = {}, {}
    for path in args.jsonl:
        recs = [json.loads(line) for line in open(path)]
        pre = torch.tensor([r.get("prefix_ids", r.get("prompt_tokens")) for r in recs])
        ar = torch.tensor([r.get("ar_ids", r.get("target_tokens", r.get("generated_tokens"))) for r in recs])
        blk = torch.tensor([r.get("block_ids", r.get("generated_tokens")) for r in recs])
        key = (pre.size(1), blk.size(1), len(recs))
        if key not in real_cache:
            rp, rc = real_rows(tok, args.data_dir, len(recs), args.loader_batch, pre.size(1), blk.size(1))
            real_cache[key] = (rp, rc, mean_nll(ref, rp, rc, args.batch, device),
                               sum(unigram_entropy(r.tolist()) for r in rc) / len(recs))
        rp, rc, real_nll, real_ent = real_cache[key]
        same = bool(torch.equal(rp, pre))
        res = {"rows": len(recs), "prompts_match_real_rows": same,
               "ref_ppl_ar": math.exp(mean_nll(ref, pre, ar, args.batch, device)),
               "ref_ppl_block": math.exp(mean_nll(ref, pre, blk, args.batch, device)),
               "entropy_ar": sum(unigram_entropy(r.tolist()) for r in ar) / len(recs),
               "entropy_block": sum(unigram_entropy(r.tolist()) for r in blk) / len(recs),
               "ref_ppl_real": math.exp(real_nll) if same else float("nan"), "entropy_real": real_ent}
        out[path] = res
        print(f"{path.split('/')[-1]}: ref ppl next-token {res['ref_ppl_ar']:.2f} (entropy {res['entropy_ar']:.3f}) | "
              f"parallel {res['ref_ppl_block']:.2f} ({res['entropy_block']:.3f}) | real {res['ref_ppl_real']:.2f} "
              f"({real_ent:.3f}){'' if same else ' [prompts differ from the loader rows: real not comparable]'}",
              flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
