"""
Offline oracle for boundary-aligned lanes: is a lane start cheap when it falls right after a
sentence, paragraph or document boundary?

A plain-lanes model (S08) predicts each later lane's first tokens with no local left context; at
d4 the first one costs about 2x a normal token, at d8 2.35x. If that cost comes from cutting
sentences, lanes whose starts sit at natural boundaries would remove most of the tax. This script
measures it on existing checkpoints, with no training: it scores the same validation rows with a
plain-lanes model and a dense reference, and splits every lane start (lanes 1..L-1) by the true
text just before it (the junction token, which the lanes model writes last):
  doc        the junction token is the document start <|bos|>
  paragraph  its text contains a newline
  sentence   it ends a sentence (. ! ? possibly followed by quotes or brackets)
  mid        anything else
For each class it reports the lanes/dense nats ratio at lane offsets 0, 1, 2-3, 4-7 and 0-7, the
excess over dense in nats per lane, and how often the class occurs. The reported excess of the
boundary classes is what aligned lanes could pay per start, the mid class what plain lanes pay.

    python -m scripts.sap_lane_start_oracle --dense-dir out/.../S11dense_x1_s1 \\
        --lanes ln64=out/.../S11ln64x4_s1:64 --tokenizer-dir tokenizer_sap --data-dir data
"""
from __future__ import annotations

import argparse
import json

import torch

from nanochat.checkpoint_manager import build_model, find_last_step
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from nanochat.lanes import LANE_TOKEN, lane_inputs, lane_mask

GROUPS = [("0", [0]), ("1", [1]), ("2-3", [2, 3]), ("4-7", [4, 5, 6, 7]), ("0-7", list(range(8)))]


def junction_classes(tok, V):
    """Class of every token id as a junction (the text just before a lane start)."""
    cls = torch.full((V,), 3, dtype=torch.long)                      # 3 = mid
    for i in range(V):
        try:
            s = tok.decode([i])
        except Exception:
            continue
        t = s.rstrip(" \"')]}”’")
        if "\n" in s:
            cls[i] = 1
        elif t.endswith((".", "!", "?")):
            cls[i] = 2
    cls[tok.get_bos_token_id()] = 0
    return cls


@torch.no_grad()
def per_token_nll(model, x, y, mask=None):
    kw = {} if mask is None else {"lane_mask": mask}
    return model(x, y, loss_reduction="none", **kw).view(y.shape).float()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dense-dir", required=True)
    p.add_argument("--lanes", action="append", required=True, help="name=checkpoint_dir:L")
    p.add_argument("--tokenizer-dir", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--prefix", type=int, default=128)
    p.add_argument("--rows", type=int, default=256)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dense, tok, _ = build_model(args.dense_dir, find_last_step(args.dense_dir), dev, phase="eval",
                                tokenizer_dir=args.tokenizer_dir)
    dense.eval()
    N = dense.config.sequence_len
    loader = tokenizing_distributed_data_loader_bos_bestfit(tok, args.batch, N, split="val", data_dir=args.data_dir)
    xs, ys = [], []
    while sum(x.size(0) for x in xs) < args.rows:
        x, y = next(loader)
        xs.append(x.to(dev))
        ys.append(y.to(dev))
    X, Y = torch.cat(xs)[:args.rows], torch.cat(ys)[:args.rows]
    cls = junction_classes(tok, dense.config.vocab_size).to(dev)
    lane_tok = tok.encode_special(LANE_TOKEN)
    ref = torch.cat([per_token_nll(dense, X[i:i + args.batch], Y[i:i + args.batch]) for i in range(0, len(X), args.batch)])
    names = ["doc", "paragraph", "sentence", "mid"]
    result = {}
    for spec in args.lanes:
        name, rest = spec.split("=", 1)
        ckdir, L = rest.rsplit(":", 1)
        L = int(L)
        model, _, _ = build_model(ckdir, find_last_step(ckdir), dev, phase="eval", tokenizer_dir=args.tokenizer_dir)
        model.eval()
        mask = lane_mask(N, args.prefix, L, dev)
        own = torch.cat([per_token_nll(model, lane_inputs(X[i:i + args.batch], args.prefix, L, lane_tok),
                                       Y[i:i + args.batch], mask) for i in range(0, len(X), args.batch)])
        S = (N - args.prefix) // L
        starts = args.prefix + S * torch.arange(1, L, device=dev)            # lane j's first input position
        jcls = cls[X[:, starts]]                                             # (rows, L-1): class of the junction token
        rep = {}
        for c, cname in enumerate(names):
            sel = jcls == c
            n = int(sel.sum())
            if n == 0:
                rep[cname] = {"lanes": 0}
                continue
            row = {"lanes": n, "share": n / sel.numel()}
            for gname, offs in GROUPS:
                pos = (starts[None, :, None] + torch.tensor(offs, device=dev)[None, None, :]).expand(len(X), -1, -1)
                o = own.gather(1, pos.reshape(len(X), -1)).view_as(pos)[sel]
                r = ref.gather(1, pos.reshape(len(X), -1)).view_as(pos)[sel]
                row[f"ratio_{gname}"] = float(o.sum() / r.sum())
                row[f"excess_nats_per_lane_{gname}"] = float((o - r).sum() / n)
            rep[cname] = row
        result[name] = rep
        print(f"{name} (L={L}, lane length {S}): lanes/dense nats ratio by lane offset, per junction class")
        for cname in names:
            r = rep[cname]
            if not r.get("lanes"):
                continue
            print(f"  {cname:9s} {r['share']:6.1%} of starts | " +
                  " ".join(f"off {g} {r['ratio_' + g]:.2f}" for g, _ in GROUPS) +
                  f" | excess {r['excess_nats_per_lane_0-7']:.2f} nats per lane (offsets 0-7)", flush=True)
        del model
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
