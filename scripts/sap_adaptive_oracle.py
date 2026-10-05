"""
Adaptive block length for trunk-depth slots, measured offline and exactly.

A trained depth model (depth_local / depth_roll, any m, shared or copies) decodes a block as one
trunk pass and then slot passes. Here the block length is not fixed: before slot k emits its
token, a causal router looks at slot k's own distribution (its max probability or entropy) and
either lets it emit, or flushes, so the trunk runs on the block so far and emits the next token
at full depth. Every token's distribution is then either the slot's or the trunk's, chosen from
the past alone, so the policy is an autoregressive model with an exact teacher-forced likelihood.

For each row this script tabulates, for every block start t and offset k <= T-1, slot k's
log-probability of the data token and its router signals (one batched pass per chunk of starts),
plus the trunk's next-token log-probability at every position. It then replays the policy along
each row for a sweep of thresholds and reports, per router:
  ratio  = the policy's NLL / the trunk's NLL on the same positions (with copies on a frozen
           dense trunk this is the block-vs-dense number C);
  speed  = AR cost / policy cost, in sequential layer passes: a trunk pass costs L + r, a slot
           pass m + r (r = readout and sampling, in layer units), and a flush wastes the slot pass
           that decided it;
  mean block length, and the share of tokens the slots emitted.
The "oracle" router lets a slot emit only if its log-prob is within a margin of the trunk's: it
reads the data token, so it is an upper bound on any causal router of this kind, not a policy.

    python -m scripts.sap_adaptive_oracle --checkpoint-dir out/.../S03h_depth_local_D1S0_T4 \\
        --tokenizer-dir tokenizer --data-dir data --rows 32 --out adaptive.json
"""
from __future__ import annotations

import argparse
import json
import math

import torch

from nanochat.checkpoint_manager import build_model, find_last_step
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit


@torch.no_grad()
def slot_tables(model, x, chunk=64):
    """Row x (1, Tq). Returns (trunk_lp (Tq-1,), slot_lp, maxp, negent) with the last three (n, K):
    entry [t, k-1] is slot k of the block started at t, predicting x[t+k+1] from position t+k."""
    head = model.sap_head
    T = head.T
    K = T - 1
    Tq = x.size(1)
    hid, states = model(x, skip_logits=True, sap_capture=True)
    logits = model._sap_readout(hid)                                       # (1, Tq, V)
    trunk_lp = torch.log_softmax(logits[0, :-1].float(), -1).gather(-1, x[0, 1:, None]).squeeze(-1)
    n = Tq - T                                    # starts whose whole block x[t+1..t+T] exists
    tgt = x[:, 1:]
    out = [torch.empty(n, K, device=x.device) for _ in range(3)]
    for c0 in range(0, n, chunk):
        st = torch.arange(c0, min(n, c0 + chunk), device=x.device)[None]  # (1, nc)
        y = torch.stack([tgt.gather(1, st + k) for k in range(T)], dim=-1)  # y[..., k] = x[t+1+k]
        slot0 = logits.gather(1, st[..., None].expand(-1, -1, logits.size(-1)))
        lg = model._sap_depth_logprob(x, states, slot0, y, st, torch.ones_like(y, dtype=torch.bool),
                                      return_slot_logits=True)[0].float()  # (nc, K, V)
        lsm = torch.log_softmax(lg, -1)
        nc = st.size(1)
        out[0][c0:c0 + nc] = lsm.gather(-1, y[0, :, 1:, None]).squeeze(-1)
        p = lsm.exp()
        out[1][c0:c0 + nc] = p.max(-1).values
        out[2][c0:c0 + nc] = (p * lsm).sum(-1)                             # negative entropy
    return trunk_lp, out[0], out[1], out[2]


def replay(trunk_lp, slot_lp, sig, tau, kmax):
    """One row of the policy: a slot emits iff sig >= tau (and its offset <= kmax). Returns
    (policy nll, trunk nll, tokens, blocks, slot tokens, wasted slot passes)."""
    n = len(slot_lp)
    nll = ref = 0.0
    blocks = slot_tok = wasted = tok = 0
    t, k = 0, 0                     # current block start; offset of the position being predicted
    for p in range(n):              # position p predicts x[p+1]
        ref += -trunk_lp[p]
        tok += 1
        if k >= 1 and k <= kmax:
            if sig[t][k - 1] >= tau:
                nll += -slot_lp[t][k - 1]
                slot_tok += 1
                k += 1
                continue
            wasted += 1
        t, k = p, 1                 # the trunk runs on the block so far and emits x[p+1]
        blocks += 1
        nll += -trunk_lp[p]
    return nll, ref, tok, blocks, slot_tok, wasted


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint-dir", type=str, required=True)
    p.add_argument("--step", type=int, default=None)
    p.add_argument("--tokenizer-dir", type=str, required=True)
    p.add_argument("--data-dir", type=str, required=True)
    p.add_argument("--rows", type=int, default=32, help="validation rows of sequence_len tokens")
    p.add_argument("--readout-cost", type=float, default=1.0, help="readout + sampling, in layer passes")
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    step = args.step if args.step is not None else find_last_step(args.checkpoint_dir)
    model, tokenizer, meta = build_model(args.checkpoint_dir, step, device, phase="eval",
                                         tokenizer_dir=args.tokenizer_dir)
    model.eval()
    head = model.sap_head
    L, m, K = model.config.n_layer, head.depth_m, head.T - 1
    loader = tokenizing_distributed_data_loader_bos_bestfit(tokenizer, 1, model.config.sequence_len,
                                                             split="val", data_dir=args.data_dir)
    rows = []
    for _ in range(args.rows):
        x, _ = next(loader)
        tables = slot_tables(model, x.to(device))
        rows.append([v.cpu().tolist() for v in tables])
    r = args.readout_cost
    taus = {"maxp": [0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 1.01],
            "negent": [-99.0, -3.0, -2.0, -1.5, -1.0, -0.7, -0.5, -0.3, -0.1, 1.0],
            "oracle": [-99.0, -1.0, -0.5, -0.25, -0.1, -0.05, 0.0, 0.05, 0.1]}
    result = {"checkpoint": args.checkpoint_dir, "step": step, "L": L, "m": m, "T": head.T,
              "mode": head.mode, "rows": args.rows, "readout_cost": r, "curves": {}}
    print(f"{head.mode} m={m} T={head.T} L={L}: {args.rows} rows; ratio = policy NLL / trunk NLL")
    for name, grid in taus.items():
        for kmax in sorted({1, K}) if name != "oracle" else [K]:
            pts = []
            for tau in grid:
                agg = [0.0, 0.0, 0, 0, 0, 0]
                for trunk_lp, slot_lp, maxp, negent in rows:
                    if name == "maxp":
                        sig = maxp
                    elif name == "negent":
                        sig = negent
                    else:   # slot log-prob minus the trunk's at the same position: reads the token
                        sig = [[slot_lp[t][k] - trunk_lp[t + k + 1] for k in range(len(slot_lp[t]))]
                               for t in range(len(slot_lp))]
                    for i, v in enumerate(replay(trunk_lp, slot_lp, sig, tau, kmax)):
                        agg[i] += v
                nll, ref, tok, blocks, slot_tok, wasted = agg
                cost = blocks * (L + r) + (slot_tok + wasted) * (m + r)
                pt = {"tau": tau, "kmax": kmax, "ratio": nll / ref, "speed": tok * (L + r) / cost,
                      "block_len": tok / max(1, blocks), "slot_share": slot_tok / tok}
                pts.append(pt)
                print(f"  {name:6s} kmax={kmax} tau={tau:6.2f}: ratio {pt['ratio']:.4f} speed {pt['speed']:.2f}x "
                      f"block {pt['block_len']:.2f} slot share {pt['slot_share']:.3f}")
            result["curves"][f"{name}_k{kmax}"] = pts
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
