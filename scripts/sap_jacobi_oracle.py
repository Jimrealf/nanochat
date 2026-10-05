"""
S10 oracle: how many coupled-noise refinement stages does one-pass emulation of AR sampling need?

A one-pass generator that conditions later positions on drafted tokens is, at best, an unrolled
Jacobi iteration on the AR fixed point t_k = Pick(noise_k, P(. | context, t_<k)) with the noise held
fixed. With the exact conditionals each stage fixes at least the first wrong position, so T stages
always suffice; how many fewer are needed depends on how often a pick survives a change of history,
which is the coupling's robustness. This script measures it with exact conditionals (no learning):

  stage 0: t^0_k = Pick(noise_k, P(. | context))   position-wise marginals, all k at once
  stage s: t^s_k = Pick(noise_k, P(. | context, t^{s-1}_<k))   for all k at once

for four couplings: inverse CDF over a scalar u in token-id order, inverse CDF in the semantic
order, the tree pick (one uniform per level of a binary tree over the semantic order), and
Gumbel-max over a V-vector of Gumbels (near-maximal: picks under p and q agree with probability
close to 1 - TV(p, q)). Reported per stage: agreement with the fixed point (the exact
sequential sample under the same noise), fraction of blocks fully converged, and invalid rate.

    python -m scripts.sap_jacobi_oracle --T 4,16,64 --stages 0,1,2,4,8,16,32,64
"""
from __future__ import annotations

import argparse
import json

import torch

from nanochat.ptp import ordered_pick, semantic_rank, tree_levels, tree_pick
from scripts.sap_synthetic import build_phrase_hmm, filter_states, sample_sequences, true_block_logprob


def hmm_conditionals(hmm, alpha, t):
    """P(t_k | context, t_<k) for every k at once along the given history t (B, T): (B, T, V)."""
    a, out = alpha, []
    for k in range(t.size(1)):
        pred = a @ hmm.A                                          # state distribution at position k
        out.append(pred @ hmm.E)
        w = pred * hmm.E[:, t[:, k]].t()
        a = w / w.sum(1, keepdim=True).clamp_min(1e-30)
    return torch.stack(out, 1).double()


def hmm_marginals(hmm, alpha, T):
    m, out = alpha, []
    for _ in range(T):
        m = m @ hmm.A
        out.append(m @ hmm.E)
    return torch.stack(out, 1).double()


class Coupling:
    def __init__(self, kind, B, T, V, rank=None, device="cpu"):
        self.kind, self.rank = kind, rank
        if kind == "gumbel":
            self.noise = -torch.log(-torch.log(torch.rand(B, T, V, dtype=torch.float64, device=device).clamp_min(1e-300)))
        elif kind == "tree":
            self.noise = torch.rand(B, T, tree_levels(V), dtype=torch.float64, device=device)
        else:
            self.noise = torch.rand(B, T, dtype=torch.float64, device=device)

    def pick(self, probs, k=None):
        n = self.noise if k is None else self.noise[:, k]
        if self.kind == "gumbel":
            return (probs.clamp_min(1e-300).log() + n).argmax(-1)
        if self.kind == "tree":
            return tree_pick(probs, n, self.rank)
        return ordered_pick(probs, n, self.rank)


@torch.no_grad()
def run(T, stages, B, hmm, alpha, kinds):
    V = hmm.E.size(1)
    ranks = {"cdf_id": torch.arange(V, device=alpha.device), "cdf_sem": semantic_rank(hmm.E),
             "tree_sem": semantic_rank(hmm.E)}
    rows = {}
    for kind in kinds:
        c = Coupling(kind.split("_")[0], B, T, V, ranks.get(kind), alpha.device)
        # Fixed point: exact sequential sampling under the same noise.
        ref = torch.zeros(B, T, dtype=torch.long, device=alpha.device)
        a = alpha
        for k in range(T):
            pred = a @ hmm.A
            ref[:, k] = c.pick((pred @ hmm.E).double(), k)
            w = pred * hmm.E[:, ref[:, k]].t()
            a = w / w.sum(1, keepdim=True).clamp_min(1e-30)
        t = c.pick(hmm_marginals(hmm, alpha, T))
        out, s = [], 0
        for target in sorted(stages):
            while s < target:
                t = c.pick(hmm_conditionals(hmm, alpha, t))
                s += 1
            ok = t == ref
            out.append({"stage": s, "agree": float(ok.double().mean()),
                        "converged": float(ok.all(1).double().mean()),
                        "lead": float(ok.long().cumprod(1).sum(1).double().mean()),
                        "invalid": float((~torch.isfinite(true_block_logprob(hmm, alpha, t))).double().mean())})
        rows[kind] = out
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--T", default="4,16,64")
    p.add_argument("--stages", default="0,1,2,3,4,8,16,32,64")
    p.add_argument("--contexts", type=int, default=2048)
    p.add_argument("--context", type=int, default=16)
    p.add_argument("--vocab", type=int, default=512)
    p.add_argument("--hmm-seed", type=int, default=1234)
    p.add_argument("--kinds", default="cdf_id,cdf_sem,tree_sem,gumbel")
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default="")
    args = p.parse_args()
    torch.manual_seed(0)
    dev = torch.device(args.device)
    hmm = build_phrase_hmm(V=args.vocab, seed=args.hmm_seed, device=dev)
    seq = sample_sequences(hmm, args.contexts, args.context, torch.Generator(device=dev).manual_seed(5))
    alpha = filter_states(hmm, seq)[:, -1]
    result = {}
    for T in [int(x) for x in args.T.split(",")]:
        stages = [s for s in (int(x) for x in args.stages.split(",")) if s <= T]
        rows = run(T, stages, args.contexts, hmm, alpha, args.kinds.split(","))
        result[T] = rows
        print(f"\nT={T}: stage | " + " | ".join(f"{k}: agree / converged / invalid" for k in rows))
        for i, st in enumerate(stages):
            print(f"  {st:3d} | " + " | ".join(f"{rows[k][i]['agree']:.3f} / {rows[k][i]['converged']:.3f} / "
                                               f"{rows[k][i]['invalid']:.3f}" for k in rows))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
