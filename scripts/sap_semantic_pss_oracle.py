"""S06-S: analytic semantic-permutation upper bound for PSS emissions.

The oracle keeps the phrase HMM's true hidden state and transition process.  Each hidden
state receives its best possible arbitrary token permutation, which is strictly more
expressive than a learned global semantic code plus an XOR/affine action.  What remains
shared is the sorted probability spectrum of the base token distribution.

No parameters are fitted to evaluation samples.  The oracle directly uses the known HMM
emission table, making this a labelled free upper bound rather than a proposed model.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import torch

from scripts.sap_synthetic import (
    build_phrase_hmm,
    filter_states,
    sample_sequences,
    true_block_logprob,
)


DEFAULT_BASES = (1, 2, 3, 5, 9)


def categorical_from_uniform(probs: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    return (u.unsqueeze(-1) > probs.cumsum(-1)).sum(-1).clamp_max(probs.size(-1) - 1)


def kmeans_rows(x: torch.Tensor, k: int, iterations: int = 50) -> torch.Tensor:
    """Deterministic farthest-first k-means; only eight topic spectra are clustered."""
    n = x.size(0)
    if k >= n:
        return torch.arange(n, device=x.device)
    centers = [0]
    min_d = (x - x[0]).square().sum(-1)
    for _ in range(1, k):
        nxt = int(min_d.argmax())
        centers.append(nxt)
        min_d = torch.minimum(min_d, (x - x[nxt]).square().sum(-1))
    c = x[torch.tensor(centers, device=x.device)].clone()
    assign = torch.zeros(n, dtype=torch.long, device=x.device)
    for _ in range(iterations):
        new_assign = (x[:, None, :] - c[None]).square().sum(-1).argmin(-1)
        if torch.equal(assign, new_assign):
            assign = new_assign
            break
        assign = new_assign
        for j in range(k):
            take = assign == j
            if take.any():
                c[j] = x[take].mean(0)
    return assign


class SemanticOracle:
    def __init__(self, hmm, bases: int):
        self.hmm, self.bases = hmm, bases
        E = hmm.E
        self.sorted_E, self.order = E.sort(-1, descending=True)  # rank -> token
        self.rank = torch.empty_like(self.order)                 # token -> rank
        ranks = torch.arange(hmm.V, device=E.device)[None].expand_as(self.order)
        self.rank.scatter_(1, self.order, ranks)

        support = (E > 0).sum(-1)
        topic_states = (support > 1).nonzero(as_tuple=False).squeeze(1)
        deterministic = support == 1
        if topic_states.numel() == 0:
            raise RuntimeError("semantic oracle expected stochastic topic states")

        # There are nine distinct spectrum types in the phrase HMM: one deterministic
        # type shared by phrase positions, plus one type for each stochastic topic state.
        self.type_of_state = torch.zeros(hmm.S, dtype=torch.long, device=E.device)
        self.type_of_state[topic_states] = 1 + torch.arange(topic_states.numel(), device=E.device)
        self.type_spectra = torch.cat((self.sorted_E[deterministic][:1],
                                       self.sorted_E[topic_states]), 0)
        self.n_types = self.type_spectra.size(0)

        if bases == 1:
            type_class = torch.zeros(self.n_types, dtype=torch.long, device=E.device)
        else:
            topic_k = min(bases - 1, topic_states.numel())
            topic_class = kmeans_rows(self.type_spectra[1:], topic_k)
            type_class = torch.cat((torch.zeros(1, dtype=torch.long, device=E.device),
                                    1 + topic_class), 0)
        self.type_class = type_class
        self.state_class = type_class[self.type_of_state]
        self.C = int(type_class.max()) + 1

    def state_marginals(self, alpha: torch.Tensor, T: int) -> torch.Tensor:
        vals, m = [], alpha
        for _ in range(T):
            m = m @ self.hmm.A
            vals.append(m)
        return torch.stack(vals, 1)  # B,T,S

    def base_distributions(self, alpha: torch.Tensor, T: int) -> torch.Tensor:
        """Optimal posterior-weighted sorted spectrum for every class/context/slot."""
        state_w = self.state_marginals(alpha, T)
        B = alpha.size(0)
        type_w = torch.zeros(B, T, self.n_types, device=alpha.device)
        for u in range(self.n_types):
            type_w[:, :, u] = state_w[:, :, self.type_of_state == u].sum(-1)
        q = torch.zeros(B, T, self.C, self.hmm.V, device=alpha.device)
        for c in range(self.C):
            take = self.type_class == c
            mass = type_w[:, :, take].sum(-1)
            numer = type_w[:, :, take] @ self.type_spectra[take]
            fallback = self.type_spectra[take].mean(0)
            q[:, :, c] = torch.where(
                (mass > 1e-30)[:, :, None], numer / mass.clamp_min(1e-30)[:, :, None],
                fallback[None, None],
            )
        return q

    def log_prob(self, alpha: torch.Tensor, y: torch.Tensor, q: torch.Tensor | None = None):
        q = self.base_distributions(alpha, y.size(1)) if q is None else q
        a = alpha
        lp = torch.zeros(y.size(0), device=y.device)
        for t in range(y.size(1)):
            pred = a @ self.hmm.A
            emit = torch.empty_like(pred)
            ranks = self.rank[:, y[:, t]].t()  # B,S
            for c in range(self.C):
                take = self.state_class == c
                emit[:, take] = q[:, t, c].gather(1, ranks[:, take])
            w = pred * emit
            norm = w.sum(-1)
            lp += norm.log()
            a = w / norm.clamp_min(1e-30)[:, None]
        return lp

    @torch.no_grad()
    def sample(self, alpha: torch.Tensor, T: int, q: torch.Tensor | None = None):
        q = self.base_distributions(alpha, T) if q is None else q
        B, S = alpha.size()
        # Draw the complete primitive random tape before any realised-state computation.
        u_context = torch.rand(B, device=alpha.device)
        u_tokens = torch.rand(B, T, device=alpha.device)
        context_state = categorical_from_uniform(alpha, u_context)
        # One vectorised draw gives one next-state choice for every (batch,time,source-state)
        # without materialising a B*T*S*S transition tensor.
        maps = torch.multinomial(self.hmm.A, B * T, replacement=True).t().reshape(B, T, S)
        # Associative Hillis-Steele prefix composition of all transition random maps.
        pref = maps
        off = 1
        while off < T:
            nxt = pref.clone()
            nxt[:, off:] = pref[:, off:].gather(-1, pref[:, :-off])
            pref = nxt
            off *= 2
        states = pref.gather(-1, context_state[:, None, None].expand(B, T, 1)).squeeze(-1)
        cls = self.state_class[states]
        b = torch.arange(B, device=alpha.device)[:, None]
        t = torch.arange(T, device=alpha.device)[None]
        selected_q = q[b, t, cls]
        ranks = categorical_from_uniform(selected_q, u_tokens)
        return self.order[states, ranks]


@torch.no_grad()
def evaluate(args):
    torch.manual_seed(args.seed)
    if args.device == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    hmm = build_phrase_hmm(V=args.vocab, seed=args.hmm_seed, device=device)
    gen = torch.Generator(device=device).manual_seed(91_000 + args.seed)
    seq = sample_sequences(hmm, args.contexts, args.context + args.T, gen)
    alpha = filter_states(hmm, seq[:, :args.context])[:, -1]
    y = seq[:, args.context:args.context + args.T]
    true_lp = true_block_logprob(hmm, alpha, y)
    bases_list = [int(x) for x in args.bases.split(",") if x]
    per_example = {}
    t0 = time.time()
    for bases in bases_list:
        oracle = SemanticOracle(hmm, bases)
        kls, invalids, nlls = [], [], []
        for start in range(0, args.contexts, args.chunk):
            aa = alpha[start:start + args.chunk]
            yy = y[start:start + args.chunk]
            q = oracle.base_distributions(aa, args.T)
            lp = oracle.log_prob(aa, yy, q)
            sample = oracle.sample(aa, args.T, q)
            slp = true_block_logprob(hmm, aa, sample)
            kls.append((true_lp[start:start + args.chunk] - lp).cpu())
            invalids.append((~torch.isfinite(slp)).float().cpu())
            nlls.append(torch.where(torch.isfinite(slp), -slp, torch.nan).cpu())
        kl = torch.cat(kls)
        invalid = torch.cat(invalids)
        nll = torch.cat(nlls)
        per_example[bases] = (kl, invalid, nll, oracle)
        print(f"C={bases} (actual {oracle.C}) complete", flush=True)

    budgets = sorted(set(min(args.contexts, n) for n in (args.small_contexts, args.contexts)))
    rows = []
    for bases in bases_list:
        kl, invalid, nll, oracle = per_example[bases]
        for n in budgets:
            x, z = kl[:n], invalid[:n]
            valid_nll = nll[:n][torch.isfinite(nll[:n])]
            row = {
                "requested_spectrum_classes": bases,
                "actual_spectrum_classes": oracle.C,
                "contexts": n,
                "block_kl": float(x.mean()),
                "block_kl_se": float(x.std(unbiased=True) / math.sqrt(n)),
                "invalid_rate": float(z.mean()),
                "invalid_se": float(z.std(unbiased=True) / math.sqrt(n)),
                "sample_nll_valid": float(valid_nll.mean()) if valid_nll.numel() else None,
                "gate": "PASS" if float(x.mean()) <= 0.10 and float(z.mean()) <= 0.03 else "FAIL",
                "oracle": "free true-state/per-state-permutation upper bound",
                "fitted_parameters": 0,
                "permutation_entries": hmm.S * hmm.V,
                "stochastic_rounds": 1,
            }
            rows.append(row)
    result = {
        "experiment": "S06-S semantic-PSS oracle",
        "T": args.T,
        "vocab": args.vocab,
        "hmm_states": hmm.S,
        "distinct_emission_spectra": per_example[bases_list[-1]][3].n_types,
        "seed": args.seed,
        "seconds": round(time.time() - t0, 2),
        "rows": rows,
    }
    print("S06S_RESULT " + json.dumps(result, sort_keys=True), flush=True)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2, sort_keys=True)
    return result


def compile_text(result: dict) -> str:
    lines = [
        "S06-S — semantic-PSS free oracle",
        "================================",
        "True HMM state/transition/context plus arbitrary optimal per-state token permutations.",
        "This strictly upper-bounds a learned global semantic code with XOR/affine actions.",
        "No evaluation-sample parameters are fitted. Gate: block KL <= 0.10 and invalid <= 3%.",
        "",
        f"T={result['T']} V={result['vocab']} HMM states={result['hmm_states']} "
        f"distinct spectra={result['distinct_emission_spectra']} seed={result['seed']}",
        "",
        f"{'C':>3s} {'N':>7s} {'KL':>10s} {'KL SE':>10s} {'invalid':>10s} {'inv SE':>10s}  gate",
        "-" * 72,
    ]
    for r in result["rows"]:
        lines.append(f"{r['actual_spectrum_classes']:3d} {r['contexts']:7d} "
                     f"{r['block_kl']:10.6f} {r['block_kl_se']:10.6f} "
                     f"{r['invalid_rate']:10.6f} {r['invalid_se']:10.6f}  {r['gate']}")
    lines += ["", "Full result", "-----------", json.dumps(result, indent=2, sort_keys=True), ""]
    return "\n".join(lines)


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--bases", default=",".join(map(str, DEFAULT_BASES)))
    p.add_argument("--contexts", type=int, default=16384)
    p.add_argument("--small-contexts", type=int, default=4096)
    p.add_argument("--chunk", type=int, default=512)
    p.add_argument("--context", type=int, default=16)
    p.add_argument("--T", type=int, default=4)
    p.add_argument("--vocab", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--hmm-seed", type=int, default=1234)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default="")
    p.add_argument("--compiled-out", default="")
    p.add_argument("--smoke", action="store_true")
    return p


def main():
    args = build_parser().parse_args()
    if args.smoke:
        args.bases, args.contexts, args.small_contexts, args.chunk = "1,2,9", 32, 16, 16
    result = evaluate(args)
    text = compile_text(result)
    print(text)
    if args.compiled_out:
        os.makedirs(os.path.dirname(args.compiled_out) or ".", exist_ok=True)
        with open(args.compiled_out, "w") as f:
            f.write(text)


if __name__ == "__main__":
    main()
