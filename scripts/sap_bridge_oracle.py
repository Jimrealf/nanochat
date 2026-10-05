"""
S11 oracle: is the phrase-HMM toy sequential, or only sequential in left-to-right order?

The S10 Jacobi oracle found that refining tokens left to right fixes about one position per stage,
so a one-pass left-to-right emulator needs about T stages. That measures the decision ORDER, not
the process. Here the same exact HMM is sampled interface-first (nested dissection, as in sparse
direct solvers and Markov-bridge sampling): draw the hidden state at the block end given the
context, then the state at every interval midpoint given the interval's two end states, all
midpoints of a level at once, then every token from its own state. Each level is one parallel
step, and the sample is exact because the hidden chain is Markov: given its two end states an
interval's interior is independent of everything outside it.

Reported per T: the number of parallel levels, the invalid rate, and the mean -log p_true of the
samples against that of true sequences (equal in expectation for an exact sampler).

    python -m scripts.sap_bridge_oracle --T 4,16,64,256
"""
from __future__ import annotations

import argparse
import math

import torch

from scripts.sap_synthetic import build_phrase_hmm, filter_states, sample_sequences, true_block_logprob


def matrix_powers(A, ks):
    out = {}
    for k in sorted(set(ks)):
        out[k] = torch.linalg.matrix_power(A, k)
    return out


@torch.no_grad()
def bridge_sample(hmm, alpha, T, gen):
    """Interface-first exact sampling. alpha (B, S): belief over the state at the last context
    position. Returns tokens (B, T) and the number of parallel levels used."""
    B, S = alpha.shape
    A = hmm.A.double()
    states = torch.full((B, T + 1), -1, dtype=torch.long, device=alpha.device)
    states[:, 0] = torch.multinomial(alpha.double(), 1, generator=gen).squeeze(1)   # current state
    P = matrix_powers(A, [T] + [2 ** j for j in range(int(math.log2(max(T, 1))) + 2)] + list(range(1, T + 1)))
    states[:, T] = torch.multinomial(P[T][states[:, 0]], 1, generator=gen).squeeze(1)
    levels = 1
    intervals = [(0, T)]
    while any(b - a > 1 for a, b in intervals):
        nxt = []
        for a, b in intervals:                       # one level: every interval's midpoint at once
            if b - a <= 1:
                continue
            m = (a + b) // 2
            w = P[m - a][states[:, a]] * P[b - m][:, states[:, b]].t()      # (B, S) bridge weights
            states[:, m] = torch.multinomial(w / w.sum(1, keepdim=True), 1, generator=gen).squeeze(1)
            nxt += [(a, m), (m, b)]
        intervals = nxt
        levels += 1
    tokens = torch.multinomial(hmm.E.double()[states[:, 1:].reshape(-1)], 1, generator=gen).view(B, T)
    return tokens, levels + 1                       # + the parallel emission step


@torch.no_grad()
def token_bisection_sample(hmm, alpha, T, gen):
    """The same bisection order with TOKENS as the decision variables: at each level every new
    midpoint token is drawn from its exact posterior given all tokens already placed (forward-
    backward over the hidden chain with those observations), independently of the other new
    midpoints. Tokens do not separate the chain the way states do, so this is not exact; the gap
    measures what the choice of decision variable costs at the same depth."""
    B, S = alpha.shape
    A, E = hmm.A.double(), hmm.E.double()
    tokens = torch.full((B, T), -1, dtype=torch.long, device=alpha.device)
    placed = torch.zeros(T, dtype=torch.bool)
    order, intervals = [T - 1], [(-1, T - 1)]
    levels = [[T - 1]]
    while intervals:
        nxt, lvl = [], []
        for a, b in intervals:
            if b - a <= 1:
                continue
            m = (a + b) // 2
            lvl.append(m)
            nxt += [(a, m), (m, b)]
        if lvl:
            levels.append(lvl)
        intervals = nxt
    for lvl in levels:
        # forward-backward with emissions only at placed positions
        lik = torch.ones(B, T, S, dtype=torch.float64)
        for k in torch.nonzero(placed).flatten().tolist():
            lik[:, k] = E[:, tokens[:, k]].t()
        f = torch.empty(B, T, S, dtype=torch.float64)
        a = alpha.double() @ A
        for k in range(T):
            a = a * lik[:, k]
            a = a / a.sum(1, keepdim=True).clamp_min(1e-300)
            f[:, k] = a
            a = a @ A
        bwd = torch.ones(B, S, dtype=torch.float64)
        post = torch.empty(B, T, S, dtype=torch.float64)
        for k in range(T - 1, -1, -1):
            post[:, k] = f[:, k] * bwd
            post[:, k] = post[:, k] / post[:, k].sum(1, keepdim=True).clamp_min(1e-300)
            bwd = A @ (lik[:, k] * bwd).t()
            bwd = bwd.t()
            bwd = bwd / bwd.sum(1, keepdim=True).clamp_min(1e-300)
        for m in lvl:                                  # all midpoints of the level, independently
            probs = torch.nan_to_num(post[:, m] @ E)
            dead = probs.sum(1) <= 0                   # an earlier level already made the block impossible
            probs[dead] = 1.0
            tokens[:, m] = torch.multinomial(probs / probs.sum(1, keepdim=True), 1, generator=gen).squeeze(1)
            placed[m] = True
    return tokens, len(levels)


def _posterior_tokens(hmm, alpha, tokens, placed):
    """P(token_k | placed tokens) for every k (forward-backward over the hidden chain). (B, T, V)"""
    B, S = alpha.shape
    T = tokens.size(1)
    A, E = hmm.A.double(), hmm.E.double()
    lik = torch.ones(B, T, S, dtype=torch.float64, device=alpha.device)
    obs = placed.nonzero(as_tuple=False)
    for b, k in obs.tolist():
        lik[b, k] = E[:, tokens[b, k]]
    f = torch.empty(B, T, S, dtype=torch.float64, device=alpha.device)
    a = alpha.double() @ A
    for k in range(T):
        a = a * lik[:, k]
        a = a / a.sum(1, keepdim=True).clamp_min(1e-300)
        f[:, k] = a
        a = a @ A
    bwd = torch.ones(B, S, dtype=torch.float64, device=alpha.device)
    post = torch.empty(B, T, S, dtype=torch.float64, device=alpha.device)
    for k in range(T - 1, -1, -1):
        p_ = f[:, k] * bwd
        post[:, k] = p_ / p_.sum(1, keepdim=True).clamp_min(1e-300)
        bwd = (A @ (lik[:, k] * bwd).t()).t()
        bwd = bwd / bwd.sum(1, keepdim=True).clamp_min(1e-300)
    return post @ E


@torch.no_grad()
def ngram_bisection_sample(hmm, alpha, T, n, gen):
    """Bisection order with n-gram decisions: at each new position m the window m-n+1..m is drawn
    jointly from the exact posterior given the tokens already placed (its own window's earlier
    tokens included), independently of the other windows of the same level. n=1 is token
    bisection; larger n places short phrases, whose last tokens can separate the chain."""
    from nanochat.bridge import bisection_levels
    levels, _ = bisection_levels(T)
    B = alpha.size(0)
    tokens = torch.zeros(B, T, dtype=torch.long, device=alpha.device)
    placed = torch.zeros(B, T, dtype=torch.bool, device=alpha.device)
    for lvl in levels:
        base_tokens, base_placed = tokens.clone(), placed.clone()
        for m in lvl:
            tk, pl = base_tokens.clone(), base_placed.clone()
            for k in range(max(m - n + 1, 0), m + 1):
                if bool(pl[:, k].all()):
                    continue
                probs = torch.nan_to_num(_posterior_tokens(hmm, alpha, tk, pl)[:, k])
                dead = probs.sum(1) <= 0
                probs[dead] = 1.0
                draw = torch.multinomial(probs / probs.sum(1, keepdim=True), 1, generator=gen).squeeze(1)
                tk[:, k] = torch.where(pl[:, k], tk[:, k], draw)
                pl[:, k] = True
            lo = max(m - n + 1, 0)
            tokens[:, lo:m + 1], placed[:, lo:m + 1] = tk[:, lo:m + 1], pl[:, lo:m + 1]
    return tokens, len(levels) * n


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--T", default="4,16,64,256")
    p.add_argument("--contexts", type=int, default=4096)
    p.add_argument("--context", type=int, default=16)
    p.add_argument("--hmm-seed", type=int, default=1234)
    p.add_argument("--token-max-T", type=int, default=64, help="largest T for the token-bisection comparison")
    p.add_argument("--ngram", default="", help="comma list of n for the n-gram-decision comparison, e.g. 2,3")
    args = p.parse_args()
    torch.manual_seed(0)
    hmm = build_phrase_hmm(V=512, seed=args.hmm_seed)
    gen = torch.Generator().manual_seed(7)
    for T in [int(t) for t in args.T.split(",")]:
        seq = sample_sequences(hmm, args.contexts, args.context + T, gen)
        alpha = filter_states(hmm, seq[:, :args.context])[:, -1]
        true_block = seq[:, args.context:]
        tokens, levels = bridge_sample(hmm, alpha, T, gen)
        lp_s = true_block_logprob(hmm, alpha, tokens).double()
        lp_t = true_block_logprob(hmm, alpha, true_block).double()
        ok = torch.isfinite(lp_s)
        se = ((-lp_s[ok]).std() ** 2 / ok.sum() + (-lp_t).std() ** 2 / lp_t.numel()).sqrt()
        print(f"T={T:4d} states as decisions: {levels} parallel levels (left-to-right needs {T}); invalid "
              f"{(~ok).double().mean():.4f}; -log p_true of samples {(-lp_s[ok]).mean():.2f} vs true sequences "
              f"{(-lp_t).mean():.2f} (difference {((-lp_s[ok]).mean() - (-lp_t).mean()) / se:+.1f} se)")
        if T <= args.token_max_T:
            n = min(args.contexts, 1024)
            tok, lv = token_bisection_sample(hmm, alpha[:n], T, gen)
            lp = true_block_logprob(hmm, alpha[:n], tok).double()
            okt = torch.isfinite(lp)
            print(f"        tokens as decisions: {lv} levels; invalid {(~okt).double().mean():.4f}; "
                  f"-log p_true of valid samples {(-lp[okt]).mean():.2f}")
        for n_ in [int(v) for v in args.ngram.split(",") if v]:
            m_ = min(args.contexts, 512)
            tok, lv = ngram_bisection_sample(hmm, alpha[:m_], T, n_, gen)
            lp = true_block_logprob(hmm, alpha[:m_], tok).double()
            okt = torch.isfinite(lp)
            print(f"        {n_}-grams as decisions: {lv} sequential draws; invalid {(~okt).double().mean():.4f}; "
                  f"-log p_true of valid samples {(-lp[okt]).mean():.2f}")


if __name__ == "__main__":
    main()
