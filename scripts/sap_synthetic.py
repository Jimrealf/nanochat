"""
SAP Stage A: every block-head mode on a synthetic language whose block joint is known exactly.

Why a synthetic language
------------------------
The question Stage A answers is whether a head that emits T tokens in one pass can learn the
DEPENDENCE between them when it is pretrained from scratch with that head. Judging that on real
text needs a reference distribution, and every reference we could use on real text is a
pretrained model that was not trained this way. Here the reference is the generator itself, so
every number is exact: block log-probabilities, the block's total correlation (how much a
product of marginals must lose), and whether a sampled block is even possible.

The generator is a "phrase HMM". Hidden states are either a free topic (emitting from a sparse
topic unigram) or a position inside a multi-token phrase (emitting that phrase's token
deterministically). Phrases come in families that share their first token and then fork, like
"New York City" and "New Jersey Turnpike", so a head that samples its slots independently can
emit a block that is impossible ("New York Turnpike"). Its probability under the generator is
exactly 0, which is the mode-mixing rate this script reports.

Per mode it reports (all per block of T tokens, in nats):
  block_kl        E_true[log p_true(block) - log p_model(block)]: exact for indep/local/cp, an
                  upper bound for the latent modes (importance-weighted) and inv_head
  tc              the true total correlation, the block-KL of the BEST possible
                  independent-slot head
  ar_block_kl     the same quantity for the trunk's own next-token factorisation
  invalid_rate    fraction of blocks sampled from the head that are impossible under truth
  sample_nll      mean -log p_true of the sampled blocks that are possible
  sensitivity     how much the slot distributions move with the latent draw (0 = ignored)

Usage
-----
    python -m scripts.sap_synthetic --smoke                       # 2 modes, ~1 min, checks wiring
    python -m scripts.sap_synthetic --T 2 4 --seeds 2             # the Stage A grid
    python -m scripts.sap_synthetic --modes indep p1_discrete --steps 3000
    MODAL_PROFILE=... modal run modal_sap.py::stage_a              # the grid in parallel on Modal

Results go to <out>/results.jsonl, one line per (mode, T, seed), and a table to stdout with the
pre-registered gate verdicts from sap_research_plan.md.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from nanochat.block_head import SAP_MODES, target_bins
from nanochat.gpt import GPT, GPTConfig


# ----------------------------------------------------------------------------- generator
@dataclass
class PhraseHMM:
    A: torch.Tensor        # (S, S) transition matrix
    E: torch.Tensor        # (S, V) emission matrix
    pi0: torch.Tensor      # (S,) initial distribution
    V: int

    @property
    def S(self):
        return self.A.size(0)


def build_phrase_hmm(V=512, K=8, families=32, per_family=3, len_lo=3, len_hi=6,
                     p_phrase=0.3, stay=0.9, alpha=0.1, seed=0, device="cpu") -> PhraseHMM:
    """A small language with forks: phrase families share a first token, then diverge."""
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)  # the Dirichlet draws below use the global generator
    V_free = V // 2
    # Free topics: sparse unigrams over the free half of the vocabulary.
    topic_uni = torch.distributions.Dirichlet(torch.full((V_free,), alpha)).sample((K,))
    phrases = []
    first_tokens = torch.randperm(V - V_free, generator=g)[:families] + V_free
    for f in range(families):
        for _ in range(per_family):
            L = int(torch.randint(len_lo, len_hi + 1, (1,), generator=g))
            rest = torch.randint(V_free, V, (L - 1,), generator=g)
            phrases.append(torch.cat([first_tokens[f:f + 1], rest]))
    P = len(phrases)
    offsets, S = [], K
    for ph in phrases:
        offsets.append(S)
        S += len(ph)
    A = torch.zeros(S, S)
    E = torch.zeros(S, V)
    E[:K, :V_free] = topic_uni
    # Topic-dependent phrase preferences and phrase-dependent topic after the phrase ends.
    pref = torch.distributions.Dirichlet(torch.full((P,), 0.3)).sample((K,))
    after = torch.distributions.Dirichlet(torch.full((K,), 0.5)).sample((P,))
    for k in range(K):
        A[k, :K] = (1 - p_phrase) * (1 - stay) / (K - 1)
        A[k, k] = (1 - p_phrase) * stay
        for p, off in enumerate(offsets):
            A[k, off] = p_phrase * pref[k, p]
    for p, (ph, off) in enumerate(zip(phrases, offsets)):
        for j, tok in enumerate(ph.tolist()):
            E[off + j, tok] = 1.0
            if j + 1 < len(ph):
                A[off + j, off + j + 1] = 1.0
            else:
                A[off + j, :K] = after[p]
    A = A / A.sum(1, keepdim=True)
    pi0 = torch.zeros(S)
    pi0[:K] = 1.0 / K
    return PhraseHMM(A.to(device), E.to(device), pi0.to(device), V)


@torch.no_grad()
def sample_sequences(hmm: PhraseHMM, B: int, L: int, gen=None) -> torch.Tensor:
    dev = hmm.A.device
    s = torch.multinomial(hmm.pi0.expand(B, -1), 1, generator=gen).squeeze(1)
    out = torch.empty(B, L, dtype=torch.long, device=dev)
    for i in range(L):
        out[:, i] = torch.multinomial(hmm.E[s], 1, generator=gen).squeeze(1)
        s = torch.multinomial(hmm.A[s], 1, generator=gen).squeeze(1)
    return out


@torch.no_grad()
def filter_states(hmm: PhraseHMM, tokens: torch.Tensor) -> torch.Tensor:
    """alpha[b, t] = P(state_t | tokens[b, :t+1]). (B, L, S)."""
    B, L = tokens.shape
    alpha = torch.empty(B, L, hmm.S, device=tokens.device)
    a = hmm.pi0[None] * hmm.E[:, tokens[:, 0]].t()
    a = a / a.sum(1, keepdim=True)
    alpha[:, 0] = a
    for t in range(1, L):
        a = (a @ hmm.A) * hmm.E[:, tokens[:, t]].t()
        a = a / a.sum(1, keepdim=True).clamp_min(1e-30)
        alpha[:, t] = a
    return alpha


@torch.no_grad()
def true_block_logprob(hmm: PhraseHMM, alpha_t: torch.Tensor, blocks: torch.Tensor) -> torch.Tensor:
    """log P(blocks | context) for alpha_t (N, S) and blocks (N, T). -inf when impossible."""
    a = alpha_t
    lp = torch.zeros(blocks.size(0), device=blocks.device)
    for k in range(blocks.size(1)):
        w = (a @ hmm.A) * hmm.E[:, blocks[:, k]].t()
        c = w.sum(1)
        lp = lp + torch.log(c.clamp_min(0.0))
        a = w / c.clamp_min(1e-30)[:, None]
    return lp


@torch.no_grad()
def true_marginal_logprob(hmm: PhraseHMM, alpha_t: torch.Tensor, blocks: torch.Tensor) -> torch.Tensor:
    """sum_k log P(token k steps ahead = blocks[:, k] | context): the ideal independent head."""
    m = alpha_t
    lp = torch.zeros(blocks.size(0), device=blocks.device)
    for k in range(blocks.size(1)):
        m = m @ hmm.A
        pk = (m @ hmm.E).gather(1, blocks[:, k:k + 1]).squeeze(1)
        lp = lp + torch.log(pk.clamp_min(1e-30))
    return lp


@torch.no_grad()
def true_next_entropy(hmm: PhraseHMM, alpha: torch.Tensor) -> torch.Tensor:
    p = (alpha @ hmm.A) @ hmm.E
    return -(p * p.clamp_min(1e-30).log()).sum(-1)


# ----------------------------------------------------------------------------- model
def build_model(args, mode, T, V, device):
    cfg = GPTConfig(
        sequence_len=args.seq_len, vocab_size=V, n_layer=args.depth,
        n_head=max(1, args.n_embd // 64), n_kv_head=max(1, args.n_embd // 64), n_embd=args.n_embd,
        window_pattern="L",
        sap_block_T=T, sap_block_mode=mode, sap_block_frac=args.block_frac,
        sap_head_layers=args.head_layers, sap_ctx_window=args.ctx_window,
        sap_latent_groups=args.latent_groups, sap_latent_codes=args.latent_codes,
        sap_latent_dim=args.latent_dim, sap_free_bits=args.free_bits,
        sap_kl_anneal_steps=args.kl_anneal, sap_cp_components=args.cp_components,
        sap_wta_k=args.wta_k,
    )
    with torch.device("meta"):
        model = GPT(cfg)
    model.to_empty(device=device)
    model.init_weights()
    return model


def lr_at(step, total, warmup):
    if step < warmup:
        return (step + 1) / warmup
    return max(0.05, 1.0 - (step - warmup) / max(1, total - warmup))


class SequencePool:
    """Fresh generator sequences, produced `size` at a time and each used once.

    Sampling the HMM is sequential in time, so a batch of 32 costs ~500 tiny kernel launches
    and dominated the step (57 of ~70 ms on an RTX 3050 Ti). Generating 1,024 rows at once
    costs about the same wall time as 32, and since no row is reused the training stream has
    exactly the distribution of per-step sampling.
    """

    def __init__(self, hmm, length, size, gen):
        self.hmm, self.length, self.size, self.gen = hmm, length, size, gen
        self.pool, self.used = None, size

    def batch(self, B):
        if self.used + B > self.size:
            self.pool = sample_sequences(self.hmm, max(self.size, B), self.length, self.gen)
            self.used = 0
        out = self.pool[self.used:self.used + B]
        self.used += B
        return out


def train(model, hmm, args, device, seed):
    gen = torch.Generator(device=device).manual_seed(10_000 + seed)
    pool = SequencePool(hmm, args.seq_len + 1, args.pool, gen)
    opt = model.setup_optimizer()
    model.train()
    t0 = time.time()
    last = {}
    for step in range(args.steps):
        mult = lr_at(step, args.steps, args.warmup)
        for grp in opt.param_groups:
            grp["lr"] = grp["initial_lr"] * mult
        tok = pool.batch(args.batch)
        loss = model(tok[:, :-1].contiguous(), tok[:, 1:].contiguous())
        loss.backward()
        opt.step()
        model.zero_grad(set_to_none=True)
        if step % args.log_every == 0 or step == args.steps - 1:
            st = {k: (v.tolist() if v.numel() > 1 else round(v.item(), 4))
                  for k, v in (getattr(model, "_sap_stats", None) or {}).items()}
            last = {"step": step, "loss": round(loss.item(), 4), **st}
            print(f"    step {step:5d} loss {loss.item():.4f} {st} ({time.time() - t0:.0f}s)", flush=True)
    return last


# ----------------------------------------------------------------------------- evaluation
@torch.no_grad()
def evaluate(model, hmm, args, device, T, seed):
    model.eval()
    head = model.sap_head
    gen = torch.Generator(device=device).manual_seed(99_999 + seed)
    tok = sample_sequences(hmm, args.eval_seqs, args.seq_len + 1, gen)
    x, y = tok[:, :-1].contiguous(), tok[:, 1:].contiguous()
    logits, hid = model(x, return_hidden=True)
    alpha = filter_states(hmm, x)
    # next-token quality of the trunk against the exact conditional entropy
    ntp_ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1)).item()
    ent = true_next_entropy(hmm, alpha.reshape(-1, hmm.S)).mean().item()

    positions = [p for p in args.eval_positions if p + T <= args.seq_len]
    B = x.size(0)
    b = torch.arange(B, device=device).repeat(len(positions))
    t = torch.tensor(positions, device=device).repeat_interleave(B)
    ks = torch.arange(T, device=device)
    yb = y[b[:, None], t[:, None] + ks[None, :]]                          # true continuations
    a_t = alpha[b, t]
    lp_true = true_block_logprob(hmm, a_t, yb)
    lp_marg = true_marginal_logprob(hmm, a_t, yb)
    tc = (lp_true - lp_marg).mean().item()
    ar = torch.log_softmax(logits[b[:, None], t[:, None] + ks[None, :]].float(), -1)
    lp_ar = ar.gather(-1, yb[..., None]).squeeze(-1).sum(-1)
    ar_block_kl = (lp_true - lp_ar).mean().item()

    ctx, cv = model._sap_window(hid, b, t)
    u_bins = None
    if head.mode == "inv_head" and T > 1:
        rows = logits[b[:, None], t[:, None] + ks[None, :T - 1]].reshape(-1, logits.size(-1))
        lo, w = target_bins(rows, yb[:, :T - 1].reshape(-1))
        u_bins = (lo.view(-1, T - 1), w.view(-1, T - 1))
    elif head.mode == "inv_head":
        u_bins = (torch.zeros(yb.size(0), 0, device=device),) * 2
    lp_model = head.block_logprob(hid[b, t], ctx, cv, yb, model._sap_readout,
                                  embed=model.transformer.wte, u_bins=u_bins,
                                  n_samples=args.iwae_samples)
    block_kl = None if lp_model is None else (lp_true - lp_model).mean().item()

    # sample-based coherence, comparable across every mode including the likelihood-free one
    S = args.samples_per_ctx
    hs = hid[b, t].repeat_interleave(S, dim=0)
    ctx_s = None if ctx is None else ctx.repeat_interleave(S, dim=0)
    cv_s = None if cv is None else cv.repeat_interleave(S, dim=0)
    gs = torch.Generator(device=device).manual_seed(4242 + seed)
    samp = head.sample(hs, ctx_s, cv_s, model._sap_readout, embed=model.transformer.wte,
                       embed_table=model.transformer.wte.weight, temperature=1.0, generator=gs)
    lp_s = true_block_logprob(hmm, a_t.repeat_interleave(S, dim=0), samp)
    invalid = (~torch.isfinite(lp_s)).float().mean().item()
    sample_nll = (-lp_s[torch.isfinite(lp_s)]).mean().item() if torch.isfinite(lp_s).any() else float("nan")
    # Plan-latent heads only: the same blocks decoded from a plan the recognition model drew
    # from the TRUE block. Clean here but dirty above means the prior is the bottleneck;
    # dirty in both means the decoder (or the plan's capacity) is.
    invalid_posterior = None
    if head.mode in ("p1_discrete", "p2_gauss"):
        samp_q = head.sample(hs, ctx_s, cv_s, model._sap_readout, embed=model.transformer.wte,
                             embed_table=model.transformer.wte.weight, temperature=1.0, generator=gs,
                             posterior_y=yb.repeat_interleave(S, dim=0))
        lp_q = true_block_logprob(hmm, a_t.repeat_interleave(S, dim=0), samp_q)
        invalid_posterior = (~torch.isfinite(lp_q)).float().mean().item()

    # the trunk's own autoregressive samples: the reference invalid rate
    ar_invalid = None
    if args.ar_reference:
        pref_ok = []
        for pos in positions:
            pre = x[:, :pos + 1].repeat_interleave(S, dim=0)
            for _ in range(T):
                lg = model(pre)[:, -1]
                nxt = torch.multinomial(torch.softmax(lg.float(), -1), 1, generator=gs)
                pre = torch.cat([pre, nxt], dim=1)
            blk = pre[:, -T:]
            lp_ar_s = true_block_logprob(hmm, alpha[:, pos].repeat_interleave(S, dim=0), blk)
            pref_ok.append(torch.isfinite(lp_ar_s).float())
        ar_invalid = 1.0 - torch.cat(pref_ok).mean().item()

    sens = head.latent_sensitivity(hid[b, t], ctx, cv, model._sap_readout, n_samples=8)
    return {
        "ntp_ce": ntp_ce, "true_entropy": ent, "ntp_excess": ntp_ce - ent,
        "tc": tc, "block_kl": block_kl, "ar_block_kl": ar_block_kl,
        "invalid_rate": invalid, "invalid_rate_posterior": invalid_posterior,
        "ar_invalid_rate": ar_invalid, "sample_nll": sample_nll,
        "sensitivity": None if sens is None else sens.mean().item(),
        "head_flops_per_token": head.flops_per_token(hmm.V),
        "contexts": int(yb.size(0)),
    }


# ----------------------------------------------------------------------------- gates
def gate_verdicts(rows):
    """The Stage A gates from sap_research_plan.md, applied per (T, seed) group."""
    out = []
    by = {}
    for r in rows:
        by.setdefault((r["T"], r["seed"]), {})[r["mode"]] = r
    for (T, seed), g in sorted(by.items()):
        b2 = g.get("indep")
        b3 = g.get("cp")
        tc = next(iter(g.values()))["tc"]
        out.append(f"T={T} seed={seed}: true total correlation {tc:.3f} nats/block")
        if b2 is not None and b2["block_kl"] is not None:
            out.append(f"  sanity B2: block_kl {b2['block_kl']:.3f} vs tc {tc:.3f} "
                       f"({'ok' if b2['block_kl'] >= 0.8 * tc else 'BELOW tc: estimator or tc suspect'})")
        for name in ("p1_discrete", "p2_gauss"):
            r = g.get(name)
            if r is None or b2 is None or r["block_kl"] is None:
                continue
            kl_used = r.get("train_kl_nats")
            checks = [
                ("block_kl <= 0.5 x B2", r["block_kl"] <= 0.5 * b2["block_kl"]),
                ("latent carries >= 0.5 x tc", kl_used is not None and kl_used >= 0.5 * tc),
            ]
            if b3 is not None and b3["block_kl"] is not None:
                checks.append(("better than cp mixture", r["block_kl"] < b3["block_kl"]))
            verdict = "ADVANCE" if all(c for _, c in checks) else "KILL"
            out.append(f"  {name}: {verdict} | " + ", ".join(f"{n}: {'yes' if c else 'no'}" for n, c in checks))
        r = g.get("p3_energy")
        if r is not None and b2 is not None:
            ok = r["invalid_rate"] <= 0.7 * b2["invalid_rate"]
            out.append(f"  p3_energy: {'ADVANCE' if ok else 'KILL'} | invalid {r['invalid_rate']:.3f} "
                       f"vs B2 {b2['invalid_rate']:.3f} (needs <= 0.7x)")
        r = g.get("plain_noise")
        if r is not None and r["sensitivity"] is not None:
            out.append(f"  control plain_noise: sensitivity {r['sensitivity']:.4f} nats "
                       f"({'ignores its noise, as predicted' if r['sensitivity'] < 0.05 else 'USES its noise: the motivating claim fails here'})")
    return out


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--modes", nargs="+", default=list(SAP_MODES), choices=list(SAP_MODES))
    p.add_argument("--T", nargs="+", type=int, default=[2, 4])
    p.add_argument("--seeds", type=int, default=1)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--n-embd", type=int, default=128)
    p.add_argument("--vocab", type=int, default=512)
    p.add_argument("--block-frac", type=float, default=0.125)
    p.add_argument("--head-layers", type=int, default=2)
    p.add_argument("--ctx-window", type=int, default=16)
    p.add_argument("--latent-groups", type=int, default=4)
    p.add_argument("--latent-codes", type=int, default=16)
    p.add_argument("--latent-dim", type=int, default=64)
    p.add_argument("--free-bits", type=float, default=0.25)
    p.add_argument("--kl-anneal", type=int, default=500)
    p.add_argument("--cp-components", type=int, default=16)
    p.add_argument("--wta-k", type=int, default=4)
    p.add_argument("--eval-seqs", type=int, default=128)
    p.add_argument("--eval-positions", nargs="+", type=int, default=[48, 96, 144, 192])
    p.add_argument("--samples-per-ctx", type=int, default=4)
    p.add_argument("--iwae-samples", type=int, default=32)
    p.add_argument("--ar-reference", type=int, default=1, help="also sample blocks from the trunk's own next-token head")
    p.add_argument("--log-every", type=int, default=250)
    p.add_argument("--hmm-seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", type=str, default="out/sap_synthetic")
    p.add_argument("--pool", type=int, default=1024, help="sequences generated per refill of the training stream (each used once)")
    p.add_argument("--smoke", action="store_true", help="2 modes, T=2, 200 steps: checks the wiring only")
    return p


def apply_smoke(args):
    if args.smoke:
        args.modes = ["indep", "p1_discrete"]
        args.T, args.steps, args.eval_seqs, args.iwae_samples = [2], 200, 32, 8
        args.log_every = 100
        args.out = args.out + "_smoke"
    return args


def run_one(args, mode, T, seed, hmm=None):
    """Train one (mode, T, seed) from scratch on the phrase HMM and score it. Returns the row."""
    device = torch.device(args.device)
    if hmm is None:
        hmm = build_phrase_hmm(V=args.vocab, seed=args.hmm_seed, device=device)
    torch.manual_seed(seed)
    print(f"== mode={mode} T={T} seed={seed}", flush=True)
    t0 = time.time()
    model = build_model(args, mode, T, hmm.V, device)
    last = train(model, hmm, args, device, seed)
    res = evaluate(model, hmm, args, device, T, seed)
    kl = last.get("kl_per_group")
    if kl is not None:
        res["train_kl_nats"] = float(sum(kl) if isinstance(kl, list) else kl)
    elif last.get("kl_per_block") is not None:
        res["train_kl_nats"] = float(last["kl_per_block"])
    row = {"mode": mode, "T": T, "seed": seed, "steps": args.steps,
           "seconds": round(time.time() - t0, 1), "final_train": last, **res}
    bk = "  n/a " if res["block_kl"] is None else f"{res['block_kl']:6.3f}"
    print(f"   -> block_kl {bk} | tc {res['tc']:.3f} | ar_block_kl {res['ar_block_kl']:.3f} "
          f"| invalid {res['invalid_rate']:.3f} (AR {res['ar_invalid_rate']}) "
          f"| ntp_excess {res['ntp_excess']:.3f} | sens {res['sensitivity']}", flush=True)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return row


def main():
    args = apply_smoke(build_parser().parse_args())
    os.makedirs(args.out, exist_ok=True)
    device = torch.device(args.device)
    hmm = build_phrase_hmm(V=args.vocab, seed=args.hmm_seed, device=device)
    print(f"phrase HMM: {hmm.S} states, V={hmm.V}")
    rows = []
    path = os.path.join(args.out, "results.jsonl")
    for T in args.T:
        for seed in range(args.seeds):
            for mode in args.modes:
                row = run_one(args, mode, T, seed, hmm)
                rows.append(row)
                with open(path, "a") as f:
                    f.write(json.dumps(row) + "\n")
    print("\n".join(["", "Stage A gates", *gate_verdicts(rows)]))
    print(f"\nresults appended to {path}")


if __name__ == "__main__":
    main()
