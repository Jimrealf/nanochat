"""
RC-PTP: reference-corrected, self-inverting Parallel Token Prediction (S10, s10_sap_tl_brainstorm.md).

One-pass generation of a whole block (T=L). Auxiliaries are drawn once per position; one causal call
emits every position's distribution from the auxiliaries of earlier positions, and every token is
read off at once with a pick, t_k = Pick(u_k, P_k) (C-PTP, Draxler et al., ICLR 2026).

What this adds to PTP:
  1. parallel self-inversion: a token-conditioned AR mode, trained by ordinary CE, gives every
     position's distribution in one teacher-forced pass, so data tokens invert to auxiliaries at once
     (PTP's from-scratch training inverts under the generator itself, one call per position).
  2. a mid-pass cut and a reference module: the generator picks a draft halfway up the stack and the
     upper layers read the drafted tokens as well as the auxiliaries (S01 measured that a reference
     module helps only on consistent anchors; these drafts are tied to the final picks through u);
  3. coupled noise: the same u_k drives the draft and the final pick, so where the draft
     distribution is already right the final token equals the draft;
  4. a semantic vocabulary order for the picks;
  5. (iteration 1) a robust pick. PTP's pick is the inverse CDF of one uniform, which is arithmetic
     decoding down a binary tree with the uniform rescaled at every level. The rescaling makes it
     fragile: two distributions at KL 0.003 pick different tokens from the same u 21% of the time
     over V=512 (Gumbel-max: 3%, the best any coupling can do: 2.4%). The tree pick keeps the tree
     and draws an independent uniform per level, which bounds the disagreement by the sum of the
     branch-share differences along the path (measured 4.8% at the same KL) with log2(V) uniforms
     per position instead of V Gumbels.

Likelihood of the one-pass sampler (the pushforward of u): log_prob is sequential importance
sampling along the generator's own posterior. p_gen(t) = E[prod_k |I_k(u_<k)|] with each u_k drawn
uniformly inside I_k, the cell that picks t_k (an interval, or a box for the tree pick, of volume
P_k(t_k)), so a path weight is unbiased for p_gen(t), its log-mean-exp is a lower bound on
log p_gen(t), and a block KL computed from it is an upper bound.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------- flat inverse-CDF pick
def _walk_cdf(probs, rank):
    """Inclusive CDF of probs walked in rank order (rank[v] = walk position of token v), in float64
    and ending at exactly 1, plus the walk order (order[r] = token at walk position r)."""
    order = torch.argsort(rank)
    cdf = probs.double()[..., order].cumsum(-1)
    return cdf / cdf[..., -1:], order


def ordered_cdf_bounds(probs, tokens, rank):
    """[lo, hi): the auxiliaries that pick each token. probs (..., V), tokens (...); float64."""
    cdf, _ = _walk_cdf(probs, rank)
    r = rank[tokens][..., None]
    hi = cdf.gather(-1, r).squeeze(-1)
    lo = cdf.gather(-1, (r - 1).clamp_min(0)).squeeze(-1).masked_fill(r.squeeze(-1) == 0, 0.0)
    return lo, hi


def ordered_pick(probs, u, rank):
    """Inverse-CDF pick: the token whose [lo, hi) contains u. probs (..., V), u (...)."""
    cdf, order = _walk_cdf(probs, rank)
    r = torch.searchsorted(cdf, u.double()[..., None].contiguous(), right=True).squeeze(-1)
    return order[r.clamp_max(probs.size(-1) - 1)]


# ----------------------------------------------------------------------------- tree pick
def tree_levels(V):
    return max(1, math.ceil(math.log2(V)))


def _tree_sums(probs, rank):
    """Prefix sums of probs in rank order, zero-padded to 2^L leaves: (..., 2^L + 1), float64."""
    V, L = probs.size(-1), tree_levels(probs.size(-1))
    q = F.pad(probs.double()[..., torch.argsort(rank)], (0, 2 ** L - V))
    return torch.cat([torch.zeros_like(q[..., :1]), q.cumsum(-1)], -1), L


def _left_share(cs, node, width):
    """Share of the node [node, node + width) mass in its left half (0 for a massless node)."""
    at = lambda i: cs.gather(-1, i[..., None]).squeeze(-1)
    base = at(node)
    total = at(node + width) - base
    return torch.where(total > 0, (at(node + width // 2) - base) / total.clamp_min(1e-300), torch.zeros_like(total))


def tree_pick(probs, u, rank):
    """Descend a binary tree over the rank order with one uniform per level, going right when
    u_l >= the left child's share of the node. probs (..., V), u (..., L)."""
    cs, L = _tree_sums(probs, rank)
    node = torch.zeros(probs.shape[:-1], dtype=torch.long, device=probs.device)
    width = 2 ** L
    for level in range(L):
        share = _left_share(cs, node, width)
        width //= 2
        node = node + (u[..., level].double() >= share).long() * width
    return torch.argsort(rank)[node.clamp_max(probs.size(-1) - 1)]


def tree_bounds(probs, tokens, rank):
    """Per-level [lo, hi) of the uniforms that pick each token, (..., L) each, float64; the product of
    the widths is the token's probability (a product of branch shares down its path)."""
    cs, L = _tree_sums(probs, rank)
    leaf = rank[tokens]
    node = torch.zeros_like(leaf)
    width = 2 ** L
    los, his = [], []
    for _ in range(L):
        share = _left_share(cs, node, width)
        width //= 2
        right = (leaf - node) >= width
        los.append(torch.where(right, share, torch.zeros_like(share)))
        his.append(torch.where(right, torch.ones_like(share), share))
        node = node + right.long() * width
    return torch.stack(los, -1), torch.stack(his, -1)


# ----------------------------------------------------------------------------- shared helpers
def uniform_in(lo, hi, gen=None):
    """u uniform in [lo, hi), kept strictly below hi after rounding. float64."""
    r = torch.rand(lo.shape, device=lo.device, dtype=torch.float64, generator=gen)
    return place(lo, hi, r)


def place(lo, hi, rel):
    """The point at relative offset rel in [lo, hi), kept strictly below hi after rounding."""
    return torch.minimum(lo + rel * (hi - lo), torch.nextafter(hi, lo)).maximum(lo)


def invert(probs, tokens, rank, gen=None):
    """Auxiliaries under which probs picks the given tokens (flat inverse CDF)."""
    return uniform_in(*ordered_cdf_bounds(probs, tokens, rank), gen=gen)


def semantic_rank(emission):
    """Walk order from an (S, V) emission matrix: tokens grouped by the state that emits them most,
    strongest first within a group. Returns rank (V,)."""
    strength, best = emission.max(0)
    order = torch.argsort(best.double() * 2.0 - strength.double(), stable=True)
    rank = torch.empty_like(order)
    rank[order] = torch.arange(order.numel(), device=order.device)
    return rank


def binary_digits(u, bits):
    """PTP's arithmetic-coding embedding of an auxiliary (their [ar] variant): its first binary
    digits. u (...) -> (..., bits)."""
    scale = torch.pow(2.0, torch.arange(1, bits + 1, device=u.device, dtype=torch.float64))
    return torch.floor(u.double()[..., None] * scale).remainder(2.0)


class _Causal(nn.Module):
    def __init__(self, width, depth):
        super().__init__()
        layer = nn.TransformerEncoderLayer(width, max(1, width // 32), 4 * width, dropout=0.0,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)

    def forward(self, h):
        T = h.size(1)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=h.device), 1)
        return self.norm(self.blocks(h, mask=mask, is_causal=True))


class RCPTP(nn.Module):
    """Toy RC-PTP over a block of T tokens given a context vector (the oracle belief).

    cut=False is self-inverting C-PTP without the reference module; coupled=False gives the draft
    its own noise; rank=None walks the vocabulary in id order; coupling is "cdf" (PTP's pick, one
    uniform per position) or "tree" (one uniform per level of a binary tree over the vocabulary).
    stages generalises the cut: the generator's layers split into that many groups, each ending in a
    head whose picks (under the same auxiliaries) feed the next group one position later, so the
    pass holds stages - 1 coupled refinement steps (default 2 with the cut, 1 without).

    inversion picks where the training auxiliaries come from:
      "ar"     the AR mode's distributions (one parallel pass); the generator distils the AR
               conditionals, the target under which its own picks reproduce the inversion map;
      "seq"    the generator's own final distributions, one position at a time (PTP's Eq. 13);
      "jacobi" the AR-mode inversion refined by n_inv parallel sweeps under the generator's own.
    With "seq" and "jacobi" the generator is trained by cross-entropy on the data (Eq. 13), so it
    conditions on its own picks rather than on another model's.

    Auxiliaries are float64 tensors (B, T, levels): levels = 1 for "cdf", ceil(log2 V) for "tree"."""
    exact = True              # log_prob is a lower bound on log p_gen, so the block KL is an upper bound
    bound = True              # trained through loss(), not -log_prob
    training_multiplier = 2   # the AR-mode pass plus the generator pass

    def __init__(self, ctx_dim, T, V, width=128, depth=4, rank=None, cut=True, coupled=True,
                 bits=24, is_samples=16, inversion="ar", n_inv=2, coupling="cdf", stages=None):
        super().__init__()
        n = stages or (2 if cut else 1)
        assert depth >= n and inversion in ("ar", "seq", "jacobi") and coupling in ("cdf", "tree")
        self.T, self.V, self.cut, self.coupled = T, V, n > 1, coupled
        self.inversion, self.n_inv, self.coupling = inversion, n_inv, coupling
        self.levels = 1 if coupling == "cdf" else tree_levels(V)
        self.bits, self.is_samples = bits, is_samples
        self.register_buffer("rank", torch.arange(V) if rank is None else rank.clone(), persistent=False)
        self.ctx = nn.Sequential(nn.Linear(ctx_dim, width), nn.SiLU(), nn.Linear(width, width))
        self.pos = nn.Parameter(torch.randn(T, width) / math.sqrt(width))
        # AR mode: token-conditioned and teacher-forced; its distributions invert data into auxiliaries.
        self.ar_tok = nn.Embedding(V + 1, width)                           # index V = block start
        self.ar = _Causal(width, depth)
        self.ar_head = nn.Linear(width, V)
        # Generator: auxiliaries of earlier positions in; each stage ends in a head, and its picks
        # feed the next stage (the first stage is the draft, the later ones the reference module).
        self.u_in = nn.Linear(self.levels * (bits + 1), width)
        self.u_start = nn.Parameter(torch.zeros(width))
        sizes = [depth // n + (1 if i >= n - depth % n else 0) for i in range(n)]
        self.gen = nn.ModuleList(_Causal(width, d) for d in sizes)
        self.heads = nn.ModuleList(nn.Linear(width, V) for _ in range(n))
        self.tok_in = nn.ModuleList(nn.Embedding(V + 1, width) for _ in range(n - 1))

    @property
    def draft_head(self):
        return self.heads[0]

    @property
    def final_head(self):
        return self.heads[-1]

    # ---------------------------------------------------------------- picks
    def noise(self, B, T, device):
        return torch.rand(B, T, self.levels, dtype=torch.float64, device=device)

    def pick(self, logits, u):
        probs = logits.double().softmax(-1)
        if self.coupling == "cdf":
            return ordered_pick(probs, u[..., 0], self.rank)
        return tree_pick(probs, u, self.rank)

    def cell(self, logits, tokens):
        """Per-level [lo, hi) of the auxiliaries that pick the tokens, (..., levels) each."""
        probs = logits.double().softmax(-1)
        if self.coupling == "cdf":
            lo, hi = ordered_cdf_bounds(probs, tokens, self.rank)
            return lo[..., None], hi[..., None]
        return tree_bounds(probs, tokens, self.rank)

    # ---------------------------------------------------------------- passes
    def _start(self, x):
        return torch.full_like(x[:, :1], self.V)

    def _u_embed(self, u):
        """Position k carries u_{k-1}, so with causal attention it sees exactly u_<k."""
        feats = torch.cat([binary_digits(u, self.bits), u.double()[..., None] - 0.5], -1).flatten(-2)
        e = self.u_in(feats.to(self.u_in.weight.dtype))
        return torch.cat([self.u_start.expand(u.size(0), 1, -1), e[:, :-1]], dim=1)

    def ar_logits(self, ctx, y):
        h = self.ctx(ctx)[:, None] + self.pos[None, :y.size(1)] + self.ar_tok(torch.cat([self._start(y), y[:, :-1]], 1))
        return self.ar_head(self.ar(h))

    def stage_logits(self, ctx, u, u_draft=None):
        """The one pass over auxiliaries u (B, L, levels), L <= T: every stage's logits, and the
        tokens each non-final stage picks, which enter the next stage one position later."""
        h = self.ctx(ctx)[:, None] + self.pos[None, :u.size(1)] + self._u_embed(u)
        logits, picks = [], []
        for s, (block, head) in enumerate(zip(self.gen, self.heads)):
            if s:
                h = h + self.tok_in[s - 1](torch.cat([self._start(picks[-1]), picks[-1][:, :-1]], 1))
            h = block(h)
            logits.append(head(h))
            if s + 1 < len(self.gen):
                with torch.no_grad():
                    picks.append(self.pick(logits[-1], u if self.coupled else u_draft))
        return logits, picks

    def generate_logits(self, ctx, u, u_draft=None):
        """(first-stage logits, final logits, last drafted tokens or None)."""
        logits, picks = self.stage_logits(ctx, u, u_draft)
        return logits[0], logits[-1], (picks[-1] if picks else None)

    # ---------------------------------------------------------------- self-inversion
    def _invert_seq(self, ctx, y, ud, rel):
        """Exact inversion under the generator's own final distributions, one position at a time."""
        u = torch.zeros(rel.shape, dtype=torch.float64, device=y.device)
        for k in range(y.size(1)):
            fl = self.generate_logits(ctx, u[:, :k + 1], ud[:, :k + 1])[1][:, k]
            u[:, k] = place(*self.cell(fl, y[:, k]), rel[:, k])
        return u

    def _invert_sweeps(self, ctx, y, u, ud, rel, sweeps):
        """Parallel fixed-point sweeps of the inversion under the generator's own final
        distributions, each position keeping its relative offsets rel; position k is exact after
        k + 1 sweeps. Returns u and the share of positions the last sweep still moved (0 once the
        generator is consistent)."""
        moved = torch.zeros((), dtype=torch.float64, device=y.device)
        for _ in range(sweeps):
            new = place(*self.cell(self.generate_logits(ctx, u, ud)[1], y), rel)
            moved, u = (new != u).any(-1).double().mean(), new
        return u, moved

    def _loss_self(self, ctx, y):
        al = self.ar_logits(ctx, y)
        loss_ar = F.cross_entropy(al.flatten(0, 1).float(), y.flatten())
        with torch.no_grad():
            rel = self.noise(*y.shape, y.device)
            ud = self.noise(*y.shape, y.device)
            if self.inversion == "seq":
                u, moved = self._invert_seq(ctx, y, ud, rel), None
            else:
                u, moved = self._invert_sweeps(ctx, y, place(*self.cell(al, y), rel), ud, rel, self.n_inv)
        logits, picks = self.stage_logits(ctx, u, ud)
        ces = [F.cross_entropy(l.flatten(0, 1).float(), y.flatten()) for l in logits]   # PTP Eq. 13
        total = loss_ar + sum(ces)
        aux = {"loss_ar": float(loss_ar.detach()), "loss_gen": float(ces[-1].detach())}
        if moved is not None:
            aux["inv_moved"] = float(moved)
        if picks:
            aux["draft_hits_data"] = float((picks[-1] == y).float().mean())
        return total, aux

    # ---------------------------------------------------------------- training / harness API
    def loss(self, ctx, y):
        if self.inversion != "ar":
            return self._loss_self(ctx, y)
        al = self.ar_logits(ctx, y)
        loss_ar = F.cross_entropy(al.flatten(0, 1).float(), y.flatten())
        with torch.no_grad():
            u = place(*self.cell(al, y), self.noise(*y.shape, y.device))    # parallel inversion
            q = al.double().softmax(-1).float()                             # the AR conditionals to distil
            ent = -(q * q.clamp_min(1e-30).log()).sum(-1).mean()
        logits, picks = self.stage_logits(ctx, u, self.noise(*y.shape, y.device))
        ces = [-(q * l.float().log_softmax(-1)).sum(-1).mean() for l in logits]
        total = loss_ar + sum(ces)
        aux = {"loss_ar": float(loss_ar.detach()), "kl_draft": float((ces[0] - ent).detach())}
        if picks:
            aux["kl_final"] = float((ces[-1] - ent).detach())
            aux["draft_hits_data"] = float((picks[-1] == y).float().mean())
        return total, aux

    @torch.no_grad()
    def sample(self, ctx, u=None, u_draft=None):
        """One-pass sampling of the whole block."""
        if u is None:
            u = self.noise(ctx.size(0), self.T, ctx.device)
        fl = self.generate_logits(ctx, u, torch.rand_like(u) if u_draft is None else u_draft)[1]
        return self.pick(fl, u)

    @torch.no_grad()
    def ar_sample(self, ctx, u):
        """The AR mode's sequential sampler under the given auxiliaries (one call per position)."""
        y = torch.zeros(u.shape[:2], dtype=torch.long, device=u.device)
        for k in range(u.size(1)):
            y[:, k] = self.pick(self.ar_logits(ctx, y[:, :k + 1])[:, k], u[:, k])
        return y

    @torch.no_grad()
    def log_prob(self, ctx, y, S=None):
        """Sequential importance sampling along the generator's own posterior, S paths per row;
        returns log of the mean path weight (B,)."""
        S = S or self.is_samples
        rows, out = max(1, 4096 // S), []
        for i in range(0, y.size(0), rows):
            c = ctx[i:i + rows].repeat_interleave(S, 0)
            t = y[i:i + rows].repeat_interleave(S, 0)
            u = torch.zeros(*t.shape, self.levels, device=t.device, dtype=torch.float64)
            ud = torch.rand_like(u)
            logw = torch.zeros(t.size(0), device=t.device, dtype=torch.float64)
            for k in range(t.size(1)):
                lo, hi = self.cell(self.generate_logits(c, u[:, :k + 1], ud[:, :k + 1])[1][:, k], t[:, k])
                logw = logw + torch.log(hi - lo).sum(-1)
                u[:, k] = uniform_in(lo, hi)
            out.append(torch.logsumexp(logw.view(-1, S), dim=1) - math.log(S))
        return torch.cat(out).float()

    @torch.no_grad()
    def diagnostics(self, ctx, y, true_lp):
        """Exact block KL of the AR mode, agreement of the one-pass generator with the AR mode's
        sequential sampler under shared auxiliaries, and the per-position gap KL(AR || generator)
        at the data under the AR-mode inversion."""
        al = self.ar_logits(ctx, y)
        ar_lp = al.double().log_softmax(-1).gather(-1, y[..., None]).squeeze(-1).sum(-1)
        u = self.noise(*y.shape, y.device)
        ref = self.ar_sample(ctx, u)
        logits, picks = self.stage_logits(ctx, u, torch.rand_like(u))
        ok = self.pick(logits[-1], u) == ref
        q = al.double().softmax(-1)
        g = self.generate_logits(ctx, place(*self.cell(al, y), self.noise(*y.shape, y.device)),
                                 torch.rand_like(u))[1].double().log_softmax(-1)
        kl_pos = (q * (q.clamp_min(1e-300).log() - g)).sum(-1).mean(0)
        out = {"ar_block_kl": float((true_lp.double() - ar_lp).mean()),
               "agree_final": float(ok.float().mean()),
               "lead_correct": float(ok.long().cumprod(-1).sum(-1).float().mean()),   # PTP's metric
               "agree_by_pos": [round(float(a), 4) for a in ok.float().mean(0)][:64],
               "gen_kl_by_pos": [round(float(k), 4) for k in kl_pos][:64]}
        if picks and self.coupled:
            out["agree_draft"] = float((picks[0] == ref).float().mean())
            out["agree_by_stage"] = [round(float((p == ref).float().mean()), 4) for p in picks]
        return out
