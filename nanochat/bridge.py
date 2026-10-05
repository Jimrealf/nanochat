"""
Bridge LM (S11-A, s11_sap_tl_brainstorm.md): one-pass T=L generation by sampling separator codes
coarse-to-fine, then every token at once.

Every position k carries a discrete code z_k, a deterministic function of the text (the decision
variables are computable from data, so everything trains by teacher forcing, with no model
sampling in the loop). Codes are generated in bisection order: the block end first, then the
midpoint of every interval between known positions, all midpoints of a level in parallel, each
drawn from a learned categorical that sees only coarser codes and the context. Tokens are then
emitted in parallel given all codes. Generation takes len(levels) + 1 parallel steps, which is
ceil(log2 T) + 2; when the codes form a Markov chain the within-level independence is exact (the
exact toy version is scripts/sap_bridge_oracle.py).

Likelihood: codes are deterministic given the text, so
    log p(z(t)) + log p(t | z(t)) <= log sum_z p(z) p(t | z) = log p_model(t),
a teacher-forced lower bound; a block KL computed from it is an upper bound.

Code sources (the decision-variable experiment):
    "oracle"  argmax of the exact HMM filtering belief after each token (toy upper bound);
    "tokens"  z = t (learned token bisection: the decision-variable control; emission is the
              identity, so the likelihood is exact);
    "ar"      learned separators, two stages, all teacher forced: a causal AR model is trained by
              cross-entropy for ar_steps, then frozen, and its state after each token is clustered
              by k-means into K codes; the bridge prior and the emission then train on those fixed
              codes. Stationary targets, no sampling in training (s11 root cause 0);
    "ar_pred" the same two stages, but pasts are clustered by what they predict: KL (Bregman)
              k-means on the frozen AR model's next-token distribution. This approximates the causal
              states of computational mechanics (pasts with the same conditional future), the minimal
              sufficient statistic of the past, whose process is Markov; Euclidean k-means on hidden
              states ignores that criterion.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def bisection_levels(T, with_brackets=False):
    """Positions grouped by generation level: the block end first, then the midpoint of every
    interval between known positions (position -1 is the context). Returns (levels, level_of), and
    with with_brackets also (T, 2) the interval each position was drawn inside (-1 = context)."""
    levels, level_of = [[T - 1]], torch.zeros(T, dtype=torch.long)
    brackets = torch.full((T, 2), -1, dtype=torch.long)
    intervals = [(-1, T - 1)]
    while True:
        nxt, lvl = [], []
        for a, b in intervals:
            if b - a > 1:
                m = (a + b) // 2
                lvl.append(m)
                brackets[m] = torch.tensor([a, b])
                nxt += [(a, m), (m, b)]
        if not lvl:
            break
        level_of[lvl] = len(levels)
        levels.append(lvl)
        intervals = nxt
    return (levels, level_of, brackets) if with_brackets else (levels, level_of)


def window_bisection_levels(T, n):
    """Generation steps for window bisection: take the bisection midpoints level by level, but at
    each midpoint m place the window m-n+1..m (its positions not placed yet), one token per step
    inside a window and every window of a level in parallel. Returns (steps, step_of): steps[s] lists
    the positions drawn at step s. n=1 is plain bisection; the windows' last tokens act as
    separators (scripts/sap_bridge_oracle.py: 2-token windows sample the toy HMM with 0% invalid
    at T=64, single tokens with 44%)."""
    levels, _ = bisection_levels(T)
    step_of = torch.full((T,), -1, dtype=torch.long)
    steps, base = [], 0
    for lvl in levels:
        width = 0
        for m in lvl:
            j = 0
            for k in range(max(m - n + 1, 0), m + 1):
                if step_of[k] < 0:
                    step_of[k] = base + j
                    j += 1
            width = max(width, j)
        base += width
    for st in range(base):
        steps.append([k for k in range(T) if step_of[k] == st])
    return [st for st in steps if st], step_of


@torch.no_grad()
def filtered_argmax_states(hmm, alpha, y):
    """Toy oracle codes: argmax_s P(state_k = s | context, y_<=k) for every block position. (B, T)."""
    a, out = alpha, []
    for k in range(y.size(1)):
        w = (a @ hmm.A) * hmm.E[:, y[:, k]].t()
        a = w / w.sum(1, keepdim=True).clamp_min(1e-30)
        out.append(a.argmax(1))
    return torch.stack(out, 1)


class _Bidirectional(nn.Module):
    def __init__(self, width, depth):
        super().__init__()
        layer = nn.TransformerEncoderLayer(width, max(1, width // 32), 4 * width, dropout=0.0,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)

    def forward(self, h):
        return self.norm(self.blocks(h))


class BridgeLM(nn.Module):
    """Toy Bridge LM over a block of T tokens given a context vector (the oracle belief)."""
    exact = True                # log_prob is a teacher-forced lower bound (exact for codes="tokens")
    bound = True                # trained through loss()
    training_multiplier = 1

    def __init__(self, ctx_dim, T, V, width=128, depth=4, emit_depth=2, codes="oracle", hmm=None,
                 K=512, ar_steps=4000, kmeans_batches=16, endpoints=False, poe=False, window=1):
        super().__init__()
        assert codes in ("oracle", "tokens", "ar", "ar_pred")
        assert codes != "oracle" or hmm is not None
        self.T, self.V, self.codes, self.hmm = T, V, codes, hmm
        self.K = V if codes == "tokens" else (hmm.S if codes == "oracle" else K)
        if codes in ("ar", "ar_pred"):
            from nanochat.ptp import _Causal
            self.ar_steps, self.kmeans_batches, self.step = ar_steps, kmeans_batches, 0
            self.ar_ctx = nn.Sequential(nn.Linear(ctx_dim, width), nn.SiLU(), nn.Linear(width, width))
            self.ar_pos = nn.Parameter(torch.randn(T + 1, width) / math.sqrt(width))
            self.ar_tok = nn.Embedding(V + 1, width)                       # index V = block start
            self.ar = _Causal(width, depth)
            self.ar_head = nn.Linear(width, V)
            self.register_buffer("codebook", torch.zeros(0, width))
            self._bank = []
        levels, level_of, brackets = bisection_levels(T, with_brackets=True)
        if window > 1:                                                   # window bisection (tokens only)
            assert codes == "tokens" and not endpoints and not poe
            levels, level_of = window_bisection_levels(T, window)
        self.levels, self.endpoints, self.poe = levels, endpoints, poe
        self.register_buffer("level_of", level_of, persistent=False)
        self.register_buffer("brackets", brackets, persistent=False)
        self.ctx = nn.Sequential(nn.Linear(ctx_dim, width), nn.SiLU(), nn.Linear(width, width))
        self.pos = nn.Parameter(torch.randn(T, width) / math.sqrt(width))
        self.lvl = nn.Embedding(len(levels), width)
        self.code_emb = nn.Embedding(self.K + 1, width)               # index K = not yet known
        if endpoints:                                                    # the Markov-bridge inductive bias:
            self.left_emb = nn.Embedding(self.K + 1, width)               # each midpoint reads the codes at
            self.right_emb = nn.Embedding(self.K + 1, width)              # its interval's two ends (K = context)
        self.prior = _Bidirectional(width, depth)
        self.prior_head = nn.Linear(width, self.K)
        if poe:
            # Exact Markov-bridge form: log P(z_m | z_a, z_b) = log P(z_m | z_a; d1) + log P(z_b | z_m; d2)
            # + const, so the logits get one learned table per (level, left end code) and one per
            # (level, right end code); the transformer's logits correct whatever is not Markov.
            n = len(levels)
            self.poe_left = nn.Parameter(torch.zeros(n, self.K + 1, self.K))
            self.poe_right = nn.Parameter(torch.zeros(n, self.K + 1, self.K))
        if codes != "tokens":
            self.emit_in = nn.Embedding(self.K, width)
            self.emit = _Bidirectional(width, emit_depth)
            self.emit_head = nn.Linear(width, V)

    # ---------------------------------------------------------------- pieces
    def ar_states(self, ctx, y):
        """State after each token: position k + 1 of the causal stack over [start, y]. (B, T, W)"""
        inp = torch.cat([torch.full_like(y[:, :1], self.V), y], 1)
        h = self.ar_ctx(ctx)[:, None] + self.ar_pos[None, :inp.size(1)] + self.ar_tok(inp)
        return self.ar(h)

    def ar_nll(self, ctx, y):
        lg = self.ar_head(self.ar_states(ctx, y)[:, :-1])
        return F.cross_entropy(lg.flatten(0, 1).float(), y.flatten(), reduction="none").view(y.shape).sum(1)

    def _assign(self, x, cb):
        if self.codes == "ar_pred":                                      # argmin_j KL(p || c_j)
            return (x @ cb.clamp_min(1e-12).log().t()).argmax(1)
        return torch.cdist(x, cb).argmin(1)

    @torch.no_grad()
    def _fit_codebook(self, states, iters=25):
        x = states.reshape(-1, states.size(-1)).float()
        cb = x[torch.randperm(x.size(0), device=x.device)[:self.K]].clone()
        for _ in range(iters):
            a = self._assign(x, cb)
            for j in torch.unique(a):                                    # keep dead centroids in place
                cb[j] = x[a == j].mean(0)                                # (the mean distribution for KL)
        self.codebook = cb

    def _code_features(self, ctx, y):
        h = self.ar_states(ctx, y)[:, 1:].float()
        if self.codes == "ar_pred":
            return self.ar_head(h).float().softmax(-1)                   # predicted next-token distribution
        return h

    @torch.no_grad()
    def code_of(self, ctx, y):
        if self.codes == "tokens":
            return y
        if self.codes == "oracle":
            return filtered_argmax_states(self.hmm, ctx, y)
        h = self._code_features(ctx, y)
        if self.codebook.numel() == 0:                                    # evaluated before stage 2
            self._fit_codebook(h)
        return self._assign(h.reshape(-1, h.size(-1)), self.codebook).view(y.shape)

    def prior_logits(self, ctx, z, level):
        """Logits for every position from the codes of strictly coarser levels (others masked)."""
        known = (self.level_of < level)[None].expand_as(z)
        inp = self.code_emb(torch.where(known, z, torch.full_like(z, self.K)))
        h = inp + self.pos[None] + self.lvl(self.level_of)[None] + self.ctx(ctx)[:, None]
        ends = None
        if self.endpoints or self.poe:                                   # only the queried level reads them
            ends = []
            for j in (0, 1):
                idx = self.brackets[:, j]
                ends.append(torch.where(idx[None] >= 0, z.gather(1, idx.clamp_min(0)[None].expand_as(z)),
                                        torch.full_like(z, self.K)))
        q = (self.level_of == level)[None, :, None]
        if self.endpoints:
            h = h + q * (self.left_emb(ends[0]) + self.right_emb(ends[1]))
        logits = self.prior_head(self.prior(h))
        if self.poe:
            logits = logits + q * (self.poe_left[level][ends[0]] + self.poe_right[level][ends[1]])
        return logits

    def emit_logits(self, ctx, z):
        h = self.emit_in(z) + self.pos[None] + self.ctx(ctx)[:, None]
        return self.emit_head(self.emit(h))

    def _prior_nll(self, ctx, z):
        """Per-row negative log-likelihood of the codes in bisection order. (B,)"""
        nll = torch.zeros(z.size(0), device=z.device)
        for l, pos in enumerate(self.levels):
            lg = self.prior_logits(ctx, z, l)[:, pos]
            nll = nll + F.cross_entropy(lg.flatten(0, 1).float(), z[:, pos].flatten(), reduction="none").view(
                z.size(0), -1).sum(1)
        return nll

    def _emit_nll(self, ctx, z, y):
        if self.codes == "tokens":
            return torch.zeros(y.size(0), device=y.device)
        lg = self.emit_logits(ctx, z)
        return F.cross_entropy(lg.flatten(0, 1).float(), y.flatten(), reduction="none").view(y.shape).sum(1)

    # ---------------------------------------------------------------- harness API
    def loss(self, ctx, y):
        if self.codes in ("ar", "ar_pred"):
            self.step += 1
            if self.step <= self.ar_steps:                               # stage 1: the AR model alone
                nll = self.ar_nll(ctx, y)
                if self.step > self.ar_steps - self.kmeans_batches:
                    with torch.no_grad():
                        self._bank.append(self._code_features(ctx, y))
                if self.step == self.ar_steps:                            # freeze, then cluster its states
                    self._fit_codebook(torch.cat(self._bank))
                    self._bank = []
                    for prm in [*self.ar_ctx.parameters(), self.ar_pos, *self.ar_tok.parameters(),
                                *self.ar.parameters(), *self.ar_head.parameters()]:
                        prm.requires_grad_(False)
                return nll.mean() / self.T, {"ar_nats": float(nll.mean().detach())}
        z = self.code_of(ctx, y)
        prior, emit = self._prior_nll(ctx, z), self._emit_nll(ctx, z, y)
        total = (prior + emit).mean() / self.T
        return total, {"prior_nats": float(prior.mean().detach()), "emit_nats": float(emit.mean().detach())}

    @torch.no_grad()
    def log_prob(self, ctx, y):
        z = self.code_of(ctx, y)
        return -(self._prior_nll(ctx, z) + self._emit_nll(ctx, z, y))

    @torch.no_grad()
    def sample(self, ctx):
        """len(levels) parallel code steps, then one parallel emission step."""
        B = ctx.size(0)
        z = torch.full((B, self.T), self.K, dtype=torch.long, device=ctx.device)
        for l, pos in enumerate(self.levels):
            probs = self.prior_logits(ctx, z, l)[:, pos].float().softmax(-1)
            z[:, pos] = torch.multinomial(probs.flatten(0, 1), 1).view(B, len(pos))
        if self.codes == "tokens":
            return z
        probs = self.emit_logits(ctx, z).float().softmax(-1)
        return torch.multinomial(probs.flatten(0, 1), 1).view(B, self.T)

    @torch.no_grad()
    def diagnostics(self, ctx, y, true_lp):
        z = self.code_of(ctx, y)
        nll_levels = []
        for l, pos in enumerate(self.levels):
            lg = self.prior_logits(ctx, z, l)[:, pos]
            nll_levels.append(round(float(F.cross_entropy(lg.flatten(0, 1).float(), z[:, pos].flatten())), 4))
        return {"parallel_steps": len(self.levels) + (0 if self.codes == "tokens" else 1),
                "prior_nats_per_code_by_level": nll_levels,
                "emit_nats_per_token": float(self._emit_nll(ctx, z, y).mean() / self.T)}
