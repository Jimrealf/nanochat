"""
SAP: a sampling-aware block head that emits the next T tokens in one parallel pass.

Why this exists
---------------
A head that reads the trunk state h_t and writes T token distributions in one pass is,
whatever its internals, a set of T distributions fixed BEFORE anything is sampled. Sampling
them independently draws from the product of marginals, so a fork ("New" York vs "Los"
Angeles) comes out mixed. Attention among the slots does not change that: it mixes features
that are all deterministic functions of the context.

What can change it is a per-sample variable that reaches every slot before the tokens are
read out. Here that variable is a plan latent z, drawn once per block, and the slots are
decoded from (context, z). Given (context, z) the slots are conditionally independent by
construction, so all of the dependence between them has to travel through z. That is also
why the latent cannot be ignored: a z that carries nothing leaves the decoder at the product
of marginals, which costs exactly the block's total correlation in extra nats.

How z is trained decides whether this works. Drawing z from the prior and applying plain CE
against the one observed block makes every z chase the same target and the decoder learns to
ignore z (Condor, arXiv 2609.06324, measures 0.02 nats of noise sensitivity this way). The
sound rules are:

  * posterior sampling: z comes from an encoder that has seen the true block, the decoder
    reproduces a block its latent already encodes, and a KL term pulls the prior onto the
    posterior (the ELBO; modes p1_discrete and p2_gauss);
  * a strictly proper scoring rule over samples: the energy score is minimised in
    expectation only by the true conditional distribution, even with one hard target per
    context (mode p3_energy, likelihood-free).

Modes (config.sap_block_mode)
-----------------------------
  indep        B2  no latent. The LCA-style head with independent slots: the product of
                   marginals every other mode is measured against. Exact likelihood.
  p1_discrete  P1  product-quantised plan, G groups x C codes, tiny autoregressive prior
                   over the groups, posterior encoder, ELBO with free bits.
  p2_gauss     P2  diagonal-Gaussian plan with the same encoder and decoder, ELBO.
  p3_energy    P3  noise drives the plan; the decoder emits vectors in the (normalised)
                   input-embedding space and is trained with the energy score.
  cp           B3  mixture of R components of independent slots. Exact likelihood.
  local        B4  local autoregressive head: slot k reads the tokens of slots < k.
                   Exact likelihood; sequential inside the head when decoding.
  inv_head     B5  PTP-style competitor restricted to the head: slot k reads uniform
                   noise u_<k that, in training, is inverted from the data tokens through
                   the next-token distribution's CDF.
  plain_noise  control: prior noise with plain CE. Expected to learn to ignore the noise.
  wta          control: winner-take-all over k noise draws (Condor's symmetry breaking).

Every mode decodes with the same attention decoder: slot queries attend to each other and to
a short window of the trunk's final hidden states, and the logits come from the model's own
lm_head (with its softcap), so the head adds no second unembedding.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

SAP_MODES = ("indep", "p1_discrete", "p2_gauss", "p3_energy", "cp", "local",
             "inv_head", "plain_noise", "wta")
LATENT_MODES = ("p1_discrete", "p2_gauss")
NOISE_MODES = ("p3_energy", "plain_noise", "wta")
# Modes whose slots may only read earlier slots (their inputs carry earlier tokens or noise).
CAUSAL_MODES = ("local", "inv_head")


def _norm(x):
    return F.rms_norm(x, (x.size(-1),)).to(x.dtype)


class _Lin(nn.Linear):
    """Bias-free linear that runs in the activation dtype, like nanochat's Linear.

    Kept local rather than imported from gpt.py, which imports this module lazily.
    """

    def __init__(self, fan_in, fan_out):
        super().__init__(fan_in, fan_out, bias=False)

    def forward(self, x):
        return F.linear(x, self.weight.to(dtype=x.dtype))


class _Attention(nn.Module):
    """Slots as queries; keys/values are the context window followed by the slots."""

    def __init__(self, d, n_head):
        super().__init__()
        assert d % n_head == 0, f"n_embd={d} must be divisible by n_head={n_head}"
        self.n_head, self.hd = n_head, d // n_head
        self.c_q, self.c_k, self.c_v, self.c_proj = _Lin(d, d), _Lin(d, d), _Lin(d, d), _Lin(d, d)

    def forward(self, xq, xkv, mask):
        N, T, d = xq.shape
        S = xkv.size(1)
        q = self.c_q(xq).view(N, T, self.n_head, self.hd).transpose(1, 2)
        k = self.c_k(xkv).view(N, S, self.n_head, self.hd).transpose(1, 2)
        v = self.c_v(xkv).view(N, S, self.n_head, self.hd).transpose(1, 2)
        q, k = _norm(q), _norm(k)  # QK norm, as in the trunk
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask[:, None])
        return self.c_proj(y.transpose(1, 2).reshape(N, T, d))


class _MLP(nn.Module):
    def __init__(self, d, mult):
        super().__init__()
        self.c_fc, self.c_proj = _Lin(d, mult * d), _Lin(mult * d, d)

    def forward(self, x):
        return self.c_proj(F.relu(self.c_fc(x)).square())


class _Layer(nn.Module):
    def __init__(self, d, n_head, mult):
        super().__init__()
        self.attn, self.mlp = _Attention(d, n_head), _MLP(d, mult)

    def forward(self, s, ctx, mask):
        sn = _norm(s)
        kv = sn if ctx is None else torch.cat([ctx, sn], dim=1)
        s = s + self.attn(sn, kv, mask)
        return s + self.mlp(_norm(s))


def _target_logprob(states, targets, readout, chunk):
    """log p(target) for each row of `states` under `readout`.

    The (rows, V) logit tensor is built one chunk at a time and recomputed in backward, so
    T slots on a fraction of positions never hold a second full-batch logit tensor.
    """
    def _chunk(s, y):
        logits = readout(s)
        return logits.gather(-1, y[:, None]).squeeze(-1) - torch.logsumexp(logits, dim=-1)

    out = []
    for i in range(0, states.size(0), chunk):
        s, y = states[i:i + chunk], targets[i:i + chunk]
        if torch.is_grad_enabled() and s.requires_grad:
            out.append(checkpoint(_chunk, s, y, use_reentrant=False))
        else:
            out.append(_chunk(s, y))
    return torch.cat(out) if out else states.new_zeros(0, dtype=torch.float32)


def u_features(u, n_freq):
    """Sinusoidal features of u in [0, 1) fine enough to resolve narrow CDF bins.

    Frequencies run up to 2^(n_freq-1) * pi, so with n_freq=24 two values 1e-7 apart still
    differ in the top feature. PTP feeds 32 binary digits instead; both are resolution knobs.
    """
    freqs = (2.0 ** torch.arange(n_freq, device=u.device, dtype=torch.float32)) * math.pi
    ang = u.float()[..., None] * freqs
    return torch.cat([u.float()[..., None], torch.sin(ang), torch.cos(ang)], dim=-1)


def pick(probs, u):
    """Inverse-CDF sampling: the token whose CDF bin contains u. probs (..., V), u (...)."""
    cdf = probs.float().cumsum(-1)
    idx = torch.searchsorted(cdf, u.float()[..., None].contiguous()).squeeze(-1)
    return idx.clamp(max=probs.size(-1) - 1)


@torch.no_grad()
def target_bins(logits_rows, targets, chunk=4096):
    """Lower edge and width of each target's CDF bin under softmax(logits_rows).

    Used by the inv_head competitor to invert data tokens into noise. ID order, as in PTP.
    """
    lo_all, p_all = [], []
    V = logits_rows.size(-1)
    ar = torch.arange(V, device=logits_rows.device)
    for i in range(0, logits_rows.size(0), chunk):
        pr = torch.softmax(logits_rows[i:i + chunk].float(), dim=-1)
        y = targets[i:i + chunk]
        p_y = pr.gather(-1, y[:, None]).squeeze(-1)
        lo = (pr * (ar[None, :] < y[:, None])).sum(-1)
        lo_all.append(lo)
        p_all.append(p_y)
    return torch.cat(lo_all), torch.cat(p_all)


def energy_score(v, e, valid):
    """Energy score of a 2-sample estimate against one observed block (beta = 1).

    v: (N, 2, T, d) two samples, e: (N, T, d) the observed block, valid: (N, T).
    ES = 1/2 (|v1 - e| + |v2 - e|) - 1/2 |v1 - v2|, distances over the whole block, scaled by
    1/sqrt(T) so the number does not grow with the block length. Strictly proper: its
    expectation over the data is minimised only by the true conditional distribution.
    """
    m = valid[:, None, :, None].to(v.dtype)
    v = v * m
    e = (e * valid[:, :, None].to(e.dtype))[:, None]
    scale = 1.0 / math.sqrt(v.size(2))
    d1 = (v - e).flatten(2).norm(dim=-1).mean(dim=1)
    d12 = (v[:, 0] - v[:, 1]).flatten(1).norm(dim=-1)
    return (d1 - 0.5 * d12) * scale, d12 * scale


def gumbel_st(logits, tau):
    """Straight-through Gumbel-softmax: hard one-hot forward, soft gradient. logits (..., C)."""
    u = torch.rand_like(logits).clamp_min(1e-10)
    g = -torch.log((-torch.log(u)).clamp_min(1e-10))
    soft = torch.softmax((logits + g) / tau, dim=-1)
    idx = soft.argmax(-1)
    hard = F.one_hot(idx, logits.size(-1)).to(soft.dtype)
    return hard - soft.detach() + soft, idx


class BlockHead(nn.Module):
    """Sampling-aware head emitting T tokens per trunk pass. See the module docstring."""

    def __init__(self, config, vocab_size: int):
        super().__init__()
        self.mode = str(getattr(config, "sap_block_mode", "indep"))
        assert self.mode in SAP_MODES, f"sap_block_mode={self.mode!r} not in {SAP_MODES}"
        self.T = int(config.sap_block_T)
        assert self.T >= 1, "sap_block_T must be >= 1"
        self.d = d = int(config.n_embd)
        self.vocab_size = int(vocab_size)
        self.W = int(getattr(config, "sap_ctx_window", 16))
        self.frac = float(getattr(config, "sap_block_frac", 0.125))
        n_head = int(getattr(config, "sap_head_heads", 0)) or max(1, d // 64)
        mult = int(getattr(config, "sap_head_mlp_mult", 2))
        L = int(getattr(config, "sap_head_layers", 2))
        self.chunk = int(getattr(config, "sap_logit_chunk", 8192))
        self.G = int(getattr(config, "sap_latent_groups", 4))
        self.C = int(getattr(config, "sap_latent_codes", 16))
        self.dz = int(getattr(config, "sap_latent_dim", 64))
        self.free_bits = float(getattr(config, "sap_free_bits", 0.25))
        self.kl_anneal = int(getattr(config, "sap_kl_anneal_steps", 2000))
        self.tau = float(getattr(config, "sap_gumbel_tau", 1.0))
        self.R = int(getattr(config, "sap_cp_components", 8))
        self.k_wta = int(getattr(config, "sap_wta_k", 4))
        self.n_freq = int(getattr(config, "sap_u_freqs", 24))
        self.u_interior = float(getattr(config, "sap_u_interior", 0.8))
        # Training forward calls, for the KL warm-up. A tensor buffer rather than a Python int
        # so torch.compile sees an in-place update instead of a new constant every step.
        self.register_buffer("_calls", torch.zeros((), dtype=torch.float32), persistent=False)

        self.h_proj = _Lin(d, d)
        self.slot_emb = nn.Parameter(torch.empty(self.T, d))
        self.ctx_pos = nn.Parameter(torch.empty(self.W, d)) if self.W > 0 else None
        self.dec = nn.ModuleList([_Layer(d, n_head, mult) for _ in range(L)])

        m = self.mode
        if m in LATENT_MODES:
            self.enc_pos = nn.Parameter(torch.empty(self.T, d))
            self.enc_cls = _Lin(d, d)
            n_enc = int(getattr(config, "sap_enc_layers", 1))
            self.enc = nn.ModuleList([_Layer(d, n_head, mult) for _ in range(n_enc)])
        if m == "p1_discrete":
            G, C = self.G, self.C
            self.q_out = _Lin(d, G * C)
            self.prior_in = _Lin(d, d)
            self.prior_code = nn.Parameter(torch.empty(G * C, d))   # embeds earlier groups' codes
            self.prior_grp = nn.Parameter(torch.empty(G, d))        # which group is being predicted
            self.prior_mlp = _MLP(d, mult)
            self.prior_out = _Lin(d, C)
            self.dec_code = nn.Parameter(torch.empty(G * C, d))     # what the decoder sees of z
        if m == "p2_gauss":
            self.q_out = _Lin(d, 2 * self.dz)
            self.prior_mlp = _MLP(d, mult)
            self.prior_out = _Lin(d, 2 * self.dz)
            self.z_in = _Lin(self.dz, d)
        if m in NOISE_MODES:
            self.z_in = _Lin(self.dz, d)
        if m == "cp":
            self.comp_emb = nn.Parameter(torch.empty(self.R, d))
            self.mix_out = _Lin(d, self.R)
        if m == "local":
            self.tok_in = _Lin(d, d)
        if m == "inv_head":
            self.u_in = _Lin(2 * self.n_freq + 1, d)

    # ------------------------------------------------------------------ init
    @torch.no_grad()
    def init_weights(self):
        """Explicit init for every tensor (GPT.init_weights NaN-poisons first)."""
        d = self.d
        self._calls.zero_()
        s = 3 ** 0.5 * d ** -0.5
        for mod in self.modules():
            if isinstance(mod, _Lin):
                torch.nn.init.uniform_(mod.weight, -s, s)
        # Residual writers start at zero, so the decoder begins as the identity on its inputs.
        for layer in list(self.dec) + list(getattr(self, "enc", [])):
            torch.nn.init.zeros_(layer.attn.c_proj.weight)
            torch.nn.init.zeros_(layer.mlp.c_proj.weight)
        for p in (self.slot_emb, self.ctx_pos, getattr(self, "enc_pos", None),
                  getattr(self, "prior_code", None), getattr(self, "prior_grp", None),
                  getattr(self, "dec_code", None), getattr(self, "comp_emb", None)):
            if p is not None:
                torch.nn.init.normal_(p, mean=0.0, std=1.0)
        # Output layers of the latent nets start small so the first KL terms are near zero.
        for name in ("q_out", "prior_out", "mix_out"):
            if hasattr(self, name):
                torch.nn.init.normal_(getattr(self, name).weight, std=0.001)
        if hasattr(self, "prior_mlp"):
            torch.nn.init.zeros_(self.prior_mlp.c_proj.weight)

    def embedding_parameters(self):
        """Lookup-style tables; the rest are matrices for Muon."""
        names = ("slot_emb", "ctx_pos", "enc_pos", "prior_code", "prior_grp", "dec_code", "comp_emb")
        return [getattr(self, n) for n in names if getattr(self, n, None) is not None]

    # ------------------------------------------------------------- pieces
    def _ctx(self, ctx):
        if ctx is None or self.W == 0:
            return None
        return _norm(ctx + self.ctx_pos[-ctx.size(1):].to(ctx.dtype))

    def _mask(self, N, T, ctx_valid, causal, device):
        slots = torch.ones(T, T, dtype=torch.bool, device=device)
        if causal:
            slots = torch.tril(slots)
        slots = slots[None].expand(N, T, T)
        if ctx_valid is None:
            return slots
        return torch.cat([ctx_valid[:, None, :].expand(N, T, ctx_valid.size(1)), slots], dim=-1)

    def _decode(self, slot_in, ctx_in, ctx_valid):
        N, T, _ = slot_in.shape
        if ctx_in is not None and ctx_in.size(0) != N:  # several samples per context
            rep = N // ctx_in.size(0)
            ctx_in = ctx_in.repeat_interleave(rep, dim=0)
            ctx_valid = ctx_valid.repeat_interleave(rep, dim=0)
        mask = self._mask(N, T, None if ctx_in is None else ctx_valid,
                          self.mode in CAUSAL_MODES, slot_in.device)
        s = slot_in
        for layer in self.dec:
            s = layer(s, ctx_in, mask)
        return s

    def _base(self, h):
        return (self.h_proj(h)[:, None, :] + self.slot_emb[None].to(h.dtype))  # (N, T, d)

    def _encode(self, h, y_safe, embed):
        """Posterior network over the true block. Returns the pooled (N, d) summary."""
        e = embed(y_safe).to(h.dtype) + self.enc_pos[None].to(h.dtype)
        x = torch.cat([self.enc_cls(h)[:, None, :], e], dim=1)  # (N, T+1, d)
        N, S, _ = x.shape
        mask = torch.ones(N, S, S, dtype=torch.bool, device=x.device)
        for layer in self.enc:
            x = layer(x, None, mask)
        return _norm(x[:, 0])

    def _prior_logits_p1(self, h, z_onehot):
        """AR prior over code groups, teacher-forced on z_onehot (N, G, C). Returns (N, G, C)."""
        N, G, C = z_onehot.shape
        # Embedding of every earlier group's chosen code: exclusive cumulative sum over groups.
        per_group = torch.einsum("ngc,gcd->ngd", z_onehot.to(h.dtype),
                                 self.prior_code.view(G, C, self.d).to(h.dtype))
        prev = torch.cumsum(per_group, dim=1) - per_group
        x = self.prior_in(h)[:, None, :] + prev + self.prior_grp[None].to(h.dtype)
        x = _norm(x)
        x = x + self.prior_mlp(x)
        return self.prior_out(_norm(x)).float()

    def _z_emb_p1(self, z_onehot):
        N, G, C = z_onehot.shape
        return z_onehot.reshape(N, G * C).to(self.dec_code.dtype) @ self.dec_code

    def _prior_p2(self, h):
        x = _norm(self.h_proj(h))
        x = x + self.prior_mlp(x)
        mu, logvar = self.prior_out(_norm(x)).float().chunk(2, dim=-1)
        return mu, logvar.clamp(-8.0, 4.0)

    def _slot_logprob(self, s, y_safe, readout):
        N, T, d = s.shape
        return _target_logprob(s.reshape(N * T, d), y_safe.reshape(N * T), readout,
                               self.chunk).view(N, T)

    def _beta(self):
        if self.kl_anneal <= 0:
            return 1.0
        return torch.clamp(self._calls / float(self.kl_anneal), max=1.0)

    def _interior(self, lo, width):
        """A point inside each CDF bin, kept off the edges.

        Uniform over the central `sap_u_interior` fraction of the bin. PTP draws from
        Beta(b, b) for the same purpose; the edges are where a slightly different CDF at
        decode time would flip the token, which is K-Forcing's fragility.
        """
        r = 0.5 + (torch.rand_like(lo) - 0.5) * self.u_interior
        return (lo + width * r).clamp(0.0, 1.0 - 1e-7)

    # ------------------------------------------------------------- training
    def loss(self, h, ctx, ctx_valid, y, readout, embed=None, u_bins=None):
        """Per-token training loss for one batch of blocks.

        h: (N, d) normalised trunk state at the block start; ctx: (N, W, d) window ending at
        that position (or None); ctx_valid: (N, W) bool; y: (N, T) targets, -1 = ignore;
        readout: (rows, d) -> (rows, V) softcapped logits; embed: ids -> input embeddings;
        u_bins: (lo, width) of the data tokens' CDF bins, only for inv_head.
        Returns (scalar loss, dict of detached diagnostics).
        """
        if self.training:
            self._calls.add_(1.0)
        valid = y >= 0
        y_safe = y.clamp_min(0)
        nvalid = valid.sum().clamp_min(1)
        ctx_in = self._ctx(ctx)
        base = self._base(h)
        N, T = y.shape
        stats = {}
        m = self.mode

        if m == "indep":
            lp = self._slot_logprob(self._decode(base, ctx_in, ctx_valid), y_safe, readout)
            loss = -(lp * valid).sum() / nvalid
        elif m == "local":
            prev = torch.cat([y_safe.new_zeros(N, 1), y_safe[:, :-1]], dim=1)
            tok = self.tok_in(_norm(embed(prev).to(h.dtype)))
            tok = tok * (torch.arange(T, device=h.device) > 0)[None, :, None].to(tok.dtype)
            lp = self._slot_logprob(self._decode(base + tok, ctx_in, ctx_valid), y_safe, readout)
            loss = -(lp * valid).sum() / nvalid
        elif m == "inv_head":
            u = self._interior(*u_bins)                                  # (N, T-1)
            lp = self._slot_logprob(self._decode(base + self._u_emb(u, h.dtype), ctx_in, ctx_valid),
                                    y_safe, readout)
            loss = -(lp * valid).sum() / nvalid
        elif m == "p1_discrete":
            q_logits = self.q_out(self._encode(h, y_safe, embed)).float().view(N, self.G, self.C)
            z_st, _ = gumbel_st(q_logits, self.tau)
            p_logits = self._prior_logits_p1(h, z_st)
            logq = torch.log_softmax(q_logits, -1)
            kl = (logq.exp() * (logq - torch.log_softmax(p_logits, -1))).sum(-1)   # (N, G)
            zemb = self._z_emb_p1(z_st).to(h.dtype)
            lp = self._slot_logprob(self._decode(base + zemb[:, None], ctx_in, ctx_valid), y_safe, readout)
            rec = -(lp * valid).sum()
            kl_term = torch.clamp(kl.mean(0), min=self.free_bits).sum() * N
            loss = (rec + self._beta() * kl_term) / nvalid
            stats["kl_per_group"] = kl.mean(0).detach()
            stats["rec_nats_per_token"] = (rec / nvalid).detach()
        elif m == "p2_gauss":
            mu_q, logvar_q = self.q_out(self._encode(h, y_safe, embed)).float().chunk(2, dim=-1)
            logvar_q = logvar_q.clamp(-8.0, 4.0)
            mu_p, logvar_p = self._prior_p2(h)
            z = mu_q + torch.randn_like(mu_q) * (0.5 * logvar_q).exp()
            kl = 0.5 * (logvar_p - logvar_q + (logvar_q.exp() + (mu_q - mu_p) ** 2) / logvar_p.exp() - 1.0)
            zemb = self.z_in(z.to(h.dtype))
            lp = self._slot_logprob(self._decode(base + zemb[:, None], ctx_in, ctx_valid), y_safe, readout)
            rec = -(lp * valid).sum()
            kl_tot = kl.sum(-1).mean()
            kl_term = torch.clamp(kl_tot, min=self.free_bits * self.G) * N
            loss = (rec + self._beta() * kl_term) / nvalid
            stats["kl_per_block"] = kl_tot.detach()
            stats["rec_nats_per_token"] = (rec / nvalid).detach()
        elif m == "p3_energy":
            eps = torch.randn(N, 2, self.dz, device=h.device, dtype=h.dtype)
            zemb = self.z_in(eps).reshape(N * 2, 1, self.d)
            s = self._decode(base.repeat_interleave(2, dim=0) + zemb, ctx_in, ctx_valid)
            v = _norm(s).float().view(N, 2, T, self.d)
            e = _norm(embed(y_safe)).detach().float()
            es, spread = energy_score(v, e, valid)
            loss = es.mean()
            stats["energy_score"] = es.mean().detach()
            stats["sample_spread"] = spread.mean().detach()
        elif m == "cp":
            log_pi = torch.log_softmax(self.mix_out(h).float(), dim=-1)                 # (N, R)
            slot_in = base[:, None] + self.comp_emb[None, :, None].to(h.dtype)          # (N, R, T, d)
            s = self._decode(slot_in.reshape(N * self.R, T, self.d), ctx_in, ctx_valid)
            lp = self._slot_logprob(s, y_safe.repeat_interleave(self.R, dim=0), readout).view(N, self.R, T)
            comp = (lp * valid[:, None]).sum(-1)
            block_ll = torch.logsumexp(log_pi + comp, dim=-1)
            loss = -block_ll.sum() / nvalid
        elif m in ("plain_noise", "wta"):
            k = 1 if m == "plain_noise" else self.k_wta
            eps = torch.randn(N, k, self.dz, device=h.device, dtype=h.dtype)
            zemb = self.z_in(eps).reshape(N * k, 1, self.d)
            s = self._decode(base.repeat_interleave(k, dim=0) + zemb, ctx_in, ctx_valid)
            lp = self._slot_logprob(s, y_safe.repeat_interleave(k, dim=0), readout).view(N, k, T)
            nll = -(lp * valid[:, None]).sum(-1)                                        # (N, k)
            loss = nll.min(dim=1).values.sum() / nvalid
        else:  # pragma: no cover - guarded in __init__
            raise ValueError(m)
        stats["block_loss"] = loss.detach()
        return loss, stats

    def _u_emb(self, u, dtype):
        """Noise embedding placed on slot k for u_{k-1}; slot 0 gets none. u: (N, T-1)."""
        N = u.size(0)
        e = self.u_in(u_features(u, self.n_freq).to(dtype))                            # (N, T-1, d)
        return torch.cat([e.new_zeros(N, 1, self.d), e], dim=1)

    # ------------------------------------------------------------- sampling
    @torch.no_grad()
    def sample(self, h, ctx, ctx_valid, readout, embed=None, embed_table=None,
               temperature=1.0, generator=None, posterior_y=None):
        """Draw one block of T tokens per row of h. Returns (N, T) long.

        temperature scales the slot distributions; 0 takes the argmax. The plan latent comes
        from the prior, since that is where the block's randomness lives at decode time.
        posterior_y (diagnostic only, p1/p2): draw the plan from the recognition model given
        that true block instead. If posterior samples are clean and prior samples are not, the
        gap is the prior's, not the decoder's.
        """
        N, T, dev = h.size(0), self.T, h.device
        ctx_in = self._ctx(ctx)
        base = self._base(h)
        m = self.mode

        def _rand(*shape):
            return torch.rand(*shape, device=dev, generator=generator)

        def _randn(*shape):
            return torch.randn(*shape, device=dev, generator=generator)

        def _choose(logits):  # (N, T, V) -> (N, T)
            if temperature <= 0:
                return logits.argmax(-1)
            probs = torch.softmax(logits / temperature, dim=-1)
            return pick(probs, _rand(*probs.shape[:-1]))

        def _logits(s):
            return readout(s.reshape(-1, self.d)).view(s.size(0), s.size(1), -1)

        if m == "indep":
            return _choose(_logits(self._decode(base, ctx_in, ctx_valid)))
        if m == "local":
            out = torch.zeros(N, T, dtype=torch.long, device=dev)
            for k in range(T):
                prev = torch.cat([out.new_zeros(N, 1), out[:, :-1]], dim=1)
                tok = self.tok_in(_norm(embed(prev).to(h.dtype)))
                tok = tok * (torch.arange(T, device=dev) > 0)[None, :, None].to(tok.dtype)
                s = self._decode(base + tok, ctx_in, ctx_valid)
                lg_k = _logits(s[:, k:k + 1])
                out[:, k] = _choose(lg_k)[:, 0]
            return out
        if m == "inv_head":
            u = _rand(N, T)
            lg = _logits(self._decode(base + self._u_emb(u[:, :-1], h.dtype), ctx_in, ctx_valid))
            if temperature <= 0:
                return lg.argmax(-1)
            return pick(torch.softmax(lg / temperature, dim=-1), u)
        if m == "p1_discrete":
            if posterior_y is not None:
                q_logits = self.q_out(self._encode(h, posterior_y.clamp_min(0), embed)).float()
                q_probs = torch.softmax(q_logits.view(N, self.G, self.C), dim=-1)
                z = F.one_hot(pick(q_probs, _rand(N, self.G)), self.C).float()
            else:
                z = torch.zeros(N, self.G, self.C, device=dev)
                for g in range(self.G):
                    p = torch.softmax(self._prior_logits_p1(h, z)[:, g], dim=-1)
                    z[:, g] = F.one_hot(pick(p, _rand(N)), self.C).to(z.dtype)
            zemb = self._z_emb_p1(z).to(h.dtype)
            return _choose(_logits(self._decode(base + zemb[:, None], ctx_in, ctx_valid)))
        if m == "p2_gauss":
            if posterior_y is not None:
                mu, logvar = self.q_out(self._encode(h, posterior_y.clamp_min(0), embed)).float().chunk(2, dim=-1)
                logvar = logvar.clamp(-8.0, 4.0)
            else:
                mu, logvar = self._prior_p2(h)
            z = mu + _randn(*mu.shape) * (0.5 * logvar).exp()
            zemb = self.z_in(z.to(h.dtype))
            return _choose(_logits(self._decode(base + zemb[:, None], ctx_in, ctx_valid)))
        if m == "cp":
            pi = torch.softmax(self.mix_out(h).float(), dim=-1)
            r = pick(pi, _rand(N))
            s = self._decode(base + self.comp_emb[r][:, None].to(h.dtype), ctx_in, ctx_valid)
            return _choose(_logits(s))
        if m in ("plain_noise", "wta"):
            zemb = self.z_in(_randn(N, self.dz).to(h.dtype))
            return _choose(_logits(self._decode(base + zemb[:, None], ctx_in, ctx_valid)))
        if m == "p3_energy":
            zemb = self.z_in(_randn(N, self.dz).to(h.dtype))
            v = _norm(self._decode(base + zemb[:, None], ctx_in, ctx_valid)).float()   # (N, T, d)
            table = _norm(embed_table.float())                                        # (V, d)
            return (v @ table.t())[..., :self.vocab_size].argmax(-1)
        raise ValueError(m)

    # ------------------------------------------------------------- likelihood
    @torch.no_grad()
    def block_logprob(self, h, ctx, ctx_valid, y, readout, embed=None, u_bins=None, n_samples=16):
        """log p(block) per row: exact where the mode has a likelihood, else a bound.

        indep / local / cp: exact. p1 / p2: importance-weighted bound with the posterior as
        proposal. inv_head: the sampled-noise bound averaged over draws (an upper bound on
        NLL). plain_noise / wta: Monte Carlo over prior noise. p3_energy: None.
        """
        valid = y >= 0
        y_safe = y.clamp_min(0)
        N, T = y.shape
        ctx_in = self._ctx(ctx)
        base = self._base(h)
        m = self.mode

        def _ll(s, yy):
            return (self._slot_logprob(s, yy, readout) * valid.repeat_interleave(
                s.size(0) // N, dim=0)).sum(-1)

        if m == "indep":
            return _ll(self._decode(base, ctx_in, ctx_valid), y_safe)
        if m == "local":
            prev = torch.cat([y_safe.new_zeros(N, 1), y_safe[:, :-1]], dim=1)
            tok = self.tok_in(_norm(embed(prev).to(h.dtype)))
            tok = tok * (torch.arange(T, device=h.device) > 0)[None, :, None].to(tok.dtype)
            return _ll(self._decode(base + tok, ctx_in, ctx_valid), y_safe)
        if m == "cp":
            log_pi = torch.log_softmax(self.mix_out(h).float(), dim=-1)
            slot_in = base[:, None] + self.comp_emb[None, :, None].to(h.dtype)
            s = self._decode(slot_in.reshape(N * self.R, T, self.d), ctx_in, ctx_valid)
            comp = _ll(s, y_safe.repeat_interleave(self.R, dim=0)).view(N, self.R)
            return torch.logsumexp(log_pi + comp, dim=-1)
        if m == "inv_head":
            acc = torch.zeros(N, device=h.device)
            for _ in range(n_samples):
                u = self._interior(*u_bins)
                acc += _ll(self._decode(base + self._u_emb(u, h.dtype), ctx_in, ctx_valid), y_safe)
            return acc / n_samples
        if m == "p1_discrete":
            q_logits = self.q_out(self._encode(h, y_safe, embed)).float().view(N, self.G, self.C)
            logq_all = torch.log_softmax(q_logits, -1)
            terms = []
            for _ in range(n_samples):
                idx = torch.distributions.Categorical(logits=q_logits).sample()        # (N, G)
                z = F.one_hot(idx, self.C).float()
                logp = torch.log_softmax(self._prior_logits_p1(h, z), -1).gather(-1, idx[..., None]).squeeze(-1).sum(-1)
                logq = logq_all.gather(-1, idx[..., None]).squeeze(-1).sum(-1)
                zemb = self._z_emb_p1(z).to(h.dtype)
                ll = _ll(self._decode(base + zemb[:, None], ctx_in, ctx_valid), y_safe)
                terms.append(ll + logp - logq)
            return torch.logsumexp(torch.stack(terms), dim=0) - math.log(n_samples)
        if m == "p2_gauss":
            mu_q, logvar_q = self.q_out(self._encode(h, y_safe, embed)).float().chunk(2, dim=-1)
            logvar_q = logvar_q.clamp(-8.0, 4.0)
            mu_p, logvar_p = self._prior_p2(h)

            def _lognormal(z, mu, logvar):
                return (-0.5 * (math.log(2 * math.pi) + logvar + (z - mu) ** 2 / logvar.exp())).sum(-1)

            terms = []
            for _ in range(n_samples):
                z = mu_q + torch.randn_like(mu_q) * (0.5 * logvar_q).exp()
                zemb = self.z_in(z.to(h.dtype))
                ll = _ll(self._decode(base + zemb[:, None], ctx_in, ctx_valid), y_safe)
                terms.append(ll + _lognormal(z, mu_p, logvar_p) - _lognormal(z, mu_q, logvar_q))
            return torch.logsumexp(torch.stack(terms), dim=0) - math.log(n_samples)
        if m in ("plain_noise", "wta"):
            terms = []
            for _ in range(n_samples):
                zemb = self.z_in(torch.randn(N, self.dz, device=h.device, dtype=h.dtype))
                terms.append(_ll(self._decode(base + zemb[:, None], ctx_in, ctx_valid), y_safe))
            return torch.logsumexp(torch.stack(terms), dim=0) - math.log(n_samples)
        return None  # p3_energy: likelihood-free

    @torch.no_grad()
    def latent_sensitivity(self, h, ctx, ctx_valid, readout, n_samples=8):
        """How much the slot distributions move with the plan/noise draw, in nats.

        Summed over slots: H(mean_z p) - mean_z H(p). Zero when the head ignores its latent,
        which is what plain_noise is expected to show. None for modes without a latent.
        """
        m = self.mode
        if m not in ("p1_discrete", "p2_gauss", "plain_noise", "wta", "cp"):
            return None
        N = h.size(0)
        ctx_in = self._ctx(ctx)
        base = self._base(h)
        probs = []
        for _ in range(n_samples):
            if m == "p1_discrete":
                z = torch.zeros(N, self.G, self.C, device=h.device)
                for g in range(self.G):
                    p = torch.softmax(self._prior_logits_p1(h, z)[:, g], dim=-1)
                    z[:, g] = F.one_hot(pick(p, torch.rand(N, device=h.device)), self.C).float()
                zemb = self._z_emb_p1(z).to(h.dtype)[:, None]
            elif m == "p2_gauss":
                mu, logvar = self._prior_p2(h)
                zemb = self.z_in((mu + torch.randn_like(mu) * (0.5 * logvar).exp()).to(h.dtype))[:, None]
            elif m == "cp":
                r = pick(torch.softmax(self.mix_out(h).float(), -1), torch.rand(N, device=h.device))
                zemb = self.comp_emb[r][:, None].to(h.dtype)
            else:
                zemb = self.z_in(torch.randn(N, self.dz, device=h.device, dtype=h.dtype))[:, None]
            s = self._decode(base + zemb, ctx_in, ctx_valid)
            probs.append(torch.softmax(readout(s.reshape(-1, self.d)), -1).view(N, self.T, -1))
        P = torch.stack(probs)                                                        # (S, N, T, V)

        def _H(p):
            return -(p * p.clamp_min(1e-12).log()).sum(-1)

        return (_H(P.mean(0)) - _H(P).mean(0)).sum(-1)                                # (N,)

    # ------------------------------------------------------------- cost
    def flops_per_token(self, vocab_size: int | None = None, train: bool = True) -> int:
        """Training FLOPs (forward + backward, 6 per MAC) the head adds per TRUNK token.

        Counts the slot rows' matmuls through the decoder, the per-block key/value
        projections of the context window, attention scores, the readout through the shared
        unembedding (vocab_size x d per slot row), and the mode-specific encoder, prior and
        mixture work. Everything is scaled by sap_block_frac, the fraction of positions that
        carry a block in training. With train=False it returns per-BLOCK forward FLOPs.
        """
        V = vocab_size or self.vocab_size
        d, T, W = self.d, self.T, self.W
        mult = self.dec[0].mlp.c_fc.weight.shape[0] // d if len(self.dec) else 0
        L = len(self.dec)
        per_layer = (T * (2 + 2 * mult) * d * d          # q, o and the MLP for T slot rows
                     + 2 * (T + W) * d * d               # k and v over context window and slots
                     + 2 * T * (T + W) * d)              # scores and weighted values
        dec = L * per_layer
        readout = T * V * d
        extra = d * d                                    # h_proj
        m = self.mode
        if m == "cp":
            dec, readout = dec * self.R, readout * self.R
            extra += self.R * d
        elif m == "wta":
            dec, readout = dec * self.k_wta, readout * self.k_wta
        elif m == "p3_energy":
            dec, readout = dec * 2, 0                    # two samples, no V-wide readout
        if m in ("plain_noise", "wta", "p2_gauss", "p3_energy"):
            extra += self.dz * d
        if m in LATENT_MODES:
            S = T + 1
            mlp_e = self.enc[0].mlp.c_fc.weight.shape[0] // d if len(self.enc) else 0
            extra += len(self.enc) * (S * (4 + 2 * mlp_e) * d * d + 2 * S * S * d) + d * d
            if m == "p1_discrete":
                extra += d * self.G * self.C + self.G * (d * d + 2 * mult * d * d + d * self.C)
                extra += self.G * self.C * d
            else:
                extra += d * 2 * self.dz + (2 * mult * d * d) + d * 2 * self.dz
        if m == "local":
            extra += T * d * d
        if m == "inv_head":
            extra += T * (2 * self.n_freq + 1) * d
        per_block = dec + readout + extra
        if not train:
            return int(2 * per_block)
        return int(6 * per_block * self.frac)


@torch.no_grad()
def evaluate_block_bpb(model, batches, steps, token_bytes, blocks_per_row=8, n_samples=16):
    """Block bpb of the SAP head next to the trunk's next-token bpb on the SAME tokens.

    For each validation row, `blocks_per_row` evenly spaced block starts are scored. The
    block number is exact (indep, local, cp) or an upper bound on the NLL (p1, p2 via an
    importance-weighted bound; inv_head via the sampled-noise bound; plain_noise and wta via
    Monte Carlo over prior noise). The next-token number scores the same T tokens with the
    trunk's own autoregressive factorisation, so the ratio is the price of emitting the block
    in one pass. Blocks containing a token with no bytes (special tokens) or an ignored
    target are skipped whole, since a joint bound cannot drop one slot.

    Returns a dict; block_bpb is None for the likelihood-free p3_energy head.
    """
    import torch.distributed as dist
    head = model.sap_head
    device = model.get_device()
    T = head.T
    ks = torch.arange(T, device=device)
    block_nats = torch.zeros((), dtype=torch.float64, device=device)
    ntp_nats = torch.zeros((), dtype=torch.float64, device=device)
    n_bytes = torch.zeros((), dtype=torch.float64, device=device)
    n_blocks = torch.zeros((), dtype=torch.float64, device=device)
    sens_sum = torch.zeros((), dtype=torch.float64, device=device)
    sens_n = torch.zeros((), dtype=torch.float64, device=device)
    likelihood_free = head.mode == "p3_energy"
    it = iter(batches)
    for step in range(steps):
        x, y = next(it)
        hid = model(x, skip_logits=True)
        B, Tq = x.shape
        n_start = Tq - T + 1
        eff_blocks = max(1, min(blocks_per_row, n_start))
        if n_start <= 1:
            starts = torch.zeros(1, dtype=torch.long, device=device)
        else:
            starts = torch.tensor([(j + 1) * n_start // (eff_blocks + 1) for j in range(eff_blocks)],
                                  device=device)
        b = torch.arange(B, device=device).repeat_interleave(eff_blocks)
        t = starts.repeat(B)
        yb = y[b[:, None], t[:, None] + ks[None, :]]
        ysafe = yb.clamp_min(0)
        bytes_b = token_bytes[ysafe] * (yb >= 0)
        keep = (bytes_b > 0).all(dim=1)
        ctx, cv = model._sap_window(hid, b, t)

        # Compute NTP logits in chunks for selected block positions
        # Avoids allocating massive tensors: OOM-safe for arbitrary T up to L
        hid_blocks = hid[b[:, None], t[:, None] + ks[None, :]]
        hid_flat = hid_blocks.reshape(-1, hid_blocks.size(-1))
        ysafe_flat = ysafe.reshape(-1)
        ntp_lp_list = []
        u_bins = None
        inv_lo, inv_w = [], []
        for c in range(0, hid_flat.size(0), 4096):
            c_hid = hid_flat[c:c + 4096]
            c_y = ysafe_flat[c:c + 4096]
            c_lg = model._sap_readout(c_hid)
            c_lp = torch.log_softmax(c_lg.float(), dim=-1).gather(-1, c_y[:, None]).squeeze(-1)
            ntp_lp_list.append(c_lp)
            if head.mode == "inv_head":
                l_i, w_i = target_bins(c_lg, c_y)
                inv_lo.append(l_i)
                inv_w.append(w_i)
        if head.mode == "inv_head" and inv_lo:
            u_bins = (torch.cat(inv_lo).view(-1, T)[:, :T - 1], torch.cat(inv_w).view(-1, T)[:, :T - 1])
        ntp_lp = torch.cat(ntp_lp_list).view(yb.shape).sum(-1)
        ntp_nats += -(ntp_lp[keep]).double().sum()
        n_bytes += bytes_b[keep].double().sum()
        n_blocks += keep.double().sum()
        if not likelihood_free:
            lp = head.block_logprob(hid[b, t], ctx, cv, yb, model._sap_readout,
                                    embed=model.transformer.wte, u_bins=u_bins, n_samples=n_samples)
            block_nats += -(lp[keep]).double().sum()
        if step == 0:
            s = head.latent_sensitivity(hid[b, t], ctx, cv, model._sap_readout)
            if s is not None:
                sens_sum += s.double().sum()
                sens_n += s.numel()
    if dist.is_initialized() and dist.get_world_size() > 1:
        for v in (block_nats, ntp_nats, n_bytes, n_blocks, sens_sum, sens_n):
            dist.all_reduce(v, op=dist.ReduceOp.SUM)
    nb = max(n_bytes.item(), 1.0)
    out = {
        "ntp_bpb_same_tokens": ntp_nats.item() / (math.log(2) * nb),
        "block_bpb": None if likelihood_free else block_nats.item() / (math.log(2) * nb),
        "blocks": int(n_blocks.item()),
        "latent_sensitivity_nats": (sens_sum.item() / sens_n.item()) if sens_n.item() > 0 else None,
    }
    if out["block_bpb"] is not None and out["ntp_bpb_same_tokens"] > 0:
        out["block_over_ntp"] = out["block_bpb"] / out["ntp_bpb_same_tokens"]
    return out
