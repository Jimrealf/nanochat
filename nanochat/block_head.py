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
  field_cp     one categorical plan selects a context-gated low-rank positional field;
               exact mixture likelihood, then every token is sampled in parallel.
  field_energy one Gaussian draw creates a correlated low-rank field over all slots.  A
               Monte-Carlo marginal likelihood is combined with an energy score over
               realised hard token blocks (sparse soft straight-through gradient).
  local        B4  local autoregressive head: slot k reads the tokens of slots < k.
                   Exact likelihood; sequential inside the head when decoding.
  inv_head     B5  PTP-style competitor restricted to the head: slot k reads uniform
                   noise u_<k that, in training, is inverted from the data tokens through
                   the next-token distribution's CDF.
  plain_noise  control: prior noise with plain CE. Expected to learn to ignore the noise.
  wta          control: winner-take-all over k noise draws (Condor's symmetry breaking).

S01 adds a second family whose stochastic variable is a sampled token field rather than one
global plan.  Every ``sir*`` mode makes a draft inside the head, then lets later layers read
that realised draft before the final tokens are sampled:

  sir          hard on-policy draft + leave-one-out reference layer.
  sir_soft     the hard draft has a sparse top-k straight-through gradient.
  sir_compat   learned multi-scale low-rank compatibility messages.
  sir_context  contextual soft-error loss over the draft's candidates.
  sir_conf     a fixed-budget second draft updates the least-confident positions.
  sir_anchor   sparse sampled anchors followed by parallel fill.
  sir_pyramid  coarse anchors, fine anchors, then parallel fill.
  sir_tree     observed anchors factorised as a balanced tree: levels sample in parallel,
               every level conditions on committed ancestors, then unanchored slots fill.
  sir_lattice  a top-k candidate lattice with learned pairwise messages.
  sir_energy   a structured-negative reference energy that also writes into the decoder.
  sir_full     soft + compatibility + context + confidence + energy.

SAP v4 (sap_research_plan.md) keeps the stochastic cut but draws it from an exact joint, so a
block costs one head pass plus a chain sampler that runs no network layers (nanochat/sap_chain.py):

  lat_crf      every slot on a top-K candidate lattice (plus an escape state), linear-chain CRF
               with low-rank pair potentials. Exact likelihood; forward-filter backward-sample.
  lat_tt       the same lattice under an HMM / tensor train with R latent states.
  lat_cp       the same lattice under a mixture of Z global codes: sample the code, then every
               slot in parallel (strictly one-shot).
  cut_crf      the sampling-cut head: even slots are anchors drawn jointly from an exact CRF,
               then one reference layer fills the odd slots in parallel from both neighbours.
  cut_tt       the same with an HMM over the anchors.
  pmi_chain    zero-parameter coherence floor: slot k's logits plus a learned gate times the
               corpus PMI row of slot k-1's token (nanochat/sap_tables.py).
  corpus_code  corpus-defined token classes as an exact latent: an exact chain over the class
               sequence, then tokens in parallel inside their classes.
  p1_selfpost  P1 whose posterior reads the trunk's own (detached) state at the block's end.

Every v4 head reads slot 1 out of the trunk state h itself, so its first token is exactly the
next-token distribution. `sap_nce_props > 0` adds self-contrastive resampling on top of the
lattice and cut heads: a scorer trained by InfoNCE against the head's own proposals reweights L
proposals at decode time.

The token-aligned reference arms mask the matching draft position, so output slot i cannot simply
copy draft token i.  The hierarchical arms are different: their sparse anchors are committed
sampled decisions, and fill positions may read those anchors (including a matching committed
anchor).  Draft vocabulary projections are chunked and retain only top-k candidates; no ``B*T*V``
tensor survives a chunk.

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

from nanochat.sap_chain import chain_logprob, chain_sample, hmm_loglik, hmm_sample
from nanochat.sap_tables import seen as _seen

SIR_MODES = ("sir", "sir_soft", "sir_compat", "sir_context", "sir_conf", "sir_anchor",
             "sir_pyramid", "sir_tree", "sir_lattice", "sir_energy", "sir_full")
LATTICE_MODES = ("lat_crf", "lat_tt", "lat_cp")
CUT_MODES = ("cut_crf", "cut_tt")
STRUCT_MODES = LATTICE_MODES + CUT_MODES
CRF_MODES = ("lat_crf", "cut_crf")
TT_MODES = ("lat_tt", "cut_tt")
V4_MODES = (*STRUCT_MODES, "pmi_chain", "corpus_code", "p1_selfpost")
# The block's tokens pass through the top sap_depth_layers trunk layers (GPT._sap_depth_logprob):
# trunk depth per slot instead of a small head. depth_local is the exact chain rule (T-1
# sequential slot passes per block); depth_tree factorises the block by bisection rounds
# (about log2 T sequential passes, mask slots for unknown left neighbours); depth_roll is
# depth_local whose slot k also enters with slot k-1's top-layer state.
DEPTH_MODES = ("depth_local", "depth_tree", "depth_roll")


def depth_tree_layout(T):
    """Rounds of depth_tree for a block of T tokens at offsets 1..T (offset o holds y_o).

    y_1 comes from the next-token head. The remaining offsets 2..T are queried by bisection
    (the interval's midpoint each round, as in sir_tree). A round runs a slot for every
    committed token at offsets <= T-1 plus a mask slot at j-1 for each queried j whose left
    neighbour is not yet committed; y_j is read off the slot at offset j-1, the next-token
    convention of the trunk. Returns [(slot offsets, slot holds a token?, [(slot index, j)])].
    """
    committed, rounds = {1}, []
    intervals = [(2, T)] if T >= 2 else []
    while intervals:
        queries, nxt = [], []
        for lo, hi in intervals:
            mid = (lo + hi) // 2
            queries.append(mid)
            if lo <= mid - 1:
                nxt.append((lo, mid - 1))
            if mid + 1 <= hi:
                nxt.append((mid + 1, hi))
        slots = sorted({o for o in committed if o <= T - 1} | {j - 1 for j in queries if j - 1 not in committed})
        read = [(slots.index(j - 1), j) for j in sorted(queries)]
        rounds.append((slots, [o in committed for o in slots], read))
        committed |= set(queries)
        intervals = nxt
    return rounds


SAP_MODES = ("indep", "p1_discrete", "p2_gauss", "p3_energy", "cp", "field_cp", "field_energy",
             "local", "local_jacobi",
             "inv_head", "plain_noise", "wta", *SIR_MODES, *V4_MODES, *DEPTH_MODES)
P1_MODES = ("p1_discrete", "p1_selfpost")
LATENT_MODES = ("p1_discrete", "p2_gauss")
NOISE_MODES = ("p3_energy", "plain_noise", "wta")
# Modes whose slots may only read earlier slots (their inputs carry earlier tokens or noise).
CAUSAL_MODES = ("local", "local_jacobi", "inv_head")


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
        if self.mode in DEPTH_MODES:
            L, self.W = 0, 0          # the trunk's own top layers do the work (GPT._sap_depth_logprob)
        self.chunk = int(getattr(config, "sap_logit_chunk", 8192))
        self.G = int(getattr(config, "sap_latent_groups", 4))
        self.C = int(getattr(config, "sap_latent_codes", 16))
        self.dz = int(getattr(config, "sap_latent_dim", 64))
        self.free_bits = float(getattr(config, "sap_free_bits", 0.25))
        self.kl_anneal = int(getattr(config, "sap_kl_anneal_steps", 2000))
        self.tau = float(getattr(config, "sap_gumbel_tau", 1.0))
        self.R = int(getattr(config, "sap_cp_components", 8))
        self.field_rank = int(getattr(config, "sap_field_rank", 8))
        self.field_samples = int(getattr(config, "sap_field_samples", 4))
        self.field_energy_weight = float(getattr(config, "sap_field_energy_weight", 0.25))
        self.field_topk = int(getattr(config, "sap_field_topk", 16))
        self.k_wta = int(getattr(config, "sap_wta_k", 4))
        self.n_freq = int(getattr(config, "sap_u_freqs", 24))
        self.u_interior = float(getattr(config, "sap_u_interior", 0.8))
        self.jacobi_sweeps = int(getattr(config, "sap_jacobi_sweeps", 2))
        self.sir_topk = int(getattr(config, "sap_sir_topk", 16))
        self.sir_rank = int(getattr(config, "sap_sir_rank", 32))
        self.sir_anchor_stride = int(getattr(config, "sap_sir_anchor_stride", 16))
        self.sir_fine_stride = int(getattr(config, "sap_sir_fine_stride", 4))
        self.sir_refine_frac = float(getattr(config, "sap_sir_refine_frac", 0.25))
        self.sir_tree_levels = int(getattr(config, "sap_sir_tree_levels", 0))
        self.sir_train_samples = int(getattr(config, "sap_sir_train_samples", 4))
        self.sir_policy_weight = float(getattr(config, "sap_sir_policy_weight", 1.0))
        self.sir_posterior_mix = float(getattr(config, "sap_sir_posterior_mix", 0.5))
        self.sir_draft_weight = float(getattr(config, "sap_sir_draft_weight", 1.0))
        self.sir_context_weight = float(getattr(config, "sap_sir_context_weight", 0.25))
        self.sir_energy_weight = float(getattr(config, "sap_sir_energy_weight", 0.25))
        # SAP v4
        self.lat_k = int(getattr(config, "sap_lattice_k", 64))
        self.pair_rank = int(getattr(config, "sap_pair_rank", 32))
        self.tt_rank = int(getattr(config, "sap_tt_rank", 32))
        self.cp_codes = int(getattr(config, "sap_cp_codes", 256))
        self.cp_chunk = 64
        self.nce_props = int(getattr(config, "sap_nce_props", 0))
        self.nce_neg = int(getattr(config, "sap_nce_neg", 4))
        self.nce_weight = float(getattr(config, "sap_nce_weight", 1.0))
        self.supp_min_count = int(getattr(config, "sap_supp_min_count", 0))
        self.supp_penalty = float(getattr(config, "sap_supp_penalty", 20.0))
        self.soft_eps = float(getattr(config, "sap_soft_eps", 0.0))
        self.code_classes = int(getattr(config, "sap_code_classes", 128))
        self.table_path = str(getattr(config, "sap_table_path", "") or "")
        # Cut heads: ordinary slot layers before the stochastic cut. The anchors are read out
        # after these, so with the default 1 an anchor three tokens ahead got one head layer.
        self.cut_pre = int(getattr(config, "sap_cut_pre_layers", 1))
        # depth_local: how many top trunk layers the block's tokens pass through, and whether
        # they use the trunk's own blocks (1, from scratch) or trainable copies (0, frozen trunk).
        self.depth_m = int(getattr(config, "sap_depth_layers", 2))
        self.depth_share = int(getattr(config, "sap_depth_share", 0))
        # Skip-middle slots: the bottom `depth_bottom` layers plus the top m - depth_bottom, the
        # skipped middle replaced by the block start's middle-layer delta (GPT._sap_depth_entry).
        self.depth_bottom = int(getattr(config, "sap_depth_bottom", 0))
        if self.mode in DEPTH_MODES:
            Ln, a = int(config.n_layer), self.depth_bottom
            assert 1 <= self.depth_m <= Ln, "sap_depth_layers must be in [1, n_layer]"
            assert 0 <= a < self.depth_m, "sap_depth_bottom must leave at least one top layer"
            assert a == 0 or self.mode != "depth_tree", "skip-middle slots are built for the chain modes"
            self.depth_layer_ids = list(range(a)) + list(range(Ln - (self.depth_m - a), Ln))
            # Trunk states the slots read: every slot layer's input, plus layer a's (the bottom of
            # the skipped middle) for the skip-middle delta.
            self.depth_capture_ids = sorted(set(self.depth_layer_ids) | ({a} if a > 0 else set()))
        if self.mode in V4_MODES:
            assert self.lat_k >= 1 and self.pair_rank >= 1 and self.tt_rank >= 1 and self.cp_codes >= 1
            if self.mode in CUT_MODES:
                assert 1 <= self.cut_pre < L, \
                    f"{self.mode}: sap_cut_pre_layers={self.cut_pre} must leave >= 1 of {L} layers for the fill"
            assert self.nce_props == 0 or self.mode in STRUCT_MODES, \
                "self-contrastive resampling (sap_nce_props) needs an exact lattice or cut head"
            assert self.supp_min_count == 0 or self.mode in CRF_MODES, \
                "the corpus support mask (sap_supp_min_count) acts on CRF pair potentials"
            assert self.soft_eps == 0.0 or self.mode in STRUCT_MODES, \
                "the n-gram soft-target auxiliary (sap_soft_eps) is wired into the lattice readout"
        if self.mode in SIR_MODES:
            assert L >= 2, f"{self.mode} needs at least two head layers (draft + reference)"
            if self.mode in ("sir_conf", "sir_pyramid", "sir_tree", "sir_full"):
                assert L >= 3, f"{self.mode} needs at least three head layers"
            assert self.sir_topk >= 1
            assert self.sir_rank >= 1
            assert self.sir_anchor_stride >= 1 and self.sir_fine_stride >= 1
            assert 0.0 < self.sir_refine_frac <= 1.0
            assert self.sir_train_samples >= 1
            assert self.sir_tree_levels >= 0
            assert 0.0 <= self.sir_posterior_mix <= 1.0
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
        if m == "field_cp":
            assert self.R >= 1 and self.field_rank >= 1
            # One sampled code selects a different low-rank field over the T slots.  The
            # context gate changes the basis amplitudes without changing the number of
            # stochastic decisions.  Unlike `cp`, the plan is not broadcast identically.
            self.field_pos = nn.Parameter(torch.empty(self.T, self.field_rank))
            self.field_code = nn.Parameter(torch.empty(self.R, self.field_rank, d))
            self.field_gate = _Lin(d, self.field_rank)
            self.mix_out = _Lin(d, self.R)
        if m == "field_energy":
            assert self.field_rank >= 1 and self.field_samples >= 2 and self.field_topk >= 1
            assert self.field_energy_weight >= 0.0
            # One draw contains q independent Gaussian vectors.  A learned positional
            # basis mixes them into a correlated T-slot field before the fixed-depth
            # decoder.  There is no token-conditioned draw or refinement step.
            self.field_pos = nn.Parameter(torch.empty(self.T, self.field_rank))
            self.field_noise_in = _Lin(self.dz, d)
            self.field_gate = _Lin(d, self.field_rank)
        if m in ("local", "local_jacobi"):
            self.tok_in = _Lin(d, d)
        if m == "inv_head":
            self.u_in = _Lin(2 * self.n_freq + 1, d)
        if m in SIR_MODES:
            r = min(self.sir_rank, d)
            self.sir_rank = r
            self.draft_in = _Lin(d, d)
            if m in ("sir_compat", "sir_context", "sir_lattice", "sir_full"):
                self.compat_down = _Lin(d, r)
                self.compat_key = _Lin(d, r)
                self.compat_up = _Lin(r, d)
            if m in ("sir_lattice",):
                self.lattice_q = _Lin(d, r)
                self.lattice_k = _Lin(d, r)
            if m in ("sir_energy", "sir_full"):
                self.energy_tok = _Lin(d, d)
                self.energy_out = _Lin(d, 1)
        if m in STRUCT_MODES:
            r = self.pair_rank
            self.lat_tab_u = _Lin(d, r)                      # candidate features from the embedding table
            if m in CRF_MODES:
                self.lat_tab_v = _Lin(d, r)
                self.lat_w = _Lin(d, r)                      # context gate on the rank-r pair terms
                self.lat_w0 = nn.Parameter(torch.empty(r))   # context-free part of that gate
            if m in TT_MODES:
                R = self.tt_rank
                self.tt_pi = _Lin(d, R)
                self.tt_q, self.tt_k = _Lin(d, R), _Lin(d, R)
                self.tt_B = nn.Parameter(torch.empty(R, R))  # static transition logits
                self.tt_E = nn.Parameter(torch.empty(R, r))  # what each latent state prefers
                self.tt_esc = nn.Parameter(torch.empty(R))
            if m == "lat_cp":
                self.cp_pi = _Lin(d, self.cp_codes)
                self.cp_E = nn.Parameter(torch.empty(self.cp_codes, r))
                self.cp_esc = nn.Parameter(torch.empty(self.cp_codes))
            if m in CUT_MODES:
                self.draft_in = _Lin(d, d)                   # anchor memory, as in sir_tree
        if m == "pmi_chain":
            self.pmi_gate = _Lin(d, 1)
        if m == "corpus_code":
            C, r = self.code_classes, self.pair_rank
            self.code_u = nn.Parameter(torch.empty(C, r))
            self.code_v = nn.Parameter(torch.empty(C, r))
            self.code_w = _Lin(d, r)
            self.code_beta = nn.Parameter(torch.empty(1))   # weight on the corpus class PMI
        if m == "p1_selfpost":
            G, C = self.G, self.C
            self.q_out = _Lin(d, G * C)
            self.prior_in = _Lin(d, d)
            self.prior_code = nn.Parameter(torch.empty(G * C, d))
            self.prior_grp = nn.Parameter(torch.empty(G, d))
            self.prior_mlp = _MLP(d, mult)
            self.prior_out = _Lin(d, C)
            self.dec_code = nn.Parameter(torch.empty(G * C, d))
            self.self_post_in = _Lin(2 * d, d)
            self.self_post_mlp = _MLP(d, mult)
        if m in DEPTH_MODES:
            self.depth_in = _Lin(d, d)                        # the slot token's extra entry, zero init
            self.depth_slot = nn.Parameter(torch.empty(max(1, self.T - 1), d))
            if m == "depth_tree":                             # x0 of a slot whose token is unknown
                self.depth_mask = nn.Parameter(torch.empty(d))
            if m == "depth_roll":                             # previous slot's top state, zero init
                self.depth_fb = _Lin(d, d)
            # GPT.__init__ attaches depth_blocks (copies of the top trunk layers) when depth_share=0.
        if self.nce_props > 0:
            self.nce_tok, self.nce_hin = _Lin(d, d), _Lin(d, d)
            self.nce_pos = nn.Parameter(torch.empty(self.T, d))
            self.nce_layer = _Layer(d, n_head, mult)
            self.nce_out = _Lin(d, 1)

        # Corpus tables (nanochat/sap_tables.py). Buffers follow the module's device; the CPU
        # copy re-fills them after a meta-device to_empty (see init_weights).
        self._table_cpu, self._tab_V = None, None
        if self._table_keys() and self.table_path:
            self.set_tables(torch.load(self.table_path, map_location="cpu", weights_only=False))

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
        layers = list(self.dec) + list(getattr(self, "enc", []))
        if hasattr(self, "nce_layer"):
            layers.append(self.nce_layer)
        for layer in layers:
            torch.nn.init.zeros_(layer.attn.c_proj.weight)
            torch.nn.init.zeros_(layer.mlp.c_proj.weight)
        for p in (self.slot_emb, self.ctx_pos, getattr(self, "enc_pos", None),
                  getattr(self, "prior_code", None), getattr(self, "prior_grp", None),
                  getattr(self, "dec_code", None), getattr(self, "comp_emb", None),
                  getattr(self, "field_pos", None), getattr(self, "field_code", None),
                  getattr(self, "tt_E", None), getattr(self, "cp_E", None),
                  getattr(self, "code_u", None), getattr(self, "code_v", None),
                  getattr(self, "nce_pos", None), getattr(self, "depth_mask", None)):
            if p is not None:
                torch.nn.init.normal_(p, mean=0.0, std=1.0)
        # Output layers of the latent nets start small so the first KL terms are near zero.
        for name in ("q_out", "prior_out", "mix_out", "tt_pi", "cp_pi"):
            if hasattr(self, name):
                torch.nn.init.normal_(getattr(self, name).weight, std=0.001)
        if hasattr(self, "field_gate"):
            # Start with a context-independent field (gate=1); context dependence has to
            # earn its way in without making the initially random code field explode.
            torch.nn.init.zeros_(self.field_gate.weight)
        for name in ("prior_mlp", "self_post_mlp"):
            if hasattr(self, name):
                torch.nn.init.zeros_(getattr(self, name).c_proj.weight)
        # Every v4 coupling starts at zero, so each structured head starts as the independent
        # head on the same lattice and has to earn its dependence. tt_E / cp_E stay random:
        # identical latent states would receive identical gradients forever.
        for name in ("lat_w", "pmi_gate", "code_w", "nce_out", "depth_in", "depth_fb"):
            if hasattr(self, name):
                torch.nn.init.zeros_(getattr(self, name).weight)
        for name in ("lat_w0", "tt_B", "tt_esc", "cp_esc", "code_beta", "depth_slot"):
            if hasattr(self, name):
                torch.nn.init.zeros_(getattr(self, name))
        self._reload_tables()

    def embedding_parameters(self):
        """Lookup-style tables; the rest are matrices for Muon."""
        names = ("slot_emb", "ctx_pos", "enc_pos", "prior_code", "prior_grp", "dec_code", "comp_emb",
                 "field_pos", "field_code",
                 "lat_w0", "tt_B", "tt_E", "tt_esc", "cp_E", "cp_esc", "code_u", "code_v",
                 "code_beta", "nce_pos", "depth_slot", "depth_mask")
        return [getattr(self, n) for n in names if getattr(self, n, None) is not None]

    # ------------------------------------------------------------------ corpus tables
    def _table_keys(self):
        m, keys = self.mode, []
        if m == "pmi_chain":
            keys += ["pmi_ids", "pmi_vals"]
        if m == "corpus_code":
            keys += ["classes", "class_pmi"]
        if self.supp_min_count > 0:
            keys += ["supp2_keys" if m in CUT_MODES else "supp_keys"]
        if self.soft_eps > 0:
            keys += ["big_ids", "big_p"]
        return keys

    def set_tables(self, tables):
        """Install corpus tables (sap_tables.build_tables output) as non-persistent buffers."""
        need = self._table_keys()
        missing = [k for k in need if k not in tables]
        assert not missing, f"tables lack {missing}, needed by mode {self.mode}"
        V = int(tables["V"])
        assert V >= self.vocab_size, f"tables built for V={V} < vocab {self.vocab_size}"
        cpu = {}
        for k in need:
            v = tables[k]
            if k.endswith("_keys"):
                stored = int(tables.get("supp_min_store", torch.tensor(1)))
                assert self.supp_min_count >= stored, \
                    f"tables keep only pairs seen >= {stored} times; sap_supp_min_count={self.supp_min_count}"
                v = v[tables[k.replace("_keys", "_counts")] >= max(1, self.supp_min_count)]
            elif k == "classes":
                v = v.long()
                assert int(v.max()) < self.code_classes, \
                    f"tables have {int(v.max()) + 1} classes, sap_code_classes={self.code_classes}"
            elif k.endswith("_ids"):
                v = v.long()
            else:
                v = v.float()
            cpu[k] = v.contiguous()
        self._table_cpu, self._tab_V = cpu, V
        dev = torch.device("cpu") if self.slot_emb.is_meta else self.slot_emb.device
        for k, v in cpu.items():
            name = "tab_" + k
            self._buffers.pop(name, None)
            # copy=True: on CPU .to() would alias the master copy, and init_weights poisons
            # buffers with NaN before _reload_tables re-fills them from that copy.
            self.register_buffer(name, v.to(dev, copy=True), persistent=False)

    @torch.no_grad()
    def _reload_tables(self):
        if self._table_cpu is None:
            return
        for k, v in self._table_cpu.items():
            getattr(self, "tab_" + k).copy_(v)

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

    def _standard_layer(self, s, ctx_in, ctx_valid, layer, causal=False):
        """One ordinary slot layer, factored out so SIR can put a stochastic cut between layers."""
        N, T, _ = s.shape
        if ctx_in is not None and ctx_in.size(0) != N:
            rep = N // ctx_in.size(0)
            ctx_in = ctx_in.repeat_interleave(rep, dim=0)
            ctx_valid = ctx_valid.repeat_interleave(rep, dim=0)
        mask = self._mask(N, T, None if ctx_in is None else ctx_valid, causal, s.device)
        return layer(s, ctx_in, mask)

    def _reference_layer(self, s, draft, ctx_in, ctx_valid, layer, leave_one_out=True):
        """Cross-attend to a realised draft; optionally forbid slot i from reading draft i.

        ``draft`` may be shorter than T for anchor modes.  In that case it is a latent skeleton,
        not a token-aligned draft, so every output is allowed to read every anchor.
        """
        N, T, _ = s.shape
        if ctx_in is not None and ctx_in.size(0) != N:
            rep = N // ctx_in.size(0)
            ctx_in = ctx_in.repeat_interleave(rep, dim=0)
            ctx_valid = ctx_valid.repeat_interleave(rep, dim=0)
        D = draft.size(1)
        dm = torch.ones(T, D, dtype=torch.bool, device=s.device)
        if leave_one_out and D == T:
            dm.fill_diagonal_(False)
        dm = dm[None].expand(N, T, D)
        dnorm = _norm(draft)
        if ctx_in is None:
            kv, mask = dnorm, dm
        else:
            kv = torch.cat([ctx_in, dnorm], dim=1)
            cm = ctx_valid[:, None, :].expand(N, T, ctx_valid.size(1))
            mask = torch.cat([cm, dm], dim=-1)
        sn = _norm(s)
        s = s + layer.attn(sn, kv, mask)
        return s + layer.mlp(_norm(s))

    def _sir_candidates(self, s, y_safe, readout):
        """Chunked exact logits -> top-k candidates and exact target log-probability.

        Only ``(rows, k)`` tensors leave each chunk.  The full vocabulary logits are recomputed
        in backward, matching `_target_logprob`'s memory discipline.
        """
        N, T, d = s.shape
        flat = s.reshape(N * T, d)
        target = None if y_safe is None else y_safe.reshape(-1)
        vals_all, idx_all, lp_all = [], [], []
        k = min(self.sir_topk, self.vocab_size)

        def _chunk_with_target(ss, yy):
            logits = readout(ss)
            vals, idx = logits.topk(k, dim=-1)
            lp = logits.gather(-1, yy[:, None]).squeeze(-1) - torch.logsumexp(logits, dim=-1)
            return vals, idx, lp

        def _chunk_no_target(ss):
            return readout(ss).topk(k, dim=-1)

        for a in range(0, flat.size(0), self.chunk):
            ss = flat[a:a + self.chunk]
            if target is None:
                vals, idx = _chunk_no_target(ss)
            else:
                yy = target[a:a + self.chunk]
                if torch.is_grad_enabled() and ss.requires_grad:
                    vals, idx, lp = checkpoint(_chunk_with_target, ss, yy, use_reentrant=False)
                else:
                    vals, idx, lp = _chunk_with_target(ss, yy)
                lp_all.append(lp)
            vals_all.append(vals)
            idx_all.append(idx)
        vals = torch.cat(vals_all).view(N, T, k)
        idx = torch.cat(idx_all).view(N, T, k)
        lp = None if target is None else torch.cat(lp_all).view(N, T)
        return vals, idx, lp

    def _sir_draw(self, vals, idx, embed, soft_bridge, temperature=1.0, generator=None):
        """Sample one candidate per slot; optionally use its sparse expectation as the gradient."""
        if temperature <= 0:
            probs = torch.softmax(vals, dim=-1)
            choice = vals.argmax(-1)
        else:
            probs = torch.softmax(vals.float() / temperature, dim=-1)
            u = torch.rand(*probs.shape[:-1], device=probs.device, generator=generator)
            choice = pick(probs, u)
        ids = idx.gather(-1, choice[..., None]).squeeze(-1)
        hard = embed(ids)
        if soft_bridge:
            cand = embed(idx).to(hard.dtype)
            soft = (probs[..., None].to(hard.dtype) * cand).sum(-2)
            emb = hard - soft.detach() + soft
        else:
            emb = hard
        confidence = probs.gather(-1, choice[..., None]).squeeze(-1)
        return emb, ids, confidence, confidence.clamp_min(1e-12).log()

    @staticmethod
    def _shift_mean(x, offsets):
        """Mean of neighbouring positions at the requested non-zero offsets; never reads self."""
        N, T, d = x.shape
        out = x.new_zeros(N, T, d)
        count = x.new_zeros(1, T, 1)
        for off in offsets:
            if off <= 0 or off >= T:
                continue
            out[:, off:] += x[:, :-off]
            count[:, off:] += 1
            out[:, :-off] += x[:, off:]
            count[:, :-off] += 1
        return out / count.clamp_min(1)

    def _sir_message(self, draft):
        multi = self.mode in ("sir_compat", "sir_context", "sir_lattice", "sir_full")
        offsets = [1]
        if multi:
            off = 2
            while off < draft.size(1):
                offsets.append(off)
                off *= 2
            z = self.compat_down(_norm(draft))
            return self.compat_up(_norm(self._shift_mean(z, offsets)))
        return self.draft_in(_norm(self._shift_mean(draft, offsets)))

    def _sir_lattice_logits(self, vals, idx, embed):
        """One exact sum-product message in each direction on the restricted top-k chain."""
        if vals.size(1) < 2:
            return vals
        # Keep the compatibility path in the model's activation dtype. ``vals`` are
        # deliberately fp32 logits, and following their dtype here would turn this
        # otherwise small low-rank path into an accidental fp32 island on GPU.
        e = _norm(embed(idx))
        q = self.lattice_q(e[:, :-1])
        k = self.lattice_k(e[:, 1:])
        pair = torch.einsum("ntur,ntvr->ntuv", q, k) / math.sqrt(self.sir_rank)
        logp_l = torch.log_softmax(vals[:, :-1].float(), -1)
        logp_r = torch.log_softmax(vals[:, 1:].float(), -1)
        to_r = torch.logsumexp(logp_l[..., :, None] + pair.float(), dim=-2)
        to_l = torch.logsumexp(logp_r[..., None, :] + pair.float(), dim=-1)
        to_r = to_r - to_r.mean(-1, keepdim=True)
        to_l = to_l - to_l.mean(-1, keepdim=True)
        out = vals.float().clone()
        out[:, 1:] += to_r
        out[:, :-1] += to_l
        return out.to(vals.dtype)

    def _sir_context_loss(self, vals, idx, y_safe, valid, embed):
        """Soft plausible-error target derived from the true neighbours, restricted to draft top-k."""
        true_e = _norm(embed(y_safe))
        ctx = self._shift_mean(true_e, [1])
        q = self.compat_down(_norm(ctx))
        cand = self.compat_key(_norm(embed(idx).to(true_e.dtype)))
        reward = torch.einsum("ntr,ntkr->ntk", q, cand) / math.sqrt(self.sir_rank)
        target = torch.softmax(reward.float(), -1).detach()
        ce = -(target * torch.log_softmax(vals.float(), -1)).sum(-1)
        return (ce * valid).sum() / valid.sum().clamp_min(1)

    def _sir_energy_loss(self, s, y_safe, valid, embed):
        """Rank a real block above a Frankenstein block assembled from another context."""
        real = embed(y_safe).to(s.dtype)
        fake = real.roll(1, dims=0) if real.size(0) > 1 else real.roll(1, dims=1)
        sr = self.energy_out(_norm(s + self.energy_tok(_norm(real)))).squeeze(-1)
        sf = self.energy_out(_norm(s + self.energy_tok(_norm(fake)))).squeeze(-1)
        denom = valid.sum(-1).clamp_min(1)
        sr = (sr * valid).sum(-1) / denom
        sf = (sf * valid).sum(-1) / denom
        return F.softplus(sf - sr).mean()

    def _sir_tree_layout(self, device):
        """Breadth-first anchor rounds and the final-fill positions.

        Position zero is the root because, for language, the first emitted token is the most
        natural branch decision.  Every later round bisects the still-uncommitted intervals.
        ``sir_tree_levels=0`` expands the full tree; a positive value stops after that many
        anchor rounds and leaves the remaining positions for one parallel fill.

        Device tensors are precomputed and cached by (device, T, levels) to prevent
        unpinned host-to-device transfers during CUDA graph capture and avoid allocation
        overhead in steady-state decode.
        """
        key = (str(device), self.T, self.sir_tree_levels)
        if not hasattr(self, "_tree_layout_cache"):
            self._tree_layout_cache = {}
        cached = self._tree_layout_cache.get(key)
        if cached is not None:
            return cached[0], cached[1]

        levels = [[0]]
        intervals = [(1, self.T - 1)] if self.T > 1 else []
        while intervals and (self.sir_tree_levels == 0 or len(levels) < self.sir_tree_levels):
            level, nxt = [], []
            for lo, hi in intervals:
                if lo > hi:
                    continue
                mid = (lo + hi) // 2
                level.append(mid)
                if lo <= mid - 1:
                    nxt.append((lo, mid - 1))
                if mid + 1 <= hi:
                    nxt.append((mid + 1, hi))
            if not level:
                break
            levels.append(level)
            intervals = nxt
        committed = {p for level in levels for p in level}
        remaining = [p for p in range(self.T) if p not in committed]
        tensors = [torch.tensor(p, dtype=torch.long, device=device) for p in levels]
        rem_tensor = torch.tensor(remaining, dtype=torch.long, device=device)
        self._tree_layout_cache[key] = (tensors, rem_tensor)
        return tensors, rem_tensor

    def _sir_tree_positions(self, device):
        """Breadth-first balanced positions; each returned tensor is one parallel round."""
        return self._sir_tree_layout(device)[0]

    def _sir_tree_memory(self, ids, pos, embed, dtype):
        """Token-plus-position values make the committed anchor set order-aware."""
        raw = embed(ids).to(dtype) + self.slot_emb.index_select(0, pos)[None].to(dtype)
        return self.draft_in(_norm(raw))

    def _sir_tree_teacher(self, base, ctx_in, ctx_valid, y_safe, valid, readout, embed):
        """Exact log p(y|h) for the tree factorisation, teacher-forcing observed ancestors."""
        s0 = self._standard_layer(base, ctx_in, ctx_valid, self.dec[0])
        levels, remaining = self._sir_tree_layout(base.device)
        memory = []
        total = base.new_zeros(base.size(0), dtype=torch.float32)
        anchor_nll, anchor_tokens = total.new_zeros(()), valid.new_zeros((), dtype=torch.long)

        for level, pos in enumerate(levels):
            q = s0.index_select(1, pos)
            if level:
                q = self._reference_layer(q, torch.cat(memory, dim=1), ctx_in, ctx_valid,
                                          self.dec[1], leave_one_out=False)
            yy, vv = y_safe.index_select(1, pos), valid.index_select(1, pos)
            lp = self._slot_logprob(q, yy, readout)
            total = total + (lp * vv).sum(-1)
            anchor_nll = anchor_nll - (lp * vv).sum()
            anchor_tokens = anchor_tokens + vv.sum()
            memory.append(self._sir_tree_memory(yy, pos, embed, base.dtype))

        fill_nll, fill_tokens = total.new_zeros(()), valid.new_zeros((), dtype=torch.long)
        if remaining.numel():
            pos = remaining
            q = s0.index_select(1, pos)
            q = self._reference_layer(q, torch.cat(memory, dim=1), ctx_in, ctx_valid,
                                      self.dec[2], leave_one_out=False)
            yy, vv = y_safe.index_select(1, pos), valid.index_select(1, pos)
            lp = self._slot_logprob(q, yy, readout)
            total = total + (lp * vv).sum(-1)
            fill_nll = -(lp * vv).sum()
            fill_tokens = vv.sum()
        return total, {
            "tree_anchor_nll": anchor_nll / anchor_tokens.clamp_min(1),
            "tree_fill_nll": fill_nll / fill_tokens.clamp_min(1),
            "tree_rounds": total.new_tensor(float(len(levels) + bool(remaining.numel()))),
            "tree_anchors": total.new_tensor(float(sum(pos.numel() for pos in levels))),
        }

    def _sir_tree_sample(self, base, ctx_in, ctx_valid, readout, embed,
                         temperature=1.0, generator=None, oracle_y=None):
        """Ancestrally sample anchor levels, parallel within a level, then parallel-fill."""
        s0 = self._standard_layer(base, ctx_in, ctx_valid, self.dec[0])
        levels, remaining = self._sir_tree_layout(base.device)
        memory = []
        out = torch.zeros(base.size(0), self.T, dtype=torch.long, device=base.device)

        for level, pos in enumerate(levels):
            q = s0.index_select(1, pos)
            if level:
                q = self._reference_layer(q, torch.cat(memory, dim=1), ctx_in, ctx_valid,
                                          self.dec[1], leave_one_out=False)
            if oracle_y is None:
                vals, idx, _ = self._sir_candidates(q, None, readout)
                _, ids, _, _ = self._sir_draw(vals, idx, embed, False, temperature, generator)
            else:
                ids = oracle_y.index_select(1, pos)
            out[:, pos] = ids
            memory.append(self._sir_tree_memory(ids, pos, embed, base.dtype))

        if remaining.numel():
            pos = remaining
            q = s0.index_select(1, pos)
            q = self._reference_layer(q, torch.cat(memory, dim=1), ctx_in, ctx_valid,
                                      self.dec[2], leave_one_out=False)
            vals, idx, _ = self._sir_candidates(q, None, readout)
            _, ids, _, _ = self._sir_draw(vals, idx, embed, False, temperature, generator)
            out[:, pos] = ids
        return out

    def _sir_forward(self, base, ctx_in, ctx_valid, readout, embed, y_safe=None,
                     valid=None, temperature=1.0, generator=None, force_teacher=False):
        """Run a SIR stochastic computation graph and return final slot states plus auxiliaries."""
        s = self._standard_layer(base, ctx_in, ctx_valid, self.dec[0])
        soft = self.mode in ("sir_soft", "sir_full")
        draft_losses, context_losses, sample_logps = [], [], []

        def draw(states, pos=None, lattice=False):
            if pos is None:
                yy, vv = y_safe, valid
            elif pos.ndim == 1:
                yy = None if y_safe is None else y_safe.index_select(1, pos)
                vv = None if valid is None else valid.index_select(1, pos)
            else:
                yy = None if y_safe is None else y_safe.gather(1, pos)
                vv = None if valid is None else valid.gather(1, pos)
            vals, idx, lp = self._sir_candidates(states, yy, readout)
            if lattice:
                vals = self._sir_lattice_logits(vals, idx, embed)
            emb, ids, conf, sample_lp = self._sir_draw(vals, idx, embed, soft, temperature, generator)
            sampled = torch.ones_like(sample_lp, dtype=torch.bool)
            # Token-space posterior: during training, some realised draft positions come
            # from the observed block and the rest from the model's current prior. The
            # leave-one-out reference mask prevents copying y_i into output i. This is a
            # denoising ELBO-style bridge: it gives the refiner a branch-correlated draft,
            # while draft CE learns the aggregate posterior used at inference.
            if force_teacher and yy is not None:
                teacher = torch.ones_like(yy, dtype=torch.bool)
            elif self.training and yy is not None and self.sir_posterior_mix > 0:
                teacher = torch.rand(yy.shape, device=yy.device, generator=generator) < self.sir_posterior_mix
            else:
                teacher = None
            if teacher is not None:
                if vv is not None:
                    teacher &= vv
                truth = embed(yy).to(emb.dtype)
                emb = torch.where(teacher[..., None], truth, emb)
                ids = torch.where(teacher, yy, ids)
                conf = torch.where(teacher, torch.ones_like(conf), conf)
                sampled &= ~teacher
            if vv is not None:
                sampled &= vv
            sample_logps.append((sample_lp * sampled).sum(-1))
            if lp is not None:
                draft_losses.append(-(lp * vv).sum() / vv.sum().clamp_min(1))
            if self.mode in ("sir_context", "sir_full") and yy is not None and pos is None:
                context_losses.append(self._sir_context_loss(vals, idx, yy, vv, embed))
            return emb, ids, conf

        mode = self.mode
        if mode in ("sir_anchor", "sir_pyramid"):
            coarse_pos = torch.arange(0, self.T, self.sir_anchor_stride, device=base.device)
            coarse, ids, conf = draw(s.index_select(1, coarse_pos), coarse_pos)
            owner = torch.div(torch.arange(self.T, device=base.device), self.sir_anchor_stride,
                              rounding_mode="floor").clamp(max=coarse.size(1) - 1)
            s = s + self.draft_in(_norm(coarse.index_select(1, owner)))
            s = self._reference_layer(s, coarse, ctx_in, ctx_valid, self.dec[1], leave_one_out=False)
            draft, draft_ids, confidence = coarse, ids, conf
            next_layer = 2
            if mode == "sir_pyramid":
                fine_pos = torch.arange(0, self.T, self.sir_fine_stride, device=base.device)
                fine, draft_ids, confidence = draw(s.index_select(1, fine_pos), fine_pos)
                owner = torch.div(torch.arange(self.T, device=base.device), self.sir_fine_stride,
                                  rounding_mode="floor").clamp(max=fine.size(1) - 1)
                s = s + self.draft_in(_norm(fine.index_select(1, owner)))
                s = self._reference_layer(s, fine, ctx_in, ctx_valid, self.dec[2], leave_one_out=False)
                draft, next_layer = fine, 3
            for layer in self.dec[next_layer:]:
                s = self._reference_layer(s, draft, ctx_in, ctx_valid, layer, leave_one_out=False)
        else:
            draft, draft_ids, confidence = draw(s, lattice=mode == "sir_lattice")
            msg = self._sir_message(draft)
            if mode in ("sir_energy", "sir_full"):
                msg = msg + self.energy_tok(_norm(msg))
            s = s + msg
            s = self._reference_layer(s, draft, ctx_in, ctx_valid, self.dec[1], leave_one_out=True)
            next_layer = 2
            if mode in ("sir_conf", "sir_full"):
                nref = max(1, int(round(self.sir_refine_frac * self.T)))
                low = confidence.topk(nref, dim=1, largest=False).indices
                selected = s.gather(1, low[..., None].expand(-1, -1, s.size(-1)))
                d2, ids2, conf2 = draw(selected, low)
                draft = draft.scatter(1, low[..., None].expand(-1, -1, draft.size(-1)), d2)
                draft_ids = draft_ids.scatter(1, low, ids2)
                confidence = confidence.scatter(1, low, conf2)
                s = s + self._sir_message(draft)
                s = self._reference_layer(s, draft, ctx_in, ctx_valid, self.dec[2], leave_one_out=True)
                next_layer = 3
            for layer in self.dec[next_layer:]:
                s = self._reference_layer(s, draft, ctx_in, ctx_valid, layer, leave_one_out=True)

        aux = {
            "draft_loss": (torch.stack(draft_losses).mean() if draft_losses else None),
            "context_loss": (torch.stack(context_losses).mean() if context_losses else None),
            "draft_confidence": confidence.mean(),
            "draft_ids": draft_ids,
            "draft_sample_logprob": torch.stack(sample_logps).sum(0),
        }
        return s, aux

    # ------------------------------------------------------- SAP v4: exact sampling cuts
    @staticmethod
    def _etab(embed, embed_weight):
        """The token-embedding table behind `embed` (gradient-scaled when the trunk is detached)."""
        return embed_weight if embed_weight is not None else embed.weight

    def _lat_k(self):
        return max(1, min(self.lat_k, self.vocab_size - 1))

    @staticmethod
    def _read_states(h, s):
        """Slot 0 reads out of the trunk state itself: its distribution IS the next-token one."""
        return torch.cat([h[:, None].to(s.dtype), s[:, 1:]], dim=1)

    def _cut_layout(self, device):
        """Anchor (even) and fill (odd) slot indices, cached on device for graph capture."""
        key = (str(device), self.T)
        cache = self.__dict__.setdefault("_cut_cache", {})
        if key not in cache:
            cache[key] = (torch.arange(0, self.T, 2, device=device),
                          torch.arange(1, self.T, 2, device=device))
        return cache[key]

    def _lattice(self, s, y_safe, readout, prev=None, esc_u=None, esc_greedy=False, first_logits=None):
        """See _lattice_rows. first_logits (N, V): slot 0's logits already computed by the trunk
        (training reuses the next-token logits, so the head pays T-1 readouts, not T)."""
        if first_logits is None:
            return self._lattice_rows(s, y_safe, readout, prev, esc_u, esc_greedy)
        head = self._lattice_rows(first_logits[:, None], None if y_safe is None else y_safe[:, :1],
                                  lambda z: z.float(), None if prev is None else prev[:, :1],
                                  None if esc_u is None else esc_u[:, :1], esc_greedy)
        if s.size(1) == 1:
            return head
        tail = self._lattice_rows(s[:, 1:], None if y_safe is None else y_safe[:, 1:], readout,
                                  None if prev is None else prev[:, 1:],
                                  None if esc_u is None else esc_u[:, 1:], esc_greedy)
        return {k: torch.cat([head[k], tail[k]], dim=1) for k in head}

    def _lattice_rows(self, s, y_safe, readout, prev=None, esc_u=None, esc_greedy=False):
        """Per-slot candidate lattice from one exact readout per row.

        s (N, J, d). Returns a dict of (N, J, ...) tensors: vals/idx, the top-K logits and ids;
        esc, the log-mass outside the candidates (the escape state's unary). With targets: st, the
        target's lattice state (K = escape) and lp_tok = log p(y | escape) (0 for a candidate).
        prev (N, J) data token before each slot: adds the n-gram soft-target CE (C-soft).
        esc_u (N, J, E) uniforms: E escape-token draws per slot (esc_greedy: one argmax draw).
        Only (rows, K) tensors leave a chunk; the V-wide logits are recomputed in backward.
        """
        N, J, d = s.shape
        K = self._lat_k()
        flat = s.reshape(N * J, d)
        ys = None if y_safe is None else y_safe.reshape(-1)
        ps = None if prev is None else prev.reshape(-1)
        us = None if esc_u is None else esc_u.reshape(N * J, -1)
        big_ids = getattr(self, "tab_big_ids", None)
        big_p = getattr(self, "tab_big_p", None)

        def _chunk(ss, yc, pc, uc):
            logits = readout(ss)
            vals, idx = logits.topk(K, dim=-1)
            rest = logits.scatter(-1, idx, float("-inf"))
            esc = torch.logsumexp(rest, dim=-1)
            out = [vals, idx, esc]
            if yc is not None:
                match = idx == yc[:, None]
                has = match.any(-1)
                st = torch.where(has, match.int().argmax(-1), torch.full_like(yc, K))
                ly = logits.gather(-1, yc[:, None]).squeeze(-1)
                out += [st, torch.where(has, torch.zeros_like(ly), ly - esc)]
            if pc is not None:
                p0 = pc.clamp_min(0)
                lse = torch.logsumexp(logits, dim=-1)
                lg = logits.gather(-1, big_ids[p0]) - lse[:, None]
                out.append(-(big_p[p0] * lg).sum(-1) * (pc >= 0))
            if uc is not None:
                cdf = torch.softmax(rest, dim=-1).cumsum(-1)
                out.append(torch.searchsorted(cdf, uc.float().contiguous()).clamp(max=logits.size(-1) - 1))
            elif esc_greedy:
                out.append(rest.argmax(-1, keepdim=True))
            return tuple(out)

        parts = []
        for a in range(0, flat.size(0), self.chunk):
            args = (flat[a:a + self.chunk],
                    None if ys is None else ys[a:a + self.chunk],
                    None if ps is None else ps[a:a + self.chunk],
                    None if us is None else us[a:a + self.chunk])
            if torch.is_grad_enabled() and args[0].requires_grad:
                parts.append(checkpoint(_chunk, *args, use_reentrant=False))
            else:
                parts.append(_chunk(*args))
        cols = [torch.cat(c) for c in zip(*parts)]
        names = ["vals", "idx", "esc"] + (["st", "lp_tok"] if ys is not None else []) \
            + (["soft"] if ps is not None else []) + (["esc_tok"] if (us is not None or esc_greedy) else [])
        out = {}
        for name, c in zip(names, cols):
            out[name] = c.view(N, J, *c.shape[1:])
        return out

    def _crf_pair(self, sa, idx, E, skip):
        """Rank-r pair potentials between consecutive chain slots' candidates (escape row/col 0).

        phi_j(a, b) = sum_r w_jr u_r(a) v_r(b) / sqrt(r): u, v are learned features of the
        candidate tokens' embeddings and w_j a gate read from the two slot states, so the
        interaction between the same two tokens depends on the context. Optionally pairs never
        seen in the corpus (two apart when skip=True) are pushed down by supp_penalty.
        """
        N, J, K = idx.shape
        if J < 2:
            return torch.zeros(N, 0, K + 1, K + 1, device=idx.device)
        En = _norm(E)
        u = self.lat_tab_u(En).float()[idx[:, :-1]]
        v = self.lat_tab_v(En).float()[idx[:, 1:]]
        w = self.lat_w(_norm(sa[:, :-1]) + _norm(sa[:, 1:])).float() + self.lat_w0.float()
        pair = torch.einsum("njar,njr,njbr->njab", u, w, v) / math.sqrt(u.size(-1))
        if self.supp_min_count > 0:
            keys = self.tab_supp2_keys if skip else self.tab_supp_keys
            with torch.no_grad():
                ok = _seen(keys, idx[:, :-1, :, None], idx[:, 1:, None, :], self._tab_V)
            pair = pair - self.supp_penalty * (~ok).float()
        return F.pad(pair, (0, 1, 0, 1))

    def _tt_parts(self, sa, lat, E):
        """HMM over the lattice: (log_pi (N,R), log_A (N,J-1,R,R), log_emit (N,J,R,K+1))."""
        f = self.lat_tab_u(_norm(E)).float()[lat["idx"]]                         # (N, J, K, r)
        bias = torch.einsum("njkr,zr->njzk", f, self.tt_E.float()) / math.sqrt(f.size(-1))
        em = torch.cat([lat["vals"][:, :, None, :] + bias,
                        (lat["esc"][:, :, None] + self.tt_esc.float())[..., None]], dim=-1)
        log_emit = torch.log_softmax(em, dim=-1)
        log_pi = torch.log_softmax(self.tt_pi(_norm(sa[:, 0])).float(), dim=-1)
        R = self.tt_rank
        if sa.size(1) > 1:
            q = self.tt_q(_norm(sa[:, :-1])).float()
            k = self.tt_k(_norm(sa[:, 1:])).float()
            log_A = torch.log_softmax(self.tt_B.float() + q[..., :, None] * k[..., None, :], dim=-1)
        else:
            log_A = em.new_zeros(sa.size(0), 0, R, R)
        return log_pi, log_A, log_emit

    def _cp_emit(self, f, vals, esc, code_e, code_esc):
        """Emission log-probs over the lattice for codes. f (N,J,K,r); code_e (N,Z',r) or (Z',r)."""
        if code_e.dim() == 2:
            bias = torch.einsum("njkr,zr->nzjk", f, code_e.float())
            esc_b = code_esc.float()[None, :, None]
        else:
            bias = torch.einsum("njkr,nzr->nzjk", f, code_e.float())
            esc_b = code_esc.float()[:, :, None]
        em = torch.cat([vals[:, None] + bias / math.sqrt(f.size(-1)),
                        (esc[:, None] + esc_b)[..., None]], dim=-1)               # (N, Z', J, K+1)
        return torch.log_softmax(em, dim=-1)

    def _cp_ll(self, h, lat, obs, E):
        """Exact mixture likelihood over the Z codes, chunked over codes (checkpointed)."""
        f = self.lat_tab_u(_norm(E)).float()[lat["idx"]]
        log_pi = torch.log_softmax(self.cp_pi(_norm(h)).float(), dim=-1)            # (N, Z)
        st, vals, esc = lat["st"], lat["vals"], lat["esc"]
        N, J = st.shape

        def _blk(f_, vals_, esc_, ce, cesc, st_, obs_):
            le = self._cp_emit(f_, vals_, esc_, ce, cesc)                          # (N, Zc, J, K+1)
            o = le.gather(3, st_[:, None, :, None].expand(N, le.size(1), J, 1)).squeeze(-1)
            return (o * obs_[:, None, :]).sum(-1)                                    # (N, Zc)

        cols = []
        for c0 in range(0, self.cp_codes, self.cp_chunk):
            args = (f, vals, esc, self.cp_E[c0:c0 + self.cp_chunk], self.cp_esc[c0:c0 + self.cp_chunk],
                    st, obs.float())
            if torch.is_grad_enabled() and f.requires_grad:
                cols.append(checkpoint(_blk, *args, use_reentrant=False))
            else:
                cols.append(_blk(*args))
        return torch.logsumexp(log_pi + torch.cat(cols, dim=-1), dim=-1)

    def _struct_ll(self, sa, h, lat, obs, E, skip):
        """Exact log p of the observed lattice states under this mode's joint + escape-token terms."""
        m = self.mode
        if m in CRF_MODES:
            unary = torch.cat([lat["vals"], lat["esc"][..., None]], dim=-1)
            ll = chain_logprob(unary, self._crf_pair(sa, lat["idx"], E, skip), lat["st"], obs)
        elif m in TT_MODES:
            log_pi, log_A, log_emit = self._tt_parts(sa, lat, E)
            st = lat["st"]
            o = log_emit.gather(3, st[:, :, None, None].expand(-1, -1, log_emit.size(2), 1)).squeeze(-1)
            ll = hmm_loglik(log_pi, log_A, o * obs[..., None].float())
        else:
            ll = self._cp_ll(h, lat, obs, E)
        return ll + (lat["lp_tok"] * obs.float()).sum(-1)

    def _struct_draw(self, sa, h, lat, E, L, temperature, generator, skip):
        """L exact joint draws of lattice states (N, L, J); temperature 0 = most probable."""
        m = self.mode
        if m in CRF_MODES:
            unary = torch.cat([lat["vals"], lat["esc"][..., None]], dim=-1)
            return chain_sample(unary, self._crf_pair(sa, lat["idx"], E, skip), L, temperature, generator)
        if m in TT_MODES:
            log_pi, log_A, log_emit = self._tt_parts(sa, lat, E)
            return hmm_sample(log_pi, log_A, log_emit, L, temperature, generator)
        # lat_cp: one code per draw, then every slot in parallel
        f = self.lat_tab_u(_norm(E)).float()[lat["idx"]]
        log_pi = torch.log_softmax(self.cp_pi(_norm(h)).float(), dim=-1)
        N = log_pi.size(0)
        if temperature <= 0:
            code = log_pi.argmax(-1, keepdim=True).expand(N, L)
        else:
            u = torch.rand(N, L, device=h.device, generator=generator)
            cdf = torch.softmax(log_pi / temperature, -1).cumsum(-1)
            code = torch.searchsorted(cdf, u).clamp(max=self.cp_codes - 1)
        le = self._cp_emit(f, lat["vals"], lat["esc"], self.cp_E[code], self.cp_esc[code])   # (N, L, J, K+1)
        if temperature <= 0:
            return le.argmax(-1)
        u = torch.rand(le.shape[:-1], device=h.device, generator=generator)
        cdf = torch.softmax(le / temperature, -1).cumsum(-1)
        return torch.searchsorted(cdf, u[..., None].contiguous()).squeeze(-1).clamp(max=le.size(-1) - 1)

    @staticmethod
    def _states_to_tokens(states, lat):
        """Lattice states (N, L, J) -> token ids, using the pre-drawn escape tokens."""
        idx, esc = lat["idx"], lat["esc_tok"]                                       # (N,J,K), (N,J,E)
        N, L, J = states.shape
        K = idx.size(-1)
        cand = idx[:, None].expand(N, L, J, K).gather(3, states.clamp(max=K - 1)[..., None]).squeeze(-1)
        e = esc.transpose(1, 2)                                                     # (N, E, J)
        e = e.expand(N, L, J) if e.size(1) == 1 else e[:, :L]
        return torch.where(states == K, e, cand)

    def _cut_slots(self, base, ctx_in, ctx_valid):
        """Slot states before the cut: the first sap_cut_pre_layers ordinary layers."""
        s = base
        for layer in self.dec[:self.cut_pre]:
            s = self._standard_layer(s, ctx_in, ctx_valid, layer)
        return s

    def _cut_fill(self, s0, anchor_ids, ctx_in, ctx_valid, embed):
        """Fill-slot states after reading the committed anchors. anchor_ids (M, Ja), M = N or N*L."""
        A, Fi = self._cut_layout(s0.device)
        q = s0.index_select(1, Fi)
        if anchor_ids.size(0) != q.size(0):
            q = q.repeat_interleave(anchor_ids.size(0) // q.size(0), dim=0)
        mem = self._sir_tree_memory(anchor_ids, A, embed, s0.dtype)
        for layer in self.dec[self.cut_pre:]:
            q = self._reference_layer(q, mem, ctx_in, ctx_valid, layer, leave_one_out=False)
        return q

    @staticmethod
    def _prev_tokens(y_safe, pos=None):
        """Data token before each slot (or before each slot in pos); -1 before the first."""
        prev = torch.cat([y_safe.new_full((y_safe.size(0), 1), -1), y_safe[:, :-1]], dim=1)
        return prev if pos is None else prev.index_select(1, pos)

    def _pmi_logprob(self, read, y_safe, prev, alpha, readout):
        """log p(y_k | ctx, y_{k-1}) with the corpus PMI row of y_{k-1} added to slot k's logits."""
        N, T, d = read.shape
        flat, ys = read.reshape(N * T, d), y_safe.reshape(-1)
        p0 = prev.reshape(-1).clamp_min(0)
        ids, vals = self.tab_pmi_ids[p0], self.tab_pmi_vals[p0]
        a = (alpha * (prev >= 0)).reshape(-1)

        def _chunk(ss, yc, ic, vc, ac):
            logits = readout(ss).scatter_add(-1, ic, ac[:, None] * vc)
            return logits.gather(-1, yc[:, None]).squeeze(-1) - torch.logsumexp(logits, dim=-1)

        out = []
        for i in range(0, flat.size(0), self.chunk):
            args = (flat[i:i + self.chunk], ys[i:i + self.chunk], ids[i:i + self.chunk],
                    vals[i:i + self.chunk], a[i:i + self.chunk])
            if torch.is_grad_enabled() and args[0].requires_grad:
                out.append(checkpoint(_chunk, *args, use_reentrant=False))
            else:
                out.append(_chunk(*args))
        return torch.cat(out).view(N, T)

    def _class_readout(self, read, y_safe, readout):
        """Per-class log-mass of each slot's readout, and the target's within-class log-prob."""
        N, T, d = read.shape
        C, classes = self.code_classes, self.tab_classes
        flat = read.reshape(N * T, d)
        ys = None if y_safe is None else y_safe.reshape(-1)

        def _chunk(ss, yc):
            logits = readout(ss)
            mx = logits.max(-1, keepdim=True).values.detach()
            sums = torch.zeros(ss.size(0), C, device=ss.device, dtype=logits.dtype).index_add(
                1, classes[:logits.size(-1)], torch.exp(logits - mx))
            lse_c = mx + torch.log(sums.clamp_min(1e-30))
            if yc is None:
                return (lse_c,)
            cy = classes[yc]
            tok = logits.gather(-1, yc[:, None]).squeeze(-1) - lse_c.gather(-1, cy[:, None]).squeeze(-1)
            return lse_c, tok, cy

        parts = []
        for i in range(0, flat.size(0), self.chunk):
            args = (flat[i:i + self.chunk], None if ys is None else ys[i:i + self.chunk])
            if torch.is_grad_enabled() and args[0].requires_grad:
                parts.append(checkpoint(_chunk, *args, use_reentrant=False))
            else:
                parts.append(_chunk(*args))
        cols = [torch.cat(c) for c in zip(*parts)]
        return [c.view(N, T, *c.shape[1:]) for c in cols]

    def _code_pair(self, s):
        """Class pair potentials (N, T-1, C, C): rank-r learned terms plus the corpus class PMI."""
        w = self.code_w(_norm(s[:, :-1]) + _norm(s[:, 1:])).float()
        r = self.code_u.size(-1)
        pair = torch.einsum("cr,njr,dr->njcd", self.code_u.float(), w, self.code_v.float()) / math.sqrt(r)
        return pair + self.code_beta.float() * self.tab_class_pmi

    def _code_ll(self, s, lse_c, cls, obs):
        """Exact log p(class sequence), chunked over blocks so N x C x C never materialises whole."""
        N = s.size(0)
        rows = max(1, (1 << 22) // max(1, (s.size(1) - 1) * self.code_classes ** 2))

        def _blk(s_, u_, c_, o_):
            return chain_logprob(u_, self._code_pair(s_), c_, o_)

        out = []
        for i in range(0, N, rows):
            args = (s[i:i + rows], lse_c[i:i + rows], cls[i:i + rows], obs[i:i + rows])
            if torch.is_grad_enabled() and s.requires_grad:
                out.append(checkpoint(_blk, *args, use_reentrant=False))
            else:
                out.append(_blk(*args))
        return torch.cat(out)

    def _encode_self(self, h, h_future):
        """P1 posterior from the trunk's own state after reading the block (detached)."""
        x = self.self_post_in(torch.cat([_norm(h), _norm(h_future.detach().to(h.dtype))], dim=-1))
        x = _norm(x)
        return _norm(x + self.self_post_mlp(x))

    def _nce_scores(self, blocks, h, ctx_in, ctx_valid, embed):
        """Scorer phi for blocks (N, L, T) -> (N, L). Bidirectional over the block, reads context."""
        N, L, T = blocks.shape
        dt = h.dtype
        x = self.nce_tok(_norm(embed(blocks.reshape(N * L, T)).to(dt))) + self.nce_pos[None].to(dt)
        x = x + self.nce_hin(_norm(h)).repeat_interleave(L, dim=0)[:, None]
        ci = None if ctx_in is None else ctx_in.repeat_interleave(L, dim=0)
        cv = None if ctx_valid is None else ctx_valid.repeat_interleave(L, dim=0)
        x = self.nce_layer(x, ci, self._mask(N * L, T, cv, False, h.device))
        return self.nce_out(_norm(x.mean(1))).float().view(N, L)

    def _v4_ll(self, h, ctx_in, ctx_valid, y_safe, valid, readout, embed, E, train=False,
               first_logits=None):
        """Exact log p(block) (N,) for the v4 structured/table heads, plus what sampling reuses."""
        m = self.mode
        base = self._base(h)
        aux = {}
        n_esc = self.nce_neg if (train and self.nce_props > 0) else 0
        esc_u = None
        if m in LATTICE_MODES:
            s = self._decode(base, ctx_in, ctx_valid)
            if n_esc:
                esc_u = torch.rand(h.size(0), self.T, n_esc, device=h.device)
            prev = self._prev_tokens(y_safe) if (train and self.soft_eps > 0) else None
            lat = self._lattice(self._read_states(h, s), y_safe, readout, prev=prev, esc_u=esc_u,
                                first_logits=first_logits)
            ll = self._struct_ll(s, h, lat, valid, E, skip=False)
            aux.update(lat=lat, sa=s)
        elif m in CUT_MODES:
            s0 = self._cut_slots(base, ctx_in, ctx_valid)
            A, Fi = self._cut_layout(h.device)
            sa = s0.index_select(1, A)
            if n_esc:
                esc_u = torch.rand(h.size(0), A.numel(), n_esc, device=h.device)
            prev = self._prev_tokens(y_safe, A) if (train and self.soft_eps > 0) else None
            ya = y_safe.index_select(1, A)
            lat = self._lattice(self._read_states(h, sa), ya, readout, prev=prev, esc_u=esc_u,
                                first_logits=first_logits)
            ll = self._struct_ll(sa, h, lat, valid.index_select(1, A), E, skip=True)
            if Fi.numel():
                q = self._cut_fill(s0, ya, ctx_in, ctx_valid, embed)
                lp_f = self._slot_logprob(q, y_safe.index_select(1, Fi), readout)
                ll = ll + (lp_f * valid.index_select(1, Fi)).sum(-1)
            aux.update(lat=lat, sa=sa, s0=s0)
        elif m == "pmi_chain":
            s = self._decode(base, ctx_in, ctx_valid)
            alpha = self.pmi_gate(_norm(s)).float().squeeze(-1)
            lp = self._pmi_logprob(self._read_states(h, s), y_safe, self._prev_tokens(y_safe), alpha, readout)
            ll = (lp * valid).sum(-1)
        elif m == "corpus_code":
            s = self._decode(base, ctx_in, ctx_valid)
            lse_c, tok, cls = self._class_readout(self._read_states(h, s), y_safe, readout)
            ll = self._code_ll(s, lse_c, cls, valid) + (tok * valid).sum(-1)
        else:
            raise ValueError(m)
        if train and "lat" in aux and "soft" in aux["lat"]:
            aux["soft"] = aux["lat"]["soft"]
        return ll, aux

    @torch.no_grad()
    def _v4_draw(self, h, ctx_in, ctx_valid, readout, embed, E, L, temperature, generator,
                 aux=None, oracle_y=None):
        """L blocks per row (N, L, T) from the v4 head's exact sampler (one head pass per row)."""
        m = self.mode
        N, dev = h.size(0), h.device
        base = self._base(h)
        greedy = temperature <= 0

        def _esc_u(J):
            return None if greedy else torch.rand(N, J, L, device=dev, generator=generator)

        if m in LATTICE_MODES:
            if aux is not None and "esc_tok" in aux["lat"]:
                s, lat = aux["sa"], aux["lat"]
            else:
                s = self._decode(base, ctx_in, ctx_valid)
                lat = self._lattice(self._read_states(h, s), None, readout,
                                    esc_u=_esc_u(self.T), esc_greedy=greedy)
            states = self._struct_draw(s, h, lat, E, L, temperature, generator, skip=False)
            return self._states_to_tokens(states, lat)
        if m in CUT_MODES:
            A, Fi = self._cut_layout(dev)
            if aux is not None and "esc_tok" in aux["lat"]:
                s0, sa, lat = aux["s0"], aux["sa"], aux["lat"]
            else:
                s0 = self._cut_slots(base, ctx_in, ctx_valid)
                sa = s0.index_select(1, A)
                lat = self._lattice(self._read_states(h, sa), None, readout,
                                    esc_u=_esc_u(A.numel()), esc_greedy=greedy)
            if oracle_y is not None:
                anchors = oracle_y.index_select(1, A)[:, None].expand(N, L, A.numel())
            else:
                anchors = self._states_to_tokens(
                    self._struct_draw(sa, h, lat, E, L, temperature, generator, skip=True), lat)
            out = torch.zeros(N, L, self.T, dtype=torch.long, device=dev)
            out[:, :, A] = anchors
            if Fi.numel():
                q = self._cut_fill(s0, anchors.reshape(N * L, -1), ctx_in, ctx_valid, embed)
                lg = readout(q.reshape(-1, self.d)).view(N, L, Fi.numel(), -1)
                out[:, :, Fi] = self._choose(lg, temperature, generator)
            return out
        if m == "pmi_chain":
            s = self._decode(base, ctx_in, ctx_valid)
            alpha = self.pmi_gate(_norm(s)).float().squeeze(-1)                        # (N, T)
            lg_all = readout(self._read_states(h, s).reshape(-1, self.d)).view(N, self.T, -1)
            out = torch.zeros(N, L, self.T, dtype=torch.long, device=dev)
            for k in range(self.T):
                lg = lg_all[:, k][:, None].expand(N, L, lg_all.size(-1))
                if k > 0:
                    prev = out[:, :, k - 1]
                    lg = lg.scatter_add(-1, self.tab_pmi_ids[prev],
                                        alpha[:, k, None, None] * self.tab_pmi_vals[prev])
                out[:, :, k] = self._choose(lg, temperature, generator)
            return out
        if m == "corpus_code":
            s = self._decode(base, ctx_in, ctx_valid)
            read = self._read_states(h, s)
            (lse_c,) = self._class_readout(read, None, readout)
            cls = chain_sample(lse_c, self._code_pair(s), L, temperature, generator)  # (N, L, T)
            lg = readout(read.reshape(-1, self.d)).view(N, 1, self.T, -1)
            member = self.tab_classes[:lg.size(-1)][None, None, None, :] == cls[..., None]
            return self._choose(lg.masked_fill(~member, float("-inf")), temperature, generator)
        raise ValueError(m)

    @staticmethod
    def _choose(logits, temperature, generator):
        """Sample (or argmax) the last dim of logits (..., V)."""
        if temperature <= 0:
            return logits.argmax(-1)
        probs = torch.softmax(logits.float() / temperature, dim=-1)
        u = torch.rand(probs.shape[:-1], device=probs.device, generator=generator)
        return pick(probs, u)

    @torch.no_grad()
    def nce_logprob_estimate(self, h, ctx, ctx_valid, y, readout, embed=None, n_props=64,
                             valid_mask=None):
        """log p(y) for p ∝ q·exp(phi): exact log q + phi(y) - log mean_L exp(phi(proposals)).

        A consistent but biased (in log space) estimate; reported next to the exact q number,
        never in place of it.
        """
        valid = (y >= 0) if valid_mask is None else ((y >= 0) & valid_mask)
        y_safe = y.clamp_min(0)
        ctx_in = self._ctx(ctx)
        E = self._etab(embed, None)
        lq, _ = self._v4_ll(h, ctx_in, ctx_valid, y_safe, valid, readout, embed, E)
        props = self._v4_draw(h, ctx_in, ctx_valid, readout, embed, E, n_props, 1.0, None)
        phi_y = self._nce_scores(y_safe[:, None], h, ctx_in, ctx_valid, embed)[:, 0]
        phi_p = self._nce_scores(props, h, ctx_in, ctx_valid, embed)
        return lq + phi_y - (torch.logsumexp(phi_p, dim=-1) - math.log(n_props))

    def _base(self, h):
        return (self.h_proj(h)[:, None, :] + self.slot_emb[None].to(h.dtype))  # (N, T, d)

    def _field_cp_all(self, h):
        """All exact-mixture plan fields, shape (N, R, T, d).

        The learned positional basis is shared across plans, while every plan owns a
        rank-by-channel coefficient table.  A context-dependent positive gate modulates the
        basis channels.  This is deliberately a deterministic tensor construction: the only
        stochastic decision at inference is the single categorical plan draw.
        """
        pos = self.field_pos.to(h.dtype)
        code = self.field_code.to(h.dtype)
        gate = 1.0 + torch.tanh(self.field_gate(_norm(h)))
        return torch.einsum("tq,nq,rqd->nrtd", pos, gate, code) / math.sqrt(self.field_rank)

    def _field_cp_selected(self, h, code_idx):
        """Selected one-shot plan field, shape (N, T, d)."""
        pos = self.field_pos.to(h.dtype)
        code = self.field_code[code_idx].to(h.dtype)
        gate = 1.0 + torch.tanh(self.field_gate(_norm(h)))
        return torch.einsum("tq,nq,nqd->ntd", pos, gate, code) / math.sqrt(self.field_rank)

    def _field_noise(self, h, eps):
        """Map one shared Gaussian draw to a correlated positional field.

        eps has shape (N, K, q, dz).  K is a training-only Monte-Carlo axis; inference
        always uses K=1.  The result is (N, K, T, d), produced without token-dependent
        sampling or recurrence.
        """
        pos = self.field_pos.to(h.dtype)
        gate = 1.0 + torch.tanh(self.field_gate(_norm(h)))                         # (N, q)
        coeff = self.field_noise_in(eps.to(h.dtype))                              # (N, K, q, d)
        return torch.einsum("tq,nq,nkqd->nktd", pos, gate, coeff) / math.sqrt(self.field_rank)

    def _hard_token_embeddings_st(self, states, readout, embed):
        """Sample real token ids in the forward pass, with a sparse soft backward pass.

        Gumbel-max is over the full vocabulary, so the realised hard token follows the
        model's actual categorical distribution.  Only the top-k perturbed candidates
        carry gradients.  Chunk checkpointing avoids retaining a rows-by-vocabulary tensor.
        The embedding table is detached so the scoring geometry cannot collapse to make
        the energy objective cheap.
        """
        table = _norm(embed.weight[:self.vocab_size]).detach().float()
        flat = states.reshape(-1, self.d)
        out = []
        k = min(self.field_topk, self.vocab_size)

        def _chunk(ss):
            logits = readout(ss).float()
            u = torch.rand_like(logits).clamp_(1e-7, 1.0 - 1e-7)
            perturbed = logits - torch.log(-torch.log(u))
            vals, ids = perturbed.topk(k, dim=-1)
            soft = torch.softmax(vals / self.tau, dim=-1)
            hard = torch.zeros_like(soft)
            hard[:, 0] = 1.0
            weights = hard - soft.detach() + soft
            return (weights[..., None] * table[ids]).sum(dim=1)

        for a in range(0, flat.size(0), self.chunk):
            ss = flat[a:a + self.chunk]
            if torch.is_grad_enabled() and ss.requires_grad:
                out.append(checkpoint(_chunk, ss, use_reentrant=False))
            else:
                out.append(_chunk(ss))
        return torch.cat(out).view(*states.shape[:-1], self.d)

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
    def loss(self, h, ctx, ctx_valid, y, readout, embed=None, u_bins=None, h_future=None,
             embed_weight=None, slot0_logits=None):
        """Per-token training loss for one batch of blocks.

        h: (N, d) normalised trunk state at the block start; ctx: (N, W, d) window ending at
        that position (or None); ctx_valid: (N, W) bool; y: (N, T) targets, -1 = ignore;
        readout: (rows, d) -> (rows, V) softcapped logits; embed: ids -> input embeddings;
        u_bins: (lo, width) of the data tokens' CDF bins, only for inv_head; h_future: (N, d)
        trunk state at the block's last position, only for p1_selfpost; embed_weight: the
        table behind `embed` when it is a gradient-scaled view (v4 heads read it directly);
        slot0_logits: (N, V) the trunk's next-token logits at the block start, which the lattice
        and cut heads reuse as slot 0's readout.
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

        if m in DEPTH_MODES:
            raise RuntimeError(f"{m} runs through the trunk's layers: GPT._sap_depth_loss "
                               "computes it with GPT._sap_depth_logprob, never BlockHead.loss")
        if m in V4_MODES and m != "p1_selfpost":
            E = self._etab(embed, embed_weight)
            ll, aux = self._v4_ll(h, ctx_in, ctx_valid, y_safe, valid, readout, embed, E, train=True,
                                  first_logits=slot0_logits if m in STRUCT_MODES else None)
            loss = -ll.sum() / nvalid
            stats["block_nll"] = loss.detach()
            if self.soft_eps > 0 and "soft" in aux:
                has_prev = (aux["soft"] != 0).float()
                soft = aux["soft"].sum() / has_prev.sum().clamp_min(1)
                loss = loss + self.soft_eps * soft
                stats["soft_ce"] = soft.detach()
            if self.nce_props > 0:
                neg = self._v4_draw(h, ctx_in, ctx_valid, readout, embed, E, self.nce_neg, 1.0, None, aux=aux)
                blocks = torch.cat([y_safe[:, None], neg], dim=1)
                phi = self._nce_scores(blocks, h.detach(), None if ctx_in is None else ctx_in.detach(),
                                       ctx_valid, embed)
                nce = F.cross_entropy(phi, torch.zeros(N, dtype=torch.long, device=h.device))
                loss = loss + self.nce_weight * nce
                stats["nce_loss"] = nce.detach()
                stats["nce_acc"] = (phi[:, 0] > phi[:, 1:].max(-1).values).float().mean().detach()
        elif m == "sir_tree":
            ll, tree_stats = self._sir_tree_teacher(
                base, ctx_in, ctx_valid, y_safe, valid, readout, embed)
            loss = -ll.sum() / nvalid
            stats.update({k: v.detach() for k, v in tree_stats.items()})
            stats["final_nll"] = loss.detach()
        elif m in SIR_MODES:
            # Optimise the distribution *after sampling*, not the mean CE under an
            # independent draft. The latter has a product-of-marginals optimum and was
            # exactly the S01-v1 failure. Multiple realised drafts define a Monte-Carlo
            # marginal likelihood; posterior reference weights then give the discrete
            # draft policy a low-variance score-function credit signal.
            sample_ll, auxes, states = [], [], []
            for _ in range(self.sir_train_samples):
                s, aux = self._sir_forward(base, ctx_in, ctx_valid, readout, embed,
                                           y_safe=y_safe, valid=valid)
                lp = self._slot_logprob(s, y_safe, readout)
                sample_ll.append((lp * valid).sum(-1))
                auxes.append(aux)
                states.append(s)
            ll = torch.stack(sample_ll)                                  # (K, N)
            ref_weight = torch.softmax(ll.detach(), dim=0)
            final = -(torch.logsumexp(ll, dim=0) - math.log(ll.size(0))).sum() / nvalid
            sample_lp = torch.stack([a["draft_sample_logprob"] for a in auxes])
            advantage = ref_weight - (1.0 / ll.size(0))
            policy = -(advantage * sample_lp).sum() / nvalid

            draft_terms = [a["draft_loss"] for a in auxes if a["draft_loss"] is not None]
            context_terms = [a["context_loss"] for a in auxes if a["context_loss"] is not None]
            draft = torch.stack(draft_terms).mean() if draft_terms else None
            context = torch.stack(context_terms).mean() if context_terms else None
            terms, weights = [final], [1.0]
            if draft is not None and self.sir_draft_weight > 0:
                terms.append(draft * self.sir_draft_weight)
                weights.append(self.sir_draft_weight)
            if context is not None and self.sir_context_weight > 0:
                terms.append(context * self.sir_context_weight)
                weights.append(self.sir_context_weight)
            energy = None
            if m in ("sir_energy", "sir_full") and self.sir_energy_weight > 0:
                energy = torch.stack([self._sir_energy_loss(s, y_safe, valid, embed)
                                      for s in states]).mean()
                terms.append(energy * self.sir_energy_weight)
                weights.append(self.sir_energy_weight)
            loss = torch.stack(terms).sum() / sum(weights)
            # Zero-valued in the forward pass: this changes only the draft policy gradient,
            # keeping loss scales and the outer sap_lambda comparable with the v1 arms.
            if self.sir_policy_weight > 0 and ll.size(0) > 1:
                loss = loss + self.sir_policy_weight * (policy - policy.detach())
            stats["final_nll"] = final.detach()
            if draft is not None:
                stats["draft_nll"] = draft.detach()
            if context is not None:
                stats["context_soft_loss"] = context.detach()
            if energy is not None:
                stats["energy_rank_loss"] = energy.detach()
            stats["draft_policy_surrogate"] = policy.detach()
            stats["draft_ref_ess"] = (1.0 / ref_weight.square().sum(0)).mean().detach()
            stats["draft_confidence"] = torch.stack(
                [a["draft_confidence"] for a in auxes]).mean().detach()
        elif m == "indep":
            lp = self._slot_logprob(self._decode(base, ctx_in, ctx_valid), y_safe, readout)
            loss = -(lp * valid).sum() / nvalid
        elif m in ("local", "local_jacobi"):
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
        elif m in P1_MODES:
            enc = self._encode(h, y_safe, embed) if m == "p1_discrete" else self._encode_self(h, h_future)
            q_logits = self.q_out(enc).float().view(N, self.G, self.C)
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
        elif m == "field_energy":
            # The prior is also the inference distribution: no recognition model and no
            # prior/posterior gap.  Log-mean-exp lets different fields specialise, while
            # the sample-level score explicitly rewards coherent *realised token blocks*.
            K = self.field_samples
            eps = torch.randn(N, K, self.field_rank, self.dz, device=h.device, dtype=h.dtype)
            field = self._field_noise(h, eps)
            s = self._decode((base[:, None] + field).reshape(N * K, T, self.d),
                             ctx_in, ctx_valid).view(N, K, T, self.d)
            lp = self._slot_logprob(
                s.reshape(N * K, T, self.d), y_safe.repeat_interleave(K, dim=0), readout
            ).view(N, K, T)
            component_ll = (lp * valid[:, None]).sum(-1)
            marginal_nll = -(torch.logsumexp(component_ll, dim=1) - math.log(K)).sum() / nvalid

            sampled = self._hard_token_embeddings_st(
                s[:, :2].reshape(N * 2, T, self.d), readout, embed
            ).view(N, 2, T, self.d)
            target = _norm(embed(y_safe)).detach().float()
            es, spread = energy_score(sampled.float(), target, valid)
            energy = es.mean()
            loss = marginal_nll + self.field_energy_weight * energy
            stats["marginal_nll"] = marginal_nll.detach()
            stats["energy_score"] = energy.detach()
            stats["sample_spread"] = spread.mean().detach()
        elif m in ("cp", "field_cp"):
            log_pi = torch.log_softmax(self.mix_out(h).float(), dim=-1)                 # (N, R)
            if m == "cp":
                slot_in = base[:, None] + self.comp_emb[None, :, None].to(h.dtype)      # (N, R, T, d)
            else:
                slot_in = base[:, None] + self._field_cp_all(h)                         # (N, R, T, d)
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
        if self.mode in DEPTH_MODES:
            raise RuntimeError(f"{self.mode} decodes through the trunk's KV cache: GPT._sap_depth_sample")
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

        if m in V4_MODES and m != "p1_selfpost":
            E = embed_table if embed_table is not None else embed.weight
            L = self.nce_props if (self.nce_props > 0 and temperature > 0) else 1
            blocks = self._v4_draw(h, ctx_in, ctx_valid, readout, embed, E, L, temperature, generator,
                                   oracle_y=posterior_y)
            if L == 1:
                return blocks[:, 0]
            # Self-contrastive resampling: one of the L exact proposals, weighted by exp(phi).
            phi = self._nce_scores(blocks, h, ctx_in, ctx_valid, embed)
            choice = pick(torch.softmax(phi, dim=-1), _rand(N))
            return blocks.gather(1, choice[:, None, None].expand(N, 1, T)).squeeze(1)
        if m == "sir_tree":
            return self._sir_tree_sample(base, ctx_in, ctx_valid, readout, embed,
                                         temperature=temperature, generator=generator,
                                         oracle_y=posterior_y)
        if m in SIR_MODES:
            s, _ = self._sir_forward(base, ctx_in, ctx_valid, readout, embed,
                                     y_safe=posterior_y,
                                     valid=None if posterior_y is None else torch.ones_like(
                                         posterior_y, dtype=torch.bool),
                                     temperature=temperature, generator=generator,
                                     force_teacher=posterior_y is not None)
            return _choose(_logits(s))
        if m == "indep":
            return _choose(_logits(self._decode(base, ctx_in, ctx_valid)))
        if m in ("local", "local_jacobi"):
            # Only local_jacobi trades exactness for sweeps. `local` must sample its exact
            # sequential factorisation: jacobi_sweeps is always set from the config (default 2),
            # so reading it here made every `local` sample a 2-sweep Jacobi approximation.
            js = self.jacobi_sweeps if m == "local_jacobi" else 0
            if js > 0:
                # Fast Parallel Jacobi: 1 initial draft pass + js parallel causal sweeps
                s = self._decode(base, ctx_in, ctx_valid)
                out = _choose(_logits(s))
                for _ in range(js):
                    prev = torch.cat([out.new_zeros(N, 1), out[:, :-1]], dim=1)
                    tok = self.tok_in(_norm(embed(prev).to(h.dtype)))
                    tok = tok * (torch.arange(T, device=dev) > 0)[None, :, None].to(tok.dtype)
                    s = self._decode(base + tok, ctx_in, ctx_valid)
                    out = _choose(_logits(s))
                return out
            else:
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
        if m in P1_MODES:
            if posterior_y is not None and m == "p1_discrete":
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
        if m in ("cp", "field_cp"):
            pi = torch.softmax(self.mix_out(h).float(), dim=-1)
            r = pick(pi, _rand(N))
            if m == "cp":
                plan = self.comp_emb[r][:, None].to(h.dtype)
            else:
                plan = self._field_cp_selected(h, r)
            s = self._decode(base + plan, ctx_in, ctx_valid)
            return _choose(_logits(s))
        if m == "field_energy":
            # Exactly one shared field draw, one fixed-depth decoder call, then all T
            # categorical draws are made concurrently from the resulting logits.
            eps = _randn(N, 1, self.field_rank, self.dz)
            field = self._field_noise(h, eps)[:, 0]
            return _choose(_logits(self._decode(base + field, ctx_in, ctx_valid)))
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
    def block_logprob(self, h, ctx, ctx_valid, y, readout, embed=None, u_bins=None, n_samples=16,
                      valid_mask=None, h_future=None):
        """log p(block) per row: exact where the mode has a likelihood, else a bound.

        indep / local / local_jacobi / cp / field_cp: exact. field_energy uses a
        prior Monte-Carlo likelihood estimate. For local_jacobi this is the likelihood of
        the underlying teacher-forced local conditionals; the finite-sweep Jacobi sampler is
        still evaluated separately by generation quality. p1 / p2: importance-weighted bound
        with the posterior as proposal. inv_head: the sampled-noise bound averaged over draws
        (an upper bound on NLL). plain_noise / wta: Monte Carlo over prior noise. p3_energy:
        None.
        """
        if self.mode in DEPTH_MODES:
            raise RuntimeError(f"{self.mode} block log-probs come from GPT._sap_depth_logprob "
                               "(evaluate_block_bpb dispatches there)")
        valid = (y >= 0) if valid_mask is None else ((y >= 0) & valid_mask)
        y_safe = y.clamp_min(0)
        N, T = y.shape
        ctx_in = self._ctx(ctx)
        base = self._base(h)
        m = self.mode

        def _ll(s, yy):
            return (self._slot_logprob(s, yy, readout) * valid.repeat_interleave(
                s.size(0) // N, dim=0)).sum(-1)

        if m in V4_MODES and m != "p1_selfpost":
            return self._v4_ll(h, ctx_in, ctx_valid, y_safe, valid, readout, embed,
                               self._etab(embed, None))[0]
        if m == "sir_tree":
            return self._sir_tree_teacher(
                base, ctx_in, ctx_valid, y_safe, valid, readout, embed)[0]
        if m in SIR_MODES:
            terms = []
            for _ in range(n_samples):
                s, _ = self._sir_forward(base, ctx_in, ctx_valid, readout, embed)
                terms.append(_ll(s, y_safe))
            return torch.logsumexp(torch.stack(terms), dim=0) - math.log(n_samples)
        if m == "indep":
            return _ll(self._decode(base, ctx_in, ctx_valid), y_safe)
        if m in ("local", "local_jacobi"):
            prev = torch.cat([y_safe.new_zeros(N, 1), y_safe[:, :-1]], dim=1)
            tok = self.tok_in(_norm(embed(prev).to(h.dtype)))
            tok = tok * (torch.arange(T, device=h.device) > 0)[None, :, None].to(tok.dtype)
            return _ll(self._decode(base + tok, ctx_in, ctx_valid), y_safe)
        if m in ("cp", "field_cp"):
            log_pi = torch.log_softmax(self.mix_out(h).float(), dim=-1)
            if m == "cp":
                slot_in = base[:, None] + self.comp_emb[None, :, None].to(h.dtype)
            else:
                slot_in = base[:, None] + self._field_cp_all(h)
            s = self._decode(slot_in.reshape(N * self.R, T, self.d), ctx_in, ctx_valid)
            comp = _ll(s, y_safe.repeat_interleave(self.R, dim=0)).view(N, self.R)
            return torch.logsumexp(log_pi + comp, dim=-1)
        if m == "inv_head":
            acc = torch.zeros(N, device=h.device)
            for _ in range(n_samples):
                u = self._interior(*u_bins)
                acc += _ll(self._decode(base + self._u_emb(u, h.dtype), ctx_in, ctx_valid), y_safe)
            return acc / n_samples
        if m in P1_MODES:
            if m == "p1_selfpost" and h_future is None:
                return None                       # the posterior needs the trunk state after the block
            enc = self._encode(h, y_safe, embed) if m == "p1_discrete" else self._encode_self(h, h_future)
            q_logits = self.q_out(enc).float().view(N, self.G, self.C)
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
        if m == "field_energy":
            terms = []
            for _ in range(n_samples):
                eps = torch.randn(N, 1, self.field_rank, self.dz,
                                  device=h.device, dtype=h.dtype)
                field = self._field_noise(h, eps)[:, 0]
                terms.append(_ll(self._decode(base + field, ctx_in, ctx_valid), y_safe))
            return torch.logsumexp(torch.stack(terms), dim=0) - math.log(n_samples)
        if m in ("plain_noise", "wta"):
            terms = []
            for _ in range(n_samples):
                zemb = self.z_in(torch.randn(N, self.dz, device=h.device, dtype=h.dtype))
                terms.append(_ll(self._decode(base + zemb[:, None], ctx_in, ctx_valid), y_safe))
            return torch.logsumexp(torch.stack(terms), dim=0) - math.log(n_samples)
        return None  # p3_energy: likelihood-free

    @torch.no_grad()
    def latent_sensitivity(self, h, ctx, ctx_valid, readout, n_samples=8,
                           max_blocks=None, max_slots=None):
        """How much the slot distributions move with the plan/noise draw, in nats.

        Summed over slots: H(mean_z p) - mean_z H(p). Zero when the head ignores its latent,
        which is what plain_noise is expected to show. None for modes without a latent.

        max_blocks and max_slots bound diagnostic memory. When either cap is active, evenly
        spaced rows are used and the slot sum is scaled back to T slots. This keeps the metric
        in per-block nats while avoiding an O(samples * blocks * T * vocab) tensor at T=L.
        """
        m = self.mode
        if m not in ("p1_discrete", "p1_selfpost", "p2_gauss", "plain_noise", "wta", "cp",
                     "field_cp", "field_energy"):
            return None
        if n_samples < 1:
            raise ValueError(f"n_samples must be positive, got {n_samples}")
        if max_blocks is not None and max_blocks < 1:
            raise ValueError(f"max_blocks must be positive, got {max_blocks}")
        if max_slots is not None and max_slots < 1:
            raise ValueError(f"max_slots must be positive, got {max_slots}")

        if max_blocks is not None and h.size(0) > max_blocks:
            block_idx = torch.linspace(0, h.size(0) - 1, max_blocks, device=h.device).round().long()
            h = h.index_select(0, block_idx)
            if ctx is not None:
                ctx = ctx.index_select(0, block_idx)
            if ctx_valid is not None:
                ctx_valid = ctx_valid.index_select(0, block_idx)
        N = h.size(0)
        ctx_in = self._ctx(ctx)
        base = self._base(h)
        slot_idx = None
        if max_slots is not None and self.T > max_slots:
            slot_idx = torch.linspace(0, self.T - 1, max_slots, device=h.device).round().long()
        scored_slots = self.T if slot_idx is None else slot_idx.numel()
        prob_sum = None
        entropy_sum = None
        for _ in range(n_samples):
            if m in P1_MODES:
                z = torch.zeros(N, self.G, self.C, device=h.device)
                for g in range(self.G):
                    p = torch.softmax(self._prior_logits_p1(h, z)[:, g], dim=-1)
                    z[:, g] = F.one_hot(pick(p, torch.rand(N, device=h.device)), self.C).float()
                zemb = self._z_emb_p1(z).to(h.dtype)[:, None]
            elif m == "p2_gauss":
                mu, logvar = self._prior_p2(h)
                zemb = self.z_in((mu + torch.randn_like(mu) * (0.5 * logvar).exp()).to(h.dtype))[:, None]
            elif m in ("cp", "field_cp"):
                r = pick(torch.softmax(self.mix_out(h).float(), -1), torch.rand(N, device=h.device))
                zemb = (self.comp_emb[r][:, None].to(h.dtype) if m == "cp"
                        else self._field_cp_selected(h, r))
            elif m == "field_energy":
                eps = torch.randn(N, 1, self.field_rank, self.dz,
                                  device=h.device, dtype=h.dtype)
                zemb = self._field_noise(h, eps)[:, 0]
            else:
                zemb = self.z_in(torch.randn(N, self.dz, device=h.device, dtype=h.dtype))[:, None]
            s = self._decode(base + zemb, ctx_in, ctx_valid)
            if slot_idx is not None:
                s = s.index_select(1, slot_idx)
            p = torch.softmax(readout(s.reshape(-1, self.d)).float(), -1).view(N, scored_slots, -1)
            entropy = -(p * p.clamp_min(1e-12).log()).sum(-1)
            if prob_sum is None:
                prob_sum = p
                entropy_sum = entropy
            else:
                prob_sum.add_(p)
                entropy_sum.add_(entropy)

        def _H(p):
            return -(p * p.clamp_min(1e-12).log()).sum(-1)

        sensitivity = _H(prob_sum / n_samples) - entropy_sum / n_samples
        slot_scale = self.T / scored_slots
        return sensitivity.sum(-1) * slot_scale                                       # (N,)

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
        if m in ("cp", "field_cp"):
            dec, readout = dec * self.R, readout * self.R
            if m == "cp":
                extra += self.R * d
            else:
                extra += d * self.field_rank + self.R * T * self.field_rank * d
        elif m == "wta":
            dec, readout = dec * self.k_wta, readout * self.k_wta
        elif m == "p3_energy":
            dec, readout = dec * 2, 0                    # two samples, no V-wide readout
        elif m == "field_energy":
            # Inference draws one q-vector field and decodes once. Training evaluates K
            # prior fields for the marginal likelihood; two of those readouts also obtain
            # sparse soft gradients from their realised hard token samples.
            K = self.field_samples if train else 1
            dec, readout = dec * K, readout * K
            extra += (d * self.field_rank
                      + K * self.field_rank * self.dz * d
                      + K * T * self.field_rank * d)
            if train and self.field_energy_weight > 0:
                extra += 2 * T * min(self.field_topk, V) * d
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
        if m == "p1_selfpost":
            extra += 2 * d * d + 2 * mult * d * d             # self posterior (no token encoder)
            extra += d * self.G * self.C + self.G * (d * d + 2 * mult * d * d + d * self.C)
            extra += self.G * self.C * d
        if m in STRUCT_MODES:
            K, r = self._lat_k(), self.pair_rank
            J = T
            if m in CUT_MODES:
                # dec[0] over all T slots, then the reference layers over the fill slots only,
                # reading the anchors' memory and the context window.
                J, Jf = (T + 1) // 2, T // 2
                ref = (Jf * (2 + 2 * mult) * d * d + 2 * (J + W) * d * d + 2 * Jf * (J + W) * d)
                dec = self.cut_pre * per_layer + (L - self.cut_pre) * ref
                extra += J * d * d                            # anchor memory projection
            if train:
                readout = (T - 1) * V * d                    # slot 0 reuses the trunk's logits
            # Candidate features are gathered rows of a projected embedding table; pricing them
            # per candidate (rather than per vocabulary row) is the sparse implementation's cost.
            extra += J * K * d * r
            if m in CRF_MODES:
                extra += J * K * d * r + (J - 1) * (d * r + K * K * r + 2 * K * K)
            elif m in TT_MODES:
                R = self.tt_rank
                extra += J * R * K * r + 2 * (J - 1) * d * R + d * R + 2 * J * R * R + J * R * K
            else:
                Z = self.cp_codes
                extra += Z * J * K * r + Z * d + Z * J * K
        if m in DEPTH_MODES:
            # Slots 1..T-1 run m trunk layers each (q/k/v/o, a 4x MLP, attention over ~W_eff prefix
            # keys); slot 0 reuses the next-token logits. With copies the prefix keys/values are
            # recomputed per layer, amortised over the ~frac*Tq block starts of a row.
            Tq = 2048
            n_slot = sum(len(r[0]) for r in depth_tree_layout(T)) if m == "depth_tree" else T - 1
            dec = self.depth_m * n_slot * (12 * d * d + 2 * Tq * d)
            if not self.depth_share:
                dec += self.depth_m * 2 * d * d * Tq / max(1.0, self.frac * Tq)
            readout = (T - 1) * V * d if train else T * V * d
            extra += d * d
        if m == "pmi_chain":
            extra += T * d
        if m == "corpus_code":
            C, r = self.code_classes, self.pair_rank
            extra += (T - 1) * (d * r + C * C * r + 2 * C * C) + T * V
        if self.nce_props > 0:
            props = 1 + self.nce_neg
            extra += props * (T * (2 + 2 * mult) * d * d + 2 * (T + W) * d * d
                              + 2 * T * (T + W) * d + T * d * d + d * d + d)
        if m in ("local", "local_jacobi"):
            extra += T * d * d
        if m == "inv_head":
            extra += T * (2 * self.n_freq + 1) * d
        if m in SIR_MODES:
            if m == "sir_tree":
                levels = self._sir_tree_positions(self.slot_emb.device)
                anchors = sum(p.numel() for p in levels)
                committed = levels[0].numel()
                # dec[0] is one ordinary full-slot layer. dec[1] is recurrent over
                # tree levels and dec[2] performs the optional final parallel fill.
                tree_dec = per_layer
                for pos in levels[1:]:
                    q = pos.numel()
                    kv = committed + W
                    tree_dec += (q * (2 + 2 * mult) * d * d
                                 + 2 * kv * d * d + 2 * q * kv * d)
                    committed += q
                fill = T - anchors
                if fill:
                    kv = anchors + W
                    tree_dec += (fill * (2 + 2 * mult) * d * d
                                 + 2 * kv * d * d + 2 * fill * kv * d)
                dec = tree_dec
                # Exactly one prediction is made for each output token: either when its
                # anchor level commits it, or in the final fill.
                readout = T * V * d
                extra += anchors * d * d
                per_block = dec + readout + extra
                if not train:
                    return int(2 * per_block)
                return int(6 * per_block * self.frac)
            # One final readout is already in `readout`; price every stochastic draft
            # readout separately. Anchor/pyramid drafts touch only their skeleton rows.
            if m == "sir_anchor":
                draft_rows = math.ceil(T / self.sir_anchor_stride)
            elif m == "sir_pyramid":
                draft_rows = (math.ceil(T / self.sir_anchor_stride)
                              + math.ceil(T / self.sir_fine_stride))
            elif m in ("sir_conf", "sir_full"):
                draft_rows = T + max(1, int(round(self.sir_refine_frac * T)))
            else:
                draft_rows = T
            readout += draft_rows * V * d
            extra += T * d * d                         # draft_in / reference message
            if m in ("sir_compat", "sir_context", "sir_lattice", "sir_full"):
                extra += T * (2 * d * self.sir_rank + self.sir_rank * d)
            if m == "sir_lattice":
                k = min(self.sir_topk, V)
                extra += 2 * T * k * d * self.sir_rank + T * k * k * self.sir_rank
            if m in ("sir_energy", "sir_full"):
                extra += T * (2 * d * d + d)
        per_block = dec + readout + extra
        if not train:
            return int(2 * per_block)
        if m in SIR_MODES:
            per_block *= self.sir_train_samples
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
    # Self-contrastive heads: block_bpb stays the exact proposal q's; the resampled p ∝ q·exp(phi)
    # gets an importance-sampling estimate beside it, never in its place.
    nce_props = int(getattr(head, "nce_props", 0))
    nce_nats = torch.zeros((), dtype=torch.float64, device=device)
    it = iter(batches)
    depth = head.mode in DEPTH_MODES
    for step in range(steps):
        x, y = next(it)
        if depth:
            hid, states = model(x, skip_logits=True, sap_capture=True)
        else:
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
        tok_valid = (bytes_b > 0) & (yb >= 0)
        # A block is admissible if it contains valid text tokens. When T=L (e.g. 2048),
        # every packed sequence contains document boundary tokens (<|bos|>, zero bytes).
        # Masking zero-byte tokens rather than discarding the entire sequence allows exact
        # byte-normalized joint evaluation on the real text tokens.
        keep = tok_valid.sum(dim=1) > 0
        ctx, cv = model._sap_window(hid, b, t)

        if keep.any():
            # Compute NTP logits in chunks only for admissible blocks. This avoids both a
            # massive logits tensor and wasted likelihood work when T=L blocks contain a
            # zero-byte special token and must all be excluded from the joint score.
            hid_blocks = hid[b[:, None], t[:, None] + ks[None, :]][keep]
            ysafe_kept = ysafe[keep]
            tok_valid_kept = tok_valid[keep]
            hid_flat = hid_blocks.reshape(-1, hid_blocks.size(-1))
            ysafe_flat = ysafe_kept.reshape(-1)
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
                u_bins = (torch.cat(inv_lo).view(-1, T)[:, :T - 1],
                          torch.cat(inv_w).view(-1, T)[:, :T - 1])
            ntp_lp = torch.cat(ntp_lp_list).view(ysafe_kept.shape)
            ntp_nats += -(ntp_lp * tok_valid_kept).double().sum()
            n_bytes += bytes_b[keep].double().sum()
            n_blocks += keep.double().sum()
            if depth:
                # depth_local scores through the trunk's own top layers, row by row.
                n_b = starts.numel()
                slot0 = model._sap_readout(hid[b, t]).view(B, n_b, -1)
                ll = model._sap_depth_logprob(x, states, slot0, yb.view(B, n_b, T),
                                              starts[None].expand(B, n_b), tok_valid.view(B, n_b, T))
                block_nats += -ll.reshape(-1)[keep].double().sum()
            elif not likelihood_free:
                ctx_kept = None if ctx is None else ctx[keep]
                cv_kept = None if cv is None else cv[keep]
                lp = head.block_logprob(hid[b, t][keep], ctx_kept, cv_kept, yb[keep],
                                        model._sap_readout, embed=model.transformer.wte,
                                        u_bins=u_bins, n_samples=n_samples,
                                        valid_mask=tok_valid_kept,
                                        h_future=hid[b, (t + T - 1).clamp(max=Tq - 1)][keep])
                block_nats += -lp.double().sum()
                if nce_props > 0:
                    lp_nce = head.nce_logprob_estimate(hid[b, t][keep], ctx_kept, cv_kept, yb[keep],
                                                       model._sap_readout, embed=model.transformer.wte,
                                                       n_props=max(64, nce_props), valid_mask=tok_valid_kept)
                    nce_nats += -lp_nce.double().sum()
        if step == 0:
            # The uncapped T=L diagnostic would materialise samples*B*T*V probabilities
            # (8*128*2048*32768 in the d8 run). A stratified estimate is ample for detecting
            # latent collapse and keeps peak memory independent of sequence length.
            s = head.latent_sensitivity(hid[b, t], ctx, cv, model._sap_readout,
                                        max_blocks=4, max_slots=64)
            if s is not None:
                sens_sum += s.double().sum()
                sens_n += s.numel()
    if dist.is_initialized() and dist.get_world_size() > 1:
        for v in (block_nats, ntp_nats, n_bytes, n_blocks, sens_sum, sens_n, nce_nats):
            dist.all_reduce(v, op=dist.ReduceOp.SUM)
    nb = n_bytes.item()
    has_blocks = n_blocks.item() > 0 and nb > 0
    out = {
        "ntp_bpb_same_tokens": (ntp_nats.item() / (math.log(2) * nb)) if has_blocks else None,
        "block_bpb": (None if likelihood_free or not has_blocks
                      else block_nats.item() / (math.log(2) * nb)),
        "blocks": int(n_blocks.item()),
        "latent_sensitivity_nats": (sens_sum.item() / sens_n.item()) if sens_n.item() > 0 else None,
    }
    if out["block_bpb"] is not None and out["ntp_bpb_same_tokens"] is not None \
            and out["ntp_bpb_same_tokens"] > 0:
        out["block_over_ntp"] = out["block_bpb"] / out["ntp_bpb_same_tokens"]
    if nce_props > 0 and has_blocks:
        out["block_bpb_nce_est"] = nce_nats.item() / (math.log(2) * nb)
        if out["ntp_bpb_same_tokens"]:
            out["block_over_ntp_nce_est"] = out["block_bpb_nce_est"] / out["ntp_bpb_same_tokens"]
    return out
