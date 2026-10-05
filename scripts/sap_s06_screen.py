"""S06-Q: fast mechanism screen for the ten one-round SAP proposals.

This is intentionally an oracle-context experiment.  Every arm receives the exact
filtering belief of the phrase HMM and must model the next T tokens jointly.  That
isolates the proposed joint sampler from context-trunk optimisation.  It is not an
end-to-end language-model result.

All samplers draw their primitive random tape in one batch before deterministic
generation.  No arm uses a teacher, distillation, an AR verifier, rejection, or a
token-conditioned resampling loop.

Typical use:
    python -m scripts.sap_s06_screen --arm pss --smoke --device cpu
    python -m scripts.sap_s06_screen --arm pss --steps 2500 --device cuda
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from scripts.sap_synthetic import (
    PhraseHMM,
    build_phrase_hmm,
    filter_states,
    sample_sequences,
    true_block_logprob,
    true_marginal_logprob,
)


ARMS = (
    "pss", "rmlt", "mif", "crc", "fnt_v", "fnt_k", "cint", "dcmf",
    "argmax", "source_oracle",
    # S09 (s09_sap_tl_gates.md): S07 hypothesis A and its control. Their log_prob is an IWAE
    # lower bound, so the reported block KL is an upper bound on the true block KL.
    "scan_flow", "coupling_flow",
    # S10 (s10_sap_tl_brainstorm.md): reference-corrected self-inverting PTP and its ablations
    # (no cut, independent draft noise, id-order CDF). log_prob is a sequential importance-sampling
    # lower bound, so their block KL is an upper bound too.
    "rcptp", "cptp_si", "rcptp_indep", "rcptp_idorder",
    # S10 iteration 1: the generator inverts data under its own CDFs and trains by CE on the data
    # (PTP's from-scratch Eq. 13). cptp_seq is PTP's own method (exact, sequential); the jac arms
    # start from the AR-mode inversion and refine it with --n-inv parallel sweeps.
    "cptp_seq", "rcptp_seq", "cptp_jac", "rcptp_jac",
    # S11 (s11_sap_tl_brainstorm.md): Bridge LM, codes sampled coarse-to-fine in bisection order,
    # tokens emitted in parallel, trained by teacher forcing. bridge = oracle separator codes (the
    # filtered HMM state); bridge_tok = codes forced to be the tokens (the decision-variable control).
    "bridge", "bridge_tok", "bridge_ar", "bridge_pred",
)
PTP_ARMS = {"rcptp", "cptp_si", "rcptp_indep", "rcptp_idorder", "cptp_seq", "rcptp_seq", "cptp_jac", "rcptp_jac"}
BRIDGE_ARMS = {"bridge", "bridge_tok", "bridge_ar", "bridge_pred"}
EXACT_ARMS = {"pss", "rmlt", "mif", "crc", "argmax", "source_oracle", "scan_flow", "coupling_flow"} | PTP_ARMS | BRIDGE_ARMS
TRANSFORMS = ("identity", "delta", "butterfly", "xor_butterfly")


def categorical_from_uniform(probs: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """Vectorised inverse-CDF sampling; u has probs.shape[:-1]."""
    return (u.unsqueeze(-1) > probs.cumsum(-1)).sum(-1).clamp_max(probs.size(-1) - 1)


def st_one_hot(logits: torch.Tensor, gumbel: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
    soft = ((logits + gumbel) / tau).softmax(-1)
    hard = F.one_hot(soft.argmax(-1), logits.size(-1)).to(soft.dtype)
    return hard + soft - soft.detach()


def transform_block(y: torch.Tensor, kind: str, inverse: bool = False, V: int = 512) -> torch.Tensor:
    """Declared exact bijections used by MIF and its source-code oracle."""
    if kind == "identity":
        return y.clone()
    z = y.clone()
    if kind in ("delta",):
        if inverse:
            for i in range(1, z.size(-1)):
                z[:, i] = (z[:, i] + z[:, i - 1]) % V
        else:
            old = y
            z[:, 1:] = (old[:, 1:] - old[:, :-1]) % V
        return z
    use_xor = kind == "xor_butterfly"
    if kind not in ("butterfly", "xor_butterfly"):
        raise ValueError(kind)
    strides = []
    stride = 1
    while stride < z.size(-1):
        strides.append(stride)
        stride *= 2
    if inverse:
        strides.reverse()
    for stride in strides:
        old = z.clone()
        for start in range(0, z.size(-1), 2 * stride):
            width = min(stride, z.size(-1) - start - stride)
            if width <= 0:
                continue
            left = old[:, start:start + width]
            right = old[:, start + stride:start + stride + width]
            z[:, start + stride:start + stride + width] = (
                torch.bitwise_xor(right, left) if use_xor else
                ((right + left) % V if inverse else (right - left) % V)
            )
    return z


class BeliefPool:
    def __init__(self, hmm: PhraseHMM, ctx: int, T: int, size: int, gen: torch.Generator):
        self.hmm, self.ctx, self.T, self.size, self.gen = hmm, ctx, T, size, gen
        self.alpha = self.blocks = None
        self.used = size

    @torch.no_grad()
    def batch(self, B: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.used + B > self.size:
            seq = sample_sequences(self.hmm, max(B, self.size), self.ctx + self.T, self.gen)
            self.alpha = filter_states(self.hmm, seq[:, :self.ctx])[:, -1]
            self.blocks = seq[:, self.ctx:self.ctx + self.T]
            self.used = 0
        out = self.alpha[self.used:self.used + B], self.blocks[self.used:self.used + B]
        self.used += B
        return out


class SlotBackbone(nn.Module):
    """Bidirectional future-slot mixer; context is observed, future tokens are not."""
    def __init__(self, context_dim: int, T: int, width: int, depth: int, noise_dim: int = 0):
        super().__init__()
        self.T, self.width, self.noise_dim = T, width, noise_dim
        self.context = nn.Sequential(nn.Linear(context_dim, width), nn.SiLU(), nn.Linear(width, width))
        self.pos = nn.Parameter(torch.randn(T, width) / math.sqrt(width))
        self.noise = nn.Linear(noise_dim, width, bias=False) if noise_dim else None
        layer = nn.TransformerEncoderLayer(
            width, max(1, width // 32), 4 * width, dropout=0.0,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)

    def forward(self, alpha: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        h = self.context(alpha)[:, None, :] + self.pos[None]
        if noise is not None:
            h = h + self.noise(noise)
        return self.norm(self.blocks(h))


class PSS(nn.Module):
    exact = True
    training_multiplier = 1

    def __init__(self, context_dim: int, T: int, V: int, width: int, depth: int, states: int):
        super().__init__()
        self.T, self.V, self.S = T, V, states
        self.backbone = SlotBackbone(context_dim, T, width, depth)
        self.q = nn.Linear(width, V)
        self.pi = nn.Linear(width, states)
        self.trans = nn.Linear(width, states * states)
        # Fixed code permutations keep inference at one V-way readout per slot.
        masks = (torch.arange(states) * 131) % V
        self.register_buffer("masks", masks.long())

    def fields(self, alpha):
        h = self.backbone(alpha)
        logq = self.q(h).log_softmax(-1)
        logpi = self.pi(h.mean(1)).log_softmax(-1)
        logA = self.trans(h[:, 1:]).view(-1, self.T - 1, self.S, self.S).log_softmax(-1)
        return logq, logpi, logA

    def log_prob(self, alpha, y):
        logq, logpi, logA = self.fields(alpha)
        idx = torch.bitwise_xor(y[:, :, None], self.masks[None, None, :])
        emit = logq.gather(-1, idx)
        f = logpi + emit[:, 0]
        for t in range(1, self.T):
            f = torch.logsumexp(f[:, :, None] + logA[:, t - 1], 1) + emit[:, t]
        return torch.logsumexp(f, -1)

    @torch.no_grad()
    def sample(self, alpha):
        logq, logpi, logA = self.fields(alpha)
        B = alpha.size(0)
        # Entire primitive tape is drawn before any state-dependent computation.
        u0 = torch.rand(B, device=alpha.device)
        umap = torch.rand(B, self.T - 1, self.S, device=alpha.device)
        ux = torch.rand(B, self.T, device=alpha.device)
        s0 = categorical_from_uniform(logpi.exp(), u0)
        maps = categorical_from_uniform(logA.exp(), umap)
        # Hillis-Steele associative prefix composition of random functions.
        pref = maps
        off = 1
        while off < self.T - 1:
            nxt = pref.clone()
            outer, inner = pref[:, off:], pref[:, :-off]
            nxt[:, off:] = outer.gather(-1, inner)
            pref = nxt
            off *= 2
        rest = pref.gather(-1, s0[:, None, None].expand(B, self.T - 1, 1)).squeeze(-1)
        states = torch.cat((s0[:, None], rest), 1)
        x = categorical_from_uniform(logq.exp(), ux)
        return torch.bitwise_xor(x, self.masks[states])


class RMLT(nn.Module):
    exact = True
    training_multiplier = 1

    def __init__(self, context_dim: int, T: int, V: int, width: int, depth: int, states: int):
        super().__init__()
        assert T == 4, "quick-pass RMLT topology is the balanced four-leaf tree"
        self.T, self.V, self.S = T, V, states
        self.backbone = SlotBackbone(context_dim, T, width, depth)
        self.q = nn.Linear(width, V)
        self.pi = nn.Linear(width, states)
        self.left = nn.Linear(width, states * states)
        self.right = nn.Linear(width, states * states)
        self.register_buffer("masks", ((torch.arange(states) * 131) % V).long())

    def fields(self, alpha):
        h = self.backbone(alpha)
        root = h.mean(1)
        return (self.q(h).log_softmax(-1), self.pi(root).log_softmax(-1),
                self.left(root).view(-1, self.S, self.S).log_softmax(-1),
                self.right(root).view(-1, self.S, self.S).log_softmax(-1))

    def log_prob(self, alpha, y):
        logq, logpi, logL, logR = self.fields(alpha)
        idx = torch.bitwise_xor(y[:, :, None], self.masks[None, None, :])
        e = logq.gather(-1, idx)
        left = torch.logsumexp(logL + (e[:, 0] + e[:, 1])[:, None, :], -1)
        right = torch.logsumexp(logR + (e[:, 2] + e[:, 3])[:, None, :], -1)
        return torch.logsumexp(logpi + left + right, -1)

    @torch.no_grad()
    def sample(self, alpha):
        logq, logpi, logL, logR = self.fields(alpha)
        B = alpha.size(0)
        uroot = torch.rand(B, device=alpha.device)
        umaps = torch.rand(B, 2, self.S, device=alpha.device)
        ux = torch.rand(B, self.T, device=alpha.device)
        root = categorical_from_uniform(logpi.exp(), uroot)
        lm = categorical_from_uniform(logL.exp(), umaps[:, 0])
        rm = categorical_from_uniform(logR.exp(), umaps[:, 1])
        ls = lm.gather(1, root[:, None]).squeeze(1)
        rs = rm.gather(1, root[:, None]).squeeze(1)
        states = torch.stack((ls, ls, rs, rs), 1)
        x = categorical_from_uniform(logq.exp(), ux)
        return torch.bitwise_xor(x, self.masks[states])


class MIF(nn.Module):
    exact = True
    training_multiplier = 1

    def __init__(self, context_dim: int, T: int, V: int, width: int, depth: int):
        super().__init__()
        self.T, self.V, self.K = T, V, len(TRANSFORMS)
        self.backbone = SlotBackbone(context_dim, T, width, depth)
        self.readout = nn.Linear(width, self.K * V)
        self.mix = nn.Linear(width, self.K)

    def fields(self, alpha):
        h = self.backbone(alpha)
        q = self.readout(h).view(-1, self.T, self.K, self.V).permute(0, 2, 1, 3)
        return q.log_softmax(-1), self.mix(h.mean(1)).log_softmax(-1)

    def component_log_probs(self, alpha, y):
        logq, logw = self.fields(alpha)
        vals = []
        for k, kind in enumerate(TRANSFORMS):
            z = transform_block(y, kind, V=self.V)
            vals.append(logq[:, k].gather(-1, z[:, :, None]).squeeze(-1).sum(-1))
        return torch.stack(vals, -1), logw

    def log_prob(self, alpha, y):
        comp, logw = self.component_log_probs(alpha, y)
        return torch.logsumexp(logw + comp, -1)

    @torch.no_grad()
    def sample(self, alpha):
        logq, logw = self.fields(alpha)
        B = alpha.size(0)
        uk = torch.rand(B, device=alpha.device)
        uz = torch.rand(B, self.T, device=alpha.device)
        k = categorical_from_uniform(logw.exp(), uk)
        chosen = logq[torch.arange(B, device=alpha.device), k].exp()
        z = categorical_from_uniform(chosen, uz)
        out = torch.empty_like(z)
        for j, kind in enumerate(TRANSFORMS):
            take = k == j
            if take.any():
                out[take] = transform_block(z[take], kind, inverse=True, V=self.V)
        return out


class CRC(nn.Module):
    exact = True
    training_multiplier = 1

    def __init__(self, context_dim: int, T: int, V: int, width: int, depth: int,
                 roots: int = 16, children: int = 16):
        super().__init__()
        assert T == 4
        self.T, self.V, self.R, self.C = T, V, roots, children
        self.context = nn.Sequential(nn.Linear(context_dim, width), nn.SiLU(), nn.Linear(width, width))
        self.root = nn.Linear(width, roots)
        self.child_l = nn.Linear(width, roots * children)
        self.child_r = nn.Linear(width, roots * children)
        # Decomposable leaves are global; context controls the circuit's sum nodes.
        self.leaf = nn.Parameter(torch.randn(2, children, 2, V) * 0.01)

    def fields(self, alpha):
        h = self.context(alpha)
        logroot = self.root(h).log_softmax(-1)
        logL = self.child_l(h).view(-1, self.R, self.C).log_softmax(-1)
        logR = self.child_r(h).view(-1, self.R, self.C).log_softmax(-1)
        return logroot, logL, logR, self.leaf.log_softmax(-1)

    def log_prob(self, alpha, y):
        logroot, logL, logR, leaf = self.fields(alpha)
        eL = leaf[0, :, 0, y[:, 0]].t() + leaf[0, :, 1, y[:, 1]].t()
        eR = leaf[1, :, 0, y[:, 2]].t() + leaf[1, :, 1, y[:, 3]].t()
        l = torch.logsumexp(logL + eL[:, None, :], -1)
        r = torch.logsumexp(logR + eR[:, None, :], -1)
        return torch.logsumexp(logroot + l + r, -1)

    @torch.no_grad()
    def sample(self, alpha):
        logroot, logL, logR, leaf = self.fields(alpha)
        B = alpha.size(0)
        ur = torch.rand(B, device=alpha.device)
        uc = torch.rand(B, 2, self.R, device=alpha.device)
        uy = torch.rand(B, 4, device=alpha.device)
        root = categorical_from_uniform(logroot.exp(), ur)
        lm = categorical_from_uniform(logL.exp(), uc[:, 0])
        rm = categorical_from_uniform(logR.exp(), uc[:, 1])
        lc = lm.gather(1, root[:, None]).squeeze(1)
        rc = rm.gather(1, root[:, None]).squeeze(1)
        probs = torch.stack((leaf[0, lc, 0].exp(), leaf[0, lc, 1].exp(),
                             leaf[1, rc, 0].exp(), leaf[1, rc, 1].exp()), 1)
        return categorical_from_uniform(probs, uy)


class NoiseGenerator(nn.Module):
    exact = False

    def __init__(self, context_dim: int, T: int, V: int, width: int, depth: int, noise_dim: int = 32):
        super().__init__()
        self.T, self.V, self.noise_dim = T, V, noise_dim
        self.backbone = SlotBackbone(context_dim, T, width, depth, noise_dim)
        self.readout = nn.Linear(width, V)

    def logits(self, alpha: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        B, K = noise.shape[:2]
        aa = alpha[:, None].expand(B, K, -1).reshape(B * K, -1)
        z = noise.reshape(B * K, self.T, self.noise_dim)
        return self.readout(self.backbone(aa, z)).view(B, K, self.T, self.V)

    def tape(self, alpha, K):
        B = alpha.size(0)
        noise = torch.randn(B, K, self.T, self.noise_dim, device=alpha.device)
        u = torch.rand(B, K, self.T, self.V, device=alpha.device).clamp_(1e-6, 1 - 1e-6)
        gumbel = -torch.log(-torch.log(u))
        return noise, gumbel

    @torch.no_grad()
    def sample(self, alpha):
        noise, gumbel = self.tape(alpha, 1)
        return (self.logits(alpha, noise) + gumbel).argmax(-1)[:, 0]


class FNTV(NoiseGenerator):
    training_multiplier = 4

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        gen = torch.Generator().manual_seed(6061)
        code = torch.randint(0, 2, (self.V, 32), generator=gen).float().mul_(2).sub_(1)
        self.register_buffer("code", code / math.sqrt(code.size(1)))

    def loss(self, alpha, y):
        K = self.training_multiplier
        noise, gumbel = self.tape(alpha, K)
        logits = self.logits(alpha, noise)
        marginal = logits.softmax(-1).mean(1)
        nll = -marginal.gather(-1, y[:, :, None]).clamp_min(1e-9).log().mean()
        draws = st_one_hot(logits, gumbel) @ self.code
        obs = self.code[y]
        scores = []
        for lag in range(1, self.T):
            # Equal token codes have zero distance. Clamp before sqrt so the derivative at
            # exactly zero is finite; the unclamped sqrt produced inf*0 -> NaN gradients.
            target = (obs[:, lag:] - obs[:, :-lag]).square().sum(-1).clamp_min(1e-6).sqrt()
            pred = ((draws[:, :, lag:] - draws[:, :, :-lag]).square().sum(-1)
                    .clamp_min(1e-6).sqrt().mean(1))
            scores.append((target - pred).square().mean() / lag)
        return nll + 5.0 * torch.stack(scores).sum(), {"marginal_nll": float(nll.detach())}


class FNTK(NoiseGenerator):
    training_multiplier = 4

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        gen = torch.Generator().manual_seed(6062)
        code = torch.randint(0, 2, (self.V, 24), generator=gen).float().mul_(2).sub_(1)
        self.register_buffer("code", code / math.sqrt(code.size(1)))

    def features(self, code):
        pieces = [code.flatten(-2)]
        for lag in range(1, self.T):
            pieces.append((code[..., lag:, :] * code[..., :-lag, :]).flatten(-2))
        return torch.cat(pieces, -1)

    @staticmethod
    def kernel(a, b):
        d2 = (a - b).square().mean(-1)
        return sum(torch.exp(-d2 / s) for s in (0.1, 0.25, 0.5, 1.0)) / 4

    def loss(self, alpha, y):
        K = self.training_multiplier
        noise, gumbel = self.tape(alpha, K)
        logits = self.logits(alpha, noise)
        marginal = logits.softmax(-1).mean(1)
        nll = -marginal.gather(-1, y[:, :, None]).clamp_min(1e-9).log().mean()
        f = self.features(st_one_hot(logits, gumbel) @ self.code)
        fy = self.features(self.code[y])
        kxx = []
        for i in range(K):
            for j in range(i + 1, K):
                kxx.append(self.kernel(f[:, i], f[:, j]))
        model_model = torch.stack(kxx).mean()
        model_data = torch.stack([self.kernel(f[:, i], fy) for i in range(K)]).mean()
        score = model_model - 2 * model_data
        return nll + 4.0 * score, {"marginal_nll": float(nll.detach()),
                                  "kernel_score": float(score.detach())}


class CINT(NoiseGenerator):
    training_multiplier = 8

    def loss(self, alpha, y):
        K = self.training_multiplier
        noise, _ = self.tape(alpha, K)
        logits = self.logits(alpha, noise)
        target = y[:, None, :, None].expand(-1, K, -1, 1)
        ce = -logits.log_softmax(-1).gather(-1, target).squeeze(-1).sum(-1)
        winner = ce.min(1).values.mean() / self.T
        marginal = logits.softmax(-1).mean(1)
        marg = -marginal.gather(-1, y[:, :, None]).clamp_min(1e-9).log().mean()
        return winner + 0.1 * marg, {"winner_ce": float(winner.detach()),
                                    "marginal_nll": float(marg.detach())}


class JVPTransformerBlock(nn.Module):
    """Small explicit self-attention block whose ops support forward-mode AD."""
    def __init__(self, width: int, heads: int):
        super().__init__()
        assert width % heads == 0
        self.heads, self.head_dim = heads, width // heads
        self.ln1, self.ln2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.proj = nn.Linear(width, width)
        self.ff = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width))

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).chunk(3, -1)
        def heads(a):
            return a.view(B, T, self.heads, self.head_dim).transpose(1, 2)
        q, k, v = heads(q), heads(k), heads(v)
        att = (q @ k.transpose(-1, -2) / math.sqrt(self.head_dim)).softmax(-1)
        mixed = (att @ v).transpose(1, 2).reshape(B, T, D)
        x = x + self.proj(mixed)
        return x + self.ff(self.ln2(x))


class MeanFlow(nn.Module):
    exact = False
    training_multiplier = 2  # forward plus JVP

    def __init__(self, context_dim: int, T: int, V: int, width: int, depth: int):
        super().__init__()
        self.T, self.V, self.width = T, V, width
        self.inp = nn.Linear(V, width)
        self.context = nn.Sequential(nn.Linear(context_dim, width), nn.SiLU(), nn.Linear(width, width))
        self.time = nn.Sequential(nn.Linear(3, width), nn.SiLU(), nn.Linear(width, width))
        self.pos = nn.Parameter(torch.randn(T, width) / math.sqrt(width))
        # PyTorch's fused SDPA lacks forward-mode AD on some CPU/GPU backends. MeanFlow's
        # defining JVP needs forward AD, so use an explicit attention block here.
        self.blocks = nn.ModuleList([JVPTransformerBlock(width, max(1, width // 32))
                                     for _ in range(depth)])
        self.norm = nn.LayerNorm(width)
        self.out = nn.Linear(width, V)

    def forward(self, z, r, t, alpha):
        rt = torch.stack((r, t, t - r), -1)
        h = self.inp(z) + self.context(alpha)[:, None] + self.time(rt)[:, None] + self.pos[None]
        for block in self.blocks:
            h = block(h)
        return self.out(self.norm(h))

    def loss(self, alpha, y):
        x = F.one_hot(y, self.V).to(alpha.dtype)
        eps = torch.randn_like(x)
        a = torch.randn(y.size(0), device=y.device).sub_(0.4).sigmoid()
        b = torch.randn(y.size(0), device=y.device).sub_(0.4).sigmoid()
        r, t = torch.minimum(a, b), torch.maximum(a, b)
        # Paper default: only 25% of examples use a nonzero interval.
        same = torch.rand_like(r) >= 0.25
        r = torch.where(same, t, r)
        z = (1 - t[:, None, None]) * x + t[:, None, None] * eps
        v = eps - x
        fn = lambda zz, rr, tt: self.forward(zz, rr, tt, alpha)
        u, dudt = torch.func.jvp(fn, (z, r, t), (v, torch.zeros_like(r), torch.ones_like(t)))
        target = (v - (t - r)[:, None, None] * dudt).detach()
        err = (u - target).square().mean((1, 2))
        weight = (err.detach() + 1e-3).rsqrt()
        return (weight * err).mean(), {"flow_mse": float(err.mean().detach())}

    @torch.no_grad()
    def sample(self, alpha):
        eps = torch.randn(alpha.size(0), self.T, self.V, device=alpha.device)
        r = torch.zeros(alpha.size(0), device=alpha.device)
        t = torch.ones_like(r)
        x = eps - self.forward(eps, r, t, alpha)
        return x.argmax(-1)


class ArgmaxCoupling(nn.Module):
    """One exact bipartite modular coupling with an ST shift estimator."""
    exact = True
    training_multiplier = 1

    def __init__(self, context_dim: int, T: int, V: int, width: int, depth: int):
        super().__init__()
        assert T % 2 == 0
        self.T, self.V = T, V
        self.backbone = SlotBackbone(context_dim, T, width, depth)
        self.base = nn.Linear(width, V)
        self.token = nn.Embedding(V, width)
        self.shift = nn.Sequential(nn.Linear(2 * width, width), nn.SiLU(), nn.Linear(width, V))
        self.register_buffer("offsets", torch.arange(V))

    def fields(self, alpha, even_tokens):
        h = self.backbone(alpha)
        logq = self.base(h).log_softmax(-1)
        pair_h = h[:, 1::2]
        shift = self.shift(torch.cat((pair_h, self.token(even_tokens)), -1))
        return logq, shift

    def log_prob(self, alpha, y):
        even, odd = y[:, 0::2], y[:, 1::2]
        logq, shift = self.fields(alpha, even)
        lp_even = logq[:, 0::2].gather(-1, even[:, :, None]).squeeze(-1).sum(-1)
        hard = F.one_hot(shift.argmax(-1), self.V).to(shift.dtype)
        soft = shift.softmax(-1)
        st = hard + soft - soft.detach()
        candidate = (odd[:, :, None] + self.offsets[None, None, :]) % self.V
        candidate_lp = logq[:, 1::2].gather(-1, candidate)
        # Forward value is exact for the argmax bijection; backward is the published ST family.
        return lp_even + (st * candidate_lp).sum((-1, -2))

    @torch.no_grad()
    def sample(self, alpha):
        h = self.backbone(alpha)
        q = self.base(h).softmax(-1)
        u = torch.rand(alpha.size(0), self.T, device=alpha.device)
        z = categorical_from_uniform(q, u)
        even = z[:, 0::2]
        shift = self.shift(torch.cat((h[:, 1::2], self.token(even)), -1))
        s = shift.argmax(-1)
        y = z.clone()
        y[:, 1::2] = (z[:, 1::2] - s) % self.V
        return y


class SourceOracle(nn.Module):
    """Same constrained marginal predictor for every declared reversible code."""
    exact = True
    training_multiplier = len(TRANSFORMS)

    def __init__(self, context_dim: int, T: int, V: int, width: int, depth: int):
        super().__init__()
        self.T, self.V, self.K = T, V, len(TRANSFORMS)
        self.backbone = SlotBackbone(context_dim, T, width, depth)
        self.readout = nn.Linear(width, self.K * V)
        self.best_kind = "identity"

    def fields(self, alpha):
        h = self.backbone(alpha)
        return self.readout(h).view(-1, self.T, self.K, self.V).permute(0, 2, 1, 3).log_softmax(-1)

    def all_log_probs(self, alpha, y):
        logq = self.fields(alpha)
        vals = []
        for k, kind in enumerate(TRANSFORMS):
            z = transform_block(y, kind, V=self.V)
            vals.append(logq[:, k].gather(-1, z[:, :, None]).squeeze(-1).sum(-1))
        return torch.stack(vals, -1)

    def log_prob(self, alpha, y):
        return self.all_log_probs(alpha, y)[:, TRANSFORMS.index(self.best_kind)]

    @torch.no_grad()
    def sample(self, alpha):
        k = TRANSFORMS.index(self.best_kind)
        q = self.fields(alpha)[:, k].exp()
        z = categorical_from_uniform(q, torch.rand(alpha.size(0), self.T, device=alpha.device))
        return transform_block(z, self.best_kind, inverse=True, V=self.V)


def make_model(args, hmm):
    common = (hmm.S, args.T, args.vocab, args.width, args.depth)
    if args.arm == "pss":
        return PSS(*common, args.states)
    if args.arm == "rmlt":
        return RMLT(*common, args.states)
    if args.arm == "mif":
        return MIF(*common)
    if args.arm == "crc":
        return CRC(*common)
    if args.arm == "fnt_v":
        return FNTV(*common)
    if args.arm == "fnt_k":
        return FNTK(*common)
    if args.arm == "cint":
        return CINT(*common)
    if args.arm == "dcmf":
        return MeanFlow(*common)
    if args.arm == "argmax":
        return ArgmaxCoupling(*common)
    if args.arm == "source_oracle":
        return SourceOracle(*common)
    if args.arm in ("scan_flow", "coupling_flow"):
        from nanochat.categorical_flow import CategoricalFlow
        return CategoricalFlow(hmm.S, args.T, args.vocab, width=args.width, depth=args.flow_cond_depth,
                               layers=args.flow_layers, scan=args.arm == "scan_flow", iwae_k=args.iwae_k,
                               bins=args.flow_bins)
    if args.arm in BRIDGE_ARMS:
        from nanochat.bridge import BridgeLM
        codes = {"bridge": "oracle", "bridge_tok": "tokens", "bridge_ar": "ar", "bridge_pred": "ar_pred"}[args.arm]
        return BridgeLM(hmm.S, args.T, args.vocab, width=args.width, depth=args.depth, codes=codes, hmm=hmm,
                        K=args.bridge_codes, ar_steps=args.bridge_ar_steps, endpoints=args.bridge_endpoints,
                        poe=args.bridge_poe, window=args.bridge_window)
    if args.arm in PTP_ARMS:
        from nanochat.ptp import RCPTP, semantic_rank
        inversion = "seq" if args.arm.endswith("_seq") else "jacobi" if args.arm.endswith("_jac") else "ar"
        return RCPTP(hmm.S, args.T, args.vocab, width=args.width, depth=args.depth,
                     rank=None if args.arm == "rcptp_idorder" else semantic_rank(hmm.E),
                     cut=not args.arm.startswith("cptp"), coupled=args.arm != "rcptp_indep",
                     is_samples=args.iwae_k, inversion=inversion, n_inv=args.n_inv, coupling=args.ptp_coupling,
                     stages=None if args.arm.startswith("cptp") else args.ptp_stages)
    raise ValueError(args.arm)


@torch.no_grad()
def independent_control(hmm, alpha, y):
    true_lp = true_block_logprob(hmm, alpha, y)
    marginal_lp = true_marginal_logprob(hmm, alpha, y)
    probs = []
    m = alpha
    for _ in range(y.size(1)):
        m = m @ hmm.A
        probs.append(m @ hmm.E)
    probs = torch.stack(probs, 1)
    samp = categorical_from_uniform(probs, torch.rand(probs.shape[:-1], device=alpha.device))
    slp = true_block_logprob(hmm, alpha, samp)
    return {
        "true_entropy": float((-true_lp).mean()),
        "ideal_indep_tc": float((true_lp - marginal_lp).mean()),
        "ideal_indep_invalid": float((~torch.isfinite(slp)).float().mean()),
    }


@torch.no_grad()
def evaluate(model, hmm, eval_data, args):
    model.eval()
    alpha, y = eval_data
    true_lp = true_block_logprob(hmm, alpha, y)
    row = independent_control(hmm, alpha, y)
    if isinstance(model, SourceOracle):
        all_lp = model.all_log_probs(alpha, y)
        kls = (true_lp[:, None] - all_lp).mean(0)
        per = {kind: float(kls[k]) for k, kind in enumerate(TRANSFORMS)}
        best = int(kls.argmin())
        model.best_kind = TRANSFORMS[best]
        row["transform_kl"] = per
        row["best_transform"] = model.best_kind
        row["block_kl"] = float(kls[best])
    elif getattr(model, "exact", False):
        lp = model.log_prob(alpha, y)
        row["block_kl"] = float((true_lp - lp).mean())
    else:
        row["block_kl"] = None
    if hasattr(model, "diagnostics"):
        row.update(model.diagnostics(alpha, y, true_lp))

    M = args.samples_per_ctx
    aa = alpha[:, None].expand(-1, M, -1).reshape(-1, alpha.size(-1))
    samp = model.sample(aa).view(alpha.size(0), M, args.T)
    flat_a, flat_y = aa, samp.reshape(-1, args.T)
    slp = true_block_logprob(hmm, flat_a, flat_y)
    valid = torch.isfinite(slp)
    row["invalid_rate"] = float((~valid).float().mean())
    row["sample_nll_valid"] = float((-slp[valid]).mean()) if valid.any() else None
    unique = []
    for blocks in samp:
        unique.append(blocks.unique(dim=0).size(0) / M)
    row["unique_rate"] = sum(unique) / len(unique)
    row["sensitivity"] = float((samp[:, 1:] != samp[:, :1]).float().mean()) if M > 1 else None
    model.train()
    return row


def verdict(arm, row):
    if arm in EXACT_ARMS:
        if row["block_kl"] <= 0.10 and row["invalid_rate"] <= 0.03:
            return "PASS"
        if row["block_kl"] <= 0.50 and row["invalid_rate"] <= 0.10:
            return "BORDERLINE"
        return "KILL_THIS_INSTANTIATION"
    if row["invalid_rate"] <= 0.10 and row["unique_rate"] >= 0.25 and row["sensitivity"] > 0.01:
        return "PASS_IMPLICIT"
    return "KILL_THIS_INSTANTIATION"


def run(args):
    torch.manual_seed(args.seed)
    if args.device == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.set_float32_matmul_precision("high")
    device = torch.device(args.device)
    hmm = build_phrase_hmm(V=args.vocab, seed=args.hmm_seed, device=device)
    gen = torch.Generator(device=device).manual_seed(70_000 + args.seed)
    train_pool = BeliefPool(hmm, args.context, args.T, args.pool, gen)
    eval_gen = torch.Generator(device=device).manual_seed(80_000 + args.seed)
    eval_pool = BeliefPool(hmm, args.context, args.T,
                           max(args.eval_contexts, args.pool), eval_gen)
    eval_data = eval_pool.batch(args.eval_contexts)
    model = make_model(args, hmm).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
    nparams = sum(p.numel() for p in model.parameters())
    t0 = time.time()
    milestones = {max(1, args.steps // 2), args.steps}
    curves = []
    last_aux = {}
    for step in range(1, args.steps + 1):
        alpha, y = train_pool.batch(args.batch)
        if isinstance(model, (FNTV, FNTK, CINT, MeanFlow)) or getattr(model, "bound", False):
            loss, last_aux = model.loss(alpha, y)
        elif isinstance(model, SourceOracle):
            loss = -model.all_log_probs(alpha, y).mean()
        else:
            loss = -model.log_prob(alpha, y).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step in milestones:
            ev = evaluate(model, hmm, eval_data, args)
            curves.append({"step": step, "loss": float(loss.detach()), **last_aux, **ev})
            print("S06Q_MILESTONE " + json.dumps({"arm": args.arm, **curves[-1]}, sort_keys=True),
                  flush=True)

    final = curves[-1]
    row = {
        "arm": args.arm,
        "seed": args.seed,
        "T": args.T,
        "depth": args.depth,
        "width": args.width,
        "steps": args.steps,
        "parameters": nparams,
        "training_multiplier": getattr(model, "training_multiplier", 1),
        "stochastic_rounds": 1,
        "oracle_context": True,
        "kl_is_upper_bound": bool(getattr(model, "bound", False)),
        "seconds": round(time.time() - t0, 2),
        "curve": curves,
        **{k: v for k, v in final.items() if k not in ("step", "loss")},
    }
    row["verdict"] = verdict(args.arm, row)
    print("S06Q_RESULT " + json.dumps(row, sort_keys=True), flush=True)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(row, f, indent=2, sort_keys=True)
    return row


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", choices=ARMS, required=True)
    p.add_argument("--steps", type=int, default=2500)
    p.add_argument("--batch", type=int, default=96)
    p.add_argument("--eval-contexts", type=int, default=256)
    p.add_argument("--samples-per-ctx", type=int, default=8)
    p.add_argument("--pool", type=int, default=2048)
    p.add_argument("--context", type=int, default=16)
    p.add_argument("--T", type=int, default=4)
    p.add_argument("--vocab", type=int, default=512)
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--states", type=int, default=64)
    p.add_argument("--flow-layers", type=int, default=8, help="scan_flow / coupling_flow: coupling layers")
    p.add_argument("--flow-cond-depth", type=int, default=2, help="scan_flow / coupling_flow: conditioner depth")
    p.add_argument("--iwae-k", type=int, default=128,
                   help="scan_flow / coupling_flow / PTP arms: importance samples per row for the eval bound")
    p.add_argument("--flow-bins", type=int, default=0, help="scan_flow / coupling_flow: rational-quadratic spline bins after the scan (0 = affine)")
    p.add_argument("--n-inv", type=int, default=2, help="*_jac PTP arms: parallel self-inversion sweeps")
    p.add_argument("--bridge-codes", type=int, default=512, help="bridge_ar: k-means codes")
    p.add_argument("--bridge-ar-steps", type=int, default=4000, help="bridge_ar: stage-1 AR training steps")
    p.add_argument("--bridge-endpoints", action="store_true",
                   help="bridge arms: each midpoint reads its interval's two end codes (Markov-bridge bias)")
    p.add_argument("--bridge-window", type=int, default=1,
                   help="bridge_tok: place n-token windows at bisection positions (window bisection)")
    p.add_argument("--bridge-poe", action="store_true",
                   help="bridge arms: add per-level left-end and right-end logit tables (exact Markov-bridge form)")
    p.add_argument("--ptp-stages", type=int, default=2,
                   help="rcptp* arms: pick stages in the generator (2 = one cut; depth must be >= stages)")
    p.add_argument("--ptp-coupling", choices=("cdf", "tree"), default="cdf",
                   help="PTP arms: flat inverse-CDF pick (PTP's) or one uniform per level of a vocabulary tree")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--hmm-seed", type=int, default=1234)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default="")
    p.add_argument("--smoke", action="store_true")
    return p


def main():
    args = build_parser().parse_args()
    if args.smoke:
        args.steps = min(args.steps, 2)
        args.batch = min(args.batch, 4)
        args.eval_contexts = min(args.eval_contexts, 4)
        args.samples_per_ctx = min(args.samples_per_ctx, 2)
        args.pool = min(args.pool, 16)
        args.width = min(args.width, 32)
        args.depth = min(args.depth, 2 if args.arm in PTP_ARMS else 1)   # the cut needs two layers
        args.states = min(args.states, 4)
    run(args)


if __name__ == "__main__":
    main()
