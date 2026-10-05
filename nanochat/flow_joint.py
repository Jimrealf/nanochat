"""One-tape continuous plan + normalized local discrete joint (toy gate).

No future token is an inference input. The recognition module sees targets only
to form a variational posterior during training/likelihood-bound evaluation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


def categorical(probs, uniform):
    # >= skips zero-mass intervals, including a possible uniform exactly zero.
    return (uniform[..., None] >= probs.cumsum(-1)).sum(-1).clamp_max(probs.size(-1) - 1)


def scan_states(initial, maps):
    """Resolve pre-sampled random functions. No RNG or neural calls in the scan."""
    prefix = maps
    offset = 1
    while offset < maps.size(1):
        prefix = torch.cat((prefix[:, :offset],
                            prefix[:, offset:].gather(-1, prefix[:, :-offset])), 1)
        offset *= 2
    rest = prefix.gather(-1, initial[:, None, None].expand(-1, maps.size(1), 1)).squeeze(-1)
    return torch.cat((initial[:, None], rest), 1)


class SlotMixer(nn.Module):
    def __init__(self, context_dim, T, width, depth, input_dim):
        super().__init__()
        self.context = nn.Sequential(nn.Linear(context_dim, width), nn.SiLU(), nn.Linear(width, width))
        self.input = nn.Linear(input_dim, width, bias=False) if input_dim else None
        self.pos = nn.Parameter(torch.randn(T, width) / math.sqrt(width))
        layer = nn.TransformerEncoderLayer(width, max(1, width // 32), 4 * width,
                                           dropout=0., activation="gelu", batch_first=True,
                                           norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)

    def forward(self, context, inputs=None):
        h = self.context(context)[:, None] + self.pos[None]
        if inputs is not None:
            h = h + self.input(inputs)
        return self.norm(self.blocks(h))


class AffineCoupling(nn.Module):
    def __init__(self, context_dim, T, width, latent, swap):
        super().__init__()
        self.swap = swap
        self.conditioner = SlotMixer(context_dim, T, width, 1, latent // 2)
        self.output = nn.Linear(width, latent)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x, context, inverse=False):
        left, right = x.chunk(2, -1)
        if self.swap:
            left, right = right, left
        shift, raw_scale = self.output(self.conditioner(context, left)).chunk(2, -1)
        log_scale = 2 * torch.tanh(raw_scale / 2)
        right = (right - shift) * (-log_scale).exp() if inverse else right * log_scale.exp() + shift
        value = torch.cat((right, left) if self.swap else (left, right), -1)
        logdet = log_scale.sum((1, 2)) * (-1 if inverse else 1)
        return value, logdet


@dataclass
class NoiseTape:
    epsilon: torch.Tensor
    initial: torch.Tensor
    maps: torch.Tensor
    tokens: torch.Tensor


class FlowJoint(nn.Module):
    def __init__(self, context_dim, T=4, vocab=512, width=128, states=64,
                 latent=8, arm="hybrid"):
        super().__init__()
        if arm not in {"hybrid", "flow_only", "chain_only"}:
            raise ValueError(arm)
        if latent % 2 or latent < 2 or T < 1:
            raise ValueError("even latent dimension >=2 and T>=1 required")
        self.arm, self.T, self.V, self.latent = arm, T, vocab, latent
        self.S = 1 if arm == "flow_only" else states
        self.has_flow = arm != "chain_only"
        self.decoder = SlotMixer(context_dim, T, width, 2, latent if self.has_flow else 0)
        self.lexical = nn.Linear(width, vocab)
        if self.S > 1:
            self.initial = nn.Linear(width, self.S)
            self.transition = nn.Linear(width, self.S * self.S)
            # Break state symmetry without using generator structure.
            self.emission_logits = nn.Parameter(torch.randn(self.S, vocab) * .2)
        if self.has_flow:
            self.flow = nn.ModuleList([AffineCoupling(context_dim, T, width, latent, swap)
                                       for swap in (False, True)])
            self.embedding = nn.Embedding(vocab, width)
            self.recognition = SlotMixer(context_dim, T, width, 2, width)
            self.posterior_head = nn.Linear(width, 2 * latent)
            nn.init.zeros_(self.posterior_head.weight)
            nn.init.zeros_(self.posterior_head.bias)

    @staticmethod
    def normal_logprob(x):
        return (-.5 * (x.square() + math.log(2 * math.pi))).sum((-1, -2))

    def transform(self, value, context, inverse=False):
        logdet = value.new_zeros(value.size(0))
        layers = reversed(self.flow) if inverse else self.flow
        for layer in layers:
            value, ld = layer(value, context, inverse=inverse)
            logdet = logdet + ld
        return value, logdet

    def posterior(self, context, targets, K=1):
        mean, raw_scale = self.posterior_head(self.recognition(context, self.embedding(targets))).chunk(2, -1)
        log_scale = 2 * torch.tanh(raw_scale / 2)
        epsilon = torch.randn((K,) + mean.shape, device=mean.device, dtype=mean.dtype)
        z = mean[None] + log_scale.exp()[None] * epsilon
        logq = self.normal_logprob(epsilon) - log_scale.sum((1, 2))[None]
        return z.flatten(0, 1), logq.flatten()

    def fields(self, context, z=None):
        h = self.decoder(context, z)
        logq = self.lexical(h).log_softmax(-1)
        if self.S == 1:
            return logq, logq.new_zeros(context.size(0), 1), logq.new_zeros(context.size(0), self.T - 1, 1, 1)
        logpi = self.initial(h[:, 0]).log_softmax(-1)
        logA = self.transition(h[:, 1:]).reshape(context.size(0), self.T - 1, self.S, self.S).log_softmax(-1)
        return logq, logpi, logA

    def target_emissions(self, logq, targets):
        base = logq.gather(-1, targets[..., None])
        if self.S == 1:
            return base
        # Separate row scaling leaves E unchanged and keeps exp(R) bounded.
        logR = self.emission_logits - self.emission_logits.amax(-1, keepdim=True)
        mass = logq.exp() @ logR.exp().T
        logZ = mass.clamp_min(torch.finfo(mass.dtype).tiny).log()
        selected = logR[:, targets].permute(1, 2, 0)
        return base + selected - logZ

    def conditional_logprob(self, context, targets, z=None):
        logq, logpi, logA = self.fields(context, z)
        emission = self.target_emissions(logq, targets)
        forward = logpi + emission[:, 0]
        for t in range(1, self.T):
            forward = torch.logsumexp(forward[:, :, None] + logA[:, t - 1], 1) + emission[:, t]
        return torch.logsumexp(forward, -1)

    def terms(self, context, targets, K=1):
        if not self.has_flow:
            ll = self.conditional_logprob(context, targets)
            return ll[None], torch.zeros_like(ll)[None]
        z, logq = self.posterior(context, targets, K)
        ctx = context.repeat(K, 1)
        epsilon, inv_logdet = self.transform(z, ctx, inverse=True)
        logp = self.normal_logprob(epsilon) + inv_logdet
        ll = self.conditional_logprob(ctx, targets.repeat(K, 1), z)
        return ll.view(K, -1), (logq - logp).view(K, -1)

    def loss(self, context, targets, beta=1.):
        ll, kl = self.terms(context, targets)
        return (-ll + beta * kl).mean()

    def noise_tape(self, context):
        B, dev, dtype = context.size(0), context.device, context.dtype
        return NoiseTape(torch.randn(B, self.T, self.latent, device=dev, dtype=dtype),
                         torch.rand(B, device=dev, dtype=dtype),
                         torch.rand(B, self.T - 1, self.S, device=dev, dtype=dtype),
                         torch.rand(B, self.T, device=dev, dtype=dtype))

    @torch.no_grad()
    def sample(self, context, tape=None, z_override=None):
        tape = self.noise_tape(context) if tape is None else tape
        z = z_override
        if self.has_flow and z is None:
            z, _ = self.transform(tape.epsilon, context)
        logq, logpi, logA = self.fields(context, z)
        if self.S > 1:
            initial = categorical(logpi.exp(), tape.initial)
            maps = categorical(logA.exp(), tape.maps)
            states = scan_states(initial, maps)
            # Only B*T*V, never B*T*S*V.
            probs = (logq + self.emission_logits[states]).softmax(-1)
        else:
            probs = logq.exp()
        return categorical(probs, tape.tokens)
