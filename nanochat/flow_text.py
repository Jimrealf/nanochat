"""Fixed-depth prompt-conditioned continuous flow with parallel lexical decoding."""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from nanochat.flow_joint import AffineCoupling, FlowJoint, categorical


@dataclass
class FlowTextConfig:
    vocab: int = 32768
    T: int = 2048
    prompt: int = 128
    width: int = 256
    latent: int = 8
    heads: int = 8
    projection_chunk: int = 128
    conditioner_depth: int = 1
    decoder_depth: int = 2
    recognition_depth: int = 2

    @property
    def generative_depth(self):
        return 2 * self.conditioner_depth + self.decoder_depth


class PromptMixer(nn.Module):
    def __init__(self, config, input_dim, depth):
        super().__init__()
        self.C, self.T = config.prompt, config.T
        self.input = nn.Linear(input_dim, config.width, bias=False)
        self.pos = nn.Parameter(torch.randn(config.prompt + config.T, config.width) / math.sqrt(config.width))
        self.blocks = nn.ModuleList([nn.TransformerEncoderLayer(
            config.width, config.heads, 4 * config.width, dropout=0., activation="gelu",
            batch_first=True, norm_first=True) for _ in range(depth)])
        self.norm = nn.LayerNorm(config.width)

    def forward(self, prompt_embeddings, inputs):
        h = torch.cat((prompt_embeddings, self.input(inputs)), 1) + self.pos[None]
        for block in self.blocks:
            h = block(h)
        return self.norm(h[:, self.C:])


class TextCoupling(AffineCoupling):
    """Reuse the tested affine bijection; replace oracle-context conditioning."""
    def __init__(self, config, swap):
        nn.Module.__init__(self)
        self.swap = swap
        self.conditioner = PromptMixer(config, config.latent // 2, config.conditioner_depth)
        self.output = nn.Linear(config.width, config.latent)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x, context, inverse=False):
        left, right = x.chunk(2, -1)
        if self.swap:
            left, right = right, left
        raw = self.output(self.conditioner(context, left))
        if raw.dtype in (torch.float16, torch.bfloat16):
            raw = raw.float()
        shift, scale = raw.chunk(2, -1)
        log_scale = 2 * torch.tanh(scale / 2)
        right = (right - shift) * (-log_scale).exp() if inverse else right * log_scale.exp() + shift
        value = torch.cat((right, left) if self.swap else (left, right), -1)
        return value, log_scale.sum((1, 2)) * (-1 if inverse else 1)


class FlowText(nn.Module):
    def __init__(self, config: FlowTextConfig):
        super().__init__()
        if config.latent < 2 or config.latent % 2 or config.width % config.heads:
            raise ValueError("even latent dimension >=2 and width divisible by heads required")
        if min(config.T, config.prompt, config.projection_chunk) < 1:
            raise ValueError("positive T, prompt and projection chunk required")
        if min(config.conditioner_depth, config.decoder_depth, config.recognition_depth) < 1:
            raise ValueError("positive conditioner, decoder and recognition depths required")
        self.config = config
        self.embedding = nn.Embedding(config.vocab, config.width)
        self.flow = nn.ModuleList([TextCoupling(config, swap) for swap in (False, True)])
        self.decoder = PromptMixer(config, config.latent, config.decoder_depth)
        self.readout = nn.Linear(config.width, config.vocab)
        self.recognition = PromptMixer(config, config.width, config.recognition_depth)
        self.posterior_head = nn.Linear(config.width, 2 * config.latent)
        nn.init.zeros_(self.posterior_head.weight)
        nn.init.zeros_(self.posterior_head.bias)

    normal_logprob = staticmethod(FlowJoint.normal_logprob)

    def transform(self, value, prompt_embeddings, inverse=False):
        # Keep density state/log-Jacobians in fp32 under bf16 neural autocast.
        logdet = value.new_zeros(value.size(0))
        for layer in reversed(self.flow) if inverse else self.flow:
            value, ld = layer(value, prompt_embeddings, inverse=inverse)
            logdet = logdet + ld
        return value, logdet

    def posterior(self, prompt_embeddings, targets, epsilon=None):
        mean, scale = self.posterior_head(self.recognition(prompt_embeddings, self.embedding(targets))).chunk(2, -1)
        if mean.dtype in (torch.float16, torch.bfloat16):
            mean, scale = mean.float(), scale.float()
        log_scale = 2 * torch.tanh(scale / 2)
        epsilon = torch.randn_like(mean) if epsilon is None else epsilon
        z = mean + log_scale.exp() * epsilon
        return z, self.normal_logprob(epsilon) - log_scale.sum((1, 2))

    def _token_nll(self, hidden, targets):
        logits = self.readout(hidden)
        if logits.dtype in (torch.float16, torch.bfloat16):
            logits = logits.float()
        return F.cross_entropy(logits.flatten(0, 1), targets.flatten(), reduction="none").view_as(targets)

    def conditional_logprob(self, hidden, targets):
        loss = hidden.new_zeros(hidden.size(0))
        for start in range(0, self.config.T, self.config.projection_chunk):
            h = hidden[:, start:start + self.config.projection_chunk]
            y = targets[:, start:start + self.config.projection_chunk]
            token_nll = (checkpoint(self._token_nll, h, y, use_reentrant=False)
                         if self.training and torch.is_grad_enabled() else self._token_nll(h, y))
            loss = loss + token_nll.sum(1)
        return -loss

    def terms(self, prompts, targets, epsilon=None):
        embedded = self.embedding(prompts)
        z, logq = self.posterior(embedded, targets, epsilon)
        noise, logdet = self.transform(z, embedded, inverse=True)
        logp = self.normal_logprob(noise) + logdet
        hidden = self.decoder(embedded, z)
        return self.conditional_logprob(hidden, targets), logq - logp

    def forward(self, prompts, targets, beta=1., epsilon=None):
        # An explicit tape enables eager/compiled equivalence checks and avoids
        # compiler-specific RNG ordering in the optimized training path.
        ll, kl = self.terms(prompts, targets, epsilon)
        return (-ll + beta * kl).mean() / self.config.T

    def noise_tape(self, prompts):
        dtype = torch.float64 if self.embedding.weight.dtype == torch.float64 else torch.float32
        return (torch.randn(prompts.size(0), self.config.T, self.config.latent, device=prompts.device, dtype=dtype),
                torch.rand(prompts.size(0), self.config.T, device=prompts.device, dtype=dtype))

    @torch.no_grad()
    def sample(self, prompts, tape=None):
        epsilon, uniforms = self.noise_tape(prompts) if tape is None else tape
        embedded = self.embedding(prompts)
        z, _ = self.transform(epsilon, embedded)
        hidden = self.decoder(embedded, z)
        pieces = []
        for start in range(0, self.config.T, self.config.projection_chunk):
            logits = self.readout(hidden[:, start:start + self.config.projection_chunk]).float()
            pieces.append(categorical(logits.softmax(-1), uniforms[:, start:start + self.config.projection_chunk]))
        return torch.cat(pieces, 1)
