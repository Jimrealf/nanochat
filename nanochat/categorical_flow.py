"""
Scan-coupled categorical flow (S07 hypothesis A) and its plain-coupling control.

A block of T tokens over V = 2**b is a sign pattern on z in R^{T x b}: bit i of token t is
[z[t, i] > 0]. One-pass generation draws eps ~ N(0, I) once, maps it through a fixed stack of
invertible coupling layers, and reads every token at once from sign(z). There is no token-by-
token neural loop.

Coupling layer (bits split into a conditioning half A and a transformed half B; the split and the
scan direction alternate across layers). A conditioner reads x_A at every position plus the
context and emits per-position (a, b, c) for the B channels:

    generative   v_t = a_t * v_{t-1} + b_t * x_t + c_t          (a scan along the block)
    normalizing  x_t = (v_t - a_t * v_{t-1} - c_t) / b_t        (parallel given v)

with |a| <= 0.95 and b in [e^-2, e^2], so the Jacobian is triangular with log|det| = sum log b. The control is
the same model with a = 0 (an ordinary affine coupling). Training only needs the normalizing
direction, which is parallel; sampling runs the scan.

Likelihood of tokens: P(y) = integral over y's sign cell of p_Z. Training maximises the ELBO
with a dequantiser q(z | y, ctx) that lives inside the cell (z = sign * softplus(u), u Gaussian);
evaluation reports an IWAE-K lower bound on log P(y), so a block KL computed from it is an upper
bound on the true KL.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def token_bits(y, b):
    """(..., T) ids -> (..., T, b) signs in {-1, +1}."""
    bits = (y[..., None] >> torch.arange(b, device=y.device)) & 1
    return bits.float() * 2 - 1


def bits_token(z):
    """(..., T, b) reals -> (..., T) ids from the sign pattern."""
    b = z.size(-1)
    return ((z > 0).long() << torch.arange(b, device=z.device)).sum(-1)


def rq_spline(x, uw, uh, ud, inverse=False, bound=5.0, min_w=1e-2, min_h=1e-2, min_d=1e-2, soft=3.0):
    """Elementwise monotone rational-quadratic spline (Durkan et al. 2019) on [-bound, bound],
    identity outside. x (...,); uw, uh (..., K) and ud (..., K-1) unnormalised. Returns (y, log
    |dy/dx|) for the forward map, or (x, log |dx/dy|) with inverse=True.

    The unnormalised parameters are soft-bounded to +-soft and the spline is evaluated in float64:
    in float32, skewed bins make the quadratic-formula inverse lose up to O(1) (measured 2026-10-03)."""
    dt = x.dtype
    x, uw, uh, ud = (t.double() for t in (x, uw, uh, ud))
    uw, uh, ud = (soft * torch.tanh(t / soft) for t in (uw, uh, ud))
    y, logd = _rq_spline64(x, uw, uh, ud, inverse, bound, min_w, min_h, min_d)
    return y.to(dt), logd.to(dt)


def _rq_spline64(x, uw, uh, ud, inverse, bound, min_w, min_h, min_d):
    K = uw.size(-1)

    def edges(u, m):
        p = m + (1 - m * K) * F.softmax(u, dim=-1)
        e = F.pad(torch.cumsum(p, dim=-1), (1, 0)) * 2 * bound - bound
        e = torch.cat([torch.full_like(e[..., :1], -bound), e[..., 1:-1], torch.full_like(e[..., :1], bound)], -1)
        return e, e[..., 1:] - e[..., :-1]

    cw, w = edges(uw, min_w)
    ch, h = edges(uh, min_h)
    one = torch.ones_like(ud[..., :1])
    d = torch.cat([one, min_d + F.softplus(ud), one], dim=-1)               # unit slope at both ends
    inside = (x > -bound) & (x < bound)
    xc = x.clamp(-bound, bound)
    loc = ch if inverse else cw
    idx = (xc[..., None] >= loc[..., 1:-1]).sum(-1, keepdim=True)          # bin index in [0, K-1]
    g = lambda t: t.gather(-1, idx).squeeze(-1)
    x_k, w_k, y_k, h_k = g(cw), g(w), g(ch), g(h)
    d_k, d_k1 = g(d[..., :-1]), g(d[..., 1:])
    s = h_k / w_k
    if not inverse:
        xi = (xc - x_k) / w_k
        den = s + (d_k1 + d_k - 2 * s) * xi * (1 - xi)
        y = y_k + h_k * (s * xi ** 2 + d_k * xi * (1 - xi)) / den
    else:
        r = xc - y_k
        a = h_k * (s - d_k) + r * (d_k1 + d_k - 2 * s)
        b = h_k * d_k - r * (d_k1 + d_k - 2 * s)
        c = -s * r
        xi = (2 * c / (-b - torch.sqrt((b ** 2 - 4 * a * c).clamp_min(0)))).clamp(0, 1)
        den = s + (d_k1 + d_k - 2 * s) * xi * (1 - xi)
        y = xi * w_k + x_k
    logd = torch.log(s ** 2 * (d_k1 * xi ** 2 + 2 * s * xi * (1 - xi) + d_k * (1 - xi) ** 2)) - 2 * torch.log(den)
    if inverse:
        logd = -logd
    return torch.where(inside, y, x), torch.where(inside, logd, torch.zeros_like(logd))


class Conditioner(nn.Module):
    """Per-position coefficients for the B channels from x_A at every position and the context."""

    def __init__(self, n_in, n_out, ctx_dim, T, width, depth):
        super().__init__()
        self.inp = nn.Linear(n_in, width)
        self.ctx = nn.Linear(ctx_dim, width)
        self.pos = nn.Parameter(torch.randn(T, width) / math.sqrt(width))
        layer = nn.TransformerEncoderLayer(width, max(1, width // 32), 4 * width, dropout=0.0,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.mix = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.out = nn.Linear(width, n_out)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, xa, ctx):
        h = self.inp(xa) + self.ctx(ctx)[:, None] + self.pos[None, :xa.size(1)]
        return self.out(self.mix(h))


class ScanCoupling(nn.Module):
    """One coupling layer over (B, T, b): channels `idx_b` transformed, `idx_a` conditioning."""

    def __init__(self, idx_a, idx_b, ctx_dim, T, width, depth, reverse, scan=True, bins=0):
        super().__init__()
        self.register_buffer("idx_a", torch.tensor(idx_a, dtype=torch.long), persistent=False)
        self.register_buffer("idx_b", torch.tensor(idx_b, dtype=torch.long), persistent=False)
        self.reverse, self.scan, self.bins = reverse, scan, bins
        nb = len(idx_b)
        n_spline = (3 * bins - 1) if bins > 0 else 0
        self.cond = Conditioner(len(idx_a), (3 + n_spline) * nb, ctx_dim, T, width, depth)

    def spline_params(self, raw):
        """(B, T, nb, 3K-1) unnormalised spline widths, heights, inner slopes for each B channel."""
        K = self.bins
        sp = raw[..., 3 * len(self.idx_b):].view(*raw.shape[:2], len(self.idx_b), 3 * K - 1)
        return sp[..., :K], sp[..., K:2 * K], sp[..., 2 * K:]

    def coeffs(self, xa, ctx):
        raw = self.cond(xa, ctx)
        self._raw = raw
        ra, rb, c = raw[..., :3 * len(self.idx_b)].chunk(3, dim=-1)
        # Bounded so the parallel inverse stays well conditioned: |a| <= 0.95, b in [e^-2, e^2]
        # (b = 1 at the zero init).
        a = 0.95 * torch.tanh(ra) if self.scan else torch.zeros_like(ra)
        b = torch.exp(2.0 * torch.tanh(rb / 2.0))
        return a, b, c

    def _flip(self, t):
        return t.flip(1) if self.reverse else t

    def normalize(self, z, ctx):
        """z -> x (parallel) and log|det dx/dz| per row. With splines, z_t = spline_t(v_t) is
        inverted first (elementwise), then the affine scan (parallel given v)."""
        xa = z.index_select(-1, self.idx_a)
        a, b, c = self.coeffs(xa, ctx)
        zb = z.index_select(-1, self.idx_b)
        logdet = torch.zeros(z.size(0), device=z.device, dtype=z.dtype)
        if self.bins > 0:
            uw, uh, ud = self.spline_params(self._raw)
            zb, ld = rq_spline(zb, uw, uh, ud, inverse=True)
            logdet = logdet + ld.sum((1, 2))
        a, b, c = (self._flip(t) for t in (a, b, c))
        v = self._flip(zb)
        v_prev = F.pad(v[:, :-1], (0, 0, 1, 0))
        xb = self._flip((v - a * v_prev - c) / b)
        x = z.clone()
        x[..., self.idx_b] = xb
        return x, logdet - torch.log(b).sum((1, 2))

    def generate(self, x, ctx):
        """x -> z (the scan along the block, then the elementwise spline)."""
        xa = x.index_select(-1, self.idx_a)
        a, b, c = (self._flip(t) for t in self.coeffs(xa, ctx))
        xb = self._flip(x.index_select(-1, self.idx_b))
        v, prev = [], torch.zeros_like(xb[:, 0])
        for t in range(xb.size(1)):
            prev = a[:, t] * prev + b[:, t] * xb[:, t] + c[:, t]
            v.append(prev)
        vb = self._flip(torch.stack(v, 1))
        if self.bins > 0:
            uw, uh, ud = self.spline_params(self._raw)
            vb, _ = rq_spline(vb, uw, uh, ud)
        z = x.clone()
        z[..., self.idx_b] = vb
        return z


class Dequantizer(nn.Module):
    """q(z | y, ctx) inside y's sign cell: z = sign * softplus(mu + sigma * eta)."""

    def __init__(self, V, b, ctx_dim, T, width, depth):
        super().__init__()
        self.emb = nn.Embedding(V, width)
        self.ctx = nn.Linear(ctx_dim, width)
        self.pos = nn.Parameter(torch.randn(T, width) / math.sqrt(width))
        layer = nn.TransformerEncoderLayer(width, max(1, width // 32), 4 * width, dropout=0.0,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.mix = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.out = nn.Linear(width, 2 * b)

    def sample(self, y, ctx, K, b):
        """K draws per row: z (K, B, T, b) and log q(z) (K, B)."""
        h = self.mix(self.emb(y) + self.ctx(ctx)[:, None] + self.pos[None, :y.size(1)])
        mu, log_sigma = self.out(h).chunk(2, dim=-1)
        sigma = F.softplus(log_sigma) + 1e-3
        eta = torch.randn((K,) + mu.shape, device=mu.device, dtype=mu.dtype)
        u = mu + sigma * eta
        s = token_bits(y, b)
        z = s * F.softplus(u)
        logq = (-0.5 * eta ** 2 - torch.log(sigma) - 0.5 * math.log(2 * math.pi)
                - F.logsigmoid(u)).sum((-1, -2))                      # change of variables u -> |z|
        return z, logq


class CategoricalFlow(nn.Module):
    """T tokens over V = 2**b in one pass: eps -> coupling stack -> sign -> ids."""
    exact = True              # log_prob is an IWAE lower bound, so block KL from it is an upper bound
    bound = True
    training_multiplier = 1

    def __init__(self, ctx_dim, T, V, width=128, depth=2, layers=8, scan=True, iwae_k=128, bins=0):
        super().__init__()
        b = int(round(math.log2(V)))
        assert 2 ** b == V, "binary sign cells need a power-of-two vocabulary"
        self.T, self.V, self.b, self.iwae_k = T, V, b, iwae_k
        half = b // 2
        lo, hi = list(range(half)), list(range(half, b))
        self.layers = nn.ModuleList(
            ScanCoupling(lo if i % 2 == 0 else hi, hi if i % 2 == 0 else lo, ctx_dim, T, width, depth,
                         reverse=(i // 2) % 2 == 1, scan=scan, bins=bins)
            for i in range(layers))
        self.deq = Dequantizer(V, b, ctx_dim, T, width, depth)

    def log_density(self, z, ctx):
        """log p_Z(z | ctx) for z (N, T, b)."""
        x, logdet = z, torch.zeros(z.size(0), device=z.device)
        for layer in reversed(self.layers):
            x, ld = layer.normalize(x, ctx)
            logdet = logdet + ld
        return (-0.5 * x ** 2 - 0.5 * math.log(2 * math.pi)).sum((1, 2)) + logdet

    def _bound(self, ctx, y, K):
        z, logq = self.deq.sample(y, ctx, K, self.b)                      # (K, B, T, b), (K, B)
        Bn = y.size(0)
        logp = self.log_density(z.reshape(K * Bn, self.T, self.b),
                                ctx.repeat(K, 1)).view(K, Bn)
        return logp - logq                                                 # (K, B) log weights

    def loss(self, ctx, y):
        w = self._bound(ctx, y, 1)
        elbo = w.mean(0)
        return -elbo.mean(), {"elbo": float(elbo.mean().detach())}

    @torch.no_grad()
    def log_prob(self, ctx, y, K=None):
        """IWAE-K lower bound on log P(y | ctx), (B,)."""
        K = K or self.iwae_k
        out = []
        for i in range(0, y.size(0), 64):
            w = self._bound(ctx[i:i + 64], y[i:i + 64], K)
            out.append(torch.logsumexp(w, 0) - math.log(K))
        return torch.cat(out)

    @torch.no_grad()
    def sample(self, ctx):
        x = torch.randn(ctx.size(0), self.T, self.b, device=ctx.device, dtype=ctx.dtype)
        for layer in self.layers:
            x = layer.generate(x, ctx)
        return bits_token(x)
