"""Scan-coupled categorical flow (nanochat/categorical_flow.py): the normalizing direction
inverts the generative scan, its log-determinant is the true Jacobian's, and the density it
trains on integrates over each sign cell to the frequencies the one-pass sampler emits."""
import itertools

import pytest
import torch

from nanochat.categorical_flow import CategoricalFlow, bits_token, token_bits


def _flow(T=3, V=4, scan=True, layers=4, seed=0, bins=0):
    torch.manual_seed(seed)
    flow = CategoricalFlow(ctx_dim=5, T=T, V=V, width=32, depth=1, layers=layers, scan=scan, iwae_k=64, bins=bins)
    with torch.no_grad():                                   # conditioner outputs start at zero
        for layer in flow.layers:
            layer.cond.out.weight.normal_(0, 0.2)
            layer.cond.out.bias.normal_(0, 0.2)
    return flow.eval()


def _normalize(flow, z, ctx):
    x, logdet = z, torch.zeros(z.size(0))
    for layer in reversed(flow.layers):
        x, ld = layer.normalize(x, ctx)
        logdet = logdet + ld
    return x, logdet


def test_bits_roundtrip():
    y = torch.arange(16)[None].expand(2, 16)
    assert torch.equal(bits_token(token_bits(y, 4)), y)


@pytest.mark.parametrize("bins", [0, 8])
@pytest.mark.parametrize("scan", [True, False])
def test_generate_inverts_normalize(scan, bins):
    """Exact in float64. (In float32, steep spline slopes amplify rounding across layers to
    about 1e-2, measured 2026-10-03; each layer alone stays near 1e-6.)"""
    dt = torch.float64 if bins else torch.float32
    flow = _flow(scan=scan, bins=bins).to(dt)
    ctx = torch.randn(6, 5, dtype=dt)
    z = torch.randn(6, 3, 2, dtype=dt)
    with torch.no_grad():
        x, _ = _normalize(flow, z, ctx)
        back = x
        for layer in flow.layers:
            back = layer.generate(back, ctx)
    assert torch.allclose(back, z, atol=1e-4)


@pytest.mark.parametrize("bins", [0, 8])
@pytest.mark.parametrize("scan", [True, False])
def test_log_determinant_is_the_jacobian(scan, bins):
    flow = _flow(scan=scan, bins=bins).double()
    ctx = torch.randn(1, 5, dtype=torch.float64)
    z = torch.randn(1, 3, 2, dtype=torch.float64)
    f = lambda v: _normalize(flow, v.view(1, 3, 2), ctx)[0].reshape(-1)
    J = torch.autograd.functional.jacobian(f, z.reshape(-1))
    _, logdet = _normalize(flow, z, ctx)
    assert torch.linalg.slogdet(J)[1].item() == pytest.approx(logdet.item(), abs=1e-8)


@pytest.mark.parametrize("bins", [0, 8])
def test_cell_mass_matches_one_pass_sampler(bins):
    """T=2 one-bit tokens (4 sign cells, a 2-D density): quadrature of the training density over
    each quadrant sums to one and matches the histogram of one-pass samples (eps -> scan stack ->
    sign). A tan-mapped midpoint grid covers each quadrant out to infinity."""
    flow = _flow(T=2, V=2, layers=4, bins=bins)
    ctx = torch.randn(1, 5)
    with torch.no_grad():
        n = 400_000
        ys = flow.sample(ctx.expand(n, 5))
        freq = torch.bincount(ys[:, 0] * 2 + ys[:, 1], minlength=4).double() / n
        N = 1500 if bins else 600            # spline densities are sharper; 600^2 leaves 1% unresolved
        th = (torch.arange(N, dtype=torch.float64) + 0.5) * (torch.pi / 2) / N
        r, jac = torch.tan(th), (1 / torch.cos(th)) ** 2 * (torch.pi / 2) / N
        R0, R1 = torch.meshgrid(r, r, indexing="ij")
        W = (jac[:, None] * jac[None, :]).reshape(-1)
        mass = torch.zeros(4, dtype=torch.float64)
        for c, (y0, y1) in enumerate(itertools.product(range(2), repeat=2)):
            s0, s1 = (1.0 if y0 else -1.0), (1.0 if y1 else -1.0)
            z = torch.stack([s0 * R0.reshape(-1), s1 * R1.reshape(-1)], dim=1)[..., None].float()
            logp = torch.cat([flow.log_density(zc, ctx.expand(zc.size(0), 5))
                              for zc in z.split(65536)]).double()
            mass[c] = (logp.exp() * W).sum()
    assert mass.sum().item() == pytest.approx(1.0, abs=1e-2 if bins else 5e-3)
    assert (mass - freq).abs().max().item() < (6e-3 if bins else 5e-3)
