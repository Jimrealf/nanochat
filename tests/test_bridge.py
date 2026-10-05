"""Bridge LM (nanochat/bridge.py): bisection levels cover the block in ceil(log2 T) + 1 steps, the
prior at a level sees only coarser codes, the token-code factorisation is a normalised
distribution sampled exactly by the level-by-level sampler, and with oracle codes the
teacher-forced bound never exceeds one."""
import itertools
from types import SimpleNamespace

import pytest
import torch

from nanochat.bridge import BridgeLM, bisection_levels, window_bisection_levels


def _hmm(S=4, V=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    A = torch.rand(S, S, generator=g) ** 3
    E = torch.rand(S, V, generator=g) ** 3
    return SimpleNamespace(A=A / A.sum(1, keepdim=True), E=E / E.sum(1, keepdim=True), S=S, V=V)


def _model(codes="tokens", V=3, T=3, hmm=None, seed=0):
    torch.manual_seed(seed)
    model = BridgeLM(ctx_dim=4, T=T, V=V, width=32, depth=2, emit_depth=1, codes=codes, hmm=hmm)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.3)
    return model.eval()


@pytest.mark.parametrize("T", [1, 2, 3, 4, 7, 64])
def test_bisection_levels_cover_the_block_in_log_steps(T):
    levels, level_of = bisection_levels(T)
    flat = sorted(p for lvl in levels for p in lvl)
    assert flat == list(range(T))
    assert len(levels) == (T - 1).bit_length() + 1                   # ceil(log2 T) + 1
    for l, lvl in enumerate(levels):
        assert (level_of[lvl] == l).all()


@pytest.mark.parametrize("endpoints,poe", [(False, False), (True, False), (False, True)])
def test_prior_sees_only_coarser_codes(endpoints, poe):
    torch.manual_seed(0)
    model = BridgeLM(ctx_dim=4, T=8, V=5, width=32, depth=2, emit_depth=1, codes="tokens", endpoints=endpoints,
                     poe=poe).eval()
    if poe:
        with torch.no_grad():
            model.poe_left.normal_()
            model.poe_right.normal_()
    ctx = torch.randn(6, 4)
    z = torch.randint(0, 5, (6, 8))
    with torch.no_grad():
        for l, pos in enumerate(model.levels):
            z2 = z.clone()
            finer = model.level_of >= l
            z2[:, finer] = torch.randint(0, 5, (6, int(finer.sum())))
            a, b = model.prior_logits(ctx, z, l), model.prior_logits(ctx, z2, l)
            assert torch.allclose(a[:, pos], b[:, pos], atol=1e-5)


def test_token_codes_give_a_normalised_distribution_and_an_exact_sampler():
    model = _model(V=3, T=3)
    ctx = torch.randn(1, 4)
    seqs = torch.tensor(list(itertools.product(range(3), repeat=3)))
    p = model.log_prob(ctx.expand(27, 4), seqs).double().exp()
    assert p.sum().item() == pytest.approx(1.0, abs=1e-5)
    torch.manual_seed(1)
    N = 200_000
    ys = model.sample(ctx.expand(N, 4))
    freq = torch.bincount(ys[:, 0] * 9 + ys[:, 1] * 3 + ys[:, 2], minlength=27).double() / N
    se = (p * (1 - p) / N).sqrt()
    assert ((freq - p).abs() < 5 * se + 1e-9).all()
    assert p.max().item() > 0.08                      # the test has power


def test_oracle_code_bound_never_exceeds_one():
    hmm = _hmm()
    model = _model(codes="oracle", V=3, T=3, hmm=hmm)
    alpha = torch.softmax(torch.randn(1, 4), -1)
    seqs = torch.tensor(list(itertools.product(range(3), repeat=3)))
    p = model.log_prob(alpha.expand(27, 4), seqs).double().exp()
    assert 0.0 < p.sum().item() <= 1.0 + 1e-6


def test_training_loss_reaches_every_module():
    hmm = _hmm()
    model = _model(codes="oracle", V=3, T=4, hmm=hmm).train()
    loss, _ = model.loss(torch.softmax(torch.randn(8, 4), -1), torch.randint(0, 3, (8, 4)))
    loss.backward()
    dead = [n for n, q in model.named_parameters() if q.grad is None or q.grad.abs().sum() == 0]
    # the last level's own code embedding is never read (nothing is finer); everything else learns
    assert all(n.startswith("code_emb") for n in dead)


@pytest.mark.parametrize("codes", ["ar", "ar_pred"])
def test_learned_codes_train_in_two_teacher_forced_stages(codes):
    """Learned codes: the AR model trains alone, is frozen and clustered into K codes at the switch
    (Euclidean on states, or KL on predicted distributions), and the bridge then trains on fixed
    codes; the bound over all sequences stays <= 1."""
    torch.manual_seed(0)
    model = BridgeLM(ctx_dim=4, T=3, V=3, width=32, depth=2, emit_depth=1, codes=codes, K=4, ar_steps=2,
                     kmeans_batches=2)
    ctx = torch.randn(16, 4)
    y = torch.randint(0, 3, (16, 3))
    _, aux1 = model.loss(ctx, y)
    assert "ar_nats" in aux1 and model.codebook.numel() == 0
    _, aux2 = model.loss(ctx, y)
    assert model.codebook.shape == (4, 32 if codes == "ar" else 3)
    assert not any(p.requires_grad for p in model.ar.parameters())
    loss, aux3 = model.loss(ctx, y)
    assert "prior_nats" in aux3
    loss.backward()
    model.eval()
    seqs = torch.tensor(list(itertools.product(range(3), repeat=3)))
    p = model.log_prob(ctx[:1].expand(27, 4), seqs).double().exp()
    assert 0.0 < p.sum().item() <= 1.0 + 1e-6


def test_poe_head_represents_exact_markov_bridges():
    """With the transformer's logits zeroed, the left/right tables alone reproduce the exact bridge
    of a Markov code chain, P(z_m | z_a, z_b) prop. to A^d1[z_a, z_m] A^d2[z_m, z_b]."""
    torch.manual_seed(0)
    K, T = 4, 4
    model = BridgeLM(ctx_dim=2, T=T, V=K, width=16, depth=1, emit_depth=1, codes="tokens", poe=True).eval()
    A = torch.softmax(torch.randn(K, K) * 2, -1)
    with torch.no_grad():
        model.prior_head.weight.zero_()
        model.prior_head.bias.zero_()
        for l, pos in enumerate(model.levels):
            p0 = pos[-1]                                                 # every position of a level shares (d1, d2)
            a, b = model.brackets[p0].tolist()
            d1, d2 = max(p0 - a, 1), max(b - p0, 1)
            left = torch.linalg.matrix_power(A, d1).log()                # rows: z_a (a >= 0)
            right = torch.linalg.matrix_power(A, d2).log().t()           # rows: z_b, columns: z_m
            model.poe_left[l, :K] = left
            model.poe_left[l, K] = 0.0                                   # a = context: uniform start
            model.poe_right[l, :K] = right
    z = torch.randint(0, K, (1, T))
    m = 2                                                                # interval (1, 3): both ends are codes
    lvl = model.level_of[m].item()
    a, b = model.brackets[m].tolist()
    assert a >= 0
    got = model.prior_logits(torch.zeros(1, 2), z, lvl)[0, m].softmax(-1)
    want = torch.linalg.matrix_power(A, m - a)[z[0, a]] * torch.linalg.matrix_power(A, b - m)[:, z[0, b]]
    assert torch.allclose(got, want / want.sum(), atol=1e-5)


@pytest.mark.parametrize("T,n", [(8, 2), (16, 3), (64, 2), (7, 2)])
def test_window_bisection_places_every_position_once_in_few_steps(T, n):
    steps, step_of = window_bisection_levels(T, n)
    assert sorted(p for st in steps for p in st) == list(range(T))
    assert len(steps) <= n * len(bisection_levels(T)[0])
    assert step_of[T - n] == 0 and step_of[T - 1] == n - 1               # the end window comes first, in order


def test_window_bisection_token_model_is_normalised():
    torch.manual_seed(0)
    model = BridgeLM(ctx_dim=4, T=4, V=3, width=32, depth=2, emit_depth=1, codes="tokens", window=2).eval()
    with torch.no_grad():
        for q in model.parameters():
            q.add_(torch.randn_like(q) * 0.3)
    seqs = torch.tensor(list(itertools.product(range(3), repeat=4)))
    p = model.log_prob(torch.randn(1, 4).expand(81, 4), seqs).double().exp()
    assert p.sum().item() == pytest.approx(1.0, abs=1e-5)
