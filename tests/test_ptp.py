"""RC-PTP (nanochat/ptp.py): picks and their inversion agree exactly for both the flat inverse-CDF
pick and the per-level tree pick, the AR mode's teacher-forced inversion reproduces its own
sequential sampler, the generator sees only earlier auxiliaries, self-inversion regenerates the data
in one pass, and the sequential importance-sampling likelihood is the one-pass sampler's."""
import itertools

import pytest
import torch

from nanochat.ptp import (RCPTP, invert, ordered_cdf_bounds, ordered_pick, place, semantic_rank, tree_bounds,
                          tree_pick)

ARMS = [(2, True), (2, False), (1, True), (3, True)]        # (stages, coupled): rcptp, rcptp_indep,
                                                             # cptp_si, and two coupled refinement steps
COUPLINGS = ["cdf", "tree"]


def _model(V=8, T=5, stages=2, coupled=True, seed=0, coupling="cdf"):
    torch.manual_seed(seed)
    model = RCPTP(ctx_dim=5, T=T, V=V, width=32, depth=3, rank=torch.randperm(V), stages=stages, coupled=coupled,
                  bits=12, is_samples=16, coupling=coupling)
    with torch.no_grad():                                     # peaked, strongly u-dependent conditionals
        for head in [model.ar_head, *model.heads]:
            head.weight.mul_(8.0)
        model.u_in.weight.mul_(4.0)
    return model.double().eval()


def test_inverted_auxiliaries_repick_their_tokens():
    torch.manual_seed(0)
    V, N = 64, 4096
    probs = (torch.randn(N, V, dtype=torch.float64) * 6).softmax(-1)      # down to ~1e-13
    rank = torch.randperm(V)
    y = torch.randint(0, V, (N,))                                         # improbable tokens included
    u = invert(probs, y, rank)
    lo, hi = ordered_cdf_bounds(probs, y, rank)
    p = probs.gather(-1, y[:, None]).squeeze(1)
    ok = (u >= lo) & (u < hi) & (ordered_pick(probs, u, rank) == y)
    # Only a token below float64 resolution at its CDF position (p < ~1e-16 near 1) has an empty
    # interval; no u picks it, so the sampler never emits it either.
    assert ok[p > 1e-15].all() and (p[~ok] < 1e-15).all()
    assert torch.allclose(hi - lo, p, rtol=1e-6, atol=1e-15)


@pytest.mark.parametrize("V", [64, 50])                      # 50: the tree pads to 64 massless leaves
def test_tree_cells_repick_their_tokens_with_volume_equal_to_probability(V):
    torch.manual_seed(5)
    N = 4096
    probs = (torch.randn(N, V, dtype=torch.float64) * 4).softmax(-1)
    rank = torch.randperm(V)
    y = torch.randint(0, V, (N,))
    lo, hi = tree_bounds(probs, y, rank)
    u = place(lo, hi, torch.rand(lo.shape, dtype=torch.float64))
    p = probs.gather(-1, y[:, None]).squeeze(1)
    assert torch.allclose((hi - lo).prod(-1), p, rtol=1e-6, atol=1e-15)     # prefix-sum differences
    assert torch.equal(tree_pick(probs, u, rank)[p > 1e-12], y[p > 1e-12])


@pytest.mark.parametrize("pick", ["cdf", "tree"])
def test_picks_sample_the_distribution_in_any_order(pick):
    torch.manual_seed(1)
    V, N = 16, 400_000
    probs = (torch.randn(V, dtype=torch.float64) * 1.5).softmax(-1)
    rank = torch.randperm(V)
    if pick == "cdf":
        t = ordered_pick(probs.expand(N, V), torch.rand(N, dtype=torch.float64), rank)
    else:
        t = tree_pick(probs.expand(N, V), torch.rand(N, 4, dtype=torch.float64), rank)
    freq = torch.bincount(t, minlength=V).double() / N
    assert (freq - probs).abs().max().item() < 4e-3


def test_tree_pick_survives_small_distribution_changes_that_flip_inverse_cdf_picks():
    """The reason for the tree pick: under the same noise, a KL ~3e-3 perturbation of a peaked
    512-token distribution flips a large share of inverse-CDF picks and few tree picks."""
    torch.manual_seed(6)
    V, N = 512, 20_000
    p = (torch.randn(V, dtype=torch.float64) * 3).softmax(-1)
    q = (p.log() + 0.1 * torch.randn(V, dtype=torch.float64)).softmax(-1)
    rank = torch.randperm(V)
    P, Q = p.expand(N, V), q.expand(N, V)
    u = torch.rand(N, dtype=torch.float64)
    ut = torch.rand(N, 9, dtype=torch.float64)
    flat = (ordered_pick(P, u, rank) != ordered_pick(Q, u, rank)).double().mean().item()
    tree = (tree_pick(P, ut, rank) != tree_pick(Q, ut, rank)).double().mean().item()
    tv = 0.5 * (p - q).abs().sum().item()
    assert tree < 3 * tv < flat


def test_semantic_rank_keeps_each_state_contiguous():
    torch.manual_seed(2)
    E = torch.rand(6, 40) ** 4
    E = E / E.sum(1, keepdim=True)
    rank = semantic_rank(E)
    walk = E.argmax(0)[torch.argsort(rank)]                               # dominant state along the walk
    assert torch.equal(walk, walk.sort().values)                          # nondecreasing: one run per state
    assert torch.equal(rank.sort().values, torch.arange(40))


@pytest.mark.parametrize("coupling", COUPLINGS)
def test_ar_mode_inversion_round_trips_through_its_sequential_sampler(coupling):
    """Data inverted under the teacher-forced distributions is regenerated exactly by the AR mode's
    sequential sampler: the parallel inversion is the sequential one."""
    model = _model(V=8, T=6, coupling=coupling)
    ctx = torch.randn(64, 5, dtype=torch.float64)
    y = torch.randint(0, 8, (64, 6))
    with torch.no_grad():
        u = place(*model.cell(model.ar_logits(ctx, y), y), model.noise(64, 6, "cpu"))
    assert torch.equal(model.ar_sample(ctx, u), y)


@pytest.mark.parametrize("coupling", COUPLINGS)
@pytest.mark.parametrize("stages,coupled", ARMS)
def test_generator_sees_only_earlier_auxiliaries(stages, coupled, coupling):
    model = _model(T=5, stages=stages, coupled=coupled, coupling=coupling)
    ctx = torch.randn(8, 5, dtype=torch.float64)
    u, ud = model.noise(8, 5, "cpu"), model.noise(8, 5, "cpu")
    with torch.no_grad():
        base = model.generate_logits(ctx, u, ud)[1]
        for j in range(5):
            u2, ud2 = u.clone(), ud.clone()
            u2[:, j], ud2[:, j] = torch.rand_like(u[:, j]), torch.rand_like(ud[:, j])
            out = model.generate_logits(ctx, u2, ud2)[1]
            assert torch.allclose(out[:, :j + 1], base[:, :j + 1], atol=1e-12)
            if j + 1 < 5:
                assert not torch.allclose(out[:, j + 1:], base[:, j + 1:], atol=1e-6)


@pytest.mark.parametrize("coupling", COUPLINGS)
@pytest.mark.parametrize("stages,coupled", ARMS)
def test_importance_sampled_likelihood_is_the_one_pass_samplers(stages, coupled, coupling):
    """V=4, T=3: the 64 sequence probabilities from log_prob sum to one and match the histogram
    of one-pass samples (u -> one call -> picks), within 5 standard errors of the two Monte Carlo
    estimates. (Path weights vary strongly with u under these random weights, so a single
    estimate is noisy: measured sums 0.990 to 1.010 at S=20k, unbiased across repeats.)"""
    model = _model(V=4, T=3, stages=stages, coupled=coupled, seed=3, coupling=coupling)
    ctx = torch.randn(1, 5, dtype=torch.float64)
    torch.manual_seed(4)
    N, R = 200_000, 6
    ys = model.sample(ctx.expand(N, 5))
    freq = torch.bincount(ys[:, 0] * 16 + ys[:, 1] * 4 + ys[:, 2], minlength=64).double() / N
    seqs = torch.tensor(list(itertools.product(range(4), repeat=3)))
    reps = torch.stack([model.log_prob(ctx.expand(64, 5), seqs, S=4_000).double().exp() for _ in range(R)])
    p, se_p = reps.mean(0), reps.std(0) / R ** 0.5
    se = (se_p ** 2 + p * (1 - p) / N).sqrt()             # binomial variance at the estimate: rare
                                                          # sequences can have zero hits
    sums = reps.sum(1)
    assert abs(sums.mean().item() - 1.0) < 5 * sums.std().item() / R ** 0.5 + 1e-9
    # The normal approximation needs about 5 expected hits; rarer sequences are pooled (a single hit
    # on a p = 3e-8 sequence in 200k draws happened once, and is consistent with p).
    common = p * N >= 5
    assert ((p - freq).abs() < 5 * se + 1e-9)[common].all()
    rare_p, rare_f = p[~common].sum(), freq[~common].sum()
    assert abs(rare_p - rare_f).item() < 5 * (rare_p / N).sqrt().item() + 5 / N
    assert freq.max().item() > 0.05 and (freq > 0.005).sum().item() >= 8      # the test has power
    assert se.max().item() < 0.2 * freq.max().item()


@pytest.mark.parametrize("coupling", COUPLINGS)
@pytest.mark.parametrize("stages,coupled", ARMS)
def test_self_inversion_regenerates_the_data_in_one_pass(stages, coupled, coupling):
    """Auxiliaries inverted under the generator's own distributions, sequentially or by T + 1
    parallel sweeps from any start, make the one-pass sampler emit exactly the data; the last sweep
    moves nothing."""
    model = _model(T=5, stages=stages, coupled=coupled, coupling=coupling)
    ctx = torch.randn(32, 5, dtype=torch.float64)
    y = torch.randint(0, 8, (32, 5))
    rel, ud = model.noise(32, 5, "cpu"), model.noise(32, 5, "cpu")
    with torch.no_grad():
        u_seq = model._invert_seq(ctx, y, ud, rel)
        u_jac, moved = model._invert_sweeps(ctx, y, torch.rand_like(rel), ud, rel, 6)
    assert torch.equal(model.sample(ctx, u_seq, ud), y)
    assert torch.allclose(u_jac, u_seq, atol=1e-12) and moved.item() == 0.0


@pytest.mark.parametrize("inversion", ["ar", "seq", "jacobi"])
@pytest.mark.parametrize("stages,coupled", ARMS)
def test_training_loss_reaches_every_module(stages, coupled, inversion):
    model = _model(T=5, stages=stages, coupled=coupled).train()
    model.inversion = inversion
    ctx = torch.randn(16, 5, dtype=torch.float64)
    loss, aux = model.loss(ctx, torch.randint(0, 8, (16, 5)))
    loss.backward()
    dead = [n for n, p in model.named_parameters() if p.grad is None or p.grad.abs().sum() == 0]
    assert dead == [] and torch.isfinite(loss)
