"""Distributional correctness of the Flow–Joint toy gate, not just tensor shapes."""
import itertools

import pytest
import torch

from nanochat.flow_joint import FlowJoint, categorical, scan_states


@pytest.fixture(autouse=True)
def small_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(71)
    yield
    torch.set_num_threads(old)


def model(arm="hybrid", T=3, V=3):
    return FlowJoint(5, T=T, vocab=V, width=16, states=2, latent=2, arm=arm).double()


def test_flow_inverse_and_jacobian():
    m = model(T=2).eval()
    with torch.no_grad():
        for layer in m.flow:
            layer.output.weight.normal_(0, .1)
            layer.output.bias.normal_(0, .1)
    ctx = torch.randn(1, 5, dtype=torch.double)
    x = torch.randn(1, 2, 2, dtype=torch.double)
    z, forward_ld = m.transform(x, ctx)
    back, inverse_ld = m.transform(z, ctx, inverse=True)
    torch.testing.assert_close(back, x)
    torch.testing.assert_close(forward_ld, -inverse_ld)
    jac = torch.autograd.functional.jacobian(lambda v: m.transform(v.view_as(x), ctx)[0].flatten(), x.flatten())
    torch.testing.assert_close(torch.linalg.slogdet(jac)[1], forward_ld[0])


def test_emissions_equal_dense_reference_and_normalize():
    m = model()
    ctx = torch.randn(2, 5, dtype=torch.double)
    z = torch.randn(2, 3, 2, dtype=torch.double)
    logq, _, _ = m.fields(ctx, z)
    dense = (logq[:, :, None, :] + m.emission_logits[None, None]).log_softmax(-1)
    y = torch.randint(3, (2, 3))
    reference = dense.gather(-1, y[:, :, None, None].expand(-1, -1, 2, 1)).squeeze(-1)
    torch.testing.assert_close(m.target_emissions(logq, y), reference)
    torch.testing.assert_close(dense.exp().sum(-1), torch.ones(2, 3, 2, dtype=torch.double))


@pytest.mark.parametrize("T", [1, 3, 4, 9])
def test_scan_matches_sequential_random_maps(T):
    initial = torch.randint(7, (5,))
    maps = torch.randint(7, (5, T - 1, 7))
    serial = [initial]
    for t in range(T - 1):
        serial.append(maps[:, t].gather(1, serial[-1][:, None]).squeeze(1))
    assert torch.equal(scan_states(initial, maps), torch.stack(serial, 1))


def test_zero_probability_is_never_selected_at_zero_uniform():
    assert categorical(torch.tensor([[0., 0., .4, .6]]), torch.tensor([0.])).item() == 2


def test_single_token_chain_has_no_transition_requirement():
    m = model(arm="chain_only", T=1).eval()
    ctx = torch.randn(3, 5, dtype=torch.double)
    assert torch.isfinite(m.conditional_logprob(ctx, torch.tensor([[0], [1], [2]]))).all()
    assert m.sample(ctx).shape == (3, 1)


@pytest.mark.parametrize("arm", ["hybrid", "chain_only", "flow_only"])
def test_joint_sums_to_one_and_sampler_matches(arm):
    m = model(arm=arm).eval()
    ctx = torch.randn(1, 5, dtype=torch.double)
    z = torch.randn(1, 3, 2, dtype=torch.double) if m.has_flow else None
    outcomes = torch.tensor(list(itertools.product(range(3), repeat=3)))
    p = m.conditional_logprob(ctx.expand(27, -1), outcomes,
                             z.expand(27, -1, -1) if z is not None else None).exp().detach()
    torch.testing.assert_close(p.sum(), torch.ones((), dtype=torch.double))
    # CPU smoke-sized statistical test. Fixed z tests the entire exact local joint.
    n = 12000
    samples = []
    for _ in range(n // 200):
        samples.append(m.sample(ctx.expand(200, -1), z_override=z.expand(200, -1, -1) if z is not None else None))
    samples = torch.cat(samples)
    ids = samples @ torch.tensor([9, 3, 1])
    frequency = torch.bincount(ids, minlength=27).double() / n
    zscores = (frequency - p).abs() / (p * (1 - p) / n).sqrt()
    assert zscores.max() < 5


@pytest.mark.parametrize("arm", ["hybrid", "chain_only", "flow_only"])
def test_finite_gradient_and_no_new_randomness_with_tape(arm, monkeypatch):
    m = model(arm=arm)
    ctx = torch.randn(3, 5, dtype=torch.double)
    y = torch.randint(3, (3, 3))
    loss = m.loss(ctx, y)
    loss.backward()
    for name, parameter in m.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
    m.eval()
    tape = m.noise_tape(ctx)
    expected = m.sample(ctx, tape=tape)
    def no_randomness(*args, **kwargs):
        raise AssertionError("sampler drew noise after being given the complete tape")
    monkeypatch.setattr(torch, "rand", no_randomness)
    monkeypatch.setattr(torch, "randn", no_randomness)
    assert torch.equal(m.sample(ctx, tape=tape), expected)


def test_iwae_density_terms_reduce_to_exact_when_latent_is_ignored():
    m = model()
    with torch.no_grad():
        m.decoder.input.weight.zero_()
    ctx = torch.randn(3, 5, dtype=torch.double)
    y = torch.randint(3, (3, 3))
    # Initial flow and posterior are both standard normal, hence importance ratio=1.
    ll, kl = m.terms(ctx, y, K=7)
    torch.testing.assert_close(kl, torch.zeros_like(kl))
    exact = m.conditional_logprob(ctx, y, torch.zeros(3, 3, 2, dtype=torch.double))
    torch.testing.assert_close(ll, exact[None].expand_as(ll))


def test_partial_results_can_be_compiled_without_final_importance_count():
    from scripts.sap_flow_joint_gate import compile_text
    row = {"config": {"seed": 0, "arm": "hybrid", "eval_k": 64, "final_k": 256},
           "training_steps": 8000, "matrix_budget_ratio": 1.,
           "curve": [{"step": 2000, "kl_k64": {"mean": 2., "se": .1},
                      "prior_invalid": {"mean": .5, "se": .01}, "training_seconds": 10.,
                      "likelihood_is_bound": True}]}
    text = compile_text([row])
    assert "updates=2000/8000 K=64" in text
