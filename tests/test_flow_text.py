import json
import math
from types import SimpleNamespace

import pytest
import torch

from nanochat.flow_text import FlowText, FlowTextConfig
from scripts.sap_flow_text import (baseline_assets, baseline_budget, count_training_flops,
                                   split_row, dense_score, training_budget, training_schedule)


@pytest.fixture(autouse=True)
def small_cpu():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.manual_seed(22)
    yield
    torch.set_num_threads(old)


def tiny(scale=1):
    return FlowText(FlowTextConfig(vocab=11, T=4, prompt=3, width=16, heads=2,
                                   latent=2, projection_chunk=2, conditioner_depth=scale,
                                   decoder_depth=2 * scale, recognition_depth=2 * scale)).double()


def test_split_preserves_prompt_boundary_and_all_T_targets():
    sequence = torch.arange(14).view(2, 7)
    prompt, target = split_row(sequence[:, :-1], sequence[:, 1:], 3)
    assert torch.equal(prompt, sequence[:, :3])
    assert torch.equal(target, sequence[:, 3:])


@pytest.mark.parametrize("scale", [1, 2])
def test_prompt_conditioned_flow_inverse_and_jacobian(scale):
    m = tiny(scale).eval()
    with torch.no_grad():
        for layer in m.flow:
            layer.output.weight.normal_(0, .12)
    p = m.embedding(torch.tensor([[1, 2, 3]]))
    epsilon = torch.randn(1, 4, 2, dtype=torch.double)
    z, ld = m.transform(epsilon, p)
    back, ild = m.transform(z, p, inverse=True)
    torch.testing.assert_close(back, epsilon)
    torch.testing.assert_close(ld, -ild)
    jac = torch.autograd.functional.jacobian(lambda x: m.transform(x.view_as(epsilon), p)[0].flatten(), epsilon.flatten())
    torch.testing.assert_close(torch.linalg.slogdet(jac)[1], ld[0])


def test_projection_chunks_have_identical_likelihood_and_gradient():
    m = tiny().train()
    h = torch.randn(2, 4, 16, dtype=torch.double, requires_grad=True)
    y = torch.randint(11, (2, 4))
    lp = m.conditional_logprob(h, y)
    expected = m.readout(h).log_softmax(-1).gather(-1, y[..., None]).sum((1, 2))
    torch.testing.assert_close(lp, expected)
    actual_grad = torch.autograd.grad(lp.sum(), (h, m.readout.weight), retain_graph=True)
    expected_grad = torch.autograd.grad(expected.sum(), (h, m.readout.weight))
    for a, b in zip(actual_grad, expected_grad):
        torch.testing.assert_close(a, b)


@pytest.mark.parametrize("scale", [1, 2])
def test_sampler_is_target_free_and_replays_one_complete_tape(monkeypatch, scale):
    m = tiny(scale).eval()
    prompts = torch.tensor([[1, 2, 3], [4, 5, 6]])
    tape = m.noise_tape(prompts)
    y = m.sample(prompts, tape)
    calls = []
    hooks = [block.register_forward_hook(lambda *_: calls.append(1)) for path in [
        m.flow[0].conditioner, m.flow[1].conditioner, m.decoder] for block in path.blocks]
    def forbidden(*args, **kwargs):
        raise AssertionError("recognition or fresh randomness used during fixed-tape generation")
    monkeypatch.setattr(m.recognition, "forward", forbidden)
    monkeypatch.setattr(torch, "rand", forbidden)
    monkeypatch.setattr(torch, "randn", forbidden)
    assert torch.equal(m.sample(prompts, tape), y)
    assert len(calls) == 4 * scale == m.config.generative_depth
    for hook in hooks:
        hook.remove()
    assert y.shape == (2, 4)


def test_decoder_distribution_normalizes_without_rejection():
    m = tiny().eval()
    p = m.embedding(torch.tensor([[1, 2, 3]]))
    noise = torch.randn(1, 4, 2, dtype=torch.double)
    z, _ = m.transform(noise, p)
    q = m.readout(m.decoder(p, z)).softmax(-1)
    torch.testing.assert_close(q.sum(-1), torch.ones(1, 4, dtype=torch.double))


@pytest.mark.parametrize("scale", [1, 2])
def test_full_training_path_has_finite_gradients_and_counted_recognition(scale):
    m = tiny(scale).train()
    p, y = torch.randint(11, (2, 3)), torch.randint(11, (2, 4))
    loss = m(p, y)
    loss.backward()
    for name, parameter in m.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
    m.zero_grad(set_to_none=True)
    f1 = count_training_flops(m, p[:1], y[:1], "cpu")
    f2 = count_training_flops(m, p, y, "cpu")
    assert f1 > 0 and f2 == 2 * f1


def test_zero_latent_readout_special_case_has_exact_importance_weights():
    m = tiny().eval()
    with torch.no_grad():
        m.decoder.input.weight.zero_()
    p, y = torch.randint(11, (2, 3)), torch.randint(11, (2, 4))
    eps = torch.randn(2, 4, 2, dtype=torch.double)
    ll, kl = m.terms(p, y, eps)
    torch.testing.assert_close(kl, torch.zeros_like(kl))
    other, other_kl = m.terms(p, y, -eps)
    torch.testing.assert_close(ll, other)
    torch.testing.assert_close(kl, other_kl)


def test_budget_comes_from_baseline_points_not_a_typed_token_count():
    record = {"returncode": 0, "train_tokens": 121110528, "flops_per_token": 94372610}
    assert baseline_budget(record) == 121110528 * 94372610
    with pytest.raises(ValueError):
        baseline_budget({**record, "returncode": 1})


@pytest.mark.parametrize("steps,multiple", [(1680, 1), (5040, 3), (8400, 5)])
def test_explicit_matched_batch_budgets_are_not_capped_to_isoflops(steps, multiple):
    args = SimpleNamespace(microbatch=4, T=2048, batch_tokens=262144, steps=steps, smoke=False)
    dense_budget = 440401920 * 352324600
    accum, actual_steps, per_update = training_budget(args, dense_budget, 1178532315136)
    assert accum == 32 and actual_steps == steps
    assert actual_steps * args.batch_tokens == multiple * 440401920
    assert per_update * actual_steps / dense_budget == pytest.approx(multiple * 1.6333098284933836)
    for step in (1, 34, 84, 168, steps):
        lr, beta = training_schedule(step, steps, 3e-4, 34, 168)
        assert 0 < lr <= 3e-4
        assert beta == min(1., step / 168)
    assert training_schedule(steps, steps, 3e-4, 34, 168) == pytest.approx((3e-5, 1.))


def test_budget_rejects_silent_batch_rounding_and_preserves_old_isoflop_mode():
    args = SimpleNamespace(microbatch=4, T=2048, batch_tokens=65536, steps=None, smoke=False)
    accum, steps, _ = training_budget(args, 440401920 * 352324600, 1178532315136)
    assert accum == 8 and steps == 4114
    args.batch_tokens = 262145
    with pytest.raises(ValueError, match="exact multiple"):
        training_budget(args, 1, 1)
    args.batch_tokens, args.steps = 262144, 0
    with pytest.raises(ValueError, match="positive"):
        training_budget(args, 1, 1)
    with pytest.raises(ValueError, match="warmup"):
        training_schedule(1, 1680, 3e-4, 34, 2000)


def test_gradient_accumulation_equals_same_effective_batch_gradient():
    model = tiny().train()
    prompts, targets = torch.randint(11, (4, 3)), torch.randint(11, (4, 4))
    # Fixed posterior randomness makes this an exact batch-reduction test.
    epsilon = torch.randn(4, 4, 2, dtype=torch.double)
    ll, kl = model.terms(prompts, targets, epsilon)
    ((-ll + kl).mean() / 4).backward()
    reference = {name: p.grad.clone() for name, p in model.named_parameters()}
    model.zero_grad(set_to_none=True)
    for start in (0, 2):
        ll, kl = model.terms(prompts[start:start + 2], targets[start:start + 2], epsilon[start:start + 2])
        ((-ll + kl).mean() / (4 * 2)).backward()
    for name, p in model.named_parameters():
        torch.testing.assert_close(p.grad, reference[name], atol=1e-10, rtol=1e-8)


def test_explicit_training_tape_and_tensor_beta_preserve_objective():
    model = tiny().train()
    prompts, targets = torch.randint(11, (2, 3)), torch.randint(11, (2, 4))
    epsilon = torch.randn(2, 4, 2, dtype=torch.double)
    for beta in (.1, .7, 1.):
        actual = model(prompts, targets, torch.tensor(beta, dtype=torch.double), epsilon)
        ll, kl = model.terms(prompts, targets, epsilon)
        torch.testing.assert_close(actual, (-ll + beta * kl).mean() / 4)


def test_sweep_collection_preserves_all_arms_and_marks_partial_results(tmp_path, monkeypatch):
    import modal_sap_flow_text as launcher
    monkeypatch.setattr(launcher, "ROOT", str(tmp_path / "remote"))
    sweep_id = "flow_d8_mb262k_test"
    for steps in launcher.MATCHED_STEPS:
        root = tmp_path / "remote" / f"{sweep_id}_u{steps}"
        root.mkdir(parents=True)
        (root / "run.log").write_text(f"step {steps}\n")
        (root / "samples.jsonl").write_text('{"sample": true}\n')
        if steps == 1680:
            (root / "COMPLETE").touch()
    result = launcher.sweep_results(sweep_id)
    assert not result["complete"] and len(result["runs"]) == 3
    assert [r["complete"] for r in result["runs"]] == [True, False, False]
    launcher.save_sweep(result, tmp_path / "local")
    compiled = (tmp_path / "local" / "compiled.log").read_text()
    assert "COMPLETE: False" in compiled
    for steps in launcher.MATCHED_STEPS:
        assert f"step {steps}" in compiled
    for steps in (5040, 8400):
        (tmp_path / "remote" / f"{sweep_id}_u{steps}" / "COMPLETE").touch()
    assert launcher.sweep_results(sweep_id)["complete"]


def test_coordinator_waits_for_existing_call_but_does_not_hide_worker_timeout(monkeypatch):
    import modal_sap_flow_text as launcher
    import time
    monkeypatch.setattr(time, "sleep", lambda _: None)
    class Call:
        def __init__(self, outputs):
            self.outputs = iter(outputs)
            self.polls = 0
        def get(self, timeout):
            assert timeout == 45
            self.polls += 1
            value = next(self.outputs)
            if isinstance(value, Exception):
                raise value
            return value
    class SyncTimeout(launcher.modal.exception.TimeoutError):
        pass
    call = Call([SyncTimeout(),
                 launcher.modal.exception.ConnectionError("transient"), "completed"])
    assert launcher.wait_for_existing_call(call) == "completed"
    assert call.polls == 3
    with pytest.raises(launcher.modal.exception.FunctionTimeoutError):
        launcher.wait_for_existing_call(Call([launcher.modal.exception.FunctionTimeoutError()]))


def test_depth_defaults_preserve_old_d4_checkpoint_layout():
    old_config = dict(vocab=11, T=4, prompt=3, width=16, heads=2, latent=2, projection_chunk=2)
    restored = FlowText(FlowTextConfig(**old_config)).double()
    restored.load_state_dict(tiny().state_dict(), strict=True)
    assert restored.config.generative_depth == 4
    with pytest.raises(ValueError, match="depths"):
        FlowText(FlowTextConfig(**old_config, conditioner_depth=0))


def test_baseline_provenance_checks_depth_step_tokens_and_tag(tmp_path):
    config = FlowTextConfig(width=512, conditioner_depth=2, decoder_depth=4, recognition_depth=4)
    record = dict(returncode=0, train_tokens=440401920, flops_per_token=352324600, val_bpb=.958955)
    meta = dict(step=1680, total_batch_size=262144, val_bpb=.958955,
                user_config=dict(model_tag="S08_dense_L_s1"),
                model_config=dict(vocab_size=32768, n_layer=8, n_embd=512, window_pattern="L"))
    (tmp_path / "model_001680.pt").touch()
    (tmp_path / "meta_001680.json").write_text(json.dumps(meta))
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"S08_dense_L_s1": record}))
    args = SimpleNamespace(baseline_record=baseline, baseline_key="S08_dense_L_s1", dense_dir=tmp_path)
    assert baseline_assets(args, config) == (record, baseline_budget(record), 1680)
    with pytest.raises(ValueError, match="architecture"):
        baseline_assets(args, FlowTextConfig())
    record["train_tokens"] -= 1
    baseline.write_text(json.dumps({"S08_dense_L_s1": record}))
    with pytest.raises(ValueError, match="budget/result"):
        baseline_assets(args, config)


def test_dense_scorer_contiguous_views_and_identical_continuation_scope():
    class Dense:
        def eval(self):
            return self
        def __call__(self, inputs, targets, loss_reduction):
            assert inputs.is_contiguous() and targets.is_contiguous()
            assert torch.equal(inputs[:, 1:], targets[:, :-1])
            # Mimic GPT's flattened loss interface and special-token handling.
            return targets.view(-1).double() + 1
    prompts = torch.tensor([[1, 2, 3], [2, 3, 4]])
    targets = torch.tensor([[0, 4, 5, 6], [7, 6, 5, 4]])
    sizes = torch.tensor([0] + [1] * 10)
    result = dense_score(Dense(), prompts, targets, sizes, "cpu")
    total = (targets.double() + 1).sum().item()
    assert result["joint_bpb"] == pytest.approx(total / (7 * math.log(2)))
    assert result["conventional_masked_bpb"] == pytest.approx((total - 1) / (7 * math.log(2)))
