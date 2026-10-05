"""
Tests for the SAP sampling-aware block head (nanochat/block_head.py).

Each test guards a claim the SAP paper rests on: that the exact-likelihood modes are
normalised, that the latent-variable bounds really are bounds, that causal slots cannot read
what they are about to emit, that the training loss is the next-token loss plus exactly the
block term, and that the KV-cached decode loop (the one that gets timed) emits the same
blocks as the reference loop.

All tests run on CPU with tiny models and finish in seconds.
"""

import itertools
import math

import pytest
import torch

from nanochat.block_head import (SAP_MODES, SIR_MODES, energy_score, evaluate_block_bpb, pick,
                                 target_bins)
from nanochat.block_head import DEPTH_MODES
from nanochat.engine import generate_block_kv
from nanochat.gpt import GPT, GPTConfig

V = 16      # tiny, so whole block distributions can be enumerated
D = 64


def build(mode, T=2, **kw):
    layers = 3 if mode in ("sir_conf", "sir_pyramid", "sir_tree", "sir_full") else (2 if mode in SIR_MODES else 1)
    opts = dict(sap_block_frac=0.5, sap_head_layers=layers, sap_ctx_window=4, sap_kl_anneal_steps=0,
                sap_latent_groups=2, sap_latent_codes=3, sap_cp_components=3)
    opts.update(kw)
    cfg = GPTConfig(n_layer=2, n_head=1, n_kv_head=1, n_embd=D, vocab_size=V,
                    sequence_len=32, window_pattern="L",
                    sap_block_T=T, sap_block_mode=mode, **opts)
    torch.manual_seed(0)
    model = GPT(cfg)
    model.init_weights(verify=True)
    # Random-init heads start near uniform, which hides normalisation bugs. Push every
    # head-side weight away from its init so the distributions under test are peaked.
    with torch.no_grad():
        for p in model.sap_head.parameters():
            p.add_(torch.randn_like(p) * 0.5)
    model.eval()
    return model


def head_inputs(model, x, b, t):
    hid = model(x, skip_logits=True)
    ctx, valid = model._sap_window(hid, b, t)
    return hid[b, t], ctx, valid


def all_blocks(T):
    return torch.tensor(list(itertools.product(range(V), repeat=T)))


@pytest.mark.parametrize("mode", ["indep", "local", "cp", "field_cp", "sir_tree"])
def test_exact_modes_are_normalised(mode):
    """Summed over every possible block, an exact mode's probabilities add up to 1."""
    model = build(mode)
    x = torch.randint(0, V, (1, 12), generator=torch.Generator().manual_seed(1))
    blocks = all_blocks(2)
    n = blocks.size(0)
    h, ctx, valid = head_inputs(model, x, torch.zeros(1, dtype=torch.long), torch.tensor([7]))
    with torch.no_grad():
        lp = model.sap_head.block_logprob(h.expand(n, -1), ctx.expand(n, -1, -1), valid.expand(n, -1),
                                          blocks, model._sap_readout, embed=model.transformer.wte)
    assert torch.logsumexp(lp, 0).exp().item() == pytest.approx(1.0, abs=1e-4)


def test_local_jacobi_reports_underlying_local_likelihood():
    """Jacobi is an approximate sampler for the exact teacher-forced local conditionals."""
    model = build("local_jacobi")
    x = torch.randint(0, V, (2, 12), generator=torch.Generator().manual_seed(14))
    b, t = torch.arange(2), torch.full((2,), 7)
    h, ctx, valid = head_inputs(model, x, b, t)
    y = torch.tensor([[3, 11], [5, 7]])
    with torch.no_grad():
        jacobi_lp = model.sap_head.block_logprob(
            h, ctx, valid, y, model._sap_readout, embed=model.transformer.wte)
        model.sap_head.mode = "local"
        local_lp = model.sap_head.block_logprob(
            h, ctx, valid, y, model._sap_readout, embed=model.transformer.wte)
    assert jacobi_lp is not None
    assert torch.equal(jacobi_lp, local_lp)


def test_latent_sensitivity_caps_readout_rows():
    """The T=L diagnostic must score only the configured block/slot sample."""
    model = build("p1_discrete", T=8)
    x = torch.randint(0, V, (6, 16), generator=torch.Generator().manual_seed(15))
    b, t = torch.arange(6), torch.full((6,), 7)
    h, ctx, valid = head_inputs(model, x, b, t)
    rows = []

    def tracked_readout(states):
        rows.append(states.size(0))
        return model._sap_readout(states)

    with torch.no_grad():
        sensitivity = model.sap_head.latent_sensitivity(
            h, ctx, valid, tracked_readout, n_samples=2, max_blocks=2, max_slots=3)
    assert sensitivity.shape == (2,)
    assert torch.isfinite(sensitivity).all()
    assert rows == [6, 6]


def test_block_eval_reports_na_when_no_joint_block_is_admissible():
    model = build("local", T=4)
    x = torch.randint(0, V, (2, 12), generator=torch.Generator().manual_seed(16))
    y = torch.full_like(x, -1)
    metrics = evaluate_block_bpb(model, [(x, y)], steps=1,
                                 token_bytes=torch.ones(V), blocks_per_row=2)
    assert metrics["blocks"] == 0
    assert metrics["ntp_bpb_same_tokens"] is None
    assert metrics["block_bpb"] is None


@pytest.mark.parametrize("mode", SIR_MODES)
def test_s01_sir_modes_train_sample_and_score(mode):
    """Every S01 arm owns a real training, sampling, and likelihood-estimation path."""
    layers = 3 if mode in ("sir_conf", "sir_pyramid", "sir_tree", "sir_full") else 2
    model = build(mode, T=4, sap_head_layers=layers, sap_sir_topk=4,
                  sap_sir_anchor_stride=2, sap_sir_fine_stride=1)
    x = torch.randint(0, V, (2, 12), generator=torch.Generator().manual_seed(21))
    y = torch.randint(0, V, (2, 12), generator=torch.Generator().manual_seed(22))
    model.train()
    loss = model(x, y)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all()
               for p in model.sap_head.parameters())

    model.eval()
    b, t = torch.arange(2), torch.full((2,), 6)
    h, ctx, valid = head_inputs(model, x, b, t)
    block = y[b[:, None], t[:, None] + torch.arange(4)[None]]
    with torch.no_grad():
        sample = model.sap_head.sample(h, ctx, valid, model._sap_readout,
                                       embed=model.transformer.wte,
                                       embed_table=model.transformer.wte.weight,
                                       generator=torch.Generator().manual_seed(23))
        lp = model.sap_head.block_logprob(h, ctx, valid, block, model._sap_readout,
                                          embed=model.transformer.wte, n_samples=2)
    assert sample.shape == (2, 4)
    assert ((0 <= sample) & (sample < V)).all()
    assert lp.shape == (2,)
    assert torch.isfinite(lp).all()


def test_sir_reference_masks_the_matching_draft_token():
    """With one reference layer, output i has no differentiable path to draft i."""
    model = build("sir", T=3, sap_head_layers=2, sap_ctx_window=0)
    head = model.sap_head
    s = torch.randn(1, 3, D, requires_grad=True)
    draft = torch.randn(1, 3, D, requires_grad=True)
    out = head._reference_layer(s, draft, None, None, head.dec[1], leave_one_out=True)
    grad = torch.autograd.grad(out[:, 0].sum(), draft)[0]
    assert grad[:, 0].abs().max().item() == pytest.approx(0.0, abs=1e-8)
    assert grad[:, 1:].abs().sum().item() > 0


def test_sir_candidate_readout_is_chunk_bounded():
    model = build("sir_soft", T=4, sap_head_layers=2, sap_logit_chunk=3, sap_sir_topk=4)
    head = model.sap_head
    states = torch.randn(2, 4, D)
    targets = torch.randint(0, V, (2, 4))
    rows = []

    def tracked(s):
        rows.append(s.size(0))
        return model._sap_readout(s)

    vals, idx, lp = head._sir_candidates(states, targets, tracked)
    assert vals.shape == idx.shape == (2, 4, 4)
    assert lp.shape == (2, 4)
    assert rows == [3, 3, 2]


def test_sir_cost_orders_anchor_core_and_confidence():
    anchor = build("sir_anchor", T=8, sap_head_layers=2, sap_sir_anchor_stride=4).sap_head
    core = build("sir", T=8, sap_head_layers=2).sap_head
    conf = build("sir_conf", T=8, sap_head_layers=3).sap_head
    assert anchor.flops_per_token(V) < core.flops_per_token(V) < conf.flops_per_token(V)


def test_sir_tree_levels_cover_every_position_once():
    head = build("sir_tree", T=8, sap_head_layers=3).sap_head
    levels = [p.tolist() for p in head._sir_tree_positions(torch.device("cpu"))]
    assert levels == [[0], [4], [2, 6], [1, 3, 5, 7]]
    assert sorted(itertools.chain.from_iterable(levels)) == list(range(8))


def test_sir_anchor_oracle_commits_observed_anchors():
    model = build("sir_anchor", T=4, sap_head_layers=2, sap_sir_anchor_stride=2,
                  sap_sir_posterior_mix=0.0)
    x = torch.randint(0, V, (2, 12), generator=torch.Generator().manual_seed(41))
    y = torch.tensor([[2, 3, 5, 7], [11, 13, 1, 4]])
    b, t = torch.arange(2), torch.full((2,), 6)
    h, ctx, valid = head_inputs(model, x, b, t)
    head = model.sap_head
    with torch.no_grad():
        _, aux = head._sir_forward(
            head._base(h), head._ctx(ctx), valid, model._sap_readout,
            model.transformer.wte, y_safe=y, valid=torch.ones_like(y, dtype=torch.bool),
            force_teacher=True)
    assert torch.equal(aux["draft_ids"], y[:, [0, 2]])


def test_sir_training_uses_post_sampling_reference_weights_and_prices_samples():
    model = build("sir", T=4, sap_head_layers=2, sap_sir_topk=4,
                  sap_sir_train_samples=3)
    x = torch.randint(0, V, (2, 12), generator=torch.Generator().manual_seed(31))
    y = torch.randint(0, V, (2, 12), generator=torch.Generator().manual_seed(32))
    model.train()
    loss = model(x, y)
    stats = model._sap_stats
    assert torch.isfinite(loss)
    assert torch.isfinite(stats["draft_policy_surrogate"])
    assert 1.0 <= stats["draft_ref_ess"].item() <= 3.0

    one = build("sir", T=4, sap_head_layers=2, sap_sir_train_samples=1).sap_head
    three = model.sap_head
    assert three.flops_per_token(V) == 3 * one.flops_per_token(V)


def _exact_p1_logprob(model, h, ctx, valid, y):
    """log p(y) for the discrete plan head by summing over all C^G plans."""
    head = model.sap_head
    G, C = head.G, head.C
    base = head._base(h)
    ctx_in = head._ctx(ctx)
    terms = []
    for combo in itertools.product(range(C), repeat=G):
        idx = torch.tensor(combo)[None].expand(h.size(0), -1)
        z = torch.nn.functional.one_hot(idx, C).float()
        logp_z = torch.log_softmax(head._prior_logits_p1(h, z), -1).gather(-1, idx[..., None]).squeeze(-1).sum(-1)
        s = head._decode(base + head._z_emb_p1(z).to(h.dtype)[:, None], ctx_in, valid)
        ll = head._slot_logprob(s, y, model._sap_readout).sum(-1)
        terms.append(logp_z + ll)
    return torch.logsumexp(torch.stack(terms), 0)


@pytest.mark.parametrize("mode", ["p1_discrete", "p1_selfpost"])
def test_discrete_plan_bound_is_a_bound_and_tightens(mode):
    """The importance-weighted estimate never exceeds the exact log-likelihood on average,
    and more samples move it up towards it. p1_selfpost's posterior reads the trunk state
    after the block, so it is a different proposal for the same prior and decoder."""
    model = build(mode)
    x = torch.randint(0, V, (1, 12), generator=torch.Generator().manual_seed(2))
    h, ctx, valid = head_inputs(model, x, torch.zeros(1, dtype=torch.long), torch.tensor([7]))
    with torch.no_grad():
        h_future = model(x, skip_logits=True)[:, 8]
    y = torch.tensor([[3, 11]])
    kw = dict(embed=model.transformer.wte, h_future=h_future)
    with torch.no_grad():
        exact = _exact_p1_logprob(model, h, ctx, valid, y).item()
        torch.manual_seed(0)
        k1 = torch.stack([model.sap_head.block_logprob(h, ctx, valid, y, model._sap_readout,
                                                       n_samples=1, **kw)
                          for _ in range(400)]).mean().item()
        k64 = torch.stack([model.sap_head.block_logprob(h, ctx, valid, y, model._sap_readout,
                                                        n_samples=64, **kw)
                           for _ in range(50)]).mean().item()
    assert k1 <= exact + 1e-3
    assert k64 <= exact + 1e-3
    assert k64 >= k1 - 1e-3


@pytest.mark.parametrize("mode", ["p1_discrete", "p1_selfpost", "p2_gauss"])
def test_kl_vanishes_when_posterior_equals_prior(mode):
    model = build(mode)
    with torch.no_grad():
        model.sap_head.q_out.weight.zero_()
        model.sap_head.prior_out.weight.zero_()
    model.train()
    x = torch.randint(0, V, (2, 32), generator=torch.Generator().manual_seed(3))
    model(x, x)
    kl = model._sap_stats.get("kl_per_group", model._sap_stats.get("kl_per_block"))
    assert torch.allclose(kl, torch.zeros_like(kl), atol=1e-6)


def test_causal_slots_cannot_read_their_own_or_later_inputs():
    """local: slot k sees tokens < k only. inv_head: slot k sees noise u_<k only."""
    model = build("local", T=3)
    x = torch.randint(0, V, (1, 12), generator=torch.Generator().manual_seed(4))
    h, ctx, valid = head_inputs(model, x, torch.zeros(1, dtype=torch.long), torch.tensor([7]))
    head = model.sap_head
    with torch.no_grad():
        def slot_logits(y):
            prev = torch.cat([y.new_zeros(1, 1), y[:, :-1]], dim=1)
            tok = head.tok_in(torch.nn.functional.rms_norm(model.transformer.wte(prev), (D,)))
            tok = tok * (torch.arange(3) > 0)[None, :, None].to(tok.dtype)
            s = head._decode(head._base(h) + tok, head._ctx(ctx), valid)
            return model._sap_readout(s.reshape(-1, D)).view(3, -1)
        a = slot_logits(torch.tensor([[1, 2, 3]]))
        b = slot_logits(torch.tensor([[1, 2, 9]]))   # changes only the last slot's token
    assert torch.allclose(a[:3], b[:3]), "no slot may read the last slot's own token"

    model = build("inv_head", T=3)
    h, ctx, valid = head_inputs(model, x, torch.zeros(1, dtype=torch.long), torch.tensor([7]))
    head = model.sap_head
    with torch.no_grad():
        def inv_logits(u):
            s = head._decode(head._base(h) + head._u_emb(u, h.dtype), head._ctx(ctx), valid)
            return model._sap_readout(s.reshape(-1, D)).view(3, -1)
        a = inv_logits(torch.tensor([[0.2, 0.7]]))
        b = inv_logits(torch.tensor([[0.2, 0.1]]))    # changes u_2, read only by slot 3
    assert torch.allclose(a[:2], b[:2])
    assert not torch.allclose(a[2], b[2])


def test_head_reads_no_future_context():
    """The block at position t depends on tokens <= t only (the trunk is causal and the
    context window ends at t)."""
    model = build("indep")
    g = torch.Generator().manual_seed(5)
    x = torch.randint(0, V, (1, 16), generator=g)
    x2 = x.clone()
    x2[0, 9:] = torch.randint(0, V, (7,), generator=g)
    y = torch.tensor([[4, 5]])
    b, t = torch.zeros(1, dtype=torch.long), torch.tensor([8])
    with torch.no_grad():
        lps = []
        for seq in (x, x2):
            h, ctx, valid = head_inputs(model, seq, b, t)
            lps.append(model.sap_head.block_logprob(h, ctx, valid, y, model._sap_readout,
                                                    embed=model.transformer.wte))
    assert torch.allclose(lps[0], lps[1])


def test_training_loss_is_next_token_loss_plus_block_term():
    model = build("indep")
    model.train()
    x = torch.randint(0, V, (2, 32), generator=torch.Generator().manual_seed(6))
    torch.manual_seed(7)
    total = model(x, x)
    block = model._sap_stats["block_loss"]
    head, model.sap_head = model.sap_head, None
    try:
        ntp = model(x, x)
    finally:
        model.sap_head = head
    assert total.item() == pytest.approx(ntp.item() + model.config.sap_lambda * block.item(), rel=1e-5)


def test_inverse_cdf_round_trip():
    """A u drawn inside a token's bin picks that token back (the inv_head competitor)."""
    g = torch.Generator().manual_seed(8)
    logits = torch.randn(64, V, generator=g) * 3
    y = torch.randint(0, V, (64,), generator=g)
    lo, w = target_bins(logits, y)
    u = lo + w * 0.5
    assert torch.equal(pick(torch.softmax(logits, -1), u), y)


def test_energy_score_prefers_the_true_distribution():
    """In expectation the energy score ranks the true sampler above a collapsed one."""
    torch.manual_seed(9)
    n = 20000
    a, b = torch.tensor([1.0, 0.0]), torch.tensor([-1.0, 0.0])
    obs = torch.where(torch.rand(n, 1) < 0.5, a, b)[:, None, :]                # (n, T=1, d=2)
    valid = torch.ones(n, 1, dtype=torch.bool)

    def draw():
        return torch.where(torch.rand(n, 1) < 0.5, a, b)[:, None, :]

    true_samples = torch.stack([draw(), draw()], dim=1)                       # (n, 2, 1, 2)
    collapsed = torch.zeros(n, 2, 1, 2)                                       # always the mean
    one_mode = a.expand(n, 2, 1, 2)                                           # mode collapse
    es_true = energy_score(true_samples, obs, valid)[0].mean()
    es_mean = energy_score(collapsed, obs, valid)[0].mean()
    es_mode = energy_score(one_mode, obs, valid)[0].mean()
    assert es_true < es_mean and es_true < es_mode


class _RecomputeTrunk:
    """Stands in for the model inside generate_block_kv but recomputes the whole sequence on
    every call, so the loop's bookkeeping (windows, last state, RNG use) can be compared with
    the reference loop exactly, without the attention kernels' bf16 noise."""

    def __init__(self, model):
        self.model, self.seq = model, None

    def __getattr__(self, name):
        return getattr(self.model, name)

    def forward(self, ids, kv_cache=None, skip_logits=False):
        self.seq = ids if self.seq is None else torch.cat([self.seq, ids], dim=1)
        return self.model.forward(self.seq, skip_logits=True)[:, -ids.size(1):]


@pytest.mark.parametrize("mode", [m for m in SAP_MODES if m not in DEPTH_MODES])  # depth: own tests below
def test_kv_block_decode_loop_matches_reference(mode):
    """The decode loop that gets timed emits exactly the reference loop's blocks."""
    model = build_v4(mode, T=2) if mode in V4_MODES else build(mode)
    prompt = [1, 2, 3, 4, 5]
    ref = [tok for blk in model.generate_block(prompt, 8, temperature=0.0, seed=11) for tok in blk][:8]
    out = generate_block_kv(_RecomputeTrunk(model), prompt, 8, temperature=0.0, seed=11)[0].tolist()
    assert len(ref) == 8 and ref == out


def test_kv_cache_states_track_full_recompute():
    """The real KV path feeds the head the same trunk states up to bf16 attention noise.

    Both attention paths in gpt.py cast q, k, v to bf16, and the cached path rounds
    differently from the full one, so states agree to ~1e-2 rather than exactly; that is
    why the loop test above recomputes instead.
    """
    from nanochat.engine import _kv_cache_for
    model = build("indep")
    seq = torch.randint(0, V, (1, 13), generator=torch.Generator().manual_seed(12))
    with torch.no_grad():
        full = model(seq, skip_logits=True)
        kv = _kv_cache_for(model, 1, 32, torch.device("cpu"), torch.float32)
        parts = [model(seq[:, :5], kv_cache=kv, skip_logits=True)]
        for i in range(5, 13, 2):
            parts.append(model(seq[:, i:i + 2], kv_cache=kv, skip_logits=True))
    assert (full - torch.cat(parts, dim=1)).abs().max().item() < 0.1


def test_flops_scale_with_the_block_fraction():
    a = build("p1_discrete").sap_head
    b = build("p1_discrete", sap_block_frac=0.25).sap_head
    assert a.flops_per_token() > 0
    assert b.flops_per_token() == pytest.approx(a.flops_per_token() / 2, rel=0.01)


def test_field_cp_is_position_specific_and_one_plan_draw():
    """A plan must change slots differently; otherwise this collapses to the old broadcast CP."""
    model = build("field_cp", T=4, sap_cp_components=3, sap_field_rank=2)
    head = model.sap_head
    h = torch.randn(2, D)
    with torch.no_grad():
        field = head._field_cp_all(h)
        selected = head._field_cp_selected(h, torch.tensor([0, 2]))
    assert field.shape == (2, 3, 4, D)
    assert torch.allclose(selected[0], field[0, 0])
    assert torch.allclose(selected[1], field[1, 2])
    assert not torch.allclose(field[:, :, 0], field[:, :, 1])


def test_field_energy_is_strict_one_round_and_trains_on_hard_samples(monkeypatch):
    """One shared field draw causes one decoder call; its sample-aware loss backpropagates."""
    model = build("field_energy", T=4, sap_field_rank=3, sap_field_samples=2,
                  sap_field_topk=4, sap_logit_chunk=3)
    head = model.sap_head
    x = torch.randint(0, V, (2, 12), generator=torch.Generator().manual_seed(51))
    y = torch.randint(0, V, (2, 12), generator=torch.Generator().manual_seed(52))
    model.train()
    loss = model(x, y)
    assert torch.isfinite(loss)
    assert {"marginal_nll", "energy_score", "sample_spread"} <= set(model._sap_stats)
    loss.backward()
    assert head.field_noise_in.weight.grad is not None
    assert torch.isfinite(head.field_noise_in.weight.grad).all()

    model.eval()
    b, t = torch.arange(2), torch.full((2,), 6)
    h, ctx, valid = head_inputs(model, x, b, t)
    calls = 0
    original = head._decode

    def tracked(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(head, "_decode", tracked)
    with torch.no_grad():
        sample = head.sample(h, ctx, valid, model._sap_readout, embed=model.transformer.wte,
                             generator=torch.Generator().manual_seed(53))
    assert sample.shape == (2, 4)
    assert calls == 1
    monkeypatch.setattr(head, "_decode", original)
    block = y[b[:, None], t[:, None] + torch.arange(4)[None]]
    with torch.no_grad():
        lp = head.block_logprob(h, ctx, valid, block, model._sap_readout,
                                embed=model.transformer.wte, n_samples=2)
    assert lp.shape == (2,)
    assert torch.isfinite(lp).all()


def test_graph_safe_kv_path_matches_the_standard_path():
    """The CUDA-graph decode path (positions read on the device, attention over the whole
    pre-allocated cache with a mask) computes the same states as the standard KV path, for a
    prefill followed by multi-token and single-token steps."""
    from nanochat.engine import _kv_cache_for
    model = build("indep")
    seq = torch.randint(0, V, (2, 13), generator=torch.Generator().manual_seed(13))
    outs = []
    for graph_safe in (False, True):
        kv = _kv_cache_for(model, 2, 32, torch.device("cpu"), torch.float32)
        kv.graph_safe = graph_safe
        with torch.no_grad():
            parts = [model(seq[:, :5], kv_cache=kv, skip_logits=True)]
            for i, j in ((5, 9), (9, 10), (10, 13)):
                parts.append(model(seq[:, i:j], kv_cache=kv, skip_logits=True))
        outs.append(torch.cat(parts, dim=1))
        assert kv.cache_seqlens.tolist() == [13, 13]
    assert torch.allclose(outs[0], outs[1], atol=1e-5)   # bit-identical on CPU


# ----------------------------------------------------------------------------- SAP v4
# The v4 heads draw their stochastic cut from an exact joint over a top-K candidate lattice
# (nanochat/sap_chain.py) or from corpus tables (nanochat/sap_tables.py). The claims under test:
# every head is a normalised distribution over ALL V^T blocks (the escape state included), the
# sampler draws from that same distribution, slot 0 is the next-token distribution itself, the
# trunk-gradient switch does what it says, and the tables count what they claim to count.

from nanochat.block_head import STRUCT_MODES, V4_MODES  # noqa: E402
from nanochat.sap_chain import (chain_logprob, chain_logz, chain_sample, chain_score,  # noqa: E402
                                chain_viterbi, hmm_loglik, hmm_sample)
from nanochat.sap_tables import seen, tables_from_sequences  # noqa: E402

V4_EXACT = ["lat_crf", "lat_tt", "lat_cp", "cut_crf", "cut_tt", "pmi_chain", "corpus_code"]


def _tables(Vt=V, classes=4):
    seqs = torch.randint(0, Vt, (64, 33), generator=torch.Generator().manual_seed(21))
    return tables_from_sequences(seqs, Vt, top_m=6, pmi_min_count=1, n_classes=classes,
                                 svd_rank=6, kmeans_iters=5)


def build_v4(mode, T=3, **kw):
    """Like build(), with small lattice/rank settings and corpus tables when the mode needs them."""
    opts = dict(sap_block_frac=0.5, sap_head_layers=2, sap_ctx_window=4, sap_lattice_k=3,
                sap_pair_rank=4, sap_tt_rank=3, sap_cp_codes=5, sap_code_classes=4)
    opts.update(kw)
    cfg = GPTConfig(n_layer=2, n_head=1, n_kv_head=1, n_embd=D, vocab_size=V, sequence_len=32,
                    window_pattern="L", sap_block_T=T, sap_block_mode=mode, **opts)
    torch.manual_seed(0)
    model = GPT(cfg)
    if model.sap_head._table_keys():
        model.sap_head.set_tables(_tables(classes=opts["sap_code_classes"]))
    model.init_weights(verify=True)
    # Push every coupling off its zero init, and give the shared readout real logits (nanochat
    # starts lm_head near zero, which would leave every slot uniform and every test trivially
    # passing). The scales keep the block distributions peaked but far from a single block.
    with torch.no_grad():
        for p in model.sap_head.parameters():
            p.add_(torch.randn_like(p) * 0.1)
        model.lm_head.weight.add_(torch.randn_like(model.lm_head.weight) * 0.3)
    model.eval()
    return model


def _all_block_logprobs(model, T):
    x = torch.randint(0, V, (1, 12), generator=torch.Generator().manual_seed(1))
    h, ctx, valid = head_inputs(model, x, torch.zeros(1, dtype=torch.long), torch.tensor([7]))
    blocks = all_blocks(T)
    n = blocks.size(0)
    with torch.no_grad():
        lp = model.sap_head.block_logprob(h.expand(n, -1), ctx.expand(n, -1, -1), valid.expand(n, -1),
                                          blocks, model._sap_readout, embed=model.transformer.wte)
    return lp, (h, ctx, valid), blocks


def test_chain_algorithms_match_enumeration():
    """Forward algorithm, masked marginal, Viterbi and the HMM likelihood against brute force."""
    g = torch.Generator().manual_seed(5)
    N, J, S = 2, 3, 4
    u, p = torch.randn(N, J, S, generator=g), torch.randn(N, J - 1, S, S, generator=g)
    states = torch.tensor(list(itertools.product(range(S), repeat=J)))
    sc = torch.stack([chain_score(u, p, st[None].expand(N, J)) for st in states], 1)
    assert torch.allclose(chain_logz(u, p), torch.logsumexp(sc, 1), atol=1e-5)
    assert torch.equal(chain_viterbi(u, p), states[sc.argmax(1)])
    obs = torch.tensor([[True, False, True]] * N)
    st = states[7][None].expand(N, J)
    marg = torch.logsumexp(sc.view(N, S, S, S)[:, st[0, 0], :, st[0, 2]], -1) - torch.logsumexp(sc, 1)
    assert torch.allclose(chain_logprob(u, p, st, obs), marg, atol=1e-5)
    log_pi = torch.log_softmax(torch.randn(N, 3, generator=g), -1)
    log_A = torch.log_softmax(torch.randn(N, J - 1, 3, 3, generator=g), -1)
    log_emit = torch.log_softmax(torch.randn(N, J, 3, S, generator=g), -1)
    tot = torch.stack([hmm_loglik(log_pi, log_A, log_emit.gather(3, s[None, :, None, None].expand(N, J, 3, 1)).squeeze(-1))
                       for s in states], 1)
    assert torch.allclose(torch.logsumexp(tot, 1), torch.zeros(N), atol=1e-5)


@pytest.mark.parametrize("sampler", ["crf", "hmm"])
def test_chain_samplers_are_exact(sampler):
    g = torch.Generator().manual_seed(6)
    N, J, S, L = 1, 3, 3, 30000
    states = torch.tensor(list(itertools.product(range(S), repeat=J)))
    if sampler == "crf":
        u, p = torch.randn(N, J, S, generator=g), torch.randn(N, J - 1, S, S, generator=g)
        target = torch.softmax(torch.stack([chain_score(u, p, s[None]) for s in states], 1), 1)[0]
        draws = chain_sample(u, p, L, 1.0, torch.Generator().manual_seed(7))
    else:
        log_pi = torch.log_softmax(torch.randn(N, 2, generator=g), -1)
        log_A = torch.log_softmax(torch.randn(N, J - 1, 2, 2, generator=g), -1)
        log_emit = torch.log_softmax(torch.randn(N, J, 2, S, generator=g), -1)
        target = torch.stack([hmm_loglik(log_pi, log_A, log_emit.gather(3, s[None, :, None, None].expand(N, J, 2, 1)).squeeze(-1))
                              for s in states], 1).exp()[0]
        draws = hmm_sample(log_pi, log_A, log_emit, L, 1.0, torch.Generator().manual_seed(7))
    code = (draws[0] * torch.tensor([S * S, S, 1])).sum(-1)
    freq = torch.bincount(code, minlength=S ** J).float() / L
    assert 0.5 * (freq - target).abs().sum().item() < 0.02


@pytest.mark.parametrize("mode", V4_EXACT + ["lat_crf+supp", "cut_crf+pre2"])
def test_v4_heads_are_normalised_over_every_block(mode):
    """Summed over all V^T blocks, including those whose tokens fall outside the lattice and
    reach it only through the escape state, each exact v4 head's probabilities add up to 1."""
    kw = {}
    if mode.endswith("+supp"):
        mode, kw = mode.split("+")[0], dict(sap_supp_min_count=1)
    elif mode.endswith("+pre2"):              # two slot layers before the cut, one fill layer
        mode, kw = mode.split("+")[0], dict(sap_head_layers=3, sap_cut_pre_layers=2)
    lp, _, _ = _all_block_logprobs(build_v4(mode, **kw), 3)
    assert torch.logsumexp(lp, 0).exp().item() == pytest.approx(1.0, abs=1e-4)


@pytest.mark.parametrize("mode", ["lat_crf", "lat_tt", "lat_cp", "cut_crf", "corpus_code"])
def test_v4_unscored_slot_is_summed_out(mode):
    """A block with an unscored (e.g. zero-byte) token keeps an exact likelihood: masking a
    slot equals summing the full joint over that slot's value."""
    model = build_v4(mode)
    lp, (h, ctx, valid), blocks = _all_block_logprobs(model, 3)
    mask = torch.tensor([[True, False, True]])
    y = torch.tensor([[4, 0, 9]])
    with torch.no_grad():
        masked = model.sap_head.block_logprob(h, ctx, valid, y, model._sap_readout,
                                              embed=model.transformer.wte, valid_mask=mask)
    sel = (blocks[:, 0] == 4) & (blocks[:, 2] == 9)
    # cut heads condition the fill on the anchors' tokens, so only the anchor (0, 2) slots of
    # a cut block can be summed out exactly; the middle slot here is a fill and is simply dropped.
    expect = torch.logsumexp(lp[sel], 0)
    if mode == "cut_crf":
        anchors_only = model.sap_head._v4_ll(h, model.sap_head._ctx(ctx), valid, y,
                                             mask, model._sap_readout, model.transformer.wte,
                                             model.transformer.wte.weight)[0]
        expect = anchors_only[0]
    assert masked.item() == pytest.approx(expect.item(), abs=1e-4)


# Multipliers on each head's coupling weights that give the T=2 fixture a measurable total
# correlation (>= 0.09 nats), so a sampler that ignored the coupling would fail the test.
_COUPLING = {"lat_crf": (0.3, ["lat_w0", "lat_w.weight"]),
             "lat_tt": (30.0, ["tt_E", "tt_B", "tt_q.weight", "tt_k.weight"]),
             "lat_cp": (3.0, ["cp_E", "cp_pi.weight"]),
             "cut_crf": (10.0, ["draft_in.weight", "dec.1.attn.c_proj.weight", "dec.1.mlp.c_proj.weight"]),
             "pmi_chain": (30.0, ["pmi_gate.weight"])}


def _couple(model, mode):
    if mode in _COUPLING:
        f, names = _COUPLING[mode]
        with torch.no_grad():
            for n, p in model.sap_head.named_parameters():
                if n in names:
                    p.mul_(f)
    return model


def _total_correlation(lp):
    P = lp.exp().view(V, V)
    indep = P.sum(1)[:, None] * P.sum(0)[None, :]
    return (P * (P.clamp_min(1e-30).log() - indep.clamp_min(1e-30).log())).sum().item()


@pytest.mark.parametrize("mode", ["lat_crf", "lat_tt", "lat_cp", "cut_crf", "cut_tt", "pmi_chain", "corpus_code", "local"])
def test_v4_sampler_draws_from_its_own_likelihood(mode):
    """The decode-time sampler and block_logprob describe the same distribution, including the
    dependence between the slots (where the fixture has enough of it to measure)."""
    model = _couple(build_v4(mode, T=2), mode)
    lp, (h, ctx, valid), blocks = _all_block_logprobs(model, 2)
    if mode in _COUPLING:
        assert _total_correlation(lp) > 0.09
    n = 20000
    with torch.no_grad():
        draws = model.sap_head.sample(h.expand(n, -1), ctx.expand(n, -1, -1), valid.expand(n, -1),
                                      model._sap_readout, embed=model.transformer.wte,
                                      embed_table=model.transformer.wte.weight, temperature=1.0,
                                      generator=torch.Generator().manual_seed(8))
    freq = torch.bincount(draws[:, 0] * V + draws[:, 1], minlength=V * V).float() / n
    assert 0.5 * (freq - lp.exp()).abs().sum().item() < 0.05


@pytest.mark.parametrize("mode", ["lat_crf", "cut_tt", "corpus_code"])
def test_v4_slot0_is_the_next_token_distribution(mode):
    """Slot 0 reads out of the trunk state: with the couplings at zero its marginal under the
    head's joint is exactly the next-token softmax."""
    model = build_v4(mode, T=2)
    head = model.sap_head
    with torch.no_grad():
        for name in ("lat_w", "code_w"):
            if hasattr(head, name):
                getattr(head, name).weight.zero_()
        for name in ("lat_w0", "code_beta"):
            if hasattr(head, name):
                getattr(head, name).zero_()
        if hasattr(head, "tt_E"):           # identical states: the chain carries no dependence
            head.tt_E.zero_()
            head.tt_esc.zero_()
    lp, (h, _, _), blocks = _all_block_logprobs(model, 2)
    marg = torch.zeros(V).index_add_(0, blocks[:, 0], lp.exp())
    ntp = torch.softmax(model._sap_readout(h), -1)[0]
    assert torch.allclose(marg, ntp, atol=1e-5)


@pytest.mark.parametrize("mode", ["cut_crf", "lat_tt"])
def test_trunk_grad_scales_the_block_gradient(mode):
    """sap_trunk_grad=0 leaves trunk, lm_head and wte without block-loss gradient (the trunk is
    the dense model's); 0.1 sends exactly a tenth of the co-trained gradient."""
    x = torch.randint(0, V, (2, 32), generator=torch.Generator().manual_seed(9))
    grads = {}
    for g in (1.0, 0.1, 0.0):
        model = build_v4(mode, sap_trunk_grad=g)
        model.train()
        hid = model(x, skip_logits=True)
        logits = model._sap_readout(hid)    # the trunk's next-token logits, reused for slot 0
        torch.manual_seed(3)                # same block starts in every run
        loss, _ = model._sap_block_loss(hid, x, logits)
        loss.backward()
        trunk = [p for n, p in model.named_parameters() if not n.startswith("sap_head")]
        grads[g] = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).flatten() for p in trunk])
        assert all(p.grad is not None for p in model.sap_head.parameters() if p.requires_grad)
    assert grads[0.0].abs().max().item() == 0.0
    assert grads[1.0].abs().max().item() > 0.0
    # The trunk runs its activations in bf16, so the backward pass rounds the scaled gradient
    # at ~3 significant digits; compare the whole gradient vector, not element by element.
    rel = (grads[0.1] - 0.1 * grads[1.0]).norm() / (0.1 * grads[1.0]).norm()
    assert rel.item() < 2e-2


def test_corpus_tables_count_what_they_claim():
    seqs = torch.tensor([[0, 1, 2, 1, 2, 3], [1, 2, 3, 0, 1, 2]])
    t = tables_from_sequences(seqs, 5, top_m=3, pmi_min_count=1, n_classes=2, svd_rank=2, kmeans_iters=3)
    pairs = {(int(k) // 5, int(k) % 5): int(c) for k, c in zip(t["supp_keys"], t["supp_counts"])}
    assert pairs == {(0, 1): 2, (1, 2): 4, (2, 1): 1, (2, 3): 2, (3, 0): 1}
    skips = {(int(k) // 5, int(k) % 5): int(c) for k, c in zip(t["supp2_keys"], t["supp2_counts"])}
    assert skips[(1, 1)] == 1 and skips[(2, 0)] == 1 and sum(skips.values()) == 8
    # p(b | a=1): only 2 follows 1
    assert int(t["big_ids"][1, 0]) == 2 and float(t["big_p"][1, 0]) == pytest.approx(1.0)
    # PMI(0, 1) = log(c(0,1) N / (c(0.) c(.1))) = log(2 * 10 / (2 * 3))
    assert float(t["pmi_vals"][0, 0]) == pytest.approx(math.log(20 / 6), abs=1e-2)
    assert seen(t["supp_keys"], torch.tensor([0, 1, 3]), torch.tensor([1, 0, 0]), 5).tolist() == [True, False, True]


def test_self_contrastive_resampler_targets_q_times_exp_phi():
    """With many proposals, importance resampling draws from p ∝ q·exp(phi)."""
    model = _couple(build_v4("lat_crf", T=2, sap_nce_props=256), "lat_crf")
    head = model.sap_head
    with torch.no_grad():
        head.nce_out.weight.mul_(5.0)           # a scorer that moves q by TV ~0.6
    lp_q, (h, ctx, valid), blocks = _all_block_logprobs(model, 2)
    with torch.no_grad():
        phi = head._nce_scores(blocks[None], h, head._ctx(ctx), valid, model.transformer.wte)[0]
        target = torch.softmax(lp_q + phi, 0)
        n = 4000
        draws = head.sample(h.expand(n, -1), ctx.expand(n, -1, -1), valid.expand(n, -1),
                            model._sap_readout, embed=model.transformer.wte,
                            embed_table=model.transformer.wte.weight, temperature=1.0,
                            generator=torch.Generator().manual_seed(10))
    freq = torch.bincount(draws[:, 0] * V + draws[:, 1], minlength=V * V).float() / n
    plain = lp_q.exp()
    assert 0.5 * (freq - target).abs().sum().item() < 0.08
    assert 0.5 * (target - plain).abs().sum().item() > 0.3      # phi actually reshapes q


@pytest.mark.parametrize("mode", ["lat_crf", "cut_tt", "lat_cp", "pmi_chain", "corpus_code", "p1_selfpost"])
def test_v4_training_step_is_finite(mode):
    kw = dict(sap_nce_props=4, sap_supp_min_count=1, sap_soft_eps=0.1) if mode == "lat_crf" else {}
    model = build_v4(mode, **kw)
    model.train()
    x = torch.randint(0, V, (2, 32), generator=torch.Generator().manual_seed(11))
    loss = model(x, x)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.sap_head.parameters() if p.grad is not None)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("mode", ["cut_crf", "lat_tt"])
def test_v4_greedy_sampler_captures_under_cuda_graphs(mode):
    """The decode benchmark replays the head inside a CUDA graph: no host sync, fixed shapes."""
    model = build_v4(mode, T=4).cuda()
    x = torch.randint(0, V, (2, 12), device="cuda")
    head = model.sap_head
    with torch.no_grad():
        hid = model(x, skip_logits=True)
        h, ctx, valid = model._sap_last(hid, hid[:, -1].contiguous())
        kw = dict(embed=model.transformer.wte, embed_table=model.transformer.wte.weight, temperature=0.0)
        eager = head.sample(h, ctx, valid, model._sap_readout, **kw)
        out = torch.empty_like(eager)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                out.copy_(head.sample(h, ctx, valid, model._sap_readout, **kw))
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out.copy_(head.sample(h, ctx, valid, model._sap_readout, **kw))
        out.zero_()
        graph.replay()
        torch.cuda.synchronize()
    assert torch.equal(out, eager)


@pytest.mark.parametrize("mode", ["lat_crf", "cut_tt"])
def test_slot0_reuses_the_trunk_logits_exactly(mode):
    """Training passes the trunk's next-token logits as slot 0's readout (T-1 readouts per
    block instead of T); the loss must equal recomputing slot 0 from the trunk state."""
    model = build_v4(mode)
    head = model.sap_head
    x = torch.randint(0, V, (2, 32), generator=torch.Generator().manual_seed(15))
    with torch.no_grad():
        hid = model(x, skip_logits=True)
        b, t = torch.tensor([0, 1, 1]), torch.tensor([3, 9, 20])
        y = x[b[:, None], t[:, None] + 1 + torch.arange(3)[None]]
        h = hid[b, t]
        ctx, valid = model._sap_window(hid, b, t)
        a, _ = head.loss(h, ctx, valid, y, model._sap_readout, embed=model.transformer.wte)
        c, _ = head.loss(h, ctx, valid, y, model._sap_readout, embed=model.transformer.wte,
                         slot0_logits=model._sap_readout(h))
    assert a.item() == pytest.approx(c.item(), abs=1e-5)


# ----------------------------------------------------------------------------- depth_local
def _depth_model(m, share, T=4, layers=3, mode="depth_local", bottom=0):
    cfg = GPTConfig(n_layer=layers, n_head=2, n_kv_head=2, n_embd=D, vocab_size=V, sequence_len=32,
                    window_pattern="L", sap_block_T=T, sap_block_mode=mode, sap_block_frac=0.25,
                    sap_depth_layers=m, sap_depth_share=share, sap_depth_bottom=bottom)
    torch.manual_seed(0)
    model = GPT(cfg)
    model.init_weights(verify=True)
    with torch.no_grad():
        for blk in model.transformer.h:
            for p in blk.parameters():
                p.add_(torch.randn_like(p) * 0.2)
        model.lm_head.weight.add_(torch.randn_like(model.lm_head.weight) * 0.3)
    model.sap_sync_depth_copies()
    return model.eval()


def _depth_ll(model, x, starts, T=4):
    with torch.no_grad():
        hid, states = model(x, skip_logits=True, sap_capture=True)
        logits = model._sap_readout(hid)
        tgt = torch.cat([x[:, 1:], x[:, :1]], 1)
        b = torch.arange(x.size(0))[:, None]
        y = torch.stack([tgt[b, starts + k] for k in range(T)], -1)
        slot0 = logits[b, starts]
        ll = model._sap_depth_logprob(x, states, slot0, y, starts, torch.ones_like(y, dtype=torch.bool))
        lp = torch.log_softmax(logits.float(), -1)
        ar = sum(lp[b, starts + k, y[..., k]] for k in range(T))
    return ll, ar


@pytest.mark.parametrize("share", [0, 1])
def test_depth_local_with_every_layer_is_the_trunk_likelihood(share):
    """With all L layers and the trunk's weights, a slot is exactly the trunk at its position:
    the block likelihood equals the trunk's teacher-forced one (bf16 attention noise aside)."""
    model = _depth_model(3, share)
    x = torch.randint(0, V, (2, 24), generator=torch.Generator().manual_seed(1))
    ll, ar = _depth_ll(model, x, torch.tensor([[3, 10], [5, 15]]))
    assert (ll - ar).abs().max().item() < 0.1


def test_depth_local_reads_no_input_after_the_block_start():
    """Slots read trunk keys at positions <= t and the block's own tokens only: changing every
    input token after t leaves the block likelihood unchanged."""
    model = _depth_model(2, 0)
    x = torch.randint(0, V, (1, 24), generator=torch.Generator().manual_seed(2))
    starts = torch.tensor([[9]])
    with torch.no_grad():
        hid, states = model(x, skip_logits=True, sap_capture=True)
        y = torch.tensor([[[4, 7, 1, 9]]])
        slot0 = model._sap_readout(hid)[:, 9][:, None]
        a = model._sap_depth_logprob(x, states, slot0, y, starts, torch.ones_like(y, dtype=torch.bool))
        x2 = x.clone()
        x2[:, 10:] = torch.randint(0, V, (1, 14), generator=torch.Generator().manual_seed(3))
        hid2, states2 = model(x2, skip_logits=True, sap_capture=True)
        b = model._sap_depth_logprob(x2, states2, model._sap_readout(hid2)[:, 9][:, None], y, starts,
                                     torch.ones_like(y, dtype=torch.bool))
    assert a.item() == pytest.approx(b.item(), abs=1e-4)


@pytest.mark.parametrize("share", [0, 1])
def test_depth_local_decode_with_every_layer_is_ar_decoding(share):
    """The timed depth_local decoder (single-token passes through the top layers reading the
    trunk's KV cache) emits exactly the AR greedy tokens when it runs every layer."""
    from nanochat.engine import generate_ar_kv
    model = _depth_model(3, share)
    prompt = torch.randint(0, V, (2, 10), generator=torch.Generator().manual_seed(5))
    a = generate_block_kv(model, prompt, 12, temperature=0.0)
    b = generate_ar_kv(model, prompt, 12, temperature=0.0)
    assert torch.equal(a, b)


def test_depth_local_decode_matches_training_conditionals_with_trained_copies():
    """With copies that have moved away from the trunk (as after training), each greedy slot
    token of the timed decoder is the argmax of the training path's conditional for that slot.
    (Reading the trunk's KV cache instead of the copies' own keys would break this.)"""
    model = _depth_model(2, 0, layers=3)
    with torch.no_grad():
        for p in model.sap_head.depth_blocks.parameters():
            p.add_(torch.randn_like(p) * 0.3)
    prompt = torch.randint(0, V, (1, 10), generator=torch.Generator().manual_seed(6))
    blk = generate_block_kv(model, prompt, 4, temperature=0.0)            # one block, T=4
    with torch.no_grad():
        hid, states = model(prompt, skip_logits=True, sap_capture=True)
        t = torch.tensor([[prompt.size(1) - 1]])
        slot0 = model._sap_readout(hid)[:, -1][:, None]
        y = blk[:, None, :]                                               # (1, 1, T) teacher-forced
        logits = model._sap_depth_logprob(prompt, states, slot0, y, t, torch.ones_like(y, dtype=torch.bool),
                                          return_slot_logits=True)       # (1, 1, T-1, V)
    assert blk[0, 0].item() == slot0[0, 0].argmax().item()
    assert torch.equal(logits[0, 0].argmax(-1), blk[0, 1:])


@pytest.mark.parametrize("mode", DEPTH_MODES)
def test_depth_trunk_grad_scales_the_shared_trunk_gradient(mode):
    """With the trunk's own top layers shared, sap_trunk_grad scales every block-loss gradient
    that reaches a trunk-owned tensor (blocks, embeddings, scalars, lm_head); 0 leaves none."""
    x = torch.randint(0, V, (2, 32), generator=torch.Generator().manual_seed(16))
    grads = {}
    for g in (1.0, 0.1, 0.0):
        cfg = GPTConfig(n_layer=3, n_head=2, n_kv_head=2, n_embd=D, vocab_size=V, sequence_len=32,
                        window_pattern="L", sap_block_T=4, sap_block_mode=mode, sap_block_frac=0.25,
                        sap_depth_layers=2, sap_depth_share=1, sap_trunk_grad=g)
        torch.manual_seed(0)
        model = GPT(cfg)
        model.init_weights(verify=True)
        model.train()
        hid, states = model(x, skip_logits=True, sap_capture=True)
        logits = model._sap_readout(hid)
        torch.manual_seed(3)
        loss, _ = model._sap_depth_loss(x, hid, x, logits, states)
        loss.backward()
        trunk = [p for n, p in model.named_parameters() if not n.startswith("sap_head")]
        grads[g] = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).flatten() for p in trunk])
    assert grads[0.0].abs().max().item() == 0.0
    assert grads[1.0].abs().max().item() > 0.0
    rel = (grads[0.1] - 0.1 * grads[1.0]).norm() / (0.1 * grads[1.0]).norm()
    assert rel.item() < 2e-2


def _perturb_head(model, scale=0.3):
    """Move every slot-path weight (copies, entry, offsets, mask) away from its init."""
    with torch.no_grad():
        for p in model.sap_head.parameters():
            p.add_(torch.randn_like(p) * scale)


@pytest.mark.parametrize("share", [0, 1])
def test_depth_tree_is_a_normalised_block_distribution(share):
    """Each bisection round reads only tokens committed in earlier rounds (mask slots carry no
    token, not even through the value embeddings), so p(y_2..y_4 | ctx, y_1) sums to one over
    all V^3 continuations."""
    model = _depth_model(2, share, mode="depth_tree")
    _perturb_head(model)
    model = model.to(torch.float32)      # bf16 rotary tables vary with the batch layout at ~1e-4
    x = torch.randint(0, V, (1, 20), generator=torch.Generator().manual_seed(7))
    t = 11
    rest = torch.tensor(list(itertools.product(range(V), repeat=3)))
    with torch.no_grad():
        hid, states = model(x, skip_logits=True, sap_capture=True)
        slot0 = model._sap_readout(hid)[:, t]                                # (1, V)
        for y1 in (2, 9):
            lls = []
            for c in rest.split(512):
                y = torch.cat([torch.full((len(c), 1), y1), c], 1)[None]     # (1, n, 4)
                n = y.size(1)
                lls.append(model._sap_depth_logprob(x, states, slot0[:, None].expand(1, n, V), y,
                                                    torch.full((1, n), t), torch.ones_like(y, dtype=torch.bool))[0])
            lp0 = torch.log_softmax(slot0[0].float(), -1)[y1]
            total = torch.logsumexp(torch.cat(lls).double() - lp0.double(), 0)
            assert total.item() == pytest.approx(0.0, abs=1e-5)


@pytest.mark.parametrize("share", [0, 1])
def test_depth_tree_decode_matches_training_conditionals(share):
    """Each greedy token of the timed round decoder is the argmax of the training path's
    conditional for that token, given the block's earlier rounds."""
    model = _depth_model(2, share, mode="depth_tree")
    _perturb_head(model)
    prompt = torch.randint(0, V, (2, 10), generator=torch.Generator().manual_seed(8))
    blk = generate_block_kv(model, prompt, 4, temperature=0.0)            # one block, T=4
    with torch.no_grad():
        hid, states = model(prompt, skip_logits=True, sap_capture=True)
        t = torch.full((2, 1), prompt.size(1) - 1)
        slot0 = model._sap_readout(hid)[:, -1][:, None]
        y = blk[:, None, :]
        logits = model._sap_depth_logprob(prompt, states, slot0, y, t, torch.ones_like(y, dtype=torch.bool),
                                          return_slot_logits=True)       # (2, 1, T-1, V)
    assert torch.equal(slot0[:, 0].argmax(-1), blk[:, 0])
    assert torch.equal(logits[:, 0].argmax(-1), blk[:, 1:])



def test_depth_roll_with_zero_feedback_is_depth_local():
    """depth_roll only adds the previous slot's top state through depth_fb (zero at init): with
    it at zero, the sequential slots give depth_local's likelihood exactly."""
    model = _depth_model(2, 0, mode="depth_roll")
    _perturb_head(model)
    with torch.no_grad():
        model.sap_head.depth_fb.weight.zero_()
    x = torch.randint(0, V, (2, 24), generator=torch.Generator().manual_seed(9))
    starts = torch.tensor([[3, 10], [5, 15]])
    roll, _ = _depth_ll(model, x, starts)
    model.sap_head.mode = "depth_local"
    local, _ = _depth_ll(model, x, starts)
    assert (roll - local).abs().max().item() < 1e-4


def test_depth_roll_slot_reads_only_the_block_so_far():
    """Slot k's logits depend on y_1..y_k only, even though its feedback comes from slot k-1's
    top state: changing y_3 moves slot 3's logits and leaves slots 1 and 2 unchanged."""
    model = _depth_model(2, 1, mode="depth_roll")
    _perturb_head(model)
    x = torch.randint(0, V, (1, 24), generator=torch.Generator().manual_seed(10))
    starts = torch.tensor([[9]])
    with torch.no_grad():
        hid, states = model(x, skip_logits=True, sap_capture=True)
        slot0 = model._sap_readout(hid)[:, 9][:, None]
        out = []
        for y3 in (2, 11):
            y = torch.tensor([[[4, 7, y3, 9]]])
            out.append(model._sap_depth_logprob(x, states, slot0, y, starts, torch.ones_like(y, dtype=torch.bool),
                                                return_slot_logits=True)[0, 0])     # (T-1, V)
    assert torch.allclose(out[0][:2], out[1][:2], atol=1e-5)
    assert (out[0][2] - out[1][2]).abs().max().item() > 1e-3


@pytest.mark.parametrize("share", [0, 1])
def test_depth_roll_decode_matches_training_conditionals(share):
    """The timed decoder feeds each slot the previous slot's top state exactly as training does:
    every greedy token is the argmax of the training path's conditional."""
    model = _depth_model(2, share, mode="depth_roll")
    _perturb_head(model)
    prompt = torch.randint(0, V, (2, 10), generator=torch.Generator().manual_seed(12))
    blk = generate_block_kv(model, prompt, 4, temperature=0.0)
    with torch.no_grad():
        hid, states = model(prompt, skip_logits=True, sap_capture=True)
        t = torch.full((2, 1), prompt.size(1) - 1)
        slot0 = model._sap_readout(hid)[:, -1][:, None]
        y = blk[:, None, :]
        logits = model._sap_depth_logprob(prompt, states, slot0, y, t, torch.ones_like(y, dtype=torch.bool),
                                          return_slot_logits=True)
    assert torch.equal(slot0[:, 0].argmax(-1), blk[:, 0])
    assert torch.equal(logits[:, 0].argmax(-1), blk[:, 1:])


@pytest.mark.parametrize("mode", ["depth_local", "depth_roll"])
def test_skip_middle_with_no_skipped_layer_is_the_trunk_likelihood(mode):
    """Skip-middle slots run the bottom layers exactly as the trunk and add the block start's
    middle-layer delta before the top ones. With nothing skipped (bottom 1 + top 2 of 3 layers)
    that delta is zero, so the block likelihood is the trunk's teacher-forced one."""
    model = _depth_model(3, 0, mode=mode, bottom=1)
    assert model.sap_head.depth_layer_ids == [0, 1, 2]
    x = torch.randint(0, V, (2, 24), generator=torch.Generator().manual_seed(13))
    ll, ar = _depth_ll(model, x, torch.tensor([[3, 10], [5, 15]]))
    assert (ll - ar).abs().max().item() < 0.1


@pytest.mark.parametrize("mode", ["depth_local", "depth_roll"])
@pytest.mark.parametrize("share", [0, 1])
def test_skip_middle_decode_matches_training_conditionals(mode, share):
    """Layers {0, 3} of 4 (the middle two skipped): the timed decoder adds the same middle delta
    at the same layer as training, so every greedy token is the training conditional's argmax."""
    model = _depth_model(2, share, layers=4, mode=mode, bottom=1)
    assert model.sap_head.depth_layer_ids == [0, 3]
    _perturb_head(model)
    prompt = torch.randint(0, V, (2, 10), generator=torch.Generator().manual_seed(14))
    blk = generate_block_kv(model, prompt, 4, temperature=0.0)
    with torch.no_grad():
        hid, states = model(prompt, skip_logits=True, sap_capture=True)
        t = torch.full((2, 1), prompt.size(1) - 1)
        slot0 = model._sap_readout(hid)[:, -1][:, None]
        y = blk[:, None, :]
        logits = model._sap_depth_logprob(prompt, states, slot0, y, t, torch.ones_like(y, dtype=torch.bool),
                                          return_slot_logits=True)
    assert torch.equal(slot0[:, 0].argmax(-1), blk[:, 0])
    assert torch.equal(logits[:, 0].argmax(-1), blk[:, 1:])


def test_skip_middle_reads_no_input_after_the_block_start():
    """The middle delta comes from the block start t: inputs after t do not move the block."""
    model = _depth_model(2, 0, layers=4, bottom=1)
    _perturb_head(model)
    x = torch.randint(0, V, (1, 24), generator=torch.Generator().manual_seed(15))
    starts = torch.tensor([[9]])
    y = torch.tensor([[[4, 7, 1, 9]]])
    out = []
    with torch.no_grad():
        for seed in (None, 16):
            xx = x.clone()
            if seed is not None:
                xx[:, 10:] = torch.randint(0, V, (1, 14), generator=torch.Generator().manual_seed(seed))
            hid, states = model(xx, skip_logits=True, sap_capture=True)
            out.append(model._sap_depth_logprob(xx, states, model._sap_readout(hid)[:, 9][:, None], y, starts,
                                                torch.ones_like(y, dtype=torch.bool)))
    assert out[0].item() == pytest.approx(out[1].item(), abs=1e-4)

