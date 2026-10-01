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

from nanochat.block_head import (SAP_MODES, energy_score, pick, target_bins)
from nanochat.engine import generate_block_kv
from nanochat.gpt import GPT, GPTConfig

V = 16      # tiny, so whole block distributions can be enumerated
D = 64


def build(mode, T=2, **kw):
    opts = dict(sap_block_frac=0.5, sap_head_layers=1, sap_ctx_window=4, sap_kl_anneal_steps=0,
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


@pytest.mark.parametrize("mode", ["indep", "local", "cp"])
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


def test_discrete_plan_bound_is_a_bound_and_tightens():
    """The importance-weighted estimate never exceeds the exact log-likelihood on average,
    and more samples move it up towards it."""
    model = build("p1_discrete")
    x = torch.randint(0, V, (1, 12), generator=torch.Generator().manual_seed(2))
    h, ctx, valid = head_inputs(model, x, torch.zeros(1, dtype=torch.long), torch.tensor([7]))
    y = torch.tensor([[3, 11]])
    with torch.no_grad():
        exact = _exact_p1_logprob(model, h, ctx, valid, y).item()
        torch.manual_seed(0)
        k1 = torch.stack([model.sap_head.block_logprob(h, ctx, valid, y, model._sap_readout,
                                                       embed=model.transformer.wte, n_samples=1)
                          for _ in range(400)]).mean().item()
        k64 = torch.stack([model.sap_head.block_logprob(h, ctx, valid, y, model._sap_readout,
                                                        embed=model.transformer.wte, n_samples=64)
                           for _ in range(50)]).mean().item()
    assert k1 <= exact + 1e-3
    assert k64 <= exact + 1e-3
    assert k64 >= k1 - 1e-3


@pytest.mark.parametrize("mode", ["p1_discrete", "p2_gauss"])
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


@pytest.mark.parametrize("mode", list(SAP_MODES))
def test_kv_block_decode_loop_matches_reference(mode):
    """The decode loop that gets timed emits exactly the reference loop's blocks."""
    model = build(mode)
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
