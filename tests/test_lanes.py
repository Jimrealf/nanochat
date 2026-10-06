"""Lanes (nanochat/lanes.py): the lane order is an exact factorisation, one lane is the ordinary
causal model, and the timed decoder draws from the same conditionals training scores."""
import itertools

import pytest
import torch

from nanochat.gpt import GPT, GPTConfig
from nanochat.lanes import generate_lanes, infill_layout, infill_rows, lane_inputs, lane_layout, lane_mask, lane_rank


def _model(V=16, layers=2, seq=32):
    cfg = GPTConfig(n_layer=layers, n_head=2, n_kv_head=2, n_embd=64, vocab_size=V, sequence_len=seq,
                    window_pattern="L")
    torch.manual_seed(0)
    model = GPT(cfg)
    model.init_weights(verify=True)
    with torch.no_grad():                    # c_proj starts at zero; move every weight off its init
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.2)
    return model.eval()


def test_lane_layout_and_rank():
    """Prefix positions keep their causal order; lane positions share one step per column."""
    S, starts = lane_layout(11, 5, 2)
    assert S == 3 and starts == [8]
    assert lane_rank(11, 5, 2).tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 5, 6, 7]
    m = lane_mask(11, 5, 2)[0, 0]
    assert m[8, :5].all() and m[8, 5] and not m[8, 6]   # lane 1 step 0: prefix and the step-0 inputs
    assert m[9, 6] and m[9, 9] and not m[9, 7]          # step 1 reads steps 0-1, not step 2
    assert not m[4, 5]                                  # the prefix stays causal


def test_one_lane_is_the_causal_model():
    """With a single lane the mask is causal and nothing is swapped: the ordinary forward."""
    model = _model()
    g = torch.Generator().manual_seed(1)
    x = torch.randint(0, 16, (2, 24), generator=g)
    y = torch.randint(0, 16, (2, 24), generator=g)
    with torch.no_grad():
        a = model(x, y, loss_reduction='none')
        b = model(lane_inputs(x, 6, 1, 15), y, loss_reduction='none', lane_mask=lane_mask(24, 6, 1))
    assert (a - b).abs().max().item() < 1e-2


def _lane_total(model, mask, V=4, P=2, L=2, N=8):
    seqs = torch.tensor(list(itertools.product(range(V), repeat=N - P + 1)))   # t_2 .. t_8
    full = torch.cat([torch.tensor([1, 2]).expand(seqs.size(0), 2), seqs], dim=1)
    x, y = full[:, :N], full[:, 1:].clone()
    y[:, 0] = -1                                    # t_1 is part of the fixed prefix
    lls = []
    with torch.no_grad():
        for xs, ys in zip(lane_inputs(x, P, L, V - 1).split(2048), y.split(2048)):
            nll = model(xs, ys, loss_reduction="none", lane_mask=mask).view(ys.shape)
            lls.append(-(nll * (ys >= 0)).sum(-1))
    return torch.logsumexp(torch.cat(lls).double(), 0).item()


def test_lane_order_is_a_normalised_distribution():
    """Enumerate every continuation after a two-token prefix (V=4, two lanes of three): the
    lane-order probabilities sum to one, so no target reads itself or a later draw. Reading
    same-step inputs is allowed (they were drawn the step before); letting a lane read its own
    next input breaks the sum, which shows the check has power."""
    model = _model(V=4, seq=16)
    mask = lane_mask(8, 2, 2)
    assert _lane_total(model, mask) == pytest.approx(0.0, abs=1e-3)       # measured ~1e-7
    leak = mask.clone()
    leak[0, 0, 2, 3] = True            # lane 0 step 0 reads its own step-1 input: its own target
    assert abs(_lane_total(model, leak)) > 5e-3                            # measured ~2e-2


def test_lane_decoder_matches_training_conditionals():
    """Greedy lane decoding (prompt pass, then one all-layer pass per lockstep step through the
    KV cache) emits the argmax of the lane-masked training forward at every drawn position."""
    model = _model()
    P, L, S = 5, 2, 3
    prompt = torch.randint(0, 15, (2, P), generator=torch.Generator().manual_seed(3))
    out = generate_lanes(model, prompt, L, S, lane_token=15, temperature=0.0)
    full = torch.cat([prompt, out], dim=1)                                     # P + 1 + L*S tokens
    N = P + L * S
    x = lane_inputs(full[:, :N], P, L, 15)
    with torch.no_grad():
        logits = model(x, lane_mask=lane_mask(N, P, L))
    assert torch.equal(logits[:, P - 1:].argmax(-1), full[:, P:])


def test_separator_slots_are_the_only_route_across_the_split():
    """S11 separator oracle: with no slots the second half is blind to the first half; with slots
    the first half reaches it, and only the slots' and first second-half targets are dropped."""
    from nanochat.lanes import separator_batch, separator_mask
    model = _model(seq=32)
    g = torch.Generator().manual_seed(3)
    x = torch.randint(0, 15, (4, 24), generator=g)
    y = torch.randint(0, 15, (4, 24), generator=g)
    x2 = x.clone()
    x2[:, :12] = torch.randint(0, 15, (4, 12), generator=g)          # change only the first half
    P = 12
    with torch.no_grad():
        for m, leaks in ((0, False), (2, True)):
            xa, ya = separator_batch(x, y, P, m, 15)
            xb, _ = separator_batch(x2, y, P, m, 15)
            mask = separator_mask(24, P, m)
            la = model(xa, ya, loss_reduction="none", lane_mask=mask).view(ya.shape)
            lb = model(xb, ya, loss_reduction="none", lane_mask=mask).view(ya.shape)
            diff = (la - lb)[:, P + m:].abs().max().item()
            assert (diff > 1e-3) if leaks else (diff < 1e-5)
    xm, ym = separator_batch(x, y, P, 2, 15)
    assert (xm[:, P:P + 2] == 15).all() and torch.equal(xm[:, P + 2:], x[:, P:-2])          # shifted, not replaced
    assert (ym[:, P - 1:P + 2] == -1).all() and torch.equal(ym[:, :P - 1], y[:, :P - 1])
    assert torch.equal(ym[:, P + 2:], y[:, P:-2])


def test_sentence_aligned_lanes_cut_at_sentence_ends_and_predict_each_text_token_once():
    """S12 S-1: lanes are cut at the last sentence end in their window (else when full), the rest of
    a cut lane is padding, and every placed text token is predicted exactly once, in text order."""
    from nanochat.lanes import aligned_lanes_rows, lane_rank
    P, L, S = 3, 3, 6
    N = P + L * S
    ends = torch.zeros(20, dtype=torch.bool)
    ends[5] = True                                                   # token 5 ends a sentence
    g = torch.Generator().manual_seed(0)
    raw = torch.randint(6, 17, (2, N + 1), generator=g)
    raw[0, [6, 12, 15]] = 5                                          # sentence ends inside row 0
    xa, ya, placed, aligned = aligned_lanes_rows(raw[:, :-1], raw[:, 1:], P, L, 18, 19, ends, 0, window=4)
    rank = lane_rank(N, P, L)
    for b in range(2):
        n = int(placed[b])
        text_targets = [(int(rank[i]), i, int(t)) for i, t in enumerate(ya[b].tolist()) if 0 <= t < 18 and i >= P - 1]
        got = [t for _, _, t in sorted(text_targets, key=lambda z: z[1])]
        assert sorted(got) == sorted(raw[b, P:P + n].tolist())       # each placed text token predicted once
        lanes_text = [t for t in xa[b, P:].tolist() if t < 18]       # inputs in slot order, minus lane/pad
        assert lanes_text == raw[b, P:P + n].tolist()                 # slots hold the text in order
    assert int(aligned[0]) >= 1 and int(placed[0]) < L * S             # row 0 has cuts, hence padding
    assert (ya[0] == 19).any() and (xa[0] == 19).any()
    assert int(placed[1]) == L * S - (L - 1)                         # no sentence ends: plain-lane cuts, full lanes


def test_lagged_lane_mask_hides_other_lanes_recent_steps_only():
    from nanochat.lanes import lagged_lane_mask, lane_mask, lane_rank
    N, P, L = 2 + 12, 2, 3                                   # three 4-step lanes after a 2-token prefix
    assert torch.equal(lagged_lane_mask(N, P, L, 0), lane_mask(N, P, L))
    m = lagged_lane_mask(N, P, L, 2)[0, 0]
    r = lane_rank(N, P, L)
    q = P + 4 + 3                                            # lane 1, step 3
    assert m[q, :P].all()                                    # the prefix in full
    assert m[q, P + 4:q + 1].all()                           # its own lane up to its step
    assert m[q, P + 0] and m[q, P + 1] and not m[q, P + 2]   # lane 0 up to step 3 - 2 = 1
    assert not m[q, P + 8 + 2] and m[q, P + 8 + 1]           # lane 2 likewise
    assert not (m & (r[None, :] > r[:, None])).any()         # never a later step


def test_lane_offset_report_gives_absolute_nats_per_lane():
    from scripts.sap_position_bpb import lane_offset_report
    N, P, L, R = 12, 4, 2, 3                   # lanes of S = 4 at input positions 4..7 and 8..11; 3 rows
    ref = torch.full((N,), 2.0 * R, dtype=torch.float64)
    own = ref.clone()
    own[8] += 3.0 * R                          # lane 1's first prediction: 3 nats more per row
    own[11] -= 1.0 * R                         # its last one (the junction): 1 nat less
    rep = lane_offset_report(N, P, L, (own, ref), (ref, ref), n_rows=R)
    assert rep["extra nats per lane"] == pytest.approx(2.0) and rep["reference nats per token"] == pytest.approx(2.0)
    assert rep["0"] == pytest.approx(2.5) and rep["share of the lanes' extra nats at offset 0"] == pytest.approx(1.5)
    assert rep["deficit nats per lane"] == pytest.approx(3.0) and rep["recovery nats per lane"] == pytest.approx(-1.0)
    assert rep["extra nats per lane by offset"] == pytest.approx([3.0, 0.0, 0.0, -1.0])



def test_infill_layout_reads_only_earlier_draws_except_at_cold_slots():
    """S16-A: perm lists every slot once (outside slots, then the middles, each in position order);
    one middle of lo..hi slots per chunk, with a slot outside on either side inside the chunk; and
    the only slots whose input is drawn later in perm order are the cold slots right after each
    middle (their input is the middle's last target), which hold the lane-start token."""
    from nanochat.lanes import infill_layout
    N, lo, hi, gap, first = 2048, 2, 64, 128, 64
    for seed in range(5):
        perm, cold = infill_layout(N, lo, hi, gap, first, generator=torch.Generator().manual_seed(seed))
        assert sorted(perm.tolist()) == list(range(N))
        down = (perm[1:] < perm[:-1]).nonzero().flatten()
        assert down.numel() == 1                                     # two increasing runs: outside, middles
        mid = torch.zeros(N, dtype=torch.bool)
        mid[perm[int(down) + 1:]] = True
        starts = (mid[1:] & ~mid[:-1]).nonzero().flatten() + 1
        assert torch.equal(cold, (mid[:-1] & ~mid[1:]).nonzero().flatten() + 1)
        c = first + gap * torch.arange(starts.numel())
        assert starts.numel() == len(range(first, N, gap))           # every chunk has room at N = 2048
        assert ((cold - starts >= lo) & (cold - starts <= hi)).all()
        assert (starts > c).all() and (cold < torch.clamp(c + gap, max=N)).all()
        rank = torch.empty(N, dtype=torch.long)
        rank[perm] = torch.arange(N)
        later = rank[:-1] > rank[1:]                                 # slot k's input t_k is slot k-1's target
        is_cold = torch.zeros(N, dtype=torch.bool)
        is_cold[cold] = True
        assert torch.equal(later, is_cold[1:])


def _infill_total(model, perm, cold, V=4, N=8):
    seqs = torch.tensor(list(itertools.product(range(V), repeat=N - 1)))  # t_2 .. t_8 after t_0 t_1 = 1 2
    full = torch.cat([torch.tensor([1, 2]).expand(seqs.size(0), 2), seqs], dim=1)
    x, y = full[:, :N], full[:, 1:].clone()
    y[:, 0] = -1                                    # t_1 is part of the fixed prefix
    causal = torch.ones(N, N, dtype=torch.bool).tril()[None, None]
    lls = []
    with torch.no_grad():
        for xs, ys in zip(x.split(2048), y.split(2048)):
            xi, yi, pid = infill_rows(xs, ys, perm, cold, V - 1)
            nll = model(xi, yi, loss_reduction="none", lane_mask=causal, pos_ids=pid).view(yi.shape)
            lls.append(-(nll * (yi >= 0)).sum(-1))
    return torch.logsumexp(torch.cat(lls).double(), 0).item()


def test_infill_rows_are_a_normalised_distribution():
    """Enumerate every continuation (V=4, eight slots, one middle) run in infill order with true
    rotary positions and a causal mask over the reordered row: the probabilities sum to one. Leaving
    the cold slot's input in place (it is the middle's last target) breaks the sum."""
    model = _model(V=4, seq=16)
    perm, cold = infill_layout(8, lo=2, hi=3, gap=8, first=1, generator=torch.Generator().manual_seed(0))
    assert cold.numel() == 1 and perm.tolist() != list(range(8))
    assert _infill_total(model, perm, cold) == pytest.approx(0.0, abs=1e-3)      # measured ~1e-8
    assert abs(_infill_total(model, perm, cold[:0])) > 5e-3                       # measured ~0.1


def test_lrb_normalized_distribution_and_gradients():
    """S16-C: With learned lane-relative attention bias and input offset embeddings,
    the model remains an exact normalized probability distribution (logsumexp == 0.0),
    and loss gradients cleanly backpropagate into both LRB and offset parameters."""
    from nanochat.lanes import compute_lrb_buckets, deduce_lane_layout

    # 1. Bucket and layout deduction checks
    m = lane_mask(32, 8, 4)
    P, L = deduce_lane_layout(m)
    assert (P, L) == (8, 4)
    vis, buckets, offsets = compute_lrb_buckets(32, 8, 4)
    assert vis.shape == (32, 32)
    assert buckets[vis].min().item() >= 0 and buckets[vis].max().item() <= 43

    # 2. Probability normalization check
    cfg = GPTConfig(n_layer=2, n_head=2, n_kv_head=2, n_embd=64, vocab_size=4, sequence_len=16,
                    window_pattern="L", lane_rel_bias=True, lane_offset_embed=True)
    torch.manual_seed(42)
    model = GPT(cfg)
    model.init_weights(verify=True)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.1)

    assert _lane_total(model, lane_mask(8, 2, 2), V=4, P=2, L=2, N=8) == pytest.approx(0.0, abs=1e-3)

    # 3. Gradient check
    model.train()
    x = torch.randint(0, 4, (2, 16))
    y = torch.randint(0, 4, (2, 16))
    x_in = lane_inputs(x, 4, 2, 3)
    loss = model(x_in, y, lane_mask=lane_mask(16, 4, 2), lane_params=(4, 2))
    loss.backward()
    assert model.lane_rel_bias.grad is not None
    assert model.lane_offset_embed.weight.grad is not None
    assert model.lane_rel_bias.grad.norm().item() > 0.0
    assert model.lane_offset_embed.weight.grad.norm().item() > 0.0


def test_lrb_roles_collapsed_control_keeps_only_offset_gap_buckets():
    """S16-C control: with the lane roles collapsed, every visible lane-to-lane pair falls in its
    offset-gap bucket (0..6), the prefix buckets and offsets are unchanged, and the lane order
    stays a normalised distribution."""
    from nanochat.lanes import compute_lrb_buckets
    N, P, L = 32, 8, 4
    vis, full, off = compute_lrb_buckets(N, P, L)
    _, col, off2 = compute_lrb_buckets(N, P, L, roles=False)
    lanes = torch.arange(N) >= P
    ll = vis & lanes[:, None] & lanes[None, :]
    assert full[ll].max() >= 7 and col[ll].max() < 7               # roles were in use, and now are not
    assert torch.equal(col[ll], full[ll] % 7)                      # the offset-gap bucket is kept
    assert torch.equal(col[~ll], full[~ll]) and torch.equal(off, off2)
    cfg = GPTConfig(n_layer=2, n_head=2, n_kv_head=2, n_embd=64, vocab_size=4, sequence_len=16,
                    window_pattern="L", lane_rel_bias=True, lane_offset_embed=True, lane_rel_bias_roles=False)
    torch.manual_seed(42)
    model = GPT(cfg)
    model.init_weights(verify=True)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.1)
    assert _lane_total(model, lane_mask(8, 2, 2), V=4, P=2, L=2, N=8) == pytest.approx(0.0, abs=1e-3)


def test_lookahead_band_runs_from_a_quarter_of_the_lane_to_before_the_junction():
    from scripts.sap_position_bpb import lookahead_band
    assert lookahead_band(list(range(30))) == sum(range(8, 29))      # L = 64: offsets 8-28
    assert lookahead_band(list(range(15))) == sum(range(4, 14))      # L = 128: offsets 4-13
    assert lookahead_band(list(range(60))) == sum(range(15, 59))     # L = 32: offsets 15-58

