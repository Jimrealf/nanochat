"""Lanes (nanochat/lanes.py): the lane order is an exact factorisation, one lane is the ordinary
causal model, and the timed decoder draws from the same conditionals training scores."""
import itertools

import pytest
import torch

from nanochat.gpt import GPT, GPTConfig
from nanochat.lanes import generate_lanes, lane_inputs, lane_layout, lane_mask, lane_rank


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

