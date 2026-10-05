"""Window bisection (nanochat/wbisect.py): the step schedule, a two-stream mask through which no
token sees its own or any same/later-step content, and an exact factorisation: the probabilities
of all rows sum to one for any parameters."""
import itertools

import pytest
import torch

from nanochat.gpt import GPT, GPTConfig
from nanochat.wbisect import (bridged_lanes_steps, lane_order_steps, pos_ids, seeded_lanes_steps, two_stream_batch,
                              two_stream_mask, wb_forward, wb_steps)


def _model(V=6, layers=2, seq=32):
    cfg = GPTConfig(n_layer=layers, n_head=2, n_kv_head=2, n_embd=64, vocab_size=V, sequence_len=seq,
                    window_pattern="L")
    torch.manual_seed(0)
    model = GPT(cfg)
    model.init_weights(verify=True)
    with torch.no_grad():                    # c_proj starts at zero; move every weight off its init
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.3)
    return model.float().eval()


def test_steps_put_the_prefix_first_and_the_block_end_window_next():
    s = wb_steps(12, 3, 2)
    assert s[:3].tolist() == [-3, -2, -1]
    assert s[3:].min() == 0 and s[10] == 0 and s[11] == 1          # block end window, in order
    assert sorted(set(s[3:].tolist())) == list(range(int(s.max()) + 1))


@pytest.mark.parametrize("n", [1, 2])
def test_a_token_sees_only_contents_of_earlier_steps(n):
    model = _model()
    N, P = 10, 2
    steps = wb_steps(N, P, n)
    g = torch.Generator().manual_seed(1)
    x = torch.randint(0, 5, (3, N), generator=g)
    mask, pid = two_stream_mask(steps), pos_ids(N)
    with torch.no_grad():
        base = model(two_stream_batch(x, 5)[0], lane_mask=mask, pos_ids=pid, head_from=N)
        for k in range(1, N):
            later = steps >= steps[k]                                # own and same/later-step contents
            x2 = x.clone()
            x2[:, later] = (x2[:, later] + 1) % 5
            out = model(two_stream_batch(x2, 5)[0], lane_mask=mask, pos_ids=pid, head_from=N)
            assert torch.allclose(out[:, k], base[:, k], atol=1e-4)
            earlier = (steps < steps[k]).nonzero().flatten()
            if len(earlier):                                         # and earlier contents do reach it
                x3 = x.clone()
                x3[:, earlier] = (x3[:, earlier] + 1) % 5
                out3 = model(two_stream_batch(x3, 5)[0], lane_mask=mask, pos_ids=pid, head_from=N)
                assert not torch.allclose(out3[:, k], base[:, k], atol=1e-4)


@pytest.mark.parametrize("n,P", [(1, 1), (2, 1), (2, 3)])
def test_window_bisection_is_a_normalised_distribution(n, P):
    """Every row x_1..x_5 over a 4-token vocabulary (x_0 fixed): exp(-NLL) sums to one."""
    model = _model(V=4)
    N = 6
    steps = wb_steps(N, P, n)
    rows = torch.tensor(list(itertools.product(range(4), repeat=N - 1)))
    x = torch.cat([torch.zeros(rows.size(0), 1, dtype=torch.long), rows], 1)
    with torch.no_grad():
        nll = torch.cat([wb_forward(model, c, steps, 3, loss_reduction="none").view(c.shape)
                         for c in x.split(256)])
    total = torch.logsumexp(-nll[:, 1:].double().sum(1), 0).exp().item()
    assert total == pytest.approx(1.0, abs=1e-4)


def test_bridged_lanes_place_separators_coarse_to_fine_then_fill_in_lockstep():
    N, P, L, n = 2 + 24, 2, 4, 2                                   # four 6-token intervals after a 2-token prefix
    s = bridged_lanes_steps(N, P, L, n)
    starts = [2, 8, 14, 20]
    seps = [list(range(st + 4, st + 6)) for st in starts]
    assert s[seps[3]].tolist() == [0, 1]                            # the block end's separator first
    assert s[seps[1]].tolist() == [2, 3]                            # then the middle interval's
    assert s[seps[0]].tolist() == s[seps[2]].tolist() == [4, 5]     # then both quarters in parallel
    for st in starts:                                               # fills: left to right, in lockstep
        assert s[st:st + 4].tolist() == [6, 7, 8, 9]
    assert int(s.max()) + 1 == 3 * n + (6 - n)


def test_lane_orders_write_lockstep_lanes_and_middle_out_seeds():
    s = lane_order_steps(2 + 12, 2, 3)                             # three 4-token lanes after a 2-token prefix
    assert s[2:].tolist() == [0, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3]  # junctions (lane starts 6, 10) last
    s = seeded_lanes_steps(2 + 12, 2, 2)                           # two 6-token intervals, seeds at offset 2
    assert s[2:8].tolist() == [2, 1, 0, 1, 2, 3]
    assert s[8:].tolist() == [2, 1, 0, 1, 2, 3]
    assert int(s.max()) + 1 == 4                                   # 12 tokens in 4 steps, 2 contextless seeds
    s = seeded_lanes_steps(2 + 10, 2, 1, m=3)                      # a 3-token seed window, then both fronts
    assert s[2:].tolist() == [5, 4, 3, 0, 1, 2, 3, 4, 5, 6]        # left front starts warm at step 3


@pytest.mark.parametrize("order", ["bl21", "bl32", "lane2", "seed1", "seed2", "window"])
def test_bridged_lanes_are_a_normalised_distribution(order):
    model = _model(V=4)
    N = 6
    steps = {"bl21": lambda: bridged_lanes_steps(N, 1, 2, 1), "bl32": lambda: bridged_lanes_steps(N, 1, 3, 2),
             "lane2": lambda: lane_order_steps(N + 1, 1, 3)[:N], "seed1": lambda: seeded_lanes_steps(N, 1, 1),
             "seed2": lambda: seeded_lanes_steps(N + 1, 1, 2)[:N], "window": lambda: seeded_lanes_steps(N, 1, 1, m=2)}[order]()
    rows = torch.tensor(list(itertools.product(range(4), repeat=N - 1)))
    x = torch.cat([torch.zeros(rows.size(0), 1, dtype=torch.long), rows], 1)
    with torch.no_grad():
        nll = torch.cat([wb_forward(model, c, steps, 3, loss_reduction="none").view(c.shape)
                         for c in x.split(256)])
    total = torch.logsumexp(-nll[:, 1:].double().sum(1), 0).exp().item()
    assert total == pytest.approx(1.0, abs=1e-4)


@pytest.mark.parametrize("kind", ["wb", "bl", "lane", "seed"])
def test_cached_generator_reproduces_the_training_conditionals(kind):
    """Teacher-forced, the one-pass-per-step generator emits exactly the logits the two-stream
    training pass assigns to every block position."""
    from nanochat.wbisect import wb_generate
    model = _model(V=8)
    N, P = 14, 3
    steps = {"wb": lambda: wb_steps(N, P, 2), "bl": lambda: bridged_lanes_steps(N, P, 3, 2),
             "lane": lambda: lane_order_steps(N + 1, P, 4)[:N], "seed": lambda: seeded_lanes_steps(N + 1, P, 2)[:N]}[kind]()
    g = torch.Generator().manual_seed(4)
    x = torch.randint(0, 7, (2, N), generator=g)
    with torch.no_grad():
        full = model(two_stream_batch(x, 7)[0], lane_mask=two_stream_mask(steps), pos_ids=pos_ids(N), head_from=N)
        _, trace = wb_generate(model, x[:, :P], steps, 7, teacher=x, return_logits=True)
    assert sum(q.numel() for q, _ in trace) == N - P
    for qpos, lg in trace:
        assert torch.allclose(lg.float(), full[:, qpos].float(), atol=2e-3), (qpos, (lg - full[:, qpos]).abs().max())


def test_mask_cache_follows_the_order_not_the_address():
    """Regression (S11 eval, 2026-10-04): the cache was keyed on data_ptr, so a new order allocated
    at a freed order's address got the old mask. Different orders, and in-place edits, must each get
    their own mask."""
    from nanochat.wbisect import _mask_and_pos, bridged_lanes_steps, two_stream_mask
    a = bridged_lanes_steps(24, 4, 4, 2)
    b = bridged_lanes_steps(24, 4, 2, 2)
    assert not torch.equal(a, b)
    assert torch.equal(_mask_and_pos(a, "cpu")[0], two_stream_mask(a, "cpu"))
    buf = a.clone()
    assert torch.equal(_mask_and_pos(buf, "cpu")[0], two_stream_mask(a, "cpu"))
    buf.copy_(b)                                   # same tensor, same address, different order
    assert torch.equal(_mask_and_pos(buf, "cpu")[0], two_stream_mask(b, "cpu"))
