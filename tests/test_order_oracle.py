"""S14 E0 order oracle (scripts/sap_order_oracle.py). Under an exact any-order oracle (a Markov
chain, conditionals by forward-backward over the visible tokens) every order's chain is the exact
block NLL, so gap = 0, and the parallel score exceeds it by the same-step dependence (TC), which
is zero for independent tokens. The masked-LM wrapper must find the logit alignment and catch an
oracle that ignores right context; sharded, resumed runs must merge to the single run."""
import json
import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from nanochat.bridge import bisection_levels
from scripts.sap_order_oracle import (MaskedLMOracle, _groups, compare, context_probe, finalize, order_steps,
                                      score_order, score_rows, separator_scores, snap_levels)


class MarkovOracle:
    """Exact conditionals q(x_i | visible) of a first-order Markov chain; mask_id = V."""

    def __init__(self, V=3, seed=0, iid=False, causal=False):
        g = torch.Generator().manual_seed(seed)
        A = torch.rand(V, V, generator=g, dtype=torch.float64) ** 4
        if iid:
            A = A[:1].expand(V, V).clone()
        self.A = A / A.sum(1, keepdim=True)
        self.pi = torch.full((V,), 1.0 / V, dtype=torch.float64)
        self.V, self.mask_id, self.causal = V, V, causal

    def sample(self, n, seed):
        g = torch.Generator().manual_seed(seed)
        x = [int(torch.multinomial(self.pi, 1, generator=g))]
        for _ in range(n - 1):
            x.append(int(torch.multinomial(self.A[x[-1]], 1, generator=g)))
        return torch.tensor(x)

    def block_nll(self, ids, P):
        return -sum(math.log(self.A[ids[k - 1], ids[k]]) for k in range(P, ids.numel()))

    def posteriors(self, x):
        N, V = x.numel(), self.V
        e = torch.ones(N, V, dtype=torch.float64)
        vis = x != self.mask_id
        e[vis] = F.one_hot(x[vis], V).double()
        alpha, beta = torch.empty(N, V, dtype=torch.float64), torch.ones(N, V, dtype=torch.float64)
        a = self.pi * e[0]
        alpha[0] = a / a.sum()
        for k in range(1, N):
            a = (alpha[k - 1] @ self.A) * e[k]
            alpha[k] = a / a.sum()
        if not self.causal:                           # causal: right context ignored (the bug to catch)
            for k in range(N - 2, -1, -1):
                b = self.A @ (e[k + 1] * beta[k + 1])
                beta[k] = b / b.sum()
        post = alpha * beta
        return post / post.sum(1, keepdim=True)

    def nll(self, x, pos, tok):
        return [-self.posteriors(x[b])[pos[b], tok[b]].log() for b in range(x.size(0))]


def _expected(oracle, P, Tb, steps, groups=0):
    """E_p[(par, grouped chain)] by enumerating every sequence of the chain."""
    N = P + 1 + Tb
    par = chain = 0.0
    for code in range(oracle.V ** N):
        ids = torch.tensor([(code // oracle.V ** k) % oracle.V for k in range(N)])
        w = float(oracle.pi[ids[0]]) * math.exp(-oracle.block_nll(ids, 1))
        p, c, _ = score_order(oracle, ids, P, steps, groups)
        par, chain = par + w * float(p.sum()), chain + w * float(c.sum())
    return par, chain


@pytest.mark.parametrize("order", ["l2r", "bisect1", "bisect2", "lanes2", "lanes4", "random3", "snap1"])
def test_every_chain_is_the_exact_block_nll(order):
    oracle, P, Tb = MarkovOracle(), 3, 8
    ids = oracle.sample(P + 1 + Tb, seed=1)
    is_start = torch.tensor([0, 0, 1, 0, 0, 0, 1, 0], dtype=torch.bool)
    steps, level = order_steps(order, Tb, is_start=is_start, seed=1)
    par, chain, passes = score_order(oracle, ids, P, steps, batch=3)
    assert float(chain.sum()) == pytest.approx(oracle.block_nll(ids, P), rel=1e-9)
    assert passes == 1 + Tb
    if order == "l2r":
        assert torch.equal(par, chain)


def test_tc_is_the_same_step_dependence():
    P, Tb = 1, 6
    steps, _ = order_steps("random2", Tb, seed=0)                    # two parallel steps of three
    iid = MarkovOracle(V=2, iid=True)
    ids = iid.sample(P + 1 + Tb, seed=0)
    par, chain, _ = score_order(iid, ids, P, steps)
    assert torch.allclose(par, chain, atol=1e-12)                    # independent tokens: TC = 0
    dep = MarkovOracle(V=2, seed=3)
    ids = dep.sample(P + 1 + 16, seed=0)
    par, chain, _ = score_order(dep, ids, P, order_steps("bisect1", 16)[0])
    assert torch.allclose(par, chain, atol=1e-12)                    # bisection midpoints of a Markov chain are
    par, chain = _expected(dep, P, Tb, steps)                        # independent given their brackets
    _, grouped = _expected(dep, P, Tb, steps, groups=2)
    assert par > grouped + 1e-3 and grouped > chain + 1e-3           # E[par] >= E[2 groups] >= E[chain]
    assert [g.tolist() for g in _groups(torch.arange(5), 2)] == [[0, 2, 4], [1, 3]]


@pytest.mark.parametrize("order", ["l2r", "bisect1", "bisect4", "lanes4", "random5", "snap2"])
def test_orders_are_complete_schedules(order):
    Tb = 32
    is_start = torch.zeros(Tb, dtype=torch.bool)
    is_start[[4, 11, 19, 26]] = True
    steps, level = order_steps(order, Tb, is_start=is_start, seed=0)
    assert steps.shape == (1 + Tb,) and steps[0] == 0 and (steps[1:] > 0).all()
    n = int(steps.max()) + 1
    assert set(steps.tolist()) == set(range(n)) and len(level) == n  # contiguous, one level per step
    assert level == sorted(level)
    if order == "lanes4":                                            # junctions last, lane starts first
        assert steps[1 + torch.arange(7, Tb, 8)].eq(n - 1).all() and steps[1 + torch.arange(0, Tb, 8)].eq(1).all()


def test_bisection_and_snap_levels():
    steps, _ = order_steps("bisect1", 8)
    assert int(steps.max()) + 1 == 5                                 # block start, then log2(8) + 1 levels
    is_start = torch.zeros(16, dtype=torch.bool)
    assert torch.equal(snap_levels(16, is_start, 4), bisection_levels(16)[1])   # nothing to snap to
    is_start[5] = True
    lv = snap_levels(16, is_start, 2)                                # midpoint 7, reach 5..9
    assert lv[15] == 0 and lv[5] == 1 and (lv >= 0).all()
    assert snap_levels(16, is_start, 1)[7] == 1                      # 5 out of reach: the midpoint


def test_one_token_separates_a_markov_chain():
    oracle, P, Tb = MarkovOracle(seed=5), 3, 16
    ids = oracle.sample(P + 1 + Tb, seed=2)
    _, full, _ = score_order(oracle, ids, P, order_steps("l2r", Tb)[0])
    hid = separator_scores(oracle, ids, P, cut=8, windows=[0, 1, 3], span=6, batch=4)
    assert torch.allclose(hid[1], full[8:14]) and torch.allclose(hid[3], full[8:14])   # far past adds nothing
    assert not torch.allclose(hid[0], full[8:14])


@pytest.mark.parametrize("shift", [0, 1])
def test_masked_lm_oracle_finds_the_alignment_and_needs_right_context(shift):
    exact, P, Tb = MarkovOracle(seed=2), 8, 64

    def model(input_ids, **_):
        lg = torch.stack([exact.posteriors(r).log() for r in input_ids])
        lg[input_ids != exact.mask_id] = 0.0                         # untrained at visible positions
        return SimpleNamespace(logits=torch.cat([lg[:, 1:], lg[:, -1:]], 1) if shift else lg)

    oracle = MaskedLMOracle(model, tok=None, mask_id=exact.mask_id, shift=-1, device="cpu")
    rows = [exact.sample(P + 1 + Tb, seed=s) for s in (5, 6)]
    oracle.detect_shift(rows, P)
    assert oracle.shift == shift
    steps, _ = order_steps("bisect1", Tb)
    assert torch.allclose(score_order(oracle, rows[0], P, steps)[0], score_order(exact, rows[0], P, steps)[0],
                          atol=1e-6)                                 # float32 logits
    assert context_probe(oracle, rows, P)["ratio"] < 0.95
    assert context_probe(MarkovOracle(seed=2, causal=True), rows, P)["ratio"] == pytest.approx(1.0)


def test_sharded_resumed_runs_merge_to_the_single_run(tmp_path):
    oracle, P, Tb, R = MarkovOracle(), 2, 8, 4
    rows = [oracle.sample(P + 1 + Tb, seed=s) for s in range(R)]
    orders = ["l2r", "bisect1", "lanes2", "random3"]
    cfg = {"oracle": "markov", "mask_id": oracle.mask_id, "shift": 0, "rows": R, "prefix": P, "block": 1 + Tb,
           "orders": orders, "groups": 0, "seed": 0, "rows_hash": "x", "block_bytes": [30] * R}
    run = lambda idx, done=None: score_rows(oracle, rows, P, Tb, orders, idx, done=done, log=lambda s: None)
    full = run(range(R))
    single = finalize([{"config": cfg, "records": full}], n_boot=50)
    row1 = run([1])
    calls, exact_nll = [], oracle.nll
    oracle.nll = lambda *a: calls.append(1) or exact_nll(*a)
    second = run([1, 3], done=row1)
    resumed_calls = len(calls)
    calls.clear()
    run([3])
    assert resumed_calls == len(calls)                               # the resumed row was not rescored
    sharded = finalize([{"config": cfg, "records": run([0, 2])}, {"config": cfg, "records": second}], n_boot=50)
    assert json.dumps(sharded, sort_keys=True) == json.dumps(single, sort_keys=True)
    for o in orders:                                                 # an exact oracle has no gap
        assert abs(single["orders"][o]["gap_pct"][0]) < 1e-9
    tot = lambda o, k: sum(float(full[i][o][k].sum()) for i in range(R))
    s = single["orders"]["lanes2"]
    assert s["TC_pct"][0] == pytest.approx(100 * (tot("lanes2", "par") - tot("lanes2", "chain")) / tot("l2r", "chain"))
    assert s["total_pct"][0] == pytest.approx(s["TC_pct"][0] + s["gap_pct"][0])
    assert s["TC_pct"][1] <= s["TC_pct"][0] <= s["TC_pct"][2] and single["orders"]["l2r"]["TC_pct"] == [0.0] * 3
    with pytest.raises(AssertionError, match="not scored"):
        finalize([{"config": cfg, "records": second}])
    paths = []
    for k in range(2):
        paths.append(tmp_path / f"r{k}.json")
        paths[-1].write_text(json.dumps(single))
    assert compare(paths, log=lambda s: None)["spearman"] == pytest.approx(1.0)


def test_command_line_run_and_sharded_merge(tmp_path, monkeypatch):
    import sys

    import pyarrow as pa
    import pyarrow.parquet as pq

    from scripts import sap_order_oracle as soo

    exact, chars = MarkovOracle(V=4, seed=4), "ab. "
    texts = ["".join(chars[i] for i in exact.sample(40, seed=s).tolist()) for s in range(5)]
    (tmp_path / "data").mkdir()
    pq.write_table(pa.table({"text": texts}), tmp_path / "data" / "shard_00000.parquet")

    class Tok:                                                      # characters as tokens
        bos_token_id = None

        def decode(self, ids, **_):
            return "".join(chars[i] for i in ids)

        def __call__(self, text, **_):
            return SimpleNamespace(input_ids=[chars.index(c) for c in text])

    def model(input_ids, **_):
        lg = torch.stack([exact.posteriors(r).log() for r in input_ids])
        lg[input_ids != exact.mask_id] = 0.0
        return SimpleNamespace(logits=lg)

    monkeypatch.setattr(soo.MaskedLMOracle, "from_pretrained",
                        classmethod(lambda cls, name, device, shift, mask_id: cls(model, Tok(), exact.mask_id, shift, "cpu")))
    common = ["--data-dir", str(tmp_path / "data"), "--rows", "3", "--prefix", "4", "--block", "16",
              "--orders", "l2r,bisect1,lanes4,snap4,random3", "--batch", "5",
              "--sep-cut", "8", "--sep-span", "8", "--sep-windows", "0,1,2"]
    commits = []

    def main(*extra):
        soo.main(common + list(extra), commit=lambda: commits.append(1))

    main("--raw", str(tmp_path / "all.pt"), "--out", str(tmp_path / "all.json"))
    assert len(commits) == 3 + 1 + 1                                 # every row, the final save, the result
    for k in range(2):
        main("--shard", str(k), "--num-shards", "2", "--raw", str(tmp_path / f"s{k}.pt"))
    main("--shard", "1", "--num-shards", "2", "--raw", str(tmp_path / "s1.pt"))     # finished shard: resumes
    monkeypatch.setattr(sys, "argv", ["x", "--merge", str(tmp_path / "s0.pt"), str(tmp_path / "s1.pt"),
                                      "--out", str(tmp_path / "merged.json")])
    soo.main()
    single, merged = (json.loads((tmp_path / f).read_text()) for f in ("all.json", "merged.json"))
    assert merged == single and single["shift"] == 0 and single["rows"] == 3
    assert single["validity"]["right_context_used"] and set(single["orders"]) == {"l2r", "bisect1", "lanes4",
                                                                                   "snap4", "random3"}
    assert single["orders"]["bisect1"]["steps_max"] == 6 and "bisect1_TC_ge_3pct" in single["readings"]
    assert single["separator"]["1"]["8"]["far_past_info_bits"] == pytest.approx([0.0] * 3, abs=1e-9)
    assert not single["readings"]["LSB_dead"] and single["validity"]["separator_info_nonnegative"]
