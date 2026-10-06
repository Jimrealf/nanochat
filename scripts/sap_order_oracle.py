"""
S14 E0, the order oracle (s14_sap_strict_tl_brainstorm.md): how much of a parallel generation
order's cost on real text is information, and how much is the difficulty of the order's own
conditionals. A large any-order model (a masked diffusion LM) serves as the conditional oracle.
Nothing is trained.

An order gives every block position a generation step; the tokens of a step are drawn together,
each given the tokens of earlier steps. With the oracle's conditionals q(x_i | visible tokens):

    NLL_par   = sum over steps and over each step's tokens of -log q(x_i | earlier steps)
                (one oracle pass per step: what the parallel order scores)
    NLL_chain = the same order with each step's tokens revealed one at a time, left to right
                (one pass per token: the order's exact chain under the oracle)
    TC        = NLL_par - NLL_chain     the same-step dependence parallel draws lose (the
                                        multi-information, the "joint dependence error" of
                                        dependency-aware parallel decoding)
    gap       = NLL_chain(order) - NLL_chain(l2r)
                                        how much harder the order's conditionals are for the oracle

Under an exact oracle the chain is order-invariant (gap = 0) and E[NLL_chain] <= E[NLL_par], so
TC >= 0 on average (tests/test_order_oracle.py). A real oracle's gap measures, at the oracle's
scale, the cost of predicting text out of left-to-right order. TC is the floor any model of that
order pays.

Every order first draws the block's first token (the prompt's next token, step 0), then orders the
other Tb positions:
    l2r          left to right
    bisect{n}    window bisection with n-token windows (nanochat/bridge.py; n=1: single tokens)
    lanes{L}     S08 plain lanes: token t >= 1 at lockstep step (t - 1) mod S, junctions last
    snap{W}      single-token bisection whose anchors move to the nearest sentence start within
                 +-W of the dyadic midpoint (S14-B). The order depends on the text and the offset
                 decisions are not scored, so its numbers are a necessary condition, not a bound
    random{R}    a random order in R equal steps, drawn afresh for every row (masked-diffusion-style
                 parallel decoding)
    conf{R}      confidence-ordered decoding in R steps, the masked-diffusion default (LLaDA's
                 low-confidence remasking): each step reveals the ceil(hidden / steps left) hidden
                 positions with the highest oracle max-probability given what is revealed so far
                 (R extra passes per row; the order depends only on visible text, so it is a valid
                 sampler and its chain is still a chain rule)
    bl{L}_{n}    S11 bridged lanes (nanochat/wbisect.py): each of L intervals' last n tokens placed
                 coarse-to-fine, then the intervals filled left to right in lockstep

Separator bound (E2a, --sep-cut; on by default): how many bits must cross a cut so that the next m
tokens lose at most 2%. With the past before the last w tokens hidden (block position c is the cut;
a BOS token stays visible), the l2r NLL of block positions c .. c+m-1 rises by I_hat bits, an
estimate of I(next m tokens; far past | the w-token window). Any code C of B bits attached to the
window gives I(next; C | window) <= H(C) <= B, so its cost over full context is at least I - B
bits, whatever the code and however it is learned: a separator needs at least I - 2% of the
span's NLL bits. Under an exact first-order Markov oracle one token is a perfect separator (the
test checks I_hat = 0 at w >= 1).

Validity checks (pre-registered; E0 is inconclusive for an oracle that fails one): right context
lowers the oracle's NLL (a causal mask left on by mistake would not, and would bias every
non-left-to-right order); TC >= 0 within its CI; the oracle's l2r bpb within 20% of an AR model's
on the same text (--ar-ref); across two oracles, Spearman >= 0.8 on the orders' total cost
(--compare); the far-past information does not grow with the window.

Cost: with --groups 0, exactly 1 + Tb oracle passes per order and row (the steps, plus one per
token after a step's first), plus windows x span for the separator bound. 32 rows x (10 orders x
1025 + 4 x 256) passes at an 8B oracle is about 5 to 9 H100-hours; --shard/--num-shards splits the rows over GPUs, --raw saves every finished row (a rerun
resumes) and --merge combines the shards. --groups G reveals each step in G round-robin groups
instead (at most G passes per step), a cheaper estimate between the chain and the parallel score.

    python -m scripts.sap_order_oracle --oracle GSAI-ML/LLaDA-8B-Base --data-dir data \\
        --rows 32 --prefix 128 --block 1024 --ar-ref Qwen/Qwen2.5-7B --out out/s14_oracle_llada.json
    python -m scripts.sap_order_oracle ... --shard 0 --num-shards 4 --raw out/llada_0.pt   # one per GPU
    python -m scripts.sap_order_oracle --merge out/llada_*.pt --out out/s14_oracle_llada.json
    python -m scripts.sap_order_oracle --compare out/s14_oracle_llada.json out/s14_oracle_dream.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import time

import torch

DEFAULT_ORDERS = "l2r,bisect1,bisect2,bisect4,bisect16,lanes8,lanes32,lanes64,snap32,random12"
MAX_REPORTED_LEVELS = 64          # per-level tables are skipped for orders with more levels (l2r)
STRICT_STEPS = 13                 # the strict thesis: <= log2 T + 2 sampling levels
SEP_BUDGETS = (8, 16, 48)         # separator sizes in bits: S14 plan E2 (2^8, 2^16, 4 x 2^12 codes)
SEP_COST_PCT = 2.0                # the cost a separator may add over full context (E2 reading)


# ----------------------------------------------------------------------------- orders
def _block_steps(local):
    """(1 + Tb,) steps: the block's first token alone at step 0, then the (Tb,) local steps + 1."""
    return torch.cat([torch.zeros(1, dtype=torch.long), local.long() + 1])


def bisect_step_levels(Tb, n):
    """The bisection level of each window-bisection step (mirrors window_bisection_levels)."""
    from nanochat.bridge import bisection_levels
    levels, _ = bisection_levels(Tb)
    placed = torch.zeros(Tb, dtype=torch.bool)
    out = []
    for li, lvl in enumerate(levels):
        width = 0
        for m in lvl:
            j = 0
            for k in range(max(m - n + 1, 0), m + 1):
                if not placed[k]:
                    placed[k] = True
                    j += 1
            width = max(width, j)
        out += [li] * width
    return out


def snap_levels(Tb, is_start, W):
    """(Tb,) level of each local position for single-token bisection whose anchors move to the
    nearest sentence start within +-W of the dyadic midpoint (ties to the left; no start in reach:
    the midpoint). is_start[k] marks local position k as the first token of a sentence."""
    level_of = torch.full((Tb,), -1, dtype=torch.long)
    level_of[Tb - 1] = 0
    intervals, lvl = [(-1, Tb - 1)], 1
    while True:
        nxt = []
        for a, b in intervals:
            if b - a > 1:
                mid = (a + b) / 2
                lo, hi = max(a + 1, math.ceil(mid - W)), min(b - 1, math.floor(mid + W))
                cand = [p for p in range(lo, hi + 1) if is_start[p]]
                m = min(cand, key=lambda p: (abs(p - mid), p)) if cand else (a + b) // 2
                level_of[m] = lvl
                nxt += [(a, m), (m, b)]
        if not nxt:
            return level_of
        intervals, lvl = nxt, lvl + 1


def order_steps(name, Tb, is_start=None, seed=0):
    """(steps (1 + Tb,), level of each step, step 0's level being -1) for one of the module
    docstring's orders."""
    m = re.fullmatch(r"(l2r|bisect|lanes|snap|random|bl)(\d*)(?:_(\d+))?", name)
    assert m, f"unknown order {name!r}"
    kind, k = m.group(1), int(m.group(2) or 0)
    assert kind == "bl" or not m.group(3), f"unknown order {name!r}"
    if kind == "bl":
        from nanochat.wbisect import bridged_lanes_steps
        assert k >= 1 and m.group(3), f"{name}: bridged lanes are bl{{L}}_{{n}}"
        local = bridged_lanes_steps(Tb, 0, k, int(m.group(3)))
        level = list(range(int(local.max()) + 1))
    elif kind == "l2r":
        local = torch.arange(Tb)
        level = list(range(Tb))
    elif kind == "bisect":
        from nanochat.bridge import window_bisection_levels
        assert k >= 1
        _, local = window_bisection_levels(Tb, k)
        level = bisect_step_levels(Tb, k)
    elif kind == "lanes":
        assert k >= 1 and Tb % k == 0, f"lanes{k} needs the block ({Tb}) to split into {k} lanes"
        local = torch.arange(Tb) % (Tb // k)
        level = list(range(Tb // k))
    elif kind == "snap":
        assert is_start is not None, "snap orders need sentence starts"
        local = snap_levels(Tb, is_start, k)
        level = list(range(int(local.max()) + 1))
    else:
        assert k >= 1
        g = torch.Generator().manual_seed(seed)
        local = torch.empty(Tb, dtype=torch.long)
        local[torch.randperm(Tb, generator=g)] = torch.arange(Tb) * k // Tb
        level = list(range(k))
    return _block_steps(local), [-1] + [int(v) for v in level]


# ----------------------------------------------------------------------------- scoring
def _groups(pos, groups):
    """Reveal order inside a step: one position at a time left to right (the exact chain), or G
    round-robin groups."""
    if groups <= 0 or groups >= pos.numel():
        return [pos[i:i + 1] for i in range(pos.numel())]
    return [pos[g::groups] for g in range(groups)]


@torch.no_grad()
def score_order(oracle, ids, P, steps, groups=0, batch=8):
    """One row. ids: (P + T,) tokens, the block at P..P+T-1; steps: (T,) step of every block
    position. oracle.nll(x, pos, tok) returns, per row b of x, -log q(tok[b] at positions pos[b])
    with everything equal to oracle.mask_id hidden. Returns per-position (par, chain) nats and the
    number of oracle passes."""
    T = steps.numel()
    par = torch.zeros(T, dtype=torch.float64)
    jobs_par, jobs_chain = [], []                     # (visible (T,) bool, block positions)
    for s in range(int(steps.max()) + 1):
        pos = (steps == s).nonzero().flatten()
        before = steps < s
        jobs_par.append((before, pos))
        if pos.numel() > 1:
            seen = before.clone()
            for gi, g in enumerate(_groups(pos, groups)):
                if gi > 0:                            # the first group sees what the step's par saw
                    jobs_chain.append((seen.clone(), g))
                seen[g] = True

    def run(jobs, out):
        for i in range(0, len(jobs), batch):
            chunk = jobs[i:i + batch]
            x = ids[None].repeat(len(chunk), 1)
            for b, (vis, _) in enumerate(chunk):
                x[b, (~vis).nonzero().flatten() + P] = oracle.mask_id
            nll = oracle.nll(x, [g + P for _, g in chunk], [ids[g + P] for _, g in chunk])
            for (_, g), v in zip(chunk, nll):
                out[g] = v

    run(jobs_par, par)
    chain = par.clone()
    run(jobs_chain, chain)
    return par, chain, len(jobs_par) + len(jobs_chain)


@torch.no_grad()
def separator_scores(oracle, ids, P, cut, windows, span, batch=8, keep_first=False):
    """{w: (span,) nats}: the l2r NLL of block positions cut .. cut+span-1 when everything before
    the w block-or-prefix tokens preceding the cut is hidden (position 0 too unless keep_first,
    for a BOS token)."""
    c, out = P + cut, {}
    for w in windows:
        nll = torch.zeros(span, dtype=torch.float64)
        for i in range(0, span, batch):
            ts = list(range(i, min(i + batch, span)))
            x = ids[None].repeat(len(ts), 1)
            x[:, int(keep_first):c - w] = oracle.mask_id
            for b, t in enumerate(ts):
                x[b, c + t:] = oracle.mask_id
            vals = oracle.nll(x, [torch.tensor([c + t]) for t in ts], [ids[c + t:c + t + 1] for t in ts])
            for t, v in zip(ts, vals):
                nll[t] = v[0]
        out[w] = nll
    return out


@torch.no_grad()
def confidence_steps(oracle, ids, P, Tb, R):
    """(1 + Tb,) steps of confidence-ordered decoding (the conf{R} order): the block's first token
    at step 0, then R steps, each revealing the ceil(hidden / steps left) hidden positions whose
    oracle max-probability (oracle.confidence) is highest given the positions revealed so far.
    One oracle pass per step."""
    steps = torch.full((1 + Tb,), -1, dtype=torch.long)
    steps[0] = 0
    for s in range(1, R + 1):
        hidden = (steps < 0).nonzero().flatten()
        k = -(-hidden.numel() // (R - s + 1))
        x = ids.clone()
        x[hidden + P] = oracle.mask_id
        conf = oracle.confidence(x[None], [hidden + P])[0]
        steps[hidden[conf.topk(k).indices]] = s
    return steps


def score_rows(oracle, rows, P, Tb, orders, indices, starts=None, groups=0, batch=8, seed=0,
               done=None, save=None, log=print, sep=None):
    """Score rows[i] for i in indices under every order: {i: {order: record}}. done holds records
    already scored (a resumed run skips them); save(records) runs after every row. Random orders
    are drawn afresh for every row (seed + i); snap orders use starts[i]. sep: {cut, windows, span,
    keep_first} adds the separator scores under the key "_sep"."""
    records = dict(done or {})
    fixed = {o: order_steps(o, Tb) for o in orders if not o.startswith(("snap", "random", "conf"))}
    todo = [i for i in indices if i not in records]
    if len(todo) < len(indices):
        log(f"resuming: {len(indices) - len(todo)} of {len(indices)} rows already scored")
    for n, i in enumerate(todo):
        t0 = time.time()
        rec = {}
        for o in orders:
            extra = 0
            if o in fixed:
                steps, level = fixed[o]
            elif o.startswith("conf"):
                extra = int(o[4:])
                steps, level = confidence_steps(oracle, rows[i], P, Tb, extra), [-1] + list(range(extra))
            else:
                steps, level = order_steps(o, Tb, is_start=None if starts is None else starts[i], seed=seed + i)
            par, chain, passes = score_order(oracle, rows[i], P, steps, groups, batch)
            passes += extra
            rec[o] = {"par": par, "chain": chain, "passes": passes, "steps": int(steps.max()) + 1,
                      "level": torch.tensor([level[s] for s in steps.tolist()])}
        if sep:
            rec["_sep"] = separator_scores(oracle, rows[i], P, sep["cut"], sep["windows"], sep["span"], batch,
                                           sep["keep_first"])
        records[i] = rec
        if save is not None:
            save(records)
        ref = rec[orders[0]]["chain"].sum()
        line = " ".join(f"{o}={float(rec[o]['par'].sum() - ref):+.1f}" for o in orders[1:])
        log(f"row {i} ({n + 1}/{len(todo)}, {time.time() - t0:.0f}s): nats over the {orders[0]} chain: {line}")
    return records


# ----------------------------------------------------------------------------- oracles
class MaskedLMOracle:
    """A masked diffusion LM as the conditional oracle: q(x_i | visible) is the model's softmax at
    position i (shift 0, LLaDA) or at i - 1 (shift 1: Dream keeps the shift of the AR model it was
    adapted from). shift=-1 detects it on the data (the wrong alignment is far worse)."""

    KNOWN = {"GSAI-ML/LLaDA-8B-Base": (126336, 0), "Dream-org/Dream-v0-Base-7B": (None, 1)}

    def __init__(self, model, tok, mask_id, shift, device):
        self.model, self.tok, self.mask_id, self.shift, self.device = model, tok, mask_id, shift, device

    @classmethod
    def from_pretrained(cls, name, device, shift=-1, mask_id=-1):
        from transformers import AutoModel, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
        model = AutoModel.from_pretrained(name, trust_remote_code=True, torch_dtype=torch.bfloat16)
        model = model.to(device).eval()
        known_mask, _ = cls.KNOWN.get(name, (None, None))
        mask_id = mask_id if mask_id >= 0 else known_mask if known_mask is not None else tok.mask_token_id
        assert mask_id is not None, f"{name}: no mask token; pass --mask-id"
        return cls(model, tok, int(mask_id), shift, device)

    def logits(self, x):
        try:
            return self.model(input_ids=x).logits
        except TypeError:                             # Dream's forward wants an explicit full mask
            return self.model(input_ids=x, attention_mask="full").logits

    @torch.no_grad()
    def nll(self, x, pos, tok):
        lg = self.logits(x.to(self.device))
        out = []
        for b in range(x.size(0)):
            z = lg[b, pos[b].to(self.device) - self.shift].float()
            t = tok[b].to(self.device)[:, None]
            out.append((torch.logsumexp(z, -1) - z.gather(1, t).squeeze(1)).double().cpu())
        return out

    @torch.no_grad()
    def confidence(self, x, pos):
        """Per row b of x: the oracle's max probability at positions pos[b] (decoding confidence)."""
        lg = self.logits(x.to(self.device))
        return [torch.softmax(lg[b, pos[b].to(self.device) - self.shift].float(), -1).amax(-1).cpu()
                for b in range(x.size(0))]

    @torch.no_grad()
    def detect_shift(self, rows, P, frac=0.15, seed=0):
        """Mean NLL of randomly hidden block tokens under shift 0 and 1; keeps the better one."""
        res = {}
        for s in (0, 1):
            self.shift = s
            g = torch.Generator().manual_seed(seed)
            tot = n = 0
            for ids in rows[:2]:
                hide = (torch.rand(ids.numel() - P, generator=g) < frac).nonzero().flatten() + P
                x = ids[None].clone()
                x[0, hide] = self.mask_id
                tot += float(self.nll(x, [hide], [ids[hide]])[0].sum())
                n += hide.numel()
            res[s] = tot / max(n, 1)
        self.shift = min(res, key=res.get)
        return res


@torch.no_grad()
def context_probe(oracle, rows, P, n=32, seed=0):
    """Mean NLL of block tokens with every other token visible, and with every later token hidden.
    An oracle that reads right context scores the first far lower; a causal one scores them equal,
    which would silently tax every order that is not left to right."""
    g = torch.Generator().manual_seed(seed)
    both = left = cnt = 0
    for ids in rows[:2]:
        T = ids.numel() - P
        for i in (torch.randperm(T - 1, generator=g)[:n] + P).tolist():   # the last token has no right
            x = ids[None].repeat(2, 1)
            x[0, i] = oracle.mask_id
            x[1, i:] = oracle.mask_id
            v = oracle.nll(x, [torch.tensor([i])] * 2, [ids[i:i + 1]] * 2)
            both, left, cnt = both + float(v[0].sum()), left + float(v[1].sum()), cnt + 1
    return {"both_sides": both / cnt, "left_only": left / cnt, "ratio": both / left}


@torch.no_grad()
def ar_reference_bpb(name, prefixes, blocks, device):
    """bpb of an AR model on the same block texts given the same prefixes: NLL(prefix + block)
    - NLL(prefix), over the block's UTF-8 bytes. Tokenizers differ, bytes do not."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16).to(device).eval()

    def nll(text):
        ids = tok(text, add_special_tokens=False).input_ids
        if tok.bos_token_id is not None:
            ids = [tok.bos_token_id] + ids
        x = torch.tensor(ids, device=device)[None]
        lg = model(x).logits[0, :-1].float()
        return float((torch.logsumexp(lg, -1) - lg.gather(1, x[0, 1:, None]).squeeze(1)).sum())

    nats = sum(nll(p + b) - nll(p) for p, b in zip(prefixes, blocks))
    nbytes = sum(len(b.encode("utf-8")) for b in blocks)
    return nats / math.log(2) / nbytes


# ----------------------------------------------------------------------------- data
def load_rows(tok, data_dir, rows, length):
    """The first `rows` validation documents with at least `length` oracle tokens (the last
    parquet shard is validation, as in nanochat/dataset.py), cut to `length`, BOS first if the
    tokenizer has one."""
    import pyarrow.parquet as pq
    files = sorted(f for f in os.listdir(data_dir) if f.endswith(".parquet") and not f.endswith(".tmp"))
    pf = pq.ParquetFile(os.path.join(data_dir, files[-1]))
    bos = tok.bos_token_id
    out = []
    for rg in range(pf.num_row_groups):
        for text in pf.read_row_group(rg).column("text").to_pylist():
            ids = tok(text, add_special_tokens=False).input_ids
            if bos is not None:
                ids = [bos] + ids
            if len(ids) >= length:
                out.append(torch.tensor(ids[:length], dtype=torch.long))
                if len(out) == rows:
                    return out
    return out


def sentence_starts(tok, ids, P, cache=None):
    """(Tb,) bool over block positions 1..Tb: the token begins a sentence, i.e. the token before it
    ends with . ! ? (before closing quotes or brackets) or holds a newline."""
    cache = {} if cache is None else cache

    def ends(t):
        if t not in cache:
            s = tok.decode([t])
            cache[t] = "\n" in s or s.rstrip(" \"')]}”’").endswith((".", "!", "?"))
        return cache[t]
    return torch.tensor([ends(int(t)) for t in ids[P:-1]], dtype=torch.bool)


def rows_hash(rows):
    h = hashlib.sha256()
    for r in rows:
        h.update(r.numpy().tobytes())
    return h.hexdigest()[:16]


# ----------------------------------------------------------------------------- summary
def lane_profile(rs, ref_rs, L):
    """S15: per lane (lanes 1..L-1, mean over rows) and by offset within the lane, the lanes order's
    parallel NLL minus the reference chain's at the same positions. Its positive part is the
    deficit (early offsets, decided without left context), its negative part the recovery (late
    offsets, which read the next lane's early tokens); tc_by_offset is the order's own same-step
    share. In the units of sap_position_bpb's lane report for trained models."""
    Tb = rs[0]["par"].numel() - 1
    S = Tb // L
    k = torch.arange(Tb)
    sel = k // S >= 1
    off = (k % S)[sel]
    ex = torch.zeros(S, dtype=torch.float64)
    tc = torch.zeros(S, dtype=torch.float64)
    for r, q in zip(rs, ref_rs):
        ex.index_add_(0, off, (r["par"][1:] - q["chain"][1:])[sel])
        tc.index_add_(0, off, (r["par"][1:] - r["chain"][1:])[sel])
    ex, tc = ex / (len(rs) * (L - 1)), tc / (len(rs) * (L - 1))
    return {"lane_len": S, "deficit": float(ex.clamp(min=0).sum()), "recovery": float(ex.clamp(max=0).sum()),
            "net": float(ex.sum()), "tc": float(tc.sum()), "excess_by_offset": ex.tolist(),
            "tc_by_offset": tc.tolist()}


def summarize(per_row, block_bytes, ref="l2r", n_boot=1000, seed=0):
    """per_row[order] = list over rows of score_rows records. Returns, per order, NLL per token,
    TC, gap and their sum as % of the reference chain NLL ([estimate, 95% row-bootstrap low,
    high]), and per-level TC for orders with at most MAX_REPORTED_LEVELS levels."""
    R = len(per_row[ref])
    par = {o: torch.tensor([float(r["par"].sum()) for r in rs], dtype=torch.float64) for o, rs in per_row.items()}
    chain = {o: torch.tensor([float(r["chain"].sum()) for r in rs], dtype=torch.float64) for o, rs in per_row.items()}
    count = sum(r["par"].numel() for r in per_row[ref])
    den = chain[ref]
    g = torch.Generator().manual_seed(seed)
    boots = [torch.randint(0, R, (R,), generator=g) for _ in range(n_boot)]

    def with_ci(num):
        vals = sorted(100.0 * float(num[b].sum()) / float(den[b].sum()) for b in boots)
        return [100.0 * float(num.sum()) / float(den.sum()), vals[int(0.025 * n_boot)], vals[int(0.975 * n_boot) - 1]]

    out = {}
    for o, rs in per_row.items():
        levels = {}
        for r in rs:
            for lv, p, c in zip(r["level"].tolist(), r["par"].tolist(), r["chain"].tolist()):
                d = levels.setdefault(lv, [0.0, 0.0, 0])
                d[0], d[1], d[2] = d[0] + p, d[1] + c, d[2] + 1
        steps = [r["steps"] for r in rs]
        out[o] = {
            "steps_mean": sum(steps) / R, "steps_max": max(steps),
            "passes_per_row": sum(r["passes"] for r in rs) / R,
            "par_nats_per_token": float(par[o].sum()) / count,
            "chain_nats_per_token": float(chain[o].sum()) / count,
            "TC_pct": with_ci(par[o] - chain[o]),
            "gap_pct": with_ci(chain[o] - chain[ref]),
            "total_pct": with_ci(par[o] - chain[ref]),
        }
        if len(levels) <= MAX_REPORTED_LEVELS:
            out[o]["per_level"] = {
                str(k): {"tokens_per_row": v[2] / R, "par_nats_per_token": v[0] / v[2],
                         "chain_nats_per_token": v[1] / v[2], "TC_nats_per_row": (v[0] - v[1]) / R,
                         "TC_pct": 100.0 * (v[0] - v[1]) / float(den.sum())}
                for k, v in sorted(levels.items())}
        if o.startswith("snap"):
            out[o]["note"] = "order depends on the text and its offset decisions are not scored: a necessary condition only"
        m = re.fullmatch(r"lanes(\d+)", o)
        if m and int(m.group(1)) > 1:
            out[o]["lane_profile"] = lane_profile(rs, per_row[ref], int(m.group(1)))
    out[ref]["bpb"] = float(den.sum()) / math.log(2) / block_bytes
    return out


def _boot_mean(v, n_boot=1000, seed=0):
    """[mean, 95% row-bootstrap low, high] of a (R,) tensor."""
    g = torch.Generator().manual_seed(seed)
    R = v.numel()
    vals = sorted(float(v[torch.randint(0, R, (R,), generator=g)].mean()) for _ in range(n_boot))
    return [float(v.mean()), vals[int(0.025 * n_boot)], vals[int(0.975 * n_boot) - 1]]


def separator_summary(records, rows, sep, ref="l2r", n_boot=1000, seed=0):
    """Per window w and span m (16, 64 and the whole span), in bits per row: the far past's
    information about the next m tokens, their full-context NLL, the bits a separator needs to keep
    them within SEP_COST_PCT, and the least cost (% of the full-context NLL) any separator of
    SEP_BUDGETS bits can have."""
    cut, span = sep["cut"], sep["span"]
    full = torch.stack([records[i][ref]["chain"][cut:cut + span] for i in rows]) / math.log(2)
    out = {}
    for w in sep["windows"]:
        hid = torch.stack([records[i]["_sep"][w] for i in rows]) / math.log(2)
        res = {}
        for m in sorted({m for m in (16, 64, span) if m <= span}):
            info, ce = (hid[:, :m] - full[:, :m]).sum(1), full[:, :m].sum(1)
            res[str(m)] = {"far_past_info_bits": _boot_mean(info, n_boot, seed),
                           "full_context_bits": float(ce.mean()),
                           "bits_needed": _boot_mean(info - SEP_COST_PCT / 100 * ce, n_boot, seed),
                           "least_cost_pct": {str(B): 100.0 * max(0.0, float(info.mean()) - B) / float(ce.mean())
                                              for B in SEP_BUDGETS}}
        out[str(w)] = res
    return out


def readings(summary, sep=None):
    """The pre-registered E0 readings for one oracle (s14_sap_strict_tl_brainstorm.md, E0 and E2a).
    Point estimates; the CIs are in the summary. Two of them need both oracles (compare)."""
    out = {}
    b = summary.get("bisect1")
    if b:
        out["bisect1_TC_ge_3pct"] = b["TC_pct"][0] >= 3.0        # strict token orders closed on information
        out["bisect1_gap_ge_5pct"] = b["gap_pct"][0] >= 5.0      # closed on computation if true for both oracles
    strict = [o for o, s in summary.items() if o != "l2r" and s["steps_max"] <= STRICT_STEPS]
    out["strict_orders"] = strict
    out["PCB_live_orders"] = [o for o in strict
                              if summary[o]["TC_pct"][0] <= 1.0 and summary[o]["gap_pct"][0] <= 2.0]
    for o in summary:
        if o.startswith("snap") and b and b["total_pct"][0] > 0:
            cut = 100.0 * (1 - summary[o]["total_pct"][0] / b["total_pct"][0])
            out[f"{o}_cut_vs_bisect1_pct"] = cut
            out[f"{o}_BSB_live"] = cut >= 15.0
    for o in summary:                                 # S15 L0b: lanes against diffusion decoding at equal steps
        if not o.startswith("lanes"):
            continue
        for d in summary:
            if d.startswith(("conf", "random")) and summary[d]["steps_max"] == summary[o]["steps_max"] \
                    and summary[d]["total_pct"][0] > 0:
                ratio = summary[o]["total_pct"][0] / summary[d]["total_pct"][0]
                out[f"{o}_over_{d}_total"] = ratio
                out[f"{o}_beats_{d}"] = ratio <= 0.5     # pre-registered: at most half the cost at equal steps
    if sep and "1" in sep:
        need = sep["1"][max(sep["1"], key=int)]["bits_needed"][0]
        out["separator_w1_bits_needed"] = need
        out["LSB_dead"] = need > 48                    # no 48-bit single-position separator keeps the span within 2%
        out["separator16_dead"] = need > 16            # the CVL strong form and the carry-select scan need <= 16 bits
    return out


def finalize(raws, n_boot=1000):
    """Merge shard files (score_rows records plus config and extras) into the result JSON."""
    cfg = raws[0]["config"]
    for r in raws[1:]:
        assert r["config"] == cfg, "shards were run with different configurations"
    records = {}
    for r in raws:
        for i, rec in r["records"].items():
            assert i not in records, f"row {i} is in two shards"
            records[i] = rec
    missing = sorted(set(range(cfg["rows"])) - set(records))
    assert not missing, f"rows not scored yet: {missing}"
    orders = cfg["orders"]
    per_row = {o: [records[i][o] for i in range(cfg["rows"])] for o in orders}
    summary = summarize(per_row, sum(cfg["block_bytes"]), n_boot=n_boot, seed=cfg["seed"])
    extra = {}
    for r in raws:                                    # shard 0 also holds the AR reference
        for k, v in r.get("extra", {}).items():
            extra.setdefault(k, v)
    validity = {"tc_nonnegative": {o: summary[o]["TC_pct"][2] >= 0 for o in orders}}
    if "context_probe" in extra:
        validity["right_context_used"] = extra["context_probe"]["ratio"] <= 0.95
    result = {**{k: cfg[k] for k in ("oracle", "mask_id", "shift", "rows", "prefix", "block", "groups",
                                     "rows_hash", "seed")}, **extra, "orders": summary}
    if "ar_ref" in extra:
        ratio = summary["l2r"]["bpb"] / extra["ar_ref"]["bpb"]
        result["ar_ref"]["oracle_l2r_bpb_ratio"] = ratio
        validity["oracle_within_20pct_of_ar"] = abs(ratio - 1) <= 0.2
    elif "ar_ref_error" in extra:
        validity["oracle_within_20pct_of_ar"] = False
    sep = None
    if cfg.get("sep"):
        sep = result["separator"] = separator_summary(records, range(cfg["rows"]), cfg["sep"], n_boot=n_boot,
                                                      seed=cfg["seed"])
        whole = [sep[str(w)][str(cfg["sep"]["span"])]["far_past_info_bits"] for w in sorted(cfg["sep"]["windows"])]
        validity["separator_info_nonnegative"] = all(v[2] >= 0 for v in whole)
        validity["separator_info_shrinks_with_window"] = all(a[0] >= b[0] for a, b in zip(whole, whole[1:]))
    result["validity"] = validity
    result["valid"] = all(v if isinstance(v, bool) else all(v.values()) for v in validity.values())
    result["readings"] = readings(summary, sep)
    return result


def report(result, log=print):
    fmt = lambda v: f"{v[0]:6.2f} [{v[1]:5.2f},{v[2]:5.2f}]"
    log(f"{result['oracle']}: {result['rows']} rows of {result['prefix']} + {result['block']} tokens, "
        f"row hash {result['rows_hash']}")
    log(f"\n{'order':10s} {'steps':>5s} {'par':>7s} {'chain':>7s} {'TC % of l2r':>20s} {'gap %':>20s} {'total %':>20s}")
    for o, s in result["orders"].items():
        log(f"{o:10s} {s['steps_max']:5d} {s['par_nats_per_token']:7.4f} {s['chain_nats_per_token']:7.4f} "
            f"{fmt(s['TC_pct']):>20s} {fmt(s['gap_pct']):>20s} {fmt(s['total_pct']):>20s}")
    for o, s in result["orders"].items():
        lp = s.get("lane_profile")
        if lp:
            log(f"{o}: per lane (lanes 1..L-1) deficit {lp['deficit']:.2f}, recovery {lp['recovery']:.2f}, "
                f"net {lp['net']:.2f} nats (same-step TC {lp['tc']:.2f})")
    line = f"oracle l2r bpb {result['orders']['l2r']['bpb']:.4f}"
    if "ar_ref" in result:
        line += f", {result['ar_ref']['name']} {result['ar_ref']['bpb']:.4f} (ratio {result['ar_ref']['oracle_l2r_bpb_ratio']:.3f})"
    log(line)
    if "separator" in result:
        log("\nseparator bound (bits per row; least cost of a B-bit separator, % of the span's NLL):")
        log(f"{'window':>6s} {'span':>5s} {'far-past info':>22s} {'bits needed':>22s} " +
            " ".join(f"{'B=' + str(b):>7s}" for b in SEP_BUDGETS))
        for w, res in result["separator"].items():
            for m, v in res.items():
                log(f"{w:>6s} {m:>5s} {fmt(v['far_past_info_bits']):>22s} {fmt(v['bits_needed']):>22s} " +
                    " ".join(f"{c:7.2f}" for c in v["least_cost_pct"].values()))
    log(f"validity ({'pass' if result['valid'] else 'FAIL: E0 inconclusive for this oracle'}): "
        f"{json.dumps(result['validity'])}")
    log(f"readings: {json.dumps(result['readings'])}")


def _spearman(a, b):
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for rank, i in enumerate(order):
            r[i] = float(rank)
        return r
    ra, rb = ranks(a), ranks(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    va = math.sqrt(sum((x - ma) ** 2 for x in ra))
    vb = math.sqrt(sum((y - mb) ** 2 for y in rb))
    return cov / (va * vb) if va > 0 and vb > 0 else float("nan")


def compare(paths, log=print):
    """The cross-oracle readings: Spearman of the orders' total cost (TC + gap), pre-registered at
    >= 0.8, and bisection's gap >= 5% in both oracles (exact strict orders closed on computation)."""
    res = [json.load(open(p)) for p in paths]
    common = [o for o in res[0]["orders"] if o != "l2r" and all(o in r["orders"] for r in res[1:])]
    totals = [[r["orders"][o]["total_pct"][0] for o in common] for r in res]
    rho = _spearman(totals[0], totals[1])
    log(f"orders compared: {', '.join(common)}")
    for r, t in zip(res, totals):
        log(f"  {r['oracle']:32s} valid={r['valid']} " + " ".join(f"{v:6.2f}" for v in t))
    log(f"Spearman of total cost (TC + gap, % of l2r): {rho:.3f} (pre-registered agreement: >= 0.8)")
    gap_both = all(r["readings"].get("bisect1_gap_ge_5pct", False) for r in res)
    log(f"bisect1 gap >= 5% in every oracle (exact strict orders closed on computation): {gap_both}")
    return {"spearman": rho, "agree": rho >= 0.8, "bisect1_gap_ge_5pct_all": gap_both,
            "all_valid": all(r["valid"] for r in res)}


# ----------------------------------------------------------------------------- main
def main(argv=None, commit=None):
    """argv: command-line arguments (default sys.argv); commit: called after every file write (a Modal
    volume commit, so finished rows survive a lost container)."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--oracle", default="GSAI-ML/LLaDA-8B-Base")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--rows", type=int, default=32)
    p.add_argument("--prefix", type=int, default=128, help="oracle tokens before the block (BOS included)")
    p.add_argument("--block", type=int, default=1024, help="Tb: block positions after its first token")
    p.add_argument("--orders", default=DEFAULT_ORDERS)
    p.add_argument("--groups", type=int, default=0, help="0: exact chain; G > 0: G round-robin groups per step")
    p.add_argument("--batch", type=int, default=8, help="oracle passes per forward call")
    p.add_argument("--shift", type=int, default=-1, help="-1: detect; 0: LLaDA-style; 1: Dream-style")
    p.add_argument("--mask-id", type=int, default=-1)
    p.add_argument("--ar-ref", default="", help="an AR model scored on the same text (validity check)")
    p.add_argument("--sep-cut", type=int, default=512, help="separator bound: the cut's block position (<= 0: off)")
    p.add_argument("--sep-windows", default="0,1,4,16", help="separator bound: tokens kept visible before the cut")
    p.add_argument("--sep-span", type=int, default=256, help="separator bound: tokens scored after the cut")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--raw", default="", help="per-row records, saved after every row; a rerun resumes from them")
    p.add_argument("--out", default="")
    p.add_argument("--merge", nargs="+", default=None, help="raw files of finished shards -> --out")
    p.add_argument("--compare", nargs="+", default=None, help="two finished result JSONs")
    args = p.parse_args(argv)
    commit = commit or (lambda: None)

    def dump(obj):
        if args.out:
            os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
            with open(args.out, "w") as f:
                json.dump(obj, f, indent=1)
            print(f"wrote {args.out}")
            commit()

    def write(result):
        report(result)
        dump(result)

    if args.compare:
        dump(compare(args.compare))
        return
    if args.merge:
        write(finalize([torch.load(f) for f in args.merge]))
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    oracle = MaskedLMOracle.from_pretrained(args.oracle, device, args.shift, args.mask_id)
    P, Tb = args.prefix, args.block
    rows = load_rows(oracle.tok, args.data_dir, args.rows, P + 1 + Tb)
    assert len(rows) == args.rows, f"only {len(rows)} validation documents reach {P + 1 + Tb} tokens"
    orders = [o for o in args.orders.split(",") if o]
    assert orders[0] == "l2r", "l2r is the reference and goes first"
    shift_probe = oracle.detect_shift(rows, P) if args.shift < 0 else None
    known_shift = MaskedLMOracle.KNOWN.get(args.oracle, (None, None))[1]
    assert shift_probe is None or known_shift is None or oracle.shift == known_shift, \
        f"detected shift {oracle.shift} ({shift_probe}), but {args.oracle} predicts with shift {known_shift}"
    probe = context_probe(oracle, rows, P)
    cache = {}
    starts = [sentence_starts(oracle.tok, r, P, cache) for r in rows]
    blocks = [oracle.tok.decode(r[P:].tolist(), skip_special_tokens=True) for r in rows]
    prefixes = [oracle.tok.decode(r[:P].tolist(), skip_special_tokens=True) for r in rows]
    sep = None
    if args.sep_cut > 0:
        sep = {"cut": args.sep_cut, "windows": [int(w) for w in args.sep_windows.split(",") if w],
               "span": args.sep_span, "keep_first": oracle.tok.bos_token_id is not None}
        assert args.sep_cut + args.sep_span <= 1 + Tb and P + args.sep_cut - max(sep["windows"]) >= 0
    config = {"oracle": args.oracle, "mask_id": oracle.mask_id, "shift": oracle.shift, "rows": len(rows),
              "prefix": P, "block": 1 + Tb, "orders": orders, "groups": args.groups, "seed": args.seed,
              "rows_hash": rows_hash(rows), "block_bytes": [len(b.encode("utf-8")) for b in blocks], "sep": sep}
    indices = list(range(args.shard, len(rows), args.num_shards))
    conf_passes = sum(int(o[4:]) for o in orders if o.startswith("conf"))
    passes = len(indices) * (len(orders) * (1 + Tb) + conf_passes + (len(sep["windows"]) * sep["span"] if sep else 0))
    print(f"oracle {args.oracle}: mask {oracle.mask_id}, shift {oracle.shift} (probe {shift_probe}), "
          f"right-context probe {probe}; shard {args.shard}/{args.num_shards}: {len(indices)} of {len(rows)} "
          f"rows of {P} + {1 + Tb} tokens, row hash {config['rows_hash']}, "
          f"{passes:,} oracle passes{' (exact chain)' if args.groups <= 0 else ' at most'}", flush=True)

    done = {}
    if args.raw and os.path.exists(args.raw):
        old = torch.load(args.raw)
        assert old["config"] == config, f"{args.raw} was made with another configuration; move it away to restart"
        done = old["records"]
    extra = {"shift_probe": shift_probe, "context_probe": probe}
    if done and "ar_ref" in old.get("extra", {}) and old["extra"]["ar_ref"]["name"] == args.ar_ref:
        extra["ar_ref"] = old["extra"]["ar_ref"]

    def save(records):
        if args.raw:
            os.makedirs(os.path.dirname(os.path.abspath(args.raw)), exist_ok=True)
            torch.save({"config": config, "records": records, "extra": extra}, args.raw + ".tmp")
            os.replace(args.raw + ".tmp", args.raw)
            commit()

    records = score_rows(oracle, rows, P, Tb, orders, indices, starts, args.groups, args.batch, args.seed,
                         done=done, save=save, log=lambda s: print(s, flush=True), sep=sep)
    if args.ar_ref and args.shard == 0 and "ar_ref" not in extra:
        del oracle.model
        if device == "cuda":
            torch.cuda.empty_cache()
        try:
            extra["ar_ref"] = {"name": args.ar_ref, "bpb": ar_reference_bpb(args.ar_ref, prefixes, blocks, device)}
            print(f"AR reference {args.ar_ref}: {extra['ar_ref']['bpb']:.4f} bpb", flush=True)
        except Exception as e:                        # the scored rows are kept; the check reads as failed
            extra["ar_ref_error"] = f"{args.ar_ref}: {e!r}"
            print(f"AR reference failed: {extra['ar_ref_error']}", flush=True)
    save(records)
    if args.num_shards == 1:
        write(finalize([{"config": config, "records": records, "extra": extra}]))


if __name__ == "__main__":
    main()
