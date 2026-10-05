"""
Window bisection (S11, s11_sap_tl_brainstorm.md): an exact-likelihood generation order in which a
block of tokens after a left-to-right prefix is produced in about n * (ceil(log2 T) + 1) parallel
steps instead of T.

Order: the bisection midpoints level by level; at each midpoint m the window m-n+1..m (its
positions not yet placed), one token per step inside a window, every window of a level in
parallel. The windows' last tokens separate the block the way a hidden state does
(scripts/sap_bridge_oracle.py: with exact conditionals, 2-token windows sample the toy HMM with 0%
invalid blocks at T=64, single tokens with 44%).

Training is two-stream (XLNet): the row is laid out twice, a content stream (the tokens) and a
query stream (a mask token at the same positions, same rotary positions). Content attends to
contents of the same or earlier steps; a query attends to contents of strictly earlier steps and to
itself, and predicts its own token. One pass scores every token under
    p(x) = prod_steps prod_{k in step} p(x_k | x at earlier steps),
a normalised distribution for any parameters (tests/test_wbisect.py), so the bpb is exact.
"""
from __future__ import annotations

import torch

from nanochat.bridge import window_bisection_levels


def wb_steps(N, P, n):
    """(N,) generation step of every row position: the prefix 0..P-1 left to right (steps -P..-1),
    the block P..N-1 by window bisection with n-token windows (steps 0, 1, ...)."""
    steps = torch.empty(N, dtype=torch.long)
    steps[:P] = torch.arange(P) - P
    if N > P:
        _, block = window_bisection_levels(N - P, n)
        steps[P:] = block
    return steps


def bridged_lanes_steps(N, P, L, n):
    """(N,) generation step of every row position for bridged lanes: the prefix 0..P-1 left to
    right; the block P..N-1 split into L contiguous intervals; first each interval's last n tokens
    (its separator window) are placed, intervals taken coarse-to-fine in bisection order, one token
    per step inside a window and every window of a level in parallel; then every interval is filled
    left to right in lockstep, one position per interval per step, between its left neighbour's
    separator (its left context) and its own (its right end). Steps: (ceil(log2 L) + 1) * n +
    (interval length - n). Same-step tokens are conditionally independent given earlier steps."""
    from nanochat.bridge import bisection_levels
    steps = torch.full((N,), -1, dtype=torch.long)
    steps[:P] = torch.arange(P) - P
    Tb = N - P
    assert 1 <= L <= Tb
    lens = [Tb // L + (1 if i < Tb % L else 0) for i in range(L)]
    starts = [P + sum(lens[:i]) for i in range(L)]
    seps = [list(range(starts[i] + lens[i] - min(n, lens[i]), starts[i] + lens[i])) for i in range(L)]
    base = 0
    for lvl in bisection_levels(L)[0]:
        width = 0
        for i in lvl:
            for j, k in enumerate(seps[i]):
                steps[k] = base + j
            width = max(width, len(seps[i]))
        base += width
    for i in range(L):
        fill = [k for k in range(starts[i], starts[i] + lens[i]) if k not in seps[i]]
        for j, k in enumerate(fill):
            steps[k] = base + j
    assert (steps[P:] >= 0).all()
    return steps


def lane_order_steps(N, P, L):
    """(N,) steps of S08 plain lanes written as a two-stream order (the same conditionals as the
    single-stream lane model): the block's first token right after the prefix (step 0); the rest
    of the block split into L lanes of S positions, lane j covering P + j*S .. P + j*S + S - 1; at
    step 1 + s every lane writes its offset 1 + s; each lane j >= 1's offset 0 (its junction with
    lane j - 1's end) comes last, at step S, with both sides known. Steps: S + 1."""
    steps = torch.full((N,), -1, dtype=torch.long)
    steps[:P] = torch.arange(P) - P
    Tb = N - P
    assert Tb % L == 0, f"{Tb} block positions must split into {L} lanes"
    S = Tb // L
    off = (torch.arange(Tb) % S)
    steps[P:] = torch.where(off == 0, torch.full_like(off, S), off)
    steps[P] = 0                                    # lane 0 starts from the prefix: no junction
    return steps


def seeded_lanes_steps(N, P, K, m=1):
    """(N,) steps of seeded bidirectional lanes (middle-out lanes): the block split into K intervals
    of l positions. In each interval a seed window of m tokens is written left to right from step 0
    (the right front's start); from step m the right front continues rightward and a left front
    writes leftward from the window's left end, so the left front starts with m tokens of context;
    each interval's last position (its junction with the next interval's left front) comes last,
    with both sides known. Intervals run in lockstep. m = 1 is middle-out from one seed token: K
    contextless starts for the tokens per step of 2K plain lanes; larger m warms the left front up
    at the cost of about m / 2 extra steps. Steps: m + max(right, left) + 1, right + left = l - 1 - m."""
    steps = torch.full((N,), -1, dtype=torch.long)
    steps[:P] = torch.arange(P) - P
    Tb = N - P
    assert Tb % K == 0, f"{Tb} block positions must split into {K} intervals"
    l = Tb // K
    assert 1 <= m <= l - 2, f"seed window {m} must leave room for a left front and a junction in {l}"
    left = (l - 1 - m) // 2                         # left-front length; the right tail gets the rest
    right = l - 1 - m - left
    last = m + max(left, right)
    for k in range(K):
        a = P + k * l
        c = a + left                                # the seed window's first position
        for j in range(m):
            steps[c + j] = j
        for s in range(right):
            steps[c + m + s] = m + s
        for s in range(left):
            steps[c - 1 - s] = m + s
        steps[a + l - 1] = last
    assert (steps[P:] >= 0).all()
    return steps


def two_stream_mask(steps, device=None):
    """(1, 1, 2N, 2N) visibility (True = attend): content rows see contents of steps <= their own;
    query rows see contents of steps < their own, and themselves."""
    s = steps.to(device)
    N = s.numel()
    none = torch.zeros(N, N, dtype=torch.bool, device=device)
    top = torch.cat([s[None, :] <= s[:, None], none], 1)
    bottom = torch.cat([s[None, :] < s[:, None], torch.eye(N, dtype=torch.bool, device=device)], 1)
    return torch.cat([top, bottom], 0)[None, None]


def two_stream_batch(x, mask_token):
    """Inputs (B, 2N): the row, then the mask token at every position. Targets (B, N): each query
    position's own token (position 0, the row's first token, is ignored)."""
    idx = torch.cat([x, torch.full_like(x, mask_token)], 1)
    tgt = x.clone()
    tgt[:, 0] = -1
    return idx, tgt


def pos_ids(N, device=None):
    return torch.arange(N, device=device).repeat(2)


class WBBatches:
    """Wraps (x, y) batches for evaluate_bpb: two-stream inputs and own-token targets."""

    def __init__(self, batches, mask_token):
        self.batches, self.mask_token = batches, mask_token

    def __iter__(self):
        for x, _ in self.batches:
            yield two_stream_batch(x, self.mask_token)


_CACHE = {}


def _mask_and_pos(steps, device):
    """Two-stream mask and position ids of the last step order seen. The cache holds the steps tensor
    itself and matches it by identity and version, so a freed order's memory reused by a different
    order (a data_ptr key's failure, which scored later models of one eval under the first model's
    order) or an in-place edit cannot return a stale mask."""
    hit = _CACHE.get("last")
    ver = -1 if steps.is_inference() else steps._version   # inference tensors keep no version counter
    if hit is None or hit[0] is not steps or hit[1] != ver or hit[2] != str(device):
        hit = (steps, ver, str(device), two_stream_mask(steps, device), pos_ids(steps.numel(), device))
        _CACHE["last"] = hit
    return hit[3], hit[4]


def wb_forward(model, x, steps, mask_token, loss_reduction="mean"):
    """Two-stream training/eval call: the exact window-bisection NLL of each row's tokens (the
    first token of the row is not scored). With loss_reduction='none' it is (B * N,) in position
    order, position k scoring token k."""
    N = x.size(1)
    idx, tgt = two_stream_batch(x, mask_token)
    mask, pid = _mask_and_pos(steps, x.device)
    return model(idx, tgt, loss_reduction=loss_reduction, lane_mask=mask, pos_ids=pid, head_from=N)


class WBDecoder:
    """Cached two-stream decoding in a step order (wb_steps or bridged_lanes_steps). After a prefill
    of the prompt's content, step s runs one pass over the previous step's tokens (content, whose
    keys and values are then cached) and the current step's queries, and draws the current tokens.
    Every step has static shapes, so each can be captured as its own CUDA graph."""

    def __init__(self, model, B, steps, mask_token, temperature=1.0, generator=None):
        import torch.nn.functional as F
        from nanochat.engine import _kv_cache_for
        self.model, self.B, self.mask_token = model, B, mask_token
        self.temperature, self.generator = temperature, generator
        self.dev = dev = model.get_device()
        dtype = torch.bfloat16 if dev.type == "cuda" else torch.float32
        self.steps = steps.to(dev)
        self.N = N = steps.numel()
        self.P = int((steps < 0).sum())
        self.kv = _kv_cache_for(model, B, N + 8, dev, dtype)
        self.blocks = list(model.transformer.h)
        self.layer_ids = list(range(len(self.blocks)))
        self.lin = lambda mod, x: F.linear(x, mod.weight.to(dtype=x.dtype))
        self.out = torch.zeros(B, N, dtype=torch.long, device=dev)
        self.plan, prev = [], self.steps.new_zeros(0)
        for st in sorted(set(self.steps[self.P:].tolist())):
            q = (self.steps == st).nonzero().flatten()
            nc, nq = prev.numel(), q.numel()
            vis = torch.zeros(nc + nq, nc + nq, dtype=torch.bool, device=dev)
            vis[:nc, :nc] = True                           # last step's content: itself and its siblings
            vis[nc:, :nc] = True                           # queries: last step's content
            vis[nc:, nc:] = torch.eye(nq, dtype=torch.bool, device=dev)
            self.plan.append((prev, q, torch.cat([prev, q])[None].expand(B, nc + nq).contiguous(), vis,
                              torch.full((B, nq), mask_token, dtype=torch.long, device=dev)))
            prev = q

    def _run(self, ids, pos, vis):
        from nanochat.common import COMPUTE_DTYPE
        from nanochat.gpt import norm
        B, Q = ids.shape
        kv = self.kv
        t_cached = kv.cache_seqlens.to(torch.long)
        Tmax = kv.k_cache.size(2)
        x0s = norm(self.model.transformer.wte(ids).to(COMPUTE_DTYPE))
        pre = (torch.arange(Tmax, device=self.dev)[None, None, :] < t_cached[:, None, None]).expand(B, Q, Tmax)
        cur = []
        xs = self.model._sap_depth_layers(x0s, x0s, ids, torch.ones(B, Q, dtype=torch.bool, device=self.dev), pos,
                                          lambda j, i, *a: kv.get_layer_cache(i), pre, vis[None], lambda v: v,
                                          self.lin, self.blocks, kv_out=cur, layer_ids=self.layer_ids)
        return xs, cur

    def _append(self, cur, n):                             # cache the first n slots' keys and values
        if n == 0:
            return
        kv = self.kv
        slot = kv.cache_seqlens.to(torch.long)[:, None] + torch.arange(n, device=self.dev)
        for i, (k, v) in enumerate(cur):
            kc, vc = kv.get_layer_cache(i)
            idx = slot[..., None, None].expand(self.B, n, kc.size(2), kc.size(3))
            kc.scatter_(1, idx, k[:, :n].to(kc.dtype))
            vc.scatter_(1, idx, v[:, :n].to(vc.dtype))
        kv.cache_seqlens.add_(n)

    def prefill(self, prompt):
        P = self.P
        self.kv.reset()
        self.out.zero_()
        self.out[:, :P] = prompt.to(self.dev)
        _, cur = self._run(self.out[:, :P], torch.arange(P, device=self.dev)[None].expand(self.B, P),
                           torch.tril(torch.ones(P, P, dtype=torch.bool, device=self.dev)))
        self._append(cur, P)

    def step(self, s, teacher=None):
        """Run step s; returns its query logits. Draws (or, with teacher, copies) its tokens."""
        from nanochat.lanes import _choose
        prev, q, pos, vis, masks = self.plan[s]
        nc = prev.numel()
        xs, cur = self._run(torch.cat([self.out[:, prev], masks], 1), pos, vis)
        self._append(cur, nc)
        lg = self.model._sap_readout(xs[:, nc:])
        self.out[:, q] = teacher[:, q].to(self.dev) if teacher is not None else \
            _choose(lg, self.temperature, self.generator)
        return lg


@torch.no_grad()
def wb_generate(model, prompt, steps, mask_token, temperature=1.0, generator=None, teacher=None,
                return_logits=False):
    """Generate a row in a two-stream step order with one cached pass per step (WBDecoder).
    prompt (B, P) fills the left-to-right prefix. teacher (B, N) feeds those tokens instead of
    sampling (tests). Returns tokens (B, N) [and [(positions, logits)] per step]."""
    dec = WBDecoder(model, prompt.size(0), steps, mask_token, temperature, generator)
    dec.prefill(prompt)
    trace = []
    for s in range(len(dec.plan)):
        lg = dec.step(s, teacher)
        if return_logits:
            trace.append((dec.plan[s][1], lg))
    return (dec.out.clone(), trace) if return_logits else dec.out.clone()
