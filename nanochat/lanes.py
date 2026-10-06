"""
Lanes: one trunk pass emits L tokens that are far apart in the same document.

A row of N input positions is a causal prefix of P positions followed by L contiguous lanes of
S = (N - P) / L positions each. The model writes the lanes in lockstep: at step s every lane emits
its s-th token. Every input of step s was drawn before the step began, so a lane input at step s
sees the whole prefix and every lane's inputs up to and including step s; only the step's draws
are mutually independent. Lane j > 0 has no predecessor input of its own at step 0 (its left
neighbour is lane j-1's last token, written last), so that input slot holds a reserved
lane-start token. RoPE keeps the true positions.

The factorisation is exact: every target is a softmax given what was visible when it was drawn,
and the only independence is between tokens of the same step, which sit S positions apart. The
adjacent-token dependence that sank every block head (SAP Stage 0: 47% of block NLL at T=4) never
has to be modelled; what the lanes pay instead is lane j starting j*S tokens beyond the text it
can see.

Training and evaluation pass `lane_mask` to GPT.forward with the lane-start inputs swapped in;
generation runs one all-layer pass per step through GPT._sap_depth_layers with the KV cache as
prefix (generate_lanes), so a block of L*S tokens costs S passes instead of L*S.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

LANE_TOKEN = "<|output_end|>"   # never occurs in pretraining text; its embedding rows become the lane start


def lane_layout(N, P, L):
    """Lane length S and the input positions that hold the lane-start token (lanes 1..L-1)."""
    assert L >= 1 and 0 <= P < N and (N - P) % L == 0, f"N - P = {N - P} must split into {L} lanes"
    S = (N - P) // L
    return S, [P + j * S for j in range(1, L)]


def lane_rank(N, P, L, device=None):
    """Generation step of every input position: p for the prefix, P + s for a lane's step s."""
    S, _ = lane_layout(N, P, L)
    p = torch.arange(N, device=device)
    return torch.where(p < P, p, P + (p - P) % S)


def lane_mask(N, P, L, device=None):
    """(1, 1, N, N) bool, True where query row may read key column: a key input from the same
    generation step or an earlier one (the prefix stays causal, since its ranks are distinct)."""
    r = lane_rank(N, P, L, device)
    return (r[None, :] <= r[:, None])[None, None]


def lagged_lane_mask(N, P, L, lag, device=None):
    """(1, 1, N, N) plain-lanes mask in which a lane sees the other lanes only `lag` steps late:
    a lane row at step s reads its own lane up to step s and every other lane up to step s - lag
    (the prefix in full). lag = 0 is `lane_mask`. With lag >= 1 lanes may advance asynchronously by
    up to `lag` steps (S12 S-3: speculative lanes) without changing any row's conditional."""
    S, _ = lane_layout(N, P, L)
    r = lane_rank(N, P, L, device)
    p = torch.arange(N, device=device)
    lane = torch.where(p < P, torch.full_like(p, -1), (p - P) // S)
    same = lane[None, :] == lane[:, None]
    other_ok = (lane[None, :] < 0) | (r[None, :] <= r[:, None] - lag)
    vis = (r[None, :] <= r[:, None]) & (same | other_ok | (lane[:, None] < 0))
    return vis[None, None]


def prompt_window_mask(N, P, k, device=None):
    """(1, 1, N, N) causal mask restricted to the first P positions (the prompt) plus each
    position's last k positions (itself included). The information oracle for any one-pass
    generator whose dependence on its own output is a short suffix (S07 hypothesis B)."""
    p = torch.arange(N, device=device)
    causal = p[None, :] <= p[:, None]
    return (causal & ((p[None, :] < P) | (p[:, None] - p[None, :] < k)))[None, None]


def lane_inputs(x, P, L, lane_token):
    """Inputs with the lane-start token at each lane j > 0's first position (a copy)."""
    _, starts = lane_layout(x.size(1), P, L)
    x = x.clone()
    if starts:
        x[:, starts] = lane_token
    return x


def sample_prefix_len(max_prefix, L, generator=None):
    """A prefix length in [0, max_prefix] that leaves the row divisible into L lanes."""
    k = int(torch.randint(0, max_prefix // L + 1, (1,), generator=generator).item())
    return k * L


def infill_layout(N, lo=2, hi=64, gap=128, first=64, generator=None):
    """S16-A position-preserving infill rows. One middle span of lo..hi slots inside every gap-slot
    chunk after the first `first` slots, with at least one slot outside on either side. Returns
    (perm, cold): perm (N,) lists the slots outside every middle in position order, then every middle
    in position order; cold lists the first slot after each middle. Every slot still predicts its own
    target, so a model run in perm order with true rotary positions scores an exact factorisation
    in which each middle is drawn after its whole right context (the lookahead lanes' late offsets
    need). A cold slot's own input is the middle's last target, drawn later, so it holds the
    lane-start token, like a lane start."""
    mid = torch.zeros(N, dtype=torch.bool)
    for c in range(first, N, gap):
        room = min(c + gap, N) - c - 2
        if room < lo:
            break
        m = int(torch.randint(lo, min(hi, room) + 1, (1,), generator=generator))
        a = c + 1 + int(torch.randint(0, room - m + 1, (1,), generator=generator))
        mid[a:a + m] = True
    perm = torch.cat([(~mid).nonzero().flatten(), mid.nonzero().flatten()])
    cold = (mid[:-1] & ~mid[1:]).nonzero().flatten() + 1
    return perm, cold


def infill_rows(x, y, perm, cold, lane_token):
    """(inputs, targets, pos_ids) for infill_layout: run the model on these with a causal mask over
    the reordered sequence and pos_ids as the rotary positions."""
    x = x.clone()
    x[:, cold] = lane_token
    return x[:, perm], y[:, perm], perm


class LaneBatches:
    """Wraps (x, y) batches for evaluate_bpb: lane-start inputs swapped in at a fixed prefix."""

    def __init__(self, batches, P, L, lane_token):
        self.batches, self.P, self.L, self.lane_token = batches, P, L, lane_token

    def __iter__(self):
        for x, y in self.batches:
            yield lane_inputs(x, self.P, self.L, self.lane_token), y


PAD_TOKEN = "<|output_start|>"   # never occurs in pretraining text; fills a sentence-aligned lane after its last sentence


def sentence_end_table(tok, V):
    """(V,) bool: token ids whose text ends a sentence (. ! ? before any closing quotes or brackets)."""
    ends = torch.zeros(V, dtype=torch.bool)
    for i in range(V):
        try:
            t = tok.decode([i])
        except Exception:
            continue
        if t.rstrip(" \"')]}\u201d\u2019").endswith((".", "!", "?")):
            ends[i] = True
    return ends


def aligned_lanes_rows(x, y, P, L, lane_token, pad_token, ends, bos, window):
    """Sentence-aligned lanes (S12 S-1) built from standard rows (x, y), on the plain-lanes layout:
    the prefix, then L slots of S = (N - P) / L positions, lane 0's slot continuing the prefix and
    every later slot starting with the lane-start token. Each lane takes the next text and is cut
    at the last sentence end (or just before a document start) inside its final `window` slots;
    with none there it is cut mid-sentence when full, as plain lanes are. The rest of a cut lane is
    padding: its last text token predicts the pad token (the lane-end decision), pads predict pads,
    and the slot's last position predicts nothing. A later lane's first text token is predicted by
    its lane-start row. The text tail that no longer fits is dropped. The ranks, and so the mask,
    are those of plain lanes (`lane_mask(N, P, L)`). Returns inputs (B, N), targets (B, N),
    text tokens placed per row (B,) and lanes cut at a boundary (B,)."""
    import numpy as np
    B, N = x.shape
    S = (N - P) // L
    assert P + L * S == N and window >= 1
    raw = torch.cat([x, y[:, -1:]], 1).cpu().numpy()                           # (B, N + 1) text
    ends_np = ends.cpu().numpy()
    xa = np.full((B, N), pad_token, dtype=np.int64)
    ya = np.full((B, N), -1, dtype=np.int64)
    placed = np.zeros(B, dtype=np.int64)
    aligned = np.zeros(B, dtype=np.int64)
    for b in range(B):
        r = raw[b]
        is_end = ends_np[r]                                                    # (N + 1,) text position ends a sentence
        is_bos = r == bos
        xa[b, :P] = r[:P]
        ya[b, :P] = r[1:P + 1]                       # the prefix's last row predicts lane 0's first token
        c = P
        for j in range(L):
            base = P + j * S
            first = base if j == 0 else base + 1
            cap = S if j == 0 else S - 1
            e, cut = c + cap, False
            for e2 in range(c + cap, max(c + cap - window, c + 1) - 1, -1):
                if is_end[e2 - 1] or is_bos[e2]:
                    e, cut = e2, True
                    break
            n = e - c
            if j > 0:
                xa[b, base] = lane_token
                ya[b, base] = r[c]
            if n > 0:
                xa[b, first:first + n] = r[c:e]
                ya[b, first:first + n - 1] = r[c + 1:e]
                q = first + n - 1
                if q < base + S - 1:                 # room left: the lane ends here, then pads
                    ya[b, q:base + S - 1] = pad_token
            aligned[b] += int(cut)
            c = e
        placed[b] = c - P
    return (torch.from_numpy(xa).to(x.device), torch.from_numpy(ya).to(x.device), torch.from_numpy(placed),
            torch.from_numpy(aligned))


def separator_mask(N, P, m, device=None):
    """S11 separator oracle: (1, 1, N, N) causal mask in which positions P..P+m-1 are summary
    slots that read the first P positions, and every position from P+m on reads only the slots and
    its own half. The second half's access to the first goes through the m slots alone."""
    p = torch.arange(N, device=device)
    q, k = p[:, None], p[None, :]
    return ((k <= q) & ~((q >= P + m) & (k < P)))[None, None]


def separator_batch(x, y, P, m, slot_token):
    """Insert m slot tokens at P and shift the second half right by m (its last m tokens drop), so
    every model sees the same real tokens up to the row end minus m. Targets of the slots, of the
    last first-half position and of the last slot are ignored; second-half targets keep their
    original values (original position k sits at k + m)."""
    if m == 0:
        return x.clone(), y.clone()
    B, N = x.shape
    slots = torch.full((B, m), slot_token, dtype=x.dtype, device=x.device)
    xs = torch.cat([x[:, :P], slots, x[:, P:N - m]], 1)
    ignore = torch.full((B, m + 1), -1, dtype=y.dtype, device=y.device)
    ys = torch.cat([y[:, :max(P - 1, 0)], ignore, y[:, P:N - m]], 1)
    return xs, ys


class SeparatorBatches:
    """Wraps (x, y) batches for evaluate_bpb with the separator layout."""

    def __init__(self, batches, P, m, slot_token):
        self.batches, self.P, self.m, self.slot_token = batches, P, m, slot_token

    def __iter__(self):
        for x, y in self.batches:
            yield separator_batch(x, y, self.P, self.m, self.slot_token)


def check_lane_windows(model, total_len):
    """Lane decoding masks cached keys by visibility, not by sliding window, so every layer's
    window must cover the whole generation (train lane models with window pattern L)."""
    for i, (left, _) in enumerate(model.window_sizes):
        assert left is None or left < 0 or left >= total_len, \
            f"layer {i} has a {left}-token window; lane decoding needs full-context layers"


@torch.no_grad()
def lane_step(model, kv_cache, ids, pos):
    """One lockstep step: the L new inputs (B, L) at positions pos (B, L) pass through every trunk
    layer, reading the cached keys/values (all earlier steps) and each other's (all known before
    the step); their keys/values are then written to the cache. Graph-safe. Returns next-token
    logits (B, L, V)."""
    from nanochat.common import COMPUTE_DTYPE
    from nanochat.gpt import norm
    B, L = ids.shape
    dev = ids.device
    blocks = list(model.transformer.h)
    t_cached = kv_cache.cache_seqlens.to(torch.long)                       # (B,)
    Tmax = kv_cache.k_cache.size(2)
    x0s = norm(model.transformer.wte(ids).to(COMPUTE_DTYPE))
    is_tok = torch.ones(B, L, dtype=torch.bool, device=dev)
    pre = (torch.arange(Tmax, device=dev)[None, None, :] < t_cached[:, None, None]).expand(B, L, Tmax)

    def prefix_kv(j, i, kv_of, rl, xl):
        return kv_cache.get_layer_cache(i)

    def lin(mod, x):
        return F.linear(x, mod.weight.to(dtype=x.dtype))

    cur = []
    xs = model._sap_depth_layers(x0s, x0s, ids, is_tok, pos, prefix_kv, pre,
                                 torch.ones(1, L, L, dtype=torch.bool, device=dev), lambda v: v, lin, blocks,
                                 kv_out=cur, layer_ids=list(range(len(blocks))))
    slot = t_cached[:, None] + torch.arange(L, device=dev)                  # (B, L) insertion index
    for i, (k, v) in enumerate(cur):
        kc, vc = kv_cache.get_layer_cache(i)
        idx = slot[..., None, None].expand(B, L, kc.size(2), kc.size(3))
        kc.scatter_(1, idx, k.to(kc.dtype))
        vc.scatter_(1, idx, v.to(vc.dtype))
    kv_cache.cache_seqlens.add_(L)
    return model._sap_readout(xs)


def _choose(logits, temperature, generator):
    if temperature <= 0:
        return logits.argmax(-1)
    probs = torch.softmax(logits.float() / temperature, dim=-1)
    from nanochat.block_head import pick
    return pick(probs, torch.rand(logits.shape[:-1], device=logits.device, generator=generator))


@torch.inference_mode()
def generate_lanes(model, prompt, L, S, lane_token, temperature=1.0, generator=None, kv_cache=None):
    """Write L lanes of S tokens after prompt (B, P): one causal pass over the prompt (which also
    draws the token at position P, lane 0's first input), then S lockstep steps. Returns
    (B, 1 + L*S): that token, then lane 0's S tokens, lane 1's, ..., in text order."""
    from nanochat.engine import _kv_cache_for
    B, P = prompt.shape
    dev = model.get_device()
    dtype = torch.bfloat16 if dev.type == "cuda" else torch.float32
    check_lane_windows(model, P + L * S)               # input positions; the last drawn token is never fed back
    kv = kv_cache if kv_cache is not None else _kv_cache_for(model, B, P + L * S + 8, dev, dtype)
    logits = model.forward(prompt.to(dev), kv_cache=kv)[:, -1]
    first = _choose(logits, temperature, generator)
    ids = torch.full((B, L), lane_token, dtype=torch.long, device=dev)
    ids[:, 0] = first
    lane_pos = P + torch.arange(L, device=dev) * S                        # (L,) lane j's first input
    out = torch.empty(B, L, S, dtype=torch.long, device=dev)
    for s in range(S):
        nxt = _choose(lane_step(model, kv, ids, (lane_pos + s)[None].expand(B, L)), temperature, generator)
        out[:, :, s] = nxt
        ids = nxt
    return torch.cat([first[:, None], out.reshape(B, L * S)], dim=1)


_LRB_CACHE = {}


def deduce_lane_layout(m):
    """Deduce (P, L) from a bool lane_mask tensor of shape (1, 1, N, N), (1, N, N), or (N, N)."""
    if m.dim() == 4:
        m = m[0, 0]
    elif m.dim() == 3:
        m = m[0]
    N = m.size(0)
    row_counts = m.sum(dim=-1)
    expected_causal = torch.arange(1, N + 1, device=m.device)
    is_causal_row = row_counts == expected_causal
    non_causal = (~is_causal_row).nonzero(as_tuple=True)[0]
    P = non_causal[0].item() if len(non_causal) > 0 else N
    if P == N:
        return P, 1
    tr_at_P = row_counts[P].item()
    L = tr_at_P - P
    return P, max(1, L)


def compute_lrb_buckets(N, P, L, device=None, roles=True):
    """S16-C: Lane-Relative Attention Bias (LRB) bucket mapping and offset layout. roles=False is
    the attribution control: the five lane roles collapse into one, so lane-to-lane pairs keep only
    their offset-gap bucket (0..6) and the bias can no longer tell the next lane from its own.
    Returns:
        vis: (N, N) bool, visibility mask under lockstep lanes.
        buckets: (N, N) int64, bucket index in [0, 43] for visible pairs, -1 where masked.
        offsets: (N,) int64, intra-lane token offset in [0, S-1] (0 for prefix).
    """
    S = (N - P) // L
    p = torch.arange(N, device=device)
    lane = torch.where(p < P, torch.full_like(p, -1), (p - P) // S)
    offset = torch.where(p < P, torch.zeros_like(p), (p - P) % S)
    rank = torch.where(p < P, p, P + offset)
    vis = rank[None, :] <= rank[:, None]

    def dist_bucket(d):
        b = torch.zeros_like(d)
        b = torch.where(d == 0, 0, b)
        b = torch.where(d == 1, 1, b)
        b = torch.where(d == 2, 2, b)
        b = torch.where(d == 3, 3, b)
        b = torch.where((d >= 4) & (d <= 7), 4, b)
        b = torch.where((d >= 8) & (d <= 15), 5, b)
        b = torch.where(d >= 16, 6, b)
        return b

    q_lane = lane[:, None]
    k_lane = lane[None, :]
    q_off = offset[:, None]
    k_off = offset[None, :]

    q_is_pre = q_lane < 0
    k_is_pre = k_lane < 0

    # Lane-to-lane relationships
    dl = k_lane - q_lane
    d_off = (q_off - k_off).clamp(min=0)
    db = dist_bucket(d_off)

    lane_role = torch.zeros_like(dl)
    lane_role = torch.where(dl == 0, 0, lane_role)
    lane_role = torch.where(dl == 1, 1, lane_role)
    lane_role = torch.where(dl == -1, 2, lane_role)
    lane_role = torch.where(dl >= 2, 3, lane_role)
    lane_role = torch.where(dl <= -2, 4, lane_role)
    if not roles:
        lane_role = torch.zeros_like(dl)

    bucket_lane_lane = lane_role * 7 + db

    # Prefix-to-prefix relationships: 8 distance buckets
    d_pre = (p[:, None] - p[None, :]).clamp(min=0)
    pb = torch.where(d_pre == 0, 0,
         torch.where(d_pre == 1, 1,
         torch.where(d_pre == 2, 2,
         torch.where(d_pre == 3, 3,
         torch.where((d_pre >= 4) & (d_pre <= 7), 4,
         torch.where((d_pre >= 8) & (d_pre <= 15), 5,
         torch.where((d_pre >= 16) & (d_pre <= 31), 6, 7)))))))
    bucket_pre_pre = 35 + pb

    # Lane-to-prefix relationship: bucket 43
    bucket_lane_pre = torch.full_like(dl, 43)

    buckets = torch.where(q_is_pre & k_is_pre, bucket_pre_pre,
              torch.where(~q_is_pre & k_is_pre, bucket_lane_pre,
              torch.where(~q_is_pre & ~k_is_pre, bucket_lane_lane, -1)))
    buckets = torch.where(vis, buckets, -1)
    return vis, buckets, offset


def get_lrb_cache(N, P, L, device=None, roles=True):
    """Cached retrieve for LRB tensors to avoid repeated allocations."""
    key = (N, P, L, str(device), roles)
    if key not in _LRB_CACHE:
        _LRB_CACHE[key] = compute_lrb_buckets(N, P, L, device=device, roles=roles)
    return _LRB_CACHE[key]

