"""
S13 SV-A: splice codes, word classes decided first at every lane junction (s13_sap_brainstorm.md).

A plain-lanes model (nanochat/lanes.py) starts every later lane blind: lane j's first tokens are
predicted before lane j-1's last tokens exist, and at equal tokens the previous lane's end recovers
only 13 to 35% of that start deficit (s13, item 3). Splice codes hand the start part of the
missing left context through a low-entropy variable decided first:

  round 0      every later lane j has a code slot that predicts c_j, the Brown class
               (scripts/sap_brown_classes.py) of the junction token e_j = lane j-1's last token,
               from the prompt alone; all codes are drawn in parallel
  rounds 1..S  plain lanes. Lane j's first input is the code (the lane-start token plus the code's
               embedding) instead of the bare lane-start token, so its first prediction knows
               the class of the token before it; lane j-1 writes e_j last, its softmax
               restricted to the tokens of class c_j

Lane 0 continues the prompt and runs in rounds 0..S-1, so a block takes S + 1 rounds. Codes are
functions of the text and the junction token is restricted to its code's class, so
log p(x) = sum_j log p(c_j | prompt) + sum_i log p(x_i | what i sees), exactly: the reported bpb
counts the code nats. Rotary positions are the text positions (a code slot sits at e_j's position).

The layout is built inside GPT.forward(splice=(P, L, lane_token)) from ordinary (x, y) rows, so
the training loop, evaluate_bpb and scripts/sap_position_bpb.py pass one static keyword. With
loss_reduction='none' the losses come back on the original positions, each code's nats added to
lane j's first prediction. Every token id has a class (special tokens included), which keeps the
factorisation normalised: sum_c p(c) sum_{e in c} p(e | c) = 1.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def class_ids(targets, class_of_token, K):
    """(B, T) Brown class of each target; K where the target is ignored (< 0)."""
    cls = class_of_token[targets.clamp_min(0)]
    return torch.where(targets < 0, torch.full_like(cls, K), cls)


_LAYOUTS = {}


@torch.compiler.disable                       # built once per (N, P, L) and cached; not worth tracing
def splice_layout(N, P, L, device=None):
    """Static layout of a spliced row built from an N-position plain-lanes row.

    Returns a dict of (N + L - 1,) tensors over the spliced positions:
      orig   original position of a token slot, -1 for a code slot
      junc   junction index j (1..L-1) of a code slot, 0 elsewhere
      first  lane j >= 1's first token slot (its input becomes the code)
      pos    rotary position (a code slot takes its junction token's position)
      rank   generation step (prefix p; codes P; lane 0 slot o at P + o; lane j slot o at P + 1 + o)
    and `mask_orig`, the (L-1,) original positions whose target is junction token e_j (lane j-1's
    last input), with `junc_orig` = P + j*S, the original input position of e_j.
    """
    key = (N, P, L, str(device))
    if key in _LAYOUTS:
        return _LAYOUTS[key]
    assert L >= 2 and (N - P) % L == 0, f"N - P = {N - P} must split into {L} lanes"
    S = (N - P) // L
    assert S >= 2, "a lane needs two slots so its last one sees the next lane's code"
    orig, junc, first, pos, rank = [], [], [], [], []
    for p in range(P):
        orig.append(p); junc.append(0); first.append(False); pos.append(p); rank.append(p)
    for o in range(S):
        orig.append(P + o); junc.append(0); first.append(False); pos.append(P + o); rank.append(P + o)
    for j in range(1, L):
        start = P + j * S
        orig.append(-1); junc.append(j); first.append(False); pos.append(start); rank.append(P)
        for o in range(S):
            orig.append(start + o); junc.append(0); first.append(o == 0); pos.append(start + o)
            rank.append(P + 1 + o)
    t = lambda v, dt=torch.long: torch.tensor(v, dtype=dt, device=device)
    starts = t([P + j * S for j in range(1, L)])
    lay = {"orig": t(orig), "junc": t(junc), "first": t(first, torch.bool), "pos": t(pos), "rank": t(rank),
           "mask_orig": starts - 1, "junc_orig": starts, "S": S}
    lay["mask"] = (lay["rank"][None, :] <= lay["rank"][:, None])[None, None]
    # integer indices (static shapes, so torch.compile needs no data-dependent sizes)
    slot = {o: i for i, o in enumerate(orig) if o >= 0}
    lay["tok_slots"] = t([i for i, o in enumerate(orig) if o >= 0])
    lay["tok_orig"] = t([o for o in orig if o >= 0])
    lay["first_slots"] = t([i for i, f in enumerate(first) if f])
    lay["code_slots"] = t([i for i, jj in enumerate(junc) if jj > 0])
    lay["mask_slots"] = t([slot[P + j * S - 1] for j in range(1, L)])
    _LAYOUTS[key] = lay
    return lay


class Splice(nn.Module):
    """The code embedding (added to a lane's first input) and the code head (K classes)."""

    def __init__(self, K, d):
        super().__init__()
        self.K = K
        self.emb = nn.Embedding(K + 1, d)                # row K: no code
        self.head = nn.Linear(d, K, bias=False)


def splice_loss(model, x, y, P, L, lane_token, loss_reduction="mean", return_parts=False):
    """Splice-code loss for ordinary rows (x, y) of N positions (y[p] = x[p+1]); see the module
    docstring. 'mean' returns nats per valid token target (codes included); 'none' returns (B*N,)
    on the original positions, each code's nats folded into its junction token's position."""
    B, N = x.shape
    dev = x.device
    sp = model.splice
    K = sp.K
    lay = splice_layout(N, P, L, dev)
    Np = lay["orig"].numel()
    ts, to, fs = lay["tok_slots"], lay["tok_orig"], lay["first_slots"]
    # inputs: original tokens, the lane-start token at code slots and at each later lane's first slot
    xs = torch.full((B, Np), lane_token, dtype=x.dtype, device=dev)
    xs[:, ts] = x[:, to]
    xs[:, fs] = lane_token
    # codes: class of junction token e_j = x[P + j*S] (the target of lane j-1's last input)
    e = y[:, lay["mask_orig"]]                                    # (B, L-1)
    codes = class_ids(e, model.class_of_token, K)                  # K where e_j is ignored
    cin = torch.full((B, Np), K, dtype=torch.long, device=dev)
    cin[:, fs] = codes
    extra = sp.emb(cin)
    extra = extra * (cin < K).unsqueeze(-1).to(extra.dtype)       # no-code rows add nothing
    logits, hidden = model(xs, lane_mask=lay["mask"], pos_ids=lay["pos"], extra_embed=extra,
                           return_hidden=True)
    # token targets on token slots
    ys = torch.full((B, Np), -1, dtype=y.dtype, device=dev)
    ys[:, ts] = y[:, to]
    tl = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), ys.reshape(-1), ignore_index=-1,
                         reduction="none").view(B, -1)
    # restrict each junction token's softmax to its code's class: add lse_class - lse_all
    mslots = lay["mask_slots"]                                     # (L-1,) spliced slots predicting e_j
    lg = logits[:, mslots].float()                                 # (B, L-1, V)
    member = model.class_of_token[None, None, :] == codes.clamp(max=K - 1)[..., None]
    corr = torch.logsumexp(lg.masked_fill(~member, float("-inf")), -1) - torch.logsumexp(lg, -1)
    corr = torch.where((codes < K) & (e >= 0), corr, torch.zeros_like(corr))
    tl[:, mslots] = tl[:, mslots] + corr
    # code nats at the code slots
    cslots = lay["code_slots"]                                     # (L-1,) in junction order 1..L-1
    clog = sp.head(hidden[:, cslots].to(sp.head.weight.dtype)).float()   # (B, L-1, K)
    ctgt = torch.where(codes < K, codes, torch.full_like(codes, -1))
    cl = F.cross_entropy(clog.reshape(-1, K), ctgt.reshape(-1), ignore_index=-1, reduction="none").view(B, -1)
    if return_parts:                                               # diagnostics: token nats on the
        out = torch.zeros(B, N, dtype=tl.dtype, device=dev)           # original positions, code nats
        out[:, to] = tl[:, ts]                                        # per junction
        return out, cl
    if loss_reduction == "none":
        out = torch.zeros(B, N, dtype=tl.dtype, device=dev)
        out[:, to] = tl[:, ts]
        out[:, lay["junc_orig"]] += cl
        return out.reshape(-1)
    n = (ys >= 0).sum().clamp_min(1)
    return (tl.sum() + cl.sum()) / n
