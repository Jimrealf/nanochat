"""
Corpus tables for the SAP v4 heads: the user's "training-set distribution tables".

Per-slot soft targets cannot make a block coherent (any loss that is a sum of per-slot terms
is minimised by matching each slot's marginal), so the tables enter the heads in the three
places where corpus statistics can carry dependence between slots:

  supp_keys / supp2_keys   every (a, b) pair seen adjacent / two apart, with its count. A CRF
                           chain can forbid pairs never seen in the corpus (C-supp).
  pmi_ids / pmi_vals       per token, its top-M successors by pointwise mutual information.
                           pmi_chain adds them to the next slot's logits (B-PMI).
  classes / class_pmi      a token-class map from k-means on PPMI-SVD vectors, and the class
                           bigram PMI. corpus_code uses the classes as an exact latent.
  big_ids / big_p          per token, its top-M successors with p(b | a), for the optional
                           n-gram soft-target auxiliary (C-soft).

Counts are exact (torch.unique over int64 pair keys a * V + b), accumulated chunk by chunk so
a few hundred million tokens fit in memory. The builder runs on whatever device the tokens are
on; the returned tables are CPU tensors.
"""

from __future__ import annotations

import math
import warnings

import torch
import torch.nn.functional as F


def _pair_keys(seqs, V, gap):
    """int64 keys a * V + b for every (x_i, x_{i+gap}) inside each row of seqs (B, L)."""
    a, b = seqs[:, :-gap], seqs[:, gap:]
    return (a.long() * V + b.long()).reshape(-1)


def _merge(keys, counts):
    uniq, inv = torch.unique(keys, return_inverse=True)
    return uniq, torch.zeros(uniq.numel(), dtype=torch.long, device=keys.device).index_add_(0, inv, counts)


class PairCounter:
    """Exact (a, b) counts at a fixed gap, accumulated over chunks of sequences."""

    def __init__(self, V, gap, flush_every=50_000_000):
        self.V, self.gap, self.flush_every = V, gap, flush_every
        self.keys, self.counts, self.pending = [], [], 0

    def add(self, seqs):
        if seqs.size(1) <= self.gap:
            return
        k, c = torch.unique(_pair_keys(seqs, self.V, self.gap), return_counts=True)
        self.keys.append(k)
        self.counts.append(c.long())
        self.pending += k.numel()
        if self.pending > self.flush_every:
            self._flush()

    def _flush(self):
        if len(self.keys) > 1:
            k, c = _merge(torch.cat(self.keys), torch.cat(self.counts))
            self.keys, self.counts = [k], [c]
        self.pending = self.keys[0].numel() if self.keys else 0

    def result(self):
        self._flush()
        if not self.keys:
            z = torch.zeros(0, dtype=torch.long)
            return z, z
        return self.keys[0], self.counts[0]


def _top_per_row(rows, cols, score, V, M):
    """For each row id, the M columns with the largest score. Returns (V, M) ids and scores.

    Rows with fewer than M entries are padded with column 0 and score 0, which is a no-op for
    the additive uses below.
    """
    dev = rows.device
    order = torch.argsort(score, descending=True, stable=True)
    order = order[torch.argsort(rows[order], stable=True)]       # by row, best score first
    r, c, s = rows[order], cols[order], score[order]
    start = torch.searchsorted(r, torch.arange(V, device=dev))
    rank = torch.arange(r.numel(), device=dev) - start[r]
    keep = rank < M
    ids = torch.zeros(V, M, dtype=torch.long, device=dev)
    val = torch.zeros(V, M, dtype=torch.float32, device=dev)
    ids[r[keep], rank[keep]] = c[keep]
    val[r[keep], rank[keep]] = s[keep].float()
    return ids, val


def _kmeans(x, C, iters, seed):
    """Spherical k-means on rows of x (n, k). Returns labels (n,)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = F.normalize(x.float(), dim=-1)
    cent = x[torch.randperm(x.size(0), generator=g)[:C].to(x.device)].clone()
    labels = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
    for _ in range(iters):
        labels = (x @ cent.t()).argmax(-1)
        new = torch.zeros_like(cent).index_add_(0, labels, x)
        empty = new.norm(dim=-1) < 1e-8
        new[empty] = cent[empty]                               # keep a centroid that lost its points
        cent = F.normalize(new, dim=-1)
    return labels


def build_tables(counts_adj, counts_skip, V, top_m=256, pmi_min_count=3, n_classes=256,
                 svd_rank=64, kmeans_iters=30, seed=0, supp_min_store=1):
    """Tables from exact pair counts. counts_* are (keys, counts) from PairCounter.result().

    supp_min_store drops pairs seen fewer times from the stored support lists (singletons are
    most distinct pairs on web text); the heads can only threshold at or above it.
    """
    keys, cnt = counts_adj
    dev = keys.device
    a, b = keys // V, keys % V
    n_tot = cnt.sum().clamp_min(1).double()
    left = torch.zeros(V, dtype=torch.float64, device=dev).index_add_(0, a, cnt.double())
    right = torch.zeros(V, dtype=torch.float64, device=dev).index_add_(0, b, cnt.double())

    # p(b | a), top-M successors
    p_cond = cnt.double() / left[a].clamp_min(1)
    big_ids, big_p = _top_per_row(a, b, p_cond, V, top_m)
    big_p = big_p / big_p.sum(-1, keepdim=True).clamp_min(1e-12)

    # PMI on pairs seen often enough to be trusted, top-M per predecessor
    pmi = torch.log(cnt.double() * n_tot / (left[a] * right[b]).clamp_min(1))
    ok = cnt >= pmi_min_count
    pmi_ids, pmi_vals = _top_per_row(a[ok], b[ok], pmi[ok].clamp(max=10.0), V, top_m)

    # Token classes: k-means on PPMI-SVD vectors (successor and predecessor profiles).
    pos = ok & (pmi > 0)
    vals = pmi[pos].float()
    with warnings.catch_warnings():      # torch warns about its own sparse invariant checks
        warnings.simplefilter("ignore")
        M = torch.sparse_coo_tensor(torch.stack([a[pos], b[pos]]), vals, (V, V),
                                    check_invariants=False).coalesce()
        k = min(svd_rank, V - 1)
        U, S, Vt = torch.svd_lowrank(M, q=k, niter=4)
    vec = torch.cat([U * S.sqrt(), Vt * S.sqrt()], dim=-1)
    C = min(n_classes, V)
    classes = _kmeans(vec, C, kmeans_iters, seed)
    cc = torch.zeros(C * C, dtype=torch.float64, device=dev).index_add_(0, classes[a] * C + classes[b], cnt.double())
    cc = cc.view(C, C) + 0.5
    class_pmi = torch.log(cc * cc.sum() / (cc.sum(1, keepdim=True) * cc.sum(0, keepdim=True)))

    skeys, scnt = counts_skip
    key_dtype = torch.int32 if V * V < 2 ** 31 else torch.long
    keep_a, keep_s = cnt >= supp_min_store, scnt >= supp_min_store
    return {
        "V": torch.tensor(V), "n_tokens": torch.tensor(int(n_tot.item())),
        "supp_min_store": torch.tensor(supp_min_store),
        "supp_keys": keys[keep_a].to(key_dtype).cpu(), "supp_counts": cnt[keep_a].int().cpu(),
        "supp2_keys": skeys[keep_s].to(key_dtype).cpu(), "supp2_counts": scnt[keep_s].int().cpu(),
        "pmi_ids": pmi_ids.int().cpu(), "pmi_vals": pmi_vals.half().cpu(),
        "big_ids": big_ids.int().cpu(), "big_p": big_p.half().cpu(),
        "classes": classes.cpu(), "class_pmi": class_pmi.float().cpu(),
    }


def tables_from_sequences(seqs, V, chunk_rows=4096, **kw):
    """Convenience for an in-memory (B, L) token tensor (the synthetic pool, tests)."""
    adj, skip = PairCounter(V, 1), PairCounter(V, 2)
    for i in range(0, seqs.size(0), chunk_rows):
        adj.add(seqs[i:i + chunk_rows])
        skip.add(seqs[i:i + chunk_rows])
    return build_tables(adj.result(), skip.result(), V, **kw)


def seen(keys_sorted, a, b, V):
    """Bool tensor: was (a, b) in the sorted key array? a, b broadcast together."""
    q = (a.long() * V + b.long()).to(keys_sorted.dtype)
    if keys_sorted.numel() == 0:
        return torch.zeros_like(q, dtype=torch.bool)
    pos = torch.searchsorted(keys_sorted, q.reshape(-1).contiguous()).clamp(max=keys_sorted.numel() - 1)
    return (keys_sorted[pos] == q.reshape(-1)).view(q.shape)


def describe(tables):
    """One-line summary for logs."""
    V = int(tables["V"])
    cover = (tables["pmi_vals"].float() != 0).float().sum(-1).mean().item()
    return (f"tables: V={V} tokens={int(tables['n_tokens']):,} pairs={tables['supp_keys'].numel():,} "
            f"skip-pairs={tables['supp2_keys'].numel():,} classes={int(tables['classes'].max()) + 1} "
            f"pmi successors/token={cover:.1f}")


def pmi_floor_sanity(tables):
    """Fraction of tokens whose PMI row is empty (they get no B-PMI coupling)."""
    return float((tables["pmi_vals"].float().abs().sum(-1) == 0).float().mean())


__all__ = ["PairCounter", "build_tables", "tables_from_sequences", "seen", "describe",
           "pmi_floor_sanity", "math"]
