"""
Brown word classes for the S13 class-information oracle (s13_sap_brainstorm.md, Q1).

Brown clustering assigns every token to one of K classes so that the class bigram model fits
adjacent text as well as possible, which is the same as maximising the mutual information between
the classes of adjacent tokens. That is the criterion the oracle needs: a class that tells a lane
start as much as possible about the token just before it.

Counts are exact adjacent pairs from the training shards (nanochat/sap_tables.py::PairCounter),
joined with BOS as in training. Clustering is the exchange algorithm (Kneser and Ney 1993; Martin,
Liermann and Ney 1998): start from frequency-ranked classes, then move each token, most frequent
first, to the class that most increases

    LL = sum_{c,d} F(M[c, d]) - sum_c F(Nl[c]) - sum_d F(Nr[d]),   F(n) = n log n,

where M is the class bigram count matrix and Nl, Nr its row and column sums. Each pass is exact
and never lowers LL; class MI = (LL + F(T)) / T, with T the number of bigrams.

    python -m scripts.sap_brown_classes --tokenizer-dir tokenizer --data-dir data \\
        --tokens 60000000 --classes 256 --out out/s13/brown256.npy
    python -m scripts.sap_brown_classes --smoke
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import scipy.sparse as sp
import torch

from nanochat.dataset import parquets_iter_batched
from nanochat.sap_tables import PairCounter
from nanochat.tokenizer import get_tokenizer


def count_pairs(tok, data_dir, n_tokens, row_len=4096):
    """Adjacent-pair counts (V x V sparse) over about n_tokens training tokens."""
    V, bos = tok.get_vocab_size(), tok.get_bos_token_id()
    adj = PairCounter(V, 1)
    buf, seen = [], 0
    for texts in parquets_iter_batched("train", data_dir=data_dir):
        for ids in tok.encode(texts, prepend=bos, num_threads=8):
            buf.extend(ids)
        n_rows = len(buf) // row_len
        if n_rows:
            adj.add(torch.tensor(buf[:n_rows * row_len], dtype=torch.long).view(n_rows, row_len))
            seen += n_rows * row_len
            buf = buf[n_rows * row_len:]
        if seen >= n_tokens:
            break
    keys, counts = adj.result()
    a, b = (keys // V).numpy(), (keys % V).numpy()
    return sp.csr_matrix((counts.numpy().astype(np.float64), (a, b)), shape=(V, V)), seen


def _F(x):
    x = np.asarray(x, dtype=np.float64)
    return np.where(x > 0, x * np.log(np.maximum(x, 1e-300)), 0.0)


def class_stats(N, cls, K):
    """Class bigram matrix, its log-likelihood term and the adjacent-class MI (nats)."""
    C = sp.csr_matrix((np.ones(N.shape[0]), (np.arange(N.shape[0]), cls)), shape=(N.shape[0], K))
    M = np.asarray((C.T @ N @ C).todense())
    T = M.sum()
    LL = _F(M).sum() - _F(M.sum(1)).sum() - _F(M.sum(0)).sum()
    return M, LL, (LL + float(_F(T))) / T


def exchange(N, K, passes, log=print, min_count=1):
    """Exact exchange clustering. Returns the (V,) class map and the MI after each pass."""
    V = N.shape[0]
    freq = np.asarray(N.sum(1)).ravel() + np.asarray(N.sum(0)).ravel()
    order = np.argsort(-freq, kind="stable")
    cls = np.full(V, K - 1, dtype=np.int64)            # frequency init: the top K-1 tokens alone,
    cls[order[:K - 1]] = np.arange(K - 1)              # everything else in the last class
    Nc = N.tocsc()
    diag = N.diagonal().astype(np.float64)
    nl = np.asarray(N.sum(1)).ravel()                  # a token's count as the left of a pair
    nr = np.asarray(N.sum(0)).ravel()
    C = sp.csr_matrix((np.ones(V), (np.arange(V), cls)), shape=(V, K))
    R = np.asarray((N @ C).todense())                  # R[u, c]: pairs (u, v) with v in class c
    Lm = np.asarray((N.T @ C).todense())               # Lm[u, c]: pairs (v, u) with v in class c
    M, LL, mi = class_stats(N, cls, K)
    history = [mi]
    log(f"init: MI {mi:.4f} nats")
    active = order[freq[order] >= min_count]
    for it in range(passes):
        moved, t0 = 0, time.time()
        for w in active:
            a = cls[w]
            r = R[w].copy()
            l = Lm[w].copy()
            r[a] -= diag[w]                            # neighbours other than w itself
            l[a] -= diag[w]
            # take w out of class a: every pair involving w leaves M
            M[a, :] -= r
            M[:, a] -= l
            M[a, a] -= diag[w]
            Nl, Nr = M.sum(1), M.sum(0)
            # gain of putting w into each class c (only the classes w has neighbours in change).
            # Pairs (v, w) add l[c'] to every row c' wherever w goes, so the row-sum term that
            # depends on c is F(Nl[c] + l[c] + nl[w]) - F(Nl[c] + l[c]); columns likewise.
            ir, il = np.nonzero(r)[0], np.nonzero(l)[0]
            gain = (_F(M[:, ir] + r[ir][None, :]) - _F(M[:, ir])).sum(1) \
                + (_F(M[il, :] + l[il][:, None]) - _F(M[il, :])).sum(0)
            dM = np.diagonal(M)
            gain += (_F(dM + r + l + diag[w]) - _F(dM)) - (_F(dM + r) - _F(dM)) - (_F(dM + l) - _F(dM))
            gain -= (_F(Nl + l + nl[w]) - _F(Nl + l)) + (_F(Nr + r + nr[w]) - _F(Nr + r))
            b = int(np.argmax(gain))
            M[b, :] += r
            M[:, b] += l
            M[b, b] += diag[w]
            if b != a:
                moved += 1
                cls[w] = b
                col = Nc.getcol(w)                     # pairs (u, w): R[u, .] changes
                rows_u, vals = col.indices, col.data
                R[rows_u, a] -= vals
                R[rows_u, b] += vals
                row = N.getrow(w)                      # pairs (w, u): Lm[u, .] changes
                cols_u, vals = row.indices, row.data
                Lm[cols_u, a] -= vals
                Lm[cols_u, b] += vals
        _, LL, mi = class_stats(N, cls, K)
        history.append(mi)
        log(f"pass {it + 1}: moved {moved}, MI {mi:.4f} nats ({time.time() - t0:.0f}s)")
        if moved == 0:
            break
    return cls, history


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tokenizer-dir", type=str, default="tokenizer")
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--tokens", type=int, default=60_000_000)
    p.add_argument("--classes", type=int, default=256)
    p.add_argument("--passes", type=int, default=4)
    p.add_argument("--out", type=str, default="out/s13/brown256.npy")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.smoke:
        args.tokens, args.classes, args.passes = 300_000, 16, 2
        args.out = args.out.replace(".npy", "_smoke.npy")
    tok = get_tokenizer(args.tokenizer_dir)
    t0 = time.time()
    N, seen = count_pairs(tok, args.data_dir, args.tokens)
    print(f"counted {seen:,} tokens, {N.nnz:,} distinct adjacent pairs ({time.time() - t0:.0f}s)", flush=True)
    rng = np.random.default_rng(0)
    _, _, mi_random = class_stats(N, rng.integers(0, args.classes, N.shape[0]), args.classes)
    cls, history = exchange(N, args.classes, args.passes, log=lambda s: print(s, flush=True))
    sizes = np.bincount(cls, minlength=args.classes)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.save(args.out, cls.astype(np.int32))
    report = {"tokens": seen, "classes": args.classes, "mi_random": mi_random, "mi_by_pass": history,
              "class_size_min": int(sizes.min()), "class_size_median": int(np.median(sizes)),
              "class_size_max": int(sizes.max()), "empty_classes": int((sizes == 0).sum())}
    freq = np.asarray(N.sum(1)).ravel()
    examples = {}
    for c in range(min(args.classes, 40)):
        members = np.where(cls == c)[0]
        top = members[np.argsort(-freq[members])][:8]
        examples[int(c)] = [tok.decode([int(t)]) for t in top]
    report["examples"] = examples
    with open(args.out.replace(".npy", ".json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"adjacent-class MI: random {mi_random:.4f}, Brown {history[-1]:.4f} nats; class sizes "
          f"min {sizes.min()}, median {int(np.median(sizes))}, max {sizes.max()}; saved {args.out}")


if __name__ == "__main__":
    main()
