"""
Corpus tables for the SAP v4 heads, counted on the FineWeb-Edu training shards.

The tables are the "training-set distributions" the heads may consult: which token pairs occur
adjacent and two apart (C-supp), the top successors of each token by PMI (pmi_chain) and by
p(b | a) (the optional soft-target auxiliary), and corpus token classes (corpus_code). See
nanochat/sap_tables.py for what each one is and why per-slot soft targets alone cannot make a
block coherent.

Documents are tokenized with the run's tokenizer, joined with BOS as in training, and counted
exactly. Only the training split is read, never validation.

    python -m scripts.sap_corpus_tables --tokens 300000000 --out out/s03_sap/tables_V32k.pt
    python -m scripts.sap_corpus_tables --smoke                       # ~1M tokens, checks the path
"""

from __future__ import annotations

import argparse
import os
import time

import torch

from nanochat.dataset import parquets_iter_batched
from nanochat.sap_tables import PairCounter, build_tables, describe, pmi_floor_sanity
from nanochat.tokenizer import get_tokenizer


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=str, default=None)
    p.add_argument("--tokenizer-dir", type=str, default=None)
    p.add_argument("--tokens", type=int, default=300_000_000, help="training tokens to count")
    p.add_argument("--row-len", type=int, default=4096, help="the token stream is cut into rows of this length")
    p.add_argument("--top-m", type=int, default=256)
    p.add_argument("--pmi-min-count", type=int, default=3)
    p.add_argument("--classes", type=int, default=128)
    p.add_argument("--svd-rank", type=int, default=64)
    p.add_argument("--supp-min-store", type=int, default=2,
                   help="keep only pairs seen at least this often in the support lists")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", type=str, default="out/s03_sap/tables_V32k.pt")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.smoke:
        args.tokens = 1_000_000
        args.out = args.out.replace(".pt", "_smoke.pt")

    tok = get_tokenizer(args.tokenizer_dir)
    V, bos = tok.get_vocab_size(), tok.get_bos_token_id()
    adj, skip = PairCounter(V, 1), PairCounter(V, 2)
    buf, seen, t0 = [], 0, time.time()
    for texts in parquets_iter_batched("train", data_dir=args.data_dir):
        for ids in tok.encode(texts, prepend=bos, num_threads=8):
            buf.extend(ids)
        n_rows = len(buf) // args.row_len
        if n_rows:
            rows = torch.tensor(buf[:n_rows * args.row_len], dtype=torch.long,
                                device=args.device).view(n_rows, args.row_len)
            buf = buf[n_rows * args.row_len:]
            adj.add(rows)
            skip.add(rows)
            seen += rows.numel()
            print(f"  counted {seen:,} tokens ({seen / max(1e-9, time.time() - t0):,.0f} tok/s)", flush=True)
        if seen >= args.tokens:
            break
    assert seen > 0, f"no training text found (data dir {args.data_dir!r}); pass --data-dir"
    tables = build_tables(adj.result(), skip.result(), V, top_m=args.top_m,
                          pmi_min_count=args.pmi_min_count, n_classes=args.classes,
                          svd_rank=args.svd_rank, supp_min_store=args.supp_min_store)
    print(describe(tables))
    print(f"tokens with an empty PMI row: {pmi_floor_sanity(tables):.3%}")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(tables, args.out)
    print(f"wrote {args.out} ({os.path.getsize(args.out) / 2 ** 20:.0f} MB)")


if __name__ == "__main__":
    main()
