"""
Matched training-FLOPs budget for the SAP sweeps.

Prints ONE number to stdout: the total training FLOPs of the DENSE arm at the given depth,
i.e. its Chinchilla token budget (the same one scripts/code_head_budget.py gives the SCH
sweeps) times its FLOPs per token.

Why FLOPs and not tokens. A block head costs extra FLOPs per token (T slot rows through
the shared unembedding on sap_block_frac of the positions, plus its decoder). The claim
under test is a position on the bpb-vs-training-FLOPs curve, so every SAP arm gets the
dense arm's FLOPs via --target-flops and base_train converts that into fewer tokens for
the arms that cost more per token. Pinning tokens instead would hand the block-head arms
free compute.

    DENSE_FLOPS=$(python3 -m scripts.sap_budget --depth 8 --tokenizer-dir tokenizer)

Diagnostics go to stderr so command substitution stays clean.
"""

from __future__ import annotations

import argparse
import sys

from scripts.code_head_budget import build_dense_meta


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--depth", type=int, required=True)
    p.add_argument("--ratio", type=float, default=10.5, help="Chinchilla tokens:params ratio")
    p.add_argument("--aspect-ratio", type=int, default=64)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--model-dim", type=int, default=0)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--window-pattern", type=str, default="SSSL")
    p.add_argument("--vocab-size", type=int, default=0, help="0 = read it from the tokenizer")
    p.add_argument("--tokenizer-dir", type=str, default=None)
    p.add_argument("--round-to", type=int, default=262144)
    args = p.parse_args()

    vocab_size = args.vocab_size
    if vocab_size <= 0:
        from nanochat.tokenizer import get_tokenizer
        vocab_size = get_tokenizer(tokenizer_dir=args.tokenizer_dir).get_vocab_size()
    model, config = build_dense_meta(args.depth, vocab_size, args.aspect_ratio, args.head_dim,
                                     args.model_dim, args.seq_len, args.window_pattern)
    sp = model.num_scaling_params()
    tokens = int(args.ratio * (sp["transformer_matrices"] + sp["lm_head"]))
    if args.round_to > 1:
        tokens = (tokens // args.round_to) * args.round_to
    flops_per_token, _, _ = model.estimate_flops()
    total = float(flops_per_token) * tokens
    print(f"dense d{config.n_layer} d_model={config.n_embd} V={vocab_size}: "
          f"{flops_per_token:,.0f} FLOPs/token x {tokens:,} tokens = {total:.6e}", file=sys.stderr)
    print(f"{total:.6e}")


if __name__ == "__main__":
    main()
