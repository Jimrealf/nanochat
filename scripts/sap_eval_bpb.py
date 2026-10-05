"""
Evaluate the SAP block head's bpb next to the trunk's next-token bpb on the same tokens from a saved checkpoint.
"""
from __future__ import annotations

import argparse
import json
import torch

from nanochat.checkpoint_manager import build_model, find_last_step
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from nanochat.tokenizer import RustBPETokenizer, get_token_bytes
from nanochat.block_head import evaluate_block_bpb


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint-dir", type=str, required=True)
    p.add_argument("--step", type=int, default=None)
    p.add_argument("--tokenizer-dir", type=str, required=True)
    p.add_argument("--data-dir", type=str, required=True)
    p.add_argument("--device-batch-size", type=int, default=8)
    p.add_argument("--eval-tokens", type=int, default=1048576, help="number of tokens to evaluate on (~1M)")
    p.add_argument("--blocks-per-row", type=int, default=1)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    step = args.step if args.step is not None else find_last_step(args.checkpoint_dir)
    print(f"Loading checkpoint {args.checkpoint_dir} at step {step}...")
    model, tokenizer, meta = build_model(args.checkpoint_dir, step, device, phase="eval",
                                         tokenizer_dir=args.tokenizer_dir)
    model.eval()

    token_bytes = get_token_bytes(device=device, tokenizer_dir=args.tokenizer_dir)
    val_loader = tokenizing_distributed_data_loader_bos_bestfit(
        tokenizer, args.device_batch_size, model.config.sequence_len,
        split="val", data_dir=args.data_dir
    )
    tokens_per_step = args.device_batch_size * model.config.sequence_len
    eval_steps = max(1, args.eval_tokens // tokens_per_step)
    print(f"Running evaluate_block_bpb over {eval_steps} steps ({eval_steps * tokens_per_step:,} tokens)...")

    results = evaluate_block_bpb(model, val_loader, eval_steps, token_bytes,
                                 blocks_per_row=args.blocks_per_row)
    print(f"SAP evaluation results for step {step}:")
    print(json.dumps(results, indent=2))

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"step": step, **results}, f, indent=2)
        print(f"Wrote results to {args.out}")


if __name__ == "__main__":
    main()
