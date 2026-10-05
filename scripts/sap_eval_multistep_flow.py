"""
scripts/sap_eval_multistep_flow.py
Evaluate the trained ChunkMeanFlowPrior across different Euler/MeanFlow integration step counts S in {1, 2, 4, 8, 16}.

Answers Option 2 immediately: Does multi-step integration over the learned flow field
reduce the latent MSE from 1.97 into the ChunkAE tolerance tube (MSE <= 0.25)?
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from nanochat.tokenizer import get_tokenizer
from scripts.sap_chunk_ae import ChunkAE
from scripts.sap_meanflow_chunk import ChunkMeanFlowPrior, compute_ngram_diversity


@torch.no_grad()
def sample_flow_multistep(prior: ChunkMeanFlowPrior, prompt_tokens: torch.Tensor, steps: int = 1) -> torch.Tensor:
    B = prompt_tokens.size(0)
    device = prompt_tokens.device
    z = torch.randn(B, prior.N, prior.dz, device=device)
    
    if steps == 1:
        r = torch.zeros(B, device=device)
        t = torch.ones(B, device=device)
        u = prior(z, r, t, prompt_tokens)
        z = z - u
    else:
        dt = 1.0 / steps
        for step in range(steps):
            t_curr = 1.0 - step * dt
            t_next = max(0.0, 1.0 - (step + 1) * dt)
            r = torch.full((B,), t_next, device=device)
            t = torch.full((B,), t_curr, device=device)
            # The prior predicts average velocity along [r, t]
            u = prior(z, r, t, prompt_tokens)
            z = z - (t_curr - t_next) * u
            
    # Unit RMS normalization
    z = z * torch.rsqrt(z.pow(2).mean(-1, keepdim=True) + 1e-6)
    return z


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--chunk-ae-weights", type=str, required=True)
    p.add_argument("--prior-weights", type=str, required=True)
    p.add_argument("--tokenizer-dir", type=str, default="tokenizer_sap")
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--K", type=int, default=8)
    p.add_argument("--dz", type=int, default=256)
    p.add_argument("--prompt-len", type=int, default=128)
    p.add_argument("--gen-len", type=int, default=1920)
    p.add_argument("--prior-width", type=int, default=512)
    p.add_argument("--prior-depth", type=int, default=8)
    p.add_argument("--prior-heads", type=int, default=8)
    p.add_argument("--val-rows", type=int, default=128)
    p.add_argument("--steps-list", type=str, default="1,2,4,8,16")
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tok = get_tokenizer(args.tokenizer_dir)
    V = tok.get_vocab_size()
    P, N, K, dz = args.prompt_len, args.gen_len // args.K, args.K, args.dz

    print(f"Loading ChunkAE from {args.chunk_ae_weights}...")
    chunk_ae = ChunkAE(K=K, dz=dz, V=V).to(device)
    ae_ckpt = torch.load(args.chunk_ae_weights, map_location=device, weights_only=True)
    chunk_ae.load_state_dict(ae_ckpt["state_dict"] if "state_dict" in ae_ckpt else ae_ckpt)
    chunk_ae.eval()

    print(f"Loading ChunkMeanFlowPrior from {args.prior_weights}...")
    prior = ChunkMeanFlowPrior(V_prompt=V, P=P, N=N, dz=dz, width=args.prior_width, depth=args.prior_depth, heads=args.prior_heads).to(device)
    prior_ckpt = torch.load(args.prior_weights, map_location=device, weights_only=True)
    prior.load_state_dict(prior_ckpt["state_dict"] if "state_dict" in prior_ckpt else prior_ckpt)
    prior.eval()

    steps_to_eval = [int(s.strip()) for s in args.steps_list.split(",") if s.strip()]

    # Collect validation batch
    val_loader = tokenizing_distributed_data_loader_bos_bestfit(
        tok, args.val_rows, 2048, split="val", data_dir=args.data_dir, device=str(device)
    )
    val_batch, _ = next(val_loader)
    val_batch = val_batch[:args.val_rows].to(device)

    prompt = val_batch[:, :P]
    target_tokens = val_batch[:, P:P + N * K]
    B = prompt.size(0)

    # Encode ground truth latents
    target_chunks = target_tokens.reshape(B * N, K)
    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            z_true = chunk_ae.encode(target_chunks).reshape(B, N, dz)

    print(f"\nEvaluating {B} validation rows across integration steps {steps_to_eval}...")
    results = {}
    print(f"{'Steps':<8} | {'Latent MSE':<12} | {'Token Acc':<12} | {'Distinct-1':<12} | {'Distinct-2':<12} | {'Distinct-3':<12} | {'Time (s)':<10}")
    print("-" * 88)

    for S in steps_to_eval:
        t0 = time.time()
        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                z_pred = sample_flow_multistep(prior, prompt, steps=S)
                logits = chunk_ae.decode(z_pred.reshape(B * N, dz))
                pred_tokens = logits.argmax(-1).reshape(B, N * K)

        dt = time.time() - t0
        mse = (z_pred.float() - z_true.float()).pow(2).mean().item()
        acc = (pred_tokens == target_tokens).float().mean().item()

        sample_tokens = pred_tokens[:16].tolist()
        d1 = compute_ngram_diversity(sample_tokens, 1)
        d2 = compute_ngram_diversity(sample_tokens, 2)
        d3 = compute_ngram_diversity(sample_tokens, 3)

        results[S] = {
            "steps": S,
            "latent_mse": mse,
            "token_accuracy": acc,
            "distinct_1": d1,
            "distinct_2": d2,
            "distinct_3": d3,
            "eval_time_sec": dt,
        }
        print(f"{S:<8} | {mse:<12.4f} | {acc * 100:<11.3f}% | {d1:<12.3f} | {d2:<12.3f} | {d3:<12.3f} | {dt:<10.2f}")

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved results to {args.out}")


if __name__ == "__main__":
    main()
