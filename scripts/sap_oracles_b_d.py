"""
scripts/sap_oracles_b_d.py
Mathematical oracles and empirical gates for Proposal D (OU Correlated Noise) and Proposal B (Schrödinger Flow Bridge).

Evaluates:
1. Proposal D Oracle:
   - Empirical lag-k autocorrelation of chunk latents: rho(k) for k in [0, 30].
   - Tests ChunkMeanFlowPrior with OU correlated noise eps ~ N(0, Sigma_OU) across tau in {1, 2, 4, 8, 16}.
2. Proposal B Oracle:
   - Residual variance of interior chunks conditioned on boundary anchors [z_0, z_W] for W in {8, 16, 32}.
   - Tests if bounding anchors pull the interior latent MSE into the ChunkAE tube (MSE <= 0.25).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from nanochat.tokenizer import get_tokenizer
from scripts.sap_chunk_ae import ChunkAE
from scripts.sap_meanflow_chunk import ChunkMeanFlowPrior, compute_ngram_diversity


def sample_ou_noise(B: int, N: int, dz: int, tau: float, device: torch.device) -> torch.Tensor:
    """Generate 1D Ornstein-Uhlenbeck correlated noise across the N sequence positions."""
    if tau <= 0:
        return torch.randn(B, N, dz, device=device)
    # 1D AR(1) process: eps_i = alpha * eps_{i-1} + sqrt(1 - alpha^2) * xi_i
    alpha = math.exp(-1.0 / tau)
    sigma_innov = math.sqrt(1.0 - alpha ** 2)
    eps = []
    curr = torch.randn(B, 1, dz, device=device)
    eps.append(curr)
    for _ in range(1, N):
        curr = alpha * curr + sigma_innov * torch.randn(B, 1, dz, device=device)
        eps.append(curr)
    return torch.cat(eps, dim=1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--chunk-ae-weights", type=str, default="/vol/out/s03_sap/s13_chunk_ae_K8_dz256.pt")
    p.add_argument("--prior-weights", type=str, default="/vol/out/s03_sap/s13_meanflow_K8_dz256_d8.pt")
    p.add_argument("--tokenizer-dir", type=str, default="/vol/tokenizer_sap")
    p.add_argument("--data-dir", type=str, default="/vol/data")
    p.add_argument("--rows", type=int, default=128)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tok = get_tokenizer(args.tokenizer_dir)
    V = tok.get_vocab_size()
    P, N, K, dz = 128, 240, 8, 256

    print("Loading ChunkAE...")
    chunk_ae = ChunkAE(K=K, dz=dz, V=V).to(device)
    ae_ckpt = torch.load(args.chunk_ae_weights, map_location=device, weights_only=True)
    chunk_ae.load_state_dict(ae_ckpt["state_dict"] if "state_dict" in ae_ckpt else ae_ckpt)
    chunk_ae.eval()

    print("Loading ChunkMeanFlowPrior...")
    prior = ChunkMeanFlowPrior(V, P, N, dz, width=512, depth=8, heads=8).to(device)
    prior_ckpt = torch.load(args.prior_weights, map_location=device, weights_only=True)
    prior.load_state_dict(prior_ckpt["state_dict"] if "state_dict" in prior_ckpt else prior_ckpt)
    prior.eval()

    # Load validation data
    loader = tokenizing_distributed_data_loader_bos_bestfit(
        tok, args.rows, 2048, split="val", data_dir=args.data_dir, device=str(device)
    )
    batch, _ = next(loader)
    batch = batch[:args.rows].to(device)
    prompt = batch[:, :P]
    target = batch[:, P:P + N * K]
    B = prompt.size(0)

    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z_true = chunk_ae.encode(target.reshape(-1, K)).reshape(B, N, dz)

    # =========================================================================
    # PART 1: Proposal D (OU Correlated Noise Oracle)
    # =========================================================================
    print("\n" + "=" * 60)
    print("PART 1: PROPOSAL D (CORRELATED NOISE / OU ORACLE)")
    print("=" * 60)

    # 1A. Measure empirical autocorrelation rho(k)
    z_f = z_true.float()  # (B, N, dz)
    autocorr = {}
    for k in [1, 2, 3, 4, 8, 16, 32]:
        dot = (z_f[:, :-k, :] * z_f[:, k:, :]).mean().item()
        autocorr[f"lag_{k}"] = dot

    print(f"Empirical Chunk Latent Autocorrelation:")
    for k, v in autocorr.items():
        print(f"  {k:7s}: {v:.4f}")

    # 1B. Test ChunkMeanFlowPrior with OU noise across tau in {0, 1, 2, 4, 8, 16}
    print("\nEvaluating ChunkMeanFlowPrior with OU correlated noise:")
    print(f"{'tau':<8} | {'Latent MSE':<12} | {'Token Acc':<12} | {'Distinct-1':<12} | {'Distinct-2':<12} | {'Distinct-3':<12}")
    print("-" * 75)

    ou_results = {}
    for tau in [0.0, 1.0, 2.0, 4.0, 8.0, 16.0]:
        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                eps = sample_ou_noise(B, N, dz, tau, device)
                r = torch.zeros(B, device=device)
                t = torch.ones(B, device=device)
                u = prior(eps, r, t, prompt)
                z_pred = eps - u
                z_pred = z_pred * torch.rsqrt(z_pred.pow(2).mean(-1, keepdim=True) + 1e-6)
                logits = chunk_ae.decode(z_pred.reshape(-1, dz))
                pred_toks = logits.argmax(-1).reshape(B, N * K)

        mse = (z_pred.float() - z_true.float()).pow(2).mean().item()
        acc = (pred_toks == target).float().mean().item()
        samples = pred_toks[:16].tolist()
        d1 = compute_ngram_diversity(samples, 1)
        d2 = compute_ngram_diversity(samples, 2)
        d3 = compute_ngram_diversity(samples, 3)

        ou_results[tau] = {"mse": mse, "acc": acc, "d1": d1, "d2": d2, "d3": d3}
        print(f"{tau:<8.1f} | {mse:<12.4f} | {acc * 100:<11.3f}% | {d1:<12.3f} | {d2:<12.3f} | {d3:<12.3f}")

    # =========================================================================
    # PART 2: Proposal B (Schrödinger Flow Bridge Oracle)
    # =========================================================================
    print("\n" + "=" * 60)
    print("PART 2: PROPOSAL B (SCHRÖDINGER FLOW BRIDGE ORACLE)")
    print("=" * 60)
    print("Measuring residual interior MSE given boundary anchors [z_0, z_W]...")

    bridge_results = {}
    # For window sizes W in {8, 16, 32} chunks (64, 128, 256 tokens)
    for W in [8, 16, 32]:
        n_windows = N // W
        z_windows = z_f[:, :n_windows * W, :].reshape(B * n_windows, W, dz)
        
        # Endpoints
        z_start = z_windows[:, 0:1, :]  # (B*n_windows, 1, dz)
        z_end = z_windows[:, W - 1:W, :]  # (B*n_windows, 1, dz)
        
        # Midpoint index
        mid_idx = W // 2
        z_mid_true = z_windows[:, mid_idx, :]  # (B*n_windows, dz)
        
        # 1. Brownian Bridge interpolation predictor: (z_start + z_end) / 2
        z_mid_interp = 0.5 * (z_start[:, 0, :] + z_end[:, 0, :])
        interp_mse = (z_mid_interp - z_mid_true).pow(2).mean().item()
        
        # 2. Optimal linear bridge predictor: fit ridge regression from [z_start, z_end] to z_mid
        X = torch.cat([z_start[:, 0, :], z_end[:, 0, :]], dim=1)  # (M, 2*dz)
        Y = z_mid_true  # (M, dz)
        
        # Solve least squares: W_opt = (X^T X + lambda I)^-1 X^T Y
        reg = 1e-2 * torch.eye(2 * dz, device=device)
        XtX = X.t() @ X + reg
        XtY = X.t() @ Y
        W_opt = torch.linalg.solve(XtX, XtY)
        z_mid_ridge = X @ W_opt
        ridge_mse = (z_mid_ridge - Y).pow(2).mean().item()
        
        # Check token reconstruction at the midpoint with ridge prediction
        with torch.no_grad():
            z_mid_norm = z_mid_ridge * torch.rsqrt(z_mid_ridge.pow(2).mean(-1, keepdim=True) + 1e-6)
            target_mid_toks = target[:, :n_windows * W * K].reshape(B, n_windows, W, K)[:, :, mid_idx, :].reshape(-1, K)
            logits_mid = chunk_ae.decode(z_mid_norm.to(chunk_ae.from_z.weight.dtype))
            pred_mid_toks = logits_mid.argmax(-1)
            mid_acc = (pred_mid_toks == target_mid_toks).float().mean().item()

        bridge_results[W] = {
            "window_chunks": W,
            "window_tokens": W * K,
            "midpoint_interp_mse": interp_mse,
            "midpoint_ridge_mse": ridge_mse,
            "midpoint_token_acc": mid_acc,
        }
        print(f"Window W = {W:2d} chunks ({W * K:3d} tokens):")
        print(f"  Simple Interp Midpoint MSE: {interp_mse:.4f}")
        print(f"  Optimal Ridge Midpoint MSE:  {ridge_mse:.4f}")
        print(f"  Midpoint Token Accuracy:     {mid_acc * 100:.3f}%")

    out_data = {
        "proposal_d_ou": ou_results,
        "proposal_b_bridge": bridge_results,
        "autocorr": autocorr,
    }

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(out_data, f, indent=2)
        print(f"\nSaved oracle summary to {args.out}")


if __name__ == "__main__":
    main()
