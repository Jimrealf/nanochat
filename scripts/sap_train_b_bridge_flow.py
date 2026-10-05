"""
scripts/sap_train_b_bridge_flow.py
Proposal B: Schrödinger Flow Bridge / 2-Pass Plan & Infill (Full training beyond oracle).

Mechanism:
Generation happens in exactly 2 sequential forward passes (T=2, ~400x speedup):
  Pass 1 (Anchor Planning): The model predicts M = N // W coarse anchor latents (e.g. W=16, M=15 anchors)
         from the prompt in 1 pass.
  Pass 2 (Bridge Infilling): For every interval k in parallel, conditioned on boundary anchors
         (z_{k*W}, z_{(k+1)*W}), a bridge flow network transports Brownian bridge noise to the
         W - 1 interior chunk latents in 1 single parallel pass.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from nanochat.tokenizer import get_tokenizer
from scripts.sap_chunk_ae import ChunkAE


class JVPTransformerBlock(nn.Module):
    """Transformer block with explicit matmuls compatible with torch.func.jvp forward AD."""
    def __init__(self, width: int, heads: int):
        super().__init__()
        assert width % heads == 0
        self.heads, self.head_dim = heads, width // heads
        self.ln1, self.ln2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.proj = nn.Linear(width, width)
        self.ff = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width))

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).chunk(3, -1)
        def heads(a):
            return a.view(B, T, self.heads, self.head_dim).transpose(1, 2)
        q, k, v = heads(q), heads(k), heads(v)
        att = (q @ k.transpose(-1, -2) / math.sqrt(self.head_dim)).softmax(-1)
        mixed = (att @ v).transpose(1, 2).reshape(B, T, D)
        x = x + self.proj(mixed)
        return x + self.ff(self.ln2(x))


class SchrodingerBridgeFlowPrior(nn.Module):
    def __init__(self, V_prompt: int, P: int, N: int, dz: int, W: int = 16,
                 width: int = 512, depth: int = 8, heads: int = 8):
        super().__init__()
        self.P = P  # prompt tokens (128)
        self.N = N  # total chunk latents (240)
        self.dz = dz
        self.W = W  # window stride (16)
        self.M = N // W  # number of anchors (15)
        self.width = width

        # Prompt & positional embeddings
        self.prompt_emb = nn.Embedding(V_prompt, width)
        self.prompt_pos = nn.Parameter(torch.randn(P, width) * 0.02)

        # Anchor backbone (Pass 1)
        self.anchor_pos = nn.Parameter(torch.randn(self.M, width) * 0.02)
        self.anchor_in = nn.Linear(dz, width)
        self.anchor_blocks = nn.ModuleList([JVPTransformerBlock(width, heads) for _ in range(depth // 2)])
        self.anchor_norm = nn.LayerNorm(width)
        self.anchor_out = nn.Linear(width, dz)

        # Bridge infill network (Pass 2)
        # Conditioned on left anchor, right anchor, and relative position within window
        self.bridge_in = nn.Linear(dz * 3, width)  # [z_t, a_left, a_right]
        self.bridge_pos = nn.Parameter(torch.randn(N, width) * 0.02)
        self.bridge_blocks = nn.ModuleList([JVPTransformerBlock(width, heads) for _ in range(depth)])
        self.bridge_norm = nn.LayerNorm(width)
        self.bridge_out = nn.Linear(width, dz)

        # Time embeddings for MeanFlow
        self.time_emb = nn.Sequential(
            nn.Linear(3, width),
            nn.SiLU(),
            nn.Linear(width, width)
        )

    def forward_anchor(self, z_anchor, r, t, prompt_tokens):
        # Pass 1: generate anchors from prompt
        B = z_anchor.size(0)
        rt = torch.stack((r, t, t - r), -1)
        t_vec = self.time_emb(rt)[:, None, :]

        h_prompt = self.prompt_emb(prompt_tokens) + self.prompt_pos[None, :, :]
        h_anchor = self.anchor_in(z_anchor) + self.anchor_pos[None, :, :] + t_vec
        h = torch.cat([h_prompt, h_anchor], dim=1)
        for block in self.anchor_blocks:
            h = block(h)
        h = self.anchor_norm(h)
        return self.anchor_out(h[:, self.P:, :])

    def forward_bridge(self, z_interp, r, t, a_left, a_right, prompt_tokens):
        # Pass 2: infill interior chunks conditioned on (a_left, a_right)
        B, N_int, dz = z_interp.shape
        rt = torch.stack((r, t, t - r), -1)
        t_vec = self.time_emb(rt)[:, None, :]

        # Concatenate interior latent with its bounding anchors
        feat = torch.cat([z_interp, a_left, a_right], dim=-1)
        h_bridge = self.bridge_in(feat) + self.bridge_pos[None, :N_int, :] + t_vec
        h_prompt = self.prompt_emb(prompt_tokens) + self.prompt_pos[None, :, :]

        h = torch.cat([h_prompt, h_bridge], dim=1)
        for block in self.bridge_blocks:
            h = block(h)
        h = self.bridge_norm(h)
        return self.bridge_out(h[:, self.P:, :])

    def loss(self, prompt_tokens, x_latents):
        # x_latents: (B, N, dz)
        B, N, dz = x_latents.shape
        device = x_latents.device
        W = self.W
        M = self.M

        # 1. Anchors at indices 0, W, 2W, ..., (M-1)W
        anchor_indices = [k * W for k in range(M)]
        anchors_true = x_latents[:, anchor_indices, :]  # (B, M, dz)

        # Anchor MeanFlow loss
        eps_anchor = torch.randn_like(anchors_true)
        a_r = torch.randn(B, device=device).sub_(0.4).sigmoid()
        b_r = torch.randn(B, device=device).sub_(0.4).sigmoid()
        r_a, t_a = torch.minimum(a_r, b_r), torch.maximum(a_r, b_r)
        same_a = torch.rand_like(r_a) >= 0.25
        r_a = torch.where(same_a, t_a, r_a)

        z_t_a = (1 - t_a[:, None, None]) * anchors_true + t_a[:, None, None] * eps_anchor
        v_a = eps_anchor - anchors_true
        fn_a = lambda zz, rr, tt: self.forward_anchor(zz, rr, tt, prompt_tokens)
        u_a, dudt_a = torch.func.jvp(fn_a, (z_t_a, r_a, t_a), (v_a, torch.zeros_like(r_a), torch.ones_like(t_a)))
        target_a = (v_a - (t_a - r_a)[:, None, None] * dudt_a).detach()
        err_a = (u_a - target_a).square().mean((1, 2))
        loss_anchor = ((err_a.detach() + 1e-3).rsqrt() * err_a).mean()

        # 2. Bridge Infill loss for interior chunks
        # Build interior chunks and their bounding anchors
        interior_chunks = []
        left_anchors = []
        right_anchors = []

        for k in range(M - 1):
            left_a = x_latents[:, k * W:k * W + 1, :]  # (B, 1, dz)
            right_a = x_latents[:, (k + 1) * W:(k + 1) * W + 1, :]  # (B, 1, dz)
            inter_z = x_latents[:, k * W + 1:(k + 1) * W, :]  # (B, W-1, dz)
            interior_chunks.append(inter_z)
            left_anchors.append(left_a.expand(-1, W - 1, -1))
            right_anchors.append(right_a.expand(-1, W - 1, -1))

        # Also handle last chunk interval if any
        if (M - 1) * W < N - 1:
            left_a = x_latents[:, (M - 1) * W:(M - 1) * W + 1, :]
            right_a = x_latents[:, -1:, :]
            inter_z = x_latents[:, (M - 1) * W + 1:, :]
            if inter_z.size(1) > 0:
                interior_chunks.append(inter_z)
                left_anchors.append(left_a.expand(-1, inter_z.size(1), -1))
                right_anchors.append(right_a.expand(-1, inter_z.size(1), -1))

        all_interior = torch.cat(interior_chunks, dim=1)  # (B, N_int, dz)
        all_left = torch.cat(left_anchors, dim=1)
        all_right = torch.cat(right_anchors, dim=1)

        # Brownian bridge prior: linear interpolation of endpoints + bridge fluctuation
        # Normalized position s in [0, 1]
        eps_bridge = torch.randn_like(all_interior)
        # Trajectory from bridge prior to data
        a_b = torch.randn(B, device=device).sub_(0.4).sigmoid()
        b_b = torch.randn(B, device=device).sub_(0.4).sigmoid()
        r_b, t_b = torch.minimum(a_b, b_b), torch.maximum(a_b, b_b)
        same_b = torch.rand_like(r_b) >= 0.25
        r_b = torch.where(same_b, t_b, r_b)

        z_t_b = (1 - t_b[:, None, None]) * all_interior + t_b[:, None, None] * eps_bridge
        v_b = eps_bridge - all_interior
        fn_b = lambda zz, rr, tt: self.forward_bridge(zz, rr, tt, all_left, all_right, prompt_tokens)
        u_b, dudt_b = torch.func.jvp(fn_b, (z_t_b, r_b, t_b), (v_b, torch.zeros_like(r_b), torch.ones_like(t_b)))
        target_b = (v_b - (t_b - r_b)[:, None, None] * dudt_b).detach()
        err_b = (u_b - target_b).square().mean((1, 2))
        loss_bridge = ((err_b.detach() + 1e-3).rsqrt() * err_b).mean()

        total_loss = loss_anchor + loss_bridge
        return total_loss, {
            "loss_anchor": float(loss_anchor.item()),
            "loss_bridge": float(loss_bridge.item()),
            "anchor_mse": float(err_a.mean().detach()),
            "bridge_mse": float(err_b.mean().detach()),
        }

    @torch.no_grad()
    def sample_two_pass(self, prompt_tokens):
        # Pass 1: Sample anchors
        B = prompt_tokens.size(0)
        device = prompt_tokens.device
        W = self.W
        M = self.M
        dz = self.dz

        eps_anchor = torch.randn(B, M, dz, device=device)
        u_a = self.forward_anchor(eps_anchor, torch.zeros(B, device=device), torch.ones(B, device=device), prompt_tokens)
        anchors_pred = eps_anchor - u_a
        anchors_pred = anchors_pred * torch.rsqrt(anchors_pred.pow(2).mean(-1, keepdim=True) + 1e-6)

        # Pass 2: Infill interior chunks in parallel
        interior_chunks = []
        left_anchors = []
        right_anchors = []

        for k in range(M - 1):
            left_a = anchors_pred[:, k:k + 1, :]
            right_a = anchors_pred[:, k + 1:k + 2, :]
            left_anchors.append(left_a.expand(-1, W - 1, -1))
            right_anchors.append(right_a.expand(-1, W - 1, -1))

        if (M - 1) * W < self.N - 1:
            n_rem = self.N - 1 - (M - 1) * W
            left_a = anchors_pred[:, -1:, :]
            right_a = anchors_pred[:, -1:, :]
            left_anchors.append(left_a.expand(-1, n_rem, -1))
            right_anchors.append(right_a.expand(-1, n_rem, -1))

        all_left = torch.cat(left_anchors, dim=1)
        all_right = torch.cat(right_anchors, dim=1)
        N_int = all_left.size(1)

        eps_bridge = torch.randn(B, N_int, dz, device=device)
        u_b = self.forward_bridge(eps_bridge, torch.zeros(B, device=device), torch.ones(B, device=device),
                                   all_left, all_right, prompt_tokens)
        interior_pred = eps_bridge - u_b
        interior_pred = interior_pred * torch.rsqrt(interior_pred.pow(2).mean(-1, keepdim=True) + 1e-6)

        # Stitch full sequence of N latents
        full_latents = torch.empty(B, self.N, dz, device=device)
        curr_int_idx = 0
        for k in range(M):
            full_latents[:, k * W, :] = anchors_pred[:, k, :]
            if k < M - 1:
                full_latents[:, k * W + 1:(k + 1) * W, :] = interior_pred[:, curr_int_idx:curr_int_idx + (W - 1), :]
                curr_int_idx += (W - 1)

        if curr_int_idx < N_int:
            full_latents[:, (M - 1) * W + 1:, :] = interior_pred[:, curr_int_idx:, :]

        return full_latents


def compute_ngram_diversity(token_lists, n=3):
    total, unique = 0, set()
    for seq in token_lists:
        for i in range(len(seq) - n + 1):
            ngram = tuple(seq[i:i + n])
            unique.add(ngram)
            total += 1
    return len(unique) / max(1, total)


def evaluate_prior(prior, chunk_ae, val_loader, P, N, K, V, dz, val_rows, device, tok):
    prior.eval()
    chunk_ae.eval()

    total_evaluated = 0
    latent_mse_sum = 0.0
    token_correct_sum = 0
    total_tokens = 0
    samples = []

    with torch.no_grad():
        while total_evaluated < val_rows:
            batch, _ = next(val_loader)
            batch = batch.to(device)
            B = min(batch.size(0), val_rows - total_evaluated)
            batch = batch[:B]

            prompt = batch[:, :P]
            target_tokens = batch[:, P:P + N * K]

            # True latents from frozen ChunkAE
            x_chunks = target_tokens.reshape(B * N, K)
            z_true = chunk_ae.encode(x_chunks).reshape(B, N, dz)

            # Sample two pass
            z_pred = prior.sample_two_pass(prompt)
            latent_mse_sum += F.mse_loss(z_pred, z_true, reduction="sum").item() / dz

            # Decode
            logits = chunk_ae.decode(z_pred.reshape(B * N, dz))
            pred_tokens = logits.argmax(dim=-1).reshape(B, N * K)

            token_correct_sum += (pred_tokens == target_tokens).sum().item()
            total_tokens += target_tokens.numel()

            for b in range(B):
                samples.append({
                    "prompt_tokens": prompt[b].tolist(),
                    "generated_tokens": pred_tokens[b].tolist(),
                    "target_tokens": target_tokens[b].tolist(),
                    "prompt_text": tok.decode(prompt[b].tolist()),
                    "generated_text": tok.decode(pred_tokens[b].tolist()),
                    "target_text": tok.decode(target_tokens[b].tolist()),
                })

            total_evaluated += B

    avg_latent_mse = latent_mse_sum / (total_evaluated * N)
    avg_token_acc = token_correct_sum / total_tokens
    gen_list = [s["generated_tokens"] for s in samples]
    d1 = compute_ngram_diversity(gen_list, 1)
    d2 = compute_ngram_diversity(gen_list, 2)
    d3 = compute_ngram_diversity(gen_list, 3)

    return {
        "latent_mse": avg_latent_mse,
        "token_accuracy": avg_token_acc,
        "diversity": {"d1": d1, "d2": d2, "d3": d3},
        "samples": samples,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-ae-path", type=str, default="/vol/out/s03_sap/s13_chunk_ae_K8_dz256.pt")
    parser.add_argument("--tokenizer-dir", type=str, default="tokenizer_sap")
    parser.add_argument("--data-dir", type=str, default="/vol/data")
    parser.add_argument("--out-dir", type=str, default="out/s03_sap")
    parser.add_argument("--tag", type=str, default="s13_b_bridge_flow_W16")
    parser.add_argument("--W", type=int, default=16)
    parser.add_argument("--K", type=int, default=8)
    parser.add_argument("--P", type=int, default=128)
    parser.add_argument("--N", type=int, default=240)
    parser.add_argument("--dz", type=int, default=256)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--val-rows", type=int, default=128)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = get_tokenizer(args.tokenizer_dir)
    V = tok.get_vocab_size()

    if args.smoke:
        args.steps = 5
        args.batch_size = 4
        args.val_rows = 4
        print(f"Running smoke test on {device}...")

    # Load frozen ChunkAE
    print(f"Loading frozen ChunkAE from {args.chunk_ae_path}...")
    chunk_ae = ChunkAE(V, args.K, d=512, dz=args.dz, layers=2).to(device)
    if os.path.exists(args.chunk_ae_path):
        ckpt = torch.load(args.chunk_ae_path, map_location=device, weights_only=True)
        if isinstance(ckpt, dict) and "model" in ckpt:
            ckpt = ckpt["model"]
        chunk_ae.load_state_dict(ckpt)
        print("ChunkAE loaded and frozen.")
    else:
        print(f"Warning: {args.chunk_ae_path} not found. Running with initialized ChunkAE.")
    chunk_ae.eval()
    for p in chunk_ae.parameters():
        p.requires_grad = False

    prior = SchrodingerBridgeFlowPrior(
        V_prompt=V, P=args.P, N=args.N, dz=args.dz, W=args.W,
        width=args.width, depth=args.depth, heads=args.heads
    ).to(device)
    params = sum(p.numel() for p in prior.parameters())
    print(f"SchrodingerBridgeFlowPrior: W={args.W} M={args.N // args.W} P={args.P} N={args.N} dz={args.dz} width={args.width} depth={args.depth} heads={args.heads} params={params:,}")

    optim = torch.optim.AdamW(prior.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)

    T_total = args.P + args.N * args.K
    train_loader = tokenizing_distributed_data_loader_bos_bestfit(
        tok, args.batch_size, T_total, split="train", data_dir=args.data_dir, device=str(device)
    )
    val_loader = tokenizing_distributed_data_loader_bos_bestfit(
        tok, min(args.batch_size, 8), T_total, split="val", data_dir=args.data_dir, device=str(device)
    )

    t0 = time.time()
    tokens_seen = 0

    for step in range(args.steps):
        prior.train()
        batch, _ = next(train_loader)
        batch = batch.to(device)
        prompt = batch[:, :args.P]
        chunks = batch[:, args.P:args.P + args.N * args.K]

        with torch.no_grad():
            B = batch.size(0)
            x_chunks = chunks.reshape(B * args.N, args.K)
            z_clean = chunk_ae.encode(x_chunks).reshape(B, args.N, args.dz)

        loss, metrics = prior.loss(prompt, z_clean)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(prior.parameters(), 1.0)
        optim.step()

        tokens_seen += batch.numel()
        if step % 100 == 0 or step == args.steps - 1 or args.smoke:
            dt = time.time() - t0
            print(f"step {step:5d}/{args.steps} | loss {loss.item():.4f} (anc={metrics['loss_anchor']:.4f}, bri={metrics['loss_bridge']:.4f}) | mse_a={metrics['anchor_mse']:.3f} mse_b={metrics['bridge_mse']:.3f} | {tokens_seen/1e6:.1f}M tokens | {dt:.0f}s")

    print("Evaluating 2-pass generation on validation set...")
    val_metrics = evaluate_prior(
        prior, chunk_ae, val_loader, args.P, args.N, args.K, V, args.dz,
        args.val_rows, device, tok
    )
    print(f"Validation latent MSE: {val_metrics['latent_mse']:.4f}")
    print(f"Validation token accuracy (two-pass): {val_metrics['token_accuracy']:.5f}")
    d = val_metrics["diversity"]
    print(f"Sample diversity: d1={d['d1']:.3f} d2={d['d2']:.3f} d3={d['d3']:.3f}")

    if val_metrics["samples"]:
        s0 = val_metrics["samples"][0]
        print(f"Sample generation (first 100 chars):")
        print(f"  Prompt:    {repr(s0['prompt_text'][:80])}")
        print(f"  Generated: {repr(s0['generated_text'][:80])}")
        print(f"  Target:    {repr(s0['target_text'][:80])}")

    os.makedirs(args.out_dir, exist_ok=True)
    out_pt = os.path.join(args.out_dir, f"{args.tag}.pt")
    out_jsonl = os.path.join(args.out_dir, f"{args.tag}_samples.jsonl")
    out_json = os.path.join(args.out_dir, f"{args.tag}.json")

    torch.save({"model": prior.state_dict(), "args": vars(args), "val_metrics": {k: v for k, v in val_metrics.items() if k != "samples"}}, out_pt)
    print(f"Saved prior weights to {out_pt}")

    with open(out_jsonl, "w") as f:
        for s in val_metrics["samples"]:
            f.write(json.dumps(s) + "\n")
    print(f"Saved generated samples to {out_jsonl}")

    with open(out_json, "w") as f:
        json.dump({
            "args": vars(args),
            "latent_mse": val_metrics["latent_mse"],
            "token_accuracy": val_metrics["token_accuracy"],
            "diversity": val_metrics["diversity"],
        }, f, indent=2)
    print(f"Saved summary metrics to {out_json}")


if __name__ == "__main__":
    main()
