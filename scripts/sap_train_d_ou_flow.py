"""
scripts/sap_train_d_ou_flow.py
Proposal D: Position-Coupled OU Flow Training (Full training beyond oracle).

Mechanism:
Instead of independent white noise across chunks, the prior is an Ornstein-Uhlenbeck (OU)
Gaussian process across sequence positions:
  Cov(eps_{m, d}, eps_{m', d'}) = delta_{d, d'} * exp(-|m - m'| / tau)
The model is trained from scratch with this position-coupled noise prior using JVP forward-mode AD.
At test time, the model samples from the OU prior and maps to chunk latents in 1 single forward pass.
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


def sample_ou_noise(B: int, N: int, dz: int, tau: float = 4.0, device: str | torch.device = "cpu") -> torch.Tensor:
    """Sample an exact AR(1) Ornstein-Uhlenbeck Gaussian process across N sequence positions."""
    if tau <= 0.0:
        return torch.randn(B, N, dz, device=device)
    alpha = math.exp(-1.0 / tau)
    beta = math.sqrt(1.0 - alpha**2)
    eps = torch.empty(B, N, dz, device=device)
    eps[:, 0] = torch.randn(B, dz, device=device)
    for m in range(1, N):
        eps[:, m] = alpha * eps[:, m - 1] + beta * torch.randn(B, dz, device=device)
    return eps


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


class ChunkOUFlowPrior(nn.Module):
    def __init__(self, V_prompt: int, P: int, N: int, dz: int, tau: float = 4.0,
                 width: int = 512, depth: int = 8, heads: int = 8):
        super().__init__()
        self.P = P  # prompt tokens (128)
        self.N = N  # chunk latents (240)
        self.dz = dz  # chunk latent dim (256)
        self.tau = tau
        self.width = width

        self.prompt_emb = nn.Embedding(V_prompt, width)
        self.prompt_pos = nn.Parameter(torch.randn(P, width) * 0.02)

        self.latent_in = nn.Linear(dz, width)
        self.latent_pos = nn.Parameter(torch.randn(N, width) * 0.02)

        self.time_emb = nn.Sequential(
            nn.Linear(3, width),
            nn.SiLU(),
            nn.Linear(width, width)
        )

        self.blocks = nn.ModuleList([JVPTransformerBlock(width, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(width)
        self.out = nn.Linear(width, dz)

    def forward(self, z, r, t, prompt_tokens):
        B = z.size(0)
        rt = torch.stack((r, t, t - r), -1)  # (B, 3)
        t_vec = self.time_emb(rt)[:, None, :]  # (B, 1, width)

        h_prompt = self.prompt_emb(prompt_tokens) + self.prompt_pos[None, :, :]
        h_latents = self.latent_in(z) + self.latent_pos[None, :, :] + t_vec

        h = torch.cat([h_prompt, h_latents], dim=1)
        for block in self.blocks:
            h = block(h)
        h = self.norm(h)
        out = self.out(h[:, self.P:, :])
        return out

    def loss(self, prompt_tokens, x_latents):
        B, N, dz = x_latents.shape
        device = x_latents.device
        eps = sample_ou_noise(B, N, dz, tau=self.tau, device=device)

        # Sample time intervals
        a = torch.randn(B, device=device).sub_(0.4).sigmoid()
        b = torch.randn(B, device=device).sub_(0.4).sigmoid()
        r, t = torch.minimum(a, b), torch.maximum(a, b)
        same = torch.rand_like(r) >= 0.25
        r = torch.where(same, t, r)

        # Linear trajectory: z_t = (1 - t) x + t eps
        t_expand = t[:, None, None]
        z_t = (1 - t_expand) * x_latents + t_expand * eps
        v = eps - x_latents

        fn = lambda zz, rr, tt: self.forward(zz, rr, tt, prompt_tokens)
        u, dudt = torch.func.jvp(fn, (z_t, r, t), (v, torch.zeros_like(r), torch.ones_like(t)))

        dt_expand = (t - r)[:, None, None]
        target = (v - dt_expand * dudt).detach()
        err = (u - target).square().mean((1, 2))
        weight = (err.detach() + 1e-3).rsqrt()
        loss = (weight * err).mean()
        return loss, {"flow_mse": float(err.mean().detach())}

    @torch.no_grad()
    def sample_one_pass(self, prompt_tokens):
        B = prompt_tokens.size(0)
        device = prompt_tokens.device
        eps = sample_ou_noise(B, self.N, self.dz, tau=self.tau, device=device)
        r = torch.zeros(B, device=device)
        t = torch.ones(B, device=device)
        u = self.forward(eps, r, t, prompt_tokens)
        x_hat = eps - u
        x_hat = x_hat * torch.rsqrt(x_hat.pow(2).mean(-1, keepdim=True) + 1e-6)
        return x_hat


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

            # Sample one pass
            z_pred = prior.sample_one_pass(prompt)
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
    parser.add_argument("--tag", type=str, default="s13_d_ou_flow_tau4")
    parser.add_argument("--tau", type=float, default=4.0)
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

    prior = ChunkOUFlowPrior(
        V_prompt=V, P=args.P, N=args.N, dz=args.dz, tau=args.tau,
        width=args.width, depth=args.depth, heads=args.heads
    ).to(device)
    params = sum(p.numel() for p in prior.parameters())
    print(f"ChunkOUFlowPrior: tau={args.tau} P={args.P} N={args.N} dz={args.dz} width={args.width} depth={args.depth} heads={args.heads} params={params:,}")

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
            print(f"step {step:5d}/{args.steps} | loss {loss.item():.4f} | flow_mse {metrics['flow_mse']:.4f} | {tokens_seen/1e6:.1f}M tokens | {dt:.0f}s")

    print("Evaluating one-pass generation on validation set...")
    val_metrics = evaluate_prior(
        prior, chunk_ae, val_loader, args.P, args.N, args.K, V, args.dz,
        args.val_rows, device, tok
    )
    print(f"Validation latent MSE: {val_metrics['latent_mse']:.4f}")
    print(f"Validation token accuracy (one-pass): {val_metrics['token_accuracy']:.5f}")
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
