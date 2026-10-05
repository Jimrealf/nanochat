"""
scripts/sap_meanflow_chunk.py
S13 Branch A: One-Pass MeanFlow Prior over Chunk Latents (SV-D, true T=L).

Architecture:
1. Frozen ChunkAE: compresses K=8 tokens into a single 256-dim continuous latent vector z (unit RMS).
2. ChunkMeanFlowPrior: given prompt tokens x[:P] (P=128), trains a one-step velocity field u(z_t, r, t, prompt)
   over N = 1920 / 8 = 240 chunk latents simultaneously using JVP forward-mode AD.
3. At test time: draws eps ~ N(0, I) and in ONE SINGLE forward pass emits all 240 chunk latents:
      x_hat = eps - u(eps, r=0, t=1, prompt)
   The frozen ChunkAE decodes all 1920 tokens in parallel. Literal T=L in 1 pass.
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


class ChunkMeanFlowPrior(nn.Module):
    def __init__(self, V_prompt: int, P: int, N: int, dz: int, width: int = 512, depth: int = 8, heads: int = 8):
        super().__init__()
        self.P = P  # prompt tokens (128)
        self.N = N  # chunk latents (240)
        self.dz = dz  # chunk latent dim (256)
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
        # z: (B, N, dz)
        # r: (B,)
        # t: (B,)
        # prompt_tokens: (B, P)
        B = z.size(0)
        rt = torch.stack((r, t, t - r), -1)  # (B, 3)
        t_vec = self.time_emb(rt)[:, None, :]  # (B, 1, width)
        
        h_prompt = self.prompt_emb(prompt_tokens) + self.prompt_pos[None, :, :]
        h_latents = self.latent_in(z) + self.latent_pos[None, :, :] + t_vec
        
        # Concatenate prompt and chunk latents: (B, P + N, width)
        h = torch.cat([h_prompt, h_latents], dim=1)
        for block in self.blocks:
            h = block(h)
        h = self.norm(h)
        # Latent predictions from the N chunk positions
        out = self.out(h[:, self.P:, :])
        return out

    def loss(self, prompt_tokens, x_latents):
        # x_latents: (B, N, dz) clean chunk latents from frozen ChunkAE
        B, N, dz = x_latents.shape
        device = x_latents.device
        eps = torch.randn_like(x_latents)
        
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
        # One-step generation from pure Gaussian noise: r=0, t=1
        B = prompt_tokens.size(0)
        device = prompt_tokens.device
        eps = torch.randn(B, self.N, self.dz, device=device)
        r = torch.zeros(B, device=device)
        t = torch.ones(B, device=device)
        u = self.forward(eps, r, t, prompt_tokens)
        x_hat = eps - u
        # Unit RMS normalization
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
    pos_correct_sum = torch.zeros(N * K, dtype=torch.float64)
    samples = []
    
    with torch.no_grad():
        while total_evaluated < val_rows:
            batch, _ = next(val_loader)
            batch = batch.to(device)
            B = min(batch.size(0), val_rows - total_evaluated)
            batch = batch[:B]
            
            prompt = batch[:, :P]
            target_tokens = batch[:, P:P + N * K]
            
            # Ground truth latents
            target_chunks = target_tokens.reshape(B * N, K)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                z_true = chunk_ae.encode(target_chunks).reshape(B, N, dz)
            
            # One-pass generation
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                z_pred = prior.sample_one_pass(prompt)
                logits = chunk_ae.decode(z_pred.reshape(B * N, dz))  # (B*N, K, V)
            
            pred_tokens = logits.argmax(-1).reshape(B, N * K)
            
            # Metrics
            mse = (z_pred.float() - z_true.float()).pow(2).mean().item()
            latent_mse_sum += mse * B
            
            hit = (pred_tokens == target_tokens)
            token_correct_sum += int(hit.sum().item())
            pos_correct_sum += hit.double().sum(0).cpu()
            total_tokens += B * N * K
            total_evaluated += B
            
            # Record first 16 samples for text inspection
            if len(samples) < 16:
                for b in range(min(B, 16 - len(samples))):
                    p_toks = prompt[b].tolist()
                    g_toks = pred_tokens[b].tolist()
                    t_toks = target_tokens[b].tolist()
                    samples.append({
                        "prompt_tokens": p_toks,
                        "generated_tokens": g_toks,
                        "target_tokens": t_toks,
                        "prompt_text": tok.decode(p_toks),
                        "generated_text": tok.decode(g_toks[:128]),
                        "target_text": tok.decode(t_toks[:128]),
                    })
    
    all_gen_tokens = [s["generated_tokens"] for s in samples]
    d1 = compute_ngram_diversity(all_gen_tokens, 1)
    d2 = compute_ngram_diversity(all_gen_tokens, 2)
    d3 = compute_ngram_diversity(all_gen_tokens, 3)
    
    prior.train()
    return {
        "val_rows": total_evaluated,
        "latent_mse": latent_mse_sum / total_evaluated,
        "token_accuracy": token_correct_sum / total_tokens,
        "distinct_1": d1,
        "distinct_2": d2,
        "distinct_3": d3,
        "samples": samples,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--chunk-ae-weights", type=str, required=True, help="Path to trained ChunkAE .pt weights")
    p.add_argument("--tokenizer-dir", type=str, default="tokenizer_sap")
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--K", type=int, default=8)
    p.add_argument("--dz", type=int, default=256)
    p.add_argument("--ae-d", type=int, default=512)
    p.add_argument("--ae-layers", type=int, default=2)
    p.add_argument("--prompt-len", type=int, default=128)
    p.add_argument("--gen-len", type=int, default=1920)
    p.add_argument("--prior-width", type=int, default=512)
    p.add_argument("--prior-depth", type=int, default=8)
    p.add_argument("--prior-heads", type=int, default=8)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--rows", type=int, default=32, help="batch size in 2048-token sequences")
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--val-rows", type=int, default=128)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--out-samples", type=str, default=None)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    if args.smoke:
        args.steps = 20
        args.rows = 2
        args.val_rows = 4
        args.prior_width = 128
        args.prior_depth = 2
        args.prior_heads = 4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = get_tokenizer(args.tokenizer_dir)
    V = tok.get_vocab_size()

    # Load frozen ChunkAE
    print(f"Loading frozen ChunkAE from {args.chunk_ae_weights}...", flush=True)
    chunk_ae = ChunkAE(V, args.K, d=args.ae_d, dz=args.dz, layers=args.ae_layers).to(device)
    state = torch.load(args.chunk_ae_weights, map_location=device, weights_only=True)
    chunk_ae.load_state_dict(state)
    chunk_ae.eval()
    for param in chunk_ae.parameters():
        param.requires_grad = False
    print("ChunkAE loaded and frozen.", flush=True)

    P = args.prompt_len
    N = args.gen_len // args.K  # 1920 / 8 = 240
    prior = ChunkMeanFlowPrior(V, P=P, N=N, dz=args.dz, width=args.prior_width,
                               depth=args.prior_depth, heads=args.prior_heads).to(device)
    prior_params = sum(q.numel() for q in prior.parameters())
    print(f"ChunkMeanFlowPrior: P={P} N={N} dz={args.dz} width={args.prior_width} "
          f"depth={args.prior_depth} heads={args.prior_heads} params={prior_params:,}", flush=True)

    opt = torch.optim.AdamW(prior.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 200) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / args.steps))))

    train_loader = tokenizing_distributed_data_loader_bos_bestfit(
        tok, args.rows, 2048, split="train", data_dir=args.data_dir, device=str(device))
    val_loader = tokenizing_distributed_data_loader_bos_bestfit(
        tok, 8, 2048, split="val", data_dir=args.data_dir, device=str(device))

    t0 = time.time()
    seen_tokens = 0
    for step in range(args.steps):
        batch, _ = next(train_loader)
        batch = batch.to(device)
        B = batch.size(0)
        prompt = batch[:, :P]
        gen_tokens = batch[:, P:P + N * args.K]
        
        # Extract target latents through frozen ChunkAE
        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                x_latents = chunk_ae.encode(gen_tokens.reshape(B * N, args.K)).reshape(B, N, args.dz)
        
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            loss, metrics = prior.loss(prompt, x_latents)
        
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(prior.parameters(), 1.0)
        opt.step()
        sched.step()
        seen_tokens += B * 2048

        if step % 200 == 0 or step == args.steps - 1:
            dt = time.time() - t0
            print(f"step {step:5d}/{args.steps} | loss {loss.item():.4f} | flow_mse {metrics['flow_mse']:.4f} "
                  f"| {seen_tokens / 1e6:.1f}M tokens | {dt:.0f}s", flush=True)

    print("Evaluating one-pass generation on validation set...", flush=True)
    eval_res = evaluate_prior(prior, chunk_ae, val_loader, P, N, args.K, V, args.dz, args.val_rows, device, tok)
    print(f"Validation latent MSE: {eval_res['latent_mse']:.4f}")
    print(f"Validation token accuracy (one-pass): {eval_res['token_accuracy']:.5f}")
    print(f"Sample diversity: d1={eval_res['distinct_1']:.3f} d2={eval_res['distinct_2']:.3f} d3={eval_res['distinct_3']:.3f}")
    print("Sample generation (first 100 chars):")
    if eval_res["samples"]:
        print(f"  Prompt:    {eval_res['samples'][0]['prompt_text'][:80]!r}")
        print(f"  Generated: {eval_res['samples'][0]['generated_text'][:80]!r}")
        print(f"  Target:    {eval_res['samples'][0]['target_text'][:80]!r}")

    out_dict = {
        "prior_params": prior_params,
        "steps": args.steps,
        "rows": args.rows,
        "lr": args.lr,
        "K": args.K,
        "dz": args.dz,
        "prompt_len": P,
        "gen_len": N * args.K,
        **eval_res
    }

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({k: v for k, v in out_dict.items() if k != "samples"}, f, indent=2)
        ckpt_path = os.path.splitext(args.out)[0] + ".pt"
        torch.save(prior.state_dict(), ckpt_path)
        print(f"Saved prior weights to {ckpt_path}", flush=True)

    if args.out_samples:
        os.makedirs(os.path.dirname(args.out_samples), exist_ok=True)
        with open(args.out_samples, "w") as f:
            for s in eval_res["samples"]:
                f.write(json.dumps(s) + "\n")
        print(f"Saved generated samples to {args.out_samples}", flush=True)


if __name__ == "__main__":
    main()
