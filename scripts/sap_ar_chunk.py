"""
scripts/sap_ar_chunk.py
S13 Option 1: Autoregressive Continuous Latent Chunk Transformer (CALM-style).

Instead of generating all 240 chunk latents in 1 flat step from noise (which failed due to
excessive variance and total correlation tax), this model generates chunk latents autoregressively:
    z_m ~ p(z_m | z_{<m}, prompt)

Key advantages over next-token AR:
- 8x reduction in sequential decode steps (240 steps vs 1920 steps).
- 8x smaller sequence length in attention (368 tokens vs 2048 tokens).
- 64x reduction in attention FLOPs and 8x smaller KV-cache.
- Clean causal teacher forcing during pretraining via MSE loss.
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
from scripts.sap_meanflow_chunk import compute_ngram_diversity


class CausalTransformerBlock(nn.Module):
    def __init__(self, width: int, heads: int):
        super().__init__()
        assert width % heads == 0
        self.heads, self.head_dim = heads, width // heads
        self.ln1, self.ln2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.proj = nn.Linear(width, width)
        self.ff = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width))

    def forward(self, x, mask=None):
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).chunk(3, -1)
        def heads(a):
            return a.view(B, T, self.heads, self.head_dim).transpose(1, 2)
        q, k, v = heads(q), heads(k), heads(v)
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask
        att = scores.softmax(-1)
        mixed = (att @ v).transpose(1, 2).reshape(B, T, D)
        x = x + self.proj(mixed)
        return x + self.ff(self.ln2(x))


class ChunkARPrior(nn.Module):
    def __init__(self, V_prompt: int, P: int, N: int, dz: int, width: int = 512, depth: int = 8, heads: int = 8):
        super().__init__()
        self.P = P  # prompt tokens (128)
        self.N = N  # chunk latents (240)
        self.dz = dz  # chunk latent dim (256)
        self.width = width
        
        self.prompt_emb = nn.Embedding(V_prompt, width)
        self.latent_in = nn.Linear(dz, width)
        self.pos_emb = nn.Parameter(torch.randn(P + N + 1, width) * 0.02)
        
        self.blocks = nn.ModuleList([CausalTransformerBlock(width, heads) for _ in range(depth)])
        self.ln_f = nn.LayerNorm(width)
        self.out_proj = nn.Linear(width, dz)

    def forward(self, prompt_tokens, latents_prefix):
        # prompt_tokens: (B, P)
        # latents_prefix: (B, M, dz) where M <= N - 1
        B = prompt_tokens.size(0)
        M = latents_prefix.size(1)
        device = prompt_tokens.device
        
        h_prompt = self.prompt_emb(prompt_tokens)
        h_latents = self.latent_in(latents_prefix) if M > 0 else torch.empty(B, 0, self.width, device=device)
        h = torch.cat([h_prompt, h_latents], dim=1)  # (B, P + M, width)
        
        T = h.size(1)
        h = h + self.pos_emb[:T][None, :, :]
        
        # Causal mask
        mask = torch.full((T, T), float("-inf"), device=device)
        mask = torch.triu(mask, diagonal=1)[None, None, :, :]
        
        for block in self.blocks:
            h = block(h, mask=mask)
        h = self.ln_f(h)
        
        # Hidden states predicting next chunk latents are at indices [P - 1, P + M - 1]
        # At index P - 1 (end of prompt): predicts z_1
        # At index P (after z_1): predicts z_2
        # ...
        pred_h = h[:, self.P - 1:]
        out = self.out_proj(pred_h)
        return out  # (B, M + 1, dz)

    def loss(self, prompt_tokens, target_latents):
        # target_latents: (B, N, dz)
        B, N, dz = target_latents.shape
        # Input latents prefix is z_1 ... z_{N-1}
        latents_prefix = target_latents[:, :-1, :]
        preds = self.forward(prompt_tokens, latents_prefix)  # (B, N, dz)
        
        mse = F.mse_loss(preds, target_latents)
        return mse, {"latent_mse": float(mse.detach())}

    @torch.no_grad()
    def generate(self, prompt_tokens, max_chunks: int = None):
        if max_chunks is None:
            max_chunks = self.N
        B = prompt_tokens.size(0)
        device = prompt_tokens.device
        
        # Initialize empty prefix
        gen_latents = torch.empty(B, 0, self.dz, device=device)
        for m in range(max_chunks):
            preds = self.forward(prompt_tokens, gen_latents)
            next_z = preds[:, -1:, :]
            # Unit RMS normalization
            next_z = next_z * torch.rsqrt(next_z.pow(2).mean(-1, keepdim=True) + 1e-6)
            gen_latents = torch.cat([gen_latents, next_z], dim=1)
        return gen_latents


def evaluate(model, chunk_ae, val_loader, P, N, K, dz, val_rows, device, tok):
    model.eval()
    chunk_ae.eval()
    
    total_eval = 0
    latent_mse_sum = 0.0
    token_correct_sum = 0
    total_tokens = 0
    samples = []
    
    with torch.no_grad():
        while total_eval < val_rows:
            batch, _ = next(val_loader)
            batch = batch.to(device)
            B = min(batch.size(0), val_rows - total_eval)
            batch = batch[:B]
            
            prompt = batch[:, :P]
            target_tokens = batch[:, P:P + N * K]
            
            target_chunks = target_tokens.reshape(B * N, K)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                z_true = chunk_ae.encode(target_chunks).reshape(B, N, dz)
            
            # Autoregressive generation
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                z_pred = model.generate(prompt, max_chunks=N)
                logits = chunk_ae.decode(z_pred.reshape(B * N, dz))
            
            pred_tokens = logits.argmax(-1).reshape(B, N * K)
            
            mse = (z_pred.float() - z_true.float()).pow(2).mean().item()
            latent_mse_sum += mse * B
            
            hit = (pred_tokens == target_tokens)
            token_correct_sum += int(hit.sum().item())
            total_tokens += B * N * K
            total_eval += B
            
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
                    
    all_gen = [s["generated_tokens"] for s in samples]
    d1 = compute_ngram_diversity(all_gen, 1)
    d2 = compute_ngram_diversity(all_gen, 2)
    d3 = compute_ngram_diversity(all_gen, 3)
    
    model.train()
    return {
        "val_rows": total_eval,
        "latent_mse": latent_mse_sum / total_eval,
        "token_accuracy": token_correct_sum / total_tokens,
        "distinct_1": d1,
        "distinct_2": d2,
        "distinct_3": d3,
        "samples": samples,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--chunk-ae-weights", type=str, required=True)
    p.add_argument("--tokenizer-dir", type=str, default="tokenizer_sap")
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--K", type=int, default=8)
    p.add_argument("--dz", type=int, default=256)
    p.add_argument("--prompt-len", type=int, default=128)
    p.add_argument("--gen-len", type=int, default=1920)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--depth", type=int, default=8)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--rows", type=int, default=32)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--val-rows", type=int, default=128)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--out-samples", type=str, default=None)
    p.add_argument("--out-weights", type=str, default=None)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    if args.smoke:
        args.steps = 20
        args.val_rows = 16

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

    print(f"Initializing ChunkARPrior: width={args.width}, depth={args.depth}, heads={args.heads}...")
    model = ChunkARPrior(V_prompt=V, P=P, N=N, dz=dz, width=args.width, depth=args.depth, heads=args.heads).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    train_loader = tokenizing_distributed_data_loader_bos_bestfit(
        tok, args.rows, 2048, split="train", data_dir=args.data_dir, device=str(device)
    )
    val_loader = tokenizing_distributed_data_loader_bos_bestfit(
        tok, min(args.rows, args.val_rows), 2048, split="val", data_dir=args.data_dir, device=str(device)
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
    
    t0 = time.time()
    for step in range(1, args.steps + 1):
        batch, _ = next(train_loader)
        batch = batch.to(device)
        prompt = batch[:, :P]
        target_tokens = batch[:, P:P + N * K]
        
        with torch.no_grad():
            target_chunks = target_tokens.reshape(-1, K)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                z_true = chunk_ae.encode(target_chunks).reshape(-1, N, dz)
                
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            loss, metrics = model.loss(prompt, z_true)
            
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if step % 100 == 0 or step == args.steps or step <= 10:
            elapsed = time.time() - t0
            toks_m = (step * args.rows * (P + N * K)) / 1e6
            print(f"step {step:5d} | loss {loss.item():.4f} | latent_mse {metrics['latent_mse']:.4f} | {toks_m:.1f}M toks | {elapsed:.1f}s")

    print("\nRunning final validation and generation...")
    val_results = evaluate(model, chunk_ae, val_loader, P, N, K, dz, args.val_rows, device, tok)
    print(f"Validation Latent MSE: {val_results['latent_mse']:.4f}")
    print(f"Validation Token Acc:  {val_results['token_accuracy'] * 100:.3f}%")
    print(f"Distinct-1/2/3:        {val_results['distinct_1']:.3f} / {val_results['distinct_2']:.3f} / {val_results['distinct_3']:.3f}")

    if val_results["samples"]:
        s0 = val_results["samples"][0]
        print(f"\n--- Sample 0 ---")
        print(f"Prompt:    {s0['prompt_text'][:100]}")
        print(f"Generated: {s0['generated_text'][:200]}")

    if args.out_weights:
        os.makedirs(os.path.dirname(args.out_weights), exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "args": vars(args)}, args.out_weights)
        print(f"Saved weights to {args.out_weights}")

    if args.out_samples and val_results["samples"]:
        os.makedirs(os.path.dirname(args.out_samples), exist_ok=True)
        with open(args.out_samples, "w") as f:
            for s in val_results["samples"]:
                f.write(json.dumps(s) + "\n")
        print(f"Saved {len(val_results['samples'])} samples to {args.out_samples}")

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({
                "steps": args.steps,
                "train_loss": loss.item(),
                "val_latent_mse": val_results["latent_mse"],
                "val_token_acc": val_results["token_accuracy"],
                "distinct_1": val_results["distinct_1"],
                "distinct_2": val_results["distinct_2"],
                "distinct_3": val_results["distinct_3"],
            }, f, indent=2)
        print(f"Saved summary to {args.out}")


if __name__ == "__main__":
    main()
