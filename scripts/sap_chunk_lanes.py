"""
scripts/sap_chunk_lanes.py
Proposal A: Chunk-Lanes (Lanes over Chunk Latents).

Architecture:
1. Frozen ChunkAE: compresses K=8 tokens into dz=256 continuous latent.
2. Layout: N=240 chunks (1920 tokens) arranged into L=16 lanes of S=15 steps each.
3. At step s in {0..14}, all 16 lanes emit their s-th chunk in parallel.
   The chunks sit S*K = 15*8 = 120 tokens apart, where total correlation is negligible.
4. Conditioning: Step s attends to the full prompt (P=128) and all prior steps 0..s-1 of ALL lanes.
5. Head: 1-step MeanFlow velocity field conditioned on the lane hidden state,
   mapping eps ~ N(0, I) to chunk latent z_{j, s}.
6. Full block generation (1920 tokens) takes EXACTLY 15 sequential steps (~15ms on H100, 220x speedup).
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


class JVPFlowHead(nn.Module):
    """1-step MeanFlow velocity head conditioned on lane hidden state."""
    def __init__(self, dz: int, width: int):
        super().__init__()
        self.time_emb = nn.Sequential(
            nn.Linear(3, width),
            nn.SiLU(),
            nn.Linear(width, width),
        )
        self.net = nn.Sequential(
            nn.Linear(dz + width, width),
            nn.GELU(),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, dz),
        )

    def forward(self, z, r, t, h_cond):
        # z: (B, M, dz)
        # r: (B,)
        # t: (B,)
        # h_cond: (B, M, width)
        B, M, dz = z.shape
        rt = torch.stack((r, t, t - r), -1)  # (B, 3)
        t_vec = self.time_emb(rt)[:, None, :]  # (B, 1, width)
        
        inp = torch.cat([z, h_cond + t_vec], dim=-1)
        return self.net(inp)


class ChunkLanesModel(nn.Module):
    def __init__(self, V_prompt: int, P: int = 128, L: int = 16, S: int = 15, K: int = 8, dz: int = 256,
                 width: int = 512, depth: int = 8, heads: int = 8):
        super().__init__()
        self.P = P  # prompt tokens (128)
        self.L = L  # parallel lanes (16)
        self.S = S  # steps per lane (15)
        self.N = L * S  # total chunks (240)
        self.K = K  # tokens per chunk (8)
        self.dz = dz  # latent dimension (256)
        self.width = width
        self.heads = heads
        self.head_dim = width // heads

        self.prompt_emb = nn.Embedding(V_prompt, width)
        self.prompt_pos = nn.Parameter(torch.randn(P, width) * 0.02)

        # Step 0 lane starts
        self.lane_start_emb = nn.Parameter(torch.randn(L, width) * 0.02)
        self.latent_in = nn.Linear(dz, width)

        # Positional embeddings: lane index (L) + step index (S)
        self.lane_pos = nn.Parameter(torch.randn(L, width) * 0.02)
        self.step_pos = nn.Parameter(torch.randn(S, width) * 0.02)

        # Transformer blocks
        from scripts.sap_meanflow_chunk import JVPTransformerBlock
        self.blocks = nn.ModuleList([JVPTransformerBlock(width, heads) for _ in range(depth)])
        self.ln_f = nn.LayerNorm(width)

        # Flow head
        self.flow_head = JVPFlowHead(dz, width)

    def _build_step_major_sequence(self, prompt_tokens, target_latents):
        # prompt_tokens: (B, P)
        # target_latents: (B, N, dz) where N = L * S
        B = prompt_tokens.size(0)
        device = prompt_tokens.device

        # Reshape target latents into (B, L, S, dz)
        Z_grid = target_latents.reshape(B, self.L, self.S, self.dz)

        # Sequence construction in step-major order:
        # At step 0: inputs are lane_start_emb (L positions)
        # At step s in 1..S-1: inputs are Z_grid[:, :, s-1, :]
        h_prompt = self.prompt_emb(prompt_tokens) + self.prompt_pos[None, :, :]  # (B, P, width)

        step_inputs = []
        for s in range(self.S):
            if s == 0:
                inp_s = self.lane_start_emb[None, :, :].expand(B, self.L, self.width)
            else:
                prev_z = Z_grid[:, :, s - 1, :]  # (B, L, dz)
                inp_s = self.latent_in(prev_z)
            # Add lane and step position
            inp_s = inp_s + self.lane_pos[None, :, :] + self.step_pos[None, s:s + 1, :]
            step_inputs.append(inp_s)

        h_lanes = torch.cat(step_inputs, dim=1)  # (B, L * S, width)
        h_full = torch.cat([h_prompt, h_lanes], dim=1)  # (B, P + L * S, width)
        return h_full

    def forward_trunk(self, h_full):
        # Attention mask: block-causal by step
        # Prompt (0..P-1) can see prompt.
        # Step s (positions P + s*L .. P + (s+1)*L - 1) can see prompt + all earlier steps 0..s-1
        # Chunks within the same step cannot see each other.
        T = h_full.size(1)
        device = h_full.device
        mask = torch.full((T, T), float("-inf"), device=device)

        # Prompt attends causally to prompt
        mask[:self.P, :self.P] = torch.triu(torch.full((self.P, self.P), float("-inf"), device=device), diagonal=1)

        # Step-major lanes
        for s in range(self.S):
            start = self.P + s * self.L
            end = start + self.L
            # Sees full prompt
            mask[start:end, :self.P] = 0.0
            # Sees all previous steps
            if s > 0:
                mask[start:end, self.P:start] = 0.0
            # Diagonal within same step: each lane sees only its own input at step s
            for j in range(self.L):
                mask[start + j, start + j] = 0.0

        h = h_full
        for block in self.blocks:
            # Explicit attention with mask
            B, seq_len, D = h.shape
            q, k, v = block.qkv(block.ln1(h)).chunk(3, -1)
            def heads_fn(a):
                return a.view(B, seq_len, block.heads, block.head_dim).transpose(1, 2)
            q, k, v = heads_fn(q), heads_fn(k), heads_fn(v)
            scores = (q @ k.transpose(-1, -2)) / math.sqrt(block.head_dim) + mask[None, None, :, :]
            att = scores.softmax(-1)
            mixed = (att @ v).transpose(1, 2).reshape(B, seq_len, D)
            h = h + block.proj(mixed)
            h = h + block.ff(block.ln2(h))

        h = self.ln_f(h)
        # Extract lane hidden states: (B, S, L, width) -> reshape to (B, L * S, width) in natural order
        h_lane_steps = h[:, self.P:, :].reshape(B, self.S, self.L, self.width)
        h_natural = h_lane_steps.permute(0, 2, 1, 3).reshape(B, self.N, self.width)
        return h_natural

    def loss(self, prompt_tokens, target_latents):
        # target_latents: (B, N, dz) in natural chunk order (0..239)
        B, N, dz = target_latents.shape
        device = target_latents.device

        h_full = self._build_step_major_sequence(prompt_tokens, target_latents)
        h_cond = self.forward_trunk(h_full)  # (B, N, width) in natural order

        eps = torch.randn_like(target_latents)
        a = torch.randn(B, device=device).sub_(0.4).sigmoid()
        b = torch.randn(B, device=device).sub_(0.4).sigmoid()
        r, t = torch.minimum(a, b), torch.maximum(a, b)
        same = torch.rand_like(r) >= 0.25
        r = torch.where(same, t, r)

        t_exp = t[:, None, None]
        z_t = (1 - t_exp) * target_latents + t_exp * eps
        v = eps - target_latents

        fn = lambda zz, rr, tt: self.flow_head(zz, rr, tt, h_cond)
        u, dudt = torch.func.jvp(fn, (z_t, r, t), (v, torch.zeros_like(r), torch.ones_like(t)))

        dt_exp = (t - r)[:, None, None]
        target = (v - dt_exp * dudt).detach()
        err = (u - target).square().mean((1, 2))
        weight = (err.detach() + 1e-3).rsqrt()
        loss = (weight * err).mean()
        return loss, {"flow_mse": float(err.mean().detach())}

    @torch.no_grad()
    def generate_15_steps(self, prompt_tokens):
        # 15 sequential steps to emit all 240 chunks!
        B = prompt_tokens.size(0)
        device = prompt_tokens.device

        generated_grid = torch.zeros(B, self.L, self.S, self.dz, device=device)
        h_prompt = self.prompt_emb(prompt_tokens) + self.prompt_pos[None, :, :]

        # We maintain running step inputs
        current_seq = h_prompt  # starts with prompt
        for s in range(self.S):
            if s == 0:
                inp_s = self.lane_start_emb[None, :, :].expand(B, self.L, self.width)
            else:
                prev_z = generated_grid[:, :, s - 1, :]
                inp_s = self.latent_in(prev_z)
            inp_s = inp_s + self.lane_pos[None, :, :] + self.step_pos[None, s:s + 1, :]

            # Run trunk with full history
            full_seq = torch.cat([current_seq, inp_s], dim=1)
            # Run forward trunk on current prefix
            h_all = self.forward_trunk_prefix(full_seq, s)
            h_curr_step = h_all[:, -self.L:, :]  # (B, L, width)

            # Emit 16 chunks in parallel via 1-step flow head
            eps = torch.randn(B, self.L, self.dz, device=device)
            r = torch.zeros(B, device=device)
            t = torch.ones(B, device=device)
            u = self.flow_head(eps, r, t, h_curr_step)
            z_s = eps - u
            z_s = z_s * torch.rsqrt(z_s.pow(2).mean(-1, keepdim=True) + 1e-6)

            generated_grid[:, :, s, :] = z_s
            current_seq = full_seq

        # Return natural chunk order: (B, L, S, dz) -> (B, L*S, dz)
        z_out = generated_grid.permute(0, 1, 2, 3).reshape(B, self.N, self.dz)
        return z_out

    def forward_trunk_prefix(self, h_full, current_s: int):
        # Forward pass for prefix up to step current_s
        T = h_full.size(1)
        device = h_full.device
        mask = torch.full((T, T), float("-inf"), device=device)

        # Prompt
        mask[:self.P, :self.P] = torch.triu(torch.full((self.P, self.P), float("-inf"), device=device), diagonal=1)

        # Steps up to current_s
        for s in range(current_s + 1):
            start = self.P + s * self.L
            end = start + self.L
            mask[start:end, :self.P] = 0.0
            if s > 0:
                mask[start:end, self.P:start] = 0.0
            for j in range(self.L):
                mask[start + j, start + j] = 0.0

        h = h_full
        for block in self.blocks:
            B, seq_len, D = h.shape
            q, k, v = block.qkv(block.ln1(h)).chunk(3, -1)
            def heads_fn(a):
                return a.view(B, seq_len, block.heads, block.head_dim).transpose(1, 2)
            q, k, v = heads_fn(q), heads_fn(k), heads_fn(v)
            scores = (q @ k.transpose(-1, -2)) / math.sqrt(block.head_dim) + mask[None, None, :, :]
            att = scores.softmax(-1)
            mixed = (att @ v).transpose(1, 2).reshape(B, seq_len, D)
            h = h + block.proj(mixed)
            h = h + block.ff(block.ln2(h))

        return self.ln_f(h)


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

            # 15-step generation
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                z_pred = model.generate_15_steps(prompt)
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
    p.add_argument("--tokenizer-dir", type=str, default="/vol/tokenizer_sap")
    p.add_argument("--data-dir", type=str, default="/vol/data")
    p.add_argument("--P", type=int, default=128)
    p.add_argument("--L", type=int, default=16)
    p.add_argument("--S", type=int, default=15)
    p.add_argument("--K", type=int, default=8)
    p.add_argument("--dz", type=int, default=256)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--depth", type=int, default=8)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--rows", type=int, default=32)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--val-rows", type=int, default=128)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--out-weights", type=str, default=None)
    p.add_argument("--out-samples", type=str, default=None)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    if args.smoke:
        args.steps = 20
        args.val_rows = 16

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tok = get_tokenizer(args.tokenizer_dir)
    V = tok.get_vocab_size()
    P, L, S, K, dz = args.P, args.L, args.S, args.K, args.dz
    N = L * S

    print(f"Loading ChunkAE from {args.chunk_ae_weights}...")
    chunk_ae = ChunkAE(K=K, dz=dz, V=V).to(device)
    ae_ckpt = torch.load(args.chunk_ae_weights, map_location=device, weights_only=True)
    chunk_ae.load_state_dict(ae_ckpt["state_dict"] if "state_dict" in ae_ckpt else ae_ckpt)
    chunk_ae.eval()

    print(f"Initializing ChunkLanesModel: L={L} lanes, S={S} steps, width={args.width}, depth={args.depth}...")
    model = ChunkLanesModel(V, P, L, S, K, dz, width=args.width, depth=args.depth, heads=args.heads).to(device)
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
            print(f"step {step:5d} | loss {loss.item():.4f} | flow_mse {metrics['flow_mse']:.4f} | {toks_m:.1f}M toks | {elapsed:.1f}s")

    print("\nRunning final validation and 15-step generation...")
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
