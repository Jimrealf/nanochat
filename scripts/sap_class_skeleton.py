"""
scripts/sap_class_skeleton.py
S13 Branch B: SV-B Class Skeleton Test (s13_sap_brainstorm.md).

Two-stage generation:
1. Stage 1 (Class Skeleton): an autoregressive model over K=256 Brown word classes
   predicts the class skeleton c_P..c_{L-1} from the prompt x_{0..P-1}.
2. Stage 2 (Class-Restricted Lanes): lanes at L=128 decode tokens in parallel,
   each position's softmax restricted to its predicted class c_t.

Gate (pre-registered): NLL_class + NLL_token|class must beat plain L=128 (1.107x dense)
by at least 3 points (<= 1.077x dense).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from nanochat.tokenizer import get_tokenizer


class ClassARModel(nn.Module):
    """Autoregressive class predictor: prompt tokens (V) -> block classes (K=256)."""
    def __init__(self, V: int, K: int = 256, d: int = 256, depth: int = 4, heads: int = 4, max_seq: int = 2048):
        super().__init__()
        self.V, self.K = V, K
        self.prompt_emb = nn.Embedding(V, d)
        self.class_emb = nn.Embedding(K, d)
        self.pos = nn.Parameter(torch.randn(max_seq, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, activation="gelu",
                                          batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, depth)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, K, bias=False)

    def forward(self, prompt_tokens, block_classes):
        # prompt_tokens: (B, P)
        # block_classes: (B, N)
        B, P = prompt_tokens.shape
        _, N = block_classes.shape
        T = P + N
        hp = self.prompt_emb(prompt_tokens)
        hc = self.class_emb(block_classes)
        h = torch.cat([hp, hc], dim=1) + self.pos[:T][None]
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=h.device)
        out = self.transformer(h, mask=mask, is_causal=True)
        logits = self.head(self.norm(out[:, P - 1:-1]))  # predict c_P..c_{L-1}
        return logits


class ClassMaskedLanesModel(nn.Module):
    """Lanes model with output softmax restricted to true Brown class."""
    def __init__(self, V: int, K: int = 256, d: int = 256, depth: int = 4, heads: int = 4,
                 P: int = 128, L: int = 128, class_map: torch.Tensor | None = None):
        super().__init__()
        self.V, self.K = V, K
        self.P, self.L = P, L
        self.tok_emb = nn.Embedding(V, d)
        self.cls_emb = nn.Embedding(K, d)
        self.lane_token = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.pos = nn.Parameter(torch.randn(2048, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, activation="gelu",
                                          batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, depth)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, V, bias=False)
        if class_map is not None:
            self.register_buffer("class_of_token", class_map)

    def forward(self, x, c, lane_mask):
        B, T = x.shape
        # Add class embedding to input
        h = self.tok_emb(x) + self.cls_emb(c) + self.pos[:T][None]
        out = self.transformer(h, mask=lane_mask, is_causal=False)
        logits = self.head(self.norm(out))
        return logits


def evaluate_class_ar(model, val_loader, class_map, P, val_rows, device):
    model.eval()
    total_nll, total_correct, total_classes = 0.0, 0, 0
    with torch.no_grad():
        seen = 0
        while seen < val_rows:
            batch, _ = next(val_loader)
            batch = batch.to(device)
            B = min(batch.size(0), val_rows - seen)
            batch = batch[:B]
            prompt = batch[:, :P]
            all_classes = class_map[batch]
            block_classes_in = all_classes[:, P - 1:-1]
            block_classes_target = all_classes[:, P:]
            
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(prompt, block_classes_in)
            
            loss = F.cross_entropy(logits.reshape(-1, model.K).float(), block_classes_target.reshape(-1), reduction="sum")
            total_nll += float(loss)
            pred = logits.argmax(-1)
            total_correct += int((pred == block_classes_target).sum())
            total_classes += block_classes_target.numel()
            seen += B
    model.train()
    return {
        "nll_nats_per_class": total_nll / total_classes,
        "class_accuracy": total_correct / total_classes
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tokenizer-dir", type=str, default="tokenizer_sap")
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--class-map", type=str, default="out/s13/brown256.npy")
    p.add_argument("--prompt-len", type=int, default=128)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--rows", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val-rows", type=int, default=128)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    if args.smoke:
        args.steps = 20
        args.rows = 2
        args.val_rows = 4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = get_tokenizer(args.tokenizer_dir)
    V = tok.get_vocab_size()

    class_np = np.load(args.class_map).astype(np.int64)
    class_map = torch.from_numpy(class_np).to(device)
    K = int(class_map.max().item()) + 1
    print(f"Loaded class map from {args.class_map}: {len(class_map)} tokens mapped to {K} classes.", flush=True)

    P = args.prompt_len
    model = ClassARModel(V, K=K, d=256, depth=4, heads=4).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 100) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / args.steps))))

    train_loader = tokenizing_distributed_data_loader_bos_bestfit(
        tok, args.rows, 2048, split="train", data_dir=args.data_dir, device=str(device))
    val_loader = tokenizing_distributed_data_loader_bos_bestfit(
        tok, 8, 2048, split="val", data_dir=args.data_dir, device=str(device))

    print(f"ClassARModel params: {sum(q.numel() for q in model.parameters()):,}. Training for {args.steps} steps...", flush=True)
    t0 = time.time()
    for step in range(args.steps):
        batch, _ = next(train_loader)
        batch = batch.to(device)
        B = batch.size(0)
        prompt = batch[:, :P]
        all_classes = class_map[batch]
        classes_in = all_classes[:, P - 1:-1]
        classes_target = all_classes[:, P:]

        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(prompt, classes_in)
            loss = F.cross_entropy(logits.reshape(-1, K), classes_target.reshape(-1))

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % 200 == 0 or step == args.steps - 1:
            print(f"step {step:5d}/{args.steps} | class NLL {loss.item():.4f} nats | {time.time() - t0:.0f}s", flush=True)

    print("Evaluating ClassAR on validation split...", flush=True)
    res = evaluate_class_ar(model, val_loader, class_map, P, args.val_rows, device)
    print(f"Class NLL: {res['nll_nats_per_class']:.4f} nats/class | accuracy: {res['class_accuracy']:.4f}", flush=True)
    
    dense_ref_nats = 3.768 # logged d4 reference
    plain_lanes_nats = 1.107 * dense_ref_nats # 4.171 nats
    print(f"Plain L=128 reference NLL: {plain_lanes_nats:.4f} nats/token", flush=True)

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)
        print(f"Saved results to {args.out}", flush=True)


if __name__ == "__main__":
    main()
