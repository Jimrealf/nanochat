"""
S13 Q2: can a chunk of K tokens be squeezed into one robust continuous latent? (SV-D's necessary
condition, s13_sap_brainstorm.md.)

SV-D is a true one-pass generator: a frozen chunk autoencoder plus a one-step (MeanFlow) prior
over the block's L/K latents, both trained from scratch with targets computed from the data. It
is only possible if the autoencoder reconstructs a chunk almost perfectly from a latent that
tolerates noise, because the prior's one-step samples will be noisy. CALM (Shao et al. 2025)
reports above 99.9% at K=4; this script asks the same at K=8.

Autoencoder, context-free per chunk as in CALM:
  encoder  token embedding + position, a small bidirectional transformer over the K tokens,
           mean pool, linear to a dz-dim latent, RMS-normalised
  noise    z + sigma * N(0, I) during training (robustness)
  decoder  the latent broadcast to K positions plus position embeddings, a small bidirectional
           transformer, one softmax per position (all K tokens in parallel)

Reported on validation chunks: token accuracy and NLL at sigma = 0 and at the training sigma, and
accuracy by position within the chunk. Kill line (pre-registered): token accuracy below 99.5% at
the training sigma closes SV-D.

    python -m scripts.sap_chunk_ae --tokenizer-dir tokenizer_sap --data-dir data --K 8 --steps 4000
    python -m scripts.sap_chunk_ae --smoke
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


class ChunkAE(nn.Module):
    def __init__(self, V, K, d=512, dz=128, layers=2, heads=8):
        super().__init__()
        self.K = K
        self.dz = dz
        self.emb = nn.Embedding(V, d)
        self.pos_e = nn.Parameter(torch.randn(K, d) * 0.02)
        self.pos_d = nn.Parameter(torch.randn(K, d) * 0.02)
        mk = lambda: nn.TransformerEncoder(nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, activation="gelu",
                                                                      batch_first=True, norm_first=True), layers)
        self.enc, self.dec = mk(), mk()
        self.to_z = nn.Linear(d, dz)
        self.from_z = nn.Linear(dz, d)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, V, bias=False)

    def encode(self, t):                                     # (n, K) -> (n, dz), unit RMS
        h = self.enc(self.emb(t) + self.pos_e)
        z = self.to_z(h.mean(1))
        return z * torch.rsqrt(z.pow(2).mean(-1, keepdim=True) + 1e-6)

    def decode(self, z):                                     # (n, dz) -> (n, K, V) logits
        h = self.dec(self.from_z(z)[:, None, :] + self.pos_d)
        return self.head(self.norm(h))

    def forward(self, t, sigma):
        z = self.encode(t)
        if sigma > 0:
            z = z + sigma * torch.randn_like(z)
        return self.decode(z)


def chunks(loader, K, device):
    x, _ = next(loader)
    x = x.to(device)
    B, N = x.shape
    return x[:, :N - N % K].reshape(-1, K)


@torch.no_grad()
def evaluate(model, val_chunks, sigma, batch=1024):
    model.eval()
    nll, correct, n = 0.0, 0, 0
    pos_correct = torch.zeros(model.K, dtype=torch.float64)
    for i in range(0, val_chunks.size(0), batch):
        t = val_chunks[i:i + batch]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=t.is_cuda):
            lg = model(t, sigma).float()
        nll += float(F.cross_entropy(lg.reshape(-1, lg.size(-1)), t.reshape(-1), reduction="sum"))
        hit = lg.argmax(-1) == t
        correct += int(hit.sum())
        pos_correct += hit.double().sum(0).cpu()
        n += t.numel()
    model.train()
    return {"token_accuracy": correct / n, "nll_per_token": nll / n,
            "accuracy_by_position": (pos_correct / (n / model.K)).tolist()}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tokenizer-dir", type=str, default="tokenizer")
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--K", type=int, default=8)
    p.add_argument("--d", type=int, default=512)
    p.add_argument("--dz", type=int, default=128)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--sigma", type=float, default=0.5, help="latent noise during training (latent has unit RMS)")
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--rows", type=int, default=32, help="rows of 2048 tokens per step")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val-rows", type=int, default=256)
    p.add_argument("--eval-batch", type=int, default=1024, help="validation chunks per forward")
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.smoke:
        args.steps, args.rows, args.val_rows, args.d, args.layers, args.eval_batch = 20, 2, 4, 128, 1, 128
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = get_tokenizer(args.tokenizer_dir)
    V = tok.get_vocab_size()
    torch.manual_seed(0)
    model = ChunkAE(V, args.K, args.d, args.dz, args.layers).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 200) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / args.steps))))
    train = tokenizing_distributed_data_loader_bos_bestfit(tok, args.rows, 2048, split="train", data_dir=args.data_dir,
                                                           device=str(device))
    val = tokenizing_distributed_data_loader_bos_bestfit(tok, 8, 2048, split="val", data_dir=args.data_dir,
                                                         device=str(device))
    val_chunks = torch.cat([chunks(val, args.K, device) for _ in range(max(1, args.val_rows // 8))])
    print(f"chunk AE: K={args.K} d={args.d} dz={args.dz} layers={args.layers} sigma={args.sigma} "
          f"params={sum(q.numel() for q in model.parameters()):,}; {val_chunks.size(0):,} validation chunks", flush=True)
    t0, seen = time.time(), 0
    for step in range(args.steps):
        t = chunks(train, args.K, device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            lg = model(t, args.sigma)
        loss = F.cross_entropy(lg.float().reshape(-1, V), t.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        seen += t.numel()
        if step % 500 == 0 or step == args.steps - 1:
            print(f"step {step:5d} | loss {loss.item():.4f} | {seen / 1e6:.0f}M tokens | {time.time() - t0:.0f}s", flush=True)
    res = {"K": args.K, "d": args.d, "dz": args.dz, "layers": args.layers, "sigma_train": args.sigma,
           "train_tokens": seen, "clean": evaluate(model, val_chunks, 0.0, args.eval_batch),
           "noisy": evaluate(model, val_chunks, args.sigma, args.eval_batch)}
    for k in ("clean", "noisy"):
        r = res[k]
        print(f"{k:5s} (sigma {0.0 if k == 'clean' else args.sigma}): token accuracy {r['token_accuracy']:.5f}, "
              f"NLL {r['nll_per_token']:.4f} nats/token, by position " +
              " ".join(f"{a:.4f}" for a in r["accuracy_by_position"]), flush=True)
    verdict = "PASS" if res["noisy"]["token_accuracy"] >= 0.995 else "KILL (below 99.5% at the training noise)"
    res["verdict"] = verdict
    print(f"SV-D necessary condition: {verdict}", flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)
        ckpt_path = os.path.splitext(args.out)[0] + ".pt"
        torch.save(model.state_dict(), ckpt_path)
        print(f"saved weights to {ckpt_path}", flush=True)


if __name__ == "__main__":
    main()
