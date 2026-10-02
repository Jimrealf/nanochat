#!/usr/bin/env python3
"""Stage 0 Degeneracy Pre-test for Sampling-Aware Pretraining (SAP).

Reference: sap_research_plan.md §3.1, §3.2, §6.1

Purpose:
  Empirically test the theoretical prediction in §3.2:
  Because h^(1) = g1(h_t) and h^(2) = g2(h_t) are deterministic functions of the
  exact same trunk representation h_t, representation-space block coupling (RBC)
  can be driven toward zero with a completely FROZEN backbone.
  If so, the InfoNCE/predictive loss exerts zero gradient pressure on the backbone
  to resolve branch structure, rendering Variants A and B inert.

Kill Criterion (§7):
  - L_RBC -> ~0 with frozen backbone AND loss fails to differentiate fork from non-fork
    prefixes -> Variants A/B are dead. Stop and advance to Variant C/D.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import pyarrow.parquet as pq

from nanochat.checkpoint_manager import build_model
from nanochat.dataset import list_parquet_files
from nanochat.tokenizer import get_tokenizer


class BilinearCoupling(nn.Module):
    """Variant A: Bilinear InfoNCE coupling head."""

    def __init__(self, dim: int, temperature: float = 0.07):
        super().__init__()
        self.M = nn.Linear(dim, dim, bias=False)
        self.temperature = temperature
        # Standard orthogonal / normal initialization
        nn.init.orthogonal_(self.M.weight)

    def forward(self, h1: torch.Tensor, h2: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """Compute InfoNCE contrastive loss over token positions in batch.
        
        h1, h2: (N, D) normalized hidden states
        """
        N, D = h1.shape
        # Normalize representations on the sphere
        h1_norm = F.normalize(h1, p=2, dim=-1)
        h2_norm = F.normalize(h2, p=2, dim=-1)

        # Bilinear mapping
        proj_h2 = self.M(h2_norm)  # (N, D)
        proj_h2 = F.normalize(proj_h2, p=2, dim=-1)

        # Pairwise similarity matrix: (N, N)
        logits = torch.matmul(h1_norm, proj_h2.t()) / self.temperature

        # Diagonal elements are the positive pairs
        labels = torch.arange(N, device=h1.device)
        loss = F.cross_entropy(logits, labels)

        # Compute top-1 accuracy for positive pair identification
        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            acc = (preds == labels).float().mean().item()

        return loss, acc


class PredictiveCoupling(nn.Module):
    """Variant B: Predictive forward-consistency coupling head."""

    def __init__(self, dim: int):
        super().__init__()
        self.f_phi = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        for m in self.f_phi.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, h1: torch.Tensor, h2: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """Compute cosine distance loss between f_phi(h1) and h2."""
        pred_h2 = self.f_phi(h1)
        pred_norm = F.normalize(pred_h2, p=2, dim=-1)
        h2_norm = F.normalize(h2, p=2, dim=-1)

        # Cosine distance: 1 - cos_sim
        cos_sim = (pred_norm * h2_norm).sum(dim=-1)
        loss = (1.0 - cos_sim).mean()
        mean_cos = cos_sim.mean().item()
        return loss, mean_cos


def get_fork_and_nonfork_prompts() -> Tuple[List[str], List[str]]:
    """Curated fork (high-entropy ambiguous branching) vs non-fork (unambiguous) prefixes."""
    fork_prompts = [
        "The flight was heading to New",
        "She decided to order an iced",
        "The startup was founded in San",
        "The capital of the United",
        "He opened the door and saw a",
        "The weather tomorrow is going to be",
        "The code was written in",
        "The recipe calls for a cup of",
        "He moved his king to",
        "The book was published in the year",
        "In this study, we investigated the effect of",
        "The conference will take place in",
        "The team celebrated their victory with",
        "The primary color chosen for the design was",
        "She picked up the phone and called her",
        "The train leaves the station at",
    ]

    nonfork_prompts = [
        "The Eiffel Tower is located in the city of Paris,",
        "To be or not to be, that is the",
        "The sun rises in the east and sets in the",
        "Water boils at one hundred degrees",
        "Two plus two equals four, and four plus four equals",
        "In the United States of America, the president resides in the White",
        "Oxygen has the atomic number",
        "The capital of France is",
        "Photosynthesis is the process by which green plants use sunlight to synthesize",
        "Mount Everest is the highest mountain peak on planet",
        "A triangle has three sides and a square has four",
        "The speed of light in a vacuum is approximately three hundred thousand kilometers per",
        "Isaac Newton formulated the laws of motion and universal",
        "The Pacific Ocean is the largest ocean on",
        "DNA is composed of four nucleotide bases: adenine, thymine, cytosine, and",
        "The author of Hamlet and Romeo and Juliet was William",
    ]
    return fork_prompts, nonfork_prompts


def extract_trunk_representations(model: nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
    """Run model forward and extract pre-lm_head normalized trunk representation h_t."""
    trunk_container = []

    def hook(module, args):
        trunk_container.append(args[0])

    handle = model.lm_head.register_forward_pre_hook(hook)
    try:
        with torch.no_grad():
            _ = model(input_ids)
        h = trunk_container[0]
    finally:
        handle.remove()

    return h


def main():
    parser = argparse.ArgumentParser(description="Stage 0 Degeneracy Pre-test")
    parser.add_argument("--checkpoint-dir", type=str, default="out/c00_sch_phase0/d4/DENSE_softmax_s1/depth_4/ckpt_base/base")
    parser.add_argument("--step", type=int, default=462)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--tokenizer-dir", type=str, default="tokenizer")
    parser.add_argument("--steps", type=int, default=150, help="Optimization steps for M / f_phi")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output-json", type=str, default="diagnostics/stage0_results.json")
    args = parser.parse_args()

    print("=" * 70)
    print("STAGE 0 DEGENERACY PRE-TEST (Sampling-Aware Pretraining §6.1)")
    print("=" * 70)
    print(f"Device: {args.device}")
    print(f"Loading checkpoint: {args.checkpoint_dir} (step {args.step})")

    # Load frozen base model
    model, tokenizer, meta = build_model(
        checkpoint_dir=args.checkpoint_dir,
        step=args.step,
        device=args.device,
        phase="eval",
        tokenizer_dir=args.tokenizer_dir,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    dim = model.config.n_embd
    print(f"Backbone loaded. Hidden dimension D = {dim}, Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("Backbone successfully FROZEN (all requires_grad=False).")

    # Define standard MTP slot heads (frozen)
    # Slot 1: identity g1(h) = h
    # Slot 2: g2(h) = LayerNorm(Linear(h)) (typical independent MTP future head)
    torch.manual_seed(42)
    g2 = nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, dim),
        nn.LayerNorm(dim),
    ).to(args.device)
    for p in g2.parameters():
        p.requires_grad = False
    g2.eval()

    # Load training data batches from parquet shards
    parquet_files = list_parquet_files(data_dir=args.data_dir)
    assert len(parquet_files) > 0, f"No parquet files found in {args.data_dir}"
    print(f"Using {len(parquet_files)} parquet shards from {args.data_dir}")

    # Read a sample of documents to tokenize
    sample_docs = []
    for pf_path in parquet_files[:5]:
        pf = pq.ParquetFile(pf_path)
        for rg_i in range(min(pf.num_row_groups, 2)):
            rg = pf.read_row_group(rg_i)
            sample_docs.extend(rg.column("text").to_pylist()[:20])
            if len(sample_docs) >= 100:
                break
        if len(sample_docs) >= 100:
            break

    # Tokenize into tensors
    token_batches = []
    for doc in sample_docs:
        ids = tokenizer.encode(doc)
        if len(ids) >= args.seq_len:
            token_batches.append(torch.tensor(ids[:args.seq_len], dtype=torch.long))
        if len(token_batches) >= args.batch_size * 20:
            break

    token_tensor = torch.stack(token_batches).to(args.device)
    print(f"Prepared {token_tensor.shape[0]} sequences of length {args.seq_len}")

    # Initialize Variant A (Bilinear InfoNCE) and Variant B (Predictive MLP)
    variant_a = BilinearCoupling(dim, temperature=0.07).to(args.device)
    variant_b = PredictiveCoupling(dim).to(args.device)

    opt_a = torch.optim.AdamW(variant_a.parameters(), lr=args.lr, weight_decay=1e-4)
    opt_b = torch.optim.AdamW(variant_b.parameters(), lr=args.lr, weight_decay=1e-4)

    history = {
        "step": [],
        "var_a_loss": [],
        "var_a_acc": [],
        "var_b_loss": [],
        "var_b_cossim": [],
    }

    print("\nStarting optimization of coupling heads (Backbone FROZEN)...")
    start_time = time.time()

    N_samples = args.batch_size * 16  # number of token positions per contrastive pool
    num_seqs = token_tensor.shape[0]

    for step in range(args.steps + 1):
        # Sample batch
        idx = torch.randint(0, num_seqs, (args.batch_size,))
        batch_ids = token_tensor[idx]

        with torch.no_grad():
            h_trunk = extract_trunk_representations(model, batch_ids).float()  # (B, T, D)
            h1 = h_trunk.reshape(-1, dim)
            h2 = g2(h_trunk).reshape(-1, dim)

            # Subsample N_samples to keep contrastive matrix tractable
            if h1.shape[0] > N_samples:
                sub_idx = torch.randperm(h1.shape[0])[:N_samples]
                h1_sub = h1[sub_idx]
                h2_sub = h2[sub_idx]
            else:
                h1_sub = h1
                h2_sub = h2

        # Step Variant A
        loss_a, acc_a = variant_a(h1_sub, h2_sub)
        opt_a.zero_grad()
        loss_a.backward()
        opt_a.step()

        # Step Variant B
        loss_b, cossim_b = variant_b(h1_sub, h2_sub)
        opt_b.zero_grad()
        loss_b.backward()
        opt_b.step()

        if step % 25 == 0 or step == args.steps:
            history["step"].append(step)
            history["var_a_loss"].append(float(loss_a.item()))
            history["var_a_acc"].append(float(acc_a))
            history["var_b_loss"].append(float(loss_b.item()))
            history["var_b_cossim"].append(float(cossim_b))

            print(f"Step {step:4d}/{args.steps:4d} | "
                  f"Variant A (InfoNCE): loss = {loss_a.item():.4f}, top-1 acc = {acc_a * 100:.1f}% | "
                  f"Variant B (Predictive): loss = {loss_b.item():.4f}, cos_sim = {cossim_b:.4f}")

    elapsed = time.time() - start_time
    print(f"\nOptimization completed in {elapsed:.1f}s.")

    # -------------------------------------------------------------------------
    # Fork vs Non-Fork Prefix Differential Test
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("EVALUATION: Fork vs Non-Fork Prefix Differential Check (§6.1)")
    print("=" * 70)

    fork_prompts, nonfork_prompts = get_fork_and_nonfork_prompts()

    def evaluate_prompts(prompts: List[str]) -> Dict[str, float]:
        a_losses, a_accs, b_losses, b_sims = [], [], [], []
        variant_a.eval()
        variant_b.eval()
        for p in prompts:
            ids = tokenizer.encode(p)
            inp = torch.tensor([ids], dtype=torch.long, device=args.device)
            with torch.no_grad():
                h_trunk = extract_trunk_representations(model, inp).float()
                h1 = h_trunk[0]  # (T, D)
                h2 = g2(h_trunk)[0]
                la, acc = variant_a(h1, h2)
                lb, sim = variant_b(h1, h2)
                a_losses.append(la.item())
                a_accs.append(acc)
                b_losses.append(lb.item())
                b_sims.append(sim)
        return {
            "a_loss_mean": float(sum(a_losses) / len(a_losses)),
            "a_acc_mean": float(sum(a_accs) / len(a_accs)),
            "b_loss_mean": float(sum(b_losses) / len(b_losses)),
            "b_cossim_mean": float(sum(b_sims) / len(b_sims)),
        }

    fork_stats = evaluate_prompts(fork_prompts)
    nonfork_stats = evaluate_prompts(nonfork_prompts)

    print(f"Fork Prompts (N={len(fork_prompts)}):")
    print(f"  Variant A (InfoNCE):    loss = {fork_stats['a_loss_mean']:.4f}, acc = {fork_stats['a_acc_mean']*100:.1f}%")
    print(f"  Variant B (Predictive): loss = {fork_stats['b_loss_mean']:.4f}, cos_sim = {fork_stats['b_cossim_mean']:.4f}")

    print(f"\nNon-Fork Prompts (N={len(nonfork_prompts)}):")
    print(f"  Variant A (InfoNCE):    loss = {nonfork_stats['a_loss_mean']:.4f}, acc = {nonfork_stats['a_acc_mean']*100:.1f}%")
    print(f"  Variant B (Predictive): loss = {nonfork_stats['b_loss_mean']:.4f}, cos_sim = {nonfork_stats['b_cossim_mean']:.4f}")

    diff_a = abs(fork_stats['a_loss_mean'] - nonfork_stats['a_loss_mean'])
    diff_b = abs(fork_stats['b_cossim_mean'] - nonfork_stats['b_cossim_mean'])
    print(f"\nDifferential |Fork - NonFork|:")
    print(f"  Variant A loss gap:    {diff_a:.4f}")
    print(f"  Variant B cos_sim gap: {diff_b:.4f}")

    # -------------------------------------------------------------------------
    # Gate Decision Evaluation (§6.1, §7)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STAGE 0 GATE DECISION")
    print("=" * 70)

    final_a_loss = history["var_a_loss"][-1]
    final_b_loss = history["var_b_loss"][-1]
    init_a_loss = history["var_a_loss"][0]
    init_b_loss = history["var_b_loss"][0]

    print(f"Variant A (InfoNCE):    init loss = {init_a_loss:.4f} -> final loss = {final_a_loss:.4f} (acc: {history['var_a_acc'][-1]*100:.1f}%)")
    print(f"Variant B (Predictive): init loss = {init_b_loss:.4f} -> final loss = {final_b_loss:.4f} (cos_sim: {history['var_b_cossim'][-1]:.4f})")

    # Degeneracy threshold: loss drops substantially (> 80% reduction or acc > 90%) without any backbone change
    deg_a = (final_a_loss < 0.5) or (history["var_a_acc"][-1] > 0.90) or (final_a_loss < 0.25 * init_a_loss)
    deg_b = (history["var_b_cossim"][-1] > 0.90) or (final_b_loss < 0.1)

    stage0_fired = deg_a and deg_b and (diff_a < 0.3)

    if stage0_fired:
        verdict = "STAGE 0 GATE FIRED: DEGENERACY CONFIRMED"
        action = "Variants A and B are inert (M/f_phi minimizes loss purely from frozen deterministic trunk; no branch signal sent to backbone). Skip A/B; Advance to Variant C (Low-Rank Logit Joint Energy) and Variant D (Token Channel)."
    else:
        verdict = "STAGE 0 PASSED (RESIDUAL SIGNAL DETECTED)"
        action = "RBC loss maintained plateau or strong fork differential. Retain Variants A/B."

    print(f"Verdict: {verdict}")
    print(f"Action:  {action}")
    print("=" * 70)

    # Save results
    results = {
        "verdict": verdict,
        "action": action,
        "history": history,
        "fork_stats": fork_stats,
        "nonfork_stats": nonfork_stats,
        "differential": {"diff_a": diff_a, "diff_b": diff_b},
        "checkpoint": args.checkpoint_dir,
        "step": args.step,
    }

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {args.output_json}")


if __name__ == "__main__":
    main()
