#!/usr/bin/env python3
"""Direct Joint Divergence Metric (TV and KL) for Multi-Token LLMs.

Reference: sap_research_plan.md §6.5 (Metric 1)

Measures:
  Total Variation (TV) and Kullback-Leibler (KL) divergence between:
  1. True joint distribution P*(x_{t+1}, x_{t+2} | x_{<=t}) obtained via 2-step
     autoregressive rollout over top-k x top-k continuations.
  2. The candidate model's block distribution Q(x_{t+1}, x_{t+2} | x_{<=t}).
     (In standard MTP: product of marginals Q = P_1 * P_2;
      In Variant C: low-rank joint energy Q ~ exp(E(a, b))).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.checkpoint_manager import build_model
from nanochat.dataset import list_parquet_files
from nanochat.tokenizer import get_tokenizer


def compute_joint_divergence_for_prefix(
    model: nn.Module,
    prefix_ids: List[int],
    top_k1: int = 16,
    top_k2: int = 16,
    device: str = "cuda",
    eps: float = 1e-12,
) -> Tuple[float, float]:
    """Compute KL(P* || Q) and TV(P*, Q) for a single sequence prefix."""
    inp = torch.tensor([prefix_ids], dtype=torch.long, device=device)
    vocab_size = model.config.vocab_size

    # Step 1: Forward trunk to get slot 1 marginal
    with torch.no_grad():
        logits1_seq = model(inp)
    logits1 = logits1_seq[0, -1, :vocab_size].float()
    probs1 = F.softmax(logits1, dim=-1)

    # Top-k1 candidates for token 1
    top1_vals, top1_indices = torch.topk(probs1, k=top_k1)
    top1_indices_list = top1_indices.tolist()

    # Step 2: In independent MTP (Arm 1), slot 2 marginal is unconditioned
    probs2_marginal = probs1  # In unaugmented model, shares marginal

    # Step 3: Autoregressive 2-step rollout for true joint P*(a, b)
    # Collect candidate pairs and true joint probabilities
    joint_true_dict = {}  # (a, b) -> P*(a, b)
    all_b_set = set()

    for a in top1_indices_list:
        p_a = float(probs1[a].item())
        inp_a = torch.tensor([prefix_ids + [a]], dtype=torch.long, device=device)
        with torch.no_grad():
            logits_ar_seq = model(inp_a)
        logits_ar = logits_ar_seq[0, -1, :vocab_size].float()
        probs_ar = F.softmax(logits_ar, dim=-1)

        top2_vals, top2_indices = torch.topk(probs_ar, k=top_k2)
        for b_idx in top2_indices.tolist():
            p_b_given_a = float(probs_ar[b_idx].item())
            joint_true_dict[(a, b_idx)] = p_a * p_b_given_a
            all_b_set.add(b_idx)

    # Step 4: Compute Q(a, b) on the exact same support
    joint_q_dict = {}
    for (a, b) in joint_true_dict.keys():
        # Product of marginals
        joint_q_dict[(a, b)] = float(probs1[a].item()) * float(probs2_marginal[b].item())

    # Step 5: Normalize both over the restricted support
    sum_true = sum(joint_true_dict.values())
    sum_q = sum(joint_q_dict.values())

    if sum_true <= 0 or sum_q <= 0:
        return 0.0, 0.0

    pairs = list(joint_true_dict.keys())
    p_true_norm = [joint_true_dict[p] / sum_true for p in pairs]
    q_norm = [joint_q_dict[p] / sum_q for p in pairs]

    # Step 6: TV and KL
    tv = 0.5 * sum(abs(p - q) for p, q in zip(p_true_norm, q_norm))
    kl = sum(p * math.log((p + eps) / (q + eps)) for p, q in zip(p_true_norm, q_norm) if p > 0)
    kl = max(0.0, kl)

    return float(tv), float(kl)


def main():
    parser = argparse.ArgumentParser(description="Direct Joint Divergence Benchmark")
    parser.add_argument("--checkpoint-dir", type=str, default="out/c00_sch_phase0/d4/DENSE_softmax_s1/depth_4/ckpt_base/base")
    parser.add_argument("--step", type=int, default=462)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tokenizer-dir", type=str, default="tokenizer")
    parser.add_argument("--num-samples", type=int, default=30)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--top-k1", type=int, default=12)
    parser.add_argument("--top-k2", type=int, default=12)
    parser.add_argument("--output-json", type=str, default="benchmarks/joint_divergence_results.json")
    args = parser.parse_args()

    print("=" * 70)
    print("DIRECT JOINT DIVERGENCE METRIC (SAP §6.5 Metric 1)")
    print("=" * 70)
    print(f"Loading checkpoint: {args.checkpoint_dir} (step {args.step})")

    model, tokenizer, meta = build_model(
        checkpoint_dir=args.checkpoint_dir,
        step=args.step,
        device=args.device,
        phase="eval",
        tokenizer_dir=args.tokenizer_dir,
    )
    model.eval()

    sample_prompts = [
        "In modern machine learning research, the efficiency of transformer",
        "The quick brown fox jumps over the lazy",
        "When designing distributed training algorithms for deep neural",
        "She decided to travel across the European continent during the summer of",
        "The economic implications of interest rate adjustments by the central",
        "In quantum physics, the observer effect states that the act of",
        "The primary function of the mitochondria in eukaryotic cells is to produce",
        "After downloading the latest version of the compiler, he executed the command",
        "The capital of Australia is Canberra, whereas the largest city is",
        "To optimize database performance, indexing frequently queried columns is",
    ]

    # Supplement with additional prompts to reach args.num_samples
    prompts = sample_prompts * (args.num_samples // len(sample_prompts) + 1)
    prompts = prompts[:args.num_samples]

    tv_list, kl_list = [], []
    print(f"\nComputing 2-step AR rollout joint over {len(prompts)} prompts...")
    start_t = time.time()

    for i, p in enumerate(prompts):
        ids = tokenizer.encode(p)
        if len(ids) > args.seq_len:
            ids = ids[:args.seq_len]
        tv, kl = compute_joint_divergence_for_prefix(
            model=model,
            prefix_ids=ids,
            top_k1=args.top_k1,
            top_k2=args.top_k2,
            device=args.device,
        )
        tv_list.append(tv)
        kl_list.append(kl)
        if (i + 1) % 10 == 0 or (i + 1) == len(prompts):
            print(f"  Processed {i+1:3d}/{len(prompts)} | TV = {tv:.4f}, KL = {kl:.4f}")

    elapsed = time.time() - start_t
    mean_tv = sum(tv_list) / len(tv_list)
    mean_kl = sum(kl_list) / len(kl_list)

    # 95% Confidence Intervals
    std_tv = math.sqrt(sum((x - mean_tv) ** 2 for x in tv_list) / max(1, len(tv_list) - 1))
    std_kl = math.sqrt(sum((x - mean_kl) ** 2 for x in kl_list) / max(1, len(kl_list) - 1))
    ci95_tv = 1.96 * std_tv / math.sqrt(len(tv_list))
    ci95_kl = 1.96 * std_kl / math.sqrt(len(kl_list))

    print("\n" + "=" * 70)
    print("JOINT DIVERGENCE SUMMARY (Arm 1 Baseline Reference)")
    print("=" * 70)
    print(f"Evaluation completed in {elapsed:.1f}s.")
    print(f"Mean TV Distance:    {mean_tv:.4f} +/- {ci95_tv:.4f} (95% CI: [{mean_tv - ci95_tv:.4f}, {mean_tv + ci95_tv:.4f}])")
    print(f"Mean KL Divergence: {mean_kl:.4f} +/- {ci95_kl:.4f} (95% CI: [{mean_kl - ci95_kl:.4f}, {mean_kl + ci95_kl:.4f}])")
    print("=" * 70)

    results = {
        "num_samples": len(prompts),
        "mean_tv": mean_tv,
        "ci95_tv": ci95_tv,
        "mean_kl": mean_kl,
        "ci95_kl": ci95_kl,
        "tv_list": tv_list,
        "kl_list": kl_list,
        "elapsed_s": elapsed,
    }

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {args.output_json}")


if __name__ == "__main__":
    main()
