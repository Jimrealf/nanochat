#!/usr/bin/env python3
"""Fork disambiguation and mode-mix benchmark for Multi-Token LLMs.

Reference: sap_research_plan.md §6.2, §6.4, §6.5 (Metric 2)

Evaluates:
  1. Mode-mix collision rate under parallel emission (e.g. "New" -> "Angeles" collision)
     at temperatures T in {0.0, 0.7, 1.0}.
  2. Head-2 Shannon entropy across test prompts.
  3. Arm 6 Oracle path: Head 2 conditioned on the realized x_{t+1}, providing the
     true headroom upper bound (Stage 0.5 headroom denominator).
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
from nanochat.tokenizer import get_tokenizer


# Curated fork benchmark items: prefix -> candidate slot1 branches -> expected/coherent slot2 continuations
FORK_BENCHMARK_DATA = [
    {
        "prompt": "She booked a flight to New",
        "branches": {
            " York": [" City", ",", " and", " to", " with", " State"],
            " Orleans": [",", " to", " and", " for", " with", " Louisiana"],
            " Jersey": [",", " and", " City", " Shore", " with"],
            " Mexico": [",", " and", " for", " to", " with"],
        },
    },
    {
        "prompt": "At the coffee shop he ordered an iced",
        "branches": {
            " tea": [",", " and", " with", " to", " please"],
            " coffee": [",", " and", " with", " to", " please"],
            " latte": [",", " and", " with", " to", " please"],
            " Americano": [",", " and", " with", " to"],
        },
    },
    {
        "prompt": "The tech company is headquartered in San",
        "branches": {
            " Francisco": [",", " and", " Bay", " California"],
            " Jose": [",", " California", " and"],
            " Diego": [",", " California", " and"],
            " Antonio": [",", " Texas", " and"],
        },
    },
    {
        "prompt": "The landmark building in the United",
        "branches": {
            " States": [" of", ",", " has", " is", " government"],
            " Kingdom": [",", " has", " is", " government"],
            " Arab": [" Emirates", ","],
            " Nations": [" headquarters", " building", " General"],
        },
    },
    {
        "prompt": "She baked fresh bread with garlic and",
        "branches": {
            " butter": [",", " and", " herbs", "."],
            " herb": [" butter", "s", " seasoning"],
            " olive": [" oil", "s"],
            " cheese": [",", " and", " on"],
        },
    },
    {
        "prompt": "The backend service was rewritten from Python to",
        "branches": {
            " Rust": [" for", " because", " to", ",", "."],
            " Go": [" for", " because", " to", ",", "."],
            " C": ["++", " to", " for"],
            " Java": [" for", " because", " to", ","],
        },
    },
    {
        "prompt": "The capital city of South",
        "branches": {
            " Korea": [" is", ",", " has"],
            " Africa": [" has", " is", ","],
            " America": [" has", " is", ","],
            " Carolina": [" is", ",", " has"],
            " Dakota": [" is", ",", " has"],
        },
    },
    {
        "prompt": "The patient complained of severe chest",
        "branches": {
            " pain": [",", " and", " radiating", " shortness"],
            " tightness": [",", " and", " with"],
            " discomfort": [",", " and", " that"],
        },
    },
    {
        "prompt": "He plays the acoustic",
        "branches": {
            " guitar": [",", " and", " in", " with"],
            " bass": [",", " and", " in"],
            " piano": [",", " and", " in"],
        },
    },
    {
        "prompt": "The database query was executed using",
        "branches": {
            " SQL": [" to", " and", " queries"],
            " PostgreSQL": [" to", " and", " database"],
            " MySQL": [" to", " and", " database"],
            " MongoDB": [" to", " and", " collection"],
        },
    },
    {
        "prompt": "They spent their summer vacation in Northern",
        "branches": {
            " California": [",", " where", " hiking"],
            " Ireland": [",", " where", " visiting"],
            " Italy": [",", " where", " enjoying"],
            " Virginia": [",", " near", " where"],
        },
    },
    {
        "prompt": "He poured a glass of red",
        "branches": {
            " wine": [",", " and", " into", " onto"],
            " fruit": [" juice", " punch"],
            " berry": [" juice", " syrup"],
        },
    },
]


def compute_shannon_entropy(probs: torch.Tensor, eps: float = 1e-12) -> float:
    """Compute Shannon entropy in nats: H(P) = -sum p log p."""
    probs_clamped = torch.clamp(probs, min=eps)
    entropy = -(probs_clamped * torch.log(probs_clamped)).sum(dim=-1)
    return float(entropy.mean().item())


def sample_next_token(logits: torch.Tensor, temperature: float = 0.0) -> int:
    """Sample next token from logits at given temperature (T=0 is greedy argmax)."""
    if temperature <= 1e-4:
        return int(torch.argmax(logits, dim=-1).item())
    probs = F.softmax(logits / temperature, dim=-1)
    sampled = torch.multinomial(probs, num_samples=1)
    return int(sampled.item())


def evaluate_fork_item(
    model: nn.Module,
    tokenizer,
    item: Dict,
    temperature: float,
    device: str,
    g2_head: Optional[nn.Module] = None,
) -> Dict:
    """Evaluate one fork benchmark item in both factorized MTP mode and Arm 6 Oracle mode."""
    prompt = item["prompt"]
    branches = item["branches"]

    prompt_ids = tokenizer.encode(prompt)
    inp = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    # 1. Forward trunk to get slot 1 and slot 2 marginals
    trunk_out = []
    def hook(module, args):
        trunk_out.append(args[0])

    handle = model.lm_head.register_forward_pre_hook(hook)
    try:
        with torch.no_grad():
            logits1_seq = model(inp)  # (1, T, V)
    finally:
        handle.remove()

    logits1 = logits1_seq[0, -1, :model.config.vocab_size].float()
    probs1 = F.softmax(logits1, dim=-1)
    entropy1 = compute_shannon_entropy(probs1)

    # Slot 2 factorized marginal (Arm 1 simulation)
    # If g2_head provided, use it on h_t; otherwise use trunk h_t through lm_head as baseline
    h_t = trunk_out[0][:, -1:, :]  # (1, 1, D)
    if g2_head is not None:
        h2 = g2_head(h_t)
        logits2 = model.lm_head(h2)[0, 0, :model.config.vocab_size].float()
    else:
        # Standard factorized assumption: slot 2 marginal before conditioning
        logits2 = logits1  # In unaugmented model, marginal at t+2 shares trunk state

    probs2 = F.softmax(logits2, dim=-1)
    entropy2 = compute_shannon_entropy(probs2)

    # Sample slot 1 token
    t1_id = sample_next_token(logits1, temperature=temperature)
    t1_text = tokenizer.decode([t1_id])

    # Sample slot 2 token under factorized prediction
    t2_factorized_id = sample_next_token(logits2, temperature=temperature)
    t2_factorized_text = tokenizer.decode([t2_factorized_id])

    # 2. Arm 6 Oracle Path: Condition head 2 on the realized t1_id
    inp_oracle = torch.tensor([prompt_ids + [t1_id]], dtype=torch.long, device=device)
    with torch.no_grad():
        logits_oracle_seq = model(inp_oracle)
    logits_oracle = logits_oracle_seq[0, -1, :model.config.vocab_size].float()
    probs_oracle = F.softmax(logits_oracle, dim=-1)
    entropy_oracle = compute_shannon_entropy(probs_oracle)

    t2_oracle_id = sample_next_token(logits_oracle, temperature=temperature)
    t2_oracle_text = tokenizer.decode([t2_oracle_id])

    # 3. Check consistency against known coherent branches
    # Identify which branch slot 1 landed in (if any)
    matched_branch = None
    for branch_key in branches.keys():
        if t1_text.strip().lower() == branch_key.strip().lower() or branch_key.strip().lower() in t1_text.strip().lower():
            matched_branch = branch_key
            break

    factorized_collision = False
    oracle_collision = False

    if matched_branch is not None:
        coherent_continuations = branches[matched_branch]
        # Incompatible continuations from competing branches
        competing_continuations = []
        for other_k, other_conts in branches.items():
            if other_k != matched_branch:
                competing_continuations.extend(other_conts)

        # Check factorized collision: emitted token matches competing branch but not own branch
        matches_coherent_fact = any(c.strip().lower() in t2_factorized_text.strip().lower() for c in coherent_continuations)
        matches_competing_fact = any(c.strip().lower() in t2_factorized_text.strip().lower() for c in competing_continuations)

        if matches_competing_fact and not matches_coherent_fact:
            factorized_collision = True

        matches_coherent_ora = any(c.strip().lower() in t2_oracle_text.strip().lower() for c in coherent_continuations)
        matches_competing_ora = any(c.strip().lower() in t2_oracle_text.strip().lower() for c in competing_continuations)
        if matches_competing_ora and not matches_coherent_ora:
            oracle_collision = True

    return {
        "prompt": prompt,
        "t1_text": t1_text,
        "matched_branch": matched_branch,
        "t2_factorized": t2_factorized_text,
        "t2_oracle": t2_oracle_text,
        "factorized_collision": factorized_collision,
        "oracle_collision": oracle_collision,
        "entropy1": entropy1,
        "entropy2_factorized": entropy2,
        "entropy2_oracle": entropy_oracle,
    }


def main():
    parser = argparse.ArgumentParser(description="Eval Fork Branching and Mode-Mix Benchmark")
    parser.add_argument("--checkpoint-dir", type=str, default="out/c00_sch_phase0/d4/DENSE_softmax_s1/depth_4/ckpt_base/base")
    parser.add_argument("--step", type=int, default=462)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tokenizer-dir", type=str, default="tokenizer")
    parser.add_argument("--temperatures", nargs="+", type=float, default=[0.0, 0.7, 1.0])
    parser.add_argument("--output-json", type=str, default="benchmarks/branching_results.json")
    args = parser.parse_args()

    print("=" * 70)
    print("FORK DISAMBIGUATION & MODE-MIX BENCHMARK (SAP §6.2, §6.5)")
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

    all_results = {}

    for temp in args.temperatures:
        print(f"\nEvaluating at Temperature T = {temp:.1f}...")
        item_reports = []
        fact_collisions = 0
        oracle_collisions = 0
        total_matched = 0
        h2_fact_entropies = []
        h2_oracle_entropies = []

        for item in FORK_BENCHMARK_DATA:
            rep = evaluate_fork_item(
                model=model,
                tokenizer=tokenizer,
                item=item,
                temperature=temp,
                device=args.device,
            )
            item_reports.append(rep)
            if rep["matched_branch"] is not None:
                total_matched += 1
                if rep["factorized_collision"]:
                    fact_collisions += 1
                if rep["oracle_collision"]:
                    oracle_collisions += 1

            h2_fact_entropies.append(rep["entropy2_factorized"])
            h2_oracle_entropies.append(rep["entropy2_oracle"])

        fact_collision_rate = fact_collisions / max(1, total_matched)
        oracle_collision_rate = oracle_collisions / max(1, total_matched)
        headroom_gap = fact_collision_rate - oracle_collision_rate
        mean_h2_fact = sum(h2_fact_entropies) / len(h2_fact_entropies)
        mean_h2_oracle = sum(h2_oracle_entropies) / len(h2_oracle_entropies)

        print(f"  Matched branch prefixes: {total_matched}/{len(FORK_BENCHMARK_DATA)}")
        print(f"  Arm 1 (Factorized) Mode-Mix Rate: {fact_collision_rate * 100:.1f}%")
        print(f"  Arm 6 (Oracle) Mode-Mix Rate:     {oracle_collision_rate * 100:.1f}%")
        print(f"  Headroom Gap (Arm 1 -> Arm 6):    {headroom_gap * 100:+.1f}%")
        print(f"  Mean Head-2 Entropy (Factorized): {mean_h2_fact:.3f} nats")
        print(f"  Mean Head-2 Entropy (Oracle):     {mean_h2_oracle:.3f} nats")

        all_results[f"T_{temp}"] = {
            "temperature": temp,
            "fact_collision_rate": fact_collision_rate,
            "oracle_collision_rate": oracle_collision_rate,
            "headroom_gap": headroom_gap,
            "mean_h2_fact_entropy": mean_h2_fact,
            "mean_h2_oracle_entropy": mean_h2_oracle,
            "reports": item_reports,
        }

    # Stage 0.5 Headroom Gate Check (§6.2, §7)
    print("\n" + "=" * 70)
    print("STAGE 0.5 HEADROOM GATE CHECK (§6.2)")
    print("=" * 70)
    greedy_gap = all_results.get("T_0.0", {}).get("headroom_gap", 0.0)
    print(f"Greedy Headroom Gap: {greedy_gap * 100:+.1f}%")
    if greedy_gap >= 0.05 or all_results.get("T_0.7", {}).get("headroom_gap", 0.0) >= 0.05:
        headroom_verdict = "STAGE 0.5 PASSED (CLEAR HEADROOM GAP DETECTED)"
        headroom_action = "Arm 1 -> Arm 6 headroom is established. Benchmark has valid experimental signal."
    else:
        headroom_verdict = "STAGE 0.5 WARNING: NARROW HEADROOM GAP"
        headroom_action = "Expand benchmark dataset or evaluate at larger model scale to maximize headroom."

    print(f"Verdict: {headroom_verdict}")
    print(f"Action:  {headroom_action}")
    print("=" * 70)

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    summary_data = {
        "gate_verdict": headroom_verdict,
        "gate_action": headroom_action,
        "results_by_temperature": all_results,
    }
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print(f"Branching benchmark results saved to {args.output_json}")


if __name__ == "__main__":
    main()
