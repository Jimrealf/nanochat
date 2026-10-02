"""
scripts/sap_llm_judge.py: Pairwise LLM-as-a-Judge evaluation for SAP block head vs. AR baselines.

Evaluates generated continuation pairs from `scripts/sap_eval_generation.py` (gen_*.jsonl)
using a high-capability judge model (e.g. GPT-4o, Claude 3.5 Sonnet, or Gemini 1.5 Pro).

Scientific standards for A* conferences (NeurIPS / ICML / ICLR):
1. Blind evaluation: Continuation source (block head vs. AR) is anonymous.
2. Positional debiasing (Order Swapping): Every prefix is judged twice:
     Pass 1: A = Block, B = AR
     Pass 2: A = AR,    B = Block
   A method only wins if it is consistently chosen regardless of order.
   Inconsistent order votes are classified as "Tie (Position Inconsistency)".
3. Chain-of-Thought Rubric: The judge must output reasoning evaluating
   coherence, factuality, and fluency before providing the categorical decision.
4. Statistical confidence: Reports Win/Tie/Loss rates and Wilson score 95% intervals.

Usage:
  # Using OpenAI (GPT-4o / GPT-4o-mini):
  export OPENAI_API_KEY="sk-..."
  python -m scripts.sap_llm_judge --pairs out/s00_sap/gen_SAP_local_T4_s1_d8.jsonl --provider openai --model gpt-4o-mini

  # Using Anthropic Claude:
  export ANTHROPIC_API_KEY="sk-ant-..."
  python -m scripts.sap_llm_judge --pairs out/s00_sap/gen_SAP_local_T4_s1_d8.jsonl --provider anthropic --model claude-3-5-sonnet-20241022

  # Mock / dry-run test without API keys:
  python -m scripts.sap_llm_judge --pairs out/s00_sap/gen_SAP_local_T4_s1_d8.jsonl --provider mock --max-pairs 10
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import urllib.request
import urllib.error


JUDGE_PROMPT_TEMPLATE = """You are an expert linguistic evaluator assessing the quality of two competing AI continuations for a given prefix.

[Prefix]:
\"\"\"{prefix}\"\"\"

[Continuation A]:
\"\"\"{cont_a}\"\"\"

[Continuation B]:
\"\"\"{cont_b}\"\"\"

Evaluate the quality of Continuation A vs. Continuation B based on:
1. Coherence & Flow: Does the continuation naturally and logically follow the prefix?
2. Fluency & Syntax: Is the grammar correct and natural, without awkward token repetitions or phrase loops?
3. Information Density: Does it provide substantive, non-degenerate text?

Instructions:
- Provide a brief 1-2 sentence rationale in "analysis".
- State your verdict strictly as "A", "B", or "Tie".
- Reply ONLY with a valid JSON object in the exact format:
{{"analysis": "<short rationale>", "verdict": "A" | "B" | "Tie"}}
"""


def _call_openai(prompt: str, model: str, api_key: str) -> str:
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are an objective AI evaluator. Always respond with strict JSON."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0,
        "response_format": {"type": "json_object"}
    }
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers)
    with urllib.request.urlopen(req, timeout=30) as resp:
        res = json.loads(resp.read().decode("utf-8"))
        return res["choices"][0]["message"]["content"]


def _call_anthropic(prompt: str, model: str, api_key: str) -> str:
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01"
    }
    payload = {
        "model": model,
        "max_tokens": 512,
        "temperature": 0.0,
        "system": "You are an objective AI evaluator. Always respond with strict JSON.",
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers)
    with urllib.request.urlopen(req, timeout=30) as resp:
        res = json.loads(resp.read().decode("utf-8"))
        return res["content"][0]["text"]


def _call_mock(prompt: str, cont_a: str, cont_b: str) -> str:
    # Heuristic mock: penalize repetitions (lower distinct n-grams) or tie
    words_a = cont_a.split()
    words_b = cont_b.split()
    uniq_a = len(set(words_a)) / max(1, len(words_a))
    uniq_b = len(set(words_b)) / max(1, len(words_b))
    diff = uniq_a - uniq_b
    if abs(diff) < 0.05:
        verdict = "Tie"
    elif diff > 0:
        verdict = "A"
    else:
        verdict = "B"
    return json.dumps({"analysis": f"Mock evaluation based on lexical diversity ({uniq_a:.2f} vs {uniq_b:.2f})",
                       "verdict": verdict})


def parse_judgment(raw: str) -> str:
    clean = raw.strip()
    if clean.startswith("```json"):
        clean = clean[7:]
    if clean.startswith("```"):
        clean = clean[3:]
    if clean.endswith("```"):
        clean = clean[:-3]
    try:
        obj = json.loads(clean.strip())
        v = str(obj.get("verdict", "")).strip()
        if v in ("A", "B", "Tie"):
            return v
    except Exception:
        pass
    # Fallback substring
    if '"verdict": "A"' in raw:
        return "A"
    if '"verdict": "B"' in raw:
        return "B"
    return "Tie"


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for binomial proportions."""
    if total == 0:
        return 0.0, 0.0
    p = successes / total
    denom = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denom
    margin = (z * math.sqrt(p * (1 - p) / total + z**2 / (4 * total**2))) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pairs", type=str, required=True, help="Path to gen_*.jsonl from sap_eval_generation.py")
    parser.add_argument("--provider", type=str, default="openai", choices=["openai", "anthropic", "mock"])
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="Judge model name")
    parser.add_argument("--max-pairs", type=int, default=200, help="Max number of pairs to evaluate (default: 200)")
    parser.add_argument("--out", type=str, default="", help="Output judgments jsonl path")
    args = parser.parse_args()

    api_key = ""
    if args.provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            print("Error: OPENAI_API_KEY environment variable not set.", file=sys.stderr)
            sys.exit(1)
    elif args.provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        if not api_key:
            print("Error: ANTHROPIC_API_KEY environment variable not set.", file=sys.stderr)
            sys.exit(1)

    out_path = args.out or args.pairs.replace(".jsonl", "_judged.jsonl")
    if out_path == args.pairs:
        out_path += ".judged.jsonl"

    pairs = []
    with open(args.pairs, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                pairs.append(json.loads(line))
            if len(pairs) >= args.max_pairs:
                break

    print(f"Loaded {len(pairs)} pairs from {args.pairs}")
    print(f"Evaluating with {args.provider} ({args.model}) with Swap-Pair Debiasing...")

    results = []
    block_wins = 0
    ar_wins = 0
    ties = 0

    out_file = open(out_path, "w", encoding="utf-8")

    for i, p in enumerate(pairs):
        prefix = p.get("prefix", "")
        cont_block = p.get("block", "")
        cont_ar = p.get("ar", "")

        # Pass 1: A = Block, B = AR
        prompt_1 = JUDGE_PROMPT_TEMPLATE.format(prefix=prefix, cont_a=cont_block, cont_b=cont_ar)
        # Pass 2: A = AR,    B = Block
        prompt_2 = JUDGE_PROMPT_TEMPLATE.format(prefix=prefix, cont_a=cont_ar, cont_b=cont_block)

        try:
            if args.provider == "openai":
                r1 = _call_openai(prompt_1, args.model, api_key)
                r2 = _call_openai(prompt_2, args.model, api_key)
            elif args.provider == "anthropic":
                r1 = _call_anthropic(prompt_1, args.model, api_key)
                r2 = _call_anthropic(prompt_2, args.model, api_key)
            else:
                r1 = _call_mock(prompt_1, cont_block, cont_ar)
                r2 = _call_mock(prompt_2, cont_ar, cont_block)

            v1 = parse_judgment(r1)  # A=Block, B=AR
            v2 = parse_judgment(r2)  # A=AR,    B=Block

            # Resolve verdict with position swap:
            # If Pass 1 voted A (Block) and Pass 2 voted B (Block) => Block Win
            # If Pass 1 voted B (AR)    and Pass 2 voted A (AR)    => AR Win
            # If both voted Tie, or if vote flipped due to order   => Tie
            if v1 == "A" and v2 == "B":
                final_verdict = "BLOCK_WIN"
                block_wins += 1
            elif v1 == "B" and v2 == "A":
                final_verdict = "AR_WIN"
                ar_wins += 1
            else:
                final_verdict = "TIE"
                ties += 1

            record = {
                "pair_id": i,
                "prefix": prefix,
                "verdict": final_verdict,
                "pass1_order": "A=Block, B=AR",
                "pass1_raw": r1,
                "pass2_order": "A=AR, B=Block",
                "pass2_raw": r2,
                "ar_ref_nll": p.get("ar_ref_nll"),
                "block_ref_nll": p.get("block_ref_nll")
            }
            out_file.write(json.dumps(record) + "\n")
            out_file.flush()

            if (i + 1) % 10 == 0 or (i + 1) == len(pairs):
                print(f"[{i+1}/{len(pairs)}] Block Wins: {block_wins} | AR Wins: {ar_wins} | Ties: {ties}")

        except Exception as e:
            print(f"Error on pair {i}: {e}", file=sys.stderr)
            time.sleep(2)

    out_file.close()

    total = block_wins + ar_wins + ties
    if total > 0:
        win_rate = (block_wins + 0.5 * ties) / total
        b_rate = block_wins / total
        a_rate = ar_wins / total
        t_rate = ties / total
        b_lo, b_hi = wilson_interval(block_wins, total)

        print("\n" + "=" * 60)
        print("           LLM-as-a-Judge Evaluation Summary           ")
        print("=" * 60)
        print(f"Total Evaluated Pairs: {total}")
        print(f"Block Wins:            {block_wins:4d} ({b_rate * 100:5.1f}%) [95% CI: {b_lo*100:.1f}% - {b_hi*100:.1f}%]")
        print(f"AR (Dense) Wins:       {ar_wins:4d} ({a_rate * 100:5.1f}%)")
        print(f"Ties (or order flip):  {ties:4d} ({t_rate * 100:5.1f}%)")
        print(f"Standard Win Rate:     {win_rate * 100:5.1f}% (Wins + 0.5 * Ties)")
        print(f"Net Score (Block - AR): {((block_wins - ar_wins) / total) * 100:+5.1f}%")
        print("=" * 60)
        print(f"Detailed judgments saved to: {out_path}")


if __name__ == "__main__":
    main()
