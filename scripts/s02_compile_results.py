"""Compile downloaded S02 Modal JSON rows into a durable learning-curve verdict."""

from __future__ import annotations

import argparse
import glob
import json
import math
import os


def load(patterns):
    rows, sources = [], []
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            sources.append(path)
            with open(path) as f:
                if path.endswith(".jsonl"):
                    rows.extend(json.loads(line) for line in f if line.strip())
                else:
                    obj = json.load(f)
                    rows.extend(obj if isinstance(obj, list) else [obj])
    unique = {(r.get("tag"), r["mode"], r["T"], r["seed"]): r for r in rows}
    return list(unique.values()), sources


def num(x):
    return f"{x:.4f}" if isinstance(x, (int, float)) and math.isfinite(x) else "n/a"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("inputs", nargs="*", default=[
        "out/s02_sap_modal/results.jsonl",
        "out/s02_sap_modal/remote/*s02*.json",
        "out/s02_sap_modal/remote/sap_stageA_modal/*s02*.json",
    ])
    p.add_argument("--out", default="s02_sap_compiled.log")
    args = p.parse_args()
    rows, sources = load(args.inputs)
    lines = [
        "S02 SAP CONVERGENCE + CORRELATED-TREE LOG",
        "=========================================",
        "Gate: T=4 block KL <= 0.10 nats/block AND invalid <= 0.03.",
        "Modal profile: blessingjim31-workspace",
        "",
        "Sources:", *[f"  {s}" for s in sources], "",
    ]
    any_tree_pass = False
    for row in sorted(rows, key=lambda r: r.get("tag", "")):
        tag = row.get("tag", "untagged")
        lines += [tag, "step    phase       block_KL  invalid  oracle_anchor"]
        for m in row.get("training_milestones", []):
            phase = "head-only" if m.get("head_only") else "joint"
            lines.append(f"{m['step']:5d}   {phase:9s}  {num(m.get('block_kl')):>8s}  "
                         f"{num(m.get('invalid_rate')):>7s}  "
                         f"{num(m.get('invalid_rate_oracle_anchors')):>13s}")
        passed = row.get("block_kl") is not None and row["block_kl"] <= 0.10 \
            and row["invalid_rate"] <= 0.03
        lines.append(f"final: KL={num(row.get('block_kl'))} invalid={num(row.get('invalid_rate'))} "
                     f"{'ADVANCE' if passed else 'KILL'}")
        lines.append("")
        any_tree_pass |= row["mode"] == "sir_tree" and passed

    lines += ["VERDICT", "-------"]
    if any_tree_pass:
        lines.append("At least one correlated-tree arm clears the exact gate; it may advance to a cost-matched systems test.")
    else:
        lines.append("No correlated-tree arm clears the exact gate; no depth-8 language sweep is authorised.")
    lines += ["", "RAW JSON ROWS", "-------------"]
    lines.extend(json.dumps(r, sort_keys=True) for r in sorted(rows, key=lambda r: r.get("tag", "")))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {args.out}: {len(rows)} rows from {len(sources)} files")


if __name__ == "__main__":
    main()
