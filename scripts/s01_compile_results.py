"""Compile the S01 Modal phrase-HMM rows into one durable text verdict.

The output is intentionally plain text plus complete JSON rows so it remains useful after the
individual Modal apps and local terminal history are gone.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
from collections import defaultdict


RUNS = {
    "s01": ("v1: one on-policy draft + expected CE", "ap-aniDuP27Q1DKrR1un2jlm2"),
    "s01_iw4": ("v2: K=4 marginal likelihood + reference-weight policy credit",
                 "ap-W81oBQQx6KeE59p2ZhJyv4 + supplemental ap-QoaBpb0sxKwbRHjr553Mqo"),
    "s01_postmix05": ("v3: v2 + fixed 0.5 token-posterior bridge", "ap-bpVVbZVLqClNexIC2XbGAG"),
    "s01_postmix05_stride21": ("v3 hierarchy correction: T=4 coarse/fine stride 2/1",
                                "ap-TTuh73gOInInS3AYlv6kGE"),
}


def load_rows(patterns):
    rows, sources = [], []
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            sources.append(path)
            with open(path) as f:
                for line in f:
                    if line.strip():
                        rows.append(json.loads(line))
    # A retried local entry point may append a returned row twice. Keep its latest copy.
    unique = {}
    for row in rows:
        unique[(row.get("tag"), row["mode"], row["T"], row["seed"])] = row
    return list(unique.values()), sources


def fnum(x, width=8, precision=3):
    return f"{x:{width}.{precision}f}" if isinstance(x, (int, float)) and math.isfinite(x) else f"{str(x):>{width}}"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "inputs",
        nargs="*",
        default=[
            "out/s01_sap_modal/stage_a_*/results.jsonl",
            "out/s01_sap_modal/remote_stage_a_json/sap_stageA_modal/*s01*.json",
        ],
    )
    p.add_argument("--out", default="s01_sap_compiled.log")
    args = p.parse_args()
    rows, sources = load_rows(args.inputs)
    by_tag = defaultdict(list)
    for row in rows:
        by_tag[row.get("tag", "untagged")].append(row)

    lines = [
        "S01 SAP COMPILED EXPERIMENT LOG",
        "===============================",
        "Exact phrase-HMM gate: T=4, block KL <= 0.10 nats/block AND invalid <= 0.03.",
        "FineWeb depth-8 was pre-registered to run only after this gate.",
        "Modal profile: blessingjim31-workspace",
        "",
        "Sources:",
        *[f"  {path}" for path in sources],
        "",
    ]
    for tag in RUNS:
        group = sorted(by_tag.get(tag, []), key=lambda r: r["mode"])
        label, app_id = RUNS[tag]
        lines += [f"{tag}: {label}", f"Modal app: {app_id}",
                  "mode             block_KL  invalid  train_s  verdict"]
        for row in group:
            advance = row.get("block_kl") is not None and row["block_kl"] <= 0.10 and row["invalid_rate"] <= 0.03
            lines.append(f"{row['mode']:16s} {fnum(row.get('block_kl'))}  {fnum(row['invalid_rate'], 7)}  "
                         f"{fnum(row.get('seconds'), 7, 1)}  {'ADVANCE' if advance else 'KILL'}")
        if group:
            best_kl = min(group, key=lambda r: float("inf") if r.get("block_kl") is None else r["block_kl"])
            best_invalid = min(group, key=lambda r: r["invalid_rate"])
            lines += [f"best KL: {best_kl['mode']} {best_kl.get('block_kl')}",
                      f"best invalid: {best_invalid['mode']} {best_invalid['invalid_rate']}"]
        else:
            lines.append("no returned rows")
        lines.append("")

    lines += [
        "VERDICT",
        "-------",
        "No S01 mechanism clears the exact gate. V1 reproduces the product-of-marginals",
        "solution. V2 proves post-sampling credit affects the latent draft but closes too little",
        "of the gap. V3 produces excellent posterior-conditioned training likelihood but fails",
        "under prior-only sampling, diagnosing aggregate-posterior mismatch. Correct scale-relative",
        "anchors improve invalidity but not likelihood; the pyramid is overconfident.",
        "Therefore the costlier depth-8 H100 sweep was not launched.",
        "",
        "RAW JSON ROWS",
        "-------------",
    ]
    for row in sorted(rows, key=lambda r: (r.get("tag", ""), r["mode"], r["T"], r["seed"])):
        lines.append(json.dumps(row, sort_keys=True))
    text = "\n".join(lines) + "\n"
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write(text)
    print(f"wrote {args.out}: {len(rows)} unique rows from {len(sources)} result files")


if __name__ == "__main__":
    main()
