"""
Validation bpb per position bucket for one or more checkpoints on identical rows.

S09 uses it as the information oracle for suffix-state one-pass generators (S07 hypothesis B):
a model trained to attend only to the prompt plus its last k positions (`--prompt-window k`) is
scored per position against a full-context model, so its loss far from the prompt measures what
forgetting the generated text beyond a k-token suffix costs.

    python -m scripts.sap_position_bpb --tokenizer-dir tokenizer --data-dir data \\
        --model dense:out/.../S09_dense_L_s1 --model win32:out/.../S09_win32_s1:32 --prompt-len 128
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math

import torch

from nanochat.checkpoint_manager import build_model, find_last_step
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from nanochat.lanes import prompt_window_mask, separator_batch, separator_mask
from nanochat.tokenizer import get_token_bytes


args_last_ignored = False


def rows_hash(rows):
    """Short sha256 of the evaluation rows (inputs and targets), to prove that two workspaces
    score identical rows (S13 compares against logged references instead of retraining them)."""
    h = hashlib.sha256()
    for x, y in rows:
        h.update(x.detach().cpu().to(torch.int64).numpy().tobytes())
        h.update(y.detach().cpu().to(torch.int64).numpy().tobytes())
    return h.hexdigest()[:16]


@torch.no_grad()
def bucket_bpb(model, rows, token_bytes, edges, mask=None, batch=8, prep=None, wb=None, lanes=None, splice=None,
               xform=None):
    """bpb per [edges[i], edges[i+1]) input-position bucket over the (x, y) rows. prep(x, y), if
    given, rewrites each batch (the S11 separator layout and its ignored targets); xform(x)
    rewrites the inputs of a one-stream order (S16-F step_inputs)."""
    nats = torch.zeros(len(edges) - 1, dtype=torch.float64)
    nbytes = torch.zeros(len(edges) - 1, dtype=torch.float64)
    pos_nats = pos_bytes = None                         # per input position, summed over rows
    for i in range(0, len(rows), batch):
        x = torch.stack([r[0] for r in rows[i:i + batch]])
        y = torch.stack([r[1] for r in rows[i:i + batch]])
        shift = 0
        if prep is not None:
            x, y, shift = prep(x, y)
        kw = {} if mask is None else {"lane_mask": mask}
        if lanes is not None:                           # S08 plain lanes: lane-start inputs, lane mask
            from nanochat.lanes import lane_inputs
            x = lane_inputs(x, *lanes)
        if xform is not None:
            x = xform(x)
        if wb is not None:                              # window bisection: query k scores token k
            from nanochat.wbisect import wb_forward     # = the dense model's position k - 1 target
            steps, mask_token = wb
            own = wb_forward(model, x, steps, mask_token, loss_reduction="none").view(x.shape).double()
            loss = torch.cat([own[:, 1:], torch.zeros_like(own[:, :1])], 1)
            y = y.clone()
            y[:, -1] = -1
        elif splice is not None:                        # S13 splice codes: the model builds the layout
            loss = model(x, y, loss_reduction="none", splice=splice).view(y.shape).double()
        else:
            loss = model(x, y, loss_reduction="none", **kw).view(y.shape).double()
        if wb is None and args_last_ignored:            # score the same targets as window bisection
            y = y.clone()
            y[:, -1] = -1
        if shift:                                       # separator layout: back to original positions
            P, m = shift
            loss = torch.cat([loss[:, :P], loss[:, P + m:], torch.zeros_like(loss[:, :m])], 1)
            y = torch.cat([y[:, :P], y[:, P + m:], torch.full_like(y[:, :m], -1)], 1)
        valid = y >= 0
        b = torch.where(valid, token_bytes[y.clamp_min(0)], torch.zeros_like(y)).double()
        loss = loss * (b > 0)
        if pos_nats is None:
            pos_nats = torch.zeros(loss.size(1), dtype=torch.float64)
            pos_bytes = torch.zeros(loss.size(1), dtype=torch.float64)
        pos_nats += loss.sum(0).cpu()
        pos_bytes += b.sum(0).cpu()
        for j in range(len(edges) - 1):
            nats[j] += loss[:, edges[j]:edges[j + 1]].sum().cpu()
            nbytes[j] += b[:, edges[j]:edges[j + 1]].sum().cpu()
    bpb = (nats / (nbytes.clamp_min(1) * torch.log(torch.tensor(2.0, dtype=torch.float64)))).tolist()
    return bpb, pos_nats, pos_bytes


def lookahead_band(by_offset):
    """S16: a lane's excess summed over offsets ceil(S/4)..S-2 (8-28 at S = 30), where the next
    lane's first tokens are 2 to about 3S/4 positions ahead. Lookahead gains there straddle the
    sign of the excess, so the deficit/recovery split books part of them as deficit (S16-C's gain
    at offsets 8-15 was); from 2026-10-06 this band is the recovery metric for S16 gates."""
    S = len(by_offset)
    return sum(by_offset[-(-S // 4):S - 1])


def lane_offset_report(N, P, L, own, ref, n_rows=0):
    """S08 plain lanes: the cost against the reference by input offset within a lane (step s),
    pooled over lanes 1..L-1 (lane 0 continues the prefix directly and is reported alone). own,
    ref: (pos_nats, pos_bytes) on identical targets, summed over n_rows rows. Offset 0 is a lane's
    first prediction, made from the lane-start token with no left context of its own; offset S-1
    predicts the next lane's first token, after that lane's later tokens are known. With n_rows,
    also the extra nats per lane in absolute units (S15 R1, comparable across model sizes and with
    the order oracle's per-lane TC), the reference's nats per token, and the per-lane excess by
    offset split into the deficit (offsets that cost more than the reference) and the recovery
    (offsets that cost less: late tokens that read the next lane's early ones), and the lookahead
    band (lookahead_band)."""
    S = (N - P) // L
    p = torch.arange(N)
    s, j = (p - P) % S, (p - P) // S
    groups = [("0", s == 0), ("1", s == 1), ("2", s == 2), ("3", s == 3), ("4-7", (s >= 4) & (s < 8)),
              ("8-15", (s >= 8) & (s < 16)), (f"16-{S - 2}", (s >= 16) & (s < S - 1)), (f"{S - 1} (end)", s == S - 1)]
    out = {}
    for name, g in groups:
        g = g & (p >= P) & (j >= 1)
        if g.any() and ref[0][g].sum() > 0:
            out[name] = float(own[0][g].sum() / ref[0][g].sum())
    lane0 = (p >= P) & (j == 0)
    out["lane 0"] = float(own[0][lane0].sum() / ref[0][lane0].sum())
    lanes = (p >= P) & (j >= 1)
    tax = float(own[0][lanes].sum() - ref[0][lanes].sum())
    first = float(own[0][lanes & (s == 0)].sum() - ref[0][lanes & (s == 0)].sum())
    out["share of the lanes' extra nats at offset 0"] = first / tax if tax > 0 else float("nan")
    if n_rows > 0:
        out["extra nats per lane"] = tax / (n_rows * (L - 1))
        out["reference nats per token"] = float(ref[0][lanes].sum()) / (n_rows * int(lanes.sum()))
        by_offset = [float(own[0][lanes & (s == k)].sum() - ref[0][lanes & (s == k)].sum()) / (n_rows * (L - 1))
                     for k in range(S)]
        out["deficit nats per lane"] = sum(v for v in by_offset if v > 0)
        out["recovery nats per lane"] = sum(v for v in by_offset if v < 0)
        out["lookahead nats per lane"] = lookahead_band(by_offset)
        out["extra nats per lane by offset"] = by_offset
    return out


@torch.no_grad()
def splice_offset_report(model, rows, token_bytes, P, L, lane_token, batch=8):
    """S13 splice codes, absolute: mean nats per scored token by offset within lanes 1..L-1, lane 0
    alone, and the mean code nats per junction. No reference model is needed; S13 compares these
    with the logged plain-lanes profile."""
    from nanochat.splice import splice_loss
    N = rows[0][0].numel()
    S = (N - P) // L
    p = torch.arange(N)
    s, j = (p - P) % S, (p - P) // S
    nats = torch.zeros(N, dtype=torch.float64)
    cnt = torch.zeros(N, dtype=torch.float64)
    code_nats, code_n = 0.0, 0
    for i in range(0, len(rows), batch):
        x = torch.stack([r[0] for r in rows[i:i + batch]])
        y = torch.stack([r[1] for r in rows[i:i + batch]])
        tl, cl = splice_loss(model, x, y, P, L, lane_token, return_parts=True)
        valid = (y >= 0) & (token_bytes[y.clamp_min(0)] > 0)
        nats += (tl.double() * valid).sum(0).cpu()
        cnt += valid.double().sum(0).cpu()
        code_nats += float(cl.double().sum())
        code_n += cl.numel()
    groups = [("0", s == 0), ("1", s == 1), ("2", s == 2), ("3", s == 3), ("4-7", (s >= 4) & (s < 8)),
              (f"8-{S - 2}", (s >= 8) & (s < S - 1)), (f"{S - 1} (end)", s == S - 1)]
    out = {}
    for name, g in groups:
        g = g & (p >= P) & (j >= 1)
        out[name] = float(nats[g].sum() / cnt[g].sum().clamp_min(1))
    lane0 = (p >= P) & (j == 0)
    out["lane 0"] = float(nats[lane0].sum() / cnt[lane0].sum().clamp_min(1))
    out["all block tokens"] = float(nats[p >= P].sum() / cnt[p >= P].sum().clamp_min(1))
    out["code per junction"] = code_nats / max(code_n, 1)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", action="append", required=True,
                   help="name:checkpoint_dir[:k] with k the prompt window (omit for full context), or "
                        "name:checkpoint_dir:sepM / sepMfull for the S11 separator oracle (M slots), or "
                        "name:checkpoint_dir:wbN for a window-bisection model with N-token windows, or "
                        "name:checkpoint_dir:blL_N for bridged lanes (L intervals, N-token separators), or "
                        "name:checkpoint_dir:lnL for S08 plain lanes (prefix --wb-prefix), or "
                        "name:checkpoint_dir:boL_N for S16-F one-stream bridged lanes, or "
                        "name:checkpoint_dir:loL for plain lanes trained as a two-stream order, or "
                        "name:checkpoint_dir:sdK or sdK_M for seeded middle-out lanes (K intervals, M-token seed window)")
    p.add_argument("--wb-prefix", type=int, default=128, help="S11: left-to-right prefix of window-bisection models")
    p.add_argument("--sep-split", type=int, default=1024, help="S11: where separator slots sit")
    p.add_argument("--within-doc", type=int, default=0,
                   help="S11: if > 0, score only rows with no document start within this many tokens of the split")
    p.add_argument("--tokenizer-dir", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--rows", type=int, default=256)
    p.add_argument("--prompt-len", type=int, default=128)
    p.add_argument("--edges", type=str, default="0,128,512,1024,2048")
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--hash-only", action="store_true", help="print the evaluation-rows hash and exit (no model)")
    p.add_argument("--seq-len", type=int, default=2048, help="row length for --hash-only")
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    edges = [int(e) for e in args.edges.split(",")]
    if args.hash_only:
        from nanochat.tokenizer import get_tokenizer
        tok = get_tokenizer(args.tokenizer_dir)
        loader = tokenizing_distributed_data_loader_bos_bestfit(tok, 8, args.seq_len, split="val",
                                                                data_dir=args.data_dir, device="cpu")
        rows = []
        while len(rows) < args.rows:
            x, y = next(loader)
            rows += list(zip(x, y))
        print(f"evaluation rows hash ({args.rows} rows of {args.seq_len}): {rows_hash(rows[:args.rows])}")
        return
    specs = [m.split(":") for m in args.model]
    first = build_model(specs[0][1], find_last_step(specs[0][1]), device, phase="eval",
                        tokenizer_dir=args.tokenizer_dir)
    tok = first[1]
    N = first[0].config.sequence_len
    loader = tokenizing_distributed_data_loader_bos_bestfit(tok, 8, N, split="val", data_dir=args.data_dir)
    rows = []
    while len(rows) < args.rows:
        x, y = next(loader)
        rows += list(zip(x.to(device), y.to(device)))
    rows = rows[:args.rows]
    print(f"evaluation rows hash ({len(rows)} rows of {N}): {rows_hash(rows)}", flush=True)
    token_bytes = get_token_bytes(device=device, tokenizer_dir=args.tokenizer_dir)
    result = {"edges": edges, "rows": len(rows), "prompt_len": args.prompt_len, "models": {}}
    sep_max = max([int(sp[2][3:].replace("full", "")) for sp in specs if len(sp) > 2 and sp[2].startswith("sep")]
                  or [0])
    slot_token = None
    if sep_max:
        from nanochat.lanes import LANE_TOKEN
        slot_token = tok.encode_special(LANE_TOKEN)
    P = args.sep_split
    within_doc = None
    if args.within_doc > 0:                             # rows whose split falls inside one document
        bos = tok.get_bos_token_id()
        w = args.within_doc
        within_doc = lambda x: ~(x[:, max(P - w, 0):P + w] == bos).any(1)
    if within_doc is not None:
        kept = int(sum(within_doc(r[0][None]).item() for r in rows))
        print(f"within-document rows: {kept} of {len(rows)} (no document start within {args.within_doc} of {P})")
        result["within_doc_rows"] = kept
    global args_last_ignored
    args_last_ignored = any(len(sp) > 2 and sp[2].startswith(("wb", "bl", "lo", "sd")) for sp in specs)
    lane_token = None
    if any(len(sp) > 2 and sp[2].startswith(("ln", "bo")) for sp in specs):
        from nanochat.lanes import LANE_TOKEN
        lane_token = tok.encode_special(LANE_TOKEN)
    wb_mask_token = None
    if args_last_ignored:
        from nanochat.lanes import LANE_TOKEN
        wb_mask_token = tok.encode_special(LANE_TOKEN)
    per_pos = {}
    for k, spec in enumerate(specs):
        name, ckdir = spec[0], spec[1]
        wbn = int(spec[2][2:]) if len(spec) > 2 and spec[2].startswith("wb") else 0
        bl = tuple(int(v) for v in spec[2][2:].split("_")) if len(spec) > 2 and spec[2].startswith("bl") else None
        ln = int(spec[2][2:]) if len(spec) > 2 and spec[2].startswith("ln") else 0
        bo = tuple(int(v) for v in spec[2][2:].split("_")) if len(spec) > 2 and spec[2].startswith("bo") else None
        spl = int(spec[2][2:]) if len(spec) > 2 and spec[2].startswith("sp") else 0
        lo = int(spec[2][2:]) if len(spec) > 2 and spec[2].startswith("lo") else 0
        sd = tuple(int(v) for v in spec[2][2:].split("_")) if len(spec) > 2 and spec[2].startswith("sd") else None
        sep = spec[2] if len(spec) > 2 and spec[2].startswith("sep") else ""
        window = int(spec[2]) if len(spec) > 2 and not sep and not wbn and not bl and not ln and not lo and not sd \
            and not spl and not bo else 0
        m = int(sep[3:].replace("full", "")) if sep else 0

        def prep(x, y, m=m):
            # Every model is scored on the same original targets: the last first-half prediction and
            # the final sep_max positions (which shifted layouts drop) are ignored for all.
            y = y.clone()
            y[:, P - 1] = -1
            y[:, y.size(1) - sep_max:] = -1
            if within_doc is not None:
                y[~within_doc(x)] = -1
            if not m:
                return x, y, 0
            xs, ys = separator_batch(x, y, P, m, slot_token)
            return xs, ys, (P, m)
        model = first[0] if k == 0 else build_model(ckdir, find_last_step(ckdir), device, phase="eval",
                                                     tokenizer_dir=args.tokenizer_dir)[0]
        model.eval()
        mask = prompt_window_mask(N, args.prompt_len, window, device) if window > 0 else None
        if m and not sep.endswith("full"):
            mask = separator_mask(N, P, m, device)
        wb = None
        if wbn:
            from nanochat.wbisect import wb_steps
            wb = (wb_steps(N, args.wb_prefix, wbn).to(device), wb_mask_token)
        if bl:
            from nanochat.wbisect import bridged_lanes_steps
            wb = (bridged_lanes_steps(N, args.wb_prefix, bl[0], bl[1]).to(device), wb_mask_token)
        if lo:
            from nanochat.wbisect import lane_order_steps
            wb = (lane_order_steps(N, args.wb_prefix, lo).to(device), wb_mask_token)
        if sd:
            from nanochat.wbisect import seeded_lanes_steps
            wb = (seeded_lanes_steps(N, args.wb_prefix, sd[0], sd[1] if len(sd) > 1 else 1).to(device), wb_mask_token)
        lanes = None
        if ln:
            from nanochat.lanes import lane_mask
            mask = lane_mask(N, args.wb_prefix, ln, device)
            lanes = (args.wb_prefix, ln, lane_token)
        xform = None
        if bo:
            from nanochat.lanes import bridged_slot_steps, step_inputs, step_mask
            ys = bridged_slot_steps(N, args.wb_prefix, bo[0], bo[1])
            mask = step_mask(ys, device)
            xform = lambda x, ys=ys: step_inputs(x, ys, lane_token)
        splice = None
        if spl:
            from nanochat.lanes import LANE_TOKEN
            splice = (args.wb_prefix, spl, tok.encode_special(LANE_TOKEN))
        bpb, pn, pb = bucket_bpb(model, rows, token_bytes, edges, mask, prep=prep if sep_max else None, wb=wb,
                                 lanes=lanes, splice=splice, xform=xform)
        per_pos[name] = (pn, pb)
        blk = lambda a: float(pn[a:].sum() / (pb[a:].sum() * math.log(2)))
        result["models"][name] = {"checkpoint": ckdir, "window": window, "separator": sep, "wb_window": wbn,
                                  "lanes": ln or spl, "splice": bool(spl), "bpb": bpb,
                                  "block_bpb_from_prefix_minus_1": blk(args.wb_prefix - 1),
                                  "block_bpb_from_prefix": blk(args.wb_prefix)}
        if spl:
            result["models"][name]["splice_offsets"] = splice_offset_report(
                model, rows, token_bytes, args.wb_prefix, spl, splice[2])
        del model
    names = list(result["models"])
    ref = result["models"][names[0]]["bpb"]
    head = " ".join(f"[{edges[j]},{edges[j + 1]})".rjust(14) for j in range(len(edges) - 1))
    print(f"bpb by input-position bucket over {len(rows)} rows (ratio to {names[0]}):\n{'':20s}{head}")
    for name in names:
        bpb = result["models"][name]["bpb"]
        print(f"{name:20s}" + " ".join((f"{b:.4f} ({b / r:.3f})" if r > 0 else "--").rjust(14)
                                       for b, r in zip(bpb, ref)))
    for name in names:
        r = result["models"][name]
        print(f"{name}: block bpb {r['block_bpb_from_prefix_minus_1']:.4f} (targets from position "
              f"{args.wb_prefix - 1}), {r['block_bpb_from_prefix']:.4f} (from {args.wb_prefix})")
        if r.get("splice_offsets"):
            print(f"{name}: splice nats per token by lane offset (lanes 1..L-1; codes separate): " +
                  ", ".join(f"{k} {v:.3f}" for k, v in r["splice_offsets"].items()))
    for name in names:
        ln = result["models"][name]["lanes"]
        if ln and len(names) > 1 and not result["models"][name]["splice"]:
            rep = lane_offset_report(N, args.wb_prefix, ln, per_pos[name], per_pos[names[0]],
                                     n_rows=result.get("within_doc_rows", len(rows)))
            result["models"][name]["lane_offsets"] = rep
            print(f"{name}: nats ratio to {names[0]} by offset within a lane (lanes 1..{ln - 1}): " +
                  ", ".join(f"{k} {v:.3f}" for k, v in rep.items() if not isinstance(v, list)))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
