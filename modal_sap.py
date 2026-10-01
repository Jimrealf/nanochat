"""
Modal launcher for the SAP experiments (sap_research_plan.md v3).

Stage A: every block-head mode on the synthetic phrase HMM, one container per run, so the
whole grid takes about as long as one run. Only metrics come back: each run scores itself
exactly against the generator, so no weights are kept. Rows also land on the volume under
out/sap_stageA_modal/.

Stage B: FineWeb-Edu at depth 8, one H100 container per arm, every arm at the dense arm's
training FLOPs. Checkpoints go to the `nanochat` volume under out/s00_sap/d<depth>/<tag>. The
decode-speed and generation evals run in a second pass, once the dense reference exists.

The `nanochat` volume already holds the FineWeb-Edu shards (data/) and the V=32,768 tokenizer
(tokenizer/). The code comes from this checkout, mounted into the image at container start, so
the volume's older copies of nanochat/ and scripts/ are never imported.

    export MODAL_PROFILE=blessingjim31
    modal run modal_sap.py::stage_a --smoke
    modal run modal_sap.py::stage_a                               # 9 modes x T in {2,4}, seed 0
    modal run modal_sap.py::stage_a --seeds 2 --modes indep,cp,p1_discrete
    modal run modal_sap.py::stage_b --smoke                       # depth 2 on an L4, checks paths
    modal run modal_sap.py::stage_b --arms indep,cp,local,p1_discrete --ts 2,4
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys

import modal

# The local entrypoints import scripts.* and nanochat.* from this checkout.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

VOLUME = modal.Volume.from_name("nanochat")
VOL = "/vol"            # data/ (FineWeb-Edu shards) and tokenizer/ live at the volume root
SRC = "/root/src"       # this checkout's nanochat/ and scripts/
WORK = "/root/work"     # cwd for every run: tokenizer/ and data/ symlinked to the volume

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("uv")
    .add_local_file("pyproject.toml", remote_path="/root/pyproject.toml", copy=True)
    .run_commands("uv pip install --system --compile-bytecode -r /root/pyproject.toml")
    .env({"PYTHONPATH": SRC, "OMP_NUM_THREADS": "8", "NANOCHAT_BASE_DIR": "/tmp/nanochat_base",
          "PYTHONUNBUFFERED": "1"})
    .add_local_dir("nanochat", remote_path=f"{SRC}/nanochat", ignore=["**/__pycache__", "**/*.pyc"])
    .add_local_dir("scripts", remote_path=f"{SRC}/scripts", ignore=["**/__pycache__", "**/*.pyc"])
)
app = modal.App("nanochat-sap", image=image)


def _workdir():
    """A cwd where the repo's default lookups ("tokenizer/tokenizer.pkl", "data") hit the volume."""
    os.makedirs(WORK, exist_ok=True)
    for name in ("tokenizer", "data"):
        link = os.path.join(WORK, name)
        if not os.path.exists(link):
            os.symlink(os.path.join(VOL, name), link)
    os.chdir(WORK)
    if SRC not in sys.path:
        sys.path.insert(0, SRC)


def _run_logged(cmd, log_path):
    """Run cmd, stream its output to the Modal log and to log_path. Returns (code, text)."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    lines = []
    with open(log_path, "w") as log:
        proc = subprocess.Popen(cmd, cwd=WORK, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1)
        for line in proc.stdout:
            print(line, end="")
            log.write(line)
            lines.append(line)
        proc.wait()
    return proc.returncode, "".join(lines)


# ----------------------------------------------------------------------------- Stage A
@app.function(gpu="L4", timeout=2 * 3600, volumes={VOL: VOLUME})
def stage_a_run(mode: str, T: int, seed: int, argv: list, tag: str = "") -> dict:
    _workdir()
    from scripts.sap_synthetic import build_parser, run_one
    args = build_parser().parse_args(list(argv) + ["--device", "cuda"])
    row = run_one(args, mode, T, seed)
    row["tag"], row["argv"] = tag, list(argv)
    out = f"{VOL}/out/sap_stageA_modal"
    os.makedirs(out, exist_ok=True)
    name = f"{mode}_T{T}_s{seed}_steps{args.steps}" + (f"_{tag}" if tag else "")
    with open(f"{out}/{name}.json", "w") as f:
        json.dump(row, f)
    VOLUME.commit()
    return row


@app.local_entrypoint()
def stage_a(smoke: bool = False, seeds: int = 1, steps: int = 2000, ts: str = "2,4",
            modes: str = "", extra: str = "", tag: str = "", out: str = "out/sap_synthetic_modal"):
    """extra: further sap_synthetic flags for every run, e.g. --extra "--latent-codes 64".
    tag: label for this variant; rows land in <out>_<tag>/ and carry it."""
    from scripts.sap_synthetic import gate_verdicts
    from nanochat.block_head import SAP_MODES
    mode_list = [m for m in modes.split(",") if m] or list(SAP_MODES)
    t_list = [int(t) for t in ts.split(",") if t]
    argv = ["--steps", str(steps)] + extra.split()
    if tag:
        out = f"{out}_{tag}"
    if smoke:
        mode_list, t_list, seeds = ["indep", "p1_discrete"], [2], 1
        argv = ["--steps", "200", "--eval-seqs", "32", "--iwae-samples", "8", "--log-every", "100"]
        out = out + "_smoke"
    grid = [(m, T, s, argv, tag) for T in t_list for s in range(seeds) for m in mode_list]
    print(f"Stage A on Modal: {len(grid)} runs in parallel ({', '.join(mode_list)}; T={t_list}; seeds={seeds})")
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "results.jsonl")
    rows = []
    for res in stage_a_run.starmap(grid, return_exceptions=True):
        if isinstance(res, BaseException):
            print(f"  run failed: {res!r}")
            continue
        rows.append(res)
        with open(path, "a") as f:
            f.write(json.dumps(res) + "\n")
        bk = "n/a" if res["block_kl"] is None else f"{res['block_kl']:.3f}"
        post = res.get("invalid_rate_posterior")
        print(f"  {res['mode']:12s} T={res['T']} s={res['seed']}: block_kl {bk} | tc {res['tc']:.3f} "
              f"| invalid {res['invalid_rate']:.3f}"
              + ("" if post is None else f" (posterior plan {post:.3f})")
              + f" | sens {res['sensitivity']} | {res['seconds']}s")
    print("\n".join(["", "Stage A gates", *gate_verdicts(rows)]))
    print(f"\n{len(rows)}/{len(grid)} runs; rows appended to {path}")


# ----------------------------------------------------------------------------- Stage B
def _parse_train_log(text):
    out = {}
    m = re.findall(r"Validation bpb: ([0-9.]+)", text)
    if m:
        out["val_bpb"] = float(m[-1])
    m = re.findall(r"SAP_EVAL_JSON (\{.*\})", text)
    if m:
        out["sap_eval"] = json.loads(m[-1])
    m = re.search(r"Estimated FLOPs per token \(total\):\s+([0-9.e+]+)", text)
    if m:
        out["flops_per_token"] = float(m.group(1))
    m = re.search(r"Total number of training tokens: ([0-9,]+)", text)
    if m:
        out["train_tokens"] = int(m.group(1).replace(",", ""))
    return out


@app.function(gpu="H100", timeout=8 * 3600, volumes={VOL: VOLUME})
def stage_b_train(tag: str, train_args: list, depth: int) -> dict:
    _workdir()
    ckdir = f"{VOL}/out/s00_sap/d{depth}"
    cmd = [sys.executable, "-m", "scripts.base_train", *train_args,
           "--checkpoints-dir", ckdir, "--model-tag", tag]
    code, text = _run_logged(cmd, f"{VOL}/out/s00_sap/logs/{tag}_d{depth}.log")
    VOLUME.commit()
    return {"tag": tag, "returncode": code, **_parse_train_log(text)}


@app.function(gpu="H100", timeout=3 * 3600, volumes={VOL: VOLUME})
def stage_b_post(tag: str, ref_tag: str, depth: int, gen_prefixes: int, gen_tokens: int,
                 bench_tokens: int) -> dict:
    _workdir()
    base = f"{VOL}/out/s00_sap"
    ckdir = f"{base}/d{depth}"
    dec_json = f"{base}/decode_{tag}_d{depth}.json"
    code_a, text_a = _run_logged(
        [sys.executable, "-m", "scripts.sap_decode_bench", "--checkpoint-dir", f"{ckdir}/{tag}",
         "--tokenizer-dir", f"{VOL}/tokenizer", "--gen-tokens", str(bench_tokens), "--out", dec_json],
        f"{base}/logs/decode_{tag}_d{depth}.log")
    code_b, text_b = _run_logged(
        [sys.executable, "-m", "scripts.sap_eval_generation", "--checkpoint-dir", f"{ckdir}/{tag}",
         "--reference-dir", f"{ckdir}/{ref_tag}", "--tokenizer-dir", f"{VOL}/tokenizer",
         "--data-dir", f"{VOL}/data", "--n-prefixes", str(gen_prefixes), "--gen-tokens", str(gen_tokens),
         "--out", f"{base}/gen_{tag}_d{depth}.jsonl"],
        f"{base}/logs/gen_{tag}_d{depth}.log")
    VOLUME.commit()
    out = {"tag": tag, "decode_rc": code_a, "gen_rc": code_b}
    if code_a == 0 and os.path.exists(dec_json):
        out["decode"] = json.load(open(dec_json))
    m = re.search(r"reference ppl\s+next-token\s+([0-9.]+) \| block\s+([0-9.]+)", text_b)
    if m:
        out["ref_ppl_ar"], out["ref_ppl_block"] = float(m.group(1)), float(m.group(2))
    m = re.search(r"distinct 3-grams\s+next-token ([0-9.]+) \| block ([0-9.]+)", text_b)
    if m:
        out["distinct3_ar"], out["distinct3_block"] = float(m.group(1)), float(m.group(2))
    return out


def _dense_flops(depth, seq_len, window, vocab=32768, ratio=10.5, round_to=262144):
    """The dense arm's total training FLOPs: its Chinchilla tokens x its FLOPs per token."""
    from scripts.code_head_budget import build_dense_meta
    model, _ = build_dense_meta(depth, vocab, 64, 128, 0, seq_len, window)
    sp = model.num_scaling_params()
    tokens = int(ratio * (sp["transformer_matrices"] + sp["lm_head"]))
    tokens = (tokens // round_to) * round_to
    flops_per_token, _, _ = model.estimate_flops()
    return float(flops_per_token) * tokens, tokens, flops_per_token


@app.local_entrypoint()
def stage_b(depth: int = 8, arms: str = "indep,cp,local,inv_head,p1_discrete,p2_gauss,p3_energy",
            ts: str = "2,4", seeds: int = 1, smoke: bool = False, post: bool = True,
            frac: float = 0.0625, cp_frac: float = 0.015625, gen_prefixes: int = 1024,
            gen_tokens: int = 128, bench_tokens: int = 256, out: str = "out/s00_sap_modal"):
    arm_list = [a for a in arms.split(",") if a]
    t_list = [int(t) for t in ts.split(",") if t]
    opts = {"max-seq-len": 2048, "window-pattern": "SSSL", "device-batch-size": 16,
            "total-batch-size": -1, "eval-tokens": 20 * 2 ** 20, "sap-eval-steps": 40, "log-every": 100}
    train_fn, post_fn = stage_b_train, stage_b_post
    if smoke:
        depth, arm_list, t_list, seeds = 2, ["p1_discrete"], [2], 1
        gen_prefixes, gen_tokens, bench_tokens = 8, 16, 16
        opts.update({"max-seq-len": 256, "window-pattern": "L", "device-batch-size": 4,
                     "total-batch-size": 1024, "eval-tokens": 16384, "sap-eval-steps": 2, "log-every": 10})
        flops = 1e12
        out = out + "_smoke"
        train_fn = stage_b_train.with_options(gpu="L4")
        post_fn = stage_b_post.with_options(gpu="L4")
        print("Stage B smoke: depth 2, sequence 256, ~35 steps, on an L4")
    else:
        flops, tokens, fpt = _dense_flops(depth, opts["max-seq-len"], opts["window-pattern"])
        print(f"Stage B: dense d{depth} {fpt:,.0f} FLOPs/token x {tokens:,} tokens = {flops:.4e} FLOPs per arm")
    common = [f"--depth", str(depth), "--target-flops", f"{flops:.6e}", "--target-param-data-ratio", "10.5",
              "--warmup-ratio", "0.005", "--warmdown-ratio", "0.65", "--final-lr-frac", "0.05",
              "--eval-every", "-1", "--core-metric-every", "0", "--sample-every", "-1", "--save-every", "-1",
              "--data-dir", f"{VOL}/data", "--tokenizer-dir", f"{VOL}/tokenizer"]
    for k, v in opts.items():
        common += [f"--{k}", str(v)]

    jobs = []
    for s in range(1, seeds + 1):
        jobs.append((f"B1_dense_s{s}", common + ["--seed", str(s)], depth))
        for T in t_list:
            for arm in arm_list:
                f_ = cp_frac if arm == "cp" else frac
                jobs.append((f"SAP_{arm}_T{T}_s{s}",
                             common + ["--seed", str(s), "--sap-block-t", str(T), "--sap-block-mode", arm,
                                       "--sap-block-frac", str(f_)], depth))
    print(f"launching {len(jobs)} training runs in parallel: {', '.join(j[0] for j in jobs)}")
    os.makedirs(out, exist_ok=True)
    results = {}
    for res in train_fn.starmap(jobs, return_exceptions=True):
        if isinstance(res, BaseException):
            print(f"  training failed: {res!r}")
            continue
        results[res["tag"]] = res
        se = res.get("sap_eval") or {}
        print(f"  {res['tag']:28s} rc={res['returncode']} val_bpb={res.get('val_bpb')} "
              f"block_bpb={se.get('block_bpb')} ntp_same={se.get('ntp_bpb_same_tokens')} "
              f"flops/tok={res.get('flops_per_token')}")
    if post:
        post_jobs = []
        for tag, res in results.items():
            seed = tag.rsplit("_s", 1)[1]
            ref = f"B1_dense_s{seed}"
            if tag.startswith("SAP_") and res["returncode"] == 0 and results.get(ref, {}).get("returncode") == 0:
                post_jobs.append((tag, ref, depth, gen_prefixes, gen_tokens, bench_tokens))
        print(f"post-run evals for {len(post_jobs)} arms")
        for res in post_fn.starmap(post_jobs, return_exceptions=True):
            if isinstance(res, BaseException):
                print(f"  post-run failed: {res!r}")
                continue
            results[res["tag"]]["post"] = res
            rows = (res.get("decode") or {}).get("rows", [])
            sp = ", ".join(f"b{r['batch']}: {r['speedup']:.2f}x" for r in rows)
            print(f"  {res['tag']:28s} decode speedup {sp} | ref ppl AR {res.get('ref_ppl_ar')} "
                  f"vs block {res.get('ref_ppl_block')}")
    path = os.path.join(out, f"summary_d{depth}.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsummary written to {path}; logs and checkpoints on the volume under out/s00_sap/")
