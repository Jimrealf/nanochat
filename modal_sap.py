"""
Modal launcher for the SAP experiments (sap_research_plan.md v3).

Stage A: every block-head mode on the synthetic phrase HMM, one container per run, so the
whole grid takes about as long as one run. Only metrics come back: each run scores itself
exactly against the generator, so no weights are kept. Rows also land on the volume under
out/sap_stageA_modal/.

Stage B: FineWeb-Edu at depth 8, one H100 container per arm, every arm at the dense arm's
training FLOPs. Checkpoints go to the `nanochat` volume under out/s00_sap/d<depth>/<tag>. The
decode-speed and generation evals run in a second pass, once the dense reference exists.

S01: the ten sampled-in-the-middle refinement mechanisms at T=L, fanned out over one H100
per arm. Checkpoints, per-job logs, post-evals, a JSON summary, and a compiled text log land
under out/s01_sap/ on the volume.

The `nanochat` volume already holds the FineWeb-Edu shards (data/) and the V=32,768 tokenizer
(tokenizer/). The code comes from this checkout, mounted into the image at container start, so
the volume's older copies of nanochat/ and scripts/ are never imported.

    export MODAL_PROFILE=blessingjim31
    modal run modal_sap.py::stage_a --smoke
    modal run modal_sap.py::stage_a                               # 9 modes x T in {2,4}, seed 0
    modal run modal_sap.py::stage_a --seeds 2 --modes indep,cp,p1_discrete
    modal run modal_sap.py::stage_b --smoke                       # depth 2 on an L4, checks paths
    modal run modal_sap.py::stage_b --arms indep,cp,local,p1_discrete --ts 2,4
    MODAL_PROFILE=blessingjim31-workspace modal run modal_sap.py::s01 --depth 8
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


def _workdir(tok: str = "tokenizer"):
    """A cwd where the repo's default lookups ("tokenizer/tokenizer.pkl", "data") hit the volume.

    tok: the volume directory that tokenizer/ points at. The S03 runs use tokenizer_sap, the
    pinned V=32,768 tokenizer B1_dense_s1 was trained with; a workspace's own tokenizer/ may be a
    different one (nanochat1's is), which silently scrambles every token id.
    """
    try:
        VOLUME.reload()
    except Exception as e:
        print(f"Notice: VOLUME.reload() skipped or failed: {e}")
    os.makedirs(WORK, exist_ok=True)
    for name, target in (("tokenizer", tok), ("data", "data")):
        link = os.path.join(WORK, name)
        if not os.path.exists(link):
            os.symlink(os.path.join(VOL, target), link)
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
@app.function(gpu="L4", timeout=8 * 3600, volumes={VOL: VOLUME})
def stage_a_run(mode: str, T: int, seed: int, argv: list, tag: str = "") -> dict:
    _workdir()
    from scripts.sap_synthetic import build_parser, run_one
    args = build_parser().parse_args(list(argv) + ["--device", "cuda"])
    out = f"{VOL}/out/sap_stageA_modal"
    os.makedirs(out, exist_ok=True)
    name = f"{mode}_T{T}_s{seed}_steps{args.steps}" + (f"_{tag}" if tag else "")
    if args.steps > 2000 and not args.resume_checkpoint:
        args.resume_checkpoint = f"{out}/checkpoints/{name}.pt"
        args.checkpoint_every = 2000
    row = run_one(args, mode, T, seed, checkpoint_commit=VOLUME.commit)
    row["tag"], row["argv"] = tag, list(argv)
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


@app.local_entrypoint()
def s02(audit_steps: int = 32000, tree_steps: int = 8000,
        freeze_after: int = 2000, out: str = "out/s02_sap_modal"):
    """Run the S02 convergence audit and correlated-tree controls concurrently.

    The two sir_anchor jobs differ only in width. The tree-l2 arm is the intended
    root/child-anchor + parallel-fill mechanism; tree-full is its exact tree-factorisation
    ceiling at T=4. Milestones are evaluated from one uninterrupted trajectory.
    """
    base = ["--T", "4", "--sir-posterior-mix", "0", "--sir-anchor-stride", "2",
            "--sir-fine-stride", "1", "--sir-train-samples", "4",
            "--sir-policy-weight", "1.0", "--freeze-trunk-after", str(freeze_after),
            "--eval-seqs", "128", "--samples-per-ctx", "4", "--iwae-samples", "32",
            "--log-every", "250"]
    jobs = []
    for width in (128, 256):
        argv = (["--steps", str(audit_steps), "--n-embd", str(width),
                 "--eval-milestones", "2000", "8000", str(audit_steps)] + base)
        jobs.append(("sir_anchor", 4, 0, argv, f"s02_audit_w{width}"))
    for levels, label in ((2, "l2"), (0, "full")):
        argv = (["--steps", str(tree_steps), "--n-embd", "128", "--head-layers", "3",
                 "--sir-tree-levels", str(levels), "--eval-milestones", "2000",
                 str(tree_steps)] + base)
        jobs.append(("sir_tree", 4, 0, argv, f"s02_tree_{label}"))

    print("S02 on Modal: four jobs in parallel (anchor widths 128/256; tree l2/full)")
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "results.jsonl")
    rows = []
    for res in stage_a_run.starmap(jobs, return_exceptions=True):
        if isinstance(res, BaseException):
            print(f"  run failed: {res!r}")
            continue
        rows.append(res)
        with open(path, "a") as f:
            f.write(json.dumps(res) + "\n")
        print(f"  {res['tag']:20s}: KL={res['block_kl']:.4f} "
              f"invalid={res['invalid_rate']:.4f} "
              f"oracle={res.get('invalid_rate_oracle_anchors')} {res['seconds']}s")
    summary = os.path.join(out, "summary.json")
    with open(summary, "w") as f:
        json.dump(rows, f, indent=2, sort_keys=True)
    print(f"\n{len(rows)}/{len(jobs)} S02 jobs completed; {summary}")


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
    target_dir = f"{ckdir}/{tag}"
    if os.path.isdir(target_dir):
        pts = [f for f in os.listdir(target_dir) if f.startswith("model_") and f.endswith(".pt")]
        if pts:
            print(f"Skipping {tag}: found existing checkpoint {pts[-1]} in {target_dir}")
            log_candidates = [
                f"{VOL}/out/s00_sap/logs/{tag}_d{depth}.log",
                f"{VOL}/out/s00_sap/{tag}_d{depth}.log",
            ]
            text = ""
            for lp in log_candidates:
                if os.path.exists(lp):
                    with open(lp) as f:
                        text = f.read()
                    break
            res = {"tag": tag, "returncode": 0, **_parse_train_log(text)}
            if "val_bpb" not in res:
                metas = [f for f in os.listdir(target_dir) if f.startswith("meta_") and f.endswith(".json")]
                if metas:
                    with open(os.path.join(target_dir, sorted(metas)[-1])) as f:
                        meta_data = json.load(f)
                    res["val_bpb"] = meta_data.get("val_bpb")
            return res

    cmd = [sys.executable, "-m", "scripts.base_train", *train_args,
           "--checkpoints-dir", ckdir, "--model-tag", tag]
    code, text = _run_logged(cmd, f"{VOL}/out/s00_sap/logs/{tag}_d{depth}.log")
    VOLUME.commit()
    return {"tag": tag, "returncode": code, **_parse_train_log(text)}


@app.function(timeout=10 * 60, volumes={VOL: VOLUME})
def stage_b_existing_reference(tag: str, depth: int) -> dict:
    """Load metadata for an existing dense reference without ever training one."""
    _workdir()
    ckdir = f"{VOL}/out/s00_sap/d{depth}/{tag}"
    pts = [] if not os.path.isdir(ckdir) else [
        f for f in os.listdir(ckdir) if f.startswith("model_") and f.endswith(".pt")]
    if not pts:
        raise FileNotFoundError(
            f"required existing reference checkpoint is missing: {ckdir}; refusing to retrain it")
    log_path = f"{VOL}/out/s00_sap/logs/{tag}_d{depth}.log"
    text = open(log_path).read() if os.path.exists(log_path) else ""
    res = {"tag": tag, "returncode": 0, "existing_reference": True,
           "checkpoint": os.path.join(ckdir, sorted(pts)[-1]), **_parse_train_log(text)}
    if "val_bpb" not in res:
        metas = [f for f in os.listdir(ckdir) if f.startswith("meta_") and f.endswith(".json")]
        if metas:
            with open(os.path.join(ckdir, sorted(metas)[-1])) as f:
                res["val_bpb"] = json.load(f).get("val_bpb")
    return res


@app.function(gpu="H100", timeout=3 * 3600, volumes={VOL: VOLUME})
def stage_b_post(tag: str, ref_tag: str, depth: int, gen_prefixes: int, gen_tokens: int,
                 bench_tokens: int, override_mode: str = "", jacobi_sweeps: int = None,
                 use_graphs: bool = False, eval_bpb: bool = True, skip_gen: bool = False,
                 skip_bpb: bool = False) -> dict:
    _workdir()
    base = f"{VOL}/out/s00_sap"
    ckdir = f"{base}/d{depth}"
    suffix = f"_{override_mode}" if override_mode else ""
    dec_json = f"{base}/decode_{tag}{suffix}_d{depth}.json"
    bpb_json = f"{base}/bpb_{tag}{suffix}_d{depth}.json"
    extra_args = []
    if override_mode:
        extra_args += ["--override-mode", override_mode]
    if jacobi_sweeps is not None:
        extra_args += ["--jacobi-sweeps", str(jacobi_sweeps)]
    bench_args = [sys.executable, "-m", "scripts.sap_decode_bench", "--checkpoint-dir",
                  f"{ckdir}/{tag}", "--tokenizer-dir", f"{VOL}/tokenizer",
                  "--gen-tokens", str(bench_tokens), "--out", dec_json]
    if not use_graphs:
        bench_args.append("--no-graphs")
    code_a, text_a = _run_logged(
        bench_args + extra_args,
        f"{base}/logs/decode_{tag}{suffix}_d{depth}.log")
    gen_log = f"{base}/logs/gen_{tag}{suffix}_d{depth}.log"
    if skip_gen and os.path.exists(gen_log):
        code_b = 0
        with open(gen_log, "r", errors="ignore") as f:
            text_b = f.read()
    else:
        code_b, text_b = _run_logged(
            [sys.executable, "-m", "scripts.sap_eval_generation", "--checkpoint-dir", f"{ckdir}/{tag}",
             "--reference-dir", f"{ckdir}/{ref_tag}", "--tokenizer-dir", f"{VOL}/tokenizer",
             "--data-dir", f"{VOL}/data", "--n-prefixes", str(gen_prefixes), "--gen-tokens", str(gen_tokens),
             "--out", f"{base}/gen_{tag}{suffix}_d{depth}.jsonl"] + extra_args,
            gen_log)
    out = {"tag": tag, "decode_rc": code_a, "gen_rc": code_b}
    if eval_bpb:
        if skip_bpb and os.path.exists(bpb_json):
            code_c = 0
            text_c = ""
        else:
            code_c, text_c = _run_logged(
                [sys.executable, "-m", "scripts.sap_eval_bpb", "--checkpoint-dir", f"{ckdir}/{tag}",
                 "--tokenizer-dir", f"{VOL}/tokenizer", "--data-dir", f"{VOL}/data",
                 "--eval-tokens", "1048576", "--out", bpb_json],
                f"{base}/logs/bpb_{tag}{suffix}_d{depth}.log")
        out["bpb_rc"] = code_c
        if code_c == 0 and os.path.exists(bpb_json):
            with open(bpb_json) as f:
                out["bpb_eval"] = json.load(f)
    VOLUME.commit()
    if code_a == 0 and os.path.exists(dec_json):
        out["decode"] = json.load(open(dec_json))
    else:
        out["decode_error"] = f"returncode={code_a}, log={base}/logs/decode_{tag}{suffix}_d{depth}.log"
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
def stage_b(depth: int = 8, arms: str = "local",
            ts: str = "8,L", seeds: int = 1, smoke: bool = False, post: bool = True,
            frac: float = 0.0625, cp_frac: float = 0.015625, gen_prefixes: int = 1024,
            gen_tokens: int = 128, bench_tokens: int = 256, out: str = "out/s00_sap_modal"):
    arm_list = [a for a in arms.split(",") if a]
    t_raw = [t.strip() for t in ts.split(",") if t.strip()]
    opts = {"max-seq-len": 2048, "window-pattern": "SSSL", "device-batch-size": 16,
            "total-batch-size": -1, "eval-tokens": 20 * 2 ** 20, "sap-eval-steps": 40, "log-every": 100}
    train_fn, post_fn = stage_b_train, stage_b_post
    if smoke:
        depth, arm_list, t_raw, seeds = 2, ["local"], ["2"], 1
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
        for t_spec in t_raw:
            if t_spec.upper() == "L":
                T = opts["max-seq-len"]
                t_label = "TL"
                f_ = 1.0 / T
            else:
                T = int(t_spec)
                t_label = f"T{T}"
                f_ = 1.0 / T if T >= opts["max-seq-len"] else frac
            for arm in arm_list:
                f_arm = cp_frac if arm == "cp" else f_
                arm_args = ["--seed", str(s), "--sap-block-t", str(T), "--sap-block-mode", arm,
                            "--sap-block-frac", str(f_arm)]
                if arm == "p1_discrete":
                    arm_args += ["--sap-latent-codes", "64"]
                jobs.append((f"SAP_{arm}_{t_label}_s{s}", common + arm_args, depth))
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


# ----------------------------------------------------------------------------- S02 depth-8 exception
@app.function(timeout=30 * 60, volumes={VOL: VOLUME})
def s02_d8_compile_logs(depth: int, results: dict) -> str:
    """Compile the user-authorised correlated-tree Stage-B run into one durable log."""
    _workdir()
    source = f"{VOL}/out/s00_sap/logs"
    base = f"{VOL}/out/s02_sap_d8"
    out_path = f"{base}/s02_sap_d{depth}_compiled.log"
    os.makedirs(base, exist_ok=True)
    with open(out_path, "w") as out:
        out.write("S02 SAP correlated-tree depth-8 exception\n")
        out.write("Arms: shallow tree (2 anchor levels + parallel fill), full balanced tree\n")
        out.write("Budget: each arm receives the dense depth-8 training-FLOP budget\n\n")
        out.write("COMPILED_RESULT_JSON\n")
        out.write(json.dumps(results, indent=2, sort_keys=True, default=str))
        out.write("\n\nPER-JOB LOGS\n")
        names = []
        for tag in sorted(results):
            names.extend((f"{tag}_d{depth}.log", f"decode_{tag}_d{depth}.log",
                          f"gen_{tag}_d{depth}.log"))
        for name in names:
            path = os.path.join(source, name)
            if not os.path.exists(path):
                continue
            out.write(f"\n{'=' * 88}\n{name}\n{'=' * 88}\n")
            with open(path, errors="replace") as src:
                while True:
                    chunk = src.read(1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
    with open(f"{base}/summary_d{depth}.json", "w") as f:
        json.dump(results, f, indent=2, sort_keys=True, default=str)
    VOLUME.commit()
    return out_path


@app.local_entrypoint()
def s02_d8(depth: int = 8, seeds: int = 1, post: bool = True, smoke: bool = False,
           gen_prefixes: int = 1024, gen_tokens: int = 128, bench_tokens: int = 256,
           out: str = "out/s02_sap_d8_modal"):
    """Cost-matched depth-8 T=L run for shallow and full correlated tree factorisations."""
    opts = {"max-seq-len": 2048, "window-pattern": "SSSL", "device-batch-size": 16,
            "total-batch-size": -1, "eval-tokens": 20 * 2 ** 20,
            "sap-eval-steps": 40, "log-every": 100}
    train_fn, post_fn = stage_b_train, stage_b_post
    if smoke:
        depth = 2
        gen_prefixes, gen_tokens, bench_tokens = 8, 16, 16
        opts.update({"max-seq-len": 64, "window-pattern": "L", "device-batch-size": 2,
                     "total-batch-size": 128, "eval-tokens": 2048,
                     "sap-eval-steps": 1, "log-every": 1})
        flops = 2e11
        out += "_smoke"
        train_fn = stage_b_train.with_options(gpu="L4", timeout=3600)
        post_fn = stage_b_post.with_options(gpu="L4", timeout=3600)
        print("S02 tree smoke: d2, T=L=64, shallow/full on L4s")
    else:
        flops, tokens, fpt = _dense_flops(depth, opts["max-seq-len"], opts["window-pattern"])
        print(f"S02 d8 trees: dense d{depth} {fpt:,.0f} FLOPs/token x {tokens:,} tokens "
              f"= {flops:.4e} FLOPs per arm")

    common = ["--depth", str(depth), "--target-flops", f"{flops:.6e}",
              "--target-param-data-ratio", "10.5", "--warmup-ratio", "0.005",
              "--warmdown-ratio", "0.65", "--final-lr-frac", "0.05",
              "--eval-every", "-1", "--core-metric-every", "0", "--sample-every", "-1",
              "--save-every", "-1", "--data-dir", f"{VOL}/data",
              "--tokenizer-dir", f"{VOL}/tokenizer"]
    for key, value in opts.items():
        common += [f"--{key}", str(value)]

    T = opts["max-seq-len"]
    tree_common = ["--sap-block-t", str(T), "--sap-block-mode", "sir_tree",
                   "--sap-block-frac", str(1.0 / T), "--sap-head-layers", "3",
                   "--sap-sir-topk", "16", "--sap-sir-posterior-mix", "0"]
    jobs, refs, results = [], {}, {}
    for seed in range(1, seeds + 1):
        # Full runs require the already-trained S00 reference. This is a read-only lookup:
        # a missing checkpoint is an error and never falls through to dense training.
        ref = f"B1_dense_s{seed}"
        refs[str(seed)] = ref
        if not smoke:
            results[ref] = stage_b_existing_reference.remote(ref, depth)
        for levels, label in ((2, "l2"), (0, "full")):
            tag = f"S02_sir_tree_{label}_TL_s{seed}"
            args = common + ["--seed", str(seed), "--sap-sir-tree-levels", str(levels)] + tree_common
            jobs.append((tag, args, depth))

    print(f"launching {len(jobs)} cost-matched jobs concurrently: {', '.join(j[0] for j in jobs)}")
    for res in train_fn.starmap(jobs, return_exceptions=True):
        if isinstance(res, BaseException):
            print(f"  training failed: {res!r}")
            continue
        results[res["tag"]] = res
        se = res.get("sap_eval") or {}
        print(f"  {res['tag']:32s} rc={res['returncode']} val_bpb={res.get('val_bpb')} "
              f"block_bpb={se.get('block_bpb')} ntp_same={se.get('ntp_bpb_same_tokens')} "
              f"tokens={res.get('train_tokens')} flops/tok={res.get('flops_per_token')}")

    if post:
        post_jobs = []
        for tag, res in results.items():
            if not tag.startswith("S02_") or res.get("returncode") != 0:
                continue
            seed = tag.rsplit("_s", 1)[1]
            # A smoke scores each random tiny model against itself; the full run uses the
            # read-only dense checkpoint loaded above.
            ref = tag if smoke else refs[seed]
            if smoke or results.get(ref, {}).get("returncode") == 0:
                # The final True enables CUDA-graph timing in addition to eager timing.
                post_jobs.append((tag, ref, depth, gen_prefixes, gen_tokens, bench_tokens,
                                  "", None, True))
        print(f"launching {len(post_jobs)} post-evals concurrently (eager + CUDA graphs)")
        for res in post_fn.starmap(post_jobs, return_exceptions=True):
            if isinstance(res, BaseException):
                print(f"  post-run failed: {res!r}")
                continue
            results[res["tag"]]["post"] = res
            rows = (res.get("decode") or {}).get("rows", [])
            speeds = ", ".join(
                f"b{r['batch']}={r['speedup']:.2f}x"
                + (f"/graph {r['graph_speedup']:.2f}x" if "graph_speedup" in r else "")
                for r in rows)
            print(f"  {res['tag']:32s} {speeds}; ref-PPL AR={res.get('ref_ppl_ar')} "
                  f"block={res.get('ref_ppl_block')}")

    compiled = s02_d8_compile_logs.remote(depth, results)
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, f"summary_d{depth}.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2, sort_keys=True, default=str)
    print(f"summary: {path}")
    print(f"compiled volume log: {compiled}")


# ----------------------------------------------------------------------------- S01
S01_MODES = ("sir", "sir_soft", "sir_compat", "sir_context", "sir_conf", "sir_anchor",
             "sir_pyramid", "sir_lattice", "sir_energy", "sir_full")


@app.function(gpu="H100", timeout=8 * 3600, volumes={VOL: VOLUME})
def s01_train(tag: str, train_args: list, depth: int, force: bool = False) -> dict:
    """Train one S01 arm into its own namespace on the shared volume."""
    _workdir()
    base = f"{VOL}/out/s01_sap"
    ckdir = f"{base}/d{depth}"
    target_dir = f"{ckdir}/{tag}"
    log_path = f"{base}/logs/{tag}_d{depth}.log"
    if not force and os.path.isdir(target_dir):
        pts = [f for f in os.listdir(target_dir) if f.startswith("model_") and f.endswith(".pt")]
        if pts:
            text = open(log_path).read() if os.path.exists(log_path) else ""
            print(f"Skipping {tag}: found {sorted(pts)[-1]} in {target_dir}")
            return {"tag": tag, "returncode": 0, "skipped": True, "log_path": log_path,
                    **_parse_train_log(text)}
    cmd = [sys.executable, "-m", "scripts.base_train", *train_args,
           "--checkpoints-dir", ckdir, "--model-tag", tag]
    code, text = _run_logged(cmd, log_path)
    VOLUME.commit()
    return {"tag": tag, "returncode": code, "skipped": False, "log_path": log_path,
            **_parse_train_log(text)}


@app.function(gpu="H100", timeout=3 * 3600, volumes={VOL: VOLUME})
def s01_post(tag: str, ref_tag: str, depth: int, gen_prefixes: int,
             gen_tokens: int, bench_tokens: int) -> dict:
    """Decode-speed and reference-PPL checks for one trained S01 arm."""
    _workdir()
    base = f"{VOL}/out/s01_sap"
    ckdir = f"{base}/d{depth}"
    dec_json = f"{base}/decode_{tag}_d{depth}.json"
    code_a, text_a = _run_logged(
        [sys.executable, "-m", "scripts.sap_decode_bench", "--checkpoint-dir", f"{ckdir}/{tag}",
         "--tokenizer-dir", f"{VOL}/tokenizer", "--gen-tokens", str(bench_tokens),
         "--no-graphs", "--out", dec_json],
        f"{base}/logs/decode_{tag}_d{depth}.log")
    code_b, text_b = _run_logged(
        [sys.executable, "-m", "scripts.sap_eval_generation", "--checkpoint-dir", f"{ckdir}/{tag}",
         "--reference-dir", f"{ckdir}/{ref_tag}", "--tokenizer-dir", f"{VOL}/tokenizer",
         "--data-dir", f"{VOL}/data", "--n-prefixes", str(gen_prefixes),
         "--gen-tokens", str(gen_tokens), "--out", f"{base}/gen_{tag}_d{depth}.jsonl"],
        f"{base}/logs/gen_{tag}_d{depth}.log")
    VOLUME.commit()
    out = {"tag": tag, "decode_rc": code_a, "gen_rc": code_b}
    if code_a == 0 and os.path.exists(dec_json):
        with open(dec_json) as f:
            out["decode"] = json.load(f)
    m = re.search(r"reference ppl\s+next-token\s+([0-9.]+) \| block\s+([0-9.]+)", text_b)
    if m:
        out["ref_ppl_ar"], out["ref_ppl_block"] = float(m.group(1)), float(m.group(2))
    m = re.search(r"distinct 3-grams\s+next-token ([0-9.]+) \| block ([0-9.]+)", text_b)
    if m:
        out["distinct3_ar"], out["distinct3_block"] = float(m.group(1)), float(m.group(2))
    return out


@app.function(timeout=30 * 60, volumes={VOL: VOLUME})
def s01_compile_logs(depth: int, results: dict) -> str:
    """Build one durable, downloadable log containing the verdict and every job log."""
    _workdir()
    base = f"{VOL}/out/s01_sap"
    out_path = f"{base}/s01_sap_d{depth}_compiled.log"
    os.makedirs(base, exist_ok=True)
    with open(out_path, "w") as out:
        out.write("S01 SAP sampled-in-the-middle refinement sweep\n")
        out.write(f"depth={depth}; mechanisms={','.join(S01_MODES)}\n\n")
        out.write("COMPILED_RESULT_JSON\n")
        out.write(json.dumps(results, indent=2, sort_keys=True, default=str))
        out.write("\n\nPER-JOB LOGS\n")
        log_dir = f"{base}/logs"
        if os.path.isdir(log_dir):
            for name in sorted(os.listdir(log_dir)):
                if not name.endswith(f"_d{depth}.log"):
                    continue
                out.write(f"\n{'=' * 88}\n{name}\n{'=' * 88}\n")
                with open(os.path.join(log_dir, name), errors="replace") as src:
                    while True:
                        chunk = src.read(1024 * 1024)
                        if not chunk:
                            break
                        out.write(chunk)
    with open(f"{base}/summary_d{depth}.json", "w") as f:
        json.dump(results, f, indent=2, sort_keys=True, default=str)
    VOLUME.commit()
    return out_path


@app.local_entrypoint()
def s01(depth: int = 8, arms: str = ",".join(S01_MODES), seeds: int = 1,
        post: bool = True, force: bool = False, smoke: bool = False,
        gen_prefixes: int = 1024, gen_tokens: int = 128, bench_tokens: int = 256,
        topk: int = 16, rank: int = 32, anchor_stride: int = 16,
        fine_stride: int = 4, refine_frac: float = 0.25,
        train_samples: int = 4, policy_weight: float = 1.0, posterior_mix: float = 0.5,
        draft_weight: float = 1.0, context_weight: float = 0.25,
        energy_weight: float = 0.25, out: str = "out/s01_sap_modal"):
    """Run the S01 T=L mechanisms concurrently, followed by concurrent post-evaluation."""
    arm_list = [a.strip() for a in arms.split(",") if a.strip()]
    unknown = sorted(set(arm_list) - set(S01_MODES))
    if unknown:
        raise ValueError(f"unknown S01 arms: {unknown}; choices are {S01_MODES}")
    opts = {"max-seq-len": 2048, "window-pattern": "SSSL", "device-batch-size": 16,
            "total-batch-size": -1, "eval-tokens": 20 * 2 ** 20,
            "sap-eval-steps": 40, "log-every": 100}
    train_fn, post_fn = s01_train, s01_post
    if smoke:
        depth, arm_list, seeds = 2, ["sir", "sir_full"], 1
        gen_prefixes, gen_tokens, bench_tokens = 8, 16, 16
        opts.update({"max-seq-len": 64, "window-pattern": "L", "device-batch-size": 2,
                     "total-batch-size": 128, "eval-tokens": 2048,
                     "sap-eval-steps": 1, "log-every": 1})
        flops = 2e11
        out += "_smoke"
        train_fn = s01_train.with_options(gpu="L4", timeout=3600)
        post_fn = s01_post.with_options(gpu="L4", timeout=3600)
        print("S01 Modal smoke: sir + sir_full, d2, T=L=64 on L4s")
    else:
        flops, tokens, fpt = _dense_flops(depth, opts["max-seq-len"], opts["window-pattern"])
        print(f"S01: dense d{depth} {fpt:,.0f} FLOPs/token x {tokens:,} tokens = {flops:.4e} per arm")
    common = ["--depth", str(depth), "--target-flops", f"{flops:.6e}",
              "--target-param-data-ratio", "10.5", "--warmup-ratio", "0.005",
              "--warmdown-ratio", "0.65", "--final-lr-frac", "0.05",
              "--eval-every", "-1", "--core-metric-every", "0", "--sample-every", "-1",
              "--save-every", "-1", "--data-dir", f"{VOL}/data",
              "--tokenizer-dir", f"{VOL}/tokenizer"]
    for k, v in opts.items():
        common += [f"--{k}", str(v)]
    T = opts["max-seq-len"]
    sir_common = ["--sap-block-t", str(T), "--sap-block-frac", str(1.0 / T),
                  "--sap-sir-topk", str(topk), "--sap-sir-rank", str(rank),
                  "--sap-sir-anchor-stride", str(anchor_stride),
                  "--sap-sir-fine-stride", str(fine_stride),
                  "--sap-sir-refine-frac", str(refine_frac),
                  "--sap-sir-train-samples", str(train_samples),
                  "--sap-sir-policy-weight", str(policy_weight),
                  "--sap-sir-posterior-mix", str(posterior_mix),
                  "--sap-sir-draft-weight", str(draft_weight),
                  "--sap-sir-context-weight", str(context_weight),
                  "--sap-sir-energy-weight", str(energy_weight)]
    jobs = []
    for seed in range(1, seeds + 1):
        jobs.append((f"B1_dense_s{seed}", common + ["--seed", str(seed)], depth, force))
        for arm in arm_list:
            layers = 3 if arm in ("sir_conf", "sir_pyramid", "sir_full") else 2
            args = common + ["--seed", str(seed), "--sap-block-mode", arm,
                             "--sap-head-layers", str(layers)] + sir_common
            jobs.append((f"SAP_{arm}_TL_s{seed}", args, depth, force))
    print(f"launching {len(jobs)} cost-matched jobs concurrently: {', '.join(j[0] for j in jobs)}")
    results = {}
    for res in train_fn.starmap(jobs, return_exceptions=True):
        if isinstance(res, BaseException):
            print(f"  training failed: {res!r}")
            continue
        results[res["tag"]] = res
        se = res.get("sap_eval") or {}
        print(f"  {res['tag']:28s} rc={res['returncode']} val_bpb={res.get('val_bpb')} "
              f"block_bpb={se.get('block_bpb')} ntp_same={se.get('ntp_bpb_same_tokens')}")
    if post:
        jobs = []
        for tag, res in results.items():
            if not tag.startswith("SAP_") or res.get("returncode") != 0:
                continue
            seed = tag.rsplit("_s", 1)[1]
            ref = f"B1_dense_s{seed}"
            if results.get(ref, {}).get("returncode") == 0:
                jobs.append((tag, ref, depth, gen_prefixes, gen_tokens, bench_tokens))
        print(f"launching {len(jobs)} post-evals concurrently")
        for res in post_fn.starmap(jobs, return_exceptions=True):
            if isinstance(res, BaseException):
                print(f"  post-run failed: {res!r}")
                continue
            results[res["tag"]]["post"] = res
            rows = (res.get("decode") or {}).get("rows", [])
            speeds = ", ".join(f"b{r['batch']}={r['speedup']:.2f}x" for r in rows)
            print(f"  {res['tag']:28s} {speeds}; ref-PPL AR={res.get('ref_ppl_ar')} "
                  f"block={res.get('ref_ppl_block')}")
    compiled = s01_compile_logs.remote(depth, results)
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, f"summary_d{depth}.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2, sort_keys=True, default=str)
    print(f"summary: {path}")
    print(f"compiled volume log: {compiled}")


@app.local_entrypoint()
def run_post(tag: str, ref: str = "B1_dense_s1", depth: int = 8,
             gen_prefixes: int = 1024, gen_tokens: int = 128, bench_tokens: int = 256,
             override_mode: str = "", jacobi_sweeps: int = None, use_graphs: bool = False,
             skip_gen: bool = False, skip_bpb: bool = False):
    """Run decode benchmarks, generation evaluation, and block BPB on an existing checkpoint on the volume.
    Example:
        modal run modal_sap.py::run_post --tag S02_sir_tree_full_TL_s1
    """
    label = f"{tag} (override_mode={override_mode}, sweeps={jacobi_sweeps})" if override_mode else tag
    print(f"Running post-eval on {label} against reference {ref} (depth {depth})...")
    res = stage_b_post.remote(tag, ref, depth, gen_prefixes, gen_tokens, bench_tokens,
                              override_mode=override_mode, jacobi_sweeps=jacobi_sweeps,
                              use_graphs=use_graphs, skip_gen=skip_gen, skip_bpb=skip_bpb)
    rows = (res.get("decode") or {}).get("rows", [])
    sp = ", ".join(
        f"b{r['batch']}: eager={r.get('speedup', 0):.2f}x"
        + (f", graph={r['graph_speedup']:.2f}x" if "graph_speedup" in r else "")
        for r in rows)
    print(f"\nResults for {tag}:")
    print(f"  Decode speedup: {sp or res.get('decode_error', 'N/A')}")
    print(f"  Ref PPL AR:     {res.get('ref_ppl_ar')}")
    print(f"  Ref PPL Block:  {res.get('ref_ppl_block')}")
    print(f"  Distinct 3 AR:  {res.get('distinct3_ar')}")
    print(f"  Distinct 3 Blk: {res.get('distinct3_block')}")
    if "bpb_eval" in res:
        bpb = res["bpb_eval"]
        print(f"  Block BPB:      {bpb.get('block_bpb')}")
        print(f"  NTP BPB:        {bpb.get('ntp_bpb_same_tokens')}")
        print(f"  Blocks scored:  {bpb.get('blocks')}")
    return res


@app.local_entrypoint()
def run_s02_d8_post(depth: int = 8, bench_tokens: int = 256,
                    gen_prefixes: int = 1024, gen_tokens: int = 128,
                    use_graphs: bool = False, skip_gen: bool = False,
                    skip_bpb: bool = False):
    """Rerun post-evaluation (decode bench + generation + BPB) on both trained depth-8 tree arms concurrently."""
    jobs = [
        ("S02_sir_tree_full_TL_s1", "B1_dense_s1", depth, gen_prefixes, gen_tokens, bench_tokens, "", None, use_graphs, True, skip_gen, skip_bpb),
        ("S02_sir_tree_l2_TL_s1", "B1_dense_s1", depth, gen_prefixes, gen_tokens, bench_tokens, "", None, use_graphs, True, skip_gen, skip_bpb),
    ]
    print(f"Launching post-evaluation on {len(jobs)} arms concurrently on H100...")
    for res in stage_b_post.starmap(jobs):
        tag = res["tag"]
        rows = (res.get("decode") or {}).get("rows", [])
        sp = ", ".join(
            f"b{r['batch']}: eager={r.get('speedup', 0):.2f}x"
            + (f", graph={r['graph_speedup']:.2f}x" if "graph_speedup" in r else "")
            for r in rows)
        print(f"\nResults for {tag}:")
        print(f"  Decode speedup: {sp or res.get('decode_error', 'N/A')}")
        print(f"  Ref PPL AR:     {res.get('ref_ppl_ar')}")
        print(f"  Ref PPL Block:  {res.get('ref_ppl_block')}")
        print(f"  Distinct 3 AR:  {res.get('distinct3_ar')}")
        print(f"  Distinct 3 Blk: {res.get('distinct3_block')}")
        if "bpb_eval" in res:
            bpb = res["bpb_eval"]
            print(f"  Block BPB:      {bpb.get('block_bpb')}")
            print(f"  NTP BPB:        {bpb.get('ntp_bpb_same_tokens')}")
            print(f"  Blocks scored:  {bpb.get('blocks')}")



# ----------------------------------------------------------------------------- S03: SAP v4
# Exact sampling cuts (sap_research_plan.md v4). Four stages, each its own entrypoint so a
# later one can be launched after reading the earlier one's numbers:
#
#   modal run modal_sap.py::s03_stage0                 # corpus tables, then the oracles
#   modal run modal_sap.py::s03_stage_a                # synthetic gate, every mechanism x grad
#   modal run modal_sap.py::s03_stage2 --modes cut_crf,lat_crf,...   # d8 heads on the frozen dense trunk
#   modal run modal_sap.py::s03_stage3 --modes cut_crf,lat_tt        # d8 from scratch, grad 0 / 0.1 / 1
#
# Everything lands under out/s03_sap/ on the volume. The dense reference is the existing
# out/s00_sap/d8/B1_dense_s1 and is never retrained. Post-evals time decode with CUDA graphs on
# (AR and block alike) at 32*T and at 256 generated tokens.
S03 = f"{VOL}/out/s03_sap"
S03_DENSE = f"{VOL}/out/s00_sap/d8/B1_dense_s1"     # d8 reference; exists on nanochat1 only
S03_TABLES = f"{S03}/tables_V32k.pt"
S03_TOK_NAME = "tokenizer_sap"       # pinned V=32,768 tokenizer (sha256 06978be3...) that B1_dense_s1 used
S03_TOK = f"{VOL}/{S03_TOK_NAME}"
S03_V4 = ["lat_crf", "lat_tt", "lat_cp", "cut_crf", "cut_tt", "pmi_chain", "corpus_code", "p1_selfpost"]


@app.function(gpu="H100", timeout=4 * 3600, volumes={VOL: VOLUME}, memory=65536)
def s03_job(name: str, cmd: list) -> dict:
    """Run one module (tables, oracle, eval) inside the work dir and log it to the volume."""
    _workdir(S03_TOK_NAME)
    code, text = _run_logged([sys.executable, "-m", *cmd], f"{S03}/logs/{name}.log")
    VOLUME.commit()
    return {"name": name, "returncode": code, "tail": text[-3000:]}


@app.local_entrypoint()
def s03_stage0(tokens: int = 300_000_000, contexts: int = 512, samples: int = 256,
               skip_tables: bool = False, smoke: bool = False):
    """Corpus tables from the training shards, then the Stage 0 oracles on the dense d8 model."""
    tables = S03_TABLES.replace(".pt", "_smoke.pt") if smoke else S03_TABLES
    tab = ["scripts.sap_corpus_tables", "--data-dir", f"{VOL}/data", "--tokenizer-dir", S03_TOK,
           "--tokens", str(tokens if not smoke else 1_000_000), "--out", tables]
    if not skip_tables:
        res = s03_job.remote("tables" + ("_smoke" if smoke else ""), tab)
        print(res["tail"])
        if res["returncode"] != 0:
            raise SystemExit("table build failed; oracles not launched")
    ora = ["scripts.sap_oracle", "--checkpoint-dir", S03_DENSE, "--tokenizer-dir", S03_TOK,
           "--data-dir", f"{VOL}/data", "--tables", tables,
           "--contexts", str(16 if smoke else contexts), "--pair-contexts", str(8 if smoke else 256),
           "--samples", str(8 if smoke else samples), "--out", f"{S03}/oracle_d8{'_smoke' if smoke else ''}.json"]
    res = s03_job.remote("oracle" + ("_smoke" if smoke else ""), ora)
    print(res["tail"])


@app.local_entrypoint()
def s03_stage_a(steps: int = 8000, seeds: int = 2, grads: str = "0,1", ts: str = "4",
                modes: str = "", smoke: bool = False, rerun: str = "", out: str = "out/s03_sap_stageA"):
    """Stage A synthetic gate: every v4 head, its component variants, and the indep / local /
    full-tree references, at each trunk gradient. Gate: block KL <= 0.10, invalid <= 3%."""
    from scripts.sap_synthetic import gate_verdicts
    specs = [m for m in modes.split(",") if m] or S03_V4 + ["indep", "local", "sir_tree"]
    variants = []
    for spec in specs:
        m, v = _s03_split(spec)
        variants.append((m, list(_S03_VARIANTS_A[v]) if v else [], v))
    if not modes:
        variants += [("lat_crf", ["--nce-props", "64"], "nce"), ("cut_crf", ["--nce-props", "64"], "nce"),
                     ("lat_crf", ["--supp-min-count", "1"], "supp"), ("cut_crf", ["--supp-min-count", "1"], "supp"),
                     ("lat_crf", ["--soft-eps", "0.1"], "soft")]
    per_mode = {"p1_selfpost": ["--latent-codes", "64"],
                "sir_tree": ["--head-layers", "3", "--sir-topk", "16", "--sir-posterior-mix", "0"]}
    base = ["--steps", str(steps), "--eval-milestones", str(min(2000, steps)), str(steps)]
    if smoke:
        base = ["--steps", "60", "--eval-seqs", "16", "--iwae-samples", "4", "--log-every", "30",
                "--table-seqs", "2000"]
        seeds, grads = 1, "0"
        variants = variants[:3]
        out += "_smoke"
    grid = []
    for g in [float(x) for x in grads.split(",") if x]:
        for T in [int(t) for t in ts.split(",") if t]:
            for s in range(seeds):
                for m, extra, vtag in variants:
                    # rerun: stage_a_run resumes any finished checkpoint with the same name, so a
                    # re-measurement after a code fix must carry its own tag (e.g. --rerun fix1).
                    tag = f"s03_g{g:g}" + (f"_{vtag}" if vtag else "") + (f"_{rerun}" if rerun else "")
                    grid.append((m, T, s, base + ["--trunk-grad", str(g)] + per_mode.get(m, []) + extra, tag))
    print(f"S03 Stage A: {len(grid)} runs in parallel")
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
        bk = "n/a" if res["block_kl"] is None else f"{res['block_kl']:.4f}"
        print(f"  {res['mode']:12s} {res['tag']:14s} T={res['T']} s={res['seed']}: block_kl {bk} "
              f"| invalid {res['invalid_rate']:.3f} | trunk excess {res['ntp_excess']:.4f} "
              f"| oracle anchors {res.get('invalid_rate_oracle_anchors')} | nce est {res.get('block_kl_nce_est')}")
    by_tag = {}
    for r in rows:
        by_tag.setdefault(r["tag"], []).append(r)
    for tag, rs in sorted(by_tag.items()):
        print(f"\nStage A gates [{tag}]")
        print("\n".join(gate_verdicts(rs)))
    print(f"\n{len(rows)}/{len(grid)} runs; rows appended to {path}")


def _s03_dense(depth):
    """Dense reference checkpoint at a depth: B1_dense_s1 at d8, else s03_dense's seed-1 run.
    Since 2026-10-03 the mainline test depth is d4 (cheaper), trained on jimpearse01."""
    return S03_DENSE if depth == 8 else f"{S03}/d{depth}/S03_dense_s1"


def _s03_common(depth=8):
    return ["--depth", str(depth), "--max-seq-len", "2048", "--window-pattern", "SSSL",
            "--device-batch-size", "16", "--warmup-ratio", "0.005", "--warmdown-ratio", "0.65",
            "--final-lr-frac", "0.05", "--eval-every", "-1", "--core-metric-every", "0",
            "--sample-every", "-1", "--save-every", "-1", "--sap-eval-steps", "40", "--log-every", "50",
            "--data-dir", f"{VOL}/data", "--tokenizer-dir", S03_TOK]


_S03_VARIANTS = {"nce": ["--sap-nce-props", "64", "--sap-nce-neg", "4"],
                 "supp": ["--sap-supp-min-count", "2"], "soft": ["--sap-soft-eps", "0.1"]}
_S03_VARIANTS_A = {"nce": ["--nce-props", "64"], "supp": ["--supp-min-count", "1"], "soft": ["--soft-eps", "0.1"]}


def _s03_split(spec):
    """'cut_crf+nce@L4P2' -> ('cut_crf', 'nce'); 'local' -> ('local', ''). Capacity options
    after '@' are read by _s03_capacity."""
    spec = spec.split("@")[0]
    mode, _, variant = spec.partition("+")
    return mode, variant


def _s03_capacity(spec):
    """'@L4M4W64P2D4' -> head layers 4, MLP mult 4, context window 64, cut pre-layers 2,
    depth_local through the top 4 trunk layers."""
    opts = {}
    if "@" in spec:
        for key, val in re.findall(r"([LMWPDSB])(\d+)", spec.split("@", 1)[1]):
            opts[key] = val
    return opts


def _s03_mode_args(spec, T, frac, lattice_k=256):
    mode, variant = _s03_split(spec)
    cap = _s03_capacity(spec)
    layers = cap.get("L", "3" if mode == "sir_tree" else "2")
    args = ["--sap-block-t", str(T), "--sap-block-mode", mode, "--sap-block-frac", str(frac),
            "--sap-head-layers", layers]
    if "M" in cap:
        args += ["--sap-head-mlp-mult", cap["M"]]
    if "W" in cap:
        args += ["--sap-ctx-window", cap["W"]]
    if "P" in cap:
        args += ["--sap-cut-pre-layers", cap["P"]]
    if "D" in cap:
        args += ["--sap-depth-layers", cap["D"]]
    if "S" in cap:
        args += ["--sap-depth-share", cap["S"]]
    if "B" in cap:
        args += ["--sap-depth-bottom", cap["B"]]
    if mode == "sir_tree":
        args += ["--sap-sir-topk", "16", "--sap-sir-posterior-mix", "0"]
    if variant:
        args += _S03_VARIANTS[variant]
        if variant in ("supp", "soft"):
            args += ["--sap-table-path", S03_TABLES]
    if mode in ("lat_crf", "lat_tt", "lat_cp", "cut_crf", "cut_tt"):
        # Stage 0: top-64 coverage of the data token is 83/71/64/57% at slots 1-4, top-256
        # 91/84/76/72%; escaped tokens lose their coupling, so the d8 lattice is 256 wide.
        args += ["--sap-lattice-k", str(lattice_k)]
    if mode in ("pmi_chain", "corpus_code"):
        args += ["--sap-table-path", S03_TABLES]
    if mode == "p1_selfpost":
        args += ["--sap-latent-codes", "64"]
    return args


@app.function(gpu="H100", timeout=6 * 3600, volumes={VOL: VOLUME})
def s03_train(tag: str, train_args: list, smoke: bool = False, depth: int = 8) -> dict:
    _workdir(S03_TOK_NAME)
    # Smoke checkpoints live apart: a full run skips any tag whose directory already holds a
    # checkpoint, and on 2026-10-03 that silently reused 4-iteration smoke heads for three arms.
    ckdir = f"{S03}/d{depth}" + ("_smoke" if smoke else "")
    if os.path.isdir(f"{ckdir}/{tag}") and any(f.startswith("model_") for f in os.listdir(f"{ckdir}/{tag}")):
        print(f"{tag}: checkpoint exists, not retraining")
        text = open(f"{S03}/logs/{tag}.log").read() if os.path.exists(f"{S03}/logs/{tag}.log") else ""
        return {"tag": tag, "returncode": 0, "skipped": True, **_parse_train_log(text)}
    cmd = [sys.executable, "-m", "scripts.base_train", *train_args, "--checkpoints-dir", ckdir, "--model-tag", tag]
    code, text = _run_logged(cmd, f"{S03}/logs/{tag}.log")
    VOLUME.commit()
    return {"tag": tag, "returncode": code, **_parse_train_log(text)}


@app.function(gpu="H100", timeout=4 * 3600, volumes={VOL: VOLUME})
def s03_post(tag: str, T: int, gen_prefixes: int = 512, gen_tokens: int = 128,
             nce_props: str = "", depth: int = 8) -> dict:
    """Decode speed with CUDA graphs (AR and block, both sampling at temperature 1, so a
    resampling head pays for its proposals) at 32*T and 256 tokens, exact block bpb on 1M
    validation tokens, and generation quality scored by the dense reference. nce_props: also
    time a self-contrastive head at these proposal counts."""
    _workdir(S03_TOK_NAME)
    ck = f"{S03}/d{depth}/{tag}"
    out = {"tag": tag}
    for L in ([int(x) for x in nce_props.split(",") if x] or [None]):
        for n_tok in sorted({32 * T, 256}):
            suffix = f"_g{n_tok}" + ("" if L is None else f"_L{L}")
            js = f"{S03}/decode_{tag}{suffix}.json"
            cmd = [sys.executable, "-m", "scripts.sap_decode_bench", "--checkpoint-dir", ck,
                   "--tokenizer-dir", S03_TOK, "--gen-tokens", str(n_tok),
                   "--graph-temperature", "1.0", "--out", js]
            if L is not None:
                cmd += ["--nce-props", str(L)]
            code, _ = _run_logged(cmd, f"{S03}/logs/decode_{tag}{suffix}.log")
            out[f"decode{suffix}"] = json.load(open(js)) if code == 0 and os.path.exists(js) else {"rc": code}
    js = f"{S03}/bpb_{tag}.json"
    code, _ = _run_logged([sys.executable, "-m", "scripts.sap_eval_bpb", "--checkpoint-dir", ck,
                           "--tokenizer-dir", S03_TOK, "--data-dir", f"{VOL}/data",
                           "--eval-tokens", "1048576", "--blocks-per-row", "64", "--out", js],
                          f"{S03}/logs/bpb_{tag}.log")
    out["bpb"] = json.load(open(js)) if code == 0 and os.path.exists(js) else {"rc": code}
    code, text = _run_logged([sys.executable, "-m", "scripts.sap_eval_generation", "--checkpoint-dir", ck,
                              "--reference-dir", _s03_dense(depth), "--tokenizer-dir", S03_TOK,
                              "--data-dir", f"{VOL}/data", "--n-prefixes", str(gen_prefixes),
                              "--gen-tokens", str(gen_tokens), "--out", f"{S03}/gen_{tag}.jsonl"],
                             f"{S03}/logs/gen_{tag}.log")
    m = re.search(r"reference ppl\s+next-token\s+([0-9.]+) \| block\s+([0-9.]+)", text)
    if m:
        out["ref_ppl_ar"], out["ref_ppl_block"] = float(m.group(1)), float(m.group(2))
    m = re.search(r"distinct 3-grams\s+next-token ([0-9.]+) \| block ([0-9.]+)", text)
    if m:
        out["distinct3_ar"], out["distinct3_block"] = float(m.group(1)), float(m.group(2))
    VOLUME.commit()
    return out


def _s03_report(results, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    for tag, r in sorted(results.items()):
        post = r.get("post") or {}
        bpb = post.get("bpb") or {}
        speeds = []
        for key in sorted(k for k in post if k.startswith("decode_g")):
            for row in (post[key] or {}).get("rows", []):
                if row.get("batch") == 16:
                    speeds.append(f"{key[7:]}: eager {row.get('speedup', 0):.2f}x graph "
                                  f"{row.get('graph_speedup', float('nan')):.2f}x")
        print(f"{tag:34s} val_bpb={r.get('val_bpb')} block/ntp={bpb.get('block_over_ntp')} "
              f"(nce est {bpb.get('block_over_ntp_nce_est')}) block_bpb={bpb.get('block_bpb')} "
              f"ref_ppl {post.get('ref_ppl_ar')}->{post.get('ref_ppl_block')} "
              f"distinct3 {post.get('distinct3_ar')}->{post.get('distinct3_block')} | b16 {'; '.join(speeds)}")
    print(f"summary written to {path}")


@app.local_entrypoint()
def s03_stage2(modes: str = "", ts: str = "4", frac: float = 0.0625, iters: int = 600,
               post: bool = True, smoke: bool = False, rerun: str = "", extra: str = "",
               depth: int = 4, out: str = "out/s03_sap_stage2"):
    """Block heads trained on the frozen dense trunk (identical to sap_trunk_grad=0 at its end
    point, so not a pretrained probe): the cheap screen for block bpb, speed and generation.
    Its ntp_bpb_same_tokens is the dense model's next-token bpb on the block-eval tokens (the
    eval's block starts are evenly spaced over a fixed validation stream, so every run scores
    the same tokens): the denominator of the block-vs-dense comparison."""
    mode_list = [m for m in modes.split(",") if m] or S03_V4 + ["indep", "local"]
    common = _s03_common(depth) + ["--num-iterations", str(iters if not smoke else 4), "--total-batch-size", "524288",
                                   "--eval-tokens", str(4 * 2 ** 20), "--sap-init-trunk", _s03_dense(depth),
                                   "--sap-freeze-trunk", "1", "--sap-trunk-grad", "0"] + extra.split()
    jobs = []
    for T in [int(t) for t in ts.split(",") if t]:
        for m in mode_list:
            jobs.append((f"S03h_{m.replace('+', '_').replace('@', '_')}_T{T}" + rerun, common + _s03_mode_args(m, T, frac),
                         smoke, depth))
    print(f"S03 stage 2: {len(jobs)} head-only runs on the frozen d{depth} dense trunk {_s03_dense(depth)}")
    results = {}
    for res in s03_train.starmap(jobs, return_exceptions=True):
        if isinstance(res, BaseException):
            print(f"  training failed: {res!r}")
            continue
        results[res["tag"]] = res
        se = res.get("sap_eval") or {}
        print(f"  {res['tag']:28s} rc={res['returncode']} block_bpb={se.get('block_bpb')} "
              f"ntp_same={se.get('ntp_bpb_same_tokens')} val_bpb={res.get('val_bpb')}")
    if post and not smoke:
        pj = [(tag, int(tag.rsplit("_T", 1)[1].split("_")[0]), 512, 128, "64,16" if "_nce_" in tag else "", depth)
              for tag, r in results.items() if r.get("returncode") == 0]
        for res in s03_post.starmap(pj, return_exceptions=True):
            if isinstance(res, BaseException):
                print(f"  post failed: {res!r}")
                continue
            results[res["tag"]]["post"] = res
    _s03_report(results, os.path.join(out, "summary_stage2.json"))


@app.local_entrypoint()
def s03_stage3(modes: str = "cut_crf,lat_tt", grads: str = "0,0.1,1", block_t: int = 4, frac: float = 0.0625,
               soft_on: str = "", seed: int = 1, post: bool = True, extra: str = "", rerun: str = "",
               depth: int = 4, out: str = "out/s03_sap_stage3"):
    """From scratch at the dense arm's training FLOPs: the chosen heads at each trunk gradient,
    plus (soft_on=<mode>) the n-gram soft-target auxiliary on that mode's fully co-trained arm."""
    T = block_t   # Modal's CLI lowercases parameter names, so the entrypoint cannot take `T`
    flops, tokens, fpt = _dense_flops(depth, 2048, "SSSL")
    print(f"S03 stage 3: dense d{depth} {fpt:,.0f} FLOPs/token x {tokens:,} tokens = {flops:.4e} FLOPs per arm")
    common = _s03_common(depth) + ["--target-flops", f"{flops:.6e}", "--target-param-data-ratio", "10.5",
                                   "--total-batch-size", "-1", "--eval-tokens", str(20 * 2 ** 20), "--seed", str(seed)]
    jobs = []
    for m in [x for x in modes.split(",") if x]:
        for g in [float(x) for x in grads.split(",") if x]:
            jobs.append((f"S03_{m.replace('+', '_').replace('@', '_')}_T{T}_g{g:g}_s{seed}" + rerun,
                         common + _s03_mode_args(m, T, frac) + ["--sap-trunk-grad", str(g)] + extra.split(),
                         False, depth))
    if soft_on:
        jobs.append((f"S03_{soft_on}_T{T}_g1_soft_s{seed}",
                     common + _s03_mode_args(soft_on, T, frac) + ["--sap-trunk-grad", "1",
                                                                  "--sap-soft-eps", "0.1", "--sap-table-path", S03_TABLES],
                     False, depth))
    print(f"launching {len(jobs)} FLOPs-matched runs: {', '.join(j[0] for j in jobs)}")
    results = {}
    for res in s03_train.starmap(jobs, return_exceptions=True):
        if isinstance(res, BaseException):
            print(f"  training failed: {res!r}")
            continue
        results[res["tag"]] = res
        print(f"  {res['tag']:34s} rc={res['returncode']} val_bpb={res.get('val_bpb')} "
              f"tokens={res.get('train_tokens')} flops/tok={res.get('flops_per_token')}")
    if post:
        pj = [(tag, T, 512, 128, "64,16" if "_nce_" in tag else "", depth)
              for tag, r in results.items() if r.get("returncode") == 0]
        for res in s03_post.starmap(pj, return_exceptions=True):
            if isinstance(res, BaseException):
                print(f"  post failed: {res!r}")
                continue
            results[res["tag"]]["post"] = res
    _s03_report(results, os.path.join(out, "summary_stage3.json"))


@app.local_entrypoint()
def s03_adaptive(tags: str, depth: int = 4, rows: int = 32):
    """Adaptive block length on trained depth checkpoints (scripts/sap_adaptive_oracle.py): the
    exact likelihood ratio against the trunk and the modelled sequential speedup of a causal
    stop-or-continue router, per threshold, one container per checkpoint."""
    jobs = [(f"adaptive_{t}", ["scripts.sap_adaptive_oracle", "--checkpoint-dir", f"{S03}/d{depth}/{t}",
                               "--tokenizer-dir", S03_TOK, "--data-dir", f"{VOL}/data", "--rows", str(rows),
                               "--out", f"{S03}/adaptive_{t}.json"]) for t in tags.split(",") if t]
    for res in s03_job.starmap(jobs, return_exceptions=True):
        if isinstance(res, BaseException):
            print(f"  failed: {res!r}")
            continue
        print(f"== {res['name']} rc={res['returncode']}\n{res['tail']}")


@app.function(gpu="H100", timeout=2 * 3600, volumes={VOL: VOLUME})
def s03_ar_speed(ckpts: list, gen_tokens: int = 256) -> dict:
    """Next-token decode speed (eager and CUDA graphs, temperature 1) of several checkpoints in
    one container, so their ratios share a GPU. ckpts: checkpoint dirs under out/s03_sap."""
    _workdir(S03_TOK_NAME)
    out = {}
    for ck in ckpts:
        js = f"{S03}/arspeed_{os.path.basename(ck)}.json"
        code, _ = _run_logged([sys.executable, "-m", "scripts.sap_decode_bench", "--checkpoint-dir", f"{S03}/{ck}",
                               "--tokenizer-dir", S03_TOK, "--gen-tokens", str(gen_tokens), "--ar-only",
                               "--graph-temperature", "1.0", "--out", js], f"{S03}/logs/arspeed_{os.path.basename(ck)}.log")
        out[ck] = json.load(open(js)) if code == 0 and os.path.exists(js) else {"rc": code}
    VOLUME.commit()
    return out


@app.local_entrypoint()
def s03_dense_frontier(depth: int = 4, layers: str = "1,2,3", seed: int = 1,
                       out: str = "out/s03_sap_dense_frontier"):
    """Dense models as wide as d<depth> but shallower, trained at d<depth>'s FLOPs: the cost of
    buying decode speed by removing layers, against which a block decoder's (speed, C) is read.
    Then every one of them and the d<depth> reference timed in one container."""
    flops, tokens, fpt = _dense_flops(depth, 2048, "SSSL")
    width = depth * 64
    print(f"dense frontier at d{depth}'s budget {flops:.4e} FLOPs, width {width}, layers {layers}")
    common = _s03_common(depth) + ["--target-flops", f"{flops:.6e}", "--target-param-data-ratio", "10.5",
                                   "--total-batch-size", "-1", "--eval-tokens", str(20 * 2 ** 20),
                                   "--seed", str(seed)]
    jobs = []
    for n in [int(x) for x in layers.split(",") if x]:
        args = list(common)
        args[args.index("--depth") + 1] = str(n)
        jobs.append((f"S03_dense_L{n}_w{width}_s{seed}", args + ["--model-dim", str(width)], False, depth))
    results = {}
    for res in s03_train.starmap(jobs, return_exceptions=True):
        if isinstance(res, BaseException):
            print(f"  training failed: {res!r}")
            continue
        results[res["tag"]] = res
        print(f"  {res['tag']:24s} rc={res['returncode']} val_bpb={res.get('val_bpb')} tokens={res.get('train_tokens')}")
    # The speed reference is the full-depth model trained here when layers includes it (so the
    # whole frontier shares one training setup), else the depth's dense reference.
    own = f"S03_dense_L{depth}_w{width}_s{seed}"
    ref = own if results.get(own, {}).get("returncode") == 0 else "S03_dense_s1"
    ckpts = [f"d{depth}/{ref}"] + [f"d{depth}/{t}" for t, r in results.items() if r.get("returncode") == 0 and t != ref]
    speeds = s03_ar_speed.remote(ckpts)
    for ck, r in speeds.items():
        rows = {row["batch"]: row for row in (r.get("rows") or [])}
        print(f"  {ck:32s} " + " ".join(f"b{b}: graph {rows[b].get('graph_ar_tok_per_s', float('nan')):,.0f} tok/s"
                                         for b in sorted(rows)))
        tag = ck.split("/")[-1]
        results.setdefault(tag, {"tag": tag})["ar_speed"] = r
    _s03_report(results, os.path.join(out, f"summary_frontier_d{depth}.json"))


@app.local_entrypoint()
def s03_dense(depth: int = 4, seeds: str = "1,2", out: str = "out/s03_sap_dense"):
    """The dense reference at a depth (seed 1 is the reference _s03_dense points at; more seeds
    give the run-to-run noise floor that SAP arms' trunk differences are read against)."""
    flops, tokens, fpt = _dense_flops(depth, 2048, "SSSL")
    print(f"S03 dense d{depth}: {fpt:,.0f} FLOPs/token x {tokens:,} tokens = {flops:.4e} FLOPs")
    common = _s03_common(depth) + ["--target-flops", f"{flops:.6e}", "--target-param-data-ratio", "10.5",
                                   "--total-batch-size", "-1", "--eval-tokens", str(20 * 2 ** 20)]
    jobs = [(f"S03_dense_s{s}", common + ["--seed", s], False, depth) for s in seeds.split(",") if s]
    results = {}
    for res in s03_train.starmap(jobs, return_exceptions=True):
        if isinstance(res, BaseException):
            print(f"  training failed: {res!r}")
            continue
        results[res["tag"]] = res
        print(f"  {res['tag']:20s} rc={res['returncode']} val_bpb={res.get('val_bpb')} tokens={res.get('train_tokens')}")
    _s03_report(results, os.path.join(out, f"summary_dense_d{depth}.json"))


# ----------------------------------------------------------------------------- S08: lanes (s08_sap_lanes_plan.md)
# (Named S08: a parallel session's s07_sap_mechanism_brainstorm.md holds the S07 name. The first d4
# lane launch, before the rename, used checkpoint tags S07_dense_L_s*, S07_lanes*_s*.)
# One trunk pass emits L tokens S positions apart (nanochat/lanes.py). Every arm, the dense
# baseline included, uses full-context attention (window pattern L): lane decoding masks cached
# keys by generation step, not by sliding window. Checkpoints share out/s03_sap/d<depth>.


def _s08_common(depth):
    args = _s03_common(depth)
    args[args.index("--window-pattern") + 1] = "L"
    return args


@app.function(gpu="H100", timeout=2 * 3600, volumes={VOL: VOLUME})
def s08_post(tag: str, L: int, depth: int = 4, gen_tokens: int = 1921) -> dict:
    """CUDA-graph lane decoding against next-token decoding of the same checkpoint."""
    _workdir(S03_TOK_NAME)
    js = f"{S03}/lanes_{tag}.json"
    code, _ = _run_logged([sys.executable, "-m", "scripts.sap_decode_bench", "--checkpoint-dir", f"{S03}/d{depth}/{tag}",
                           "--tokenizer-dir", S03_TOK, "--lanes", str(L), "--gen-tokens", str(gen_tokens),
                           "--prompt-len", "64", "--batch-sizes", "1", "16", "--graph-temperature", "1.0",
                           "--out", js], f"{S03}/logs/lanes_{tag}.log")
    VOLUME.commit()
    return {"tag": tag, "rc": code, **(json.load(open(js)) if code == 0 and os.path.exists(js) else {})}


@app.function(gpu="H100", timeout=3 * 3600, volumes={VOL: VOLUME})
def s08_gen_job(tag: str, L: int, depth: int, ar_tag: str, ref_tag: str, n_prefixes: int = 256,
                gen_tokens: int = 1921, temperature: float = 1.0, ar_temperature: float = -1.0, ar_top_p: float = 1.0,
                ar_rep: float = 1.0) -> dict:
    """Reference PPL and distinct 3-grams of lane samples against another model's next-token
    samples, both scored by a third (dense) model, so neither side is scored by itself."""
    _workdir(S03_TOK_NAME)
    ck = lambda t: f"{S03}/d{depth}/{t}"
    code, text = _run_logged([sys.executable, "-m", "scripts.sap_eval_generation", "--checkpoint-dir", ck(tag),
                              "--ar-dir", ck(ar_tag), "--reference-dir", ck(ref_tag), "--tokenizer-dir", S03_TOK,
                              "--data-dir", f"{VOL}/data", "--lanes", str(L), "--n-prefixes", str(n_prefixes),
                              "--batch", "16", "--gen-tokens", str(gen_tokens), "--real", "--temperature", str(temperature),
                              *(["--ar-temperature", str(ar_temperature)] if ar_temperature > 0 else []),
                              *(["--ar-top-p", str(ar_top_p)] if ar_top_p < 1 else []),
                              *(["--ar-rep-penalty", str(ar_rep)] if ar_rep != 1 else []),
                              "--out", f"{S03}/gen_{tag}_t{temperature:g}_{ar_tag}_t{ar_temperature:g}_p{ar_top_p:g}_r{ar_rep:g}.jsonl"],
                             f"{S03}/logs/gen_{tag}_t{temperature:g}_{ar_tag}_t{ar_temperature:g}_p{ar_top_p:g}_r{ar_rep:g}.log")
    out = {"tag": tag, "temperature": temperature, "ar_tag": ar_tag,
           "ar_temperature": ar_temperature if ar_temperature > 0 else temperature, "ar_top_p": ar_top_p,
           "ar_rep": ar_rep, "rc": code}
    m = re.search(r"reference ppl\s+next-token\s+([0-9.]+) \| block\s+([0-9.]+)", text)
    if m:
        out["ref_ppl_ar"], out["ref_ppl_lanes"] = float(m.group(1)), float(m.group(2))
    m = re.search(r"distinct 3-grams\s+next-token ([0-9.]+) \| block ([0-9.]+)", text)
    if m:
        out["distinct3_ar"], out["distinct3_lanes"] = float(m.group(1)), float(m.group(2))
    m = re.search(r"unigram entropy\s+next-token ([0-9.]+) \| block ([0-9.]+)", text)
    if m:
        out["entropy_ar"], out["entropy_lanes"] = float(m.group(1)), float(m.group(2))
    m = re.search(r"real continuation\s+reference ppl\s+([0-9.]+) \| distinct 3-grams ([0-9.]+) \| unigram entropy ([0-9.]+)", text)
    if m:
        out["ref_ppl_real"], out["distinct3_real"], out["entropy_real"] = (float(g) for g in m.groups())
    VOLUME.commit()
    return out


@app.local_entrypoint()
def s08_profile(tags: str, depth: int = 4, ref_tag: str = "S07_dense_L_s2"):
    """Per-position reference profile of s08_gen outputs (tags: comma list of '<tag>:<L>')."""
    jobs = []
    for t in [x for x in tags.split(",") if x]:
        tag, L = t.split(":")
        jobs.append((f"profile_{tag}", ["scripts.sap_lane_gen_profile", "--gen-jsonl", f"{S03}/gen_{tag}.jsonl",
                                        "--reference-dir", f"{S03}/d{depth}/{ref_tag}", "--tokenizer-dir", S03_TOK,
                                        "--lanes", L, "--out", f"{S03}/profile_{tag}.json"]))
    for res in s03_job.starmap(jobs, return_exceptions=True):
        print(res if isinstance(res, BaseException) else f"== {res['name']} rc={res['returncode']}\n{res['tail'][-1500:]}")


@app.local_entrypoint()
def s08_gen(tags: str, depth: int = 4, ar_tag: str = "S07_dense_L_s1", ref_tag: str = "S07_dense_L_s2",
            n_prefixes: int = 256, gen_tokens: int = 1921, temperatures: str = "1.0", ar_settings: str = ""):
    """Generation quality of lane checkpoints (tags: comma list of '<tag>:<L>'). The prompt is 64
    tokens, so gen_tokens = 1 + L * S must give the lane length the model trained with at that
    prompt, S = (2048 - 64) / L: gen_tokens 1985 for any L dividing 1984."""
    jobs = [(t.split(":")[0], int(t.split(":")[1]), depth, ar_tag, ref_tag, n_prefixes, gen_tokens, float(temp))
            for t in tags.split(",") if t for temp in temperatures.split(",") if temp]
    if ar_settings:            # 'AR_TAG:T:TOP_P:REP' list, each with the lane model at the first temperature
        t0 = float(temperatures.split(",")[0])
        jobs += [(t.split(":")[0], int(t.split(":")[1]), depth, a.split(":")[0], ref_tag, n_prefixes, gen_tokens, t0,
                  float(a.split(":")[1]), float(a.split(":")[2]), float(a.split(":")[3]))
                 for t in tags.split(",") if t for a in ar_settings.split(",") if a]
    for res in s08_gen_job.starmap(jobs, return_exceptions=True):
        print(res)


@app.local_entrypoint()
def s08_lanes(depth: int = 4, lanes: str = "2,4,8", seeds: str = "1,2", post: bool = True,
              extra: str = "", rerun: str = "", mult: float = 1.0, out: str = "out/s08_lanes"):
    """Lane tax (pre-registered L1/L3 in s08_sap_lanes_plan.md): lane-order val bpb of models
    trained from scratch in lane order against a dense causal model of the same architecture and
    FLOPs, two seeds each, then lane decoding speed for seed 1. mult scales the token budget of
    every arm, dense included (the SAP longer-training protocol: run mult=1, 2, 4; tags get x<mult>
    when mult != 1)."""
    flops, tokens, fpt = _dense_flops(depth, 2048, "L")
    flops *= mult
    mt = "" if mult == 1 else f"x{mult:g}"
    print(f"S08 lanes at d{depth}: {fpt:,.0f} FLOPs/token x {int(tokens * mult):,} tokens = {flops:.4e} FLOPs per arm")
    common = _s08_common(depth) + ["--target-flops", f"{flops:.6e}", "--target-param-data-ratio", "10.5",
                                   "--total-batch-size", "-1", "--eval-tokens", str(20 * 2 ** 20),
                                   "--lane-prefix-max", "256", "--lane-eval-prefix", "128"] + extra.split()
    lane_list = [int(x) for x in lanes.split(",") if x]
    seed_list = [s for s in seeds.split(",") if s]
    jobs = [(f"S08_dense_L{mt}_s{sd}{rerun}", common + ["--seed", sd, "--lanes", "0"], False, depth)
            for sd in seed_list]
    jobs += [(f"S08_lanes{L}{mt}_s{sd}{rerun}", common + ["--seed", sd, "--lanes", str(L)], False, depth)
             for L in lane_list for sd in seed_list]
    results = {}
    for res in s03_train.starmap(jobs, return_exceptions=True):
        if isinstance(res, BaseException):
            print(f"  training failed: {res!r}")
            continue
        results[res["tag"]] = res
        print(f"  {res['tag']:22s} rc={res['returncode']} val_bpb={res.get('val_bpb')}")
    dense = [r["val_bpb"] for t, r in results.items() if t.startswith("S08_dense") and r.get("val_bpb")]
    if dense:
        d = sum(dense) / len(dense)
        print(f"dense (window L) val bpb {d:.6f} over {len(dense)} seeds: {dense}")
        for L in lane_list:
            v = [r["val_bpb"] for t, r in results.items() if t.startswith(f"S08_lanes{L}{mt}_") and r.get("val_bpb")]
            if v:
                print(f"  lanes L={L}: val bpb {sum(v) / len(v):.6f} (seeds {v}) | tax {sum(v) / len(v) / d - 1:+.4%}")
    if post:
        pj = [(t, int(t.split("_")[1][5:].split("x")[0]), depth) for t, r in results.items()
              if t.startswith("S08_lanes") and "_s1" in t and r.get("returncode") == 0]
        for res in s08_post.starmap(pj, return_exceptions=True):
            if isinstance(res, BaseException):
                print(f"  post failed: {res!r}")
                continue
            results[res["tag"]]["post"] = res
            for row in res.get("rows", []):
                print(f"  {res['tag']:22s} b{row['batch']}: lanes {row['graph_lane_tok_per_s']:,.0f} tok/s vs AR "
                      f"{row['graph_ar_tok_per_s']:,.0f} -> {row['graph_speedup']:.2f}x")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, f"summary_lanes_d{depth}.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"summary written to {out}/summary_lanes_d{depth}.json")


# ----------------------------------------------------------------------------- S09: T=L gates (s09_sap_tl_gates.md)
# Information oracle for S07 hypothesis B: d4 models that attend only to a 128-token prompt plus
# their last k positions, scored per position against full-context dense models of the same
# FLOPs. Checkpoints under out/s03_sap/d<depth>.


@app.local_entrypoint()
def s09_window_oracle(depth: int = 4, windows: str = "8,32,128", second_seed_window: int = 32,
                      prompt_len: int = 128, rows: int = 256):
    flops, tokens, fpt = _dense_flops(depth, 2048, "L")
    common = _s08_common(depth) + ["--target-flops", f"{flops:.6e}", "--target-param-data-ratio", "10.5",
                                   "--total-batch-size", "-1", "--eval-tokens", str(20 * 2 ** 20)]
    jobs = [(f"S09_dense_L_s{sd}", common + ["--seed", str(sd)], False, depth) for sd in (1, 2)]
    ks = [int(k) for k in windows.split(",") if k]
    for k in ks:
        seeds = (1, 2) if k == second_seed_window else (1,)
        jobs += [(f"S09_win{k}_s{sd}", common + ["--seed", str(sd), "--prompt-window", str(k),
                                                 "--prompt-len", str(prompt_len)], False, depth) for sd in seeds]
    print(f"S09 window oracle at d{depth}: {len(jobs)} runs at {flops:.4e} FLOPs")
    ok = []
    for res in s03_train.starmap(jobs, return_exceptions=True):
        if isinstance(res, BaseException):
            print(f"  training failed: {res!r}")
            continue
        print(f"  {res['tag']:18s} rc={res['returncode']} val_bpb={res.get('val_bpb')}")
        if res.get("returncode") == 0:
            ok.append(res["tag"])
    models = []
    for tag in sorted(ok, key=lambda t: (not t.startswith("S09_dense"), t)):
        k = tag.split("_")[1][3:] if tag.startswith("S09_win") else ""
        models += ["--model", f"{tag}:{S03}/d{depth}/{tag}" + (f":{k}" if k else "")]
    res = s03_job.remote("s09_position_bpb", ["scripts.sap_position_bpb", "--tokenizer-dir", S03_TOK,
                                              "--data-dir", f"{VOL}/data", "--rows", str(rows),
                                              "--prompt-len", str(prompt_len),
                                              "--out", f"{S03}/s09_position_bpb_d{depth}.json", *models])
    print(res["tail"][-2500:])


@app.local_entrypoint()
def s11_separator_oracle(depth: int = 4, slots: str = "1,4,16,64", control: int = 64, split: int = 1024,
                         seeds: str = "1", extra_seed_slots: int = 16, rows: int = 512, within_doc: int = 128,
                         prefix: str = "S11b", smoke: bool = False):
    """S11 real-text separator oracle (s11_sap_tl_brainstorm.md, pre-registered): the second half of
    each 2048-token row reaches the first half only through m learned summary slots at --split.
    Scored per position against full-context dense (S09_dense_L_s1/s2) and a slots-inserted
    full-attention control, at the dense FLOPs. Kill the Markov form of separators if m=64 costs
    more than 3% on the second half; go if m <= 16 costs at most 1%."""
    flops, tokens, fpt = _dense_flops(depth, 2048, "L")
    common = _s08_common(depth) + ["--target-flops", f"{flops:.6e}", "--target-param-data-ratio", "10.5",
                                   "--total-batch-size", "-1", "--eval-tokens", str(20 * 2 ** 20),
                                   "--sep-split", str(split)]
    ms = [int(m) for m in slots.split(",") if m]
    sds = [int(sd) for sd in seeds.split(",") if sd]
    if smoke:                                          # a few steps, checkpoints under d<depth>_smoke
        common += ["--num-iterations", "20", "--eval-tokens", str(2 ** 18)]
    jobs = [(f"{prefix}_sep{m}_s{sd}", common + ["--seed", str(sd), "--sep-slots", str(m)], smoke, depth)
            for m in ms for sd in sds]
    if extra_seed_slots:
        jobs.append((f"{prefix}_sep{extra_seed_slots}_s2", common + ["--seed", "2", "--sep-slots", str(extra_seed_slots)],
                     smoke, depth))
    if control:
        jobs.append((f"{prefix}_sep{control}full_s1",
                     common + ["--seed", "1", "--sep-slots", str(control), "--sep-full"], smoke, depth))
    print(f"S11 separator oracle at d{depth}: {len(jobs)} runs at {flops:.4e} FLOPs")
    ok = []
    for res in s03_train.starmap(jobs, return_exceptions=True):
        if isinstance(res, BaseException):
            print(f"  training failed: {res!r}")
            continue
        print(f"  {res['tag']:20s} rc={res['returncode']} val_bpb={res.get('val_bpb')}")
        if res.get("returncode") == 0:
            ok.append(res["tag"])
    models = [f"--model=dense_s{sd}:{S03}/d{depth}/S09_dense_L_s{sd}" for sd in (1, 2)]
    for tag in sorted(ok):
        spec = tag.split("_")[1]                       # sep16 or sep64full
        models.append(f"--model={tag}:{S03}/d{depth}{'_smoke' if smoke else ''}/{tag}:{spec}")
    edges = f"0,{split - 1},{split + 64},{split + 128},{split + 256},{split + 512},{2048 - 64}"
    for wd in sorted({0, within_doc}):
        name = f"s11_separator_bpb_d{depth}_{prefix}_wd{wd}{'_smoke' if smoke else ''}"
        res = s03_job.remote(name, ["scripts.sap_position_bpb", "--tokenizer-dir", S03_TOK,
                                    "--data-dir", f"{VOL}/data", "--rows", str(rows), "--within-doc", str(wd),
                                    "--sep-split", str(split), "--edges", edges, "--out", f"{S03}/{name}.json",
                                    *models])
        print(f"--- {'all rows' if wd == 0 else f'rows whose split is >= {wd} tokens inside one document'}")
        print(res["tail"][-2500:])


@app.local_entrypoint()
def s11_wbisect(depth: int = 4, windows: str = "1,4,16", prefix: int = 128, seeds: str = "1", rows: int = 256,
                device_batch: int = 8, tag_prefix: str = "S11wb", smoke: bool = False):
    """S11 real-text window bisection (s11_sap_tl_brainstorm.md): d4 models trained in the
    window-bisection order (two-stream, exact likelihood) on the same tokens as dense
    (S09_dense_L_s1/s2), scored per position against them. n=1 is single-token bisection, the
    order whose tax killed token-level trees; larger n tests the separator principle on text.
    Kill: bpb on the block (positions >= prefix) above 1.05x dense at every n; decisive go: some n
    within 1%."""
    flops, tokens, fpt = _dense_flops(depth, 2048, "L")
    common = _s08_common(depth) + ["--target-flops", f"{flops:.6e}", "--target-param-data-ratio", "10.5",
                                   "--total-batch-size", "-1", "--eval-tokens", str(20 * 2 ** 20),
                                   "--wb-prefix", str(prefix)]
    common[common.index("--device-batch-size") + 1] = str(device_batch)
    if smoke:
        common += ["--num-iterations", "20", "--eval-tokens", str(2 ** 18)]
    ns = [int(n) for n in windows.split(",") if n]
    jobs = [(f"{tag_prefix}{n}_s{sd}", common + ["--seed", str(sd), "--wb-window", str(n)], smoke, depth)
            for n in ns for sd in seeds.split(",") if sd]
    print(f"S11 window bisection at d{depth}: {len(jobs)} runs, matched tokens with dense ({tokens:,} tokens)")
    ok = []
    for res in s03_train.starmap(jobs, return_exceptions=True):
        if isinstance(res, BaseException):
            print(f"  training failed: {res!r}")
            continue
        print(f"  {res['tag']:16s} rc={res['returncode']} val_bpb={res.get('val_bpb')}")
        if res.get("returncode") == 0:
            ok.append(res["tag"])
    models = [f"--model=dense_s{sd}:{S03}/d{depth}/S09_dense_L_s{sd}" for sd in (1, 2)]
    for tag in sorted(ok):
        n = tag[len(tag_prefix):].split("_")[0]
        models.append(f"--model={tag}:{S03}/d{depth}{'_smoke' if smoke else ''}/{tag}:wb{n}")
    edges = f"0,{prefix},{prefix + 128},{prefix + 384},1024,2047"
    name = f"s11_wbisect_bpb_d{depth}_{tag_prefix}{'_smoke' if smoke else ''}"
    res = s03_job.remote(name, ["scripts.sap_position_bpb", "--tokenizer-dir", S03_TOK, "--data-dir", f"{VOL}/data",
                                "--rows", str(rows), "--wb-prefix", str(prefix), "--edges", edges,
                                "--out", f"{S03}/{name}.json", *models])
    print(res["tail"][-2500:])


@app.local_entrypoint()
def s11_blanes(depth: int = 4, configs: str = "128:4,64:4,64:8,32:8,16:8", prefix: int = 128, seeds: str = "1",
               mults: str = "1", dense_mults: str = "", rows: int = 256, device_batch: int = 8,
               tag_prefix: str = "S11bl", smoke: bool = False):
    """S11 bridged lanes on real text (s11_sap_tl_brainstorm.md): L intervals whose last n tokens are
    placed coarse-to-fine, then filled left to right in lockstep (two-stream, exact likelihood).
    configs "L:n,...". mults: training-token multiples of the dense compute-optimal budget (the
    SAP longer-training protocol); dense_mults trains dense references at those multiples too.
    Scored per position against every dense reference on identical targets."""
    flops, tokens, fpt = _dense_flops(depth, 2048, "L")
    base = _s08_common(depth) + ["--target-param-data-ratio", "10.5", "--total-batch-size", "-1",
                                 "--eval-tokens", str(20 * 2 ** 20)]
    if smoke:
        base += ["--num-iterations", "20", "--eval-tokens", str(2 ** 18)]
    cfgs = [tuple(int(v) for v in c.split(":")) for c in configs.split(",") if c]
    jobs = []
    for mult in [float(m) for m in mults.split(",") if m]:
        mt = f"x{mult:g}"
        for L, n in cfgs:
            for sd in [sd for sd in seeds.split(",") if sd]:
                a = base + ["--target-flops", f"{flops * mult:.6e}", "--seed", sd, "--wb-prefix", str(prefix),
                            "--wb-window", str(n), "--wb-lanes", str(L)]
                a[a.index("--device-batch-size") + 1] = str(device_batch)
                jobs.append((f"{tag_prefix}{L}n{n}{mt}_s{sd}", a, smoke, depth))
    for mult in [float(m) for m in dense_mults.split(",") if m]:
        for sd in [sd for sd in seeds.split(",") if sd]:
            jobs.append((f"S11dense_x{mult:g}_s{sd}",
                         base + ["--target-flops", f"{flops * mult:.6e}", "--seed", sd], smoke, depth))
    print(f"S11 bridged lanes at d{depth}: {len(jobs)} runs (dense budget {tokens:,} tokens = x1)")
    ok = []
    for res in s03_train.starmap(jobs, return_exceptions=True):
        if isinstance(res, BaseException):
            print(f"  training failed: {res!r}")
            continue
        print(f"  {res['tag']:24s} rc={res['returncode']} val_bpb={res.get('val_bpb')}")
        if res.get("returncode") == 0:
            ok.append(res["tag"])
    d = f"{S03}/d{depth}{'_smoke' if smoke else ''}"
    models = [f"--model=dense_s{sd}:{S03}/d{depth}/S09_dense_L_s{sd}" for sd in (1, 2)]
    for tag in sorted(ok):
        if tag.startswith("S11dense"):
            models.append(f"--model={tag}:{d}/{tag}")
        else:
            L, rest = tag[len(tag_prefix):].split("n", 1)
            n = rest.split("x")[0]
            models.append(f"--model={tag}:{d}/{tag}:bl{L}_{n}")
    edges = f"0,{prefix},{prefix + 128},{prefix + 384},1024,2047"
    name = f"s11_blanes_bpb_d{depth}_{tag_prefix}_m{mults.replace(',', '-')}{'_smoke' if smoke else ''}"
    res = s03_job.remote(name, ["scripts.sap_position_bpb", "--tokenizer-dir", S03_TOK, "--data-dir", f"{VOL}/data",
                                "--rows", str(rows), "--wb-prefix", str(prefix), "--edges", edges,
                                "--out", f"{S03}/{name}.json", *models])
    print(res["tail"][-3000:])


@app.local_entrypoint()
def s11_ladder(specs: str = "dense:1:1", depth: int = 4, prefix: int = 128, rows: int = 256, device_batch: int = 8,
               ref: str = "S11dense_x1_s1", name: str = "ladder", smoke: bool = False):
    """S11 real-text ladder in one app (the SAP longer-training protocol, s11_sap_tl_brainstorm.md).
    specs, comma separated:
        dense:MULT:SEED      dense causal model at MULT x the compute-optimal tokens
        bl:L:N:MULT:SEED     bridged lanes (L intervals, N-token separators)
        wb:N:MULT:SEED       window bisection (N-token windows)
        ln:L:MULT:SEED       S08 plain lanes
        lo:L:MULT:SEED       S08 plain lanes as a two-stream order (the control for sd)
        sd:K:MULT:SEED       seeded middle-out lanes, K seeds (two-stream)
        sd:K:M:MULT:SEED     seeded lanes with an M-token seed window (the left front starts warm)
        la:L:W:MULT:SEED     S12 sentence-aligned lanes, cut within the last W slots (scored by s12_aligned_eval)
        sp:L:MULT:SEED       S13 SV-A splice codes: lanes whose junction classes are drawn first
                             (Brown classes at {VOL}/out/s13/brown256.npy; nanochat/splice.py)
        ppi:L:F:MULT:SEED    S16-A plain lanes with a fraction F of position-preserving infill micro-steps
        mix:L1+L2+..:MULT:SEED  S16-B any-L lanes (each micro-step draws L from the list; scored at the largest
                             listed L <= 64, the paper's default; s16_score scores it at every L)
    Every model trained here is scored per position on identical targets, as a ratio to `ref`.
    Bar (user decision 2026-10-04): within 1% of dense-1x bpb at <= 4x tokens with >= 10x fewer
    sequential decode steps; the gap at equal tokens is reported alongside."""
    flops, tokens, fpt = _dense_flops(depth, 2048, "L")
    base = _s08_common(depth) + ["--target-param-data-ratio", "10.5", "--total-batch-size", "-1",
                                 "--eval-tokens", str(20 * 2 ** 20)]
    if smoke:
        base += ["--num-iterations", "20", "--eval-tokens", str(2 ** 18)]
    jobs, evspec = [], {}
    for spec in [x for x in specs.split(",") if x]:
        kind, *v = spec.split(":")
        mult, seed = float(v[-2]), v[-1]
        a = base + ["--target-flops", f"{flops * mult:.6e}", "--seed", seed]
        if kind == "dense":
            tag, ev = f"S11dense_x{mult:g}_s{seed}", ""
        elif kind == "bl":
            L, n = v[0], v[1]
            tag, ev = f"S11bl{L}n{n}x{mult:g}_s{seed}", f":bl{L}_{n}"
            a += ["--wb-prefix", str(prefix), "--wb-window", n, "--wb-lanes", L]
            a[a.index("--device-batch-size") + 1] = str(device_batch)
        elif kind == "wb":
            n = v[0]
            tag, ev = f"S11wb{n}x{mult:g}_s{seed}", f":wb{n}"
            a += ["--wb-prefix", str(prefix), "--wb-window", n]
            a[a.index("--device-batch-size") + 1] = str(device_batch)
        elif kind == "ln":
            L = v[0]
            tag, ev = f"S11ln{L}x{mult:g}_s{seed}", f":ln{L}"
            a += ["--lanes", L, "--lane-prefix-max", "256", "--lane-eval-prefix", str(prefix)]
        elif kind == "la":                                # S12 sentence-aligned lanes: la:L:W:MULT:SEED
            L, W = v[0], v[1]
            tag, ev = f"S11la{L}w{W}x{mult:g}_s{seed}", None
            a += ["--lanes", L, "--lane-prefix-max", "256", "--lane-eval-prefix", str(prefix),
                  "--lane-align-window", W]
        elif kind == "sp":                                # S13 SV-A splice codes: sp:L:MULT:SEED
            L = v[0]
            tag, ev = f"S13sp{L}x{mult:g}_s{seed}", f":sp{L}"
            a += ["--lanes", L, "--lane-prefix-max", "256", "--lane-eval-prefix", str(prefix),
                  "--splice", "1", "--class-map", f"{VOL}/out/s13/brown256.npy"]
        elif kind == "ppi":                               # S16-A: ppi:L:F:MULT:SEED
            L, F = v[0], float(v[1])
            tag, ev = f"S16ppi{round(F * 100)}L{L}x{mult:g}_s{seed}", f":ln{L}"
            a += ["--lanes", L, "--lane-prefix-max", "256", "--lane-eval-prefix", str(prefix),
                  "--lane-infill-frac", str(F)]
        elif kind == "mix":                               # S16-B: mix:16+32+64+128:MULT:SEED
            Ls = [int(x) for x in v[0].split("+")]
            L = max([x for x in Ls if x <= 64] or Ls)
            tag, ev = f"S16mix{'_'.join(map(str, Ls))}x{mult:g}_s{seed}", f":ln{L}"
            a += ["--lanes", str(L), "--lanes-mix", ",".join(map(str, Ls)), "--lane-prefix-max", "256",
                  "--lane-eval-prefix", str(prefix)]
        elif kind in ("lo", "sd"):                        # two-stream plain lanes / seeded middle-out lanes
            L = v[0]
            m = v[1] if kind == "sd" and len(v) == 4 else "1"     # sd:K:M:MULT:SEED (seed window M) or sd:K:MULT:SEED
            mt = f"w{m}" if m != "1" else ""
            tag, ev = f"S11{kind}{L}{mt}x{mult:g}_s{seed}", f":{kind}{L}" + (f"_{m}" if m != "1" else "")
            a += ["--wb-prefix", str(prefix), "--wb-order", "lanes" if kind == "lo" else "seeded", "--wb-lanes", L]
            if kind == "sd":
                a += ["--wb-window", m]
            a[a.index("--device-batch-size") + 1] = str(device_batch)
        else:
            raise ValueError(spec)
        jobs.append((tag, a, smoke, depth))
        evspec[tag] = ev
    print(f"S11 ladder at d{depth}: {len(jobs)} runs (x1 = {tokens:,} tokens)")
    ok = []
    for res in s03_train.starmap(jobs, return_exceptions=True):
        if isinstance(res, BaseException):
            print(f"  training failed: {res!r}")
            continue
        print(f"  {res['tag']:24s} rc={res['returncode']} val_bpb={res.get('val_bpb')}")
        if res.get("returncode") == 0:
            ok.append(res["tag"])
    d = f"{S03}/d{depth}{'_smoke' if smoke else ''}"
    # The reference goes first (the lane reports read every lanes model against it): this run's, or,
    # when the specs do not retrain it, an earlier run's at this depth.
    keep_ref = ref in ok or (ref not in evspec and not smoke)
    order = ([ref] if keep_ref else []) + sorted(t for t in ok if t != ref and evspec[t] is not None)
    models = [f"--model={t}:{d}/{t}{evspec.get(t, '')}" for t in order]      # aligned lanes: s12_aligned_eval
    edges = f"0,{prefix},{prefix + 128},{prefix + 384},1024,2047"
    out_name = f"s11_ladder_bpb_d{depth}_{name}{'_smoke' if smoke else ''}"
    res = s03_job.remote(out_name, ["scripts.sap_position_bpb", "--tokenizer-dir", S03_TOK, "--data-dir",
                                    f"{VOL}/data", "--rows", str(rows), "--wb-prefix", str(prefix), "--edges", edges,
                                    "--out", f"{S03}/{out_name}.json", *models])
    print(res["tail"][-3500:])


@app.function(gpu="H100", timeout=2 * 3600, volumes={VOL: VOLUME})
def s13_chunk_ae_job(K: int = 8, dz: int = 256, steps: int = 8000, sigma: float = 0.5) -> dict:
    """S13 Q2 (s13_sap_brainstorm.md): SV-D's necessary condition, a robust chunk autoencoder."""
    _workdir(S03_TOK_NAME)
    js = f"{S03}/s13_chunk_ae_K{K}_dz{dz}.json"
    code, text = _run_logged([sys.executable, "-m", "scripts.sap_chunk_ae", "--tokenizer-dir", S03_TOK,
                              "--data-dir", f"{VOL}/data", "--K", str(K), "--dz", str(dz), "--steps", str(steps),
                              "--sigma", str(sigma), "--out", js], f"{S03}/logs/s13_chunk_ae_K{K}_dz{dz}.log")
    VOLUME.commit()
    return {"rc": code, "tail": text[-2500:]}


@app.local_entrypoint()
def s13_chunk_ae(chunk: int = 8, dz: int = 256, steps: int = 8000, sigma: float = 0.5):
    r = s13_chunk_ae_job.remote(chunk, dz, steps, sigma)
    print(f"rc={r['rc']}\n{r['tail']}")


@app.function(gpu="H100", timeout=3 * 3600, volumes={VOL: VOLUME})
def s13_meanflow_job(K: int = 8, dz: int = 256, steps: int = 4000, rows: int = 32, lr: float = 5e-4,
                     prior_width: int = 512, prior_depth: int = 8, prior_heads: int = 8,
                     val_rows: int = 128) -> dict:
    """S13 Branch A: One-step MeanFlow Prior over ChunkAE latents (true T=L 1-pass generator)."""
    _workdir(S03_TOK_NAME)
    ae_weights = f"{S03}/s13_chunk_ae_K{K}_dz{dz}.pt"
    if not os.path.exists(ae_weights):
        raise FileNotFoundError(f"ChunkAE checkpoint not found: {ae_weights}")
    js = f"{S03}/s13_meanflow_K{K}_dz{dz}_d{prior_depth}.json"
    samples_js = f"{S03}/s13_meanflow_K{K}_dz{dz}_d{prior_depth}_samples.jsonl"
    log = f"{S03}/logs/s13_meanflow_K{K}_dz{dz}_d{prior_depth}.log"
    cmd = [
        sys.executable, "-m", "scripts.sap_meanflow_chunk",
        "--chunk-ae-weights", ae_weights,
        "--tokenizer-dir", S03_TOK,
        "--data-dir", f"{VOL}/data",
        "--K", str(K),
        "--dz", str(dz),
        "--prompt-len", "128",
        "--gen-len", "1920",
        "--prior-width", str(prior_width),
        "--prior-depth", str(prior_depth),
        "--prior-heads", str(prior_heads),
        "--steps", str(steps),
        "--rows", str(rows),
        "--lr", str(lr),
        "--val-rows", str(val_rows),
        "--out", js,
        "--out-samples", samples_js
    ]
    code, text = _run_logged(cmd, log)
    VOLUME.commit()
    return {"rc": code, "tail": text[-3000:], "json_path": js, "samples_path": samples_js}


@app.local_entrypoint()
def s13_meanflow(chunk: int = 8, dz: int = 256, steps: int = 4000, rows: int = 32, lr: float = 5e-4,
                 prior_width: int = 512, prior_depth: int = 8, prior_heads: int = 8,
                 val_rows: int = 128):
    r = s13_meanflow_job.remote(chunk, dz, steps, rows, lr, prior_width, prior_depth, prior_heads, val_rows)
    print(f"rc={r['rc']}\n{r['tail']}")


@app.function(gpu="H100", timeout=1 * 3600, volumes={VOL: VOLUME})
def s13_multistep_eval_job(K: int = 8, dz: int = 256, prior_depth: int = 8, val_rows: int = 128,
                           steps_list: str = "1,2,4,8,16") -> dict:
    """Option 2: Multi-step integration of the trained ChunkMeanFlowPrior."""
    _workdir(S03_TOK_NAME)
    ae_weights = f"{S03}/s13_chunk_ae_K{K}_dz{dz}.pt"
    prior_weights = f"{S03}/s13_meanflow_K{K}_dz{dz}_d{prior_depth}.pt"
    if not os.path.exists(ae_weights):
        raise FileNotFoundError(f"ChunkAE checkpoint not found: {ae_weights}")
    if not os.path.exists(prior_weights):
        raise FileNotFoundError(f"MeanFlowPrior checkpoint not found: {prior_weights}")
    js = f"{S03}/s13_multistep_eval_d{prior_depth}.json"
    log = f"{S03}/logs/s13_multistep_eval_d{prior_depth}.log"
    cmd = [
        sys.executable, "-m", "scripts.sap_eval_multistep_flow",
        "--chunk-ae-weights", ae_weights,
        "--prior-weights", prior_weights,
        "--tokenizer-dir", S03_TOK,
        "--data-dir", f"{VOL}/data",
        "--K", str(K),
        "--dz", str(dz),
        "--prompt-len", "128",
        "--gen-len", "1920",
        "--prior-width", "512",
        "--prior-depth", str(prior_depth),
        "--prior-heads", "8",
        "--val-rows", str(val_rows),
        "--steps-list", steps_list,
        "--out", js,
    ]
    code, text = _run_logged(cmd, log)
    VOLUME.commit()
    return {"rc": code, "tail": text[-3000:], "json_path": js}


@app.local_entrypoint()
def s13_multistep_eval(chunk: int = 8, dz: int = 256, prior_depth: int = 8, val_rows: int = 128,
                       steps_list: str = "1,2,4,8,16"):
    r = s13_multistep_eval_job.remote(chunk, dz, prior_depth, val_rows, steps_list)
    print(f"rc={r['rc']}\n{r['tail']}")


@app.function(gpu="H100", timeout=1 * 3600, volumes={VOL: VOLUME})
def s13_probe_pos_job() -> dict:
    _workdir(S03_TOK_NAME)
    code, text = _run_logged([sys.executable, "-m", "scripts.sap_probe_flow_positions"], f"{S03}/logs/s13_probe_pos.log")
    return {"rc": code, "tail": text[-3000:]}


@app.local_entrypoint()
def s13_probe_pos():
    r = s13_probe_pos_job.remote()
    print(f"rc={r['rc']}\n{r['tail']}")


@app.function(gpu="H100", timeout=1 * 3600, volumes={VOL: VOLUME})
def s13_oracle_bd_job(rows: int = 128) -> dict:
    """Mathematical oracles and empirical gates for Proposals B and D."""
    _workdir(S03_TOK_NAME)
    js = f"{S03}/s13_oracles_bd.json"
    log = f"{S03}/logs/s13_oracles_bd.log"
    cmd = [
        sys.executable, "-m", "scripts.sap_oracles_b_d",
        "--chunk-ae-weights", f"{S03}/s13_chunk_ae_K8_dz256.pt",
        "--prior-weights", f"{S03}/s13_meanflow_K8_dz256_d8.pt",
        "--tokenizer-dir", S03_TOK,
        "--data-dir", f"{VOL}/data",
        "--rows", str(rows),
        "--out", js,
    ]
    code, text = _run_logged(cmd, log)
    VOLUME.commit()
    return {"rc": code, "tail": text[-3000:], "json_path": js}


@app.local_entrypoint()
def s13_oracle_bd(rows: int = 128):
    r = s13_oracle_bd_job.remote(rows)
    print(f"rc={r['rc']}\n{r['tail']}")




@app.function(gpu="H100", timeout=2 * 3600, volumes={VOL: VOLUME})
def s13_chunk_lanes_job(L: int = 16, S: int = 15, K: int = 8, dz: int = 256, steps: int = 3000,
                        rows: int = 32, lr: float = 5e-4, width: int = 512, depth: int = 8,
                        heads: int = 8, val_rows: int = 128) -> dict:
    """Proposal A: Chunk-Lanes (L=16 lanes, S=15 steps, 15-step generation)."""
    _workdir(S03_TOK_NAME)
    ae_weights = f"{S03}/s13_chunk_ae_K{K}_dz{dz}.pt"
    if not os.path.exists(ae_weights):
        raise FileNotFoundError(f"ChunkAE checkpoint not found: {ae_weights}")
    js = f"{S03}/s13_chunk_lanes_L{L}_S{S}_d{depth}.json"
    weights = f"{S03}/s13_chunk_lanes_L{L}_S{S}_d{depth}.pt"
    samples_js = f"{S03}/s13_chunk_lanes_L{L}_S{S}_d{depth}_samples.jsonl"
    log = f"{S03}/logs/s13_chunk_lanes_L{L}_S{S}_d{depth}.log"
    cmd = [
        sys.executable, "-m", "scripts.sap_chunk_lanes",
        "--chunk-ae-weights", ae_weights,
        "--tokenizer-dir", S03_TOK,
        "--data-dir", f"{VOL}/data",
        "--P", "128",
        "--L", str(L),
        "--S", str(S),
        "--K", str(K),
        "--dz", str(dz),
        "--width", str(width),
        "--depth", str(depth),
        "--heads", str(heads),
        "--steps", str(steps),
        "--rows", str(rows),
        "--lr", str(lr),
        "--val-rows", str(val_rows),
        "--out", js,
        "--out-weights", weights,
        "--out-samples", samples_js,
    ]
    code, text = _run_logged(cmd, log)
    VOLUME.commit()
    return {"rc": code, "tail": text[-3000:], "json_path": js, "samples_path": samples_js}


@app.local_entrypoint()
def s13_chunk_lanes(lanes: int = 16, lane_steps: int = 15, chunk: int = 8, dz: int = 256,
                    steps: int = 3000, rows: int = 32, lr: float = 5e-4, width: int = 512,
                    depth: int = 8, heads: int = 8, val_rows: int = 128):
    r = s13_chunk_lanes_job.remote(lanes, lane_steps, chunk, dz, steps, rows, lr, width, depth, heads, val_rows)
    print(f"rc={r['rc']}\n{r['tail']}")


@app.function(gpu="H100", timeout=2 * 3600, volumes={VOL: VOLUME})
def s13_d_ou_flow_job(tau: float = 4.0, K: int = 8, dz: int = 256, steps: int = 3000,
                      rows: int = 32, lr: float = 3e-4, width: int = 512, depth: int = 8,
                      heads: int = 8, val_rows: int = 128) -> dict:
    """Proposal D: Position-Coupled OU Flow Training (full training beyond oracle)."""
    _workdir(S03_TOK_NAME)
    ae_weights = f"{S03}/s13_chunk_ae_K{K}_dz{dz}.pt"
    if not os.path.exists(ae_weights):
        raise FileNotFoundError(f"ChunkAE checkpoint not found: {ae_weights}")
    tag = f"s13_d_ou_flow_tau{int(tau)}_d{depth}"
    js = f"{S03}/{tag}.json"
    weights = f"{S03}/{tag}.pt"
    samples_js = f"{S03}/{tag}_samples.jsonl"
    log = f"{S03}/logs/{tag}.log"
    cmd = [
        sys.executable, "-m", "scripts.sap_train_d_ou_flow",
        "--chunk-ae-path", ae_weights,
        "--tokenizer-dir", S03_TOK,
        "--data-dir", f"{VOL}/data",
        "--out-dir", S03,
        "--tag", tag,
        "--tau", str(tau),
        "--K", str(K),
        "--dz", str(dz),
        "--width", str(width),
        "--depth", str(depth),
        "--heads", str(heads),
        "--steps", str(steps),
        "--batch-size", str(rows),
        "--lr", str(lr),
        "--val-rows", str(val_rows),
    ]
    code, text = _run_logged(cmd, log)
    VOLUME.commit()
    return {"rc": code, "tail": text[-3000:], "json_path": js, "samples_path": samples_js}


@app.local_entrypoint()
def s13_d_ou_flow(tau: float = 4.0, chunk: int = 8, dz: int = 256, steps: int = 3000,
                  rows: int = 32, lr: float = 3e-4, width: int = 512, depth: int = 8,
                  heads: int = 8, val_rows: int = 128):
    r = s13_d_ou_flow_job.remote(tau, chunk, dz, steps, rows, lr, width, depth, heads, val_rows)
    print(f"rc={r['rc']}\n{r['tail']}")


@app.function(gpu="H100", timeout=2 * 3600, volumes={VOL: VOLUME})
def s13_b_bridge_flow_job(W: int = 16, K: int = 8, dz: int = 256, steps: int = 3000,
                          rows: int = 32, lr: float = 3e-4, width: int = 512, depth: int = 8,
                          heads: int = 8, val_rows: int = 128) -> dict:
    """Proposal B: Schrödinger Flow Bridge 2-Pass Plan & Infill Training."""
    _workdir(S03_TOK_NAME)
    ae_weights = f"{S03}/s13_chunk_ae_K{K}_dz{dz}.pt"
    if not os.path.exists(ae_weights):
        raise FileNotFoundError(f"ChunkAE checkpoint not found: {ae_weights}")
    tag = f"s13_b_bridge_flow_W{W}_d{depth}"
    js = f"{S03}/{tag}.json"
    weights = f"{S03}/{tag}.pt"
    samples_js = f"{S03}/{tag}_samples.jsonl"
    log = f"{S03}/logs/{tag}.log"
    cmd = [
        sys.executable, "-m", "scripts.sap_train_b_bridge_flow",
        "--chunk-ae-path", ae_weights,
        "--tokenizer-dir", S03_TOK,
        "--data-dir", f"{VOL}/data",
        "--out-dir", S03,
        "--tag", tag,
        "--W", str(W),
        "--K", str(K),
        "--dz", str(dz),
        "--width", str(width),
        "--depth", str(depth),
        "--heads", str(heads),
        "--steps", str(steps),
        "--batch-size", str(rows),
        "--lr", str(lr),
        "--val-rows", str(val_rows),
    ]
    code, text = _run_logged(cmd, log)
    VOLUME.commit()
    return {"rc": code, "tail": text[-3000:], "json_path": js, "samples_path": samples_js}


@app.local_entrypoint()
def s13_b_bridge_flow(w: int = 16, chunk: int = 8, dz: int = 256, steps: int = 3000,
                      rows: int = 32, lr: float = 3e-4, width: int = 512, depth: int = 8,
                      heads: int = 8, val_rows: int = 128):
    r = s13_b_bridge_flow_job.remote(w, chunk, dz, steps, rows, lr, width, depth, heads, val_rows)
    print(f"rc={r['rc']}\n{r['tail']}")


@app.function(gpu="H100", timeout=2 * 3600, volumes={VOL: VOLUME})
def s13_discrete_chunk_lanes_job(L: int = 16, S: int = 15, K: int = 8, steps: int = 3000,
                                 rows: int = 32, lr: float = 5e-4, width: int = 512, depth: int = 8,
                                 heads: int = 8, val_rows: int = 128) -> dict:
    """Discrete Chunk-Lanes (DCL): 16 Parallel Lanes x 15 Steps with Discrete Multi-Token Chunk Head."""
    _workdir(S03_TOK_NAME)
    tag = f"s13_discrete_chunk_lanes_L{L}_S{S}_d{depth}"
    js = f"{S03}/{tag}.json"
    weights = f"{S03}/{tag}.pt"
    samples_js = f"{S03}/{tag}_samples.jsonl"
    log = f"{S03}/logs/{tag}.log"
    cmd = [
        sys.executable, "-m", "scripts.sap_discrete_chunk_lanes",
        "--tokenizer-dir", S03_TOK,
        "--data-dir", f"{VOL}/data",
        "--out-dir", S03,
        "--tag", tag,
        "--P", "128",
        "--L", str(L),
        "--S", str(S),
        "--K", str(K),
        "--width", str(width),
        "--depth", str(depth),
        "--heads", str(heads),
        "--steps", str(steps),
        "--batch-size", str(rows),
        "--lr", str(lr),
        "--val-rows", str(val_rows),
    ]
    code, text = _run_logged(cmd, log)
    VOLUME.commit()
    return {"rc": code, "tail": text[-3000:], "json_path": js, "samples_path": samples_js}


@app.local_entrypoint()
def s13_discrete_chunk_lanes(lanes: int = 16, lane_steps: int = 15, chunk: int = 8,
                             steps: int = 3000, rows: int = 32, lr: float = 5e-4, width: int = 512,
                             depth: int = 8, heads: int = 8, val_rows: int = 128):
    r = s13_discrete_chunk_lanes_job.remote(lanes, lane_steps, chunk, steps, rows, lr, width, depth, heads, val_rows)
    print(f"rc={r['rc']}\n{r['tail']}")


@app.function(gpu="H100", timeout=30 * 60, volumes={VOL: VOLUME})
def s13_eval_dcl_job(L: int = 16, S: int = 15, K: int = 8, depth: int = 8,
                     width: int = 512, heads: int = 8, temperature: float = 0.8,
                     top_p: float = 0.9, val_rows: int = 128) -> dict:
    """Evaluate already trained Discrete Chunk-Lanes model with temperature/top-p sampling."""
    _workdir(S03_TOK_NAME)
    tag = f"s13_discrete_chunk_lanes_L{L}_S{S}_d{depth}"
    weights = f"{S03}/{tag}.pt"
    if not os.path.exists(weights):
        raise FileNotFoundError(f"Checkpoint not found: {weights}")
    eval_tag = f"{tag}_temp{temperature}_top{top_p}"
    js = f"{S03}/{eval_tag}.json"
    samples_js = f"{S03}/{eval_tag}_samples.jsonl"
    log = f"{S03}/logs/{eval_tag}.log"
    cmd = [
        sys.executable, "-m", "scripts.sap_discrete_chunk_lanes",
        "--tokenizer-dir", S03_TOK,
        "--data-dir", f"{VOL}/data",
        "--out-dir", S03,
        "--tag", eval_tag,
        "--eval-only",
        "--load-weights", weights,
        "--temperature", str(temperature),
        "--top-p", str(top_p),
        "--P", "128",
        "--L", str(L),
        "--S", str(S),
        "--K", str(K),
        "--width", str(width),
        "--depth", str(depth),
        "--heads", str(heads),
        "--val-rows", str(val_rows),
    ]
    code, text = _run_logged(cmd, log)
    VOLUME.commit()
    return {"rc": code, "tail": text[-3000:], "json_path": js, "samples_path": samples_js, "eval_tag": eval_tag}


@app.local_entrypoint()
def s13_eval_dcl(lanes: int = 16, lane_steps: int = 15, chunk: int = 8, depth: int = 8,
                 width: int = 512, heads: int = 8, temperature: float = 0.8,
                 top_p: float = 0.9, val_rows: int = 128):
    r = s13_eval_dcl_job.remote(lanes, lane_steps, chunk, depth, width, heads, temperature, top_p, val_rows)
    print(f"rc={r['rc']}\n{r['tail']}")


@app.function(gpu="H100", timeout=2 * 3600, volumes={VOL: VOLUME})
def s13_ar_chunk_job(K: int = 8, dz: int = 256, steps: int = 2000, rows: int = 32, lr: float = 5e-4,
                     width: int = 512, depth: int = 8, heads: int = 8, val_rows: int = 128) -> dict:
    """Option 1: Autoregressive Continuous Latent Chunk Transformer (CALM-style)."""
    _workdir(S03_TOK_NAME)
    ae_weights = f"{S03}/s13_chunk_ae_K{K}_dz{dz}.pt"
    if not os.path.exists(ae_weights):
        raise FileNotFoundError(f"ChunkAE checkpoint not found: {ae_weights}")
    js = f"{S03}/s13_ar_chunk_K{K}_dz{dz}_d{depth}.json"
    weights = f"{S03}/s13_ar_chunk_K{K}_dz{dz}_d{depth}.pt"
    samples_js = f"{S03}/s13_ar_chunk_K{K}_dz{dz}_d{depth}_samples.jsonl"
    log = f"{S03}/logs/s13_ar_chunk_K{K}_dz{dz}_d{depth}.log"
    cmd = [
        sys.executable, "-m", "scripts.sap_ar_chunk",
        "--chunk-ae-weights", ae_weights,
        "--tokenizer-dir", S03_TOK,
        "--data-dir", f"{VOL}/data",
        "--K", str(K),
        "--dz", str(dz),
        "--prompt-len", "128",
        "--gen-len", "1920",
        "--width", str(width),
        "--depth", str(depth),
        "--heads", str(heads),
        "--steps", str(steps),
        "--rows", str(rows),
        "--lr", str(lr),
        "--val-rows", str(val_rows),
        "--out", js,
        "--out-weights", weights,
        "--out-samples", samples_js,
    ]
    code, text = _run_logged(cmd, log)
    VOLUME.commit()
    return {"rc": code, "tail": text[-3000:], "json_path": js, "samples_path": samples_js}


@app.local_entrypoint()
def s13_ar_chunk(chunk: int = 8, dz: int = 256, steps: int = 2000, rows: int = 32, lr: float = 5e-4,
                 width: int = 512, depth: int = 8, heads: int = 8, val_rows: int = 128):
    r = s13_ar_chunk_job.remote(chunk, dz, steps, rows, lr, width, depth, heads, val_rows)
    print(f"rc={r['rc']}\n{r['tail']}")


@app.function(gpu="H100", timeout=2 * 3600, volumes={VOL: VOLUME})
def s13_class_skeleton_job(steps: int = 3000, rows: int = 32, lr: float = 1e-3, val_rows: int = 128) -> dict:
    """S13 Branch B: SV-B Class Skeleton autoregressive check."""
    _workdir(S03_TOK_NAME)
    js = f"{S03}/s13_class_skeleton.json"
    log = f"{S03}/logs/s13_class_skeleton.log"
    cmd = [
        sys.executable, "-m", "scripts.sap_class_skeleton",
        "--tokenizer-dir", S03_TOK,
        "--data-dir", f"{VOL}/data",
        "--class-map", f"{VOL}/out/s13/brown256.npy",
        "--prompt-len", "128",
        "--steps", str(steps),
        "--rows", str(rows),
        "--lr", str(lr),
        "--val-rows", str(val_rows),
        "--out", js
    ]
    code, text = _run_logged(cmd, log)
    VOLUME.commit()
    return {"rc": code, "tail": text[-2500:], "json_path": js}


@app.local_entrypoint()
def s13_class_skeleton(steps: int = 3000, rows: int = 32, lr: float = 1e-3, val_rows: int = 128):
    r = s13_class_skeleton_job.remote(steps, rows, lr, val_rows)
    print(f"rc={r['rc']}\n{r['tail']}")


@app.function(gpu="H100", timeout=2 * 3600, volumes={VOL: VOLUME})
def s13_roofline_job(depths: str = "8,20", batches: str = "1,16,64", rounds: str = "8,16,32,64",
                     gen_tokens: int = 1920, prompt: int = 128) -> dict:
    """S13 Q3 (s13_sap_brainstorm.md): is one pass worth it? On random-weight models of each depth
    (timing depends on shape and schedule, not weights): next-token decoding, lockstep lanes at R
    rounds, and one pass over all generated rows, under CUDA graphs, on one H100."""
    _workdir(S03_TOK_NAME)
    out = {}
    for D in [int(d) for d in depths.split(",")]:
        js = f"{S03}/s13_roofline_d{D}.json"
        code, text = _run_logged([sys.executable, "-m", "scripts.sap_decode_bench", "--roofline", str(D),
                                  "--gen-tokens", str(gen_tokens), "--prompt-len", str(prompt), "--rounds", rounds,
                                  "--batch-sizes", *batches.split(","), "--graph-temperature", "1.0",
                                  "--repeats", "3", "--out", js], f"{S03}/logs/s13_roofline_d{D}.log")
        out[D] = {"rc": code, "tail": text[-2500:]}
    VOLUME.commit()
    return out


@app.local_entrypoint()
def s13_roofline(depths: str = "8,20", batches: str = "1,16,64", rounds: str = "8,16,32,64"):
    res = s13_roofline_job.remote(depths, batches, rounds)
    for D, r in res.items():
        print(f"d{D} rc={r['rc']}\n{r['tail']}")


@app.function(gpu="H100", timeout=3 * 3600, volumes={VOL: VOLUME})
def s11_speed_job(ckpt: str, schedules: list, depth: int = 4, prefix: int = 128, batches: str = "1,16,64") -> dict:
    """CUDA-graph two-stream decoding of the block (one captured graph per step) against CUDA-graph
    next-token decoding of the same number of tokens, on one checkpoint, in one container so every
    ratio shares a GPU. Timing depends on the model shape and the step schedule, not on the weights,
    so one checkpoint of the d<depth> shape times every schedule. schedules: ["bl:L:n" | "wb:n"]."""
    _workdir(S03_TOK_NAME)
    out = {}
    for sch in schedules:
        kind, *v = sch.split(":")
        if kind == "ln":                                  # S08 plain lanes: the first token, then L lanes
            flags = ["--lanes", v[0], "--prompt-len", str(prefix), "--gen-tokens", "1921"]
        else:
            flags = ["--wb-window", v[-1], "--wb-prefix", str(prefix)] + (["--wb-lanes", v[0]] if kind == "bl" else [])
        js = f"{S03}/s11speed_{ckpt}_{sch.replace(':', '_')}.json"
        code, _ = _run_logged([sys.executable, "-m", "scripts.sap_decode_bench", "--checkpoint-dir",
                               f"{S03}/d{depth}/{ckpt}", "--tokenizer-dir", S03_TOK, *flags,
                               "--batch-sizes", *batches.split(","), "--graph-temperature", "1.0", "--out", js],
                              f"{S03}/logs/s11speed_{ckpt}_{sch.replace(':', '_')}.log")
        out[sch] = json.load(open(js)) if code == 0 and os.path.exists(js) else {"rc": code}
    VOLUME.commit()
    return out


@app.local_entrypoint()
def s11_speed(ckpt: str = "S11bl64n4x1_s1", schedules: str = "bl:64:4,bl:32:8,bl:128:4,bl:16:8,wb:16,wb:4",
              depth: int = 4, prefix: int = 128, batches: str = "1,16,64"):
    """Decode speed of S11 step schedules on an H100 (the bar needs measured speed, not step counts)."""
    res = s11_speed_job.remote(ckpt, [x for x in schedules.split(",") if x], depth, prefix, batches)
    for sch, r in res.items():
        if "rows" not in r:
            print(f"{sch}: failed {r}")
            continue
        steps = r.get("steps", r.get("lane_len"))
        print(f"{sch}: {steps} steps for {r['gen_tokens']} tokens")
        for row in r["rows"]:
            fast = row.get("graph_wb_tok_per_s", row.get("graph_lane_tok_per_s"))
            print(f"  batch {row['batch']:3d}: next-token {row['graph_ar_tok_per_s']:10,.0f} tok/s | parallel "
                  f"{fast:10,.0f} tok/s | {row['graph_speedup']:.2f}x")


@app.function(gpu="H100", timeout=3 * 3600, volumes={VOL: VOLUME})
def s11_gen_job(tag: str, sched: str, depth: int, ar_tag: str, ref_tag: str, n_prefixes: int = 256,
                prefix: int = 128, gen_tokens: int = 1920, temperature: float = 1.0, ar_temperature: float = -1.0,
                ar_top_p: float = 1.0) -> dict:
    """Reference perplexity and distinct 3-grams of two-stream samples (sched "bl:L:n" or "wb:n")
    against another model's next-token samples, both scored by a third dense model, on the same
    prefixes, at temperature 1. The row (prefix + gen_tokens) matches the schedule trained on."""
    _workdir(S03_TOK_NAME)
    kind, *v = sched.split(":")
    flags = ["--wb-window", v[-1]] + (["--wb-lanes", v[0]] if kind == "bl" else [])
    ck = lambda t: f"{S03}/d{depth}/{t}"
    code, text = _run_logged([sys.executable, "-m", "scripts.sap_eval_generation", "--checkpoint-dir", ck(tag),
                              "--ar-dir", ck(ar_tag), "--reference-dir", ck(ref_tag), "--tokenizer-dir", S03_TOK,
                              "--data-dir", f"{VOL}/data", *flags, "--n-prefixes", str(n_prefixes), "--batch", "16",
                              "--prefix-len", str(prefix), "--gen-tokens", str(gen_tokens), "--real",
                              "--temperature", str(temperature),
                              *(["--ar-temperature", str(ar_temperature)] if ar_temperature > 0 else []),
                              *(["--ar-top-p", str(ar_top_p)] if ar_top_p < 1 else []),
                              "--out", f"{S03}/s11gen_{tag}_t{temperature:g}_{ar_tag}_t{ar_temperature:g}_p{ar_top_p:g}.jsonl"],
                             f"{S03}/logs/s11gen_{tag}_t{temperature:g}_{ar_tag}_t{ar_temperature:g}_p{ar_top_p:g}.log")
    out = {"tag": tag, "sched": sched, "temperature": temperature, "ar_tag": ar_tag,
           "ar_temperature": ar_temperature if ar_temperature > 0 else temperature, "ar_top_p": ar_top_p, "rc": code}
    m = re.search(r"reference ppl\s+next-token\s+([0-9.]+) \| block\s+([0-9.]+)", text)
    if m:
        out["ref_ppl_ar"], out["ref_ppl_two_stream"] = float(m.group(1)), float(m.group(2))
    m = re.search(r"distinct 3-grams\s+next-token ([0-9.]+) \| block ([0-9.]+)", text)
    if m:
        out["distinct3_ar"], out["distinct3_two_stream"] = float(m.group(1)), float(m.group(2))
    m = re.search(r"unigram entropy\s+next-token ([0-9.]+) \| block ([0-9.]+)", text)
    if m:
        out["entropy_ar"], out["entropy_two_stream"] = float(m.group(1)), float(m.group(2))
    m = re.search(r"real continuation\s+reference ppl\s+([0-9.]+) \| distinct 3-grams ([0-9.]+) \| unigram entropy ([0-9.]+)", text)
    if m:
        out["ref_ppl_real"], out["distinct3_real"], out["entropy_real"] = (float(g) for g in m.groups())
    VOLUME.commit()
    return out


@app.local_entrypoint()
def s11_gen(tags: str = "S11bl64n4x4_s1:bl:64:4,S11bl32n8x4_s1:bl:32:8", depth: int = 4,
            ar_tag: str = "S11dense_x1_s1", ref_tag: str = "S11dense_x4_s2", n_prefixes: int = 256,
            temperatures: str = "1.0", ar_settings: str = ""):
    """Sample quality of S11 checkpoints (tags: comma list of 'TAG:bl:L:n' or 'TAG:wb:n') against
    next-token samples, scored by a held-out dense model, at each temperature (the quality-
    diversity frontier: compare reference ppl at matched unigram entropy). ar_settings, if given,
    sweeps the next-token side alone: comma list of 'AR_TAG:TEMPERATURE:TOP_P', each run with
    the parallel model at the first of `temperatures`."""
    jobs = []
    for t in [x for x in tags.split(",") if x]:
        tag, sched = t.split(":", 1)
        temps = [float(v) for v in temperatures.split(",") if v]
        if ar_settings:
            for a in [x for x in ar_settings.split(",") if x]:
                at, atemp, ap = a.split(":")
                jobs.append((tag, sched, depth, at, ref_tag, n_prefixes, 128, 1920, temps[0], float(atemp), float(ap)))
        else:
            for temp in temps:
                jobs.append((tag, sched, depth, ar_tag, ref_tag, n_prefixes, 128, 1920, temp))
    for res in s11_gen_job.starmap(jobs, return_exceptions=True):
        print(res)


@app.local_entrypoint()
def s11_score(tags: str, depth: int = 4, prefix: int = 128, rows: int = 256, ref: str = "S11dense_x1_s1",
              name: str = "rescore"):
    """Per-position bpb of existing S11 checkpoints (no training), each scored in its own order,
    which the tag names: S11dense_* (left to right), S11bl{L}n{n}x* (bridged lanes), S11wb{n}x*
    (window bisection), S11ln{L}x* (plain lanes). Ratios are to `ref` on identical targets."""
    def spec(t):
        m = re.match(r"S11bl(\d+)n(\d+)x", t)
        if m:
            return f":bl{m.group(1)}_{m.group(2)}"
        m = re.match(r"S11wb(\d+)x", t)
        if m:
            return f":wb{m.group(1)}"
        m = re.match(r"S11sd(\d+)w(\d+)x", t)
        if m:
            return f":sd{m.group(1)}_{m.group(2)}"
        m = re.match(r"S11(ln|lo|sd)(\d+)x", t)
        if m:
            return f":{m.group(1)}{m.group(2)}"
        assert t.startswith("S11dense"), t
        return ""
    ts = [t for t in tags.split(",") if t]
    order = ([ref] if ref in ts else []) + [t for t in ts if t != ref]
    models = [f"--model={t}:{S03}/d{depth}/{t}{spec(t)}" for t in order]
    edges = f"0,{prefix},{prefix + 128},{prefix + 384},1024,2047"
    out_name = f"s11_score_d{depth}_{name}"
    res = s03_job.remote(out_name, ["scripts.sap_position_bpb", "--tokenizer-dir", S03_TOK, "--data-dir",
                                    f"{VOL}/data", "--rows", str(rows), "--wb-prefix", str(prefix), "--edges", edges,
                                    "--out", f"{S03}/{out_name}.json", *models])
    print(res["tail"][-4000:])


D16_REF = f"{VOL}/out/p10_isotoken/d16/ISO_mst_ve_gattn_s1/depth_16/ckpt_base/base"   # d16, val bpb 0.908


@app.local_entrypoint()
def s11_rescore_samples(files: str, ref: str = D16_REF, name: str = "d16"):
    """Re-score saved generation samples (comma list of jsonl names under out/s03_sap) under a
    stronger reference, with each prompt's true continuation scored by the same model."""
    paths = [f"{S03}/{f}" for f in files.split(",") if f]
    res = s03_job.remote(f"s11_rescore_{name}", ["scripts.sap_rescore_samples", "--reference-dir", ref,
                                                  "--tokenizer-dir", S03_TOK, "--data-dir", f"{VOL}/data",
                                                  "--jsonl", *paths, "--out", f"{S03}/s11_rescore_{name}.json"])
    print(res["tail"][-3000:])


@app.local_entrypoint()
def s11_lane_start_oracle(lanes: str = "ln64x4=S11ln64x4_s1:64,ln32x4=S11ln32x4_s1:32,ln64x1=S11ln64x1_s1:64",
                          dense: str = "S11dense_x1_s1", depth: int = 4, rows: int = 256):
    """Offline oracle for boundary-aligned lanes: plain-lanes start cost split by the junction
    token's class (document, paragraph, sentence boundary, or mid-sentence)."""
    specs = []
    for t in [x for x in lanes.split(",") if x]:
        name, rest = t.split("=")
        tag, L = rest.split(":")
        specs += ["--lanes", f"{name}={S03}/d{depth}/{tag}:{L}"]
    res = s03_job.remote(f"s11_lane_start_oracle_d{depth}", ["scripts.sap_lane_start_oracle", "--dense-dir",
                                                              f"{S03}/d{depth}/{dense}", "--tokenizer-dir", S03_TOK,
                                                              "--data-dir", f"{VOL}/data", "--rows", str(rows), *specs,
                                                              "--out", f"{S03}/s11_lane_start_oracle_d{depth}.json"])
    print(res["tail"][-3000:])


@app.local_entrypoint()
def s12_aligned_eval(models: str = "dense=S11dense_x1_s1,ln:64=S11ln64x1_s1,ln:32=S11ln32x1_s1,la:64:15=S11la64w15x1_s1,"
                                   "la:32:30=S11la32w30x1_s1", depth: int = 4, rows: int = 256, prefix: int = 128):
    """S12 S-1 gate: block bpb of dense, plain lanes and sentence-aligned lanes on the same rows."""
    specs = []
    for t in [x for x in models.split(",") if x]:
        kind, tag = t.split("=")
        specs.append(f"--model={kind}={S03}/d{depth}/{tag}")
    res = s03_job.remote(f"s12_aligned_eval_d{depth}", ["scripts.sap_aligned_lanes_eval", "--tokenizer-dir", S03_TOK,
                                                         "--data-dir", f"{VOL}/data", "--rows", str(rows), "--prefix",
                                                         str(prefix), *specs, "--out", f"{S03}/s12_aligned_eval_d{depth}.json"])
    print(res["tail"][-3000:])


S09 = f"{VOL}/out/s09_sap"


@app.function(gpu="L4", timeout=3 * 3600, volumes={VOL: VOLUME})
def s09_flow_run(arm: str, seed: int, argv: list) -> dict:
    """One S07-hypothesis-A toy gate run (scripts/sap_s06_screen.py, oracle context)."""
    _workdir()
    from scripts.sap_s06_screen import build_parser, run
    os.makedirs(f"{S09}/rows", exist_ok=True)
    tag = "_".join(argv[i + 1] for i, a in enumerate(argv) if a in ("--flow-bins", "--steps"))
    args = build_parser().parse_args(["--arm", arm, "--seed", str(seed), "--device", "cuda",
                                      "--out", f"{S09}/rows/{arm}_s{seed}_{tag}.json", *argv])
    row = run(args)
    VOLUME.commit()
    return row


@app.local_entrypoint()
def s09_flow_toy(arms: str = "scan_flow,coupling_flow", seeds: str = "0,1", steps: int = 8000, block_t: int = 4,
                 width: int = 128, batch: int = 96, layers: int = 8, cond_depth: int = 2, iwae_k: int = 128,
                 bins: int = 0, out: str = "out/s09_flow_toy"):
    """Phrase-HMM toy gate for the scan-coupled categorical flow against its no-scan control
    (pre-registered in s09_sap_tl_gates.md): pass = IWAE block-KL upper bound <= 0.10 and invalid
    <= 3%; the scan must beat the control by >= 10% KL to count as the mechanism. (block_t, not T:
    Modal's CLI lowercases parameter names.)"""
    T = block_t
    argv = ["--steps", str(steps), "--T", str(T), "--width", str(width), "--batch", str(batch),
            "--flow-layers", str(layers), "--flow-cond-depth", str(cond_depth), "--iwae-k", str(iwae_k),
            "--flow-bins", str(bins)]
    jobs = [(a, int(sd), argv) for a in arms.split(",") if a for sd in seeds.split(",") if sd]
    rows = []
    for res in s09_flow_run.starmap(jobs, return_exceptions=True):
        if isinstance(res, BaseException):
            print(f"  run failed: {res!r}")
            continue
        rows.append(res)
        print(f"  {res['arm']:14s} seed {res['seed']}: KL<= {res['block_kl']:.4f} invalid {res['invalid_rate']:.4f} "
              f"unique {res['unique_rate']:.3f} | {res['verdict']} | {res['seconds']:.0f}s")
    by = {}
    for r in rows:
        by.setdefault(r["arm"], []).append(r["block_kl"])
    for a, v in by.items():
        print(f"{a}: mean KL upper bound {sum(v) / len(v):.4f} over {len(v)} seeds")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, f"rows_T{T}_bins{bins}.json"), "w") as f:
        json.dump(rows, f, indent=2)


S11 = f"{VOL}/out/s11_sap"


@app.function(gpu="L4", timeout=4 * 3600, volumes={VOL: VOLUME})
def s11_toy_run(arm: str, seed: int, argv: list) -> dict:
    """One Bridge LM toy gate run (scripts/sap_s06_screen.py, oracle context)."""
    _workdir()
    from scripts.sap_s06_screen import build_parser, run
    os.makedirs(f"{S11}/rows", exist_ok=True)
    tag = "_".join(a.strip("-") + argv[i + 1] for i, a in enumerate(argv) if a in ("--T", "--steps", "--bridge-codes"))
    tag += ("_ep" if "--bridge-endpoints" in argv else "") + ("_poe" if "--bridge-poe" in argv else "")
    if "--bridge-window" in argv:
        tag += "_w" + argv[argv.index("--bridge-window") + 1]
    args = build_parser().parse_args(["--arm", arm, "--seed", str(seed), "--device", "cuda",
                                      "--out", f"{S11}/rows/{arm}_s{seed}_{tag}.json", *argv])
    row = run(args)
    VOLUME.commit()
    return row


@app.local_entrypoint()
def s11_toy(arms: str = "bridge,bridge_tok", seeds: str = "0,1", steps: int = 8000, block_ts: str = "4,16,64",
            width: int = 128, depth: int = 4, batch: int = 96, codes: int = 512, ar_steps: int = 4000,
            endpoints: bool = False, poe: bool = False, window: int = 1, out: str = "out/s11_toy"):
    """Phase 2 of s11_sap_tl_brainstorm.md (pre-registered): Bridge LM on the phrase-HMM toy.
    Pass: KL <= 0.10 and invalid <= 3% at T=4; T=L viability: KL <= 0.5 at T=64; kill if KL > 5 at
    T=64. bridge_ar runs ar_steps of stage 1 inside --steps."""
    jobs = []
    for T in [int(t) for t in block_ts.split(",") if t]:
        argv = ["--steps", str(steps), "--T", str(T), "--width", str(width), "--depth", str(depth),
                "--batch", str(batch), "--bridge-codes", str(codes), "--bridge-ar-steps", str(ar_steps)]
        if endpoints:
            argv.append("--bridge-endpoints")
        if poe:
            argv.append("--bridge-poe")
        if window > 1:
            argv += ["--bridge-window", str(window)]
        jobs += [(a, int(sd), argv) for a in arms.split(",") if a for sd in seeds.split(",") if sd]
    rows = []
    for res in s11_toy_run.starmap(jobs, return_exceptions=True):
        if isinstance(res, BaseException):
            print(f"  run failed: {res!r}")
            continue
        rows.append(res)
        nll = res.get("sample_nll_valid")
        print(f"  {res['arm']:11s} T={res['T']:<3d} seed {res['seed']}: KL<= {res['block_kl']:.3f} invalid "
              f"{res['invalid_rate']:.4f} valid-sample nll {nll if nll is None else round(nll, 2)} "
              f"(true entropy {res['true_entropy']:.2f}) steps {res.get('parallel_steps')} | {res['verdict']} "
              f"| {res['seconds']:.0f}s")
    by = {}
    for r in rows:
        by.setdefault((r["T"], r["arm"]), []).append(r)
    print("\nmeans over seeds: T arm | KL bound | invalid | valid-sample nll - entropy")
    for (T, a), rs in sorted(by.items()):
        m = lambda k: sum(r[k] for r in rs) / len(rs)
        gap = sum((r["sample_nll_valid"] or float("nan")) - r["true_entropy"] for r in rs) / len(rs)
        print(f"  T={T:<3d} {a:11s} | {m('block_kl'):.3f} | {m('invalid_rate'):.4f} | {gap:+.2f}  ({len(rs)} seeds)")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, f"rows_{arms.replace(',', '-')}_T{block_ts.replace(',', '-')}_s{steps}"
                                f"{'_ep' if endpoints else ''}{'_poe' if poe else ''}"
                                f"{f'_w{window}' if window > 1 else ''}.json"), "w") as f:
        json.dump(rows, f, indent=2)


S10 = f"{VOL}/out/s10_sap"


@app.function(gpu="L4", timeout=3 * 3600, volumes={VOL: VOLUME})
def s10_toy_run(arm: str, seed: int, argv: list) -> dict:
    """One RC-PTP toy gate run (scripts/sap_s06_screen.py, oracle context)."""
    _workdir()
    from scripts.sap_s06_screen import build_parser, run
    os.makedirs(f"{S10}/rows", exist_ok=True)
    tag = "_".join(a.strip("-") + argv[i + 1] for i, a in enumerate(argv)
                   if a in ("--T", "--depth", "--steps", "--n-inv", "--ptp-coupling", "--ptp-stages"))
    args = build_parser().parse_args(["--arm", arm, "--seed", str(seed), "--device", "cuda",
                                      "--out", f"{S10}/rows/{arm}_s{seed}_{tag}.json", *argv])
    row = run(args)
    VOLUME.commit()
    return row


@app.local_entrypoint()
def s10_toy(arms: str = "rcptp,cptp_si,rcptp_indep,rcptp_idorder", seeds: str = "0,1", steps: int = 8000,
            block_ts: str = "4,16,64", depth: int = 4, deep: int = 8, deep_t: int = 64, width: int = 128,
            batch: int = 96, iwae_k: int = 128, n_inv: int = 2, coupling: str = "cdf", stages: int = 2,
            out: str = "out/s10_toy"):
    """Phase 1 of s10_sap_tl_brainstorm.md, pre-registered: pass = generator block-KL upper bound
    <= 0.10 and invalid <= 3% at T=4; each mechanism counts only if the full arm beats its ablation
    by >= 10% KL at T=16 or 64; T=L viability = KL <= 0.5 at T=64 (depth 4) or a clear gain at depth
    8. Depth `deep` runs only at T=`deep_t`. (block_ts, not T: Modal's CLI lowercases names.)"""
    jobs = []
    for T in [int(t) for t in block_ts.split(",") if t]:
        for d in [depth] + ([deep] if deep and T == deep_t else []):
            argv = ["--steps", str(steps), "--T", str(T), "--depth", str(d), "--width", str(width),
                    "--batch", str(batch), "--iwae-k", str(iwae_k), "--n-inv", str(n_inv),
                    "--ptp-coupling", coupling, "--ptp-stages", str(stages)]
            jobs += [(a, int(sd), argv) for a in arms.split(",") if a for sd in seeds.split(",") if sd]
    rows = []
    for res in s10_toy_run.starmap(jobs, return_exceptions=True):
        if isinstance(res, BaseException):
            print(f"  run failed: {res!r}")
            continue
        rows.append(res)
        print(f"  {res['arm']:14s} T={res['T']:<3d} d{res['depth']} seed {res['seed']}: KL<= {res['block_kl']:.4f} "
              f"(AR mode {res['ar_block_kl']:.4f}) invalid {res['invalid_rate']:.4f} agree {res['agree_final']:.3f} "
              f"lead {res['lead_correct']:.2f} | {res['verdict']} | {res['seconds']:.0f}s")
    by = {}
    for r in rows:
        by.setdefault((r["T"], r["depth"], r["arm"]), []).append(r)
    print("\nmeans over seeds: T depth arm | KL upper bound | AR-mode KL | invalid | agree | lead")
    for (T, d, a), rs in sorted(by.items()):
        m = lambda k: sum(r[k] for r in rs) / len(rs)
        print(f"  T={T:<3d} d{d} {a:14s} | {m('block_kl'):.4f} | {m('ar_block_kl'):.4f} | {m('invalid_rate'):.4f} "
              f"| {m('agree_final'):.3f} | {m('lead_correct'):.2f}  ({len(rs)} seeds)")
    for (T, d, a), rs in sorted(by.items()):
        full = by.get((T, d, "rcptp"))
        if a != "rcptp" and full:
            kf = sum(r["block_kl"] for r in full) / len(full)
            ka = sum(r["block_kl"] for r in rs) / len(rs)
            print(f"  T={T} d{d}: rcptp beats {a} by {100 * (ka - kf) / ka:+.1f}% KL (needs >= +10%)")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, f"rows_{arms.replace(',', '-')}_{coupling}_st{stages}_T{block_ts.replace(',', '-')}_d{depth}.json"), "w") as f:
        json.dump(rows, f, indent=2)


# ----------------------------------------------------------------------------- S04: strict one-shot positional plan field
S04 = f"{VOL}/out/s04_sap"


@app.function(timeout=10 * 60, volumes={VOL: VOLUME})
def s04_compile(rows: list, smoke: bool = False) -> str:
    """Persist the pre-registered toy gate in one inspectable volume artifact."""
    _workdir()
    os.makedirs(S04, exist_ok=True)
    suffix = "_smoke" if smoke else ""
    path = f"{S04}/s04_field_cp{suffix}_compiled.log"
    with open(path, "w") as f:
        f.write("S04 exact one-shot positional plan field\n")
        f.write("One categorical plan draw; one fixed-depth head; all token draws parallel.\n")
        f.write("Pre-registered gate: T=4 block KL <= 0.10 and invalid rate <= 0.03.\n\n")
        f.write(json.dumps(rows, indent=2, sort_keys=True, default=str))
        f.write("\n")
    with open(f"{S04}/summary{suffix}.json", "w") as f:
        json.dump(rows, f, indent=2, sort_keys=True, default=str)
    VOLUME.commit()
    return path


@app.local_entrypoint()
def s04(steps: int = 8000, seeds: int = 1, smoke: bool = False,
        components: int = 16, field_rank: int = 8,
        out: str = "out/s04_sap_modal"):
    """Synthetic kill gate for the strict one-shot field mixture before any d8 spend."""
    from scripts.sap_synthetic import gate_verdicts

    if smoke:
        steps, seeds = 80, 1
        out += "_smoke"
    base = ["--steps", str(steps), "--T", "4", "--cp-components", str(components),
            "--field-rank", str(field_rank), "--eval-milestones", str(steps),
            "--log-every", str(max(1, steps // 4))]
    if smoke:
        base += ["--eval-seqs", "16", "--samples-per-ctx", "2", "--iwae-samples", "4"]
    jobs = [("field_cp", 4, seed, base, "s04_field_cp") for seed in range(seeds)]
    print(f"S04: {len(jobs)} strict one-shot field job(s); C={components}, rank={field_rank}")
    rows = []
    for res in stage_a_run.starmap(jobs, return_exceptions=True):
        if isinstance(res, BaseException):
            print(f"  run failed: {res!r}")
            continue
        rows.append(res)
        print(f"  seed {res['seed']}: KL={res.get('block_kl')} invalid={res.get('invalid_rate')} "
              f"sensitivity={res.get('sensitivity')}")
    print("\n".join(["", "S04 gate", *gate_verdicts(rows)]))
    os.makedirs(out, exist_ok=True)
    local = os.path.join(out, "summary.json")
    with open(local, "w") as f:
        json.dump(rows, f, indent=2, sort_keys=True)
    compiled = s04_compile.remote(rows, smoke)
    print(f"summary: {local}\ncompiled volume log: {compiled}")


# ----------------------------------------------------------------------------- S05: strict one-shot correlated noise field
S05 = f"{VOL}/out/s05_sap"


@app.function(timeout=10 * 60, volumes={VOL: VOLUME})
def s05_compile(rows: list, smoke: bool = False) -> str:
    """Persist the pre-registered hard-sample correlated-field gate."""
    _workdir()
    os.makedirs(S05, exist_ok=True)
    suffix = "_smoke" if smoke else ""
    path = f"{S05}/s05_field_energy{suffix}_compiled.log"
    with open(path, "w") as f:
        f.write("S05 strict one-round correlated noise field\n")
        f.write("One shared Gaussian field draw; one fixed-depth head; all token draws parallel.\n")
        f.write("Training: K=4 prior marginal likelihood + energy score on hard token samples.\n")
        f.write("Pre-registered gate: T=4 block KL <= 0.10 and invalid rate <= 0.03.\n\n")
        f.write(json.dumps(rows, indent=2, sort_keys=True, default=str))
        f.write("\n")
    with open(f"{S05}/summary{suffix}.json", "w") as f:
        json.dump(rows, f, indent=2, sort_keys=True, default=str)
    VOLUME.commit()
    return path


@app.local_entrypoint()
def s05(steps: int = 8000, seeds: int = 1, smoke: bool = False,
        field_rank: int = 8, field_samples: int = 4,
        energy_weight: float = 0.25, field_topk: int = 16,
        out: str = "out/s05_sap_modal"):
    """Toy kill gate for a genuinely one-round correlated sampler before any d8 spend."""
    from scripts.sap_synthetic import gate_verdicts

    if smoke:
        steps, seeds = 80, 1
        out += "_smoke"
    base = ["--steps", str(steps), "--T", "4", "--field-rank", str(field_rank),
            "--field-samples", str(field_samples), "--field-energy-weight", str(energy_weight),
            "--field-topk", str(field_topk), "--eval-milestones", str(steps),
            "--log-every", str(max(1, steps // 4))]
    if smoke:
        base += ["--eval-seqs", "16", "--samples-per-ctx", "2", "--iwae-samples", "4"]
    jobs = [("field_energy", 4, seed, base, "s05_field_energy") for seed in range(seeds)]
    print(f"S05: {len(jobs)} strict one-round field job(s); q={field_rank}, "
          f"Ktrain={field_samples}, energy={energy_weight}, topk={field_topk}")
    rows = []
    for res in stage_a_run.starmap(jobs, return_exceptions=True):
        if isinstance(res, BaseException):
            print(f"  run failed: {res!r}")
            continue
        rows.append(res)
        print(f"  seed {res['seed']}: KL={res.get('block_kl')} invalid={res.get('invalid_rate')} "
              f"sensitivity={res.get('sensitivity')}")
    print("\n".join(["", "S05 gate", *gate_verdicts(rows)]))
    os.makedirs(out, exist_ok=True)
    local = os.path.join(out, "summary.json")
    with open(local, "w") as f:
        json.dump(rows, f, indent=2, sort_keys=True)
    compiled = s05_compile.remote(rows, smoke)
    print(f"summary: {local}\ncompiled volume log: {compiled}")


# ----------------------------------------------------------------------------- S06-Q: ten one-round mechanisms
S06 = f"{VOL}/out/s06_sap"


@app.function(gpu="L4", timeout=3 * 3600, volumes={VOL: VOLUME})
def s06_screen_run(arm: str, argv: list) -> dict:
    """One isolated L4 job for one S06-Q mechanism."""
    _workdir()
    from scripts.sap_s06_screen import build_parser, run
    os.makedirs(f"{S06}/rows", exist_ok=True)
    args = build_parser().parse_args([
        "--arm", arm, "--device", "cuda", "--out", f"{S06}/rows/{arm}.json", *argv,
    ])
    row = run(args)
    VOLUME.commit()
    return row


def _s06_text(rows: list, config: dict) -> str:
    lines = [
        "S06-Q — one-round SAP mechanism screen",
        "========================================",
        "Oracle context: exact phrase-HMM filtering belief (mechanism test, not end-to-end).",
        "All primitive randomness is drawn once; no teacher, distillation, AR verifier,",
        "rejection, or token-conditioned resampling loop.",
        "Exact gate: PASS KL<=0.10 and invalid<=3%; BORDERLINE KL<=0.50 and invalid<=10%.",
        "Implicit gate: invalid<=10%, unique>=25%, sensitivity>0.01; no KL claim.",
        "",
        "Configuration: " + json.dumps(config, sort_keys=True),
        "",
        f"{'arm':16s} {'KL':>9s} {'invalid':>9s} {'unique':>9s} {'sens':>9s} {'params':>11s}  verdict",
        "-" * 92,
    ]
    for r in sorted(rows, key=lambda x: x.get("arm", "")):
        kl = "n/a" if r.get("block_kl") is None else f"{r['block_kl']:.4f}"
        lines.append(f"{r['arm']:16s} {kl:>9s} {r.get('invalid_rate', float('nan')):9.4f} "
                     f"{r.get('unique_rate', float('nan')):9.4f} "
                     f"{r.get('sensitivity', float('nan')):9.4f} "
                     f"{r.get('parameters', 0):11,d}  {r.get('verdict', 'FAILED')}")
    lines += [
        "",
        "Interpretation guardrails",
        "-------------------------",
        "A pass only earns an end-to-end learned-context reproduction and an L=2048 kernel benchmark.",
        "PSS/RMLT test fixed XOR state emissions; failure is scoped to that scalable instantiation.",
        "MIF tests four declared reversible transforms; source_oracle reports each transform separately.",
        "CRC is a fixed decomposable circuit; DCMF is true JVP MeanFlow with one inference evaluation.",
        "Argmax is the exact-parallel prior-work control, not a novelty claim.",
        "",
        "Full rows",
        "---------",
        json.dumps(rows, indent=2, sort_keys=True, default=str),
        "",
    ]
    return "\n".join(lines)


@app.function(timeout=10 * 60, volumes={VOL: VOLUME})
def s06_compile(rows: list, config: dict) -> dict:
    _workdir()
    os.makedirs(S06, exist_ok=True)
    if config.get("merge"):
        config = dict(config)
        config["rerun_arms"] = list(config.get("arms", []))
        summary_path = f"{S06}/summary.json"
        if os.path.exists(summary_path):
            with open(summary_path) as f:
                old_rows = json.load(f)
            replacements = {r.get("arm"): r for r in rows}
            rows = [replacements.pop(r.get("arm"), r) for r in old_rows]
            rows.extend(replacements.values())
        config["arms"] = sorted(r.get("arm") for r in rows)
    text = _s06_text(rows, config)
    suffix = "_smoke" if config.get("smoke") else ""
    path = f"{S06}/s06_sap_compiled{suffix}.log"
    with open(path, "w") as f:
        f.write(text)
    with open(f"{S06}/summary{suffix}.json", "w") as f:
        json.dump(rows, f, indent=2, sort_keys=True, default=str)
    VOLUME.commit()
    return {"path": path, "text": text}


@app.local_entrypoint()
def s06_q(steps: int = 2500, depth: int = 4, width: int = 128, batch: int = 96,
          eval_contexts: int = 256, samples_per_ctx: int = 8, states: int = 64,
          arms: str = "", smoke: bool = False, merge: bool = False,
          out: str = "s06_sap_compiled.log"):
    """Fan the ten pre-registered S06-Q screens across independent L4s."""
    from scripts.sap_s06_screen import ARMS
    arm_list = [x for x in arms.split(",") if x] or list(ARMS)
    if smoke:
        steps, depth, width, batch, eval_contexts, samples_per_ctx, states = 2, 1, 32, 4, 4, 2, 4
    argv = [
        "--steps", str(steps), "--depth", str(depth), "--width", str(width),
        "--batch", str(batch), "--eval-contexts", str(eval_contexts),
        "--samples-per-ctx", str(samples_per_ctx), "--states", str(states),
    ]
    config = dict(steps=steps, depth=depth, width=width, batch=batch,
                  eval_contexts=eval_contexts, samples_per_ctx=samples_per_ctx,
                  states=states, smoke=smoke, arms=arm_list, merge=merge)
    print(f"S06-Q: {len(arm_list)} mechanisms in parallel on L4; "
          f"T=4 depth={depth} steps={steps}")
    rows = []
    for res in s06_screen_run.starmap([(arm, argv) for arm in arm_list], return_exceptions=True):
        if isinstance(res, BaseException):
            print(f"  run failed: {res!r}")
            continue
        rows.append(res)
        kl = "n/a" if res.get("block_kl") is None else f"{res['block_kl']:.4f}"
        print(f"  {res['arm']:16s} KL={kl} invalid={res['invalid_rate']:.4f} "
              f"unique={res['unique_rate']:.4f} {res['verdict']}")
    compiled = s06_compile.remote(rows, config)
    with open(out, "w") as f:
        f.write(compiled["text"])
    print(f"\n{len(rows)}/{len(arm_list)} completed")
    print(f"local compiled log: {os.path.abspath(out)}")
    print(f"volume compiled log: {compiled['path']}")


@app.local_entrypoint()
def s06_recompile(out: str = "s06_sap_compiled.log"):
    """Re-render the persisted ten-arm summary without launching a GPU job."""
    config = dict(steps=2500, depth=4, width=128, batch=96, eval_contexts=256,
                  samples_per_ctx=8, states=64, smoke=False, arms=[], merge=True)
    compiled = s06_compile.remote([], config)
    with open(out, "w") as f:
        f.write(compiled["text"])
    print(f"local compiled log: {os.path.abspath(out)}")
    print(f"volume compiled log: {compiled['path']}")


# ----------------------------------------------------------------------------- S06-S: semantic-PSS free oracle
S06S = f"{VOL}/out/s06_semantic_pss"


@app.function(gpu="L4", timeout=60 * 60, volumes={VOL: VOLUME})
def s06_semantic_oracle_run(argv: list) -> dict:
    _workdir()
    from scripts.sap_semantic_pss_oracle import build_parser, compile_text, evaluate
    os.makedirs(S06S, exist_ok=True)
    args = build_parser().parse_args([
        "--device", "cuda", "--out", f"{S06S}/summary.json", *argv,
    ])
    result = evaluate(args)
    text = compile_text(result)
    path = f"{S06S}/s06_semantic_pss_oracle.log"
    with open(path, "w") as f:
        f.write(text)
    VOLUME.commit()
    return {"result": result, "text": text, "path": path}


@app.local_entrypoint()
def s06_semantic(contexts: int = 16384, small_contexts: int = 4096,
                 bases: str = "1,2,3,5,9", chunk: int = 512,
                 out: str = "s06_semantic_pss_oracle.log"):
    """Run the analytic true-state semantic-permutation ceiling and download its log."""
    argv = ["--contexts", str(contexts), "--small-contexts", str(small_contexts),
            "--bases", bases, "--chunk", str(chunk)]
    compiled = s06_semantic_oracle_run.remote(argv)
    with open(out, "w") as f:
        f.write(compiled["text"])
    print(compiled["text"])
    print(f"local compiled log: {os.path.abspath(out)}")
    print(f"volume compiled log: {compiled['path']}")


# ----------------------------------------------------------------------------- S06-L: learned spectrum-class semantic PSS
S06L = f"{VOL}/out/s06_learned_scp_ss"


@app.function(gpu="L4", timeout=3 * 3600, volumes={VOL: VOLUME})
def s06_learned_run(argv: list) -> dict:
    _workdir()
    from scripts.sap_scp_ss_learned import build_parser, compile_text, run
    os.makedirs(S06L, exist_ok=True)
    args = build_parser().parse_args([
        "--device", "cuda", "--out", f"{S06L}/summary.json", *argv,
    ])
    result = run(args, checkpoint_commit=VOLUME.commit)
    text = compile_text(result)
    path = f"{S06L}/s06_learned_scp_ss.log"
    with open(path, "w") as f:
        f.write(text)
    VOLUME.commit()
    return {"result": result, "text": text, "path": path}


@app.local_entrypoint()
def s06_learned(steps: int = 8000, depth: int = 4, width: int = 128,
                states: int = 64, spectra: int = 2, batch: int = 96,
                out: str = "s06_learned_scp_ss.log"):
    """Run the learned SC-PSS toy gate; later stages remain conditional on its result."""
    argv = ["--steps", str(steps), "--depth", str(depth), "--width", str(width),
            "--states", str(states), "--spectra", str(spectra), "--batch", str(batch),
            "--eval-contexts", "512", "--samples-per-ctx", "8",
            "--eval-milestones", "4000", "8000"]
    compiled = s06_learned_run.remote(argv)
    with open(out, "w") as f:
        f.write(compiled["text"])
    print(compiled["text"])
    print(f"local compiled log: {os.path.abspath(out)}")
    print(f"volume compiled log: {compiled['path']}")


# ----------------------------------------------------------------------------- S14: strict T=L
# s14_sap_strict_tl_brainstorm.md. E0 (the order oracle) and E2a (the separator bound) score text
# with any-order models from the Hugging Face hub, in images pinned to the transformers and torch
# their model cards name: their remote code predates this repo's transformers. E1 needs no new
# code: modal run modal_sap.py::s11_ladder --depth 8 --specs dense:1:1,wb:1:1:1,wb:16:1:1 --name s14_e1
S14 = f"{VOL}/out/s14"
S14_ORACLES = {"llada": "GSAI-ML/LLaDA-8B-Base", "dream": "Dream-org/Dream-v0-Base-7B"}


def _s14_oracle_image(transformers):
    return (
        modal.Image.debian_slim(python_version="3.11")
        .pip_install("torch==2.5.1", f"transformers=={transformers}", "numpy<2", "pyarrow", "sentencepiece",
                     "protobuf", "einops")
        .env({"PYTHONPATH": SRC, "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false"})
        .add_local_dir("nanochat", remote_path=f"{SRC}/nanochat", ignore=["**/__pycache__", "**/*.pyc"])
        .add_local_dir("scripts", remote_path=f"{SRC}/scripts", ignore=["**/__pycache__", "**/*.pyc"])
    )


class _Tee:
    """stdout to the Modal log and to a log file on the volume."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
        return len(s)

    def flush(self):
        for st in self.streams:
            st.flush()


def _s14_oracle_run(argv, name):
    """One E0 shard in this process: every finished row is saved and committed to the volume, so a
    rerun of the same command resumes where a lost container stopped."""
    import contextlib
    import traceback
    try:
        VOLUME.reload()
    except Exception as e:
        print(f"Notice: VOLUME.reload() skipped or failed: {e}")
    if SRC not in sys.path:
        sys.path.insert(0, SRC)
    from scripts.sap_order_oracle import main
    local_log, code = f"/tmp/{name}.log", 0           # no file stays open on the volume across commits
    with open(local_log, "w") as log:
        tee = _Tee(sys.stdout, log)
        with contextlib.redirect_stdout(tee):
            try:
                main(argv, commit=VOLUME.commit)
            except Exception:
                traceback.print_exc(file=tee)
                code = 1
    text = open(local_log).read()
    os.makedirs(f"{S14}/logs", exist_ok=True)
    with open(f"{S14}/logs/{name}.log", "a") as f:
        f.write(text)
    VOLUME.commit()
    return {"name": name, "returncode": code, "tail": text[-4000:]}


@app.function(image=_s14_oracle_image("4.38.2"), gpu="H100", timeout=10 * 3600, volumes={VOL: VOLUME})
def s14_oracle_llada(argv: list, name: str) -> dict:
    return _s14_oracle_run(argv, name)


@app.function(image=_s14_oracle_image("4.46.2"), gpu="H100", timeout=10 * 3600, volumes={VOL: VOLUME})
def s14_oracle_dream(argv: list, name: str) -> dict:
    return _s14_oracle_run(argv, name)


@app.function(timeout=30 * 60, volumes={VOL: VOLUME})
def s14_cpu_job(name: str, cmd: list) -> dict:
    """A CPU step (merge, compare) in the work dir, logged to the volume."""
    _workdir(S03_TOK_NAME)
    code, text = _run_logged([sys.executable, "-m", *cmd], f"{S14}/logs/{name}.log")
    VOLUME.commit()
    return {"name": name, "returncode": code, "tail": text[-4000:]}


@app.local_entrypoint()
def s14_order_oracle(oracle: str = "llada", rows: int = 0, shards: int = 4, orders: str = "",
                     ar_ref: str = "Qwen/Qwen2.5-7B", batch: int = 8, block: int = 1024, prefix: int = 128,
                     sep_cut: int = 512, name: str = "", merge_only: bool = False, smoke: bool = False):
    """S14 E0 + E2a: per-order TC (information) and gap (oracle difficulty) on real text, and the
    bits a separator must carry, under an 8B any-order oracle (scripts/sap_order_oracle.py). Rows
    are split over `shards` H100s; rerunning the same command resumes from the saved rows, then
    merges into out/s14/s14_oracle_<oracle>.json.
        modal run modal_sap.py::s14_order_oracle --smoke            # 2 short rows: images, paths, probes
        modal run modal_sap.py::s14_order_oracle                    # LLaDA-8B-Base, 32 rows (primary)
        modal run modal_sap.py::s14_order_oracle --oracle dream     # Dream-v0-Base-7B, 16 rows (check)
        modal run modal_sap.py::s14_order_oracle_compare            # cross-oracle readings
    S15 L0b (s15_lanes_paper_plan.md), the paper's 1920-token block, no separator bound:
        modal run modal_sap.py::s14_order_oracle --block 1920 --sep-cut 0 --rows 16 --name t1920 \
            --orders l2r,lanes32,lanes64,lanes128,bl32_8,random30,conf30,conf60
    `name` keeps a run's raw files and result apart (out/s14/s14_oracle_<oracle>_<name>.json).
    --merge-only re-merges a finished run's raw shard files on CPU (no model load), e.g. to add a
    newer summary such as the lanes deficit/recovery profile."""
    fn = {"llada": s14_oracle_llada, "dream": s14_oracle_dream}[oracle]
    rows = rows or (32 if oracle == "llada" else 16)
    tag = f"s14_oracle_{oracle}" + (f"_{name}" if name else "")
    argv = ["--oracle", S14_ORACLES[oracle], "--data-dir", f"{VOL}/data", "--rows", str(rows), "--batch", str(batch),
            "--block", str(block), "--prefix", str(prefix), "--sep-cut", str(sep_cut)]
    argv += ["--ar-ref", ar_ref] if ar_ref else []
    argv += ["--orders", orders] if orders else []
    if smoke:
        tag, rows, shards = tag + "_smoke", 2, 2
        argv += ["--rows", "2", "--prefix", "32", "--block", "128", "--orders", "l2r,bisect1,lanes8,snap8,random6",
                 "--sep-cut", "64", "--sep-span", "32", "--sep-windows", "0,1,4"]
    jobs = [(argv + ["--shard", str(k), "--num-shards", str(shards), "--raw", f"{S14}/raw/{tag}_{k}of{shards}.pt"],
             f"{tag}_{k}of{shards}") for k in range(shards)]
    print(f"S14 E0 with {S14_ORACLES[oracle]}: {rows} rows over {shards} H100 shards" +
          (" (merge only)" if merge_only else ""))
    failed = []
    for (_, name), res in zip(jobs, [] if merge_only else fn.starmap(jobs, return_exceptions=True)):
        if isinstance(res, BaseException) or res["returncode"] != 0:
            failed.append(name)
            print(f"--- {name} FAILED: {res!r}" if isinstance(res, BaseException) else f"--- {name} FAILED\n{res['tail']}")
        else:
            print(f"--- {name}\n{res['tail'][-1200:]}")
    if failed:
        raise SystemExit(f"shards failed: {failed}. Rerun the same command: finished rows are kept.")
    res = s14_cpu_job.remote(f"{tag}_merge", ["scripts.sap_order_oracle", "--merge", *[a[-1] for a, _ in jobs],
                                              "--out", f"{S14}/{tag}.json"])
    print(res["tail"])


@app.local_entrypoint()
def s16_score(models: str, depth: int = 8, ref: str = "S11dense_x1_s1", prefix: int = 128, rows: int = 256,
              name: str = "gates"):
    """S16 (s16_lanes_recovery_brainstorm.md): score lane checkpoints at any lane count against one
    dense reference, with the deficit/recovery lane report. models: comma list of TAG@L (e.g.
    S16mix16_32_64_128x1_s1@32,S11ln64x1_s1@64). Eval only."""
    specs = [f"--model={ref}:{S03}/d{depth}/{ref}"]
    for m in [x for x in models.split(",") if x]:
        tag, L = m.split("@")
        specs.append(f"--model={tag}@{L}:{S03}/d{depth}/{tag}:ln{L}")
    edges = f"0,{prefix},{prefix + 128},{prefix + 384},1024,2047"
    out = f"s16_score_d{depth}_{name}"
    res = s03_job.remote(out, ["scripts.sap_position_bpb", "--tokenizer-dir", S03_TOK, "--data-dir", f"{VOL}/data",
                               "--rows", str(rows), "--wb-prefix", str(prefix), "--edges", edges,
                               "--out", f"{S03}/{out}.json", *specs])
    print(res["tail"][-4000:])


@app.local_entrypoint()
def s14_order_oracle_compare(a: str = "s14_oracle_llada", b: str = "s14_oracle_dream"):
    """The cross-oracle E0 readings: Spearman of the orders' cost and bisection's gap in both."""
    res = s14_cpu_job.remote("s14_oracle_compare", ["scripts.sap_order_oracle", "--compare", f"{S14}/{a}.json",
                                                    f"{S14}/{b}.json", "--out", f"{S14}/s14_oracle_compare.json"])
    print(res["tail"])
