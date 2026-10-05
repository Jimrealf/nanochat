"""One full-budget d4 or d8 T=L flow-only run on nanochat2, existing dense assets.

MODAL_PROFILE=nanochat2 modal run --detach modal_sap_flow_text.py --run-id flow_d4_tl_s1
MODAL_PROFILE=nanochat2 modal run modal_sap_flow_text.py --run-id flow_d4_tl_s1 --collect
MODAL_PROFILE=nanochat2 modal run --detach modal_sap_flow_text.py --depth 8 --run-id flow_d8_tl_s1
MODAL_PROFILE=nanochat2 modal run --detach modal_sap_flow_text.py --sweep --depth 8 --run-id flow_d8_mb262k_s1_20261004
MODAL_PROFILE=nanochat2 modal run modal_sap_flow_text.py --sweep --collect --watch --depth 8 --run-id flow_d8_mb262k_s1_20261004
"""
import json
import os
from pathlib import Path

import modal


ROOT = "/vol/out/sap_flow_text"
ASSETS = "/root/flow_assets"
MATCHED_STEPS = (1680, 5040, 8400)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("uv")
    .add_local_file("pyproject.toml", remote_path="/root/pyproject.toml", copy=True)
    .run_commands("uv pip install --system --compile-bytecode -r /root/pyproject.toml")
    .env({"PYTHONPATH": "/root/src", "OMP_NUM_THREADS": "8", "PYTHONUNBUFFERED": "1",
          "NANOCHAT_BASE_DIR": "/tmp/nanochat_base"})
    .add_local_dir("nanochat", remote_path="/root/src/nanochat", ignore=["**/__pycache__", "**/*.pyc"])
    .add_local_dir("scripts", remote_path="/root/src/scripts", ignore=["**/__pycache__", "**/*.pyc"])
    .add_local_dir("out/sap_flow_text_assets/dense_d4/S07_dense_L_s1", remote_path=f"{ASSETS}/dense_d4")
    .add_local_dir("out/sap_flow_text_assets/dense_d8/S08_dense_L_s1", remote_path=f"{ASSETS}/dense_d8")
    .add_local_dir("out/sap_flow_text_assets/tokenizer_v32k/tokenizer_sap", remote_path=f"{ASSETS}/tokenizer")
    .add_local_file("out/s07_lanes/summary_lanes_d4.json", remote_path=f"{ASSETS}/baseline_d4.json")
    .add_local_file("out/s08_lanes/summary_lanes_d8.json", remote_path=f"{ASSETS}/baseline_d8.json")
    .add_local_file("sap_flow_text_d8_matched_batch_plan.md", remote_path=f"{ASSETS}/matched_batch_plan.md")
)
app = modal.App("nanochat-sap-flow-text", image=image)
volume = modal.Volume.from_name("nanochat")


@app.function(gpu="H100", cpu=8, memory=32768, timeout=1800, volumes={"/vol": volume})
def training_perf(run_id: str):
    import contextlib
    from scripts.sap_flow_text_perf import run
    root = Path(ROOT) / run_id
    if root.exists():
        raise RuntimeError("refusing to overwrite an existing performance audit")
    root.mkdir(parents=True)
    try:
        with (root / "run.log").open("w") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            return run(root, commit=volume.commit)
    finally:
        volume.commit()


@app.function(timeout=120, volumes={"/vol": volume})
def collect_perf(run_id: str):
    volume.reload()
    root = Path(ROOT) / run_id
    p = root / "results.json"
    rows = json.loads(p.read_text()) if p.exists() else []
    return {"complete": len(rows) == 5, "rows": rows,
            "log": (root / "run.log").read_text() if (root / "run.log").exists() else ""}


@app.function(gpu="H100", cpu=8, memory=32768, timeout=21600, max_containers=3, volumes={"/vol": volume})
def full_run(run_id: str, eval_only: bool = False, depth: int = 4, steps: int = 0):
    import contextlib
    import sys
    from scripts.sap_flow_text import train, finish, parser

    if depth not in (4, 8):
        raise ValueError("only registered d4 and d8 runs are supported")
    if steps and (depth != 8 or steps not in MATCHED_STEPS):
        raise ValueError("only the three registered d8 explicit budgets are supported")
    root = Path(ROOT) / run_id
    if root.exists() and not eval_only:
        raise RuntimeError(f"refusing to overwrite {root}")
    root.mkdir(parents=True, exist_ok=True)
    if steps and not eval_only:
        (root / "preregistration.md").write_bytes(Path(f"{ASSETS}/matched_batch_plan.md").read_bytes())
    class Tee:
        def __init__(self, *streams):
            self.streams = streams
        def write(self, value):
            for stream in self.streams:
                stream.write(value); stream.flush()
            return len(value)
        def flush(self):
            for stream in self.streams:
                stream.flush()
    argv = [
        "--data-dir", "/vol/data", "--tokenizer-dir", f"{ASSETS}/tokenizer",
        "--dense-dir", f"{ASSETS}/dense_d{depth}", "--baseline-record", f"{ASSETS}/baseline_d{depth}.json",
        "--baseline-key", "S07_dense_L_s1" if depth == 4 else "S08_dense_L_s1",
        "--width", str(depth * 64), "--heads", str(depth * 2),
        "--conditioner-depth", str(depth // 4), "--decoder-depth", str(depth // 2),
        "--recognition-depth", str(depth // 2),
        "--out", str(root),
    ]
    if steps:
        argv += ["--steps", str(steps), "--batch-tokens", "262144", "--match-dense-batch",
                 "--lr-warmup-steps", "34", "--kl-warmup-steps", "168", "--eval-every", "420",
                 "--microbatch", "32", "--compile-training", "--fused-adamw"]
    args = parser().parse_args(argv)
    with (root / ("post_retry.log" if eval_only else "run.log")).open("a") as log:
        try:
            with contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
                if not eval_only:
                    train(args, commit=volume.commit)
                finish(args, commit=volume.commit)
        except Exception as exc:
            import traceback
            traceback.print_exc(file=log)
            log.flush()
            (root / "FAILED.json").write_text(json.dumps({"error": repr(exc)}, indent=2))
            volume.commit()
            raise
    (root / "COMPLETE").write_text("training and evaluation completed\n")
    volume.commit()
    return str(root)


def read_result(run_id: str):
    root = Path(ROOT) / run_id
    result = {"run_id": run_id, "complete": (root / "COMPLETE").exists()}
    for name in ("manifest.json", "training_summary.json", "evaluation.json", "FAILED.json"):
        p = root / name
        if p.exists():
            result[name] = json.loads(p.read_text())
    logs = "\n\n".join(p.read_text() for p in (root / "run.log", root / "post_retry.log") if p.exists())
    result["log"] = logs
    result["samples"] = (root / "samples.jsonl").read_text() if (root / "samples.jsonl").exists() else ""
    return result


@app.function(timeout=120, volumes={"/vol": volume})
def collect_result(run_id: str):
    volume.reload()
    return read_result(run_id)


def sweep_results(sweep_id: str):
    runs = [read_result(f"{sweep_id}_u{steps}") for steps in MATCHED_STEPS]
    status_path = Path(ROOT) / sweep_id / "coordinator.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {"state": "starting"}
    return {"sweep_id": sweep_id, "complete": all(r["complete"] for r in runs),
            "coordinator": status, "runs": runs}


def compiled_sweep(result):
    header = (f"FLOW D8 MATCHED-BATCH SWEEP: {result['sweep_id']}\n"
              f"COMPLETE: {result['complete']}\n"
              "T=L=2048; 262144 target tokens/update; fresh starts; no dense retraining.\n"
              "Budgets: 1680 / 5040 / 8400 updates. BPB bounds are not exact inference BPB.\n")
    return header + "\n".join(f"\n===== {r['run_id']} (complete={r['complete']}) =====\n{r['log']}"
                              for r in result["runs"])


def save_sweep(result, local):
    local.mkdir(parents=True, exist_ok=True)
    compact = {**result, "runs": [{k: v for k, v in r.items() if k not in ("log", "samples")}
                                  for r in result["runs"]]}
    (local / "results.json").write_text(json.dumps(compact, indent=2))
    (local / "compiled.log").write_text(compiled_sweep(result))
    for run in result["runs"]:
        folder = local / run["run_id"]
        folder.mkdir(exist_ok=True)
        (folder / "compiled.log").write_text(run["log"])
        (folder / "samples.jsonl").write_text(run["samples"])
        (folder / "result.json").write_text(json.dumps({k: v for k, v in run.items()
                                                       if k not in ("log", "samples")}, indent=2))


def wait_for_existing_call(call, limit_seconds=21900):
    """Retry read-side timeouts/connections, never resubmit a GPU invocation."""
    import time
    deadline = time.monotonic() + limit_seconds
    while time.monotonic() < deadline:
        try:
            return call.get(timeout=45)
        except modal.exception.FunctionTimeoutError:
            raise  # A worker's actual execution timeout is terminal.
        except (modal.exception.TimeoutError, TimeoutError):
            pass  # The sync SDK can wrap the polling exception in a subclass.
        except (modal.exception.ConnectionError, ConnectionError) as exc:
            print(f"Waiting for existing call after connection error: {exc!r}", flush=True)
            time.sleep(5)
    raise TimeoutError("waiting deadline reached; inspect the original call, do not duplicate it")


@app.function(timeout=22200, volumes={"/vol": volume})
def matched_batch_sweep(sweep_id: str):
    """Durable CPU parent: all three GPU calls survive a disconnected local client."""
    root = Path(ROOT) / sweep_id
    if root.exists():
        raise RuntimeError(f"refusing to relaunch existing sweep {root}")
    for steps in MATCHED_STEPS:
        if (Path(ROOT) / f"{sweep_id}_u{steps}").exists():
            raise RuntimeError("a sweep arm already exists; collect it instead of duplicating training")
    root.mkdir(parents=True)
    status = {"state": "running", "calls": {}, "outcomes": {}}
    def persist():
        (root / "coordinator.json").write_text(json.dumps(status, indent=2))
        volume.commit()
    persist()
    calls = []
    for steps in MATCHED_STEPS:
        run_id = f"{sweep_id}_u{steps}"
        call = full_run.spawn(run_id, False, 8, steps)
        calls.append((run_id, call))
        status["calls"][run_id] = call.object_id
        persist()
        print(f"SPAWNED {run_id}: {call.object_id}", flush=True)
    for run_id, call in calls:
        try:
            status["outcomes"][run_id] = {"status": "complete", "path": wait_for_existing_call(call)}
        except Exception as exc:
            status["outcomes"][run_id] = {"status": "error", "error": repr(exc)}
        volume.reload()
        persist()
        result = sweep_results(sweep_id)
        save_sweep(result, root)
        volume.commit()
        print(f"OUTCOME {run_id}: {status['outcomes'][run_id]}", flush=True)
    status["state"] = "complete" if all(v["status"] == "complete" for v in status["outcomes"].values()) else "finished_with_errors"
    persist()
    volume.reload()
    save_sweep(sweep_results(sweep_id), root)
    volume.commit()
    return status


@app.function(timeout=120, volumes={"/vol": volume})
def collect_sweep(sweep_id: str):
    volume.reload()
    return sweep_results(sweep_id)


def matched_batch_main(run_id, collect, watch):
    import time
    local = Path("out/sap_flow_text") / run_id
    local.mkdir(parents=True, exist_ok=True)
    if not collect:
        if watch:
            raise ValueError("use --collect --watch in a separate process; do not detach a collector instead of the coordinator")
        call = matched_batch_sweep.spawn(run_id)
        (local / "coordinator_call.json").write_text(json.dumps({"id": call.object_id, "sweep_id": run_id}, indent=2))
        print(f"Coordinator call {call.object_id}", flush=True)
        # Keep this the last spawned call in the launch app for detached durability.
        wait_for_existing_call(call)
        return
    deadline = time.monotonic() + 7 * 3600
    while True:
        try:
            result = collect_sweep.remote(run_id)
            save_sweep(result, local)
            Path("sap_flow_text_d8_matched_batch_compiled.log").write_text(compiled_sweep(result))
            progress = [{"run_id": r["run_id"], "complete": r["complete"],
                         "last_eval_step": (r.get("training_summary.json", {}).get("curves") or [{"step": None}])[-1]["step"]}
                        for r in result["runs"]]
            print(json.dumps({"complete": result["complete"], "runs": progress}), flush=True)
            if not watch or result["complete"] or result["coordinator"]["state"] == "finished_with_errors":
                return
        except Exception as exc:
            if not watch:
                raise
            print(f"Read-only collection failed; training is not restarted: {exc!r}", flush=True)
        if time.monotonic() >= deadline:
            raise TimeoutError("collector reached its seven-hour limit; remote artifacts remain collectable")
        for _ in range(4):
            time.sleep(30)


@app.local_entrypoint()
def main(run_id: str = "flow_d4_tl_s1", collect: bool = False, eval_only: bool = False,
         depth: int = 4, steps: int = 0, sweep: bool = False, watch: bool = False, perf: bool = False):
    if os.environ.get("MODAL_PROFILE") != "nanochat2":
        raise RuntimeError("explicit MODAL_PROFILE=nanochat2 required")
    if not run_id or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for c in run_id):
        raise ValueError("invalid run ID")
    if depth not in (4, 8) or not run_id.startswith(f"flow_d{depth}_"):
        raise ValueError("registered depth and run ID prefix must agree")
    if perf:
        if depth != 8 or any((sweep, watch, steps, eval_only)):
            raise ValueError("performance audit is a standalone d8 operation")
        local = Path("out/sap_flow_text") / run_id
        local.mkdir(parents=True, exist_ok=True)
        if collect:
            result = collect_perf.remote(run_id)
            (local / "results.json").write_text(json.dumps(result["rows"], indent=2))
            (local / "compiled.log").write_text(result["log"])
            print(json.dumps({k: v for k, v in result.items() if k != "log"}, indent=2))
            return
        call = training_perf.spawn(run_id)
        (local / "call.json").write_text(json.dumps({"id": call.object_id}, indent=2))
        print(f"Performance audit call {call.object_id}", flush=True)
        result = wait_for_existing_call(call, limit_seconds=1900)
        (local / "results.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        return
    if sweep:
        if depth != 8 or steps or eval_only or not run_id.startswith("flow_d8_mb262k_"):
            raise ValueError("matched sweep requires d8, a flow_d8_mb262k_ ID, and no single-run overrides")
        return matched_batch_main(run_id, collect, watch)
    if watch:
        raise ValueError("watch is only implemented for sweep collection")
    local = Path("out/sap_flow_text") / run_id
    local.mkdir(parents=True, exist_ok=True)
    if not collect:
        call = full_run.spawn(run_id, eval_only, depth, steps)
        (local / "call.json").write_text(json.dumps({"id": call.object_id, "run_id": run_id}, indent=2))
        print(f"Function call {call.object_id}", flush=True)
        call.get()
    result = collect_result.remote(run_id)
    (local / "result.json").write_text(json.dumps({k: v for k, v in result.items() if k not in ("log", "samples")}, indent=2))
    (local / "compiled.log").write_text(result["log"])
    (local / "samples.jsonl").write_text(result["samples"])
    name = (f"{run_id}_compiled.log" if result.get("manifest.json", {}).get("budget_mode") == "explicit_steps"
            else f"sap_flow_text_d{depth}_compiled.log")
    Path(name).write_text(result["log"])
    print(json.dumps({"run_id": run_id, "complete": result["complete"], "local": str(local)}))
