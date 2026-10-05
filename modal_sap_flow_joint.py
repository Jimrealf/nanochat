"""Isolated Flow–Joint gate; does not modify or launch existing SAP experiments.

MODAL_PROFILE=nanochat2 modal run --detach modal_sap_flow_joint.py --run-id <unique-id>
MODAL_PROFILE=nanochat2 modal run modal_sap_flow_joint.py --run-id <unique-id> --collect
"""
import json
from pathlib import Path

import modal

# Keep the existing dependency image/cache, but do not import another launcher:
# Modal mounts this entrypoint, not arbitrary sibling modules, in remote workers.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("uv")
    .add_local_file("pyproject.toml", remote_path="/root/pyproject.toml", copy=True)
    .run_commands("uv pip install --system --compile-bytecode -r /root/pyproject.toml")
    .env({"PYTHONPATH": "/root/src", "OMP_NUM_THREADS": "8", "NANOCHAT_BASE_DIR": "/tmp/nanochat_base",
          "PYTHONUNBUFFERED": "1"})
    .add_local_dir("nanochat", remote_path="/root/src/nanochat", ignore=["**/__pycache__", "**/*.pyc"])
    .add_local_dir("scripts", remote_path="/root/src/scripts", ignore=["**/__pycache__", "**/*.pyc"])
)
app = modal.App("nanochat-sap-flow-joint", image=image)
volume = modal.Volume.from_name("nanochat")
ROOT = "/vol/out/sap_flow_joint"


@app.function(gpu="L4", timeout=3600, volumes={"/vol": volume}, max_containers=6)
def experiment(run_id: str, arm: str, seed: int):
    import contextlib
    import sys
    from scripts.sap_flow_joint_gate import parser, run

    class Tee:
        def __init__(self, *streams):
            self.streams = streams
        def write(self, value):
            for stream in self.streams:
                stream.write(value)
                stream.flush()
            return len(value)
        def flush(self):
            for stream in self.streams:
                stream.flush()

    out = Path(ROOT) / run_id / f"{arm}_s{seed}"
    if out.exists():
        raise RuntimeError(f"Refusing to overwrite existing run {out}")
    out.mkdir(parents=True)
    args = parser().parse_args(["--arm", arm, "--seed", str(seed), "--device", "cuda", "--out", str(out)])
    with (out / "run.log").open("w") as log:
        try:
            with contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
                result = run(args, checkpoint_commit=volume.commit)
        except Exception:
            import traceback
            traceback.print_exc(file=log)
            log.flush()
            volume.commit()
            raise
    volume.commit()
    return result


@app.function(timeout=120, volumes={"/vol": volume})
def collect_results(run_id: str):
    from scripts.sap_flow_joint_gate import compile_text
    volume.reload()
    root = Path(ROOT) / run_id
    rows, pending = [], []
    for arm in ("hybrid", "flow_only", "chain_only"):
        for seed in (0, 1):
            file = root / f"{arm}_s{seed}" / "summary.json"
            if not file.exists():
                pending.append(f"{arm}_s{seed}: no summary")
                continue
            row = json.loads(file.read_text())
            if row["curve"][-1]["step"] != row["training_steps"]:
                pending.append(f"{arm}_s{seed}: {row['curve'][-1]['step']}/{row['training_steps']}")
            rows.append(row)
    compiled = compile_text(rows)
    (root / "sap_flow_joint_compiled.log").write_text(compiled)
    (root / "summary.json").write_text(json.dumps({"pending": pending, "rows": rows}, indent=2))
    volume.commit()
    return {"pending": pending, "rows": rows, "compiled": compiled}


@app.function(timeout=3900, volumes={"/vol": volume})
def sweep(run_id: str):
    """One durable parent call owns all six children if the local client disconnects."""
    root = Path(ROOT) / run_id
    if root.exists():
        raise RuntimeError(f"Refusing to reuse run directory {root}")
    root.mkdir(parents=True)
    calls = []
    for arm in ("hybrid", "flow_only", "chain_only"):
        for seed in (0, 1):
            call = experiment.spawn(run_id, arm, seed)
            calls.append({"arm": arm, "seed": seed, "function_call_id": call.object_id})
            print(json.dumps(calls[-1]), flush=True)
    (root / "calls.json").write_text(json.dumps(calls, indent=2))
    volume.commit()
    errors = []
    for entry in calls:
        try:
            modal.FunctionCall.from_id(entry["function_call_id"]).get()
        except Exception as exc:
            errors.append({**entry, "error": str(exc)})
    result = collect_results.remote(run_id)
    result["errors"], result["calls"] = errors, calls
    return result


@app.local_entrypoint()
def main(run_id: str, collect: bool = False):
    import os
    if os.environ.get("MODAL_PROFILE") != "nanochat2":
        raise RuntimeError("Set MODAL_PROFILE=nanochat2 explicitly for this gate")
    if not run_id or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for c in run_id):
        raise ValueError("run-id must use lowercase letters, numbers, hyphens or underscores")
    out = Path("out/sap_flow_joint") / run_id
    out.mkdir(parents=True, exist_ok=True)
    result = collect_results.remote(run_id) if collect else sweep.remote(run_id)
    (out / "summary.json").write_text(json.dumps({k: v for k, v in result.items() if k != "compiled"}, indent=2))
    (out / "sap_flow_joint_compiled.log").write_text(result["compiled"])
    if not result["pending"]:
        Path("sap_flow_joint_compiled.log").write_text(result["compiled"])
    print(result["compiled"].split("Full configurations")[0])
    print("Pending:", result["pending"])
