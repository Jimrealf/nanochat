"""Full-budget T=L text experiment. See sap_flow_text_d{4,8}_plan.md."""
from __future__ import annotations

import argparse
import contextlib
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

import torch
from torch.utils.flop_counter import FlopCounterMode

from nanochat.flow_text import FlowText, FlowTextConfig
from nanochat.flow_joint import categorical


def amp(device):
    return torch.autocast("cuda", dtype=torch.bfloat16) if str(device).startswith("cuda") else contextlib.nullcontext()


def sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def split_row(inputs, targets, prompt_length):
    """Loader's (x,y) overlap by one; neither lose nor duplicate boundary tokens."""
    row = torch.cat((inputs[:, :1], targets), 1)
    return row[:, :prompt_length], row[:, prompt_length:]


def baseline_budget(record):
    if record.get("returncode") != 0 or min(record["train_tokens"], record["flops_per_token"]) <= 0:
        raise ValueError("a successful baseline with positive token/FLOP counts is required")
    return int(record["train_tokens"]) * float(record["flops_per_token"])


def baseline_assets(args, config):
    """Refuse budget/checkpoint mismatches before spending on a training run."""
    from nanochat.checkpoint_manager import find_last_step
    record = json.loads(Path(args.baseline_record).read_text())[args.baseline_key]
    budget = baseline_budget(record)
    step = find_last_step(args.dense_dir)
    meta = json.loads((Path(args.dense_dir) / f"meta_{step:06d}.json").read_text())
    mc = meta["model_config"]
    expected = (config.vocab, config.generative_depth, config.width, "L")
    actual = (mc["vocab_size"], mc["n_layer"], mc["n_embd"], mc["window_pattern"])
    if actual != expected:
        raise ValueError(f"wrong baseline architecture: {actual} != {expected}")
    if (meta["step"] != step or step * meta["total_batch_size"] != record["train_tokens"]
            or meta["user_config"]["model_tag"] != args.baseline_key
            or not math.isclose(meta["val_bpb"], record["val_bpb"], abs_tol=1e-5)):
        raise ValueError("baseline checkpoint does not match the recorded training budget/result")
    return record, budget, step


def count_training_flops(model, prompt, target, device):
    model.train()
    with FlopCounterMode(display=False) as counter, amp(device):
        model(prompt, target).backward()
    model.zero_grad(set_to_none=True)
    return counter.get_total_flops()


def training_budget(args, baseline_flops, row_flops):
    micro_tokens = args.microbatch * args.T
    if min(args.microbatch, args.T, args.batch_tokens) < 1 or args.batch_tokens % micro_tokens:
        raise ValueError("effective batch must be a positive exact multiple of microbatch * T")
    if args.steps is not None and args.steps < 1:
        raise ValueError("explicit training steps must be positive")
    accum = args.batch_tokens // micro_tokens
    per_update = row_flops * args.microbatch * accum
    steps = args.steps if args.steps is not None else max(1, int(baseline_flops // per_update))
    if args.smoke:
        steps = 2
    return accum, steps, per_update


def training_schedule(step, steps, peak_lr, lr_warmup_steps=None, kl_warmup_steps=None):
    lr_warmup = max(1., .02 * steps) if lr_warmup_steps is None else lr_warmup_steps
    kl_warmup = max(1., .1 * steps) if kl_warmup_steps is None else kl_warmup_steps
    if not (0 < lr_warmup <= steps and 0 < kl_warmup <= steps):
        raise ValueError("warmup must be positive and no longer than the training run")
    beta = min(1., step / kl_warmup)
    lr = peak_lr * min(1., step / lr_warmup) * (.1 + .9 * .5 * (1 + math.cos(math.pi * step / steps)))
    return lr, beta


@torch.no_grad()
def evaluate(model, prompts, targets, token_bytes, device, K=1, batch=2):
    model.eval()
    bounds, ess, recon, kl_values = [], [], [], []
    with torch.random.fork_rng(devices=[torch.cuda.current_device()] if str(device).startswith("cuda") else []):
        torch.manual_seed(97131)
        for start in range(0, len(prompts), batch):
            p, y = prompts[start:start + batch].to(device), targets[start:start + batch].to(device)
            weights, lls, kls = [], [], []
            with amp(device):
                for _ in range(K):
                    ll, kl = model.terms(p, y)
                    weights.append((ll - kl).double())
                    lls.append(ll.double())
                    kls.append(kl.double())
            w = torch.stack(weights)
            bounds.append((torch.logsumexp(w, 0) - math.log(K)).cpu())
            ess.append(w.softmax(0).square().sum(0).reciprocal().cpu())
            recon.append(torch.stack(lls).mean(0).cpu())
            kl_values.append(torch.stack(kls).mean(0).cpu())
    nll = -torch.cat(bounds)
    bytes_per_row = token_bytes.cpu()[targets.cpu()].sum(-1).double()
    ratio = nll.sum() / (math.log(2) * bytes_per_row.sum())
    # Ratio-of-sums standard error across independent rows (delta method).
    residual = nll / math.log(2) - ratio * bytes_per_row
    se = residual.std() / math.sqrt(len(nll)) / bytes_per_row.mean()
    model.train()
    return {"K": K, "joint_bpb_bound": ratio.item(), "bpb_se": se.item(),
            "ess_mean": torch.cat(ess).mean().item(), "ess_min": torch.cat(ess).min().item(),
            "reconstruction_nats_per_token": (-torch.cat(recon).mean() / targets.size(1)).item(),
            "latent_kl_nats_per_token": (torch.cat(kl_values).mean() / targets.size(1)).item(),
            "rows": len(nll), "bytes": bytes_per_row.sum().item(),
            "row_nll_bound": nll.tolist(), "row_bytes": bytes_per_row.tolist(),
            "bound_in_expectation_not_pointwise": True, "includes_special_token_nll": True}


@torch.no_grad()
def dense_score(dense, prompts, targets, token_bytes, device, batch=2):
    dense.eval()
    nlls, conventional = [], []
    for start in range(0, len(prompts), batch):
        p, y = prompts[start:start + batch].to(device), targets[start:start + batch].to(device)
        row = torch.cat((p, y), 1)
        with amp(device):
            losses = dense(row[:, :-1].contiguous(), row[:, 1:].contiguous(), loss_reduction="none").view(len(p), -1)
        losses = losses[:, p.size(1) - 1:]
        assert losses.shape == y.shape
        nlls.append(losses.double().sum(1).cpu())
        conventional.append((losses * (token_bytes.to(device)[y] > 0)).double().sum(1).cpu())
    nll = torch.cat(nlls)
    sizes = token_bytes.cpu()[targets.cpu()].sum(1).double()
    denom = math.log(2) * sizes.sum()
    return {"joint_bpb": (nll.sum() / denom).item(),
            "conventional_masked_bpb": (torch.cat(conventional).sum() / denom).item(),
            "row_nll": nll.tolist(), "row_bytes": sizes.tolist(), "rows": len(nll),
            "includes_special_token_nll": True}


def train(args, commit=None):
    from nanochat.tokenizer import get_tokenizer, get_token_bytes
    from nanochat.dataloader import tokenizing_distributed_data_loader_with_state_bos_bestfit
    torch.set_num_threads(8 if args.device.startswith("cuda") else 1)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = get_tokenizer(args.tokenizer_dir)
    tokenizer_digest = hashlib.sha256((Path(args.tokenizer_dir) / "tokenizer.pkl").read_bytes()).hexdigest()
    if tokenizer_digest != "06978be3b6fa73254b3adc4a2f7b499bc7f9d7eb9d74f37b7a1620bd84e02808":
        raise ValueError("tokenizer SHA256 does not match the baseline's pinned tokenizer")
    vocab = tokenizer.get_vocab_size()
    if vocab != 32768 and not args.smoke:
        raise ValueError(f"expected pinned V=32768, got {vocab}")
    config = FlowTextConfig(vocab=vocab, T=args.T, prompt=args.prompt, width=args.width,
                            heads=args.heads, latent=args.latent, projection_chunk=args.projection_chunk,
                            conditioner_depth=args.conditioner_depth, decoder_depth=args.decoder_depth,
                            recognition_depth=args.recognition_depth)
    model = FlowText(config).to(args.device)
    token_bytes = get_token_bytes(device="cpu", tokenizer_dir=args.tokenizer_dir)
    dense_record, total_budget, dense_step = baseline_assets(args, config)
    # Evaluation rows are cloned because the streaming loader reuses its buffers.
    loader_args = dict(tokenizer=tokenizer, B=args.microbatch, T=args.prompt + args.T - 1,
                       device=args.device, data_dir=args.data_dir)
    val_loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(
        split="val", **{**loader_args, "B": args.eval_microbatch})
    held_p, held_y = [], []
    for _ in range(math.ceil(args.eval_rows / args.eval_microbatch)):
        x, y, _ = next(val_loader)
        p, target = split_row(x, y, args.prompt)
        held_p.append(p.cpu().clone()); held_y.append(target.cpu().clone())
    prompts, targets = torch.cat(held_p)[:args.eval_rows], torch.cat(held_y)[:args.eval_rows]
    torch.save({"prompts": prompts, "targets": targets}, out / "heldout.pt")
    del val_loader
    with torch.random.fork_rng(devices=[torch.cuda.current_device()] if args.device.startswith("cuda") else []):
        probe_flops = count_training_flops(model, prompts[:1].to(args.device), targets[:1].to(args.device), args.device)
        model.eval()
        with amp(args.device):
            sample = model.sample(prompts[:1].to(args.device))
        if sample.shape != (1, args.T):
            raise RuntimeError("full-length generation preflight failed")
        model.train()
    accum, steps, per_update = training_budget(args, total_budget, probe_flops)
    batch_tokens = args.batch_tokens
    training_schedule(1, steps, args.lr, args.lr_warmup_steps, args.kl_warmup_steps)
    if args.eval_every < 0:
        raise ValueError("evaluation interval cannot be negative")
    dense_meta = json.loads((Path(args.dense_dir) / f"meta_{dense_step:06d}.json").read_text())
    if args.match_dense_batch and batch_tokens != dense_meta["total_batch_size"]:
        raise ValueError("requested effective batch does not match dense")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(.9, .95),
                                 weight_decay=.01, fused=args.fused_adamw)
    from scripts.sap_flow_text_perf import make_train_callable
    train_model = make_train_callable(model, args.compile_training)
    beta_tensor = torch.tensor(1., device=args.device)
    compile_warmup_seconds = 0.
    if args.compile_training:
        # Preserve initialization and RNG; this warms code, not model parameters.
        with torch.random.fork_rng(devices=[torch.cuda.current_device()] if args.device.startswith("cuda") else []):
            warm_p = prompts[:args.microbatch].to(args.device).contiguous()
            warm_y = targets[:args.microbatch].to(args.device).contiguous()
            if len(warm_p) != args.microbatch:
                raise ValueError("enough held-out rows for the compile preflight are required")
            epsilon = torch.randn(args.microbatch, config.T, config.latent, device=args.device)
            sync(args.device); started_compile = time.perf_counter()
            with amp(args.device):
                warm_loss = train_model(warm_p, warm_y, beta_tensor, epsilon)
            warm_loss.backward()
            sync(args.device)
            compile_warmup_seconds = time.perf_counter() - started_compile
            model.zero_grad(set_to_none=True)
            del warm_loss, warm_p, warm_y, epsilon
        print(f"COMPILE_WARMUP_SECONDS {compile_warmup_seconds:.3f}", flush=True)
    repo = Path(__file__).resolve().parents[1]
    source_files = ("nanochat/flow_text.py", "nanochat/flow_joint.py", "scripts/sap_flow_text.py",
                    "nanochat/dataloader.py", "nanochat/tokenizer.py", "scripts/sap_flow_text_perf.py")
    hashes = {name: hashlib.sha256((repo / name).read_bytes()).hexdigest() for name in source_files}
    source_dir = out / "source"
    for name in source_files:
        dest = source_dir / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((repo / name).read_bytes())
    tokenizer_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(args.tokenizer_dir).iterdir() if p.is_file()}
    metadata = {"args": vars(args), "config": asdict(config), "baseline": dense_record,
                "baseline_total_reported_flops": total_budget, "probe_matrix_flops_per_row": probe_flops,
                "update_matrix_flops": per_update, "steps": steps, "batch_tokens": batch_tokens,
                "gradient_accumulation_steps": accum,
                "compile_warmup_seconds": compile_warmup_seconds,
                "budget_mode": "explicit_steps" if args.steps is not None else "baseline_flops",
                "training_compute_ratio_to_dense": per_update * steps / total_budget,
                "training_token_ratio_to_dense": steps * batch_tokens / dense_record["train_tokens"],
                "effective_batch_matches_dense": batch_tokens == dense_meta["total_batch_size"],
                "heldout_token_sha256": hashlib.sha256(prompts.numpy().tobytes() + targets.numpy().tobytes()).hexdigest(),
                "train_tokens": steps * batch_tokens, "total_matrix_flops": per_update * steps,
                "parameters": sum(p.numel() for p in model.parameters()), "source_sha256": hashes,
                "tokenizer_sha256": tokenizer_hashes, "torch": torch.__version__,
                "gpu": torch.cuda.get_device_name() if args.device.startswith("cuda") else "cpu",
                "generative_blocks": config.generative_depth,
                "recognition_blocks_training_only": config.recognition_depth,
                "baseline_key": args.baseline_key, "baseline_step": dense_step,
                "new_dense_training": False, "toy_gate_overridden_by_user": True}
    from nanochat.dataset import list_parquet_files
    shards = list_parquet_files(data_dir=args.data_dir)
    metadata["data_shards"] = [{"name": Path(path).name, "bytes": Path(path).stat().st_size} for path in shards]
    metadata["validation_shard"] = Path(shards[-1]).name
    metadata["baseline_checkpoint_sha256"] = hashlib.sha256((Path(args.dense_dir) / f"model_{dense_step:06d}.pt").read_bytes()).hexdigest()
    (out / "manifest.json").write_text(json.dumps(metadata, indent=2))
    print("MANIFEST " + json.dumps(metadata), flush=True)
    if commit:
        commit()
    curves = []
    began = time.perf_counter()
    def record(step, loss=None, beta=1.):
        metrics = evaluate(model, prompts, targets, token_bytes, args.device, K=1, batch=args.eval_microbatch)
        row = {"step": step, "loss": loss, "beta": beta, "elapsed_seconds": time.perf_counter() - began,
               "training_tokens": step * batch_tokens, **metrics}
        curves.append(row)
        with (out / "curve.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        print("EVAL " + json.dumps({k: v for k, v in row.items() if not k.startswith("row_")}), flush=True)
        (out / "training_summary.json").write_text(json.dumps({"metadata": metadata, "curves": curves}, indent=2))
    record(0)
    loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(split="train", **loader_args)
    milestones = {max(1, steps // 4), max(1, steps // 2), max(1, 3 * steps // 4), steps}
    if args.eval_every:
        milestones.update(range(args.eval_every, steps, args.eval_every))
    for step in range(1, steps + 1):
        tick = time.perf_counter()
        lr, beta = training_schedule(step, steps, args.lr, args.lr_warmup_steps, args.kl_warmup_steps)
        beta_tensor.fill_(beta)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for _ in range(accum):
            x, y, data_state = next(loader)
            p, target = split_row(x, y, args.prompt)
            p, target = p.contiguous(), target.contiguous()
            epsilon = torch.randn(args.microbatch, config.T, config.latent, device=args.device)
            with amp(args.device):
                loss = train_model(p, target, beta_tensor, epsilon)
            (loss / accum).backward()
            losses.append(loss.detach())
        loss_values = torch.stack(losses)
        if not torch.isfinite(loss_values).all():
            raise RuntimeError(f"nonfinite loss at step {step}")
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        if not torch.isfinite(norm):
            raise RuntimeError(f"nonfinite gradients at step {step}")
        optimizer.step()
        sync(args.device)
        mean_loss = loss_values.mean().item()
        if step == 1 or step % 25 == 0:
            print(json.dumps({"step": step, "steps": steps, "loss": mean_loss, "beta": beta, "lr": lr,
                              "tokens_per_second": batch_tokens / (time.perf_counter() - tick),
                              "elapsed_seconds": time.perf_counter() - began}), flush=True)
        if step in milestones:
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step,
                        "metadata": metadata, "cpu_rng": torch.get_rng_state(),
                        "cuda_rng": torch.cuda.get_rng_state_all() if args.device.startswith("cuda") else [],
                        "dataloader_state": data_state, "resume_data_is_approximate": True}, out / "checkpoint.pt")
            record(step, mean_loss, beta)
            if commit:
                commit()
    return {"metadata": metadata, "curves": curves}


@torch.no_grad()
def generation_bench(flow, dense, prompts, device, repeats=3):
    """End-to-end outputs, prompt processing and sampling included for both models."""
    from nanochat.engine import _kv_cache_for
    from scripts.sap_decode_bench import _capture
    flow.eval(); dense.eval()
    B, C = prompts.shape
    T = flow.config.T
    cache = _kv_cache_for(dense, B, C + T + 16, torch.device(device), torch.bfloat16)
    cache.graph_safe = True
    output = torch.empty(B, T, device=device, dtype=torch.long)
    ids = torch.empty(B, 1, device=device, dtype=torch.long)
    position = torch.ones((), device=device, dtype=torch.long)
    def pick(logits):
        return categorical(logits.float().softmax(-1), torch.rand(B, device=device))[:, None]
    def prefill():
        cache.reset()
        position.fill_(1)
        ids.copy_(pick(dense(prompts, kv_cache=cache)[:, -1]))
        output[:, :1].copy_(ids)
    def ar_step():
        ids.copy_(pick(dense(ids, kv_cache=cache)[:, -1]))
        output.scatter_(1, position.expand(B, 1), ids)
        position.add_(1)
    def ar_eager():
        prefill()
        for _ in range(T - 1):
            ar_step()
        return output
    def measure(fn):
        values = []
        for _ in range(repeats):
            sync(device); tick = time.perf_counter(); result = fn(); sync(device)
            assert result.shape == (B, T)
            values.append(time.perf_counter() - tick)
        return {"seconds": statistics.median(values), "trials_seconds": values,
                "tokens_per_second": B * T / statistics.median(values)}
    with amp(device):
        flow.sample(prompts)
        ar_eager()
        eager_flow = measure(lambda: flow.sample(prompts))
        eager_ar = measure(ar_eager)
        flow_out = torch.empty_like(output)
        def flow_step():
            flow_out.copy_(flow.sample(prompts))
        flow_graph = _capture(flow_step, 3)
        prefill_graph = _capture(prefill, 3)
        prefill()
        ar_graph = _capture(ar_step, 3)
        def graph_flow():
            flow_graph.replay()
            return flow_out
        def graph_ar():
            prefill_graph.replay()
            for _ in range(T - 1):
                ar_graph.replay()
            return output
        graph_f, graph_a = measure(graph_flow), measure(graph_ar)
    return {"batch": B, "actual_generated_tokens_per_row": T, "prompt": C,
            "eager_flow": eager_flow, "eager_ar": eager_ar,
            "graph_flow": graph_f, "graph_ar": graph_a,
            "eager_speedup": eager_flow["tokens_per_second"] / eager_ar["tokens_per_second"],
            "graph_speedup": graph_f["tokens_per_second"] / graph_a["tokens_per_second"]}


def finish(args, commit=None):
    from nanochat.checkpoint_manager import build_model
    from nanochat.tokenizer import get_token_bytes
    torch.backends.mha.set_fastpath_enabled(False)
    out = Path(args.out)
    checkpoint_data = torch.load(out / "checkpoint.pt", map_location=args.device, weights_only=False)
    config = FlowTextConfig(**checkpoint_data["metadata"]["config"])
    if checkpoint_data["step"] != checkpoint_data["metadata"]["steps"]:
        raise ValueError("final evaluation requires the completed training budget")
    _, _, dense_step = baseline_assets(args, config)
    digest = hashlib.sha256((Path(args.dense_dir) / f"model_{dense_step:06d}.pt").read_bytes()).hexdigest()
    if digest != checkpoint_data["metadata"]["baseline_checkpoint_sha256"]:
        raise ValueError("evaluation baseline differs from the registered training reference")
    flow = FlowText(config).to(args.device)
    flow.load_state_dict(checkpoint_data["model"])
    del checkpoint_data
    held = torch.load(out / "heldout.pt", weights_only=True)
    prompts, targets = held["prompts"], held["targets"]
    token_bytes = get_token_bytes(tokenizer_dir=args.tokenizer_dir)
    dense, tokenizer, _ = build_model(args.dense_dir, dense_step, args.device, "eval", tokenizer_dir=args.tokenizer_dir)
    result = {"dense": dense_score(dense, prompts, targets, token_bytes, args.device), "flow_bounds": []}
    for K in (1, 8, 32):
        row = evaluate(flow, prompts, targets, token_bytes, args.device, K=K, batch=2)
        row["ratio_to_dense_joint_bpb"] = row["joint_bpb_bound"] / result["dense"]["joint_bpb"]
        result["flow_bounds"].append(row)
        print("FINAL_BOUND " + json.dumps({k: v for k, v in row.items() if not k.startswith("row_")}), flush=True)
        (out / "evaluation.json").write_text(json.dumps(result, indent=2))
        if commit:
            commit()
    flow.eval()
    with torch.no_grad(), amp(args.device):
        samples = torch.cat([flow.sample(prompts[a:a + 4].to(args.device)).cpu() for a in range(0, 16, 4)])
    result["prior_sample_reference"] = dense_score(dense, prompts[:16], samples, token_bytes, args.device)
    distinct = [len(set(zip(row[:-2], row[1:-1], row[2:]))) / max(1, len(row) - 2) for row in samples.tolist()]
    result["prior_sample_distinct3"] = statistics.mean(distinct)
    with (out / "samples.jsonl").open("w") as f:
        for p, y, diversity in zip(prompts[:16].tolist(), samples.tolist(), distinct):
            f.write(json.dumps({"prompt_ids": p, "generated_ids": y, "prompt": tokenizer.decode(p),
                                "completion": tokenizer.decode(y), "distinct3": diversity}) + "\n")
    result["benchmarks"] = []
    for B in (1, 16):
        torch.cuda.reset_peak_memory_stats()
        bench = generation_bench(flow, dense, prompts[:B].to(args.device), args.device)
        bench["peak_allocated_bytes_both_models"] = torch.cuda.max_memory_allocated()
        result["benchmarks"].append(bench)
        print("BENCH " + json.dumps(bench), flush=True)
        (out / "evaluation.json").write_text(json.dumps(result, indent=2))
        if commit:
            commit()
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name, default in (("T", 2048), ("prompt", 128), ("width", 256), ("heads", 8),
                          ("latent", 8), ("projection-chunk", 128), ("microbatch", 4),
                          ("batch-tokens", 65536), ("eval-rows", 128), ("seed", 1),
                          ("conditioner-depth", 1), ("decoder-depth", 2), ("recognition-depth", 2)):
        p.add_argument("--" + name, type=int, default=default)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--steps", type=int, default=None, help="explicit update budget; overrides iso-FLOP step derivation")
    p.add_argument("--lr-warmup-steps", type=int, default=None)
    p.add_argument("--kl-warmup-steps", type=int, default=None)
    p.add_argument("--eval-every", type=int, default=0)
    p.add_argument("--match-dense-batch", action="store_true")
    p.add_argument("--compile-training", action="store_true")
    p.add_argument("--fused-adamw", action="store_true")
    p.add_argument("--eval-microbatch", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--tokenizer-dir", required=True)
    p.add_argument("--dense-dir", required=True)
    p.add_argument("--baseline-record", required=True)
    p.add_argument("--baseline-key", default="S07_dense_L_s1")
    p.add_argument("--out", required=True)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--eval-only", action="store_true")
    return p


if __name__ == "__main__":
    arguments = parser().parse_args()
    finish(arguments) if arguments.eval_only else train(arguments)
