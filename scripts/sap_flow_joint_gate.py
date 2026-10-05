"""Bounded, cost-audited three-arm Flow–Joint experiment; see sap_flow_joint_gate.md."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import torch
from torch.utils.flop_counter import FlopCounterMode

from nanochat.flow_joint import FlowJoint
from scripts.sap_s06_screen import BeliefPool, independent_control
from scripts.sap_synthetic import build_phrase_hmm, true_block_logprob


ARMS = ("hybrid", "flow_only", "chain_only")


def construct(args, context_dim, arm=None):
    return FlowJoint(context_dim, T=args.T, vocab=args.vocab, width=args.width,
                     states=args.states, latent=args.latent, arm=arm or args.arm).to(args.device)


def training_flops(model, context, targets):
    """Audited matrix-FLOPs proxy, including attention and recognition backward."""
    model.train()
    with FlopCounterMode(display=False) as counter:
        model.loss(context, targets).backward()
    model.zero_grad(set_to_none=True)
    return counter.get_total_flops()


def stats(values):
    values = values.double()
    return {"mean": values.mean().item(), "se": (values.std(unbiased=True) / math.sqrt(values.numel())).item(),
            "n_contexts": values.numel()}


@torch.no_grad()
def likelihood(model, context, targets, K, batch=64, importance_chunk=16):
    bound, ess, cond, kl = [], [], [], []
    for start in range(0, context.size(0), batch):
        a, y = context[start:start + batch], targets[start:start + batch]
        weights, likelihoods, kls = [], [], []
        for k0 in range(0, K if model.has_flow else 1, importance_chunk):
            ll, kk = model.terms(a, y, min(importance_chunk, K - k0) if model.has_flow else 1)
            weights.append(ll - kk)
            likelihoods.append(ll)
            kls.append(kk)
        w = torch.cat(weights)
        bound.append(torch.logsumexp(w, 0) - math.log(w.size(0)))
        ess.append(w.softmax(0).square().sum(0).reciprocal())
        cond.append(torch.cat(likelihoods).mean(0))
        kl.append(torch.cat(kls).mean(0))
    return [torch.cat(v) for v in (bound, ess, cond, kl)]


@torch.no_grad()
def evaluate(model, hmm, evaluation, args, final=False):
    model.eval()
    context, targets = evaluation
    if not final:
        context, targets = context[:min(256, len(context))], targets[:min(256, len(context))]
    # Evaluation never perturbs the training noise stream.
    with torch.random.fork_rng(devices=[torch.cuda.current_device()] if context.is_cuda else []):
        torch.manual_seed(92001)
        true_lp = true_block_logprob(hmm, context, targets)
        result = {"true_entropy": stats(-true_lp), "likelihood_is_bound": model.has_flow}
        for K in (args.eval_k, args.final_k) if final and model.has_flow else (args.eval_k,):
            lp, ess, cond, kl = likelihood(model, context, targets, K)
            result[f"kl_k{K}"] = stats(true_lp - lp)
            result[f"ess_k{K}"] = stats(ess)
            result["posterior_conditional_nll"] = stats(-cond)
            result["latent_kl"] = stats(kl)
        M = args.samples_per_ctx
        invalid, posterior_invalid, zero_invalid, unique = [], [], [], []
        for start in range(0, len(context), 64):
            torch.manual_seed(93000 + start)  # common prior tape, independent of importance K
            aa = context[start:start + 64].repeat_interleave(M, 0)
            yy = targets[start:start + 64].repeat_interleave(M, 0)
            samples = model.sample(aa)
            invalid.append((~torch.isfinite(true_block_logprob(hmm, aa, samples))).float().view(-1, M).mean(-1))
            unique.extend([len(torch.unique(row, dim=0)) / M for row in samples.view(-1, M, args.T)])
            if model.has_flow:
                z, _ = model.posterior(aa, yy)
                posterior_samples = model.sample(aa, z_override=z)
                posterior_invalid.append((~torch.isfinite(true_block_logprob(hmm, aa, posterior_samples))).float().view(-1, M).mean(-1))
                no_latent = model.sample(aa, z_override=torch.zeros_like(z))
                zero_invalid.append((~torch.isfinite(true_block_logprob(hmm, aa, no_latent))).float().view(-1, M).mean(-1))
        result["prior_invalid"] = stats(torch.cat(invalid))
        result["unique_fraction"] = sum(unique) / len(unique)
        if posterior_invalid:
            result["posterior_invalid"] = stats(torch.cat(posterior_invalid))
            result["zero_latent_invalid"] = stats(torch.cat(zero_invalid))
        if final:
            result["independent_oracle"] = independent_control(hmm, context, targets)
    model.train()
    return result


def source_hashes():
    root = Path(__file__).resolve().parents[1]
    paths = ("nanochat/flow_joint.py", "scripts/sap_flow_joint_gate.py", "scripts/sap_synthetic.py", "scripts/sap_s06_screen.py")
    return {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in paths}


def run(args, checkpoint_commit=None):
    torch.set_num_threads(4 if args.device.startswith("cuda") else 1)
    torch.backends.mha.set_fastpath_enabled(False)  # consistent countable train/eval operators
    torch.set_float32_matmul_precision("highest")
    device = torch.device(args.device)
    hmm = build_phrase_hmm(V=args.vocab, seed=args.hmm_seed, device=device)
    torch.manual_seed(args.seed)  # generator construction resets RNG; model seed comes AFTER it
    train = BeliefPool(hmm, args.context, args.T, args.pool,
                       torch.Generator(device=device).manual_seed(70000 + args.seed))
    heldout = BeliefPool(hmm, args.context, args.T, args.eval_contexts,
                         torch.Generator(device=device).manual_seed(81000))
    evaluation = heldout.batch(args.eval_contexts)
    model = construct(args, hmm.S)
    probe_context, probe_y = evaluation[0][:2], evaluation[1][:2]
    with torch.random.fork_rng(devices=[torch.cuda.current_device()] if device.type == "cuda" else []):
        per_step = training_flops(model, probe_context, probe_y) * args.batch // len(probe_context)
        reference = construct(args, hmm.S, "hybrid")
        reference_flops = training_flops(reference, probe_context, probe_y) * args.batch // len(probe_context)
        del reference
    steps = max(1, round(args.hybrid_steps * reference_flops / per_step))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(.9, .95), weight_decay=.01)
    metadata = {"config": vars(args), "source_sha256": source_hashes(),
                "parameters": sum(p.numel() for p in model.parameters()),
                "training_steps": steps, "matrix_flops_per_update": per_step,
                "reference_matrix_flops_per_update": reference_flops,
                "total_matrix_flops": per_step * steps,
                "matrix_budget_ratio": per_step * steps / (reference_flops * args.hybrid_steps),
                "torch": torch.__version__, "oracle_context": True,
                "gpu": torch.cuda.get_device_name() if device.type == "cuda" else "cpu",
                "neural_inference_blocks": 4 if model.has_flow else 2,
                "recognition_blocks_training_only": 2 if model.has_flow else 0,
                "inference_token_conditioned_neural_calls": 0}
    (out / "manifest.json").write_text(json.dumps(metadata, indent=2))
    curve = []
    started, train_seconds = time.perf_counter(), 0.
    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize()
    def record(step, metrics):
        row = {"step": step, "arm": args.arm, "seed": args.seed,
               "elapsed_seconds": time.perf_counter() - started, "training_seconds": train_seconds, **metrics}
        curve.append(row)
        with (out / "curve.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
    record(0, evaluate(model, hmm, evaluation, args))
    milestones = {max(1, steps // 4), max(1, steps // 2), steps}
    for step in range(1, steps + 1):
        sync()
        tick = time.perf_counter()
        a, y = train.batch(args.batch)
        beta = min(1., step / max(1, .1 * steps))
        warm = min(1., step / max(1, .02 * steps))
        lr = args.lr * warm * (.1 + .9 * .5 * (1 + math.cos(math.pi * step / steps)))
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        loss = model.loss(a, y, beta=beta)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        if not torch.isfinite(norm):
            raise RuntimeError(f"non-finite gradient at step {step}")
        optimizer.step()
        sync()
        train_seconds += time.perf_counter() - tick
        if step % 500 == 0:
            print(json.dumps({"arm": args.arm, "seed": args.seed, "step": step, "steps": steps,
                              "loss": loss.item(), "beta": beta, "training_seconds": train_seconds}), flush=True)
        if step in milestones:
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "step": step, "metadata": metadata, "cpu_rng": torch.get_rng_state(),
                        "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
                        "data_rng": train.gen.get_state(), "pool_alpha": train.alpha,
                        "pool_blocks": train.blocks, "pool_used": train.used}, out / "checkpoint.pt")
            record(step, {"loss": loss.item(), "beta": beta, "lr": lr,
                          **evaluate(model, hmm, evaluation, args, final=step == steps)})
            (out / "summary.json").write_text(json.dumps({**metadata, "curve": curve}, indent=2))
            if checkpoint_commit:
                checkpoint_commit()
    return {**metadata, "curve": curve}


def compile_text(rows):
    lines = ["SAP Flow–Joint toy gate: T=4, oracle context; NOT d4 real-text BPB or L=2048 speed.",
             "No teacher/distillation/verifier. IWAE KL is an upper bound in expectation only.",
             "Costs match counted forward/backward matrix FLOPs, not exact wall time.", ""]
    for row in sorted(rows, key=lambda r: (r["config"]["seed"], r["config"]["arm"])):
        c, end = row["config"], row["curve"][-1]
        k = max(int(key.removeprefix("kl_k")) for key in end if key.startswith("kl_k"))
        kl, inv = end[f"kl_k{k}"], end["prior_invalid"]
        lines.append(f"{c['arm']:12} seed={c['seed']} updates={end['step']}/{row['training_steps']} K={k} "
                     f"KL{' bound' if end['likelihood_is_bound'] else ''}={kl['mean']:.6f} ± {1.96*kl['se']:.6f} "
                     f"invalid={100*inv['mean']:.3f}% ± {196*inv['se']:.3f}% "
                     f"train_s={end['training_seconds']:.1f} budget_ratio={row['matrix_budget_ratio']:.6f}")
    lines.extend(["", "Full configurations, hashes, learning curves, and diagnostics:", json.dumps(rows, indent=2)])
    return "\n".join(lines) + "\n"


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", choices=ARMS, default="hybrid")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--hmm-seed", type=int, default=0)
    p.add_argument("--T", type=int, default=4)
    p.add_argument("--vocab", type=int, default=512)
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--states", type=int, default=64)
    p.add_argument("--latent", type=int, default=8)
    p.add_argument("--context", type=int, default=64)
    p.add_argument("--hybrid-steps", type=int, default=8000)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--pool", type=int, default=4096)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--eval-contexts", type=int, default=1024)
    p.add_argument("--eval-k", type=int, default=64)
    p.add_argument("--final-k", type=int, default=256)
    p.add_argument("--samples-per-ctx", type=int, default=8)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default="out/sap_flow_joint_local")
    return p


if __name__ == "__main__":
    run(parser().parse_args())
