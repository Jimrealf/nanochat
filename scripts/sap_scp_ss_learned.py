"""Learned Spectrum-Class Semantic PSS on the exact phrase-HMM gate.

This is the post-oracle S06-L gate.  It learns a 64-state random-map chain, two
state-routed base emission spectra, and an arbitrary bijection per latent state.
The neural parameters use exact marginal likelihood.  Bijections use a generalized
EM step: posterior state responsibilities define a token-to-base-rank assignment
score, then Hungarian assignment enforces an exact permutation.

The only oracle input retained from S06-Q is the exact HMM filtering belief used as
context, so this tests the joint generator rather than a context encoder.  The model
does not read the HMM transition/emission parameters and uses no teacher/verifier.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from scripts.sap_s06_screen import (
    BeliefPool,
    SlotBackbone,
    categorical_from_uniform,
    evaluate,
)
from scripts.sap_synthetic import build_phrase_hmm


class LearnedSCPSS(nn.Module):
    exact = True
    training_multiplier = 1

    def __init__(self, context_dim: int, T: int, V: int, width: int, depth: int,
                 states: int = 64, spectra: int = 2, seed: int = 0):
        super().__init__()
        assert states % spectra == 0
        self.T, self.V, self.S, self.C = T, V, states, spectra
        self.backbone = SlotBackbone(context_dim, T, width, depth)
        self.readout = nn.Linear(width, spectra * V)
        self.pi = nn.Linear(width, states)
        self.trans = nn.Linear(width, states * states)
        classes = torch.arange(states) * spectra // states
        self.register_buffer("state_class", classes.long())

        # Diverse exact bijections, subsequently learned by assignment-EM.
        gen = torch.Generator().manual_seed(60_600 + seed)
        perm = torch.stack([torch.randperm(V, generator=gen) for _ in range(states)])
        inv = torch.empty_like(perm)
        ranks = torch.arange(V)[None].expand_as(perm)
        inv.scatter_(1, perm, ranks)
        self.register_buffer("perm", perm)          # base rank -> output token
        self.register_buffer("inv_perm", inv)      # output token -> base rank

        # Spectrum 0 begins sharp; spectrum 1 begins diffuse. These are generic shape
        # priors learned from the oracle diagnosis, not HMM transition/emission values.
        with torch.no_grad():
            bias = self.readout.bias.view(spectra, V)
            bias.zero_()
            bias[0, 0] = 4.0
            if spectra > 1:
                bias[1, :min(32, V)] = 1.0

    def fields(self, alpha):
        h = self.backbone(alpha)
        logq = self.readout(h).view(-1, self.T, self.C, self.V).log_softmax(-1)
        logpi = self.pi(h.mean(1)).log_softmax(-1)
        logA = self.trans(h[:, 1:]).view(-1, self.T - 1, self.S, self.S).log_softmax(-1)
        return logq, logpi, logA

    def emission_log_probs(self, logq, y):
        B = y.size(0)
        out = torch.empty(B, self.T, self.S, device=y.device, dtype=logq.dtype)
        for c in range(self.C):
            states = (self.state_class == c).nonzero(as_tuple=False).squeeze(1)
            # Advanced indexing gives S_c,B,T; transpose to B,T,S_c for the gather.
            ranks = self.inv_perm[states][:, y].permute(1, 2, 0)
            vals = logq[:, :, c].gather(2, ranks)
            out[:, :, states] = vals
        return out

    def forward_backward(self, logq, logpi, logA, y, posterior: bool = False):
        emit = self.emission_log_probs(logq, y)
        fwd = [logpi + emit[:, 0]]
        for t in range(1, self.T):
            fwd.append(torch.logsumexp(fwd[-1][:, :, None] + logA[:, t - 1], 1) + emit[:, t])
        logz = torch.logsumexp(fwd[-1], -1)
        if not posterior:
            return logz, None
        bwd = [None] * self.T
        bwd[-1] = torch.zeros_like(fwd[-1])
        for t in range(self.T - 2, -1, -1):
            bwd[t] = torch.logsumexp(logA[:, t] + emit[:, t + 1, None, :] +
                                     bwd[t + 1][:, None, :], -1)
        gamma = torch.stack([(fwd[t] + bwd[t] - logz[:, None]).exp()
                             for t in range(self.T)], 1)
        return logz, gamma

    def log_prob(self, alpha, y):
        return self.forward_backward(*self.fields(alpha), y)[0]

    @torch.no_grad()
    def sample(self, alpha):
        logq, logpi, logA = self.fields(alpha)
        B = alpha.size(0)
        # Entire primitive random tape is materialized before state-dependent operations.
        u0 = torch.rand(B, device=alpha.device)
        umaps = torch.rand(B, self.T - 1, self.S, device=alpha.device)
        ux = torch.rand(B, self.T, device=alpha.device)
        s0 = categorical_from_uniform(logpi.exp(), u0)
        maps = categorical_from_uniform(logA.exp(), umaps)
        pref = maps
        off = 1
        while off < self.T - 1:
            nxt = pref.clone()
            nxt[:, off:] = pref[:, off:].gather(-1, pref[:, :-off])
            pref = nxt
            off *= 2
        rest = pref.gather(-1, s0[:, None, None].expand(B, self.T - 1, 1)).squeeze(-1)
        states = torch.cat((s0[:, None], rest), 1)
        cls = self.state_class[states]
        b = torch.arange(B, device=alpha.device)[:, None]
        t = torch.arange(self.T, device=alpha.device)[None]
        q = logq.exp()[b, t, cls]
        x = categorical_from_uniform(q, ux)
        return self.perm[states, x]


@torch.no_grad()
def update_permutations(model: LearnedSCPSS, pool: BeliefPool, batches: int, batch: int):
    """Generalized M-step: exact bijection maximizing posterior-weighted rank score."""
    was_training = model.training
    model.eval()
    dev = model.perm.device
    score = torch.zeros(model.S, model.V, model.V, device=dev)  # state, output token, base rank
    occupancy = torch.zeros(model.S, device=dev)
    for _ in range(batches):
        alpha, y = pool.batch(batch)
        logq, logpi, logA = model.fields(alpha)
        _, gamma = model.forward_backward(logq, logpi, logA, y, posterior=True)
        flat_y = y.reshape(-1)
        flat_gamma = gamma.reshape(-1, model.S)
        occupancy += flat_gamma.sum(0)
        for s in range(model.S):
            c = int(model.state_class[s])
            values = logq[:, :, c].reshape(-1, model.V)
            weighted = flat_gamma[:, s:s + 1] * values
            score[s].index_add_(0, flat_y, weighted)

    old = model.inv_perm.clone()
    new_inv = old.clone()
    # Assignment is CPU/C code and exact for the accumulated generalized-M objective.
    score_cpu = score.float().cpu().numpy()
    occ_cpu = occupancy.cpu()
    for s in range(model.S):
        if occ_cpu[s] < 1e-3:
            continue
        rows, cols = linear_sum_assignment(score_cpu[s], maximize=True)
        new_inv[s, torch.from_numpy(rows).to(dev)] = torch.from_numpy(cols).to(dev)
    new_perm = torch.empty_like(new_inv)
    tokens = torch.arange(model.V, device=dev)[None].expand_as(new_inv)
    new_perm.scatter_(1, new_inv, tokens)
    model.inv_perm.copy_(new_inv)
    model.perm.copy_(new_perm)
    changed = float((new_inv != old).float().mean())
    active = int((occupancy > 1e-3).sum())
    if was_training:
        model.train()
    return {"permutation_change": changed, "active_states": active,
            "min_occupancy": float(occupancy.min()), "max_occupancy": float(occupancy.max())}


def gate(row):
    return "PASS" if row["block_kl"] <= 0.10 and row["invalid_rate"] <= 0.03 else "FAIL"


def run(args, checkpoint_commit=None):
    torch.manual_seed(args.seed)
    if args.device == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.set_float32_matmul_precision("high")
    device = torch.device(args.device)
    hmm = build_phrase_hmm(V=args.vocab, seed=args.hmm_seed, device=device)
    train_gen = torch.Generator(device=device).manual_seed(92_000 + args.seed)
    train_pool = BeliefPool(hmm, args.context, args.T, args.pool, train_gen)
    em_gen = torch.Generator(device=device).manual_seed(93_000 + args.seed)
    em_pool = BeliefPool(hmm, args.context, args.T, args.pool, em_gen)
    eval_gen = torch.Generator(device=device).manual_seed(94_000 + args.seed)
    eval_pool = BeliefPool(hmm, args.context, args.T, max(args.pool, args.eval_contexts), eval_gen)
    eval_data = eval_pool.batch(args.eval_contexts)
    model = LearnedSCPSS(hmm.S, args.T, args.vocab, args.width, args.depth,
                         args.states, args.spectra, args.seed).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
    milestones = set(args.eval_milestones or [args.steps // 2, args.steps])
    curves, em_rows = [], []
    t0 = time.time()
    for step in range(1, args.steps + 1):
        alpha, y = train_pool.batch(args.batch)
        logq, logpi, logA = model.fields(alpha)
        lp, gamma = model.forward_backward(logq, logpi, logA, y, posterior=True)
        nll = -lp.mean()
        occ = gamma.mean((0, 1))
        balance = (occ * (occ.clamp_min(1e-9).log() + math.log(model.S))).sum()
        q = logq.exp()
        m = 0.5 * (q[:, :, 0] + q[:, :, 1])
        js = 0.5 * ((q[:, :, 0] * (logq[:, :, 0] - m.clamp_min(1e-9).log())).sum(-1) +
                    (q[:, :, 1] * (logq[:, :, 1] - m.clamp_min(1e-9).log())).sum(-1)).mean()
        loss = nll + args.balance_weight * balance - args.diversity_weight * js
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if (step >= args.perm_warmup and args.perm_every > 0 and
                step % args.perm_every == 0 and step < args.steps):
            info = update_permutations(model, em_pool, args.perm_batches, args.batch)
            info["step"] = step
            em_rows.append(info)
            print("S06L_EM " + json.dumps(info, sort_keys=True), flush=True)
            if checkpoint_commit is not None:
                checkpoint_commit()

        if step in milestones:
            ev = evaluate(model, hmm, eval_data, args)
            row = {"step": step, "loss": float(loss.detach()), "nll": float(nll.detach()),
                   "state_balance_kl": float(balance.detach()), "spectrum_js": float(js.detach()),
                   **ev}
            curves.append(row)
            print("S06L_MILESTONE " + json.dumps(row, sort_keys=True), flush=True)

    final = curves[-1]
    result = {
        "experiment": "S06-L learned spectrum-class semantic PSS",
        "depth": args.depth, "width": args.width, "T": args.T,
        "states": args.states, "spectra": args.spectra, "steps": args.steps,
        "parameters": sum(p.numel() for p in model.parameters()),
        "oracle_context": True, "oracle_emissions_or_transitions": False,
        "permutation_learning": "posterior-weighted Hungarian generalized EM",
        "stochastic_rounds": 1, "seconds": round(time.time() - t0, 2),
        "curve": curves, "permutation_updates": em_rows,
        **{k: v for k, v in final.items() if k not in ("step", "loss", "nll")},
    }
    result["verdict"] = gate(result)
    print("S06L_RESULT " + json.dumps(result, sort_keys=True), flush=True)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2, sort_keys=True)
    return result


def compile_text(result):
    lines = [
        "S06-L — learned two-spectrum semantic PSS",
        "=========================================",
        "Depth-4 oracle-context mechanism gate; no HMM emissions/transitions, teacher, or verifier.",
        "Exact likelihood; bijections learned by posterior-weighted Hungarian generalized EM.",
        "Gate: block KL <= 0.10 and invalid <= 3%.", "",
        f"S={result['states']} C={result['spectra']} depth={result['depth']} "
        f"width={result['width']} steps={result['steps']} params={result['parameters']:,}", "",
        f"final KL={result['block_kl']:.6f} invalid={result['invalid_rate']:.6f} "
        f"unique={result['unique_rate']:.6f} verdict={result['verdict']}", "",
        "Full result", "-----------", json.dumps(result, indent=2, sort_keys=True), "",
    ]
    return "\n".join(lines)


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--batch", type=int, default=96)
    p.add_argument("--eval-contexts", type=int, default=512)
    p.add_argument("--samples-per-ctx", type=int, default=8)
    p.add_argument("--pool", type=int, default=2048)
    p.add_argument("--context", type=int, default=16)
    p.add_argument("--T", type=int, default=4)
    p.add_argument("--vocab", type=int, default=512)
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--states", type=int, default=64)
    p.add_argument("--spectra", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--balance-weight", type=float, default=0.002)
    p.add_argument("--diversity-weight", type=float, default=0.01)
    p.add_argument("--perm-warmup", type=int, default=1000)
    p.add_argument("--perm-every", type=int, default=1000)
    p.add_argument("--perm-batches", type=int, default=16)
    p.add_argument("--eval-milestones", type=int, nargs="*", default=[])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--hmm-seed", type=int, default=1234)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default="")
    p.add_argument("--compiled-out", default="")
    p.add_argument("--smoke", action="store_true")
    return p


def main():
    args = build_parser().parse_args()
    if args.smoke:
        args.steps, args.batch, args.eval_contexts = 4, 4, 4
        args.samples_per_ctx, args.pool = 2, 16
        args.width, args.depth, args.states = 32, 1, 4
        args.perm_warmup, args.perm_every, args.perm_batches = 2, 2, 1
        args.eval_milestones = [2, 4]
    result = run(args)
    text = compile_text(result)
    print(text)
    if args.compiled_out:
        os.makedirs(os.path.dirname(args.compiled_out) or ".", exist_ok=True)
        with open(args.compiled_out, "w") as f:
            f.write(text)


if __name__ == "__main__":
    main()
