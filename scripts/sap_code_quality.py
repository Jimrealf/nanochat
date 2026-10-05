"""
S11 diagnostic: are learned codes separators?

The Bridge LM samples codes coarse-to-fine and assumes codes split the block (given the codes at an
interval's ends, its interior is independent of the outside). Oracle codes (the argmax of the exact
HMM filtering belief) reach 24% invalid at T=64; learned codes (k-means of a trained AR model's
states, or KL k-means of its predicted next-token distributions) reach 77 to 85%. This script
separates code quality from prior learnability: for each code type it fits the best Markov model
OVER THE CODES by counting (code transitions and token emissions on 200k sampled positions), then
samples it exactly with the interface-first bridge sampler. If that fails, the codes are not
Markov separators and no prior can fix it.

Reported per code type: codes used, H(true state | code) in nats (0 = the code determines the
state), and the exact-bridge invalid rate at T = 16 and 64.

    python -m scripts.sap_code_quality --ar-steps 4000 --K 512
"""
from __future__ import annotations

import argparse
import math
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from nanochat.bridge import BridgeLM, filtered_argmax_states
from scripts.sap_bridge_oracle import bridge_sample
from scripts.sap_synthetic import build_phrase_hmm, filter_states, sample_sequences, true_block_logprob


def train_ar(hmm, T, steps, width, depth, device, gen):
    """The Bridge LM's stage-1 AR model, trained alone on toy blocks."""
    model = BridgeLM(hmm.S, T, hmm.E.size(1), width=width, depth=depth, codes="ar", K=8, ar_steps=10 ** 9).to(device)
    opt = torch.optim.AdamW([p for n, p in model.named_parameters() if n.startswith("ar")], lr=3e-4,
                            betas=(0.9, 0.95), weight_decay=0.01)
    for step in range(steps):
        seq = sample_sequences(hmm, 96, 16 + T, gen)
        alpha = filter_states(hmm, seq[:, :16])[:, -1]
        loss = model.ar_nll(alpha, seq[:, 16:]).mean() / T
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if (step + 1) % 1000 == 0:
            print(f"  AR step {step + 1}: {loss.item():.3f} nats/token", flush=True)
    return model.eval()


@torch.no_grad()
def kmeans(x, K, kl, iters=30):
    cb = x[torch.randperm(x.size(0), device=x.device)[:K]].clone()
    for _ in range(iters):
        a = (x @ cb.clamp_min(1e-12).log().t()).argmax(1) if kl else torch.cdist(x, cb).argmin(1)
        for j in torch.unique(a):
            cb[j] = x[a == j].mean(0)
    return cb, (lambda y: (y @ cb.clamp_min(1e-12).log().t()).argmax(1) if kl else torch.cdist(y, cb).argmin(1))


@torch.no_grad()
def code_hmm(codes, tokens, first_alpha_codes, K, V):
    """Count-based Markov model over codes: transitions, emissions (token given code), start."""
    A = torch.full((K, K), 1e-6, dtype=torch.float64)
    E = torch.full((K, V), 1e-6, dtype=torch.float64)
    A.index_put_((codes[:, :-1].flatten(), codes[:, 1:].flatten()), torch.ones(codes[:, 1:].numel(), dtype=torch.float64),
                 accumulate=True)
    E.index_put_((codes.flatten(), tokens.flatten()), torch.ones(codes.numel(), dtype=torch.float64), accumulate=True)
    return SimpleNamespace(A=(A / A.sum(1, keepdim=True)).float(), E=(E / E.sum(1, keepdim=True)).float(), S=K)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ar-steps", type=int, default=4000)
    p.add_argument("--K", type=int, default=512)
    p.add_argument("--fit-T", type=int, default=16)
    p.add_argument("--fit-rows", type=int, default=12000)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    torch.manual_seed(0)
    dev = torch.device(args.device)
    hmm = build_phrase_hmm(V=512, seed=1234, device=dev)
    gen = torch.Generator(device=dev).manual_seed(11)
    T = args.fit_T
    ar = train_ar(hmm, T, args.ar_steps, 128, 4, dev, gen)

    seq = sample_sequences(hmm, args.fit_rows, 16 + T, gen)
    alpha = filter_states(hmm, seq[:, :16])[:, -1]
    y = seq[:, 16:]
    with torch.no_grad():
        h = torch.cat([ar.ar_states(alpha[i:i + 1024], y[i:i + 1024])[:, 1:] for i in range(0, y.size(0), 1024)])
        pred = ar.ar_head(h).float().softmax(-1)
    truth = filtered_argmax_states(hmm, alpha, y)
    feats = {"euclidean k-means of AR states": (h.float(), False),
             "KL k-means of AR next-token predictions": (pred, True)}
    code_sets = {"oracle (filtered argmax state)": truth}
    for name, (x, kl) in feats.items():
        flat = x.reshape(-1, x.size(-1))
        sub = flat[torch.randperm(flat.size(0), device=dev)[:60000]]
        _, assign = kmeans(sub, args.K, kl)
        code_sets[name] = torch.cat([assign(c) for c in flat.split(65536)]).view(y.shape)

    print(f"\ncode quality (K={args.K}, fit on {args.fit_rows} rows of T={T}):")
    for name, codes in code_sets.items():
        K = int(codes.max()) + 1
        joint = torch.zeros(K, hmm.S, device=dev)
        joint.index_put_((codes.flatten(), truth.flatten()), torch.ones(codes.numel(), device=dev), accumulate=True)
        pc = joint.sum(1, keepdim=True)
        cond = (joint / pc.clamp_min(1)).clamp_min(1e-12)
        h_state = float(-(joint * cond.log()).sum() / joint.sum())
        model = code_hmm(codes.cpu(), y.cpu(), None, K, 512)
        res = []
        # Context-free comparison for every code type: start from the codes seen at the block start,
        # and call a block valid if the true HMM can produce it from some state (uniform start).
        start = torch.bincount(codes[:, 0].cpu(), minlength=K).float()
        start = (start / start.sum())[None].expand(2048, -1)
        anywhere = torch.full((2048, hmm.S), 1.0 / hmm.S)
        hmm_cpu = SimpleNamespace(A=hmm.A.cpu(), E=hmm.E.cpu(), S=hmm.S)
        for Tg in (16, 64):
            toks, _ = bridge_sample(model, start, Tg, torch.Generator().manual_seed(Tg))
            lp = true_block_logprob(hmm_cpu, anywhere, toks)
            res.append(float((~torch.isfinite(lp)).double().mean()))
        print(f"  {name:45s} codes used {int((pc > 0).sum()):4d}  H(state|code) {h_state:.3f}  "
              f"exact-bridge invalid T=16 {res[0]:.3f}  T=64 {res[1]:.3f}")


if __name__ == "__main__":
    main()
