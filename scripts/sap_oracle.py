"""
SAP v4 Stage 0: how much dependence real text puts inside a T-token block, and how much of it
each structured block-head family could carry, measured on a trained dense model's own joint.

These are properties of FineWeb-Edu text as the best model we have sees it. Every fit below is
labelled: a FREE fit (its own parameters per context) is an upper bound on what a head of that
family could reach; a CONSTRAINED fit shares its pair projections across contexts the way a
real head does, and the gap between the two is the price of computing them from h_t.

  O1   block total correlation at T in {2, 4, 8}: sum_k CE(y_k under the Monte Carlo
       marginal p(y_k | ctx)) - NLL_AR(block). This is exactly what a product-of-marginals
       head loses over the AR model, i.e. the most any sampling-aware mechanism can recover.
       Pre-registered (sap_research_plan.md v4): TC_4 < 5% of NLL_4 means coherence is not
       the bottleneck.
  O2   lattice coverage: P(y_k in the top-K of the marginal), K in {16, 64, 256}.
  O2b  pair capture at T=2 on the exact joint over the 64 most likely first tokens and the 64
       most likely second tokens (+ an escape column): rank-r CRF (free and constrained) and
       CP with Z codes, as the fraction of I(y1; y2 | ctx, y1 in top-64) recovered.
  O2c  cut mutual information I(first half; second half | ctx) at T=4 and 8. A tensor train
       needs bond dimension of at least e^MI to carry it.
  O3   (with --tables) the share of the pair MI left inside corpus classes, which corpus_code
       cannot carry.

    python -m scripts.sap_oracle --checkpoint-dir /vol/out/s00_sap/d8/B1_dense_s1 \\
        --tokenizer-dir /vol/tokenizer --data-dir /vol/data --tables /vol/out/s03_sap/tables_V32k.pt
    python -m scripts.sap_oracle --smoke          # tiny random model, checks every code path
"""

from __future__ import annotations

import argparse
import json
import math
import time

import torch
import torch.nn.functional as F


# ----------------------------------------------------------------------------- data
@torch.no_grad()
def val_contexts(tokenizer, data_dir, n, ctx_len, T, seq_len, device, seed):
    """n (context, block) pairs from validation rows: ctx (n, ctx_len), block (n, T)."""
    from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
    g = torch.Generator().manual_seed(seed)
    loader = tokenizing_distributed_data_loader_bos_bestfit(tokenizer, 8, seq_len, split="val",
                                                            data_dir=data_dir)
    ctxs, blocks = [], []
    for x, y in loader:
        for r in range(x.size(0)):
            t = int(torch.randint(ctx_len - 1, x.size(1) - T, (1,), generator=g))
            ctxs.append(x[r, t - ctx_len + 1:t + 1])
            blocks.append(y[r, t:t + T])
        if len(ctxs) >= n:
            break
    return torch.stack(ctxs[:n]).to(device), torch.stack(blocks[:n]).to(device)


def last_logits(model, seq):
    """Next-token logits after each row of seq, without materialising (B, L, V)."""
    hid = model(seq, skip_logits=True)
    return model._sap_readout(hid[:, -1]).float()


def tf_logprobs(model, ctx, block):
    """log p(block_k | ctx, block_<k) for every k: (n, T)."""
    seq = torch.cat([ctx, block[:, :-1]], dim=1)
    hid = model(seq, skip_logits=True)[:, ctx.size(1) - 1:]
    lp = torch.log_softmax(model._sap_readout(hid.reshape(-1, hid.size(-1))).float(), -1)
    return lp.gather(-1, block.reshape(-1, 1)).view(block.shape)


# ----------------------------------------------------------------------------- O1 / O2 / O2c
@torch.no_grad()
def o1_o2_o2c(model, ctx, block, S, chunk, gen):
    """Monte Carlo marginals of every slot, coverage ranks, and the cut MI of each half split."""
    n, T = block.shape
    log_marg = torch.zeros(n, T, device=ctx.device)
    rank = torch.zeros(n, T, dtype=torch.long, device=ctx.device)
    cut = {}
    halves = [h for h in (1, 2, 4) if 2 * h <= T]
    cut_lp = {h: torch.zeros(n, device=ctx.device) for h in halves}
    for i in range(0, n, chunk):
        c, b = ctx[i:i + chunk], block[i:i + chunk]
        m = c.size(0)
        seq = c.repeat_interleave(S, dim=0)
        for k in range(T):
            probs = torch.softmax(last_logits(model, seq), dim=-1)                # (m*S, V)
            pk = probs.view(m, S, -1).mean(1)                                     # MC marginal
            pd = pk.gather(1, b[:, k:k + 1])
            log_marg[i:i + m, k] = pd.squeeze(1).clamp_min(1e-30).log()
            rank[i:i + m, k] = (pk > pd).sum(-1)
            nxt = torch.multinomial(probs, 1, generator=gen)
            seq = torch.cat([seq, nxt], dim=1)
            if k + 1 in halves:
                # p(data second half | ctx, sampled first half), averaged over the samples
                h = k + 1
                tail = b[:, h:2 * h].repeat_interleave(S, dim=0)
                lp = tf_logprobs(model, seq, tail).sum(-1).view(m, S)
                cut_lp[h][i:i + m] = torch.logsumexp(lp, dim=1) - math.log(S)
    tf = tf_logprobs(model, ctx, block)
    for h in halves:
        cond = tf[:, h:2 * h].sum(-1)                                             # given data first half
        cut[2 * h] = (cond - cut_lp[h])                                           # per context, nats
    return log_marg, rank, tf, cut


# ----------------------------------------------------------------------------- O2b
@torch.no_grad()
def pair_joint(model, ctx, K, chunk):
    """Exact joint over (top-K first tokens) x (top-K second tokens + escape), per context.

    Rows are restricted to y1 in the top-K (renormalised); the escape column holds the
    second-token mass outside its top-K. Returns P (n, K, K+1), cand1 (n, K), cand2 (n, K),
    and the first-token mass the restriction keeps.
    """
    n = ctx.size(0)
    Ps, c1s, c2s, masses = [], [], [], []
    for i in range(0, n, chunk):
        c = ctx[i:i + chunk]
        m = c.size(0)
        p1 = torch.softmax(last_logits(model, c), -1)
        top_p, top_a = p1.topk(K, dim=-1)                                          # (m, K)
        seq = torch.cat([c.repeat_interleave(K, dim=0), top_a.reshape(-1, 1)], dim=1)
        p2 = torch.softmax(last_logits(model, seq), -1).view(m, K, -1)             # (m, K, V)
        w = top_p / top_p.sum(-1, keepdim=True)
        q2 = (w[..., None] * p2).sum(1)
        cand2 = q2.topk(K, dim=-1).indices                                          # (m, K)
        inner = p2.gather(2, cand2[:, None, :].expand(m, K, K))                     # (m, K, K)
        P = torch.cat([inner, (1 - inner.sum(-1, keepdim=True)).clamp_min(0)], dim=-1) * w[..., None]
        Ps.append(P)
        c1s.append(top_a)
        c2s.append(cand2)
        masses.append(top_p.sum(-1))
    return torch.cat(Ps), torch.cat(c1s), torch.cat(c2s), torch.cat(masses)


def _kl(P, logQ):
    return (P * (P.clamp_min(1e-30).log() - logQ)).sum((-1, -2))


def mutual_info(P):
    indep = P.sum(-1, keepdim=True) * P.sum(-2, keepdim=True)
    return _kl(P, indep.clamp_min(1e-30).log())


def fit_crf_free(P, r, steps=400, lr=0.05):
    """Free per-context rank-r CRF: log Q = u_a + v_b + sum_r U_ar V_br - log Z."""
    n, A, B = P.shape
    g = torch.Generator(device=P.device).manual_seed(0)
    u = P.sum(-1).clamp_min(1e-30).log().clone().requires_grad_(True)
    v = P.sum(-2).clamp_min(1e-30).log().clone().requires_grad_(True)
    U = (0.1 * torch.randn(n, A, r, device=P.device, generator=g)).requires_grad_(True)
    W = (0.1 * torch.randn(n, B, r, device=P.device, generator=g)).requires_grad_(True)
    opt = torch.optim.Adam([u, v, U, W], lr=lr)
    for _ in range(steps):
        logit = u[:, :, None] + v[:, None, :] + U @ W.transpose(1, 2)
        logQ = logit - torch.logsumexp(logit.flatten(1), -1)[:, None, None]
        loss = _kl(P, logQ).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        logit = u[:, :, None] + v[:, None, :] + U @ W.transpose(1, 2)
        return _kl(P, logit - torch.logsumexp(logit.flatten(1), -1)[:, None, None])


def fit_crf_shared(P, e1, e2, r, steps=600, lr=0.03, held_out=None):
    """Constrained rank-r CRF: pair term sum_r w_r(ctx) (e1 Pu)_r (e2 Pv)_r with Pu, Pv shared
    across contexts (e = normalised embeddings of the candidates), free unaries and gates per
    context, as in lat_crf. With held_out (index tensor), Pu/Pv are fit on the rest only and the
    held-out contexts refit just their own unaries and gates."""
    n, A, B = P.shape
    d = e1.size(-1)
    g = torch.Generator(device=P.device).manual_seed(1)
    Pu = (torch.randn(d, r, device=P.device, generator=g) / math.sqrt(d)).requires_grad_(True)
    Pv = (torch.randn(d, r, device=P.device, generator=g) / math.sqrt(d)).requires_grad_(True)

    def _fit(idx, train_shared, steps_):
        u = P[idx].sum(-1).clamp_min(1e-30).log().clone().requires_grad_(True)
        v = P[idx].sum(-2).clamp_min(1e-30).log().clone().requires_grad_(True)
        w = torch.ones(len(idx), r, device=P.device).requires_grad_(True)
        params = [u, v, w] + ([Pu, Pv] if train_shared else [])
        opt = torch.optim.Adam(params, lr=lr)
        for _ in range(steps_):
            a, b = e1[idx] @ Pu, e2[idx] @ Pv                                      # (m, A, r), (m, B-1, r)
            pair = torch.einsum("mar,mr,mbr->mab", a, w, b)
            pair = F.pad(pair, (0, 1))                                             # escape column
            logit = u[:, :, None] + v[:, None, :] + pair
            logQ = logit - torch.logsumexp(logit.flatten(1), -1)[:, None, None]
            loss = _kl(P[idx], logQ).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        return _kl(P[idx], logQ.detach())

    all_idx = torch.arange(n, device=P.device)
    if held_out is None:
        return _fit(all_idx, True, steps), None
    train_idx = all_idx[~torch.isin(all_idx, held_out)]
    kl_train = _fit(train_idx, True, steps)
    kl_held = _fit(held_out, False, steps)
    return kl_train, kl_held


def fit_cp_free(P, Z, iters=200):
    """Free per-context mixture of Z products fit by EM on the soft counts P."""
    n, A, B = P.shape
    g = torch.Generator(device=P.device).manual_seed(2)
    pi = torch.full((n, Z), 1.0 / Z, device=P.device)
    f = torch.softmax(torch.randn(n, Z, A, device=P.device, generator=g), -1)
    h = torch.softmax(torch.randn(n, Z, B, device=P.device, generator=g), -1)
    for _ in range(iters):
        joint = pi[:, :, None, None] * f[..., None] * h[:, :, None, :]            # (n, Z, A, B)
        resp = joint / joint.sum(1, keepdim=True).clamp_min(1e-30)
        wz = resp * P[:, None]
        pi = wz.sum((-1, -2))
        f = wz.sum(-1) / pi[..., None].clamp_min(1e-30)
        h = wz.sum(-2) / pi[..., None].clamp_min(1e-30)
    Q = (pi[:, :, None, None] * f[..., None] * h[:, :, None, :]).sum(1)
    return _kl(P, Q.clamp_min(1e-30).log())


def class_residual(P, cand1, cand2, classes):
    """I(y1; y2 | c1, c2) on the lattice joint (the escape column is its own class).

    Within each (class c1, class c2) cell of the joint, the MI between the tokens; summed with
    the cell masses. Vectorised: row sums by column class, column sums by row class, cell masses.
    """
    n, A, B = P.shape
    C = int(classes.max()) + 1
    c1 = classes[cand1]                                                           # (n, A)
    c2 = torch.cat([classes[cand2], torch.full((n, 1), C, device=P.device, dtype=classes.dtype)], 1)
    o1 = F.one_hot(c1, C + 1).float()                                             # (n, A, C+1)
    o2 = F.one_hot(c2, C + 1).float()                                             # (n, B, C+1)
    row_by_c2 = P @ o2                                                            # (n, A, C+1)
    col_by_c1 = o1.transpose(1, 2) @ P                                            # (n, C+1, B)
    mass = o1.transpose(1, 2) @ P @ o2                                            # (n, C+1, C+1)
    rs = row_by_c2.gather(2, c2[:, None, :].expand(n, A, B))                      # P(a, cell cols)
    cs = col_by_c1.gather(1, c1[:, :, None].expand(n, A, B))                      # P(cell rows, b)
    ms = mass.gather(1, c1[:, :, None].expand(n, A, C + 1)).gather(2, c2[:, None, :].expand(n, A, B))
    term = P * (P.clamp_min(1e-30).log() + ms.clamp_min(1e-30).log()
                - rs.clamp_min(1e-30).log() - cs.clamp_min(1e-30).log())
    return term.sum((-1, -2))


# ----------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint-dir", type=str, default=None)
    p.add_argument("--step", type=int, default=None)
    p.add_argument("--tokenizer-dir", type=str, default=None)
    p.add_argument("--data-dir", type=str, default=None)
    p.add_argument("--tables", type=str, default="")
    p.add_argument("--contexts", type=int, default=512)
    p.add_argument("--pair-contexts", type=int, default=256)
    p.add_argument("--ctx-len", type=int, default=256)
    p.add_argument("--samples", type=int, default=256)
    p.add_argument("--T", type=int, default=8)
    p.add_argument("--chunk", type=int, default=8, help="contexts per sampling chunk (x samples rows)")
    p.add_argument("--out", type=str, default="out/s03_sap/oracle_d8.json")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    dev = torch.device(args.device)
    gen = torch.Generator(device=dev).manual_seed(0)
    t0 = time.time()

    if args.smoke:
        from nanochat.gpt import GPT, GPTConfig
        cfg = GPTConfig(sequence_len=128, vocab_size=256, n_layer=2, n_head=2, n_kv_head=2, n_embd=64,
                        window_pattern="L")
        model = GPT(cfg).to(dev)
        model.init_weights()
        with torch.no_grad():
            model.lm_head.weight.add_(torch.randn_like(model.lm_head.weight) * 0.3)
        args.contexts, args.pair_contexts, args.samples, args.ctx_len = 8, 6, 8, 16
        ctx = torch.randint(0, 256, (args.contexts, args.ctx_len), device=dev)
        block = torch.randint(0, 256, (args.contexts, args.T), device=dev)
        token_bytes = torch.ones(256, device=dev)
        tables = None
        args.out = args.out.replace(".json", "_smoke.json")
    else:
        from nanochat.checkpoint_manager import build_model, find_last_step
        from nanochat.tokenizer import get_token_bytes
        step = args.step if args.step is not None else find_last_step(args.checkpoint_dir)
        model, tok, _ = build_model(args.checkpoint_dir, step, dev, phase="eval", tokenizer_dir=args.tokenizer_dir)
        ctx, block = val_contexts(tok, args.data_dir, args.contexts, args.ctx_len, args.T,
                                  model.config.sequence_len, dev, seed=0)
        token_bytes = get_token_bytes(device=dev, tokenizer_dir=args.tokenizer_dir).float()
        tables = torch.load(args.tables, map_location="cpu", weights_only=False) if args.tables else None
    model.eval()
    V = model.config.vocab_size
    out = {"settings": {k: v for k, v in vars(args).items()}, "notes": {
        "O1": "Monte Carlo marginals overstate CE slightly (log of an average); compare S=64 vs S=all",
        "O2b": "rows restricted to y1 in the model's top-64 (mass reported); free fits are upper bounds",
        "O2c": "log p(second half | data first half) - log mean_s p(second half | sampled first half), "
               "on data blocks: equals the MI when the data follow the model, so it can dip below 0 "
               "only under model/data mismatch",
    }}

    # O1 / O2 / O2c
    log_marg, rank, tf, cut = o1_o2_o2c(model, ctx, block, args.samples, args.chunk, gen)
    nbytes = token_bytes[block].clamp_min(1)
    o1 = {}
    for T in (2, 4, 8):
        if T > args.T:
            continue
        ce_marg = -log_marg[:, :T].sum(-1)
        nll = -tf[:, :T].sum(-1)
        tc = ce_marg - nll
        bytes_T = nbytes[:, :T].sum(-1)
        o1[f"T{T}"] = {
            "tc_nats_per_block": tc.mean().item(),
            "nll_ar_nats_per_block": nll.mean().item(),
            "tc_over_nll": (tc.mean() / nll.mean()).item(),
            "tc_bits_per_byte": (tc.sum() / (math.log(2) * bytes_T.sum())).item(),
            "ar_bits_per_byte": (nll.sum() / (math.log(2) * bytes_T.sum())).item(),
            "per_slot_marginal_ce": (-log_marg[:, :T]).mean(0).tolist(),
            "per_slot_teacher_forced_nll": (-tf[:, :T]).mean(0).tolist(),
        }
    out["O1"] = o1
    out["O1_pre_registered_coherence_bottleneck"] = (o1.get("T4", {}).get("tc_over_nll", 0.0) >= 0.05)
    out["O2_coverage"] = {f"K{K}": (rank < K).float().mean(0).tolist() for K in (16, 64, 256)}
    out["O2c_cut_mi_nats"] = {f"T{T}": {"mean": v.mean().item(), "tt_rank_floor": math.exp(v.mean().item())}
                              for T, v in cut.items()}
    print(json.dumps({"O1": o1, "O2c": out["O2c_cut_mi_nats"]}, indent=2), flush=True)

    # O2b
    K = min(64, V - 1)
    for n_pairs in sorted({args.pair_contexts, 2 * args.pair_contexts}):
        n_pairs = min(n_pairs, ctx.size(0))
        P, c1, c2, mass = pair_joint(model, ctx[:n_pairs], K, max(1, args.chunk // 2))
        mi = mutual_info(P)
        res = {"contexts": n_pairs, "first_token_mass_kept": mass.mean().item(),
               "pair_mi_nats": mi.mean().item(), "entries_per_context": K * (K + 1)}
        fits = {}
        for r in (2, 4, 8, 16):
            kl = fit_crf_free(P, r)
            npar = 2 * K + 1 + r * (2 * K + 1)
            fits[f"crf_free_r{r}"] = {"captured": (1 - kl.mean() / mi.mean()).item(),
                                      "params_per_context": npar, "interpolates": npar >= K * (K + 1)}
        E = F.normalize(model.transformer.wte.weight.detach().float(), dim=-1)
        held = torch.arange(0, n_pairs, 4, device=dev)
        for r in (8, 32):
            kl_tr, kl_ho = fit_crf_shared(P, E[c1], E[c2], r, held_out=held)
            keep = ~torch.isin(torch.arange(n_pairs, device=dev), held)
            fits[f"crf_shared_r{r}"] = {
                "captured_train": (1 - kl_tr.mean() / mi[keep].mean()).item(),
                "captured_held_out": (1 - kl_ho.mean() / mi[held].mean()).item(),
                "shared_params": 2 * E.size(-1) * r, "params_per_context": 2 * K + 1 + r}
        for Z in (4, 16, 64):
            kl = fit_cp_free(P, Z)
            npar = Z * (2 * K + 2)
            fits[f"cp_free_Z{Z}"] = {"captured": (1 - kl.mean() / mi.mean()).item(),
                                     "params_per_context": npar, "interpolates": npar >= K * (K + 1)}
        res["fits"] = fits
        if tables is not None:
            classes = tables["classes"].to(dev).long()
            resid = class_residual(P, c1, c2, classes)
            res["O3_within_class_share"] = (resid.mean() / mi.mean()).item()
        out[f"O2b_n{n_pairs}"] = res
        print(json.dumps({f"O2b_n{n_pairs}": res}, indent=2), flush=True)
    out["seconds"] = round(time.time() - t0, 1)

    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
