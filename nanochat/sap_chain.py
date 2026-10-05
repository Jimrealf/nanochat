"""
Exact chain-structured joints for the SAP v4 block heads.

A block head emits T tokens in one pass. If every slot is sampled from its own distribution,
the block is drawn from the product of marginals, whatever the head computed before the
readout. S01 tried to repair this with a stochastic cut inside the head, but it sampled the
cut's variables independently too, and its `sir_lattice` arm used a candidate chain only to
sharpen per-slot logits before sampling each slot on its own. The repair here is to make the
cut a normalised joint and to draw from that joint exactly:

  linear-chain CRF   p(s) ∝ exp(sum_j unary[j, s_j] + sum_j pair[j, s_j, s_{j+1}])
  HMM (tensor train) p(s) = sum_z pi(z_0) prod_j emit_j(s_j | z_j) prod_j A_j(z_j, z_{j+1})

Both are exact in O(J S^2) by the forward algorithm, and both are sampled exactly by a
forward filter followed by a backward (CRF) or ancestral (HMM) pass. Temperature 0 gives the
Viterbi path. The state sets are small (a top-K candidate lattice plus an escape state, or R
latent states), so a draw costs microseconds and runs no network layers: this is the "cheap
sampler" that keeps the head at one neural pass per block.

Every loop runs over the chain length, which is static, and every shape is fixed, so the
samplers capture under CUDA graphs. Nothing here calls .item().
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _draw(probs, generator=None):
    """One inverse-CDF draw per row of probs (..., S). Same rule as block_head.pick."""
    u = torch.rand(probs.shape[:-1], device=probs.device, generator=generator)
    cdf = probs.float().cumsum(-1)
    idx = torch.searchsorted(cdf, u.float()[..., None].contiguous()).squeeze(-1)
    return idx.clamp(max=probs.size(-1) - 1)


def chain_logz(unary, pair):
    """log normaliser of the chain. unary (N, J, S), pair (N, J-1, S, S). Returns (N,)."""
    alpha = unary[:, 0]
    for j in range(1, unary.size(1)):
        alpha = unary[:, j] + torch.logsumexp(alpha[:, :, None] + pair[:, j - 1], dim=1)
    return torch.logsumexp(alpha, dim=-1)


def chain_score(unary, pair, states):
    """Unnormalised log score of fully observed states (N, J)."""
    N, J, _ = unary.shape
    score = unary.gather(2, states[:, :, None]).squeeze(-1).sum(-1)
    if J > 1:
        rows = pair.gather(2, states[:, :-1, None, None].expand(N, J - 1, 1, pair.size(-1))).squeeze(2)
        score = score + rows.gather(2, states[:, 1:, None]).squeeze(-1).sum(-1)
    return score


def chain_logprob(unary, pair, states, observed=None):
    """log p(states at the observed slots); unobserved slots are summed out exactly.

    observed (N, J) bool or None (all observed). With every slot observed this is
    score - logZ; otherwise the observed slots are clamped and the rest marginalised, which is
    how a block with an unscored (zero-byte) token keeps an exact likelihood.
    """
    logz = chain_logz(unary, pair)
    if observed is None:
        return chain_score(unary, pair, states) - logz
    keep = F.one_hot(states, unary.size(-1)).bool() | ~observed[..., None]
    return chain_logz(unary.masked_fill(~keep, float("-inf")), pair) - logz


def chain_viterbi(unary, pair):
    """Most probable state sequence. Returns (N, J) long."""
    N, J, _ = unary.shape
    delta, back = unary[:, 0], []
    for j in range(1, J):
        best, arg = (delta[:, :, None] + pair[:, j - 1]).max(dim=1)
        back.append(arg)
        delta = unary[:, j] + best
    s = delta.argmax(-1)
    path = [s]
    for arg in reversed(back):
        s = arg.gather(1, s[:, None]).squeeze(1)
        path.append(s)
    return torch.stack(path[::-1], dim=1)


def chain_sample(unary, pair, n_samples=1, temperature=1.0, generator=None):
    """Exact draws: forward filter, then sample backwards. Returns (N, L, J) long."""
    N, J, S = unary.shape
    if temperature <= 0:
        return chain_viterbi(unary, pair)[:, None, :].expand(N, n_samples, J).contiguous()
    u, p = unary.float() / temperature, pair.float() / temperature
    alphas = [u[:, 0]]
    for j in range(1, J):
        alphas.append(u[:, j] + torch.logsumexp(alphas[-1][:, :, None] + p[:, j - 1], dim=1))
    out = torch.empty(N, n_samples, J, dtype=torch.long, device=unary.device)
    s = _draw(torch.softmax(alphas[-1], -1)[:, None, :].expand(N, n_samples, S), generator)
    out[:, :, J - 1] = s
    for j in range(J - 2, -1, -1):
        # p(s_j | s_{j+1}) ∝ exp(alpha_j(s_j) + pair_j[s_j, s_{j+1}])
        col = p[:, j].gather(2, s[:, None, :].expand(N, S, n_samples))      # (N, S, L)
        logits = (alphas[j][:, :, None] + col).transpose(1, 2)               # (N, L, S)
        s = _draw(torch.softmax(logits, -1), generator)
        out[:, :, j] = s
    return out


def hmm_loglik(log_pi, log_A, log_obs):
    """log sum_z pi(z_0) prod_j obs_j(z_j) prod_j A_j(z_j, z_{j+1}).

    log_pi (N, R); log_A (N, J-1, R, R) row-normalised; log_obs (N, J, R) is the log
    probability of what slot j emitted given each state (0 for a slot that is summed out).
    """
    a = log_pi + log_obs[:, 0]
    for j in range(1, log_obs.size(1)):
        a = log_obs[:, j] + torch.logsumexp(a[:, :, None] + log_A[:, j - 1], dim=1)
    return torch.logsumexp(a, dim=-1)


def hmm_sample(log_pi, log_A, log_emit, n_samples=1, temperature=1.0, generator=None):
    """Ancestral draw of the emitted states. log_emit (N, J, R, S) normalised over S.

    Returns (N, L, J) long. At temperature 0 it returns the emissions on the jointly most
    probable (state, emission) path.
    """
    N, J, R, S = log_emit.shape
    if temperature <= 0:
        best_e, best_s = log_emit.max(-1)                                   # (N, J, R)
        unary = best_e.clone()
        unary[:, 0] = unary[:, 0] + log_pi
        z = chain_viterbi(unary, log_A)                                      # (N, J)
        s = best_s.gather(2, z[:, :, None]).squeeze(-1)
        return s[:, None, :].expand(N, n_samples, J).contiguous()
    t = float(temperature)
    z = _draw(torch.softmax(log_pi.float() / t, -1)[:, None, :].expand(N, n_samples, R), generator)
    out = torch.empty(N, n_samples, J, dtype=torch.long, device=log_emit.device)
    for j in range(J):
        em = log_emit[:, j].float().gather(1, z[:, :, None].expand(N, n_samples, S))    # (N, L, S)
        out[:, :, j] = _draw(torch.softmax(em / t, -1), generator)
        if j < J - 1:
            tr = log_A[:, j].float().gather(1, z[:, :, None].expand(N, n_samples, R))   # (N, L, R)
            z = _draw(torch.softmax(tr / t, -1), generator)
    return out
