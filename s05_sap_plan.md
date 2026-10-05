# S05 — one-round correlated noise field with sample-aware training

## Non-negotiable inference contract

One random object is sampled before prediction, one fixed-depth block head is evaluated, and all
`T` final token distributions and token draws are produced concurrently. There is no draft-token,
anchor, confidence, tree, denoising, or resampling loop:

\[
\epsilon\sim\mathcal N(0,I)^{q\times d_z},\qquad
F_i=\frac1{\sqrt q}\sum_{a=1}^q\Phi_{ia}G_a(h)W\epsilon_a,
\]
\[
s_{1:T}=D(h_{1:W},\,b_{1:T}(h)+F_{1:T}),\qquad
y_i\sim p(y_i\mid s_i)\quad\text{for every }i\text{ in parallel}.
\]

The draw is high-bandwidth (`q` continuous factors), but remains one vectorised sampling event.
Shared basis coefficients make positions correlated before the categorical token draws.

## Why this mechanism survived the funnel

Thirty-six candidate mechanisms were screened against four hard constraints: sequential sampling
depth exactly one; no target-only posterior at inference; a trainable joint objective; and an
inference cost capable of beating dense AR wall clock. The pool comprised compact mixtures (6),
posterior/VAE variants (5), copula/flow variants (6), exact graphical models (5), iterative or
tree samplers (6), proposal/reranking methods (4), and sample-scoring methods (4).

- Compact mixtures and VAE variants were removed by the measured S00/S04 capacity and
  prior-posterior failures.
- CRF, tensor-train, tree, diffusion, Jacobi and anchor methods were removed because their sample
  depends on token-conditioned rounds, even if each round is parallel.
- Multi-proposal reranking was removed because `M` full 2,048-token proposals spends away the
  intended speedup.
- Discrete normalising flows and copulas were removed at this stage because they either hide a
  sequential inverse or lack a tractable, differentiable categorical likelihood at this scale.
- Deterministic soft messages were removed because they do not alter the product-of-marginals
  sampling graph.

Two sample-scoring variants survived analysis: a characteristic-kernel/MMD loss and the energy
score. Energy is the first experiment because it is linear in samples, works on the whole realised
block, and already has a local implementation to audit. MMD remains a fallback only if energy has
useful sensitivity but unstable scale; it is not a post-failure hyperparameter arm.

## Training objective

For `K=4` prior field draws, train the actual inference prior with a Monte-Carlo marginal
likelihood:

\[
\mathcal L_{MC}=-\log\frac1K\sum_{k=1}^K
\prod_i p(y_i\mid h,\epsilon^{(k)}).
\]

For two of those fields, sample actual categorical token ids by full-vocabulary Gumbel-max. The
forward value is hard and therefore matches inference. The backward path is a top-16 soft
straight-through relaxation, and a whole-block energy score compares the concatenated sampled
token embeddings with the observed block. The embedding table is detached so it cannot collapse
the metric:

\[
\mathcal L=\mathcal L_{MC}+0.25\left[
\tfrac12\sum_{k=1}^2\lVert \hat e^{(k)}-e(y)\rVert
-\tfrac12\lVert \hat e^{(1)}-\hat e^{(2)}\rVert\right]/\sqrt T.
\]

This repairs P3's key mismatch: P3 scored continuous decoder states and never trained on realised
token identities. S05's energy forward pass sees the same hard categorical object that inference
emits. The soft path is only a gradient estimator, not the generated representation.

## Cost, closest work, and conference bar

- Inference: one field projection, one fixed-depth decoder, one vocabulary readout, `T` parallel
  categorical draws. Sequential sampling depth is exactly one.
- Training: four decoder/readout evaluations plus two sparse soft embedding projections on the
  sampled fraction of positions. This is a training-only cost and is explicitly reported.
- Closest local references are Condor's coupled one-step noise, straight-through differentiable
  scheduled sampling, and one-step continuous denoising LMs. The proposed delta is a positional
  low-rank correlated field for an AR trunk's `T=L` block plus a joint proper score evaluated on
  realised discrete token blocks. A novelty claim is not made until the toy gate survives.
- The paper bar remains either better quality at matched cost, or quality neutrality plus measured
  wall-clock speedup. Lower sequential depth alone does not count.

## Pre-registered gate

1. Phrase-HMM, `T=4`, seed 0, 8,000 steps; `q=8`, `d_z=64`, `K=4`, top-k backward width 16,
   energy weight 0.25.
2. Advance only if MC block KL is <= 0.10 nats/block and impossible generations are <= 3%.
   Also require nonzero field sensitivity; a likelihood pass with ignored noise is a kill.
3. Only after passing, confirm seed 1. Only after both seeds pass, run `T=L=2048`, depth 8 at a
   cost-matched budget, reusing the existing dense result rather than retraining it.
4. Depth-8 gates remain: trunk BPB within 1% of dense, generated reference PPL within 10% of its
   AR generation, no diversity collapse, and at least 2x batch-16 decode speed in the symmetric
   benchmark.

Failure kills the correlated Gaussian field plus MC-likelihood/energy training combination. It
does not justify a rank, sample-count, top-k, or loss-weight sweep; the next proposal must change
the joint family or the proper scoring rule.

## Result — 2026-10-03

The `nanochat2` seed-0 run completed all 8,000 steps:

| metric | result | gate / reference |
|---|---:|---:|
| MC block KL | 3.3549 nats/block | <= 0.10 |
| impossible generations | 53.86% | <= 3% |
| field sensitivity | 1.0662 nats/block | nonzero required |
| true total correlation | 4.270 nats/block | diagnostic |
| AR block KL | 0.013 nats/block | diagnostic |
| AR impossible rate | 0.049% | diagnostic |
| trunk NTP excess | 0.001 nats/token | diagnostic |

**KILL.** The field is used, but it removes only about 0.915 nats/block (21%) of the independent
factorisation's total-correlation penalty and still emits an impossible phrase more than half the
time. It is also materially worse than S04's exact categorical field (KL 1.7078, invalid 33.94%).
This localises the failure to credit assignment: Euclidean energy between sampled token embeddings
is too weak a proxy for discrete sequence compatibility, even though its hard forward pass matches
inference. Per the registration, seed 1, rank/weight/sample sweeps, and depth 8 were not run. The
compiled artifact is `s05_sap_compiled.log`.
