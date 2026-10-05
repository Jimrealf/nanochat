# S04 — exact one-shot positional plan field

## Claim being tested

Generate a block from one shared random decision and one fixed-depth neural head evaluation:

\[
p(y\mid h)=\sum_{c=1}^{C}p(c\mid h)\prod_{i=1}^{T}
p\!\left(y_i\mid h,\;\Phi_i\,G(h)\,A_c\right).
\]

`c` is one categorical plan sampled once per block. `Phi` is a learned low-rank positional
basis, `G(h)` gates its channels from the trunk context, and `A_c` maps a plan into a distinct
field over all output positions. Conditional on the plan, every token distribution and every
token draw is evaluated in parallel. There is no token-dependent refinement or resampling loop.

This differs from the measured `cp` head because `cp` broadcasts one component vector to every
slot and asks the decoder to discover all position-specific consequences. It differs from P1/P2
because its mixture likelihood is exact: training takes a log-sum-exp over the same plan prior
used at inference. The posterior responsibility over plans is soft, but inference samples one
hard plan.

## Cost and conference bar

- Inference: one trunk pass, one categorical plan draw, one ordinary block-head evaluation and
  `T` parallel vocabulary readouts. Sequential sampling depth is one.
- Training: exact likelihood evaluates `C` component heads. This is acceptable for the synthetic
  gate; any depth-8 arm must be cost-matched using the measured `field_cp` FLOP count.
- The proposal only survives if it either improves quality at matched cost or reaches quality
  neutrality with a measured speedup. Parameter count alone is not a result.

## Pre-registered funnel

1. Phrase-HMM, `T=4`, seed 0, 8k steps. Use `C=16`, field rank 8, and the same trunk/head width
   as the prior S03 gate.
2. Advance only if exact block KL is at most 0.10 nats/block and impossible generations are at
   most 3%. These are the already-registered S03 thresholds, not post-hoc thresholds.
3. If it passes, confirm seed 1 and run a strict `T=L=2048` depth-8 experiment at the dense
   training-FLOP budget. Reuse `B1_dense_s1`; never retrain it.
4. At depth 8 require: trunk BPB no more than 1% above dense, generated reference PPL no more
   than 10% above the arm's AR generation, distinct-3 without collapse, and at least 2x batch-16
   decode speed under a symmetric benchmark.

## Kill interpretation

- Failure on the phrase-HMM kills this exact positional-field mixture, not all one-shot joints.
- Passing the toy gate but failing `T=L` indicates that a compact global plan does not have enough
  conditional capacity for 2,048-token text, even when its positional action and likelihood are
  repaired. Do not respond by sweeping code count or field rank; the next mechanism must change
  the joint factorisation.

## Result — 2026-10-03

The pre-registered seed-0 run completed at 8,000 steps under Modal profile `nanochat2`:

| metric | result | gate |
|---|---:|---:|
| exact block KL | 1.7078 nats/block | <= 0.10 |
| impossible generations | 33.94% | <= 3% |
| latent sensitivity | 3.2596 nats/block | diagnostic |
| trunk NTP excess | 0.001 nats/token | diagnostic |

**KILL.** This is not latent collapse: the sampled plan substantially changes the output
distributions and recovers much of the 4.270-nat total correlation, but a 16-way global plan still
cannot select a coherent phrase branch reliably. Per the pre-registration, no component-count,
rank, second-seed, or depth-8 sweep is authorised. The compiled log is `s04_sap_compiled.log`.
