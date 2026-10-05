# SAP Flow–Joint: completed toy gate, 2026-10-03

**Historical gate verdict: FAIL at the registered budget.** Its original
no-promotion decision was explicitly overridden by the user on 2026-10-04 for
one full-budget d4 **flow-only** real-text run at T=L=2048; see
`sap_flow_text_d4_plan.md`. The toy numbers and hybrid verdict remain unchanged.
This closes the tested hybrid instantiation at that budget, not every
possible continuous/discrete hybrid or the original SAP research objective.

Registration: `sap_flow_joint_gate.md`. Full local record:
`sap_flow_joint_compiled.log`; machine-readable record:
`out/sap_flow_joint/fj_20261003_b/summary.json`.

## Results

Six L4 runs on explicit profile `nanochat2`, two seeds per arm, T=4, V=512,
oracle context, width 128. Hybrid/flow-only have four generative mixing blocks
(two affine-flow conditioners plus a two-block decoder), not a real-text d4 LM.
Both also have a two-block training-only recognition network. Hybrid/chain use
64 learned discrete states. No teacher, distillation, verifier, or token-conditioned
neural generation loop was used. The existing dense AR model was not retrained.

| Arm | Updates per seed | KL, seed 0 / 1 (nats/block) | Invalid prior blocks, seed 0 / 1 |
|---|---:|---:|---:|
| Hybrid | 8,000 | 1.4004 / 1.4415 (bound) | 35.25% / 35.75% |
| Flow-only | 10,576 | 1.2447 / 1.2364 (bound) | 31.58% / 30.63% |
| Chain-only | 15,572 | 4.0022 / 3.9749 (exact model likelihood) | 57.07% / 58.80% |
| Registered gate | — | <=0.10 | <=3.00% |

Controls were assigned updates by measured forward+backward matrix FLOPs,
including attention and recognition. Total-budget ratios to hybrid: flow-only
0.999974, chain-only 1.000001. Pointwise, log-space reductions, and optimizer
work are outside this proxy. It is not an equal-wall-time comparison.

Each final measurement uses the same 1,024 held-out contexts and target blocks,
with eight prior samples/context (8,192 generated blocks/run). Conditional HMM
likelihood is exact; latent-marginal likelihood uses IWAE K=256. These are
Monte Carlo estimates of an upper bound on KL **in expectation**, not exact
latent-marginal KL or guaranteed pointwise bounds. Chain-only KL is still an
empirical expectation over held-out data, not an enumerated population KL.

Hybrid invalid-rate approximate 95% intervals, using context-cluster standard
errors: 35.25 +/- 1.47 and 35.75 +/- 1.55 percentage points. The failure is far
outside sampling uncertainty. Reported bound estimates are also worse than
flow-only on both seeds; unequal bound tightness prevents treating their
difference as a proven true-KL ranking. The invalidity result already fails
the gate independently of that qualification.

## What the diagnostics establish

- Increasing importance samples from 64 to 256 changes hybrid KL estimates
  from 1.4156 to 1.4004 and 1.4532 to 1.4415. Mean effective sample sizes at
  K=256 are 146.9 and 142.2. This is a useful stability check, not proof of
  bound tightness. A likelihood-estimator issue cannot explain 35% invalid samples.
- Posterior-conditioned hybrid samples are still 28.56% / 28.61% invalid,
  versus 35.25% / 35.75% from the prior. Thus this is **not merely** a good
  posterior decoder with a bad generation prior. The posterior/decoder pair
  also fails the local-compatibility test. This does not separately identify
  decoder expressivity, inference-family limitations, and optimization.
- Estimated posterior/prior latent KL is 2.598 / 2.691 nats/block. Fixing the
  latent to zero changes invalidity to 15.36% / 14.71%, still a failure. This
  diagnostic is not a calibrated alternative sampler: removing latent randomness
  changes its distribution and might reduce diversity.
- The hybrid improves greatly over chain-only, but the local discrete joint adds
  cost without a demonstrated benefit over spending the same matrix-FLOP budget
  on flow-only. The intended global/local complementarity is not established.
- Interim evaluations use the first 256 contexts; final evaluations use all 1,024.
  Do not interpret a difference between interim and final means as a pure learning
  improvement on an unchanged evaluation set. The two interim points are comparable.

No L=2048 decode throughput, real-text BPB, or d8 quality claim follows from this
toy test. The user-requested full-training follow-up is an explicit scale/learning
test of the strongest control, not a claim of a new mechanism or a toy-gate pass.

## Engineering and provenance

Implementation: `nanochat/flow_joint.py`; experiment:
`scripts/sap_flow_joint_gate.py`; isolated launcher: `modal_sap_flow_joint.py`.
Sixteen focused local tests pass, including normalization, empirical sampler
agreement, flow inverse/Jacobian, deterministic noise-tape replay, finite gradients,
T=1 behavior, a known exact special case of importance weights, and partial-log
collection. A two-update CPU smoke run also completed. No local full training.

Remote app: `ap-O7cF3FljNeQB4jmGIu2fBm` (completed). Volume `nanochat` under
`out/sap_flow_joint/fj_20261003_b/` retains each final checkpoint, optimizer/RNG/data
pool state, manifest, JSONL learning curve, and stdout log. Local summary includes
source hashes/configs; `calls.json` records all six function-call IDs. Only logs
and summaries were downloaded, not large checkpoints.

The first submission (`fj_20261003_a`) failed on a launcher import before training;
its empty app was stopped. The successful submission used the same registered
model/configuration after packaging was fixed. Post-launch launcher hardening adds
a durable remote sweep parent for future client disconnects. Post-launch runner
edits change only partial-summary formatting (reporting completed versus planned
updates and the available importance count), not model/training/evaluation behavior.
The compiled log retains the original successful-run formatting and source hashes.

Inference retains dense transition fields O(B T S²), lexical logits O(B T V),
and attention work O(B T² d). State resolution uses O(T S log T) work and
O(log T) span; its current implementation is not a work-efficient scan kernel.
Training adds O(B T V S) emission contractions and sequential-in-T exact HMM
likelihood. Neither low sequential neural depth nor FLOP counting proves a
wall-clock speedup at L=2048.
