# Flow-only d4 at T=L=2048 — completed full-budget test

2026-10-04. Requested explicitly after the user challenged the T=4 screening veto.
**Outcome: the full-length mechanism is fast, but not quality-neutral. The learned
latent is barely informative under the recognition distribution.** This is a
result for this architecture/training recipe and budget, not a proof against scaling.

## What actually ran

- One H100 on Modal profile `nanochat2`; seed 1, FineWeb-Edu, pinned V=32,768.
- T=L=2048 throughout training and inference; 128 observed prompt tokens.
- Width 256; four generative attention blocks (two flow conditioners + two decoder
  blocks), two training-only recognition blocks; 23,856,672 total parameters.
- 1,183 updates, 65,536 targets/update, **77,529,088 target tokens**.
- Measured flow matrix FLOPs 1.1425778017e16 versus existing dense recorded budget
  1.1429516626e16 (0.0327% lower). Recognition and vocabulary projection checkpoint
  recomputation are charged. The two FLOP estimators and optimizers are not identical;
  this is an approximately budget-matched architecture test, not an optimizer ablation.
- Complete scheduled budget, no early quality stopping, no toy invalidity gate,
  no distillation/AR verifier, no second LM trained. Dense `S07_dense_L_s1` was reused.

## Quality: same held-out continuations, fresh dense evaluation

128 held-out blocks, 262,144 target tokens, 1,239,960 text bytes. Both joint metrics
include special-token likelihood; they are not directly comparable to older
conventional BPB that masks special-token losses.

| Measurement | Joint BPB | Relative to dense |
|---|---:|---:|
| Existing dense d4, exact AR likelihood | 1.144624 | 1.0000x |
| Flow, K=1 ELBO estimate | 2.344598 | 2.0484x |
| Flow, K=8 importance bound estimate | 2.344510 | 2.0483x |
| Flow, K=32 importance bound estimate | 2.344498 | 2.0483x |

The flow numbers estimate an NLL upper bound **in expectation**, not exact inference
BPB or a guaranteed pointwise bound. Thus +104.83% is the **bound gap**, not a proven
exact marginal-likelihood gap. K=32 mean ESS is 14.12/32 (minimum 2.95); changing
K=1 to 32 changes BPB by only 0.00010. That stability is useful but not a proof of
tightness. The bound's across-row standard error is 0.01736 BPB.

Prior samples are visibly incoherent word/subword mixtures. Their dense-reference
joint BPB is 2.9419 over 16 samples; this is a descriptive reference score, not the
flow's own likelihood and not a dense-generated-sample head-to-head comparison.
Distinct trigrams are 99.985%: diversity alone does not establish language quality.
Full unedited 2048-token samples are saved in the local run directory.

## Speed: whole generation, not a kernel or padded-token estimate

Same H100, same 128-token prompts, temperature 1, **2,048 actual generated tokens
per row**, including prompt processing, latent/noise generation, vocabulary
projection and sampling. Three timing trials, median; capture and warmup excluded
for both. Dense uses KV caching. Generation outputs are materialized for both.

| Batch | Flow graph latency | Dense graph latency | Flow / dense graph throughput | Speedup |
|---|---:|---:|---:|---:|
| 1 | 3.112 ms | 1,784.250 ms | 658,150 / 1,148 tokens/s | 573.39x |
| 16 | 31.314 ms | 2,077.138 ms | 1,046,422 / 15,776 tokens/s | 66.33x |

Eager speedups: 1186.74x at batch 1, 160.76x at batch 16. These larger eager
ratios are strongly affected by launch overhead and are not the headline comparison.
Peak PyTorch allocated memory during the benchmark with **both models resident**:
0.420 GB (batch 1), 1.751 GB (batch 16). This is not isolated flow training memory
or total GPU reservation. No claim of a deployable efficient LM follows from speed
without acceptable quality.

## What full training taught us

| Updates | Held-out ELBO BPB | Posterior/prior latent KL, nats/token |
|---|---:|---:|
| 0 | 3.22195 | 0 |
| 295 | 2.34909 | 0.01530 |
| 591 | 2.34707 | 0.00184 |
| 887 | 2.34565 | 0.00109 |
| 1,183 | 2.34458 | 0.00086 |

All checkpoints use the same evaluation rows. Most measured improvement happens
in the first quarter, then the likelihood curve flattens while latent KL approaches
zero. This is strong evidence consistent with **posterior collapse / underuse of
the sampled plan** in this training recipe. It is not merely an artifact of requiring
3% impossible sequences on a tiny toy. It also does not separately establish whether
the limiting cause is the objective, optimizer, recognition family, or capacity.

The useful next problem is inducing meaningful latent information and dependence
under the prior while preserving this fixed-computation generator. Blindly repeating
the same full run, claiming more training cannot help, or relabeling this speedup as
quality-neutral would all overstate the evidence. This exploratory run is one seed;
it completes the requested baseline-sized budget, not a proof of global convergence.

## Artifacts and engineering

- Local compiled log: `sap_flow_text_d4_compiled.log`.
- Local structured results and samples: `out/sap_flow_text/flow_d4_tl_s1_20261004/`.
- Modal volume `nanochat`: `out/sap_flow_text/flow_d4_tl_s1_20261004/`, containing
  final checkpoint/optimizer/RNG, source snapshot, held-out rows, manifest, curves,
  evaluation JSON, samples, logs and completion marker.
- Training app `ap-FFS3HLbiUJ9tZ6CNmN3QpD`; final evaluation app
  `ap-eodS6yLxSkk2goGCTK7J8d`. Training finished before evaluation was retried.
- The first dense evaluation failed because GPT's flattened target `.view()` needs
  contiguous storage. Fixed the caller's input/target slices, added a regression
  test, then evaluated the existing checkpoint. Neither model was retrained.
- Twenty-five focused tests pass (text model + prior Flow–Joint tests), including
  bijection/Jacobian, chunked likelihood gradients, exactly four generative blocks,
  no recognition/no fresh RNG when a full tape is supplied, boundary alignment,
  budget derivation, and dense comparison accounting. CPU work was smoke-only.

Plan: `sap_flow_text_d4_plan.md`; implementation `nanochat/flow_text.py`, runner
`scripts/sap_flow_text.py`, isolated launcher `modal_sap_flow_text.py`.
