# Flow-only d8, T=L=2048 — full-budget result

Completed 2026-10-04 on Modal profile `nanochat2`. All training, likelihood
evaluation, sample generation and both decode benchmarks completed. The remote
`COMPLETE` marker is present and the full log is downloaded.

**Verdict:** scaling this instantiation improves the BPB-bound estimate by 1.10%
relative to d4, but does not restore coherent generation or establish quality
neutrality. End-to-end CUDA-graph speed is excellent: 877.2x at batch 1 and 89.3x
at batch 16. Speed alone does not meet the research bar.

## What was trained

- Eight generative Transformer blocks, width 512, 16 attention heads: two affine
  coupling layers with two-block conditioners, then a four-block lexical decoder.
  Four additional recognition blocks are training-only and charged to compute.
- Full-length eight-channel latent, prompt 128, T=L=2048 from update one, V=32768.
  Generation uses one upfront random tape and fixed neural computation. No teacher,
  distillation, AR verifier, or token-dependent resampling loop.
- 76,163,104 total parameters including recognition. 4,114 updates at 65,536 target
  tokens/update = **269,615,104 target tokens**. Final training evaluation elapsed
  1,574.29 seconds (~26.2 minutes), excluding setup and final postprocessing.
- Counted training matrix FLOPs: 155,151,422,223,024,128. Existing dense recorded
  budget: 440,401,920 tokens * 352,324,600 FLOPs/token = 155,164,430,303,232,000.
  Difference: -0.00838%. Recognition and vocabulary-projection recomputation count.
  This is a matrix-FLOP proxy match, not identical optimizer or wall-clock cost.
- Same optimizer/objective as d4: AdamW, lr 3e-4, KL warmup over the first 10%,
  beta=1 thereafter. Complete registered budget; no synthetic invalidity early stop.
- Dense reference reused, never retrained: `S08_dense_L_s1`, step 1680, width 512,
  depth 8, full attention. Metadata step, token budget, tag and validation score
  checked before training. Checkpoint SHA256:
  `3288b1328b14ee3ea106732f44b9bc92c1f0f4ed64fb58969880a13988586c38`.

Depth, width and training budget grow together versus d4; this is not a pure depth
ablation. The d8 compute budget is 13.58x the d4 budget.

## Likelihood and quality

128 held-out rows, 262,144 continuation tokens, 1,239,960 text bytes. Downloaded
d4/d8 held-out tensors and verified exact prompt/target equality. Dense and flow
score the same continuation tokens; both joint numerators include special-token
losses. The pinned tokenizer and data manifest are recorded in the result.

| Measurement | BPB | Relative to d8 dense joint BPB |
|---|---:|---:|
| Dense d8 joint likelihood | 0.951791 | 1.0000x |
| Flow K=1 ELBO estimate | 2.320787 | 2.4383x |
| Flow K=8 importance-bound estimate | 2.319134 | 2.4366x |
| Flow K=32 importance-bound estimate | 2.318681 | 2.4361x |

Dense conventional masked BPB on these rows is 0.951220; the older 0.958955 value
used another evaluation scope and is not the comparison denominator here.

The K=32 **bound-estimate gap** is +143.61%, not a measured exact marginal/inference
BPB gap. These importance objectives bound NLL in expectation, not pointwise for
each finite Monte Carlo realization. ESS is poor: mean 2.17/32, minimum ~1.00.
Increasing K=8 to 32 changes the estimate by 0.000453 BPB, but does not prove the
bound is tight. Row-level standard error is 0.01693 under the evaluator's row
independence approximation; it does not account for variational-bound looseness.

The d4 K=32 estimate was 2.344498: d8 is 1.10% lower on identical held-out rows.
The respective dense baselines improve from 1.144624 to 0.951791, so the relative
bound gap grows despite the small absolute flow improvement.

All 16 saved prior samples contain exactly 2,048 output tokens. Inspection of
their openings shows incoherent word/subword mixtures, not ordinary prose. For
example, a prompt about audio editing produces:

> , been this's happens if myself aarn:

> time moreaxis $ in the. However like probiotics claim proper. oil details tiny.

Dense-reference sample BPB is 2.626882 and distinct trigrams 0.999695. These are
descriptive only: the former is not the flow's likelihood or a matched comparison
against dense-generated samples; the latter plainly does not establish coherence.

## Does scale fix latent underuse?

| Budget | Step | Held-out K=1 BPB estimate | Posterior/prior KL, nats/token |
|---|---:|---:|---:|
| Initial | 0 | 3.226414 | 0 |
| 25% | 1,028 | 2.335378 | 0.017393 |
| 50% | 2,057 | 2.329608 | 0.005876 |
| 75% | 3,085 | 2.324088 | 0.004804 |
| 100% | 4,114 | 2.321011 | 0.004475 |

The final K=32 diagnostic gives KL 0.004927 nats/token, about 5.73x d4's 0.000860.
Thus latent use did increase with scale; it would be inaccurate to call it exactly
unchanged or zero. Nevertheless it remains very small and decreases after warmup,
consistent with substantial latent underuse/near-collapse. The bound continues
improving slowly; completing the budget does not establish optimization convergence.
Final evaluation K=1 differs slightly from the last training evaluation because
the evaluation microbatch and finite latent samples differ, not because of extra
training. This run does not isolate the objective, posterior family and optimizer
as competing causes, nor rule out every larger model or alternative flow mechanism.

## Actual generation speed

One H100 80GB, torch 2.9.1+cu128. Exactly 2,048 outputs per row, prompt length 128,
temperature-one sampling. Prompt processing, latent/noise generation, all flow
and decoder computation, vocabulary projection and token sampling are included.
Three timed trials after warmup; medians below. Graph capture/warmup excluded for
both. The reference is ordinary dense AR decoding, not speculative decoding.

| Batch | Flow graph latency | Dense graph latency | Flow / dense tokens/s | Speedup |
|---|---:|---:|---:|---:|
| 1 | 4.156 ms | 3,645.881 ms | 492,750 / 562 | 877.20x |
| 16 | 45.934 ms | 4,099.804 ms | 713,374 / 7,993 | 89.25x |

Eager speedups are 2,159.21x and 301.05x, respectively; use the less inflated graph
figures. First eager-flow trials had large timing outliers; raw trials are retained.
Allocator peaks with **both models resident**: 1.961 GB at batch 1, 3.712 GB at
batch 16. These are neither isolated flow memory nor training peaks. Vocabulary
projection is chunked at 128 positions to bound temporary logits; global attention
still has quadratic-in-length work. This run does not claim linear-time attention.

## Reproducibility and limits

- Registration: `sap_flow_text_d8_plan.md`.
- Local complete log: `sap_flow_text_d8_compiled.log` (71,401 bytes at collection).
- Structured results, samples and held-out tensors:
  `out/sap_flow_text/flow_d8_tl_s1_20261004/`.
- Modal volume `nanochat`, profile `nanochat2`, same relative run directory holds
  checkpoint/optimizer/RNG state, source snapshot, manifest, curves, samples and logs.
- Training app `ap-b9wNkZXJodxJF7Y6Z48nvX`, call `fc-01M43368V65YZ2Y7BMS8CMS79D`.
  The local waiting client timed out; the detached remote job continued and completed.
  Independent CPU collection recovered everything. No training was restarted.
- Thirty focused tests pass, including d4 checkpoint-layout compatibility, d4/d8
  affine inverse/Jacobian, target-free fixed-tape sampling with the exact generative
  block count, finite training gradients, FLOP accounting, and baseline provenance.
- One seed, one registered budget, finite-sample likelihood bounds, no claim of
  convergence or main-track novelty. No extra scaling sweep is implied. The
  >=2x speed requirement passes; the <=1.01x dense quality target is not established,
  and inspected sample coherence fails. This instantiation is not the SAP solution.
