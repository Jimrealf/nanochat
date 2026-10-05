# Flow-only full-training override: d4, T=L=2048

**Completed:** all 1,183 updates and final evaluation/benchmark. See
`sap_flow_text_d4_results.md` and `sap_flow_text_d4_compiled.log`. Batch-16 graph
speedup is 66.33x, but joint BPB-bound estimate 2.34450 versus dense 1.14462 is
not quality-neutral. Registration below is retained for provenance.

2026-10-04. Explicitly requested by the user after the Flow–Joint toy screen.
This supersedes the earlier instruction not to advance after its T=4 failure.
The old numbers remain valid; their use as a universal scaling veto does not.
T=4 isolated local dependence cheaply. It did not test long-block representation,
real-text optimization, or the final T=L system. The 3% invalidity criterion belongs
to that synthetic distribution, not to natural-language quality in general.

## Scope and accounting (before full training)

- One flow-only model, seed 1, from scratch on the existing FineWeb-Edu shards.
  T=L=2048 **from the first training update**; fixed observed prompt of 128 tokens.
  No T=4 training, short-block curriculum, distillation, verifier, AR training mode,
  token-dependent neural loop, or retraining of a dense model.
- Width 256. Four generative Transformer blocks total: two affine coupling
  conditioners (one block each), then a two-block lexical decoder. Full-length
  eight-channel continuous latent. Two extra recognition blocks are training-only.
  All three paths attend to embedded **observed prompt tokens**, replacing the
  toy's exact HMM context belief. No uncounted context encoder is added.
- Prompt and future slots are concatenated in each bidirectional mixer. Future
  decoder inputs are continuous latents, not target token embeddings. Only the
  recognition network receives future training tokens. Train/eval input leakage
  and single-tape generation receive explicit tests.
- Same V=32768 tokenizer as the existing `S07_dense_L_s1` checkpoint. Retrieve it
  read-only from `jimpearse01`; all new paid training/evaluation uses `nanochat2`.
  The local baseline record `out/s07_lanes/summary_lanes_d4.json` reports
  121,110,528 training tokens, 94,372,610 FLOPs/token, conventional val BPB 1.151842.
  Derive the total budget by multiplying those recorded values (~1.143e16 FLOPs).
  Charge the flow model's measured forward/backward matrix FLOPs including
  recognition and vocabulary-checkpoint recomputation. Report that this is a
  matrix-FLOP proxy versus the baseline's recorded estimator, not identical
  optimizers or perfectly equal hardware cost.
- AdamW, lr 3e-4, batch 65,536 generated target tokens/update, microbatch 4;
  cosine decay to 0.1x, 2% LR warmup, KL weight warmup to 1 over 10% of budget.
  Complete the derived budget irrespective of toy-style quality thresholds.
  Abort only on engineering faults (nonfinite loss/gradients, data/tokenizer
  mismatch, resource failure); checkpoint first where possible.

## Evaluation

**Launched:** `flow_d4_tl_s1_20261004`, Modal app
`ap-FFS3HLbiUJ9tZ6CNmN3QpD`, call `fc-01M42JJ7CNGWRZJXZ6PD62KDSJ`.
The on-device FLOP audit derives 1,183 updates and 77,529,088 target tokens,
11,425,778,017,239,040 matrix FLOPs versus the baseline's recorded
11,429,516,625,838,080 FLOPs (0.0327% below). Parameters: 23,856,672.
Both volume listings have 301 shards and validation shard `shard_06542.parquet`;
tokenizer SHA256 matches `06978be3b6fa73254b3adc4a2f7b499bc7f9d7eb9d74f37b7a1620bd84e02808`.
The volume run directory contains an exact source snapshot as well as hashes.

- The same fixed held-out 128 prompt/continuation rows at every checkpoint;
  intermediate ELBO, final importance counts K=1,8,32, ESS and uncertainty.
- Compare flow block **joint BPB upper-bound estimates** to dense joint BPB on
  exactly the same continuation tokens. Include special-token likelihood in both
  joint numerators; bytes count actual text bytes. This differs from the historical
  conventional BPB that masks special-token losses. Re-evaluate dense; never compare
  these new joint figures directly to its old 1.151842 number.
- Importance estimates are bounds in expectation, not guaranteed per-realization
  bounds. ESS collapse at L=2048 is reported as an evaluation limitation. A loose
  bound cannot prove the actual inference distribution has that exact BPB gap.
- Save prior-generated 2048-token samples, repetition/diversity and dense-reference
  scoring as descriptive diagnostics. Dense is evaluation-only, not a teacher or
  generation verifier. Natural text has no oracle impossible-sequence rate.
- Time actual 2048-token generation at batch 1 and 16 on the same H100, including
  prompt processing, latent draw/flow, vocabulary projection and token sampling.
  Report eager and CUDA-graph measurements separately. Graph capture/warmup is
  excluded for both; no kernel-only or padded-token speedup claim. Preserve timings,
  memory, checkpoints, curves, samples and a downloaded compiled log.
- Long-term success remains <=1.01x dense joint BPB and >=2x graph throughput,
  with plausible samples/diversity. This exploratory full-budget run is not stopped
  early merely because it misses that final research bar. It also does not prove
  optimization convergence, robustness across seeds, or a main-track novelty claim.

## Implementation and resource safeguards

Keep this experiment isolated from the existing dirty GPT/base_train branches.
Projection chunks bound O(B*T*V) temporary storage, with checkpoint recomputation
charged in training FLOPs. Hidden activations cost O(B*T*d); full attention costs
O(B*(T+128)^2*d) work (use SDPA's fused backend). Chunking is a memory strategy,
not token-conditioned decoding. Data uses existing remote shards, not a dataset
copy on the crowded local drive. One H100 run, maximum two hours, with incremental
volume commits and retrieval independent of training. CPU runs are tiny smoke tests.

This is a real-text adaptation of the previous `flow_only` control, not the
failed sign-cell categorical flow, and not a new hybrid mechanism. Nearest work
remains FlowSeq (`Literature Review/D19-1437_FlowSeq_Ma_2019.pdf`); the experiment
tests feasibility at the SAP operating point, not novelty by combination.
