# SAP Flow–Joint gate (registered 2026-10-03, before training)

**Post-run status:** all six registered runs completed; hybrid fails on both seeds.
Results in `sap_flow_joint_results.md`; registration below is retained unchanged.

User-authorized implementation of Direction 1 in `sap_next_mechanisms_review.md`.
No teacher, distillation, AR verifier, rejection, or token-conditioned neural loop.
This is a constrained learned model, not a free fit to the generator's parameters.

## Frozen first test

- Phrase HMM: the existing V=512 generator, generator seed 0, context 64, T=4.
  Exact filtering belief is supplied as **oracle context only**. The learned model
  never receives generator transitions, emissions, phrase identities, or state labels.
- Three arms, seeds 0 and 1: `hybrid`, `flow_only`, `chain_only`.
- Width 128; decoder two bidirectional Transformer blocks. Flow arms additionally
  have two affine channel-coupling layers, each with one bidirectional Transformer
  conditioner, and a full-length latent with 8 channels per slot. Thus inference
  has four neural mixing blocks, not four plus an undisclosed deep flow. A separate
  two-block Gaussian recognition network is used **only in training/evaluation**.
- Hybrid/chain: 64 learned states and learned positive lexical weights R[S,V].
  Emissions are exactly normalized q_t(y)R[s,y]/sum_v q_t(v)R[s,v]. No fixed code,
  hard assignment, HMM-informed initialization, or external language model.
- AdamW, lr 3e-4, batch 128; hybrid budget 8,000 updates. Other arms get a derived
  number of updates matching hybrid's measured forward+backward matrix FLOPs.
  PyTorch's FLOP counter includes attention; pointwise/reduction/optimizer work
  is not represented by that proxy and wall time is reported separately. All
  recognition and density-evaluation matmuls are charged. No hand-matched widths.
- Train ELBO (one posterior draw). KL weight warms linearly to 1 over the first
  10% of updates; the remaining 90% uses the proper ELBO. Report the weight.
  Common cosine learning-rate schedule by fraction of each arm's budget.
- Independent streaming training data; common held-out contexts/targets and
  evaluation tapes across arms/seeds. Evaluate at 0, 25%, 50%, 100% of budget.
  Final evaluation: 1,024 held-out contexts, 8 prior samples/context; IWAE K=64
  and K=256 with per-context uncertainty and importance effective sample size.
  Final checkpoint, optimizer, RNG, configs, per-stage metrics and logs retained.

## Gate / interpretation

Require all of: estimated held-out KL upper bound <=0.10 nats/block, invalid prior
blocks <=3%, and >=10% better comparable held-out KL than each ablation, on both
seeds. Sampling uncertainty is reported, not silently rounded into a pass.
An IWAE estimate is a lower bound **in expectation**; its single Monte Carlo
realization is not a guaranteed pointwise bound. K convergence and ESS are checked.
Bound differences alone cannot establish true-KL superiority if bound tightness
differs: an apparent pass with unresolved bounds is inconclusive, not promotion.
High prior invalidity directly fails the gate, independently of bound tightness.
Record posterior-conditioned invalidity and latent KL to distinguish inadequate
local modeling from a prior/posterior gap or ignored latent.

A pass earns a learned-context and T=16/64 coherence test, then full L=2048
throughput/memory measurement. It does not earn a d8 real-text run automatically.
Failure closes this instantiation, not all continuous/discrete hybrids. No dense
baseline is retrained. This toy cannot establish BPB neutrality or a paper claim.
The eventual bar remains <=1.01x dense joint BPB and >=2x batch-16 CUDA-graph
throughput for 2,048 actually emitted tokens at matched training compute.

## Execution and costs

Use explicitly selected Modal profile `nanochat2`, not the machine's active profile.
Six isolated L4 tasks, maximum one hour each, committed checkpoints/JSONL at each
milestone. No automatic widening, extra seeds, or longer runs after inspecting results.
Save a local compiled log and JSON summary after collection.

One primitive noise tape feeds the fixed flow and decoder. Counterfactual transition
maps are sampled in parallel; a deterministic Hillis–Steele composition scan resolves
states in O(log T) span and O(T S log T) work. Only the selected state's emission is
normalized at inference. All tokens then emit concurrently, with no resampling.
Training normalizers cost O(B T V S); transition projection O(B T d S²);
log-space HMM likelihood O(B T S²) work and O(T) sequential training span.
No B*T*S*V emission tensor is materialized. Dense logits and transitions still cost
O(B T V + B T S²) memory, and attention costs O(B T² d) work. At L=2048 these
remain serious bottlenecks requiring measurement, not a presumed speedup.

Nearest work and mechanism review: FlowSeq (D19-1437), Categorical Normalizing Flows
(2006.09790), CoDD (2603.00045); downloaded papers in `Literature Review/` and the
comparison in `sap_next_mechanisms_review.md`. The hybrid combination alone is not
a novelty claim.
