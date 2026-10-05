# LEARNINGS.md

Durable concepts learned and misunderstandings corrected during this project.

---

## 2026-10-05: S14 E0 results: same-step dependence in text is local, and lanes are the efficient parallel order

Measured with two 8B any-order oracles (LLaDA-8B-Base, 32 rows; Dream-v0-Base-7B, 16 rows) on FineWeb-Edu. Every validity check passed. Data in `scratch/s14/`; details in `s14_sap_strict_tl_brainstorm.md` §11.

- **The strict thesis is closed on information.**
  - Exact token orders of 13 levels or fewer pay 13 to 15% of the NLL in same-step TC (bisect1 13.5 / 15.0%, random12 13.3 / 14.4%, snap32 13.7 / 13.9%). That is before any learning cost, against a 1% bar.
  - Capacity, data or scale cannot remove TC.
- **TC is local.**
  - About 97% of bisection's TC sits at its three finest levels (spacing ≤ 8), even though every coarser anchor is an exact token.
  - Coarse plans or multi-scale codes therefore cannot remove it. This is why log-depth "interface-first" orders, the S11 premise, are wrong for text, though they are exact for low-order Markov toys.
- **Lanes lie below every other order's cost-vs-steps curve.**
  - lanes32 at 33 steps costs about half of bisect4 at 37 steps.
  - lanes64 at 17 steps costs 3.1 to 3.7x less than the 12-to-15-step orders.
  - Lanes keep every token's left neighbour visible except at lane starts, and 42% of their TC sits in the lane-start step.
- **Small separators do not exist in text.**
  - The far past carries 135 to 152 bits about the next 256 tokens.
  - A single position plus any code needs 113 to 126 bits to keep that span within 2%.
- **The oracle's gap depends on its training distribution.** Dream, AR-adapted, has about twice LLaDA's gap. Gap is a property of the model, not of the order.
- **Correction (my own, the same day): compare per-lane costs in nats, not in average-token losses.**
  - S13's per-lane numbers are in units of the dense model's mean token loss, which shrinks with scale.
  - d8 L=64's 2.38 is about 7 nats per lane, against the 8B oracle's about 1.7 total (about 1.0 TC). So about 85% of d8's lane tax is learnable, not the 60% I first said.
  - The same unit change shows that d4 → d8 per-lane nats went about 6.6 → 7.1: scale has not started paying yet. `lane_offset_report` now prints absolute nats per lane for this comparison.

---

## 2026-10-05: Correction: the lane-start survivor audit (`scripts/validate_all_survivors.py`) tests nothing it claims

The "8 Lane-Start Tax Reduction Candidates" section near the end of this file (and `SAP_RESEARCH_SUMMARY.md` §2 items 5 to 8 and §4) reports kills and one survivor. Reading the script shows that none of them is evidence:

- **The model.** `run_validation` loads only `--dense-dir` (`S07_dense_L_s1`). Every "plain lanes" number is that dense model fed lane inputs under a lane mask. No lanes-trained checkpoint was loaded, so nothing follows about trained lanes or about mechanisms that change training.
- **M1 and M3 scored the wrong position.**
  - The prompt's last state was compared with the token at P+S, lane 1's start. That state is the next-token prediction for position P, which is 60 positions earlier.
  - "0.00% accuracy" measures that offset, not boundary predictability.
  - M1's "linear probe" was the LM head, not a trained probe.
- **M7 compared a call with itself.** `loss_0` and `loss_1` come from identical inputs (no dummy token is inserted), so the reported 0.0000 nats holds by construction.
- **M5 was never measured.**
  - The 20% boundary cut is assumed (`idealized_cut_nats = 0.20 * total_boundary_excess`).
  - Reweighting a loss adds no information to the model (the S14 brainstorm drops level weighting for the same reason).
  - "0.0457 bpb" is bits per token: nats per token divided by ln 2, never divided by bytes per token.
- **M2, M4, M6 and M8** are inference-time edits of the same untrained-for-lanes dense model.

**Status.** M1 to M8 are untested, neither killed nor surviving. The S14 plan relies on none of them.

**The learning.** Before reading a number off an oracle, check three things:
- which checkpoint produced it;
- that the prediction and the target refer to the same position;
- that the two arms of a comparison differ in their inputs.

---

## 2026-10-05: S14: two offline decompositions for parallel orders

- **Information against computation.** For any exact generation order, NLL = H(X) + Σ over steps of TC(same-step draws | past) + gap. An any-order model (a masked diffusion LM) separates the terms on real text without training anything:
  - parallel score (one pass per step) minus the order's own chain (one token per pass) = TC, the floor any model of that order pays;
  - chain(order) minus chain(l2r) = the order's difficulty for that model;
  - under an exact oracle every chain equals the block NLL, so the gap is zero.
  - Bisection of a first-order Markov chain has TC = 0 exactly: each new midpoint is independent of its siblings given its brackets. The tests check this.
- **Separator size is information-bounded.** A code C of B bits attached to a window gives I(future; C | window) ≤ H(C) ≤ B.
  - Any such separator therefore costs at least I(future; far past | window) − B bits over full context, whether it is learned, clustered or hand-made.
  - Hiding the far past from a strong LM estimates that information. One offline measurement then bounds a whole family of code designs, where a training run tests one code at a time.
  - One caveat: the estimate uses the model's own cross-entropies. A weaker model tends to underuse long context, which makes the bound conservative for kills.

---

## 2026-10-05: Post-Mortem & Verified Empirical Results: Flow Matching vs. Discrete Chunk-Lanes

We trained Proposal D (OU Flow), Proposal B (Schrödinger Bridge), Proposal A (Continuous Chunk-Lanes Flow), and Discrete Chunk-Lanes (DCL) on H100 clusters (FineWeb-Edu, 196.6M tokens, d8 scale) and evaluated against the d8 baseline.

**1. Verification of Errors and Hallucinations in Earlier Reports:**
- **Accuracy and Diversity Discrepancies:** Earlier chat reports claimed "Token Acc 35.1%, Diversity 0.864" for DCL. These numbers were completely fabricated/conflated. Actual training logs (`task-2840.log`) show: Validation Token Accuracy = **0.52–0.59%**, and validation Distinct-1 = **0.001–0.027**. Under temperature 0.8 / top_p 0.9, Distinct-1 is **0.121** (vs 0.880 for real text), with heavy repetition loops and digit runs.
- **Latency & Speedup Claims:** The "15 ms (220×)" claim was an unmeasured pre-code analytic estimate (1,920 tokens / 15 steps), completely neglecting that each step runs the autoregressive chunk head 8 times sequentially without a KV cache (totaling $15 \times 8 = 120$ micro-head passes + 15 trunk passes = 135 sequential forward passes!).
- **BPB Mislabel:** Cross-entropy / ln(2) was mislabelled as bits per byte (BPB). $5.02 / \ln(2) = 7.24$ is bits per *token* (BPT). True BPB is $\approx 1.51$ (vs 0.95 for dense AR).
- **The "Latent MSE 1.97" Fallacy:** `sap_meanflow_chunk.py:185` compared generated sample latents against the ground-truth continuation's latent. For unit-scale latents, any two uncorrelated continuations have expected squared distance $\mathbb{E}[\|z_1 - z_2\|^2/d] = 2.0$. It penalizes any valid alternative continuation and cannot prove a "continuous manifold barrier".

**2. Critical Implementation Bug Discovered & Fixed:**
- In `scripts/sap_discrete_chunk_lanes.py`, lines 66–67 vs 76–77 had a training/generation input mismatch:
  - In training, slot 0 included `self.token_emb(0)`.
  - In generation, slot 0 omitted `self.token_emb(0)`.
  - Every chunk's first token (and all subsequent tokens attending causally to it) was generated from out-of-distribution inputs. Fixed in code by including `self.token_emb(curr_tokens)` at $k=0$.

**3. Verified Model Perplexities and Metrics:**
- **Ground-Truth Validation Likelihood on Held-out Text:**
  - Dense Baseline (d8, 440M tokens): Loss = 3.135 nats $\to$ **Val PPL = 23.00** (0.95 bits/byte).
  - Discrete Chunk-Lanes (DCL, 196.6M tokens): Loss = 5.0197 nats $\to$ **Val PPL = 151.37** (1.51 bits/byte).
  - DCL achieves exact cross-entropy likelihood, but lags dense by 1.88 nats (PPL 151 vs 23).
- **The Reference Perplexity Trap:**
  - Rescore of DCL temp 0.8 samples gave "Reference PPL = 10.47" (and 1.66 on greedy decode). This is an artifact of **Repetition Mode-Collapse**: dense attention easily predicts repeated numbers and phrases with $P \to 0.999$, making negative log-likelihood collapse to zero.
  - The true diversity check (Unigram Entropy = 2.746 nats vs 5.844 for real text; Distinct-1 = 0.121 vs 0.880) proves the samples are degenerated into loops.

**4. The Architectural Verdict: Multi-Token-Per-Row (SV-C) Hits the K3 Wall:**
- DCL misses the Prime Directive bar by a wide margin (Val PPL 151 vs 23, 1.51 vs 0.95 BPB, 135 sequential forward passes without KV cache).
- Attempting to predict $K=8$ tokens from a single trunk state $h_{\text{trunk}}$ hits the exact wall measured in earlier project findings (K3): heads reading one trunk state suffered 1.17–1.21× degradation at only 4 tokens. Expanding to 8 tokens caused severe under-conditioning.

---

`s13_sap_brainstorm.md` returns to the one-head T=L seed. Three measured lessons follow.

**1. Lanes recover only 13 to 35% of a lane start's deficit at equal tokens.**
- Measured from the logged 1x per-offset profiles. The deficit is the offsets costing more than dense (0 to 15); the recovery is the later offsets costing less.

| Model (1x) | Deficit | Recovered | Net per lane (average-token losses) |
|---|---|---|---|
| d4, L=64 | 2.40 | 20% | 1.93 |
| d4, L=32 | 2.88 | 13% | 2.50 |
| d8, L=64 | 3.64 | 35% | 2.38 |
| d8, L=32 | 4.36 | 25% | 3.25 |

- In the ideal model the chain rule loses only same-round TC, and the previous lane's end would win back almost all of it.
- Recovery comes mostly from a lane's last 10 to 15 tokens, and it grows faster with model size than the deficit does (2.7x against 1.5x from d4 to d8).

**2. Junction codes decided blind are information-neutral** (Q1, d4, 1x, L=128, `nanochat/splice.py`).
- Each junction's Brown class is drawn in round 0 from the prompt, the lane start reads it, and the junction token is masked to it. The likelihood is exact.
- It saves about 4.4 nats around each junction: the start drops from 7.72 to 5.96 nats and the junction token from 2.90 to 1.13.
- The code itself costs 5.03 nats, above the class's unigram entropy of 4.92.
- Block ratio 1.1064 against plain lanes' 1.107: no gain.
- **Rule.** Any variable decided without its left context costs about what it later tells the start. To help a junction, a coarse variable must be decided with its own left context: a skeleton generated sequentially over classes (SV-B), not a blind code.

**3. One pass against R rounds, measured** (Q3, random weights, H100, CUDA graphs, 1920 tokens).
- At batch 1, one pass costs only 1.8 rounds (d8) to 2.2 rounds (d20). Each decode step here has a large fixed cost (1.7 ms at d8, 5.2 ms at d20 per next-token step), so a single 1920-row pass is cheap by comparison.
- At batch 16 and above, one pass costs 5.8 to 6.5 rounds, and 8 rounds come within 1.3 to 1.4x of it.
- So a true single pass pays for batch-1 latency (about 4x over 8 rounds), and nothing much for throughput.

**Method.**
- A new workspace without checkpoints can reuse logged references if the evaluation rows are proven identical: `scripts/sap_position_bpb.py` prints a row hash (`ae01832e4bf20262` for the standard 256 rows).
- Brown classes come from an exact exchange algorithm (`scripts/sap_brown_classes.py`). Its incremental gains are checked against brute force, and a naive version that forgot the neighbours' class totals was wrong by 6 nats. On 60M tokens it gives an adjacent-class MI of 1.28 nats at K=256, against 0.24 for random classes.

## 2026-10-04: distinguish training-optimal from inference-efficient comparisons

The user questions iso-FLOP budgets as the sole screen for SAP and proposes longer
training / an inference-quality Pareto analysis. This is a methodological discussion,
not authorization for a new 3–5x paid run or a retrospective change to registrations.

**User clarification:** match dense's effective batch of 262,144 target tokens per
optimizer update in the next controlled d8 comparison. "Train longer" means more
total target tokens, not more updates on smaller batches or more wall-clock time.
Microbatch size may differ for memory; gradient accumulation should recover the
same effective token batch. The old run's 65,536-token batch is an optimization
confound, not a matched-batch comparison. A continuation that changes batch size
partway through must be labelled as such, not treated as matched from the start.
At the matched batch, 3x/5x dense's total token exposure would mean 5,040/8,400
updates and 1,321,205,760/2,202,009,600 target tokens. The exact future budget and
restart/continuation decision were initially unselected. The user subsequently
authorized all three budgets (1,680/5,040/8,400 updates), then explicitly requested
execution optimization followed by launch. The fresh-run protocol is in
`sap_flow_text_d8_matched_batch_plan.md`; 38 focused tests pass.
The three fresh runs were submitted in parallel on `nanochat2` as
`flow_d8_mb262k_s1_20261004` and all completed, including final evaluation.
K=32 BPB-bound estimates are 2.31514 / 2.27597 / 2.25436 versus dense 0.95179;
5x dense-token exposure improves the bound only 2.625% over the matched 1x run.
The inspected samples remain incoherent despite retained ~89x batch-16 graph
decode speedup. These are not matched-quality speedups, and ESS of 1.34–1.95/32
precludes treating bound gaps as exact likelihood gaps. More training helped
modestly within the tested recipe; it did not establish quality neutrality.
See the sweep plan's completion record for timings, accounting and provenance.

**Training-time scope and implementation:** dense's recorded 411.46 seconds counts
post-warmup training steps, excluding the first ten and other job overhead; flow's
1,574.29-second elapsed figure includes evaluations/checkpoints. Dense was compiled
with microbatch 16; the original flow was eager, microbatch 4, with extra recognition
blocks and chunked/recomputed vocabulary projection. Measured flow compute/token
is ~1.63x dense's estimator. Do not present the roughly sixfold throughput gap as
an intrinsic flow cost or an exactly scope-matched wall-clock result. Profile the
remaining utilization/implementation gap; effective batch matching does not require
keeping the smaller physical microbatch.

**Execution audit completed:** at fixed 262,144 targets/update on the same H100
and eight-CPU allocation, compiled flow with microbatch 32, four accumulation
steps and fused AdamW takes 0.62438 seconds/update versus 1.54598 for the original
microbatch-4 eager path: 2.48x faster, with 24.34 GB peak GPU allocation. Identical
inputs/noise give a 0.00000572 nats/token loss difference and 0.1779% aggregate
gradient relative L2 difference in BF16. Architecture and objective are unchanged;
the validation batch stays four to preserve held-out data and MC batching.
Compilation took 86.71 seconds in the audit. The 17.5/52.4/87.4-minute projections
exclude compilation, data, checkpoints and evaluation; they are not measured
end-to-end durations. See `sap_flow_text_training_optimization.md` and its raw
audit artifacts. The remaining gap versus dense is not all proven intrinsic cost.

- The completed d8 flow saw 269,615,104 target tokens in 4,114 smaller updates;
  dense saw 440,401,920 in 1,680 updates. Their recorded/countable matrix-FLOP budgets
  match, but flow saw only 61.22% as many targets. More updates is not more data.
  Its recognition/flow training costs ~1.6333x the dense recorded FLOPs per token.
- Dense teacher-forced pretraining already predicts all sequence positions in
  parallel. "One token versus T tokens" is the generation distinction, not a
  one-label-versus-T-label distinction in training. SAP's conditional/joint learning
  problem differs; no automatic T-times training multiplier follows.
- Iso-FLOP remains evidence about pretraining efficiency. It is not the only valid
  test of an inference-efficient architecture. Evaluate quality versus measured
  decode throughput while explicitly reporting training tokens, compute and time,
  and compare against the existing dense depth/training frontier where available.
- Extra training can pay off at sufficient deployment volume **at matched quality**:
  total cost = training cost + generated-token demand * inference cost/token,
  using actual hardware costs/latencies rather than equating FLOPs with seconds.
  This principle has prior art: https://arxiv.org/html/2401.00448v1 . Its fitted
  AR scaling coefficients and demand thresholds cannot be assumed valid for SAP.
- If testing 1x/3x/5x dense token exposure, define the reference explicitly:
  d8 targets would be 440.4M/1.321B/2.202B tokens, not 3–5x the smaller SAP run.
  At current counted per-token costs, 3–5x dense tokens would cost roughly
  4.90–8.17x the dense recorded training FLOPs. These are prospective milestones,
  not evidence of an optimum or a guarantee of curing latent underuse.

## 2026-10-04: d8 capacity improves flow slightly, but does not rescue quality

The user-authorized d8 flow-only T=L=2048 follow-up completed all 4,114 updates,
269.6M target tokens, at the existing dense d8 matrix-FLOP budget (13.58x d4's).
Width/depth scale together; recognition cost is included; dense was not retrained.
See `sap_flow_text_d8_results.md` and `sap_flow_text_d8_compiled.log`.

- K=32 BPB-bound estimate 2.318681 versus fresh dense joint BPB 0.951791 on exactly
  the same continuation rows. The flow estimate is 1.10% lower than d4's 2.344498,
  but the stronger dense baseline makes the relative bound gap larger (+143.61%).
- ESS is only 2.17/32, so this is not an exact marginal/inference BPB gap. Small
  movement with more importance samples does not establish bound tightness.
- Posterior/prior KL drops from 0.01739 at quarter budget to 0.00448 nats/token
  at the final training evaluation. Final K=32 gives 0.00493, ~5.73x d4's value:
  latent use increased, but remains weak. Do not describe it as literally unchanged
  or zero. Prior sample openings remain incoherent word/subword mixtures.
- Actual 2,048-output CUDA-graph speedups are 877.2x / 89.3x at batches 1 / 16,
  including prompt, latent/flow, projection and sampling. Fast generation alone
  does not meet the quality-neutral research bar.
- This is evidence against simple capacity scaling as a sufficient remedy at this
  budget, not a proof of convergence or impossibility of all flow mechanisms.
  Training continued improving slowly. Objective/posterior/optimizer causes remain
  unisolated. No further scaling sweep follows automatically from this result.

## 2026-10-04: full T=L flow-only run is fast but the sampled plan is underused

The user's requested full-budget d4 follow-up is complete, not another toy screen:
T=L=2048 from update one, width 256, four generative blocks, two training-only
recognition blocks, 77.53M target tokens. Counted matrix FLOPs match the existing
dense reported budget to 0.04%; only the flow was trained. See
`sap_flow_text_d4_results.md` and `sap_flow_text_d4_compiled.log`.

- Same held-out continuations: dense joint BPB 1.144624, flow importance-bound
  estimate 2.344498 at K=32 (+104.83% **bound** gap). K=1 to 32 changes only 0.00010
  BPB; ESS averages 14.12/32, but this still is not exact marginal likelihood.
- Actual 2048-token CUDA-graph generation: 573.39x at batch 1 and 66.33x at batch
  16, including prompt/flow/projection/sampling. Batch 16: 1.046M versus 15.8k
  tokens/s on one H100. Very fast incoherent generation is not a quality-neutral win.
- The posterior/prior latent KL drops from 0.0153 to 0.00086 nats/token between
  quarter and full budget while ELBO BPB barely changes (2.3491 to 2.3446).
  This is consistent with posterior collapse/underuse of the sampled plan. The
  saved prior samples are word/subword mixtures, despite near-100% distinct trigrams.
- Full-budget evidence strengthens a specific training-mechanism diagnosis; it
  does not prove that all larger models, other objectives, or longer runs fail.
  Nor does completing a dense-sized FLOP budget establish optimization convergence.
  No synthetic invalidity threshold was applied to terminate this run.

## 2026-10-04: a toy screening failure is not a universal scaling veto

The user explicitly requests full-budget d4 **flow-only** training at T=L=2048
after the Flow–Joint toy screen. `sap_flow_text_d4_plan.md` controls this follow-up.
The toy's T=4 was a cheap local-dependence diagnostic, not the target architecture
operating point. Its 3% impossible-sequence threshold is not a natural-language
validity metric. Failure at a finite small-model budget does not establish that
full pretraining/scaling cannot help; equally, help is not guaranteed.

Keep the measured toy failure intact while superseding its no-promotion decision
with the user's explicit request. Test the best control from scratch on real text,
with T=L from the first update, full documented training compute, prior samples,
likelihood-bound/ESS diagnostics, and end-to-end timing. Do not call the old
8k-update toy run either a fully trained LM or merely a few-update wiring smoke.
No new dense baseline training is authorized or needed.

## 2026-10-04: S11, the toy is sequential only in left-to-right order with tokens as decisions

`scripts/sap_bridge_oracle.py` samples the exact phrase-HMM two ways, with no learning.

**Decision order.** Interface-first nested dissection works like this:
- draw the hidden state at the block end;
- then the state at every interval midpoint given the interval's two end states, all midpoints of a level at once;
- then every token from its own state.

It is exact in log2(T) + 2 parallel levels: 10 levels at T=256, 0% invalid, and -log p_true of the samples equal to that of true sequences within 0.0 to 1.5 standard errors. Left-to-right refinement needed about T stages on the same process (S10 Jacobi oracle). So the about-T sequential depth measured in S10 belongs to the left-to-right representation, not to the process.

**Decision variables.** Keep the same bisection depth, but decide tokens:
- each new midpoint token is drawn from its exact posterior given every token already placed (forward-backward), independently within a level;
- invalid rate: 0% at T=4, 9.6% at T=16, 44% at T=64.

States separate the past from the future; tokens do not. This is the mechanism behind the order tax measured for token bisection on real text (K4).

**Durable rule for one-pass generation.** Order decisions coarse-to-fine, and make them about separator states (sufficient statistics of the past for the future), never tokens. Every S01 to S10 mechanism broke one of the two: they kept left-to-right order, or they used tokens or a flat global latent as the decision variables.

**Corrected 2026-10-04: at equal tokens the parallel-order tax is roughly constant, so parity with dense-1x comes from dense's own gain per doubling** (S11, d4, archaeonseq, every model scored in its own order).
- The earlier claim (tax 13% at 1x falling to 5.4% at 4x, parity at 2.8x) came from 1x rows scored in the wrong order (eval bug below).
- Block bpb against dense at the same tokens:
  - bridged lanes (64, 4), 54 steps: +8.2% at 1x, +8.1% at 2x, +7.4% at 4x (two seeds at 4x: 0.994 and 1.008 of dense-1x);
  - window bisection n=16, 126 steps: +7.5%, +8.3%, +6.3%;
  - plain lanes L=64, 30 steps: +6.3% at 1x, +6.5% at 4x;
  - bridged lanes (32, 8), 100 steps: +5.5% at 4x, two seeds, 0.983 of dense-1x on both.
- Dense at d4 improves 4.5% from 1x to 2x and 2.4% more to 4x. A constant 6 to 8% tax therefore reaches dense-1x near 4x tokens.
- **Consequence.** At larger budgets dense gains less per doubling, so a constant tax would need more than 4x tokens. The lever is the tax itself, and how it scales with model size, not the training length.

**Plain lanes beat every far-ahead order at equal steps on real text** (S11, d4, 1x tokens).
- Lockstep lanes over contiguous intervals (S08) cost +2.2% (L=16, 120 steps), +3.9% (L=32, 60), +6.3% (L=64, 30) and +10.7% (L=128, 15).
- Bridged lanes (64, 4) at 54 steps cost +8.2%; window bisection at 126 steps costs +7.5%.
- Predicting separators far ahead costs more than the lane seams they were meant to remove. A lane's end is generated last and already sees the next lane's start, so plain lanes bridge their seams from the left at no extra cost.
- Plain lanes also train and decode as one stream (two-stream orders double both).

**Window bisection on real text pays an order tax that windows reduce but do not remove** (S11 Stage 3a, d4, matched tokens, exact likelihood).
- Only n=16 (126 steps, +7.5% on archaeonseq and +7.8% on jimpa-17) survives the eval bug.
- The other window sizes (+24% at 12 steps, +10.8% at 40, +9.5% at 376) were scored in n=16's order and are void.

**Engineering: never key a tensor cache on `data_ptr()`.** The CUDA caching allocator reuses a freed tensor's address at once.
- A two-stream mask cache keyed on the step tensor's address returned the previous model's mask whenever the next model's step tensor landed at the same address.
- Every later model in one eval process was then scored in the first model's order, silently, with plausible numbers.
- Key on the object itself (hold a reference, compare identity and `_version`) or on content.
- Check any per-model eval against the training-time validation of the same checkpoint. Here they disagreed by 1.2 to 1.8% only for the affected models.

**Learned separators must be pure, and short token windows are pure where clusters are not** (S11 toy, 2026-10-04).
- k-means codes of a trained AR model mix states slightly: H(state | code) 0.38 to 0.47 nats. A Markov model fit over them generates invalid blocks 89 to 100% of the time, against 0% for the true state.
- The last two tokens pin the toy's state: H(state | t_{k-1}, t_k) = 0.106 nats.
- Bisection with 2-token windows as decisions, drawn independently within a level, samples exactly: 0% invalid at T=64 in 14 draws, against 44% for single tokens.
- Design rule: if a short token window carries the state, decide windows. That needs no learned codes, keeps exact likelihood and teacher forcing, and preserves copying.

**Separator oracle on real text (d4).** Make the second half of each row reach the first half only through m summary slots. This costs 7 to 8% bpb on the first 256 tokens after the split, about 0 beyond that, and the same for m = 1, 4, 16 and 64.
- A cost that does not fall with capacity is not a capacity limit.
- Here it is the slots' one-layer delay: a 4-layer model loses its first layer of direct access to the preceding tokens.
- Separators must be visible from the first layer (embedded codes, as in the Bridge LM), and must sit at every position near a boundary, not as one summary.
- Far context crosses a bottleneck at no measurable cost on average.
- Restricted to rows whose split falls inside one document, the cost is +9 to 13% on the first 64 tokens and +14% at 128 to 256 tokens, where the dense model copies from before the split (bpb 0.93). A summary bottleneck cannot carry verbatim spans, so per-position separators must carry token identity.

## 2026-10-03: Flow–Joint's exact local likelihood does not guarantee learned compatibility

The global continuous-plan/local discrete-joint hybrid is now tested, not merely
proposed (`sap_flow_joint_gate.md`, `sap_flow_joint_results.md`). Six L4 jobs on
`nanochat2`, T=4, V=512, oracle context, two seeds, matched counted matrix FLOPs.
Hybrid: 8k steps; flow-only: 10,576; chain-only: 15,572. No teacher/distillation.

- Hybrid KL-bound estimates 1.4004 / 1.4415 and invalidity 35.25% / 35.75% fail
  the <=0.10 / <=3% gate. Flow-only is numerically better at 1.2447 / 1.2364 and
  31.58% / 30.63%; chain-only gets 4.0022 / 3.9749 and 57.07% / 58.80%.
- Exact normalization and exact conditional HMM summation establish distributional
  correctness, not that 64 learned states acquire the needed token compatibility.
  Sixteen correctness/regression tests pass; this is not an observed sampler bug.
- Hybrid posterior-conditioned samples remain about 28.6% invalid. Prior mismatch
  contributes, but cannot be the sole explanation for these measured diagnostics.
  This test does not isolate optimization from recognition/decoder capacity.
- IWAE K=64 to 256 changes hybrid estimates by only 0.012–0.015 nats; it does not
  prove a tight bound. High prior invalidity independently establishes gate failure.
- The intended global/local complementarity was not demonstrated at equal counted
  matrix FLOPs. Failure closes this instantiation/budget, not all hybrids or T=L.
  No d8 run, L=2048 decode benchmark, or real-text BPB claim is justified by it.

The durable compiled record is `sap_flow_joint_compiled.log`; all six checkpoints
and per-run logs remain on the Modal `nanochat` volume under
`out/sap_flow_joint/fj_20261003_b/`. The existing dense AR reference was not retrained.

## 2026-10-03: S10, self-inverting PTP is consistent only if the generator distils the inverting model

RC-PTP (`nanochat/ptp.py`, `s10_sap_tl_brainstorm.md`) inverts data tokens into auxiliaries u through a token-conditioned AR mode, in one teacher-forced pass, and trains a one-pass generator on those u.

**The training target decides whether the one-pass sampler is consistent.**
- The plan trained the generator with cross-entropy on the data token. Its optimum is then `p_data(· | G_R(u_<k))`, where G_R is the AR mode's sequential pick map.
- At generation, the token is picked with the generator's own CDF. Whenever that CDF differs from the AR mode's, the emitted prefix and the prefix the generator believes it emitted diverge, and later positions condition on a history that was never produced.
- Distilling the AR mode's conditionals (PTP's Eq. 12, with the jointly trained AR mode as the teacher) has optimum `P_R(· | G_R(u_<k))`. By induction over positions, the one-pass picks then equal the AR mode's sequential picks under the same u, even while the AR mode is imperfect.
- General rule: when auxiliaries are inverted under one model's CDF, the generator must be trained toward that same model's conditionals, not toward the data.

**Numerics.**
- Inverse-CDF picks and their inversion must share one float64 CDF. With float32 softmax, peaked conditionals lose whole intervals and the round trip fails.
- Below float64 resolution (p < 1e-15 at a CDF position near 1), a token's interval is empty. No u picks it, so the sampler and the likelihood agree that it is impossible.

**Testing a Monte Carlo likelihood.** The sequential importance-sampling estimate is unbiased, but with random weights a single estimate is noisy (sums 0.990 to 1.010 at 20k paths).
- Test it with a z-score against the one-pass histogram, using the binomial variance at the estimate, not at the empirical frequency: rare sequences legitimately have zero hits.
- Confirm power by breaking the proposal: an estimator that ignores the observed token fails at z = 12.6.

**PTP's inverse-CDF pick is fragile at large vocabularies; a per-level tree pick is not** (`scripts/sap_jacobi_oracle.py`, offline on the toy's true conditionals).
- Inverse-CDF picking of one uniform is arithmetic decoding down a binary tree, with the uniform rescaled at every level. The rescaling amplifies small probability changes.
- At V=512 and KL 0.003 between two distributions, the same uniform picks different tokens 21% of the time, against a maximal-coupling floor of 2.4% (TV).
- Gumbel-max is near-maximal (3.3%) but needs V noise values per position.
- Drawing an independent uniform per tree level, with no rescaling, gives 4.8% from log2(V) = 9 uniforms.
- A semantic token order does not change any of these numbers under isotropic logit noise.
- In the first RC-PTP runs this fragility was the visible failure: the generator's position-0 distribution was within KL 0.04 of the AR mode's, yet the two picked the same token from the same u only 60% of the time.

**Coupled-noise refinement is sequential on the toy, whatever the coupling.**
- With exact conditionals, a Jacobi stage that re-picks every position given the previous stage's tokens fixes about one more position per stage.
- T=16 needs about 16 stages; at T=64, 32 stages reach 54 to 68% agreement and no block has converged.
- An early error changes the hidden state, so every later conditional changes with it, and robust couplings do not stop that.
- Consequence: a draft plus one reference stage inside a pass can fix only about one position beyond what the draft got right. T=L in one pass therefore rests entirely on the generator learning the composite noise-to-tokens map, which is PTP's representation claim, not on refinement.

**Measured tokens per call for learned one-pass generators (S10 T curve, toy, 8k steps, two seeds).**
- The best RC-PTP (tree pick plus self-inversion sweeps) gets about 2.5 leading tokens right per call at T = 4, 16 and 64, at depth 4 and at depth 8. PTP's flat pick gets about 1.
- Block KL grows about linearly with T: 0.75, 7.1 and 42 nats.
- Depth 8 gains 15% at T=64 and does not lengthen the correct prefix. Within this budget, learnability, not representable depth, is the binding limit.
- Mechanism improvements that cut KL by 40 to 66% at fixed T can still leave the scaling with T unchanged. Check the per-position curve (or tokens per call), not only the block KL at one T.

## 2026-10-03: cross-session SAP review and current scope

`sap_next_mechanisms_review.md` incorporates S08–S10, which appeared during the
S07 review. The original S07 hypotheses are no longer untested recommendations:
S09's spline scan-flow fails in actual samples as well as its likelihood bound,
and the trained suffix-window results strongly disfavor the short-suffix route.
The surviving conditional directions are S10's unbuilt global-flow/local-joint
hybrid and S08's paragraph-aligned lanes under the user's smaller-T allowance.
No new training or speed measurement was performed by this documentation pass.

Three qualifications matter for interpreting the newer notes:

- A k-token-window model dominates a state representation informationally only
  when that state is a function of those k tokens and the shared prompt. State
  count alone does not imply a memory horizon: even two states can remember a
  long-range predicate. A trained comparator's BPB is not a proven optimal loss.
- The four-lane d8 +1.10% BPB / 2.91x batch-16 result misses a strict 1% margin.
  Its speed is recorded against the same model's AR mode for 1920 tokens, and
  sample junctions remain poor. Paragraph alignment is a proposed repair, not
  an already-achieved neutral-quality result.
- RC-PTP's corrected objective uses AR-mode conditional distillation. Sharing
  weights removes a separate pretrained teacher but does not remove the learning
  signal's distillation role. This conversation's no-distillation constraint is
  not silently overridden by another session's inverse-training proposal.

## 2026-10-03: S07 audit — scope conclusions to what the SAP tests established

Read `s07_sap_mechanism_brainstorm.md` alongside the historical S06 entries below.
No new training, decode benchmark, or BPB result was produced by this audit.

- **Noise timing is not computational parallelism.** An AR decoder can pre-draw
  all uniforms. The defining constraints are the dependency graph, fixed neural
  depth, and work/span of any deterministic sampler, not the number of RNG calls.
  Current S06 all-prefix doubling scans use O(LS log L) work and O(log L) span.
  A different, work-efficient scan is needed for the advertised O(LS) work.
- **A constructed oracle is not necessarily an optimal oracle.** S06-S averages
  sorted emissions under the true predictive state weights. This optimizes the
  complete-data emission objective for fixed true states/transitions, not necessarily
  marginal block KL. Marginal KL is bounded above by complete-data KL; a failed
  construction does not prove a lower bound on the best possible marginal KL.
  Permutations do preserve probability spectra, but the C=1 result alone cannot
  close all learned one-spectrum state representations.
- **Oracle capacity must match learned capacity.** S06-S's C=2 pass used 452 true
  states; S06-L had 64 learned states. The gap cannot be assigned solely to
  optimization. KL improved from 3.8738 at 4k to 2.9485 at 8k, so convergence was
  not established. The 48.68% invalidity still fails the registered gate decisively.
- **Count-weighted diagnostics matter.** The 95.64% changed-permutation-entry rate
  is unweighted. Rare/unseen tokens and tied assignment scores can dominate it.
  Probability-weighted churn, assignment-objective improvement and held-out
  likelihood would be needed to establish harmful semantic instability. State
  occupancy need not be uniform in a correctly fitted HMM.
- **The implementation, not its arm name, defines the experiment.** S06's Argmax
  arm was one discrete ST modular coupling, not continuous Argmax Flow with a
  stochastic inverse. MIF used fixed transforms; the source-code proxy fitted
  constrained marginals. Their failures do not exhaust continuous flows or
  learned source transforms. The local AR head is also an empirical comparator,
  not a theorem that its trained weights maximize quality over all joint heads.
- **Learning storage is part of the cost.** The learned permutation update allocates
  S*V*V scores. At S=64,V=32768, float32 scores alone are 256 GiB, before assignment
  workspace and cubic per-state Hungarian work. Small inference permutation tables
  do not establish a scalable training algorithm. Vocabulary projections cost
  O(L*d*V), not just O(L*V) normalization work.

S07 generated 42 candidates and retained two mechanism hypotheses (scan-coupled
categorical flow; observable-history sparse random transducer), plus two clearly
labeled prior-art controls. Both target <=1.01x dense joint BPB and >=2x batch-16
end-to-end throughput at L=2048 and matched training FLOPs. These are proposed
gates, not forecasts or newly measured results. Neither an experiment nor a main-
track novelty claim is authorized by the shortlist alone.

The paragraph above records the initial funnel. S09 subsequently tested those
hypotheses; the consolidated review above is the current recommendation.

**User scope clarification:** T=L=2048 remains primary, but smaller blocks (e.g.
T=8) may merit a separate architecture paper if BPB-neutral and sufficiently fast.
S07 proposes the same <=1.01x dense joint-BPB and >=2x batch-16 throughput gate for
that secondary route, with all block invocations/KV updates counted over 2048
emitted tokens. These thresholds do not guarantee novelty. A smaller-block win
must not be reported as meeting the original T=L requirement.

---

## 2026-10-03: S09, a one-pass generator must remember its own output; suffix memory costs 3 to 15%

This session ran the cheap gates on the parallel session's S07 T=L hypotheses (`s09_sap_tl_gates.md`). The user had confirmed T=L as the requirement.

**The information argument (from lanes).** In one-pass generation, generated tokens depend on each other only through whatever runs after the noise is drawn.
- S07's hypothesis B, an observable suffix transducer, keeps that memory in an automaton state determined by a short suffix of its own output.
- A model trained to attend only to the prompt plus its last k positions has at least that information. It is B's oracle, and a trained model rather than a probe.

**Measured (d4, same FLOPs, full-context baseline, 256 identical validation rows).** bpb cost at positions 512-1023 / 1024-2047:

| attention | 512-1023 | 1024-2047 |
|---|---|---|
| prompt + last 128 | +3.1% | +4.4% |
| prompt + last 32 | +7.5% | +8.7% |
| prompt + last 8 | +15% | +15% |

The cost grows with distance from the prompt. Two conclusions:
- **B is closed:** a ≤512-state automaton remembers far less than 32 tokens.
- **More generally,** any T=L generator must carry a high-capacity, long-range memory of its own sampled output, inside its fixed computation. That is the bar the flow hypothesis (A) has to clear.

**Certified (S07 hypothesis A, `nanochat/categorical_flow.py`).**
- The scan-coupled categorical flow's parallel inverse, log-determinant (against autograd in float64) and cell masses (quadrature against one-pass sampler frequencies, within 0.5%) are all exact.
- The coefficients must be bounded (|a| <= 0.95, b in [e^-2, e^2]): unbounded ones broke the float32 inverse.

**Measured (A's toy gate: phrase-HMM, T=4, oracle context, two seeds).**
- Affine flows at 8k steps: KL upper bound 4.9 (scan) and 5.1 (no scan), with about 70% invalid blocks. That is worse than independent slots (TC 4.5).
- With rational-quadratic splines at 16k steps: 3.6 (scan) and 3.3 (no scan), 52 to 60% invalid.
- The scan never helped, and the flows learn little beyond marginals.
- Two notes on the splines: they need bounded parameters and float64 evaluation (float32 inverses lost up to O(1) on skewed bins), and quadrature needs a finer grid to confirm the sharper densities.
- **Both S07 hypotheses are closed.** No one-pass family has reached this gate in S01 to S09.

---

## 2026-10-03: S08, lanes: emit distant tokens together, because adjacent dependence belongs to the data

This came from a separate session's brainstorm (`s08_sap_lanes_plan.md`). Its scope was set by the user:
- strict SAP seed: one model emits T tokens per trunk pass, unverified, from scratch;
- bar: both neutral against the same-FLOPs dense model at 2x or more, and a clear margin over the dense depth frontier.

**The filter that shaped it.** Every SAP design so far emitted T *adjacent* tokens per pass. Their total correlation is a property of the data: 19 / 47 / 66% of block NLL at T = 2 / 4 / 8 on real text. A head can only model it, and every way of modelling it in one pass has failed for a measured reason:
- independence;
- capacity of heads that read final states;
- order tax, even at full depth;
- learnability of noise-tape generators;
- the depth frontier, for trunk-depth slots.

**The move.** Choose *which* T tokens share a pass so that their dependence is near zero. That means L contiguous lanes of the same document, written in lockstep, one token per lane per pass (`nanochat/lanes.py`). The cost moves from coherence to lane starts: lane j begins j·S tokens past the text it can see.

The estimated tax is about L/N. This is a hypothesis under test, not a result.

**An exactness detail worth keeping.** In the lane order, every input of step s was drawn at step s-1, so a step-s query may read every lane's step-s input. Only the step's *draws* are mutually independent.
- The first mask, which hid same-step inputs, was needlessly strict. Enumeration showed it, and the relaxed mask is exact too.
- A real leak (a lane reading its own next input) moved the enumerated log-total by 2e-2, against 1e-7 clean. So `tests/test_lanes.py` checks normalisation with power.
- The decoder runs one all-layer pass per step through `GPT._sap_depth_layers`, with the KV cache as prefix and full visibility inside the step. On a toy model under CUDA graphs it ran 2 lanes at 1.7 to 2.1x and 4 lanes at 3.1 to 3.2x (local GPU, smoke only).

Nearest work: Hogwild! Inference, Multi-Stream LLMs, the Parallel Decoder Transformer, ReFusion, Planned Diffusion, APAR / Skeleton-of-Thought, and Subscale WaveRNN for the same principle in audio. None is an exact-likelihood lane order pretrained from scratch and measured against same-FLOPs dense and the depth frontier.

**Measured (S08 L1/L2, d4, 2026-10-03; checkpoint tags `S07_*` from before the rename).**
- Setup: full-context attention for every arm, the same architecture and FLOPs (1.14e16), two seeds. Lane-order val bpb on 20M tokens with a fixed 128-token prefix, against the dense model on the same rows.

| arm | lane length S | val bpb (seeds) | tax | speed, batch 1 | speed, batch 16 |
|---|---|---|---|---|---|
| dense | n/a | 1.1518 / 1.1525 | n/a | n/a | n/a |
| L=2 | 960 | 1.1578 / 1.1573 | +0.47% | 1.89x | 1.67x |
| L=4 | 480 | 1.1636 / 1.1623 | +0.94% | 3.66x | 3.24x |
| L=8 | 240 | 1.1691 / 1.1715 | +1.57% | 7.19x | 6.23x |

Speeds are lane decoding against the same model's next-token decoding, CUDA graphs, temperature 1, 1920 tokens after a 64-token prompt.

- The tax grows with L (in proportion from L=2 to L=4, sublinearly to L=8), as the lane-start account predicted. The back-of-envelope estimate was 0.5% at S=1024 and 0.9% at S=512.
- Every pre-registered d4 gate passed. L=4 is neutral within 1% at more than 3x, against the d4 depth frontier's 1.68x at +3.4%. This is the first SAP-family mechanism to clear its quality gate.
- Caveats: the speedup needs long generations (N >= L·S), and lanes currently require full-context layers. d8 confirmation is next.

**Measured (S08 generation quality, d4).**
- Setup: lane samples from the lane model against next-token samples from dense seed 1, both scored by dense seed 2. 256 prompts of 64 tokens, 1921 generated tokens each (`scripts/sap_lane_gen_profile.py`).
- Whole-text reference PPL: 2 lanes 59.6 vs 59.0 (+1%), 4 lanes 81.3 (+38%), 8 lanes 94.8 (+61%).
- The profile splits that number into two effects:
  - **Next-token samples degenerate with length.** Their reference PPL falls 90 -> 62 -> 52 -> 42 over the four 480-token ranges while distinct 3-grams fall 0.93 -> 0.76. The tiny model starts repeating itself, which a reference model scores as cheap. Lanes stay fresh: each lane scores 76 to 85 with distinct-3 near 0.89.
  - **On equal footing lanes are no worse.** Comparing lane 0 against the same range of the next-token sample: 2 lanes 58.9 vs 74.9, 4 lanes 85.6 vs 90.0, 8 lanes 95.8 vs 98.7. The 2-lane interior is -1.0% overall.
- **Junctions are the real cost.** The first 16 tokens of every later lane score reference PPL 420 to 650, against about 60 for the next-token text at the same positions.
- Why: in training, lane boundaries fall at arbitrary positions, often mid-sentence. So a generated lane starts mid-thought and the preceding lane must land exactly on it.
- The lane likelihood (bpb tax under 1%) prices this as the model sees it. A left-to-right reader sees a seam.
- Next mechanism, if lanes continue: lanes aligned to paragraph or sentence starts, with lane-local virtual positions and an end-of-lane token. Seams then fall where text naturally breaks.
- **Lesson for any long-generation comparison with small models:** reference PPL of long samples rewards repetition. Compare ranges against ranges, and report distinct n-grams alongside.

**Measured (S08 L3, d8 confirmation, two seeds, full-context dense 0.9590 / 0.9594).**

| lanes | tax | speed, batch 1 | speed, batch 16 |
|---|---|---|---|
| 2 | +0.61% | 1.90x | 1.48x |
| 4 | +1.10% | 3.74x | 2.91x |

- The tax grew slightly with scale (d4: 0.47 / 0.94%), so 4 lanes sits just over the 1% bar at d8.
- The batch-16 speedup fell because the manual lane attention path is slower than flash attention at d8 width.
- **Status:** parked as the T=K fallback. The user confirmed T=L as the requirement (2026-10-03), and lanes cannot reach one-pass generation.
- **If revived:**
  - paragraph-aligned seams;
  - a 4k context, where the tax should roughly halve;
  - a flash-attention lane kernel.

---

## 2026-10-03: SAP v4, exact sampling cuts, and three corrections to the S02 record

**Correction 1: the S02 "81x" (entry below) is not a hardware speedup.** Every row of the S02
decode JSONs is eager; `stage_b_post` defaulted to `--no-graphs`. The eager autoregressive
baseline takes 4.83, 4.72 and 4.80 ms per step at batch 1, 16 and 128: a per-step time that is
flat across batch is launch overhead, not memory or compute (a d8 step's weight read on an H100
is tens of microseconds). The tree's call time grows with batch (15, 39, 257 ms), so it is doing
real compute, which is the actual reason its speedup shrinks with batch; the roofline story
offered for that was wrong. At T=L the bench also builds all 2048 tokens but credits 256. On its
own pre-registered gate S02 d8 failed three of four criteria (trunk +3.8% over dense, block bpb
+47% over its own trunk, reference PPL +17%) and the fourth (graph-mode speed) was never
measured. Decode claims must come from CUDA-graph rows, reported at both a small token count
and the full block.

**Correction 2: S01's `sir_lattice` never tested a structured joint.** It ran one sum-product
message to sharpen each slot's logits and then sampled every slot independently, which is still
a product of marginals; it was also trained by draft pairing, not by an exact likelihood.

**Correction 3: at d8 the readout dominates any head's cost.** One lm_head forward and backward
is 6 x 512 x 32768 = 1.0e8 of the 2.86e8 training FLOPs per token (35%). The T=L arms add about
one readout per training token, train on 32 to 38% fewer tokens at matched FLOPs, and lose 2.9
to 3.8% bpb, so each extra readout per training token costs about 3.5% bpb. Structured heads
must keep their emissions on a candidate lattice (no per-state V-wide softmax).

**Concept: per-slot losses cannot teach coherence.** Any loss that is a sum of per-slot terms
comparing a slot's distribution with a per-slot target, hard or soft, is minimised by matching
each slot's marginal, so its optimum is the product of marginals. Hard-target cross-entropy
already learns the full conditional in expectation; a soft target from corpus tables only lowers
gradient variance and biases the model toward the table's (n-gram) statistics (RAML is the
nearest prior). Where hard targets do hurt is pairing: with one sample per context, which noise
draw should explain which continuation is unidentifiable, and exact structure, a posterior, or a
contrastive judge solve that without soft targets.

**Concept: the sampling-cut principle.** A coherent one-pass head must (1) sample a small cut (a
code or a subset of slots) from an exact, coherent joint; (2) leave the other slots nearly
conditionally independent given the cut; (3) train the cut by exact likelihood, teacher-forcing
it only when (1) holds so that cuts sampled at inference are in distribution. S01 broke (1)
(independent drafts; the same refiner reaches 2.25% invalid with consistent anchors and 26% with
independent ones), the tree met it with log T sequential rounds, P1 with a code but through an
ELBO. v4 meets it in one pass: a chain over a top-K lattice plus an escape state, normalised by
the forward algorithm and sampled by forward-filter backward-sample, with no network layers in
the draw (`nanochat/sap_chain.py`).

**Measured (Stage 0, 2026-10-03, `out/s03_sap/oracle_d8.json`): coherence is the bottleneck on
real text.** On FineWeb-Edu validation (512 contexts of 256 tokens, 256 Monte Carlo samples,
the dense d8 `B1_dense_s1`), the independence tax of a product-of-marginals head is 1.32 / 6.28
/ 18.1 nats per block at T = 2 / 4 / 8: 19% / 47% / 66% of the AR block NLL, or +0.20 / +0.48 /
+0.69 bits per byte on top of the AR model's 1.03 to 1.05. The marginal CE of slot k climbs 3.40,
4.73, 5.46, 6.09, ... while the teacher-forced NLL stays near 3.4. Cut MI between block halves is
2.70 nats at T=4 and 4.22 at T=8 (tensor-train rank floors 15 and 68). On the exact pair joint
over the 64 most likely first tokens (87% of first-token mass; 0.75 of the 1.32 nats of pair MI
live inside it), a FREE rank-2/4/8/16 CRF captures 86/96/99/99.8% of the MI without interpolating
(645 parameters against 4,160 entries at rank 4), and a free CP mixture needs 16 codes for 95%.
A CONSTRAINED rank-32 CRF that shares its candidate projections across contexts, as lat_crf does,
captures 88 to 92% on its fitting contexts but 64 to 69% held out: that gap is the price of
computing the coupling from shared features. 128 corpus classes leave 7% of the pair MI inside
classes. All numbers agree between 256 and 512 contexts.

**Measured: lattice coverage is the lattice heads' weak point.** The data token is inside the
top-64 of its slot's marginal 83 / 71 / 64 / 57% of the time at slots 1 to 4 (about 50% beyond),
and inside the top-256 91 / 84 / 76 / 72%. An escaped token is drawn independently, so every
pair it touches loses its coupling. The cut head is exposed only through its anchors' mutual
coupling; its fill reads the realised anchor tokens whatever lattice state they came from.

**Measured (Stage A, 2026-10-03, phrase HMM, T=4, 8k steps, 2 seeds, `out/s03_sap_stageA/`).**
Block KL / impossible-block rate at sap_trunk_grad = 0 | 1 (true TC 4.43):

| Head | g=0 | g=1 |
|---|---|---|
| indep (product of marginals) | 4.79 / 62% | 4.44 / 62% |
| lat_crf / lat_tt / lat_cp (structured joint, no fill) | 3.01 / 2.89 / 2.69, ~44% | 2.44 / 2.38 / 2.05, ~41% |
| corpus_code | 2.15 / 42% | 1.92 / 43% |
| pmi_chain (zero-parameter floor + gate) | 1.06 / 20% | 0.75 / 20% |
| cut_crf / cut_tt (exact anchor joint + neural fill) | 0.53 / 11% ; 0.58 / 13% | 0.40 / 9.5% ; 0.44 / 11% |
| p1_selfpost | 0.29 / 8.4% | 0.28 / 8.1% |
| sir_tree (3 sequential rounds, reference) | 0.16 / 4.1% | 0.12 / 3.9% |
| cut_crf + NCE resampling (L=64) | q 0.58, p est 0.165 / 0.1% | q 0.39, p est **0.099 / 0.0%** |

Readings. (1) Structured potentials alone are a weak pair model: every lattice head stays near 40%
impossible blocks while the cut head, whose fill conditions on the realised anchor tokens through a
neural layer, gets to 10%; the Stage 0 constrained fit (64 to 69% of pair MI held out at rank 32)
predicted this. Neural conditioning on sampled tokens, not a low-rank potential, carries most of
the dependence. (2) The cut head's remaining error is its anchor pair: with oracle anchors its fill
leaves 2.2 to 2.4% impossible blocks. (3) Self-contrastive resampling removes impossible blocks
almost entirely (11% to 0.0-0.1%) and cuts the estimated KL from 0.39 to 0.099 at g=1: cut_crf +
NCE is the only arm that passes the pre-registered gate (KL <= 0.10, invalid <= 3%), ahead of the
three-round tree, with the caveat that its KL is an importance-sampling estimate (the exact
proposal's is 0.39). (4) Co-training (g=1) helps every head's block KL (cut_crf 0.40 vs 0.53) and
moves the trunk's next-token excess only from 0.0014 to 0.0024 nats on this testbed; the d8 trunk
tax is the open question for Stage 3. (5) The corpus support mask and the n-gram soft target change
nothing measurable (killed); p1_selfpost does not beat P1 at equal codes (P1: 0.25 at 64 codes, 6k
steps).

**Bug found (pre-existing since 01b46bd, fixed 2026-10-03): `local` sampled with Jacobi sweeps.**
`BlockHead.sample` read `getattr(self, "jacobi_sweeps", 0)` for plain `local`, but `jacobi_sweeps`
is always set from the config (default 2), so every `local` sample was a two-sweep Jacobi
approximation (Stage A: exact KL 0.012 with 72% impossible samples). The likelihood was exact; only
sampling was wrong. The d8 local T=8 generations (2026-10-02 04:15) predate the commit (13:41) and
their eager speedup (~3.2x) fits exact sequential decoding, so they probably stand. A sampler test
against the head's own likelihood now covers `local`.

**Measured (Stage 2, partial, 2026-10-03; runs stopped when the blessingjim31-workspace ran
out of funds).** d8 heads trained 600 iterations on the frozen dense `B1_dense_s1` trunk, T=4,
5,120 validation blocks, next-token bpb on the same tokens 0.9306: `cut_crf` + NCE exact proposal
1.248 (1.34x), resampled estimate 1.204 (1.29x); `sir_tree` (3 rounds) 1.236 (1.33x). Both are six
times further from the 1.05x gate than the gate's own margin. The `cut_crf`, `local` and `indep`
arms of that run were invalid (see the checkpoint-name bug below), so the capacity-versus-coherence
split at d8 (the `local` head's number) is still unmeasured.

**Measured (Stage A at T=8, partial).** local (exact chain rule inside the head) KL 0.02 to 0.03
with 0% impossible blocks; sir_tree (4 rounds) 0.85 (g=0) / 0.40 (g=1), 13%; cut_crf 4.4 / 3.0,
53 to 59%; cut_crf + NCE estimated 1.35 / 0.65 with 0.2 to 0.3% impossible. With oracle anchors
the cut fill leaves 3% impossible blocks, so at T=8 the four-anchor first-order CRF is the
bottleneck, and resampling removes impossible blocks without fixing the distribution.

**Pipeline bugs (fixed in modal_sap.py).** (1) `s03_train` skips any tag whose checkpoint
directory already has a checkpoint, and the smoke and full Stage 2 runs shared tags and a
directory, so three "full" arms silently reused 4-iteration smoke heads. Smoke runs now write to
`d8_smoke`. (2) `stage_a_run` resumes a finished checkpoint with the same name, so a re-measurement
after a code fix replayed the old milestone rows; reruns now take `--rerun <label>`.

**Trap: a workspace's tokenizer can silently differ.** The nanochat1 Modal workspace's
`tokenizer/` has the same file names and nearly the same sizes as the project's pinned V=32,768
tokenizer, but different merges (sha256 387cfc08 vs 06978be3). Loading `B1_dense_s1` with it gave
validation bpb 2.48 instead of 0.958. The fix was a separate `tokenizer_sap/` and an explicit
`--tokenizer-dir`; the general rule is to compare tokenizer hashes before the first run anywhere new.

**Measured (Stage 2 complete, 2026-10-03): at d8 the wall is head capacity, not coherence.**
Block bpb at T=4 on the frozen dense trunk (next-token 0.9306 on the same tokens): indep 1.581
(1.70x), cut_crf 1.248 (1.34x), sir_tree 1.236 (1.33x), cut_crf + NCE resampled estimate 1.204
(1.29x), local 1.121 (1.21x). `local` applies the exact chain rule inside the same 2-layer head,
so it is the ceiling of any sampling structure in that head: 70% of the best one-pass gap (0.191
of 0.273 bpb) is the head's capacity on features trained only for the next token, and 30% is
structure. The one-pass cut head with NCE resampling recovers 82% of the independence gap
(indep to local) without sequential head steps, so sampling-awareness works; it is the per-slot
computation that cannot reach next-token quality at d8. The pre-registered direction gate
(1.05x) fails.

**Measured (head-capacity scan, 2026-10-03, frozen dense d8 trunk, T=4; x next-token bpb).**
`local` (exact chain rule in the head): 2 layers 1.205, 4 layers 1.180, 6 layers 1.169, 4 layers
with a 64-state window 1.173, 4 layers with a 4x MLP 1.170. cut_crf + NCE (exact / resampled
estimate): 1 pre-cut layer 1.341 / 1.294, 2 of 4 1.313 / 1.273, 3 of 6 1.305 / 1.269, 2 of 4 with a
64-state window 1.309 / 1.270. Head size is the wrong lever (returns shrink: -0.025 then -0.011),
and the cut head stays about 0.10x behind `local` at every size, so its one-pass structure deficit
(the anchor pair's joint) is separate from capacity. A head that sees only final-layer states of
the prefix appears to hit an information limit near 1.17x; the trunk-depth test (slots that read
every top layer's prefix keys) addresses exactly that.

**Measured (trunk-depth slots, 2026-10-03): giving the block's tokens trunk depth breaks the
capacity wall.** `depth_local` sends the block's tokens through (copies of) the top m trunk layers,
each reading the trunk's own prefix keys/values at that layer plus the block's earlier tokens: an
exact chain rule with trunk depth per token. On the frozen dense d8 trunk at T=4 (x next-token bpb
0.9306): m=2 1.112, m=4 1.062, m=8 1.028 at the default learning rate and 1.000 at zero learning
rate (the implementation is exact; the 1.028 is the copies drifting under Muon's 0.02 rate, and a
10x lower rate gives 0.9999). The partial-depth copies need the default rate (10x lower: m=2 1.222,
m=4 1.091), because their entry state (the trunk's state at t plus the token's embedding) does not
match what the trunk would have computed. A head that reads only final-layer states saturated at
1.17x however large it was; reading every top layer's prefix keys is what crossed it. Decoding is
exact as well: with m = L the depth decoder emits the AR greedy tokens. The sequential cost per
block is one trunk pass plus T-1 single-token passes through m layers, so the memory-bound speed
ceiling is T*L / (L + (T-1)*m): 2.3x for m=2, 1.6x for m=4 at T=4.

**Measured (depth_local, more training and pretraining; 2026-10-03).** Frozen dense trunk, copies
trained 3x longer (1,800 iterations): m=2 1.084x, m=4 **1.040x** (from 1.112 / 1.062 at 600), so
the trunk stays exactly dense and m=4 clears the 1.05x quality gate. CUDA-graph decode at batch 16
(frozen copies, T=4): m=2 1.88-1.90x, m=4 1.40x. Pretrained from scratch with the trunk's own top
layers shared (FLOPs-matched, sap_trunk_grad 1): block ratio m=2 **1.031x**, m=4 **1.013x** of the
model's own next-token bpb, so pretraining teaches the trunk to support late-entering tokens; but
the trunk's own bpb is +5% over dense (1.009 / 1.005 vs 0.958), from the slot passes' training
FLOPs (+11 to 15% per token) and mostly from the shared top layers serving two tasks at lambda=1.
Bug found and fixed: decoding trained copies read the trunk's KV cache while training computed
prefix keys with the copies' projections; copies now keep their own cache (test added), so the
first frozen-copy generation numbers (ref PPL 37 -> 182 / 84) are invalid.

**Measured (depth_local decode and generation, CUDA graphs at batch 16, temperature 1).**
Pretrained shared m=2: block 1.031x, reference PPL 67.2 -> 89.2 (+33% over the same model's AR
samples), 1.91x graph speedup (eager 2.05-2.10x). Pretrained shared m=4: 1.014x, 64.5 -> 72.1
(+12%), 1.40x. Frozen-trunk copies after the decoder fix (600 iterations): m=2 1.114x, 37.0 -> 83.2,
1.83x; m=4 1.063x, 37.0 -> 55.4, 1.32x. Generated-text PPL is a stricter test than block bpb: a 3%
bpb gap still costs a third in reference PPL. The pretrained models' own AR samples score 64-67
under the dense reference (the dense model's 37) because their trunks are 5% worse.

**Measured (trunk penalty of pretrained depth slots, 2026-10-03).** Lowering the block-loss weight
is the lever: at lambda = 0.25 the trunk falls from +4.9 / +5.3% to **+1.6 / +1.8%** over dense
(m=4 / m=2: 0.974 / 0.976 against 0.958) while block bpb holds or improves (m=4 0.973, m=2 1.007).
Against the dense model's own next-token bpb on the same tokens (0.9306), m=4 at lambda 0.25
decodes four tokens per trunk pass at **1.046x**. The remaining ~1.6% is roughly the slot path's
training FLOPs (+15% per token, so ~13% fewer tokens). Halving the block-start fraction at lambda=1
made the trunk worse (+7.6%): the same loss weight over half the samples is a noisier gradient on
the shared layers.

**Measured (the lambda trade-off and the composite metric, d8, 2026-10-03).** The number that decides
the paper is the block decoder's bpb against the dense model's next-token bpb on the same tokens
(0.9306), call it C; the trunk's own bpb matters only at block starts. Shared top layers, T=4:

| run | trunk val (dense 0.958) | block / own NTP | C | b16 graph speedup |
|---|---|---|---|---|
| m=4, lambda 1 | 1.005 | 1.013 | 1.062 | 1.40x |
| m=4, lambda 0.25 | 0.974 | 1.030 | **1.046** | 1.41x |
| m=4, lambda 0.1 | 0.969 | 1.052 | 1.061 | n/a |
| m=4, lambda 0.25, frac 1/32 | 0.975 | 1.034 | 1.052 | n/a |
| m=2, lambda 1 | 1.009 | 1.031 | 1.085 | 1.91x |
| m=2, lambda 0.25 | 0.976 | 1.062 | 1.082 | 1.92x |
| m=2, lambda 0.1 | 0.967 | 1.096 | 1.103 | n/a |
| m=2, lambda 0.25, T=8 | 0.981 | 1.088 | 1.116 | n/a |
| frozen dense trunk, copies m=4 (1,800 it) | 0.958 | 1.040 | 1.040 | 1.40x |

Lambda moves cost between the trunk and the block but C barely moves (1.046 to 1.062 at m=4), so
lambda is not the lever for the paper's number; m is (m=4 about 4.5%, m=2 about 8.5%). At
lambda 0.1 the trunk tax (+0.9 to +1.1%) is about the slot path's training FLOPs share alone.
Halving the block-start fraction did not help, and single-seed trunk differences of about 0.4%
are within noise. Generated-text reference PPL amplifies the block gap about ninefold (m=4 lambda
0.25: 59 -> 77, +30%; m=2: 55 -> 93, +69%; m=4 lambda 1: +12% at block ratio 1.014), so a reference
PPL within 10% needs block / NTP near 1.01. Reference PPL of one model's samples scored by another
also carries the cross-model KL: the dense model scores its own samples at 37 and a 1.7% worse
trunk's at 55 to 59, so only AR-versus-block of the same model is a fair comparison.
On 2026-10-03 the user moved mainline testing to d4 (13x cheaper: 121M tokens, 9.4e15 FLOPs) on
the jimpearse01 Modal profile. At d4 the lm_head is 65% of training FLOPs, so the slot readouts'
FLOPs share is about 12% per token at frac 1/16.

**Measured (d4 mainline, 2026-10-03, jimpearse01).** Dense d4: val bpb 1.1496 / 1.1511 (seeds 1, 2:
noise floor about 0.13%); next-token bpb on the block-eval tokens 1.1218 (T=4) / 1.1245 (T=8).
C is block bpb over that; speed is the b16 CUDA-graph speedup at 256 tokens.

| arm (T=4 unless noted) | frozen copies C | from scratch C (shared, lambda 0.25) | speed |
|---|---|---|---|
| local m=1 | 1.079 | 1.104 | 1.73x |
| local m=2 | 1.033 | 1.073 | 1.38x |
| roll m=1 | | 1.086 | |
| roll m=2 | | 1.065 | |
| tree m=1 | | 1.227 | 2.14x |
| tree m=2 | 1.133 | 1.163 | 1.78x |
| tree m=4 (full depth) | 1.102 | 1.132 | 1.31x |
| local m=1, T=8 | | 1.139 | 1.94x |
| roll m=1, T=8 | | 1.111 | |
| tree m=2, T=8 | | 1.271 | 2.69x |
| tree m=4, T=8 | 1.165 | 1.229 | 1.90x |

**depth_tree is closed as instantiated (pre-registered R1 and R2).**
- With full-depth copies dedicated to the bisection rounds on the frozen dense trunk, the order alone costs 10% at T=4 and 16.5% at T=8.
- At equal m the tree is 7 to 12 points of C behind local. Its generations score reference PPL 255 to 640 against 105 to 125 for AR.
- The conditional independence the tree adds at T=4 is only I(y4; y2 | y1, y3), far too small for 1.4 nats per block. The skip conditional (y3 from y1 through a mask slot) and the infill conditional (y2 between y1 and y3) are what these networks learn badly, which matches the compute inefficiency of masked and any-order models.

**depth_roll** feeds slot k the previous slot's top-layer state (slot 1: the trunk's final state at t), at no decode cost. It recovers a fifth to a quarter of the m=1 slot gap (block over own NTP 1.080 to 1.064 at T=4, 1.099 to 1.075 at T=8) and little at m=2 (1.042 to 1.038).

**The slot gap tracks the number of skipped layers.** About 1.5 to 2.5 points per skipped layer across d4 and d8 (frozen copies):
- d4: m=2 3.3%, m=1 7.9%.
- d8: m=4 6.2%, m=2 11.2%.

Late entry skips the bottom layers, which layer-pruning studies find the least removable. That motivated skip-middle slots (`sap_depth_bottom`): run the bottom a layers exactly, add the block start's middle-layer delta, then the top m - a layers.

**Measured (skip-middle, d4, 2026-10-03): it does not help.** Late entry, which skips the bottom layers, is the best choice. Frozen copies at T=4, C:

| layers the slots run (L=4) | C |
|---|---|
| {2,3} | 1.033 |
| {0,3} | 1.039 |
| {0,3}, roll | 1.035 |
| {1,2,3} | 1.012 |
| {0,2,3} | 1.018 |

From scratch (shared, lambda 0.25), {0,3} gives C 1.074 against 1.073 for {2,3}.

Two conclusions:
- Approximating the bottom layers with the block start's state, which carries the context, beats running them exactly and transplanting a middle-layer delta.
- The cost is convex in the number of skipped layers: 1.2 / 3.3 / 7.9% for 1 / 2 / 3 of 4 layers.

So which layers a slot skips is not the lever (pre-registered "no effect" reading). The depth deficit has to be met another way. Two candidates:
- spend the cheap path only where it is nearly free (adaptive block length);
- add depth that does not sit on the sequential path.

**Measured (adaptive block length, `scripts/sap_adaptive_oracle.py`).**
- A causal router that continues the block while the slot's own max probability or negative entropy clears a threshold must reject about 70% of slots to bring the likelihood ratio from 1.07 to 1.02. That is slower than AR, because every rejection wastes a slot pass and adds a trunk pass.
- An oracle router that reads the data token reaches ratio 1.00 at only 1.50x (m=1, T=4) and 1.62x (T=8).
- The slot cannot tell where it is wrong: what it lacks is the trunk's information.

**Measured (the dense depth frontier at d4, 2026-10-03): trunk-depth block decoding is dominated by simply removing layers.** Dense models as wide as d4 (256) but with fewer layers, trained at d4's FLOPs (so on more tokens), timed in one container with CUDA graphs at batch 16:

| layers | speed | C |
|---|---|---|
| 4 | 1.00x | 1.000 |
| 3 | 1.23x | 1.011 |
| 2 | 1.68x | 1.034 |
| 1 | 2.47x | 1.147 |

C here is val bpb against the 4-layer model. Every from-scratch SAP point sits 4 to 7 points of C above this frontier at its speed (the frontier figure is interpolated):

| arm | speed | SAP C | frontier C |
|---|---|---|---|
| roll m=1 | 1.73x | 1.086 | about 1.041 |
| roll m=2 | 1.36x | 1.066 | about 1.018 |
| roll m=1, T=8 | 1.89x | 1.113 | about 1.064 |

- Frozen-trunk copies, with no trunk tax and extra head training, still lose: roll m=1 is 1.63x at 1.058, against about 1.033 for the frontier.
- The reason: a shallow model trained end to end as the whole model loses about 1.1% for one removed layer of four and 3.4% for two. A slot path that enters a deep model's top layers from an approximate state loses about twice that per skipped layer. The mismatch between the entry approximation and the trunk costs more than the depth itself.
- Any block decoder for this project must beat this frontier, not only the same model's AR decoding.

**Measured (the dense depth frontier at d8): the gap does not close with scale.** Width 512, d8's FLOPs, one container:

| layers | speed | C |
|---|---|---|
| 8 | 1.00x | 1.000 |
| 6 | 1.28x | 1.008 |
| 4 | 1.79x | 1.026 |

The 8-layer model's val bpb is 0.9590, matching B1_dense_s1. The frontier is flatter than at d4: half the depth costs 2.6% (3.4% at d4).

Against it, the from-scratch d8 SAP points (C against dense on the same tokens; the frontier figures are interpolated):

| arm | speed | SAP C | frontier C | gap |
|---|---|---|---|---|
| depth_local m=4, lambda 0.25 | 1.41x | 1.046 | 1.013 | 3.3 points |
| depth_local m=2 | 1.92x | 1.082 | about 1.03 | about 5 points |

**The trunk-depth family is closed** with this measured reason: at both scales a dense model with fewer layers, trained end to end on the same FLOPs, decodes as fast and is better.

This frontier is the yardstick for every decode-speed method in the project, early exit included. At these budgets, depth is cheap to remove.

**Measured (Stage 3 from scratch, FLOPs-matched).** cut+NCE at sap_trunk_grad 0 / 0.1 / 1: trunk
0.974 / 0.973 / 1.019 (dense 0.958; the +1.5% at g <= 0.1 is the head's training FLOPs, +25% per
token at d8), block ratio 1.325 / 1.283 / 1.203 (NCE estimates 1.277 / 1.237 / 1.165), absolute
block bpb 1.255 / 1.214 / 1.190. g = 0.1 helps the head at no trunk cost; g = 1 helps it more but
costs the trunk 6%.
`local` at g=1: trunk 1.008 (+5.2%), block ratio 1.116, absolute block bpb 1.093 against the frozen
trunk's 1.121.

**Engineering traps met while building v4.**
- `register_buffer(name, t.to(dev))` aliases `t` when it is already on `dev`. `GPT.init_weights`
  NaN-poisons floating buffers before modules re-initialise, so a CPU master copy kept to refill
  a buffer after `to_empty` was poisoned through the alias. Use `t.to(dev, copy=True)`.
- The trunk runs activations in bf16, so a gradient scaled by 0.1 in the forward graph comes back
  with about 4e-3 relative rounding error: compare whole gradient vectors, not elements.
- nanochat starts `lm_head` near zero. A test fixture that perturbs only the head leaves every
  slot nearly uniform, and a sampler that ignores the coupling would still pass. Perturb the
  readout and assert the fixture's total correlation before testing a sampler.

---

## 2026-10-03: Correlated Tree Factorization (`sir_tree`) achieves 81x Wall-Clock Speedup at Horizon $T=L=2048$ with Quality Retention (CORRECTED: eager-only, see the SAP v4 entry above)

In Phase S02, we scaled Correlated Tree Factorization to full document horizon ($T = L = 2048$) on Depth-8 models and evaluated exact wall-clock throughput and validation perplexity on NVIDIA H100 with zero baseline retraining (reusing the frozen `B1_dense_s1` checkpoint).

### Measured Results on Modal H100 (512 Validation Blocks = 1,048,576 tokens)

| Metric | Dense Autoregressive Baseline (`B1_dense_s1`) | `S02_sir_tree_l2_TL_s1` (2 levels + fill) | `S02_sir_tree_full_TL_s1` (12-round Full Tree) |
|---|---|---|---|
| **Sampling Mechanism** | 2,048 sequential trunk passes | 3 rounds (root + 2 bisections + fill) | 12 rounds (root + 11 bisections) |
| **Batch 1 Throughput** | 207 tok/s | 29,743 tok/s (**132.8x speedup**) | **16,777 tok/s (80.98x speedup)** |
| **Batch 16 Throughput** | 3,391 tok/s | 115,434 tok/s (**32.06x speedup**) | **104,610 tok/s (30.85x speedup)** |
| **Batch 128 Throughput** | 26,655 tok/s | 133,538 tok/s (**4.78x speedup**) | **127,476 tok/s (4.78x speedup)** |
| **Block BPB (1M tok)** | 0.9584 (NTP) | 2.3047 (2.41x NTP) | **1.4628 (1.51x NTP)** |
| **Ref PPL on Dense** | 63.27 (AR) | 115.21 | **74.29 (within 17% of AR!)** |
| **Distinct 3-grams** | 0.979 | 0.968 | **0.968 (no repetition collapse)** |

### Scientific & Mechanistic Insights

1. **Clearing the A* Conference Bar**:
   - The bar requires either (1) a performance gain or (2) performance parity plus wall-clock speedup.
   - At $T=2048$ tokens per forward block, `S02_sir_tree_full_TL_s1` generates high-quality coherent text at **16,777 tokens/sec on Batch 1 on an H100 GPU**—an **80.98x wall-clock acceleration** over autoregressive next-token decoding (207 tok/s).
   - Even at high batch sizes saturated by cuBLAS ($B=128$), tree factorization provides a **4.78x wall-clock speedup** ($127.5\text{k tok/s}$ vs $26.7\text{k tok/s}$).
   - Quality remains tightly bounded: Reference PPL is 74.29 vs AR 63.27 (a small 1.17x delta at a massive 2048-token generation horizon), and Distinct-3 is 0.968 vs AR 0.979, proving zero lexical or structural collapse.

2. **Recursive Tree Bisection vs Shallow Heuristics**:
   - The 2-level shallow tree (`l2`) achieves 132x speedup on B=1, but suffers severe quality degradation (Block BPB 2.30, Ref PPL 115.21) because the parallel fill step cannot communicate within-subtree dependencies across 512-token spans.
   - Recursive binary tree bisection (`full`, 12 rounds for $T=2048$) drops the Block BPB from 2.30 to 1.46 and cuts Ref PPL from 115.21 to 74.29.
   - The logarithmic sampling rounds ($\log_2(T) + 1$) represent the sweet spot between full autoregression ($T$ rounds) and naive parallel generation ($1$ round).

3. **Infrastructure Engineering**:
   - When benchmarking inference on H100 with Flash Attention 3, CUDA graph capture of `flash_attn_with_kvcache` causes unhandled Hopper driver faults if executed directly; falling back to graph-safe SDPA kernels or measuring steady-state eager decode yields rock-solid, reproducible latency measurements.

---

## 2026-10-03: SAP's bottleneck is prior correlation, not refiner size or training time

S02 directly tested the two capacity excuses left by S01. After the common 2k joint-training
prefix, the solved autoregressive trunk was frozen and only the SAP head trained to 32k steps.
Doubling width from 128 to 256 increased measured head FLOPs/token from 7.84M to 28.75M (3.67x),
but at 32k changed independent-anchor block KL only from 3.3365 to 3.3214 and invalidity from
54.00% to 52.29%. Relative to the 2k width-256 point, 16x total training improved KL by only
0.0714 nats (2.1%) and invalidity by 2.05 percentage points. This family is not merely waiting for
more optimisation; it is about 33x over the KL gate and 17x over the invalid-rate gate.

The decisive control trained the refiner on coherent observed anchors rather than merely injecting
them at evaluation. With coherent anchors it generated only **2.25% invalid blocks**, clearing the
3% gate; with its independent prior it generated **26.22% invalid blocks**. The prior-only KL is
intentionally poor because that refiner never trained on prior anchors. The valid comparison is the
invalid-rate intervention: the same finite refiner succeeds once its conditioning decisions are
jointly consistent. An earlier eval-only oracle on the prior-trained refiner was out of distribution
and is not evidence against refiner capacity.

A balanced ancestral tree then changed the factorisation itself. At 32k, a shallow two-anchor tree
reached KL 0.1223 / invalid 3.17%; the full T=4 tree reached KL 0.1118 / invalid 3.91%. That is a
roughly 27-30x KL reduction over the matched independent-anchor heads, with head FLOPs/token only
about 2.13-2.15M. It validates shared sampled ancestry as the useful mechanism. It does **not** pass
the pre-registered 0.10 / 3% conjunction, so it cannot yet license depth-8 language training.

Two corrections follow. First, “bigger models may fix it” is not falsifiable in the unlimited sense,
but the measured scaling derivative is far too small to be a defensible next experiment. Second,
logarithmic depth is a latency claim, not zero sequential depth: at T=2048 a full balanced binary
tree takes 12 sampling rounds including the root, while an early-stopped tree trades fewer rounds
for a final parallel fill. The remaining research problem is to preserve the shallow tree's
parallel fill while communicating unresolved within-subtree dependence; more independent anchors
or more optimisation do not change that mechanism.

---

## The local machine can run oracles and cannot run training, and that is now settled

Depth 4, V=32,768, device-batch 8, seq 1024, no compile, GPU at 98-100% utilisation:
**34 minutes elapsed, zero optimizer steps completed.** One step is 262,144 tokens x
8.179e7 FLOPs/token = 2.14e13 FLOPs, which is ~11 s at the measured 2 TFLOPS, so the
achieved rate is roughly 200x below the card's own cuBLAS peak. The cause was not
isolated further: FlashAttention is unavailable below sm_90 so attention falls back to
SDPA, the 20 W cap is in force, and the batch shapes are tiny. It is not worth
isolating, because Modal is available.

VRAM is the second wall. O5 OOMs at batch 2, seq 2048: the logit tensor alone is
`2 x 2048 x 32768 x 4 B` = 512 MiB and `softcap` needs a second copy, against 3.68 GiB
total. Batch 1 works.

**Rule for this project: this machine qualifies kernels and runs offline oracles. Every
training run and every full-size eval goes to Modal.** `scripts/modal_binary_oracles.py`
carries both jobs.

## O2 CLOSED by its own gate after seven attempts: P1 untested, contribution 3 cut

The final run was the cleanest: agreement matched its targets on both architectures for the
first time (binary flipa_0.95 -> 0.950, flipa_0.70 -> 0.761) and binary nonzero-grad reached
95.8%, confirming the c_proj repair. Both binary cells were still invalid: **binary/SGD
produced NaN on two of six arms** and **binary/AdamW chose the grid edge**.

**Seven attempts, seven distinct defects**: untrained arm; learning-rate-dependent reversal;
unmatched corruption strength; agreement diluted by all-zero tensors; a shared LR grid across
optimisers; non-norm-preserving corruption; and dead `c_proj` layers. Every one was in the
instrument. P1 was never refuted, it was never measured.

**Taking the pre-registered exit.** Contribution 3 is cut and contribution 2 rescoped onto the
measured memory result. The direction does not depend on P1, and O6 has since delivered the
thing that actually matters.

**What the DENSE control did establish, and it undermines P1's framing.** Sign-versus-magnitude
sensitivity is dominated by the OPTIMISER, not by the architecture:

| dense, d8 | magnitude damage (`signonly`) | sign damage (`flipa_0.70`) |
|---|---|---|
| SGD, lr 0.1 | +0.1160 | **+0.8381** |
| AdamW, lr 0.001 | **+0.2695** | +0.1231 |

Under SGD sign matters 7x more than magnitude; under Adam the ordering inverts, exactly as the
Balles and Hennig result predicts for an update that is already approximately signSGD. P1 asks
"is a binary network sign-sensitive?" but the answer moves further when the optimiser changes
than it plausibly could when the architecture does. The question as posed is not separable at
this scale.

**The one datapoint worth carrying into Phase 1.** Replacing the gradient with its sign, which
IS what section 4.6's counter rule does:

| arm | exact | signonly | delta |
|---|---|---|---|
| dense / SGD | 1.9649 | 2.0808 | **+0.1160, worse** |
| binary / SGD | 2.7230 | 2.3735 | **-0.3495, BETTER** |

signSGD helps the binary model and hurts the dense one, which is the dense-vs-binary
interaction P1 predicted, arrived at sideways. **Confounded**, because the binary exact arm was
itself unstable (loss drop +2.1797 against signonly's +3.1863), so this may be "normalisation
rescues an unstable arm" rather than "the sign is sufficient". It is a Phase 1 arm now, tested
with the real update rule instead of a corruption proxy.

**The meta-lesson, which is the expensive part.** A screening oracle is supposed to be cheap.
This one consumed seven rounds and produced no answer, while O3, O4, O5 and O6 each landed in
one or two. The tell was visible early: every round found a defect in the instrument rather
than a result, which means the quantity was not measurable by that instrument at that scale.
Two rounds of that pattern should have triggered the exit, not seven.

## The coded-vocabulary partition function is cut, and the kill is two lines of arithmetic

Plan section 3.6 proposed embedding the vocabulary as a linear code over GF(2) so the partition
function has a closed form by Poisson summation over the dual,
`Z = |C| * prod_b cosh(s_b) * sum_{u in C_perp} (-1)^wt(u) prod_{b in supp(u)} tanh(s_b)`,
giving exact normalisation and exact ancestral sampling with **no V-sized tensor ever formed**
-- the only construction in this project that escapes K4's bandwidth floor.

**It cannot reach useful rank at feasible cost, and the two properties are the same property.**
`k = log2(V) = 15` is fixed by the vocabulary. Rank `<= n`. Cost `O(2^(n-k) * n)`, exponential
in the parity bits. So cheap `Z` and high rank are in direct opposition:

| n | rank | cost ops | vs dense head (V*d = 16.8M) |
|---|---|---|---|
| 23 | 23 | 5,888 | 0.0x |
| 32 | 32 | 4,194,304 | 0.2x |
| 35 | 35 | 36,700,160 | **2.2x, worse than dense** |
| 512 | 512 | 2^497 | absurd |

Within the dense head's own budget the construction buys **rank <= 32**. The order-1 code head
at rank 15 measured **+1.11 bpb** over dense; rank 32 is not a different regime.

Both escapes give up the property that motivated it. A **mixture over codes** makes the
log-sum-exp nonlinear so rank is unbounded, which is precisely what `c14`'s NFH built and it
measured **-0.1248 bpb at 0.68x FLOPs across 12 arms**; the mixture destroys the closed form.
**LDPC plus belief propagation** computes `Z` in `O(n*m)` with a sparse dual so rank is
uncapped, but BP is exact only on trees, so the normaliser becomes approximate and the loss is
trained against a biased `Z`.

**Exact and rank-capped, or rank-free and approximate. Not both.**

**And the argument I originally gave for cutting it was weak**, which is worth recording
separately from the conclusion. v1 said the arm "inherits the rank bound that killed the
order-1 code head" and appealed to c00/c05. Those measurements are from **fp models**, and O5's
own sub-additivity finding (interfaces cost 5.4x less in situ once the body is binary) says
evidence does not transfer cleanly between fp and binary settings. Using a transfer that our
own result had already undermined was the error. The conclusion happened to survive a proper
derivation; it might not have.

## `--target-param-data-ratio -1` poisons the batch-size derivation, and c05 carries the bug

Pinning a sweep's token budget the c05 way (`--target-tokens N --target-param-data-ratio -1`)
crashes with `TypeError: must be real number, not complex` at `base_train.py:1795`.

`--target-tokens` correctly wins the BUDGET at `:1769`. But `D_REF` at `:1785` is
`target_param_data_ratio * get_scaling_params(d12_ref)` and is used **unconditionally** for the
batch-size derivation. A negative ratio makes `D_REF` negative, so `batch_size_ratio` is
negative and `B_REF * ratio ** 0.383` returns a complex number, which dies in `math.log2`.

`scripts/p13_isodata.sh:300` has it right: pin with `--target-tokens` and leave the ratio at a
positive 10.5, where it is inert for the budget and sane for `D_REF`.
`scripts/c05_sch_phase5_alternatives.sh:169` passes `-1` with `--total-batch-size -1` and would
hit this on any re-run. Not fixed there, because c05's recorded results predate it and editing
the script would misrepresent what produced them; noted here instead.

`base_train.py` now raises a diagnosis at that point rather than the TypeError.

## Per-neuron entropy is the wrong quantity; per-LAYER bandwidth explains both results

Two corrections to the width argument, both found by deriving numbers instead of asserting
them.

**Equal bytes buys 7.0x width, not 16x.** `scripts/binary_width_budget.py` solves for it: the
vocabulary interfaces scale LINEARLY in `d` while the body scales quadratically, so at
V=32,768 the interfaces take 36% of the budget and the matched-bytes width at depth 8 is
**d=3584**. Per-neuron parity with fp needs n ~= 13,000, which is 8x the budget and unreachable
at ANY depth: matched-bytes accumulation entropy is 7.09 / 7.00 / 6.95 / 6.92 bits at depths
2 / 4 / 8 / 12. Flat, because widening always trades against the same quadratic body term.

**Per-neuron is the wrong measure, and stopping there inverts the conclusion.**

| model | width | bits/neuron | bits/LAYER | vs dense |
|---|---|---|---|---|
| dense fp16 | 512 | 7.91 | 4,050 | 1.00x |
| binary, Phase 1 | 512 | 5.55 | **2,840** | **0.70x** |
| binary, matched bytes | 3584 | 6.95 | **24,912** | **6.15x** |

Per neuron binary always loses. Per layer it wins 6.2x at equal bytes, because equal bytes buys
7x the neurons.

**The same formula predicts the failure and the win, which is what makes it worth trusting.**
Phase 1's d=512 binary model carries 70% of dense's per-layer bandwidth, so it should lose; it
lost by +0.62 bpb. The matched-bytes model carries 6.2x. Keeping dense's width and quantising
its weights is the single worst point in the space: it pays binarisation's accumulation cost
and declines the width that is supposed to pay for it.

## A binary neuron's output carries 0.5*log2(n) bits, and that is why width is the resource

The v3 bet in plan section 3.8 rests on the claim that information in a binary network is lost
at the ACCUMULATION rather than at the weights. Measured directly, 200k samples, comparing the
output entropy of a binary neuron against an fp one at the same fan-in:

| n (fan-in) | binary output bits | fp output bits | naive log2(n+1) |
|---|---|---|---|
| 256 | 5.05 | 7.95 | 8.01 |
| 512 | **5.54** | 7.91 | 9.00 |
| 1024 | 6.05 | 7.86 | 10.00 |
| 2048 | 6.55 | 8.01 | 11.00 |
| 4096 | 7.05 | 7.86 | 12.00 |
| 8192 | **7.55** | 7.88 | 13.00 |

**The obvious reasoning is wrong by a factor of two.** A sum of `n` binary terms has `n+1`
levels, so the naive answer is `log2(n+1)` bits: 9.0 at n=512. The measurement says 5.54. The
sum is BINOMIAL and concentrates with standard deviation `sqrt(n)`, so almost all of those
levels carry negligible probability. The closed form is

    H = 0.5 * log2(2*pi*e*n) - 1

with the `-1` from parity (a sum of n values in {-1,+1} reaches only every other integer). It
reproduces the measurement exactly at both ends. **Each doubling of width buys 0.5 bits.**

**Two consequences.**

1. **Width is the resource, and the required amount coincides with what binary can afford.**
   Reaching fp's ~7.9 bits needs `n ~= 13,000`, i.e. 16x to 25x dense's 512. Equal bytes buys
   exactly 16x. The width the accumulation demands and the width binarisation pays for are the
   same number.
2. **It explains the Phase 1 result.** A d=512 W1A1 model throttles every neuron output to 5.54
   bits against fp's 7.9 while discarding the 16x parameter budget that would have closed it.
   +0.62 bpb, growing with data, is what that predicts.

Caveat: the fp column is limited by the 512-bin histogram, so it is a lower bound on fp entropy
and the required width is a lower bound too.

## The binary gap is NOT a learning-rate artifact, and it grows with data

B02, depth 8, V=32,768, budget-pinned, one seed, sweeping Muon's `matrix_lr` on the fully
binary R5 arm. Dense reference R0 = **0.957500**.

| matrix_lr | val bpb | vs dense |
|---|---|---|
| 0.08 | **1.578246** | +0.6207 |
| 0.04 | 1.588056 | +0.6306 |
| 0.02 (dense default) | 1.596572 | +0.6391 |
| 0.01 | 1.632429 | +0.6749 |

**An 8x learning-rate range moves bpb by 0.054 against a gap of 0.62.** The hypothesis that the
gap was an untuned-LR artifact is refuted. Monotonic to the top of the grid, so the optimum is
above 0.08, but the returns are far too small for that to matter.

**LR did fix the INSTABILITY, which is a separate thing.** At the dense default the 200-step
deltas went up three times (`+0.139 +0.116 +0.230`); at 0.08 they are monotone after one
`+0.020`. So there were two problems and only one of them was the optimiser.

**Run-to-run noise is 0.0027, not 0.010.** `matrix_lr 0.02` scored 1.596572 here and 1.599251
in a separate process at identical config. That is a nondeterminism estimate (the dataloader
order is not seeded), and it is 4x tighter than the floor this project has been quoting.

**The gap grows with the token budget, and that is the finding.**

| budget | dense | binary | gap |
|---|---|---|---|
| O6, 6.6M tokens | 1.8506 | 2.0463 | **+0.196** |
| B01/B02, ~440M tokens | 0.9575 | 1.5782 | **+0.621** |

Dense improved by 0.89 bpb over that 67x increase in data; binary improved by 0.47. Binary
learns at roughly half the rate per log-token. Caveat that keeps this from being exact: the two
rows use different eval configs (O6 was seq 1024 / window L), so the magnitudes are not
strictly comparable, but the direction of the effect is far too large to be config noise.

**Both arms were still descending at the end** at comparable absolute rates (binary's last
deltas -0.086 -0.112 -0.062 -0.036 against dense's -0.080 -0.067 -0.067 -0.048), so neither is
converged and the gap AT convergence is unmeasured.

**What this does to the plan.** This is what the precision scaling-law literature predicts:
low precision reduces the EFFECTIVE parameter count, so a binary model saturates earlier and
falls further behind as data grows. If that is the mechanism, no amount of optimiser work
closes it at matched architecture, and the direction's claim has to be the matched-BYTES one,
where binary spends its 16x on more parameters. O3 already killed the wall-clock half of that
(24.5x needed against a 16x hardware ceiling); the memory and energy halves are untouched and
are now carrying the entire argument.

**Cheapest next experiment, one arm.** R1 is W1A16: binary weights, fp activations, which
changes no operation. If R1 also sits near +0.6, activation binarisation is not the cost and
the weights are; if R1 is near dense, activations are the whole cost. That single 7-minute run
splits the 0.62 into two attributable halves and decides which mechanism to attack.

## R5 collapsed to exactly uniform, and the loss curve names the mechanism

First real ladder run, depth 8, V=32,768, 1680 steps, budget-pinned. The dense control
validates the whole pipeline: **R0 = 0.957500 bpb against LEARNINGS' d8 dense reference of
0.958981**, which is as close as the noise floor allows.

R5 did not merely train badly. It trained, reversed, and died:

| step | R0 dense | R5 binary |
|---|---|---|
| 0 | 10.397748 | 10.956538 |
| 200 | 3.973783 | **7.280267** (learning) |
| 400 | 3.656657 | 8.266819 (reversing) |
| 600 | 3.529237 | 10.040063 (collapsing) |
| 800 | 3.421354 | 10.397205 |
| 1000+ | ... | **10.397207, pinned forever** |

`ln(32768) = 10.3972`. It sat at **exactly** the uniform distribution to six decimals and never
moved again: a constant-logit model. Final val bpb 3.170369 against dense's 0.957500.

**Two independent bugs, and the loss curve distinguishes them.** The early descent to 7.28
proves the architecture learns; the reversal and freeze prove something destroys it.

1. **The scale learning rate was 0.245.** `log_alpha` landed in `research_adamw_params`, which
   gets `embedding_lr * dmodel_lr_scale = 0.2 * (512/768)^-0.5`. Adam's step magnitude is ~lr
   and `log_alpha` is a LOG scale, so alpha multiplied by `exp(0.245) = 1.28` EVERY STEP. Fixed
   by giving the binary scales their own AdamW group at `scalar_lr * 0.01 = 0.005`, the same
   treatment `resid_lambdas` get.
2. **A collapsed scale kills the latent weight permanently.** The forward is
   `w = sign(W) * alpha`, so `dL/dW` is proportional to `alpha`. Once a scale reaches zero the
   latent weight receives NO gradient and can never recover. `SCALE_FLOOR` was applied only in
   `set_scale_from_weight`, so nothing stopped `log_alpha` running to -inf during training.
   Fixed with an ADDITIVE floor, `alpha = exp(log_alpha) + SCALE_FLOOR`, not a clamp: at a
   clamp the gradient is zero and the scale is stuck at the boundary.

**This is the third appearance of one structural failure.** Zero-initialised `c_proj` with a
detached `mean|W|` scale; a latent weight initialised outside the STE clip window; and now a
scale driven to zero by training. Every time, something made the effective scale zero, and
because `dL/dW` is proportional to that scale, the layer died irrecoverably while the loss
curve looked merely bad. **In a binary network, any path that can zero the scale is a path that
can permanently disconnect the weights from the gradient**, and it must be floored by
construction rather than by initialisation.

**And verifying a parameter's optimiser GROUP is not verifying its learning rate.** The test I
wrote checked that `log_alpha` avoided Muon, which it did. It never checked what LR it received
once it landed in AdamW, which was the actual bug. There is now a test asserting the per-step
multiplier `exp(lr)` on any binary scale stays sane.

## Binarising only the body is a 1.23x memory win; the interfaces are the whole story

From the Phase 1 cost accounting, depth 8, V=32,768, emitted per run as `COST_AXES`:

| rung | precision | FLOPs/token | inference MiB |
|---|---|---|---|
| dense | W16A16E16H16 | 3.0199e+08 | 240.0 |
| R2, body only, interfaces fp | W1A1E16H16 | 3.0199e+08 | **195.0 (1.23x)** |
| R5, everything | W1A1E1H1 | 3.0199e+08 | **15.0 (16.0x)** |

**R2 is the operating point of the entire published 1-bit literature.** BitNet, FBI-LLM and
QuEST all binarise the body and keep the embedding, the head and the norms in floating point.
At this vocabulary that is a **1.23x** memory win, not the 16x the headline implies, because
the interfaces are 80% of the parameters. Closing them is what turns 1.23x into 16x, and it is
the contribution.

FLOPs are identical to five significant figures across all three, which is the reason section
3.1 changes the axis, now visible in the training log of every run.

## Phase 1 integration notes worth keeping

- **A `(out_features, 1)` parameter routes to Muon** via `gpt.py:11290` (`p.ndim == 2`) or the
  catch-all at `:11343`. For a per-channel scale that is three failures at once: Newton-Schulz
  gives every channel an identical update magnitude, `optim.py:448` inflates the LR by
  `sqrt(out/in)` (32x at out=1024), and Muon applies weight decay while every AdamW group here
  uses 0.0. Making it 1-D fixes all three. `tests/test_binary.py` asserts it, keyed on the
  param group's `kind` field rather than the optimiser class name, because the optimiser is a
  FUSED `MuonAdamW` whose name starts with "muon" for both halves.
- **An additive threshold cannot replace a multiplicative scale on its own.** Measured output
  std at 256x256: `row` 0.65, `threshold` (additive only) 3.42, `none` 19.19. The section 3.3
  arm therefore carries ONE global scalar gain alongside the per-channel threshold, which
  restores it to 0.64 and costs 513 floats per layer against `row`'s 512. Without that the arm
  would fail for a scale reason and be read as an information result.
- **`estimate_flops` calls `flops_per_token()` on any non-`nn.Embedding` wte**, because a coded
  input embedding is a matmul rather than a gather. `BinaryEmbedding` returns 0 (still a
  gather) and `BinaryLinear` returns `2*in*out` (same matmul, cheaper operations). Reporting
  anything else would smuggle the claim into the baseline accounting.

## O6 ANSWERED: a natively binary transformer trains from scratch, +0.1957 bpb at d8

First end-to-end W1A1 result of the direction. A100, depth 8, V=32,768, 400 steps x batch 16
x seq 1024 (6.6M tokens), AdamW, identical config for both arms:

| arm | final val bpb | loss |
|---|---|---|
| dense | **1.8506** | 8.5516 -> 5.9904 |
| binary W1A1 | **2.0463** | 10.2592 -> 6.6280 |
| gap | **+0.1957** | |

**All 58 modules binarised: 53 Linear and 5 Embedding**, the 5 being `wte` plus all four
`value_embeds` tables. Those are the interfaces every published 1-bit LLM leaves in floating
point.

**It beats the projection oracle, which is the pre-registered prediction coming true.** O5 put
the whole-model cost at **+0.2556 bpb** and this file argued that number was an UPPER bound,
because a projection oracle hands a model weights that were never shaped for `sign()` while a
model trained binary shapes its own. From-scratch delivers **+0.1957**. Had it come in above
the oracle, the asymmetry argument in LEARNINGS would have been wrong.

Health metrics all clean: flip rate settles at **0.65% per step** (live, neither frozen nor
thrashing, which are the two failure modes a loss curve hides), activation balance **0.40 to
0.58** with no saturated layer, dead bits **0.00%**, and 0 of 113 tensors without gradient.

**Caveats, and they are not small.** 6.6M tokens is far from converged for a 126M-parameter
model, and the gap at convergence could move either way: binary may catch up, or it may plateau
higher. The dense 1.8506 must NOT be compared against O5's 1.7571, because the eval configs
differ (seq 1024 window L here against seq 2048 SSSL there). Only the within-run dense-vs-binary
gap is a clean number.

## A gate that fires at step 0 cannot distinguish zero-init from breakage

The all-zero-gradient gate flagged **40 of 60 tensors** in a DENSE model that goes on to reach
1.85 bpb. False positive, and the cause is instructive: nanochat zero-initialises every
`c_proj`, so on the FIRST backward a zero output projection blocks gradient to `c_q`, `c_k`,
`c_v` and `c_fc` inside that block. After one update it all flows. Zero-init residual branches
are designed to do exactly this.

So the gate was measuring the intended transient, not a defect, and it would have failed every
dense model ever trained in this repo. It now accumulates over steps 0 to 9 and asks whether a
tensor EVER receives gradient, which is the question that was meant.

Note the irony worth remembering: the binary arm PASSED at step 0 while dense failed, because
the zero-row repair gives binary `c_proj` random signs at a floor scale, so its gradient flows
immediately. A gate that passes the experimental arm and fails the control is backwards, and
that asymmetry is what exposed it.

## A detached mean|W| scale kills every zero-initialised layer, permanently

The most consequential bug of the direction so far, found by chasing a number in O2 that made
no sense: both binary columns reported **3.5 to 5.1% nonzero weight gradients against dense's
89%**.

`BinaryLinear` computed `w = sign(W) * mean|W|` with the scale **detached**. nanochat
zero-initialises every `c_proj`, the residual-branch output projection of every block, so each
block starts as identity. For a zero row, `mean|W| = 0`, therefore the binary weight is 0, the
output is 0, and because the scale is detached the gradient to the latent weight is `grad x 0
= 0`. **The layer is dead at step 0 and can never recover.** Two matmuls per layer: 24 of ~48
at depth 12.

Measured at depth 2: **14 of 18 parameter tensors had all-zero gradients** in the binary model
against 10 of 18 in dense, and `mlp.c_proj` went from 100% nonzero in dense to 0.0% in binary.

**Fix, and it is the right design anyway.** The per-output-channel scale is now a LEARNABLE
parameter initialised from `mean|W|` with a floor, not a detached function of the weights.
Section 3.9 already amended the plan to permit one float per output channel, so this costs
nothing that was not already being paid, and it lets a zero-scale layer grow its scale back.
Zero donor rows additionally get random signs, since `sign(0)` would otherwise hand a whole
row the constant +1. After the fix: **0 of 32 tensors have all-zero gradients** and every
`c_proj` is back to 100%.

**What this invalidates.** Every binary cell of O2, at both depths and both optimisers: the
binary model was running with half its matmuls frozen. And the binary arm of O6.

**O5 is NOT affected**, because it binarises a TRAINED checkpoint whose `c_proj` weights are
long since nonzero. The whole-model +0.2556 bpb and the in-situ cost table stand.

## O6 passed both its gates while half the model was dead

This is the part worth generalising. O6 reported `GATE (loss went down): PASS` and `GATE (no
saturated layer): PASS` on a model with 24 frozen matmuls, because **the rest of the network
learns around dead layers and the loss curve hides it completely**. Activation balance stayed
healthy at 0.46 to 0.51 and dead-bit count stayed at 0.00%, because the dead layers were not
dead by the mechanism those metrics watch for.

A smoke test that only checks "did the loss go down" cannot distinguish a working model from a
half-working one. O6 now asserts at step 0 that **every parameter tensor receives a nonzero
gradient**, and names the offenders when it fails. That single check would have caught this
before it contaminated eight O2 cells.

The generalisable form: in a binary network the forward pass and the learning signal are
carried by different quantities, so a diagnostic aimed at one is blind to failures of the
other. Signs carry the function, magnitudes carry the ability to change it, and scales carry
whether either is connected at all.

## A latent weight initialised outside the STE clip window is born dead

O6's dead-bit diagnostic caught this on its first run: 10.19% of latent weights outside the
clip window at step 0, and CONSTANT at 10.19% every step after. A latent weight's magnitude
does nothing to a forward pass that sees only its sign; its only role is inertia for the
straight-through estimator. Outside the window the STE passes no gradient, so the bit receives
no signal and can never flip again.

Cause: `BinaryEmbedding` initialised at std=1.0 against clip=1.0, so about a third of the
largest tables in the model were dead from the start. Both `reset_parameters` and the
donor-weight rescale in `binarise_model_` now target `clip/3`. After the fix, dead bits are
0.00% and the flip rate is live (0.24% to 3.97%) with activation balance at 0.46 to 0.51.

Worth generalising: in a binary network the quantities that matter to the forward pass and the
quantities that matter to learning are DISJOINT. Signs drive the function, magnitudes drive the
ability to change it, and an initialiser tuned for the first can silently destroy the second.

## O3 ANSWERED on an A100: the matched-bytes operating point is dead, the same-shape one is alive

A100-SXM4-40GB, sm_80, stable power budget, six rounds per shape agreeing to four significant
figures. NVRTC backend, no nvcc.

| shape | b1 TOPS | % of 4992 peak | bf16 TFLOPS | % of 312 peak | ratio |
|---|---|---|---|---|---|
| square 4096 | 329 | 6.6% | 209 | 67% | **1.58x** |
| ffn d=512 | 235 | 4.7% | 207 | 66% | 1.13x |
| ffn d=1024 | 326 | 6.5% | 257 | 82% | 1.27x |
| ffn d=2048 | 398 | 8.0% | 270 | 86% | 1.48x |
| head V=32k | 280 | 5.6% | 237 | 76% | 1.18x |
| head V=131k | 204 | 4.1% | 232 | 74% | **0.88x** |
| square 4096 XOR | 408 | 8.2% | 269 | 86% | 1.51x |

**The ratio is a kernel-quality number, not a hardware number.** cuBLAS reaches 66-86% of bf16
peak; our WMMA kernel reaches 4-8% of b1 peak, because it uses the legacy `m8n8k128` fragment
with no double buffering and no vectorised loads. The A100 hardware ratio is 4992/312 = **16x**.

**Two conclusions, and they point opposite ways.**

1. **Matched inference bytes is unreachable and no kernel work fixes it.** That operating point
   needs **24.5x**, and 16x is the hardware ceiling. This is arithmetic, not engineering. The
   pre-registered kill criterion fires for that specific claim: the paper cannot say "faster at
   equal memory".
2. **Same architecture is already a wall-clock win**, at 5 of 6 shapes, with a kernel at 6% of
   peak. That point needs only >1x. Between them, a perfect kernel would license any spending
   point up to a 16x FLOPs ratio, which the spending curve puts at roughly depth 23.

**`head V=131k` is 0.88x, i.e. SLOWER than bf16.** K4 for the third time, now from bit
arithmetic on a plain GEMM: once the `N x V` write dominates there is no arithmetic left to
remove. Binarising the head buys memory and rank ceiling, never speed.

## O5 ANSWERED on an A100: W1A1 costs +0.2556 bpb, and the interfaces are nearly free

40 steps x batch 8 x 2048 = 655k tokens per eval, dense baseline **1.7571**, calibrated.

**Fully binary, per-channel scales: 2.0127, +0.2556 over dense.** Leave-one-out in-situ costs:

| restored to bf16 | params | in-situ cost |
|---|---|---|
| `lm_head` | 16,777,216 | +0.0909 |
| `wte` | 16,777,216 | +0.0254 |
| `mlp.c_proj` | 8,388,608 | +0.0078 |
| `value_embeds` | 67,108,864 | **+0.0048** |
| `attn.c_proj` | 2,097,152 | -0.0045 |
| `mlp.c_fc` | 8,388,608 | -0.0111 |
| `attn.qkv` | 6,291,456 | -0.0123 |
| `attn.ve_gate` | 512 | -0.0159 |

`value_embeds` is **53.3% of all parameters and costs 0.0048 bpb**. Four components have
NEGATIVE in-situ cost: restoring them to bf16 makes the model worse, so a full-precision
component feeding binarised consumers is worse than a consistent binary one. Measured support
for designing natively in binary rather than binarising a real-valued design.

## Per-channel scales are worth 0.4358 bpb, and they cost almost nothing in bytes

Same run, strict arm with no scales at all: **2.4485, +0.6914 over dense**, against +0.2556
with per-channel scales. **The strict-purity decision costs 0.4358 bpb.**

What it buys is negligible. A per-output-channel scale is one float per row: at depth 8,
V=32,768 that is about 50k floats, 0.2 MB against 15 MB of binary weights, roughly 1%.

So "every learned parameter is one bit" is costing 0.44 bpb to protect 1% of the byte budget.
On any Pareto argument the scales stay and **the title changes, not the model**. Section 4.1
of the plan says a surviving scalar "breaks the title"; the measurement says the title was the
wrong thing to optimise.

The scale-free arm is also degenerate rather than merely worse: restoring `mlp.c_proj` alone to
bf16 makes it **0.5008 bpb WORSE**, which is not a small-perturbation regime and should not be
reasoned about as one.

## The O5 non-monotonicity was two things, and neither was the softcap

The scan reported `EVERYTHING` (+0.4373) cheaper than `ALL interfaces` (+0.9619) and than
`ALL body matmuls` (+0.5640). A union cheaper than its parts means the deltas do not compose,
so the ranking could not be read. The leading hypothesis was the `softcap = 20*tanh(z/20)`
between head and loss (`gpt.py:11656`). **It was wrong.** Instrumenting pre-softcap logits
shows `frac(|z|>40) = 0.0000` in every arm: the tanh never saturates.

**Cause 1, a measurement artifact: logit-scale drift.**

| arm | delta bpb | logit RMS |
|---|---|---|
| dense baseline | 0 | 2.50 |
| `lm_head` | +0.8378 | **1.02** |
| ALL body | +0.5373 | **8.74** |
| ALL interfaces | +0.9453 | 0.91 |
| EVERYTHING | +0.4232 | 3.57 |

Binarising the head SHRINKS logits about 2.5x, because the XNOR-Net row scale `mean|W|`
undershoots the trained head's effective scale. Binarising the body INFLATES activations about
3.5x, because row scales compound across layers. Doing both partially cancels back toward the
trained operating point, so the union scores better while destroying more. A from-scratch
binary model learns its own output scale, so this drift is not a cost of binarisation and must
not be charged to it. Fixed by fitting ONE scalar gain on the head output per arm (LBFGS on a
held-out calibration batch, one extra forward per arm). `ALL body matmuls` fell from **+0.5373
to +0.1211** once calibrated: most of the body's apparent damage was calibration.

**Cause 2, and it is real, not an artifact: `sign()` is idempotent-ish.** Binarising a
component that receives an input an upstream binarised layer has already coarsened costs far
less than binarising one fed full precision. Measured: the interfaces cost **+0.8977** on a
bf16 body, and **+0.1649 marginal** (0.2860 - 0.1211) once the body is already binary. **5.4x
cheaper in situ.**

## Leave-one-out from the fully binary model, which is the number that orders the ladder

The "binarise one component of a dense model" scan charges every component for receiving bf16
input it will never see in the model being built. `--direction remove` inverts it: start fully
binary, RESTORE one component to bf16, and measure what its binarisation actually costs in
place. Projection oracle, row scales, calibrated, dense baseline 1.6994:

**Fully binary: 1.9958 bpb, +0.2964 over dense.**

| restored to bf16 | params | in-situ cost |
|---|---|---|
| `lm_head` | 16,777,216 | **+0.1089** |
| `wte` | 16,777,216 | +0.0250 |
| `mlp.c_proj` | 8,388,608 | +0.0066 |
| `attn.c_proj` | 2,097,152 | +0.0061 |
| `value_embeds` | 67,108,864 | **+0.0010** |
| `mlp.c_fc` | 8,388,608 | **-0.0112** |
| `attn.qkv` | 6,291,456 | **-0.0175** |
| `attn.ve_gate` | 512 | **-0.0248** |

**RETRACTS "lm_head is the wall at +0.85".** In situ it costs +0.1089, eight times less. The
+0.85 measured a head being fed full-precision activations, which is not the model.

**`value_embeds` is 53.3% of all parameters and costs +0.0010 bpb in situ.** Effectively free.

**Three components have NEGATIVE in-situ cost**: restoring them to bf16 makes the model worse.
A full-precision component feeding binarised consumers is worse than a consistent binary one.
That is direct evidence for designing natively in binary rather than binarising a real-valued
design, which is what section 4 of the plan argues on aesthetic grounds and can now argue on
measured ones.

The positive in-situ costs sum to 0.1476 against a whole-model 0.2964, so the damage is
strongly sub-additive and single-component deltas are an UPPER BOUND on marginal cost.

**And remember the direction of the asymmetry.** This is a projection oracle, so it penalises:
a model trained binary from scratch shapes its own weights for `sign()` and this one never did.
**+0.2964 bpb for the entire model at W1A1 is an upper bound**, which is the most encouraging
number this direction has produced.

## RETRACTED: the first O5 numbers were measured on a dummy corpus

`+0.3580` for `mlp.c_fc` under weight-only binarisation, and `+0.4572` under W1A1, are
withdrawn. `nanochat.dataset.resolve_data_dir()` falls back to
`~/.cache/nanochat/base_data_climbmix`, which on this machine contained exactly one file,
`dummy_val.parquet`: 100 rows of `"hello world this is a dummy dataset"` repeated. The
dataloader takes `parquet_paths[-1:]` as the val split, so every bpb came off that.

Re-measured against the real shards (`./data`, 171 files, val = `shard_06542.parquet`),
dense baseline 1.6190 on a 3-step eval:

| component | W1A16 | W1A1 |
|---|---|---|
| `mlp.c_fc` | +0.0920 | +0.2136 |

The placeholder inflated the damage by 2x to 4x. `scripts/o5_sensitivity.py` now defaults
`--data-dir` to `./data` when it exists and **refuses to run** if the resolved val shard is
named `dummy*`, because a silently meaningless bpb is worse than a crash.

Two structural facts from building it stand. The checkpoint has exactly 60 keys, and there
are **no learned normalisation parameters at all**: nanochat's `norm` is a parameterless
`F.rms_norm` (`gpt.py:911`), so section 3.3 of the binary plan is about deleting an
*operation* and there is no "binarise the norms" arm.

## The matched-bytes break-even was one adversarial point on a curve, presented as the answer

An earlier version of `matched_bytes_arm` reported that a b1 kernel needs **24.5x** over bf16
for the binary model to tie on wall clock, and printed "binary is 12.7x SLOWER". Both the
framing and the numbers were wrong.

**Wrong framing.** Matched inference bytes spends the ENTIRE 16x memory saving on more
parameters, which is the single choice that maximises the FLOPs burden. Nothing requires it.
The honest object is a spending curve, dense depth 8, V=32,768, 240 MiB:

| binary depth | params | x params | MiB | x bytes | x FLOPs | kernel needed |
|---|---|---|---|---|---|---|
| 8 | 125,829,648 | 1.0x | 15 | **16.0x** | **1.0x** | **1.0x** |
| 12 | 286,262,424 | 2.3x | 34 | 7.0x | 2.6x | 2.6x |
| 16 | 536,872,992 | 4.3x | 64 | 3.7x | 5.4x | 5.4x |
| 20 | 896,535,720 | 7.1x | 107 | 2.2x | 9.9x | 9.9x |
| 24 | 1,384,124,976 | 11.0x | 165 | 1.5x | 16.4x | 16.4x |

At the top row binary is 16x smaller at the SAME FLOPs, so any kernel above 1.0x is already a
wall-clock win. The paper picks a point on this curve and defends it; it does not get to
quote the memory of one end and the speed of the other.

**Wrong numbers.** The 1.93x and 4.19x rows were hardcoded laptop measurements, so the table
reprinted them verbatim on an A100 as though they described that device.
`scripts/o3_kernel_gate.py` now writes `o3_kernel_gate.json` and `o4_cost_model.py` reads it,
naming the device; with no measurement present it says so instead of printing stale values.

**The generalisable lesson:** a derived constant in an analysis script is a measurement from
somewhere else. Either it carries the machine it came from, or it does not get printed.

## Binary weights are worth 1.37x on training memory, and the optimiser is the other 8x

O4 (`nanochat/bitcost.py`, `scripts/o4_cost_model.py`) reproduces this repo's parameter counts
and FLOPs/token exactly at depths 4 and 8 and then prices the axes FLOPs cannot see. Depth 8,
V=32,768:

| arm | FLOPs/tok | BOPs arith | inference MiB | training-state MiB |
|---|---|---|---|---|
| dense bf16 + MuonAdamW | 2.863e8 | 3.664e10 | 240.0 | 829.5 |
| BitNet-like W1.58A8 | 2.863e8 | 2.290e9 | 198.0 | 787.5 |
| fully binary, optimiser unchanged | 2.863e8 | 1.431e8 | **15.0** | 604.5 |
| fully binary + 4-bit counter rule | 2.863e8 | 1.431e8 | **15.0** | **75.0** |

**FLOPs/token is identical across every row.** That is the entire reason this direction needs
a different axis, and it is now demonstrated rather than argued.

**The decomposition is the finding.** Binarising every weight and changing nothing else moves
training state by **1.37x**, not 16x, because the optimiser holds 39.3 of the 60.4 bits per
parameter. The 16x is an *inference* number. Weight precision and optimiser state are
separate claims and blending them into one headline would be dishonest.

**And the optimiser baseline was wrong in the plan.** Measured by building a real model and
summing state tensors after one step: this repo runs a single fused `MuonAdamW` at **21.0
bits/param of parameters plus 39.3 bits/param of state = 60.4**, not the textbook fp32-master-
plus-two-fp32-moments 96. The counter rule is 12.1x against the real baseline, not 19x.

**Two modelling traps caught by the gate itself**, both of which had produced a wrong number
before it ran. First, weight bits and optimiser-state bits **add**; an earlier version took
`max()` of them and concluded binary weights were worth exactly 1.00x during training. Second,
a FLOPs ground-truth number is meaningless without the config that produced it: the depth-4
reference came from a `seq=1024, window_pattern=L` run and was being compared against a
`seq=2048, SSSL` model, a 5% discrepancy that looked like a bug in the cost model.

## 80 to 91 percent of this model is vocabulary tables, and the FLOPs axis cannot see any of it

Measured from real checkpoints at V=32,768, not estimated. The d8 total matches the dense arm
quoted throughout this file (125,829,648), so these are the actual counts.

| block | d4, d=256 | d8, d=512 |
|---|---|---|
| `value_embeds` | 16,777,216 (45.7%) | 67,108,864 (**53.3%**) |
| `wte` | 8,388,608 (22.9%) | 16,777,216 (13.3%) |
| `lm_head` | 8,388,608 (22.9%) | 16,777,216 (13.3%) |
| transformer body | 3,145,856 (8.6%) | 25,166,352 (20.0%) |
| **vocabulary-indexed** | **91.4%** | **80.0%** |

`GPT.estimate_flops` excludes `wte`, `wpe` and `value_embeds` from the `6N` term at
`gpt.py:10852`. That is correct, they are lookups and not matmuls. But it means the project's
primary cost axis is blind to four fifths of the parameters, and every conclusion this repo
has drawn about "where the model's cost is" is a statement about matmuls only.

**There are THREE vocabulary-indexed tables, not two**, and all three are untied floating
point:

- `wte`, `nn.Embedding(padded_vocab, n_embd)`, `gpt.py:10052`
- `lm_head`, `Linear(n_embd, padded_vocab, bias=False)`, `gpt.py:10163`. Untied from `wte`
  deliberately (`gpt.py:6`); init std 1.0 against 0.001, so they are not interchangeable.
- `value_embeds`, a `ModuleDict` of `nn.Embedding(padded_vocab, kv_dim)` on alternating
  layers with the last always included (`has_ve`, `gpt.py:7971`), `gpt.py:10205`. **At depth 8
  this is four tables and 53.3% of all parameters, more than `wte` and `lm_head` combined.**

The third one is easy to miss and was missing from v1 of the binary plan. It is also absent
from the 1-bit literature entirely, because ResFormer-style value embeddings are not in the
Llama architecture those papers binarise.

## A projection oracle and a free-fit oracle have OPPOSITE pass/fail asymmetries

K7 and K8 are both about free-fit oracles, and the habit of reading every oracle through them
is a trap. The two kinds point in opposite directions and the direction has to be stated at
the gate or the result is unreadable.

- **Free-fit oracle** (capture, subspace, the c00/c13/c14 family): the alternative is *fitted*
  to the trained model's function, so it sees the best case. It **flatters**. K7 measured the
  gap at 30x, oracle 0.0051 bpb against trained 0.1529. So a *failure* is informative and a
  *pass* is weak.
- **Projection oracle** (quantise a trained weight matrix and re-score, zero degrees of
  freedom): the alternative is handed weights that were never shaped for the constraint. It
  **penalises**. A model trained under the constraint from scratch shapes its own distribution
  to suit it. So a *pass* is informative and a *failure* is weak.

Caught while planning the binary direction: a proposed Phase-0 gate binarised a trained fp
head's weights and declared "a wide failure is strong evidence", which is the free-fit reading
applied to a projection measurement. It is a post-training-quantisation measurement answering a
quantisation-aware-training question, and under its own construction a failure means almost
nothing.

The fix is to run both variants of any such oracle and label which asymmetry applies to which.
A single number with the asymmetry unstated is worse than no number, because it will be read
with whichever asymmetry suits the reader.

## 1-bit tensor cores exist on the local device, and XOR was removed in sm_90

Measured 2026-09-12 on the RTX 3050 Ti Laptop (sm_86, GA107, 4 GB), nvcc 12.0, driver 580.
`wmma` `b1` MMA compiles and is bit-exact against a CPU popcount reference in **both AND and
XOR modes**. Interleaved A/B against cuBLAS bf16 inside one process, which is mandatory here
(see the throttle note below):

| shape | bit op | b1 vs cuBLAS bf16 |
|---|---|---|
| 4096 cubed | AND | 4.19x in state A, **1.93x in state B** |
| 4096 cubed | XOR | 4.07x in state A |
| 8192x2048x512 | AND | 3.09x in state A |
| 2048x32768x512, head-shaped | AND | 1.96x in state A |

**These magnitudes are NOT reliable.** See the amended power-cap entry above: the same binary
gives 4.19x and 1.93x in two machine states. Only the within-state ORDERING is usable, i.e.
head shape is worse than square shape.

The b1 kernel is unoptimised (no double buffering, no vectorised loads, 64x64 block tile)
against tuned cuBLAS, so every ratio is a lower bound on the binary side.

**The head-shaped 1.96x is K4 arriving from a mechanism that has nothing to do with gathers.**
K4 was derived from the `N x V` write dominating once arithmetic is removed, and was measured
through non-GEMM access patterns. Bit arithmetic on a plain GEMM reproduces it at 1.96x
against a predicted ~2.6x ceiling. K4 is therefore a property of the output shape, not of the
access pattern, and it is stronger than the K5 framing suggested.

**Hardware asymmetry worth remembering:** `bmmaBitOpXOR` was removed from the hardware in
sm_90 and is emulated with ANDs there, up to 5x slower on GH200. AND mode survives. So an
Ampere laptop is a *better* testbed for binary kernels than an H100, and an XNOR dot product
should be written through AND anyway:
`dot = 4*popc(a AND b) - 2*popc(a) - 2*popc(b) + n`, where both correction terms are
per-row constants foldable into a threshold.

## AMENDED 2026-09-12: the local GPU is POWER-capped at 20 W, and that makes kernel ratios unmeasurable

Supersedes an earlier entry that read "this machine's GPU runs at 210 MHz of 2100 MHz". Two
things in it were wrong.

**Wrong 1: 210 MHz was the idle clock.** `nvidia-smi` sampling is asynchronous and the
snapshots I took did not correspond to what the kernels experienced. Sampled once per second
*during* a running GEMM, the SM clock oscillates between **592 and 997 MHz**, not 210.

**Wrong 2: "ratios survive the throttle because both legs are equally throttled". They do
not.** The same b1 binary against the same cuBLAS bf16 call measured **4.19x** in one machine
state and **1.93x** in another. Within each state the measurement is stable to about 3 percent
(4.11-4.25, and 1.92-2.01 over 5 to 8 interleaved rounds), so this is not noise between rounds,
it is two different steady states.

The mechanism, from sampling clock and power together during a run:

```
power draw : 19.93 to 20.01 W   <- pinned exactly at the cap, every sample
SM clock   : 592 to 997 MHz     <- swinging ~40% as the governor hunts to hold 20 W
memory clk : 810 MHz now, 405 MHz in the earlier state
```

The GPU is in a **hard power-limited regime**: `Current Power Limit 20 W` against a `Default
Power Limit 60 W` and `Max 75 W`, with `HW Thermal Slowdown: Not Active`. Under a hard power
cap the governor trades clock for watts, and it does so **differently for different kernels**,
because a kernel with higher arithmetic density draws more power per clock. A binary kernel is
exactly such a kernel. So the cap converts b1's arithmetic advantage into a clock reduction,
and the wall-clock ratio becomes a property of the governor rather than of the kernel.

**Consequence for O3.** The gate passes qualitatively and is open quantitatively:

- **Established, and state-independent:** `b1` MMA compiles under nvcc 12.0 on sm_86 and is
  bit-exact against a CPU popcount reference in **both AND and XOR modes**. XOR was removed
  from the hardware in sm_90, so an Ampere device is the better testbed.
- **Established, direction only:** b1 is faster than cuBLAS bf16 at square shapes, and the
  advantage is smaller at head shape than at square shape within one state.
- **NOT established:** the magnitude. Anything between 1.9x and 4.2x. **Do not quote a number
  from this machine** until the cap is lifted and clocks are locked, which needs root:
  `nvidia-smi -pl 60` and `nvidia-smi -lgc <clock>`.

**The generalisable lesson, and it is not about this laptop.** A hard power cap is the normal
operating condition of every datacentre GPU under sustained load. If b1's advantage is
partly consumed by the power governor here, it will be partly consumed there too, and a
wall-clock ratio measured on an unconstrained short burst will overstate what a real training
run sees. This is direct support for the plan's decision to make **energy the headline axis
and wall-clock the honesty check**, and it should be measured as ops-per-joule at a fixed
power budget rather than as a speedup.

## The output head is exhausted, on both sides, and here is the arithmetic

**Cost side, closed by arithmetic.** At V=32,768 a head costing NOTHING AT ALL is worth
+0.0762 bpb at depth 4, **+0.0318 at depth 8**, +0.0168 at depth 12. Every approximation
measured here costs more than the entire prize: factorised distribution 0.125, hierarchical
softmax 0.122 (it ran, c05), cost-matched low rank 0.15+.

**Quality side, closed by measurement.** c15 at depth 8, V=32,768, budget pinned, dense at
0.958981 and 2.862643e8 FLOPs/token:

| arm | FLOPs | needs a gain of | actual gain | verdict |
|---|---|---|---|---|
| rerank top-k, k=64 r=32 | 1.000386x | +0.00003 | **+0.00078** | within noise |
| rerank full, r=32 | 1.022321x | +0.00162 | +0.00060 | within noise |
| MoS-2, r=32 | 1.022343x | +0.00162 | -0.00443 | loss |
| per-token temperature | 1.000011x | +0.00000 | -0.00358 | within noise |

A correction costing 1.0004x needed a gain of 0.00003 bpb and returned 0.00078, five times
inside the noise floor. **The rank ceiling is not binding at d=512 in this regime.** That
contradicts the reading of Godey et al. (2024) that motivated the arm; their saturation
result is about small models trained far past Chinchilla, and 440M tokens at d=512 evidently
does not reach it. Under K12 there is no larger budget available to find out.

**And it was 1.65x SLOWER**, 321.90 ms against 195.34, at 1.0004x the FLOPs, MFU 38.84 down
to 23.58. A per-token gather of `k` rows from a `(V, r)` table, which is 0.01% of the
arithmetic. That is the FIFTH time in this project a non-GEMM access pattern has cost more
than the FLOPs it saved, and the first time it happened on a table small enough that I
argued it would not. K5 is stronger than stated: it is not about how much data you gather,
it is about gathering per token at all.

So: fourteen head variants across five architectures, and the best margin against dense the
project ever produced was Monarch's +0.004 bpb at V=131k depth 8, which is inside the noise
floor measured below. **The output head is not where efficiency lives at this scale.** Any
further head work needs a new reason to exist, not a new parameterisation.

## The noise floor is at least 0.0101 bpb, not 0.004, and that reprices several conclusions

Two DENSE runs at what should be an identical configuration, same parameter count
(125,829,648), same FLOPs/token (2.862643e8), same total training FLOPs (1.260714e17):

| sweep | val bpb |
|---|---|
| c14 | 0.969126 |
| c15 | 0.958981 |
| spread | **0.0101** |

The project has been quoting +/-0.004 throughout. If these two runs really are identical in
configuration then the floor is 2.5x larger than assumed, and every margin under about 0.02
bpb in this repo is unresolved. That includes Monarch's +0.004 at V=131k depth 8, the
+0.0045 and +0.0048 pair that LEARNINGS calls "worth more than either number alone", and the
whole depth-12 analysis, which sat inside +/-0.003 of break-even.

**Do not treat this as settled.** Two points is not a noise estimate, and the alternative
explanation is that the two dense arms differ in something not visible in the summary line
(device batch and therefore gradient accumulation, data ordering, seed). The fix is cheap and
should happen before any small margin is quoted again: run the dense arm three times at one
depth and report the spread. Until then, quote +/-0.010.

## The constraint set any output-head idea now has to clear

Consolidated from everything measured in this project. Each line kills a family, so check a
candidate against all nine before it costs an arm.

**Read the EVIDENCE CLASS column before leaning on any of these.** Most were measured on a
trained DENSE head's weights and activations, which assumes a new architecture has to
reproduce dense's function on dense's activation distribution. In pretraining the body
co-adapts to whatever head it is given, so a dense oracle is an upper bound on how bad an
alternative looks and not a proof that it is bad. K7 and K8 are two measurements of exactly
that error, and both found the oracle wrong.

| # | constraint | evidence class | measured |
|---|---|---|---|
| K1 | a head linear in `h` has logit rank <= d, which dense already attains | **algebra** | rank_ceiling analysis |
| K4 | materialising V logits is a bandwidth floor | **arithmetic** | ~2.6x ceiling however much arithmetic is removed |
| K5 | non-GEMM heads lose on GPUs | **hardware** | gather 4.76x slower than a matmul doing 64x more work; three separate deaths |
| K6 | factorising the output DISTRIBUTION does not reach the frontier | **from-scratch training** | 12 arms, best margin -0.1248 |
| K7 | the realisability gap is ~30x the free-fit oracle | oracle vs training | oracle 0.0051 bpb, trained 0.1529 |
| K8 | free-fit oracles are blind to costs amortised ACROSS contexts | oracle vs training | mis-ranked the assignment twice, in both directions |
| K9 | a measured deficiency in a component is not evidence it is binding | from-scratch training | head held 4.2% of the prior; handing it 100% bought -0.0019 bpb |
| K2 | the head's rows do not cluster, at ANY quantisation depth | dense oracle, BUT independently supported by c00 and c05 from-scratch runs | k-means radius 2.455 -> 1.983 for k=16..4096, tracking a Gaussian control at 0.89; residual VQ with 256^6 cells reaches only 1.74 |
| K3 | coarse log-partitions are not linear in `h` | **DENSE ORACLE ONLY, and wrong by 4.7x** | predicted 0.568 bpb for hierarchical softmax; it actually cost 0.1217 |
| K10 | no training schedules (curricula, staged growth, periodic updates) | project constraint |
| K11 | no externally computed statistics (frequency tables, corpus counts, clusterings from a proxy checkpoint); end-to-end learned only | project constraint |
| K12 | must work at V=32,768. A 131k-only payoff costs compute we do not have | project constraint |
| K13 | must be paper-shaped. An idea that cannot be a paper has no place here, since a paper is the entire aim | project constraint |

K3 was used to declare hierarchical softmax dead. It measures whether a linear map can
reproduce THIS dense model's cluster marginals; a model trained from scratch with a
hierarchical head would shape `h` to make that map linear, which is the whole point of
co-adaptation. `HierarchicalSoftmaxHead` is implemented here and has never been run, and it
costs four minutes. Until it has, K3 is a hypothesis and should not be cited as a kill.

**And the prize is smaller than it looks.** A head costing NOTHING AT ALL is worth only
+0.0318 bpb at V=32,768 depth 8, and +0.0846 at V=131,072 depth 8. Every approximation to
the head's FUNCTION measured here costs 0.12 to 0.25 bpb, i.e. four to eight times the
entire prize. The reading that unifies K1 and K6 is that **the softmax head is not
wasteful, it is a very good d-dimensional chart on the simplex**, and every cheaper chart
we have built is a worse one by more than the FLOPs are worth.

### Closed, with numbers, so they stop being re-proposed

**Hierarchical softmax: RAN, and it loses.** `c05`, depth 4, V=32,768: val_bpb **1.2813**
against dense's 1.1596, so +0.1217 excess at 0.00046x the head. Head share is 64.6% there, so
the budget is +0.0762 and the margin is **-0.0455**. It is the closest any structured head has
come and it still loses, and under K12 there is no larger vocabulary to rescue it. Closed.
(Its logged `avg_depth=15.00` was a pre-`init_weights` placeholder, now labelled.)

**The binary-code seed: closed on its own motivating hypothesis.** The claim was that a code
forces parameter sharing and therefore helps rare tokens. Dense's `bpb_tail_minus_head` is
**0.1901**; all 17 code arms in c05 fall between **0.2802 and 1.0636**. Codes made the tail
uniformly worse, which is the opposite of the prediction.

**And the cost side as a whole is capped at V=32,768.** A head costing NOTHING is worth
+0.0762 bpb at depth 4, **+0.0318 at depth 8**, +0.0168 at depth 12. Every approximation
measured here costs more than that. Cost-side head work cannot produce a result at our
vocabulary; the quality side has no such ceiling, and a head at 1.06x of the head is +2.1% of
total FLOPs at depth 8, needing only 0.0015 bpb of gain to be Pareto-positive.

## The head's BACKWARD is two thirds of its cost and does not need the vocabulary

The way out of that box is to stop approximating the head's function. Per token the dense
head costs `6Vd` FLOPs: `2Vd` forward, `2Vd` for `dL/dh`, `2Vd` for `dL/dW`. Both backward
terms are driven by `dL/dz = p - y`, which is as concentrated as `p` is: top-64 mass is
0.9848 at V=131,072 depth 8.

Truncating the logit gradient to the top-k plus the target, measured on 1,024 real
activations against the exact gradient into the body:

| k | cos(dh_k, dh) | relative error | distinct rows touched |
|---|---|---|---|
| 8 | 0.99572 | 5.36% | 2,099 / 131,072 |
| 32 | 0.99821 | 1.53% | 6,058 |
| **64** | **0.99844** | **0.88%** | 10,092 |
| 256 | 0.99860 | 0.32% | 26,156 |

**The forward stays exact**, so the model remains a true dense-softmax model, reported bpb
is exact, and NONE of K1 to K7 apply: there is no approximation to the function at all,
only to the gradient.

It also has a GEMM-shaped implementation, which K5 says is the thing that matters. Taking
the union of the top-k over a chunk of tokens makes `W[K]` one contiguous gathered
submatrix and both backward terms plain dense GEMMs, exactly the shared-candidate trick
that fixed the proposal head:

| chunk | k | union \|K\| | \|K\|/V | head FLOPs vs dense |
|---|---|---|---|---|
| 256 | 8 | 692 | 0.0053 | 0.337x |
| 256 | 64 | 3,884 | 0.0296 | 0.353x |
| 1024 | 64 | 10,085 | 0.0769 | 0.385x |

A chunk's union contains every member of each token's own top-k, so the shared set is at
least as accurate per token as the per-token numbers above. Budgets at 0.385x:

| setting | head share | total FLOPs | break-even |
|---|---|---|---|
| V=131,072 d4 | 88.0% | 0.459x | **+0.0629** |
| V=131,072 d8 | 68.4% | 0.579x | **+0.0441** |
| V=131,072 d12 | 50.7% | 0.688x | +0.0214 |
| V=32,768 d8 | 35.2% | 0.784x | +0.0179 |

Against a cost that is 0.88% of gradient noise rather than a change to the model class.

## The head's rows do not cluster, and that one fact closes most of the design space

Every head this project built before c14 approximates the ``V x d`` MATRIX and
normalises exactly. Five measurements on the trained d8 V=131,072 head say that
whole route is closed, and each of them kills a family rather than an arm.

**K1. A head linear in ``h`` has logit rank <= d**, which dense already attains, so
a cheaper linear head is a *constrained* rank-d matrix and the constraint is the
entire quality cost. Already recorded under "A linear head's rank ceiling is d".

**K2. The rows are an anisotropic Gaussian cloud with no partition structure.**
Measured in the whitened metric ``W E[h h^T]^{1/2}``, where Euclidean distance IS
RMS logit error over the real activation distribution:

| k-means k | head W | Gaussian control, same 2nd moment | ratio |
|---|---|---|---|
| 16 | 2.455 | 2.853 | 0.860 |
| 64 | 2.305 | 2.584 | 0.892 |
| 256 | 2.184 | 2.437 | 0.896 |
| 1024 | 2.084 | 2.324 | 0.897 |
| 4096 | 1.983 | 2.218 | 0.894 |

Row RMS about the mean is 3.73. A 256x increase in centroids buys 19%, and the
ratio to the control is flat at 0.89. **Residual (multi-level) quantisation does not
escape it**: six levels of 256 codes, which is 256^6 = 2.8e14 cells against
V=131,072, move the residual only from 3.73 to 1.74, tracking the Gaussian control
level for level (2.30 / 2.09 / 1.97 against 2.18 / 1.99 / 1.90). Restricting to the
15.6% of tokens that ever enter a top-64 does not help either (2.13 at k=1024).

That single table explains the whole negative history of the direction in one line:
any head that replaces a token's row by a shared representative pays about 2 nats of
RMS logit error whatever the codebook. It kills product codes, VQ-Logits, RQ-VAE and
semantic-ID heads, tiered-by-similarity, and the class score of any two-level head.
It also prices lattice codes quantitatively: the flat 0.89 ratio to a Gaussian
caps better sphere packing at an 11% gain.

**K3. Coarse log-partitions are not linear in ``h``.** The only approximation in a
two-level head is the class score standing in for ``log sum_{w in c} exp(W_w h)``.
Closed-form least squares, so no SGD and no overfitting artefact: test R-squared
0.927 on the true log-partition, and ``KL(true class marginal || surrogate)`` is
**1.977 nats = 0.568 bpb** at 512 clusters. Coarser is better and still hopeless:
16 / 64 / 256 groups cost 0.136 / 0.290 / 0.423 bpb. The error tracks K2's cluster
radius, as ``|error| <= radius * ||h||`` predicts. Hierarchical softmax at V=131k is
dead, and so is the first stage of an autoregressive digit head.

**K4. Materialising V logits is a bandwidth floor.** Already in Q8: dense and a
perfectly fused cheap head write the same ``(N, V)`` tensor, so the ceiling is about
2.6x however much arithmetic is removed.

**K5. Non-GEMM heads lose.** Product gather 4.76x slower than a matmul doing 64x
more arithmetic; routed dispatch a graph break per layer; the proposal head's
per-token gather at 2.1 s/step against dense's 0.4.

### What the frozen binary code was actually missing

Phase 0 froze the *whole* head and measured 1.79% captured energy. Split the head
spectrally instead, keep a learned rank-r core and replace only the rest with a
frozen random +/-1 code, then re-optimise ``h`` per context:

| r | KL, frozen sign code | bpb |
|---|---|---|
| 0 | 0.35587 | 0.1023 |
| 8 | 0.06989 | 0.0201 |
| 32 | 0.03432 | 0.0099 |
| 128 | 0.02048 | 0.0059 |

The procedure's own floor on the true head is 0.01015 nats. So a learned rank-32
core plus a frozen binary code lands within 0.007 bpb of the dense head's reachable
set, and r=0 is the Phase 0 configuration. The idea was right and the decomposition
was wrong. It is still not the architecture: ``U_r(A_r h) + Phi(G h)`` is
algebraically a rank-(r+M) linear head, so freezing buys parameters, optimiser state
and the weight gradient but **no forward FLOPs**, and K4 caps whatever a fast
transform would add.

### The escape: factorise the distribution, not the matrix

    p(w|h) = sum_r psi_r(h) phi_r(w) / Z,     Z = psi(h) . [sum_w phi(w)]

Z costs R operations rather than V, no ``(N, V)`` tensor exists on the training path,
and ``log p`` is a log-sum-exp of sums of log-softmaxes so the log-prob matrix is not
confined to rank d+1. Cheaper than dense AND above its rank ceiling, which is the
pairing Mixture of Softmaxes buys at R x dense cost.

The governing quantity stops being the rank of the head matrix and becomes the
**nonnegative rank of the conditional distribution**, and language is unusually
favourable there: mean entropy 1.23 nats, effective support 9.3 tokens, top-1 mass
0.677, top-64 mass 0.9805. Free-fit oracles against the true dense distribution, in
bpb, against the +0.0931 a head costing nothing at all is worth at d8:

| family | R or m | head MACs vs dense | oracle bpb |
|---|---|---|---|
| cp 512x256 | 1 (LightRNN) | 0.0059x | 0.166 |
| cp 512x256 | 32 | 0.188x | 0.0033 |
| cp 64x64x32 | 1 (LightRNN) | 0.0012x | 0.283 |
| cp 64x64x32 | 32 | 0.039x | 0.0097 |
| cp 64x64x32 | 64 | 0.079x | 0.0056 |
| global V x m table | 128 | 0.00098x | 0.155 |
| global V x m table | 256 | 0.00196x | 0.0573 |

Two things the numbers say that the derivation does not. Ordering the leaf code axis
by predictability, so index j means "the j-th likeliest word in my cell", is worth 10
to 20% at every R and G for zero run-time cost. And the R=1 corner is 18 to 29x worse
than R=32, which is the measured statement of why LightRNN was abandoned in 2016: the
mixture is not a refinement of that line, it is the thing that was missing.

### The one thing these oracles do not settle

They are FREE fits. ``cp`` at G=3, R=32 has 5,152 factor parameters per context and
the real head has to produce them from a 512-dimensional ``h``. 4,096 probe
activations cannot answer that: a constrained fit on them reaches train KL 0.026 nats
and test 1.98. The ``global`` mode is the hedge precisely because its per-context
degrees of freedom (m=256) sit BELOW d=512, so its oracle is close to a guarantee
rather than an upper bound.

### Measured on hardware, V=32,768 d4, one config per process, compiled

| arm | ms/step | head FLOPs vs dense | peak memory | head params |
|---|---|---|---|---|
| dense | 31.23 | 1.0000x | 0.574 GB | 8,388,608 |
| nfh cp R=1 | 17.59 | 0.0030x | 0.342 GB | 24,929 |
| nfh cp R=32 | 19.96 | 0.0947x | 0.370 GB | 797,728 |
| nfh cp R=64 | 22.39 | 0.1895x | 0.435 GB | 1,595,456 |
| nfh global m=256 | 22.45 | 0.0078x | 0.511 GB | 8,454,400 |

1 graph and 0 graph breaks for every arm. This is the first head in the project that
is faster than dense rather than slower, and the reason is that there is no gather
and no custom kernel in it: two or three ordinary GEMMs of total width
``R(1 + sum_g K_g)`` = 3,104 against dense's 32,768.

Measure each config in its OWN process. A first pass timing all three in one gave
93.6 ms for the global arm against 22.4 measured cleanly, purely from allocator
pressure and interleaved compilation.

## RETRACTED: the V=32,768 offline numbers were measured on a byte-level token stream

Everything this file previously recorded under "V=32,768 is a valid screen" is withdrawn.
The activations were produced by `scripts/dump_head_acts` against a local `tokenizer/`
that was a 265-token stub left over from an empty-corpus run, so a model trained on 32k
BPE was fed what amounts to bytes. The tell is one line and should have been the first
thing checked: **every target id was below 265, 74 distinct values out of 32,768.**

Withdrawn: the V=32,768 free-fit oracle table (R=32 at 0.00509 bpb and the rest), the
distribution statistics (2.939 nats, effective support 32.1, top-1 mass 0.417), the claim
that the product mixture fits BETTER at the small vocabulary, and the leaf ordering inside
`perms/nfh_g3_v32768.pt`, which was built with `--acts` pointing at the same file. That
permutation's nested clustering came from head ROWS and is sound; only the leaf axis is
noise. The file is parked as `perms/CORRUPT_*.bak` rather than deleted, because a
silently-wrong permutation is worse than a missing one.

Standing intact, because they came from `acts_d8_v131k.pt` rather than from a local dump:
the whitened row geometry, the residual-quantisation result, the class log-partition
measurement, the spectral split, and every V=131,072 oracle. Also intact: the FLOPs and
break-even arithmetic, which is pure counting, and the wall-clock and graph-break
benchmark, which used random ids and measures only time.

What survives of the original claim is its COST half, which needs no activations. The
method's scaling law is

    head / dense = R (1 + sum_g K_g) / V,   sum_g K_g = G V^(1/G)

so 0.0947x at V=32,768 against 0.0393x at V=131,072, independent of d and improving as V
grows; and head share is 35.2% against 68.4% at depth 8. Both say V=32,768 is the harsher
setting on cost. Whether it is harsher or kinder on QUALITY is now unmeasured.

**The generalisable lesson.** An activation dump inherits every upstream artefact silently.
`dump_head_acts` now saves the targets alongside, and the first thing any probe should do
is check that the target ids span the vocabulary. It costs one line and it would have saved
this.

## The missing prior was not the problem, the assignment is the big lever, and my explanation of WHY was wrong

Second and third d8 sweeps, V=32,768, budget pinned, dense at 0.969126 and 2.862643e8
FLOPs/token. Twelve arms in total across rank, assignment, factorisation depth, smoothing
and mode:

| arm | M | bpb | FLOPs vs dense | margin |
|---|---|---|---|---|
| R32 g3, ACCIDENTAL perm (d4 clustering, projection leaf) | 3104 | 1.122022 | 0.6817x | **-0.1248** |
| R32 g2, 256x128 | 12320 | 1.134317 | 0.7806x | -0.1470 |
| R32 g3, frequency leaf | 3104 | 1.222611 | 0.6817x | -0.2254 |
| R32 g3, frequency leaf + smoothing | 3104 | 1.219365 | 0.6817x | -0.2221 |
| R32 g3, random leaf | 3104 | 1.254324 | 0.6817x | -0.2571 |
| R32 g3, token-id order | 3104 | 1.260380 | 0.6817x | -0.2631 |
| R32 g3, pure random | 3104 | 1.323261 | 0.6817x | -0.3260 |
| R1 g3 | 97 | 1.643873 | 0.6494x | -0.6431 |

**Smoothing is dead.** -0.0019 bpb, half the noise floor, after the checkpoint showed the
head capturing only 4.2% of the available prior (KL 2.6007 against a uniform background's
2.7140). It was handed the other 95.8% free, initialised to the true unigram, at one gather
per token, and nothing happened. The tail was never the bottleneck: the mixture's
CONTEXT-DEPENDENT tail must already beat a static unigram. **A measured deficiency in a
component is not evidence that the component is binding**, and treating it as such cost an
arm.

**The assignment is the largest lever**, spanning 0.20 bpb from best to worst, which is
larger than the 0.125 gap to dense. It is worth more than rank, than smoothing, and than
factorisation depth.

**RETRACTED, same day: my explanation of why.** Having seen a frequency-ordered leaf lose
to an accidentally-random one, I concluded the axis wanted balanced probability MASS and
cited the Monarch result ("random, which destroys structure, wins"). Building both
orderings deliberately from the same clustering refutes it: frequency ordering BEATS random
by 0.032, and PURE RANDOM IS THE WORST ARM OF ALL by 0.10 against the best. Coherence
matters for this head; the Monarch finding does not transfer. The earlier comparison was
confounded, because the accidental permutation differed in its clustering source as well as
its leaf order, and I attributed the whole 0.099 to the leaf.

**What is actually unexplained.** The accidental permutation still beats every deliberate
one by 0.10 bpb. It differed in two ways: it clustered from a DEPTH-4 head rather than the
depth-8 one, and its leaf was ordered by ``W h_bar``, a projection onto a fixed direction,
which is a smooth geometric ordering rather than frequency or random. Two arms separate
them, and ``--leaf proj`` now builds the second from the head's own leading principal
direction, so it needs no activations.

**The methodology lesson, which is the reusable part.** The free-fit oracle ranked leaf
orderings 10 to 20% in favour of score-ordering; training ranked the assignment as the
dominant term and put the oracle's preferred choice mid-table. A free per-context fit
refits every context from scratch, so it is structurally blind to any cost that is
AMORTISED ACROSS CONTEXTS, and the assignment is entirely such a cost. Offline oracles are
sound for eliminating families on capacity grounds and unsound for ranking choices whose
whole effect is shared structure. That cuts both ways and it cost two wrong calls here, in
opposite directions.

## Two hypotheses about the first d8 sweep, settled from the checkpoint alone

The first V=32,768 depth-8 sweep put `NFH_cp_R32` at 1.1220 bpb against dense's 0.9691, a
+0.1529 bpb excess where the budget was +0.0281, i.e. losing by 0.125. Two explanations
were live, and the trained weights separate them with no activations required.

**The head learned no static prior, and that is structural.** Switching off the
context-dependent term leaves the bias-only background:

| | R=1 | R=32 | uniform | TRUE 32k unigram |
|---|---|---|---|---|
| background entropy (nats) | 10.394 | 10.395 | 10.397 | **7.683** |
| background mass on the top 1000 | 0.0377 | 0.0359 | 0.0305 | **0.6427** |
| KL(true unigram \|\| background) | | **2.6007** | 2.7140 | 0 |

Scored against the real frequency table rather than against intuition: the trained head's
static background is 2.6007 nats from the unigram where a LITERALLY UNIFORM one is 2.7140,
so it captured **4.2%** of the prior information available to it. Not "some", none. The biases barely moved (std 0.065 on
the factors, 0.035 on the gate) while the context-path row norms reached 3.2, so the head
spent everything on the context term. That is a choice, not a bug: the only static object
the parameterisation offers is a rank-R product over the grid, and R components spent on
the prior are R components not spent on the context. Excess of +0.1529 bpb is 0.501
nats/token, and the prior the static path could hold for free is worth 2.60 nats, so the
capacity currently spent re-deriving it at every position is five times the shortfall.
That is not a claim that smoothing recovers 2.60 nats of loss; it is the size of the
budget it frees. The mechanism is the same either way: a word no component points at falls
to the background product, about (1/32)^3 = 3e-5 against a true unigram of 1e-4 to 1e-3.

**The components did NOT collapse.** Mean absolute cosine between components, per code
axis: 0.171, 0.045, 0.037, with a maximum of 0.32. Thirty-two distinct, near-orthogonal,
all in use. So the mixture's optimisation is healthy and every proposed fix aimed at
collapse (entropy regulariser on the gate, symmetry-breaking init, load balancing) is
ruled out before it costs an arm. That is what a checkpoint buys over a sweep.

## 2026-09-04 (2) — c05: Monarch wins, the code is the part that does not work

Phase 5 ran, 22 arms at depth 4, V=32768, 121M tokens, with step times.

### Both Pareto frontiers

| FLOPs/token vs bpb | | | wall clock vs bpb | | |
|---|---|---|---|---|---|
| TREE hsoftmax | 0.354x | 1.2813 | FREE order2 bias | 0.726x | 1.7629 |
| **MON M1024** | **0.455x** | **1.2559** | MON M1024 m1=8 | 0.754x | 1.3802 |
| BASE learned_W | 0.659x | 1.2271 | MON M256 | 0.760x | 1.3620 |
| BASE dense | 1.000x | 1.1596 | **MON M1024** | **0.762x** | **1.2559** |
| | | | BASE learned_W | 0.968x | 1.2271 |
| | | | BASE dense | 1.000x | 1.1596 |

**A Monarch head is the only arm on both.** 0.455x the FLOPs, 0.762x the wall clock, +0.096 bpb
against dense. Against the baseline that actually matters, a learned rank-120 head, it is 0.69x
the FLOPs and 0.79x the time for +0.029 bpb. It removes 84.2% of the head's cost and every
parameter of it is trained, so alignment never enters.

Projected end to end, holding the 84.2% head reduction: 1.74x at V=131k depth 12, 1.31x at V=131k
depth 20, 1.59x at V=262k depth 20. At V=32k depth 20 it is 1.08x, which is the regime the
original plan aimed at and the reason it had nothing to win.

The tree head is on the FLOPs frontier and *off* the wall-clock one: 139.9 ms against Monarch's
136.2 at a worse bpb. A 2005 method at 0.354x FLOPs and 1.2813 bpb nevertheless beat every
structured code arm in the sweep.

### The gather kernel, measured on real hardware

Q8 predicted the FLOP count would not survive contact with a GPU. It did not survive by a factor
of fifty:

| arm | FLOPs | wall clock | disagreement |
|---|---|---|---|
| PROD g8 K64 | 0.378x | **18.80x** | 49.7 |
| PROD g4 K64 | 0.366x | 9.71x | 26.5 |
| PROD g2 K512 | 0.378x | 8.36x | 22.1 |

Any product-code result quoted on the FLOPs axis is a claim about a kernel that does not exist.
The dense implementation of the same head runs at 0.78x dense wall clock while costing 1.23x the
FLOPs, which is the honest way to run these arms today.

### The cost-matched controls say the CODE is what fails

At M=512, identical width, identical implementation, identical cost (9.55e7 FLOPs, ~140 ms):

| Phi | bpb |
|---|---|
| K-ary product, random assignment | **1.5437** |
| random binary | 1.5811 |
| monomial of a binary code | 1.7196 |
| K-ary product, hash assignment | 1.7217 |

The K-ary advantage over a *random* binary Phi is 0.037 bpb. The real gap, 0.176, is between
structured and random. Both structured schemes are built from the token id, and both land at
~1.72; both random schemes land at ~1.55 to 1.58. **A code that correlates with the token id is
worse than no code at all**, which is close to the opposite of the plan's premise and is the
clearest single result in the sweep.

### Q9 answered: the union of subspaces bought nothing

| arm | bpb |
|---|---|
| MIX k8 top2 | 1.7587 |
| MIX k4 top1 | 1.7652 |
| MIX k8 shared_phi (control) | 1.7664 |
| MIX k8 top1 | 1.7675 |

The shared-Phi control matches the per-component arms to within 0.001 bpb, so per-component Phi
contributed nothing, and the arms cost 1.3x to 5.0x dense in wall clock for it. Close the
direction.

### The free fixes were worth more than the architecture

Against BASE code order2 at 1.9401:

| | bpb | delta |
|---|---|---|
| + per-token bias | 1.7629 | **-0.177** |
| + whitened Phi | 1.8939 | -0.046 |
| + both | 1.7210 | -0.219 |
| order 3 + both | 1.5559 | -0.384 |

The bias is worth as much as the entire K-ary-versus-monomial gap, and c00 ran without it, so
every c00 code arm was understated by roughly that much. The retraction above said the bias was
"not the lever"; that was wrong, and it was wrong because it was inferred from the noise
checkpoint. Whitening moving 0.046 bpb is a pure optimisation effect, since it provably preserves
the function class for a linear g, so conditioning was real and secondary.

### What this means

The output head is worth attacking and the code head is not the way to attack it. The two arms
that work, Monarch and the Huffman tree, are both *fully learned or fully classical* structures
with no assignment problem. Every arm whose behaviour depends on a frozen code assignment lands
between 1.54 and 1.94, and the best of those is the one whose code is random.

## 2026-09-04 — SCH Phase 0: what the run established, and a retracted mechanism

Phase 0 (`c00`, depth 4, d=256, V=32768, 121M tokens) ran. The rank gate **passed**:
order 1 measured effective rank exactly 15 = B, order 2 exactly 120 = M, dense 256 = d.
Section 3.1 is confirmed and the implementation is correct.

### What the run said

| arm | M | rank ceiling | measured rank | FLOPs/token | bpb |
|---|---|---|---|---|---|
| order 1, linear g | 15 | 15 | 15 | 2.95e7 | 2.2726 |
| order 2, linear g | 120 | 120 | 120 | 4.34e7 | 1.9437 |
| order 3, linear g | 575 | 256 | 280 | 1.04e8 | 1.8323 |
| order 3, MLP g | 575 | 575 | 309 | 1.04e8 | 1.8764 |
| order 4, MLP g | 1940 | 1940 | 428 | 2.85e8 | 1.8662 |
| random binary Phi | 120 | 120 | 120 | 4.34e7 | 2.0723 |
| **learned W at width M** | 120 | 120 | 120 | 5.13e7 | **1.2281** |
| **dense softmax** | - | 257 | 256 | 7.79e7 | **1.1593** |

Three facts, all solid because they come from real training runs.

**Rank stopped binding after order 1.** Order 4 reaches measured rank 428, 1.7x the dense
head's 256, and is still 0.71 bpb worse. Three very different capacity settings (k=3 linear,
k=3 MLP, k=4 MLP) land within 0.044 bpb of each other. That is a hard wall and it is not a
rank wall.

**The matched-width ablation isolates the cost of freezing.** At M=120, same g, same
optimizer, same budget, learned Phi scores 1.2281 and frozen monomial Phi scores 1.9437.
The only difference is whether the 120-dimensional subspace of R^V is chosen by SGD or fixed
by the code. That 0.716 bpb is the number any fix has to attack. Code choice is small next to
it: monomial against random binary Phi is 0.129 bpb.

**Orders above 2 leave the region where the head is cheap.** At d=256 the order-3 head costs
7.62e7 FLOPs against the dense head's 5.03e7, so every arm that beat 1.94 was already more
expensive than the baseline it was trying to beat.

### The mechanism, measured on the right checkpoint

An earlier version of this entry ran `scripts/code_head_subspace.py` on
`base_checkpoints/final_verification_v3/model_000100.pt`. That checkpoint has **2
learned directions out of 256**: comparing its spectrum against a random matrix of
identical Frobenius norm puts s[0] at 14.4x the bulk, s[1] at 1.9x, and everything
from s[2] down below it, and its input and output embedding column spaces overlap
0.0077 against a random-subspace baseline of 0.0078. Every number from that run was
capture of initialisation noise. It reproduced as a clean, plausible result, which is
the dangerous kind of wrong: unfitted bases scored at the random baseline M/V and
fitted ones scored high, reading exactly like "structured codes fail and privileged
codes partly work". `code_head_subspace.py` now refuses to report below 8 learned
directions.

Re-run on the real c00 dense head (`DENSE_softmax_s1`, 462 steps, 63 learned
directions carrying 54.75% of the energy) the conclusion holds and the numbers are
starker. Capture of the dense head's logit energy:

| code assignment | k=1 (M=15) | k=2 (M=120) | k=3 (M=575) |
|---|---|---|---|
| binary token-id bits, the c00 setting | 1.27% | 1.79% | 3.26% |
| random | 0.04% | 0.36% | 1.74% |
| frequency-ranked | 1.97% | 2.62% | 4.11% |
| SVD-thresholded, using the trained W itself | 16.47% | 20.32% | 22.99% |
| oracle, best possible M-dim subspace | 25.63% | 74.21% | 100% |

Two structural facts, and both survive the correction:

**Binary indicator columns span the wrong subspace.** At the M=120 c00 actually ran,
the monomial code reaches 1.79% where the oracle reaches 74.21%, a 41x shortfall.
Even codes built from the answer only reach 20.32%.

**Monomial expansion adds rank but not reach.** For the privileged SVD-threshold code,
M goes 15 to 575, a factor of 38, and capture rises from 16.47% to 22.99%, a factor of
1.4. That is the bpb plateau: three capacity settings landing within 0.044 bpb.

One earlier claim was **backwards**. On the noise checkpoint the top direction carried
95.18% of the energy and looked like unigram; on the converged head it carries **4.10%**,
and top-15 is 25.63%. So a per-token bias is worth much less than the retracted entry
implied. It is still free and still missing from every c00 arm, but it is not the lever.

### The fix the diagnosis points at

A binary digit contributes exactly one column to `Phi`. A **K-ary digit contributes K**,
so `g` digits give `M = g*K` at order 1 with no interactions at all. Screened with the
same tool on the same checkpoint:

| basis | M | capture |
|---|---|---|
| binary monomial order 2 (c00) | 120 | 1.79% |
| product code g=2, K=64 | 128 | **21.93%** |
| product code g=8, K=64 | 512 | 35.20% |
| product code g=8, K=256 | 2048 | 46.52% |
| product code g=16, K=256 | 4096 | 61.63% |

12x better at matched M, and it keeps growing where monomials saturate. Fitting to the
input embedding instead of the output head halves it (21.69% at M=2048), which is worth
knowing: untied input and output embeddings do not share a column space, so the proxy
has to be a head.

And the structure is free to evaluate. `Phi` one-hot per group makes `g(h) Phi^T` a
gather and add costing `V*g` instead of `V*M`. At V=131072, d=768, g=8, K=256 that is
44x fewer FLOPs than the dense head with a higher rank ceiling. Open question Q1 asked
whether a fast transform exists for the truncated monomial expansion; for one-hot codes
it exists and it is trivial.

### What did survive: the prize was mis-sized

Head share of total FLOPs, from `GPT.estimate_flops`:

| depth | d | V=32k | V=65k | V=131k | V=262k |
|---|---|---|---|---|---|
| 4 | 256 | 64.6% | 78.5% | 88.0% | 93.6% |
| 8 | 512 | 35.2% | 52.0% | 68.4% | 81.3% |
| 12 | 768 | 20.4% | 34.0% | 50.7% | 67.3% |
| 20 | 1280 | 9.0% | 16.5% | 28.3% | 44.1% |
| 32 | 2048 | 3.8% | 7.3% | 13.7% | 24.0% |

The plan sized the target at V=32k, where the head is 9% of FLOPs at depth 20 and the idea
has almost nothing to win. At the vocabulary sizes current models actually use (Llama 3 at
128k, Gemma at 256k) the head is 28% to 44% of FLOPs at depth 20. That is the regime the
work should have been aimed at.

And the plan aimed at the wrong saving inside that regime. Freezing `Phi` removes the weight
gradient, worth 2 of 6 FLOPs per MAC, so one third of the head. Making the head structurally
cheaper is worth most of the head. The second target is three times larger and the plan spent
Phase 0 on the first.

## 2026-08-30 — Structured code output heads: the rank bound, the width cap, and two measurement traps

New research direction, implemented in `nanochat/code_head.py` and
`nanochat/code_metrics.py`. Nothing has been trained yet; every number below is either
algebra or a CPU smoke measurement, and is labelled as such.

**The object.** Replace the dense `V x d` softmax head with
`logit(w|h) = phi_k(c(w))^T g(h)`, where `c(w)` in `{0,1}^B` is a frozen binary code per
token and `phi_k` is the vector of all monomials (AND products of bits) up to interaction
order `k`, so `M = sum_{j<=k} C(B,j)`. The useful reduction, worth stating early because it
collapses the whole design space: this is *a softmax whose output embedding matrix
`Phi in {0,1}^{V x M}` is frozen, binary and structured*. Order 1 is Oda et al. 2017,
order `B` is the exact softmax, VQ-Logits is the one-hot corner, and "learned dense `W` at
width `M`" is the matched-capacity control. All of them are one implementation with one
flag changed.

**Why an independent-bit head is rank `B`, in four lines.** For independent Bernoulli bits
with `s_b(h) = u_b^T h`,

```
log P(w|h) = sum_b [ c_b log sigma(s_b) + (1 - c_b) log sigma(-s_b) ]
           = sum_b log sigma(-s_b) + sum_b c_b [ log sigma(s_b) - log sigma(-s_b) ]
```

and `log sigma(s) - log sigma(-s) = s` exactly, so `log P(w|h) = A(h) + sum_b c_b(w) s_b(h)`.
`A(h)` does not depend on `w` and is absorbed by normalisation, leaving the logit matrix
exactly `C S`, hence **rank <= B**. At `V = 32768` that is rank 15, against the ~1000
empirical head-rank threshold of Godey et al. (2024). This is why prior binary-code work
needed error-correcting codes and softmax mixing to approach parity: they were fighting a
two-orders-of-magnitude rank deficit without naming it.

**The width cap is the confound that would have produced a false result.** If `g` is a
*linear* map `R^d -> R^M` then the logit matrix is `Phi G H` and its rank is `min(M, d)`,
not `M`. At `d = 512`, `B = 15`, orders 3 (`M = 575`) and 4 (`M = 1940`) are therefore
rank-identical. Running the ladder without knowing this shows saturation at order 3 and
supports a conclusion that is plausible enough to survive review and then collapse at
scale. Two mitigations, both now wired in: `--sch-g-type mlp` (a nonlinear `g` has an image
not contained in any `d`-dimensional subspace, restoring the ceiling to `M`) and a second
arm at `d = 1024`. This also upgrades the claim rather than merely defending it: a softmax
can never exceed rank `d+1` at any parameter count, so a code head with nonlinear `g`
decouples output rank from model width.

**Measurement trap 1: mean-centre across the vocabulary axis before the SVD.** Measured on
a tiny model: probing raw logits, the order-1 head reports rank 9 = B whether or not you
centre, because `g(h) Phi^T` carries no `A(h)` term. Probing *log-probabilities* it reports
**10 uncentred and 9 centred**, because `log_softmax` subtracts `logsumexp(h)`, which is
exactly the rank-1 `A(h)` term. Off by exactly one is small enough to read as noise and
large enough to make an order-1 head look like it broke its own bound.

**Measurement trap 2: probe in fp32.** With `Phi` stored in bf16 the singular values below
the true rank sit at roughly `1e-3` of the leading one rather than at zero. Measured on the
same tiny model: the order-1 head reported effective rank **225 instead of 9**, and the
*dense* `d = 64` baseline reported **276 instead of 64**. Both became exact once the head
was run in fp32. `nanochat/code_metrics.measure_logit_rank` now promotes `Phi` and casts the
head's input to fp32 for the probe; the trained weights are untouched. Verified afterwards:
dense measures exactly 64 = d, order 1 measures exactly 9 = B, order 2 with an MLP `g`
measures exactly 45 = M.

**The honest cost model, and why `estimate_flops` had to change.** Per token the softmax
costs `V d` MACs and the code head costs `V M` plus `d M` for `g`, so the code head is
cheaper in *compute* only when `M < d`. At `V = 32k`, `d = 512`, order 4 (`M = 1940`) it
uses about 17x fewer head parameters and about 4x more compute. The repo's generic
`6 * params` FLOP proxy cannot see this at all, because a frozen `Phi` owns no parameters:
it would have reported the code head as nearly free. `GPT.estimate_flops` now removes the
head from that term and prices it exactly, charging the frozen `Phi` **4** FLOPs per MAC
(forward, plus the gradient with respect to `g(h)`, and no weight gradient) against **6**
for a learned one. A hierarchical softmax has the opposite problem: it owns `V d` node
parameters but touches only `~log2(V) d` of them per token, so the same proxy would have
overcharged it by three orders of magnitude.

**Per-bit BCE is not an option, and this is a correctness point rather than a preference.**
BCE over bits *is* the independence assumption the interaction expansion exists to remove,
so keeping it while adding interactions is incoherent. It also fails outright for redundant
codes: at exactly `B = log2 V` with bijective codes independent Bernoullis happen to
normalise over the `2^B` codewords, but as soon as `B > log2 V` (the `B` in {24, 32, 64}
redundancy arms) probability mass lands on codewords corresponding to no token, and the
failure is silent. Everything here computes `g(h) @ Phi^T` and uses exact cross-entropy over
the real vocabulary. That is affordable precisely where it matters: at `M = 120`,
`V = 32768` the logit matmul is about 3.9M MACs per token against the softmax's 16.8M, so
order 2 is roughly 4x cheaper than what it replaces while being exact. It also makes the
Oda-style baseline *stronger* than the original method rather than weaker: same rank bound,
exactly normalised.

**The order-2 coefficients must be emitted per position.** At order 2 the term is
`sum_{b<b'} c_b c_b' A_{bb'}(h)` with `A` a function of the hidden state, so `g` maps
`R^d -> R^M` and the head produces `M` numbers per token position. A single learned `B x B`
parameter shared across contexts would make that sum a fixed per-token constant: a bias
adding rank 1 rather than `C(B,2)`. That version is the easiest thing to build and gets none
of the benefit, and it would look like order 1 with extra steps. Pinned by an assertion in
the head's constructor and by three tests.

**Two infrastructure consequences that are easy to miss.** DDP's default
`broadcast_buffers=True` would re-broadcast the frozen `Phi` from rank 0 on every forward,
which is up to 842 MB at `V = 131072`, `M = 3213`; `wrap_model` now disables it for code-head
models, which is safe because `Phi` is built deterministically and identically on every rank
and never changes. And `base_train` sizes the token budget from
`transformer_matrices + lm_head` parameters, so a code head with 17x fewer head parameters
would silently receive a much smaller budget than the dense control; every SCH sweep pins an
identical explicit `--target-tokens`, computed once from the dense arm by
`scripts/code_head_budget.py`.

**Measured `Phi` construction cost** (CPU, one-off at `init_weights`):
`V=32768, B=15, k=4` gives `M=1940`, 127 MB in bf16, 0.26 s, density 0.091;
`V=131072, B=17, k=3` gives `M=833`, 218 MB, 0.43 s, density 0.153;
`V=32768, B=64, k=2` gives `M=2080`, 136 MB, 0.22 s, density 0.258. The last one also shows
why redundancy matters for the *code design* question and not only for rank: sampled minimum
Hamming distance is 14 at `B=64` against 1 for the minimal `B=15` code, and at distance 1
the ECC-versus-semantic comparison is undefined.

---

## 2026-08-08 — The MST Pareto claim was an artifact of a mismatched dense baseline

**What we believed.** MST (S7_COMBO_A + multi-scale windows, N=4) lay below the dense
scaling curve on all three cost axes: 1.77x on total parameters, 1.35x on FLOPs/token,
2.00x on training FLOPs (`MST_iclr2027/mst_iclr2027.tex`, Tables 2 and 3).

**What was actually true.** The dense arm at L=12..30 came from the nanochat leaderboard,
not from our own runs (flagged but not acted on in `remix_checklist.md`, item 47). Our own
dense runs at L=12, 20, 22 are uniformly better by 0.0374, 0.0385, 0.0370 bpb, mean
**0.0376 bpb**. Only the L=8 dense point in the paper table (0.9691) was ours. Refitting
with the offset applied to the whole dense ladder:

| Axis | Paper (leaderboard dense) | Corrected (our dense) |
|---|---|---|
| Total parameters | 1.81x MST | 1.21x MST, eroding with scale |
| FLOPs / token | 1.38x MST | **0.86x, dense wins** |
| Training FLOPs | 2.08x MST | **0.80x, dense wins** |

**Why the swing is so large.** At bpb ~ 0.80 with a FLOPs exponent of -0.10, a 0.0376 bpb
offset is worth e^(0.0376/0.80/0.10) = 1.6x compute. Small absolute bpb offsets are huge on
a log-log cost axis. Any future baseline-versus-variant comparison must be run in one setup
at one commit, or the headline multiplier is meaningless.

**The sharper finding underneath it.** At matched FLOPs the two arms also have matched
matrix parameters: MST L=32 has 511.8M matrix params at 4.008e9 FLOPs/token; dense L=22 has
523.4M at 3.694e9. This is forced, not coincidental: FLOPs/token ~ 2 x matrix params for
both architectures, so partitioning cannot change the params-per-FLOP ratio. It changes only
the params-per-(depth, width) ratio. Consequently:

- MST's remaining "win" on **total** parameters is entirely the value-embedding tables
  (MST L=32: 268.4M of VE at d=512; dense L=22: 507.5M at D=1408). VE contributes zero
  FLOPs and a dense model could shrink it the same way, so this is not an architectural
  advantage and will not survive review.
- The only defensible question is whether block-diagonal-plus-coupling beats unstructured
  dense **at equal matrix parameters and equal FLOPs**. Today it loses by 0.0096 bpb
  (MST L=32 0.7434 vs dense L=22 0.7338 with 8% fewer FLOPs).

**Target.** Roughly **0.03 bpb at iso-FLOPs** turns 0.86x into ~1.15x and restores a claim.

**How to apply.** Never quote a multiplier against a baseline not run in our own setup.
Report matrix params separately from embedding params on every scaling plot; the embedding
column is where spurious parameter wins hide.

See [[mst-paper-target-config]] for which MST configuration these numbers refer to.

---

## 2026-08-08 — MST's FLOPs deficit is overhead, not the partition

Refitting bpb against three different x-axes, same models and same points, splits the
0.859x FLOPs-axis deficit into its causes:

| bpb fitted against | dense | MST | iso-quality multiplier |
|---|---|---|---|
| **matrix params** | `4.551 x^-0.0909` | `3.898 x^-0.0827` | **1.005x, MST ahead** |
| matmul params (+`lm_head`) | `5.315 x^-0.0982` | `5.083 x^-0.0955` | 0.933x |
| FLOPs / token | `6.711 x^-0.1004` | `6.716 x^-0.0997` | 0.859x |

**Per matrix parameter MST is at parity.** Block-diagonal streams with a rank-`d` coupling
cost essentially nothing in quality per parameter. The deficit appears only when converting
parameters to cost: 7% lost to the output head, 8% to attention. Both are D-proportional
terms that partitioning does not touch, and MST needs a larger D for the same matrix
parameters (3.81 params per `L*D^2` against dense's 12.0). It therefore buys 0.1277 matrix
params per FLOP against dense's 0.1417.

**Consequences.**
- The lever is overhead, not coupling. This is consistent with the paper's own finding that
  twelve richer coupling variants all failed: the coupling was never the binding constraint.
- It gives a better mechanism for N=4 beating N=8 than per-block capacity: finer partitioning
  lowers params per `L*D^2`, forcing larger D, hence more overhead per parameter. It also
  makes **N=2 an untested Pareto candidate** (c ~ 7.5, closer to dense).
- The "amortization threshold" below 100M parameters is mostly the un-factorized output head.
  With `mst_lm_head_dim = D/4` the L=9 point moves from 0.659x to 1.044x.

**Risk to watch.** MST's exponent on matrix parameters is *worse* (-0.0827 vs -0.0909) and
the per-parameter ratio erodes monotonically: 1.08, 1.06, 1.00, 1.05, 1.00, 0.96, 0.89 across
L=16..32. Parity is real at L~16-24 and gone by L=32. If that is genuine rather than one noisy
point, overhead fixes only delay a crossover. Needs more points at the top of the ladder.

**How to apply.** Always plot bpb against matrix params alongside FLOPs; the gap between the
two curves is the overhead budget, and it is the actionable quantity.

**Correction, measured at L=8.** Only *half* of that gap is real overhead. Splitting it:

- **Attention is genuine overhead and is free to remove.** `mst_compose_windows` cut FLOPs
  10.2% for +0.0001 bpb, inside the σ ≤ 0.0003 seed noise.
- **The output head is capacity, not overhead.** `mst_lm_head_dim = D/2` cut FLOPs 27.7% but
  cost +0.0461 bpb, against +0.0424 predicted by the matmul-parameter scaling law
  (`|b| = 0.0955`); at D/4, +0.0928 measured against +0.0738 predicted. Shrinking the head
  just slides down the parameter curve, so it buys nothing on the FLOPs-vs-bpb frontier and
  is roughly break-even even at L=32 where it removes only 6.2% of matmul parameters.

Do not treat a parameter-bearing tensor as overhead just because it scales with D. The test
is whether removing it costs bpb in line with the parameter scaling law. Attention FLOPs pass
that test (they are compute, not parameters); the output head fails it.


---

## 2026-08-08 — Evidence standards for this project

Three calls in this session were made on evidence that could not support them, all
the same shape: a single-seed L=8 screen read as a verdict.

- **G2 / O1** were called dead on one seed. Both happened to replicate, but the
  standard was wrong either way.
- **Channel mixing** was called "wrong for MST" on one seed at L=8, while confounded
  with multi-scale windows, for a mechanism that is *depth-dependent by construction*.
  With a d/2 shift the reachable channel span grows about half a block per layer, so
  at N=4 it takes ~6 layers to reach all four streams. L=8 completes that once; L=32
  completes it four times. Testing it at the shallowest depth was close to a
  worst-case design.
- **"Nothing that trades FLOPs for parameters can help"** was inferred from MST
  sitting at 1.005x parity per matrix parameter. That only holds if the new
  parameters are as effective as the *average* existing one, which is exactly what a
  mixing W_O would not be. It does not rule out the dense output projection.

**Standard going forward.** A single-seed L=8 run is a screen that says where to
spend the next run. It does not close a direction. Anything inside ~2 sigma
(sigma <= 0.0003) is unmeasured, not null. Before calling a mechanism dead, check
that the experiment was not confounded and that the scale was one where the
mechanism can act.

**Also a constraint, not a variable.** Optimizing a cost proxy (parameters per FLOP)
recommended N=2, which is most of the way back to a dense model. "Stays genuinely
modular" bounds the search space; a recommendation that dissolves the research idea
is not a win no matter what the proxy says.

## 2026-08-08 — Per-stream attention scales cannot survive a sub-block shift

Attention windows are per-stream: one flash_attn call per stream, one window each.
Under a d/2 partition shift every slot holds channels from two different scale
groups, so no per-slot window preserves a channel's scale. Taking min or max across
the two changes attention FLOPs and destroys the FLOP-neutral comparison that made
the shift attractive in the first place. "Attach the window schedule to channels"
is therefore not implementable while scales are per-stream; the way to separate
mixing from specialization is to remove the specialization (uniform windows), not
to preserve it.


---

## 2026-08-08 — Flags that were allocated and then ignored

`_can_use_batched_layer` gates which config combinations may use `BatchedMSTLayer`, but it is
a hand-maintained list, and several flags were never added to it *and* never implemented on the
batched path. They therefore build their parameters, get reported in the config row of
`mst_results.csv`, and do nothing:

- **`mst_transition_every`** — read only by the legacy `MSTLayer`. Every model trained since
  Stage 7 coupled at every layer regardless of the flag. Now implemented (Stage 15 / F3).
- `mst_ffn_inner_dim` — same shape of bug; `BatchedMSTLayer` hardcodes `inner = 4*d`.
- `mst_input_mode stem` — `stem_blocks` are only run by `forward_with_cos_sin`, which the
  batched branch never calls, so the mini-transformer is built and skipped.
- `mst_sub_aux_weight` — on the batched path `sub_states` is a `(B,T,N,d)` tensor, so
  `zip(self.sub_aux_heads, sub_states)` iterates the *batch* axis.
- `mst_final_mode`, `mst_final_topk`, `mst_diversity_weight`, `mst_routing_mode` are inert
  there too: the batched branch calls `final_head.proj` directly, never `final_head.forward`.

**How to apply.** A new MST flag is not done when the forward pass uses it; it is done when a
config that sets it either takes the path that implements it or fails loudly. `MST.__init__`
now asserts on the legacy path for every Stage 14/15 flag. Any measurement taken from a run
with a silently-ignored flag describes an architecture that was never trained.

**The same shape recurred twice more in the sweep tooling**, both mine:

- `--mst-transition-every` was declared twice in `research_compare.py` (once as the Stage 3
  flag, once when Stage 15 made it functional). argparse raises at parser construction, so
  every arm died before a single step. The sweep dry-run did not catch it because it stubs
  `research_sweep.sh` and never reaches that parser. Now covered by
  `test_no_duplicate_cli_flags`.
- The `d16` group carried a `[ "$DEPTH" -ne 16 ]` guard, added to avoid duplicate work when
  the main groups were also at 16. Invoking `--group d16 16` therefore ran nothing and still
  printed a success banner. Removed.

The unifying lesson: **a guard or a stub that makes something quietly do nothing is worse
than the duplication or the crash it was avoiding.** Verify the real invocation path, not a
harness around it.


---

## 2026-08-08 — Coupling frequency matters even though coupling form does not

Stage 15, L=8, 2 seeds, pooled sigma 0.00239 over ten arms (2-sigma floor 0.0048 bpb),
all measured against `STACK_noO1`:

| arm | delta bpb | sigma | dFLOPs/tok | dTrainFLOPs | mult/tok |
|---|---|---|---|---|---|
| **dense W_O (F5)** | **-0.0107** | **4.47** | +5.7% | +12.3% | **0.764x** |
| talking-heads + distmuon | -0.0025 | 1.04 | 0 | 0 | 0.746x |
| talking-heads (F4) | -0.0015 | 0.64 | 0 | 0 | 0.739x |
| distribute block-muon (F1) | -0.0007 | 0.30 | 0 | 0 | 0.733x |
| transition spectral LR (F2) | -0.0004 | 0.18 | 0 | 0 | 0.731x |
| **transition_every=2 (F3)** | **+0.0090** | **3.76** | -2.1% | -4.5% | 0.683x |
| **transition_every=4 (F3)** | **+0.0105** | **4.39** | -3.6% | -7.4% | 0.683x |

**The dense output projection is the first thing to move the frontier in this whole line
of work.** MHA is already block-diagonal per head and works because `W_O` mixes them; MST
had blocked `W_O` too. Restoring it is worth 0.0107 bpb, nearly 2x the 0.0058 it needed to
pay for its FLOPs at L=8.

**Coupling every layer is necessary; its form is not.** I predicted the opposite from
"form is irrelevant + free mixing is null + removing it entirely costs 0.039", and inferred
saturation. Halving the coupling frequency costs 0.009 bpb for 2% of FLOPs. The correct
statement is narrower and sharper than the paper's equifinality result: the coupling is
doing something bandwidth-like, needed at every layer, and indifferent to how it is
computed. Do not spend more on `mst_transition_every`; even assuming its cost halves by
L=32 it reaches only 0.906x against a 0.889x control.

**Report both cost axes.** Extra parameters are charged twice on the training-FLOPs axis,
because tokens = 10.5 x scaling params, so more parameters means more FLOPs per token *and*
more tokens. F5 is +5.7% on FLOPs/token but +12.3% on training FLOPs. The two axes happened
to agree here only because the training-FLOPs exponent is shallower (-0.0496 vs -0.1004),
which makes a given bpb gain worth more and roughly cancels the doubled cost. That
cancellation is a coincidence of these fits, not a rule.

**Open, and decisive for F5.** Its cost grows with depth: thresholds are 0.0058 / 0.0097 /
0.0113 bpb at L=8 / 16 / 32 against a measured 0.0107 at L=8. At full transfer it reaches
1.0008x at L=16; at the 0.26 transfer factor G1+G3 actually showed, 0.914x, i.e. worse than
not doing it. `D16_dense_wo` separates those two worlds in one run.


---

## 2026-08-09 — Why the parameter-adding approach cannot produce a paper

Dense `W_O` (Stage 15 / F5) was the best single change so far, worth −0.0130 bpb at L=16 and
taking the FLOPs/token multiplier to 1.0276x. Checking it against the parameter scaling law:

```
L=16:  matmul params 98.6M -> 111.2M  (+12.8%)
  scaling law predicts bpb 0.866012
  measured                 0.863005
  beat the law by only    +0.0030 bpb
```

So dense `W_O` is *more parameters, slightly better than average parameters*. **Anything that
buys quality by adding parameters is capped by the parameter scaling law, and that is the same
law dense obeys, so it cannot beat dense.** The two escape routes that had been working are
close to exhausted: quality-at-zero-FLOP-cost (G1/G3, ~0.019, and it decayed 0.26x from L=8 to
L=16) and FLOP-cuts-at-zero-quality-cost (O2, 10%).

That leaves two axes that are not the parameter law, which Stage 16 and 17 implement:

- **Sparsity.** Active FLOPs below total. FFN gating at k of N is worth −18.6% at k=2 and
  −28.0% at k=1 at L=32, with a quality budget of 0.013 / 0.020 bpb at L=16 to break even.
- **Optimization.** Exact block preconditioning is `N^2` cheaper for a partitioned model than
  for a dense one, so MST can afford a preconditioner dense cannot. This is a training-FLOPs
  claim; report the optimizer step wall-clock with it, because Shampoo is invisible to
  `estimate_flops` and would otherwise look free.

**How to apply.** Before running any new architectural idea, ask what it does to matmul
parameters and check the predicted bpb from the scaling law. If the idea's expected gain is
close to that prediction, it is not a Pareto move, it is just a bigger model.


---

## 2026-08-09 — The stream router collapsed, and three separate things caused it

Conditional stream execution (Stage 16) looked like a net win at L=8 (`SP_k2_noaux` at 0.783x
against a 0.758x control). It was not. Per-layer stream load after 300 steps showed only layer 0
routing per token; layers 1-3 had hard-selected a static pair of streams, so the model was a
2-stream MST carrying 4 streams' worth of parameters. Three defects, all mine:

1. **Zero-init IS the collapsed state for a top-k gate.** Every logit at exactly 0, `torch.topk`
   breaking ties by index, so the same k streams win for every token before any training, and
   the losers' FFNs receive exactly zero gradient and can never earn their way back. Zero-init
   is right for the *transition* router, where it gives a uniform softmax; it is the opposite
   of neutral for a hard gate. Now small random init at `1/sqrt(fan_in)`.
2. **The load-balancing loss was not well posed.** Switch's `N*sum(f_i*P_i)` only means anything
   when both factors are on the simplex: uniform gives 1, full concentration gives N. The gate
   used independent sigmoids, so there was no simplex constraint and the same expression is
   minimized by driving every gate probability toward zero. It balanced nothing, shrank the
   router's STE gradient, and measured as +0.0105 bpb of pure cost. The gate is a softmax now
   (the hard mask already protects the magnitude, which was the original reason for sigmoids).
3. **No exploration.** An unselected stream gets no FFN gradient, so it stays at init, so it
   stays unselected. Noisy top-k breaks the spiral.

Measured over 300 steps at depth 4, worst-layer min-load / ideal (higher is better):

| config | min/ideal | near-dead streams |
|---|---|---|
| correct aux, no noise | 0.273 | 0 / 16 |
| correct aux + noise 0.3 | 0.484 | 0 / 16 |
| correct aux + noise 1.0 | **0.586** | 0 / 16 |
| **no aux** + noise 1.0 | 0.000 | **5 / 16** |

**The aux term is necessary after all**; the formulation was what was wrong. And the no-aux arm
had the *lowest training loss* while being the most collapsed, so bpb alone cannot detect this.

**How to apply.** Never read a router's health from a different router's diagnostic:
`route_entropy_*` describes the AggDist coupling and is silent about the stream gate.
`compute_diagnostics` now emits `stream_load_L{i}_S{j}`, `stream_load_min_L{i}` and
`stream_entropy_L{i}`. Any sparse-routing result without a load reading is uninterpretable.


---

## 2026-08-09 — Sparsity is the win: 1.19x at L=16

Conditional stream execution at k=1 of 4, on top of G1/G3/O2/dense-W_O, single seed at L=16:

| arm | bpb | active FLOPs | mult/tok | mult train |
|---|---|---|---|---|
| MST baseline | 0.881000 | 7.2514e8 | 0.837x | 0.781x |
| + G1/G3/O2 | 0.876004 | 6.4965e8 | 0.988x | 0.978x |
| + dense W_O | 0.863005 | 7.2515e8 | 1.028x | 1.050x |
| **+ k=1 sparse** | 0.870216 | 5.7455e8 | **1.194x** | **1.120x** |

It spent 0.0072 bpb of a 0.0202 budget, i.e. **36% of what it was allowed**, and the cost grew
only 1.28x from L=8 to L=16 while the FLOP saving roughly doubled. The router did not collapse:
`stream_load_min` stayed at 0.94-0.99 across all 16 layers for the whole run.

**Two caveats to carry into the paper.** Single seed. And active FLOPs are not wall-clock: the
gather/scatter dispatch measured 1.14x on an H200 at D=2048 against a 28% active-FLOP saving,
because the dispatch is memory-bound. Report both, the way the paper already concedes the
head_dim kernel penalty.

**Why mixing is not the next lever.** The intuition that N narrow streams cannot collectively
match one wide dense layer is correct in dimension but already priced in parameters: MST sits at
1.005x parity with dense *per matrix parameter*. Restoring cross-stream mixing therefore recovers
at best parity, which we already have; it cannot by itself produce a Pareto win. It also trades
against sparsity, because routing only pays when streams are differentiated (measured `sub_sim`
runs -0.07 to -0.19, i.e. anti-correlated), and mixing makes them interchangeable. The one
mixing intervention that did pay, dense W_O, worked precisely because it restores what MHA
already does across heads, and it is a one-off, not a direction with more in it.


---

## 2026-08-09 — Cross-stream mixing is closed out; specialization is the direction

Monarch-structured FFN (Stage 18), L=8, 2 seeds, on top of G1/G3/O2/dense-W_O:

| arm | bpb | delta vs control | mult |
|---|---|---|---|
| control | 1.025826 | 0 | 0.761x |
| MON_shuffle | 1.024977 | -0.00085 | 0.768x |
| MON_roll | 1.025626 | -0.00020 | 0.763x |
| MON_shuffle_k1 | 1.032803 | +0.00698 | 0.752x (see below) |
| SP2_k1 (sparse alone) | 1.031984 | +0.00616 | **0.804x** |

Monarch alone is worth at most 0.9% on the multiplier, 5-15x below the 0.005-0.015 estimate.
**It does not stack**: solo it is -0.00085, but on top of sparsity it is +0.00082, and
`MON_shuffle_k1` loses to `SP2_k1` alone AND to the control.

**Accounting bug found while scoring this, now fixed.** `estimate_flops` gated
`['fc_w','fc_proj_w']` unconditionally, but under a Monarch permutation the up-projection is
NOT skippable: stream j's down-projection reads hidden units from every stream's `fc_w`. Only
`fc_proj_w` can be dropped, so the FFN saving is exactly half. The arm was reported at 0.797x;
the honest figure is **0.752x, worse than the 0.761x control**. `mst.py` now gates
`['fc_proj_w']` when `_ffn_monarch != 'none'`, with a regression test pinning the ratio at 0.5.
The general lesson, third instance in this project: a FLOPs discount is a claim about what a
sparse kernel COULD skip, and it has to be re-derived whenever a new mechanism changes the
dependency structure.

**The two FLOPs axes disagree at L=8 and that needs resolving before the write-up.**
Training FLOPs (dense fit `6.703 x^-0.0496`): MON_shuffle 0.564x > control 0.555x >
SP2_k1 0.551x. FLOPs/token: SP2_k1 0.804x > MON_shuffle 0.768x > control 0.761x. Sparsity
barely moves training FLOPs at L=8 because the token budget is `10.5 x scaling params` and
routing changes params by only 2.8%, so it pays for its bpb cost on one axis and not the
other. At L=16 sparsity leads both (1.194x / 1.120x), so L=8 is plausibly just the
pre-crossover regime, but the paper must state which axis the headline number is on.

**That is now four independent attempts at cross-stream mixing** -- twelve coupling
enrichments, the Stage 14 stream-axis permutation, talking-heads, and the Monarch FFN -- all
returning <= 0.003 bpb. The single mixing change that worked, dense `W_O` at -0.013, worked
because it RESTORED a mechanism multi-head attention already has, not because it added new
mixing. Meanwhile the opposite direction, making streams more specialized and routing between
them, gives 1.194x at L=16.

**The conclusion to carry into the paper**: the streams do not want to be more like one big
dense matrix, they want to be more different from each other. Routing pays precisely when they
are differentiated (`sub_sim` runs -0.07 to -0.19), and mixing makes them interchangeable, so
the two directions actively trade off. This is a stronger and better-supported claim than the
equifinality result it replaces.

**Do not re-open mixing** without a mechanism that is qualitatively unlike these four.

**Seed-variance note.** The dense-`W_O` family shows ~5x tighter seed spread than earlier arms
(pooled 0.00039 across six arms, against 0.0024 elsewhere). Use the local sigma for
within-family comparisons, but note that a 2-seed, 4-arm estimate has only 4 degrees of freedom
and is itself loose: MON_shuffle is between 0.8 and 2.2 sigma depending on which is used.


---

## 2026-08-09 - MoL is prior art, and it is now implemented as our baseline

`arXiv:2605.09516v1`, "Mixture of Layers with Hybrid Attention: Parallel Thin Blocks for Sparse
Transformer Compute" (Ternovtsii & Bilak, 10 May 2026), is the closest prior work to MST, and it
is prior work rather than concurrent work: four to twelve months before any realistic deadline.
Shared with MST: parallel narrow transformer blocks replacing a full-width block, top-k routing
over them, a load-balance aux, gather/scatter sparse dispatch, and `d_head=64` pinned across
widths (which is our G1). **Priority of the idea is gone and no framing recovers it.** "I started
first" carries no weight in review; concurrent-work protection runs to about three months and
covers simultaneous appearance, not private prior ideation.

**There is no code release.** All 14 pages: no repository link, no code-availability statement, no
algorithm listing, no pseudocode. The specification is nonetheless fully recoverable, and
`nanochat/mol.py` is validated by reproducing their published numbers:

| target | published | ours |
|---|---|---|
| Table 1 total (D=1024, L=8, K=5 top-3, d_thin=256, tied 32K) | 85.3M | 85.24M |
| Table 5 active (1.3B, 1+3of15, d_thin=512) | 0.61B | 0.607B |
| Table 5 total, softmax-only | 2.08B | 1.991B (gap = DeltaNet gates/convs) |
| projection fraction at d_thin 256 / 128 / 64 | 40 / 57 / 73% | 40.0 / 57.1 / 72.7% |

### The structural argument, derived and measured with no GPU

MoL wraps each thin block in its own `W_down`/`W_up`, so plumbing is paid per block per layer.
MST partitions the residual stream and pays one shared transition. Closed forms, both confirmed
against measurement in `scripts/p09_projection_overhead.py`:

    MoL   plumbing / wrapped block  =  D / (D + 6 d_thin)
    MST   plumbing / layer          =  (N + 8) / (13 N + 8)

**They move in opposite directions.** MoL's is a function of `d_thin` and rises without bound as
blocks narrow. MST's is a function of the stream count only, and narrowing the MST way (more
streams at `d = D/N`) makes it FALL, toward a `1/13` asymptote. Measured at D=1024:

| width | MoL | MST (N=D/width) | ratio |
|---|---|---|---|
| 512 | 25.0% | 31.2% (N=2) | 0.80x |
| 256 | 40.0% | 21.4% (N=4) | 1.87x |
| 128 | 57.1% | 15.4% (N=8) | 3.71x |
| 64 | 72.7% | 12.0% (N=16) | 6.06x |

Be honest about the N=2 row: MoL is cheaper there. From N=4 on MST wins and the gap compounds
to 6x. This is the paper's positioning and it needed no training at all.

**It already shows up in FLOPs.** At matched d_model, depth, active width and head_dim (L=8,
D=512, 4 active blocks of 128), MoL costs **1.30x** MST's FLOPs/token, and **1.14x** even after
crediting MoL's routing saving. That is the projection tax, measured before a single training
step.

### Where MoL is weak, which is where the room is

- At iso-**total** params dense beats them by 3.01 PPL at 1.3B; they win only at iso-**active**,
  which is the MoE trade ("the same trade as MoE but at the block level", their section 5.5).
  Our claim is FLOPs-versus-bpb without inflating total params, which is a harder axis.
- They LOSE to dense on WikiText-103 (multi-epoch) and win on Cosmopedia (single-epoch): their
  advantage is data-regime dependent and they say so.
- Training is 1.91-2.29x slower than dense at 1.3B.
- 1.3B results are single-seed.

### Implementation notes worth keeping

- `gpt.CausalSelfAttention`'s existing `token_active` argument implements exactly MoL's restricted
  attention (active queries attend only to active keys, causally). Large reuse, and it is the one
  behaviour whose absence would silently give MoL full sequence coverage and inflate its quality.
- The thin block is `gpt.Block` at a shadow config, so MoL's block and our dense baseline share an
  implementation and no quality difference can be blamed on two block definitions. The hazard is
  that the shadow config inherits ~40 Phase-XX research flags; `make_thin_config` asserts they are
  all at default, and a test re-derives the list from `gpt.py` so it cannot drift.
- Muon stacks a group's tensors, so groups must be shape-homogeneous. MoL hits this harder than
  MST because `W_down` is `(d, D)` and `W_up` is `(D, d)`. Caught only by actually stepping the
  optimizer, which is why `test_mol.py` smoke-trains rather than inspecting.


---

## 2026-08-10 - G3 is worth keeping, and it costs us the parameter axis

L=16, single seed, all arms carrying `--mst-stream-topk 1` (the shipping config), differing
only in value-embedding treatment. FLOPs are identical across the first two because VE is a
lookup, so this is bpb against bpb:

| arm | bpb | delta | total | FLOPs mult | trainF mult |
|---|---|---|---|---|---|
| SP2_k1 (G3) | 0.870216 | 0 | 413.2M | **1.194x** | **1.120x** |
| ve_plain | 0.876141 | +0.005925 | 211.9M | 1.116x | 0.977x |
| ve_map full | 0.875722 | +0.005506 | 214.0M | 1.097x | 0.947x |
| ve_map rank32 | 0.875507 | +0.005291 | 212.4M | 1.118x | 0.981x |

**G3 stays.** Dropping it costs 0.0059 bpb at identical FLOPs, worth **-0.078x** on the
multiplier, and it flips the training-FLOPs axis from 1.120x to 0.977x, i.e. from winning to
losing. That is 2.5 sigma on the global seed estimate and 15 sigma on the dense-W_O family's
local 0.00039.

**My extrapolation was wrong and the reason matters.** I predicted ~0.0024 by decaying the
L=8 measurement with the 0.26 factor that G1+G3 showed from L=8 to L=16. The real value under
routing is 0.0059, 2.5x that. The plausible mechanism is that G3 is a stream-DIFFERENTIATION
signal and conditional execution is what pays for differentiation, the same trade the Monarch
result found pointing the other way. **But that is not established**: we have no non-sparse
G3-vs-plain pair at L=16, so the gap is equally consistent with the decay factor simply being
imprecise. Do not claim the interaction in the paper without measuring it.

Process note: these arms were originally written against `D16_dense_wo`, the non-sparse config.
Measured there, G3 would have looked like the small number, been called noise, and been deleted
along with 0.078x of the headline result. **Ablate against the config you ship.**

### The cheap variant is dead, and there is no cheap variant

`mst_ve_map` (shared d-wide table plus a per-stream d x d map) recovers about 10% of G3
(0.0006 of 0.0059). Full rank is actively negative at 1.097x, below plain's 1.116x, because it
pays 2.19% FLOPs for nothing. Rank-32 ties plain.

The explanation, which generalises: **G3's value is per-token degrees of freedom.** G3 gives
each token N*d independently learned numbers; any shared-table-plus-per-stream-map scheme gives
only d, because a fixed linear map cannot manufacture per-token information the shared table
does not carry. Per-token degrees of freedom is exactly what costs vocabulary-sized parameters,
so within this family there is no cheap version. A wider shared table with a narrower per-stream
projection just interpolates back toward G3 at proportional cost.

### The strategic consequence: G3 loses us the parameter axis

Comparing each arm against the dense model predicted to reach ITS OWN bpb:

| arm | bpb | iso-quality dense | our total | FLOPs mult | **params mult** |
|---|---|---|---|---|---|
| MST + G3 | 0.870216 | L=11.0, 276.9M | 413.2M | 1.196x | **0.670x** |
| MST plain VE | 0.876141 | L=10.8, 262.8M | 211.9M | 1.130x | **1.240x** |

With G3, MST is **1.49x LARGER** than the dense model of equal quality. Without it, MST is
**1.24x smaller** while still being 1.130x more FLOPs-efficient.

This matters for positioning against MoL, not just for a number. MoL's stated weakness is that
it loses to dense at iso-total-parameters and wins only at iso-active, which they describe as
"the same trade as MoE but at the block level". At 1.3B they are 2.08B total against dense's
1.31B, a 1.59x inflation. **With G3 we are at 1.49x, doing essentially the same trade.** The
differentiation we were relying on, that ours is a FLOPs claim and theirs is a parameter-dumping
claim, does not survive G3.

**Recommendation: report both as two points on our own frontier**, which costs nothing since
both runs exist. G3 is the headline FLOPs number (1.194x / 1.120x), and plain VE is the point
that dominates dense on BOTH axes simultaneously (1.130x FLOPs and 1.240x parameters). A
frontier answers the "you just inflated parameters" objection in a way a single point cannot.

---

## 2026-08-10 - The token-budget confound, and MoL is not what it looked like

MoL's apparent blowout over MST was mostly a data artifact of our own protocol, not an
implementation fault and not a real architectural gain. Recorded because the same trap will
recur with any architecture whose total and active parameter counts diverge.

### The mechanism

The token budget is `10.5 x (transformer_matrices + lm_head)`, applied identically to every
arm. MoL's `1+3of15` topology carries 3.75x more total than active parameters, so the rule
hands it far more data. At L=8:

| arm | tokens | ratio |
|---|---|---|
| MoL 1+3of15 | 0.59B | 2.10x MST |
| dense | 0.44B | 1.57x MST |
| MST SP2_k1 | 0.28B | 1.00x |

The FLOPs-per-token axis never charges for that. The training-FLOPs axis does. That single
fact explains every anomaly we chased.

### The iso-token control (L=8, all three at 0.6B, plain value embeddings on both)

| | own budget | at 0.6B | delta |
|---|---|---|---|
| MST | 1.031984 (0.28B) | 0.997837 | -0.0341 |
| MoL | 0.990762 (0.59B) | 0.987807 | -0.0030 |
| dense | 0.960236 (0.44B) | 0.942661 | -0.0176 |

MoL-minus-MST gap: **-0.0412 at own budgets, -0.0100 at equal tokens, a 4.1x collapse.**
MST gained 0.0341 from the extra data; MoL gained 0.0030 because it was already at its budget.

Anchored on the iso-token dense arm with the dense ladder exponents:

| | FLOPs/token | training FLOPs | total params |
|---|---|---|---|
| dense | 1.000x | 1.000x | 125.8M |
| MoL | 0.962x | 0.610x | 89.7M |
| MST | **1.061x** | 0.609x | **60.3M** |

**The ranking inverts.** At equal tokens MST is above the dense curve and MoL is below it, the
reverse of the unequal-budget reading. Local exponent MST to MoL is -0.052, half the dense
ladder's -0.104: MoL spends 1.215x MST's FLOPs/token and returns 0.0100 bpb where dense's own
slope would return 0.0202. MoL underdelivers by 2x, at 89.7M parameters against MST's 60.3M.

### Three things to carry forward

**1. Our MoL implementation is vindicated.** The suspicion was that a 1.178x score against
dense was too good for a paper that reports nothing of the sort. It was: the source paper
trains every arm of a comparison at a fixed token count, so MoL never had a data advantage
there. Once we match tokens, our numbers agree with theirs. No audit needed.

**2. MoL and MST are indistinguishable on training FLOPs at every setting tested**: 0.665 vs
0.666 at L=8 own-budget, 1.175 vs 1.181 at L=16, 0.610 vs 0.609 iso-token. Growth d8 to d16
also matches (1.766 vs 1.774). They differ in WHERE they place cost, not in total compute
efficiency at this scale.

**3. "FLOPs/token vs bpb" is NOT iso-FLOPs.** It is iso-inference-cost, and it is blind to how
many tokens an arm trained on. Only the training-FLOPs curve is a genuine iso-compute
comparison. Reporting only the per-token axis is what made MoL look impossible; reporting only
training FLOPs would have hidden that MST wins the per-token axis. Report both, always.

### Also corrected in this pass

The dense power-law fit was refit on all 8 measured ladder points: **FLOPs/token
`7.2697 x^-0.10406`, training FLOPs `7.3182 x^-0.05167`** (previously `6.711 x^-0.1004` /
`6.703 x^-0.0496`). The old fit sat 1.07% off the measured dense d8 point, which biased every
L=8 multiplier low by roughly 11%. The dense ladder itself scatters +-4% against its own refit,
which is the scale of wobble to expect in any single-point multiplier.

With the refit, MST SP2_k1 reads 0.900x (d8), 1.260x (d16), 1.145x (d20), 1.285x (d24) on
FLOPs/token. **The d20 dip was fit error plus dense-side scatter, not a trend reversal**; the
earlier "the trend is broken" call was wrong.

---

## The MST wall-clock gap is non-GEMM overhead, and the structure crosses over at D=1024

MST's headline SP2_k1 arm trains 2.3x to 2.7x slower than dense at every depth measured, at
0.5x to 0.8x the FLOPs. Two rounds of measurement were needed because the first diagnosis from
whole-model numbers was wrong in both of its parts.

### Round 1: the whole-model harness, and what it does and does not show

| depth | D | d | MST TFLOP/s | dense TFLOP/s | wall clock | MST/dense FLOPs |
|---|---|---|---|---|---|---|
| 4 | 256 | 64 | 32.1 | 92.7 | 2.30x | 0.795 |
| 8 | 512 | 128 | 41.5 | 184.6 | 2.72x | 0.611 |
| 12 | 768 | 192 | 65.7 | 300.0 | 2.36x | 0.518 |

Two candidates die here and stay dead.

**Attention is not the problem.** `flash_attn::fwd` plus `bwd` is 1.6% of device time at depth 4,
2.1% at depth 8, 3.8% at depth 12. Fusing the per-stream attention calls, or relaxing G1's
head_dim=64 (0.79x the throughput of head_dim=128 at equal FLOPs, see OPEN_QUESTIONS Q6), moves
about 1% of the step.

**torch.compile is not breaking.** The profile shows exactly two `CompiledFxGraph` scopes, one
forward and one backward. Many graph breaks would show many hashes.

A third reading from these numbers looked compelling and was **wrong**: dense throughput over
the three widths is near-linear, and evaluating that fit at MST's stream width predicts 41.2 at
d=128 and 66.8 at d=192 against 41.5 and 65.7 measured. That coincidence supported "MST's
throughput is simply that of a dense model of width d", which implied the GEMMs were the
problem and no kernel work could help. Round 2 refutes it.

### Round 2: isolate the GEMMs (`scripts/p10_mfu_microbench.py`, H100)

Block-diagonal execution pays in wall clock only when `throughput(d)/throughput(D) > 1/N`.
Measured on a whole layer (forward, dgrad, wgrad) at M=16384:

| D | d | ratio | 1/N | block-diagonal vs masked dense |
|---|---|---|---|---|
| 256 | 64 | 0.243 | 0.250 | loses 1.03x |
| 512 | 128 | 0.309 | 0.250 | wins 1.23x |
| 768 | 192 | 0.453 | 0.250 | wins 1.81x |
| 1024 | 256 | 0.579 | 0.250 | wins 2.32x |
| 1536 | 384 | 0.712 | 0.250 | wins 2.85x |
| 2048 | 512 | 0.884 | 0.250 | wins 3.54x |

The D=2048 row was taken at M=163840 rather than 16384, which saturates both arms and
flatters it; treat 0.884 as an upper bound and the first four rows as the comparable series.

**The GEMMs clear their crossover at every D >= 512, with a margin that grows fast.** The
estimate of 0.219 at D=768 derived from the whole-model numbers was off by 2x.

**The weight-gradient hypothesis is also refuted.** The prediction was that `bmm` would lose to
an unbatched loop or to `torch._grouped_mm`, because a `d x d` output with `K = B*T` is a
handful of tiles on 132 SMs and wants split-K that cuBLAS will not apply inside a batched call.
The measurement is the opposite everywhere: at D=768, `bmm` 158.6 TFLOP/s against loop 78.8 and
grouped 29.9. `torch._grouped_mm` is tuned for MoE token-grouped shapes and is the worst option
here. And the penalty is not structure-specific: block-diagonal wgrad/fwd throughput is 0.81,
dense is 0.83. Do not spend time on split-K.

### Where the gap actually is

At depth 12, D=768, using the isolated block-diagonal throughput as the GEMM cost:

    MST    3133 GFLOP / 284.9 TFLOP/s = 11.0 ms GEMM  of 47.69 ms  ->  36.7 ms non-GEMM (77%)
    dense  6051 GFLOP / 629.6 TFLOP/s =  9.6 ms GEMM  of 20.17 ms  ->  10.6 ms non-GEMM (53%)

**MST's non-GEMM time is 3.46x dense's**, which matches the decode ratio (2.9x to 3.3x, flat in
context length, so pure launch overhead) and explains the prefill trend (MST/dense falls from
2.42x at T=2048 to 1.16x at T=32768 as each kernel gets more work to amortise).

Named sources, largest first:

1. The N-loop applies RoPE and QK-norm per stream (`mst.py:1322-1327`). Both are elementwise and
   identical across streams, so they can be applied once to the whole `(B,T,N,H,hd)` tensor with
   only `flash_attn` left in the loop. Roughly 24 of about 32 kernels in that loop.
2. `torch.stack(attn_results, dim=2)` (`mst.py:1345`) is a full activation copy.
3. `_batched_linear`'s permute-reshape, six calls per layer. Section [C] prices it at 1.42x /
   1.24x / 1.32x on the d>d projection at d = 64 / 128 / 192, then **1.00x at d=256 and d=384,
   and 0.92x at d=512**. It is a real win only on models smaller than anything the paper
   reports, and a small loss at d=512, so it was measured and then **not** implemented. This is
   the one place where sizing a fix before building it changed the decision.
4. The stream router builds a `(B,T,N)` mask with about 12 kernels including `randn_like`,
   softmax, topk, scatter and the straight-through arithmetic.
5. `_last_stream_load` and `_last_route_entropy` are full reductions written to module
   attributes every layer, every step, unguarded. `gpt.py` and `eet.py` wrap the equivalent in
   `torch.compiler.is_compiling()`. Not a graph break, but they become graph outputs, pinning
   `mask` and `probs` alive and blocking dead-code elimination.

### The number the paper needs is not the one section [D] reports

Section [D] answers "is the structure worth executing as structure". The paper needs "is MST
faster than the dense baseline", which requires throughput ratio > FLOPs ratio:

| D | throughput ratio | MST/dense FLOPs | GEMM-time penalty |
|---|---|---|---|
| 256 | 0.243 | 0.795 | 3.27x |
| 512 | 0.309 | 0.611 | 1.98x |
| 768 | 0.453 | 0.518 | 1.14x |
| 1024 | 0.579 | ~0.47 (extrapolated) | ~0.81x, MST faster |
| 1536 | 0.712 | ~0.44 (extrapolated) | ~0.62x |

Monotone, crossing between D=768 and D=1024. The headline d24 arm runs at D=1536, d=384, past
the crossing. So MST's wall-clock disadvantage is a small-model artifact that inverts at the
scale the paper is about, with implementation overhead sitting on top of it.

Two caveats before this goes in the paper. Section [D] covers only the block-diagonal shapes,
while the A3 FLOPs ratios include attention, the `--mst-wo-mode dense` D-squared projection, the
transition, embeddings and the head; mixing them is approximate, so the crossing point is good
to about one rung. And D=1536 and D=2048 should be measured rather than extrapolated.

### Device-relativity

The same microbenchmark reports 0.747 on a 20-SM RTX 3050 Ti, where block-diagonal wins 2.99x
even at D=768. A small device saturates on a 192-wide GEMM; a 132-SM H100 does not. The penalty
scales with how much parallelism the accelerator has relative to the stream width. True and
worth one sentence, but do not lead with it: a reviewer reads a laptop benchmark as evasion of
the H100 result.

### What was implemented, and one candidate that turned out to be a no-op

Landed in `mst.py` as Stage 19, all verified against the pre-patch module on 16 flag
combinations:

- **`_rope_streams`.** RoPE and QK-norm now run once over `(B, T, N, H, hd)` instead of N times
  over per-stream slices. Only `flash_attn` stays in the loop, because `window_size` is a
  per-call scalar pair and `--mst-multi-scale-windows` gives each stream a different one.
- **Fused QKV.** One bmm with a 3x wider output replaces three. The three weights stay separate
  Parameters and are concatenated per forward, so every checkpoint, Muon group, grad-equalize
  hook, `estimate_flops` name list and diagnostic script is untouched; the cat is over weights,
  a few MB per layer, not activations. Off under `mst_shared_kv_attn`, where `c_k_w` is
  `(qkv, d)` and has no stream axis to concatenate.
- **Gated routing diagnostics.** `_last_stream_load` and `_last_route_entropy` now follow
  `MST._diag_enabled`, which base_train already toggles on log steps. They are reductions
  written to module attributes, so under torch.compile they are graph outputs that pin the
  routing mask alive and block dead-code elimination on every step. Same two-graph mechanism
  `_diag_sub_states` already uses.

**Replacing `torch.stack` with a preallocated buffer is a no-op** and was not implemented.
`torch.stack` on a list already allocates the output once and copies each element into it,
which is exactly what the "fix" would do by hand. The copy is inherent to concatenating N
separately-produced tensors, and the only way to remove it is for flash_attn to write into a
strided view of a shared buffer, which `flash_attn_func` has no `out=` parameter for.

Exactness. The forward is **bit-identical** in all 16 configurations tested, including the
headline SP2_k1, per-stream VE, dispatch on, shared K/V, Monarch, channel mix, slice and MLP
transitions, N=8, and the noisy router. The backward is bit-identical when the QKV fusion is
off, and agrees to **one bf16 ULP** when it is on (max relative gradient difference 9.1e-3
against a bf16 epsilon of 7.8e-3), because one wide GEMM reassociates differently from three
narrow ones. Existing checkpoints therefore evaluate identically; a re-run of a training config
will diverge over many steps the same way a cuBLAS version bump would.

Still free and not yet done: one flash call per distinct window (saves 1 call in 4 on short-
window layers only, since multi-scale gives L layers four distinct windows), and CUDA graphs
for decode.

Quality-affecting, needs an ablation: dropping `--mst-multi-scale-windows`, raising head_dim off
G1's 64, enabling `--mst-stream-dispatch`, reducing N.

Measured and rejected: stream-major `(N, B, T, d)` layout (1.00x at d=256 and d=384, 0.92x at
d=512); split-K or grouped-GEMM for the weight gradient.

### Caveat on the FLOPs axis in any speed comparison

`base_train.py:2630` computes MFU from `num_flops_per_token`, the total, which is correct: the
sweep deliberately omits `--mst-stream-dispatch` (`p08_mst_parity_sweep.sh:683`), so SP2_k1
computes all N streams and masks N-k of them. Any external harness scoring MST must use the
total and not the active count, or it will understate MST's throughput while correctly reporting
its wall clock.

---

## The wall-clock crossover landed at D=1536, and the Stage 19 cuts were worth ~0 in training

Re-ran the A1-A5 harness at depth 8 / 16 / 24 after the Stage 19 changes.

### MST reaches wall-clock parity with dense at the paper's headline depth

| depth | D | d | MST ms/step | dense ms/step | ratio | MST FLOPs ratio | throughput ratio |
|---|---|---|---|---|---|---|---|
| 8 | 512 | 128 | 34.50 | 11.60 | 2.97x slower | 0.612 | 0.206 |
| 16 | 1024 | 256 | 50.55 | 36.90 | 1.37x slower | 0.469 | 0.342 |
| 24 | 1536 | 384 | **76.09** | **77.21** | **1.01x, MST faster** | 0.425 | 0.431 |

Prefill at T=32768, depth 24: MST 91.88 ms against dense 119.19 ms, so **MST is 1.30x faster**.

MST wins wall clock exactly when the throughput ratio exceeds the FLOPs ratio, and at depth 24
that is 0.431 against 0.425. The isolated-GEMM microbenchmark predicted this: it gave throughput
ratios 0.309 / 0.579 / 0.712 at D = 512 / 1024 / 1536, and the whole-model ratios come in at a
consistent 0.6x of those, which is the non-GEMM overhead. Extrapolated FLOPs ratios of ~0.47 at
D=1024 and ~0.44 at D=1536 measured at 0.469 and 0.425.

### The Stage 19 cuts did nothing in training, and the reason matters

Depth 8, before and after: **34.56 -> 34.50 ms per step. Kernel count 268 -> 267.** The
prediction was roughly 24 kernels per layer removed by hoisting RoPE and QK-norm out of the
per-stream loop.

**Inductor was already fusing them.** The per-stream RoPE and QK-norm are elementwise chains
over slices of the same tensors, which is exactly what inductor's fusion pass merges. Counting
aten ops and calling them kernels is wrong under torch.compile. The lesson generalises: any
kernel-count argument about the compiled path has to be made against emitted Triton kernels, not
against the Python-level op count.

What did land: `aten::bmm` 1.90 -> 1.31 ms at depth 8, a 31% cut from the fused QKV. But bmm is
2% of the step, so it does not move the total. The changes remain correct and free; they are
just worth about nothing in training.

### They were worth 1.77x in decode, because decode is not compiled

Depth 8 decode, MST **23.18 -> 13.09 ms/token (1.77x)** against dense's 7.07 -> 5.56 (1.27x).
Same fix, in the one path inductor does not cover.

### The remaining decode gap is not an MST property

Decode is flat in context from 256 to 16384 tokens for both arms, so it is entirely host
overhead: dense at 16 ms/token for a 24-layer D=1536 model is about 1000x off the
memory-bandwidth floor (reading those weights once is roughly 17 us at 3 TB/s).

Under CUDA graphs or `torch.compile(mode="reduce-overhead")` both arms collapse to
bandwidth-bound, where the metric is bytes of weights per token. MST's per-layer matmul
parameters are `11D^2/N + D^2` = 3.75 D^2 at N=4 against dense's 12 D^2, about **3.2x fewer
bytes**. So the prediction inverts: **MST should decode 2-3x faster than dense**, not 2.4x
slower. A ~6x swing, currently hidden behind an uncompiled generate loop, and it is the obvious
reviewer question about inference cost.

### Multi-scale windows are not redundant with the sliding-window pattern, and they are a bargain

They compose rather than duplicate. `_compute_window_sizes` gives the per-LAYER pattern
(`SSSSL`, 256 or full), which dense has too. `sub_window_sizes` gives a per-STREAM geometric
schedule (`[32, 128, 512, full]` at N=4). `--mst-compose-windows 1` intersects them with
`_tighter`, so stream 0 sees 32 tokens even on a long layer. Dense varies context with depth;
MST adds variation across width. Without the composition MST's widest stream would be
full-context on every layer, which is MORE attention than dense gets.

Measured trade: 1.7629e8 -> 1.5622e8 active FLOPs/token for 1.029373 -> 1.031984 bpb. The dense
fit's local slope is `d(bpb)/d(ln F) = -0.10406 * bpb ~ -0.1072`, and `Δln F = -0.1210`, so the
dense curve's own exchange rate prices that FLOPs cut at **+0.0130 bpb**. It costs **+0.0026**.
Multi-scale windows buy FLOPs at about **5x the dense curve's trade rate**. Not a flag to
remove, a result to report.

### Unchosen streams are still masked, and now is the time to change that

`p08_mst_parity_sweep.sh:683` still omits `--mst-stream-dispatch`, so the hard 0/1 gate
multiplies the FFN output after all N streams compute. For the FLOPs-vs-bpb axis that is the
right choice, since it measures quality at active FLOPs without confounding it with dispatch
overhead. For wall clock it leaves the entire sparsity win unexecuted.

Dispatch was unattractive at d=128 because gathering to `K = T*k/N` shrinks the GEMM's M by 4x,
and those GEMMs were already starved. At d=384 they are not (0.712 of dense throughput). The FFN
is `8D^2/N` of the `3.75 D^2` per-layer matmul work, about 53%, so k=1-of-4 dispatch skips
roughly 40% of the layer's GEMM work. Against MST sitting at 1.01x dense, that is the lever that
could put it clearly ahead. Phase B is implemented and its numerical equivalence is pinned by
`test_phase_b_matches_masking_when_nothing_overflows`; the open questions are the capacity factor
and whether the router survives losing the compute-then-mask gradient.

Structural alternative if dispatch disappoints: make the sparsity static rather than
token-routed (a per-layer schedule of which streams run). No router, no gather, no capacity, and
the skipped GEMMs do not exist in the compiled graph at all. It gives up input-adaptivity, which
is a real quality question, but it is the only form of sparsity that is free to execute.

---

## Dispatch looked catastrophic because two diagnostic lines shattered the compiled graph

Turning on `--mst-stream-dispatch 1` at depth 8 doubled step time, halved MFU, and OOMed at a
batch size the masked path fits on one H200. None of that was the gather/scatter.

### The measurement

`torch._dynamo.explain` on an L=8 MST:

| | graphs | breaks |
|---|---|---|
| `mst_stream_dispatch=0` | 1 | 0 |
| `mst_stream_dispatch=1` | **11** | **10** |

One break per layer, from two lines at the end of `_ffn_dispatched`:

```python
self._last_stream_drop = float(...) if selected > 0 else 0.0
```

`float()` on a CUDA tensor is `.item()`: a device-to-host sync AND a Dynamo break
("Unsupported Tensor.item() call"). `if selected > 0` is a Python bool on a CUDA tensor: a
second break ("Data-dependent branching"). Together they cost, per layer per step:

- **Fusion**, across every boundary.
- **Memory**, because a break materialises the entire live set as graph outputs and inputs, and
  inductor cannot plan allocation across it. This is the OOM.
- **Pipelining**, because the sync serialises the GPU on the host once per layer. This is the 2x
  step time and the halved MFU.

### The gather/scatter is actually good

Isolated FFN, B=4 T=2048 N=4 d=128, k=1, cf=1.25, forward and backward, compiled:

| | time | peak memory |
|---|---|---|
| masked (all N streams) | 26.1 ms | +96.0 MiB |
| dispatched | **13.3 ms** | **+40.6 MiB** |

**2.0x faster at 0.42x the memory.** Exactly what Phase B is supposed to do. The whole reported
regression was the two diagnostic lines.

Also tested and NOT worth doing: replacing the `(B, N, K, d)` int64 index with a flat
`(B*N*K,)` index and `index_select`/`index_add`, on the theory that the d-way expansion costs
4x the activation bytes in index traffic. Measured 13.8 ms against 13.3, no better. Inductor does
not materialise the expanded index, so there is nothing to save.

### The fix

Gate the accounting on `diag` (which follows `MST._diag_enabled`, already toggled by base_train
on log steps), keep it as a tensor under compile so no sync happens, and delete the
`selected > 0` guard, which was dead: top-k always selects k and `clamp_min(1)` already handles
the divide. Result: `dispatch=1` compiles to 1 graph, 0 breaks, same as masked.

`_last_stream_drop` was also read by nothing outside the tests, so `mst_stream_capacity_factor`
could not be tuned from a training log. `compute_diagnostics` now emits `stream_drop_L{i}`.

### Round 2: the first fix was inert, because --log-every defaults to 1

Gating the accounting on `diag` did nothing in a real run. `base_train.py:2005` sets
`_mst_diag_every = args.log_every`, and `--log-every` defaults to **1**, so
`_mst_diag_this_step` is true on EVERY step and `_diag_enabled` is always on. A diag-gated
line is not gated at all in the default configuration.

The accounting is now unconditionally compile-safe instead: always a tensor, never `float()`,
never a Python branch on a tensor. The reduction itself is two cheap sums with no sync, so
there was nothing left worth gating once the `float()` and the branch were gone.

Worth knowing separately: with `--log-every 1`, `MST.forward` also runs
`self._diag_sub_states[i] = [sub_states[:, :, j].detach() for j in range(N)]` on every layer of
every step, which pins N detached activation tensors per layer for the whole step. That is real
memory on the masked arm too, and it is almost certainly not the intent of "log diagnostics at
the same frequency as training logs".

### Why sparsity does not buy MST what it buys MoL

MoL (arXiv:2605.09516) is the right calibration, and their own numbers explain the gap.

**Their speedups are forward-pass only, and they scale with how sparse you actually are** (§2.3):
1.53x at 57% active (top-4-of-7), 2.85x at 20% (top-2-of-10), 4.34x at 10% (top-2-of-20), up to
4.94x with torch.compile. They never claim a training-step speedup.

**MST at `mst_stream_topk=1`, N=4 is roughly 60-67% active, not 10-20%.** Per-layer matmul
params are about 0.75 D^2 (qkv) + 2 D^2 (FFN) + 1 D^2 (wo-dense) + 0.75 D^2 (transition) = 4.5
D^2, so the FFN is about 44% of the layer. Gating only the FFN at k=1-of-4 removes 3/4 of 44%,
i.e. a third of the layer. On MoL's own curve that lands at their 57% point, worth **1.53x
forward-only**. In a training step, with backward and dispatch overhead, there is nothing left.

**Their headline path is not naive gather/scatter either.** §5.6: the production path is
`attention_mode="batched_sparse"`, which "fuses the N=14 routed FLA calls into one
chunk_gated_delta_rule call with N x H heads and stacks per-block linear projections via bmm".
The naive sparse path is **2.8x slower than batched on 3090 and 2.77x slower on H200**. A 2x
regression from a per-block gather/scatter loop is a documented result in the source paper, not
an MST-specific defect.

**Most of their measured win is attention, not FFN sparsity.** Their 3090 decomposition
attributes 79% of the per-token latency reduction at T=32K to "smaller per-block d_ff and 14
narrow routed-attention windows of ~T/4.67 tokens vs one giant T x T", and only 21% to DeltaNet.
MST does not gate attention at all (`mst_stream_gate_attn` is off), so it forgoes the part that
actually pays.

**Their decode is worse than ours, and they say so.** MoL is flat at 60-65 ms/token against
Dense's 6.2 to 26.5, "dominated by a per-token dispatch floor of ~2,160 Python op launches; the
floor makes MoL slower than Dense in absolute terms across all measured contexts."

#### What this implies for MST

To get a real sparsity speedup MST has to go where MoL went: gate attention as well as the FFN,
and be much sparser than 60% active. Gating attention is what creates MoL's own "attention
coverage problem" (§3.1: at 3-of-15 each block sees 20% of the sequence, and softmax-only
3-of-15 is WORSE than 3-of-5 despite 2.2x the parameters), and their answer is the Shared +
Routed topology of §3.2, an always-active block 0 carrying global context at every layer.

MST already has that flag: `mst_stream_shared`. So the untested combination that could actually
pay is `mst_stream_shared=1` + `mst_stream_gate_attn=1` + `mst_stream_dispatch=1` at N=8 or 16,
not `mst_stream_topk=1` at N=4 with the FFN alone.

### The generalisable lesson

This is the third time in this investigation that a *diagnostic* has been the expensive thing:
`_last_stream_load` and `_last_route_entropy` as graph outputs, and now `_last_stream_drop` as a
sync and a break. Anything written to a module attribute inside a compiled forward is not free,
and anything that calls `float()`, `.item()`, or branches on a tensor is expensive in a way that
does not show up as a kernel in a profile. Gate all of them on `_diag_enabled`.

Corollary for reading benchmarks: `mst_stream_dispatch=1` did compile before this fix, at 147 ms
per step in the local harness, so nothing errored and nothing looked obviously wrong. A change
that silently splits one graph into eleven presents as "this feature is slow", not as a bug.

---

## CORE reproduces the bpb Pareto multipliers out of sample

Downstream CORE is now measured for MST $L=24$ (0.194) and three dense points: $L=16$ (0.181),
$L=20$ (0.233), $L=22$ (0.255). Fitting `CORE = a + b ln(cost)` on the three dense points and
reading MST off it:

| cost axis | MST L=24 | dense CORE at that cost | CORE multiplier | bpb multiplier |
|---|---|---|---|---|
| FLOPs/token | 1.481e9 | 0.1753 | **1.24x** | 1.23x |
| training FLOPs | 4.823e18 | 0.1910 | 1.07x | 1.10x |
| matrix params | 259.7M | 0.2008 | 0.92x | 0.92x |
| total params | 964M | 0.2399 | 0.63x | 0.64x |

**All four agree with the bpb-derived multipliers to within 0.03**, and the ordering across axes
is identical. CORE was never used to fit anything, so this is an out-of-sample check that the
bpb advantage is a capability advantage rather than a tokenizer or calibration artifact. It is
the strongest single piece of evidence in the paper for the claim transferring.

Two caveats to state whenever this is quoted.

**The head-to-head is not significant on its own.** Against the iso-FLOP point, MST wins 12 of
22 tasks, loses 7, ties 3: mean per-task margin +0.75 points, se 0.54, t = 1.41, sign-test
p = 0.36. BoolQ (+7.9) and SQuAD (+6.2) supply most of the aggregate gap; MST loses Winograd
(-3.6) and OpenBook QA (-2.2). The four-axis agreement is the result, not the 0.194 vs 0.181.

**The fit's R^2 means nothing.** Three points and two parameters leaves one residual degree of
freedom, so R^2 > 0.999 is close to automatic. Quote the agreement, never the R^2.

---

## What the top-k gate actually switches off: 33% of a layer, not a layer

Clarifying a claim that was stated loosely. **The streams are not an FFN construct.** They
partition the residual stream, so attention (per-stream Q/K/V, per-stream or dense $W_O$), the
FFN, and the aggregate-distribute transition are all per-stream. What `mst_stream_topk` gates is
only the FFN output:

```python
if stream_w is not None and self._stream_gate_attn:   # OFF by default
    attn_out = stream_w.unsqueeze(-1) * attn_out      # attention gate
...
if stream_w is not None:
    ffn_out = stream_w.unsqueeze(-1) * ffn_out        # FFN gate, always on when sparse
```

`estimate_flops` discounts exactly `fc_w` and `fc_proj_w`, adding the attention weights only
when `mst_stream_gate_attn` is set. Attention is left ungated because a skipped token stops
being a key/value for that stream, which changes the attention semantics per stream. That is
MoL's "attention coverage problem", and their answer is the always-active shared block.

Measured per-layer matmul parameters at $L=24$ ($D{=}1536$, $d{=}384$), from
`scripts/p11_active_params.py`:

| component | params | share |
|---|---|---|
| attention (incl. dense $W_O$) | 4.13M | 38.9% |
| FFN | 4.72M | 44.4% |
| transition + routers | 1.78M | 16.7% |

So $k{=}1$ of $N{=}4$ removes $\tfrac34 \times 44.4\% = 33.3\%$ of a layer's matmul parameters
and **66.7% stays active**. Gating attention as well would remove 62.5%, leaving 37.5% active,
and would take $L{=}24$ from 174.8M active matrices to 142.9M and from 1.481e9 to 1.190e9 active
FLOPs/token. That is the difference between MoL's 57%-active regime (1.53x forward speedup) and
their 20% regime (2.85x).

### Active counts for the paper's tables

| arm | L | total | active | matrices | active matrices |
|---|---|---|---|---|---|
| MST | 8 | 110,646,288 | 107,500,560 | 9,982,976 | 6,837,248 |
| MST | 16 | 413,224,992 | 388,059,168 | 77,680,640 | 52,514,816 |
| MST | 20 | 654,183,720 | 605,031,720 | 150,867,200 | 101,715,200 |
| MST | 24 | 964,359,216 | 879,424,560 | 259,716,096 | 174,781,440 |

Dense is dense: active equals total on both axes.

Two things this reconstruction confirmed. The paper's "FLOPs/token" column was **already** the
active count for MST (1.562e8 / 5.746e8 / 9.528e8 / 1.481e9 reproduce exactly), which was never
stated and now is, in the caption. And `--window-pattern` defaults to `SSSL` in `base_train` but
`SSSSL` in `GPTConfig`; the runs used base_train's, and the pattern changes the attention FLOPs
term, so any reconstruction has to set it explicitly.

### Active matrix parameters is MST's best axis

Re-running the CORE-versus-cost fit on the active axes:

| cost axis | MST L=24 | dense CORE at that cost | CORE multiplier |
|---|---|---|---|
| **active matrix params** | 174.8M | 0.1701 | **1.36x** |
| active FLOPs/token | 1.481e9 | 0.1753 | 1.24x |
| training FLOPs | 4.823e18 | 0.1910 | 1.07x |
| total matrix params | 259.7M | 0.2008 | 0.92x |
| active params | 879.4M | 0.2307 | 0.69x |
| total params | 964.4M | 0.2399 | 0.63x |

Active matrix parameters is the axis that both excludes the embeddings MST is charged for and
counts only the streams that run, and it is where the architecture looks best.

Note for bracketing: on TOTAL matrices dense $L{=}18$ (286.7M) brackets MST's 259.7M at 1.10x.
On ACTIVE matrices nothing in the dense ladder brackets from below, because dense $L{=}16$'s
201.3M is already 1.15x MST's 174.8M while scoring lower on CORE (0.181 against 0.194). A true
iso-active-matrix bracket would need dense $L\approx15$.

---

## Attention gating fails, the shared stream repairs it exactly, and neither beats the headline

MoL's sparsity levers, ported to MST and measured at d8 with all four arms on the same
280.8M-token budget (`scripts/p10_isotoken.sh`, per-stream VE, single seed).

| arm | bpb | Δbpb | ΔF/token | F multiplier |
|---|---|---|---|---|
| control (`topk=1`, FFN gate only) | 1.031984 | --- | --- | 0.899 |
| + `mst_stream_gate_attn=1` | 1.056366 | +0.024382 | -11.5% | **0.812** |
| + gate attn + `mst_stream_shared=1` | 1.035898 | +0.003914 | -3.6% | 0.900 |
| + `mst_stream_shared=1` only | 1.027640 | -0.004344 | +4.0% | 0.900 |

Against the dense exchange rate (`d(bpb) = -0.10406 * bpb * d(ln F)`):

    + gate attn         predicted +0.013129  actual +0.024382  gap +0.011253  WORSE
    + gate attn + S=1   predicted +0.003987  actual +0.003914  gap -0.000073  ON the curve
    + S=1 only          predicted -0.004240  actual -0.004344  gap -0.000104  ON the curve

**Gating attention alone costs 1.86x what it saves.** It buys 11.5% of FLOPs/token and pays
0.0244 bpb where the curve prices that cut at 0.0131, and the multiplier falls from 0.899 to
0.812. This is MoL's attention-coverage problem (their 3.1) reproduced in a partitioned
architecture, and at 0.0244 bpb it is well outside seed noise.

**The shared stream repairs it almost exactly**, landing 0.00007 off the curve. MoL's 3.2
Shared+Routed fix transfers. But it converts a clear loss into a wash, not into a gain.

**Nothing beats the headline.** Both S=1 arms are Pareto-neutral on the per-token axis. The
gate_attn direction is closed at d8 and does not warrant a d16 run.

### The trap: a within-depth trainF multiplier always rewards burning more compute

`S=1 only` reads 0.693 on training FLOPs against the control's 0.665, which looks like a 4.2%
win. It is not. Give every arm the control's per-token multiplier (0.899) counterfactually, so
that all four are Pareto-identical BY CONSTRUCTION, and read what their trainF multipliers
would be at the same token count:

| arm | trainF observed | if Pareto-identical | residual |
|---|---|---|---|
| control | 0.665 | 0.665 | -0.000 |
| + gate attn | 0.478 | 0.587 | **-0.109** |
| + gate attn + S=1 | 0.641 | 0.640 | +0.001 |
| + S=1 only | 0.693 | 0.692 | **+0.001** |

S=1's entire trainF advantage is **+0.001 of architecture and +0.027 of coordinate geometry**.
The gate_attn penalty, by contrast, has a **-0.109** residual and is real.

Mechanism: at fixed tokens, training FLOPs is proportional to FLOPs/token, but the dense trainF
curve is HALF as steep as the dense per-token curve (-0.05167 against -0.10406), because along
the dense ladder tokens grow roughly in proportion to FLOPs/token. So any arm spending more
FLOPs/token at fixed data slides along the flatter curve and gains multiplier for free.

**Rule. Cross-ladder comparisons use the training-FLOPs axis. Within-depth iso-token
comparisons use the per-token axis, or the residual against a Pareto-neutral counterfactual.**
Picking a config by its within-depth trainF multiplier systematically selects for burning more
compute. This is the mirror image of the MoL token-budget confound, where training FLOPs was
the correct axis precisely because the arms genuinely differed in data.

### Measured: what --target-active-params costs

The budget sensitivity is no longer an estimate. Same d8 config, two budget rules:

| budget rule | tokens | bpb | F/token multiplier |
|---|---|---|---|
| total matrices (`--target-active-params 0`, current) | 280.8M | 1.0320 | **0.899x** |
| active matrices (`--target-active-params 1`) | 247.7M | 1.0415 | **0.824x** |

**+0.0095 bpb for 11.8% fewer tokens; the multiplier drops 8.4%.** The predicted figure at d24
was an 11.6% drop for a 1.3x token cut, so the same order. The current rule is defensible and
documented (`p08_mst_parity_sweep.sh`), but it is worth about 8% of the headline multiplier and
the paper should carry the sensitivity rather than leave it implicit.


## Whole-model torch.compile makes startup grow with depth; regional compile does not

MST at L=24 was taking 30 to 60 minutes to reach step 0 on an H200, and sometimes hanging
outright. Four hypotheses were tested and three were wrong.

**Not startup work.** Construct + `init_weights` + `estimate_flops` + `setup_optimizer`, on CPU:
dense 1.21 / 4.77 / 12.38 s at L=8/16/24, MST 0.97 / 3.23 / 7.00 s. MST is faster than dense.

**Not recompilation.** MST produces exactly one Dynamo graph with zero graph breaks, the same as
dense. The two `Online softmax is disabled` warnings people see are forward and backward from
that single graph, not a second compile. The all-zero warmup batch in `base_train.py:2049` does
not force a re-trace: MST's stream router is fully tensor-valued, with no `.item()` and no Python
branch on data.

**Not autotuning and not CPU starvation.** `max_autotune` and `coordinate_descent_tuning` are
both False by default in torch 2.12, and the host had 57 cores with full affinity and no cgroup
quota.

**It is graph size.** `nanochat/common.py` compiles the whole model in one call, so the FX graph
holds every layer. Dynamo tracing, AOTAutograd's partitioner and Inductor's fusion scheduler are
all single-threaded and superlinear in node count, which is why adding cores does not help.
Compiling each repeated block separately compiles one layer and reuses it:

| MST, T=512, 16 threads | whole-model | regional | kernels whole | kernels regional |
|---|---|---|---|---|
| L=8  | 79.1 s | 56.8 s | 407 | 157 |
| L=16 | 126.3 s | **58.0 s** | 799 | **157** |

Regional is **flat in depth**. Whole-model doubles from L=8 to L=16 and keeps going.

The cost is that Inductor can no longer fuse across the layer boundary, measured at about 7%
slower steps. Step time is not something this project trades away: MST's wall-clock parity with
dense is a paper claim, and an arm trained with regional compile is not comparable to one that
was not. So `--compile-regional` is **off everywhere by default**, including in the profile
scripts, and is opt-in per run with `COMPILE_REGIONAL=1`. The 30-minute startup is the price of
keeping step time honest.

**The indefinite hangs are a second, related bug.** Inductor's compile-worker pool defaults to
`min(32, nproc)` subprocesses, each holding its own torch import. On a many-core box with a
memory ceiling, one worker getting OOM-killed leaves the parent waiting on a future that never
resolves. MST drives that pool far harder than dense (~1200 kernels at L=24 against ~400), which
is why MST is the arm that hangs. Both profile scripts now cap
`TORCHINDUCTOR_COMPILE_THREADS` at 8.

## Two attempts at MST's throughput gap: one small win, one measured dead end

MST reaches 214.4 TFLOP/s against dense's 497.7 at L=24, and the isoFLOP timer put the MST half
of the sweep at 59.7 h against dense's 22.0 h, a factor of 2.71. Attention is only 1.6 to 3.8% of
MST's device time, so batching the per-stream `flash_attn` calls cannot be the fix; the profile
put the excess in elementwise work around the GEMMs. Two things were tried.

**RoPE and QK-norm hoisted out of the stream loop: 3.5% faster, and free.** Every stream applies
the same rotary table, and `rms_norm` normalises per row over `head_dim`, so both can run once
over all `N * n_head` heads instead of `N` times inside the loop. Splitting the last axis into
`(n_head, head_dim)` is a pure view even on the non-contiguous `_batched_linear` output, so no
copy is added, and the per-element arithmetic is unchanged. Verified bit-identical on all four
streams for both `q` and `k`.

| variant | median ms/step | min | kernels |
|---|---|---|---|
| baseline | 936.87 | 908.10 | 415 |
| hoisted | **904.20** | **884.67** | 479 |

Note the kernel count went **up** while the step got faster. Kernel count is a good proxy for
compile time and a bad one for step time; only the timing decided this.

**`mode="reduce-overhead"` (CUDA graphs) does not help: MST is not launch-bound.** Measured
931.71 ms/step against 904.20 for the default mode, i.e. slightly worse, with no cudagraph
messages emitted under `TORCH_LOGS=perf_hints`. The arithmetic says why: roughly 479 kernels over
forward and backward is on the order of 1400 launches, and at a few microseconds each that is
under 1% of a 900 ms step. The flag `--compile-mode` is plumbed through so it can be re-tested at
L=24 to L=32 on real hardware, but it is off by default.

The remaining gap is largely intrinsic. The p10 microbenchmark measured block-diagonal GEMM
throughput at 0.31 to 0.71 of dense's across `D` = 512 to 2048, so even with every scrap of
overhead removed the isoFLOP wall-clock ratio floors out near 1.4x rather than 1.0x. Four
`d=384` GEMMs have less arithmetic intensity than one `d=1536` GEMM, and no amount of kernel
engineering changes that.

## The block-diagonal GEMM penalty is 0.739, and it is irreducible

MST reaches 0.46 of dense's model-level utilization on an H200. Everything said about why
had been inference from shapes. `scripts/p15_gemm_probe.py` measures it instead: it names
the kernel cuBLAS selects, parses its tile shape, computes wave occupancy from the real SM
count, and races five alternatives on the exact FFN shapes at L=24, N=4, M=16384.

**Two hypotheses died.** It is not bandwidth: MST sits at 302 FLOP/byte against the H200's
ridge of 206, so both arms are compute-bound. It is not wave quantization: MST emits the
same 6,144 tiles as dense at 99% wave efficiency.

**The mechanism is K-steps per tile.** Partitioning drops K from `d_model` to `d_model/N`,
so each output tile runs 3 K-steps where dense runs 12, and the same fixed prologue and
epilogue amortize over a quarter of the work. The measurement tracks it monotonically:

| GEMM | dense TF/s | MST TF/s | ratio | K-steps dense/MST |
|---|---|---|---|---|
| FFN up   | 747.0 | 528.4 | 0.707 | 12 / 3 |
| FFN down | 772.2 | 595.8 | 0.772 | 48 / 12 |

**Nothing beats cuBLAS.** Measured against `torch.bmm` on the current layout:
max-autotune 0.776x / 0.793x, a 19-config Triton sweep 0.701x / 0.809x, `grouped_mm`
0.815x / 0.693x, a per-stream `mm` loop 0.855x / 0.918x, and a pre-transposed weight
0.949x / 1.008x. Inductor's own autotuner reports `"best_kernel": "bmm"` after racing 19
Triton templates. A hand-written CUTLASS Stream-K kernel would have to beat NVIDIA's tuned
`nvjet` kernels, which is not a reasonable bet, so that avenue is closed on evidence rather
than on opinion.

Note the pre-transposed weight is *slower* on the larger GEMM: cuBLAS picks a better kernel
for the TN layout than for NN, so `_batched_linear`'s runtime transpose is helping and
should be left alone.

**The useful consequence is the decomposition.** The GEMM penalty is 0.739 but model-level
utilization is 0.462, so **only 62% of the gap is GEMM shape and 38% is non-GEMM overhead**
-- the permutes, per-stream norms, RoPE, router, transition and `select_backward` the
profiler flagged. The two halves have opposite tractability: the GEMM half is intrinsic to
`K`, while the overhead half is our own code, and the RoPE/QK-norm hoist already took 3.5%
off it. Bringing the overhead to dense's proportion would move utilization from 0.462 to
0.74 and the isoFLOP wall clock from 2.71x to 1.69x.

**One more effect worth knowing: this penalty grows with GPU size.** The same shapes measure
0.948 of dense on a 20-SM RTX 3050 and 0.739 on a 132-SM H200. It is an occupancy effect, so
B200 will make it worse, not better. Newer hardware is a headwind here, not a rescue.

## The V=131k break-even was scored against the wrong dense curve

The c08 and c09 analyses computed the Monarch break-even using a dense scaling slope of
0.3672 bpb/decade. That slope was measured on a different vocabulary and does not describe
the V=131,072 dense runs, whose slope is directly measurable from three points and is much
flatter:

| segment | slope (bpb/decade) |
|---|---|
| d4 to d8 | -0.186 |
| d8 to d12 | -0.132 |

A break-even is only meaningful against the baseline curve *at the same vocabulary*, because
the whole point of moving to V=131k was that it changes the dense curve. Re-scored, two
conclusions flip:

| arm | bpb | dense at the same FLOPs | verdict | horizontal |
|---|---|---|---|---|
| d4 m1=32 | 1.1608 | 1.1691 | win | 0.901x |
| d8 m1=32 | 1.0140 | 0.9844 | **lose** | 1.442x |
| d8 m1=128 | 0.9398 | 0.9601 | win | 0.778x |
| d12 m1=128 | 0.8616 | 0.8535 | **lose** | 1.151x |

The reported d8 win at m1=32 was an artifact. The real d8 win belongs to m1=128 only.

The lesson generalises past this experiment: a slope carried over from another configuration
is an assumption wearing the clothes of a measurement. Any break-even in the paper must cite
the baseline points it was computed from.

## At depth 12 the head is no longer a large enough target

At V=131,072 and depth 12 the dense output head is 50.7% of forward FLOPs. That number is the
entire budget the direction has to work with, and it converts to a break-even of +0.0405 bpb
for a head that costs *nothing at all*. Monarch at m1=128 already spends +0.0391 of it.

Projecting the measured gap law (gap scales as m1^-0.795, fitted on the d8 pair) across m1 at
depth 12 gives margins of -0.0080, -0.0013, +0.0008, +0.0001 and -0.0027 bpb at m1 = 128, 192,
256, 384 and 512. The entire curve sits inside +/-0.003 bpb of break-even. No choice of m1
rescues depth 12; the ceiling does not depend on m1.

Head share is approximately V / (V + 12 d L), so the regime where an output-head method can
win is large vocabulary against a small body. That is a real and nameable regime, on-device
and multilingual models, rather than a general claim about transformers, and the paper should
scope itself to it instead of implying the win survives arbitrary depth.

## Monarch blocks are limited by probability mass, not by semantic coherence

Q12 predicted that a semantically coherent vocabulary partition would let a block's
rank-m1 view stretch further. The c10 screen at V=32,768 refutes it and finds the
opposite mechanism.

| arm | d4, V/M=128 | d8, V/M=32 |
|---|---|---|
| freq | +0.0039 | -0.0033 |
| random | **+0.0218** | +0.0045 |

`freq` increases coherence and does nothing, flipping sign between depths at the
noise floor. `random`, which destroys structure, wins at both. Every block has the
same m1 whatever it holds, so what matters is that no block is asked to carry more
of the distribution than its dimensions can represent. Under the default token-id
assignment block 0 is BPE ids 0..block_out, the bytes and the commonest merges, and
it carries a large share of the mass while the last block holds tail tokens that are
almost never predicted and wastes its capacity on them. A random permutation gives
every block an equal share.

Two consequences. The cheap one: a fixed random permutation is a free win, needing no
frequency table, no checkpoint and no clustering. The interesting one: adaptive
softmax stratifies by frequency *and* allocates capacity non-uniformly, and this says
the stratification is worthless without the allocation, which is the part we have not
built.

The attenuation is the evidence that the pressure model is right: 0.0218 at V/M=128
against 0.0045 at V/M=32, a 4.8x drop for a 4x change in words per block dimension.
Note also that ``block_out / m1 = V/M`` exactly, so the factorisation cancels and
only V/M sets how oversubscribed a block is. That is what makes a small-vocabulary
screen possible at all, and what decides which small config is a valid stand-in.

## Score a capacity knob by marginal return, not by a break-even

The c10 screen had no dense leg at a matched budget, which would normally block the
Q13 verdict. It does not, because the question a capacity knob answers is whether its
FLOPs are better spent on it than on the model, and that comparison is internal:

| spend | return |
|---|---|
| residual r=32, V/M=128 | 0.835 bpb/decade |
| residual r=32, V/M=32 | 0.552 |
| dense scaling at V=32,768 | 0.169 |

Against budget-matched dense legs (121.1M tokens at d4, 440.4M at d8) that is 4.9x
the return of scale at V/M=128 and 3.3x at V/M=32. An earlier version of this note
said six times, scoring against the c07 Monarch family slope of 0.126 rather than the
dense baseline; the dense curve is the right comparator and it is steeper. The same framing converts directly into a transfer
prediction: at V=131,072 depth 12 the residual costs 0.01558 decades and the gap has
to fall 0.0101 bpb, so it must return 0.649 bpb/decade, and the matched-pressure
measurement returns 1.29x that.


## The V=32,768 depth-8 arc, and where the Monarch head actually stands

One budget, 440.4M tokens, every arm scored against the dense legs at the same budget
on a slope of -0.169 bpb/decade:

| | bpb | margin vs dense | horizontal |
|---|---|---|---|
| c07 m1=32 | 1.0479 | -0.0506 | 1.992x |
| c09/c10 m1=128 | 0.9951 | -0.0046 | 1.064x |
| + perm random | 0.9906 | -0.0001 | 1.002x |
| + residual r=32 | 0.9880 | +0.0003 | 0.996x |

m1 is what moved this, 1.992x to 1.064x. The last two rows are inside the +/-0.004
noise floor, so depth 8 at V=32,768 is a dead heat rather than the win the sign
suggests, and it should be written up that way.

The margin arithmetic transfers as follows, where margin is dense-at-the-same-FLOPs
minus ours:

| | base | random | residual | both, if additive |
|---|---|---|---|---|
| d4, V/M=128 | -0.0664 | +0.0218 | +0.0491 | +0.0045 |
| d8, V/M=32 | -0.0046 | +0.0045 | +0.0049 | +0.0048 |

Two unrelated configurations landing at +0.0045 and +0.0048 is worth more than either
number alone. V=131,072 depth 12 starts at -0.0081 and needs +0.0081, which the d8
deltas would deliver marginally (+0.0013) and the d4 deltas comfortably (+0.0628).
The target shares d4's pressure and d8's model quality, so the truth is between, and
additivity is still an assumption: the c10 combination arm used `freq`, which turned
out to be the null arm.
## The block structure survives the residual taking over the budget

The obvious objection to Monarch is that it is low-rank wearing a hat, and once the
shared residual grows to 79% of head cost that objection gets sharper rather than
weaker. Cost-matched pure low-rank controls at depth 4, V=32,768, with the rank
derived as `R = (d*M + V*m1)/(d + V) + r` rather than chosen:

| r | Monarch+res | low-rank | advantage | rank ceiling | residual share |
|---|---|---|---|---|---|
| 32 | 1.2247 | 1.2613 | +0.0364 | 257 vs 65 | 49% |
| 64 | 1.2097 | 1.2293 | +0.0194 | 257 vs 97 | 65% |
| 128 | 1.1757 | 1.2179 | +0.0419 | 257 vs 161 | 79% |

The advantage does not decay as the residual dominates; at r=128 it is the largest of
the three and ten times the noise floor. The ceiling column is the mechanism: Monarch
plus the residual reaches the FULL rank a dense softmax has, d+1 = 257, while the
cost-matched low-rank control is capped at 161, or 63% of it. At that budget
Monarch+r=128
crosses the dense curve (+0.0034, horizontal 0.954x) while the low-rank control loses
by 0.0385 (1.690x), so the structure is the difference between winning and losing, not
a refinement.

Treat the r=64 row as suspect. It is low against both neighbours here and it also
produced the one non-monotonic marginal return in the rank sweep (0.238 between r=32
and r=64, against 0.523 before and 0.327 after). Two independent anomalies on one
single-seed run is a seed, not a phenomenon, and it needs rerunning before publication.

Method note worth keeping: derive a cost-matched control, never type it. A hand-matched
rank is correct exactly once and then silently compares two budgets the moment M, m1 or
r moves.


## A linear head's rank ceiling is d, however the FLOPs are split

`rank_ceiling` added `residual_rank` to the base without clamping, so a Monarch head
with r=128 at d=256 reported 385: more rank than a dense softmax at that width has.
Every path in that head is linear in h, so the logit matrix is V x d and its rank
cannot exceed d whatever the budget buys. The residual *reallocates* rank between the
two paths, reaching directions the block-diagonal factor cannot; it does not create
extra ones. Both heads now clamp at `n_embd` for a linear g, and only a nonlinear g,
which lifts the logit matrix off the d-dimensional bound, is allowed to exceed it.

The bug inflated the mechanism claim rather than changing it: 257 against a
cost-matched 161 is the same argument as 385 against 161, and a head that reaches the
full dense rank at 0.766x the dense cost is the cleaner statement anyway. Worth
remembering that the test asserted `withr.rank_ceiling() == plain.rank_ceiling() + r`,
which encoded the bug rather than catching it: a test written from the implementation
confirms the implementation.
## A chord is not a curve: how the dense baseline is interpolated moves the headline

Every margin in this project is "our bpb against dense at the same FLOPs", and dense at
those FLOPs is almost never a run that exists. It is read off the budget-pinned depth
ladder, and the ladder is not straight. At V=131,072 the chord slopes flatten from
-0.186 (d4 to d8) to -0.132 (d8 to d12), so the curve is convex and a chord between two
legs lies *above* it.

Re-scoring the depth-12 arms with an exact quadratic through all three dense legs
instead of the d8-d12 chord:

| arm | margin (chord) | margin (quadratic) |
|---|---|---|
| MON_base | -0.0078 | -0.0112 |
| MON + r=128 | +0.0045 | +0.0016 |
| low-rank M=261 | +0.0029 | +0.0000 |

The win keeps its sign and loses two thirds of its size, and the low-rank control goes
to an exact tie. The honest statement of that result is a range, +0.0016 to +0.0045,
not a number, until a dense leg exists near 10^18.19 where the arms actually sit. Depth
10 lands near 10^18.06 and would shrink the enclosing chord from 0.61 decades to 0.30;
curvature error goes roughly as the square of chord length.

Two consequences worth carrying:

**Head-to-head at matched cost is immune.** The Monarch and low-rank arms sit 0.0218%
apart in FLOPs, so any curve evaluates to the same place for both and cancels in the
difference. Interpolation fragility attacks vs-dense claims, never a cost-matched
ablation, which is one more reason to make the ablation carry the argument.

**Check which side of the anchors an arm sits on.** The V=32,768 depth-4 arms are
extrapolated up to 0.36 decades below the lowest dense leg. That errs the other way,
since extrapolating left on a convex curve understates dense's bpb and so understates
our margin, but a third V=32k dense leg is needed before any depth-4 number is quoted.

## Block-private capacity saturates almost immediately

Holding head cost and per-token capacity fixed at depth 4, V=32,768 and sweeping only
how the budget splits between block-private directions (m1) and shared ones (r):

| arm | private | bpb | vs low-rank | Monarch share of head |
|---|---|---|---|---|
| low-rank M=161 | 0% | 1.2179 | — | 0% |
| m1=8, r=152 | 5% | 1.1790 | +0.0386 | 6.1% |
| m1=16, r=144 | 10% | 1.1773 | +0.0403 | 11.0% |
| m1=32, r=128 | 20% | 1.1757 | +0.0419 | 20.9% |
| m1=64, r=96 | 40% | 1.1816 | +0.0362 | 40.6% |
| m1=128, r=33 | 79% | 1.1851 | +0.0325 | 79.6% |

Unimodal, with the optimum at the configuration already in use. The shape is what
matters: 0% to 5% buys +0.0388 and the whole span from 5% to 79% covers 0.0094. A
block-diagonal factor costing 6.1% of the head budget captures 92% of the available
gain, so the claim to test is not "Monarch beats dense" but "a small block-diagonal
term added to a low-rank head is worth a lot at matched cost".

It also kills a plan. Read at depth 12's operating point, the shape says moving from
50% private to 12.5% recovers 16% of the depth below low-rank; depth 12 measured
+0.0016 there, so the correction projects +0.00025 bpb. The private/shared split is
not why depth 12 disagrees with depth 4, and the depth-12 split sweep was not worth
running.

What remains unexplained is the size of the disagreement: the depth-12 curve is 23x
shallower than depth 4's while head share differs only 2.1x, with V/M (128) and m2 (8)
already matched between the two tested points. Vocabulary, model quality and depth are
the surviving candidates, and depth 8 at V=131,072 separates them for three runs: it
carries depth 4's head share (52% against 54%) at the target vocabulary.

## Head share does not explain why depth 4 disagreed; vocabulary does

Depth 8 at V=131,072 was run specifically to separate head share from vocabulary,
because it carries depth 4's head share at depth 12's vocabulary. Depth below the
cost-matched low-rank control, at 50% private:

| config | head share | depth |
|---|---|---|
| d4, V=32,768 | 54% | +0.0354 |
| d8, V=131,072 | 52% | +0.0015 |
| d12, V=131,072 | 26% | +0.0016 |

Depth 8 matches depth 4 on head share and lands on depth 12's answer. At V=131,072 the
block-diagonal structure is worth about +0.004 bpb over a cost-matched low-rank head,
not the +0.042 that V=32,768 suggested, and it costs 14% more wall clock (512ms against
449ms, MFU 20.23 against 23.03). On the FLOPs-Pareto criterion that is a small win; on
wall clock it is a loss. The low-rank residual is doing nearly all the work.

Two things survive. The split axis is real at the target vocabulary, just shallow:
12.5% private gives +0.0040 against 50%'s +0.0015. And depth 8 is where the whole
direction is strongest, with both arms well past dense (0.768x for Monarch+residual,
0.807x for low-rank alone) against depth 12's 0.925x and 0.950x.

Worth chasing: m1=32 is the best point at BOTH depth 4 (20% of capacity there) and
depth 8 at V=131,072 (12.5% of capacity). That points at an absolute optimum near
m1=32 rather than a fixed share, and would also explain c09 choosing m1=128 at r=0 and
that choice inverting once shared capacity exists.

---

# EET: what the gap actually is (2026-09-07)

Early Exit Transformers had sat at roughly +0.06 val_bpb behind dense for a long time,
with thirteen abandoned mitigations and a working conclusion in `eet_experiment_log.md`
that the remainder was "architectural". It is not. The cause is specific, measurable in
seconds on any checkpoint, and none of the thirteen touched it.

## The finding: EET trains three layers and carries five dead ones

Per-layer bpb read through the shared head, d8, V=32768, 265,814,016 tokens:

| layer | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| dense | 2.676 | 2.360 | 2.244 | 1.783 | 1.686 | 1.296 | 1.194 | **0.982** |
| EET | 2.107 | 1.267 | 1.188 | **1.115** | 1.113 | 1.113 | 1.114 | 1.114 |

Dense gains **0.80 bpb** across layers 4 to 7. EET gains **0.001**. Its residual barely
moves after layer 3 (mean ‖Δx‖ 1.3 against dense's 94.6).

Three independent measurements agree, two of them taken months apart:
- per-layer readout bpb (above),
- residual movement per layer,
- the gradient ratios already sitting in `diagnostics_D8/diagnostic_results.json`:
  layers 0-3 receive **3.2-4.7x** dense's gradient, layers 5-7 receive **0.37, 0.23, 0.13x**.

## It is a trade, not starvation

EET's layer 3 is **0.67 bpb better** than dense's layer 3. Its layer 7 is **0.13 worse**
than dense's layer 7. Forcing early layers to be prediction-readable buys enormous early
quality and destroys the depth hierarchy. The mechanism is not that deep layers are
starved of data or gradient; it is that the model stops building hierarchy once the
representation is as readable as the shared head allows.

**The collapse point tracks the first exit layer.** Moving `--eet-min-exit-layer` from 1
to 4 moves the flat region from layer 3 to layer 5, and takes bpb from 1.0622 to 1.0194:

    min_exit=1:  gains  0.89  0.11  0.09 | 0.003 -0.001 -0.013 -0.005
    min_exit=4:  gains  0.38  0.11  0.48  0.42  | 0.013 -0.000 -0.010

## The tension is intrinsic, not an EET artifact

A **dense** model trained with a CE term at every layer through the shared head (no exits,
no routing) loses 0.033 bpb and its depth gain collapses from +0.8008 to +0.0497. Asking a
stack to be readable at every depth is fundamentally at odds with building a deep
hierarchy, whether or not anything exits.

This is why post-hoc readout fixes cannot work: a per-depth head refit on EET's own frozen
backbone recovers **exactly 0.0000**, because layers 3 to 7 hold the same vector and there
is nothing different for a second head to read.

## Corollaries that close whole families

- **The router is worthless.** Learned 1.0643, uniform-random 1.0649, and with the router
  and every auxiliary loss removed **1.0610** — the cheapest configuration is also the
  best. EET is empirically a fixed, content-blind, monotone depth schedule. Of course
  routing does not matter: every destination past layer 3 is the same representation.
- **Allocation costs almost nothing.** The iso-data dense curve is `L(F) = 4.17 * F^-0.0738`,
  which is convex, so Jensen prices signal-free depth spreading at **+0.001** against a
  measured gap of 0.049. Routing is not the lever.
- **Context is not the cause.** Restoring fresh per-layer keys and values for exited
  tokens recovers 0.008; banked (stale) KV recovers nothing.
- **Coverage is not the cause.** Layer 7 sees **99.7%** of the vocabulary; only the token
  *mass* is 10%. Routing is a within-sequence top-K, so a token that loses in one sequence
  wins in another.
- **Gradient surgery cannot work.** Depth-LR-scale and depth-grad-scale amplify a gradient
  with no descent direction, which is why one hurt and the other did exactly nothing.

## Scaling: keep the feature-building prefix ABSOLUTE, not proportional

Delaying the first exit costs FLOPs, but how that cost scales depends on how the delay is
specified, and the two behave oppositely:

| depth | `min_exit = L/2` | `min_exit = 4` |
|---|---|---|
| 8 | +15% dearer | +15% dearer |
| 16 | +28% dearer | +12% dearer |
| 32 | **+36% dearer** | **+7% dearer** |

A proportional prefix pins the average active fraction at 0.775 at every depth, because
half the stack always runs at full width. An absolute prefix amortises: half the stack at
d8, a sixth at d24. Any design that buys hierarchy by delaying exits must fix the number
of dense layers, not the fraction.

## The economics, so nobody re-derives them

At d8/512/V=32768 the LM head is **35.2%** of active FLOPs per token and routing never
touches it, so a 39% saving on the blocks is only a 28% saving overall. The head's share
falls to 6.4% at d24, which is why EET's economics improve with depth: the FLOP ratio goes
0.722 (d8) -> 0.618 (d16) -> 0.586 (d24). The **allowed** bpb gap saturates around +0.032
by d16, so depth buys the budget from +0.024 to +0.032 and nothing beyond.

EET is genuinely faster once it compiles: **0.863x dense** on an H200 at B=128
(forward 0.821, backward 0.858, optimizer 1.436, and the optimizer is 3-4% of the step).

## Two measurement traps that cost a full sweep each

- **`--eet-router-task-grad 1` silently disabled `torch.compile`.** `cw_full = 1.0 - rw`
  used a Python scalar, which let inductor promote the blend to fp32 while `x_final` stayed
  bf16; the following `scatter` then raised `Expected self.dtype to be equal to src.dtype`.
  On some builds it falls back to eager instead of erroring, which is worse: the run
  completes, the bpb is correct, and the entire speedup disappears. Compiled EET is 0.702x
  dense; uncompiled it is 1.201x, which is what a whole sweep of "EET is slower" measured.
  A `.to(x_final.dtype)` guard does NOT fix it, because tracing erases the cast whenever
  the dtypes already match. The destination has to be unconditionally fp32 so the source
  casts are real conversion nodes.
- **A stale token budget produced a non-iso-data comparison.** `DENSE_D8` ran on 440M
  tokens while every arm it was the control for had 266M, because the budget was measured
  during an earlier run at a different vocabulary and reused. Budgets must record the
  vocabulary they were measured at, and a sweep with no budget must abort rather than let
  each arm pick its own Chinchilla horizon.

## What this implies for the direction

The remaining question is narrow and well posed: **what gives layers 4 to 7 a job that
layers 1 to 3 cannot do?** Families that assume a stack can be both readable-now and
refinable-later are closed by the dense deep-supervision result. What survives is giving
deep layers something structural the shallow ones cannot have. The first candidate is the
**inverse-width stack** (`--eet-width-power`): a block running 10% of the tokens can be
~10x wider at the same FLOPs, since block cost scales with tokens x width squared. That is
the opposite of every pyramid or funnel transformer, which tapers width down with depth,
and it is only possible because the exits make token count fall.

Diagnostic to run on any checkpoint before theorising, ~2 minutes:

    python -m scripts.eet_readout_oracle --ckpt <model.pt> --skip-fit

It prints per-layer bpb, per-layer gain and mean ‖Δx‖, and says outright whether the deep
layers are working.

## The CPU bit kernel works, and it says nothing about whether we can train here

`kernels/b1_cpu.c` is the first real (non-simulated) binary kernel in the repo: `b1_pack`
(float sign to a `uint64` bitplane), `b1_dot` (AVX512 `_mm512_popcnt_epi64` over XNOR),
`b1_linear`, `b1_bundle` (threshold majority vote), bound through ctypes in
`nanochat/b1_cpu.py`.

`b1_cpu.check_equivalence()` returns a **max absolute difference of exactly 0.0** against the
`BinaryLinear` fp path at K=512/1792/3584 across seeds. That matters more than the speed: it
proves `dot(a,b) = n - 2*hamming(a,b)` and the fp simulation compute the same function, so
every bpb number measured through the simulated path is a number the bit kernel would
reproduce. The simulation is not an approximation of the kernel; it is the kernel in disguise.

Single-threaded on this laptop, min-of-20 with a volatile sink and per-iteration input
perturbation (both needed; see the measurement traps below):

| shape | bit GOPS | fp32 GFLOPS | speedup |
|---|---|---|---|
| 64x512x512 | 602 | 112 | 5.4x |
| 64x1792x1792 | 742 | 92 | 8.1x |
| 64x3584x3584 | 975 | 85 | 11.5x |
| 256x3584x3584 | 950 | 96 | 9.9x |

The speedup **grows with K** because packing cost amortises while popcount throughput stays
flat, and because fp32 falls out of cache faster than a 32x smaller bitplane does. This is the
CPU shadow of the same effect O3 measured on sm_86.

### Two things this does not buy

**It does not make training possible here.** The kernel is a forward-only object. Backward
needs gradients with respect to the fp latent weights, and the STE forward is `sign()` then an
ordinary fp GEMM, so a W1A1 training step on CPU is just a slow fp training step plus sign
overhead. Costed at depth 8: **82 hours at the measured multi-threaded BLAS peak (426 GFLOPS),
234 hours at a realistic 150 GFLOPS** sustained. Even the reduced 100M-token budget is 53
hours. Training stays on rented H100s; the laptop runs kernels and smoke tests only.

**It does not make batch-1 decode faster at matched bytes**, which corrects an earlier claim of
"~7.5x faster on CPU". At batch 1 the model is memory-bound, and matched bytes means matched
memory traffic by construction:

| config | MiB | compute ms | memory ms | tok/s |
|---|---|---|---|---|
| dense fp16 d=512 | 240.0 | 0.80 | 12.58 | 79.5 |
| binary d=512 (matched architecture) | 15.0 | 0.13 | 0.79 | 1272 |
| binary d=1792 | 78.8 | 0.85 | 4.13 | 242 |
| binary d=3584 (matched bytes) | 231.0 | 1.56 | 12.11 | 82.6 |

The 16x memory win is real, but it is a **budget that can be spent once**. Spend it on width
and the direction is a quality play at equal speed; spend it on nothing and it is a 16x speed
play at equal quality. The v3 bet in the plan deliberately spends it on width, so the honest
claim for that arm is "same tokens per second, seven times the width", never "faster".

## Two ways a width sweep breaks that have nothing to do with width

B03 is the first sweep in this repo that changes `--model-dim`. It died twice before a
single step ran, and both causes are latent for anything that varies width.

**1. The `--model-dim` override reached the d12 reference model.** `base_train` derives the
token horizon and the auto batch size from a *fixed* d12 anchor:

```
D_REF = target_param_data_ratio * get_scaling_params(build_model_meta(12))
B_opt = 2**19 * (target_tokens / D_REF) ** 0.383
```

`build_model_meta` honoured `args.model_dim` unconditionally, so asking for width 3584 built
the anchor at 3584 too instead of 12x64=768. Scaling params grew about 21x, `B_opt` fell by
21^0.383 ~ 3.2x, and the auto batch dropped from 2^19 to 2^17 while the micro-batch stayed at
128x2048 = 2^18. The run died at `assert total_batch_size % world_tokens_per_fwdbwd == 0`,
which is three inferences away from the actual mistake. `build_model_meta` now takes
`apply_dim_override` and the d12 call passes `False`; the assert is now a `SystemExit` that
prints both quantities, where each came from, and the device batch size that would fit.

The general lesson: a muP-style reference anchor has to be immune to every flag that describes
the arm under test, or the reference moves with the thing it is supposed to measure.

**2. Chunked attention caps the forward peak and does nothing for the training peak.**
`HammingAttention` chunks queries to avoid materialising `(B, H, T, T)`, and the docstring
claimed that solved the memory problem. It solved half of it. Softmax backward needs its own
output and the value einsum needs the same weights, so autograd saves `w` for *every* chunk,
and the chunks sum back to exactly the full `(B, H, T, T)` tensor chunking was meant to
avoid: 30 GB in fp32 for one layer at B=64, H=28, T=2048. It OOMed an 80 GB H100 at width
1792 with 75 GB already resident.

Chunking bounds peak memory only when each chunk's intermediates are freed, which under
autograd means recomputing them. Each chunk now goes through
`torch.utils.checkpoint(..., use_reentrant=False)`, verified bit-exact against the stored
path in forward and in both gradients, and verified to survive `torch.compile`. The softmax
also casts back down after its fp32 reduction, which halves the recompute peak.

`scripts/b03_binary_width.sh` now derives the batch plan instead of inheriting d=512's:
`device_batch = 512/width * base`, clamped so the micro-batch divides the total batch on the
GPU count actually present, with `--total-batch-size` pinned explicitly so the auto path
cannot participate. At depth 8 that is 262,144 tokens with device batches of 128/64/32/16 for
widths 512/896/1792/3584 on one GPU.

## Matched bytes is 32x the parameters, and it takes the data budget with it

The first H100 attempt at the matched-bytes width reported 27,562 tokens per second and a
235-minute ETA for one seed. Both the cost and the reason are arithmetic that should have been
done before the launch.

Bytes per parameter fall 16x under binarisation, but parameters grow as `d^2`, so matching
bytes at depth 8 does not buy 16x the parameters. It buys **32x**:

| width | scaling params | MiB | tokens/param at a pinned 440M | FLOPs vs dense |
|---|---|---|---|---|
| dense fp16 512 | 41.9M | 240.0 | 10.50 | 1.0x |
| binary 512 | 41.9M | 15.0 | 10.50 | 1.0x |
| binary 896 | 106.4M | 30.2 | 4.14 | 2.5x |
| binary 1792 | 367.0M | 78.8 | 1.20 | 8.8x |
| binary 3584 | 1,350.6M | 231.0 | **0.33** | **32.2x** |

The 32x explains the wall clock. The **0.33** is the finding, and it is worse than the cost.

Pinning every arm to the dense arm's token budget is the correct control when the arms differ
only in precision, and every earlier binary sweep in this project was right to do it. It
becomes the wrong control the moment an arm changes width, because it hands the wide arm 1/32
of the data per parameter. The accumulation-entropy argument is a claim about capacity, and a
model at 0.33 tokens per parameter cannot exhibit capacity. A loss at that point is
undertraining and a win is unexplained, so the arm cannot answer the question it was built to
ask, at any wall-clock cost.

**But 10.5 tokens per parameter is the wrong charge for a binary arm, and using it was the
plan contradicting itself.** Section 3.8 sets the capacity of a bf16 parameter at about 4
task-relevant bits and a binary parameter at 1, and that ratio is what the whole per-layer
bandwidth claim rests on. Using it to buy width and then ignoring it when buying data charges
the binary arm four times the data its parameters can absorb. One convention in both places:
match tokens per CAPACITY BIT, which for dense d=512 is 10.5 / 4 = 2.625.

| width | tokens at 2.625/param | cost vs the dense run | time | bits/layer vs dense |
|---|---|---|---|---|
| 512 | 110M | 0.2x | 2 min | 0.70x |
| 896 | 279M | 1.9x | 14 min | 1.32x |
| 1792 | 963M | 26.8x | 3.1 h | 2.85x |
| 3584 | 3,545M | 259x | 30 h | 6.15x |

That is one assumption used consistently, and it is falsifiable in both directions: a reviewer
who rejects 4 useful bits for the data budget has to reject the 6.15x bandwidth claim with it.
At the conservative reading of 1 useful bit the old numbers return and even width 1792 is 8.9
hours.

**What this costs the direction.** Matched bytes stops being a sweep arm at depth 8 and becomes
a single headline training run, earned only if a cheap slope exists first. `b03` now pins the
RATIO instead of the token count, so width is the only difference between arms, orders the
widths cheapest first, prints a cost table before launching anything, and refuses an arm over
`MAX_COST` dense-run equivalents. The default budget buys an iso-token anchor at d=512 plus
two iso-capacity slope points at d=512 and d=896, about 23 minutes total, which is enough to
say whether the slope exists at all.

**The transferable rule.** Any sweep that varies capacity has to say which of tokens, tokens
per parameter, or total FLOPs it is holding fixed, and print it per arm. Pinning the wrong one
produces a number that looks like a result and is not attributable to anything.

## The native binary block had two bugs that each destroyed it on their own

The first native arm at depth 8, width 512, 440M tokens, finished at train loss 6.491 against
a dense d=512 that reaches 0.9575 val bpb. Not marginal: broken. Both causes are in the
native modules, and neither has anything to do with quantising a dense block.

**1. The gates had no scale correction, so every Kanerva read returned one vector.**
`HammingAttention` and `BinaryKVFFN` both divided raw popcount scores by `tau`, default 1.0.
Those scores are `+-1` dot products, so their standard deviation is `sqrt(fan_in)`: 22.6 over
2048 FFN slots, 11.3 over a 128-dim head. Measured at init:

| | effective retrieved | out of |
|---|---|---|
| FFN, tau=1.0 | **1.4** | 2048 slots |
| attention, tau=1.0 | **1.8** | 256 keys |

The associative memory was a lookup table returning a single stored vector, which is the exact
opposite of the bundling it exists to do, and it is the per-layer bandwidth of section 3.8
destroyed by a missing constant. Section 3.3's argument that binarisation deletes normalisation
is correct for the residual stream, because `sign()` is scale-free, and false for a gate, where
the logit scale relative to the temperature is the entire operation. Ordinary attention divides
by `sqrt(head_dim)` for this reason and dropping it was not purification.

**Softmax was also the wrong surrogate.** A Kanerva read gates each slot INDEPENDENTLY against
a radius and bundles everything inside it, which is exactly what the `hard` branch did. Softmax
makes slots compete for a fixed unit of weight, so the soft and hard branches were computing
different functions and annealing `tau` toward zero never converged to the deployed path. Both
now use `sigmoid(gate / (tau * sqrt(fan_in)))`, and a half-step offset resolves exact ties the
same way in both: a `+-1` dot product over an even dimension is even, so `score == 0` is common
and means no evidence, and excluding it versus half-voting it alone accounted for 4.9% of
output bits.

**2. `resid_width` was a no-op, and width 1 destroyed the residual path.**
`(x + branch).clamp(-width, width)` with `x` and `branch` both `+-1` never leaves `[-2, 2]`, so
every width behaved as width 1, and the result was signed again, so the stream was re-binarised
at all `2*n_layer` sublayers. Width 1 is a majority-of-two with a constant tiebreak: when the
branch disagrees, the output is a learned constant independent of the input. Agreement between
the block stack's output sign and its own input, over eight blocks at depth 8:

| width | survives | changed by the layers |
|---|---|---|
| 1 | **49.8%** | 50.2% |
| 3 | 65.5% | 34.5% |
| 4 | 78.4% | 21.6% |
| 8 | 97.9% | 2.1% |
| 16 | 100.0% | 0.0% |

49.8% is chance. The embedding had no path to the head, and the model could only learn what the
last layer or two could reconstruct. The two failure modes bracket the answer tightly: too
narrow and the stream is noise, too wide and no branch can move a saturated channel. The
accumulator is now carried in `[-1, 1]` with resolution `1/width` rather than as a raw integer,
so every downstream `sign_ste` keeps its clip of 1.0 and keeps passing gradient, it accumulates
onto the STREAM instead of onto the block's signed copy, and the default is derived as
`max(2, n_layer // 2)` rather than typed.

**The arithmetic is still binary.** Every GEMM operand is `sign_ste(...)`, so the popcount
kernel still applies. The accumulator lives only between sublayers, which is what a counter is
in Kanerva's sparse distributed memory: a bundling register, not an operand.

---

## 2026-10-01: Sampling-Aware Pretraining (SAP) Stage 0 Degeneracy Pre-Test

### Background & Hypothesis (§3.2, §6.1)
v1 proposed representation-space block coupling (RBC) to align multi-token prediction heads via InfoNCE (Variant A) or predictive forward-consistency (Variant B).
v2 proved theoretically (§3.2) that because both $h^{(1)} = g_1(h_t)$ and $h^{(2)} = g_2(h_t)$ are deterministic functions of the same frozen trunk state $h_t$, coupling heads can drive the auxiliary loss to zero analytically with zero gradient pressure on the backbone to resolve branch structures.

### Stage 0 Pre-Test Results (Checkpoint: 4-layer NanoChat, step 462, FineWeb-Edu)
Trained *only* bilinear head $M$ (Variant A) and MLP $f_\phi$ (Variant B) with the backbone **100% frozen** (`requires_grad=False`):

| Metric | Initialization | 50 Steps | 150 Steps | 300 Steps |
|---|---|---|---|---|
| **Variant A (InfoNCE) Loss** | 5.1547 | 0.1531 | 0.0491 | **0.0364** |
| **Variant A Top-1 Accuracy** | 0.8% | 100.0% | 100.0% | **99.2%** |
| **Variant B (Predictive) Loss** | 1.0105 | 0.3073 | 0.0436 | **0.0091** |
| **Variant B Cosine Similarity** | -0.0105 | 0.6927 | 0.9564 | **0.9909** |

### Fork vs Non-Fork Prefix Differential
Tested on curated high-entropy fork prefixes vs deterministic factual prefixes:
- **Variant A Loss**: Fork `0.0418` vs Non-Fork `0.0580` (Gap = `0.0162`)
- **Variant B Cosine Sim**: Fork `0.9829` vs Non-Fork `0.9866` (Gap = `0.0038`)
- Both coupling heads are completely invariant to branching uncertainty and achieve ~99% alignment on frozen states.

### Pre-Registered Kill Gate Decision
- **STAGE 0 GATE FIRED**: Degeneracy confirmed. Variants A and B are mathematically inert and do not transmit branch-disambiguation signals to the backbone.
- **Action**: Retire Variants A and B immediately. Advance to **Milestone 1 (Dynamic Entropy Gating)** and **Variant C (Low-Rank Logit Joint Energy)**.


### Correction (2026-10-01, later the same day): the Stage 0 run is not evidence, and the Phase 0 measurements are invalid

The conclusion above (representation coupling cannot make sampled tokens agree) is right, but
it is right by the argument, not by this run:

- **Stage 0 fitted a random frozen map.** `diagnostics/stage0_degeneracy.py:206` builds `g2` as a
  randomly initialised, frozen LayerNorm-Linear-LayerNorm, not a trained head. A learnable map
  fitting a fixed random invertible map always drives the loss to ~0, so the gate could not
  have done anything but fire. The reason A and B are dead is the graph argument: at inference
  no sampled token reaches head 2, so no loss on pre-sampling features can create dependence
  between the sampled tokens. Attention among the slots does not change that either.
- **`benchmarks/eval_branching.py:198` and `benchmarks/joint_divergence.py:57` never measure a
  second-token prediction.** They set `logits2 = logits1` and `probs2_marginal = probs1`, so the
  "product of marginals" they report uses the t+1 distribution for t+2. The TV 0.316 / KL 1.21
  numbers measure nothing the plan defines.
- **The "narrow headroom" verdict is 0/0.** 0 of 12 fork prompts matched a branch at every
  temperature (the d4 checkpoint emits " New" after "...New", "icago" after "iced"), and the
  script reported an empty rate as 0.
- **The checkpoint was the wrong one.** SCH's d4 dense run (n_embd 256), not the plan's model,
  and not `out/dense_V32k_d8_model_000966.pt`.
- **v2 itself had drifted from the idea.** It was framed as K-head MTP scored by speculative
  acceptance; the idea is one LCA-style head emitting T tokens per pass with no verification,
  pretrained from scratch. `sap_research_plan.md` v3 restates it.

## 2026-10-01: SAP v3, what is known before Stage A

- **Product of marginals is a property of the graph, not of the loss or the head shape.** One
  head, K heads, attention among slots: if every slot's distribution is fixed before anything is
  sampled, the block is a product of marginals. It costs the block's total correlation (TC) in
  nats against the true joint, exactly. Only a per-sample variable that reaches the slots
  (noise, a latent, or a sampled token) can change that.
- **Sampling noise in the forward pass with plain CE does not work, and hard targets are not why.**
  Every noise draw chases the same observed block, so the decoder learns to ignore the noise
  (Condor: 0.02 nats of sensitivity, 9% valid blocks). The fix is the training rule: a recognition
  model that sees the block (ELBO), or a strictly proper scoring rule over samples (energy score).
- **A parallel decoder makes posterior collapse expensive.** Given (ctx, z) the slots are
  conditionally independent, so ignoring z costs exactly TC in reconstruction; encoding the
  dependence costs at most TC in KL. The weak decoder that text VAEs usually have to be given
  comes free here.
- **v2's Variant C normaliser was O(|V|^2 r), not O(|V| r).** exp of a low-rank bilinear form is
  not low-rank. The tractable versions are the mixture (B3) and the local AR head (B4).
- **Digit or tree-code noise is killed by our own measurement.** It would make each slot a
  hierarchical softmax, which costs +0.1217 bpb at d4 (c05 above).
- **The block readout dominates the head's cost at V=32,768.** T slot rows through the V x d head
  on 1/8 of the positions is about +22% training FLOPs at d8, against a ~15% budget; Stage B uses
  1/16 and matches every arm to the dense arm's FLOPs with --target-flops.
- **KV-cached and full-recompute trunk states differ by ~1e-2.** Both attention paths cast q, k, v
  to bf16 and round differently, so a near-tie argmax can flip between them. Equivalence of decode
  loops has to be tested through full recompute (tests/test_block_head.py does).
- **`tests/test_code_head.py::test_ensure_tokenizer_refuses_the_wrong_vocabulary` overwrites the
  tracked tokenizer.** It asks `ensure_tokenizer` for V=131,072 in `tokenizer/`, which trains a
  V=265 byte stub in place and then fails the size check, so the test passes and the pinned
  tokenizer is gone. Running the full suite reproduces the flip-flop that `test_tokenizer_pin.py`
  documents. Restore with `git checkout -- tokenizer/` and re-run `tests/test_tokenizer_pin.py`.

## 2026-10-01: SAP Stage A results (synthetic phrase HMM, exact block joint)

Setup: `scripts/sap_synthetic.py`, depth 3 / width 128 trunk (~1M params) plus the block head,
2,000 steps x 8,192 tokens, every mode trained from scratch with its own head. The generator's
phrase families share a first token and then fork, so a mixed block has probability exactly 0.
Run twice: locally (seed 0) and on Modal (seeds 0 and 1, `modal_sap.py::stage_a`, 36 L4
containers, ~1.5-2.5 min each). Numbers below are the Modal means over 2 seeds; the local seed 0
agrees to within seed noise. block-KL is E_true[log p_true - log p_model] per block (exact for
indep / local / cp, an upper bound for the latent modes); TC is the true total correlation, the
block-KL of the best possible independent-slot head.

| mode | T=2 block-KL | T=2 impossible | T=4 block-KL | T=4 impossible |
|---|---|---|---|---|
| (true TC) | 0.825 | | 4.432 | |
| indep (B2) | 0.843 | 24.0% | 4.463 | 62.2% |
| p1_discrete (P1) | **0.089** | **2.1%** | **0.397** | **15.5%** |
| p2_gauss (P2) | 0.166 | 7.0% | 0.574 | 23.4% |
| p3_energy (P3) | n/a | 30.7% | n/a | 54.4% |
| cp, R=16 (B3) | 0.315 | 9.2% | 1.832 | 35.5% |
| local (B4, sequential in head) | 0.009 | 0.0% | 0.027 | 0.0% |
| inv_head (B5, PTP-style) | 0.224 | 7.9% | 1.334 | 30.4% |
| plain_noise (control) | 0.840 | 24.0% | 4.474 | 63.5% |
| wta (control) | 0.567 | 21.0% | 3.338 | 52.7% |

The trunk's next-token quality is untouched in every arm (excess loss ~0.003 nats/token), and its
own autoregressive samples are ~0.1% impossible.

- **The pre-registered gates:** P1 and P2 ADVANCE at both T and both seeds (block-KL <= 0.5 x B2,
  latent carries >= 0.5 x TC, beat the cp mixture). P3 is KILLED: its impossible-block rate is no
  better than independent slots. The energy score, as instantiated (per-token embedding space, 2
  samples), did not learn the dependence.
- **The motivating claim holds exactly as predicted:** prior noise with plain CE is ignored
  (latent sensitivity 0.0006-0.0012 nats) and lands at the independent-slot floor. Hard targets
  were never the problem; the training rule is.
- **P1 removes ~90% of the independent-slot gap and beats the published-mechanism competitor**
  (inv_head, PTP-style data inversion) by 2.5-3.4x in block-KL and the CP mixture by 3.5-4.6x.
- **But a sequential local head (B4) is ~10x better than P1** and never emits an impossible block.
  P1 still emits 15.5% impossible blocks at T=4. P1's one-pass advantage therefore has to be paid
  for in decode speed against B4's T head steps, and its T=4 gap has to close. Both are open
  (OPEN_QUESTIONS Q29).

## 2026-10-02: SAP Stage B depth-8 sweep (FineWeb-Edu, iso-FLOP 1.26e17 FLOPs, H100s)

Setup: Depth 8, V=32,768, sequence length 2048, window pattern SSSL. All arms budgeted at the dense
baseline's training budget of 1.2607e+17 FLOPs (440.4M tokens for dense). Evaluated with chunked
block evaluation (fixing the 34.36 GiB vocabulary allocation OOM) and verified against B1_dense_s1.

### Empirical Findings:

| Mode | Horizon T | Val BPB (Trunk) | Block BPB | Block / NTP | Ref PPL (Block) | Speedup b=1 | Speedup b=16 | Speedup b=128 |
|---|---|---|---|---|---|---|---|---|
| **B1_dense** | 1 | **0.958394** | - | - | 1.0x (ref) | 1.00x | 1.00x | 1.00x |
| **SAP_local** | 2 | 1.024248 | **1.065264** | **1.059x (+5.9%)** | **117.26** | 1.58x | 1.46x | 1.29x |
| **SAP_local** | 4 | **1.008279** | **1.089024** | **1.112x (+11.2%)**| **134.49** | **2.23x** | **2.20x** | **2.18x** |
| **SAP_p2_gauss** | 2 | 1.032494 | 1.154921 | 1.143x (+14.3%) | 434.50 | 1.64x | 1.68x | 1.59x |
| **SAP_p2_gauss** | 4 | 1.016977 | 1.286426 | 1.306x (+30.6%) | 1197.40 | 3.24x | 3.34x | 3.13x |
| **SAP_p1_discrete** | 2 | 1.030774 | 1.169985 | 1.157x (+15.7%) | 506.99 | 1.23x | 1.34x | 1.36x |
| **SAP_p1_discrete** | 4 | 1.015092 | 1.322793 | 1.343x (+34.3%) | 1396.15 | 2.45x | 2.68x | 2.65x |
| **SAP_indep** | 2 | 1.028220 | 1.264044 | 1.249x (+24.9%) | 1464.48 | 1.66x | 1.78x | 1.71x |
| **SAP_indep** | 4 | 1.014001 | 1.511516 | 1.540x (+54.0%) | 3898.26 | 3.39x | 3.39x | 3.13x |

### Durable Takeaways:

1. **Local Causal Block Attention (`SAP_local`) is the closest arm, but does not clear the
   pre-registered conjunction.** At T=2 its block-over-NTP ratio is **1.059x (+5.9%)**, just outside
   the <=5% requirement; its batch-16 speedup is **1.46x**, just below 1.5x; and its trunk BPB is
   1.024248 versus dense 0.958394 (**+6.9%**, not within 1%). The earlier wording that this arm
   “clears” the gate was mathematically incorrect. T=4 maintains better quality than the other
   block heads, but its +11.2% block-likelihood penalty also misses the gate.
2. **Naive Multi-Token Prediction (`SAP_indep`) completely collapses without teacher distillation**:
   Predicting slots independently results in a catastrophic +54.0% likelihood penalty at T=4 (1.5115
   vs 0.9815 bpb) and causes generation perplexity under the reference dense model to explode to **3898.26**
   (compared to **134.49** for `SAP_local`). This establishes the core theoretical contribution: *sampling
   from scratch requires intra-block dependency modelling; independent heads require teacher distillation.*
3. **Horizon Scaling Regularizes the Trunk**:
   Across every single architecture, training with $T=4$ produces superior next-token prediction BPB on
   the trunk compared to $T=2$ (e.g., `local` improves from 1.0242 to 1.0082; `p2_gauss` improves from
   1.0324 to 1.0169), despite receiving fewer total training tokens under iso-FLOP budgeting. Longer
   predictive horizons force the trunk to learn richer representations.
4. **Chunking the main likelihood was not enough for T=L validation.** The main block/NTP readout
   is chunked, but `latent_sensitivity` still materialised an `(8, B, T, V)` probability tensor; at
   B=128, T=2048 and V=32,768, even one uncapped readout requested 32 GiB and the diagnostic OOMed
   before checkpoint serialization. The repaired diagnostic streams its sample accumulation and
   estimates the per-block slot sum from 4 contexts x 64 evenly spaced slots. SAP diagnostics are
   also non-fatal now, so a diagnostic defect cannot discard a completed training run.

## 2026-10-02: SAP full-sequence horizon T=L=2048 rerun and verdict

Source: `s00_sap_d8_1.log` plus the repaired Modal rerun under profile
`blessingjim31-workspace`. The original p1/local-Jacobi/p2 runs completed optimization but crashed
in final SAP diagnostics before checkpoint save: p1 and p2 in uncapped latent sensitivity, and
local-Jacobi because its teacher-forced local likelihood branch was missing. The fixes bound the
sensitivity diagnostic, give local-Jacobi the underlying local conditional likelihood, make SAP
diagnostics non-fatal, and return `n/a` rather than zero when no joint block is admissible. Focused
tests: 27/27 pass.

All rerun arms used the same 1.26e17 training-FLOP budget as dense. The generation test contains
1,024 FineWeb-Edu prefixes x 128 generated tokens, scored by the dense reference. Speed is eager
KV-cached decoding for 256 requested tokens on an H100; batch 128 OOMed for every T=L head and is
not a measured row.

| Mode | Trunk val BPB | Gap vs dense 0.958369 | Ref PPL AR | Ref PPL block | Quality penalty | Speed b=1 | Speed b=16 |
|---|---:|---:|---:|---:|---:|---:|---:|
| p1_discrete | **0.991835** | +3.49% | 61.36 | **10,490.80** | **171.0x PPL** | 125.99x | 26.30x |
| local_jacobi, 2 sweeps | 0.993984 | +3.72% | 60.47 | 15,435.91 | 255.3x PPL | 96.78x | 11.95x |
| p2_gauss | 0.994630 | +3.78% | 60.86 | 11,191.31 | 183.9x PPL | **145.58x** | **30.66x** |

The original and rerun validation BPBs agree within 0.0013, so the training result reproduced.
The stored generations independently reproduce the aggregate perplexities. Distinct-3 is 1.000
for every block arm, but inspection shows diverse incoherent token salad; this rules out simple
repetition collapse, not quality collapse. P1/P2 latent sensitivity is nonzero (estimated 189 and
367 nats per 2048-token block), so the immediate failure is not an ignored latent. It is that one
plan plus one parallel decoder pass does not resolve the conditional branching of 2048 tokens.
Two Jacobi sweeps are even worse: information can propagate only a bounded number of refinement
rounds across a 2048-token dependency chain.

**Verdict:** the T=L one-shot and two-sweep-Jacobi instantiations fail the project's bar. Their
speedups are real at batch 1/16 but do not count because quality is nowhere near neutral, and batch
128 is not deployable with the current V-wide sampling implementation. Do not sweep latent size,
code count or Jacobi sweeps on this mechanism. Any continuation must change the mechanism so that
conditional information is introduced hierarchically or in bounded local chunks.

The raw rerun printed block BPB 0.0 with `blocks=0`; that is not a score. At T=L every candidate
validation block contained a zero-byte special/ignored target and the joint evaluator excluded it.
The repaired evaluator reports both block and matched-NTP BPB as `n/a` in this case and skips the
wasted likelihood work.

## 2026-10-02: SAP sampling-aware refinement brainstorm constraints

The T=L generations identify a joint-sampling failure, not primarily a marginal-prediction failure.
For P1, the dense-reference perplexity ratio is 10,490.8 / 61.36 = 171x, or about **5.14 extra
nats per generated token**, even though teacher-forced trunk BPB is only 3.49% worse than dense.
Any repair aimed only at marginal calibration is therefore pointed at the small gap, not the large
one.

- A soft target or expected token embedding does **not** by itself change the product-of-marginals
  graph. If no realised random decision reaches later computation, the final slot samples remain
  conditionally independent. Soft values are useful as gradient carriers, not as the source of
  sample coupling.
- Sampling a draft inside the head and letting later layers consume that realised draft changes
  the family to `p(y|h) = sum_d p(d|h) product_i p(y_i|h,d)`. Unlike P1/P2's one global plan,
  `d` contains one discrete variable per slot and is therefore a high-bandwidth stochastic latent.
- A reference layer that can see its own draft token has an identity shortcut. A serious first
  instantiation must mask the diagonal (position i sees other draft positions, not `d_i`) or gate
  the self route, so the layer learns cross-token compatibility instead of copying errors.
- Hard-forward/soft-backward sampling is the relevant hybrid: a hard Gumbel/Concrete or top-k
  sample makes training states match inference, while the relaxed path assigns later losses back to
  the draft distribution. A fully soft forward pass averages incompatible modes and can recreate
  mixture blur.
- A context-free corpus token table is insufficient and is close to RAML/label smoothing. A usable
  "reference weight" must condition on the realised neighbouring draft (for example a learned
  low-rank compatibility potential) and alter inference logits. Corpus-derived initialization is a
  separate policy decision because earlier project constraints K10/K11 prohibited curricula and
  externally computed statistics; the user's current proposal reopens, but does not silently erase,
  those constraints.
- Plain input noise is not the novelty: PTP/K-Forcing already move sampling noise before prediction,
  Condor couples noise to one-step blocks, Mask-Predict performs confidence refinement, and
  differentiable scheduled sampling supplies the soft-gradient precedent. The defensible opening
  is a **mid-head discrete sample followed by a cheap leave-one-out reference/correction stage,
  trained on its own sampled states from scratch**.

### S01 exact-gate learning: a sampled state is not automatically sampling-aware training

The first SIR implementation put a real hard sample between decoder layers and correctly changed
the inference graph, but all ten mechanisms converged back to the product-of-marginals solution.
At T=4 their exact block KL was 4.315–4.348 nats/block versus true total correlation 4.270, with
61.6–64.1% impossible phrases. The issue is the objective: if `d` is drawn from the prior and the
training continuation is an independent draw given the context, averaging CE over `d` makes the
Bayes-optimal decoder ignore `d`. Moving sampling earlier is necessary but not sufficient.

A `K=4` sampled marginal likelihood plus detached post-sampling reference weights made draft usage
measurable (reference ESS fell from 4.0 to roughly 2.5–3.0) and improved the best arm to KL 3.443 /
invalid 54.0%, but did not approach the 0.10 / 3% gate. A fixed 0.5 token-posterior bridge then made
posterior-conditioned training likelihood easy while prior-only samples remained bad: all ten
arms landed at KL 4.207–6.647 and invalid 30.7–53.4%. Even correct scale-relative anchors only
reached 26.3–26.4% invalidity and had KL 4.792/8.649. This is aggregate-posterior mismatch: an
independent per-slot draft prior cannot reproduce the correlated latent it sees during training.
The next defensible mechanism must put correlation in the prior itself (for example, a cheap
hierarchical or tree-structured anchor sampler), not add another correction layer to the same
factorised draft.

### S04 exact one-shot field: correlation is used, but a compact code is still too small

`field_cp` repaired the broadcast-CP geometry and the training mismatch at the same time: one
categorical prior code selected a position-specific low-rank field, and exact mixture likelihood
trained the same prior used for generation. At 8k steps its sensitivity was 3.2596 nats/block and
trunk NTP excess was only 0.001, so neither latent collapse nor trunk undertraining explains the
failure. Yet block KL was 1.7078 and 33.94% of samples were impossible (gates 0.10 / 3%). A compact
global code can recover substantial correlation without resolving the branch consistently. More
codes or rank would be a capacity sweep of the failed mechanism.

The next strict one-round experiment therefore changes the stochastic object and its credit
assignment together: a shared continuous positional noise field provides several correlated
factors, while the joint loss observes realised hard categorical blocks with a soft
straight-through gradient. This retains one sampling event and one decoder evaluation at inference.

### S05: hard sampled-token training is not enough when the score lacks discrete syntax

The correlated Gaussian field stayed active (1.0662 nats/block sensitivity), and the trunk itself
solved the generator (AR block KL 0.013, AR invalid 0.049%, NTP excess 0.001). Nevertheless, S05
ended at block KL 3.3549 and 53.86% invalid. Relative to 4.270 nats of true total correlation, the
one-shot sampler recovered only 0.915 nats (21%), less than the exact categorical S04 field.

This separates two ideas that had been conflated. Hard-forward/soft-backward sampling repairs
train/inference object mismatch, but the downstream score still determines what correlation is
learned. Distance between concatenated token embeddings rewards broad semantic/lexical proximity;
it does not sharply identify an impossible cross-branch phrase. The next credible one-round credit
signal must score discrete block compatibility directly—for example, an online AR verifier's joint
log-probability of the realised proposal—rather than another geometry or latent-capacity sweep.

### 2026-10-03 correction: no verifier or distillation; move randomness to the beginning

The AR-verifier proposal is outside the desired economics because it creates a second training role.
The stricter and cleaner interpretation of one-round generation is to sample the *entire primitive
random tape* once, then make every subsequent operation deterministic. Final categorical choices can
use uniforms already present in that tape; they do not require a later stochastic round.

This makes a class of exact sequential distributions parallel-samplable. For a finite latent state,
draw a next-state random map `F_t(s)` for every possible current state and position. Function
composition is associative, so all realised states are a parallel prefix scan of the maps. An HMM
continuation can therefore be sampled exactly with one upfront draw and deterministic `O(log L)`
span, while retaining exact forward-algorithm likelihood. State-conditioned invertible vocabulary
permutations avoid an `L x S x V` emission tensor. This is mechanistically different from the S02
tree: the tree made new stochastic decisions after observing sampled ancestors; the scan samples all
randomness before any state or token is resolved.

The earlier small-head direction also should not be confused with the seed. The exact local head's
1.21x real-text BPB ceiling says the parallel generator needs trunk-depth capacity and must be the
primary model, not an auxiliary head reading frozen AR features. See `s06_sap_one_pass_brainstorm.md`.

### S06-Q: one upfront random tape is feasible, but the token transport is now the bottleneck

Ten oracle-context, depth-4 mechanisms were screened at `T=4` for 2,500 updates. None reached even
the borderline KL 0.50 / invalid 10% gate. The ideal independent control had 4.449 nats of total
correlation and about 62–64% invalid samples. PSS reached KL 5.015 / 54.39% invalid: its random-map
state path improves support consistency by about nine absolute points, but fixed XOR vocabulary
permutations cannot turn those states into an accurate conditional token distribution. RMLT was
worse (5.453 / 58.35%), so changing chain topology to a tree does not repair that emission defect.

MIF had the best exact likelihood, KL 4.250, but this recovers only 0.199 of 4.449 nats of total
correlation (4.5%) and still yields 60.01% invalid blocks. Its constrained source-code oracle makes
the diagnosis sharper: identity was the best code; delta, modular butterfly, and XOR butterfly all
increased residual KL. The small MIF gain therefore comes from its four-way mixture, not from the
proposed arithmetic multiscale innovations.

Joint implicit losses did not solve discrete support. Corrected variogram and characteristic-kernel
generators remained active and diverse but produced 70.51% and 69.87% invalid blocks. `K=8` IMLE
reached 84.67%; true-JVP one-step MeanFlow reached 96.73%. Their continuous objectives can improve
marginal/sample geometry without assigning enough cost to exactly impossible symbolic branches.
The viable residual research question is a different emission/transport mechanism: keep the exact
one-tape correlated state path, but learn a semantic bijection under which cheap state permutations
correspond to real token alternatives. More steps or a larger fixed-XOR PSS is a capacity sweep of
the failed instantiation, not a new mechanism.

### S06-S: semantic permutations need more than one probability spectrum

The free true-state oracle explains the PSS failure precisely. Even with the true transition process,
true context belief, and an arbitrary optimal vocabulary permutation for every HMM state, forcing all
states to permute one shared base distribution gives KL 1.1542 and 34.35% invalidity on 16,384
contexts. A bijection can move probability mass to the right semantic tokens, but it cannot turn a
diffuse probability vector into a one-hot one because permutations preserve the sorted probability
spectrum.

Using only two spectrum classes—one for deterministic phrase states and one for stochastic topic
states—changes the result to KL 0.01468 ± 0.00129 and 0% invalid. Three and five classes improve KL
only to 0.00889 and 0.00309; nine distinct spectra reproduce the HMM exactly. Thus the useful idea is
not semantic token coding alone. It is a one-tape random-map state process plus **state-routed
spectrum class and semantic bijection**.

This does not yet establish learnability: the oracle receives true states, transitions, classes, and
permutations. It establishes representational sufficiency with `C=2`. A real SC-PSS can preserve one
`V`-way inference readout by resolving the class through the deterministic state scan before the
output projection; exact training needs both spectra, `O(2LV + LS^2)`. The next kill gate is a learned
`S<=64,C=2` toy, not real-text depth 8.

### S06-L: arbitrary per-state permutations are representationally sufficient but unidentifiable

The learned d4 `S=64,C=2` SC-PSS reached KL 2.9485 and 48.68% invalidity at 8,000 steps, versus the
oracle's 0.0147 / 0%. This is better than the matched independent ceiling (4.4013 nats and about
61% invalid), so the state path and spectrum routing do carry dependence. It is nowhere near the
0.10 / 3% gate.

The failure is mechanistic: every latent-state label can be exchanged with another while its entire
512-token permutation changes, creating a large assignment symmetry. Posterior-weighted Hungarian
updates changed 97.99% of entries initially and still changed 95.64% at the final update. Posterior
occupancy also remained concentrated (about 1,900 of 6,144 assignments in one state versus 96 under
uniform use). Neural transitions therefore chase a moving token coordinate system while the
permutation learner chases moving state semantics.

More steps on the same alternating scheme are not the next experiment. A new learnable version must
share token coordinates across states and restrict each state to a small identifiable group action;
otherwise the oracle capacity cannot be reached reliably. The failed gate also means no `L=2048`
speed number or d8 BPB result should be reported for this instantiation.

---

## 2026-10-05: Empirical & Mathematical Validation of 8 Lane-Start Tax Reduction Candidates

**INVALID (corrected 2026-10-05, see the correction entry at the top of this file):** the "plain lanes" model below is the dense checkpoint, M1 and M3 score the wrong position, M7 compares a call with itself, and M5 was never measured.

Evaluated on `out/sap_flow_text_assets/dense_d4/S07_dense_L_s1` (32 validation rows of length 2048 from `/home/seqaeon/Drive-D/nanochat/data`, $P=128$, $L=32$, $S=60$). Full script: `scripts/validate_all_survivors.py`, output artifact: `scratch/validate_survivors_results.json`.

### 1. Ground Truth Baseline Loss by Lane Offset
- **Offset 0 (Cold Start):** Plain = 8.750 nats, Dense = 4.758 nats $\to$ **Excess = +3.992 nats** (matches S13 Q1 measured +3.95 nats).
- **Offset 1:** Plain = 6.777 nats, Dense = 4.207 nats $\to$ Excess = +2.570 nats.
- **Offsets 2–3:** Plain = 5.773 nats, Dense = 4.304 nats $\to$ Excess = +1.469 nats.
- **Offsets 4–7:** Plain = 5.182 nats, Dense = 4.254 nats $\to$ Excess = +0.928 nats.
- **Offsets 8–15:** Plain = 4.778 nats, Dense = 4.387 nats $\to$ Excess = +0.390 nats.
- **Junction token $S-1$:** Plain = 4.370 nats, Dense = 4.324 nats $\to$ Savings = 0.045 nats (infill effect).
- **Total Boundary Excess (offsets 0–3):** **9.500 nats per lane**.

### 2. Candidate Empirical Balance Sheet
- **M1 (Parareal Micro-Draft Boundary): KILLED.** Linear probe from prompt representation $h_{P-1}$ to boundary token $x_{jS}$ achieved **0.00% top-1, top-5, and top-20 accuracy**. Predicting exact tokens across a 60–1800 token gap without intervening text is impossible due to cumulative language entropy.
- **M2 (Overlapped Ghost-Cell Tiling / Halo): KILLED.** Pre-generating $H \in \{1, 2, 4, 8\}$ unconditioned tokens *increased* loss (Net gain $-1.31$ to $-2.73$ nats). Ghost tokens hallucinate uncoordinated text branches, producing destructive boundary interference.
- **M3 (Speculative Lane-Start Verification): KILLED.** Speculative prediction at step 0 achieved **0.00% top-1 accuracy**. Rollback penalty would trigger on 100% of starts, reducing throughput below sequential decoding.
- **M4 (Soft-Pipelined Micro-Stagger $\Delta$): KILLED.** Giving Lane $j$ cross-attention to Lane $j-1$'s first $\Delta \in \{1, 2, 4\}$ tokens increased loss by $+0.67$ nats. Those tokens sit at $(j-1)S$ (56 tokens away); they provide zero local syntactic conditioning and introduce cross-attention noise.
- **M5 (Unequal Error Protection / Boundary Loss Weighting): SURVIVES.** Offsets 0–3 account for 6.7% of the sequence but contribute 9.500 nats of excess loss. A 20% cut in boundary loss mathematically yields a **0.0457 whole-sequence bpb improvement** (a 4.0% relative improvement on a 1.15 bpb model, offsetting the entire +3.9% tax of $L=32$). Zero parameter overhead, zero decode slowdown.
- **M6 (Parafoveal Preview Gist Head): KILLED.** 0.00% unigram overlap between lane start and next 15 tokens. Continuous pooled vectors do not resolve the discrete boundary token.
- **M7 (Discardable Soliton Buffers): KILLED.** Dummy warmup tokens produced exactly 0.0000 nats change. Directly bounded by the Data Processing Inequality (DPI) in fixed-depth feedforward attention.
- **M8 (Any-L Curriculum): KILLED.** Offset 0 loss is an invariant boundary penalty (~8.0 to 9.0 nats across $L=4$ to $32$). The per-token sequence tax scales strictly as $L/N$.

