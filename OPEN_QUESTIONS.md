# Open Questions

Unresolved decisions, deferred work, and things that need research before they can be settled.
Entries get removed when they are answered, and the answer goes to `LEARNINGS.md`.

Latest SAP scope/evidence audit: `s07_sap_mechanism_brainstorm.md`. Historical v3/v4
and S06 statements below are retained as chronology, not overriding user decisions.
Current cross-session recommendation: `sap_next_mechanisms_review.md` incorporates
S08/S09/S10 and supersedes the untested-S07 questions below. Open next mechanisms:
the global-flow/local-joint hybrid for T=L, and paragraph-aligned lanes as a
separate smaller-T route. S10's AR-mode distillation is not assumed authorized
under this conversation's no-distillation boundary.

---

## Sampling-aware pretraining, SAP (opened 2026-10-01)

See `sap_research_plan.md` v3. One head emits T tokens per trunk pass; a plan latent drawn
once per block makes them agree.

### Q24. Which training signals count as "no teacher forcing"? PARTLY SETTLED 2026-10-01.

The user accepts an ELBO with a recognition model (P1, P2) and a strictly proper scoring rule
(P3, energy score). Data-inverted noise (PTP-style) and self-consistency against the model's
own next-token mode count as teacher forcing or distillation, and appear only as competitor
baselines. Still open: whether the recognition model of an ELBO reading the true block is
itself acceptable in the paper's framing, since it conditions on ground truth during training.
It is standard VAE practice and nothing reads it at inference, but it should be said plainly.

### Q25. What block fraction keeps the head inside the FLOPs budget?

The readout is T rows through the V x d head per carried position. At V=32,768, d8, a
fraction of 1/8 costs about +22% training FLOPs (back-of-envelope); Stage B defaults to 1/16
(and 1/64 for the cp baseline). Unknown: how much head quality the smaller fraction gives up.
It only learns from 1/16 of the positions. Measure block bpb at 1/32, 1/16 and 1/8 at matched
FLOPs before fixing it.

### Q26. Is the phrase-HMM testbed a valid proxy for language?

It has exact block probabilities and impossible blocks, which is why Stage A uses it. It does
not have long-range structure or the heavy-tailed branching of real text. A method can pass
Stage A and still fail Stage B. Stage A is a kill filter, not evidence for the paper.

### Q27. Will the Gaussian plan (P2) have prior holes?

Prior samples landing where the decoder never trained produce incoherent blocks while the
ELBO looks fine. Gate: prior-sample quality within 2x of posterior-sample quality. If it
fails, fit a one-step generator over the aggregate posterior (MeanFlow, teacher-free) before
killing P2.

### Q28. Is the novelty defensible against VADD and Block Transformer?

VADD (ICLR 2026) already uses a recognition model and an ELBO to capture inter-token
dependence, in masked diffusion. Block Transformer already gets 10-20x throughput at equal
perplexity from scratch, with a sequential local decoder. Read both in full before Stage B.
The comparison that decides the paper is P1/P2 against Block Transformer at matched FLOPs and
matched decode cost.

### Q29. Does P1's single pass beat a sequential local head, and why does P1 leave 15% impossible blocks at T=4? ANSWERED 2026-10-03.

Stage A (LEARNINGS 2026-10-01): P1 reaches 9-11% of the independent-slot gap; the local head (B4,
T sequential steps inside the head) reaches ~1% with no impossible blocks. Two things decide the
direction. (1) Speed: P1 decodes a block in G tiny prior steps plus one decoder pass, B4 in T
decoder passes, each with a V-wide readout; Stage B's decode benchmark must show P1 faster at
batch 1/16/128, or B4 (Block-Transformer-like) simply wins. (2) Quality: is the T=4 gap capacity
(C^G = 65,536 plans, a 2-layer decoder, 2,000 steps) or structural (the prior cannot match the
aggregate posterior)? One diagnostic separates them: P1 at C in {16, 64} x steps in {2k, 6k}, and
posterior-sampled vs prior-sampled impossible rates. If posterior samples are clean and prior
samples are not, it is the prior (fix: a better prior); if both are dirty, it is capacity.

The d8 Stage B comparison settles the first half against P1: the local head is much better in
block likelihood and reference generation quality. P1 is only modestly faster at T=4 and neither
arm clears the pre-registered conjunction. The T=L follow-up is more decisive: P1 is 26.30x faster
at batch 16 but its block generations have 10,490.8 reference PPL versus 61.36 for AR, so the speed
does not count; two-sweep local-Jacobi is worse at 15,435.9. Still open is the mechanistic cause of
P1's T=4 prior/posterior gap. S02 answers it: independent anchors remain at KL 3.321 / 52.29%
invalid after 32k steps and 2x width, while an oracle-trained refiner reaches 2.25% invalidity when
given coherent anchors. The prior factorisation, not refiner capacity, is the bottleneck.

### Q30. Can a hierarchical block mechanism preserve most of T=L's speed without one-shot conditional collapse?

The T=L result kills one global plan + one parallel slot pass and two global Jacobi sweeps. A new
mechanism must introduce conditional information at more than one scale while keeping sequential
depth sublinear in T—for example coarse anchor tokens followed by independent bounded local chunks,
or a tree of plans with O(log T) refinement rounds. Before implementation, derive the exact readout
and decoder FLOPs and pre-register a quality-neutral speed target against the T=2/4 local head; more
latent codes or more sweeps on the existing head do not answer this question.

**2026-10-02 brainstorm funnel.** The leading mechanism is Sample-Inject-Reference (SIR): split
the current two-layer block decoder, produce and hard-sample a draft after the first part, inject a
sparse relaxed embedding of that draft, and let the remaining reference layer revise the block.
Mask the draft self-edge so each final slot must use other realised choices. Train on self-sampled
drafts, with a hard forward path and straight-through top-k relaxation. The first cheap kill test is
the phrase-HMM: SIR must beat P1's T=4 block-KL 0.397 and impossible rate 15.5% while using at most
two V-wide readouts. On FineWeb-Edu the pre-registered target should be dense-reference generated
PPL within 10% of AR (equivalently, excess NLL <= `ln(1.10) = 0.095` nats/token) and batch-16
wall-clock at least 2x AR; a softer gate would allow another fast but unusable head.

Two policy constraints need an explicit decision before implementation. A gradual ground-truth to
self-sampled roll-in violates the earlier no-curriculum constraint (K10), and a precomputed
corpus-confusion/PMI table violates the no-external-statistics constraint (K11). The architecture
does not require either: on-policy drafts from step one and an end-to-end learned compatibility
potential are the clean default. If corpus statistics are admitted, use them only to initialise or
regularise a context-conditioned reference potential, not as context-free soft labels.

**2026-10-02 measured correction.** Pure on-policy CE did not merely miss the gate: all ten SIR
arms reproduced total correlation (`KL 4.315–4.348` vs TC `4.270`, invalid `61.6–64.1%`). The
reason is mathematical: under `d ~ p(d|h)`, the observed `y` is independent of `d`, so
`E_d[-log p(y|h,d)]` rewards ignoring the draft. A `K=4` Monte-Carlo marginal with post-sampling
reference weights reduced the best KL to `3.443` and invalid rate to `54.0%`, showing that credit
assignment works but does not supply enough branch correlation. S01-v3 then mixed observed tokens
and prior samples at a fixed rate of 0.5. All ten arms still failed: KL `4.207–6.647`, invalid
`30.7–53.4%`. Correct T=4 hierarchical strides lowered invalidity further to `26.3–26.4%` but
worsened KL (`4.792` anchor, `8.649` pyramid). The open mechanism question is now how to sample a
*correlated prior* whose draws match the branch-correlated training latent; another independent
per-position prior or refiner-capacity sweep does not answer it. The depth-8 H100 gate remains
closed.

**2026-10-03 correlated-tree result.** A balanced ancestral tree reduces T=4 block KL from the
best independent-anchor value of 3.321 to 0.112-0.122 and invalidity from 52.29% to 3.17-3.91%.
The shallow tree is the better parallelism point but still misses both registered thresholds at
32k (KL 0.1223, invalid 3.17%); the full tree also misses (0.1118, 3.91%). Therefore correlated
ancestry is validated as the mechanism, but Q30 remains open at useful parallel depth and the d8
gate remains closed. Next work must improve dependence inside the shallow tree's parallel fill,
not add capacity or training to the independent-anchor family.


### Q31. Does an exact sampling cut close the one-pass gap at T=4, and at which trunk gradient? (opened 2026-10-03)

SAP v4 (`sap_research_plan.md` v4). Three policy points were settled by the user on 2026-10-03:

- **"One pass"** means one neural head pass plus a cheap sampler (a chain over ~64 candidates per
  slot, no network layers), with at most one mid-pass sampling cut followed by one reference layer.
- **Corpus tables are admitted** (the user's idea: "training-set distributions which we compute a
  table for before training"), counted on the training split only. This lifts K11 above. By the
  per-slot corollary (LEARNINGS 2026-10-03) they cannot make a block coherent as context-free soft
  labels, so they enter as a CRF support mask, a PMI floor and corpus classes; the n-gram soft
  target survives only as C-soft, a labelled gradient-variance reducer on co-trained arms.
- **Trunk gradient**: compare detach (0), 0.1 and full co-training (1) from scratch at d8.

Open, in the order the stages answer them:

1. Is coherence the bottleneck on real text at T=4 (Stage 0, O1: TC_4 >= 5% of block NLL)?
2. Does any exact-cut head pass the phrase-HMM gate (KL <= 0.10, invalid <= 3%) where the full
   tree (0.11, 3.9%) did not, and with one neural pass instead of three rounds?
3. On the frozen dense d8 trunk, does the best head reach block bpb <= 1.05x NTP at T=4 with a
   CUDA-graph speedup >= 2x at batch 16? If no mechanism does, the one-pass direction closes.
4. Does the frozen-trunk screen transfer to from-scratch co-training (g = 0.1, 1), or does the
   trunk tax (+3 to 15% so far) reappear? If the capacity tax dominates instead, C-depth (slot
   queries through the top trunk layers) is the next mechanism.

Status 2026-10-03. Items 1 to 4 are answered: coherence matters, and no one-pass head reached the gate. C-depth (trunk-depth slots) became the mainline. It is not one pass: it is one trunk pass plus T-1 sequential passes through m layers. That drift from the one-head seed is flagged here and in the plan.

### Q32. Which comparison is the paper's bar for a lossy block decoder? (opened 2026-10-03, user decision)

Against the same dense model, trunk-depth slots pay about 2 points of bpb per trunk layer a slot skips, and the cost is convex. None of the following reaches C <= 1.02 at 2x:
- late entry, rolling entry and skip-middle at d4 and d8;
- the bisection tree;
- adaptive block length, even with an oracle router (1.5x at ratio 1.0).

The other standard comparison for a lossy decoder is the cheapest alternative way to decode as fast: a shallower dense model of the same width, at the same training FLOPs.
- If SAP sits clearly below that frontier, it is a performance gain at matched decode latency and matched training compute.
- That is the bar's first clause read at a matched cost, not "neutral against the same model".

The frontier (`modal_sap.py::s03_dense_frontier`, width 256, 1 to 3 layers at d4's budget) is being measured. Whether that reading is acceptable for the paper is the user's decision.

Measured at d4 the same day: the reading fails as well.
- Every SAP point is 4 to 7 points of C above the dense depth frontier at matched speed and matched training FLOPs. Example: roll m=1 at 1.73x, C 1.086, against 1.041 for the frontier.
- So at d4 there is no comparison under which trunk-depth slots clear the bar.
- A d8 frontier (4, 6, 8 layers at width 512) is running to check whether the gap shrinks with scale.

Resolved (2026-10-03): the frontier is flatter at d8, with 4 layers at 1.79x and C 1.026. SAP-d8 sits 3 to 5 points above it. Neither reading of the bar is met by any SAP mechanism built so far.

## Fully binary transformer (opened 2026-09-11)

See `fully-binary-transformer-plan.md`. These are the four Phase-0 gates; all but Q22 run on
this machine against checkpoints already on disk. Q22 is answered.

### Q20. Where in a whole transformer is floating point actually load-bearing?

Forward-only on `out/dense_d8_V32k_model_001014.pt`, a full depth-8 V=32,768 dense checkpoint.
Binarise one component class at a time holding the rest fp, and measure val bpb: attention
`c_q/c_k/c_v`, attention `c_proj`, FFN `c_fc`, FFN `c_proj`, head, embedding, norms, then
per-layer within whichever class is most sensitive.

This replaces an earlier head-only formulation of Q20, which was wrong twice: it used the code
head arc's leftover artifacts (`head_d8_v131k.pt`, V=131,072, one component) to plan a
whole-model direction, and it read a projection oracle with the free-fit asymmetry. See
LEARNINGS, "A projection oracle and a free-fit oracle have OPPOSITE pass/fail asymmetries".

Run BOTH variants and label them:
- `sign(W)*scale`, zero degrees of freedom. Penalises, because the weights were never shaped
  for the constraint. A pass is informative, a failure is weak.
- fitted 1-bit `W'` at width `d_b` minimising output error against the frozen teacher on real
  activations. Flatters. A wide failure is informative, a pass is weak.

Answers the ordering of the Phase 1 ladder, and puts the two interfaces and the body on one
footing for the first time, which is what section 2.3 of the plan asserts without evidence.
Does not answer whether a from-scratch W1A1 model reaches the frontier; nothing offline does.


### Q21. Does quality in a binary network track gradient-sign agreement or gradient magnitude error?

A binary weight's update is a flip decision, so the backward pass has to deliver at most one
bit per weight per step. If final bpb is a function of the sign-agreement rate with exact
backprop and roughly invariant to magnitude error, the whole programme of cheapening the
backward pass is licensed and the counter optimiser follows. If it tracks gradient MSE
instead, the backward contribution collapses to low-bit backward with error feedback, which is
published, and the direction loses one of its four contributions.

Cheap: a tiny model at `tests/test_tiny_model.py` scale, per-layer agreement against exact
backprop, bpb plotted against agreement rate.

### Q22. PARTLY ANSWERED 2026-09-12, magnitude REOPENED: b1 works on sm_86 in both AND and XOR modes, bit-exact. The speed ratio is NOT measurable here: the same binary gives 4.19x and 1.93x in two machine states, because the GPU is hard-capped at 20 W of a 60 W default and the governor trades clock for watts differently per kernel. Needs root to lock clocks (`nvidia-smi -pl 60`, `-lgc`). See LEARNINGS.

`b1` MMA with `bmmaBitOpXOR` is deprecated and the XOR operand was removed from the hardware
in sm_90; it is emulated with ANDs and measured up to 5x slower on GH200. AND-mode `b1`
survives, and on Hopper the fast `16x8x256` fragment is reachable only through inline PTX, not
WMMA. Blackwell `tcgen05` is unverified.

Needs the target GPU, extending `scripts/p10_mfu_microbench.py` and `scripts/p15_gemm_probe.py`.
NOTHING TRAINS BEFORE THIS RUNS. The three outcomes and their consequences for what the paper
claims are tabulated in plan section 3.4; the short version is that if there is no competitive
1-bit GEMM and decode gives under 2x, the speed claim is dropped and the abstract says so.
This project has five recorded cases of an access pattern costing more than the arithmetic it
saved, and a bit-op table is not a speedup.

### Q23. What is the state-bytes break-even at depth 8, V=32,768?

The kill criterion in plan section 8 for the W1A1 body currently carries a 0.10 bpb
placeholder. It must be replaced by a derived number once the cost model (bit-ops, energy,
training-state bytes) reports, in the same way every FLOPs break-even in this repo is read off
the budget-pinned depth ladder rather than typed. Note the chord-versus-curve trap: the dense
ladder is convex, so a chord between two legs lies above it and flatters the margin.

---

## Nonnegative factorized output heads (opened 2026-09-06)

### Q15. ANSWERED 2026-09-06: realisability costs 30x the free-fit oracle, and the reason is a missing prior.

First V=32,768 depth-8 sweep, budget pinned at 440.4M tokens:

| arm | FLOPs vs dense | break-even | excess | margin |
|---|---|---|---|---|
| NFH_cp_R1 (LightRNN) | 0.6494x | +0.0317 | +0.5410 | -0.5093 |
| NFH_cp_R32 | 0.6817x | +0.0281 | +0.1529 | **-0.1248** |
| NFH_global_m256 | 0.6511x | +0.0315 | +1.0442 | -1.0127 |

**The claim survived**: 0.388 bpb between R=1 and R=32, about 97x the noise floor, so
probability-space coupling is the mechanism and it is what the binary code head was
missing. **The head still loses to dense** by 0.125 bpb against its budget, and the
free-fit oracle had said 0.0051, so realisability is 30x the oracle.

The checkpoint says which kind of realisability, with no activations needed. The static
background is uniform to four decimal places and the components are near-orthogonal and
all in use, so the head holds no unigram prior and its mixture optimises fine. See
LEARNINGS, "Two hypotheses about the first d8 sweep". `--sch-nfh-smooth` follows directly;
every collapse-oriented fix is ruled out.

**What is still open** is whether smoothing plus a wider factor map recover the 0.125.
Pre-registered: if they recover less than half of it, the realisability floor is real,
this family does not reach the Pareto frontier, and the direction closes.

### Q16. Does an untied Tucker core beat tied CP components at the same cost?

`cp` ties component r's factors together. A shared context-independent core
`G[r_1..r_G]` gives `R^G` effective components for `R` context cost: at R=16 that is
4,096 combinations from 48 factor vectors, and the core contraction is `R^G` = 4,096
operations per token against the projection's 1.3M.

Deliberately NOT built. It needs its own contraction on both the training and the
eval paths, and writing tests for a mode the first sweep will not run is how a
codebase acquires unexercised branches. `NFH_MODES` is `("cp", "global")` until the
CP result says the direction is alive. The CP number is a lower bound on it by
construction, so nothing is lost by waiting.

### Q17. Do several bijections beat one fitted one?

`sch_nfh_views` is implemented: view 0 takes the configured assignment and views 1..
take random ones, mixed uniformly. It is the sketching argument, several independent
hashes rather than one perfect one, and it attacks `cp`'s main fragility, which is
that the whole head rests on a single fitted permutation. Not in the default sweep
because L=2 at R=16 has to be compared against L=1 at R=32 at matched cost, and that
comparison is only worth its GPU time once R itself is settled.

### Q18. The composers that were designed and deferred

Each is a one-flag addition on top of `cp`, each has a measurement behind it, and
none is in v1 because the first sweep does not need them:

- **Entropy-adaptive R.** A variable-R mixture is still exactly normalised, so R can
  be routed per token with zero bias. Measured spread at R=8: median 0.0126 nats,
  p90 0.129, p99 0.375, and correlation with context entropy 0.783; the easy half
  costs 0.0087 and the hard half 0.0876. Cost is set by the tail, not the median.
- **Multi-token prediction on one shared code table**, which amortises the head over
  two positions and is independently a quality gain.
- **Progressive rank growth**, R from 4 to 32. Exactly normalised at every R, so
  growth is a no-op on the objective.
- **A unigram background component**, one component pinned to the marginals, as tail
  calibration insurance for 1/R of the budget.

### Q19. `scripts/c13_proposal_head.sh` arms are not covered by any test

`_sweep_arms` in `tests/test_code_head.py` reads top-level `VAR="..."` assignments,
and c13 builds its arms from a multi-line `PROP="..."` that the extractor cannot
read, so it silently extracts nothing. The extractor now accepts both run-line
conventions (`run TAG "$DEPTH" ...` and c13's `run TAG ...`), which was the other
half of the gap, but multi-line assignments still are not handled. c14 works around
it by keeping every value on one line and by deriving-then-checking the two values
that cannot be (`LOWRANK_C`, `DIMS`). Fixing the extractor properly belongs with c13.

---

## Structured code output heads (opened 2026-08-30)

### Q1. REOPENED and answered for one-hot codes, 2026-09-04.

Closing this as moot was wrong: it assumed the target was freezing `Phi`, worth one third of
the head, when the target should be structural cheapness, worth most of it. For a **K-ary
product code** the fast transform exists and is trivial. `Phi` is one-hot within each group,
so `g(h) @ Phi^T` is a gather and add costing `V*g` rather than `V*M`. At V=131072, d=768,
g=8, K=256 that is 44x fewer FLOPs than the dense head with a *higher* rank ceiling. See
`output-head-efficiency-directions.md` section 3.

Still open for truncated *monomial* expansion, which is what was originally asked.

### Q1 (original). Is there a fast transform for a truncated-order monomial expansion?

The cost model rests on `g(h) @ Phi^T` being a dense `M x V` matmul. A Walsh-Hadamard
transform computes the *full* order-`B` expansion in `V log V`, but we deliberately truncate
at order `k`, and it is not obvious whether the truncation admits a fast algorithm or
destroys the structure the transform exploits.

**Why it matters.** If one exists, the code head becomes cheaper than the softmax at every
order rather than only when `M < d`, and the efficiency story stops being a trade. If it does
not, the paper's cost section stands as written.

**How to resolve.** Literature check first (subset-sum / zeta-Mobius transforms over the
subset lattice are the obvious place to look), then a prototype benchmarked against the dense
matmul at the widths we actually use.

**Position for now.** Section 3.5 of the plan says flag it, do not promise it. The current
FLOP accounting assumes no fast transform, which is the conservative direction.

### Q2. REOPENED 2026-09-04.

Closed on the same bad framing as Q1. The observation that stands is narrower: at d=256 the
order-3 *binary* head costs 7.62e7 FLOPs against the dense head's 5.03e7, so sparsity would
have to overcome a 1.5x deficit before saving anything. That is an argument against high-order
binary codes, not against structured heads.

### Q2 (original). Can `Phi`'s sparsity at high order be turned into a real speedup?

Expected density is roughly `(3/2)^B / M`, and the measured values confirm it falls with
order: 0.258 at `B=64, k=2`, 0.153 at `B=17, k=3`, 0.091 at `B=15, k=4`. So the order-4 arm,
which is the expensive one, is also the sparsest.

**Why it is not obviously a win.** Sparse kernels need structured sparsity to beat dense
tensor cores, and this sparsity is unstructured. At 9% density a dense bf16 matmul may still
win outright.

**How to resolve.** Measure. `nanochat/code_metrics.measure_head_cost` already reports head
wall-clock and peak memory separately from the model total, so a CSR or block-sparse
prototype can be compared directly. Do not assume the saving; the plan is explicit about this.

### Q3. Which vocabulary do the headline experiments use?

`V = 32768` is the right vocabulary to build on (`2^15` exactly, so `B = 15` with no rounding
waste, which makes the Phase 0 rank check unambiguous) and the wrong one to publish on. At
`B = 15` the code is a bijection onto `{0,1}^15`, so the minimum Hamming distance is 1 and the
ECC-versus-semantic comparison is *undefined*; measured sampled minimum distance is 1 there
against 14 at `B = 64`. The tail is also too thin: at 32k even the rarest tokens occur
thousands of times in 600M tokens, so the data-scarcity regime the whole hypothesis lives in
does not exist and the decile crossover will not fire.

**The risk if this is not settled.** Reading a 32k null result as a dead hypothesis, which
would be a false negative on the central claim.

**Blocked on.** Compute. `c01`'s `vocab` group and all of `c04` run at 131k, but both need a
separate tokenizer trained on the same corpus plus its own frequency table. 262k is worth
doing if affordable, because one point is not a trend.

### Q4. Should the held-out-vocabulary control mask targets or remove tokens entirely?

Both are implemented (`--sch-holdout-mode target|full`) and they prove different things.

- `target` masks the held-out ids as prediction targets only. The head receives no gradient
  for producing them, which is exactly the output-side claim, while their input embeddings
  still train. Clean, but a reviewer can say the model still saw the tokens.
- `full` also rewrites them in the inputs, so the model never sees them at all. This is the
  honest end-to-end extension test, and it is what a coded *input* side exists to support.
  It does perturb the context distribution, which is a confound of its own.

**Current position.** Run both and quote both; `c03` and `c04` do. What is unresolved is
which one the headline number comes from, and that should be decided before the results are
seen rather than after.

### Q5. AMENDED 2026-09-04, not resolved.

Every c00 arm ran with `--sch-bias 0` while the head's dominant logit direction is unigram
frequency, which a bias captures exactly for `2V` FLOPs. Any decile structure measured under
that setting is confounded by withholding a free parameter from the code arms only. Re-run
with `--sch-bias 1` before reading anything into deciles.

The 95.18% energy share quoted earlier came from a 100-step checkpoint with two learned
directions and should not be cited. See the retraction in `LEARNINGS.md`.

### Q5 (original). What happens if the frequency-decile crossover does not fire anywhere?

Section 7 makes this a kill criterion for the tail-generalisation framing: drop it and pivot
to extension-only. But "did not fire" is ambiguous while the only vocabulary tested is 32k,
where section 8 predicts it *cannot* fire for reasons unrelated to the hypothesis.

**Decision rule to fix in advance.** The criterion should be evaluated at 131k or above, and
only after confirming the rarest decile actually contains scarce tokens (the diagnostics
report types and evaluation tokens per decile, so this is checkable rather than assumed).
Declaring the thesis dead on 32k evidence would be the wrong call.

### Q6. Do the head's `g` parameters belong in the Muon group?

Currently every head parameter goes to the AdamW unembedding group, matching how the dense
`lm_head` is treated, so the comparison is not confounded by optimizer treatment. But `g` is
a `d x M` matrix and the repo's convention is that 2D matrices get Muon.

**Why it is deferred.** Changing it would make the code head and the dense baseline differ in
two ways at once. Worth a controlled ablation once the main results exist, not before.

---

### Q7. Is a rank-120 learned head a real Pareto point, or an artifact of depth 4? (opened 2026-09-04)

The only line from the SCH work worth following. `ctrl_learned_w` reached 1.2281 bpb at
5.13e7 FLOPs against dense's 1.1593 at 7.79e7, so 66% of the compute for 0.069 bpb. The SVD
explains why: the dense head's logit matrix is 97.43% explained by 15 directions, so a
rank-120 head is barely truncating anything.

**Why it may not survive.** Depth 4 puts 64.6% of all FLOPs in the head. At depth 20 the
head is 9.0%, so the same rank reduction buys almost nothing while the bpb cost is unlikely
to shrink at the same rate.

**What settling it needs.** The dense scaling arm c00 did not run. Without it we cannot say
whether the point sits above or below the dense FLOPs-bpb curve.

**Prior art check first.** Factorized and low-rank softmax heads are old: adaptive softmax,
ALBERT's factorized embedding, the tied low-rank literature. This needs a novelty argument
before it needs more compute.

---

### Q8. Does a fused product-gather kernel exist that reaches the memory floor? (opened 2026-09-04)

**Blocking for the whole product-code direction.** The arithmetic is `V*g` instead of `V*M`, a
96x reduction at V=131072, d=768, g=8. The wall clock cannot follow it that far, and the current
implementation moves in the wrong direction entirely.

**The floor.** Both the dense head and the product head write the same `N x V` logit tensor,
17.2 GB per forward at 65536 tokens. The dense head is compute bound at 768 FLOP/byte, so
removing its arithmetic lands on that write. H100 roofline: dense 13.3 ms, fused product 5.1 ms.
**Achievable 2.6x, not 96x.**

**The current state.** `product_gather` runs one `index_select` per group, each materialising a
full `N x V` tensor. About `2g` passes over the output, 275 GB, roofline 82 ms, 6x slower than
dense. `flops_per_token` reports the 96x anyway, so any sweep run before this is fixed will show
a FLOP win alongside a step-time regression.

**MEASURED 2026-09-04, and it is worse than the roofline suggested.** A first c05 run at depth 4
came in at **3.36 s/step and 0.23 MFU**. On CPU at N=2048, V=32768, g=8, K=64 the gather is
**4.76x slower** than the dense `z @ Phi^T` that performs **64x more arithmetic**. The backward
is the dominant term and the roofline above does not model it: `index_select`'s backward is an
`index_add` scattering V values into K slots per position, which at V=32768 and K=64 is 512-way
atomic contention per slot.

**Resolved for now by separating the two questions.** `--sch-product-impl dense` (the default)
materialises the one-hot Phi and runs a normal GEMM at `4*V*M`, so the quality question is
answerable today and the product arms are cost-matched to monomial and random-binary arms at the
same M. `--sch-product-impl gather` keeps the `4*V*g` path, and `c05 --group kernel` runs it on
purpose so the gap is on the record. `flops_per_token` branches on the implementation, so the
FLOP column cannot report `V*g` while the GEMM runs.

**What to try, in order.** `torch.compile` on the loop, since the sweeps already pass `--compile`
and inductor fuses gather chains well. Then a Triton kernel accumulating the `g` lookups in
registers with a single write, and a sort-based segmented reduction for the backward to avoid
the atomics entirely. Benchmark every attempt against the dense path, not against the FLOP count.

**Why it decides the direction.** CLAUDE.md's bar is FLOPs-efficient without being much slower.
At 2.6x on a head that is 50.7% of FLOPs at V=131k depth 12, the end-to-end win is about 1.4x.
That is worth having. At 0.16x it is a regression. The gap between those is entirely this kernel.

---

### Q9. RESOLVED 2026-09-04: no. Close the direction.

c05 at depth 4: MIX k8 top2 1.7587, MIX k4 top1 1.7652, **MIX k8 shared_phi (the control) 1.7664**,
MIX k8 top1 1.7675. The shared-Phi control matches the per-component arms to within 0.001 bpb, so
per-component Phi contributed nothing measurable, and the arms cost 1.3x to 5.0x dense in wall
clock to contribute it. The union of subspaces is not where the head's problem was.

Original question follows.

### Q9 (original). Does a routed union of frozen subspaces beat one shared subspace?

The c05 `mixture` group. Per-component Phi with top-1 routing gives reach up to `K*M`
dimensions at the per-token cost of `M`, against c00's mixture which computed all K components
(cost `K*4VM`) and shared a single Phi between them.

**Two failure modes found while building it, both now guarded.** A zero-initialised router plus
hard top-1 sends every token to component 0 forever, because identical logits tie and topk
breaks ties by index; the components then get no gradient and DDP refuses to step. And without
a load-balance term top-1 concentrates on whichever component wins early. The router is now
symmetry-broken at init, `sch_mixture_aux` defaults to 0.01, and `wrap_model` sets
`find_unused_parameters` when routing is sparse.

**Memory, not compute, is the binding constraint, and it took three passes to fix.**
The first version held three full-width buffers per slot (a `-inf` fill, a broadcast add, and a
logaddexp result). Removing those was not enough: the head still returned log-probabilities, so
`F.cross_entropy` re-normalised an already normalised vector, which is the identity and costs a
second full `(N, V)` fp32 tensor saved for backward. A hard top-1 mixture has nothing to mix, so
it now emits raw logits and lets the caller normalise once; genuinely self-normalised heads use
`F.nll_loss` instead of `F.cross_entropy`; and the dispatch buffer matches the components' dtype
rather than promoting each slice to fp32 before the copy. Measured memory slope in batch size,
in units of one full-width fp32 buffer: dense 4.16, top-1 5.05, top-2 9.57.

**The residual 1.2x is a graph break.** Routing dispatches tokens with `nonzero`, which is data
dependent, so the head runs eager while the dense baseline is compiled and fused. The sweep gives
the routed arms a smaller device batch (`MIX_DBS`, default 32); gradient accumulation keeps the
total batch identical. A token-chunked `torch.utils.checkpoint` around the dispatch would remove
the need, at the cost of recomputing the head in backward.

**Original note.** One `(N, V)` fp32 tensor is 34 GB at 262144
tokens and V=32768, which is already what the dense baseline holds. A first implementation held
three per slot (a `-inf` fill, a broadcast add for the router weight, and a logaddexp result) and
died on `loss.backward()` with 106 GB resident on a 140 GB card. With k=1 none of the three is
needed: the renormalised weight over a single element is 0, there is one slot to combine, and the
components partition the token axis so the buffer needs no fill. k>1 is inherently k log-softmax
outputs plus the combination and gets a smaller device batch instead; gradient accumulation keeps
the total batch identical.

**The control that decides it** is `MIX_k8_shared_phi`: same K, same routing, one shared Phi.
If it matches `MIX_k8_top1`, the union bought nothing and only the log-sum-exp mixing mattered,
which c00 already had.

---

### Q10. Does the head gap grow or shrink with depth? (opened 2026-09-04)

**The single fact that decides whether any of c05 is a paper.** Every arm was measured at depth 4,
where the head is 64.6% of FLOPs at V=32768. A Monarch head removes 84.2% of the head, which is
0.455x total FLOPs and 0.762x wall clock there, but projects to 1.08x end to end at V=32k depth 20
and 1.74x at V=131k depth 12.

The projection assumes the **bpb cost stays at +0.096**. Nobody knows that it does. At depth 4 the
head carries an unusually large share of the modelling work, so a weaker head may cost less at
depth 20, or more. If the gap shrinks the method gets better with scale and there is a paper; if it
grows there is not, and no amount of head engineering fixes it.

**The budget rule does not apply to this class of change.** `base_train` sizes the horizon from
`transformer_matrices + lm_head`, chosen empirically in `dev/LOG.md` because the Kaplan-style count
held the ratio near 10.5 across 1e18 to 1e19 FLOPs where the all-parameter count drifted 3.0 to
4.0. That fit was made on models where head size is a function of `d`. Once the head is the thing
being varied, the rule pays a smaller budget for a better head. Unpinned, MON_M1024 draws 0.389x,
0.638x, 0.788x and 0.866x of dense's tokens at depths 4, 8, 12 and 16, and because that shortfall
*shrinks with depth* it would manufacture exactly the result this question is asking about.
Excluding the head is not the fix either: it yields 264,246,528 at depth 8, matching neither the
dense legs at 440,401,920 nor any other convention. Pin the dense budget. This is the same
argument `mst_iclr2027.tex` already makes for value embeddings, applied to the head.

**What settles it.** A depth ladder, dense and Monarch at 4/8/12/16, at one pinned token budget, at
V=131072 rather than V=32768. Eight runs. The deliverable is two curves and the depth at which they
cross, not another head variant at depth 4.

---

## MoL baseline (opened 2026-08-09)

### Q1. Gated DeltaNet in MoL's routed blocks

`nanochat/mol.py` implements MoL with softmax attention in every block. Their headline
configuration (1+3of15) uses Gated DeltaNet linear attention in the routed blocks, with softmax
only in the shared block.

**Why deferring is defensible.** Their section 5.3 reports a dense DeltaNet control matching dense
softmax within 0.01 PPL, so MoL's structural gain is not the attention swap, and they run a "MoL
all-softmax" control themselves (Table 7).

**Why it is still a risk.** Their Table 2 prices DeltaNet at 0.85 PPL *inside* MoL at
`d_thin=256`, and their section 3.3 argues the delta rule helps precisely when attention capacity
is constrained, which is the thin-block regime. So a reviewer can fairly say we did not run their
best configuration.

**Blocked on**: a chunked delta-rule kernel (they use FLA Triton) and GPU access to validate it.
`mol_routed_attn` exists so this is a new enum value rather than a refactor.

**Decide by**: whether the head-to-head is close. If MST wins by a wide margin on FLOPs-vs-bpb,
0.85 PPL of MoL-internal improvement does not change the conclusion and a stated limitation is
enough. If it is close, this has to be built.

### Q2. Is the shared block inside or outside the routing softmax?

Their S+KofN notation says 1+3of15 "selects 3 from 14 routed blocks", which we read as: the
router covers the routed blocks only, and shared blocks are always on with unit weight. Eq (2)
is then exact when S=0, which is their Table 1 configuration.

The alternative reading is that the router covers all N and shared blocks are force-included with
the softmax renormalised over the S+k active. The paper does not disambiguate. Our reading is
implemented; if a MoL result looks anomalous, this is the first thing to vary.

### Q3. Do we ever need to reproduce their WikiText-103 perplexity?

Currently we validate by parameter and FLOP accounting against their tables, which reproduces
Table 1 (85.3M), Table 5 active (0.61B) and all three projection-overhead figures. That is strong
evidence the architecture is right but says nothing about the training recipe.

A full reproduction needs WikiText-103, a custom 32K BPE, `T=255`, and multi-epoch training, none
of which this repo has. Deferred unless a reviewer demands it or a MoL arm underperforms in a way
that looks like an implementation fault rather than an architectural one.

---

## MST

### Q7. RESOLVED 2026-08-10: G3 stays; see LEARNINGS.md.

Measured at L=16 under k=1 routing: dropping G3 costs 0.0059 bpb at identical FLOPs, worth
-0.078x on the multiplier, and flips training FLOPs from 1.120x to 0.977x. The cheap
`mst_ve_map` variant recovers ~10% and its full-rank form is net negative; the flag stays in
the repo as a recorded negative result.

**What replaced it, Q8**: G3 makes MST 1.49x LARGER than the iso-quality dense model (413.2M
against 276.9M), where plain VE is 1.24x smaller. That is the same parameter-for-compute trade
MoL makes (1.59x at 1.3B), and it dissolves our positioning against them. Decide whether the
paper leads with the G3 point (1.196x FLOPs, 0.670x params) or reports both as a frontier. The
recommendation is both, since both runs exist.

### Q4. Which FLOPs axis is the paper's headline?

At L=8 the two axes disagree and the ordering inverts:

- training FLOPs: MON_shuffle 0.564x > control 0.555x > SP2_k1 0.551x
- FLOPs/token: SP2_k1 0.804x > MON_shuffle 0.768x > control 0.761x

Sparsity is the best arm on one axis and the worst on the other, because the token budget is
`10.5 x scaling params` and routing changes params by only 2.8%. At L=16 sparsity leads both
(1.194x / 1.120x), so L=8 is plausibly the pre-crossover regime.

This has to be settled before the write-up, and settled on principle rather than after seeing the
numbers, because picking the favourable axis post hoc is exactly what sinks an efficiency paper.

### Q5. Does the Pareto multiplier keep growing with scale?

0.804x at L=8, 1.194x at L=16, so it crosses 1.0 between them and grew 1.49x over one doubling.
Two points cannot separate a trend from noise. `--group d32` in
`scripts/p08_mst_parity_sweep.sh` is written and waiting on compute.

Independent support: the block-diagonal GEMM efficiency measurement (24% of dense throughput at
d=64, 31% at 128, 67% at 256, 103% at 512) says the wall-clock penalty is a small-model artifact
that vanishes by d=512, which is exactly L=32. Two mechanisms, same crossover region.

**Caveat that needs checking on the real hardware**: that measurement is from an RTX 3050 Ti,
whose ridge point is near 90 FLOP/byte. An H100 is near 295, which moves the crossover to
d ~ 600, i.e. nearer L=37 than L=32. The crossover moves the wrong way on faster accelerators.

### Q6. RESOLVED 2026-09-01: the `d_h=32` kernel penalty is real on the training hardware.

Section 7 cites a 2.13x penalty at `head_dim=32`, part of the motivation for G1. The laptop GPU
had said the opposite (1.19x *faster*), because consumer Ampere does not take the real
FlashAttention path for these shapes. The A1 harness settles it on the training GPU, at fixed
total attention FLOPs:

| head_dim | n_head | ms | TFLOP/s | vs 128 |
|---|---|---|---|---|
| 32 | 64 | 0.398 | 172.5 | 0.49x |
| 64 | 32 | 0.245 | 280.6 | 0.79x |
| 128 | 16 | 0.194 | 354.1 | 1.00x |

2.05x at `d_h=32`, close enough to the cited 2.13x to quote. Note the second row: G1's own
`head_dim=64` also pays 1.26x against dense's 128. That is a real cost, but attention is only
1.6% to 3.8% of MST's device time (LEARNINGS, occupancy section), so it is worth about 1% of the
step and does not motivate revisiting G1.

### Q9. RESOLVED 2026-09-01: the structure is above its crossover; the overhead is not.

Opened on an estimate of 0.219 for `throughput(d)/throughput(D)` at D=768, derived from
whole-model numbers against a threshold of `1/N = 0.250`. Isolating the GEMMs with
`scripts/p10_mfu_microbench.py` on the H100 gives **0.453**, and 0.243 / 0.309 / 0.579 at
D = 256 / 512 / 1024. Block-diagonal execution clears its crossover at every D >= 512. The
split-K weight-gradient hypothesis is refuted in the same run. See LEARNINGS.

The gap is non-GEMM overhead: MST spends 36.7 ms of a 47.69 ms step outside its GEMMs against
dense's 10.6 ms of 20.17 ms, a 3.46x ratio that matches the decode kernel-count ratio.

### Q10. RESOLVED 2026-09-01: yes, and it crosses at D=1536.

Measured on the A1-A5 harness: MST 76.09 ms/step against dense 77.21 at depth 24 (D=1536,
d=384), so **MST is 1.01x faster in training**, and 1.30x faster in prefill at T=32768. The
ladder is 2.97x slower at D=512, 1.37x at D=1024, parity at D=1536. Throughput ratio 0.431
against FLOPs ratio 0.425. See LEARNINGS.

### Q12. Does stream dispatch pay now that d=384? (opened 2026-09-01)

**First attempt was invalid and has been re-enabled.** The initial run showed 2x step time, half
the MFU, and OOM at a batch the masked path fits. Cause: `float()` and a data-dependent `if` in
`_ffn_dispatched` put a Dynamo graph break in every layer, taking the compiled model from 1 graph
to 11 at L=8. Fixed 2026-09-01; dispatch now compiles to 1 graph, 0 breaks. In isolation the
dispatched FFN is 2.0x faster than masked at 0.42x the peak memory. **Re-run before drawing any
conclusion.**


The headline arm still masks unchosen streams rather than skipping them
(`p08_mst_parity_sweep.sh:683` omits `--mst-stream-dispatch`), so the k=1-of-4 saving is an
accounting claim. Dispatch was unattractive at d=128 because gathering to `K = T*k/N` shrinks the
GEMM's M by 4x and those GEMMs were already starved; at d=384 they run at 0.712 of dense
throughput, so the arithmetic changes. The FFN is about 53% of the per-layer matmul work, so
k=1-of-4 dispatch skips roughly 40% of it.

Phase B is implemented and `test_phase_b_matches_masking_when_nothing_overflows` pins numerical
equivalence when nothing overflows. Two things to settle: the capacity factor, and whether the
router survives losing the compute-then-mask gradient (exploration noise carries the whole burden
on the dispatched path). Run it at depth 24 against the masked arm on both bpb and ms/step.

Fallback if it disappoints: static sparsity (a per-layer schedule of which streams run) removes
the router, the gather and the capacity logic entirely, and the skipped GEMMs do not appear in
the compiled graph. Costs input-adaptivity.

### Q13. Does MST win decode once the generate loop is compiled? (opened 2026-09-01)

Decode currently reads 2.4x slower than dense at every depth, but it is flat in context from 256
to 16384 tokens for BOTH arms, so it is pure host overhead: dense at 16 ms/token for a 24-layer
D=1536 model is about 1000x off the bandwidth floor. Under CUDA graphs or
`torch.compile(mode="reduce-overhead")` the metric becomes bytes of weights per token, where MST
reads 3.75 D^2 per layer against dense's 12 D^2, so **MST should be 2-3x faster**. A ~6x swing
from the current number, and the natural reviewer question about inference cost. Measure before
the paper.

### Q14. RESOLVED 2026-09-02: attention gating does not pay, at d8.

Measured with all four arms on the same 280.8M-token budget. Gating attention alone costs 1.86x
what it saves (multiplier 0.899 -> 0.812), reproducing MoL's attention-coverage problem.
`mst_stream_shared=1` repairs it to within 0.00007 of the dense exchange rate, confirming their
Shared+Routed fix transfers, but only to a wash. Both S=1 arms are Pareto-neutral. No d16 run
needed. See LEARNINGS, including why the apparent training-FLOPs win for `S=1 only` is a
coordinate artifact with a +0.001 residual.

### Q15. Should the paper carry the --target-active-params sensitivity? (opened 2026-09-02)

Measured at d8: the active-matrices budget gives 247.7M tokens and 1.0415 bpb against the
total-matrices budget's 280.8M and 1.0320, so **+0.0095 bpb and an 8.4% drop in the FLOPs/token
multiplier** (0.899x to 0.824x). Not a defect, and the choice is argued in
`p08_mst_parity_sweep.sh`, but a reviewer who notices that sparse arms draw a total-parameter
budget will ask. Open question is placement: a sensitivity paragraph in the limitations section,
or a footnote on the cost table. One d16 or d24 confirmation would make the number a trend
rather than a single point.

### Q11. How much of the 3.46x non-GEMM overhead do the Stage 19 cuts recover? (opened 2026-09-01)

Three cuts landed in `mst.py` on 2026-09-01, forward bit-identical on 16 flag combinations
(backward within one bf16 ULP when the QKV fusion is on). See LEARNINGS for the exactness
argument and `tests/test_mst_parity_fixes.py` O3/O4/O5 for the regressions.

- `_rope_streams`: RoPE and QK-norm hoisted out of the per-stream attention loop.
- Fused QKV: one bmm with a 3x wider output instead of three, weights left as three Parameters
  so nothing downstream changes.
- Routing diagnostics gated on `MST._diag_enabled`, so the reductions stop being graph outputs
  on non-log steps.

**The measurement that closes this: re-run the A1-A5 harness at depth 4, 8 and 12 and compare
against 15.79 / 34.56 / 47.69 ms per step (MST) with dense unchanged at 6.88 / 12.71 / 20.17.**
The prediction is a partial recovery, not parity: the RoPE hoist is the large item, the QKV
fusion is worth about 1% to 2%, and the diagnostic gating only matters under compile.

Two candidates were sized and rejected rather than built, which is the useful part of the record:

- Replacing `torch.stack` with a preallocated buffer is a **no-op**. `torch.stack` already
  allocates once and copies each element in; the copy is inherent to concatenating N
  separately-produced tensors and `flash_attn_func` has no `out=` to write into a shared view.
- Stream-major `(N, B, T, d)` layout measures 1.42x / 1.24x / 1.32x at d = 64 / 128 / 192 but
  **1.00x at d=256 and d=384, and 0.92x at d=512**. It helps only below the scales the paper
  reports and hurts at the top.

Still open and free: one flash call per distinct window (worth 1 call in 4 on short-window
layers only, because multi-scale gives long-window layers four distinct windows), and CUDA
graphs for decode, where both arms are host-bound at 3.7 ms/token for a four-layer model.

Blocked on one measurement to attribute the remainder: re-run the A2 profile with
`## Call CompiledFxGraph` parent scopes filtered out and the top 25 *leaf* kernels shown with
shapes. Those two scopes are 57% to 72% of device time and the listed leaves sum to well under
30%, so the 36.7 ms is still attributed by inference rather than by observation.

## Q12: does a clustered vocabulary partition raise Monarch's per-block capacity?

`MonarchHead` assigns word `w` to block `w // block_out`, so blocks own contiguous ranges of
token id. BPE ids are roughly merge-order, which stratifies blocks by frequency but not by
meaning. A block only has to separate the words inside its own shard, so a semantically
coherent shard should be separable in fewer than m1 dimensions.

The change is a fixed output permutation: zero FLOPs, no new parameters, one buffer. Cluster
by input-embedding similarity from a trained dense run, or by frequency, and compare against
the current identity assignment at matched m1. This is the cheapest untested lever in the
direction and it attacks the exact quantity, per-block capacity, that the depth-12 ceiling is
made of.

## Q13: does a shared low-rank residual beat spending the same FLOPs on larger m1?

Monarch gives every word m1 block-private directions and no shared ones. Adding
`logits += C (A h)` with A of shape (r, d) and C of shape (V, r) costs r(d + V) MACs and gives
every word r globally shared directions. At r=32 that is 4.2M MACs on top of 17.6M, moving the
FLOPs ratio from 0.582x to 0.603x and the break-even from +0.0310 to +0.0289, so the residual
has to buy 0.0102 bpb to pay for itself. Growing m1 from 128 to 160 for comparable cost is
projected to buy only 0.0063. The question is whether shared directions, which can carry
unigram frequency and syntactic class, are worth more per dimension than block-private ones.

## Q12 (resolved, and inverted): block assignment is about mass, not meaning

A frequency-sorted partition is indistinguishable from the default; a random one wins
at both screened depths. See LEARNINGS.md. The clustered arm was never run and is now
low priority: clustering by embedding similarity concentrates mass the same way `freq`
does, which is the thing that hurts.

Still open and cheap: whether `random` and the shared residual add or overlap. The
c10 combination arm used `freq`, which turned out to be the null arm, so it only
showed that the residual dominates a no-op.

## Q14: non-uniform m1 per block

Q12 says uniform capacity across blocks is the constraint. The response is to keep the
frequency stratification and reallocate capacity to match it: block j gets its own m1_j,
large for the head of the distribution and small for the tail, with sum_j m1_j held
fixed so FLOPs do not move. That is adaptive softmax's actual idea, which the current
head has half of.

It costs more than a permutation: the second factor stops being one stacked
`(m2, block_out, m1)` tensor and becomes m2 separate matrices, so the single batched
GEMM becomes a grouped one with ragged K. Worth it only if the Q13 residual saturates,
because both moves buy the same thing, shared capacity where the mass is, and the
residual buys it without touching the kernel.

---

## EET: is the remaining 0.06 bpb architectural? (open, P02 decides)

`eet_experiment_log.md` concludes the gap is architectural after thirteen failed
mitigations. That conclusion is not yet supported: every one of the thirteen targeted
gradient flow, learning rates, representation alignment, distillation or scheduling, and
none targeted the two mechanisms that actually differ from a dense model.

- **Defect 1, context destruction.** Exited tokens lose their keys and values in every
  later layer, so at d8 the survivors attend over 10-25% of the sequence. Restoring
  full-context reads costs about 1.6% of a dense layer.
- **Defect 2, data starvation.** Exit depth is a per-vocabulary-item lookup table
  (`use_pos_embed` is off, so the router's input is `norm(wte(idx))`), so deep layers train
  on a fixed ~10% slice of the vocabulary. This is the likeliest explanation of the
  measured "78% of the gap is backbone co-training" and of why depth-LR-scale hurt and
  depth-grad-scale did nothing.

`scripts/eet_p02_tests.sh` resolves this with pre-registered gates. **Do not brainstorm
new EET mechanisms until it has run**: if both defects come back negative, the
architectural verdict is confirmed and the direction closes; if either is positive, the
mechanism that closed the gap determines what the paper is about.

Two further questions stay open regardless of the outcome:

1. **What is the right dense control?** No iso-FLOP iso-data dense run at d8's reduced
   active-FLOP budget exists. Every EET Pareto statement so far compares against dense at
   full budget, which is not the comparison a reviewer will accept. P02 adds d5 and d6
   controls; the exact matched depth still has to be pinned once their FLOP counts land.
2. **Is "matches dense, 25% faster" publishable at all?** Almost certainly not on its own
   in 2026: Mixture-of-Recursions (NeurIPS 2025) and N-vium (57.9% wallclock at 1.5B with
   no perplexity cost) already occupy that claim. If P02 succeeds, the framing question is
   whether **write-depth versus read-depth as two separately routed budgets** is a strong
   enough primitive to carry a main-track paper. That is unresolved and should be settled
   before any scaling runs are booked.

---

## SAP S05: can a one-round continuous field learn discrete block correlation? (resolved: no)

S04 is closed: an exact categorical positional-field mixture used its latent but missed the toy
gate badly (KL 1.7078, invalid 33.94%). S05 tested a different mechanism: one vectorised Gaussian
field draw, a fixed-depth decoder, and parallel final token draws, trained by MC marginal
likelihood plus a proper score on hard realised token blocks. It failed the phrase-HMM gate at KL
3.3549 and 53.86% invalid despite nonzero field sensitivity. No seed 1, hyperparameter sweep, or
depth-8 run is authorised. A train-time AR verifier was considered and explicitly rejected by the
user because it creates a second training role. See `s05_sap_plan.md`.

## SAP S06: can random-map scan turn exact sequential dependence into one-round generation? (screen resolved)

The verifier question above is superseded by the user's no-distillation/no-second-model boundary.
Permutation-State Scan was tested with a depth-4 oracle-context backbone, 64 states, exact joint
likelihood, and fixed XOR vocabulary permutations. It reduced invalidity from roughly 63% to 54.39%
but finished at KL 5.015, far outside the 0.10 / 3% gate. All nine companion mechanisms also failed;
MIF was best in exact likelihood at KL 4.250 / 60.01% invalid. No depth-8 run is warranted. See
`s06_sap_one_pass_brainstorm.md` and `s06_sap_compiled.log`.

The semantic oracle resolved representational capacity. One shared spectrum fails even with true
states and arbitrary per-state bijections (KL 1.1542, invalid 34.35%), so token relabelling alone is
not sufficient. Two state-routed spectrum classes pass decisively (KL 0.01468 ± 0.00129, 0%
invalid), while nine classes reproduce the HMM exactly.

The remaining open question is learnability of **Spectrum-Class Semantic PSS** at `S<=64,C=2`:

1. Can the model learn state transitions and the binary emission-spectrum route from context?
2. Can semantic bijections be learned or constructed without true HMM emissions, a teacher, or a
   second model?
3. Can exact training score both spectra in `O(2LV + LS^2)` while inference routes only one `LV`
   readout after the deterministic state scan?

The learned toy failed at KL 2.9485 / invalid 48.68% after 8,000 updates, despite improving over the
independent control. Its arbitrary permutation EM remained unstable (95.64% entry change at the last
update) and state occupancy remained concentrated. Therefore no speed kernel or depth-8 run was
performed.

The open question is now narrower: can a *shared, identifiable* semantic coordinate system replace
64 independently assigned permutations? A candidate must constrain state actions to a small closed
family (for example affine or Feistel actions on one shared code), specify a stable learning rule,
and preserve exact inversion/normalisation. Merely increasing the EM sample count, balance loss,
steps, or state count is a sweep of the failed mechanism and does not reopen d8.

## SAP S07: which learnable joint mechanism merits a fresh gate? (open 2026-10-03)

The S06 gate failure stands. Its claim of proven S=64 sufficiency/optimization-only
failure is superseded: the semantic construction used 452 true states and was not a
certified marginal-KL optimum. Permutation churn was unweighted; it does not by
itself diagnose harmful instability. See the newest LEARNINGS entry.

Two proposed mechanism questions, not scheduled runs:

1. Can a continuous categorical flow with invertible affine scans and nonlinear
   couplings learn coherent token-cell probabilities at fixed neural depth, and
   beat a genuine continuous Argmax coupling control at matched training cost?
   The S06 discrete ST shift is not that control. A small training-only stochastic
   inverse is part of this proposal; it is not an AR teacher, but its acceptability
   and cost must be made explicit before implementation (related to Q24).
2. Can a corpus-defined observable suffix/phrase state retain enough dependence
   for a sparse random-map transducer? It avoids hidden-state/permutation EM and
   supplies exact conditional likelihood, but sparse backoff and finite memory
   may recreate the old quality gap. A constrained offline fit comes first.

Both retain T=L=2048 as the primary target, no verifier/distillation/refinement,
and the numerical gate in S07. The user also admits smaller blocks such as T=8
as a separate route if BPB-neutral and sufficiently fast/novel; S07 records a
distinct secondary evaluation, not a redefinition of full-block success.
Existing dense checkpoints are references; no retraining is implied.
The main-track novelty of either mechanism remains unestablished. No Modal jobs
were launched during this brainstorming/documentation pass.

## SAP S08: does a lockstep lane order cost <= 1% bpb at 2x or more? (open 2026-10-03)

The strict-seed brainstorm from a separate session (`s08_sap_lanes_plan.md`) kept one family. Its lead is fixed lockstep lanes: L contiguous lanes of one document, one token per lane per trunk pass, with an exact likelihood.

Pre-registered at d4, full-context attention for every arm, two seeds:
- **Kill:** lane tax above 1.0% at L=2 (S=1024), or a tax that does not fall with lane length.
- **Bar candidate:** tax at L=4 of 1% or less.
- **Tax at L=4 between 1% and 2%:** run paragraph-aligned lanes before any d8 spend.

Then d8, against the measured d8 frontier (4 layers: 1.79x at +2.6%).

Interaction with S07, which targets T=L=2048: this session's user instruction was "strict SAP seed only", meaning T tokens per pass. Lanes read T as distant tokens. Which reading governs is the user's decision.

**d4 result (2026-10-03):** passed. Tax +0.47% / +0.94% / +1.57% at L = 2 / 4 / 8, at 1.67x / 3.24x / 6.23x on batch 16 with CUDA graphs. Open: does the tax hold at d8 and beyond? Do lanes work with sliding-window layers? How does generation quality (reference PPL) compare with AR samples?

**d8 (2026-10-03):** tax +0.61% at L=2 (1.90x / 1.48x on batch 1 / 16) and +1.10% at L=4 (3.74x / 2.91x). The seams dominate the generation-quality cost. **Parked:** the user confirmed T=L is required, so lanes are the documented T=K fallback, not the lead.

## SAP S09: do S07's T=L hypotheses survive their cheap gates? (open 2026-10-03)

This is the session that built lanes, now on T=L at the user's direction (`s09_sap_tl_gates.md`).
- **B (observable suffix transducer) is closed.** Its information oracle (prompt + last-k attention, d4) costs 7.4 to 8.9% at k=32 for positions >= 512, against a pre-registered kill line of 1%.
- **A (scan-coupled categorical flow) is closed.** It is numerically certified, but on the toy gate it reaches only KL <= 3.3 to 3.6 with 52 to 60% invalid blocks, splines included. The scan arm is no better than its no-scan control.
- **Open, for the user:** is there any T=L mechanism with a long-range memory of its own output that is learnable in one fixed computation?
- Every family tried (S01 to S09) says no. The T=K fallback (lanes) has the only near-neutral result.

## SAP Flow–Joint: does a continuous plan plus exact local joint fix the toy? (tested 2026-10-03)

The first registered instantiation fails: two seeds at equal counted training
matrix FLOPs give hybrid KL bound 1.4004 / 1.4415, invalidity 35.25% / 35.75%,
against <=0.10 / <=3%. Cost-matched flow-only is numerically better, so the
intended hybrid advantage is not established. See `sap_flow_joint_results.md`.
The initial no-promotion decision is superseded on 2026-10-04 by the user's
explicit full-budget d4 flow-only T=L=2048 request (`sap_flow_text_d4_plan.md`).
That is a scale/learning test of the best control, not a reversal of these results
or authorization for a broad sweep. The user subsequently authorized one full-budget
d8 flow-only capacity test on 2026-10-04; see `sap_flow_text_d8_plan.md`.

**Full d4 follow-up completed:** T=L=2048, 77.53M target tokens, full baseline-sized
budget. Flow joint BPB-bound estimate 2.34450 versus dense 1.14462, but batch-16
CUDA-graph speedup 66.33x. Latent KL falls to 0.00086 nats/token, consistent with
posterior collapse; samples are incoherent. See `sap_flow_text_d4_results.md`.
Open: what training mechanism makes the sampled plan carry useful future dependence
without a teacher or token-dependent neural loop? This run does not settle whether
objective/optimizer, posterior family, or capacity is chiefly responsible.

**User-authorized d8 capacity follow-up completed (2026-10-04):** same T=L and
objective, width 512, eight generative/four recognition blocks, 269.6M target tokens,
full existing d8 dense compute budget. K=32 BPB-bound estimate 2.31868 versus dense
0.95179; batch-16 graph speedup 89.25x. Exact d4/d8 held-out token equality verified.
Flow's bound improves 1.10% over d4, but samples remain incoherent and latent KL
remains small (~0.00493 nats/token), despite being ~5.73x d4. ESS 2.17/32 limits
interpretation of the bound gap. See `sap_flow_text_d8_results.md`. Capacity scaling
alone did not rescue this instantiation at the registered budget; the unresolved
training-mechanism question remains. No further scaling sweep is authorized here.

**Budget/frontier question raised by the user (2026-10-04):** iso-FLOP should not
be the only lens for an inference-efficient SAP. How much extra training buys
quality neutrality at retained decode speed, and at what inference demand does
that training cost amortize? The completed flow saw fewer targets than dense
(269.6M vs 440.4M), despite more optimizer updates. Candidate 1x/3x/5x dense-token
milestones and a quality/speed/training-cost frontier need an explicit protocol,
including a defensible continuation LR schedule and learning-curve diagnostics.
The user clarified that the comparison should match dense's effective batch
(262,144 target tokens/update at d8), and "longer" refers to total target tokens.
Use gradient accumulation if the physical microbatch must differ. Whether to start
a clean matched-batch run or explicitly label a batch-changing continuation was
initially open. The user has now authorized all three budgets, 1,680/5,040/8,400
updates. The user subsequently authorized execution optimization followed by
launch. The H100 audit selected compiled microbatch 32, four accumulation steps
and fused AdamW: 0.62438 seconds/update versus 1.54598 for the original eager path,
2.48x faster at the same effective batch. Full-model fixed-tape BF16 loss and
gradient checks passed; 38 focused tests pass. See
`sap_flow_text_training_optimization.md`. All three completed on `nanochat2` as
`flow_d8_mb262k_s1_20261004`, sustaining about 398–404K targets/s in logged updates.
The 1x/3x/5x-token K=32 bounds are 2.31514 / 2.27597 / 2.25436 versus dense 0.95179,
with ~89x B16 graph decode speedups but incoherent inspected samples. These tested
budgets did not establish quality neutrality; low importance ESS limits exact-gap
interpretation. The sweep plan records completed results, timing scopes and IDs.
Open: distinguish remaining objective/recognition/decoder limitations rather than
assuming another duration multiplier will resolve them. No further run is implied.
Do not silently retrofit the old run's criterion, transfer AR scaling coefficients
to flow, or launch extra training from this discussion alone.

Still unresolved: which mechanism can learn symbolic compatibility without
compressing all dependence into a small state or relying entirely on continuous
geometry? Posterior-conditioned invalidity is still 28.6%, so improving only the
prior is not a sufficient diagnosis of the observed failure. Distinguishing
optimization from recognition/decoder capacity would require a separately
justified test, not an assertion that the entire hybrid family is impossible.

## SAP S10: can one call carry T=L through reference-corrected, self-inverting PTP? (open 2026-10-03)

The user made PTP-style inversion admissible and asked for its inversion problem to be fixed with pieces of earlier failed concepts while keeping generation to one pass (`s10_sap_tl_brainstorm.md`). The survivor is RC-PTP (`nanochat/ptp.py`):
- one pass over binary-digit embeddings of uniform auxiliaries;
- a draft pick at the cut, with a reference module above it reading the drafted tokens;
- the same u for the draft and the final pick;
- a semantic CDF order;
- training fully parallel: a token-conditioned AR mode inverts data in one teacher-forced pass, and the generator distils its conditionals.

Pre-registered toy gates (L4, two seeds, 8k steps):
- **Pass:** KL upper bound <= 0.10 and invalid <= 3% at T=4.
- **Mechanism claims:** each piece must beat its ablation by >= 10% KL at T=16 or 64.
- **T=L viability:** KL <= 0.5 at T=64 (depth 4), or a clear gain at depth 8.

Open: the tokens-per-call against depth curve (T9). It decides whether one call can cover 2048 tokens at all.

**Phase 1 result (2026-10-03).** The T=4 gate fails (best KL 0.75, 28% invalid), and the T=L viability gate fails by about 80x (T=64 KL 42 at depth 4, 36 at depth 8).
- Two new mechanisms each pass the >= 10% contribution test at T=16 and 64:
  - a tree pick (one uniform per vocabulary-tree level), which fixes the flat inverse-CDF pick's fragility;
  - self-inversion by parallel sweeps.
- Together they raise correct tokens per call from about 1 to about 2.5, which does not change the scaling.
- Open, for the user: keep T=L as a hard requirement, or move to T=K, where lanes at d8 (+1.10% at 2.9 to 3.7x) already sit above the dense depth frontier (4 layers: 1.79x at +2.6%)?


## SAP S11: can learned separator codes, sampled interface-first, carry T=L? (open 2026-10-04)

The user kept T=L as a hard requirement and asked for a brainstorm from the obstacle outward (`s11_sap_tl_brainstorm.md`). Two exact oracles on the toy say:
- left-to-right order, not the process, costs about T stages: interface-first sampling is exact in log2(T) + 2 levels;
- tokens as decision variables cost 44% invalid at T=64, against 0% for states.

The lead survivor is a Bridge LM:
- learned causal state codes, product-quantised;
- a bisection-order prior with within-level independence;
- parallel token emission;
- teacher-forced training with a likelihood lower bound.

Open, and decided by the pre-registered toy gates:
- Can the codes be learned? (KL <= 0.10 at T=4 and <= 0.5 at T=64; kill above 5 at T=64.)
- On real text, how much does routing cross-boundary information through codes cost? (Kill above 3% at d4.)

**Stage 1 separator oracle (2026-10-04, d4, done).**
- Routing the second half through 1 to 64 slots costs +1.3% on the second half over all rows, and +2.5% when the split falls inside one document. That is under the 3% kill line, but there is no go.
- The cost sits in the first 64 tokens (local continuity) and in copied spans, and it does not depend on slot count.
- Consequence: Bridge LM codes must be per position, visible from the first layer, and carry token identity.

Stage 2 (toy Bridge LM, two seeds, T=64) is running.

**Stage 2 and 3a (2026-10-04).**
- The toy Bridge LM with oracle separator codes reaches 6.9% invalid at T=64 with 24k steps (bound 5.0), against RC-PTP's 100%.
- Learned clustered codes fail because they are impure.
- Window bisection (decide token windows) is exact and needs no learned codes.
- On real text at d4 it costs +7.8% at best (n=16, 126 parallel steps for 1920 tokens). The pre-registered kill (> 5% at every n) is met.
- Open: is there an order between left to right and bisection that keeps text's conditionals learnable at O(log T) or O(sqrt T) depth?

**Stage 3b (2026-10-04, superseded the same day).** It reported bridged lanes at 4x beating dense-1x (-1.8% and -1.5%) with parity at 2.8x tokens. The (64, 4) rows and the 1x rows were scored in another model's order (eval bug, fixed; see `s11_sap_tl_brainstorm.md`, "Correction").

**State after the fix and the ladders (2026-10-04, archaeonseq, d4).**
- **Bar confirmed by the user:** within 1% of dense-1x bpb at 4x tokens or fewer, with at least 10x fewer steps.
- **Met on bpb by:**

| Configuration | Block bpb vs dense-1x | Steps for 1920 tokens | Measured speedup, batch 1 |
|---|---|---|---|
| Plain lanes L=64, 4x | 0.993 and 0.992 (two seeds) | 30 | 53x |
| Plain lanes L=32, 2x | 0.994 | 60 | 28x |
| Bridged lanes (32, 8), 4x | 0.983 and 0.983 (two seeds) | 100 | 17x |

- **At equal tokens the tax is constant, not shrinking:** about 6.5% at L=64 and 4% at L=32. Parity at 2 to 4x comes from dense's own gain per doubling at d4.
- **Seeded middle-out lanes are killed:** the tax doubles.

**Open: sample quality.** Samples from every parallel order have 2.1 to 2.6x the reference ppl of dense-1x next-token samples (held-out dense-4x as the scorer), with distinct 3-grams of 0.97 against 0.81:
- plain lanes L=64, 4x: 151 against 61;
- bridged lanes (32, 8), 4x: 120 against 57.

Next-token samples from d4 repeat themselves, which lowers their reference ppl, so a real-text anchor (`sap_eval_generation.py --real`) is running.

**Decisions for the user:**
- add a sample-quality criterion to the bar (bpb parity does not imply sample parity for parallel orders);
- whether the paper's comparison class is masked and block diffusion at equal steps (their papers accept 2 to 3x generative-ppl gaps against next-token models), which would need an MDLM baseline in this codebase;
- whether to test at d8 whether the tax and the sample gap shrink with model size.

## SAP S13: back to the seed, constant rounds or one pass? (open 2026-10-04)

The user asked to go back to the one-head T=L seed (`s13_sap_brainstorm.md`).

**Findings from logged runs.**
- **Junction loss.** At equal tokens, a lane's end recovers only 13 to 35% of the next lane's start deficit. Recovery grows with model size (d4 20% to d8 35% at L=64).
- **Lanes are already T=L by the accepted definition.** With a fixed lane length S, they decode any L in S rounds. The open problem is the tax at small S.
- **d8 ladder closed.** 64 lanes at 5x tokens give about 1.010 of dense-1x, on the line.

**Measured (Q3).** One 1920-row pass costs 1.8 to 2.2 rounds at batch 1 and 5.8 to 6.5 rounds at batch 16 and above, in this codebase's CUDA-graph decoder at d8 and d20. Below about 8 rounds, the gain is real only at small batch.

**Open, and decided by pre-registered quick tests:**
- **SV-A splice codes: killed (2026-10-04).** Block 1.1064 against plain lanes' 1.107: about 0% of the tax removed. A blind code costs 5.03 nats and saves about 4.4. Blind junction variables are information-neutral.
- **SV-B skeleton:** not blind, since each class is drawn given the earlier classes, so the SV-A argument does not kill it. The pre-registered gate on SV-A was wrong; the user decides whether to run its two-run oracle.
- **SV-D: necessary condition passed (2026-10-04).** A K=8 chunk autoencoder (256-dim latent) reconstructs 100.000% clean and 99.974% at noise sigma 0.5. Next is the one-step MeanFlow prior, which needs a sample scorer (a d8 dense on eqyve967, or scoring on seqaeon).
- Is the unrecovered junction loss TC or learnability? Answered by the optional toy oracle Q4; it gates SV-E.

**Decision for the user.** One pass (SV-D, true T=L, samples-only bar) or constant rounds (SV-A, exact bpb, S+1 rounds)? The Q3 timing above is the evidence for that choice.


## SAP S14: is the strict seed limited by information or by computation? (RESOLVED 2026-10-05: information)

**Resolved.** Under both 8B oracles, exact token orders of 13 levels or fewer lose 13 to 15% to same-step TC, against a 1% bar.
- Lanes need about 36 or more steps for TC ≤ 1%.
- Separators need 113 to 126 bits.
- PCB and BSB are not live; LSB and CVL are dead; VAR-T is disfavoured.
- Details in `s14_sap_strict_tl_brainstorm.md` §11; the learning is in `LEARNINGS.md`. The original entry is kept below as the record.


The user asked for the strict version of the seed (`s14_sap_strict_tl_brainstorm.md`):
- 1920 tokens in at most 13 parallel levels;
- exact likelihood or an honest bound;
- from scratch, with no teacher;
- bar: within 1.01x of dense-1x at ≤ 4x tokens, with ≥ 20x decode at batch 1.

**Funnel.** 53 candidates → 5 mechanisms plus 3 ingredients:
- PCB, pyramid-capacity bisection;
- BSB, boundary-snapped bisection;
- LSB, learned-separator Bridge LM;
- CVL, per-level coordination variable;
- VAR-T, temporal multi-scale codes.

**Decisive unknown.** Every measured strict-order tax (window bisection 7.8 to 24%, lanes 3.9 to 10.7%) mixes same-step TC (information) with the learning/computation gap. Nothing has separated them on real text.

**Stage 0, ready to run on Modal.** About 10 to 13 H100-hours.
- E0 + E2a: `modal run modal_sap.py::s14_order_oracle` (LLaDA-8B-Base, 32 rows), `--oracle dream` (Dream-v0-Base-7B, 16 rows), then `s14_order_oracle_compare`.
  - Measures TC and gap per order with bootstrap CIs.
  - Measures the bits any single-position separator must carry (an information bound covering every code of B bits).
- E1: `modal run modal_sap.py::s11_ladder --depth 8 --specs dense:1:1,wb:1:1:1,wb:16:1:1 --name s14_e1`.
  - Measures whether the window-bisection tax shrinks from d4 to d8.
- E2b (training with codes) runs only if E2a leaves a ≤ 48-bit separator possible.

**Pre-registered readings** (in the brainstorm, §6):

| reading | condition |
|---|---|
| Strict token orders closed on information | bisect1 TC ≥ 3% |
| Exact strict orders closed on computation | bisect1 gap ≥ 5% in both oracles |
| PCB is live | some ≤ 13-step order with TC ≤ 1% and gap ≤ 2% |
| BSB is live | snapping cuts TC + gap by ≥ 15% |
| LSB is dead | a single-position separator needs > 48 bits |
| PCB is supported | tax(d8)/tax(d4) ≤ 0.7 |

**Decision for the user.** None until Stage 0 returns. Then pick which Stage 1 gates to run.

## SAP S15: does the plain-lanes tax turn down with scale? (open 2026-10-05)

**Stage L0 (2026-10-06).** Details in `s15_lanes_paper_plan.md` §7.
- R1 in between: extra nats per lane 7.41 / 7.56 / 7.14 at d4 / d8 / d12 (d12 / d8 = 0.944), so the d16 tiebreak is pre-registered.
- Equal-token tax 6.4 → 7.8 → 8.3%; parity about 4.5x.
- Samples 1.55x worse at temperature 1; at real-text entropy, pending.
- Speed 54x / 31x / 16x at batch 1 / 16 / 64.
- Lanes beat random-order decoding 3.8x at equal steps under the oracle.
- **New target for L1: recovery.** The lane-start deficit is at the 8B oracle's information level; recovery is 5.0 against 9.7 nats per lane.
- Open: d16 (R1), R3 at real-text entropy, and the L1 brainstorm (`s16_lanes_recovery_brainstorm.md`).

S14 closed the strict thesis. The user chose the plain-lanes paper (`s15_lanes_paper_plan.md`).

**Frank status: not at the A* bar yet.**
- The evidence is only at d4 and d8, and the equal-token tax grew from d4 to d8.
- Parity needs 2 to 5x tokens, and that multiple can grow with scale.
- The speedup is batch-1 only, and there is no iso-quality baseline.
- Samples and the OWT / BD3-LM protocol are still open.

**Decides now (R1).** Extra nats per lane at L=64 and 1x tokens, d12 against d8.
- ≤ 0.90x: go.
- ≥ 1.05x: no-go for plain lanes as the core.
- In between: a d16 pair decides.

**In parallel (L0b).** The S14 oracle at T=1920: lanes' floor per L, and lanes against confidence-ordered diffusion decoding at equal steps (win if lanes' total ≤ 0.5x).

**Decision for the user, after R1.** Which mechanism for the learnable lane-start cost (L1). At d8 about 85% of the per-lane loss is learnable.

