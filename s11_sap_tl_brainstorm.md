# S11: T=L brainstorm from the obstacle out

> 2026-10-04.
>
> User decisions:
> - T=L stays a hard requirement: one fixed computation emits the whole block.
> - Every candidate must state up front how it beats the measured limits before anything is built.
> - Reason from what is actually stopping us, using heuristics from neuroscience, psychology, philosophy, mathematics, statistics, biology, chemistry, physics and fiction, not only from ML papers.
> - (2026-10-04, revision) A fixed number of internal sampling levels counts as T=L (13 at T=2048), with all positions within a level in parallel.
> - (revision) The user's diagnosis: sampling during pretraining fights the hard-prediction training signal. The analysis is reworked around it (root cause 0), the brainstorm is broadened beyond hierarchy, and the real-text separator oracle runs next.

## 1. What is stopping us

S01 to S10 tried fields, noise tapes, scans, flows, energy and kernel scores, IMLE, MeanFlow, suffix transducers, latent plans and PTP-style generators. All failed the phrase-HMM toy gate. The S10 T curve made the failure quantitative:
- The best learned one-pass generator gets about 2.5 leading tokens right per call. This is limit **C1**.
- Coupled refinement with exact conditionals fixes about one position per stage. This is limit **C2**.

Two new exact oracles (`scripts/sap_bridge_oracle.py`, on the true toy HMM, no learning) show that neither limit belongs to the process. Both belong to how the decisions were organised.

**Root cause 0 (unifying, the user's diagnosis): what the training signal is made of.** Sort every S01 to S10 mechanism by whether its training loop needed the model's own samples.

| Training signal | Mechanisms | Toy KL at T=4 |
|---|---|---|
| Teacher forcing on hard targets derived from data, exact factorisation | AR mode (S10, 8k steps); local head B4; correlated tree (sequential rounds); cut head with the fill conditioned on true anchors | 0.17 to 0.27; 0.027; 0.11; 0.40 to 0.53 |
| The same, but with a structurally crippled model | PSS (one emission distribution permuted by 64 states); S04 flat positional mixture | 2.95; 1.71 |
| The model's own samples, noise or posterior draws inside training | S05 (Monte Carlo marginal likelihood over prior draws); energy, kernel, IMLE, MeanFlow; flows (dequantisation noise, ELBO); P1 and P2 (ELBO posterior draws); RC-PTP (inverted noise, on-policy drafts); S01 independent anchors | 3.35; 1.7 to 5.0; 3.3; 0.40 and 0.57; 0.75 to 1.66; 3.3 |

- Every mechanism that learned well was trained like an AR model: cross-entropy on targets computable from the data, under an exact factorisation.
- Every mechanism that needed sampling inside training learned poorly. The best of them, P1, is still 4x off the gate.
- The two failures without sampling had crippled structure.

The mechanism is not mysterious:
- inputs that depend on the model (inverted noise, drafts, posterior draws) are non-stationary;
- they add variance;
- discrete picks block gradients.

So a one-pass generator must not only have the right structure (root causes 1 and 2 below). It must be trainable by teacher forcing: every decision variable a deterministic function of the data, every conditional fit by cross-entropy on hard data-derived targets, and no model sampling in the training loop. Sampling happens only at inference, exactly as it does for an AR model.

**How S11 handles it.** The decision variables are codes computed deterministically from the text:
- quantised states of a causal encoder;
- in the cleanest version, quantised states of a separately trained recurrent LM, whose states are Markov by construction.

The bisection prior and the token emission are then fit by teacher-forced cross-entropy on those codes and tokens. log p(codes(t)) + log p(t | codes(t)) <= log p(t), so the reported bpb is an honest upper bound.

The remaining non-stationarity is the code encoder. It is handled in one of two ways:
- two stages: train the encoder, freeze it, then fit the prior on fixed codes, as VQ-VAE-2 does;
- joint training with EMA codebooks and stop-gradients on the prior's inputs.

Exposure bias at inference touches only log2(T) + 2 levels, not T.

**Root cause 1: the order of decisions.** Every mechanism so far decided tokens left to right, or all at once in a flat way.
- In left-to-right order the dependency chain has length T. A wrong early choice changes the hidden state, and with it every later conditional (S10 Jacobi oracle).
- The same HMM sampled interface-first is exact in log2(T) + 2 parallel levels. Interface-first means nested dissection: draw the hidden state at the block end, then the state at every interval midpoint given the interval's two end states, all midpoints of a level at once, then every token from its own state.

| T | Left-to-right stages | Interface-first levels | Invalid | -log p_true, samples vs true sequences |
|---|---|---|---|---|
| 4 | 4 | 4 | 0% | 10.09 vs 10.06 (+0.2 se) |
| 16 | 16 | 6 | 0% | 40.17 vs 40.27 (-0.4 se) |
| 64 | 64 | 8 | 0% | 160.88 vs 160.21 (+1.5 se) |
| 256 | 256 | 10 | 0% | 641.02 vs 641.01 (+0.0 se) |

The toy is not sequential. Its left-to-right representation is.

**Root cause 2: the decision variables.** Keep the bisection order, but make the decisions tokens instead of states.
- Each new midpoint token is drawn from its exact posterior given every token already placed (forward-backward), independently of the other midpoints of its level.
- Invalid rate: 0% at T=4, 9.6% at T=16, 44% at T=64.
- Tokens do not separate the past from the future; the state does. This is also why token bisection paid an order tax on real text (K4).

**Root cause 3: no inference path for latent decisions.**
- Latent decisions trained without an encoder do not learn: S05, a Gaussian field with a decoder, reached KL 3.35.
- With an encoder but one global plan, capacity runs out: P1 and P2 reached KL 0.40 and 0.57 at T=4, and +31 to 34% bpb on d8 text (T6).
- The decision variables must be inferable from data and must scale with T.

**Root cause 4: discrete inversion is hard to learn.**
- C1 shows that learning to recover earlier picks from their noise stalls at about 2.5 tokens, at depth 4 or 8.
- Any mechanism whose layers must invert picks inherits that limit. One whose every decision is an explicit draw from a model trained like an AR model does not.

**The requirements that follow.** A T=L generator needs:
- (a) decisions ordered coarse-to-fine, with depth O(log T), not a chain;
- (b) decision variables that separate the block, that is, states (sufficient statistics of the past for the future), not tokens;
- (c) every decision an explicit draw, never a learned inversion;
- (d) variables computable from data for training;
- (e) state capacity sized to what must cross a boundary. T2 limits token-suffix memory, not learned continuous or product-quantised states. State-space LMs show a fixed-size learned state carries most of what a transformer uses.
- (f) trainable by teacher forcing: decision variables computed from the data, conditionals fit by cross-entropy on hard targets, and no model sampling inside training (root cause 0).

## 2. Heuristics, field by field

| Field | Heuristic | What it suggests here |
|---|---|---|
| Physics (statistical) | Text's mutual information decays as a power law, the signature of hierarchical (critical, grammar-like) processes; Markov chains decay exponentially (Lin and Tegmark). | The left-to-right chain is the wrong decomposition for text's statistics. A hierarchical generator matches them, and hierarchies are sampled top-down in O(log T). |
| Physics (renormalization, tensor networks) | MERA represents critical systems with layers of local disentanglers between scales; tree tensor networks sample top-down. | Coarse-to-fine levels with local mixing at each level. Finite bond dimension is too small for text, so the bonds must be learned continuous or product-coded states. |
| Numerical analysis | Nested dissection and domain decomposition: solve the separators first, then the interiors independently in parallel, with O(log n) depth. | Interface-first generation. Separators are states, and interiors are conditionally independent given their boundary states. |
| Statistics | Markov bridges: given both end states, an interval's interior is independent of the outside. Parallel smoothers (Sarkka) parallelise HMMs. | Exact within-level independence is available once the decision variables are Markov states. |
| Neuroscience | Lashley's serial-order problem: associative chaining is too slow for skilled sequences, so the brain activates a hierarchical plan in parallel and serialises it. Competitive queuing encodes order as an activation gradient. Hippocampal replay compresses sequences in time. | Plan in parallel at several timescales, realise locally. Replay is still sequential, so it is rejected. |
| Neuroscience (predictive coding) | Higher areas predict lower ones at slower timescales. | One top-down sweep through timescales; iterated settling is rejected (C2). |
| Psychology | Levelt (message, then frame, then lemmas, then form) and Garrett's frame-and-slot model: content is selected in parallel into a syntactic frame. Exchange errors show parallel planning within a clause. Chunking (Miller) extends span. | Two tiers of separators: a small frame alphabet with exact bridges, plus content states. |
| Biology (development) | The Drosophila segmentation cascade (maternal gradients, then gap, pair-rule and segment-polarity genes) patterns a whole embryo in a few parallel steps. Boundaries come from thresholds on fields; lateral inhibition makes them sharp. Wolpert's positional information. | A few parallel refinement levels, with boundaries emerging from fields rather than counted sequentially. |
| Biology (plants) | Lindenmayer systems rewrite every symbol at once, so a string of length T grows in O(log T) steps. | Span-first (constituent) generation as a parallel stochastic grammar. |
| Biology (folding, immunity) | Proteins fold hierarchically, local structure first (Levinthal's resolution). V(D)J recombination joins independently chosen segments at designed junctions. | Interfaces designed for independent filling. Bottom-up assembly needs iteration, so it is rejected. |
| Chemistry | DNA origami and self-assembly: parts bind in one step because their interfaces encode the global shape. | Encode global structure into boundary states, then fill parts independently. |
| Mathematics | Parallel prefix needs an associative update. Sampling feedback (the pick depends on the state, the state on the pick) breaks associativity unless the state is the decision variable. Fourier and wavelet bases make Gaussian components independent. | Make the state, not the token, the decision variable. The Gaussian-only basis change is rejected, since text is not Gaussian. |
| Philosophy | Teleology (decide the end, then the path); holism (the whole fixes the parts); Leibniz's pre-established harmony (monads coordinate through a shared initial condition, each mirroring the whole). | Bridge sampling from both ends. Shared-noise coordination is what PTP does, and each position must simulate the whole: C1. |
| Fiction | Chiang's heptapods write a whole sentence at once, by a variational, end-known principle. Queneau's *Cent mille milliards de poèmes* gets 10^14 always-valid sonnets by choosing every line independently, because every line at a position shares the same interface (rhyme, metre). Borges' Library of Babel: generation as selection. | Design the representation so parts are interchangeable given their interfaces. Selection from an index is the inverse-CDF chain again, so it is rejected. |

## 3. Pool and filter

About 45 candidates went in. They are grouped below by the constraint that kills them, where:
- C1, C2: the S10 limits above;
- O: the decision-order oracle;
- V: the decision-variable oracle;
- T1 to T9: the S10 table.

| Killed by | Candidates |
|---|---|
| C1 (learned inversion of picks) | PTP variants (flat or tree pick, sweeps, cut); O-PTP; Condor-style noise tickets; K-Forcing push-forward; Gumbel-noise PTP; Leibniz-style shared-noise monads. |
| C2 / O (left-to-right refinement) | Learned Jacobi in depth (measured: about one position per stage); Parareal with a coarse predictor; deep equilibrium sampling; one-sweep checkerboard Gibbs; predictive-coding settling; hippocampal replay (fast but sequential); crystal-growth fronts; cellular automata. |
| V (tokens as decisions) | Bisection and multiscale token orders; sparse AR skeleton with parallel fill; anchor-token cut heads. |
| T2 (small fixed state) | Finite-state random-map scans (S06 PSS: shared emission permuted by state, 64 states); perturb-and-MAP chain CRFs; tree tensor networks with finite bond dimension; PCFGs over small nonterminal sets. |
| T6 / root cause 3 (flat or global latents) | Global plan with factorised decoder (P1, P2); positional-information field with independent token fates; de Finetti exchangeable latent; Gaussian-process or path-integration prior with a Gaussian-only basis; bag-of-content plus competitive-queuing sort (it solves order, but the content set's mutual dependence remains flat). |
| T3 / T4 (implicit or bijective over discrete cells) | Scoring-rule generators; IMLE; GANs; attractor or Hopfield settling; Gaussian copulas; sign-cell flows; one-step consistency from scratch. |
| Derived cost | Importance resampling of parallel proposals (the number of proposals grows as exp(KL), and KL grows linearly in T); retrieval template plus parallel edit (novel 2048-token blocks are far from any stored block, so edits regenerate most tokens). |
| Not T=L | Lanes (kept as the T=K fallback); SuperBPE and phrase units; chunk latents with sequential steps (CALM); dual-process sparse planner. |

**Non-hierarchical heuristics considered in the revision, and why each still fails.**

| Heuristic | Mechanism it suggests | Verdict |
|---|---|---|
| Coding theory: self-synchronising codes (Huffman decoding resynchronises from any offset). Coupling from the past. | Sample segments in parallel from overlapping burn-in regions under shared noise, and join where the trajectories coalesce. | Killed by C2. Under shared noise, trajectories with different histories do not coalesce on the toy: a wrong early state persists, and refinement fixes about one position per stage. |
| Dynamical systems: symbolic dynamics, beta-expansions. | Read every token off the orbit of one point. | The orbit's itinerary is the inverse-CDF chain, so this is PTP: C1. |
| Chemistry: templated ligation (short oligos hybridise to a template at once, then are joined). | Draw a scaffold, then fill short pieces in parallel against it. | Survives, but it is interface-first generation in other words: the scaffold is the separator tier. Folded into A. |
| Music: a jazz chorus over a chord progression; sonata form before themes. | The progression separates the bars, and a bar's melody is local given its chords. | Same family as A, and it has a human precedent: people plan the separators first. |
| Physics: holography, where every fragment encodes the whole. | A distributed plan held by every position. | Generating the distributed plan is the flat-latent problem: the control D. |
| Optimality theory, the heptapods' variational writing. | The block as the optimum of a global harmony. | Exact only for chain or tree energies. Chains hit T2; trees are the span cascade B. |

All of them either reduce to interface-first or span-first hierarchy, or fail on a measured limit. That convergence is itself evidence. It is also the brainstorm's main blind spot (section 5).

**Survivors:** one family, interface-first generation over learned state variables, in three forms, plus one flat control. They are the only candidates that satisfy requirements (a) to (e).

## 4. Survivors

### S11-A (lead): Bridge LM, interface-first sampling of learned state codes

**Mechanism.**
- **Encoder.** A causal encoder reads the block and gives a state at every position. A product quantiser turns it into a code z_k (for example 4 groups of 1024). Codes are deterministic functions of the text up to k, as an AR trunk's states are.
- **Bridge prior.** Codes are generated in bisection order.
  - First z_T given the prompt, then z_{T/2}, then the quarter points, and so on.
  - Every code of a level is drawn in parallel from a learned categorical (one per product group) that attends only to coarser codes and the prompt.
- **Emission.** Every token is drawn in parallel given its codes.
- **Training.** Teacher forcing: codes computed from the data, cross-entropy on the codes in bisection order, cross-entropy on the tokens, and VQ commitment with straight-through into the encoder.
  - Because codes are discrete and deterministic given the text, log p(z(t)) + log p(t | z(t)) <= log p(t). The reported bpb is an upper bound with no KL balancing and no posterior collapse.
- **Generation.** log2(T) + 2 parallel levels: 13 for T=2048. This is one fixed computation.

**How it beats each measured limit.**
- C2: there is no refinement, and the depth is log2(T) by construction. The decision-order oracle samples the toy exactly in 10 levels at T=256.
- V and K4: decisions are separator states, not tokens. The decision-variable oracle: tokens give 44% invalid at T=64, states give 0%.
- C1: there is no learned inversion. Every level is an explicit categorical draw from a model trained by teacher-forced cross-entropy, like an AR model.
- T2: the code alphabet is product-quantised (2^40 at 4 x 1024) and each level attends to every coarser code. Memory is not a token suffix.
- T6: capacity scales with T, one code per position.
- S05: the encoder is the inference path.

**Cost.**
- Training: an encoder pass, a prior pass and an emission pass, about 2 to 2.5x dense FLOPs per token. Sharing the encoder trunk with the prior is the obvious reduction, and is counted in every FLOPs-matched comparison.
- Generation: 13 sequential levels, each a pass over at most T positions, plus emission. That is roughly 2 to 3 dense prefill passes against T dense decode steps. Batch-1 speedup should be an order of magnitude at T=2048.

**Bar.** Performance-neutral plus speedup: the bound bpb within 1% of the same-FLOPs dense model, at 2x or more decode speed at T=2048 (batch 1 and 16, CUDA graphs).

**Pre-registered kills.**
- Toy (phrase-HMM, oracle context, two seeds, 16k steps):
  - pass is KL <= 0.10 and invalid <= 3% at T=4;
  - T=L viability is KL <= 0.5 at T=64;
  - kill if KL at T=64 exceeds 5, or the KL per token is at least 0.3 nats (half of RC-PTP's slope).
- Real text (d4):
  - kill if the bound exceeds 1.05x dense at T=128;
  - kill if the bisection order over the codes costs more than 3% against left-to-right over the same codes.

**Nearest work and delta.**
- Multiscale and bisection token orders (sigma-GPT; S03 depth_tree): their decisions are tokens. The oracle and K4 show what that costs. Here the decisions are learned separator states.
- VQ-VAE-2 and hierarchical discrete latents: their priors over codes are autoregressive, hence sequential. Here the prior is in bisection order, with within-level independence that is exact when the codes are Markov.
- Hierarchical VAEs (VDVAE, NVAE): they use spatial pooling hierarchies over continuous latents for images, trained by ELBO. Here a temporal interface hierarchy over deterministic causal codes trains like an AR model.
- Parallel HMM smoothers and Markov bridges: exact for a known model. Here the state space and the bridges are learned.

### Broadened beyond hierarchy (revision)

Requirement (f) re-ranks the family and admits two non-hierarchical designs.

- **S11-N1: Two-tier flat separators.**
  - Mechanism:
    - separator codes every s tokens, generated by a short teacher-forced AR over separators (T/s levels, for example 32 at s=64);
    - every segment filled in parallel given its two boundary codes;
    - within the segment, an exact first-order token chain sampled by parallel scan.
  - It satisfies (b) to (f), and (a) in the weaker sense of a fixed number of levels.
  - Risk: a 64-token segment's internal dependence beyond one token of memory must come from the boundary codes (C1 inside the segment). Smaller s means more levels.
  - Kill: at matched levels, the toy KL at T=64 is worse than A's.
- **S11-N2: Large neural HMM sampled by random-map scan.**
  - Mechanism:
    - an explicit chain over 2^16 or more states, with low-rank, context-conditioned transitions and free emissions;
    - trained by exact forward-algorithm likelihood (teacher forced, no sampling);
    - sampled exactly by associative prefix composition in O(log T).
  - Non-hierarchical, it satisfies (a) to (d) and (f).
  - It is likely killed by (e) on text: published 2^15-state neural HMMs trail LSTMs by about 50% in perplexity (Chiu and Rush 2020, PTB 125 against 83). It stays as the exact-chain reference on the toy only.
- **Ranking by (f).**
  - Deterministic-code designs (A, C, N1) lead.
  - Continuous-latent and span designs need ELBO posterior draws, which root cause 0 predicts will learn poorly. They drop from lead options to fallbacks: A's continuous variant, B, and the flat control D.

### S11-B: Constituent cascade, span-first

This is the dual of A, from L-systems and the Drosophila cascade.
- Latents live on spans (tree nodes) and are generated top-down. Children are drawn jointly given the parent, with lateral attention between neighbours at the same level (MERA-style disentanglers).
- Boundaries can be adaptive, as thresholds on fields, rather than dyadic.
- Leaves emit tokens.

**Beats the limits** the same way as A: depth O(log T), explicit draws, capacity scaling with T.

**Weaker than A in two ways.**
- Spans are not separators, so there is no exactness argument.
- Span latents are not deterministic functions of the text, so training needs an ELBO with an encoder.

**Kill:** the same toy gates as A, or no gain over A at T=64.

### S11-C: Two-tier separators (Levelt)

A with two tiers:
- a small learned frame alphabet (for example 256 states), whose bridges are computed exactly from a learned transition matrix by matrix powers, as in the oracle;
- content codes for everything else.

The frame tier carries exact long-range structure. It is an extension to test once A works, and it is gated on A.

### S11-D (control): Flat latent with an exact local chain (FlowCRF)

A coupling-flow prior over per-position continuous latents, with an exact first-order token chain given the latents, sampled by parallel scan. It satisfies (c) to (e) but not (a): there is no coarse-to-fine order. It runs as the control that isolates the value of interface-first ordering.

## 5. Red team: where S11 is weakest

1. **The toy oracle is close to tautological.** Any HMM can be bridge-sampled; the oracle shows that order and decision variables matter, and that is all. It says nothing about whether text has a compact separator state, or whether bridges are learnable. Everything that matters for the paper is decided on real text.
2. **The coarse levels may carry the same order tax token bisection paid.** The bisection order keeps the same total entropy as left to right, but its coarse conditionals are long-range predictions: the separator 1024 tokens ahead given the start and the end. Learned codes may shrink the tax measured for tokens (K4). They may not remove it.
3. **Training FLOPs.** Encoder, bisection prior and emission cost about 2 to 2.5x dense per token. At matched FLOPs the Bridge LM sees fewer tokens. Sharing the trunk between the encoder and the prior has to work, or the bar is lost before the order tax is even counted.
4. **Vector quantisation is fragile** (dead codes, codebook collapse). Product codes help. The continuous-latent variant (an ELBO with Gaussian bridges and an encoder) is the fallback, and it brings posterior-collapse risk.
5. **What "one pass" means.** At T=2048 the Bridge LM makes 13 internal sampling levels, each a network pass over the newly decided positions, and emits everything at the end. That is one fixed computation, with explicit draws inside, like RC-PTP's mid-pass cut. It is not a single trunk pass. The user decides whether this counts as T=L.
6. **Convergence or bias.** Every heuristic led to hierarchy. That fits text's power-law statistics and nested dissection, but it is also the family the ML literature knows best (VDVAE, NVAE). The real-text oracle below must be allowed to kill it.

## 6. Staged tests (pre-registered, cheapest first)

0. **Done: the two exact toy oracles above.** Order: log2(T) + 2 levels, exact. Variables: token decisions give 44% invalid at T=64.
1. **First: the real-text separator oracle (d4, Modal, before any generator is built).**
   - Train d4 models whose second 1024 tokens reach the first 1024 only through m learned summary slots. The slots are computed from the first half; the bottleneck is the whole interface.
   - Sweep m in {1, 4, 16, 64} slots of model width. Score the second half's bpb against full attention, at the same FLOPs, two seeds for the deciding m.
   - This measures what any separator must carry across a boundary. It is the text-side test of root cause 2 for the whole family.
   - This tests the strongest, Markov form of the separator idea: a small boundary state that makes the halves conditionally independent.
   - **Kill the Markov form** (endpoint-only bridges, exact within-level independence) if m = 64 costs more than 3% on the second half. A would then have to use per-position codes with attention over every coarser code, where within-level independence is only approximate.
   - **Go with the Markov form** if m <= 16 costs at most 1%.
   - Either way the result fixes how much state a separator needs on real text, which is the T2 question asked of learned states rather than token suffixes.
2. **Toy, learned (local GPU, then Modal for two seeds).**
   - Arms:
     - `bridge` (A);
     - `bridge_tok`: A with codes forced to equal the tokens, i.e. learned token bisection. This is the decision-variable control.
     - the RC-PTP T curve as the reference.
   - T = 4, 16 and 64; gates as in A.
   - Diagnostics:
     - the prior's KL per level;
     - purity of the learned codes against the true HMM state.
3. **Real-text Bridge LM (d4).** Bound bpb against dense at matched FLOPs, then one-pass speed at T=2048.

## 7. Results

### Stage 1, real-text separator oracle (d4, jimpa-17, 2026-10-04)

Setup:
- d4 at the dense FLOPs (1.14e16), window pattern L.
- m summary slots are inserted at position 1024 by shifting the second half, so every model sees the same real tokens.
- The second half reaches the first only through the slots.
- Scored on 512 validation rows against the S09 full-context dense models, as a ratio to dense seed 1.

| Model | [1088,1152) | [1152,1280) | [1280,1536) | [1536,1984) | Second half, token-weighted |
|---|---|---|---|---|---|
| dense, seed 2 | 1.005 | 0.997 | 0.998 | 1.005 | about 1.002 |
| 64 slots inserted, full attention (control) | 1.007 | 0.996 | 1.000 | 1.003 | about 1.001 |
| 1 slot | 1.081 | 1.075 | 1.004 | 0.997 | 1.015 |
| 4 slots | 1.071 | 1.074 | 0.997 | 0.996 | 1.012 |
| 16 slots (seeds 1 / 2) | 1.084 / 1.078 | 1.077 / 1.076 | 1.005 / 1.000 | 0.998 / 1.004 | 1.017 / 1.016 |
| 64 slots | 1.087 | 1.074 | 0.996 | 0.997 | 1.014 |

- **Slot insertion itself is harmless.** The control matches dense.
- **The cost sits in the first 256 tokens after the split, at 7 to 8%.** Beyond 256 tokens it is zero.
- **The cost does not depend on the slot count** (1 to 64), and seeds agree within 0.6%.
- **Against the pre-registered lines:**
  - 64 slots cost +1.3% on the second half against the control, under the 3% kill line, so the Markov form is not killed.
  - 16 slots cost +1.6%, over the 1% go line, so there is no go.
- **Diagnosis.** A capacity limit would shrink with more slots, and this cost does not. The slot path has a one-layer delay: slots hold first-half information only from their own first layer on. A 4-layer model therefore loses its first layer of direct access to the tokens just before the split, which is exactly where local continuity lives.
- **For the Bridge LM:**
  - its codes are embedded inputs at every position, visible from the first layer, so this delay does not apply;
  - the far context (beyond 256 tokens) is carried fine through a bottleneck;
  - the near context needs per-position separators, not one boundary summary, so the attend-to-all-coarser-codes design is the right one.
- **Caveat.** A delay-free oracle (slots computed by a separate compressor and injected as embeddings) would isolate capacity. It has not been built.
- **Rows whose split falls inside one document** (320 of 512; no document start within 128 tokens of the split).

| Model | First 64 after the split | 64 to 128 | 128 to 256 | 256 to 512 | Beyond 512 |
|---|---|---|---|---|---|
| Slots, 1 to 64, both seeds | +9 to 13% | +2.5 to 3.7% | +14% | 0% | 0% |
| Full-attention control | +1% | 0% | -1% | 0% | 0% |

- Dense bpb in the 128-to-256 bucket is unusually low (0.93 against about 1.19 nearby), which points to text copied across the split. A bottleneck cannot carry verbatim spans.
- The second-half average against the control is +2.5%: under the 3% kill line, but marginal.
- **Design consequence for A.** A single boundary summary loses both local continuity and long-range copying. The Bridge LM's codes must sit at every position, be visible from the first layer, and carry token identity as well as state, for example a code (state cluster, token) per position, so that copying is possible through the codes.

### Stage 2, toy Bridge LM (local GPU, one seed, 8k steps; running)

| Arm | T | Parallel steps | KL bound | Invalid | Valid-sample -log p_true vs true entropy |
|---|---|---|---|---|---|
| `bridge` (oracle separator codes) | 4 | 4 | 0.59 (single-term bound) | 3.3% | 9.57 vs 9.50 |
| `bridge_tok` (token codes) | 4 | 3 | 0.37 (exact) | 4.5% | 10.19 vs 9.50 |
| `bridge` | 16 | 6 | 2.41 (bound) | 10.2% | 42.9 vs 39.2 |
| `bridge_tok` | 16 | 5 | 1.98 (exact) | 31.4% | 45.0 vs 39.2 |
| RC-PTP best (S10) | 16 | 1 | 7.1 | 98% | n/a |

- At T=16, separator codes make about a third as many invalid blocks as token codes (10% against 31%), as the decision-variable oracle predicted.
- The Bridge LM is far ahead of every earlier one-pass mechanism: RC-PTP had 98% invalid at T=16.
- Both arms were still improving at 8k steps; invalid fell from 20% to 10% between 4k and 8k.
- The oracle-code KL is a single-term bound and is loose. Its invalid rate and valid-sample NLL are the comparable quality numbers.

### Stage 2 results, two seeds on Modal L4 (2026-10-04)

| Arm (codes) | T | Steps | Parallel steps | KL | Invalid | Valid-sample NLL above entropy |
|---|---|---|---|---|---|---|
| `bridge` (oracle separator: the filtered HMM state) | 64 | 8k | 8 | <= 9.27 (bound) | 23.9% | +8.4 |
| `bridge_tok` (tokens) | 64 | 8k | 7 | 7.87 (exact) | 80.7% | +22.6 |
| `bridge_ar` (Euclidean k-means of AR states) | 4 / 16 / 64 | 12k | 4 / 6 / 8 | <= 2.0 / 7.4 / 28.1 | 6.6% / 27% / 77% | +0.7 / +4.0 / +15.0 |
| RC-PTP best (S10) | 64 | 8k | 1 | <= 42.1 | 100% | n/a |

**Against the pre-registered gates.**
- **T=L viability (KL <= 0.5 at T=64): fails.**
- **"KL > 5" kill:**
  - Token codes are killed: exact KL 7.9.
  - For oracle codes it is undecided, since 9.3 is only a single-term bound.
- **Slope kill (>= 0.3 nats per token): not met for oracle codes.** They are at 0.145 nats per token, 4x better than RC-PTP's 0.6.

**Readings.**
1. Separator codes are the first one-pass mechanism whose quality scales usefully with T: 24% invalid at T=64 against 81% for tokens and 100% for RC-PTP.
2. Learning the bridge prior is not saturated: oracle-code invalid fell from 61% to 24% between 4k and 8k steps.
3. Euclidean k-means on AR hidden states does not find separators. Those codes are as bad as tokens (27% invalid at T=16, 77% at T=64).

**Iteration (running).** Each is a change of mechanism plus one budget check.
- **Causal-state codes (`bridge_pred`).** Pasts are clustered by their predicted next-token distribution (KL k-means). This is the computational-mechanics criterion: separators are pasts with the same conditional future, the minimal sufficient statistic, and Markov by construction.
- **Endpoint conditioning (`--bridge-endpoints`).** Each midpoint reads its interval's two end codes explicitly, the exact Markov-bridge structure.
- **Budget check.** Oracle codes at 24k steps, T=64.

### Diagnosis of learned codes, and the fix: window bisection (2026-10-04)

**The iteration results.**
- Endpoint conditioning: no effect (T=16: 9.4 to 13% invalid, against 10.2% without).
- Causal-state codes by KL k-means: worse than Euclidean (T=64: 85% invalid against 77%).
- Product-of-experts bridge head: 8.8% at T=16, within noise of no head.

**Code quality, separated from the learned prior** (`scripts/sap_code_quality.py`). Fit the best count-based Markov model over each code type and sample it with the exact bridge:

| Codes | H(state \| code) | Exact-bridge invalid, T=16 / 64 |
|---|---|---|
| Oracle (filtered argmax state) | 0.000 nats | 0.0% / 0.1% |
| Euclidean k-means of AR states | 0.376 | 89% / 100% |
| KL k-means of predicted next tokens | 0.469 | 98% / 100% |

Even codes that mix states only slightly (one code is about 1.5 states) destroy long-range consistency. Clustering does not find pure separators.

**What does.** In this toy the last two tokens nearly determine the state: H(state | token) = 0.747 nats, against 0.106 for the previous token plus the token. So make each bisection decision a short contiguous window rather than a single token:
- at each midpoint m, draw the window m-n+1..m from the exact posterior given every window already placed, independently of the other windows of the same level;
- each window's last tokens are then a separator.

| Decisions (exact posteriors) | Invalid, T=16 | Invalid, T=64 | Sequential draws at T=64 |
|---|---|---|---|
| single tokens | 12.7% | 44% | 7 |
| **2-token windows** | **0%** | **0%** | 14 |
| 3-token windows | 0% | 0% | 21 |

**Window bisection removes the learned-code problem.**
- The decision variables are tokens, so training is pure teacher forcing and the likelihood is exact.
- Copying works, since tokens are in the decisions; the real-text separator oracle showed that matters.
- Generation takes about n x (ceil(log2 T) + 1) steps.
- The learned version (`--bridge-window n` on `bridge_tok`) is running on Modal at T=16 and 64, two seeds.

**Nearest work.** The Insertion Transformer's balanced binary insertion is n=1, single-token decisions: 44% invalid here even with exact posteriors. Masked diffusion with a spread-out schedule also decides single tokens. Window bisection differs by the separator principle: each decision is long enough to pin the state, which makes within-level independence nearly exact.

### Stage 2, iteration results (two seeds each, Modal L4)

| Arm | T | Steps | KL | Invalid | Valid-sample NLL above entropy |
|---|---|---|---|---|---|
| oracle codes | 16 / 64 | 8k | <= 2.41 / 9.27 | 10% / 24% | +2.8 / +8.4 |
| oracle codes + endpoint conditioning | 16 / 64 | 8k | <= 2.51 / 9.19 | 11% / 24% | +2.8 / +9.2 |
| oracle codes + product-of-experts bridge head | 16 / 64 | 8k | <= 2.20 / 8.92 | 8% / 23% | +2.6 / +7.9 |
| **oracle codes, longer training** | 64 | **24k** | <= **4.99** | **6.9%** | **+3.7** |
| causal-state codes (KL k-means) + endpoints | 4 / 16 / 64 | 12k | <= 2.4 / 10.4 / 37.3 | 7% / 41% / 85% | +0.5 / +5.6 / +17.3 |
| window bisection, 2-token windows | 16 / 64 | 8k | 1.50 / 5.07 (exact) | 19% / 53% | +4.1 / +18.8 |
| window bisection, 3-token windows | 16 / 64 | 8k | 1.33 / 4.94 (exact) | 17% / 45% | +3.9 / +14.3 |
| single-token bisection (control) | 16 / 64 | 8k | 1.98 / 7.87 (exact) | 31% / 81% | +5.8 / +22.6 |

**Readings.**
- Training budget is the largest lever. Tripling steps took oracle codes at T=64 from 24% invalid to 6.9%, and the KL bound from 9.3 to 5.0.
- Endpoint conditioning and the product-of-experts head change little.
- Window bisection beats single-token bisection at every T: exact KL -33% at T=16 and -37% at T=64, about half the invalid rate. It is the first exact-likelihood one-pass arm to reach KL near 5 at T=64; RC-PTP's bound was 42.
- **Gate status.**
  - T=L viability (KL <= 0.5 at T=64) is not met by any arm.
  - The "KL > 5" kill line is borderline for window bisection (4.94 and 5.07 at 8k).
  - Window bisection at 32k steps is running, to separate undertraining from a structural limit.

### Stage 3 design for real text (drafted 2026-10-04, before building)

**Codes.**
- Each position's code is a pair: the token itself, plus a state cluster.
- The cluster is a k-means or product-quantised index of a frozen, teacher-forced causal LM's state after that token.
- Carrying the token makes copying through codes possible (separator oracle, within-document rows) and makes the emission trivial, so the likelihood bound is the bisection-order chain rule over (cluster, token) pairs. The cluster gives the separation that tokens alone lack (the decision-variable oracle: tokens alone give 44% invalid at T=64).
- The prior factorises as p(cluster | coarser) times p(token | cluster, coarser), with two heads.

**Training in one pass (FLOPs).** Level-by-level teacher forcing costs ceil(log2 T) + 1 passes per step, 12 at T=2048, which no FLOPs-matched comparison survives. Use XLNet-style two-stream attention with a bisection mask instead:
- a content stream, where a position sees codes of its own and coarser levels;
- a query stream, where a position sees only strictly coarser codes and predicts its own.

That is one forward pass with about 2x dense attention FLOPs, the same budget class as RC-PTP's two passes. The frozen code LM adds one forward pass without gradients.

**Generation.** ceil(log2 T) + 1 levels. Each level computes only its new positions, reading cached content-stream keys and values of the coarser levels, so total work is about one to two dense passes over T. At d4, batch 1 and T=2048 that is about 12 passes of around 4 ms each against 2048 cached decode steps of about 0.3 ms: roughly a 12x latency win before kernel work.

**Kill lines (pre-registered).**
- The bound bpb at d4 exceeds 1.05x dense at matched FLOPs (decode tokens counted once, code LM counted).
- Or the bisection order over (cluster, token) codes costs more than 3% against left to right over the same codes. That measures the order tax directly.

### Stage 2 learning curve: window bisection keeps improving with training (toy, T=64, 3-token windows, 16 parallel steps, two seeds)

| Training steps | 8k | 16k | 32k |
|---|---|---|---|
| Exact KL | 4.94 | 2.4 to 3.1 | **1.70** (1.55 / 1.85) |
| Invalid | 45% | 22 to 25% | **13%** |
| Valid-sample NLL above entropy | +14.3 | n/a | +5.9 |

- KL roughly halves with each doubling of steps and has not flattened. The gate (0.5) is about two more doublings away if the trend holds.
- This supports the user's point: these orders learn more slowly than left to right, so a fixed iso-FLOP budget understates them. The longer-training protocol (Stage 3b) applies on real text.

### Stage 3a, real-text window bisection (d4, pre-registered 2026-10-04, before results)

**Model.** Window bisection is trained two-stream (`nanochat/wbisect.py`, `--wb-window n`).
- A 128-token prefix runs left to right. The 1920-token block is generated in window-bisection order.
- The factorisation is exact: `tests/test_wbisect.py` shows all rows sum to probability 1, and that no token sees its own or any same- or later-step content.
- Parallel steps for the 1920-token block: n=1 takes 12, n=4 takes 40, and n=16 and n=64 more (logged per run). Left to right takes 1920.

**Comparison.** Matched tokens with the S09 dense d4 models (121M tokens). Scored per position on 256 validation rows, on identical targets.

**Gates.**
- **Kill:** bpb on the block (positions >= 128) above 1.05x dense at every n.
- **Go:** some n within 1% of dense on the block, at its step count.
- n=1 is single-token bisection, the order whose tax killed token-level trees. The gap between n=1 and larger n measures the separator principle on text.

**Cost note.** Two-stream training costs about 2x a dense step per token at d4 (the head runs on the query half only). This first comparison is at matched tokens. A FLOPs-matched comparison follows only if the order passes.

**Results (2026-10-04, d4, matched tokens, one seed, 256 rows; ratio to dense seed 1, and dense seed 2 is within 1%).**

| Window n | Parallel steps for 1920 tokens | Prefix [0,128) | [128,256) | [256,512) | [512,1024) | [1024,2047) | Block, token-weighted |
|---|---|---|---|---|---|---|---|
| 1 (single-token bisection) | 12 | 1.104 | 1.164 | 1.200 | 1.253 | 1.251 | **+24%** |
| 4 | 40 | 1.045 | 1.037 | 1.090 | 1.113 | 1.118 | **+10.8%** |
| 16 | 126 | 1.020 | 1.007 | 1.065 | 1.085 | 1.086 | **+7.8%** |
| 64 | 376 | 1.010 | 1.030 | 1.078 | 1.096 | 1.106 | **+9.5%** |

**Verdict: the pre-registered kill is met.** The block exceeds 1.05x dense at every n; the best is +7.8% at n=16.

**Readings.**
1. **The separator principle carries over to text, but only partly.** Windows cut the single-token order tax from 24% to 8%. The toy predicted near zero.
2. **The tax does not vanish as windows grow; it turns back up at n=64.** Larger windows put more tokens into the coarse windows (64 tokens each, about 1900 tokens ahead of the prefix), whose conditionals the model learns poorly.
   - The block's cost is spread evenly from 256 tokens on (+6.5 to 8.6%).
   - Even the left-to-right prefix pays 1 to 10%. Capacity spent on the hard far-ahead conditionals is taken from everything else.
3. **The two constraints pull against each other.**
   - On text, a 16-token window does not carry the state that separates siblings: T2 measured +7 to 9% for 32-token suffix memory, about the same size.
   - Making windows long enough to carry it front-loads high-entropy far-future prediction.
4. **Against the dense depth frontier** (d4: 1 layer, 2.5x at +14.7%), 126 steps at +7.8% is a far better speed trade.
5. **But the 1% bar is missed by about 8x, and FLOPs-matched it would be worse:** two-stream training costs about 2x per token.

### Stage 3b, bridged lanes and the longer-training protocol (pre-registered 2026-10-04, before results)

**Mechanism (user-approved).** Keep text's left-to-right conditionals for the bulk of the block, and bisect only the separators (`nanochat/wbisect.py::bridged_lanes_steps`, `--wb-lanes L --wb-window n`).
- The 1920-token block is split into L intervals.
- Each interval's last n tokens (its separator) are placed coarse-to-fine in bisection order over intervals.
- Then every interval is filled left to right in lockstep. Each fill lane starts after its left neighbour's separator and ends at its own, which fixes the seams that hurt lanes (S08).
- The likelihood is exact (normalisation test). Parallel steps: (ceil(log2 L) + 1) * n + (interval length - n).

| Configuration | (128, 4) | (64, 4) | (64, 8) | (32, 8) | (16, 8) |
|---|---|---|---|---|---|
| Steps for 1920 tokens (left to right needs 1920) | 43 | 54 | 78 | 100 | 152 |

**Gates at matched tokens (1x).**
- **Kill:** block bpb above 1.05x dense at every configuration.
- **Go signal:** 1.02x or better at 100 steps or fewer.

**Longer-training protocol (user decision, 2026-10-04).** SAP variants predict many tokens per step, a harder learning problem, so a strict iso-FLOP comparison under-trains them.
- SAP and dense are both trained at 1x, 2x and 4x the dense compute-optimal tokens (d4: 121M, 242M, 484M), for (64, 4) and (32, 8).
- **Readouts:**
  - the gap at equal tokens, at each budget, which shows whether the order tax shrinks with training;
  - tokens-to-parity: the multiple of dense's 1x tokens SAP needs to reach dense-1x bpb.
- **Framing:** inference-aware scaling (Sardana et al. 2023, "Beyond Chinchilla-Optimal"). When deployment decoding dominates lifetime compute, a k-times training cost buys an s-times decode speedup.
- **Proposed bar, pending user confirmation:** SAP reaches dense-1x bpb within 1% at 4x tokens or fewer (about 8x FLOPs with two-stream training), with at least 10x fewer sequential decode steps. The bpb at equal tokens is reported alongside it.

**Results at 1x tokens (2026-10-04, d4, one seed, 256 rows; ratio to dense seed 1).**

| (L, n) | Steps | [128,256) | [256,512) | [512,1024) | [1024,2047) | Block, token-weighted |
|---|---|---|---|---|---|---|
| (128, 4) | 43 | 1.051 | 1.095 | 1.114 | 1.118 | **+10.9%** |
| (64, 4) | 54 | 1.064 | 1.120 | 1.127 | 1.138 | +12.8% |
| (64, 8) | 78 | 1.067 | 1.117 | 1.127 | 1.133 | +12.5% |
| (32, 8) | 100 | 1.082 | 1.131 | 1.135 | 1.147 | +13.7% |
| (16, 8) | 152 | 1.098 | 1.143 | 1.155 | 1.164 | +15.4% |

**Verdict at 1x: the kill line is met** (above 1.05x at every configuration). Bridged lanes do not beat window bisection at equal step counts: window bisection gives +10.8% at 40 steps and +7.8% at 126.

**Diagnosis.**
1. The separators are the expensive tokens. They are L x n tokens (256 to 512 here) placed far ahead with little context, which window bisection already showed the model predicts poorly.
2. The fill does not have the context the design assumed. A fill lane's first tokens see only their left neighbour's n-token separator as immediate context. The rest of the neighbouring interval is generated in the same lockstep steps, so longer intervals leave a larger hole (up to 112 tokens at L=16). That is why fewer, longer lanes do worse here, the opposite of the intent.
3. Across families the cost tracks the number of parallel steps:
   - window bisection: 12 steps +24%, 40 steps +11%, 126 steps +8%;
   - plain lanes (S08, no far-ahead tokens): 240 steps +1.6%, 480 steps +0.9%, 960 steps +0.5%.
   - Plain lanes at L = 16 to 128 (120 down to 15 steps) have not been measured. They are the direct test of whether far-ahead placement or seams dominate the cost. They are queued for the next Modal profile (jimpa-17 is out of credit).

**Longer-training protocol results (2026-10-04, d4, one seed, 256 rows; block bpb as a ratio to dense at 1x, seed 1, same rows).**

| Model | 1x tokens (121M) | 2x (242M) | 4x (484M) |
|---|---|---|---|
| dense | 1.000 | 0.956 | 0.933 |
| bridged lanes (64, 4), 54 steps | 1.128 | 1.021 | **0.982** |
| bridged lanes (32, 8), 100 steps | 1.137 | 1.016 | **0.985** |
| Gap at equal tokens, (64, 4) / (32, 8) | +12.8% / +13.7% | +6.9% / +6.3% | +5.3% / +5.6% |

**Readings.**
1. **The tax at equal tokens shrinks with training,** from 13% to 6.5% to 5.4%, which supports the user's point that these orders learn more slowly. It is levelling off near 5% by 4x.
2. **Tokens-to-parity against dense at 1x: about 2.8x by log-linear interpolation.** At 2x both configurations are within 1.6 to 2.1% of dense-1x; at 4x both are below it (-1.8% and -1.5%).
3. **Against the proposed longer-training bar** (dense-1x bpb within 1% at 4x tokens or fewer, with at least 10x fewer sequential steps), both configurations pass:
   - (64, 4): 54 steps for 1920 tokens (35x fewer), 1.8% better than dense-1x at 4x tokens;
   - (32, 8): 100 steps (19x fewer), 1.5% better.
4. **Caveats before any claim.**
   - One seed. Dense seeds differ by up to 1% per bucket.
   - 4x tokens with two-stream training is about 8x dense-1x FLOPs.
   - The step counts are not measured decode speed: a cached, CUDA-graph generation loop for these orders is not built yet.
   - At equal tokens the order still costs about 5%.
   - Window bisection (n=16: +7.8% at 1x, the lowest 1x tax) and plain lanes at L >= 16 have not been run at 2x and 4x.

**Next, on the new Modal profile (jimpa-17 is out of credit):**
- a second seed for (64, 4) at 4x and for dense at 4x;
- window bisection n=16 at 2x and 4x;
- plain lanes at L = 16 to 128, at 1x and 4x (`s08_lanes(mult=...)`);
- a generation loop with timing for bridged lanes.

**User decision (2026-10-04): this is now the SAP bar.** Within 1% of dense-1x bpb using at most 4x dense's tokens, with at least 10x fewer sequential decode steps; the gap at equal tokens is reported alongside, framed as inference-aware scaling. The earlier "within 1% at the same FLOPs" bar is retired.


### Measured decode speed (2026-10-04, archaeonseq H100, d4, `modal_sap.py::s11_speed`)

Both decoders run under CUDA graphs at temperature 1, timed on the same GPU in one container.
- **Next-token decoding:** one captured step, replayed once per token.
- **Two-stream decoding:** `WBDecoder` runs one cached pass per step, over the previous step's tokens (content, whose keys and values are then cached) and the current step's queries. Each step has its own static shapes, so each is captured as its own graph, and the graphs are replayed in order.
- **Workload:** 1920 tokens after a 128-token prompt. Prefill is excluded for both.
- **Weights:** timing depends on the model shape and the schedule, not the weights, so one d4 checkpoint (`S11bl64n4x1_s1`) times every schedule.

| Schedule | Steps | Batch 1 | Batch 16 | Batch 64 |
|---|---|---|---|---|
| Next-token, tok/s | 1920 | 1,179 | 16,075 | 61,325 |
| Bridged lanes (128, 4) | 43 | 37.3x | 21.9x | 9.2x |
| Bridged lanes (64, 4) | 54 | 30.0x | 19.0x | 8.6x |
| Bridged lanes (32, 8) | 100 | 16.6x | 12.1x | 6.5x |
| Bridged lanes (16, 8) | 152 | 11.1x | 8.8x | 5.1x |
| Window bisection n=4 | 40 | 40.1x | 22.8x | 9.3x |
| Window bisection n=16 | 126 | 13.4x | 10.3x | 5.7x |

**Readings.**
1. **At batch 1 the speedup tracks the step ratio.** One two-stream step costs about 1.0 ms and one next-token step 0.85 ms. Both are bound by kernel count: profiling a d4-shaped model locally gives 264 CUDA kernels per next-token step (113 of them small elementwise kernels) and 255 per two-stream step. The two-stream step's 128 rows (64 content and 64 query rows at batch 1) cost almost nothing extra.
2. **The speedup shrinks with batch.** A two-stream step does about two rows of work per generated token (its content row and its query row), plus attention under an explicit mask over the whole cache. At batch 64 that makes it compute-bound (about 4.3 ms per step for (64, 4)), while next-token decoding at d4 is still overhead-bound.
3. **Caveat a reviewer will raise.** Neither decoder is fused or compiled. Fusing elementwise kernels (`torch.compile`) would cut the next-token step's 264 kernels, and probably the two-stream step's by a similar factor, but the two-stream step keeps its extra masked attention and its double rows. The batch-1 ratio should therefore be re-measured with both decoders compiled before it is cited, and at a larger model, where batch-1 decoding is bound by reading the weights and the step ratio should carry over directly.

### Correction (2026-10-04): the two-stream eval scored later models in the first model's order

**Bug.** `nanochat/wbisect.py::_mask_and_pos` cached the two-stream mask under the key `(steps.data_ptr(), numel, device)`. `scripts/sap_position_bpb.py` builds a fresh step tensor for each model. Once the previous model's tensor was freed, the CUDA allocator could return the same address for the next model's tensor. That gave a cache hit, and the later model was scored under the earlier model's generation order.
- **Only the first two-stream model of each eval process is guaranteed correct.** Training-time validation is unaffected, since each run holds one step tensor.
- **Fix:** the cache now holds the tensor itself and matches it by identity and version. A regression test (`test_mask_cache_follows_the_order_not_the_address`) covers it.

**What it invalidated.**
- Every per-position number for a two-stream model other than the first in its eval:
  - the 1x bridged-lanes table above, except (128, 4);
  - the bridged-lanes (64, 4) longer-training row on jimpa-17;
  - the Stage 3a window-bisection step curve, except n=16. Its non-monotone point (+9.5% at 376 steps, worse than +7.8% at 126) was a symptom.
- The diagnosis built on those rows: that longer lanes do worse and that bridged lanes do not beat window bisection at equal steps.

**What it did not touch.** Dense models, plain lanes, separators, the toy, every training-time validation number, and bridged lanes (32, 8), which was the first two-stream model in its evals. Training-time validation and the corrected eval agree to 0.1% for dense and (32, 8). Before the fix they disagreed by 1.2 to 1.8% for (64, 4) and window bisection.

**Corrected ladder (archaeonseq, d4, 256 rows, every model scored in its own order).** Values are block bpb, token-weighted over positions 128 to 2047, as a ratio to the mean of the two dense-1x seeds.

| Model | Steps for 1920 tokens | 1x (121M) | 2x | 4x (484M) |
|---|---|---|---|---|
| Dense | 1920 | 1.000 (seeds 0.9998, 1.0002) | 0.955 | 0.932 (0.9312, 0.9321) |
| Bridged lanes (32, 8) | 100 | not run | not run | **0.983, 0.983** (two seeds) |
| Bridged lanes (64, 4) | 54 | 1.082 | 1.032 | 0.994, 1.008 (two seeds) |
| Window bisection n=16 | 126 | 1.075 | 1.034 | 0.991 |
| Plain lanes L=16 | 120 | 1.022 | | |
| Plain lanes L=32 | 60 | 1.039 | | |
| Plain lanes L=64 | 30 | 1.063 | | **0.993** |
| Plain lanes L=128 | 15 | 1.107 | | |

**Readings.**
1. **Plain lanes dominate at equal step counts.** At 1x:
   - L=32 (60 steps) costs +3.9%, against +8.2% for bridged lanes (64, 4) at 54 steps;
   - L=16 (120 steps) costs +2.2%, against +7.5% for window bisection at 126 steps.
   - At 4x, plain lanes L=64 reach 0.993 in 30 steps, against 0.994 to 1.008 for bridged lanes (64, 4) in 54.
   - So placing separators far ahead costs more than the seams it removes. The S08 lanes were the better instantiation all along.
2. **Plain lanes are also cheaper to train and to run.** One stream instead of two:
   - 4x tokens is 4x dense-1x FLOPs (two-stream training is 8x);
   - one row per generated token at decode instead of two.
3. **Against the bar** (within 1% of dense-1x at 4x tokens or fewer, at least 10x fewer steps):
   - bridged lanes (32, 8) pass on two seeds (-1.7%, 100 steps);
   - plain lanes L=64 pass on one seed (-0.7%, 30 steps, 64x fewer);
   - bridged lanes (64, 4) are borderline (-0.6% and +0.8%).
4. **The 1x lane tax grows with L at about 0.7 to 0.8 doublings per doubling of L** (2.2, 3.9, 6.3, 10.7%). That is consistent with a roughly fixed cost per lane start. The plain-lanes ladder (L = 16 to 128 at 2x and 4x, a second seed for L=64 at 4x) and a per-offset breakdown of where in a lane the cost falls are running.

### Stage 3c, seeded middle-out lanes (pre-registered 2026-10-04, before results)

**Diagnosis it acts on.** Plain lanes pay their tax mostly at contextless lane starts.
- Cost per start, in average-token losses: about 2.8 at L=16, 2.4 at L=32, 1.9 at L=64 and 1.6 at L=128.
- Lanes near the prompt are nearly free (ratio 1.001 on positions 128 to 256 at L=64); lanes far from it cost about 7%.
- A lane's end is cheaper than dense, because it is written after the next lane's start is known. The per-offset breakdown in the plain-lanes ladder tests this.

**Mechanism.** `nanochat/wbisect.py::seeded_lanes_steps`.
- Split the block into K intervals and put a seed near each interval's middle (step 0).
- At step s, write seed - s and seed + s, every interval in lockstep. Text is predicted in both directions, as in middle-out decoding (Mehri and Sigal, 2018), but with many seeds at once and an exact likelihood (two-stream).
- Each interval's last position, the junction with the next interval's left front, comes last, with both sides known.
- For the same tokens per step as 2K plain lanes, this has K contextless starts instead of 2K - 1.

**Control.** `lane_order_steps` writes S08 plain lanes as a two-stream order (the same conditionals). This separates the order's effect from single-stream versus two-stream training.

**Arms** (d4, 1x tokens, seed 1):

| Arm | Steps for 1920 tokens | Contextless starts |
|---|---|---|
| plain lanes, two-stream, L=64 (`lo64`) | 31 | 63 |
| seeded, K=32 (`sd32`) | 31 | 32 |
| plain lanes, two-stream, L=128 (`lo128`) | 16 | 127 |
| seeded, K=64 (`sd64`) | 16 | 64 |

**Gates (block bpb against the mean of the two dense-1x seeds).**
- **Control validity:** `lo64` within 1.5 points of single-stream `ln64` (1.063). Otherwise two-stream training changes the lane result, and the comparison below is read only within two-stream.
- **Kill:** at both step counts, seeding removes less than 25% of the plain-lane tax, i.e. sd32 > 1 + 0.75 (lo64 - 1) and sd64 > 1 + 0.75 (lo128 - 1). Then contextless starts are not the dominant cost, and middle-out seeding closes.
- **Go:** seeding removes at least 40% of the tax at either step count. Then run the 2x and 4x ladder and build a one-stream seeded decoder (two-stream doubles training and decoding work).

**Addendum before results (2026-10-04), after the per-offset breakdown of plain lanes.** The plain-lanes ladder measured the cost by offset within a lane, as a nats ratio to dense-1x for lanes 1 to L-1:

| Configuration | Offset 0 | 1 | 2 | 3 | 4-7 | 8-15 | Lane end | Lane 0 |
|---|---|---|---|---|---|---|---|---|
| L=64, 1x | 1.96 | 1.36 | 1.21 | 1.18 | 1.09 | 1.04 | 0.86 | 0.93 |
| L=64, 4x | 1.95 | 1.34 | 1.18 | 1.13 | 1.04 | 0.98 | 0.63 | 0.89 |
| L=128, 1x | 2.05 | 1.35 | 1.19 | 1.15 | 1.06 | 0.95 | 0.77 | 0.93 |

**Readings.**
- **A lane start costs about twice a normal token, and training does not reduce it** (1.96, 1.96, 1.95 at 1x, 2x and 4x). It is an information deficit.
- **Offsets 1 to 7 cost about as much again in total.** A young lane has little context of its own.
- **Lane ends get cheaper with training** (0.86 to 0.63). They see the next lane's start.
- **The factorisation itself loses nothing:** H(left, start) = H(start) + H(left | start). The net tax is the part of the start's deficit that the model fails to recover at the previous lane's end.

**What this predicts for seeding.** Middle-out with one seed token removes one offset-0 cost per seed, but it gives each seed two young fronts. Predicted tax removed: about 20 to 25%. A seed window of m tokens, written left to right before the left front starts, lets the left front begin with m tokens of context. At equal steps that means fewer starts and no young left fronts, predicted 35 to 45% removed.

**Added arms** (`seeded_lanes_steps(..., m)`, same steps as the plain-lane controls):

| Arm | Steps | Contextless starts |
|---|---|---|
| sd40, m = 12 | 31 | 40 (left fronts start with 12 tokens of context) |
| sd80, m = 6 | 16 | 80 |

The kill and go thresholds above apply to each seeded variant against its own equal-step control.

**Stage 3c results (2026-10-04, d4, 1x tokens, seed 1; block bpb as a ratio to the mean of the dense-1x seeds).**

| Steps | Plain lanes, one stream | Plain lanes, two-stream (control) | Seeded, m = 1 | Seeded, warm window |
|---|---|---|---|---|
| 31 | 1.063 (L=64) | 1.076 (L=64) | 1.147 (K=32) | 1.157 (K=40, m=12) |
| 16 | 1.107 (L=128) | 1.124 (L=128) | 1.164 (K=64) | 1.165 (K=80, m=6) |

**Verdict: killed.** Seeding does not remove tax; it roughly doubles it at 31 steps (+14.7% against the control's +7.6%), and the warm seed window does not help.
- **Control validity holds.** Two-stream plain lanes are within 1.3 to 1.7 points of the one-stream model, so two-stream training itself costs about 1.5 points.
- **Even the prompt positions get worse** (1.05 to 1.07 against 1.02 on positions 0 to 127). The shared weights pay for learning right-to-left prediction as well as left-to-right.
- **Fronts that converge on a junction write positions two apart in the same step,** which is the adjacent-independence cost the block heads (S00 to S06) paid.
- **Durable reading.** Fewer contextless starts do not help if the order adds backward prediction or converging fronts. Within one-token-per-front orders, plain left-to-right lanes are the best instantiation measured.

### Sample quality: plain lanes pass on bpb but fail on samples (2026-10-04)

`scripts/sap_eval_generation.py --lanes` with 256 validation prompts of 64 tokens and 1985 generated tokens.
- The lane length (31) matches the layout the model trained with at that prompt length.
- Comparison: dense-1x next-token samples on the same prompts, temperature 1.
- Scorer: held-out dense-4x seed 2.

| Model | Reference ppl (next-token / parallel) | Distinct 3-grams (next-token / parallel) |
|---|---|---|
| Plain lanes L=64, 4x (block bpb 0.993 of dense-1x) | 61.5 / **150.8** | 0.822 / 0.979 |
| Plain lanes L=32, 2x (0.994) | 61.5 / **136.5** | 0.822 / 0.971 |

An earlier run used a lane length the model had not trained with at that prompt length (30 instead of 31). It gave 152.8 and 140.5, so the layout was not the cause.

**Reading.** bpb parity does not imply sample parity for a parallel order.
- The lanes model concentrates its uncertainty in lane starts: offset 0 costs 2x, and it is compensated by cheaper lane ends and interiors.
- At sampling, each start is drawn blind. Only the previous lane's last few tokens can bridge into it.
- S08 measured the same at L=2 to 8: the first 16 tokens of each later lane scored reference ppl 420 to 650, against about 60. At L=64 there are 63 seams, which predicts about 180 overall.
- The bpb tax measures the probability mass placed on incoherent seams. In sample space that mass shows up as visible breaks in most samples.

**Consequence for the bar.** A generation-speed paper needs a sample-quality criterion beside bpb, such as reference ppl of samples within a stated margin of dense-1x samples (with distinct-n reported, since small next-token models repeat themselves). Proposed to the user, not adopted unilaterally.

**Next test, running.** Bridged lanes and window bisection fill every interval between two known ends, so a fill can bridge over its whole length rather than its last few tokens. Their samples are being scored the same way (`modal_sap.py::s11_gen`).

**Real-text anchor (2026-10-04, `--real`, same prompts).** Reference ppl, distinct 3-grams and unigram entropy (nats) of each prompt's true continuation, beside the samples.

| Samples | Reference ppl | Distinct 3-grams | Unigram entropy |
|---|---|---|---|
| Real continuation (1920 tokens) | 31.5 | 0.917 | 5.87 |
| Dense-1x next-token, temperature 1 | 52.6 | 0.794 | 5.15 |
| Bridged lanes (32, 8), 4x, temperature 1 | 115.3 | 0.970 | 6.10 |
| Real continuation (1985 tokens, 64-token prompt) | 31.9 | 0.916 | 5.88 |
| Dense-1x next-token, temperature 1 | 55.4 | 0.808 | 5.21 |
| Plain lanes L=64, 4x, temperature 1 | 147.9 | 0.979 | 6.19 |

**Reading.** At temperature 1 the two kinds of sampler miss real text in opposite directions:
- next-token samples are under-diverse (entropy 0.7 nats below real text; they repeat);
- parallel samples are over-diverse (0.2 to 0.3 nats above).

So a temperature-1 reference-ppl ratio mixes diversity with coherence. The fair comparison is the quality-diversity frontier: reference ppl at matched unigram entropy, at real text's 5.87, with temperature swept for both samplers. That sweep is running (next-token at 0.8 to 1.2; bridged lanes at 0.8 to 1.2; plain lanes at 0.8 and 0.9).

**Quality-diversity frontier (2026-10-04, temperature sweep, same prompts and scorer).**

| Sampler | Temperature | Unigram entropy | Reference ppl | Distinct 3-grams |
|---|---|---|---|---|
| Dense-1x next-token | 0.8 | 3.26 | 3.8 | 0.355 (degenerate loops) |
| | 0.9 | 4.23 | 12.7 | 0.629 |
| | 1.0 | 5.15 | 52.6 | 0.794 |
| | 1.1 | 6.06 | 282.6 | 0.906 |
| | 1.2 | 6.75 | 1531 | 0.974 |
| Bridged lanes (32, 8), 4x | 0.8 | 5.01 | 14.3 | 0.757 |
| | 0.9 | 5.63 | 38.3 | 0.913 |
| | 1.0 | 6.10 | 115.3 | 0.970 |
| | 1.1 | 6.53 | 421.5 | 0.990 |
| Plain lanes L=64, 4x | 0.8 | 5.09 | 17.2 | 0.791 |
| | 0.9 | 5.72 | 49.0 | 0.935 |
| | 1.0 | 6.19 | 147.9 | 0.979 |
| Real text | | 5.87 | 31.5 | 0.917 |

**Reading (corrects the temperature-1 reading above).** Interpolated in log ppl to real text's unigram entropy (5.87), reference ppl is:
- about 199 for dense-1x next-token sampling;
- about 67 for bridged lanes;
- about 71 for plain lanes.

Bridged lanes at temperature 0.9 match real text's diversity (distinct 3-grams 0.913 against 0.917) at reference ppl 38 against real text's 31.5. Over 1920 tokens the d4 next-token sampler falls into repetition loops below temperature 1 and becomes incoherent above it.

So on this frontier the parallel orders do not produce worse samples; at real-text diversity they produce better ones.

**Caveats, being run.**
- Next-token baselines are normally sampled with nucleus sampling, which suppresses loops.
- The parallel models trained on 4x tokens, so dense-4x is the equal-tokens comparison. The scorer, dense-4x seed 2, is that model's sibling, which may favour it.

**Next-token baselines with nucleus sampling and at equal tokens (2026-10-04).** Same prompts, same scorer (dense-4x seed 2).

| Next-token sampler | Temperature | Top-p | Unigram entropy | Reference ppl | Distinct 3-grams |
|---|---|---|---|---|---|
| Dense-1x | 1.0 | 0.90 | 4.41 | 13.6 | 0.666 |
| Dense-1x | 1.0 | 0.95 | 4.79 | 24.4 | 0.739 |
| Dense-1x | 1.1 | 0.95 | 5.31 | 86.2 | 0.803 |
| Dense-4x | 1.0 | 0.95 | 4.89 | 20.3 | 0.755 |
| Dense-4x | 1.0 | 1.0 | 5.39 | 50.0 | 0.833 |
| Dense-4x | 1.1 | 0.95 | 5.70 | 90.8 | 0.859 |
| Dense-4x | 1.05 | 1.0 | 5.84 | 115.2 | 0.893 |
| *Bridged lanes (32, 8), 4x* | 0.9 | 1.0 | 5.63 | **38.3** | 0.913 |
| *Plain lanes L=64, 4x* | 0.9 | 1.0 | 5.72 | 49.0 | 0.935 |
| *Real text* | | | 5.87 | 31.5 | 0.917 |

**Reading.** At matched unigram entropy (5.63), dense-4x next-token sampling scores reference ppl of about 78 to 80, with or without nucleus sampling, against 38 for bridged lanes, and has fewer distinct 3-grams. No next-token setting reaches real-text diversity without its reference ppl rising past 80.

**Candidate mechanism.** A parallel order's longest self-conditioning chain is 30 to 100 steps instead of 1920, so sampling errors compound less (exposure bias).

**Confound to rule out first.** A d4 reference judges local fluency but is weak on global coherence, such as lanes drifting onto different subtopics. That is exactly the failure a parallel order would have. A stronger scorer is needed: a larger model on the same tokenizer, or an LLM judge, which needs the user's go-ahead because it spends their API funds.

**Stronger-scorer attempt, invalid (2026-10-04).** The d16 checkpoint on the archaeonseq volume (`out/p10_isotoken/d16/ISO_mst_ve_gattn_s1/depth_16/ckpt_base/base`, an MST model, results table bpb 0.908) scores real text at reference ppl 2365, and highly repetitive next-token samples at 19.
- It can exploit repetition but cannot read our token ids.
- The volume's `tokenizer/` and `tokenizer_sap/` have identical hashes (06978be3). So either the model trained with a tokenizer that has since been replaced, or its MST code path no longer loads it correctly.
- Its numbers are void. A valid stronger scorer needs a dense model trained here with `tokenizer_sap` (d8 or d12), or an LLM judge.

### User decisions (2026-10-04, late)

- **Sample criterion added to the bar.** Matched-entropy check: sweep temperature for both samplers and interpolate log reference ppl to real text's unigram entropy. SAP samples must score no worse there than dense-1x next-token samples, with distinct 3-grams reported. The scorer must be stronger than the models compared.
- **Coherence test: both.** The DeepSeek pairwise judge, and a dense d8 scorer trained with `tokenizer_sap`.
- **Next spend: all three.**
  - A d8 scale test of the lane tax and samples.
  - An MDLM (masked diffusion) baseline at d4 at equal steps and tokens.
  - A lane-start mechanism brainstorm.
- **Modal profile `seqaeon`.** Its `tokenizer/` is a stub; `tokenizer_sap/` was uploaded and verified.
