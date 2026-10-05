# S07 — learnable joint generation, without leaving the one-pass seed

Date: 2026-10-03. Status: original research proposals, **subsequently tested in S09**.

> Latest status: the separately recorded S09 gates reject the scan-flow instantiation
> and strongly disfavor the short-suffix transducer. Do not follow the historical
> build recommendation below as a current queue. See `sap_next_mechanisms_review.md`
> for the S08/S09/S10 cross-check, current user scope, and updated shortlist.

## Scope and evidence reviewed

Read the SAP planning chain: `sap_research_plan.md`, its v2/v3 archives,
`s01_sap_plan.md`, `s02_sap_plan.md`, `s04_sap_plan.md`, `s05_sap_plan.md`, and
`s06_sap_one_pass_brainstorm.md`; cross-check against SAP entries in `LEARNINGS.md`,
`OPEN_QUESTIONS.md`, `PROJECT_MAP.md`, the S06 logs, and current S06 implementation.
The unrelated MST/EET/linear-layer plans are not new SAP evidence.

The primary user requirement is a model pretrained from scratch that emits
**T=L=2048** in one fixed neural computation. No dense teacher, distillation, AR
verifier, rejection, or token-dependent neural refinement loop.

**User clarification, 2026-10-03:** smaller blocks such as T=8 are admissible as a
separate research route if the speed gain and BPB neutrality are strong enough to
support a novel contribution. They are not success on the T=L requirement. A
T=8 system generating 2048 tokens makes 256 block invocations; report that plainly.
The old small-head failures and the existing dense frontier remain its controls.

One upfront random tape is a bookkeeping convention, not a parallelism theorem:
ordinary AR can also pre-draw its uniforms. A proposal must show its actual work,
memory, and dependency span. Fixed neural layers and deterministic parallel scans
fit the S06 contract; a serial token loop hidden inside `forward()` does not.

## What survives the evidence audit

| Evidence | Defensible conclusion | Not established |
|---|---|---|
| S01 anchor, 32k steps, width 128/256: KL 3.34/3.32, invalid 54.0/52.3% | Scaling that independent-anchor instantiation did not repair its prior | All sampling-aware training is impossible |
| Consistent-anchor refiner: invalid 2.25%, versus independent anchors 26.22% | That refiner can use coherent conditioning | A learned one-round anchor prior exists |
| S02 full tree: KL 0.1118, invalid 3.91% | Correlation removes most of the toy gap | A pass of the 0.10/3% conjunction, or one-pass generation |
| Frozen d8 local head: 1.121 BPB versus 0.9306 NTP | A small appended head leaves a substantial measured gap | An exact AR factorization makes this trained head a proven optimum over all other architectures |
| Dense frontier: d4→d2 gives 1.68x at 1.034 quality ratio; d8→d4 gives 1.79x at 1.026 | SAP must beat simply using a shallower dense model | These 256-token timings predict the L=2048 frontier |
| S06 learned SC-PSS: KL 2.9485, invalid 48.68% | The registered learned instantiation failed; no speed/d8 gate unlocked | Proven convergence, a pure optimization failure, or a family-wide impossibility |

Three important S06 corrections:

1. **The semantic construction is not a certified marginal-likelihood optimum.** It
   keeps the true HMM's 452 states/transitions and averages rank-sorted emissions
   using predictive state weights. This minimizes the complete-data emission KL
   for that fixed latent construction. Marginalizing states changes the objective:
   `KL(P_Y || Q_Y) <= KL(P_SY || Q_SY)`. A bad constructed Q does not lower-bound
   the best possible marginal KL. Its C=1 failure is informative about that
   construction, not a proof that every one-spectrum latent model fails.
2. **452 oracle states are not 64 learned states.** C=2's KL 0.01468 demonstrates
   feasibility with the oracle state process, not representational sufficiency for
   the learned S=64 model. The learned KL improved from 3.8738 at 4k to 2.9485 at
   8k; that is not a demonstrated plateau. Unweighted 95.64% permutation churn can
   include swaps among low-mass/tied entries. It does not prove high-mass semantic
   instability or unidentifiability. Those remain hypotheses.
3. **The screen names exceed some implementations.** `ArgmaxCoupling` is one
   discrete modular coupling with a straight-through shift; it is not a continuous
   Argmax Flow. MIF mixed four fixed transforms, not learned semantic lifting.
   The source-code arm fitted constrained neural marginals, not the optimal
   independent distribution in every transformed coordinate system. These are
   failed proxy instantiations, not exhaustive tests of those published families.

The mathematical spectrum invariant itself is sound: a permutation cannot change
the multiset of emission probabilities. Also, zero observed invalid samples is not
automatically zero population invalidity; with 16,384 independent draws, the usual
one-sided 95% rule-of-three bound is about 0.0183%.

## Numerical bar, before a build

Both new hypotheses below target **quality neutrality plus measured speedup**:

- Actual generator joint BPB <= 1.01 times dense BPB on identical held-out bytes,
  pinned tokenizer, and equal total training FLOPs; >=2x end-to-end batch-16 decode
  throughput for 2048 emitted tokens on the same GPU with symmetric CUDA graphs.
- The 1% margin is a proposed noninferiority tolerance, not an observed result or
  a theorem converting BPB into sample quality. Reference-scored generation NLL,
  repetition and diversity must be reported together; reference PPL <=1.10x dense
  samples under the same fixed scorer is a supplementary guardrail, not sufficient
  evidence of distributional agreement.
- Also compare with the existing dense depth frontier. Reuse trained checkpoints;
  re-time them at the actual output length. Do not extrapolate the d8 frontier
  beyond its measured 1.79x point or combine different evaluation-token BPBs.
- Toy gate remains KL <=0.10 nats/block AND invalid <=3%, initially at T=4.
  A pass must be repeated with learned context and a second seed. T=16/64
  composition/copy tests precede any 2048-token language claim.
- An exact density per example still produces a Monte Carlo estimate of expected
  KL. Report uncertainty. A variational NLL bound must be labeled as a bound; it
  must not silently become an exact BPB number.

The same <=1.01x joint-BPB and >=2x batch-16 end-to-end speed gates are proposed for
the secondary T=8 route, measured over 2048 actually emitted tokens and including
all 256 block calls and prefix/KV updates. This is deliberately stricter than the
failed v4 head's 1.05x block-over-own-trunk gate: comparison is to dense absolute
BPB, not to an already degraded SAP trunk. The measured dense d8 half-depth point
(1.79x, 1.026 quality ratio at its original benchmark length) motivates using a
strong frontier control, but does not mathematically imply the 2x/1% thresholds.
These are proposed thresholds, not a guarantee of conference-level novelty.
Block Transformer/local AR and genuine parallel-flow controls must also be priced
before a paper claim. Keep primary and secondary results in separate tables.

Training budgets include all conditioning networks, readouts, training-only inverse
modules and auxiliary losses. Derive token counts from measured/accounted FLOPs.
No d8 run is authorized by this brainstorm.

## Pool: 42 candidates, filtered by mechanism

| IDs | Candidates | Disposition |
|---|---|---|
| 01–06 | independent hard CE; softened per-slot CE; deterministic slot attention; unpaired prior-noise CE; CRF marginals sampled independently; corpus-unigram smoothing | These context-only product decoders retain the marginal optimum; no joint mechanism |
| 07–12 | AR verifier; distilled one-step generator; Gibbs/refinement; sampled token-tree rounds; AR inverse-CDF with preloaded noise; proposal rejection | Outside the requested inference/training contract |
| 13–18 | more PSS states; longer permutation EM; balance-weight tuning; affine/Feistel state relabelings alone; more spectrum classes; larger global CP code | Sweeps or emission-coordinate variants without a demonstrated new learning mechanism; not the next brainstorm |
| 19–25 | dense V-by-V transitions; exponentiated low-rank energy with a claimed cheap normalizer; full PCFG chart; exact factorial HMM; enumerated block plans; unrestricted per-state Hungarian assignment; generic Born/tensor sampler | Respectively quadratic vocabulary cost, invalid normalizer shortcut, cubic chart, exponential joint state, exponential plans, SV² memory/SV³ assignment, or unresolved conditional-sampling span |
| 26–32 | another MMD weight; another variogram scale; larger-K IMLE; plain categorical MeanFlow; single linear Gaussian field; fixed-ID ordered copula; fixed arithmetic lifting | No identified repair to the tested training/capacity problem; deferred, not mathematically impossible families |
| 33–34 | genuine continuous Argmax coupling flow; non-AR learned categorical-embedding flow | Retain as prior-art controls, not new ideas |
| 35–39 | plain DA-Transformer; plain conditional probabilistic circuit; larger lexical tokens/SuperBPE; unrestricted latent copy forest; arbitrary nonlinear recurrent-map composition | First three already occupy their novelty claim; copy-forest likelihood unresolved; nonlinear function composition has no compact closed scan representation |
| 40 | scan-coupled categorical flow | New mechanism hypothesis A |
| 41 | observed-history sparse random transducer | New mechanism hypothesis B |
| 42 | lexicalized span circuit with atomic phrase leaves | Reserve only: nearest PC/phrase-token work already close, and independent joins retain unresolved cross-span dependence; no sufficiently distinct proposal yet |

**Funnel: 42 -> two mechanism hypotheses + two necessary controls.** The remaining
38 are rejected or deferred for the reasons above. This is not ten configurations
of another PSS head, and none is yet a validated main-track contribution.

## A. Scan-coupled categorical flow — preferred architectural hypothesis

### Mechanism

Replace finite hidden-state labels and vocabulary permutations with a continuous
transport whose probability mass over discrete token cells can change smoothly.
Use `b=log2(V)` real coordinates per token for the power-of-two vocabularies in
these experiments: 9 at V=512 and 15 at V=32768. Signs identify a token's binary
code. Bits are NOT sampled independently at the output: their joint signs come
from a coupled distribution across the whole block.

Draw `epsilon[1:L,1:b]` once. A fixed stack of invertible coupling layers transforms
it into z; output all token codes from `sign(z)` at the end. No intermediate token
is committed and no denoising solver is run.

In a coupling layer split channels into A and B. A parallel conditioner reading A
and the observed prefix predicts diagonal a, positive b, c and monotone spline
parameters. Transform B by

`v_t = a_t * v_(t-1) + b_t * x_t + c_t`, followed by `z_t = spline_t(v_t)`.

All coefficients are available before the scan; none reads the newly realized
v_(t-1). Affine-map composition is associative, so v is a parallel scan. Given the
output sequence, first invert the splines in parallel, then compute

`x_t = (v_t - a_t * v_(t-1) - c_t) / b_t`.

The conditional Jacobian is triangular with log determinant `sum log(b_t)` plus
the spline derivatives. Alternate channel partitions/directions across layers.
This gives globally propagated continuous information within each layer without
tabulating all possible discrete states. Stable coefficients and numerical inverse
checks are essential; real-valued state is not an unlimited-precision free memory.

### Training and difference from S05/S06

For a training token, sample continuous z only inside its sign cell using a small
trainable stochastic-inverse module, and maximize

`E_q[log p_theta(z | prefix) - log q_phi(z | tokens, prefix)]`.

This is a lower bound on categorical log likelihood, not an embedding-distance
score or a hard-sampling straight-through gradient. It directly trains a normalized
continuous joint to put mass in observed token cells. A simple inverse can share
token embedding/table parameters; it is a training-only part of the same model,
not a dense AR teacher. Its cost and the earlier open policy question about
recognition modules must be stated before implementation, not hidden.

Prior/posterior mismatch is still possible. The cell construction removes arbitrary
S-by-V assignments; it does not magically solve variational optimization.

### Cost and kill tests

- With D layers and fixed spline-bin count, diagonal scans have O(D L b) work using
  a work-efficient implementation and O(D log L) span. Conditioners dominate:
  dense attention adds O(D L² d), FFNs O(D L d²), and prefix cross-attention its
  own O(D L C d). Count these, rather than advertising only scan FLOPs.
- Flow-state memory is O(D B L b) during training plus conditioner activations.
  There is no mandatory L-by-V generation tensor for binary cells. This does not
  mean a binary output code is automatically an efficient or accurate representation.
- First certify inverse/logdet correctness on a tiny continuous example; estimate
  cell probabilities with low-dimensional quadrature or controlled Monte Carlo.
  Check likelihood-versus-sampler agreement, not only low training loss.
- One bounded d4 toy comparison against the genuine continuous coupling control:
  fail if the joint-KL upper confidence bound cannot get <=0.10 and invalid <=3%.
  If only a loose variational bound fails, call the gate inconclusive/closed to
  scaling, not a proof that the generator family fails. Compare prior samples too.
- Reject the scan primitive as a research lead if it gives neither at least 10%
  lower held-out KL at equal training FLOPs nor at least 20% lower generation
  latency at noninferior toy KL versus its continuous coupling control. These are
  proposed mechanism-selection thresholds; passing is not the language-model bar.
- Reject scaling if a realistic L=2048 graph benchmark cannot achieve the common
  2x speed target, or numerical instability/expensive conditioners consume it.

### Closest work and potential contribution

[Argmax Flows](https://arxiv.org/abs/2102.05379) already provides continuous-to-token
surjections, binary Cartesian coding and probabilistic inverses. [FlowSeq](https://aclanthology.org/D19-1437/)
already uses non-AR Transformer coupling layers; [parallel linear recurrence](https://arxiv.org/abs/1709.04057)
already provides the scan algebra. The proposed delta is an invertible scan coupling
with cheap likelihood inversion and long-range sample interaction, evaluated as
the primary from-scratch block generator. Novelty of that combination is **not
established** by this search; simply applying Argmax Flow to SAP is not a paper.

The Argmax paper's text8 table gives coupling flow 1.82 bpc versus its AR flow
1.39 and cited AR TransformerXL 1.08, on that paper's settings. This is a warning
about the parallel-flow quality gap, not a cost-matched comparison with nanochat.

## B. Observed-history sparse random transducer — cheapest structural oracle

### Mechanism

Change *what a state means*. Use a training-corpus suffix/phrase automaton with a
deterministic, observable update `s_next = delta(s, token)`, not exchangeable hidden
states with separately learned vocabulary permutations. State labels are fixed by
token histories. The full-depth context network predicts every future position's
transition parameters at once.

Use a normalized full-support emission with sparse history-dependent exceptions:

`p_t(y | s,h) = (1-g_t(s,h))*q_t(y|h) + g_t(s,h)*r_t(y|s,h)`.

q is one full-vocabulary distribution per position. r has a fixed training-derived
support of at most k tokens per state, with neural weights. This changes both
emission shape and identity; it is not a relabeling of one base spectrum. Corpus
tables define observable topology/support, not externally generated target labels.

During training, each target block determines its state trajectory. The joint
likelihood is the exact sum of these conditional log probabilities, with no EM,
posterior state assignment or sampled-noise/target pairing. This is not the S01
marginal-CE objective: its conditions include the actual observable automaton state.

During generation, pre-sample a token `Y_t(s)` for every possible source state.
Define `F_t(s)=delta(s,Y_t(s))`; scan these finite maps and then gather the token
for each realized source state. All neural predictions precede this deterministic
scan, so token histories do not trigger another neural call.

One q sample per position can be reused across counterfactual states: only one
state is visited, and its distribution has the correct mixture marginal. Randomness
must remain independent across time. Counterfactual coupling does not change the
realized chain's law. Prove this and test against exhaustive tiny-sequence likelihoods.

### Cost and kill tests

- Core generation work is one context network/readout O(L d V), sparse-row scoring
  and draws O(L S k r) for rank-r logits, and work-efficient map scan O(L S).
  No dense L S² transitions or L S V emissions. Ordinary all-prefix doubling instead
  costs O(L S log L), as in current S06 code; do not conflate the algorithms.
- Store O(B L S) maps and emitted labels, O(S k) sparse supports, and either a
  compiled O(S V) deterministic update table or a measured sparse lookup structure.
  At B=16, L=2048, S=512 one int32 map buffer is 64 MiB; a uint16 S-by-V update
  table at V=32768 is 32 MiB. Multiple maps, token labels, gradients and scores add
  to this. Materializing all sparse logits may dominate and must be tiled.
- Start with a constrained, training-only suffix topology and held-out likelihood
  fit on existing data, with S<=512 and k<=16 as a proposed initial systems budget.
  Report parameters/sample count, support coverage by frequency, and a 2x-data
  refit. Do not call this learned fit a global optimum or use true HMM states as
  if they were learned token histories.
- Kill this budgeted construction if its oracle-context T=4 fit misses 0.10/3%,
  or if the sparse exception path cannot account for the observed dependence.
  Full support does not remove the S03 escape problem: whenever q dominates,
  current-token probabilities ignore the realized history. Measure that tax.
- Before neural pretraining, test the observable-state sufficiency on longer
  held-out blocks. A state that only remembers a short suffix can forget distant
  choices. The proposal fails if meeting the quality target requires a state table
  beyond the measured speed/memory budget; do not respond with an open-ended S sweep.
- For a finite-state separator, `I(left;right | h) <= log S` under the modeled law.
  Existing d8 cut-MI estimates 2.70/4.22 nats at T=4/8 warn against tiny separators;
  they do not certify sufficiency at S=512 or predict the L=2048 requirement.

### Closest work and potential contribution

Variable-order/backoff language models and observable finite-state models are old;
[Scaling HMM LMs](https://arxiv.org/abs/2011.04640) already studies large sparse
state/emission structures. [Parallel HMM inference](https://arxiv.org/abs/2102.05743)
is also prior art, although filtering is not the same operation as this sampler.
The proposed distinction is context-conditioned sparse innovations on an observable
transducer, with exact supervised likelihood and compiled parallel random-map
generation. Neither an n-gram model nor the use of a scan alone supplies novelty;
only a demonstrated new quality/latency regime would justify pursuing this framing.

## Controls retained, not padded into new suggestions

1. **Genuine continuous Argmax coupling flow.** Necessary to distinguish the new
   scan primitive from established parallel transport and repair the S06 naming gap.
   Use the same cell encoding/inverse and derive its conditioning-network budget.
   It must meet the same toy and real-text gates; a win is a baseline win, not new
   architecture evidence. Its primary risk is the published coupling-versus-AR gap.
2. **Learned categorical-embedding flow.** A non-AR version of
   [Categorical Normalizing Flows](https://arxiv.org/abs/2006.09790)/FlowSeq is the
   representation control if fixed sign geometry appears limiting. Token-local
   stochastic embeddings and decoder put inter-token dependence in the flow. Count
   the V-wide decoder and variational bound; do not use an AR flow inverse, an AR
   ranking model, or multiple-proposal decoding. Failure of the first sign-coded
   control alone does not establish that learned semantic cells are impossible.

## Proposed sequence, with no jobs launched

1. Preserve the S06 failure and its unpassed gate, but correct its broader claims.
2. Do the cheap observable-state sufficiency/oracle audit for B and numerical
   invertibility/density checks for A. Analytic success only earns a toy comparison.
3. If building next, prioritize **A plus its genuine continuous-coupling control**;
   this changes the joint transport and the discrete learning signal together.
   B advances only if its structural oracle says a feasible table retains dependence.
4. Learned-context toy reproduction, longer-block dependence tests, then symmetric
   L=2048 end-to-end inference and training-memory measurements. Only then propose
   a cost-matched real-text run using the existing dense checkpoints as references.

The user's smaller-block allowance adds a predeclared secondary T=8 evaluation of
these mechanisms, not an automatic revival of every old head. If T=8 clears the
quality/speed/novelty bar while T=2048 fails, report a smaller-block result and the
unmet original requirement. Do not describe it as full-block one-pass SAP.

Serving measurements must include peak VRAM and request concurrency. Whole-block
generation delays the first usable token and creates bursty output; queueing and
backpressure at high batch can erase throughput gains. No claim is based on an
isolated scan, fewer sampling calls, or eager AR launch overhead.

## Primary literature saved and inspected

New downloads in `Literature Review/`:

- `2102.05379_Argmax_Flows_Hoogeboom_2021.pdf`: stochastic inverse, binary Cartesian
  products, continuous coupling versus AR results.
- `2006.09790_Categorical_Normalizing_Flows_Lippe_2021.pdf`: token-local learned
  encodings, decoder factorization, variational objective.
- `D19-1437_FlowSeq_Ma_2019.pdf`: parallel coupling architecture and decoding policies;
  its AR-reranked NPD and multi-proposal IWD are not the proposed inference route.
- `2011.04640_Scaling_HMM_LMs_Chiu_2020.pdf`: blocked emissions, neural parameterization,
  state-size and inference costs.
- `1709.04057_Parallel_Linear_RNN_Martin_Cundy_2018.pdf`: associative recurrence,
  work-efficient scan and diagonal-versus-dense composition costs.

Also inspected the existing DA-Transformer and CoDD PDFs. DA-Transformer already
learns a latent path through parallel word representations; CoDD already composes
factorized neural potentials with a tractable circuit prior. Neither should be
reintroduced as a new SAP mechanism under a different name. This is a targeted
mechanism review, not a claim of exhaustive prior-art clearance.
