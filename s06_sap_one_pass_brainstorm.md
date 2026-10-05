# S06 — returning to the seed: one sampled random tape, one parallel generator

> Evidence audit, 2026-10-03: see `s07_sap_mechanism_brainstorm.md` before using the
> broader conclusions below. The logged gate failures stand. However, the semantic
> construction was not a certified optimum of marginal block likelihood; its 452
> true states do not prove capacity for learned S=64. Unweighted permutation churn
> does not prove unidentifiability, and the discrete ST coupling arm did not test
> continuous Argmax Flow. Current doubling scans do O(LS log L) work; O(LS) requires
> a work-efficient implementation. One upfront noise tape alone is not parallelism.

## Boundary restored

The target is not another small head on an autoregressive model. It is a model trained from scratch
whose primary generative computation emits the entire block in one forward invocation.

Hard constraints:

1. one model; no dense teacher, distillation, AR verifier, rejection sampler, or speculative pair;
2. sample every primitive random variable in one vectorised operation near the start;
3. after that draw, only deterministic feed-forward, circuit, or parallel-scan computation;
4. no token-conditioned sampling or refinement loop;
5. all final tokens become available together;
6. either exact joint likelihood or a genuinely joint, correlation-sensitive proper objective;
7. the generator has trunk-depth capacity rather than a small appendage to frozen AR features.

This definition allows a normal neural network's fixed layer stack and deterministic GPU scan. It
forbids the correlated tree's repeated `sample -> condition -> sample` rounds.

## Constraints derived from the completed experiments

- Per-slot CE, softened targets, deterministic messages, and slot attention all retain the
  product-of-marginals optimum.
- S01/S02 show that a realised draft does not help when its prior is independent or its training
  pairing is wrong.
- S04 shows that an exact, active 16-way global plan is still too small or too rigid: KL 1.7078,
  invalid 33.94%.
- S05 shows that matching hard sampled objects is insufficient when the score is insensitive to
  discrete correlation: KL 3.3549, invalid 53.86%, despite active noise.
- The exact local-chain ceiling on real text was about 1.21x the trunk's BPB. Therefore a small
  block head is a capacity bottleneck even when its factorisation is correct.
- The successful tree diagnostic establishes that correlated decisions can solve the toy process;
  its failure against the seed is sequential sampling depth, not the existence of hierarchy.

The architecture must therefore move dependence into an exact transport from a single random tape,
and it must make that transport part of the main model.

## Brainstorm pool: 48 mechanisms

The pool was intentionally broad before filtering.

| family | candidates |
|---|---|
| finite-state / automata (8) | HMM chain, random-map chain, semi-Markov segments, factorial HMM, weighted finite automaton, latent tree automaton, chain-tree hybrid, neural PCFG |
| reversible transports (10) | discrete bipartite flow, integer affine flow, XOR code flow, butterfly flow, learned lifting wavelet, continuous dequantisation flow, Argmax Flow, Gaussian copula, spline-simplex flow, Brenier map |
| one-step paths (7) | MeanFlow, rectified flow, categorical flow matching, consistency training, one-step score model, Schrödinger bridge, one-step latent diffusion |
| implicit joint objectives (8) | Euclidean energy, Hamming energy, variogram score, n-gram kernel score, random-checksum MMD, learned-kernel MMD, conditional IMLE, adversarial sequence GAN |
| noise-to-sequence architectures (6) | broadcast latent, low-rank field, full noise-tape Transformer, reversible neural cellular automaton, stochastic routing circuit, random Fourier program |
| decoding/training policies (9) | sampled tree rounds, CRF ancestral chain, Gibbs, rejection, proposal reranking, AR verifier, distillation, speculation, soft per-slot targets |

## Funnel

| reason removed | count | conclusion |
|---|---:|---|
| measured failure or mathematical product optimum | 9 | broadcast/field latents, Euclidean or Hamming additive scores, deterministic/soft slot objectives do not survive |
| violates the one-random-round boundary | 8 | tree rounds, Gibbs, rejection, reranking, AR verification and speculation are out |
| requires another trained model or teacher | 3 | distillation and adversarial/verifier variants are out |
| likelihood or inverse is sequential/intractable | 8 | generic PCFG, copula, autoregressive flow and unconstrained transport variants are out |
| prior art leaves no clean main-track delta alone | 6 | plain discrete bipartite flow, Argmax Flow, MeanFlow, rectified flow, consistency and ordinary SPNs are controls/ingredients only |
| projected cost cannot retain a speed claim | 4 | full-state emissions, factorial exact inference, dense grammar charts and many-proposal IMLE are removed |

Ten mechanisms remain technically defensible. Only the first three currently have a clean enough
novelty-and-performance story to lead a main-track paper; the others are oracle-backed fallbacks or
controls, not ten flags on one head.

## Survivor 1 — Permutation-State Scan (PSS), lead

Let the full-depth parallel backbone predict an initial latent-state distribution, transition
matrices `A_t`, and one base categorical token distribution `q_t` for every future position. Sample
one random tape containing all state and token uniforms:

\[
U=\{u_0, u^S_{t,s}, u^X_t: t\le L, s\le S\}.
\]

Each `u^S_{t,s}` turns row `s` of `A_t` into a deterministic random map

\[
F_t(s)=\operatorname{ICDF}(A_t(s,:),u^S_{t,s}).
\]

Function composition is associative. A parallel prefix scan computes every
`s_t=(F_t\circ\cdots\circ F_1)(s_0)` in `O(log L)` span and `O(LS)` scan work after the maps have
been sampled. Base tokens `x_t=ICDF(q_t,u^X_t)` are drawn concurrently, and the state applies an
invertible vocabulary-code permutation:

\[
y_t=\pi_{s_t}(x_t).
\]

The entire stochasticity was drawn in `U`; scan and permutation are deterministic. Exact likelihood
is the ordinary forward algorithm with emission
`e_t(s)=q_t(pi_s^{-1}(y_t))`. There is no posterior or prior mismatch.

- **Cost:** backbone once; transition construction `O(LS^2)` (sparse/low-rank later), scan `O(LS)`
  work and `O(log L)` span, one `L x V` readout. No `L x S x V` tensor.
- **Performance bar:** exact likelihood should close the toy coherence gap; at real scale it must
  achieve quality neutrality and >=2x batch-16 wall-clock speed against AR.
- **Kill:** known-parameter phrase-HMM oracle fails numerical exactness; learned `S<=64` model
  misses KL 0.10/invalid 3%; or an H100 `L=2048,S=64` scan/readout cannot retain 2x speed.
- **Nearest work:** discrete bipartite flows already provide exact parallel categorical transport;
  parallel-HMM work parallelises filtering and Viterbi. The delta is exact neural sequence
  *generation* by sampling all random transition functions once and composing them with a scan,
  plus state-conditioned invertible vocabulary transport.

## Survivor 2 — Random-Map Latent Tree (RMLT), co-lead

Use a balanced latent tree rather than a chain. Every edge owns random maps from parent state to
child state; all maps and leaf token uniforms are sampled once. Deterministic tree contraction then
propagates a root decision to all leaves, which emit tokens through state permutations.

- **Mechanism:** global branch information reaches distant leaves in `log L` deterministic circuit
  depth, without the correlated tree's repeated stochastic rounds.
- **Cost:** exact tree likelihood and sampling `O(LS^2)` work, `O(log L)` span; one vocabulary
  readout through permutation emissions.
- **Bar:** performance gain over PSS on long-range coherence at equal wall clock, or the same
  quality with a faster/smaller transition circuit.
- **Kill:** no improvement over PSS on a synthetic process with explicitly hierarchical branches;
  or state count must grow faster than PSS to fit local syntax.
- **Nearest work:** tree tensor networks and sum-product networks have exact likelihoods; S02 has a
  sampled token tree. The delta is an upfront random-map sampler integrated into a from-scratch
  block LM, not top-down sample/condition rounds.

## Survivor 3 — Multiscale Innovation Flow (MIF), lead if the discrete optimiser works

Encode token IDs with a bijective binary/finite-field code. A butterfly of reversible coupling
gates transforms independent innovation symbols into the entire output block. Gates at scale `2^k`
mix positions separated by that scale, so 11 stages cover 2,048 tokens. Sample innovations once;
the inverse butterfly emits all tokens.

- **Cost:** `O(L log L b r)` gate work for `b=ceil(log2 V)` code bits, with exact inverse and exact
  base likelihood; no vocabulary readout if the code head itself predicts innovation bits.
- **Bar:** quality neutral plus a large wall-clock gain; this is the most aggressive output-cost
  reduction in the shortlist.
- **Kill:** an offline learned transform leaves >30% of the original block total correlation in
  its innovations, or straight-through discrete gate learning misses the phrase-HMM gate.
- **Nearest work:** discrete bipartite flows directly cover reversible coupling and warn that
  straight-through learning degrades with many classes. The proposed delta must be the semantic
  token code plus multiscale source-coding objective; a plain flow is only a baseline.

## Survivor 4 — Contextual Region Circuit (CRC)

A balanced decomposable probabilistic circuit alternates sum nodes (shared latent alternatives)
and product nodes (conditionally independent regions). A full-depth context network predicts sum
weights. Sample every sum-node uniform once, then use deterministic activation propagation to the
leaves; exact likelihood is a bottom-up circuit evaluation.

- **Cost:** linear in circuit edges, targeted below `O(LR^2 log L)`; logarithmic deterministic
  depth; leaf token distributions are read once.
- **Bar:** match PSS quality with better global/local allocation at equal wall clock.
- **Kill:** circuit size needed for the phrase-HMM exceeds the PSS state budget, or real-text
  likelihood remains >1.10x the model's matched token baseline.
- **Nearest work:** SPNs and CoDD make this novelty crowded. It survives as a mechanism, but not as
  a paper lead unless the upfront-noise circuit or scaling law is materially new.

## Survivor 5 — Full-Noise Transformer + Variogram Score (FNT-V)

Give every output position its own noise vector, process the whole noise tape with a full-depth
Transformer, and deterministically decode tokens using uniforms included in the original tape.
Replace S05's Euclidean block energy with a multiscale variogram score over token-pair features.
Variogram scores are specifically more sensitive than energy scores to misspecified correlations.

- **Cost:** one full backbone at inference; training needs two generated blocks and `O(L log L r)`
  sampled pair differences, not an AR teacher.
- **Bar:** better quality than PSS at comparable inference cost, or equal quality with simpler
  kernels.
- **Kill:** an offline valid-versus-Frankenstein discrimination audit is not at least 5x stronger
  than S05's embedding distance, or the T=4 gate fails.
- **Nearest work:** proper variogram scoring and implicit generators are established. The delta is
  a discrete multiscale sequence score for a one-round LM; objective novelty alone is insufficient
  unless it unlocks AR-level generation.

## Survivor 6 — Full-Noise Transformer + Characteristic Sequence Kernel (FNT-K)

Use a strictly proper kernel score/MMD on dyadic n-gram and random-checksum sketches of the complete
hard token block. Product kernels and hashed n-gram spectra detect cross-branch Frankenstein blocks
that Euclidean distance treats as nearby. Two generator samples provide the model-model term; one
observed block gives an unbiased data-model term.

- **Cost:** one model at inference; two samples and `O(L log L R)` random features during training.
- **Bar:** same as FNT-V, with a measurable quality gain over Euclidean energy at fixed training
  cost.
- **Kill:** kernel effect size audit fails, gradient signal vanishes by `L=128`, or diversity
  collapses despite low invalidity.
- **Nearest work:** MMD generators and learned kernels include text experiments. The multiscale
  checksum construction and conditional one-round LM are the possible delta.

## Survivor 7 — Conditional IMLE Noise Transport (CINT)

For each observed block, draw `K` full noise tapes, generate `K` blocks in one batched invocation,
and update only the nearest tape under a joint sequence metric. This assigns noise to modes without
a posterior network, KL, discriminator, or teacher.

- **Cost:** `K`-fold training generation, one-fold inference; `K=8` is the maximum defensible gate.
- **Bar:** quality neutrality plus inference speed, while reporting the training multiplier.
- **Kill:** `K=8` misses the toy gate or matched-training-compute quality loses to PSS.
- **Nearest work:** conditional IMLE and multiple-choice learning. S01 WTA is not the same capacity
  class, but this is still a weaker novelty position than exact state transports.

## Survivor 8 — Direct Categorical MeanFlow (DCMF), baseline/reserve

Train a full sequence velocity/map directly from Gaussian noise to one-hot token vertices, from
scratch, with one evaluation at inference. There is no distillation or pretrained flow teacher.

- **Cost:** roughly one full backbone at inference; training requires Jacobian-vector products and
  is expected to cost several ordinary forwards.
- **Bar:** AR-level generation at one evaluation and a clear gain over published one-step text
  flows under matched data/compute.
- **Kill:** direct training cannot pass the toy gate, terminal argmax collapses entropy, or the
  novelty review shows the method is merely MeanFlow applied to tokens.
- **Nearest work:** MeanFlow is explicitly self-contained and teacher-free; FLM obtains its reported
  one-step results by distilling a many-step flow. This survives technically, not as a standalone
  main-track claim.

## Survivor 9 — Argmax Coupling Flow, mandatory exact-parallel control

Use a full-depth discrete bipartite/Argmax Flow over the whole sequence. It has parallel generation
and a likelihood or variational likelihood, and is the closest established implementation of the
seed idea.

- **Cost:** fixed coupling-layer depth; `O(LV)` terminal categorical work.
- **Bar:** a control, not the paper claim; any new exact transport must beat it per wall-clock FLOP.
- **Kill:** straight-through gradients fail at the project vocabulary, as the original paper warns,
  or quality trails the independent baseline.
- **Nearest work:** Discrete Flows and Argmax Flows directly own this territory.

## Survivor 10 — Static multiscale source-code oracle

Before learning a transport, fit a reversible dyadic transform to existing token blocks offline and
measure total correlation in its innovations. This is not a model proposal; it is the cheapest
validity oracle for the entire reversible-circuit family.

- **Cost:** CPU/GPU offline analysis only.
- **Bar:** eliminate or justify MIF before training.
- **Kill:** held-out residual total correlation >30%, with parameters far fewer than samples and a
  converged larger-sample refit.
- **Nearest work:** wavelet source coding and discrete lifting schemes.

## Recommended order

1. **PSS known-model oracle:** implement the random-map scan against the existing phrase-HMM's true
   `A/E`; verify its samples reproduce exact block probabilities and invalid rate. This tests the
   sampler, not learning.
2. **PSS speed kernel:** benchmark `L=2048`, `S in {32,64,128}` on H100 before building a model.
3. **PSS learned toy:** full-depth slot backbone, exact likelihood, `S=32/64`; use fixed code-XOR
   vocabulary permutations first. No depth-8 run unless the existing 0.10/3% gate passes.
4. In parallel only at the oracle level, run the **variogram/kernel discrimination audit** on valid
   versus Frankenstein phrase blocks and stored S01/S04/S05 samples.
5. Run MIF's static transform oracle. Build RMLT only if PSS passes the toy but a deliberately
   hierarchical generator exposes a chain-state ceiling.

The first build should therefore be PSS, not another continuous global latent and not another
training loss on the existing small head.

## S06-Q quick-pass protocol (pre-registered 2026-10-03)

This screen is deliberately a mechanism test rather than ten small end-to-end language models.
Every learned arm receives the exact phrase-HMM filtering belief as its context. This removes
context-encoder optimisation as a confound; it is an oracle input and therefore cannot establish
an end-to-end result. A survivor must later work with a learned context trunk.

- `T=4`, `V=512`, one seed, at most four future-slot mixing layers, and one L4 per learned arm.
- One primitive random tape is drawn before deterministic generation. No token-dependent sampling,
  rejection, refinement, teacher, distillation, AR verifier, or second model is permitted.
- Exact-density arms report held-out block KL and sampled invalid rate. **Pass:** KL <= 0.10 and
  invalid <= 3%. **Borderline:** KL <= 0.50 and invalid <= 10%. Anything worse is killed at this
  instantiation.
- Implicit-density arms report invalid rate, valid-sample NLL, unique-block rate, and noise
  sensitivity. **Pass:** invalid <= 10%, at least 25% unique blocks, and nonzero sensitivity.
  They cannot be promoted over an exact arm merely because their unavailable KL is omitted.
- The independent optimum and the already measured S04/S05 results remain controls. Parameters,
  training multiplier, stochastic rounds, and deterministic circuit depth are reported.
- PSS uses fixed code-XOR state permutations in this gate; failure kills that scalable emission
  parameterisation, not every possible state emission. RMLT uses the same emissions for a fair
  chain-versus-tree test.
- MIF tests a small, declared bank of exact reversible lifting transforms. CRC uses a fixed-depth
  decomposable sum-product topology. The static source-code oracle fits every transform with the
  same constrained marginal predictor at two sample budgets; it is not a free per-context table.
- DCMF implements the MeanFlow identity with the required JVP and one function evaluation at
  inference. Argmax coupling uses the straight-through discrete shift from the prior-work family
  and is explicitly a control, not a novelty claim.

The screen only answers whether a defining mechanism contains signal at `T=4`. Passing does not
justify a depth-8 real-text run. A pass earns an end-to-end learned-context reproduction, then an
`L=2048` wall-clock kernel benchmark before real-text training.

## S06-Q results (2026-10-03)

The ten depth-4/width-128 screens ran for 2,500 updates on separate L4s with fixed held-out oracle
contexts. None passed or reached the borderline gate.

| mechanism | block KL | invalid | conclusion |
|---|---:|---:|---|
| PSS, 64 states + fixed XOR emissions | 5.015 | 54.39% | best invalidity, but likelihood worse than the ideal independent model |
| RMLT, 64 states + fixed XOR emissions | 5.453 | 58.35% | tree topology did not improve the chain |
| MIF, four-transform exact mixture | **4.250** | 60.01% | best exact likelihood; only 0.199/4.449 nats (4.5%) of TC recovered |
| contextual region circuit | 14.348 | 96.73% | hard capacity failure |
| FNT + variogram | n/a | 70.51% | active/diverse, but worse than independent invalidity |
| FNT + characteristic kernel | n/a | 69.87% | active/diverse, but worse than independent invalidity |
| conditional IMLE (`K=8`) | n/a | 84.67% | expensive and incoherent |
| direct categorical MeanFlow | n/a | 96.73% | one-step transport learned its regression loss, not discrete support |
| Argmax coupling control | 5.023 | 67.24% | exact prior-work control trails independence |
| constrained source-code oracle | 5.091 | 65.53% | identity won; every arithmetic/XOR transform was worse |

The matched ideal independent control had total correlation 4.449 nats/block and roughly 62–64%
invalid samples. Thus PSS has a real sample-support signal (about 8.9 absolute invalid points better
than its matched control), while MIF has a small likelihood signal. Neither is remotely close to the
3% invalid / 0.10 KL bar even with the true context belief. The source-code result rejects the
specific fixed arithmetic/XOR innovation family: `identity` KL 5.091, versus `delta` 6.717,
`butterfly` 7.144, and `xor_butterfly` 6.921.

The FNT-V initial run exposed a zero-distance `sqrt` derivative singularity in the variogram
implementation. It was fixed with a pre-square-root clamp and rerun from scratch; the table contains
only the corrected finite result.

**Verdict:** there is no depth-8 candidate in these ten instantiations. The useful residue is not a
request for more steps: PSS says shared latent-state paths can reduce off-support sampling, while the
fixed code permutations prevent those states from expressing the right semantic token families.
The next mechanism, if pursued, must change that transport—for example a learned bijective semantic
token code with cheap state-conditioned permutations—rather than enlarge the same XOR state model.

## S06-S semantic-PSS oracle (pre-registered 2026-10-03)

Before trying to learn a global semantic code, test a strict upper bound. The oracle receives the
true HMM state, true transition matrix, true filtering belief, and an arbitrary optimal vocabulary
permutation for every hidden state. Arbitrary per-state permutations strictly contain a learned
global code followed by XOR/affine state actions, so failure here closes semantic relabelling as a
repair for the one-spectrum PSS.

A permutation can rearrange an emission distribution but cannot change its sorted probability
spectrum. For state `s`, sort `E_s` and let its oracle permutation map probability rank back to the
correct semantic token. At each context and future position, the optimal shared base distribution
is the state-posterior-weighted mean of those sorted spectra. Exact forward likelihood and one-tape
random-map sampling then use the oracle permutations.

Run spectrum budgets `C in {1,2,3,5,9}`:

- `C=1` is the proposed semantic-PSS upper bound: one base distribution, arbitrary state bijections.
- `C>=2` keeps deterministic phrase states in one spectrum class and clusters the eight stochastic
  topic spectra into `C-1` classes. This diagnoses whether semantic alignment or spectrum mismatch
  is the bottleneck.
- `C=9` assigns all nine distinct HMM emission spectra their own class and must reproduce the true
  HMM up to floating-point and Monte Carlo error; it is the correctness control.

This is explicitly a free oracle, not a learned head: it uses generator parameters unavailable to a
real model and fits no sample-level parameters. Report 4,096- and 16,384-context estimates with
Monte Carlo standard errors. The pre-registered gate remains block KL <= 0.10 and invalid <= 3%.

- If `C=1` fails, a learned semantic code alone is killed regardless of optimiser or model size.
- If `C<=2` passes, continue only with **spectrum-class semantic PSS** and account for its routed
  readout/training cost; do not claim that token relabelling alone solved the problem.
- If only large `C` passes, close the direction because multiplying vocabulary readouts destroys the
  intended efficiency argument unless a single selected-class readout can be trained exactly.

### S06-S result

The analytic CPU evaluation completed in 5.5 seconds. The 4,096- and 16,384-context estimates are
converged and the nine-spectrum correctness control reproduces the true HMM to numerical precision.

| spectrum classes | KL @ 4,096 | KL @ 16,384 | invalid @ 16,384 | gate |
|---:|---:|---:|---:|---|
| 1 | 1.12297 ± 0.01726 | 1.15420 ± 0.00885 | 34.3506% ± 0.3710% | fail |
| 2 | 0.01459 ± 0.00249 | **0.01468 ± 0.00129** | **0.0000%** | pass |
| 3 | 0.00815 ± 0.00197 | 0.00889 ± 0.00100 | 0.0000% | pass |
| 5 | 0.00244 ± 0.00103 | 0.00309 ± 0.00055 | 0.0000% | pass |
| 9 | ~0 | ~0 | 0.0000% | correctness control |

An arbitrary semantic permutation improves the learned fixed-XOR PSS substantially, but a single
base spectrum still cannot map both one-hot phrase emissions and diffuse topic emissions: a
permutation preserves the multiset of probabilities. Splitting only those two shapes is sufficient.
The original semantic-code-only proposal is therefore rejected, but the broader mechanism survives
as **Spectrum-Class Semantic PSS (SC-PSS)**:

1. sample the complete state-transition and token-uniform tape once;
2. compose the random state maps by deterministic parallel scan;
3. route each realised state to one of two emission-spectrum classes;
4. form one selected class-conditioned token distribution and apply the state's semantic bijection.

At inference this can retain one `V`-way readout per position: the deterministic scan resolves the
class before a routed output projection. Exact training requires both class distributions, costing
`O(2LV + LS^2)` rather than `O(LSV)`. An arbitrary `S x V` permutation and its inverse require about
8 MiB total at `S=64,V=32768` with 16-bit entries. The unresolved difficulty is learning the latent
states, binary spectrum routing, and semantic bijections without the oracle HMM parameters. No
depth-8 run is justified until a learned `S<=64,C=2` toy reproduces the oracle gate.

## S06-L learned SC-PSS gate (pre-registered 2026-10-03)

Train a depth-4, width-128, `S=64,C=2` model for 8,000 updates on the same `T=4` phrase-HMM gate.
As in S06-Q it receives the exact filtering belief so the experiment isolates the generator. It does
not receive the HMM transition matrix, emissions, hidden states, spectrum labels, or oracle
permutations.

- Neural transition and two-spectrum parameters use exact block marginal likelihood.
- Half the latent states are assigned to each spectrum class; the learned transition process decides
  which class is active. A small state-occupancy term prevents unused-state interpolation, and a
  Jensen-Shannon term prevents the two spectra collapsing to the same distribution.
- State/token bijections start as random exact permutations. Every 1,000 steps after a 1,000-step
  warmup, posterior state responsibilities are accumulated over 1,536 fresh blocks. A Hungarian
  generalized-M step maximizes the responsibility-weighted token-to-base-rank score under the exact
  bijection constraint. This is part of the single model's training, not a teacher or verifier.
- Evaluate uninterrupted checkpoints at 4,000 and 8,000 steps on 512 fixed contexts with eight
  samples each. **Pass:** block KL <= 0.10 and invalid <= 3%. No post-hoc width/state/EM sweep.
- A pass unlocks an `L=2048,S=64,C=2` scan+routed-readout H100 benchmark. A failure stops before
  speed and d8, because an efficient kernel cannot rescue a generator that fails with oracle context.

### S06-L result

The uninterrupted d4 run completed all 8,000 updates and seven permutation M-steps.

| step | block KL | invalid | independent TC | independent invalid |
|---:|---:|---:|---:|---:|
| 4,000 | 3.8738 | 56.45% | 4.4013 | 62.70% |
| 8,000 | **2.9485** | **48.68%** | 4.4013 | 60.74% |

The model learns genuine dependence, but misses the 0.10 / 3% gate by a large margin. The oracle
ceiling was KL 0.0147 / 0% invalid, so the remaining gap is learnability rather than representational
capacity. The alternating bijection learner did not converge: permutation change fell only from
97.99% at the first M-step to 95.64% at the seventh, while one latent state absorbed roughly
1,900 of 6,144 posterior state-position assignments and the least-used state received fewer than 5.
State balance improved but remained far from uniform (`KL=0.795`).

**Gate verdict: fail.** Consequently the conditional `L=2048` scan/readout benchmark and d8
real-text training were not run. Reporting a kernel speed for a generator with 48.7% impossible toy
blocks would not establish performance-neutral speedup, and a d8 spend would violate the
pre-registration. The next mechanism must remove the independently relearned `S x V` assignment
symmetry—for example, a shared semantic code with a small identifiable family of state actions—rather
than tune the balance coefficient or extend the same unstable EM trajectory.

## Literature boundary checked

- Tran et al., *Discrete Flows: Invertible Generative Models of Discrete Data* (2019): exact
  parallel bipartite generation; large-class straight-through limitation. Downloaded as
  `Literature Review/1905.10347_Discrete_Flows_Tran_2019.pdf`.
- Hassan, Särkkä & García-Fernández, *Temporal Parallelization of Inference in Hidden Markov
  Models* (2021): associative scan for filtering, smoothing and Viterbi, not random-map neural
  generation. Downloaded as
  `Literature Review/2102.05743_Parallel_HMM_Inference_Sarkka_2021.pdf`.
- Geng et al., *Mean Flows for One-step Generative Modeling* (2025): from-scratch, teacher-free
  one-step continuous generation.
- FLM/FMLM (2026): continuous one-hot language flow; its reported one-step FMLM is obtained by
  distilling a many-step FLM, which is outside the user boundary.
- Kernel/variogram scoring literature: energy scores are known to have weak discrimination of
  dependence; variogram and characteristic-kernel scores directly motivate survivors 5 and 6.
