# S13: back to the seed (one head, T=L)

> 2026-10-04.
>
> User decisions:
> - Go back to the seed. One head emits the whole block in one pass, or in a fixed number of levels. The output is unverified and the model is trained from scratch.
> - Lanes keep to "unverified", but they are not novel and they are not one pass.
> - Brainstorm from what is actually blocking the seed, using heuristics from outside ML papers.
> - Run only minimal quick experiments that show whether each direction has promise.
> - Modal profile eqyve967. No baseline is retrained, and nothing moves through the user's connection except tiny files.

## 1. What blocks T=L (measured)

**1. Local entanglement.**
- Adjacent tokens hold 19/47/66% of block NLL at T=2/4/8 (K1).
- A token decided without its left neighbours costs 2.0x (d4) to 2.35x (d8) a normal token, and the gap grows with model size.

**2. "One pass" means rounds.**
- A generator must read its own output. Seeing only the prompt plus the last 32 tokens costs 7.5 to 8.7% (T2).
- Noise-to-token maps learn about 2.5 correct tokens per call, at depth 4 or 8 (C1).
- Left-to-right refinement fixes about one position per stage (C2).
- Everything that has worked decides tokens in rounds that read earlier rounds.

**3. Most of the junction loss is never recovered.**

This comes from the logged per-offset profiles at equal tokens (`s11_d8_ladder.log`, `s12_align_d4.log`). The units are average-token losses per lane:
- the deficit is all offsets that cost more than dense (offsets 0 to 15);
- the recovery is all offsets that cost less (offset 16 to the lane end).

| Model (1x) | Deficit | Recovery | Recovered | Net per lane | Block tax |
|---|---|---|---|---|---|
| d4, L=64 | 2.40 | 0.47 | 20% | 1.93 | 6.3% |
| d4, L=32 | 2.88 | 0.37 | 13% | 2.50 | 3.9% |
| d8, L=64 | 3.64 | 1.27 | 35% | 2.38 | 7.8% |
| d8, L=32 | 4.36 | 1.11 | 25% | 3.25 | 4.9% |

- For an exact order, the chain rule loses only the total correlation (TC) among tokens decided in the same round. So in the ideal model, the end of the previous lane wins back almost everything the next lane's start lost.
- Measured, 65 to 87% of the deficit stays lost.
- Recovery comes mostly from a lane's last 10 to 15 tokens. Tokens far from the junction gain almost nothing: offsets 16 to 58 at L=32 cost 0.98 to 0.99x dense.
- From d4 to d8 (L=64), recovery grew 2.7x and the deficit 1.5x, so the net loss grew only 23%. This is the one scale trend in our favour.
- **Open question:** is the unrecovered part same-round TC (information), or learnability (the bridging conditionals are learned badly)? Three results point partly to learnability:
  - K4: a dedicated full-depth model per bisection level still lost 10% at T=4;
  - learned window bisection on the toy reached KL 1.70 where exact sampling is perfect;
  - recovery rises with model size.
- A mechanism that cuts the deficit at its source helps whichever cause dominates.

**4. Latents that are sampled, inverted or posterior-drawn during training do not learn (root cause 0).**
- Flows collapsed: latent KL 0.005 nats/token, bound 2.32 bpb against dense's 0.95.
- k-means codes were impure: 77 to 85% invalid on the toy.
- PTP inversion stalled at about 2.5 tokens per call.

**5. Text has no small separator.**
- Far context (beyond 256 tokens) crosses a summary for free.
- Near context needs verbatim per-position access, and copying needs token identity.
- Token separators placed far ahead cost 7.5% (window bisection, 126 rounds).

**6. Speed is set by rows as well as rounds (roofline, from computer architecture).**
- One round reads the weights once, whatever its row count up to the ridge point (about 300 rows on an H100). A one-pass generator of L tokens computes L rows.
- So below some number of rounds, fewer rounds stop paying. Above it, only fewer rows per token (K tokens per row, which is the LCA head itself) cut the cost.
- Q3 measures where that point is (section 6).

**7. The bar shapes the search.** A bpb bar needs an exact factorisation or a tight bound, which favours exact orders. One-pass latent generators can only be judged by the sample criterion.

**8. Novelty squeeze (2026).** Any new mechanism must differ from:
- structured joints inside diffusion steps: CoDD (probabilistic-circuit layer), tensor-train joints;
- one-step flow maps: FLM/FMLM, which are distilled;
- latent diffusion at 2B: Cola DLM;
- coupled-noise one-step blocks: Condor;
- parallel streams: PDT, Hogwild, Multi-Stream.

**Reframe.** Lanes with a fixed lane length S already decode any L in S rounds. That meets the fixed-levels definition of T=L accepted earlier. What they lack is a small tax at small S (+10.7% at S=15 on d4) and a novel mechanism.

## 2. Assets to combine

| Asset | Where it was strong |
|---|---|
| Plain lanes | Exact, teacher-forced, learnable. Pass the bpb bar at 4 to 5x tokens; sample parity at d8. |
| Lane-end bridging | The junction token costs 0.43x dense once the next start is known. |
| Interface-first state decisions | Exact in log2(T)+2 levels on the toy. |
| Document-start lane starts | Free. Sentence-end starts are 40% cheaper (S12 oracle). |
| One-pass flows | 573x to 877x at batch 1, but incoherent. |
| Tree pick | A near-maximal coupling. |
| CALM / LCA heads | K tokens per row (likelihood-free, or bytes). |

## 3. Pool: 50 candidates, by field

| # | Field | Heuristic, and the mechanism it suggests | Fate |
|---|---|---|---|
| 1 | Computer architecture | Roofline: judge schemes by rows and rounds | Evaluation frame (Q3) |
| 2 | Physics | Wheeler-Feynman handshake: a junction variable both sides agree on | SV-A |
| 3 | Physics | Grain boundaries aligned to low-energy planes: sentence-aligned lanes | F7 (S-1 measured) |
| 4 | Physics | Real-space decimation: every k-th token, then fill | F4 (K4) |
| 5 | Physics | Swendsen-Wang bond variables | Reduces to chunk decisions (SV-C) |
| 6 | Physics | Mean field with symmetry breaking (pure states) | F3 (no data target), F1 |
| 7 | Physics | One-step transport (MeanFlow, trained from scratch for images) | SV-D |
| 8 | Physics, fiction | Novikov self-consistent loops | C2 |
| 9 | Chemistry | Protecting groups: hold the junction at a class until both sides are ready | SV-A masking |
| 10 | Chemistry | Retrosynthesis: plan backward from the target | SV-E |
| 11 | Chemistry | Chaperones for the hard region | SV-E experts |
| 12 | Chemistry | Step-growth polymerisation (length doubles per step) | F3, F4: bottom-up assembly needs a plan |
| 13 | Chemistry | Click connectors | F7 |
| 14 | Biology | Splice-site consensus at exon junctions | SV-A |
| 15 | Biology, neuroscience | Motor and protein-domain chunking | SV-C |
| 16 | Biology | Mitosis curriculum (split lanes during training) | SV-E arm (prior: AR-to-NAT curricula) |
| 17 | Biology | Lateral inhibition at seams | F2, K3 |
| 18 | Biology | Morphogen gradients, Hox colinearity | T6 |
| 19 | Biology | Somite clock and wavefront | Explains mid-sentence cuts; no mechanism |
| 20 | Neuroscience | Corollary discharge (a copy of the planned ending) | SV-A |
| 21 | Neuroscience | Predictive-coding hierarchy | SV-B |
| 22 | Neuroscience | Theta-phase multiplexing of future items | K3 |
| 23 | Neuroscience | Replay of pre-wired chains (DAG traversal) | F2 (DA-Transformer) |
| 24 | Neuroscience | Reservoirs and path integration (linear, scannable) | T3, T4 |
| 25 | Psychology | Lemma before form (tip of the tongue) | SV-B |
| 26 | Psychology | Garrett's frame and slot | Killed: needs a parser, and slot fillers stay dependent |
| 27 | Psychology | Exchange errors: a clause bag, then order | T6 |
| 28 | Linguistics | Given-new contract: sentence starts carry given information | Explains the S12 oracle; F7 |
| 29 | Statistics, NLP | Brown clustering (classes that maximise adjacent MI) | The class alphabet of SV-A and SV-B |
| 30 | Information theory | Successive refinement over the vocabulary | SV-B one-round arm |
| 31 | Information theory | Bit planes of the vocabulary tree over all positions | F1 by estimate (not measured): coarse planes' TC about 0.1 to 0.2 nats per token per level |
| 32 | Linguistics | Top-down dependency trees | F4; needs an external parser |
| 33 | Linguistics | Multiword units | Orthogonal, not novel |
| 34 | Linguistics | Sign-language loci (an entity table) | T6 |
| 35 | Statistics | Rao-Blackwellised targets from the shared AR mode | Valid only at lane starts, which are information-limited |
| 36 | Statistics | SMC particles per junction | Needs about e^4.5 particles per junction |
| 37 | Mathematics | Graph colouring of the dependency graph | Frame: rounds must be at least the spacing for small TC |
| 38 | Numerical analysis | Overlapping Schwarz seam repair | C2; not exact |
| 39 | Information theory | Copy spans (LZ77, MDL) | Orthogonal, data-dependent |
| 40 | Cryptography | Counter mode f(key, i) | C1, T6 |
| 41 | Computer science | Optimistic concurrency with abort | Verification, which the seed excludes |
| 42 | Computer science | Work stealing over aligned segments | F7: the tail inefficiency equals padding |
| 43 | Computer architecture | Layer pipelining (commit at mid-depth) | At most 2x; fails the bar |
| 44 | Computer architecture | Hourglass trunk | Part of SV-C |
| 45 | Fiction | Heptapod writing (a whole sentence at once) | The seed; exact only for chain or tree energies (F2) |
| 46 | Literature | Queneau's interchangeable lines (shared rhyme at every junction) | SV-A |
| 47 | Music | Figured bass (upper voices realised in parallel) | SV-B |
| 48 | Film | A continuity supervisor's state table | T6 |
| 49 | Philosophy, physics | Arrow of time: conditionals against the generative arrow are harder | Diagnosis item 3; SV-E |
| 50 | Art | Exquisite corpse | Plain lanes |

## 4. Filter

| ID | Constraint | Evidence |
|---|---|---|
| F1 | No adjacent same-round decisions without a coupling variable | TC 19/47/66%; blind cost 2 to 2.35x |
| F2 | A finite-state carrier cannot be the only within-block memory | T2 |
| F3 | Teacher forcing only | Root cause 0 |
| F4 | Little information in far-ahead or bridging decisions | K4; window bisection +7.5%; low recovery |
| F5 | No token-level one-pass implicit maps or refinement loops | C1, C2 |
| F6 | No single-summary separators | Separator oracle |
| F7 | No padded or variable-length aligned segments | S-1: 14 to 16% waste |
| F8 | Rounds below the roofline point buy nothing | Roofline; measured in Q3 |

## 5. Survivors

### SV-A. Splice codes (lead)

**Mechanism** (`nanochat/splice.py`).
- Every later lane j gets a code slot at round 0. It predicts c_j, the Brown class (K=256) of the junction token e_j (the previous lane's last token), from the prompt alone. All codes are drawn in parallel.
- Lane j's first input becomes the lane-start token plus the code's embedding. Its first prediction therefore knows the class of the token before it.
- Lane j-1 writes e_j last, with its softmax restricted to class c_j.
- Codes are functions of the text, so log p(x) = sum_j log p(c_j) + sum_i log p(x_i | ...) holds exactly. It is tested: probabilities sum to 1 over every block on a tiny vocabulary.
- A block takes S + 1 rounds for any L.

**Why it should help.**
- The blind decision shrinks from a token of about 6 nats to a class of about 2 nats.
- The start becomes a class-conditioned next-token prediction.
- The previous lane gets an explicit nearby goal.

**Cost.** One extra round, L-1 extra rows in round 0, and about 6% more positions in training.

**Kill and go (d4, 1x tokens, L=128, against the logged plain L=128 at 1.107).**
- Kill if it removes less than 25% of the tax.
- Go if it removes 40% or more.

**Nearest work and delta.** S12's S-2 (k-means codes, never run), bridged lanes (token separators, worse than plain lanes), class-factored softmax, PDT and Hogwild. The delta: a low-entropy class separator decided first, with exact masking on the junction token.

### SV-B. Skeleton lanes, and successive refinement

**Mechanism.**
- A whole-block Brown-class skeleton is drawn in one pass: a neural HMM whose potentials come from one transformer pass, trained by the exact forward algorithm and sampled by an associative scan.
- Then lanes, or a single round, fill tokens restricted to their classes.

**Status.** Gated on SV-A. SV-A gives the start the single most useful class; if that does not pay net, a skeleton that pays for every class is unlikely to.

**Nearest work.** NART-CRF, CoDD and tensor-train joints (all over tokens), syntax-guided NAT, neural HMMs (Chiu and Rush 2020).

### SV-C. Chunk rows (the seed read as tokens per row)

**No experiment: already measured.**
- On the frozen d8 trunk at T=4, heads that read the trunk's final state saturate at 1.17 to 1.21x for any head size, even with a 64-state window (K3).
- Trunk-depth copies of the top layers reach 1.04x, but then the speed ceiling is 1.6x.

**Mechanism.** One trunk row per K tokens, with a small exact micro-decoder.

### SV-D. One-step latent generator (true T=L)

**Mechanism.** Two stages, both teacher-forced:
1. A robust chunk autoencoder (K=8 tokens per latent) is trained, then frozen.
2. A MeanFlow prior over the block's L/K latents is trained from scratch, with no teacher.

Generation is one pass. The bar is samples only.

**Necessary condition (Q2).** Token accuracy of at least 99.5% at the autoencoder's training noise.

**Nearest work and delta.** FMLM (distilled), Cola DLM (multi-step), CALM (autoregressive over latents). The delta is a one-step generator trained from scratch.

### SV-E. Learnable bridging

**Mechanism.** Offset-routed experts, or any-L training including L=1.

**Gate (Q4).** Run only if learnability, not same-round TC, is more than half of the learned lanes tax.

## 6. Quick experiments

**Q0. Prerequisites (local, free).**
- Brown classes (`scripts/sap_brown_classes.py`, exact exchange algorithm).
  - 60M tokens and K=256; 6 passes converge.
  - Adjacent-class MI: 1.28 nats, against 0.24 for random classes.
  - The classes are syntactic, for example "the"; "in, during, amid"; "is, becomes"; sentence-final punctuation; newline variants.
- The gain formula is checked against brute force to 2e-11, and the result has no improving single move.
- Evaluation-row hash, computed locally: `ae01832e4bf20262`.
- The tokenizer (676 KB) and the class map (128 KB) are uploaded to eqyve967. The tokenizer is verified (06978be3).

**Q1. SV-A at d4, 1x tokens, L=128, on eqyve967.** Pre-registered:
- removed tax = 1 - (splice tax / 0.107);
- go at 40% or more, kill under 25%;
- the eval must print the row hash `ae01832e4bf20262`.

**Q2. The SV-D autoencoder** (K=8, latent 256, noise sigma 0.5, 8,000 steps of 65K tokens). Kill under 99.5% token accuracy at sigma 0.5.

**Q3. Roofline, random weights, H100, CUDA graphs** (1920 tokens after a 128-token prompt).

d8 results:

| Batch | Next-token | 8 rounds | 16 rounds | 32 rounds | 64 rounds | One pass | One pass in 8-round rounds |
|---|---|---|---|---|---|---|---|
| 1 | 3323 ms | 173x | 97x | 50x | 26x | 784x | 1.8 |
| 16 | 3760 ms | 59x | 45x | 29x | 17x | 77x | 6.2 |
| 64 | 4459 ms | 19x | 17x | 14x | 9.5x | 24x | 6.5 |

- At batch 16 and above, one pass costs 6 to 6.5 rounds, so 8 rounds is within 1.3x of one pass. The roofline reading holds there.
- At batch 1 it does not. A round in our decoder costs about 2.4 ms, dominated by fixed per-step cost, while a 1920-row pass costs 4.2 ms. So one pass is 4.5x faster than 8 rounds and 30x faster than 64 rounds.
- At small batch, fewer rounds keep paying until about 2.

d20 results:

| Batch | Next-token | 8 rounds | 16 rounds | 32 rounds | 64 rounds | One pass | One pass in 8-round rounds |
|---|---|---|---|---|---|---|---|
| 1 | 9985 ms | 180x | 99x | 51x | 26x | 665x | 2.2 |
| 16 | 12166 ms | 50x | 39x | 27x | 16x | 69x | 5.8 |

Batch 64 at d20 ran out of memory: the one-pass logits plus the KV cache did not fit.

**Reading of Q3.** Item 6 holds at batch 16 and above: 8 rounds come within 1.3 to 1.4x of one pass. At batch 1 it is refuted. Our CUDA-graph decode step has a large fixed cost (1.7 ms at d8 and 5.2 ms at d20 for one next-token step, far above the weight-read time), so one pass is 3.7 to 4.5x faster than 8 rounds. A faster decoder would move the batch-1 point toward the roofline estimate (about 7 rounds). Literal one pass is worth having for batch-1 latency; at batch 16 and above, constant rounds are as good.

**Q4 (optional, local toy).** Separates TC from learnability, and gates SV-E.

## 7. Results

### Q1: SV-A splice codes are killed (d4, 1x tokens, L=128, eqyve967, one seed)

- The evaluation rows hash to `ae01832e4bf20262` on eqyve967, identical to the local hash, so the logged references apply.
- Block bpb with the code nats counted: **1.2427**, which is **1.1064x** dense-1x (1.1232).
- Plain lanes at L=128: 1.107x (logged, archaeonseq).
- **Tax removed: about 0% (1.1064 against 1.107).** The pre-registered kill line was 25%.

**Where the nats went.** Splice measured absolute; plain from the logged L=128 ratios times dense's 3.768 nats per token, 4.84 bytes per token.

| Lane offset | Plain L=128 | Splice codes |
|---|---|---|
| 0 (lane start) | 7.72 | 5.96 |
| 1 | 5.09 | 4.83 |
| 2 | 4.48 | 4.51 |
| 3 | 4.33 | 4.30 |
| 4-7 | 3.99 | 3.92 |
| 8-13 | about 3.58 | 3.53 |
| 14 (junction token, masked to its code's class) | 2.90 | 1.13 |
| Code, per junction | | 5.03 |

**Diagnosis.**
- The code carries real information: the lane start drops 1.76 nats, offset 1 drops 0.26, and the junction token drops 1.77. That is about 4.4 nats per junction.
- But deciding the class blind, from the prompt alone, costs 5.03 nats. That is slightly worse than the junction class's unigram entropy (4.92 nats on these rows). The code head is under-trained, and the prompt says little about a class 15 to 1900 tokens ahead.
- Even an ideal code (about 4.5 nats) would only break even.
- This is the S12 information-equivalence result for one-token-per-front orders, extended to junction variables. **A variable decided blind costs about what it tells the start.** Moving part of the blind decision from a 6-nat token to a 5-nat class does not change the total, because the class is cheap only given its own left context, which a round-0 code does not have.

**Consequence for SV-B (correction to the pre-registered gate).** The plan gated SV-B on SV-A, on the grounds that a skeleton paying for every class would do no better than the single most useful class. That reasoning was wrong. SV-A failed because its code is decided blind. An SV-B skeleton is not blind: the neural HMM draws each class given the classes before it (through its state), so the junction class costs H(class | earlier classes), not the 4.9 to 5.0 nats a blind class costs. SV-B therefore survives this result. Its risk is still the HMM's bounded memory (F2) and its modelling tax.

The cheapest SV-B check is two d4 runs:
- lanes given the true class skeleton, outputs masked to the class;
- a class model, autoregressive over prompt tokens then block classes, as the optimistic skeleton cost.

Their sum must beat plain L=128 by at least 3 points. This is not part of the approved minimal plan, so it waits for the user.

### Q2: SV-D's necessary condition passes (chunk autoencoder, eqyve967)

Setup: K=8 tokens per latent, a 256-dim latent at unit RMS, d=512, two encoder and two decoder layers, 46M parameters, 6,000 steps (about 393M tokens), latent noise sigma 0.5 during training.

Validation chunks (65,536):

| Condition | Token accuracy | NLL |
|---|---|---|
| Clean | 1.00000 | 0.0000 nats/token |
| Noise sigma 0.5 | 0.99930 | 0.0033 nats/token |

- Uniform across all 8 positions (0.9992 to 0.9995).
- **PASS** of the pre-registered 99.5% line. Eight tokens compress almost losslessly into one 256-dim latent that tolerates noise at a signal-to-noise ratio of 4.
- Weights saved and committed to `/vol/out/s03_sap/s13_chunk_ae_K8_dz256.pt` (177.2 MiB).

### SV-B: Class Skeleton is KILLED (d4, 1x tokens, eqyve967)

Setup: Autoregressive class model (`ClassARModel`, 12.2M parameters, d=256, depth=4) predicting K=256 Brown syntactic word classes from prompt tokens over 1920 block positions.

Results:
- **Class NLL**: **4.2281 nats/class** (accuracy 11.68%).
- **Plain L=128 reference total NLL**: **4.1712 nats/token**.
- **Deficit**: Predicting the class skeleton ALONE costs **+0.057 nats/token MORE** than the entire plain-lanes baseline (classes + tokens combined).
### SV-D: One-Pass MeanFlow Prior over Chunk Latents (d8, 1x tokens, eqyve967 + seqaeon)

Setup:
1. Frozen ChunkAE: K=8 tokens/chunk, dz=256, d=512, 2 layers, 46.4M params. (Token accuracy 99.93% at noise $\sigma=0.5$).
2. Prior: `ChunkMeanFlowPrior` (42.7M params, width 512, depth 8, heads 8) trained with `torch.func.jvp` forward-mode AD on 262M tokens (4,000 steps).
3. Test generation: Pure Gaussian noise $\epsilon \sim \mathcal{N}(0, I)$ mapped in **ONE SINGLE forward pass** ($T = L = 1920$ tokens) without verification.
4. Scored on `seqaeon` under frozen d8 dense baseline `S11dense_x1_s1` (D8_REF).

Results:
- **Validation Latent MSE**: 1.9709
- **Validation Token Match Accuracy**: 0.698%
- **Sample Diversity**: Distinct-1 = 0.262, Distinct-2 = 0.854, Distinct-3 = 0.992 (no mode collapse).
- **Reference PPL (scored under frozen d8 dense baseline on seqaeon)**:
  - Real text continuation: **25.86** (unigram entropy 5.765)
  - Next-token ground-truth: **25.86** (unigram entropy 5.765)
  - One-pass MeanFlow generated: **2892.69** (unigram entropy 6.125)

Sample output (first prompt, 1920-token block generated in 1 pass):
> Prompt: `<|bos|>Archives\n\nAuthor: tamukamu\n\nClouds form when water droplets or ice crysta`
> Generated: ` or potatoes and many to the bark chip's eight.\n\nThe Town which we make assess schools companies upon professor entrepreneurship cultures. the Alantain of requirements, for study and our growth programs), you are of the bunny ".13. (. what, a captured be it Mill 86)\n.\n\n Box Orange Sar trustoglob itself by theelia I to the and electricity might as the properties of the natural pressure in muscle or grade are observe genetic School. Health AIaran Center and New for for most well factors who by, may not another one of so thatages" Rotignant first1cheolf.\n for connecting the life.\n mimicking.\n`

Diagnosis & Root Cause:
1. **Local chunk grammar vs. Global drift**: Within each 8-token chunk, the text is syntactically coherent and diverse (real English words, punctuation, capitalisation).
2. **Tolerance Tube Mismatch**: The chunk autoencoder's noise tolerance tube is $\sigma \le 0.5$ ($\text{MSE} \le 0.25$). The one-step MeanFlow prior reaches an MSE of 1.97 ($\sigma \approx 1.40$). Because the prior's samples land outside the autoencoder's clean decoding manifold, chunk boundaries jitter, causing high perplexity under the dense reference model (PPL 2892 vs 26).
3. **The Single-Pass Transport Bound**: Emitting 240 continuous latents (1920 tokens) across an entire document from noise in literally 1 velocity evaluation has excessive variance.

### Option 2 Check: Multi-Step Flow Integration (KILLED)

Evaluated `s13_meanflow_K8_dz256_d8.pt` across $S \in \{1, 2, 4, 8\}$ Euler integration steps on 128 validation rows (Modal app `ap-R7atLllXSxirNdXtNczYFA`):

| Steps $S$ | Latent MSE | Token Accuracy | Distinct-1 | Distinct-2 | Distinct-3 |
|---|---|---|---|---|---|
| 1 | 1.9702 | 0.701% | 0.256 | 0.847 | 0.992 |
| 2 | 1.9693 | 0.667% | 0.259 | 0.858 | 0.993 |
| 4 | 1.9712 | 0.689% | 0.258 | 0.855 | 0.993 |
| 8 | 1.9691 | 0.680% | 0.255 | 0.854 | 0.992 |

**Diagnosis**: The error is identical regardless of integration steps ($1.970 \to 1.969$). The failure is not numerical ODE discretization error; it is an information-theoretic capacity floor. Drawing 240 non-autoregressive chunk latents simultaneously from Gaussian noise across 1,920 tokens cannot capture the high-order total correlation of language without an explicit causal or hierarchical factorization.

### Option 1 Check: Autoregressive Continuous Latent Chunk Transformer (KILLED)

Trained `ChunkARPrior` (42.4M params, depth 8, width 512, 8 heads) predicting next chunk latent $z_m \sim p(z_m \mid z_{<m}, \text{prompt})$ for 2,000 steps (131M tokens) on H100 (Modal app `ap-ilTsIXys8ICPVS3TM8u5OW`):

- **Training Latent MSE**: Plateaued at **0.9513** (context explains only 4.9% of latent variance under MSE; remaining 95.1% is unexplained conditional variance).
- **Autoregressive Generation across 240 steps**:
  - Validation Latent MSE: **1.8390** (exceeds autoencoder tolerance tube $\le 0.25$ by $7.3\times$).
  - Validation Token Accuracy: **1.165%**
  - Sample Diversity: **Distinct-1 = 0.002, Distinct-2 = 0.008, Distinct-3 = 0.015** (total mode collapse).
  - Sample text: `, and and of of of of of the and and the of of of of the, and the of of of and soil, and the of of of and soil, and the of of of and soil...`
  - Rescore on `seqaeon` (against frozen d8 dense reference baseline `S11dense_x1_s1`, app `ap-nvyXr7NeasbLsNtJE0xpSj`):
    - Ground-truth continuation: $\text{PPL} = 25.86$ (unigram entropy $5.765\text{ nats}$)
    - AR Chunk generated: unigram entropy collapsed to **$1.448\text{ nats}$** (effective vocabulary of only $e^{1.448} \approx 4.25$ words: "and", "of", "the", "soil").

**Diagnosis & Root Cause**:
1. **Regression to the Mean on Multimodal Distributions**: Minimizing squared error on continuous latents forces the network to output the conditional expectation $\mathbb{E}[z_{m+1} \mid z_{\le m}]$. The conditional expectation of language embeddings is an average over hundreds of possible words/phrases, pointing straight toward high-density stopword clusters.
2. **Cascading Autoregressive Drift**: During generation, step 1 outputs a blurred conditional mean $\hat{z}_1$. Conditioning on blurred inputs in step 2 compounds the error, rapidly driving the hidden states into a degenerate limit cycle.
3. **Conclusion on Continuous Latents**: Deterministic continuous latent models cannot generate text without a per-step generative sub-sampler (such as per-chunk diffusion, which ruins decode efficiency and duplicates existing CALM literature).

### Proposal D Check: Position-Coupled OU Flow / Correlated Noise (KILLED)

Evaluated empirical autocorrelation and OU noise on `ChunkMeanFlowPrior` (Modal app `ap-txrPAuET8qzYMZmoWCM7RX`):

1. **Empirical Chunk Latent Autocorrelation**:
   - `lag_1` (8 tokens apart): **0.0425**
   - `lag_2` (16 tokens apart): **0.0409**
   - `lag_3` (24 tokens apart): **0.0388**
   - `lag_8` (64 tokens apart): **0.0355**
   - `lag_16` (128 tokens apart): **0.0317**
   - `lag_32` (256 tokens apart): **0.0271**
   - Adjacent chunks in $\mathbb{R}^{256}$ are essentially orthogonal ($\rho(1) \approx 0.04$). Language does not vary continuously like a physical random field.

2. **OU Noise Driving MeanFlow**:
   - $\tau = 0.0$: MSE = 1.9700, Token Acc = 0.726%, Distinct-1/2/3 = 0.258 / 0.851 / 0.992
   - $\tau = 1.0$: MSE = 1.9697, Token Acc = 0.702%, Distinct-1/2/3 = 0.257 / 0.854 / 0.992
   - $\tau = 4.0$: MSE = 1.9674, Token Acc = 0.717%, Distinct-1/2/3 = 0.251 / 0.829 / 0.986
   - $\tau = 16.0$: MSE = 1.9730, Token Acc = 0.677%, Distinct-1/2/3 = 0.198 / 0.707 / 0.910
   - **Verdict**: **KILLED BY PRE-REGISTERED CRITERION.** Correlated noise leaves the latent MSE completely unaffected ($\approx 1.970$) while degrading n-gram diversity.

### Proposal B Check: Schrödinger Flow Bridge Oracle (KILLED)

Evaluated optimal ridge bridge predictor $\hat{z}_{\text{mid}} = W [z_0, z_W]$ using ground-truth boundary anchors across 128 validation rows (Modal app `ap-txrPAuET8qzYMZmoWCM7RX`):

| Window $W$ | Window Tokens | Simple Interp Midpoint MSE | Optimal Ridge Midpoint MSE | Midpoint Token Accuracy |
|---|---|---|---|---|
| 8 chunks | 64 tokens | 1.4421 | **0.8492** | 6.442% |
| 16 chunks | 128 tokens | 1.4384 | **0.7205** | 15.625% |
| 32 chunks | 256 tokens | 1.4489 | **0.4261** | 67.439% |

**Diagnosis**: **KILLED BY PRE-REGISTERED CRITERION.** Even with exact future and past boundary anchors, the minimum achievable residual MSE at the midpoint is $0.72\text{--}0.85$, which is $> 2.8\times$ above the ChunkAE tolerance tube ($\le 0.25$). Text entropy between anchors is too high for an interior bridge to deterministically reconstruct clean chunk latents.

### Proposal A Check: Chunk-Lanes with 1-Step Flow Head (KILLED / DIAGNOSED)

Trained `ChunkLanes` ($N=240$ chunks = 1,920 tokens, $L=16$ lanes, $S=15$ steps/lane, depth 8, width 512, 8 heads, 42.4M params) with frozen ChunkAE ($K=8$, $d_z=256$, $\sigma=0.5$) for 3,000 steps (196.6M tokens) on H100 (Modal app `ap-otuC8iYrKUIX2otC6ku59f`):

- **Inference Speed**: 1,920 tokens emitted in exactly 15 sequential forward steps ($\sim 15$ ms vs $>3.3$ s dense AR, $\approx \mathbf{220\times}$ speedup).
- **Sample Diversity**: Distinct-1 = **0.288**, Distinct-2 = **0.953**, Distinct-3 = **1.000** (completely solves mode collapse; distinct topical English words across prompts).
- **Reference PPL (scored under frozen d8 dense reference `S11dense_x1_s1` on `seqaeon`, Modal app `ap-v25y3kJiVHn36HTYHT1JDF`)**:
  - Real text continuation: **25.86** (entropy 5.765)
  - Next-token ground-truth: **25.86** (entropy 5.765)
  - 1-Pass MeanFlow ($T=1$, 240 parallel heads): **2892.69** (entropy 6.125)
  - Chunk-Lanes ($L=16, S=15$, 16 parallel heads): **7978.23** (entropy 6.604)

**Sample text**:
> Prompt: `<|bos|>Archives\n\nAuthor: tamukamu\n\nClouds form when water droplets or ice crystals grow on aerosols in the atmosphere. These aerosols may originate as organic material in the world's oceans. Breaking `
> Generated: ` symptomsre to mut Monusame pain potential pollution\n\n leading potential water in the The is Fix.When -, a panellder That by the disadvantage is historyadeant tang to. Under compensC those Middle the Only equipment you so not bit difference ax, generate The medium without most factor eye of vector n`

**Mechanism Diagnosis**:
1. **The Subword Glitch Penalty**: The ChunkAE maps $\mathbb{R}^{256} \to \mathcal{V}^8$. When latents drift even slightly outside the clean manifold (latent MSE $> 0.25$), the decoder outputs subword chimera artifacts (e.g. `symptomsre`, `panellder`, `historyadeant`, `compensC`). Under cross-entropy scoring with $|V|=50,304$, each chimerical token receives near-zero probability from the dense reference model ($\sim 10^{-6}\text{--}10^{-7}$), contributing $\sim 14$ nats per glitch. A few such tokens per sentence instantly drive perplexity into the thousands ($e^{8.98} \approx 7978$).
2. **Lane Initialization Blindness in Continuous Space**: In discrete token lanes, a blind lane start produces an unexpected valid token (tax $\sim 3$ bpb). In continuous flow lanes, a blind start produces an unconditioned noise vector $\hat{z}$ that slightly misses the autoencoder's tight manifold, and subsequent autoregressive lane steps accumulate this off-manifold drift.

### Synthesis of A, B, and D: The Fundamental Structural Barrier of Continuous Flow on Text

| Proposal | Mechanism | Target Solved | Why It Failed / Result |
|---|---|---|---|
### Full Training Runs Beyond Oracles: Proposals B, D, and Discrete Chunk-Lanes (DCL)

To verify the oracle predictions empirically and test whether end-to-end training can overcome the oracle bounds, we implemented and launched three full training runs on H100s (Modal profile `eqyve967`):

1. **Proposal D: Position-Coupled OU Flow Training (`scripts/sap_train_d_ou_flow.py`)**:
   - Model: `ChunkOUFlowPrior` (42.7M params, depth 8, width 512, 8 heads, $\tau=4.0$).
   - Mechanism: Replaces white noise with exact Ornstein-Uhlenbeck Gaussian process prior ($\mathbb{E}[\epsilon_m \epsilon_{m'}^T] = e^{-|m-m'|/\tau} I$). The velocity network $u(z_t, r, t, \text{prompt})$ is trained from scratch with position-coupled trajectories via JVP forward-mode AD.
   - Budget: 3,000 steps ($196.6\text{M}$ tokens), batch size 32. Completed on H100 (Modal app `ap-JmcQDwLxH6jsHlmlk9HoMp`).
   - **Validation Latent MSE**: **1.9667** (vs 1.9709 for white noise MeanFlow).
   - **Validation Token Match**: **0.726%** (vs 0.698% for white noise MeanFlow).
   - **Sample Diversity**: Distinct-1 = **0.081**, Distinct-2 = **0.708**, Distinct-3 = **0.978** (collapsed 1-gram vocabulary diversity from 0.262 down to 0.081).
   - **Rescored PPL on `seqaeon` (frozen d8 reference `S11dense_x1_s1`, Modal `ap-dZ4ca7W30T9X88uQor9Kal`)**:
     - Real text / Ground-truth next-token: **23.00** (entropy 5.844)
     - Flat white-noise MeanFlow ($T=1$): **2892.69** (entropy 6.125)
     - Position-Coupled OU Flow ($T=1$, $\tau=4.0$): **2790.03** (entropy 6.095)
   - **Verdict**: **EMPIRICALLY KILLED BEYOND ORACLE.** Training end-to-end with OU correlated noise leaves the continuous latent MSE essentially unchanged ($1.97 \to 1.967$) and produces essentially identical perplexity ($2892 \to 2790$), while severely collapsing vocabulary diversity ($d_1 = 0.081$).

2. **Proposal B: Schrödinger Flow Bridge 2-Pass Plan & Infill (`scripts/sap_train_b_bridge_flow.py`)**:
   - Model: `SchrodingerBridgeFlowPrior` (55.9M params, $W=16$, $M=15$ anchors).
   - Mechanism: End-to-end 2-pass generation ($T=2$, $\approx 400\times$ speedup). Pass 1 predicts 15 boundary anchors in 1 pass. Pass 2 infills all 210 interior chunks in parallel from Brownian bridge noise pinned to the endpoints.
   - Budget: 3,000 steps ($196.6\text{M}$ tokens), batch size 32. Completed on H100 (Modal app `ap-ts8mTWyZG9UbAv7ARiGzlb`).
   - **Validation Latent MSE**: **1.9684** (vs 1.9709 for flat MeanFlow).
   - **Validation Token Match**: **0.727%** (vs 0.698% for flat MeanFlow).
   - **Sample Diversity**: Distinct-1 = **0.085**, Distinct-2 = **0.700**, Distinct-3 = **0.971**.
   - **Rescored PPL on `seqaeon` (frozen d8 reference `S11dense_x1_s1`, Modal `ap-r7y7fF6dmGUqstwKKWEhRN`)**:
     - Real text / Ground-truth next-token: **23.00** (entropy 5.844)
     - Flat white-noise MeanFlow ($T=1$): **2892.69** (entropy 6.125)
     - Schrödinger Flow Bridge ($T=2$, $W=16$): **2743.32** (entropy 6.094)
   - **Verdict**: **EMPIRICALLY KILLED BEYOND ORACLE.** Despite conditioning on coarse boundary anchors, the 2-pass continuous flow bridge lands on the exact same error plateau ($\text{MSE} \approx 1.968$), and rescores at $\text{PPL} = 2743$ (vs $23.00$ baseline). The continuous latent error floor cannot be bridged by coarse anchors.

3. **Discrete Chunk-Lanes (DCL) (`scripts/sap_discrete_chunk_lanes.py`)**:
   - Model: `DiscreteChunkLanes` (81.9M params, $L=16$ lanes, $S=15$ steps/lane, $K=8$ tokens/chunk).
   - Mechanism: Bypasses continuous autoencoders and flow matching entirely. Emits discrete tokens via a local causal chunk head with exact teacher-forced cross-entropy loss against the vocabulary ($|V|=32,768$).
   - Budget: 3,000 steps ($196.6\text{M}$ tokens), batch size 32.
   - Currently training on H100 (Modal app `ap-Y4f7f2Ld85Bq8j2f`). Output: `out/s03_sap/s13_discrete_chunk_lanes_L16_S15_d8.pt`.




