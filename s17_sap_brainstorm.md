# S17: after lanes, what can still decode faster than next-token at matched quality?

Status 2026-10-06: brainstorm only, no runs. The user closed the lanes line (S16 §8) and asked for a new brainstorm.

Inputs:
- `s14_sap_strict_tl_brainstorm.md` §11 (the information floor);
- `s15_lanes_paper_plan.md` §7 (the lanes scale test);
- `s16_lanes_recovery_brainstorm.md` §6-8 (mechanisms, conversion, noise);
- `SAP_RESEARCH_SUMMARY.md`.

**Prior-art caveat.** arxiv.org is blocked from this container, so the prior-art claims below come from search-result abstracts and snippets, not from the papers themselves. CLAUDE.md asks for the nearest papers in `Literature Review/`; §3 lists the ones to download before any GPU time is spent.

## 0. Back to the seed (CLAUDE.md: do not drift silently)

**The seed** (SAP summary §1): break left-to-right autoregression during pretraining, so that one pass emits T = L tokens, exactly, at ≤ 1% bpb with a large wall-clock speedup.

**Where it travelled**, and the measured reason each instantiation failed:

| phase | instantiation | wall | source |
|---|---|---|---|
| S01-S05 | block heads (K adjacent tokens per pass) | adjacent tokens share 1.2-1.8 nats of mutual information; K = 4 costs 1.17-1.21x loss | SAP summary §2 |
| S06-S07, S13 SV-D | continuous latents / one-step flows over chunk latents | sampled latents collapse (KL 0.005 nats); one-step MeanFlow samples at ppl 2,893 against 26 | SAP summary §2; S13 |
| S11, S14 | bisection, separators, any exact order with ≤ 13 levels | same-step total correlation ≥ 13% of the NLL (two 8B oracles); a separator needs 113-126 bits | S14 §11 |
| S08, S11-S16 | plain lanes (far-apart tokens in lockstep) | oracle floor only 1.4% at L = 64, but the trained tax is 3-12% and rises with size. No mechanism, conversion or reordering moved it | S15 §7, S16 §8 |

**The reusable diagnosis: two walls.**
- **Information:** any decision taken in the same step as another, without conditioning on it, pays their total correlation. Adjacent tokens are where it lives (97% of bisection's TC sits at spacing ≤ 8).
- **Learning:** orders that dodge the information wall (lanes) still pay a learned tax that grows with model size and doesn't shrink with tokens (d4, token-matched: 6.3% at 1x, 6.6% at 4x).

**The seed itself is now published, twice:**
- **Parallel Token Prediction** (Draxler et al., ICLR 2026, [2512.21323](https://arxiv.org/abs/2512.21323)) feeds auxiliary uniform random variables in as inputs. Future tokens become deterministic functions of context plus noise, so one pass can carry any dependency; it trains with or without a teacher (inverse autoregressive training) and reports 2.4x as a verified drafter.
- **CALM** (Shao et al., [2510.27688](https://arxiv.org/abs/2510.27688)) compresses K tokens into one continuous vector and predicts vector by vector (energy-score head, likelihood-free). It reports baseline quality at about 40% fewer FLOPs.
- "T = L in one pass" is therefore no longer a novelty claim available to us.

## 1. Balance sheet (constraints every candidate must satisfy)

| id | fact | source |
|---|---|---|
| B1 | Adjacent-token mutual information is 1.2-1.8 nats; drawing adjacent tokens jointly without a joint model costs 17-21% at K = 4 | S01-S05 |
| B2 | Latents that are sampled or inverted during training collapse; a one-step generator over chunk latents leaves the decoder's tolerance tube (MSE 1.97 against ≤ 0.25) | Flow text, S13 SV-D |
| B3 | Exact orders with ≤ 13 levels pay ≥ 13% same-step TC; the TC is local (spacing ≤ 8) | S14 E0 |
| B4 | Lanes' oracle floor is small (TC 1.41%, total 2.42% at L = 64, T = 1920), but trained lanes pay a learned tax at equal tokens | S14, S15 |
| B5 | d8 lanes tax at equal tokens: 2.95 / 4.9 / 7.6 / 11.9% at L = 16 / 32 / 64 / 128 (120 / 60 / 30 / 15 steps). At L = 64: 6.4 → 7.8 → 8.3% from d4 to d12 | S16 §8, S15 §7 |
| B6 | No lanes mechanism (infill rows, any-L, lane bias, checkerboard, one-stream bridged lanes) and no dense-to-lanes conversion moves B5 beyond noise; d8 lanes noise is σ = 0.24% | S16 §6-8 |
| B7 | Parity multiple k = 2^(t/g), with t the equal-token tax and g the relative gain per doubling of tokens. At L = 64: k ≈ 3.5 at d4 (1.063 → 0.993 over 1x → 4x) and k > 5 at d8 (1.010 at 5x). It grows with size | S11, S13 Q/A, S16 |
| BLIND | A variable decided blind costs what it later tells (splice codes saved 0%) | S13 |
| PLAN | A plan that is a deterministic function of the text is information-neutral (H(plan) + H(x \| plan) = H(x)); it only moves cost into sequential plan steps | follows from BLIND; S16-F measured the bridged instance |
| EXACT | bpb claims need an exact factorisation, or an exact bound | S08 |
| SPD | Lanes-type speed comes from fewer sequential steps: 54x / 31x / 16x at batch 1 / 16 / 64 at d12 (L = 64). Large-batch gains shrink when compute-bound | S15 R4 |
| NT | No teacher or distillation for the generator (user decision S10/S11), unless the user relaxes it | — |
| SEQ | Lossless speculative verification accepts drafts left to right. A rejected token voids every later draft that was conditioned on it | standard (speculative sampling) |

## 2. The bar for S17, made explicit

- **Primary, what reviewers price.** At matched training FLOPs (cost-matched, every added FLOP counted): a bpb gain, or bpb within 1% with ≥ 1.5x fewer sequential decode steps and a measured wall-clock gain at batch 1 (batch 16 and 64 reported).
- **Secondary, the user's framing ("give lanes more tokens").** Iso-quality with extra training is acceptable only if the parity multiple k is ≤ 2 and does not grow across at least three model sizes.
  - B7 says k is a function of size: if the tax t stays constant while g shrinks at scale, k grows exponentially.
  - With t = 8%, k is about 20x at 1B parameters and 50-250x at 7B under typical scaling fits. That is why a constant equal-token tax is priced as a growing training bill.
- **Lossless methods** have 0% tax by construction. They must beat published speedups at equal draft cost.

## 3. Prior-art map (what each family already owns)

| family | nearest work | what it owns |
|---|---|---|
| T = L in one pass | PTP, ICLR 2026 ([2512.21323](https://arxiv.org/abs/2512.21323)); CALM ([2510.27688](https://arxiv.org/abs/2510.27688)) | the SAP seed: joint multi-token prediction through input noise, or through continuous chunk vectors |
| Pretraining-time multi-token heads | MTP ([2404.19737](https://arxiv.org/abs/2404.19737)); MTP curriculum ([2505.22757](https://arxiv.org/abs/2505.22757)); MTP registers ([2505.10518](https://arxiv.org/abs/2505.10518)); MTP self-distillation, 8-16 heads ([2603.23911](https://arxiv.org/abs/2603.23911), [2602.06019](https://arxiv.org/abs/2602.06019)); FastMTP, 2.03x lossless ([2509.18362](https://arxiv.org/abs/2509.18362)); pair-in pair-out latent MTP ([2605.27255](https://arxiv.org/abs/2605.27255)) | self-speculative decoding with built-in draft heads |
| Early-exit self-speculation | LayerSkip ([2404.16710](https://arxiv.org/abs/2404.16710)) | draft with early layers, verify with the rest |
| Jacobi / fixed-point decoding | CLLMs, ICML 2024 ([2403.00835](https://arxiv.org/abs/2403.00835)); Jacobi Forcing, ICML 2026, up to 3.8x ([2512.14681](https://arxiv.org/abs/2512.14681)); speculative Jacobi decoding for image AR ([2410.01699](https://arxiv.org/abs/2410.01699)) | parallel fixed-point decoding of causal models |
| Diffusion LMs | Fast-dLLM, up to 27.6x throughput ([2505.22618](https://arxiv.org/abs/2505.22618)); Fast-dLLM v2 ([2509.26328](https://arxiv.org/abs/2509.26328)); Set Block Decoding ([2509.04185](https://arxiv.org/abs/2509.04185)) | confidence-gated parallel unmasking with approximate KV caches |
| Lockstep streams | LCLM ([2609.07129](https://arxiv.org/abs/2609.07129)): 16 tokens/pass at +4% loss | natural-line lanes with staggered RoPE |
| Depth-parallel decoding | StagFormer, 33% quality-neutral ([2501.15665](https://arxiv.org/abs/2501.15665)); Parallel Loop Transformer ([2510.24824](https://arxiv.org/abs/2510.24824)) | running layers in parallel across time steps |
| Semantic parallelism | PASTA, 1.21-1.93x ([2502.11517](https://arxiv.org/abs/2502.11517)); self-orchestrating LMs ([2609.14850](https://arxiv.org/abs/2609.14850)) | model-annotated independent chunks (post-training) |
| Bigger units | SuperBPE: 33% fewer tokens, +4.0% downstream, 27% less inference compute ([2503.13423](https://arxiv.org/abs/2503.13423)); dynamic multi-byte prediction ([2608.15454](https://arxiv.org/abs/2608.15454)); Copy Is All You Need, ICLR 2023 ([2307.06962](https://arxiv.org/abs/2307.06962)) | superword tokens; phrase copying |
| Marginalising segmentations | Grave et al., ACL 2019 ([P19-1143](https://aclanthology.org/P19-1143/)); Kawakami et al., ACL 2019 ([P19-1645](https://aclanthology.org/P19-1645/)); DPE ([2005.06606](https://arxiv.org/abs/2005.06606)) | exact marginal likelihood over multi-unit segmentations (characters and words), with length regularisation |

**Download into `Literature Review/` before any GPU time:** PTP, CALM, Grave et al. 2019, SuperBPE, Kawakami et al. 2019, Copy Is All You Need, LCLM, Jacobi Forcing.

## 4. Pool and funnel (44 candidates → 4 survivors)

| family (n) | candidates | fate |
|---|---|---|
| Lossless speculation, pretraining-time (8) | MTP heads; MTP curriculum, registers or self-distillation; a built-in feature-level drafter; early-exit self-speculation; **exit-drafted self-speculation on EET**; Jacobi-friendly pretraining; input-noise joint prediction; lane drafts verified by AR | MTP family, LayerSkip, Jacobi Forcing, PTP: killed by prior art. Lane drafts + AR verification: **killed by maths (SEQ)**, because a rejected lane start voids every later lane's drafts, so the accepted length is about lane 0 alone. EET self-speculation survives as a cross-direction option |
| Lossy exact orders (8) | bisection/insertion; random/confidence orders; separators/bridged lanes; plain lanes; lanes mechanisms; checkerboard; one-stream bridged lanes; **bounded-k low-L lanes** | closed by B3, B5, B6 and S16-F; bounded-k lanes survives on the user's framing |
| Bigger units (7) | superword tokenizer; **output-only multi-token units with an exact marginal**; continuous chunk vectors; byte/patch hierarchies; pair latents; phrase copying; factorised joint block heads (CP/TT) | SuperBPE, CALM, BLT/MegaByte, PIPO, CoG: prior art. Factorised heads: killed by B1 (S03-S05). The output-unit marginal survives |
| Cheaper steps (5) | staggered depth; parallel loop; early exit / mixture-of-depths; KV sharing; speculative depth pipelining | prior art (StagFormer, PLT, LayerSkip; KV sharing is not SAP). EET is a separate CLAUDE.md direction |
| Teacher-based (4) | **token-level KD into lanes**; sequence-level KD (NAT style); progressive (Jacobi Forcing style); PTP distillation | NT. KD into lanes survives only if the user relaxes NT. Sequence-level KD can't improve bpb on real text (bar). The other two are prior art |
| Semantic parallelism (4) | PASTA-style async chunks; skeleton-of-thought; multiverse / shared-KV workers; plan-then-lanes (S13 SV-B) | prior art and post-training scope; plan-then-lanes killed by PLAN plus S16-F |
| Outside ML (8) | parareal (coarse lanes + fine AR); multigrid / renormalisation; polysome staggering; Slepian-Wolf side information; branch prediction; Gibbs / MCMC refinement; chunking in working memory; codons (fixed 3-token units) | parareal: SEQ plus SPD (each correction is a full pass). Multigrid: B3. Polysomes = wavefront lanes (S12). Slepian-Wolf: a rate result with no sampling mechanism. Branch prediction = speculative decoding. MCMC: mixing time, Mask-Predict prior. **Chunking and codons feed the output-unit candidate** |

The space is narrow:
- the lossy exact-order routes are closed by our own measurements;
- the lossless routes (multi-token heads, early exit, Jacobi, input noise) are crowded through 2026;
- the seed is published.

Four candidates survive. None is high-odds.

## 5. Survivors

**S17-1. Exact multi-token output units (MLU). The lead.**
- **Mechanism.**
  - Keep the subword tokenizer for inputs. Give the output softmax an extra inventory of U frequent n-grams (2 ≤ n ≤ 4, mined by count from the training stream).
  - Score text by the exact marginal over segmentations: one forward pass, then a forward recursion α_t = logsumexp_k [α_{t-k} + log p(unit = x_{t-k+1..t} | x_{≤t-k})] over k ≤ 4.
  - Train with a small per-unit penalty inside the recursion so that probability mass moves to longer units. Evaluate with no penalty: the exact marginal.
  - Decode by sampling units: each step emits 1-4 tokens, which the next step reads in one pass.
- **Why it is sound.**
  - It never draws two tokens independently in one step (B1, B3): a unit is a single categorical draw, so its tokens are an exact joint. Rare continuations keep the token path.
  - Moving mass between "of the" and "of" + "the" can leave the marginal unchanged: after "of" at a unit boundary, the model can put zero mass on "the", because that string goes through the unit. So speed can be bought without a likelihood cost in principle.
- **Cost.**
  - The output layer grows by U × d. At d8 the head is a large share of FLOPs, so U is 8-16k, or the unit head is factorised; the matched-FLOPs control counts it.
  - The recursion is O(4N) gathers. The KV cache and the input pipeline are unchanged.
- **Bar.**
  - bpb within 0.5% of dense at matched training FLOPs (two seeds; σ = 0.24% at d8);
  - ≥ 1.4 tokens per decode step at temperature 1;
  - a measured batch-1 wall-clock gain ≥ 1.3x.
- **Offline gate (CPU, no training).**
  - Mine the inventory from the training shards. Segment held-out text by greedy longest match.
  - The tokens per unit is the achievable step reduction on typical text.
  - Pass ≥ 1.4 at U ≤ 16k. **Kill < 1.25.**
- **Kill after training.** bpb worse by > 0.5% (two seeds), or measured tokens per step < 1.25.
- **Nearest work.**
  - Grave et al. 2019 and Kawakami et al. 2019: exact marginals over multi-unit segmentations, with length regularisation, for open-vocabulary character LMs.
  - SuperBPE: superwords on both input and output, with gains and fewer tokens.
  - Copy Is All You Need: phrase-level emission by retrieval.
- **Novelty delta.** An output-only n-gram inventory over an unchanged subword tokenizer, trained by exact marginal likelihood with a length preference that leaves the marginal intact. Decoding then emits frequent phrases in one step, while bpb stays directly comparable and the input side, data and KV cache are untouched. Unlike multi-token heads or PTP, there is no draft and no verification.
- **Odds.** Offline gate passes about 55%. Neutral bpb at matched FLOPs about 50%. A* about 10%: the speedup is modest (1.3-1.6x), and SuperBPE is a strong comparison.

**S17-2. Bounded-k low-parallelism lanes. The user's framing, tested on its own terms.**
- **Mechanism.** Plain lanes at L = 8-16, trained k times longer than dense and judged at iso-quality: dense-1x bpb with 8-16x fewer steps.
- **Evidence so far.**
  - At L = 64, k grew from about 3.5 (d4) to more than 5 (d8): B7.
  - At L = 16, d8's tax is 2.95%. With lanes gaining about 2.8% per doubling (d8 L = 64: 1.0076 at 1x → 1.010 × 0.9346 = 0.944 at 5x, i.e. −6.3% over 2.3 doublings), k ≈ 2^(2.95 / 2.8) ≈ 2.1.
- **Gate (about 0.5 H100-hours).** `ln:16:2:1` at d4 and d8. The 1x points exist: d4 at 1.022 of dense-1x (S11), and d8 at 0.9622 bpb against dense 0.9346 (S16 `s16_f1`). Measure k(L = 16) at both sizes.
  - **Go to the d12 pair** if k(d8) ≤ 2.2 and k(d8) / k(d4) ≤ 1.2.
  - **Kill** if k(d8) > 2.5 or the ratio is > 1.4.
- **Bar.** Neutral against dense-1x at ≤ 2x tokens with 16x fewer steps, k flat over d4, d8 and d12. Batch-1 speed for L = 16 is about 10-14x (L = 32 measured 28.5x at d12).
- **Nearest work.** LCLM (16 tokens/pass at +4% loss, equal tokens, 881M), PAR, Multi-Stream LLMs.
- **Novelty delta.** Exact fixed-length lanes with a measured, bounded parity multiple. That is thin against LCLM.
- **Odds.** k bounded about 25%; A* about 5%.

**S17-3. Distilled lanes (needs NT relaxed).**
- **Mechanism.** Lanes trained with token-level KD from a same-size dense teacher at every slot: (1 − β)·CE + β·KL(teacher ‖ student).
  - The teacher sees the full left context. Its soft targets estimate the same objective with lower variance at lane starts and late offsets, which is the recovery band where the learnable gap is.
  - The deficit is information and cannot move.
- **Cost.** One teacher forward per student token (about +1/3 of training FLOPs), plus the teacher's own training, counted in k.
- **Bar.** The L = 64 tax at 1x student tokens drops from 7.6% to ≤ 5%, or L = 16 reaches neutral at 1x, with all teacher FLOPs in the accounting.
- **Kill.** A tax cut < 1.5 points at d8 (two seeds).
- **Nearest work.** Knowledge distillation for non-autoregressive translation, the classic NAT result; Jacobi Forcing (progressive distillation); MTP self-distillation; PTP's distillation variant.
- **Novelty delta.** Thin. It is listed because it is the one lever aimed at the learnable recovery gap that S16 did not try, and NT forbids it.
- **Odds.** A tax cut ≥ 1.5 points about 30%; A* about 5%.

**S17-4. Exit-drafted self-speculation on EET (cross-direction).**
- **Mechanism.** Use the Early Exit Transformer (a CLAUDE.md direction) as its own drafter. Tokens exit at learned depths to draft; one full-depth pass verifies the drafts, so it is lossless.
- **Bar.** 0% tax by construction. At least 2x at batch 1, and above LayerSkip on the same model.
- **Kill.**
  - EET's own gap to dense is not closed: OPEN_QUESTIONS records about 0.06 bpb open, and the base must be neutral;
  - or the accepted draft length falls below LayerSkip's at equal draft depth.
- **Nearest work.** LayerSkip and its successors.
- **Novelty delta.** Learned per-token exit depth from pretraining as the draft policy. Thin.
- **Odds.** A* about 5%. Listed because it is the SAP-compatible use of an active direction, not because it is strong.

## 6. Gates, budget and order (about 10.5 H100-hours left)

1. **S17-1 offline gate (CPU, minutes; no GPU).** It needs the tokenizer and the training shards (Modal CPU job or the user's machine). Code: an inventory miner plus a greedy longest-match segmenter, `scripts/sap_unit_coverage.py`, written once the user picks.
2. **S17-2 gate (about 0.5 H100-hours):**

   `modal run modal_sap.py::s11_ladder --depth 4 --name s17_k --specs ln:16:2:1`

   `modal run modal_sap.py::s11_ladder --depth 8 --name s17_k --specs ln:16:2:1`
3. **S17-1 d8 pair (about 0.6 H100-hours, if the offline gate passes):** an MLU arm against the d8 dense at matched FLOPs (head included), two seeds if within 0.5%. Code: the unit head, the marginal recursion with a length penalty, unit sampling in the decoder, and a test that the marginal sums to one by enumeration.
4. **S17-3** only if the user relaxes NT. **S17-4** only after EET's own gap is closed.

## 7. Honest odds and recommendation

| what | my estimate |
|---|---|
| S17-1 passes its offline gate | about 55% |
| S17-1 is neutral at matched FLOPs with ≥ 1.4 tokens/step at d8 | about 25% |
| S17-2 shows a bounded k at L = 16 | about 25% |
| Any S17 survivor carries a main-track A* paper | **about 10%** |

**Recommendation.**
1. Run the two cheap gates (S17-1 offline, S17-2 at about 0.5 H100-hours) before anything else.
2. If both fail, SAP has no A*-grade path left in this codebase:
   - the seed is published (PTP, CALM);
   - the lossless space is crowded;
   - the lossy exact space is closed by our own data.

   The remaining budget would then do more on the other CLAUDE.md directions (MST, EET, the linear-layer line), whose bar is the FLOPs-bpb Pareto front, than on another SAP mechanism.
