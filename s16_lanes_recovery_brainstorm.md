# S16: raising lanes' recovery (the L1 mechanism for the lanes paper)

Status 2026-10-06, after the S16-C settlement (§7).
- S16-A (infill rows) is killed.
- S16-B (any-L) is no worse at L ≤ 64 against the old baseline, but it is not a recovery mechanism.
- S16-C (lane bias) is worth at most about 0.1-0.3% bpb. It is within the d8 noise and not a recovery mechanism.
- d8 lanes vary by 0.3-0.5% between seeds or code versions (§7). The noise control is pre-registered in §7.
- M0 is void (confirmed by its file, §7); the token-matched rerun is pending.
- Next (user decision): the conversion test (§7), then S16-F, one-stream bridged lanes, which is coded.
- Gates read recovery on the lookahead band (§6), not on the sign split.
- S16-E (checkerboard lanes) was killed while coding: it is plain lanes relabelled (§3).

Inputs: `s15_lanes_paper_plan.md` §7 (Stage L0) and `s14_sap_strict_tl_brainstorm.md` §11 (the order oracle).

## 0. The target, in one paragraph

Plain lanes (L contiguous lanes in lockstep, exact likelihood, KV cache) lose about 7 nats per lane at L = 64, at every size from d4 to d12. The equal-token tax is 6.4 → 7.8 → 8.3%. Split by offset (§1, B1 and B2):
- **The early offsets' deficit is information.** It is 11.9 nats per lane at d12, against 11.3 for an 8B any-order oracle.
- **The late offsets' recovery is learnable.** These are tokens that read the next lane's early tokens. d12 recovers 5.0 nats per lane; the oracle recovers 9.7.
- **The shortfall is at medium range.** It sits at offsets 8 to 28, where the next lane's start is 2 to 22 tokens ahead. The junction token itself is already at the oracle level (−1.49 against −1.56).

The goal: raise recovery by about 2 to 4 nats per lane, with no extra decode steps and ≤ 10% FLOPs. At d12 that would cut the L = 64 tax from 8.3% to about 4 to 6%, and the 8B floor for L = 64 is about 2%.

## 1. Balance sheet (constraints every candidate must satisfy)

| id | fact | source |
|---|---|---|
| B1 | Deficit (offsets 0-15) 9.0 / 11.4 / 11.9 nats per lane at d4 / d8 / d12; oracle 11.3. Information: it can be cut only by adding information (more steps, or codes that are themselves paid for) | S15 §7.2, S14 §11 |
| B2 | Recovery (offsets 16-29) −1.8 / −4.0 / −5.0; oracle −9.7. Offsets 16-28: d12 −3.5 against oracle −7.7; junction (29): −1.49 against −1.56 | S15 §7.2 |
| B3 | Recovery grows with size (+2.2 from d4 to d8, +1.0 from d8 to d12); the d16 pair is pending | S15 §7.4 |
| B4 | Same-step TC floor at T = 1920 (oracle): lanes32 0.51%, lanes64 1.41%, lanes128 3.39%. Bridged lanes (32, 8): 0.07% at 101 steps | S15 §7.1 |
| B5 | Trained bridged lanes (two-stream) lose to plain lanes at equal steps at 1x, but (32, 8) at 4x tokens reached 0.983 of dense-1x (plain L = 64 at 4x: 0.993) | S11 3b |
| BLIND | A variable decided blind costs about what it later tells: splice codes at junctions removed 0% | S13 Q1 |
| SEP | A single-position separator needs 113 to 126 bits to keep the next 256 tokens within 2% | S14 E2a |
| CAP | Capacity taken from one objective costs the others ("even the l2r prefix paid 1-10%" in window bisection) | S11 3a |
| SPD | Lanes' speed comes from S steps for L x S tokens and a step-causal KV cache. Extra steps or recomputation must stay ≤ about 10% | S15 R4 |
| EXACT | The paper's bpb claim needs an exact factorisation: each target is a softmax given what was visible when it was drawn | S08 |
| NT | No teacher or distillation (user decision, S10/S11), unless the user relaxes it | — |

**Prior art that squeezes the novelty** (checked 2026-10-06):
- **LCLM, Line-Coupled Language Model** (arXiv 2609.07129, Sept 2026).
  - Mechanism: natural text lines advanced in lockstep, with cross-line conditioning through an ordinary causal mask, line-staggered RoPE, and KV cache. 881M parameters trained from scratch on 15.5B tokens.
  - Results: 2.94 tokens per pass at a loss of 2.44 against 2.39 for AR; 16 tokens per pass at a loss of 2.34 against 2.25 for its AR baseline.
  - It is the nearest work to plain lanes. Plain lanes alone reads as "LCLM with fixed-length lines".
- **Multi-Stream LLMs** (2605.12460): role streams in lockstep.
- **PAR** (2412.15119): image regions in lockstep.
- **Set Block Decoding** (2509.04185): AR plus masked future blocks.
- **DUS, dilated scheduling** (2506.19037, ICML 2026): coarse-to-fine dilated unmasking for masked diffusion models. Bisection-like; S14 measured that structure at 13.5% TC.
- **FIM** (2207.14255); **UL2**; **Belief State Transformer** (2410.23506, ICLR 2025: forward plus backward prediction for goal-conditioned decoding); **planned diffusion** (2510.18087).

So the paper's novelty has to come from three places:
1. the high-parallelism regime (32 to 64 tokens per pass, where recovery dominates);
2. the information analysis (TC floor; deficit against recovery);
3. a mechanism that closes recovery.

## 2. Pool and funnel (42 candidates → 5 survivors + 2 ingredients)

| family (n) | candidates | fate |
|---|---|---|
| lane-start-targeted (5) | extra compute at offset 0 (mixture-of-depths for starts); lane-start experts; lane-start loss weighting; start-token embeddings per lane; start re-ranking | **killed by B1**: the deficit is information. This is the bundled report's recommendation |
| junction codes (3) | splice codes; boundary classes; junction latents | killed by BLIND, SEP |
| extra-step repair (4) | junction refinement step; turbo message passing between lanes; Gibbs repair of lane ends; multiple shooting with re-solves | killed by SPD (steps), and repair breaks EXACT |
| staggered / natural-line lanes (4) | wavefront lanes; sentence-aligned lanes; natural-line lanes; staggered RoPE | killed by prior art: LCLM owns natural lines and staggered RoPE; S12 killed wavefront and padding |
| two-stream variants (2) | XLNet lanes; two-stream with query side-cars | killed: measured −1.5 points (S11 control) |
| decoding-time only (3) | offset-dependent temperature; lane-start top-k; nucleus by offset | kept as **ingredient I1** (samples, R3); no bpb change |
| teacher-based (2) | distil bridging from LLaDA; recovery KD from a bidirectional teacher | killed by NT (listed in case the user relaxes it) |
| training signal (8) | **position-preserving infill rows**; MDM auxiliary (bidirectional masks); span-infill with spans 2-64; **any-L training**; L curriculum; late-offset loss weighting; reverse-lanes auxiliary; **backward (previous-token) heads** | 3 survive (bold, two merged); loss weighting is allocation only (CAP), dropped; curriculum merged into any-L |
| addressing (4) | **lane-relative attention bias**; offset embeddings; lane-local RoPE heads; lookahead pointer heads | 1 survives (the four merge into one) |
| capacity (3) | **offset-routed capacity for late offsets**; bridge cross-attention module; depth-wise feedback (top states of earlier steps) | 1 survives; feedback needs sequential training (cost ×S), dropped; bridge module merged |
| order design (4) | checkerboard lanes (odd lanes, then even lanes with both neighbours known); **one-stream bridged lanes**; rotated lanes; hierarchical lanes | 1 survives. Checkerboard and rotated lanes are plain lanes relabelled (no gain, by construction; §3, S16-E); hierarchical merged into bridged |
| scale / conversion (2) | **convert a pretrained model to lanes**; more tokens | conversion survives as an option for the user; tokens become a diagnostic (M0) |

## 3. Survivors

Each card gives mechanism, cost, bar (gate), pre-registered kill, and nearest work with the delta.

All gates are read in the same units:
- deficit and recovery nats per lane, and extra nats per lane, from `lane_offset_report`;
- block bpb against dense-1x at the same depth and tokens;
- d8 at 1x tokens, L = 64, against `S11ln64x1_s1` rescored with the new report. Stage M1's `s16_score` call includes it and `S11ln32x1_s1`, so it replaces S15 §7.4's `s15_d8b` rescoring.

**S16-A. Position-preserving infill rows (PPI). The lead, cheapest to test.**
- **Mechanism.**
  - A fraction f of training rows are re-ordered as prefix, then suffix, then middle, keeping each token's true rotary position (`pos_ids`), under a causal mask on the re-ordered sequence.
  - The suffix's first input slot holds the lane-start token. The middle is predicted left to right with the whole suffix known, ahead in position.
  - Every target is predicted once, so it is an exact 3-segment order, a two-lane order with the second lane drawn first.
  - Middles are 2 to 64 tokens long, and several per row keep the infill share high.
- **Why.**
  - Ordinary causal training never shows the model keys ahead of the query in position. Lanes' recovery is exactly that pattern, at distances 1 to S.
  - Infill rows train it on most tokens, not just late offsets.
  - FIM showed infill training is nearly free for left-to-right loss.
- **Cost.** 0 extra FLOPs (rows replace lanes rows) or +f (rows added). One stream.
- **Gate (d8).** At f = 0.25, recovery ≥ +1.0 nats per lane, deficit within +0.3, and lanes block bpb ≥ 0.5% lower than the plain-lanes baseline at matched tokens.
- **Kill.** Recovery gain < 0.4, or block bpb not lower (two seeds if within 0.3%).
- **Nearest work.** FIM (2207.14255): position-reordered, sentinel-based, for infilling; UL2 / GLM: mixtures of objectives; LCLM: no infill objective.
- **Delta.** Infill rows with true positions, as a training signal for exact lockstep lanes' lookahead, measured by the recovery split.

**S16-B. Any-L lanes (one model, any parallelism).**
- **Mechanism.** Each micro-step draws L from {16, 32, 64, 128} (prefix lengths stay multiples of L). One model decodes at any L, dialled at inference.
- **Why.** It varies the lookahead distances (S = 120 to 15) and their mix.
- **Paper value.** A single exact model offers a speed/quality dial, which LCLM's line structure does not give.
- **Cost.** 0.
- **Gate (d8).** At L = 64, extra nats per lane ≤ the single-L model's (no worse) and recovery ≥ +0.5. At L = 32 and 128, within 0.5% bpb of their single-L models (the L = 32 d8 single-L model exists).
- **Kill.** > 1% worse than single-L at L = 64.
- **Nearest work.** LCLM (variable lines, fixed per model); σ-GPT / any-order AR (random orders, not lockstep lanes); elastic or anytime decoding.
- **Delta.** Exact lockstep lanes with a train-time distribution over parallelism.

**S16-C. Lane-relative attention bias (LRB).**
- **Mechanism.** A learned per-head scalar bias added to attention scores, as a function of:
  - the key's lane relative to the query's: same, next, previous, or other;
  - the bucketed offset gap.

  The lanes path already uses an explicit SDPA mask, which becomes a float mask with bias. An offset embedding goes in the input.
- **Why.** It makes "the next lane's early tokens" addressable as a role, not only through RoPE distances.
- **Cost.** About 0 FLOPs; a (T, T) bias per head in training (memory), cached per step at decode.
- **Gate (d8).** Recovery ≥ +0.7, block bpb ≥ 0.3% lower.
- **Kill.** Recovery gain < 0.3.
- **Nearest work.** T5 relative biases, ALiBi, LCLM's line-staggered RoPE (positions only).
- **Delta.** Lane-role attention biases for cross-lane lookahead.
- **Coded** by the user's agent in `Seqaeon/nanochat` (80101b8) and ported unchanged:
  - `nanochat/lanes.py::compute_lrb_buckets`: 44 buckets per head, made of 5 lane roles × 7 offset-gap buckets, 8 prefix-distance buckets, and 1 lane-to-prefix bucket;
  - `GPT.lane_rel_bias` and `lane_offset_embed`.
- **Control** (added here): `--lane-rel-bias-roles 0` (ladder spec `lrbd`) collapses the roles.
- **Not yet in the KV-cache decoder.** Result: §6.

**S16-D. Offset-routed recovery capacity (ORC).**
- **Mechanism.** Positions at offsets ≥ S/2 (where recovery happens) pass through one extra MLP expert per layer (deterministic routing by offset), or k extra top layers.
- **Cost.** +5 to 10% FLOPs. Compare with spreading the same capacity uniformly at matched FLOPs.
- **Why.** Recovery grows with size (B3), so it may be capacity-limited.
- **Gate (d8).** Recovery ≥ +0.7 beyond the uniform-capacity control at matched FLOPs.
- **Kill.** No better than the uniform control.
- **Nearest work.** Mixture-of-Depths (2404.02258), PCB (S14-A), CALM.
- **Delta.** Capacity routed by lane offset, from the measured recovery profile. Not coded yet.

**S16-E. Checkerboard lanes: killed by construction (found while coding the oracle gate).**
- **The proposal.** 2L lanes of S/2 tokens. The odd lanes are drawn in lockstep (S/2 steps), then the even lanes (S/2 steps), each with both neighbours known.
- **Why it is dead.** An order is fully defined by the step at which each position is drawn. Take position 2j(S/2) + o, with 0 ≤ o < S:
  - for o < S/2 it is in a phase-1 lane at offset o, so it is drawn at step o + 1;
  - for o ≥ S/2 it is in a phase-2 lane at offset o − S/2, so it is drawn at step S/2 + (o − S/2) + 1 = o + 1.

  Plain `lanes{L}` also draws that position at step o + 1. Every token sees the same set, so the oracle's total is identical and a trained one-stream model gets the same mask. The "warm start" of the phase-2 lanes is simply the second half of a plain lane, and their "lookahead" is the next lane's first half.
- **What would differ.** Drawing phase 2 right to left, or from both ends, changes the order. But it keeps the same L cold starts drawn together at step 1, so no TC gain is expected. A causal model would also need backward prediction (I2).
- **Lesson.** Before proposing an order, write down each position's step and compare it with the plain orders (LEARNINGS 2026-10-06).

**S16-F. One-stream bridged lanes (BL1).**
- **Mechanism.** S11's bridged lanes (separator windows first, coarse to fine, then fills), trained in one stream:
  - per-slot generation steps;
  - lane-start tokens in slots whose input is not yet known;
  - a step-causal mask with true positions.

  This removes the two-stream penalty (about 1.5 points, 2x FLOPs).
- **Why.** The oracle measures it as the most information-efficient order: TC 0.07%, total 0.86% at 101 steps.
- **Cost.** (log2 L + 1) x n extra steps, so 101 against 61 for 32 lanes.
- **Gate (d8).** At 101 steps, tax below plain lanes L = 16 at 121 steps.
- **Kill.** Not better than plain lanes at equal or fewer steps (S11's reading for two-stream).
- **Nearest work.** S11 bridged lanes; insertion / bisection orders; DUS.
- **Delta.** A one-stream exact bridged order. Not coded yet; after M1.

**Ingredients.**
- **I1.** Offset-dependent sampling temperature: lower at lane starts, for R3; no bpb effect.
- **I2.** Backward (previous-token) heads as an auxiliary on all positions, in the Belief State Transformer style. It gives a product-of-experts bridge at late offsets. It rides on S16-A if PPI passes.

**Option for the user: S16-G, convert a pretrained model (PMC).**
- **Mechanism.** Fine-tune a pretrained AR model (for example a 0.5B to 1.5B Qwen) or an MDM into plain lanes: lane mask, lane-start token, a few billion tokens.
- **Why.** The 8B any-order oracle already recovers 9.7 nats per lane zero-shot (total 2.42% at L = 64), so pretrained models carry the bridging skill.
- **Cost.** 10 to 40 H100-hours.
- **Bar.** ≤ 2% tax at L = 32 against the base model's own AR bpb, with ≥ 20x batch-1 speed.
- **Constraint.** It changes the paper from "pretrained from scratch" to "conversion", so it is the user's call.
- **Nearest work.** Dream (AR → MDM), SBD (AR → masked future blocks), LCLM (from scratch).

## 4. Stages and pre-registered readings

**Stage M0: eval-only (CPU, plus well under 1 H100-hour)**
1. **The oracle's exact per-offset profile.** This is S15 §7.4's `--merge-only` re-merge of t1920 (CPU, no model load). It gives the oracle's deficit, recovery and per-step TC at L = 32, 64 and 128 on the same rows, which is the reference every recovery number is read against.
2. **Recovery against training tokens.** Rescore d4 plain lanes at L = 64 at 1x and 4x (two seeds), each token budget against its own dense model:

   `modal run modal_sap.py::s16_score --depth 4 --name tokens1x --models S11ln64x1_s1@64,S11ln32x1_s1@32`

   `modal run modal_sap.py::s16_score --depth 4 --name tokens4x --ref S11dense_x4_s2 --models S11ln64x4_s1@64,S11ln64x4_s2@64,S11ln32x4_s1@32`

   - **Corrected 2026-10-06.** As first written, this step was one call that scored the 4x models against dense-1x. That books the general gain from 4x tokens as recovery (§6, M0).
   - Read the change on the lookahead band (§6) and on recovery.
   - If recovery grows by ≥ 2 nats per lane from 1x to 4x (≥ 1 per doubling of tokens), recovery is sample-limited, and the training-signal mechanisms (S16-A, B) get priority.
   - If it grows by < 0.5, the capacity mechanism (S16-D) gets priority.
   - In between, both stay in.
   - The 4x tags are S11's two seeds (0.993 and 0.992 of dense-1x). If either is not on the volume, the job names the missing directory.

**Stage M1: d8 gates (1x tokens, about 0.3 H100-hours per arm)**
- **Train** (the existing d8 dense `S11dense_x1_s1` is the reference; it is not retrained):

  `modal run modal_sap.py::s11_ladder --depth 8 --name s16_m1 --specs ppi:64:0.25:1:1,mix:16+32+64+128:1:1,ln:128:1:1`

  - `S16ppi25L64x1_s1`: S16-A at f = 0.25, middles of 2 to 64 slots, one per 128 slots (about 16 per row, a quarter of the slots).
  - `S16mix16_32_64_128x1_s1`: S16-B, L drawn from {16, 32, 64, 128} per micro-step. The ladder scores it at L = 64.
  - `S11ln128x1_s1`: the single-L control S16-B needs at L = 128 (d8 has single-L models at 32 and 64 only).
- **Score** the baseline with the new report, and S16-B at its other lane counts, all against the same dense model:

  `modal run modal_sap.py::s16_score --depth 8 --name m1 --models S11ln64x1_s1@64,S11ln32x1_s1@32,S16mix16_32_64_128x1_s1@32,S16mix16_32_64_128x1_s1@128`

- **Readings**, as in each card, all from `lane_offset_report`'s deficit and recovery nats per lane and block bpb:
  - S16-A against `S11ln64x1_s1`;
  - S16-B at 32, 64 and 128 against `S11ln32x1_s1`, `S11ln64x1_s1` and `S11ln128x1_s1`.
- A go needs two seeds if the bpb gain is within 0.3%. Rerun the arm with seed 2 (`...:1:2`).

**Stage M2: only for an M1 go.**
- The winner at d12 (and d16 if the R1 tiebreak runs).
- Samples at real-text entropy (R3).
- Then the paper protocol (OWT, 350M).
- The A* claim to aim for: at 32 to 64 tokens per pass, a tax of ≤ 3% at ≥ 350M parameters. That sits beyond LCLM's frontier (16 tokens per pass at +4%) and next to the information floor.

## 5. Honest odds

| what | my estimate |
|---|---|
| PPI (S16-A) passes its d8 gate | about 40% → **killed** (§6) |
| Any-L (S16-B) passes "no worse" | about 70%; a recovery gain, about 30% → **no worse at L ≤ 64; no recovery gain** |
| Some survivor closes ≥ 50% of the recovery gap at d12 | about 25% → about 15% (the best so far, S16-C, closes about 12% at d8, on one seed) |
| The resulting paper reaches A* (it still needs scale and the protocol) | about 15 to 25% → **about 10 to 15%** |

## 6. Stage M1 results (2026-10-06)

**Data.**
- Files: `scratch/s16/`, copied from `Seqaeon/nanochat` (`scratch/sync_d8/` and `scratch/`).
- Setup: d8 models at 1x tokens (440.4M), 256 rows, against dense `S11dense_x1_s1` (0.9346).
- Checks passed:
  - the baseline reproduces S15 exactly (1.0076 bpb; 7.562 extra nats per lane);
  - every arm has the same hyperparameters and 1,680 steps;
  - the report's numbers match the JSONs.

| model | L | block bpb | against single-L | extra nats/lane | deficit | recovery | lookahead band (gain) |
|---|---|---|---|---|---|---|---|
| `S11ln32x1_s1` | 32 | 0.9806 | — | 10.434 | 15.756 | −5.322 | −1.864 |
| `S11ln64x1_s1` | 64 | 1.0076 | — | 7.562 | 11.652 | −4.090 | −1.118 |
| `S11ln128x1_s1` | 128 | 1.0457 | — | 5.663 | 8.872 | −3.209 | −0.555 |
| `S16ppi25L64x1_s1` (S16-A) | 64 | 1.0105 | +0.29% | 7.823 | 11.811 | −3.989 | −1.116 (−0.002) |
| `S16mix16_32_64_128x1_s1` (S16-B) | 32 | 0.9834 | +0.29% | 10.970 | 16.382 | −5.412 | −1.274 (−0.590) |
| same | 64 | 1.0081 | +0.05% | 7.563 | 11.585 | −4.023 | −1.134 (+0.016) |
| same | 128 | 1.0571 | +1.09% | 6.241 | 9.059 | −2.817 | −0.180 (−0.376) |
| `S16lrbL64x1_s1` (S16-C) | 64 | 1.0025 | **−0.51%** | 7.018 | 11.333 | −4.314 | −1.684 (**+0.566**) |

**How to read the table.**
- All nats are per lane, over lanes 1..L-1.
- The lookahead band (`lookahead_band`) is the excess summed over offsets ⌈S/4⌉..S−2. Its gain is the single-L model's band minus the arm's, so a positive gain means more lookahead benefit.

**Verdicts.**
- **S16-A (infill rows): kill.**
  - Recovery fell by 0.10 (the bundled report gave it as a +0.10 gain), and bpb is 0.29% worse. The band is unchanged.
  - Its costs are at offsets 0-7 (+0.14 nats per lane) and at the junction (+0.12).
- **S16-B (any-L): neither go nor kill.**
  - At L = 64 it is no worse (+0.05%) but gains nothing: recovery −0.07, band +0.02.
  - At 32 it is +0.29%. At 128 it is +1.09%, which fails the 0.5% clause.
  - Lookahead drops at the lane counts other than 64 (band −0.59 at 32, −0.38 at 128).
  - Keep it as a speed/quality dial for L ≤ 64, not as a recovery mechanism.
- **S16-C (lane bias plus offset embedding): a kill by the pre-registered letter, between kill and go on the corrected metric.**
  - Its card needed recovery ≥ +0.7 and bpb ≥ 0.3% lower to go, and killed it at recovery < +0.3. Measured recovery was +0.22. The bundled report quoted only the bpb half of the gate.
  - **The metric was the flaw.** S16-C's whole gain is at offsets 8-28: −0.30 nats per lane at 8-15 and −0.27 at 16-28; all other offsets sum to +0.02. That is the band §0 names as the learnable shortfall. The sign split books offsets 8-15 as deficit, because their excess is still positive.
  - **On the band it is +0.57**: above the kill (0.3), below the go (0.7).
  - **The bpb gain is probably real.** The any-L arm, on the same code and machines, lands within 0.05% of the baseline at L = 64, and the seed spread at d4 is 0.04-0.1%.
  - **Not yet counted:**
    - it is one seed;
    - it has no decoder support. `generate_lanes` and `lane_step` pass no lane mask, so the bias and the offset embeddings are skipped at decode. Samples would not come from the scored model, and the 30-step speed is unmeasured;
    - training wall-clock rose 32% (1,027 s against 777 s);
    - the bias also covers prefix-to-prefix distances, so part of the gain may be a generic relative-position bias (the prefix bucket moved 0.4%).

**Metric correction (pre-registered from now).**
- S16 gates read recovery on `lookahead nats per lane` (`scripts/sap_position_bpb.py::lookahead_band`), with the same bars: go ≥ +0.7, kill < +0.3.
- The deficit/recovery split stays as a diagnostic.

**M0: void.**
- The single call first written in §4 scored the 4x lanes against dense-1x. Lanes-4x is 0.993 of dense-1x, so its net excess against dense-1x is about −0.8 nats per lane, against +7.4 at 1x. The sign split books most of that general gain as recovery: the reported +5.655 is what this looks like.
- Token-matched, the net tax is flat (1.0655 against 1.063 at 1x). A +5.7 recovery gain would then need a d4 deficit near 14 nats; d12's is 11.9 and the 8B oracle's 11.3.
- Neither repo holds a JSON or log of the run.
- Rerun with the corrected commands in §4. M1 already shows that the two training-signal arms do not move recovery.

**S16-C settlement (about 1 H100-hour), pre-registered before the runs:**
```
modal run modal_sap.py::s11_ladder --depth 8 --name s16_c2 --specs lrb:64:1:2,ln:64:1:2,lrbd:64:1:1
```
- **Real.** The two-seed mean of `S16lrbL64x1_s1/_s2` is ≥ 0.3% lower bpb than that of `S11ln64x1_s1/_s2`, and both seeds are lower.
- **Recovery mechanism**, on the two-seed mean band gain:
  - ≥ 0.7: go to d12;
  - 0.3 to 0.7: a free add-on, not a headline;
  - < 0.3: drop the recovery claim.
- **Attribution** (`S16lrbdL64x1_s1`, lane roles collapsed into offset-gap buckets):
  - if it keeps ≥ 80% of seed 1's bpb gain, the gain is a generic relative bias. There is then no lane-addressing claim, and dense needs the same bias as a fair baseline;
  - if it keeps ≤ 50%, the lane roles carry the gain.
- Decoder support (bias rows and offset embeddings in `lane_step`, plus a decoder-matches-training test) comes only if S16-C survives this.

**Paper status (frank).**
- No mechanism closes recovery yet. S16-C cuts the d8 L = 64 tax from 7.8% to 7.3%, about 12% of the roughly 4.7 nats per lane that a ≤ 3% tax needs.
- **Remaining levers:**
  - S16-F, one-stream bridged lanes: the oracle's most efficient order, 0.86% total at 101 steps, about 19 tokens per pass. It is to be coded after the settlement runs; its gate is unchanged;
  - S16-G, converting a pretrained model: the user's call;
  - scale: d16 costs about 9 of the roughly 12.7 H100-hours left.
- **Not now:** d16, the R3 samples sweep, S16-D.

## 7. S16-C settlement, the noise floor, and what runs next (2026-10-06)

**Data.**
- Files: `scratch/s16/s16_score_d8_s16_c2_full.json` and `scratch/s16/s16_score_d4_tokens.json`, from `Seqaeon/nanochat` 733ae88.
- Setup: d8 at 1x, 256 rows, against dense `S11dense_x1_s1` (0.9346). The report's numbers match the JSON.

| model | trained on | block bpb | extra nats/lane | deficit | recovery | band | prefix bucket [0, 128) |
|---|---|---|---|---|---|---|---|
| `S11ln64x1_s1` | code before Oct 5 | 1.0076 | 7.562 | 11.652 | −4.090 | −1.118 | 1.1733 |
| `S11ln64x1_s2` | current code | 1.0029 | 7.046 | 11.393 | −4.347 | −1.666 | 1.1653 |
| `S16lrbL64x1_s1` | current | 1.0025 | 7.018 | 11.333 | −4.314 | −1.684 | 1.1688 |
| `S16lrbL64x1_s2` | current | 1.0011 | 6.877 | 11.345 | −4.469 | −1.710 | 1.1669 |
| `S16lrbdL64x1_s1` (roles collapsed) | current | 1.0036 | 7.155 | 11.436 | −4.280 | −1.543 | 1.1681 |

How to read it: nats per lane over lanes 1..63; band = `lookahead_band`; "trained on" comes from the config fields saved with each checkpoint.

**Verdicts by the pre-registered letter (§6).**
- **Real: pass.** The two-seed mean is −0.343%, and both seeds are lower.
- **Recovery: free add-on.** The band gain is +0.305 (seed 1 +0.566, seed 2 +0.044): 0.005 above the drop line.
- **Attribution: no verdict.** The roles-collapsed control keeps 78% (bpb) or 75% (band) of seed 1's gain, between the 50% and 80% bars. The bundled report rounded it to "generic".

**What the letter hides.**
- The two baseline seeds differ by 0.47% bpb and 0.55 on the band, as large as every effect above. The two-seed comparison has Welch t ≈ 1.4.
- The outlier is `S11ln64x1_s1`, the only model trained on code from before Oct 5. It is the worst of the five on every column, including the plain causal prefix bucket.
- Against the same-code baseline `S11ln64x1_s2`:
  - S16-C gains −0.04% and −0.18% bpb, and +0.02 and +0.04 on the band;
  - the roles-collapsed control is +0.07% (worse) and −0.12 on the band.
- **So S16-C is worth at most about 0.1-0.3%.** It is not a recovery mechanism. Keep it off by default.
- My "seed spread 0.04-0.1%" in §6 came from d4 dense and was wrong for d8 lanes.
- If the old code is what separates `s1`, the M1 readings shift as well:
  - infill rows: +0.29% → +0.76% against `s2`, still a kill;
  - any-L at 64: +0.05% → +0.52%.

  The noise control decides which applies.

**M0 is void, confirmed by its file.**
- Against dense-1x, the 4x models score −0.510 and −0.564 extra nats per lane, better than the reference.
- Their deficit falls from 9.338 to 7.083 and 7.013, and "recovery" from −1.930 to −7.593 and −7.578. That is the general gain from 4x tokens, split by sign.
- The token-matched rerun (§4) is pending.

**Next: the user chose conversion first, then S16-F.** Pre-registered before the runs:

1. **Noise control:**

   `modal run modal_sap.py::s11_ladder --depth 8 --name s16_noise --tag-suffix _cur --specs dense:1:1,ln:64:1:1`

   This retrains seed 1 of both baselines on current code (`tag_suffix` sidesteps the skip-if-exists rule).
   - `S11ln64x1_s1_cur` within ±0.1% of 1.0076: no code shift. The 0.47% is seed noise, and effects under 0.5% need four or more seeds per arm.
   - ≤ 1.0046: a code shift. Every reading against the old `S11ln64x1_s1`, `S11ln32x1_s1` or old dense is void and is re-read against same-code baselines.
   - The same rule applies to `S11dense_x1_s1_cur` against 0.9346, on which the d8 tax numbers rest.
2. **Conversion (S16-G at d8, in-house):**

   `modal run modal_sap.py::s11_ladder --depth 8 --name s16_g1 --specs cv:64:0.25:1,dc:0.25:1,cv:64:1:1,dc:1:1`

   - `cv` starts L = 64 lanes training from the trained dense `S11dense_x1_s1` (`--sap-init-trunk`). `dc` continues the same dense, as dense, for the same tokens. Both use the default LR schedule.
   - T_m = bpb(`cv_m`) / bpb(`dc_m`) − 1. From-scratch lanes sit at 7.3-7.8% at 1x.
   - **Validity:** both `dc` arms must land below 0.9346, or the continuation schedule is broken and the test is void.
   - **Go:** T_0.25 ≤ 3.0%. A public pretrained model then becomes the paper's core, and the claim changes to conversion.
   - **Promising:** T_1 ≤ 4.5%, one more probe.
   - **Dead:** T_1 ≥ 6.5%. Then run S16-F.
3. **S16-F, one-stream bridged lanes (coded):**
   - `nanochat/lanes.py::bridged_slot_steps`, `step_inputs`, `step_mask`; `scripts/base_train.py --lane-bridge N`; scoring spec `boL_N`.
   - Each slot runs at its target's step, and a slot whose input is drawn later holds the lane token.
   - In bridged lanes, the only such slots are the separator windows' first slots (32 at L = 32); every fill starts warm. That is 100 steps at (2048, 128, 32, 8).
   - On plain lanes' steps it reproduces `lane_mask` and `lane_inputs` exactly. There is no decoder until it passes.

   `modal run modal_sap.py::s11_ladder --depth 8 --name s16_f1 --specs bo:32:8:1:1,ln:16:1:1`

   - **Pass:** `S16bo32n8x1_s1` (100 steps) has bpb ≤ `S11ln16x1_s1` (120 steps, trained in the same call).
   - **Kill:** ≥ +0.5% against it.
   - **In between:** a second seed of both.

