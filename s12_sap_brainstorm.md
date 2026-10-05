# S12: after lanes, what removes the lane-start cost (brainstorm, 2026-10-04)

The user asked for a quick lane-start brainstorm that is free to drop lanes. It should combine what each earlier mechanism did well. Training up to 5x tokens is acceptable (user, 2026-10-04): the benefit is at inference.

## 1. Measured facts the pool is filtered by

**Assets (what worked, and where).**

| Asset | Evidence |
|---|---|
| A1. Plain lanes | One stream, exact likelihood, KV-cached. 30 to 60 steps for 1920 tokens; 53x at batch 1 (L=64), 28x (L=32). Pass the bpb bar at 4x tokens on d4 (L=64, 32) and on d8 (L=32: 0.987; L=64: 1.015, so 5x is needed). |
| A2. Lane samples | Better than next-token samples at matched diversity. DeepSeek judge, whole-text coherence rubric, d4, about 105 valid pairs each: 75% win rate against dense-1x (60 wins, 7 losses, 38 ties) and 68% against dense-4x (50, 12, 45). The DeepSeek balance ran out (HTTP 402) before the other pairs. |
| A3. Right context is cheap and learnable | A lane's last token, written after the next lane's start is known, costs 0.43 to 0.86x dense. |
| A4. Teacher-forced exact factorisations learn | Sampling inside training does not (root cause 0). |
| A5. Far context compresses | A summary bottleneck costs about 0 beyond 64 tokens (separator oracle). Local continuity and copying need per-position access. |
| A6. Coarse-to-fine decisions on states are exact | Toy: log2(T) + 2 levels when the decision variables are pure separators. |
| A7. Document-boundary lane starts are free | This oracle, below. |

**Liabilities.**

| Liability | Evidence |
|---|---|
| L1. Cold lane starts | Offset 0 costs 1.95x (d4) and 2.35x (d8), and offsets 1 to 7 cost up to 1.7x. Flat in training tokens and growing with model size, so the tax grows from d4 to d8: 3.9 to 4.9% at L=32, 6.3 to 7.8% at L=64. |
| L2. Far-ahead token windows are expensive | Window bisection, bridged-lanes separators. |
| L3. Two-stream training | Costs about 1.5 points and doubles compute. |
| L4. Backward prediction with shared weights | Hurts everything (seeded lanes, killed). |
| L5. Adjacent tokens drawn in the same step | Cost about 19% of next-token NLL at T=2 (K1). Local chain heads saturate at 1.17 to 1.21x (K3). |
| L6. Learned latents | Codes, noise tapes and flows did not learn on the gates (S04 to S06, S09, S11 toy). |

## 2. A structural fact that removes a family

In any order where each front writes one token per step, every front has exactly one blind end:
- **Plain lanes** have a blind start and a bridging end, written last with the next start known.
- **Right-to-left lanes** mirror this.
- **Alternating directions** put two blind starts, or two ends written in the same step, next to each other, which is worse.
- **Deferring each lane's first m tokens to the last m steps** looks like a "ligase" region. It is information-equivalent to plain lanes with every boundary shifted by m.
- **Wavefront (staggered) starts** give each new start only a sparse context: the first tokens of the lanes before it.

So with L tokens per step there are about L cold starts. Only two levers remain: where the starts fall (what they are blind to), and what cheap information reaches them first.

## 3. The oracle that decides the lever (`scripts/sap_lane_start_oracle.py`, no training)

Plain-lanes checkpoints were scored against dense-1x on 256 validation rows. Each lane start is split by the true token just before it (the junction, which the lanes model writes last). The table gives excess nats per lane over offsets 0 to 7.

| Junction class (share of starts) | d4 L=64 4x | d8 L=64 4x | d8 L=32 4x |
|---|---|---|---|
| mid-sentence (93 to 94%) | 6.97 | 9.54 | 10.89 |
| sentence end, `. ! ?` (2 to 3%) | 4.35 (-38%) | 5.76 (-40%) | 6.44 (-41%) |
| newline (3%) | 6.62 (-5%) | 8.46 (-11%) | 9.67 (-11%) |
| document start, `<|bos|>` (0.6 to 1.2%) | -0.18 (free) | -0.49 (free) | -0.13 (free) |

- After a sentence end, the first token costs 2.5 to 3.5x, but offsets 1 to 7 cost about the same as dense. After a mid-sentence start, offsets 1 to 7 cost 1.36 to 1.70x.
- Newlines barely help. Many are list, heading or code structure that continues across the break.
- These models were trained with 94% mid-sentence starts. A model trained with aligned starts could do better than this oracle reads.

## 4. Pool (36 candidates, by source field) and the filter

**Inside lanes, where starts fall or what reaches them:**
1. Sentence-aligned lanes.
2. Paragraph-aligned lanes.
3. Document-aligned lanes.
4. Sentence-interleaved lanes with virtual positions.
5. Junction state codes generated first.
6. Coarse-to-fine junction codes.
7. Lanes conditioned on a fast sequential draft (Parareal-like).
8. Speculative lanes (multi-token drafts verified per lane).
9. Any-L lanes (mixed lane counts in training).
10. Best-of-K worlds reranked by seam scores.
11. Gibbs or Jacobi seam refinement.
12. Lanes with lengths growing with distance from the prompt.
13. Wavefront lanes.
14. Deferred-head lanes.
15. Boustrophedon lanes.
16. Seeded middle-out lanes (killed).
17. Lanes plus local two-token chain heads.
18. A dedicated "start expert" path.
19. Coupled-noise starts (tree pick across lanes).
20. Lanes plus SuperBPE (fewer tokens).

**Without lanes:**
21. Window bisection.
22. Bridged lanes.
23. Bisection over sentences.
24. Multiscale dilated sketch and fill (Reed 2017 for text).
25. CALM-style next-chunk vectors.
26. Block diffusion.
27. MDLM at few steps.
28. Hidden-state bisection with a continuous generator.
29. One-pass SAP heads (S01 to S10).

**From other fields:**
30. Multigrid V-cycle (numerical analysis), which is 21.
31. Okazaki fragments and ligase (DNA replication), which is 14.
32. Nucleation and growth (chemistry), which is 16.
33. Sequential Monte Carlo resampling of starts (statistics), which is 10.
34. LDPC message passing between lanes (coding theory), which is 11.
35. Discourse planning (linguistics), which is 5 and 6.
36. Phrase chunking in speech production (psychology), which is 1.

**Killed by the filter.**

| Reason | Candidates |
|---|---|
| Section 2 (information-equivalent, or worse) | 13, 14, 15, 31 |
| L4 | 16, 32 |
| L2, measured | 21, 22, 23, 30 |
| L5 | 17, 24 (the K4 bisection tax of 10 to 16%) |
| L6 | 25, 28, 29 |
| Not exact, or samples only, with low novelty | 10, 11, 33, 34 |
| Information-limited, not capacity-limited | 18 |
| No information added about the missing left context | 19 |
| Breaks lockstep for little gain | 12 |
| Breaks T=L (N sequential draft steps) | 7 |
| Rare in long-form text | 3 (kept as a training-data rule; see S-1) |
| The comparison class, not a contribution | 26, 27 |
| Orthogonal and not novel | 20 |

**Survivors:** 1 (with 2 and 4 as its variants), 5 and 6, 8, 9.

## 5. Survivors

**S-1. Sentence-aligned lanes (lead).**
- **Mechanism.** Lane boundaries move to the last sentence end inside each slot's final window. The rest of the slot is padded with a no-text token, scored and predicted (an exact likelihood of the canonical segmentation, so bpb is an upper bound). Lanes that hit a document start are aligned there too.
- **Cost.** Pad waste is about half a sentence per lane. At L=32 (60-token slots) that is about 20%: about 75 steps instead of 60, 26x instead of 32x.
- **Bar.** Oracle: about 40% less start excess, so the d8 tax at equal steps falls from about 4.9% toward 3%.
- **Kill.** At d4 1x, the tax at equal decode steps drops by less than 25% against plain lanes.
- **Variant 4 (no padding).** Lanes write consecutive sentences with virtual positions. More engineering; only if S-1 passes.

**S-2. Junction state codes (state-first plus lanes; the toy's lesson on real text).**
- **Mechanism.** Each junction gets a discrete code: k-means of a frozen small dense model's hidden state at the junction, plus the boundary class. The code is a deterministic function of the text, so the likelihood stays exact. All codes are generated at step 0, then lanes run conditioned on them.
- **Cost.** One extra step.
- **Bar.** Tax reduction, if a coarse code is easier to predict blind than tokens.
- **Kill.** Start excess falls by less than 30%, or the codes' own cost exceeds the saving (net bpb at d4 1x no better than plain lanes at equal steps).
- **Risk.** L6. Codes were impure as exact separators, but here they only need to inform.

**S-3. Speculative lanes.**
- **Mechanism.** Each lane drafts k tokens per step with a cheap multi-token head. The next step verifies every lane's draft against the lanes model's own conditionals and keeps an accepted prefix per lane. Samples stay exactly from the lanes distribution.
- **Cost.** k x L rows per step (free at small batch) and a small auxiliary head.
- **Bar.** Performance neutral plus speed: tokens per step multiply by the mean accepted length.
- **Kill.** Mean accepted length below 1.3 tokens per lane-step at d4.

**S-4. Any-L lanes.**
- **Mechanism.** Train with L drawn per batch, including L=1. One model then decodes at any lane count, and at L=1 as a plain next-token model.
- **Bar.** Practicality; possibly less tax through shared learning.
- **Kill.** Per-L tax more than 1 point above the dedicated models.

**The paper hook to verify.**
- A2 says lane samples beat next-token samples at matched diversity. The candidate mechanism: errors compound over 30 to 100 self-conditioning steps instead of 1920.
- It is shown only at d4, where next-token samples degenerate into loops.
- It must be re-measured at d8 against good next-token decoding (nucleus, repetition penalty) before anything is built on it. The d8 dense and lanes checkpoints already exist on seqaeon.

## 6. Results so far

**d8 sample frontier (2026-10-04, seqaeon).**
- Setup: 256 prompts of 64 tokens, 1985 generated tokens.
- Scorer: the other d8 dense-1x seed.
- Real text: unigram entropy 5.88, reference ppl 21.7, distinct 3-grams 0.916.

| Sampler | Entropy | Reference ppl | Distinct 3-grams |
|---|---|---|---|
| Dense-1x next-token, T=0.9 | 5.32 | 14.6 | 0.826 |
| Dense-1x next-token, T=1.0, top-p 0.95 | 5.56 | 21.9 | 0.881 |
| Dense-1x next-token, T=1.0 | 5.94 | 55.9 | 0.930 |
| Plain lanes L=64, 4x, T=0.85 | 5.56 | 21.7 | 0.911 |
| Plain lanes L=64, 4x, T=0.9 | 5.76 | 33.8 | 0.945 |
| Plain lanes L=64, 4x, T=1.0 | 6.18 | 101.4 | 0.981 |
| Plain lanes L=32, 4x, T=0.9 | 5.70 | 26.5 | 0.927 |
| Plain lanes L=32, 4x, T=1.0 | 6.15 | 86.4 | 0.977 |

**Reading.**
- Interpolated to real text's entropy (5.88), reference ppl is about 48 for next-token, 46 for L=64 and 42.5 for L=32. At entropy 5.56 the two samplers tie (21.9 against 21.7), and the lane samples have more distinct 3-grams.
- **The d4 advantage (about 199 against 67) was mostly next-token degeneration at tiny scale.** At d8, next-token sampling at temperature 1 no longer loops. Lanes are at parity, or slightly better, which passes the user's matched-entropy criterion.
- **The paper hook is not "better samples than next-token". It is sample parity with 28 to 53x faster decoding.**
- The CTRL repetition penalty of 1.2 over prompt and output destroys 2000-token generations (entropy 7.2 to 7.3, reference ppl 573 to 1938) by suppressing every common token. It is not a usable baseline at this length.
- The DeepSeek judge for d8 pairs waits for the user's balance top-up (HTTP 402).

**S-1 sentence-aligned lanes: killed (2026-10-04, d4, 1x, seqaeon, `scripts/sap_aligned_lanes_eval.py`).** Block bpb after a 128-token prefix over 256 rows, with the aligned lanes' pad (lane-end) decisions counted in the nats.

| Model | Ratio to dense | Text tokens per step | Plain lanes at the same tokens per step (log2 interpolation) |
|---|---|---|---|
| Plain lanes L=64 | 1.0636 | 64 | |
| Plain lanes L=32 | 1.0382 | 32 | |
| Aligned L=64, window 15 | 1.0665 | 55.0 (86% of slots) | about 1.058 |
| Aligned L=32, window 30 | 1.0368 | 26.7 (84% of slots) | about 1.034 |

- At equal text per decode step, aligned lanes are slightly worse, not 25% better.
- Only 45% (L=64) and 74% (L=32) of lanes find a boundary in their window. The padding costs 14 to 16% of slots.
- The roughly 40% saving the oracle measured per aligned start does not survive the lost text per step.
- Without the pad decisions the numbers barely move (1.1968 against 1.1979), so the end-of-lane decision is cheap. The loss is the waste.
