# S15: can plain lanes reach the A* bar?

Status 2026-10-05. Opened after S14 closed the strict thesis on information (`s14_sap_strict_tl_brainstorm.md` §11). The user chose the plain-lanes paper, with S14's order oracle as its foundation.

## 1. Frank assessment: not at the A* main-track bar yet

What exists:
- plain lanes, exact likelihood, trained from scratch;
- bpb within 1% of dense-1x at 2 to 5x training tokens (d4 L=64 0.993 / 0.992; L=32 at 2x 0.994; d8 L=64 at 5x about 1.010);
- 28x (L=32) and 53x (L=64) decode speedup at batch 1 in the CUDA-graph decoder.

Gaps, most serious first:

1. **The scale trend points the wrong way.**
   - All evidence is at d4 and d8 (11.5M and 41.9M scaling parameters).
   - The equal-token tax grew from d4 to d8: L=64 6.3 → 7.8%, L=32 3.9 → 4.9%.
   - Per-lane loss in absolute nats also grew, about 6.6 → 7.1 (S13's 1.93 → 2.38 average-token losses).
2. **Parity is bought with training tokens, and the multiple can grow with scale.** See §2.
3. **The speedup is batch-1 latency.**
   - At batch 16 and above, rows set the cost (S13 Q3), so fewer rounds stop paying.
   - There is no iso-quality baseline yet: a smaller dense model with the same bpb, or speculative decoding.
4. **Samples.** At d4, lane samples had 2.1 to 2.6x the reference perplexity of dense-1x samples (S11). Not shown resolved at scale.
5. **Protocol and novelty.**
   - Nothing is on the field's protocol yet: OWT, GPT-2 tokenizer, about 110M parameters, against AR, MDLM and BD3-LM at matched decoding steps (NFE).
   - PAR (2412.15119) generates distant image regions in lockstep, so plain lanes alone reads as a layout. A mechanism is needed.

What S14 adds to the paper:
- Under two 8B any-order oracles, lanes lie below the cost-vs-steps curve of every other parallel order: bisection, random (diffusion-style) and sentence-snapped.
- The tax splits into information (TC, about 1 nat per lane at L=64) and learnable cost.
- At d8 about 85% of the per-lane loss is learnable: about 7 nats against the 8B oracle's about 1.7.

Odds of reaching A* along this plan: about 20 to 30%, decided mostly by R1.

## 2. Is equal training compute the right yardstick? (user question, 2026-10-05)

Equal training compute is not the only fair comparison.
- Lane starts are genuinely harder conditionals, and training is paid once while decoding is paid per request.
- Accepted work routinely buys inference speed with training: speculative-decoding drafters, multi-token prediction heads, distillation, overtraining.
- Lanes add no FLOPs per token, so k x tokens is exactly k x training compute, which is easy to report.

What reviewers will ask is different:

1. **Does k stay bounded at scale?**
   - Parity needs k = 2^(tax / g), where g is dense's relative gain per doubling of tokens.
   - At d4 and d8, g is about 3%, so a 6 to 8% tax needs 4 to 5x.
   - At larger scale g shrinks toward 1 to 1.5% per doubling. A tax that stays near 5% then needs about 10x; a shrinking tax keeps k modest.
2. **Is lanes faster at matched quality and matched training budget?** The alternatives are:
   - a smaller dense model with the same bpb, itself 2 to 4x faster at batch 1;
   - dense plus speculative decoding, lossless, about 2 to 3x;
   - BD3-LM.

   A large margin over the best of these (10x or more at batch 1) makes the training cost a fair price.

## 3. Stage L0a: the decisive scale test (about 8 to 9 H100-hours)

Everything below uses existing code.
- `s11_ladder` trains and scores at any depth.
- `lane_offset_report` now prints "extra nats per lane" and "reference nats per token" in absolute nats, so model sizes, and the S14 oracle, are comparable.

```
modal run modal_sap.py::s11_ladder --depth 12 --specs dense:1:1,dense:2:1,ln:64:1:1,ln:32:1:1 --name s15_d12
modal run modal_sap.py::s11_score --depth 8 --tags S11dense_x1_s1,S11ln64x1_s1,S11ln32x1_s1 --name s15_d8
modal run modal_sap.py::s11_score --depth 4 --tags S11dense_x1_s1,S11ln64x1_s1,S11ln32x1_s1 --name s15_d4
modal run modal_sap.py::s08_gen --depth 12 --tags S11ln64x1_s1:64,S11ln32x1_s1:32 --ar-tag S11dense_x1_s1 --ref-tag S11dense_x2_s1 --gen-tokens 1985
modal run modal_sap.py::s11_speed --ckpt S11ln64x1_s1 --depth 12 --schedules ln:64,ln:32
```

- d12 is 110M scaling parameters and 1.16B tokens at 1x, about 1.0e18 FLOPs, or about 1 H100-hour per dense-1x run.
- The rescoring at d4 and d8 is eval-only.

**Readings, pre-registered.**

- **R1, decides.** Extra nats per lane, L=64, 1x tokens: d12 against d8, both from the updated report.

  | outcome | meaning |
  |---|---|
  | ≤ 0.90 x d8 | **go**: scale moves the cost toward the S14 floor |
  | ≥ 1.05 x d8 | **no-go for plain lanes as the core**: only a mechanism that removes the learnable part can carry a paper, and it must pass its own offline gate first |
  | in between | one d16 pair (dense + ln64 at 1x, about 10 H100-hours) decides |

- **R2, reported.**
  - Block tax % at 1x for L=64 and L=32 (d4 6.3 / 3.9, d8 7.8 / 4.9).
  - The parity multiple k = 2^(tax / g), with g from dense-1x against dense-2x at d12.
- **R3, samples.** Reference perplexity of lane samples over that of dense-1x next-token samples at matched unigram entropy, scored by dense-2x. Parity bar ≤ 1.2x.
- **R4, speed.** Lane tokens/s over next-token tokens/s at batch 1, 16 and 64 (H100, CUDA graphs).

## 4. Stage L0b: the S14 oracle at the paper's block (offline; about 5 H100-hours; parallel with L0a)

```
modal run modal_sap.py::s14_order_oracle --block 1920 --sep-cut 0 --rows 16 --name t1920 \
    --orders l2r,lanes32,lanes64,lanes128,bl32_8,random30,conf30,conf60
```

- `conf{R}` is confidence-ordered decoding, the LLaDA/Dream default: each step reveals the most confident ceil(hidden / steps left) positions.
- `bl32_8` is S11's bridged lanes.

**Readings, pre-registered.** Printed in the JSON's `readings`.
- TC and gap per lane and in %, for L = 32, 64 and 128 at T = 1920: the floor in R1's units, and the choice of L.
- **Lanes against diffusion decoding at equal steps: `lanes{L}_beats_{conf|random}{R}` when lanes' total is ≤ 0.5x.** Pairs:
  - lanes64 (31 steps) against conf30 and random30;
  - lanes32 (61 steps) against conf60.

## 5. Later stages, only after an R1 go

- **L1. A mechanism for the learnable lane-start cost.**
  - A fresh pool of 30 to 50 ideas, filtered offline against the S14 per-lane numbers before any training.
  - The iso-quality speed baseline: `s03_dense_frontier --depth 12` (shallower dense at matched FLOPs, timed), plus speculative decoding on dense.
- **L2. Paper protocol.**
  - OWT, GPT-2 tokenizer, about 110M parameters (350M check).
  - Against AR, MDLM and BD3-LM at matched NFE: likelihood, generative perplexity at matched entropy, and wall-clock at batch 1 / 16 / 64.

## 6. Verification rules

- R1 compares numbers from the same updated report at every depth. Reference tags and token budgets are named in every ratio.
- Each model is scored in its own order, and its val bpb is checked against training (±0.5%).
- Every per-position JSON must print the row hash `ae01832e4bf20262`: the same 256 rows and tokenizer at every depth.
