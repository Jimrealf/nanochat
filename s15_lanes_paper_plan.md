# S15: can plain lanes reach the A* bar?

Status 2026-10-06: Stage L0 is done (§7).
- R1 is in between (0.944), so a d16 tiebreak is set up.
- R3 fails as run.
- Speed and the equal-step oracle comparison are strong.
- The learnable gap is lanes' recovery, not their lane starts, so L1 targets recovery (`s16_lanes_recovery_brainstorm.md`).

Opened 2026-10-05, after S14 closed the strict thesis on information (`s14_sap_strict_tl_brainstorm.md` §11). The user chose the plain-lanes paper, with S14's order oracle as its foundation.

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

## 7. Stage L0 results (2026-10-06)

All numbers below are from `scratch/s15/`.
- Rows hash `ae01832e4bf20262`: 256 rows, block positions 128 to 2047, one seed per arm.
- Oracle rows hash `6bcd550836218394`: LLaDA-8B-Base, 16 rows of 128 + 1921.

I checked the bundled `s15_stage_l0_comprehensive_report.md`: its numbers match the files, but its diagnosis and parts of its framing do not (§7.3).

### 7.1 Readings

| reading | result | verdict |
|---|---|---|
| R1: extra nats per lane (L=64, 1x), d4 / d8 / d12 | 7.408 / 7.562 / 7.136; d12 / d8 = **0.944** | in between: the d16 tiebreak (§7.4) applies |
| R2: block bpb, dense-1x / lanes64-1x | d4 1.1232 / 1.1946 (+6.36%); d8 0.9346 / 1.0076 (+7.81%); d12 0.8367 / 0.9058 (**+8.26%**) | — |
| R2: lanes32-1x | d4 1.1662 (+3.82%); d8 0.9805 (+4.92%); d12 trained to step 2204/2205 but not scored | — |
| R2: parity multiple at d12 | dense-2x 0.8049, so 3.80% per doubling; k = 2^(8.26 / 3.80) ≈ **4.5x** (about 4 to 5x at d4 and d8 too) | — |
| R3: samples at d12, temperature 1, scored by dense-2x | reference ppl next-token 56.78, lanes64 87.82 (**1.55x**); unigram entropy 6.100 and 6.158; distinct 3-grams 0.962 and 0.979; real text 14.27 at entropy 5.881 | **fails ≤ 1.2x as run**; at real-text entropy, not measured yet (§7.4) |
| R4: tokens/s ratio to next-token, H100, CUDA graphs, 1920 tokens | L=64: **54.3x / 30.7x / 16.4x** at batch 1 / 16 / 64; L=32: 28.5x / 17.9x / 10.7x | strong at d12 |
| L0b: lanes64 against random30 at 31 steps | total 2.42% against 9.09% (**3.8x lower**) | `lanes64_beats_random30` |
| L0b: lanes64 against conf30, lanes32 against conf60 | 2.42% against 84.6%; 1.15% against 72.7% | beats, but conf is a weak baseline (§7.3) |
| L0b: floor at T = 1920 | lanes32 TC 0.51%, total 1.15%; lanes64 TC 1.41% (0.94 nats per lane), total 2.42% (1.61 nats per lane); lanes128 TC 3.39%, total 5.11% | — |
| L0b: bridged lanes (32, 8), 101 steps | TC **0.07%**, total 0.86% | the most information-efficient structure measured |
| E1 at d8 (moot after S14) | window bisection n=1 +28.5% (d4 +24%); n=16 +7.0% (d4 +7.8%) | — |

**Iso-quality point.** Lanes64 at d12 (0.906 block bpb) beats dense at d8 (0.935), while decoding 54x faster than d12's own next-token decoder at batch 1.

### 7.2 Where the per-lane cost sits: deficit against recovery

Approximate, from the offset-group ratios times reference nats per token. The oracle's figures use par minus the block-average l2r. The exact per-offset values come from the new report fields (`deficit` / `recovery nats per lane`, `lane_profile`).

| L=64, nats per lane | deficit (offsets 0-15) | recovery (offsets 16-29) | net |
|---|---|---|---|
| d4 | 9.0 | −1.8 | 7.3 |
| d8 | 11.4 | −4.0 | 7.4 |
| d12 | 11.9 | −5.0 | 6.9 |
| LLaDA-8B oracle, T = 1920 | 11.3 | −9.7 | 1.6 |

**The deficit is information.**
- Early offsets are decided with no left context in their lane.
- The trained models' deficit stopped growing (+0.5 from d8 to d12) and already matches what an 8B any-order model pays.

**The learnable gap is recovery.**
- Late offsets read the next lane's early tokens (lookahead).
- d12 recovers 5.0 nats per lane; the oracle recovers 9.7.
- Recovery grows with scale (1.8 → 4.0 → 5.0), which is the turn in R1.

**Consequence: the L1 mechanism must raise recovery.** A lane-start inductive bias targets the part that cannot shrink.

### 7.3 Corrections to the bundled report

1. **"Lanes crush confidence decoding by 35 to 63x."**
   - True as measured, but the baseline is pathological: global confidence unmasking of 64 tokens per step over a fully masked 1920-token span.
   - Its same-step TC stays at 47 to 164 nats per step to the last step, as hard spans cluster and are unmasked together.
   - LLaDA is run in semi-autoregressive blocks in practice.
   - The robust claim is against random-order decoding, 3.8x at equal steps. Report conf as "naive global confidence decoding".
2. **"An inductive bias at lane starts closes the gap."** No: the deficit is at the information level (§7.2), and the gap is recovery.
3. **R3 fails its pre-registered bar as run** (1.55x > 1.2x). The report leaves this out.
   - The S11 d4 result (parallel samples better at real-text entropy) came from a d4 next-token sampler that loops.
   - At d12 the next-token sampler no longer loops (distinct 3-grams 0.962), and lanes sample worse.
4. **Batch 16 and 64 speedups are a small-model regime.**
   - d12 is latency-bound, so 64 rows per step cost about as much as one.
   - At 7B the batch-64 gain should shrink (not measured).
   - My own earlier "the advantage mostly disappears at batch ≥ 16" was wrong for d12.
5. **The proposed title overclaims.** The 54x is batch 1 on 110M parameters, with an 8.3% equal-token bpb tax.

### 7.4 Next runs, readings pre-registered before running

```
modal run modal_sap.py::s11_ladder --depth 16 --specs dense:1:1,ln:64:1:1 --name s15_d16                        # ~9 H100-h
modal run modal_sap.py::s08_gen --depth 12 --tags S11ln64x1_s1:64 --ar-tag S11dense_x1_s1 --ref-tag S11dense_x2_s1 \
    --gen-tokens 1985 --temperatures 0.85,0.9,0.95                                                                # ~1.5 H100-h
modal run modal_sap.py::s14_order_oracle --merge-only --block 1920 --sep-cut 0 --rows 16 --name t1920            # CPU, minutes
modal run modal_sap.py::s11_score --depth 12 --tags S11dense_x1_s1,S11ln64x1_s1 --name s15_d12b                 # exact deficit/recovery
modal run modal_sap.py::s11_score --depth 8 --tags S11dense_x1_s1,S11ln64x1_s1,S11ln32x1_s1 --name s15_d8b
```

- **d16, the R1 tiebreak.** Extra nats per lane (L=64, 1x):

  | outcome | meaning |
  |---|---|
  | ≤ 6.81 (0.90 x d8) | go |
  | ≥ 7.14 (no decline from d12) | no-go for plain lanes as the core |
  | in between | the scale trend alone is too slow; L1 decides |

- **Mechanistic prediction at d16.**
  - The deficit stays within ±5% of d12's (exact value from `s15_d12b`), and recovery grows past d12's.
  - If the deficit keeps rising instead, the information-level reading of §7.2 is wrong.
- **R3 at real-text entropy.** Interpolate log reference ppl to entropy 5.88 for both samplers.
  - Lanes / next-token ≤ 1.2x passes.
  - Above 1.5x fails.
  - In between is reported as is.
