# S08: lanes, emitting T distant tokens per pass (strict SAP seed)

> 2026-10-03. This brainstorm and its experiments come from a separate session from the parallel `s07_sap_mechanism_brainstorm.md`, which targets T=L=2048. Here the user set the scope to "strict SAP seed only" and the bar to "both" (neutral against same-FLOPs dense at 2x or more, plus a clear margin over the depth frontier), and approved this plan. Code: `nanochat/lanes.py`, `tests/test_lanes.py`, `scripts/base_train.py --lanes`, `scripts/sap_decode_bench.py --lanes`, `modal_sap.py::s08_lanes`. The first d4 launch used checkpoint tags `S07_*` (before the rename to S08).

## Context

The user asked to go back to the drawing board, account for everything learned, and keep strictly to the SAP seed:
- one model emits T jointly coherent tokens per trunk pass;
- no verification, no teacher or distillation;
- trained from scratch.

Bar (user decision, 2026-10-03), **both** of:
- bpb within about 1% of the same-FLOPs dense model at 2x or more decode speed;
- a clear margin over the dense depth frontier.

The d8 frontier for reference:

| layers | speed | C |
|---|---|---|
| 6 | 1.28x | 1.008 |
| 4 | 1.79x | 1.026 |

The SAP work so far comprises:
- S00 to S06 (one-pass heads, mid-pass sampling with reference modules, fields, permutation-state scans);
- v4 (exact cut heads, lattices, corpus tables, soft targets);
- the trunk-depth family: depth_local, depth_roll, skip-middle, depth_tree, adaptive length.

All of it is closed with measured reasons.

**The one reinterpretation this plan makes, flagged for the user:** every SAP design so far emitted T *adjacent* tokens per pass. The lead survivor emits T *distant* tokens per pass: lanes of the same document, advancing in lockstep. The reason is below (constraint K1). The seed's other requirements all hold: one model, T tokens per trunk pass, unverified, from scratch, exact likelihood. If this drift is unacceptable, the strict-seed brainstorm has no survivors.

## Constraints from measurements (the filter)

| | constraint | evidence |
|---|---|---|
| K1 | Adjacent-token dependence is a property of the data, and large | TC = 19 / 47 / 66% of AR block NLL at T = 2 / 4 / 8 (Stage 0 on dense d8). No head or objective can remove it; it can only be modelled. |
| K2 | Per-slot losses cannot create coherence | Proved corollary. Soft targets: Stage A KL 2.92 vs 3.01. Soft inputs (`sir_soft`): KL 3.8 to 4.3. |
| K3 | Heads reading final trunk states hit a capacity ceiling | The exact local chain saturates at 1.17 to 1.21x for any head size. Cut + NCE: 1.29 to 1.34x frozen, 1.20x co-trained. |
| K4 | Non-left-to-right in-block orders are expensive even at full depth | Bisection with dedicated full-depth copies: 10% (T=4) and 16.5% (T=8). |
| K5 | Noise-tape and one-random-round generators are not learnable on the gate | S04 KL 1.71; S05 3.35; S06 PSS 2.95 (its oracle ceiling was 0.015). |
| K6 | Shallow paths bolted onto deep models lose to shallow models trained end to end | The trunk-depth family sits 3 to 7 points of C above the frontier at d4 and d8. |
| K7 | Causal routers cannot find where a cheap path fails | Must reject about 70% of slots to reach 1.02. Even an oracle router gives 1.5x at ratio 1.0. |
| K8 | Depth is cheap to remove at these budgets | Frontier: d8 at half depth costs +2.6%. Any speed claim must beat it. |
| K9 | Reference PPL amplifies bpb gaps about 9x | The bpb bar must be strict. |

## Funnel: 42 candidates, 39 killed, 3 survive

| kill reason | n | candidates |
|---|---|---|
| K1 or K2 (product of marginals over adjacent tokens) | 6 | independent MTP heads; soft targets and soft inputs; deterministic drafts; linear-conditioning in-block RNN; Gumbel coupling; early-exit MTP |
| K3 (final-state capacity) | 5 | lattice CRF / TT / CP heads; cut + reference fill; NCE resampling as lead; latent plan (P1, self-posterior); copy-span head |
| K4 (order tax) | 5 | cut + trunk-depth fill; mask-slot full-depth CRF (NAT-CRF style); coarse-to-fine anchors; strided or interleaved lanes (each token loses its left neighbour); LDPC-style parallel decoding |
| K5 (learnability, or likelihood-free so no bpb) | 4 | noise-tape PTP from scratch; CALM-style continuous chunk autoencoder with energy head; discrete chunk VQ codes (lossy, so not an LM); Swendsen-Wang cluster sampling |
| K6 or K8 (frontier) | 6 | depth_local, depth_roll, skip-middle, depth_tree; SAP on a shallow-wide base; two-rate fast/slow (outside strict scope) |
| K7 | 2 | adaptive block length (confidence routers, learned router) |
| Boundary (verification, rounds or teacher) | 5 | Jacobi / CLLM; speculative trees; Parareal; planner + diffusion fill; self-speculation |
| Speed arithmetic | 3 | learned phrase units / segmental LM (SuperBPE gets 1.5x at a 200k vocab, so 2x is out of reach); multi-token input units; dynamic vocabulary of n-grams |
| Knob of a survivor, not a mechanism | 3 | staggered lanes; lane headers; hierarchical lane depth |

## Survivors

All three choose *which* T tokens share a pass so that their dependence is near zero. That attacks K1 at its source instead of modelling it.

### S1 (lead): fixed lockstep lanes

**Mechanism.**
- A training row is a prefix of P tokens (causal, the "prompt"), followed by L contiguous lanes of S tokens each.
- At step s, every lane emits its s-th token in parallel. A lane token at step s sees the whole prefix, every lane's tokens from steps before s, and itself, but no other step-s token.
- Each lane's first prediction comes from a learned lane-start vector at position jS-1. That input slot is otherwise lane j-1's last token, which never serves as an input in this order.
- RoPE uses the true positions.
- The likelihood is exact: a product of softmaxes, with independence only *within* a step, between tokens S apart.

**Expected tax.**
- It is concentrated at lane starts: lane j begins jS tokens beyond the context it can see.
- Lane j-1's tail sees lane j's head, so the junction behaves like infill.
- Back-of-envelope: about 17 nats per lane start against S × 3.6 nats per lane gives about 0.5% at S=1024 and about 0.9% at S=512.
- So the tax should scale as L/N.

**Cost.**
- Training FLOPs are the same as dense; only the mask changes.
- Decode: one trunk pass per L tokens. That is about L x at small batch for long generations (N >= L·S).

**Bar.** L=4 at <= 1% beats both the same-FLOPs dense model and the frontier (about 3.8x against 1.79x at +2.6%).

**Kill.**
- Lane tax at L=2 (S=1024) above 1.0% at d4 (two-seed means).
- Or the tax does not fall with S, meaning the cost is not a lane-start effect.

**Prior work and delta.**
- Prior work:
  - Hogwild! Inference (2504.06261): an inference protocol on pretrained models.
  - Multi-Stream LLMs (2605.12460): role streams, instruction-tuned.
  - Parallel Decoder Transformer (2512.10054): frozen trunk, planner, no results yet.
  - ReFusion (2512.13586): a diffusion LLM with slot-level AR.
  - Planned Diffusion (2510.18087): planner plus diffusion, measured by win rate.
  - APAR and Skeleton-of-Thought: fine-tuning or prompting.
  - Subscale WaveRNN and Reed 2017: the same principle for audio and images.
- Delta: the first exact-likelihood lane factorization pretrained from scratch on raw text. bpb is measured against same-FLOPs dense and the depth frontier, with no planner, diffusion, fine-tuning or verifier, and the tax is characterised as about L/N.

### S2: paragraph-aligned lanes (conditional on S1's tax being 1 to 2%)

**Mechanism.**
- Lane boundaries fall on paragraph starts (`\n\n`), so lanes start at topic shifts rather than mid-sentence.
- Lanes are variable length, with an end-of-lane token.
- Virtual positions jR + s use a fixed stride R, so lane offsets are known at decode.

**Cost.** Same as S1, but decode steps equal the longest lane.

**Kill.** Tax not below S1's at equal L.

### S3: forking lanes (conditional on S2)

**Mechanism.**
- A lane may emit FORK at a paragraph boundary, spawning a new lane at the next virtual stride. Training places forks at randomly chosen paragraph boundaries.
- Parallelism grows with document length, under the model's own control.

**Cost.** Decode needs lane padding to a maximum L for CUDA graphs.

**Kill.** Forks rarely predicted, or tax above S2.

## Staged execution (pre-registered; cheap first; jimpearse01, `tokenizer_sap`)

**L1, d4 lane tax (about 10 H100-minutes).**
- Lanes L in {2, 4, 8} (S = 1024 / 512 / 256 at sequence length 2048), two seeds each.
- Prefix P ~ U[0, 256].
- Baseline: a dense model with the same window pattern. Use `L` everywhere for both arms, so lane masks need no sliding windows, and retrain the dense d4 with `L`, two seeds.
- Report lane-order val bpb over dense bpb, and fit tax against 1/S. Apply the kill criteria above.
- 1% < tax(4) <= 2%: run S2 before any d8 spend.

**L1 outcome (d4, 2026-10-03): passed.**
- Dense, full-context attention: 1.1522.
- Tax by lane count (S in brackets): L=2 (960) +0.47%, L=4 (480) +0.94%, L=8 (240) +1.57%, two seeds each.
- The tax grows with L, as a lane-start effect should.

**L2, decode.**
- Lane decoder under CUDA graphs: 2048-token generations at batch 1 and 16, against AR on the same model.
- Gate: at least 0.8·L x.
- Also report the same model's reference PPL for lane samples against its AR samples.

**L2 outcome: passed.** Batch 1 / 16 with CUDA graphs: L=2 1.89x / 1.67x, L=4 3.66x / 3.24x, L=8 7.19x / 6.23x. L=8 at batch 16 is just under the 0.8·L gate. Generation quality (d4, `scripts/sap_eval_generation.py --lanes`, `scripts/sap_lane_gen_profile.py`):
- Lane interiors match or beat the next-token sample range for range. Next-token samples degenerate into repetition over 1921 tokens, which lowers their reference PPL.
- The cost is at lane seams: the first 16 tokens of each later lane score reference PPL 420 to 650, against about 60.
- That moves S2 (paragraph-aligned lanes) from conditional to the next mechanism, if lanes continue.

**L3, d8 confirmation (about 1 H100-hour).**
- L in {2, 4}, two seeds, against the measured d8 frontier.
- "Both" is cleared at tax(4) <= 1% with speed >= 3x.

## Implementation (done 2026-10-03)

**`nanochat/gpt.py`**
- `CausalSelfAttention.forward`: add a `lane_mask` branch beside the existing custom-mask SDPA path (the EET `token_active` branch around line 8210).
- `GPT.forward`: accept a lane spec (L, S, P), build the boolean step mask, and swap in a learned `lane_start` vector at lane-start inputs.

**`nanochat/engine.py`**
- `generate_lanes`: per step, run the L new tokens through all layers with the KV cache as prefix and identity slot visibility, then write their K/V into the cache. This reuses `GPT._sap_depth_layers` (prefix-KV callback, slot-visibility mask, `kv_out`, positions) with m = L, as the `depth_tree` decoder already does.

**Training, benchmarking and launch**
- `scripts/base_train.py`: `--lanes L --lane-prefix-max P`, with val bpb in lane order.
- `scripts/sap_decode_bench.py`: `--lanes`.
- `modal_sap.py`: an `s07_lanes` entrypoint reusing `s03_train`, `s03_post`, `_dense_flops` and `s03_dense`.

**Tests (extend `tests/test_block_head.py` or add `tests/test_lanes.py`)**
- L=1 gives exactly the standard causal loss.
- Lane-order likelihood sums to 1 by enumeration on a tiny vocabulary.
- No leak: changing step >= s tokens leaves step-s predictions unchanged.
- The greedy lane decoder equals the argmax of the training conditionals.

**Literature, downloaded into `Literature Review/`** (IDs verified at download; each under 10 MB): 2512.10054, 2512.13586, 2510.18087, 2504.06261, 2605.12460, the parallel-generation survey 2508.08712, Skeleton-of-Thought, APAR, Subscale WaveRNN 1802.08435, Reed 2017 parallel multiscale AR, σ-GPT.

**Docs**
- This file is the S08 plan. `sap_research_plan.md` is left to its current scope note (the parallel S07 session). v4 is also archived as `sap_research_plan_v4_archive.md`.
- LEARNINGS gets the funnel and constraints; OPEN_QUESTIONS gets Q33 (lanes); PROJECT_MAP gets the lane code.

## Verification

- Run `pytest tests/test_block_head.py` with the tokenizer test deselected, then check `git status tokenizer/`.
- Smoke runs:
  - `base_train` at d2 with `--lanes 2` for a few steps;
  - `sap_decode_bench --smoke --lanes 2` with graphs on the local GPU.
- Each stage reports two-seed means against the 0.13% d4 seed noise, before any further spend.

## Outcome (2026-10-03): parked as the T=K fallback

- d8 (two seeds): tax +0.61% at L=2, +1.10% at L=4; speed 1.90x / 3.74x on batch 1, 1.48x / 2.91x on batch 16.
- The user then confirmed that T=L (one fixed computation for the whole block) is the requirement. Lanes reduce passes to N/K and cannot reach that.
- This session moved to the T=L gates in `s09_sap_tl_gates.md`.
