# S14: the strict seed, T=L in a fixed number of levels

**Status 2026-10-05: CLOSED on information.** E0 and E2a ran in both oracles, and every validity check passed (§11).
- Exact token orders of 13 levels or fewer lose 13 to 15% to same-step dependence alone, against a 1% bar.
- All five survivors are dead, not live, or disfavoured.
- E1 and Stage 1 are not run.

The work continues as the non-strict lanes paper (`s15_lanes_paper_plan.md`).

## 0. Changes from the approved plan, and why

- **E2 is now two steps.**
  - E2a is an information bound computed inside the E0 job, for about 10% of E0's cost. It covers every possible code of B bits, learned or not.
  - The planned training run (d4 with k-means product-VQ codes of the d8 dense state) tested only one code family. That family already failed on the toy: k-means codes were 77% invalid against 6.9% for oracle codes.
  - The training run is now E2b. It runs only if E2a leaves a separator of 48 bits or fewer possible.
- **E1 needs no new code.** `s11_ladder` at depth 8 is protocol-matched to the d4 window-bisection runs (same FLOPs target, prefix, eval rows and edges). `s11_wbisect` would have needed a d8 dense reference it does not have.
- **E0 gained three things.**
  - A right-context probe: a causal mask left on by mistake would leave the l2r score looking normal while taxing every other order.
  - A check of the detected logit shift against the known one.
  - Random orders drawn per row, plus sharding over GPUs with per-row resume.

## 1. The strict thesis and its bar

**Thesis.** The seed plus the S11 and S13 user decisions:
- one model emits a 1920-token block in one fixed computation of at most 13 sampling levels (log2 T + 2);
- every position in a level is drawn in parallel;
- no verification;
- trained from scratch, with no teacher or distillation for the generator;
- exact likelihood or an honest upper bound on bpb.

**Bar.** CLAUDE.md "performance neutral plus a speedup", in numbers:
- bound bpb at most 1.01x dense-1x, at no more than 4x training tokens;
  - references: `S09_dense_L_s1/s2` at d4 and `S11dense_x1_s1` at d8, same V=32,768 vocabulary;
- at most 13 network levels for 1920 tokens;
- measured decode at least 20x faster at batch 1 and 5x at batch 16, in the existing CUDA-graph decoder.

The paper then moves to the OWT / GPT-2 tokenizer / ~110M protocol, the one MDLM and BD3-LM (ICLR 2025 oral) were accepted on.

## 2. Balance sheet: facts any strict mechanism must satisfy

| id | measured or proved fact | source |
|---|---|---|
| K1 | Adjacent tokens drawn in one round lose 19 / 47 / 66% of block NLL at T = 2 / 4 / 8 | S08 |
| K3 | Heads reading final trunk states saturate at 1.17 to 1.21x, at any head size. DCL (8 tokens per row): 5.02 against 3.14 nats | S08, S13 |
| K4 | Non-left-to-right orders cost even at full depth: 10% / 16.5% at T = 4 / 8 | S08 |
| RC0 | Only teacher-forced exact factorisations learn. Latents sampled, inverted or posterior-drawn in training do not | S11 |
| T2 | A one-pass generator must carry long memory of its own output: a last-32 suffix costs 7.5 to 8.7% | S09 |
| SEP | Far context crosses a summary for free. Near context needs per-position verbatim access: 7 to 8% in the first 256 tokens, partly a one-layer delay. Copying needs token identity | S11 stage 1 |
| WB | Window bisection at d4, 1x tokens: n=1 **+24%** (12 steps), n=4 +10.8% (40), n=16 **+7.8%** (126), n=64 +9.5%. Even the l2r prefix paid 1 to 10% | S11 3a |
| TOY | Bridge LM with oracle separator codes at T=64: 24% → **6.9%** invalid with 3x steps. k-means codes 77%, tokens 81%. Window-bisection KL halves per doubling of steps | S11 stage 2 |
| BLIND | A variable decided blind costs about what it later tells: splice codes removed 0% of the tax | S13 Q1 |
| LANES | Tax is proportional to L/N and flat in training tokens. 65 to 87% of each lane-start deficit is never recovered, though an ideal model recovers all but same-step TC. Recovery grows with size, 20% → 35% from d4 to d8 | S11, S13 |
| NC | Deterministic one-step maps cannot commit before a sharp categorical readout; the escapes are commit-per-level or noise re-injection. SV-D one-pass MeanFlow: PPL 2893 against 26 | arXiv 2606.30705; S13 |
| AE | 8 tokens → one 256-d latent at 100% / 99.93% accuracy (sigma 0.5) | S13 Q2 |
| RF | At batch ≥ 16, 8 rounds are within 1.3x of one pass. At batch 1, one pass is about 4x faster than 8 rounds | S13 Q3 |
| TH | Constant steps can match perplexity but not sequence-level correctness (Feng et al., NeurIPS'25). TC0 backbones cannot do inherently serial work at any step count (Serial Scaling Hypothesis). Backward and non-causal prediction is consistently harder (Arrows of Time, ICML'24) | literature |

**Chain rule.** For any exact order:

NLL = H(X) + Σ over steps of TC(the step's draws | past) + learning/computation gap.

## 3. The decisive unknown

Every measured strict-order tax mixes two terms that nobody has separated on real text: window bisection costs 7.8 to 24%, lanes 3.9 to 10.7%. The two terms are:
- **same-step TC:** information lost by drawing tokens together;
- **the learning/computation gap:** how hard the order's conditionals are to learn and compute.

What each outcome means:

| outcome | meaning | mechanisms |
|---|---|---|
| Small TC, gap shrinking with capacity | The strict thesis is an architecture problem | S14-A, S14-B |
| Large TC | Token decisions are dead; only decision variables that separate can work | S14-C, S14-D, S14-E, gated on E2 |
| Gap large even for an 8B any-order model | The serial "arrow" binds, and exact strict orders are closed with a measured reason | — |

E0 measures both terms offline in a few H100-hours.

## 4. Pool and funnel: 53 candidates → 5 mechanisms + 3 ingredients

| family (n) | candidates | fate |
|---|---|---|
| reversible flows (5) | discrete bipartite flow (Tran'19), argmax flow, categorical spline flow, integer flow, flow over chunk latents | killed: S09 measured that flows learn little beyond marginals; T3/T4 |
| one-step continuous (5) | MeanFlow (SV-D), consistency/shortcut from scratch, FMLM flow map, Coupling Models, SDE chunk latents | killed by the strict bar (samples only, no tight bound), NC and RC0. FMLM is distilled; Coupling Models (May'26) already owns "one-step from scratch" |
| noise inversion (3) | PTP / RC-PTP, shared-noise monads, coupled-noise tickets | C1 (about 2.5 tokens per call) |
| refinement (5) | Jacobi/CLLM, Parareal, checkerboard Gibbs, predictive-coding settling, coupling from the past | C2 (one position per stage), verification |
| finite state / scan (4) | random-map HMM scan, 2^16-state neural HMM, MERA with finite bond, carry-select segment scan (draft each segment for every incoming boundary state, then pick by associative scan) | T2 / SEP: needs separators of ≤ 16 bits; E2a decides |
| flat plans (4) | global plan (S04/P1/P2), de Finetti latent, holographic plan, bag-of-content + competitive queuing | measured (KL 1.7, +31 to 34%) / flat TC |
| token-decision orders (5) | single-token bisection, window bisection, strided lanes, insertion-transformer slots, random-order few-step | WB measured; strict versions gated on E0; strided lanes killed by K4 |
| within-round joints (4) | chain CRF per level, DCL rows, lattice/TT/CP heads, NAT-CRF | K3 / DCL measured |
| decision codes (6) | k-means codes, Brown-class skeleton, blind splice codes, **learned separator codes**, bidirectional plan codes, **multi-scale residual codes** | measured kills for the first three; the two in bold survive (plan codes merged in as an arm) |
| order design (4) | **boundary-snapped bisection**, **pyramid capacity**, **per-level coordination variable**, oracle-driven schedule search | the three in bold survive; schedule search becomes a tool inside E0 |
| training side (4) | multi-horizon planning loss, interval coordinates, AR multi-task, level-weighted loss | ingredients; level weighting dropped (reweighting adds no information) |
| other fields (4) | kinetic proofreading, Hayek prices, counterpoint, V(D)J junctions | map to verification / coordination variable / skeleton / splice codes |

## 5. Survivors

Each survivor states its mechanism, levels, cost, bar, pre-registered kill, and nearest work with the delta.

### S14-A. Pyramid-capacity bisection (PCB): the lead if E0 says "gap"

- **Mechanism.** The existing exact window-bisection order (two-stream, `nanochat/wbisect.py`), with capacity routed by decoding level:
  - each MLP gets E = 4 level-experts, routed deterministically by the query's level bucket;
  - the l2r prefix gets its own expert;
  - optionally, coarse queries (≤ 6% of positions) pass through k extra planner layers.
- **Why.** WB measured capacity contention, and recovery grows with model size.
- **Levels.** n=1 gives 12 levels (strict). n=16 is read for the mechanism only.
- **Cost.** About two-stream dense FLOPs plus ≤ 10%. MLP parameters x4 at matched FLOPs; both are reported.
- **Bar.** The n=1 tax goes from +24% to ≤ +5% at d4 at matched tokens, then ≤ 1.01x at ≤ 4x tokens on d8.
- **Kill.** Either of:
  - less than a 40% reduction of the n=1 and n=16 taxes at matched FLOPs (d4, two seeds);
  - a prefix penalty above 0.5%.
- **Nearest work.**
  - Hourglass and Funnel: compute allocated by resolution, but in l2r order.
  - VAR: one shared network across scales.
  - Shih et al., NeurIPS'22: any-order AR redundancy.
- **Delta.** Capacity is allocated by position in an *exact parallel* order. The claim is that the order tax is capacity contention, which can be removed without changing the factorisation.

### S14-B. Boundary-snapped bisection (BSB)

- **Mechanism.** Each bisection anchor moves from the dyadic midpoint to the nearest natural boundary within ±W: document start, then paragraph, then sentence end. The S12 oracle found document-start starts free and sentence-end starts about 40% cheaper.
  - The offset is itself a scored decision: a categorical over ±W, or "none", which means the midpoint.
  - In the same pass, queries at all 2W+1 candidate positions give the token at whichever offset is drawn.
  - Intervals become variable length with no padding. S12's aligned lanes died of 14 to 16% pad waste.
- **Levels.** 12, one pass each.
- **Cost.** Per-row two-stream masks and a negligible offset head.
- **Bar.** At least 25% lower block tax than midpoint bisection at equal levels (d4, n = 1 and 4).
- **Kill.** Any of:
  - E0 shows that snapping lowers TC + gap by less than 15%;
  - the d4 gain is below 10%;
  - the offset decisions cost more than 0.5% bpb.
- **Nearest work.** Insertion Transformer (slot plus token, log-n insertion, translation with distillation); S12 aligned lanes; SoT and APAR.
- **Delta.** An exact-likelihood pretraining order whose separators sit where text separates itself, with variable intervals instead of padding.

### S14-C. Learned-separator Bridge LM (LSB): the lead if E0 says "information" and E2a finds small separators

- **Mechanism.** Per-position decisions (token, code) in bisection order, following the S11 stage-3 design. The code is *learned for separation*:
  - a shared causal trunk feeds a product VQ (4 x 2^12);
  - training is end to end on the bisection-order bound, −log p(codes) − log p(tokens | codes), with straight-through gradients;
  - a cut head must predict the next m tokens from the (token, code) at the cut alone;
  - one arm uses bidirectional "plan codes".
- **Bound.** Codes are deterministic functions of the text, so the bound is honest.
- **Levels.** 12, each one pass plus a code-to-token head.
- **Cost.** About 2.2 to 2.5x dense training FLOPs per token.
- **Bar.**
  - toy T=64: ≤ 15% invalid;
  - d4: bound ≤ 1.05x dense, with the order tax over codes ≤ 3% against l2r over the same codes.
- **Kill.** Either of:
  - E2a: one position needs more than 48 bits to keep the next 256 tokens within 2%. Windows would then bring back K1, and the codes cannot be the separators the Bridge LM's exactness argument assumes;
  - toy invalid above 30% at 24k steps.
- **Nearest work.**
  - S11 Bridge LM: k-means codes impure, 77 to 85% invalid.
  - HDLM, NeurIPS'25: fixed embedding clusters as a coarse scale inside masked diffusion.
  - VQ-VAE-2: an AR prior.
  - Learned causal states (Zhang'19).
- **Delta.** Codes trained by the exact-order bound to *be* separators, decided jointly with their token, in a log-depth exact order.

### S14-D. Per-level coordination variable (CVL)

- **Mechanism.** Before each level, one K-way variable c is drawn (K = 256 to 1024). The level's positions are then independent given (past, c).
- **Exact marginalisation.** p(level) = Σ_c p(c) Π_i p(x_i | c), summed exactly with no sampling in training, so it is RC0-safe.
  - c enters through a class-factorised softmax, which costs K·C per position rather than K·V.
  - It captures up to log K nats of same-level TC per level.
- **Levels.** One extra tiny step per level.
- **Bar.** Removes at least 30% of E0's TC for its order, and cuts the tax by at least 20% at d4.
- **Kill.** Either of:
  - E0's per-level TC exceeds 3·log K where the tax sits;
  - the d4 gain is below 10%.
- **Nearest work.** Mixture of Softmaxes (one position); DA-Transformer (finite-state path); coupled-noise one-step blocks (distilled); Wyner VAE.
- **Delta.** An exactly marginalised public signal per parallel round, sized from the measured same-round TC.

### S14-E. Next-scale prediction over discrete temporal multi-scale codes (VAR-T)

- **Mechanism.**
  - Extend the S13 chunk AE (K=8 tokens, 256-d) and quantise residually along time. Scale s has 2^s codes per block (s = 0..8 over 240 chunks); each code is the residual of the coarser scales.
  - A VAR-style block-causal transformer draws one scale per pass.
  - A lossless decoder emits the tokens.
- **Levels.** 9 scales plus the decode, so 10.
- **Bound.** Codes are deterministic, and the decoder NLL is counted.
- **Bar.**
  - Tokenizer: at least 99.5% token accuracy at ≤ 64 bits per chunk. This is necessary but not sufficient.
  - Prior: ≤ 1.05x dense at d4.
- **Kill.** The tokenizer gate fails, or finest-scale sibling TC exceeds 5% (measured with a small any-order model over the codes).
- **Nearest work.** VAR (NeurIPS'24 best paper); HDLM (semantic scales, diffusion); CALM (AR chunk latents); ResGen (RVQ); Large Concept Models; SV-D (failed).
- **Delta.** Discrete *temporal* residual scales for text with a likelihood bound and categorical commitment at every level, which is the escape the non-commitment paper names.

### Ingredients

These ride on a survivor and are not proposals on their own:
- I1: interval coordinates (distance to the left and right anchor, interval length);
- I2: a multi-horizon future-prediction loss;
- I3: AR multi-task (L=1), the control for contention.

### Outside ML

| survivor | sources |
|---|---|
| PCB | big.LITTLE cores; cortical hierarchy |
| BSB | grain boundaries; the given-new contract |
| LSB | epsilon-machine causal states; nested dissection |
| CVL | Wyner's common information; Aumann's correlated equilibrium |
| VAR-T | wavelets; successive refinement; block-spin renormalisation |

## 6. Stage 0: what to run, what to send back, how it is read

**Cost.** Stage 0 is about 10 to 13 H100-hours (roughly $40 to $55 on Modal). Run the smoke test first.

### E0 + E2a: order oracle and separator bound (offline, nothing trained)

```
modal run modal_sap.py::s14_order_oracle --smoke             # ~20 min: images, model loads, probes, merge
modal run modal_sap.py::s14_order_oracle                     # LLaDA-8B-Base, 32 rows on 4 H100s
modal run modal_sap.py::s14_order_oracle --oracle dream      # Dream-v0-Base-7B, 16 rows on 4 H100s
modal run modal_sap.py::s14_order_oracle_compare
```

**Setup.**
- Rows are 32 FineWeb-Edu validation documents (the last shard on the volume), re-encoded in the oracle's tokenizer: prefix 128, then a 1025-token block.
- Rerunning a command resumes from the rows already scored.
- Images pin the transformers and torch versions the model cards name. The models run their own remote code from the official repos (`trust_remote_code`).

**Cost.** LLaDA is about 5 to 6 H100-hours, Dream about 2.5 to 3. Over 4 shards, wall-clock is about 1.5 hours per oracle.

**Measured per order.** Orders: l2r, bisect1/2/4/16, lanes8/32/64, snap32, random12.

| quantity | definition |
|---|---|
| NLL_par | one oracle pass per step |
| NLL_chain | the order's exact chain under the oracle, one token per pass |
| TC | NLL_par − NLL_chain |
| gap | NLL_chain − NLL_chain(l2r) |

All are reported as % of the l2r chain, with row-bootstrap CIs and per-level TC (nats per row).

**Separator bound (E2a).**
- At block position 512, the past before the last w tokens (w ∈ {0, 1, 4, 16}) is hidden; the BOS token stays visible.
- The l2r NLL of the next 16 / 64 / 256 tokens rises by I_hat bits. This estimates I(future; far past | window).
- Any code C of B bits attached to the window has I(future; C | window) ≤ H(C) ≤ B. Its cost over full context is therefore at least I − B bits, however the code is made or learned.
- So keeping the span within 2% needs at least I − 2%·NLL bits.
- Limits:
  - This is the strong, sufficient-statistic form of a separator, which is what the Bridge LM's exactness argument assumes.
  - The estimate uses the oracle's own cross-entropies. A weaker model tends to underuse long context, which makes a kill conservative.

**Validity, pre-registered.** An oracle that fails any check is inconclusive.
- Right context lowers the oracle's NLL (probe ratio ≤ 0.95).
- TC ≥ 0 within its CI, for every order.
- Oracle l2r bpb within 20% of Qwen2.5-7B on the same bytes.
- Far-past information ≥ 0 within its CI, and not growing with the window.
- Across the two oracles, Spearman ≥ 0.8 on the orders' total cost (TC + gap).
- **Contingency, decided now.** LLaDA may fail only the AR-proximity check with a ratio ≤ 1.3 while passing the rest. In that case Dream (adapted from Qwen2.5-7B) becomes the primary oracle and LLaDA the check.

**Readings, pre-registered.** These are printed in the JSON's `readings` and by `--compare`.

| reading | condition |
|---|---|
| Strict token orders closed on information | `bisect1_TC_ge_3pct`: single-token bisection TC ≥ 3% |
| Exact strict orders closed on computation | bisect1 gap ≥ 5% in both oracles |
| PCB is live | `PCB_live_orders`: some order of ≤ 13 steps with TC ≤ 1% and gap ≤ 2% |
| BSB is live | `snap32_BSB_live`: snapping cuts TC + gap by ≥ 15% against bisect1. Necessary only: the offset decisions are not scored here |
| CVL's K | set by per-level TC: K must satisfy log K ≥ TC/3 at the levels that hold the tax |
| LSB is dead | `LSB_dead`: separator_w1_bits_needed > 48 in both oracles |
| The CVL strong form and the carry-select scan are dead | `separator16_dead`: needed > 16 bits |

**Send back.**
- `out/s14/s14_oracle_llada.json`, `s14_oracle_dream.json` and `s14_oracle_compare.json`;
- the logs in `out/s14/logs/`.

### E1: order-tax scaling on our stack (~3 to 4 H100-hours)

```
modal run modal_sap.py::s11_ladder --depth 8 --specs dense:1:1,wb:1:1:1,wb:16:1:1 --name s14_e1
```

- Window bisection n = 1 and 16 at d8, at matched tokens with `S11dense_x1_s1` (skipped if the checkpoint exists, but it must be in the specs to be scored as the reference).
- Each model is scored in its own order, and compared with d4's +24% and +7.8%.

**Readings, pre-registered.**
- tax(d8)/tax(d4) ≤ 0.7: the gap is shrinking, which supports PCB.
- ≥ 1.0: it is growing. Together with an E0 gap ≥ 5%, this closes exact strict orders.

**Send back.** `out/s03_sap/s11_ladder_bpb_d8_s14_e1.json` and the train logs.

### E2b: conditional on E2a

Run only if E2a's `separator_w1_bits_needed` is ≤ 48 in either oracle. The run is d4 with codes added to the window embeddings, the post-cut attention cut from the far past, and product-VQ or learned codes. That code is not written yet.

## 7. Later stages: only for what Stage 0 leaves alive

**Stage 1: mechanism gates.** d4 at matched tokens, about 1 to 3 H100-hours each, with two seeds for the deciding arm.
- G-PCB: n = 1 and 16, plain against E=4 level experts.
- G-BSB: n = 1 and 4, snapped against midpoint.
- G-LSB: the toy first (`s11_toy`, new arm `bridge_vq`, T = 16 / 64, 24k steps), then d4 text.
- G-CVL: K from E0, on the best Stage-1 order.
- G-VART: the tokenizer gate (`scripts/sap_chunk_ae.py` with PQ/RVQ), then the d4 prior.

**Stage 2: d8 confirmation of any Stage-1 go.**
- Two seeds at 1x and 4x tokens.
- Speed with `s11_speed` at batch 1 / 16 / 64.
- Samples with `s11_gen` plus `s11_rescore_samples` (d16 scorer, matched entropy).

**Stage 3: the paper protocol.**
- OWT, GPT-2 tokenizer, 1024 context, about 110M parameters, plus a 350M confirmation.
- Baselines: AR, MDLM, BD3-LM (public code and checkpoints), plain lanes, and insertion-style bisection.
- Reported: exact or bound bpb, generative PPL at matched entropy, level count, wall-clock.

## 8. Literature

arXiv is blocked from the cloud session, so download these on your machine into `Literature Review/` (gitignored):

```
mkdir -p "Literature Review" && cd "Literature Review"
for id in 2404.02905 2510.08632 1902.03249 2502.09622 2507.12549 2606.30705 2401.17505 2205.13554 \
          2502.09992 2603.12996 2604.02560 1905.10945 1906.10437 2110.13711 1905.10347 2605.07193 \
          2602.16813 2503.09573 2406.07524 2412.15119 2404.09562 2504.20456 2601.13228; do
  curl -sSL -o "$id.pdf" "https://arxiv.org/pdf/$id"; done
```

| arXiv id | paper |
|---|---|
| 2404.02905 | VAR |
| 2510.08632 | HDLM |
| 1902.03249 | Insertion Transformer |
| 2502.09622 | Feng et al. |
| 2507.12549 | Serial Scaling Hypothesis |
| 2606.30705 | non-commitment |
| 2401.17505 | Arrows of Time |
| 2205.13554 | Shih et al., AO-ARM |
| 2502.09992 | LLaDA |
| 2603.12996 | DAPD |
| 2604.02560 | dependency-guided parallel decoding |
| 1905.10945 | Wyner VAE |
| 1906.10437 | learned causal states |
| 2110.13711 | Hourglass |
| 1905.10347 | Discrete Flows |
| 2605.07193 | Coupling Models |
| 2602.16813 | FMLM |
| 2503.09573 | BD3-LM |
| 2406.07524 | MDLM |
| 2412.15119 | PAR |
| 2404.09562 | sigma-GPT |
| 2504.20456 | any-subset AR |
| 2601.13228 | A3 |

Dream 7B is the `Dream-org/Dream-v0-Base-7B` model card and blog.

## 9. How I check what you send

- **Row hash.** Every per-position JSON from `sap_position_bpb` must print `ae01832e4bf20262` (same data and tokenizer at d4 and d8). E0 prints its own row hash per oracle; the merge refuses shards whose hashes differ.
- **References.** Every ratio cites its reference tag and token budget.
- **Seeds.** Deciding arms have two seeds. A gain counts only above 2x the d4 seed noise (0.13%).
- **Own order.** Each model is scored in its own order and checked against its training-time val bpb (±0.5%).

## 10. Honest odds

The strict thesis at the A* bar is a long shot. Stage 0 decides within days whether any strict family is alive.

My prior:
- exact token orders at ≤ 13 levels fail on information (E0 TC) or on computation (E0 gap, E1);
- LSB needs single-position separators of ≤ 48 bits, which SEP and the excess entropy of text make unlikely;
- PCB is the best bet if E0 shows small TC and a shrinking gap.

If Stage 0 closes all five, the strict thesis is closed with measured reasons. The A* path is then the non-strict speculative lanes.

## 11. Results (2026-10-05)

**Runs.** E0 + E2a in `scratch/s14/` (JSONs and merge logs from `out/s14/` on the volume).
- `GSAI-ML/LLaDA-8B-Base`: 32 rows, row hash `a249add6812c7098`.
- `Dream-org/Dream-v0-Base-7B`: 16 rows, row hash `0757c59a556d13c2`.
- Rows are 128 + 1025 tokens. About 6 H100-hours: 481 s per row for LLaDA, 396 s for Dream.

**Validity: every check passed in both oracles.**

| check | LLaDA | Dream |
|---|---|---|
| shift probe (mean NLL, shift 0 / shift 1) | 0.56 / 12.1 nats | 7.3 / 0.58 nats |
| right-context probe ratio | 0.155 | 0.313 |
| l2r bpb against Qwen2.5-7B on the same bytes | 0.6646 vs 0.5984 (1.111x) | 0.6181 vs 0.5785 (1.069x) |

- TC ≥ 0 for every order.
- Separator information is non-negative and shrinks with the window.
- Cross-oracle Spearman of total cost: 0.967.

### Per order

% of the l2r chain NLL, with row-bootstrap 95% CIs.

| order | steps (LLaDA / Dream) | TC % LLaDA | TC % Dream | gap % LLaDA | gap % Dream | total % LLaDA | total % Dream |
|---|---|---|---|---|---|---|---|
| bisect1 | 12 / 12 | 13.53 [12.59, 14.59] | 14.97 [13.53, 16.46] | 4.68 [3.68, 5.89] | 9.38 [7.01, 12.75] | 18.21 [16.42, 20.29] | 24.35 [20.80, 28.85] |
| bisect2 | 21 / 21 | 4.85 [4.37, 5.40] | 5.18 [4.66, 5.78] | 3.86 [2.96, 4.95] | 8.26 [6.15, 10.93] | 8.71 [7.54, 10.09] | 13.44 [11.01, 16.56] |
| bisect4 | 37 / 37 | 1.63 [1.39, 1.91] | 1.89 [1.46, 2.34] | 3.06 [2.33, 3.96] | 6.43 [4.62, 8.75] | 4.69 [3.85, 5.69] | 8.32 [6.30, 10.82] |
| bisect16 | 113 / 113 | 0.25 [0.15, 0.35] | 0.20 [0.03, 0.39] | 1.44 [0.97, 2.01] | 2.98 [1.90, 4.46] | 1.68 [1.19, 2.28] | 3.17 [2.11, 4.68] |
| lanes8 | 129 / 129 | 0.16 [0.04, 0.34] | 0.06 [-0.07, 0.24] | 0.29 [0.04, 0.55] | 0.95 [0.43, 1.71] | 0.45 [0.16, 0.82] | 1.01 [0.43, 1.85] |
| lanes32 | 33 / 33 | 1.20 [0.90, 1.63] | 1.25 [0.84, 1.80] | 1.18 [0.73, 1.70] | 2.52 [1.44, 3.95] | 2.38 [1.72, 3.24] | 3.77 [2.62, 5.43] |
| lanes64 | 17 / 17 | 3.06 [2.49, 4.02] | 3.45 [2.86, 4.13] | 1.86 [1.25, 2.58] | 3.73 [2.27, 5.77] | 4.92 [3.90, 6.37] | 7.18 [5.56, 9.60] |
| snap32 | 15 / 14 | 13.65 [12.55, 14.82] | 13.86 [12.27, 15.54] | 3.99 [3.09, 5.01] | 8.52 [6.54, 11.32] | 17.64 [15.91, 19.68] | 22.38 [19.07, 26.33] |
| random12 | 13 / 13 | 13.34 [12.16, 15.00] | 14.43 [12.84, 16.40] | 3.95 [3.06, 5.04] | 7.98 [6.01, 10.77] | 17.29 [15.50, 19.59] | 22.42 [19.10, 26.75] |

### Separator bound (E2a)

The cut is at block position 512 and the span is the next 256 tokens.

| quantity | LLaDA | Dream |
|---|---|---|
| far past's information (window 0) | 152 bits | 135 bits |
| far past's information (window 16) | 95 bits | 89 bits |
| bits a separator needs (w = 1) | **126 [102, 154]** | **113 [88, 140]** |
| bits a separator needs (w = 16) | 80 | 74 |

### Pre-registered readings

| reading | LLaDA | Dream | verdict |
|---|---|---|---|
| bisect1 TC ≥ 3% (strict token orders closed on information) | yes | yes | **closed** |
| bisect1 gap ≥ 5% in both (closed on computation) | no (4.68) | yes | not established, and moot |
| PCB: some ≤13-step order with TC ≤ 1% and gap ≤ 2% | none | none | **not live** |
| BSB: snapping cuts TC + gap by ≥ 15% | 3.2% | 8.1% | **not live**; it also adds steps (15 / 14) |
| LSB: separator needs ≤ 48 bits | 126 | 113 | **dead** |
| CVL strong form, carry-select scan: ≤ 16 bits | 126 | 113 | **dead** |
| CVL's own kill: per-level TC > 3 ln K where the tax sits (K=1024: 21 nats) | level 9: 158 nats | level 9: 180 nats | **dead** |

VAR-T was not tested. The diagnosis below disfavours it, and it is not run.

### Diagnosis (reusable)

1. **Same-step dependence in text is local.**
   - 96.6% (LLaDA) and 98.1% (Dream) of bisection's TC sits in its three finest levels, at spacing ≤ 8.
   - That holds although every coarser anchor is an exact token, so coarse plans, codes or scales cannot remove it.
2. **Lanes lie below every other order's cost-vs-steps curve.** The step counts are not exactly matched, and the matched comparison against diffusion decoding is S15 L0b. The margins:
   - lanes32 at 33 steps costs about half of bisect4 at 37 steps (2.4 vs 4.7% LLaDA; 3.8 vs 8.3% Dream);
   - lanes64 at 17 steps costs 3.1 to 3.7x less than the 12-to-15-step orders (4.9 / 7.2% against 17.3 to 24.4%);
   - lanes8 at 129 steps costs 3.1 to 3.7x less than bisect16 at 113 steps.
   - Interface-first bisection, the S11 premise, is the wrong parallel structure for text.
3. **Lanes' information floor.**
   - TC per lane: 0.43 / 0.82 / 1.04 nats at S = 128 / 32 / 16 (LLaDA). 42 to 43% of it sits in the lane-start step.
   - TC ≤ 1% needs about 36 or more steps per lane. This is extrapolated from S = 16 and 32; T = 1920 is measured in S15 L0b.
4. **Excess entropy.**
   - The far past carries 135 to 152 bits about the next 256 tokens.
   - 89 to 95 bits of that sit beyond a 16-token window.
5. **The gap depends on how the oracle was trained.** Dream, AR-adapted, has about 2x the gap of LLaDA, which was trained on any order.
6. **S13's open question is answered: the unrecovered junction loss is mostly learnability.** Refined in S15 §7.2: the lane-start deficit is information, and the learnable part is recovery.
   - d8 L=64 loses 2.38 average-token losses per lane, about 7 nats at about 3.0 nats per token. S15 R1 measures this exactly.
   - The 8B oracle's total is about 1.7 nats per lane, of which about 1.0 is TC.
   - So about 85% of d8's lane tax is learnable.

**E1, run anyway (2026-10-05, d8, matched tokens with `S11dense_x1_s1`).** Moot after the closure; recorded for completeness (`scratch/s15/s11_ladder_bpb_d8_s14_e1.json`).
- Window bisection n=1: +28.5% (d4 +24%).
- n=16: +7.0% (d4 +7.8%).
- tax(d8) / tax(d4): 1.19 for n=1 and 0.90 for n=16, against the pre-registered 0.7 line for "shrinking".

