# PROJECT_MAP.md

Quick-reference map of the codebase for AI agents and human researchers. This document details the files, key classes, functions, and brief summaries of their operational roles across the repository.

---

## 1. Architecture & Repository Overview

`nanochat` is an LLM pretraining and model efficiency research repository designed to research, benchmark, and evaluate novel FLOP-efficient architectural innovations against dense Transformer baselines:
- **RemixedLinear & ConditionedLinear**: Dense linear layer replacements utilizing context-conditioned routing and template mixing.
- **Modular Sub-Transformers (MST)**: Replacing standard dense layers with $K$-sized parallel sub-transformer blocks.
- **Early Exit Transformer (EET)**: Exiting tokens dynamically at early layers to reduce compute per token during inference/training.
- **Structured Code Output Heads (SCH)**: Replacing the dense $V \times d$ softmax with a frozen, binary, structured output embedding built from monomials of a per-token binary code, trading head parameters and output rank against compute.

---

## 2. Core Package (`nanochat/`)

Located at [`nanochat/`](file:///home/seqaeon/Downloads/nanochat/nanochat). Contains all neural network layers, model definitions, optimizers, dataloaders, and core evaluation engines.

### 2.1 Core Models & Layer Implementations

#### [`nanochat/gpt.py`](file:///home/seqaeon/Downloads/nanochat/nanochat/gpt.py)
Primary entry point for the standard GPT architecture and experimental linear layer replacements (RemixedLinear, ConditionedLinear).

- `class Config`: Configuration dataclass for GPT models (n_layer, n_head, n_embd, vocab_size, sequence_length, linear_type, etc.).
- `class GPT(nn.Module)`: Full Decoder-only Transformer model.
  - `forward(idx, targets=None, kv_cache=None, loss_reduction='mean')`: Standard causal forward pass with optional KV caching and targets.
  - `setup_optimizer(...)`: Configures mixed AdamW / Muon parameter groups (separating 2D matrices for Muon and 1D/scalars for AdamW).
  - `estimate_flops()`: Returns estimated FLOPs per token for the current model configuration.
- `class RemixedLinear(nn.Module)`: Drop-in replacement for `nn.Linear` using context-conditioned basis expansion & template bank routing.
  - `forward(x, ctx=None)`: Projects input using dynamically mixed template weights based on router decisions.
- `class RemixedLinearFused(nn.Module)`: Optimized CUDA/Triton fused implementation of `RemixedLinear`.
  - `forward(x, ctx=None)`: Executes template mixing and matmul via optimized batched/Triton kernels.
- `class ConditionedLinear(nn.Module)`: Context-conditioned linear layer variant.
- `class Block(nn.Module)`: Transformer decoder block combining attention, MLP (or RemixedLinear MLP), and residual connections.
- `class CausalSelfAttention(nn.Module)`: Multi-Head Causal Self-Attention supporting FlashAttention, RoPE, and KV caching.
- `class MLP(nn.Module)`: Standard or Remixed SwiGLU / GeLU feed-forward network.
- `class AffineQuantileRouter(nn.Module)`: Quantile-balanced routing mechanism for dynamic template selection.
- `class QuantileBalancedRouter(nn.Module)`: Router ensuring uniform capacity distribution across templates.
- `class AttentionRouter(nn.Module)`: Cross-attention routing module for conditioning linear weights on context.

#### [`nanochat/mol.py`](file:///home/seqaeon/Downloads/nanochat/nanochat/mol.py)
Faithful reimplementation of **MoL** (Ternovtsii & Bilak 2026, arXiv:2605.09516v1), the closest
prior art to MST, used as the head-to-head baseline. No official code exists; recovered from
their equations and validated against their published parameter counts.

- `make_thin_config(config)`: clones GPTConfig down to `d_thin` so `gpt.Block` builds a genuine
  narrow block, pinning `d_head=64` at every width. Asserts every Phase-XX research flag is at
  its default, because the shadow config would otherwise inherit them and a thin block would
  silently stop matching the dense baseline.
- `class ThinBlock`: Eq (1), `W_up . (Block_thin(W_down . x) - W_down . x)`. The subtraction
  strips the block's inner residual so it emits only its delta.
- `class SplitStage`: Eq (2) plus the shared blocks of their section 3.2. S always-active blocks
  with full attention, N-S routed blocks selected top-k with a softmax router and CV^2 balance
  loss. Routed blocks use `token_active` so they attend only over their own tokens, which is the
  "dense restricted attention" their section 2.3 defines sparse dispatch against.
- `class MoL(nn.Module)`: model. Same surface as `MST` (`forward`, `num_scaling_params`,
  `estimate_flops`, `compute_diagnostics`, `setup_optimizer`, `generate`) so it drops into the
  training loop unchanged, and `setup_optimizer` deliberately mirrors MST's so the head-to-head
  is not confounded by optimizer treatment.
- Not yet implemented: Gated DeltaNet in routed blocks (`mol_routed_attn` is the slot for it).

#### [`nanochat/mst.py`](file:///home/seqaeon/Downloads/nanochat/nanochat/mst.py)
Implementation of Modular Sub-Transformers (MST).

- `class MSTConfig`: Configuration dataclass for MST models (n_sub, sub_dim, shared_kv, transition_type, etc.).
- `class MST(nn.Module)`: Modular Sub-Transformer backbone.
  - `forward(idx, targets=None, kv_cache=None, loss_reduction='mean')`: Forward pass executing $N$ parallel sub-transformers across sequence blocks.
  - `setup_optimizer(...)`: MST-specific parameter grouping for Muon/AdamW.
  - `compute_diagnostics()`: Computes sub-transformer routing entropy, cross-sub agreement, and load distribution.
- `class MSTLayer(nn.Module)`: Individual MST layer containing $N$ sub-transformer blocks and transition routing.
- `class BatchedMSTLayer(nn.Module)`: High-performance batched PyTorch implementation executing sub-transformers via 3D tensor ops.
- `class SubBlock(nn.Module)`: Individual sub-transformer block within an MST layer.
- `def _batched_linear(x, weight)`: Batched linear projection `x (B, T, N, in) @ weight (N, out, in).T`.
- `def _rope_streams(x, cos, sin)`: rotary embedding for a `(B, T, N, H, hd)` tensor, stream axis kept. `gpt.apply_rotary_emb` asserts `ndim == 4` and concatenates on dim 3, so it cannot take the stream axis. Exists so RoPE and QK-norm run ONCE per layer instead of once per stream (Stage 19), and so the fused QKV output can stay a strided view: q/k/v are slices of one `(B, T, N, 3, H, hd)` buffer, where N and H are not adjacent and a flatten into the head axis would copy. Arithmetic is identical to `apply_rotary_emb`.
- `def resolve_sub_heads(config)`: Single source of truth for per-stream attention geometry, returning `(n_head_per_sub, head_dim)`. With `mst_sub_head_dim > 0` (Stage 12 / G1) it pins `head_dim` and derives the head count so `qkv_dim` stays equal to `sub_dim`, making the change parameter- and FLOP-neutral; otherwise it reproduces the legacy `d // config.n_head` geometry (32 at N=4, against dense's 128).

- `def mix_channels(x, mode, N, d, inverse=False)`: permutes the flattened `(N*d)` channel axis of a `(B,T,N,d)` stream tensor. `'roll'` shifts by `d//2` (Swin's shifted windows on the channel axis; streams keep half their channels so specialization survives); `'shuffle'` is ShuffleNet's transpose (every stream draws `d/N` channels from every other; maximal mixing, no stream identity). Distinct from `mst_feature_cycle`, which rolls by exactly `d` and only relabels which weights see which block.
- `def MST._build_layer_sub_windows(N)`: per-(layer, sub) attention windows, the single source of truth for the forward pass, the legacy list path, and `estimate_flops`. Without multi-scale windows every stream inherits the layer window; with them, legacy behaviour lets the per-sub schedule *replace* the layer pattern (so the widest stream is full-context at every layer), while `mst_compose_windows` intersects the two.

**Stage 12 dense-parity flags** (all FLOP-neutral; see `LEARNINGS.md` 2026-08-08): `mst_sub_head_dim` (G1, head geometry), `mst_final_norm` (G2, RMSNorm before `lm_head` as dense does), `mst_per_stream_ve` (G3, per-stream value-embedding slices from an `N*d`-wide table instead of one `d`-wide table broadcast to every stream). Measured at L=8: G1 −0.0101, G3 −0.0092, **G2 +0.0021 (a regression, do not use)**.

**G3-cheap** (`mst_ve_map`, `mst_ve_map_rank`): keep ONE `d`-wide VE table and give each stream its own learned view of it, instead of widening the table to `N*d`. Saves 201M params at L=16 and 805M at L=32. Unlike G3 it is a matmul, so it DOES cost FLOPs: +1.735% at L=16 full rank, +0.434% at rank 32. Both forms are exactly identity at init, so they start at the plain-VE baseline. Strictly less expressive than G3 (all N vectors are linear images of one shared vector rather than N independent lookups).

**Stage 19 non-GEMM overhead cuts** (no flags, always on, no FLOP or parameter change; see `LEARNINGS.md` 2026-09-01). Measurement showed MST's block-diagonal GEMMs clear their wall-clock crossover at every D >= 512, and that the 2.4x training gap against dense is non-GEMM time: 36.7 ms of a 47.7 ms step at depth 12 against dense's 10.6 of 20.2. Three cuts: (1) RoPE and QK-norm hoisted out of the per-stream attention loop via `_rope_streams`, leaving only `flash_attn` in it (it stays because `window_size` is a per-call scalar pair and `mst_multi_scale_windows` gives each stream a different one); (2) the three QKV projections fused into one bmm with a 3x wider output, with the three weights left as separate Parameters and concatenated per forward so checkpoints, Muon groups, grad-equalize hooks and `estimate_flops` name lists are untouched (`BatchedMSTLayer._fuse_qkv`, off under `mst_shared_kv_attn`); (3) `_last_stream_load` and `_last_route_entropy` gated on `MST._diag_enabled` so they stop being torch.compile graph outputs on non-log steps. Forward is bit-identical across 16 flag combinations; the backward agrees to one bf16 ULP when the QKV fusion is on. Regressions: `tests/test_mst_parity_fixes.py` O3/O4/O5. Two candidates were sized and rejected: a preallocated attention-output buffer (a no-op, `torch.stack` already does exactly that) and the stream-major `(N, B, T, d)` layout (1.00x at d=256 and d=384, 0.92x at d=512).

**Stage 14 free-mixing flags** (zero parameters, zero FLOPs, verified identical at L=16/32): `mst_channel_mix` (`none|roll|shuffle`) and `mst_channel_mix_site` (`layer` = alternate the partition offset on odd layers, `ffn` = permute between attention and FFN as ShuffleNet does, `both`). Motivation: an MST layer is block-diagonal, and composing block-diagonal maps under a fixed partition stays block-diagonal, so all cross-stream flow is forced through the rank-`d` coupling. Permuting the partition makes the composition mix instead. Implemented in `BatchedMSTLayer` only; the legacy list path asserts.

**Stage 16 conditional-stream flags**: `mst_stream_topk` (k of N streams active per token, 0=dense), `mst_stream_router_aux`, `mst_stream_gate_attn`. `BatchedMSTLayer._stream_gate` returns a 0/1 gate carrying gradient through a straight-through estimator, from a per-token causal router (`stream_router_w`, `(N, N*d)`). Phase A is compute-then-mask, to price what k-of-N sparsity costs; the fixed-capacity gather/scatter that realises the saving is Phase B. `MST.estimate_flops` now genuinely discounts `active_flops` (the first place in the repo that does) by `6 * (1-k/N) * ffn_params`, plus the QK term when attention is gated. Measured active-FLOP savings at L=32: k=3 −9.3%, k=2 −18.6%, k=1 −28.0%.

**Stage 18 Monarch FFN**: `mst_ffn_monarch` (`none|shuffle|roll`). `fc_w` (per-stream `d→4d`) and `fc_proj_w` (per-stream `4d→d`) are already the two block-diagonal factors of a Monarch matrix with the permutation set to identity; this inserts the missing `P` on the `N*4d` hidden axis, reusing `mix_channels` with the *hidden* width rather than the stream width. Zero parameters, zero FLOPs. Applied once and **never inverted**, unlike the Stage 14 stream-axis permutation which is a change of basis. Asserted incompatible with `mst_stream_dispatch`: Monarch needs every stream's up-projection to exist, and in the dispatched path a capacity-buffer slot holds a different token per stream.

**Stage 16 Phase B dispatch**: `mst_stream_dispatch`, `mst_stream_capacity_factor`. `BatchedMSTLayer._ffn_dispatched` gathers each stream's selected tokens into a fixed-capacity `(B, K, N, d)` buffer, runs the same single batched matmul, and scatters back, so the FFN genuinely runs on `K < T` tokens. Bit-exact against the masked path when nothing overflows; overflow is reported as `_last_stream_drop`. Capacity is resolved in **position order**, which keeps token-choice routing causal (expert-choice would let a later token evict an earlier one). Measured 1.14x forward wall-clock at D=2048 against a 28% active-FLOP saving.

**Stage 17 optimizer flags**: `mst_shampoo`, `mst_precond_every`, `mst_shampoo_beta` route the stacked per-stream weights to `kind='shampoo'` in `nanochat/optim.py`. See `shampoo_step` and `_inverse_fourth_root` there, and `tests/test_shampoo.py`.

**Stage 15 coupling/attention flags**: `mst_distribute_block_muon` (F1, puts `distribute_w` in `setup_optimizer`'s `stacked_names` so it finally gets block-diagonal Newton-Schulz and the sub-LR; it is `(N*d, d)` like `c_proj_w` but was omitted, so Muon was orthogonalizing across all N coupling blocks jointly), `mst_trans_spectral_lr` (F2, `agg_up_w` and `agg_down_w` get LRs scaled by `sqrt(N)` and `1/sqrt(N)`), `mst_transition_every` (F3, **now live on the batched path** — it was previously allocated-and-ignored there; non-coupling layers allocate no transition weights, so params and FLOPs both drop: −9.2%/−14.1% params and −7.6%/−11.7% FLOPs at L=32 for k=2/4), `mst_talking_heads` (F4, learned `(N*n_head)^2` mixing along the head axis before `c_proj`, identity-init, +0.006% params), `mst_wo_mode` (F5, `block`|`dense`; dense is +19.7% params / +16.3% FLOPs at L=32). `MST.__init__` now asserts that none of these are set on the legacy list path, which cannot implement them.

**Stage 13 overhead-cut flags** (these *reduce* FLOPs, unlike Stage 12): `mst_compose_windows` (O2, see `_build_layer_sub_windows`) and `mst_lm_head_dim` (O1, factorizes the output head as `D → Dh → V`). Measured at L=8: **O2 is free** (−10.2% FLOPs, +0.0001 bpb, inside seed noise) and is the recommended default; **O1 is not** (−27.7% FLOPs but +0.0461 bpb, matching what the parameter scaling law predicts, so the head is capacity rather than overhead). O2's FLOP saving is −10.2% / −10.4% / −7.5% at L=8/16/32.

#### [`nanochat/code_head.py`](file:///home/seqaeon/Downloads/nanochat/nanochat/code_head.py)
Structured Code Output Heads (SCH). Replaces `lm_head` with
`logit(w|h) = phi_k(c(w))^T g(h)`, where `c(w)` in `{0,1}^B` is a frozen binary code per token
and `phi_k` is the vector of all monomials (AND products of bits) up to interaction order `k`,
so `M = sum_{j<=k} C(B,j)`. Equivalently: a softmax whose output embedding matrix
`Phi in {0,1}^{V x M}` is frozen, binary and structured. Order 1 is Oda et al. 2017
(logit rank <= B), order B is the exact softmax, and everything between is the research question.

- `minimal_bits(V)` / `full_phi_width(bits, order)`: `ceil(log2 V)` and `sum_j C(B,j)`, the two
  quantities every sweep row is indexed by.
- `build_codes(vocab_size, bits, mode, seed, freqs, path, ecc_bits)`: the code-assignment arms.
  `binary` (bijective expansion), `random` (distinct random codes), `ecc` (systematic random
  linear code over GF(2), so injective by construction with distance from the parity part),
  `frequency` (rank-ordered), `file` (semantic codes from `scripts/code_assign.py`). `ecc_bits`
  appends parity bits to any base code, which is the single axis interpolating the two opposed
  objectives of contribution 4. Asserts injectivity: two tokens sharing a code share a row of
  `Phi` and are indistinguishable at any order, which puts a hard floor under the loss.
- `code_statistics(C)`: density, sampled mean and minimum Hamming distance. Minimum distance 1
  means no error-correction slack, which is why the minimal `B = 15` code at `V = 32768` makes
  the ECC-versus-semantic comparison undefined.
- `enumerate_monomials(bits, order, max_m, seed)` / `build_phi_monomial(C, groups, ...)`: the
  index sets and the materialised `Phi`. A monomial over binary variables is an AND, so
  `phi_S(c) = min_{b in S} c_b`; the build is chunked over columns because the intermediate is
  `V x chunk x order` bytes. An `max_m` cap truncates *within* the highest kept order rather than
  dropping the order, so `M` stays a continuous knob for the saturation sweep.
- `build_phi_random_binary` / `build_phi_onehot`: the two frozen non-monomial controls. The
  random-binary one matches the monomial arm's expected row density, so it isolates *structure*
  from *binariness*; the one-hot one is VQ-Logits, which is the `k=1`, one-bit-set corner of this
  same family.
- `class CodeProjection`: `g`, linear or MLP, `R^d -> R^M`. **A linear `g` caps the logit rank at
  `min(M, d)`**, which makes orders 3 and 4 rank-identical at `d=512` and fakes ladder saturation;
  the MLP form is what separates "saturated" from "hit `d`".
- `class StructuredCodeHead`: the head. Drop-in for `Linear(n_embd, padded_vocab_size)`.
  `Phi` is a **non-persistent** buffer rebuilt deterministically from the persistent `codes`
  buffer, so a checkpoint carries `V x B` uint8 rather than `V x M` floats, and a
  `load_state_dict` post-hook rebuilds it. `rank_ceiling()` reports the configuration's
  theoretical bound; `flops_per_token()` prices the frozen `Phi` at `4 * V * M` (no weight
  gradient) against `6 * V * M` for a learned one.
- `class HierarchicalSoftmaxHead`: the Huffman tree baseline (Morin and Bengio 2005). Owns its
  loss because a tree head never materialises a `V`-wide logit vector; refuses to produce logits,
  so generation and the rank probe raise rather than silently working.
- `class CodeInputEmbedding`: the Phase 3 input-side arms, `linear` (`E = C U`, rank <= B, the
  predicted collapse), `expanded` (`E = phi_k(c) U`), `nonlinear` (`E = MLP(c)`), `tied`
  (the output head's final projection transposed, so one matrix serves both directions).
- `resolve_sch_config(config, padded_vocab_size)`: single source of truth for `B`, `M` and the
  width cap, shared by the head, the FLOP estimate, the sweeps and the diagnostics.
- `build_code_head` / `describe_head`: factory used by `GPT.__init__`, and the startup line that
  prints `B`, `k`, `M`, the rank ceiling, head parameters and head FLOPs against the dense
  equivalent.

**SCH configuration flags** (all on `GPTConfig`, all exposed as `--sch-*` on `scripts/base_train.py`
and plumbed through `research_sweep.sh` and `research_compare.py`):
- *master*: `use_code_head`, `sch_head_type` (`code` | `hsoftmax`).
- *code assignment*: `sch_bits` (B), `sch_code_mode`, `sch_code_path`, `sch_code_ecc_bits`,
  `sch_code_seed`.
- *expansion*: `sch_order` (k), `sch_max_m` (cap on M), `sch_phi_mode`
  (`monomial` | `random_binary` | `onehot` | `learned` | `gaussian`), `sch_phi_density`,
  `sch_phi_dtype`, `sch_phi_normalize`, `sch_phi_center`.
- *the projection g*: `sch_g_type` (`linear` | `mlp`), `sch_g_hidden`, `sch_g_layers`,
  `sch_g_out_std`.
- *rank mitigations*: `sch_mixture` (log-sum-exp mixture of code heads, escapes the bound
  entirely), `sch_logit_act` (`sigsoftmax` | `monotonic`), `sch_residual_rank` (dense residual
  hybrid, buys `r` rank for `rV` params), `sch_bias` (per-token bias: +1 rank but it BREAKS
  zero-shot vocabulary extension).
- *input side*: `sch_input_mode`, `sch_input_hidden`.

**Two facts that are expensive to rediscover.** (1) The rank probe must run in **fp32**: with a
bf16 `Phi` the singular values below the true rank sit at ~1e-3 of the leading one, and a
genuinely rank-15 head reads as full rank. Measured: a dense `d=64` head reported rank 276 with a
bf16 probe and exactly 64 with an fp32 one. (2) Every SCH sweep must pin an explicit
`--target-tokens`, because `base_train` sizes the token budget from head parameters and a code
head has up to 17x fewer of them, so the default would quietly give the code arms less data than
the dense control. `scripts/code_head_budget.py` computes the matched value.

#### [`nanochat/block_head.py`](file:///home/seqaeon/Downloads/nanochat/nanochat/block_head.py)
SAP (sampling-aware pretraining, `sap_research_plan.md` v4): one head that emits the next T tokens per trunk pass. A plan latent drawn once per block reaches every slot before readout; that is what makes the T tokens agree, since attention among slots alone still yields a product of marginals.

- `class BlockHead`: all modes in one module (`config.sap_block_mode`): `indep` (B2, independent slots), `p1_discrete` / `p2_gauss` (P1/P2, plan latent trained by an ELBO with a recognition model), `p3_energy` (P3, energy score, likelihood-free), `cp` (B3 mixture), `local` (B4 local AR head), `inv_head` (B5, PTP-style data-inverted noise), and the controls `plain_noise` / `wta`. Slots attend to each other and to a window of `sap_ctx_window` trunk states; logits come from the model's own `lm_head` + softcap.
  - `loss(h, ctx, ctx_valid, y, readout, embed, u_bins)`: per-token training loss for a batch of blocks; returns diagnostics (KL per group, reconstruction, energy score).
  - `sample(...)`: draws one block per row (prior latent, then one decoder pass; `local` is sequential inside the head).
  - `block_logprob(...)`: exact (indep/local/cp), importance-weighted bound (p1/p2), sampled-noise bound (inv_head), Monte Carlo (plain_noise/wta), None (p3).
  - `latent_sensitivity(...)`: H(mean p) - mean H(p) over latent draws, in nats; ~0 means the latent is ignored.
  - `flops_per_token(V)`: training FLOPs the head adds per trunk token (scaled by `sap_block_frac`); priced into `GPT.estimate_flops`.
- `def evaluate_block_bpb(model, batches, steps, token_bytes)`: block bpb next to the trunk's next-token bpb on the SAME tokens; called by `base_train` at every eval (`SAP_EVAL_JSON`).
- `def energy_score`, `def pick` (inverse-CDF sampling), `def target_bins` (data token -> CDF bin, for `inv_head`), `def gumbel_st`.
- GPT integration (`nanochat/gpt.py`): `sap_*` config fields; `GPT._sap_block_loss` adds the block loss in training only (`loss_reduction='mean'`), so every bpb eval still measures the trunk; `forward(..., skip_logits=, return_hidden=)`; `GPT.generate_block()` is the uncached reference decoder.
- **SAP v4 (`sap_research_plan.md` v4): exact sampling cuts.** Modes `lat_crf` / `lat_tt` / `lat_cp` (exact CRF, HMM/tensor-train or code-mixture joint over a top-K candidate lattice plus an escape state, all T slots), `cut_crf` / `cut_tt` (lead: even slots are anchors drawn jointly from that exact joint, one reference layer fills the odd slots), `pmi_chain` (corpus PMI floor), `corpus_code` (corpus token classes as an exact latent), `p1_selfpost` (P1 with the trunk's own block-end state as posterior). Every v4 head reads slot 0 out of the trunk state, so its first token is the next-token distribution. Pieces: `_lattice` (chunked readout to candidates, escape mass, target lattice state, optional soft-target CE and escape draws), `_crf_pair` (rank-r context-gated pair terms, optional corpus support penalty), `_tt_parts`, `_cp_ll` / `_cp_emit`, `_struct_ll` / `_struct_draw`, `_cut_fill`, `_pmi_logprob`, `_class_readout` / `_code_ll`, `_encode_self`, `_nce_scores` (self-contrastive resampling, `sap_nce_props`), `nce_logprob_estimate`, `_v4_ll` / `_v4_draw`. Corpus tables live in non-persistent `tab_*` buffers (`set_tables`, refilled in `init_weights`). `GPT._sap_block_loss` scales the block loss's gradient into trunk, lm_head and wte by `sap_trunk_grad` (0 = detached, trunk exactly dense).

#### [`nanochat/sap_chain.py`](file:///home/seqaeon/Downloads/nanochat/nanochat/sap_chain.py)
Exact chain joints for the v4 heads, CUDA-graph safe (static loops, fixed shapes, no `.item()`): `chain_logz` (forward algorithm), `chain_score`, `chain_logprob` (observed slots clamped, the rest summed out exactly), `chain_viterbi`, `chain_sample` (forward-filter backward-sample; Viterbi at temperature 0), `hmm_loglik`, `hmm_sample` (ancestral; joint Viterbi at temperature 0).

#### [`nanochat/sap_tables.py`](file:///home/seqaeon/Downloads/nanochat/nanochat/sap_tables.py)
Corpus tables (the user's training-set distribution tables): `PairCounter` (exact (a, b) counts at a gap, chunked `torch.unique`), `build_tables` (adjacent and skip-1 support with counts, PMI and p(b|a) top-M rows, PPMI-SVD k-means token classes and class PMI), `tables_from_sequences`, `seen` (sorted-key membership), `describe`.

#### [`nanochat/eet.py`](file:///home/seqaeon/Downloads/nanochat/nanochat/eet.py)
Implementation of Early Exit Transformers (EET).

- `class EETConfig`: Configuration dataclass for EET (exit_rates, exit_loss_weight, routing_type, etc.).
- `class EET(nn.Module)`: Early Exit Transformer model.
  - `forward(idx, targets=None, kv_cache=None, loss_reduction='mean')`: Forward pass dynamically dropping tokens at intermediate layers.
  - `compute_exit_loss(...)`: Computes multi-exit auxiliary losses and depth-weighted cross-entropy.
- `class EETBlock(nn.Module)`: EET layer block supporting per-token routing, skip-connection passes, and early exit decisions.
- `class ExitRouter(nn.Module)`: Predicts per-token exit probabilities or hard exit decisions per layer.

---

### 2.2 Optimization, Execution & Infrastructure

#### [`nanochat/optim.py`](file:///home/seqaeon/Downloads/nanochat/nanochat/optim.py)
Mixed Muon + AdamW optimizer implementations for extreme efficiency.

- `def shampoo_step(...)`: block-diagonal Shampoo (Stage 17). Preconditioning a dense `D x D` weight is `O(D^3)`; MST's stacked per-stream weights are N blocks of `d x d`, so exact block preconditioning is `D^3/N^2`, `N^2 = 16x` cheaper at N=4. K-FAC and Shampoo both *approximate* block-diagonality; MST makes it exact by construction. Reuses the `block_diagonal` reshape already in `_step_muon`, plus its Nesterov momentum and cautious weight decay, so only the orthogonalization is replaced.
- `def _inverse_fourth_root(M, eps, fallback)`: batched `(M + ridge)^(-1/4)` in fp32. The ridge is **relative to the mean diagonal**, not absolute: after one step `L` is rank-1 and an absolute ridge leaves it singular so `eigh` fails to converge. Eigenvalues get a relative floor, and a genuinely degenerate matrix returns `fallback` (the previous preconditioner) rather than crashing training.
- `MuonAdamW.load_state_dict` / `DistMuonAdamW.load_state_dict` restore fp32 on `L`/`R`/`QL`/`QR`: `torch.optim` casts state to the owning parameter's dtype, which would silently downcast the preconditioner statistics to bf16 on resume.
- `kind='shampoo'` is dispatched at all nine `kind` sites. `_reduce_muon` is reused verbatim for the distributed reduce phase since gradient communication is identical; only `_compute_shampoo` is new.

- `class MuonAdamW(torch.optim.Optimizer)`: Single-GPU optimizer applying Muon (polar-decomposition-based update) to 2D matrix weights and AdamW to 1D/scalar weights.
- `class DistMuonAdamW(torch.optim.Optimizer)`: Multi-GPU Distributed DDP optimizer with asynchronous `all_reduce` and `all_gather` for Muon matrix steps.
- `def adamw_step_fused(...)`: CUDA-fused AdamW update step.
- `def muon_step_fused(...)`: CUDA-fused Newton-Schulz / Muon matrix update step.

#### [`nanochat/engine.py`](file:///home/seqaeon/Downloads/nanochat/nanochat/engine.py)
Inference, sampling, and KV-cache management engine.

- `class KVCache`: Manages Key/Value state for FlashAttention-3 and standard causal generation.
  - `prefill(x)` / `insert(k, v)`: Handles prefill sequence caching and token-by-token update.
- `class Engine`: High-level generation engine.
  - `generate(prompt, max_tokens, temperature=1.0, top_k=None, seed=None)`: Auto-regressive stream generation with top-$k$/temperature sampling.
- `def generate_block_kv(model, tokens, max_tokens, ...)` / `def generate_ar_kv(...)`: SAP's symmetric KV-cached decode loops (one trunk pass per T tokens vs per token), used for timing. Accept one prompt or a (B, L) tensor of prompts. Note: KV-cached and full-recompute trunk states differ by ~1e-2 because both attention paths cast to bf16.

#### [`nanochat/common.py`](file:///home/seqaeon/Downloads/nanochat/nanochat/common.py)
Shared training utilities and distributed coordination.

- `def setup_ddp()`: Initializes DistributedDataParallel environment (`torch.distributed`).
- `def print0(*args, **kwargs)`: Rank 0-only print utility.
- `def get_lr(it, warmup_iters, max_iters, min_lr, max_lr)`: Cosine decay learning rate scheduler with linear warmup.
- `def autodetect_dtype()`: Detects bfloat16 / float16 hardware support.

#### [`nanochat/checkpoint_manager.py`](file:///home/seqaeon/Downloads/nanochat/nanochat/checkpoint_manager.py)
State persistence for distributed models.

- `class CheckpointManager`: Handles saving/loading model weights, optimizer states, step counters, and random states safely across distributed ranks.

#### [`nanochat/dataloader.py`](file:///home/seqaeon/Downloads/nanochat/nanochat/dataloader.py) & [`nanochat/dataset.py`](file:///home/seqaeon/Downloads/nanochat/nanochat/dataset.py)
Data pipelines for pretraining.

- `class DistributedDataLoader`: Streams tokenized parquet shards across DDP ranks with perfect sharding and no duplicate sequences.
- `def get_dataset(split, data_dir)`: Loads data files for pretraining or validation.

#### [`nanochat/tokenizer.py`](file:///home/seqaeon/Downloads/nanochat/nanochat/tokenizer.py)
BPE Tokenizer wrappers.

- `class RustBPETokenizer`: High-speed `tiktoken` wrapper with custom Rust BPE training capabilities.
  - `encode(text)` / `decode(ids)`: Convert between text strings and token ID sequences.
  - `render_conversation(conversation)`: Tokenize structured chat turn conversations.
- `class HuggingFaceTokenizer`: Fallback wrapper for HuggingFace `tokenizers`.

#### [`nanochat/flash_attention.py`](file:///home/seqaeon/Downloads/nanochat/nanochat/flash_attention.py)
FlashAttention-2 / FlashAttention-3 wrappers and Triton fallback kernels.

#### [`nanochat/fp8.py`](file:///home/seqaeon/Downloads/nanochat/nanochat/fp8.py)
FP8 precision quantization helpers and FP8 linear layers.

#### [`nanochat/execution.py`](file:///home/seqaeon/Downloads/nanochat/nanochat/execution.py)
Hardware environment setup, CUDA device assignment, and multi-node initialization.

#### [`nanochat/code_metrics.py`](file:///home/seqaeon/Downloads/nanochat/nanochat/code_metrics.py)
Every metric section 6 of the SCH plan requires on every run, behind one entry point so an arm is
either fully instrumented or not run at all.

- `run_all_diagnostics(model, build_val_loader, token_bytes, vocab_size, ...)`: the orchestrator
  called at the end of `scripts/base_train.py`. `build_val_loader` is a zero-argument factory
  because each probe takes its own independent pass over the validation stream.
- `frequency_deciles(freqs, vocab_size, n_bins)`: bins balanced by *corpus mass*, not by
  vocabulary rank, so every decile has comparable statistical power. Types per bin are reported
  alongside, since that is the quantity the parameter-sharing argument is about.
- `evaluate_bpb_grouped(...)`: bits per byte overall, per decile, and over an arbitrary token
  subset (the held-out vocabulary). Same sum-nats / sum-bytes construction as
  `loss_eval.evaluate_bpb`, so the numbers are comparable to the training log and across
  tokenizers.
- `measure_logit_rank(...)`: SVD of the pre-softcap logit matrix after mean-centring across the
  vocabulary axis. Promotes `Phi` to fp32 and casts the head's input to fp32 for the probe (see
  the note above). Reports effective rank, the rank carrying 99% of the spectral energy, and the
  threshold-free stable rank. Rows are capped and columns subsampled, both of which preserve rank.
- `measure_anisotropy(...)`, `measure_holdout_rank(...)`, `measure_head_cost(...)`: representation
  collapse, where the true token lands in the ranking for held-out ids, and head wall-clock and
  peak memory reported separately from the model total.
- `write_sch_row(...)` and `SCH_CSV_COLUMNS`: one row per run appended to `sch_results.csv` next
  to the run directory, plus a `sch_metrics.json` sidecar carrying the full spectrum.

#### [`nanochat/loss_eval.py`](file:///home/seqaeon/Downloads/nanochat/nanochat/loss_eval.py) & [`nanochat/core_eval.py`](file:///home/seqaeon/Downloads/nanochat/nanochat/core_eval.py)
Validation cross-entropy loss evaluation and downstream evaluation benchmarks (Hellaswag, ARC, MMLU).

#### [`nanochat/report.py`](file:///home/seqaeon/Downloads/nanochat/nanochat/report.py)
Generates structured Markdown training reports, tracking GPU utilization, throughput (tokens/sec), parameter FLOPs, and loss metrics.

---

## 3. Top-Level Entry Points & Scripts

### 3.1 Main Pretraining Entry Points

- [`train.py`](file:///home/seqaeon/Downloads/nanochat/train.py): Top-level wrapper launching training jobs via `scripts/base_train.py`.
- [`scripts/base_train.py`](file:///home/seqaeon/Downloads/nanochat/scripts/base_train.py): Core pretraining script for Dense baseline, RemixedLinear, MST, and EET models. Sets up loss logging, validation, DDP, and optimizer steps.

### 3.2 Sweeps & Hyperparameter Research (`scripts/`)

- [`scripts/lr_sweep.py`](file:///home/seqaeon/Downloads/nanochat/scripts/lr_sweep.py): Grid search for optimal learning rates across dense and experimental model variants.
- [`scripts/actual_lr_research_sweep.py`](file:///home/seqaeon/Downloads/nanochat/scripts/actual_lr_research_sweep.py): Multi-stage coordinate-descent learning rate sweep script.
- [`scripts/warmup_sweep.py`](file:///home/seqaeon/Downloads/nanochat/scripts/warmup_sweep.py): Warmup iteration ratio sweeps for RemixedLinear & CCL models.
- [`scripts/p01_mst_sweep.sh`](file:///home/seqaeon/Downloads/nanochat/scripts/p01_mst_sweep.sh) – [`scripts/p35_conditioned_sweep.sh`](file:///home/seqaeon/Downloads/nanochat/scripts/p35_conditioned_sweep.sh): Shell scripts orchestrating multi-run experimental sweeps for paper iterations.
- [`scripts/c00_sch_phase0_rank.sh`](file:///home/seqaeon/Downloads/nanochat/scripts/c00_sch_phase0_rank.sh) – [`scripts/c04_sch_phase4_scale.sh`](file:///home/seqaeon/Downloads/nanochat/scripts/c04_sch_phase4_scale.sh): the five Structured Code Head phase sweeps, one per phase of `structured-code-output-heads-plan.md`. Each pins a matched `--target-tokens` from `scripts/code_head_budget.py`, keeps JSON completion state so a killed sweep resumes, and orders its groups so the go/no-go arm runs first.
  - `c00` Phase 0, the rank gate. Order-1 at the minimal code must measure effective rank exactly `B`; anything else is an implementation bug and a hard stop. Also measures the dense baseline's *achieved* rank and demonstrates the width cap (linear `g` pinned at `d`, MLP `g` reaching `M`).
  - `c01` Phase 1, the core experiment: the interaction-order ladder, the redundancy axis (the "redundancy beats depth" prediction, run first), the full baseline table (dense softmax, learned `W` at width `M`, frozen random binary `Phi`, Huffman hierarchical softmax, Oda-style order 1, VQ-Logits), the code-assignment arms, and a `V=131072` slice.
  - `c02` Phase 2, the three rank mitigations that can actually work: mixture of code heads, pointwise nonlinearity, dense residual hybrid. Head architecture cannot fix rank, so nothing else is run.
  - `c03` Phase 3, the input side: table, `E = CU` (the predicted rank-`B` collapse, run because a dramatic failure at exactly the predicted rank confirms the mechanism), expanded, nonlinear, tied. Runs the held-out vocabulary in both `target` and `full` modes.
  - `c04` Phase 4, scale confirmation at `d=1024`, 12 layers, `V=131072`. Confirms the *ordering* of methods holds, and re-runs the width-cap test at a second width.
- [`scripts/p08_mst_parity_sweep.sh`](file:///home/seqaeon/Downloads/nanochat/scripts/p08_mst_parity_sweep.sh): Stage 12 dense-parity ablation (G1/G2/G3) at L=8 with 3 seeds, plus an opt-in L=16 transfer check. Groups: `control combo g1 g2 g3 d16`; the combined arm runs before the singles as a go/no-go. Guards head_dim choices against `sub_dim` divisibility.

### 3.3 Architecture Diagnostics & Paper Probes (`scripts/`)

- [`scripts/eet_dense_diagnostic.py`](file:///home/seqaeon/Downloads/nanochat/scripts/eet_dense_diagnostic.py): Comparative layer-by-layer diagnostic between Early Exit Transformers and Dense baselines.
- [`scripts/mst_router_diagnostics.py`](file:///home/seqaeon/Downloads/nanochat/scripts/mst_router_diagnostics.py): Computes sub-transformer routing balance and gate entropy from MST checkpoints.
- [`scripts/p11_active_params.py`](file:///home/seqaeon/Downloads/nanochat/scripts/p11_active_params.py): Active parameters, active matrix parameters and active FLOPs for every arm in the paper's cost and downstream tables, built on the meta device so it needs no GPU. Reconstructs the headline SP2_k1 config from `p08_mst_parity_sweep.sh` and reads `estimate_flops()` rather than re-deriving the discount, so it cannot drift from the model. Also prints the per-layer matmul breakdown by role (attention / FFN / transition), which is what makes the "top-k gates 33.3% of a layer" claim a measured number: the gate reaches the FFN only, and the FFN is 44.4% of a layer at L=24. `--gate-attn` prices the alternative (62.5% gated, 37.5% active). Note it sets `window_pattern='SSSL'` to match base_train's default rather than GPTConfig's `SSSSL`, which changes the attention FLOPs term.
- [`scripts/p12_isoflop.sh`](file:///home/seqaeon/Downloads/nanochat/scripts/p12_isoflop.sh): isoFLOP profile. Every arm trains to the same ACTIVE training-FLOPs budget (`--target-active-flops`, default 9.0e18). C is set by the CORPUS, not by affordability: a shard holds ~252.8M characters, so `MAX_SHARDS=300` is roughly 18B tokens, the protocol is single-epoch, and each arm consumes `C / active FLOPs per token`, so the cheapest-per-token arm fixes the ceiling (MST 16/24/28/32 caps C at 1.03e19; adding L=12 would cap it at 5.73e18). 9.0e18 sits at 87% of that, and puts dense L=24 at 3.99x its own budget rather than the 7.45x a smaller C would give, which is what makes the 1B-scale points citable. Dense L=16/18/20/24 (537M to 1.38B active) against MST L=16/24/28/32 (388M to 1.62B active), four points each; both arms straddle their vertex, dense between L=18 and L=20 and MST between L=24 and L=28. Cost 8 x 9.0e18 = 7.2e19.
- MST L=20 is excluded as its off-trend ladder point (1.126x against 1.223x at L=16 and 1.284x at L=24). No mechanism was found: multi-scale windows are assigned per stream, not per head, so head count cannot disturb them; and `sub_dim = 64 mod 128` does break tensor-core alignment for MST where it never does for dense, but that is a speed effect while the anomaly is in bpb per FLOP. L=12 shares L=20's geometry and has never shown the effect, so a single bad run is likelier than anything structural. L=28 is in the same geometric class and is kept deliberately, so its residual tests the hypothesis; if it lands off the curve, fit 16/24/32 instead, which is clean end to end. The MoL arm exists but is excluded from `--arms all` because its per-block projections make it far slower to train than either baseline; `--mol-only` still runs it.
- **`--timer-only`** costs a sweep before you commit to it. Every arm runs `TIMER_STEPS` (default 12) steps with `--timing-probe-steps`, which derives `num_iterations`, the LR schedule and the batch size from the real budget and prints them, then ends through the normal `last_step` path so the final eval and save are timed too. Each arm is projected as measured startup overhead plus `full_iterations x dt`, and the projections are summed. A timer pass writes to `<out>/timer` and never calls `mark_done`, so it cannot mark a real arm complete.
- [`scripts/p13_isodata.sh`](file:///home/seqaeon/Downloads/nanochat/scripts/p13_isodata.sh): iso-data profile, the same shape but fixing tokens instead of FLOPs (default 1,167,968,256, again MST L=16's budget). Results are in `mst_isodata.html`. Read with care: a single token count puts arms whose own optimal budgets span 5.3x at very different distances from their optima, and in the measured run MST's multiplier and its starvation fraction fall together, so the profile cannot separate depth from undertraining.

Both profile scripts take `--arms dense|mst|mol|all` (aliases `--dense-only`, `--mst-only`, `--mol-only`) plus optional positional depths, so a single arm can be launched on its own machine: `bash scripts/p12_isoflop.sh --mst-only 24` runs exactly that one. Positional depths replace the built-in list for whichever arms run, and `SWEEP_LOG` overrides the log path as it does in the other sweep scripts. The MoL arm is `1+3of15` at `d_thin = D/4` with `--mol-per-block-ve 1`, the topology that reproduces the published parameter counts exactly (L=16: matrices 314,817,536, total 449,035,296, active FLOPs 8.094035e8) and the configuration their paper proposes, matching MST's `--mst-per-stream-ve`. Its defaults are 8/12/16 rather than MST's 12/16/24 because the per-block projections make it far costlier per token (8.09e8 against MST's 5.75e8 at L=16), so those depths bracket the same budget. Pointing both halves at the same `OUT_BASE` merges their state files and each half skips the other's finished arms; pointing them at different ones keeps them independent and the halves are combined at plotting time.

Both profile scripts are resumable at two levels. Arm level: a completed arm is recorded in `<out>/p1{2,3}_state.json` and skipped on re-run, and an arm counts as complete only if it left `<arm>/depth_<d>/results_depth_<d>.tsv`, so a sweep that exits 0 without training is retried rather than silently dropping a point. Mid-arm: `research_compare.py` finds the last checkpoint under the arm's stable `--out-dir` and passes `--resume-from-step`, so an interrupted arm continues from its last periodic save (`--save-every 200`). INT and TERM stop the whole script after the current arm rather than being read as an arm failure and advancing to the next multi-hour run; the exit status is 130 when interrupted, 1 when arms remain, 0 only when the profile is complete.
- [`scripts/p10_mfu_microbench.py`](file:///home/seqaeon/Downloads/nanochat/scripts/p10_mfu_microbench.py): Isolates MST's wall-clock gap against dense using pure GEMM shapes (no model, no flash_attn, no data), so it runs anywhere with CUDA. Four sections. `[A]` block-diagonal forward at `d` against the dense GEMM at `D` it replaces, testing whether MST's throughput is simply that of a dense model of width `d`. `[B]` the weight-gradient GEMM (`d x d` output, `K = B*T`), comparing one batched call against a loop of unbatched calls and against `torch._grouped_mm`, since only the unbatched forms let cuBLAS pick split-K. `[C]` prices `_batched_linear`'s permute-reshape against a contiguous stream-major layout. `[D]` the decisive one: runs a whole layer (forward, dgrad, wgrad) as block-diagonal and as masked dense, and reports `throughput(d)/throughput(D)` against the crossover threshold `1/N` below which the block-diagonal structure is a net wall-clock loss. Device-relative by construction: a 20-SM laptop reports 0.747, a 132-SM H100 about 0.22, so only the training GPU's numbers mean anything. `--sweep` runs the D ladder; set `--tokens` to the training harness's actual `B*T` per rank.
- [`scripts/remix_diagnostics.py`](file:///home/seqaeon/Downloads/nanochat/scripts/remix_diagnostics.py): Analyzes template bank orthogonality and collapse in RemixedLinear models.
- [`scripts/conditioning_headroom.py`](file:///home/seqaeon/Downloads/nanochat/scripts/conditioning_headroom.py): Measures theoretical conditioning capacity per layer before training.
- [`scripts/code_assign.py`](file:///home/seqaeon/Downloads/nanochat/scripts/code_assign.py): Builds the `(V, B)` uint8 code matrices consumed by `--sch-code-mode file`. Semantic codes from pretrained embeddings via ITQ (rotation-optimised binary hashing) or a residual balanced binary partition in the spirit of RQ-VAE semantic IDs, plus ECC, random, frequency and semantic-plus-parity modes. Repairs code collisions by moving each duplicate to its nearest free code in Hamming distance. `--report` prints density, sampled minimum Hamming distance, `M` at orders 1 to 4 against the ~1000 rank threshold, and a semantic-coherence number (Spearman correlation between code Hamming distance and embedding cosine distance, which is approximately 0 for a random code). `--build-freq-table` computes `freq_table.pt`, shared with EET's `FrequencyPrior`.
- [`scripts/code_head_budget.py`](file:///home/seqaeon/Downloads/nanochat/scripts/code_head_budget.py): Prints the single integer token budget every SCH sweep pins with `--target-tokens`, computed from the DENSE arm so the code arms are not silently under-trained.
- [`scripts/code_head_diagnostics.py`](file:///home/seqaeon/Downloads/nanochat/scripts/code_head_diagnostics.py): Post-hoc diagnostics on a trained checkpoint, and the explicit PASS/FAIL print of the Phase 0 rank gate.
- [`scripts/paper_bench.py`](file:///home/seqaeon/Downloads/nanochat/scripts/paper_bench.py): Standardized GPU benchmark suite for paper evaluation.
- [`scripts/paper_throughput.py`](file:///home/seqaeon/Downloads/nanochat/scripts/paper_throughput.py): Generates throughput curves (tokens/sec vs FLOPs) comparing Dense vs RemixedLinear.
- [`scripts/paper_probe.py`](file:///home/seqaeon/Downloads/nanochat/scripts/paper_probe.py): Probes internal representations and stream dynamics across model layers.
- [`scripts/verify_flops.py`](file:///home/seqaeon/Downloads/nanochat/scripts/verify_flops.py): Analytical FLOP and parameter count validator.

### 3.4 Data & Log Utilities

- [`prepare.py`](file:///home/seqaeon/Downloads/nanochat/prepare.py): Downloads and tokenizes training datasets into parquet shards.
- [`parse_log.py`](file:///home/seqaeon/Downloads/nanochat/parse_log.py): Extracts loss metrics, step timing, and validation scores from sweep log files.
- [`get_dim.py`](file:///home/seqaeon/Downloads/nanochat/get_dim.py): Computes dimension matching for target FLOP budgets.

---

## 4. Test Suite (`tests/`)

Located at [`tests/`](file:///home/seqaeon/Downloads/nanochat/tests).

- [`tests/test_remixed_linear.py`](file:///home/seqaeon/Downloads/nanochat/tests/test_remixed_linear.py): Unit tests for `RemixedLinear` weight initialization, router integration, and bias handling.
- [`tests/test_remixedlinear_fused.py`](file:///home/seqaeon/Downloads/nanochat/tests/test_remixedlinear_fused.py): Numerical parity tests and speed benchmarks comparing fused `RemixedLinearFused` vs reference PyTorch implementation.
- [`tests/test_triton_fused_kernel.py`](file:///home/seqaeon/Downloads/nanochat/tests/test_triton_fused_kernel.py): Triton kernel tests verifying template mixing matmul against PyTorch reference.
- [`tests/test_conditioned_linear.py`](file:///home/seqaeon/Downloads/nanochat/tests/test_conditioned_linear.py): Verifies conditioning gate mechanisms and backward gradients.
- [`tests/test_quantile_router_topk.py`](file:///home/seqaeon/Downloads/nanochat/tests/test_quantile_router_topk.py): Tests top-$k$ quantile router semantics, gradient flow, and causal masking.
- [`tests/test_mst_parity_fixes.py`](file:///home/seqaeon/Downloads/nanochat/tests/test_mst_parity_fixes.py): Stage 12 dense-parity fixes — proves G1 moves `head_dim` at constant matrix params/FLOPs with the batched path intact, G2 delivers unit-RMS activations into `lm_head`, G3 widens the VE table to `N*d` with distinct per-stream slices and no FLOP change, plus a combined forward/backward step.
- [`tests/test_eet_losses.py`](file:///home/seqaeon/Downloads/nanochat/tests/test_eet_losses.py): Unit tests for EET early exit losses, exit depth weighting, and router entropy penalties.
- [`tests/test_eet_new_features.py`](file:///home/seqaeon/Downloads/nanochat/tests/test_eet_new_features.py): Tests depth affine transforms, capacity annealing, and departure summaries in EET.
- [`tests/test_engine.py`](file:///home/seqaeon/Downloads/nanochat/tests/test_engine.py): Tests KV-cache behavior, sampling determinism, temperature scaling, and multi-sample generation.
- [`tests/test_block_head.py`](file:///home/seqaeon/Downloads/nanochat/tests/test_block_head.py): SAP block head: exact modes normalise over all blocks, the IWAE bound is a bound and tightens with samples, KL = 0 when posterior = prior, causal slots cannot read their own or later inputs, no future context, training loss = next-token loss + lambda x block loss, inverse-CDF round trip, energy-score properness, decode-loop equivalence (through full recompute), KV numerics, FLOPs scaling.
- [`tests/test_attention_fallback.py`](file:///home/seqaeon/Downloads/nanochat/tests/test_attention_fallback.py): Tests fallback PyTorch attention when FlashAttention is disabled or unavailable.

### `scripts/ensure_tokenizer.py`
Makes a tokenizer directory usable, building whatever is missing and skipping whatever is not.
Safe to call on every sweep invocation.
- Ensures `tokenizer.pkl` and `token_bytes.pt` (via `scripts.tok_train`) and `freq_table.pt`
  (via `code_assign.build_freq_table`). The frequency table is not needed to train, but
  without it the decile breakdown silently disables itself and a hierarchical-softmax arm
  falls back to a balanced tree instead of a Huffman one.
- Refuses a directory whose `vocab_size` is not the one requested, so a sweep pinned to one
  vocabulary cannot silently run against another. The message distinguishes "BPE ran out of
  corpus" from "wrong tokenizer in this directory".
- `scaled_defaults(vocab_size)` scales `max_chars` and `doc_cap` linearly from tok_train's
  32768 baseline (2e9 chars, 10k per document), because byte-pair frequency is roughly
  Zipfian: the rank-131072 pair occurs about 4x less often than the rank-32768 one, so
  reusing the 32k slice estimates the tail merges from proportionally less evidence. At
  V=131072 that is 8e9 chars and a 40,000 doc cap. Both overridable.
- `_preflight_data` refuses before training when the corpus is empty. `list_parquet_files`
  CREATES the directory and returns `[]` when the path is missing, so `tok_train` processes
  0 sequences, writes a 265-token tokenizer and exits 0; the failure would otherwise surface
  as a vocabulary mismatch after the training run had been spent.
- A tokenizer of the wrong size on disk is REBUILT, not skipped. Skipping it and failing the
  size check afterwards is an unbreakable loop.
- `--max-chars` builds a cheap tokenizer for a smoke test; `--eval` additionally runs
  `tok_eval`, which is diagnostics and never fatal.
Called by `c01`, `c04` and `c08`, which previously printed the build commands and stopped.

### `scripts/c07_monarch_depth_ladder.sh`
OPEN_QUESTIONS Q10. One arm, `MON_M1024`, at whatever depths are given (default 8 12 16);
dense legs are assumed to exist. Refuses any depth where `M < d`, because the rank ceiling
would drop below `d+1` and the arm would stop being the full-rank head c06 validated. Pins the
dense token budget and prints it with an instruction to check it against the dense leg.

### `scripts/c08_vocab131k_bench.sh`
Dense and `MON_M1024` at V=131072, default depth 8. Builds the 131k tokenizer if absent. The
decisive comparison is the depth-8 gap against the known V=32768 depth-8 gap of +0.0788: same
depth, same width, only the vocabulary moves, which is the one manipulation the c07 ladder could
not perform. Pre-registers the break-even gap (+0.1478) and what each outcome would mean.

### `scripts/c15_rerank.sh`
The rerank head at V=32,768 depth 8. Prints the cost-side ceiling in its header, because
that is why the sweep exists: a free head is worth +0.0318 bpb at depth 8 and the
headline arm costs 1.0011x, so its hurdle is a 0.00003 bpb GAIN. Five arms: `DENSE`,
`RERANK_k${TK}_r${TR}`, `RERANK_full_r${TR}`, `MOS2_r${TR}`, `TEMP`, plus `RERANK_r0`
under `--with-control`. `TK` and `TR` are plain scalars rather than a loop, because the
arm extractor in `tests/test_code_head.py` cannot resolve a variable inside a `for` list
and substitutes `$VAR` by word boundary, so a loop variable that prefixes an environment
variable silently corrupts the arm.

### `scripts/c14_nfh.sh`
The nonnegative-factorized head. `NFH_cp_R1` (LightRNN) and `NFH_cp_R32` run by
default, because the gap between them IS the claim, plus `NFH_global` as the hedge.
`--with-controls` adds the assignment controls and the two-axis factorisation;
`--with-baselines` adds a cost-matched low-rank arm, `HSOFTMAX` and `TIERED` (both
implemented and never run), and a fresh dense leg.

Vocabulary-aware by construction, because both quantities that decide the trade move
with V and in opposite directions: the family fits better at V=32,768 (oracle 0.00509
bpb against 0.00970) while the cost ratio `R(1 + sum_g K_g)/V` gets worse (0.0947x
against 0.0393x). `VOCAB=32768` is therefore a conservative screen at roughly 5x less
compute, with `DIMS`, `DIMS2`, `SLOPE` and `PERM_PATH` all deriving from it; the axes
are literals with a tiling guard that replaces them if they do not tile `VOCAB`, and
`LOWRANK_C` is likewise a literal that the baselines block recomputes and refuses to
run on if it has drifted. Both are literals only because the arm extractor in
`tests/test_code_head.py` cannot read a command substitution.

Pins the dense token budget through `scripts/code_head_budget`, which matters more
here than for any previous head: `cp` has 25x fewer head parameters than a dense
softmax, so on base_train's default sizing the arms would draw a smaller budget and
the result would be confounded by data rather than architecture.

### `scripts/c05_sch_phase5_alternatives.sh`
Phase 5 sweep: the five directions that survive the c00 post-mortem. Groups
`baseline product mixture monarch tree free`, 21 arms. Same format and token pinning as
`c00_sch_phase0_rank.sh`. `PROXY_CKPT=<dense ckpt>` additionally fits product codes to
that checkpoint's output embedding; without it the fitted arms are skipped rather than
silently falling back to the hash control. The comparison that decides the sweep is
against `BASE_learned_W`, not `BASE_dense`.

### `output-head-efficiency-directions.md`
Companion to the SCH plan, written after Phase 0. Records the reframe (the head is 28% to 51%
of FLOPs at V=131k-262k, not 9% at V=32k; structural cheapness is worth 3x what freezing is
worth), the three live hypotheses for the measured 0.716 bpb freezing cost, the K-ary
product-code proposal with its gather-and-add fast path, and the ranked screening plan.

### `scripts/sweep_report.py`
Measures every arm of a finished sweep in one pass, replacing a per-arm loop over
`code_head_diagnostics` and `code_head_subspace`. Point it at the depth directory
(`python -m scripts.sweep_report out/c05_sch_phase5/d4`).
- `find_arms(dir, pattern)` keys on the presence of `model_*.pt`, so arms that crashed are
  skipped instead of killing the run.
- `cached_metrics` / `quality` flatten what `base_train` already wrote and prefer `bpb`,
  falling back to `val_bpb`, so an arm is never dropped for a metric it was not asked to produce.
- `build_reference` finds the sweep's own dense arm automatically and refuses one with fewer
  than 8 learned directions, because capture measured against an under-trained head is capture
  of noise.
- `head_basis` / `capture_of` add the column the per-arm tools cannot: how much of the dense
  arm's logit energy each frozen head can actually reach, reported both raw and after the
  dominant unigram direction that a per-token bias supplies for free.
`--cached` harvests instead of recomputing; `--arms REGEX` restricts; one failing arm is
reported and does not stop the rest. Writes `<dir>/sweep_report.csv`.

### `scripts/code_head_subspace.py`
CPU-only subspace diagnostics for code heads, run against a trained dense checkpoint.
Answers "is colspace(Phi) the *right* M-dimensional subspace", where the rank probe in
`code_metrics.py` only answers "how many dimensions does it have".
- `learned_direction_count(S, V, d)` compares the head's spectrum against a random
  matrix of identical Frobenius norm and reports how many directions rise above the
  bulk. The tool refuses to report below 8, because on an early checkpoint every
  unfitted basis scores at the random baseline and every fitted one scores high, which
  reads exactly like a real result.
- `load_dense_head(path)` loads and validates a dense `lm_head.weight`, rejecting the
  NaN-poisoned checkpoints nanochat writes for aborted runs.
- `orthonormal_basis(phi)` centres Phi on the vocabulary axis and returns a QR basis.
- `analyse(...)` reports, per interaction order: capture of the dense head's logit energy by
  colspace(Phi); capture with one extra freely chosen direction, which is what a per-token
  bias buys; capture of the residual after the dominant unigram direction is removed; and the
  oracle top-M SVD subspace for both.
- Builds privileged codes from the dense head's own singular vectors (median threshold,
  dyadic staircase on u1, hybrid) to bound what any binary code assignment could reach.
Read the residual column. The others are dominated by the unigram direction a bias supplies
for free.

Phase 5 additions (see `output-head-efficiency-directions.md`):
- `build_product_codes(V, groups, codebook, source, seed, path)` assigns each token a
  K-ary codeword. Sources `hash` (the K-ary analogue of `code_mode=binary`), `random`
  (null control), `file` (k-means assignment from `code_assign.py --mode product`).
- `product_gather(z, assign)` is `g(h) Phi^T` when Phi is one-hot per group: g gathers and g-1
  adds costing `V*g` instead of `V*M`. Selected by `sch_product_impl=gather`, and MEASURED 4.76x
  slower than the matmul it replaces because the backward is a 512-way-contended `index_add`.
  The default `sch_product_impl=dense` materialises Phi and runs a GEMM at `4*V*M`;
  `flops_per_token` branches on which one is active. See OPEN_QUESTIONS Q8.
- `_whiten_phi(phi)` returns `Phi (Phi^T Phi)^-1/2`. Same column space, orthonormal
  columns. For a linear g this is provably a reparameterisation, so any bpb movement in
  the sweep is an optimisation effect and nothing else.
- `StructuredCodeHead._build_one_phi(V, M, dtype, k)` builds component k's Phi from a
  per-component seed; `_sparse_mixture` routes top-k over components with dispatch on the
  flattened token axis, so the cost tracks k rather than K. The constructor refuses
  `sch_mixture_per_phi` with a full monomial expansion, which would be a silent no-op
  because reseeding the full expansion returns the same monomial set.
- `MonarchHead`: two block-diagonal factors with a transpose between, costing
  `d*M + V*m1` instead of `V*d`, fully learned so alignment is not a question.
  `m1` is per-token capacity (the K of each block's GEMM), `m2` the number of
  independent blocks, each owning `V/m2` vocabulary rows. Two capacity knobs sit on
  top of it, both aimed at the depth-12 ceiling where the factorisation axis runs
  out: `sch_monarch_perm` (Q12) changes which words share a block and costs nothing,
  and `sch_residual_rank` (Q13) adds `r` directions shared across all blocks for
  `r(d + V)` MACs. The bias, the residual and the reshape all write into one
  full-width tensor in place; a second would not fit at V=131072.
- `RerankHead` (`sch_head_type=rerank`): the first head here that goes for a QUALITY
  gain rather than a cost saving, because at V=32,768 the cost side is capped: a head
  costing nothing is worth +0.0762 / +0.0318 / +0.0168 bpb at depth 4 / 8 / 12, and
  every approximation measured in this project costs more than that.

      z     = W h            dense, exact, unchanged
      K     = topk(z)        the model's OWN top-k, no corpus statistics
      z[K] += U[K] . f(h)    rank-r nonlinear correction, on K only
      p     = softmax(z)     over all V, exact

  Cost is `k*r` rather than `V*r`: 1.0011x the dense head at k=64, r=32, V=32,768, so
  it needs a gain of 0.00003 bpb to be Pareto-positive. `f` is nonlinear and `K` is
  data dependent, so `rank_ceiling` is not `d+1`; that is what Mixture of Softmaxes
  buys at R times dense cost. `custom_loss=False` and `emits_logits=True`, so the
  ordinary `F.cross_entropy` path and every existing eval, decile metric and rank
  probe work unchanged, which is the point of keeping the forward dense.
  Four modes, so the ablations are the evidence: `topk` is the mechanism, `full`
  applies the same correction over all V and prices what the restriction buys, `mos2`
  is the published comparator, `temp` is a per-token logit scale and the floor the
  mechanism must clear. `sch_rerank_rank=0` reduces it to dense bit-exactly.
  Two traps paid for here: `up` is initialised small but NOT zero, because the
  correction is a product and a zero start leaves `down` with identically zero
  gradient on step 0, which DDP reports as a parameter that received none; and `corr`
  is cast to `z.dtype` before `scatter_add_`, which requires them to match exactly and
  which autocast does not guarantee.
- `NonnegFactorHead` (`sch_head_type=nfh`): factorises the DISTRIBUTION, not the
  matrix, which is the first head here that is not bounded by the three measured
  walls in LEARNINGS ("The head's rows do not cluster"). Two modes.
  `cp`: `p(w|h) = sum_r pi_r(h) prod_g alpha^g_{r,a_g(w)}(h)` over a bijection
  `w -> (a_1..a_G)` tiling the PADDED vocabulary, every factor a softmax, so the
  partition function is 1 by construction. Cost `d R (1 + sum_g K_g)`, which is
  0.039x the dense head at V=131,072 with 64x64x32 and R=32. R=1 is exactly
  LightRNN and is the paper's control.
  `global`: `log p(w|h) = lse_j(u_j + v_{w,j}) - lse_j(u_j + s_j)` with a learned
  `V x m` log-table and `s_j = lse_w v_{w,j}` reduced once per FORWARD rather than
  per token. Cost `d m`, independent of V.
  `loss()` is the training path and never builds a V-wide tensor: two or three GEMMs
  of width `R*K_g`, then `R` gathered scalars per axis. `forward()` builds full
  log-probs for evaluation and generation only, in TOKEN order, so `permutes_vocab`
  stays False and the generation path needs no special handling; it carries
  `@torch._dynamo.disable` because its chunk loop would otherwise unroll into one
  graph, which is how the proposal head hit a 30-minute compile. `rank_ceiling`
  returns V rather than d+1: the log-sum-exp lifts the log-prob matrix off the
  d-dimensional bound every linear head in this file is clamped to.
  Measured at V=32,768 d4 compiled, 1 graph and 0 breaks: 19.96 ms/step for cp R=32
  against dense's 31.23, at 0.370 GB peak against 0.574.
- `_nfh_dims(spec, padded_vocab_size, groups)`: parses the code axes or derives them
  for a power-of-two vocabulary, largest axis first. They must multiply to the padded
  vocabulary exactly, since anything else silently drops or duplicates words while
  still returning a normalised vector.
- `build_vocab_permutation`: builds the Q12 permutation (`none | random | freq |
  file`). Confined to `[0, vocab_size)` and identity on the padding tail, because
  `GPT.forward` slices the padding off before the loss. Rebuilt in
  `MonarchHead.init_weights`, not just `__init__`, because `base_train` builds on
  meta and then `to_empty`s every buffer to garbage.

#### [`scripts/build_vocab_permutation.py`](file:///home/seqaeon/Downloads/nanochat/scripts/build_vocab_permutation.py)

Offline builder for `--sch-monarch-perm=file`. `freq` and `random` mirror the in-head
modes; `cluster` runs k-means over a dense checkpoint's `lm_head` rows (the matrix
Monarch replaces) and then a capacitated assignment, because the head needs exactly
equal blocks and plain k-means does not give them. `nested` (for
`--sch-nfh-perm=file`) recurses that same capacitated k-means once per code axis and
orders the LEAF axis by a predictability proxy from `--acts`, so index j means "the
j-th likeliest word in my cell"; measured worth 10 to 20% of the offline
reconstruction error at every rank and factorisation depth, for zero run-time cost.
`vocab_rows` also accepts the `{"acts", "lm_head"}` payload that
`dump_head_acts.py --also-head` writes, which is now the usual source because one
file supplies both the rows to cluster and the activations that order the leaf. Tokens are placed in order of how
much they prefer their best centroid over their second, so the tokens that pay for
the balancing are the ones that cared least.


### `fully-binary-transformer-plan.md`
Research-direction plan (v1) for the fully binary transformer: strict W1A1, every learned
parameter one bit and every matmul operand one bit, interfaces and optimiser included. Records
why the cost axis changes from FLOPs to (bit-ops, energy, training-state bytes), since a W1A1
model has identical FLOPs/token to its dense twin; what the closed binary-code-head arc
contributes (the five numbers that kill families here too); the hole in the 1-bit literature,
which is that BitNet/FBI-LLM/QuEST all exempt the embedding, the head and the norms, and those
are 35-68% of the model at our vocabularies and depths; the one-bit-of-credit argument that
licenses a 5-bit-per-parameter counter optimiser; the hardware confound (b1 XOR MMA removed in
sm_90, AND mode survives, XNOR recoverable as `4*popc(a AND b) - 2*popc(a) - 2*popc(b) + n`);
and the closed-form partition function over a linear-coded vocabulary that is the only
construction in this project to escape K4. Four Phase-0 oracles, all but the kernel gate
runnable without a GPU. Pre-registered kill table in section 8.

### `scripts/b00_binary_phase0.sh`
Phase 0 driver for the fully binary transformer, three groups `cost kernel sensitivity`,
same conventions as the `c0*` sweeps: `--force`, `--group`, positional depths, JSON state at
`out/b00_binary_phase0/state.json` so a killed run resumes, and `SWEEP_LOG` to tee. Env:
`CKPT VOCAB_SIZE SEQ_LEN WINDOW_PATTERN TOKENIZER_DIR DATA_DIR MAX_SHARDS DEVICE_BATCH_SIZE
EVAL_STEPS KERNEL_ROUNDS`. `cost` runs anywhere; `kernel` and `sensitivity` want a rented card.

### `nanochat/bitcost.py`
The three axes FLOPs cannot see: bit-operations, energy, training-state bytes. Exists because
a W1A1 model has identical FLOPs/token to its dense twin. `PrecisionSpec` carries per-component
bit widths (`w/a/embed/head/accum`); `OptimizerSpec` carries optimiser STATE bits per parameter,
and weight bits and state bits ADD rather than overlapping (an earlier `max()` made binary
weights look worth exactly 1.00x during training). `NANOCHAT_MUONADAMW` is 39.3 state bits,
MEASURED on this repo, not the textbook 96. Energy figures are marked PROVISIONAL and must be
pinned from Horowitz ISSCC 2014 before any joule number is quoted.

### `scripts/o4_cost_model.py`
O4. Gate first: reproduces parameter counts and FLOPs/token exactly at depths 4 and 8, and each
reference number carries the exact config that produced it, because the depth-4 figure came from
a `seq=1024, wp=L` run and comparing it against a `seq=2048, SSSL` model looked like a bug in the
cost model. `validate_optimizer_bytes` builds a real model, takes one step and sums optimiser
state. `matched_bytes_arm` is the one that matters: at equal inference bytes the binary model is
depth 27 / 1.98B params, issues 24.5x the MACs, and therefore needs a b1 kernel at >= 24.5x over
bf16 to tie on wall clock.

### `scripts/o5_sensitivity.py`
O5. Whole-model binarisation sensitivity: one component class at a time, everything else bf16,
val bpb. Two axes because "binarise" is easy to measure wrongly: `--binarise weights` is W1A16
and changes no operation at all, `acts` isolates the half BitNet will not take below 8 bits, and
`both` is the only arm matching the plan. Crossed with `--scale none|row`, a per-row float scale
being a float parameter. Component patterns verified against the 60 keys of
`out/dense_d8_V32k_model_001014.pt`. There is no norm arm: nanochat's `norm` is a parameterless
`F.rms_norm`. Projection oracle, so it PENALISES; a pass is strong evidence and a failure is weak,
the opposite of how K7/K8 read a free-fit oracle.

### `scripts/o3_kernel_gate.py`, `kernels/b1_probe.cu`
O3. Interleaved A/B of a `wmma` b1 GEMM against cuBLAS bf16 inside one process, because two
separate runs measure the thermal state rather than the kernel. Refuses sm_89 (Ada dropped INT1)
and warns on sm_90+ (XOR removed from hardware, emulated up to 5x slower), so a headline number
must come from T4, A10G or A100. Prints the 24.5x break-even alongside every ratio.

### `nanochat/binary.py`
Natively binary layers. `BinaryLinear` is `y = (sign(W)*alpha) @ (sign(x)*beta)` with alpha one
float per output channel and beta one per token, both the L1-optimal `mean|.|`; `binarise_acts=False`
is the W1A16 rung, which changes no operation. `BinaryEmbedding` is the learned 1-bit table of
section 4.4, `V x D` bits plus one float per row. `binarise_model_` swaps `nn.Linear` and
`nn.Embedding` in place, with `skip` for building the Phase 1 ladder in the order the in-situ
cost table gives. `_SignSTE` is a clipped straight-through estimator and is deliberately not a
contribution. **Latent weights must be initialised INSIDE the clip window**: a latent weight's
magnitude has no effect on a forward pass that sees only its sign, its only role is STE inertia,
and outside the window it receives no gradient and can never flip again. Initialising the
embedding at std=1.0 against clip=1.0 left 10.19% of weights dead at step 0 and frozen there;
`reset_parameters` and the donor-weight rescale in `binarise_model_` both now target `clip/3`.

### `scripts/o2_sign_agreement.py`
O2. Tests P1: does quality in a binary network track gradient SIGN agreement or magnitude?
Corruptions are chosen to move the two independently: `signonly` and `lognormal_s` hold agreement
at 1.00 while destroying magnitude, `flip_p` holds magnitude while setting agreement to 1-p.
**Optimiser is an axis because Adam is already approximately signSGD** (Balles and Hennig 2018),
so an effect visible only under Adam says nothing about binarity; SGD is the magnitude-sensitive
control and the result that matters is the dense-vs-binary interaction. Corrupts `p.grad` after
backward, which is exactly section 3.2's claim about the weight update; cheapening
activation-gradient propagation is a separate experiment.

### `scripts/o6_binary_smoke.py`
O6. Does a natively binary transformer train from scratch at all? NOT a result and barred from
being quoted as one. Three diagnostics a loss curve hides: **flip rate** (a rule that never flips
and one that flips constantly both look like convergence failure in the loss), **activation
balance** (a layer stuck near 0 or 1 is dead and stays invisible for a long time), and **dead
bits** (latent weights outside the STE clip window, which found the init bug above on its first
run). Gates on loss decreasing and no saturated layer.

### `scripts/b01_binary_ladder.sh`
Phase 1 ladder for the binary direction. Seven rungs built by SUBTRACTION from the fully binary
model via `--binary-skip`, ordered by O5's in-situ cost table: R0 dense, R1 W1A16 body (changes
no operation), R2 body W1A1 with fp interfaces (the BitNet-comparable point, and only a 1.23x
memory win), R3 +value_embeds, R4 +wte, R5 +lm_head (equals O6, should reproduce +0.1957), R6
threshold replacing scales. Budget pinned to the DENSE arm's via `scripts/code_head_budget.py`
the c05 way, because binary arms carry extra scale parameters and `get_scaling_params` is
`transformer_matrices + lm_head`, so per-arm Chinchilla would confound data with architecture.
Same conventions as c05: `--force`, `--rungs`, `--seeds`, positional depths, JSON state,
`SWEEP_LOG`. Tags are `${RUNG}_s${seed}` so `scripts/paper_collect.py` aggregates unmodified.
The open question it exists to answer: does O5's in-situ ORDERING survive from-scratch training?

### `tests/test_binary.py`
Eleven tests, each written because the failure actually happened. `log_alpha` is 1-D so it does
not reach Muon (checked on the param group's `kind`, not the optimiser class name, since
`MuonAdamW` is fused); every tensor receives gradient within ten steps, NOT at step 0, because
nanochat zero-initialises every `c_proj` and that blocks upstream gradient for exactly one
backward; zero-initialised rows are not frozen; binary and dense FLOPs are equal; and the three
scale modes behave, with `none` asserted to blow the output up because that is the point.

### SAP scripts (`sap_research_plan.md` v3)
- `scripts/sap_synthetic.py`: Stage A. Trains every block-head mode from scratch on a phrase HMM whose block joint is exact (forward algorithm); reports block-KL, the true total correlation, the trunk's own block-KL, the rate of impossible sampled blocks (mode mixing), latent sensitivity, and the pre-registered gate verdicts. `--smoke` checks the wiring in ~1 minute.
- `scripts/s00_sap_d8.sh`: Stage B on FineWeb-Edu at depth 8 (rented GPU). Every arm gets the dense arm's training FLOPs via `--target-flops` from `scripts/sap_budget.py`; `--post` runs the decode benchmark and generation eval per arm.
- `scripts/sap_budget.py`: prints the dense arm's total training FLOPs (Chinchilla tokens x FLOPs/token) for FLOPs-matched SAP arms.
- `scripts/sap_decode_bench.py`: tokens/sec of block decoding vs next-token decoding, same model, KV-cached, batch 1/16/128.
- `scripts/sap_eval_generation.py`: continues validation prefixes with both decoders; reference-model perplexity, distinct 3-grams, and pairs for an LLM judge.
- `scripts/sap_corpus_tables.py` (v4): counts adjacent and skip-1 token pairs on the training shards and writes the corpus tables (`nanochat/sap_tables.py`).
- `scripts/sap_oracle.py` (v4 Stage 0): on a trained dense model, block total correlation at T=2/4/8 (Monte Carlo marginals), lattice coverage, pair-MI capture by free and constrained rank-r CRFs and free CP mixtures (with parameter/entry ratios and a held-out split), cut MI (TT rank floor), and the within-class residual for `corpus_code`.
- `modal_sap.py` S03 section: `s03_stage0` (tables, then oracles), `s03_stage_a` (synthetic gate over every v4 head and variant x trunk gradient), `s03_dense` (dense reference at a depth, several seeds), `s03_stage2` (heads on the frozen dense trunk of `--depth`), `s03_stage3` (from scratch at the dense FLOPs of `--depth` x trunk gradient); post-evals (`s03_post`) time decode with CUDA graphs on at 32T and 256 tokens. Checkpoints go to `out/s03_sap/d<depth>/`; since 2026-10-03 the mainline depth is 4 (d8 reference `B1_dense_s1` exists on the nanochat1 workspace only). Capacity specs: `mode@D<m>S<share>` sets `--sap-depth-layers` / `--sap-depth-share`.
- Lanes, SAP S08 (`nanochat/lanes.py`, plan `s08_sap_lanes_plan.md`). One trunk pass emits L tokens S positions apart: a causal prefix, then L contiguous lanes written in lockstep.
  - `lane_layout`, `lane_rank`, `lane_mask`: a step-s input reads every input of steps up to s.
  - `lane_inputs`: the lane-start token `<|output_end|>` at each later lane's first input.
  - `LaneBatches`: lane-order val bpb through `evaluate_bpb(lane_mask=...)`.
  - `lane_step` and `generate_lanes`: one all-layer pass per step through `GPT._sap_depth_layers`, with the KV cache as prefix; graph-safe.
  - Plumbing: `GPT.forward(lane_mask=...)`, the attention's explicit-mask SDPA branch, `scripts/base_train.py --lanes / --lane-prefix-max / --lane-eval-prefix`, `scripts/sap_decode_bench.py --lanes` (CUDA-graph lane vs next-token timing) and `scripts/sap_eval_generation.py --lanes --ar-dir` (reference PPL of lane samples against another model's next-token samples).
  - Modal: `modal_sap.py::s08_lanes` (lane tax against full-context dense, two seeds, then speed), `s08_gen`, `s03_dense_frontier` (fewer-layer dense at matched FLOPs, timed in one container).
  - Tests: `tests/test_lanes.py`.
- RC-PTP, SAP S10 (`nanochat/ptp.py`, doc `s10_sap_tl_brainstorm.md`). A one-pass T=L generator in the style of C-PTP: each position reads embedded auxiliaries of earlier positions, and every token is picked at once.
  - Picks: `ordered_pick` / `ordered_cdf_bounds` (PTP's flat inverse CDF over a rank order), and `tree_pick` / `tree_bounds` (one uniform per level of a binary tree over the rank order; near-maximal coupling; cells are boxes whose volume is the token's probability). Helpers: `place`, `uniform_in`, `invert`, `semantic_rank`, `binary_digits`.
  - `RCPTP`:
    - a token-conditioned AR mode (`ar_logits`, `ar_sample`);
    - a staged generator (`stage_logits`): stage heads pick under the same auxiliaries and feed the next stage one position later (`stages`, `coupled`);
    - training auxiliaries from the AR mode (`inversion="ar"`, distil), exact self-inversion (`"seq"`, PTP Eq. 13) or AR init plus parallel sweeps (`"jacobi"`, `n_inv`);
    - `log_prob`: a sequential importance-sampling lower bound;
    - `diagnostics`: AR-mode KL, agreement and lead against the AR sampler, per-position gap.
  - Toy arms in `scripts/sap_s06_screen.py`: `rcptp`, `cptp_si`, `rcptp_indep`, `rcptp_idorder`, `cptp_seq`, `rcptp_seq`, `cptp_jac`, `rcptp_jac`, with `--ptp-coupling`, `--ptp-stages` and `--n-inv`. `evaluate` merges any model's `diagnostics`.
  - `scripts/sap_jacobi_oracle.py`: with the toy's exact conditionals, how many coupled refinement stages one-pass emulation needs (flat CDF, tree, Gumbel).
  - `scripts/sap_bridge_oracle.py` (S11): exact interface-first (nested-dissection) sampling of the toy HMM in log2(T) + 2 levels, against token-valued bisection with exact posteriors.
  - S11 window bisection: `nanochat/wbisect.py` (`wb_steps`: prefix left to right, then the block in window-bisection order; `two_stream_mask`, `two_stream_batch`, `pos_ids`, `WBBatches`, `wb_forward`: one two-stream pass scores every token exactly); `GPT.forward(pos_ids=..., head_from=...)`; `scripts/base_train.py --wb-window / --wb-prefix`; `scripts/sap_position_bpb.py` specs `name:ckdir:wbN`; Modal `modal_sap.py::s11_wbisect`; tests in `tests/test_wbisect.py`. Toy versions: `nanochat/bridge.py` (`BridgeLM`, `window_bisection_levels`, oracle/token/learned codes, endpoint and product-of-experts bridge heads), `scripts/sap_code_quality.py`, harness arms `bridge*`, Modal `s11_toy`.
  - S11 bridged lanes and decoding: `nanochat/wbisect.py` `bridged_lanes_steps` (L intervals; each interval's last n tokens placed coarse-to-fine in bisection order over intervals, then all intervals filled left to right in lockstep; (ceil(log2 L) + 1) * n + (interval length - n) steps), `WBDecoder` (cached two-stream decoding: one pass per step over the previous step's content and the current step's queries, static shapes per step, so each step captures as its own CUDA graph) and `wb_generate`; `scripts/base_train.py --wb-lanes`; `scripts/sap_position_bpb.py` specs `blL_N` and `lnL`; `scripts/sap_decode_bench.py --wb-window / --wb-lanes / --wb-prefix` (`graph_wb_seconds`: one graph per step replayed in order, against `graph_ar_seconds` for the same tokens); Modal `modal_sap.py::s11_blanes`, `s11_ladder` (dense / bl / wb / ln specs at token multiples, scored per position against one reference) and `s11_speed` (H100 timing of step schedules on one checkpoint).
  - S11 orders and sample quality: `nanochat/wbisect.py` `lane_order_steps` (S08 plain lanes as a two-stream order, the control) and `seeded_lanes_steps(N, P, K, m)` (middle-out lanes from an m-token seed window; killed in Stage 3c); `scripts/base_train.py --wb-order {bisect,lanes,seeded}`; `scripts/sap_position_bpb.py` specs `loL` / `sdK[_M]` and a per-offset lane cost report for `lnL` models (`lane_offset_report`); `scripts/sap_eval_generation.py --real` (true continuation scored as an anchor), unigram entropy, `--ar-temperature` / `--ar-top-p` (nucleus sampling for the next-token side via `nanochat/engine.py::sample_next_token(top_p=...)`); `scripts/sap_rescore_samples.py` (re-score saved samples and the real continuations under another reference); Modal `modal_sap.py::s11_gen` (temperatures, next-token settings), `s11_score` (eval-only rescoring of checkpoints in their own orders), `s11_rescore_samples`, and `s08_gen(gen_tokens=..., temperatures=...)`.
  - S11 separator oracle: `nanochat/lanes.py` `separator_mask` / `separator_batch` / `SeparatorBatches` (positions from split+m on reach the first half only through m slot positions); `scripts/base_train.py --sep-slots / --sep-split / --sep-full`; `scripts/sap_position_bpb.py` model specs `name:ckdir:sepM[full]`; Modal `modal_sap.py::s11_separator_oracle`; test in `tests/test_lanes.py`.
  - S13 (`s13_sap_brainstorm.md`):
    - `nanochat/splice.py`, SV-A splice codes. `splice_layout(N, P, L)` inserts a round-0 code slot before each later lane. `Splice` holds the code embedding and the code head. `splice_loss` gives an exact likelihood: code nats plus junction tokens masked to their class, folded onto the original positions for `loss_reduction='none'`. It is reached through `GPT.forward(splice=(P, L, lane_token))`.
    - `GPT.forward(extra_embed=...)`; `GPTConfig.splice_k` and the persistent `class_of_token` buffer.
    - `scripts/base_train.py --splice 1 --class-map PATH`.
    - `scripts/sap_brown_classes.py`: exact exchange-algorithm Brown classes from adjacent counts, with gains checked against brute force.
    - `scripts/sap_position_bpb.py`:
      - spec `name:ckdir:spL`;
      - `rows_hash` (printed on every eval; `--hash-only` computes it locally with no model);
      - `splice_offset_report`: absolute nats by lane offset and code nats per junction, with no reference model needed;
      - block bpb from P-1 and from P.
    - `scripts/sap_decode_bench.py --roofline D`: random-weight next-token against R-round lanes against one pass (`graph_onepass_seconds`), under CUDA graphs.
    - `scripts/sap_chunk_ae.py`: Q2 chunk autoencoder for SV-D's necessary condition.
    - Modal: `modal_sap.py::s11_ladder` spec `sp:L:MULT:SEED`, `s13_roofline`, `s13_chunk_ae`.
    - Tests: `tests/test_splice.py` (layout and visibility; the factorisation sums to 1 over every block).
  - S14 (`s14_sap_strict_tl_brainstorm.md`), the strict seed:
    - `scripts/sap_order_oracle.py`: the E0 order oracle and the E2a separator bound. Nothing is trained.
      - An any-order masked diffusion LM (LLaDA-8B-Base or Dream-v0-Base-7B) scores real text in a generation order twice: in parallel per step, and as the order's exact chain (one token per pass) (`score_order`).
      - TC = par − chain is same-step dependence. gap = chain(order) − chain(l2r) is the order's difficulty for the oracle.
      - Orders (`order_steps`): l2r, `bisectN` (window bisection via `nanochat/bridge.py`), `lanesL`, `snapW` (anchors snapped to sentence starts, `snap_levels`), `randomR`, and `blL_n` (S11 bridged lanes via `nanochat/wbisect.py::bridged_lanes_steps`).
      - `confR` (`confidence_steps`): confidence-ordered decoding, the masked-diffusion default. One pass per step via `MaskedLMOracle.confidence`.
      - `readings` also compares lanes with `conf` and `random` orders at equal steps (S15 L0b).
      - `separator_scores` and `separator_summary`: the NLL after a cut with the far past hidden, giving the bits any separator must carry.
      - `MaskedLMOracle`: logit-shift detection and known mask ids. `context_probe` flags an oracle that ignores right context. `ar_reference_bpb`.
      - `score_rows` / `finalize` / `readings`: sharded (`--shard`, `--num-shards`), resumable (`--raw`), `--merge`, and `--compare` (cross-oracle Spearman). Pre-registered `validity` and `readings` are written into the JSON.
    - Modal: `modal_sap.py::s14_order_oracle` (shards on H100s in images pinned to each model card's transformers, then a merge; `block`, `prefix`, `sep_cut` and `name` set the run) and `s14_order_oracle_compare`. The workers are `s14_oracle_llada`, `s14_oracle_dream` and `s14_cpu_job`. Results: `scratch/s14/`.
  - S15 (`s15_lanes_paper_plan.md`), the lanes paper:
    - `scripts/sap_position_bpb.py::lane_offset_report(..., n_rows)` adds absolute extra nats per lane, the reference's nats per token, and the per-lane excess by offset with its deficit (positive part) and recovery (negative part). The R1 scale test compares model sizes in the same units as the S14 oracle.
    - `scripts/sap_order_oracle.py::lane_profile`: the same per-offset deficit and recovery for `lanesL` orders in the oracle (the order's par minus the l2r chain at the same positions, lanes 1..L-1).
    - `modal_sap.py::s14_order_oracle --merge-only` re-merges a finished run's raw shards on CPU, for example to add `lane_profile`.
    - Results: `scratch/s15/`.
    - Runs reuse `s11_ladder`, `s11_score`, `s08_gen` and `s11_speed` at depth 12.
    - Tests: `tests/test_order_oracle.py`. It uses an exact Markov-chain oracle to check:
      - every chain equals the block NLL;
      - TC = 0 for independent tokens and for bisection of a Markov chain;
      - one token separates a Markov chain;
      - shift detection, and the causal-oracle probe;
      - sharded and resumed runs merge to the single run.
  - S16 (`s16_lanes_recovery_brainstorm.md`), mechanisms that raise lanes' recovery:
    - `nanochat/lanes.py::infill_layout` / `infill_rows` (S16-A, position-preserving infill rows):
      - one middle span per chunk; slots outside every middle in position order, then the middles;
      - true rotary positions via `pos_ids`, and a causal mask over the reordered row;
      - the first slot after each middle holds the lane-start token, so the row is an exact order.
    - `scripts/base_train.py`:
      - `--lane-infill-frac` / `--lane-infill-span` / `--lane-infill-gap`: that fraction of micro-steps trains on infill rows instead of lane rows;
      - `--lanes-mix` (S16-B, any-L lanes): each micro-step draws its lane count from the list.
    - Modal:
      - `modal_sap.py::s11_ladder` specs `ppi:L:F:MULT:SEED` and `mix:L1+L2+..:MULT:SEED`. Its scoring keeps an earlier run's dense reference when the call does not retrain it.
      - `modal_sap.py::s16_score`: scores checkpoints at any lane count (`TAG@L`) against one dense reference, with the deficit/recovery report.
    - Tests in `tests/test_lanes.py`:
      - the infill layout reads only earlier draws, except at the cold slots;
      - infill rows are a normalised distribution, and leaving a cold slot's input in place breaks the sum.
  - Modal: `modal_sap.py::s10_toy`, which runs `s10_toy_run` jobs on L4 for arms x T x depth x seeds.
  - Tests: `tests/test_ptp.py` (45).
- Trunk-depth slots (`nanochat/gpt.py`): `_sap_depth_tools` (gradient scaling and shared-or-copy blocks), `_sap_depth_entry` (slot entry state and x0: the trunk's state at the block start plus the token's embedding, or the `depth_mask` vector for a slot whose token is unknown), `_sap_depth_layers` (any slot set through the top m layers, reading the prefix keys/values at each layer under a slot-visibility mask), `_sap_depth_local_logprob` (exact chain rule), `_sap_depth_tree_logprob` (exact bisection-round factorisation; layout from `block_head.depth_tree_layout`), `_sap_depth_sample` / `_sap_depth_tree_sample` (graph-safe decoders that read the trunk's KV cache, or the trained copies' own cache from `sap_depth_copy_cache` / `sap_depth_extend_copy_cache`), `_sap_depth_loss` (training loss over per-row block starts).
- `scripts/base_train.py` v4 flags: `--sap-trunk-grad`, `--sap-lattice-k`, `--sap-pair-rank`, `--sap-tt-rank`, `--sap-cp-codes`, `--sap-nce-*`, `--sap-table-path`, `--sap-supp-*`, `--sap-soft-eps`, `--sap-code-classes`, and `--sap-init-trunk` / `--sap-freeze-trunk` (load a dense checkpoint, train only the head, frozen tensors pruned from the optimizer).
