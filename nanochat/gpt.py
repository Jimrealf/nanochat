"""
GPT model (rewrite, a lot simpler)
Notable features:
- rotary embeddings (optionally combined with learned absolute positional embeddings)
- QK norm
- untied weights for token embedding and lm_head
- relu^2 activation in MLP
- norm after token embedding
- no learnable params in rmsnorm
- no bias in linear layers
- Group-Query Attention (GQA) support for more efficient inference
- Flash Attention 3 integration
"""

from functools import partial
from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import get_dist_info, print0, COMPUTE_DTYPE
from nanochat.optim import MuonAdamW, DistMuonAdamW

# Our custom Flash Attention module that automatically uses FA3 on Hopper+ and SDPA fallback elsewhere
from nanochat.flash_attention import flash_attn
from nanochat.block_head import DEPTH_MODES

@dataclass
class GPTConfig:
    """Configuration for :class:`GPT` and optional research branches.

    Router defaults are explicit to keep notebook experiments reproducible:
    ``router_context_window=-1`` (full context), ``router_causal=True``,
    ``router_num_heads=4``, ``router_num_queries=8``, ``router_n_layers=2``,
    and ``router_use_vocab_prior=False``.
    """
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6 # number of query heads
    n_kv_head: int = 6 # number of key/value heads (GQA)
    n_embd: int = 768

    # S16-C: Lane-Relative Attention Bias (LRB) and Offset Embedding
    lane_rel_bias: bool = False
    lane_offset_embed: bool = False
    lane_max_offsets: int = 256
    lane_rel_bias_roles: bool = True    # False: the roles-collapsed control (offset-gap buckets only)

    # Research branches
    use_moe: bool = False
    use_perm: bool = False
    moe_num_experts: int = 8
    moe_router_dim: int = 64
    moe_embed_dim: int = 64
    dropout: float = 0.0
    use_remix_linear: bool = False
    remix_context_dim: int = 64
    # If > 0, overrides remix_context_dim with n_embd // remix_context_dim_ratio.
    # Recommended: 8 (gives 96-dim context for 768-dim model, 128-dim for 1024-dim, etc.)
    # Set to 0 to use the fixed remix_context_dim value instead.
    remix_context_dim_ratio: int = 6
    remix_basis_size: int = 0  # 0 = auto (match n_embd)
    # Rank of the low-rank output gate. Smaller r = more stable at long T.
    # r=8 works well from T=64 to T=2048. Increase to 16 only if needed.
    remix_output_gate_rank: int = 16
    remixed_linear_kwargs: dict | None = None
    use_pos_embed: bool = False
    moe_use_abs_pos_embed: bool = False

    # Fix 1A: per-layer context updaters for remix_linear (zero-init deltas applied at each block)
    use_layer_context: bool = True
    # Fix 1B: auto-scale basis_size to max(remix_basis_size, in_features // 4) to prevent rank bottleneck
    scale_basis_size: bool = True
    # Use DualGateLinear instead of RemixedLinear (single dense + dual D-dim gate, no basis compression)
    remix_use_dual_gate: bool = False
    # Explicitly enable RemixedLinearFused for 29C chunk routing (default False: use standard RemixedLinear)
    remix_use_fused: bool = False
    # Phase 26: OutputGatedLinear — single W + low-rank output gate, no W_b/W_m factorization
    p26_output_gated_linear: int = 0
    # Phase 28: FLOPs-efficient template routing experiments
    p28_shared_basis: int = 0               # 28C: share single W_b projection across all attn Q/K/V/O layers per block
    p28_chunk_routing_size: int = 0         # 28D: amortize template routing over N-token chunks (0=per-token, 64/256=chunk)
    p28_global_template_bank: str = 'none'  # 28E/F: 'none'|'ffn'|'all' — cross-layer global template bank
    p28_attn_proj_templates: int = 0        # 28C2: override n_templates for attn c_proj (0=use default from remixed_linear_kwargs)
    p28_attn_qk_templates: int = 0          # 28C3: override n_templates for attn c_q and c_k (0=use default)
    # Phase 29: RemixedLinear optimizer optimizations
    p29_template_block_diag: int = 0        # Block-diagonal Muon for template_bank (0=off, 1=on — uses K blocks)
    p29_template_lr_scale: float = 1.0      # LR multiplier for template_bank Muon group (e.g. 2.0 for 2× LR)
    p29_grad_equalize: int = 0              # gradient equalization across templates (0=off, 1=on; MST-inspired)
    # Phase 34: FFN-shape study. The dense FFN spends its capacity on a 4x width
    # expansion (D -> 4D -> D). If per-token weight mixing supplies that capacity
    # instead, the expansion is redundant and the FFN gets much cheaper. These
    # knobs let you test that directly, and to isolate it from attention.
    p34_ffn_mult: float = 4.0     # FFN hidden width as a multiple of n_embd (4.0 = standard)
    p34_ffn_single: int = 0       # 1 = one RemixedLinear D->D, no hidden layer and no activation
    p34_dense_attn: int = 0       # 1 = plain dense attention, so only the FFN is RemixedLinear
    # Reproduce the pre-fix initialization, where the generic Linear fallback ran
    # after each specialized branch and overwrote it (RemixedLinear's orthogonal
    # basis became xavier; its basis_modulator bias of 2.0 read back as 0.0).
    # Only for re-running checkpoints trained before that was corrected.
    legacy_init_fallback: int = 0
    # Phase 35: template bank diversity regularization. Penalizes high pairwise
    # cosine similarity between templates within each RemixedLinear layer to
    # prevent template collapse. 0 = off. Reasonable range: 0.01–0.1.
    p35_template_diversity_lambda: float = 0.0
    # Phase 36: affine-hull routing for the template bank. 0 = softmax (simplex),
    # 1 = alpha = 1/K + s*(beta - mean(beta)), unconstrained and possibly negative.
    # Targets the two mechanisms behind the measured collapse: the alpha_k
    # prefactor on the softmax gradient, and the fact that on a simplex the only
    # way to grow operator norm is to concentrate. See RemixedLinear._affine_weights.
    p22_route_affine: int = 0
    # Phase 36: the controls for the p35 result.
    p34_ffn_schedule: str = ''      # per-layer FFN width: list, or 'measured'/'shrink'/'last'/'every2'/'every4'(_iso)
    p34_ffn_no_ffn_replacement: str = 'none'  # what to put in layers with schedule=0: 'none' | 'linear' (D->D)
    p34_ffn_last_depth: int = 1     # FFN depth in the last layer: 1=D->4D->D, 2=D->4D->4D->D, 3=D->D->4D->D
    p36_swiglu_ffn: int = 0        # dense baseline with a SwiGLU FFN instead of ReLU^2
    p36_swiglu_mult: float = 8.0 / 3.0  # hidden width multiple, 8/3 = parameter parity
    cond_sites: str = 'both'       # 'both' | 'attn' | 'ffn': where cond layers go
    # Phase 35: ConditionedLinear moves conditioning from weight space to
    # activation space.  Selected with --cclblock-modulation cond.  A degree of
    # freedom costs O(d) here instead of O(d^2), and no operator is ever
    # materialized per token, so every parameter is read once per batch like a
    # dense weight.  See the ConditionedLinear docstring for the DOF/byte
    # accounting these defaults come from.
    cond_rank: int = 256          # R: additive branch rank (0 = no additive branch)
    cond_mult_steps: int = 0      # m: multiplicative composition steps (0 = off)
    cond_gate_source: str = 'router'  # 'router' (own proj of x) | 'tied' (reuse V^T x) | 'ctx' (context stream)
    cond_coeff_act: str = 'centered'  # 'centered' (1+tanh) | 'linear' (1+z) | 'sigmoid' (2σ)
    cond_router_rank: int = 0     # >0 factorizes the router as d_sig -> rr -> R (0 = full)
    cond_live_init: float = 0.0   # >0: draw every conditioning factor nonzero and subtract the
                                  # branch's mean contribution from W0, so the composite starts at
                                  # the dense init with no dead branch. The value scales the branch.
    cond_router_act: str = 'none' # 'none' | 'gelu' | 'relu2': nonlinearity in the router's
                                  # hidden layer. Needs cond_router_rank > 0. Same parameters
                                  # and same FLOPs as the linear rank-rr router it replaces.
    cond_chunk_size: int = 0      # >0 routes once per chunk from its first token (0 = per token)
    cond_mult_scale: float = -1.0 # bound on |c| for the composition branch (-1 = 1/m, the stable default)
    cond_mult_impl: str = 'wy'    # 'wy' (compact, 2 GEMMs) | 'loop' (sequential reference)
    cond_attn_projs: str = 'qkvo' # 'qkvo' (default, all) | 'qko' (skip c_v) | 'qo' | 'o'
    cond_layer_frac: float = 1.0  # fraction of layers (from start) that get ConditionedLinear
    # Fix 1D: PermutationMoE expert mode — 'full' (original D×D), 'low_rank', or 'factored'
    perm_expert_mode: str = 'low_rank'
    # Fix 1D: rank for low_rank mode (rank = max(8, base_embed_dim // perm_rank_ratio))
    #          or block_size for factored mode (num_blocks = base_embed_dim // perm_rank)
    perm_rank: int = 16

    # Shared context-aware router defaults used by embedding/context branches
    router_context_window: int = -1  # -1 = full context. For sequence_len > 512, set this to
                                      # 256-512 to prevent the GlobalContextManager's attention
                                      # from becoming a bottleneck (O(T²)) and diluting the
                                      # context signal. The per-layer residual-stream updaters
                                      # now carry the long-range signal instead.
    router_causal: bool = True
    router_num_heads: int = 4
    router_num_queries: int = 16
    router_n_layers: int = 2
    router_use_vocab_prior: bool = False

    # Sliding window attention pattern string, tiled across layers. Final layer always L.
    # Characters: L=long (full context), S=short (half context)
    # Examples: "L"=all full context, "SL"=alternating, "SSL"=two short then one long
    window_pattern: str = "SSSSL"

    # CCL block modulation strategy (only used when use_remix_linear=True):
    #   'weight'        — RemixedLinear gates in activation space.
    #   'normalization' — CCLBlock with AdaRMSNorm conditioning.
    #   'householder'   — operator-space basis transport via context-conditioned
    #                     Householder reflection(s).
    #   'spectral'      — operator-space spectral scaffold (near-isometric scaling).
    #   'ocd'           — orthogonal-complement-style dynamic delta with overlap penalty.
    #   'decoupled'     — orthogonal static/dynamic input channel routing.
    #   'tucker'        — Tucker-routed operator simplex.
    #   'svs'           — singular-value steering over fixed factors.
    #   'vq'            — vector-quantized regime-conditioned operator deltas.
    #   'dcu'           — deferred-context-unlock style tail-space steering.
    #   'fsi'           — Frozen Subspace Indexing: frozen orthogonal rotations + frozen routing.
    #   'aesp'          — Attention-Entropy Stratified Projection: entropy-routed per-stratum deltas.
    #   'ckr'           — Causal Kernel Reparameterization: position-dependent branch mixing.
    cclblock_modulation: str = 'weight'
    cclblock_orth_lambda: float = 0.0
    # The context stream logic:
    #   'local'      — (default) derived directly from norm(x) inside the block. No cross-block threading.
    #   'ema'        — fixed ema factor, with .detach() to prevent gradient explosion.
    #   'selective'  — GRU-style input-dependent gating, no detach.
    #   'multiscale' — 3 parallel selective temporal channels (Fast, Med, Slow).
    #   'ssm'        — linear state-space style context highway.
    cclblock_context_stream: str = 'local'
    cclblock_ema_factor: float = 0.99
    # Design C: Cross-layer stale context lag (0 = disabled, k>=1 = use context from k blocks ago).
    # When k>0, block i is conditioned on the context emitted by block i-k in the same forward pass.
    # This breaks the circular dependency because i-k is a genuinely different layer's computation.
    # The stale context is detached — only the within-block gradient path is active.
    cclblock_stale_ctx_lag: int = 0
    # --- Novel Ablation Designs ---
    # Design 1: 'shifted' residual uses norm(x_entry) from the START of the block
    #            (= previous layer output). Decouples basis from gate signal source.
    #            Use via --cclblock-context-stream "shifted"
    # Design 3: Sparse top-k basis gate. 0=soft sigmoid, N=activate top-N basis functions.
    #            Uses straight-through estimator for gradients.
    cclblock_sparse_gate_k: int = 0
    # Design 6: Basis gate temperature. <1.0=sharper (more selective), >1.0=softer (more uniform).
    cclblock_gate_temperature: float = 1.0
    # Design 4: Context prototype bank. 0=disabled, N=use N learned prototype vectors.
    #            Forces model to learn a vocab of text-type contexts via soft lookup.
    cclblock_context_bank_size: int = 0
    # Design 7: Per-module context. True=separate ctx projections for attn vs ffn.
    #            Attn gets pre-attn ctx, FFN gets post-attn ctx.
    cclblock_per_head_ctx: bool = False
    # Design 2: Context source for the FFN gate. 'norm_x' uses the residual stream (default).
    #            'attn_heads' uses mean query vectors (q after RoPE+QKnorm, detached).
    #            This gives the gate a direct signal about "what each position is searching for"
    #            without any recurrence or cross-block dependency.
    cclblock_context_source: str = 'norm_x'  # 'norm_x' | 'attn_heads' | 'attn_geometry'
    # Phase 8: Boundary-Gated / Chunk Context / Auxiliary Objective
    # Design 8 (boundary gate): learned soft gate that fires only at segment boundaries.
    #   Use --cclblock-context-stream "boundary"
    # Design 9 (hard chunk): context = pooled summary of previous K-token chunk.
    #   Strictly causal, zero recurrence. Use --cclblock-chunk-size N (e.g. 64)
    cclblock_chunk_size: int = 0
    # Design 10 (aux objective): context branch predicts an auxiliary target.
    #   'boundary': predicts whether current token is a structural boundary.
    #   'entropy':  predicts per-token cross-entropy (difficulty signal).
    #   Forces context to encode non-trivial info, prevents identity collapse.
    cclblock_aux_objective: str = 'none'  # 'none' | 'boundary' | 'entropy'
    cclblock_aux_lambda: float = 0.1      # weight of auxiliary loss
    cclblock_boundary_token_id: int = 198 # token ID for boundary detection (default=\n)
    # Phase 9: Residual Adaptive Linear + New Context Streams + FiLM Gate
    # Proposal A: ResidualAdaptiveLinear — dense base + context-conditioned low-rank delta.
    #   y = base(x) + scale * (x @ U * ctx_coeffs(ctx)) @ V
    #   V=0 at init → zero delta → exact dense baseline at step 0. No regression guarantee.
    use_ral: bool = False    # use ResidualAdaptiveLinear instead of RemixedLinear
    ral_rank: int = 32       # rank r for the context-conditioned additive delta
    # Proposal C: FiLM basis conditioning — affine (scale+shift) instead of sigmoid gate.
    #   gate = 1 + tanh(scale * 0.1); h_gated = h * gate + shift
    #   Zero-init → scale=0→gate=1, shift=0 → exact identity at init.
    cclblock_film_gate: bool = False
    # Dual-V shadow routing: append shadow channels to attention V and route the
    # shadow output directly into FFN context conditioning.
    cclblock_attn_shadow_dim: int = 0
    # Paradigm 1: Decoupled Feature-Space Routing
    cclblock_dynamic_ratio: float = 0.25
    cclblock_gate_rank: int = 8
    # Paradigm 2: Evidence Accumulation SSM context stream
    cclblock_num_regimes: int = 8
    cclblock_regime_temperature: float = 1.0
    # Additional operator-family ablations
    cclblock_poly_order: int = 2
    cclblock_lie_generators: int = 4
    cclblock_grassmann_bank_size: int = 4
    cclblock_tucker_rank: int = 32
    cclblock_tucker_modes: int = 8
    cclblock_svs_rank: int = 64
    cclblock_svs_eps: float = 0.1
    cclblock_vq_codes: int = 8
    cclblock_vq_temperature: float = 1.0
    cclblock_dcu_warmup_steps: int = 0
    # Phase 12: Novel zero-friction context-conditioned linear layers
    # FSI: Frozen Subspace Indexing — K frozen orthogonal rotations + frozen routing
    cclblock_fsi_rotations: int = 8       # number of frozen orthogonal rotation matrices
    cclblock_fsi_selector_dim: int = 64   # dimension of frozen routing projection
    # AESP: Attention-Entropy Stratified Projection
    cclblock_aesp_strata: int = 4         # number of entropy strata (stratum 0 = pure dense)
    cclblock_aesp_delta_rank: int = 4     # rank of per-stratum low-rank deltas
    # CKR: Causal Kernel Reparameterization
    cclblock_ckr_branches: int = 4        # number of parallel dense branches
    cclblock_ckr_kernel_size: int = 64    # causal conv1d kernel size
    # Phase 13: CKR enhancements
    cclblock_ckr_pos_channels: int = 1    # 13a: multi-channel position signal (1=original, 3=fast/med/slow)
    cclblock_ckr_dual_optim: int = 0      # 13b: route CKR gate params to dedicated AdamW group
    cclblock_ckr_content_bias: float = 0.0  # 13c: frozen content hash bias scale (0=pure position)
    # Phase 14: Gradient-isolated content conditioning
    cclblock_giad_rank: int = 32              # 14a: GIAD low-rank bottleneck dimension
    cclblock_psg_kernel_size: int = 64        # 14b: PSG causal conv kernel size
    cclblock_ss_dynamic_ratio: float = 0.25   # 14c: SplitStream dynamic channel fraction
    cclblock_ss_branches: int = 2             # 14c: SplitStream CKR branches on dynamic path
    cclblock_ss_kernel_size: int = 64         # 14c: SplitStream causal conv kernel size
    # Phase 15: LoKR — Low-rank Kernel Reparameterization
    cclblock_lokr_branches: int = 8           # 15: LoKR number of low-rank perturbation branches
    cclblock_lokr_rank: int = 16              # 15: LoKR rank of each perturbation
    # Phase 16: CKR-Anneal temperature annealing
    cclblock_ckr_temp_start: float = 2.0      # 16A: initial softmax temperature (soft routing)
    cclblock_ckr_temp_end: float = 0.3        # 16A: final softmax temperature (sharp routing)
    # Phase 16: Causal Output Mixer
    cclblock_com_kernel_size: int = 32        # 16C: COM causal depthwise conv kernel size
    # Phase 17: CKR enhancements
    cclblock_ckr_ortho_init: int = 0          # 17D: orthogonal branch initialization (0/1)
    cclblock_ckr_branch_dropout: float = 0.0  # 17E: branch dropout probability (0=off)
    cclblock_ckr_diversity_lambda: float = 0.0 # 17H: auxiliary branch diversity loss weight
    cclblock_ckr_layer_selective: int = 0     # 17F: 0=all layers CKR, 1=odd layers only
    # Phase 17: New architectures kernel sizes
    cclblock_pgr_kernel_size: int = 64        # 17C: PGR causal conv kernel size
    cclblock_cil_kernel_size: int = 64        # 17I: CIL causal conv kernel size
    cclblock_prb_kernel_size: int = 64        # 17J: PRB causal conv kernel size
    # Phase 18: Beyond CKR — orthogonal improvements
    p18_layer_drop: float = 0.0               # 18E: stochastic depth drop probability
    p18_dynamic_activation: int = 0           # 18I: learned activation mix (0/1)
    p18_mixture_norm: int = 0                 # 18H: RMSNorm+LayerNorm mixture (0/1)
    p18_causal_attn_bias: int = 0             # 18J: learned causal attention score bias (0/1)
    p18_aux_sim_lambda: float = 0.0           # 18G: layer similarity penalty weight
    p18_gradient_penalty: float = 0.0         # 18B: gradient penalty weight
    p18_per_channel_scale: int = 0            # 18F: learnable per-channel output scale (0/1)
    # Phase 19: Zero-overhead indirect modulation
    p19_residual_gate: int = 0                # 19A: per-layer learned scalar on block output (0/1)
    p19_head_importance: int = 0              # 19B: per-head learned scalar on attn output (0/1)
    p19_residual_mix_groups: int = 0          # 19C: grouped 1x1 conv between blocks (0=off, N=group_size)
    p19_attn_logit_bias: int = 0              # 19D: per-head learned QK temperature (0/1)
    p19_residual_decay: int = 0               # 19E: learned depth-dependent x0 decay (0/1)
    p19_grad_equilibrium: float = 0.0         # 19F: gradient equilibrium regularization lambda (0=off)
    p19_spectral_reparam: int = 0             # 19G: spectral reparameterization (0=off, 1=c_proj, 2=c_fc+c_proj)
    p19_weight_anticollapse: float = 0.0      # 19H: weight anti-collapse penalty lambda (0=off)
    p19_ve_bias: int = 0                      # 19I: add learnable bias to VE gate (0/1)
    p19_weight_noise: float = 0.0             # 19J: training-time weight perturbation epsilon (0=off)
    # Phase 20: Context-conditioned dynamic weight computation
    p20_hrcs_scale: int = 0                   # 20A: Hash-routed column selection (0=off, scale=D_stored/D_active)
    p20_lswr_scale: int = 0                   # 20B: LSH weight routing (0=off, scale factor)
    p20_lswr_planes: int = 8                  # 20B: number of LSH hash planes
    p20_lrcfb_branches: int = 0               # 20C: Content-routed branches (0=off, K=num branches)
    p20_lrcfb_narrow: int = 0                 # 20C: narrow branches for param parity (0=full-size, 1=H//K per branch)
    p20_lrcfb_learned: int = 0                # 20C: learned routing (0=frozen random, 1=learnable projection)
    p20_lrcfb_topk: int = 0                   # 20C: top-k sparse routing (0=soft/all, K=top-K only)
    p20_dgcr_branches: int = 0                # 20D: Detached-gradient content-routed branches (0=off, K=num branches)
    p20_dgcr_aux_weight: float = 0.01         # 20D: weight for auxiliary routing loss
    p20_mone_experts: int = 0                 # 20F: Mixture of Narrow Experts (0=off, K=num experts)
    p20_mone_topk: int = 0                    # 20F: top-k expert routing (0=compute all, K=top-k sparse)
    p20_mone_narrow: int = 1                  # 20F: narrow experts (1=hidden=4D/K param parity, 0=full 4D each)
    p20_mone_frozen: int = 0                  # 20F: frozen routing (0=learned Linear router, 1=frozen random proj)
    p20_ncea_branches: int = 0                # 20H: Noise-contrastive expert assignment (0=off, K=branches)
    p20_ncea_eps: float = 0.1                 # 20H: perturbation magnitude for NCEA
    p20_adwi: int = 0                         # 20I: Attention-derived weight interpolation (0=off, 1=on)
    # Phase 20 (require pre-trained checkpoint — Phase 2 training)
    p20_pwu_branches: int = 0                 # 20E: Progressive weight unfreezing (0=off, K=branches forked from pretrained)
    p20_pwu_phase: int = 1                    # 20E: training phase (1=normal pretrain, 2=fork+route, 3=joint finetune)
    p20_fsvd_gate: int = 0                    # 20G: Frozen-SVD σ gating (0=off, 1=on, loads SVD from pretrained)
    p20_wbfc_clusters: int = 0                # 20J: Weight bank frozen clustering (0=off, K=clusters)
    p20_wbfc_active: int = 0                  # 20J: number of active clusters per token (M < K)
    # Phase 21: Pervasive Expert Routing (PER) — MoE-ify every Linear layer
    p21_per_experts: int = 0                  # 21: number of experts per MoELinear (0=off, K=experts)
    p21_per_topk: int = 0                     # 21: top-k routing (0=soft/all, K=top-K)
    p21_per_learned: int = 0                  # 21: learned routing (0=frozen, 1=learnable)
    p21_per_attn: int = 0                     # 21: also replace attention Q/K/V/O (0=MLP only, 1=all)
    # Phase 22: MoE attention projections
    p22_attn_moe_route: str = 'none'          # 22: MoE routing for attn Q/K/V/Proj ('none'|'sequence'|'token')
    # Phase 23: Tiny-Experts RemixedLinear
    # Each expert has intermediate dim = remix_basis_size // p23_topk  (compute parity with dense).
    # K_total experts in the bank, topk active per token (FFN) or per sequence (attention).
    p23_tiny_expert: int = 0                  # 23: 0=off, 1=Tiny Experts mode for RemixedLinear templates
    p23_n_experts: int = 64                   # 23: total experts in the bank
    p23_topk: int = 16                        # 23: active experts per forward pass (expert_dim = basis_size // topk)
    p23_learned_route: int = 0                # 23: 0=frozen random routing projection, 1=learned
    # Phase 23: Standard MoE baseline (full-size experts, learned router)
    p23_std_moe_experts: int = 0              # 23: 0=off, K=standard MoE with K full-size experts
    p23_std_moe_topk: int = 1                 # 23: top-k active experts (1=sparse, 0=dense all)
    p23_std_moe_aux_weight: float = 0.01      # 23: load-balance auxiliary loss weight
    p23_lokr: int = 0                         # 23: enable LoKR mode in RemixedLinear
    p23_lokr_rank: int = 4                    # 23: low-rank bottleneck for each LoKR expert
    p23_use_shared_block_router: int = 0      # 23: block-level single pass router for all RemixedLinear inner experts
    p23_linear_moe_experts: int = 0           # 23: enable weight-space LinearMoE with K experts (0=off)
    p23_linear_moe_topk: int = 0              # 23: top-k selected experts in LinearMoE (0=soft all-expert blend)
    p23_quantile_route: int = 0               # 23: use EMA quantile-balanced routing without aux loss
    remix_shared_context_gates: int = 0       # 23: batch all 6 per-RL context gate computations into 3 block-level matmuls
    # Phase 24: Linear-layer variant experiments
    p24_use_sliced_weight: int = 0
    p24_sliced_weight_reduction_scale: int = 8
    p24_sliced_weight_min_select: int = 128
    p24_sliced_weight_scope: str = "per_token"         # per_token|per_block|global
    p24_sliced_weight_balance_coeff: float = 0.01
    p24_quantile_route: int = 0
    p24_use_folded_mod: int = 0
    p24_folded_mod_reduction_scale: int = 8
    p24_folded_mod_min_dim: int = 128                      # floor on folded_dim (0 = no floor)
    p24_folded_mod_scope: str = "per_layer"            # per_layer|per_block|global
    p24_folded_mod_gate_act: str = "sigmoid"
    p24_use_sequence_gated_linear: int = 0
    p24_sequence_gated_scope: str = "per_layer"        # per_layer|per_block|global
    p24_sequence_gated_act: str = "sigmoid"
    # Phase 30: LayerNorm ablation
    remix_disable_ln_basis: int = 0           # 30B: 1=replace intermediate LayerNorm in RemixedLinear with Identity
    dense_intermediate_ln: int = 0            # 30A: 1=add intermediate LayerNorm to dense linear projections
    # ── MoL: Mixture of Layers (Ternovtsii & Bilak 2026, arXiv:2605.09516) ──
    # Reimplemented as a baseline for MST; see nanochat/mol.py. Their notation is
    # S+KofN: mol_n_shared + mol_topk of mol_n_blocks.
    use_mol: bool = False                      # master switch for MoL mode
    mol_n_blocks: int = 5                      # N = thin blocks per split stage
    mol_n_shared: int = 1                      # S = always-active blocks (full attention)
    mol_topk: int = 3                          # k = routed blocks selected per token
    mol_thin_dim: int = 256                    # d_thin
    mol_head_dim: int = 64                     # pinned across all block widths (their §2.2)
    mol_ffn_mult: float = 4.0                  # d_ff,thin = mult * d_thin
    mol_router_aux: float = 0.05               # CV² load-balance weight (their α)
    mol_routed_attn: str = 'softmax'           # 'softmax'; 'deltanet' is phase 2
    mol_dispatch: int = 1                      # 1=gather/scatter (default), 0=masked reference
    mol_capacity_factor: float = 1.0           # per-block capacity when dispatching
    mol_block_lr_scale: float = 1.0            # per-thin-block LR multiplier (their recipe has none)
    mol_per_block_ve: int = 0                  # G3-equivalent: each thin block reads its own VE slice
    # ── Fully binary transformer (fully-binary-transformer-plan.md) ──
    use_binary: bool = False                   # master switch: swap Linear/Embedding for binary
    binary_acts: bool = True                   # False = W1A16 rung, which changes no operation
    binary_weight_scale: str = "row"           # row | none | threshold  (see nanochat/binary.py)
    binary_act_scale: str = "token"            # token | none
    binary_clip: float = 1.0                   # STE clip window; latent weights init inside it
    binary_linear: bool = True                 # binarise nn.Linear
    binary_embeddings: bool = True             # binarise nn.Embedding (wte + value_embeds)
    binary_skip: str = ""                      # comma-separated name substrings left in fp
    binary_native: bool = False                # replace Block with BinaryBlock: Hamming
                                               # attention, binary KV-memory FFN, bundled
                                               # residual, no normalisation (sections 4.2/4.3)
    binary_tau: float = 1.0                    # retrieval temperature; anneal toward hard
    binary_hard: bool = False                  # hard threshold retrieval (the popcount path)
    binary_resid_width: int = 0                # 0 = derive from depth; 1 = majority bundling (destroys the stream); >1 = accumulator
    binary_attn_chunk: int = 256               # query-block size; 0 = unchunked, which
                                               # needs 84GB for one layer at d=3584 T=2048
    # ── MST: Modular Sub-Transformer Architecture ──
    use_mst: bool = False                      # master switch for MST mode
    mst_n_subs: int = 8                        # N = number of sub-transformers per layer
    mst_sub_dim: int = 64                      # d = dimension per sub-transformer (D/N)
    mst_head_dim: int = 0                      # attention head_dim (0=auto: d//n_head; else expands QKV to n_head*head_dim)
    # Axis 1 — First-Layer Input
    mst_input_mode: str = 'fixed_slice'        # 'fixed_slice'|'learned_proj'|'rotated_slice'|'per_sub_embed'|'stem'
    mst_rotated_slice_learned: bool = False    # I-C: learn the rotation matrix vs random orthogonal
    # Axis 2 — Routing
    mst_routing_mode: str = 'soft_weighted'    # 'soft_weighted'|'topk_hard'|'sequence_path'
    mst_routing_topk: int = 4                  # R-B: k for top-k hard routing
    mst_routing_aux_weight: float = 0.01       # load balancing loss coefficient
    mst_diversity_weight: float = 0.0          # cosine diversity penalty (0=off; e.g. 0.01 to encourage sub specialization)
    # Axis 3 — FFN Internal Transition
    mst_ffn_mode: str = 'standard'             # 'standard'|'no_downproj'|'shared_dense'|'grouped_up_shared_down'|'shared_swiglu'
    mst_ffn_every: int = 1                     # apply FFN every N layers (1=every, 2=even-only for sandwiched FFN)
    # Axis 4 — Layer-to-Layer Transition
    mst_transition_mode: str = 'parallel'      # 'parallel'|'aggregate_distribute'|'cross_attend'|'concat_proj'|'free_for_all'
    # Axis 5 — Final Layer → Vocabulary
    mst_final_mode: str = 'aggregate_proj'     # 'aggregate_proj'|'weighted_logits'|'concat_proj'
    mst_final_topk: int = -1                    # topk for final head (-1=same as routing_topk, 0=all subs)
    # Stage 3+ features
    mst_ffn_shared_up: int = 0                  # share FFN up-projection across all subs (0=off, 1=on)
    mst_ffn_inner_dim: int = 0                  # inner dim for (shared) FFN (0=4*d default)
    mst_sub_dropout: float = 0.0                # drop entire sub outputs during training (0=off)
    mst_transition_every: int = 1               # apply transition every N layers (1=every, 2=even-only)
    mst_ffa_temperature: float = 1.0            # temperature for FFA softmax routing (>1=softer, <1=sharper)
    mst_global_residual: int = 0                # maintain D-dim global residual stream (0=off, 1=on)
    mst_hybrid_dense: int = 0                   # alternate dense (D) and MST (N×d) layers (0=off, 1=on)
    mst_cross_sub_kv: int = 0                   # share K,V projections across all subs (0=off, 1=on)
    # Stage 5 features
    mst_sub_aux_weight: float = 0.0             # H3: per-sub auxiliary prediction loss weight (0=off)
    mst_progressive_merge: int = 0              # N1: pyramid sub-merging (0=off, 1=on)
    mst_multi_scale_windows: int = 0            # W1: per-sub window sizes (0=off, 1=on)
    # Stage 6 features
    mst_delta_residual: int = 0                 # DR1: delta residual mode — subs produce corrections to full-D stream
    mst_sub_layers: int = 1                     # SL1: layers per sub-transformer (1=current, 2/4/8=deeper subs)
    # Stage 7: Scaling improvement experiments (P07)
    mst_grad_equalize: int = 0                  # 1A: per-sub gradient equalization (0=off, 1=on)
    mst_block_diagonal_muon: int = 0            # 1B: block-diagonal Newton-Schulz in Muon (0=off, 1=on)
    mst_transition_width_mult: float = 1.0      # 1C: transition bottleneck width multiplier (1.0=d, 4.0=D for N=4)
    mst_sub_lr_scale: float = 1.0               # per-sub Muon LR multiplier (√N ≈ 2.0 for N=4 to fix μP scaling)
    mst_shared_expert: int = 0                  # 2A: DeepSeek-style shared expert — sub 0 always fully weighted (0=off, 1=on)
    mst_router_entropy_weight: float = 0.0      # 2C: router entropy regularization weight (0=off, e.g. 0.1)
    mst_shared_kv_attn: int = 0                 # 3A: share K,V projections in main attention across subs (0=off, 1=on)
    mst_contrastive_diversity_weight: float = 0.0  # 3B: contrastive diversity loss on FFN activations (0=off)
    # Stage 8: Transition expressivity
    mst_transition_nonlinear: int = 0               # add SiLU activation at AggDist bottleneck (0=off, 1=on)
    mst_transition_gated: int = 0                   # input-dependent routing via concat→gate (0=off, 1=on)
    mst_transition_mlp: int = 0                     # replace AggDist with concat→MLP→split (0=off, 1=on)
    # Stage 9: Cross-sub expressivity
    mst_cross_sub_gate: int = 0                     # A: cross-sub FFN gating rank (0=off, e.g. 32)
    mst_hyper_connect: int = 0                      # B: hyper-connected sub residuals (0=off, 1=EMA lookback)
    mst_cross_kv_inject: int = 0                    # C: cross-sub KV injection attention (0=off, 1=on)
    # Stage 10: Structural transition improvements
    mst_slice_transition: int = 0                   # SliceMoE: per-slice routing (0=off, S=num slices e.g. 4)
    mst_lookback_layers: int = 0                    # DenseFormer lookback depth (0=off, K=num past layers e.g. 2)
    mst_bilinear_transition: int = 0                # Bilinear aggregate: 2nd-order cross-sub interaction (0=off, 1=on)
    # Stage 11: Attention bottleneck + structural improvements
    mst_cross_sub_qmod: int = 0                     # Low-rank cross-sub query modulation rank (0=off, e.g. 16)
    mst_feature_cycle: int = 0                      # Sub-feature cycling at transition (0=off, 1=on)
    mst_mean_transition: int = 0                    # Parameter-free mean-add transition replacing AggDist (0=off, 1=on)
    # Stage 12: dense-parity fixes (each FLOP-neutral; see LEARNINGS.md 2026-08-08)
    mst_sub_head_dim: int = 0                       # G1: per-stream attention head_dim, heads derived as d//head_dim
                                                    #     so qkv_dim stays == d (0=off: n_head heads of d//n_head).
                                                    #     Dense runs head_dim=128; MST's implicit 32 costs quality
                                                    #     and flash-attn efficiency at identical params and FLOPs.
    mst_final_norm: int = 0                         # G2: RMSNorm the final projection before lm_head, as dense does
    mst_per_stream_ve: int = 0                      # G3: give each stream its own slice of an (N*d)-wide value
    mst_ve_map: int = 0                         # G3-cheap: shared d-wide VE table + per-stream d x d map
    mst_ve_map_rank: int = 0                    # 0 = full d x d map; r>0 = identity + rank-r correction
                                                    #     embedding instead of broadcasting one d-wide table to all
    # Stage 13: overhead cuts. Unlike Stage 12 these DO reduce FLOPs, which is the
    # point: MST is at parity with dense per matrix parameter, and its whole
    # FLOPs-axis deficit is D-proportional overhead (~7% output head, ~8% attention).
    mst_lm_head_dim: int = 0                        # O1: bottleneck width between the final projection and
                                                    #     lm_head (0 = n_embd, i.e. off). The head then costs
                                                    #     D*Dh + V*Dh instead of D*D + V*D.
    mst_compose_windows: int = 0                    # O2: intersect per-sub windows with the layer window pattern
                                                    #     instead of letting the per-sub windows replace it, so the
                                                    #     widest stream is not full-context at every single layer
    # Stage 14: free cross-stream mixing. MST is MHA with a block-diagonal output
    # projection: composing block-diagonal layers under a FIXED partition stays
    # block-diagonal forever, so all cross-channel flow is squeezed through the
    # rank-d coupling. Permuting the partition makes the composition mix at zero
    # parameter and zero FLOP cost (ShuffleNet channel shuffle; Swin shifted windows;
    # Monarch's block-diagonal-permutation-block-diagonal factorization).
    mst_channel_mix: str = 'none'                   # 'none' | 'roll' (shift by d//2, Swin-style, streams keep
                                                    #   half their channels) | 'shuffle' (ShuffleNet transpose,
                                                    #   every stream draws d/N channels from every other)
    mst_channel_mix_site: str = 'layer'             # 'layer' (alternate the offset on odd layers)
                                                    # | 'ffn' (permute between attention and FFN, which is where
                                                    #   ShuffleNet puts its shuffle) | 'both'
    # Stage 15: coupling optimization + cross-stream mixing in attention.
    mst_distribute_block_muon: int = 0              # F1: give distribute_w the same block-diagonal Newton-Schulz
                                                    #     and sub-LR that the per-stream weights get. It is
                                                    #     (N*d, d), structurally identical to c_proj_w, but was
                                                    #     omitted, so Muon orthogonalizes across all N coupling
                                                    #     blocks jointly (the exact cross-contamination 1B fixes).
    mst_trans_spectral_lr: int = 0                  # F2: agg_up_w (D,d) and agg_down_w (d,D) get LR scaled by
                                                    #     sqrt(N) and 1/sqrt(N) per the spectral condition,
                                                    #     instead of sharing one matrix_lr.
    mst_talking_heads: int = 0                      # F4: learned (N*n_head)^2 mixing along the head axis before
                                                    #     c_proj. MHA is already block-diagonal per head and works
                                                    #     because W_O mixes them; MST blocks W_O too. ~1K params.
    mst_wo_mode: str = 'block'                      # F5: 'block' (per-stream c_proj) | 'dense' (full D x D over
                                                    #     the concatenated attention outputs, i.e. MHA's W_O)
    # Stage 16: conditional stream execution. Everything above buys quality by adding
    # parameters, which is capped by the parameter scaling law -- the same law dense
    # obeys, so it cannot beat dense. Sparsity is a different axis: make the model
    # sparse-active so active FLOPs fall below total. MST is the natural host because
    # its streams are already independent modules with their own attention AND FFN,
    # i.e. mixture-of-experts where the expert is a whole transformer block.
    mst_stream_topk: int = 0                        # k of N streams active per token (0 = dense, today's model).
    mst_stream_shared: int = 0                 # S always-active streams; top-k runs over the remaining N-S
                                                    #   FFN gating at N=4: k=2 is ~19% of total FLOPs, k=1 ~28%.
    mst_stream_router_aux: float = 0.01             # Switch-style load-balancing weight. A prior attempt
                                                    #   (free_for_all + topk1) collapsed to uniform routing on
                                                    #   replicate seeds, so this is not optional decoration.
    mst_stream_router_noise: float = 0.0            # noisy top-k exploration (Shazeer et al.), training only.
                                                    #   Without it an unselected stream gets no FFN gradient, so
                                                    #   it stays at init, so it stays unselected: the death
                                                    #   spiral that collapsed layers 1-3 to a static pair.
    mst_stream_dispatch: int = 0                    # Phase B: gather/scatter so the FFN genuinely runs on
                                                    #   K < T tokens per stream, instead of masking. Numerically
                                                    #   identical to masking when nothing overflows.
    mst_stream_capacity_factor: float = 1.0         # per-stream capacity K = ceil(T*k/N*factor). Above 1.0
                                                    #   buys headroom against routing imbalance at the cost of
                                                    #   doing more work than the ideal.
    mst_stream_gate_attn: int = 0                   # also gate attention QKV, not just the FFN. Bigger saving
                                                    #   (~26% at k=2) but a skipped token stops being a key/value
                                                    #   for that stream, which changes attention semantics.
    # Stage 17: block-diagonal Shampoo. Preconditioning a dense D x D weight costs O(D^3);
    # MST's are N blocks of d x d, so exact block preconditioning is D^3/N^2, 16x cheaper
    # at N=4. K-FAC and Shampoo both approximate block-diagonality; MST makes it exact by
    # construction, so at equal optimizer cost it affords a stronger preconditioner than
    # dense can. This is a training-efficiency claim, not a FLOPs/token one.
    # Stage 18: Monarch-structured FFN. The FFN is already two thirds of a Monarch
    # factorization -- fc_w (per-stream d->4d) then fc_proj_w (per-stream 4d->d) is
    # block-diagonal . permutation . block-diagonal with the permutation set to IDENTITY.
    # Permuting the N*4d hidden axis between them makes it a genuine Monarch matrix, so
    # each stream's down-projection reads hidden units produced by other streams' up-
    # projections. Zero parameters, negligible FLOPs.
    #
    # NOTE this cannot be combined with conditional stream execution's dispatch: Monarch
    # needs every stream's up-projection to exist, and sparsity's whole saving is not
    # computing them. They are alternatives, not a stack.
    mst_ffn_monarch: str = 'none'                   # 'none' | 'shuffle' (ShuffleNet/Monarch transpose,
                                                    #   every stream draws 4d/N units from every other)
                                                    #   | 'roll' (shift by inner//2, neighbour trade only)
    mst_shampoo: int = 0                            # route the stacked per-stream weights to kind='shampoo'
    mst_precond_every: int = 10                     # steps between inverse-fourth-root refreshes
    mst_shampoo_beta: float = 0.95                  # EMA decay for the L/R Kronecker factors
    # ── EET: Early Exit Transformer ──
    use_eet: bool = False                          # master switch for EET mode
    eet_frozen_kv: bool = True                     # True=Option B (frozen KV injection), False=Option A (masked attention)
    eet_reenter_final: bool = False                # True = force exited tokens to re-enter and be processed by the final layer
    eet_router_type: str = 'mlp2'                  # 'linear', 'mlp1', 'mlp2' — exit router architecture
    eet_router_hidden: int = 0                     # router MLP hidden dim (0 = n_embd // 4)
    eet_freq_prior_alpha: float = 0.0              # weight of frequency-based exit bias (0=disabled)
    eet_pos_prior_beta: float = 0.0                # weight of POS-category exit bias (0=disabled)
    eet_domain_prior: bool = False                 # enable domain-conditioned routing
    # Three-phase training schedule
    eet_warmup_frac: float = 0.02                  # Phase 1: dense warmup fraction of total steps
    eet_explore_frac: float = 0.15                 # Phase 2: exploration phase fraction
    eet_reconstruct_lambda: float = 1.0            # Phase 2: reconstruction loss weight (λ_r)
    eet_efficiency_lambda_start: float = 0.01      # Phase 3: initial efficiency loss weight (λ_e)
    eet_efficiency_lambda_end: float = 0.1         # Phase 3: final efficiency loss weight
    eet_translator_rank: int = 0                   # TunedLens translator rank (0 = full rank d→d)
    eet_max_frozen_kv_frac: float = 0.75           # safety cap: max fraction of tokens that can exit
    eet_exit_threshold: float = 0.5                # sigmoid threshold for exit decision
    eet_min_exit_layer: int = 1                    # earliest layer a token can exit at
    # Loss variant selection
    eet_loss_variant: str = 'reconstruct'           # 'reconstruct' | 'entropy_surprise' | 'adversarial' | 'quality' | 'ce_guided'
    eet_topk_vocab: int = 512                       # vocab approximation for entropy computation
    eet_entropy_lambda: float = 0.3                 # Variant A: entropy loss weight
    eet_surprise_lambda: float = 0.1                # Variant A: surprise loss weight
    eet_adv_lambda: float = 1.0                     # Variant B: adversarial gap weight
    eet_adv_entropy_lambda: float = 0.2             # Variant B: entropy stabilizer weight
    eet_quality_lambda: float = 1.0                 # REINFORCE quality loss weight
    eet_quality_entropy_bonus: float = 0.1           # entropy bonus to prevent exit distribution collapse
    # New paradigms
    eet_gumbel_temp_start: float = 0.0             # > 0 means use Gumbel-Softmax
    eet_gumbel_temp_end: float = 0.0
    eet_gumbel_hard: int = 1                       # 1=straight-through estimator, 0=soft blending
    eet_commitment_beta: float = 0.1               # commitment loss scale (Direction 3)
    eet_global_router: bool = False                # upfront single global exit router
    eet_freq_efficiency_alpha: float = 0.0         # per-token frequency-scaled efficiency (0=uniform, >0=frequent tokens penalized more)
    eet_diversity_lambda: float = 0.0              # exit diversity pressure: penalizes uniform exit depth across tokens
    eet_ce_guided_lambda: float = 1.0              # CE-guided routing loss weight (loss_variant='ce_guided')
    eet_router_lr_mult: float = 5.0                # Router LR multiplier (relative to gate_lr). Routers need higher LR to break symmetry.
    eet_model_lr_mult: float = 1.0                 # Backbone LR multiplier when EET is active. Scales all non-router LRs (Muon + AdamW).
    eet_depth_weight_type: str = 'none'            # Token CE loss weighting type by exit depth ('none', 'linear', 'ema', 'sqrt')
    eet_depth_weight_max: float = 2.5              # Maximum weighting factor for deep tokens in linear strategy
    eet_use_override: int = 0                      # 1 = use stochastic depth override, 0 = disabled
    eet_override_prob_start: float = 0.5           # initial override probability during training
    eet_override_prob_end: float = 0.1             # final/minimum override probability
    # MoD-style fixed-capacity compute skipping
    eet_compute_skip: bool = False                 # True = gather/scatter active tokens for real FLOP savings
    eet_target_active_frac: float = 0.125          # fraction of tokens still active at deepest routable layer (geometric decay)
    eet_capacity_schedule: str = 'bell'            # capacity schedule ('uniform' | 'linear' | 'geometric' | 'bell')
    eet_exit_fracs: list[float] | None = None      # custom per-slot exit fractions override (comma-separated list of floats)
    eet_capacity_alignment_lambda: float = 0.0     # weight for load-balancing/capacity alignment loss (0=disabled)
    # --- P02 Test 1: context restoration for exited tokens (compute_skip path) ---
    # 'none'  : active queries attend only to active keys (current behaviour, destroys context)
    # 'fresh' : keys/values projected at THIS layer from every token's current-or-frozen state.
    #           Upper bound on quality; costs an extra 4*d^2 per exited token per layer.
    # 'stale' : keys/values banked at the layer each token exited (projected by that layer's
    #           W_k/W_v), reused unchanged afterwards. Near free: only active tokens project.
    eet_kv_mode: str = 'none'
    # --- P02 Test 2: stochastic routing (training only; eval stays deterministic) ---
    # Gumbel noise scale added to the per-slot exit score before top-K. Scores are softmax
    # probabilities in [0, 1], so 0.0 = deterministic, ~0.3 = mild exploration,
    # ~1.0 = strong, >=10 = effectively uniform-random routing.
    eet_route_noise: float = 0.0
    eet_route_noise_end: float = -1.0              # <0 = hold eet_route_noise constant; else linear anneal
    # --- P02 Test 0B: per-layer vocabulary coverage diagnostic ---
    eet_coverage_diag: bool = False                # accumulate which token ids reach which layer
    # --- Tier 0 test 2: deep supervision -------------------------------------
    # Adds a CE term at every layer through the SHARED head, so every layer is trained to
    # be prediction-readable. On a dense model this prices what EET gets for free and pays
    # for: EET's layer 3 reads at 1.041 against dense's 1.783, but its layer 7 reads at
    # 1.039 against dense's 0.982. If a healthy backbone cannot be made readable at every
    # depth without losing more than the EET gap, no architecture can have both.
    # Evaluated on a random subset of positions per layer so the cost is ~1 extra head
    # pass rather than n_layer of them.
    deep_supervision_lambda: float = 0.0
    deep_supervision_frac: float = 0.125
    # --- Tier 2 idea 5: the inverse-width stack ------------------------------
    # Depth collapse is not the deep layers being starved; it is the model refusing to
    # build hierarchy past the first exit (measured: moving min_exit_layer 1 -> 4 moves
    # the flat region from L3 to L5 and takes bpb 1.0622 -> 1.0194). Rather than fight
    # the readability pressure, accept it and give the deep layers something the shallow
    # ones cannot have: width. A layer running 10% of the tokens can be many times wider
    # at the same FLOPs, because block cost scales with (tokens x width^2).
    #
    #   ffn_mult[i] = base * (1 / active_frac[i]) ** eet_width_power
    #
    # power 0 is the current uniform stack; power 1 makes every layer's FFN cost the same
    # in FLOPs regardless of how few tokens reach it. This is the opposite of every
    # pyramid/funnel transformer, which tapers width DOWN with depth.
    eet_width_power: float = 0.0
    eet_width_cap: float = 16.0    # ceiling on the multiplier; 1/a hits 10x at a=0.10
    # Escape hatch: run the split-KV attention outside the compiled graph. Costs speed but
    # sidesteps any inductor stride-guard failure on the mask, so the T1 quality gate can
    # still produce a bpb number. Never use it for a wallclock claim.
    eet_kv_eager: bool = False
    eet_router_task_grad: bool = True              # allow task loss gradients to propagate to router through continue weights
    eet_reinforce_interval: int = 0                 # two-pass REINFORCE every N steps (0=disabled). Runs dense forward to get counterfactual CE.
    eet_reinforce_lambda: float = 0.1               # REINFORCE loss weight
    eet_exit_adapter_rank: int = 0                  # per-exit low-rank adapter rank (0=disabled). Maps early-exit representations to lm_head space.
    eet_router_after_block: int = 0                 # run global router after this block index (0=use raw embedding x0). Post-attention context helps routing.
    # A³D: Attention-Anchored Adaptive Depth (FFN-only skip)
    eet_ffn_skip: bool = False                      # True = A³D mode: skip FFN only, preserve attention at all layers
    eet_ffn_target_frac: float = 0.50               # fraction of tokens that GET FFN at each layer (rest skip FFN)
    eet_ffn_full_attn: bool = True                  # True = attention on all T tokens; False = gather for attention too
    # Novel EET improvements
    eet_depth_affine: bool = False                   # per-exit-depth learned (γ, β) affine applied before LM head (cheap alignment)
    eet_capacity_anneal_frac: float = 0.0           # fraction of training to anneal target_active_frac from 0.5 → configured (0=disabled)
    eet_learned_schedule: bool = False               # learn exit fractions instead of using fixed bell/uniform/geometric
    eet_departure_summary: bool = False              # inject mean of exiting tokens' hidden states into remaining active tokens
    eet_route_consistency_lambda: float = 0.0       # consistency loss weight penalizing routing oscillation (0=disabled)
    eet_dense_distill_interval: int = 0             # concurrent dense distillation every N steps (0=disabled)
    eet_dense_distill_lambda: float = 0.5           # KL distillation loss weight
    # Diagnostic-guided gradient fixes
    eet_depth_lr_scale: bool = False                 # Option A: per-layer LR scaling inversely proportional to surviving token fraction
    eet_depth_grad_scale: bool = False               # Option B: scale per-token CE by inverse of active fraction at exit depth
    eet_detach_aux_from_backbone: bool = False       # Detach aux losses (CE-guided, surprise) from backbone — only train router params
    eet_detach_exit_from_backbone: bool = False      # Detach exiting token representations from backbone graph — backbone only trains from final-layer tokens



    # -- SCH: Structured Code Output Heads -------------------------------------
    # Replaces the dense V x d softmax head with logit(w|h) = phi_k(c(w))^T g(h),
    # where c(w) is a frozen binary code and phi_k its monomial expansion up to
    # interaction order k. See nanochat/code_head.py for the derivation; the short
    # version is that order 1 is Oda et al. 2017 (logit rank <= B), order B is the
    # exact softmax, and everything between is unexplored.
    use_code_head: bool = False                     # master switch for the code output head
    sch_head_type: str = 'code'                     # code | hsoftmax | monarch | tiered | proposal | nfh (nonnegative factorized)
    sch_bits: int = 0                               # B, code length (0 = ceil(log2 V), the minimal/degenerate code)
    sch_order: int = 2                              # k, highest monomial interaction order kept
    sch_max_m: int = 0                              # cap on M (0 = uncapped); also sets M directly for non-monomial phi modes
    sch_phi_mode: str = 'monomial'                  # monomial | random_binary (structure control) | onehot (VQ-Logits) | learned (matched-capacity control) | gaussian
    sch_code_mode: str = 'binary'                   # binary | random | ecc | frequency | file
    sch_code_path: str = ''                         # .pt with a (V, B) uint8 code matrix (semantic codes from scripts/code_assign.py)
    sch_code_ecc_bits: int = 0                      # parity bits appended to the base code; sweeps the ECC-vs-semantic tension on one axis
    sch_code_seed: int = 1234                       # seed for code assignment and monomial subsampling
    sch_phi_density: float = 0.0                    # row density for phi_mode=random_binary (0 = match the monomial arm)
    sch_phi_dtype: str = 'bf16'                     # bf16 | fp32 storage for the frozen Phi
    sch_phi_normalize: int = 1                      # rescale Phi to unit mean row norm (reparameterisation; needed for stability at order 3+)
    sch_phi_center: int = 0                         # mean-centre Phi columns (ablation; adds the constant direction to the span)
    sch_g_type: str = 'linear'                      # linear | mlp. MLP separates "ladder saturated" from "hit the width cap d"
    sch_g_hidden: int = 0                           # hidden width of the MLP g (0 = n_embd)
    sch_g_layers: int = 2                           # depth of the MLP g
    sch_g_out_std: float = 0.001                    # init std of g's final layer, matching lm_head's
    sch_mixture: int = 1                            # m code heads mixed by log-sum-exp (>1 escapes the rank bound entirely)
    sch_residual_rank: int = 0                      # r for the dense residual hybrid: logit += v_w . W_r h, buys r rank for rV params
    sch_logit_act: str = 'none'                     # none | sigsoftmax | monotonic (pointwise nonlinearity on code logits)
    sch_bias: int = 0                               # learned per-token bias (V params, +1 rank, BREAKS zero-shot vocab extension)
    sch_input_mode: str = 'table'                   # table | linear | expanded | nonlinear | tied - Phase 3 input-side arms
    sch_input_hidden: int = 0                       # hidden width for sch_input_mode=nonlinear (0 = 4 * n_embd)
    # --- Phase 5 alternatives (see output-head-efficiency-directions.md) -----
    sch_product_groups: int = 8                     # g: digits in the K-ary product code (phi_mode=product)
    sch_product_codebook: int = 256                 # K: symbols per digit; M = g*K at order 1, no interactions
    sch_product_source: str = 'hash'                # hash | random | file (k-means assignment from code_assign.py)
    sch_product_impl: str = 'dense'                 # dense = materialise the one-hot Phi and GEMM (fast); gather = V*g arithmetic but needs a fused kernel (Q8)
    sch_phi_whiten: int = 0                         # Phi (Phi^T Phi)^-1/2: same span, cond 1, pure reparameterisation
    sch_mixture_per_phi: int = 0                    # each mixture component gets its OWN Phi (union of subspaces)
    sch_mixture_topk: int = 0                       # components evaluated per token (0 = all); cost tracks k, not K
    sch_mixture_aux: float = 0.01                   # load-balance weight for sparse routing; 0 lets top-1 collapse to one component
    sch_monarch_m1: int = 0                         # Monarch inner factor (0 = sqrt(M)); cost is d*M + V*m1
    sch_monarch_perm: str = 'none'                  # none | random (control) | freq | file. Which words share a Monarch block; costs nothing
    sch_monarch_perm_path: str = ''                 # .pt holding a permutation of range(vocab_size) (scripts/build_vocab_permutation.py)
    sch_tier_bounds: str = ''                       # cumulative vocab edges for sch_head_type=tiered, e.g. '1024,4096,16384,65536'
    sch_tier_caps: str = ''                         # dims per tier, e.g. '512,512,511,480,18' (scripts/solve_tiers.py solves these)
    sch_tier_order: str = 'freq'                    # none | random | freq | file. Which words land in which tier; freq is the point
    sch_tier_perm_path: str = ''                    # .pt permutation of range(vocab_size) for sch_tier_order=file
    sch_proposal_rank: int = 32                     # c for sch_head_type=proposal. Trained to RANK, so it stays 16-32 at any d
    sch_proposal_topk: int = 16384                   # K words scored exactly per token, plus the target
    sch_proposal_samples: int = 4096                # S importance samples for the tail; 0 makes the estimator biased
    sch_proposal_warmup: int = 0                    # steps of exact softmax before switching; the proposal is noise at init
    sch_proposal_chunk: int = 2048                  # tokens sharing one candidate set; also sets how many dense (V,d) grad buffers per step
    sch_proposal_aux: float = 1.0                   # weight on the ranking loss that trains the proposal from the exact logits
    # --- Nonnegative factorized heads (sch_head_type=nfh) -------------------
    # Reparameterises the DISTRIBUTION so the partition function factorises, rather
    # than approximating the V x d matrix. See nanochat/code_head.py.
    sch_nfh_mode: str = 'cp'                        # cp (mixture of product codes) | global (V x m log-table, cost independent of V)
    sch_nfh_rank: int = 32                          # R components for cp; m table columns for global. R=1 is exactly LightRNN
    sch_nfh_dims: str = ''                          # code axis sizes, e.g. '64,64,32'; must multiply to the PADDED vocab. Empty = auto
    sch_nfh_groups: int = 3                         # G used only when sch_nfh_dims is empty
    sch_nfh_views: int = 1                          # independent bijections mixed uniformly; view 0 uses sch_nfh_perm, the rest are random
    sch_nfh_perm: str = 'none'                      # none | random (control) | freq | file. Which words share a code cell
    sch_nfh_perm_path: str = ''                     # .pt permutation of range(vocab_size) (scripts/build_vocab_permutation.py)
    sch_nfh_g_type: str = 'linear'                  # linear | mlp. The map into the factors; mlp is nearly free once the head is 0.04x
    sch_nfh_g_hidden: int = 0                       # hidden width for sch_nfh_g_type=mlp (0 = n_embd)
    # --- Rerank head (sch_head_type=rerank): dense forward + a cheap learned
    # correction applied only where the probability mass is. Cost-side head work is
    # capped at +0.032 bpb at V=32,768 depth 8; this goes for a GAIN instead.
    sch_rerank_mode: str = 'topk'                   # topk | full | mos2 | temp
    sch_rerank_k: int = 64                          # words corrected; top-64 carries 98% of the mass
    sch_rerank_rank: int = 32                       # r; 0 reduces the head to dense exactly
    sch_nfh_smooth: int = 0                         # interpolate a static per-token prior into the mixture (Jelinek-Mercer). V params, one gather/token
    sch_nfh_chunk: int = 256                        # tokens per chunk on the EVAL path only; the training path builds no V-wide tensor
    # --- SAP: sampling-aware block head (nanochat/block_head.py, sap_research_plan.md) ---
    # One head emits the next T tokens per trunk pass. A plan latent drawn once per block is
    # what makes the T tokens agree; the trunk keeps its ordinary next-token loss.
    sap_block_T: int = 0                            # tokens per block; 0 disables the head entirely
    sap_block_mode: str = 'indep'                   # see nanochat.block_head.SAP_MODES
    sap_block_frac: float = 0.125                   # fraction of positions carrying a block in training; the head's cost scales with it
    sap_lambda: float = 1.0                         # weight of the block loss next to the next-token loss
    sap_head_layers: int = 2                        # attention layers in the slot decoder
    sap_head_heads: int = 0                         # 0 = n_embd // 64
    sap_head_mlp_mult: int = 2                      # MLP width multiple inside the head
    sap_enc_layers: int = 1                         # posterior encoder layers (p1, p2)
    sap_ctx_window: int = 16                        # trunk positions the slots cross-attend to (0 = h_t only)
    sap_latent_groups: int = 4                      # G code groups (p1); free-bits groups (p2)
    sap_latent_codes: int = 16                      # C codes per group (p1): C^G joint plans
    sap_latent_dim: int = 64                        # z dim (p2) and noise dim (p3, plain_noise, wta)
    sap_free_bits: float = 0.25                     # nats of KL per group that are not penalised
    sap_kl_anneal_steps: int = 2000                 # micro-steps to ramp the KL weight from 0 to 1
    sap_gumbel_tau: float = 1.0                     # straight-through Gumbel temperature (p1)
    sap_cp_components: int = 8                      # R for the cp mixture baseline
    sap_field_rank: int = 8                         # positional basis rank of the exact one-shot field_cp mixture
    sap_field_samples: int = 4                      # training-only prior draws for field_energy (inference is always one)
    sap_field_energy_weight: float = 0.25           # realised hard-block energy-score weight
    sap_field_topk: int = 16                        # sparse soft-gradient candidates; hard draw still uses full V
    sap_wta_k: int = 4                              # noise draws for the winner-take-all control
    sap_u_freqs: int = 24                           # sinusoidal frequencies resolving u (inv_head)
    sap_u_interior: float = 0.8                     # central fraction of a CDF bin u is drawn from (inv_head)
    sap_logit_chunk: int = 8192                     # slot rows per checkpointed readout chunk
    sap_jacobi_sweeps: int = 2                      # parallel refinement sweeps for local_jacobi (or local)
    # --- S01 Sample-Inject-Reference heads ---------------------------------
    sap_sir_topk: int = 16                          # retained draft candidates per slot (bounds B*T*V memory)
    sap_sir_rank: int = 32                          # low-rank compatibility/lattice width
    sap_sir_anchor_stride: int = 16                 # coarse sampled anchor spacing
    sap_sir_fine_stride: int = 4                    # second pyramid spacing
    sap_sir_refine_frac: float = 0.25               # fixed fraction resampled by confidence refinement
    sap_sir_tree_levels: int = 0                    # 0=full balanced tree; >0=anchor rounds before parallel fill
    sap_sir_train_samples: int = 4                  # realised drafts in the training marginal-likelihood estimate
    sap_sir_policy_weight: float = 1.0              # score-function credit from post-sampling reference weights
    sap_sir_posterior_mix: float = 0.5              # probability a training draft slot comes from the observed block
    sap_sir_draft_weight: float = 1.0               # draft CE weight; objectives are renormalised by total weight
    sap_sir_context_weight: float = 0.25             # contextual plausible-error auxiliary
    sap_sir_energy_weight: float = 0.25              # structured-negative energy auxiliary
    # --- SAP v4: exact sampling cuts (sap_research_plan.md v4) -------------
    sap_trunk_grad: float = 1.0                     # gradient the block loss sends into the trunk, lm_head and wte: 0 = detached (trunk exactly dense), 1 = co-trained
    sap_lattice_k: int = 64                         # candidates per slot on the lattice; the rest share one escape state
    sap_pair_rank: int = 32                         # rank of the CRF pair terms / TT and CP candidate features / class pairs
    sap_tt_rank: int = 32                           # latent states R of the tensor-train (HMM) heads
    sap_cp_codes: int = 256                         # global codes Z of lat_cp
    sap_nce_props: int = 0                          # >0: self-contrastive resampling over this many proposals at decode
    sap_nce_neg: int = 4                            # head-sampled negatives per block for the InfoNCE scorer
    sap_nce_weight: float = 1.0                     # weight of the InfoNCE scorer loss
    sap_table_path: str = ''                        # corpus tables (scripts/sap_corpus_tables.py) for pmi_chain, corpus_code, supp, soft
    sap_supp_min_count: int = 0                     # >0: CRF pairs seen fewer times in the corpus get -sap_supp_penalty
    sap_supp_penalty: float = 20.0
    sap_soft_eps: float = 0.0                       # weight of the n-gram soft-target auxiliary on slot readouts (C-soft)
    sap_code_classes: int = 128                     # corpus token classes for corpus_code
    sap_cut_pre_layers: int = 1                     # cut heads: ordinary slot layers before the stochastic cut (the rest fill)
    sap_depth_layers: int = 2                       # depth_local: top trunk layers the block's tokens pass through
    sap_depth_share: int = 0                        # depth_local: 1 = the trunk's own blocks, 0 = trainable copies (synced from the trunk)
    sap_depth_bottom: int = 0                       # depth slots: of the m layers, how many are the bottom ones (skip the middle)
    splice_k: int = 0                               # S13 SV-A splice codes: number of word classes (0 = off; nanochat/splice.py)


# Used by notebooks to validate kwargs passed to GPTConfig.
RESEARCH_ALLOWED_KEYS = {
    "use_moe", "use_perm",
    "moe_num_experts", "moe_router_dim", "moe_embed_dim", "dropout",
    "use_remix_linear", "remix_context_dim", "remix_context_dim_ratio", "remix_basis_size", "remix_output_gate_rank", "remixed_linear_kwargs",
    "remix_shared_context_gates",
    "use_pos_embed", "moe_use_abs_pos_embed",
    "router_context_window", "router_causal", "router_num_heads",
    "router_num_queries", "router_n_layers", "router_use_vocab_prior",
    # Fix 1A
    "use_layer_context",
    # Fix 1B
    "scale_basis_size",
    # Fix 1D
    "perm_expert_mode", "perm_rank",
    # CCL block redesign
    "cclblock_modulation", "cclblock_orth_lambda", "cclblock_context_stream", "cclblock_ema_factor", "cclblock_stale_ctx_lag",
    # Novel ablation designs
    "cclblock_sparse_gate_k", "cclblock_gate_temperature", "cclblock_context_bank_size", "cclblock_per_head_ctx",
    "cclblock_context_source",
    # Phase 8
    "cclblock_chunk_size", "cclblock_aux_objective", "cclblock_aux_lambda", "cclblock_boundary_token_id",
    # Phase 9
    "use_ral", "ral_rank", "cclblock_film_gate", "cclblock_attn_shadow_dim",
    # New paradigms
    "cclblock_dynamic_ratio", "cclblock_gate_rank", "cclblock_num_regimes", "cclblock_regime_temperature",
    "cclblock_poly_order", "cclblock_lie_generators", "cclblock_grassmann_bank_size",
    "cclblock_tucker_rank", "cclblock_tucker_modes", "cclblock_svs_rank", "cclblock_svs_eps",
    "cclblock_vq_codes", "cclblock_vq_temperature", "cclblock_dcu_warmup_steps",
    # Phase 12: FSI/AESP/CKR
    "cclblock_fsi_rotations", "cclblock_fsi_selector_dim",
    "cclblock_aesp_strata", "cclblock_aesp_delta_rank",
    "cclblock_ckr_branches", "cclblock_ckr_kernel_size",
    # Phase 13
    "cclblock_ckr_pos_channels", "cclblock_ckr_dual_optim", "cclblock_ckr_content_bias",
    # Phase 14
    "cclblock_giad_rank", "cclblock_psg_kernel_size",
    "cclblock_ss_dynamic_ratio", "cclblock_ss_branches", "cclblock_ss_kernel_size",
    # Phase 15
    "cclblock_lokr_branches", "cclblock_lokr_rank",
    # Phase 16
    "cclblock_ckr_temp_start", "cclblock_ckr_temp_end",
    "cclblock_com_kernel_size",
    # Phase 18
    "p18_layer_drop", "p18_dynamic_activation", "p18_mixture_norm", "p18_causal_attn_bias",
    "p18_aux_sim_lambda", "p18_gradient_penalty", "p18_per_channel_scale",
    # Phase 19
    "p19_residual_gate", "p19_head_importance", "p19_residual_mix_groups",
    "p19_attn_logit_bias", "p19_residual_decay", "p19_grad_equilibrium",
    "p19_spectral_reparam", "p19_weight_anticollapse", "p19_ve_bias", "p19_weight_noise",
    # Phase 20
    "p20_hrcs_scale", "p20_lswr_scale", "p20_lswr_planes",
    "p20_lrcfb_branches",
    "p20_lrcfb_narrow", "p20_lrcfb_learned", "p20_lrcfb_topk",
    "p20_dgcr_branches", "p20_dgcr_aux_weight",
    "p20_mone_experts", "p20_mone_topk", "p20_mone_narrow", "p20_mone_frozen",
    "p20_ncea_branches", "p20_ncea_eps",
    "p20_adwi",
    "p20_pwu_branches", "p20_pwu_phase",
    "p20_fsvd_gate",
    "p20_wbfc_clusters", "p20_wbfc_active",
    # Phase 21
    "p21_per_experts", "p21_per_topk", "p21_per_learned", "p21_per_attn",
    # Phase 22
    "p22_attn_moe_route",
    # Phase 23
    "p23_tiny_expert", "p23_n_experts", "p23_topk", "p23_learned_route",
    "p23_std_moe_experts", "p23_std_moe_topk", "p23_std_moe_aux_weight",
    "p23_lokr", "p23_lokr_rank",
    "p23_use_shared_block_router", "p23_linear_moe_experts", "p23_linear_moe_topk",
    # Phase 24
    "p24_use_sliced_weight", "p24_sliced_weight_reduction_scale", "p24_sliced_weight_min_select",
    "p24_sliced_weight_scope", "p24_sliced_weight_balance_coeff", "p24_quantile_route",
    "p24_use_folded_mod", "p24_folded_mod_reduction_scale", "p24_folded_mod_min_dim",
    "p24_folded_mod_scope",
    "p24_folded_mod_gate_act", "p24_use_sequence_gated_linear", "p24_sequence_gated_scope",
    "p24_sequence_gated_act",
    # Phase 30: LayerNorm ablation
    "remix_disable_ln_basis", "dense_intermediate_ln", "legacy_init_fallback",
    "p22_route_affine", "p34_ffn_schedule", "p34_ffn_no_ffn_replacement", "p34_ffn_last_depth",
    "p36_swiglu_ffn", "p36_swiglu_mult", "cond_sites",
    # Phase 35: ConditionedLinear
    "cond_rank", "cond_mult_steps", "cond_gate_source", "cond_coeff_act",
    "cond_router_rank", "cond_router_act", "cond_live_init", "cond_chunk_size",
    "cond_mult_impl", "cond_mult_scale",
    "cond_attn_projs", "cond_layer_frac",
    # MoL: Mixture of Layers (baseline, arXiv:2605.09516)
    # SCH: Structured Code Output Heads
    "use_code_head", "sch_head_type", "sch_bits", "sch_order", "sch_max_m",
    "sch_phi_mode", "sch_code_mode", "sch_code_path", "sch_code_ecc_bits",
    "sch_code_seed", "sch_phi_density", "sch_phi_dtype", "sch_phi_normalize",
    "sch_phi_center", "sch_g_type", "sch_g_hidden", "sch_g_layers", "sch_g_out_std",
    "sch_mixture", "sch_residual_rank", "sch_logit_act", "sch_bias",
    "sch_input_mode", "sch_input_hidden",
    "sch_product_groups", "sch_product_codebook", "sch_product_source", "sch_product_impl",
    "sch_phi_whiten", "sch_mixture_per_phi", "sch_mixture_topk", "sch_mixture_aux",
    "sch_monarch_m1", "sch_monarch_perm", "sch_monarch_perm_path",
    "sch_tier_bounds", "sch_tier_caps", "sch_tier_order", "sch_tier_perm_path",
    "sch_proposal_rank", "sch_proposal_topk", "sch_proposal_samples",
    "sch_proposal_warmup", "sch_proposal_chunk", "sch_proposal_aux",
    "sch_nfh_mode", "sch_nfh_rank", "sch_nfh_dims", "sch_nfh_groups",
    "sch_nfh_views", "sch_nfh_perm", "sch_nfh_perm_path", "sch_nfh_g_type",
    "sch_nfh_g_hidden", "sch_nfh_chunk", "sch_nfh_smooth",
    "sch_rerank_mode", "sch_rerank_k", "sch_rerank_rank",
    # SAP: sampling-aware block head
    "sap_block_T", "sap_block_mode", "sap_block_frac", "sap_lambda", "sap_head_layers",
    "sap_head_heads", "sap_head_mlp_mult", "sap_enc_layers", "sap_ctx_window",
    "sap_latent_groups", "sap_latent_codes", "sap_latent_dim", "sap_free_bits",
    "sap_kl_anneal_steps", "sap_gumbel_tau", "sap_cp_components", "sap_field_rank",
    "sap_field_samples", "sap_field_energy_weight", "sap_field_topk", "sap_wta_k",
    "sap_u_freqs", "sap_u_interior", "sap_logit_chunk", "sap_jacobi_sweeps",
    "sap_sir_topk", "sap_sir_rank", "sap_sir_anchor_stride", "sap_sir_fine_stride",
    "sap_sir_refine_frac", "sap_sir_tree_levels", "sap_sir_train_samples", "sap_sir_policy_weight", "sap_sir_posterior_mix",
    "sap_sir_draft_weight", "sap_sir_context_weight",
    "sap_sir_energy_weight",
    "sap_trunk_grad", "sap_lattice_k", "sap_pair_rank", "sap_tt_rank", "sap_cp_codes",
    "sap_nce_props", "sap_nce_neg", "sap_nce_weight", "sap_table_path", "sap_supp_min_count",
    "sap_supp_penalty", "sap_soft_eps", "sap_code_classes", "sap_cut_pre_layers",
    "sap_depth_layers", "sap_depth_share", "sap_depth_bottom",
    "splice_k",
    "use_mol", "mol_n_blocks", "mol_n_shared", "mol_topk", "mol_thin_dim",
    "mol_head_dim", "mol_ffn_mult", "mol_router_aux", "mol_routed_attn",
    "mol_dispatch", "mol_capacity_factor", "mol_block_lr_scale", "mol_per_block_ve",
    # MST: Modular Sub-Transformer
    "use_mst", "mst_n_subs", "mst_sub_dim", "mst_head_dim",
    "mst_input_mode", "mst_rotated_slice_learned",
    "mst_routing_mode", "mst_routing_topk", "mst_routing_aux_weight",
    "mst_diversity_weight",
    "mst_ffn_mode", "mst_ffn_every", "mst_transition_mode", "mst_final_mode",
    "mst_sub_aux_weight", "mst_progressive_merge", "mst_multi_scale_windows",
    "mst_delta_residual", "mst_sub_layers",
    # MST Stage 7: Scaling improvements
    "mst_grad_equalize", "mst_block_diagonal_muon", "mst_transition_width_mult",
    "mst_sub_lr_scale", "mst_shared_expert", "mst_router_entropy_weight",
    "mst_shared_kv_attn", "mst_contrastive_diversity_weight",
    # MST Stage 8: Transition expressivity
    "mst_transition_nonlinear", "mst_transition_gated", "mst_transition_mlp",
    # MST Stage 9: Cross-sub expressivity
    "mst_cross_sub_gate", "mst_hyper_connect", "mst_cross_kv_inject",
    # MST Stage 10: Structural transition improvements
    "mst_slice_transition", "mst_lookback_layers", "mst_bilinear_transition",
    # MST Stage 11: Attention bottleneck + structural improvements
    "mst_cross_sub_qmod", "mst_feature_cycle", "mst_mean_transition",
    # MST Stage 12: dense-parity fixes
    "mst_sub_head_dim", "mst_final_norm", "mst_per_stream_ve",
    "mst_ve_map", "mst_ve_map_rank",
    # MST Stage 13: overhead cuts
    "mst_lm_head_dim", "mst_compose_windows",
    # MST Stage 14: free cross-stream mixing
    "mst_channel_mix", "mst_channel_mix_site",
    # MST Stage 15: coupling optimization + attention cross-stream mixing
    "mst_distribute_block_muon", "mst_trans_spectral_lr",
    "mst_talking_heads", "mst_wo_mode",
    # MST Stage 16: conditional stream execution
    "mst_stream_topk", "mst_stream_router_aux", "mst_stream_gate_attn",
    "mst_stream_shared",
    "mst_stream_router_noise", "mst_stream_dispatch", "mst_stream_capacity_factor",
    # MST Stage 17: block-diagonal Shampoo
    "mst_shampoo", "mst_precond_every", "mst_shampoo_beta",
    # MST Stage 18: Monarch-structured FFN
    "mst_ffn_monarch",
    # EET: Early Exit Transformer
    "use_eet", "eet_frozen_kv", "eet_router_type", "eet_router_hidden",
    "eet_freq_prior_alpha", "eet_pos_prior_beta", "eet_domain_prior",
    "eet_warmup_frac", "eet_explore_frac", "eet_reconstruct_lambda",
    "eet_efficiency_lambda_start", "eet_efficiency_lambda_end",
    "eet_translator_rank", "eet_max_frozen_kv_frac",
    "eet_exit_threshold", "eet_min_exit_layer",
    "eet_loss_variant", "eet_topk_vocab",
    "eet_entropy_lambda", "eet_surprise_lambda",
    "eet_adv_lambda", "eet_adv_entropy_lambda",
    "eet_gumbel_temp_start", "eet_gumbel_temp_end", "eet_gumbel_hard",
    "eet_commitment_beta", "eet_global_router",
    "eet_freq_efficiency_alpha", "eet_diversity_lambda", "eet_ce_guided_lambda",
    "eet_depth_weight_type", "eet_depth_weight_max",
    "eet_use_override", "eet_override_prob_start", "eet_override_prob_end",
    "eet_reenter_final",
    "eet_compute_skip", "eet_target_active_frac",
    "eet_capacity_schedule", "eet_exit_fracs",
    "eet_capacity_alignment_lambda", "eet_router_task_grad",
    "eet_kv_mode", "eet_route_noise", "eet_route_noise_end", "eet_coverage_diag",
    "deep_supervision_lambda", "deep_supervision_frac",
    "eet_width_power", "eet_width_cap",
    "eet_kv_eager",
    "eet_depth_affine", "eet_capacity_anneal_frac", "eet_learned_schedule",
    "eet_departure_summary", "eet_route_consistency_lambda",
    "eet_dense_distill_interval", "eet_dense_distill_lambda",
}


def norm(x):
    return F.rms_norm(x, (x.size(-1),)).to(x.dtype)

class Linear(nn.Linear):
    """nn.Linear that casts weights to match input dtype in forward.
    Replaces autocast: master weights stay fp32 for optimizer precision,
    but matmuls run in the activation dtype (typically bf16 from embeddings)."""
    def forward(self, x):
        return F.linear(x, self.weight.to(dtype=x.dtype), self.bias.to(dtype=x.dtype) if self.bias is not None else None)


class ImprovedContextAwareRouter(nn.Module):
    """Context-aware router used by research embedding branches.

    Defaults: ``context_window=-1``, ``causal=True``, ``num_heads=4``,
    ``num_queries=8``, ``n_layers=1``, ``use_vocab_prior=False``.
    """
    def __init__(
        self,
        vocab_size,
        num_experts,
        router_dim,
        full_embed_dim,
        context_window=-1,
        causal=True,
        num_heads=4,
        num_queries=8,
        n_layers=1,
        use_vocab_prior=False,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.router_dim = router_dim
        self.context_window = context_window
        self.causal = causal
        self.num_heads = num_heads
        self.head_dim = router_dim // num_heads
        self.n_layers = n_layers
        self.use_vocab_prior = use_vocab_prior

        self.embed_proj = Linear(full_embed_dim, router_dim, bias=True)
        self.qkv_proj = Linear(router_dim, 3 * router_dim, bias=False)
        self.out_proj = Linear(router_dim, router_dim, bias=True)
        self.ln = nn.LayerNorm(router_dim)
        self.routing_queries = nn.Parameter(torch.randn(num_queries, router_dim))
        self.temperature_predictor = Linear(router_dim, 1, bias=True)
        self.expert_proj = Linear(router_dim, num_experts, bias=True)
        self.cross_expert_proj = Linear(router_dim, num_experts, bias=True)
        self.alpha_gate = Linear(router_dim, 1, bias=True)
        if use_vocab_prior:
            self.vocab_routing_bias = nn.Embedding(vocab_size, num_experts)

    def _create_mask(self, seq_len, device):
        positions = torch.arange(seq_len, device=device)
        pos_diff = positions.unsqueeze(0) - positions.unsqueeze(1)
        if self.context_window == -1:
            return pos_diff > 0 if self.causal else torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)
        if self.causal:
            return (pos_diff > 0) | (pos_diff < -self.context_window)
        window_half = self.context_window // 2
        return torch.abs(pos_diff) > window_half

    def _multi_head_attention(self, x):
        # Fix 1C: SDPA + QK-norm. Mirrors the main model for stability at scale.
        batch_size, seq_len, _ = x.shape
        qkv = self.qkv_proj(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        # QK-norm stabilises attention logit scale at larger router_dims
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        # Use fused SDPA when causal+full-context (most common case)
        is_full_causal = self.causal and (self.context_window == -1 or self.context_window >= seq_len)
        if is_full_causal:
            context = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            # Windowed or non-causal: fall back to explicit masked attention
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
            mask = self._create_mask(seq_len, x.device).unsqueeze(0).unsqueeze(0)
            attn_scores = attn_scores.masked_fill(mask, float('-inf'))
            attn_weights = F.softmax(attn_scores.float(), dim=-1).to(q.dtype)
            context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).reshape(batch_size, seq_len, self.router_dim)
        return self.out_proj(context) + x

    def _cross_attention(self, queries, context):
        """Cross-attention from learned routing queries into the full token context.

        The routing queries are fixed learned parameters (not derived from any
        specific token), so they cannot directly leak future token identity.
        Without a causal mask, each position receives a summary of the full
        visible context — a richer signal than a causal prefix average.
        At inference time the context is always the causal prefix anyway.

        Uses SDPA (Flash Attention when available) to avoid materializing the
        full (B, T, Q, T) attention matrix that caused Triton XBLOCK > 4096.

        Returns (B, T, D): per-position summary (same for all positions since
        queries are position-independent and context is the full sequence).
        """
        batch_size, seq_len, dim = context.shape
        queries = queries.to(dtype=context.dtype)  # (Q, D)

        # Keys/Values: (B, 1, T, D) — same context for every routing query.
        k = context.unsqueeze(1)  # (B, 1, T, D)

        outputs = []
        for q_vec in queries.unbind(0):  # iterate over Q routing queries
            # q: (B, 1, T, D) — same routing query at every target position.
            # .contiguous() lets SDPA pick Flash Attention when available.
            q = q_vec.view(1, 1, 1, dim).expand(batch_size, 1, seq_len, dim).contiguous()
            # Non-causal: each routing query attends to all T context tokens.
            out = F.scaled_dot_product_attention(q, k, k, is_causal=True)  # (B, 1, T, D)
            outputs.append(out)

        # (B, Q, T, D) → mean across Q routing queries → (B, T, D)
        return torch.cat(outputs, dim=1).mean(dim=1)

    def forward(self, full_embeds, input_ids=None):
        dtype = full_embeds.dtype
        x = self.embed_proj(full_embeds)
        for _ in range(self.n_layers):
            x = self._multi_head_attention(x)
        # Fix: ensure x matches LN's parameter dtype to avoid RuntimeError in generation/inference
        x = self.ln(x.to(dtype=self.ln.weight.dtype)).to(dtype=dtype)
        self_attn_logits = self.expert_proj(x)
        cross_attn_logits = self.cross_expert_proj(self._cross_attention(self.routing_queries, x))
        alpha = torch.sigmoid(self.alpha_gate(x))
        expert_logits = alpha * self_attn_logits + (1 - alpha) * cross_attn_logits
        if self.use_vocab_prior and input_ids is not None:
            expert_logits = expert_logits + self.vocab_routing_bias(input_ids)
        adaptive_temp = torch.sigmoid(self.temperature_predictor(x)) * 2.0 + 0.1
        adaptive_temp = torch.clamp(adaptive_temp, min=1e-6)
        return expert_logits, adaptive_temp, self.expert_proj.weight


class DirectContextualEmbedding(nn.Module):
    """Direct contextual embedding with context-aware routing over K learned expert codes.

    Fix 1E: Routes among ``num_experts`` (K, typically 8) learned direction vectors rather
    than misusing num_experts=dim. Each token's embedding = seed + router-weighted expert code.
    This is parameter-efficient (K×dim extra params) and conceptually correct per the paper.
    """

    def __init__(
        self,
        vocab_size,
        dim,
        num_experts,
        router_dim,
        context_window,
        dropout=0.0,
        router_causal=True,
        router_num_heads=4,
        router_num_queries=8,
        router_n_layers=2,
        router_use_vocab_prior=False,
    ):
        super().__init__()
        self.seed_embeddings = nn.Embedding(vocab_size, dim)
        # K learned expert code vectors (context-independent, shared across vocab)
        # Small: K × dim params (e.g. 8 × 256 = 2048)
        self.expert_codes = nn.Parameter(torch.zeros(num_experts, dim))
        self.router = ImprovedContextAwareRouter(
            vocab_size=vocab_size,
            num_experts=num_experts,   # correctly K, not dim
            router_dim=router_dim,
            full_embed_dim=dim,
            context_window=context_window,
            causal=router_causal,
            num_heads=router_num_heads,
            num_queries=router_num_queries,
            n_layers=router_n_layers,
            use_vocab_prior=router_use_vocab_prior,
        )
        self.dropout = nn.Dropout(dropout)
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, input_ids):
        seeds = self.seed_embeddings(input_ids)                              # (B, T, dim)
        expert_logits, adaptive_temp, _ = self.router(seeds, input_ids)     # (B, T, K)
        expert_weights = F.softmax(expert_logits / adaptive_temp, dim=-1)   # (B, T, K)
        # Context-modulated perturbation: weighted sum of K expert codes
        # Ensure codes match weights/seeds dtype for einsum and addition
        expert_codes = self.expert_codes.to(dtype=seeds.dtype)
        perturbation = torch.einsum('btk,kd->btd', expert_weights.to(dtype=seeds.dtype), expert_codes)  # (B, T, dim)
        output = seeds + perturbation
        return self.out_norm(output), {'expert_weights': expert_weights}


class PermutationMoE(nn.Module):
    """Permutation MoE embedding with configurable expert router defaults.

    Defaults to learned absolute positional embeddings (``moe_use_abs_pos_embed=True``).

    Fix 1D: Supports three expert modes controlled by ``perm_expert_mode``:
    - ``'full'``:     Original D×D selection matrix per expert (O(D²) cost).
    - ``'low_rank'``: D×rank selection weights over learned rank basis vectors (O(D·rank)).
                     rank = max(8, base_embed_dim // perm_rank) where perm_rank acts as divisor.
    - ``'factored'``: Block-diagonal — D split into (D//perm_rank) blocks of perm_rank each.
                     Independent perm_rank×perm_rank permutation per block (O(D·perm_rank)).
    Fix 1H: adaptive_temp is clamped to [0.5, 2.1] to prevent routing collapse from near-zero temps.
    """

    def __init__(
        self,
        vocab_size,
        block_size,
        base_embed_dim,
        num_experts=8,
        router_dim=64,
        selection_mode='soft',
        allow_replacement=True,
        dropout=0.0,
        router_context_window=-1,
        router_causal=True,
        router_num_heads=4,
        router_num_queries=8,
        router_n_layers=2,
        router_use_vocab_prior=False,
        moe_use_abs_pos_embed=True,
        perm_expert_mode='low_rank',
        perm_rank=16,
    ):
        super().__init__()
        self.base_embed_dim = base_embed_dim
        self.num_experts = num_experts
        self.selection_mode = selection_mode
        self.allow_replacement = allow_replacement
        self.perm_expert_mode = perm_expert_mode
        self.embeddings = nn.Embedding(vocab_size, base_embed_dim)
        self.position_embeddings = nn.Embedding(block_size, base_embed_dim) if moe_use_abs_pos_embed else None

        D = base_embed_dim
        if perm_expert_mode == 'full':
            # Original: each expert outputs D×D logits (O(D²) cost)
            self.perm_rank = D
            self.dim_selectors = nn.ModuleList([
                nn.Sequential(
                    Linear(D, router_dim, bias=False),
                    nn.LayerNorm(router_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    Linear(router_dim, D * D, bias=False),
                ) for _ in range(num_experts)
            ])
        elif perm_expert_mode == 'low_rank':
            # Fix 1D low_rank: D×rank selection weights + learned (rank, D) basis per expert
            # rank = max(8, D // perm_rank)  — perm_rank acts as a divisor, e.g. perm_rank=16 ⇒ rank=D//16
            r = max(8, D // perm_rank)
            self.perm_rank = r
            self.dim_selectors = nn.ModuleList([
                nn.Sequential(
                    Linear(D, router_dim, bias=False),
                    nn.LayerNorm(router_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    Linear(router_dim, D * r, bias=False),  # D*rank outputs
                ) for _ in range(num_experts)
            ])
            # Learned basis: (K, rank, D) — K sets of rank D-dimensional basis vectors
            self.perm_basis = nn.ParameterList([
                nn.Parameter(torch.randn(r, D) * (D ** -0.5))
                for _ in range(num_experts)
            ])
        elif perm_expert_mode == 'factored':
            # Fix 1D factored (block-diagonal): block_size=perm_rank, num_blocks=D//perm_rank
            # perm_rank must evenly divide D; if not, round to nearest divisor.
            bs = perm_rank
            while D % bs != 0 and bs > 1:
                bs -= 1
            nb = D // bs
            self.perm_rank = bs   # block_size stored in perm_rank attr
            self.perm_num_blocks = nb
            self.dim_selectors = nn.ModuleList([
                nn.Sequential(
                    Linear(D, router_dim, bias=False),
                    nn.LayerNorm(router_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    Linear(router_dim, nb * bs * bs, bias=False),  # nb independent bs×bs matrices
                ) for _ in range(num_experts)
            ])
        else:
            raise ValueError(f"Unknown perm_expert_mode: {perm_expert_mode!r}. Use 'full', 'low_rank', or 'factored'.")

        self.expert_router = ImprovedContextAwareRouter(
            vocab_size=vocab_size,
            num_experts=num_experts,
            router_dim=router_dim,
            full_embed_dim=base_embed_dim,
            context_window=router_context_window,
            causal=router_causal,
            num_heads=router_num_heads,
            num_queries=router_num_queries,
            n_layers=router_n_layers,
            use_vocab_prior=router_use_vocab_prior,
        )
        self.ln = nn.LayerNorm(base_embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('temperature', torch.tensor(1.0))

    def forward(self, input_ids):
        B, T = input_ids.shape
        D = self.base_embed_dim
        embeds = self.embeddings(input_ids)
        if self.position_embeddings is not None:
            positions = torch.arange(T, device=input_ids.device)
            positions = torch.clamp(positions, max=self.position_embeddings.num_embeddings - 1)
            embeds = embeds + self.position_embeddings(positions)

        expert_outputs = []
        for expert_idx in range(self.num_experts):
            raw = self.dim_selectors[expert_idx](embeds)  # shape depends on mode

            if self.perm_expert_mode == 'full':
                selection_logits = raw.view(B, T, D, D).clamp(-30, 30)
                if self.selection_mode == 'hard' and not self.allow_replacement:
                    selection_weights = F.gumbel_softmax(selection_logits, tau=self.temperature, hard=True, dim=-1)
                elif self.selection_mode == 'hard':
                    selection_weights = F.gumbel_softmax(selection_logits, tau=self.temperature, hard=False, dim=-1)
                else:
                    selection_weights = F.softmax(selection_logits / self.temperature, dim=-1)
                selected = torch.einsum('bloi,bli->blo', selection_weights, embeds)

            elif self.perm_expert_mode == 'low_rank':
                r = self.perm_rank
                # (B, T, D, r) selection weights: for each output dim, weights over r basis vectors
                sel_w = F.softmax(raw.view(B, T, D, r).clamp(-30, 30) / self.temperature, dim=-1)
                # Project embeds onto rank-r basis: (B, T, r) scalar projections
                basis = self.perm_basis[expert_idx]  # (r, D)
                basis_proj = torch.einsum('bti,ri->btr', embeds, basis)  # (B, T, r)
                # selected[b,t,d] = sum_r( sel_w[b,t,d,r] * basis_proj[b,t,r] )
                selected = torch.einsum('btdr,btr->btd', sel_w, basis_proj)  # (B, T, D)

            else:  # factored
                bs = self.perm_rank
                nb = self.perm_num_blocks
                selection_logits = raw.view(B, T, nb, bs, bs).clamp(-30, 30)
                selection_weights = F.softmax(selection_logits / self.temperature, dim=-1)  # (B, T, nb, bs, bs)
                embeds_blocked = embeds.view(B, T, nb, bs)
                selected = torch.einsum('btnoi,btni->btno', selection_weights, embeds_blocked).reshape(B, T, D)

            expert_outputs.append(selected)

        expert_outputs = torch.stack(expert_outputs, dim=2)  # (B, T, K, D)
        expert_logits, adaptive_temp, _ = self.expert_router(embeds, input_ids)
        # Fix 1H: clamp adaptive_temp from below to prevent near-zero softmax collapse
        adaptive_temp = adaptive_temp.clamp(min=0.5)
        expert_weights = F.softmax(expert_logits / (self.temperature * adaptive_temp), dim=-1)
        expert_weights = self.dropout(expert_weights)
        # expert_outputs is (B, T, K, D), expert_weights is (B, T, K)
        # Match dtypes for final summation
        output = (expert_weights.to(dtype=expert_outputs.dtype).unsqueeze(-1) * expert_outputs).sum(dim=2)
        return self.ln(output), {'expert_weights': expert_weights}


class GlobalContextManager(nn.Module):
    """Global context manager built on the context-aware router.

    Defaults: ``router_num_heads=4``, ``router_num_queries=8``,
    ``router_n_layers=2``, ``router_use_vocab_prior=False``.
    """

    def __init__(
        self,
        vocab_size,
        d_model,
        router_dim=64,
        context_window=128,
        router_causal=True,
        router_num_heads=4,
        router_num_queries=8,
        router_n_layers=2,
        router_use_vocab_prior=False,
    ):
        super().__init__()
        self.router = ImprovedContextAwareRouter(
            vocab_size=vocab_size,
            num_experts=router_dim,
            router_dim=router_dim,
            full_embed_dim=d_model,
            context_window=context_window,
            causal=router_causal,
            num_heads=router_num_heads,
            num_queries=router_num_queries,
            n_layers=router_n_layers,
            use_vocab_prior=router_use_vocab_prior,
        )

    def forward(self, x_embeds, input_ids=None):
        context_logits, _, _ = self.router(x_embeds, input_ids)
        return F.layer_norm(context_logits, context_logits.shape[-1:])


class EMAContextStream(nn.Module):
    """Legacy Exponential Moving Average context stream (Design A).

    Uses a fixed EMA factor and detaches history to prevent long-range gradient 
    explosion, stabilizing the model at the cost of theoretically shorter context gradient flow.
    """
    def __init__(self, n_embd, ctx_dim, ema_factor=0.99):
        super().__init__()
        self.ema_factor = ema_factor
        self.proj = Linear(n_embd, ctx_dim, bias=False)

    def forward(self, attn_out, prev_ctx):
        new_ctx = self.proj(attn_out)
        if prev_ctx is not None:
            return self.ema_factor * prev_ctx.detach() + (1 - self.ema_factor) * new_ctx
        return new_ctx


class LocalContextStream(nn.Module):
    """Purely local context logic derived from pre-MLP norm(x). Ignores previous block states."""
    def __init__(self, n_embd, ctx_dim):
        super().__init__()
        self.proj = Linear(n_embd, ctx_dim, bias=True)
        nn.init.zeros_(self.proj.weight) # CRITICAL: zero-init so gates start at identity

    def forward(self, pre_mlp_norm_x, prev_ctx=None):
        return self.proj(pre_mlp_norm_x)


class ContextBank(nn.Module):
    """Design 4: Soft lookup into a learned vocabulary of N context prototypes.

    Projects norm(x) to form a query, retrieves a weighted mixture of learned
    prototype vectors. Forces the model to learn a discrete vocabulary of text-type
    contexts (code, prose, math, etc.) as prototypes.

    Zero-init prototypes + small std init ensures identity-like behaviour at t=0.
    Gradient path: loss → ffwd gate → ctx → weights (softmax) → query_proj → norm(x).
    No recurrence, no chain across blocks.
    """
    def __init__(self, n_embd, ctx_dim, n_prototypes):
        super().__init__()
        self.query_proj = Linear(n_embd, ctx_dim, bias=False)
        nn.init.zeros_(self.query_proj.weight)  # stable identity start
        self.prototypes = nn.Parameter(torch.zeros(n_prototypes, ctx_dim))
        nn.init.normal_(self.prototypes, std=0.02)
        self.scale = ctx_dim ** -0.5

    def forward(self, norm_x, prev_ctx=None):
        q = self.query_proj(norm_x)                                 # (B, T, ctx_dim)
        scores = q @ self.prototypes.to(q.dtype).T * self.scale     # (B, T, n_proto)
        weights = F.softmax(scores, dim=-1)
        return weights @ self.prototypes.to(q.dtype)                # (B, T, ctx_dim)


class SelectiveContextStream(nn.Module):
    """GRU-style selective update for the cross-block context state (Design A).

    Replaces the fixed-λ EMA + detach used in the old RemixedBlock with
    input-dependent gating:
      - alpha = sigmoid(gate(attn_out))  — update gate, input-dependent
      - content = tanh(write(attn_out)) — new content
      - ctx = (1-alpha)*prev_ctx + alpha*content  — no detach, full grad flow

    Zero-init on write weight/bias ensures no spurious content at step 0.
    gate bias=0 → sigmoid(0)=0.5 (balanced at init).
    """
    def __init__(self, n_embd, ctx_dim):
        super().__init__()
        self.gate  = Linear(n_embd, ctx_dim, bias=True)
        self.write = Linear(n_embd, ctx_dim, bias=True)
        nn.init.zeros_(self.gate.bias)
        nn.init.zeros_(self.write.weight)
        nn.init.zeros_(self.write.bias)

    def forward(self, attn_out, prev_ctx):
        alpha   = torch.sigmoid(self.gate(attn_out))          # (B, T, ctx_dim)
        content = torch.tanh(self.write(attn_out))            # (B, T, ctx_dim)
        if prev_ctx is None:
            return alpha * content
        # No detach: full gradient flow through context history
        return (1 - alpha) * prev_ctx + alpha * content


class MultiScaleContext(nn.Module):
    """Three parallel selective state channels with different temporal scales (Design D).

    Fast  (bias=+2.0 → α≈0.88): updates aggressively — captures recent syntax/tokens.
    Medium(bias= 0.0 → α≈0.50): balanced — captures paragraph-level topic.
    Slow  (bias=-2.0 → α≈0.12): retains history — captures document-level style.

    ctx_dim is auto-corrected to the nearest multiple of 3 when this class is
    instantiated, so the caller should read self.ctx_dim after construction.
    """
    GATE_BIASES = [2.0, 0.0, -2.0]   # sigmoid → [0.88, 0.50, 0.12]

    def __init__(self, n_embd, ctx_dim):
        super().__init__()
        # Auto-correct to nearest multiple of 3 (required for 3 equal sub-channels)
        corrected = max(3, round(ctx_dim / 3) * 3)
        self.ctx_dim   = corrected
        self.scale_dim = corrected // 3
        self.gates  = nn.ModuleList([Linear(n_embd, self.scale_dim, bias=True)  for _ in range(3)])
        self.writes = nn.ModuleList([Linear(n_embd, self.scale_dim, bias=True)  for _ in range(3)])
        for gate, bias_val in zip(self.gates, self.GATE_BIASES):
            nn.init.constant_(gate.bias, bias_val)
        for write in self.writes:
            nn.init.zeros_(write.weight)
            nn.init.zeros_(write.bias)

    def forward(self, attn_out, prev_ctx):
        chunks = []
        for i in range(3):
            alpha   = torch.sigmoid(self.gates[i](attn_out))
            content = torch.tanh(self.writes[i](attn_out))
            if prev_ctx is None:
                chunks.append(alpha * content)
            else:
                prev = prev_ctx[..., i*self.scale_dim:(i+1)*self.scale_dim]
                chunks.append((1 - alpha) * prev + alpha * content)
        return torch.cat(chunks, dim=-1)


class ParallelLinearContextStream(nn.Module):
    """Linear state-space style context highway with stable exponential decay."""
    def __init__(self, n_embd, ctx_dim):
        super().__init__()
        self.ctx_dim = ctx_dim
        self.dt_proj = Linear(n_embd, ctx_dim, bias=True)
        self.x_proj = Linear(n_embd, ctx_dim, bias=True)
        self.a_log = nn.Parameter(torch.zeros(ctx_dim))
        nn.init.zeros_(self.dt_proj.weight)
        nn.init.zeros_(self.dt_proj.bias)
        nn.init.zeros_(self.x_proj.weight)
        nn.init.zeros_(self.x_proj.bias)

    def forward(self, x, prev_ctx=None):
        bsz, seqlen, _ = x.shape
        dt = F.softplus(self.dt_proj(x))
        decay = torch.exp(-dt * torch.exp(self.a_log).view(1, 1, -1))
        write = self.x_proj(x)
        state = torch.zeros(bsz, self.ctx_dim, device=x.device, dtype=x.dtype)
        outputs = []
        for t in range(seqlen):
            state = state * decay[:, t, :] + write[:, t, :]
            outputs.append(state)
        return torch.stack(outputs, dim=1)


class DetachedAttnContextStream(nn.Module):
    """Proposal B: Projects norm(attn_out).detach() to ctx_dim.
    
    attn_out is intrinsically orthogonal to x_t (it's the cross-token mixture).
    """
    def __init__(self, n_embd, ctx_dim):
        super().__init__()
        self.proj = Linear(n_embd, ctx_dim, bias=True)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, attn_out, prev_ctx=None):
        return self.proj(norm(attn_out).detach())


class CausalPrefixContextStream(nn.Module):
    """Proposal E: Causal Prefix Context via cumsum (O(T), no recurrence)."""
    def __init__(self, n_embd, ctx_dim):
        super().__init__()
        self.proj = Linear(n_embd, ctx_dim, bias=True)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x, prev_ctx=None):
        cumsum = x.cumsum(dim=1)
        counts = torch.arange(1, x.size(1) + 1, device=x.device, dtype=x.dtype).view(1, -1, 1)
        return self.proj(norm(cumsum / counts).detach())


class WarmupEMAContextStream(nn.Module):
    """Proposal D: Position-Aware EMA Warmup."""
    def __init__(self, n_embd, ctx_dim, ema_factor=0.95):
        super().__init__()
        self.ema_factor = ema_factor
        self.ema_proj = Linear(n_embd, ctx_dim, bias=False)
        self.local_proj = Linear(n_embd, ctx_dim, bias=True)
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.bias = nn.Parameter(torch.tensor(-2.0))

        nn.init.zeros_(self.ema_proj.weight)
        nn.init.zeros_(self.local_proj.weight)

    def forward(self, x, prev_ctx=None):
        B, T, D = x.shape
        ema_signal = self.ema_proj(x)
        local_signal = self.local_proj(norm(x).detach())
        
        t = torch.arange(T, device=x.device, dtype=x.dtype)
        blend = torch.sigmoid(self.alpha * torch.log(t + 1.0) + self.bias).view(1, -1, 1)
        
        if prev_ctx is None:
            ema_ctx = ema_signal
        else:
            ema_ctx = self.ema_factor * prev_ctx.detach() + (1 - self.ema_factor) * ema_signal
            
        return blend * ema_ctx + (1 - blend) * local_signal


class DACSEMAContextStream(nn.Module):
    """Proposal F: Combines DACS (attn_out source) with Selective EMA smoothing."""
    def __init__(self, n_embd, ctx_dim):
        super().__init__()
        self.gate  = Linear(n_embd, ctx_dim, bias=True)
        self.write = Linear(n_embd, ctx_dim, bias=True)
        nn.init.zeros_(self.write.weight)
        nn.init.zeros_(self.gate.weight)
        if self.gate.bias is not None:
            nn.init.constant_(self.gate.bias, -2.0)  # closed at init

    def forward(self, attn_out, prev_ctx=None):
        content = torch.tanh(self.write(attn_out.detach()))
        alpha = torch.sigmoid(self.gate(attn_out.detach()))
        if prev_ctx is None:
            return alpha * content
        return (1 - alpha) * prev_ctx.detach() + alpha * content


class DecayPrefixContextStream(nn.Module):
    """Proposal G: Exponentially-decayed causal prefix mean.
    Computed via loop over T to approximate vector-scan.
    """
    def __init__(self, n_embd, ctx_dim, gamma=0.9):
        super().__init__()
        self.proj = Linear(n_embd, ctx_dim, bias=True)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        self.gamma = gamma

    def forward(self, x, prev_ctx=None):
        B, T, D = x.shape
        states = []
        curr = torch.zeros(B, D, device=x.device, dtype=x.dtype)
        for t in range(T):
            curr = self.gamma * curr + (1 - self.gamma) * x[:, t]
            states.append(curr)
        out = torch.stack(states, dim=1)
        return self.proj(norm(out).detach())


class BoundaryGatedContextStream(nn.Module):
    """Design 8: Soft boundary-gated context — update context only at learned segment boundaries.

    A lightweight binary gate decides per-position whether to refresh the context vector
    or hold the previous segment's context state fixed.

    Init strategy:
      - ctx_proj.weight = 0   → context starts as pure bias (neutral)
      - boundary_proj.bias = -2 → sigmoid(-2) ≈ 0.12, gate starts nearly closed
        (context is held fixed 88% of the time at init)

    Gradient flow:
      - prev_ctx is DETACHED: no BPTT through the full sequence chain (avoids the
        SelectiveContextStream failure mode).
      - Only the local ctx_new path and boundary_prob gate carry gradients.
      - context learns to update at genuine boundaries via the gate gradient signal.
    """
    def __init__(self, n_embd, ctx_dim):
        super().__init__()
        self.ctx_proj = Linear(n_embd, ctx_dim, bias=True)
        nn.init.zeros_(self.ctx_proj.weight)
        self.boundary_proj = Linear(n_embd, 1, bias=True)
        nn.init.zeros_(self.boundary_proj.weight)
        nn.init.constant_(self.boundary_proj.bias, -2.0)  # gate≈0.12 at init

    def forward(self, norm_x, prev_ctx=None):
        ctx_new = self.ctx_proj(norm_x)                                    # (B, T, ctx_dim)
        boundary_prob = torch.sigmoid(self.boundary_proj(norm_x))          # (B, T, 1)
        if prev_ctx is None:
            return ctx_new
        return boundary_prob * ctx_new + (1 - boundary_prob) * prev_ctx.detach()


class HardChunkContextStream(nn.Module):
    """Design 9: Context = pooled summary of the immediately preceding K-token chunk.

    Strictly causal implementation:
      - Tokens 0..K-1  receive zero context (no prior chunk exists).
      - Tokens K..2K-1 receive ctx = ctx_proj(mean(norm_x[0:K])).
      - Tokens 2K..3K-1 receive ctx = ctx_proj(mean(norm_x[K:2K])), etc.

    Fully vectorized via unfold + shift — no Python loops over T, compile-safe.
    Zero recurrence: each chunk is fully independent of all others.
    """
    def __init__(self, n_embd, ctx_dim, chunk_size):
        super().__init__()
        self.chunk_size = chunk_size
        self.ctx_dim = ctx_dim
        self.ctx_proj = Linear(n_embd, ctx_dim, bias=True)
        nn.init.zeros_(self.ctx_proj.weight)

    def forward(self, norm_x, prev_ctx=None):
        B, T, C = norm_x.shape
        K = self.chunk_size
        ctx_dim = self.ctx_dim
        # Pad to multiple of K
        pad = (K - T % K) % K
        x_pad = F.pad(norm_x, (0, 0, 0, pad)) if pad > 0 else norm_x  # (B, T', C)
        n_chunks = x_pad.shape[1] // K
        # Compute mean of each K-token chunk → project to ctx_dim
        x_chunks = x_pad.view(B, n_chunks, K, C).mean(dim=2)        # (B, n_chunks, C)
        ctx_chunks = self.ctx_proj(x_chunks)                         # (B, n_chunks, ctx_dim)
        # Causal shift: chunk i receives summary of chunk i-1 (zero for i=0)
        ctx_chunks_shifted = F.pad(ctx_chunks, (0, 0, 1, 0))[:, :-1, :]  # (B, n_chunks, ctx_dim)
        # Expand back to token level
        ctx = ctx_chunks_shifted.unsqueeze(2).expand(-1, -1, K, -1).contiguous()
        ctx = ctx.view(B, -1, ctx_dim)[:, :T, :]                    # (B, T, ctx_dim)
        return ctx.to(norm_x.dtype)


class PredictiveChunkContextStream(nn.Module):
    """Paradigm 3: predictive, zero-lag chunk context.

    Predict chunk i+1 context from chunk i summary, then hold it constant within
    chunk i+1. This avoids stale chunk lag and removes per-token context noise.
    """
    def __init__(self, n_embd, ctx_dim, chunk_size=64):
        super().__init__()
        self.chunk_size = chunk_size
        self.ctx_dim = ctx_dim
        self.future_predictor = nn.Sequential(
            Linear(n_embd, n_embd, bias=True),
            nn.GELU(),
            Linear(n_embd, ctx_dim, bias=True),
        )
        nn.init.zeros_(self.future_predictor[-1].weight)
        nn.init.zeros_(self.future_predictor[-1].bias)

    def forward(self, norm_x, prev_ctx=None):
        B, T, C = norm_x.shape
        K = self.chunk_size
        pad = (K - T % K) % K
        x_pad = F.pad(norm_x, (0, 0, 0, pad)) if pad > 0 else norm_x
        n_chunks = x_pad.shape[1] // K
        x_chunks = x_pad.view(B, n_chunks, K, C).mean(dim=2)
        predicted_future_ctx = self.future_predictor(x_chunks)
        ctx_chunks_shifted = F.pad(predicted_future_ctx, (0, 0, 1, 0))[:, :-1, :]
        ctx = ctx_chunks_shifted.unsqueeze(2).expand(-1, -1, K, -1).contiguous()
        ctx = ctx.view(B, -1, self.ctx_dim)[:, :T, :]
        return ctx.to(norm_x.dtype)


class EvidenceAccumulationContextStream(nn.Module):
    """Paradigm 2: linear low-dim evidence SSM for regime selection context."""
    def __init__(self, n_embd, ctx_dim, num_regimes=8, temperature=1.0):
        super().__init__()
        self.num_regimes = num_regimes
        self.temperature = max(float(temperature), 1e-6)
        self.evidence_proj = Linear(n_embd, num_regimes, bias=True)
        self.decay_proj = Linear(n_embd, num_regimes, bias=True)
        self.ctx_from_regime = Linear(num_regimes, ctx_dim, bias=False)
        nn.init.zeros_(self.evidence_proj.weight)
        nn.init.zeros_(self.evidence_proj.bias)
        nn.init.zeros_(self.decay_proj.weight)
        nn.init.constant_(self.decay_proj.bias, 4.5)  # sigmoid(4.5)≈0.99
        nn.init.zeros_(self.ctx_from_regime.weight)

    def forward(self, norm_x, prev_ctx=None):
        B, T, _ = norm_x.shape
        evidence = self.evidence_proj(norm_x)
        decay = torch.sigmoid(self.decay_proj(norm_x))
        states = []
        curr_state = torch.zeros(B, self.num_regimes, device=norm_x.device, dtype=norm_x.dtype)
        for t in range(T):
            curr_state = decay[:, t, :] * curr_state + evidence[:, t, :]
            states.append(curr_state)
        S_t = torch.stack(states, dim=1)
        R_t = F.softmax(S_t / self.temperature, dim=-1)
        return self.ctx_from_regime(R_t.to(dtype=norm_x.dtype))


class AdaRMSNorm(nn.Module):
    """Context-conditioned RMSNorm for the 'normalization' CCL path (Design B).

    Instead of modulating weight matrices (RemixedLinear), the context predicts
    per-position (scale, shift) for the post-norm representation — the DiT insight
    applied to language model blocks.

    At init: proj.weight=0, proj.bias[:n_embd]=1.0 (scale), proj.bias[n_embd:]=0.0 (shift).
    This guarantees exact standard RMSNorm at initialization regardless of ctx quality.
    Clean gradient path: loss → output → (scale*norm_x + shift) → ctx → context stream.
    """
    def __init__(self, n_embd, ctx_dim):
        super().__init__()
        self.n_embd = n_embd
        self.proj = Linear(ctx_dim, 2 * n_embd, bias=True)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        self.proj.bias.data[:n_embd] = 1.0   # scale starts at 1.0 (identity)
        # shift stays at 0.0 (no bias at init)

    def forward(self, x, ctx):
        x_norm = F.rms_norm(x, (x.shape[-1],))
        if ctx is None:
            return x_norm
        params = self.proj(ctx.to(dtype=x.dtype))    # (B, T, 2*n_embd)
        scale, shift = params.chunk(2, dim=-1)
        return x_norm * scale + shift


class CCLBlock(nn.Module):
    """Clean Context-Conditioned Language Block — 'normalization' modulation path.

    Standard dense CausalSelfAttention + MLP with AdaRMSNorm conditioning.
    No RemixedLinear complexity (no basis gates, output gates, template mixing).

    Context lifecycle:
      1. Attention input is conditioned by prev_ctx via AdaRMSNorm (scale+shift).
      2. attn_out is fed to SelectiveContextStream/MultiScaleContext → fresh ctx.
         (AFTER attention, so ctx carries cross-token aggregated signal.)
      3. MLP input is conditioned by freshly updated ctx via AdaRMSNorm.
      4. ctx is returned as next block's prev_ctx.

    This breaks the circular conditioning of the old AG-CCL:
      - ctx is derived from attn_out, NOT from norm(x) the MLP will read.
      - AdaRMSNorm replaces RemixedLinear as the modulation mechanism.
    """
    def __init__(self, config, layer_idx):
        super().__init__()
        ctx_dim = config.remix_context_dim
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp  = MLP(config)
        stream_type = getattr(config, 'cclblock_context_stream', 'local')
        if stream_type == 'multiscale':
            self.ctx_stream = MultiScaleContext(config.n_embd, ctx_dim)
            ctx_dim = self.ctx_stream.ctx_dim  # possibly auto-corrected
        elif stream_type == 'ssm':
            self.ctx_stream = ParallelLinearContextStream(config.n_embd, ctx_dim)
        elif stream_type == 'selective':
            self.ctx_stream = SelectiveContextStream(config.n_embd, ctx_dim)
        elif stream_type == 'ema':
            self.ctx_stream = EMAContextStream(config.n_embd, ctx_dim, ema_factor=getattr(config, 'cclblock_ema_factor', 0.99))
        else:
            self.ctx_stream = LocalContextStream(config.n_embd, ctx_dim)
        self.ada_norm_attn = AdaRMSNorm(config.n_embd, ctx_dim)
        self.ada_norm_mlp  = AdaRMSNorm(config.n_embd, ctx_dim)

    def forward(self, x, ve, cos_sin, window_size, kv_cache, prev_ctx=None):
        is_local = isinstance(self.ctx_stream, LocalContextStream)
        
        # Attention with context-conditioned norm
        attn_in  = self.ada_norm_attn(x, None if is_local else prev_ctx)
        attn_out = self.attn(attn_in, ve, cos_sin, window_size, kv_cache)
        x        = x + attn_out

        if is_local:
            ctx = self.ctx_stream(norm(x), prev_ctx)
        else:
            ctx = self.ctx_stream(attn_out, prev_ctx)

        # MLP with freshly updated context-conditioned norm
        x = x + self.mlp(self.ada_norm_mlp(x, ctx))
        
        return x, ctx.detach() if is_local else ctx


# Phase 31: set True only if torch._grouped_mm actually raises, so we stop
# retrying it.  Device/dtype eligibility is checked per call — caching that
# would let one CPU call disable the fused path for CUDA.
_GROUPED_MM_FAILED = False


def _can_grouped_mm(t):
    return (not _GROUPED_MM_FAILED) and hasattr(torch, '_grouped_mm') \
        and t.is_cuda and t.dtype in (torch.bfloat16, torch.float16)


def _grouped_top1_chunk_apply(h_p, idx, T_stack, coef):
    """Phase 31: apply top-1 chunk routing as K dense GEMMs instead of composing
    a per-chunk effective weight matrix.

    The compose formulation (`einsum('bnk,koc->bnoc')`) contracts over all K
    templates whether the routing weights are one-hot or not, so top-1 costs
    exactly what soft routing costs and materializes B*n_chunks matrices of
    (out, basis) in HBM.  With top-1 there are at most K *distinct* effective
    weights, so the right formulation is the standard MoE one: permute chunks
    into template-contiguous order, run one GEMM per template, permute back.
    The template bank is then read K times total rather than B*n_chunks times.

    Args:
        h_p:     (B, n_chunks * chunk, basis) — gated basis, chunk-padded.
        idx:     (B, n_chunks) int64 — selected template per chunk.
        T_stack: (K, out_features, basis) — template bank in h_p's dtype.
        coef:    (B, n_chunks, 1) — per-chunk routing coefficient, or None.

    Returns:
        (B, n_chunks * chunk, out_features)
    """
    B, T_pad, basis_sz = h_p.shape
    n_chunks = idx.shape[1]
    chunk = T_pad // n_chunks
    K, out_features, _ = T_stack.shape
    BN = B * n_chunks

    flat_idx = idx.reshape(-1)                                    # (BN,)
    order = torch.argsort(flat_idx)                               # chunk -> sorted slot
    # Histogram of template usage. NOT torch.bincount: its output length is
    # max(minlength, values.max()+1), which Dynamo classifies as a dynamic shape
    # operator ("output shape depends on input Tensor data") and refuses to put
    # in a graph. That single call broke the graph once per projection per layer,
    # which cost far more than this path saves: measured at d4 it turned a 320 ms
    # step into 650 ms and raised peak memory per sample by 1.73x, enough to push
    # device-batch 128 into OOM. scatter_add_ into a preallocated (K,) buffer is
    # the same histogram with a shape the compiler knows statically.
    counts = torch.zeros(K, dtype=torch.int64, device=idx.device)
    counts.scatter_add_(0, flat_idx, torch.ones_like(flat_idx))
    h_sorted = h_p.reshape(BN, chunk, basis_sz).index_select(0, order)

    out_sorted = None
    if _can_grouped_mm(h_p):
        # One fused grouped GEMM over the K variable-sized token groups. Every
        # shape here is static: only the *values* in offs depend on the routing,
        # and the output is always (BN*chunk, out_features).
        offs = torch.cumsum(counts * chunk, dim=0).to(torch.int32)
        try:
            out_sorted = torch._grouped_mm(
                h_sorted.reshape(BN * chunk, basis_sz),
                T_stack.transpose(1, 2),                          # (K, basis, out)
                offs=offs,
            )
        except (RuntimeError, NotImplementedError, AttributeError):
            # Latch off permanently rather than retrying every call. Under
            # torch.compile this handler is not traced (Dynamo traces the try
            # body only), so it costs nothing on the compiled path; it is here so
            # an unsupported card degrades to the loop below instead of killing
            # a training run mid-step.
            global _GROUPED_MM_FAILED
            _GROUPED_MM_FAILED = True
            out_sorted = None

    if out_sorted is None:
        # Eager-only fallback: K dense GEMMs on contiguous slices. .tolist()
        # forces a sync and a graph break, so _grouped_route declines the whole
        # path whenever _can_grouped_mm is False and this stays untraced. It
        # exists for CPU and fp32, where the tests exercise it.
        h_flat = h_sorted.reshape(BN * chunk, basis_sz)
        segs, start = [], 0
        for k, n_k in enumerate(counts.tolist()):
            if n_k == 0:
                continue
            segs.append(h_flat[start * chunk:(start + n_k) * chunk] @ T_stack[k].t())
            start += n_k
        out_sorted = torch.cat(segs, dim=0)

    out_sorted = out_sorted.reshape(BN, chunk, out_features)
    inv = torch.empty_like(order)
    inv.scatter_(0, order, torch.arange(BN, device=order.device))
    out = out_sorted.index_select(0, inv)                          # (BN, chunk, out)
    if coef is not None:
        out = out * coef.reshape(BN, 1, 1)
    return out.reshape(B, T_pad, out_features)


class RemixedLinear(nn.Module):
    def __init__(self, in_features, out_features, context_dim, basis_size=64, remixed_linear_kwargs=None, scale_basis=True, film_gate=False, routing_scope='per_sequence'):
        super().__init__()
        self._film_gate_flag = film_gate
        # routing_scope: 'per_token' (FFN layers) or 'per_sequence' (attention layers)
        # Per-sequence routing pools h_gated over T before computing routing logits.
        self.routing_scope = routing_scope
        if remixed_linear_kwargs is None:
            remixed_linear_kwargs = dict(use_basis_gate=True, use_output_gate=True, use_context=True)
        # Fix 1B: prevent rank bottleneck — but basis_size should compress relative to
        # the *smaller* of in/out (template_mixing is out_features x basis_size, so using
        # in_features // 4 when in_features >> out_features makes the layer *more* expensive
        # than dense: e.g. c_proj 3072->768 would get basis_size=768 = full-rank output).
        _bsf = int(remixed_linear_kwargs.get('basis_scale_factor', 4)) if remixed_linear_kwargs else 4
        if scale_basis:
            basis_size = max(basis_size, min(in_features, out_features) // max(1, _bsf))
        self.basis_size      = basis_size  # may be updated below for LoKR after b_shrunk calc
        self._gate_stats     = {}          # populated during forward() when self.training
        self.use_basis_gate = remixed_linear_kwargs.get('use_basis_gate', True)
        self.use_output_gate = remixed_linear_kwargs.get('use_output_gate', True)
        self.use_context = remixed_linear_kwargs.get('use_context', True)
        # Design 3: sparse top-k gate. 0=off, N=top-N active basis functions.
        self.sparse_gate_k = remixed_linear_kwargs.get('sparse_gate_k', 0)
        # Design 6: gate temperature. divides sigmoid logits. <1=sharper, >1=softer.
        self.gate_temperature = max(remixed_linear_kwargs.get('gate_temperature', 1.0), 1e-6)
        # Operator-space modulation: none|householder|spectral|ocd
        self.operator_modulation = remixed_linear_kwargs.get('operator_modulation', 'none')
        self._last_orth_loss = None
        self._template_diversity_loss = None
        # Phase 22: MoE-style overparameterized template mixing
        # n_templates > 1 creates K template_mixing matrices with content routing
        self.n_templates = remixed_linear_kwargs.get('n_templates', 1)
        self.template_routing_learned = remixed_linear_kwargs.get('template_routing_learned', False)
        # Phase 23: Tiny Experts mode
        self.tiny_expert = remixed_linear_kwargs.get('tiny_expert', False)
        self.tiny_expert_topk = remixed_linear_kwargs.get('tiny_expert_topk', 16)
        self.use_quantile_route = int(remixed_linear_kwargs.get('use_quantile_route', 0))

        # Phase 23 LoKR: iso-parameter shared-base + low-rank expert adapters
        # Shared base uses shrunk basis (B_shrunk = basis_size - K*rank) for param parity.
        # Low-rank expert adapters: down(in→rank) + up(rank→out), top-k routed from x.
        self.lokr_expert = remixed_linear_kwargs.get('lokr_expert', False)
        self.lokr_n_experts = remixed_linear_kwargs.get('lokr_n_experts', 64)
        self.lokr_topk = remixed_linear_kwargs.get('lokr_topk', 16)
        self.lokr_rank = remixed_linear_kwargs.get('lokr_rank', 4)
        self.lokr_learned = remixed_linear_kwargs.get('lokr_learned', False)

        # Whether routing is handled externally by SharedBlockRouter (injected via forward kwarg)
        self._use_shared_route = remixed_linear_kwargs.get('use_shared_route', False)

        # Phase 28: FLOPs-efficient routing
        # 28D: amortize template routing over N-token chunks (0 = per-token, N = chunk size)
        self.chunk_routing_size = int(remixed_linear_kwargs.get('chunk_routing_size', 0))
        # Phase 31: how the chunk-routed product is evaluated.
        #   'compose' — build W_eff = sum_k a_k T_k per chunk, then contract (legacy).
        #   'grouped' — top-1 only: permute chunks by template, one GEMM per template.
        # Numerically equivalent for topk=1; 'compose' is kept as the default so
        # existing runs and checkpoints are unaffected.
        self.chunk_route_impl = remixed_linear_kwargs.get('chunk_route_impl', 'compose')
        # Phase 31: top-1 routing coefficient.
        #   'ones'   — legacy. softmax over a single selected logit is *exactly* 1.0,
        #              so the router receives zero gradient: top-1 trains with a
        #              frozen random routing projection.
        #   'switch' — coefficient is the full-softmax probability of the selected
        #              template (Switch-Transformer style), which is differentiable.
        self.top1_gate = remixed_linear_kwargs.get('top1_gate', 'ones')
        # Phase 31: low-rank template deltas. 0 = off (full (K, out, basis) bank).
        # r > 0 replaces the bank with a shared base plus K rank-r deltas, which
        # removes the per-chunk (out, basis) materialization from soft routing.
        self.template_delta_rank = int(remixed_linear_kwargs.get('template_delta_rank', 0))
        # Phase 31: which of the two factors carries the routing.
        #   'output' — route W_m (out, basis); W_b is a single shared matrix. Legacy.
        #   'basis'  — route W_b (basis, in);  W_m is a single shared matrix.
        #   'narrow' — pick whichever routed factor has fewer elements, per
        #              projection.  basis*in < out*basis  <=>  in < out, so this
        #              resolves to 'basis' for expanding projections (c_fc) and
        #              'output' for contracting ones (c_proj) and square ones.
        # The point is that the per-chunk materialized matrix is the *routed*
        # factor, so routing the narrow side shrinks it without touching rank.
        self._route_side_req = remixed_linear_kwargs.get('route_side', 'output')
        # Phase 31: drop W_b entirely — h = LN(x), then a routed (out, in) template.
        # Exactly one matmul per projection, i.e. dense FLOPs, at the cost of a
        # larger routed factor for contracting projections.
        self.drop_basis_proj = bool(remixed_linear_kwargs.get('drop_basis_proj', False))
        # 28E/F: use global template bank (injected by GPT after construction) instead of per-layer template_bank
        # When True AND self._global_bank is set, forward uses global bank and skips local template_bank.
        _global_bank_mode = remixed_linear_kwargs.get('global_bank_mode', 'none')
        _layer_role = remixed_linear_kwargs.get('layer_role', 'ffn')
        self._use_global_bank = (
            _global_bank_mode == 'all' or
            (_global_bank_mode == 'ffn' and _layer_role == 'ffn')
        )
        self._global_bank = None          # injected by GPT.__init__ post-construction
        self._global_bank_key = 'fc' if (out_features > in_features) else 'proj'
        self._layer_idx_for_bank = 0      # injected by GPT.__init__ post-construction

        # Initialise all optional attrs to None so non_gate_parameters() /
        # gate_parameters() can always use `is not None` checks on every code path.
        self.template_bank = None
        self.template_mixing = None
        self.route_affine = False      # affine-hull routing; set where template_route is built
        self.route_affine_scale = None
        self.template_base = None      # Phase 31 delta mode: shared base (out, basis)
        self.template_delta_v = None   # Phase 31 delta mode: (basis, K*r)
        self.template_delta_u = None   # Phase 31 delta mode: (K*r, out)
        self.expert_up_w = None    # 3D stacked: (K, expert_dim, basis_size)
        self.expert_down_w = None  # 3D stacked: (K, out_features, expert_dim)
        self.template_route = None
        self.lokr_down_w = None
        self.lokr_up_w = None
        self.lokr_route_proj = None

        if self.lokr_expert:
            # Iso-param: B_shrunk = basis_size - K*rank
            # Ensures (B_shrunk + K*rank)*(in+out) = basis_size*(in+out)
            K, R = self.lokr_n_experts, self.lokr_rank
            b_shrunk = basis_size - K * R
            if b_shrunk <= 0:
                # Auto-clamp: reduce rank so at least 1 dim remains for shared base
                import warnings
                R_old = R
                R = max(1, (basis_size - 1) // K)
                b_shrunk = basis_size - K * R
                self.lokr_rank = R
                warnings.warn(
                    f"LoKR: auto-clamped rank {R_old}→{R} "
                    f"(basis={basis_size}, K={K})"
                )
            assert b_shrunk > 0, (
                f"LoKR: basis_size ({basis_size}) too small even after rank clamp "
                f"(K={K}, rank={R}). Reduce n_experts or increase basis_size."
            )
            # Override basis_size for the shared-base path
            basis_size = b_shrunk
            self.basis_size = b_shrunk   # keep self.basis_size consistent with actual self.basis output

        # ── Phase 31: resolve route_side / drop_basis_proj ───────────────────
        # Both are only meaningful for the legacy multi-template bank; the tiny-expert,
        # LoKR, global-bank and delta paths have no separate W_b/W_m pair to swap.
        _legacy_bank = (self.n_templates > 1 and not self.tiny_expert
                        and not self.lokr_expert and not self._use_global_bank
                        and self.template_delta_rank == 0)
        # Gate drop_basis_proj on in_features <= basis_size.  Removing W_b forces the
        # bank's contraction dim from basis_size to in_features, so for a contracting
        # projection (c_proj: in = 4*basis) every one of the K templates grows 4x —
        # costing K*(in-basis)*out extra parameters and quadrupling the per-chunk
        # materialized matrix, to save one matmul.  Measured at d8 that made c_proj
        # 0.45x fwd / 0.26x f+b and added ~5.2M params, swamping the gain elsewhere.
        # Where in == basis (c_fc, attention) the bank is unchanged and W_b is pure
        # overhead, so the drop is a clean win.
        # Note: if in < basis strictly, this also shrinks the contraction dim — with
        # scale_basis and remix_basis_size = model_dim, in <= basis only ever holds
        # as in == basis, so no rank is lost in practice.
        self.drop_basis_proj = (self.drop_basis_proj and _legacy_bank
                                and in_features <= basis_size)
        if self.drop_basis_proj:
            # No W_b at all: LN acts on x directly, so the bank's contraction dim
            # is in_features and the templates are (out, in).
            basis_size = in_features
            self.basis_size = in_features
            self.route_side = 'output'
        elif _legacy_bank and self._route_side_req in ('basis', 'narrow'):
            self.route_side = ('basis' if (self._route_side_req == 'basis'
                                           or in_features < out_features)
                               else 'output')
        else:
            # Either not a legacy bank, or route_side left at 'output'.  A projection
            # whose drop was gated out lands here and stays exactly legacy.
            self.route_side = 'output'
        # Phase 31: the materialized per-chunk matrix on the basis side is
        # (basis, in) — its size does NOT depend on K.  So a basis-routed
        # projection can afford out/basis times more templates at the *same*
        # compose cost and the same materialization, which is how you keep
        # parameter count matched to the output-side config instead of losing
        # K*(out-basis)*basis parameters.
        #   N > 0  : use exactly N templates here
        #   N == -1: auto — scale to n_templates * (out // basis), i.e. iso-param
        if self.route_side == 'basis':
            _bst = int(remixed_linear_kwargs.get('basis_side_templates', 0))
            if _bst == -1:
                self.n_templates = self.n_templates * max(1, out_features // basis_size)
            elif _bst > 0:
                self.n_templates = _bst
        # W_b exists only when the shared input map is still a separate matrix.
        self._has_basis_proj = not (self.drop_basis_proj or self.route_side == 'basis')

        self.basis = Linear(in_features, basis_size, bias=False) if self._has_basis_proj else None
        if self.lokr_expert:
            # Low-rank expert adapters: stacked tensors for efficient batched matmul
            # lokr_down_w: (K, rank, in_features)  — projects x to rank-r subspace
            # lokr_up_w:   (K, out_features, rank) — zero-init for identity start
            K, R = self.lokr_n_experts, self.lokr_rank
            self.lokr_down_w = nn.Parameter(torch.empty(K, R, in_features))
            self.lokr_up_w   = nn.Parameter(torch.zeros(K, out_features, R))
            # Router: frozen (Parameter) or learned (weight of a Linear)
            if self.lokr_learned:
                self.lokr_route_proj = nn.Parameter(torch.empty(K, in_features))
            else:
                self.lokr_route_proj = nn.Parameter(torch.empty(K, in_features),
                                                     requires_grad=False)
            # template_mixing serves as the shrunk shared-base output matrix
            self.template_mixing = nn.Parameter(torch.empty(out_features, basis_size))
            # Store shrunk basis_size for diagnostics
            self._lokr_basis_size = basis_size
        elif self.tiny_expert and self.n_templates > 1:
            # Phase 23 Tiny Experts — vectorized 3D stacked parameters.
            # expert_dim = max(basis_size // topk, 32) — floor of 32 prevents impossibly narrow
            # bottlenecks (e.g. basis_size=128, topk=16 → 8 dims → near-zero relu² outputs).
            topk = self.tiny_expert_topk
            assert topk > 0, "tiny_expert_topk must be > 0"
            raw_expert_dim = basis_size // topk
            self.expert_dim = raw_expert_dim
            K = self.n_templates
            self.expert_up_w   = nn.Parameter(torch.empty(K, self.expert_dim, basis_size))
            self.expert_down_w = nn.Parameter(torch.empty(K, out_features, self.expert_dim))
            self.template_bank = None
            self.template_mixing = None
            # Routing: quantile-balanced or simple learned/frozen softmax
            if not self._use_shared_route:
                if self.use_quantile_route == 2:
                    self._qrouter = QuantileCrossAttentionRouter(
                        in_features, K, topk)
                    self.template_route = None
                elif self.use_quantile_route == 1:
                    self._qrouter = QuantileBalancedRouter(
                        in_features, K, topk, learned=self.template_routing_learned)
                    self.template_route = None  # handled by _qrouter
                else:
                    # Affine-hull routing (see _affine_weights): drops the
                    # simplex constraint that makes the softmax router collapse.
                    self.route_affine = bool(remixed_linear_kwargs.get('route_affine', 0))
                    # Only allocated when the mode is on: an unused parameter in
                    # every model would sit in the optimizer with no gradient.
                    if self.route_affine:
                        self.route_affine_scale = nn.Parameter(torch.ones(1) / max(1, self.n_templates))
                    route_init = torch.randn(in_features, K) / (in_features ** 0.5)
                    if self.template_routing_learned:
                        self.template_route = nn.Parameter(route_init)
                    else:
                        if hasattr(self, 'template_route') and not isinstance(self.template_route, (nn.Parameter, torch.Tensor)):
                            del self.template_route
                        self.register_buffer('template_route', route_init)
                    self._qrouter = None
            else:
                self._qrouter = None
            # Diagnostics
            self.register_buffer('_template_entropy_buf', torch.zeros(1), persistent=False)
            self.register_buffer('_template_weights_buf', torch.zeros(self.lokr_n_experts), persistent=False)
        elif self.n_templates > 1:
            self.template_topk = int(remixed_linear_kwargs.get('template_topk', 0))
            if self._use_global_bank:
                # Phase 28E/F: templates live in GlobalTemplateBank — no local allocation.
                # _global_bank reference is injected by GPT.__init__ after construction.
                # template_bank, template_route, and template_mixing stay None.
                self.template_bank = None
                self.template_mixing = None
                self.template_route = None
                self.register_buffer('_template_entropy_buf', torch.zeros(1), persistent=False)
                self.register_buffer('_template_weights_buf', torch.zeros(self.n_templates), persistent=False)
            elif self.template_delta_rank > 0:
                # Phase 31: shared base + K rank-r deltas.  T_k = T_0 + U_k V_k^T, so
                #     y = h T_0^T + sum_k a_k (h V_k) U_k^T
                # Soft mixing over all K templates becomes three dense GEMMs and never
                # forms an (out, basis) matrix, so nothing is written to HBM per chunk.
                # Extra cost over dense is K*r*(in+out) per token instead of K*out*basis
                # per chunk, which also makes per-token routing affordable again.
                _r = self.template_delta_rank
                self.template_bank = None
                self.template_base = nn.Parameter(torch.randn(out_features, basis_size))
                self.template_delta_v = nn.Parameter(
                    torch.randn(basis_size, self.n_templates * _r) / (basis_size ** 0.5)
                )
                # Zero-init U so every delta starts at exactly zero: the layer begins
                # life as the K=1 factorized layer, matching the identity-preserving
                # init used for output_gate_basis and lokr_up_w.
                self.template_delta_u = nn.Parameter(
                    torch.zeros(self.n_templates * _r, out_features)
                )
                if self.use_quantile_route == 2:
                    self._qrouter = QuantileCrossAttentionRouter(in_features, self.n_templates, self.template_topk)
                    self.template_route = None
                elif self.use_quantile_route == 1:
                    self._qrouter = QuantileBalancedRouter(in_features, self.n_templates, self.template_topk, learned=self.template_routing_learned)
                    self.template_route = None
                else:
                    # Affine-hull routing (see _affine_weights): drops the
                    # simplex constraint that makes the softmax router collapse.
                    self.route_affine = bool(remixed_linear_kwargs.get('route_affine', 0))
                    # Only allocated when the mode is on: an unused parameter in
                    # every model would sit in the optimizer with no gradient.
                    if self.route_affine:
                        self.route_affine_scale = nn.Parameter(torch.ones(1) / max(1, self.n_templates))
                    route_init = torch.randn(in_features, self.n_templates) / (in_features ** 0.5)
                    if self.template_routing_learned:
                        self.template_route = nn.Parameter(route_init)
                    else:
                        self.register_buffer('template_route', route_init)
                    self._qrouter = None
                self.register_buffer('_template_entropy_buf', torch.zeros(1), persistent=False)
                self.register_buffer('_template_weights_buf', torch.zeros(self.n_templates), persistent=False)
            elif self.route_side == 'basis':
                # Phase 31: the routed factor is W_b. Bank is (K, basis, in) and the
                # output map W_m is a single shared (out, basis) matrix. For an
                # expanding projection this shrinks the per-chunk materialized matrix
                # by out/basis (4x at c_fc) and turns the *large* basis→out matmul
                # into a weight-shared dense GEMM, without reducing any rank.
                self.template_bank = nn.Parameter(
                    torch.randn(self.n_templates, basis_size, in_features)
                )
                self.template_mixing = nn.Parameter(torch.randn(out_features, basis_size))
                if self.use_quantile_route == 2:
                    self._qrouter = QuantileCrossAttentionRouter(in_features, self.n_templates, getattr(self, 'template_topk', 0))
                    self.template_route = None
                elif self.use_quantile_route == 1:
                    self._qrouter = QuantileBalancedRouter(in_features, self.n_templates, getattr(self, 'template_topk', 0), learned=self.template_routing_learned)
                    self.template_route = None
                else:
                    # Affine-hull routing (see _affine_weights): drops the
                    # simplex constraint that makes the softmax router collapse.
                    self.route_affine = bool(remixed_linear_kwargs.get('route_affine', 0))
                    # Only allocated when the mode is on: an unused parameter in
                    # every model would sit in the optimizer with no gradient.
                    if self.route_affine:
                        self.route_affine_scale = nn.Parameter(torch.ones(1) / max(1, self.n_templates))
                    route_init = torch.randn(in_features, self.n_templates) / (in_features ** 0.5)
                    if self.template_routing_learned:
                        self.template_route = nn.Parameter(route_init)
                    else:
                        self.register_buffer('template_route', route_init)
                    self._qrouter = None
                self.register_buffer('_template_entropy_buf', torch.zeros(1), persistent=False)
                self.register_buffer('_template_weights_buf', torch.zeros(self.n_templates), persistent=False)
            else:
                # Legacy: K separate template_mixing matrices: each (out_features, basis_size)
                # With drop_basis_proj this is (out, in) — LN acts on x directly.
                self.template_bank = nn.Parameter(
                    torch.randn(self.n_templates, out_features, basis_size)
                )
                self.template_mixing = None  # use template_bank instead
                # Quantile routing or standard routing for template selection
                if self.use_quantile_route == 2:
                    self._qrouter = QuantileCrossAttentionRouter(in_features, self.n_templates, getattr(self, 'template_topk', 0))
                    self.template_route = None
                elif self.use_quantile_route == 1:
                    self._qrouter = QuantileBalancedRouter(in_features, self.n_templates, getattr(self, 'template_topk', 0), learned=self.template_routing_learned)
                    self.template_route = None
                else:
                    # Affine-hull routing (see _affine_weights): drops the
                    # simplex constraint that makes the softmax router collapse.
                    self.route_affine = bool(remixed_linear_kwargs.get('route_affine', 0))
                    # Only allocated when the mode is on: an unused parameter in
                    # every model would sit in the optimizer with no gradient.
                    if self.route_affine:
                        self.route_affine_scale = nn.Parameter(torch.ones(1) / max(1, self.n_templates))
                    route_init = torch.randn(in_features, self.n_templates) / (in_features ** 0.5)
                    if self.template_routing_learned:
                        self.template_route = nn.Parameter(route_init)
                    else:
                        if hasattr(self, 'template_route') and not isinstance(self.template_route, (nn.Parameter, torch.Tensor)):
                            del self.template_route
                        self.register_buffer('template_route', route_init)
                    self._qrouter = None
                # Diagnostics
                self.register_buffer('_template_entropy_buf', torch.zeros(1), persistent=False)
                self.register_buffer('_template_weights_buf', torch.zeros(self.n_templates), persistent=False)
        else:
            self.template_mixing = nn.Parameter(torch.randn(out_features, basis_size))
        # Phase 30: optionally disable intermediate LayerNorm for ablation
        if remixed_linear_kwargs.get('disable_ln_basis', False):
            self.ln_basis = nn.Identity()
        else:
            self.ln_basis = nn.LayerNorm(basis_size)

        self.bias = nn.Parameter(torch.zeros(out_features))

        # Phase 29: per-template gradient equalization (MST-inspired).
        # Normalizes each template's gradient to the mean norm across all K
        # templates, preventing the dominant-template feedback loop where the
        # most-routed template receives disproportionately more gradient and
        # reinforces its dominance (same pathology as MST sub-collapse).
        if self.template_bank is not None and remixed_linear_kwargs.get('grad_equalize', False):
            K = self.template_bank.shape[0]
            def _template_grad_equalize(grad, _K=K):
                # grad: (K, out, basis) or (K, basis, in) for route_side='basis'
                per_tmpl_norms = grad.norm(dim=tuple(range(1, grad.ndim)), keepdim=True)  # (K, 1, 1)
                mean_norm = per_tmpl_norms.mean()
                scale = mean_norm / (per_tmpl_norms + 1e-8)
                return grad * scale
            self.template_bank.register_hook(_template_grad_equalize)

        if self.use_context:
            # --- Gate network design ---
            #
            # PROBLEM AT LONG SEQUENCES:
            # output_modulator = Linear(ctx_dim, out_features) has ctx_dim × out_features
            # parameters (e.g. 128 × 3072 = 393K for c_fc). At T=2048, positions 0-~100
            # have weak causal context, so their context_state is near-zero. Those positions
            # produce near-random output gates that MULTIPLY the pre_output, corrupting
            # gradients for basis and template_mixing everywhere. The dense baseline has
            # no such corruption — this is why RemixedLinear degrades at long T.
            #
            # FIX 1: Low-rank output gate.
            # Instead of predicting out_features independent gate values per position,
            # predict r << out_features coefficients and apply them to r learned gate
            # basis vectors: gate = sigmoid(coeffs @ gate_basis), where
            #   coeffs  = Linear(ctx_dim, r)        — small, stable to train
            #   gate_basis = Parameter(r, out_features)  — learned gate directions
            # This reduces gate noise by forcing it onto a low-dimensional manifold.
            # Gradient of gate_basis weights is summed over T positions, reducing variance.
            # Gradient of coeffs weights is summed over out_features directions per position,
            # amplifying signal by r.
            #
            # FIX 2: Centered gate activation for output gate.
            # sigmoid starts at 0.88 (with bias=2.0) but the identity operation is 1.0.
            # Using gate = 1 + tanh(scale * coeffs @ gate_basis) centers the gate around
            # 1.0 and allows it to range in [0, 2], providing both attenuation AND
            # amplification. Gradient at init = sech²(0) = 1.0 vs sigmoid'(2.0) ≈ 0.10.
            # 10× better gradient flow for the output gate at initialization.
            basis_hidden = max(context_dim // 2, min(basis_size, context_dim * 2))
            film_gate = getattr(self, '_film_gate_flag', False)
            _gate_out_size = 2 * basis_size if self.use_basis_gate and film_gate else basis_size
            self.basis_gate_mode = remixed_linear_kwargs.get('basis_gate_mode', 'mlp')
            # Initialise all gate module attrs so non_gate_parameters/gate_parameters can use
            # `is not None` checks unconditionally on every code path.
            self.basis_modulator = None
            self.basis_gate_content = None
            self.basis_gate_context = None
            if self.use_basis_gate:
                if self.basis_gate_mode == 'mlp':
                    self.basis_modulator = nn.Sequential(
                        Linear(context_dim, basis_hidden, bias=True),
                        nn.GELU(),
                        Linear(basis_hidden, _gate_out_size, bias=True),
                    )
                    nn.init.zeros_(self.basis_modulator[-1].weight)
                    if self.basis_modulator[-1].bias is not None:
                        nn.init.zeros_(self.basis_modulator[-1].bias)
                elif self.basis_gate_mode == 'linear':
                    # Single linear gate: half the cost of MLP gate
                    self.basis_modulator = Linear(context_dim, _gate_out_size, bias=True)
                    nn.init.zeros_(self.basis_modulator.weight)
                    nn.init.zeros_(self.basis_modulator.bias)
                elif self.basis_gate_mode == 'random':
                    # Random-init gate: same architecture as linear but weight/bias
                    # are kept at their default normal init (std≈1/sqrt(context_dim)).
                    # Gate values at step 0 are spread across (0,1) rather than
                    # collapsing to 0.5 (linear zero-init) or 1.0 (centered).
                    # Empirical test of whether random gate values at init are
                    # beneficial or harmful vs stable initializations.
                    self.basis_modulator = Linear(context_dim, _gate_out_size, bias=True)
                    # Do NOT zero-init — leave PyTorch default kaiming_uniform init
                elif self.basis_gate_mode == 'centered':
                    # Centered gate: 1 + tanh(s * W_gate * ctx)
                    # Starts at 1.0 (full passthrough) instead of sigmoid's 0.5.
                    # Structural weights (W_b, W_m) see clean gradients from step 0;
                    # gate learns to symmetrically amplify (>1) or suppress (<1) channels.
                    self.basis_modulator = Linear(context_dim, _gate_out_size, bias=True)
                    nn.init.zeros_(self.basis_modulator.weight)
                    nn.init.zeros_(self.basis_modulator.bias)
                    # Learnable scale — init small so gate stays near 1.0 early in training
                    self.centered_gate_scale = nn.Parameter(torch.ones(1) * 0.1)
                elif self.basis_gate_mode == 'attn':
                    # Bilinear attention gate: gate = σ(W_content·h ⊙ W_context·ctx)
                    # Jointly conditioned on input content AND context; cheaper than MLP.
                    self.basis_gate_content = Linear(basis_size, _gate_out_size, bias=False)
                    self.basis_gate_context = Linear(context_dim, _gate_out_size, bias=True)
                    nn.init.normal_(self.basis_gate_content.weight, std=0.02)
                    nn.init.zeros_(self.basis_gate_context.weight)
                    nn.init.zeros_(self.basis_gate_context.bias)
                elif self.basis_gate_mode == 'lowrank':
                    # Low-rank basis gate (symmetric to output gate, negligible FLOPs)
                    # gate = 1 + tanh(scale * (W_g * ctx) @ gate_vecs)
                    r_bg = int(remixed_linear_kwargs.get('basis_gate_rank', 8))
                    self.basis_gate_coeffs  = Linear(context_dim, r_bg, bias=True)
                    self.basis_gate_vectors = nn.Parameter(torch.zeros(r_bg, _gate_out_size))
                    self.basis_gate_lr_scale = nn.Parameter(torch.ones(1) * 0.1)
                    nn.init.zeros_(self.basis_gate_coeffs.weight)
                    nn.init.zeros_(self.basis_gate_coeffs.bias)
                # else 'none': gate_basis = ones in forward, no module
            # Low-rank output gate: ctx_dim → r coefficients, r basis vectors → out_features
            # r is a small constant (default 8), making this O(ctx_dim*r + r*out_features)
            # rather than O(ctx_dim*out_features).
            r = remixed_linear_kwargs.get('output_gate_rank', 8)
            self.output_gate_coeffs = Linear(context_dim, r, bias=True)
            # Small random init (not zeros!) — zero basis creates a zero-gradient
            # trap for output_gate_scale: d tanh(s*0)/ds = 0 × sech²(0) = 0.
            self.output_gate_basis  = nn.Parameter(torch.randn(r, out_features) * 0.01)
            self.output_gate_scale  = nn.Parameter(torch.ones(1) * 0.1)  # learnable scale, starts small
            if self.operator_modulation == 'householder':
                self.operator_householder = Linear(context_dim, basis_size, bias=True)
                nn.init.zeros_(self.operator_householder.weight)
                nn.init.zeros_(self.operator_householder.bias)
            elif self.operator_modulation == 'spectral':
                self.operator_spectral = Linear(context_dim, basis_size, bias=True)
                nn.init.zeros_(self.operator_spectral.weight)
                nn.init.zeros_(self.operator_spectral.bias)
            elif self.operator_modulation == 'ocd':
                # Low-rank dynamic delta: sum_r coeff_r * (out_r ⊗ in_r)
                ocd_rank = remixed_linear_kwargs.get('ocd_rank', 16)
                self.ocd_coeffs = Linear(context_dim, ocd_rank, bias=True)
                self.ocd_in = nn.Parameter(torch.zeros(ocd_rank, basis_size))
                self.ocd_out = nn.Parameter(torch.zeros(ocd_rank, out_features))
                self.ocd_scale = nn.Parameter(torch.tensor(0.1))
                nn.init.zeros_(self.ocd_coeffs.weight)
                nn.init.zeros_(self.ocd_coeffs.bias)
            elif self.operator_modulation == 'lie':
                m = int(remixed_linear_kwargs.get('lie_generators', 4))
                self.lie_coeffs = Linear(context_dim, m, bias=True)
                self.lie_generators = nn.Parameter(torch.zeros(m, basis_size, basis_size))
                nn.init.zeros_(self.lie_coeffs.weight)
                nn.init.zeros_(self.lie_coeffs.bias)
            elif self.operator_modulation == 'polynomial':
                p = int(remixed_linear_kwargs.get('poly_order', 2))
                self.poly_order = max(1, p)
                self.poly_coeffs = Linear(context_dim, self.poly_order + 1, bias=True)
                self.poly_A = nn.Parameter(torch.eye(basis_size))
                nn.init.zeros_(self.poly_coeffs.weight)
                nn.init.zeros_(self.poly_coeffs.bias)
                with torch.no_grad():
                    self.poly_coeffs.bias[0] = 1.0
            elif self.operator_modulation == 'grassmann':
                k = int(remixed_linear_kwargs.get('grassmann_bank_size', 4))
                self.grassmann_alpha = Linear(context_dim, k, bias=True)
                self.grassmann_bank = nn.Parameter(torch.ones(k, basis_size))
                nn.init.zeros_(self.grassmann_alpha.weight)
                nn.init.zeros_(self.grassmann_alpha.bias)

    def gate_parameters(self):
        """Yield context-gate-specific parameters (basis_modulator, output_gate_*).
        These are routed to a lower-LR optimizer group to reduce gradient noise at long T."""
        if self.use_context:
            if self.basis_modulator is not None:
                yield from self.basis_modulator.parameters()
            if self.basis_gate_content is not None:
                yield from self.basis_gate_content.parameters()
            if self.basis_gate_context is not None:
                yield from self.basis_gate_context.parameters()
            yield self.output_gate_coeffs.weight
            if self.output_gate_coeffs.bias is not None:
                yield self.output_gate_coeffs.bias
            yield self.output_gate_basis
            yield self.output_gate_scale
            if hasattr(self, 'centered_gate_scale'):
                yield self.centered_gate_scale
            if hasattr(self, "operator_householder"):
                yield from self.operator_householder.parameters()
            if hasattr(self, "operator_spectral"):
                yield from self.operator_spectral.parameters()
            if hasattr(self, "ocd_coeffs"):
                yield from self.ocd_coeffs.parameters()
                yield self.ocd_scale
            if hasattr(self, "lie_coeffs"):
                yield from self.lie_coeffs.parameters()
            if hasattr(self, "poly_coeffs"):
                yield from self.poly_coeffs.parameters()
            if hasattr(self, "grassmann_alpha"):
                yield from self.grassmann_alpha.parameters()
            # lowrank basis gate — basis_gate_coeffs (Linear) + basis_gate_vectors + basis_gate_lr_scale
            if self.basis_gate_mode == 'lowrank' and hasattr(self, 'basis_gate_coeffs') and self.basis_gate_coeffs is not None:
                yield from self.basis_gate_coeffs.parameters()
                yield self.basis_gate_vectors
                yield self.basis_gate_lr_scale
        # Routing parameters — yielded regardless of use_context, since routing
        # is orthogonal to the context/gate system.  When use_context=False these
        # would otherwise fall into the catch-all optimizer group with the wrong LR.
        if self.template_route is not None and isinstance(self.template_route, nn.Parameter):
            yield self.template_route
        if getattr(self, '_qrouter', None) is not None:
            yield from self._qrouter.gate_parameters()
        # LoKR route projection (only when learned — i.e., requires_grad=True)
        if self.lokr_route_proj is not None and self.lokr_route_proj.requires_grad:
            yield self.lokr_route_proj
        if self.route_affine_scale is not None:
            yield self.route_affine_scale

    def _affine_weights(self, sig, dtype):
        """Affine-hull routing: alpha = 1/K + s*(beta - mean(beta)), beta unconstrained.

        The softmax puts the routing weights on the SIMPLEX, i.e. the reachable
        operators are the convex hull of the K templates. Two things follow, and
        both are visible in scripts/paper_template_analysis.py on a trained d12
        checkpoint (59/72 modules at K_eff < 1.05, usage CV = sqrt(K-1) exactly,
        which is the analytic maximum):

          Gradient starvation.  dL/dz_k = alpha_k * (g_k - sum_j alpha_j g_j).
          The alpha_k prefactor means a template's gradient is proportional to
          how much it is already being used, so a losing template cannot recover.
          Rich-get-richer, with one-hot as the fixed point.

          Norm coupling.  For templates with mean pairwise cosine c, a uniform
          mixture has Frobenius norm sqrt(1/K + (K-1)c/K) relative to a single
          template, which at the measured cosines (0.02 to 0.23) is 0.40 to 0.55.
          On the simplex the ONLY way to increase operator norm is to concentrate,
          so there is a large task-independent gradient toward a vertex. That
          predicts more orthogonal banks collapse harder, and they do:
          Spearman(bank cos, K_eff) = 0.943 across the six projections.

        Dropping nonnegativity moves the reachable set from the convex hull to the
        AFFINE hull: still sum(alpha) = 1, so the mixture is still a proper
        combination and nothing about operator scale changes, but the weights may
        now be negative (subtracting a template becomes available) and, crucially,
        beta can grow in any direction. That decouples "make the operator bigger"
        from "concentrate on one template", and it removes the alpha_k prefactor
        entirely: dL/dbeta_k is the full projection of the weight gradient onto
        template k, independent per k, at every step, forever.

        To be precise about what this does and does not fix: it does not make the
        mean template T-bar any larger (T-bar is still the shrunken average, norm
        ~1/sqrt(K) of one template). What it removes is the coupling, since norm
        can now be recovered by scaling beta instead of by collapsing.

        route_affine_scale is learned and starts at 1/K, which keeps the initial
        spread of alpha comparable to a softmax's and avoids a dead start where
        all K templates would receive identical gradient.
        """
        sig = norm(sig)   # the anchor is a raw residual stream whose scale grows
                          # with depth; softmax tolerates that, an affine map does not
        beta = torch.einsum('bnc,ck->bnk', sig.float(), self.template_route.float())
        beta = beta - beta.mean(dim=-1, keepdim=True)
        s = self.route_affine_scale.float()
        return (1.0 / self.n_templates + s * beta).to(dtype)

    def _template_weights(self, sig, dtype):
        """Routing weights from a float32 signal (B, N, in_features) → (B, N, K).

        Mirrors the topk/softmax logic of the compose paths; used by the Phase 31
        delta path, where the same code serves chunk-level and per-token routing.
        """
        K = self.n_templates
        if getattr(self, '_qrouter', None) is not None:
            return self._qrouter(sig).to(dtype)
        if self.template_route is None:
            return torch.full(sig.shape[:-1] + (K,), 1.0 / K,
                              device=sig.device, dtype=dtype)
        if getattr(self, 'route_affine', False):
            return self._affine_weights(sig, dtype)
        logits = torch.einsum('bnc,ck->bnk', sig, self.template_route.float())
        topk = getattr(self, 'template_topk', 0)
        if 0 < topk < K:
            vals, idx = logits.topk(topk, dim=-1)
            if topk == 1 and self.top1_gate == 'switch':
                # Coefficient from the full softmax, so the router gets a gradient.
                probs = F.softmax(logits, dim=-1)
                return (F.one_hot(idx.squeeze(-1), K).to(dtype)
                        * probs.gather(-1, idx).to(dtype))
            one_hot = F.one_hot(idx, K).to(dtype)
            return (one_hot * F.softmax(vals, dim=-1).unsqueeze(-1).to(dtype)).sum(dim=-2)
        return F.softmax(logits, dim=-1).to(dtype)

    def _effective_topk(self):
        """How many templates one routing decision actually selects.

        The quantile routers resolve their own topk (0 means all K), so asking
        `self.template_topk` is not enough to know whether routing is hard top-1.
        """
        K = self.n_templates
        qr = getattr(self, '_qrouter', None)
        if qr is not None:
            return _resolve_topk(getattr(qr, 'topk', 0), K)
        topk = getattr(self, 'template_topk', 0)
        return K if (topk <= 0 or topk >= K) else topk

    def _grouped_route(self, sig, dtype):
        """(sel_idx, coef, weights_all) for the grouped top-1 path, or None.

        None means the caller must fall back to compose, so this is the single
        place that decides whether the fast path applies. It applies only when
        routing selects exactly one template, in which case the compose einsum
        would be contracting against a one-hot vector: same result, but it
        materializes B*n_chunks weight matrices to get there.

        The quantile routers return a normalised weight vector rather than
        logits, so when their resolved topk is 1 that vector is already one-hot
        and selecting from it is a topk(1), not a re-derivation. `coef` is the
        gathered weight, numerically exactly 1.0. It is kept as a tensor rather
        than skipped so the autograd edge to route_proj survives: DDP raises on a
        parameter that requires grad and receives None. The gradient through it
        is zero either way, exactly as in the compose path, so this is a
        throughput change and nothing else.
        """
        if self.chunk_route_impl != 'grouped' or self._effective_topk() != 1:
            return None
        K = self.n_templates
        qr = getattr(self, '_qrouter', None)
        if qr is not None:
            weights_all = qr(sig).to(dtype)
            sel_vals, sel_idx = weights_all.topk(1, dim=-1)
            return sel_idx.squeeze(-1), sel_vals.to(dtype), weights_all
        if getattr(self, 'template_route', None) is None:
            return None
        logits_all = torch.einsum('bnc,ck->bnk', sig, self.template_route.float())
        # topk rather than argmax so tie-breaking matches the compose path exactly
        # and the two implementations stay bit-identical.
        sel_vals, sel_idx = logits_all.topk(1, dim=-1)
        sel_idx = sel_idx.squeeze(-1)
        if self.top1_gate == 'switch':
            # Coefficient from the full softmax, so the router gets a gradient.
            probs_all = F.softmax(logits_all, dim=-1)
            return (sel_idx, probs_all.gather(-1, sel_idx.unsqueeze(-1)).to(dtype),
                    probs_all.to(dtype))
        # Legacy coefficient: softmax over a single logit is exactly 1.0.
        return (sel_idx, F.softmax(sel_vals.float(), dim=-1).to(dtype),
                F.one_hot(sel_idx, K).to(dtype))

    def _routed_basis_proj(self, x, dtype):
        """Phase 31 route_side='basis': h_pre = x @ W_b_eff(chunk)^T.

        Identical machinery to the output-side chunk routing, applied to the *input*
        map instead: the bank is (K, basis, in) and the result is (B, T, basis),
        which then flows through ln_basis and the gates exactly as before.
        """
        B, T_len, C = x.shape
        K = self.n_templates
        B_stack = self.template_bank.to(dtype=dtype)          # (K, basis, in)
        chunk = self.chunk_routing_size

        if chunk <= 0:
            # Per-token routing. Apply-first (all K, then mix) is the only option
            # here without materializing a matrix per token.
            weights_all = self._template_weights(x.float(), dtype)          # (B,T,K)
            all_h = torch.einsum('btc,kdc->btkd', x, B_stack)               # (B,T,K,basis)
            h_pre = (all_h * weights_all.unsqueeze(-1)).sum(dim=2)
        else:
            n_chunks = (T_len + chunk - 1) // chunk
            pad = n_chunks * chunk - T_len
            x_p = F.pad(x, (0, 0, 0, pad)) if pad > 0 else x
            sig = x_p.reshape(B, n_chunks, chunk, C)[:, :, 0, :].float()
            topk = getattr(self, 'template_topk', 0)
            _g = self._grouped_route(sig, dtype)
            if _g is not None:
                sel_idx, coef, weights_all = _g
                h_p = _grouped_top1_chunk_apply(x_p, sel_idx, B_stack, coef)
                h_pre = h_p[:, :T_len, :]
            else:
                weights_all = self._template_weights(sig, dtype)            # (B,n_chunks,K)
                W_eff_all = torch.einsum('bnk,kdc->bndc', weights_all, B_stack)
                x_chunked = x_p.reshape(B, n_chunks, chunk, C)
                h_chunked = torch.einsum('bnsc,bndc->bnsd', x_chunked, W_eff_all)
                h_pre = h_chunked.reshape(B, n_chunks * chunk, -1)[:, :T_len, :]

        with torch.no_grad():
            w_f = weights_all.float()
            ent = -(w_f * torch.log(w_f.clamp(min=1e-8))).sum(dim=-1).mean()
            self._template_entropy_buf.copy_(ent.detach())
            if hasattr(self, '_template_weights_buf') and self._template_weights_buf is not None:
                w_mean = w_f.mean(dim=tuple(range(w_f.ndim - 1)))
                if self._template_weights_buf.shape == w_mean.shape:
                    self._template_weights_buf.copy_(w_mean.detach())
            if self.training and not torch.compiler.is_compiling():
                self._gate_stats['template_weights'] = w_f.mean(dim=tuple(range(w_f.ndim - 1))).detach()
                self._gate_stats['template_entropy'] = ent.detach()
                self._gate_stats['template_weights_std'] = w_f.std(dim=tuple(range(w_f.ndim - 1))).mean().detach()
        return h_pre

    def non_gate_parameters(self):
        """Yield structural parameters (basis, template_mixing/experts, bias) — Muon/normal LR."""
        if self.basis is not None:
            yield self.basis.weight
        if self.template_mixing is not None:
            yield self.template_mixing
        if self.template_bank is not None:
            yield self.template_bank
        # Phase 31 low-rank template deltas
        if self.template_base is not None:
            yield self.template_base
            yield self.template_delta_v
            yield self.template_delta_u
        # Tiny Expert stacked 3D weight tensors → structural AdamW group (ndim=3)
        if self.expert_up_w is not None:
            yield self.expert_up_w
        if self.expert_down_w is not None:
            yield self.expert_down_w
        # LoKR expert weight matrices
        if self.lokr_down_w is not None:
            yield self.lokr_down_w
        if self.lokr_up_w is not None:
            yield self.lokr_up_w
        yield self.bias
        if hasattr(self, "ocd_in"):
            yield self.ocd_in
        if hasattr(self, "ocd_out"):
            yield self.ocd_out
        if hasattr(self, "lie_generators"):
            yield self.lie_generators
        if hasattr(self, "poly_A"):
            yield self.poly_A
        if hasattr(self, "grassmann_bank"):
            yield self.grassmann_bank

    def forward(self, x, context_state, route_weights=None, context_gates=None, **kwargs):
        """
        route_weights:   optional (B, T, K) pre-computed routing tensor from SharedBlockRouter.
        context_gates:   optional dict from SharedContextGates with keys:
                           'basis_gate'     → (B, T, basis_size) raw gate logits (before sigmoid)
                           'output_coeffs'  → (B, T, gate_rank) raw output gate coefficients
                         When provided, local basis_modulator/output_gate_coeffs are skipped.
        """
        dtype = x.dtype
        # Phase 28C: shared basis injection — RemixedBlock pre-computes h_basis once
        # for all Q/K/V/O attention projections and sets self._h_basis before calling forward.
        _injected_h_basis = getattr(self, '_h_basis', None)
        if _injected_h_basis is not None:
            h_basis = _injected_h_basis.to(dtype=dtype)
        else:
            _ln_dtype = self.ln_basis.weight.dtype if hasattr(self.ln_basis, 'weight') else dtype
            if self.route_side == 'basis':
                # Phase 31: routing lives in the input map — W_b is per-chunk, W_m shared.
                _pre = self._routed_basis_proj(x, dtype)
            elif self.basis is None:
                # Phase 31 drop_basis_proj: no input map at all, LN acts on x.
                _pre = x
            else:
                _pre = self.basis(x)
            h_basis = self.ln_basis(_pre.to(dtype=_ln_dtype)).to(dtype=dtype)

        if self.use_context and context_state is not None:
            ctx = context_state.to(dtype=dtype)
            if self.operator_modulation == 'householder':
                v = F.normalize(self.operator_householder(ctx), dim=-1, eps=1e-6)
                h_basis = h_basis - 2.0 * (h_basis * v).sum(dim=-1, keepdim=True) * v
            elif self.operator_modulation == 'spectral':
                scale = 1.0 + torch.tanh(self.operator_spectral(ctx) * 0.1)
                h_basis = h_basis * scale.to(dtype=dtype)
            elif self.operator_modulation == 'lie':
                coeffs = self.lie_coeffs(ctx)
                generators = self.lie_generators.to(dtype=dtype)
                skew = 0.5 * (generators - generators.transpose(-1, -2))
                transport = torch.einsum('btm,mij->btij', coeffs, skew)
                eye = torch.eye(self.basis_size, device=x.device, dtype=dtype).view(1, 1, self.basis_size, self.basis_size)
                h_basis = torch.einsum('bti,btij->btj', h_basis, eye + 0.1 * transport)
            elif self.operator_modulation == 'polynomial':
                a = self.poly_coeffs(ctx)
                A = self.poly_A.to(dtype=dtype)
                z = torch.zeros_like(h_basis)
                h_pow = h_basis
                z = z + a[..., :1] * h_pow
                for i in range(1, self.poly_order + 1):
                    h_pow = torch.matmul(h_pow, A)
                    z = z + a[..., i:i+1] * h_pow
                h_basis = z
            elif self.operator_modulation == 'grassmann':
                alpha = F.softmax(self.grassmann_alpha(ctx), dim=-1)
                scale = torch.einsum('btk,kd->btd', alpha, self.grassmann_bank.to(dtype=dtype))
                h_basis = h_basis * scale

            # Basis gate: sparse or dense sigmoid with configurable temperature
            if self.use_basis_gate:
                if context_gates is not None and 'basis_gate' in context_gates:
                    gate_logits = context_gates['basis_gate'].to(dtype=dtype)
                elif self.basis_gate_mode == 'attn':
                    # Bilinear: content (from h_basis) ⊙ context projection → gate logits
                    gate_logits = (self.basis_gate_content(h_basis) *
                                   self.basis_gate_context(ctx))
                elif self.basis_gate_mode == 'lowrank':
                    coeffs_bg = self.basis_gate_coeffs(ctx)
                    gate_logits = torch.matmul(
                        coeffs_bg, self.basis_gate_vectors.to(dtype=dtype)
                    )
                elif self.basis_gate_mode == 'none':
                    gate_logits = None
                else:  # 'mlp' or 'linear'
                    gate_logits = self.basis_modulator(ctx)
                if gate_logits is None:
                    gate_basis = torch.ones_like(h_basis)
                elif self._film_gate_flag:
                    scale_logits, shift = gate_logits.chunk(2, dim=-1)
                    gate_basis = (1.0 + torch.tanh(scale_logits * 0.1)).to(dtype=dtype)
                    h_basis = h_basis * gate_basis + shift.to(dtype=dtype)
                    gate_basis = None
                elif self.sparse_gate_k > 0:
                    k = min(self.sparse_gate_k, self.basis_size)
                    topk_vals, topk_idx = torch.topk(gate_logits, k=k, dim=-1)
                    sparse = torch.zeros_like(gate_logits).scatter_(-1, topk_idx, F.softmax(topk_vals, dim=-1))
                    soft = torch.sigmoid(gate_logits / self.gate_temperature)
                    gate_basis = (sparse + (soft - soft.detach())).to(dtype=dtype)
                elif self.basis_gate_mode == 'centered':
                    # Centered gate: passthrough at init, symmetric amplify/suppress
                    s = self.centered_gate_scale.to(dtype=dtype)
                    gate_basis = (1.0 + torch.tanh(s * gate_logits)).to(dtype=dtype)
                else:
                    gate_basis = torch.sigmoid(gate_logits / self.gate_temperature).to(dtype=dtype)
                # Track basis gate stats (zero overhead when not training; skip under torch.compile)
                if self.training and gate_basis is not None and not torch.compiler.is_compiling():
                    with torch.no_grad():
                        gb = gate_basis.float()
                        self._gate_stats['basis_mean'] = gb.mean().detach()
                        self._gate_stats['basis_std']  = gb.std().detach()
                        self._gate_stats['basis_dead'] = (gb < 0.10).float().mean().detach()
                        self._gate_stats['basis_sat']  = (gb > 0.90).float().mean().detach()
            else:
                gate_basis = torch.ones_like(h_basis)

            # Output gate: LOW-RANK + CENTERED ACTIVATION
            # gate = 1 + tanh(scale * coeffs @ gate_basis_vectors)
            if self.use_output_gate:
                if context_gates is not None and 'output_coeffs' in context_gates:
                    coeffs = context_gates['output_coeffs'].to(dtype=dtype)
                else:
                    coeffs = self.output_gate_coeffs(ctx)                       # (B, T, r)
                gate_logits = torch.matmul(coeffs, self.output_gate_basis.to(dtype=dtype))  # (B, T, out)
                gate_out = 1.0 + torch.tanh(self.output_gate_scale.to(dtype=dtype) * gate_logits)
                # Track output gate stats
                if self.training and not torch.compiler.is_compiling():
                    with torch.no_grad():
                        go = gate_out.float()
                        self._gate_stats['out_mean'] = go.mean().detach()
                        self._gate_stats['out_std']  = go.std().detach()
            else:
                gate_out = None
        else:
            gate_basis = torch.ones_like(h_basis)
            gate_out = None

        h_gated = (h_basis * gate_basis).to(dtype=dtype)

        if self.route_side == 'basis':
            # Phase 31: routing was already applied to W_b above, so the output map
            # is a single shared matrix — one full-efficiency dense GEMM, and for an
            # expanding projection this is the *large* one.
            pre_output = F.linear(h_gated, self.template_mixing.to(dtype=dtype))
        elif self.lokr_expert:
            # ──────────────────────────────────────────────────────────────
            # LoKR-Remix forward: shared base + top-k low-rank expert deltas
            # ──────────────────────────────────────────────────────────────
            # 1. Shared-base output (always active, shrunk template_mixing)
            base_out = F.linear(h_gated, self.template_mixing.to(dtype=dtype))

            # 2. Router: per-token (FFN) or per-sequence (attn)
            if self.routing_scope == 'per_sequence':
                route_input = x.float().mean(dim=1, keepdim=True)                  # (B, 1, in)
            else:
                route_input = x.float()                                            # (B, T, in)
            # route_proj: (K, in) — frozen or learned; performed in float32 for stability
            route_logits = F.linear(route_input, self.lokr_route_proj.float())  # (B, *, K)
            if self.routing_scope == 'per_sequence':
                route_logits = route_logits.expand(-1, h_gated.shape[1], -1)  # (B, T, K)

            # 3. Top-k routing and batched low-rank expert computation
            K = self.lokr_n_experts
            topk = self.lokr_topk
            if topk > 0 and topk < K:
                topk_w, topk_idx = route_logits.topk(topk, dim=-1)  # (B, T, topk)
                topk_w = F.softmax(topk_w.float(), dim=-1).to(dtype)

                # Batched low-rank matmul over ALL experts, then gather top-k
                # lokr_down_w: (K, R, in) → all_h: (B, T, K, R)
                all_h = torch.einsum('bti,kri->btkr', x.to(self.lokr_down_w.dtype),
                                     self.lokr_down_w)  # (B, T, K, R)
                # lokr_up_w: (K, out, R) → all_out: (B, T, K, out)
                all_out = torch.einsum('btkr,kor->btko', all_h,
                                       self.lokr_up_w)  # (B, T, K, out)
                all_out = all_out.to(dtype)

                # Gather: (B, T, topk, out)
                idx_exp = topk_idx.unsqueeze(-1).expand(*topk_idx.shape, all_out.shape[-1])
                topk_out = torch.gather(all_out, dim=-2, index=idx_exp)
                expert_delta = (topk_out * topk_w.unsqueeze(-1)).sum(dim=-2)  # (B, T, out)
            else:
                # Soft routing: all experts, weighted sum
                soft_w = F.softmax(route_logits.float(), dim=-1).to(dtype)  # (B, T, K)
                all_h = torch.einsum('bti,kri->btkr', x.to(self.lokr_down_w.dtype),
                                     self.lokr_down_w)
                all_out = torch.einsum('btkr,kor->btko', all_h,
                                       self.lokr_up_w).to(dtype)
                expert_delta = (all_out * soft_w.unsqueeze(-1)).sum(dim=-2)  # (B, T, out)

            # Diagnostics: router entropy
            with torch.no_grad():
                prob_f = F.softmax(route_logits.float(), dim=-1)
                ent = -(prob_f * torch.log(prob_f.clamp(min=1e-8))).sum(dim=-1).mean()
                if hasattr(self, '_template_entropy_buf'):
                    self._template_entropy_buf.copy_(ent.detach())
                if hasattr(self, '_template_weights_buf') and self._template_weights_buf is not None:
                    w_mean = prob_f.mean(dim=tuple(range(prob_f.ndim - 1)))
                    if self._template_weights_buf.shape == w_mean.shape:
                        self._template_weights_buf.copy_(w_mean.detach())
                if self.training and not torch.compiler.is_compiling():
                    self._gate_stats['template_weights'] = prob_f.mean(dim=tuple(range(prob_f.ndim - 1))).detach()
                    self._gate_stats['template_entropy'] = ent.detach()
                    self._gate_stats['template_weights_std'] = prob_f.std(dim=tuple(range(prob_f.ndim - 1))).mean().detach()

            pre_output = base_out + expert_delta

        elif self.tiny_expert and self.n_templates > 1:
            # Phase 23 Tiny Experts — vectorized via stacked 3D parameter tensors.
            # expert_up_w:   (K, expert_dim, basis_size)
            # expert_down_w: (K, out_features, expert_dim)
            # ALL K experts are computed simultaneously with two einsums; top-k gather
            # selects contributions.  No Python loop → no DDP deadlock → torch.compile-safe.
            K = self.n_templates
            B, T, _ = h_gated.shape
            out_features = self.expert_down_w.shape[1]

            # ── Routing ──────────────────────────────────────────────────────────
            if route_weights is not None:
                # Pre-computed by SharedBlockRouter — skip local routing entirely
                rw = route_weights  # (B, T, K) already softmaxed
            elif getattr(self, '_qrouter', None) is not None:
                # Quantile-balanced routing
                rw = self._qrouter(x)                                      # (B, T, K)
            elif not self._use_shared_route:
                if self.routing_scope == 'per_sequence':
                    x_pool = x.float().mean(dim=1, keepdim=True)              # (B, 1, D)
                    rw = F.softmax(x_pool @ self.template_route.float(), dim=-1).to(dtype)  # (B, 1, K)
                    rw = rw.expand(-1, T, -1)                                  # (B, T, K)
                else:
                    rw = F.softmax(x.float() @ self.template_route.float(), dim=-1).to(dtype)  # (B, T, K)
            else:
                raise RuntimeError("TinyExpert with use_shared_route=True requires route_weights kwarg")

            with torch.no_grad():
                w_f = rw.float()
                ent = -(w_f * torch.log(w_f.clamp(min=1e-8))).sum(dim=-1).mean()
                self._template_entropy_buf.copy_(ent.detach())
                if hasattr(self, '_template_weights_buf') and self._template_weights_buf is not None:
                    w_mean = w_f.mean(dim=tuple(range(w_f.ndim - 1)))
                    if self._template_weights_buf.shape == w_mean.shape:
                        self._template_weights_buf.copy_(w_mean.detach())
                if self.training and not torch.compiler.is_compiling():
                    self._gate_stats['template_weights'] = w_f.mean(dim=tuple(range(w_f.ndim - 1))).detach()
                    self._gate_stats['template_entropy'] = ent.detach()
                    self._gate_stats['template_weights_std'] = w_f.std(dim=tuple(range(w_f.ndim - 1))).mean().detach()

            # ── Fully vectorized expert computation (single pair of einsums) ────────
            # Accumulated in float32 to avoid instability with relu.square() and massive reductions
            # up:  h_gated (B,T,S) × expert_up_w (K,H,S) → (B,T,K,H)
            all_up  = torch.einsum('bts,khs->btkh', h_gated.float(), self.expert_up_w.float())
            all_up  = F.relu(all_up).square()
            # down: (B,T,K,H) × expert_down_w (K,O,H) → (B,T,K,O)
            all_out = torch.einsum('btkh,koh->btko', all_up, self.expert_down_w.float()).to(dtype)

            topk_val = self.tiny_expert_topk
            if 0 < topk_val < K:
                # Sparse gather: pick top-k expert outputs then weighted-sum
                topk_w, topk_idx = rw.topk(topk_val, dim=-1)                # (B,T,tk)
                topk_w = topk_w / (topk_w.sum(-1, keepdim=True) + 1e-8)
                idx_exp = topk_idx.unsqueeze(-1).expand(*topk_idx.shape, out_features)  # (B,T,tk,O)
                topk_out = torch.gather(all_out, dim=2, index=idx_exp)       # (B,T,tk,O)
                pre_output = (topk_out * topk_w.unsqueeze(-1)).sum(dim=2)   # (B,T,O)
            else:
                # Soft routing: weighted sum over all K experts
                pre_output = (all_out * rw.unsqueeze(-1)).sum(dim=2)         # (B,T,O)
        elif self.n_templates > 1:
            if self.template_base is not None:
                # ──────────────────────────────────────────────────────────────
                # Phase 31: shared base + K low-rank deltas.
                #   W_eff = T_0 + sum_k a_k U_k V_k^T
                #   y     = h T_0^T + ((h V) * a) U^T
                # Three dense GEMMs.  Soft mixing over all K templates costs
                # K*r*(basis+out) per token instead of K*out*basis per chunk, and
                # nothing of shape (out, basis) is ever written to HBM — so this
                # path is insensitive to chunk_routing_size and works equally well
                # per-token.
                # ──────────────────────────────────────────────────────────────
                r = self.template_delta_rank
                K = self.n_templates
                B, T_len, C = x.shape
                base = F.linear(h_gated, self.template_base.to(dtype=dtype))   # (B,T,out)
                z = h_gated @ self.template_delta_v.to(dtype=dtype)            # (B,T,K*r)

                if self.chunk_routing_size > 0:
                    chunk = self.chunk_routing_size
                    n_chunks = (T_len + chunk - 1) // chunk
                    pad = n_chunks * chunk - T_len
                    x_p = F.pad(x, (0, 0, 0, pad)) if pad > 0 else x
                    z_p = F.pad(z, (0, 0, 0, pad)) if pad > 0 else z
                    sig = x_p.reshape(B, n_chunks, chunk, C)[:, :, 0, :].float()
                    weights_all = self._template_weights(sig, dtype)           # (B,n_chunks,K)
                    # Broadcast the chunk-constant coefficient over the chunk's tokens
                    # instead of expanding it into a (B, T, K) tensor.
                    z_p = (z_p.reshape(B, n_chunks, chunk, K, r)
                           * weights_all.reshape(B, n_chunks, 1, K, 1))
                    z = z_p.reshape(B, n_chunks * chunk, K * r)[:, :T_len, :]
                else:
                    # Route per token, matching the legacy per-token compose branch
                    # below — which ignores routing_scope even though every projection
                    # is built with routing_scope='per_sequence'. Mirroring it keeps
                    # delta mode a pure throughput change and nothing else.
                    weights_all = self._template_weights(x.float(), dtype)     # (B,T,K)
                    z = (z.reshape(B, T_len, K, r)
                         * weights_all.reshape(B, T_len, K, 1)).reshape(B, T_len, K * r)

                pre_output = base + (z @ self.template_delta_u.to(dtype=dtype))

                with torch.no_grad():
                    w_f = weights_all.float()
                    ent = -(w_f * torch.log(w_f.clamp(min=1e-8))).sum(dim=-1).mean()
                    self._template_entropy_buf.copy_(ent.detach())
                    if hasattr(self, '_template_weights_buf') and self._template_weights_buf is not None:
                        w_mean = w_f.mean(dim=tuple(range(w_f.ndim - 1)))
                        if self._template_weights_buf.shape == w_mean.shape:
                            self._template_weights_buf.copy_(w_mean.detach())
                    if self.training and not torch.compiler.is_compiling():
                        self._gate_stats['template_weights'] = w_f.mean(dim=tuple(range(w_f.ndim - 1))).detach()
                        self._gate_stats['template_entropy'] = ent.detach()
                        self._gate_stats['template_weights_std'] = w_f.std(dim=tuple(range(w_f.ndim - 1))).mean().detach()
            elif self._global_bank is not None:
                # Phase 28E/F: global template bank — compute W_eff once per forward, apply as bmm.
                # Routing is done by the per-layer router inside GlobalTemplateBank.
                W_eff = self._global_bank.get_effective_weight(
                    self._global_bank_key, self._layer_idx_for_bank, x
                ).to(dtype=dtype)  # (B, out_features, basis_size) — cast to h_gated dtype
                pre_output = torch.bmm(h_gated, W_eff.transpose(1, 2))  # (B, T, out_features)
                with torch.no_grad():
                    self._template_entropy_buf.copy_(torch.zeros(1, device=x.device))
            elif self.chunk_routing_size > 0:
                # Phase 28D: amortized chunk routing — vectorized (no Python loop).
                # Compute routing once per chunk from the first token, then apply
                # a single chunk-level effective weight to all tokens in the chunk.
                T_stack = self.template_bank.to(dtype=dtype)  # (K, out_features, basis_size)
                chunk = self.chunk_routing_size
                B, T_len, C = x.shape
                n_chunks = (T_len + chunk - 1) // chunk
                pad = n_chunks * chunk - T_len

                # Pad T to a clean multiple of chunk (zero-pad the tail)
                x_p = F.pad(x, (0, 0, 0, pad)) if pad > 0 else x          # (B, n*chunk, C)
                h_p = F.pad(h_gated, (0, 0, 0, pad)) if pad > 0 else h_gated  # (B, n*chunk, basis)

                # Extract causal routing anchor: first token of each chunk.
                # Reshape → (B, n_chunks, chunk, C), take index 0 along the chunk dim.
                x_anchors = x_p.reshape(B, n_chunks, chunk, C)[:, :, 0, :].float()  # (B, n_chunks, C)

                # Route all chunks: (B, n_chunks, K)
                topk = getattr(self, 'template_topk', 0)
                # Phase 31: top-1 + 'grouped' skips the W_eff materialization
                # entirely. _grouped_route decides whether that applies, for both
                # the plain and the quantile routers, and returns None when it
                # does not so the compose path below runs unchanged.
                _g = self._grouped_route(x_anchors, dtype)
                _grouped = _g is not None
                if _grouped:
                    sel_idx, coef, weights_all = _g
                elif getattr(self, '_qrouter', None) is not None:
                    weights_all = self._qrouter(x_anchors).to(dtype)
                elif getattr(self, 'route_affine', False) and getattr(self, 'template_route', None) is not None:
                    weights_all = self._affine_weights(x_anchors, dtype)
                elif hasattr(self, 'template_route') and self.template_route is not None:
                    logits_all = torch.einsum('bnc,ck->bnk', x_anchors, self.template_route.float())
                    if topk > 0 and topk < self.n_templates:
                        topk_vals, topk_idx = logits_all.topk(topk, dim=-1)
                        if topk == 1:
                            one_hot = F.one_hot(topk_idx.squeeze(-1), num_classes=self.n_templates).to(dtype)
                            weights_all = one_hot * F.softmax(topk_vals.float(), dim=-1).to(dtype)
                        else:
                            one_hot = F.one_hot(topk_idx, num_classes=self.n_templates).to(dtype)
                            weights_all = (one_hot * F.softmax(topk_vals.float(), dim=-1).unsqueeze(-1).to(dtype)).sum(dim=-2)
                    else:
                        weights_all = F.softmax(logits_all.float(), dim=-1).to(dtype)
                else:
                    weights_all = torch.full((B, n_chunks, self.n_templates), 1.0 / self.n_templates, device=x.device, dtype=dtype)

                basis_sz = h_p.shape[-1]
                if _grouped:
                    # K dense GEMMs, template bank read K times instead of B*n_chunks.
                    out_p = _grouped_top1_chunk_apply(h_p, sel_idx, T_stack, coef)
                    pre_output = out_p[:, :T_len, :]
                else:
                    # Build one effective weight matrix per chunk: (B, n_chunks, out, basis)
                    W_eff_all = torch.einsum('bnk,koc->bnoc', weights_all, T_stack)

                    # Apply: reshape h to (B, n_chunks, chunk, basis), batch-contract on basis
                    h_chunked = h_p.reshape(B, n_chunks, chunk, basis_sz)            # (B, n_chunks, chunk, basis)
                    out_chunked = torch.einsum('bnsc,bnoc->bnso', h_chunked, W_eff_all)  # (B, n_chunks, chunk, out)
                    pre_output = out_chunked.reshape(B, n_chunks * chunk, -1)[:, :T_len, :]  # (B, T, out)

                with torch.no_grad():
                    w_f = weights_all.float()
                    ent = -(w_f * torch.log(w_f.clamp(min=1e-8))).sum(dim=-1).mean()
                    self._template_entropy_buf.copy_(ent.detach())
                    if hasattr(self, '_template_weights_buf') and self._template_weights_buf is not None:
                        w_mean = w_f.mean(dim=tuple(range(w_f.ndim - 1)))
                        if self._template_weights_buf.shape == w_mean.shape:
                            self._template_weights_buf.copy_(w_mean.detach())
                    if self.training and not torch.compiler.is_compiling():
                        self._gate_stats['template_weights'] = w_f.mean(dim=tuple(range(w_f.ndim - 1))).detach()
                        self._gate_stats['template_entropy'] = ent.detach()
                        self._gate_stats['template_weights_std'] = w_f.std(dim=tuple(range(w_f.ndim - 1))).mean().detach()
            else:
                # Legacy Phase 22: MoE-style per-token template routing
                if getattr(self, '_qrouter', None) is not None:
                    route_logits = self._qrouter(x)
                elif hasattr(self, 'template_route') and self.template_route is not None:
                    route_logits = x.float() @ self.template_route.float()  # (B, T, K)
                else:
                    route_logits = torch.zeros(x.shape[0], x.shape[1], self.n_templates, device=x.device)
                with torch.no_grad():
                    w_f = F.softmax(route_logits.float(), dim=-1)
                    ent = -(w_f * torch.log(w_f.clamp(min=1e-8))).sum(dim=-1).mean()
                    self._template_entropy_buf.copy_(ent.detach())
                    if hasattr(self, '_template_weights_buf') and self._template_weights_buf is not None:
                        w_mean = w_f.mean(dim=tuple(range(w_f.ndim - 1)))
                        if self._template_weights_buf.shape == w_mean.shape:
                            self._template_weights_buf.copy_(w_mean.detach())
                    if self.training and not torch.compiler.is_compiling():
                        self._gate_stats['template_weights'] = w_f.mean(dim=tuple(range(w_f.ndim - 1))).detach()
                        self._gate_stats['template_entropy'] = ent.detach()
                        self._gate_stats['template_weights_std'] = w_f.std(dim=tuple(range(w_f.ndim - 1))).mean().detach()
                topk = getattr(self, 'template_topk', 0)
                if topk > 0 and topk < self.n_templates:
                    # Hard top-k sparse routing — stack all templates, matmul, gather
                    topk_vals, topk_idx = route_logits.topk(topk, dim=-1)   # (B, T, topk)
                    topk_weights = F.softmax(topk_vals.float(), dim=-1).to(dtype)  # (B, T, topk)
                    T_stack = self.template_bank.to(dtype=dtype)  # (K, out, basis)
                    all_out = torch.einsum('bts,kos->btko', h_gated, T_stack)  # (B, T, K, out)
                    idx_exp = topk_idx.unsqueeze(-1).expand(
                        *topk_idx.shape, all_out.shape[-1]
                    )  # (B, T, topk, out)
                    topk_out = torch.gather(all_out, dim=-2, index=idx_exp)   # (B, T, topk, out)
                    pre_output = (topk_out * topk_weights.unsqueeze(-1)).sum(dim=-2)  # (B, T, out)
                else:
                    # Soft routing: batched weighted sum over all K templates (single einsum)
                    route_weights = F.softmax(route_logits, dim=-1).to(dtype)  # (B, T, K)
                    T_stack = self.template_bank.to(dtype=dtype)  # (K, out, basis)
                    all_out = torch.einsum('bts,kos->btko', h_gated, T_stack)  # (B, T, K, out)
                    pre_output = (all_out * route_weights.unsqueeze(-1)).sum(dim=2)  # (B, T, out)
        else:
            pre_output = F.linear(h_gated, self.template_mixing.to(dtype=dtype))


        if self.use_context and context_state is not None and self.operator_modulation == 'ocd':
            coeffs = self.ocd_coeffs(ctx)                                         # (B, T, r)
            proj = torch.matmul(h_basis, self.ocd_in.to(dtype=dtype).transpose(0, 1))  # (B, T, r)
            delta = torch.matmul(proj * coeffs, self.ocd_out.to(dtype=dtype))      # (B, T, out)
            pre_output = pre_output + self.ocd_scale.to(dtype=dtype) * delta
            # Soft overlap penalty proxy: ||W_s W_d^T||_F^2 where W_d is mean dynamic map.
            c_mean = coeffs.mean(dim=(0, 1))                                       # (r,)
            w_dyn = torch.einsum(
                'r,ro,ri->oi',
                c_mean,
                self.ocd_out.to(dtype=dtype),
                self.ocd_in.to(dtype=dtype),
            )                                                                       # (out, basis)
            # Guard: template_bank may be None (n_templates=1 or tiny_expert/lokr modes)
            _tmix = self.template_mixing
            if _tmix is None and self.template_bank is not None:
                _tmix = self.template_bank[0]  # (out, basis) — first template from stacked 3D
            tmix = _tmix
            overlap = torch.matmul(tmix.to(dtype=dtype), w_dyn.transpose(0, 1))
            self._last_orth_loss = overlap.pow(2).mean()
        else:
            self._last_orth_loss = None

        # Phase 35: template diversity loss — penalize pairwise cosine similarity
        # between templates to prevent collapse (especially in attention Q/K/V).
        if (self.training and self.template_bank is not None
                and self.n_templates > 1 and not torch.compiler.is_compiling()):
            bank = self.template_bank.float()                          # (K, out, basis)
            K = bank.shape[0]
            flat = bank.reshape(K, -1)                                 # (K, out*basis)
            flat = F.normalize(flat, dim=-1)                           # unit vectors
            cos_mat = flat @ flat.T                                    # (K, K)
            # Upper triangle only (exclude diagonal self-similarity)
            mask = torch.triu(torch.ones(K, K, device=bank.device, dtype=torch.bool), diagonal=1)
            pairwise_cos = cos_mat[mask]                               # (K*(K-1)/2,)
            # Mean squared cosine: 0 = perfectly diverse, 1 = collapsed
            self._template_diversity_loss = pairwise_cos.pow(2).mean()
        else:
            self._template_diversity_loss = None

        if gate_out is not None:
            pre_output = pre_output * gate_out

        return (pre_output + self.bias.to(dtype=dtype)).to(dtype=dtype) if self.bias is not None else pre_output.to(dtype=dtype)


class RemixedLinearFused(nn.Module):
    """Throughput-optimized RemixedLinear for chunk routing (29C config).

    Only supports the n_templates>1, chunk_routing_size>0 path.
    All other routing modes (per-token, LoKR, TinyExpert, global bank,
    operator modulation) are dropped for clarity and speed.

    Key optimizations over RemixedLinear:
      1. Template assembly einsum('bnk,koc→bnoc') → single cuBLAS GEMM
         via reshape + matmul: (BN,K) @ (K, O*B_sz) → reshape
      2. Chunk mixing einsum('bnsc,bnoc→bnso') → torch.bmm:
         bmm((BN, chunk, B_sz), (BN, B_sz, O)) → (BN, chunk, O)
      3. Skips gate multiply entirely when basis gate is disabled
         (avoids multiplying by ones tensor)

    Wrap with torch.compile() for additional elementwise fusion of
    LN + gate + bias into fewer kernel launches.
    """

    def __init__(self, in_features, out_features, context_dim,
                 basis_size=64, remixed_linear_kwargs=None, scale_basis=True,
                 film_gate=False, routing_scope='per_sequence'):
        super().__init__()
        if remixed_linear_kwargs is None:
            remixed_linear_kwargs = {}
        kw = remixed_linear_kwargs

        # ── Extract config from kwargs (same keys as RemixedLinear) ───────
        n_templates = int(kw.get('n_templates', 8))
        chunk_routing_size = int(kw.get('chunk_routing_size', 64))
        assert n_templates > 1, "RemixedLinearFused requires n_templates > 1"
        assert chunk_routing_size > 0, "RemixedLinearFused requires chunk_routing_size > 0"

        use_basis_gate = kw.get('use_basis_gate', True)
        use_output_gate = kw.get('use_output_gate', True)
        basis_gate_mode = kw.get('basis_gate_mode', 'mlp')
        output_gate_rank = int(kw.get('output_gate_rank', 8))
        gate_temperature = max(float(kw.get('gate_temperature', 1.0)), 1e-6)
        template_routing_learned = kw.get('template_routing_learned', False)
        template_topk = int(kw.get('template_topk', 0))
        _bsf = int(kw.get('basis_scale_factor', 4))

        if scale_basis:
            basis_size = max(basis_size, min(in_features, out_features) // max(1, _bsf))

        self.in_features = in_features
        self.out_features = out_features
        self.basis_size = basis_size
        self.n_templates = n_templates
        self.chunk_routing_size = chunk_routing_size
        self.use_basis_gate = use_basis_gate
        self.use_output_gate = use_output_gate
        self.use_context = kw.get('use_context', True)
        self.routing_scope = routing_scope
        self.template_topk = template_topk
        self.gate_temperature = gate_temperature
        self._gate_stats = {}
        self._film_gate_flag = film_gate
        self._last_orth_loss = None  # compat with RemixedLinear diagnostics

        # ── Compat attrs (None stubs for shared code: weight init, FLOPs) ─
        self.template_mixing = None
        self.expert_up_w = None
        self.expert_down_w = None
        self.lokr_down_w = None
        self.lokr_up_w = None
        self.lokr_route_proj = None
        self.lokr_expert = False
        self.tiny_expert = False
        self._use_global_bank = False
        self._global_bank = None
        self.basis_gate_content = None
        self.basis_gate_context = None

        # ── Structural parameters ────────────────────────────────────────
        self.basis = Linear(in_features, basis_size, bias=False)
        self.ln_basis = nn.LayerNorm(basis_size)
        self.bias = nn.Parameter(torch.zeros(out_features))

        # Template bank: (K, out_features, basis_size)
        self.template_bank = nn.Parameter(
            torch.randn(n_templates, out_features, basis_size)
        )

        # Routing: quantile-balanced or simple learned/frozen softmax
        self.use_quantile_route = int(kw.get('use_quantile_route', 0))
        self._qrouter = None

        if self.use_quantile_route == 2 and template_topk > 0:
            self._qrouter = QuantileCrossAttentionRouter(
                in_features, n_templates, template_topk)
            self.template_route = None
        elif self.use_quantile_route == 1 and template_topk > 0:
            self._qrouter = QuantileBalancedRouter(
                in_features, n_templates, template_topk,
                learned=template_routing_learned)
            self.template_route = None
        else:
            # Standard routing projection: (in_features, K) — frozen by default
            route_init = torch.randn(in_features, n_templates) / (in_features ** 0.5)
            if template_routing_learned:
                self.template_route = nn.Parameter(route_init)
            else:
                self.register_buffer('template_route', route_init)

        # ── Context gates ────────────────────────────────────────────────
        self.basis_modulator = None
        self.basis_gate_mode = basis_gate_mode if use_basis_gate else 'none'

        if self.use_context and use_basis_gate:
            _gate_out_size = 2 * basis_size if (use_basis_gate and film_gate) else basis_size
            if basis_gate_mode == 'linear':
                self.basis_modulator = Linear(context_dim, _gate_out_size, bias=True)
                nn.init.zeros_(self.basis_modulator.weight)
                nn.init.zeros_(self.basis_modulator.bias)
            elif basis_gate_mode == 'mlp':
                basis_hidden = max(context_dim // 2, min(basis_size, context_dim * 2))
                self.basis_modulator = nn.Sequential(
                    Linear(context_dim, basis_hidden, bias=True),
                    nn.GELU(),
                    Linear(basis_hidden, _gate_out_size, bias=True),
                )
                nn.init.zeros_(self.basis_modulator[-1].weight)
                if self.basis_modulator[-1].bias is not None:
                    nn.init.zeros_(self.basis_modulator[-1].bias)

        if self.use_context and use_output_gate:
            r = output_gate_rank
            self.output_gate_coeffs = Linear(context_dim, r, bias=True)
            self.output_gate_basis = nn.Parameter(torch.randn(r, out_features) * 0.01)
            self.output_gate_scale = nn.Parameter(torch.ones(1) * 0.1)

        # Entropy tracking (diagnostic only)
        self.register_buffer('_template_entropy_buf', torch.zeros(1), persistent=False)

    # ── Optimizer group helpers ──────────────────────────────────────────

    def gate_parameters(self):
        """Yield context-gate parameters (lower-LR optimizer group)."""
        if self.basis_modulator is not None:
            yield from self.basis_modulator.parameters()
        if self.use_output_gate:
            yield self.output_gate_coeffs.weight
            if self.output_gate_coeffs.bias is not None:
                yield self.output_gate_coeffs.bias
            yield self.output_gate_basis
            yield self.output_gate_scale
        if isinstance(self.template_route, nn.Parameter):
            yield self.template_route
        if self._qrouter is not None:
            yield from self._qrouter.parameters()

    def non_gate_parameters(self):
        """Yield structural parameters (Muon / normal-LR group)."""
        yield self.basis.weight
        yield self.template_bank
        yield self.bias

    # ── Weight conversion ────────────────────────────────────────────────

    @classmethod
    def from_remixed_linear(cls, orig):
        """Convert a trained RemixedLinear to RemixedLinearFused.

        Copies all matching parameters/buffers.  Raises if the source
        RemixedLinear does not use chunk routing with multiple templates.
        """
        assert isinstance(orig, RemixedLinear), "Expected a RemixedLinear instance"
        assert orig.n_templates > 1, "Requires multi-template RemixedLinear"
        assert orig.chunk_routing_size > 0, "Requires chunk routing (chunk_routing_size > 0)"
        # This class assumes the bank is (K, out, basis) and W_b is a separate shared
        # matrix, which Phase 31's route_side='basis' / drop_basis_proj both break.
        assert getattr(orig, 'route_side', 'output') == 'output', \
            "RemixedLinearFused does not support route_side='basis'"
        assert not getattr(orig, 'drop_basis_proj', False), \
            "RemixedLinearFused does not support drop_basis_proj"

        ctx_dim = (orig.output_gate_coeffs.weight.shape[1]
                   if orig.use_output_gate else 64)
        out_gate_rank = (orig.output_gate_coeffs.weight.shape[0]
                         if orig.use_output_gate else 8)

        kw = dict(
            n_templates=orig.n_templates,
            chunk_routing_size=orig.chunk_routing_size,
            use_basis_gate=orig.use_basis_gate,
            use_output_gate=orig.use_output_gate,
            use_context=orig.use_context,
            basis_gate_mode=getattr(orig, 'basis_gate_mode', 'linear'),
            output_gate_rank=out_gate_rank,
            gate_temperature=orig.gate_temperature,
            template_routing_learned=(isinstance(orig.template_route, nn.Parameter)
                                      if orig.template_route is not None else False),
            template_topk=getattr(orig, 'template_topk', 0),
            use_quantile_route=getattr(orig, 'use_quantile_route', 0),
        )
        fused = cls(
            in_features=orig.basis.weight.shape[1],
            out_features=orig.template_bank.shape[1],
            context_dim=ctx_dim,
            basis_size=orig.basis_size,
            remixed_linear_kwargs=kw,
            scale_basis=False,            # already scaled in orig
            film_gate=getattr(orig, '_film_gate_flag', False),
            routing_scope=orig.routing_scope,
        )

        # Copy matching parameters and buffers
        orig_sd = orig.state_dict()
        fused_sd = fused.state_dict()
        for key in fused_sd:
            if key in orig_sd and fused_sd[key].shape == orig_sd[key].shape:
                fused_sd[key].copy_(orig_sd[key])
        fused.load_state_dict(fused_sd)
        return fused

    # ── torch.compile integration ────────────────────────────────────────

    def compile(self, mode='reduce-overhead', fullgraph=True):
        """Compile the forward pass for elementwise fusion (LN + gate + bias).

        This fuses the remaining elementwise operations (LayerNorm, sigmoid
        gate, output gate tanh, bias add) into fewer GPU kernel launches on
        top of the GEMM/bmm optimizations.

        Args:
            mode: torch.compile mode.
                  'reduce-overhead' — best for inference (uses CUDA graphs).
                  'default'         — best for training (no CUDA graph overhead).
            fullgraph: If True, compile the entire forward as a single graph.
                       Set False if you hit graph-break issues.

        Returns:
            self (with forward replaced by the compiled version).
        """
        self.forward = torch.compile(self.forward, mode=mode, fullgraph=fullgraph)
        return self

    # ── Forward pass ─────────────────────────────────────────────────────

    def forward(self, x, context_state=None, route_weights=None, **kwargs):
        dtype = x.dtype
        B, T_len, C = x.shape
        K = self.n_templates
        O = self.out_features
        B_sz = self.basis_size
        chunk = self.chunk_routing_size

        # ── Step 1: Basis projection + LayerNorm ──────────────────────
        _ln_dtype = self.ln_basis.weight.dtype
        h = self.ln_basis(self.basis(x).to(dtype=_ln_dtype)).to(dtype=dtype)

        # ── Step 2-3: Context gating ──────────────────────────────────
        gate_out = None
        if context_state is not None:
            ctx = context_state.to(dtype=dtype)

            # Basis gate (optional — skip multiply when disabled)
            if self.use_basis_gate and self.basis_modulator is not None:
                gate_logits = self.basis_modulator(ctx)
                h = h * torch.sigmoid(gate_logits / self.gate_temperature).to(dtype=dtype)

            # Output gate (low-rank centered: 1 + tanh(s · coeffs @ basis))
            if self.use_output_gate:
                coeffs = self.output_gate_coeffs(ctx)                          # (B, T, r)
                gl = torch.matmul(coeffs, self.output_gate_basis.to(dtype=dtype))  # (B, T, O)
                gate_out = 1.0 + torch.tanh(self.output_gate_scale.to(dtype=dtype) * gl)

        # ── Step 4-5: Chunk routing with GEMM / bmm ──────────────────
        T_stack = self.template_bank.to(dtype=dtype)   # (K, O, B_sz)
        n_chunks = (T_len + chunk - 1) // chunk
        pad = n_chunks * chunk - T_len

        # Pad to clean multiple of chunk size
        x_p = F.pad(x, (0, 0, 0, pad)) if pad > 0 else x
        h_p = F.pad(h, (0, 0, 0, pad)) if pad > 0 else h

        # Routing anchors: first token of each chunk
        x_anchors = x_p.reshape(B, n_chunks, chunk, C)[:, :, 0, :]  # (B, n_chunks, C)

        # ── Routing: quantile-balanced or standard matmul+softmax ─────
        if self._qrouter is not None:
            anchor_flat = x_anchors.reshape(B * n_chunks, 1, C)
            weights_flat = self._qrouter(anchor_flat)              # (BN, 1, K)
            weights_all = weights_flat.reshape(B, n_chunks, K).to(dtype)
        elif self.template_route is not None:
            logits_all = x_anchors.float() @ self.template_route.float()  # (B, n_chunks, K)
            topk = self.template_topk
            if topk > 0 and topk < K:
                topk_vals, topk_idx = logits_all.topk(topk, dim=-1)
                if topk == 1:
                    one_hot = F.one_hot(topk_idx.squeeze(-1), num_classes=K).to(dtype)
                    weights_all = one_hot * F.softmax(topk_vals.float(), dim=-1).to(dtype)
                else:
                    one_hot = F.one_hot(topk_idx, num_classes=K).to(dtype)
                    weights_all = (one_hot * F.softmax(topk_vals.float(), dim=-1).unsqueeze(-1).to(dtype)).sum(dim=-2)
            else:
                weights_all = F.softmax(logits_all.float(), dim=-1).to(dtype)  # (B, n_chunks, K)
        else:
            raise RuntimeError("RemixedLinearFused: no routing method configured "
                               "(need template_route or _qrouter)")

        # ── OPT 1: Template assembly — single GEMM ───────────────────
        # einsum('bnk,koc→bnoc') → (BN, K) @ (K, O*B_sz) → reshape
        BN = B * n_chunks
        W_eff_flat = weights_all.reshape(BN, K) @ T_stack.reshape(K, O * B_sz)
        # W_eff_flat: (BN, O * B_sz)

        # ── OPT 2: Chunk mixing — torch.bmm ──────────────────────────
        # einsum('bnsc,bnoc→bnso') → bmm((BN, chunk, B_sz), (BN, B_sz, O))
        h_chunked = h_p.reshape(BN, chunk, B_sz)
        W_eff_t = W_eff_flat.reshape(BN, O, B_sz).transpose(1, 2)  # (BN, B_sz, O)
        pre_output = torch.bmm(h_chunked, W_eff_t)                 # (BN, chunk, O)
        pre_output = pre_output.reshape(B, n_chunks * chunk, O)[:, :T_len, :]

        # Entropy tracking (no-grad diagnostic — skip under torch.compile)
        if not torch.compiler.is_compiling():
            with torch.no_grad():
                w_f = weights_all.float()
                ent = -(w_f * torch.log(w_f.clamp(min=1e-8))).sum(dim=-1).mean()
                self._template_entropy_buf.copy_(ent.detach())

        # ── Step 6-7: Output gate + bias ──────────────────────────────
        if gate_out is not None:
            pre_output = pre_output * gate_out

        return (pre_output + self.bias.to(dtype=dtype)).to(dtype=dtype)


class DualGateLinear(nn.Module):
    """DualGateLinear: single dense projection + dual multiplicative context gate.

    Forward computation:
        h = LayerNorm(W · x)                        # standard dense projection, LN in output space
        y = h ⊙ σ(W_basis_gate · ctx / T)          # Stage-1: D-dim sigmoid gate from causal context
        y = y ⊙ (1 + tanh(s · U · ctx))            # Stage-2: low-rank output gate (same as RemixedLinear)

    Key structural differences vs RemixedLinear (basis_size < D):
    - No factorized W_b × W_m — single W ∈ ℝ^{D×C} (standard dense matmul)
    - LayerNorm applied to the FULL output space (D-dim), not a compressed basis
    - Stage-1 gate is D-dimensional (vs B-dim in RemixedLinear where B << D)
    - Equivalent FLOPs to dense + gate overhead; no basis compression savings

    This makes it a fair structural ablation: same causal context conditioning
    and dual gating as RemixedLinear, but without the basis-space bottleneck.

    Compatible with the same gate_parameters/non_gate_parameters interface as
    RemixedLinear so it can be dropped in anywhere RemixedLinear is used.
    """

    def __init__(self, in_features, out_features, context_dim,
                 basis_size=None,          # ignored — kept for API compat with RemixedLinear
                 remixed_linear_kwargs=None,
                 scale_basis=None,         # ignored — no basis to scale
                 film_gate=False,          # if True: FiLM-style (scale+shift) instead of sigmoid
                 routing_scope='per_sequence',
                 **_ignored):
        super().__init__()
        if remixed_linear_kwargs is None:
            remixed_linear_kwargs = {}

        self.in_features = in_features
        self.out_features = out_features
        self.context_dim = context_dim
        self.use_context = remixed_linear_kwargs.get('use_context', True)
        self.use_basis_gate = remixed_linear_kwargs.get('use_basis_gate', True)
        self.use_output_gate = remixed_linear_kwargs.get('use_output_gate', True)
        self.gate_temperature = float(remixed_linear_kwargs.get('gate_temperature', 1.0))
        self.basis_gate_mode = remixed_linear_kwargs.get('basis_gate_mode', 'linear')
        self._film_gate_flag = film_gate
        # stage-1 gate output size: 2×D for FiLM (scale+shift), D for sigmoid
        _gate_out = out_features * 2 if film_gate else out_features

        # ── Dense projection (single matmul — no W_b × W_m factorization) ────
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.normal_(self.weight, std=0.02)
        self.bias = nn.Parameter(torch.zeros(out_features))
        # LayerNorm in OUTPUT space (D-dim)
        self.ln_out = nn.LayerNorm(out_features)
        # compat alias used by setup_optimizer's ln_basis probe
        self.ln_basis = self.ln_out

        # ── Stage-1: basis gate (D-dim sigmoid/FiLM gate from ctx) ───────────
        if self.use_context and self.use_basis_gate:
            if self.basis_gate_mode == 'mlp':
                self.basis_gate = nn.Sequential(
                    Linear(context_dim, context_dim // 2, bias=False),
                    nn.SiLU(),
                    Linear(context_dim // 2, _gate_out, bias=True),
                )
                nn.init.zeros_(self.basis_gate[-1].weight)
                nn.init.zeros_(self.basis_gate[-1].bias)
            else:  # 'linear' (default)
                self.basis_gate = Linear(context_dim, _gate_out, bias=True)
                nn.init.zeros_(self.basis_gate.weight)
                nn.init.zeros_(self.basis_gate.bias)
        else:
            self.basis_gate = None

        # ── Stage-2: low-rank output gate (rank-r from ctx) ──────────────────
        if self.use_context and self.use_output_gate:
            r = int(remixed_linear_kwargs.get('output_gate_rank', 8))
            self.output_gate_coeffs = Linear(context_dim, r, bias=True)
            self.output_gate_basis  = nn.Parameter(torch.randn(r, out_features) * 0.01)
            self.output_gate_scale  = nn.Parameter(torch.ones(1) * 0.1)
            nn.init.zeros_(self.output_gate_coeffs.weight)
            nn.init.zeros_(self.output_gate_coeffs.bias)
        else:
            self.output_gate_coeffs = None
            self.output_gate_basis  = None
            self.output_gate_scale  = None
        self._gate_stats = {}  # populated during forward() when self.training

    # ── Optimizer parameter grouping ──────────────────────────────────────────
    def gate_parameters(self):
        """Gate-side params → lower LR optimizer group."""
        if self.use_context:
            if self.basis_gate is not None:
                yield from self.basis_gate.parameters()
            if self.output_gate_coeffs is not None:
                yield self.output_gate_coeffs.weight
                if self.output_gate_coeffs.bias is not None:
                    yield self.output_gate_coeffs.bias
                yield self.output_gate_basis
                yield self.output_gate_scale

    def non_gate_parameters(self):
        """Structural params → Muon / normal-LR group."""
        yield self.weight
        yield self.bias

    # ── Forward ───────────────────────────────────────────────────────────────
    def forward(self, x, context_state, route_weights=None, context_gates=None, **_ignored):
        """
        x:             (B, T, in_features)
        context_state: (B, T, context_dim)  — causal context from CCLBlock
        """
        dtype = x.dtype

        # ── Step 1: Standard dense projection + LN in output space ───────────
        h = self.ln_out(
            F.linear(x, self.weight.to(dtype=dtype), self.bias.to(dtype=dtype))
            .to(dtype=self.ln_out.weight.dtype)
        ).to(dtype=dtype)

        if self.use_context and context_state is not None:
            ctx = context_state.to(dtype=dtype)

            # ── Step 2: Stage-1 gate (D-dim, in output space) ────────────────
            if self.use_basis_gate and self.basis_gate is not None:
                if context_gates is not None and 'basis_gate' in context_gates:
                    gate_logits = context_gates['basis_gate'].to(dtype=dtype)
                else:
                    gate_logits = self.basis_gate(ctx)

                if self._film_gate_flag:
                    scale_logits, shift = gate_logits.chunk(2, dim=-1)
                    gate1 = (1.0 + torch.tanh(scale_logits * 0.1)).to(dtype=dtype)
                    h = h * gate1 + shift.to(dtype=dtype)
                else:
                    gb_sig = torch.sigmoid(gate_logits / self.gate_temperature).to(dtype=dtype)
                    if self.training and not torch.compiler.is_compiling():
                        with torch.no_grad():
                            gb = gb_sig.float()
                            self._gate_stats['basis_mean'] = gb.mean().detach()
                            self._gate_stats['basis_std']  = gb.std().detach()
                            self._gate_stats['basis_dead'] = (gb < 0.10).float().mean().detach()
                            self._gate_stats['basis_sat']  = (gb > 0.90).float().mean().detach()
                    h = h * gb_sig

            # ── Step 3: Stage-2 low-rank output gate ─────────────────────────
            if self.use_output_gate and self.output_gate_coeffs is not None:
                if context_gates is not None and 'output_coeffs' in context_gates:
                    coeffs = context_gates['output_coeffs'].to(dtype=dtype)
                else:
                    coeffs = self.output_gate_coeffs(ctx)
                gate_logits2 = torch.matmul(coeffs, self.output_gate_basis.to(dtype=dtype))
                gate2 = 1.0 + torch.tanh(self.output_gate_scale.to(dtype=dtype) * gate_logits2)
                if self.training and not torch.compiler.is_compiling():
                    with torch.no_grad():
                        go = gate2.float()
                        self._gate_stats['out_mean'] = go.mean().detach()
                        self._gate_stats['out_std']  = go.std().detach()
                h = h * gate2

        return h


class OutputGatedLinear(nn.Module):
    """Minimal context-adaptive linear: single dense W + low-rank centered output gate.

    Forward:
        y = W x + bias                               # plain dense matmul (no factorization)
        g = 1 + tanh(s (W_c ctx) @ G)               # low-rank centered output gate
        return y * g

    ~1.01x FLOPs vs dense baseline (gate overhead negligible at r=8).
    Tests whether W_b@W_m factorization is needed for the quality gain seen in P25,
    or if a single W with only the output gate suffices.
    """

    def __init__(
        self,
        in_features,
        out_features,
        context_dim,
        basis_size=None,
        remixed_linear_kwargs=None,
        scale_basis=None,
        film_gate=False,
        routing_scope=None,
        **_ignored,
    ):
        super().__init__()
        if remixed_linear_kwargs is None:
            remixed_linear_kwargs = {}
        self.in_features  = in_features
        self.out_features = out_features
        self.use_context  = remixed_linear_kwargs.get('use_context', True)
        self.use_output_gate = remixed_linear_kwargs.get('use_output_gate', True)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.normal_(self.weight, std=0.02)
        self.bias = nn.Parameter(torch.zeros(out_features))
        if self.use_context and self.use_output_gate:
            r = int(remixed_linear_kwargs.get('output_gate_rank', 8))
            self.output_gate_coeffs = Linear(context_dim, r, bias=True)
            self.output_gate_basis  = nn.Parameter(torch.randn(r, out_features) * 0.01)
            self.output_gate_scale  = nn.Parameter(torch.ones(1) * 0.1)
            nn.init.zeros_(self.output_gate_coeffs.weight)
            nn.init.zeros_(self.output_gate_coeffs.bias)
        else:
            self.output_gate_coeffs = None
            self.output_gate_basis  = None
            self.output_gate_scale  = None
        self.ln_basis = None  # compat alias for setup_optimizer probes

    def gate_parameters(self):
        if self.use_context and self.output_gate_coeffs is not None:
            yield self.output_gate_coeffs.weight
            if self.output_gate_coeffs.bias is not None:
                yield self.output_gate_coeffs.bias
            yield self.output_gate_basis
            yield self.output_gate_scale

    def non_gate_parameters(self):
        yield self.weight
        yield self.bias

    def forward(self, x, context_state, route_weights=None, context_gates=None, **_ignored):
        dtype = x.dtype
        y = F.linear(x, self.weight.to(dtype=dtype), self.bias.to(dtype=dtype))
        if self.use_context and context_state is not None and self.output_gate_coeffs is not None:
            ctx = context_state.to(dtype=dtype)
            coeffs = self.output_gate_coeffs(ctx)
            gate = 1.0 + torch.tanh(
                self.output_gate_scale.to(dtype=dtype)
                * torch.matmul(coeffs, self.output_gate_basis.to(dtype=dtype))
            )
            y = y * gate
        return y


class DecoupledAdaptiveLinear(nn.Module):
    """Paradigm 1: split static/dynamic channels for gradient decoupling."""
    def __init__(self, in_features, out_features, context_dim, dynamic_ratio=0.25, basis_size=64, gate_rank=8, **_ignored):
        super().__init__()
        self.dyn_features = max(1, int(in_features * dynamic_ratio))
        self.stat_features = in_features - self.dyn_features
        if self.stat_features <= 0:
            self.stat_features = in_features - 1
            self.dyn_features = 1

        self.static_proj = Linear(self.stat_features, out_features, bias=True)
        self.dyn_basis = Linear(self.dyn_features, basis_size, bias=False)
        self.dyn_mix = nn.Parameter(torch.randn(out_features, basis_size) / (basis_size ** 0.5))
        self.gate_coeffs = Linear(context_dim, gate_rank, bias=True)
        self.gate_basis = nn.Parameter(torch.zeros(gate_rank, out_features))
        self.gate_scale = nn.Parameter(torch.ones(1) * 0.1)
        self.ln_basis = nn.Identity()
        nn.init.zeros_(self.gate_coeffs.weight)
        nn.init.zeros_(self.gate_coeffs.bias)

    @torch.no_grad()
    def reset_parameters(self, bound=None):
        """Owns this module's entire subtree; GPT.init_weights claims it.

        Same two defects as ResidualAdaptiveLinear: dyn_mix/gate_basis/gate_scale
        were never initialized, and zeroing both gate_coeffs and gate_basis is a
        fixed point (dL/dgate_basis is proportional to coeffs, dL/dgate_coeffs to
        gate_basis, so both stay zero forever). A unit coefficient bias keeps
        gate == 1 at init, since gate_basis is still zero, and makes it live.
        """
        for lin in (self.static_proj, self.dyn_basis):
            nn.init.xavier_uniform_(lin.weight)
            if lin.bias is not None:
                nn.init.zeros_(lin.bias)
        nn.init.normal_(self.dyn_mix, std=self.dyn_mix.shape[-1] ** -0.5)
        nn.init.zeros_(self.gate_basis)          # gate == 1 at init
        nn.init.zeros_(self.gate_coeffs.weight)
        nn.init.ones_(self.gate_coeffs.bias)     # coeffs == 1, so dL/dgate_basis != 0
        self.gate_scale.fill_(0.1)

    def gate_parameters(self):
        yield from self.gate_coeffs.parameters()
        yield self.gate_basis
        yield self.gate_scale

    def non_gate_parameters(self):
        yield from self.static_proj.parameters()
        yield self.dyn_basis.weight
        yield self.dyn_mix

    def forward(self, x, context_state, **_ignored):
        x_stat = x[..., :self.stat_features]
        x_dyn = x[..., self.stat_features:]
        out_stat = self.static_proj(x_stat)
        h_basis = self.dyn_basis(x_dyn)
        out_dyn = F.linear(h_basis, self.dyn_mix.to(dtype=x.dtype))
        if context_state is not None:
            coeffs = self.gate_coeffs(context_state.to(dtype=x.dtype))
            gate_logits = torch.matmul(coeffs, self.gate_basis.to(dtype=x.dtype))
            gate = 1.0 + torch.tanh(self.gate_scale.to(dtype=x.dtype) * gate_logits)
            out_dyn = out_dyn * gate
        return out_stat + out_dyn


class TuckerAdaptiveLinear(nn.Module):
    """Tucker-decomposed context routing: W_eff = Σ_k v_k(ctx) A G_k B^T."""
    def __init__(self, in_features, out_features, context_dim, tucker_rank=32, tucker_modes=8, **_ignored):
        super().__init__()
        r = max(1, int(tucker_rank))
        k = max(1, int(tucker_modes))
        self.A = nn.Parameter(torch.randn(in_features, r) * (in_features ** -0.5))
        self.B = nn.Parameter(torch.randn(out_features, r) * (out_features ** -0.5))
        self.core = nn.Parameter(torch.randn(r, r, k) * (r ** -0.5))
        self.ctx_router = Linear(context_dim, k, bias=True)
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.ln_basis = nn.Identity()
        nn.init.zeros_(self.ctx_router.weight)
        nn.init.zeros_(self.ctx_router.bias)

    @torch.no_grad()
    def reset_parameters(self, bound=None):
        """Owns this module's entire subtree; GPT.init_weights claims it.

        A/B/core/bias are bare nn.Parameters and held uninitialized storage
        before this existed. The router stays zero, so the core is mixed
        uniformly at init and every mode receives gradient.
        """
        nn.init.normal_(self.A, std=self.A.shape[0] ** -0.5)
        nn.init.normal_(self.B, std=self.B.shape[0] ** -0.5)
        nn.init.normal_(self.core, std=self.core.shape[0] ** -0.5)
        nn.init.zeros_(self.bias)
        nn.init.zeros_(self.ctx_router.weight)
        nn.init.zeros_(self.ctx_router.bias)

    def gate_parameters(self):
        yield from self.ctx_router.parameters()

    def non_gate_parameters(self):
        yield self.A
        yield self.B
        yield self.core
        yield self.bias

    def forward(self, x, context_state, **_ignored):
        h = torch.matmul(x, self.A.to(dtype=x.dtype))  # (B,T,r)
        if context_state is None:
            weights = torch.full((*h.shape[:2], self.core.shape[-1]), 1.0 / self.core.shape[-1], device=x.device, dtype=x.dtype)
        else:
            weights = F.softmax(self.ctx_router(context_state.to(dtype=x.dtype)), dim=-1)
        mixed = torch.einsum('btk,ijk->btij', weights, self.core.to(dtype=x.dtype))
        h2 = torch.einsum('bti,btij->btj', h, mixed)
        y = torch.matmul(h2, self.B.to(dtype=x.dtype).transpose(0, 1))
        return y + self.bias.to(dtype=x.dtype)


class SingularValueSteeringLinear(nn.Module):
    """SVS: context steers singular-value-like gains over fixed low-rank factors."""
    def __init__(self, in_features, out_features, context_dim, svs_rank=64, svs_eps=0.1, dcu_warmup_steps=0, **_ignored):
        super().__init__()
        r = max(1, min(int(svs_rank), in_features, out_features))
        self.V = nn.Parameter(torch.randn(in_features, r) * (in_features ** -0.5))
        self.U = nn.Parameter(torch.randn(out_features, r) * (out_features ** -0.5))
        self.sigma = nn.Parameter(torch.ones(r))
        self.ctx = Linear(context_dim, r, bias=True)
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.svs_eps = float(svs_eps)
        self.dcu_warmup_steps = max(0, int(dcu_warmup_steps))
        self._step = 0
        self.ln_basis = nn.Identity()
        nn.init.zeros_(self.ctx.weight)
        nn.init.zeros_(self.ctx.bias)

    @torch.no_grad()
    def reset_parameters(self, bound=None):
        """Owns this module's entire subtree; GPT.init_weights claims it.

        V/U/sigma/bias are bare nn.Parameters and held uninitialized storage
        before this existed. ctx stays zero so steer == 1 at init, which is the
        identity start this layer is designed around; sigma carries the scale so
        the branch is live from step 0.
        """
        nn.init.normal_(self.V, std=self.V.shape[0] ** -0.5)
        nn.init.normal_(self.U, std=self.U.shape[0] ** -0.5)
        nn.init.ones_(self.sigma)
        nn.init.zeros_(self.bias)
        nn.init.zeros_(self.ctx.weight)
        nn.init.zeros_(self.ctx.bias)

    def gate_parameters(self):
        yield from self.ctx.parameters()

    def non_gate_parameters(self):
        yield self.V
        yield self.U
        yield self.sigma
        yield self.bias

    def forward(self, x, context_state, **_ignored):
        h = torch.matmul(x, self.V.to(dtype=x.dtype))
        if context_state is None:
            steer = torch.ones_like(h)
        else:
            steer = 1.0 + self.svs_eps * torch.tanh(self.ctx(context_state.to(dtype=x.dtype)))
        if self.dcu_warmup_steps > 0:
            self._step += 1
            unlock = min(1.0, float(self._step) / float(self.dcu_warmup_steps))
            steer = 1.0 + unlock * (steer - 1.0)
        h = h * (self.sigma.to(dtype=x.dtype) * steer)
        y = torch.matmul(h, self.U.to(dtype=x.dtype).transpose(0, 1))
        return y + self.bias.to(dtype=x.dtype)


class VQAdaptiveLinear(nn.Module):
    """VQ regime routing: dense base + regime-selected full-rank delta."""
    def __init__(self, in_features, out_features, context_dim, vq_codes=8, vq_temperature=1.0, **_ignored):
        super().__init__()
        self.base = Linear(in_features, out_features, bias=True)
        self.K = max(1, int(vq_codes))
        self.temperature = max(float(vq_temperature), 1e-6)
        self.codebook = nn.Parameter(torch.randn(self.K, context_dim) * (context_dim ** -0.5))
        self.delta = nn.Parameter(torch.zeros(self.K, out_features, in_features))
        self.ln_basis = nn.Identity()

    @torch.no_grad()
    def reset_parameters(self, bound=None):
        """Owns this module's entire subtree; GPT.init_weights claims it.

        codebook and delta are bare nn.Parameters and held uninitialized storage
        before this existed. delta=0 gives an exact dense start and still leaves
        dL/ddelta alive, since it is multiplied by the code assignment and x
        rather than by another zero.
        """
        nn.init.xavier_uniform_(self.base.weight)
        if self.base.bias is not None:
            nn.init.zeros_(self.base.bias)
        nn.init.normal_(self.codebook, std=self.codebook.shape[-1] ** -0.5)
        nn.init.zeros_(self.delta)

    def gate_parameters(self):
        yield self.codebook

    def non_gate_parameters(self):
        yield from self.base.parameters()
        yield self.delta

    def forward(self, x, context_state, **_ignored):
        y = self.base(x)
        if context_state is None:
            return y
        ctx = context_state.to(dtype=x.dtype)
        B, T, C = ctx.shape
        flat = ctx.reshape(B * T, C)
        dist = torch.cdist(flat, self.codebook.to(dtype=x.dtype))
        idx = torch.argmin(dist, dim=-1)
        hard = F.one_hot(idx, num_classes=self.K).to(dtype=x.dtype)
        soft = F.softmax(-dist / self.temperature, dim=-1)
        w = hard + (soft - soft.detach())
        w = w.view(B, T, self.K)
        w_delta = torch.einsum('btk,koi->btoi', w, self.delta.to(dtype=x.dtype))
        y_delta = torch.einsum('btoi,bti->bto', w_delta, x)
        return y + y_delta


class FrozenSubspaceIndexedLinear(nn.Module):
    """Frozen Subspace Indexing (FSI): context-dependent input transformation
    with ZERO trainable routing parameters.

    y = base_dense(x - 2 * sum_k w_k * v_k * (v_k^T x))

    Uses K frozen Householder reflections (unit vectors v_k) with soft routing
    weights w_k from a frozen random projection of the attention signal.

    Key properties:
    - base_dense is STANDARD nn.Linear — identical gradient dynamics to baseline
    - Reflectors are FROZEN (registered buffers) — zero gradient interference
    - Routing uses FROZEN random projections — no controller to train
    - Memory: O(K*D) reflector buffer, NOT O(K*D²) rotation matrices
    - Compute: O(B*T*K*D) — linear in D, not quadratic
    - torch.compile friendly: no gather, no einsum over D², just matmuls + element-wise

    Compatible with setup_optimizer: gate_parameters() yields nothing (no gates),
    non_gate_parameters() yields base weight/bias. ln_basis = Identity for compat.
    """
    def __init__(self, in_features, out_features, n_rotations=8, selector_dim=64, signal_dim=None, **_ignored):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_rotations = n_rotations
        self.signal_dim = signal_dim or in_features
        self.base = Linear(in_features, out_features, bias=True)
        self.ln_basis = nn.Identity()  # compat with setup_optimizer

        # FROZEN: K random unit vectors defining Householder reflections
        # P_k = I - 2 * v_k * v_k^T  (orthogonal reflection through hyperplane ⊥ v_k)
        # Each reflection provides a distinct "view" of the input space
        vecs = torch.randn(n_rotations, in_features)
        vecs = F.normalize(vecs, dim=-1)  # unit vectors
        self.register_buffer('reflectors', vecs)  # (K, D_in)

        # FROZEN: random projection for routing signal extraction
        # signal (signal_dim) → selector_dim → K-dim soft weights
        self.register_buffer('selector_proj',
            torch.randn(self.signal_dim, selector_dim) / (self.signal_dim ** 0.5))
        self.register_buffer('selector',
            torch.randn(selector_dim, n_rotations) / (selector_dim ** 0.5))

    @torch.no_grad()
    def reset_parameters(self, bound=None):
        """Owns this module's entire subtree; GPT.init_weights claims it.

        Everything this layer routes with lives in buffers, and to_empty()
        empties buffers exactly like parameters. Before this existed the
        'frozen orthogonal rotations' were uninitialized memory: the reflectors
        were not unit vectors, so I - 2vv^T was not a reflection, and the frozen
        routing projections were arbitrary. Nothing about the layer's premise
        held in an actual run.
        """
        nn.init.xavier_uniform_(self.base.weight)
        if self.base.bias is not None:
            nn.init.zeros_(self.base.bias)
        vecs = torch.randn_like(self.reflectors)
        self.reflectors.copy_(F.normalize(vecs, dim=-1))   # unit => genuine reflections
        self.selector_proj.copy_(
            torch.randn_like(self.selector_proj) / (self.selector_proj.shape[0] ** 0.5))
        self.selector.copy_(
            torch.randn_like(self.selector) / (self.selector.shape[0] ** 0.5))

    def gate_parameters(self):
        """No gate parameters — routing is entirely frozen."""
        return iter([])

    def non_gate_parameters(self):
        """Only the base dense layer is trained."""
        yield self.base.weight
        if self.base.bias is not None:
            yield self.base.bias

    def forward(self, x, context_state, **_ignored):
        """
        x: (B, T, D_in)
        context_state: (B, T, signal_dim) — detached attention output for routing.
                       If None, uses pure base projection (identity transform).
        """
        if context_state is None:
            return self.base(x)

        dtype = x.dtype
        signal = context_state.to(dtype=dtype).detach()

        # Frozen routing: signal → soft weights over K reflections
        routing = (signal @ self.selector_proj.to(dtype)) @ self.selector.to(dtype)  # (B, T, K)
        weights = torch.softmax(routing, dim=-1)  # (B, T, K)

        # Soft-weighted Householder reflections (compile-friendly, O(BTK*D))
        # dot_k = v_k^T x  →  dots: (B, T, K)
        dots = torch.einsum('btd,kd->btk', x, self.reflectors.to(dtype))

        # weighted_dots = sum_k w_k * dot_k * v_k  →  delta: (B, T, D_in)
        # This is the soft mixture of reflections applied to x
        weighted_dots = weights * dots  # (B, T, K)
        delta = torch.einsum('btk,kd->btd', weighted_dots, self.reflectors.to(dtype))

        # x_transformed = x - 2 * delta  (soft mixture of Householder reflections)
        x_transformed = x - 2.0 * delta

        return self.base(x_transformed)


class AttentionEntropyStratifiedLinear(nn.Module):
    """Attention-Entropy Stratified Projection (AESP): per-stratum dense
    projection selected by attention entropy.

    y = base(x) + alpha_k * (x @ U_k) @ V_k    where k = stratum(entropy)

    Uses attention entropy (already computed, zero extra FLOPs for routing)
    to partition positions into K strata. Each stratum has a tiny low-rank
    delta (rank 4 by default) with FROZEN scaling alpha_k.

    Key properties:
    - Stratum 0 (highest entropy = weakest context) has alpha=0 → EXACT dense baseline
    - Higher strata allow progressively more specialization (alpha grows)
    - Deltas are tiny rank-4 → minimal parameter overhead
    - Routing (entropy → stratum) is free (entropy computed in attention already)
    - Frozen alphas prevent the network from "turning off" specialization

    Compatible with setup_optimizer: gate_parameters() yields nothing (routing is
    frozen), non_gate_parameters() yields base + U/V + scale.
    """
    def __init__(self, in_features, out_features, n_strata=4, delta_rank=4, **_ignored):
        super().__init__()
        self.n_strata = n_strata
        self.delta_rank = delta_rank
        self.base = Linear(in_features, out_features, bias=True)
        self.ln_basis = nn.Identity()  # compat

        # Per-stratum low-rank deltas (V=0 at init → zero delta)
        self.U = nn.ParameterList([
            nn.Parameter(torch.empty(in_features, delta_rank))
            for _ in range(n_strata)
        ])
        self.V = nn.ParameterList([
            nn.Parameter(torch.zeros(delta_rank, out_features))
            for _ in range(n_strata)
        ])
        for u in self.U:
            nn.init.kaiming_uniform_(u, a=0.01)

        # FROZEN scaling: stratum 0 = zero delta (pure baseline)
        # Higher strata get progressively more specialization budget
        alphas = torch.linspace(0, 0.2, n_strata)
        alphas[0] = 0.0  # stratum 0 is always pure dense baseline
        self.register_buffer('alphas', alphas)

    @torch.no_grad()
    def reset_parameters(self, bound=None):
        """Owns this module's entire subtree; GPT.init_weights claims it.

        The U/V ParameterLists and the alphas buffer held uninitialized storage
        before this existed; to_empty() replaces buffer storage too, so the
        stratum mixing coefficients were garbage rather than the intended ramp.
        Only V is zeroed, so the deltas start at zero with dL/dV alive.
        """
        nn.init.xavier_uniform_(self.base.weight)
        if self.base.bias is not None:
            nn.init.zeros_(self.base.bias)
        for u in self.U:
            nn.init.kaiming_uniform_(u, a=0.01)
        for v in self.V:
            nn.init.zeros_(v)
        self.alphas.copy_(torch.linspace(0, 0.2, self.n_strata, device=self.alphas.device))

    def gate_parameters(self):
        """No gate parameters — routing/scaling is frozen."""
        return iter([])

    def non_gate_parameters(self):
        """Base projection + per-stratum U/V deltas."""
        yield self.base.weight
        if self.base.bias is not None:
            yield self.base.bias
        for u in self.U:
            yield u
        for v in self.V:
            yield v

    def forward(self, x, context_state, **_ignored):
        """
        x: (B, T, D)
        context_state: (B, T) — attention entropy per position (scalar).
                       If None or wrong shape, uses pure base projection.
        """
        y = self.base(x)

        if context_state is None:
            return y

        dtype = x.dtype
        entropy = context_state.to(dtype=dtype)

        # Handle case where context_state is (B, T, D) — take mean to get scalar
        if entropy.ndim == 3:
            entropy = entropy.mean(dim=-1)  # (B, T)

        # Quantize entropy into strata via percentile boundaries
        with torch.no_grad():
            q_levels = torch.linspace(0, 1, self.n_strata + 1, device=x.device)[1:-1]
            boundaries = torch.quantile(entropy.float().flatten(), q_levels)
            stratum = torch.bucketize(entropy.float(), boundaries)  # (B, T) in [0, K-1]

        # Add stratum-specific deltas (skip stratum 0 which has alpha=0)
        for k in range(1, self.n_strata):
            alpha_k = self.alphas[k].to(dtype)
            if alpha_k == 0:
                continue
            mask = (stratum == k).unsqueeze(-1).to(dtype)  # (B, T, 1)
            delta = (x @ self.U[k].to(dtype)) @ self.V[k].to(dtype)  # (B, T, out)
            y = y + mask * alpha_k * delta

        return y


class CausalKernelLinear(nn.Module):
    """Causal Kernel Reparameterization (CKR): position-dependent mixing
    of K parallel dense branches via tiny causal conv1d.

    Training:  y_t = sum_k  w_k(t) · (W_k · x_t)
    Inference: y_t = W_merged(t) · x_t  (structural reparameterization)

    Phase 13 enhancements:
    - 13a: Multi-channel position signal (C_pos > 1) for multi-scale temporal patterns
    - 13c: Optional frozen content hash bias for document-type fingerprinting

    Key properties:
    - w_k(t) depends on POSITION (+ optional frozen content hash), not learned content
    - No chicken-and-egg: W_k learn features, w_k learn positions, independently
    - Position weights from causal conv capture local structure naturally
    - At init: conv output ≈ 0 → softmax(0) = 1/K → equal branch weighting
    - Reparameterizable at inference for zero overhead

    Compatible with setup_optimizer: gate_parameters() yields position signal +
    conv weights (small), non_gate_parameters() yields branch projections.
    """
    def __init__(self, in_features, out_features, n_branches=4, kernel_size=64,
                 max_seq_len=2048, n_pos_channels=1, content_bias_scale=0.0,
                 signal_dim=None, temperature=1.0,
                 ortho_init=False, branch_dropout=0.0, **_ignored):
        super().__init__()
        self.n_branches = n_branches
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_size = kernel_size
        self.content_bias_scale = content_bias_scale
        # Temperature as a buffer tensor (NOT Python float) to avoid torch.compile
        # guard breaks when annealing. Buffer tensors are treated as dynamic values.
        # _init_temperature (Python float) survives to_empty() for re-init.
        self._init_temperature = temperature
        self.register_buffer('_temperature', torch.tensor([temperature]), persistent=False)
        self.branch_dropout = branch_dropout  # 17E: probability of dropping each branch
        self.ln_basis = nn.Identity()  # compat

        # K parallel projection branches
        self.branches = nn.ModuleList([
            Linear(in_features, out_features, bias=(i == 0))
            for i in range(n_branches)
        ])
        # 17D: Orthogonal initialization for branch diversity
        self._ortho_init = ortho_init
        if ortho_init:
            for branch in self.branches:
                nn.init.orthogonal_(branch.weight)

        # 13a: Multi-channel position signal (C_pos channels → conv → K branches)
        # C_pos > 1 allows the conv to learn multi-scale temporal patterns
        self.n_pos_channels = n_pos_channels
        self.pos_signal = nn.Parameter(torch.zeros(1, n_pos_channels, max_seq_len))
        self.branch_conv = nn.Conv1d(
            n_pos_channels, n_branches, kernel_size=kernel_size,
            padding=kernel_size - 1, bias=True  # causal: we trim the right padding
        )
        # Init: conv output ≈ 0 → softmax(0) = 1/K → equal branch weighting
        nn.init.zeros_(self.branch_conv.weight)
        nn.init.zeros_(self.branch_conv.bias)

        # 13c: Frozen content hash for document-type fingerprinting
        # Projects detached input → K-dim frozen logit bias
        if content_bias_scale > 0:
            sig_dim = signal_dim or in_features
            self.register_buffer('content_proj',
                torch.randn(sig_dim, n_branches) / (sig_dim ** 0.5))
        else:
            self.content_proj = None

    def gate_parameters(self):
        """Position signal and conv weights — the 'routing' side."""
        yield self.pos_signal
        yield from self.branch_conv.parameters()

    def non_gate_parameters(self):
        """Branch projection weights — the 'feature' side."""
        for branch in self.branches:
            yield from branch.parameters()

    def forward(self, x, context_state=None, **kwargs):
        """
        x: (B, T, D). context_state: optional (B, T, signal_dim) for content bias.
        """
        B, T, D = x.shape
        dtype = x.dtype

        # Position-dependent branch weights (NOT content-dependent)
        sig = self.pos_signal[:, :, :T].expand(B, -1, -1)  # (B, C_pos, T)
        raw_weights = self.branch_conv(sig.to(dtype=self.branch_conv.weight.dtype))[:, :, :T]  # (B, K, T) — causal trim
        logits = raw_weights.to(dtype)  # (B, K, T)

        # 13c: Add frozen content hash bias (if enabled)
        if self.content_proj is not None and context_state is not None:
            # Frozen projection of detached context → (B, T, K) logit bias
            ctx = context_state.to(dtype=dtype).detach()
            content_logits = ctx @ self.content_proj.to(dtype)  # (B, T, K)
            logits = logits + self.content_bias_scale * content_logits.permute(0, 2, 1)  # (B, K, T)

        w = F.softmax(logits / self._temperature.to(dtype), dim=1)  # (B, K, T)

        # 17E: Branch dropout — randomly zero some branches during training
        if self.training and self.branch_dropout > 0 and self.n_branches > 1:
            # Create per-branch mask (same for all positions in batch)
            mask = torch.ones(self.n_branches, device=w.device, dtype=dtype)
            drop_idx = torch.randint(0, self.n_branches, (1,)).item()
            mask[drop_idx] = 0.0
            w = w * mask.view(1, -1, 1)
            w = w / (w.sum(dim=1, keepdim=True) + 1e-8)  # renormalize

        # Parallel branch computation
        y = torch.zeros(B, T, self.out_features, device=x.device, dtype=dtype)
        for k in range(self.n_branches):
            y_k = self.branches[k](x)  # (B, T, out)
            y = y + w[:, k, :].unsqueeze(-1) * y_k  # position-weighted sum

        return y


class GradientIsolatedDeltaLinear(nn.Module):
    """Phase 14a: Gradient-Isolated Additive Delta (GIAD).

    y = base(x) + delta_net(x)

    Approximate gradient isolation via initialization:
    - delta_up.weight = 0 at init → delta output is exactly zero
    - As training progresses, delta_up weights grow from zero,
      gradually introducing the delta path's contribution
    - Early training is virtually identical to the dense baseline

    Note: no separate scale parameter — nn.Parameter(torch.zeros(1))
    is float32, which causes dtype promotion (float32 × bf16 → float32)
    under torch.compile + fp8/bf16 autocast, crashing FlashAttention.
    """
    def __init__(self, in_features, out_features, delta_rank=32, **_ignored):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.base = Linear(in_features, out_features, bias=True)
        self.ln_basis = nn.Identity()  # compat with setup_optimizer

        # Low-rank bottleneck: x → rank-r → output (delta_up is zero-init)
        self.delta_down = Linear(in_features, delta_rank, bias=False)
        self.delta_up = Linear(delta_rank, out_features, bias=False)

    def gate_parameters(self):
        """Delta path params are the 'gate' side (lower LR)."""
        yield self.delta_down.weight
        yield self.delta_up.weight

    def non_gate_parameters(self):
        """Base projection weights — standard dense, unaffected by delta."""
        yield self.base.weight
        if self.base.bias is not None:
            yield self.base.bias

    def forward(self, x, context_state=None, **_ignored):
        """
        x: (B, T, D_in). context_state is ignored.
        """
        y = self.base(x)
        # delta_up zero-init → starts at 0, grows naturally during training
        delta = self.delta_up(F.gelu(self.delta_down(x)))
        return y + delta


class PositionalScalarGatedLinear(nn.Module):
    """Phase 14b: Positional Scalar Gating (PSG).

    y_t = W · (s(t) ⊙ x_t)

    where s(t) is a D_in-dimensional position-dependent scaling vector from a
    tiny causal conv1d over a learned positional signal.

    Effective weight at position t: W_eff(t) = W · diag(s(t))
    This is a rank-D perturbation — can modulate every element of W_eff —
    yet costs only O(conv_params) extra, NOT O(K × D × D) like CKR.

    Init: s(t) = 1.0 for all t → y = W · x (identity with dense baseline).
    Position-only routing: no content-dependent gates → no friction tax.
    """
    def __init__(self, in_features, out_features, kernel_size=64,
                 max_seq_len=2048, **_ignored):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.base = Linear(in_features, out_features, bias=True)
        self.ln_basis = nn.Identity()  # compat

        # Learned 1D positional signal → causal conv → D_in-dim scaling per position
        self.pos_signal = nn.Parameter(torch.zeros(1, 1, max_seq_len))
        self.scale_conv = nn.Conv1d(
            1, in_features, kernel_size=kernel_size,
            padding=kernel_size - 1, bias=True
        )
        # Init: conv output ≈ 0 → s(t) = 1.0 (identity scaling)
        nn.init.zeros_(self.scale_conv.weight)
        nn.init.zeros_(self.scale_conv.bias)
        # Controls the maximum deviation from identity
        self.scale_magnitude = nn.Parameter(torch.tensor(0.01))

    @torch.no_grad()
    def reset_parameters(self, bound=None):
        """Owns this module's entire subtree; GPT.init_weights claims it.

        pos_signal and scale_magnitude are bare nn.Parameters and held
        uninitialized storage before this existed.
        """
        nn.init.xavier_uniform_(self.base.weight)
        if self.base.bias is not None:
            nn.init.zeros_(self.base.bias)
        nn.init.normal_(self.pos_signal, std=0.01)
        nn.init.zeros_(self.scale_conv.weight)
        if self.scale_conv.bias is not None:
            nn.init.zeros_(self.scale_conv.bias)
        self.scale_magnitude.fill_(0.01)

    def gate_parameters(self):
        """Position signal + conv = routing side."""
        yield self.pos_signal
        yield from self.scale_conv.parameters()
        yield self.scale_magnitude

    def non_gate_parameters(self):
        """Base projection weights — standard dense."""
        yield self.base.weight
        if self.base.bias is not None:
            yield self.base.bias

    def forward(self, x, context_state=None, **_ignored):
        """
        x: (B, T, D_in). context_state is ignored (position-only scaling).
        """
        B, T, D = x.shape
        dtype = x.dtype

        # Position-dependent per-channel scaling
        sig = self.pos_signal[:, :, :T].expand(B, -1, -1)  # (B, 1, T)
        raw = self.scale_conv(sig.to(dtype=self.scale_conv.weight.dtype))[:, :, :T]  # (B, D_in, T)
        # s(t) = 1 + magnitude * tanh(raw) → starts at 1.0, bounded perturbation
        s = 1.0 + self.scale_magnitude * torch.tanh(raw.to(dtype))  # (B, D_in, T)
        x_scaled = x * s.permute(0, 2, 1)  # (B, T, D_in)
        return self.base(x_scaled)


class SplitStreamLinear(nn.Module):
    """Phase 14c: SplitStream — Decoupled channel split + CKR dynamic path.

    y = concat(
        W_static · x[:, :, :D_s],       # static channels: pure dense
        CKR_branches(x[:, :, D_s:])      # dynamic channels: position-dependent
    )

    Combines the two winning principles:
    - Decoupled (Phase 11): Reduce gradient competition via channel partitioning
    - CKR (Phase 12): Position-only routing avoids content-gradient interference

    Static path (majority ~75%): exact dense baseline gradient dynamics.
    Dynamic path (minority ~25%): CKR-style position routing on fewer channels.
    Zero parameter sharing between paths.
    """
    def __init__(self, in_features, out_features, dynamic_ratio=0.25,
                 n_branches=2, kernel_size=64, max_seq_len=2048, **_ignored):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ln_basis = nn.Identity()  # compat

        # Channel split dimensions
        self.d_dynamic = max(1, int(in_features * dynamic_ratio))
        self.d_static = in_features - self.d_dynamic
        d_out_dynamic = max(1, int(out_features * dynamic_ratio))
        d_out_static = out_features - d_out_dynamic

        # Static path: plain dense projection
        self.static_proj = Linear(self.d_static, d_out_static, bias=True)

        # Dynamic path: CKR-style multi-branch with position routing
        self.n_branches = n_branches
        self.dynamic_branches = nn.ModuleList([
            Linear(self.d_dynamic, d_out_dynamic, bias=(i == 0))
            for i in range(n_branches)
        ])
        self.pos_signal = nn.Parameter(torch.zeros(1, 1, max_seq_len))
        self.branch_conv = nn.Conv1d(
            1, n_branches, kernel_size=kernel_size,
            padding=kernel_size - 1, bias=True
        )
        nn.init.zeros_(self.branch_conv.weight)
        nn.init.zeros_(self.branch_conv.bias)

    @torch.no_grad()
    def reset_parameters(self, bound=None):
        """Owns this module's entire subtree; GPT.init_weights claims it.

        pos_signal and the branch conv held uninitialized storage before this
        existed, so the position-routed branch weights were arbitrary.
        """
        nn.init.xavier_uniform_(self.static_proj.weight)
        if self.static_proj.bias is not None:
            nn.init.zeros_(self.static_proj.bias)
        for br in self.dynamic_branches:
            nn.init.xavier_uniform_(br.weight)
            if br.bias is not None:
                nn.init.zeros_(br.bias)
        nn.init.normal_(self.pos_signal, std=0.01)
        nn.init.zeros_(self.branch_conv.weight)
        if self.branch_conv.bias is not None:
            nn.init.zeros_(self.branch_conv.bias)

    def gate_parameters(self):
        """Position signal + conv + dynamic branch weights = routing side."""
        yield self.pos_signal
        yield from self.branch_conv.parameters()
        for branch in self.dynamic_branches:
            yield from branch.parameters()

    def non_gate_parameters(self):
        """Static projection weights — pure dense, zero interference."""
        yield self.static_proj.weight
        if self.static_proj.bias is not None:
            yield self.static_proj.bias

    def forward(self, x, context_state=None, **_ignored):
        """
        x: (B, T, D_in). context_state is ignored (position-only dynamic path).
        """
        B, T, D = x.shape
        dtype = x.dtype

        # Channel split
        x_static = x[:, :, :self.d_static]
        x_dynamic = x[:, :, self.d_static:]

        # Static path: standard dense projection
        y_static = self.static_proj(x_static)

        # Dynamic path: CKR-style position-dependent branch mixing
        sig = self.pos_signal[:, :, :T].expand(B, -1, -1)
        raw_w = self.branch_conv(sig.to(dtype=self.branch_conv.weight.dtype))[:, :, :T]
        w = F.softmax(raw_w.to(dtype), dim=1)  # (B, K, T)

        y_dynamic = torch.zeros(B, T, self.dynamic_branches[0].weight.shape[0],
                                device=x.device, dtype=dtype)
        for k in range(self.n_branches):
            y_k = self.dynamic_branches[k](x_dynamic)
            y_dynamic = y_dynamic + w[:, k, :].unsqueeze(-1) * y_k

        return torch.cat([y_static, y_dynamic], dim=-1)


class LoKRLinear(nn.Module):
    """Phase 15: Low-rank Kernel Reparameterization (LoKR).

    W_eff(t) = W + Σ_k  w_k(t) · U_k @ V_k^T

    Combines CKR's key insight (position-dependent multi-branch mixing) with
    dramatically lower parameter cost by using low-rank perturbations instead
    of full-rank duplicate branches.

    Key properties:
    - W is shared full-rank base → standard dense gradient (Pattern 1 ✓)
    - K low-rank perturbation branches (U_k, V_k) provide diversity (Pattern 2 ✓)
    - Position-only routing via causal conv (Pattern 3 ✓)
    - At init: V_k=0 → perturbation=0 → exact dense baseline
    - 21% param overhead vs CKR's 300% (at K=8, r=16 for D=768)

    Computation:
        y = W·x + Σ_k w_k(t) · U_k @ (V_k @ x)
    Each V_k@x is (B,T,r), then U_k@result is (B,T,D_out). Very efficient.
    """
    def __init__(self, in_features, out_features, n_branches=8, rank=16,
                 kernel_size=64, max_seq_len=2048, **_ignored):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_branches = n_branches
        self.rank = rank
        self.ln_basis = nn.Identity()  # compat

        # Shared full-rank base projection
        self.base = Linear(in_features, out_features, bias=True)

        # K low-rank perturbation branches: U_k (out×r) and V_k (r×in)
        self.lora_down = nn.ParameterList([
            nn.Parameter(torch.zeros(rank, in_features))   # V_k: r × D_in
            for _ in range(n_branches)
        ])
        self.lora_up = nn.ParameterList([
            nn.Parameter(torch.zeros(out_features, rank))  # U_k: D_out × r
            for _ in range(n_branches)
        ])
        # V_k = 0 at init → perturbation = 0 → exact dense baseline
        # U_k gets xavier init for gradient flow once V_k starts learning

        # Position-dependent branch weights (same as CKR)
        self.pos_signal = nn.Parameter(torch.zeros(1, 1, max_seq_len))
        self.branch_conv = nn.Conv1d(
            1, n_branches, kernel_size=kernel_size,
            padding=kernel_size - 1, bias=True
        )
        nn.init.zeros_(self.branch_conv.weight)
        nn.init.zeros_(self.branch_conv.bias)

    @torch.no_grad()
    def reset_parameters(self, bound=None):
        """Owns this module's entire subtree; GPT.init_weights claims it.

        The lora_down/lora_up ParameterLists held uninitialized storage before
        this existed, and the constructor zeroed BOTH factors, which its own
        comment says was not the intent ("U_k gets xavier init for gradient flow
        once V_k starts learning"). Two zeroed factors is a fixed point: the
        perturbation and every gradient into it stay zero for the whole run.
        Zeroing only lora_down keeps the exact dense start and leaves dL/d(down)
        alive through lora_up.
        """
        nn.init.xavier_uniform_(self.base.weight)
        if self.base.bias is not None:
            nn.init.zeros_(self.base.bias)
        for down in self.lora_down:
            nn.init.zeros_(down)               # perturbation == 0 at init
        for up in self.lora_up:
            nn.init.xavier_uniform_(up)        # nonzero, so dL/d(down) != 0
        nn.init.normal_(self.pos_signal, std=0.01)
        nn.init.zeros_(self.branch_conv.weight)
        if self.branch_conv.bias is not None:
            nn.init.zeros_(self.branch_conv.bias)

    def gate_parameters(self):
        """Position signal + conv = routing side."""
        yield self.pos_signal
        yield from self.branch_conv.parameters()

    def non_gate_parameters(self):
        """Base projection + low-rank perturbation matrices = feature side."""
        yield self.base.weight
        if self.base.bias is not None:
            yield self.base.bias
        for v in self.lora_down:
            yield v
        for u in self.lora_up:
            yield u

    def forward(self, x, context_state=None, **_ignored):
        """
        x: (B, T, D_in). context_state ignored (position-only routing).
        """
        B, T, D = x.shape
        dtype = x.dtype

        # Standard dense base
        y = self.base(x)

        # Position-dependent branch weights
        sig = self.pos_signal[:, :, :T].expand(B, -1, -1)
        raw_w = self.branch_conv(sig.to(dtype=self.branch_conv.weight.dtype))[:, :, :T]
        w = F.softmax(raw_w.to(dtype), dim=1)  # (B, K, T)

        # Low-rank position-dependent perturbations
        delta = torch.zeros_like(y)
        for k in range(self.n_branches):
            # V_k @ x: (B, T, D_in) → (B, T, r)
            h_k = F.linear(x, self.lora_down[k].to(dtype))  # (B, T, r)
            # U_k @ h_k: (B, T, r) → (B, T, D_out)
            d_k = F.linear(h_k, self.lora_up[k].to(dtype))  # (B, T, D_out)
            delta = delta + w[:, k, :].unsqueeze(-1) * d_k

        return y + delta


class CausalOutputMixer(nn.Module):
    """Phase 16C: Causal Output Mixer (COM).

    y = Linear(x) + gate(pos) ⊙ causal_depthwise_conv(Linear(x))

    Fundamentally different from all previous approaches:
    - The Linear layer is COMPLETELY UNMODIFIED (true zero friction)
    - Position-dependent causal convolution adds sequence-level mixing
    - Each position can selectively mix with previous positions
    - Gate starts at 0 (zero-init bias) → pure dense baseline at init

    Similar to what works in Mamba/RWKV: conv + gating on features.
    ~0.5% parameter overhead vs CKR's 300%.
    """
    def __init__(self, in_features, out_features, kernel_size=32,
                 max_seq_len=2048, **_ignored):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_size = kernel_size
        self.ln_basis = nn.Identity()  # compat with setup_optimizer

        # Standard dense projection (unmodified)
        self.base = Linear(in_features, out_features, bias=True)

        # Causal depthwise convolution on output features
        # Groups = out_features → each channel convolved independently
        self.causal_conv = nn.Conv1d(
            out_features, out_features, kernel_size=kernel_size,
            padding=kernel_size - 1,  # causal: trim right side
            groups=out_features, bias=False
        )
        nn.init.zeros_(self.causal_conv.weight)

        # Position-dependent gate: learned pos signal → per-channel gate
        self.pos_signal = nn.Parameter(torch.zeros(1, 1, max_seq_len))
        self.gate_conv = nn.Conv1d(
            1, out_features, kernel_size=kernel_size,
            padding=kernel_size - 1, bias=True
        )
        nn.init.zeros_(self.gate_conv.weight)
        nn.init.constant_(self.gate_conv.bias, -3.0)  # sigmoid(-3) ≈ 0.05 → nearly off at init

    @torch.no_grad()
    def reset_parameters(self, bound=None):
        """Owns this module's entire subtree; GPT.init_weights claims it.

        pos_signal and both convs held uninitialized storage before this
        existed. The -3.0 gate bias is the intended near-off start.
        """
        nn.init.xavier_uniform_(self.base.weight)
        if self.base.bias is not None:
            nn.init.zeros_(self.base.bias)
        nn.init.zeros_(self.causal_conv.weight)
        if self.causal_conv.bias is not None:
            nn.init.zeros_(self.causal_conv.bias)
        nn.init.normal_(self.pos_signal, std=0.01)
        nn.init.zeros_(self.gate_conv.weight)
        if self.gate_conv.bias is not None:
            nn.init.constant_(self.gate_conv.bias, -3.0)

    def gate_parameters(self):
        """Routing side: position signal + gate conv + causal conv."""
        yield self.pos_signal
        yield from self.gate_conv.parameters()
        yield from self.causal_conv.parameters()

    def non_gate_parameters(self):
        """Feature side: base projection weights."""
        yield from self.base.parameters()

    def forward(self, x, context_state=None, **_ignored):
        """
        x: (B, T, D_in). context_state ignored.
        """
        B, T, _ = x.shape
        dtype = x.dtype

        # Standard dense projection
        y = self.base(x)  # (B, T, D_out)

        # Causal depthwise conv on output (channels-first), cast weights to match input dtype
        y_t = y.transpose(1, 2)  # (B, D_out, T)
        conv_out = F.conv1d(
            y_t, self.causal_conv.weight.to(dtype), bias=None,
            padding=self.kernel_size - 1, groups=y_t.shape[1]
        )[:, :, :T]  # causal trim → (B, D_out, T)

        # Position-dependent gate (cast gate_conv weights to match)
        sig = self.pos_signal[:, :, :T].expand(B, -1, -1).to(dtype)  # (B, 1, T)
        gate = torch.sigmoid(
            F.conv1d(sig, self.gate_conv.weight.to(dtype),
                     self.gate_conv.bias.to(dtype),
                     padding=self.kernel_size - 1)[:, :, :T]
        )  # (B, D_out, T)

        # Gate ⊙ conv_out → residual
        mixed = gate * conv_out  # (B, D_out, T)
        return y + mixed.transpose(1, 2)  # (B, T, D_out)


class PositionGatedResidual(nn.Module):
    """Phase 17C: Position-Gated Residual (PGR).

    y = W₀x + g(pos) · W₁x

    Two full-rank branches with explicit base+delta structure.
    g(pos) is a scalar gate per position from causal conv on learned position signal.
    W₁ zero-init → starts as pure W₀ (dense baseline).

    Compatible with setup_optimizer: gate_parameters() yields pos_signal + gate_conv,
    non_gate_parameters() yields W₀ and W₁.
    """
    def __init__(self, in_features, out_features, kernel_size=64, max_seq_len=2048, **_ignored):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_size = kernel_size
        self.ln_basis = nn.Identity()

        # Base projection (standard init)
        self.base = Linear(in_features, out_features, bias=True)
        # Delta projection (zero-init for dense-baseline start)
        self.delta = Linear(in_features, out_features, bias=False)
        nn.init.zeros_(self.delta.weight)

        # Position gate: learned signal → scalar gate per position
        self.pos_signal = nn.Parameter(torch.zeros(1, 1, max_seq_len))
        nn.init.normal_(self.pos_signal, std=0.01)
        self.gate_conv = nn.Conv1d(1, 1, kernel_size, padding=kernel_size - 1, bias=True)
        nn.init.zeros_(self.gate_conv.weight)
        nn.init.constant_(self.gate_conv.bias, -3.0)  # sigmoid(-3) ≈ 0.05

    def gate_parameters(self):
        yield self.pos_signal
        yield from self.gate_conv.parameters()

    def non_gate_parameters(self):
        yield from self.base.parameters()
        yield from self.delta.parameters()

    def forward(self, x, context_state=None, **_ignored):
        B, T, _ = x.shape
        dtype = x.dtype
        base_out = self.base(x)  # (B, T, out)
        delta_out = self.delta(x)  # (B, T, out)
        # Position gate
        sig = self.pos_signal[:, :, :T].expand(B, -1, -1).to(dtype)
        gate = torch.sigmoid(
            F.conv1d(sig, self.gate_conv.weight.to(dtype),
                     self.gate_conv.bias.to(dtype),
                     padding=self.kernel_size - 1)[:, :, :T]
        )  # (B, 1, T)
        gate = gate.permute(0, 2, 1)  # (B, T, 1)
        return base_out + gate * delta_out


class CausalInterpolationLinear(nn.Module):
    """Phase 17I: Causal Interpolation Linear (CIL).

    y = ((1-α(pos)) · W₀ + α(pos) · W₁) · x

    Interpolates between two weight matrices in weight space using a position-dependent
    scalar α from causal conv. α starts at 0 → pure W₀ at init.

    Key: only ONE matmul with the interpolated weight, not two separate matmuls.
    """
    def __init__(self, in_features, out_features, kernel_size=64, max_seq_len=2048, **_ignored):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_size = kernel_size
        self.ln_basis = nn.Identity()

        # Two full-rank weight matrices
        self.weight_0 = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_1 = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        nn.init.kaiming_uniform_(self.weight_0, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.weight_1, a=math.sqrt(5))

        # Position interpolation: learned signal → α(pos) ∈ [0,1]
        self.pos_signal = nn.Parameter(torch.zeros(1, 1, max_seq_len))
        nn.init.normal_(self.pos_signal, std=0.01)
        self.alpha_conv = nn.Conv1d(1, 1, kernel_size, padding=kernel_size - 1, bias=True)
        nn.init.zeros_(self.alpha_conv.weight)
        nn.init.constant_(self.alpha_conv.bias, -5.0)  # sigmoid(-5) ≈ 0.007 → nearly pure W₀

    @torch.no_grad()
    def reset_parameters(self, bound=None):
        """Owns this module's entire subtree; GPT.init_weights claims it.

        weight_0/weight_1/bias/pos_signal are bare nn.Parameters and held
        uninitialized storage before this existed. The -5.0 alpha bias is the
        intended near-pure-W0 start.
        """
        nn.init.kaiming_uniform_(self.weight_0, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.weight_1, a=math.sqrt(5))
        nn.init.zeros_(self.bias)
        nn.init.normal_(self.pos_signal, std=0.01)
        nn.init.zeros_(self.alpha_conv.weight)
        if self.alpha_conv.bias is not None:
            nn.init.constant_(self.alpha_conv.bias, -5.0)

    def gate_parameters(self):
        yield self.pos_signal
        yield from self.alpha_conv.parameters()

    def non_gate_parameters(self):
        yield self.weight_0
        yield self.weight_1
        yield self.bias

    def forward(self, x, context_state=None, **_ignored):
        B, T, _ = x.shape
        dtype = x.dtype
        # Position-dependent alpha
        sig = self.pos_signal[:, :, :T].expand(B, -1, -1).to(dtype)
        alpha = torch.sigmoid(
            F.conv1d(sig, self.alpha_conv.weight.to(dtype),
                     self.alpha_conv.bias.to(dtype),
                     padding=self.kernel_size - 1)[:, :, :T]
        )  # (B, 1, T)  → scalar per position
        alpha = alpha.permute(0, 2, 1)  # (B, T, 1)

        # Interpolate weight matrices: W_eff = (1-α)·W₀ + α·W₁
        # Per-position matmul: x @ W_eff^T = (1-α)·(x@W₀^T) + α·(x@W₁^T)
        y0 = F.linear(x, self.weight_0.to(dtype), self.bias.to(dtype))  # (B, T, out)
        y1 = F.linear(x, self.weight_1.to(dtype))  # (B, T, out)
        return (1 - alpha) * y0 + alpha * y1


class PositionalResidualBias(nn.Module):
    """Phase 17J: Positional Residual Bias (PRB).

    y = Wx + b(pos)

    Position-dependent bias vector (NOT weight modulation). ~1% overhead.
    Diagnostic experiment: if PRB matches CKR, the benefit was never about
    weight modulation — just position-dependent output.
    """
    def __init__(self, in_features, out_features, kernel_size=64, max_seq_len=2048, **_ignored):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_size = kernel_size
        self.ln_basis = nn.Identity()

        self.base = Linear(in_features, out_features, bias=True)
        # Position-dependent bias from causal conv
        self.pos_signal = nn.Parameter(torch.zeros(1, 1, max_seq_len))
        nn.init.normal_(self.pos_signal, std=0.01)
        # Conv: 1 channel → out_features channels (generates bias vector per position)
        self.bias_conv = nn.Conv1d(1, out_features, kernel_size, padding=kernel_size - 1, bias=True)
        nn.init.zeros_(self.bias_conv.weight)
        nn.init.zeros_(self.bias_conv.bias)

    def gate_parameters(self):
        yield self.pos_signal
        yield from self.bias_conv.parameters()

    def non_gate_parameters(self):
        yield from self.base.parameters()

    def forward(self, x, context_state=None, **_ignored):
        B, T, _ = x.shape
        dtype = x.dtype
        y = self.base(x)  # (B, T, out)
        sig = self.pos_signal[:, :, :T].expand(B, -1, -1).to(dtype)
        bias = F.conv1d(
            sig, self.bias_conv.weight.to(dtype), self.bias_conv.bias.to(dtype),
            padding=self.kernel_size - 1
        )[:, :, :T]  # (B, out, T)
        return y + bias.permute(0, 2, 1)  # (B, T, out)


# ============================================================================
# Phase 18: Beyond CKR — Fundamentally Different Approaches
# ============================================================================

class AdaptiveGatedLinear(nn.Module):
    """Phase 18A: Adaptive ReLU² Gate (ARG).

    y = W·x · g(x)  where g(x) = sigmoid(w_g·x + b_g) per output channel.

    Content-dependent (NOT position), minimal overhead (~D_out params).
    g starts at 1.0 (bias=+5 → sigmoid(5) ≈ 0.993).
    """
    def __init__(self, in_features, out_features, **_ignored):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ln_basis = nn.Identity()
        self.base = Linear(in_features, out_features, bias=True)
        # Gate: content-dependent per-channel gate
        self.gate_w = nn.Parameter(torch.zeros(out_features, in_features))
        self.gate_b = nn.Parameter(torch.full((out_features,), 5.0))  # sigmoid(5)≈0.993

    def gate_parameters(self):
        yield self.gate_w
        yield self.gate_b

    def non_gate_parameters(self):
        yield from self.base.parameters()

    def forward(self, x, context_state=None, **_ignored):
        dtype = x.dtype
        y = self.base(x)  # (B, T, out)
        gate_logits = F.linear(x, self.gate_w.to(dtype), self.gate_b.to(dtype))  # (B, T, out)
        gate = torch.sigmoid(gate_logits)
        return y * gate


class KroneckerLinear(nn.Module):
    """Phase 18C: Kronecker-Factored Linear (KFL).

    W = A ⊗ B where A is (d1, d1') and B is (d2, d2').
    d1*d2 = D_out, d1'*d2' = D_in.

    FEWER parameters than dense: d1² + d2² vs (d1*d2)².
    Computation: reshape x to (..., d2', d1'), multiply by B left and A right, reshape.
    """
    def __init__(self, in_features, out_features, **_ignored):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ln_basis = nn.Identity()
        # Find factorization closest to sqrt
        d2_in = self._find_factor(in_features)
        d1_in = in_features // d2_in
        d2_out = self._find_factor(out_features)
        d1_out = out_features // d2_out
        self.d1_in, self.d2_in = d1_in, d2_in
        self.d1_out, self.d2_out = d1_out, d2_out
        self.A = nn.Parameter(torch.empty(d1_out, d1_in))
        self.B = nn.Parameter(torch.empty(d2_out, d2_in))
        self.bias = nn.Parameter(torch.zeros(out_features))
        # Kronecker init: effective weight is A⊗B, so var(AB) = var(A)*var(B).
        # Target var = 2/fan_in. So each factor gets var = sqrt(2/fan_in).
        target_std = math.sqrt(math.sqrt(2.0 / in_features))
        nn.init.normal_(self.A, std=target_std)
        nn.init.normal_(self.B, std=target_std)

    @staticmethod
    def _find_factor(n):
        """Find factor of n closest to sqrt(n)."""
        s = int(math.sqrt(n))
        for i in range(s, 0, -1):
            if n % i == 0:
                return i
        return 1

    def gate_parameters(self):
        return iter([])

    def non_gate_parameters(self):
        yield self.A
        yield self.B
        yield self.bias

    def forward(self, x, context_state=None, **_ignored):
        B, T, _ = x.shape
        dtype = x.dtype
        # Reshape: (B, T, d2_in, d1_in)
        x_r = x.view(B, T, self.d2_in, self.d1_in)
        # B @ x_r → (B, T, d2_out, d1_in)
        y = torch.matmul(self.B.to(dtype), x_r)
        # y @ A^T → (B, T, d2_out, d1_out)
        y = torch.matmul(y, self.A.to(dtype).t())
        # Reshape back: (B, T, d2_out * d1_out)
        y = y.reshape(B, T, self.out_features)
        return y + self.bias.to(dtype)


class MixtureNorm(nn.Module):
    """Phase 18H: Mixture of Norms.

    norm(x) = w₁·RMSNorm(x) + w₂·LayerNorm(x)

    Learned mixture lets model keep magnitude info where useful.
    Starts at pure RMSNorm (w₁=1, w₂=0).
    """
    def __init__(self, n_embd):
        super().__init__()
        self.rms = nn.RMSNorm(n_embd)
        self.ln = nn.LayerNorm(n_embd, elementwise_affine=False)
        self.w1 = nn.Parameter(torch.tensor(1.0))  # RMSNorm weight
        self.w2 = nn.Parameter(torch.tensor(0.0))  # LayerNorm weight (starts off)

    def forward(self, x):
        return self.w1 * self.rms(x) + self.w2 * self.ln(x)


class DynamicActivation(nn.Module):
    """Phase 18I: Dynamic Activation Function.

    f(x) = α·ReLU²(x) + β·GELU(x) + γ·SiLU(x)

    Per-layer learned scalars. ~3 params per layer. Starts as pure ReLU².
    """
    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(1.0))  # ReLU² weight
        self.beta = nn.Parameter(torch.tensor(0.0))   # GELU weight
        self.gamma = nn.Parameter(torch.tensor(0.0))  # SiLU weight

    def forward(self, x):
        return (self.alpha * F.relu(x).square()
                + self.beta * F.gelu(x)
                + self.gamma * F.silu(x))


class CausalAttnBias(nn.Module):
    """Phase 18J: Causal Attention Score Bias.

    Learned lower-triangular bias added to attention scores before softmax.
    Uses a factored representation: bias(i,j) = f(i-j) where f is a learned
    function of relative distance, stored as a 1D vector of max_dist values.

    Much cheaper than a full T×T matrix, and generalizes better.
    """
    def __init__(self, n_head, max_dist=2048):
        super().__init__()
        self.n_head = n_head
        self.max_dist = max_dist
        # Per-head learned bias as function of relative distance
        # Shape: (n_head, max_dist) — bias for distance 0, 1, 2, ...
        self.rel_bias = nn.Parameter(torch.zeros(n_head, max_dist))

    def get_bias(self, T, device):
        """Return (1, n_head, T, T) lower-triangular bias matrix."""
        # Create relative distance matrix
        pos = torch.arange(T, device=device)
        dist = pos.unsqueeze(0) - pos.unsqueeze(1)  # (T, T), negative = future
        dist = dist.clamp(min=0, max=self.max_dist - 1)  # causal: only past
        # Mask future positions
        causal_mask = (pos.unsqueeze(0) - pos.unsqueeze(1)) >= 0  # (T, T)
        # Look up bias values
        bias = self.rel_bias[:, dist]  # (n_head, T, T)
        bias = bias * causal_mask.unsqueeze(0).float()
        return bias.unsqueeze(0)  # (1, n_head, T, T)


class PerChannelScale(nn.Module):
    """Phase 18F: Per-Channel Learnable Scale.

    Applied after a linear layer: y = Linear(x) * scale
    where scale is a per-channel learned parameter initialized to 1.0.

    Achieves similar effect to per-channel LR: channels that don't contribute
    have their scale shrink, effectively reducing their influence.
    """
    def __init__(self, n_features):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(n_features))

    def forward(self, x):
        return x * self.scale.to(x.dtype)


class ModulationDiagnostics:
    """Universal diagnostic collector for ALL position-conditioned linear modules.

    Supports: CKR, LoKR, PGR, CIL, PRB, COM, PSG, and any future module
    with pos_signal or similar position-dependent routing.

    Computes diagnostics DIRECTLY from model parameters using torch.no_grad().
    This is compile-safe because it runs OUTSIDE the compiled forward graph.

    Key diagnostics:
    - branch_entropy: how uniform the routing is (1.0=uniform, 0.0=collapsed)
    - position_std: how much routing varies by position (0=static, high=dynamic)
    - gate_magnitude: average gate value (for gated architectures like PGR/PRB)
    - weight_divergence: how different branch weights are from each other
    - grad_norm: gradient norm of position signal (learning signal strength)
    - temperature: current softmax temperature (CKR only)
    """
    # All module types we can diagnose
    DIAG_TYPES = (CausalKernelLinear, LoKRLinear, PositionGatedResidual,
                  CausalInterpolationLinear, PositionalResidualBias,
                  CausalOutputMixer, PositionalScalarGatedLinear,
                  AdaptiveGatedLinear, KroneckerLinear)  # Phase 18
    # Phase 18 types diagnosed from model-level (not per-layer module)
    P18_TYPES = (DynamicActivation, MixtureNorm, PerChannelScale)

    def __init__(self, model):
        self.model = model
        self._layers = []
        for name, mod in model.named_modules():
            if isinstance(mod, self.DIAG_TYPES):
                self._layers.append((name, mod))
        # Phase 18: Also track P18 modules
        self._p18_layers = []
        for name, mod in model.named_modules():
            if isinstance(mod, self.P18_TYPES):
                self._p18_layers.append((name, mod))

    @torch.no_grad()
    def _collect_branch_routing(self, mod):
        """Collect branch routing stats for modules with pos_signal + branch_conv (CKR, LoKR)."""
        sig = mod.pos_signal  # (1, C_pos, T_max)
        raw_w = mod.branch_conv(sig.to(dtype=mod.branch_conv.weight.dtype))  # (1, K, T_max)
        temp = 1.0
        if hasattr(mod, '_temperature'):
            temp = mod._temperature.item()
        w = F.softmax(raw_w.float() / max(temp, 1e-8), dim=1)  # (1, K, T)

        # Branch weight entropy (normalized)
        log_w = torch.log(w.clamp(min=1e-8))
        entropy = -(w * log_w).sum(dim=1).mean().item()
        max_entropy = math.log(mod.n_branches)
        entropy_ratio = entropy / max(max_entropy, 1e-8)

        # Position variance (how much routing varies across positions)
        weight_std = w.std(dim=2).mean().item()

        # Weight divergence: mean pairwise cosine distance between branch weights
        weight_div = 0.0
        if hasattr(mod, 'branches'):
            norms = []
            for b in mod.branches:
                norms.append(b.weight.float().reshape(-1))
            if len(norms) >= 2:
                cos_sims = []
                for i in range(len(norms)):
                    for j in range(i+1, len(norms)):
                        cos = F.cosine_similarity(norms[i].unsqueeze(0), norms[j].unsqueeze(0)).item()
                        cos_sims.append(cos)
                weight_div = 1.0 - (sum(cos_sims) / len(cos_sims))  # 0=identical, 1=orthogonal

        return {
            'type': type(mod).__name__,
            'entropy_ratio': entropy_ratio,
            'position_std': weight_std,
            'temperature': temp,
            'weight_divergence': weight_div,
            'pos_signal_norm': sig.float().norm().item(),
            'pos_signal_grad_norm': sig.grad.float().norm().item() if sig.grad is not None else 0.0,
        }

    @torch.no_grad()
    def _collect_gate_stats(self, mod):
        """Collect gate stats for modules with a scalar position gate (PGR, CIL, PRB, COM, PSG)."""
        sig = mod.pos_signal  # (1, 1, T_max)
        # Determine gate conv
        if isinstance(mod, PositionGatedResidual):
            conv = mod.gate_conv
            gate_name = 'position_gate'
        elif isinstance(mod, CausalInterpolationLinear):
            conv = mod.alpha_conv
            gate_name = 'alpha'
        elif isinstance(mod, PositionalResidualBias):
            conv = mod.bias_conv
            gate_name = 'bias_magnitude'
        elif isinstance(mod, CausalOutputMixer):
            conv = mod.gate_conv
            gate_name = 'com_gate'
        elif isinstance(mod, PositionalScalarGatedLinear):
            conv = mod.gate_conv if hasattr(mod, 'gate_conv') else None
            gate_name = 'psg_gate'
        else:
            return {'type': type(mod).__name__, 'error': 'unknown_gate_type'}

        if conv is None:
            return {'type': type(mod).__name__, 'error': 'no_conv'}

        raw = conv(sig.to(dtype=conv.weight.dtype))  # (1, C_out, T_max)
        gate_vals = torch.sigmoid(raw.float())

        # For PRB, the "gate" is the bias magnitude, not sigmoid
        if isinstance(mod, PositionalResidualBias):
            gate_vals = raw.float()

        result = {
            'type': type(mod).__name__,
            'gate_name': gate_name,
            'gate_mean': gate_vals.mean().item(),
            'gate_std': gate_vals.std().item(),
            'gate_min': gate_vals.min().item(),
            'gate_max': gate_vals.max().item(),
            'pos_signal_norm': sig.float().norm().item(),
            'pos_signal_grad_norm': sig.grad.float().norm().item() if sig.grad is not None else 0.0,
        }

        # Weight divergence for PGR (base vs delta)
        if isinstance(mod, PositionGatedResidual):
            base_w = mod.base.weight.float().reshape(-1)
            delta_w = mod.delta.weight.float().reshape(-1)
            delta_norm = delta_w.norm().item()
            base_norm = base_w.norm().item()
            result['delta_to_base_ratio'] = delta_norm / max(base_norm, 1e-8)

        # Weight divergence for CIL (w0 vs w1)
        if isinstance(mod, CausalInterpolationLinear):
            w0 = mod.weight_0.float().reshape(-1)
            w1 = mod.weight_1.float().reshape(-1)
            cos = F.cosine_similarity(w0.unsqueeze(0), w1.unsqueeze(0)).item()
            result['weight_cosine_sim'] = cos
            result['weight_divergence'] = 1.0 - cos

        return result

    @torch.no_grad()
    def _collect_p18_stats(self, mod):
        """Collect stats for Phase 18 modules (DynamicActivation, MixtureNorm, PerChannelScale)."""
        result = {'type': type(mod).__name__}
        if isinstance(mod, DynamicActivation):
            result['alpha'] = mod.alpha.item()
            result['beta'] = mod.beta.item()
            result['gamma'] = mod.gamma.item()
        elif isinstance(mod, MixtureNorm):
            result['w_rms'] = mod.w1.item()
            result['w_ln'] = mod.w2.item()
        elif isinstance(mod, PerChannelScale):
            result['scale_mean'] = mod.scale.float().mean().item()
            result['scale_std'] = mod.scale.float().std().item()
        return result

    @torch.no_grad()
    def _collect_arg_stats(self, mod):
        """Collect stats for AdaptiveGatedLinear (18A)."""
        gate_b = mod.gate_b.float()
        gate_w_norm = mod.gate_w.float().norm().item()
        return {
            'type': 'AdaptiveGatedLinear',
            'gate_mean': torch.sigmoid(gate_b).mean().item(),
            'gate_std': torch.sigmoid(gate_b).std().item(),
            'gate_w_norm': gate_w_norm,
            'gate_b_mean': gate_b.mean().item(),
        }

    @torch.no_grad()
    def _collect_kfl_stats(self, mod):
        """Collect stats for KroneckerLinear (18C)."""
        return {
            'type': 'KroneckerLinear',
            'A_norm': mod.A.float().norm().item(),
            'B_norm': mod.B.float().norm().item(),
            'factorization': f'{mod.d1_out}x{mod.d1_in} ⊗ {mod.d2_out}x{mod.d2_in}',
            'param_ratio': (mod.A.numel() + mod.B.numel()) / (mod.in_features * mod.out_features),
        }

    @torch.no_grad()
    def collect(self):
        """Compute diagnostics for all tracked layers."""
        if not self._layers and not self._p18_layers:
            return {}
        all_metrics = {}
        for name, mod in self._layers:
            if isinstance(mod, (CausalKernelLinear, LoKRLinear)):
                all_metrics[name] = self._collect_branch_routing(mod)
            elif isinstance(mod, AdaptiveGatedLinear):
                all_metrics[name] = self._collect_arg_stats(mod)
            elif isinstance(mod, KroneckerLinear):
                all_metrics[name] = self._collect_kfl_stats(mod)
            else:
                all_metrics[name] = self._collect_gate_stats(mod)
        # P18 modules: aggregate by type (only report first instance per type)
        seen_types = set()
        for name, mod in self._p18_layers:
            tname = type(mod).__name__
            if tname not in seen_types:
                all_metrics[f'p18/{tname}'] = self._collect_p18_stats(mod)
                seen_types.add(tname)
        return all_metrics

    def format(self, metrics):
        """Format diagnostics as a compact log string."""
        if not metrics:
            return "  [diag] no conditioning layers found"
        lines = ["  [modulation diagnostics]"]
        # Group by module type
        by_type = {}
        for name, m in metrics.items():
            t = m.get('type', 'unknown')
            by_type.setdefault(t, []).append(m)

        for mod_type, mods in by_type.items():
            lines.append(f"    {mod_type} ({len(mods)} layers):")
            if 'entropy_ratio' in mods[0]:
                ent = sum(m['entropy_ratio'] for m in mods) / len(mods)
                pstd = sum(m['position_std'] for m in mods) / len(mods)
                wdiv = sum(m.get('weight_divergence', 0) for m in mods) / len(mods)
                gnorm = sum(m.get('pos_signal_grad_norm', 0) for m in mods) / len(mods)
                lines.append(f"      entropy={ent:.3f}  pos_std={pstd:.4f}  "
                           f"wt_div={wdiv:.3f}  grad_norm={gnorm:.4f}")
                if any(m.get('temperature', 1.0) != 1.0 for m in mods):
                    lines.append(f"      temperature={mods[0]['temperature']:.3f}")
            elif 'gate_mean' in mods[0]:
                gm = sum(m['gate_mean'] for m in mods) / len(mods)
                gs = sum(m['gate_std'] for m in mods) / len(mods)
                gnorm = sum(m.get('pos_signal_grad_norm', 0) for m in mods) / len(mods)
                lines.append(f"      gate_mean={gm:.4f}  gate_std={gs:.4f}  "
                           f"grad_norm={gnorm:.4f}")
                if any('delta_to_base_ratio' in m for m in mods):
                    ratio = sum(m.get('delta_to_base_ratio', 0) for m in mods) / len(mods)
                    lines.append(f"      delta/base_ratio={ratio:.4f}")
                if any('weight_divergence' in m for m in mods):
                    wdiv = sum(m.get('weight_divergence', 0) for m in mods) / len(mods)
                    lines.append(f"      weight_divergence={wdiv:.4f}")
            elif 'alpha' in mods[0]:
                # DynamicActivation
                a = mods[0]['alpha']
                b = mods[0]['beta']
                g = mods[0]['gamma']
                lines.append(f"      α(ReLU²)={a:.3f}  β(GELU)={b:.3f}  γ(SiLU)={g:.3f}")
            elif 'w_rms' in mods[0]:
                # MixtureNorm
                w1 = mods[0]['w_rms']
                w2 = mods[0]['w_ln']
                lines.append(f"      w_rms={w1:.3f}  w_ln={w2:.3f}")
            elif 'scale_mean' in mods[0]:
                # PerChannelScale
                sm = mods[0]['scale_mean']
                ss = mods[0]['scale_std']
                lines.append(f"      scale_mean={sm:.4f}  scale_std={ss:.4f}")
            elif 'A_norm' in mods[0]:
                # KroneckerLinear
                an = sum(m['A_norm'] for m in mods) / len(mods)
                bn = sum(m['B_norm'] for m in mods) / len(mods)
                pr = mods[0].get('param_ratio', 0)
                lines.append(f"      A_norm={an:.3f}  B_norm={bn:.3f}  param_ratio={pr:.4f}")
            elif 'gate_w_norm' in mods[0]:
                # AdaptiveGatedLinear
                gm = sum(m['gate_mean'] for m in mods) / len(mods)
                gs = sum(m['gate_std'] for m in mods) / len(mods)
                gwn = sum(m['gate_w_norm'] for m in mods) / len(mods)
                lines.append(f"      gate_mean={gm:.4f}  gate_std={gs:.4f}  gate_w_norm={gwn:.4f}")
        return "\n".join(lines)

    def to_dict(self, metrics):
        """Flatten metrics into a single dict for wandb/logging."""
        if not metrics:
            return {}
        result = {}
        # Compute means across all layers
        all_vals = list(metrics.values())
        if 'entropy_ratio' in all_vals[0]:
            result['diag/branch_entropy'] = sum(m['entropy_ratio'] for m in all_vals) / len(all_vals)
            result['diag/position_std'] = sum(m['position_std'] for m in all_vals) / len(all_vals)
            result['diag/weight_divergence'] = sum(m.get('weight_divergence', 0) for m in all_vals) / len(all_vals)
            result['diag/temperature'] = all_vals[0].get('temperature', 1.0)
        if 'gate_mean' in all_vals[0]:
            result['diag/gate_mean'] = sum(m['gate_mean'] for m in all_vals) / len(all_vals)
            result['diag/gate_std'] = sum(m['gate_std'] for m in all_vals) / len(all_vals)
        result['diag/pos_signal_grad_norm'] = sum(m.get('pos_signal_grad_norm', 0) for m in all_vals) / len(all_vals)
        result['diag/pos_signal_norm'] = sum(m.get('pos_signal_norm', 0) for m in all_vals) / len(all_vals)
        if any('delta_to_base_ratio' in m for m in all_vals):
            result['diag/delta_to_base_ratio'] = sum(m.get('delta_to_base_ratio', 0) for m in all_vals) / len(all_vals)
        if any('weight_divergence' in m for m in all_vals):
            result['diag/weight_divergence'] = sum(m.get('weight_divergence', 0) for m in all_vals) / len(all_vals)
        return result

    def save_to_file(self, metrics, step, filepath):
        """Append metrics as JSONL to a file."""
        import json, os
        if not metrics:
            return
        flat = self.to_dict(metrics)
        flat['step'] = step
        # Also save per-layer breakdown for first 4 layers
        for i, (name, m) in enumerate(metrics.items()):
            if i >= 4:
                break
            prefix = f"layer_{i}"
            for k, v in m.items():
                if isinstance(v, (int, float)):
                    flat[f"{prefix}/{k}"] = v
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'a', encoding='utf-8') as f:
            f.write(json.dumps(flat) + '\n')

    # ── Phase 19: Expanded Diagnostics ──────────────────────────────────────

    @torch.no_grad()
    def collect_p19(self, model):
        """Collect Phase 19 expanded diagnostics from model state.

        Returns a dict of metrics covering:
        - Gradient health (per-layer grad norms, SNR, flow ratio)
        - Weight dynamics (weight norms per layer)
        - Phase 19 proposal-specific metrics (alpha, head_importance, logit_scale,
          sigma stats, depth_decay, mixer gamma, etc.)
        - Residual stream parameters (resid_lambda, x0_lambda ratios)
        """
        metrics = {}

        # ── 1. Gradient health per block ──
        grad_norms_attn = []
        grad_norms_ffn = []
        weight_norms = []
        for i, block in enumerate(model.transformer.h):
            prefix = f"block_{i}"
            # Get key weight references
            if hasattr(block, 'attn'):
                cq_w = block.attn.c_q.weight if (hasattr(block.attn, 'c_q') and hasattr(block.attn.c_q, 'weight')) else None
                cp_w = block.attn.c_proj.weight if (hasattr(block.attn, 'c_proj') and hasattr(block.attn.c_proj, 'weight')) else None
                if cp_w is None:
                    mat_params = [p for p in block.attn.parameters() if p.ndim == 2]
                    cp_w = mat_params[-1] if mat_params else None
            else:
                cq_w = cp_w = None
            
            if hasattr(block, 'mlp'):
                fc_w = getattr(block.mlp, 'c_fc', None)
                fc_w = fc_w.weight if (fc_w is not None and hasattr(fc_w, 'weight')) else None
                proj_w = getattr(block.mlp, 'c_proj', None)
                proj_w = proj_w.weight if (proj_w is not None and hasattr(proj_w, 'weight')) else None
                # P20 MLP variants: fall back to first/last 2D param
                if fc_w is None:
                    mat_params = [p for p in block.mlp.parameters() if p.ndim == 2]
                    fc_w = mat_params[0] if mat_params else None
                    proj_w = mat_params[-1] if len(mat_params) > 1 else fc_w
            elif hasattr(block, 'ffwd'):
                fc_w = getattr(block.ffwd, 'c_fc', None)
                fc_w = fc_w.weight if (fc_w is not None and hasattr(fc_w, 'weight')) else None
                proj_w = getattr(block.ffwd, 'c_proj', None)
                proj_w = proj_w.weight if (proj_w is not None and hasattr(proj_w, 'weight')) else None
                if fc_w is None:
                    mat_params = [p for p in block.ffwd.parameters() if p.ndim == 2]
                    fc_w = mat_params[0] if mat_params else None
                    proj_w = mat_params[-1] if len(mat_params) > 1 else fc_w
            else:
                fc_w = proj_w = None

            # Grad norms
            attn_gn = cp_w.grad.float().norm().item() if (cp_w is not None and cp_w.grad is not None) else 0.0
            ffn_gn = fc_w.grad.float().norm().item() if (fc_w is not None and fc_w.grad is not None) else 0.0
            grad_norms_attn.append(attn_gn)
            grad_norms_ffn.append(ffn_gn)

            # Grad SNR (mean / std) for FFN
            if fc_w is not None and fc_w.grad is not None:
                g = fc_w.grad.float()
                snr = (g.mean().abs() / (g.std() + 1e-8)).item()
            else:
                snr = 0.0
            metrics[f"{prefix}/grad_norm_attn"] = attn_gn
            metrics[f"{prefix}/grad_norm_ffn"] = ffn_gn
            metrics[f"{prefix}/grad_snr"] = snr
            # Gradient flow ratio (attn vs ffn balance)
            metrics[f"{prefix}/grad_flow_ratio"] = attn_gn / max(ffn_gn, 1e-8)

            # Weight norms
            wn = fc_w.float().norm().item() if fc_w is not None else 0.0
            weight_norms.append(wn)
            metrics[f"{prefix}/weight_norm_fc"] = wn

        # Aggregate gradient stats
        if grad_norms_ffn:
            gn_tensor = torch.tensor(grad_norms_ffn)
            gn_mean = gn_tensor.mean().item()
            gn_std = gn_tensor.std().item()
            metrics["global/grad_cv"] = gn_std / max(gn_mean, 1e-8)  # coefficient of variation
            metrics["global/grad_norm_mean"] = gn_mean

        # ── 2. Residual stream parameters ──
        rl = model.resid_lambdas.float()
        x0l = model.x0_lambdas.float()
        for i in range(len(rl)):
            metrics[f"block_{i}/resid_lambda"] = rl[i].item()
            metrics[f"block_{i}/x0_lambda"] = x0l[i].item()
            metrics[f"block_{i}/resid_scale_ratio"] = rl[i].item() / max(abs(x0l[i].item()), 1e-8)

        # ── 3. Phase 19 proposal-specific metrics ──

        # 19A: Residual Gate Scaling (alpha per layer)
        for i, block in enumerate(model.transformer.h):
            if hasattr(block, 'residual_alpha') and block.residual_alpha is not None:
                alpha_val = F.softplus(block.residual_alpha).item()
                metrics[f"block_{i}/p19a_alpha"] = alpha_val
                if block.residual_alpha.grad is not None:
                    metrics[f"block_{i}/p19a_alpha_grad"] = block.residual_alpha.grad.float().norm().item()

        # 19B: Head Importance Scaling
        for i, block in enumerate(model.transformer.h):
            attn = block.attn
            if hasattr(attn, 'head_importance') and attn.head_importance is not None:
                his = F.softplus(attn.head_importance.float())
                metrics[f"block_{i}/p19b_head_scale_mean"] = his.mean().item()
                metrics[f"block_{i}/p19b_head_scale_std"] = his.std().item()
                metrics[f"block_{i}/p19b_head_scale_min"] = his.min().item()
                metrics[f"block_{i}/p19b_head_scale_max"] = his.max().item()
                # Entropy of head importance (normalized)
                p = his / his.sum()
                ent = -(p * torch.log(p + 1e-8)).sum().item()
                max_ent = math.log(len(his))
                metrics[f"block_{i}/p19b_head_entropy"] = ent / max(max_ent, 1e-8)

        # 19C: Residual Stream Mixing
        if model.residual_mix_gamma is not None:
            for i, gamma_p in enumerate(model.residual_mix_gamma):
                metrics[f"block_{i}/p19c_gamma"] = gamma_p.item()
            if model.residual_mixers is not None:
                for i, mixer in enumerate(model.residual_mixers):
                    metrics[f"block_{i}/p19c_mix_weight_norm"] = mixer.weight.float().norm().item()

        # 19D: Attention Logit Bias
        for i, block in enumerate(model.transformer.h):
            attn = block.attn
            if hasattr(attn, 'attn_logit_scale') and attn.attn_logit_scale is not None:
                scale = F.softplus(attn.attn_logit_scale.float())
                metrics[f"block_{i}/p19d_logit_scale_mean"] = scale.mean().item()
                metrics[f"block_{i}/p19d_logit_scale_std"] = scale.std().item()

        # 19E: Learned Residual Decay
        if model.depth_decay_raw is not None:
            decay_base = torch.sigmoid(model.depth_decay_raw.float()).item()
            metrics["global/p19e_decay_base"] = decay_base
            for i in range(model.config.n_layer):
                metrics[f"block_{i}/p19e_x0_factor"] = decay_base ** i

        # 19G: Spectral Reparameterization
        for i, block in enumerate(model.transformer.h):
            ffwd = block.mlp if hasattr(block, 'mlp') else (block.ffwd if hasattr(block, 'ffwd') else None)
            if ffwd is None:
                continue
            for label, srp in [('proj', getattr(ffwd, 'srp_proj', None)), ('fc', getattr(ffwd, 'srp_fc', None))]:
                if srp is None:
                    continue
                sigma = srp.sigma.float()
                metrics[f"block_{i}/p19g_{label}_sigma_max"] = sigma.max().item()
                metrics[f"block_{i}/p19g_{label}_sigma_min"] = sigma.min().item()
                metrics[f"block_{i}/p19g_{label}_sigma_ratio"] = (sigma.max() / (sigma.min() + 1e-8)).item()
                # Sigma entropy (normalized) — measures spectral concentration
                p = sigma.abs() / (sigma.abs().sum() + 1e-8)
                ent = -(p * torch.log(p + 1e-8)).sum().item()
                max_ent = math.log(len(sigma))
                metrics[f"block_{i}/p19g_{label}_sigma_entropy"] = ent / max(max_ent, 1e-8)

        # 19I: VE gate bias values
        for i, block in enumerate(model.transformer.h):
            attn = block.attn
            if hasattr(attn, 've_gate') and attn.ve_gate is not None and attn.ve_gate.bias is not None:
                bias = attn.ve_gate.bias.float()
                metrics[f"block_{i}/p19i_ve_bias_mean"] = bias.mean().item()
                metrics[f"block_{i}/p19i_ve_bias_std"] = bias.std().item()

        return metrics

    def format_p19(self, p19_metrics):
        """Format Phase 19 diagnostics as a compact log string."""
        if not p19_metrics:
            return ""
        lines = ["  [p19 diagnostics]"]
        # Global metrics
        for k in sorted(p19_metrics):
            if k.startswith("global/"):
                lines.append(f"    {k}: {p19_metrics[k]:.4f}")
        # Per-block summary (just first and last block for brevity)
        n_blocks = sum(1 for k in p19_metrics if k.startswith("block_0/"))
        if n_blocks > 0:
            block_keys = sorted(set(k.split("/")[1] for k in p19_metrics if k.startswith("block_")))
            for bk in block_keys:
                vals = [p19_metrics.get(f"block_{i}/{bk}", None) for i in range(20) if f"block_{i}/{bk}" in p19_metrics]
                if vals:
                    lines.append(f"    {bk}: [{vals[0]:.4f}...{vals[-1]:.4f}] (n={len(vals)})")
        return "\n".join(lines)

    def to_dict_p19(self, p19_metrics):
        """Flatten P19 metrics into wandb dict with 'p19/' prefix."""
        if not p19_metrics:
            return {}
        result = {}
        for k, v in p19_metrics.items():
            if isinstance(v, (int, float)):
                result[f"p19/{k}"] = v
        return result

    # ── Phase 20: Dynamic Weight Computation Diagnostics ──────────────────

    @torch.no_grad()
    def collect_p20(self, model):
        """Collect Phase 20 diagnostics: routing entropy, expert load, branch divergence."""
        metrics = {}
        for i, block in enumerate(model.transformer.h):
            mlp = block.mlp if hasattr(block, 'mlp') else (block.ffwd if hasattr(block, 'ffwd') else None)
            if mlp is None:
                continue
            prefix = f"block_{i}"

            if isinstance(mlp, MoNE_MLP):
                # Router entropy
                metrics[f"{prefix}/p20f_router_entropy"] = mlp._last_router_entropy
                # Expert load balance (std of loads — 0 = perfect balance)
                if mlp._last_expert_load is not None:
                    load = mlp._last_expert_load.float()
                    metrics[f"{prefix}/p20f_load_std"] = load.std().item()
                    metrics[f"{prefix}/p20f_load_max"] = load.max().item()
                    metrics[f"{prefix}/p20f_load_min"] = load.min().item()
                # Expert weight divergence (cosine distance between expert fc weights)
                norms = [e.weight.float().reshape(-1) for e in mlp.experts_fc]
                if len(norms) >= 2:
                    cos_sims = []
                    for ii in range(len(norms)):
                        for jj in range(ii + 1, len(norms)):
                            cos = F.cosine_similarity(norms[ii].unsqueeze(0), norms[jj].unsqueeze(0)).item()
                            cos_sims.append(cos)
                    metrics[f"{prefix}/p20f_expert_divergence"] = 1.0 - (sum(cos_sims) / len(cos_sims))
                # Router weight gradient norm
                if mlp.router.weight.grad is not None:
                    metrics[f"{prefix}/p20f_router_grad_norm"] = mlp.router.weight.grad.float().norm().item()

            elif isinstance(mlp, FrozenRoutedMLP):
                metrics[f"{prefix}/p20c_routing_entropy"] = mlp._last_routing_entropy
                if mlp._last_routing_weights is not None:
                    load = mlp._last_routing_weights.float()
                    metrics[f"{prefix}/p20c_load_std"] = load.std().item()
                # Branch weight divergence
                norms = [b.weight.float().reshape(-1) for b in mlp.branches_fc]
                if len(norms) >= 2:
                    cos_sims = []
                    for ii in range(len(norms)):
                        for jj in range(ii + 1, len(norms)):
                            cos = F.cosine_similarity(norms[ii].unsqueeze(0), norms[jj].unsqueeze(0)).item()
                            cos_sims.append(cos)
                    metrics[f"{prefix}/p20c_branch_divergence"] = 1.0 - (sum(cos_sims) / len(cos_sims))

            elif isinstance(mlp, DetachedRoutedMLP):
                metrics[f"{prefix}/p20d_routing_entropy"] = mlp._last_routing_entropy
                # Router gradient norm (from aux loss)
                router_last = mlp.router[-1]  # last Linear in router Sequential
                if router_last.weight.grad is not None:
                    metrics[f"{prefix}/p20d_router_grad_norm"] = router_last.weight.grad.float().norm().item()
                # Branch divergence
                norms = [b.weight.float().reshape(-1) for b in mlp.branches_fc]
                if len(norms) >= 2:
                    cos_sims = []
                    for ii in range(len(norms)):
                        for jj in range(ii + 1, len(norms)):
                            cos = F.cosine_similarity(norms[ii].unsqueeze(0), norms[jj].unsqueeze(0)).item()
                            cos_sims.append(cos)
                    metrics[f"{prefix}/p20d_branch_divergence"] = 1.0 - (sum(cos_sims) / len(cos_sims))

            elif isinstance(mlp, HashRoutedMLP):
                metrics[f"{prefix}/p20a_routing_entropy"] = mlp._last_routing_entropy

            elif isinstance(mlp, LSHRoutedMLP):
                metrics[f"{prefix}/p20b_routing_entropy"] = mlp._last_routing_entropy
                if mlp._last_bucket_usage is not None:
                    usage = mlp._last_bucket_usage.float()
                    metrics[f"{prefix}/p20b_bucket_usage_std"] = usage.std().item()
                    metrics[f"{prefix}/p20b_buckets_used"] = (usage > 1e-6).sum().item()

            elif isinstance(mlp, NoiseCEA_MLP):
                metrics[f"{prefix}/p20h_routing_entropy"] = mlp._last_routing_entropy
                if mlp.router.weight.grad is not None:
                    metrics[f"{prefix}/p20h_router_grad_norm"] = mlp.router.weight.grad.float().norm().item()
                # Delta magnitude (how big are the perturbations?)
                delta_norms = [d.float().norm().item() for d in mlp.delta_fc]
                metrics[f"{prefix}/p20h_delta_fc_norm_mean"] = sum(delta_norms) / len(delta_norms)

            elif isinstance(mlp, AttnDerivedMLP):
                metrics[f"{prefix}/p20i_routing_entropy"] = mlp._last_routing_entropy
                # Expert divergence
                norms = [e.weight.float().reshape(-1) for e in mlp.experts_fc]
                if len(norms) >= 2:
                    cos_sims = []
                    for ii in range(len(norms)):
                        for jj in range(ii + 1, len(norms)):
                            cos = F.cosine_similarity(norms[ii].unsqueeze(0), norms[jj].unsqueeze(0)).item()
                            cos_sims.append(cos)
                    metrics[f"{prefix}/p20i_expert_divergence"] = 1.0 - (sum(cos_sims) / len(cos_sims))

            elif isinstance(mlp, PWU_MLP):
                metrics[f"{prefix}/p20e_routing_entropy"] = mlp._last_routing_entropy
                if mlp.router.weight.grad is not None:
                    metrics[f"{prefix}/p20e_router_grad_norm"] = mlp.router.weight.grad.float().norm().item()
                # Branch divergence
                norms = [b.weight.float().reshape(-1) for b in mlp.branches_fc]
                if len(norms) >= 2:
                    cos_sims = []
                    for ii in range(len(norms)):
                        for jj in range(ii + 1, len(norms)):
                            cos = F.cosine_similarity(norms[ii].unsqueeze(0), norms[jj].unsqueeze(0)).item()
                            cos_sims.append(cos)
                    metrics[f"{prefix}/p20e_branch_divergence"] = 1.0 - (sum(cos_sims) / len(cos_sims))

            elif isinstance(mlp, FSVD_MLP):
                metrics[f"{prefix}/p20g_routing_entropy"] = mlp._last_routing_entropy
                metrics[f"{prefix}/p20g_sigma_sparsity"] = mlp._last_sigma_sparsity
                metrics[f"{prefix}/p20g_sigma_fc_mean"] = mlp.sigma_fc.float().mean().item()
                metrics[f"{prefix}/p20g_sigma_fc_std"] = mlp.sigma_fc.float().std().item()

            elif isinstance(mlp, WBFC_MLP):
                metrics[f"{prefix}/p20j_routing_entropy"] = mlp._last_routing_entropy

            # Phase 21: MoELinear diagnostics (MoELinear replaces layers inside standard MLP/Attn)
            if isinstance(mlp, MLP):
                for name, layer in [("fc", getattr(mlp, 'c_fc', None)),
                                    ("proj", getattr(mlp, 'c_proj', None))]:
                    if isinstance(layer, MoELinear):
                        metrics[f"{prefix}/p21_{name}_routing_entropy"] = layer._last_routing_entropy
            # Attention MoELinear layers
            attn = block.attn if hasattr(block, 'attn') else None
            if attn is not None:
                for name in ['c_q', 'c_k', 'c_v', 'c_proj']:
                    layer = getattr(attn, name, None)
                    if isinstance(layer, MoELinear):
                        metrics[f"{prefix}/p21_attn_{name}_entropy"] = layer._last_routing_entropy

        return metrics

    def format_p20(self, p20_metrics):
        """Format Phase 20 diagnostics as a compact log string."""
        if not p20_metrics:
            return ""
        lines = ["  [p20 diagnostics]"]
        block_keys = sorted(set(k.split("/")[1] for k in p20_metrics if k.startswith("block_")))
        for bk in block_keys:
            vals = [p20_metrics.get(f"block_{i}/{bk}", None) for i in range(64) if f"block_{i}/{bk}" in p20_metrics]
            if vals:
                lines.append(f"    {bk}: [{vals[0]:.4f}...{vals[-1]:.4f}] (n={len(vals)})")
        return "\n".join(lines)

    def to_dict_p20(self, p20_metrics):
        """Flatten P20 metrics into wandb dict with 'p20/' prefix."""
        if not p20_metrics:
            return {}
        result = {}
        for k, v in p20_metrics.items():
            if isinstance(v, (int, float)):
                result[f"p20/{k}"] = v
        return result


class ResidualAdaptiveLinear(nn.Module):
    """Proposal A: Dense base layer + context-conditioned additive low-rank delta.

    y = base(x) + scale * (x @ U * ctx_coeffs(ctx)) @ V

    Key properties:
    - V=0 at init → delta=0 → EXACT dense baseline at step 0 (no regression risk)
    - base(x) always has full D_in×D_out rank — no bottleneck from basis compression
    - Gradient path through base is pure dense-linear (no interference)
    - Delta grows only as optimizer finds it useful, gated by ctx_coeffs
    - Same forward(x, ctx) / gate_parameters() / non_gate_parameters() API as RemixedLinear

    Compatible with setup_optimizer: gate_parameters() returns ctx_coeffs params,
    non_gate_parameters() returns base+U+V+scale. No ln_basis → optimizer loop skips it.
    """
    def __init__(self, in_features, out_features, context_dim, rank=32, **_ignored):
        super().__init__()
        self.rank = rank
        self.in_features  = in_features
        self.out_features = out_features
        # Full-rank dense base — always active, no rank deficit
        self.base = Linear(in_features, out_features, bias=True)
        # Low-rank adaptive delta matrices
        self.U = nn.Parameter(torch.empty(in_features, rank))
        self.V = nn.Parameter(torch.zeros(rank, out_features))    # ZERO: delta=0 at init
        # Context → rank-dimensional coefficient vector
        self.ctx_coeffs = Linear(context_dim, rank, bias=True)
        self.scale = nn.Parameter(torch.tensor(0.1))
        # Dummy attribute so setup_optimizer can safely call .ln_basis.parameters() check
        self.ln_basis = nn.Identity()  # has no parameters

        nn.init.kaiming_uniform_(self.U, a=0.01)
        nn.init.zeros_(self.ctx_coeffs.weight)
        nn.init.zeros_(self.ctx_coeffs.bias)

    @torch.no_grad()
    def reset_parameters(self, bound=None):
        """Owns this module's entire subtree; GPT.init_weights claims it.

        Two defects are repaired here. The parameters were never initialized at
        all (bare nn.Parameters that init_weights' Linear-only fallback never
        reached, so they held whatever to_empty() left behind), and the intended
        init was a fixed point: V=0 and ctx_coeffs=0 together make dL/dU, dL/dV,
        dL/dscale and dL/dctx_coeffs all exactly zero, so the delta branch could
        never leave zero for the entire run. Setting the coefficient bias to 1
        keeps delta identically zero at init (V is still 0, which is what makes
        this an exact dense baseline at step 0) while leaving dL/dV alive.
        """
        nn.init.xavier_uniform_(self.base.weight)
        if self.base.bias is not None:
            nn.init.zeros_(self.base.bias)
        nn.init.kaiming_uniform_(self.U, a=0.01)
        nn.init.zeros_(self.V)                 # delta == 0 at init
        nn.init.zeros_(self.ctx_coeffs.weight)
        nn.init.ones_(self.ctx_coeffs.bias)    # coeffs == 1, so dL/dV != 0
        self.scale.fill_(0.1)

    def gate_parameters(self):
        """Context gate params routed to lower-LR optimizer group."""
        yield from self.ctx_coeffs.parameters()
        yield self.scale

    def non_gate_parameters(self):
        """Structural params (base weights, U, V) routed to normal-LR group."""
        yield self.base.weight
        yield self.base.bias
        yield self.U
        yield self.V

    def forward(self, x, context_state, **_ignored):
        y = self.base(x)
        if context_state is not None:
            ctx = context_state.to(x.dtype)
            coeffs = self.ctx_coeffs(ctx)                                     # (B, T, rank)
            h = (x @ self.U.to(x.dtype))                                      # (B, T, rank)
            delta = (h * coeffs) @ self.V.to(x.dtype)                         # (B, T, out_features)
            y = y + self.scale.to(x.dtype) * delta
        return y


class ConditionedLinear(nn.Module):
    """Phase 35: condition the activations, not the weights.

    RemixedLinear conditions a token by picking a point in the simplex over K
    stored templates. That costs K·out·basis parameters, and K·out·basis bytes of
    HBM traffic for every routed chunk, to buy exactly K-1 degrees of freedom per
    token, because the router emits K numbers and each is broadcast across a whole
    (out, basis) matrix. The low-rank delta variant (template_delta_rank) does not
    escape this: `weights_all` is (B, n_chunks, K) and is broadcast across the
    rank dimension, so widening r widens the subspace the correction lives in
    without adding a single independent knob. Three measurements agree that this
    is the defect rather than a tuning problem: ~25 FLOP/byte against an H200
    ridge of 206, a K× parameter count that inflates the token budget under a
    fixed data:param ratio, and K-1 DOF/token no matter how large K·d² gets.

    So the metric to design against is degrees of freedom per token per byte
    moved. Conditioning in weight space costs O(d²) per knob and has to cross
    HBM; conditioning in activation space costs O(d) per knob and never
    materializes an operator, so every parameter is read exactly once per batch
    like a dense weight. Two forms, both available here:

        additive        y = W₀x + U (c(x) ⊙ Vᵀx)                    c(x) ∈ R^R
        multiplicative  z₀ = x,  z_i = z_{i-1} + c_i(x)·u_i(v_iᵀz_{i-1})
                        y = W₀ z_m                                  c(x) ∈ R^m

    The additive form is a gated low-rank branch, i.e. a GLU relative. That is
    not a mark against it, since it is why GLU-family layers work and the bank
    does not, but it does mean it is the safe floor rather than the novel part.
    At d=768, R=256 it is 1.67× dense parameters (2.0× with a full-rank router)
    for 256 DOF/token, against 8.0× for 7 DOF/token at K=8.

    The multiplicative form composes the rank-1 generators sequentially instead
    of summing them, so it reaches the group they generate rather than their
    span. Written naively it is a loop of m memory-bound passes over the
    activations, which at m=16 costs several times the base GEMM and would repeat
    the original mistake of counting FLOPs and paying bandwidth. It does not have
    to be: with p = Vᵀx and S = VᵀU (strictly lower, since v_iᵀu_j only enters for
    j<i), the recursion closes in the coefficients alone,

        q_i = c_i·(p_i + Σ_{j<i} S_ij q_j),      z_m = x + U q,
        y   = W₀x + (W₀U) q

    which is the compact WY form of the product Π(I + c_i u_i v_iᵀ). Two width-m
    GEMMs plus an O(m²)-per-token triangular solve, one pass over the
    activations, and W₀U is a single (out,in)×(in,m) GEMM amortized over the
    batch. `cond_mult_impl='loop'` keeps the sequential version as an executable
    reference; tests/test_conditioned_linear.py pins the two together.

    Read side by side, the two forms differ by exactly one thing: additive
    applies diag(c) between V and U, multiplicative applies a token-dependent
    lower-triangular m×m matrix generated by c. Same bandwidth, curved operator
    manifold.

    Init is identity-preserving in the strong sense: every U is zero and every
    router weight is zero, so c ≡ 1 and the correction is identically zero, and
    W₀ takes exactly the init the dense baseline's weight would have (uniform for
    in-projections, zeros for out-projections). Crucially this leaves a live
    gradient, dL/dU ∝ (c ⊙ Vᵀx) ≠ 0 at step 0, unlike zero-initializing both
    factors of a product, which is a fixed point the optimizer never leaves.
    """

    def __init__(self, in_features, out_features, rank=256, mult_steps=0,
                 signal_dim=None, gate_source='router', coeff_act='centered',
                 router_rank=0, chunk_size=0, mult_impl='wy', mult_scale=-1.0,
                 zero_init_base=False, router_act='none', live_init=0.0, **_ignored):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.rank         = max(0, int(rank))
        self.mult_steps   = max(0, int(mult_steps))
        self.gate_source  = gate_source
        self.coeff_act    = coeff_act
        self.chunk_size   = max(0, int(chunk_size))
        self.mult_impl    = mult_impl
        self.zero_init_base = bool(zero_init_base)
        self.live_init    = max(0.0, float(live_init))
        if self.rank == 0 and self.mult_steps == 0:
            raise ValueError("ConditionedLinear needs cond_rank > 0 or cond_mult_steps > 0; "
                             "both zero is a plain dense Linear, use --cclblock-modulation weight")
        if gate_source not in ('router', 'tied', 'ctx'):
            raise ValueError(f"Unknown cond_gate_source {gate_source!r}")
        if coeff_act not in ('centered', 'linear', 'sigmoid', 'one'):
            raise ValueError(f"Unknown cond_coeff_act {coeff_act!r}")
        if mult_impl not in ('wy', 'loop'):
            raise ValueError(f"Unknown cond_mult_impl {mult_impl!r}")
        if router_act not in ('none', 'gelu', 'relu2'):
            raise ValueError(f"Unknown cond_router_act {router_act!r}")
        self.router_act = router_act

        # Every parameter is a bare nn.Parameter rather than a Linear submodule on
        # purpose: GPT.init_weights walks modules() and xavier-initializes any
        # Linear it does not recognize, which would silently destroy the zero-init
        # this layer's identity-preserving start depends on.
        self.base_w = nn.Parameter(torch.empty(out_features, in_features))

        sig_dim = in_features if signal_dim is None else int(signal_dim)
        self.signal_dim = sig_dim
        self.router_rank = max(0, int(router_rank))
        n_coeff = self.rank + self.mult_steps  # one router emits both branches' coefficients

        # Router: signal → coefficients. 'tied' reuses Vᵀx and costs nothing, which
        # is the control that separates "conditioning" from "just more
        # nonlinearity": under 'tied' this layer is a plain gated low-rank branch
        # with no independent conditioning signal at all.
        # router_act != 'none' puts a nonlinearity between route_down and route_w,
        # making the router  sig -> Linear(sig_dim, rr) -> act -> Linear(rr, n_coeff)
        # at exactly the parameter count and FLOP count of the linear rank-rr
        # router it replaces. conditioning_headroom.py --nonlinear-reach measures
        # this exact pair on a trained checkpoint: on dense d22 the nonlinearity
        # raised held-out reach from 0.338 to 0.445 at r=64, +0.107 overall and
        # +0.120 to +0.150 on every attention projection, against a permutation
        # control of -0.024. A rank-64 nonlinear router recovers 87% of what a
        # FULL-rank (1408) linear router reaches, at 1/22 of its cost.
        # The bias is not decoration: without it the hidden units cannot place
        # their nonlinearity anywhere but the origin, and the measured router had
        # one.
        self.route_w = None
        self.route_down = None
        self.route_b = None
        if gate_source in ('router', 'ctx'):
            if self.router_rank > 0:
                self.route_down = nn.Parameter(torch.empty(sig_dim, self.router_rank))
                self.route_w    = nn.Parameter(torch.empty(self.router_rank, n_coeff))
                if router_act != 'none':
                    self.route_b = nn.Parameter(torch.empty(self.router_rank))
            else:
                if router_act != 'none':
                    raise ValueError("cond_router_act needs cond_router_rank > 0: the "
                                     "nonlinearity goes in the router's hidden layer, and a "
                                     "full-rank router has none")
                self.route_w    = nn.Parameter(torch.empty(sig_dim, n_coeff))

        # Additive branch
        self.add_v = nn.Parameter(torch.empty(in_features, self.rank)) if self.rank else None
        self.add_u = nn.Parameter(torch.empty(self.rank, out_features)) if self.rank else None
        # Multiplicative branch: generators live in input space, so both factors
        # are (in, m), because the composition acts on x before W₀ sees it.
        self.mul_v = nn.Parameter(torch.empty(in_features, self.mult_steps)) if self.mult_steps else None
        self.mul_u = nn.Parameter(torch.empty(in_features, self.mult_steps)) if self.mult_steps else None
        if self.mult_steps:
            # Fixed, not learned: a learnable bound is not a bound. Raise it with
            # cond_mult_scale only if you want to trade the guarantee for range.
            sc = float(mult_scale) if mult_scale and mult_scale > 0 else 1.0 / self.mult_steps
            self._mult_scale_value = sc   # python float survives to_empty()
            self.register_buffer('mult_scale', torch.tensor(sc), persistent=False)

        self._gate_stats = {}  # populated during forward() when self.training

    # ── parameter groups ────────────────────────────────────────────────────
    def gate_parameters(self):
        """Conditioning params, routed to the lower-LR gate group."""
        if self.route_down is not None:
            yield self.route_down
        if self.route_b is not None:
            yield self.route_b
        if self.route_w is not None:
            yield self.route_w

    def non_gate_parameters(self):
        """Structural params: base operator and the generators perturbing it."""
        yield self.base_w
        for p in (self.add_v, self.add_u, self.mul_v, self.mul_u):
            if p is not None:
                yield p

    @torch.no_grad()
    def reset_parameters(self, base_bound):
        """Called from GPT.init_weights, where storage is garbage after to_empty().

        base_bound is the dense baseline's uniform init bound (sqrt(3)/sqrt(n_embd)),
        so W₀ starts exactly where the dense weight it replaces would have.
        """
        if self.zero_init_base:
            nn.init.zeros_(self.base_w)
        else:
            nn.init.uniform_(self.base_w, -base_bound, base_bound)
        # Zero router → c ≡ 1; zero U → correction ≡ 0; random V → live dL/dU.
        if self.route_down is not None:
            nn.init.normal_(self.route_down, std=self.signal_dim ** -0.5)
        if self.route_b is not None:
            # Nonzero so the hidden units do not all start at the same point on
            # the activation curve, which would make them exactly redundant and
            # cost the router most of its rank at step 0.
            nn.init.normal_(self.route_b, std=0.5)

        if self.live_init > 0 and self.add_v is not None:
            # LIVE INIT. The default below zeroes add_u and route_w, so the
            # branch contributes exactly nothing at step 0 and its gradient
            # unlocks in a chain: add_u at step 1, route_w at 2, route_down and
            # route_b at 3. Measured cost of that cold start, at matched tokens
            # in the flat-LR region, is 0.117 train loss at step 200 and 0.045
            # at step 400 against dense. This codebase has been bitten by the
            # same class of bug twice already (output_gate_basis zero-init, and
            # template_route exactly zero in every shipped run).
            #
            # The fix keeps the FUNCTION at init and changes only the
            # PARAMETERIZATION. Draw every factor nonzero, then subtract the
            # branch's expected contribution from W0:
            #
            #     W0 <- W_dense - U diag(cbar) V^T
            #
            # cbar is 1 for every coeff_act at zero-mean logits ('centered'
            # 1+tanh, 'linear' 1+z, 'sigmoid' 2*sigmoid, 'one'), so the
            # correction is just the plain product. The composite operator
            # starts exactly where the dense weight it replaces would start,
            # including under zero_init_base where W_dense is 0, while every
            # parameter receives gradient at step 0.
            s = float(self.live_init)
            nn.init.normal_(self.add_v, std=self.in_features ** -0.5)
            nn.init.normal_(self.add_u, std=s * self.rank ** -0.5)
            if self.route_w is not None:
                fan = self.router_rank if self.route_down is not None else self.signal_dim
                nn.init.normal_(self.route_w, std=fan ** -0.5)  # z ~ O(1), tanh unsaturated
            self.base_w.sub_((self.add_v @ self.add_u).T)
        else:
            if self.route_w is not None:
                nn.init.zeros_(self.route_w)
            if self.add_v is not None:
                nn.init.normal_(self.add_v, std=self.in_features ** -0.5)
                nn.init.zeros_(self.add_u)
        if getattr(self, 'mult_scale', None) is not None:
            self.mult_scale.fill_(self._mult_scale_value)
        if self.mul_v is not None:
            # Both random now. Identity at init comes from c = 0, not from u = 0,
            # so u must be nonzero or the router has no gradient at step 0.
            nn.init.normal_(self.mul_v, std=self.in_features ** -0.5)
            nn.init.normal_(self.mul_u, std=self.in_features ** -0.5)

    # ── conditioning coefficients ───────────────────────────────────────────
    def _activate(self, z):
        """Map router logits to coefficients that are exactly 1 at zero logits."""
        if self.coeff_act == 'centered':
            return 1.0 + torch.tanh(z)
        if self.coeff_act == 'sigmoid':
            return 2.0 * torch.sigmoid(z)
        if self.coeff_act == 'one':
            # c == 1 identically, so the branch is U V^T x with no nonlinearity
            # and W_eff = W0 + U V^T is a plain dense matrix. Any gain here is a
            # reparameterization/optimization effect, not expressivity. This is
            # the control that decides which of the two the p35 result is.
            return torch.ones_like(z)
        return 1.0 + z  # 'linear': unbounded, can flip sign

    def _mult_coefficients(self, z, dtype):
        """Coefficients for the composition branch: c = tanh(z) * scale.

        The general form is unstable and this is why the p35 C arms produced NaN.
        The recursion q_i = c_i (p_i + sum_{j<i} S_ij q_j) with S = V^T U is
        bounded at init only because U = 0 makes S = 0. Nothing constrains ||u||
        during training, Muon supplies a fixed-magnitude update regardless of
        gradient scale, and the amplification is geometric in m, so divergence is
        a question of when rather than whether.

        Making it a proper exponential map fixes it by construction. With u and v
        unit-normalized and |c| <= 1/m, every factor I + c u v^T has spectral norm
        at most 1 + 1/m, so the whole product is bounded by (1 + 1/m)^m < e for
        ANY m. That is also the honest version of the Lie-algebra-to-group story
        the design started from: prod(I + g_i/m) -> exp(sum g_i).

        Identity at init now comes from c = 0 rather than u = 0, which is the
        better end to zero: dL/d(router) is nonzero at step 0 because u is not,
        where zeroing u would have left the router with no gradient.
        """
        return (torch.tanh(z.float()) * self.mult_scale).to(dtype)

    def _coefficients(self, signal, p_add, p_mul, n_coeff, dtype):
        """(B, T, n_coeff) conditioning coefficients for this token (or chunk).

        p_add / p_mul are the already-computed Vᵀx projections, so the 'tied'
        source costs no extra matmul on top of the branch it gates.
        """
        if self.gate_source == 'tied':
            # No router at all: the branch's own projection is the gate signal.
            # This is the control that separates conditioning from nonlinearity:
            # under 'tied' the layer is a plain gated low-rank branch whose gate
            # carries no information the value path does not already have.
            parts = [p for p in (p_add, p_mul) if p is not None]
            return torch.cat(parts, dim=-1) if len(parts) > 1 else parts[0]

        sig = signal
        B, T = sig.shape[0], sig.shape[1]
        if self.chunk_size > 0:
            # Route once per chunk off its first token: strictly causal, and the
            # knob that measures whether per-token conditioning buys anything
            # over chunk-level when both are equally affordable.
            chunk = self.chunk_size
            n_chunks = (T + chunk - 1) // chunk
            pad = n_chunks * chunk - T
            sig_p = F.pad(sig, (0, 0, 0, pad)) if pad > 0 else sig
            sig = sig_p.reshape(B, n_chunks, chunk, sig.shape[-1])[:, :, 0, :]

        h = norm(sig).to(dtype=dtype)  # rmsnorm: c_proj sees ReLU² activations
        if self.route_down is not None:
            h = h @ self.route_down.to(dtype=dtype)
            if self.route_b is not None:
                h = h + self.route_b.to(dtype=dtype)
                h = F.gelu(h) if self.router_act == 'gelu' else torch.relu(h).square()
        c = h @ self.route_w.to(dtype=dtype)

        if self.chunk_size > 0:
            c = (c.unsqueeze(2)
                  .expand(B, c.shape[1], self.chunk_size, n_coeff)
                  .reshape(B, c.shape[1] * self.chunk_size, n_coeff)[:, :T, :])
        return c

    def _compose(self, x, p_mul, c_mul, dtype):
        """Sequential rank-1 composition, returning q such that z_m = x + U q.

        'wy' solves the coefficient recursion directly and touches the activation
        tensor once; 'loop' is the literal definition, kept as the reference the
        fast path is tested against. m is meant to be small (8 to 32): the compact
        path unrolls O(m²) scalar ops on (B, T) tensors.
        """
        # Unit columns make |c| the exact spectral norm of each rank-1 update,
        # which is what turns the 1/m bound on c into a bound on the product.
        u = F.normalize(self.mul_u.to(dtype=dtype), dim=0)   # (in, m)
        v = F.normalize(self.mul_v.to(dtype=dtype), dim=0)   # (in, m)
        m = self.mult_steps
        if self.mult_impl == 'loop':
            z = x
            qs = []
            for i in range(m):
                q_i = c_mul[..., i] * (z @ v[:, i])          # (B, T)
                z = z + q_i.unsqueeze(-1) * u[:, i]
                qs.append(q_i)
            return torch.stack(qs, dim=-1)                    # (B, T, m)
        # Compact form: q_i = c_i (p_i + Σ_{j<i} S_ij q_j), S = VᵀU strictly lower.
        S = v.transpose(0, 1) @ u                             # (m, m), params only
        qs = []
        for i in range(m):
            acc = p_mul[..., i]
            for j in range(i):
                acc = acc + S[i, j] * qs[j]
            qs.append(c_mul[..., i] * acc)
        return torch.stack(qs, dim=-1)                        # (B, T, m)

    def forward(self, x, context_state=None, route_weights=None, **_ignored):
        dtype = x.dtype
        n_coeff = self.rank + self.mult_steps

        p_add = x @ self.add_v.to(dtype=dtype) if self.rank else None        # (B, T, R)
        # Normalized here too: _compose uses unit columns, and p_mul feeds its
        # recursion directly, so the two must use the same v or the compact path
        # silently computes a different model from the sequential one.
        p_mul = (x @ F.normalize(self.mul_v.to(dtype=dtype), dim=0)) if self.mult_steps else None

        signal = x
        if self.gate_source == 'ctx':
            if context_state is None:
                raise ValueError(
                    "ConditionedLinear(gate_source='ctx') received context_state=None. "
                    "The router is sized for the context dim, so there is no silent "
                    "fallback to x. Attention layers are built with gate_source='router' "
                    "for exactly this reason (see RemixedMultiAttention)."
                )
            signal = context_state
        z = self._coefficients(signal, p_add, p_mul, n_coeff, dtype)
        # Each branch maps the router logits its own way: the additive branch to
        # coefficients that are 1 at zero logits (so its zero-init U is what
        # gives identity), the composition branch to coefficients that are 0 at
        # zero logits and bounded by 1/m (so identity and stability come from c).
        c_add = self._activate(z[..., :self.rank]) if self.rank else None
        c_mul = self._mult_coefficients(z[..., self.rank:], dtype) if self.mult_steps else None
        c = torch.cat([t for t in (c_add, c_mul) if t is not None], dim=-1)

        w0 = self.base_w.to(dtype=dtype)
        y = F.linear(x, w0)
        if self.mult_steps:
            q = self._compose(x, p_mul, c_mul, dtype)                # (B, T, m)
            # y = W₀(x + U q), with W₀U formed once per forward so the activation
            # tensor is read once regardless of m.
            w0u = w0 @ F.normalize(self.mul_u.to(dtype=dtype), dim=0)  # (out, m)
            y = y + q @ w0u.transpose(0, 1)
        if self.rank:
            y = y + (p_add * c_add) @ self.add_u.to(dtype=dtype)

        if self.training and not torch.compiler.is_compiling():
            with torch.no_grad():
                self._record_conditioning(c, n_coeff)
        return y

    @torch.no_grad()
    def _record_conditioning(self, c, n_coeff):
        """Measure the conditioning this layer is actually using.

        cond_dof_pr is the participation ratio of the eigenvalue spectrum of the
        coefficient covariance, (tr Σ)² / tr(Σ²), computed without an eigen-
        decomposition. It is the number this whole design is about: how many
        independent directions the per-token coefficient vector explores, out of
        R+m available. RemixedLinear's ceiling on the same measure is 1 per
        routed factor, because its one coefficient per template is broadcast
        across the entire rank dimension, and a marginal per-coefficient
        variance would score that same broadcast tensor a full R, which is
        exactly the accounting that makes a K× parameter bank look expressive.

        Read it knowing the estimator is biased low. From N token samples of an
        R-dimensional coefficient the sample covariance is Wishart, and even a
        perfectly isotropic field reports R/(1+R/N) rather than R. N is sized at
        16R below so the shortfall stays under ~6%; a reading well under that is
        real collapse, a reading a few percent under it is the estimator. No
        correction is applied because the obvious one (multiplying by 1+R/N)
        would inflate a genuinely rank-1 field from 1.0 to 1.5 and hide the
        failure this metric exists to catch.

        Cost is N·(R+m)², and the whole block is skipped under torch.compile.
        """
        max_rows = max(2048, 16 * n_coeff)
        cf = c.float()
        self._gate_stats['cond_c_mean'] = cf.mean()
        self._gate_stats['cond_c_std']  = cf.std()
        flat = cf.reshape(-1, n_coeff)
        # Deterministic stride rather than a permutation: no RNG, no sync.
        if flat.shape[0] > max_rows:
            flat = flat[:: (flat.shape[0] + max_rows - 1) // max_rows]
        var = flat.var(dim=0)
        # Mean per-coefficient std: 0 means the router has collapsed to a
        # constant and the layer is a static low-rank branch.
        self._gate_stats['cond_tok_std'] = var.mean().sqrt()
        centered = flat - flat.mean(dim=0, keepdim=True)
        cov = (centered.transpose(0, 1) @ centered) / max(flat.shape[0] - 1, 1)
        tr = cov.diagonal().sum()
        self._gate_stats['cond_dof_pr'] = (tr * tr) / cov.square().sum().clamp(min=1e-12)


def _cond_site_enabled(config, site):
    """Is ConditionedLinear active at this site ('attn' or 'ffn')?

    The p35 gain has to come from somewhere. The FFN already has a nonlinearity
    and attention's Q/K/V/O do not, so localizing tells you whether the result is
    'the FFN wanted a better activation' or 'the projections wanted one at all'.
    """
    sites = getattr(config, 'cond_sites', 'both')
    return sites == 'both' or sites == site


def _conditioned_linear_kwargs(config):
    """Shared ConditionedLinear settings, so attention and FFN cannot drift apart."""
    gate_source = getattr(config, 'cond_gate_source', 'router')
    return dict(
        rank=getattr(config, 'cond_rank', 256),
        mult_steps=getattr(config, 'cond_mult_steps', 0),
        gate_source=gate_source,
        coeff_act=getattr(config, 'cond_coeff_act', 'centered'),
        router_rank=getattr(config, 'cond_router_rank', 0),
        router_act=getattr(config, 'cond_router_act', 'none'),
        live_init=getattr(config, 'cond_live_init', 0.0),
        chunk_size=getattr(config, 'cond_chunk_size', 0),
        mult_impl=getattr(config, 'cond_mult_impl', 'wy'),
        mult_scale=getattr(config, 'cond_mult_scale', -1.0),
        # 'ctx' reads what RemixedBlock's bypass path hands the FFN, which is
        # attn_out.detach() at full n_embd, not a projected remix_context_dim
        # vector, and not differentiable back into attention. So the FFN is
        # conditioned on what attention just retrieved, with the conditioning
        # path gradient-isolated from the retrieval that produced it.
        signal_dim=config.n_embd if gate_source == 'ctx' else None,
    )


def _column_quantile(flat, q):
    """Per-column quantile of `flat` (N, K), safe under torch.compile.

    torch.quantile calls numel() on its input, which raises
    "Cannot call numel() on tensor with symbolic sizes/strides" the moment Dynamo
    gives the batch a symbolic shape. That happens as soon as anything varies the
    token count (a prefill sweep, a different device batch, chunk padding), so a
    compiled training run with --p23-quantile-route 1 dies with a TorchRuntimeError
    rather than a clean error. Sorting and indexing gives the same value with no
    numel() call, and keeps the row count symbolic instead of forcing a guard.

    This takes the nearest rank where torch.quantile interpolates linearly, so the
    two can differ by up to one order statistic. That cannot change routing: the
    thresholded set is unioned with the hard top-k set and the final selection is
    the top-k of the scores regardless of what the threshold admitted, so the
    criterion is inert either way (see the ablation in the paper, and
    tests/test_quantile_router_topk.py which pins it).
    """
    srt, _ = torch.sort(flat, dim=0)
    n = srt.shape[0]
    pos = torch.clamp(
        (torch.as_tensor(float(q), device=flat.device, dtype=torch.float32) * (n - 1)).round().long(),
        min=0,
    )
    return srt.index_select(0, pos.reshape(1)).squeeze(0)


def _resolve_topk(topk, K):
    """How many experts a router selects, given a `template_topk` setting.

    topk <= 0 means "all K", i.e. soft mixing over the whole bank. That is the
    convention everywhere else in this file (see RemixedLinear._template_weights
    and both compose branches, which fall through to a plain softmax when topk is
    0 or >= K), and it is what --p22-template-topk 0 means in the sweep scripts.

    The quantile routers previously clamped with max(1, min(topk, K)), which
    turned 0 into 1 and silently made every --p22-template-topk 0 run hard top-1
    with coefficient exactly 1.0. Passing --p22-template-topk 1 explicitly still
    reproduces that behaviour bit for bit, which is the way to re-run anything
    trained before this fix.
    """
    return K if topk <= 0 else min(int(topk), K)


class QuantileBalancedRouter(nn.Module):
    """Per-Batch Quantile Load Balancing Router (No EMA).

    Computes raw affinity scores (B, T, K) and dynamically calculates
    per-batch quantile thresholds to balance expert utilization per step
    without stateful EMA buffers or aux loss penalties.

    ROUTING SCOPE. `route_scope='token'` (the default) scores every position of
    the input independently, so when the chunk-routing path hands this router one
    anchor per chunk it produces one routing decision per chunk, and when the
    per-token path hands it (B, T, D) it produces one per token.

    `route_scope='sequence'` restores the earlier behaviour, which mean-pooled
    the affinity signal over dim 1 and broadcast a single weight vector to every
    position. That had two consequences worth remembering, because every run
    trained before this default changed carries them:

      * the chunk axis was the pooled axis, so all chunks of a sequence received
        the same template and chunk-amortised routing was numerically identical
        to per-sequence routing, whatever the chunk size;
      * pooling spans the whole sequence, so a chunk's routing depended on tokens
        that come after it. Routing was not causal.

    Keep 'sequence' only to reproduce those runs.
    """

    def __init__(self, in_features: int, n_experts: int, topk: int,
                 learned: bool = True, route_scope: str = 'token', **kwargs):
        super().__init__()
        assert route_scope in ('token', 'sequence'), route_scope
        self.n_experts = n_experts
        self.topk      = topk
        self.route_scope = route_scope

        # Affinity projection: (K, D) — frozen or learned
        if learned:
            self.route_proj = nn.Parameter(torch.empty(n_experts, in_features))
            nn.init.normal_(self.route_proj, std=in_features ** -0.5)
        else:
            self.register_buffer('route_proj',
                torch.randn(n_experts, in_features) / (in_features ** 0.5))

    def gate_parameters(self):
        if isinstance(self.route_proj, nn.Parameter):
            yield self.route_proj

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D) → weights: (B, T, K) normalized top-k weights."""
        B, T, D = x.shape
        K = self.n_experts
        topk = _resolve_topk(self.topk, K)

        if self.route_scope == 'sequence':
            # Legacy: one decision per sequence, non-causal. See the class docstring.
            sig = x.float().mean(dim=1, keepdim=True) if x.ndim == 3 else x.float()
        else:
            sig = x.float()
        scores = F.linear(sig, self.route_proj.float())     # (B, T, K), (B, 1, K) or (B, K)
        if scores.ndim == 2:
            scores = scores.unsqueeze(1)
        if scores.shape[1] != T:
            scores = scores.expand(B, T, K)

        if self.training:
            # Per-batch quantile thresholding (no EMA)
            flat = scores.detach().float().reshape(-1, K)
            target_q = 1.0 - topk / K
            thresh = _column_quantile(flat, target_q)                    # (K,)
            q_mask = scores > thresh.unsqueeze(0).unsqueeze(0)           # (B, T, K) bool

            # Guarantee topk via hard top-k union
            _, topk_idx = scores.topk(topk, dim=-1)
            hard_mask   = torch.zeros_like(q_mask).scatter_(-1, topk_idx, True)
            mask        = q_mask | hard_mask

            masked_scores = scores.masked_fill(~mask, -1e9)
            _, top_idx    = masked_scores.topk(topk, dim=-1)
            mask          = torch.zeros_like(mask).scatter_(-1, top_idx, True)

            gated   = scores.masked_fill(~mask, -1e9)
            weights = F.softmax(gated.float(), dim=-1).to(x.dtype)
        else:
            topk_s, topk_idx = scores.topk(topk, dim=-1)
            weights = torch.zeros(B, T, K, device=x.device, dtype=x.dtype)
            weights.scatter_(-1, topk_idx, F.softmax(topk_s.float(), dim=-1).to(x.dtype))

        return weights


class QuantileCrossAttentionRouter(nn.Module):
    """Linear Causal Cross-Attention routing with Per-Batch Quantile Load Balancing (No EMA)."""

    def __init__(self, in_features: int, n_experts: int, topk: int,
                 head_dim: int = 64, **kwargs):
        super().__init__()
        self.n_experts = n_experts
        self.topk = topk
        self.head_dim = head_dim
        
        self.q_exp = nn.Parameter(torch.empty(n_experts, head_dim))
        nn.init.normal_(self.q_exp, std=head_dim ** -0.5)
        
        self.k_proj = nn.Linear(in_features, head_dim, bias=False)
        self.v_proj = nn.Linear(in_features, head_dim, bias=False)
        self.q_tok_proj = nn.Linear(in_features, head_dim, bias=False)

    def gate_parameters(self):
        yield self.q_exp
        yield self.k_proj.weight
        yield self.v_proj.weight
        yield self.q_tok_proj.weight

    def forward(self, x: torch.Tensor, kv_state: dict = None) -> torch.Tensor:
        B, T, D = x.shape
        K, D_h = self.n_experts, self.head_dim
        topk = _resolve_topk(self.topk, K)
        
        x_in = x.to(dtype=self.k_proj.weight.dtype)
        k = self.k_proj(x_in)       # (B, T, D_h)
        v = self.v_proj(x_in)       # (B, T, D_h)
        q_tok = self.q_tok_proj(x_in) # (B, T, D_h)
        
        qk = torch.einsum('bth,kh->btk', k, self.q_exp)
        S = F.elu(qk) + 1.0     # (B, T, K)
        
        if kv_state is not None:
            prev_N = kv_state.get('qca_N', torch.zeros(B, K, D_h, dtype=x.dtype, device=x.device))
            prev_Z = kv_state.get('qca_Z', torch.zeros(B, K, dtype=x.dtype, device=x.device))
            
            N = prev_N + torch.einsum('btk,bth->bkh', S, v)
            Z = prev_Z + S.squeeze(1)
            
            kv_state['qca_N'] = N.clone()
            kv_state['qca_Z'] = Z.clone()
            
            N = N.unsqueeze(1)
            Z = Z.unsqueeze(1)
        else:
            terms_N = S.unsqueeze(-1) * v.unsqueeze(2)
            N = torch.cumsum(terms_N.float(), dim=1).to(x.dtype)
            Z = torch.cumsum(S.float(), dim=1).to(x.dtype)
            
        C = N / (Z.unsqueeze(-1) + 1e-6)
        scores = torch.einsum('btkh,bth->btk', C.to(dtype=q_tok.dtype), q_tok)

        if self.training:
            # Per-batch quantile thresholding (no EMA)
            flat = scores.detach().float().reshape(-1, K)
            target_q = 1.0 - topk / K
            thresh = _column_quantile(flat, target_q)                    # (K,)
            q_mask = scores > thresh.unsqueeze(0).unsqueeze(0)           # (B, T, K) bool

            # Guarantee topk via hard top-k union
            _, topk_idx = scores.topk(topk, dim=-1)
            hard_mask   = torch.zeros_like(q_mask).scatter_(-1, topk_idx, True)
            mask        = q_mask | hard_mask

            masked_scores = scores.masked_fill(~mask, -1e9)
            _, top_idx    = masked_scores.topk(topk, dim=-1)
            mask          = torch.zeros_like(mask).scatter_(-1, top_idx, True)

            gated   = scores.masked_fill(~mask, -1e9)
            weights = F.softmax(gated.float(), dim=-1).to(x.dtype)
        else:
            topk_s, topk_idx = scores.topk(topk, dim=-1)
            weights = torch.zeros(B, T, K, device=x.device, dtype=x.dtype)
            weights.scatter_(-1, topk_idx, F.softmax(topk_s.float(), dim=-1).to(x.dtype))

        return weights


class LinearMoE(nn.Module):

    """Weight-space Mixture of Experts — blends K expert weight matrices before a single F.linear call.

    Route: W_eff[b,t] = Σ_k w_k(x[b,t]) * W_k  then  y[b,t] = F.linear(x[b,t], W_eff[b,t])

    Properties:
      - Fully differentiable everywhere (no discrete routing decisions)
      - No dead-expert gradient problem (all weights get gradient via chain rule)
      - Compatible with torch.compile (no Python-level branches on dynamic tensors)
      - Efficient: soft blending in weight-space → one batched einsum, not K forward passes
    """
    def __init__(self, in_features: int, out_features: int, n_experts: int = 8,
                 topk: int = 0, bias: bool = True, use_quantile_route: int = 0):
        super().__init__()
        self.n_experts = n_experts
        self.topk = topk  # 0 = soft (all experts), N = hard top-N
        # Expert weight bank: (K, out_features, in_features)
        self.experts_w = nn.Parameter(torch.empty(n_experts, out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        # Router: quantile-balanced or simple learned softmax
        if use_quantile_route == 2 and topk > 0:
            self.router = QuantileCrossAttentionRouter(in_features, n_experts, topk)
            self._use_quantile = True
        elif use_quantile_route == 1 and topk > 0:
            self.router = QuantileBalancedRouter(in_features, n_experts, topk, learned=True)
            self._use_quantile = True
        else:
            self.router = nn.Linear(in_features, n_experts, bias=False)
            nn.init.zeros_(self.router.weight)
            self._use_quantile = False
        # Kaiming init for expert weights
        nn.init.kaiming_uniform_(self.experts_w.view(n_experts * out_features, in_features),
                                 a=math.sqrt(5))
        self.experts_w.data = self.experts_w.data.view(n_experts, out_features, in_features)
        # Compat attrs expected by setup_optimizer / diagnostics
        self.ln_basis = nn.Identity()

    def gate_parameters(self):
        if self._use_quantile:
            yield from self.router.gate_parameters()
        else:
            yield from self.router.parameters()

    def non_gate_parameters(self):
        yield self.experts_w
        if self.bias is not None:
            yield self.bias

    def forward(self, x, context_state=None, route_weights=None, **kwargs):
        """x: (B, T, in_features) → (B, T, out_features)."""
        B, T, _ = x.shape
        if self._use_quantile:
            w = self.router(x)                                           # (B, T, K)
        else:
            logits = F.linear(x, self.router.weight.to(x.dtype))        # (B, T, K)
            if self.topk > 0 and self.topk < self.n_experts:
                topk_w, topk_idx = logits.topk(self.topk, dim=-1)
                w = torch.zeros_like(logits).scatter_(-1, topk_idx,
                                                      F.softmax(topk_w, dim=-1))
            else:
                w = F.softmax(logits, dim=-1)                           # (B, T, K) soft
        # Blend expert weights in parameter space: W_eff = Σ_k w_k * W_k
        W_eff = torch.einsum('btk,koi->btoi',
                             w.float(), self.experts_w.float()).to(x.dtype)  # (B, T, out, in)
        # Single batched matmul (no expert loop)
        out = torch.einsum('bti,btoi->bto', x, W_eff)
        if self.bias is not None:
            out = out + self.bias
        return out

class SlicedWeightLinear(nn.Module):
    """Weight-bank column slicing with Product-Key routing.

    The weight bank is the same shape as a dense linear: (out_features, in_features).
    A Product-Key router selects n_selected column indices per routing decision:
        n_selected = max(min_select, in_features // reduction_scale)
    Only those (input_dim, weight_column) pairs are computed, giving a compute reduction
    of n_selected / in_features vs dense while keeping parameter count identical.

    Scope:
        per_token  — each token selects its own columns (T independent routing decisions)
        per_block  — one column set per sequence, shared across all T tokens
        global     — one column set shared across all layers in the forward pass
    """
    def __init__(self, in_features: int, out_features: int, reduction_scale: int = 8,
                 min_select: int = 128, scope: str = "per_token",
                 balance_coeff: float = 0.01, bias: bool = True, use_quantile_route: int = 0,
                 signal_dim: int = 0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.reduction_scale = max(1, int(reduction_scale))
        self.min_select = max(1, int(min_select))
        self.scope = scope
        self.balance_coeff = float(balance_coeff)
        self.use_quantile_route = int(use_quantile_route)
        # Weight bank: same shape as a dense linear — no parameter bloat
        self.weight_bank = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        # n_selected: columns (and matching input dims) active per routing decision
        # = max(min_select_floor, in_features // reduction_scale)
        self.n_selected = min(in_features, max(self.min_select, in_features // self.reduction_scale))
        # Product-Key router: n_keys = ceil(sqrt(in_features))
        n_keys = int(math.isqrt(in_features))
        if n_keys * n_keys < in_features:
            n_keys += 1
        self.n_keys = n_keys
        # Local routers (per_token): input dim = in_features
        self.router_a = Linear(in_features, n_keys, bias=False)
        self.router_b = Linear(in_features, n_keys, bias=False)
        # Small random init so EMA balance loss is non-zero from step 1.
        # Zero init → uniform probs → imbalance=log(1)=0 → zero gradient → router never trains.
        nn.init.normal_(self.router_a.weight, std=0.02)
        nn.init.normal_(self.router_b.weight, std=0.02)
        # External routers (per_block/global): input dim = signal_dim (n_embd)
        self.signal_dim = signal_dim if signal_dim > 0 else in_features
        self.ext_router_a = Linear(self.signal_dim, n_keys, bias=False)
        self.ext_router_b = Linear(self.signal_dim, n_keys, bias=False)
        nn.init.normal_(self.ext_router_a.weight, std=0.02)
        nn.init.normal_(self.ext_router_b.weight, std=0.02)
        nn.init.kaiming_uniform_(self.weight_bank, a=math.sqrt(5))
        self.register_buffer("usage_a", torch.full((n_keys,), 1.0 / n_keys))
        self.register_buffer("usage_b", torch.full((n_keys,), 1.0 / n_keys))
        self._last_balance_loss = None
        self.ln_basis = nn.Identity()

    def _compute_balance_loss(self, logits_a: torch.Tensor, logits_b: torch.Tensor, dtype: torch.dtype):
        """EMA-based balance loss: penalise routing to already-overused sub-keys."""
        probs_a = F.softmax(logits_a.float(), dim=-1)
        probs_b = F.softmax(logits_b.float(), dim=-1)
        with torch.no_grad():
            self.usage_a.mul_(0.99).add_(0.01 * probs_a.mean(dim=(0, 1)))
            self.usage_b.mul_(0.99).add_(0.01 * probs_b.mean(dim=(0, 1)))
        target = 1.0 / self.n_keys
        imbalance_a = (self.usage_a / target).log()
        imbalance_b = (self.usage_b / target).log()
        aux_a = (probs_a * imbalance_a.detach()).mean()
        aux_b = (probs_b * imbalance_b.detach()).mean()
        self._last_balance_loss = (aux_a + aux_b).to(dtype=dtype)

    def forward(self, x, context_state=None, route_weights=None, p24_route_input=None, **kwargs):
        B, T, C = x.shape
        use_external = p24_route_input is not None

        if use_external:
            # per_block / global: route once from pooled signal, share column selection across T
            ri = p24_route_input
            if ri.ndim == 3:
                ri = ri.mean(dim=1)          # (B, signal_dim)
            route_input = ri.unsqueeze(1)    # (B, 1, signal_dim)
            router_a, router_b = self.ext_router_a, self.ext_router_b
        else:
            # per_token: each token selects its own columns
            route_input = x                  # (B, T, in_features)
            router_a, router_b = self.router_a, self.router_b

        logits_a = router_a(route_input)     # (B, T_r, n_keys)  T_r=T or 1
        logits_b = router_b(route_input)
        self._compute_balance_loss(logits_a, logits_b, x.dtype)

        # Product-Key: top-k_a × top-k_b → column indices in [0, in_features)
        topk_a = max(1, int(math.isqrt(self.n_selected)))
        topk_b = max(1, self.n_selected // topk_a)
        topk_a_c = min(topk_a, self.n_keys)
        topk_b_c = min(topk_b, self.n_keys)
        idx_a = logits_a.topk(topk_a_c, dim=-1).indices   # (B, T_r, topk_a)
        idx_b = logits_b.topk(topk_b_c, dim=-1).indices   # (B, T_r, topk_b)
        T_r = idx_a.size(1)
        pair_idx = (idx_a.unsqueeze(-1) * self.n_keys + idx_b.unsqueeze(-2))  # (B, T_r, topk_a, topk_b)
        pair_idx = pair_idx.reshape(B, T_r, -1) % self.in_features            # (B, T_r, n_pairs)
        n_pairs = pair_idx.size(-1)
        if n_pairs < self.n_selected:
            rep = pair_idx[..., :1].expand(-1, -1, self.n_selected - n_pairs)
            pair_idx = torch.cat([pair_idx, rep], dim=-1)
        pair_idx = pair_idx[..., :self.n_selected]         # (B, T_r, n_selected)

        # Expand shared column selection to all T tokens (per_block / global)
        if T_r == 1 and T > 1:
            pair_idx = pair_idx.expand(-1, T, -1)          # (B, T, n_selected)

        # Gather matching input dims from x (same indices as weight columns)
        x_sel = torch.gather(x, 2, pair_idx)               # (B, T, n_selected)

        # Gather weight columns from bank: weight_bank is (out, in_features)
        flat_idx = pair_idx.reshape(B * T, self.n_selected)           # (B*T, n_selected)
        W_slice = self.weight_bank.t()[flat_idx].to(x.dtype)          # (B*T, n_selected, out)

        x_flat = x_sel.reshape(B * T, 1, self.n_selected)             # (B*T, 1, n_selected)
        out = torch.bmm(x_flat, W_slice).squeeze(1)                   # (B*T, out)
        out = out.reshape(B, T, self.out_features)
        if self.bias is not None:
            out = out + self.bias
        return out


class FoldedModulationLinear(nn.Module):
    """LinearMoE3-style fold-by-sum + modulation on folded channels.

    Plan spec (linear_moe_variants_plan.md §2):
    - weight: (out_features, C//R)  ← genuine parameter reduction
    - Fold: x_fold = x.view(B, T, C//R, R).sum(-1)   (B, T, C//R)
    - Gate: pool = x_fold.mean(dim=1); gate = sigmoid(gate_proj(pool))  (B, C//R)
    - x_mod = x_fold * gate.unsqueeze(1)
    - out = F.linear(x_mod, weight, bias)
    For per_block / global scope, p24_gate_input injects the pre-computed gate logits.
    """
    def __init__(self, in_features: int, out_features: int, reduction_scale: int = 8,
                 scope: str = "per_layer", gate_act: str = "sigmoid", bias: bool = True,
                 signal_dim: int = 0, min_folded_dim: int = 0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.reduction_scale = max(1, int(reduction_scale))
        self.scope = scope
        self.gate_act = gate_act
        # Compute folded_dim with optional minimum floor.
        # If folded_dim_raw < min_folded_dim, reduce effective_R so folded_dim >= min_folded_dim.
        folded_dim_raw = max(1, in_features // self.reduction_scale)
        if min_folded_dim > 0 and folded_dim_raw < min_folded_dim:
            effective_R = max(1, in_features // min_folded_dim)
        else:
            effective_R = self.reduction_scale
        self.effective_R = effective_R
        self.folded_dim = max(1, in_features // self.effective_R)
        self.used_in = self.folded_dim * self.effective_R  # may be < in_features if not divisible
        # Weight operates on folded (compressed) input: genuine param reduction vs dense
        self.weight = nn.Parameter(torch.empty(out_features, self.folded_dim))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        # Local gate: folded_dim → folded_dim (used for per_layer scope)
        self.gate_proj = Linear(self.folded_dim, self.folded_dim, bias=True)
        # External gate: signal_dim → folded_dim (per_block/global; signal is always n_embd, not in_features)
        self.signal_dim = signal_dim if signal_dim > 0 else in_features
        self.ext_gate_proj = Linear(self.signal_dim, self.folded_dim, bias=True)
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.zeros_(self.gate_proj.bias)
        nn.init.zeros_(self.ext_gate_proj.weight)
        nn.init.zeros_(self.ext_gate_proj.bias)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.ln_basis = nn.Identity()

    def _act(self, g: torch.Tensor) -> torch.Tensor:
        if self.gate_act == "tanh_centered":
            return 1.0 + torch.tanh(g)
        return torch.sigmoid(g)

    def forward(self, x, context_state=None, route_weights=None, p24_gate_input=None, **kwargs):
        B, T, C = x.shape
        # Step 1: fold by summing consecutive groups of effective_R dims
        x_use = x[..., :self.used_in]                                             # (B, T, used_in)
        x_fold = x_use.view(B, T, self.folded_dim, self.effective_R).sum(-1)      # (B, T, folded_dim)
        # Step 2: compute gate
        if p24_gate_input is None:
            # per_layer: pool x_fold over T, project to gate
            pool = x_fold.mean(dim=1)                                             # (B, folded_dim)
            # Use F.linear directly to bypass Float8Linear wrapper (FP8 needs row-dim % 16)
            gate_logits = F.linear(pool.float(),
                                   self.gate_proj.weight.float(),
                                   self.gate_proj.bias.float())                   # (B, folded_dim)
        else:
            # per_block / global: external input (B, C) or (B, 1, C) → project to folded_dim
            ext = p24_gate_input
            if ext.ndim == 3:
                ext = ext.mean(dim=1)                                             # (B, C)
            gate_logits = F.linear(ext.float(),
                                   self.ext_gate_proj.weight.float(),
                                   self.ext_gate_proj.bias.float())               # (B, folded_dim)
        gate = self._act(gate_logits).to(x.dtype).unsqueeze(1)                   # (B, 1, folded_dim)
        # Step 3: modulate and project
        x_mod = x_fold * gate                                                     # (B, T, folded_dim)
        return F.linear(x_mod, self.weight.to(x.dtype),
                        self.bias.to(x.dtype) if self.bias is not None else None)


class SequenceGatedLinear(nn.Module):
    """Dense linear with sequence-pooled content gate."""
    def __init__(self, in_features: int, out_features: int, gate_act: str = "sigmoid", bias: bool = True,
                 signal_dim: int = 0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.gate_act = gate_act
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        # Local gate (per_layer): in_features → in_features
        self.gate_proj = Linear(in_features, in_features, bias=True)
        # External gate (per_block/global): signal_dim (n_embd) → in_features
        self.signal_dim = signal_dim if signal_dim > 0 else in_features
        self.ext_gate_proj = Linear(self.signal_dim, in_features, bias=True)
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.zeros_(self.gate_proj.bias)
        nn.init.zeros_(self.ext_gate_proj.weight)
        nn.init.zeros_(self.ext_gate_proj.bias)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.ln_basis = nn.Identity()

    def _act(self, g: torch.Tensor) -> torch.Tensor:
        if self.gate_act == "tanh_centered":
            return 1.0 + torch.tanh(g)
        return torch.sigmoid(g)

    def forward(self, x, context_state=None, route_weights=None, p24_gate_input=None, **kwargs):
        if p24_gate_input is None:
            pooled = x.mean(dim=1)                                    # (B, in_features)
            proj_w = self.gate_proj.weight.float()
            proj_b = self.gate_proj.bias.float()
        else:
            ext = p24_gate_input
            pooled = ext.mean(dim=1) if ext.ndim == 3 else ext       # (B, signal_dim)
            proj_w = self.ext_gate_proj.weight.float()
            proj_b = self.ext_gate_proj.bias.float()
        # Use F.linear directly to bypass Float8Linear wrapper (FP8 needs row-dim % 16)
        gate = self._act(
            F.linear(pooled.float(), proj_w, proj_b)
        ).to(x.dtype).unsqueeze(1)                                    # (B, 1, in_features)
        return F.linear(x * gate, self.weight.to(x.dtype),
                        self.bias.to(x.dtype) if self.bias is not None else None)


class _FFNPassThrough(nn.Module):
    """Stands in for `c_proj` when the FFN is a single layer.

    A None would be cleaner but every consumer of the block iterates
    [c_q, c_k, c_v, c_proj, c_fc, c_proj] and would need a guard: GPT.init_weights,
    setup_optimizer, RemixedBlock's shared-basis list and the diagnostics. A
    parameter-free module falls through all of them: it has no `.weight`, no
    `gate_parameters`, and `parameters()` yields nothing.
    """
    def forward(self, x, *args, **kwargs):
        return x


class _DenseAttnAdapter(nn.Module):
    """Plain CausalSelfAttention with RemixedMultiAttention's call signature.

    Used by p34_dense_attn to hold attention fixed while the FFN varies, so an FFN
    result is not confounded by attention also being remixed. The two forwards
    differ only in the trailing arguments (context_state / route_weights vs
    token_active), so the adapter drops what dense attention has no use for.
    Exposes c_q/c_k/c_v/c_proj and ve_gate so the init and optimizer paths, which
    reach into block.attn by name, keep working.
    """
    def __init__(self, config, layer_idx):
        super().__init__()
        self.inner = CausalSelfAttention(config, layer_idx)

    def __getattr__(self, name):
        if name in ("c_q", "c_k", "c_v", "c_proj", "ve_gate", "head_dim", "n_head"):
            return getattr(self._modules["inner"], name)
        return super().__getattr__(name)

    def forward(self, x, ve, cos_sin, window_size, kv_cache, context_state=None,
                route_weights=None, **kwargs):
        return self.inner(x, ve, cos_sin, window_size, kv_cache)


class RemixedFeedForward(nn.Module):
    """Feedforward path using RemixedLinear (or RAL) in the base MLP framework."""
    def __init__(self, config, layer_idx=0):
        super().__init__()
        self._sched_mult = resolve_ffn_schedule(config, config.n_layer)[layer_idx]
        use_ral = getattr(config, 'use_ral', False)
        mode = getattr(config, 'cclblock_modulation', 'weight')
        use_decoupled = mode == 'decoupled'
        ral_rank = getattr(config, 'ral_rank', 32)
        film_gate = getattr(config, 'cclblock_film_gate', False)
        kwargs = config.remixed_linear_kwargs if config.remixed_linear_kwargs is not None else dict(use_basis_gate=True, use_output_gate=True, use_context=True)
        scale = getattr(config, 'scale_basis_size', True)
        use_p24_sliced = bool(getattr(config, 'p24_use_sliced_weight', 0))
        use_p24_folded = bool(getattr(config, 'p24_use_folded_mod', 0))
        use_p24_seqgate = bool(getattr(config, 'p24_use_sequence_gated_linear', 0))
        # Phase 23 LinearMoE mode: weight-space blending of K expert matrices
        linear_moe_k = getattr(config, 'p23_linear_moe_experts', 0)
        use_qroute = bool(getattr(config, 'p23_quantile_route', 0))
        if use_p24_sliced:
            self.c_fc = SlicedWeightLinear(
                config.n_embd, 4 * config.n_embd,
                reduction_scale=getattr(config, 'p24_sliced_weight_reduction_scale', 8),
                min_select=getattr(config, 'p24_sliced_weight_min_select', 128),
                scope=getattr(config, 'p24_sliced_weight_scope', 'per_token'),
                balance_coeff=getattr(config, 'p24_sliced_weight_balance_coeff', 0.01),
                use_quantile_route=getattr(config, 'p24_quantile_route', 0),
                signal_dim=config.n_embd,
            )
            self.c_proj = SlicedWeightLinear(
                4 * config.n_embd, config.n_embd,
                reduction_scale=getattr(config, 'p24_sliced_weight_reduction_scale', 8),
                min_select=getattr(config, 'p24_sliced_weight_min_select', 128),
                scope=getattr(config, 'p24_sliced_weight_scope', 'per_token'),
                balance_coeff=getattr(config, 'p24_sliced_weight_balance_coeff', 0.01),
                use_quantile_route=getattr(config, 'p24_quantile_route', 0),
                signal_dim=config.n_embd,
            )
        elif use_p24_folded:
            self.c_fc = FoldedModulationLinear(
                config.n_embd, 4 * config.n_embd,
                reduction_scale=getattr(config, 'p24_folded_mod_reduction_scale', 8),
                min_folded_dim=getattr(config, 'p24_folded_mod_min_dim', 128),
                scope=getattr(config, 'p24_folded_mod_scope', 'per_layer'),
                gate_act=getattr(config, 'p24_folded_mod_gate_act', 'sigmoid'),
                signal_dim=config.n_embd,
            )
            self.c_proj = FoldedModulationLinear(
                4 * config.n_embd, config.n_embd,
                reduction_scale=getattr(config, 'p24_folded_mod_reduction_scale', 8),
                min_folded_dim=getattr(config, 'p24_folded_mod_min_dim', 128),
                scope=getattr(config, 'p24_folded_mod_scope', 'per_layer'),
                gate_act=getattr(config, 'p24_folded_mod_gate_act', 'sigmoid'),
                signal_dim=config.n_embd,
            )
        elif use_p24_seqgate:
            self.c_fc = SequenceGatedLinear(
                config.n_embd, 4 * config.n_embd,
                gate_act=getattr(config, 'p24_sequence_gated_act', 'sigmoid'),
                signal_dim=config.n_embd,
            )
            self.c_proj = SequenceGatedLinear(
                4 * config.n_embd, config.n_embd,
                gate_act=getattr(config, 'p24_sequence_gated_act', 'sigmoid'),
                signal_dim=config.n_embd,
            )
        elif linear_moe_k > 0:
            linear_moe_topk = getattr(config, 'p23_linear_moe_topk', 0)
            self.c_fc   = LinearMoE(config.n_embd, 4 * config.n_embd, n_experts=linear_moe_k, topk=linear_moe_topk, use_quantile_route=use_qroute)
            self.c_proj = LinearMoE(4 * config.n_embd, config.n_embd, n_experts=linear_moe_k, topk=linear_moe_topk, use_quantile_route=use_qroute)
        elif mode == 'fsi':
            n_rot = getattr(config, 'cclblock_fsi_rotations', 8)
            sel_dim = getattr(config, 'cclblock_fsi_selector_dim', 64)
            self.c_fc   = FrozenSubspaceIndexedLinear(config.n_embd, 4 * config.n_embd, n_rotations=n_rot, selector_dim=sel_dim, signal_dim=config.n_embd)
            self.c_proj = FrozenSubspaceIndexedLinear(4 * config.n_embd, config.n_embd, n_rotations=n_rot, selector_dim=sel_dim, signal_dim=config.n_embd)
        elif mode == 'aesp':
            n_str = getattr(config, 'cclblock_aesp_strata', 4)
            d_rank = getattr(config, 'cclblock_aesp_delta_rank', 4)
            self.c_fc   = AttentionEntropyStratifiedLinear(config.n_embd, 4 * config.n_embd, n_strata=n_str, delta_rank=d_rank)
            self.c_proj = AttentionEntropyStratifiedLinear(4 * config.n_embd, config.n_embd, n_strata=n_str, delta_rank=d_rank)
        elif mode == 'cond' and not _cond_site_enabled(config, 'ffn'):
            _h = max(1, int(round(config.n_embd * self._sched_mult)))
            self.c_fc   = _DenseProj(config.n_embd, _h)
            self.c_proj = _DenseProj(_h, config.n_embd)
        elif mode == 'cond' and _cond_site_enabled(config, 'ffn'):
            # Phase 35: hidden width follows p34_ffn_mult so the FFN shape stays a
            # free knob, and zero_init_base mirrors the dense baseline's init
            # (in-projection uniform, out-projection zeros).
            ck = _conditioned_linear_kwargs(config)
            _h = max(1, int(round(config.n_embd * self._sched_mult)))
            self.c_fc   = ConditionedLinear(config.n_embd, _h, zero_init_base=False, **ck)
            self.c_proj = ConditionedLinear(_h, config.n_embd, zero_init_base=True, **ck)
        elif mode == 'ckr' or mode == 'ckr_ffn':
            n_br = getattr(config, 'cclblock_ckr_branches', 4)
            ksz = getattr(config, 'cclblock_ckr_kernel_size', 64)
            n_pc = getattr(config, 'cclblock_ckr_pos_channels', 1)
            cb_scale = getattr(config, 'cclblock_ckr_content_bias', 0.0)
            temp = getattr(config, 'cclblock_ckr_temp_start', 1.0)
            ortho = bool(getattr(config, 'cclblock_ckr_ortho_init', 0))
            bdrop = float(getattr(config, 'cclblock_ckr_branch_dropout', 0.0))
            ckr_common = dict(n_branches=n_br, kernel_size=ksz, max_seq_len=config.sequence_len, n_pos_channels=n_pc, content_bias_scale=cb_scale, signal_dim=config.n_embd, temperature=temp, ortho_init=ortho, branch_dropout=bdrop)
            self.c_fc   = CausalKernelLinear(config.n_embd, 4 * config.n_embd, **ckr_common)
            self.c_proj = CausalKernelLinear(4 * config.n_embd, config.n_embd, **ckr_common)
        elif mode == 'pgr':
            pgr_ks = getattr(config, 'cclblock_pgr_kernel_size', 64)
            self.c_fc   = PositionGatedResidual(config.n_embd, 4 * config.n_embd, kernel_size=pgr_ks, max_seq_len=config.sequence_len)
            self.c_proj = PositionGatedResidual(4 * config.n_embd, config.n_embd, kernel_size=pgr_ks, max_seq_len=config.sequence_len)
        elif mode == 'cil':
            cil_ks = getattr(config, 'cclblock_cil_kernel_size', 64)
            self.c_fc   = CausalInterpolationLinear(config.n_embd, 4 * config.n_embd, kernel_size=cil_ks, max_seq_len=config.sequence_len)
            self.c_proj = CausalInterpolationLinear(4 * config.n_embd, config.n_embd, kernel_size=cil_ks, max_seq_len=config.sequence_len)
        elif mode == 'prb':
            prb_ks = getattr(config, 'cclblock_prb_kernel_size', 64)
            self.c_fc   = PositionalResidualBias(config.n_embd, 4 * config.n_embd, kernel_size=prb_ks, max_seq_len=config.sequence_len)
            self.c_proj = PositionalResidualBias(4 * config.n_embd, config.n_embd, kernel_size=prb_ks, max_seq_len=config.sequence_len)
        elif mode == 'arg':
            # 18A: Adaptive ReLU² Gate — content-dependent per-channel gate
            self.c_fc   = AdaptiveGatedLinear(config.n_embd, 4 * config.n_embd)
            self.c_proj = AdaptiveGatedLinear(4 * config.n_embd, config.n_embd)
        elif mode == 'kfl':
            # 18C: Kronecker-Factored Linear — fewer params than dense
            self.c_fc   = KroneckerLinear(config.n_embd, 4 * config.n_embd)
            self.c_proj = KroneckerLinear(4 * config.n_embd, config.n_embd)
        elif mode == 'com':
            com_ks = getattr(config, 'cclblock_com_kernel_size', 32)
            self.c_fc   = CausalOutputMixer(config.n_embd, 4 * config.n_embd, kernel_size=com_ks, max_seq_len=config.sequence_len)
            self.c_proj = CausalOutputMixer(4 * config.n_embd, config.n_embd, kernel_size=com_ks, max_seq_len=config.sequence_len)
        elif mode == 'giad':
            giad_rank = getattr(config, 'cclblock_giad_rank', 32)
            self.c_fc   = GradientIsolatedDeltaLinear(config.n_embd, 4 * config.n_embd, delta_rank=giad_rank)
            self.c_proj = GradientIsolatedDeltaLinear(4 * config.n_embd, config.n_embd, delta_rank=giad_rank)
        elif mode == 'psg':
            psg_ks = getattr(config, 'cclblock_psg_kernel_size', 64)
            self.c_fc   = PositionalScalarGatedLinear(config.n_embd, 4 * config.n_embd, kernel_size=psg_ks, max_seq_len=config.sequence_len)
            self.c_proj = PositionalScalarGatedLinear(4 * config.n_embd, config.n_embd, kernel_size=psg_ks, max_seq_len=config.sequence_len)
        elif mode == 'splitstream':
            ss_ratio = float(getattr(config, 'cclblock_ss_dynamic_ratio', 0.25))
            ss_br = getattr(config, 'cclblock_ss_branches', 2)
            ss_ks = getattr(config, 'cclblock_ss_kernel_size', 64)
            self.c_fc   = SplitStreamLinear(config.n_embd, 4 * config.n_embd, dynamic_ratio=ss_ratio, n_branches=ss_br, kernel_size=ss_ks, max_seq_len=config.sequence_len)
            self.c_proj = SplitStreamLinear(4 * config.n_embd, config.n_embd, dynamic_ratio=ss_ratio, n_branches=ss_br, kernel_size=ss_ks, max_seq_len=config.sequence_len)
        elif mode == 'lokr':
            lokr_br = getattr(config, 'cclblock_lokr_branches', 8)
            lokr_r = getattr(config, 'cclblock_lokr_rank', 16)
            lokr_ks = getattr(config, 'cclblock_ckr_kernel_size', 64)
            self.c_fc   = LoKRLinear(config.n_embd, 4 * config.n_embd, n_branches=lokr_br, rank=lokr_r, kernel_size=lokr_ks, max_seq_len=config.sequence_len)
            self.c_proj = LoKRLinear(4 * config.n_embd, config.n_embd, n_branches=lokr_br, rank=lokr_r, kernel_size=lokr_ks, max_seq_len=config.sequence_len)
        elif mode == 'tucker':
            self.c_fc = TuckerAdaptiveLinear(config.n_embd, 4 * config.n_embd, context_dim=config.remix_context_dim, tucker_rank=getattr(config, 'cclblock_tucker_rank', 32), tucker_modes=getattr(config, 'cclblock_tucker_modes', 8))
            self.c_proj = TuckerAdaptiveLinear(4 * config.n_embd, config.n_embd, context_dim=config.remix_context_dim, tucker_rank=getattr(config, 'cclblock_tucker_rank', 32), tucker_modes=getattr(config, 'cclblock_tucker_modes', 8))
        elif mode in ('svs', 'dcu'):
            self.c_fc = SingularValueSteeringLinear(config.n_embd, 4 * config.n_embd, context_dim=config.remix_context_dim, svs_rank=getattr(config, 'cclblock_svs_rank', 64), svs_eps=getattr(config, 'cclblock_svs_eps', 0.1), dcu_warmup_steps=getattr(config, 'cclblock_dcu_warmup_steps', 0) if mode == 'dcu' else 0)
            self.c_proj = SingularValueSteeringLinear(4 * config.n_embd, config.n_embd, context_dim=config.remix_context_dim, svs_rank=getattr(config, 'cclblock_svs_rank', 64), svs_eps=getattr(config, 'cclblock_svs_eps', 0.1), dcu_warmup_steps=getattr(config, 'cclblock_dcu_warmup_steps', 0) if mode == 'dcu' else 0)
        elif mode == 'vq':
            self.c_fc = VQAdaptiveLinear(config.n_embd, 4 * config.n_embd, context_dim=config.remix_context_dim, vq_codes=getattr(config, 'cclblock_vq_codes', 8), vq_temperature=getattr(config, 'cclblock_vq_temperature', 1.0))
            self.c_proj = VQAdaptiveLinear(4 * config.n_embd, config.n_embd, context_dim=config.remix_context_dim, vq_codes=getattr(config, 'cclblock_vq_codes', 8), vq_temperature=getattr(config, 'cclblock_vq_temperature', 1.0))
        elif use_decoupled:
            ratio = float(getattr(config, 'cclblock_dynamic_ratio', 0.25))
            gate_rank = int(getattr(config, 'cclblock_gate_rank', 8))
            self.c_fc   = DecoupledAdaptiveLinear(config.n_embd, 4 * config.n_embd, context_dim=config.remix_context_dim, dynamic_ratio=ratio, basis_size=config.remix_basis_size, gate_rank=gate_rank)
            self.c_proj = DecoupledAdaptiveLinear(4 * config.n_embd, config.n_embd, context_dim=config.remix_context_dim, dynamic_ratio=ratio, basis_size=config.remix_basis_size, gate_rank=gate_rank)
        elif use_ral:
            self.c_fc   = ResidualAdaptiveLinear(config.n_embd, 4 * config.n_embd, context_dim=config.remix_context_dim, rank=ral_rank)
            self.c_proj = ResidualAdaptiveLinear(4 * config.n_embd, config.n_embd, context_dim=config.remix_context_dim, rank=ral_rank)
        else:
            # Enrich kwargs with Phase 23 LoKR / Tiny Expert settings
            lokr_expert = bool(getattr(config, 'p23_lokr', 0))
            tiny_expert = bool(getattr(config, 'p23_tiny_expert', 0))
            if lokr_expert or tiny_expert or True:  # always copy to add p28 fields safely
                kwargs = dict(kwargs)  # shallow copy to avoid mutating shared dict
                kwargs['use_quantile_route'] = int(getattr(config, 'p23_quantile_route', 0))
                kwargs['template_topk'] = int(getattr(config, 'p22_template_topk', kwargs.get('template_topk', 0)))
                kwargs['route_affine'] = int(getattr(config, 'p22_route_affine', 0))
                kwargs['n_templates'] = int(getattr(config, 'p22_n_templates', kwargs.get('n_templates', 1)))
            # Phase 28: tag with layer_role and p28 routing settings
            kwargs['layer_role'] = 'ffn'
            kwargs['global_bank_mode'] = getattr(config, 'p28_global_template_bank', 'none')
            kwargs['chunk_routing_size'] = getattr(config, 'p28_chunk_routing_size', 0)
            if lokr_expert:
                kwargs['lokr_expert'] = True
                kwargs['lokr_n_experts'] = getattr(config, 'p23_n_experts', 64)
                kwargs['lokr_topk'] = getattr(config, 'p23_topk', 16)
                kwargs['lokr_rank'] = getattr(config, 'p23_lokr_rank', 4)
                kwargs['lokr_learned'] = bool(getattr(config, 'p23_learned_route', 0))
            elif tiny_expert:
                kwargs['tiny_expert'] = True
                kwargs['tiny_expert_topk'] = getattr(config, 'p23_topk', 16)
            # Auto-select RemixedLinearFused for 29C chunk routing only if explicitly enabled via remix_use_fused
            _is_29c_fused = (
                getattr(config, 'remix_use_fused', False)
                # RemixedLinearFused has its own inline routing and does not
                # implement the affine path, so the flag would silently no-op.
                and not int(getattr(config, 'p22_route_affine', 0))
                and kwargs.get('n_templates', 1) > 1
                and int(kwargs.get('chunk_routing_size', 0)) > 0
                and not lokr_expert and not tiny_expert
                and kwargs.get('global_bank_mode', 'none') == 'none'
            )
            _linear_cls = (
                DualGateLinear if getattr(config, 'remix_use_dual_gate', False)
                else OutputGatedLinear if getattr(config, 'p26_output_gated_linear', 0)
                else RemixedLinearFused if _is_29c_fused
                else RemixedLinear
            )
            # Phase 34: the hidden width is a knob, not a constant. The dense FFN
            # needs 4x because a static matrix has to cover every context with one
            # set of weights; a per-token effective operator may not. p34_ffn_single
            # goes further and drops the hidden layer entirely, leaving one
            # RemixedLinear D->D whose LayerNorm and output gate supply the
            # nonlinearity that relu^2 supplied before.
            _hidden = max(1, int(round(config.n_embd * self._sched_mult)))
            self._single = bool(getattr(config, 'p34_ffn_single', 0))
            if self._single:
                self.c_fc = _linear_cls(config.n_embd, config.n_embd, context_dim=config.remix_context_dim, basis_size=config.remix_basis_size, remixed_linear_kwargs=kwargs, scale_basis=scale, film_gate=film_gate, routing_scope='per_sequence')
                self.c_proj = _FFNPassThrough()
            else:
                self.c_fc   = _linear_cls(config.n_embd, _hidden, context_dim=config.remix_context_dim, basis_size=config.remix_basis_size, remixed_linear_kwargs=kwargs, scale_basis=scale, film_gate=film_gate, routing_scope='per_sequence')
                self.c_proj = _linear_cls(_hidden, config.n_embd, context_dim=config.remix_context_dim, basis_size=config.remix_basis_size, remixed_linear_kwargs=kwargs, scale_basis=scale, film_gate=film_gate, routing_scope='per_sequence')

        # 18I: Dynamic Activation (learned mix of ReLU², GELU, SiLU)
        self.dynamic_act = DynamicActivation() if getattr(config, 'p18_dynamic_activation', 0) else None
        # 18F: Per-Channel Scale after projection
        self.channel_scale = PerChannelScale(config.n_embd) if getattr(config, 'p18_per_channel_scale', 0) else None

    def forward(self, x, context_state, route_weights=None, context_gates=None, p24_shared=None):
        """route_weights:  optional dict from SharedBlockRouter with 'c_fc'/'c_proj_ffn' keys.
        context_gates:  optional dict from SharedContextGates with layer-key sub-dicts."""
        rw_fc   = route_weights['c_fc']      if (route_weights and isinstance(self.c_fc,   (RemixedLinear, RemixedLinearFused, OutputGatedLinear))) else None
        rw_proj = route_weights['c_proj_ffn'] if (route_weights and isinstance(self.c_proj, (RemixedLinear, RemixedLinearFused, OutputGatedLinear))) else None
        cg_fc   = context_gates['c_fc']      if (context_gates and isinstance(self.c_fc,   (RemixedLinear, RemixedLinearFused, OutputGatedLinear))) else None
        cg_proj = context_gates['c_proj_ffn'] if (context_gates and isinstance(self.c_proj, (RemixedLinear, RemixedLinearFused, OutputGatedLinear))) else None
        route_in = p24_shared.get('route_input') if p24_shared is not None else None
        gate_in = p24_shared.get('gate_input') if p24_shared is not None else None
        x = self.c_fc(x, context_state, route_weights=rw_fc, context_gates=cg_fc, p24_route_input=route_in, p24_gate_input=gate_in)
        if getattr(self, '_single', False):
            # One RemixedLinear is the whole FFN: no hidden activation to apply and
            # no second projection. Returning here rather than passing through a
            # no-op keeps the graph identical to a single linear layer.
            return self.channel_scale(x) if self.channel_scale is not None else x
        if self.dynamic_act is not None:
            x = self.dynamic_act(x)
        else:
            x = F.relu(x).square()
        x = self.c_proj(x, context_state, route_weights=rw_proj, context_gates=cg_proj, p24_route_input=route_in, p24_gate_input=gate_in)
        if self.channel_scale is not None:
            x = self.channel_scale(x)
        return x


class RemixedMultiAttention(nn.Module):
    """Attention path using RemixedLinear (or RAL) for Q/K/V/proj."""
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        self.ve_gate_channels = min(self.n_embd, 32)
        self.shadow_dim = max(0, int(getattr(config, 'cclblock_attn_shadow_dim', 0)))
        self.shadow_dim_per_head = 0
        if self.shadow_dim > 0:
            raw_per_head = (self.shadow_dim + self.n_kv_head - 1) // self.n_kv_head
            if raw_per_head % 8 != 0:
                raw_per_head = ((raw_per_head + 7) // 8) * 8
            corrected = raw_per_head * self.n_kv_head
            if corrected != self.shadow_dim:
                print0(f"cclblock_attn_shadow_dim auto-corrected from {self.shadow_dim} to {corrected} to ensure FlashAttention V-head multiple of 8")
                self.shadow_dim = corrected
            self.shadow_dim_per_head = self.shadow_dim // self.n_kv_head
        self.v_head_dim = self.head_dim + self.shadow_dim_per_head
        use_ral = getattr(config, 'use_ral', False)
        mode = getattr(config, 'cclblock_modulation', 'weight')
        use_decoupled = mode == 'decoupled'
        ral_rank = getattr(config, 'ral_rank', 32)
        film_gate = getattr(config, 'cclblock_film_gate', False)
        kwargs = config.remixed_linear_kwargs if config.remixed_linear_kwargs is not None else dict(use_basis_gate=True, use_output_gate=True, use_context=True)
        scale = getattr(config, 'scale_basis_size', True)
        if mode == 'fsi':
            n_rot = getattr(config, 'cclblock_fsi_rotations', 8)
            sel_dim = getattr(config, 'cclblock_fsi_selector_dim', 64)
            self.c_q    = FrozenSubspaceIndexedLinear(self.n_embd, self.n_head * self.head_dim, n_rotations=n_rot, selector_dim=sel_dim, signal_dim=self.n_embd)
            self.c_k    = FrozenSubspaceIndexedLinear(self.n_embd, self.n_kv_head * self.head_dim, n_rotations=n_rot, selector_dim=sel_dim, signal_dim=self.n_embd)
            self.c_v    = FrozenSubspaceIndexedLinear(self.n_embd, self.n_kv_head * self.v_head_dim, n_rotations=n_rot, selector_dim=sel_dim, signal_dim=self.n_embd)
            self.c_proj = FrozenSubspaceIndexedLinear(self.n_embd, self.n_embd, n_rotations=n_rot, selector_dim=sel_dim, signal_dim=self.n_embd)
        elif mode == 'aesp':
            n_str = getattr(config, 'cclblock_aesp_strata', 4)
            d_rank = getattr(config, 'cclblock_aesp_delta_rank', 4)
            self.c_q    = AttentionEntropyStratifiedLinear(self.n_embd, self.n_head * self.head_dim, n_strata=n_str, delta_rank=d_rank)
            self.c_k    = AttentionEntropyStratifiedLinear(self.n_embd, self.n_kv_head * self.head_dim, n_strata=n_str, delta_rank=d_rank)
            self.c_v    = AttentionEntropyStratifiedLinear(self.n_embd, self.n_kv_head * self.v_head_dim, n_strata=n_str, delta_rank=d_rank)
            self.c_proj = AttentionEntropyStratifiedLinear(self.n_embd, self.n_embd, n_strata=n_str, delta_rank=d_rank)
        elif mode == 'cond' and not _cond_site_enabled(config, 'attn'):
            self.c_q    = _DenseProj(self.n_embd, self.n_head * self.head_dim)
            self.c_k    = _DenseProj(self.n_embd, self.n_kv_head * self.head_dim)
            self.c_v    = _DenseProj(self.n_embd, self.n_kv_head * self.v_head_dim)
            self.c_proj = _DenseProj(self.n_embd, self.n_embd)
        elif mode == 'cond' and _cond_site_enabled(config, 'attn'):
            ck = _conditioned_linear_kwargs(config)
            if ck['gate_source'] == 'ctx':
                # Attention runs before the block has a context: the bypass path
                # derives ctx from attn_out, so context_state is None here by
                # construction. 'ctx' therefore means "condition the FFN on what
                # attention just retrieved", and attention conditions on its own
                # input, since nothing else is available at that point.
                ck = dict(ck, gate_source='router', signal_dim=None)
            # Per-projection conditioning: cond_attn_projs controls which of
            # q/k/v/o get ConditionedLinear vs plain dense.  Default 'qkvo' = all.
            _projs = str(getattr(config, 'cond_attn_projs', 'qkvo')).lower()
            # Per-layer conditioning: cond_layer_frac restricts conditioning to
            # the first N% of layers (where unmet demand is highest). Default 1.0 = all.
            _frac = float(getattr(config, 'cond_layer_frac', 1.0))
            _layer_ok = (layer_idx / max(1, config.n_layer)) < _frac
            def _mk(in_f, out_f, proj_char, zero_init):
                if proj_char in _projs and _layer_ok:
                    return ConditionedLinear(in_f, out_f, zero_init_base=zero_init, **ck)
                return _DenseProj(in_f, out_f)
            self.c_q    = _mk(self.n_embd, self.n_head * self.head_dim,      'q', False)
            self.c_k    = _mk(self.n_embd, self.n_kv_head * self.head_dim,   'k', False)
            self.c_v    = _mk(self.n_embd, self.n_kv_head * self.v_head_dim, 'v', False)
            self.c_proj = _mk(self.n_embd, self.n_embd,                      'o', True)
        elif mode == 'ckr':
            n_br = getattr(config, 'cclblock_ckr_branches', 4)
            ksz = getattr(config, 'cclblock_ckr_kernel_size', 64)
            n_pc = getattr(config, 'cclblock_ckr_pos_channels', 1)
            cb_scale = getattr(config, 'cclblock_ckr_content_bias', 0.0)
            temp = getattr(config, 'cclblock_ckr_temp_start', 1.0)
            ortho = bool(getattr(config, 'cclblock_ckr_ortho_init', 0))
            bdrop = float(getattr(config, 'cclblock_ckr_branch_dropout', 0.0))
            ckr_common = dict(n_branches=n_br, kernel_size=ksz, max_seq_len=config.sequence_len, n_pos_channels=n_pc, content_bias_scale=cb_scale, signal_dim=self.n_embd, temperature=temp, ortho_init=ortho, branch_dropout=bdrop)
            self.c_q    = CausalKernelLinear(self.n_embd, self.n_head * self.head_dim, **ckr_common)
            self.c_k    = CausalKernelLinear(self.n_embd, self.n_kv_head * self.head_dim, **ckr_common)
            self.c_v    = CausalKernelLinear(self.n_embd, self.n_kv_head * self.v_head_dim, **ckr_common)
            self.c_proj = CausalKernelLinear(self.n_embd, self.n_embd, **ckr_common)
        elif mode == 'ckr_ffn':
            # 16B: Plain Linear for attention (CKR only on FFN)
            self.c_q    = Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
            self.c_k    = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
            self.c_v    = Linear(self.n_embd, self.n_kv_head * self.v_head_dim, bias=False)
            self.c_proj = Linear(self.n_embd, self.n_embd, bias=False)
        elif mode == 'pgr':
            pgr_ks = getattr(config, 'cclblock_pgr_kernel_size', 64)
            self.c_q    = PositionGatedResidual(self.n_embd, self.n_head * self.head_dim, kernel_size=pgr_ks, max_seq_len=config.sequence_len)
            self.c_k    = PositionGatedResidual(self.n_embd, self.n_kv_head * self.head_dim, kernel_size=pgr_ks, max_seq_len=config.sequence_len)
            self.c_v    = PositionGatedResidual(self.n_embd, self.n_kv_head * self.v_head_dim, kernel_size=pgr_ks, max_seq_len=config.sequence_len)
            self.c_proj = PositionGatedResidual(self.n_embd, self.n_embd, kernel_size=pgr_ks, max_seq_len=config.sequence_len)
        elif mode == 'cil':
            cil_ks = getattr(config, 'cclblock_cil_kernel_size', 64)
            self.c_q    = CausalInterpolationLinear(self.n_embd, self.n_head * self.head_dim, kernel_size=cil_ks, max_seq_len=config.sequence_len)
            self.c_k    = CausalInterpolationLinear(self.n_embd, self.n_kv_head * self.head_dim, kernel_size=cil_ks, max_seq_len=config.sequence_len)
            self.c_v    = CausalInterpolationLinear(self.n_embd, self.n_kv_head * self.v_head_dim, kernel_size=cil_ks, max_seq_len=config.sequence_len)
            self.c_proj = CausalInterpolationLinear(self.n_embd, self.n_embd, kernel_size=cil_ks, max_seq_len=config.sequence_len)
        elif mode == 'prb':
            prb_ks = getattr(config, 'cclblock_prb_kernel_size', 64)
            self.c_q    = PositionalResidualBias(self.n_embd, self.n_head * self.head_dim, kernel_size=prb_ks, max_seq_len=config.sequence_len)
            self.c_k    = PositionalResidualBias(self.n_embd, self.n_kv_head * self.head_dim, kernel_size=prb_ks, max_seq_len=config.sequence_len)
            self.c_v    = PositionalResidualBias(self.n_embd, self.n_kv_head * self.v_head_dim, kernel_size=prb_ks, max_seq_len=config.sequence_len)
            self.c_proj = PositionalResidualBias(self.n_embd, self.n_embd, kernel_size=prb_ks, max_seq_len=config.sequence_len)
        elif mode == 'com':
            com_ks = getattr(config, 'cclblock_com_kernel_size', 32)
            self.c_q    = CausalOutputMixer(self.n_embd, self.n_head * self.head_dim, kernel_size=com_ks, max_seq_len=config.sequence_len)
            self.c_k    = CausalOutputMixer(self.n_embd, self.n_kv_head * self.head_dim, kernel_size=com_ks, max_seq_len=config.sequence_len)
            self.c_v    = CausalOutputMixer(self.n_embd, self.n_kv_head * self.v_head_dim, kernel_size=com_ks, max_seq_len=config.sequence_len)
            self.c_proj = CausalOutputMixer(self.n_embd, self.n_embd, kernel_size=com_ks, max_seq_len=config.sequence_len)
        elif mode == 'giad':
            giad_rank = getattr(config, 'cclblock_giad_rank', 32)
            self.c_q    = GradientIsolatedDeltaLinear(self.n_embd, self.n_head * self.head_dim, delta_rank=giad_rank)
            self.c_k    = GradientIsolatedDeltaLinear(self.n_embd, self.n_kv_head * self.head_dim, delta_rank=giad_rank)
            self.c_v    = GradientIsolatedDeltaLinear(self.n_embd, self.n_kv_head * self.v_head_dim, delta_rank=giad_rank)
            self.c_proj = GradientIsolatedDeltaLinear(self.n_embd, self.n_embd, delta_rank=giad_rank)
        elif mode == 'psg':
            psg_ks = getattr(config, 'cclblock_psg_kernel_size', 64)
            self.c_q    = PositionalScalarGatedLinear(self.n_embd, self.n_head * self.head_dim, kernel_size=psg_ks, max_seq_len=config.sequence_len)
            self.c_k    = PositionalScalarGatedLinear(self.n_embd, self.n_kv_head * self.head_dim, kernel_size=psg_ks, max_seq_len=config.sequence_len)
            self.c_v    = PositionalScalarGatedLinear(self.n_embd, self.n_kv_head * self.v_head_dim, kernel_size=psg_ks, max_seq_len=config.sequence_len)
            self.c_proj = PositionalScalarGatedLinear(self.n_embd, self.n_embd, kernel_size=psg_ks, max_seq_len=config.sequence_len)
        elif mode == 'splitstream':
            ss_ratio = float(getattr(config, 'cclblock_ss_dynamic_ratio', 0.25))
            ss_br = getattr(config, 'cclblock_ss_branches', 2)
            ss_ks = getattr(config, 'cclblock_ss_kernel_size', 64)
            self.c_q    = SplitStreamLinear(self.n_embd, self.n_head * self.head_dim, dynamic_ratio=ss_ratio, n_branches=ss_br, kernel_size=ss_ks, max_seq_len=config.sequence_len)
            self.c_k    = SplitStreamLinear(self.n_embd, self.n_kv_head * self.head_dim, dynamic_ratio=ss_ratio, n_branches=ss_br, kernel_size=ss_ks, max_seq_len=config.sequence_len)
            self.c_v    = SplitStreamLinear(self.n_embd, self.n_kv_head * self.v_head_dim, dynamic_ratio=ss_ratio, n_branches=ss_br, kernel_size=ss_ks, max_seq_len=config.sequence_len)
            self.c_proj = SplitStreamLinear(self.n_embd, self.n_embd, dynamic_ratio=ss_ratio, n_branches=ss_br, kernel_size=ss_ks, max_seq_len=config.sequence_len)
        elif mode == 'lokr':
            lokr_br = getattr(config, 'cclblock_lokr_branches', 8)
            lokr_r = getattr(config, 'cclblock_lokr_rank', 16)
            lokr_ks = getattr(config, 'cclblock_ckr_kernel_size', 64)
            self.c_q    = LoKRLinear(self.n_embd, self.n_head * self.head_dim, n_branches=lokr_br, rank=lokr_r, kernel_size=lokr_ks, max_seq_len=config.sequence_len)
            self.c_k    = LoKRLinear(self.n_embd, self.n_kv_head * self.head_dim, n_branches=lokr_br, rank=lokr_r, kernel_size=lokr_ks, max_seq_len=config.sequence_len)
            self.c_v    = LoKRLinear(self.n_embd, self.n_kv_head * self.v_head_dim, n_branches=lokr_br, rank=lokr_r, kernel_size=lokr_ks, max_seq_len=config.sequence_len)
            self.c_proj = LoKRLinear(self.n_embd, self.n_embd, n_branches=lokr_br, rank=lokr_r, kernel_size=lokr_ks, max_seq_len=config.sequence_len)
        elif mode == 'tucker':
            self.c_q = TuckerAdaptiveLinear(self.n_embd, self.n_head * self.head_dim, context_dim=config.remix_context_dim, tucker_rank=getattr(config, 'cclblock_tucker_rank', 32), tucker_modes=getattr(config, 'cclblock_tucker_modes', 8))
            self.c_k = TuckerAdaptiveLinear(self.n_embd, self.n_kv_head * self.head_dim, context_dim=config.remix_context_dim, tucker_rank=getattr(config, 'cclblock_tucker_rank', 32), tucker_modes=getattr(config, 'cclblock_tucker_modes', 8))
            self.c_v = TuckerAdaptiveLinear(self.n_embd, self.n_kv_head * self.v_head_dim, context_dim=config.remix_context_dim, tucker_rank=getattr(config, 'cclblock_tucker_rank', 32), tucker_modes=getattr(config, 'cclblock_tucker_modes', 8))
            self.c_proj = TuckerAdaptiveLinear(self.n_embd, self.n_embd, context_dim=config.remix_context_dim, tucker_rank=getattr(config, 'cclblock_tucker_rank', 32), tucker_modes=getattr(config, 'cclblock_tucker_modes', 8))
        elif mode in ('svs', 'dcu'):
            dcu_steps = getattr(config, 'cclblock_dcu_warmup_steps', 0) if mode == 'dcu' else 0
            common = dict(context_dim=config.remix_context_dim, svs_rank=getattr(config, 'cclblock_svs_rank', 64), svs_eps=getattr(config, 'cclblock_svs_eps', 0.1), dcu_warmup_steps=dcu_steps)
            self.c_q = SingularValueSteeringLinear(self.n_embd, self.n_head * self.head_dim, **common)
            self.c_k = SingularValueSteeringLinear(self.n_embd, self.n_kv_head * self.head_dim, **common)
            self.c_v = SingularValueSteeringLinear(self.n_embd, self.n_kv_head * self.v_head_dim, **common)
            self.c_proj = SingularValueSteeringLinear(self.n_embd, self.n_embd, **common)
        elif mode == 'vq':
            common = dict(context_dim=config.remix_context_dim, vq_codes=getattr(config, 'cclblock_vq_codes', 8), vq_temperature=getattr(config, 'cclblock_vq_temperature', 1.0))
            self.c_q = VQAdaptiveLinear(self.n_embd, self.n_head * self.head_dim, **common)
            self.c_k = VQAdaptiveLinear(self.n_embd, self.n_kv_head * self.head_dim, **common)
            self.c_v = VQAdaptiveLinear(self.n_embd, self.n_kv_head * self.v_head_dim, **common)
            self.c_proj = VQAdaptiveLinear(self.n_embd, self.n_embd, **common)
        elif use_decoupled:
            ratio = float(getattr(config, 'cclblock_dynamic_ratio', 0.25))
            gate_rank = int(getattr(config, 'cclblock_gate_rank', 8))
            self.c_q    = DecoupledAdaptiveLinear(self.n_embd, self.n_head * self.head_dim, context_dim=config.remix_context_dim, dynamic_ratio=ratio, basis_size=config.remix_basis_size, gate_rank=gate_rank)
            self.c_k    = DecoupledAdaptiveLinear(self.n_embd, self.n_kv_head * self.head_dim, context_dim=config.remix_context_dim, dynamic_ratio=ratio, basis_size=config.remix_basis_size, gate_rank=gate_rank)
            self.c_v    = DecoupledAdaptiveLinear(self.n_embd, self.n_kv_head * self.v_head_dim, context_dim=config.remix_context_dim, dynamic_ratio=ratio, basis_size=config.remix_basis_size, gate_rank=gate_rank)
            self.c_proj = DecoupledAdaptiveLinear(self.n_embd, self.n_embd, context_dim=config.remix_context_dim, dynamic_ratio=ratio, basis_size=config.remix_basis_size, gate_rank=gate_rank)
        elif use_ral:
            self.c_q    = ResidualAdaptiveLinear(self.n_embd, self.n_head * self.head_dim,    context_dim=config.remix_context_dim, rank=ral_rank)
            self.c_k    = ResidualAdaptiveLinear(self.n_embd, self.n_kv_head * self.head_dim, context_dim=config.remix_context_dim, rank=ral_rank)
            self.c_v    = ResidualAdaptiveLinear(self.n_embd, self.n_kv_head * self.v_head_dim, context_dim=config.remix_context_dim, rank=ral_rank)
            self.c_proj = ResidualAdaptiveLinear(self.n_embd, self.n_embd,                    context_dim=config.remix_context_dim, rank=ral_rank)
        else:
            # Enrich kwargs with Phase 23 LoKR / Tiny Expert settings
            lokr_expert = bool(getattr(config, 'p23_lokr', 0))
            tiny_expert = bool(getattr(config, 'p23_tiny_expert', 0))
            if lokr_expert or tiny_expert or True:  # always copy to add p28 fields safely
                kwargs = dict(kwargs)  # shallow copy
                kwargs['use_quantile_route'] = int(getattr(config, 'p23_quantile_route', 0))
                kwargs['template_topk'] = int(getattr(config, 'p22_template_topk', kwargs.get('template_topk', 0)))
                kwargs['route_affine'] = int(getattr(config, 'p22_route_affine', 0))
            # Phase 28: tag with layer_role so global bank knows not to replace attn (mode='ffn')
            kwargs['layer_role'] = 'attn'
            kwargs['global_bank_mode'] = getattr(config, 'p28_global_template_bank', 'none')
            kwargs['chunk_routing_size'] = getattr(config, 'p28_chunk_routing_size', 0)
            if lokr_expert:
                kwargs['lokr_expert'] = True
                kwargs['lokr_n_experts'] = getattr(config, 'p23_n_experts', 64)
                kwargs['lokr_topk'] = getattr(config, 'p23_topk', 16)
                kwargs['lokr_rank'] = getattr(config, 'p23_lokr_rank', 4)
                kwargs['lokr_learned'] = bool(getattr(config, 'p23_learned_route', 0))
            elif tiny_expert:
                kwargs['tiny_expert'] = True
                kwargs['tiny_expert_topk'] = getattr(config, 'p23_topk', 16)
            # Phase 28 per-module template count overrides (from similarity analysis)
            # Build separate kwarg dicts so c_q/c_k, c_v, and c_proj can each have different K
            attn_proj_K = int(getattr(config, 'p28_attn_proj_templates', 0))
            attn_qk_K   = int(getattr(config, 'p28_attn_qk_templates', 0))
            kwargs_proj = dict(kwargs)
            kwargs_qk   = dict(kwargs)
            if attn_proj_K > 0:
                kwargs_proj['n_templates'] = attn_proj_K
            if attn_qk_K > 0:
                kwargs_qk['n_templates'] = attn_qk_K
            # Attention layers use per-sequence routing (one routing decision per sequence)
            # Auto-select RemixedLinearFused for 29C chunk routing only if explicitly enabled via remix_use_fused
            _is_29c_fused = (
                getattr(config, 'remix_use_fused', False)
                # RemixedLinearFused has its own inline routing and does not
                # implement the affine path, so the flag would silently no-op.
                and not int(getattr(config, 'p22_route_affine', 0))
                and kwargs.get('n_templates', 1) > 1
                and int(kwargs.get('chunk_routing_size', 0)) > 0
                and not lokr_expert and not tiny_expert
                and kwargs.get('global_bank_mode', 'none') == 'none'
            )
            _linear_cls = (
                DualGateLinear if getattr(config, 'remix_use_dual_gate', False)
                else OutputGatedLinear if getattr(config, 'p26_output_gated_linear', 0)
                else RemixedLinearFused if _is_29c_fused
                else RemixedLinear
            )
            self.c_q    = _linear_cls(self.n_embd, self.n_head * self.head_dim,      context_dim=config.remix_context_dim, basis_size=config.remix_basis_size, remixed_linear_kwargs=kwargs_qk,   scale_basis=scale, film_gate=film_gate, routing_scope='per_sequence')
            self.c_k    = _linear_cls(self.n_embd, self.n_kv_head * self.head_dim,   context_dim=config.remix_context_dim, basis_size=config.remix_basis_size, remixed_linear_kwargs=kwargs_qk,   scale_basis=scale, film_gate=film_gate, routing_scope='per_sequence')
            self.c_v    = _linear_cls(self.n_embd, self.n_kv_head * self.v_head_dim, context_dim=config.remix_context_dim, basis_size=config.remix_basis_size, remixed_linear_kwargs=kwargs,      scale_basis=scale, film_gate=film_gate, routing_scope='per_sequence')
            self.c_proj = _linear_cls(self.n_embd, self.n_embd,                      context_dim=config.remix_context_dim, basis_size=config.remix_basis_size, remixed_linear_kwargs=kwargs_proj, scale_basis=scale, film_gate=film_gate, routing_scope='per_sequence')

        self.ve_gate = Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None
        self._attn_mode = mode  # store for forward dispatch

    def forward(self, x, ve, cos_sin, window_size, kv_cache, context_state, route_weights=None):
        """route_weights: optional dict from SharedBlockRouter with 'c_q','c_k','c_v','c_proj_attn'."""
        is_rl = isinstance(self.c_q, (RemixedLinear, RemixedLinearFused, OutputGatedLinear))  # True for weight-mod path
        rw = route_weights if (route_weights is not None and is_rl) else None
        B, T, C = x.size()
        if self._attn_mode == 'ckr_ffn':
            # Plain Linear for attention — no context_state
            q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
            k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
            v_full = self.c_v(x).view(B, T, self.n_kv_head, self.v_head_dim)
        else:
            q = self.c_q(x, context_state, route_weights=rw['c_q']      if rw else None).view(B, T, self.n_head, self.head_dim)
            k = self.c_k(x, context_state, route_weights=rw['c_k']      if rw else None).view(B, T, self.n_kv_head, self.head_dim)
            v_full = self.c_v(x, context_state, route_weights=rw['c_v'] if rw else None).view(B, T, self.n_kv_head, self.v_head_dim)
        v = v_full[..., :self.head_dim]

        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            if self.ve_gate is not None:
                gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
                v = v + gate.unsqueeze(-1) * ve

        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)

        if kv_cache is None:
            y = flash_attn.flash_attn_func(q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16), causal=True, window_size=window_size)
        else:
            k_cache, v_cache = kv_cache.get_layer_cache(self.layer_idx)
            y = flash_attn.flash_attn_with_kvcache(
                q.to(torch.bfloat16), k_cache, v_cache,
                k=k.to(torch.bfloat16), v=v.to(torch.bfloat16),
                cache_seqlens=kv_cache.cache_seqlens,
                causal=True,
                window_size=window_size,
            )
            if self.layer_idx == kv_cache.n_layers - 1:
                kv_cache.advance(T)

        y = y.contiguous().view(B, T, -1)
        if self._attn_mode == 'ckr_ffn':
            return self.c_proj(y)
        return self.c_proj(y, context_state, route_weights=rw['c_proj_attn'] if rw else None)

    def forward_with_shadow(self, x, ve, cos_sin, window_size, kv_cache, context_state):
        """Dual-V routing: return content output plus shadow stream routed by same attention map."""
        B, T, C = x.size()
        q = self.c_q(x, context_state).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x, context_state).view(B, T, self.n_kv_head, self.head_dim)
        v_full = self.c_v(x, context_state).view(B, T, self.n_kv_head, self.v_head_dim)
        v = v_full[..., :self.head_dim]

        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            if self.ve_gate is not None:
                gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
                v = v + gate.unsqueeze(-1) * ve
        if self.shadow_dim_per_head > 0:
            v_full = torch.cat([v, v_full[..., self.head_dim:]], dim=-1)
        else:
            v_full = v

        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)

        if kv_cache is None:
            y_full = flash_attn.flash_attn_func(q.to(torch.bfloat16), k.to(torch.bfloat16), v_full.to(torch.bfloat16), causal=True, window_size=window_size)
        else:
            k_cache, v_cache = kv_cache.get_layer_cache(self.layer_idx)
            y_full = flash_attn.flash_attn_with_kvcache(
                q.to(torch.bfloat16), k_cache, v_cache,
                k=k.to(torch.bfloat16), v=v_full.to(torch.bfloat16),
                cache_seqlens=kv_cache.cache_seqlens,
                causal=True,
                window_size=window_size,
            )
            if self.layer_idx == kv_cache.n_layers - 1:
                kv_cache.advance(T)

        y_content = y_full[..., :self.head_dim].contiguous().view(B, T, -1)
        y_proj = self.c_proj(y_content, context_state)
        if self.shadow_dim_per_head > 0:
            y_shadow = y_full[..., self.head_dim:].contiguous().view(B, T, -1)
        else:
            y_shadow = torch.zeros(B, T, 0, device=x.device, dtype=x.dtype)
        return y_proj, y_shadow

    def forward_with_q_stats(self, x, ve, cos_sin, window_size, kv_cache, context_state):
        """Design 2: run forward, also returning mean-query-vector as context signal.

        q after RoPE+QKnorm is the model's per-position 'search intent'. Its mean across
        attention heads gives a (B, T, head_dim) tensor capturing the query distribution
        without any temporal recurrence. Detached so it contributes no gradient to Q/K/V.
        """
        B, T, C = x.size()
        q = self.c_q(x, context_state).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x, context_state).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x, context_state).view(B, T, self.n_kv_head, self.head_dim)

        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            if self.ve_gate is not None:
                gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
                v = v + gate.unsqueeze(-1) * ve

        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)

        # Capture query statistics BEFORE flash_attn (q is post-RoPE+QKnorm)
        # Mean across heads: (B, T, head_dim) — the average 'search direction'
        q_stats = q.mean(dim=2).detach()  # detach: no gradient through ctx into Q/K weights

        if kv_cache is None:
            y = flash_attn.flash_attn_func(q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16), causal=True, window_size=window_size)
        else:
            k_cache, v_cache = kv_cache.get_layer_cache(self.layer_idx)
            y = flash_attn.flash_attn_with_kvcache(
                q.to(torch.bfloat16), k_cache, v_cache,
                k=k.to(torch.bfloat16), v=v.to(torch.bfloat16),
                cache_seqlens=kv_cache.cache_seqlens,
                causal=True,
                window_size=window_size,
            )
            if self.layer_idx == kv_cache.n_layers - 1:
                kv_cache.advance(T)

        y = y.contiguous().view(B, T, -1)
        return self.c_proj(y, context_state), q_stats


class GlobalTemplateBank(nn.Module):
    """Phase 28E: Cross-layer global template bank with per-layer tiny routers.

    Instead of each RemixedLinear owning K templates (params scale with layers*K*D*B),
    this module maintains a single shared set of K templates, and each layer selects
    from them via a tiny Linear(n_embd, K) router.

    Two sub-banks handle different output dimensions:
      - fc_bank:   (K, 4*n_embd, basis_size)  — for FFN c_fc layers
      - proj_bank: (K, n_embd,   basis_size)  — for FFN c_proj / attn Q,K,V,O

    FLOPs-per-token: router(n_embd -> K) amortized, then single bmm(B, T, D).
    Parameter savings: ~(n_layers-1)/(n_layers) of per-layer template params.
    """
    def __init__(self, n_templates, n_embd, basis_size, n_layers):
        super().__init__()
        self.n_templates = n_templates
        self.n_embd = n_embd
        self.basis_size = basis_size
        self.n_layers = n_layers
        # Two banks for the two output-dimension shapes present in RemixedLinear
        self.fc_bank   = nn.Parameter(torch.empty(n_templates, 4 * n_embd, basis_size))
        self.proj_bank = nn.Parameter(torch.empty(n_templates, n_embd, basis_size))
        # Per-layer tiny routers: input pooled mean of residual stream -> template weights
        self.layer_routers_fc       = nn.ModuleList([nn.Linear(n_embd, n_templates, bias=False) for _ in range(n_layers)])
        self.layer_routers_ffn_proj = nn.ModuleList([nn.Linear(4 * n_embd, n_templates, bias=False) for _ in range(n_layers)])
        self.layer_routers_proj     = nn.ModuleList([nn.Linear(n_embd, n_templates, bias=False) for _ in range(n_layers)])

    def get_effective_weight(self, bank_key, layer_idx, x):
        """Compute the blended effective weight matrix for this layer's forward pass.

        Args:
            bank_key:  'fc' for c_fc layers (out=4*n_embd), 'proj' for c_proj / attn layers
            layer_idx: which layer's router to use
            x:         (B, T, n_embd) residual stream input

        Returns:
            W_eff: (B, out_features, basis_size) — batch-specific effective weight matrix
        """
        ctx = x.float().mean(dim=1)  # (B, n_embd) — pool across sequence for per-sequence routing
        if bank_key == 'fc':
            logits = self.layer_routers_fc[layer_idx](ctx)     # (B, K)
            bank   = self.fc_bank                               # (K, 4*n_embd, basis_size)
        else:
            if ctx.shape[-1] == self.n_embd * 4:
                logits = self.layer_routers_ffn_proj[layer_idx](ctx) # (B, K)
            else:
                logits = self.layer_routers_proj[layer_idx](ctx)   # (B, K)
            bank   = self.proj_bank                             # (K, n_embd, basis_size)
        weights = F.softmax(logits, dim=-1).to(bank.dtype)     # (B, K)
        # Weighted sum over templates: (B, out_features, basis_size)
        W_eff = torch.einsum('bk,koc->boc', weights, bank)
        return W_eff


class SharedContextGates(nn.Module):
    """Batches all 6 per-RemixedLinear gate computations into 3 shared matmuls per block.

    Without this: each of the 6 RemixedLinear modules independently calls:
      - basis_modulator(ctx): 2 matmuls (Linear+GELU+Linear) = 12 per block
      - output_gate_coeffs(ctx): 1 matmul = 6 per block  → 18 total.

    With SharedContextGates:
      - hidden_proj(ctx): 1 matmul + GELU
      - basis_gate_proj(hidden): 1 matmul → all 6 basis gates at once
      - output_coeff_proj(ctx): 1 matmul → all 6 output gate coefficients at once
      → 3 total. ~6× fewer kernel launches for the gating overhead.

    Per-layer personalisation is preserved:
      - basis gates share the MLP but each RL applies sigmoid with its own gate_temperature
      - output gates share the ctx→coeff projection but each RL multiplies by its own
        output_gate_basis (r × out_features), whose shape differs across layers.
    """
    # SCG only serves the 2 FFN layers — attention layers always use their local basis_modulator
    # because block_ctx_gates is only passed to ffwd(), not attn().
    N_LAYERS  = 2
    LAYER_KEYS = ('c_fc', 'c_proj_ffn')

    def __init__(self, ctx_dim: int, basis_size: int, gate_rank: int = 8,
                 film_gate: bool = False):
        super().__init__()
        self.basis_size = basis_size
        self.gate_rank  = gate_rank
        self.film_gate  = film_gate
        N = self.N_LAYERS
        gate_out_size = 2 * basis_size if film_gate else basis_size

        # Shared hidden for basis gate MLP (serves 2 FFN layers)
        hidden = max(ctx_dim // 2, min(basis_size, ctx_dim * 2))
        self.hidden_proj     = nn.Linear(ctx_dim, hidden, bias=True)
        self.basis_gate_proj = nn.Linear(hidden, N * gate_out_size, bias=True)
        # Output gate coefficients (ctx → N*r, no hidden needed — small projection)
        self.output_coeff_proj = nn.Linear(ctx_dim, N * gate_rank, bias=True)

        # Zero-init so gates start as identity (sigmoid(0)=0.5, matching local basis_modulator init)
        nn.init.zeros_(self.hidden_proj.weight);     nn.init.zeros_(self.hidden_proj.bias)
        nn.init.zeros_(self.basis_gate_proj.weight); nn.init.zeros_(self.basis_gate_proj.bias)
        nn.init.zeros_(self.output_coeff_proj.weight); nn.init.zeros_(self.output_coeff_proj.bias)

    def forward(self, context_state: 'Tensor') -> dict:
        """context_state: (B, T, ctx_dim)
        Returns dict: layer_key → {'basis_gate': (B,T,basis_size), 'output_coeffs': (B,T,r)}
        Only keys 'c_fc' and 'c_proj_ffn' are emitted (attn layers use local basis_modulator).
        """
        dtype = context_state.dtype
        N, K, r = self.N_LAYERS, self.basis_size, self.gate_rank
        gate_out = 2 * K if self.film_gate else K

        # 3 matmuls total for both FFN gates — explicit dtype cast avoids BFloat16/float32 mismatch
        # under FP8 training where SharedContextGates is not converted by the FP8 converter.
        hp_w = self.hidden_proj.weight.to(dtype)
        hp_b = self.hidden_proj.bias.to(dtype) if self.hidden_proj.bias is not None else None
        hidden  = F.gelu(F.linear(context_state, hp_w, hp_b))         # (B, T, hidden)
        bg_w = self.basis_gate_proj.weight.to(dtype)
        bg_b = self.basis_gate_proj.bias.to(dtype) if self.basis_gate_proj.bias is not None else None
        bg_all  = F.linear(hidden, bg_w, bg_b)                         # (B, T, N * gate_out)
        oc_w = self.output_coeff_proj.weight.to(dtype)
        oc_b = self.output_coeff_proj.bias.to(dtype) if self.output_coeff_proj.bias is not None else None
        oc_all  = F.linear(context_state, oc_w, oc_b)                 # (B, T, N * r)

        bg_all  = bg_all.view(*bg_all.shape[:-1], N, gate_out)        # (B, T, N, gate_out)
        oc_all  = oc_all.view(*oc_all.shape[:-1], N, r)               # (B, T, N, r)

        return {
            key: {
                'basis_gate':    bg_all[..., i, :].to(dtype),         # (B, T, gate_out)
                'output_coeffs': oc_all[..., i, :].to(dtype),         # (B, T, r)
            }
            for i, key in enumerate(self.LAYER_KEYS)
        }

    def gate_parameters(self):
        """All SharedContextGates params go to the gate LR group."""
        yield from self.hidden_proj.parameters()
        yield from self.basis_gate_proj.parameters()
        yield from self.output_coeff_proj.parameters()


class SharedBlockRouter(nn.Module):
    """Pre-computes routing weights for ALL RemixedLinear layers in a block in two matmuls.

    Instead of each of the 6 RemixedLinear layers independently computing K routing logits
    (6 separate matmuls), this module batches all logits into two combined matmuls at block entry:

      - Token-level  (c_fc, c_proj_ffn)            : (B, T, K×2) in one shot
      - Sequence-level (c_q, c_k, c_v, c_proj_attn): (B, 1, K×4) then expand to (B, T, K×4)

    Each RemixedLinear receives its corresponding slice via the `route_weights` kwarg
    and skips local routing entirely.  This cuts routing kernel launches from 6 to 2 per block.
    """
    N_TOKEN_LAYERS = 2   # c_fc, c_proj_ffn
    N_SEQ_LAYERS   = 4   # c_q, c_k, c_v, c_proj_attn
    TOKEN_KEYS     = ('c_fc', 'c_proj_ffn')
    SEQ_KEYS       = ('c_q', 'c_k', 'c_v', 'c_proj_attn')

    def __init__(self, n_embd: int, n_experts: int, learned: bool = True):
        super().__init__()
        self.n_experts = n_experts
        K = n_experts
        if learned:
            self.token_proj = nn.Parameter(torch.empty(n_embd, K * self.N_TOKEN_LAYERS))
            self.seq_proj   = nn.Parameter(torch.empty(n_embd, K * self.N_SEQ_LAYERS))
            nn.init.normal_(self.token_proj, std=n_embd ** -0.5)
            nn.init.normal_(self.seq_proj,   std=n_embd ** -0.5)
        else:
            self.register_buffer('token_proj',
                torch.randn(n_embd, K * self.N_TOKEN_LAYERS) / (n_embd ** 0.5))
            self.register_buffer('seq_proj',
                torch.randn(n_embd, K * self.N_SEQ_LAYERS)   / (n_embd ** 0.5))

    def forward(self, x: 'Tensor') -> dict:
        """x: (B, T, D) — normalized residual stream.
        Returns dict: layer_name → (B, T, K) softmax routing weights.
        """
        B, T, _ = x.shape
        K       = self.n_experts
        dtype   = x.dtype
        x_pool      = x.float().mean(dim=1, keepdim=True)                                 # (B, 1, D)
        tok_logits  = x_pool @ self.token_proj.float()                                    # (B, 1, K*2)
        tok_weights = F.softmax(
            tok_logits.view(B, 1, self.N_TOKEN_LAYERS, K), dim=-1
        ).to(dtype).expand(B, T, self.N_TOKEN_LAYERS, K).contiguous()                     # (B, T, 2, K)
        # Sequence routing: mean-pool → one matmul → expand
        seq_logits  = x_pool @ self.seq_proj.float()                                      # (B, 1, K*4)
        seq_weights = F.softmax(
            seq_logits.view(B, 1, self.N_SEQ_LAYERS, K), dim=-1
        ).to(dtype).expand(B, T, self.N_SEQ_LAYERS, K).contiguous()                         # (B, T, 4, K)
        return {
            'c_fc':         tok_weights[:, :, 0, :],   # (B, T, K)
            'c_proj_ffn':   tok_weights[:, :, 1, :],
            'c_q':          seq_weights[:, :, 0, :],
            'c_k':          seq_weights[:, :, 1, :],
            'c_v':          seq_weights[:, :, 2, :],
            'c_proj_attn':  seq_weights[:, :, 3, :],
        }

    def gate_parameters(self):
        """Routing projection tensors go into the lower-LR gate group."""
        if isinstance(self.token_proj, nn.Parameter):
            yield self.token_proj
            yield self.seq_proj


class RemixedBlock(nn.Module):
    """Attention-Grounded Context-Conditioned Linear block — 'weight' modulation path.

    Supports all novel ablation designs via config toggles:
      - cclblock_context_stream: 'local'|'shifted'|'ema'|'selective'|'multiscale'|'ssm'|
                                 'predictive_chunk'|'evidence_ssm'
      - cclblock_context_bank_size: 0=off, N=ContextBank with N prototypes
      - cclblock_per_head_ctx: separate context projections for attn vs ffn
      - cclblock_sparse_gate_k: 0=soft sigmoid, N=sparse top-k gate (via remixed_linear_kwargs)
      - cclblock_gate_temperature: sigmoid temperature for basis gate
    """
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = (_DenseAttnAdapter(config, layer_idx)
                     if getattr(config, 'p34_dense_attn', 0)
                     else RemixedMultiAttention(config, layer_idx))
        self.ffwd = (None if resolve_ffn_schedule(config, config.n_layer)[layer_idx] <= 0
                     else RemixedFeedForward(config, layer_idx))
        self._p24_sliced_scope = getattr(config, 'p24_sliced_weight_scope', 'per_token')
        self._p24_folded_scope = getattr(config, 'p24_folded_mod_scope', 'per_layer')
        self._p24_seq_scope = getattr(config, 'p24_sequence_gated_scope', 'per_layer')
        self._p24_any = bool(getattr(config, 'p24_use_sliced_weight', 0) or
                             getattr(config, 'p24_use_folded_mod', 0) or
                             getattr(config, 'p24_use_sequence_gated_linear', 0))
        ctx_dim = config.remix_context_dim
        stream_type = getattr(config, 'cclblock_context_stream', 'local')
        bank_size  = getattr(config, 'cclblock_context_bank_size', 0)
        per_head   = getattr(config, 'cclblock_per_head_ctx', False)
        ctx_source = getattr(config, 'cclblock_context_source', 'norm_x')
        chunk_size = getattr(config, 'cclblock_chunk_size', 0)
        self._modulation_mode = getattr(config, 'cclblock_modulation', 'weight')
        # 18H: Mixture norm
        if getattr(config, 'p18_mixture_norm', 0):
            self.norm_attn = MixtureNorm(config.n_embd)
            self.norm_mlp = MixtureNorm(config.n_embd)
        else:
            self.norm_attn = None
            self.norm_mlp = None
        # 18E: Stochastic depth (LayerDrop)
        self.layer_drop = getattr(config, 'p18_layer_drop', 0.0)
        # Phase 23: Block-level shared router.
        # Computes routing logits for ALL RemixedLinear layers in a single pair of matmuls per block.
        # When enabled, each RemixedLinear skips its own local template_route computation.
        self._shared_router = None
        if (getattr(config, 'p23_tiny_expert', 0) and
                getattr(config, 'p23_n_experts', 1) > 1 and
                getattr(config, 'p23_use_shared_block_router', 0)):
            self._shared_router = SharedBlockRouter(
                n_embd=config.n_embd,
                n_experts=getattr(config, 'p23_n_experts', 8),
                learned=bool(getattr(config, 'p23_learned_route', 0)),
            )

        # Phase 23: Block-level shared context gates.
        # Batches all 6 per-RL gate computations (basis_modulator + output_gate_coeffs)
        # into 3 shared matmuls, cutting gate overhead ~6×.
        self._ctx_gates = None
        use_context_flag = (config.remixed_linear_kwargs or {}).get('use_context', True)
        if getattr(config, 'remix_shared_context_gates', 0) and use_context_flag:
            # Read basis_size from the ACTUAL first RemixedLinear layer (self.attn is already built).
            # This correctly captures Fix-1B auto-scaling (max(default,min(in,out)//4)) without
            # re-implementing the formula here. Disable for LoKR since b_shrunk varies per-layer
            # making a single shared basis_size impossible — those blocks fall back to per-RL gates.
            _first_rl = next(
                (m for m in ([self.attn.c_q, self.attn.c_k, self.attn.c_v,
                              getattr(self.attn, 'c_proj', None)]
                             + ([self.ffwd.c_fc, self.ffwd.c_proj] if self.ffwd is not None else []))
                 if isinstance(m, (RemixedLinear, RemixedLinearFused))),
                None
            )
            if _first_rl is not None and not getattr(_first_rl, 'lokr_expert', False):
                _basis_size = _first_rl.basis_size
                _gate_rank  = getattr(config, 'remix_output_gate_rank', 8)
                _film_gate  = getattr(config, 'cclblock_film_gate', False)
                self._ctx_gates = SharedContextGates(
                    ctx_dim=ctx_dim,
                    basis_size=_basis_size,
                    gate_rank=_gate_rank,
                    film_gate=_film_gate,
                )

        # Phase 28C: shared W_b across attn Q/K/V/O.
        # Instead of 4 independent basis projections, compute h_basis ONCE per block
        # and inject it into all attention RemixedLinear layers before forward.
        # Saves ~3/4 of attention basis FLOPs with no quality cost (all layers see same input x).
        self._shared_basis_proj = None
        self._shared_ln_basis   = None
        self._attn_rl_layers    = []
        if getattr(config, 'p28_shared_basis', 0):
            # Collect all RemixedLinear layers in attention (Q, K, V, O)
            _attn_rl = [m for m in [
                getattr(self.attn, 'c_q', None), getattr(self.attn, 'c_k', None),
                getattr(self.attn, 'c_v', None), getattr(self.attn, 'c_proj', None),
            ] if isinstance(m, (RemixedLinear, RemixedLinearFused))]
            if _attn_rl:
                self._attn_rl_layers = _attn_rl
                _bs = _attn_rl[0].basis_size  # basis_size may differ if auto-scaled
                self._shared_basis_proj = nn.Linear(config.n_embd, _bs, bias=False)
                self._shared_ln_basis   = nn.LayerNorm(_bs)

        self.is_shifted   = (stream_type == 'shifted')
        self.per_head_ctx = per_head
        self.ctx_source   = ctx_source  # 'norm_x' | 'attn_heads' | 'attn_geometry'
        self.attn_shadow_dim = max(0, int(getattr(config, 'cclblock_attn_shadow_dim', 0)))
        if self.attn_shadow_dim > 0:
            self.shadow_ctx_proj = Linear(self.attn.shadow_dim, ctx_dim, bias=True)
            nn.init.zeros_(self.shadow_ctx_proj.weight)
            nn.init.zeros_(self.shadow_ctx_proj.bias)

        # Design 2: query-based context projection (head_dim → ctx_dim)
        # Only created when attn_heads source is active; replaces ctx_stream for FFN gating.
        if ctx_source == 'attn_heads':
            head_dim = config.n_embd // config.n_head
            self.ctx_proj_q = Linear(head_dim, ctx_dim, bias=True)
            nn.init.zeros_(self.ctx_proj_q.weight)  # start neutral
        elif ctx_source == 'attn_geometry':
            self.ctx_proj_geom = Linear(4, ctx_dim, bias=True)
            nn.init.zeros_(self.ctx_proj_geom.weight)

        def _make_stream():
            """Factory: returns the configured context stream module."""
            if bank_size > 0:
                return ContextBank(config.n_embd, ctx_dim, bank_size)
            elif chunk_size > 0 or stream_type == 'chunk':
                effective_chunk_size = chunk_size if chunk_size > 0 else 64
                return HardChunkContextStream(config.n_embd, ctx_dim, effective_chunk_size)
            elif stream_type == 'predictive_chunk':
                effective_chunk_size = chunk_size if chunk_size > 0 else 64
                return PredictiveChunkContextStream(config.n_embd, ctx_dim, effective_chunk_size)
            elif stream_type in ('local', 'shifted'):
                return LocalContextStream(config.n_embd, ctx_dim)
            elif stream_type == 'selective':
                return SelectiveContextStream(config.n_embd, ctx_dim)
            elif stream_type == 'boundary':
                return BoundaryGatedContextStream(config.n_embd, ctx_dim)
            elif stream_type == 'ema':
                return EMAContextStream(config.n_embd, ctx_dim, ema_factor=getattr(config, 'cclblock_ema_factor', 0.99))
            elif stream_type == 'dacs':
                return DetachedAttnContextStream(config.n_embd, ctx_dim)
            elif stream_type == 'prefix':
                return CausalPrefixContextStream(config.n_embd, ctx_dim)
            elif stream_type == 'warmup_ema':
                return WarmupEMAContextStream(config.n_embd, ctx_dim, ema_factor=getattr(config, 'cclblock_ema_factor', 0.99))
            elif stream_type == 'dacs_ema':
                return DACSEMAContextStream(config.n_embd, ctx_dim)
            elif stream_type == 'decay_prefix':
                return DecayPrefixContextStream(config.n_embd, ctx_dim, gamma=0.9)
            elif stream_type == 'ssm':
                return ParallelLinearContextStream(config.n_embd, ctx_dim)
            elif stream_type == 'evidence_ssm':
                return EvidenceAccumulationContextStream(
                    config.n_embd,
                    ctx_dim,
                    num_regimes=getattr(config, 'cclblock_num_regimes', 8),
                    temperature=getattr(config, 'cclblock_regime_temperature', 1.0),
                )
            else:  # multiscale
                return MultiScaleContext(config.n_embd, ctx_dim)

        # FSI/AESP/CKR/ARG/KFL/cond: no context stream needed (bypass in forward).
        # 'cond' conditions on the layer's own input, and under gate_source='ctx'
        # on the attention output the bypass path already hands the FFN. Either
        # way a context stream would only add parameters, and parameters set the
        # token budget under a fixed data:param ratio.
        if self._modulation_mode in ('fsi', 'aesp', 'ckr', 'giad', 'psg', 'splitstream', 'lokr', 'arg', 'kfl', 'cond'):
            self.ctx_stream = None
        elif per_head:
            # Design 7: two independent streams; attn uses pre-attn repr, ffn uses post-attn repr
            self.ctx_stream_attn = _make_stream()
            self.ctx_stream_ffn  = _make_stream()
        else:
            self.ctx_stream = _make_stream()

    def _is_local(self, stream):
        """Returns True if stream doesn't need prev_ctx (local or bank or shifted)."""
        return isinstance(stream, (LocalContextStream, ContextBank, HardChunkContextStream, PredictiveChunkContextStream, EvidenceAccumulationContextStream, BoundaryGatedContextStream, DetachedAttnContextStream, CausalPrefixContextStream, DecayPrefixContextStream))

    def _inject_shared_basis(self, norm_x):
        """Phase 28C: pre-compute shared h_basis from norm_x and inject into all attn RL layers."""
        if self._shared_basis_proj is not None:
            # Cast to projection weight dtype (model may be in bfloat16 while Linear starts float32)
            dtype = self._shared_basis_proj.weight.dtype
            h = self._shared_ln_basis(self._shared_basis_proj(norm_x.to(dtype=dtype)))
            for rl in self._attn_rl_layers:
                rl._h_basis = h

    def _clear_shared_basis(self):
        """Phase 28C: clear injected _h_basis to avoid stale state across forward calls."""
        for rl in self._attn_rl_layers:
            rl._h_basis = None

    def forward(self, x, ve, cos_sin, window_size, kv_cache, prev_ctx=None, p24_global_signal=None):
        x_entry = x  # snapshot before attn: = previous layer output (used for 'shifted' mode)
        p24_shared = None
        if self._p24_any:
            block_signal = norm(x_entry).mean(dim=1)
            use_global = ((self._p24_sliced_scope == 'global') or
                          (self._p24_folded_scope == 'global') or
                          (self._p24_seq_scope == 'global'))
            use_block = ((self._p24_sliced_scope == 'per_block') or
                         (self._p24_folded_scope == 'per_block') or
                         (self._p24_seq_scope == 'per_block'))
            shared_signal = p24_global_signal if (use_global and p24_global_signal is not None) else (block_signal if use_block else None)
            if shared_signal is not None:
                p24_shared = {'route_input': shared_signal, 'gate_input': shared_signal}

        # Phase 23 shared routing: compute all K*N_layers routing decisions in 2 matmuls
        # from norm(x) before any transformation.  Each sub-module receives its slice.
        block_route = None
        if self._shared_router is not None:
            block_route = self._shared_router(norm(x))

        # FSI/AESP/CKR/PGR/CIL/PRB/ARG/KFL bypass: these modes don't use context streams at all.
        # They derive their modulation signal directly from the attention output.
        if self._modulation_mode in ('fsi', 'aesp', 'ckr', 'ckr_ffn', 'com', 'giad', 'psg', 'splitstream', 'lokr', 'pgr', 'cil', 'prb', 'arg', 'kfl', 'cond'):
            norm_fn_attn = self.norm_attn if self.norm_attn is not None else norm
            norm_fn_mlp = self.norm_mlp if self.norm_mlp is not None else norm
            _nx = norm_fn_attn(x); self._inject_shared_basis(_nx)
            attn_out = self.attn(_nx, ve, cos_sin, window_size, kv_cache, context_state=None)
            self._clear_shared_basis()
            x = x + attn_out
            ctx_signal = attn_out.detach()
            # 18E: Stochastic depth — randomly skip FFN during training
            if self.training and self.layer_drop > 0 and torch.rand(1).item() < self.layer_drop:
                pass  # Skip FFN
            else:
                scale = 1.0 / (1.0 - self.layer_drop) if self.training and self.layer_drop > 0 else 1.0
                if self.ffwd is not None:
                    x = x + scale * self.ffwd(norm_fn_mlp(x), ctx_signal, p24_shared=p24_shared)
            self._last_ctx = ctx_signal
            return x, ctx_signal

        if self.attn_shadow_dim > 0:
            _nx = norm(x); self._inject_shared_basis(_nx)
            attn_out, shadow_ctx = self.attn.forward_with_shadow(
                _nx, ve, cos_sin, window_size, kv_cache, None
            )
            self._clear_shared_basis()
            x = x + attn_out
            ctx = self.shadow_ctx_proj(shadow_ctx)
            if self.ffwd is not None:
                x = x + self.ffwd(norm(x), ctx, p24_shared=p24_shared)
            self._last_ctx = ctx.detach()
            return x, self._last_ctx

        if self.ctx_source in ('attn_heads', 'attn_geometry'):
            # Design 2: context derived from query vectors (post-RoPE+QKnorm, detached)
            _nx = norm(x); self._inject_shared_basis(_nx)
            attn_out, q_stats = self.attn.forward_with_q_stats(
                _nx, ve, cos_sin, window_size, kv_cache, None)
            self._clear_shared_basis()
            x = x + attn_out
            if self.ctx_source == 'attn_heads':
                ctx = self.ctx_proj_q(q_stats)   # (B, T, ctx_dim)
            else:
                q_norm = q_stats.norm(dim=-1, keepdim=True)
                q_mean = q_stats.mean(dim=-1, keepdim=True)
                q_std = q_stats.std(dim=-1, keepdim=True)
                q_peak = q_stats.abs().amax(dim=-1, keepdim=True)
                geom = torch.cat([q_norm, q_mean, q_std, q_peak], dim=-1)
                ctx = self.ctx_proj_geom(geom)
            if self.ffwd is not None:
                x = x + self.ffwd(norm(x), ctx, p24_shared=p24_shared)
            self._last_ctx = ctx.detach()
            return x, self._last_ctx

        elif self.per_head_ctx:
            # Design 7: separate contexts from pre-attn and post-attn residuals
            ctx_src_attn = norm(x_entry)                  # before attention
            ctx_attn = self.ctx_stream_attn(ctx_src_attn) # (B, T, ctx_dim)
            # Run attention with dedicated attn context
            _nx = norm(x); self._inject_shared_basis(_nx)
            attn_out = self.attn(_nx, ve, cos_sin, window_size, kv_cache, ctx_attn)
            self._clear_shared_basis()
            x = x + attn_out
            ctx_src_ffn = norm(x_entry if self.is_shifted else x)
            ctx_ffn = self.ctx_stream_ffn(ctx_src_ffn)   # (B, T, ctx_dim)
            if self.ffwd is not None:
                x = x + self.ffwd(norm(x), ctx_ffn, p24_shared=p24_shared)
            self._last_ctx = ctx_ffn.detach()
            return x, self._last_ctx
        else:
            stream = self.ctx_stream
            local = self._is_local(stream)
            is_dacs = isinstance(stream, (DetachedAttnContextStream, DACSEMAContextStream))

            if is_dacs:
                _nx = norm(x); self._inject_shared_basis(_nx)
                attn_out = self.attn(_nx, ve, cos_sin, window_size, kv_cache,
                                     context_state=None, route_weights=block_route)
                self._clear_shared_basis()
                x = x + attn_out
                ctx = stream(attn_out, None if local else prev_ctx)
            else:
                _nx = norm(x); self._inject_shared_basis(_nx)
                attn_out = self.attn(_nx, ve, cos_sin, window_size, kv_cache,
                                     None if (local or self.is_shifted) else prev_ctx,
                                     route_weights=block_route)
                self._clear_shared_basis()
                x = x + attn_out
                ctx_src = norm(x_entry if self.is_shifted else x)
                ctx = stream(ctx_src, None if local else prev_ctx)

            # Phase 23: shared context gates — 3 matmuls for all 6 RL gate decisions
            block_ctx_gates = self._ctx_gates(ctx) if self._ctx_gates is not None else None

            if self.ffwd is not None:
                x = x + self.ffwd(norm(x), ctx, route_weights=block_route,
                                   context_gates=block_ctx_gates, p24_shared=p24_shared)
            out_ctx = ctx.detach() if (local or self.is_shifted or is_dacs) else ctx
            self._last_ctx = out_ctx
            return x, out_ctx


def has_ve(layer_idx, n_layer):
    """Returns True if GPT layer should have Value Embedding (alternating, last layer always included)."""
    return layer_idx % 2 == (n_layer - 1) % 2

def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:] # split up last dim into two halves
    y1 = x1 * cos + x2 * sin # rotate pairs of dims
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)

class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        # Phase 21: optionally replace attention projections with MoELinear
        _per_k = getattr(config, 'p21_per_experts', 0)
        _per_attn = getattr(config, 'p21_per_attn', 0)
        # Phase 22: MoNE/LRCFB attention projections (same pattern as MoNE_MLP/FrozenRoutedMLP)
        _p22_route = getattr(config, 'p22_attn_moe_route', 'none')
        _lrcfb_k = getattr(config, 'p20_lrcfb_branches', 0)
        _mone_k = getattr(config, 'p20_mone_experts', 0)
        _attn_moe_k = max(_lrcfb_k, _mone_k) if _p22_route != 'none' else 0
        self._attn_moe_k = _attn_moe_k  # Python int, compile-safe branch condition
        if _attn_moe_k > 0:
            # K expert projection sets — mirrors MoNE_MLP.experts_fc / FrozenRoutedMLP.branches_fc
            D = self.n_embd
            D_q = self.n_head * self.head_dim
            D_kv = self.n_kv_head * self.head_dim
            self.c_q = nn.ModuleList([Linear(D, D_q, bias=False) for _ in range(_attn_moe_k)])
            self.c_k = nn.ModuleList([Linear(D, D_kv, bias=False) for _ in range(_attn_moe_k)])
            self.c_v = nn.ModuleList([Linear(D, D_kv, bias=False) for _ in range(_attn_moe_k)])
            self.c_proj = nn.ModuleList([Linear(D, D, bias=False) for _ in range(_attn_moe_k)])
            # Shared router for all 4 projections (ensures Q/K coordinate coherence)
            # Mirrors MoNE_MLP.router (learned) / FrozenRoutedMLP.content_proj (frozen)
            self._attn_moe_seq = (_p22_route == 'sequence')
            if _lrcfb_k > 0:
                _learned = bool(getattr(config, 'p20_lrcfb_learned', 0))
                self._attn_moe_topk = getattr(config, 'p20_lrcfb_topk', 0)
            else:
                _learned = not bool(getattr(config, 'p20_mone_frozen', 0))
                self._attn_moe_topk = getattr(config, 'p20_mone_topk', 0)
            if _learned:
                self._attn_router = Linear(D, _attn_moe_k, bias=False)
            else:
                self.register_buffer('_attn_router_proj', torch.randn(D, _attn_moe_k) / (D ** 0.5))
        elif _per_k > 0 and _per_attn:
            _per_topk = getattr(config, 'p21_per_topk', 0)
            _per_learned = bool(getattr(config, 'p21_per_learned', 0))
            self.c_q = MoELinear(self.n_embd, self.n_head * self.head_dim, K=_per_k,
                                  topk=_per_topk, learned_route=_per_learned)
            self.c_k = MoELinear(self.n_embd, self.n_kv_head * self.head_dim, K=_per_k,
                                  topk=_per_topk, learned_route=_per_learned)
            self.c_v = MoELinear(self.n_embd, self.n_kv_head * self.head_dim, K=_per_k,
                                  topk=_per_topk, learned_route=_per_learned)
            self.c_proj = MoELinear(self.n_embd, self.n_embd, K=_per_k,
                                    topk=_per_topk, learned_route=_per_learned)
        else:
            self.c_q = Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
            self.c_k = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
            self.c_v = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
            self.c_proj = Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = min(self.n_embd, 32)
        # 19I: VE gate with optional learnable bias
        use_ve_bias = bool(getattr(config, 'p19_ve_bias', 0))
        self.ve_gate = Linear(self.ve_gate_channels, self.n_kv_head, bias=use_ve_bias) if has_ve(layer_idx, config.n_layer) else None
        # 19B: Per-head importance scaling — learned multiplier on attention output
        # softplus(0.5413) ≈ 1.0, so head_scale_raw inited to 0.5413 gives scale=1.0 (identity)
        self.head_importance = None
        if getattr(config, 'p19_head_importance', 0):
            self.head_importance = nn.Parameter(torch.full((self.n_head,), 0.5413))
        # 19D: Per-head attention logit temperature — scales Q before flash_attn
        # softplus(0.5413) ≈ 1.0, so scale=1.0 at init (identity)
        self.attn_logit_scale = None
        if getattr(config, 'p19_attn_logit_bias', 0):
            self.attn_logit_scale = nn.Parameter(torch.full((self.n_head,), 0.5413))

    def _compute_attn_moe_route(self, x):
        """Phase 22: compute shared routing weights for all 4 attention projections.
        Same routing logic as MoNE_MLP.forward / FrozenRoutedMLP.forward."""
        if hasattr(self, '_attn_router'):
            logits = self._attn_router(x)  # (B, T, K) — learned, like MoNE_MLP.router
        else:
            logits = torch.matmul(x.float(), self._attn_router_proj.float())  # frozen, like FrozenRoutedMLP.content_proj
        if self._attn_moe_seq:
            logits = logits.mean(dim=1, keepdim=True)  # (B, 1, K) per-sequence
        topk = self._attn_moe_topk
        if topk > 0 and topk < self._attn_moe_k:
            _, topk_idx = logits.topk(topk, dim=-1)
            mask = torch.zeros_like(logits).scatter_(-1, topk_idx, 1.0)
            logits = logits.masked_fill(mask == 0, float('-inf'))
        return F.softmax(logits, dim=-1).to(x.dtype)

    def forward(self, x, ve, cos_sin, window_size, kv_cache, token_active=None, eet_frozen_kv=False, frozen_k=None, frozen_v=None,
                attn_mask=None):
        B, T, C = x.size()

        # Project the input to get queries, keys, and values
        # Shape: (B, T, H, D) - FA3's native layout, no transpose needed!
        if self._attn_moe_k > 0:
            # Phase 22: MoNE/LRCFB routing — same weighted-sum pattern as MoNE_MLP.forward
            _rw = self._compute_attn_moe_route(x)  # (B, 1, K) or (B, T, K)
            q = sum(_rw[..., k:k+1] * self.c_q[k](x) for k in range(self._attn_moe_k))
            q = q.view(B, T, self.n_head, self.head_dim)
            k = sum(_rw[..., k:k+1] * self.c_k[k](x) for k in range(self._attn_moe_k))
            k = k.view(B, T, self.n_kv_head, self.head_dim)
            v = sum(_rw[..., k:k+1] * self.c_v[k](x) for k in range(self._attn_moe_k))
            v = v.view(B, T, self.n_kv_head, self.head_dim)
        else:
            q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
            k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
            v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Value residual (ResFormer): mix in value embedding with input-dependent gate per head
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))  # (B, T, n_kv_head), range (0, 2)
            v = v + gate.unsqueeze(-1) * ve

        # Apply Rotary Embeddings to queries and keys to get relative positional encoding
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k) # QK norm

        # 19D: Per-head learned QK temperature — scale Q to control attention sharpness
        # Effective dot product: (scale * q) · k = scale * (q · k)
        # FA3-safe: we scale Q before the call, not the attention matrix
        if self.attn_logit_scale is not None:
            scale = F.softplus(self.attn_logit_scale).to(q.dtype)  # (n_head,)
            q = q * scale.view(1, 1, self.n_head, 1)  # broadcast: (1, 1, H, 1)

        # If Option B: frozen KV injection
        if token_active is not None and eet_frozen_kv and frozen_k is not None and frozen_v is not None:
            active_mask = token_active.unsqueeze(-1).unsqueeze(-1)  # (B, T, 1, 1)
            k = torch.where(active_mask, k, frozen_k)
            v = torch.where(active_mask, v, frozen_v)

        # Cache actual computed/injected keys and values
        self._last_k = k
        self._last_v = v

        # Flash Attention (FA3 on Hopper+, PyTorch SDPA fallback elsewhere)
        # window_size is (left, right) tuple: (N, 0) for causal, (-1, 0) for full context
        if kv_cache is None and attn_mask is not None:
            # Lanes: an explicit (1 or B, 1, T, T) visibility mask over the row (True = attend),
            # built by lane_mask(); it replaces the causal mask, so no sliding window applies.
            dt = torch.bfloat16 if q.is_cuda else q.dtype
            y = F.scaled_dot_product_attention(
                q.transpose(1, 2).to(dt), k.transpose(1, 2).to(dt), v.transpose(1, 2).to(dt),
                attn_mask=attn_mask, enable_gqa=self.n_kv_head != self.n_head).transpose(1, 2)
        elif kv_cache is None:
            # Training: causal attention with optional sliding window
            if token_active is not None:
                # Build custom attention mask
                causal_mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
                active_queries = token_active.unsqueeze(1).unsqueeze(3)  # (B, 1, T, 1)
                
                if not eet_frozen_kv:
                    # Option A: Active queries attend ONLY to active keys (complete skip)
                    active_keys = token_active.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T)
                    combined_mask = causal_mask.unsqueeze(0).unsqueeze(1) & active_keys & active_queries
                else:
                    # Option B: Active queries attend to active + frozen keys
                    combined_mask = causal_mask.unsqueeze(0).unsqueeze(1) & active_queries
                
                y = F.scaled_dot_product_attention(
                    q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                    attn_mask=combined_mask
                ).transpose(1, 2)
            else:
                y = flash_attn.flash_attn_func(q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16), causal=True, window_size=window_size)
        else:
            # Inference: use flash_attn_with_kvcache which handles cache management
            k_cache, v_cache = kv_cache.get_layer_cache(self.layer_idx)
            y = flash_attn.flash_attn_with_kvcache(
                q.to(torch.bfloat16), k_cache, v_cache,
                k=k.to(torch.bfloat16), v=v.to(torch.bfloat16),
                cache_seqlens=kv_cache.cache_seqlens,
                causal=True,
                window_size=window_size,
                graph_safe=getattr(kv_cache, 'graph_safe', False),
            )
            # Advance position after last layer processes
            if self.layer_idx == kv_cache.n_layers - 1:
                kv_cache.advance(T)

        # 19B: Per-head importance scaling — learned multiplier on post-attention output
        if self.head_importance is not None:
            # y shape: (B, T, n_head, head_dim)
            his = F.softplus(self.head_importance).to(y.dtype)  # (n_head,)
            y = y * his.view(1, 1, self.n_head, 1)  # broadcast

        # Re-assemble the heads and project back to residual stream
        y = y.contiguous().view(B, T, -1)
        if self._attn_moe_k > 0:
            y = sum(_rw[..., k:k+1] * self.c_proj[k](y) for k in range(self._attn_moe_k))
        else:
            y = self.c_proj(y)
        return y

    def _project_kv(self, x_kv, ve_kv, cos_kv, sin_kv):
        """Project a full-length (B, T, C) state into RoPE'd, QK-normed keys and values."""
        B, T, _ = x_kv.size()
        k = self.c_k(x_kv).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x_kv).view(B, T, self.n_kv_head, self.head_dim)
        if ve_kv is not None and self.ve_gate is not None:
            ve_kv = ve_kv.view(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(x_kv[..., :self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve_kv
        k = norm(apply_rotary_emb(k, cos_kv, sin_kv))
        return k, v

    def forward_split(self, x_q, cos_sin_q, q_pos, k, v, window_size, ve_q=None,
                      key_mask=None, n_q=None):
        """EET context-restoring attention: few queries, full-length keys/values.

        Standard ``forward`` requires queries and keys to be the same token set, which is
        why the compute-skip path currently drops exited tokens out of attention entirely.
        This variant keeps the query set small (the surviving tokens) while letting them
        read every position, so removing a token from the computation no longer removes it
        from everyone else's context.

        Args:
            x_q:       (B, K, C) hidden states of the active tokens (queries only).
            cos_sin_q: RoPE tables already gathered at the active positions.
            q_pos:     (B, K) absolute position of each active token, ascending.
            k, v:      (B, T, n_kv_head, head_dim) keys/values for ALL positions, already
                       RoPE'd and QK-normed (see ``_project_kv``).
            window_size: (left, right) as elsewhere; left < 0 means full context.
            ve_q:      unused for keys/values, kept for signature symmetry.
            key_mask:  optional (B, T) bool; False positions are excluded from every
                       query's attention. Used by the context oracle to reproduce EET's
                       context loss on a dense model without changing its depth.
            n_q:       the query count as a plain Python int. Pass it whenever the caller
                       knows it. Reading it off x_q.size(1) instead lets torch.compile
                       carry it as a symbolic size, and inductor then bakes one layer's
                       mask stride into a kernel that another layer reuses:
                         assert_size_stride(constant_pad_nd, (64, 1, s0, 2056), ...)
                         AssertionError: expected size 64==64, stride 430848==3691776
                       (430848 = 204*2112, 3691776 = 1748*2112: two different layers'
                       capacities sharing one compiled backward).

        Returns:
            (B, K, C) attention output for the active tokens.
        """
        B, K, C = x_q.size()
        if n_q is not None:
            K = int(n_q)          # concrete: forces a guard instead of a symbolic size
        T = k.size(1)
        q = self.c_q(x_q).view(B, K, self.n_head, self.head_dim)
        cos_q, sin_q = cos_sin_q
        q = norm(apply_rotary_emb(q, cos_q, sin_q))
        if self.attn_logit_scale is not None:
            scale = F.softplus(self.attn_logit_scale).to(q.dtype)
            q = q * scale.view(1, 1, self.n_head, 1)

        # Causal + sliding-window mask over absolute positions: (B, 1, K, T)
        k_pos = torch.arange(T, device=x_q.device).view(1, 1, 1, T)
        qp = q_pos.view(B, 1, K, 1)
        mask = k_pos <= qp
        left = window_size[0] if window_size is not None else -1
        if left is not None and left >= 0:
            mask = mask & (k_pos > qp - left)
        if key_mask is not None:
            mask = mask & key_mask.view(B, 1, 1, T)
        # A query whose whole row is masked would make softmax produce NaN. Keep the
        # diagonal alive so every query can at least attend to itself.
        self_ok = k_pos == qp
        mask = mask | self_ok

        # Canonical layout: SDPA pads the mask's last dim for alignment, and a
        # non-contiguous input makes that pad's stride harder for inductor to guard.
        # Keys and values come from a full-length state the caller assembles, and one
        # fp32 parameter anywhere in that expression (resid_lambdas, the x0 weights)
        # promotes the whole thing. Align here rather than trusting every caller.
        y = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.to(q.dtype).transpose(1, 2), v.to(q.dtype).transpose(1, 2),
            attn_mask=mask.contiguous()
        ).transpose(1, 2)

        if self.head_importance is not None:
            his = F.softplus(self.head_importance).to(y.dtype)
            y = y * his.view(1, 1, self.n_head, 1)
        y = y.contiguous().view(B, K, -1)
        return self.c_proj(y)


@torch._dynamo.disable
def forward_split_eager(block, *args, **kwargs):
    """Run Block.forward_split outside the compiled graph (config.eet_kv_eager)."""
    return block.forward_split(*args, **kwargs)


class SpectralReparamLinear(nn.Module):
    """Phase 19G: Spectral Reparameterization.

    W = U @ diag(sigma) @ V^T where U, V are frozen (from SVD of initial W),
    only sigma (the singular values) is learned.

    At init: sigma = actual singular values of W → exact identity with original W.
    At inference: reparameterizable back to a single W_eff = U @ diag(sigma) @ V^T.
    Params added: min(in_features, out_features) per layer.
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # These will be filled in init_weights after the base Linear is initialized
        self.register_buffer('U', torch.zeros(out_features, min(in_features, out_features)))
        self.register_buffer('Vt', torch.zeros(min(in_features, out_features), in_features))
        self.sigma = nn.Parameter(torch.ones(min(in_features, out_features)))

    def init_from_weight(self, weight):
        """Decompose a weight matrix and store frozen U, V^T + learned sigma.
        Call this from init_weights() AFTER the weight has been properly initialized."""
        with torch.no_grad():
            W = weight.float()  # SVD in float32 for precision
            U, S, Vt = torch.linalg.svd(W, full_matrices=False)
            self.U.copy_(U.to(self.U.dtype))
            self.Vt.copy_(Vt.to(self.Vt.dtype))
            self.sigma.data.copy_(S)

    def forward(self, x):
        dtype = x.dtype
        # W_eff = U @ diag(sigma) @ V^T
        # Compute as: x @ V^T^T @ diag(sigma) @ U^T = x @ V @ diag(sigma) @ U^T
        # But it's more efficient to compute the effective weight directly:
        # y = (x @ V^T.T) * sigma @ U^T = x @ (V^T.T * sigma) @ U^T
        Vt = self.Vt.to(dtype)  # (k, in)
        U = self.U.to(dtype)    # (out, k)
        sigma = self.sigma.to(dtype)  # (k,)
        # x: (B, T, in) → (B, T, k) → (B, T, out)
        h = F.linear(x, Vt)  # x @ V = x @ Vt.T → (B, T, k) — F.linear does x @ W^T
        h = h * sigma  # element-wise scale by singular values
        return F.linear(h, U)  # h @ U^T → (B, T, out)


# ── Phase 20: Context-Conditioned Dynamic Weight Computation ────────────────

class MoNE_MLP(nn.Module):
    """20F: Mixture of Experts MLP.

    K expert MLPs with content-based routing.

    Modes:
      - narrow=True (default):  each expert has hidden_dim = 4*D/K (param parity)
      - narrow=False:           each expert has hidden_dim = 4*D (full-size, K× params)
      - frozen_route=False (default): learned Linear router
      - frozen_route=True:      frozen random projection (no gradient through routing)
      - topk=0:                 soft MoE (compute all, weighted sum)
      - topk=N:                 sparse top-N routing

    Training: compute active experts, weight by softmax routing scores.
    """
    def __init__(self, config, n_experts=4, topk=0, narrow=True, frozen_route=False):
        super().__init__()
        self.n_experts = n_experts
        self.topk = topk  # 0 = compute all (soft MoE), >0 = sparse top-k
        self.narrow = narrow
        self.frozen_route = frozen_route
        D = config.n_embd
        if topk > 0:
            expert_hidden = 4 * D // topk
            assert expert_hidden * topk == 4 * D, (
                f"4*D={4*D} must be divisible by topk={topk}")
        elif narrow:
            expert_hidden = 4 * D // n_experts
            assert expert_hidden * n_experts == 4 * D, (
                f"4*D={4*D} must be divisible by n_experts={n_experts}")
        else:
            expert_hidden = 4 * D  # full-size: each expert is as wide as the dense MLP
        self.expert_hidden = expert_hidden
        # K expert MLPs
        self.experts_fc = nn.ModuleList([
            Linear(D, expert_hidden, bias=False) for _ in range(n_experts)])
        self.experts_proj = nn.ModuleList([
            Linear(expert_hidden, D, bias=False) for _ in range(n_experts)])
        # Router: learned or frozen
        if frozen_route:
            route_init = torch.randn(D, n_experts) / (D ** 0.5)
            self.register_buffer('router_proj', route_init)
            self.router = None
        else:
            self.router = Linear(D, n_experts, bias=False)
        # Load-balancing auxiliary loss coefficient (stored, not a parameter)
        self._aux_balance_coeff = 0.01
        # Cache routing weights for diagnostics (not a parameter, not a buffer)
        self._last_router_entropy = 0.0
        self._last_expert_load = None

    def forward(self, x):
        B, T, D = x.shape
        dtype = x.dtype
        # Router: content-dependent routing scores (learned or frozen)
        if self.router is not None:
            router_logits = self.router(x)  # (B, T, K)
        else:
            router_logits = x.float() @ self.router_proj.float()  # (B, T, K)
        router_weights = F.softmax(router_logits.float(), dim=-1).to(dtype)  # (B, T, K)

        # Cache for diagnostics
        with torch.no_grad():
            rw_float = router_weights.float()
            log_rw = torch.log(rw_float.clamp(min=1e-8))
            self._last_router_entropy = -(rw_float * log_rw).sum(dim=-1).mean().item()
            self._last_expert_load = rw_float.mean(dim=(0, 1))  # (K,) — average load per expert

        if self.topk > 0 and self.topk < self.n_experts:
            # Sparse top-k: only compute selected experts
            topk_weights, topk_indices = router_weights.topk(self.topk, dim=-1)  # (B, T, topk)
            topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-8)  # renormalize
            y = torch.zeros(B, T, D, device=x.device, dtype=dtype)
            for rank_idx in range(self.topk):
                expert_idx = topk_indices[..., rank_idx]  # (B, T)
                weight = topk_weights[..., rank_idx]  # (B, T)
                # Compute each expert for all tokens, mask by selection
                for k in range(self.n_experts):
                    mask = (expert_idx == k)  # (B, T) bool
                    if mask.any():
                        h = self.experts_fc[k](x)
                        h = F.relu(h).square()
                        h = self.experts_proj[k](h)
                        y = y + (weight * mask.to(dtype)).unsqueeze(-1) * h
        else:
            # Soft routing: compute all experts, weight by routing scores
            y = torch.zeros(B, T, D, device=x.device, dtype=dtype)
            for k in range(self.n_experts):
                h = self.experts_fc[k](x)
                h = F.relu(h).square()
                h = self.experts_proj[k](h)
                y = y + router_weights[..., k:k+1] * h

        return y


def _moe_optimal_topk(n_experts: int, c: float = 0.3) -> int:
    """Compute theoretically optimal number of active experts for a Standard MoE.

    From the effective-parameter scaling law  P_eff ≈ P_active · E^c:
    * P_active is fixed by param-parity (topk · H_expert = 4D, where H_expert = 4D/E).
    * The model should use exactly E tokens per token when compute parity ≡ dense,
      i.e. topk=E  (0% sparsity) is optimal when E is small.
    * As E → ∞ the optimal sparsity fraction approaches 1-c, giving
          topk_opt = E^(1-c)   [rounds to nearest int, clamped ≥ 1]

    With c=0.3: topk_opt(8) = 8^0.7 ≈ 5   (37.5% sparsity)
                topk_opt(4) = 4^0.7 ≈ 3   (25% sparsity)
                topk_opt(2) = 2^0.7 ≈ 1.6 → 2 (0% sparsity ≡ dense)
    """
    return max(1, round(n_experts ** (1.0 - c)))


class StandardMoE_MLP(nn.Module):
    """Phase 23: Standard Mixture-of-Experts MLP with learned top-k routing.

    Expert sizing (compute parity per active expert):
        H_expert = 4 * D // topk
    so topk active experts together match the dense FFN FLOPs exactly.
        topk=1 → H = 4*D  (full-size experts, Switch-Transformer style)
        topk=2 → H = 2*D
        topk=E → H = 4*D//E

    Load-balance aux loss (Switch Transformer):
        L_aux = aux_weight * E * Σ_e (mean_gate_prob_e * mean_dispatch_frac_e)

    Memory: uses per-expert gradient checkpointing so h_e intermediates
    are recomputed during backward rather than stored, keeping activation
    memory at O(N*H) regardless of E.
    """

    def __init__(self, config, n_experts: int = 8, topk: int = 1, aux_weight: float = 0.01, use_quantile_route: int = 0):
        super().__init__()
        D = config.n_embd
        self.n_embd    = D
        self.n_experts = n_experts
        self.aux_weight = aux_weight

        # Resolve topk=-1 → optimal
        if topk == -1:
            topk = max(1, round(n_experts ** 0.7))
        self.topk = topk

        # Full-size FFN per expert — same as Switch Transformer.
        # Active FLOPs per token = topk × (2 × D × 4D) = topk × dense_MLP_flops.
        # Total params = E × dense_MLP_params (E× overhead, but full expressivity per expert).
        H = 4 * D
        self.expert_hidden = H
        print(f"StandardMoE_MLP: E={n_experts}, H={H}, topk={topk}  "
              f"(active FLOPs ≈ dense)")

        # Expert weights: stacked so all participate in every forward → DDP-safe
        #   fc_w:   (E, H, D)   up-projection
        #   proj_w: (E, D, H)   down-projection
        self.fc_w   = nn.Parameter(torch.empty(n_experts, H, D))
        self.proj_w = nn.Parameter(torch.empty(n_experts, D, H))

        # Router: QuantileBalanced or simple nn.Linear
        self.use_quantile_route = use_quantile_route
        if use_quantile_route == 1 and topk > 0:
            self.router = QuantileBalancedRouter(D, n_experts, topk, learned=True)
        else:
            self.router = nn.Linear(D, n_experts, bias=False)
            # Small random init to break expert symmetry at step 0.
            # Zero-init causes all experts to receive identical gradients
            # (uniform softmax → identical dispatch) and never differentiate.
            nn.init.normal_(self.router.weight, std=D ** -0.5)

        # Weight init (Kaiming up, zeros down — same as dense MLP convention)
        nn.init.kaiming_uniform_(self.fc_w.view(n_experts * H, D), a=math.sqrt(5))
        self.fc_w.data = self.fc_w.data.view(n_experts, H, D)
        nn.init.zeros_(self.proj_w)

        # Diagnostics (not checkpointed, not saved)
        self._last_router_logits = None
        self._last_expert_load   = None

    # ------------------------------------------------------------------ #
    #  Single-expert forward (called through checkpoint to save memory)   #
    # ------------------------------------------------------------------ #
    def _expert_fwd(self, e: int, x_flat):
        """Run expert e on all N tokens. Returns (N, D)."""
        # Cast weights to input dtype: fc_w/proj_w are float32 by default (torch.empty),
        # but x_flat is bfloat16 during training. torch.compile enforces strict dtype matching.
        dtype = x_flat.dtype
        h = F.relu(x_flat @ self.fc_w[e].T.to(dtype)).square()
        return h @ self.proj_w[e].T.to(dtype)

    # ------------------------------------------------------------------ #
    #  Aux loss (call after forward, before optimizer step)               #
    # ------------------------------------------------------------------ #
    def compute_aux_loss(self):
        """Switch Transformer load-balance auxiliary loss.

        For soft/top-all routing (topk >= n_experts), there is no load-balance
        problem (every expert always fires), so we skip the aux loss entirely.
        For sparse top-k routing, we compute dispatch fractions using the actual
        top-k mask, not the hard top-1 argmax.
        """
        if self._last_router_logits is None:
            return None
        # No load-balance problem when all experts are active
        if self.topk >= self.n_experts:
            return None
        logits = self._last_router_logits          # (N, E)
        E = self.n_experts
        probs = F.softmax(logits.float(), dim=-1)  # (N, E)
        mean_prob = probs.mean(dim=0)              # (E,)
        # Dispatch fraction: fraction of tokens routed to each expert via top-k
        topk_idx = logits.topk(self.topk, dim=-1).indices  # (N, k)
        dispatch = torch.zeros(logits.shape[0], E, device=logits.device)
        dispatch.scatter_(1, topk_idx, 1.0 / self.topk)   # (N, E) uniform over selected experts
        mean_frac = dispatch.mean(dim=0)                    # (E,)
        return self.aux_weight * E * (mean_prob * mean_frac).sum()

    # ------------------------------------------------------------------ #
    #  Forward                                                            #
    # ------------------------------------------------------------------ #
    def forward(self, x):
        B, T, D = x.shape
        E = self.n_experts
        N = B * T

        x_flat = x.reshape(N, D)                              # (N, D)

        # Route — cast to router's parameter dtype to avoid BF16/FP32 mismatch under compile
        if self.use_quantile_route == 1 and self.topk > 0:
            gate = self.router(x)                                 # (B, T, E) -> normalised weights
            logits = None # Quantile doesn't produce raw logits for aux loss
        else:
            logits  = self.router(x_flat.to(self.router.weight.dtype))  # (N, E)
            self._last_router_logits = logits   # kept alive for compute_aux_loss grad
            gate = F.softmax(logits.float(), dim=-1).to(x.dtype)  # (N, E)
            gate = gate.reshape(B, T, E)                          # (B, T, E)

        # Flatten gate for dispatch
        gate_flat = gate.reshape(N, E)

        # Diagnostics
        with torch.no_grad():
            self._last_expert_load = gate_flat.float().mean(dim=0)  # (E,)

        # Build dispatch weights (N, E): sparse top-k or full soft
        if 0 < self.topk < E and (self.use_quantile_route != 1):
            topk_w, topk_idx = gate_flat.topk(self.topk, dim=-1)   # (N, k)
            topk_w = topk_w / (topk_w.sum(-1, keepdim=True) + 1e-8)
            dispatch = torch.zeros(N, E, dtype=x.dtype, device=x.device)
            dispatch.scatter_add_(1, topk_idx, topk_w)         # (N, E)
        else:
            # If quantile route is used, 'gate' is already sparsified and normalised by the router
            dispatch = gate_flat                                     # (N, E)

        # Run each expert with gradient checkpointing:
        #   h_e is recomputed during backward → not stored → no 32 GiB OOM
        output = torch.zeros(N, D, dtype=x.dtype, device=x.device)
        for e in range(E):
            w_e = dispatch[:, e].unsqueeze(-1)                 # (N, 1)
            # checkpoint avoids storing the (N, H) hidden state between fwd/bwd
            out_e = torch.utils.checkpoint.checkpoint(
                self._expert_fwd, e, x_flat, use_reentrant=False
            )                                                   # (N, D)
            output = output + out_e * w_e

        return output.reshape(B, T, D)


class FrozenRoutedMLP(nn.Module):
    """20C: Content-Routed Branched MLP (LRCFB).

    K MLP branches mixed by a content-dependent routing signal.

    Modes (controlled by config flags):
      - narrow=False (default): each branch is full-size (K× params overhead)
      - narrow=True:  each branch has H//K hidden dim (param parity with baseline)
      - learned_route=False (default): routing projection R is FROZEN (buffer, no grad)
      - learned_route=True:  routing projection R is a learnable parameter
      - topk=0 (default): soft routing (all branches computed, weighted sum)

      - topk=N: hard top-N routing (only top-N branches computed, rest zeroed)

    alpha = softmax(x @ R)  — content-dependent routing
    y = Σ_k alpha_k * MLP_k(x)

    With narrow=True:
      Normal MLP: fc (128→512), proj (512→128) = 131K params
      K=4 narrow: 4× [fc (128→128), proj (128→128)] = 131K params (same!)
      But each token gets a DIFFERENT blend of 4 specialized sub-MLPs.
    """
    def __init__(self, config, n_branches=4, narrow=False, learned_route=False, topk=0):
        super().__init__()
        self.n_branches = n_branches
        self.narrow = narrow
        self.topk = topk
        D = config.n_embd
        H_full = 4 * D
        if topk > 0:
            H_per_branch = H_full // topk
            assert H_per_branch * topk == H_full, f"4*D={H_full} must be divisible by topk={topk}"
        elif narrow:
            H_per_branch = H_full // n_branches
            assert H_per_branch * n_branches == H_full, f"4*D={H_full} must be divisible by n_branches={n_branches}"
        else:
            H_per_branch = H_full
        self.H_per_branch = H_per_branch
        # K MLP branches (narrow or full-size)
        self.branches_fc = nn.ModuleList([
            Linear(D, H_per_branch, bias=False) for _ in range(n_branches)])
        self.branches_proj = nn.ModuleList([
            Linear(H_per_branch, D, bias=False) for _ in range(n_branches)])
        # Content routing projection
        # Frozen (buffer) or learned (parameter) depending on flag
        route_init = torch.randn(D, n_branches) / (D ** 0.5)
        if learned_route:
            self.content_proj = nn.Parameter(route_init)
        else:
            self.register_buffer('content_proj', route_init)
        self.learned_route = learned_route
        # Cache for diagnostics
        self._last_routing_entropy = 0.0
        self._last_routing_weights = None

    def forward(self, x):
        B, T, D = x.shape
        dtype = x.dtype
        # Content-dependent routing
        logits = x.float() @ self.content_proj.float()  # (B, T, K)
        weights = F.softmax(logits, dim=-1).to(dtype)  # (B, T, K)

        # Cache for diagnostics
        with torch.no_grad():
            w_float = weights.float()
            log_w = torch.log(w_float.clamp(min=1e-8))
            self._last_routing_entropy = -(w_float * log_w).sum(dim=-1).mean().item()
            self._last_routing_weights = w_float.mean(dim=(0, 1))

        # Top-k sparsification: zero out non-top-k weights
        if self.topk > 0 and self.topk < self.n_branches:
            topk_vals, topk_idx = weights.topk(self.topk, dim=-1)
            mask = torch.zeros_like(weights)
            mask.scatter_(-1, topk_idx, 1.0)
            weights = weights * mask
            # Re-normalize so weights sum to 1
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        # Compute active branches, weight by routing
        y = torch.zeros(B, T, D, device=x.device, dtype=dtype)
        for k in range(self.n_branches):
            if self.topk > 0 and self.topk < self.n_branches:
                # Skip branches with zero weight (top-k optimization)
                # Note: this is a per-batch check, not per-token. Full per-token
                # skipping would require scatter/gather which breaks compile.
                # Keep simple for now.
                pass
            h = self.branches_fc[k](x)
            h = F.relu(h).square()
            h = self.branches_proj[k](h)
            y = y + weights[..., k:k+1] * h

        return y


class DetachedRoutedMLP(nn.Module):
    """20D: Detached-Gradient Content Routing (DGCR).

    K full-rank MLP branches with a LEARNED router, but the router's main-loss
    gradient is stopped. The router only receives gradient through the auxiliary
    routing loss, which trains it to predict which branch would minimize
    per-token loss ex-post.

    Main forward path:
        alpha = router(x.detach()).detach()  # stop grad BOTH ways
        y = Σ_k alpha_k * MLP_k(x)

    Auxiliary loss (computed separately):
        For each branch k, compute per-token error ||MLP_k(x) - y_target||²
        Train router to assign high weight to the lowest-error branch

    This avoids FM1 (gradient fracturing) because the main loss never
    backprops through the router. The router learns independently via aux loss.

    NOTE: K× parameter overhead. The aux loss requires computing all K branch
    outputs even if we use top-k routing for the main path.
    """
    def __init__(self, config, n_branches=4, aux_weight=0.01):
        super().__init__()
        self.n_branches = n_branches
        self.aux_weight = aux_weight
        D = config.n_embd
        H = 4 * D
        # K full-rank MLP branches
        self.branches_fc = nn.ModuleList([
            Linear(D, H, bias=False) for _ in range(n_branches)])
        self.branches_proj = nn.ModuleList([
            Linear(H, D, bias=False) for _ in range(n_branches)])
        # Learned router (receives grad only through aux loss)
        self.router = nn.Sequential(
            Linear(D, D // 4, bias=False),
            nn.GELU(),
            Linear(D // 4, n_branches, bias=False),
        )
        # Zero-init final router layer for uniform routing at start
        nn.init.zeros_(self.router[-1].weight)
        # Cache for aux loss computation and diagnostics
        self._last_branch_outputs = None  # set during forward
        self._last_router_logits = None
        self._last_routing_entropy = 0.0

    def forward(self, x):
        B, T, D = x.shape
        dtype = x.dtype

        # Router: content-dependent but gradient-isolated from main loss
        # x.detach() stops main-loss gradient from reaching router
        router_logits = self.router(x.detach().to(dtype))  # (B, T, K)
        self._last_router_logits = router_logits  # save for aux loss
        # .detach() on output stops router gradient from reaching branch weights
        router_weights = F.softmax(router_logits.float(), dim=-1).detach().to(dtype)  # (B, T, K)

        # Cache for diagnostics
        with torch.no_grad():
            rw_float = router_weights.float()
            log_rw = torch.log(rw_float.clamp(min=1e-8))
            self._last_routing_entropy = -(rw_float * log_rw).sum(dim=-1).mean().item()

        # Compute all branches (needed for aux loss)
        branch_outputs = []
        for k in range(self.n_branches):
            h = self.branches_fc[k](x)
            h = F.relu(h).square()
            h = self.branches_proj[k](h)
            branch_outputs.append(h)

        # Save for aux loss (detached from main graph)
        self._last_branch_outputs = [bo.detach() for bo in branch_outputs]

        # Main output: weighted sum of branches
        y = torch.zeros(B, T, D, device=x.device, dtype=dtype)
        for k in range(self.n_branches):
            y = y + router_weights[..., k:k+1] * branch_outputs[k]

        return y

    def compute_aux_loss(self):
        """Compute auxiliary routing loss.

        Uses the mean of all branch outputs as the target. Trains the router
        to predict which branch is closest to this consensus output, which
        encourages branch specialization.
        """
        if self._last_branch_outputs is None or self._last_router_logits is None:
            return torch.tensor(0.0)
        # Target = mean of all branch outputs (consensus)
        target = torch.stack(self._last_branch_outputs, dim=0).mean(dim=0)  # (B, T, D)
        errors = []
        for bo in self._last_branch_outputs:
            err = ((bo - target) ** 2).mean(dim=-1)  # (B, T)
            errors.append(err)
        errors = torch.stack(errors, dim=-1)  # (B, T, K)
        best_branch = errors.argmin(dim=-1)  # (B, T)
        logits = self._last_router_logits  # (B, T, K) — has grad to router params
        aux_loss = F.cross_entropy(
            logits.reshape(-1, self.n_branches),
            best_branch.reshape(-1),
        )
        return self.aux_weight * aux_loss


class HashRoutedMLP(nn.Module):
    """20A: Hash-Routed Column Selection (HRCS).

    Stores a fat weight matrix W ∈ ℝ^{D_stored × D_out} but at each token
    position selects only D_active columns based on a DETERMINISTIC hash
    of the input. No learned gate — routing is a frozen, content-sensitive
    hash function applied to each token's feature vector.

    The hash function is: hash(x) = top-D_active indices of |x @ H_frozen|
    where H_frozen is a random frozen projection. This gives deterministic,
    input-dependent column selection with no gradient through the hash.

    Avoids FM1 (frozen hash), FM2 (content-based, not norm-based), FM3 (full columns).
    Risk: fat W must live in VRAM. The gather may not be compile-friendly.
    """
    def __init__(self, config, scale_factor=4):
        super().__init__()
        D = config.n_embd
        D_stored = D * scale_factor
        D_out = D
        # Fat fc weight: D → D_stored (stores scale_factor × more columns)
        self.c_fc = Linear(D, D_stored, bias=False)
        # Projection back: D_stored → D_out (only top-D_active columns used)
        self.c_proj = Linear(D_stored, D_out, bias=False)
        # Frozen hash projection: (D, D_stored) for column scoring
        self.register_buffer('hash_proj',
            torch.randn(D, D_stored) / (D ** 0.5))
        self.D_active = 4 * D  # active columns = same as standard MLP hidden dim
        self.D_stored = D_stored
        # Diagnostics
        self._last_routing_entropy = 0.0

    def forward(self, x):
        B, T, D = x.shape
        dtype = x.dtype
        # Full fc projection
        h = self.c_fc(x)  # (B, T, D_stored)
        # Hash-based column selection: score each hidden dim
        scores = (x.float() @ self.hash_proj.float()).abs()  # (B, T, D_stored)
        # Select top-D_active columns
        _, top_indices = scores.topk(self.D_active, dim=-1)  # (B, T, D_active)
        # Mask: zero out non-selected columns
        mask = torch.zeros_like(h)
        mask.scatter_(-1, top_indices, 1.0)
        h = h * mask  # (B, T, D_stored) — only D_active columns are non-zero
        h = F.relu(h).square()
        y = self.c_proj(h)  # (B, T, D_out)

        # Diagnostics: how spread out is the column selection?
        with torch.no_grad():
            # Compute normalized selection frequency across tokens
            sel_freq = mask.float().mean(dim=(0, 1))  # (D_stored,) — fraction of tokens selecting each col
            sel_freq = sel_freq / (sel_freq.sum() + 1e-8)
            log_sf = torch.log(sel_freq.clamp(min=1e-8))
            self._last_routing_entropy = -(sel_freq * log_sf).sum().item()

        return y


class LSHRoutedMLP(nn.Module):
    """20B: Locality-Sensitive Weight Routing (LSWR).

    Like 20A but uses proper LSH: similar inputs hash to similar column subsets.
    Frozen random hyperplanes partition input space into 2^n_planes buckets.
    Each bucket maps to a fixed set of active columns (precomputed at init).

    alpha = sign(x @ H)  → binary hash → bucket_id
    columns = bucket_to_columns[bucket_id]
    y = MLP with only those columns active

    Avoids FM1 (frozen hash), FM2 (content-based). LSH guarantees locality
    preservation: nearby inputs in feature space → similar column selections.
    """
    def __init__(self, config, scale_factor=4, n_planes=8):
        super().__init__()
        D = config.n_embd
        D_stored = D * scale_factor
        self.D_active = 4 * D
        self.D_stored = D_stored
        self.n_planes = n_planes
        self.n_buckets = 2 ** n_planes
        # Fat MLP layers
        self.c_fc = Linear(D, D_stored, bias=False)
        self.c_proj = Linear(D_stored, D, bias=False)
        # Frozen LSH hyperplanes: (n_planes, D) — random unit vectors
        planes = torch.randn(n_planes, D)
        planes = planes / planes.norm(dim=-1, keepdim=True)
        self.register_buffer('hyperplanes', planes)
        # Precompute bucket → column mapping (frozen)
        # Each bucket gets a random subset of D_active columns from D_stored
        bucket_map = torch.zeros(self.n_buckets, D_stored)
        for b in range(self.n_buckets):
            perm = torch.randperm(D_stored)[:self.D_active]
            bucket_map[b, perm] = 1.0
        self.register_buffer('bucket_map', bucket_map)  # (n_buckets, D_stored)
        # Diagnostics
        self._last_routing_entropy = 0.0
        self._last_bucket_usage = None

    def forward(self, x):
        B, T, D = x.shape
        dtype = x.dtype
        # LSH: compute binary hash
        hash_bits = (x.float() @ self.hyperplanes.float().T) > 0  # (B, T, n_planes) bool
        # Convert to bucket IDs
        powers = (2 ** torch.arange(self.n_planes, device=x.device)).float()
        bucket_ids = (hash_bits.float() @ powers).long()  # (B, T) integers in [0, n_buckets)
        # Look up column masks for each bucket
        mask = self.bucket_map[bucket_ids]  # (B, T, D_stored)
        mask = mask.to(dtype)
        # MLP with masked columns
        h = self.c_fc(x)  # (B, T, D_stored)
        h = h * mask
        h = F.relu(h).square()
        y = self.c_proj(h)

        # Diagnostics
        with torch.no_grad():
            # Bucket usage entropy
            bucket_counts = torch.zeros(self.n_buckets, device=x.device)
            bucket_counts.scatter_add_(0, bucket_ids.reshape(-1),
                                       torch.ones(B * T, device=x.device))
            bucket_prob = bucket_counts / (B * T)
            log_bp = torch.log(bucket_prob.clamp(min=1e-8))
            self._last_routing_entropy = -(bucket_prob * log_bp).sum().item()
            self._last_bucket_usage = bucket_prob

        return y


class NoiseCEA_MLP(nn.Module):
    """20H: Noise-Contrastive Expert Assignment (NCEA).

    A shared base MLP + K learned perturbation directions. Extends 19J
    (weight noise works) with STRUCTURED noise: instead of random
    perturbation, route tokens to the perturbation direction that is
    most beneficial, trained via reconstruction comparison (not main loss).

    W_k = W_base + eps * delta_k
    Router predicts which perturbation helps most (aux loss, not main loss).
    Main forward uses detached routing weights.

    Avoids FM1 (aux-loss router only). Extends the proven 19J signal.
    Risk: computing all K outputs at training time for aux loss is expensive.
    """
    def __init__(self, config, n_branches=4, eps=0.1):
        super().__init__()
        self.n_branches = n_branches
        self.eps = eps
        D = config.n_embd
        H = 4 * D
        # Shared base MLP
        self.c_fc = Linear(D, H, bias=False)
        self.c_proj = Linear(H, D, bias=False)
        # K perturbation directions for fc (small init)
        self.delta_fc = nn.ParameterList([
            nn.Parameter(torch.randn(H, D) * 0.01) for _ in range(n_branches)])
        # K perturbation directions for proj (small init)
        self.delta_proj = nn.ParameterList([
            nn.Parameter(torch.randn(D, H) * 0.01) for _ in range(n_branches)])
        # Router: predicts best perturbation (trained via aux loss only)
        self.router = Linear(D, n_branches, bias=False)
        nn.init.zeros_(self.router.weight)
        # Diagnostics
        self._last_routing_entropy = 0.0
        self._last_branch_outputs = None
        self._last_router_logits = None

    def forward(self, x):
        B, T, D = x.shape
        dtype = x.dtype
        # Router: content-dependent, gradient-isolated from main loss
        router_logits = self.router(x.detach())  # (B, T, K)
        self._last_router_logits = router_logits
        router_weights = F.softmax(router_logits.float(), dim=-1).detach().to(dtype)

        # Diagnostics
        with torch.no_grad():
            rw = router_weights.float()
            log_rw = torch.log(rw.clamp(min=1e-8))
            self._last_routing_entropy = -(rw * log_rw).sum(dim=-1).mean().item()

        # Compute all K perturbed outputs (needed for aux loss)
        branch_outputs = []
        base_fc_w = self.c_fc.weight  # (H, D)
        base_proj_w = self.c_proj.weight  # (D, H)
        for k in range(self.n_branches):
            # Cast deltas to match weight dtype to avoid bf16/float32 mismatch
            fc_w = base_fc_w + self.eps * self.delta_fc[k].to(base_fc_w.dtype)
            proj_w = base_proj_w + self.eps * self.delta_proj[k].to(base_proj_w.dtype)
            h = F.linear(x, fc_w)
            h = F.relu(h).square()
            out = F.linear(h, proj_w)
            branch_outputs.append(out)

        self._last_branch_outputs = [bo.detach() for bo in branch_outputs]

        # Weighted sum
        y = torch.zeros(B, T, D, device=x.device, dtype=dtype)
        for k in range(self.n_branches):
            y = y + router_weights[..., k:k+1] * branch_outputs[k]
        return y

    def compute_aux_loss(self):
        """Train router to predict which perturbation direction is best."""
        if self._last_branch_outputs is None or self._last_router_logits is None:
            return torch.tensor(0.0)
        # Target = mean of all branch outputs (consensus)
        target = torch.stack(self._last_branch_outputs, dim=0).mean(dim=0)
        errors = []
        for bo in self._last_branch_outputs:
            err = ((bo - target) ** 2).mean(dim=-1)
            errors.append(err)
        errors = torch.stack(errors, dim=-1)
        best_branch = errors.argmin(dim=-1)
        logits = self._last_router_logits
        return 0.01 * F.cross_entropy(
            logits.reshape(-1, self.n_branches), best_branch.reshape(-1))


class AttnDerivedMLP(nn.Module):
    """20I: Attention-Derived Weight Interpolation (ADWI).

    Uses the attention mechanism's per-head output magnitudes as a ROUTING
    signal for FFN expert selection. No new router — reuses the existing
    attention computation. Each head's contribution magnitude determines
    which narrow expert to weight.

    head_magnitudes = [||attn_head_k||² for k in range(H)]
    alpha = softmax(head_magnitudes)
    y = Σ_h alpha_h * Expert_h(x)

    Avoids FM4 (zero routing overhead — reuses attention). Content-dependent
    routing derived from an already-trained mechanism.

    Implementation: This module receives `attn_head_norms` from the
    CausalSelfAttention layer in Block.forward.
    """
    def __init__(self, config):
        super().__init__()
        n_heads = config.n_head
        D = config.n_embd
        expert_hidden = 4 * D // n_heads
        self.n_experts = n_heads
        assert expert_hidden * n_heads == 4 * D, (
            f"4*D={4*D} must be divisible by n_heads={n_heads}")
        # H narrow expert MLPs (one per attention head)
        self.experts_fc = nn.ModuleList([
            Linear(D, expert_hidden, bias=False) for _ in range(n_heads)])
        self.experts_proj = nn.ModuleList([
            Linear(expert_hidden, D, bias=False) for _ in range(n_heads)])
        # Diagnostics
        self._last_routing_entropy = 0.0
        self._attn_head_norms = None

    def set_attn_head_norms(self, head_norms):
        """Called by Block.forward to pass attention head magnitudes."""
        self._attn_head_norms = head_norms

    def forward(self, x):
        B, T, D = x.shape
        dtype = x.dtype
        if self._attn_head_norms is not None:
            # Use attention head magnitudes as routing signal
            weights = F.softmax(self._attn_head_norms.float(), dim=-1).to(dtype)  # (B, T, H)
        else:
            # Fallback: uniform routing
            weights = torch.ones(B, T, self.n_experts, device=x.device, dtype=dtype) / self.n_experts

        # Diagnostics
        with torch.no_grad():
            w_float = weights.float()
            log_w = torch.log(w_float.clamp(min=1e-8))
            self._last_routing_entropy = -(w_float * log_w).sum(dim=-1).mean().item()

        # Compute all experts, weight by routing
        y = torch.zeros(B, T, D, device=x.device, dtype=dtype)
        for k in range(self.n_experts):
            h = self.experts_fc[k](x)
            h = F.relu(h).square()
            h = self.experts_proj[k](h)
            y = y + weights[..., k:k+1] * h
        return y


class PWU_MLP(nn.Module):
    """20E: Progressive Weight Unfreezing (PWU).

    Phase 2 architecture: takes a pre-trained MLP's weights, forks them into K
    branches with random perturbations, and trains a router.

    Three training phases controlled by p20_pwu_phase:
      Phase 1: Normal pretraining (use standard MLP — this class is not used)
      Phase 2: Freeze forked branches, train router only (router_only=True)
      Phase 3: Unfreeze everything, joint training with reduced branch LR

    from_pretrained_mlp() creates this from a trained standard MLP.
    """
    def __init__(self, config, n_branches=4, router_only=False):
        super().__init__()
        self.n_branches = n_branches
        self.router_only = router_only
        D = config.n_embd
        H = 4 * D
        # K branches (forked from pretrained weights)
        self.branches_fc = nn.ModuleList([
            Linear(D, H, bias=False) for _ in range(n_branches)])
        self.branches_proj = nn.ModuleList([
            Linear(H, D, bias=False) for _ in range(n_branches)])
        # Content-dependent router
        self.router = Linear(D, n_branches, bias=False)
        nn.init.zeros_(self.router.weight)
        # Diagnostics
        self._last_routing_entropy = 0.0

    @classmethod
    def from_pretrained_mlp(cls, config, pretrained_mlp, n_branches=4, router_only=True, perturb_scale=0.01):
        """Create PWU_MLP by forking a pre-trained MLP's weights into K branches."""
        module = cls(config, n_branches=n_branches, router_only=router_only)
        fc_w = pretrained_mlp.c_fc.weight.data.clone()
        proj_w = pretrained_mlp.c_proj.weight.data.clone()
        for k in range(n_branches):
            module.branches_fc[k].weight.data.copy_(fc_w + perturb_scale * torch.randn_like(fc_w))
            module.branches_proj[k].weight.data.copy_(proj_w + perturb_scale * torch.randn_like(proj_w))
            if router_only:
                # Phase 2: freeze branches, only train router
                module.branches_fc[k].weight.requires_grad_(False)
                module.branches_proj[k].weight.requires_grad_(False)
        return module

    def forward(self, x):
        B, T, D = x.shape
        dtype = x.dtype
        # Router
        router_logits = self.router(x)  # (B, T, K)
        weights = F.softmax(router_logits.float(), dim=-1).to(dtype)

        with torch.no_grad():
            w = weights.float()
            log_w = torch.log(w.clamp(min=1e-8))
            self._last_routing_entropy = -(w * log_w).sum(dim=-1).mean().item()

        y = torch.zeros(B, T, D, device=x.device, dtype=dtype)
        for k in range(self.n_branches):
            h = self.branches_fc[k](x)
            h = F.relu(h).square()
            h = self.branches_proj[k](h)
            y = y + weights[..., k:k+1] * h
        return y


class FSVD_MLP(nn.Module):
    """20G: Frozen-SVD Weight Subspace Traversal.

    Phase 2 architecture: decomposes pre-trained MLP weights via SVD,
    freezes U and V (directional structure), and lets a content-dependent
    frozen gate SELECT which singular values to amplify/suppress per token.

    Unlike 19G (which failed), U/V come from a PRE-TRAINED weight (not zero-init),
    and the σ modulation is content-dependent via a frozen random projection.

    W_eff = U @ diag(sigma * gate(x)) @ V^T per token
    gate = sigmoid(x @ R_frozen)  — frozen, no gradient

    Same total params as pretrained MLP (sigma replaces the original weight).
    """
    def __init__(self, config):
        super().__init__()
        D = config.n_embd
        H = 4 * D
        self.D = D
        self.H = H
        # SVD components (set by from_pretrained_mlp)
        # fc: W = U_fc @ diag(sigma_fc) @ V_fc^T, shape (H, D) → U:(H,r), sigma:(r,), V:(D,r)
        self.register_buffer('U_fc', torch.zeros(H, D))   # placeholder, set by from_pretrained
        self.register_buffer('V_fc', torch.zeros(D, D))
        self.sigma_fc = nn.Parameter(torch.ones(D))        # learnable singular values
        # proj: W = U_proj @ diag(sigma_proj) @ V_proj^T, shape (D, H)
        self.register_buffer('U_proj', torch.zeros(D, D))
        self.register_buffer('V_proj', torch.zeros(H, D))
        self.sigma_proj = nn.Parameter(torch.ones(D))
        # Frozen content-dependent gate for sigma modulation
        self.register_buffer('gate_proj_fc', torch.randn(D, D) / (D ** 0.5))
        self.register_buffer('gate_proj_proj', torch.randn(D, D) / (D ** 0.5))
        # Diagnostics
        self._last_routing_entropy = 0.0
        self._last_sigma_sparsity = 0.0

    @classmethod
    def from_pretrained_mlp(cls, config, pretrained_mlp):
        """Create FSVD_MLP by decomposing a pre-trained MLP's weights via SVD."""
        module = cls(config)
        D = config.n_embd
        # SVD of fc weight (H, D)
        fc_w = pretrained_mlp.c_fc.weight.data.float()
        U_fc, S_fc, Vh_fc = torch.linalg.svd(fc_w, full_matrices=False)
        # Keep rank = min(H, D) = D components
        r = min(fc_w.shape)
        module.U_fc = U_fc[:, :r]         # (H, r)
        module.sigma_fc = nn.Parameter(S_fc[:r])  # (r,)
        module.V_fc = Vh_fc[:r, :].T      # (D, r)  — note: Vh transposed

        # SVD of proj weight (D, H)
        proj_w = pretrained_mlp.c_proj.weight.data.float()
        U_proj, S_proj, Vh_proj = torch.linalg.svd(proj_w, full_matrices=False)
        r2 = min(proj_w.shape)
        module.U_proj = U_proj[:, :r2]
        module.sigma_proj = nn.Parameter(S_proj[:r2])
        module.V_proj = Vh_proj[:r2, :].T

        return module

    def forward(self, x):
        B, T, D = x.shape
        dtype = x.dtype

        # Content-dependent sigma gating (frozen — no gradient through gate)
        gate_fc = torch.sigmoid(x.float() @ self.gate_proj_fc.float())     # (B, T, D)
        gate_proj = torch.sigmoid(x.float() @ self.gate_proj_proj.float()) # (B, T, D)

        # Modulated sigma
        sigma_fc_eff = self.sigma_fc.float() * gate_fc      # (B, T, r)
        sigma_proj_eff = self.sigma_proj.float() * gate_proj # (B, T, r)

        # Diagnostics
        with torch.no_grad():
            # Sparsity: what fraction of sigma values are effectively zero?
            self._last_sigma_sparsity = (gate_fc.mean(dim=(0, 1)) < 0.1).float().mean().item()
            # Entropy of gate values
            g = gate_fc.mean(dim=(0, 1)).clamp(1e-8, 1 - 1e-8)
            self._last_routing_entropy = -(g * g.log() + (1-g) * (1-g).log()).mean().item()

        # W_eff_fc = U_fc @ diag(sigma_fc_eff) @ V_fc^T per token
        # h = x @ V_fc → scale by sigma → multiply by U_fc^T
        h = x.float() @ self.V_fc.float()           # (B, T, r)
        h = h * sigma_fc_eff                         # (B, T, r) — per-token sigma gating
        h = h @ self.U_fc.float().T                  # (B, T, H)
        h = F.relu(h).square()

        # W_eff_proj = U_proj @ diag(sigma_proj_eff) @ V_proj^T per token
        y = h @ self.V_proj.float()                  # (B, T, r)
        y = y * sigma_proj_eff
        y = y @ self.U_proj.float().T                # (B, T, D)
        return y.to(dtype)


class WBFC_MLP(nn.Module):
    """20J: Weight Bank with Frozen Clustering (WBFC).

    Phase 2 architecture: clusters a pre-trained MLP's hidden dimensions
    into K groups using K-means, then routes tokens to top-M clusters
    via a frozen content projection.

    Online: for each token, select top-M clusters (M < K)
    Only compute with the hidden dims belonging to those clusters.
    This is the most direct realization of the user's vision:
    a large weight bank where context selects a subspace.

    Saves FLOPs at inference: only M/K of the hidden dims are computed.
    """
    def __init__(self, config, n_clusters=8, n_active=2):
        super().__init__()
        D = config.n_embd
        H = 4 * D
        self.n_clusters = n_clusters
        self.n_active = n_active
        self.D = D
        self.H = H
        # Standard MLP weights (full — we mask at runtime)
        self.c_fc = Linear(D, H, bias=False)
        self.c_proj = Linear(H, D, bias=False)
        # Cluster assignments: which hidden dim belongs to which cluster
        # Shape: (H,) — integer cluster ID for each hidden dim
        self.register_buffer('cluster_ids', torch.zeros(H, dtype=torch.long))
        # Cluster centroids for routing: (K, D)
        self.register_buffer('centroids', torch.randn(n_clusters, D))
        # Frozen content projection for cluster affinity
        self.register_buffer('route_proj', torch.randn(D, n_clusters) / (D ** 0.5))
        # Diagnostics
        self._last_routing_entropy = 0.0

    @classmethod
    def from_pretrained_mlp(cls, config, pretrained_mlp, n_clusters=8, n_active=2):
        """Create WBFC_MLP by clustering a pre-trained MLP's hidden dimensions."""
        D = config.n_embd
        H = 4 * D
        module = cls(config, n_clusters=n_clusters, n_active=n_active)

        # Copy pretrained weights
        module.c_fc.weight.data.copy_(pretrained_mlp.c_fc.weight.data)
        module.c_proj.weight.data.copy_(pretrained_mlp.c_proj.weight.data)

        # K-means clustering on fc weight columns (each hidden dim is a D-vector)
        fc_w = pretrained_mlp.c_fc.weight.data.float()  # (H, D)
        # Simple K-means (10 iterations)
        centroids = fc_w[torch.randperm(H)[:n_clusters]]  # random init
        for _ in range(10):
            dists = torch.cdist(fc_w, centroids)  # (H, K)
            assignments = dists.argmin(dim=-1)      # (H,)
            for k in range(n_clusters):
                mask = (assignments == k)
                if mask.sum() > 0:
                    centroids[k] = fc_w[mask].mean(dim=0)
        module.cluster_ids.copy_(assignments)
        module.centroids.copy_(centroids)

        return module

    def forward(self, x):
        B, T, D = x.shape
        dtype = x.dtype

        # Cluster affinity: which clusters are relevant for this token?
        affinity = x.float() @ self.route_proj.float()  # (B, T, K)
        _, top_clusters = affinity.topk(self.n_active, dim=-1)  # (B, T, M)

        # Build mask: which hidden dims to activate
        # cluster_ids: (H,) — for each hidden dim, its cluster
        # Expand and compare
        mask = torch.zeros(B, T, self.H, device=x.device, dtype=dtype)
        for m in range(self.n_active):
            cluster_k = top_clusters[..., m]  # (B, T)
            # For each (b, t), activate all hidden dims belonging to cluster_k[b,t]
            cluster_k_expanded = cluster_k.unsqueeze(-1).expand(-1, -1, self.H)  # (B, T, H)
            cluster_ids_expanded = self.cluster_ids.unsqueeze(0).unsqueeze(0).expand(B, T, -1)  # (B, T, H)
            mask = mask + (cluster_ids_expanded == cluster_k_expanded).to(dtype)
        mask = mask.clamp(max=1.0)

        # MLP with masked hidden dims
        h = self.c_fc(x)  # (B, T, H)
        h = h * mask
        h = F.relu(h).square()
        y = self.c_proj(h)

        # Diagnostics
        with torch.no_grad():
            # How many clusters are used on average?
            active_frac = mask.float().mean(dim=-1).mean()  # fraction of active dims
            # Cluster selection entropy
            cluster_probs = affinity.float().softmax(dim=-1).mean(dim=(0, 1))  # (K,)
            log_cp = torch.log(cluster_probs.clamp(min=1e-8))
            self._last_routing_entropy = -(cluster_probs * log_cp).sum().item()

        return y


# ─────────────────────────────────────────────────────────────────────────────
# Phase 22: Full-size MoE Linear (for attention projections)
# ─────────────────────────────────────────────────────────────────────────────

_moe_full_counter = 0

class MoEFullLinear(nn.Module):
    """Phase 22: Full-size MoE drop-in replacement for nn.Linear.

    Each of the K experts is a full (D_out, D_in) weight matrix — no bottleneck.
    Routing can be provided externally (for shared Q/K/V/Proj routing in attention)
    or computed internally.

    Per-sequence routing (sequence_route=True): computes routing from sequence-mean,
    builds a single effective weight W_eff = Σ_k w_k W_k, then applies as one bmm.
    This preserves Q-K dot-product coherence because all tokens share the same
    projection matrix.
    """
    def __init__(self, D_in, D_out, K, topk=0, learned_route=False,
                 has_router=True, sequence_route=True):
        super().__init__()
        global _moe_full_counter
        self.D_in = D_in
        self.D_out = D_out
        self.K = K
        self.topk = topk if topk > 0 else K  # 0 = soft (all experts)
        self.sequence_route = sequence_route
        # K full-size expert weight matrices: (K, D_out, D_in)
        self.expert_weights = nn.Parameter(
            torch.randn(K, D_out, D_in) * (D_in ** -0.5))
        # Optional internal router
        self.has_router = has_router
        if has_router:
            gen = torch.Generator().manual_seed(137 + _moe_full_counter)
            _moe_full_counter += 1
            route_init = torch.randn(D_in, K, generator=gen) / (D_in ** 0.5)
            if learned_route:
                self.route_proj = nn.Parameter(route_init)
            else:
                self.register_buffer('route_proj', route_init)
        # Diagnostics
        self.register_buffer('_entropy_buf', torch.zeros(1), persistent=False)

    @property
    def weight(self):
        """Mean expert weight for init_weights / diagnostics compatibility."""
        return self.expert_weights.mean(dim=0)  # (D_out, D_in)

    def gate_parameters(self):
        if self.has_router and isinstance(getattr(self, 'route_proj', None), nn.Parameter):
            yield self.route_proj

    def non_gate_parameters(self):
        yield self.expert_weights

    def _compute_routing(self, x):
        """Compute routing weights from input using internal router."""
        logits = torch.matmul(x.float(), self.route_proj.float())  # (B, T, K)
        if self.sequence_route:
            logits = logits.mean(dim=1, keepdim=True)  # (B, 1, K)
        if self.topk < self.K:
            _, topk_idx = logits.topk(self.topk, dim=-1)
            mask = torch.zeros_like(logits).scatter_(-1, topk_idx, 1.0)
            logits = logits.masked_fill(mask == 0, float('-inf'))
        return F.softmax(logits, dim=-1)

    def forward(self, x, route_weights=None):
        """Forward pass. If route_weights is provided, uses those instead of internal router."""
        orig_shape = x.shape
        if x.dim() == 2:
            x = x.unsqueeze(0)
        B, T, D = x.shape
        dtype = x.dtype

        # Get routing weights
        if route_weights is None:
            assert self.has_router, "MoEFullLinear: no router and no external route_weights"
            route_weights = self._compute_routing(x).to(dtype)
        else:
            route_weights = route_weights.to(dtype)

        # Entropy diagnostic
        with torch.no_grad():
            probs = route_weights.float()
            ent = -(probs * (probs + 1e-8).log()).sum(-1).mean()
            self._entropy_buf.copy_(ent)

        # Compute output
        experts = self.expert_weights.to(dtype)  # (K, D_out, D_in)
        if route_weights.shape[1] == 1:
            # Per-sequence: build effective weight, single bmm (very efficient)
            w = route_weights.squeeze(1)  # (B, K)
            eff = torch.einsum('bk,koi->boi', w, experts)  # (B, D_out, D_in)
            y = torch.bmm(x, eff.transpose(1, 2))  # (B, T, D_out)
        else:
            # Per-token: full K-way computation
            all_out = torch.einsum('btd,kod->btko', x, experts)  # (B, T, K, D_out)
            y = (all_out * route_weights.unsqueeze(-1)).sum(dim=2)  # (B, T, D_out)

        if len(orig_shape) == 2:
            y = y.squeeze(0)
        return y


# ─────────────────────────────────────────────────────────────────────────────
# Phase 21: Pervasive Expert Routing (PER) — MoE-ify every Linear layer
# ─────────────────────────────────────────────────────────────────────────────

# Global counter for assigning unique seeds to each MoELinear instance
_moe_linear_counter = 0

class MoELinear(nn.Module):
    """Phase 21: Drop-in replacement for Linear(D_in, D_out) with K expert sub-layers.

    Each expert is a narrow Linear(D_in, D_out) using a hidden bottleneck:
        expert_k(x) = A_k @ (B_k @ x)   where B_k: (r, D_in), A_k: (D_out, r), r = D_in // K

    All expert weights are stored as single stacked tensors. The for-loop over K
    experts is unrolled by torch.compile (K is a Python int constant). No graph
    breaks, no ParameterList, fully compile + FP8 friendly.

    Routing: frozen content projection R (different per instance via unique seed).
    """
    def __init__(self, D_in, D_out, K, topk=0, learned_route=False, bias=False):
        super().__init__()
        global _moe_linear_counter
        self.D_in = D_in
        self.D_out = D_out
        self.K = K
        self.topk = topk if topk > 0 else K  # 0 = use all (soft routing)
        # Bottleneck rank per expert
        self.rank = max(1, D_in // K)
        # Stacked expert weights: B (K, rank, D_in), A (K, D_out, rank)
        self.experts_B = nn.Parameter(
            torch.randn(K, self.rank, D_in) * (D_in ** -0.5))
        self.experts_A = nn.Parameter(
            torch.randn(K, D_out, self.rank) * (self.rank ** -0.5))
        # Per-instance frozen/learned routing projection
        gen = torch.Generator().manual_seed(42 + _moe_linear_counter)
        _moe_linear_counter += 1
        route_init = torch.randn(D_in, K, generator=gen) / (D_in ** 0.5)
        if learned_route:
            self.route_proj = nn.Parameter(route_init)
        else:
            self.register_buffer('route_proj', route_init)
        self.learned_route = learned_route
        # Diagnostics — use register_buffer + .copy_() so compile can trace
        # buffer mutations (like BatchNorm's running_mean). No Python attribute
        # assignment = no graph break.
        self.register_buffer('_entropy_buf', torch.zeros(1), persistent=False)

    @property
    def _last_routing_entropy(self):
        """Lazy .item() — only called outside compile when diagnostics are read."""
        return self._entropy_buf.item()

    def forward(self, x):
        orig_shape = x.shape
        if x.dim() == 2:
            x = x.unsqueeze(0)
        B, T, D = x.shape
        dtype = x.dtype

        # Routing scores — float for stable softmax, back to dtype
        logits = x.float() @ self.route_proj.float()  # (B, T, K)
        weights = F.softmax(logits, dim=-1).to(dtype)  # (B, T, K)

        # Routing entropy diagnostic — buffer mutation is compile-safe
        with torch.no_grad():
            w_f = weights.float()
            entropy = -(w_f * torch.log(w_f.clamp(min=1e-8))).sum(dim=-1).mean()
            self._entropy_buf.copy_(entropy.detach())

        # Top-k sparsification (static branch — self.topk/self.K are Python ints)
        if self.topk < self.K:
            _, topk_idx = weights.topk(self.topk, dim=-1)
            mask = torch.zeros_like(weights)
            mask.scatter_(-1, topk_idx, 1.0)
            weights = weights * mask
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        # Sequential expert computation — K is a Python int so the loop is
        # unrolled by torch.compile. Each iteration only materializes (B, T, r)
        # and (B, T, D_out), avoiding the (B, T, K, D_out) memory spike.
        y = torch.zeros(B, T, self.D_out, device=x.device, dtype=dtype)
        for k in range(self.K):
            h = F.linear(x, self.experts_B[k].to(dtype))    # (B, T, r)
            out = F.linear(h, self.experts_A[k].to(dtype))  # (B, T, D_out)
            y = y + weights[:, :, k:k+1] * out

        if len(orig_shape) == 2:
            y = y.squeeze(0)
        return y

    @property
    def weight(self):
        """Reconstruct effective weight for compatibility (e.g., init_weights).
        Returns mean of A_k @ B_k — shape (D_out, D_in).
        """
        # experts_A: (K, D_out, r), experts_B: (K, r, D_in)
        w = torch.einsum('kor,krd->kod', self.experts_A, self.experts_B)  # (K, D_out, D_in)
        return w.mean(dim=0)  # (D_out, D_in)


class MLP(nn.Module):
    def __init__(self, config, layer_idx=0):
        super().__init__()
        _mult = resolve_ffn_schedule(config, config.n_layer)[layer_idx]
        _hidden = max(1, int(round(config.n_embd * _mult)))
        _per_k = getattr(config, 'p21_per_experts', 0)
        if _per_k > 0:
            _per_topk = getattr(config, 'p21_per_topk', 0)
            _per_learned = bool(getattr(config, 'p21_per_learned', 0))
            self.c_fc = MoELinear(config.n_embd, 4 * config.n_embd, K=_per_k,
                                  topk=_per_topk, learned_route=_per_learned)
            self.c_proj = MoELinear(4 * config.n_embd, config.n_embd, K=_per_k,
                                    topk=_per_topk, learned_route=_per_learned)
        else:
            self.c_fc = Linear(config.n_embd, _hidden, bias=False)
            self.c_proj = Linear(_hidden, config.n_embd, bias=False)
        # 18I: Dynamic Activation (learned mix of ReLU², GELU, SiLU)
        self.dynamic_act = DynamicActivation() if getattr(config, 'p18_dynamic_activation', 0) else None
        # 18F: Per-Channel Scale after projection
        self.channel_scale = PerChannelScale(config.n_embd) if getattr(config, 'p18_per_channel_scale', 0) else None
        # 19G: Spectral Reparameterization
        srp_mode = getattr(config, 'p19_spectral_reparam', 0)
        self.srp_proj = SpectralReparamLinear(_hidden, config.n_embd) if srp_mode >= 1 else None
        self.srp_fc = SpectralReparamLinear(config.n_embd, _hidden) if srp_mode >= 2 else None
        # Phase 30: optional intermediate LayerNorm for dense+LN ablation
        self.ln_intermediate = nn.LayerNorm(_hidden) if getattr(config, 'dense_intermediate_ln', 0) else None

    def forward(self, x):
        x = self.c_fc(x)
        if self.ln_intermediate is not None:
            _ln_in_dtype = self.ln_intermediate.weight.dtype
            x = self.ln_intermediate(x.to(dtype=_ln_in_dtype)).to(dtype=x.dtype)
        if self.dynamic_act is not None:
            x = self.dynamic_act(x)
        else:
            x = F.relu(x).square()
        # 19G: If spectral reparam is on for c_proj, use it; else use standard linear
        if self.srp_proj is not None:
            x = self.srp_proj(x)
        else:
            x = self.c_proj(x)
        if self.channel_scale is not None:
            x = self.channel_scale(x)
        return x


class _DenseProj(nn.Module):
    """Plain dense projection that tolerates the research call signature.

    Used when cond_sites restricts ConditionedLinear to one half of the block:
    the other half must be the DENSE layer it is being compared against, not a
    RemixedLinear, or the ablation measures two things at once.
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.proj = Linear(in_features, out_features, bias=False)

    def forward(self, x, context_state=None, **_ignored):
        return self.proj(x)


# Measured on the dense d22 checkpoint by scripts/conditioning_headroom.py.
# reachable = per-layer demand x the fraction a linear router could predict from x;
# supply    = per-layer Jacobian dispersion the FFN's nonlinearity already provides.
# Both are participation ratios over 4096 tokens. All 22 layers had IDENTICAL width,
# so the spread is a property of the activations, not of the architecture.
_D22_REACHABLE = (40.1, 63.9, 48.9, 90.2, 91.9, 87.3, 97.1, 121.0, 144.2, 181.0, 86.1,
                  86.2, 136.0, 325.5, 503.8, 376.8, 170.6, 302.6, 354.7, 17.7, 39.6, 59.8)
_D22_SUPPLY    = (85.2, 22.4, 166.4, 101.4, 199.1, 172.2, 229.0, 170.4, 124.1, 170.2, 216.3,
                  164.5, 182.7, 111.6, 145.8, 129.4, 106.9, 86.4, 59.9, 45.4, 70.7, 28.1)

# Native per-depth reachable profiles, measured on actual checkpoints at each depth.
# These are the CORRECT profiles to use for FFN reallocation, not d22 interpolation.
# From headroom_results.log (FFN c_fc reachable demand column).
_NATIVE_REACHABLE = {
    8:  (232.5, 214.7, 208.6, 71.0, 115.4, 107.9, 233.8, 120.1),
    12: (513.4, 468.6, 231.1, 119.7, 224.3, 225.4, 289.5, 137.1, 283.1, 162.1, 283.0, 191.1),
    20: (53.3, 94.4, 208.2, 292.3, 287.1, 206.7, 223.9, 226.0, 277.4, 325.0,
         190.2, 136.5, 125.0, 166.7, 267.8, 208.8, 200.8, 101.7, 92.9, 175.4),
    22: (95.7, 251.8, 203.9, 293.5, 273.8, 182.3, 179.0, 210.5, 246.6, 368.6, 142.7,
         169.1, 236.5, 436.2, 719.4, 456.3, 207.0, 323.4, 419.4, 28.6, 44.8, 50.8),
}


def _interp_profile(profile, n_layer):
    """Resample a 22-layer profile onto n_layer by relative depth."""
    m = len(profile)
    if n_layer == m:
        return list(profile)
    out = []
    for i in range(n_layer):
        pos = (i / max(1, n_layer - 1)) * (m - 1)
        lo = int(pos); hi = min(lo + 1, m - 1); f = pos - lo
        out.append(profile[lo] * (1 - f) + profile[hi] * f)
    return out


def eet_active_fractions(n_layer, min_exit_layer, target_frac, schedule='bell'):
    """Fraction of tokens still active when each block runs, under EET's capacity schedule.

    Mirrors the computation inside EarlyExitGPT.forward's compute-skip path. It lives here
    because the FFN widths have to be chosen at construction time, before any forward has
    run. tests/test_eet_width.py asserts the two stay in agreement.
    """
    routing_layers = list(range(min_exit_layer, n_layer - 1))
    n_rl = len(routing_layers)
    if n_rl <= 0:
        return [1.0] * n_layer
    if schedule == 'uniform':
        fracs = [(1.0 - target_frac) / n_rl] * n_rl
    elif schedule == 'linear':
        w = [k + 1 for k in range(n_rl)]
        fracs = [x / sum(w) * (1.0 - target_frac) for x in w]
    elif schedule == 'geometric':
        per = 1.0 - target_frac ** (1.0 / n_rl)
        fracs = [per] * n_rl
    else:  # bell
        mid = (n_rl - 1) / 2.0
        sigma = max(1.0, n_rl / 4.0)
        w = [math.exp(-((k - mid) / sigma) ** 2) for k in range(n_rl)]
        fracs = [x / sum(w) * (1.0 - target_frac) for x in w]
    survivor, caps = 1.0, []
    for f in fracs:
        survivor -= f
        caps.append(max(1e-6, survivor))
    per_block, rl, cur = [], 0, 1.0
    for i in range(n_layer):
        per_block.append(cur)
        if i in routing_layers and rl < n_rl:
            cur = caps[rl]
            rl += 1
    return per_block


def resolve_ffn_schedule(config, n_layer):
    """Per-layer FFN width multiplier. 0 at a layer means NO FFN in that block.

    p34_ffn_schedule accepts a comma-separated list of multipliers, or a preset:

      ''            uniform p34_ffn_mult, the standard transformer
      'measured'    reallocate at FIXED total parameters, proportional to the
                    measured reachable demand. FLOP-neutral, so any gain is free.
      'shrink'      cut only the over-provisioned layers to their reachable level
                    and leave the rest. Removes compute, which is the direction
                    that actually moves the efficiency metric.
      'last'        FFN only in the final block
      'every2' / 'every4'          FFN every 2nd / 4th block, same width
      'every2_iso' / 'every4_iso'  same, width scaled up to hold parameters fixed
    """
    base = float(getattr(config, 'p34_ffn_mult', 4.0))
    spec = str(getattr(config, 'p34_ffn_schedule', '') or '').strip()

    # Tier 2 idea 5: widen the FFN where tokens are scarce. Takes precedence over the
    # p34 presets, which were fitted on a dense stack where every layer sees every token.
    _wp = float(getattr(config, 'eet_width_power', 0.0))
    if _wp > 0.0 and getattr(config, 'use_eet', False):
        a = eet_active_fractions(n_layer,
                                 int(getattr(config, 'eet_min_exit_layer', 1)),
                                 float(getattr(config, 'eet_target_active_frac', 0.125)),
                                 str(getattr(config, 'eet_capacity_schedule', 'bell')))
        cap = float(getattr(config, 'eet_width_cap', 16.0))
        return [base * min(cap, (1.0 / max(ai, 1e-6)) ** _wp) for ai in a]

    if not spec:
        return [base] * n_layer
    if ',' in spec:
        vals = [float(v) for v in spec.split(',') if v.strip() != '']
        if len(vals) != n_layer:
            raise ValueError(f"p34_ffn_schedule has {len(vals)} entries for {n_layer} layers")
        return vals
    if spec == 'measured':
        r = _interp_profile(_D22_REACHABLE, n_layer)
        return [base * n_layer * x / sum(r) for x in r]
    if spec == 'measured_native':
        if n_layer in _NATIVE_REACHABLE:
            r = list(_NATIVE_REACHABLE[n_layer])
        else:
            # Fall back to interpolating from nearest measured depth
            nearest = min(_NATIVE_REACHABLE, key=lambda k: abs(k - n_layer))
            r = _interp_profile(_NATIVE_REACHABLE[nearest], n_layer)
            print0(f"p34_ffn_schedule='measured_native': no d{n_layer} profile, "
                   f"interpolating from d{nearest}")
        return [base * n_layer * x / sum(r) for x in r]
    if spec == 'shrink':
        r = _interp_profile(_D22_REACHABLE, n_layer)
        sup = _interp_profile(_D22_SUPPLY, n_layer)
        return [base * min(1.0, ri / max(si, 1e-9)) for ri, si in zip(r, sup)]
    if spec == 'last':
        return [0.0] * (n_layer - 1) + [base]
    for stride, iso in ((2, False), (4, False), (2, True), (4, True)):
        if spec == f"every{stride}" + ("_iso" if iso else ""):
            keep = [i for i in range(n_layer) if i % stride == stride - 1]
            w = base * (n_layer / max(1, len(keep))) if iso else base
            return [w if i in keep else 0.0 for i in range(n_layer)]
    raise ValueError(f"Unknown p34_ffn_schedule {spec!r}")


class LinearMLP(nn.Module):
    """D -> D linear projection, no activation. Cheap placeholder for layers
    where the FFN schedule removes the FFN entirely.

    Gives the residual stream a channel-recombination opportunity without the
    expressivity (and cost) of a full nonlinear FFN. This tests whether the FFN's
    feature recombination or its nonlinearity is the binding constraint in
    sparse-FFN architectures.
    """
    def __init__(self, config):
        super().__init__()
        d = config.n_embd
        self.proj = Linear(d, d, bias=False)

    def forward(self, x):
        return self.proj(x)


class DeepMLP(nn.Module):
    """Multi-stage FFN. Two variants behind p34_ffn_last_depth:

      depth=2  D -> 4D (ReLU^2) -> 4D (ReLU^2) -> D    'wide-wide'
      depth=3  D -> D  (ReLU^2) -> 4D (ReLU^2) -> D    'bottleneck'

    From the Montufar/Telgarsky depth-separation results, the number of linear
    regions grows exponentially in depth but only polynomially in width. If the
    headroom diagnostic showed unmet demand at the deeper layers, concentrating
    depth in one FFN may recover more expressivity per FLOP than distributing
    width across many shallow FFNs.
    """
    def __init__(self, config, depth_mode=2):
        super().__init__()
        d = config.n_embd
        mult = float(getattr(config, 'p34_ffn_mult', 4.0))
        h = max(1, int(round(d * mult)))
        if depth_mode == 2:
            # D -> 4D -> 4D -> D
            self.c_fc1 = Linear(d, h, bias=False)
            self.c_fc2 = Linear(h, h, bias=False)
            self.c_proj = Linear(h, d, bias=False)
        elif depth_mode == 3:
            # D -> D -> 4D -> D
            self.c_fc1 = Linear(d, d, bias=False)
            self.c_fc2 = Linear(d, h, bias=False)
            self.c_proj = Linear(h, d, bias=False)
        else:
            raise ValueError(f"DeepMLP depth_mode must be 2 or 3, got {depth_mode}")
        self.depth_mode = depth_mode

    def forward(self, x):
        x = F.relu(self.c_fc1(x)).square()
        x = F.relu(self.c_fc2(x)).square()
        x = self.c_proj(x)
        return x


class SwiGLUMLP(nn.Module):
    """SwiGLU feedforward: y = W_down( SiLU(W_gate x) * (W_up x) ).

    The control for Phase 35. ConditionedLinear with gate_source='tied' reduces
    exactly to y = W0 x + U * SiLU(2 V^T x), because c = 1 + tanh(t) and
    t*(1 + tanh(t)) = (2t)*sigmoid(2t) = SiLU(2t). So the arm that won the p35
    sweep is a parallel low-rank SiLU branch on every projection, not a
    conditioned operator. Before that can be called an architecture result it has
    to beat the ordinary way of adding a gated nonlinearity to a transformer,
    which is this.

    Width is set for PARAMETER PARITY with the ReLU^2 FFN it replaces: that one
    holds 2*4*D^2 = 8D^2, this one holds 3*D*H, so H = 8D/3.
    """
    def __init__(self, config):
        super().__init__()
        d = config.n_embd
        mult = float(getattr(config, 'p36_swiglu_mult', 8.0 / 3.0))
        h = max(1, int(round(d * mult)))
        self.hidden = h
        self.c_gate = Linear(d, h, bias=False)
        self.c_up   = Linear(d, h, bias=False)
        self.c_proj = Linear(h, d, bias=False)

    def forward(self, x):
        return self.c_proj(F.silu(self.c_gate(x)) * self.c_up(x))


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        # Phase 20: Select MLP variant based on config
        _hrcs = getattr(config, 'p20_hrcs_scale', 0)
        _lswr = getattr(config, 'p20_lswr_scale', 0)
        _lrcfb_k = getattr(config, 'p20_lrcfb_branches', 0)
        _dgcr_k = getattr(config, 'p20_dgcr_branches', 0)
        _mone_k = getattr(config, 'p20_mone_experts', 0)
        _ncea_k = getattr(config, 'p20_ncea_branches', 0)
        _adwi = getattr(config, 'p20_adwi', 0)
        _ffn_sched = resolve_ffn_schedule(config, config.n_layer)[layer_idx]
        _no_ffn_repl = str(getattr(config, 'p34_ffn_no_ffn_replacement', 'none')).strip()
        _last_depth = int(getattr(config, 'p34_ffn_last_depth', 1))
        _is_last_layer = (layer_idx == config.n_layer - 1)
        if _ffn_sched <= 0:
            # Schedule says no FFN in this block.
            if _no_ffn_repl == 'linear':
                self.mlp = LinearMLP(config)
            else:
                self.mlp = None
        elif _is_last_layer and _last_depth > 1 and _ffn_sched > 0:
            # Deep FFN in the final layer: depth_mode 2 = D->4D->4D->D, 3 = D->D->4D->D
            self.mlp = DeepMLP(config, depth_mode=_last_depth)
        elif getattr(config, 'p36_swiglu_ffn', 0):
            self.mlp = SwiGLUMLP(config)
        elif _hrcs > 0:
            self.mlp = HashRoutedMLP(config, scale_factor=_hrcs)
        elif _lswr > 0:
            self.mlp = LSHRoutedMLP(config, scale_factor=_lswr,
                                    n_planes=getattr(config, 'p20_lswr_planes', 8))
        elif _lrcfb_k > 0:
            self.mlp = FrozenRoutedMLP(config, n_branches=_lrcfb_k,
                                        narrow=bool(getattr(config, 'p20_lrcfb_narrow', 0)),
                                        learned_route=bool(getattr(config, 'p20_lrcfb_learned', 0)),
                                        topk=getattr(config, 'p20_lrcfb_topk', 0))
        elif _dgcr_k > 0:
            self.mlp = DetachedRoutedMLP(config, n_branches=_dgcr_k,
                                         aux_weight=getattr(config, 'p20_dgcr_aux_weight', 0.01))
        elif _mone_k > 0:
            self.mlp = MoNE_MLP(config, n_experts=_mone_k,
                                topk=getattr(config, 'p20_mone_topk', 0),
                                narrow=bool(getattr(config, 'p20_mone_narrow', 1)),
                                frozen_route=bool(getattr(config, 'p20_mone_frozen', 0)))
        elif _ncea_k > 0:
            self.mlp = NoiseCEA_MLP(config, n_branches=_ncea_k,
                                    eps=getattr(config, 'p20_ncea_eps', 0.1))
        elif _adwi:
            self.mlp = AttnDerivedMLP(config)
        elif getattr(config, 'p23_std_moe_experts', 0) > 0:
            # Phase 23: Standard MoE baseline (full-size experts, learned router)
            self.mlp = StandardMoE_MLP(config,
                                       n_experts=config.p23_std_moe_experts,
                                       topk=getattr(config, 'p23_std_moe_topk', 1),
                                       aux_weight=getattr(config, 'p23_std_moe_aux_weight', 0.01),
                                       use_quantile_route=getattr(config, 'p23_quantile_route', 0))
        else:
            self.mlp = MLP(config, layer_idx)

        # 18H: Mixture norm (learned RMSNorm + LayerNorm blend)
        if getattr(config, 'p18_mixture_norm', 0):
            self.norm_attn = MixtureNorm(config.n_embd)
            self.norm_mlp = MixtureNorm(config.n_embd)
        else:
            self.norm_attn = None
            self.norm_mlp = None
        # 18E: Stochastic depth (LayerDrop)
        self.layer_drop = getattr(config, 'p18_layer_drop', 0.0)
        # 19A: Residual Gate Scaling — learned scalar on block output
        # softplus(0.5413) ≈ 1.0 → identity at init
        self.residual_alpha = None
        if getattr(config, 'p19_residual_gate', 0):
            self.residual_alpha = nn.Parameter(torch.tensor(0.5413))
        # 19J: Training-time weight noise epsilon
        self._weight_noise_eps = float(getattr(config, 'p19_weight_noise', 0.0))

    def forward(self, x, ve, cos_sin, window_size, kv_cache, token_active=None, eet_frozen_kv=False, frozen_k=None, frozen_v=None,
                attn_mask=None):
        norm_fn_attn = self.norm_attn if self.norm_attn is not None else norm
        norm_fn_mlp = self.norm_mlp if self.norm_mlp is not None else norm
        # 19J: Weight noise — add isotropic noise to key weights during training
        if self.training and self._weight_noise_eps > 0:
            eps = self._weight_noise_eps
            for p in (self.mlp.parameters() if self.mlp is not None else ()):
                if p.ndim == 2:  # only perturb weight matrices, not biases
                    p.data.add_(eps * torch.randn_like(p))
        attn_kw = {} if attn_mask is None else {"attn_mask": attn_mask}
        attn_out = self.attn(norm_fn_attn(x), ve, cos_sin, window_size, kv_cache,
                             token_active=token_active, eet_frozen_kv=eet_frozen_kv,
                             frozen_k=frozen_k, frozen_v=frozen_v, **attn_kw)
        # 20I: If ADWI, extract per-head norms from attention output for FFN routing
        if isinstance(self.mlp, AttnDerivedMLP):
            _B, _T, _ = attn_out.shape
            n_head = self.attn.n_head
            head_dim = self.attn.head_dim
            # Reshape attn output to (B, T, H, D) to get per-head norms
            attn_reshaped = attn_out.view(_B, _T, n_head, head_dim)
            head_norms = (attn_reshaped ** 2).sum(dim=-1).detach()  # (B, T, H) — detach to avoid coupling
            self.mlp.set_attn_head_norms(head_norms)
        x = x + attn_out
        # 18E: Stochastic depth — randomly skip FFN during training
        if self.training and self.layer_drop > 0 and torch.rand(1).item() < self.layer_drop:
            pass  # Skip FFN
        else:
            scale = 1.0 / (1.0 - self.layer_drop) if self.training and self.layer_drop > 0 else 1.0
            # self.mlp is None when p34_ffn_schedule zeroes this layer's FFN.
            if self.mlp is not None:
                block_out = scale * self.mlp(norm_fn_mlp(x))
                # 19A: Residual Gate Scaling
                if self.residual_alpha is not None:
                    block_out = F.softplus(self.residual_alpha).to(block_out.dtype) * block_out
                x = x + block_out
        # 19J: Undo weight noise (subtract it back so gradients update the clean weights)
        if self.training and self._weight_noise_eps > 0:
            # NOTE: We don't undo here — the noise is effectively a different perturbation each step.
            # The noise added to .data before forward is part of this step's computation only.
            # The gradient update will apply to the noised weights, which is the intended SAM-like behavior.
            pass
        return x

    def forward_split(self, x_q, cos_sin_q, q_pos, k, v, window_size, key_mask=None,
                      n_q=None):
        """Block forward where attention reads full-length keys/values (EET kv_mode)."""
        norm_fn_attn = self.norm_attn if self.norm_attn is not None else norm
        norm_fn_mlp = self.norm_mlp if self.norm_mlp is not None else norm
        attn_out = self.attn.forward_split(
            norm_fn_attn(x_q), cos_sin_q, q_pos, k, v, window_size,
            key_mask=key_mask, n_q=n_q
        )
        x = x_q + attn_out
        if self.mlp is not None:
            block_out = self.mlp(norm_fn_mlp(x))
            if self.residual_alpha is not None:
                block_out = F.softplus(self.residual_alpha).to(block_out.dtype) * block_out
            x = x + block_out
        return x

    def forward_attn_only(self, x, ve, cos_sin, window_size, kv_cache):
        """A³D: Run attention only (no FFN). Returns x + attn_out."""
        norm_fn_attn = self.norm_attn if self.norm_attn is not None else norm
        attn_out = self.attn(norm_fn_attn(x), ve, cos_sin, window_size, kv_cache)
        return x + attn_out

    def forward_ffn_only(self, x):
        """A³D: Run FFN only on pre-attended x. Returns x + mlp(norm(x))."""
        norm_fn_mlp = self.norm_mlp if self.norm_mlp is not None else norm
        block_out = self.mlp(norm_fn_mlp(x))
        if self.residual_alpha is not None:
            block_out = F.softplus(self.residual_alpha).to(block_out.dtype) * block_out
        return x + block_out



# Research modules that own their entire subtree's initialization. GPT.init_weights
# calls reset_parameters(bound) on each and does not apply the generic fallback to
# them. Everything here holds bare nn.Parameters or buffers, which to_empty() fills
# with uninitialized memory and which the Linear/Embedding/Norm fallback cannot
# reach, so before this registry existed each of these ran on garbage.
_SELF_INIT_MODULES = (
    ConditionedLinear,
    ResidualAdaptiveLinear,
    SingularValueSteeringLinear,
    TuckerAdaptiveLinear,
    VQAdaptiveLinear,
    DecoupledAdaptiveLinear,
    AttentionEntropyStratifiedLinear,
    FrozenSubspaceIndexedLinear,
    PositionalScalarGatedLinear,
    SplitStreamLinear,
    LoKRLinear,
    CausalOutputMixer,
    CausalInterpolationLinear,
)

class GPT(nn.Module):
    def __init__(self, config, pad_vocab_size_to=64):
        """
        NOTE a major footgun: this __init__ function runs in meta device context (!!)
        Therefore, any calculations inside here are shapes and dtypes only, no actual data.
        => We actually initialize all data (parameters, buffers, etc.) in init_weights() instead.
        """
        super().__init__()
        self.config = config
        # Compute per-layer window sizes for sliding window attention
        # window_size is (left, right) tuple: (-1, 0) for full context, (N, 0) for sliding window
        self.window_sizes = self._compute_window_sizes(config)
        self.use_moe = config.use_moe
        self.use_perm = config.use_perm
        self.moe_num_experts = config.moe_num_experts
        self.moe_router_dim = config.moe_router_dim
        self.moe_embed_dim = config.moe_embed_dim
        self.use_remix_linear = config.use_remix_linear
        if config.use_remix_linear and getattr(config, 'remix_context_dim_ratio', 0) > 0:
            config.remix_context_dim = max(config.remix_context_dim, config.n_embd // config.remix_context_dim_ratio)
            print0(f"remix_context_dim auto-scaled to {config.remix_context_dim} (n_embd={config.n_embd} // ratio={config.remix_context_dim_ratio})")
        
        self.cclblock_modulation   = getattr(config, 'cclblock_modulation', 'weight')
        self.cclblock_stale_ctx_lag = getattr(config, 'cclblock_stale_ctx_lag', 0)
        
        # Auto-correct to nearest multiple of 48 (LCM of 3 and 16) if using Multiscale
        # This ensures 3 equal channels AND fp8 compatibility (divisible by 16)
        if getattr(config, 'cclblock_context_stream', 'ema') == 'multiscale':
            orig_dim = config.remix_context_dim
            config.remix_context_dim = max(48, round(orig_dim / 48) * 48)
            print0(f"remix_context_dim auto-corrected for MultiScale/FP8 from {orig_dim} to {config.remix_context_dim}")
        # Auto-cap router_context_window at long sequences.
        # The GlobalContextManager runs O(T²) causal attention.  At T=2048 with the default
        # router_context_window=-1 (full context), this produces a highly diluted context signal
        # AND is very expensive.  Cap it automatically so the residual-stream context_updaters
        # carry the long-range signal instead (they read from norm(x) which is well-conditioned).
        if config.use_remix_linear and config.router_context_window == -1 and config.sequence_len > 512:
            config.router_context_window = 256
            print0(f"router_context_window auto-capped to 256 (sequence_len={config.sequence_len} > 512). "
                   f"Set router_context_window explicitly to override.")
        self.remix_context_dim = config.remix_context_dim
        # Auto-scale: remix_basis_size=0 means "match model_dim"
        if config.remix_basis_size <= 0:
            config.remix_basis_size = config.n_embd
        self.remix_basis_size = config.remix_basis_size
        self.use_pos_embed = config.use_pos_embed
        if config.remixed_linear_kwargs is None:
            self.remixed_linear_kwargs = dict(use_basis_gate=True, use_output_gate=True, use_context=True)
        else:
            self.remixed_linear_kwargs = config.remixed_linear_kwargs
        # Wire output_gate_rank into kwargs so RemixedLinear picks it up
        self.remixed_linear_kwargs['output_gate_rank'] = getattr(config, 'remix_output_gate_rank', 8)
        if self.cclblock_modulation in ('householder', 'spectral', 'ocd', 'lie', 'polynomial', 'grassmann'):
            self.remixed_linear_kwargs['operator_modulation'] = self.cclblock_modulation
        else:
            self.remixed_linear_kwargs['operator_modulation'] = 'none'
        self.remixed_linear_kwargs['lie_generators'] = getattr(config, 'cclblock_lie_generators', 4)
        self.remixed_linear_kwargs['poly_order'] = getattr(config, 'cclblock_poly_order', 2)
        self.remixed_linear_kwargs['grassmann_bank_size'] = getattr(config, 'cclblock_grassmann_bank_size', 4)
        config.remixed_linear_kwargs = self.remixed_linear_kwargs
        # Pad vocab for efficiency (DDP, tensor cores). This is just an optimization - outputs are cropped in forward().
        # https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.resize_token_embeddings
        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab_size != config.vocab_size:
            print0(f"Padding vocab_size from {config.vocab_size} to {padded_vocab_size} for efficiency")
        if self.use_remix_linear:
            if self.cclblock_modulation == 'normalization':
                block_cls = CCLBlock
            else:
                block_cls = RemixedBlock  # 'weight' path
        else:
            block_cls = Block
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(padded_vocab_size, config.n_embd),
            "h": nn.ModuleList([block_cls(config, layer_idx) for layer_idx in range(config.n_layer)]),
        })
        if self.use_pos_embed:
            self.transformer["wpe"] = nn.Embedding(config.sequence_len, config.n_embd)

        # Phase 28E/F: GlobalTemplateBank — cross-layer shared template pool.
        # Must be created AFTER transformer blocks so we can inject references into RemixedLinear layers.
        _bank_mode = getattr(config, 'p28_global_template_bank', 'none')
        if config.use_remix_linear and _bank_mode != 'none':
            _n_templates = config.remixed_linear_kwargs.get('n_templates', 1) if config.remixed_linear_kwargs else 1
            self.p28_global_bank = GlobalTemplateBank(
                n_templates=_n_templates,
                n_embd=config.n_embd,
                basis_size=config.remix_basis_size,
                n_layers=config.n_layer,
            )
            # Inject bank reference into matching RemixedLinear layers
            for layer_idx, block in enumerate(self.transformer['h']):
                if not isinstance(block, RemixedBlock):
                    continue
                # FFN layers always get the bank
                for sub in [getattr(block.ffwd, 'c_fc', None), getattr(block.ffwd, 'c_proj', None)]:
                    if isinstance(sub, (RemixedLinear, RemixedLinearFused)) and sub._use_global_bank:
                        sub._global_bank = self.p28_global_bank
                        sub._layer_idx_for_bank = layer_idx
                        # bank_key is already set in __init__ based on in/out dim ratio
                # For 'all' mode, also inject into attention layers
                if _bank_mode == 'all':
                    for sub in [getattr(block.attn, 'c_q', None), getattr(block.attn, 'c_k', None),
                                 getattr(block.attn, 'c_v', None), getattr(block.attn, 'c_proj', None)]:
                        if isinstance(sub, (RemixedLinear, RemixedLinearFused)) and sub._use_global_bank:
                            sub._global_bank = self.p28_global_bank
                            sub._layer_idx_for_bank = layer_idx
            print0(f"[P28] GlobalTemplateBank created: mode={_bank_mode}, K={_n_templates}, "
                   f"n_layers={config.n_layer}, basis_size={config.remix_basis_size}")
        else:
            self.p28_global_bank = None

        self.embedding_model = None
        if self.use_moe:
            if self.use_perm:
                self.embedding_model = PermutationMoE(
                    vocab_size=padded_vocab_size,
                    block_size=config.sequence_len,
                    base_embed_dim=self.moe_embed_dim,
                    num_experts=self.moe_num_experts,
                    router_dim=self.moe_router_dim,
                    selection_mode='soft',
                    allow_replacement=True,
                    dropout=config.dropout,
                    router_context_window=config.router_context_window,
                    router_causal=config.router_causal,
                    router_num_heads=config.router_num_heads,
                    router_num_queries=config.router_num_queries,
                    router_n_layers=config.router_n_layers,
                    router_use_vocab_prior=config.router_use_vocab_prior,
                    moe_use_abs_pos_embed=config.moe_use_abs_pos_embed,
                    # Fix 1D: expert mode selection
                    perm_expert_mode=getattr(config, 'perm_expert_mode', 'low_rank'),
                    perm_rank=getattr(config, 'perm_rank', 16),
                )
            else:
                self.embedding_model = DirectContextualEmbedding(
                    vocab_size=padded_vocab_size,
                    dim=self.moe_embed_dim,
                    # Fix 1E: pass correct num_experts (K) and router_dim
                    num_experts=self.moe_num_experts,
                    router_dim=self.moe_router_dim,
                    context_window=config.router_context_window,
                    dropout=config.dropout,
                    router_causal=config.router_causal,
                    router_num_heads=config.router_num_heads,
                    router_num_queries=config.router_num_queries,
                    router_n_layers=config.router_n_layers,
                    router_use_vocab_prior=config.router_use_vocab_prior,
                )
            assert self.moe_embed_dim == config.n_embd, "moe_embed_dim must match n_embd"
        self.context_manager = None
        self.context_updaters = None
        # CCL: context is derived per-block inside RemixedBlock/CCLBlock via SelectiveContextStream.
        # GlobalContextManager and context_updaters are not used for remix_linear.
        # SCH: the output head is either the dense V x d softmax or a structured
        # code head. It keeps the name `lm_head` on purpose so every downstream
        # consumer (setup_optimizer's unembedding group, num_scaling_params,
        # checkpointing, the forward pass) keeps working unchanged.
        self.use_code_head = bool(getattr(config, 'use_code_head', False))
        if self.use_code_head:
            from nanochat.code_head import build_code_head, describe_head, CodeInputEmbedding
            if getattr(config, 'sch_head_type', 'code') == 'hsoftmax':
                # The tree head returns a loss, not logits, so forward() returns
                # before the auxiliary-objective block. Refuse the combination
                # instead of silently dropping those terms.
                assert not config.use_remix_linear and not getattr(config, 'use_moe', False), \
                    "sch_head_type=hsoftmax returns a loss directly and cannot carry the " \
                    "remix/MoE auxiliary objectives; run it on the dense backbone"
            self.lm_head = build_code_head(config, padded_vocab_size, config.n_embd)
            print0(describe_head(self.lm_head))
            # Phase 3: coding the input side too. Only meaningful for the
            # capability claim, since a code head alone lets new tokens be
            # generated but not consumed, which is half a capability.
            in_mode = getattr(config, 'sch_input_mode', 'table')
            if in_mode != 'table':
                assert getattr(config, 'sch_head_type', 'code') == 'code', \
                    "sch_input_mode requires the code head (sch_head_type=code)"
                self.transformer.wte = CodeInputEmbedding(
                    in_mode, self.lm_head, config.n_embd,
                    hidden=getattr(config, 'sch_input_hidden', 0))
                print0(f"[SCH] input embedding replaced: mode={in_mode} "
                       f"in_dim={self.transformer.wte.in_dim}")
        else:
            self.lm_head = Linear(config.n_embd, padded_vocab_size, bias=False)
        # SAP: the block head reads out through this same lm_head, so it adds no second
        # unembedding. It needs a dense V x d head to read out through.
        self.sap_head = None
        if int(getattr(config, 'sap_block_T', 0)) > 0:
            assert not self.use_code_head, \
                "the SAP block head reads out through the dense lm_head; do not combine it with use_code_head"
            from nanochat.block_head import BlockHead
            self.sap_head = BlockHead(config, config.vocab_size)
            if self.sap_head.mode in DEPTH_MODES and not self.sap_head.depth_share:
                # Trainable copies of the top trunk layers for the slot path (synced from the
                # trunk in init_weights and after --sap-init-trunk). They live under sap_head so
                # the head's optimizer groups and --sap-freeze-trunk treat them as head weights.
                self.sap_head.depth_blocks = nn.ModuleList(
                    [block_cls(config, i) for i in self.sap_head.depth_layer_ids])
            print0(f"[SAP] block head: mode={self.sap_head.mode} T={self.sap_head.T} "
                   f"frac={self.sap_head.frac} window={self.sap_head.W} "
                   f"layers={len(self.sap_head.dec)} params={sum(p.numel() for p in self.sap_head.parameters()):,}")
        # S13 SV-A splice codes (nanochat/splice.py): a code embedding, a code head, and the token
        # class map, which is a persistent buffer so checkpoints carry it.
        self.splice = None
        if int(getattr(config, 'splice_k', 0)) > 0:
            from nanochat.splice import Splice
            self.splice = Splice(int(config.splice_k), config.n_embd)
            self.register_buffer("class_of_token", torch.zeros(config.vocab_size, dtype=torch.long))
            print0(f"[S13] splice codes: K={config.splice_k} "
                   f"params={sum(p.numel() for p in self.splice.parameters()):,}")
        # S16-C: Lane-Relative Attention Bias (LRB) and Offset Embedding
        self.lane_rel_bias = None
        if getattr(config, 'lane_rel_bias', False):
            self.lane_rel_bias = nn.Parameter(torch.empty(config.n_head, 44))
        self.lane_offset_embed = None
        if getattr(config, 'lane_offset_embed', False):
            self.lane_offset_embed = nn.Embedding(getattr(config, 'lane_max_offsets', 256), config.n_embd)

        # Design 10 (auxiliary objective): lightweight head predicts boundary or entropy from
        # the mean context vector across all RemixedBlocks. Forces context to encode
        # non-trivial information and prevents gradient-collapse to identity.
        self.aux_head = None
        if config.use_remix_linear and getattr(config, 'cclblock_aux_objective', 'none') != 'none':
            self.aux_head = Linear(config.remix_context_dim, 1, bias=True)
            nn.init.zeros_(self.aux_head.weight)
            nn.init.constant_(self.aux_head.bias, 0.0)
        # Per-layer learnable scalars (inspired by modded-nanogpt)
        # resid_lambdas: scales the residual stream at each layer (init 1.0 = neutral)
        # x0_lambdas: blends initial embedding back in at each layer (init 0.0 = disabled)
        # Separate parameters so they can have different optimizer treatment
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))   # fake init, real init in init_weights()
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))     # fake init, real init in init_weights()
        # 19C: Residual Stream Depthwise Mixing — grouped 1x1 conv between blocks
        mix_groups = getattr(config, 'p19_residual_mix_groups', 0)
        self._residual_mix_groups = mix_groups
        if mix_groups > 0:
            # Per-layer: gamma (init=0 → identity) + group-linear mixer
            self.residual_mix_gamma = nn.ParameterList([
                nn.Parameter(torch.tensor(0.0)) for _ in range(config.n_layer)
            ])
            self.residual_mixers = nn.ModuleList([
                nn.Conv1d(config.n_embd, config.n_embd, 1, groups=config.n_embd // mix_groups)
                for _ in range(config.n_layer)
            ])
        else:
            self.residual_mix_gamma = None
            self.residual_mixers = None
        # 19E: Learned Residual Decay — global depth-dependent x0 decay
        # x0_factor_i = sigmoid(depth_decay_raw) ^ i. When depth_decay_raw is large,
        # sigmoid ≈ 1 so x0 persists across layers; when small, x0 fades quickly.
        self._use_residual_decay = bool(getattr(config, 'p19_residual_decay', 0))
        if self._use_residual_decay:
            # init such that sigmoid(2.0)^i starts with a moderate decay
            self.depth_decay_raw = nn.Parameter(torch.tensor(2.0))
        else:
            self.depth_decay_raw = None
        # Value embeddings (ResFormer-style): alternating layers, last layer always included
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict({str(i): nn.Embedding(padded_vocab_size, kv_dim) for i in range(config.n_layer) if has_ve(i, config.n_layer)})
        # To support meta device initialization, we init the rotary embeddings here, but it's just "fake" meta tensors only.
        # As for rotary_seq_len, these rotary embeddings are pretty small/cheap in memory,
        # so let's just over-compute them by 10X, but assert fail if we ever reach that amount.
        # In the future we can dynamically grow the cache, for now it's fine.
        self.rotary_seq_len = config.sequence_len * 10 # 10X over-compute should be enough, TODO make nicer?
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False) # persistent=False means it's not saved to the checkpoint
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def _report_uninitialized(self):
        """Name every tensor init_weights left as uninitialized memory.

        GPT is built on meta and then to_empty()'d, which replaces the storage of
        parameters AND buffers with whatever the allocator hands back. Anything
        init_weights does not explicitly write is therefore uninitialized memory
        for the whole run: all-zero on a fresh process (so the branch is dead and
        every gradient into it is exactly zero), arbitrary recycled values once
        the allocator has history.

        This is not hypothetical. Before the reset_parameters dispatch existed,
        twelve modulation modes ran this way, including every bare nn.Parameter
        in svs/dcu/tucker/vq/decoupled/aesp/psg/splitstream/lokr/com, the U/V/
        scale of --use-ral, and the frozen reflectors and routing projections of
        fsi, which live in buffers. So this stays as a permanent tripwire: any
        future research module that forgets its init gets named here rather than
        silently training on garbage.

        Returns a list of (name, kind) for anything still poisoned.
        """
        bad = []
        for n, p in self.named_parameters():
            if p.is_floating_point() and torch.isnan(p).any():
                bad.append((n, 'param'))
        for n, b in self.named_buffers():
            if b.is_floating_point() and torch.isnan(b).any():
                bad.append((n, 'buffer'))
        return bad

    @torch.no_grad()
    def init_weights(self, verify=True):
        """
        Initialize the full model in this one function for maximum clarity.

        wte (embedding):     normal, std=1.0
        lm_head:             normal, std=0.001
        for each block:
            attn.c_q:        uniform, std=1/sqrt(n_embd)
            attn.c_k:        uniform, std=1/sqrt(n_embd)
            attn.c_v:        uniform, std=1/sqrt(n_embd)
            attn.c_proj:     zeros
            mlp.c_fc:        uniform, std=1/sqrt(n_embd)
            mlp.c_proj:      zeros
        """
        # Poison first, so anything init_weights fails to write is detectable
        # afterwards instead of silently becoming uninitialized memory. Two
        # extra passes over the parameters, once per training run.
        if verify:
            for p in self.parameters():
                if p.is_floating_point():
                    p.fill_(float('nan'))
            for b in self.buffers():
                if b.is_floating_point():
                    b.fill_(float('nan'))

        def _generic_init(sub: nn.Module):
            """Default init for plain layers a specialized branch does not own."""
            if isinstance(sub, (Linear, nn.Linear)):
                torch.nn.init.xavier_uniform_(sub.weight)
                if sub.bias is not None:
                    torch.nn.init.zeros_(sub.bias)
            elif isinstance(sub, nn.Embedding):
                torch.nn.init.normal_(sub.weight, mean=0.0, std=0.02)
            elif isinstance(sub, (nn.LayerNorm, nn.RMSNorm)):
                if getattr(sub, 'weight', None) is not None:
                    torch.nn.init.ones_(sub.weight)
                if getattr(sub, 'bias', None) is not None:
                    torch.nn.init.zeros_(sub.bias)

        # Research branches are also on to_empty() storage and need explicit init.
        def _init_research_module(mod: nn.Module):
            # The generic fallback runs as a PRE-pass, not inline. modules()
            # yields parents before children, so when the fallback sat at the
            # bottom of this same loop it executed after a parent's specialized
            # branch and silently overwrote it one iteration later:
            # RemixedLinear's orthogonal basis came out xavier, and its
            # basis_modulator's intended bias of 2.0 (sigmoid -> 0.88) read back
            # as 0.0 (sigmoid -> 0.50). Running it first means a specialized
            # branch always wins, and nothing is left uninitialized either,
            # which claiming subtrees would have risked (ln_basis, for one).
            # Set legacy_init_fallback=1 on the config to reproduce runs made
            # before this was fixed.
            legacy = bool(getattr(self.config, 'legacy_init_fallback', 0))
            if not legacy:
                for sub in mod.modules():
                    _generic_init(sub)

            # We use a non-recursive approach to avoid overwriting specialized inits
            for sub in mod.modules():
                # Modules that own their entire subtree's init. Everything here
                # holds bare nn.Parameters or buffers that the Linear-only
                # fallback cannot reach, so before this dispatch existed they
                # ran on whatever to_empty() left in the storage.
                if isinstance(sub, _SELF_INIT_MODULES):
                    sub.reset_parameters(s)
                    continue
                if isinstance(sub, CCLBlock):
                    # AdaRMSNorm: zero proj weights, scale bias=1.0, shift bias=0.0.
                    # Guarantees identity RMSNorm at init regardless of ctx quality.
                    for ada in [sub.ada_norm_attn, sub.ada_norm_mlp]:
                        torch.nn.init.zeros_(ada.proj.weight)
                        torch.nn.init.zeros_(ada.proj.bias)
                        ada.proj.bias.data[:ada.n_embd] = 1.0   # scale
                    # SelectiveContextStream / MultiScaleContext: handled by their own
                    # __init__ zero-inits already; but re-apply here for safety.
                    _init_ctx_stream(sub.ctx_stream)
                    # Attn and MLP: same as regular Block
                    torch.nn.init.uniform_(sub.attn.c_q.weight, -s, s)
                    torch.nn.init.uniform_(sub.attn.c_k.weight, -s, s)
                    torch.nn.init.uniform_(sub.attn.c_v.weight, -s, s)
                    torch.nn.init.zeros_(sub.attn.c_proj.weight)
                    torch.nn.init.uniform_(sub.mlp.c_fc.weight, -s, s)
                    torch.nn.init.zeros_(sub.mlp.c_proj.weight)
                    if sub.attn.ve_gate is not None:
                        torch.nn.init.zeros_(sub.attn.ve_gate.weight)
                    continue  # Skip further processing of sub-components

                if isinstance(sub, RemixedBlock):
                    # SelectiveContextStream / MultiScaleContext replaces ctx_from_attn + ctx_ema_gate.
                    # FSI/AESP/CKR modes have ctx_stream=None (no context stream needed).
                    if sub.ctx_stream is not None:
                        _init_ctx_stream(sub.ctx_stream)
                if isinstance(sub, (RemixedLinear, RemixedLinearFused)):
                    # basis is None under Phase 31 route_side='basis' / drop_basis_proj
                    if getattr(sub, 'basis', None) is not None:
                        torch.nn.init.orthogonal_(sub.basis.weight)
                    if sub.template_mixing is not None:
                        torch.nn.init.kaiming_normal_(sub.template_mixing)
                    if sub.template_bank is not None:
                        if sub.template_bank.ndim == 3:
                            # Stacked 3D tensor: init each template slice independently
                            for k in range(sub.template_bank.shape[0]):
                                torch.nn.init.kaiming_normal_(sub.template_bank.data[k])
                        else:
                            torch.nn.init.kaiming_normal_(sub.template_bank)
                    # Phase 31 low-rank template deltas: base gets the same init the
                    # bank would have; U stays zero so each delta starts at exactly 0.
                    if sub.template_base is not None:
                        torch.nn.init.kaiming_normal_(sub.template_base)
                        torch.nn.init.kaiming_uniform_(sub.template_delta_v, a=math.sqrt(5))
                        torch.nn.init.zeros_(sub.template_delta_u)
                    # Phase 23 Stacked Tiny Experts
                    if sub.expert_up_w is not None:
                        K, H, B_sz = sub.expert_up_w.shape
                        torch.nn.init.kaiming_uniform_(sub.expert_up_w.view(K * H, B_sz), a=math.sqrt(5))
                    if sub.expert_down_w is not None:
                        torch.nn.init.zeros_(sub.expert_down_w)
                    # Diagnostic buffer, overwritten every forward, but it must
                    # still be written once or the tripwire below reports it and
                    # a noisy tripwire is one people learn to ignore.
                    if hasattr(sub, '_template_entropy_buf'):
                        sub._template_entropy_buf.zero_()
                    # The routing projection. Created in __init__ as
                    # randn(in_features, K)/sqrt(in_features), which on meta
                    # storage produces nothing, and no branch here ever wrote
                    # it: in the shipped K=8 arm template_route came out
                    # EXACTLY ZERO, so every token routed uniformly (1/K) at
                    # init in every layer. That is the maximum-averaging point,
                    # where mixing near-orthogonal templates costs a factor
                    # sqrt(1/K + (K-1)c/K) in operator norm, so the run starts
                    # with the strongest possible gradient to collapse onto one
                    # template. With --p22-template-routing-learned 0 it is a
                    # frozen buffer instead, so 'frozen random routing' was
                    # frozen UNIFORM routing: a static average of the bank.
                    if getattr(sub, 'template_route', None) is not None:
                        torch.nn.init.normal_(sub.template_route,
                                              std=sub.template_route.shape[0] ** -0.5)
                    if getattr(sub, 'route_affine_scale', None) is not None:
                        # 1/K keeps the initial spread of alpha comparable to a
                        # softmax's. Zero would be a dead start: every template
                        # would get identical gradient until the scale moved.
                        torch.nn.init.constant_(sub.route_affine_scale, 1.0 / sub.n_templates)
                    # Quantile routers hold their projection as a bare Parameter
                    # or buffer, with the same consequence.
                    _qr = getattr(sub, '_qrouter', None)
                    if _qr is not None:
                        if getattr(_qr, 'route_proj', None) is not None:
                            torch.nn.init.normal_(_qr.route_proj, std=_qr.route_proj.shape[-1] ** -0.5)
                        if getattr(_qr, 'q_exp', None) is not None:
                            torch.nn.init.normal_(_qr.q_exp, std=_qr.q_exp.shape[-1] ** -0.5)
                        for _b in ('ema_thresholds', '_ema_init'):
                            if hasattr(_qr, _b):
                                getattr(_qr, _b).zero_()
                    # LoKR: kaiming for down projections, zeros for up (identity start)
                    if sub.lokr_down_w is not None:
                        torch.nn.init.kaiming_uniform_(sub.lokr_down_w, a=math.sqrt(5))
                    if sub.lokr_up_w is not None:
                        torch.nn.init.zeros_(sub.lokr_up_w)
                    if sub.lokr_route_proj is not None:
                        torch.nn.init.normal_(sub.lokr_route_proj, std=0.02)
                    # Phase 23 Stacked Tiny Experts
                    if sub.expert_up_w is not None:
                        K, H, B_sz = sub.expert_up_w.shape
                        torch.nn.init.kaiming_uniform_(sub.expert_up_w.view(K * H, B_sz), a=math.sqrt(5))
                    if sub.expert_down_w is not None:
                        torch.nn.init.zeros_(sub.expert_down_w)
                    torch.nn.init.zeros_(sub.bias)
                    if sub.use_context:
                        # basis_modulator: may be None when basis_gate_mode='none' or 'linear'/'attn'
                        # Guard against None before iterating modules.
                        if sub.basis_modulator is not None:
                            linears = [m for m in sub.basis_modulator.modules() if isinstance(m, (Linear, nn.Linear))]
                            for m in linears:
                                torch.nn.init.xavier_uniform_(m.weight)
                                if m.bias is not None: torch.nn.init.zeros_(m.bias)
                            if linears: torch.nn.init.constant_(linears[-1].bias, 2.0)
                        # basis_gate_content / basis_gate_context (attn mode): xavier init
                        if sub.basis_gate_content is not None:
                            torch.nn.init.xavier_uniform_(sub.basis_gate_content.weight)
                            if sub.basis_gate_content.bias is not None:
                                torch.nn.init.zeros_(sub.basis_gate_content.bias)
                        if sub.basis_gate_context is not None:
                            torch.nn.init.xavier_uniform_(sub.basis_gate_context.weight)
                            if sub.basis_gate_context.bias is not None:
                                torch.nn.init.zeros_(sub.basis_gate_context.bias)
                        # Low-rank output gate: xavier for coeffs, small random for basis.
                        # Zero basis creates a zero-gradient trap for output_gate_scale:
                        # d tanh(s*0)/ds = 0 × sech²(0) = 0, so scale can never learn.
                        if hasattr(sub, 'output_gate_coeffs'):
                            torch.nn.init.xavier_uniform_(sub.output_gate_coeffs.weight)
                            torch.nn.init.zeros_(sub.output_gate_coeffs.bias)
                            torch.nn.init.normal_(sub.output_gate_basis, std=0.01)
                            torch.nn.init.constant_(sub.output_gate_scale, 0.1)
                    continue  # Skip further processing of this module's sub-components here

                if isinstance(sub, StandardMoE_MLP):
                    if hasattr(sub, 'experts_fc_w'):
                        E, H, D = sub.experts_fc_w.shape
                        torch.nn.init.kaiming_uniform_(sub.experts_fc_w.view(E * H, D), a=math.sqrt(5))
                        torch.nn.init.zeros_(sub.experts_proj_w)
                    if hasattr(sub, 'router'):
                        torch.nn.init.zeros_(sub.router.weight)
                    continue

                if isinstance(sub, LinearMoE):
                    K, O, I = sub.experts_w.shape
                    torch.nn.init.kaiming_uniform_(sub.experts_w.view(K * O, I), a=math.sqrt(5))
                    if sub.bias is not None:
                        torch.nn.init.zeros_(sub.bias)
                    # LinearMoE supports two router types:
                    # 1) nn.Linear router with .weight
                    # 2) QuantileBalancedRouter with .route_proj
                    if hasattr(sub.router, "weight"):
                        torch.nn.init.zeros_(sub.router.weight)
                    elif hasattr(sub.router, "route_proj"):
                        torch.nn.init.normal_(sub.router.route_proj, std=I ** -0.5)
                        if hasattr(sub.router, "ema_thresholds"):
                            sub.router.ema_thresholds.zero_()
                        if hasattr(sub.router, "_ema_init"):
                            sub.router._ema_init.zero_()
                    continue

                if isinstance(sub, SharedBlockRouter):
                    n_embd = sub.token_proj.shape[0]
                    if isinstance(sub.token_proj, nn.Parameter):
                        torch.nn.init.normal_(sub.token_proj, std=n_embd ** -0.5)
                        torch.nn.init.normal_(sub.seq_proj, std=n_embd ** -0.5)
                    continue

                if isinstance(sub, ImprovedContextAwareRouter):
                    torch.nn.init.normal_(sub.routing_queries, mean=0.0, std=sub.router_dim ** -0.5)
                    # expert_proj: normal init (per-position self-attention logits, well-behaved)
                    torch.nn.init.normal_(sub.expert_proj.weight, mean=0.0, std=0.02)
                    if sub.expert_proj.bias is not None: torch.nn.init.zeros_(sub.expert_proj.bias)
                    # Fix 2 (Bug 2): zero-init cross_expert_proj to disable the non-causal global-mean
                    # cross-attention path at init. _cross_attention returns the mean over all T tokens
                    # (leaks future tokens, signal dilutes as mean of 2048 things → near-constant).
                    # Zero-init lets training decide if this path is ever useful; alpha_gate will
                    # initially route all weight to the causal self_attn_logits path.
                    torch.nn.init.normal_(sub.cross_expert_proj.weight, mean=0.0, std=0.02)  # original normal init
                    if sub.cross_expert_proj.bias is not None: torch.nn.init.zeros_(sub.cross_expert_proj.bias)
                    
                    # Projections and gates (Xavier or Normal)
                    for m in [sub.embed_proj, sub.out_proj, sub.temperature_predictor, sub.alpha_gate]:
                        torch.nn.init.xavier_uniform_(m.weight)
                        if m.bias is not None: torch.nn.init.zeros_(m.bias)
                    if sub.use_vocab_prior:
                        torch.nn.init.normal_(sub.vocab_routing_bias.weight, mean=0.0, std=0.02)
                    continue

                # Fallback for remaining research linear layers/embeddings.
                # Under the fixed path this already ran in the pre-pass above;
                # repeating it here would restore exactly the clobbering the
                # pre-pass exists to remove, so it is legacy-only.
                if legacy:
                    _generic_init(sub)

                # Phase 17: Re-initialize buffers and position signals for
                # position-routed modules. to_empty() replaces tensor storage
                # with garbage; Python float attrs survive for re-init.
                if isinstance(sub, CausalKernelLinear):
                    sub._temperature.fill_(sub._init_temperature)
                    torch.nn.init.normal_(sub.pos_signal, std=0.01)
                    torch.nn.init.zeros_(sub.branch_conv.weight)
                    torch.nn.init.zeros_(sub.branch_conv.bias)
                    if sub.content_proj is not None:
                        sig_dim = sub.content_proj.shape[0]
                        sub.content_proj.copy_(torch.randn_like(sub.content_proj) / (sig_dim ** 0.5))
                    if getattr(sub, '_ortho_init', False):
                        for branch in sub.branches:
                            torch.nn.init.orthogonal_(branch.weight)
                elif isinstance(sub, LoKRLinear):
                    torch.nn.init.normal_(sub.pos_signal, std=0.01)
                    torch.nn.init.zeros_(sub.branch_conv.weight)
                    torch.nn.init.zeros_(sub.branch_conv.bias)
                elif isinstance(sub, (PositionGatedResidual, PositionalResidualBias)):
                    torch.nn.init.normal_(sub.pos_signal, std=0.01)
                    if isinstance(sub, PositionGatedResidual):
                        torch.nn.init.zeros_(sub.delta.weight)
                        torch.nn.init.zeros_(sub.gate_conv.weight)
                        torch.nn.init.constant_(sub.gate_conv.bias, -3.0)
                    elif isinstance(sub, PositionalResidualBias):
                        torch.nn.init.zeros_(sub.bias_conv.weight)
                        torch.nn.init.zeros_(sub.bias_conv.bias)
                elif isinstance(sub, CausalInterpolationLinear):
                    torch.nn.init.normal_(sub.pos_signal, std=0.01)
                    torch.nn.init.zeros_(sub.alpha_conv.weight)
                    torch.nn.init.constant_(sub.alpha_conv.bias, -5.0)
                    torch.nn.init.zeros_(sub.bias)
                # Phase 18: re-init after to_empty()
                elif isinstance(sub, KroneckerLinear):
                    target_std = math.sqrt(math.sqrt(2.0 / sub.in_features))
                    torch.nn.init.normal_(sub.A, std=target_std)
                    torch.nn.init.normal_(sub.B, std=target_std)
                    torch.nn.init.zeros_(sub.bias)
                elif isinstance(sub, AdaptiveGatedLinear):
                    torch.nn.init.zeros_(sub.gate_w)
                    torch.nn.init.constant_(sub.gate_b, 5.0)  # sigmoid(5)≈0.993
                elif isinstance(sub, DynamicActivation):
                    sub.alpha.data.fill_(1.0)
                    sub.beta.data.fill_(0.0)
                    sub.gamma.data.fill_(0.0)
                elif isinstance(sub, MixtureNorm):
                    sub.w1.data.fill_(1.0)
                    sub.w2.data.fill_(0.0)
                elif isinstance(sub, PerChannelScale):
                    sub.scale.data.fill_(1.0)

        # Embedding and unembedding
        if self.use_code_head:
            # Must run AFTER the NaN-poison pass above: the head materialises its
            # frozen Phi here (never in __init__, which runs on the meta device
            # and is followed by to_empty(), so a constructor-built buffer would
            # hold garbage).
            self.lm_head.init_weights()
            if not isinstance(self.transformer.wte, nn.Embedding):
                self.transformer.wte.init_weights()
            else:
                torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        else:
            torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
            torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        if "wpe" in self.transformer:
            torch.nn.init.normal_(self.transformer.wpe.weight, mean=0.0, std=1.0)

        # Transformer blocks: uniform init with bound = sqrt(3) * std (same standard deviation as normal)
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5 # sqrt(3) multiplier makes sure Uniform achieves the same std as Normal

        def _init_ctx_stream(stream):
            """Common init for SelectiveContextStream and MultiScaleContext."""
            if isinstance(stream, SelectiveContextStream):
                # gate bias=0 (sigmoid(0)=0.5 balanced), write zero-init (no spurious content)
                torch.nn.init.zeros_(stream.gate.bias)
                torch.nn.init.zeros_(stream.write.weight)
                torch.nn.init.zeros_(stream.write.bias)
                # gate weight: xavier for good gradient flow
                torch.nn.init.xavier_uniform_(stream.gate.weight)
            elif isinstance(stream, MultiScaleContext):
                for gate, bias_val in zip(stream.gates, MultiScaleContext.GATE_BIASES):
                    torch.nn.init.xavier_uniform_(gate.weight)
                    torch.nn.init.constant_(gate.bias, bias_val)
                for write in stream.writes:
                    torch.nn.init.zeros_(write.weight)
                    torch.nn.init.zeros_(write.bias)

        for block in self.transformer.h:
            if isinstance(block, (RemixedBlock, CCLBlock)):
                _init_research_module(block)
                # RemixedBlock attn ve_gate not handled in _init_research_module — do it here
                if isinstance(block, RemixedBlock) and block.attn.ve_gate is not None:
                    torch.nn.init.zeros_(block.attn.ve_gate.weight)
                    # 19I: VE gate bias zero-init (neutral at init)
                    if block.attn.ve_gate.bias is not None:
                        torch.nn.init.zeros_(block.attn.ve_gate.bias)
            else:
                # Phase 22: MoEFullLinear attention projections
                if getattr(block.attn, '_attn_moe_k', 0) > 0:
                    for proj, is_output in [(block.attn.c_q, False), (block.attn.c_k, False),
                                             (block.attn.c_v, False), (block.attn.c_proj, True)]:
                        for k_idx in range(proj.K):
                            if is_output:
                                torch.nn.init.zeros_(proj.expert_weights.data[k_idx])
                            else:
                                torch.nn.init.uniform_(proj.expert_weights.data[k_idx], -s, s)
                else:
                    torch.nn.init.uniform_(block.attn.c_q.weight, -s, s) # weights use Uniform to avoid outliers
                    torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
                    torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
                    torch.nn.init.zeros_(block.attn.c_proj.weight) # projections are zero
                # MLP init: handles both standard MLP and P20 variants
                # Check specific P20 types first (some have c_fc too)
                if isinstance(block.mlp, MoNE_MLP):
                    for expert_fc in block.mlp.experts_fc:
                        torch.nn.init.uniform_(expert_fc.weight, -s, s)
                    for expert_proj in block.mlp.experts_proj:
                        torch.nn.init.zeros_(expert_proj.weight)
                    torch.nn.init.zeros_(block.mlp.router.weight)
                elif isinstance(block.mlp, (FrozenRoutedMLP, DetachedRoutedMLP)):
                    for branch_fc in block.mlp.branches_fc:
                        torch.nn.init.uniform_(branch_fc.weight, -s, s)
                    for branch_proj in block.mlp.branches_proj:
                        torch.nn.init.zeros_(branch_proj.weight)
                    if isinstance(block.mlp, DetachedRoutedMLP):
                        for sub in block.mlp.router:
                            if hasattr(sub, 'weight'):
                                torch.nn.init.zeros_(sub.weight)
                elif isinstance(block.mlp, AttnDerivedMLP):
                    for expert_fc in block.mlp.experts_fc:
                        torch.nn.init.uniform_(expert_fc.weight, -s, s)
                    for expert_proj in block.mlp.experts_proj:
                        torch.nn.init.zeros_(expert_proj.weight)
                elif isinstance(block.mlp, NoiseCEA_MLP):
                    # Base c_fc/c_proj + perturbation deltas + router
                    torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
                    torch.nn.init.zeros_(block.mlp.c_proj.weight)
                    torch.nn.init.zeros_(block.mlp.router.weight)
                    for delta in block.mlp.delta_fc:
                        torch.nn.init.normal_(delta, std=0.01)
                    for delta in block.mlp.delta_proj:
                        torch.nn.init.normal_(delta, std=0.01)
                elif isinstance(block.mlp, SwiGLUMLP):
                    # Same recipe as the ReLU^2 FFN: in-projections uniform, the
                    # out-projection zero so the block starts as the identity.
                    torch.nn.init.uniform_(block.mlp.c_gate.weight, -s, s)
                    torch.nn.init.uniform_(block.mlp.c_up.weight, -s, s)
                    torch.nn.init.zeros_(block.mlp.c_proj.weight)
                elif hasattr(block.mlp, 'c_fc'):
                    # Standard MLP (or HashRoutedMLP / LSHRoutedMLP which have c_fc/c_proj)
                    torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
                    torch.nn.init.zeros_(block.mlp.c_proj.weight)
                # 19I: VE gate zero-init for dense Block too. The weight was
                # never initialized here (only the bias), so it held whatever
                # to_empty() left; the RemixedBlock/CCLBlock paths zero both.
                if block.attn.ve_gate is not None:
                    torch.nn.init.zeros_(block.attn.ve_gate.weight)
                    if block.attn.ve_gate.bias is not None:
                        torch.nn.init.zeros_(block.attn.ve_gate.bias)

            # Phase 19 inits (shared across Block/RemixedBlock/CCLBlock)
            # 19G: Spectral Reparameterization — decompose c_proj AFTER it is initialized
            mlp = block.mlp if hasattr(block, 'mlp') else (block.ffwd if hasattr(block, 'ffwd') else None)
            if mlp is not None:
                if hasattr(mlp, 'srp_proj') and mlp.srp_proj is not None:
                    mlp.srp_proj.init_from_weight(mlp.c_proj.weight)
                if hasattr(mlp, 'srp_fc') and mlp.srp_fc is not None:
                    mlp.srp_fc.init_from_weight(mlp.c_fc.weight)

        # Per-layer scalars
        self.resid_lambdas.fill_(1.0)   # 1.0 => typical residual connections at init
        self.x0_lambdas.fill_(0.1)      # 0.1 => small initial weight for skip connection to input embedding

        # 19C: Residual mixer init — gamma=0 (identity), conv weights small random
        if self.residual_mix_gamma is not None:
            for gamma_p in self.residual_mix_gamma:
                gamma_p.data.fill_(0.0)
        if self.residual_mixers is not None:
            for mixer in self.residual_mixers:
                torch.nn.init.normal_(mixer.weight, std=0.01)
                if mixer.bias is not None:
                    torch.nn.init.zeros_(mixer.bias)

        # 19E: Depth decay init
        if self.depth_decay_raw is not None:
            self.depth_decay_raw.data.fill_(2.0)  # sigmoid(2.0) ≈ 0.88 moderate decay

        # Value embeddings (init like c_v: uniform with same std)
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)

        if self.embedding_model is not None:
            _init_research_module(self.embedding_model)
            # Fix 1E: zero-init expert_codes so initial behavior = standard seed embedding
            if hasattr(self.embedding_model, 'expert_codes'):
                torch.nn.init.zeros_(self.embedding_model.expert_codes)
        if self.context_manager is not None:
            _init_research_module(self.context_manager)
        # Fix 1A: zero-init the *last* linear in each context_updater MLP so initial
        # behaviour is identical to static context (delta starts at zero)
        if self.context_updaters is not None:
            for updater in self.context_updaters:
                # updater is now a Sequential(Linear, GELU, Linear)
                linears = [m for m in updater.modules() if isinstance(m, (Linear, nn.Linear))]
                # Xavier init first layer, zero-init last layer for identity-start
                if len(linears) >= 1:
                    torch.nn.init.xavier_uniform_(linears[0].weight)
                    if linears[0].bias is not None:
                        torch.nn.init.zeros_(linears[0].bias)
                if len(linears) >= 2:
                    torch.nn.init.zeros_(linears[-1].weight)
                    if linears[-1].bias is not None:
                        torch.nn.init.zeros_(linears[-1].bias)

        # Rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin

        # SAP block head: last, so no generic research-module pass can overwrite it.
        if getattr(self, 'sap_head', None) is not None:
            self.sap_head.init_weights()
            self.sap_sync_depth_copies()
        if getattr(self, 'splice', None) is not None:
            torch.nn.init.normal_(self.splice.emb.weight, mean=0.0, std=1.0)
            torch.nn.init.normal_(self.splice.head.weight, mean=0.0, std=0.001)
            self.class_of_token.zero_()                 # base_train loads the class map after init
        if getattr(self, 'lane_rel_bias', None) is not None:
            self.lane_rel_bias.zero_()
        if getattr(self, 'lane_offset_embed', None) is not None:
            torch.nn.init.normal_(self.lane_offset_embed.weight, mean=0.0, std=0.02)

        # Cast embeddings to COMPUTE_DTYPE: optimizer can tolerate reduced-precision
        # embeddings and it saves memory. Exception: fp16 requires fp32 embeddings
        # because GradScaler cannot unscale fp16 gradients.
        if COMPUTE_DTYPE != torch.float16:
            # Only a real lookup table is safe to keep in reduced precision. A
            # coded input embedding (sch_input_mode != table) is a matmul, so its
            # master weights stay fp32 like every other matrix in the model.
            if isinstance(self.transformer.wte, nn.Embedding):
                self.transformer.wte.to(dtype=COMPUTE_DTYPE)
            if getattr(self, 'lane_offset_embed', None) is not None:
                self.lane_offset_embed.to(dtype=COMPUTE_DTYPE)
            for ve in self.value_embeds.values():
                ve.to(dtype=COMPUTE_DTYPE)

        if verify:
            self._verify_initialized()

    @torch.no_grad()
    def _verify_initialized(self):
        """Name and repair anything init_weights left as uninitialized memory.

        Separate from init_weights so a subclass can run its own initialization first and
        then check the FINAL state. EarlyExitGPT initializes its routers and buffers after
        super().init_weights() returns, so verifying inside the base call reported eight
        tensors that were about to be set correctly, printed "results from this run are
        suspect" on every EET run, and overwrote the routers with a generic init that the
        subclass then replaced anyway.
        """
        missed = self._report_uninitialized()
        if not missed:
            return
        # Repair conservatively so the run does not produce NaN, but say
        # so loudly: a generic init is a guess, and the module that owns
        # these tensors is the only thing that knows what they should be.
        by_class = {}
        for name, kind in missed:
            # A tensor registered directly on GPT has no dot in its name, and rsplit
            # would then hand get_submodule the tensor's own name and raise. '' is the
            # root module, which is the correct owner. Before this, the tripwire crashed
            # on exactly the case it exists to catch.
            owner_path = name.rsplit('.', 1)[0] if '.' in name else ''
            owner = self.get_submodule(owner_path)
            by_class.setdefault(type(owner).__name__, []).append(f"{name} ({kind})")
        print0("=" * 72)
        print0("init_weights: UNINITIALIZED TENSORS, results from this run are suspect")
        for cls_name, names in sorted(by_class.items()):
            print0(f"  {cls_name}: {len(names)} tensors, e.g. {', '.join(names[:3])}")
        print0("  Give the owning class a reset_parameters(bound) and add it to")
        print0("  _SELF_INIT_MODULES. Falling back to a generic init for now.")
        print0("=" * 72)
        for name, _kind in missed:
            t = self.get_parameter(name) if _kind == 'param' else self.get_buffer(name)
            if t.ndim >= 2:
                torch.nn.init.normal_(t, std=t.shape[-1] ** -0.5)
            else:
                torch.nn.init.zeros_(t)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=200000, device=None):
        # TODO: bump base theta more? e.g. 100K is more common more recently
        # autodetect the device from model embeddings
        if device is None:
            device = self.transformer.wte.weight.device
        # stride the channels
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        # stride the time steps
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        # calculate the rotation frequencies at each (time, channel) pair
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.to(COMPUTE_DTYPE), sin.to(COMPUTE_DTYPE)
        cos, sin = cos[None, :, None, :], sin[None, :, None, :] # add batch and head dims for later broadcasting
        return cos, sin

    def _compute_window_sizes(self, config):
        """
        Compute per-layer window sizes for sliding window attention.

        Returns list of (left, right) tuples for FA3's window_size parameter:
        - left: how many tokens before current position to attend to (-1 = unlimited)
        - right: how many tokens after current position to attend to (0 for causal)

        Pattern string is tiled across layers. Final layer always gets L (full context).
        Characters: L=long (full context), S=short (half context)
        """
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern), f"Invalid window_pattern: {pattern}. Use only S and L."
        # Map characters to window sizes
        long_window = config.sequence_len
        short_window = 256
        char_to_window = {
            "L": (long_window, 0),
            "S": (short_window, 0),
        }
        # Tile pattern across layers
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        # Final layer always gets full context
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def get_device(self):
        return self.transformer.wte.weight.device

    @property
    def max_seq_len(self):
        """Property for use by external evaluation scripts (like core_eval.py)."""
        return self.config.sequence_len

    def estimate_flops(self):
        """
        Return (total_flops_per_token, active_flops_per_token) for the model (forward + backward).

        total_flops: 6 × all matmul params + attention QK kernel FLOPs (same formula as before).
        active_flops: same as total but expert param counts are scaled by topk/K_total for MoE
                      layers, reflecting that only a fraction of expert weights are active per token.

        Each matmul weight contributes 2 FLOPs in forward, 4 in backward => 6 total.
        Ref: https://medium.com/@dzmitrybahdanau/the-flops-calculus-of-language-model-training-3b19c1f025e4
        Attention QK FLOPs: 12 * h * q * effective_seq_len per layer.
        Ref: https://arxiv.org/abs/2204.02311 (PaLM paper).
        """
        nparams = sum(p.numel() for p in self.parameters())
        # Exclude non-matmul params: embeddings and per-layer scalars
        value_embeds_numel = sum(ve.weight.numel() for ve in self.value_embeds.values())
        wpe_numel = self.transformer.wpe.weight.numel() if "wpe" in self.transformer else 0
        wte_numel = sum(p.numel() for p in self.transformer.wte.parameters())
        nparams_exclude = (wte_numel + wpe_numel + value_embeds_numel +
                          self.resid_lambdas.numel() + self.x0_lambdas.numel())
        if getattr(self, 'lane_offset_embed', None) is not None:
            nparams_exclude += sum(p.numel() for p in self.lane_offset_embed.parameters())
        if getattr(self, 'lane_rel_bias', None) is not None:
            nparams_exclude += self.lane_rel_bias.numel()
        # Binary per-channel scales (log_alpha / theta / log_g) are elementwise
        # rescales of the output, not matmul weights, so the 6N proxy overcharges
        # them. Tiny in absolute terms (512 floats against 3.1M matrix params per
        # layer at d=512) but the accounting claim is the point of this direction.
        for _n, _p in self.named_parameters():
            if _n.rsplit(".", 1)[-1] in ("log_alpha", "theta", "log_g"):
                nparams_exclude += _p.numel()
        # SCH: the head's FLOPs are NOT 6 * head_params.
        #   - the frozen Phi does V*M MACs per token while owning zero parameters,
        #     so the 6N proxy would report the code head as almost free;
        #   - a hierarchical softmax owns V*d node parameters but touches only
        #     ~log2(V)*d of them per token, so the same proxy would overcharge it
        #     by three orders of magnitude;
        #   - a coded input embedding is a matmul, not the free gather that an
        #     nn.Embedding is.
        # Both are therefore removed from the generic term and priced exactly.
        # This is the "honest cost model" the head has to be argued against: at
        # V=32k, d=512, order 4 (M=1940) the code head is 17x smaller in
        # parameters and ~4x more expensive in compute than the dense softmax.
        sch_head_flops = 0
        if getattr(self, 'use_code_head', False):
            nparams_exclude += sum(p.numel() for p in self.lm_head.parameters())
            sch_head_flops += self.lm_head.flops_per_token()
        if not isinstance(self.transformer.wte, nn.Embedding):
            sch_head_flops += self.transformer.wte.flops_per_token()
        # SAP: the block head runs on sap_block_frac of the positions, T slot rows each, and
        # reads out through the shared V x d head per slot row. 6N over its parameters would
        # charge it as if it ran once on every token, so it is priced explicitly instead.
        if getattr(self, 'sap_head', None) is not None:
            nparams_exclude += sum(p.numel() for p in self.sap_head.parameters())
            sch_head_flops += self.sap_head.flops_per_token(self.config.vocab_size)
        h, q, t = self.config.n_head, self.config.n_embd // self.config.n_head, self.config.sequence_len
        # Sum attention FLOPs per layer, accounting for sliding window
        attn_flops = 0
        for window_size in self.window_sizes:
            window = window_size[0]  # (left, right) tuple, we use left
            effective_seq = t if window < 0 else min(window, t)
            attn_flops += 12 * h * q * effective_seq
        total_flops = 6 * (nparams - nparams_exclude) + attn_flops + sch_head_flops

        # Compute active FLOPs: count inactive expert params as fractionally active
        # For each MoE layer: active_fraction = topk / K_total
        # We subtract the "inactive" portion of expert param counts.
        inactive_expert_params = 0
        for block in self.transformer.h:
            mlp = getattr(block, 'mlp', None)
            if mlp is None:
                continue
            if isinstance(mlp, StandardMoE_MLP):
                K = mlp.n_experts
                topk = mlp.topk if mlp.topk > 0 else K  # topk=0 means all
                if topk < K:
                    # Total expert params in this MLP block
                    expert_params = mlp.fc_w.numel() + mlp.proj_w.numel()
                    # Active fraction: topk/K; rest is inactive
                    inactive_frac = 1.0 - (topk / K)
                    inactive_expert_params += int(expert_params * inactive_frac)

        # Flat scan over all submodules for MoE/P24 inactive param accounting.
        # Note: SlicedWeightLinear/FoldedModulationLinear/SequenceGatedLinear are NOT
        # necessarily inside a RemixedLinear, so we must scan the full module tree.
        #
        # derived_matmul_flops: FLOPs from matmuls on derived weight matrices (e.g. W_eff
        # in chunk routing) that the 6N parameter-counting proxy cannot capture because
        # the derived matrix is an intermediate tensor, not an nn.Parameter.
        # For chunk routing, the forward does TWO ops with template_bank:
        #   1. Assembly: einsum('bnk,koc->bnoc') — captured by amortized param count
        #   2. Mixing:   einsum('bnsc,bnoc->bnso') — NOT captured (W_eff is derived)
        # We add 6 * out_features * basis_size per such projection to compensate.
        derived_matmul_flops = 0
        for submod in self.modules():
            if isinstance(submod, (RemixedLinear, RemixedLinearFused)):
                if getattr(submod, 'tiny_expert', False) and submod.n_templates > 1:
                    topk = submod.tiny_expert_topk
                    K = submod.n_templates
                    if topk > 0 and topk < K:
                        expert_params = submod.expert_up_w.numel() + submod.expert_down_w.numel()
                        inactive_frac = 1.0 - (topk / K)
                        inactive_expert_params += int(expert_params * inactive_frac)
                # Phase 31 delta mode (template_bank is None, template_base is set) is
                # deliberately excluded: every delta param is read for every token, so
                # there are no inactive params and no derived W_eff matmul. The plain
                # 6N proxy is already exact — active and hardware FLOPs coincide.
                elif (not getattr(submod, 'tiny_expert', False)
                      and not getattr(submod, 'lokr_expert', False)
                      and getattr(submod, 'n_templates', 1) > 1
                      and submod.template_bank is not None):
                    chunk = getattr(submod, 'chunk_routing_size', 0)
                    if chunk > 0:
                        # Chunk routing: routing + mixing ops amortized over chunk_size tokens
                        template_params = submod.template_bank.numel()
                        route_params = submod.template_route.numel() if submod.template_route is not None else 0
                        
                        topk = getattr(submod, 'template_topk', 0)
                        K = submod.n_templates
                        
                        # Active parameters per token is what's read from memory.
                        # For soft routing (topk=0): we read ALL templates once per chunk.
                        # For sparse routing (topk>0): we read only TOPK templates once per chunk.
                        active_templates = topk if (0 < topk < K) else K
                        active_template_params = template_params * (active_templates / K)
                        
                        # Active per token = (active_template_params + route_params) / chunk
                        active_per_token = (active_template_params + route_params) / chunk
                        
                        # inactive = total - active_per_token
                        inactive = (template_params + route_params) - active_per_token
                        inactive_expert_params += int(inactive)

                        # Fix: the mixing matmul (W_eff × h_gated) performs
                        # 2 * out_features * basis_size FLOPs per token (forward), i.e.
                        # 6 * out_features * basis_size per token (fwd+bwd). W_eff is a
                        # derived tensor constructed from template_bank, not an nn.Parameter,
                        # so the 6N formula doesn't see it. We must add it explicitly.
                        out_features = submod.template_bank.shape[1]
                        basis_size = submod.template_bank.shape[2]
                        derived_matmul_flops += 6 * out_features * basis_size
                    else:
                        # Legacy template bank with hard top-k routing
                        topk = getattr(submod, 'template_topk', 0)
                        K = submod.n_templates
                        if topk > 0 and topk < K:
                            template_params = submod.template_bank.numel()
                            inactive_frac = 1.0 - (topk / K)
                            inactive_expert_params += int(template_params * inactive_frac)
                elif getattr(submod, 'lokr_expert', False):
                    topk = submod.lokr_topk
                    K = submod.lokr_n_experts
                    if topk > 0 and topk < K:
                        expert_params = submod.lokr_down_w.numel() + submod.lokr_up_w.numel()
                        inactive_frac = 1.0 - (topk / K)
                        inactive_expert_params += int(expert_params * inactive_frac)
            elif isinstance(submod, LinearMoE):
                # LinearMoE blends weights, so only 1 dense matmul per token.
                # Active fraction ≈ 1/K.
                K = submod.n_experts
                if K > 1:
                    expert_params = submod.experts_w.numel()
                    inactive_frac = 1.0 - (1.0 / K)
                    inactive_expert_params += int(expert_params * inactive_frac)
            elif isinstance(submod, SlicedWeightLinear):
                # Selects n_selected columns out of in_features per token.
                # weight_bank is (out, in_features); active fraction = n_selected / in_features.
                if submod.n_selected < submod.in_features:
                    expert_params = submod.weight_bank.numel()
                    inactive_frac = 1.0 - (submod.n_selected / submod.in_features)
                    inactive_expert_params += int(expert_params * inactive_frac)
            # FoldedModulationLinear: weight is already (out, folded_dim) = the true active size.
            # No inactive params — total and active are equal by construction. No adjustment needed.
            # SequenceGatedLinear: weight is identical to dense baseline. total == active.

        total_flops = total_flops + derived_matmul_flops
        active_flops = total_flops - 6 * inactive_expert_params
        active_params = nparams - inactive_expert_params

        # EET: routing physically removes tokens from later blocks, so a block reached by
        # a fraction a of the tokens does a of its matmul work and a^2 of its attention
        # kernel work (queries and keys both shrink). Nothing above models that, so before
        # this every EET point was reported at essentially dense cost -- EET base came out
        # at 2.867e8 against dense's 2.863e8 -- and every Pareto plot in the repo had EET
        # at the wrong x-coordinate. Worse, the inverse-width arms had their extra
        # parameters counted while their routing saving was not, so they looked ~1.9x
        # dense when the honest figure is 0.86x.
        if getattr(self.config, 'use_eet', False) and getattr(self.config, 'eet_compute_skip', False):
            a = eet_active_fractions(
                self.config.n_layer,
                int(getattr(self.config, 'eet_min_exit_layer', 1)),
                float(getattr(self.config, 'eet_target_active_frac', 0.125)),
                str(getattr(self.config, 'eet_capacity_schedule', 'bell')))
            saved = 0.0
            for i, (blk, ai) in enumerate(zip(self.transformer.h, a)):
                bp = sum(p_.numel() for p_ in blk.parameters())
                window = self.window_sizes[i][0]
                eff = t if window < 0 else min(window, t)
                saved += 6 * bp * (1.0 - ai)                    # matmuls scale with a
                saved += 12 * h * q * eff * (1.0 - ai * ai)     # attention scales with a^2
            active_flops = max(active_flops - saved, 0.0)
        return total_flops, active_flops, active_params


    def num_scaling_params(self):
        """
        Return detailed parameter counts for scaling law analysis.
        Different papers use different conventions:
        - Kaplan et al. excluded embedding parameters
        - Chinchilla included all parameters
        Ref: https://arxiv.org/abs/2203.15556 (Chinchilla paper)
        Ref: https://arxiv.org/abs/2001.08361 (Kaplan et al. original scaling laws paper)

        Returns a dict with counts for each parameter group, so downstream analysis
        can experiment with which combination gives the cleanest scaling laws.
        """
        # Count each group separately (mirrors the grouping in setup_optimizers)
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        wpe = sum(p.numel() for p in self.transformer.wpe.parameters()) if "wpe" in self.transformer else 0
        value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
        research = 0
        if self.embedding_model is not None:
            research += sum(p.numel() for p in self.embedding_model.parameters())
        if self.aux_head is not None:
            research += sum(p.numel() for p in self.aux_head.parameters())
        if getattr(self, 'sap_head', None) is not None:
            research += sum(p.numel() for p in self.sap_head.parameters())
        if getattr(self, 'splice', None) is not None:     # S13 splice code table and head
            research += sum(p.numel() for p in self.splice.parameters())
        if getattr(self, 'lane_offset_embed', None) is not None:
            research += sum(p.numel() for p in self.lane_offset_embed.parameters())
        if getattr(self, 'lane_rel_bias', None) is not None:
            research += self.lane_rel_bias.numel()
        # AG-CCL: ctx_from_attn and ctx_ema_gate live inside transformer.h (RemixedBlock)
        # and are already counted in transformer_matrices above.
        scalars = self.resid_lambdas.numel() + self.x0_lambdas.numel()
        # P19: GPT-level scalar params (depth_decay_raw, residual_mix_gamma, residual_mixers)
        # These are nn.Parameters on GPT, not inside transformer.h, so we must count them.
        if self.depth_decay_raw is not None:
            scalars += self.depth_decay_raw.numel()
        if self.residual_mix_gamma is not None:
            for gamma_p in self.residual_mix_gamma:
                scalars += gamma_p.numel()
        if self.residual_mixers is not None:
            for mixer in self.residual_mixers:
                scalars += sum(p.numel() for p in mixer.parameters())
        total = wte + wpe + value_embeds + lm_head + transformer_matrices + research + scalars
        assert total == sum(p.numel() for p in self.parameters()), "Parameter count mismatch"
        return {
            'wte': wte,
            'wpe': wpe,
            'value_embeds': value_embeds,
            'lm_head': lm_head,
            'transformer_matrices': transformer_matrices,
            'research': research,
            'scalars': scalars,
            'total': total,
        }

    def convert_to_phase2(self):
        """Convert standard MLP modules in each Block to Phase 2 P20 variants.

        Called AFTER loading a Phase 1 checkpoint. Replaces each block's MLP
        with the Phase 2 variant (PWU_MLP, FSVD_MLP, or WBFC_MLP) by converting
        the pre-trained MLP weights in-place.

        Usage:
            model.load_state_dict(checkpoint)
            model.convert_to_phase2()  # converts MLPs using pretrained weights
            optimizer = model.setup_optimizer(...)  # re-setup optimizer for new params
        """
        config = self.config
        converted = 0
        pwu_k = getattr(config, 'p20_pwu_branches', 0)
        pwu_phase = getattr(config, 'p20_pwu_phase', 1)
        fsvd = getattr(config, 'p20_fsvd_gate', 0)
        wbfc_k = getattr(config, 'p20_wbfc_clusters', 0)
        wbfc_m = getattr(config, 'p20_wbfc_active', 0)

        if pwu_k == 0 and fsvd == 0 and wbfc_k == 0:
            return 0  # nothing to convert

        for i, block in enumerate(self.transformer.h):
            if not isinstance(block, Block):
                continue
            if not isinstance(block.mlp, MLP):
                continue  # skip if already converted or using another P20 variant

            old_mlp = block.mlp
            device = next(old_mlp.parameters()).device

            if pwu_k > 0:
                router_only = (pwu_phase == 2)
                new_mlp = PWU_MLP.from_pretrained_mlp(
                    config, old_mlp, n_branches=pwu_k,
                    router_only=router_only)
            elif fsvd > 0:
                new_mlp = FSVD_MLP.from_pretrained_mlp(config, old_mlp)
            elif wbfc_k > 0:
                new_mlp = WBFC_MLP.from_pretrained_mlp(
                    config, old_mlp, n_clusters=wbfc_k,
                    n_active=wbfc_m if wbfc_m > 0 else max(1, wbfc_k // 4))
            else:
                continue

            block.mlp = new_mlp.to(device)
            converted += 1
            del old_mlp

        print(f"Phase 2 conversion: converted {converted} MLP modules")
        return converted

    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0, adam_betas=(0.8, 0.95), scalar_lr=0.5, disable_mu_p=False, mu_p_scale_override=-1.0, gate_lr_scale=0.3):
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()

        # Separate out all parameters into groups
        # For RemixedLinear, split gate params (basis_modulator, output_gate_*) from
        # structural params (basis, template_mixing) — gate params get a lower LR (suggestion 5).
        gate_matrix_params   = []  # 2D gate params  → lower-LR Muon
        gate_adamw_params    = []  # 1D gate params  → lower-LR AdamW
        struct_matrix_params = []  # 2D struct params → normal Muon
        struct_adamw_params  = []  # 1D struct params → normal AdamW
        ckr_gate_adamw_params = []  # 13b: CKR gate params → dedicated conservative AdamW
        eet_router_matrix_params = []  # EET router 2D params → dedicated high-LR Muon
        eet_router_adamw_params  = []  # EET router 1D params → dedicated high-LR AdamW
        template_bank_params = []  # P29: 3D template_bank params → block-diagonal Muon

        def _sort_ctx_stream_params(stream):
            """Route SelectiveContextStream/MultiScaleContext params to gate groups."""
            for p in stream.parameters():
                (gate_matrix_params if p.ndim == 2 else gate_adamw_params).append(p)

        if self.use_remix_linear:
            for block in self.transformer.h:
                if isinstance(block, CCLBlock):
                    # 'normalization' path: ctx_stream and AdaRMSNorm.proj are gate-side.
                    # Standard CausalSelfAttention + MLP weights are structural.
                    _sort_ctx_stream_params(block.ctx_stream)
                    for ada in [block.ada_norm_attn, block.ada_norm_mlp]:
                        for p in ada.parameters():
                            (gate_matrix_params if p.ndim == 2 else gate_adamw_params).append(p)
                    # Structural: attn Q/K/V/proj, MLP fc/proj
                    for p in [block.attn.c_q.weight, block.attn.c_k.weight,
                               block.attn.c_v.weight, block.attn.c_proj.weight,
                               block.mlp.c_fc.weight, block.mlp.c_proj.weight]:
                        struct_matrix_params.append(p)
                    if block.attn.ve_gate is not None:
                        struct_matrix_params.append(block.attn.ve_gate.weight)
                else:
                    assert isinstance(block, RemixedBlock), "Expected RemixedBlock or CCLBlock in remix_linear mode"
                    # 'weight' path: ctx streams and projs are gate-side.
                    if hasattr(block, 'ctx_stream') and block.ctx_stream is not None:
                        _sort_ctx_stream_params(block.ctx_stream)
                    if hasattr(block, 'ctx_stream_attn'):
                        _sort_ctx_stream_params(block.ctx_stream_attn)
                    if hasattr(block, 'ctx_stream_ffn'):
                        _sort_ctx_stream_params(block.ctx_stream_ffn)
                    
                    if hasattr(block, 'ctx_proj_q'):
                        for p in block.ctx_proj_q.parameters():
                            (gate_matrix_params if p.ndim == 2 else gate_adamw_params).append(p)
                    if hasattr(block, 'shadow_ctx_proj'):
                        for p in block.shadow_ctx_proj.parameters():
                            (gate_matrix_params if p.ndim == 2 else gate_adamw_params).append(p)

                    # RemixedLinear: sort gate vs structural
                    remix_linears = [
                        block.attn.c_q, block.attn.c_k, block.attn.c_v, block.attn.c_proj,
                    ] + ([block.ffwd.c_fc, block.ffwd.c_proj] if block.ffwd is not None else [])
                    # 13b: Dual-optimizer CKR — route CKR gate params to dedicated AdamW
                    ckr_dual = (self.cclblock_modulation == 'ckr' and
                                getattr(self.config, 'cclblock_ckr_dual_optim', 0))
                    for rl in remix_linears:
                        if hasattr(rl, 'gate_parameters'):
                            for p in rl.gate_parameters():
                                if ckr_dual:
                                    # 13b: CKR gate params → dedicated AdamW (conservative)
                                    ckr_gate_adamw_params.append(p)
                                else:
                                    (gate_matrix_params if p.ndim == 2 else gate_adamw_params).append(p)
                        if hasattr(rl, 'non_gate_parameters'):
                            for p in rl.non_gate_parameters():
                                if p.ndim == 3 and hasattr(rl, 'template_bank') and p is rl.template_bank:
                                    # P29: 3D stacked template_bank → dedicated block-diagonal Muon group
                                    template_bank_params.append(p)
                                elif p.ndim >= 2:
                                    struct_matrix_params.append(p)
                                else:
                                    struct_adamw_params.append(p)
                        else:
                            # Plain Linear/Float8Linear — all params are structural
                            for p in rl.parameters():
                                (struct_matrix_params if p.ndim == 2 else struct_adamw_params).append(p)
                        # ln_basis (LayerNorm) — only on research modules (None on OutputGatedLinear)
                        if hasattr(rl, 'ln_basis') and rl.ln_basis is not None:
                            for p in rl.ln_basis.parameters():
                                struct_adamw_params.append(p)
                    # ve_gate (if present) is structural
                    if block.attn.ve_gate is not None:
                        struct_matrix_params.append(block.attn.ve_gate.weight)
                        # 19I: VE gate bias (if present)
                        if block.attn.ve_gate.bias is not None:
                            struct_adamw_params.append(block.attn.ve_gate.bias)
                    # Phase 18: MixtureNorm params (scalar weights)
                    if hasattr(block, 'norm_attn') and block.norm_attn is not None:
                        for p in block.norm_attn.parameters():
                            (struct_matrix_params if p.ndim == 2 else struct_adamw_params).append(p)
                    if hasattr(block, 'norm_mlp') and block.norm_mlp is not None:
                        for p in block.norm_mlp.parameters():
                            (struct_matrix_params if p.ndim == 2 else struct_adamw_params).append(p)
                    # Phase 18: DynamicActivation params (α, β, γ scalars)
                    if hasattr(block.ffwd, 'dynamic_act') and block.ffwd.dynamic_act is not None:
                        for p in block.ffwd.dynamic_act.parameters():
                            struct_adamw_params.append(p)
                    # Phase 18: PerChannelScale params (1D scale vector)
                    if hasattr(block.ffwd, 'channel_scale') and block.ffwd.channel_scale is not None:
                        for p in block.ffwd.channel_scale.parameters():
                            struct_adamw_params.append(p)
                    # Phase 19: Per-block P19 params
                    # 19A: residual_alpha scalar
                    if hasattr(block, 'residual_alpha') and block.residual_alpha is not None:
                        struct_adamw_params.append(block.residual_alpha)
                    # 19B: head_importance (per-head scalars)
                    if hasattr(block.attn, 'head_importance') and block.attn.head_importance is not None:
                        struct_adamw_params.append(block.attn.head_importance)
                    # 19D: attn_logit_scale (per-head scalars)
                    if hasattr(block.attn, 'attn_logit_scale') and block.attn.attn_logit_scale is not None:
                        struct_adamw_params.append(block.attn.attn_logit_scale)
                    # 19G: Spectral Reparameterization sigma params
                    ffwd = block.ffwd if hasattr(block, 'ffwd') else (block.mlp if hasattr(block, 'mlp') else None)
                    if ffwd is not None:
                        if hasattr(ffwd, 'srp_proj') and ffwd.srp_proj is not None:
                            struct_adamw_params.append(ffwd.srp_proj.sigma)
                        if hasattr(ffwd, 'srp_fc') and ffwd.srp_fc is not None:
                            struct_adamw_params.append(ffwd.srp_fc.sigma)
                    # Phase 23: SharedBlockRouter routing projections → gate group
                    if hasattr(block, '_shared_router') and block._shared_router is not None:
                        for p in block._shared_router.gate_parameters():
                            (gate_matrix_params if p.ndim == 2 else gate_adamw_params).append(p)
                    # Phase 23: SharedContextGates → gate group (batched basis + output gate projections)
                    if hasattr(block, '_ctx_gates') and block._ctx_gates is not None:
                        for p in block._ctx_gates.gate_parameters():
                            (gate_matrix_params if p.ndim == 2 else gate_adamw_params).append(p)
                    # 19J: no params (noise epsilon is a config float, not learned)
        else:
            # Regular Block: split standard struct parameters from MoE router parameters
            gate_param_ids = set()
            # EET early-exit routers → dedicated group (NOT gate group)
            eet_router_matrix_params = []
            eet_router_adamw_params = []
            for name, p in self.named_parameters():
                if "eet_router" in name:
                    (eet_router_matrix_params if p.ndim == 2 else eet_router_adamw_params).append(p)
                    gate_param_ids.add(id(p))
            for m in self.transformer.h.modules():
                # MoNE_MLP router
                if isinstance(m, MoNE_MLP) and getattr(m, 'router', None) is not None:
                    for p in m.router.parameters():
                        (gate_matrix_params if p.ndim == 2 else gate_adamw_params).append(p)
                        gate_param_ids.add(id(p))
                # FrozenRoutedMLP router (if learned)
                elif isinstance(m, FrozenRoutedMLP) and getattr(m, 'learned_route', False) and hasattr(m, 'content_proj'):
                    p = m.content_proj
                    if isinstance(p, nn.Parameter):
                        (gate_matrix_params if p.ndim == 2 else gate_adamw_params).append(p)
                        gate_param_ids.add(id(p))

            # Phase 22: shared attention MoE router → gate group
            for block in self.transformer.h:
                if hasattr(block.attn, '_attn_shared_route'):
                    p = block.attn._attn_shared_route
                    if isinstance(p, nn.Parameter):
                        (gate_matrix_params if p.ndim == 2 else gate_adamw_params).append(p)
                        gate_param_ids.add(id(p))

            # All other parameters go to candidate_matrix_params / struct_adamw_params
            for p in self.transformer.h.parameters():
                if id(p) not in gate_param_ids:
                    if p.ndim == 2:
                        struct_matrix_params.append(p)
                    else:
                        struct_adamw_params.append(p)

        # Embedding model (MoE), if present
        if self.embedding_model is not None:
            cand = list(self.embedding_model.parameters())
            struct_matrix_params += [p for p in cand if p.ndim == 2]
            struct_adamw_params  += [p for p in cand if p.ndim != 2]

        # Aux head (Design 10), if present
        if self.aux_head is not None:
            for p in self.aux_head.parameters():
                (struct_matrix_params if p.ndim == 2 else struct_adamw_params).append(p)

        # SAP block head: its lookup tables (slot, code and window-position embeddings) train
        # like embeddings with AdamW; its projections are matrices for Muon.
        sap_embedding_params = []
        if getattr(self, 'sap_head', None) is not None:
            _sap_emb_ids = {id(p) for p in self.sap_head.embedding_parameters()}
            for p in self.sap_head.parameters():
                if id(p) in _sap_emb_ids:
                    sap_embedding_params.append(p)
                else:
                    (struct_matrix_params if p.ndim == 2 else struct_adamw_params).append(p)

        # S13 splice codes: the code table trains like an embedding, the code head like lm_head.
        if getattr(self, 'splice', None) is not None:
            sap_embedding_params.append(self.splice.emb.weight)
        if getattr(self, 'lane_offset_embed', None) is not None:
            sap_embedding_params.append(self.lane_offset_embed.weight)
        if getattr(self, 'lane_rel_bias', None) is not None:
            struct_adamw_params.append(self.lane_rel_bias)

        research_adamw_params = gate_adamw_params + struct_adamw_params

        value_embeds_params = list(self.value_embeds.parameters())
        embedding_params = list(self.transformer.wte.parameters()) + sap_embedding_params
        if "wpe" in self.transformer:
            embedding_params += list(self.transformer.wpe.parameters())
        lm_head_params = list(self.lm_head.parameters())
        if getattr(self, 'splice', None) is not None:
            lm_head_params.append(self.splice.head.weight)
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        # Phase 19: Collect GPT-level P19 params
        p19_scalar_params = []
        # 19C: Residual mixer gamma + conv weights
        if self.residual_mix_gamma is not None:
            for gamma_p in self.residual_mix_gamma:
                p19_scalar_params.append(gamma_p)
        if self.residual_mixers is not None:
            for mixer in self.residual_mixers:
                for p in mixer.parameters():
                    (struct_matrix_params if p.ndim == 2 else p19_scalar_params).append(p)
        # 19E: Depth decay scalar
        if self.depth_decay_raw is not None:
            p19_scalar_params.append(self.depth_decay_raw)
        all_params = (gate_matrix_params + struct_matrix_params + research_adamw_params +
                      embedding_params + lm_head_params + value_embeds_params +
                      resid_params + x0_params + ckr_gate_adamw_params + p19_scalar_params +
                      eet_router_matrix_params + eet_router_adamw_params +
                      template_bank_params)

        # Safety catch-all: route any uncovered params (e.g. lokr_route_proj when
        # use_context=False, which is gated inside gate_parameters()) to struct groups.
        covered_ids = {id(p) for p in all_params}
        orphan_params = [p for p in self.parameters() if id(p) not in covered_ids]
        if orphan_params:
            orphan_names = [n for n, p in self.named_parameters()
                            if id(p) in {id(o) for o in orphan_params}]
            print0(f"[setup_optimizer] Catch-all: routing {len(orphan_params)} uncovered "
                   f"params to struct groups: {orphan_names}")
            for p in orphan_params:
                (struct_matrix_params if p.ndim == 2 else struct_adamw_params).append(p)
            research_adamw_params = gate_adamw_params + struct_adamw_params  # recompute with new entries
            all_params = (gate_matrix_params + struct_matrix_params + research_adamw_params +
                          embedding_params + lm_head_params + value_embeds_params +
                          resid_params + x0_params + ckr_gate_adamw_params + p19_scalar_params +
                          eet_router_matrix_params + eet_router_adamw_params +
                          template_bank_params)

        # Binary per-channel scales get their OWN group at a low LR.
        #
        # log_alpha / log_g are LOG-parameterised, and everything else in this model
        # is not. Left in research_adamw_params they receive
        # embedding_lr * dmodel_lr_scale = 0.2 * (512/768)^-0.5 ~= 0.245, and since
        # Adam's step magnitude is ~lr, the scale multiplies by exp(0.245) ~= 1.28
        # EVERY STEP. The scales blow up or collapse within a few dozen steps and the
        # model cannot learn. Checking that log_alpha avoids Muon was not enough; the
        # AdamW LR it landed on was the actual problem.
        _bin_scale_names = ("log_alpha", "theta", "log_g")
        binary_scale_params = [q for n, q in self.named_parameters()
                               if n.rsplit(".", 1)[-1] in _bin_scale_names]
        if binary_scale_params:
            _bs_ids = {id(q) for q in binary_scale_params}
            _drop = lambda lst: [q for q in lst if id(q) not in _bs_ids]
            gate_matrix_params = _drop(gate_matrix_params)
            struct_matrix_params = _drop(struct_matrix_params)
            gate_adamw_params = _drop(gate_adamw_params)
            struct_adamw_params = _drop(struct_adamw_params)
            research_adamw_params = _drop(research_adamw_params)
            embedding_params = _drop(embedding_params)
            lm_head_params = _drop(lm_head_params)
            value_embeds_params = _drop(value_embeds_params)
            all_params = _drop(all_params) + binary_scale_params

        assert len(list(self.parameters())) == len(all_params), (
            f"Parameter count mismatch even after catch-all: model has "
            f"{len(list(self.parameters()))} params, optimizer groups cover {len(all_params)}")

        # Scale the LR for the AdamW parameters by ∝1/√dmodel (tuned for 768 dim model, μP-style).
        # Can be disabled via disable_mu_p=True so raw flags are used directly (e.g. for research models
        # that may not satisfy the μP assumptions).
        if mu_p_scale_override > 0.0:
            dmodel_lr_scale = mu_p_scale_override
            print0(f"μP LR scaling OVERRIDDEN to {dmodel_lr_scale:.6f}")
        elif disable_mu_p:
            dmodel_lr_scale = 1.0
            print0(f"μP LR scaling DISABLED — using raw LR flags directly (model_dim={model_dim})")
        else:
            dmodel_lr_scale = (model_dim / 768) ** -0.5
            print0(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")

        # EET backbone LR multiplier: scale all non-router LRs when EET is active.
        # Computed BEFORE router LR so the router remains independently controlled.
        _eet_model_lr_mult = getattr(self.config, 'eet_model_lr_mult', 1.0)
        _orig_matrix_lr = matrix_lr  # save unscaled for router LR computation
        if _eet_model_lr_mult != 1.0 and getattr(self.config, 'use_eet', False):
            print0(f"[EET] Scaling backbone LRs by eet_model_lr_mult={_eet_model_lr_mult:.3f}")
            matrix_lr *= _eet_model_lr_mult
            unembedding_lr *= _eet_model_lr_mult
            embedding_lr *= _eet_model_lr_mult
            scalar_lr *= _eet_model_lr_mult

        # Build param_groups with all required fields explicit
        param_groups = [
            # AdamW groups (embeddings, lm_head, scalars)
            dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=research_adamw_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            # scalar_lr * 0.01, the same treatment resid_lambdas get, because a log
            # scale needs to move slowly and multiplicatively.
            dict(kind='adamw', params=binary_scale_params, lr=scalar_lr * 0.01, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=resid_params, lr=scalar_lr * 0.01, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),
        ]
        # Structural matrix params: normal Muon LR (or depth-scaled when eet_depth_lr_scale=True)
        _use_depth_lr = getattr(self.config, 'eet_depth_lr_scale', False) and getattr(self.config, 'use_eet', False)
        if _use_depth_lr:
            # Compute per-layer LR multipliers from the capacity schedule
            import math as _math
            _n_layer = self.config.n_layer
            _min_exit = getattr(self.config, 'eet_min_exit_layer', 1)
            _n_rl = _n_layer - 1 - _min_exit  # number of routing layers
            _target_frac = getattr(self.config, 'eet_target_active_frac', 0.125)
            _schedule = getattr(self.config, 'eet_capacity_schedule', 'bell')
            # Compute exit fracs same way as forward pass
            if _schedule == 'uniform':
                _exit_fracs = [(1.0 - _target_frac) / _n_rl] * _n_rl
            elif _schedule == 'linear':
                _ws = [k + 1 for k in range(_n_rl)]
                _tw = sum(_ws)
                _exit_fracs = [w / _tw * (1.0 - _target_frac) for w in _ws]
            elif _schedule == 'bell':
                _mid = (_n_rl - 1) / 2.0
                _sigma = max(1.0, _n_rl / 4.0)
                _ws = [_math.exp(-((_k - _mid) / _sigma) ** 2) for _k in range(_n_rl)]
                _tw = sum(_ws)
                _exit_fracs = [w / _tw * (1.0 - _target_frac) for w in _ws]
            else:  # geometric
                _psf = 1.0 - _target_frac ** (1.0 / _n_rl)
                _exit_fracs = [_psf] * _n_rl
            # Compute per-layer surviving fraction
            _survivor = [1.0] * _n_layer  # fraction of tokens active at each layer
            _surv = 1.0
            _rl_idx = 0
            for layer_i in range(_n_layer):
                _survivor[layer_i] = _surv
                if layer_i in range(_min_exit, _n_layer - 1) and _rl_idx < len(_exit_fracs):
                    _surv -= _exit_fracs[_rl_idx]
                    _rl_idx += 1
            # Per-layer LR scale = 1 / surviving_fraction (capped at 10×)
            _layer_lr_scale = [min(10.0, 1.0 / max(s, 0.05)) for s in _survivor]
            print0(f"[EET] Depth-scaled LR multipliers: {[f'{s:.2f}' for s in _layer_lr_scale]}")
            # Map each param in struct_matrix_params to its layer index
            _layer_param_ids = {}
            for layer_i, block in enumerate(self.transformer.h):
                for p in block.parameters():
                    _layer_param_ids[id(p)] = layer_i
            # Group struct_matrix_params by (layer, shape) with scaled LR
            _depth_grouped = {}  # (layer_idx, shape) -> [params]
            _non_layer_params = []
            for p in struct_matrix_params:
                li = _layer_param_ids.get(id(p))
                if li is not None:
                    key = (li, p.shape)
                    _depth_grouped.setdefault(key, []).append(p)
                else:
                    _non_layer_params.append(p)
            for (li, shape), gp in sorted(_depth_grouped.items()):
                param_groups.append(dict(
                    kind='muon', params=gp, lr=matrix_lr * _layer_lr_scale[li],
                    momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=weight_decay,
                ))
            # Non-layer params (e.g. embedding model) at base LR
            if _non_layer_params:
                for shape in sorted({p.shape for p in _non_layer_params}):
                    gp = [p for p in _non_layer_params if p.shape == shape]
                    param_groups.append(dict(
                        kind='muon', params=gp, lr=matrix_lr,
                        momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=weight_decay,
                    ))
        else:
            for shape in sorted({p.shape for p in struct_matrix_params}):
                group_params = [p for p in struct_matrix_params if p.shape == shape]
                param_groups.append(dict(
                    kind='muon', params=group_params, lr=matrix_lr,
                    momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=weight_decay,
                ))
        # Gate matrix params: gate_lr_scale× Muon LR (default 0.3×, configurable)
        gate_lr = matrix_lr * gate_lr_scale
        for shape in sorted({p.shape for p in gate_matrix_params}):
            group_params = [p for p in gate_matrix_params if p.shape == shape]
            param_groups.append(dict(
                kind='muon', params=group_params, lr=gate_lr,
                momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=weight_decay,
            ))
        # 13b: Dedicated conservative AdamW for CKR position routing params
        if ckr_gate_adamw_params:
            param_groups.append(dict(
                kind='adamw', params=ckr_gate_adamw_params,
                lr=embedding_lr * dmodel_lr_scale * 0.5,  # conservative: half of embedding LR
                betas=(0.9, 0.999),  # high β₂ for slow, stable adaptation
                eps=1e-10, weight_decay=0.0,  # no decay on positional params
            ))
        # EET router params: dedicated group with configurable LR multiplier
        # Routers need higher LR than gate params to break the constant-function
        # equilibrium (MoE literature: gating networks need 2-10× higher LR).
        # Uses _orig_matrix_lr (before eet_model_lr_mult scaling) so the two are independent.
        eet_router_lr_mult = getattr(self.config, 'eet_router_lr_mult', 5.0)
        eet_router_lr = _orig_matrix_lr * gate_lr_scale * eet_router_lr_mult
        if eet_router_matrix_params:
            for shape in sorted({p.shape for p in eet_router_matrix_params}):
                group_params = [p for p in eet_router_matrix_params if p.shape == shape]
                param_groups.append(dict(
                    kind='muon', params=group_params, lr=eet_router_lr,
                    momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=weight_decay,
                ))
        if eet_router_adamw_params:
            param_groups.append(dict(
                kind='adamw', params=eet_router_adamw_params, lr=eet_router_lr,
                betas=adam_betas, eps=1e-10, weight_decay=0.0,
            ))
        # Phase 19: GPT-level scalar params (mixer gammas, depth decay, etc.)
        if p19_scalar_params:
            param_groups.append(dict(
                kind='adamw', params=p19_scalar_params,
                lr=scalar_lr, betas=adam_betas, eps=1e-10, weight_decay=0.0,
            ))
        # Phase 29: Template bank params → block-diagonal Muon with optional LR scaling
        # IMPORTANT: block_diagonal=K is ALWAYS used for template_bank. Before P0, each
        # template was a separate 2D nn.Parameter, so Muon ran Newton-Schulz independently
        # per template. After stacking into (K, out, basis), we must use block_diagonal=K
        # to preserve that per-template independence. Without it, all K templates get a
        # single joint Newton-Schulz pass, causing cross-template gradient contamination.
        _p29_lr_scale = getattr(self.config, 'p29_template_lr_scale', 1.0)
        if template_bank_params:
            template_lr = matrix_lr * _p29_lr_scale
            for shape in sorted({p.shape for p in template_bank_params}):
                group_params = [p for p in template_bank_params if p.shape == shape]
                K = shape[0]  # n_templates dimension
                group_dict = dict(
                    kind='muon', params=group_params, lr=template_lr,
                    momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=weight_decay,
                    _template_bank_3d=True,  # flag for 3D→2D reshape handling
                    block_diagonal=K,        # per-template Newton-Schulz (mandatory for correctness)
                )
                param_groups.append(group_dict)
            if _p29_lr_scale != 1.0:
                print0(f"[P29] Template bank Muon LR scaled by {_p29_lr_scale:.2f}× → {template_lr:.6f}")
            print0(f"[P29] Template bank: block-diagonal Muon (K={template_bank_params[0].shape[0]} blocks)")

        Factory = DistMuonAdamW if ddp else MuonAdamW
        optimizer = Factory(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean',
                skip_logits=False, return_hidden=False, sap_capture=False, lane_mask=None, pos_ids=None,
                head_from=0, extra_embed=None, splice=None, lane_params=None):
        """skip_logits returns the final normalised hidden states (B, T, d) without the
        lm_head, which is all block decoding needs from a trunk pass. return_hidden makes
        the inference path return (logits, hidden). pos_ids (T,) gives each input its rotary
        position explicitly (S11 two-stream window bisection, where both streams reuse 0..N-1);
        it needs an explicit lane_mask and no kv_cache. head_from > 0 runs the lm_head only on
        positions head_from.. (the query stream), and targets then cover only those positions.
        splice=(P, L, lane_token) scores ordinary rows as S13 splice-code lanes (nanochat/splice.py);
        extra_embed (B, T, d) is added to the token embeddings."""
        if splice is not None:
            from nanochat.splice import splice_loss
            assert targets is not None, "splice=... is a scoring path; it needs targets"
            return splice_loss(self, idx, targets, *splice, loss_reduction=loss_reduction)
        B, T = idx.size()

        # Grab the rotary embeddings for the current sequence length (they are of shape (1, seq_len, 1, head_dim/2))
        if pos_ids is not None:
            assert kv_cache is None and "wpe" not in self.transformer
            cos_sin = self.cos.index_select(1, pos_ids), self.sin.index_select(1, pos_ids)
            T0, T_total = None, None
        elif kv_cache is not None and getattr(kv_cache, 'graph_safe', False):
            # CUDA-graph decode: positions come from the cache on the device, with no host
            # sync, so the step can be captured once and replayed. The rotary cache must
            # already cover the whole generation (sequence_len * 10 by construction).
            pos = kv_cache.cache_seqlens[0].to(torch.long) + torch.arange(T, device=idx.device)
            cos_sin = self.cos.index_select(1, pos), self.sin.index_select(1, pos)
            T0, T_total = None, None
        else:
            T0 = 0 if kv_cache is None else kv_cache.get_pos()
            T_total = T0 + T

        if T_total is not None and T_total > self.cos.size(1):
            # Dynamic cache growth: double the cache or use T_total, whichever is larger
            new_len = max(T_total, self.cos.size(1) * 2)
            print0(f"Growing rotary embeddings cache from {self.cos.size(1)} to {new_len}")
            head_dim = self.config.n_embd // self.config.n_head
            cos, sin = self._precompute_rotary_embeddings(new_len, head_dim)
            # Re-register buffers to update their size (persistent=False as in __init__)
            self.register_buffer("cos", cos, persistent=False)
            self.register_buffer("sin", sin, persistent=False)

        if T_total is not None:
            cos_sin = self.cos[:, T0:T_total], self.sin[:, T0:T_total] # truncate cache to current sequence length

        # Forward the trunk of the Transformer
        if self.embedding_model is None:
            x = self.transformer.wte(idx) # embed current token
        else:
            x, _ = self.embedding_model(idx)
        if extra_embed is not None:                     # S13 splice: code embeddings on lane-first inputs
            x = x + extra_embed.to(x.dtype)
        if lane_mask is not None and getattr(self.config, 'lane_offset_embed', False) and getattr(self, 'lane_offset_embed', None) is not None:
            from nanochat.lanes import deduce_lane_layout, get_lrb_cache
            _lp_P, _lp_L = lane_params if lane_params is not None else deduce_lane_layout(lane_mask)
            _, _, _offsets = get_lrb_cache(x.size(1), _lp_P, _lp_L, device=x.device,
                                           roles=getattr(self.config, 'lane_rel_bias_roles', True))
            _max_o = getattr(self.config, 'lane_max_offsets', 256)
            x = x + self.lane_offset_embed(_offsets.clamp(max=_max_o - 1))
        if "wpe" in self.transformer:
            if T_total is None:  # graph-safe decode: positions already on the device
                positions = pos
            else:
                assert T_total <= self.config.sequence_len, f"use_pos_embed=True requires sequence <= {self.config.sequence_len}, got {T_total}"
                positions = torch.arange(T0, T_total, device=idx.device)
            x = x + self.transformer.wpe(positions)
        x = x.to(COMPUTE_DTYPE) # ensure activations are in compute dtype (no-op usually, but active for fp16 code path)
        x = norm(x)
        x0 = x  # save initial normalized embedding for x0 residual
        _sap_states = {}  # depth_local: residual entering each top trunk layer (filled below)
        p24_global_signal = None
        if self.use_remix_linear:
            use_global = (
                getattr(self.config, 'p24_sliced_weight_scope', 'per_token') == 'global' or
                getattr(self.config, 'p24_folded_mod_scope', 'per_layer') == 'global' or
                getattr(self.config, 'p24_sequence_gated_scope', 'per_layer') == 'global'
            )
            if use_global and (
                getattr(self.config, 'p24_use_sliced_weight', 0) or
                getattr(self.config, 'p24_use_folded_mod', 0) or
                getattr(self.config, 'p24_use_sequence_gated_linear', 0)
            ):
                p24_global_signal = x.mean(dim=1)
        # CCL context threading: context flows block-to-block, starting as None at block 0.
        # RemixedBlock and CCLBlock both return (x, ctx) and accept prev_ctx.
        # Design C (cclblock_stale_ctx_lag > 0): block i receives context from block i-k
        # (guaranteed to be independent because it's from a different layer's computation).
        # The stale context is detached — within-block gradient path remains intact.
        lag = self.cclblock_stale_ctx_lag
        if self.use_remix_linear and lag > 0:
            ctx_history = []
            for i, block in enumerate(self.transformer.h):
                x  = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
                ve = self.value_embeds[str(i)](idx).to(x.dtype) if str(i) in self.value_embeds else None
                stale_ctx = ctx_history[i - lag].detach() if i >= lag else None
                x, new_ctx = block(x, ve, cos_sin, self.window_sizes[i], kv_cache, stale_ctx, p24_global_signal=p24_global_signal)
                ctx_history.append(new_ctx)
        else:
            prev_ctx = None
            _ds_lambda = float(getattr(self.config, 'deep_supervision_lambda', 0.0))
            _ds_n = max(1, int(float(getattr(self.config, 'deep_supervision_frac', 0.125))
                               * idx.size(0) * idx.size(1)))
            _ds_loss = torch.zeros((), device=idx.device, dtype=torch.float32)
            collect_sim = float(getattr(self.config, 'p18_aux_sim_lambda', 0.0)) > 0
            if collect_sim:
                self._layer_outputs = []
            # 19E: Precompute x0 decay factors
            if self._use_residual_decay and self.depth_decay_raw is not None:
                decay_base = torch.sigmoid(self.depth_decay_raw)  # scalar in (0, 1)
            else:
                decay_base = None
            # depth_local reads the residual stream entering each of the top trunk layers.
            _sap_depth = (self.sap_head is not None and self.sap_head.mode in DEPTH_MODES
                          and (sap_capture or (kv_cache is None and self.training and targets is not None)))
            _sap_ids = set(self.sap_head.depth_capture_ids) if _sap_depth else set()
            _sap_states = {}
            _lrb_mask = lane_mask
            if lane_mask is not None and getattr(self.config, 'lane_rel_bias', False) and getattr(self, 'lane_rel_bias', None) is not None:
                from nanochat.lanes import deduce_lane_layout, get_lrb_cache
                _lp_P, _lp_L = lane_params if lane_params is not None else deduce_lane_layout(lane_mask)
                _vis, _buckets, _ = get_lrb_cache(x.size(1), _lp_P, _lp_L, device=x.device,
                                                  roles=getattr(self.config, 'lane_rel_bias_roles', True))
                _dt = torch.bfloat16 if x.is_cuda else x.dtype
                _bias = self.lane_rel_bias[:, _buckets.clamp(min=0)][None].to(_dt)
                _lrb_mask = torch.where(_vis[None, None], _bias, torch.tensor(float('-inf'), device=x.device, dtype=_dt))
            for i, block in enumerate(self.transformer.h):
                # 19E: Apply depth-dependent x0 decay
                x0_w = self.x0_lambdas[i]
                if decay_base is not None:
                    x0_w = x0_w * (decay_base ** i)
                if i in _sap_ids:
                    _sap_states[i] = x      # pre-mix residual entering block i, every position
                x = self.resid_lambdas[i] * x + x0_w * x0
                ve = self.value_embeds[str(i)](idx).to(x.dtype) if str(i) in self.value_embeds else None
                if self.use_remix_linear:
                    x, prev_ctx = block(x, ve, cos_sin, self.window_sizes[i], kv_cache, prev_ctx, p24_global_signal=p24_global_signal)
                elif lane_mask is not None:
                    x = block(x, ve, cos_sin, self.window_sizes[i], kv_cache, attn_mask=_lrb_mask)
                else:
                    x = block(x, ve, cos_sin, self.window_sizes[i], kv_cache)
                # Tier 0 test 2: make this layer prediction-readable through the shared head.
                if (_ds_lambda > 0.0 and self.training and targets is not None
                        and i < len(self.transformer.h) - 1):
                    _sel = torch.randint(0, x.size(0) * x.size(1), (_ds_n,), device=x.device)
                    _h = norm(x).view(-1, x.size(-1))[_sel]
                    _lg = self.lm_head(_h)[..., :self.config.vocab_size].float()
                    _lg = 20.0 * torch.tanh(_lg / 20.0)
                    _ds_loss = _ds_loss + F.cross_entropy(
                        _lg, targets.view(-1)[_sel], ignore_index=-1)
                # 19C: Residual stream mixing after block output
                if self.residual_mixers is not None:
                    gamma = self.residual_mix_gamma[i].to(x.dtype)
                    # Conv1d expects (B, C, T), but x is (B, T, C)
                    mixed = self.residual_mixers[i](x.transpose(1, 2)).transpose(1, 2)
                    x = x + gamma * mixed
                if collect_sim:
                    self._layer_outputs.append(x.detach())
        x = norm(x)
        if head_from:
            x = x[:, head_from:]
        if self.sap_head is not None and self.sap_head.mode == "depth_roll":
            _sap_states["top"] = x       # depth_roll: slot 1's feedback is the trunk's final state at t
        if skip_logits:
            if sap_capture:
                return x, (_sap_states if self.sap_head is not None and self.sap_head.mode in DEPTH_MODES else {})
            return x

        # Forward the lm_head (compute logits)
        softcap = 20 # smoothly cap the logits to the range [-softcap, softcap]
        if self.use_code_head and getattr(self.lm_head, 'custom_loss', False):
            # Some heads own their loss. Hierarchical softmax does so because the full
            # V-wide logit vector is never materialised at all. The proposal head does
            # so because WHICH logits it computes exactly depends on the targets, but
            # it can still produce a logit vector when asked (evaluation and generation
            # run its exact dense head), so it is not held to the tree head's rule.
            if targets is not None:
                return self.lm_head.loss(x, targets, reduction=loss_reduction)
            assert getattr(self.lm_head, 'emits_logits', False), (
                "hierarchical softmax cannot produce a logit vector; it supports the "
                "loss path only (use a code or dense head for generation / rank probes)")
        logits = self.lm_head(x) # (B, T, padded_vocab_size) <- very big tensor, large amount of memory
        logits = logits[..., :self.config.vocab_size] # slice to remove padding
        logits = logits.float() # switch to fp32 for logit softcap and loss computation
        # A self-normalised head (log-sum-exp mixture, sigsoftmax) already returns
        # log-probabilities. Squashing those with tanh would silently change the
        # distribution rather than bound a logit, so softcap is skipped there;
        # F.cross_entropy is still exact, because log_softmax of a normalised
        # log-prob vector is the identity.
        if not (self.use_code_head and getattr(self.lm_head, 'self_normalized', False)):
            logits = softcap * torch.tanh(logits / softcap) # squash the logits

        # A permuted-vocabulary head returns logits in block-position order rather
        # than token order. The loss path permutes the *targets* instead of the
        # logits, which is a gather of N integers rather than a second (N, V)
        # tensor; every other consumer (generation, rank probes) gets vocabulary
        # order restored, at one gather on a path that is not the bottleneck.
        if self.use_code_head and getattr(self.lm_head, 'permutes_vocab', False):
            inv = self.lm_head.inv_perm[:self.config.vocab_size]
            if targets is not None:
                # -1 is the ignore index and must survive the gather untouched.
                targets = torch.where(targets < 0, targets, inv[targets.clamp_min(0)])
            else:
                logits = logits.index_select(-1, inv)

        if targets is not None:
            # training: given the targets, compute and return the loss
            if self.use_code_head and getattr(self.lm_head, 'self_normalized', False):
                # `logits` are already log-probabilities. F.cross_entropy would
                # log_softmax them again, which is the identity on a normalised
                # vector and costs a second full (V-wide) tensor saved for
                # backward: 34 GB at 262144 tokens and V=32768. nll_loss skips it
                # and is exactly equal.
                loss = F.nll_loss(logits.view(-1, logits.size(-1)), targets.view(-1),
                                  ignore_index=-1, reduction=loss_reduction)
            else:
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=loss_reduction)
            # Tier 0 test 2: averaged over the supervised layers so the weight means the
            # same thing at any depth. Only ever added to a scalar ('mean') loss.
            if _ds_lambda > 0.0 and self.training and loss_reduction == 'mean':
                n_sup = max(1, len(self.transformer.h) - 1)
                loss = loss + _ds_lambda * (_ds_loss / n_sup)

            # SAP: the block loss rides on the next-token loss in training only, so every
            # bpb evaluation (loss_reduction='none') still measures the trunk's next-token
            # quality and the block head is evaluated separately (sap_block_eval).
            if self.sap_head is not None and self.training and loss_reduction == 'mean':
                if self.sap_head.mode in DEPTH_MODES:
                    _sap_loss, _sap_stats = self._sap_depth_loss(idx, x, targets, logits, _sap_states)
                else:
                    _sap_loss, _sap_stats = self._sap_block_loss(x, targets, logits)
                loss = loss + float(self.config.sap_lambda) * _sap_loss
                if not torch.compiler.is_compiling():
                    self._sap_stats = _sap_stats

            # Design 10: Auxiliary context objective
            # Reads _last_ctx stored on each RemixedBlock during this forward pass.
            if self.aux_head is not None and loss_reduction == 'mean':
                aux_obj = getattr(self.config, 'cclblock_aux_objective', 'none')
                ctx_vecs = []
                for block in self.transformer.h:
                    if isinstance(block, RemixedBlock) and hasattr(block, '_last_ctx') and block._last_ctx is not None:
                        ctx_vecs.append(block._last_ctx)
                if ctx_vecs:
                    # Mean context across all layers: (B, T, ctx_dim)
                    ctx_mean = torch.stack(ctx_vecs, dim=0).mean(dim=0).to(logits.dtype)
                    aux_logits = self.aux_head(ctx_mean).squeeze(-1)  # (B, T)
                    if aux_obj == 'boundary':
                        boundary_id = getattr(self.config, 'cclblock_boundary_token_id', 198)
                        boundary_target = (targets == boundary_id).float()
                        aux_loss = F.binary_cross_entropy_with_logits(aux_logits, boundary_target)
                    elif aux_obj == 'entropy':
                        # Teach context to predict per-token loss magnitude (detached)
                        with torch.no_grad():
                            token_loss = F.cross_entropy(
                                logits.view(-1, logits.size(-1)), targets.view(-1),
                                ignore_index=-1, reduction='none').view(B, T)
                        aux_loss = F.mse_loss(aux_logits, token_loss.detach() / 3.0)
                    else:
                        aux_loss = torch.zeros(1, device=loss.device, dtype=loss.dtype)
                    aux_lambda = getattr(self.config, 'cclblock_aux_lambda', 0.1)
                    loss = loss + aux_lambda * aux_loss

            # OCD overlap penalty from RemixedLinear layers (if enabled)
            orth_lambda = float(getattr(self.config, 'cclblock_orth_lambda', 0.0))
            if orth_lambda > 0.0 and loss_reduction == 'mean':
                orth_terms = []
                for mod in self.modules():
                    if isinstance(mod, (RemixedLinear, RemixedLinearFused)) and mod._last_orth_loss is not None:
                        orth_terms.append(mod._last_orth_loss.to(dtype=loss.dtype))
                if orth_terms:
                    loss = loss + orth_lambda * torch.stack(orth_terms).mean()

            # Phase 35: template diversity regularization — penalizes template collapse.
            div_lambda = float(getattr(self.config, 'p35_template_diversity_lambda', 0.0))
            if div_lambda > 0.0 and loss_reduction == 'mean':
                div_terms = []
                for mod in self.modules():
                    if isinstance(mod, RemixedLinear) and mod._template_diversity_loss is not None:
                        div_terms.append(mod._template_diversity_loss.to(dtype=loss.dtype))
                if div_terms:
                    loss = loss + div_lambda * torch.stack(div_terms).mean()

            # Phase 24: sliced-weight routing balance loss
            # SCH: load balancing for a sparsely routed code head. Without it
            # top-1 routing concentrates on whichever component wins early, and
            # the sweep pays for K subspaces while using one.
            sch_aux = float(getattr(self.config, 'sch_mixture_aux', 0.0))
            if (sch_aux > 0.0 and loss_reduction == 'mean' and self.use_code_head
                    and getattr(self.lm_head, '_aux_loss', None) is not None):
                loss = loss + sch_aux * self.lm_head._aux_loss.to(dtype=loss.dtype)

            p24_balance_coeff = float(getattr(self.config, 'p24_sliced_weight_balance_coeff', 0.0))
            if p24_balance_coeff > 0.0 and loss_reduction == 'mean':
                bal_terms = []
                for mod in self.modules():
                    if isinstance(mod, SlicedWeightLinear) and mod._last_balance_loss is not None:
                        bal_terms.append(mod._last_balance_loss.to(dtype=loss.dtype))
                if bal_terms:
                    loss = loss + p24_balance_coeff * torch.stack(bal_terms).mean()

            # 18G: Auxiliary representation similarity loss
            # Penalize high cosine similarity between adjacent layers
            sim_lambda = float(getattr(self.config, 'p18_aux_sim_lambda', 0.0))
            if sim_lambda > 0.0 and loss_reduction == 'mean' and hasattr(self, '_layer_outputs'):
                sim_losses = []
                for i in range(len(self._layer_outputs) - 1):
                    h_curr = self._layer_outputs[i].float()
                    h_next = self._layer_outputs[i + 1].float()
                    # Mean cosine similarity across batch and sequence
                    cos_sim = F.cosine_similarity(h_curr, h_next, dim=-1).mean()
                    sim_losses.append(cos_sim)
                if sim_losses:
                    loss = loss + sim_lambda * torch.stack(sim_losses).mean()

            # 19H: Anti-Collapse Weight Norm Penalty
            # Penalizes high cosine similarity between adjacent layers' WEIGHT matrices
            # (different from 18G which penalized activations — weights being similar means redundant layers)
            ac_lambda = float(getattr(self.config, 'p19_weight_anticollapse', 0.0))
            if ac_lambda > 0.0 and loss_reduction == 'mean':
                ac_losses = []
                blocks = list(self.transformer.h)
                for j in range(len(blocks) - 1):
                    b_curr, b_next = blocks[j], blocks[j + 1]
                    # Get MLP c_fc weights (the largest weight matrices)
                    w_curr = b_curr.mlp.c_fc.weight if hasattr(b_curr, 'mlp') else (b_curr.ffwd.c_fc.weight if hasattr(b_curr, 'ffwd') else None)
                    w_next = b_next.mlp.c_fc.weight if hasattr(b_next, 'mlp') else (b_next.ffwd.c_fc.weight if hasattr(b_next, 'ffwd') else None)
                    if w_curr is not None and w_next is not None:
                        sim = F.cosine_similarity(w_curr.flatten().unsqueeze(0).float(),
                                                  w_next.flatten().unsqueeze(0).float(), dim=-1)
                        # Penalize only when similarity exceeds threshold (0.8)
                        ac_losses.append(F.relu(sim - 0.8))
                if ac_losses:
                    loss = loss + ac_lambda * torch.stack(ac_losses).mean()

            return loss
        else:
            # inference: just return the logits directly
            if return_hidden:
                return logits, x
            return logits

    # ------------------------------------------------------------------ SAP block head
    def _sap_readout(self, s):
        """Slot states (rows, d) -> softcapped logits (rows, V), exactly as the main head."""
        logits = self.lm_head(norm(s))[..., :self.config.vocab_size].float()
        return 20.0 * torch.tanh(logits / 20.0)

    def _sap_window(self, x, b, t):
        """The W trunk states ending at position t of row b, front-padded. (N, W, d), (N, W)."""
        W = self.sap_head.W
        if W == 0:
            return None, None
        offs = torch.arange(W - 1, -1, -1, device=x.device)
        pos = t[:, None] - offs[None, :]
        valid = pos >= 0
        return x[b[:, None], pos.clamp_min(0)], valid

    def _sap_block_loss(self, x, targets, logits):
        """Block loss on a random sap_block_frac of the positions of this batch.

        x: (B, Tq, d) final normalised hidden states; targets: (B, Tq) where
        targets[:, j] is the token after position j; logits: next-token logits (only the
        inv_head competitor reads them, to invert the data tokens into noise).
        """
        head = self.sap_head
        B, Tq, _ = x.shape
        Tb = head.T
        n_start = Tq - Tb + 1
        assert n_start > 0, f"sap_block_T={Tb} longer than the sequence ({Tq})"
        N = max(1, int(round(head.frac * B * Tq)))
        flat = torch.randint(0, B * n_start, (N,), device=x.device)
        b, t = flat // n_start, flat % n_start
        ks = torch.arange(Tb, device=x.device)
        y = targets[b[:, None], t[:, None] + ks[None, :]]                  # (N, Tb)
        # sap_trunk_grad scales the gradient the block loss sends into everything the dense
        # model owns: the trunk states, and the shared lm_head and wte the head reads through.
        # 0 leaves the trunk exactly the dense model's (the head trains on what it is given).
        g = float(getattr(self.config, 'sap_trunk_grad', 1.0))

        def _gs(v):
            if v is None or g == 1.0:
                return v
            return v.detach() if g == 0.0 else v.detach() + g * (v - v.detach())

        h = _gs(x[b, t])                                                    # (N, d)
        ctx, ctx_valid = self._sap_window(x, b, t)
        ctx = _gs(ctx)
        h_future = x[b, (t + Tb - 1).clamp(max=Tq - 1)].detach()           # only p1_selfpost reads it
        readout, embed, embed_weight = self._sap_readout, self.transformer.wte, None
        if g != 1.0:
            W_out, V = _gs(self.lm_head.weight), self.config.vocab_size
            embed_weight = _gs(self.transformer.wte.weight)

            def readout(s):
                logits = F.linear(norm(s), W_out.to(dtype=s.dtype))[..., :V].float()
                return 20.0 * torch.tanh(logits / 20.0)

            embed = partial(F.embedding, weight=embed_weight)
        u_bins = None
        if head.mode == 'inv_head':
            from nanochat.block_head import target_bins
            with torch.no_grad():
                fb = b[:, None].expand(N, Tb - 1).reshape(-1)
                ft = (t[:, None] + ks[None, :Tb - 1]).reshape(-1)
                yy = y[:, :Tb - 1].reshape(-1).clamp_min(0)
                lo, width = [], []
                for i in range(0, fb.numel(), 4096):
                    l_i, w_i = target_bins(logits[fb[i:i + 4096], ft[i:i + 4096]], yy[i:i + 4096])
                    lo.append(l_i)
                    width.append(w_i)
                u_bins = (torch.cat(lo).view(N, Tb - 1), torch.cat(width).view(N, Tb - 1)) \
                    if lo else (x.new_zeros(N, 0, dtype=torch.float32),) * 2
        # The lattice and cut heads read slot 0 out of the trunk state, which is exactly the
        # next-token logit row the trunk just computed: reuse it (same gradient scaling).
        slot0 = None if logits is None else _gs(logits[b, t])
        return head.loss(h, ctx, ctx_valid, y, readout, embed=embed, u_bins=u_bins,
                         h_future=h_future, embed_weight=embed_weight, slot0_logits=slot0)

    # ------------------------------------------------------------------ SAP depth slots
    @torch.no_grad()
    def sap_sync_depth_copies(self):
        """Copy the trunk's top layers into depth_local's trainable copies (a no-op otherwise)."""
        head = getattr(self, 'sap_head', None)
        if head is None or not hasattr(head, 'depth_blocks'):
            return
        for blk, i in zip(head.depth_blocks, head.depth_layer_ids):
            blk.load_state_dict(self.transformer.h[i].state_dict())

    def _sap_depth_tools(self):
        """(gs, lin, blocks): gs scales a trunk-owned tensor's gradient by sap_trunk_grad in
        training; lin applies a block linear (gradient-scaled when the slots share the trunk's
        blocks); blocks are the m layers the slots run through (copies or the trunk's own)."""
        head = self.sap_head
        g = float(getattr(self.config, 'sap_trunk_grad', 1.0)) if torch.is_grad_enabled() else 1.0

        def gs(v):
            if g == 1.0:
                return v
            return v.detach() if g == 0.0 else v.detach() + g * (v - v.detach())

        shared = not hasattr(head, 'depth_blocks')

        def lin(mod, x):
            w = gs(mod.weight) if shared else mod.weight
            return F.linear(x, w.to(dtype=x.dtype))

        blocks = [self.transformer.h[i] for i in head.depth_layer_ids] if shared else list(head.depth_blocks)
        return gs, lin, blocks

    def _sap_depth_layers(self, xs, x0s, slot_ids, slot_is_tok, pos, prefix_kv, pre_mask, slot_vis,
                          gs, lin, blocks, pre_pos=None, kv_out=None, mid=None, layer_ids=None):
        """Run Q slots through the top m layers, each layer reproducing a trunk block.

        xs, x0s (B, Q, d): entry state and the slot's x0 stream; slot_ids (B, Q) token ids for
        the value embeddings, zeroed where slot_is_tok (B, Q) is False (mask slots); pos (B, Q)
        absolute positions (RoPE); prefix_kv(j, i, kv_of, rl, xl) -> (kp, vp) the prefix's keys
        and values at layer i, (B, Tp, Hkv, hd); pre_mask (B, Q, Tp) which prefix keys a slot may
        read (the layer's sliding window is applied here); slot_vis (1 or B, Q, Q) which slots a
        slot may read. pre_pos (B, Tp): the prefix keys' positions for the window, default
        0..Tp-1. kv_out: a list that receives each layer's slot (keys, values). mid (j, delta):
        added to the residual before the j-th of the m layers (skip-middle slots: the block
        start's middle-layer delta). layer_ids: the trunk layer index of each block (default:
        the block head's depth layers; lane decoding passes every layer). Returns xs after the m
        layers.
        """
        B, Q, d = xs.shape
        cos_s, sin_s = self.cos[0][pos], self.sin[0][pos]                  # (B, Q, 1, hd/2)
        Tp = pre_mask.size(-1)
        kpos = torch.arange(Tp, device=xs.device)[None] if pre_pos is None else pre_pos   # (1 or B, Tp)
        on = slot_is_tok[..., None, None]
        ids = self.sap_head.depth_layer_ids if layer_ids is None else layer_ids
        for j, (blk_mod, i) in enumerate(zip(blocks, ids)):
            if mid is not None and j == mid[0]:
                xs = xs + mid[1]
            rl, xl = gs(self.resid_lambdas[i]), gs(self.x0_lambdas[i])
            xs = rl * xs + xl * x0s
            attn = blk_mod.attn
            assert attn._attn_moe_k == 0 and attn.head_importance is None and attn.attn_logit_scale is None, \
                "depth slots reproduce the plain trunk block only"
            H, Hkv, hd = attn.n_head, attn.n_kv_head, attn.head_dim
            has_v = str(i) in self.value_embeds and attn.ve_gate is not None
            ve_w = gs(self.value_embeds[str(i)].weight) if has_v else None

            def kv_of(xn, ids, cos, sin, Tn, mask_on=None):
                kx = lin(attn.c_k, xn).view(B, Tn, Hkv, hd)
                vx = lin(attn.c_v, xn).view(B, Tn, Hkv, hd)
                if has_v:
                    ve = F.embedding(ids, ve_w).to(xn.dtype).view(B, Tn, Hkv, hd)
                    if mask_on is not None:
                        ve = ve * mask_on.to(ve.dtype)
                    vx = vx + 2 * torch.sigmoid(lin(attn.ve_gate, xn[..., :attn.ve_gate_channels])).unsqueeze(-1) * ve
                return norm(apply_rotary_emb(kx, cos, sin)), vx

            hn = norm(xs)
            q = norm(apply_rotary_emb(lin(attn.c_q, hn).view(B, Q, H, hd), cos_s, sin_s))
            ks, vs = kv_of(hn, slot_ids, cos_s, sin_s, Q, on)
            if kv_out is not None:
                kv_out.append((ks, vs))
            kp, vp = prefix_kv(j, i, kv_of, rl, xl)
            pre = pre_mask
            left = self.window_sizes[i][0]
            if left is not None and left >= 0:
                pre = pre & (kpos[:, None, :] > pos[..., None] - left)
            mask = torch.cat([pre, slot_vis.expand(B, Q, Q)], dim=-1)
            kall = torch.cat([kp.to(ks.dtype), ks], dim=1).to(q.dtype)
            vall = torch.cat([vp.to(vs.dtype), vs], dim=1).to(q.dtype)
            if Hkv != H:
                kall = kall.repeat_interleave(H // Hkv, dim=2)
                vall = vall.repeat_interleave(H // Hkv, dim=2)
            att = F.scaled_dot_product_attention(q.transpose(1, 2), kall.transpose(1, 2), vall.transpose(1, 2),
                                                 attn_mask=mask[:, None])
            xs = xs + lin(attn.c_proj, att.transpose(1, 2).reshape(B, Q, d))
            mlp = blk_mod.mlp
            if mlp is not None:
                assert type(mlp).__name__ == "MLP" and mlp.dynamic_act is None and mlp.srp_proj is None \
                    and mlp.ln_intermediate is None, "depth slots reproduce the plain MLP only"
                xs = xs + lin(mlp.c_proj, F.relu(lin(mlp.c_fc, norm(xs))).square())
        return xs

    def _sap_depth_base(self, st):
        """The per-position state a slot's entry is built from, from trunk states st {i: (..., d)}:
        the state entering the first slot layer (late entry), or, with skip-middle slots, the
        middle layers' delta at the block start, state(L-b) - state(a)."""
        head = self.sap_head
        a = head.depth_bottom
        if a > 0:
            return st[head.depth_layer_ids[a]] - st[a]
        return st[head.depth_layer_ids[0]]

    def sap_depth_state(self, states):
        """Decode-time entry base (B, d) at the last position of a trunk pass's captured states."""
        return self._sap_depth_base({i: v[:, -1] for i, v in states.items() if isinstance(i, int)})

    def _sap_depth_entry(self, gs, states_first, t_q, toks, is_tok, offs):
        """Entry of slots: (xs, x0s, mid). Late entry: the trunk's state at the block start t (or
        the slot's own embedding when m = L) plus the token's embedding and its offset. Skip-middle
        (depth_bottom = a > 0): the slot starts exactly as the trunk would (its embedding), runs
        the bottom a layers, and mid = (a, delta) adds the block start's middle-layer delta (plus
        the same learned entry terms) before the top layers. states_first is _sap_depth_base."""
        head = self.sap_head
        first = head.depth_layer_ids[0]
        dt = states_first.dtype
        B, Q = toks.shape
        d = self.config.n_embd
        emb = F.embedding(toks, gs(self.transformer.wte.weight)).to(dt)
        if hasattr(head, "depth_mask"):
            emb = torch.where(is_tok[..., None], emb, head.depth_mask.to(dt)[None, None].expand(B, Q, d))
        x0s = norm(emb)
        if states_first.dim() == 3:                                       # training: (B, Tq, d)
            base = states_first.gather(1, t_q[..., None].expand(B, Q, d))
        else:                                                             # decode: (B, d) at t
            base = states_first[:, None].expand(B, Q, d)
        learned = head.depth_in(x0s) + head.depth_slot[offs - 1].to(dt)
        if head.depth_bottom > 0:
            return x0s, x0s, (head.depth_bottom, base + learned)
        xs = x0s if first == 0 else base
        return xs + learned, x0s, None

    def _sap_depth_readout(self, gs):
        V, W_out = self.config.vocab_size, gs(self.lm_head.weight)

        def readout(z):
            logits = F.linear(norm(z), W_out.to(dtype=z.dtype))[..., :V].float()
            return 20.0 * torch.tanh(logits / 20.0)
        return readout

    def _sap_depth_logprob(self, idx, states, slot0_logits, y, starts, valid, return_slot_logits=False):
        """Exact log p(block) through the top m trunk layers; (B, n). See the two layouts below."""
        if self.sap_head.mode == "depth_tree":
            return self._sap_depth_tree_logprob(idx, states, slot0_logits, y, starts, valid, return_slot_logits)
        if self.sap_head.mode == "depth_roll":
            return self._sap_depth_roll_logprob(idx, states, slot0_logits, y, starts, valid, return_slot_logits)
        return self._sap_depth_local_logprob(idx, states, slot0_logits, y, starts, valid, return_slot_logits)

    def _sap_depth_prefix_train(self, idx, st, x0t):
        """prefix_kv for training/eval: the trunk's keys/values at layer i over the whole row."""
        Tq = idx.size(1)
        cos_p, sin_p = self.cos[:, :Tq], self.sin[:, :Tq]

        def prefix_kv(j, i, kv_of, rl, xl):
            return kv_of(norm(rl * st[i] + xl * x0t), idx, cos_p, sin_p, Tq)
        return prefix_kv

    def _sap_depth_local_logprob(self, idx, states, slot0_logits, y, starts, valid, return_slot_logits=False):
        """depth_local: exact autoregressive factorisation of the block through the top m layers.

        idx (B, Tq) the input ids; states {i: (B, Tq, d)} the residual stream entering trunk
        layer i (before its resid/x0 mixing), for the top m layers; slot0_logits (B, n, V) the
        next-token logits at the starts; y (B, n, T) block targets; starts (B, n) block starts
        (position t, whose next token is y[..., 0]); valid (B, n, T). Returns (B, n).

        Slot 0 is the trunk's own next-token distribution. Slot k >= 1 is the block's token
        y[..., k-1] placed at position t+k: it enters layer L-m from the trunk's state at t (or,
        when m = L, from its own embedding, exactly as the trunk does), then each layer
        reproduces a trunk block (resid/x0 scalars, value embedding, RoPE at its true position,
        QK norm) and attends to the trunk's keys/values at positions <= t and to the block's
        earlier slots. With m = L and the trunk's own weights it is the trunk's teacher-forced
        likelihood. In training, every trunk-owned tensor on this path passes sap_trunk_grad of
        its gradient.
        """
        head = self.sap_head
        B, n, T = y.shape
        d = self.config.n_embd
        gs, lin, blocks = self._sap_depth_tools()
        lp0 = torch.log_softmax(slot0_logits.float(), -1).gather(-1, y[..., :1].clamp_min(0)).squeeze(-1)
        ll = lp0 * valid[..., 0]
        if T == 1 and not return_slot_logits:
            return ll
        K = T - 1
        Q = n * K
        dev = idx.device
        assert not (self._use_residual_decay and self.depth_decay_raw is not None), "x0 decay unsupported"
        tok = y[..., :K].clamp_min(0).reshape(B, Q)                       # slot k's token: y[k-1]
        kk = torch.arange(1, T, device=dev).repeat(n)                     # (Q,) slot offset
        blk = torch.arange(n, device=dev).repeat_interleave(K)            # (Q,) block of each slot
        t_q = starts.repeat_interleave(K, dim=1)                          # (B, Q) block start
        pos = t_q + kk[None, :]
        st = {i: gs(states[i]) for i in head.depth_capture_ids}
        is_tok = torch.ones(B, Q, dtype=torch.bool, device=dev)
        xs, x0s, mid = self._sap_depth_entry(gs, self._sap_depth_base(st), t_q, tok, is_tok, kk[None].expand(B, Q))
        x0t = norm(F.embedding(idx, gs(self.transformer.wte.weight)).to(xs.dtype))
        pre = torch.arange(idx.size(1), device=dev)[None, None, :] <= t_q[..., None]
        vis = ((blk[:, None] == blk[None, :]) & (kk[None, :] <= kk[:, None]))[None]
        xs = self._sap_depth_layers(xs, x0s, tok, is_tok, pos, self._sap_depth_prefix_train(idx, st, x0t),
                                    pre, vis, gs, lin, blocks, mid=mid)
        readout = self._sap_depth_readout(gs)
        if return_slot_logits:                       # (B, n, K, V): slot k's next-token logits
            return readout(norm(xs).reshape(B * Q, d)).view(B, n, K, -1)
        from nanochat.block_head import _target_logprob
        lp = _target_logprob(norm(xs).reshape(B * Q, d), y[..., 1:].clamp_min(0).reshape(B * Q),
                             readout, head.chunk).view(B, n, K)
        return ll + (lp * valid[..., 1:]).sum(-1)

    def _sap_depth_roll_logprob(self, idx, states, slot0_logits, y, starts, valid, return_slot_logits=False):
        """depth_roll: depth_local whose slot k also enters with the previous slot's top-layer
        state (slot 1: the trunk's final state at t, states["top"]), through head.depth_fb.

        That state is what produced y_k's distribution, so decoding gets it for free; in exchange
        the block's newest token reaches the next prediction through 2m layers rather than m. Still
        the exact chain rule; training runs the T-1 slots in sequence, each reading the prefix and
        the earlier slots' keys/values at every layer. Returns (B, n), or with return_slot_logits
        slot k's next-token logits, (B, n, T-1, V).
        """
        head = self.sap_head
        B, n, T = y.shape
        d = self.config.n_embd
        dev = idx.device
        gs, lin, blocks = self._sap_depth_tools()
        lp0 = torch.log_softmax(slot0_logits.float(), -1).gather(-1, y[..., :1].clamp_min(0)).squeeze(-1)
        ll = lp0 * valid[..., 0]
        if T == 1 and not return_slot_logits:
            return ll
        assert not (self._use_residual_decay and self.depth_decay_raw is not None), "x0 decay unsupported"
        first = head.depth_layer_ids[0]
        st = {i: gs(states[i]) for i in head.depth_capture_ids}
        base = self._sap_depth_base(st)
        Tq = idx.size(1)
        x0t = norm(F.embedding(idx, gs(self.transformer.wte.weight)).to(st[first].dtype))
        cos_p, sin_p = self.cos[:, :Tq], self.sin[:, :Tq]
        pre_kv, past = {}, [[] for _ in blocks]           # trunk prefix K/V; earlier slots' K/V per layer

        def prefix_kv(j, i, kv_of, rl, xl):
            if j not in pre_kv:
                pre_kv[j] = kv_of(norm(rl * st[i] + xl * x0t), idx, cos_p, sin_p, Tq)
            kp, vp = pre_kv[j]
            return (torch.cat([kp] + [a for a, _ in past[j]], 1),
                    torch.cat([vp] + [b for _, b in past[j]], 1))

        prev = gs(states["top"]).gather(1, starts[..., None].expand(B, n, d))     # (B, n, d)
        same = torch.eye(n, dtype=torch.bool, device=dev)[None].expand(B, n, n)   # a block's own slots
        pre0 = torch.arange(Tq, device=dev)[None, None, :] <= starts[..., None]    # (B, n, Tq)
        pos0 = torch.arange(Tq, device=dev)[None].expand(B, Tq)
        is_tok = torch.ones(B, n, dtype=torch.bool, device=dev)
        readout = self._sap_depth_readout(gs)
        from nanochat.block_head import _target_logprob
        slot_logits = []
        for k in range(1, T):
            tok = y[..., k - 1].clamp_min(0)                                      # (B, n): y_k
            offs = torch.full((B, n), k, dtype=torch.long, device=dev)
            xs, x0s, mid = self._sap_depth_entry(gs, base, starts, tok, is_tok, offs)
            fb = head.depth_fb(norm(prev))
            if mid is None:
                xs = xs + fb
            else:
                mid = (mid[0], mid[1] + fb)
            pre = torch.cat([pre0] + [same] * (k - 1), dim=-1)
            kpos = torch.cat([pos0] + [starts + kk for kk in range(1, k)], dim=1)
            cur = []
            xs = self._sap_depth_layers(xs, x0s, tok, is_tok, starts + k, prefix_kv, pre, same,
                                        gs, lin, blocks, pre_pos=kpos, kv_out=cur, mid=mid)
            for j, kv in enumerate(cur):
                past[j].append(kv)
            if return_slot_logits:
                slot_logits.append(readout(norm(xs).reshape(B * n, d)).view(B, n, -1))
            else:
                lp = _target_logprob(norm(xs).reshape(B * n, d), y[..., k].clamp_min(0).reshape(-1),
                                     readout, head.chunk).view(B, n)
                ll = ll + lp * valid[..., k]
            prev = xs
        if return_slot_logits:
            return torch.stack(slot_logits, dim=2)
        return ll

    def _sap_depth_tree_logprob(self, idx, states, slot0_logits, y, starts, valid, return_slot_logits=False):
        """depth_tree: exact likelihood of the bisection factorisation through the top m layers.

        log p(y) = log p(y_1 | ctx) + sum over rounds of sum over that round's queried y_j of
        log p(y_j | ctx, tokens committed in earlier rounds). Each round runs the committed
        tokens' slots (and mask slots where a queried token's left neighbour is unknown) through
        the top m layers with every slot of the round visible to every other; nothing a round
        queries is an input to it. Returns (B, n), or with return_slot_logits the logits each
        y_j (j = 2..T) is read from, (B, n, T-1, V).
        """
        head = self.sap_head
        B, n, T = y.shape
        d = self.config.n_embd
        dev = idx.device
        gs, lin, blocks = self._sap_depth_tools()
        lp0 = torch.log_softmax(slot0_logits.float(), -1).gather(-1, y[..., :1].clamp_min(0)).squeeze(-1)
        ll = lp0 * valid[..., 0]
        first = head.depth_layer_ids[0]
        st = {i: gs(states[i]) for i in head.depth_capture_ids}
        base = self._sap_depth_base(st)
        x0t = norm(F.embedding(idx, gs(self.transformer.wte.weight)).to(st[first].dtype))
        prefix = self._sap_depth_prefix_train(idx, st, x0t)
        readout = self._sap_depth_readout(gs)
        y_safe = y.clamp_min(0)
        from nanochat.block_head import _target_logprob, depth_tree_layout
        per_j = {}
        for slots, tokflag, read in depth_tree_layout(T):
            S = len(slots)
            offs = torch.tensor(slots, device=dev)
            flag = torch.tensor(tokflag, device=dev)
            Q = n * S
            o_q = offs.repeat(n)                                          # (Q,)
            t_q = starts.repeat_interleave(S, dim=1)                      # (B, Q)
            pos = t_q + o_q[None, :]
            tok = y_safe.index_select(2, offs - 1).reshape(B, Q)          # y_o (offset o -> y[o-1])
            is_tok = flag.repeat(n)[None].expand(B, Q)
            blk = torch.arange(n, device=dev).repeat_interleave(S)
            xs, x0s, mid = self._sap_depth_entry(gs, base, t_q, tok, is_tok, o_q[None].expand(B, Q))
            pre = torch.arange(idx.size(1), device=dev)[None, None, :] <= t_q[..., None]
            vis = (blk[:, None] == blk[None, :])[None]
            xs = self._sap_depth_layers(xs, x0s, tok, is_tok, pos, prefix, pre, vis, gs, lin, blocks, mid=mid)
            xs = xs.view(B, n, S, d)
            si = torch.tensor([a for a, _ in read], device=dev)
            jj = torch.tensor([j for _, j in read], device=dev)
            h = xs.index_select(2, si)                                    # (B, n, R, d)
            if return_slot_logits:
                lg = readout(norm(h).reshape(-1, d)).view(B, n, len(read), -1)
                per_j.update({j: lg[:, :, r] for r, (_, j) in enumerate(read)})
                continue
            tgt = y_safe.index_select(2, jj - 1)                          # (B, n, R)
            lp = _target_logprob(norm(h).reshape(-1, d), tgt.reshape(-1), readout, head.chunk).view(tgt.shape)
            ll = ll + (lp * valid.index_select(2, jj - 1)).sum(-1)
        if return_slot_logits:
            return torch.stack([per_j[j] for j in range(2, T + 1)], dim=2)
        return ll

    def sap_depth_copy_cache(self, batch_size, seq_len, device, dtype):
        """KV buffers for depth_local's trained copies, or None when the slots share the trunk's
        layers (then the trunk's own KV cache is the right one). (m, B, Tmax, Hkv, hd) each."""
        head = self.sap_head
        if not hasattr(head, 'depth_blocks'):
            return None
        a = head.depth_blocks[0].attn
        shape = (len(head.depth_blocks), batch_size, seq_len, a.n_kv_head, a.head_dim)
        return torch.zeros(shape, device=device, dtype=dtype), torch.zeros(shape, device=device, dtype=dtype)

    @torch.no_grad()
    def sap_depth_extend_copy_cache(self, copy_kv, new_ids, states, kv_cache):
        """Append the copies' own keys/values for the tokens the trunk just cached. Training
        computes prefix keys with the copies' projections of the trunk's residual stream, so
        decoding must too; the trunk's cache holds the original layers' keys. Graph-safe."""
        if copy_kv is None:
            return
        head = self.sap_head
        B, Tn = new_ids.shape
        end = kv_cache.cache_seqlens.to(torch.long)                       # after this pass
        pos = end[:, None] - Tn + torch.arange(Tn, device=new_ids.device)  # (B, Tn)
        dt = states[head.depth_layer_ids[0]].dtype
        x0n = norm(self.transformer.wte(new_ids).to(dt))
        cos, sin = self.cos[0][pos], self.sin[0][pos]                      # (B, Tn, 1, hd/2)
        for j, (blk_mod, i) in enumerate(zip(head.depth_blocks, head.depth_layer_ids)):
            a = blk_mod.attn
            xt = self.resid_lambdas[i] * states[i] + self.x0_lambdas[i] * x0n
            has_v = str(i) in self.value_embeds and a.ve_gate is not None
            ve = self.value_embeds[str(i)](new_ids).to(dt) if has_v else None
            k, v = a._project_kv(norm(xt), ve, cos, sin)
            idx = pos[..., None, None].expand(B, Tn, a.n_kv_head, a.head_dim)
            copy_kv[0][j].scatter_(1, idx, k.to(copy_kv[0].dtype))
            copy_kv[1][j].scatter_(1, idx, v.to(copy_kv[1].dtype))

    @torch.no_grad()
    def _sap_depth_sample(self, kv_cache, state_first, slot0_logits, temperature=1.0, generator=None,
                          copy_kv=None, top_first=None):
        """Decode one depth_local block: T tokens, the first from the trunk's next-token logits,
        each later one from a single-token pass through the top m layers that reads the trunk's
        KV cache (positions <= t) and the block's earlier slots. Graph-safe: positions and masks
        come from the cache on the device. state_first (B, d): sap_depth_state of the last trunk
        pass (the pre-mix residual entering layer L-m at the last cached position t, or with
        skip-middle slots the middle layers' delta there). top_first (B, d): the trunk's final (normed)
        state at t, depth_roll's feedback into slot 1. Returns (B, T) long."""
        from nanochat.block_head import pick
        head = self.sap_head
        if head.mode == "depth_tree":
            return self._sap_depth_tree_sample(kv_cache, state_first, slot0_logits, temperature, generator, copy_kv)
        T, dev = head.T, slot0_logits.device
        B, d = state_first.shape
        first = head.depth_layer_ids[0]
        blocks = head.depth_blocks if hasattr(head, 'depth_blocks') else [self.transformer.h[i] for i in head.depth_layer_ids]

        def _choose(logits):
            if temperature <= 0:
                return logits.argmax(-1)
            probs = torch.softmax(logits.float() / temperature, dim=-1)
            return pick(probs, torch.rand(B, device=dev, generator=generator))

        out = torch.zeros(B, T, dtype=torch.long, device=dev)
        out[:, 0] = _choose(slot0_logits)
        if T == 1:
            return out
        t = kv_cache.cache_seqlens.to(torch.long) - 1                     # (B,) last cached position
        Tmax = kv_cache.k_cache.size(2)
        kpos = torch.arange(Tmax, device=dev)
        H0 = blocks[0].attn
        kbuf = [torch.zeros(B, T - 1, H0.n_kv_head, H0.head_dim, device=dev, dtype=kv_cache.k_cache.dtype) for _ in blocks]
        vbuf = [torch.zeros_like(kb) for kb in kbuf]
        slot_idx = torch.arange(T - 1, device=dev)
        dt = state_first.dtype
        roll = head.mode == "depth_roll"
        assert not roll or top_first is not None, "depth_roll decoding needs the trunk's final state"
        a = head.depth_bottom
        prev = top_first
        for k in range(1, T):
            tok = out[:, k - 1]
            pos = t + k                                                   # (B,)
            x0s = norm(self.transformer.wte(tok).to(dt))
            learned = head.depth_in(x0s) + head.depth_slot[k - 1].to(dt)
            if roll:
                learned = learned + head.depth_fb(norm(prev).to(dt))
            if a > 0:            # skip-middle: exact bottom layers, then the block start's middle delta
                xs, delta = x0s, state_first + learned
            else:
                xs = (x0s if first == 0 else state_first) + learned
            cos_s = self.cos[0][pos][:, None]                             # (B, 1, 1, hd/2)
            sin_s = self.sin[0][pos][:, None]
            for j, (blk_mod, i) in enumerate(zip(blocks, head.depth_layer_ids)):
                if a > 0 and j == a:
                    xs = xs + delta
                xs = self.resid_lambdas[i] * xs + self.x0_lambdas[i] * x0s
                attn = blk_mod.attn
                H, Hkv, hd = attn.n_head, attn.n_kv_head, attn.head_dim
                hn = norm(xs)[:, None]                                    # (B, 1, d)
                q = norm(apply_rotary_emb(attn.c_q(hn).view(B, 1, H, hd), cos_s, sin_s))
                kk_ = attn.c_k(hn).view(B, 1, Hkv, hd)
                vv_ = attn.c_v(hn).view(B, 1, Hkv, hd)
                if str(i) in self.value_embeds and attn.ve_gate is not None:
                    ve = self.value_embeds[str(i)](tok).to(xs.dtype).view(B, 1, Hkv, hd)
                    vv_ = vv_ + 2 * torch.sigmoid(attn.ve_gate(hn[..., :attn.ve_gate_channels])).unsqueeze(-1) * ve
                kk_ = norm(apply_rotary_emb(kk_, cos_s, sin_s))
                kbuf[j][:, k - 1] = kk_[:, 0].to(kbuf[j].dtype)
                vbuf[j][:, k - 1] = vv_[:, 0].to(vbuf[j].dtype)
                if copy_kv is not None:
                    kc, vc = copy_kv[0][j], copy_kv[1][j]                  # the copies' own keys
                else:
                    kc, vc = kv_cache.get_layer_cache(i)                   # (B, Tmax, Hkv, hd)
                pre = kpos[None, :] <= t[:, None]                          # (B, Tmax)
                left = self.window_sizes[i][0]
                if left is not None and left >= 0:
                    pre = pre & (kpos[None, :] > pos[:, None] - left)
                own = (slot_idx < k)[None].expand(B, T - 1)                # slots 1..k (self included)
                mask = torch.cat([pre, own], dim=-1)[:, None, None, :]     # (B, 1, 1, Tmax+T-1)
                kall = torch.cat([kc, kbuf[j]], dim=1).to(q.dtype)
                vall = torch.cat([vc, vbuf[j]], dim=1).to(q.dtype)
                if Hkv != H:
                    kall = kall.repeat_interleave(H // Hkv, dim=2)
                    vall = vall.repeat_interleave(H // Hkv, dim=2)
                att = F.scaled_dot_product_attention(q.transpose(1, 2), kall.transpose(1, 2),
                                                     vall.transpose(1, 2), attn_mask=mask)
                xs = xs + attn.c_proj(att.transpose(1, 2).reshape(B, d))
                if blk_mod.mlp is not None:
                    xs = xs + blk_mod.mlp(norm(xs))
            out[:, k] = _choose(self._sap_readout(norm(xs)))
            prev = xs
        return out

    @torch.no_grad()
    def _sap_depth_tree_sample(self, kv_cache, state_first, slot0_logits, temperature=1.0, generator=None,
                               copy_kv=None):
        """Decode one depth_tree block: y_1 from the trunk's next-token logits, then one pass of
        each bisection round's slots through the top m layers (depth_tree_layout), reading the
        KV cache at positions <= t (the copies' own cache when the slots use trained copies).
        Graph-safe: indices are Python ints and fills, positions come from the cache. (B, T)."""
        from nanochat.block_head import depth_tree_layout, pick
        head = self.sap_head
        T, dev = head.T, slot0_logits.device
        B, d = state_first.shape
        gs, lin, blocks = self._sap_depth_tools()

        def _choose(logits):                                              # (B, R, V) -> (B, R)
            if temperature <= 0:
                return logits.argmax(-1)
            probs = torch.softmax(logits.float() / temperature, dim=-1)
            return pick(probs, torch.rand(logits.shape[:-1], device=dev, generator=generator))

        out = torch.zeros(B, T, dtype=torch.long, device=dev)
        out[:, 0] = _choose(slot0_logits[:, None])[:, 0]
        if T == 1:
            return out
        t = kv_cache.cache_seqlens.to(torch.long) - 1                     # (B,) last cached position
        pre = torch.arange(kv_cache.k_cache.size(2), device=dev)[None, :] <= t[:, None]

        def prefix_kv(j, i, kv_of, rl, xl):
            return (copy_kv[0][j], copy_kv[1][j]) if copy_kv is not None else kv_cache.get_layer_cache(i)

        for slots, tokflag, read in depth_tree_layout(T):
            S = len(slots)
            tok = torch.stack([out[:, o - 1] for o in slots], dim=1)       # (B, S); masked where unknown
            # torch.full, not t[a] = o: an int setitem copies from the host, which capture forbids.
            is_tok = torch.cat([torch.full((B, 1), f, dtype=torch.bool, device=dev) for f in tokflag], dim=1)
            offs = torch.cat([torch.full((1,), o, dtype=torch.long, device=dev) for o in slots])
            xs, x0s, mid = self._sap_depth_entry(gs, state_first, None, tok, is_tok, offs[None].expand(B, S))
            pos = t[:, None] + offs[None, :]
            vis = torch.ones(1, S, S, dtype=torch.bool, device=dev)
            xs = self._sap_depth_layers(xs, x0s, tok, is_tok, pos, prefix_kv, pre[:, None].expand(B, S, pre.size(-1)),
                                        vis, gs, lin, blocks, mid=mid)
            h = torch.stack([xs[:, a] for a, _ in read], dim=1)            # (B, R, d)
            ys = _choose(self._sap_readout(norm(h)))
            for r, (_, j) in enumerate(read):
                out[:, j - 1] = ys[:, r]
        return out

    def _sap_depth_loss(self, idx, x, targets, logits, states):
        """depth_local's training loss: the same number of block starts as _sap_block_loss, laid
        out as a fixed count per row so each row's slots attend to that row's keys only."""
        head = self.sap_head
        B, Tq, _ = x.shape
        T = head.T
        nps = max(1, int(round(head.frac * Tq)))
        starts = torch.randint(0, Tq - T + 1, (B, nps), device=x.device)
        ks = torch.arange(T, device=x.device)
        y = targets.gather(1, (starts[..., None] + ks).reshape(B, nps * T)).view(B, nps, T)
        slot0 = logits.gather(1, starts[..., None].expand(B, nps, logits.size(-1)))
        g = float(getattr(self.config, 'sap_trunk_grad', 1.0))
        if g != 1.0:                                  # slot 0 is the trunk's own next-token row
            slot0 = slot0.detach() if g == 0.0 else slot0.detach() + g * (slot0 - slot0.detach())
        ll = self._sap_depth_logprob(idx, states, slot0, y, starts, y >= 0)
        loss = -ll.sum() / (y >= 0).sum().clamp_min(1)
        return loss, {"block_nll": loss.detach(), "block_loss": loss.detach()}

    def _sap_last(self, hist, h_last):
        """Head inputs for the next block from the most recent trunk states."""
        W = self.sap_head.W
        if W == 0:
            return h_last, None, None
        n = hist.size(1)
        if n >= W:
            return h_last, hist[:, -W:], torch.ones(hist.size(0), W, dtype=torch.bool, device=hist.device)
        pad = hist.new_zeros(hist.size(0), W - n, hist.size(2))
        valid = torch.cat([torch.zeros(hist.size(0), W - n, dtype=torch.bool, device=hist.device),
                           torch.ones(hist.size(0), n, dtype=torch.bool, device=hist.device)], dim=1)
        return h_last, torch.cat([pad, hist], dim=1), valid

    @torch.inference_mode()
    def generate_block(self, tokens, max_tokens, temperature=1.0, seed=42):
        """Block generation without a KV cache: one trunk pass per T tokens.

        The reference path (it recomputes the prefix every block); engine.generate_block_kv
        is the one to time. Yields the T new token ids of each block. batch size 1.
        """
        assert self.sap_head is not None, "generate_block needs sap_block_T > 0"
        device = self.get_device()
        gen = torch.Generator(device=device)
        gen.manual_seed(seed)
        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        produced = 0
        while produced < max_tokens:
            x = self.forward(ids, skip_logits=True)
            h, ctx, valid = self._sap_last(x, x[:, -1])
            blk = self.sap_head.sample(h, ctx, valid, self._sap_readout, embed=self.transformer.wte,
                                       embed_table=self.transformer.wte.weight,
                                       temperature=temperature, generator=gen)
            ids = torch.cat([ids, blk], dim=1)
            produced += blk.size(1)
            yield blk[0].tolist()

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        """
        Naive autoregressive streaming inference.
        To make it super simple, let's assume:
        - batch size is 1
        - ids and the yielded tokens are simple Python lists and ints
        """
        assert isinstance(tokens, list)
        device = self.get_device()
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
        ids = torch.tensor([tokens], dtype=torch.long, device=device) # add batch dim
        for _ in range(max_tokens):
            logits = self.forward(ids) # (B, T, vocab_size)
            logits = logits[:, -1, :] # (B, vocab_size)
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)
            ids = torch.cat((ids, next_ids), dim=1)
            token = next_ids.item()
            yield token
