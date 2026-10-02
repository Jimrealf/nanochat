import argparse
import sys
import os
import json
import glob
import shutil
import datetime
from pathlib import Path
import signal
import subprocess

import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from scripts._sweep_utils import resolve_runner, estimate_tokens_from_base, model_dims, check_and_prepare_env
from nanochat.checkpoint_manager import find_last_step
from nanochat.chunked_remix_config import ChunkedRemixConfig


RUNNER = resolve_runner()

# ---------------------------------------------------------------------------
# Persistent sweep state helpers
# sweep_state.json lives inside --run-dir and survives re-runs.
# Format:
# {
#   "completed":  { "model_name": {"val_bpb": 1.23, "checkpoint": "...", "ckpt_dir": "..."} },
#   "unfinished": { "model_name": {"ckpt_dir": "...", "started_at": "..."} }
# }
# ---------------------------------------------------------------------------

def _state_path(run_dir_path: Path) -> Path:
    return run_dir_path / "sweep_state.json"

def load_sweep_state(run_dir_path: Path) -> dict:
    p = _state_path(run_dir_path)
    if p.exists():
        try:
            with open(p) as f:
                return json.load(f)
        except Exception:
            pass
    return {"completed": {}, "unfinished": {}}

def save_sweep_state(run_dir_path: Path, state: dict) -> None:
    p = _state_path(run_dir_path)
    with open(p, "w") as f:
        json.dump(state, f, indent=2)


def _last_resumable_step(ckpt_dir):
    """Latest step that can actually be resumed INTO, not merely the latest weights.

    find_last_step globs model_*.pt alone. base_train drops the optimizer from the FINAL
    checkpoint (--save-final-optimizer 0, the default) because a finished run has nothing
    to resume into and the shards are 2-3x the weights. So the newest model_*.pt may have
    no optim_*.pt beside it, and load_checkpoint(load_optimizer=True) would raise on it.
    A completed arm should never be retried, but the sweep scripts do retry an arm whose
    results row is missing, and that path would otherwise crash instead of starting fresh.
    """
    models = glob.glob(os.path.join(ckpt_dir, "model_*.pt"))
    steps = sorted({int(os.path.basename(f).split("_")[-1].split(".")[0]) for f in models},
                   reverse=True)
    for st in steps:
        if glob.glob(os.path.join(ckpt_dir, f"optim_{st:06d}_rank*.pt")):
            return st
    raise FileNotFoundError(
        f"no checkpoint in {ckpt_dir} carries optimizer state, so none can be resumed")


def _iso_flops(args):
    """True when the horizon comes from a FLOPs budget rather than a token budget."""
    return (getattr(args, 'target_flops', 0) or 0) > 0 or (getattr(args, 'target_active_flops', 0) or 0) > 0


def run_training_sweep(args):
    # Ensure environment is ready
    check_and_prepare_env(args)
    
    depth = args.depth
    run_dir = args.run_dir
    run_dir_path = Path(run_dir)
    run_dir_path.mkdir(parents=True, exist_ok=True)

    # Load (or create) persistent state for this run_dir
    state = load_sweep_state(run_dir_path)
    if state["unfinished"]:
        print("="*64)
        print("Unfinished experiments found from a previous run:")
        for name, info in state["unfinished"].items():
            print(f"  [{name}] ckpt_dir = {info.get('ckpt_dir', '?')}")
        print("These will be resumed from their last checkpoint.")
        print("="*64)
    
    if args.target_tokens > 0:
        target_tokens = args.target_tokens
    elif args.target_tokens == 0:
        target_tokens = estimate_tokens_from_base(depth, tokenizer_dir=args.tokenizer_dir)
    else:
        target_tokens = -1

    print("=" * 64)
    print(f"Starting Sweep for Depth {depth}")
    print(f"Target Tokens: {'Auto (per-model param count)' if target_tokens == -1 else f'{target_tokens:,}'}")
    print("=" * 64)

    
    aspect_ratio, head_dim, model_dim, target_dim = model_dims(depth, aspect_ratio=args.aspect_ratio)
    # Explicit head-dim override. model_dims() hardcodes 128; the paper's dense
    # head-dim control needs 32 to match MST's per-stream head dimension.
    if getattr(args, 'head_dim', 0) > 0 and args.head_dim != head_dim:
        print(f"  Overriding head_dim ({head_dim}) with --head-dim {args.head_dim}")
        head_dim = args.head_dim
        assert model_dim % head_dim == 0, (
            f"model_dim {model_dim} not divisible by head_dim {head_dim}")
    if args.research_dim > 0:
        print(f"  Overriding default target_dim ({target_dim}) with --research-dim {args.research_dim}")
        target_dim = args.research_dim
    elif args.research_dim == -1:
        print(f"  Overriding default target_dim ({target_dim}) with full model_dim {model_dim}")
        target_dim = model_dim
    if args.model_dim > 0:
        model_dim = args.model_dim
    max_seq_len = args.sequence_len
    
    device_batch_size = args.device_batch_size if args.device_batch_size > 0 else {4: 8, 8: 32, 16: 16, 24: 8}.get(depth, 16)
    total_batch_size = args.total_batch_size if args.total_batch_size > 0 else 262144
    eval_every = args.eval_every
    log_every = args.log_every
    
    warm_up_ratio = args.warmup_ratio
    adam_beta2 = 0.99
    
    # Common kwargs for all models
    common_args = [
        "--depth", str(depth),
        "--aspect-ratio", str(aspect_ratio),
        "--head-dim", str(head_dim),
        "--model-dim", str(model_dim),
        "--max-seq-len", str(max_seq_len),
        "--device-batch-size", str(device_batch_size),
        "--total-batch-size", str(total_batch_size), # standard for reference
        # An isoFLOP profile fixes the horizon from a FLOPs budget, on either the total
        # or the active axis, and both of those override the token budget entirely.
        "--target-tokens", str(-1 if _iso_flops(args) else target_tokens),
        *(["--target-flops", str(args.target_flops)] if getattr(args, 'target_flops', 0) > 0 else []),
        *(["--target-active-flops", str(args.target_active_flops)] if getattr(args, 'target_active_flops', 0) > 0 else []),
        *(["--target-param-data-ratio", str(args.target_param_data_ratio)] if (args.target_param_data_ratio > 0 and not _iso_flops(args)) else []),
        # An explicit iteration count overrides every budget above in base_train, so it is
        # only forwarded when the caller actually asked for one (short diagnostic runs).
        *(["--num-iterations", str(args.num_iterations)] if getattr(args, 'num_iterations', -1) > 0 else []),
        "--eval-every", str(eval_every),        
        "--log-every", str(log_every),
        "--core-metric-every", "0" if args.skip_core else str(args.core_metric_every),
        "--save-every", str(args.save_every),
        "--warmup-ratio", str(warm_up_ratio),
        "--warmdown-ratio", str(getattr(args, 'warmdown_ratio', 0.5)),
        "--final-lr-frac", str(getattr(args, 'final_lr_frac', 0.0)),    # Safer for research models
        "--adam-beta2", str(adam_beta2),     # Matches notebook
        "--research-warmup-ratio", str(args.research_warmup_ratio),
        "--use-onecycle", str(args.use_onecycle),
        "--router-context-window", str(args.router_context_window),
        "--remix-use-basis-gate", str(getattr(args, 'remix_use_basis_gate', 1)),
        "--remix-use-output-gate", str(getattr(args, 'remix_use_output_gate', 1)),
        "--remix-use-context", str(getattr(args, 'remix_use_context', 1)),
        "--remix-basis-gate-mode", str(getattr(args, 'remix_basis_gate_mode', 'mlp')),
        "--remix-gate-lr-scale", str(getattr(args, 'remix_gate_lr_scale', 0.3)),
        "--p22-n-templates", str(getattr(args, 'p22_n_templates', 1)),
        "--p22-template-routing-learned", str(getattr(args, 'p22_template_routing_learned', 0)),
        "--p22-template-topk", str(getattr(args, 'p22_template_topk', 0)),
        "--p22-attn-moe-route", str(getattr(args, 'p22_attn_moe_route', 'none')),
        "--p26-output-gated-linear", str(getattr(args, 'p26_output_gated_linear', 0)),
        "--p28-shared-basis", str(getattr(args, 'p28_shared_basis', 0)),
        "--p28-chunk-routing-size", str(getattr(args, 'p28_chunk_routing_size', 0)),
        "--p28-global-template-bank", str(getattr(args, 'p28_global_template_bank', 'none')),
        "--p28-attn-proj-templates", str(getattr(args, 'p28_attn_proj_templates', 0)),
        "--p28-attn-qk-templates",   str(getattr(args, 'p28_attn_qk_templates', 0)),
        "--p31-chunk-route-impl", str(getattr(args, 'p31_chunk_route_impl', 'compose')),
        "--p31-top1-gate", str(getattr(args, 'p31_top1_gate', 'ones')),
        "--p31-template-delta-rank", str(getattr(args, 'p31_template_delta_rank', 0)),
        "--p31-route-side", str(getattr(args, 'p31_route_side', 'output')),
        "--p31-drop-basis-proj", str(getattr(args, 'p31_drop_basis_proj', 0)),
        "--p31-basis-side-templates", str(getattr(args, 'p31_basis_side_templates', 0)),
        "--remix-template-block-diag", str(getattr(args, 'remix_template_block_diag', 0)),
        "--remix-template-lr-scale",   str(getattr(args, 'remix_template_lr_scale', 1.0)),
        "--p29-grad-equalize",         str(getattr(args, 'p29_grad_equalize', 0)),
        "--target-active-params",     str(getattr(args, 'target_active_params', 0)),
        "--compile-regional",         str(getattr(args, 'compile_regional', 0)),
        "--timing-probe-steps",       str(getattr(args, 'timing_probe_steps', 0)),
        "--compile-mode",             str(getattr(args, 'compile_mode', 'default')),
        "--remix-basis-gate-rank", str(getattr(args, 'remix_basis_gate_rank', 8)),
        "--cclblock-modulation", str(args.cclblock_modulation),
        "--cclblock-orth-lambda", str(getattr(args, 'cclblock_orth_lambda', 0.0)),
        "--p35-template-diversity-lambda", str(getattr(args, 'p35_template_diversity_lambda', 0.0)),
        "--cclblock-context-stream", str(args.cclblock_context_stream),
        "--cclblock-ema-factor", str(args.cclblock_ema_factor),
        "--cclblock-stale-ctx-lag", str(args.cclblock_stale_ctx_lag),
        # Novel ablation designs
        "--cclblock-sparse-gate-k", str(getattr(args, 'cclblock_sparse_gate_k', 0)),
        "--cclblock-gate-temperature", str(getattr(args, 'cclblock_gate_temperature', 1.0)),
        "--cclblock-context-bank-size", str(getattr(args, 'cclblock_context_bank_size', 0)),
        "--cclblock-per-head-ctx", str(getattr(args, 'cclblock_per_head_ctx', 0)),
        "--cclblock-context-source", str(getattr(args, 'cclblock_context_source', 'norm_x')),
        # Phase 8
        "--cclblock-chunk-size",        str(getattr(args, 'cclblock_chunk_size', 0)),
        "--cclblock-aux-objective",     str(getattr(args, 'cclblock_aux_objective', 'none')),
        "--cclblock-aux-lambda",        str(getattr(args, 'cclblock_aux_lambda', 0.1)),
        "--cclblock-boundary-token-id", str(getattr(args, 'cclblock_boundary_token_id', 198)),
        "--use-ral", str(getattr(args, 'use_ral', 0)),
        "--ral-rank", str(getattr(args, 'ral_rank', 32)),
        # Phase 35: ConditionedLinear
        "--p34-ffn-schedule", str(getattr(args, 'p34_ffn_schedule', '')),
        "--p34-ffn-no-ffn-replacement", str(getattr(args, 'p34_ffn_no_ffn_replacement', 'none')),
        "--p34-ffn-last-depth", str(getattr(args, 'p34_ffn_last_depth', 1)),
        "--p36-swiglu-ffn", str(getattr(args, 'p36_swiglu_ffn', 0)),
        "--p36-swiglu-mult", str(getattr(args, 'p36_swiglu_mult', 8.0/3.0)),
        "--cond-sites", str(getattr(args, 'cond_sites', 'both')),
        "--p22-route-affine", str(getattr(args, 'p22_route_affine', 0)),
        "--cond-rank", str(getattr(args, 'cond_rank', 256)),
        "--cond-mult-steps", str(getattr(args, 'cond_mult_steps', 0)),
        "--cond-gate-source", str(getattr(args, 'cond_gate_source', 'router')),
        "--cond-coeff-act", str(getattr(args, 'cond_coeff_act', 'centered')),
        "--cond-router-rank", str(getattr(args, 'cond_router_rank', 0)),
        "--cond-router-act", str(getattr(args, 'cond_router_act', 'none')),
        "--cond-live-init", str(getattr(args, 'cond_live_init', 0.0)),
        "--cond-chunk-size", str(getattr(args, 'cond_chunk_size', 0)),
        "--cond-mult-scale", str(getattr(args, 'cond_mult_scale', -1.0)),
        "--cond-mult-impl", str(getattr(args, 'cond_mult_impl', 'wy')),
        "--cond-attn-projs", str(getattr(args, 'cond_attn_projs', 'qkvo')),
        "--cond-layer-frac", str(getattr(args, 'cond_layer_frac', 1.0)),
        "--cclblock-film-gate", str(getattr(args, 'cclblock_film_gate', 0)),
        "--cclblock-attn-shadow-dim", str(getattr(args, 'cclblock_attn_shadow_dim', 0)),
        "--cclblock-dynamic-ratio", str(getattr(args, 'cclblock_dynamic_ratio', 0.25)),
        "--cclblock-gate-rank", str(getattr(args, 'cclblock_gate_rank', 8)),
        "--cclblock-num-regimes", str(getattr(args, 'cclblock_num_regimes', 8)),
        "--cclblock-regime-temperature", str(getattr(args, 'cclblock_regime_temperature', 1.0)),
        "--cclblock-poly-order", str(getattr(args, 'cclblock_poly_order', 2)),
        "--cclblock-lie-generators", str(getattr(args, 'cclblock_lie_generators', 4)),
        "--cclblock-grassmann-bank-size", str(getattr(args, 'cclblock_grassmann_bank_size', 4)),
        "--cclblock-tucker-rank", str(getattr(args, 'cclblock_tucker_rank', 32)),
        "--cclblock-tucker-modes", str(getattr(args, 'cclblock_tucker_modes', 8)),
        "--cclblock-svs-rank", str(getattr(args, 'cclblock_svs_rank', 64)),
        "--cclblock-svs-eps", str(getattr(args, 'cclblock_svs_eps', 0.1)),
        "--cclblock-vq-codes", str(getattr(args, 'cclblock_vq_codes', 8)),
        "--cclblock-vq-temperature", str(getattr(args, 'cclblock_vq_temperature', 1.0)),
        "--cclblock-dcu-warmup-steps", str(getattr(args, 'cclblock_dcu_warmup_steps', 0)),
        # Phase 12: FSI/AESP/CKR
        "--cclblock-fsi-rotations", str(getattr(args, 'cclblock_fsi_rotations', 8)),
        "--cclblock-fsi-selector-dim", str(getattr(args, 'cclblock_fsi_selector_dim', 64)),
        "--cclblock-aesp-strata", str(getattr(args, 'cclblock_aesp_strata', 4)),
        "--cclblock-aesp-delta-rank", str(getattr(args, 'cclblock_aesp_delta_rank', 4)),
        "--cclblock-ckr-branches", str(getattr(args, 'cclblock_ckr_branches', 4)),
        "--cclblock-ckr-kernel-size", str(getattr(args, 'cclblock_ckr_kernel_size', 64)),
        # Phase 13: CKR enhancements
        "--cclblock-ckr-pos-channels", str(getattr(args, 'cclblock_ckr_pos_channels', 1)),
        "--cclblock-ckr-dual-optim", str(getattr(args, 'cclblock_ckr_dual_optim', 0)),
        "--cclblock-ckr-content-bias", str(getattr(args, 'cclblock_ckr_content_bias', 0.0)),
        # Phase 14: GIAD/PSG/SplitStream
        "--cclblock-giad-rank", str(getattr(args, 'cclblock_giad_rank', 32)),
        "--cclblock-psg-kernel-size", str(getattr(args, 'cclblock_psg_kernel_size', 64)),
        "--cclblock-ss-dynamic-ratio", str(getattr(args, 'cclblock_ss_dynamic_ratio', 0.25)),
        "--cclblock-ss-branches", str(getattr(args, 'cclblock_ss_branches', 2)),
        "--cclblock-ss-kernel-size", str(getattr(args, 'cclblock_ss_kernel_size', 64)),
        # Phase 15: LoKR
        "--cclblock-lokr-branches", str(getattr(args, 'cclblock_lokr_branches', 8)),
        "--cclblock-lokr-rank", str(getattr(args, 'cclblock_lokr_rank', 16)),
        # Phase 16: CKR-Anneal / COM
        "--cclblock-ckr-temp-start", str(getattr(args, 'cclblock_ckr_temp_start', 2.0)),
        "--cclblock-ckr-temp-end", str(getattr(args, 'cclblock_ckr_temp_end', 0.3)),
        "--cclblock-com-kernel-size", str(getattr(args, 'cclblock_com_kernel_size', 32)),
        # Phase 17: CKR enhancements + new architectures
        "--cclblock-ckr-ortho-init", str(getattr(args, 'cclblock_ckr_ortho_init', 0)),
        "--cclblock-ckr-branch-dropout", str(getattr(args, 'cclblock_ckr_branch_dropout', 0.0)),
        "--cclblock-ckr-diversity-lambda", str(getattr(args, 'cclblock_ckr_diversity_lambda', 0.0)),
        "--cclblock-pgr-kernel-size", str(getattr(args, 'cclblock_pgr_kernel_size', 64)),
        "--cclblock-cil-kernel-size", str(getattr(args, 'cclblock_cil_kernel_size', 64)),
        "--cclblock-prb-kernel-size", str(getattr(args, 'cclblock_prb_kernel_size', 64)),
        # Phase 18: Beyond CKR
        "--p18-layer-drop", str(getattr(args, 'p18_layer_drop', 0.0)),
        "--p18-dynamic-activation", str(getattr(args, 'p18_dynamic_activation', 0)),
        "--p18-mixture-norm", str(getattr(args, 'p18_mixture_norm', 0)),
        "--p18-aux-sim-lambda", str(getattr(args, 'p18_aux_sim_lambda', 0.0)),
        "--p18-gradient-penalty", str(getattr(args, 'p18_gradient_penalty', 0.0)),
        "--p18-per-channel-scale", str(getattr(args, 'p18_per_channel_scale', 0)),
        # Phase 19: Zero-overhead indirect modulation
        "--p19-residual-gate", str(getattr(args, 'p19_residual_gate', 0)),
        "--p19-head-importance", str(getattr(args, 'p19_head_importance', 0)),
        "--p19-residual-mix-groups", str(getattr(args, 'p19_residual_mix_groups', 0)),
        "--p19-attn-logit-bias", str(getattr(args, 'p19_attn_logit_bias', 0)),
        "--p19-residual-decay", str(getattr(args, 'p19_residual_decay', 0)),
        "--p19-grad-equilibrium", str(getattr(args, 'p19_grad_equilibrium', 0.0)),
        "--p19-spectral-reparam", str(getattr(args, 'p19_spectral_reparam', 0)),
        "--p19-weight-anticollapse", str(getattr(args, 'p19_weight_anticollapse', 0.0)),
        "--p19-ve-bias", str(getattr(args, 'p19_ve_bias', 0)),
        "--p19-weight-noise", str(getattr(args, 'p19_weight_noise', 0.0)),
        # Phase 20
        "--p20-hrcs-scale", str(getattr(args, 'p20_hrcs_scale', 0)),
        "--p20-lswr-scale", str(getattr(args, 'p20_lswr_scale', 0)),
        "--p20-lswr-planes", str(getattr(args, 'p20_lswr_planes', 8)),
        "--p20-lrcfb-branches", str(getattr(args, 'p20_lrcfb_branches', 0)),
        "--p20-lrcfb-narrow", str(getattr(args, 'p20_lrcfb_narrow', 0)),
        "--p20-lrcfb-learned", str(getattr(args, 'p20_lrcfb_learned', 0)),
        "--p20-lrcfb-topk", str(getattr(args, 'p20_lrcfb_topk', 0)),
        "--p20-dgcr-branches", str(getattr(args, 'p20_dgcr_branches', 0)),
        "--p20-dgcr-aux-weight", str(getattr(args, 'p20_dgcr_aux_weight', 0.01)),
        "--p20-mone-experts", str(getattr(args, 'p20_mone_experts', 0)),
        "--p20-mone-topk", str(getattr(args, 'p20_mone_topk', 0)),
        "--p20-mone-narrow", str(getattr(args, 'p20_mone_narrow', 1)),
        "--p20-mone-frozen", str(getattr(args, 'p20_mone_frozen', 0)),
        "--p20-ncea-branches", str(getattr(args, 'p20_ncea_branches', 0)),
        "--p20-ncea-eps", str(getattr(args, 'p20_ncea_eps', 0.1)),
        "--p20-adwi", str(getattr(args, 'p20_adwi', 0)),
        # Phase 2 proposals
        "--p20-pwu-branches", str(getattr(args, 'p20_pwu_branches', 0)),
        "--p20-pwu-phase", str(getattr(args, 'p20_pwu_phase', 1)),
        "--p20-fsvd-gate", str(getattr(args, 'p20_fsvd_gate', 0)),
        "--p20-wbfc-clusters", str(getattr(args, 'p20_wbfc_clusters', 0)),
        "--p20-wbfc-active", str(getattr(args, 'p20_wbfc_active', 0)),
        # Phase 21
        "--p21-per-experts", str(getattr(args, 'p21_per_experts', 0)),
        "--p21-per-topk", str(getattr(args, 'p21_per_topk', 0)),
        "--p21-per-learned", str(getattr(args, 'p21_per_learned', 0)),
        "--p21-per-attn", str(getattr(args, 'p21_per_attn', 0)),
        # Phase 23: Tiny Experts RemixedLinear + Standard MoE baseline
        "--p23-tiny-expert", str(getattr(args, 'p23_tiny_expert', 0)),
        "--p23-n-experts", str(getattr(args, 'p23_n_experts', 64)),
        "--p23-topk", str(getattr(args, 'p23_topk', 16)),
        "--p23-learned-route", str(getattr(args, 'p23_learned_route', 0)),
        "--p23-std-moe-experts", str(getattr(args, 'p23_std_moe_experts', 0)),
        "--p23-std-moe-topk", str(getattr(args, 'p23_std_moe_topk', -1)),
        "--p23-std-moe-aux-weight", str(getattr(args, 'p23_std_moe_aux_weight', 0.01)),
        "--p23-lokr", str(getattr(args, 'p23_lokr', 0)),
        "--p23-lokr-rank", str(getattr(args, 'p23_lokr_rank', 4)),
        "--p23-use-shared-block-router", str(getattr(args, 'p23_use_shared_block_router', 0)),
        "--p23-linear-moe-experts", str(getattr(args, 'p23_linear_moe_experts', 0)),
        "--p23-linear-moe-topk", str(getattr(args, 'p23_linear_moe_topk', 0)),
        "--p23-quantile-route", str(getattr(args, 'p23_quantile_route', 0)),
        "--seed", str(getattr(args, 'seed', -1)),
        "--p34-ffn-mult", str(getattr(args, 'p34_ffn_mult', 4.0)),
        "--p34-ffn-single", str(getattr(args, 'p34_ffn_single', 0)),
        "--p34-dense-attn", str(getattr(args, 'p34_dense_attn', 0)),
        "--p24-use-sliced-weight", str(getattr(args, 'p24_use_sliced_weight', 0)),
        "--p24-sliced-weight-reduction-scale", str(getattr(args, 'p24_sliced_weight_reduction_scale', 8)),
        "--p24-sliced-weight-min-select", str(getattr(args, 'p24_sliced_weight_min_select', 128)),
        "--p24-sliced-weight-scope", str(getattr(args, 'p24_sliced_weight_scope', 'per_token')),
        "--p24-sliced-weight-balance-coeff", str(getattr(args, 'p24_sliced_weight_balance_coeff', 0.01)),
        "--p24-quantile-route", str(getattr(args, 'p24_quantile_route', 0)),
        "--p24-use-folded-mod", str(getattr(args, 'p24_use_folded_mod', 0)),
        "--p24-folded-mod-reduction-scale", str(getattr(args, 'p24_folded_mod_reduction_scale', 8)),
        "--p24-folded-mod-scope", str(getattr(args, 'p24_folded_mod_scope', 'per_layer')),
        "--p24-folded-mod-gate-act", str(getattr(args, 'p24_folded_mod_gate_act', 'tanh_centered')),
        "--p24-use-sequence-gated-linear", str(getattr(args, 'p24_use_sequence_gated_linear', 0)),
        "--p24-sequence-gated-scope", str(getattr(args, 'p24_sequence_gated_scope', 'per_layer')),
        "--p24-sequence-gated-act", str(getattr(args, 'p24_sequence_gated_act', 'tanh_centered')),
        "--p24-folded-mod-min-dim", str(getattr(args, 'p24_folded_mod_min_dim', 128)),
        "--remix-shared-context-gates", str(getattr(args, 'remix_shared_context_gates', 0)),
        "--remix-use-dual-gate", str(getattr(args, 'remix_use_dual_gate', 0)),
        "--remix-basis-scale-factor", str(getattr(args, 'remix_basis_scale_factor', 4)),
        "--remix-output-gate-rank", str(getattr(args, 'remix_output_gate_rank', 16)),
        "--p24-use-sliced-weight", str(getattr(args, 'p24_use_sliced_weight', 0)),
        "--p24-sliced-weight-reduction-scale", str(getattr(args, 'p24_sliced_weight_reduction_scale', 8)),
        "--p24-sliced-weight-min-select", str(getattr(args, 'p24_sliced_weight_min_select', 128)),
        "--p24-sliced-weight-scope", str(getattr(args, 'p24_sliced_weight_scope', "global")),
        "--p24-sliced-weight-balance-coeff", str(getattr(args, 'p24_sliced_weight_balance_coeff', 0.01)),
        "--p24-quantile-route", str(getattr(args, 'p24_quantile_route', 0)),
        "--p24-use-folded-mod", str(getattr(args, 'p24_use_folded_mod', 0)),
        "--p24-folded-mod-reduction-scale", str(getattr(args, 'p24_folded_mod_reduction_scale', 8)),
        "--p24-folded-mod-scope", str(getattr(args, 'p24_folded_mod_scope', "global")),
        "--p24-folded-mod-gate-act", str(getattr(args, 'p24_folded_mod_gate_act', "tanh_centered")),
        "--p24-use-sequence-gated-linear", str(getattr(args, 'p24_use_sequence_gated_linear', 0)),
        "--p24-sequence-gated-scope", str(getattr(args, 'p24_sequence_gated_scope', "global")),
        "--p24-sequence-gated-act", str(getattr(args, 'p24_sequence_gated_act', "tanh_centered")),
        # Phase 30: LayerNorm ablation
        "--remix-disable-ln-basis", str(getattr(args, 'remix_disable_ln_basis', 0)),
        "--dense-intermediate-ln", str(getattr(args, 'dense_intermediate_ln', 0)),
        # MoL: Mixture of Layers baseline (arXiv:2605.09516)
        "--use-mol", str(getattr(args, 'use_mol', 0)),
        "--mol-n-blocks", str(getattr(args, 'mol_n_blocks', 5)),
        "--mol-n-shared", str(getattr(args, 'mol_n_shared', 1)),
        "--mol-topk", str(getattr(args, 'mol_topk', 3)),
        "--mol-thin-dim", str(getattr(args, 'mol_thin_dim', 256)),
        "--mol-head-dim", str(getattr(args, 'mol_head_dim', 64)),
        "--mol-ffn-mult", str(getattr(args, 'mol_ffn_mult', 4.0)),
        "--mol-router-aux", str(getattr(args, 'mol_router_aux', 0.05)),
        "--mol-routed-attn", str(getattr(args, 'mol_routed_attn', 'softmax')),
        "--mol-dispatch", str(getattr(args, 'mol_dispatch', 1)),
        "--mol-capacity-factor", str(getattr(args, 'mol_capacity_factor', 1.0)),
        "--mol-block-lr-scale", str(getattr(args, 'mol_block_lr_scale', 1.0)),
        "--mol-per-block-ve", str(getattr(args, 'mol_per_block_ve', 0)),
        # MST: Modular Sub-Transformer
        "--use-binary", str(getattr(args, "use_binary", 0)),
        "--binary-acts", str(getattr(args, "binary_acts", 1)),
        "--binary-weight-scale", str(getattr(args, "binary_weight_scale", 'row')),
        "--binary-act-scale", str(getattr(args, "binary_act_scale", 'token')),
        "--binary-clip", str(getattr(args, "binary_clip", 1.0)),
        "--binary-linear", str(getattr(args, "binary_linear", 1)),
        "--binary-embeddings", str(getattr(args, "binary_embeddings", 1)),
        "--binary-skip", str(getattr(args, "binary_skip", '')),
        "--use-mst", str(getattr(args, 'use_mst', 0)),
        "--mst-n-subs", str(getattr(args, 'mst_n_subs', 8)),
        "--mst-sub-dim", str(getattr(args, 'mst_sub_dim', 64)),
        "--mst-head-dim", str(getattr(args, 'mst_head_dim', 0)),
        "--mst-input-mode", str(getattr(args, 'mst_input_mode', 'fixed_slice')),
        "--mst-rotated-slice-learned", str(getattr(args, 'mst_rotated_slice_learned', 0)),
        "--mst-routing-mode", str(getattr(args, 'mst_routing_mode', 'soft_weighted')),
        "--mst-routing-topk", str(getattr(args, 'mst_routing_topk', 4)),
        "--mst-routing-aux-weight", str(getattr(args, 'mst_routing_aux_weight', 0.01)),
        "--mst-diversity-weight", str(getattr(args, 'mst_diversity_weight', 0.0)),
        "--mst-ffn-mode", str(getattr(args, 'mst_ffn_mode', 'standard')),
        "--mst-ffn-every", str(getattr(args, 'mst_ffn_every', 1)),
        "--mst-transition-mode", str(getattr(args, 'mst_transition_mode', 'parallel')),
        "--mst-final-mode", str(getattr(args, 'mst_final_mode', 'aggregate_proj')),
        "--mst-final-topk", str(getattr(args, 'mst_final_topk', -1)),
        "--mst-ffn-shared-up", str(getattr(args, 'mst_ffn_shared_up', 0)),
        "--mst-ffn-inner-dim", str(getattr(args, 'mst_ffn_inner_dim', 0)),
        "--mst-sub-dropout", str(getattr(args, 'mst_sub_dropout', 0.0)),
        "--mst-transition-every", str(getattr(args, 'mst_transition_every', 1)),
        "--mst-ffa-temperature", str(getattr(args, 'mst_ffa_temperature', 1.0)),
        "--mst-global-residual", str(getattr(args, 'mst_global_residual', 0)),
        "--mst-hybrid-dense", str(getattr(args, 'mst_hybrid_dense', 0)),
        "--mst-cross-sub-kv", str(getattr(args, 'mst_cross_sub_kv', 0)),
        "--mst-sub-aux-weight", str(getattr(args, 'mst_sub_aux_weight', 0.0)),
        "--mst-progressive-merge", str(getattr(args, 'mst_progressive_merge', 0)),
        "--mst-multi-scale-windows", str(getattr(args, 'mst_multi_scale_windows', 0)),
        "--mst-delta-residual", str(getattr(args, 'mst_delta_residual', 0)),
        "--mst-sub-layers", str(getattr(args, 'mst_sub_layers', 1)),
        # MST Stage 7: Scaling improvements
        "--mst-grad-equalize", str(getattr(args, 'mst_grad_equalize', 0)),
        "--mst-block-diagonal-muon", str(getattr(args, 'mst_block_diagonal_muon', 0)),
        "--mst-transition-width-mult", str(getattr(args, 'mst_transition_width_mult', 1.0)),
        "--mst-sub-lr-scale", str(getattr(args, 'mst_sub_lr_scale', 1.0)),
        "--mst-shared-expert", str(getattr(args, 'mst_shared_expert', 0)),
        "--mst-router-entropy-weight", str(getattr(args, 'mst_router_entropy_weight', 0.0)),
        "--mst-shared-kv-attn", str(getattr(args, 'mst_shared_kv_attn', 0)),
        "--mst-contrastive-diversity-weight", str(getattr(args, 'mst_contrastive_diversity_weight', 0.0)),
        # MST Stage 8: Transition expressivity
        "--mst-transition-nonlinear", str(getattr(args, 'mst_transition_nonlinear', 0)),
        "--mst-transition-gated", str(getattr(args, 'mst_transition_gated', 0)),
        "--mst-transition-mlp", str(getattr(args, 'mst_transition_mlp', 0)),
        # MST Stage 9: Cross-sub expressivity
        "--mst-cross-sub-gate", str(getattr(args, 'mst_cross_sub_gate', 0)),
        "--mst-hyper-connect", str(getattr(args, 'mst_hyper_connect', 0)),
        "--mst-cross-kv-inject", str(getattr(args, 'mst_cross_kv_inject', 0)),
        # MST Stage 10: Structural transition improvements
        "--mst-slice-transition", str(getattr(args, 'mst_slice_transition', 0)),
        "--mst-lookback-layers", str(getattr(args, 'mst_lookback_layers', 0)),
        "--mst-bilinear-transition", str(getattr(args, 'mst_bilinear_transition', 0)),
        # MST Stage 11: Attention bottleneck + structural improvements
        "--mst-cross-sub-qmod", str(getattr(args, 'mst_cross_sub_qmod', 0)),
        "--mst-feature-cycle", str(getattr(args, 'mst_feature_cycle', 0)),
        "--mst-mean-transition", str(getattr(args, 'mst_mean_transition', 0)),
        "--mst-sub-head-dim", str(getattr(args, 'mst_sub_head_dim', 0)),
        "--mst-final-norm", str(getattr(args, 'mst_final_norm', 0)),
        "--mst-per-stream-ve", str(getattr(args, 'mst_per_stream_ve', 0)),
        "--mst-ve-map", str(getattr(args, 'mst_ve_map', 0)),
        "--mst-ve-map-rank", str(getattr(args, 'mst_ve_map_rank', 0)),
        "--mst-lm-head-dim", str(getattr(args, 'mst_lm_head_dim', 0)),
        "--mst-compose-windows", str(getattr(args, 'mst_compose_windows', 0)),
        "--mst-channel-mix", str(getattr(args, 'mst_channel_mix', 'none')),
        "--mst-channel-mix-site", str(getattr(args, 'mst_channel_mix_site', 'layer')),
        "--mst-distribute-block-muon", str(getattr(args, 'mst_distribute_block_muon', 0)),
        "--mst-trans-spectral-lr", str(getattr(args, 'mst_trans_spectral_lr', 0)),
        "--mst-talking-heads", str(getattr(args, 'mst_talking_heads', 0)),
        "--mst-wo-mode", str(getattr(args, 'mst_wo_mode', 'block')),
        "--mst-stream-topk", str(getattr(args, 'mst_stream_topk', 0)),
        "--mst-stream-shared", str(getattr(args, 'mst_stream_shared', 0)),
        "--mst-stream-router-aux", str(getattr(args, 'mst_stream_router_aux', 0.01)),
        "--mst-stream-router-noise", str(getattr(args, 'mst_stream_router_noise', 0.0)),
        "--mst-stream-dispatch", str(getattr(args, 'mst_stream_dispatch', 0)),
        "--mst-stream-capacity-factor", str(getattr(args, 'mst_stream_capacity_factor', 1.0)),
        "--mst-stream-gate-attn", str(getattr(args, 'mst_stream_gate_attn', 0)),
        "--mst-ffn-monarch", str(getattr(args, 'mst_ffn_monarch', 'none')),
        "--mst-shampoo", str(getattr(args, 'mst_shampoo', 0)),
        "--mst-precond-every", str(getattr(args, 'mst_precond_every', 10)),
        "--mst-shampoo-beta", str(getattr(args, 'mst_shampoo_beta', 0.95)),
        # EET: Early Exit Transformer
        "--use-eet", str(getattr(args, 'use_eet', 0)),
        "--eet-frozen-kv", str(getattr(args, 'eet_frozen_kv', 1)),
        "--eet-reenter-final", str(getattr(args, 'eet_reenter_final', 0)),
        "--eet-compute-skip", str(getattr(args, 'eet_compute_skip', 0)),
        "--eet-target-active-frac", str(getattr(args, 'eet_target_active_frac', 0.125)),
        "--eet-capacity-schedule", str(getattr(args, 'eet_capacity_schedule', 'bell')),
        "--eet-exit-fracs", str(getattr(args, 'eet_exit_fracs', '')),
        "--eet-router-type", str(getattr(args, 'eet_router_type', 'mlp2')),
        "--eet-router-hidden", str(getattr(args, 'eet_router_hidden', 0)),
        "--eet-freq-prior-alpha", str(getattr(args, 'eet_freq_prior_alpha', 0.0)),
        "--eet-pos-prior-beta", str(getattr(args, 'eet_pos_prior_beta', 0.0)),
        "--eet-domain-prior", str(getattr(args, 'eet_domain_prior', 0)),
        "--eet-warmup-frac", str(getattr(args, 'eet_warmup_frac', 0.02)),
        "--eet-explore-frac", str(getattr(args, 'eet_explore_frac', 0.15)),
        "--eet-reconstruct-lambda", str(getattr(args, 'eet_reconstruct_lambda', 1.0)),
        "--eet-efficiency-lambda-start", str(getattr(args, 'eet_efficiency_lambda_start', 0.01)),
        "--eet-efficiency-lambda-end", str(getattr(args, 'eet_efficiency_lambda_end', 0.1)),
        "--eet-translator-rank", str(getattr(args, 'eet_translator_rank', 0)),
        "--eet-max-frozen-kv-frac", str(getattr(args, 'eet_max_frozen_kv_frac', 0.75)),
        "--eet-exit-threshold", str(getattr(args, 'eet_exit_threshold', 0.5)),
        "--eet-min-exit-layer", str(getattr(args, 'eet_min_exit_layer', 1)),
        "--eet-loss-variant", str(getattr(args, 'eet_loss_variant', 'reconstruct')),
        "--eet-topk-vocab", str(getattr(args, 'eet_topk_vocab', 512)),
        "--eet-entropy-lambda", str(getattr(args, 'eet_entropy_lambda', 0.3)),
        "--eet-surprise-lambda", str(getattr(args, 'eet_surprise_lambda', 0.1)),
        "--eet-adv-lambda", str(getattr(args, 'eet_adv_lambda', 1.0)),
        "--eet-adv-entropy-lambda", str(getattr(args, 'eet_adv_entropy_lambda', 0.2)),
        "--eet-quality-lambda", str(getattr(args, 'eet_quality_lambda', 1.0)),
        "--eet-quality-entropy-bonus", str(getattr(args, 'eet_quality_entropy_bonus', 0.1)),
        "--eet-gumbel-temp-start", str(getattr(args, 'eet_gumbel_temp_start', 0.0)),
        "--eet-gumbel-temp-end", str(getattr(args, 'eet_gumbel_temp_end', 0.1)),
        "--eet-gumbel-hard", str(getattr(args, 'eet_gumbel_hard', 1)),
        "--eet-commitment-beta", str(getattr(args, 'eet_commitment_beta', 0.1)),
        "--eet-global-router", str(getattr(args, 'eet_global_router', 0)),
        "--eet-freq-efficiency-alpha", str(getattr(args, 'eet_freq_efficiency_alpha', 0.0)),
        "--eet-diversity-lambda", str(getattr(args, 'eet_diversity_lambda', 0.0)),
        "--eet-ce-guided-lambda", str(getattr(args, 'eet_ce_guided_lambda', 1.0)),
        "--eet-router-lr-mult", str(getattr(args, 'eet_router_lr_mult', 5.0)),
        "--eet-model-lr-mult", str(getattr(args, 'eet_model_lr_mult', 1.0)),
        "--eet-depth-weight-type", str(getattr(args, 'eet_depth_weight_type', 'none')),
        "--eet-depth-weight-max", str(getattr(args, 'eet_depth_weight_max', 2.5)),
        "--eet-use-override", str(getattr(args, 'eet_use_override', 0)),
        "--eet-override-prob-start", str(getattr(args, 'eet_override_prob_start', 0.5)),
        "--eet-override-prob-end", str(getattr(args, 'eet_override_prob_end', 0.1)),
        "--eet-capacity-alignment-lambda", str(getattr(args, 'eet_capacity_alignment_lambda', 0.0)),
        "--eet-router-task-grad", str(getattr(args, 'eet_router_task_grad', 1)),
        "--eet-reinforce-interval", str(getattr(args, 'eet_reinforce_interval', 0)),
        "--eet-reinforce-lambda", str(getattr(args, 'eet_reinforce_lambda', 0.1)),
        "--eet-exit-adapter-rank", str(getattr(args, 'eet_exit_adapter_rank', 0)),
        "--eet-router-after-block", str(getattr(args, 'eet_router_after_block', 0)),
        "--eet-ffn-skip", str(getattr(args, 'eet_ffn_skip', 0)),
        "--eet-ffn-target-frac", str(getattr(args, 'eet_ffn_target_frac', 0.50)),
        "--eet-ffn-full-attn", str(getattr(args, 'eet_ffn_full_attn', 1)),
        "--eet-depth-affine", str(getattr(args, 'eet_depth_affine', 0)),
        "--eet-capacity-anneal-frac", str(getattr(args, 'eet_capacity_anneal_frac', 0.0)),
        "--eet-learned-schedule", str(getattr(args, 'eet_learned_schedule', 0)),
        "--eet-departure-summary", str(getattr(args, 'eet_departure_summary', 0)),
        "--eet-route-consistency-lambda", str(getattr(args, 'eet_route_consistency_lambda', 0.0)),
        "--eet-dense-distill-interval", str(getattr(args, 'eet_dense_distill_interval', 0)),
        "--eet-dense-distill-lambda", str(getattr(args, 'eet_dense_distill_lambda', 0.5)),
        "--eet-depth-lr-scale", str(getattr(args, 'eet_depth_lr_scale', 0)),
        "--eet-depth-grad-scale", str(getattr(args, 'eet_depth_grad_scale', 0)),
        "--eet-detach-aux-from-backbone", str(getattr(args, 'eet_detach_aux_from_backbone', 0)),
        "--eet-detach-exit-from-backbone", str(getattr(args, 'eet_detach_exit_from_backbone', 0)),
        # EET P02: context restoration, stochastic routing, coverage diagnostic
        "--eet-kv-mode", str(getattr(args, 'eet_kv_mode', 'none')),
        "--eet-route-noise", str(getattr(args, 'eet_route_noise', 0.0)),
        "--eet-route-noise-end", str(getattr(args, 'eet_route_noise_end', -1.0)),
        "--eet-coverage-diag", str(getattr(args, 'eet_coverage_diag', 0)),
        "--eet-kv-eager", str(getattr(args, 'eet_kv_eager', 0)),
        "--deep-supervision-lambda", str(getattr(args, 'deep_supervision_lambda', 0.0)),
        "--deep-supervision-frac", str(getattr(args, 'deep_supervision_frac', 0.125)),
        "--eet-width-power", str(getattr(args, 'eet_width_power', 0.0)),
        "--eet-width-cap", str(getattr(args, 'eet_width_cap', 16.0)),
        # SCH: Structured Code Output Heads
        "--use-code-head", str(getattr(args, 'use_code_head', 0)),
        "--sch-head-type", str(getattr(args, 'sch_head_type', 'code')),
        "--sch-bits", str(getattr(args, 'sch_bits', 0)),
        "--sch-order", str(getattr(args, 'sch_order', 2)),
        "--sch-max-m", str(getattr(args, 'sch_max_m', 0)),
        "--sch-phi-mode", str(getattr(args, 'sch_phi_mode', 'monomial')),
        "--sch-code-mode", str(getattr(args, 'sch_code_mode', 'binary')),
        "--sch-code-path", str(getattr(args, 'sch_code_path', '')),
        "--sch-code-ecc-bits", str(getattr(args, 'sch_code_ecc_bits', 0)),
        "--sch-code-seed", str(getattr(args, 'sch_code_seed', 1234)),
        "--sch-phi-density", str(getattr(args, 'sch_phi_density', 0.0)),
        "--sch-phi-dtype", str(getattr(args, 'sch_phi_dtype', 'bf16')),
        "--sch-phi-normalize", str(getattr(args, 'sch_phi_normalize', 1)),
        "--sch-phi-center", str(getattr(args, 'sch_phi_center', 0)),
        "--sch-g-type", str(getattr(args, 'sch_g_type', 'linear')),
        "--sch-g-hidden", str(getattr(args, 'sch_g_hidden', 0)),
        "--sch-g-layers", str(getattr(args, 'sch_g_layers', 2)),
        "--sch-g-out-std", str(getattr(args, 'sch_g_out_std', 0.001)),
        "--sch-mixture", str(getattr(args, 'sch_mixture', 1)),
        "--sch-residual-rank", str(getattr(args, 'sch_residual_rank', 0)),
        "--sch-logit-act", str(getattr(args, 'sch_logit_act', 'none')),
        "--sch-bias", str(getattr(args, 'sch_bias', 0)),
        "--sch-input-mode", str(getattr(args, 'sch_input_mode', 'table')),
        "--sch-input-hidden", str(getattr(args, 'sch_input_hidden', 0)),
        "--sch-product-groups", str(getattr(args, 'sch_product_groups', 8)),
        "--sch-product-codebook", str(getattr(args, 'sch_product_codebook', 256)),
        "--sch-product-source", str(getattr(args, 'sch_product_source', 'hash')),
        "--sch-product-impl", str(getattr(args, 'sch_product_impl', 'dense')),
        "--sch-phi-whiten", str(getattr(args, 'sch_phi_whiten', 0)),
        "--sch-mixture-per-phi", str(getattr(args, 'sch_mixture_per_phi', 0)),
        "--sch-mixture-topk", str(getattr(args, 'sch_mixture_topk', 0)),
        "--sch-mixture-aux", str(getattr(args, 'sch_mixture_aux', 0.01)),
        "--sch-monarch-m1", str(getattr(args, 'sch_monarch_m1', 0)),
        "--sch-monarch-perm", str(getattr(args, 'sch_monarch_perm', 'none')),
        "--sch-monarch-perm-path", str(getattr(args, 'sch_monarch_perm_path', '')),
        "--sch-tier-bounds", str(getattr(args, 'sch_tier_bounds', '')),
        "--sch-tier-caps", str(getattr(args, 'sch_tier_caps', '')),
        "--sch-tier-order", str(getattr(args, 'sch_tier_order', 'freq')),
        "--sch-tier-perm-path", str(getattr(args, 'sch_tier_perm_path', '')),
        "--sch-proposal-rank", str(getattr(args, 'sch_proposal_rank', 32)),
        "--sch-proposal-topk", str(getattr(args, 'sch_proposal_topk', 16384)),
        "--sch-proposal-samples", str(getattr(args, 'sch_proposal_samples', 4096)),
        "--sch-proposal-warmup", str(getattr(args, 'sch_proposal_warmup', 0)),
        "--sch-proposal-chunk", str(getattr(args, 'sch_proposal_chunk', 2048)),
        "--sch-proposal-aux", str(getattr(args, 'sch_proposal_aux', 1.0)),
        "--sch-nfh-mode", str(getattr(args, 'sch_nfh_mode', 'cp')),
        "--sch-nfh-rank", str(getattr(args, 'sch_nfh_rank', 32)),
        "--sch-nfh-dims", str(getattr(args, 'sch_nfh_dims', '')),
        "--sch-nfh-groups", str(getattr(args, 'sch_nfh_groups', 3)),
        "--sch-nfh-views", str(getattr(args, 'sch_nfh_views', 1)),
        "--sch-nfh-perm", str(getattr(args, 'sch_nfh_perm', 'none')),
        "--sch-nfh-perm-path", str(getattr(args, 'sch_nfh_perm_path', '')),
        "--sch-nfh-g-type", str(getattr(args, 'sch_nfh_g_type', 'linear')),
        "--sch-nfh-g-hidden", str(getattr(args, 'sch_nfh_g_hidden', 0)),
        "--sch-nfh-chunk", str(getattr(args, 'sch_nfh_chunk', 256)),
        "--sch-nfh-smooth", str(getattr(args, 'sch_nfh_smooth', 0)),
        "--sch-rerank-mode", str(getattr(args, 'sch_rerank_mode', 'topk')),
        "--sch-rerank-k", str(getattr(args, 'sch_rerank_k', 64)),
        "--sch-rerank-rank", str(getattr(args, 'sch_rerank_rank', 32)),

        "--sch-holdout-tokens", str(getattr(args, 'sch_holdout_tokens', 0)),
        "--sch-holdout-seed", str(getattr(args, 'sch_holdout_seed', 7)),
        "--sch-holdout-min-id", str(getattr(args, 'sch_holdout_min_id', 256)),
        "--sch-holdout-mode", str(getattr(args, 'sch_holdout_mode', 'target')),
        "--sch-decile-metrics", str(getattr(args, 'sch_decile_metrics', 1)),
        "--sch-rank-probe", str(getattr(args, 'sch_rank_probe', 0)),
        "--sch-eval-steps", str(getattr(args, 'sch_eval_steps', 100)),
    ]
    if args.compile:
        common_args.append("--compile")
    else:
        common_args.append("--no-compile")
    if getattr(args, "fp8", False):
        common_args.append("--fp8")
    if getattr(args, "tokenizer_dir", None):
        common_args.extend(["--tokenizer-dir", args.tokenizer_dir])
    if getattr(args, "data_dir", None):
        common_args.extend(["--data-dir", args.data_dir])
    if getattr(args, "max_shards", -1) != -1:
        common_args.extend(["--max-shards", str(args.max_shards)])
    
    # --- Optimal LR Configurations (from actual_lr_research_sweep) ---
    BEST_LRS = {
        "moe_no_perm": {
            "embedding_lr":   0.104074,
            "unembedding_lr": 0.0245175,
            "matrix_lr":      0.0329274,
            "scalar_lr":      0.152507,
        },
        "moe_perm": {
            "embedding_lr":   0.104074,
            "unembedding_lr": 0.0245175,
            "matrix_lr":      0.0329274,
            "scalar_lr":      0.152507,
        },
        "remixed-linear": {
            "embedding_lr":   0.104074,
            "unembedding_lr": 0.0245175,
            "matrix_lr":      0.0329274,
            "scalar_lr":      0.152507,
        },
    }
    # ---------------------------------------------------------------

    models = {
        "base": [], # Base model relies on base_train.py defaults
    }

    # Research branch configurations (architecture only)
    research_configs = {
        "moe_no_perm": [
            "--use-moe",
            "--moe-num-experts", "8",
            "--moe-router-dim", str(target_dim),
            "--moe-embed-dim",  str(target_dim),
        ],
        "moe_perm": [
            "--use-moe",
            "--use-perm",
            "--moe-num-experts", "8",
            "--moe-router-dim", str(target_dim),
            "--moe-embed-dim",  str(target_dim),
        ],
        "remixed-linear": [
            "--use-remix-linear",
            "--moe-embed-dim",    str(target_dim),
            "--moe-router-dim",   str(target_dim),
            "--remix-context-dim", str(target_dim),
            # NOTE: basis_size is NOT set here by default.
            # scale_basis_size=True auto-computes max(64, min(in,out) // basis_scale_factor).
            # To force full-rank, pass --remix-basis-size explicitly (e.g. = model_dim).
        ] + (["--remix-basis-size", str(getattr(args, 'remix_basis_size', 0))] if getattr(args, 'remix_basis_size', 0) > 0 else [])
    }

    for name, flags in research_configs.items():
        lrs = BEST_LRS[name]
        models[name] = flags + [
            "--embedding-lr",   str(lrs["embedding_lr"]),
            "--unembedding-lr", str(lrs["unembedding_lr"]),
            "--matrix-lr",      str(lrs["matrix_lr"]),
            "--scalar-lr",      str(lrs["scalar_lr"]),
        ]
    
    # Filter models if requested
    target_models = args.models.split(",") if args.models != "all" else models.keys()
    filtered_models = {k: v for k, v in models.items() if k in target_models}
    if not filtered_models:
        print(f"No matching models found in {list(models.keys())} for selection '{args.models}'")
        return

    results = {}
    
    for model_name, extra_args in filtered_models.items():
        print(f"\n--- Training {model_name} ---")
        
        # Check for resumption. We resolve a fallback ckpt_dir in the current run environment
        # in case absolute paths stored in the sweep state from a previous run/different machine are invalid.
        fallback_ckpt_dir = (run_dir_path / f"ckpt_{model_name}").resolve()

        saved_ckpt_dir = None
        if model_name in state.get("unfinished", {}):
            saved_ckpt_dir = state["unfinished"][model_name].get("ckpt_dir")
        elif model_name in state.get("completed", {}):
            saved_ckpt_dir = state["completed"][model_name].get("ckpt_dir")

        if saved_ckpt_dir:
            temp_ckpt_dir = Path(saved_ckpt_dir)
            temp_actual = temp_ckpt_dir / model_name
            # Verify if this saved checkpoint directory exists and actually contains model checkpoint files
            if temp_actual.exists() and glob.glob(str(temp_actual / "model_*.pt")):
                ckpt_dir = temp_ckpt_dir
            else:
                ckpt_dir = fallback_ckpt_dir
        else:
            ckpt_dir = fallback_ckpt_dir

        # ChunkedRemixConfig: when --use-chunked-remix 1, inject canonical config defaults
        # as a *prefix* before common_args so explicit sweep flags (which come later) win.
        chunked_prefix: list[str] = []
        if getattr(args, 'use_chunked_remix', False) and model_name == "remixed-linear":
            _cfg = ChunkedRemixConfig()
            chunked_prefix = _cfg.to_cli_args(model_dim=model_dim)
            print(f"  [ChunkedRemixConfig] {_cfg.summary()}")

        train_cmd_args = chunked_prefix + common_args + extra_args + [
            "--checkpoints-dir", str(ckpt_dir),
            "--model-tag", model_name
        ]
        
        # Handle mu-P scaling based on the new mode system
        if args.mu_p_mode == "disable":
            if model_name != "base":
                train_cmd_args.append("--disable-mu-p")
        elif args.mu_p_mode == "base_only":
            if model_name != "base":
                # Force research models to use the exact same multiplier base would've used
                base_multiplier = (model_dim / 768) ** -0.5
                train_cmd_args.extend(["--mu-p-scale-override", str(base_multiplier)])
        # if "enable", neither flag is passed; models calculate their own inherently

        # ── Record as unfinished BEFORE launching so a crash is visible ──
        actual_model_ckpt_dir = ckpt_dir / model_name
        state.setdefault("unfinished", {})[model_name] = {
            "ckpt_dir": str(ckpt_dir),
            "actual_model_ckpt_dir": str(actual_model_ckpt_dir),
            "started_at": datetime.datetime.now().isoformat(),
        }
        save_sweep_state(run_dir_path, state)

        # Check for resumption
        try:
            last_step = _last_resumable_step(str(actual_model_ckpt_dir))
            print(f"\n  ┌─────────────────────────────────────────────────────┐")
            print(f"  │  ⏩  RESUMING [{model_name}] from step {last_step:,}")
            print(f"  │     {str(actual_model_ckpt_dir)}")
            print(f"  └─────────────────────────────────────────────────────┘\n")
            train_cmd_args.extend(["--resume-from-step", str(last_step)])
        except FileNotFoundError:
            print(f"\n  ┌─────────────────────────────────────────────────────┐")
            print(f"  │  🆕  STARTING FRESH: [{model_name}]")
            print(f"  │     No checkpoints found — training from scratch.")
            print(f"  └─────────────────────────────────────────────────────┘\n")
        

        # Need to preserve environment variables, especially LD_LIBRARY_PATH for cusparseLt
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"

        # Each model is trained as a proper DDP job via torchrun.
        cmd = RUNNER + ["-m", "scripts.base_train"] + train_cmd_args
        print(f"Running: {' '.join(cmd)}")
        
        try:
            # We stream stdout so user isn't stuck waiting blindly
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
            if process.stdout:
                for line in iter(process.stdout.readline, ""):
                    print(line, end="", flush=True)
            process.communicate()
            
            if process.returncode != 0:
                # A negative returncode means the worker was killed by a signal rather
                # than exiting on its own. Name it: -15 (SIGTERM) from an external
                # supervisor reads identically to a genuine training error otherwise,
                # and the two want opposite responses (retry vs. fix the config).
                if process.returncode < 0:
                    sig = signal.Signals(-process.returncode).name
                    print(f"Training {model_name} was KILLED by {sig} "
                          f"({-process.returncode}) from outside this process. "
                          f"This is not a training error; the run resumes from its "
                          f"last checkpoint.")
                print(f"Error training {model_name}. Marking as failed and continuing to next model.")
                results[f"{model_name}"] = "FAILED"
                continue
                
            # Extract final checkpoint val_bpb
            # Checkpoint format is usually checkpoints_dir/model/state_*.pt etc
            # Base train saves it with step index. We find the largest one.
            model_ckpt_dir = ckpt_dir / model_name
            if model_ckpt_dir.exists():
                meta_files = glob.glob(str(model_ckpt_dir / "meta_*.json"))
                if meta_files:
                    meta_files.sort()
                    last_meta = meta_files[-1]
                    try:
                        with open(last_meta, "r") as f:
                            meta_data = json.load(f)
                        if "val_bpb" in meta_data and meta_data["val_bpb"] is not None:
                            val_bpb = float(meta_data["val_bpb"])
                            results[model_name] = {"val_bpb": val_bpb, "checkpoint": last_meta}
                            # ── Promote to completed in state ──
                            state.setdefault("completed", {})[model_name] = {
                                "val_bpb": val_bpb,
                                "checkpoint": last_meta,
                                "ckpt_dir": str(ckpt_dir),
                            }
                            state.get("unfinished", {}).pop(model_name, None)
                            save_sweep_state(run_dir_path, state)
                            print(f"Final Validation BPB for {model_name}: {val_bpb:.4f}")
                        else:
                            print(f"No val_bpb found in {last_meta}")
                    except Exception as e:
                        print(f"Failed to load metadata {last_meta}: {e}")
                else:
                    print(f"No meta_*.json files found in {model_ckpt_dir}")
            else:
                 print(f"Checkpoint directory {model_ckpt_dir} does not exist.")
                 
        except Exception as e:
            print(f"Exception during {model_name}: {e}")
            
    # --- Fail fast if any model errored ---
    failed_models = [n for n, v in results.items() if v == "FAILED"]

    # --- Generate Report and Plot ---
    if not results:
        print("No results collected to plot.")
        if failed_models:
            print(f"Failed models: {failed_models}")
            sys.exit(1)
        return
        
    print("\n--- Generating Report ---")
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    
    names = list(results.keys())
    # Filter out failed runs (stored as string "FAILED", not a result dict)
    names = [n for n in names if isinstance(results[n], dict)]
    if not names:
        print("No successful runs to plot.")
        if failed_models:
            print(f"\n[ERROR] The following models FAILED: {failed_models}")
            sys.exit(1)
        return
    # Ensure all collected BPBs are floats for math
    bpbs = [float(results[n]["val_bpb"]) for n in names]

    bars = plt.bar(names, bpbs, color=sns.color_palette("husl", len(names)))
    
    plt.title(f"Validation BPB Comparison at Depth {depth} ({target_tokens:,} tokens)", fontsize=14)
    plt.ylabel("Validation Bits Per Byte (lower is better)", fontsize=12)
    plt.ylim(float(min(bpbs)) * 0.95, float(max(bpbs)) * 1.05) # Zoom in for better contrast

    # Add exact values on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.4f}', va='bottom', ha='center', fontsize=10)
        
    plt.tight_layout()
    plot_path = run_dir_path / f"comparison_depth_{depth}.png"
    plt.savefig(plot_path)
    print(f"Saved plot to {plot_path}")
    
    # Save TSV data
    tsv_path = run_dir_path / f"results_depth_{depth}.tsv"
    with open(tsv_path, "w") as f:
        f.write("model_name\tval_bpb\n")
        for name, data in results.items():
            if isinstance(data, dict):
                f.write(f"{name}\t{data['val_bpb']}\n")
            else:
                f.write(f"{name}\tFAILED\n")
    print(f"Saved TSV data to {tsv_path}")

    if failed_models:
        print(f"\n[ERROR] The following models FAILED: {failed_models}")
        sys.exit(1)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", type=int, required=True)
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--aspect-ratio", type=int, default=0, help="model_dim = depth * aspect_ratio (0 = use defaults)")
    parser.add_argument("--head-dim", type=int, default=0, help="attention head dimension (0 = default 128)")
    parser.add_argument("--model-dim", type=int, default=0, help="Explicit model_dim override for base_train.py")
    parser.add_argument("--fp8", action="store_true", help="Enable FP8 training (Blackwell optimization)")
    parser.add_argument("--tokenizer-dir", type=str, default=None, help="explicit tokenizer directory")
    parser.add_argument("--data-dir", type=str, default=None, help="explicit data directory")
    parser.add_argument("--max-shards", type=int, default=-1, help="maximum number of dataset shards to use")
    parser.add_argument("--target-tokens", type=int, default=-1, help="explicit number of tokens to train for per model")
    parser.add_argument("--num-iterations", type=int, default=-1, help="explicit iteration count, overriding every token/FLOP budget (-1 = disable). For short diagnostic runs.")
    parser.add_argument("--target-param-data-ratio", type=float, default=-1.0, help="Chinchilla token:param ratio (e.g. 20.0); -1 = use base_train.py default (10.5)")
    parser.add_argument("--target-flops", type=float, default=-1.0, help="calculate num_iterations per model architecture to reach exact target FLOPs (-1 = disable)")
    parser.add_argument("--target-active-flops", type=float, default=-1.0, help="as --target-flops but on the ACTIVE FLOPs/token axis, which is the one the Pareto plots use. Required for an isoFLOP profile that mixes gated and dense arms (-1 = disable)")
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True, help="enable/disable torch.compile")
    parser.add_argument("--warmup-ratio", type=float, default=0.05, help="base warmup ratio passed to all runs")
    parser.add_argument("--warmdown-ratio", type=float, default=0.7, help="ratio of iterations for LR warmdown (rest is constant LR)")
    parser.add_argument("--final-lr-frac", type=float, default=0.0, help="final LR as fraction of peak LR (eta_min)")
    parser.add_argument("--models", type=str, default="all", help="Comma-separated list of models to run (e.g. 'base,remixed-linear'), or 'all'")
    parser.add_argument("--research-warmup-ratio", type=float, default=0.05, help="research-branch warmup ratio for OneCycle")
    parser.add_argument("--use-onecycle", type=int, default=1, choices=[0, 1], help="research branches: 1=OneCycle, 0=use base schedule")
    
    # New flags for run configuration
    parser.add_argument("--device-batch-size", type=int, default=-1, help="override per-device batch size")
    parser.add_argument("--total-batch-size", type=int, default=-1, help="override total batch size")
    parser.add_argument("--log-every", type=int, default=1, help="logging frequency")
    parser.add_argument("--eval-every", type=int, default=-1, help="evaluation frequency (-1 = at end)")
    parser.add_argument("--save-every", type=int, default=-1, help="checkpoint frequency")
    parser.add_argument("--core-metric-every", type=int, default=-1, help="core metric frequency")
    parser.add_argument("--skip-core", action="store_true", help="completely disable CORE metric evaluation")
    parser.add_argument("--mu-p-mode", type=str, default="base_only", choices=["disable", "base_only", "enable"], help="mu-P scaling logic")
    parser.add_argument("--sequence-len", type=int, default=2048, help="override max sequence length")
    parser.add_argument("--router-context-window", type=int, default=-1, help="override sliding window size for contextual router (-1 for full sequence)")
    # Research dimension override
    parser.add_argument("--research-dim", type=int, default=0, help="override default 1/8th model_dim for research branches (MoE/Remix)")
    parser.add_argument("--remix-basis-size", type=int, default=0, help="explicit basis_size for remixed-linear (0 = auto via scale_basis_size; set to model_dim for full-rank)")
    # Remixed-linear components
    parser.add_argument("--remix-use-basis-gate", type=int, default=1, choices=[0, 1], help="enable basis gating in remixed linear (1/0)")
    parser.add_argument("--remix-use-output-gate", type=int, default=1, choices=[0, 1], help="enable output gating in remixed linear (1/0)")
    parser.add_argument("--remix-use-context", type=int, default=1, choices=[0, 1], help="enable context modulation in remixed linear (1/0)")
    parser.add_argument("--remix-basis-gate-mode", type=str, default="mlp", choices=["mlp", "linear", "centered", "attn", "none", "random", "lowrank"], help="basis gate architecture")
    parser.add_argument("--p22-n-templates", type=int, default=1, help="22: number of template_mixing matrices (1=standard, K>1=MoE routing)")
    parser.add_argument("--p22-template-routing-learned", type=int, default=0, choices=[0, 1], help="22: learned template routing (0=frozen, 1=learned)")
    parser.add_argument("--p22-template-topk", type=int, default=0, help="22: hard top-k for legacy template bank")
    parser.add_argument("--p22-attn-moe-route", type=str, default="none", choices=["none", "sequence", "token"], help="22: MoE routing for attention Q/K/V/Proj")
    parser.add_argument("--remix-gate-lr-scale", type=float, default=0.3, help="remix: learning rate scale for gate parameters")
    parser.add_argument("--p26-output-gated-linear", type=int, default=0, choices=[0, 1], help="26: use OutputGatedLinear (single W + low-rank output gate, no factorization)")
    parser.add_argument("--p28-shared-basis", type=int, default=0, choices=[0, 1], help="28C: share single W_b projection across all attn Q/K/V/O per block")
    parser.add_argument("--p28-chunk-routing-size", type=int, default=0, help="28D: amortize template routing over N-token chunks (0=per-token)")
    parser.add_argument("--p28-global-template-bank", type=str, default="none", choices=["none", "ffn", "all"], help="28E/F: cross-layer global template bank mode")
    parser.add_argument("--p28-attn-proj-templates", type=int, default=0, help="28C2: override n_templates for attn c_proj (0=default)")
    parser.add_argument("--p28-attn-qk-templates",   type=int, default=0, help="28C3: override n_templates for attn c_q/c_k (0=default)")
    parser.add_argument("--p31-chunk-route-impl", type=str, default="compose", choices=["compose", "grouped"], help="31: chunk-routing evaluation. 'compose'=materialize per-chunk W_eff (legacy); 'grouped' (topk=1 only)=one GEMM per template, no W_eff in HBM")
    parser.add_argument("--p31-top1-gate", type=str, default="ones", choices=["ones", "switch"], help="31: top-1 routing coefficient. 'ones'=legacy (zero router gradient); 'switch'=full-softmax prob of selected template (differentiable)")
    parser.add_argument("--p31-template-delta-rank", type=int, default=0, help="31: replace the (K, out, basis) template bank with a shared base + K rank-r deltas (0=off)")
    parser.add_argument("--p31-route-side", type=str, default="output", choices=["output", "basis", "narrow"], help="31: which factor carries the routing. 'output'=route W_m (legacy); 'basis'=route W_b with shared W_m; 'narrow'=route the smaller factor per projection")
    parser.add_argument("--p31-basis-side-templates", type=int, default=0, help="31: template count for basis-routed projections (0=use n_templates, -1=auto scale by out//basis for param parity)")
    parser.add_argument("--p31-drop-basis-proj", type=int, default=0, choices=[0, 1], help="31: drop W_b entirely — h=LN(x) then a routed (out,in) template. Applied only where in_features <= basis_size (c_fc, attention); skipped on c_proj")
    parser.add_argument("--remix-template-block-diag", type=int, default=0, choices=[0, 1], help="29: block-diagonal Muon for template_bank")
    parser.add_argument("--remix-template-lr-scale",   type=float, default=1.0, help="29: LR multiplier for template_bank Muon group")
    parser.add_argument("--p29-grad-equalize",         type=int, default=0, choices=[0, 1], help="29: per-template gradient equalization to prevent template collapse")
    parser.add_argument("--target-active-params",     type=int, default=0, choices=[0, 1], help="use active params for target_tokens budget")
    parser.add_argument("--compile-regional",         type=int, default=0, choices=[0, 1], help="compile each transformer layer separately; compile time stops growing with depth")
    parser.add_argument("--compile-mode",             type=str, default="default", help="torch.compile mode passed through to base_train (default | reduce-overhead | max-autotune)")
    parser.add_argument("--timing-probe-steps",       type=int, default=0, help="stop each run after N steps, having derived the schedule from the real budget; used to cost a sweep")
    parser.add_argument("--remix-basis-gate-rank", type=int, default=8, help="rank for lowrank basis gate mode")
    # CCL block modulation
    parser.add_argument("--cclblock-modulation", type=str, default="weight",
                        choices=["weight", "normalization", "householder", "spectral", "ocd", "lie", "polynomial", "grassmann", "decoupled", "tucker", "svs", "vq", "dcu", "fsi", "aesp", "ckr", "ckr_ffn", "com", "giad", "psg", "splitstream", "lokr", "pgr", "cil", "prb", "arg", "kfl", "cond"],
                        help="CCL block strategy: 'weight' (RemixedLinear+SelectiveContextStream) "
                             "or 'normalization' (CCLBlock with AdaRMSNorm)")
    parser.add_argument("--cclblock-orth-lambda", type=float, default=0.0,
                        help="OCD overlap penalty weight (0 disables)")
    parser.add_argument("--p35-template-diversity-lambda", type=float, default=0.0,
                        help="Template bank diversity loss weight — penalizes pairwise cosine similarity (0 disables)")
    parser.add_argument("--cclblock-context-stream", type=str, default="local", 
                        choices=["local", "shifted", "ema", "selective", "multiscale", "ssm", "boundary", "chunk", "predictive_chunk", "evidence_ssm", "dacs", "prefix", "warmup_ema", "dacs_ema", "decay_prefix"],
                        help="Context stream type")
    parser.add_argument("--cclblock-ema-factor", type=float, default=0.99,
                        help="EMA factor for the legacy EMAContextStream")
    parser.add_argument("--cclblock-stale-ctx-lag", type=int, default=0,
                        help="Design C stale context lag (0=disabled, k>=1 = context from k blocks ago)")
    # Novel ablation designs
    parser.add_argument("--cclblock-sparse-gate-k", type=int, default=0,
                        help="Design 3: sparse top-k basis gate (0=off, N=top-N)")
    parser.add_argument("--cclblock-gate-temperature", type=float, default=1.0,
                        help="Design 6: gate temperature (<1=sharper, >1=softer)")
    parser.add_argument("--cclblock-context-bank-size", type=int, default=0,
                        help="Design 4: context prototype bank size (0=off, e.g. 16)")
    parser.add_argument("--cclblock-per-head-ctx", type=int, default=0, choices=[0, 1],
                        help="Design 7: separate attn/ffn context projections (0=off, 1=on)")
    parser.add_argument("--cclblock-context-source", type=str, default="norm_x",
                        choices=["norm_x", "attn_heads", "attn_geometry"],
                        help="Design 2: context source ('norm_x'=residual, 'attn_heads'=query vectors)")
    # Phase 8
    parser.add_argument("--cclblock-chunk-size", type=int, default=0)
    parser.add_argument("--cclblock-aux-objective", type=str, default="none", choices=["none", "boundary", "entropy"])
    parser.add_argument("--cclblock-aux-lambda", type=float, default=0.1)
    parser.add_argument("--cclblock-boundary-token-id", type=int, default=198)
    # Phase 9
    parser.add_argument("--use-ral", type=int, default=0, choices=[0, 1])
    parser.add_argument("--ral-rank", type=int, default=32)
    # Phase 35: ConditionedLinear (--cclblock-modulation cond)
    parser.add_argument("--p34-ffn-schedule", type=str, default="")
    parser.add_argument("--p34-ffn-no-ffn-replacement", type=str, default="none", choices=["none", "linear"])
    parser.add_argument("--p34-ffn-last-depth", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--p36-swiglu-ffn", type=int, default=0, choices=[0, 1])
    parser.add_argument("--p36-swiglu-mult", type=float, default=8.0/3.0)
    parser.add_argument("--cond-sites", type=str, default="both", choices=["both", "attn", "ffn"])
    parser.add_argument("--p22-route-affine", type=int, default=0, choices=[0, 1])
    parser.add_argument("--cond-rank", type=int, default=256)
    parser.add_argument("--cond-mult-steps", type=int, default=0)
    parser.add_argument("--cond-gate-source", type=str, default="router", choices=["router", "tied", "ctx"])
    parser.add_argument("--cond-coeff-act", type=str, default="centered", choices=["centered", "linear", "sigmoid", "one"])
    parser.add_argument("--cond-router-rank", type=int, default=0)
    parser.add_argument("--cond-router-act", type=str, default="none")
    parser.add_argument("--cond-live-init", type=float, default=0.0)
    parser.add_argument("--cond-chunk-size", type=int, default=0)
    parser.add_argument("--cond-mult-scale", type=float, default=-1.0)
    parser.add_argument("--cond-mult-impl", type=str, default="wy", choices=["wy", "loop"])
    parser.add_argument("--cond-attn-projs", type=str, default="qkvo")
    parser.add_argument("--cond-layer-frac", type=float, default=1.0)
    parser.add_argument("--cclblock-film-gate", type=int, default=0, choices=[0, 1])
    parser.add_argument("--cclblock-attn-shadow-dim", type=int, default=0)
    parser.add_argument("--cclblock-dynamic-ratio", type=float, default=0.25)
    parser.add_argument("--cclblock-gate-rank", type=int, default=8)
    parser.add_argument("--cclblock-num-regimes", type=int, default=8)
    parser.add_argument("--cclblock-regime-temperature", type=float, default=1.0)
    parser.add_argument("--cclblock-poly-order", type=int, default=2)
    parser.add_argument("--cclblock-lie-generators", type=int, default=4)
    parser.add_argument("--cclblock-grassmann-bank-size", type=int, default=4)
    parser.add_argument("--cclblock-tucker-rank", type=int, default=32)
    parser.add_argument("--cclblock-tucker-modes", type=int, default=8)
    parser.add_argument("--cclblock-svs-rank", type=int, default=64)
    parser.add_argument("--cclblock-svs-eps", type=float, default=0.1)
    parser.add_argument("--cclblock-vq-codes", type=int, default=8)
    parser.add_argument("--cclblock-vq-temperature", type=float, default=1.0)
    parser.add_argument("--cclblock-dcu-warmup-steps", type=int, default=0)
    # Phase 12: FSI/AESP/CKR
    parser.add_argument("--cclblock-fsi-rotations", type=int, default=8)
    parser.add_argument("--cclblock-fsi-selector-dim", type=int, default=64)
    parser.add_argument("--cclblock-aesp-strata", type=int, default=4)
    parser.add_argument("--cclblock-aesp-delta-rank", type=int, default=4)
    parser.add_argument("--cclblock-ckr-branches", type=int, default=4)
    parser.add_argument("--cclblock-ckr-kernel-size", type=int, default=64)
    # Phase 13: CKR enhancements
    parser.add_argument("--cclblock-ckr-pos-channels", type=int, default=1)
    parser.add_argument("--cclblock-ckr-dual-optim", type=int, default=0, choices=[0, 1])
    parser.add_argument("--cclblock-ckr-content-bias", type=float, default=0.0)
    # Phase 14: GIAD/PSG/SplitStream
    parser.add_argument("--cclblock-giad-rank", type=int, default=32)
    parser.add_argument("--cclblock-psg-kernel-size", type=int, default=64)
    parser.add_argument("--cclblock-ss-dynamic-ratio", type=float, default=0.25)
    parser.add_argument("--cclblock-ss-branches", type=int, default=2)
    parser.add_argument("--cclblock-ss-kernel-size", type=int, default=64)
    # Phase 15: LoKR
    parser.add_argument("--cclblock-lokr-branches", type=int, default=8)
    parser.add_argument("--cclblock-lokr-rank", type=int, default=16)
    # Phase 16: CKR-Anneal / COM
    parser.add_argument("--cclblock-ckr-temp-start", type=float, default=2.0)
    parser.add_argument("--cclblock-ckr-temp-end", type=float, default=0.3)
    parser.add_argument("--cclblock-com-kernel-size", type=int, default=32)
    # Phase 17
    parser.add_argument("--cclblock-ckr-ortho-init", type=int, default=0, choices=[0, 1])
    parser.add_argument("--cclblock-ckr-branch-dropout", type=float, default=0.0)
    parser.add_argument("--cclblock-ckr-diversity-lambda", type=float, default=0.0)
    parser.add_argument("--cclblock-pgr-kernel-size", type=int, default=64)
    parser.add_argument("--cclblock-cil-kernel-size", type=int, default=64)
    parser.add_argument("--cclblock-prb-kernel-size", type=int, default=64)
    parser.add_argument("--modulation-diagnostics", type=int, default=0, choices=[0, 1])
    # Phase 18
    parser.add_argument("--p18-layer-drop", type=float, default=0.0)
    parser.add_argument("--p18-dynamic-activation", type=int, default=0, choices=[0, 1])
    parser.add_argument("--p18-mixture-norm", type=int, default=0, choices=[0, 1])
    parser.add_argument("--p18-aux-sim-lambda", type=float, default=0.0)
    parser.add_argument("--p18-gradient-penalty", type=float, default=0.0)
    parser.add_argument("--p18-per-channel-scale", type=int, default=0, choices=[0, 1])
    # Phase 19
    parser.add_argument("--p19-residual-gate", type=int, default=0, choices=[0, 1])
    parser.add_argument("--p19-head-importance", type=int, default=0, choices=[0, 1])
    parser.add_argument("--p19-residual-mix-groups", type=int, default=0)
    parser.add_argument("--p19-attn-logit-bias", type=int, default=0, choices=[0, 1])
    parser.add_argument("--p19-residual-decay", type=int, default=0, choices=[0, 1])
    parser.add_argument("--p19-grad-equilibrium", type=float, default=0.0)
    parser.add_argument("--p19-spectral-reparam", type=int, default=0, choices=[0, 1, 2])
    parser.add_argument("--p19-weight-anticollapse", type=float, default=0.0)
    parser.add_argument("--p19-ve-bias", type=int, default=0, choices=[0, 1])
    parser.add_argument("--p19-weight-noise", type=float, default=0.0)
    # Phase 20
    parser.add_argument("--p20-hrcs-scale", type=int, default=0)
    parser.add_argument("--p20-lswr-scale", type=int, default=0)
    parser.add_argument("--p20-lswr-planes", type=int, default=8)
    parser.add_argument("--p20-lrcfb-branches", type=int, default=0)
    parser.add_argument("--p20-lrcfb-narrow", type=int, default=0, choices=[0, 1])
    parser.add_argument("--p20-lrcfb-learned", type=int, default=0, choices=[0, 1])
    parser.add_argument("--p20-lrcfb-topk", type=int, default=0)
    parser.add_argument("--p20-dgcr-branches", type=int, default=0)
    parser.add_argument("--p20-dgcr-aux-weight", type=float, default=0.01)
    parser.add_argument("--p20-mone-experts", type=int, default=0)
    parser.add_argument("--p20-mone-topk", type=int, default=0)
    parser.add_argument("--p20-mone-narrow", type=int, default=1, choices=[0, 1])
    parser.add_argument("--p20-mone-frozen", type=int, default=0, choices=[0, 1])
    parser.add_argument("--p20-ncea-branches", type=int, default=0)
    parser.add_argument("--p20-ncea-eps", type=float, default=0.1)
    parser.add_argument("--p20-adwi", type=int, default=0, choices=[0, 1])
    # Phase 2 proposals
    parser.add_argument("--p20-pwu-branches", type=int, default=0)
    parser.add_argument("--p20-pwu-phase", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--p20-fsvd-gate", type=int, default=0, choices=[0, 1])
    parser.add_argument("--p20-wbfc-clusters", type=int, default=0)
    parser.add_argument("--p20-wbfc-active", type=int, default=0)
    # Phase 21
    parser.add_argument("--p21-per-experts", type=int, default=0)
    parser.add_argument("--p21-per-topk", type=int, default=0)
    parser.add_argument("--p21-per-learned", type=int, default=0, choices=[0, 1])
    parser.add_argument("--p21-per-attn", type=int, default=0, choices=[0, 1])
    # Phase 23: Tiny Experts RemixedLinear + Standard MoE baseline
    parser.add_argument("--p23-tiny-expert", type=int, default=0, choices=[0, 1])
    parser.add_argument("--p23-n-experts", type=int, default=64)
    parser.add_argument("--p23-topk", type=int, default=16)
    parser.add_argument("--p23-learned-route", type=int, default=0, choices=[0, 1])
    parser.add_argument("--p23-std-moe-experts", type=int, default=0)
    parser.add_argument("--p23-std-moe-topk", type=int, default=-1)
    parser.add_argument("--p23-std-moe-aux-weight", type=float, default=0.01)
    parser.add_argument("--p23-lokr", type=int, default=0, choices=[0, 1])
    parser.add_argument("--p23-lokr-rank", type=int, default=4)
    parser.add_argument("--p23-use-shared-block-router", type=int, default=0, choices=[0, 1])
    parser.add_argument("--p23-linear-moe-experts", type=int, default=0, help="23: enable weight-space LinearMoE with K experts (0=off)")
    parser.add_argument("--p23-linear-moe-topk", type=int, default=0, help="23: top-k selected experts in LinearMoE (0=soft all-expert blend)")
    parser.add_argument("--p23-quantile-route", type=int, default=0, choices=[0, 1, 2], help="23: 1=EMA quantile routing, 2=Causal Expert Cross-Attention")
    parser.add_argument("--p34-ffn-mult", type=float, default=4.0, help="34: FFN hidden width multiple of n_embd")
    parser.add_argument("--p34-ffn-single", type=int, default=0, choices=[0, 1], help="34: single RemixedLinear D->D as the whole FFN")
    parser.add_argument("--p34-dense-attn", type=int, default=0, choices=[0, 1], help="34: dense attention, remix the FFN only")
    parser.add_argument("--seed", type=int, default=-1, help="RNG seed for weight init (-1 = unseeded). Used by the seed-variance ablation.")
    def _add_unique(*opt, **kwargs):
        # Guard against accidental duplicate definitions across merged branches.
        if any(o in parser._option_string_actions for o in opt):
            return
        parser.add_argument(*opt, **kwargs)

    _add_unique("--p24-use-sliced-weight", type=int, default=0, choices=[0, 1])
    _add_unique("--p24-sliced-weight-reduction-scale", type=int, default=8)
    _add_unique("--p24-sliced-weight-min-select", type=int, default=128)
    _add_unique("--p24-sliced-weight-scope", type=str, default="per_token", choices=["per_token", "per_block", "global"])
    _add_unique("--p24-sliced-weight-balance-coeff", type=float, default=0.01)
    _add_unique("--p24-quantile-route", type=int, default=0, choices=[0, 1, 2])
    _add_unique("--p24-use-folded-mod", type=int, default=0, choices=[0, 1])
    _add_unique("--p24-folded-mod-reduction-scale", type=int, default=8)
    _add_unique("--p24-folded-mod-scope", type=str, default="per_layer", choices=["per_layer", "per_block", "global"])
    _add_unique("--p24-folded-mod-gate-act", type=str, default="tanh_centered", choices=["sigmoid", "tanh_centered"])
    _add_unique("--p24-folded-mod-min-dim", type=int, default=128, help="floor on folded_dim (0=no floor, 128=match min_select default)")
    _add_unique("--p24-use-sequence-gated-linear", type=int, default=0, choices=[0, 1])
    _add_unique("--p24-sequence-gated-scope", type=str, default="per_layer", choices=["per_layer", "per_block", "global"])
    _add_unique("--p24-sequence-gated-act", type=str, default="tanh_centered", choices=["sigmoid", "tanh_centered"])
    parser.add_argument("--remix-shared-context-gates", type=int, default=0, choices=[0, 1], help="23: batch context gates")
    parser.add_argument("--remix-use-dual-gate", type=int, default=0, choices=[0, 1], help="25: use DualGateLinear instead of RemixedLinear")
    parser.add_argument("--remix-basis-scale-factor", type=int, default=4, help="basis compression: 4=C//4, 1=full rank")
    parser.add_argument("--remix-output-gate-rank", type=int, default=16, help="output gate rank")
    parser.add_argument("--use-chunked-remix", type=int, default=0, choices=[0, 1],
                        help="1 = activate ChunkedRemixConfig canonical P29 defaults for remixed-linear runs; "
                             "individual flags in the sweep still override on top")
    # Phase 30: LayerNorm ablation
    parser.add_argument("--remix-disable-ln-basis", type=int, default=0, choices=[0, 1], help="30B: disable intermediate LN in RemixedLinear")
    parser.add_argument("--dense-intermediate-ln", type=int, default=0, choices=[0, 1], help="30A: add intermediate LN to dense projections")
    # MoL: Mixture of Layers baseline (arXiv:2605.09516)
    parser.add_argument("--use-mol", type=int, default=0, choices=[0, 1], help="MoL: enable Mixture-of-Layers baseline")
    parser.add_argument("--mol-n-blocks", type=int, default=5, help="MoL: N thin blocks per split stage")
    parser.add_argument("--mol-n-shared", type=int, default=1, help="MoL: S always-active blocks")
    parser.add_argument("--mol-topk", type=int, default=3, help="MoL: k routed blocks per token")
    parser.add_argument("--mol-thin-dim", type=int, default=256, help="MoL: d_thin")
    parser.add_argument("--mol-head-dim", type=int, default=64, help="MoL: head dim, pinned across widths")
    parser.add_argument("--mol-ffn-mult", type=float, default=4.0, help="MoL: d_ff,thin = mult * d_thin")
    parser.add_argument("--mol-router-aux", type=float, default=0.05, help="MoL: CV^2 balance weight")
    parser.add_argument("--mol-routed-attn", type=str, default="softmax", help="MoL: routed-block attention")
    parser.add_argument("--mol-dispatch", type=int, default=1, choices=[0, 1], help="MoL: 1=gather/scatter (default), 0=masked reference")
    parser.add_argument("--mol-capacity-factor", type=float, default=1.0, help="MoL: dispatch capacity factor")
    parser.add_argument("--mol-per-block-ve", type=int, default=0, choices=[0,1], help="MoL: per-block VE slices")
    parser.add_argument("--mol-block-lr-scale", type=float, default=1.0, help="MoL: per-thin-block LR multiplier")
    # MST: Modular Sub-Transformer
    parser.add_argument("--use-binary", type=int, default=0)
    parser.add_argument("--binary-acts", type=int, default=1)
    parser.add_argument("--binary-weight-scale", type=str, default='row')
    parser.add_argument("--binary-act-scale", type=str, default='token')
    parser.add_argument("--binary-clip", type=float, default=1.0)
    parser.add_argument("--binary-linear", type=int, default=1)
    parser.add_argument("--binary-embeddings", type=int, default=1)
    parser.add_argument("--binary-skip", type=str, default='')
    parser.add_argument("--use-mst", type=int, default=0, choices=[0, 1], help="MST: enable Modular Sub-Transformer mode")
    parser.add_argument("--mst-n-subs", type=int, default=8, help="MST: number of sub-transformers")
    parser.add_argument("--mst-sub-dim", type=int, default=64, help="MST: dimension per sub-transformer")
    parser.add_argument("--mst-head-dim", type=int, default=0, help="MST: attention head_dim (0=auto d//n_head)")
    parser.add_argument("--mst-input-mode", type=str, default="fixed_slice",
                        choices=["fixed_slice", "learned_proj", "rotated_slice", "per_sub_embed", "stem"])
    parser.add_argument("--mst-rotated-slice-learned", type=int, default=0, choices=[0, 1])
    parser.add_argument("--mst-routing-mode", type=str, default="soft_weighted",
                        choices=["soft_weighted", "topk_hard", "sequence_path"])
    parser.add_argument("--mst-routing-topk", type=int, default=4)
    parser.add_argument("--mst-routing-aux-weight", type=float, default=0.01)
    parser.add_argument("--mst-diversity-weight", type=float, default=0.0)
    parser.add_argument("--mst-ffn-mode", type=str, default="standard",
                        choices=["standard", "no_downproj", "linear", "shared_dense", "grouped_up_shared_down", "shared_swiglu"])
    parser.add_argument("--mst-ffn-every", type=int, default=1, help="MST: apply FFN every N layers (1=every, 2=even-only)")
    parser.add_argument("--mst-transition-mode", type=str, default="parallel",
                        choices=["parallel", "aggregate_distribute", "cross_attend", "concat_proj", "free_for_all", "micro_attention", "micro_attention_shared_kv"])
    parser.add_argument("--mst-final-mode", type=str, default="aggregate_proj",
                        choices=["aggregate_proj", "weighted_logits", "concat_proj"])
    parser.add_argument("--mst-final-topk", type=int, default=-1)
    parser.add_argument("--mst-ffn-shared-up", type=int, default=0)
    parser.add_argument("--mst-ffn-inner-dim", type=int, default=0)
    parser.add_argument("--mst-sub-dropout", type=float, default=0.0)
    parser.add_argument("--mst-transition-every", type=int, default=1)
    parser.add_argument("--mst-ffa-temperature", type=float, default=1.0)
    parser.add_argument("--mst-global-residual", type=int, default=0)
    parser.add_argument("--mst-hybrid-dense", type=int, default=0)
    parser.add_argument("--mst-cross-sub-kv", type=int, default=0)
    # Stage 5 features
    parser.add_argument("--mst-sub-aux-weight", type=float, default=0.0, help="H3: per-sub auxiliary prediction loss weight")
    parser.add_argument("--mst-progressive-merge", type=int, default=0, choices=[0, 1], help="N1: pyramid sub-merging")
    parser.add_argument("--mst-multi-scale-windows", type=int, default=0, choices=[0, 1], help="W1: per-sub multi-scale windows")
    parser.add_argument("--mst-delta-residual", type=int, default=0, choices=[0, 1], help="DR1: delta residual mode")
    parser.add_argument("--mst-sub-layers", type=int, default=1, help="SL1: layers per sub-transformer")
    # Stage 7 features (P07)
    parser.add_argument("--mst-grad-equalize", type=int, default=0, choices=[0, 1])
    parser.add_argument("--mst-block-diagonal-muon", type=int, default=0, choices=[0, 1])
    parser.add_argument("--mst-transition-width-mult", type=float, default=1.0)
    parser.add_argument("--mst-sub-lr-scale", type=float, default=1.0)
    parser.add_argument("--mst-shared-expert", type=int, default=0, choices=[0, 1])
    parser.add_argument("--mst-router-entropy-weight", type=float, default=0.0)
    parser.add_argument("--mst-shared-kv-attn", type=int, default=0, choices=[0, 1])
    parser.add_argument("--mst-contrastive-diversity-weight", type=float, default=0.0)
    # MST Stage 8: Transition expressivity
    parser.add_argument("--mst-transition-nonlinear", type=int, default=0, choices=[0, 1])
    parser.add_argument("--mst-transition-gated", type=int, default=0, choices=[0, 1])
    parser.add_argument("--mst-transition-mlp", type=int, default=0, choices=[0, 1])
    # MST Stage 9: Cross-sub expressivity
    parser.add_argument("--mst-cross-sub-gate", type=int, default=0)
    parser.add_argument("--mst-hyper-connect", type=int, default=0, choices=[0, 1])
    parser.add_argument("--mst-cross-kv-inject", type=int, default=0, choices=[0, 1])
    # MST Stage 10: Structural transition improvements
    parser.add_argument("--mst-slice-transition", type=int, default=0)
    parser.add_argument("--mst-lookback-layers", type=int, default=0)
    parser.add_argument("--mst-bilinear-transition", type=int, default=0, choices=[0, 1])
    # MST Stage 11: Attention bottleneck + structural improvements
    parser.add_argument("--mst-cross-sub-qmod", type=int, default=0)
    parser.add_argument("--mst-feature-cycle", type=int, default=0, choices=[0, 1])
    parser.add_argument("--mst-mean-transition", type=int, default=0, choices=[0, 1])
    # Stage 12: dense-parity fixes
    parser.add_argument("--mst-sub-head-dim", type=int, default=0,
                        help="G1: per-stream head_dim, heads derived so qkv_dim == sub_dim")
    parser.add_argument("--mst-final-norm", type=int, default=0, choices=[0, 1],
                        help="G2: RMSNorm before lm_head")
    parser.add_argument("--mst-ve-map", type=int, default=0, help="MST: G3-cheap VE map")
    parser.add_argument("--mst-ve-map-rank", type=int, default=0, help="MST: rank of the VE map (0=full)")
    parser.add_argument("--mst-per-stream-ve", type=int, default=0, choices=[0, 1],
                        help="G3: per-stream value embedding slices")
    # Stage 13: overhead cuts
    parser.add_argument("--mst-lm-head-dim", type=int, default=0,
                        help="O1: output-head bottleneck width (0 = n_embd, off)")
    parser.add_argument("--mst-compose-windows", type=int, default=0, choices=[0, 1],
                        help="O2: intersect per-sub windows with the layer window pattern")
    # Stage 14: free cross-stream mixing
    parser.add_argument("--mst-channel-mix", type=str, default='none', choices=['none', 'roll', 'shuffle'],
                        help="permute the stream partition so block-diagonal layers compose into mixing")
    parser.add_argument("--mst-channel-mix-site", type=str, default='layer', choices=['layer', 'ffn', 'both'])
    # Stage 15: coupling optimization + attention cross-stream mixing.
    # NOTE --mst-transition-every is NOT declared here: it already exists above as the
    # Stage 3 flag. Stage 15 only made it functional on the batched path.
    parser.add_argument("--mst-distribute-block-muon", type=int, default=0, choices=[0, 1],
                        help="F1: block-diagonal Muon + sub-LR for distribute_w")
    parser.add_argument("--mst-trans-spectral-lr", type=int, default=0, choices=[0, 1],
                        help="F2: spectral LR scaling for agg_up/agg_down")
    parser.add_argument("--mst-talking-heads", type=int, default=0, choices=[0, 1],
                        help="F4: cross-stream talking-heads mixing before c_proj")
    parser.add_argument("--mst-wo-mode", type=str, default='block', choices=['block', 'dense'],
                        help="F5: attention output projection, block or dense")
    # Stage 16: conditional stream execution
    parser.add_argument("--mst-stream-shared", type=int, default=0, help="MST: S always-active streams")
    parser.add_argument("--mst-stream-topk", type=int, default=0,
                        help="activate k of N streams per token (0=dense)")
    parser.add_argument("--mst-stream-router-aux", type=float, default=0.01,
                        help="load-balancing weight for the stream router")
    parser.add_argument("--mst-stream-router-noise", type=float, default=0.0,
                        help="noisy top-k exploration std on the router logits")
    parser.add_argument("--mst-stream-dispatch", type=int, default=0, choices=[0, 1],
                        help="Phase B gather/scatter dispatch for the gated FFN")
    parser.add_argument("--mst-stream-capacity-factor", type=float, default=1.0)
    parser.add_argument("--mst-stream-gate-attn", type=int, default=0, choices=[0, 1],
                        help="also gate attention QKV, not just the FFN")
    # Stage 17: block-diagonal Shampoo
    parser.add_argument("--mst-ffn-monarch", type=str, default='none',
                        choices=['none', 'shuffle', 'roll'],
                        help="permute the FFN hidden axis between the two block-diagonal factors")
    parser.add_argument("--mst-shampoo", type=int, default=0, choices=[0, 1],
                        help="block-diagonal Shampoo on the stacked per-stream weights")
    parser.add_argument("--mst-precond-every", type=int, default=10,
                        help="steps between Shampoo inverse-root refreshes")
    parser.add_argument("--mst-shampoo-beta", type=float, default=0.95)
    # EET: Early Exit Transformer
    parser.add_argument("--use-eet", type=int, default=0, choices=[0, 1], help="EET: enable Early Exit Transformer")
    parser.add_argument("--eet-frozen-kv", type=int, default=1, choices=[0, 1], help="EET: frozen KV injection (1) or masked attention (0)")
    parser.add_argument("--eet-reenter-final", type=int, default=0, choices=[0, 1], help="EET: force exited tokens to re-enter and be processed by the final layer (1) or not (0)")
    parser.add_argument("--eet-compute-skip", type=int, default=0, choices=[0, 1], help="EET: enable compute-level skipping of intermediate blocks (1/0)")
    parser.add_argument("--eet-target-active-frac", type=float, default=0.125, help="EET: target active token fraction at the deepest routable layer")
    parser.add_argument("--eet-capacity-schedule", type=str, default="bell", choices=["uniform", "linear", "geometric", "bell"])
    parser.add_argument("--eet-exit-fracs", type=str, default="")
    parser.add_argument("--eet-router-type", type=str, default="mlp2", choices=["linear", "mlp1", "mlp2", "attention", "attn"])
    parser.add_argument("--eet-router-hidden", type=int, default=0)
    parser.add_argument("--eet-freq-prior-alpha", type=float, default=0.0)
    parser.add_argument("--eet-pos-prior-beta", type=float, default=0.0)
    parser.add_argument("--eet-domain-prior", type=int, default=0, choices=[0, 1])
    parser.add_argument("--eet-warmup-frac", type=float, default=0.02)
    parser.add_argument("--eet-explore-frac", type=float, default=0.15)
    parser.add_argument("--eet-reconstruct-lambda", type=float, default=1.0)
    parser.add_argument("--eet-efficiency-lambda-start", type=float, default=0.01)
    parser.add_argument("--eet-efficiency-lambda-end", type=float, default=0.1)
    parser.add_argument("--eet-translator-rank", type=int, default=0)
    parser.add_argument("--eet-max-frozen-kv-frac", type=float, default=0.75)
    parser.add_argument("--eet-exit-threshold", type=float, default=0.5)
    parser.add_argument("--eet-min-exit-layer", type=int, default=1)
    parser.add_argument("--eet-loss-variant", type=str, default="reconstruct")
    parser.add_argument("--eet-topk-vocab", type=int, default=512)
    parser.add_argument("--eet-entropy-lambda", type=float, default=0.3)
    parser.add_argument("--eet-surprise-lambda", type=float, default=0.1)
    parser.add_argument("--eet-adv-lambda", type=float, default=1.0)
    parser.add_argument("--eet-adv-entropy-lambda", type=float, default=0.2)
    parser.add_argument("--eet-quality-lambda", type=float, default=1.0)
    parser.add_argument("--eet-quality-entropy-bonus", type=float, default=0.1)
    parser.add_argument("--eet-gumbel-temp-start", type=float, default=0.0)
    parser.add_argument("--eet-gumbel-temp-end", type=float, default=0.1)
    parser.add_argument("--eet-gumbel-hard", type=int, default=1, choices=[0, 1])
    parser.add_argument("--eet-commitment-beta", type=float, default=0.1)
    parser.add_argument("--eet-global-router", type=int, default=0, choices=[0, 1])
    parser.add_argument("--eet-freq-efficiency-alpha", type=float, default=0.0)
    parser.add_argument("--eet-diversity-lambda", type=float, default=0.0)
    parser.add_argument("--eet-ce-guided-lambda", type=float, default=1.0)
    parser.add_argument("--eet-router-lr-mult", type=float, default=5.0)
    parser.add_argument("--eet-model-lr-mult", type=float, default=1.0)
    parser.add_argument("--eet-depth-weight-type", type=str, default="none", choices=["none", "linear", "ema", "sqrt"])
    parser.add_argument("--eet-depth-weight-max", type=float, default=2.5)
    parser.add_argument("--eet-use-override", type=int, default=0, choices=[0, 1])
    parser.add_argument("--eet-override-prob-start", type=float, default=0.5)
    parser.add_argument("--eet-override-prob-end", type=float, default=0.1)
    parser.add_argument("--eet-capacity-alignment-lambda", type=float, default=0.0)
    parser.add_argument("--eet-router-task-grad", type=int, default=1, choices=[0, 1])
    parser.add_argument("--eet-reinforce-interval", type=int, default=0)
    parser.add_argument("--eet-reinforce-lambda", type=float, default=0.1)
    parser.add_argument("--eet-exit-adapter-rank", type=int, default=0)
    parser.add_argument("--eet-router-after-block", type=int, default=0)
    parser.add_argument("--eet-ffn-skip", type=int, default=0, choices=[0, 1])
    parser.add_argument("--eet-ffn-target-frac", type=float, default=0.50)
    parser.add_argument("--eet-ffn-full-attn", type=int, default=1, choices=[0, 1])
    parser.add_argument("--eet-depth-affine", type=int, default=0, choices=[0, 1])
    parser.add_argument("--eet-capacity-anneal-frac", type=float, default=0.0)
    parser.add_argument("--eet-learned-schedule", type=int, default=0, choices=[0, 1])
    parser.add_argument("--eet-departure-summary", type=int, default=0, choices=[0, 1])
    parser.add_argument("--eet-route-consistency-lambda", type=float, default=0.0)
    parser.add_argument("--eet-dense-distill-interval", type=int, default=0)
    parser.add_argument("--eet-dense-distill-lambda", type=float, default=0.5)
    parser.add_argument("--eet-depth-lr-scale", type=int, default=0, choices=[0, 1])
    parser.add_argument("--eet-depth-grad-scale", type=int, default=0, choices=[0, 1])
    parser.add_argument("--eet-detach-aux-from-backbone", type=int, default=0, choices=[0, 1])
    parser.add_argument("--use-code-head", type=int, default=0, help="SCH: passthrough to base_train (see nanochat/code_head.py)")
    parser.add_argument("--sch-head-type", type=str, default='code', help="SCH: passthrough to base_train (see nanochat/code_head.py)")
    parser.add_argument("--sch-bits", type=int, default=0, help="SCH: passthrough to base_train (see nanochat/code_head.py)")
    parser.add_argument("--sch-order", type=int, default=2, help="SCH: passthrough to base_train (see nanochat/code_head.py)")
    parser.add_argument("--sch-max-m", type=int, default=0, help="SCH: passthrough to base_train (see nanochat/code_head.py)")
    parser.add_argument("--sch-phi-mode", type=str, default='monomial', help="SCH: passthrough to base_train (see nanochat/code_head.py)")
    parser.add_argument("--sch-code-mode", type=str, default='binary', help="SCH: passthrough to base_train (see nanochat/code_head.py)")
    parser.add_argument("--sch-code-path", type=str, default='', help="SCH: passthrough to base_train (see nanochat/code_head.py)")
    parser.add_argument("--sch-code-ecc-bits", type=int, default=0, help="SCH: passthrough to base_train (see nanochat/code_head.py)")
    parser.add_argument("--sch-code-seed", type=int, default=1234, help="SCH: passthrough to base_train (see nanochat/code_head.py)")
    parser.add_argument("--sch-phi-density", type=float, default=0.0, help="SCH: passthrough to base_train (see nanochat/code_head.py)")
    parser.add_argument("--sch-phi-dtype", type=str, default='bf16', help="SCH: passthrough to base_train (see nanochat/code_head.py)")
    parser.add_argument("--sch-phi-normalize", type=int, default=1, help="SCH: passthrough to base_train (see nanochat/code_head.py)")
    parser.add_argument("--sch-phi-center", type=int, default=0, help="SCH: passthrough to base_train (see nanochat/code_head.py)")
    parser.add_argument("--sch-g-type", type=str, default='linear', help="SCH: passthrough to base_train (see nanochat/code_head.py)")
    parser.add_argument("--sch-g-hidden", type=int, default=0, help="SCH: passthrough to base_train (see nanochat/code_head.py)")
    parser.add_argument("--sch-g-layers", type=int, default=2, help="SCH: passthrough to base_train (see nanochat/code_head.py)")
    parser.add_argument("--sch-g-out-std", type=float, default=0.001, help="SCH: passthrough to base_train (see nanochat/code_head.py)")
    parser.add_argument("--sch-mixture", type=int, default=1, help="SCH: passthrough to base_train (see nanochat/code_head.py)")
    parser.add_argument("--sch-residual-rank", type=int, default=0, help="SCH: passthrough to base_train (see nanochat/code_head.py)")
    parser.add_argument("--sch-logit-act", type=str, default='none', help="SCH: passthrough to base_train (see nanochat/code_head.py)")
    parser.add_argument("--sch-bias", type=int, default=0, help="SCH: passthrough to base_train (see nanochat/code_head.py)")
    parser.add_argument("--sch-input-mode", type=str, default='table', help="SCH: passthrough to base_train (see nanochat/code_head.py)")
    parser.add_argument("--sch-input-hidden", type=int, default=0, help="SCH: passthrough to base_train (see nanochat/code_head.py)")
    parser.add_argument("--sch-product-groups", type=int, default=8, help="SCH: passthrough to base_train (see nanochat/code_head.py)")
    parser.add_argument("--sch-product-codebook", type=int, default=256, help="SCH: passthrough to base_train (see nanochat/code_head.py)")
    parser.add_argument("--sch-product-source", type=str, default="hash", help="SCH: passthrough to base_train (see nanochat/code_head.py)")
    parser.add_argument("--sch-product-impl", type=str, default="dense", help="SCH: passthrough to base_train (see nanochat/code_head.py)")
    parser.add_argument("--sch-phi-whiten", type=int, default=0, help="SCH: passthrough to base_train (see nanochat/code_head.py)")
    parser.add_argument("--sch-mixture-per-phi", type=int, default=0, help="SCH: passthrough to base_train (see nanochat/code_head.py)")
    parser.add_argument("--sch-mixture-topk", type=int, default=0, help="SCH: passthrough to base_train (see nanochat/code_head.py)")
    parser.add_argument("--sch-mixture-aux", type=float, default=0.01, help="SCH: passthrough to base_train (see nanochat/code_head.py)")
    parser.add_argument("--sch-monarch-m1", type=int, default=0, help="SCH: passthrough to base_train (see nanochat/code_head.py)")
    parser.add_argument("--sch-monarch-perm", type=str, default="none", help="SCH: passthrough to base_train (see nanochat/code_head.py)")
    parser.add_argument("--sch-monarch-perm-path", type=str, default="", help="SCH: passthrough to base_train (see nanochat/code_head.py)")
    parser.add_argument("--sch-tier-bounds", type=str, default="", help="SCH: passthrough to base_train (see nanochat/code_head.py)")
    parser.add_argument("--sch-tier-caps", type=str, default="", help="SCH: passthrough to base_train (see nanochat/code_head.py)")
    parser.add_argument("--sch-tier-order", type=str, default="freq", help="SCH: passthrough to base_train (see nanochat/code_head.py)")
    parser.add_argument("--sch-tier-perm-path", type=str, default="", help="SCH: passthrough to base_train (see nanochat/code_head.py)")
    parser.add_argument("--sch-proposal-rank", type=int, default=32, help="SCH: passthrough to base_train (see nanochat/code_head.py)")
    parser.add_argument("--sch-proposal-topk", type=int, default=16384, help="SCH: passthrough to base_train (see nanochat/code_head.py)")
    parser.add_argument("--sch-proposal-samples", type=int, default=4096, help="SCH: passthrough to base_train (see nanochat/code_head.py)")
    parser.add_argument("--sch-proposal-warmup", type=int, default=0, help="SCH: passthrough to base_train (see nanochat/code_head.py)")
    parser.add_argument("--sch-proposal-chunk", type=int, default=2048, help="SCH: passthrough to base_train (see nanochat/code_head.py)")
    parser.add_argument("--sch-proposal-aux", type=float, default=1.0, help="SCH: passthrough to base_train (see nanochat/code_head.py)")
    parser.add_argument("--sch-nfh-mode", type=str, default='cp', help="SCH: passthrough to base_train (see nanochat/code_head.py)")
    parser.add_argument("--sch-nfh-rank", type=int, default=32, help="SCH: passthrough to base_train (see nanochat/code_head.py)")
    parser.add_argument("--sch-nfh-dims", type=str, default='', help="SCH: passthrough to base_train (see nanochat/code_head.py)")
    parser.add_argument("--sch-nfh-groups", type=int, default=3, help="SCH: passthrough to base_train (see nanochat/code_head.py)")
    parser.add_argument("--sch-nfh-views", type=int, default=1, help="SCH: passthrough to base_train (see nanochat/code_head.py)")
    parser.add_argument("--sch-nfh-perm", type=str, default='none', help="SCH: passthrough to base_train (see nanochat/code_head.py)")
    parser.add_argument("--sch-nfh-perm-path", type=str, default='', help="SCH: passthrough to base_train (see nanochat/code_head.py)")
    parser.add_argument("--sch-nfh-g-type", type=str, default='linear', help="SCH: passthrough to base_train (see nanochat/code_head.py)")
    parser.add_argument("--sch-nfh-g-hidden", type=int, default=0, help="SCH: passthrough to base_train (see nanochat/code_head.py)")
    parser.add_argument("--sch-nfh-chunk", type=int, default=256, help="SCH: passthrough to base_train (see nanochat/code_head.py)")
    parser.add_argument("--sch-nfh-smooth", type=int, default=0, help="SCH: passthrough to base_train (see nanochat/code_head.py)")
    parser.add_argument("--sch-rerank-mode", type=str, default='topk', help="SCH: passthrough to base_train (see nanochat/code_head.py)")
    parser.add_argument("--sch-rerank-k", type=int, default=64, help="SCH: passthrough to base_train (see nanochat/code_head.py)")
    parser.add_argument("--sch-rerank-rank", type=int, default=32, help="SCH: passthrough to base_train (see nanochat/code_head.py)")

    parser.add_argument("--sch-holdout-tokens", type=int, default=0, help="SCH: passthrough to base_train (see nanochat/code_head.py)")
    parser.add_argument("--sch-holdout-seed", type=int, default=7, help="SCH: passthrough to base_train (see nanochat/code_head.py)")
    parser.add_argument("--sch-holdout-min-id", type=int, default=256, help="SCH: passthrough to base_train (see nanochat/code_head.py)")
    parser.add_argument("--sch-holdout-mode", type=str, default='target', help="SCH: passthrough to base_train (see nanochat/code_head.py)")
    parser.add_argument("--sch-decile-metrics", type=int, default=1, help="SCH: passthrough to base_train (see nanochat/code_head.py)")
    parser.add_argument("--sch-rank-probe", type=int, default=0, help="SCH: passthrough to base_train (see nanochat/code_head.py)")
    parser.add_argument("--sch-eval-steps", type=int, default=100, help="SCH: passthrough to base_train (see nanochat/code_head.py)")
    parser.add_argument("--eet-detach-exit-from-backbone", type=int, default=0, choices=[0, 1])
    # EET P02 decision tests (see scripts/eet_p02_tests.sh)
    parser.add_argument("--eet-kv-mode", type=str, default="none", choices=["none", "fresh", "stale"])
    parser.add_argument("--eet-route-noise", type=float, default=0.0)
    parser.add_argument("--eet-route-noise-end", type=float, default=-1.0)
    parser.add_argument("--eet-coverage-diag", type=int, default=0, choices=[0, 1])
    parser.add_argument("--eet-kv-eager", type=int, default=0, choices=[0, 1])
    parser.add_argument("--deep-supervision-lambda", type=float, default=0.0)
    parser.add_argument("--deep-supervision-frac", type=float, default=0.125)
    parser.add_argument("--eet-width-power", type=float, default=0.0)
    parser.add_argument("--eet-width-cap", type=float, default=16.0)

    args = parser.parse_args()
    
    run_training_sweep(args)
