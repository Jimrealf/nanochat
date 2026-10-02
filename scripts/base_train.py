"""
Train model. From root directory of the project, run as:

python -m scripts.base_train

or distributed as:

torchrun --nproc_per_node=8 -m scripts.base_train

If you are only on CPU/Macbook, you'll want to train a much much smaller LLM. Example:
python -m scripts.base_train --depth=4 --max-seq-len=512 --device-batch-size=1 --eval-tokens=512 --core-metric-every=-1 --total-batch-size=512 --num-iterations=20
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
import torch
import torch.nn.functional as F
torch.set_num_threads(8)
import gc
import json
import time
import math
import argparse
from dataclasses import asdict
from contextlib import contextmanager

# import wandb
import torch
import torch._dynamo
torch._dynamo.config.cache_size_limit = 1000
torch._dynamo.config.optimize_ddp = False
import torch.distributed as dist

from nanochat.gpt import GPT, GPTConfig, Linear
# Single source of truth for the --sch-* choice lists.  Restating them here by
# hand is how --sch-phi-mode product reached a GPU and died in argparse.
from nanochat.code_head import (CODE_MODES, G_TYPES, HEAD_TYPES, HOLDOUT_MODES,
                                INPUT_MODES, LOGIT_ACTS, MONARCH_PERMS, PHI_DTYPES,
                                PHI_MODES,
                                PRODUCT_IMPLS, PRODUCT_SOURCES)
from nanochat.inductor_compat import guard_triton_block_size
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit, tokenizing_distributed_data_loader_with_state_bos_bestfit
from nanochat.common import compute_init, compute_cleanup, print0, DummyWandb, print_banner, get_base_dir, autodetect_device_type, get_peak_flops, COMPUTE_DTYPE, COMPUTE_DTYPE_REASON, is_ddp_initialized, wrap_model
from nanochat.tokenizer import get_tokenizer, get_token_bytes
from nanochat.checkpoint_manager import save_checkpoint, load_checkpoint
from nanochat.loss_eval import evaluate_bpb
from nanochat.engine import Engine
from nanochat.flash_attention import HAS_FA3, HAS_FA4
from scripts.base_eval import evaluate_core
print_banner()


def check_router_learned(exit_probs, token_ids, orig_model):
    """
    Run this at end of Phase 2 before committing to Phase 3.
    Check if exit_probs correlates with token frequency as TRS predicted.
    Uses orig_model._freq_prior.freq_bias as the authoritative frequency source
    (the exact signal the router saw during training).
    """
    import numpy as np
    from scipy import stats
    
    n_exits = exit_probs.size(-1)
    
    # Per-exit-slot probability stats (diagnose collapse)
    probs_cpu = exit_probs.detach().cpu().float()
    print0(f'[EET DIAGNOSTIC] Exit probability distribution across {n_exits} slots:')
    for slot in range(n_exits):
        slot_probs = probs_cpu[:, :, slot].numpy().ravel()
        label = 'final_layer' if slot == n_exits - 1 else f'exit_{slot}'
        print0(f'  {label}: mean={slot_probs.mean():.6f}  std={slot_probs.std():.6f}  '
               f'min={slot_probs.min():.6f}  max={slot_probs.max():.6f}')
    
    # Expected exit layer under soft distribution
    layer_indices = torch.arange(n_exits, device=exit_probs.device).float()
    expected_exit = (exit_probs * layer_indices).sum(dim=-1)  # (B, T)
    
    flat_ids  = token_ids.view(-1)
    flat_exit = expected_exit.view(-1).detach().cpu().numpy()
    
    print0(f'[EET DIAGNOSTIC] Mean expected exit layer: {flat_exit.mean():.4f}')
    print0(f'[EET DIAGNOSTIC] Std expected exit layer:  {flat_exit.std():.6f}')
    
    # Guard against constant router output (std=0 → Spearman undefined)
    if flat_exit.std() < 1e-8:
        print0('[EET DIAGNOSTIC] ⚠ Router output is CONSTANT across all tokens — no structure learned.')
        print0('[EET DIAGNOSTIC] Router has NOT learned structure yet. Extend Phase 2 or check gradients.')
        return 0.0
    
    # Build frequency signal — prefer _freq_prior.freq_bias (authoritative: exact signal the router saw)
    freq_prior = getattr(orig_model, '_freq_prior', None)
    if freq_prior is not None and hasattr(freq_prior, 'freq_bias'):
        freq_bias = freq_prior.freq_bias  # (vocab_size,), already log-normalized to [0,1]
        log_freq  = freq_bias[flat_ids.clamp(0, freq_bias.size(0) - 1)].cpu().float().numpy()
        freq_src  = '_freq_prior.freq_bias'
    else:
        # Fallback: load raw counts from disk and apply log1p
        import os
        from nanochat.common import get_base_dir
        cache_path = os.path.join(get_base_dir(), 'tokenizer', 'freq_table.pt')
        if os.path.exists(cache_path):
            ft = torch.load(cache_path, map_location='cpu')
            flat_ids_np = flat_ids.cpu().numpy()
            log_freq   = np.log1p(np.array([ft[i].item() for i in flat_ids_np]))
            freq_src   = f'freq_table.pt ({cache_path})'
        else:
            print0('[EET DIAGNOSTIC] ⚠ No frequency source available (no _freq_prior and no freq_table.pt). Skipping correlation.')
            return 0.0
    
    print0(f'[EET DIAGNOSTIC] Frequency source: {freq_src}')
    print0(f'[EET DIAGNOSTIC] log_freq — mean={log_freq.mean():.6f}  std={log_freq.std():.6f}'
           f'  min={log_freq.min():.6f}  max={log_freq.max():.6f}')
    
    if log_freq.std() < 1e-8:
        print0('[EET DIAGNOSTIC] ⚠ Frequency signal is constant (freq_table was saved with uniform fallback).')
        print0('[EET DIAGNOSTIC]   Run FrequencyPrior with access to training data shards to rebuild freq_table.pt.')
        return 0.0
    
    rho, p = stats.spearmanr(log_freq, flat_exit)
    print0(f'[EET DIAGNOSTIC] Spearman ρ (freq vs expected exit layer): {rho:.4f} (p={p:.4f})')
    
    if rho < -0.1 and p < 0.05:
        print0('[EET DIAGNOSTIC] ✓ Router has learned frequency-depth structure. Safe to enter Phase 3.')
    else:
        print0('[EET DIAGNOSTIC] ✗ Router has NOT learned structure yet. Extend Phase 2 or check gradients.')
    
    return rho

# -----------------------------------------------------------------------------
# CLI arguments
parser = argparse.ArgumentParser(description="Pretrain base model")
# Logging
parser.add_argument("--run", type=str, default="dummy", help="wandb run name ('dummy' disables wandb logging)")
parser.add_argument("--data-dir", type=str, default=None, help="dataset parquet directory (default: nanochat.dataset.DATA_DIR)")
parser.add_argument("--checkpoints-dir", type=str, default=None, help="base checkpoint root directory (default: <base_dir>/base_checkpoints)")
# Runtime
parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
parser.add_argument("--parallel", type=str, default="ddp", choices=["ddp", "dp"], help="ddp: DistributedDataParallel (via torchrun), dp: nn.DataParallel (for Kaggle/notebooks)")
# FP8 training
parser.add_argument("--fp8", action="store_true", help="enable FP8 training (requires H100+ GPU and torchao)")
parser.add_argument("--fp8-recipe", type=str, default="tensorwise", choices=["rowwise", "tensorwise"], help="FP8 scaling recipe: tensorwise (faster, recommended) or rowwise (more accurate but slower)")
# Model architecture
parser.add_argument("--depth", type=int, default=20, help="depth of the Transformer model")
parser.add_argument("--aspect-ratio", type=int, default=64, help="model_dim = depth * aspect_ratio")
parser.add_argument("--model-dim", type=int, default=0, help="explicit model_dim override (0 = use aspect-ratio)")
parser.add_argument("--head-dim", type=int, default=128, help="target head dimension for attention")
parser.add_argument("--max-seq-len", type=int, default=2048, help="max context length")
parser.add_argument("--window-pattern", type=str, default="SSSL", help="sliding window pattern tiled across layers: L=full, S=half context (e.g. 'SSL')")
# Research branches
parser.add_argument("--use-moe", action="store_true", help="enable research MoE embedding branch")
parser.add_argument("--use-perm", action="store_true", help="use permutation MoE branch (only with --use-moe)")
parser.add_argument("--moe-num-experts", type=int, default=8, help="number of experts for research MoE branch")
parser.add_argument("--moe-router-dim", type=int, default=64, help="router dim for MoE embedding branch")
parser.add_argument("--moe-embed-dim", type=int, default=64, help="embedding dim for MoE branch (must match n_embd)")
parser.add_argument("--use-remix-linear", action="store_true", help="enable remixed linear blocks")
parser.add_argument("--remix-context-dim", type=int, default=64, help="context dim for remixed linear control")
parser.add_argument("--remix-basis-size", type=int, default=64, help="basis size for remixed linear")
parser.add_argument("--use-pos-embed", action="store_true", help="add learned absolute positional embeddings on top of token/research embeddings")
parser.add_argument("--moe-use-abs-pos-embed", type=int, default=0, choices=[0, 1], help="use learned absolute positional embeddings inside permutation MoE embeddings (1/0)")
parser.add_argument("--remix-use-basis-gate", type=int, default=1, choices=[0, 1], help="enable basis gating in remixed linear (1/0)")
parser.add_argument("--remix-use-output-gate", type=int, default=1, choices=[0, 1], help="enable output gating in remixed linear (1/0)")
parser.add_argument("--remix-use-context", type=int, default=1, choices=[0, 1], help="enable context modulation in remixed linear (1/0)")
parser.add_argument("--remix-basis-gate-mode", type=str, default="mlp",
                    choices=["mlp", "linear", "centered", "attn", "random", "none", "lowrank"],
                    help="Gate architecture: mlp=2-layer, linear=single layer (zero-init), centered=1+tanh passthrough init, attn=bilinear, random=linear with random init (no zero-init), none=no basis gate, lowrank=low rank centered output gate symmetry")

parser.add_argument("--p22-n-templates", type=int, default=1, help="22: number of template_mixing matrices (1=standard, K>1=MoE routing)")
parser.add_argument("--p22-template-routing-learned", type=int, default=0, choices=[0, 1], help="22: learned template routing (0=frozen, 1=learned)")
parser.add_argument("--p22-template-topk", type=int, default=0, help="22: hard top-k for legacy template bank (0=soft over all K, N=top-N sparse routing)")
parser.add_argument("--p22-attn-moe-route", type=str, default="none", choices=["none", "sequence", "token"], help="22: MoE routing for attention Q/K/V/Proj ('none'=off, 'sequence'=per-seq, 'token'=per-tok)")
# Phase 23: Tiny-Experts RemixedLinear + Standard MoE baseline
parser.add_argument("--p23-tiny-expert", type=int, default=0, choices=[0, 1], help="23: enable Tiny Experts mode in RemixedLinear (each expert_dim=basis_size//topk for compute parity)")
parser.add_argument("--p23-n-experts", type=int, default=64, help="23: total experts in the Tiny Expert bank (K_total)")
parser.add_argument("--p23-topk", type=int, default=16, help="23: active experts per forward pass (expert_dim=basis_size//topk); 0=soft all-expert routing")
parser.add_argument("--p23-learned-route", type=int, default=0, choices=[0, 1], help="23: learned routing projection in Tiny Expert (0=frozen, 1=learned)")
parser.add_argument("--p23-std-moe-experts", type=int, default=0, help="23: enable StandardMoE_MLP with K full-size experts (0=off)")
parser.add_argument("--p23-std-moe-topk", type=int, default=-1, help="23: top-k active experts for StandardMoE_MLP (-1=optimal sparsity via E^(1-c), 0=all/soft, N=fixed)")
parser.add_argument("--p23-std-moe-aux-weight", type=float, default=0.01, help="23: load-balance auxiliary loss weight for StandardMoE_MLP")
parser.add_argument("--p23-lokr", type=int, default=0, choices=[0, 1], help="23: enable LoKR mode in RemixedLinear")
# Phase 26: Streamlined Full-Rank — OutputGatedLinear (single W + low-rank output gate)
parser.add_argument("--p26-output-gated-linear", type=int, default=0, choices=[0, 1], help="26: replace RemixedLinear with single-W + low-rank output gate (no W_b/W_m factorization)")
# Phase 28: FLOPs-efficient template routing
parser.add_argument("--p28-shared-basis", type=int, default=0, choices=[0, 1], help="28C: share a single W_b projection across all attn Q/K/V/O RL layers per block (saves 3/4 of attn basis FLOPs)")
parser.add_argument("--p28-chunk-routing-size", type=int, default=0, help="28D: amortize template routing over N-token chunks (0=per-token, 64/256=chunk-level)")
parser.add_argument("--p28-global-template-bank", type=str, default="none", choices=["none", "ffn", "all"], help="28E/F: cross-layer global template bank mode ('none'=off, 'ffn'=FFN only, 'all'=FFN+attn)")
parser.add_argument("--p28-attn-proj-templates", type=int, default=0, help="28C2: override n_templates for attn c_proj per-layer (0=use default from remixed_linear_kwargs)")
parser.add_argument("--p28-attn-qk-templates",   type=int, default=0, help="28C3: override n_templates for attn c_q and c_k per-layer (0=use default)")
parser.add_argument("--p31-chunk-route-impl", type=str, default="compose", choices=["compose", "grouped"], help="31: chunk-routing evaluation. 'compose' materializes per-chunk W_eff (legacy); 'grouped' (topk=1 only) permutes chunks by template and runs one GEMM per template — same math, no W_eff in HBM")
parser.add_argument("--p31-top1-gate", type=str, default="ones", choices=["ones", "switch"], help="31: top-1 routing coefficient. 'ones'=legacy (softmax over a single logit is exactly 1.0, so the router gets zero gradient); 'switch'=full-softmax probability of the selected template (differentiable)")
parser.add_argument("--p31-template-delta-rank", type=int, default=0, help="31: replace the (K, out, basis) template bank with a shared base + K rank-r deltas (0=off). Soft mixing over all K becomes three dense GEMMs with no per-chunk W_eff in HBM, so cost is K*r*(basis+out) per token instead of K*out*basis per chunk")
parser.add_argument("--p31-route-side", type=str, default="output", choices=["output", "basis", "narrow"], help="31: which factor carries the routing. 'output'=route W_m (out,basis), legacy; 'basis'=route W_b (basis,in) with a shared W_m; 'narrow'=route whichever is smaller per projection (basis for c_fc, output for c_proj/attn). Shrinks the per-chunk materialized matrix without reducing any rank")
parser.add_argument("--p31-basis-side-templates", type=int, default=0, help="31: template count for projections that route the basis side (0=use --p22-n-templates, -1=auto scale by out//basis for parameter parity with output-side routing). The per-chunk materialized matrix on the basis side is (basis,in) and independent of K, so more templates cost nothing extra in bandwidth")
parser.add_argument("--p31-drop-basis-proj", type=int, default=0, choices=[0, 1], help="31: drop W_b entirely — h=LN(x) then a routed (out,in) template, one matmul per projection. Applied ONLY where in_features <= basis_size (c_fc, attention); skipped on contracting projections like c_proj, where it would grow every template by in/basis and quadruple the per-chunk materialized matrix")
parser.add_argument("--target-active-params",     type=int, default=0, choices=[0, 1], help="use active (topk/chunk-adjusted) params instead of total params when computing target_tokens")
parser.add_argument("--remix-basis-gate-rank", type=int, default=8, help="rank for lowrank basis gate mode (basis_gate_mode=lowrank)")
parser.add_argument("--p23-lokr-rank", type=int, default=4, help="23: low-rank bottleneck for each LoKR expert")
parser.add_argument("--p23-use-shared-block-router", type=int, default=0, choices=[0, 1], help="23: block-level single pass router for all RemixedLinear inner experts")
parser.add_argument("--p23-linear-moe-experts", type=int, default=0, help="23: enable weight-space LinearMoE with K experts (0=off)")
parser.add_argument("--p23-linear-moe-topk", type=int, default=0, help="23: top-k selected experts in LinearMoE (0=soft all-expert blend)")
parser.add_argument("--p23-quantile-route", type=int, default=0, choices=[0, 1, 2], help="23: 1=EMA quantile routing, 2=Causal Expert Cross-Attention")
parser.add_argument("--remix-shared-context-gates", type=int, default=0, choices=[0, 1], help="23: batch all 6 per-RL context gate computations into 3 block-level matmuls (~6x fewer gate kernel launches)")
parser.add_argument("--remix-use-dual-gate", type=int, default=0, choices=[0, 1], help="25: use DualGateLinear instead of RemixedLinear (single dense W + dual D-dim gate, no basis compression)")
parser.add_argument("--remix-use-fused", type=int, default=0, choices=[0, 1], help="enable RemixedLinearFused for 29C chunk routing (default 0=use standard RemixedLinear)")
parser.add_argument("--gate-stats-every", type=int, default=50, help="log gate activation/gradient stats every N steps to gate_stats.log (0=off); only active when --use-remix-linear")
parser.add_argument("--remix-basis-scale-factor", type=int, default=4, help="basis compression ratio: factor=4 → C//4 (default), factor=1 → full rank C. Depth-adaptive via min(in,out)//factor")
parser.add_argument("--remix-output-gate-rank", type=int, default=16, help="rank of the low-rank output gate in RemixedLinear (default 16)")
parser.add_argument("--remix-gate-lr-scale", type=float, default=0.3, help="LR multiplier for gate params relative to structural matrix LR (default 0.3; lower = slower gate learning)")
# Phase 30: LayerNorm ablation flags
parser.add_argument("--remix-disable-ln-basis", type=int, default=0, choices=[0, 1], help="30B: disable intermediate LayerNorm in RemixedLinear basis projection (0=keep LN, 1=remove LN)")
parser.add_argument("--dense-intermediate-ln", type=int, default=0, choices=[0, 1], help="30A: add intermediate LayerNorm to dense linear projections for controlled ablation (0=off, 1=on)")
# Phase 29: RemixedLinear optimizer optimizations
parser.add_argument("--remix-template-block-diag", type=int, default=0, choices=[0, 1], help="29: block-diagonal Muon for template_bank (prevents cross-template gradient contamination)")
parser.add_argument("--remix-template-lr-scale", type=float, default=1.0, help="29: LR multiplier for template_bank Muon group (e.g. 2.0 for 2× LR, MST-inspired)")
parser.add_argument("--p29-grad-equalize", type=int, default=0, choices=[0, 1], help="29: per-template gradient equalization to prevent template collapse (MST-inspired)")
# ── MST: Modular Sub-Transformer ──
# MoL baseline (Ternovtsii & Bilak 2026, arXiv:2605.09516). Their notation is S+KofN.
parser.add_argument("--use-mol", type=int, default=0, choices=[0, 1], help="MoL: enable Mixture-of-Layers baseline")
parser.add_argument("--mol-n-blocks", type=int, default=5, help="MoL: N thin blocks per split stage")
parser.add_argument("--mol-n-shared", type=int, default=1, help="MoL: S always-active blocks (full attention)")
parser.add_argument("--mol-topk", type=int, default=3, help="MoL: k routed blocks selected per token")
parser.add_argument("--mol-thin-dim", type=int, default=256, help="MoL: d_thin")
parser.add_argument("--mol-head-dim", type=int, default=64, help="MoL: head dim, pinned across block widths")
parser.add_argument("--mol-ffn-mult", type=float, default=4.0, help="MoL: d_ff,thin = mult * d_thin")
parser.add_argument("--mol-router-aux", type=float, default=0.05, help="MoL: CV^2 load-balance weight (their alpha)")
parser.add_argument("--mol-routed-attn", type=str, default="softmax", help="MoL: routed-block attention (softmax; deltanet is phase 2)")
parser.add_argument("--mol-dispatch", type=int, default=1, choices=[0, 1], help="MoL: 1=gather/scatter (default), 0=masked reference")
parser.add_argument("--mol-capacity-factor", type=float, default=1.0, help="MoL: per-block capacity when dispatching")
parser.add_argument("--mol-per-block-ve", type=int, default=0, choices=[0,1], help="MoL: each thin block reads its own VE slice (G3 equivalent)")
parser.add_argument("--mol-block-lr-scale", type=float, default=1.0, help="MoL: per-thin-block LR multiplier (fairness ablation; their recipe has none)")
# ── Fully binary transformer ─────────────────────────────────────────────────
parser.add_argument("--use-binary", type=int, default=0, help="swap Linear/Embedding for binary (fully-binary-transformer-plan.md)")
parser.add_argument("--binary-acts", type=int, default=1, help="0 = W1A16 rung: weights binary, activations fp, which changes NO operation")
parser.add_argument("--binary-weight-scale", type=str, default="row", choices=["row", "none", "threshold"], help="row=per-channel multiplicative; none=strict (costs 0.4358 bpb); threshold=per-channel additive + one global gain")
parser.add_argument("--binary-act-scale", type=str, default="token", choices=["token", "none"])
parser.add_argument("--binary-clip", type=float, default=1.0, help="STE clip window; latent weights are initialised inside it or they are born dead")
parser.add_argument("--binary-linear", type=int, default=1)
parser.add_argument("--binary-embeddings", type=int, default=1, help="binarise wte and value_embeds, the interfaces the 1-bit literature leaves in fp")
parser.add_argument("--binary-native", type=int, default=0, help="replace Block with BinaryBlock: Hamming-retrieval attention, binary KV-memory FFN, bundled residual, no normalisation. This is the CONSISTENCY change; --model-dim is the WIDTH change; they are separate claims")
parser.add_argument("--binary-tau", type=float, default=1.0, help="retrieval temperature for the differentiable surrogate; -> 0 recovers hard top-k")
parser.add_argument("--binary-hard", type=int, default=0, help="hard threshold retrieval (the path a popcount kernel runs); no gradient, so inference/eval only")
parser.add_argument("--binary-resid-width", type=int, default=0, help="residual accumulator width. 0 = derive as max(2, n_layer//2). 1 = pure majority bundling, which measures 49.8%% stream survival at depth 8 and destroys the residual path. >1 = bounded accumulator carried across sublayers")
parser.add_argument("--binary-skip", type=str, default="", help="comma-separated name substrings left in fp; this is how the Phase 1 ladder rungs are built")
parser.add_argument("--use-mst", type=int, default=0, choices=[0, 1], help="MST: enable Modular Sub-Transformer mode")
parser.add_argument("--mst-n-subs", type=int, default=8, help="MST: number of sub-transformers N per layer")
parser.add_argument("--mst-sub-dim", type=int, default=64, help="MST: dimension d per sub-transformer")
parser.add_argument("--mst-head-dim", type=int, default=0,
                    help="MST: attention head_dim (0=auto d//n_head; e.g. 64 expands QKV to n_head*64 then projects back to d)")
parser.add_argument("--mst-input-mode", type=str, default="fixed_slice",
                    choices=["fixed_slice", "learned_proj", "rotated_slice", "per_sub_embed", "stem"],
                    help="MST Axis 1: input distribution mode")
parser.add_argument("--mst-rotated-slice-learned", type=int, default=0, choices=[0, 1],
                    help="MST I-C: learn rotation matrix (0=frozen orthogonal, 1=learned)")
parser.add_argument("--mst-routing-mode", type=str, default="soft_weighted",
                    choices=["soft_weighted", "topk_hard", "sequence_path"],
                    help="MST Axis 2: routing/combination mode")
parser.add_argument("--mst-routing-topk", type=int, default=4, help="MST R-B: k for top-k hard routing")
parser.add_argument("--mst-routing-aux-weight", type=float, default=0.01, help="MST: load balance aux loss weight")
parser.add_argument("--mst-diversity-weight", type=float, default=0.0, help="MST: cosine diversity penalty weight (0=off, e.g. 0.01)")
parser.add_argument("--mst-ffn-mode", type=str, default="standard",
                    choices=["standard", "no_downproj", "linear", "shared_dense", "grouped_up_shared_down", "shared_swiglu"],
                    help="MST Axis 3: FFN mode (standard, no_downproj, shared_dense, grouped_up_shared_down, shared_swiglu)")
parser.add_argument("--mst-ffn-every", type=int, default=1, help="MST: apply FFN every N layers (1=every, 2=even-only)")
parser.add_argument("--mst-transition-mode", type=str, default="parallel",
                    choices=["parallel", "aggregate_distribute", "cross_attend", "concat_proj", "free_for_all", "micro_attention", "micro_attention_shared_kv"],
                    help="MST Axis 4: layer-to-layer transition mode")
parser.add_argument("--mst-final-mode", type=str, default="aggregate_proj",
                    choices=["aggregate_proj", "weighted_logits", "concat_proj"],
                    help="MST Axis 5: final layer output mode")
parser.add_argument("--mst-final-topk", type=int, default=-1,
                    help="MST Axis 5: topk for final head (-1=same as routing_topk, 0=all subs)")
parser.add_argument("--mst-ffn-shared-up", type=int, default=0, choices=[0, 1],
                    help="MST: share FFN up-projection across all subs")
parser.add_argument("--mst-ffn-inner-dim", type=int, default=0,
                    help="MST: inner dim for shared FFN (0=4*d)")
parser.add_argument("--mst-sub-dropout", type=float, default=0.0,
                    help="MST: dropout entire sub outputs during training")
parser.add_argument("--mst-transition-every", type=int, default=1,
                    help="MST: apply transition every N layers (1=every, 2=alternate)")
parser.add_argument("--mst-ffa-temperature", type=float, default=1.0,
                    help="MST: temperature for FFA softmax routing")
parser.add_argument("--mst-global-residual", type=int, default=0, choices=[0, 1],
                    help="MST: D-dim global residual stream across layers")
parser.add_argument("--mst-hybrid-dense", type=int, default=0, choices=[0, 1],
                    help="MST: alternate dense (D) and MST (N×d) layers")
parser.add_argument("--mst-cross-sub-kv", type=int, default=0, choices=[0, 1],
                    help="MST: share K,V projections across all subs")
parser.add_argument("--mst-sub-aux-weight", type=float, default=0.0,
                    help="MST: per-sub auxiliary prediction loss weight (H3)")
parser.add_argument("--mst-progressive-merge", type=int, default=0, choices=[0, 1],
                    help="MST: progressive sub-merging pyramid (N1)")
parser.add_argument("--mst-multi-scale-windows", type=int, default=0, choices=[0, 1],
                    help="MST: per-sub window sizes for multi-scale attention (W1)")
parser.add_argument("--mst-delta-residual", type=int, default=0, choices=[0, 1],
                    help="MST: delta residual mode — subs produce corrections to full-D stream (DR1)")
parser.add_argument("--mst-sub-layers", type=int, default=1,
                    help="MST: layers per sub-transformer (SL1, default 1)")
# MST Stage 7: Scaling improvements
parser.add_argument("--mst-grad-equalize", type=int, default=0, choices=[0, 1],
                    help="MST: enable per-sub gradient equalization (1A)")
parser.add_argument("--mst-block-diagonal-muon", type=int, default=0, choices=[0, 1],
                    help="MST: enable block-diagonal Muon Newton-Schulz (1B)")
parser.add_argument("--mst-transition-width-mult", type=float, default=1.0,
                    help="MST: width multiplier for transition bottleneck (1C)")
parser.add_argument("--mst-sub-lr-scale", type=float, default=1.0,
                    help="MST: learning rate multiplier for sub parameters")
parser.add_argument("--mst-shared-expert", type=int, default=0, choices=[0, 1],
                    help="MST: enable DeepSeek-style shared expert (2A)")
parser.add_argument("--mst-router-entropy-weight", type=float, default=0.0,
                    help="MST: weight for router entropy regularization (2C)")
parser.add_argument("--mst-shared-kv-attn", type=int, default=0, choices=[0, 1],
                    help="MST: share K,V projections across sub-transformers (3A)")
parser.add_argument("--mst-contrastive-diversity-weight", type=float, default=0.0,
                    help="MST: weight for contrastive representation diversity loss (3B)")
# MST Stage 8: Transition expressivity
parser.add_argument("--mst-transition-nonlinear", type=int, default=0, choices=[0, 1],
                    help="MST: add SiLU activation at AggDist bottleneck")
parser.add_argument("--mst-transition-gated", type=int, default=0, choices=[0, 1],
                    help="MST: input-dependent routing via concat→gate (replaces mean-based)")
parser.add_argument("--mst-transition-mlp", type=int, default=0, choices=[0, 1],
                    help="MST: replace AggDist with concat→MLP→split (nonlinear cross-sub mixing)")
# MST Stage 9: Cross-sub expressivity
parser.add_argument("--mst-cross-sub-gate", type=int, default=0,
                    help="MST: cross-sub FFN gating rank (0=off, e.g. 32)")
parser.add_argument("--mst-hyper-connect", type=int, default=0, choices=[0, 1],
                    help="MST: hyper-connected sub residuals via EMA lookback (0=off, 1=on)")
parser.add_argument("--mst-cross-kv-inject", type=int, default=0, choices=[0, 1],
                    help="MST: cross-sub KV injection attention (0=off, 1=on)")
# MST Stage 10: Structural transition improvements
parser.add_argument("--mst-slice-transition", type=int, default=0,
                    help="MST: SliceMoE per-slice routing (0=off, S=num slices e.g. 4)")
parser.add_argument("--mst-lookback-layers", type=int, default=0,
                    help="MST: DenseFormer lookback depth (0=off, K=num past layers e.g. 2)")
parser.add_argument("--mst-bilinear-transition", type=int, default=0, choices=[0, 1],
                    help="MST: bilinear aggregate for 2nd-order cross-sub interaction (0=off, 1=on)")
# MST Stage 11: Attention bottleneck + structural improvements
parser.add_argument("--mst-cross-sub-qmod", type=int, default=0,
                    help="MST: low-rank cross-sub query modulation rank (0=off, e.g. 16)")
parser.add_argument("--mst-feature-cycle", type=int, default=0, choices=[0, 1],
                    help="MST: sub-feature cycling at transition (0=off, 1=on)")
parser.add_argument("--mst-mean-transition", type=int, default=0, choices=[0, 1],
                    help="MST: parameter-free mean-add transition replacing AggDist (0=off, 1=on)")
# MST Stage 12: dense-parity fixes (each FLOP-neutral)
parser.add_argument("--mst-sub-head-dim", type=int, default=0,
                    help="MST G1: per-stream attention head_dim; heads = sub_dim//head_dim so "
                         "qkv_dim stays == sub_dim (0=off, legacy 32 at N=4; dense uses 128)")
parser.add_argument("--mst-final-norm", type=int, default=0, choices=[0, 1],
                    help="MST G2: RMSNorm the final projection before lm_head, as dense does (0=off, 1=on)")
parser.add_argument("--mst-ve-map", type=int, default=0, help="MST: G3-cheap — shared d-wide VE table + per-stream dxd map")
parser.add_argument("--mst-ve-map-rank", type=int, default=0, help="MST: 0=full dxd VE map; r>0=identity+rank-r (cheaper in FLOPs)")
parser.add_argument("--mst-per-stream-ve", type=int, default=0, choices=[0, 1],
                    help="MST G3: per-stream value embedding slices from an (N*d)-wide table "
                         "instead of broadcasting one d-wide table (0=off, 1=on)")
# MST Stage 13: overhead cuts (these DO reduce FLOPs, unlike Stage 12)
parser.add_argument("--mst-lm-head-dim", type=int, default=0,
                    help="MST O1: bottleneck width between the final projection and lm_head "
                         "(0=n_embd, i.e. off). Head costs D*Dh + V*Dh instead of D*D + V*D")
parser.add_argument("--mst-compose-windows", type=int, default=0, choices=[0, 1],
                    help="MST O2: intersect per-sub windows with the layer window pattern rather "
                         "than replacing it, so the widest stream is not full-context every layer")
# MST Stage 14: free cross-stream mixing (zero params, zero FLOPs)
parser.add_argument("--mst-channel-mix", type=str, default='none', choices=['none', 'roll', 'shuffle'],
                    help="MST: permute the stream partition so composed block-diagonal layers mix. "
                         "roll=shift by d//2 (Swin-style, keeps specialization); "
                         "shuffle=ShuffleNet transpose (maximal mixing, loses stream identity)")
parser.add_argument("--mst-channel-mix-site", type=str, default='layer', choices=['layer', 'ffn', 'both'],
                    help="MST: where to permute. layer=alternate offset on odd layers; "
                         "ffn=between attention and FFN; both")
# MST Stage 15: coupling optimization + attention cross-stream mixing
parser.add_argument("--mst-distribute-block-muon", type=int, default=0, choices=[0, 1],
                    help="MST F1: give distribute_w block-diagonal Newton-Schulz + sub-LR, like the "
                         "per-stream weights already get (it is (N*d, d) but was omitted)")
parser.add_argument("--mst-trans-spectral-lr", type=int, default=0, choices=[0, 1],
                    help="MST F2: scale agg_up/agg_down LRs by sqrt(N) and 1/sqrt(N) per the "
                         "spectral condition instead of sharing one matrix_lr")
parser.add_argument("--mst-talking-heads", type=int, default=0, choices=[0, 1],
                    help="MST F4: learned (N*n_head)^2 mixing along the head axis before c_proj, "
                         "restoring what dense MHA's W_O does across heads")
parser.add_argument("--mst-wo-mode", type=str, default='block', choices=['block', 'dense'],
                    help="MST F5: attention output projection. block=per-stream (default), "
                         "dense=full D x D over concatenated outputs (+20% layer params)")
# MST Stage 16: conditional stream execution
parser.add_argument("--mst-stream-shared", type=int, default=0, help="MST: S always-active streams; top-k runs over the remaining N-S (MoL S+KofN topology)")
parser.add_argument("--mst-stream-topk", type=int, default=0,
                    help="MST: activate k of N streams per token (0=dense). Gates the FFN, so "
                         "k=2 of 4 is ~19%% of total FLOPs and k=1 ~28%%")
parser.add_argument("--mst-stream-router-aux", type=float, default=0.01,
                    help="MST: Switch-style load-balancing weight for the stream router")
parser.add_argument("--mst-stream-router-noise", type=float, default=0.0,
                    help="MST: noisy top-k exploration std on the router logits (training only). "
                         "Breaks the unselected-stream death spiral")
parser.add_argument("--mst-stream-dispatch", type=int, default=0, choices=[0, 1],
                    help="MST: gather/scatter so the FFN really runs on K<T tokens per stream "
                         "(Phase B). Identical to masking when nothing overflows")
parser.add_argument("--mst-stream-capacity-factor", type=float, default=1.0,
                    help="MST: per-stream capacity K = ceil(T*k/N*factor)")
parser.add_argument("--mst-stream-gate-attn", type=int, default=0, choices=[0, 1],
                    help="MST: also gate attention QKV, not just the FFN (bigger saving, but a "
                         "skipped token stops being a key/value for that stream)")
# MST Stage 17: block-diagonal Shampoo
# MST Stage 18: Monarch-structured FFN
parser.add_argument("--mst-ffn-monarch", type=str, default='none', choices=['none', 'shuffle', 'roll'],
                    help="MST: permute the FFN hidden axis between the two block-diagonal factors, "
                         "making the FFN a genuine Monarch matrix. Free. Incompatible with "
                         "--mst-stream-dispatch (Monarch needs every stream's up-projection)")
parser.add_argument("--mst-shampoo", type=int, default=0, choices=[0, 1],
                    help="MST: block-diagonal Shampoo on the stacked per-stream weights. Exact "
                         "block preconditioning costs D^3/N^2, 16x less than a dense D x D")
parser.add_argument("--mst-precond-every", type=int, default=10,
                    help="MST: steps between Shampoo inverse-fourth-root refreshes")
parser.add_argument("--mst-shampoo-beta", type=float, default=0.95,
                    help="MST: EMA decay for the Shampoo L/R factors")
# ── EET: Early Exit Transformer ──
parser.add_argument("--use-eet", type=int, default=0, choices=[0, 1], help="EET: enable Early Exit Transformer mode")
parser.add_argument("--eet-frozen-kv", type=int, default=1, choices=[0, 1], help="EET: 1=frozen KV injection (Option B), 0=masked attention (Option A)")
parser.add_argument("--eet-router-type", type=str, default="mlp2", choices=["linear", "mlp1", "mlp2", "attention", "attn"], help="EET: exit router architecture")
parser.add_argument("--eet-router-hidden", type=int, default=0, help="EET: router MLP hidden dim (0=n_embd//4)")
parser.add_argument("--eet-freq-prior-alpha", type=float, default=0.0, help="EET: frequency prior weight (0=disabled)")
parser.add_argument("--eet-pos-prior-beta", type=float, default=0.0, help="EET: POS prior weight (0=disabled)")
parser.add_argument("--eet-domain-prior", type=int, default=0, choices=[0, 1], help="EET: enable domain-conditioned routing")
parser.add_argument("--eet-warmup-frac", type=float, default=0.02, help="EET: Phase 1 dense warmup fraction")
parser.add_argument("--eet-explore-frac", type=float, default=0.15, help="EET: Phase 2 exploration fraction")
parser.add_argument("--eet-reconstruct-lambda", type=float, default=1.0, help="EET: reconstruction loss weight (λ_r)")
parser.add_argument("--eet-efficiency-lambda-start", type=float, default=0.01, help="EET: initial efficiency loss weight")
parser.add_argument("--eet-efficiency-lambda-end", type=float, default=0.1, help="EET: final efficiency loss weight")
parser.add_argument("--eet-translator-rank", type=int, default=0, help="EET: TunedLens translator rank (0=full)")
parser.add_argument("--eet-max-frozen-kv-frac", type=float, default=0.75, help="EET: max fraction of tokens that can exit")
parser.add_argument("--eet-exit-threshold", type=float, default=0.5, help="EET: sigmoid threshold for exit decision")
parser.add_argument("--eet-min-exit-layer", type=int, default=1, help="EET: earliest layer a token can exit at")
parser.add_argument("--eet-loss-variant", type=str, default="reconstruct", choices=["none", "reconstruct", "entropy_surprise", "adversarial", "quality", "layer_weighted", "ce_guided"], help="EET: loss variant for early exit training. 'none' runs routing with no auxiliary loss at all: the router then gets no gradient and is a frozen projection, which is the right control given random routing was measured to tie the learned router")
parser.add_argument("--eet-topk-vocab", type=int, default=512, help="EET: top-k vocabulary size for cheap entropy calculation")
parser.add_argument("--eet-entropy-lambda", type=float, default=0.3, help="EET: entropy pressure weight (λ_ent) for entropy_surprise loss")
parser.add_argument("--eet-surprise-lambda", type=float, default=0.1, help="EET: surprise pressure weight (λ_sur) for entropy_surprise loss")
parser.add_argument("--eet-adv-lambda", type=float, default=1.0, help="EET: adversarial loss weight (λ_adv) for adversarial loss")
parser.add_argument("--eet-adv-entropy-lambda", type=float, default=0.2, help="EET: adversarial entropy pressure weight (λ_adv_ent) for adversarial loss")
parser.add_argument("--eet-quality-lambda", type=float, default=1.0, help="EET: REINFORCE quality loss weight for quality variant")
parser.add_argument("--eet-quality-entropy-bonus", type=float, default=0.1, help="EET: entropy bonus coefficient to prevent exit distribution collapse")
parser.add_argument("--eet-gumbel-temp-start", type=float, default=0.0, help="EET: Gumbel-Softmax starting temperature (0=disabled)")
parser.add_argument("--eet-gumbel-temp-end", type=float, default=0.1, help="EET: Gumbel-Softmax ending temperature")
parser.add_argument("--eet-gumbel-hard", type=int, default=1, choices=[0, 1], help="EET: enable Straight-Through Estimator for Gumbel")
parser.add_argument("--eet-commitment-beta", type=float, default=0.1, help="EET: commitment loss weight beta (0=disabled)")
parser.add_argument("--eet-global-router", type=int, default=0, choices=[0, 1], help="EET: use an upfront single global exit router predicting exit layer distribution")
parser.add_argument("--eet-freq-efficiency-alpha", type=float, default=0.0, help="EET: per-token frequency-scaled efficiency loss (0=uniform, >0=frequent tokens penalized more for late exits)")
parser.add_argument("--eet-diversity-lambda", type=float, default=0.0, help="EET: exit diversity pressure - penalizes uniform exit depth across tokens (0=disabled)")
parser.add_argument("--eet-ce-guided-lambda", type=float, default=1.0, help="EET: CE-guided routing loss weight (loss_variant='ce_guided')")
parser.add_argument("--eet-router-lr-mult", type=float, default=5.0, help="EET: Router LR multiplier relative to gate_lr (default 5.0). Higher = faster router learning.")
parser.add_argument("--eet-model-lr-mult", type=float, default=1.0, help="EET: Backbone LR multiplier (default 1.0). Scales all non-router LRs when EET is active.")
parser.add_argument("--eet-depth-weight-type", type=str, default="none", choices=["none", "linear", "ema", "sqrt"], help="EET: Token-wise CE loss weighting by exit depth")
parser.add_argument("--eet-depth-weight-max", type=float, default=2.5, help="EET: Maximum weighting factor for deep tokens in linear strategy")
parser.add_argument("--eet-use-override", type=int, default=0, choices=[0, 1], help="EET: 1 = enable stochastic depth override, 0 = disabled")
parser.add_argument("--eet-override-prob-start", type=float, default=0.5, help="EET: initial override probability during training")
parser.add_argument("--eet-override-prob-end", type=float, default=0.1, help="EET: minimum/terminal override probability during training")
parser.add_argument("--eet-reenter-final", type=int, default=0, choices=[0, 1], help="EET: force exited tokens to re-enter and be processed by the final layer (1/0)")
parser.add_argument("--eet-compute-skip", type=int, default=0, choices=[0, 1], help="EET: enable compute-level skipping of intermediate blocks (1/0)")
parser.add_argument("--eet-target-active-frac", type=float, default=0.125, help="EET: target active token fraction at the deepest routable layer")
parser.add_argument("--eet-capacity-schedule", type=str, default="bell", choices=["uniform", "linear", "geometric", "bell"], help="EET: capacity schedule for compute skip")
parser.add_argument("--eet-exit-fracs", type=str, default="", help="EET: comma-separated list of float exit fractions (sums to ~1.0) overriding schedule")
parser.add_argument("--eet-capacity-alignment-lambda", type=float, default=0.0, help="EET: weight for load-balancing/capacity alignment loss (0=disabled)")
parser.add_argument("--eet-kv-mode", type=str, default="none", choices=["none", "fresh", "stale"], help="EET P02 T1: context for exited tokens. none=active keys only (current), fresh=re-project keys/values for every position at each layer (quality upper bound), stale=reuse the keys/values banked at each token's exit layer (near free)")
parser.add_argument("--eet-route-noise", type=float, default=0.0, help="EET P02 T2: Gumbel noise scale added to the exit score before top-K, training only. 0=deterministic, 0.3=mild, 1.0=strong, >=10=uniform random")
parser.add_argument("--eet-route-noise-end", type=float, default=-1.0, help="EET P02 T2: final noise scale for a linear anneal over training (<0 = hold --eet-route-noise constant)")
parser.add_argument("--eet-coverage-diag", type=int, default=0, choices=[0, 1], help="EET P02 T0B: accumulate per-layer vocabulary coverage during eval and report it")
parser.add_argument("--eet-kv-eager", type=int, default=0, choices=[0, 1], help="EET P02: run the split-KV attention outside torch.compile. Slower, but sidesteps inductor stride-guard failures on the attention mask so the quality gate still produces a bpb. Never use for a wallclock claim")
parser.add_argument("--deep-supervision-lambda", type=float, default=0.0, help="Tier 0: weight on a CE term at every layer through the shared head. Prices what it costs a healthy backbone to be readable at every depth")
parser.add_argument("--deep-supervision-frac", type=float, default=0.125, help="fraction of positions supervised per layer; keeps the cost near one extra head pass")
parser.add_argument("--eet-width-power", type=float, default=0.0, help="Tier 2 idea 5: FFN multiplier per layer = base*(1/active_frac)^power. 0 = uniform stack, 1 = every layer costs the same FLOPs however few tokens reach it. Widens where tokens are scarce, the opposite of a funnel transformer")
parser.add_argument("--eet-width-cap", type=float, default=16.0, help="ceiling on that multiplier")
parser.add_argument("--eet-router-task-grad", type=int, default=1, choices=[0, 1], help="EET: allow task loss gradients to propagate to router through continue weights (1/0)")
parser.add_argument("--eet-reinforce-interval", type=int, default=0, help="EET: two-pass REINFORCE every N steps (0=disabled)")
parser.add_argument("--eet-reinforce-lambda", type=float, default=0.1, help="EET: REINFORCE loss weight")
parser.add_argument("--eet-exit-adapter-rank", type=int, default=0, help="EET: per-exit low-rank adapter rank (0=disabled)")
parser.add_argument("--eet-router-after-block", type=int, default=0, help="EET: run global router after this block index (0=use raw embedding)")
parser.add_argument("--eet-ffn-skip", type=int, default=0, choices=[0, 1], help="EET A³D: skip FFN only, preserve attention at all layers")
parser.add_argument("--eet-ffn-target-frac", type=float, default=0.50, help="EET A³D: fraction of tokens that get FFN at each layer")
parser.add_argument("--eet-ffn-full-attn", type=int, default=1, choices=[0, 1], help="EET A³D: 1=attention on all T tokens, 0=gather for attention too")
parser.add_argument("--eet-depth-affine", type=int, default=0, choices=[0, 1], help="EET: apply learned (γ, β) depth-conditional affine alignment before LM head")
parser.add_argument("--eet-capacity-anneal-frac", type=float, default=0.0, help="EET: fraction of training steps to anneal target active frac from 0.5 to configured value")
parser.add_argument("--eet-learned-schedule", type=int, default=0, choices=[0, 1], help="EET: learn per-exit scheduling prior logits instead of fixed schedule")
parser.add_argument("--eet-departure-summary", type=int, default=0, choices=[0, 1], help="EET: inject mean state of exiting tokens into continuing active tokens")
parser.add_argument("--eet-route-consistency-lambda", type=float, default=0.0, help="EET: weight for EMA-based routing consistency loss")
parser.add_argument("--eet-dense-distill-interval", type=int, default=0, help="EET: concurrent dense distillation step interval (0=disabled)")
parser.add_argument("--eet-dense-distill-lambda", type=float, default=0.5, help="EET: concurrent dense distillation KL loss weight")
parser.add_argument("--eet-depth-lr-scale", type=int, default=0, choices=[0, 1], help="EET: per-layer LR scaling by inverse surviving fraction (Option A)")
parser.add_argument("--eet-depth-grad-scale", type=int, default=0, choices=[0, 1], help="EET: scale per-token CE by inverse active fraction at exit depth (Option B)")
parser.add_argument("--eet-detach-aux-from-backbone", type=int, default=0, choices=[0, 1], help="EET: detach aux losses (CE-guided, surprise) from backbone gradients")
parser.add_argument("--eet-detach-exit-from-backbone", type=int, default=0, choices=[0, 1], help="EET: detach exiting token representations from backbone — backbone only trains from final-layer tokens")
parser.add_argument("--p24-use-sliced-weight", type=int, default=0, choices=[0, 1], help="24: enable SlicedWeightLinear (LinearMoE2-style)")
parser.add_argument("--p24-sliced-weight-reduction-scale", type=int, default=8, help="24: big_dim = in_features * reduction_scale")
parser.add_argument("--p24-sliced-weight-min-select", type=int, default=128, help="24: minimum selected columns from weight bank")
parser.add_argument("--p24-sliced-weight-scope", type=str, default="per_token", choices=["per_token", "per_block", "global"], help="24: routing scope for sliced weight")
parser.add_argument("--p24-sliced-weight-balance-coeff", type=float, default=0.01, help="24: aux loss coeff for sliced router usage balance")
parser.add_argument("--p24-quantile-route", type=int, default=0, choices=[0, 1, 2], help="24: reserved routing mode selector")
parser.add_argument("--p24-use-folded-mod", type=int, default=0, choices=[0, 1], help="24: enable FoldedModulationLinear (LinearMoE3-style)")
parser.add_argument("--p24-folded-mod-reduction-scale", type=int, default=8, help="24: fold R consecutive dims by summation")
parser.add_argument("--p24-folded-mod-min-dim", type=int, default=128, help="24: floor on folded_dim")
parser.add_argument("--p24-folded-mod-scope", type=str, default="per_layer", choices=["per_layer", "per_block", "global"], help="24: gate sharing scope for folded modulation")
parser.add_argument("--p24-folded-mod-gate-act", type=str, default="tanh_centered", choices=["sigmoid", "tanh_centered"], help="24: gate activation for folded modulation")
parser.add_argument("--p24-use-sequence-gated-linear", type=int, default=0, choices=[0, 1], help="24: enable SequenceGatedLinear (dense + sequence gate)")
parser.add_argument("--p24-sequence-gated-scope", type=str, default="per_layer", choices=["per_layer", "per_block", "global"], help="24: gate sharing scope for dense sequence gating")
parser.add_argument("--p24-sequence-gated-act", type=str, default="tanh_centered", choices=["sigmoid", "tanh_centered"], help="24: gate activation for dense sequence gating")
# CCL block modulation (only active when --use-remix-linear is set)

parser.add_argument("--cclblock-modulation", type=str, default="weight",
                    choices=["weight", "normalization", "householder", "spectral", "ocd", "lie", "polynomial", "grassmann", "decoupled", "tucker", "svs", "vq", "dcu", "fsi", "aesp", "ckr", "ckr_ffn", "com", "giad", "psg", "splitstream", "lokr", "pgr", "cil", "prb", "arg", "kfl", "cond"],
                    help="CCL block strategy")
# Phase 35: ConditionedLinear (--cclblock-modulation cond)
parser.add_argument("--p34-ffn-schedule", type=str, default="",
                    help="34: per-layer FFN width. Comma list of multipliers (0 = no FFN in that block), or a preset: 'measured' (reallocate at fixed params by measured reachable demand), 'shrink' (cut only over-provisioned layers), 'last', 'every2', 'every4', 'every2_iso', 'every4_iso'")
parser.add_argument("--p34-ffn-no-ffn-replacement", type=str, default="none",
                    choices=["none", "linear"],
                    help="34: what to put in layers where the FFN schedule is 0. 'none' = attention only, 'linear' = D->D linear projection (channel recombination without nonlinearity)")
parser.add_argument("--p34-ffn-last-depth", type=int, default=1, choices=[1, 2, 3],
                    help="34: FFN depth for the last layer. 1=standard D->4D->D, 2=deep D->4D->4D->D, 3=bottleneck D->D->4D->D")
parser.add_argument("--p36-swiglu-ffn", type=int, default=0, choices=[0, 1],
                    help="36: dense baseline with a SwiGLU FFN instead of ReLU^2, at parameter parity. The control for the p35 cond-tied result, which reduces exactly to y = W0 x + U*SiLU(2 V^T x)")
parser.add_argument("--p36-swiglu-mult", type=float, default=8.0/3.0,
                    help="36: SwiGLU hidden width as a multiple of n_embd (8/3 = parameter parity with a 4x ReLU^2 FFN)")
parser.add_argument("--cond-sites", type=str, default="both", choices=["both", "attn", "ffn"],
                    help="35: where ConditionedLinear goes. 'ffn' and 'attn' localize which half of the model the gain comes from")
parser.add_argument("--p22-route-affine", type=int, default=0, choices=[0, 1],
                    help="22: affine-hull routing instead of softmax. alpha = 1/K + s*(beta - mean beta): still sums to 1, may go negative, no simplex. Removes the alpha_k prefactor that starves losing templates and the norm/concentration coupling that drives collapse")
parser.add_argument("--cond-rank", type=int, default=256,
                    help="35: R, rank of the additive conditioned branch y = W0 x + U(c(x) * V^T x). 0 disables it")
parser.add_argument("--cond-mult-steps", type=int, default=0,
                    help="35: m, sequential rank-1 composition steps z_i = z_{i-1} + c_i(x) u_i (v_i^T z_{i-1}). 0 disables it")
parser.add_argument("--cond-gate-source", type=str, default="router", choices=["router", "tied", "ctx"],
                    help="35: coefficient source. 'router' (own projection of x), 'tied' (reuse V^T x, zero extra params, the control for 'is this just a GLU'), 'ctx' (block context stream; ignores --cond-chunk-size on the attention path where ctx is None)")
parser.add_argument("--cond-coeff-act", type=str, default="centered", choices=["centered", "linear", "sigmoid", "one"],
                    help="35: coefficient activation, all equal to 1 at zero logits: 'centered' 1+tanh, 'linear' 1+z (unbounded, can flip sign), 'sigmoid' 2*sigmoid, 'one' c=1")
parser.add_argument("--cond-router-rank", type=int, default=0,
                    help="35: factor the router as d_sig -> rr -> R (0 = full-rank router). The router is ~1/3 of the added params at R=256")
parser.add_argument("--cond-router-act", type=str, default="none", choices=["none", "gelu", "relu2"],
                    help="35: nonlinearity in the router's hidden layer (needs --cond-router-rank > 0). Same params and same FLOPs as the linear rank-rr router. conditioning_headroom.py --nonlinear-reach on dense d22 measured +0.107 held-out reach for this at r=64, +0.12 to +0.15 on every attention projection")
parser.add_argument("--cond-live-init", type=float, default=0.0,
                    help="35: >0 draws every conditioning factor nonzero and subtracts the branch mean from W0, so the composite starts at the dense init with no dead branch and no multi-step gradient unlock. The value scales the branch")
parser.add_argument("--cond-chunk-size", type=int, default=0,
                    help="35: route once per chunk from its first token (0 = per token). Ignored when --cond-gate-source tied")
parser.add_argument("--cond-mult-scale", type=float, default=-1.0,
                    help="35: bound on |c| for the composition branch. -1 uses 1/m, which makes every factor I + c u v^T have spectral norm <= 1+1/m and the whole product <= e for any m. Raising it trades the stability guarantee for range")
parser.add_argument("--cond-mult-impl", type=str, default="wy", choices=["wy", "loop"],
                    help="35: multiplicative implementation. 'wy' compact (one pass over activations), 'loop' sequential reference")
parser.add_argument("--cond-attn-projs", type=str, default="qkvo",
                    help="35: which attention projections get ConditionedLinear. 'qkvo' = all (default), 'qko' = skip c_v, 'qo' = Q and output only, 'o' = output only. Headroom data shows c_v DOF is ~7-12 across all depths, so 'qko' saves 25%% cost on a projection whose ceiling is near zero")
parser.add_argument("--cond-layer-frac", type=float, default=1.0,
                    help="35: fraction of layers (from first) that get ConditionedLinear on attention. 0.5 = first half only. Headroom data shows early layers have lowest FFN excess (most unmet demand), so concentrating nonlinearity there may help")
parser.add_argument("--cclblock-orth-lambda", type=float, default=0.0,
                    help="OCD overlap penalty weight (0 disables)")
parser.add_argument("--p35-template-diversity-lambda", type=float, default=0.0,
                    help="Template bank diversity loss — penalizes pairwise cosine similarity between templates (0 disables)")
parser.add_argument("--cclblock-context-stream", type=str, default="local", 
                    choices=["local", "shifted", "ema", "selective", "multiscale", "ssm", "boundary", "chunk", "predictive_chunk", "evidence_ssm", "dacs", "prefix", "warmup_ema", "dacs_ema", "decay_prefix"],
                    help="Context stream: 'local', 'shifted', 'ema', 'selective', 'multiscale', 'ssm', 'boundary', 'chunk', 'predictive_chunk', 'evidence_ssm', 'dacs', 'prefix', 'warmup_ema', 'dacs_ema', 'decay_prefix'")
parser.add_argument("--cclblock-ema-factor", type=float, default=0.99,
                    help="EMA factor for the legacy EMAContextStream")
parser.add_argument("--cclblock-stale-ctx-lag", type=int, default=0,
                    help="Design C stale context lag (0=disabled)")
# Novel ablation designs
parser.add_argument("--cclblock-sparse-gate-k", type=int, default=0,
                    help="Design 3: sparse top-k basis gate (0=soft sigmoid, N=activate top-N basis functions)")
parser.add_argument("--cclblock-gate-temperature", type=float, default=1.0,
                    help="Design 6: basis gate temperature (<1=sharper, >1=softer, 1.0=standard sigmoid)")
parser.add_argument("--cclblock-context-bank-size", type=int, default=0,
                    help="Design 4: context prototype bank size (0=disabled, e.g. 16=16 learned prototypes)")
parser.add_argument("--cclblock-per-head-ctx", type=int, default=0, choices=[0, 1],
                    help="Design 7: separate ctx projections for attn vs ffn (0=off, 1=on)")
parser.add_argument("--cclblock-context-source", type=str, default="norm_x",
                    choices=["norm_x", "attn_heads", "attn_geometry"],
                    help="Context source for FFN gate/router ('norm_x', 'attn_heads', 'attn_geometry')")
# Phase 8: Boundary-Gated / Chunk Context / Auxiliary Objective
parser.add_argument("--cclblock-chunk-size", type=int, default=0,
                    help="Design 9: hard chunk pooling stride in tokens (0=off, e.g. 64)")
parser.add_argument("--cclblock-aux-objective", type=str, default="none",
                    choices=["none", "boundary", "entropy"],
                    help="Design 10: aux context objective ('none'=off, 'boundary'=boundary BCE, 'entropy'=difficulty MSE)")
parser.add_argument("--cclblock-aux-lambda", type=float, default=0.1,
                    help="Design 10: weight of auxiliary context loss (default 0.1)")
parser.add_argument("--cclblock-boundary-token-id", type=int, default=198,
                    help="Design 10: token ID for boundary detection (default 198=newline in many tokenizers)")
# Phase 9: RAL & FiLM
parser.add_argument("--use-ral", type=int, default=0, choices=[0, 1], help="Proposal A: Use ResidualAdaptiveLinear instead of RemixedLinear")
parser.add_argument("--ral-rank", type=int, default=32, help="Proposal A: Rank for the RAL context delta")
parser.add_argument("--cclblock-film-gate", type=int, default=0, choices=[0, 1], help="Proposal C: Use FiLM affine basis gate in RemixedLinear")
parser.add_argument("--cclblock-attn-shadow-dim", type=int, default=0, help="Dual-V shadow routing width (0=off)")
parser.add_argument("--cclblock-dynamic-ratio", type=float, default=0.25, help="Paradigm 1 (decoupled): fraction of channels routed to dynamic path")
parser.add_argument("--cclblock-gate-rank", type=int, default=8, help="Paradigm 1 (decoupled): low-rank context gate rank")
parser.add_argument("--cclblock-num-regimes", type=int, default=8, help="Paradigm 2 (evidence_ssm): number of latent regimes K")
parser.add_argument("--cclblock-regime-temperature", type=float, default=1.0, help="Paradigm 2 (evidence_ssm): softmax temperature over regimes")
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
# Phase 12: FSI / AESP / CKR
parser.add_argument("--cclblock-fsi-rotations", type=int, default=8, help="FSI: number of frozen orthogonal rotations")
parser.add_argument("--cclblock-fsi-selector-dim", type=int, default=64, help="FSI: frozen routing projection dim")
parser.add_argument("--cclblock-aesp-strata", type=int, default=4, help="AESP: number of entropy strata")
parser.add_argument("--cclblock-aesp-delta-rank", type=int, default=4, help="AESP: rank of per-stratum low-rank deltas")
parser.add_argument("--cclblock-ckr-branches", type=int, default=4, help="CKR: number of parallel dense branches")
parser.add_argument("--cclblock-ckr-kernel-size", type=int, default=64, help="CKR: causal conv1d kernel size")
# Phase 13: CKR enhancements
parser.add_argument("--cclblock-ckr-pos-channels", type=int, default=1, help="CKR: multi-channel position signal (1=original, 3=multi-scale)")
parser.add_argument("--cclblock-ckr-dual-optim", type=int, default=0, choices=[0, 1], help="CKR: route gate params to dedicated conservative AdamW")
parser.add_argument("--cclblock-ckr-content-bias", type=float, default=0.0, help="CKR: frozen content hash bias scale (0=pure position)")
# Phase 14: Gradient-isolated content conditioning
parser.add_argument("--cclblock-giad-rank", type=int, default=32, help="GIAD: low-rank bottleneck dimension")
parser.add_argument("--cclblock-psg-kernel-size", type=int, default=64, help="PSG: causal conv kernel size")
parser.add_argument("--cclblock-ss-dynamic-ratio", type=float, default=0.25, help="SplitStream: dynamic channel fraction")
parser.add_argument("--cclblock-ss-branches", type=int, default=2, help="SplitStream: CKR branches on dynamic path")
parser.add_argument("--cclblock-ss-kernel-size", type=int, default=64, help="SplitStream: causal conv kernel size")
# Phase 15: LoKR + diagnostics
parser.add_argument("--cclblock-lokr-branches", type=int, default=8, help="LoKR: number of low-rank perturbation branches")
parser.add_argument("--cclblock-lokr-rank", type=int, default=16, help="LoKR: rank of each perturbation")
# Phase 16: CKR-Anneal / COM
parser.add_argument("--cclblock-ckr-temp-start", type=float, default=2.0, help="CKR-Anneal: initial softmax temperature")
parser.add_argument("--cclblock-ckr-temp-end", type=float, default=0.3, help="CKR-Anneal: final softmax temperature")
parser.add_argument("--cclblock-com-kernel-size", type=int, default=32, help="COM: causal output mixer kernel size")
# Phase 17: CKR enhancements + new architectures
parser.add_argument("--cclblock-ckr-ortho-init", type=int, default=0, choices=[0, 1], help="17D: orthogonal branch init")
parser.add_argument("--cclblock-ckr-branch-dropout", type=float, default=0.0, help="17E: branch dropout probability")
parser.add_argument("--cclblock-ckr-diversity-lambda", type=float, default=0.0, help="17H: branch diversity loss weight")
parser.add_argument("--cclblock-pgr-kernel-size", type=int, default=64, help="17C: PGR causal conv kernel size")
parser.add_argument("--cclblock-cil-kernel-size", type=int, default=64, help="17I: CIL causal conv kernel size")
parser.add_argument("--cclblock-prb-kernel-size", type=int, default=64, help="17J: PRB causal conv kernel size")
parser.add_argument("--modulation-diagnostics", type=int, default=0, choices=[0, 1], help="enable modulation layer diagnostics logging")
# Phase 18: Beyond CKR — orthogonal improvements
parser.add_argument("--p18-layer-drop", type=float, default=0.0, help="18E: stochastic depth drop probability (0=off)")
parser.add_argument("--p18-dynamic-activation", type=int, default=0, choices=[0, 1], help="18I: learned activation mix (ReLU²+GELU+SiLU)")
parser.add_argument("--p18-mixture-norm", type=int, default=0, choices=[0, 1], help="18H: learned RMSNorm+LayerNorm mixture")
parser.add_argument("--p18-aux-sim-lambda", type=float, default=0.0, help="18G: layer similarity penalty weight (0=off)")
parser.add_argument("--p18-gradient-penalty", type=float, default=0.0, help="18B: gradient penalty weight for Lipschitz regularization (0=off)")
parser.add_argument("--p18-per-channel-scale", type=int, default=0, choices=[0, 1], help="18F: learnable per-channel output scale")
# Phase 19: Zero-overhead indirect modulation
parser.add_argument("--p19-residual-gate", type=int, default=0, choices=[0, 1], help="19A: per-layer learned scalar on block output (0/1)")
parser.add_argument("--p19-head-importance", type=int, default=0, choices=[0, 1], help="19B: per-head learned scalar on attn output (0/1)")
parser.add_argument("--p19-residual-mix-groups", type=int, default=0, help="19C: grouped 1x1 conv between blocks (0=off, N=group_size)")
parser.add_argument("--p19-attn-logit-bias", type=int, default=0, choices=[0, 1], help="19D: per-head learned QK temperature (0/1)")
parser.add_argument("--p19-residual-decay", type=int, default=0, choices=[0, 1], help="19E: learned depth-dependent x0 decay (0/1)")
parser.add_argument("--p19-grad-equilibrium", type=float, default=0.0, help="19F: gradient equilibrium regularization lambda (0=off)")
parser.add_argument("--p19-spectral-reparam", type=int, default=0, choices=[0, 1, 2], help="19G: spectral reparameterization (0=off, 1=c_proj, 2=c_fc+c_proj)")
parser.add_argument("--p19-weight-anticollapse", type=float, default=0.0, help="19H: weight anti-collapse penalty lambda (0=off)")
parser.add_argument("--p19-ve-bias", type=int, default=0, choices=[0, 1], help="19I: add learnable bias to VE gate (0/1)")
parser.add_argument("--p19-weight-noise", type=float, default=0.0, help="19J: training-time weight perturbation epsilon (0=off)")
# Phase 20: Context-conditioned dynamic weight computation
parser.add_argument("--p20-hrcs-scale", type=int, default=0, help="20A: Hash-routed column selection (0=off, scale=D_stored/D_active)")
parser.add_argument("--p20-lswr-scale", type=int, default=0, help="20B: LSH weight routing (0=off, scale factor)")
parser.add_argument("--p20-lswr-planes", type=int, default=8, help="20B: number of LSH hash planes")
parser.add_argument("--p20-lrcfb-branches", type=int, default=0, help="20C: Content-routed branches (0=off, K=branches)")
parser.add_argument("--p20-lrcfb-narrow", type=int, default=0, choices=[0, 1], help="20C: narrow branches (0=full-size, 1=H//K param parity)")
parser.add_argument("--p20-lrcfb-learned", type=int, default=0, choices=[0, 1], help="20C: learned routing (0=frozen, 1=learnable)")
parser.add_argument("--p20-lrcfb-topk", type=int, default=0, help="20C: top-k sparse routing (0=soft/all)")
parser.add_argument("--p20-dgcr-branches", type=int, default=0, help="20D: Detached-gradient content-routed branches (0=off, K=branches)")
parser.add_argument("--p20-dgcr-aux-weight", type=float, default=0.01, help="20D: auxiliary routing loss weight")
parser.add_argument("--p20-mone-experts", type=int, default=0, help="20F: Mixture of Narrow Experts (0=off, K=num experts)")
parser.add_argument("--p20-mone-topk", type=int, default=0, help="20F: top-k expert routing (0=compute all, K=top-k sparse)")
parser.add_argument("--p20-mone-narrow", type=int, default=1, choices=[0, 1], help="20F: narrow experts (1=4D/K, 0=full 4D each)")
parser.add_argument("--p20-mone-frozen", type=int, default=0, choices=[0, 1], help="20F: frozen routing (0=learned, 1=frozen random proj)")
parser.add_argument("--p20-ncea-branches", type=int, default=0, help="20H: Noise-contrastive expert assignment (0=off, K=branches)")
parser.add_argument("--p20-ncea-eps", type=float, default=0.1, help="20H: perturbation magnitude")
parser.add_argument("--p20-adwi", type=int, default=0, choices=[0, 1], help="20I: Attention-derived weight interpolation (0=off, 1=on)")
# Phase 20 — Phase 2 proposals (require pre-trained checkpoint)
parser.add_argument("--p20-pwu-branches", type=int, default=0, help="20E: Progressive weight unfreezing (0=off, K=branches)")
parser.add_argument("--p20-pwu-phase", type=int, default=1, choices=[1, 2, 3], help="20E: training phase (1=pretrain, 2=router only, 3=joint)")
parser.add_argument("--p20-fsvd-gate", type=int, default=0, choices=[0, 1], help="20G: Frozen-SVD σ gating (0=off, 1=on)")
parser.add_argument("--p20-wbfc-clusters", type=int, default=0, help="20J: Weight bank frozen clustering (0=off, K=clusters)")
parser.add_argument("--p20-wbfc-active", type=int, default=0, help="20J: active clusters per token (0=auto K//4)")
# Phase 21: Pervasive Expert Routing
parser.add_argument("--p21-per-experts", type=int, default=0, help="21: MoELinear experts per layer (0=off, K=experts)")
parser.add_argument("--p21-per-topk", type=int, default=0, help="21: top-k routing (0=soft/all)")
parser.add_argument("--p21-per-learned", type=int, default=0, choices=[0, 1], help="21: learned routing (0=frozen, 1=learnable)")
parser.add_argument("--p21-per-attn", type=int, default=0, choices=[0, 1], help="21: also replace attention Q/K/V/O (0=MLP only, 1=all)")
# Fix 1A: per-layer context updaters
parser.add_argument("--use-layer-context", type=int, default=1, choices=[0, 1], help="per-layer context deltas for remix_linear: 1=enable (Fix 1A), 0=static base context")
parser.add_argument("--router-context-window", type=int, default=-1, help="sliding window size for GlobalContextManager (-1 for full)")
# Fix 1B: basis scaling
parser.add_argument("--scale-basis-size", type=int, default=1, choices=[0, 1], help="auto-scale RemixedLinear basis_size to max(basis_size, in_features//4) (Fix 1B)")
# Fix 1D: PermutationMoE expert mode
parser.add_argument("--perm-expert-mode", type=str, default="low_rank", choices=["full", "low_rank", "factored"], help="PermutationMoE expert mode: 'full' (original D×D), 'low_rank', or 'factored' (Fix 1D)")
parser.add_argument("--perm-rank", type=int, default=16, help="rank divisor for 'low_rank' mode or block size for 'factored' mode (Fix 1D)")
# Fix 4C: gradient clipping
parser.add_argument("--max-grad-norm", type=float, default=1.0, help="gradient norm clip threshold (-1 to disable, Fix 4C)")
# Fix 1H: PermutationMoE temperature scheduling
parser.add_argument("--perm-temp-start", type=float, default=5.0, help="initial PermutationMoE temperature (decays to 1.0 over first 50%% of training, Fix 1H)")
parser.add_argument("--research-onecycle", type=int, default=1, choices=[0, 1], help="for research runs: 1=use OneCycle LR schedule, 0=fallback to base warmup/flat/warmdown")
parser.add_argument("--use-onecycle", type=int, default=None, choices=[0, 1], help="alias for --research-onecycle")
parser.add_argument("--research-warmup-ratio", type=float, default=0.05, help="research-only warmup ratio/pct_start for OneCycle")
# Training horizon (only one used, in order of precedence)
parser.add_argument("--num-iterations", type=int, default=-1, help="explicit number of optimization steps (-1 = disable)")
parser.add_argument("--target-tokens", type=int, default=-1, help="explicit number of tokens to train for (-1 = disable)")
parser.add_argument("--target-flops", type=float, default=-1.0, help="calculate num_iterations to reach target_flops, using TOTAL FLOPs/token (-1 = disable)")
parser.add_argument("--target-active-flops", type=float, default=-1.0, help="as --target-flops but using ACTIVE FLOPs/token, which is the axis the Pareto curves use. For dense the two are identical; for a gated model they differ (1.34x at MST L=24), so an isoFLOP profile mixing both arms must pick one and this is the one that matches the plots (-1 = disable)")
parser.add_argument("--target-param-data-ratio", type=float, default=10.5, help="calculate num_iterations to maintain data:param ratio (Chinchilla=20, -1 = disable)")
# Optimization
parser.add_argument("--device-batch-size", type=int, default=32, help="per-device batch size. good number to reduce to 16,8,4,... if you OOM on VRAM.")
parser.add_argument("--total-batch-size", type=int, default=-1, help="total batch size in tokens. decent numbers are e.g. 524288. (-1 = auto-compute optimal)")
parser.add_argument("--embedding-lr", type=float, default=0.8, help="learning rate for embedding parameters (Adam)")
parser.add_argument("--unembedding-lr", type=float, default=0.008, help="learning rate for unembedding parameters (Adam)")
parser.add_argument("--weight-decay", type=float, default=0.2, help="cautious weight decay for the Muon optimizer (for weights)")
parser.add_argument("--matrix-lr", type=float, default=0.02, help="learning rate for matrix parameters (Muon)")
parser.add_argument("--scalar-lr", type=float, default=0.5, help="learning rate for scalars (resid_lambdas, x0_lambdas)")
parser.add_argument("--adam-beta1", type=float, default=0.8, help="Adam beta1 for embedding/unembedding")
parser.add_argument("--adam-beta2", type=float, default=0.95, help="Adam beta2 for embedding/unembedding")
parser.add_argument("--disable-mu-p", action="store_true",
                    help="disable μP-style LR scaling (model_dim/768)^-0.5 for AdamW params. "
                         "Use when sweeping absolute LRs directly for research models.")
parser.add_argument("--mu-p-scale-override", type=float, default=-1.0, help="force a specific mu-P scale")
parser.add_argument("--warmup-ratio", type=float, default=0.005, help="ratio of iterations for LR warmup")
parser.add_argument("--warmdown-ratio", type=float, default=0.65, help="ratio of iterations for LR warmdown")
parser.add_argument("--final-lr-frac", type=float, default=0.05, help="final LR as fraction of initial LR")
parser.add_argument("--resume-from-step", type=int, default=-1, help="resume training from this step (-1 = disable)")
# Evaluation
parser.add_argument("--eval-every", type=int, default=250, help="evaluate val bpb every N steps (-1 = only at end, 0 = disable)")
parser.add_argument("--log-every", type=int, default=1, help="print step log to console every N steps")
parser.add_argument("--eval-tokens", type=int, default=80*524288, help="number of tokens to evaluate val loss on")
parser.add_argument("--core-metric-every", type=int, default=2000, help="evaluate CORE metric every N steps (-1 = only at end, 0 = disable)")
parser.add_argument("--core-metric-max-per-task", type=int, default=500, help="examples per task for CORE metric")
parser.add_argument("--sample-every", type=int, default=2000, help="sample from model every N steps (-1 = disable)")
parser.add_argument("--save-final-optimizer", type=int, default=0, choices=[0, 1], help="keep optimizer shards in the FINAL checkpoint. They are 2-3x the weights and are only needed to resume, which a finished run cannot do, so the default drops them. Set 1 if you intend to continue training from the end of this run")
parser.add_argument("--save-every", type=int, default=-1, help="save checkpoints every N steps (-1 = only at end)")
parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True, help="enable/disable torch.compile")
parser.add_argument("--timing-probe-steps", type=int, default=0, help="stop after this many steps, but compute num_iterations, the LR schedule and the batch size from the real budget first. Used by the profile scripts' --timer-only to cost a sweep without running it: startup, per-step and final-validation time are all real, and the full run is projected from the measured dt (0 = disabled, run to completion)")
parser.add_argument("--compile-mode", type=str, default="default", choices=["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"], help="torch.compile mode. 'reduce-overhead' adds CUDA graphs, which removes per-kernel launch overhead; MST emits ~3x dense's kernels so it is the arm that benefits. Costs memory for the graph pool. Any non-default value changes step time, so do not mix modes within a comparison")
parser.add_argument("--compile-regional", type=int, default=0, choices=[0, 1], help="compile each transformer layer separately instead of the whole model. Compile time then does not grow with depth (MST L=16: 58s vs 126s), at the cost of cross-layer fusion (~7%% slower steps). Use when only bpb is wanted; leave off for reported wall-clock numbers")
parser.add_argument("--tokenizer-dir", type=str, default=None, help="explicit tokenizer directory (overrides default)")
parser.add_argument("--max-shards", type=int, default=-1, help="maximum number of dataset shards to use (-1 = all)")
# Output
parser.add_argument("--model-tag", type=str, default=None, help="override model tag for checkpoint directory name")
parser.add_argument("--p34-ffn-mult", type=float, default=4.0, help="34: FFN hidden width as a multiple of n_embd (4.0 = standard dense FFN shape, 1.0 = D->D->D)")
parser.add_argument("--p34-ffn-single", type=int, default=0, choices=[0, 1], help="34: replace the FFN with a single RemixedLinear D->D (no hidden layer, no relu^2)")
parser.add_argument("--p34-dense-attn", type=int, default=0, choices=[0, 1], help="34: keep attention dense so only the FFN is RemixedLinear")
# SCH: Structured Code Output Heads (see nanochat/code_head.py and
# structured-code-output-heads-plan.md). The head is logit(w|h) = phi_k(c(w))^T g(h)
# with a frozen binary code c(w) and its monomial expansion up to order k.
parser.add_argument("--use-code-head", type=int, default=0, choices=[0, 1], help="SCH: replace the dense softmax head with a structured code head")
parser.add_argument("--sch-head-type", type=str, default="code", choices=list(HEAD_TYPES), help="SCH: 'code' = code head (monomial or K-ary product Phi); 'hsoftmax' = Huffman tree baseline; 'monarch' = block-diagonal fully-learned head")
parser.add_argument("--sch-bits", type=int, default=0, help="SCH: code length B (0 = ceil(log2 V), the minimal/degenerate code)")
parser.add_argument("--sch-order", type=int, default=2, help="SCH: highest monomial interaction order k (1 = Oda et al. independent bits, B = exact softmax)")
parser.add_argument("--sch-max-m", type=int, default=0, help="SCH: cap on the expansion width M (0 = uncapped); also sets M directly for non-monomial phi modes")
parser.add_argument("--sch-phi-mode", type=str, default="monomial", choices=list(PHI_MODES), help="SCH: what fills Phi. 'product' is the K-ary product code (M = g*K at order 1, gather-add cost V*g); 'random_binary' isolates structure from binariness; 'learned' is the matched-capacity dense-W-at-width-M control; 'onehot' is the g=1 corner (VQ-Logits)")
parser.add_argument("--sch-code-mode", type=str, default="binary", choices=list(CODE_MODES), help="SCH: code assignment arm")
parser.add_argument("--sch-code-path", type=str, default="", help="SCH: .pt holding a (V, B) uint8 code matrix (semantic codes from scripts/code_assign.py)")
parser.add_argument("--sch-code-ecc-bits", type=int, default=0, help="SCH: parity bits appended to the base code; sweeps the ECC-vs-semantic tension on one axis")
parser.add_argument("--sch-code-seed", type=int, default=1234, help="SCH: seed for code assignment and monomial subsampling")
parser.add_argument("--sch-phi-density", type=float, default=0.0, help="SCH: row density for phi-mode=random_binary (0 = match the monomial arm's density)")
parser.add_argument("--sch-phi-dtype", type=str, default="bf16", choices=list(PHI_DTYPES), help="SCH: storage dtype of the frozen Phi. Use fp32 for the Phase 0 rank gate: bf16 rounding puts a noise floor ~1e-3 of the top singular value, which reads as spurious rank")
parser.add_argument("--sch-phi-normalize", type=int, default=1, choices=[0, 1], help="SCH: rescale Phi to unit mean row norm (pure reparameterisation, needed for stability at order 3+)")
parser.add_argument("--sch-phi-center", type=int, default=0, choices=[0, 1], help="SCH: mean-centre Phi columns (ablation)")
parser.add_argument("--sch-g-type", type=str, default="linear", choices=list(G_TYPES), help="SCH: g. A LINEAR g caps logit rank at min(M, d), which makes orders 3 and 4 rank-identical at d=512 and fakes ladder saturation. Run mlp for the ladder sweep")
parser.add_argument("--sch-g-hidden", type=int, default=0, help="SCH: hidden width of the MLP g (0 = n_embd)")
parser.add_argument("--sch-g-layers", type=int, default=2, help="SCH: depth of the MLP g")
parser.add_argument("--sch-g-out-std", type=float, default=0.001, help="SCH: init std of g's final layer (matches lm_head's)")
parser.add_argument("--sch-mixture", type=int, default=1, help="SCH: m code heads mixed by log-sum-exp; m>1 escapes the rank bound entirely (Phase 2 mitigation 1)")
parser.add_argument("--sch-residual-rank", type=int, default=0, help="SCH: dense residual hybrid rank r; buys r rank for rV params (Phase 2 mitigation 3)")
parser.add_argument("--sch-logit-act", type=str, default="none", choices=list(LOGIT_ACTS), help="SCH: pointwise nonlinearity on code logits (Phase 2 mitigation 2)")
parser.add_argument("--sch-bias", type=int, default=0, choices=[0, 1], help="SCH: learned per-token bias. V params and +1 rank, but it BREAKS zero-shot vocabulary extension (a new token has no bias)")
parser.add_argument("--sch-input-mode", type=str, default="table", choices=list(INPUT_MODES), help="SCH Phase 3: input-side arm. 'linear' is expected to collapse at exactly rank B and is run for that reason")
parser.add_argument("--sch-input-hidden", type=int, default=0, help="SCH: hidden width for --sch-input-mode nonlinear (0 = 4 * n_embd)")
parser.add_argument("--sch-product-groups", type=int, default=8, help="SCH: g, digits in the K-ary product code (--sch-phi-mode product)")
parser.add_argument("--sch-product-codebook", type=int, default=256, help="SCH: K, symbols per digit; M = g*K at order 1")
parser.add_argument("--sch-product-source", type=str, default="hash", choices=list(PRODUCT_SOURCES), help="SCH: hash | random | file (assignment from scripts/code_assign.py)")
parser.add_argument("--sch-product-impl", type=str, default="dense", choices=list(PRODUCT_IMPLS), help="SCH: 'dense' materialises the one-hot Phi and runs a GEMM (fast, costs 4*V*M); 'gather' is the V*g arithmetic path and is MEASURABLY SLOWER until a fused kernel exists (OPEN_QUESTIONS Q8)")
parser.add_argument("--sch-phi-whiten", type=int, default=0, help="SCH: whiten Phi; same span, condition number 1, pure reparameterisation")
parser.add_argument("--sch-mixture-per-phi", type=int, default=0, help="SCH: give each mixture component its own Phi (union of subspaces)")
parser.add_argument("--sch-mixture-topk", type=int, default=0, help="SCH: components evaluated per token (0 = all); cost tracks k not K")
parser.add_argument("--sch-mixture-aux", type=float, default=0.01, help="SCH: load-balance weight for sparse mixture routing. 0 lets top-1 collapse onto a single component while paying for K")
parser.add_argument("--sch-monarch-m1", type=int, default=0, help="SCH: Monarch inner factor m1 (0 = sqrt(M))")
parser.add_argument("--sch-monarch-perm", type=str, default="none", choices=list(MONARCH_PERMS), help="SCH: which words share a Monarch block. Costs no FLOPs and no parameters; 'random' is the control that says whether coherence is what helps")
parser.add_argument("--sch-monarch-perm-path", type=str, default="", help="SCH: .pt permutation of range(vocab_size) for --sch-monarch-perm=file (scripts/build_vocab_permutation.py)")
parser.add_argument("--sch-tier-bounds", type=str, default="", help="SCH: cumulative vocabulary edges for --sch-head-type=tiered, e.g. '1024,4096,16384,65536'. N edges define N+1 tiers")
parser.add_argument("--sch-tier-caps", type=str, default="", help="SCH: dimensions per tier, e.g. '512,512,511,480,18'. Solved by scripts/solve_tiers.py against a trained dense head; not a value to guess")
parser.add_argument("--sch-tier-order", type=str, default="freq", choices=list(MONARCH_PERMS), help="SCH: which words land in which tier. 'freq' is the whole point; 'none' (token id) is the control that shows the ordering is what pays")
parser.add_argument("--sch-tier-perm-path", type=str, default="", help="SCH: .pt permutation for --sch-tier-order=file")
parser.add_argument("--sch-proposal-rank", type=int, default=32, help="SCH: c, the proposal rank. Trained to rank rather than reconstruct, so it stays 16-32 at any depth; a truncated SVD would need 0.33d")
parser.add_argument("--sch-proposal-topk", type=int, default=16384, help="SCH: candidates scored EXACTLY, shared across a chunk of tokens so the gather stays a GEMM")
parser.add_argument("--sch-proposal-samples", type=int, default=4096, help="SCH: S importance samples for the partition tail. 0 substitutes the cheap logits instead, which is biased by -0.03 to -1.6 nats")
parser.add_argument("--sch-proposal-warmup", type=int, default=0, help="SCH: steps of exact softmax before switching. The proposal selects nothing useful at initialisation")
parser.add_argument("--sch-proposal-chunk", type=int, default=2048, help="SCH: tokens sharing one candidate set. Per-token sets read the weight 10,240x more than a dense matmul and measured 18.7 s/step")
parser.add_argument("--sch-proposal-aux", type=float, default=1.0, help="SCH: weight on the ranking loss that trains the proposal against the exact logits. 0 leaves it with no gradient at all when --sch-proposal-samples=0")
parser.add_argument("--sch-nfh-mode", type=str, default='cp', help="SCH: cp = mixture of product codes; global = a V x m log-table whose per-token cost is independent of V")
parser.add_argument("--sch-nfh-rank", type=int, default=32, help="SCH: R mixture components (cp) or m table columns (global). R=1 is exactly LightRNN and is the decisive ablation")
parser.add_argument("--sch-nfh-dims", type=str, default='', help="SCH: code axis sizes, e.g. 64,64,32. Must multiply to the PADDED vocabulary; empty derives them")
parser.add_argument("--sch-nfh-groups", type=int, default=3, help="SCH: G, used only when --sch-nfh-dims is empty")
parser.add_argument("--sch-nfh-views", type=int, default=1, help="SCH: independent bijections mixed uniformly; view 0 uses --sch-nfh-perm and the rest are random")
parser.add_argument("--sch-nfh-perm", type=str, default='none', help="SCH: none | random (control) | freq | file. Which words share a code cell; costs nothing")
parser.add_argument("--sch-nfh-perm-path", type=str, default='', help="SCH: .pt permutation of range(vocab_size) from scripts/build_vocab_permutation.py")
parser.add_argument("--sch-nfh-g-type", type=str, default='linear', help="SCH: linear | mlp. The map into the factors. mlp is nearly free once the head is 0.04x of dense")
parser.add_argument("--sch-nfh-g-hidden", type=int, default=0, help="SCH: hidden width for --sch-nfh-g-type mlp (0 = n_embd)")
parser.add_argument("--sch-rerank-mode", type=str, default='topk', help="SCH: topk (the mechanism) | full (same correction over all V) | mos2 (two-component MoS baseline) | temp (per-token logit scale, the floor)")
parser.add_argument("--sch-rerank-k", type=int, default=64, help="SCH: words the correction is applied to, chosen by the model's own top-k. Cost is k*r rather than V*r")
parser.add_argument("--sch-rerank-rank", type=int, default=32, help="SCH: r for the correction. 0 reduces the head to dense EXACTLY, which is the control")
parser.add_argument("--sch-nfh-smooth", type=int, default=0, help="SCH: interpolate a static per-token prior into the mixture. A product mixture cannot give a word mass without pointing a component at it; this buys the unigram back for V params and one gather per token")
parser.add_argument("--sch-nfh-chunk", type=int, default=256, help="SCH: tokens per chunk on the EVAL path only; the training path builds no V-wide tensor")
# Held-out vocabulary: the headline capability experiment. Instrument from day one.
parser.add_argument("--sch-holdout-tokens", type=int, default=0, help="SCH: hold N token ids out of TRAINING so their zero-shot perplexity can be measured against an untrained softmax row")
parser.add_argument("--sch-holdout-seed", type=int, default=7, help="SCH: seed selecting the held-out token ids (must match across arms being compared)")
parser.add_argument("--sch-holdout-min-id", type=int, default=256, help="SCH: never hold out ids below this (byte fallbacks and special tokens)")
parser.add_argument("--sch-holdout-mode", type=str, default="target", choices=list(HOLDOUT_MODES), help="SCH: 'target' masks held-out ids as prediction targets only (isolates the output head, the claim under test); 'full' also rewrites them in the inputs, so the model never sees them at all")
parser.add_argument("--sch-decile-metrics", type=int, default=0, choices=[0, 1], help="SCH: report validation bpb per token-frequency decile at the end of training. DEFAULT 0: this defaulted to 1, which made the end-of-training SCH diagnostics block run for EVERY architecture (the trigger at the bottom of this file is an OR over the sch_* flags), and it OOMs. All 16 c0*/c1* sweeps pass this flag explicitly, so none of them depended on the default.")
parser.add_argument("--sch-rank-probe", type=int, default=0, help="SCH: contexts to use for the end-of-training logit-rank SVD (0 = skip; 50000 is the Phase 0 setting)")
parser.add_argument("--sch-eval-steps", type=int, default=100, help="SCH: validation batches used by the end-of-training diagnostics")
# SAP: sampling-aware block head (nanochat/block_head.py, sap_research_plan.md)
parser.add_argument("--sap-block-t", type=int, default=0, help="SAP: tokens emitted per block by the head (0 = off)")
parser.add_argument("--sap-block-mode", type=str, default="indep", choices=["indep", "p1_discrete", "p2_gauss", "p3_energy", "cp", "local", "local_jacobi", "inv_head", "plain_noise", "wta"], help="SAP: indep (independent slots), p1_discrete / p2_gauss (plan-latent ELBO), p3_energy (energy score), cp / local / local_jacobi / inv_head (baselines), plain_noise / wta (controls)")
parser.add_argument("--sap-block-frac", type=float, default=0.125, help="SAP: fraction of positions carrying a block in training; the head's FLOPs scale with it")
parser.add_argument("--sap-lambda", type=float, default=1.0, help="SAP: weight of the block loss next to the next-token loss")
parser.add_argument("--sap-head-layers", type=int, default=2, help="SAP: attention layers in the slot decoder")
parser.add_argument("--sap-head-mlp-mult", type=int, default=2, help="SAP: MLP width multiple inside the head")
parser.add_argument("--sap-enc-layers", type=int, default=1, help="SAP: posterior encoder layers (p1, p2)")
parser.add_argument("--sap-ctx-window", type=int, default=16, help="SAP: trunk positions the slots cross-attend to (0 = h_t only)")
parser.add_argument("--sap-latent-groups", type=int, default=4, help="SAP: G code groups (p1), free-bits groups (p2)")
parser.add_argument("--sap-latent-codes", type=int, default=16, help="SAP: C codes per group (p1)")
parser.add_argument("--sap-latent-dim", type=int, default=64, help="SAP: z dim (p2) and noise dim (p3, plain_noise, wta)")
parser.add_argument("--sap-free-bits", type=float, default=0.25, help="SAP: unpenalised KL nats per group")
parser.add_argument("--sap-kl-anneal-steps", type=int, default=2000, help="SAP: micro-steps over which the KL weight ramps to 1")
parser.add_argument("--sap-cp-components", type=int, default=8, help="SAP: R for the cp mixture baseline")
parser.add_argument("--sap-wta-k", type=int, default=4, help="SAP: noise draws for the winner-take-all control")
parser.add_argument("--sap-jacobi-sweeps", type=int, default=2, help="SAP: parallel refinement sweeps for local_jacobi (or local)")
parser.add_argument("--sap-eval-steps", type=int, default=20, help="SAP: validation batches for the block bpb at each eval (0 = skip)")
parser.add_argument("--seed", type=int, default=-1, help="RNG seed for weight init and data-order-independent randomness (-1 = unseeded, the historical default). Needed for seed-variance runs; note the dataloader order is not seeded by this.")
parser.add_argument("--early-stop-tokens", type=int, default=-1, help="terminate training after this many tokens without affecting the LR schedule (-1 = disabled)")
parser.add_argument("--step-loss-file", type=str, default="", help="optional JSONL file to write per-step training loss for external sweep plotting")
args = parser.parse_args()
if args.data_dir is not None:
    os.environ["NANOCHAT_DATA_DIR"] = args.data_dir
if args.use_onecycle is not None:
    args.research_onecycle = args.use_onecycle
user_config = vars(args).copy()  # for logging
# -----------------------------------------------------------------------------
# Compute init and wandb logging

device_type = autodetect_device_type() if args.device_type == "" else args.device_type
# torchrun launched us but CUDA was not detected: autodetect fell back to cpu, so no
# process group is initialised and the run would either crash later in wrap_model or,
# worse, silently train on CPU for hours. Neither is something to discover from the
# wall-clock. Fail here with the reason.
if all(k in os.environ for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE")) and device_type != "cuda":
    import torch as _t
    raise RuntimeError(
        f"launched under torchrun but device_type resolved to '{device_type}'. "
        f"torch.cuda.is_available()={_t.cuda.is_available()}, "
        f"device_count={_t.cuda.device_count() if _t.cuda.is_available() else 0}. "
        "The GPU was not visible at startup. Re-launch, or pass --device-type explicitly."
    )
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
print0(f"Device: {device_type} (ddp={ddp}, world_size={ddp_world_size})")

# nn.DataParallel optimization (multi-GPU without torchrun)
is_dp = args.parallel == "dp" and device_type == "cuda" and torch.cuda.device_count() > 1
if is_dp:
    # When using DP, we act like a single world but with larger device_batch_size
    ddp_world_size = torch.cuda.device_count()
    ddp_rank = 0
    ddp_local_rank = 0
    print0(f"✓ Using nn.DataParallel (detected {ddp_world_size} GPUs)")
else:
    if args.parallel == "dp":
        print0(f"i DataParallel requested but suppressed: device_type={device_type}, gpu_count={torch.cuda.device_count()}")

master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0
if device_type == "cuda":
    gpu_device_name = torch.cuda.get_device_name(0)
    gpu_peak_flops = get_peak_flops(gpu_device_name)
    print0(f"GPU: {gpu_device_name} | Peak FLOPS (BF16): {gpu_peak_flops:.2e}")
else:
    gpu_peak_flops = float('inf')  # MFU not meaningful for CPU/MPS
print0(f"COMPUTE_DTYPE: {COMPUTE_DTYPE} ({COMPUTE_DTYPE_REASON})")
if is_dp:
    print0(f"DataParallel enabled: world_size={ddp_world_size}")

# wandb logging init
use_dummy_wandb = True # args.run == "dummy" or not master_process
wandb_run = DummyWandb() # if use_dummy_wandb else wandb.init(project="nanochat", name=args.run, config=user_config)

# Flash Attention backend status
from nanochat.flash_attention import USE_FA4, USE_FA3, _BACKEND as _FA_BACKEND
if USE_FA4:
    print0(f"✓ Using Flash Attention 4 (Blackwell GPU detected) — fastest possible attention.")
elif USE_FA3:
    major, _ = torch.cuda.get_device_capability() if device_type == 'cuda' else (0, 0)
    hw = "Blackwell" if major >= 10 else "Hopper"
    print0(f"✓ Using Flash Attention 3 ({hw} GPU detected) — efficient and fast.")
else:
    print0("!" * 80)
    if (HAS_FA4 or HAS_FA3) and COMPUTE_DTYPE != torch.bfloat16:
        print0(f"WARNING: Flash Attention 3/4 only support bf16, but COMPUTE_DTYPE={COMPUTE_DTYPE}. Using PyTorch SDPA fallback")
    else:
        major, _ = torch.cuda.get_device_capability() if device_type == 'cuda' else (0, 0)
        if major >= 10:
            print0("WARNING: Blackwell GPU detected but FA4 (flash-attn-4) and FA3 (kernels) not found.")
            print0("WARNING: Install FA3:  pip install kernels  (or pip install flash-attn-4 for FA4)")
        elif major >= 9:
            print0("WARNING: Hopper GPU detected but FA3 (kernels package) not found.")
            print0("WARNING: Install FA3:  pip install kernels")
        else:
            print0("WARNING: Flash Attention 3/4 not available (requires Hopper sm90+ or Blackwell sm100+)")
        print0("WARNING: Falling back to PyTorch SDPA — training will be less efficient.")
    if args.window_pattern != "L":
        print0(f"WARNING: SDPA has no support for sliding window attention (window_pattern='{args.window_pattern}'). Your GPU utilization will be terrible.")
        print0("WARNING: Recommend using --window-pattern L for full context attention without alternating sliding window patterns.")
    print0("!" * 80)

# -----------------------------------------------------------------------------
# Tokenizer will be useful for evaluation and also we need the vocab size to init the model
tokenizer = get_tokenizer(tokenizer_dir=args.tokenizer_dir)
token_bytes = get_token_bytes(device=device, tokenizer_dir=args.tokenizer_dir)
vocab_size = tokenizer.get_vocab_size()
print0(f"Vocab size: {vocab_size:,}")

# -----------------------------------------------------------------------------
# Initialize the Model

def build_model_meta(depth, apply_dim_override=True):
    """Build a model on meta device for a given depth (shapes/dtypes only, no data).

    apply_dim_override=False ignores --model-dim and uses the aspect ratio. The d12
    REFERENCE model must be built that way: it is a fixed anchor for the batch-size and
    LR derivations (D_REF, B_REF), so widening it along with the arm under test moves the
    reference and the arm together and silently rescales the whole schedule.
    """
    # Model dim is nudged up to nearest multiple of head_dim for clean division
    # (FA3 requires head_dim divisible by 8, and this guarantees head_dim == args.head_dim exactly)
    if apply_dim_override and getattr(args, 'model_dim', 0) > 0:
        base_model_dim = args.model_dim
    else:
        base_dim = depth * args.aspect_ratio
        base_model_dim = ((base_dim + args.head_dim - 1) // args.head_dim) * args.head_dim
    base_num_heads = base_model_dim // args.head_dim
    if args.use_moe or args.use_remix_linear:
        model_dim = args.moe_embed_dim
        # Keep research branches compatible with repo attention constraints while
        # preferring a head count close to the base model's count.
        def _choose_research_heads(embed_dim: int, preferred_heads: int) -> int:
            pow2 = [1 << i for i in range(0, 12)]  # up to 2048 heads, way above practical usage
            valid_pow2 = [h for h in pow2 if h <= embed_dim and embed_dim % h == 0 and (embed_dim // h) % 8 == 0]
            if valid_pow2:
                return min(valid_pow2, key=lambda h: (abs(h - preferred_heads), -h))
            # Fallback: any divisor that keeps integer head_dim.
            valid_any = [h for h in range(1, embed_dim + 1) if embed_dim % h == 0]
            return min(valid_any, key=lambda h: abs(h - preferred_heads)) if valid_any else 1

        num_heads = _choose_research_heads(model_dim, base_num_heads)
        assert model_dim % num_heads == 0, f"moe_embed_dim must be divisible by n_head ({num_heads}), got {model_dim}"
    else:
        model_dim = base_model_dim
        num_heads = base_num_heads
    config = GPTConfig(
        sequence_len=args.max_seq_len, vocab_size=vocab_size,
        n_layer=depth, n_head=num_heads, n_kv_head=num_heads, n_embd=model_dim,
        window_pattern=args.window_pattern,
        use_moe=args.use_moe,
        use_perm=args.use_perm,
        moe_num_experts=args.moe_num_experts,
        moe_router_dim=args.moe_router_dim,
        moe_embed_dim=args.moe_embed_dim,
        use_remix_linear=args.use_remix_linear,
        remix_context_dim=args.remix_context_dim,
        remix_basis_size=args.remix_basis_size,
        use_pos_embed=args.use_pos_embed,
        moe_use_abs_pos_embed=bool(args.moe_use_abs_pos_embed),
        remixed_linear_kwargs=dict(
            use_basis_gate=bool(args.remix_use_basis_gate),
            use_output_gate=bool(args.remix_use_output_gate),
            use_context=bool(args.remix_use_context),
            basis_gate_mode=getattr(args, 'remix_basis_gate_mode', 'mlp'),
            sparse_gate_k=getattr(args, 'cclblock_sparse_gate_k', 0),
            gate_temperature=getattr(args, 'cclblock_gate_temperature', 1.0),
            basis_scale_factor=getattr(args, 'remix_basis_scale_factor', 4),
            # n_templates = K_total (total experts in the bank):
            # When p23_tiny_expert=1, p23_n_experts sets K_total.
            # Otherwise falls back to p22_n_templates (legacy template bank).
            n_templates=(
                getattr(args, 'p23_n_experts', 64)
                if getattr(args, 'p23_tiny_expert', 0)
                else getattr(args, 'p22_n_templates', 1)
            ),
            template_routing_learned=bool(
                getattr(args, 'p23_learned_route', 0)
                if getattr(args, 'p23_tiny_expert', 0) or getattr(args, 'p23_lokr', 0)
                else getattr(args, 'p22_template_routing_learned', 0)
            ),
            template_topk=getattr(args, 'p22_template_topk', 0),
            tiny_expert=bool(getattr(args, 'p23_tiny_expert', 0)),
            tiny_expert_topk=getattr(args, 'p23_topk', 16),
            lokr_expert=bool(getattr(args, 'p23_lokr', 0)),
            lokr_n_experts=getattr(args, 'p23_n_experts', 64),
            lokr_topk=getattr(args, 'p23_topk', 16),
            lokr_rank=getattr(args, 'p23_lokr_rank', 4),
            lokr_learned=bool(getattr(args, 'p23_learned_route', 0)),
            basis_gate_rank=getattr(args, 'remix_basis_gate_rank', 8),
            # Phase 30: LayerNorm ablation
            disable_ln_basis=bool(getattr(args, 'remix_disable_ln_basis', 0)),
            # Phase 31: chunk-routing throughput
            chunk_route_impl=getattr(args, 'p31_chunk_route_impl', 'compose'),
            top1_gate=getattr(args, 'p31_top1_gate', 'ones'),
            template_delta_rank=getattr(args, 'p31_template_delta_rank', 0),
            route_side=getattr(args, 'p31_route_side', 'output'),
            drop_basis_proj=bool(getattr(args, 'p31_drop_basis_proj', 0)),
            basis_side_templates=getattr(args, 'p31_basis_side_templates', 0),
            grad_equalize=bool(getattr(args, 'p29_grad_equalize', 0)),
        ),

        # Fix 1A
        use_layer_context=bool(getattr(args, 'use_layer_context', 1)),
        # Fix 1B
        scale_basis_size=bool(getattr(args, 'scale_basis_size', 1)),
        # Fix 1D
        perm_expert_mode=getattr(args, 'perm_expert_mode', 'low_rank'),
        perm_rank=getattr(args, 'perm_rank', 16),
        router_context_window=getattr(args, 'router_context_window', -1),
        # CCL block redesign
        cclblock_modulation=getattr(args, 'cclblock_modulation', 'weight'),
        cclblock_orth_lambda=getattr(args, 'cclblock_orth_lambda', 0.0),
        p35_template_diversity_lambda=getattr(args, 'p35_template_diversity_lambda', 0.0),
        cclblock_context_stream=getattr(args, 'cclblock_context_stream', 'local'),
        cclblock_ema_factor=getattr(args, 'cclblock_ema_factor', 0.99),
        cclblock_stale_ctx_lag=getattr(args, 'cclblock_stale_ctx_lag', 0),
        # Novel ablation designs
        cclblock_sparse_gate_k=getattr(args, 'cclblock_sparse_gate_k', 0),
        cclblock_gate_temperature=getattr(args, 'cclblock_gate_temperature', 1.0),
        cclblock_context_bank_size=getattr(args, 'cclblock_context_bank_size', 0),
        cclblock_per_head_ctx=bool(getattr(args, 'cclblock_per_head_ctx', 0)),
        cclblock_context_source=getattr(args, 'cclblock_context_source', 'norm_x'),
        # Phase 8
        cclblock_chunk_size=getattr(args, 'cclblock_chunk_size', 0),
        cclblock_aux_objective=getattr(args, 'cclblock_aux_objective', 'none'),
        cclblock_aux_lambda=getattr(args, 'cclblock_aux_lambda', 0.1),
        cclblock_boundary_token_id=getattr(args, 'cclblock_boundary_token_id', 198),
        use_ral=bool(getattr(args, 'use_ral', 0)),
        ral_rank=getattr(args, 'ral_rank', 32),
        # Phase 35: ConditionedLinear
        p34_ffn_schedule=getattr(args, 'p34_ffn_schedule', ''),
        p34_ffn_no_ffn_replacement=getattr(args, 'p34_ffn_no_ffn_replacement', 'none'),
        p34_ffn_last_depth=getattr(args, 'p34_ffn_last_depth', 1),
        p36_swiglu_ffn=getattr(args, 'p36_swiglu_ffn', 0),
        p36_swiglu_mult=getattr(args, 'p36_swiglu_mult', 8.0/3.0),
        cond_sites=getattr(args, 'cond_sites', 'both'),
        p22_route_affine=getattr(args, 'p22_route_affine', 0),
        cond_rank=getattr(args, 'cond_rank', 256),
        cond_mult_steps=getattr(args, 'cond_mult_steps', 0),
        cond_gate_source=getattr(args, 'cond_gate_source', 'router'),
        cond_coeff_act=getattr(args, 'cond_coeff_act', 'centered'),
        cond_router_rank=getattr(args, 'cond_router_rank', 0),
        cond_router_act=getattr(args, 'cond_router_act', 'none'),
        cond_live_init=getattr(args, 'cond_live_init', 0.0),
        cond_chunk_size=getattr(args, 'cond_chunk_size', 0),
        cond_mult_scale=getattr(args, 'cond_mult_scale', -1.0),
        cond_mult_impl=getattr(args, 'cond_mult_impl', 'wy'),
        cond_attn_projs=getattr(args, 'cond_attn_projs', 'qkvo'),
        cond_layer_frac=getattr(args, 'cond_layer_frac', 1.0),
        cclblock_film_gate=bool(getattr(args, 'cclblock_film_gate', 0)),
        cclblock_attn_shadow_dim=getattr(args, 'cclblock_attn_shadow_dim', 0),
        cclblock_dynamic_ratio=getattr(args, 'cclblock_dynamic_ratio', 0.25),
        cclblock_gate_rank=getattr(args, 'cclblock_gate_rank', 8),
        cclblock_num_regimes=getattr(args, 'cclblock_num_regimes', 8),
        cclblock_regime_temperature=getattr(args, 'cclblock_regime_temperature', 1.0),
        cclblock_poly_order=getattr(args, 'cclblock_poly_order', 2),
        cclblock_lie_generators=getattr(args, 'cclblock_lie_generators', 4),
        cclblock_grassmann_bank_size=getattr(args, 'cclblock_grassmann_bank_size', 4),
        cclblock_tucker_rank=getattr(args, 'cclblock_tucker_rank', 32),
        cclblock_tucker_modes=getattr(args, 'cclblock_tucker_modes', 8),
        cclblock_svs_rank=getattr(args, 'cclblock_svs_rank', 64),
        cclblock_svs_eps=getattr(args, 'cclblock_svs_eps', 0.1),
        cclblock_vq_codes=getattr(args, 'cclblock_vq_codes', 8),
        cclblock_vq_temperature=getattr(args, 'cclblock_vq_temperature', 1.0),
        cclblock_dcu_warmup_steps=getattr(args, 'cclblock_dcu_warmup_steps', 0),
        # Phase 12: FSI/AESP/CKR
        cclblock_fsi_rotations=getattr(args, 'cclblock_fsi_rotations', 8),
        cclblock_fsi_selector_dim=getattr(args, 'cclblock_fsi_selector_dim', 64),
        cclblock_aesp_strata=getattr(args, 'cclblock_aesp_strata', 4),
        cclblock_aesp_delta_rank=getattr(args, 'cclblock_aesp_delta_rank', 4),
        cclblock_ckr_branches=getattr(args, 'cclblock_ckr_branches', 4),
        cclblock_ckr_kernel_size=getattr(args, 'cclblock_ckr_kernel_size', 64),
        # Phase 13: CKR enhancements
        cclblock_ckr_pos_channels=getattr(args, 'cclblock_ckr_pos_channels', 1),
        cclblock_ckr_dual_optim=getattr(args, 'cclblock_ckr_dual_optim', 0),
        cclblock_ckr_content_bias=getattr(args, 'cclblock_ckr_content_bias', 0.0),
        # Phase 14: Gradient-isolated content conditioning
        cclblock_giad_rank=getattr(args, 'cclblock_giad_rank', 32),
        cclblock_psg_kernel_size=getattr(args, 'cclblock_psg_kernel_size', 64),
        cclblock_ss_dynamic_ratio=getattr(args, 'cclblock_ss_dynamic_ratio', 0.25),
        cclblock_ss_branches=getattr(args, 'cclblock_ss_branches', 2),
        cclblock_ss_kernel_size=getattr(args, 'cclblock_ss_kernel_size', 64),
        # Phase 15: LoKR
        cclblock_lokr_branches=getattr(args, 'cclblock_lokr_branches', 8),
        cclblock_lokr_rank=getattr(args, 'cclblock_lokr_rank', 16),
        # Phase 18: Beyond CKR
        p18_layer_drop=getattr(args, 'p18_layer_drop', 0.0),
        p18_dynamic_activation=getattr(args, 'p18_dynamic_activation', 0),
        p18_mixture_norm=getattr(args, 'p18_mixture_norm', 0),
        p18_aux_sim_lambda=getattr(args, 'p18_aux_sim_lambda', 0.0),
        p18_gradient_penalty=getattr(args, 'p18_gradient_penalty', 0.0),
        p18_per_channel_scale=getattr(args, 'p18_per_channel_scale', 0),
        # Phase 19: Zero-overhead indirect modulation
        p19_residual_gate=getattr(args, 'p19_residual_gate', 0),
        p19_head_importance=getattr(args, 'p19_head_importance', 0),
        p19_residual_mix_groups=getattr(args, 'p19_residual_mix_groups', 0),
        p19_attn_logit_bias=getattr(args, 'p19_attn_logit_bias', 0),
        p19_residual_decay=getattr(args, 'p19_residual_decay', 0),
        p19_grad_equilibrium=getattr(args, 'p19_grad_equilibrium', 0.0),
        p19_spectral_reparam=getattr(args, 'p19_spectral_reparam', 0),
        p19_weight_anticollapse=getattr(args, 'p19_weight_anticollapse', 0.0),
        p19_ve_bias=getattr(args, 'p19_ve_bias', 0),
        p19_weight_noise=getattr(args, 'p19_weight_noise', 0.0),
        # Phase 20: Context-conditioned dynamic weight computation
        p20_hrcs_scale=getattr(args, 'p20_hrcs_scale', 0),
        p20_lswr_scale=getattr(args, 'p20_lswr_scale', 0),
        p20_lswr_planes=getattr(args, 'p20_lswr_planes', 8),
        p20_lrcfb_branches=getattr(args, 'p20_lrcfb_branches', 0),
        p20_lrcfb_narrow=getattr(args, 'p20_lrcfb_narrow', 0),
        p20_lrcfb_learned=getattr(args, 'p20_lrcfb_learned', 0),
        p20_lrcfb_topk=getattr(args, 'p20_lrcfb_topk', 0),
        p20_dgcr_branches=getattr(args, 'p20_dgcr_branches', 0),
        p20_dgcr_aux_weight=getattr(args, 'p20_dgcr_aux_weight', 0.01),
        p20_mone_experts=getattr(args, 'p20_mone_experts', 0),
        p20_mone_topk=getattr(args, 'p20_mone_topk', 0),
        p20_mone_narrow=getattr(args, 'p20_mone_narrow', 1),
        p20_mone_frozen=getattr(args, 'p20_mone_frozen', 0),
        p20_ncea_branches=getattr(args, 'p20_ncea_branches', 0),
        p20_ncea_eps=getattr(args, 'p20_ncea_eps', 0.1),
        p20_adwi=getattr(args, 'p20_adwi', 0),
        # Phase 2 proposals
        p20_pwu_branches=getattr(args, 'p20_pwu_branches', 0),
        p20_pwu_phase=getattr(args, 'p20_pwu_phase', 1),
        p20_fsvd_gate=getattr(args, 'p20_fsvd_gate', 0),
        p20_wbfc_clusters=getattr(args, 'p20_wbfc_clusters', 0),
        p20_wbfc_active=getattr(args, 'p20_wbfc_active', 0),
        # Phase 21
        p21_per_experts=getattr(args, 'p21_per_experts', 0),
        p21_per_topk=getattr(args, 'p21_per_topk', 0),
        p21_per_learned=getattr(args, 'p21_per_learned', 0),
        p21_per_attn=getattr(args, 'p21_per_attn', 0),
        # Phase 22
        p22_attn_moe_route=getattr(args, 'p22_attn_moe_route', 'none'),
        # Phase 23: Tiny Expert RemixedLinear + Standard MoE baseline
        p23_tiny_expert=getattr(args, 'p23_tiny_expert', 0),
        p23_n_experts=getattr(args, 'p23_n_experts', 64),
        p23_topk=getattr(args, 'p23_topk', 16),
        p23_learned_route=getattr(args, 'p23_learned_route', 0),
        p23_std_moe_experts=getattr(args, 'p23_std_moe_experts', 0),
        p23_std_moe_topk=getattr(args, 'p23_std_moe_topk', 1),
        p23_std_moe_aux_weight=getattr(args, 'p23_std_moe_aux_weight', 0.01),
        p23_lokr=getattr(args, 'p23_lokr', 0),
        p23_lokr_rank=getattr(args, 'p23_lokr_rank', 4),
        p23_use_shared_block_router=getattr(args, 'p23_use_shared_block_router', 0),
        p23_linear_moe_experts=getattr(args, 'p23_linear_moe_experts', 0),
        p23_linear_moe_topk=getattr(args, 'p23_linear_moe_topk', 0),
        p23_quantile_route=getattr(args, 'p23_quantile_route', 0),
        p34_ffn_mult=getattr(args, 'p34_ffn_mult', 4.0),
        p34_ffn_single=getattr(args, 'p34_ffn_single', 0),
        p34_dense_attn=getattr(args, 'p34_dense_attn', 0),
        remix_shared_context_gates=getattr(args, 'remix_shared_context_gates', 0),
        remix_use_dual_gate=bool(getattr(args, 'remix_use_dual_gate', 0)),
        remix_use_fused=bool(getattr(args, 'remix_use_fused', 0)),
        p26_output_gated_linear=getattr(args, 'p26_output_gated_linear', 0),
        # Phase 28: FLOPs-efficient template routing
        p28_shared_basis=getattr(args, 'p28_shared_basis', 0),
        p28_chunk_routing_size=getattr(args, 'p28_chunk_routing_size', 0),
        p28_global_template_bank=getattr(args, 'p28_global_template_bank', 'none'),
        p28_attn_proj_templates=getattr(args, 'p28_attn_proj_templates', 0),
        p28_attn_qk_templates=getattr(args, 'p28_attn_qk_templates', 0),
        # Phase 24
        p24_use_sliced_weight=getattr(args, 'p24_use_sliced_weight', 0),
        p24_sliced_weight_reduction_scale=getattr(args, 'p24_sliced_weight_reduction_scale', 8),
        p24_sliced_weight_min_select=getattr(args, 'p24_sliced_weight_min_select', 128),
        p24_sliced_weight_scope=getattr(args, 'p24_sliced_weight_scope', 'per_token'),
        p24_sliced_weight_balance_coeff=getattr(args, 'p24_sliced_weight_balance_coeff', 0.01),
        p24_quantile_route=getattr(args, 'p24_quantile_route', 0),
        p24_use_folded_mod=getattr(args, 'p24_use_folded_mod', 0),
        p24_folded_mod_reduction_scale=getattr(args, 'p24_folded_mod_reduction_scale', 8),
        p24_folded_mod_min_dim=getattr(args, 'p24_folded_mod_min_dim', 128),
        p24_folded_mod_scope=getattr(args, 'p24_folded_mod_scope', 'per_layer'),
        p24_folded_mod_gate_act=getattr(args, 'p24_folded_mod_gate_act', 'tanh_centered'),
        p24_use_sequence_gated_linear=getattr(args, 'p24_use_sequence_gated_linear', 0),
        p24_sequence_gated_scope=getattr(args, 'p24_sequence_gated_scope', 'per_layer'),
        p24_sequence_gated_act=getattr(args, 'p24_sequence_gated_act', 'tanh_centered'),
        remix_output_gate_rank=getattr(args, 'remix_output_gate_rank', 16),
        # Phase 30: LayerNorm ablation
        remix_disable_ln_basis=getattr(args, 'remix_disable_ln_basis', 0),
        dense_intermediate_ln=getattr(args, 'dense_intermediate_ln', 0),
        # Phase 29: RemixedLinear optimizer optimizations
        p29_template_block_diag=getattr(args, 'remix_template_block_diag', 0),
        p29_template_lr_scale=getattr(args, 'remix_template_lr_scale', 1.0),
        p29_grad_equalize=getattr(args, 'p29_grad_equalize', 0),
        # MST: Modular Sub-Transformer
        use_mol=bool(getattr(args, 'use_mol', 0)),
        mol_n_blocks=getattr(args, 'mol_n_blocks', 5),
        mol_n_shared=getattr(args, 'mol_n_shared', 1),
        mol_topk=getattr(args, 'mol_topk', 3),
        mol_thin_dim=getattr(args, 'mol_thin_dim', 256),
        mol_head_dim=getattr(args, 'mol_head_dim', 64),
        mol_ffn_mult=getattr(args, 'mol_ffn_mult', 4.0),
        mol_router_aux=getattr(args, 'mol_router_aux', 0.05),
        mol_routed_attn=getattr(args, 'mol_routed_attn', 'softmax'),
        mol_dispatch=getattr(args, 'mol_dispatch', 0),
        mol_capacity_factor=getattr(args, 'mol_capacity_factor', 1.0),
        mol_block_lr_scale=getattr(args, 'mol_block_lr_scale', 1.0),
        mol_per_block_ve=getattr(args, 'mol_per_block_ve', 0),
        use_binary=bool(getattr(args, 'use_binary', 0)),
        binary_acts=bool(getattr(args, 'binary_acts', 1)),
        binary_weight_scale=getattr(args, 'binary_weight_scale', 'row'),
        binary_act_scale=getattr(args, 'binary_act_scale', 'token'),
        binary_clip=float(getattr(args, 'binary_clip', 1.0)),
        binary_linear=bool(getattr(args, 'binary_linear', 1)),
        binary_embeddings=bool(getattr(args, 'binary_embeddings', 1)),
        binary_skip=getattr(args, 'binary_skip', ''),
        binary_native=bool(getattr(args, 'binary_native', 0)),
        binary_tau=float(getattr(args, 'binary_tau', 1.0)),
        binary_hard=bool(getattr(args, 'binary_hard', 0)),
        binary_resid_width=int(getattr(args, 'binary_resid_width', 0)),
        use_mst=bool(getattr(args, 'use_mst', 0)),
        mst_n_subs=getattr(args, 'mst_n_subs', 8),
        mst_sub_dim=getattr(args, 'mst_sub_dim', 64),
        mst_head_dim=getattr(args, 'mst_head_dim', 0),
        mst_input_mode=getattr(args, 'mst_input_mode', 'fixed_slice'),
        mst_rotated_slice_learned=bool(getattr(args, 'mst_rotated_slice_learned', 0)),
        mst_routing_mode=getattr(args, 'mst_routing_mode', 'soft_weighted'),
        mst_routing_topk=getattr(args, 'mst_routing_topk', 4),
        mst_routing_aux_weight=getattr(args, 'mst_routing_aux_weight', 0.01),
        mst_diversity_weight=getattr(args, 'mst_diversity_weight', 0.0),
        mst_ffn_mode=getattr(args, 'mst_ffn_mode', 'standard'),
        mst_ffn_every=getattr(args, 'mst_ffn_every', 1),
        mst_transition_mode=getattr(args, 'mst_transition_mode', 'parallel'),
        mst_final_mode=getattr(args, 'mst_final_mode', 'aggregate_proj'),
        mst_final_topk=getattr(args, 'mst_final_topk', -1),
        mst_ffn_shared_up=getattr(args, 'mst_ffn_shared_up', 0),
        mst_ffn_inner_dim=getattr(args, 'mst_ffn_inner_dim', 0),
        mst_sub_dropout=getattr(args, 'mst_sub_dropout', 0.0),
        mst_transition_every=getattr(args, 'mst_transition_every', 1),
        mst_ffa_temperature=getattr(args, 'mst_ffa_temperature', 1.0),
        mst_global_residual=getattr(args, 'mst_global_residual', 0),
        mst_hybrid_dense=getattr(args, 'mst_hybrid_dense', 0),
        mst_cross_sub_kv=getattr(args, 'mst_cross_sub_kv', 0),
        mst_sub_aux_weight=getattr(args, 'mst_sub_aux_weight', 0.0),
        mst_progressive_merge=getattr(args, 'mst_progressive_merge', 0),
        mst_multi_scale_windows=getattr(args, 'mst_multi_scale_windows', 0),
        mst_delta_residual=getattr(args, 'mst_delta_residual', 0),
        mst_sub_layers=getattr(args, 'mst_sub_layers', 1),
        # Stage 7: Scaling improvements (P07)
        mst_grad_equalize=getattr(args, 'mst_grad_equalize', 0),
        mst_block_diagonal_muon=getattr(args, 'mst_block_diagonal_muon', 0),
        mst_transition_width_mult=getattr(args, 'mst_transition_width_mult', 1.0),
        mst_sub_lr_scale=getattr(args, 'mst_sub_lr_scale', 1.0),
        mst_shared_expert=getattr(args, 'mst_shared_expert', 0),
        mst_router_entropy_weight=getattr(args, 'mst_router_entropy_weight', 0.0),
        mst_shared_kv_attn=getattr(args, 'mst_shared_kv_attn', 0),
        mst_contrastive_diversity_weight=getattr(args, 'mst_contrastive_diversity_weight', 0.0),
        # Stage 8: Transition expressivity
        mst_transition_nonlinear=getattr(args, 'mst_transition_nonlinear', 0),
        mst_transition_gated=getattr(args, 'mst_transition_gated', 0),
        mst_transition_mlp=getattr(args, 'mst_transition_mlp', 0),
        # Stage 9: Cross-sub expressivity
        mst_cross_sub_gate=getattr(args, 'mst_cross_sub_gate', 0),
        mst_hyper_connect=getattr(args, 'mst_hyper_connect', 0),
        mst_cross_kv_inject=getattr(args, 'mst_cross_kv_inject', 0),
        # Stage 10: Structural transition improvements
        mst_slice_transition=getattr(args, 'mst_slice_transition', 0),
        mst_lookback_layers=getattr(args, 'mst_lookback_layers', 0),
        mst_bilinear_transition=getattr(args, 'mst_bilinear_transition', 0),
        # Stage 11: Attention bottleneck + structural improvements
        mst_cross_sub_qmod=getattr(args, 'mst_cross_sub_qmod', 0),
        mst_feature_cycle=getattr(args, 'mst_feature_cycle', 0),
        mst_mean_transition=getattr(args, 'mst_mean_transition', 0),
        # Stage 12: dense-parity fixes
        mst_sub_head_dim=getattr(args, 'mst_sub_head_dim', 0),
        mst_final_norm=getattr(args, 'mst_final_norm', 0),
        mst_per_stream_ve=getattr(args, 'mst_per_stream_ve', 0),
        mst_ve_map=getattr(args, 'mst_ve_map', 0),
        mst_ve_map_rank=getattr(args, 'mst_ve_map_rank', 0),
        # Stage 13: overhead cuts
        mst_lm_head_dim=getattr(args, 'mst_lm_head_dim', 0),
        mst_compose_windows=getattr(args, 'mst_compose_windows', 0),
        # Stage 14: free cross-stream mixing
        mst_channel_mix=getattr(args, 'mst_channel_mix', 'none'),
        mst_channel_mix_site=getattr(args, 'mst_channel_mix_site', 'layer'),
        # Stage 15: coupling optimization + attention cross-stream mixing
        mst_distribute_block_muon=getattr(args, 'mst_distribute_block_muon', 0),
        mst_trans_spectral_lr=getattr(args, 'mst_trans_spectral_lr', 0),
        mst_talking_heads=getattr(args, 'mst_talking_heads', 0),
        mst_wo_mode=getattr(args, 'mst_wo_mode', 'block'),
        # Stage 16: conditional stream execution
        mst_stream_topk=getattr(args, 'mst_stream_topk', 0),
        mst_stream_shared=getattr(args, 'mst_stream_shared', 0),
        mst_stream_router_aux=getattr(args, 'mst_stream_router_aux', 0.01),
        mst_stream_router_noise=getattr(args, 'mst_stream_router_noise', 0.0),
        mst_stream_dispatch=getattr(args, 'mst_stream_dispatch', 0),
        mst_stream_capacity_factor=getattr(args, 'mst_stream_capacity_factor', 1.0),
        mst_stream_gate_attn=getattr(args, 'mst_stream_gate_attn', 0),
        # Stage 17: block-diagonal Shampoo
        mst_ffn_monarch=getattr(args, 'mst_ffn_monarch', 'none'),
        mst_shampoo=getattr(args, 'mst_shampoo', 0),
        mst_precond_every=getattr(args, 'mst_precond_every', 10),
        mst_shampoo_beta=getattr(args, 'mst_shampoo_beta', 0.95),
        # EET: Early Exit Transformer
        use_eet=bool(getattr(args, 'use_eet', 0)),
        eet_frozen_kv=bool(getattr(args, 'eet_frozen_kv', 1)),
        eet_reenter_final=bool(getattr(args, 'eet_reenter_final', 0)),
        eet_router_type=getattr(args, 'eet_router_type', 'mlp2'),
        eet_router_hidden=getattr(args, 'eet_router_hidden', 0),
        eet_freq_prior_alpha=getattr(args, 'eet_freq_prior_alpha', 0.0),
        eet_pos_prior_beta=getattr(args, 'eet_pos_prior_beta', 0.0),
        eet_domain_prior=bool(getattr(args, 'eet_domain_prior', 0)),
        eet_warmup_frac=getattr(args, 'eet_warmup_frac', 0.02),
        eet_explore_frac=getattr(args, 'eet_explore_frac', 0.15),
        eet_reconstruct_lambda=getattr(args, 'eet_reconstruct_lambda', 1.0),
        eet_efficiency_lambda_start=getattr(args, 'eet_efficiency_lambda_start', 0.01),
        eet_efficiency_lambda_end=getattr(args, 'eet_efficiency_lambda_end', 0.1),
        eet_translator_rank=getattr(args, 'eet_translator_rank', 0),
        eet_max_frozen_kv_frac=getattr(args, 'eet_max_frozen_kv_frac', 0.75),
        eet_exit_threshold=getattr(args, 'eet_exit_threshold', 0.5),
        eet_min_exit_layer=getattr(args, 'eet_min_exit_layer', 1),
        eet_loss_variant=getattr(args, 'eet_loss_variant', 'reconstruct'),
        eet_topk_vocab=int(getattr(args, 'eet_topk_vocab', 512)),
        eet_entropy_lambda=float(getattr(args, 'eet_entropy_lambda', 0.3)),
        eet_surprise_lambda=float(getattr(args, 'eet_surprise_lambda', 0.1)),
        eet_adv_lambda=float(getattr(args, 'eet_adv_lambda', 1.0)),
        eet_adv_entropy_lambda=float(getattr(args, 'eet_adv_entropy_lambda', 0.2)),
        eet_quality_lambda=float(getattr(args, 'eet_quality_lambda', 1.0)),
        eet_quality_entropy_bonus=float(getattr(args, 'eet_quality_entropy_bonus', 0.1)),
        eet_gumbel_temp_start=float(getattr(args, 'eet_gumbel_temp_start', 0.0)),
        eet_gumbel_temp_end=float(getattr(args, 'eet_gumbel_temp_end', 0.1)),
        eet_gumbel_hard=bool(getattr(args, 'eet_gumbel_hard', 1)),
        eet_commitment_beta=float(getattr(args, 'eet_commitment_beta', 0.1)),
        eet_global_router=bool(getattr(args, 'eet_global_router', 0)),
        eet_freq_efficiency_alpha=float(getattr(args, 'eet_freq_efficiency_alpha', 0.0)),
        eet_diversity_lambda=float(getattr(args, 'eet_diversity_lambda', 0.0)),
        eet_ce_guided_lambda=float(getattr(args, 'eet_ce_guided_lambda', 1.0)),
        eet_router_lr_mult=float(getattr(args, 'eet_router_lr_mult', 5.0)),
        eet_model_lr_mult=float(getattr(args, 'eet_model_lr_mult', 1.0)),
        eet_depth_weight_type=str(getattr(args, 'eet_depth_weight_type', 'none')),
        eet_depth_weight_max=float(getattr(args, 'eet_depth_weight_max', 2.5)),
        eet_use_override=int(getattr(args, 'eet_use_override', 0)),
        eet_override_prob_start=float(getattr(args, 'eet_override_prob_start', 0.5)),
        eet_override_prob_end=float(getattr(args, 'eet_override_prob_end', 0.1)),
        eet_compute_skip=bool(getattr(args, 'eet_compute_skip', 0)),
        eet_target_active_frac=float(getattr(args, 'eet_target_active_frac', 0.125)),
        eet_capacity_schedule=getattr(args, 'eet_capacity_schedule', 'bell'),
        eet_kv_mode=getattr(args, 'eet_kv_mode', 'none'),
        eet_route_noise=float(getattr(args, 'eet_route_noise', 0.0)),
        eet_route_noise_end=float(getattr(args, 'eet_route_noise_end', -1.0)),
        eet_coverage_diag=bool(getattr(args, 'eet_coverage_diag', 0)),
        eet_kv_eager=bool(getattr(args, 'eet_kv_eager', 0)),
        deep_supervision_lambda=float(getattr(args, 'deep_supervision_lambda', 0.0)),
        deep_supervision_frac=float(getattr(args, 'deep_supervision_frac', 0.125)),
        eet_width_power=float(getattr(args, 'eet_width_power', 0.0)),
        eet_width_cap=float(getattr(args, 'eet_width_cap', 16.0)),
        eet_exit_fracs=[float(x.strip()) for x in getattr(args, 'eet_exit_fracs', '').split(',') if x.strip()] if getattr(args, 'eet_exit_fracs', '') else None,
        eet_capacity_alignment_lambda=float(getattr(args, 'eet_capacity_alignment_lambda', 0.0)),
        eet_router_task_grad=bool(getattr(args, 'eet_router_task_grad', 1)),
        eet_reinforce_interval=int(getattr(args, 'eet_reinforce_interval', 0)),
        eet_reinforce_lambda=float(getattr(args, 'eet_reinforce_lambda', 0.1)),
        eet_exit_adapter_rank=int(getattr(args, 'eet_exit_adapter_rank', 0)),
        eet_router_after_block=int(getattr(args, 'eet_router_after_block', 0)),
        eet_ffn_skip=bool(getattr(args, 'eet_ffn_skip', 0)),
        eet_ffn_target_frac=float(getattr(args, 'eet_ffn_target_frac', 0.50)),
        eet_ffn_full_attn=bool(getattr(args, 'eet_ffn_full_attn', 1)),
        eet_depth_affine=bool(getattr(args, 'eet_depth_affine', 0)),
        eet_capacity_anneal_frac=float(getattr(args, 'eet_capacity_anneal_frac', 0.0)),
        eet_learned_schedule=bool(getattr(args, 'eet_learned_schedule', 0)),
        eet_departure_summary=bool(getattr(args, 'eet_departure_summary', 0)),
        eet_route_consistency_lambda=float(getattr(args, 'eet_route_consistency_lambda', 0.0)),
        eet_dense_distill_interval=int(getattr(args, 'eet_dense_distill_interval', 0)),
        eet_dense_distill_lambda=float(getattr(args, 'eet_dense_distill_lambda', 0.5)),
        eet_depth_lr_scale=bool(int(getattr(args, 'eet_depth_lr_scale', 0))),
        eet_depth_grad_scale=bool(int(getattr(args, 'eet_depth_grad_scale', 0))),
        eet_detach_aux_from_backbone=bool(int(getattr(args, 'eet_detach_aux_from_backbone', 0))),
        eet_detach_exit_from_backbone=bool(int(getattr(args, 'eet_detach_exit_from_backbone', 0))),
        # SCH: Structured Code Output Heads
        use_code_head=bool(int(getattr(args, 'use_code_head', 0))),
        sch_head_type=getattr(args, 'sch_head_type', 'code'),
        sch_bits=int(getattr(args, 'sch_bits', 0)),
        sch_order=int(getattr(args, 'sch_order', 2)),
        sch_max_m=int(getattr(args, 'sch_max_m', 0)),
        sch_phi_mode=getattr(args, 'sch_phi_mode', 'monomial'),
        sch_code_mode=getattr(args, 'sch_code_mode', 'binary'),
        sch_code_path=getattr(args, 'sch_code_path', ''),
        sch_code_ecc_bits=int(getattr(args, 'sch_code_ecc_bits', 0)),
        sch_code_seed=int(getattr(args, 'sch_code_seed', 1234)),
        sch_phi_density=float(getattr(args, 'sch_phi_density', 0.0)),
        sch_phi_dtype=getattr(args, 'sch_phi_dtype', 'bf16'),
        sch_phi_normalize=int(getattr(args, 'sch_phi_normalize', 1)),
        sch_phi_center=int(getattr(args, 'sch_phi_center', 0)),
        sch_g_type=getattr(args, 'sch_g_type', 'linear'),
        sch_g_hidden=int(getattr(args, 'sch_g_hidden', 0)),
        sch_g_layers=int(getattr(args, 'sch_g_layers', 2)),
        sch_g_out_std=float(getattr(args, 'sch_g_out_std', 0.001)),
        sch_mixture=int(getattr(args, 'sch_mixture', 1)),
        sch_residual_rank=int(getattr(args, 'sch_residual_rank', 0)),
        sch_logit_act=getattr(args, 'sch_logit_act', 'none'),
        sch_bias=int(getattr(args, 'sch_bias', 0)),
        sch_input_mode=getattr(args, 'sch_input_mode', 'table'),
        sch_input_hidden=int(getattr(args, 'sch_input_hidden', 0)),
        sch_product_groups=int(getattr(args, 'sch_product_groups', 8)),
        sch_product_codebook=int(getattr(args, 'sch_product_codebook', 256)),
        sch_product_source=str(getattr(args, 'sch_product_source', 'hash')),
        sch_product_impl=str(getattr(args, 'sch_product_impl', 'dense')),
        sch_phi_whiten=int(getattr(args, 'sch_phi_whiten', 0)),
        sch_mixture_per_phi=int(getattr(args, 'sch_mixture_per_phi', 0)),
        sch_mixture_topk=int(getattr(args, 'sch_mixture_topk', 0)),
        sch_mixture_aux=float(getattr(args, 'sch_mixture_aux', 0.01)),
        sch_monarch_m1=int(getattr(args, 'sch_monarch_m1', 0)),
        sch_monarch_perm=str(getattr(args, 'sch_monarch_perm', 'none')),
        sch_monarch_perm_path=str(getattr(args, 'sch_monarch_perm_path', '')),
        sch_tier_bounds=str(getattr(args, 'sch_tier_bounds', '')),
        sch_tier_caps=str(getattr(args, 'sch_tier_caps', '')),
        sch_tier_order=str(getattr(args, 'sch_tier_order', 'freq')),
        sch_tier_perm_path=str(getattr(args, 'sch_tier_perm_path', '')),
        sch_proposal_rank=int(getattr(args, 'sch_proposal_rank', 32)),
        sch_proposal_topk=int(getattr(args, 'sch_proposal_topk', 4096)),
        sch_proposal_samples=int(getattr(args, 'sch_proposal_samples', 1024)),
        sch_proposal_warmup=int(getattr(args, 'sch_proposal_warmup', 0)),
        sch_proposal_chunk=int(getattr(args, 'sch_proposal_chunk', 128)),
        sch_proposal_aux=float(getattr(args, 'sch_proposal_aux', 1.0)),
        sch_nfh_mode=str(getattr(args, 'sch_nfh_mode', 'cp')),
        sch_nfh_rank=int(getattr(args, 'sch_nfh_rank', 32)),
        sch_nfh_dims=str(getattr(args, 'sch_nfh_dims', '')),
        sch_nfh_groups=int(getattr(args, 'sch_nfh_groups', 3)),
        sch_nfh_views=int(getattr(args, 'sch_nfh_views', 1)),
        sch_nfh_perm=str(getattr(args, 'sch_nfh_perm', 'none')),
        sch_nfh_perm_path=str(getattr(args, 'sch_nfh_perm_path', '')),
        sch_nfh_g_type=str(getattr(args, 'sch_nfh_g_type', 'linear')),
        sch_nfh_g_hidden=int(getattr(args, 'sch_nfh_g_hidden', 0)),
        sch_nfh_chunk=int(getattr(args, 'sch_nfh_chunk', 256)),
        sch_nfh_smooth=int(getattr(args, 'sch_nfh_smooth', 0)),
        sch_rerank_mode=str(getattr(args, 'sch_rerank_mode', 'topk')),
        sch_rerank_k=int(getattr(args, 'sch_rerank_k', 64)),
        sch_rerank_rank=int(getattr(args, 'sch_rerank_rank', 32)),
        # SAP: sampling-aware block head
        sap_block_T=int(getattr(args, 'sap_block_t', 0)),
        sap_block_mode=str(getattr(args, 'sap_block_mode', 'indep')),
        sap_block_frac=float(getattr(args, 'sap_block_frac', 0.125)),
        sap_lambda=float(getattr(args, 'sap_lambda', 1.0)),
        sap_head_layers=int(getattr(args, 'sap_head_layers', 2)),
        sap_head_mlp_mult=int(getattr(args, 'sap_head_mlp_mult', 2)),
        sap_enc_layers=int(getattr(args, 'sap_enc_layers', 1)),
        sap_ctx_window=int(getattr(args, 'sap_ctx_window', 16)),
        sap_latent_groups=int(getattr(args, 'sap_latent_groups', 4)),
        sap_latent_codes=int(getattr(args, 'sap_latent_codes', 16)),
        sap_latent_dim=int(getattr(args, 'sap_latent_dim', 64)),
        sap_free_bits=float(getattr(args, 'sap_free_bits', 0.25)),
        sap_kl_anneal_steps=int(getattr(args, 'sap_kl_anneal_steps', 2000)),
        sap_cp_components=int(getattr(args, 'sap_cp_components', 8)),
        sap_wta_k=int(getattr(args, 'sap_wta_k', 4)),
        sap_jacobi_sweeps=int(getattr(args, 'sap_jacobi_sweeps', 2)),

    )
    # Stash tokenizer_dir on config for lazy prior loading in EET
    config._tokenizer_dir = getattr(args, 'tokenizer_dir', None)

    with torch.device("meta"):
        if config.use_eet:
            from nanochat.eet import EarlyExitGPT
            model_meta = EarlyExitGPT(config)
        elif config.use_mol:
            from nanochat.mol import MoL
            model_meta = MoL(config)
        elif config.use_mst:
            from nanochat.mst import MST
            model_meta = MST(config)
        else:
            model_meta = GPT(config)
    return model_meta

# Build the model, move to device, init the weights
model = build_model_meta(args.depth) # 1) Build on meta device (only shapes/dtypes, no data)
model_config = model.config
model_config_kwargs = asdict(model_config)
print0(f"Model config:\n{json.dumps(model_config_kwargs, indent=2)}")
model.to_empty(device=device) # 2) All tensors get storage on target device but with uninitialized (garbage) data
# Seed immediately before init_weights so --seed controls weight initialisation.
# Offset by rank: every rank must draw *different* dropout/noise streams but the
# same weights, and init_weights runs identically everywhere, so seeding with the
# same base value here and letting rank offset apply afterwards keeps replicas
# consistent while giving each rank its own downstream stream.
if args.seed >= 0:
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    print0(f"Seeded weight init with --seed {args.seed}")
model.init_weights() # 3) All tensors get initialized
# 3b) Swap in the binary layers. AFTER init_weights because binarise_model_ copies
# the donor weights in as latent values (and repairs zero-initialised rows, which
# nanochat produces for every c_proj and which a detached mean|W| scale would
# otherwise freeze forever). BEFORE setup_optimizer so the new scale parameters are
# seen by the param-group sort, and before the DDP wrap.
if getattr(model_config, "use_binary", False):
    from nanochat.binary import binarise_model_, nativise_model_
    _skip = tuple(x for x in model_config.binary_skip.split(",") if x)
    if getattr(model_config, "binary_native", False):
        _nv = nativise_model_(model, model_config,
                              tau=model_config.binary_tau,
                              hard=model_config.binary_hard,
                              resid_width=model_config.binary_resid_width)
        print0(f"BINARY-NATIVE: replaced {len(_nv)} Blocks with BinaryBlock "
               f"(Hamming attention, KV-memory FFN, bundled residual, no norm), "
               f"tau={model_config.binary_tau}, resid_width={model_config.binary_resid_width}")
    _sw = binarise_model_(
        model,
        binarise_acts=model_config.binary_acts,
        linear=model_config.binary_linear,
        embeddings=model_config.binary_embeddings,
        skip=_skip,
        weight_scale=model_config.binary_weight_scale,
        act_scale=model_config.binary_act_scale,
        clip=model_config.binary_clip,
    )
    print0(f"BINARY: swapped {len(_sw)} modules "
           f"({sum(1 for _, k in _sw if k == 'Linear')} Linear, "
           f"{sum(1 for _, k in _sw if k == 'Embedding')} Embedding), "
           f"acts={'1b' if model_config.binary_acts else 'fp'}, "
           f"scale={model_config.binary_weight_scale}, skip={_skip or '()'}")
if args.seed >= 0:
    torch.manual_seed(args.seed + 1000 * ddp_rank)

# Phase 17: Auto-enable modulation diagnostics for any research model
# (always on for research, no need for --modulation-diagnostics flag)
mod_diag = None
diag_metrics = None  # Track last-collected metrics for wandb logging
if args.use_remix_linear:
    from nanochat.gpt import ModulationDiagnostics
    mod_diag = ModulationDiagnostics(model)
    if mod_diag._layers:
        print0(f"Modulation diagnostics enabled: tracking {len(mod_diag._layers)} conditioning layers")
    else:
        print0(f"Modulation diagnostics: no position-conditioned layers found (mode={args.cclblock_modulation})")
        mod_diag = None

# ── Gate stats collector ──────────────────────────────────────────────────────
def collect_gate_stats(model, step):
    """Walk all RemixedLinear / DualGateLinear layers and aggregate gate stats.

    Returns a dict with:
      basis_mean / basis_std / basis_dead / basis_sat   — sigmoid(gate_logits) stats
      out_mean   / out_std                              — 1+tanh(output_gate) stats
      cond_c_mean / cond_c_std                            ConditionedLinear coefficient stats
      cond_tok_std                                        how much c varies across tokens (0 = router collapsed)
      cond_dof_pr                                         participation ratio of the per-coefficient
                                                          variance spectrum: DOF/token actually in use
      gate_grad_norm   — L2 norm of all gate param gradients (0 if not yet computed)
      struct_grad_norm — L2 norm of all structural param gradients
    Each value is the mean across all tracked layers that have that stat.
    """
    from nanochat.gpt import RemixedLinear, DualGateLinear, ConditionedLinear
    accum = {}
    counts = {}
    layers_seen = 0

    raw = model.module if hasattr(model, 'module') else model
    for mod in raw.modules():
        if not isinstance(mod, (RemixedLinear, DualGateLinear, ConditionedLinear)):
            continue
        gs = getattr(mod, '_gate_stats', {})
        # Under torch.compile, _gate_stats is empty because its writes are
        # guarded by `not torch.compiler.is_compiling()`.  But the registered
        # buffers (_template_entropy_buf, _template_weights_buf) are always
        # populated via .copy_() inside the compiled graph, so fall back to
        # those to recover template routing diagnostics.
        if not gs:
            gs = {}
            _ent_buf = getattr(mod, '_template_entropy_buf', None)
            _tw_buf  = getattr(mod, '_template_weights_buf', None)
            if _ent_buf is not None and _ent_buf.numel() > 0 and _ent_buf.abs().sum().item() > 0:
                gs['template_entropy'] = _ent_buf.detach()
            if _tw_buf is not None and _tw_buf.numel() > 0 and _tw_buf.abs().sum().item() > 0:
                gs['template_weights'] = _tw_buf.detach()
                # Reconstruct template_weights_std from the per-template means
                # (the buffer holds the mean weight per template across the batch)
                gs['template_weights_std'] = _tw_buf.std().detach()
        if not gs:
            continue
        layers_seen += 1
        for k, v in gs.items():
            if isinstance(v, torch.Tensor):
                if v.numel() == 1:
                    v_val = v.item()
                    accum[k]  = accum.get(k, 0.0) + v_val
                    counts[k] = counts.get(k, 0) + 1
                else:
                    v_list = [x.item() for x in v]
                    if k not in accum:
                        accum[k] = [0.0] * len(v_list)
                        counts[k] = [0] * len(v_list)
                    # Extend accumulator if this layer has more elements
                    while len(accum[k]) < len(v_list):
                        accum[k].append(0.0)
                        counts[k].append(0)
                    for i_k, val_k in enumerate(v_list):
                        accum[k][i_k] += val_k
                        counts[k][i_k] += 1
            elif isinstance(v, (list, tuple)):
                if k not in accum:
                    accum[k] = [0.0] * len(v)
                    counts[k] = [0] * len(v)
                while len(accum[k]) < len(v):
                    accum[k].append(0.0)
                    counts[k].append(0)
                for i_k, val_k in enumerate(v):
                    accum[k][i_k] += val_k
                    counts[k][i_k] += 1
            elif isinstance(v, (int, float)):
                accum[k]  = accum.get(k, 0.0) + float(v)
                counts[k] = counts.get(k, 0) + 1

    result = {'step': step, 'layers': layers_seen}
    for k in accum:
        if isinstance(accum[k], list):
            result[k] = [x / max(c, 1) for x, c in zip(accum[k], counts[k])]
        else:
            result[k] = accum[k] / counts[k]

    # Gradient norms (only meaningful after loss.backward())
    gate_sq, gate_n, struct_sq, struct_n = 0.0, 0, 0.0, 0
    for mod in raw.modules():
        if not isinstance(mod, (RemixedLinear, DualGateLinear, ConditionedLinear)):
            continue
        for p in mod.gate_parameters():
            if p.grad is not None:
                gate_sq += p.grad.float().norm().item() ** 2
                gate_n  += 1
        for p in mod.non_gate_parameters():
            if p.grad is not None:
                struct_sq += p.grad.float().norm().item() ** 2
                struct_n  += 1
    result['gate_grad_norm']   = (gate_sq   ** 0.5) if gate_n   > 0 else 0.0
    result['struct_grad_norm'] = (struct_sq ** 0.5) if struct_n > 0 else 0.0
    return result

# If we are resuming, overwrite the model parameters with those of the checkpoint
output_dirname = args.model_tag if args.model_tag else f"d{args.depth}" # e.g. d12
if args.checkpoints_dir:
    checkpoints_root = os.path.abspath(args.checkpoints_dir)
else:
    # default to a "base_checkpoints" folder inside the nanochat base directory
    checkpoints_root = os.path.join(get_base_dir(), "base_checkpoints")

checkpoint_dir = os.path.abspath(os.path.join(checkpoints_root, output_dirname))
print0(f"Checkpoints directory: {checkpoint_dir}")
# Gate stats log (only written by master process when --use-remix-linear and gate-stats-every > 0)
gate_stats_log = os.path.join(checkpoint_dir, "gate_stats.log") if (args.use_remix_linear and getattr(args, 'gate_stats_every', 0) > 0) else None
if gate_stats_log and master_process:
    os.makedirs(checkpoint_dir, exist_ok=True)
    with open(gate_stats_log, 'w') as _f:
        _f.write('# gate_stats.log — one JSON line per logged step\n')
        _f.write('# Keys: step, layers, basis_mean, basis_std, basis_dead, basis_sat, out_mean, out_std, gate_grad_norm, struct_grad_norm\n')
if args.step_loss_file and master_process:
    step_loss_dir = os.path.dirname(os.path.abspath(args.step_loss_file))
    if step_loss_dir:
        os.makedirs(step_loss_dir, exist_ok=True)
    # Fresh file per run.
    with open(args.step_loss_file, "w", encoding="utf-8"):
        pass
eet_ever_routed = False
resuming = args.resume_from_step != -1
if resuming:
    print0(f"Resuming optimization from step {args.resume_from_step}")
    model_data, optimizer_data, meta_data = load_checkpoint(checkpoint_dir, args.resume_from_step, device, load_optimizer=True, rank=ddp_rank)
    model.load_state_dict(model_data, strict=True, assign=True)
    del model_data # free up this memory after the copy
    eet_ever_routed = meta_data.get("eet_ever_routed", False)

# -----------------------------------------------------------------------------
# Inductor can hand a fused pointwise kernel a block size larger than its own
# documented maximum and then assert on it. Clamping is a no-op unless that
# happens; see nanochat/inductor_compat.py, including why it needs
# TORCHINDUCTOR_COMPILE_THREADS=1 (or a sitecustomize) to reach compile workers.
guard_triton_block_size()

# FP8 training initialization and management (this has to be done before torch.compile)

# Convert Linear layers to Float8Linear if --fp8 is set
if args.fp8:
    if device_type != "cuda":
        print0("Warning: FP8 training requires CUDA, ignoring --fp8 flag")
        args.fp8 = False
    else:
        # Check compute capability (requires 8.9+ for L4/4090 or 9.0+ for H100)
        major, minor = torch.cuda.get_device_capability()
        if major < 8 or (major == 8 and minor < 9):
            print0(f"Warning: FP8 training requires compute capability >= 8.9 (e.g. H100, L4, 4090), but detected {major}.{minor}. Disabling FP8.")
            args.fp8 = False

if args.fp8:
    # our custom fp8 is simpler than torchao, written for exact API compatibility
    from nanochat.fp8 import Float8LinearConfig, convert_to_float8_training
    # from torchao.float8 import Float8LinearConfig, convert_to_float8_training
    import torch.nn as nn

    # SCH: the code head runs its matmuls through F.linear with an explicit weight
    # cast rather than through the child module's forward, so swapping those
    # nn.Linear children for Float8Linear would report a conversion that never
    # executes. Its dominant cost is the frozen Phi product, which is not an
    # nn.Linear and cannot be converted at all. Leave the head out of the count.
    _sch_head_active = bool(int(getattr(args, 'use_code_head', 0)))

    # Filter: dims must be divisible by 16 (FP8 hardware requirement) large enough
    def fp8_module_filter(mod: nn.Module, fqn: str) -> bool:
        if not isinstance(mod, nn.Linear):
            return False
        if _sch_head_active and (fqn.startswith('lm_head') or fqn.startswith('transformer.wte')):
            return False
        if mod.in_features % 16 != 0 or mod.out_features % 16 != 0:
            return False
        if min(mod.in_features, mod.out_features) < 128:
            return False
        return True

    fp8_config = Float8LinearConfig.from_recipe_name(args.fp8_recipe)
    num_linear = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
    convert_to_float8_training(model, config=fp8_config, module_filter_fn=fp8_module_filter)
    num_fp8 = sum(1 for m in model.modules() if 'Float8' in type(m).__name__)
    num_skipped = num_linear - num_fp8
    print0(f"✓ FP8 training enabled ({args.fp8_recipe} scaling) - converted {num_fp8}/{num_linear} linear layers, skipped {num_skipped} (too small)")

# Context manager to temporarily disable FP8 so that model evaluation remains in BF16
@contextmanager
def disable_fp8(model):
    """Temporarily swap Float8Linear modules with nn.Linear for BF16 evaluation.

    CastConfig is a frozen dataclass, so we can't mutate scaling_type. Instead,
    we swap out Float8Linear modules entirely and restore them after.
    """
    import torch.nn as nn

    # Find all Float8Linear modules and their locations
    fp8_locations = []  # list of (parent_module, attr_name, fp8_module)
    for name, module in model.named_modules():
        if 'Float8' in type(module).__name__:
            if '.' in name:
                parent_name, attr_name = name.rsplit('.', 1)
                parent = model.get_submodule(parent_name)
            else:
                parent = model
                attr_name = name
            fp8_locations.append((parent, attr_name, module))

    if not fp8_locations:
        yield  # No FP8 modules, nothing to do
        return

    # Swap Float8Linear -> Linear (our custom class that casts weights to match input dtype)
    for parent, attr_name, fp8_module in fp8_locations:
        linear = Linear(
            fp8_module.in_features,
            fp8_module.out_features,
            bias=fp8_module.bias is not None,
            device=fp8_module.weight.device,
            dtype=fp8_module.weight.dtype,
        )
        linear.weight = fp8_module.weight  # share, don't copy
        if fp8_module.bias is not None:
            linear.bias = fp8_module.bias
        setattr(parent, attr_name, linear)

    try:
        yield
    finally:
        # Restore Float8Linear modules
        for parent, attr_name, fp8_module in fp8_locations:
            setattr(parent, attr_name, fp8_module)

# Disable requires_grad for EET parameters during Phase 1 warmup
# to avoid DDP find_unused_parameters overhead and match dense speed perfectly.
if model_config.use_eet:
    for param in model.eet_routers.parameters():
        param.requires_grad = False
    for param in model.eet_translators.parameters():
        param.requires_grad = False

orig_model = model # original, uncompiled model, for saving raw model state_dict and for inference/evaluation (because the shapes may change shape)
model = wrap_model(model, parallel_type=args.parallel, compile=args.compile, device=device,
                   compile_regional=bool(getattr(args, "compile_regional", 0)),
                   compile_mode=getattr(args, "compile_mode", "default"))

# -----------------------------------------------------------------------------
# Scaling laws and muP extrapolations to determine the optimal training horizon, batch size, learning rates, weight decay.

# Get the parameter counts of our model
param_counts = orig_model.num_scaling_params()
print0(f"Parameter counts:")
for key, value in param_counts.items():
    print0(f"{key:24s}: {value:,}")
num_params = param_counts['total']
num_flops_per_token, num_active_flops_per_token, num_active_params = orig_model.estimate_flops()
print0(f"Estimated FLOPs per token (total):  {num_flops_per_token:e}")
# The three axes FLOPs cannot see. A W1A1 model has IDENTICAL FLOPs/token to its
# dense twin, so the line above is blind to the entire claim; it is also blind to
# wte/wpe/value_embeds, which are 80% of the parameters at depth 8 V=32,768.
# Emitted on every run so Phase 4's Pareto figure needs no re-runs.
try:
    from nanochat.bitcost import (cost_report, DENSE_BF16, FULLY_BINARY,
                                  NANOCHAT_MUONADAMW, PrecisionSpec)
    _counts = orig_model.num_scaling_params()
    if getattr(model_config, "use_binary", False):
        _wb = 1
        _ab = 1 if model_config.binary_acts else 16
        _eb = 1 if model_config.binary_embeddings else 16
        _hb = 16 if "lm_head" in model_config.binary_skip else 1
        _prec = PrecisionSpec(w_bits=_wb, a_bits=_ab, embed_bits=_eb, head_bits=_hb)
    else:
        _prec = DENSE_BF16
    _rep = cost_report(_counts, num_flops_per_token, _prec, NANOCHAT_MUONADAMW,
                       seq_len=args.max_seq_len)
    print0(f"COST_AXES precision={_prec.name} "
           f"bops_arith={_rep.bops_arithmetic:.4e} "
           f"energy_pj_per_token={_rep.energy_pj_per_token:.4e} "
           f"inference_MiB={_rep.total_inference_bytes/2**20:.2f} "
           f"training_state_MiB={_rep.total_state_bytes/2**20:.2f}")
except Exception as _e:
    print0(f"COST_AXES unavailable ({type(_e).__name__}: {_e})")
print0(f"Estimated FLOPs per token (active): {num_active_flops_per_token:e}")
print0(f"Estimated active params:            {num_active_params:,}")


# 1) Use scaling laws to determine the optimal training horizon in tokens
# The compute-optimal models satisfy the Tokens:Params ratio of --target-param-data-ratio (derived experimentally via scaling laws analysis).
# We've already initialized the model so we have Params. Optimal Tokens is now simply target-param-data-ratio * Params
def get_scaling_params(m):
    # As for which params to use exactly, transformer matrices + lm_head gives cleanest scaling laws (see dev/LOG.md Jan 27, 2026)
    params_counts = m.num_scaling_params()
    scaling_params = params_counts['transformer_matrices'] + params_counts['lm_head']
    return scaling_params
num_scaling_params = get_scaling_params(orig_model)
if args.target_tokens > 0:
    target_tokens = args.target_tokens
else:
    # When --target-active-params=1, adjust for inactive template/expert params so sparse models
    # receive a Chinchilla-optimal token budget proportional to their *active* parameter count.
    if getattr(args, 'target_active_params', 0) and num_active_params < num_params:
        inactive_params = num_params - num_active_params
        active_scaling_params = max(1, num_scaling_params - inactive_params)
        print0(f"Active scaling params: {active_scaling_params:,}  (total: {num_scaling_params:,}, inactive: {inactive_params:,})")
    else:
        active_scaling_params = num_scaling_params
    target_tokens = int(args.target_param_data_ratio * active_scaling_params)

# Our reference model is d12, this is where a lot of hyperparameters are tuned and then transfered to higher depths (muP style)
d12_ref = build_model_meta(12, apply_dim_override=False) # the FIXED reference anchor;
# --model-dim must NOT reach it. It did until b03: widening the arm to d=3584 also widened
# the d12 reference, inflating D_REF ~21x, shrinking the auto batch size from 2**19 to 2**17,
# and killing the run at the grad-accum assert with DEVICE_BATCH_SIZE=128.
D_REF = args.target_param_data_ratio * get_scaling_params(d12_ref) # compute-optimal d12 training horizon in tokens (measured empirically)
B_REF = 2**19 # optimal batch size at d12 ~= 524,288 tokens (measured empirically)

# 2) Now that we have the token horizon, we can calculate the optimal batch size
# We follow the Power Lines paper (Bopt ∝ D^0.383), ref: https://arxiv.org/abs/2505.13738
# The optimal batch size grows as approximately D^0.383, so e.g. if D doubles from d12 to d24, B should grow by 2^0.383 ≈ 1.3x.
total_batch_size = args.total_batch_size # user-provided override is possible
if total_batch_size == -1:
    # D_REF is derived from --target-param-data-ratio UNCONDITIONALLY, even when the
    # budget is pinned by --target-tokens (which wins above). A negative ratio, used
    # by some sweeps to mean "disable per-arm Chinchilla", makes D_REF negative, so
    # batch_size_ratio is negative and `ratio ** 0.383` returns a COMPLEX number that
    # dies three lines later inside math.log2 as "must be real number, not complex".
    # Diagnose it here instead of there.
    if D_REF <= 0 or target_tokens <= 0:
        raise SystemExit(
            f"Cannot auto-compute the batch size: D_REF={D_REF:,.0f}, "
            f"target_tokens={target_tokens:,}.\n"
            f"  --target-param-data-ratio is {args.target_param_data_ratio}, and D_REF is "
            f"that times the d12 scaling params, so a negative ratio poisons the batch-size\n"
            f"  derivation even though --target-tokens correctly overrides the BUDGET.\n"
            f"  To pin the budget, pass --target-tokens N and leave the ratio POSITIVE "
            f"(10.5), as scripts/p13_isodata.sh does. Or set --total-batch-size explicitly.")
    batch_size_ratio = target_tokens / D_REF
    predicted_batch_size = B_REF * batch_size_ratio ** 0.383
    total_batch_size = 2 ** round(math.log2(predicted_batch_size)) # clamp to nearest power of 2 for efficiency
    print0(f"Auto-computed optimal batch size: {total_batch_size:,} tokens")

# 3) Knowing the batch size, we can now calculate a learning rate correction (bigger batch size allows higher learning rates)
batch_lr_scale = 1.0
batch_ratio = total_batch_size / B_REF # B/B_ref
if batch_ratio != 1.0:
    # SGD: linear scaling with batch size is standard (not used in nanochat)
    # AdamW: sqrt scaling is standard: η ∝ √(B/B_ref)
    # Muon: we will use the same scaling for Muon as for AdamW: η ∝ √(B/B_ref) (not studied carefully, assumption!)
    batch_lr_scale = batch_ratio ** 0.5 # η ∝ √(B/B_ref)
    print0(f"Scaling LRs by {batch_lr_scale:.4f} for batch size {total_batch_size:,} (reference: {B_REF:,})")

# 4) Knowing the batch size and the token horizon, we can now calculate the appropriate weight decay scaling
# We adopt the T_epoch framework from https://arxiv.org/abs/2405.13698
# Central idea of the paper is that T_epoch = B/(η·λ·D) should remain constant.
# Above, we used learning rate scaling η ∝ √(B/B_ref). So it's a matter of ~10 lines of math to derive that to keep T_epoch constant, we need:
# λ = λ_ref · √(B/B_ref) · (D_ref/D)
# Note that these papers study AdamW, *not* Muon. We are blindly following AdamW theory for scaling hoping it ~works for Muon too.
weight_decay_scaled = args.weight_decay * math.sqrt(total_batch_size / B_REF) * (D_REF / target_tokens)
if weight_decay_scaled != args.weight_decay:
    print0(f"Scaling weight decay from {args.weight_decay:.6f} to {weight_decay_scaled:.6f} for depth {args.depth}")

# Phase 20 (E/G/J): Convert standard MLP to Phase 2 variant if flags are set
# This must happen AFTER weights are initialized/loaded but BEFORE optimizer setup
_p20_pwu = getattr(args, 'p20_pwu_branches', 0)
_p20_fsvd = getattr(args, 'p20_fsvd_gate', 0)
_p20_wbfc = getattr(args, 'p20_wbfc_clusters', 0)
if _p20_pwu > 0 or _p20_fsvd > 0 or _p20_wbfc > 0:
    n_converted = orig_model.convert_to_phase2()
    print0(f"Phase 2 conversion complete: {n_converted} MLP modules converted")

# -----------------------------------------------------------------------------
# Initialize the Optimizer (combined MuonAdamW: Muon for matrix params, AdamW for rest)
optimizer = orig_model.setup_optimizer(
    # AdamW hyperparameters
    unembedding_lr=args.unembedding_lr * batch_lr_scale,
    embedding_lr=args.embedding_lr * batch_lr_scale,
    scalar_lr=args.scalar_lr * batch_lr_scale,
    adam_betas=(args.adam_beta1, args.adam_beta2),
    # Muon hyperparameters
    matrix_lr=args.matrix_lr * batch_lr_scale,
    weight_decay=weight_decay_scaled,
    # μP
    disable_mu_p=args.disable_mu_p,
    mu_p_scale_override=args.mu_p_scale_override,
    # Gate LR scale (default 0.3×; lower values slow gate learning relative to structural weights)
    gate_lr_scale=args.remix_gate_lr_scale,
)

if resuming:
    optimizer.load_state_dict(optimizer_data)
    del optimizer_data

# -----------------------------------------------------------------------------
# GradScaler for fp16 training (bf16/fp32 don't need it — bf16 has the same exponent range as fp32)
scaler = torch.amp.GradScaler() if COMPUTE_DTYPE == torch.float16 else None
if scaler is not None:
    print0("GradScaler enabled for fp16 training")

# -----------------------------------------------------------------------------
# Initialize the DataLoaders for train/val
dataloader_resume_state_dict = None if not resuming else meta_data["dataloader_state_dict"]
train_loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(
    tokenizer,
    args.device_batch_size * (ddp_world_size if is_dp else 1),
    args.max_seq_len,
    split="train",
    device=device,
    resume_state_dict=dataloader_resume_state_dict,
    data_dir=args.data_dir,
    max_shards=args.max_shards,
)
build_val_loader = lambda: tokenizing_distributed_data_loader_bos_bestfit(
    tokenizer,
    args.device_batch_size * (ddp_world_size if is_dp else 1),
    args.max_seq_len,
    split="val",
    device=device,
    data_dir=args.data_dir,
    max_shards=args.max_shards,
)
# ── SCH: held-out vocabulary ─────────────────────────────────────────────────
# The flagship claim is zero-shot vocabulary extension: add a token after
# training, compute its code from its semantics, and the model assigns it a
# calibrated probability immediately, because the per-bit and per-monomial
# parameters are already trained.  A softmax must fit a fresh row from scratch.
#
# To measure that we remove N token ids from TRAINING and score them at
# validation time.  Two modes, and the difference matters in the write-up:
#   target  the ids are masked as prediction targets only.  The head therefore
#           gets no gradient for producing them, which is exactly the claim
#           under test, while their input embeddings still train.
#   full    the ids are also rewritten in the inputs, so the model never sees
#           them at all.  A stronger control, but it perturbs the context
#           distribution, so quote both if you use it.
sch_holdout_ids = None
sch_holdout_mask = None
if args.sch_holdout_tokens > 0:
    _g = torch.Generator().manual_seed(args.sch_holdout_seed)
    _lo = min(args.sch_holdout_min_id, vocab_size - 1)
    _pool = torch.arange(_lo, vocab_size)
    assert args.sch_holdout_tokens < _pool.numel(), (
        f"--sch-holdout-tokens {args.sch_holdout_tokens} exceeds the {_pool.numel()} "
        f"eligible ids in [{_lo}, {vocab_size})")
    sch_holdout_ids = _pool[torch.randperm(_pool.numel(), generator=_g)[:args.sch_holdout_tokens]].sort().values
    sch_holdout_mask = torch.zeros(vocab_size, dtype=torch.bool, device=device)
    sch_holdout_mask[sch_holdout_ids.to(device)] = True
    print0(f"[SCH] holding out {args.sch_holdout_tokens:,} of {vocab_size:,} token ids "
           f"({100.0 * args.sch_holdout_tokens / vocab_size:.2f}%) from training, "
           f"mode={args.sch_holdout_mode}, seed={args.sch_holdout_seed}")
    if master_process:
        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save(sch_holdout_ids, os.path.join(checkpoint_dir, 'sch_holdout_ids.pt'))

def sch_apply_holdout(x, y):
    """Strip held-out ids from a TRAINING batch.  Validation batches are left
    untouched: their perplexity on those ids is the metric."""
    if sch_holdout_mask is None:
        return x, y
    y = torch.where(sch_holdout_mask[y.clamp_min(0)] & (y >= 0), torch.full_like(y, -1), y)
    if args.sch_holdout_mode == 'full':
        x = torch.where(sch_holdout_mask[x], torch.zeros_like(x), x)
    return x, y

x, y, dataloader_state_dict = next(train_loader) # kick off load of the very first batch of data
x, y = sch_apply_holdout(x, y)

# -----------------------------------------------------------------------------
# Calculate the number of iterations we will train for and set up the various schedulers

# num_iterations: either it is given, or from target flops, or from target data:param ratio (in that order)
assert args.num_iterations > 0 or args.target_param_data_ratio > 0 or args.target_flops > 0 or args.target_active_flops > 0
if args.num_iterations > 0:
    # Override num_iterations to a specific value if given
    num_iterations = args.num_iterations
    print0(f"Using user-provided number of iterations: {num_iterations:,}")
elif args.target_flops > 0:
    # Calculate the number of iterations from the target flops (used in scaling laws analysis, e.g. runs/scaling_laws.sh)
    num_iterations = round(args.target_flops / (num_flops_per_token * total_batch_size))
    print0(f"Calculated number of iterations from target FLOPs: {num_iterations:,}")
elif args.target_active_flops > 0:
    # Same, but on the ACTIVE axis. An isoFLOP profile that mixes a gated model with a
    # dense one has to put both arms on the same contour, and the contour the Pareto
    # plots use is active FLOPs/token x tokens. Using the total count instead would place
    # MST on a 1.34x tighter budget than dense at L=24 and silently bias the profile.
    num_iterations = round(args.target_active_flops / (num_active_flops_per_token * total_batch_size))
    print0(f"Calculated number of iterations from target ACTIVE FLOPs: {num_iterations:,}")
elif args.target_param_data_ratio > 0:
    # Calculate the number of iterations from the target param data ratio (the most common use case)
    num_iterations = target_tokens // total_batch_size
    print0(f"Calculated number of iterations from target data:param ratio: {num_iterations:,}")
else:
    raise ValueError("No training horizon specified")
total_tokens = total_batch_size * num_iterations # the actual number of tokens we will train for
print0(f"Total number of training tokens: {total_tokens:,}")
if args.timing_probe_steps > 0:
    # NOTE: do NOT force log_every=1 here. _mst_diag_every is tied to args.log_every
    # (see the [MST] diagnostics block below), so setting it to 1 makes MST run per-layer
    # stream diagnostics on EVERY step while dense, which has no tracker, runs none. That
    # is an MST-only cost that does not exist in real training at --log-every 200, and it
    # inflated MST's dt by roughly a quarter in the first timing probes. The probe reads
    # dt from TIMING_PROBE_RESULT below, which is computed from total_training_time and
    # is emitted regardless of the logging interval, so no override is needed.
    print0(f"TIMING_PROBE full_iterations={num_iterations} probe_steps={args.timing_probe_steps} "
           f"total_batch_size={total_batch_size}")
print0(f"Tokens : Scaling params ratio: {total_batch_size * num_iterations / num_scaling_params:.2f}") # e.g. Chinchilla was ~20
print0(f"Total training FLOPs estimate: {num_flops_per_token * total_tokens:e}")
print0(f"Total training FLOPs (active):  {num_active_flops_per_token * total_tokens:e}")
print0(f"Total active training FLOPs estimate: {num_active_flops_per_token * total_tokens:e}")

# Research branches use a OneCycle-style schedule; base keeps the original warmup/flat/warmdown schedule
use_research_mode = args.use_moe or args.use_perm or args.use_remix_linear
use_research_scheduler = use_research_mode and bool(args.research_onecycle)
if use_research_scheduler:
    print0("Using research scheduler: OneCycle-style LR multiplier")
elif use_research_mode:
    print0("Research mode with OneCycle disabled: using base warmup/flat/warmdown schedule")
else:
    print0("Using base scheduler: linear warmup/flat/linear warmdown")

# Learning rate schedule (linear warmup, constant, linear warmdown)
def get_lr_multiplier(it):
    warmup_iters = round(args.warmup_ratio * num_iterations)
    warmdown_iters = round(args.warmdown_ratio * num_iterations)
    if it < warmup_iters:
        return (it + 1) / warmup_iters
    elif it <= num_iterations - warmdown_iters:
        return 1.0
    else:
        progress = (num_iterations - it) / warmdown_iters
        return progress * 1.0 + (1 - progress) * args.final_lr_frac


def get_lr_multiplier_onecycle(it):
    """
    OneCycle-style multiplier in [final_lr_frac, 1.0].
    - rise phase: cosine from final_lr_frac -> 1.0
    - decay phase: cosine from 1.0 -> final_lr_frac
    Uses warmup_ratio as pct_start for the peak.
    """
    if num_iterations <= 1:
        return 1.0
    t = min(max(it, 0), num_iterations - 1)
    pct = t / (num_iterations - 1)
    warmup_src = args.warmup_ratio if args.research_warmup_ratio < 0 else args.research_warmup_ratio
    pct_start = min(max(warmup_src, 0.01), 0.99)
    low = args.final_lr_frac
    if pct <= pct_start:
        phase = pct / pct_start
        return low + (1.0 - low) * (1 - math.cos(math.pi * phase)) * 0.5
    phase = (pct - pct_start) / (1 - pct_start)
    return low + (1.0 - low) * (1 + math.cos(math.pi * phase)) * 0.5

# Momentum scheduler for Muon optimizer (warms up to 0.95 over the first 300 steps)
def get_muon_momentum(it):
    frac = min(it / 300, 1)
    momentum = (1 - frac) * 0.85 + frac * 0.95
    return momentum

# Weight decay scheduler for Muon optimizer (linearly decays to zero over the course of training)
def get_weight_decay(it):
    return weight_decay_scaled * (1 - it / num_iterations)

# Fix 1H: PermutationMoE temperature scheduler
# Exponentially decays from perm_temp_start -> 1.0 over the first 50% of training
# Prevents early routing collapse from logit saturation and hard-argmax-like softmax behavior
def get_perm_temperature(it):
    t_start = args.perm_temp_start
    t_end = 1.0
    frac = min(it / max(num_iterations * 0.5, 1), 1.0)
    return t_start * (t_end / t_start) ** frac

# -----------------------------------------------------------------------------
# Training loop

# Loop state (variables updated by the training loop)
mfu = 0.0  # updated each training step; initialized here so the post-loop report always has a value
if not resuming:
    step = 0
    val_bpb = None
    min_val_bpb = float("inf")
    min_val_loss = float("inf")
    smooth_train_loss = 0 # EMA of training loss
    total_training_time = 0 # total wall-clock time of training
    last_periodic_ckpt_step = -1 # step of the last periodic (non-final) checkpoint saved
else:
    step = meta_data["step"]
    loop_state = meta_data["loop_state"]
    val_bpb = meta_data["val_bpb"]
    min_val_bpb = loop_state["min_val_bpb"]
    min_val_loss = loop_state.get("min_val_loss", float("inf"))
    smooth_train_loss = loop_state["smooth_train_loss"]
    total_training_time = loop_state["total_training_time"]
    # On resume, the checkpoint we resumed from is the last known periodic save
    last_periodic_ckpt_step = step

# ── MST Experiment Tracker ────────────────────────────────────────────────────
if model_config.use_mst and master_process:
    class _MSTTracker:
        """Accumulates per-step MST router diagnostics and val_loss milestones,
        then writes a single summary CSV row at end of training."""
        def __init__(self, config, run_dir):
            self.config = config
            self.run_dir = run_dir
            self.entropy_vals, self.balance_vals, self.grad_norm_vals = [], [], []
            self.val_at = {}  # step -> bpb

        def collect(self, model, grad_norm=None):
            """Call after each optimizer step to sample router diagnostics."""
            from nanochat.mst import MSTRouter
            for m in model.modules():
                if isinstance(m, MSTRouter):
                    if m._last_entropy is not None:
                        self.entropy_vals.append(float(m._last_entropy))
                    if m._last_balance is not None:
                        self.balance_vals.append(float(m._last_balance))
            if grad_norm is not None:
                self.grad_norm_vals.append(float(grad_norm))

        def record_val(self, step, bpb):
            self.val_at[step] = bpb

        def write_csv(self, step, val_bpb, train_loss_final, total_training_time,
                      num_flops_per_token, num_active_flops_per_token, num_active_params,
                      total_batch_size, num_params, sp):
            import csv, os, statistics
            c = self.config
            row = {
                'config_id':           getattr(args, 'model_tag', '') or f'd{args.depth}',
                'input_mode':          c.mst_input_mode,
                'routing_mode':        c.mst_routing_mode,
                'ffn_mode':            c.mst_ffn_mode,
                'ffn_every':           c.mst_ffn_every,
                'transition_mode':     c.mst_transition_mode,
                'final_mode':          c.mst_final_mode,
                'n_subs':              c.mst_n_subs,
                'sub_dim':             c.mst_sub_dim,
                'head_dim':            c.mst_head_dim if c.mst_head_dim > 0 else c.mst_sub_dim // c.n_head,
                'val_loss_1k':         self.val_at.get(1000, ''),
                'val_loss_5k':         self.val_at.get(5000, ''),
                'val_loss_final':      val_bpb if val_bpb is not None else '',
                'train_loss_final':    f'{train_loss_final:.6f}' if train_loss_final is not None else '',
                'router_entropy_mean': f'{statistics.mean(self.entropy_vals):.4f}' if self.entropy_vals else '',
                'router_entropy_min':  f'{min(self.entropy_vals):.4f}'              if self.entropy_vals else '',
                'load_balance_score':  f'{statistics.mean(self.balance_vals):.4f}'  if self.balance_vals else '',
                'grad_norm_mean':      f'{statistics.mean(self.grad_norm_vals):.4f}' if self.grad_norm_vals else '',
                'wall_time_min':       f'{total_training_time / 60:.2f}',
                'flops_per_token':     f'{num_flops_per_token:.4e}',
                'active_flops_per_token': f'{num_active_flops_per_token:.4e}',
                'total_flops':         f'{num_flops_per_token * total_batch_size * step:.4e}',
                'active_total_flops':  f'{num_active_flops_per_token * total_batch_size * step:.4e}',
                'params_full':         num_params,
                'active_params':       num_active_params,
                'params_transformer':  sp.get('transformer_matrices', '') if isinstance(sp, dict) else '',
                'params_value_embeds': sp.get('value_embeds', '')         if isinstance(sp, dict) else '',
                'params_total_sp':     sp.get('total', '')                if isinstance(sp, dict) else '',
                # P07 scaling improvement flags
                'grad_equalize':       c.mst_grad_equalize,
                'block_diagonal_muon': c.mst_block_diagonal_muon,
                'transition_width_mult': c.mst_transition_width_mult,
                'sub_lr_scale':        c.mst_sub_lr_scale,
                'shared_expert':       c.mst_shared_expert,
                'router_entropy_weight': c.mst_router_entropy_weight,
                'shared_kv_attn':      c.mst_shared_kv_attn,
                'contrastive_div_weight': c.mst_contrastive_diversity_weight,
                # Stage 8: Transition expressivity
                'transition_nonlinear': c.mst_transition_nonlinear,
                'transition_gated':     c.mst_transition_gated,
                'transition_mlp':       c.mst_transition_mlp,
                # Stage 9: Cross-sub expressivity
                'cross_sub_gate':       c.mst_cross_sub_gate,
                'hyper_connect':        c.mst_hyper_connect,
                'cross_kv_inject':      c.mst_cross_kv_inject,
                # Stage 10: Structural transition improvements
                'slice_transition':     c.mst_slice_transition,
                'lookback_layers':      c.mst_lookback_layers,
                'bilinear_transition':  c.mst_bilinear_transition,
                # Stage 11: Attention bottleneck + structural improvements
                'cross_sub_qmod':       c.mst_cross_sub_qmod,
                'feature_cycle':        c.mst_feature_cycle,
                'mean_transition':      c.mst_mean_transition,
                'global_residual':      c.mst_global_residual,
                # Stage 12: dense-parity fixes
                'sub_head_dim':         c.mst_sub_head_dim,
                'final_norm':           c.mst_final_norm,
                'per_stream_ve':        c.mst_per_stream_ve,
                # Stage 13: overhead cuts
                'lm_head_dim':          c.mst_lm_head_dim,
                'compose_windows':      c.mst_compose_windows,
                # Stage 14: free cross-stream mixing
                'channel_mix':          c.mst_channel_mix,
                'channel_mix_site':     c.mst_channel_mix_site,
                # Stage 15
                'transition_every':     c.mst_transition_every,
                'distribute_block_muon': c.mst_distribute_block_muon,
                'trans_spectral_lr':    c.mst_trans_spectral_lr,
                'talking_heads':        c.mst_talking_heads,
                'wo_mode':              c.mst_wo_mode,
                # Stage 16
                'stream_topk':          c.mst_stream_topk,
                'stream_router_aux':    c.mst_stream_router_aux,
                'stream_router_noise':  c.mst_stream_router_noise,
                'stream_dispatch':      c.mst_stream_dispatch,
                'stream_capacity_factor': c.mst_stream_capacity_factor,
                'stream_gate_attn':     c.mst_stream_gate_attn,
                # Stage 17
                'ffn_monarch':          c.mst_ffn_monarch,
                'shampoo':              c.mst_shampoo,
                'precond_every':        c.mst_precond_every,
            }
            # Write to checkpoint parent dir (original location)
            csv_path = os.path.normpath(os.path.join(self.run_dir, '..', 'mst_results.csv'))
            self._append_row(csv_path, row)
            # Also write to cwd so results survive regardless of checkpoint persistence
            cwd_csv = os.path.join(os.getcwd(), 'mst_results.csv')
            if os.path.abspath(cwd_csv) != os.path.abspath(csv_path):
                self._append_row(cwd_csv, row)

        @staticmethod
        def _append_row(csv_path, row):
            """Append one result row, never misaligning against an older header.

            NOTE: csv/os are imported here rather than relying on write_csv's local
            import, which is not visible from this scope.

            Each MST stage adds config columns, so a CSV written by an earlier stage
            has fewer fields than `row`. Appending under the new fieldnames would put
            extra values on every line while leaving the header short, silently
            skewing every column for whoever reads it later. Instead, divert to a
            numbered sibling file when the header does not match.
            """
            import csv, os
            fieldnames = list(row.keys())
            if os.path.exists(csv_path):
                with open(csv_path, newline='') as f:
                    existing = next(csv.reader(f), None)
                if existing != fieldnames:
                    stem, ext = os.path.splitext(csv_path)
                    n = 2
                    while os.path.exists(f"{stem}_v{n}{ext}"):
                        with open(f"{stem}_v{n}{ext}", newline='') as f:
                            if next(csv.reader(f), None) == fieldnames:
                                break
                        n += 1
                    print0(f"[MST] {os.path.basename(csv_path)} has an older column set; "
                           f"writing to {os.path.basename(stem)}_v{n}{ext} instead")
                    csv_path = f"{stem}_v{n}{ext}"
            write_header = not os.path.exists(csv_path)
            with open(csv_path, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if write_header:
                    writer.writeheader()
                writer.writerow(row)
            print0(f"[MST] Results appended → {csv_path}")

    _mst_tracker = _MSTTracker(model_config, checkpoint_dir)
    _mst_diag_log = os.path.join(checkpoint_dir, 'mst_diagnostics.jsonl')
    _mst_diag_every = args.log_every  # log diagnostics at same frequency as training logs
    print0(f"[MST] Diagnostics will be logged to: {_mst_diag_log}")
else:
    _mst_tracker = None
    _mst_diag_log = None
    _mst_diag_every = 0


# Figure out the needed gradient accumulation micro-steps to reach the desired total batch size per step
effective_device_batch_size = args.device_batch_size * (ddp_world_size if is_dp else 1)
tokens_per_fwdbwd = effective_device_batch_size * args.max_seq_len # tokens per iteration for a single rank
world_tokens_per_fwdbwd = tokens_per_fwdbwd * (1 if is_dp else ddp_world_size) # total tokens per iteration for all ranks
if total_batch_size % world_tokens_per_fwdbwd != 0:
    raise SystemExit(
        f"Gradient accumulation does not divide evenly.\n"
        f"  total_batch_size          = {total_batch_size:,} tokens"
        f"{' (auto-computed)' if args.total_batch_size == -1 else ' (--total-batch-size)'}\n"
        f"  world_tokens_per_fwdbwd   = {world_tokens_per_fwdbwd:,} tokens"
        f"  = device_batch_size {args.device_batch_size} x max_seq_len {args.max_seq_len}"
        f"{f' x world_size {ddp_world_size}' if not is_dp else ''}\n"
        + (f"  The micro-batch is LARGER than the whole batch. Lower --device-batch-size to "
           f"{max(1, total_batch_size // (args.max_seq_len * (1 if is_dp else ddp_world_size)))} "
           f"or below.\n" if world_tokens_per_fwdbwd > total_batch_size else
           f"  Pick a --device-batch-size that divides it, or pass --total-batch-size explicitly.\n")
        + f"  Both are normally powers of two, so a mismatch usually means one of them was "
          f"derived from the wrong model.")
grad_accum_steps = total_batch_size // world_tokens_per_fwdbwd
print0(f"Tokens / micro-batch / rank: {args.device_batch_size} x {args.max_seq_len} = {tokens_per_fwdbwd:,}")
print0(f"Tokens / micro-batch: {world_tokens_per_fwdbwd:,}")
print0(f"Total batch size {total_batch_size:,} => gradient accumulation steps: {grad_accum_steps}")
EMA_BETA = 0.9

# -----------------------------------------------------------------------------
# Pre-compilation warmup
# On the first real training step, three things overlap to create a memory spike
# that doesn't exist on subsequent steps:
#   1. torch.compile graph capture allocates temporary scratch buffers
#   2. Muon + AdamW m/v state tensors are lazily allocated on the first .step()
#   3. The CUDA caching allocator reserves large contiguous blocks on first use
# Together these can push peak VRAM 10–20 GB above the true steady-state cost,
# preventing you from sizing the batch to use the available headroom.
#
# Fix: run one throwaway forward + backward + step before real training.
# After this, all lazy allocations are done, compiler scratch has been released,
# and the allocator has defragmented. Step 0 of real training then looks like step 2.
#
# We use zero-filled tensors with the real batch shapes so the compiler traces
# with correct sizes but the data ordering is completely unaffected.
if not resuming and device_type == "cuda":
    print0("Running pre-compilation warmup (1 dummy forward+backward to init lazy allocations)...")
    torch.cuda.reset_peak_memory_stats()
    _wx = torch.zeros_like(x)
    _wy = torch.zeros_like(y)
    if model_config.use_eet:
        # Phase 1 delegates to super().forward() which ignores EET kwargs,
        # but we still need to pass eet_do_route=False so the guard matches.
        _wloss = model(_wx, _wy, eet_do_route=False)
    else:
        _wloss = model(_wx, _wy)
    if is_dp:
        _wloss = _wloss.mean()
    (_wloss / grad_accum_steps).backward()
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"]  # sensible default for the dummy step
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    del _wx, _wy, _wloss
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()  # reset so step-0 peak reflects real training only
    print0(f"Warmup complete. Steady-state VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB allocated")

# Go!
while True:
    last_step = step == num_iterations # normal end

    # Timing probe: end early, but only after num_iterations, the schedules and the batch
    # size have been derived from the real budget, so dt is the dt the full run would see.
    # Ending through last_step (rather than break) keeps the final eval and save, which is
    # exactly the tail latency the caller is trying to measure.
    if args.timing_probe_steps > 0 and step >= args.timing_probe_steps:
        last_step = True

    # early stop: force this to be the last step if we hit the token limit
    if args.early_stop_tokens > 0 and step * total_batch_size >= args.early_stop_tokens:
        print0(f"[early stop] Reached {step * total_batch_size:,} tokens (limit: {args.early_stop_tokens:,}). Initiating final eval/save.")
        last_step = True

    flops_so_far = num_flops_per_token * total_batch_size * step

    # once in a while: evaluate the val bpb (all ranks participate)
    do_eval = (args.eval_every > 0 and (last_step or step % args.eval_every == 0)) or (last_step and args.eval_every == -1)
    if do_eval:
        model.eval()
        val_loader = build_val_loader()
        eval_steps = args.eval_tokens // (args.device_batch_size * args.max_seq_len * ddp_world_size)
        with disable_fp8(model):
            eval_kwargs = {}
            if args.use_eet:
                if eet_ever_routed:
                    eval_kwargs['eet_do_route'] = True
                    eval_kwargs['eet_phase'] = 3
                else:
                    eval_kwargs['eet_do_route'] = False
                    eval_kwargs['eet_phase'] = 1
            val_bpb, val_loss = evaluate_bpb(model, val_loader, eval_steps, token_bytes, **eval_kwargs)
        print0(f"Step {step:05d} | Validation bpb: {val_bpb:.6f} | val_loss: {val_loss:.6f}")
        # SAP: the block head's bpb next to the trunk's next-token bpb on the same tokens.
        # Run on the uncompiled module: block shapes differ from the training shapes.
        if getattr(orig_model, 'sap_head', None) is not None and args.sap_eval_steps > 0:
            from nanochat.block_head import evaluate_block_bpb
            with disable_fp8(model):
                _sap_eval = evaluate_block_bpb(orig_model, build_val_loader(), args.sap_eval_steps, token_bytes)
            _bb = _sap_eval["block_bpb"]
            print0(f"Step {step:05d} | SAP block bpb: {'n/a' if _bb is None else f'{_bb:.6f}'} "
                   f"| next-token bpb on the same tokens: {_sap_eval['ntp_bpb_same_tokens']:.6f} "
                   f"| latent sensitivity: {_sap_eval['latent_sensitivity_nats']} nats")
            print0(f"SAP_EVAL_JSON {json.dumps({'step': step, **_sap_eval})}")
            wandb_run.log({"step": step, "total_training_flops": flops_so_far,
                           **{f"sap/{k}": v for k, v in _sap_eval.items() if v is not None}})
        # EET P02 T0B: per-layer vocabulary coverage of the routing policy
        if getattr(args, 'eet_coverage_diag', 0):
            _cov_model = model.module if hasattr(model, 'module') else model
            _cov = _cov_model.coverage_report() if hasattr(_cov_model, 'coverage_report') else None
            if _cov is not None:
                _cov_str = " ".join(
                    f"L{r['layer']}:{r['vocab_frac']*100:.1f}%/{r['mass_frac']*100:.1f}%" for r in _cov
                )
                print0(f"Step {step:05d} | EET coverage (vocab/mass reaching layer): {_cov_str}")
                print0(f"EET_COVERAGE_JSON {json.dumps(_cov)}")
                _cov_model.reset_coverage()
        if val_bpb < min_val_bpb:
            min_val_bpb = val_bpb
        if val_loss < min_val_loss:
            min_val_loss = val_loss
        if _mst_tracker is not None:
            _mst_tracker.record_val(step, val_bpb)
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "val/bpb": val_bpb,
        })
        model.train()

    # once in a while: estimate the CORE metric (all ranks participate)
    # use the original uncompiled model because the inputs keep changing shape
    # disable FP8 for evaluation to use BF16 for more consistent/accurate results
    results = {}
    if (args.core_metric_every != 0) and (last_step or (args.core_metric_every > 0 and step > 0 and step % args.core_metric_every == 0)):
        model.eval()
        with disable_fp8(orig_model):
            results = evaluate_core(orig_model, tokenizer, device, max_per_task=args.core_metric_max_per_task)
        print0(f"Step {step:05d} | CORE metric: {results['core_metric']:.4f}")
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "core_metric": results["core_metric"],
            "centered_results": results["centered_results"],
        })
        model.train()

    # once in a while: sample from the model (only on master process)
    # use the original uncompiled model because the inputs keep changing shape
    _can_sample = getattr(orig_model, 'supports_kv_cache_generation', True)
    # A tree head owns its own loss and never materialises a logit vector: that
    # is the entire point of O(d log V). Generation needs one, so it is not a
    # missing feature to be fixed but a property of the method, and the run must
    # not be thrown away at the last step because of it.
    _head_emits_logits = not getattr(getattr(orig_model, 'lm_head', None), 'custom_loss', False)
    _can_sample = _can_sample and _head_emits_logits
    if args.sample_every > 0 and master_process and not _can_sample and (last_step or (step > 0 and step % args.sample_every == 0)):
        print0("[sample] skipped: "
               + ("this head computes its loss without a logit vector, so it cannot generate"
                  if not _head_emits_logits
                  else "this architecture does not implement KV-cache generation"))
    if args.sample_every > 0 and master_process and _can_sample and (last_step or (step > 0 and step % args.sample_every == 0)):
        model.eval()
        prompts = [
            "The capital of France is",
            "The chemical symbol of gold is",
            "If yesterday was Friday, then tomorrow will be",
            "The opposite of hot is",
            "The planets of the solar system are:",
            "My favorite color is",
            "If 5*x + 3 = 13, then x is",
        ]
        engine = Engine(orig_model, tokenizer) # use orig_model to avoid recompilation
        for prompt in prompts:
            tokens = tokenizer(prompt, prepend="<|bos|>")
            with disable_fp8(orig_model):
                sample, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=16, temperature=0)
            print0(tokenizer.decode(sample[0]))
        model.train()

    # save checkpoint: at the end of the run, or every save_every steps, except at the first step or the resume step
    is_periodic_save = (step > 0 and step != args.resume_from_step and args.save_every > 0 and step % args.save_every == 0)
    if last_step or is_periodic_save:
        save_checkpoint(
            checkpoint_dir,
            step,
            orig_model.state_dict(), # model parameters
            # Optimizer state exists only so a preempted run can resume. The FINAL
            # checkpoint has nothing to resume into, and its optimizer shards are the
            # largest single artifact a run leaves behind: AdamW keeps exp_avg and
            # exp_avg_sq and Muon a momentum buffer, so they run 2-3x the size of the
            # weights themselves. Downstream consumers (CORE eval, paper_probe's stream
            # diagnostics, any inference benchmark) load model_*.pt and never touch them.
            (None if (last_step and not args.save_final_optimizer)
             else optimizer.state_dict()),
            { # metadata saved as json
                "step": step,
                "val_bpb": val_bpb, # loss at last step
                "model_config": model_config_kwargs,
                "eet_ever_routed": eet_ever_routed,
                "user_config": user_config, # inputs to the training script
                "device_batch_size": args.device_batch_size,
                "max_seq_len": args.max_seq_len,
                "total_batch_size": total_batch_size,
                "dataloader_state_dict": dataloader_state_dict,
                "loop_state": { # all loop state (other than step) so that we can resume training
                    "min_val_bpb": min_val_bpb,
                    "min_val_loss": min_val_loss,
                    "smooth_train_loss": smooth_train_loss,
                    "total_training_time": total_training_time,
                },
            },
            rank=ddp_rank,
        )
        # Rolling checkpoints: after a periodic save, delete the previous periodic checkpoint
        # to avoid disk bloat. The final checkpoint (last_step) is always kept.
        # Also clean up on the FINAL save, not just periodic ones. Without this a run
        # whose last step is 1680 with --save-every 200 leaves BOTH 1600 and 1680 on
        # disk, weights and optimizer shards, because the rolling delete below was
        # gated on is_periodic_save alone.
        if (is_periodic_save or last_step) and master_process and last_periodic_ckpt_step >= 0 \
                and last_periodic_ckpt_step != step:
            prev = last_periodic_ckpt_step
            for pattern in [
                os.path.join(checkpoint_dir, f"model_{prev:06d}.pt"),
                os.path.join(checkpoint_dir, f"meta_{prev:06d}.json"),
            ]:
                if os.path.exists(pattern):
                    os.remove(pattern)
            # Remove optimizer shards from all ranks (rank files we don't own are best-effort)
            for r in range(ddp_world_size):
                optim_path = os.path.join(checkpoint_dir, f"optim_{prev:06d}_rank{r:d}.pt")
                if os.path.exists(optim_path):
                    os.remove(optim_path)
            print0(f"Removed previous periodic checkpoint at step {prev:06d}")
        if is_periodic_save:
            last_periodic_ckpt_step = step

    # termination conditions
    if last_step:
        # Ensure sweep parsers always see an end-of-run loss line, even when
        # early-stop triggers before hitting a log_every boundary.
        debiased_at_step = smooth_train_loss / (1 - EMA_BETA**max(step, 1))
        print0(f"step {step:05d}/{num_iterations:05d} (final) | loss: {debiased_at_step:.6f} | early_stop: {int(args.early_stop_tokens > 0)}")
        # Machine-readable result for --timer-only. Emitted unconditionally at the end of
        # a probe so the caller never has to scrape a human log line, which turned out to
        # be printed or not depending on --log-every and where the loop exits.
        # total_training_time accumulates dt only for step > 10, so this is the mean of
        # the post-warmup steps: exactly the steady-state figure a projection needs.
        if args.timing_probe_steps > 0:
            _timed = max(step - 10, 0)
            _dt_ms = (total_training_time / _timed * 1000) if _timed > 0 else float("nan")
            print0(f"TIMING_PROBE_RESULT dt_ms={_dt_ms:.2f} timed_steps={_timed} "
                   f"ran_steps={step} full_iterations={num_iterations} "
                   f"total_batch_size={total_batch_size}")
        break

    # -------------------------------------------------------------------------
    # single training step
    # evaluate the gradient
    synchronize()
    t0 = time.time()
    # Enable EET Phase 2 to Phase 3 transition diagnostic check
    if model_config.use_eet:
        from nanochat.eet import EETPhaseScheduler
        _eet_sched = EETPhaseScheduler(
            num_iterations,
            warmup_frac=model_config.eet_warmup_frac,
            explore_frac=model_config.eet_explore_frac,
            reconstruct_lambda=model_config.eet_reconstruct_lambda,
            efficiency_lambda_start=model_config.eet_efficiency_lambda_start,
            efficiency_lambda_end=model_config.eet_efficiency_lambda_end,
        )
        # Only run the Phase 3 transition diagnostic when there actually is a Phase 3 and we had an exploration phase
        if _eet_sched.explore_end < num_iterations and _eet_sched.explore_end > _eet_sched.warmup_end and step == _eet_sched.explore_end:
            print0(f"\n[EET DIAGNOSTIC] Step {step:05d}: Running router structure check before entering Phase 3...")
            orig_model.train()  # must be training=True so do_route = eet_do_route and self.training → True
            with torch.no_grad():
                # Forward pass in soft-routing mode to populate _last_exit_probs
                _ = orig_model(x, eet_do_route=True, eet_phase=2, eet_lambda_r=0.0, eet_lambda_e=0.0)
            
            if hasattr(orig_model, '_last_exit_probs'):
                try:
                    check_router_learned(orig_model._last_exit_probs, x, orig_model)
                except Exception as e:
                    print0(f"[EET DIAGNOSTIC] Warning: correlation check failed with error: {e}")
            else:
                print0("[EET DIAGNOSTIC] Warning: exit probabilities not captured during forward pass.")
            
            orig_model.train()

    # Proposal head: the exact/approximate switch is read inside the compiled
    # forward, so it is decided OUT here. A step counter in the head would mean a
    # tensor .item() inside the graph, which breaks it every step; a Python bool set
    # from outside costs one recompile at the boundary and nothing after.
    _ph = getattr(orig_model, 'lm_head', None)
    if hasattr(_ph, 'exact_mode'):
        _ph.exact_mode = step < getattr(_ph, 'warmup', 0)

    # Enable MST diagnostic capture on log steps (last micro-step only)
    _mst_diag_this_step = (_mst_tracker is not None and _mst_diag_every > 0 and
                           (step % _mst_diag_every == 0 or step == num_iterations - 1))
    for micro_step in range(grad_accum_steps):
        # Only capture diagnostics on the last micro-step to avoid overhead
        if _mst_diag_this_step and micro_step == grad_accum_steps - 1:
            orig_model._diag_enabled = True
        if model_config.use_eet:
            # Phase scheduling computed OUTSIDE compiled forward to prevent
            # torch.compile from generating a single graph covering all phases
            # (which pre-allocates buffers for reconstruction loss etc.)
            from nanochat.eet import EETPhaseScheduler
            _eet_phase_info = EETPhaseScheduler(
                num_iterations,
                warmup_frac=model_config.eet_warmup_frac,
                explore_frac=model_config.eet_explore_frac,
                reconstruct_lambda=model_config.eet_reconstruct_lambda,
                efficiency_lambda_start=model_config.eet_efficiency_lambda_start,
                efficiency_lambda_end=model_config.eet_efficiency_lambda_end,
            ).get_phase(step)

            use_gumbel = getattr(model_config, 'eet_gumbel_temp_start', 0.0) > 0.0
            is_layer_weighted = (model_config.eet_loss_variant == 'layer_weighted')
            bypass_phases = use_gumbel or is_layer_weighted

            eet_do_route = _eet_phase_info['do_route']
            eet_phase = _eet_phase_info['phase']
            if eet_do_route:
                eet_ever_routed = True

            if eet_phase == 1:
                # Phase 1: take the EXACT same code path as non-EET dense training.
                # No extra kwargs, no tensor allocations, no requires_grad loops.
                # This ensures torch.compile generates the identical graph as dense.
                loss = model(x, y)
            else:
                # Phase 2/3: full EET forward with routing

                # Check for transition from Phase 1 to Phase 2/3
                if (hasattr(orig_model, 'eet_current_phase') and 
                    orig_model.eet_current_phase == 1):
                    orig_model.eet_current_phase = eet_phase
                    orig_model.eet_phase_tracker[0] = eet_phase
                    print0(f"[EET] Transitioning from Phase 1 (Dense Warmup) to Phase {eet_phase} (Routing active).")

                # Ensure routers and translators remain trainable (avoid writing if already True)
                for param in orig_model.eet_routers.parameters():
                    if not param.requires_grad:
                        param.requires_grad = True
                for param in orig_model.eet_translators.parameters():
                    if not param.requires_grad:
                        param.requires_grad = True

                # Capacity annealing: update target_active_frac at discrete intervals to avoid recompilation breaks
                _anneal_frac = getattr(model_config, 'eet_capacity_anneal_frac', 0.0)
                if _anneal_frac > 0.0:
                    progress = step / max(num_iterations, 1)
                    if progress < _anneal_frac:
                        t = progress / _anneal_frac
                        t_discrete = round(t * 10) / 10.0
                        base_target_frac = getattr(args, 'eet_target_active_frac', 0.125)
                        model_config.eet_target_active_frac = 0.5 + t_discrete * (base_target_frac - 0.5)
                    else:
                        model_config.eet_target_active_frac = getattr(args, 'eet_target_active_frac', 0.125)

                eet_gumbel_temp_tensor = torch.tensor(1.0, device=x.device, dtype=torch.float32)
                if model_config.eet_gumbel_temp_start > 0.0:
                    t_start = model_config.eet_gumbel_temp_start
                    t_end = model_config.eet_gumbel_temp_end
                    progress = min(max(step / max(num_iterations, 1), 0.0), 1.0)
                    temp_val = t_start * ((t_end / t_start) ** progress)
                    eet_gumbel_temp_tensor = torch.tensor(temp_val, device=x.device, dtype=torch.float32)

                eet_step_tensor = torch.tensor(step, device=x.device, dtype=torch.float32)
                eet_total_steps_tensor = torch.tensor(num_iterations, device=x.device, dtype=torch.float32)

                # Two-pass ce_guided: run dense forward first to get per-token CE baseline
                _reinforce_interval = getattr(model_config, 'eet_reinforce_interval', 0)
                if (_reinforce_interval > 0 and eet_phase == 3 and
                    step % _reinforce_interval == 0 and eet_do_route):
                    dense_ce = orig_model._compute_dense_per_token_ce(x, y)
                    orig_model._reinforce_dense_ce = dense_ce

                # Concurrent dense distillation: run dense forward pass to get teacher targets
                _distill_interval = getattr(model_config, 'eet_dense_distill_interval', 0)
                eet_dense_x = None
                if (_distill_interval > 0 and eet_phase in {2, 3} and
                    step % _distill_interval == 0 and eet_do_route):
                    with torch.no_grad():
                        eet_dense_x = orig_model._compute_dense_logits(x)

                loss = model(x, y,
                             eet_do_route=eet_do_route,
                             eet_phase=eet_phase,
                             eet_lambda_r=torch.tensor(_eet_phase_info['lambda_r'], device=x.device, dtype=torch.float32),
                             eet_lambda_e=torch.tensor(_eet_phase_info['lambda_e'], device=x.device, dtype=torch.float32),
                             eet_gumbel_temp=eet_gumbel_temp_tensor,
                             eet_step=eet_step_tensor,
                             eet_total_steps=eet_total_steps_tensor,
                             eet_dense_x=eet_dense_x)
        else:
            loss = model(x, y)
            
        # Capture the final training step probabilities on the last micro-step of the final training step
        if model_config.use_eet and (last_step or step == num_iterations - 1) and micro_step == grad_accum_steps - 1:
            if hasattr(orig_model, '_last_exit_probs') and orig_model._last_exit_probs is not None:
                orig_model._final_train_exit_probs = orig_model._last_exit_probs.detach().cpu().clone()
                orig_model._final_train_tokens = x.detach().cpu().clone()
            if hasattr(orig_model, '_last_enforced_capacities'):
                orig_model._final_enforced_capacities = list(orig_model._last_enforced_capacities)
                orig_model._final_active_counts = list(orig_model._last_active_counts)
                orig_model._final_T = orig_model._last_T
                
        if _mst_diag_this_step and micro_step == grad_accum_steps - 1:
            orig_model._diag_enabled = False
        if is_dp:
            loss = loss.mean()
        train_loss = loss.detach() # for logging
        # 18B: Gradient penalty — weight Frobenius norm regularization
        # Penalizes large weight norms to constrain Lipschitz constant
        gp_lambda = float(getattr(args, 'p18_gradient_penalty', 0.0))
        if gp_lambda > 0:
            gp_loss = sum(
                p.float().square().mean() for p in model.parameters()
                if p.ndim >= 2 and p.requires_grad
            )
            loss = loss + gp_lambda * gp_loss
        # Phase 20: Auxiliary routing loss for DGCR (20D) and NCEA (20H)
        # These train the router to predict which branch is best, separate from main loss
        _p20_dgcr = getattr(args, 'p20_dgcr_branches', 0)
        _p20_ncea = getattr(args, 'p20_ncea_branches', 0)
        _p23_std_moe = getattr(args, 'p23_std_moe_experts', 0)
        if _p20_dgcr > 0 or _p20_ncea > 0 or _p23_std_moe > 0:
            aux_loss_total = torch.tensor(0.0, device=loss.device)
            for block in orig_model.transformer.h:
                if hasattr(block, 'mlp') and hasattr(block.mlp, 'compute_aux_loss'):
                    aux = block.mlp.compute_aux_loss()
                    if aux is not None:
                        aux_loss_total = aux_loss_total + aux
            if aux_loss_total.item() > 0:
                loss = loss + aux_loss_total
        # EET: ce_guided depth classification — computed OUTSIDE the compiled forward
        # to avoid changing the compiled graph. Re-runs the tiny global router on the
        # stored embedding to get fresh router_logits WITH gradients.
        if (model_config.use_eet and model_config.eet_compute_skip and
            getattr(model_config, 'eet_loss_variant', '') == 'ce_guided' and
            hasattr(orig_model, '_last_x0_for_ce') and
            hasattr(orig_model, '_last_per_token_ce')):
            _x0 = orig_model._last_x0_for_ce         # (B, T, C), detached
            _ptce = orig_model._last_per_token_ce     # (B, T), detached
            _tgt = y

            # Re-run the global router (tiny: 2 linear layers) to get differentiable logits
            _global_router = orig_model.eet_routers[0]
            _freq_bias = getattr(orig_model, '_freq_bias', None)
            _pos_bias = getattr(orig_model, '_pos_bias', None)
            _rl = _global_router(
                _x0,
                freq_bias=_freq_bias,
                pos_bias=_pos_bias,
                freq_alpha=model_config.eet_freq_prior_alpha,
                pos_beta=model_config.eet_pos_prior_beta
            )  # (B, T, n_exits) — WITH gradients to router params

            _n_exits = _rl.shape[-1]
            _valid_mask = (_tgt != -1).float()
            _n_valid = _valid_mask.sum().clamp(min=1.0)

            # Two-pass: use differential CE if dense baseline is available
            _dense_ce = getattr(orig_model, '_reinforce_dense_ce', None)
            _ce_signal = (_ptce - _dense_ce) if _dense_ce is not None else _ptce

            with torch.no_grad():
                _valid_ce = _ce_signal[_tgt != -1]
                if _valid_ce.numel() > 1:
                    _ce_mean = _valid_ce.mean()
                    _ce_std = _valid_ce.std().clamp(min=0.1)
                    _normalized = torch.sigmoid((_ce_signal - _ce_mean) / _ce_std)
                    _target_exit = (_normalized * _n_exits).long().clamp(0, _n_exits - 1)
                else:
                    _target_exit = torch.zeros_like(_tgt)
                _target_exit = _target_exit * (_tgt != -1).long()

            _depth_loss = F.cross_entropy(
                _rl.float().view(-1, _n_exits),
                _target_exit.view(-1),
                ignore_index=-100,
                reduction='none',
            )
            _depth_loss = (_depth_loss.view_as(_tgt) * _valid_mask).sum() / _n_valid
            _ce_lambda = getattr(model_config, 'eet_ce_guided_lambda', 1.0)
            loss = loss + _ce_lambda * _depth_loss

        loss = loss / grad_accum_steps # each .backward() is a grad sum => normalize loss here
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        x, y, dataloader_state_dict = next(train_loader) # prefetch the next batch while the GPU is busy with forward/backward
        x, y = sch_apply_holdout(x, y)
    # Capture MST per-sub grad norms BEFORE optimizer step / zero_grad clears them
    if _mst_diag_this_step:
        _cached_sub_grad_norms = {}
        N = model_config.get('mst_n_subs', 0) if isinstance(model_config, dict) else getattr(model_config, 'mst_n_subs', 0)
        # Accumulate grad norms per sub index across layers.
        # With progressive merge, later layers have fewer subs — only iterate valid indices.
        sub_grad_sq = [0.0] * N
        for layer in orig_model.layers:
            # BatchedMSTLayer stores weights as fused 2D tensors — no sub_blocks
            if not hasattr(layer, 'sub_blocks'):
                # For batched layers, attribute per-sub grad norms from the fused
                # attention/FFN weights. Each weight is (N*out, in), so we split by N.
                for attr in ('c_q_w', 'c_k_w', 'c_v_w', 'c_proj_w', 'fc_w', 'fc_proj_w'):
                    p = getattr(layer, attr, None)
                    if p is not None and p.grad is not None:
                        n_subs = getattr(layer, 'N', N)
                        chunks = p.grad.view(n_subs, -1, p.grad.shape[-1])
                        for j in range(min(n_subs, N)):
                            sub_grad_sq[j] += float(chunks[j].float().norm() ** 2)
            else:
                n_subs_this_layer = len(layer.sub_blocks)
                for j in range(n_subs_this_layer):
                    block = layer.sub_blocks[j]
                    for p in block.parameters():
                        if p.grad is not None:
                            # Map merged sub indices back: after merge, sub j in this layer
                            # encompasses original subs [j*stride, (j+1)*stride).
                            # For simplicity, accumulate to sub index j (valid for uniform N).
                            if j < N:
                                sub_grad_sq[j] += float(p.grad.float().norm() ** 2)
        for j in range(N):
            _cached_sub_grad_norms[f'grad_norm_S{j}'] = float(sub_grad_sq[j] ** 0.5)
        orig_model._cached_grad_norms = _cached_sub_grad_norms
    # Placeholder: EET gradient diagnostics are captured AFTER clipping (see below)
    _eet_grad_pending = model_config.use_eet and hasattr(orig_model, 'eet_routers') and eet_phase >= 2
    # step the optimizer
    lrm = get_lr_multiplier_onecycle(step) if use_research_scheduler else get_lr_multiplier(step)
    muon_momentum = get_muon_momentum(step)
    muon_weight_decay = get_weight_decay(step)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
        if group['kind'] in ('muon', 'shampoo'):
            group["momentum"] = muon_momentum
            group["weight_decay"] = muon_weight_decay
    if scaler is not None:
        scaler.unscale_(optimizer)
        # Clip early-exit router gradients specifically to keep routing updates slow and stable
        if hasattr(orig_model, 'eet_routers') and orig_model.eet_routers is not None and eet_phase >= 2:
            torch.nn.utils.clip_grad_norm_(orig_model.eet_routers.parameters(), max_norm=10.0)
        # In distributed training, all ranks must agree on whether to skip the step.
        # Each rank may independently encounter inf/nan gradients, so we all-reduce
        # the found_inf flag (MAX = if any rank found inf, all ranks skip).
        if is_ddp_initialized():
            for v in scaler._found_inf_per_device(optimizer).values():
                dist.all_reduce(v, op=dist.ReduceOp.MAX)
        scaler.step(optimizer)
        scaler.update()
    else:
        # Clip early-exit router gradients specifically to keep routing updates slow and stable
        if hasattr(orig_model, 'eet_routers') and orig_model.eet_routers is not None and eet_phase >= 2:
            torch.nn.utils.clip_grad_norm_(orig_model.eet_routers.parameters(), max_norm=10.0)
    # Capture EET router/translator gradient norms AFTER clipping so logged values
    # reflect what the optimizer actually sees (not the raw pre-clip norms)
    if _eet_grad_pending:
        _eet_grad_info = {'step': step}
        total_router_grad_sq = 0.0
        n_router_params_with_grad = 0
        for layer_idx, router in enumerate(orig_model.eet_routers):
            layer_grad_sq = 0.0
            layer_n = 0
            for pname, p in router.named_parameters():
                if p.grad is not None:
                    gnorm = float(p.grad.float().norm())
                    layer_grad_sq += gnorm ** 2
                    layer_n += 1
                    _eet_grad_info[f'router_{layer_idx}_{pname}_grad_norm'] = gnorm
            layer_gnorm = layer_grad_sq ** 0.5
            _eet_grad_info[f'router_{layer_idx}_total_grad_norm'] = layer_gnorm
            total_router_grad_sq += layer_grad_sq
            n_router_params_with_grad += layer_n
        _eet_grad_info['router_total_grad_norm'] = total_router_grad_sq ** 0.5
        _eet_grad_info['n_router_params_with_grad'] = n_router_params_with_grad
        total_trans_grad_sq = 0.0
        for layer_idx, translator in enumerate(orig_model.eet_translators):
            layer_grad_sq = 0.0
            for pname, p in translator.named_parameters():
                if p.grad is not None:
                    gnorm = float(p.grad.float().norm())
                    layer_grad_sq += gnorm ** 2
                    _eet_grad_info[f'translator_{layer_idx}_{pname}_grad_norm'] = gnorm
            _eet_grad_info[f'translator_{layer_idx}_total_grad_norm'] = layer_grad_sq ** 0.5
            total_trans_grad_sq += layer_grad_sq
        _eet_grad_info['translator_total_grad_norm'] = total_trans_grad_sq ** 0.5
        orig_model._eet_grad_info = _eet_grad_info
    # Resume non-scaler optimizer step logic (Fix 4C, GER, optimizer.step)
    if scaler is None:
        # Fix 4C: gradient clipping before optimizer step (protects against large gradients
        # in adaptive gate pathways early in training, e.g. PermutationMoE, RemixedLinear)
        clip_val = 10.0 if (model_config.use_eet and args.max_grad_norm == 1.0) else args.max_grad_norm
        if clip_val > 0:
            torch.nn.utils.clip_grad_norm_(orig_model.parameters(), clip_val)
        # 19F: Gradient Equilibrium Regularization (gradient modifier)
        # Equalizes per-block gradient norms to prevent gradient starvation/domination
        ger_lambda = float(getattr(args, 'p19_grad_equilibrium', 0.0))
        if ger_lambda > 0:
            with torch.no_grad():
                block_grad_norms = []
                block_key_params = []
                for block in orig_model.transformer.h:
                    mlp = block.mlp if hasattr(block, 'mlp') else (block.ffwd if hasattr(block, 'ffwd') else None)
                    if mlp is not None and hasattr(mlp, 'c_fc') and mlp.c_fc.weight.grad is not None:
                        gn = mlp.c_fc.weight.grad.float().norm()
                        block_grad_norms.append(gn)
                        block_key_params.append(mlp.c_fc.weight)
                if len(block_grad_norms) >= 2:
                    gn_stack = torch.stack(block_grad_norms)
                    gn_mean = gn_stack.mean()
                    # Correction factor: scale each layer's gradients toward the mean
                    for idx, (gn, param) in enumerate(zip(block_grad_norms, block_key_params)):
                        if gn > 1e-12:
                            correction = (gn_mean / gn).clamp(1.0 - ger_lambda, 1.0 + ger_lambda)
                            param.grad.mul_(correction.to(param.grad.dtype))
        optimizer.step()
    # Fix 1H: update PermutationMoE temperature each step
    if use_research_mode:
        perm_temp = get_perm_temperature(step)
        for module in orig_model.modules():
            if hasattr(module, 'temperature') and isinstance(getattr(module, 'temperature'), torch.Tensor):
                module.temperature.fill_(perm_temp)
    # Phase 16A: CKR temperature annealing (exponential: temp_start → temp_end)
    if getattr(args, 'cclblock_modulation', '') in ('ckr', 'ckr_ffn'):
        temp_start = getattr(args, 'cclblock_ckr_temp_start', 1.0)
        temp_end = getattr(args, 'cclblock_ckr_temp_end', 1.0)
        if temp_start != temp_end and num_iterations > 1:
            progress = min(step / (num_iterations - 1), 1.0)
            current_temp = temp_start * (temp_end / temp_start) ** progress
            from nanochat.gpt import CausalKernelLinear
            for module in orig_model.modules():
                if isinstance(module, CausalKernelLinear):
                    module._temperature.fill_(current_temp)
    model.zero_grad(set_to_none=True)
    train_loss_f = train_loss.item() # .item() is a CPU-GPU sync point
    # Collect MST router diagnostics (reads tensors already computed in forward)
    if _mst_tracker is not None:
        _mst_tracker.collect(orig_model)
        # Compute and log full diagnostics if this was a diagnostic step
        if _mst_diag_this_step and master_process:
            try:
                mst_diag = orig_model.compute_diagnostics()
                mst_diag['step'] = step
                mst_diag['train_loss'] = train_loss_f
                with open(_mst_diag_log, 'a') as f:
                    # Convert any non-serializable values
                    f.write(json.dumps(mst_diag, default=str) + '\n')
                # Print summary to stdout
                sim_keys = [k for k in mst_diag if 'sub_sim' in k and '_mean' in k]
                ent_keys = [k for k in mst_diag if 'route_entropy' in k]
                grad_keys = [k for k in mst_diag if 'grad_norm' in k]
                sim_vals = [mst_diag[k] for k in sorted(sim_keys)]
                ent_vals = [mst_diag[k] for k in sorted(ent_keys)]
                grad_vals = [mst_diag[k] for k in sorted(grad_keys)]
                sim_str = ', '.join(f'{v:.3f}' for v in sim_vals) if sim_vals else 'n/a'
                ent_str = ', '.join(f'{v:.3f}' for v in ent_vals) if ent_vals else 'n/a'
                grad_str = ', '.join(f'{v:.3f}' for v in grad_vals) if grad_vals else 'n/a'
                # Stage 16: route_ent above is the TRANSITION router and says nothing about
                # conditional stream execution. Without this line a collapsed stream gate is
                # invisible in the console, and bpb alone cannot detect it: the most
                # collapsed config measured the LOWEST training loss.
                # stream_load_min is the least-used stream's share of its k/N ideal, so
                # 1.0 is balanced and 0.0 means that stream is never selected and its FFN
                # is frozen at initialization.
                load_keys = [k for k in mst_diag if k.startswith('stream_load_min_')]
                load_str = ', '.join(f'{mst_diag[k]:.2f}' for k in sorted(load_keys))
                stream_part = f" | stream_load_min=[{load_str}]" if load_keys else ""
                print0(f"  [MST diag] sub_sim=[{sim_str}] | route_ent=[{ent_str}] | "
                       f"grad_norm=[{grad_str}]{stream_part}")
            except Exception as e:
                print0(f"  [MST diag] ERROR: {e}")
    synchronize()
    t1 = time.time()
    dt = t1 - t0
    # -------------------------------------------------------------------------

    # logging (CPU action only)
    smooth_train_loss = EMA_BETA * smooth_train_loss + (1 - EMA_BETA) * train_loss_f # EMA the training loss
    debiased_smooth_loss = smooth_train_loss / (1 - EMA_BETA**(step + 1)) # debias the EMA
    if args.step_loss_file and master_process:
        with open(args.step_loss_file, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "step": int(step),
                "tokens": int(step * total_batch_size),
                "loss": float(debiased_smooth_loss),
            }) + "\n")
    pct_done = 100 * step / num_iterations
    tok_per_sec = int(total_batch_size / dt)
    flops_per_sec = num_flops_per_token * total_batch_size / dt
    mfu = 100 * flops_per_sec / (gpu_peak_flops * ddp_world_size)
    if step > 10:
        total_training_time += dt # only count the time after the first 10 steps
    # Calculate ETA based on average time per step (excluding first 10 steps)
    steps_done = step - 10
    if steps_done > 0:
        avg_time_per_step = total_training_time / steps_done
        remaining_steps = num_iterations - step
        eta_seconds = remaining_steps * avg_time_per_step
        eta_str = f" | eta: {eta_seconds/60:.1f}m"
    else:
        eta_str = ""
    epoch = f"{dataloader_state_dict['epoch']} pq: {dataloader_state_dict['pq_idx']} rg: {dataloader_state_dict['rg_idx']}"
    adamw_lrs = [g["lr"] for g in optimizer.param_groups if g.get("kind") == "adamw"]
    muon_lrs = [g["lr"] for g in optimizer.param_groups if g.get("kind") == "muon"]
    lr_msg = f"lr(adamw:{(sum(adamw_lrs)/len(adamw_lrs)) if adamw_lrs else 0:.3e}, muon:{(sum(muon_lrs)/len(muon_lrs)) if muon_lrs else 0:.3e})"
    if step % args.log_every == 0 or step == num_iterations - 1 or last_step:
        print0(f"step {step:05d}/{num_iterations:05d} ({pct_done:.2f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | {lr_msg} | dt: {dt * 1000:.2f}ms | tok/sec: {tok_per_sec:,} | bf16_mfu: {mfu:.2f} | epoch: {epoch} | total time: {total_training_time/60:.2f}m{eta_str}")
        # SAP block-head diagnostics (recorded in eager mode only; compiled runs report
        # the block head through the SAP eval instead).
        _sap_stats = getattr(orig_model, '_sap_stats', None)
        if _sap_stats:
            print0("SAP train | " + " | ".join(
                f"{k}: {v.tolist() if v.numel() > 1 else round(v.item(), 4)}" for k, v in _sap_stats.items()))
        # Phase 17: Modulation diagnostics at log intervals
        if mod_diag is not None:
            diag_metrics = mod_diag.collect()
            if diag_metrics:
                print0(mod_diag.format(diag_metrics))
                if master_process:
                    diag_file = os.path.join(checkpoint_dir, "modulation_diagnostics.jsonl")
                    mod_diag.save_to_file(diag_metrics, step, diag_file)
            else:
                diag_metrics = None
            # Phase 19: Expanded diagnostics (always collected, even without CKR layers)
            p19_metrics = mod_diag.collect_p19(orig_model)
            if p19_metrics:
                p19_log = mod_diag.format_p19(p19_metrics)
                if p19_log:
                    print0(p19_log)
            # Phase 20: Dynamic weight computation diagnostics
            p20_metrics = mod_diag.collect_p20(orig_model)
            if p20_metrics:
                p20_log = mod_diag.format_p20(p20_metrics)
                if p20_log:
                    print0(p20_log)
        else:
            p19_metrics = None
            p20_metrics = None
        # ── Gate stats logging ────────────────────────────────────────────────
        if gate_stats_log and args.gate_stats_every > 0 and (step % args.gate_stats_every == 0 or last_step):
            gs = collect_gate_stats(model, step)
            tmpl_str = ""
            if 'template_weights' in gs:
                tw = gs['template_weights']
                if isinstance(tw, list):
                    tw_fmt = "[" + ", ".join(f"{w:.3f}" for w in tw) + "]"
                else:
                    tw_fmt = f"{tw:.3f}"
                tmpl_str = f" | tmpl_w={tw_fmt}"
            if 'template_entropy' in gs:
                tmpl_str += f" H={gs['template_entropy']:.3f}"
            print0(
                f"  gate_stats | layers={gs['layers']} "
                f"{tmpl_str} "
                f"| basis µ={gs.get('basis_mean',float('nan')):.3f} "
                f"σ={gs.get('basis_std',float('nan')):.3f} "
                f"dead={gs.get('basis_dead',float('nan')):.1%} "
                f"sat={gs.get('basis_sat',float('nan')):.1%} "
                f"| out µ={gs.get('out_mean',float('nan')):.3f} "
                f"σ={gs.get('out_std',float('nan')):.3f} "
                f"| ∇gate={gs['gate_grad_norm']:.3e} "
                f"∇struct={gs['struct_grad_norm']:.3e}"
            )
            if master_process:
                with open(gate_stats_log, 'a') as _gf:
                    _gf.write(json.dumps(gs) + '\n')
        # EET: Early Exit diagnostics (safe to call .item() here — outside compiled forward)
        if model_config.use_eet and hasattr(orig_model, '_eet_diagnostics'):
            _eet_diag = orig_model._eet_diagnostics
            _eet_phase = _eet_diag.get('phase', 0)
            _eet_active = _eet_diag['active_frac'].item() if hasattr(_eet_diag.get('active_frac', 0), 'item') else _eet_diag.get('active_frac', 1.0)
            _eet_exit = _eet_diag['total_exit_frac'].item() if hasattr(_eet_diag.get('total_exit_frac', 0), 'item') else _eet_diag.get('total_exit_frac', 0.0)
            _active_counts = getattr(orig_model, '_last_active_counts', None)
            _counts_str = f" | tokens={_active_counts}" if _active_counts else ""
            _a3d_str = " [A³D]" if _eet_diag.get('a3d', False) else ""
            print0(f"  eet{_a3d_str} | phase={_eet_phase} | active={_eet_active:.3f} | exit_frac={_eet_exit:.3f}{_counts_str}")
        # EET: Router gradient diagnostics — console print + JSONL log file
        if model_config.use_eet and hasattr(orig_model, '_eet_grad_info'):
            _gi = orig_model._eet_grad_info
            _r_gnorm = _gi.get('router_total_grad_norm', 0.0)
            _t_gnorm = _gi.get('translator_total_grad_norm', 0.0)
            _n_with_grad = _gi.get('n_router_params_with_grad', 0)
            # Per-layer summary
            per_layer_parts = []
            for layer_idx in range(len(orig_model.eet_routers)):
                ln = _gi.get(f'router_{layer_idx}_total_grad_norm', 0.0)
                per_layer_parts.append(f'L{layer_idx}={ln:.3e}')
            per_layer_str = ' '.join(per_layer_parts) if per_layer_parts else 'none'
            print0(f"  eet_grad | ∇router={_r_gnorm:.3e} ∇trans={_t_gnorm:.3e} params_with_grad={_n_with_grad} | {per_layer_str}")
            # Write full grad info to JSONL log file (only when routing is active)
            if master_process and _gi.get('n_router_params_with_grad', 0) > 0:
                import json
                _gi['eet_phase'] = _eet_diag.get('phase', 0) if model_config.use_eet and hasattr(orig_model, '_eet_diagnostics') else 0
                _gi['active_frac'] = _eet_active if model_config.use_eet and hasattr(orig_model, '_eet_diagnostics') else 1.0
                eet_grad_log_path = os.path.join(checkpoint_dir, "eet_grad_log.jsonl")
                os.makedirs(checkpoint_dir, exist_ok=True)
                with open(eet_grad_log_path, 'a') as _ef:
                    _ef.write(json.dumps(_gi) + '\n')
    if step % 100 == 0:
        log_data = {
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "train/loss": debiased_smooth_loss,
            "train/lrm": lrm,
            "train/dt": dt,
            "train/tok_per_sec": tok_per_sec,
            "train/mfu": mfu,
            "train/epoch": epoch,
        }
        # Add modulation diagnostics to wandb if available
        if mod_diag is not None and diag_metrics is not None:
            log_data.update(mod_diag.to_dict(diag_metrics))
        # Phase 19: expanded diagnostics
        if mod_diag is not None and p19_metrics:
            log_data.update(mod_diag.to_dict_p19(p19_metrics))
        # Phase 20: dynamic weight diagnostics
        if mod_diag is not None and p20_metrics:
            log_data.update(mod_diag.to_dict_p20(p20_metrics))
        # EET: add exit diagnostics to step log for post-hoc effective FLOPs analysis
        if model_config.use_eet and hasattr(orig_model, '_eet_diagnostics'):
            _eet_diag = orig_model._eet_diagnostics
            _af = _eet_diag.get('active_frac', None)
            _ef = _eet_diag.get('total_exit_frac', None)
            log_data['eet/phase'] = _eet_diag.get('phase', 0)
            log_data['eet/active_frac'] = _af.item() if hasattr(_af, 'item') else float(_af or 1.0)
            log_data['eet/exit_frac'] = _ef.item() if hasattr(_ef, 'item') else float(_ef or 0.0)
        wandb_run.log(log_data)

    # state update
    first_step_of_run = (step == 0) or (resuming and step == args.resume_from_step)
    step += 1

    # The garbage collector is sadly a little bit overactive and for some poorly understood reason,
    # it spends ~500ms scanning for cycles quite frequently, just to end up cleaning up very few tiny objects each time.
    # So we manually manage and help it out here
    if first_step_of_run:
        gc.collect() # manually collect a lot of garbage from setup
        gc.freeze() # immediately freeze all currently surviving objects and exclude them from GC
        gc.disable() # nuclear intervention here: disable GC entirely except:
    elif step % 5000 == 0: # every 5000 steps...
        gc.collect() # manually collect, just to be safe for very, very long runs

# print a few more stats
print0(f"Peak memory usage: {get_max_memory() / 1024 / 1024:.2f}MiB")
print0(f"Total training time: {total_training_time/60:.2f}m")
if val_bpb is not None:
    print0(f"Minimum validation bpb: {min_val_bpb:.6f}")
    print0(f"Minimum validation loss (nats/byte): {min_val_loss:.6f}")

# EET: Print comprehensive final router diagnostics at the end of the run
if model_config.use_eet:
    print0("\n================================================================================")
    print0("[EET FINAL ROUTER DIAGNOSTICS]")
    print0("================================================================================")

    # 1. Enforced physical capacity distribution (what actually happened with tokens)
    if hasattr(orig_model, '_final_enforced_capacities') and hasattr(orig_model, '_final_active_counts'):
        capacities = orig_model._final_enforced_capacities
        active_counts = orig_model._final_active_counts
        T = orig_model._final_T
        n_blocks = len(active_counts)
        n_rl = len(capacities)

        B = 1
        if hasattr(orig_model, '_final_train_exit_probs') and orig_model._final_train_exit_probs is not None:
            B = orig_model._final_train_exit_probs.size(0)

        total_tokens = B * T
        print0(f"Enforced physical exit distribution ({total_tokens:,} tokens, B={B}, T={T}, {n_blocks} blocks, {n_rl} routing slots):")
        # Compute per-slot exit counts from active_counts
        # active_counts[i] = K_cur at block i (before block runs)
        # Exit at slot k = active_counts[routing_block_k] - active_counts[routing_block_k+1]
        # Routing slots live at blocks [min_exit_layer .. n_layer-2], NOT at 1..n_rl.
        # Hardcoding `slot + 1` is only correct when min_exit_layer == 1; with
        # --eet-min-exit-layer 4 it read blocks 1-3, where nothing has exited yet, and
        # reported "0 tokens" at every slot while the final layer showed 9.96% active --
        # a contradiction, since the other 90% had to have exited somewhere.
        _min_exit = int(getattr(args, 'eet_min_exit_layer', 1))
        for slot in range(n_rl):
            block_idx = _min_exit + slot
            k_before = active_counts[block_idx]
            if block_idx + 1 < n_blocks:
                k_after = active_counts[block_idx + 1]
            else:
                k_after = active_counts[-1]  # last block's active count
            exited = k_before - k_after
            pct = (exited / T) * 100
            label = f'exit_{slot}'
            print0(f"  {label:12s}: {pct:6.2f}% ({exited * B:,} tokens) [capacity: {capacities[slot] * B:,}]")
        # Final layer
        final_active = active_counts[-1]
        final_pct = (final_active / T) * 100
        print0(f"  {'final_layer':12s}: {final_pct:6.2f}% ({final_active * B:,} tokens active)")

        # Enforced active fraction
        enforced_active = sum(active_counts) / (n_blocks * T)
        enforced_exit = 1.0 - (active_counts[-1] / T)
        print0(f"\nEnforced active fraction: {enforced_active:.3f} (exit_frac={enforced_exit:.3f})")
    else:
        print0("(No enforced capacity data available — compute_skip may not have been enabled)")

    # 2. Soft probability diagnostics (router's learned preferences)
    if hasattr(orig_model, '_final_train_exit_probs') and orig_model._final_train_exit_probs is not None:
        exit_probs = orig_model._final_train_exit_probs.detach().cpu().float() # (B, T, n_exits)
        n_exits = exit_probs.size(-1)

        # Argmax exit layer for every token
        argmax_exits = exit_probs.argmax(dim=-1).numpy().ravel()
        total_tokens_evaluated = len(argmax_exits)

        print0(f"\nRouter soft probability diagnostics ({total_tokens_evaluated:,} tokens):")
        print0("Router preferred exit (argmax of soft probs):")
        for slot in range(n_exits):
            count = (argmax_exits == slot).sum()
            pct = (count / max(total_tokens_evaluated, 1)) * 100
            label = 'final_layer' if slot == n_exits - 1 else f'exit_{slot}'
            print0(f"  {label:12s}: {pct:6.2f}% ({count:,} tokens)")

        # Soft probability stats per slot
        print0("\nSoft exit probability stats per slot:")
        for slot in range(n_exits):
            slot_probs = exit_probs[:, :, slot].numpy().ravel()
            label = 'final_layer' if slot == n_exits - 1 else f'exit_{slot}'
            print0(f"  {label:12s}: mean={slot_probs.mean():.6f} std={slot_probs.std():.6f} min={slot_probs.min():.6f} max={slot_probs.max():.6f}")

        # Mean and std of expected exit layer
        layer_indices = torch.arange(n_exits).float()
        expected_exit = (exit_probs * layer_indices).sum(dim=-1).numpy().ravel()
        print0(f"\nMean expected exit layer (soft): {expected_exit.mean():.4f}")
        print0(f"Std expected exit layer (soft):  {expected_exit.std():.6f}")

        # Collapse warnings
        if expected_exit.std() < 0.01:
            print0(f"\n[EET WARNING] ⚠ ROUTER COLLAPSE: std={expected_exit.std():.6f} — router is near-constant across all tokens.")
        elif expected_exit.std() < 0.1:
            print0(f"\n[EET WARNING] ⚠ LOW DIFFERENTIATION: std={expected_exit.std():.6f} — router is barely differentiating tokens.")
        else:
            print0("\n[EET INFO] ✓ Router has learned token-dependent exit behavior!")
    else:
        print0("\n[EET FINAL DIAGNOSTIC] Warning: No router exit probabilities captured during training steps.")
    print0("================================================================================\n")

# MST: write per-run summary row to mst_results.csv
if _mst_tracker is not None:
    sp = orig_model.num_scaling_params() if hasattr(orig_model, 'num_scaling_params') else {}
    final_train_loss = smooth_train_loss / (1 - EMA_BETA**max(step, 1)) if step > 0 else None
    _mst_tracker.write_csv(
        step=step,
        val_bpb=val_bpb,
        train_loss_final=final_train_loss,
        total_training_time=total_training_time,
        num_flops_per_token=num_flops_per_token,
        num_active_flops_per_token=num_active_flops_per_token,
        num_active_params=num_active_params,
        total_batch_size=total_batch_size,
        num_params=num_params,
        sp=sp,
    )

# ── SCH: end-of-training diagnostics ─────────────────────────────────────────
# Every metric in section 6 of the plan, written once per run to sch_results.csv
# so a sweep is a table rather than a log-scraping exercise.
# NOTE: this is an OR over the sch_* flags, so any one of them being on by default
# makes this run for every architecture. --sch-decile-metrics used to default to 1
# and did exactly that, OOMing runs that have no code head at all.
if (bool(int(getattr(args, 'use_code_head', 0))) or args.sch_holdout_tokens > 0
        or args.sch_decile_metrics or args.sch_rank_probe) and master_process:
    try:
        from nanochat.code_metrics import run_all_diagnostics, write_sch_row
        _sch = run_all_diagnostics(
            orig_model,
            build_val_loader=build_val_loader,
            token_bytes=token_bytes,
            vocab_size=vocab_size,
            steps=args.sch_eval_steps,
            decile=bool(args.sch_decile_metrics),
            rank_contexts=int(args.sch_rank_probe),
            holdout_ids=sch_holdout_ids,
            tokenizer_dir=args.tokenizer_dir,
            device=device,
        )
        write_sch_row(
            run_dir=checkpoint_dir,
            args=args, config=model_config, metrics=_sch,
            val_bpb=val_bpb, min_val_bpb=min_val_bpb,
            num_flops_per_token=num_flops_per_token,
            num_params=num_params, total_tokens=total_tokens,
            total_training_time=total_training_time,
            scaling_params=orig_model.num_scaling_params(),
        )
    except Exception as e:
        import traceback
        print0(f"[SCH] diagnostics failed (training results are unaffected): {e}")
        traceback.print_exc()

# Log to report
from nanochat.report import get_report
section_name = "Base model training"
if args.model_tag:
    section_name += f" ({args.model_tag})"
get_report().log(section=section_name, data=[
    user_config, # CLI args
    { # stats about the training setup
        "Number of parameters": num_params,
        "Number of FLOPs per token": f"{num_flops_per_token:e}",
        "Number of active FLOPs per token": f"{num_active_flops_per_token:e}",
        "Calculated number of iterations": num_iterations,
        "Number of training tokens": total_tokens,
        "Tokens : Scaling params ratio": total_batch_size * num_iterations / num_scaling_params,
        "DDP world size": ddp_world_size,
        "warmup_ratio": args.warmup_ratio,
        "warmdown_ratio": args.warmdown_ratio,
        "final_lr_frac": args.final_lr_frac,
        "research_warmup_ratio": args.research_warmup_ratio,
    },
    { # stats about training outcomes
        "Minimum validation bpb": min_val_bpb if val_bpb is not None else None,
        "Final validation bpb": val_bpb,
        "CORE metric estimate": results.get("core_metric", None),
        "MFU %": f"{mfu:.2f}%",
        "Total training flops": f"{flops_so_far:e}",
        "Total active training flops": f"{num_active_flops_per_token * total_batch_size * step:e}",
        "Total training time": f"{total_training_time/60:.2f}m",
        "Peak memory usage": f"{get_max_memory() / 1024 / 1024:.2f}MiB",
    }
])

# cleanup
wandb_run.finish() # wandb run finish
compute_cleanup()
