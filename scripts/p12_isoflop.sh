#!/usr/bin/env bash
# ============================================================================
# P12: isoFLOP profile. Every arm gets the SAME training compute.
#
# WHY THIS EXISTS
#   The compute-optimal ladder sets each arm's token budget from its own
#   parameter count (10.5 x (matrices + lm_head)). That is a defensible
#   protocol but it is a protocol, and it is the one a reviewer will question,
#   because MST draws 1.43x the tokens of the dense model matched to it on
#   active FLOPs (see p11_active_params.py). An isoFLOP profile removes the
#   question entirely: fix C, vary model size, plot final bpb against C. This
#   is Chinchilla's "approach 2" and the default convention in efficiency work.
#
#   Read it as: at a fixed training-compute budget, which architecture's
#   best-performing model is better, and how big is that model?
#
# THE BUDGET
#   C = 6.710562e17 active FLOPs. That is MST L=16's existing training budget to six
#   figures (10.5 x 111,235,072 scaling params x 5.7455e8 active FLOPs/token), so that
#   run is already a point on the profile and costs nothing to reuse. Six new runs at
#   6.71e17 each is 4.0e18 total, about 0.8x of one L=24 run. isoFLOP cost does not
#   depend on which depths you pick, only on how many.
#
#   A second contour at C=4.8e18 (L=24's budget) would reuse MST L=24 the same
#   way, but costs ~6x an L=24 run. Only worth it if this one is favourable.
#
# ACTIVE, NOT TOTAL
#   --target-active-flops, not --target-flops. base_train's --target-flops
#   divides by estimate_flops()[0], the TOTAL count, while the Pareto plots use
#   [1], the active one. For MST those differ by 1.34x at L=24, so using the
#   total would put MST on a tighter contour than dense and bias the profile
#   against it without anyone noticing.
#
# EXPECTED SHAPE OF THE ANSWER
#   MST's training-FLOPs multiplier on the ladder is 1.08-1.14x, and the bpb
#   gap on that axis is 0.003-0.005, so a small separation is the honest
#   prediction. State that before looking. This profile is for credibility,
#   not for a big number; if you want the big number it is on the inference
#   axis and p13_isodata.sh is the run that gets it.
#
#   bash scripts/p12_isoflop.sh
#   FLOPS=4.8e18 bash scripts/p12_isoflop.sh
# ============================================================================
set -o pipefail

FORCE=0; SEEDS=1; ARMS=all; CLI_DEPTHS=(); TIMER=0
usage() {
    echo "usage: $0 [--force] [--seeds N] [--arms dense|mst|mol|all] [--timer-only] [depth ...]"
    echo "  --timer-only runs TIMER_STEPS (default 20) steps of every arm and projects the"
    echo "  full sweep from the measured dt, including startup and final-validation time."
    echo "  depths given positionally replace the built-in list for whichever arms run,"
    echo "  so '--mst-only 24' runs exactly one arm and nothing else."
}
while [[ $# -gt 0 ]]; do
    case "$1" in
        --force) FORCE=1; shift ;;
        --seeds) SEEDS="$2"; shift 2 ;;
        # Split the profile across machines: run the dense arms in one place and the
        # MST arms in another. Each half keeps its own state file, so pointing both at
        # the same OUT_BASE (a shared volume) merges them; pointing them at different
        # ones keeps them independent and the halves are combined when plotting.
        --arms) ARMS="$2"; shift 2 ;;
        --dense-only) ARMS=dense; shift ;;
        --mst-only) ARMS=mst; shift ;;
        --mol-only) ARMS=mol; shift ;;
        --timer-only) TIMER=1; shift ;;
        -*) echo "unknown arg: $1"; usage; exit 1 ;;
        *) CLI_DEPTHS+=("$1"); shift ;;
    esac
done
for d in "${CLI_DEPTHS[@]}"; do
    [[ "$d" =~ ^[0-9]+$ ]] || { echo "depth must be a positive integer, got '$d'"; usage; exit 1; }
done
[ "$ARMS" = "both" ] && ARMS=all          # accepted for compatibility with earlier invocations
case "$ARMS" in
    all|dense|mst|mol) ;;
    *) echo "--arms must be one of: dense, mst, mol, all (got '$ARMS')"; exit 1 ;;
esac

# C is pinned by two constraints that leave almost no freedom.
#
# 1. STRADDLING. A depth set brackets its optimum when C falls between the own-budgets of
#    its 2nd and 3rd points. That gives MST 24/28/32/36 the window [7.69e18, 1.56e19] and
#    dense 18/20/22/24 the window [1.29e19, 2.21e19]; the overlap is [1.29e19, 1.56e19]
#    and 1.42e19 is its geometric centre. Outside that window one arm sits entirely on one
#    side of its parabola and its vertex becomes an extrapolation.
#
# 2. THE CORPUS. One shard holds ~252.8M characters, so MAX_SHARDS=300 is roughly 18B
#    tokens and the protocol is single-epoch. The cheapest-per-token arm fixes the ceiling:
#    at 1.42e19 the binding arm is MST L=24 at 9.6B tokens, 0.53 epochs, comfortable.
#    (The previous 9.0e18 was capped by MST L=16 at 0.87 epochs. Dropping L=16 is what
#    made a larger budget possible at all: it lifted the MST ceiling from 1.03e19 to
#    2.68e19.)
#
# own budget / C at 1.42e19:
#   dense  L=18 0.51x  L=20 0.91x  L=22 1.56x  L=24 2.53x   (702M .. 1.38B params)
#   MST    L=24 0.25x  L=28 0.54x  L=32 1.10x  L=36 2.08x   (879M .. 2.09B params)
# Both straddle, dense between L=20 and L=22, MST between L=28 and L=32.
# Cost: 8 arms x 1.42e19 = 1.14e20. Run --timer-only before committing.
FLOPS="${FLOPS:-1.42e19}"
N_SUBS="${N_SUBS:-4}"
ASPECT_RATIO="${ASPECT_RATIO:-64}"
# Dense L=18/20/22/24 spans 702M to 1.38B and straddles a vertex between L=20 and L=22.
# L=16 was dropped when C rose to 1.42e19: at 0.27x its own budget it is heavily
# overtrained and adds nothing the other three do not already cover.
DENSE_DEPTHS="${DENSE_DEPTHS-18 20 22 24}"
# MST needs mst_sub_head_dim (64) to divide sub_dim = D/N, i.e. D a multiple of 256.
# L=24,28,32,36 give D=1536,1792,2048,2304 -> d=384,448,512,576, all divisible by 64.
# L=18,22,26,30,34 do NOT and have no MST arm at all.
#
# L=16 is dropped: it is the cheapest-per-token MST arm and capped the whole profile at
# C=1.03e19, and at 1.42e19 it would need 24.7B tokens, past a single epoch of the corpus.
# L=20 stays excluded as MST's known off-trend ladder point.
#
# ALIGNMENT. sub_dim is a multiple of 128 only at L=8,16,24,32,40; L=28 (448) and L=36
# (576) are 64 mod 128 and miss tensor-core alignment, which cost L=28 an off-trend MFU of
# 18.7% against 21.7% at the smaller L=24. There is no four-point aligned set in this
# range: 24/32/40 is aligned but L=40's own budget is 5.3e19, far outside any affordable C,
# and 16/24/32/40 reintroduces the corpus cap. So two of four MST points are misaligned by
# necessity. Check their residuals before trusting the fitted vertex, and if either lands
# off the curve, fit 24/32 plus whichever of 28/36 is clean.
#
# Note the shared-FFN headline does NOT remove this: the shared FFN is only 16% of MST's
# matrix parameters at L=32, and the other 84% (attention qkv, output projection, the
# transition) is still block-diagonal at K = sub_dim.
MST_DEPTHS="${MST_DEPTHS-24 28 32 36}"
# MoL (Ternovtsii & Bilak 2026) as its own arm, in the 1+3of15 topology at d_thin = D/4
# that reproduces their published parameter counts exactly. Its thin blocks are wrapped in
# per-block W_down/W_up, so it costs far more per token than MST at equal depth (8.09e8
# against 5.75e8 at L=16); 8/12/16 therefore brackets the same budget that 12/16/24 brackets
# for MST. d_thin must be divisible by mol_head_dim 64, which rules out L=10, 14, 18, 22.
# Per-block value embeddings are ON, matching MST's --mst-per-stream-ve: each arm runs the
# configuration its own paper proposes, which is the comparison tab:isotoken16 already makes.
MOL_DEPTHS="${MOL_DEPTHS-8 12 16}"
# Positional depths override the built-in lists, so a single arm can be launched on its
# own machine. Applied before the --arms filter so "--mst-only 24" means exactly that.
if [ ${#CLI_DEPTHS[@]} -gt 0 ]; then
    DENSE_DEPTHS="${CLI_DEPTHS[*]}"
    MST_DEPTHS="${CLI_DEPTHS[*]}"
    MOL_DEPTHS="${CLI_DEPTHS[*]}"
fi
# MoL is parked: its per-block W_down/W_up wrappers make it far slower to train than
# either baseline at these sizes, so it is excluded from the default sweep rather than
# deleted. "--arms mol" and "--mol-only" still run it with the depths above.
case "$ARMS" in
    dense) MST_DEPTHS="";   MOL_DEPTHS="" ;;
    mst)   DENSE_DEPTHS=""; MOL_DEPTHS="" ;;
    mol)   DENSE_DEPTHS=""; MST_DEPTHS=""  ;;
    all)   MOL_DEPTHS="" ;;
esac
OUT_BASE="${OUT_BASE:-out/p12_isoflop}"
mkdir -p "$OUT_BASE"
TIMER_STEPS="${TIMER_STEPS:-20}"

# A long arm killed mid-training (an OOM daemon's SIGTERM, a host preemption) resumes
# from its last periodic checkpoint, so retrying it in place is far cheaper than losing
# the point. RETRY_MIN_SECONDS separates a kill from a genuine config error: anything
# that dies faster than this never trained, so retrying it would only spin.
RETRIES="${RETRIES:-3}"
RETRY_MIN_SECONDS="${RETRY_MIN_SECONDS:-600}"
RETRY_DELAY="${RETRY_DELAY:-30}"
if [ "$TIMER" -eq 1 ]; then
    # A costing pass must not touch the real sweep: its own out tree, its own state file,
    # and no mark_done, so a later real run still sees every arm as outstanding.
    OUT_BASE="${OUT_BASE}/timer"
    mkdir -p "$OUT_BASE"
    echo "TIMER-ONLY: ${TIMER_STEPS} steps per arm, writing to ${OUT_BASE}"
fi
LOGFILE="${SWEEP_LOG:-${OUT_BASE}/p12.log}"

# Compiled-kernel caches, following runpod_env.sh and p30/p33/p35. Without these each
# arm compiles from an empty cache, and on an ephemeral runner every relaunch pays the
# full cost again: MST compiles far more kernels than dense (N=4 streams x 4 window
# scales, per-stream value embeddings, block-diagonal GEMMs), so it is the arm that
# suffers. Defaulting them under OUT_BASE means pointing OUT_BASE at a persistent
# volume also persists the caches, and keeps them inside the gitignored out/ tree.
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${OUT_BASE}/.inductor_cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${OUT_BASE}/.triton_cache}"

# Inductor's compile-worker pool defaults to min(32, nproc) subprocesses, each holding
# its own torch import. On a many-core box with a memory ceiling that is enough RSS to
# get a worker OOM-killed, and the parent then waits on a future that never resolves.
# MST drives the pool far harder than dense (~1200 kernels at L=24 against ~400), which
# is why it is the arm that hangs. Cap it: this costs no step time, unlike
# --compile-regional, which is left off by default because it loses cross-layer fusion.
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-8}"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"

# Arm boundaries and verdicts have to reach the log file, not just stdout. Only the
# sweep command is piped through tee, so without this the log is undelimited training
# output and the structure survives only in whatever captured stdout, which on a remote
# runner is a different place from the volume the log is written to.
log() { echo "$*" | tee -a "$LOGFILE"; }
STATE="${OUT_BASE}/p12_state.json"
[ "$FORCE" -eq 1 ] && rm -f "$STATE"
[ -f "$STATE" ] || echo '{"completed":{}}' > "$STATE"

# Interrupt handling. Without this an INT lands on the foreground process group, kills
# the arm's torchrun, and the loop reads the nonzero status as "this arm failed" and
# starts the NEXT multi-hour arm. One Ctrl-C then costs a run rather than stopping one.
ABORT=0
ALL_ARMS=()
FAILED_ARMS=()
on_signal() {
    ABORT=1
    printf '\n>>> interrupt received. Stopping. The current arm keeps its checkpoints\n'
    printf '>>> and resumes from its last step when this script is re-run.\n'
}
trap on_signal INT TERM

done_already() {
    [ "$FORCE" -eq 1 ] && return 1
    python3 -c "
import json,sys
sys.exit(0 if '$1' in json.load(open('$STATE')).get('completed',{}) else 1)" 2>/dev/null
}
mark_done() {
    python3 -c "
import json,datetime
s=json.load(open('$STATE'))
s.setdefault('completed',{})['$1']=datetime.datetime.now().isoformat()
json.dump(s,open('$STATE','w'),indent=2)"
}

# --- timer-only projection -------------------------------------------------
# Per arm: measured wall time covers startup (env, data, compile) + TIMER_STEPS steps +
# the final eval and save. Splitting off the measured steps leaves the fixed overhead,
# and the full run is that overhead plus full_iterations x dt. base_train prints
# "TIMING_PROBE full_iterations=... " before training so the real horizon is known even
# though the loop stops early.
TIMER_TOTAL=0
TIMER_ROWS=()
project_arm() {                           # project_arm <tag> <log> <t_start> <t_end>
    local tag="$1" alog="$2"
    local elapsed
    elapsed=$(awk "BEGIN{printf \"%.2f\", $4 - $3}")
    # base_train emits TIMING_PROBE_RESULT at the end of every probe. Prefer it: the
    # human step line is printed or not depending on --log-every and where the loop exits.
    local res
    res=$(grep -oE 'TIMING_PROBE_RESULT [^|]*' "$alog" 2>/dev/null | tail -1)
    local full dt timed
    if [ -n "$res" ]; then
        dt=$(printf '%s' "$res"   | grep -oE 'dt_ms=[0-9.]+'          | cut -d= -f2)
        timed=$(printf '%s' "$res"| grep -oE 'timed_steps=[0-9]+'     | cut -d= -f2)
        full=$(printf '%s' "$res" | grep -oE 'full_iterations=[0-9]+' | cut -d= -f2)
        [ "${timed:-0}" -lt 1 ] && dt=""     # no post-warmup step was timed
    fi
    [ -z "$full" ] && full=$(grep -oE 'TIMING_PROBE full_iterations=[0-9]+' "$alog" 2>/dev/null | tail -1 | grep -oE '[0-9]+$')
    # Fallback: base_train announces the horizon before training even without the probe flag.
    [ -z "$full" ] && full=$(grep -oE 'Calculated number of iterations from target ACTIVE FLOPs: [0-9,]+' "$alog" 2>/dev/null \
                             | tail -1 | grep -oE '[0-9,]+$' | tr -d ',')
    [ -z "$dt" ] && dt=$(grep -oE 'dt: [0-9.]+ *ms' "$alog" 2>/dev/null | tail -1 | grep -oE '[0-9.]+')
    if [ -z "$full" ] || [ -z "$dt" ]; then
        log "TIMER $tag: could not parse (full_iterations='${full:-?}' dt='${dt:-?}'); measured ${elapsed}s only"
        if [ ! -s "$alog" ]; then
            log "   arm log ${alog} is missing or empty"
        else
            log "   arm log ${alog} ($(wc -l < "$alog") lines)"
            grep -q 'TIMING_PROBE' "$alog" \
                || log "   --timing-probe-steps did not reach base_train (no TIMING_PROBE line)"
            grep -qE 'dt: [0-9.]+ *ms' "$alog" \
                || log "   the arm never logged a training step"
            # Surface the failure itself rather than making the caller go and find it.
            local marks
            marks=$(grep -nEi 'traceback|error|assert|out of memory|killed|Exception|abort' "$alog" \
                    | tail -6 | cut -c1-160)
            if [ -n "$marks" ]; then
                log "   failure markers in the arm log:"
                while IFS= read -r l; do log "     $l"; done <<< "$marks"
            fi
            log "   last 20 lines of the arm log:"
            while IFS= read -r l; do log "     $(printf '%s' "$l" | cut -c1-160)"; done < <(tail -20 "$alog")
        fi
        TIMER_ROWS+=("$tag|?|?|?|?")
        return
    fi
    local overhead proj
    # clamp: dt is the steady-state step, so the first few steps can exceed it and drive
    # the residual negative on a fast arm. Overhead is never less than zero.
    overhead=$(awk "BEGIN{o=$elapsed - $TIMER_STEPS * $dt/1000; if(o<0)o=0; printf \"%.1f\", o}")
    proj=$(awk "BEGIN{printf \"%.1f\", $overhead + $full * $dt/1000}")
    TIMER_TOTAL=$(awk "BEGIN{printf \"%.1f\", $TIMER_TOTAL + $proj}")
    log "TIMER $tag: ${full} steps x ${dt}ms + ${overhead}s overhead = $(awk "BEGIN{printf \"%.2f\", $proj/3600}")h"
    TIMER_ROWS+=("$tag|$overhead|$full|$dt|$proj")
}

COMMON="--device-batch-size ${DEVICE_BATCH_SIZE:-32} --total-batch-size -1 \
  --use-onecycle 0 --log-every ${LOG_EVERY:-200} --skip-core \
  --data-dir ${DATA_DIR:-data} --tokenizer-dir ${TOKENIZER_DIR:-tokenizer} \
  --sequence-len 2048 --target-param-data-ratio 10.5 \
  --warmup-ratio 0.005 --warmdown-ratio 0.65 --final-lr-frac 0.05 \
  --research-dim -1 --target-active-params 0 --target-tokens -1 \
  --compile-regional ${COMPILE_REGIONAL:-0} \
  --save-every 200 --eval-every -1 --target-active-flops ${FLOPS}"
[ -n "${MAX_SHARDS:-}" ] && COMMON="$COMMON --max-shards $MAX_SHARDS"

run() {
    local tag="$1"; shift
    local depth="$1"; shift
    for s in $(seq 1 "$SEEDS"); do
        local t="${tag}_s${s}"
        ALL_ARMS+=("$t")
        [ "$ABORT" -eq 1 ] && continue
        if done_already "$t"; then log "SKIP $t"; continue; fi
        log ""; log "=== $t (depth $depth, C=${FLOPS} active FLOPs) ==="
        local dir="${OUT_BASE}/${t}"
        [ "$FORCE" -eq 1 ] && rm -rf "$dir"
        # research_compare.py keeps its OWN sweep_state.json inside the arm dir and skips
        # any model it has already recorded as completed. A probe trains 12 steps and
        # saves, so the arm is marked complete and every later probe skips training
        # entirely, leaving a log that stops at the depth banner. A costing pass is
        # throwaway, so start it from a clean directory every time.
        [ "$TIMER" -eq 1 ] && rm -rf "$dir"
        local rc=0 attempt=0
        local t_start t_end elapsed
        local armlog="${dir}.probe.log"
      while :; do
        rc=0
        t_start=$(date +%s.%N)
        if [ "$TIMER" -eq 1 ]; then
            # A real pipeline, not a variable that expands to "|": bash parses redirections
            # before expanding, so the old ${TIMER:+| tee ...} became literal filenames.
            # Deliberately quiet on the terminal: a costing pass should print its TIMER
            # lines and nothing else. Everything still lands in the arm log and $LOGFILE.
            bash scripts/research_sweep.sh $COMMON --timing-probe-steps "$TIMER_STEPS" \
                 --out-dir "$dir" --seed "$s" "$@" "$depth" 2>&1 \
                 | tee "$armlog" >> "$LOGFILE" || rc=$?
        else
            bash scripts/research_sweep.sh $COMMON --out-dir "$dir" --seed "$s" \
                 "$@" "$depth" 2>&1 | tee -a "$LOGFILE" || rc=$?
        fi
        t_end=$(date +%s.%N)
        elapsed=$(awk -v a="$t_start" -v b="$t_end" 'BEGIN{printf "%.0f", b-a}')
        [ "$TIMER" -eq 1 ] && project_arm "$t" "$armlog" "$t_start" "$t_end"
        if [ "$ABORT" -eq 1 ] || [ "$rc" -eq 130 ] || [ "$rc" -eq 143 ]; then
            ABORT=1
            log "INTERRUPTED $t  (resumes from its last checkpoint on the next run)"
            break
        fi
        # An arm counts as complete only if it left the result row this profile reads.
        # research_sweep.sh can exit 0 without training anything, because its own
        # per-model state may already believe the models are finished; marking that
        # done would drop a point from the profile with no error anywhere.
        if [ "$TIMER" -eq 1 ]; then
            [ "$rc" -eq 0 ] || { FAILED_ARMS+=("$t"); log "FAIL $t (rc=$rc)"; }
            break
        elif [ "$rc" -eq 0 ] && [ -f "${dir}/depth_${depth}/results_depth_${depth}.tsv" ]; then
            mark_done "$t"; log "OK $t"
            break
        fi
        # Failed. A long arm that dies mid-training is almost always the host killing
        # the worker (SIGTERM from an OOM daemon, a preemption), not a bad config:
        # training resumes from the last periodic checkpoint, so a retry costs one
        # recompile and at most SAVE_EVERY steps of lost work. Retry it in place
        # rather than dropping the point and waiting for a human to re-invoke.
        # An arm that dies quickly is a real error -- a bad flag, a missing shard --
        # and retrying that only spins, so require the attempt to have run a while.
        if [ "$attempt" -lt "$RETRIES" ] && [ "$elapsed" -ge "$RETRY_MIN_SECONDS" ]; then
            attempt=$((attempt + 1))
            log "RETRY $t (rc=$rc, attempt $attempt/$RETRIES, ran ${elapsed}s; resuming from last checkpoint)"
            sleep "$RETRY_DELAY"
            # The trap can fire during that sleep; without this the retry would
            # relaunch an arm the user has just asked to stop.
            if [ "$ABORT" -eq 1 ]; then
                log "INTERRUPTED $t  (resumes from its last checkpoint on the next run)"
                break
            fi
            continue
        fi
        FAILED_ARMS+=("$t")
        if [ "$attempt" -gt 0 ]; then
            log "FAIL $t (rc=$rc, gave up after $attempt retries)"
        elif [ "$elapsed" -lt "$RETRY_MIN_SECONDS" ]; then
            log "FAIL $t (rc=$rc after only ${elapsed}s -- looks like a config error, not a kill; not retrying)"
        else
            log "FAIL $t (rc=$rc)"
        fi
        break
      done
    done
}

# The MST arm is the paper's headline SP2_k1 config. NOT the S=1 variant: the point of
# this profile is to validate the configuration the ladder is built on, and a profile
# run on a different config validates nothing. The two are Pareto-equivalent anyway
# (measured neutral at L=8 and, after correcting for its token excess, at L=16), and
# the control has the lower FLOPs/token, so it is both the cheaper run and the stronger
# inference-cost position.
mol_config() {                            # mol_config <depth>
    local D=$(( (($1 * ASPECT_RATIO + 127) / 128) * 128 ))
    echo "--use-mol 1 --models base --mol-n-blocks 15 --mol-n-shared 1 --mol-topk 3 \
      --mol-thin-dim $(( D / 4 )) --mol-head-dim 64 --mol-ffn-mult 4.0 \
      --mol-router-aux 0.05 --mol-routed-attn softmax --mol-dispatch 1 \
      --mol-per-block-ve 1"
}

mst_config() {                            # mst_config <depth>
    # The headline is now the SHARED-FFN variant (p16's shared_d arm), not the top-1
    # gated one. Measured against the gated version at matched depth, on meta device:
    #   active FLOPs/token  0.999x   active params 0.9999x   matrix params 0.672x
    # so it reaches the same compute and the same active footprint with a third fewer
    # matrices, and it does so with no router, no load-balance auxiliary, no
    # straight-through estimator and no gating multiply. Its total/active matrix ratio is
    # 1.00 against the gated version's 1.48 and MoL's 3.74.
    #
    # FFN_INNER_DIM defaults to sub_dim, which is what makes the FLOPs match: a shared
    # D -> d -> D FFN costs the same per token as four gated d -> 4d -> d ones at top-1.
    # Set --mst-stream-topk 0 with it; there is nothing left to gate.
    local D=$(( (($1 * ASPECT_RATIO + 127) / 128) * 128 ))
    local SD=$(( D / N_SUBS ))
    local INNER="${FFN_INNER_DIM:-$SD}"
    echo "--use-mst 1 --models base --mst-n-subs $N_SUBS --mst-sub-dim $SD \
      --mst-head-dim 0 --mst-input-mode learned_proj \
      --mst-routing-mode soft_weighted --mst-routing-topk 0 \
      --mst-ffn-mode shared_dense --mst-ffn-inner-dim $INNER \
      --mst-transition-mode aggregate_distribute \
      --mst-final-mode concat_proj --mst-final-topk 0 \
      --mst-routing-aux-weight 0.01 --mst-diversity-weight 0.0 \
      --mst-grad-equalize 1 --mst-block-diagonal-muon 1 \
      --mst-transition-width-mult ${N_SUBS}.0 --mst-sub-lr-scale 2.0 \
      --mst-multi-scale-windows 1 \
      --mst-sub-head-dim 64 --mst-per-stream-ve 1 --mst-compose-windows 1 \
      --mst-wo-mode dense --mst-stream-topk 0"
}

# The previous headline, kept so the two can be run against each other on one budget:
#   MST_GATED=1 bash scripts/p12_isoflop.sh --mst-only
mst_gated_config() {                      # mst_gated_config <depth>
    local D=$(( (($1 * ASPECT_RATIO + 127) / 128) * 128 ))
    local SD=$(( D / N_SUBS ))
    echo "--use-mst 1 --models base --mst-n-subs $N_SUBS --mst-sub-dim $SD \
      --mst-head-dim 0 --mst-input-mode learned_proj \
      --mst-routing-mode soft_weighted --mst-routing-topk 0 --mst-ffn-mode standard \
      --mst-transition-mode aggregate_distribute \
      --mst-final-mode concat_proj --mst-final-topk 0 \
      --mst-routing-aux-weight 0.01 --mst-diversity-weight 0.0 \
      --mst-grad-equalize 1 --mst-block-diagonal-muon 1 \
      --mst-transition-width-mult ${N_SUBS}.0 --mst-sub-lr-scale 2.0 \
      --mst-multi-scale-windows 1 \
      --mst-sub-head-dim 64 --mst-per-stream-ve 1 --mst-compose-windows 1 \
      --mst-wo-mode dense --mst-stream-topk 1 --mst-stream-router-noise 1.0"
}


echo "============================================================"
echo "  P12 isoFLOP profile   C = ${FLOPS} active FLOPs"
echo "  arms: ${ARMS}"
echo "  dense: ${DENSE_DEPTHS:-(none)}   MST: ${MST_DEPTHS:-(none)}   MoL: ${MOL_DEPTHS:-(none)}"
echo "  out ${OUT_BASE}"
echo "============================================================"

for d in $DENSE_DEPTHS; do
    run "ISOF_dense_d${d}" "$d" --models base
done
for d in $MST_DEPTHS; do
    SD=$(( (((d * ASPECT_RATIO + 127) / 128) * 128) / N_SUBS ))
    if [ $(( SD % 64 )) -ne 0 ]; then
        echo "SKIP MST d${d}: sub_dim ${SD} not divisible by mst_sub_head_dim 64"
        continue
    fi
    if [ "${MST_GATED:-0}" -eq 1 ]; then
        run "ISOF_mstgated_d${d}" "$d" $(mst_gated_config "$d")
    else
        run "ISOF_mst_d${d}" "$d" $(mst_config "$d")
    fi
done
for d in $MOL_DEPTHS; do
    TD=$(( (((d * ASPECT_RATIO + 127) / 128) * 128) / 4 ))
    if [ $(( TD % 64 )) -ne 0 ]; then
        echo "SKIP MoL d${d}: thin_dim ${TD} not divisible by mol_head_dim 64"
        continue
    fi
    run "ISOF_mol_d${d}" "$d" $(mol_config "$d")
done

echo ""
echo "============================================================"
if [ "$TIMER" -eq 1 ]; then
    printf '  %-24s %11s %8s %10s %10s\n' arm "startup(s)" steps "dt(ms)" projected
    for r in "${TIMER_ROWS[@]}"; do
        IFS='|' read -r a m f d pj <<< "$r"
        printf '  %-24s %9ss %8s %10s %9sh\n' "$a" "$m" "$f" "$d" \
            "$(awk "BEGIN{printf \"%.2f\", ${pj:-0}/3600}")"
    done
    echo "  ------------------------------------------------------------------"
    echo "  projected total for the whole sweep: $(awk "BEGIN{printf \"%.2f\", $TIMER_TOTAL/3600}")h"
    echo "  startup(s) = env setup, compile and final validation; projected = startup + steps x dt"
    echo "============================================================"
    [ ${#FAILED_ARMS[@]} -gt 0 ] && { echo "  failed: ${FAILED_ARMS[*]}"; exit 1; }
    exit 0
fi
REMAINING=()
for a in "${ALL_ARMS[@]}"; do done_already "$a" || REMAINING+=("$a"); done
log "  arms complete: $(( ${#ALL_ARMS[@]} - ${#REMAINING[@]} )) / ${#ALL_ARMS[@]}"
[ ${#REMAINING[@]} -gt 0 ] && log "  still to run:  ${REMAINING[*]}"
[ ${#FAILED_ARMS[@]} -gt 0 ] && log "  failed:        ${FAILED_ARMS[*]}"
[ ${#REMAINING[@]} -gt 0 ] && echo "  re-run this script to continue; finished arms are skipped."
echo "============================================================"
echo ""
echo "============================================================"
echo "  done: ${OUT_BASE}/"
echo "  Every run saw the same ${FLOPS} active training FLOPs, so bpb is"
echo "  directly comparable. Plot bpb against depth per arm; the lower"
echo "  envelope of each arm is that architecture's isoFLOP frontier."
echo "============================================================"

[ "$ABORT" -eq 1 ] && exit 130
[ ${#REMAINING[@]} -gt 0 ] && exit 1
exit 0
