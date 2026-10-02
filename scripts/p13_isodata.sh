#!/usr/bin/env bash
# ============================================================================
# P13: iso-data profile. Every arm sees the SAME tokens, and the baseline is a
# ladder rather than a single point, so a multiplier can be read off directly.
#
# WHY THIS IS THE ONE TO RUN FIRST
#   MST's claim is an inference-cost claim: better quality per FLOP spent at
#   serving time. That is the MoE convention (Mixtral reports 12.9B active of
#   46.7B total), and the standard way to support it is to fix the training
#   data, then compare quality against inference cost.
#
#   Fixing the data removes the objection the compute-optimal ladder invites.
#   Under the ladder's rule MST draws 1.43x the tokens of the dense model
#   matched to it on active FLOPs (p11_active_params.py). Here nobody draws
#   more than anyone else, so the resulting multiplier needs no protocol
#   defence at all.
#
# WHY A DENSE LADDER AND NOT A SINGLE DENSE POINT
#   One dense run at D tokens gives a comparison, not a multiplier: dense L=16
#   will simply beat MST L=16 on bpb while costing 2.76x the FLOPs per token,
#   and neither number alone is the claim. To say "MST reaches this bpb at
#   Nx fewer FLOPs" you need the dense bpb-versus-FLOPs curve AT THIS TOKEN
#   COUNT, which means several dense depths. That is the real price of an
#   unimpeachable comparison and there is no way around it.
#
# THE BUDGET
#   D = 1,167,968,256 tokens, MST L=16's own compute-optimal budget EXACTLY
#   (10.5 x 111,235,072 scaling params), so that run is already a point and costs
#   nothing to reuse. The exact figure matters: total_batch_size is auto-computed from
#   target_tokens and snapped to a power of two, so a rounded budget can land on a
#   different batch size and stop being the same run.
#
#   Cost: dense L=10/12/14/16 at this D is 0.56+0.86+1.29+1.81 = 4.5e18, plus MST L=12
#   and L=24 at 0.37e18 and 1.73e18. About 6.6e18 total, roughly 1.4x one L=24 run.
#
# STATE THE PREDICTION BEFORE LOOKING
#   At L=8 iso-token, MST read 1.061x against dense's 1.000x. On its own budget
#   at L=16 it reads 1.222x. The truth is somewhere between and it is NOT the
#   2.4x FLOPs ratio between the two architectures at equal depth: at that
#   ratio MST is a worse model. Expect roughly 1.1-1.2x.
#
#   bash scripts/p13_isodata.sh
#   TOKENS=3657905664 bash scripts/p13_isodata.sh    # MoL L=16's budget
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

TOKENS="${TOKENS:-1167968256}"
N_SUBS="${N_SUBS:-4}"
ASPECT_RATIO="${ASPECT_RATIO:-64}"
DENSE_DEPTHS="${DENSE_DEPTHS-10 12 14 16}"
# L=20 is deliberately excluded: it is MST's off-trend ladder point (1.125x against
# 1.222x at L=16 and 1.284x at L=24). L=12/16/24 give an over-trained, a
# compute-optimal and an under-trained point at this fixed D, which is the spread a
# fixed-data curve needs.
MST_DEPTHS="${MST_DEPTHS-12 16 24}"
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
OUT_BASE="${OUT_BASE:-out/p13_isodata}"
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
LOGFILE="${SWEEP_LOG:-${OUT_BASE}/p13.log}"

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
STATE="${OUT_BASE}/p13_state.json"
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
  --research-dim -1 --target-active-params 0 \
  --compile-regional ${COMPILE_REGIONAL:-0} \
  --save-every 200 --eval-every -1 --target-tokens ${TOKENS}"
[ -n "${MAX_SHARDS:-}" ] && COMMON="$COMMON --max-shards $MAX_SHARDS"

run() {
    local tag="$1"; shift
    local depth="$1"; shift
    for s in $(seq 1 "$SEEDS"); do
        local t="${tag}_s${s}"
        ALL_ARMS+=("$t")
        [ "$ABORT" -eq 1 ] && continue
        if done_already "$t"; then log "SKIP $t"; continue; fi
        log ""; log "=== $t (depth $depth, ${TOKENS} tokens) ==="
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

# Headline SP2_k1, same reasoning as p12: this profile exists to validate the config
# the ladder is built on. S=1 measured Pareto-neutral against it at L=8 and, after
# correcting its 9.75% token excess, at L=16, so it would not change the answer, and
# the control's lower FLOPs/token makes it both cheaper here and better positioned on
# the inference axis this profile is about.
mol_config() {                            # mol_config <depth>
    local D=$(( (($1 * ASPECT_RATIO + 127) / 128) * 128 ))
    echo "--use-mol 1 --models base --mol-n-blocks 15 --mol-n-shared 1 --mol-topk 3 \
      --mol-thin-dim $(( D / 4 )) --mol-head-dim 64 --mol-ffn-mult 4.0 \
      --mol-router-aux 0.05 --mol-routed-attn softmax --mol-dispatch 1 \
      --mol-per-block-ve 1"
}

mst_config() {                            # mst_config <depth>
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
echo "  P13 iso-data profile   D = ${TOKENS} tokens for every arm"
echo "  arms: ${ARMS}"
echo "  dense: ${DENSE_DEPTHS:-(none)}   MST: ${MST_DEPTHS:-(none)}   MoL: ${MOL_DEPTHS:-(none)}"
echo "  out ${OUT_BASE}"
echo "============================================================"

for d in $DENSE_DEPTHS; do
    run "ISOD_dense_d${d}" "$d" --models base
done
for d in $MST_DEPTHS; do
    SD=$(( (((d * ASPECT_RATIO + 127) / 128) * 128) / N_SUBS ))
    if [ $(( SD % 64 )) -ne 0 ]; then
        echo "SKIP MST d${d}: sub_dim ${SD} not divisible by mst_sub_head_dim 64"
        continue
    fi
    run "ISOD_mst_d${d}" "$d" $(mst_config "$d")
done
for d in $MOL_DEPTHS; do
    TD=$(( (((d * ASPECT_RATIO + 127) / 128) * 128) / 4 ))
    if [ $(( TD % 64 )) -ne 0 ]; then
        echo "SKIP MoL d${d}: thin_dim ${TD} not divisible by mol_head_dim 64"
        continue
    fi
    run "ISOD_mol_d${d}" "$d" $(mol_config "$d")
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
echo "  Fit bpb = a * (active FLOPs/token)^b on the DENSE runs only, then read"
echo "  each MST run's multiplier off that fit. Every arm saw ${TOKENS} tokens,"
echo "  so the fit and the multipliers carry no budget-rule assumption."
echo "============================================================"

[ "$ABORT" -eq 1 ] && exit 130
[ ${#REMAINING[@]} -gt 0 ] && exit 1
exit 0
