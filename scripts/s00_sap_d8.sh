#!/usr/bin/env bash
# ============================================================================
# S00: SAP Stage B, the sampling-aware block head on FineWeb-Edu at depth 8.
#
# WHAT THIS MEASURES
#   One head emits the next T tokens per trunk pass (nanochat/block_head.py).
#   Every arm keeps the ordinary next-token loss on the trunk; the block head
#   rides on top of it on a fraction of the positions. Two numbers per arm:
#
#     next-token bpb   the trunk's quality, the "neutral" half of the bar
#     block bpb        the head's own likelihood of the next T tokens (exact for
#                      indep / local / cp, an upper bound for p1 / p2 / inv_head),
#                      next to the next-token bpb on the SAME tokens
#
#   plus decode tokens/sec and generation quality from the post-run scripts.
#
# WHY --target-flops (with a POSITIVE --target-param-data-ratio)
#   base_train derives the step count from --target-flops when it is set, and the
#   auto batch size from the ratio times the scaling params (transformer matrices +
#   lm_head, which exclude the block head), so every arm gets the same batch size.
#   A negative ratio would make the auto batch size fail.
#
#   A block head costs extra FLOPs per token: T slot rows through the shared
#   V x d unembedding on sap_block_frac of the positions. The claim is a point on
#   the bpb-vs-training-FLOPs curve, so every arm gets the DENSE arm's FLOPs and
#   base_train turns that into fewer tokens for the arms that cost more. The
#   block fraction defaults to 1/16 here: at d8, V=32768 the readout alone makes
#   1/8 cost about +22% FLOPs, against the plan's ~15% budget.
#
# ARMS (per T in $TS)
#   B1  dense                       no block head
#   B2  indep                       the seed idea, independent slots
#   B3  cp                          mixture of R independent-slot components
#   B4  local                       local AR head (sequential inside the head)
#   B5  inv_head                    PTP-style competitor (data-inverted noise)
#   P1  p1_discrete                 plan latent, product-quantised, ELBO
#   P2  p2_gauss                    plan latent, Gaussian, ELBO
#   P3  p3_energy                   energy-score plan sampler (likelihood-free)
#   controls (--with-controls)      plain_noise, wta
#
#   Run only the methods that passed Stage A (scripts/sap_synthetic.py);
#   restrict with ARMS="indep p1_discrete ...".
#
# WHAT WOULD KILL THE DIRECTION (pre-registered, sap_research_plan.md)
#   No method within 5% block-over-next-token bpb at T=2 with >= 1.5x decode
#   speedup at batch 16. There is no analysis-paper fallback.
#
# USAGE (Modal notebook or a rented box; this laptop runs smoke tests only)
#   bash scripts/s00_sap_d8.sh 8                    # all arms, T=2 and 4, 1 seed, post-run evals
#   bash scripts/s00_sap_d8.sh --seeds 3 8          # 3 seeds
#   ARMS="indep p1_discrete" TS="4" bash scripts/s00_sap_d8.sh 8
#   POST=0 bash scripts/s00_sap_d8.sh 8             # training only
#   From a code snapshot next to the data (the Modal volume layout):
#   cd nanochat_sap && DATA_DIR=../data TOKENIZER_DIR=../tokenizer \
#       DEVICE_BATCH_SIZE=64 SWEEP_LOG=s00_sap.log bash scripts/s00_sap_d8.sh 8
#
#   Batch constraint: base_train needs total_batch % (DEVICE_BATCH_SIZE * 2048 * GPUs) == 0.
#   The auto total batch at d8 is 2^18 tokens, so DEVICE_BATCH_SIZE * GPUs <= 128.
# ============================================================================
set -o pipefail

FORCE=0
SEEDS=1
POST="${POST:-1}"
CONTROLS=0
DEPTHS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --force)          FORCE=1; shift ;;
        --seeds)          SEEDS="$2"; shift 2 ;;
        --post)           POST=1; shift ;;
        --no-post)        POST=0; shift ;;
        --with-controls)  CONTROLS=1; shift ;;
        [0-9]*)           DEPTHS+=("$1"); shift ;;
        *) echo "unknown arg: $1"
           echo "usage: $0 [--force] [--seeds N] [--post|--no-post] [--with-controls] [DEPTH ...]"
           exit 1 ;;
    esac
done
[ ${#DEPTHS[@]} -eq 0 ] && DEPTHS=(8)

VOCAB="${VOCAB:-32768}"
TOK="${TOKENIZER_DIR:-tokenizer}"
TS="${TS:-2 4}"
# p3_energy was killed in Stage A (LEARNINGS 2026-10-01); add it back with ARMS=... to rerun it.
ARMS="${ARMS:-indep cp local inv_head p1_discrete p2_gauss}"
LATENT_CODES="${LATENT_CODES:-64}"   # P1: Stage A's diagnostic found C=64 beats C=16 (capacity-limited)
[ "$CONTROLS" -eq 1 ] && ARMS="$ARMS plain_noise wta"
FRAC="${FRAC:-0.0625}"
CP_FRAC="${CP_FRAC:-0.015625}"   # R readouts per slot row; keeps cp's head cost near the others'
SEQ_LEN="${SEQ_LEN:-2048}"
WINDOW="${WINDOW:-SSSL}"
OUT_BASE="${OUT_BASE:-out/s00_sap}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-16}"
if [ -z "${NPROC_PER_NODE:-}" ]; then
    NPROC_PER_NODE=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ')
    [ "${NPROC_PER_NODE:-0}" -ge 1 ] 2>/dev/null || NPROC_PER_NODE=1
fi
# Run THIS checkout's code even if another copy of nanochat is pip-installed in editable mode.
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
EVAL_TOKENS="${EVAL_TOKENS:-20971520}"
SAP_EVAL_STEPS="${SAP_EVAL_STEPS:-40}"

if command -v torchrun &> /dev/null; then
    RUNNER="torchrun --standalone --nproc_per_node=${NPROC_PER_NODE}"
else
    RUNNER="python3 -m torch.distributed.run --standalone --nproc_per_node=${NPROC_PER_NODE}"
fi

# Check, do not build: ensure_tokenizer would also compute a frequency table from the corpus,
# which this sweep never reads. A V=265 stub (the tokenizer clobber in LEARNINGS) fails here.
if ! python3 -c "
import sys, os
from nanochat.tokenizer import get_tokenizer
v = get_tokenizer(tokenizer_dir='$TOK').get_vocab_size()
assert os.path.exists(os.path.join('$TOK', 'token_bytes.pt')), 'token_bytes.pt missing'
sys.exit(0 if v == $VOCAB else f'tokenizer at $TOK has vocab {v}, expected $VOCAB')"; then
    echo "tokenizer check failed for '${TOK}'; nothing was run."
    exit 1
fi
if ! ls "${DATA_DIR:-data}"/*.parquet > /dev/null 2>&1; then
    echo "no parquet shards in '${DATA_DIR:-data}'; set DATA_DIR. Nothing was run."
    exit 1
fi
mkdir -p "$OUT_BASE"

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

for DEPTH in "${DEPTHS[@]}"; do

LOGFILE="${SWEEP_LOG:-${OUT_BASE}/s00_d${DEPTH}.log}"
STATE="${OUT_BASE}/s00_state_d${DEPTH}.json"
[ "$FORCE" -eq 1 ] && rm -f "$STATE"
[ -f "$STATE" ] || echo '{"completed":{}}' > "$STATE"

DENSE_FLOPS="${DENSE_FLOPS:-$(python3 -m scripts.sap_budget --depth "$DEPTH" --ratio "${RATIO:-10.5}" \
    --tokenizer-dir "$TOK" --seq-len "$SEQ_LEN" --window-pattern "$WINDOW")}"

COMMON="--depth $DEPTH --max-seq-len $SEQ_LEN --window-pattern $WINDOW \
  --device-batch-size $DEVICE_BATCH_SIZE --total-batch-size -1 \
  --target-flops $DENSE_FLOPS --target-param-data-ratio ${RATIO:-10.5} \
  --warmup-ratio 0.005 --warmdown-ratio 0.65 --final-lr-frac 0.05 \
  --eval-every ${EVAL_EVERY:--1} --eval-tokens $EVAL_TOKENS \
  --core-metric-every ${CORE_EVERY:-0} --sample-every -1 --save-every -1 \
  --log-every ${LOG_EVERY:-100} --data-dir ${DATA_DIR:-data} --tokenizer-dir $TOK \
  --sap-eval-steps $SAP_EVAL_STEPS"
[ -n "${MAX_SHARDS:-}" ] && COMMON="$COMMON --max-shards $MAX_SHARDS"
# Extra base_train flags for every arm, e.g. EXTRA_ARGS="--no-compile" or a pinned
# --total-batch-size for a smoke run (the auto batch size is identical across arms anyway).
[ -n "${EXTRA_ARGS:-}" ] && COMMON="$COMMON $EXTRA_ARGS"

run() {
    local tag="$1"; shift
    for s in $(seq 1 "$SEEDS"); do
        local t="${tag}_s${s}"
        if done_already "$t"; then echo "SKIP  $t (already completed)"; continue; fi
        echo ""
        echo "--- $t  (depth $DEPTH, V=${VOCAB}, target FLOPs ${DENSE_FLOPS}) ---"
        local dir="${OUT_BASE}/d${DEPTH}"
        [ "$FORCE" -eq 1 ] && rm -rf "${dir:?}/${t}"
        if $RUNNER -m scripts.base_train $COMMON --checkpoints-dir "$dir" --model-tag "$t" \
               --seed "$s" "$@" 2>&1 | tee -a "$LOGFILE" "${OUT_BASE}/${t}_d${DEPTH}.log"; then
            mark_done "$t"; echo "OK    $t"
            if [ "$POST" -eq 1 ] && [ "$tag" != "B1_dense" ]; then
                python3 -m scripts.sap_decode_bench --checkpoint-dir "${dir}/${t}" \
                    --gen-tokens "${BENCH_TOKENS:-256}" \
                    --out "${OUT_BASE}/decode_${t}_d${DEPTH}.json" 2>&1 | tee -a "$LOGFILE"
                python3 -m scripts.sap_eval_generation --checkpoint-dir "${dir}/${t}" \
                    --reference-dir "${dir}/B1_dense_s${s}" --tokenizer-dir "$TOK" \
                    --data-dir "${DATA_DIR:-data}" --n-prefixes "${GEN_PREFIXES:-1024}" \
                    --gen-tokens "${GEN_TOKENS:-128}" \
                    --out "${OUT_BASE}/gen_${t}_d${DEPTH}.jsonl" 2>&1 | tee -a "$LOGFILE"
            fi
        else
            echo "FAIL  $t (will retry on the next invocation)"
        fi
    done
}

echo "============================================================"
echo "  S00: SAP block head, V=${VOCAB}, depth ${DEPTH}, T in {${TS}}"
echo "  arms: ${ARMS}"
echo "  block fraction ${FRAC} (cp ${CP_FRAC}), P1 codes ${LATENT_CODES}, GPUs ${NPROC_PER_NODE}"
echo "  target FLOPs ${DENSE_FLOPS} per arm (the dense arm's)   post-run evals: ${POST}"
echo "============================================================"

run "B1_dense"
for T in $TS; do
    for arm in $ARMS; do
        frac="$FRAC"
        [ "$arm" = "cp" ] && frac="$CP_FRAC"
        run "SAP_${arm}_T${T}" --sap-block-t "$T" --sap-block-mode "$arm" --sap-block-frac "$frac" \
            --sap-latent-codes "$LATENT_CODES"
    done
done

done

echo ""
echo "============================================================"
echo "  S00 complete. Per-arm lines to read:"
echo "    'Validation bpb'   next-token bpb (compare with B1_dense at equal FLOPs)"
echo "    'SAP_EVAL_JSON'    block bpb, next-token bpb on the same tokens, latent sensitivity"
echo "  With --post: decode_*.json (tokens/sec at batch 1/16/128) and gen_*.jsonl"
echo "  (generation pairs for the LLM judge; reference-model perplexity inside)."
echo "============================================================"
