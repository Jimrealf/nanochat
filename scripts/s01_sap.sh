#!/usr/bin/env bash
# S01: sampled-in-the-middle refinement (SIR) for the T=L SAP head.
#
# The ten arms are mechanism ablations, not a hyperparameter sweep:
#   sir          hard on-policy draft + leave-one-out reference
#   sir_soft     sparse top-k straight-through bridge
#   sir_compat   multiscale low-rank compatibility messages
#   sir_context  contextual plausible-error soft objective
#   sir_conf     fixed-budget confidence-triggered second draft
#   sir_anchor   sparse sampled skeleton followed by parallel fill
#   sir_pyramid  coarse-to-fine sampled skeletons
#   sir_lattice  top-k chain sum-product before draft sampling
#   sir_energy   structured-negative reference energy
#   sir_full     soft + compatibility + context + confidence + energy
#
# Gates are pre-registered in s01_sap_plan.md. Training is cost matched to the
# dense arm by --target-flops; T=L uses one block start per training sequence.
#
# Local wiring smoke (safe on this machine):
#   bash scripts/s01_sap.sh --smoke
# Synthetic phrase-HMM mechanism test:
#   bash scripts/s01_sap.sh --synthetic
# FineWeb depth-8 sweep (GPU box / Modal container):
#   bash scripts/s01_sap.sh --full 8
# Restrict arms without editing this file:
#   ARMS="sir sir_soft" bash scripts/s01_sap.sh --full --no-post 8
set -o pipefail

SIR_ARMS="sir sir_soft sir_compat sir_context sir_conf sir_anchor sir_pyramid sir_lattice sir_energy sir_full"
ARMS="${ARMS:-$SIR_ARMS}"
STAGE="full"
FORCE=0
POST="${POST:-1}"
SEEDS=1
DEPTH=8
while [[ $# -gt 0 ]]; do
    case "$1" in
        --smoke)     STAGE="smoke"; shift ;;
        --synthetic) STAGE="synthetic"; shift ;;
        --full)      STAGE="full"; shift ;;
        --force)     FORCE=1; shift ;;
        --seeds)     SEEDS="$2"; shift 2 ;;
        --post)      POST=1; shift ;;
        --no-post)   POST=0; shift ;;
        [0-9]*)      DEPTH="$1"; shift ;;
        *) echo "unknown arg: $1"; exit 1 ;;
    esac
done

export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
OUT_BASE="${OUT_BASE:-out/s01_sap}"
mkdir -p "$OUT_BASE"
if [ -z "${PYTHON_BIN:-}" ]; then
    if [ -x /home/seqaeon/Downloads/venv/bin/python ]; then
        PYTHON_BIN=/home/seqaeon/Downloads/venv/bin/python
    else
        PYTHON_BIN=python3
    fi
fi

if [ "$STAGE" = "smoke" ] || [ "$STAGE" = "synthetic" ]; then
    if [ "$STAGE" = "smoke" ]; then
        STEPS="${SYNTH_STEPS:-2}"
        EXTRA_SYNTH="--batch 2 --seq-len 24 --depth 1 --n-embd 64 --vocab 64 --eval-seqs 4 --eval-positions 8 --samples-per-ctx 1 --iwae-samples 1 --log-every 1"
        SYNTH_OUT="${OUT_BASE}/smoke"
    else
        STEPS="${SYNTH_STEPS:-2000}"
        EXTRA_SYNTH="--batch ${SYNTH_BATCH:-32} --seq-len ${SYNTH_SEQ_LEN:-256} --depth ${SYNTH_DEPTH:-3} --n-embd ${SYNTH_EMBD:-128} --vocab ${SYNTH_VOCAB:-512} --eval-seqs ${SYNTH_EVAL_SEQS:-128} --samples-per-ctx ${SYNTH_SAMPLES:-4} --iwae-samples ${SYNTH_IWAE:-32} --log-every ${SYNTH_LOG_EVERY:-250}"
        SYNTH_OUT="${OUT_BASE}/synthetic"
    fi
    # sap_synthetic automatically raises only the three multi-pass arms to the
    # structurally required third layer; the other mechanisms remain two-layer.
    "$PYTHON_BIN" -m scripts.sap_synthetic --modes $ARMS --T 4 --seeds "$SEEDS" \
        --steps "$STEPS" --head-layers 2 --sir-topk "${SIR_TOPK:-16}" \
        --sir-rank "${SIR_RANK:-32}" --sir-anchor-stride "${SYNTH_ANCHOR_STRIDE:-2}" \
        --sir-fine-stride "${SYNTH_FINE_STRIDE:-1}" --sir-refine-frac "${SIR_REFINE_FRAC:-0.25}" \
        --sir-train-samples "${SIR_TRAIN_SAMPLES:-4}" --sir-policy-weight "${SIR_POLICY_WEIGHT:-1.0}" \
        --sir-posterior-mix "${SIR_POSTERIOR_MIX:-0.5}" \
        --sir-draft-weight "${SIR_DRAFT_WEIGHT:-1.0}" \
        --sir-context-weight "${SIR_CONTEXT_WEIGHT:-0.25}" \
        --sir-energy-weight "${SIR_ENERGY_WEIGHT:-0.25}" \
        $EXTRA_SYNTH --out "$SYNTH_OUT" 2>&1 | tee "${OUT_BASE}/s01_${STAGE}.log"
    exit "${PIPESTATUS[0]}"
fi

VOCAB="${VOCAB:-32768}"
TOK="${TOKENIZER_DIR:-tokenizer}"
SEQ_LEN="${SEQ_LEN:-2048}"
WINDOW="${WINDOW:-SSSL}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-16}"
EVAL_TOKENS="${EVAL_TOKENS:-20971520}"
SAP_EVAL_STEPS="${SAP_EVAL_STEPS:-40}"
LOGFILE="${SWEEP_LOG:-${OUT_BASE}/s01_sap_d${DEPTH}_${SEEDS}.log}"
STATE="${OUT_BASE}/s01_state_d${DEPTH}.json"

if [ -z "${NPROC_PER_NODE:-}" ]; then
    NPROC_PER_NODE=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ')
    [ "${NPROC_PER_NODE:-0}" -ge 1 ] 2>/dev/null || NPROC_PER_NODE=1
fi
RUNNER="$PYTHON_BIN -m torch.distributed.run --standalone --nproc_per_node=${NPROC_PER_NODE}"

"$PYTHON_BIN" -c "
import os, sys
from nanochat.tokenizer import get_tokenizer
v = get_tokenizer(tokenizer_dir='$TOK').get_vocab_size()
assert os.path.exists(os.path.join('$TOK', 'token_bytes.pt')), 'token_bytes.pt missing'
sys.exit(0 if v == $VOCAB else f'tokenizer at $TOK has vocab {v}, expected $VOCAB')" || exit 1
if ! ls "${DATA_DIR:-data}"/*.parquet >/dev/null 2>&1; then
    echo "no parquet shards in '${DATA_DIR:-data}'; set DATA_DIR. Nothing was run."
    exit 1
fi

[ "$FORCE" -eq 1 ] && rm -f "$STATE"
[ -f "$STATE" ] || echo '{"completed":{}}' > "$STATE"
DENSE_FLOPS="${DENSE_FLOPS:-$("$PYTHON_BIN" -m scripts.sap_budget --depth "$DEPTH" \
    --ratio "${RATIO:-10.5}" --tokenizer-dir "$TOK" --seq-len "$SEQ_LEN" --window-pattern "$WINDOW") }"
DENSE_FLOPS="${DENSE_FLOPS//[[:space:]]/}"

COMMON="--depth $DEPTH --max-seq-len $SEQ_LEN --window-pattern $WINDOW \
  --device-batch-size $DEVICE_BATCH_SIZE --total-batch-size -1 \
  --target-flops $DENSE_FLOPS --target-param-data-ratio ${RATIO:-10.5} \
  --warmup-ratio 0.005 --warmdown-ratio 0.65 --final-lr-frac 0.05 \
  --eval-every -1 --eval-tokens $EVAL_TOKENS --core-metric-every 0 \
  --sample-every -1 --save-every -1 --log-every ${LOG_EVERY:-100} \
  --data-dir ${DATA_DIR:-data} --tokenizer-dir $TOK --sap-eval-steps $SAP_EVAL_STEPS"
[ -n "${MAX_SHARDS:-}" ] && COMMON="$COMMON --max-shards $MAX_SHARDS"
[ -n "${EXTRA_ARGS:-}" ] && COMMON="$COMMON $EXTRA_ARGS"

done_already() {
    [ "$FORCE" -eq 1 ] && return 1
    "$PYTHON_BIN" -c "import json,sys; sys.exit(0 if '$1' in json.load(open('$STATE')).get('completed',{}) else 1)" 2>/dev/null
}
mark_done() {
    "$PYTHON_BIN" -c "
import datetime,json
p='$STATE'; s=json.load(open(p)); s.setdefault('completed',{})['$1']=datetime.datetime.now().isoformat()
json.dump(s,open(p,'w'),indent=2)"
}
run_arm() {
    local tag="$1"; shift
    for seed in $(seq 1 "$SEEDS"); do
        local seeded="${tag}_s${seed}"
        if done_already "$seeded"; then echo "SKIP  $seeded"; continue; fi
        local dir="${OUT_BASE}/d${DEPTH}"
        [ "$FORCE" -eq 1 ] && rm -rf "${dir:?}/${seeded}"
        echo "--- $seeded | equal budget ${DENSE_FLOPS} FLOPs ---" | tee -a "$LOGFILE"
        if $RUNNER -m scripts.base_train $COMMON --checkpoints-dir "$dir" \
            --model-tag "$seeded" --seed "$seed" "$@" 2>&1 | tee -a "$LOGFILE" "${OUT_BASE}/${seeded}_d${DEPTH}.log"; then
            mark_done "$seeded"
            if [ "$POST" -eq 1 ] && [ "$tag" != "B1_dense" ]; then
                "$PYTHON_BIN" -m scripts.sap_decode_bench --checkpoint-dir "${dir}/${seeded}" \
                    --gen-tokens "${BENCH_TOKENS:-256}" --no-graphs \
                    --out "${OUT_BASE}/decode_${seeded}_d${DEPTH}.json" 2>&1 | tee -a "$LOGFILE"
                "$PYTHON_BIN" -m scripts.sap_eval_generation --checkpoint-dir "${dir}/${seeded}" \
                    --reference-dir "${dir}/B1_dense_s${seed}" --tokenizer-dir "$TOK" \
                    --data-dir "${DATA_DIR:-data}" --n-prefixes "${GEN_PREFIXES:-1024}" \
                    --gen-tokens "${GEN_TOKENS:-128}" \
                    --out "${OUT_BASE}/gen_${seeded}_d${DEPTH}.jsonl" 2>&1 | tee -a "$LOGFILE"
            fi
        else
            echo "FAIL  $seeded" | tee -a "$LOGFILE"
        fi
    done
}

echo "S01 d${DEPTH}: T=L=${SEQ_LEN}; arms: ${ARMS}; seeds=${SEEDS}; post=${POST}" | tee -a "$LOGFILE"
[ "${RUN_DENSE:-1}" -eq 1 ] && run_arm "B1_dense"
FRAC=$("$PYTHON_BIN" -c "print(1.0 / $SEQ_LEN)")
for arm in $ARMS; do
    layers=2
    case "$arm" in sir_conf|sir_pyramid|sir_full) layers=3 ;; esac
    run_arm "SAP_${arm}_TL" --sap-block-t "$SEQ_LEN" --sap-block-mode "$arm" \
        --sap-block-frac "$FRAC" --sap-head-layers "$layers" \
        --sap-sir-topk "${SIR_TOPK:-16}" --sap-sir-rank "${SIR_RANK:-32}" \
        --sap-sir-anchor-stride "${SIR_ANCHOR_STRIDE:-16}" \
        --sap-sir-fine-stride "${SIR_FINE_STRIDE:-4}" \
        --sap-sir-refine-frac "${SIR_REFINE_FRAC:-0.25}" \
        --sap-sir-train-samples "${SIR_TRAIN_SAMPLES:-4}" \
        --sap-sir-policy-weight "${SIR_POLICY_WEIGHT:-1.0}" \
        --sap-sir-posterior-mix "${SIR_POSTERIOR_MIX:-0.5}" \
        --sap-sir-draft-weight "${SIR_DRAFT_WEIGHT:-1.0}" \
        --sap-sir-context-weight "${SIR_CONTEXT_WEIGHT:-0.25}" \
        --sap-sir-energy-weight "${SIR_ENERGY_WEIGHT:-0.25}"
done

echo "S01 complete: $LOGFILE"
