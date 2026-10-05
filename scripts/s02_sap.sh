#!/usr/bin/env bash
# S02 exact-gate reproduction. The Modal entry point runs the two convergence widths and
# the shallow/full correlated-tree controls concurrently.
set -euo pipefail

case "${1:---smoke}" in
    --smoke)
        PYTHON_BIN="${PYTHON_BIN:-/home/seqaeon/Downloads/venv/bin/python}"
        PYTHONPATH=. "$PYTHON_BIN" -m scripts.sap_synthetic \
            --modes sir_anchor sir_tree --T 4 --steps 8 \
            --freeze-trunk-after 4 --eval-milestones 4 8 \
            --batch 2 --pool 8 --seq-len 24 --depth 1 --n-embd 64 --vocab 64 \
            --eval-seqs 4 --eval-positions 8 --samples-per-ctx 1 --iwae-samples 2 \
            --sir-posterior-mix 0 --sir-anchor-stride 2 --sir-tree-levels 2 \
            --log-every 1 --out out/s02_sap/smoke
        ;;
    --modal)
        modal run modal_sap.py::s02
        ;;
    *)
        echo "usage: $0 [--smoke|--modal]" >&2
        exit 2
        ;;
esac

