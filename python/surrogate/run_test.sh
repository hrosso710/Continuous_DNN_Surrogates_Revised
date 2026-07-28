#!/bin/bash
# run_test.sh -- quick validation run for the surrogate-dataset (ELM/CDR/DCR)
# neural ODE experiment.
#
# Runs just CDR (smallest of the three datasets: 800 total samples, vs.
# ELM's ~2486 and DCR's 10000) at reduced --niters, one seed, all three
# basis conditions, so the full pipeline can be verified end-to-end quickly
# before committing to the full run_experiment.sh (45 runs across all three
# datasets). Writes to results_test/ so output never collides with real
# results.

set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p results_test

echo "Running quick validation on CDR only: static, Legendre d=3, monomial d=3"
echo "(reduced niters=100 -- NOT paper-quality results)"

for BASIS in none legendre monomial; do
    echo ""
    echo "===== CDR, basis=${BASIS} ====="
    python3 train_surrogate_node.py \
        --dataset CDR \
        --basis "${BASIS}" \
        --degree 3 \
        --seed 0 \
        --niters 100 \
        --test_freq 20 \
        --grad_clip 1.0 \
        --early_stop_patience 5 \
        --results_json "results_test/test_CDR_${BASIS}.json" \
        --gpu 0
done

echo ""
echo "Quick validation complete. Check results_test/*.json for sane (non-NaN) relative errors."
