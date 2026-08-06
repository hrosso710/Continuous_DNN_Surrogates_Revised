#!/bin/bash
# run_test.sh -- quick validation run for the stationary ODE experiment.
#
# Runs a small handful of configs at drastically reduced --niters/--data_size
# so the full pipeline (training -> early stopping -> results_json write) can
# be verified end-to-end in a couple minutes, before committing to the full
# ~2000-iteration, multi-seed run_experiment.sh. Uses a distinct results
# subdirectory (results_test/) so test output never collides with or gets
# mistaken for real paper results.

set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p results_test

echo "Running quick validation: static baseline, Legendre d=3, monomial d=3"
echo "(reduced niters=100, data_size=100 -- NOT paper-quality results)"

for BASIS in none legendre monomial; do
    echo ""
    echo "===== basis=${BASIS} ====="
    python3 ode_demo_NEW.py \
        --basis "${BASIS}" \
        --degree 3 \
        --seed 0 \
        --niters 100 \
        --data_size 100 \
        --test_freq 20 \
        --grad_clip 1.0 \
        --early_stop_patience 5 \
        --train_end_frac 0.64 \
        --val_end_frac 0.80 \
        --results_json "results_test/test_${BASIS}.json" \
        --gpu 0
done

echo ""
echo "Quick validation complete. Check results_test/*.json for sane (non-NaN) relative errors."
