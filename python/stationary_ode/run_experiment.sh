#!/bin/bash
#SBATCH --job-name=timeparam-table4-full
#SBATCH -c 8
#SBATCH --mem 32G
#SBATCH --gres=gpu:2
#SBATCH -t 24:00:00
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err

set -euo pipefail

# Resolves to this script's own directory, regardless of where the repo is
# cloned or what directory `sbatch` was invoked from -- no hardcoded path.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate your own Python environment BEFORE submitting this job (e.g.
# `conda activate <env>` or `source /path/to/venv/bin/activate`), the same
# way the advisor's rampde repo documents `conda activate torch28` as a
# prerequisite rather than hardcoding it into each script -- environment
# location is inherently machine-specific and shouldn't be baked in here.
# See the top-level README for the exact dependency list.

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "GPUs visible: ${CUDA_VISIBLE_DEVICES:-none}"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv

# CONSOLIDATION NOTE: this replaces the old run_table4_degrees.sh (degrees
# 4-5) + run_degree3_table4.sh (degree 3, rerun under matching flags) as two
# separate scripts. Same protocol throughout, all three degrees now run in
# one place so there's no risk of degree 3 silently drifting out of sync
# with 4/5 again.
DATA_SIZE=500
TEST_FREQ=100
GRAD_CLIP=1.0
LR_DECAY_EVERY=500
LR_DECAY_GAMMA=0.5
EARLY_STOP_PATIENCE=5
TRAIN_END_FRAC=0.64   # t in [0, 16]
VAL_END_FRAC=0.80     # t in [16, 20]; test is t in [20, 25]
BATCH_TIME=10          # fixed throughout -- no curriculum growth

mkdir -p results

# ⚠️ GAP, not yet resolved: this script (like its two predecessors) does NOT
# regenerate the static/identity baseline -- aggregate_table4.py's
# --static_baseline_relerr defaults to a HARDCODED 8.2132, sourced from an
# earlier run whose protocol hasn't been confirmed to match --grad_clip/
# --lr_decay_every/--early_stop_patience used here. If that baseline predates
# those flags the same way the old degree-3 numbers did, it has the same
# apples-to-oranges problem this script just fixed for degree 3 -- just not
# yet caught. Confirm this before treating Table 4/6 as final, and if it
# needs rerunning, add that run to this script (basis=none, 5 seeds) rather
# than leaving it as a silent hardcoded fallback.

# 3 degrees x 2 bases (monomial, legendre) x 5 seeds = 30 runs. Only 2 GPUs
# requested, so per (degree, seed) pair: monomial (GPU 0) + legendre (GPU 1)
# run in parallel, then the loop waits for both before moving on.
#
# Background launches are inline (not wrapped in a function/subshell) --
# `wait` cannot reliably track a PID backgrounded inside a subshell (verified
# failure mode, per the original scripts' comments).
OVERALL_STATUS=0

for DEGREE in 3 4 5; do
    for SEED in 0 1 2 3 4; do
        echo ""
        echo "===== Degree ${DEGREE}, Seed ${SEED}: launching monomial (GPU 0) + legendre (GPU 1) ====="

        # --- monomial run: GPU 0 ---
        BASIS="monomial"
        LOG="results/slurm-${SLURM_JOB_ID}-${BASIS}-degree-${DEGREE}-seed${SEED}.log"
        python3 -u ode_demo_NEW.py \
            --test_freq ${TEST_FREQ} \
            --basis ${BASIS} \
            --degree ${DEGREE} \
            --data_size ${DATA_SIZE} \
            --seed ${SEED} \
            --grad_clip ${GRAD_CLIP} \
            --lr_decay_every ${LR_DECAY_EVERY} \
            --lr_decay_gamma ${LR_DECAY_GAMMA} \
            --early_stop_patience ${EARLY_STOP_PATIENCE} \
            --train_end_frac ${TRAIN_END_FRAC} \
            --val_end_frac ${VAL_END_FRAC} \
            --batch_time ${BATCH_TIME} \
            --results_json "results/results_table4_${BASIS}_d${DEGREE}_seed${SEED}.json" \
            --gpu 0 \
            &> "${LOG}" &
        PID_MONO=$!

        # --- legendre run: GPU 1 ---
        BASIS="legendre"
        LOG="results/slurm-${SLURM_JOB_ID}-${BASIS}-degree-${DEGREE}-seed${SEED}.log"
        python3 -u ode_demo_NEW.py \
            --test_freq ${TEST_FREQ} \
            --basis ${BASIS} \
            --degree ${DEGREE} \
            --data_size ${DATA_SIZE} \
            --seed ${SEED} \
            --grad_clip ${GRAD_CLIP} \
            --lr_decay_every ${LR_DECAY_EVERY} \
            --lr_decay_gamma ${LR_DECAY_GAMMA} \
            --early_stop_patience ${EARLY_STOP_PATIENCE} \
            --train_end_frac ${TRAIN_END_FRAC} \
            --val_end_frac ${VAL_END_FRAC} \
            --batch_time ${BATCH_TIME} \
            --results_json "results/results_table4_${BASIS}_d${DEGREE}_seed${SEED}.json" \
            --gpu 1 \
            &> "${LOG}" &
        PID_LEG=$!

        echo "Launched monomial run (PID ${PID_MONO}) on GPU 0, degree ${DEGREE}, seed ${SEED}"
        echo "Launched legendre run (PID ${PID_LEG}) on GPU 1, degree ${DEGREE}, seed ${SEED}"

        wait ${PID_MONO} || { echo "degree ${DEGREE} seed ${SEED} monomial run FAILED"; OVERALL_STATUS=1; }
        wait ${PID_LEG}  || { echo "degree ${DEGREE} seed ${SEED} legendre run FAILED"; OVERALL_STATUS=1; }
        echo "===== Degree ${DEGREE}, Seed ${SEED} complete ====="
    done
done

echo ""
echo "All 30 runs (3 degrees x 2 bases x 5 seeds) complete."
echo "Run ./process_results.sh separately to aggregate."

exit ${OVERALL_STATUS}
