#!/bin/bash
#SBATCH --job-name=timeparam-table3-full
#SBATCH -c 8
#SBATCH --mem 32G
#SBATCH --gres=gpu:2
#SBATCH -t 24:00:00
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err

set -euo pipefail

# Resolves to this script's own directory, regardless of where the repo is
# cloned or what directory `sbatch` was invoked from without hardcoding.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate your own Python environment BEFORE submitting this job (e.g.
# `conda activate <env>` or `source /path/to/venv/bin/activate`), the environment
# location is inherently machine-specific.
# See the top-level README for the exact dependency list.

# Not every environment this runs in is SLURM (reviewers reproducing results
# locally won't have SLURM_JOB_ID set at all), and not every machine has a
# NVIDIA GPU. Both are made optional below.

SLURM_JOB_ID="${SLURM_JOB_ID:-local}"

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "GPUs visible: ${CUDA_VISIBLE_DEVICES:-none}"

if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv
    NUM_GPUS=$(nvidia-smi -L | wc -l)
else
    echo "nvidia-smi not found -- assuming no GPU (CPU-only run)."
    NUM_GPUS=0
fi
echo "Detected ${NUM_GPUS} GPU(s) -- $([ "${NUM_GPUS}" -ge 2 ] && echo 'running monomial/legendre in parallel' || echo 'running serially')."

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

OVERALL_STATUS=0

echo ""
echo "===== Static/identity baseline: launching 5 seeds (basis=none, GPU 0) ====="
for SEED in 0 1 2 3 4; do
    LOG="results/slurm-${SLURM_JOB_ID}-none-seed${SEED}.log"
    python3 -u ode_demo_NEW.py \
        --test_freq ${TEST_FREQ} \
        --basis none \
        --data_size ${DATA_SIZE} \
        --seed ${SEED} \
        --grad_clip ${GRAD_CLIP} \
        --lr_decay_every ${LR_DECAY_EVERY} \
        --lr_decay_gamma ${LR_DECAY_GAMMA} \
        --early_stop_patience ${EARLY_STOP_PATIENCE} \
        --train_end_frac ${TRAIN_END_FRAC} \
        --val_end_frac ${VAL_END_FRAC} \
        --batch_time ${BATCH_TIME} \
        --results_json "results/results_table3_none_d0_seed${SEED}.json" \
        --gpu 0 \
        &> "${LOG}" || { echo "static baseline seed ${SEED} run FAILED"; OVERALL_STATUS=1; }
    echo "Completed static baseline seed ${SEED}"
done
echo "Static/identity baseline complete (5/5 seeds)."

if [ "${NUM_GPUS}" -ge 2 ]; then
    GPU_LEG=1
else
    GPU_LEG=0
fi

for DEGREE in 3 4 5; do
    for SEED in 0 1 2 3 4; do
        echo ""
        echo "===== Degree ${DEGREE}, Seed ${SEED}: launching monomial (GPU 0) + legendre (GPU ${GPU_LEG}) ====="

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
            --results_json "results/results_table3_${BASIS}_d${DEGREE}_seed${SEED}.json" \
            --gpu 0 \
            &> "${LOG}" &
        PID_MONO=$!

        # If fewer than 2 GPUs, wait for monomial to finish before starting
        # legendre -- they'd otherwise contend for the same device.
        if [ "${NUM_GPUS}" -lt 2 ]; then
            wait ${PID_MONO} || { echo "degree ${DEGREE} seed ${SEED} monomial run FAILED"; OVERALL_STATUS=1; }
        fi

        # --- legendre run: GPU ${GPU_LEG} ---
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
            --results_json "results/results_table3_${BASIS}_d${DEGREE}_seed${SEED}.json" \
            --gpu ${GPU_LEG} \
            &> "${LOG}" &
        PID_LEG=$!

        echo "Launched legendre run (PID ${PID_LEG}) on GPU ${GPU_LEG}, degree ${DEGREE}, seed ${SEED}"

        if [ "${NUM_GPUS}" -ge 2 ]; then
            wait ${PID_MONO} || { echo "degree ${DEGREE} seed ${SEED} monomial run FAILED"; OVERALL_STATUS=1; }
        fi
        wait ${PID_LEG}  || { echo "degree ${DEGREE} seed ${SEED} legendre run FAILED"; OVERALL_STATUS=1; }
        echo "===== Degree ${DEGREE}, Seed ${SEED} complete ====="
    done
done

echo ""
echo "All 35 runs complete (5 static/identity baseline + 3 degrees x 2 bases x 5 seeds)."
echo "Run ./process_results.sh separately to aggregate."

exit ${OVERALL_STATUS}
