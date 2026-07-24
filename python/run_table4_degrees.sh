#!/bin/bash
#SBATCH --job-name=timeparam-table4-degree-sweep
#SBATCH -c 8
#SBATCH --mem 32G
#SBATCH --gres=gpu:2
#SBATCH -t 24:00:00
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err

set -euo pipefail

cd /local/scratch/hrosso
source venv/bin/activate
cd TimeParamNODE

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "GPUs visible: ${CUDA_VISIBLE_DEVICES:-none}"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv

# IMPORTANT: these settings intentionally match the EXACT protocol already
# used to generate the degree-3 numbers in Table 4/6 -- data_size, test_freq,
# stability flags, and the train/val/test split are identical. There is
# deliberately NO --batch_time_max / curriculum here: that fix was explored
# and abandoned (made results worse, see conversation notes), and the
# degree-3 numbers already locked in for the table predate it. Running
# degree 4/5 under a different training protocol than degree 3 would make
# the comparison across degrees in Table 4 apples-to-oranges.
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

# 2 degrees x 2 bases (monomial, Legendre only -- static/identity has no
# "degree" and its data already exists) x 5 seeds = 20 runs. Only 2 GPUs
# requested, so per (degree, seed) pair: monomial (GPU 0) + Legendre (GPU 1)
# run in parallel, then the loop waits for both before moving on.
#
# Background launches are inline (not wrapped in a function/subshell) for
# the same reason as run_table6_seeds.sh -- `wait` cannot reliably track a
# PID backgrounded inside a subshell (verified this failure mode earlier).
OVERALL_STATUS=0

for DEGREE in 4 5; do
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
echo "All 20 runs (2 degrees x 2 bases x 5 seeds) complete. Aggregating results..."
python3 aggregate_table4.py --results_dir results

exit ${OVERALL_STATUS}