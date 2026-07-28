#!/bin/bash
#SBATCH --job-name=timeparam-table5-surrogate
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

# Activate your own Python environment BEFORE submitting this job -- see the
# note in stationary_ode/run_experiment.sh for why this isn't hardcoded here.

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "GPUs visible: ${CUDA_VISIBLE_DEVICES:-none}"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv

# Table 5 (ELM/CDR/DCR neural ODE surrogate results), via train_surrogate_node.py.
# Mirrors run_table6_seeds.sh's structure and protocol (degree=3, 5 seeds,
# static/monomial/legendre) -- ASSUMPTION, not yet confirmed against a
# manuscript draft of Table 5. Change DEGREE/SEEDS below if that's wrong.
#
# Path to the directory containing NNERDS.mat, CDR_Data.mat, DCR_Data.mat.
# Sits alongside train_surrogate_node.py (matches its own --data_dir
# default) -- works for anyone who clones the repo and cd's into
# python/surrogate/ before running, regardless of where the repo itself
# lives on disk.
DATA_DIR="data"

DEGREE=3
DATASETS="ELM CDR DCR"

# batch_size/niters/test_freq are train_surrogate_node.py's defaults
# (20/2000/20) -- these were only smoke-tested on synthetic data, not real
# ELM/CDR/DCR signal. Worth a single real-data smoke run before trusting
# this full sweep (see conversation notes).
GRAD_CLIP=1.0
LR_DECAY_EVERY=500
LR_DECAY_GAMMA=0.5
EARLY_STOP_PATIENCE=5

mkdir -p results

# For each dataset: 3 basis conditions x 5 seeds = 15 runs. Same pattern as
# run_table6_seeds.sh -- monomial (GPU 0) + legendre (GPU 1) in parallel,
# then static baseline alone on GPU 0. Background launches stay INLINE (not
# wrapped in a function/subshell) -- `wait $PID` cannot reliably track a PID
# backgrounded inside a subshell (verified failure mode, see
# run_table4_degrees.sh / run_table6_seeds.sh comments).
OVERALL_STATUS=0

for DATASET in ${DATASETS}; do
    echo ""
    echo "########## Dataset: ${DATASET} ##########"

    for SEED in 0 1 2 3 4; do
        echo ""
        echo "===== ${DATASET}, Seed ${SEED}: launching monomial (GPU 0) + legendre (GPU 1) ====="

        # --- monomial run: GPU 0 ---
        BASIS="monomial"
        LOG="results/slurm-${SLURM_JOB_ID}-${DATASET}-${BASIS}-degree-${DEGREE}-seed${SEED}.log"
        python3 -u train_surrogate_node.py \
            --dataset ${DATASET} \
            --data_dir ${DATA_DIR} \
            --basis ${BASIS} \
            --degree ${DEGREE} \
            --seed ${SEED} \
            --grad_clip ${GRAD_CLIP} \
            --lr_decay_every ${LR_DECAY_EVERY} \
            --lr_decay_gamma ${LR_DECAY_GAMMA} \
            --early_stop_patience ${EARLY_STOP_PATIENCE} \
            --results_json "results/results_table5_${DATASET}_${BASIS}_d${DEGREE}_seed${SEED}.json" \
            --gpu 0 \
            &> "${LOG}" &
        PID_MONO=$!

        # --- legendre run: GPU 1 ---
        BASIS="legendre"
        LOG="results/slurm-${SLURM_JOB_ID}-${DATASET}-${BASIS}-degree-${DEGREE}-seed${SEED}.log"
        python3 -u train_surrogate_node.py \
            --dataset ${DATASET} \
            --data_dir ${DATA_DIR} \
            --basis ${BASIS} \
            --degree ${DEGREE} \
            --seed ${SEED} \
            --grad_clip ${GRAD_CLIP} \
            --lr_decay_every ${LR_DECAY_EVERY} \
            --lr_decay_gamma ${LR_DECAY_GAMMA} \
            --early_stop_patience ${EARLY_STOP_PATIENCE} \
            --results_json "results/results_table5_${DATASET}_${BASIS}_d${DEGREE}_seed${SEED}.json" \
            --gpu 1 \
            &> "${LOG}" &
        PID_LEG=$!

        echo "Launched monomial run (PID ${PID_MONO}) on GPU 0, ${DATASET}, seed ${SEED}"
        echo "Launched legendre run (PID ${PID_LEG}) on GPU 1, ${DATASET}, seed ${SEED}"

        wait ${PID_MONO} || { echo "${DATASET} seed ${SEED} monomial run FAILED"; OVERALL_STATUS=1; }
        wait ${PID_LEG}  || { echo "${DATASET} seed ${SEED} legendre run FAILED"; OVERALL_STATUS=1; }
        echo "===== ${DATASET}, Seed ${SEED}: monomial + legendre complete ====="

        # --- static (identity) baseline run: GPU 0, alone ---
        echo "===== ${DATASET}, Seed ${SEED}: launching static baseline (GPU 0) ====="
        BASIS="none"
        LOG="results/slurm-${SLURM_JOB_ID}-${DATASET}-${BASIS}-degree-${DEGREE}-seed${SEED}.log"
        python3 -u train_surrogate_node.py \
            --dataset ${DATASET} \
            --data_dir ${DATA_DIR} \
            --basis ${BASIS} \
            --degree ${DEGREE} \
            --seed ${SEED} \
            --grad_clip ${GRAD_CLIP} \
            --lr_decay_every ${LR_DECAY_EVERY} \
            --lr_decay_gamma ${LR_DECAY_GAMMA} \
            --early_stop_patience ${EARLY_STOP_PATIENCE} \
            --results_json "results/results_table5_${DATASET}_${BASIS}_d${DEGREE}_seed${SEED}.json" \
            --gpu 0 \
            &> "${LOG}" &
        PID_NONE=$!
        echo "Launched static baseline run (PID ${PID_NONE}) on GPU 0, ${DATASET}, seed ${SEED}"

        wait ${PID_NONE} || { echo "${DATASET} seed ${SEED} static baseline run FAILED"; OVERALL_STATUS=1; }
        echo "===== ${DATASET}, Seed ${SEED} fully complete (3/3 conditions) ====="
    done
done

echo ""
echo "All datasets x seeds x bases complete (45 runs total). Aggregation script"
echo "(aggregate_table5.py) not yet written -- results are in results/results_table5_*.json."

exit ${OVERALL_STATUS}