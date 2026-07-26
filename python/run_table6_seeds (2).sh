#!/bin/bash
#SBATCH --job-name=timeparam-table6-seeds
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

DEGREE=3
DATA_SIZE=500       # dt = 25/499 ~= 0.05; resolves the ~3.14-time-unit oscillation
                     # comfortably without the extra cost of the old default (1000).
TEST_FREQ=100        # eval calls now only integrate the 4-unit VAL region (t in
                     # [16,20]), not the full 25-unit domain, so this should be
                     # substantially cheaper per check than earlier runs.

# --- Stability fixes, added after observing late-training divergence in the
# unseeded diagnostic run. ---
GRAD_CLIP=1.0
LR_DECAY_EVERY=500
LR_DECAY_GAMMA=0.5

# Early stopping on VALIDATION relative error only (t in [16,20]) -- the
# checkpoint it selects is what gets used for the final train/val/test
# reporting, so this IS the final-vs-best decision, made once and disclosed
# rather than picked post-hoc. TEST (t in [20,25]) is never touched until
# after training stops.
EARLY_STOP_PATIENCE=5

# Train/val/test split (Referee 2, comment 2.6.5; "Option A" -- contiguous,
# non-overlapping time regions). Defaults in ode_demo_NEW.py already match
# these; set explicitly here for visibility/documentation.
TRAIN_END_FRAC=0.64   # t in [0, 16]
VAL_END_FRAC=0.80     # t in [16, 20]; test is t in [20, 25]

# batch_time is FIXED at 10 throughout -- no curriculum growth. A batch-time
# curriculum (grow 10 -> 100 over training) was tried to address
# compounding/exposure-bias error, but was abandoned: see
# run_table4_degrees.sh's comment (made results worse; also never got past
# "verify via smoke_test_curriculum.sh before trusting this matrix", so it
# was never confirmed to actually help). The degree-3 numbers already
# locked into the manuscript (Table 4's baseline) used this same fixed
# batch_time=10 protocol -- matching it here is required so Table 6's
# static/monomial/legendre numbers are comparable to Table 4's, not
# apples-to-oranges.
BATCH_TIME=10

mkdir -p results

# 3 basis conditions x 5 seeds = 15 runs. Only 2 GPUs requested, so per seed:
# monomial (GPU 0) + legendre (GPU 1) run in parallel first, then the static
# baseline ("none") runs alone on GPU 0 -- it has no time-parameterization
# overhead and should be the fastest of the three.
#
# NOTE: background launches are written INLINE below, not wrapped in a
# function called via command substitution ($(...)) -- that pattern was
# tried and reverted after testing showed `wait` cannot track a PID that was
# backgrounded inside a subshell (the process gets reparented once the
# subshell exits, so `wait $PID` fails with "not a child of this shell" and
# does not actually block). Keeping every `&> ... &` launch directly in the
# main script body is what makes `wait $!` reliable.
OVERALL_STATUS=0

for SEED in 0 1 2 3 4; do
    echo ""
    echo "===== Seed ${SEED}: launching monomial (GPU 0) + legendre (GPU 1) ====="

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
        --results_json "results/results_table6_${BASIS}_seed${SEED}.json" \
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
        --results_json "results/results_table6_${BASIS}_seed${SEED}.json" \
        --gpu 1 \
        &> "${LOG}" &
    PID_LEG=$!

    echo "Launched monomial run (PID ${PID_MONO}) on GPU 0, seed ${SEED}"
    echo "Launched legendre run (PID ${PID_LEG}) on GPU 1, seed ${SEED}"

    wait ${PID_MONO} || { echo "seed ${SEED} monomial run FAILED"; OVERALL_STATUS=1; }
    wait ${PID_LEG}  || { echo "seed ${SEED} legendre run FAILED"; OVERALL_STATUS=1; }
    echo "===== Seed ${SEED}: monomial + legendre complete ====="

    # --- static (identity) baseline run: GPU 0, alone ---
    echo "===== Seed ${SEED}: launching static baseline (GPU 0) ====="
    BASIS="none"
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
        --results_json "results/results_table6_${BASIS}_seed${SEED}.json" \
        --gpu 0 \
        &> "${LOG}" &
    PID_NONE=$!
    echo "Launched static baseline run (PID ${PID_NONE}) on GPU 0, seed ${SEED}"

    wait ${PID_NONE} || { echo "seed ${SEED} static baseline run FAILED"; OVERALL_STATUS=1; }
    echo "===== Seed ${SEED} fully complete (3/3 conditions) ====="
done

echo ""
echo "All 5 seeds (15 runs total) complete. Aggregating results..."
python3 aggregate_table6.py --results_dir results

exit ${OVERALL_STATUS}