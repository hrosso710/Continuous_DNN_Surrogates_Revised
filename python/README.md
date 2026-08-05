# Python: optimize-then-discretize neural ODE experiments

This repo contains per-experiment (matlab and python) subdirectories with `run_test.sh` (quick validation) / `run_experiment.sh` (full paper run) / `process_results.sh` (figure/table generation), plus scripts split by concern (running vs. analyzing). 

## Structure

```
python/
├── experiment_runtime.py   Shared utilities for RUNNING experiments
├── analysis_utils.py       Shared utilities for PROCESSING results
├── process_all_results.py  Master script: runs every experiment's process_results.sh
├── timeparam_NEW.py        TimeParameterizedNet -- shared across both experiments
├── .gitignore
├── stationary_ode/         Table 4, Table 6 (synthetic 2D linear ODE task) COMBINE!!! isn't there a figure too?
│   ├── ode_demo_NEW.py
│   ├── aggregate_table4.py
│   ├── aggregate_table6.py
│   ├── run_test.sh
│   ├── run_experiment.sh       consolidated: degree 3, 4, 5, all in one script
│   ├── process_results.sh
│   ├── results/                 (gitignored)
│   └── outputs/
└── surrogate/               Table 5, Figure 6 (ELM/CDR/DCR surrogate datasets)
    ├── train_surrogate_node.py
    ├── aggregate_table5.py
    ├── data/                 NNERDS.mat, CDR_Data.mat, DCR_Data.mat -- a SEPARATE copy from matlab/data/, not shared/symlinked. 
    ├── run_test.sh
    ├── run_experiment.sh
    ├── process_results.sh      
    ├── results/                 (gitignored)
    └── outputs/
```


## Shared modules

- **`experiment_runtime.py`**: `relative_error` (shared metric convention with the MATLAB side), `makedirs`, `RunningAverageMeter`, a flush-forcing `print` override.
- **`analysis_utils.py`**: `load_all_results` / `summarize_by_config` for loading `results_*.json` summaries and aggregating mean±std across seeds per config.
- **`timeparam_NEW.py`**: `TimeParameterizedNet`, used by both `stationary_ode/ode_demo_NEW.py` and `surrogate/train_surrogate_node.py`.

## Environment setup (do this before running anything)

```bash
# Create and activate your own environment however you prefer, e.g.:
python3 -m venv venv && source venv/bin/activate
# or: conda create -n <env> python=3.x && conda activate <env>

pip install torch torchdiffeq numpy scipy tqdm matplotlib pandas
```

The `run_experiment.sh`/`run_test.sh` scripts do **not** activate an environment for you. This is a separate prerequisite step, not hardcoded into each script, since environment location is inherently specific to your machine and can't be baked into a script meant to run on anyone's setup. Activate your environment, *then* run/submit the scripts below.

All scripts resolve their own directory dynamically (`cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"`) rather than hardcoding a path — so they work regardless of where you clone the repo or what directory you run `sbatch`/`./script.sh` from.

## Dependencies

```
torch
torchdiffeq
numpy
scipy
tqdm
matplotlib
pandas
```

*(No `requirements.txt` pinned yet.)*

## Train / validation / test split

Per referee comment 2.6.5, contiguous non-overlapping regions -- see each experiment's docstring for specifics (`stationary_ode` splits by time fraction of `[0, T_MAX]`; `surrogate` uses ELM's baked-in `.mat` split for ELM, and an independent 400/200/rest NumPy split for CDR/DCR -- **not** bit-identical to the MATLAB-side split for CDR/DCR, since MATLAB's `randperm` state isn't reproducible from Python; see `train_surrogate_node.py`'s module docstring). The reported result is always the best-on-validation checkpoint.

## Quick validation

```bash
cd stationary_ode && ./run_test.sh
cd ../surrogate && ./run_test.sh
```

## Full reproduction

```bash
cd stationary_ode
./run_experiment.sh    # 30 runs: 3 degrees x 2 bases x 5 seeds
./process_results.sh   # -> Table 4/6 numbers

cd ../surrogate
./run_experiment.sh    # 45 runs: 3 datasets x 3 bases x 5 seeds
./process_results.sh  
```
