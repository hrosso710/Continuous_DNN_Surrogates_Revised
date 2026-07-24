# MATLAB: discretize-then-optimize ResNet/Hamiltonian experiments

Reproduces Table 2, Table 3, and Figures 1–5.

## Status

> **This directory is not yet runnable end-to-end.** `experiments/runExperiment_v2.m` depends on several Meganet library files not yet included here — see [Dependencies](#dependencies) below. Everything else (setup scripts, batch launchers, result aggregation) is in place and ready once those land.

## Directory layout

```
matlab/
├── meganet/       Core network layers and optimizers (forked from Meganet)
├── data/          Dataset setup functions + ELM data (NNERDS.mat)
├── experiments/   Experiment driver, batch launchers, results aggregation
└── README.md
```

## Dependencies

- MATLAB (version TBD — please confirm minimum tested version)
- The following Meganet files, not yet included in this repo:
  `Meganet.m`, `NN.m`, `getPolynomialBasis.m`, `getDenseAntiSym.m`, `doubleSymLayer.m`, `ResNNrk4.m`, `opEye.m`, `tikhonovReg.m`

## Data

- **ELM**: `data/NNERDS.mat` is included directly (420 KB).
- **CDR**: `data/CDR_Data.mat` is included directly (620 KB).
- **DCR**: `data/DCR_Data.mat` (65 MB) is hosted on Zenodo rather than committed directly: [10.5281/zenodo.21537966](https://doi.org/10.5281/zenodo.21537966). Download it and place it in `matlab/data/` before running any DCR experiment.

## Reproducing results

All experiments write to a `results/` directory (created automatically) as one `.mat` file per `(dataset, architecture, basis, degree, T, optimizer)` configuration, plus a raw per-iteration history `.txt` file.

### Running a single configuration

```matlab
addpath(genpath('meganet'), genpath('data'), genpath('experiments'))
runExperiment_v2('hamiltonian', 1, 3, 'sgd', 'Legendre', 'DCR')
```

Arguments: `dynamic` (`'ResNN'`, `'hamiltonian'`, `'leapfrog'`, `'antiSym-ResNN'`), `T`, `d` (basis degree), `opti` (`'sgd'` for ADAM, `'GNvpro'`), `basis` (`'monomial'`, `'Legendre'`, `'none'`), `dataset` (`'ELM'`, `'CDR'`, `'DCR'`).

### Running the full Table 2 sweep

```matlab
run_all_table2   % all 18 (dataset, T, optimizer) groups, 12 configs each; calls collectResults() at the end
```

Or run any single `(dataset, T, optimizer)` batch directly:

```matlab
runConfigBatch('DCR', 5, 'GNvpro')
```

### Aggregating results

```matlab
collectResults()
```

Produces, inside `results/`:
- `summary_all_runs.csv` — one row per completed run, all metrics and provenance
- `table2_pivot.csv` — Table 2-shaped pivot (dataset × T × basis → architecture × optimizer), with blank cells marking runs not yet completed
- `tikzdata/*.dat` — per-iteration train/validation loss traces for Figures 2–5 (ADAM/sgd, T=1 runs only)

## Notes on optimizers

- **ADAM (`'sgd'`)**: trains both the dynamics parameters (`theta`) and the linear readout (`W`) jointly. `W` is warm-started via closed-form ridge regression at the initial `theta` (rather than a raw random draw) and both `theta` and `W` are Tikhonov-regularized with the same `alpha1`/`alpha2` weights GNvpro uses — see the comments in `runExperiment_v2.m`'s `sgd` branch for the full rationale.
- **GNvpro**: a Gauss-Newton variable-projection method that solves for `W` in closed form at each outer iteration.
