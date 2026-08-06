# MATLAB: discretize-then-optimize ResNet/Hamiltonian experiments

Reproduces Table 2 and Figures 2-3.

## Directory layout

```
matlab/
├── meganet/       Core network layers and optimizers (forked from Meganet)
├── data/          Dataset setup functions + ELM data (NNERDS.mat)
├── experiments/   Experiment driver, batch launchers, results aggregation
└── README.md
```

## Dependencies

- MATLAB R2024b and newer
- Symbolic Math Toolbox — `meganet/getPolynomialBasis.m`'s `'Legendre'` basis calls `legendreP`, which lives in this toolbox rather than core MATLAB 

## Data

- **ELM**: `data/NNERDS.mat` is included directly (420 KB).
- **CDR**: `data/CDR_Data.mat` is included directly (620 KB).
- **DCR**: `data/DCR_Data.mat` (65 MB) is hosted on Zenodo rather than committed directly: [10.5281/zenodo.21537966](https://doi.org/10.5281/zenodo.21537966). `setupDCR.m` will offer to download it automatically (via the Zenodo API) the first time it's needed; alternatively, download it yourself from that DOI and place it in `matlab/data/` before running any DCR experiment.

## Reproducing results

All experiments write to a `results/` directory (created automatically) as one `.mat` file per `(dataset, architecture, basis, degree, T, optimizer)` configuration, plus a raw per-iteration history `.txt` file.

### Running a single configuration

```matlab
addpath(genpath('meganet'), genpath('data'), genpath('experiments'))
runExperiment_v2('hamiltonian', 1, 3, 'sgd', 'Legendre', 'DCR')
```

Arguments: `dynamic` (`'ResNN'`, `'hamiltonian'`, `'leapfrog'`, `'antiSym-ResNN'`), `T`, `d` (basis degree), `opti` (`'sgd'` for ADAM, `'GNvpro'`), `basis` (`'monomial'`, `'Legendre'`, `'none'`), `dataset` (`'ELM'`, `'CDR'`, `'DCR'`).

### Running the full Table 2 sweep

UPDATE

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

