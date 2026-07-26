# Table/Figure Reproduction — Status

Compact cross-reference. For exact commands, see [`matlab/TABLE_FIGURE_MAP.md`](matlab/TABLE_FIGURE_MAP.md) and [`python/README.md`](python/README.md).

| # | Description | Section | Metric | File Name
|---|---|---|---|---|
| Table 2 | Relative error by architecture/depth/optimizer | matlab | test error | `run_all_table2.m` |
| Table 3 | DCR Hamiltonian, ΔError baseline vs. Legendre degree (3–6) | matlab | train error | `collectTable3.m` (ΔError = absolute difference) |
| Figs 2–3 | Training/validation error curves (ResNet & Hamiltonian, ADAM, monomial vs. Legendre) | matlab | train + validation error| `collectResults.m` tikz export |
| Figs 4–5 | Validation error convergence across Legendre degree 3–6 (all 3 datasets, both architectures) | matlab | validation error | `collectResults.m` tikz export, filtered by degree instead of basis |
| Fig 6 | Neural ODE loss vs. function evaluations | python | train + validation error | - |
| Table 4 | Neural ODE parameter count and test error by degree | python | test error | `run_table4_degrees.sh` but pending...|
| Tables 5–6 | Neural ODE train/val/test error, stationary ODE task | python | train/val/test error | pending |
