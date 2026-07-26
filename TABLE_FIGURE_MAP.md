# Table/Figure Reproduction — Status

Compact cross-reference. For exact commands, see [`matlab/TABLE_FIGURE_MAP.md`](matlab/TABLE_FIGURE_MAP.md) and [`python/README.md`](python/README.md) (Python side not yet populated).

| # | Description | Section | Metric | Status |
|---|---|---|---|---|
| Table 2 | Relative error by architecture/depth/optimizer | matlab | test error | ✅ confirmed, `run_all_table2.m` |
| Table 3 | DCR Hamiltonian, ΔError vs. Legendre degree (3–6) | matlab | train error | ✅ confirmed, `collectTable3.m` (ΔError = absolute difference) |
| Figs 2–3 | Training/validation error curves (ResNet, Hamiltonian; ADAM; monomial vs. Legendre) | matlab | train + validation | ✅ confirmed, `collectResults.m` tikz export |
| Figs 4–5 | Validation error convergence across Legendre degree 3–6 (all 3 datasets, both architectures) | matlab | validation error | ✅ confirmed, same `collectResults.m` tikz export, filtered by degree instead of basis |
| Fig 6 | Neural ODE loss vs. function evaluations | python | — | ⏳ pending (Python pipeline not yet in repo) |
| Table 4 | Neural ODE parameter count and test error by degree | python | test error | ⏳ pending — `run_table4_degrees.sh` present but depends on an uncommitted aggregation script and pre-existing degree-3 results |
| Tables 5–6 | Neural ODE train/val/test error, stationary ODE task | python | train/val/test | ⏳ pending |
