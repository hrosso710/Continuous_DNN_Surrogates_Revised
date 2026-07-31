# Weight-Parameterization in Continuous Time Deep Neural Networks for Surrogate Modeling

Code accompanying the paper *"Weight-Parameterization in Continuous Time Deep Neural Networks for Surrogate Modeling"* by Haley Rosso, Lars Ruthotto, and Khachik Sargsyan.

This repository is organized into two independent sections, corresponding to the two training paradigms studied in the paper:

- [`matlab/`](matlab) — discretize-then-optimize ResNet and Hamiltonian-inspired architectures (Table 2, Table 3, Figures 1–5). Trained with ADAM and GNvpro via a fork of the [Meganet](https://github.com/EmoryMLIP) library.
- `python/` — optimize-then-discretize neural ODE experiments (Table 4, Table 5, Table 6, Figure 6). 

- [`python/`](python) — optimize-then-discretize neural ODE experiments (Table 4, Table 5, Table 6, Figure 6).

Each section has its own README with setup and exact reproduction commands per table/figure. [`TABLE_FIGURE_MAP.md`](TABLE_FIGURE_MAP.md) is a map that directs every table/figure in the paper to the exact script that generates it, spanning both sections.

## Citation

If you use this code, please cite:

```bibtex
@article{RossoRuthottoSargsyan,
      title={Weight-Parameterization in Continuous Time Deep Neural Networks for Surrogate Modeling}, 
      author={Haley Rosso and Lars Ruthotto and Khachik Sargsyan},
      year={2025},
      eprint={2507.22045},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2507.22045}, 
}
```

## License

MIT — see [LICENSE](LICENSE).

## Funding

This work was partially supported by the US National Science Foundation under grant DMS 2038118, and by Sandia National Laboratories' Laboratory Directed Research and Development (LDRD) program.
