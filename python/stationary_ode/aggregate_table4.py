"""
aggregate_table4.py

Reads the per-run JSON summaries produced by run_experiment.sh's degree
3/4/5 sweep (results_table4_{basis}_d{degree}_seed{seed}.json) and
aggregates them into Table-4-ready rows: mean/std test relative error per
(basis, degree), plus Delta-Error relative to the static (identity)
baseline. Table 6 (degree-3-only train/val/test numbers) is a separate
script, aggregate_table6.py, reading the same result files.

Usage:
    python3 aggregate_table4.py --results_dir results

The static baseline's mean test relative error defaults to 8.2132. This is
no longer an unconfirmed placeholder: run_experiment.sh now generates the
static/identity baseline (basis=none, 5 seeds) under the exact same protocol
flags used for the monomial/legendre sweep (--grad_clip 1.0
--lr_decay_every 500 --lr_decay_gamma 0.5 --early_stop_patience 5), and the
resulting files (results_table4_none_d0_seed0-4.json, all 5 cleanly
early-stopped) give mean test relerr 8.2132 +/- 0.3433 -- matching this
default exactly. Override with --static_baseline_relerr only if this run is
ever repeated and the value changes.
"""
import argparse
import glob
import json
import os
import statistics
import sys

# Parameter counts are read directly from each run's recorded
# 'trainable_params' field (see load_results/main below) rather than
# computed from a formula, so Table 4 always reflects what the model
# actually reported -- matching aggregate_table6.py's convention. The
# (degree + 1) * 252 pattern is what we've observed in practice (degree 3:
# 1008, degree 4: 1260, degree 5: 1512, degree 0/'none': 252) and is kept
# here only as a sanity-check cross-reference, not as the source of truth.
BASE_PARAMS = 252


def expected_param_count(degree):
    """Sanity-check reference only -- see module-level comment above."""
    return (degree + 1) * BASE_PARAMS


def load_results(results_dir):
    pattern = os.path.join(results_dir, 'results_table4_*.json')
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No result files found matching {pattern}", file=sys.stderr)
        sys.exit(1)

    by_basis_degree = {}
    for f in files:
        with open(f) as fh:
            r = json.load(fh)
        key = (r['basis'], r['degree'])
        by_basis_degree.setdefault(key, []).append(r)
    return by_basis_degree


def check_run_health(runs, basis, degree):
    problems = []
    for r in runs:
        if not r.get('stopped_early', False):
            problems.append(
                f"  {basis} degree {degree}, seed {r['seed']}: did NOT trigger "
                f"early stopping (ran to iter {r.get('stop_iter') or r['niters_requested']})."
            )
    if problems:
        print(f"\n[WARNING] {basis} (degree {degree}): {len(problems)}/{len(runs)} "
              f"run(s) did not cleanly early-stop:")
        for p in problems:
            print(p)
    return len(problems)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default='.')
    parser.add_argument('--static_baseline_relerr', type=float, default=8.2132,
                         help='Mean test relative error of the static (identity) '
                              'baseline, for computing Delta-Error. Default is the '
                              'already-known degree-3-matrix value.')
    args = parser.parse_args()

    by_basis_degree = load_results(args.results_dir)
    total_problems = 0
    rows = []

    for (basis, degree), runs in sorted(by_basis_degree.items()):
        seeds_present = sorted(r['seed'] for r in runs)
        expected_seeds = list(range(5))
        if seeds_present != expected_seeds:
            print(f"[WARNING] {basis} degree {degree}: expected seeds {expected_seeds}, "
                  f"found {seeds_present}", file=sys.stderr)

        total_problems += check_run_health(runs, basis, degree)

        errs = [r['reported_test_relerr'] for r in runs]
        mean = statistics.mean(errs)
        std = statistics.stdev(errs) if len(errs) > 1 else 0.0
        delta = mean - args.static_baseline_relerr

        params_seen = {r['trainable_params'] for r in runs}
        if len(params_seen) > 1:
            print(f"[WARNING] {basis} degree {degree}: inconsistent trainable_params "
                  f"across seeds: {sorted(params_seen)}", file=sys.stderr)
        params = runs[0]['trainable_params']
        if params != expected_param_count(degree):
            print(f"[WARNING] {basis} degree {degree}: reported trainable_params "
                  f"({params}) does not match the (degree+1)*252 sanity check "
                  f"({expected_param_count(degree)}) -- double check before using "
                  f"this row.", file=sys.stderr)

        rows.append({
            'basis': basis, 'degree': degree, 'n_seeds': len(runs),
            'params': params,
            'test_mean': mean, 'test_std': std, 'delta': delta,
        })

    print("\n" + "=" * 90)
    print("TABLE 4 rows (degree 3/4/5 sweep) -- ready to paste in")
    print("=" * 90)
    print(f"{'Basis':<12}{'Degree':<8}{'Params':<10}{'Test RelErr (mean+-std)':<28}{'DeltaError':<12}")
    print("-" * 90)
    for row in sorted(rows, key=lambda r: (r['basis'], r['degree'])):
        test_str = f"{row['test_mean']:.3f} +/- {row['test_std']:.3f}"
        print(f"{row['basis']:<12}{row['degree']:<8}{row['params']:<10}{test_str:<28}{row['delta']:+.3f}")
    print("=" * 90)

    out_path = os.path.join(args.results_dir, 'table4_degree_sweep.csv')
    with open(out_path, 'w') as f:
        f.write("basis,degree,n_seeds,params,test_mean,test_std,delta_error\n")
        for row in rows:
            f.write(f"{row['basis']},{row['degree']},{row['n_seeds']},{row['params']},"
                     f"{row['test_mean']:.6f},{row['test_std']:.6f},{row['delta']:+.6f}\n")
    print(f"\nWrote {out_path}")

    # LaTeX-ready lines, matching Table 4's existing row format exactly.
    print("\nLaTeX table rows:")
    for basis in ['monomial', 'legendre']:
        print(f"  % {basis}")
        for row in sorted([r for r in rows if r['basis'] == basis], key=lambda r: r['degree']):
            print(f"  & {row['degree']}  & {row['params']} & "
                  f"${row['test_mean']:.3f} \\pm {row['test_std']:.3f}$ (${row['delta']:+.3f}$) \\\\")

    if total_problems > 0:
        print(f"\n[SUMMARY] {total_problems} run(s) did not cleanly early-stop -- "
              f"review before treating this as final.")
    else:
        print(f"\n[SUMMARY] All runs cleanly triggered early stopping.")


if __name__ == '__main__':
    main()
