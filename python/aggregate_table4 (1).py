"""
aggregate_table4.py

Reads the per-run JSON summaries produced by the degree 4/5 sweep
(run_table4_degrees.sh -> results_table4_{basis}_d{degree}_seed{seed}.json)
and aggregates them into Table-4-ready rows: mean/std test relative error
per (basis, degree), plus Delta-Error relative to the static (identity)
baseline.

Usage:
    python3 aggregate_table4.py --results_dir results
    python3 aggregate_table4.py --results_dir results --table6_dir ../table6/results

Parameter counts are read directly from each run's `trainable_params`
field (already saved by ode_demo_NEW.py) rather than recomputed from a
formula -- avoids the count silently drifting from the ground truth if
the architecture ever changes.

The static baseline's mean test relative error is computed live from
results_table6_none_seed*.json (the degree-3 static/identity runs
produced by run_table6_seeds.sh), found in --results_dir by default or
--table6_dir if given separately. Falls back to --static_baseline_relerr
(default 8.2132, the last known value) only if no such files are found --
a warning is printed in that case since it means the number could be stale.
"""

import argparse
import glob
import json
import os
import statistics
import sys


def compute_static_baseline(table6_dir, override):
    """Compute the static/identity baseline's mean test relerr from
    results_table6_none_seed*.json. Falls back to `override` with a
    warning if no such files are found."""
    pattern = os.path.join(table6_dir, 'results_table6_none_seed*.json')
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"[WARNING] No static-baseline files found matching {pattern} -- "
              f"falling back to --static_baseline_relerr={override}, which may be "
              f"stale. Run run_table6_seeds.sh and pass --table6_dir to compute "
              f"this live instead.", file=sys.stderr)
        return override, 0

    errs = [json.load(open(f))['reported_test_relerr'] for f in files]
    mean = statistics.mean(errs)
    print(f"Computed static baseline from {len(files)} file(s) in {table6_dir}: "
          f"mean test relerr = {mean:.4f}")
    return mean, len(files)


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
    parser.add_argument('--table6_dir', type=str, default=None,
                         help='Directory containing results_table6_none_seed*.json '
                              '(the static/identity baseline runs). Defaults to '
                              '--results_dir if not given.')
    parser.add_argument('--static_baseline_relerr', type=float, default=8.2132,
                         help='Fallback mean test relative error of the static '
                              '(identity) baseline, used only if no '
                              'results_table6_none_seed*.json files are found.')
    args = parser.parse_args()

    table6_dir = args.table6_dir or args.results_dir
    static_baseline_relerr, n_baseline_files = compute_static_baseline(
        table6_dir, args.static_baseline_relerr)

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
        delta = mean - static_baseline_relerr

        params_seen = {r['trainable_params'] for r in runs}
        if len(params_seen) > 1:
            print(f"[WARNING] {basis} degree {degree}: inconsistent trainable_params "
                  f"across seeds: {sorted(params_seen)} -- using the first run's value.",
                  file=sys.stderr)
        params = runs[0]['trainable_params']

        rows.append({
            'basis': basis, 'degree': degree, 'n_seeds': len(runs),
            'params': params,
            'test_mean': mean, 'test_std': std, 'delta': delta,
        })

    print("\n" + "=" * 90)
    print("TABLE 4 rows (degree 4/5 sweep) -- ready to paste in")
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