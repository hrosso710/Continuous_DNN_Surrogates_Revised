"""
aggregate_table5.py

Reads the per-run JSON summaries produced by train_surrogate_node.py
(one per dataset x basis x seed) and aggregates them into a Table-5-shaped
mean/std report: mean and standard deviation of TRAIN, VALIDATION, and TEST
relative error, plus test-region NFE, per (dataset, basis), over 5 seeds.

Mirrors aggregate_table6.py's structure/health-check pattern exactly, with
a dataset axis added.

Usage:
    python3 aggregate_table5.py --results_dir results
"""

import argparse
import glob
import json
import os
import statistics
import sys

DATASET_ORDER = ['ELM', 'CDR', 'DCR']
BASIS_ORDER = ['none', 'monomial', 'legendre']
BASIS_LABELS = {'none': 'static (identity)', 'monomial': 'monomial', 'legendre': 'legendre'}


def load_results(results_dir):
    pattern = os.path.join(results_dir, 'results_table5_*.json')
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No result files found matching {pattern}", file=sys.stderr)
        sys.exit(1)

    by_dataset_basis = {}
    for f in files:
        with open(f) as fh:
            r = json.load(fh)
        key = (r['dataset'], r['basis'])
        by_dataset_basis.setdefault(key, []).append(r)
    return by_dataset_basis


def check_run_health(runs, dataset, basis):
    problems = []
    for r in runs:
        if not r.get('stopped_early', False):
            problems.append(
                f"  {dataset}/{basis}, seed {r['seed']}: did NOT trigger early stopping "
                f"(ran to iter {r.get('stop_iter') or r['niters_requested']})."
            )
    if problems:
        print(f"\n[WARNING] {dataset}/{basis}: {len(problems)}/{len(runs)} run(s) did not "
              f"cleanly early-stop:")
        for p in problems:
            print(p)
    return len(problems)


def check_seed_variance(runs, dataset, basis, threshold=2.0):
    """Flag (not fail) datasets/bases where seed-to-seed test relerr varies
    by more than `threshold`x (max/min) -- a signal the mean+-std summary
    may be hiding unstable convergence rather than a genuine basis effect."""
    errs = [r['reported_test_relerr'] for r in runs]
    if min(errs) <= 0:
        return
    ratio = max(errs) / min(errs)
    if ratio > threshold:
        print(f"[WARNING] {dataset}/{basis}: test relerr varies {ratio:.1f}x across seeds "
              f"(min={min(errs):.4f}, max={max(errs):.4f}) -- mean/std below may understate "
              f"how unstable this is. Worth checking per-seed convergence before treating "
              f"as final.", file=sys.stderr)


def aggregate(by_dataset_basis):
    rows = []
    total_problems = 0

    for dataset in DATASET_ORDER:
        present_bases = [b for b in BASIS_ORDER if (dataset, b) in by_dataset_basis]
        missing_bases = [b for b in BASIS_ORDER if (dataset, b) not in by_dataset_basis]
        if missing_bases:
            print(f"[WARNING] {dataset}: no results for basis(es) {missing_bases}",
                  file=sys.stderr)

        for basis in present_bases:
            runs = by_dataset_basis[(dataset, basis)]
            seeds_present = sorted(r['seed'] for r in runs)
            expected_seeds = list(range(5))
            if seeds_present != expected_seeds:
                print(f"[WARNING] {dataset}/{basis}: expected seeds {expected_seeds}, "
                      f"found {seeds_present}", file=sys.stderr)

            total_problems += check_run_health(runs, dataset, basis)
            check_seed_variance(runs, dataset, basis)

            def stats(key):
                vals = [r[key] for r in runs]
                mean = statistics.mean(vals)
                std = statistics.stdev(vals) if len(vals) > 1 else 0.0
                return mean, std

            train_mean, train_std = stats('reported_train_relerr')
            val_mean, val_std = stats('reported_val_relerr')
            test_mean, test_std = stats('reported_test_relerr')
            nfe_mean, nfe_std = stats('reported_test_nfe')

            params_seen = {r['trainable_params'] for r in runs}
            if len(params_seen) > 1:
                print(f"[WARNING] {dataset}/{basis}: inconsistent trainable_params across "
                      f"seeds: {sorted(params_seen)}", file=sys.stderr)
            params = runs[0]['trainable_params']

            rows.append({
                'dataset': dataset, 'basis': basis, 'label': BASIS_LABELS[basis],
                'n_seeds': len(runs), 'params': params,
                'train_mean': train_mean, 'train_std': train_std,
                'val_mean': val_mean, 'val_std': val_std,
                'test_mean': test_mean, 'test_std': test_std,
                'nfe_mean': nfe_mean, 'nfe_std': nfe_std,
            })
    return rows, total_problems


def print_table(rows):
    print("\n" + "=" * 115)
    print("TABLE 5 (ELM/CDR/DCR neural ODE surrogate) -- aggregated over seeds, relative error")
    print("=" * 115)
    header = (f"{'Dataset':<8} {'Basis':<18} {'N':<3} {'Params':<8} {'Train (mean±std)':<20} "
              f"{'Val (mean±std)':<20} {'Test (mean±std)':<20}")
    print(header)
    print("-" * 115)
    for row in rows:
        train_str = f"{row['train_mean']:.4f}±{row['train_std']:.4f}"
        val_str = f"{row['val_mean']:.4f}±{row['val_std']:.4f}"
        test_str = f"{row['test_mean']:.4f}±{row['test_std']:.4f}"
        print(f"{row['dataset']:<8} {row['label']:<18} {row['n_seeds']:<3} {row['params']:<8} "
              f"{train_str:<20} {val_str:<20} {test_str:<20}")
    print("=" * 115)


def write_csv(rows, path):
    with open(path, 'w') as f:
        f.write("dataset,basis,n_seeds,params,train_mean,train_std,val_mean,val_std,"
                "test_mean,test_std,nfe_mean,nfe_std\n")
        for row in rows:
            f.write(f"{row['dataset']},{row['basis']},{row['n_seeds']},{row['params']},"
                    f"{row['train_mean']:.6f},{row['train_std']:.6f},"
                    f"{row['val_mean']:.6f},{row['val_std']:.6f},"
                    f"{row['test_mean']:.6f},{row['test_std']:.6f},"
                    f"{row['nfe_mean']:.1f},{row['nfe_std']:.1f}\n")
    print(f"Wrote {path}")


def write_markdown(rows, path):
    with open(path, 'w') as f:
        f.write("| Dataset | Basis | N | Params | Train RelErr | Val RelErr | Test RelErr |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for row in rows:
            f.write(f"| {row['dataset']} | {row['label']} | {row['n_seeds']} | {row['params']} | "
                     f"{row['train_mean']:.4f} ± {row['train_std']:.4f} | "
                     f"{row['val_mean']:.4f} ± {row['val_std']:.4f} | "
                     f"{row['test_mean']:.4f} ± {row['test_std']:.4f} |\n")
    print(f"Wrote {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default='.')
    args = parser.parse_args()

    by_dataset_basis = load_results(args.results_dir)
    rows, total_problems = aggregate(by_dataset_basis)
    print_table(rows)

    write_csv(rows, os.path.join(args.results_dir, 'table5_results.csv'))
    write_markdown(rows, os.path.join(args.results_dir, 'table5_results.md'))

    if total_problems > 0:
        print(f"\n[SUMMARY] {total_problems} run(s) did not cleanly early-stop -- review "
              f"before treating this table as final.")
    else:
        print(f"\n[SUMMARY] All runs cleanly triggered early stopping.")


if __name__ == '__main__':
    main()
