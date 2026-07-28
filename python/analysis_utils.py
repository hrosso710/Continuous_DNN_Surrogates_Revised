"""
Analysis utilities for PROCESSING completed experiment results (loading and
aggregating the JSON summaries written by ode_demo_NEW.py / the future
surrogate-dataset experiment script).

For utilities used while RUNNING an experiment, see experiment_runtime.py.
"""

import glob
import json
import os
from typing import List, Optional

import pandas as pd


def load_run_json(path: str) -> dict:
    """Load a single run's results_json summary."""
    with open(path) as f:
        return json.load(f)


def load_all_results(results_dir: str, pattern: str = "results_table4_*.json") -> pd.DataFrame:
    """
    Load every run summary matching `pattern` under `results_dir` into one
    DataFrame, one row per run. Matches the field names ode_demo_NEW.py
    actually writes (see its `summary` dict) -- notably `reported_train_relerr`,
    `reported_val_relerr`, `reported_test_relerr`, `reported_test_nfe`,
    `reported_iter`, all computed on the same best-on-validation checkpoint.
    """
    paths = sorted(glob.glob(os.path.join(results_dir, pattern)))
    if not paths:
        return pd.DataFrame()

    rows = []
    for p in paths:
        try:
            row = load_run_json(p)
            row['_source_file'] = os.path.basename(p)
            rows.append(row)
        except Exception as e:
            print(f"Error loading {p}: {e}")

    return pd.DataFrame(rows)


def summarize_by_config(
    df: pd.DataFrame,
    group_cols: Optional[List[str]] = None,
    metric_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Aggregate mean +/- std across seeds for each (basis, degree) config --
    the shape Table 4/6 need: one row per config, mean/std over the 5 seeds.
    """
    if group_cols is None:
        group_cols = ['basis', 'degree']
    if metric_cols is None:
        metric_cols = [
            'reported_train_relerr', 'reported_val_relerr',
            'reported_test_relerr', 'reported_test_nfe',
        ]

    agg = df.groupby(group_cols)[metric_cols].agg(['mean', 'std', 'count'])
    agg.columns = ['_'.join(c) for c in agg.columns]
    return agg.reset_index()
