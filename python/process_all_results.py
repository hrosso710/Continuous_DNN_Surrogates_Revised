#!/usr/bin/env python3
"""
Master script to process all Python-side experiment results for the paper.

Mirrors the structural convention of the advisor's rampde paper/ directory
(process_all_results.py calling each experiment's own process_results.sh),
adapted to this paper's experiments.

Experiments processed:
- stationary_ode: Table 3, Figure 5
- surrogate: Table 4

Usage:
    python process_all_results.py [--experiments EXPS]
"""

import subprocess
import sys
from pathlib import Path
import argparse
from typing import Dict, List


def run_experiment_script(experiment: str, script_path: Path) -> bool:
    print("\n" + "=" * 70)
    print(f"Processing {experiment.upper()}")
    print("=" * 70)
    try:
        subprocess.run([str(script_path)], cwd=script_path.parent, check=True)
        print(f"\n{experiment.upper()} processing completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n{experiment.upper()} processing failed with exit code {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"\nProcessing script not found: {script_path}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Process all Python-side experiment results")
    parser.add_argument(
        "--experiments", type=str, default="stationary_ode,surrogate",
        help="Comma-separated experiment list (default: all available)"
    )
    args = parser.parse_args()

    experiments_to_run = [e.strip() for e in args.experiments.split(",")]
    results: Dict[str, bool] = {}

    for experiment in experiments_to_run:
        script_path = Path(f"{experiment}/process_results.sh")
        if not script_path.exists():
            print(f"\nSkipping {experiment}: {script_path} not found")
            results[experiment] = False
            continue
        results[experiment] = run_experiment_script(experiment, script_path)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for experiment, ok in results.items():
        print(f"{'OK' if ok else 'FAILED'}: {experiment}")

    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
