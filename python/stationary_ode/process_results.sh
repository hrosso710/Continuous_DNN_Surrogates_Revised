#!/bin/bash
# process_results.sh -- aggregates stationary_ode/results/*.json into
# Table 3-ready numbers. Run after run_experiment.sh (or run_test.sh, to
# sanity-check this script itself) completes.
#
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python3 aggregate_table3.py --results_dir results "$@"
