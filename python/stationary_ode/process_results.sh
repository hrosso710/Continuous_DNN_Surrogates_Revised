#!/bin/bash
# process_results.sh -- aggregates stationary_ode/results/*.json into
# Table 4/6-ready numbers. Run after run_experiment.sh (or run_test.sh, to
# sanity-check this script itself) completes.
#
# NOTE: aggregate_table4.py's static (identity) baseline value is a
# HARDCODED default (8.2132) -- see the ⚠️ note in run_experiment.sh. If
# that's been confirmed stale and rerun, pass the new value here:
#   ./process_results.sh --static_baseline_relerr <new_value>

set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python3 aggregate_table4.py --results_dir results "$@"
