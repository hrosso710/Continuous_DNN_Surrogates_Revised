#!/bin/bash
# process_results.sh -- aggregates stationary_ode/results/*.json into
# Table 4 and Table 6-ready numbers. Run after run_experiment.sh (or
# run_test.sh, to sanity-check this script itself) completes.
#
# NOTE: aggregate_table4.py's static (identity) baseline value is a
# HARDCODED default (8.2132) -- see the ⚠️ note in run_experiment.sh. If
# that's been confirmed stale and rerun, pass the new value here:
#   ./process_results.sh --static_baseline_relerr <new_value>
# aggregate_table6.py does not use this flag -- it reports the real degree-3
# monomial/legendre train/val/test numbers directly, and will warn (not
# silently substitute the hardcoded value) if degree-3 basis='none' results
# are missing, since that run doesn't exist in run_experiment.sh yet.

set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python3 aggregate_table4.py --results_dir results "$@"
python3 aggregate_table6.py --results_dir results
