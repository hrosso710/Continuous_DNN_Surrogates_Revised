#!/bin/bash
# process_results.sh -- aggregates stationary_ode/results/*.json into
# Table 3-ready numbers (manuscript numbering; this script's own filename
# stayed aggregate_table4.py -- see the note in the top-level README for why).
# Run after run_experiment.sh (or run_test.sh, to sanity-check this script
# itself) completes.
#
# NOTE: aggregate_table4.py's static (identity) baseline value is a
# HARDCODED default (8.2132) -- see the note in run_experiment.sh. If
# that's been confirmed stale and rerun, pass the new value here:
#   ./process_results.sh --static_baseline_relerr <new_value>
#
# RESOLVED (2026-08-06): Table 6 (a separate degree-3-only slice of this same
# sweep) was dropped from the manuscript -- that data is just the degree=3
# row already inside Table 3, so there's nothing distinct left to aggregate.
# aggregate_table6.py is accordingly unused; it's kept in the directory only
# as a historical/standalone reference, not called here.

set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python3 aggregate_table4.py --results_dir results "$@"