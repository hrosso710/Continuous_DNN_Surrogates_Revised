#!/bin/bash
# process_results.sh -- aggregates surrogate/results/results_table5_*.json
# into Table 5-ready numbers.
#
# ⚠️ NOT YET FUNCTIONAL: aggregate_table5.py hasn't been added to this repo
# yet. Once you add it (mirroring aggregate_table4.py's structure -- see
# ../stationary_ode/aggregate_table4.py), replace the `exit 1` block below
# with:
#     python3 aggregate_table5.py --results_dir results "$@"

set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "process_results.sh: aggregate_table5.py not yet added to this repo." >&2
echo "Results are sitting in results/results_table5_*.json, uncombined." >&2
exit 1
