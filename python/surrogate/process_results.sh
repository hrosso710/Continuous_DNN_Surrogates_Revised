#!/bin/bash
# process_results.sh -- aggregates surrogate/results/results_table5_*.json
# into Table 5-ready numbers.

set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python3 aggregate_table5.py --results_dir results "$@"
