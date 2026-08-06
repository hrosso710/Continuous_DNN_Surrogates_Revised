#!/bin/bash
# process_results.sh -- aggregates surrogate/results/results_table4_*.json
# into Table 4-ready numbers.

set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python3 aggregate_table4.py --results_dir results "$@"
