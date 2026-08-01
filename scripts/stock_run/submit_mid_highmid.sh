#!/bin/bash
# Submit all 12 (set, cohort) pooled EXAMM training arrays for mid_highmid_20yr_portfolios.
# Run this ON ANVIL (needs sbatch). bash only.
#
#   sh scripts/stock_run/submit_mid_highmid.sh              # full 10-run arrays, all 12 combos
#   sh scripts/stock_run/submit_mid_highmid.sh --canary     # array=0 only, all 12 combos (run
#                                                            # this first; check logs before
#                                                            # the full submission)
#
# Requires datasets/walkforward/mid_highmid/<set>/<cohort>_aligned/ to already exist on
# Anvil (build locally with build_mid_highmid_splits.sh, then transfer).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT="$REPO/scripts/stock_run/anvil_mse_pooled_mid_highmid.sb"

ARRAY="0-9"
[ "${1:-}" = "--canary" ] && ARRAY="0"

mkdir -p "$REPO/slurm_logs"

n=0
for SET in set1 set2 set3 set4; do
    for COHORT in cohort_2020 cohort_2021 cohort_2022; do
        echo "submitting SET=$SET COHORT=$COHORT --array=$ARRAY"
        sbatch --export=ALL,SET="$SET",COHORT="$COHORT" --array="$ARRAY" "$SCRIPT"
        n=$((n + 1))
    done
done

echo "submitted $n jobs (array=$ARRAY each)"
[ "$ARRAY" = "0" ] && echo "canary mode -- check slurm_logs/mh_*_0.out for 'FINISHED' on all 12 before running without --canary"
