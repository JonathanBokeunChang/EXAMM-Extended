#!/bin/bash
# Evaluate + ensemble all 12 (set, cohort) mid_highmid_20yr_portfolios pooled training
# campaigns. Thin loop around the EXISTING, unmodified eval_ic_run.sh (the same tool
# that produced ensemble_test/ for the original pooled baseline) -- it already reads
# data_dir from each run's .config, so nothing about it needed to change for this
# dataset. Run this ON ANVIL (needs build/rnn_examples/evaluate_rnn). bash only.
#
#   sh scripts/stock_run/eval_mid_highmid.sh [SPLIT]     # SPLIT defaults to test
#
# For each combo, writes:
#   test_output/mse_pooled_mid_highmid/<set>_<cohort>/ensemble_<split>/  (ensembled preds)
#   test_output/mse_pooled_mid_highmid/<set>_<cohort>/run_*/eval_<split>/  (per-run preds)
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SPLIT=${1:-test}

n=0
for SET in set1 set2 set3 set4; do
    for COHORT in cohort_2020 cohort_2021 cohort_2022; do
        OUT_ROOT="$REPO/test_output/mse_pooled_mid_highmid/${SET}_${COHORT}"
        if [ ! -d "$OUT_ROOT" ]; then
            echo "SKIP $SET/$COHORT: $OUT_ROOT not found" >&2
            continue
        fi
        echo "=== $SET / $COHORT ==="
        sh "$REPO/scripts/stock_run/eval_ic_run.sh" "$OUT_ROOT" "$SPLIT"
        echo
        n=$((n + 1))
    done
done

echo "evaluated $n / 12 combos (split=$SPLIT)"
