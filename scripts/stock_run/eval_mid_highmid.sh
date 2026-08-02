#!/bin/bash
# Evaluate + ensemble all 12 (set, cohort) mid_highmid_20yr_portfolios pooled training
# campaigns. Thin loop around the EXISTING, unmodified eval_ic_run.sh (the same tool
# that produced ensemble_test/ for the original pooled baseline) -- it already reads
# data_dir from each run's .config, so nothing about it needed to change for this
# dataset. Run this ON ANVIL (needs build/rnn_examples/evaluate_rnn). bash only.
#
#   sh scripts/stock_run/eval_mid_highmid.sh [SPLIT] [RUN_TAG]  # SPLIT defaults to test
#
# RUN_TAG selects which campaign to evaluate, matching the suffix
# anvil_mse_pooled_mid_highmid.sb appends to OUT_ROOT. Omit it for the baseline
# (the original 12 campaigns, whose dirs carry no suffix); pass the arm name for a
# hyperparameter ablation or a seed replicate, e.g.
#   sh scripts/stock_run/eval_mid_highmid.sh test k_bp20
#   sh scripts/stock_run/eval_mid_highmid.sh test rep2
#
# For each combo, writes:
#   test_output/mse_pooled_mid_highmid/<set>_<cohort>[_<tag>]/ensemble_<split>/  (ensembled)
#   test_output/mse_pooled_mid_highmid/<set>_<cohort>[_<tag>]/run_*/eval_<split>/ (per-run)
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SPLIT=${1:-test}
RUN_TAG=${2:-}

n=0
for SET in set1 set2 set3 set4; do
    for COHORT in cohort_2020 cohort_2021 cohort_2022; do
        OUT_ROOT="$REPO/test_output/mse_pooled_mid_highmid/${SET}_${COHORT}${RUN_TAG:+_$RUN_TAG}"
        if [ ! -d "$OUT_ROOT" ]; then
            echo "SKIP $SET/$COHORT: $OUT_ROOT not found" >&2
            continue
        fi
        echo "=== $SET / $COHORT${RUN_TAG:+ [$RUN_TAG]} ==="
        sh "$REPO/scripts/stock_run/eval_ic_run.sh" "$OUT_ROOT" "$SPLIT"
        echo
        n=$((n + 1))
    done
done

echo "evaluated $n / 12 combos (split=$SPLIT${RUN_TAG:+, tag=$RUN_TAG})"
