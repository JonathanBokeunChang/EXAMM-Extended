#!/bin/bash
# Build all 12 (set, cohort) walk-forward splits for mid_highmid_20yr_portfolios, then
# align each for cross-sectional training. bash only (zsh does not word-split unquoted
# vars -- has bitten this project three times already).
#
#   sh scripts/stock_run/build_mid_highmid_splits.sh
#
# Requires datasets/mid_highmid_20yr_portfolios_clean/ to already exist (run
# clean_mid_highmid.py first). Cohorts named by train-cutoff year, matching this
# project's existing convention (not by trade year):
#   cohort_2020: train <=2020-12-31, val 2021, test 2022 (trades 2022)
#   cohort_2021: train <=2021-12-31, val 2022, test 2023 (trades 2023)
#   cohort_2022: train <=2022-12-31, val 2023, test 2024 (trades 2024)
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CLEAN_SRC="$REPO/datasets/mid_highmid_20yr_portfolios_clean"
OUT_BASE="$REPO/datasets/walkforward/mid_highmid"

if [ ! -d "$CLEAN_SRC" ]; then
    echo "ERROR: $CLEAN_SRC not found -- run clean_mid_highmid.py first" >&2
    exit 1
fi

# cohort_name cutoff val-year test-year
COHORTS=(
    "cohort_2020 2020-12-31 2021 2022"
    "cohort_2021 2021-12-31 2022 2023"
    "cohort_2022 2022-12-31 2023 2024"
)

n=0
for SET in set1 set2 set3 set4; do
    for spec in "${COHORTS[@]}"; do
        read -r COHORT CUTOFF VALYEAR TESTYEAR <<< "$spec"
        OUT="$OUT_BASE/$SET/$COHORT"
        echo "=== $SET / $COHORT (cutoff $CUTOFF, val $VALYEAR, test $TESTYEAR) ==="
        python3 "$REPO/scripts/stock_run/build_walkforward_splits.py" --continuous \
            --source-dir "$CLEAN_SRC/$SET" \
            --cutoff "$CUTOFF" --val-year "$VALYEAR" --test-year "$TESTYEAR" \
            --out "$OUT"
        python3 "$REPO/scripts/stock_run/align_cohort.py" \
            --in "$OUT" --out "${OUT}_aligned"
        echo
        n=$((n + 1))
    done
done

echo "wrote $n (set, cohort) walk-forward splits + aligned copies -> $OUT_BASE"
