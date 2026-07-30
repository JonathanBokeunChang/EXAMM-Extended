#!/bin/bash
# Regenerate the per-run trading table for the results section.
#
# MUST be run with bash, not zsh: `set -- $cfg` and other unquoted word-splitting do not split in
# zsh, which silently collapses each config triple into a single argument. This has now bitten the
# project three times (examm multi-file inputs, coal column names, and this).
#
#   bash scripts/stock_run/regen_trading_table.sh [outdir]
#
# Strategy is dollar-neutral daily long/short, 10 long / 10 short, transaction costs ON, PRC prices.
# Benchmarks reported alongside: equal-weight buy-and-hold of the same 50 names, and the S&P (sprtrn).
# Equal-weight B&H is the correct benchmark for this book; the S&P is reported only for context.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT=${1:-$REPO/results/trading}
mkdir -p "$OUT"
CSV=$OUT/trading_per_run.csv
echo "cohort,year,arm,strategy_pct,bh_pct,sp_pct" > "$CSV"

cell() {  # cohort year window_start window_end arm pred_dir
  local cohort=$1 year=$2 ws=$3 we=$4 arm=$5 pdir=$6
  local r s b p
  r=$(cd "$REPO" && timeout 600 python3 scripts/stock_run/trade_portfolio.py \
        --pred-dir "$pdir" --data-dir "datasets/walkforward/$cohort" \
        --strategy daily_long_short_return --long 10 --short 10 --use-tc \
        --window-start "$ws" --window-end "$we" --expect-n 50 2>&1)
  if ! grep -q "strategy return" <<<"$r"; then
    echo "FAILED: $cohort $year $arm" >&2; echo "$r" | tail -3 >&2; return 1
  fi
  s=$(grep "strategy return" <<<"$r" | grep -oE '[-+][0-9.]+%' | head -1 | tr -d '%')
  b=$(grep "equal-weight"    <<<"$r" | grep -oE '[-+][0-9.]+%' | head -1 | tr -d '%')
  p=$(grep "S&P"             <<<"$r" | grep -oE '[-+][0-9.]+%' | head -1 | tr -d '%')
  echo "$cohort,$year,$arm,$s,$b,$p" >> "$CSV"
  printf '  %-22s %-6s %-10s %8s%%\n' "$cohort" "$year" "$arm" "$s"
}

sweep() {  # cohort year window_start window_end
  local cohort=$1 year=$2 ws=$3 we=$4
  local ex="${cohort/cohort/mse_cohort}"
  echo "=== $cohort / $year ==="
  for d in "$REPO"/test_output/"$ex"/run_*/eval_test; do
    [ -d "$d" ] || continue
    cell "$cohort" "$year" "$ws" "$we" "$(basename "$(dirname "$d")")" "$d"
  done
  local ens="$REPO/test_output/$ex/ensemble_test"
  [ -d "$ens" ] && cell "$cohort" "$year" "$ws" "$we" "ENSEMBLE" "$ens"
}

sweep cohort_2020_aligned 2022 2022-01-03 2022-12-30
sweep cohort_2020_aligned 2023 2023-01-03 2023-12-29
sweep cohort_2021_aligned 2023 2023-01-03 2023-12-29

echo; echo "wrote $CSV ($(( $(wc -l < "$CSV") - 1 )) rows)"
