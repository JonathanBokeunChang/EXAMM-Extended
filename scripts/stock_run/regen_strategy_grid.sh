#!/bin/bash
# Ensemble trading across STRATEGIES x TC on/off x year, for the unified results table.
# bash only (zsh does not word-split).
#
#   bash scripts/stock_run/regen_strategy_grid.sh
#
# Strategies swept (the three that consume long/short counts, spanning the exposure spectrum):
#   daily_long_short_return         dollar-neutral, 10 long / 10 short  <- headline
#   daily_hybrid_long_short_return  hybrid exposure
#   daily_long_return               long-only, 10 names
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT=${1:-$REPO/results/trading}; mkdir -p "$OUT"
CSV=$OUT/strategy_grid.csv
echo "cohort,year,strategy,tc,strategy_pct,bh_pct,sp_pct" > "$CSV"

cell() { # cohort ws we year strategy tc_flag tc_label pred_dir
  local r s b p
  r=$(cd "$REPO" && timeout 600 python3 scripts/stock_run/trade_portfolio.py \
        --pred-dir "$8" --data-dir "datasets/walkforward/$1" --strategy "$5" \
        --long 10 --short 10 $6 --window-start "$2" --window-end "$3" --expect-n 50 2>&1)
  grep -q "strategy return" <<<"$r" || { echo "FAILED $1 $4 $5 $7" >&2; return 1; }
  s=$(grep "strategy return" <<<"$r" | grep -oE '[-+][0-9.]+%' | head -1 | tr -d '%')
  b=$(grep "equal-weight"    <<<"$r" | grep -oE '[-+][0-9.]+%' | head -1 | tr -d '%')
  p=$(grep "S&P"             <<<"$r" | grep -oE '[-+][0-9.]+%' | head -1 | tr -d '%')
  echo "$1,$4,$5,$7,$s,$b,$p" >> "$CSV"
  printf '  %-20s %-9s %-32s %-7s %8s%%  (B&H %7s%%  S&P %7s%%)\n' "$1" "$4" "$5" "$7" "$s" "$b" "$p"
}

for strat in daily_long_short_return daily_hybrid_long_short_return daily_long_return; do
  echo "=== $strat ==="
  for tc in "--use-tc:withTC" ":noTC"; do
    flag="${tc%%:*}"; lab="${tc##*:}"
    cell cohort_2020_aligned 2022-01-03 2022-12-30 2022 "$strat" "$flag" "$lab" \
         "$REPO/test_output/mse_cohort_2020_aligned/ensemble_test"
    cell cohort_2020_aligned 2023-01-03 2023-12-29 2023 "$strat" "$flag" "$lab" \
         "$REPO/test_output/mse_cohort_2020_aligned/ensemble_test"
    cell cohort_2020_aligned 2022-01-03 2023-12-29 FULL "$strat" "$flag" "$lab" \
         "$REPO/test_output/mse_cohort_2020_aligned/ensemble_test"
    cell cohort_2021_aligned 2023-01-03 2023-12-29 2023 "$strat" "$flag" "$lab" \
         "$REPO/test_output/mse_cohort_2021_aligned/ensemble_test"
  done
done
echo; echo "wrote $CSV ($(( $(wc -l < "$CSV") - 1 )) rows)"
