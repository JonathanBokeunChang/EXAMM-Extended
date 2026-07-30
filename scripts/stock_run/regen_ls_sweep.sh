#!/bin/bash
# Long/short book-size sweep for the two TRUE walk-forward cells.
# bash only (zsh does not word-split).
#
#   bash scripts/stock_run/regen_ls_sweep.sh
#
# WHY ONLY TWO CELLS
# ------------------
# The walk-forward design is: train through year N-1, validate on year N, trade year N+1. Only two
# cells satisfy it:
#   cohort_2020  train <=2020, val 2021, trade 2022
#   cohort_2021  train <=2021, val 2022, trade 2023
# Trading cohort_2020 on 2023 uses a model two years stale and selected on a validation year that is
# no longer adjacent to the trading period. It is a different (easier to look good on, harder to
# defend) experiment and is excluded here.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT=${1:-$REPO/results/trading}; mkdir -p "$OUT"
CSV=$OUT/ls_sweep.csv
echo "cohort,trade_year,strategy,n_long,n_short,tc,strategy_pct,bh_pct,sp_pct" > "$CSV"

cell() { # cohort ws we year strategy n tc_flag tc_label
  local r s b p
  r=$(cd "$REPO" && timeout 600 python3 scripts/stock_run/trade_portfolio.py \
        --pred-dir "$REPO/test_output/mse_${1}/ensemble_test" \
        --data-dir "datasets/walkforward/$1" --strategy "$5" \
        --long "$6" --short "$6" $7 --window-start "$2" --window-end "$3" --expect-n 50 2>&1)
  grep -q "strategy return" <<<"$r" || { echo "FAILED $1 $4 $5 n=$6 $8" >&2; return 1; }
  s=$(grep "strategy return" <<<"$r" | grep -oE '[-+][0-9.]+%' | head -1 | tr -d '%')
  b=$(grep "equal-weight"    <<<"$r" | grep -oE '[-+][0-9.]+%' | head -1 | tr -d '%')
  p=$(grep "S&P"             <<<"$r" | grep -oE '[-+][0-9.]+%' | head -1 | tr -d '%')
  echo "$1,$4,$5,$6,$6,$8,$s,$b,$p" >> "$CSV"
  printf '  %-20s %s  %-30s n=%-3s %-7s %8s%%\n' "$1" "$4" "$5" "$6" "$8" "$s"
}

for n in 3 5 10 15 20; do
  for tc in "--use-tc:withTC" ":noTC"; do
    flag="${tc%%:*}"; lab="${tc##*:}"
    cell cohort_2020_aligned 2022-01-03 2022-12-30 2022 daily_long_short_return "$n" "$flag" "$lab"
    cell cohort_2021_aligned 2023-01-03 2023-12-29 2023 daily_long_short_return "$n" "$flag" "$lab"
  done
done
# hybrid at the headline size, for sensitivity
for tc in "--use-tc:withTC" ":noTC"; do
  flag="${tc%%:*}"; lab="${tc##*:}"
  cell cohort_2020_aligned 2022-01-03 2022-12-30 2022 daily_hybrid_long_short_return 10 "$flag" "$lab"
  cell cohort_2021_aligned 2023-01-03 2023-12-29 2023 daily_hybrid_long_short_return 10 "$flag" "$lab"
done
echo; echo "wrote $CSV ($(( $(wc -l < "$CSV") - 1 )) rows)"
