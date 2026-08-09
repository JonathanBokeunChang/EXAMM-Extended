#!/bin/bash
# Ensemble the fixed-topology LSTM/GRU campaign (anvil_train_rnn.sb) and package the result.
#
#   bash scripts/stock_run/eval_rnn_campaign.sh              # all 28 cells
#   bash scripts/stock_run/eval_rnn_campaign.sh lstm         # one architecture
#
# RUN THIS ON ANVIL. It deliberately does ENSEMBLING ONLY, not trading: eval_ensemble_ic.py is
# stdlib-only and runs under Anvil's bare python, while trade_portfolio.py needs numpy/pandas, which
# the compute nodes' `module load gcc/openmpi` python does not have. Scoring returns on the laptop
# also keeps LSTM/GRU going through the exact same scorer every other model in the study used --
# a second implementation would be a second thing to trust.
#
# It also shrinks the transfer by 20x. The raw campaign is 280 runs x 50 tickers ~ 280 MB; the 28
# ensembles are 28 x 50 ~ 14 MB, and the ensemble is the only thing the paper reports anyway.
#
# THE 10 RUNS ARE REPEATED TRAINING RUNS, NOT SEEDS. train_rnn has no --seed flag; the runs differ
# because initialisation is stochastic (verified by sha256 -- two identical invocations give
# different genomes). This is exactly the protocol Lyu et al. describe ("10 repeated training
# runs"), and the prediction mean over them is what gets reported.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO" || exit 1
TYPES=${1:-"lstm gru"}
# Which normalisation arm to score. avg_std_dev writes rnn_<type>_<cell>, instance writes
# rnn_<type>_inst_<cell>; the two must be scored separately or the ensembles mix schemes.
NORM=${NORM:-avg_std_dev}
case "$NORM" in
  avg_std_dev) NORM_TAG="" ;;
  instance)    NORM_TAG="_inst" ;;
  *) echo "ERROR: NORM must be avg_std_dev or instance (got '$NORM')" >&2; exit 2 ;;
esac
SPLIT=${SPLIT:-test}
# Drop runs whose average prediction exceeds K standard deviations of the target. One diverged run
# shifts a prediction-MEAN ensemble by level/N, and because Algorithm 2 gates on signs that closes
# the book completely while leaving IC untouched -- measured on set3/2022, where run_5 sat at -0.463
# against nine runs in -0.003..+0.007 and took the cell from 251 tradeable days to 0.
# Set MAX_LEVEL_SD= (empty) to reproduce the unscreened numbers.
MAX_LEVEL_SD=${MAX_LEVEL_SD-10}
OUT_TAR=${OUT_TAR:-$REPO/rnn_ensembles${NORM_TAG}.tar.gz}

CELLS=()
for S in set1 set2 set3 set4; do
  for C in cohort_2020 cohort_2021 cohort_2022; do CELLS+=( "${S}_${C}_aligned" ); done
done
CELLS+=( "dsA_cohort_2020_aligned" "dsA_cohort_2021_aligned" )

SUMMARY="$REPO/rnn_campaign_ic${NORM_TAG}.csv"
echo "type,cell,runs,dropped,stocks,dates,mean_ic,ic_ir,hit_rate,gate,gate_pct" > "$SUMMARY"

ok=0; miss=0; fail=0
for T in $TYPES; do
  for CELL in "${CELLS[@]}"; do
    ROOT="test_output/rnn_${T}${NORM_TAG}_${CELL}"
    if [ ! -d "$ROOT" ]; then
      echo "SKIP  $T $CELL: no $ROOT" >&2; miss=$((miss+1)); continue
    fi
    n=$(ls -d "$ROOT"/run_*/ 2>/dev/null | wc -l | tr -d ' ')
    LOG=$(mktemp)
    # --eval-subdir . because anvil_train_rnn.sb runs evaluate_rnn inline and writes the per-ticker
    # CSVs straight into run_N/, where EXAMM's pipeline would have put them in run_N/eval_test/.
    python3 scripts/stock_run/eval_ensemble_ic.py \
      --run-root "$ROOT" --split "$SPLIT" --eval-subdir . \
      ${MAX_LEVEL_SD:+--max-level-sd "$MAX_LEVEL_SD"} \
      --emit-dir "$ROOT/ensemble_${SPLIT}" > "$LOG" 2>&1
    rc=$?
    if [ $rc -ne 0 ]; then
      echo "FAIL  $T $CELL ($n runs): $(tail -2 "$LOG" | tr '\n' ' ')" >&2
      fail=$((fail+1)); rm -f "$LOG"; continue
    fi
    IC=$(grep -oE 'mean IC       : [-+0-9.]+' "$LOG" | awk '{print $NF}')
    IR=$(grep -oE 'IC info ratio : [-+0-9.]+' "$LOG" | awk '{print $NF}')
    HR=$(grep -oE 'hit rate      : [0-9.]+%' "$LOG" | awk '{print $NF}')
    ST=$(grep -oE 'universe      : [0-9]+' "$LOG" | awk '{print $NF}')
    DT=$(grep -oE 'dates         : [0-9]+' "$LOG" | awk '{print $NF}')
    RN=$(grep -oE 'runs ensembled: [0-9]+' "$LOG" | awk '{print $NF}')
    # grep -c ALREADY prints 0 when it matches nothing; it just exits non-zero doing so. The
    # `|| echo 0` that used to be here appended a SECOND zero, making DR the two-line string
    # "0\n0" and breaking the [ -gt ] test below on every clean cell.
    DR=$(grep -c '^  DROPPED ' "$LOG" 2>/dev/null); DR=${DR:-0}
    GA=$(grep -oE 'gate days     : [0-9]+/[0-9]+' "$LOG" | awk '{print $NF}')
    GP=$(grep -oE 'gate days     :.*\(([0-9.]+)%' "$LOG" | grep -oE '[0-9.]+%' | tail -1)
    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' "$T" "$CELL" "${RN:-$n}" "$DR" "$ST" "$DT" "$IC" "$IR" "$HR" "${GA:-}" "${GP:-}" >> "$SUMMARY"
    printf '  %-4s %-26s %2s runs (%s drop)  IC %-10s IR %-7s gate %-9s hit %s\n' "$T" "$CELL" "${RN:-$n}" "$DR" "$IC" "$IR" "${GA:-?}" "$HR"
    [ "$DR" -gt 0 ] && grep '^  DROPPED ' "$LOG" | sed 's/^/     /' 
    ok=$((ok+1)); rm -f "$LOG"
  done
done

echo
echo "ensembled $ok cell(s); $miss missing, $fail failed"
[ $fail -eq 0 ] || echo "WARNING: $fail cell(s) failed -- do not treat the tar as complete" >&2

# Package the ensembles plus the per-run timing/config, so the laptop can report training cost and
# provenance without a second transfer. Built as an explicit list rather than a find|xargs pipeline:
# the members live at three different depths, and a pipeline that silently emits nothing produces a
# tar that looks fine and is missing the timings.
echo "==> packaging -> $OUT_TAR"
LIST=$(mktemp)
# EXACT arm membership, not a glob. With NORM_TAG empty the pattern rnn_*${NORM_TAG}_*/ degrades
# to rnn_*_*/, which also matches every rnn_<type>_inst_<cell> directory -- so the avg_std_dev
# tarball silently carried BOTH arms (1,174 entries where 28 cells give ~589). Rebuild the list
# from the cells this run actually scored instead.
for d in $(for T in $TYPES; do for C in "${CELLS[@]}"; do
             printf 'test_output/rnn_%s%s_%s/\n' "$T" "$NORM_TAG" "$C"; done; done); do
  [ -d "$d" ] || continue
  [ -d "$d/ensemble_${SPLIT}" ] && printf '%s\n' "${d}ensemble_${SPLIT}"
  for f in "$d"run_*/timing.json "$d"run_*/.config; do [ -f "$f" ] && printf '%s\n' "$f"; done
done > "$LIST"
printf '%s\n' "$(basename "$SUMMARY")" >> "$LIST"
COUNT=$(wc -l < "$LIST" | tr -d ' ')
[ "$COUNT" -gt 1 ] || { echo "ERROR: nothing to package -- did any cell ensemble?" >&2; rm -f "$LIST"; exit 1; }
tar czf "$OUT_TAR" -T "$LIST"
rm -f "$LIST"
echo "    $(du -h "$OUT_TAR" | cut -f1)  $OUT_TAR  ($COUNT entries)"
echo
echo "Copy it to the laptop with:"
echo "  scp anvil:$OUT_TAR ."
