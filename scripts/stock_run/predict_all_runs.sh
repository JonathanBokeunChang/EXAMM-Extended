#!/bin/sh
# Generate + verify held-out TEST predictions for EVERY run (not just the
# validation-selected one) of an individual-stock cohort campaign -- the input
# to an ensemble trading comparison. Companion to predict_selected.sh, which
# only ever fetches the single best-per-stock genome; this fetches all of them
# so runs can be averaged into an ensemble the way pooled EXAMM already is.
#
# Usage: sh scripts/stock_run/predict_all_runs.sh <COHORT>
#   e.g. sh scripts/stock_run/predict_all_runs.sh cohort_2020_aligned
#
# Expects test_output/<COHORT>_individual/<TICKER>/run_<R>/global_best_genome_*.bin
# (anvil_cohort_individual.sb's output layout) and datasets/walkforward/<COHORT>/.
#
# For each (ticker, run) with a .done marker: runs evaluate_rnn on <ticker>_test.csv,
# writes predictions/<ticker>_run<R>_test_predictions.csv. Same per-output guards as
# predict_selected.sh (row count = test rows - 1, no NaN/Inf -- a NaN prediction
# would be silently longed/shorted by the trading code's argsort).
#
# On full success writes predictions/MANIFEST.csv (ticker,run,n_pred,md5) and tars
# the folder for download via Open OnDemand.

COHORT=$1
if [ -z "$COHORT" ]; then
    echo "usage: predict_all_runs.sh <COHORT>  (e.g. cohort_2020_aligned)" >&2
    exit 1
fi

REPO=$(cd "$(dirname "$0")/../.." && pwd)
BASE=$REPO/test_output/${COHORT}_individual
DATA=$REPO/datasets/walkforward/$COHORT
BIN=$REPO/build/rnn_examples/evaluate_rnn
PRED_DIR=$BASE/predictions
MANIFEST=$PRED_DIR/MANIFEST.csv
TAR=$BASE/${COHORT}_individual_predictions.tar.gz

if [ ! -x "$BIN" ]; then
    echo "ERROR: $BIN not found -- build evaluate_rnn first" >&2
    exit 1
fi
if [ ! -d "$BASE" ]; then
    echo "ERROR: $BASE not found -- run anvil_cohort_individual.sb first" >&2
    exit 1
fi
if [ ! -d "$DATA" ]; then
    echo "ERROR: $DATA not found" >&2
    exit 1
fi

if command -v md5sum > /dev/null 2>&1; then
    md5_of() { md5sum "$1" | awk '{print $1}'; }
elif command -v md5 > /dev/null 2>&1; then
    md5_of() { md5 -q "$1"; }
else
    echo "ERROR: neither md5sum nor md5 found" >&2
    exit 1
fi

rm -rf "$PRED_DIR"
rm -f "$TAR"   # a stale tar from a previous (partial) attempt must never survive
mkdir -p "$PRED_DIR"
echo "ticker,run,n_pred,md5" > "$MANIFEST"

n_ok=0
n_missing=0
n_fail=0

for tdir in "$BASE"/*/; do
    ticker=$(basename "$tdir")
    [ "$ticker" = "predictions" ] && continue
    testcsv="$DATA/${ticker}_test.csv"
    if [ ! -f "$testcsv" ]; then
        echo "FAIL: $ticker: no test file at $testcsv" >&2
        n_fail=$((n_fail + 1)); continue
    fi
    for rdir in "$tdir"run_*/; do
        [ -d "$rdir" ] || continue
        run=$(basename "$rdir" | sed 's/^run_//')
        if [ ! -f "${rdir}.done" ]; then
            echo "SKIP: $ticker run $run (no .done -- unfinished or failed training)" >&2
            n_missing=$((n_missing + 1)); continue
        fi
        bin=$(ls "$rdir"global_best_genome_*.bin 2>/dev/null | head -1)
        if [ -z "$bin" ]; then
            echo "FAIL: $ticker run $run: .done present but no genome .bin found" >&2
            n_fail=$((n_fail + 1)); continue
        fi

        out_tag="${ticker}_run${run}"
        "$BIN" --genome_file "$bin" --testing_filenames "$testcsv" \
            --time_offset 1 --output_directory "$PRED_DIR" \
            --std_message_level ERROR --file_message_level NONE > /dev/null 2>&1

        # evaluate_rnn names its output after the genome's input stem
        # (<ticker>_test_predictions.csv); rename per-run so 10 runs don't collide.
        raw="$PRED_DIR/${ticker}_test_predictions.csv"
        pred="$PRED_DIR/${out_tag}_test_predictions.csv"
        if [ ! -f "$raw" ]; then
            echo "FAIL: $ticker run $run: no predictions produced" >&2
            n_fail=$((n_fail + 1)); continue
        fi
        mv "$raw" "$pred"

        n_test=$(($(wc -l < "$testcsv") - 1))
        n_pred=$(($(wc -l < "$pred") - 1))
        if [ "$n_pred" -ne "$((n_test - 1))" ]; then
            echo "FAIL: $ticker run $run: $n_pred pred rows, expected $((n_test - 1))" >&2
            rm -f "$pred"
            n_fail=$((n_fail + 1)); continue
        fi
        if grep -qiE '(^|,)-?(nan|inf)(,|$)' "$pred"; then
            echo "FAIL: $ticker run $run: non-finite value in predictions" >&2
            rm -f "$pred"
            n_fail=$((n_fail + 1)); continue
        fi

        echo "${ticker},${run},${n_pred},$(md5_of "$pred")" >> "$MANIFEST"
        n_ok=$((n_ok + 1))
    done
done

echo "###-------------------###"
echo "predictions: $n_ok ok, $n_missing missing (.done absent), $n_fail failed"

if [ "$n_ok" -eq 0 ]; then
    echo "ERROR: nothing succeeded -- nothing to package" >&2
    exit 1
fi

tar -czf "$TAR" -C "$BASE" predictions
echo "packaged for download:"
echo "  $TAR"
echo "download via https://ondemand.anvil.rcac.purdue.edu -> Files, then extract"
echo "into test_output/${COHORT}_individual/ on your machine."
