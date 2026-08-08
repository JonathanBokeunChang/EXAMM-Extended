#!/bin/bash
# Evaluate a COMBINED (wide) EXAMM campaign: genome -> wide predictions -> 50 per-ticker files
# -> 10-run ensemble -> IC, and print the trade command.
#
#   bash scripts/stock_run/eval_combined_run.sh test_output/mse_combined_cohort_2021_aligned
#   SPLIT=val bash scripts/stock_run/eval_combined_run.sh <OUT_ROOT>
#
# WHY A SEPARATE DRIVER. eval_ic_run.sh loops evaluate_rnn over 50 narrow *_test.csv files, which is
# the pooled/individual shape. Combined has ONE wide test file and emits ONE ~400-column prediction
# file, so that loop cannot be reused. evaluate_combined.sh (untracked) does run evaluate_rnn on a
# wide file but is hardcoded to datasets/701515_split -- the legacy 70/15/15 calendar whose test
# window opens inside the ablation cohorts' training span -- and reports only MAE/MSE, not IC or
# trading. This is the missing first link in the chain.
#
# AFTER THE SPLIT, EVERYTHING DOWNSTREAM IS SHARED. run_N/eval_<split>/ ends up holding exactly the
# same 50 <TICKER>_<split>_predictions.csv files pooled produces, so eval_ensemble_ic.py and
# trade_portfolio.py run unchanged. That shared path is the whole reason the three constructions in
# tab:pooling-ablation are comparable rather than three separate measurements.
#
# TWO DATA DIRS, AND THEY ARE NOT INTERCHANGEABLE. The WIDE panel supplies the model's input; the
# PER-TICKER cohort supplies ground truth for the splitter's expected==RET[t+1] assert. Passing the
# same path for both would make that assert vacuous.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO" || exit 1
OUT_ROOT=${1:?usage: eval_combined_run.sh OUT_ROOT [DATA_DIR]}
SPLIT=${SPLIT:-test}
BIN=${BIN:-build/rnn_examples/evaluate_rnn}

# Derive the per-ticker cohort from the run tree name: mse_combined_<COHORT>_combined -> <COHORT>
BASE=$(basename "$OUT_ROOT"); BASE=${BASE#mse_combined_}; BASE=${BASE%%_rep*}
DATA_DIR=${2:-datasets/walkforward/${BASE%_combined}}
WIDE_DIR=${WIDE_DIR:-datasets/walkforward/$BASE}
WIDE="$WIDE_DIR/combined_predictors_${SPLIT}.csv"

[ -x "$BIN" ]     || { echo "ERROR: $BIN not found -- build evaluate_rnn first" >&2; exit 1; }
[ -f "$WIDE" ]    || { echo "ERROR: $WIDE missing -- run build_combined_panel.py" >&2; exit 1; }
[ -d "$DATA_DIR" ]|| { echo "ERROR: $DATA_DIR missing (per-ticker truth for the alignment check)" >&2; exit 1; }
echo "OUT_ROOT=$OUT_ROOT  SPLIT=$SPLIT  wide=$WIDE  truth=$DATA_DIR"

n_runs=0
for RUNDIR in "$OUT_ROOT"/run_*/; do
    [ -d "$RUNDIR" ] || continue
    if [ ! -f "$RUNDIR/.done" ]; then echo "skip $(basename "$RUNDIR"): no .done"; continue; fi
    GB=$(ls "$RUNDIR"/global_best_genome_*.bin 2>/dev/null \
         | sed 's/.*_\([0-9]*\)\.bin/\1 &/' | sort -rn | head -1 | cut -d' ' -f2-)
    if [ -z "$GB" ]; then echo "skip $(basename "$RUNDIR"): no genome"; continue; fi

    EVAL="$RUNDIR/eval_$SPLIT"
    RAW="$RUNDIR/eval_${SPLIT}_wide"
    rm -rf "$EVAL" "$RAW"; mkdir -p "$EVAL" "$RAW"

    "$BIN" --genome_file "$GB" --testing_filenames "$WIDE" --time_offset 1 \
        --output_directory "$RAW" --std_message_level ERROR --file_message_level ERROR \
        > "$RAW/evaluate.log" 2>&1
    PRED="$RAW/combined_predictors_${SPLIT}_predictions.csv"
    if [ ! -f "$PRED" ]; then
        echo "  $(basename "$RUNDIR"): FAILED -- no predictions written; see $RAW/evaluate.log" >&2
        continue
    fi

    # The splitter asserts expected==RET[t+1] to 1e-10 per ticker and refuses on mismatch, so a
    # wrong-calendar pairing stops here rather than becoming a plausible-looking IC.
    if ! python3 scripts/stock_run/split_combined_predictions.py \
            --pred "$PRED" --data-dir "$DATA_DIR" --out "$EVAL" --split "$SPLIT" > "$RAW/split.log" 2>&1; then
        echo "  $(basename "$RUNDIR"): SPLIT FAILED -- $(tail -1 "$RAW/split.log")" >&2
        continue
    fi
    # evaluate_rnn/split name files <TICKER>_test_predictions.csv; eval_ensemble_ic.py globs
    # *_<split>_predictions.csv, which is the same string for SPLIT=test.
    n_pred=$(ls "$EVAL"/*_"$SPLIT"_predictions.csv 2>/dev/null | wc -l | tr -d ' ')
    echo "  $(basename "$RUNDIR"): genome=$(basename "$GB")  split into $n_pred tickers"
    # HARD FAIL on a short run -- the same trap eval_ic_run.sh documents: a genome that aborts
    # leaves an empty eval dir, the loop carries on, and the ensemble is silently computed from
    # whichever runs survived.
    if [ "$n_pred" -ne 50 ]; then
        echo "ERROR: expected 50 per-ticker files, got $n_pred in $EVAL" >&2; exit 1
    fi
    n_runs=$((n_runs + 1))
done

if [ "$n_runs" -eq 0 ]; then echo "ERROR: no finished runs evaluated" >&2; exit 1; fi
echo ""
python3 scripts/stock_run/eval_ensemble_ic.py \
    --run-root "$OUT_ROOT" --split "$SPLIT" --target-col RET \
    --emit-dir "$OUT_ROOT/ensemble_$SPLIT"
echo ""
echo "To trade the ensemble:"
echo "  python3 scripts/stock_run/trade_portfolio.py \\"
echo "    --pred-dir $OUT_ROOT/ensemble_$SPLIT --data-dir $DATA_DIR \\"
echo "    --align suffix --expect-n 50 --long 10 --short 10 \\"
echo "    --strategy daily_hybrid_long_short_return"
