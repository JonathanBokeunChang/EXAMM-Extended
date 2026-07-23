#!/bin/bash
# Evaluate an IC (or MSE) training campaign and report the ENSEMBLE cross-sectional IC.
#
# For every finished run_R/ under OUT_ROOT, takes the best genome (highest-index
# global_best_genome_*.bin = best fitness found), evaluates it on all 50 stocks'
# <split> files, then averages predictions across runs and computes the daily
# cross-sectional Spearman IC (scripts/stock_run/eval_ensemble_ic.py).
#
# Usage: sh scripts/stock_run/eval_ic_run.sh <OUT_ROOT> [SPLIT]
#   e.g. sh scripts/stock_run/eval_ic_run.sh test_output/ic_pearson_cohort_2021_aligned test
#        sh scripts/stock_run/eval_ic_run.sh test_output/mse_cohort_2021_aligned test
#
# SPLIT defaults to test (the real number); use val to cross-check against the
# genomes' recorded validation fitness. Emits ensembled predictions to
# <OUT_ROOT>/ensemble_<split>/ for trade_portfolio.py.
set -u

OUT_ROOT=${1:?usage: eval_ic_run.sh <OUT_ROOT> [SPLIT]}
SPLIT=${2:-test}
BIN=build/rnn_examples/evaluate_rnn

if [ ! -x "$BIN" ]; then echo "ERROR: $BIN not found -- build evaluate_rnn first" >&2; exit 1; fi

# recover the dataset used, from any run's provenance
CFG=$(ls "$OUT_ROOT"/run_*/.config 2>/dev/null | head -1)
if [ -z "$CFG" ]; then echo "ERROR: no run_*/.config under $OUT_ROOT" >&2; exit 1; fi
DATA_DIR=$(sed -n 's/^data_dir=//p' "$CFG")
if [ ! -d "$DATA_DIR" ]; then echo "ERROR: data dir '$DATA_DIR' (from .config) missing" >&2; exit 1; fi
# The trained output column: RET for raw/IC arms, RET_CS for the z-score-MSE arm.
# (older .config files predate this key -> default RET.)
TARGET=$(sed -n 's/^target=//p' "$CFG"); TARGET=${TARGET:-RET}
echo "OUT_ROOT=$OUT_ROOT  SPLIT=$SPLIT  DATA_DIR=$DATA_DIR  TARGET=$TARGET"

n_runs=0
for RUNDIR in "$OUT_ROOT"/run_*/; do
    [ -d "$RUNDIR" ] || continue
    if [ ! -f "$RUNDIR/.done" ]; then echo "skip $(basename "$RUNDIR"): no .done"; continue; fi
    GB=$(ls "$RUNDIR"/global_best_genome_*.bin 2>/dev/null \
         | sed 's/.*_\([0-9]*\)\.bin/\1 &/' | sort -rn | head -1 | cut -d' ' -f2-)
    if [ -z "$GB" ]; then echo "skip $(basename "$RUNDIR"): no genome"; continue; fi

    EVAL="$RUNDIR/eval_$SPLIT"
    rm -rf "$EVAL"; mkdir -p "$EVAL"
    # per-file eval: evaluate_rnn uses the genome's stored normalize bounds, so
    # per-file == batch, and it sidesteps the multi-file header-read quirk.
    for f in "$DATA_DIR"/*_"$SPLIT".csv; do
        "$BIN" --genome_file "$GB" --testing_filenames "$f" --time_offset 1 \
            --output_directory "$EVAL" --std_message_level ERROR --file_message_level ERROR >/dev/null 2>&1
    done
    n_pred=$(ls "$EVAL"/*_"$SPLIT"_predictions.csv 2>/dev/null | wc -l | tr -d ' ')
    echo "  $(basename "$RUNDIR"): genome=$(basename "$GB")  evaluated $n_pred stocks"
    n_runs=$((n_runs + 1))
done

if [ "$n_runs" -eq 0 ]; then echo "ERROR: no finished runs evaluated" >&2; exit 1; fi

echo ""
python3 scripts/stock_run/eval_ensemble_ic.py \
    --run-root "$OUT_ROOT" --split "$SPLIT" --target-col "$TARGET" \
    --emit-dir "$OUT_ROOT/ensemble_$SPLIT"

if [ "$TARGET" != "RET" ]; then
    echo ""
    echo "NOTE: target=$TARGET is cross-sectionally z-scored -- the IC above is valid"
    echo "      (rank-preserving per date), but expected_$TARGET is NOT a raw return, so"
    echo "      trade_portfolio.py P&L would need raw RET joined from the source cohort."
fi

echo ""
echo "To trade the ensemble (test only):"
echo "  python3 scripts/stock_run/trade_portfolio.py \\"
echo "    --pred-dir $OUT_ROOT/ensemble_$SPLIT --data-dir $DATA_DIR \\"
echo "    --align suffix --expect-n 50 --long 10 --short 10"
