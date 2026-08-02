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

# Score in the regime the genome was TRAINED in. A run with seq_len=N only ever saw N-step
# sequences from a reset hidden state; evaluating it as one unbroken pass over the whole test
# year lets state accumulate far beyond anything it saw, and confounds "this hyperparameter is
# bad" with "we scored it out of regime". Read from the run's own .config so no caller has to
# remember, and so a sliced arm cannot be scored unsliced by accident. seq_len=0 (every arm to
# date) yields no flag at all, which is byte-identical to the previous behaviour.
# SEQ_EVAL_OVERRIDE forces a value, for deliberately measuring the size of that confound.
SEQ_LEN=$(sed -n 's/^seq_len=//p' "$CFG"); SEQ_LEN=${SEQ_EVAL_OVERRIDE:-${SEQ_LEN:-0}}
SEQ_ARGS=""
if [ "$SEQ_LEN" -gt 0 ] 2>/dev/null; then SEQ_ARGS="--test_sequence_length $SEQ_LEN"; fi
N_EXPECT=$(ls "$DATA_DIR"/*_"$SPLIT".csv 2>/dev/null | wc -l | tr -d ' ')
if [ "$N_EXPECT" -eq 0 ]; then
    echo "ERROR: no *_$SPLIT.csv under '$DATA_DIR'" >&2; exit 1
fi
echo "OUT_ROOT=$OUT_ROOT  SPLIT=$SPLIT  DATA_DIR=$DATA_DIR  TARGET=$TARGET  seq_len=$SEQ_LEN  stocks=$N_EXPECT"

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
        "$BIN" --genome_file "$GB" --testing_filenames "$f" --time_offset 1 $SEQ_ARGS \
            --output_directory "$EVAL" --std_message_level ERROR --file_message_level ERROR >/dev/null 2>&1
    done
    n_pred=$(ls "$EVAL"/*_"$SPLIT"_predictions.csv 2>/dev/null | wc -l | tr -d ' ')
    echo "  $(basename "$RUNDIR"): genome=$(basename "$GB")  evaluated $n_pred stocks"
    # HARD FAIL on a short run. evaluate_rnn is invoked per stock with output suppressed, so
    # a genome that aborts on every stock produced "evaluated 0 stocks", this loop carried on,
    # and eval_ensemble_ic.py then printed "ensembling 10 run(s)" and a mean IC computed from
    # whichever runs happened to survive. That is how hp_k_seq20_cohort_2021_aligned reported
    # IC +0.000071 off a single working genome out of ten -- a number indistinguishable from a
    # real result. An incomplete run must stop the pipeline, not be quietly averaged.
    if [ "$n_pred" -ne "$N_EXPECT" ]; then
        echo "ERROR: $(basename "$RUNDIR") produced $n_pred/$N_EXPECT predictions." >&2
        echo "       Re-run that one genome WITHOUT output suppression to see why:" >&2
        echo "         $BIN --genome_file $GB \\" >&2
        echo "           --testing_filenames $(ls "$DATA_DIR"/*_"$SPLIT".csv | head -1) \\" >&2
        echo "           --time_offset 1 $SEQ_ARGS --output_directory /tmp/dbg \\" >&2
        echo "           --std_message_level INFO --file_message_level NONE" >&2
        exit 1
    fi
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
