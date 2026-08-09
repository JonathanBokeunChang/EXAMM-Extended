#!/bin/sh
# Evaluate the best evolved genome for a stock on its held-out TEST set.
#
# Usage: sh scripts/stock_run/evaluate_stock.sh <TICKER>
#
# Picks the best global_best_genome_*.bin from test_output/stock_<TICKER>/
# by the best_validation_mse recorded in each .txt (NOT by filename order,
# which is unreliable), then runs evaluate_rnn on <TICKER>_test.csv.
# The reported MSE/MAE is the number to compare against the paper's
# Tables 2-3 (this data was never seen during evolution).

if [ -z "$1" ]; then
    echo "usage: sh scripts/stock_run/evaluate_stock.sh <TICKER>"
    exit 1
fi

S=$1
exp_name="test_output/stock_${S}"

if [ ! -d "$exp_name" ]; then
    echo "ERROR: no training output at ${exp_name} -- run train_stock.sh ${S} first"
    exit 1
fi

# rank saved best genomes by their recorded validation MSE, take the lowest
BEST_TXT=$(for f in $exp_name/global_best_genome_*.txt; do
    mse=$(grep -o 'best_validation_mse: [0-9.e+-]*' "$f" | awk '{print $2}')
    echo "$mse $f"
done | sort -n | head -1 | awk '{print $2}')

if [ -z "$BEST_TXT" ]; then
    echo "ERROR: no global_best_genome_*.txt found in ${exp_name}"
    exit 1
fi

BEST_BIN="${BEST_TXT%.txt}.bin"
echo "Best genome: ${BEST_BIN}"
grep 'best_validation_mse' "$BEST_TXT"

cd build

eval_dir="../test_output/stock_${S}_eval"
rm -rf $eval_dir
mkdir -p $eval_dir

./rnn_examples/evaluate_rnn \
--genome_file ../$BEST_BIN \
--testing_filenames ../datasets/701515_split/${S}_test.csv \
--time_offset 1 \
--output_directory $eval_dir \
--std_message_level INFO \
--file_message_level ERROR

# NOTE: the MSE/MAE printed above are in NORMALIZED space when the genome was
# trained with --normalize. The predictions CSV is denormalized by
# write_predictions() (rnn/rnn.cxx), so compute the paper-comparable
# raw-scale numbers from it:
echo ""
echo "### paper-comparable (denormalized) test metrics ###"
python3 - "$eval_dir/${S}_test_predictions.csv" <<'EOF'
import csv, sys
exp, pred = [], []
with open(sys.argv[1]) as f:
    r = csv.DictReader(f)
    ecol = [c for c in r.fieldnames if c.startswith('expected')][0]
    pcol = [c for c in r.fieldnames if c.startswith('predicted')][0]
    for row in r:
        exp.append(float(row[ecol])); pred.append(float(row[pcol]))
n = len(exp)
mae = sum(abs(e - p) for e, p in zip(exp, pred)) / n
mse = sum((e - p) ** 2 for e, p in zip(exp, pred)) / n
mae0 = sum(abs(e) for e in exp) / n
mse0 = sum(e * e for e in exp) / n
print(f"n={n}  test MAE: {mae:.5f}  test MSE: {mse:.3e}")
print(f"predict-zero baseline  MAE: {mae0:.5f}  MSE: {mse0:.3e}")
EOF
