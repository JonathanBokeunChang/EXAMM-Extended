#!/bin/sh
# Evaluate the best evolved COMBINED genome (300 inputs -> 50 outputs) on the
# held-out combined test set, and report paper-comparable metrics.
#
# Usage: sh scripts/stock_run/evaluate_combined.sh [TRAIN_OUTPUT_DIR]
#   default TRAIN_OUTPUT_DIR: test_output/combined   (train_combined.sh's output)
#   e.g. sh scripts/stock_run/evaluate_combined.sh                       # usual case
#        sh scripts/stock_run/evaluate_combined.sh test_output/combined_run2
#
# What it does:
#   1. Picks the best global_best_genome_*.bin by the best_validation_mse
#      recorded in each .txt (NOT by filename number, which is unreliable).
#   2. Runs evaluate_rnn on combined_predictors_test.csv -- the first and only
#      time test data is touched. The genome's .bin carries its normalization
#      stats; write_predictions() denormalizes the output CSV.
#      NOTE: the MSE/MAE evaluate_rnn prints are in NORMALIZED space when the
#      genome was trained with --normalize. Ignore them; use the summary below.
#   3. Computes the paper's Table 5 metric from the denormalized predictions:
#      per-stock mean MAE/MSE, SUMMED over the 50 stocks ("the results ...
#      present the sum of errors over 50 stocks", paper p.12), plus:
#        - the predict-zero baseline (no model; the no-skill floor),
#        - pooled directional accuracy (sign agreement, as in the lab's
#          Financial_toolbox get_prediction_acc.py),
#        - the paper's published Table 5 numbers for reference.
#      Also writes a per-stock breakdown CSV into the eval directory.

exp_name=${1:-test_output/combined}

if [ ! -d "$exp_name" ]; then
    echo "ERROR: no training output at ${exp_name} -- run train_combined.sh first"
    exit 1
fi

# rank saved best genomes by their recorded validation MSE, take the lowest
# (RNN_Genome writes these via to_string() -> always fixed notation, so sort -n is safe)
BEST_TXT=$(for f in $exp_name/global_best_genome_*.txt; do
    [ -f "$f" ] || continue
    mse=$(grep -o 'best_validation_mse: [0-9.e+-]*' "$f" | awk '{print $2}')
    echo "$mse $f"
done | sort -n | head -1 | awk '{print $2}')

if [ -z "$BEST_TXT" ]; then
    echo "ERROR: no global_best_genome_*.txt found in ${exp_name}"
    echo "       (a crashed run saves none -- rerun the training)"
    exit 1
fi

BEST_BIN="${BEST_TXT%.txt}.bin"
echo "Best genome: ${BEST_BIN}"
grep 'best_validation_mse' "$BEST_TXT"

cd build

eval_dir="../${exp_name}_eval"
rm -rf $eval_dir
mkdir -p $eval_dir

./rnn_examples/evaluate_rnn \
--genome_file ../$BEST_BIN \
--testing_filenames ../datasets/701515_split/combined_predictors_test.csv \
--time_offset 1 \
--output_directory $eval_dir \
--std_message_level INFO \
--file_message_level ERROR

echo ""
echo "### paper-comparable (denormalized) test metrics -- Table 5 format ###"
python3 - "$eval_dir/combined_predictors_test_predictions.csv" "$eval_dir/per_stock_metrics.csv" <<'EOF'
import sys
import pandas as pd

pred_file, out_file = sys.argv[1], sys.argv[2]
df = pd.read_csv(pred_file)
df.columns = [c.lstrip('#') for c in df.columns]  # header's first column starts with '#'

tickers = sorted({c[len('expected_'):-len('_RET')]
                  for c in df.columns
                  if c.startswith('expected_') and c.endswith('_RET')})
if len(tickers) != 50:
    sys.exit(f"ERROR: expected 50 stocks in predictions, found {len(tickers)}")

rows = []
mae_sum = mse_sum = mae0_sum = mse0_sum = 0.0
correct = total = 0
for t in tickers:
    e = df[f'expected_{t}_RET']
    p = df[f'predicted_{t}_RET']
    mae, mse = (e - p).abs().mean(), ((e - p) ** 2).mean()
    mae0, mse0 = e.abs().mean(), (e ** 2).mean()          # predict-zero baseline
    acc = ((e * p) > 0).mean()                            # sign agreement
    rows.append((t, mae, mse, mae0, mse0, acc))
    mae_sum += mae; mse_sum += mse
    mae0_sum += mae0; mse0_sum += mse0
    correct += ((e * p) > 0).sum(); total += len(e)

pd.DataFrame(rows, columns=['ticker', 'mae', 'mse', 'mae_zero', 'mse_zero',
                            'dir_acc']).to_csv(out_file, index=False)

print(f"test days: {len(df)}, stocks: {len(tickers)}")
print(f"directional accuracy (pooled): {correct/total:.4f}  (0.5 = coin flip)")
print()
print(f"{'':44s}{'MAE sum':>10s}{'MSE sum':>10s}")
print(f"{'THIS RUN':44s}{mae_sum:10.5f}{mse_sum:10.5f}")
print(f"{'predict-zero baseline (same test days)':44s}{mae0_sum:10.5f}{mse0_sum:10.5f}")
print()
print("reference -- paper Table 5 (published, 10k genomes):")
for name, mae, mse in [('EXAMM combined',            0.73902, 0.02058),
                       ('EXAMM combined fine-tuned', 0.70799, 0.01923),
                       ('Crossformer',               0.72581, 0.01983),
                       ('DeformTime',                0.66872, 0.01756),
                       ('TSMixer',                   0.72293, 0.02340),
                       ('PatchTST',                  8.36444, 3.19147)]:
    print(f"{name:44s}{mae:10.5f}{mse:10.5f}")
print()
print(f"per-stock breakdown written to: {out_file}")
EOF