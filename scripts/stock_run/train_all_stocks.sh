#!/bin/sh
# Train EXAMM on every stock in the 70/15/15 split dataset, one after another.
#
# Usage: sh scripts/stock_run/train_all_stocks.sh [MAX_GENOMES]
#
# Derives the ticker list from the data directory (excluding the
# combined_predictors dataset, which is a separate 300-input experiment).

MAX=${1:-2000}
DATA=datasets/701515_split

TICKERS=$(ls $DATA | sed -E 's/_(train|val|test)\.csv//' | sort -u | grep -v combined_predictors)

echo "Training EXAMM on stocks: "$TICKERS
echo "max_genomes per stock: ${MAX}"
echo "###-------------------###"

for S in $TICKERS; do
    sh scripts/stock_run/train_stock.sh $S $MAX
done