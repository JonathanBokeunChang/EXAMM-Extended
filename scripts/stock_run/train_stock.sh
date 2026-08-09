#!/bin/sh
# Train EXAMM on one stock from the 70/15/15 split dataset (2025 Evostar stock paper).
#
# Usage: sh scripts/stock_run/train_stock.sh <TICKER> [MAX_GENOMES]
#   e.g. sh scripts/stock_run/train_stock.sh ATO 2000
#
# Trains on <TICKER>_train.csv with <TICKER>_val.csv as the validation set
# (EXAMM's fitness). The held-out <TICKER>_test.csv is NOT touched here --
# use evaluate_stock.sh for the one-time test evaluation.
#
# Settings match the professor's actual cluster script (2023_full runs):
# 10 islands x 10, 20 BP epochs, lr 0.001, z-score normalization, and
# repopulation: every 500 genomes, EraseWorst ranking, exterminate 1,
# bestGenome refill, repeat_extinction on. (Note: prof's script uses
# bp_iterations 20 even though the paper text says 10.)
# Prof's script runs 10 repeats per stock; MAX_GENOMES there is 10000.
# MAX_GENOMES defaults to 2000 for faster iteration.
# (Prof's --test_filenames == our --validation_filenames; flag was renamed.)

if [ -z "$1" ]; then
    echo "usage: sh scripts/stock_run/train_stock.sh <TICKER> [MAX_GENOMES]"
    exit 1
fi

S=$1
MAX=${2:-2000}

cd build

DATA=../datasets/701515_split
INPUTS="RET VOL_CHANGE BA_SPREAD ILLIQUIDITY sprtrn TURNOVER"
OUTPUTS="RET"

exp_name="../test_output/stock_${S}"
# fresh output dir per run -- EXAMM appends into existing dirs, mixing stale genomes
rm -rf $exp_name
mkdir -p $exp_name

echo "Training EXAMM on ${S} (max_genomes=${MAX}), results in: ${exp_name}"
echo "###-------------------###"

./multithreaded/examm_mt --number_threads 8 \
--training_filenames $DATA/${S}_train.csv \
--validation_filenames $DATA/${S}_val.csv \
--time_offset 1 \
--input_parameter_names $INPUTS \
--output_parameter_names $OUTPUTS \
--number_islands 10 \
--island_size 10 \
--max_genomes $MAX \
--bp_iterations 20 \
--num_mutations 2 \
--learning_rate 0.001 \
--normalize avg_std_dev \
--extinction_event_generation_number 500 \
--repeat_extinction \
--island_ranking_method EraseWorst \
--islands_to_exterminate 1 \
--repopulation_method bestGenome \
--output_directory $exp_name \
--possible_node_types simple UGRNN MGU GRU delta LSTM \
--std_message_level INFO \
--file_message_level INFO