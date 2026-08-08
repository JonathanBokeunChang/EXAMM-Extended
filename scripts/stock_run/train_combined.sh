#!/bin/sh
# Train EXAMM on the COMBINED stock dataset (300 inputs -> 50 outputs).
#
# Usage: sh scripts/stock_run/train_combined.sh [MAX_GENOMES]
#   e.g. sh scripts/stock_run/train_combined.sh 20      # ~10-20 min timing probe
#        sh scripts/stock_run/train_combined.sh 500     # overnight pilot
#        sh scripts/stock_run/train_combined.sh 10000   # paper budget (cluster-scale!)
#
# This is the paper's "combined" experiment (Table 5): one model forecasting
# all 50 stock returns simultaneously from all 300 predictors, exploiting
# inter-stock correlations (research question 2).
#
# COST WARNING: the minimal seed genome alone is 300x50 = 15,000 edges.
# One combined genome costs ~500-1300x a single-stock genome to train.
# Paper: 10k genomes took 3-4 days on 8 Xeon cores. Locally, ~4000 genomes
# is roughly 1-2.5 days; run a 20-genome probe first to measure.
#
# Input/output parameter names are derived from the CSV header at runtime:
#   inputs  = every column except `date` (300 of them, named TICKER_PREDICTOR)
#   outputs = the 50 columns ending in _RET
#
# Hyperparameter provenance:
#   - islands 10 x size 10, bp 20, lr 0.001, avg_std_dev normalization,
#     repopulation (500 / EraseWorst / exterminate 1 / bestGenome /
#     repeat_extinction), num_mutations 2, time_offset 1, node types:
#     professor's cluster script (2023_full runs).
#   - The paper text says 10 BP epochs; prof's actual script uses 20 -> 20.
#   - MAX_GENOMES: paper used 10000 for the combined dataset. Default here
#     is 500 (local pilot scale).

MAX=${1:-500}

cd build

DATA=../datasets/701515_split
TRAIN_FILE=$DATA/combined_predictors_train.csv
VAL_FILE=$DATA/combined_predictors_val.csv

if [ ! -f "$TRAIN_FILE" ] || [ ! -f "$VAL_FILE" ]; then
    echo "ERROR: combined dataset not found at $DATA"
    exit 1
fi

# ---- derive input/output parameter names from the CSV header ----
# header: date,NDSN_BA_SPREAD,...  (301 columns; tr -d '\r' guards against CRLF)
HEADER=$(head -1 "$TRAIN_FILE" | tr -d '\r')

# inputs: every column except `date`
INPUTS=$(echo "$HEADER" | tr ',' '\n' | grep -v '^date$' | tr '\n' ' ')

# outputs: exactly the *_RET columns (suffix match -- must not catch
# TICKER_sprtrn or tickers containing similar letters)
OUTPUTS=$(echo "$HEADER" | tr ',' '\n' | grep '_RET$' | tr '\n' ' ')

N_IN=$(echo "$INPUTS" | wc -w | tr -d ' ')
N_OUT=$(echo "$OUTPUTS" | wc -w | tr -d ' ')
echo "derived $N_IN input parameters and $N_OUT output parameters from header"
if [ "$N_IN" -ne 300 ] || [ "$N_OUT" -ne 50 ]; then
    echo "ERROR: expected 300 inputs and 50 outputs -- header format changed?"
    exit 1
fi

exp_name="../test_output/combined"
# fresh output dir per run -- EXAMM appends into existing dirs
rm -rf $exp_name
mkdir -p $exp_name

echo "Training EXAMM on combined dataset (max_genomes=${MAX}), results in: ${exp_name}"
echo "###-------------------###"

./multithreaded/examm_mt --number_threads 2 \
--training_filenames $TRAIN_FILE \
--validation_filenames $VAL_FILE \
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
//repop 200 genomes
///10k
///standard normal
///10x each experiment