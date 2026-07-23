#!/bin/sh
# Train EXAMM with the cross-sectional IC (Information Coefficient) ranking loss
# on the POOLED walk-forward cohort: one tiny shared 6->1 genome applied to all
# 50 stocks, optimizing the daily cross-sectional rank correlation between the
# model's predicted returns and realized returns.
#
# Usage: sh scripts/stock_run/train_ic.sh [MAX_GENOMES] [IC_MODE] [NUM_THREADS] [IC_VAR_LAMBDA]
#   e.g. sh scripts/stock_run/train_ic.sh 100 pearson 4       # smoke / overfit test
#        sh scripts/stock_run/train_ic.sh 2000 pearson 8      # local pilot
#        sh scripts/stock_run/train_ic.sh 200 pearson 4 0     # pure IC (collapses; for reference)
#   IC_MODE: pearson (default, closed-form differentiable IC) | spearman (soft-rank)
#   IC_VAR_LAMBDA: weight on the anti-collapse variance-floor term (default 1.0; 0 = pure IC).
#     Objective L = -IC + lambda * mean_j max(0, tau - std_j)^2. Pure IC is scale-invariant
#     and collapses to near-constant predictions; the variance floor (VICReg, LeCun ICLR
#     2022) forces per-date cross-sectional std up toward tau, forbidding the collapse.
#
# Requires the ALIGNED cohort (all stocks share one calendar so the cross-section
# is well defined). Build it first:
#   python3 scripts/stock_run/align_cohort.py \
#       --in datasets/walkforward/cohort_2021 \
#       --out datasets/walkforward/cohort_2021_aligned
#
# Split (Dr. Lyu): train <= 2021, validate 2022, trade 2023. Fitness = validation
# Spearman IC (stored internally as -IC so EXAMM's minimize-fitness stack maximizes
# it). The held-out *_test.csv (2023) is NOT touched here.

MAX=${1:-100}
IC_MODE=${2:-pearson}
THREADS=${3:-4}
IC_VAR_LAMBDA=${4:-1.0}
IC_FITNESS=${5:-ic}          # ic (mean daily IC) | icir (IC information ratio; rewards consistency)

cd build

DATA=../datasets/walkforward/cohort_2021_aligned
INPUTS="RET VOL_CHANGE BA_SPREAD ILLIQUIDITY sprtrn TURNOVER"
OUTPUTS="RET"

if [ ! -d "$DATA" ]; then
    echo "ERROR: aligned cohort not found at $DATA"
    echo "Build it: python3 scripts/stock_run/align_cohort.py --in datasets/walkforward/cohort_2021 --out datasets/walkforward/cohort_2021_aligned"
    exit 1
fi

# pool every stock's train / val file (space-separated lists for examm_mt)
TRAIN_FILES=$(ls $DATA/*_train.csv | tr '\n' ' ')
VAL_FILES=$(ls $DATA/*_val.csv | tr '\n' ' ')
N_STOCKS=$(ls $DATA/*_train.csv | wc -l | tr -d ' ')

exp_name="../test_output/ic_${IC_MODE}"
rm -rf $exp_name
mkdir -p $exp_name

echo "Training EXAMM (loss=ic, ic_mode=${IC_MODE}, ic_var_lambda=${IC_VAR_LAMBDA}, fitness=${IC_FITNESS}) on ${N_STOCKS} pooled stocks,"
echo "max_genomes=${MAX}, threads=${THREADS}, results in: ${exp_name}"
echo "###-------------------###"

./multithreaded/examm_mt --number_threads $THREADS \
--training_filenames $TRAIN_FILES \
--validation_filenames $VAL_FILES \
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
--loss ic \
--ic_mode $IC_MODE \
--ic_var_lambda $IC_VAR_LAMBDA \
--ic_fitness $IC_FITNESS \
--extinction_event_generation_number 500 \
--repeat_extinction \
--island_ranking_method EraseWorst \
--islands_to_exterminate 1 \
--repopulation_method bestGenome \
--output_directory $exp_name \
--possible_node_types simple UGRNN MGU GRU delta LSTM \
--std_message_level INFO \
--file_message_level INFO
