#!/bin/bash
# Leave-one-REGION-out on Barkjohn's own 49-sensor panel, with per-region ensembles.
#
# REGION, NOT SENSOR. Holding out one sensor leaves 44 others, many from the same
# metropolitan area and season, so a sensor-level split measures interpolation. The
# deployment question is whether a corrector fitted in some regions works in a region it
# has never seen, which is also the failure we already measured on the LTPP panel: the
# held-out wi site failed by 85-98% under every hyperparameter setting, with +3.7 ug/m3
# bias, because six training sites were not enough to learn a site-invariant mapping.
# Five regions gives five folds, each a genuine out-of-region test.
#
# THREE-WAY, as on the LTPP panel: TEST is the held-out region, VALID is a DIFFERENT
# region (the next one, wrapping) so EXAMM's fitness must already generalise across
# regions, and TRAIN is the remaining three.
#
# VARIANT is the ablation that matters. `minimal` gives the evolved model exactly the two
# inputs the deployed equation uses (cf_1, RH); `full` adds temperature, dewpoint and the
# six particle size bins, which a three-term linear form cannot represent. Beating the
# incumbent on `minimal` would mean our search fits better; beating it on `full` but not
# `minimal` would mean the win comes from information the equation form cannot hold. Those
# are different claims and only the data can separate them.
#
# Usage: VARIANT=full ARMS="examm exagpmin" REPEATS=5 MAXG=500 bash scripts/purpleair/bk_loro.sh
set -u
cd "$(dirname "$0")/../.."

VARIANT=${VARIANT:-full}
REPEATS=${REPEATS:-5}
MAXG=${MAXG:-500}
THREADS=${THREADS:-2}
ARMS=${ARMS:-"examm exagpmin"}
BP_ITER=${BP_ITER:-10}
TAG=${TAG:-bk_$VARIANT}
DATA="datasets/purpleair_bk/$VARIANT"
LOG=test_output/purpleair_${TAG}.log

[ -d "$DATA" ] || { echo "ERROR: $DATA missing -- run build_barkjohn_dataset.py" >&2; exit 1; }

if [ "$VARIANT" = "full" ]; then
    INPUTS="pa_cf1 pa_rh pa_temp pa_dew b03 b05 b1 b25 b5 b10"
else
    INPUTS="pa_cf1 pa_rh"
fi

REGIONS=($(ls "$DATA"))
N=${#REGIONS[@]}

{
echo "### BK LORO | variant $VARIANT | genomes $MAXG | repeats $REPEATS | bp $BP_ITER | $(date '+%H:%M:%S')"
for ARM in $ARMS; do
    case "$ARM" in
        examm)    ARM_ARGS=(--possible_node_types simple UGRNN MGU GRU delta LSTM) ;;
        exagpmin) ARM_ARGS=(--weight_initialize gp
                            --possible_node_types sigmoid_gp tanh_gp sum_gp multiply_gp inverse_gp sin_gp cos_gp) ;;
        *) echo "unknown arm $ARM" >&2; exit 1 ;;
    esac
    for i in $(seq 0 $((N - 1))); do
        TEST_R=${REGIONS[$i]}
        VAL_R=${REGIONS[$(( (i + 1) % N ))]}
        TRAIN=(); VALID=(); TESTF=()
        for j in $(seq 0 $((N - 1))); do
            R=${REGIONS[$j]}
            for f in "$DATA/$R"/*.csv; do
                if [ "$R" = "$TEST_R" ]; then TESTF+=("$f")
                elif [ "$R" = "$VAL_R" ]; then VALID+=("$f")
                else TRAIN+=("$f"); fi
            done
        done
        for R in $(seq 1 "$REPEATS"); do
            OUT="test_output/purpleair_${TAG}_${ARM}_r${R}_holdout_${TEST_R}"
            rm -rf "$OUT"; mkdir -p "$OUT"
            printf 'variant=%s\ntest_region=%s\nval_region=%s\nn_train=%s\nn_valid=%s\nn_test=%s\n' \
                "$VARIANT" "$TEST_R" "$VAL_R" "${#TRAIN[@]}" "${#VALID[@]}" "${#TESTF[@]}" > "$OUT/.split"
            build/multithreaded/examm_mt --number_threads "$THREADS" \
                --training_filenames "${TRAIN[@]}" --validation_filenames "${VALID[@]}" \
                --time_offset 0 \
                --input_parameter_names $INPUTS --output_parameter_names ref_pm25 \
                --normalize min_max --number_islands 4 --island_size 5 \
                --max_genomes "$MAXG" --bp_iterations "$BP_ITER" --num_mutations 2 \
                ${ARM_ARGS[@]+"${ARM_ARGS[@]}"} \
                --output_directory "$OUT" --save_genome_option none \
                --std_message_level ERROR --file_message_level NONE > "$OUT/train.log" 2>&1
            GB=$(ls "$OUT"/global_best_genome_*.bin 2>/dev/null \
                 | sed 's/.*_\([0-9]*\)\.bin/\1 &/' | sort -rn | head -1 | cut -d' ' -f2-)
            [ -n "$GB" ] || { echo "  FAILED $ARM/$TEST_R/r$R (no genome)"; continue; }
            mkdir -p "$OUT/eval_test"
            for f in "${TESTF[@]}"; do
                build/rnn_examples/evaluate_rnn --genome_file "$GB" --testing_filenames "$f" \
                    --time_offset 0 --output_directory "$OUT/eval_test" \
                    --std_message_level ERROR --file_message_level ERROR >/dev/null 2>&1
            done
        done
        echo "=== $ARM / $VARIANT / held-out $TEST_R (val $VAL_R) ==="
        python3 scripts/purpleair/score_bk.py --tag "$TAG" --arm "$ARM" \
            --region "$TEST_R" --variant "$VARIANT" --repeats "$REPEATS"
    done
done
echo "### BK LORO COMPLETE $(date '+%Y-%m-%d %H:%M:%S')"
} >> "$LOG" 2>&1
