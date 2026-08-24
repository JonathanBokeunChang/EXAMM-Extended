#!/bin/bash
# Evolve a PurpleAir PM2.5 correction and compare it to the deployed EPA equation.
#
# Task framing: this is a CORRECTION, not a forecast -- we map the PurpleAir
# sensor's own readings at time t onto the collocated FEM/FRM reference at the
# SAME time t. Hence --time_offset 0.
#
# The reference (ref_pm25) is the OUTPUT only. It must never appear in
# --input_parameter_names: at deployment there is no reference monitor next to
# the sensor, so feeding it back in measures an oracle, not a correction.
#
# Validation is LEAVE-ONE-SITE-OUT. The incumbent is a single US-wide equation,
# so the only fair comparison is generalisation to a site never trained on.
#
# Usage: bash purpleair_run.sh <HELD_OUT_SITE> [MAX_GENOMES] [THREADS] [TAG]
#   e.g. bash purpleair_run.sh az 2000 8 exagp
#
# NOTE: the machine may be loaded. Default budget here is a SMOKE TEST.

set -euo pipefail

SITE=${1:-az}
MAX_GENOMES=${2:-200}
THREADS=${3:-2}
TAG=${4:-smoke}

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DATA="$REPO_ROOT/datasets/purpleair"
OUT="$REPO_ROOT/test_output/purpleair_${TAG}_holdout_${SITE}"

# ARM picks the operator library, exactly as scripts/stock_run/run_exagp_local.sh does.
# The whole point of the comparison is EXA-GP's readable equation against EXAMM's memory
# cells on a task whose incumbent is itself an equation, so both arms must be reachable
# from one runner at a matched budget.
# Hyperparameters were never tuned: they are the defaults of the original smoke-test
# wiring script. bp_iterations 2 in particular means every candidate genome gets two
# backprop epochs before being judged, against 10 in the stock campaign, so the search
# may be selecting among underfit models rather than among architectures. Exposed here
# so the sensitivity can be measured instead of assumed.
BP_ITER=${BP_ITER:-2}
ISLANDS=${ISLANDS:-4}
ISLAND_SIZE=${ISLAND_SIZE:-5}
NORMALIZE=${NORMALIZE:-min_max}

ARM=${ARM:-examm}
case "$ARM" in
    examm)    ARM_ARGS=(--possible_node_types simple UGRNN MGU GRU delta LSTM) ;;
    exagp)    ARM_ARGS=(--possible_node_types sigmoid tanh sum multiply inverse sin cos) ;;
    exagpmin) ARM_ARGS=(--weight_initialize gp
                        --possible_node_types sigmoid_gp tanh_gp sum_gp multiply_gp inverse_gp sin_gp cos_gp) ;;
    *) echo "ERROR: unknown ARM '$ARM' (want examm, exagp or exagpmin)" >&2; exit 1 ;;
esac

# PurpleAir-only inputs. Both channels, both raw scalings, onboard met.
INPUT_PARAMETERS="pa_cf1_a pa_cf1_b pa_cf1 pa_atm pa_rh pa_temp"
OUTPUT_PARAMETERS="ref_pm25"

# THREE-WAY SPLIT, NOT TWO. The earlier version passed the held-out site as
# --validation_filenames, which is the split EXAMM's fitness SELECTS on -- so the
# "held-out" site was chosen against and its score was optimistic. That is exactly the
# mechanism that makes a first result look better than it is, and the smoke test's -13%
# against the incumbent carries the flaw.
#
#   TEST  = $SITE      never seen during evolution or selection; scored afterwards by
#                      rnn_examples/evaluate_rnn using the genome's stored normalisation.
#   VALID = $VAL_SITE  EXAMM's fitness signal -- a DIFFERENT site, so selection still has
#                      to generalise across sites rather than merely across time.
#   TRAIN = the rest
#
# VAL_SITE defaults to the next site alphabetically after TEST, wrapping, so the pairing
# is deterministic and reproducible rather than a per-run judgement call.
shopt -s nullglob
SITES=()
for f in "$DATA"/*_seg*.csv; do
    b=$(basename "$f"); SITES+=("${b%%_*}")
done
SITES=($(printf '%s\n' "${SITES[@]}" | sort -u))
if [ ${#SITES[@]} -lt 3 ]; then
    echo "ERROR: need >=3 sites for a three-way split, found ${#SITES[@]}" >&2; exit 1
fi

VAL_SITE=${VAL_SITE:-}
if [ -z "$VAL_SITE" ]; then
    for i in "${!SITES[@]}"; do
        if [ "${SITES[$i]}" = "$SITE" ]; then
            VAL_SITE="${SITES[$(( (i + 1) % ${#SITES[@]} ))]}"
        fi
    done
fi
if [ -z "$VAL_SITE" ]; then
    echo "ERROR: test site '$SITE' not found in $DATA" >&2; exit 1
fi
if [ "$VAL_SITE" = "$SITE" ]; then
    echo "ERROR: validation site equals test site ('$SITE')" >&2; exit 1
fi

TRAIN=(); VALID=(); TEST=()
for f in "$DATA"/*_seg*.csv; do
    b=$(basename "$f")
    case "$b" in
        ${SITE}_*)     TEST+=("$f") ;;
        ${VAL_SITE}_*) VALID+=("$f") ;;
        *)             TRAIN+=("$f") ;;
    esac
done

if [ ${#TEST[@]} -eq 0 ] || [ ${#VALID[@]} -eq 0 ] || [ ${#TRAIN[@]} -eq 0 ]; then
    echo "ERROR: empty split (train ${#TRAIN[@]} / valid ${#VALID[@]} / test ${#TEST[@]})" >&2
    exit 1
fi

# Record the split so a score can never be attributed to the wrong partition later.
mkdir -p "$OUT"
{
  printf 'test_site=%s\nval_site=%s\n' "$SITE" "$VAL_SITE"
  printf 'n_train=%s\nn_valid=%s\nn_test=%s\n' "${#TRAIN[@]}" "${#VALID[@]}" "${#TEST[@]}"
} > "$OUT/.split"

mkdir -p "$OUT"
echo "test site     : $SITE  (never seen; scored after evolution)"
echo "valid site    : $VAL_SITE  (EXAMM fitness signal)"
echo "train files   : ${#TRAIN[@]}"
echo "valid files   : ${#VALID[@]}"
echo "test files    : ${#TEST[@]}"
echo "max_genomes   : $MAX_GENOMES   threads: $THREADS"
echo "output        : $OUT"
echo "###-------------------###"

"$REPO_ROOT/build/multithreaded/examm_mt" \
    --number_threads "$THREADS" \
    --training_filenames "${TRAIN[@]}" \
    --validation_filenames "${VALID[@]}" \
    --time_offset 0 \
    --input_parameter_names $INPUT_PARAMETERS \
    --output_parameter_names $OUTPUT_PARAMETERS \
    --normalize "$NORMALIZE" \
    --number_islands "$ISLANDS" \
    --island_size "$ISLAND_SIZE" \
    --max_genomes "$MAX_GENOMES" \
    --bp_iterations "$BP_ITER" \
    --num_mutations 2 \
    --output_directory "$OUT" \
    ${ARM_ARGS[@]+"${ARM_ARGS[@]}"} \
    --std_message_level INFO \
    --file_message_level INFO
RC=$?
if [ $RC -ne 0 ]; then echo "ERROR: evolution failed rc=$RC" >&2; exit $RC; fi

# ---- score the TEST site -------------------------------------------------------------
# Until this ran, the three-way split only EXCLUDED the test site; it never produced a
# number from it. evaluate_rnn replays the winning genome using the normalisation bounds
# stored in the genome itself, so the test site is never used to set any statistic.
GB=$(ls "$OUT"/global_best_genome_*.bin 2>/dev/null \
     | sed 's/.*_\([0-9]*\)\.bin/\1 &/' | sort -rn | head -1 | cut -d' ' -f2-)
if [ -z "$GB" ]; then echo "ERROR: no global_best_genome in $OUT" >&2; exit 1; fi
echo "### scoring $(basename "$GB") on held-out site $SITE"

EVAL="$OUT/eval_test"; rm -rf "$EVAL"; mkdir -p "$EVAL"
for f in "${TEST[@]}"; do
    "$REPO_ROOT/build/rnn_examples/evaluate_rnn" \
        --genome_file "$GB" --testing_filenames "$f" --time_offset 0 \
        --output_directory "$EVAL" \
        --std_message_level ERROR --file_message_level ERROR >/dev/null 2>&1
done
N_PRED=$(ls "$EVAL"/*_predictions.csv 2>/dev/null | wc -l | tr -d ' ')
if [ "$N_PRED" -ne "${#TEST[@]}" ]; then
    echo "ERROR: $N_PRED/${#TEST[@]} test segments scored -- refusing to report a partial number" >&2
    exit 1
fi

python3 "$REPO_ROOT/scripts/purpleair/score_purpleair.py" \
    --eval-dir "$EVAL" --data-dir "$DATA" --site "$SITE" | tee "$OUT/test_score.txt"
