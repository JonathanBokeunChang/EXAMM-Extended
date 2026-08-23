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

# PurpleAir-only inputs. Both channels, both raw scalings, onboard met.
INPUT_PARAMETERS="pa_cf1_a pa_cf1_b pa_cf1 pa_atm pa_rh pa_temp"
OUTPUT_PARAMETERS="ref_pm25"

shopt -s nullglob
TRAIN=()
VALID=()
for f in "$DATA"/*_seg*.csv; do
    b=$(basename "$f")
    if [[ "$b" == ${SITE}_* ]]; then VALID+=("$f"); else TRAIN+=("$f"); fi
done

if [ ${#VALID[@]} -eq 0 ]; then
    echo "ERROR: no segments for held-out site '$SITE' in $DATA" >&2
    exit 1
fi

mkdir -p "$OUT"
echo "held-out site : $SITE"
echo "train files   : ${#TRAIN[@]}"
echo "valid files   : ${#VALID[@]}"
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
    --normalize min_max \
    --number_islands 4 \
    --island_size 5 \
    --max_genomes "$MAX_GENOMES" \
    --bp_iterations 2 \
    --num_mutations 2 \
    --output_directory "$OUT" \
    --possible_node_types simple UGRNN MGU GRU delta LSTM \
    --std_message_level INFO \
    --file_message_level INFO
