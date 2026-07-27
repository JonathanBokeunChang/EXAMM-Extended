#!/bin/bash
# Pooled EXAMM on the sequence-form CSI300 dataset.
#
# The baseline to clear: GRU 2x64 scored +0.0571 (HAC t 12.35) on this exact data, exact eval
# index, exact target, at 38,849 parameters. That is the number, not published 0.0584 -- the
# sequence encoding differs from Alpha360's flat form, so only the internal comparison is
# controlled. (For context the encoding costs almost nothing: our Alpha360 GRU was +0.0581.)
#
# Design notes:
#   POOLED  one shared-weight genome over all stocks. Per-stock would mean ~500 genomes on ~1,700
#           rows each, and would not be comparable to a single pooled GRU. Pooling is also required
#           for cross-sectional rank IC to be defined at all.
#   TARGET  LABEL_CSRANK, because the GRU trains on it. That target was worth +0.0277 (HAC t 3.97)
#           on Alpha158; training EXAMM on raw returns against a rank-trained baseline would hand
#           the baseline a large known advantage and make any efficiency claim indefensible.
#   DEPTH   --max_recurrent_depth 60 lets EXAMM evolve recurrent edges spanning the full 60-day
#           window. This is the multi-scale hypothesis and EXAMM's genuinely unexploited
#           capability: every RNN baseline on this leaderboard has ONE fixed recurrence depth,
#           while ALSTM's +0.005 over LSTM shows better temporal routing pays here.
#   BASH    filenames MUST go through a bash array -- zsh does not word-split unquoted $(...),
#           which silently passes one giant filename and wedges the run.
#
# Usage:
#   bash scripts/stock_run/run_examm_seq.sh smoke     # tiny budget, verifies the pipeline
#   bash scripts/stock_run/run_examm_seq.sh full      # real run, 3 seeds
set -u
cd /Users/jonathanchang/EXAMM-Extended

MODE="${1:-smoke}"
D=datasets/csi300_master_replica_invdata_seq
OUT_ROOT="${OUT_ROOT:-/tmp/csi300_seq_examm}"

# EXAMM pairs training/validation filenames INDEX-WISE and segfaults on a count mismatch
# (515 train vs 364 val files here). _train_universe.txt is the frozen intersection -- 306 stocks
# with BOTH splits -- and the GRU baseline is restricted to the SAME list via --train-universe,
# so neither model trains on more data than the other. Costs 30.6% of train rows; disclosed.
UNIV="$D/_train_universe.txt"
[ -f "$UNIV" ] || { echo "ERROR: $UNIV missing"; exit 1; }
TRAIN=(); VAL=()
while read -r s; do
  case "$s" in ''|\#*) continue;; esac
  TRAIN+=( "$D/${s}_train.csv" ); VAL+=( "$D/${s}_val.csv" )
done < "$UNIV"
echo "[data] ${#TRAIN[@]} train / ${#VAL[@]} val files (frozen shared universe)"
[ "${#TRAIN[@]}" -eq "${#VAL[@]}" ] || { echo "ERROR: train/val count mismatch"; exit 1; }

if [ "$MODE" = "smoke" ]; then
  MAXG=60; SEEDS="1"; ISLANDS=2; ISIZE=3; BPI=2
  echo "[mode] SMOKE -- $MAXG genomes, verifies train -> global_best_genome -> evaluate end to end"
else
  MAXG=${MAXG:-10000}; SEEDS="1 2 3"; ISLANDS=10; ISIZE=10; BPI=${BPI:-10}
  echo "[mode] FULL -- $MAXG genomes x $(echo $SEEDS | wc -w | tr -d ' ') seeds"
fi

for seed in $SEEDS; do
  OUT="$OUT_ROOT/run_$seed"; rm -rf "$OUT"; mkdir -p "$OUT"
  t0=$(date +%s)
  ./build/multithreaded/examm_mt --number_threads 8 \
    --training_filenames "${TRAIN[@]}" --validation_filenames "${VAL[@]}" \
    --time_offset 0 \
    --input_parameter_names RET OPEN_C HIGH_C LOW_C VWAP_C VOLR \
    --output_parameter_names LABEL_CSRANK \
    --number_islands $ISLANDS --island_size $ISIZE \
    --max_genomes $MAXG --bp_iterations $BPI \
    --num_mutations 1 --normalize avg_std_dev \
    --max_recurrent_depth 60 \
    --extinction_event_generation_number 500 --islands_to_exterminate 1 \
    --repopulation_method bestGenome \
    --seed $seed --output_directory "$OUT" \
    --std_message_level ERROR --file_message_level NONE > "$OUT/train.log" 2>&1
  rc=$?
  gb=$(ls "$OUT"/global_best_genome_*.bin 2>/dev/null | wc -l | tr -d ' ')
  echo "[seed $seed] rc=$rc  global_best=$gb  $(( $(date +%s)-t0 ))s"
  if [ "$rc" != "0" ] || [ "$gb" = "0" ]; then
    echo "  FAILED -- last lines of train.log:"; tail -5 "$OUT/train.log"
    exit 1
  fi
done
echo "EXAMM_SEQ_${MODE}_COMPLETE -> $OUT_ROOT"
