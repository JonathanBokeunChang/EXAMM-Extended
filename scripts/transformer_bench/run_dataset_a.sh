#!/bin/bash
# Run one or more model arms on DATASET A (the original 50-stock universe), sharded across GPUs.
#
#   bash scripts/transformer_bench/run_dataset_a.sh DLinearOfficial
#   NSHARDS=9 bash scripts/transformer_bench/run_dataset_a.sh DLinearOfficial ITransformerOfficial
#
# WHY NOT run_portfolio_campaign.sh. That script sets UNIVERSE=mid_highmid unconditionally (line 41)
# and threads SET through out_dir/mine/preflight; Dataset A has no SET at all, and anvil_transformer.sb
# actively REFUSES a SET when UNIVERSE=original. Retrofitting a universe axis into a 240-line campaign
# driver to run 20 jobs is more risk than writing the 20-job loop, so this exists instead.
#
# DATASET A IS 2 COHORTS x 10 SEEDS = 20 RUNS PER MODEL, against the portfolio campaign's 120. It is
# the panel behind tab:results-2022-2023, where DLinear is still dashed.
#
# TIMING IS RECORDED AUTOMATICALLY. anvil_transformer.sb writes timing.json (epochs, epoch_sec,
# train_sec, wall_sec, batch_size, num_workers, gpu, cuda_visible_devices, host) before it writes
# .done, and refuses to mark a run complete if no epoch timings were parsed. That is why the older
# Dataset A runs have no timings -- iTransformer's kept only predictions.csv -- and why anything
# re-run through this path will.
#
# SEED-OUTERMOST enumeration, then round-robin to shards: every shard gets a mix of both cohorts, and
# an interrupted campaign has completed whole seeds rather than one cohort twice over.
set -uo pipefail
[ $# -ge 1 ] || { echo "usage: run_dataset_a.sh MODEL [MODEL...]" >&2; exit 2; }
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO" || exit 1
NSHARDS=${NSHARDS:-9}
SEQ_LEN=${SEQ_LEN:-96}
PROFILE=${PROFILE:-author}
SEEDS=${SEEDS:-"1 2 3 4 5 6 7 8 9 10"}
COHORTS=${COHORTS:-"cohort_2020_aligned cohort_2021_aligned"}
NGPU=${NGPU:-$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ')}
[ "${NGPU:-0}" -ge 1 ] || NGPU=1
LOG=${LOG:-/workspace/dataset_a.log}
say(){ echo "[$(date -u '+%H:%M:%SZ')] $*" | tee -a "$LOG"; }

for c in $COHORTS; do
  d="datasets/walkforward/$c"
  n=$(ls "$d"/*_train.csv 2>/dev/null | wc -l | tr -d ' ')
  [ "$n" -ge 50 ] || { echo "ERROR: $d has $n train files (need >=50)" >&2; exit 1; }
done

shard_worker() {   # $1 = shard index, $2.. = models
  local sh=$1; shift
  local idx=0
  for SEED in $SEEDS; do
    for M in "$@"; do
      for COHORT in $COHORTS; do
        if [ $(( idx % NSHARDS )) -eq "$sh" ]; then
          OUT="$REPO/results/transformer_bench/$PROFILE/$COHORT/$M/L$SEQ_LEN/seed_$SEED"
          if [ ! -f "$OUT/.done" ]; then
            # SLURM_ARRAY_TASK_ID, NOT SEED. anvil_transformer.sb line 53 is
            # SEED=${SLURM_ARRAY_TASK_ID:-1} -- it ignores an env SEED entirely and falls back to 1
            # outside SLURM. Passing SEED= silently ran all 20 jobs as seed 1: two completed and
            # eighteen hit the .done check and reported "already complete -- skipping", which reads
            # like success. run_portfolio_campaign.sh sets SLURM_ARRAY_TASK_ID for this reason.
            SLURM_SUBMIT_DIR="$REPO" MODEL="$M" COHORT="$COHORT" SEQ_LEN="$SEQ_LEN" \
            PROFILE="$PROFILE" SLURM_ARRAY_TASK_ID="$SEED" NUM_WORKERS=4 \
            CUDA_VISIBLE_DEVICES=$(( sh % NGPU )) \
              bash scripts/transformer_bench/anvil_transformer.sb \
              >> "/workspace/dsA_${M}_shard${sh}.log" 2>&1 \
              || echo "[shard $sh] FAILED $M $COHORT seed $SEED" >> "$LOG"
          fi
        fi
        idx=$(( idx + 1 ))
      done
    done
  done
}

TOTAL=$(( $(echo $SEEDS | wc -w) * $# * $(echo $COHORTS | wc -w) ))
say "=== dataset A | arms: $* | $TOTAL runs | $NSHARDS shards over $NGPU GPU(s) ==="
for s in $(seq 0 $(( NSHARDS - 1 ))); do shard_worker "$s" "$@" & done
wait

for M in "$@"; do
  n=$(find results/transformer_bench/$PROFILE -path "*/$M/*" -name .done 2>/dev/null | wc -l)
  say "=== $M on dataset A: $n done ==="
done
say "=== DATASET A COMPLETE ==="
