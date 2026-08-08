#!/bin/bash
# Launch ONE model arm across 9 shards on a 3-GPU pod, then wait and report.
#
#   bash scripts/transformer_bench/run_arm.sh DLinearOfficial
#   bash scripts/transformer_bench/run_arm.sh LSTMBaseline GRUBaseline    # sequential arms
#
# WHY THIS EXISTS SEPARATELY FROM run_three.sh. That script hardcodes the PatchTST -> DeformTime ->
# DLinear order for the first pod. A second pod runs a different, shorter list, and editing a
# running orchestrator is not possible -- so the arm list is an argument here rather than a literal.
#
# SEQUENTIAL ARMS, NOT ONE COMBINED CAMPAIGN. run_portfolio_campaign.sh loops seed-outermost, so a
# single campaign over several models reaches seed 1 of every model before seed 2 of any -- good for
# balanced degradation ACROSS models, but no model is complete and analysable until near the end.
# Run as separate arms and the first one is scoreable as soon as it finishes.
#
# NUM_WORKERS=4 THROUGHOUT, matching dataset A and every completed portfolio campaign. Some early
# variant arms used 2, which changes no result but makes wall_sec non-comparable across arms.
set -uo pipefail
[ $# -ge 1 ] || { echo "usage: run_arm.sh MODEL [MODEL...]" >&2; exit 2; }
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG=${LOG:-/workspace/arms.log}
NSHARDS=${NSHARDS:-9}
NGPU=${NGPU:-$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ')}
[ "${NGPU:-0}" -ge 1 ] || { echo "ERROR: no GPUs visible" >&2; exit 1; }
cd "$REPO" || exit 1
say(){ echo "[$(date -u '+%H:%M:%SZ')] $*" | tee -a "$LOG"; }

say "=== run_arm.sh | arms: $* | $NSHARDS shards over $NGPU GPU(s) ==="
for M in "$@"; do
  before=$(find results/transformer_bench/mid_highmid -path "*/author/*/${M}/*" -name .done 2>/dev/null | wc -l)
  say "=== starting $M (already done: $before/120) ==="
  for s in $(seq 0 $(( NSHARDS - 1 ))); do
    MODELS="$M" NUM_WORKERS=4 CUDA_VISIBLE_DEVICES=$(( s % NGPU )) SHARD=$s NSHARDS=$NSHARDS \
      setsid bash scripts/transformer_bench/run_portfolio_campaign.sh \
        > "/workspace/${M}_shard${s}.log" 2>&1 &
  done
  wait
  n=$(find results/transformer_bench/mid_highmid -path "*/author/*/${M}/*" -name .done 2>/dev/null | wc -l)
  say "=== $M complete: $n/120 ==="
  [ "$n" -eq 120 ] || say "    WARNING: $M finished with $n/120 -- check /workspace/${M}_shard*.log"
done
say "=== ALL ARMS COMPLETE ==="
