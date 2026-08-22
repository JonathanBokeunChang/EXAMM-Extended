#!/bin/bash
# Full EXA-GP vs EXAMM campaign on the pooled stock panel, sequentially, on one machine.
#
# Phase order is deliberate: the two EXA-GP-MIN cohorts come FIRST because they are the
# thing being asked about, so a partial campaign still answers the question. The local
# EXAMM control follows (it is the slowest arm, and the paper already has an Anvil EXAMM
# row to fall back on), and plain EXA-GP is last because it is the least informative --
# a middle point between the two arms that bracket it.
#
# One run at a time, 8 threads each: the box has 8 cores, so running phases concurrently
# would only trade wall-clock between them without improving total throughput.
#
# Usage:  nohup bash scripts/stock_run/run_exagp_campaign.sh &
#         tail -f test_output/exagp_campaign.log
set -u
cd "$(dirname "$0")/../.."
LOG=${LOG:-test_output/exagp_campaign.log}
SEEDS=${SEEDS:-10}
MAXG=${MAXG:-10000}

run_phase() {
    echo "=== PHASE $1 $2 | started $(date '+%Y-%m-%d %H:%M:%S') ==="
    bash scripts/stock_run/run_exagp_local.sh "$1" "$2" "$SEEDS" "$MAXG"
    echo "=== PHASE $1 $2 | ended $(date '+%Y-%m-%d %H:%M:%S') rc=$? ==="
}

{
  run_phase exagpmin cohort_2021_aligned
  run_phase exagpmin cohort_2020_aligned
  run_phase examm    cohort_2021_aligned
  run_phase examm    cohort_2020_aligned
  run_phase exagp    cohort_2021_aligned
  run_phase exagp    cohort_2020_aligned
  echo "=== CAMPAIGN COMPLETE $(date '+%Y-%m-%d %H:%M:%S') ==="
} >> "$LOG" 2>&1
