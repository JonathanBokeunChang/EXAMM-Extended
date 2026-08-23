#!/bin/bash
# EXA-GP-MIN vs EXAMM at a matched 500-genome budget across the twelve mid_highmid_price
# cells -- the other twelve of the paper's fourteen.
#
# WHY THIS EXISTS, AND WHY IT OUTRANKS A DEEPER BUDGET SWEEP. The two Dataset A cells at a
# matched 500-genome budget disagree:
#     cohort_2021 -> 2023   EXA-GP-MIN +0.0088 / +0.0360 / +0.0423  (h=1/3/5)
#                           EXAMM      -0.0137 / +0.0005 / -0.0012
#     cohort_2020 -> 2022   EXA-GP-MIN -0.0066 / +0.0057 / +0.0128
#                           EXAMM      -0.0007 / +0.0073 / +0.0155
# One cell strongly favours EXA-GP-MIN, the other mildly favours EXAMM. With single-run seed
# sd of 0.007-0.011, two cells cannot separate the methods however many seeds each carries --
# averaging ten seeds shrinks the error on a CELL, not the sampling error across cells. The
# fix is more cells, not more seeds or more genomes, and twelve of them already exist.
#
# Fourteen paired cells support a paired test over per-cell differences, which is the form
# the claim should take: not "EXA-GP-MIN scored X here" but "EXA-GP-MIN beats EXAMM in k of
# 14 cells". score_forecasting.reps_for() already documents that per-cell values swing 26-36
# points between trees of an identical config while the MEAN barely moves -- so the mean over
# cells is the estimand that survives, and a two-cell mean is not one.
#
# MATCHED BUDGET, SAME MACHINE, SAME BINARY, SAME SEEDS. Every cell runs both arms at 500
# genomes. The 10,000-genome parity question is separate and belongs on Anvil; nothing here
# tries to answer it.
#
# Usage:  nohup bash scripts/stock_run/run_exagp_panel_sweep.sh &
#         tail -f test_output/exagp_panel_sweep.log
set -u
cd "$(dirname "$0")/../.."
LOG=${LOG:-test_output/exagp_panel_sweep.log}
SEEDS=${SEEDS:-10}
G=${G:-500}
export THREADS=${THREADS:-4}
# Reversible so a second worker can walk the matrix from the other end; run_exagp_local.sh
# locks each run directory, so the two divide the work instead of colliding.
PANELS=${PANELS:-"set1 set2 set3 set4"}
COHORTS=${COHORTS:-"cohort_2020_aligned cohort_2021_aligned cohort_2022_aligned"}
ARMS=${ARMS:-"exagpmin examm"}

{
  for P in $PANELS; do
      for COHORT in $COHORTS; do
          for ARM in $ARMS; do
              echo "=== PANEL $P $COHORT $ARM g$G | started $(date '+%Y-%m-%d %H:%M:%S') ==="
              PANEL="$P" REPLICATE_TAG="g$G" bash scripts/stock_run/run_exagp_local.sh \
                  "$ARM" "$COHORT" "$SEEDS" "$G"
              echo "=== PANEL $P $COHORT $ARM g$G | ended $(date '+%Y-%m-%d %H:%M:%S') rc=$? ==="
          done
      done
  done
  echo "=== PANEL SWEEP COMPLETE $(date '+%Y-%m-%d %H:%M:%S') ==="
} >> "$LOG" 2>&1
