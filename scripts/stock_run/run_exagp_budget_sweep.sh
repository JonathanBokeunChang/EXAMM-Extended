#!/bin/bash
# EXA-GP-MIN vs EXAMM at MATCHED, SMALL genome budgets on the two Dataset A cells.
#
# WHY A BUDGET AXIS EXISTS AT ALL. Measured on the 10,000-genome run_1: between genome 99
# and genome 4600 the best validation MSE improved 1.2% (0.00180872 -> 0.00178720) while the
# network grew from 5 enabled edges to 55. The search spends ~98% of its budget making the
# model an order of magnitude less readable for a rounding error on its own objective. The
# compact closed forms that make EXA-GP worth trying live at 500-2000 genomes, and they cost
# minutes rather than hours. So model size is reported as an axis, not left as an accident of
# how long the search happened to run.
#
# MSE IS THE SEARCH OBJECTIVE, NOT THE ANSWER. score_forecasting.py's own header records that
# MSE and IC orderings do not track each other on this panel. A flat MSE curve therefore does
# NOT establish that the extra genomes are useless -- only that the search stopped improving
# the thing it optimises. Whether IC keeps climbing is exactly what scoring each budget
# settles, and it is why the 10,000-genome arm keeps running rather than being cancelled.
#
# BOTH ARMS AT EVERY BUDGET. Comparing EXA-GP-MIN at 500 against EXAMM at 10,000 would
# confound method with search budget, which is the single easiest way to manufacture a result
# in either direction. Every budget here carries its own EXAMM control on the same machine,
# same binary, same seeds.
#
# CHEAPEST BUDGET FIRST, deliberately: the g500 block completes in about an hour and yields a
# complete four-cell comparison, so a partial sweep is still a finished result rather than an
# unusable fragment. g2000 then refines it over several more hours.
#
# THREADS DEFAULTS TO 4, NOT 8, because the 10,000-genome campaign is running concurrently and
# already draws ~5.7 of this box's 8 cores (4 performance + 4 efficiency). Oversubscribing by
# ~1.4x keeps both saturated; handing this sweep 8 threads would just make the two fight.
#
# Usage:  nohup bash scripts/stock_run/run_exagp_budget_sweep.sh &
#         tail -f test_output/exagp_budget_sweep.log
set -u
cd "$(dirname "$0")/../.."
LOG=${LOG:-test_output/exagp_budget_sweep.log}
SEEDS=${SEEDS:-10}
BUDGETS=${BUDGETS:-"500 2000"}
export THREADS=${THREADS:-4}

{
  for G in $BUDGETS; do
      for ARM in exagpmin examm; do
          for COHORT in cohort_2021_aligned cohort_2020_aligned; do
              echo "=== BUDGET $G $ARM $COHORT | started $(date '+%Y-%m-%d %H:%M:%S') ==="
              # REPLICATE_TAG is what keeps a budget arm in its own tree. Without it every
              # budget would land in test_output/<arm>_<cohort>, and the SECOND budget would
              # find the first's .done markers, skip all ten runs and exit 0 -- reporting
              # success while producing nothing.
              REPLICATE_TAG="g$G" bash scripts/stock_run/run_exagp_local.sh \
                  "$ARM" "$COHORT" "$SEEDS" "$G"
              echo "=== BUDGET $G $ARM $COHORT | ended $(date '+%Y-%m-%d %H:%M:%S') rc=$? ==="
          done
      done
      echo "### BUDGET $G COMPLETE $(date '+%Y-%m-%d %H:%M:%S') ###"
  done
  echo "=== SWEEP COMPLETE $(date '+%Y-%m-%d %H:%M:%S') ==="
} >> "$LOG" 2>&1
