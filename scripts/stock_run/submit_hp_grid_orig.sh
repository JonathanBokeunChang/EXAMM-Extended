#!/bin/bash
# Island-count x repopulation-frequency GRID, ORIGINAL universe only (cohort_2020_aligned,
# cohort_2021_aligned -- trading years 2022/2023, the same 2 cohorts as paper Table 1). No
# mid_highmid submission here; that venue is out of scope for this grid on request.
#
# Full factorial requested: number_islands in {10,20,40} x
#                            extinction_event_generation_number in {250,500,1000}.
# "island size" was disambiguated as number_islands (--number_islands), matching the existing
# k_isl20 arm, NOT --island_size (population per island, baseline 10, never swept by anything).
#
# 3 of 9 cells ALREADY EXIST -- do NOT resubmit:
#   (10,500) baseline  -- test_output/hp_baseline_cohort_202{0,1}_aligned
#                          (submitted via anvil_examm_hpsweep.sb ARM=baseline). NOISY REFERENCE:
#                          this re-run disagrees with the paper's own (10,500) campaign
#                          (test_output/mse_cohort_202{0,1}_aligned) by 0.0182 IC / 38pp of
#                          Algorithm-2 return on cohort_2021 alone -- see submit_hp_ablation.sh's
#                          header (added 2026-08-02). Both campaigns exist; neither is touched here.
#   (20,500) k_isl20    -- test_output/hp_k_isl20_cohort_202{0,1}_aligned
#   (10,250) k_ext250   -- test_output/hp_k_ext250_cohort_202{0,1}_aligned
#
# This script submits the other 6 cells: (10,1000) (20,250) (20,1000) (40,250) (40,500) (40,1000).
#
# SAME CAVEAT AS submit_hp_ablation.sh: validation fitness does not predict test IC on this venue
# (r=+0.065, n=6, measured 2026-07-21) -- there is no honest rule for picking a "best" cell out of
# this grid after the fact. This is descriptive coverage (does the knob move the mean, with error
# bars), not a tuned-config search. The orig universe is also underpowered on its own (MDE ~0.0132
# IC per cell from just 2 cohorts) -- read single cells as suggestive, not decisive; the ablation's
# actual generalization verdict came from the 12-cell mid_highmid venue, not this one.
#
#   sh scripts/stock_run/submit_hp_grid_orig.sh          # submit all 6 missing cells
#   DRY=1 sh scripts/stock_run/submit_hp_grid_orig.sh    # print, do not submit

set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ORIG="$REPO/scripts/stock_run/anvil_examm_hpsweep.sb"
DRY=${DRY:-0}

# tag : env assignments (ARM=custom is added by sub())
declare -a CELLS=(
  "k_isl10_ext1000  ISLANDS_OVERRIDE=10,EXTINCTION_OVERRIDE=1000"
  "k_isl20_ext250   ISLANDS_OVERRIDE=20,EXTINCTION_OVERRIDE=250"
  "k_isl20_ext1000  ISLANDS_OVERRIDE=20,EXTINCTION_OVERRIDE=1000"
  "k_isl40_ext250   ISLANDS_OVERRIDE=40,EXTINCTION_OVERRIDE=250"
  "k_isl40_ext500   ISLANDS_OVERRIDE=40,EXTINCTION_OVERRIDE=500"
  "k_isl40_ext1000  ISLANDS_OVERRIDE=40,EXTINCTION_OVERRIDE=1000"
)

sub() {  # extra-env
  if [ "$DRY" = "1" ]; then echo "  DRY sbatch --export=ALL,$1 --array=0-9 $(basename "$ORIG")"
  else sbatch --export=ALL,"$1" --array=0-9 "$ORIG" >/dev/null; fi
}

n=0
for spec in "${CELLS[@]}"; do
  read -r TAG ENVS <<< "$spec"
  echo "=== cell $TAG ($ENVS) ==="
  for C in cohort_2020_aligned cohort_2021_aligned; do
    sub "ARM=custom,RUN_TAG=$TAG,COHORT=$C,$ENVS"; n=$((n+1))
  done
done
echo
echo "submitted $n array jobs x 10 seeds = $((n*10)) runs"
[ "$DRY" = "1" ] && echo "(dry run -- nothing was actually submitted)"
