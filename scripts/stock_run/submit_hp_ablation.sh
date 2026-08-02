#!/bin/bash
# One-factor-at-a-time hyperparameter ABLATION across BOTH universes.
# Run on Anvil. bash only.
#
#   sh scripts/stock_run/submit_hp_ablation.sh                # all 4 arms, both venues
#   sh scripts/stock_run/submit_hp_ablation.sh k_bp20         # one arm only
#   DRY=1 sh scripts/stock_run/submit_hp_ablation.sh          # print, do not submit
#
# WHY "ABLATION" AND NOT "SEARCH": validation fitness does not predict test IC on
# this venue (r = +0.065, n = 6, measured 2026-07-21). There is therefore NO honest
# selection rule -- picking the best arm on TEST is selection bias (measured at
# +5-6 pp on trading elsewhere in this project), and picking on validation is
# picking noise. This experiment can measure whether a knob helps ON AVERAGE, with
# error bars. It cannot output "use these settings" as a tuned configuration.
#
# WHY BOTH UNIVERSES: the goal is settings that generalise, so a knob that helps on
# one universe and hurts on the other is worthless. Consistency across two
# independent universes IS the evidence; the power gain is a side benefit
# (14 cells -> minimum detectable effect ~0.0050 IC, vs ~0.0132 on the original
# universe's 2 cohorts alone, against a measured per-10-seed-ensemble noise of
# 0.0047).
#
# PRE-REGISTERED DECISION RULE (fix this BEFORE looking at results):
#   a knob PASSES iff  (a) mean effect vs baseline > +0.0050 IC, AND
#                      (b) the effect is positive on BOTH universes separately.
#   Anything smaller is inside noise. Anything positive on only one venue does not
#   generalise. Phase 2 then combines only the passers and re-tests; if nothing
#   passes, that is the result and the sweep stops.
#
# BASELINES ALREADY EXIST -- do NOT re-run them:
#   original    : test_output/mse_cohort_202{0,1}_aligned
#   mid_highmid : test_output/mse_pooled_mid_highmid/<set>_<cohort>
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ORIG="$REPO/scripts/stock_run/anvil_examm_hpsweep.sb"
MIDHI="$REPO/scripts/stock_run/anvil_mse_pooled_mid_highmid.sb"
DRY=${DRY:-0}

# arm_name : env assignments (identical semantics in both scripts)
declare -a ARMS=(
  "k_bp20    BPI=20"
  "k_isl20   ISLANDS_OVERRIDE=20"
  "k_ext250  EXTINCTION_OVERRIDE=250"
  "k_seq20   SEQLEN=20"
)

want="${1:-}"
sub() {  # script, extra-env, array
  if [ "$DRY" = "1" ]; then echo "  DRY sbatch --export=ALL,$2 --array=0-9 $(basename "$1")"
  else sbatch --export=ALL,"$2" --array=0-9 "$1" >/dev/null; fi
}

n=0
for spec in "${ARMS[@]}"; do
  read -r ARM ENVS <<< "$spec"
  [ -n "$want" ] && [ "$want" != "$ARM" ] && continue
  echo "=== arm $ARM ($ENVS) ==="
  # ---- original universe: 2 cohorts
  for C in cohort_2020_aligned cohort_2021_aligned; do
    sub "$ORIG" "ARM=custom,RUN_TAG=$ARM,COHORT=$C,$ENVS"; n=$((n+1))
  done
  # ---- mid_highmid: 4 sets x 3 cohorts
  for S in set1 set2 set3 set4; do
    for C in cohort_2020 cohort_2021 cohort_2022; do
      sub "$MIDHI" "SET=$S,COHORT=$C,RUN_TAG=$ARM,$ENVS"; n=$((n+1))
    done
  done
done
echo
echo "submitted $n array jobs x 10 seeds = $((n*10)) runs"
[ "$DRY" = "1" ] && echo "(dry run -- nothing was actually submitted)"
