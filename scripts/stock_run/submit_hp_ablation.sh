#!/bin/bash
# One-factor-at-a-time hyperparameter ABLATION across BOTH universes.
# Run on Anvil. bash only.
#
#   sh scripts/stock_run/submit_hp_ablation.sh                     # all 4 arms, both venues
#   sh scripts/stock_run/submit_hp_ablation.sh k_isl20 k_ext250    # named arms only
#   DRY=1 sh scripts/stock_run/submit_hp_ablation.sh k_seq20       # print, do not submit
#
# k_bp20 is FINISHED and its verdict is recorded below -- naming it again would
# resubmit 140 runs on top of completed .done files. The three remaining arms are
# k_isl20, k_ext250, k_seq20.
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
# CLARIFICATION to (a), added 2026-08-02 after arm k_bp20 returned and BEFORE the
# other three arms were submitted. "Mean effect" was ambiguous between the
# cell-weighted mean over all 14 cells and the mean of the two per-universe means.
# It is the CELL-WEIGHTED MEAN. Averaging a 12-cell estimate with a 2-cell estimate
# as equals gives the noisier one six times its due weight, and the 2-cell universe
# is the noisy one: two byte-identical baseline campaigns (ARM=baseline here vs the
# paper's own runs in test_output/mse_cohort_*_aligned) differ by 0.0182 IC and
# 38 pp of Algorithm-2 return on cohort_2021. That spread is 3.6x the decision
# threshold, so those two cells cannot resolve a +0.0050 effect at all and must not
# be allowed to carry the verdict. The threshold itself is UNCHANGED at +0.0050.
#
# RESULT SO FAR -- k_bp20 (--bp_iterations 20), all 140 runs, reported under BOTH
# readings so the clarification above cannot be mistaken for moving the bar:
#   cell-weighted mean   +0.0055 IC   (passes (a) by 0.0005, i.e. 1/9th of the
#                                      0.0099 per-cell sd; t=2.06, p=0.060)
#   mean-of-universe-means +0.0111 IC
#   BUT: leave-one-out flips it to FAIL on any of the three largest cells, and it is
#   carried by orig/2023 (+0.0331) whose BASELINE is the known low draw above --
#   rescore that cell against the paper's campaign and the mean is +0.0042, FAIL.
#   The 12 powered cells alone give +0.0032 (7/12, sign p=0.77).
#   Trading, Algorithm 2 @ L/S 10: +2.13 pp overall (t=0.65, p=0.53, 8/14), and
#   +0.49 pp on the 12 powered cells (6/12) -- a coin flip.
#   VERDICT: k_bp20 FAILS. Both metrics agree once the 2 uninformative cells are set
#   aside. Raw numbers: results/gate_threshold_sweep.csv's sibling bp20_ic.csv.
#
# ============================ SWEEP COMPLETE: 4/4 NULL ============================
# All four arms returned, 560 runs. NOTHING PASSES, so per the rule above the sweep
# stops here and there is no phase 2. Raw numbers in results/arms3_ic.csv and
# results/arms3_trade.csv (k_bp20 in results/bp20_ic.csv, results/bp20_trade.csv).
#
#             IC delta                          Algorithm-2 delta @ L/S 10
#   arm       all 14    powered 12  wins   t     p       all 14    powered 12  wins
#   k_bp20    +0.0055   +0.0032     9/14  2.06  0.060    +2.13 pp   +0.49 pp   8/14
#   k_isl20   +0.0018   +0.0019    10/14  1.11  0.288    +0.98 pp   -2.91 pp   7/14
#   k_ext250  +0.0027   +0.0028     9/14  1.53  0.150    +0.51 pp   -2.08 pp   7/14
#   k_seq20   +0.0014   +0.0009    10/14  0.40  0.698    -6.78 pp   -5.14 pp   7/14
#
# Every arm is below the +0.0050 bar on the 12 powered cells, no arm clears p=0.05,
# and every arm is NEGATIVE on trading over those 12 cells. k_bp20 came closest and
# only by riding one cell whose baseline was a known low draw. This is the fifth
# independent measurement agreeing with the "do not tune" finding: on this venue
# validation fitness does not predict test IC (r = +0.065), and the knobs do not move
# the mean past the noise floor either.
#
# k_seq20 is the informative failure. Sequence slicing takes weight updates per epoch
# from 50 to ~3,250 -- the only arm with a real mechanism behind it -- and it produced
# the LARGEST per-cell spread in the sweep (IC sd 0.0128 vs ~0.0060 for the others;
# trading sd 20.4 pp) around a mean of zero. More gradient steps bought variance, not
# skill. Its two worst cells (set4/2023 -0.0247, set1/2024 -0.0201) are the biggest
# single swings anywhere in the sweep.
#
# This arm is also the one whose evaluation was WRONG until it was fixed twice, so its
# numbers above are the in-regime ones (--test_sequence_length 20, matching training):
#   - scoring it unsliced, as the old evaluator did, differs by -0.0045 (cohort_2020)
#     and +0.0127 (cohort_2021) -- at or above the decision threshold, and NOT
#     directional, so the old setup was folding a coin-flip into the arm's estimate
#   - blocked scoring initially crashed 9 of 10 genomes on a 249-row test year and the
#     evaluator averaged the one survivor into a confident-looking +0.000071
# Both fixed (rnn_recurrent_edge.cxx short-block clamp; eval_ic_run.sh hard-fails on a
# short run). Anything re-derived from this arm must use the in-regime scoring.
# =================================================================================
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

WANT=("$@")            # empty = all arms
wanted() {
  [ "${#WANT[@]}" -eq 0 ] && return 0
  for w in "${WANT[@]}"; do [ "$w" = "$1" ] && return 0; done
  return 1
}
# Reject a typo'd arm name outright. Silently submitting nothing looks identical to a
# successful no-op, and you would not find out until the morning.
for w in ${WANT[@]+"${WANT[@]}"}; do
  ok=0; for spec in "${ARMS[@]}"; do set -- $spec; [ "$1" = "$w" ] && ok=1; done
  [ "$ok" = "1" ] || { echo "ERROR: unknown arm '$w' (known: k_bp20 k_isl20 k_ext250 k_seq20)" >&2; exit 1; }
done

sub() {  # script, extra-env, array
  if [ "$DRY" = "1" ]; then echo "  DRY sbatch --export=ALL,$2 --array=0-9 $(basename "$1")"
  else sbatch --export=ALL,"$2" --array=0-9 "$1" >/dev/null; fi
}

n=0
for spec in "${ARMS[@]}"; do
  read -r ARM ENVS <<< "$spec"
  wanted "$ARM" || continue
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
