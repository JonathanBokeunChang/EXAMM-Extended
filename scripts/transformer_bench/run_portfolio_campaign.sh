#!/bin/bash
# The four-portfolio transformer campaign at PUBLISHED (author) settings, on a rented GPU box with
# no SLURM (RunPod, Lambda, vast.ai, ...). Loops SEED x MODEL x SET x COHORT and calls
# anvil_transformer.sb directly for each combination, substituting the SLURM_* env vars it expects.
#
#   bash scripts/transformer_bench/run_portfolio_campaign.sh
#
# THE GRID: 4 models x 4 sets x 3 cohorts x 10 seeds = 480 runs.
#   models   PatchTSTOfficial, CrossformerOfficial, ITransformerOfficial, DeformTime
#            -- every transformer in this study for which an AUTHORS' implementation is vendored.
#               DLinear is deliberately absent: it has no author profile here (it is a linear
#               model with no published ETTh1-style script in this harness) and it is already
#               complete on all 12 portfolio cells under the shared L=20 protocol.
#   sets     set1..set4, the four random 50-stock draws of Dr Lyu's 200-ticker mid/high-mid panel
#   cohorts  cohort_2020/2021/2022_aligned (train<=2020/21/22 -> trade 2022/23/24)
#
# WHY PROFILE=author AND L=96. This mirrors what already exists on the original universe
# (results/transformer_bench/author/<cohort>/<model>/L96/, 4 models x 2 cohorts x 10 seeds, all
# complete). Each model runs at ITS OWN published configuration, so the models are NOT matched to
# each other and these numbers must never share a table column with the shared-protocol L=20 runs
# -- including the DLinear already sitting at
# results/transformer_bench/mid_highmid/<set>/<cohort>/DLinear/. The two regimes answer different
# questions; anvil_transformer.sb keeps them in separate trees for exactly this reason.
#
# SEED IS THE OUTERMOST LOOP, AND THAT IS THE POINT. A rented pod can vanish (preemption, budget,
# a dropped ssh) at any moment, and 480 runs is many hours. Ordering by cell would leave a
# half-finished campaign with some cells at 10 seeds and the rest at zero -- an unbalanced design
# that cannot be analysed. Ordering by seed leaves EVERY cell at the same seed depth, which is a
# complete, balanced, immediately analysable campaign at whatever depth was reached. Stopping early
# then costs precision, not the experiment.
#
# Resumable: anvil_transformer.sb skips any run whose OUT/.done marker exists, so killing this loop
# and re-running it later picks up where it left off. One bad run must not cost the other 479, so
# failures are logged and the loop continues; a missing-.done summary prints at the end so nothing
# fails silently.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SEQ_LEN=${SEQ_LEN:-96}
PROFILE=${PROFILE:-author}
UNIVERSE=mid_highmid

# MODEL SPECS ARE `name` or `name:cf_config`, NOT bare names, because Crossformer's configuration is
# not uniquely determined by "the authors' settings" -- the authors publish several, and this study
# reports a SPECIFIC one.
#
# THE PAPER'S HEADLINE CROSSFORMER ROW IS THE **TRAFFIC** SCRIPT, NOT ETTh1. 05-results.tex reports
# Crossformer at 722,032 parameters (d_model 64, d_ff 128, n_heads 2, seg_len 12) and cites
# scripts/Traffic.sh; the ETTh1 script (11,185,944 params, d_model 256, d_ff 512) appears only as a
# secondary "(ETTh1, see text)" row, because it never opens the Algorithm 2 gate in either year.
# Traffic is also the ONLY published configuration across all four models that natively uses
# in_len 96 -- this study's protocol length -- so it is on-config on lookback where ETTh1 is not.
#
# Getting this wrong is invisible in the output: an etth1 campaign completes perfectly happily and
# produces a Crossformer column that simply is not the one the paper reports.
#
# WHY PER MODEL AND NOT ONE CAMPAIGN-WIDE CF_CONFIG. CF_CONFIG=traffic makes anvil_transformer.sb
# derive VARIANT=cftraffic, which redirects the OUTPUT PATH into an author/var_cftraffic/ tree. Set
# globally, that would file PatchTST, iTransformer and DeformTime -- none of which read CF_CONFIG at
# all -- under a "traffic" label describing a flag they never saw, in a tree separate from their own
# published settings. So the config travels WITH the model that owns it.
read -r -a MODELS  <<< "${MODELS:-PatchTSTOfficial CrossformerOfficial:traffic ITransformerOfficial DeformTime}"
read -r -a SETS    <<< "${SETS:-set1 set2 set3 set4}"
read -r -a COHORTS <<< "${COHORTS:-cohort_2020_aligned cohort_2021_aligned cohort_2022_aligned}"
read -r -a SEEDS   <<< "${SEEDS:-1 2 3 4 5 6 7 8 9 10}"
# Read as strings then split, never as arrays directly: with MODELS already an array,
# ${MODELS:-...} expands to its FIRST ELEMENT only and the campaign quietly runs one model.

# Split "name:cfg" into the model name and the Crossformer config it carries. A bare name means
# etth1, which is anvil_transformer.sb's own default and the only value that leaves VARIANT empty.
spec_model() { echo "${1%%:*}"; }
spec_cfg()   { local c="${1#*:}"; [ "$c" = "$1" ] && c=etth1; echo "$c"; }
spec_var()   { local c; c=$(spec_cfg "$1"); [ "$c" = "etth1" ] && echo "" || echo "cf${c}"; }

# The output path MUST be derived exactly as anvil_transformer.sb derives it, because this script
# uses it to decide whether a run is already finished. If the two ever disagree, this loop reads
# .done markers at a path nothing writes to -- and either redoes the whole campaign or, far worse,
# finds another tree's markers, skips all 480 runs and exits 0 reporting success. Keep this
# function and the .sb's OUT/MID block in step.
out_dir() {  # $1=set $2=cohort $3=model $4=seed $5=variant(may be empty)
  if [ "$PROFILE" = "author" ]; then
    echo "$REPO/results/transformer_bench/mid_highmid/$1/author${5:+/var_$5}/$2/$3/L${SEQ_LEN}/seed_$4"
  elif [ "$SEQ_LEN" = "20" ]; then
    echo "$REPO/results/transformer_bench/mid_highmid/$1/$2/$3/seed_$4"
  else
    echo "$REPO/results/transformer_bench/mid_highmid/$1/longctx/$2/$3/L${SEQ_LEN}/seed_$4"
  fi
}

# SHARDING. One process per GPU, each taking a disjoint slice, so N GPUs give ~Nx throughput.
#
# SHARD BY A HASH OF THE RUN IDENTITY, NOT BY `counter % NSHARDS`. A positional counter ALIASES
# against the loop structure whenever a loop's size shares a factor with NSHARDS, and it does so
# invisibly -- the per-shard COUNTS stay perfectly even while the content stratifies. Measured on
# this exact grid with the innermost loop over 3 cohorts:
#
#   NSHARDS=3, positional:  shard 0 -> 160 runs, ALL cohort_2020
#                           shard 1 -> 160 runs, ALL cohort_2021
#                           shard 2 -> 160 runs, ALL cohort_2022
#
# 160/160/160 looks like a clean split and is not one. It breaks two things at once: cohort_2022
# trains on ~13% more rows than cohort_2020, so shard 2 runs systematically longer than shard 0 and
# the GPUs finish far apart; and a pod that dies mid-campaign leaves cohort_2020 deep and
# cohort_2022 shallow, which is precisely the unbalanced design the seed-outermost loop order above
# exists to prevent. (run_remote_campaign.sh is not affected only because its innermost loop is 10
# seeds and 10 % 3 = 1, so it happens to rotate. That is luck, not design.)
#
# cksum over "set/cohort/model/seed" is deterministic, stable across restarts, independent of loop
# order, and cannot alias against any NSHARDS. Shard sizes are then approximately rather than
# exactly equal, which is the right trade: an even MIX matters here and a +/-15-run imbalance
# across a 10-hour campaign does not.
#
# Two properties make this safe without locking: every run writes to its own OUT directory, and a
# run is claimed by exactly one shard. The .done check still runs inside each shard, so a shard
# restarted after a crash resumes correctly and never redoes another shard's finished work.
#
#   CUDA_VISIBLE_DEVICES=0 SHARD=0 NSHARDS=2 bash scripts/transformer_bench/run_portfolio_campaign.sh &
#   CUDA_VISIBLE_DEVICES=1 SHARD=1 NSHARDS=2 bash scripts/transformer_bench/run_portfolio_campaign.sh &
SHARD=${SHARD:-0}
NSHARDS=${NSHARDS:-1}
[ "$SHARD" -lt "$NSHARDS" ] || { echo "ERROR: SHARD=$SHARD must be < NSHARDS=$NSHARDS" >&2; exit 1; }

# ONE PROCESS PER SHARD, ENFORCED. The .done markers make a shard resumable, but they only protect
# against SEQUENTIAL re-entry: two processes started at the same time both see no .done, both train
# the same run, into the same OUT directory, under the same --model_id. They then race on the
# harness's shared results/ tree and on predictions.csv, and the survivor is whichever finished
# last -- with no error anywhere.
#
# This is not hypothetical. Launching this campaign's bootstrap three times by accident produced
# three concurrent trainers on one output directory, and nothing in the output said so; it was
# visible only in `ps`. A lock costs nothing and removes the whole class.
#
# mkdir is the lock because it is atomic on POSIX and fails if the directory exists -- no flock
# dependency, works over the network filesystem. The PID/host inside is for diagnosing a stale one.
LOCKDIR="$REPO/results/transformer_bench/.campaign_locks"
mkdir -p "$LOCKDIR"
LOCK="$LOCKDIR/shard_${SHARD}_of_${NSHARDS}.lock"
if ! mkdir "$LOCK" 2>/dev/null; then
  echo "ERROR: shard $SHARD is already running (lock $LOCK held by $(cat "$LOCK/owner" 2>/dev/null || echo '?'))." >&2
  echo "       If that process is gone, remove the lock and retry:  rm -rf '$LOCK'" >&2
  exit 1
fi
printf 'pid %s on %s since %s\n' "$$" "$(hostname)" "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" > "$LOCK/owner"
# Release on any exit, including Ctrl-C and SIGTERM, so a normal stop never leaves a stale lock.
trap 'rm -rf "$LOCK"' EXIT INT TERM

# OWNERSHIP: RANK IN HASH ORDER, NOT HASH MODULO. Plain `hash % NSHARDS` is unaliased but only
# APPROXIMATELY balanced, and the error grows as the grid shrinks. Measured on the 240-run
# two-model campaign at NSHARDS=3: 78/72/90. Wall-clock is set by the SLOWEST shard, so that is
# 3 GPUs delivering 2.7x -- paying for three cards and idling two while the third finishes.
#
# Sorting by hash and then taking every NSHARDS-th row gives exactly equal counts (+/-1) while
# keeping the mix scrambled, so it inherits the anti-aliasing property and drops the imbalance.
#
# Ownership is decided here, up front, but EXECUTION still follows the seed-outermost order of the
# main loop below -- the sort determines who runs what, never in what order. Those are separate
# concerns and conflating them would trade the balanced-degradation property away for balance.
#
# NOTE this balances COUNTS, not TIME. Crossformer-traffic and iTransformer differ in per-run cost,
# so equal counts still leave some spread; it is far smaller than 78-vs-90 and needs no locking.
OWNED=$'\n'
if [ "$NSHARDS" -gt 1 ]; then
  # NO `trap ... EXIT` here: bash keeps ONE handler per signal, so setting one would silently
  # replace the lock-release trap above and every run would leave a stale lock behind. The temp
  # file is removed inline instead, right after it is consumed.
  _tmp=$(mktemp)
  for _s in "${SEEDS[@]}"; do for _m in "${MODELS[@]}"; do
    _mm=$(spec_model "$_m")
    for _st in "${SETS[@]}"; do for _c in "${COHORTS[@]}"; do
      _k="$_st/$_c/$_mm/$_s"
      printf '%s\t%s\n' "$(printf '%s' "$_k" | cksum | cut -d' ' -f1)" "$_k" >> "$_tmp"
    done; done
  done; done
  OWNED=$'\n'$(sort -k1,1n -k2,2 "$_tmp" | awk -v s="$SHARD" -v n="$NSHARDS" 'NR%n==s{print $2}')$'\n'
  rm -f "$_tmp"
fi

# Returns 0 if this run belongs to this shard.
mine() {  # $1=set $2=cohort $3=model $4=seed
  [ "$NSHARDS" -eq 1 ] && return 0
  case "$OWNED" in *$'\n'"$1/$2/$3/$4"$'\n'*) return 0 ;; *) return 1 ;; esac
}

TOTAL=$(( ${#SEEDS[@]} * ${#MODELS[@]} * ${#SETS[@]} * ${#COHORTS[@]} ))

# ---- preflight. All of this is cheap and all of it has bitten this project before, so it runs
# BEFORE the first GPU-hour rather than failing 40 runs deep.
for SET in "${SETS[@]}"; do
  for COHORT in "${COHORTS[@]}"; do
    D="$REPO/datasets/walkforward/mid_highmid_price/$SET/$COHORT"
    [ -d "$D" ] || { echo "ERROR: missing data dir $D" >&2; exit 1; }
    n=$(ls "$D"/*_train.csv 2>/dev/null | wc -l | tr -d ' ')
    [ "$n" -eq 50 ] || { echo "ERROR: $D has $n train files, expected 50" >&2; exit 1; }
    # test borrows seq_len-pred_len rows of lookback from the tail of val, so a val split shorter
    # than that silently truncates the test window set. At L=96 we need 95; val is ~250.
    v=$(( $(wc -l < "$(ls "$D"/*_val.csv | head -1)") - 1 ))
    [ "$v" -ge "$SEQ_LEN" ] || { echo "ERROR: $D val has $v rows, too short for L=$SEQ_LEN" >&2; exit 1; }
  done
done
echo "### preflight ok: ${#SETS[@]} sets x ${#COHORTS[@]} cohorts, 50 stocks each, val >= L=$SEQ_LEN"
echo "### campaign: $TOTAL runs | profile=$PROFILE | L=$SEQ_LEN | shard $SHARD/$NSHARDS"
for SPEC in "${MODELS[@]}"; do
  v=$(spec_var "$SPEC"); m=$(spec_model "$SPEC")
  # Only Crossformer reads CF_CONFIG (anvil_transformer.sb consults it inside that case branch
  # alone), so printing "cf_config=etth1" beside the other three would suggest they are running an
  # ETTh1 configuration they never see. Show the config only where it actually binds.
  if [ "$m" = "CrossformerOfficial" ]; then
    echo "###   $m | config=$(spec_cfg "$SPEC") | tree=author${v:+/var_$v}"
  else
    echo "###   $m | tree=author${v:+/var_$v}"
  fi
done

done_n=0
for SET in "${SETS[@]}"; do
  for COHORT in "${COHORTS[@]}"; do
    for SPEC in "${MODELS[@]}"; do
      for SEED in "${SEEDS[@]}"; do
        [ -f "$(out_dir "$SET" "$COHORT" "$(spec_model "$SPEC")" "$SEED" "$(spec_var "$SPEC")")/.done" ] \
          && done_n=$((done_n + 1))
      done
    done
  done
done
echo "### already complete: $done_n / $TOTAL"

fail_count=0
started=$(date +%s)
for SEED in "${SEEDS[@]}"; do            # outermost: see the header
  for SPEC in "${MODELS[@]}"; do
    MODEL=$(spec_model "$SPEC"); CFG=$(spec_cfg "$SPEC"); VAR=$(spec_var "$SPEC")
    for SET in "${SETS[@]}"; do
      for COHORT in "${COHORTS[@]}"; do
        mine "$SET" "$COHORT" "$MODEL" "$SEED" || continue
        OUT="$(out_dir "$SET" "$COHORT" "$MODEL" "$SEED" "$VAR")"
        if [ -f "$OUT/.done" ]; then
          echo "=== $MODEL | $SET/$COHORT | seed $SEED -- already done, skipping ==="
          continue
        fi
        el=$(( $(date +%s) - started ))
        echo "=== $MODEL${VAR:+ ($CFG)} | $SET/$COHORT | L=$SEQ_LEN | profile=$PROFILE | seed $SEED | +$((el/60))m ==="
        SLURM_SUBMIT_DIR="$REPO" SLURM_ARRAY_TASK_ID="$SEED" MODEL="$MODEL" COHORT="$COHORT" \
          UNIVERSE="$UNIVERSE" SET="$SET" SEQ_LEN="$SEQ_LEN" PROFILE="$PROFILE" \
          CF_CONFIG="$CFG" VARIANT="$VAR" \
          bash "$REPO/scripts/transformer_bench/anvil_transformer.sb"
        rc=$?
        if [ "$rc" -ne 0 ]; then
          echo "FAILED $MODEL $SET/$COHORT seed $SEED rc=$rc -- continuing with next" >&2
          fail_count=$((fail_count + 1))
        fi
      done
    done
  done
done

echo
echo "### campaign loop complete ($fail_count failed run(s)) in $(( ($(date +%s) - started) / 60 )) min"
echo "### missing .done markers (across ALL shards, not just this one):"
missing=0
for SET in "${SETS[@]}"; do
  for COHORT in "${COHORTS[@]}"; do
    for SPEC in "${MODELS[@]}"; do
      MODEL=$(spec_model "$SPEC"); VAR=$(spec_var "$SPEC")
      for SEED in "${SEEDS[@]}"; do
        [ -f "$(out_dir "$SET" "$COHORT" "$MODEL" "$SEED" "$VAR")/.done" ] \
          || { echo "  $SET/$COHORT/${MODEL}${VAR:+[$VAR]}/seed_$SEED"; missing=$((missing + 1)); }
      done
    done
  done
done
if [ "$missing" -eq 0 ]; then
  echo "  (none -- all $TOTAL runs complete)"
  exit 0
fi
# Exit non-zero on an incomplete campaign, explicitly rather than by falling off the end of a
# failed test. This is what makes `bash run_portfolio_campaign.sh && bash eval...` safe to chain,
# and what lets a supervising loop tell "finished" from "the pod died at run 300".
echo "### $missing of $TOTAL runs still missing -- re-run this script to resume"
exit 1
