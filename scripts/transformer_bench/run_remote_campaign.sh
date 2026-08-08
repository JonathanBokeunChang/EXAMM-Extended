#!/bin/bash
# Unattended full transformer campaign on a rented GPU box with no SLURM (RunPod, Lambda, vast.ai,
# ...). Loops MODEL x COHORT x SEED and calls anvil_transformer.sb directly for each combination,
# substituting the SLURM_* env vars it expects instead of going through sbatch.
#
#   bash scripts/transformer_bench/run_remote_campaign.sh
#
# Resumable: anvil_transformer.sb itself skips any run whose OUT/.done marker already exists, so
# killing this loop (pod stopped, ssh dropped, etc.) and re-running it later just picks up where it
# left off -- nothing here needs its own checkpointing.
#
# One bad run should not cost the other 59: failures are logged and the loop continues rather than
# aborting, and a missing-.done summary is printed at the end so nothing failed silently.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
COHORTS=(cohort_2020_aligned cohort_2021_aligned)
SEEDS=(1 2 3 4 5 6 7 8 9 10)
SEQ_LEN=${SEQ_LEN:-20}
# PROFILE must be threaded through for the same reason SEQ_LEN is: it changes the output path in
# anvil_transformer.sb, so if out_dir() below did not account for it this loop would test the wrong
# .done markers -- either re-running finished work or, far worse, seeing another profile's markers
# and skipping every run while exiting 0. MODELS can also be narrowed (e.g. MODELS=PatchTST) because
# PROFILE=author is pinned per model and deliberately refuses to run un-pinned ones.
PROFILE=${PROFILE:-shared}
# Read as a string from the environment so MODELS="PatchTST" narrows the campaign. This must stay
# ABOVE any array assignment of MODELS: if MODELS were already an array here, ${MODELS:-...} would
# silently expand to its FIRST ELEMENT only and the campaign would quietly run one model.
read -r -a MODELS <<< "${MODELS:-Crossformer DeformTime PatchTST}"

# The output path MUST be derived exactly as anvil_transformer.sb derives it, because this script
# uses it to decide whether a run is already finished. It previously hardcoded the L=20 path
# (results/transformer_bench/<cohort>/<model>/seed_N). At SEQ_LEN=96 the job script writes to
# results/transformer_bench/longctx/<cohort>/<model>/L96/seed_N instead, so once an L=20 campaign
# had populated the default tree, an L=96 campaign would have found 60 stale .done markers at the
# wrong path, skipped every single run, and exited 0 reporting success -- a silent total loss of
# the campaign. Keep this function and anvil_transformer.sb's path block in agreement.
# VARIANT must be threaded exactly as PROFILE and SEQ_LEN are, and for the same reason. A
# CF_CONFIG=traffic campaign writes to author/var_cftraffic/..., but if out_dir() still pointed at
# author/... it would find the 10 EXISTING ETTh1 .done markers, skip all 20 runs, and exit 0
# reporting success -- a silent total loss of the campaign, invisible in the output.
# CF_CONFIG != etth1 derives VARIANT the same way anvil_transformer.sb does; keep the two in step.
CF_CONFIG=${CF_CONFIG:-etth1}
VARIANT=${VARIANT:-}
if [ "$CF_CONFIG" != "etth1" ] && [ -z "$VARIANT" ]; then VARIANT="cf${CF_CONFIG}"; fi

out_dir() {  # $1=cohort $2=model $3=seed
  if [ "$PROFILE" = "author" ]; then
    echo "$REPO/results/transformer_bench/author${VARIANT:+/var_$VARIANT}/$1/$2/L${SEQ_LEN}/seed_$3"
  elif [ "$SEQ_LEN" = "20" ]; then
    echo "$REPO/results/transformer_bench/$1/$2/seed_$3"
  else
    echo "$REPO/results/transformer_bench/longctx/$1/$2/L${SEQ_LEN}/seed_$3"
  fi
}

# SHARDING. One process per GPU, each taking a disjoint slice of the work, so N GPUs give ~Nx
# throughput. Slicing is by a global counter over (model, cohort, seed) rather than by seed alone,
# so the split stays even no matter how many models or cohorts are selected.
#
# Two properties make this safe without any locking: every run writes to its own OUT directory, and
# a run is claimed by exactly one shard. The .done check still runs inside each shard, so a shard
# restarted after a crash resumes correctly and never redoes another shard's finished work.
#
#   CUDA_VISIBLE_DEVICES=0 SHARD=0 NSHARDS=3 bash run_remote_campaign.sh &
#   CUDA_VISIBLE_DEVICES=1 SHARD=1 NSHARDS=3 bash run_remote_campaign.sh &
#   CUDA_VISIBLE_DEVICES=2 SHARD=2 NSHARDS=3 bash run_remote_campaign.sh &
SHARD=${SHARD:-0}
NSHARDS=${NSHARDS:-1}
[ "$SHARD" -lt "$NSHARDS" ] || { echo "ERROR: SHARD=$SHARD must be < NSHARDS=$NSHARDS" >&2; exit 1; }

fail_count=0
idx=-1
for MODEL in "${MODELS[@]}"; do
  for COHORT in "${COHORTS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
      idx=$((idx + 1))
      [ $((idx % NSHARDS)) -eq "$SHARD" ] || continue
      OUT="$(out_dir "$COHORT" "$MODEL" "$SEED")"
      if [ -f "$OUT/.done" ]; then
        echo "=== $MODEL | $COHORT | L=$SEQ_LEN | profile=$PROFILE | seed $SEED -- already done, skipping ==="
        continue
      fi
      echo "=== $MODEL | $COHORT | L=$SEQ_LEN | profile=$PROFILE | seed $SEED ==="
      SLURM_SUBMIT_DIR="$REPO" SLURM_ARRAY_TASK_ID="$SEED" MODEL="$MODEL" COHORT="$COHORT" \
        SEQ_LEN="$SEQ_LEN" PROFILE="$PROFILE" CF_CONFIG="$CF_CONFIG" VARIANT="$VARIANT" \
        bash "$REPO/scripts/transformer_bench/anvil_transformer.sb"
      rc=$?
      if [ "$rc" -ne 0 ]; then
        echo "FAILED $MODEL $COHORT seed $SEED rc=$rc -- continuing with next" >&2
        fail_count=$((fail_count + 1))
      fi
    done
  done
done

echo
echo "### campaign loop complete ($fail_count failed run(s))"
echo "### missing .done markers (across ALL shards, not just this one):"
missing=0
for MODEL in "${MODELS[@]}"; do
  for COHORT in "${COHORTS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
      OUT="$(out_dir "$COHORT" "$MODEL" "$SEED")"
      [ -f "$OUT/.done" ] || { echo "  L=$SEQ_LEN $COHORT/$MODEL/seed_$SEED"; missing=$((missing + 1)); }
    done
  done
done
[ "$missing" -eq 0 ] && echo "  (none -- all $(( ${#MODELS[@]} * ${#COHORTS[@]} * ${#SEEDS[@]} )) runs complete)"
