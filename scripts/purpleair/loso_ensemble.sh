#!/bin/bash
# Leave-one-site-out with a MULTI-RUN ENSEMBLE per site, which single-run LOSO is not.
#
# WHY. EXAMM is nondeterministic regardless of --seed: worker threads race for examm_mutex
# in multithreaded/examm_mt.cxx:60 and the MPI backend consumes results in arrival order via
# MPI_Probe(MPI_ANY_SOURCE), so the same configuration explores a different search each time.
# Measured on this dataset: EXA-GP-MIN, held-out az, 200 genomes, run twice -> 24-hour RMSE
# 1.857 and 2.470. The first beats the deployed incumbent (2.105), the second loses to it.
# A one-run-per-site LOSO therefore reports a coin flip, and the stock campaign already
# settled this by ensembling ten runs per cell.
#
# Predictions are averaged across repeats BEFORE scoring, matching what
# eval_ensemble_ic.py does on the stock side, so the reported number is the deployed
# ensemble's rather than a mean of individually-scored runs.
#
# Usage:  MAXG=200 REPEATS=10 TAG=ens200 bash scripts/purpleair/loso_ensemble.sh
set -u
cd "$(dirname "$0")/../.."

MAXG=${MAXG:-200}
REPEATS=${REPEATS:-10}
THREADS=${THREADS:-2}
TAG=${TAG:-ens}
ARMS=${ARMS:-"exagpmin examm"}
LOG=test_output/purpleair_${TAG}.log

LOCK="test_output/.purpleair_sweep_${TAG}.lock"
if ! mkdir "$LOCK" 2>/dev/null; then
    OWNER=$(cat "$LOCK/pid" 2>/dev/null || echo "")
    if [ -n "$OWNER" ] && kill -0 "$OWNER" 2>/dev/null; then
        echo "ERROR: tag '$TAG' already running as pid $OWNER" >&2; exit 1
    fi
    rm -rf "$LOCK"; mkdir "$LOCK" || exit 1
fi
echo $$ > "$LOCK/pid"
trap 'rm -rf "$LOCK"' EXIT

SITES=${SITES:-$(ls datasets/purpleair/*_seg*.csv | xargs -n1 basename | sed 's/_.*//' | sort -u)}

{
  echo "### LOSO ENSEMBLE | genomes $MAXG | repeats $REPEATS | threads $THREADS | $(date '+%H:%M:%S')"
  for ARM in $ARMS; do
      for SITE in $SITES; do
          for R in $(seq 1 "$REPEATS"); do
              ARM="$ARM" bash scripts/purpleair/purpleair_run.sh \
                  "$SITE" "$MAXG" "$THREADS" "${TAG}_${ARM}_r${R}" >/dev/null 2>&1 \
                  || echo "  FAILED $ARM/$SITE/r$R"
          done
          echo "=== $ARM / $SITE : $REPEATS repeats done $(date '+%H:%M:%S') ==="
          python3 scripts/purpleair/score_purpleair_ensemble.py \
              --tag "$TAG" --arm "$ARM" --site "$SITE" --repeats "$REPEATS"
      done
  done
  echo "### LOSO ENSEMBLE COMPLETE $(date '+%Y-%m-%d %H:%M:%S')"
} >> "$LOG" 2>&1
