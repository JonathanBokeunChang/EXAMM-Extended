#!/bin/bash
# Leave-one-site-out across all seven collocation sites, both arms, matched budget.
#
# WHY THIS IS CHEAP AND THE STOCK SWEEP WAS NOT. The PurpleAir panel is 46,126 rows spread
# over 335 short segments (~138 rows each), against the stock panel's 50 files of 3,542
# rows. Backprop through time costs per timestep, so a genome here trains on far less tape:
# measured 202 genomes in 9.2 s at 2 threads, against ~44 s for 500 genomes on 50 stock
# files at 8 threads. A 2,000-genome run is about 90 s, so the whole 7-site x 2-arm sweep
# is well under an hour on two threads and does not need the box to itself.
#
# ONE SITE HELD OUT AT A TIME, and purpleair_run.sh assigns the validation site itself (the
# next one alphabetically, wrapping), so no site is ever both selected on and scored.
#
# Usage:  bash scripts/purpleair/loso_sweep.sh [max_genomes] [threads] [tag]
#     or: MAXG=200 THREADS=2 TAG=loso200 bash scripts/purpleair/loso_sweep.sh
set -u
cd "$(dirname "$0")/../.."

# Positional args OR env vars. Both, because a detached launcher that forwards VAR=VAL but
# not argv silently started a SECOND sweep on this script's defaults, which shared an output
# tree with the running one and overwrote a completed site's fitness log before it was
# noticed. Accepting env vars removes that failure mode at the source.
MAXG=${MAXG:-${1:-2000}}
THREADS=${THREADS:-${2:-2}}
TAG=${TAG:-${3:-loso}}
LOG=test_output/purpleair_${TAG}.log

# One sweep per tag. Two sweeps sharing a tag write to the same run directories, and
# purpleair_run.sh does not lock, so the second silently clobbers the first's results.
LOCK="test_output/.purpleair_sweep_${TAG}.lock"
if ! mkdir "$LOCK" 2>/dev/null; then
    OWNER=$(cat "$LOCK/pid" 2>/dev/null || echo "")
    if [ -n "$OWNER" ] && kill -0 "$OWNER" 2>/dev/null; then
        echo "ERROR: sweep tag '$TAG' already running as pid $OWNER. Use a different TAG." >&2
        exit 1
    fi
    rm -rf "$LOCK"; mkdir "$LOCK" || exit 1
fi
echo $$ > "$LOCK/pid"
trap 'rm -rf "$LOCK"' EXIT

SITES=$(ls datasets/purpleair/*_seg*.csv | xargs -n1 basename | sed 's/_.*//' | sort -u)

{
  echo "### LOSO sweep | max_genomes $MAXG | threads $THREADS | sites: $(echo $SITES | tr '\n' ' ')"
  for ARM in exagpmin examm; do
      for SITE in $SITES; do
          echo "=== $ARM / held-out $SITE | $(date '+%H:%M:%S') ==="
          ARM="$ARM" bash scripts/purpleair/purpleair_run.sh "$SITE" "$MAXG" "$THREADS" "${TAG}_${ARM}" \
              2>&1 | grep -E "^(test site|valid site|Barkjohn|raw pa_cf1|evolved|---|=== held)" \
              || echo "  FAILED $ARM/$SITE"
      done
  done
  echo "### LOSO SWEEP COMPLETE $(date '+%Y-%m-%d %H:%M:%S')"
} >> "$LOG" 2>&1
