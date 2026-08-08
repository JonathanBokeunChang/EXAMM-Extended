#!/bin/bash
# Pull finished transformer runs off a rented pod, incrementally, while the campaign is still going.
#
#   bash scripts/transformer_bench/sync_from_pod.sh root@1.2.3.4:22 /workspace/EXAMM-Extended
#   bash scripts/transformer_bench/sync_from_pod.sh root@1.2.3.4:22 /workspace/EXAMM-Extended --watch 600
#
# WHY THIS EXISTS. The campaign's .done markers make it resumable against a dying PROCESS -- crash,
# preemption, dropped ssh: restart the shards and they resume exactly where they stopped. They do
# nothing about a dying DISK. A pod stopped for zero balance, or terminated, takes its container
# disk with it, and then resumability is worthless because there is nothing left to resume from.
# Running out of credit mid-campaign is exactly that case, so the results have to live somewhere
# other than the machine that is billing you.
#
# WHAT IT COPIES. Only the deliverable and its provenance -- about 1.6 MB per run, ~400 MB for a
# 240-run campaign, which is small enough to re-run every few minutes over ssh without noticeable
# cost. Checkpoints and the harness's duplicate pred.npy/true.npy are deliberately NOT copied:
# they are ~1.2 MB/run, are never read by any downstream script, and are reproducible from the
# genome anyway. train_index.csv is also skipped -- 3.3 MB/run, the single largest file, and only
# ever used to build the run it came from.
#
# TWO PASSES, AND THE ORDER MATTERS. .done is synced only AFTER every data file has landed. If a
# sync were interrupted halfway through a run directory in one pass, rsync could leave a local
# .done sitting beside a missing or truncated predictions.csv -- a directory that claims to be
# complete and is not. That is the same class of silent corruption that lost the timing logs, so
# the marker is written last, on purpose.
set -uo pipefail

DEST_DEFAULT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HOST=${1:?usage: sync_from_pod.sh USER@HOST[:PORT] REMOTE_REPO_DIR [--watch SECONDS]}
REMOTE=${2:?usage: sync_from_pod.sh USER@HOST[:PORT] REMOTE_REPO_DIR [--watch SECONDS]}
WATCH=0
[ "${3:-}" = "--watch" ] && WATCH=${4:-600}

# RunPod hands out a non-22 ssh port; accept USER@HOST:PORT and split it for -p.
PORT=22
case "$HOST" in *:*) PORT="${HOST##*:}"; HOST="${HOST%:*}" ;; esac

# The pod key is not the default id_*, so ssh must be told which identity to use or every rsync
# fails with "Permission denied (publickey)". Overridable for a different pod.
SSH_KEY=${SSH_KEY:-$HOME/.ssh/runpod_examm}
[ -f "$SSH_KEY" ] || { echo "ERROR: no ssh key at $SSH_KEY (set SSH_KEY=...)" >&2; exit 1; }
SSH_CMD="ssh -p $PORT -i $SSH_KEY -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=15"

SUB=results/transformer_bench/mid_highmid
LOCAL="$DEST_DEFAULT/$SUB"
mkdir -p "$LOCAL"

# COUNT ONLY WHAT THIS CAMPAIGN PRODUCES. results/transformer_bench/mid_highmid ALSO holds the 120
# shared-protocol L=20 DLinear runs from an earlier campaign, at <set>/<cohort>/DLinear/. Counting
# everything under $LOCAL reported "120 complete runs" before a single file had transferred --
# a progress number that looks like success and measures nothing. Restrict to the author tree.
count_author() { find "$LOCAL" -path "*/author*" -name "$1" 2>/dev/null | wc -l | tr -d ' '; }

sync_once() {
  local rc1 rc2
  # pass 1 -- data files, never .done
  rsync -rlt --partial --prune-empty-dirs -e "$SSH_CMD" \
    --include='*/' \
    --include='predictions.csv' --include='timing.json' --include='train.log' \
    --include='test_index.csv' --include='scaler.json' \
    --exclude='*' \
    "$HOST:$REMOTE/$SUB/" "$LOCAL/"
  rc1=$?
  # pass 2 -- markers only, after their data is safely down
  rsync -rlt --partial --prune-empty-dirs -e "$SSH_CMD" \
    --include='*/' --include='.done' --exclude='*' \
    "$HOST:$REMOTE/$SUB/" "$LOCAL/"
  rc2=$?
  [ $rc1 -eq 0 ] && [ $rc2 -eq 0 ]
}

report() {
  local d p
  d=$(count_author .done)
  p=$(count_author predictions.csv)
  echo "### local copy: $d complete run(s), $p predictions.csv, $(du -sh "$LOCAL" 2>/dev/null | cut -f1) on disk"
  # A .done without predictions.csv means pass 2 ran against a pass-1 that had not finished.
  # Harmless -- the next sync fixes it -- but say so rather than let it look like a complete run.
  [ "$d" -gt "$p" ] && echo "### NOTE: $((d-p)) marker(s) ahead of their data; next sync will reconcile" >&2
  return 0
}

if [ "$WATCH" -gt 0 ]; then
  echo "### syncing $HOST:$REMOTE/$SUB every ${WATCH}s -- Ctrl-C to stop"
  while true; do
    sync_once && report || echo "### sync failed (pod down?); retrying in ${WATCH}s" >&2
    sleep "$WATCH"
  done
else
  sync_once || { echo "### sync FAILED" >&2; report; exit 1; }
  report
fi
