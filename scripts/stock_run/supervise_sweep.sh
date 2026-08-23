#!/bin/bash
# Keep a sweep running across external kills.
#
# WHY. These sweeps get terminated from outside the run -- a harness reaping background
# tasks, a laptop sleeping, an OOM. It has happened twice, and both times the sweep stopped
# silently after a completed run with no error line, so the only symptom was progress
# quietly ceasing. Since every sweep is already resumable (a finished seed writes .done and
# is skipped; a stale lock is reclaimed once its owner pid is gone), the correct response to
# an unexplained death is simply to start it again.
#
# WHAT IT DOES NOT DO. It does not retry a sweep that is FAILING -- run_exagp_local.sh exits
# non-zero on a genuinely broken run, and restarting that in a loop would spin forever
# burning CPU. So a restart only happens when the sweep made progress since the last attempt
# (at least one new .done). No progress means something is actually wrong; stop and say so.
#
# Usage:
#   python3 <launcher> scripts/stock_run/supervise_sweep.sh <log> SWEEP=<script> [VAR=VAL ...]
#   SWEEP=scripts/stock_run/run_exagp_panel_sweep.sh bash scripts/stock_run/supervise_sweep.sh
set -u
cd "$(dirname "$0")/../.."

SWEEP=${SWEEP:?set SWEEP=<path to sweep script>}
MAX_RESTARTS=${MAX_RESTARTS:-40}
SETTLE=${SETTLE:-20}

count_done() { ls -d test_output/*/run_*/.done 2>/dev/null | wc -l | tr -d ' '; }

prev=$(count_done)
echo "[supervisor] $(date '+%H:%M:%S') starting $SWEEP with $prev runs already complete"

for attempt in $(seq 1 "$MAX_RESTARTS"); do
    bash "$SWEEP"
    rc=$?
    now=$(count_done)
    echo "[supervisor] $(date '+%H:%M:%S') attempt $attempt exited rc=$rc; done $prev -> $now"

    if [ "$rc" -eq 0 ]; then
        echo "[supervisor] sweep reported completion, stopping"
        exit 0
    fi
    if [ "$now" -le "$prev" ]; then
        echo "[supervisor] no progress since last attempt -- this is a real failure, not a kill." >&2
        echo "[supervisor] refusing to restart. Check the sweep log." >&2
        exit 1
    fi
    prev=$now
    sleep "$SETTLE"
done
echo "[supervisor] hit MAX_RESTARTS=$MAX_RESTARTS; stopping" >&2
exit 1
