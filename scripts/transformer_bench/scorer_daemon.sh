#!/bin/bash
# Start (or report) the cell-scoring daemon that keeps .cell_scores.json warm for the monitor.
#
#   bash scripts/transformer_bench/scorer_daemon.sh [start|stop|status] [INTERVAL_SEC]
#
# WHY A SCRIPT RATHER THAN AN INLINE ssh ONE-LINER. The liveness check has to look for a process by
# its command line, and any one-liner that both CHECKS for "score_cells.py --loop" and CONTAINS the
# string to launch it will match its own shell -- pgrep -f sees the invoking command line too. That
# misfired twice here, reporting a running daemon when none existed. Inside a script the shell's own
# command line is just "bash scorer_daemon.sh", so the check cannot match itself, and a PID file
# makes it exact rather than pattern-based.
set -uo pipefail
REPO=${REPO:-/workspace/EXAMM-Extended}
ACTION=${1:-start}
INTERVAL=${2:-120}
PIDF=/workspace/scorer.pid
LOG=/workspace/scorer.log
cd "$REPO" || exit 1

alive() { [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null; }

case "$ACTION" in
  status)
    if alive; then echo "running pid $(cat $PIDF)"; tail -3 "$LOG" 2>/dev/null
    else echo "not running"; fi ;;
  stop)
    if alive; then kill "$(cat "$PIDF")" && rm -f "$PIDF" && echo "stopped"
    else echo "not running"; fi ;;
  start)
    if alive; then echo "already running pid $(cat $PIDF)"; exit 0; fi
    PY=$(cat external/.tfenv_path)/bin/python
    FIN_TOOLBOX_DIR=${FIN_TOOLBOX_DIR:-/workspace/Financial_toolbox} \
      setsid nohup "$PY" scripts/transformer_bench/score_cells.py --loop "$INTERVAL" \
      >> "$LOG" 2>&1 < /dev/null &
    echo $! > "$PIDF"
    sleep 4
    if alive; then echo "started pid $(cat $PIDF) (every ${INTERVAL}s)"
    else echo "FAILED TO START"; tail -5 "$LOG"; exit 1; fi ;;
  *) echo "usage: scorer_daemon.sh [start|stop|status] [INTERVAL_SEC]"; exit 2 ;;
esac
