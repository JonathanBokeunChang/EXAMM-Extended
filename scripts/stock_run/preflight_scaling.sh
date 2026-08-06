#!/bin/bash
# Preflight for the EXAMM scaling study. Run this on ANVIL, from the repo root, BEFORE
# submitting anything. Every check here corresponds to a way the study can fail after the
# jobs are already queued -- at which point you have burned SU and lost hours of queue time.
#
#   bash scripts/stock_run/preflight_scaling.sh
#
# Exits non-zero if anything is wrong. Prints what to do about it.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

FAIL=0
ok()   { echo "  [ok]   $*"; }
bad()  { echo "  [FAIL] $*"; FAIL=1; }
warn() { echo "  [warn] $*"; }

DATA_DIR=${DATA_DIR:-datasets/walkforward/cohort_2021_aligned}
BIN=build/mpi/examm_mpi
ACCT=$(grep -m1 -oE '^#SBATCH --account=[A-Za-z0-9_]+' scripts/stock_run/anvil_examm_scaling.sb | cut -d= -f2)
PART=$(grep -m1 -oE '^#SBATCH --partition=[A-Za-z0-9_]+' scripts/stock_run/anvil_examm_scaling.sb | cut -d= -f2)

echo "=== 1. environment ==="
command -v sbatch >/dev/null && ok "sbatch found ($(sbatch --version 2>/dev/null | head -1))" \
                             || bad "sbatch not found -- are you on an Anvil login node?"
command -v srun   >/dev/null && ok "srun found"   || bad "srun not found"
command -v python3 >/dev/null && ok "python3 found ($(python3 -V 2>&1))" \
                             || bad "python3 not found -- the .sb parses fitness_log.csv with it"
command -v bc >/dev/null && ok "bc found" || bad "bc not found -- the .sb computes timings with it"

echo
echo "=== 2. the binary (the trap: a macOS build rsync'd from a laptop) ==="
if [ ! -x "$BIN" ]; then
  bad "$BIN missing or not executable"
  echo "         build it:  module load gcc/11.2.0 openmpi/4.1.6 && cmake --build build -j8"
else
  ok "$BIN present"
  # A macOS Mach-O binary is executable on Linux by the [ -x ] test but dies at exec. The repo
  # contains a macOS build, so an rsync of the whole tree lands one here and the [ -x ] guard
  # inside the .sb would happily pass it to srun.
  ft=$(file -b "$BIN" 2>/dev/null)
  case "$ft" in
    *ELF*x86-64*)  ok "binary is a Linux x86-64 ELF" ;;
    *Mach-O*)      bad "binary is a macOS Mach-O -- this is the laptop build, it will not run"
                   echo "         rm -rf build && mkdir build && cd build && cmake .. && make -j8" ;;
    *)             warn "unrecognised binary type: $ft" ;;
  esac
  # Proof it actually launches under MPI, which no static check can establish.
  if command -v srun >/dev/null; then
    if timeout 90 srun -n 2 --time=00:02:00 "$BIN" --help >/dev/null 2>&1; then
      ok "binary launches under srun -n 2"
    else
      warn "could not launch under srun here (login nodes often disallow it)"
      warn "  -> the single-point smoke in the walkthrough is what actually proves this"
    fi
  fi
fi

echo
echo "=== 3. data ==="
if [ ! -d "$DATA_DIR" ]; then
  bad "$DATA_DIR missing"
else
  nt=$(ls "$DATA_DIR"/*_train.csv 2>/dev/null | wc -l | tr -d ' ')
  nv=$(ls "$DATA_DIR"/*_val.csv   2>/dev/null | wc -l | tr -d ' ')
  [ "$nt" -ge 50 ] && [ "$nt" -eq "$nv" ] && ok "$DATA_DIR: $nt train / $nv val files" \
                                          || bad "$DATA_DIR: $nt train / $nv val (need equal, >=50)"
  # Equal-length files are required by the pooled loader; a partial transfer is the usual cause
  # and it fails deep inside EXAMM rather than here.
  nu=$(for f in "$DATA_DIR"/*_train.csv; do wc -l < "$f"; done | sort -u | wc -l | tr -d ' ')
  [ "$nu" -eq 1 ] && ok "train files are equal-length (aligned)" \
                  || bad "train files are NOT equal-length -- rebuild with align_cohort.py"
fi

echo
echo "=== 4. SLURM account and partition ==="
if command -v sacctmgr >/dev/null 2>&1; then
  if sacctmgr -nP show assoc user="$USER" format=Account 2>/dev/null | grep -qx "$ACCT"; then
    ok "account $ACCT is available to $USER"
  else
    warn "could not confirm account $ACCT (sacctmgr returned nothing) -- verify manually"
  fi
fi
if command -v sinfo >/dev/null 2>&1; then
  if sinfo -h -p "$PART" -o "%P" 2>/dev/null | grep -q .; then
    cores=$(sinfo -h -p "$PART" -o "%c" 2>/dev/null | head -1)
    ok "partition $PART exists (${cores:-?} cores/node)"
    if [ -n "${cores:-}" ] && [ "$cores" -lt 64 ] 2>/dev/null; then
      bad "partition has only $cores cores/node -- the 64-rank arm cannot fit on one node"
    fi
  else
    bad "partition $PART not found -- check the #SBATCH --partition line"
  fi
fi

echo
echo "=== 5. output paths writable ==="
for d in slurm_logs results/scaling test_output/scaling; do
  mkdir -p "$d" 2>/dev/null && [ -w "$d" ] && ok "$d writable" || bad "$d not writable"
done

echo
echo "=== 6. no stale results that would be silently mixed in ==="
if [ -s results/scaling/timings.csv ]; then
  n=$(( $(wc -l < results/scaling/timings.csv) - 1 ))
  warn "results/scaling/timings.csv already has $n row(s)"
  warn "  the .sb APPENDS -- old rows from a different MAXG or dataset would be averaged in"
  warn "  -> mv results/scaling/timings.csv results/scaling/timings.\$(date +%F).csv"
else
  ok "no stale timings.csv"
fi

echo
if [ "$FAIL" -eq 0 ]; then
  echo "PREFLIGHT PASSED -- safe to run the single-point smoke next."
else
  echo "PREFLIGHT FAILED -- fix the [FAIL] items above before submitting."
fi
exit "$FAIL"
