#!/bin/bash
# Submit the EXAMM strong-scaling study: one sbatch per rank count, each an array of
# repeats. SLURM cannot vary --ntasks across array indices, which is why this wrapper
# exists rather than a single array job.
#
#   bash scripts/stock_run/submit_examm_scaling.sh                 # defaults below
#   MAXG=2000 REPS=5 bash scripts/stock_run/submit_examm_scaling.sh
#   CORES="8 16 32 64 128" bash scripts/stock_run/submit_examm_scaling.sh
#   DRY=1 bash scripts/stock_run/submit_examm_scaling.sh           # print, do not submit
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"

# 128 IS INCLUDED DELIBERATELY, and it is the cheapest point in the grid (~91 SU, ~14 min/run).
# EXAMM's master (mpi/examm_mpi.cxx:143) is a single serial loop: MPI_Probe, generate_genome,
# insert. It caps NOTHING -- every worker that requests work gets a genome immediately -- so the
# Amdahl knee sits at roughly t_worker/t_master workers. Measured from a real run, a genome takes
# ~10.0 s to evaluate, so the master would have to spend >160 ms per genome to bottleneck at 63
# workers and >79 ms at 127. Graph mutation plus insertion is single-digit milliseconds. The
# prediction is therefore near-linear scaling through 64, i.e. a table of ~100% efficiency that
# never locates the knee. 128 ranks (one full Anvil node) is the only point in reach that might.
CORES=${CORES:-"8 16 32 64 128"}
REPS=${REPS:-3}
# 10000 = the paper's production budget. See the .sb header: a smaller budget lets constant
# startup cost contaminate the fast arms (27% of the 64-rank arm at MAXG=1000, 4% at 10000).
MAXG=${MAXG:-10000}
DATA_DIR=${DATA_DIR:-datasets/walkforward/cohort_2021_aligned}
DRY=${DRY:-0}

# REPEATS ARE NOT OPTIONAL. A single timing per rank count cannot be distinguished from a
# slow node or a stray page-cache miss, and the whole claim here is a ratio between two
# timings. Three is the minimum that gives any spread at all; five is better if SU allows.
[ "$REPS" -ge 3 ] || echo "WARNING: REPS=$REPS gives no usable variance estimate" >&2

mkdir -p slurm_logs results/scaling test_output/scaling

echo "=== EXAMM strong-scaling submission ==="
echo "  rank counts : $CORES"
echo "  repeats     : $REPS"
echo "  max_genomes : $MAXG   (per run; the paper's production value is 10000)"
echo "  data        : $DATA_DIR"
echo

# Cost estimate. wholenode charges all 128 cores whatever we use, so SU is driven by
# wall-clock alone. Strong scaling means the small-rank arms dominate: an 8-rank run is
# ~8x the wall-clock of a 64-rank one, so it is ~8x the SU despite using fewer cores.
# Stating this up front because the intuition usually runs the other way.
echo "  COST NOTE: wholenode bills 128 cores regardless of --ntasks, so SU is pure wall-clock."
echo "  The 8-RANK arm is the expensive one (longest wall-clock), not the 64-rank arm."
case "$MAXG" in
  10000) echo "  Estimated: ~6.8 node-h total = ~875 SU. 8-rank arm ~72 min per repeat." ;;
   2000) echo "  Estimated: ~1.4 node-h total = ~175 SU. Startup is 39% of the 128-rank arm." ;;
   1000) echo "  Estimated: ~0.7 node-h total =  ~90 SU. WARNING: startup dominates the" ;;
esac
[ "$MAXG" -lt 2000 ] 2>/dev/null && \
  echo "  64/128-rank arms at this budget -- they will understate speedup badly."
echo "  (anchored on the paper's OWN mse runs: 10,031 genomes at 32 ranks = ~16 min)"
echo "  NOTE the anchor is objective-specific. An earlier version of this estimate used an"
echo "  ic_pearson run (42-80 min) and was ~3.4x too high: --loss ic computes a"
echo "  cross-sectional statistic per batch, --loss mse does not. Do not re-anchor this on"
echo "  a run from a different objective."
echo

for c in $CORES; do
  if [ "$c" -lt 2 ]; then echo "  skip $c (EXAMM needs >=2 ranks: 1 master + workers)"; continue; fi
  if [ "$c" -gt 128 ]; then echo "  skip $c (>128 would span nodes; see the .sb header)"; continue; fi
  cmd=(sbatch --ntasks="$c" --array=1-"$REPS"
       --export=ALL,MAXG="$MAXG",DATA_DIR="$DATA_DIR"
       --job-name="examm_c$c"
       scripts/stock_run/anvil_examm_scaling.sb)
  if [ "$DRY" = "1" ]; then
    echo "  DRY: ${cmd[*]}"
  else
    out=$("${cmd[@]}" 2>&1) && echo "  submitted ranks=$c reps=$REPS  -> $out" \
                            || echo "  FAILED ranks=$c: $out" >&2
  fi
done

echo
echo "When all jobs finish:"
echo "  python3 scripts/stock_run/scaling_report.py"
