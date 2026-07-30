#!/bin/bash
# Submit EXAMM hyperparameter arms across the walk-forward cohorts.
#
#   bash scripts/stock_run/submit_hpsweep.sh --dry-run    # print, submit nothing
#   bash scripts/stock_run/submit_hpsweep.sh              # submit
#   ARMS="seq50" bash scripts/stock_run/submit_hpsweep.sh                 # one arm
#   COHORTS="cohort_2019_aligned" bash scripts/stock_run/submit_hpsweep.sh  # one cohort
#
# Default: 2 arms x 3 cohorts = 6 array jobs = 60 runs, on the CPU allocation (cis251123), so this
# does not wait on GPU approval.
#
# PREREQUISITE: cohort_2019_aligned may not exist on the cluster. Build it once, it takes seconds:
#   python3 scripts/stock_run/align_cohort.py \
#       --in datasets/walkforward/cohort_2019 --out datasets/walkforward/cohort_2019_aligned
# The preflight below fails loudly if it is missing rather than silently submitting 2 of 3 cohorts.
#
# COHORTS AND THEIR WALK-FORWARD TRADING YEAR
#   cohort_2019  train ->2019, val 2020, test 2021-2023 (753 rows)  -> trade 2021
#   cohort_2020  train ->2020, val 2021, test 2022-2023 (501 rows)  -> trade 2022
#   cohort_2021  train ->2021, val 2022, test 2023      (250 rows)  -> trade 2023
# Each cohort's test window extends past its walk-forward year; score with
# compare_to_examm.py --year <trading year> so the model is never judged on a stale horizon.
#
# THE ARMS  (baseline for all: islands 10x10, num_mutations 1, extinction 500, no slicing,
#            --normalize avg_std_dev, 10,000 genomes, bp_iterations 10 -- i.e. anvil_ic.sb)
#   seq50   --train/validation_sequence_length 50    one change, real mechanism
#   seq25   --train/validation_sequence_length 25    one change, real mechanism
#   search  islands 20, mutations 2, extinction 200, slicing 50    four changes at once
#
# Slicing is the arm worth watching: compute-neutral, but weight updates per epoch go from 50 to
# 3,250 (len 50) or 6,550 (len 25), and it brings EXAMM's context length near the transformers'
# L=20 window. `search` bundles four knobs and cannot attribute a result to any one of them.
#
# CONTROLS: cohorts 2020 and 2021 already have a 10-seed baseline in
# test_output/mse_cohort_{2020,2021}_aligned -- do NOT re-run those. Cohort 2019 has no baseline;
# add ARMS="baseline" to generate one, or its arms have nothing to be compared against.
#
# EXPECTATION, stated before the results exist: seed-to-seed sd of test IC on this venue is 0.0026
# and validation fitness predicts test IC at r = +0.065. With 10 seeds SEM is 0.0008, so arm-to-arm
# differences below ~0.003 are not resolvable. A null result is the expected result here.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
JOB=scripts/stock_run/anvil_examm_hpsweep.sb
DRY=0; [ "${1:-}" = "--dry-run" ] && DRY=1

read -r -a ARMS    <<< "${ARMS:-seq50 search}"
read -r -a COHORTS <<< "${COHORTS:-cohort_2019_aligned cohort_2020_aligned cohort_2021_aligned}"

# ---- preflight; every one of these has bitten this project at least once
[ -f "$JOB" ] || { echo "ERROR: $JOB missing -- git pull on Anvil first" >&2; exit 1; }
[ -x build/mpi/examm_mpi ] || { echo "ERROR: build/mpi/examm_mpi missing -- build it first" >&2; exit 1; }
command -v sbatch >/dev/null || { echo "ERROR: sbatch not found -- are you on Anvil?" >&2; exit 1; }
mkdir -p slurm_logs
for C in "${COHORTS[@]}"; do
  d=datasets/walkforward/$C
  if [ ! -d "$d" ]; then
    echo "ERROR: $d missing." >&2
    [ "$C" = "cohort_2019_aligned" ] && echo "  Build it: python3 scripts/stock_run/align_cohort.py --in datasets/walkforward/cohort_2019 --out $d" >&2
    exit 1
  fi
  n=$(ls "$d"/*_train.csv 2>/dev/null | wc -l | tr -d ' ')
  [ "$n" -eq 50 ] || { echo "ERROR: $d has $n train files, expected 50" >&2; exit 1; }
done
echo "preflight OK -- ${#ARMS[@]} arm(s) x ${#COHORTS[@]} cohort(s), 50 stocks each"
echo

n=0
for ARM in "${ARMS[@]}"; do
  for C in "${COHORTS[@]}"; do
    out="test_output/hp_${ARM}_${C}"
    done_n=$(ls -d "$out"/run_*/.done 2>/dev/null | wc -l | tr -d ' ')
    if [ "$done_n" -eq 10 ]; then echo "  SKIP   $ARM / $C  (10/10 already complete)"; continue; fi
    [ "$done_n" -gt 0 ] && echo "  note   $ARM / $C has $done_n/10 done; the job skips those"
    cmd=(sbatch --export="ALL,ARM=$ARM,COHORT=$C" --array=0-9
         --job-name="hp_${ARM}_${C#cohort_}" "$JOB")
    if [ "$DRY" = "1" ]; then
      echo "  DRY    ${cmd[*]}"
    else
      id=$("${cmd[@]}" | grep -oE '[0-9]+$')
      echo "  SUBMIT $ARM / $C -> job ${id:-?}  (-> $out)"
    fi
    n=$((n+1))
  done
done

echo
echo "$n array job(s) $([ "$DRY" = 1 ] && echo 'would be submitted' || echo submitted), 10 runs each = $((n*10)) runs."
[ "$DRY" = 1 ] || cat <<'EOF'

Watch:     squeue -u $USER
Progress:  ls -d test_output/hp_*/run_*/.done | wc -l
Logs:      tail slurm_logs/hp_*.out

Score each arm on its OWN walk-forward year (2019->2021, 2020->2022, 2021->2023):
  python3 scripts/transformer_bench/compare_to_examm.py \
      --cohort cohort_2019_aligned --year 2021 \
      --examm-root test_output/hp_seq50_cohort_2019_aligned \
      --tf-root results/transformer_bench/cohort_2019_aligned
EOF
