#!/bin/bash
# E2 walk-forward campaign: progress check and result collection.
#
#   ./scripts/stock_run/collect_vol_cohorts.sh            # progress table
#   ./scripts/stock_run/collect_vol_cohorts.sh --bundle   # tar the finished runs
#
# Run from the repo root on Anvil. OUT_ROOT must match what anvil_vol_cold.sb derived from
# DATA_ROOT -- if you launched with a non-default DATA_ROOT, export the same OUT_ROOT here.
set -u
OUT_ROOT=${OUT_ROOT:-test_output/vol_qlib_vol_cohorts_full}
COHORT_YEARS=(2016 2017 2018 2019 2020)
RUNS=10
MAXG=${MAXG:-10000}
MIN_LOGGED=$(( MAXG * 9 / 10 ))

total=0; done_n=0; bad=0
printf '%-10s %-12s %s\n' "cohort" "done/total" "runs still missing"
for y in "${COHORT_YEARS[@]}"; do
    d=0; missing=""
    for r in $(seq 1 $RUNS); do
        total=$((total+1))
        o="$OUT_ROOT/cohort_$y/run_$r"
        if [ -f "$o/.done" ]; then
            d=$((d+1)); done_n=$((done_n+1))
            # A .done is necessary but not sufficient -- re-verify the two conditions the
            # launcher checked, in case a directory was touched or copied by hand.
            gb=$(ls "$o"/global_best_genome_*.bin 2>/dev/null | head -1)
            lg=$(wc -l < "$o/fitness_log.csv" 2>/dev/null || echo 0)
            if [ -z "$gb" ] || [ ! -s "$gb" ] || [ "$lg" -lt "$MIN_LOGGED" ]; then
                echo "  !! cohort $y run $r has .done but genome='${gb:-none}' logged=$lg" >&2
                bad=$((bad+1))
            fi
        else
            missing="$missing $r"
        fi
    done
    printf '%-10s %-12s %s\n' "$y" "$d/$RUNS" "${missing:-  -}"
done
echo
echo "TOTAL: $done_n/$total complete$([ $bad -gt 0 ] && echo "  ($bad FAILED verification)")"

if [ "${1:-}" = "--bundle" ]; then
    if [ "$done_n" -ne "$total" ]; then
        echo "refusing to bundle: only $done_n/$total runs are done" >&2
        echo "(re-run without --bundle to see which are missing, or resubmit those indices)" >&2
        exit 1
    fi
    [ $bad -gt 0 ] && { echo "refusing to bundle: $bad runs failed verification" >&2; exit 1; }
    tar czf vol_cohorts_genomes.tgz \
        $OUT_ROOT/cohort_*/run_*/global_best_genome_*.bin \
        $OUT_ROOT/cohort_*/run_*/fitness_log.csv \
        $OUT_ROOT/cohort_*/run_*/.config \
        $OUT_ROOT/cohort_*/run_*/.dataset
    echo
    ls -lh vol_cohorts_genomes.tgz
    echo "genomes bundled: $(tar tzf vol_cohorts_genomes.tgz | grep -c global_best_genome)"
    echo
    echo "now from your LOCAL machine:"
    echo "  scp $USER@anvil.rcac.purdue.edu:~/EXAMM-Extended/vol_cohorts_genomes.tgz ~/"
fi
