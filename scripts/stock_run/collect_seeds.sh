#!/bin/sh
# Collect per-run verified global bests from a finished wave into a seeds dir
# for the NEXT warm wave: seeds/<STOCK>_run<R>.bin.
#
# Usage: sh scripts/stock_run/collect_seeds.sh <WAVE_OUT_DIR> <SEEDS_DIR>
#   e.g. sh scripts/stock_run/collect_seeds.sh test_output/warm_2016 seeds/warm_2016
#
# FAILS LOUD (nonzero exit, nothing partial trusted) if any expected run is
# missing, unfinished (.done absent -- which also means its per-run genome
# verification never passed), or has no saved global best. A warm chain must
# never silently continue from a hole.

set -u
WAVE=$1
SEEDS=$2

if [ ! -d "$WAVE" ]; then
    echo "ERROR: wave dir $WAVE not found" >&2
    exit 1
fi
mkdir -p "$SEEDS"

missing=0
collected=0
for stock_dir in "$WAVE"/*/; do
    S=$(basename "$stock_dir")
    for run_dir in "$stock_dir"run_*/; do
        R=${run_dir%/}
        R=${R##*run_}
        if [ ! -f "$run_dir/.done" ]; then
            echo "MISSING/UNVERIFIED: $S run_$R (no .done)" >&2
            missing=$((missing + 1))
            continue
        fi
        # lowest recorded-MSE global best in the run dir (same rule as
        # collect_best_genomes.sh / verify_genome.sh)
        best_txt=$(for f in "$run_dir"global_best_genome_*.txt; do
            [ -f "$f" ] || continue
            mse=$(grep -o 'best_validation_mse: [0-9.eE+-]*' "$f" | awk '{print $2}')
            [ -n "$mse" ] && echo "$mse $f"
        done | sort -n | head -1 | awk '{print $2}')
        if [ -z "$best_txt" ]; then
            echo "MISSING: $S run_$R has .done but no global_best_genome_*.txt" >&2
            missing=$((missing + 1))
            continue
        fi
        best_bin=${best_txt%.txt}.bin
        if [ ! -f "$best_bin" ]; then
            echo "MISSING: $best_bin" >&2
            missing=$((missing + 1))
            continue
        fi
        cp "$best_bin" "$SEEDS/${S}_run${R}.bin"
        collected=$((collected + 1))
    done
done

echo "collected $collected seeds -> $SEEDS"
if [ $missing -gt 0 ]; then
    echo "ERROR: $missing runs missing/unverified -- fix the wave before chaining" >&2
    exit 1
fi
# a stock whose directory never got created would not appear in the loop at all,
# so also require the exact expected seed count (50 stocks x 10 runs by default)
EXPECT=${EXPECT:-500}
if [ "$collected" -ne "$EXPECT" ]; then
    echo "ERROR: collected $collected seeds but expected $EXPECT -- a stock is missing entirely" >&2
    exit 1
fi
