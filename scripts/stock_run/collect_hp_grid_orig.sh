#!/bin/sh
# Collect each run's global-best genome from the island-count x repopulation-frequency
# grid (orig universe) into one flat folder + index, before evaluation -- same purpose
# as collect_best_genomes.sh, adapted for this campaign's POOLED layout
# (test_output/hp_<tag>_<cohort>/run_<R>/, one genome per run trained jointly across all
# 50 stocks, not one genome per stock).
#
# Covers the FULL 3x3 grid: the 3 cells that already existed (baseline, k_isl20,
# k_ext250) plus the 6 this campaign added (k_isl10_ext1000, k_isl20_ext250,
# k_isl20_ext1000, k_isl40_ext250, k_isl40_ext500, k_isl40_ext1000) -- one consolidated
# snapshot of all 9 cells x 2 cohorts x 10 runs = 180 genomes, so evaluation and any
# later re-check both read from the same durable place instead of the transient
# run_R/ dirs (which also hold every intermediate rnn_genome_*.bin, not just the winner).
#
# Usage: sh scripts/stock_run/collect_hp_grid_orig.sh [DEST]
#   DEST defaults to test_output/hp_grid_orig_best
#
# Runs without .done, or completed runs missing a genome, are warned and skipped --
# never silently dropped.

REPO=$(cd "$(dirname "$0")/../.." && pwd)
DEST=${1:-$REPO/test_output/hp_grid_orig_best}

TAGS="baseline k_isl20 k_ext250 k_isl10_ext1000 k_isl20_ext250 k_isl20_ext1000 k_isl40_ext250 k_isl40_ext500 k_isl40_ext1000"
COHORTS="cohort_2020_aligned cohort_2021_aligned"

mkdir -p "$DEST"
INDEX="$DEST/index.csv"
echo "tag,cohort,run,generation_id,best_validation_mse" > "$INDEX"

collected=0
missing_genome=0
not_done=0
missing_dir=0

for tag in $TAGS; do
  for cohort in $COHORTS; do
    OUT="$REPO/test_output/hp_${tag}_${cohort}"
    if [ ! -d "$OUT" ]; then
      echo "WARN: $OUT not found, skipping" >&2
      missing_dir=$((missing_dir + 1))
      continue
    fi
    for run_dir in "$OUT"/run_*/; do
      [ -d "$run_dir" ] || continue
      run_name=$(basename "$run_dir")
      R=${run_name#run_}

      if [ ! -f "$run_dir/.done" ]; then
        echo "WARN: $tag/$cohort run $R has no .done (incomplete), skipping" >&2
        not_done=$((not_done + 1))
        continue
      fi

      # usually exactly one global_best per run; if more, take the lowest recorded MSE
      best_txt=$(for f in "$run_dir"global_best_genome_*.txt; do
        [ -f "$f" ] || continue
        mse=$(grep -o 'best_validation_mse: [0-9.eE+-]*' "$f" | awk '{print $2}')
        [ -n "$mse" ] && echo "$mse $f"
      done | sort -n | head -1 | awk '{print $2}')

      if [ -z "$best_txt" ]; then
        echo "WARN: $tag/$cohort run $R is done but has no global_best_genome_*.txt, skipping" >&2
        missing_genome=$((missing_genome + 1))
        continue
      fi

      base=${best_txt%.txt}
      if [ ! -f "$base.bin" ] || [ ! -f "$base.gv" ]; then
        echo "WARN: $tag/$cohort run $R missing .bin or .gv alongside $(basename "$base"), skipping" >&2
        missing_genome=$((missing_genome + 1))
        continue
      fi

      gen_id=$(basename "$base" | sed 's/global_best_genome_//')
      mse=$(grep -o 'best_validation_mse: [0-9.eE+-]*' "$best_txt" | awk '{print $2}')

      cp "$base.bin" "$DEST/${tag}_${cohort}_run${R}.bin"
      cp "$base.gv"  "$DEST/${tag}_${cohort}_run${R}.gv"
      cp "$best_txt" "$DEST/${tag}_${cohort}_run${R}.txt"
      echo "${tag},${cohort},${R},${gen_id},${mse}" >> "$INDEX"
      collected=$((collected + 1))
    done
  done
done

echo "###-------------------###"
echo "collected: $collected genomes -> $DEST"
echo "index:     $INDEX"
[ "$missing_dir" -gt 0 ] && echo "WARNING: $missing_dir cell/cohort dir(s) not found on disk"
[ "$not_done" -gt 0 ] && echo "WARNING: $not_done run dir(s) had no .done (incomplete)"
[ "$missing_genome" -gt 0 ] && echo "WARNING: $missing_genome completed run(s) had no usable genome"
if [ "$collected" -ne 180 ]; then
  echo "NOTE: expected 180 (9 cells x 2 cohorts x 10 runs), got $collected -- check warnings above"
fi
