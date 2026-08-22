#!/bin/bash
# EXA-GP vs EXAMM on the pooled stock panel -- LOCAL, sequential, one machine.
#
# WHY THIS EXISTS. Zimeng's suggestion: EXA-GP beats EXAMM on other datasets and has
# never been run on stock data. EXA-GP is not a separate program -- it is this same
# binary with the memory cells swapped for genetic-programming operators, selected
# purely by --possible_node_types (and, for EXA-GP-MIN, --weight_initialize gp). So the
# arms below differ from anvil_mse_pooled.sb in exactly those two flags and NOTHING else.
#
# COST, AND WHY THE OBVIOUS EXTRAPOLATION IS WRONG. The first 500 genomes take 44 s for
# EXA-GP-MIN on 8 threads, which invites the estimate "10,000 genomes in 15 minutes". That is
# wrong by an order of magnitude, because the networks GROW and per-genome cost grows with
# them. Measured on a real 10,000-genome run of this exact arm:
#     genome  500 ->   43 s elapsed,  7.6 genomes/s,   4 enabled edges
#     genome 2000 ->  428 s elapsed,  3.1 genomes/s,  11 enabled edges
#     genome 4600 -> 2339 s elapsed,  1.0 genomes/s,  55 enabled edges
# so a full run is ~2.8 h, and a 10-seed cell is a day rather than three hours. Budget from
# the decay curve, never from the first 500. A 10-seed cell at 500 genomes, by contrast, is
# about ten minutes -- see run_exagp_budget_sweep.sh for why that budget is scientifically
# interesting and not merely cheap.
#
# To scale up on Anvil instead, copy anvil_mse_pooled.sb and add the same two flags this
# script's ARM_ARGS sets -- nothing else about the protocol changes.
#
# WHY THERE IS A LOCAL EXAMM ARM AT ALL, when the paper already has EXAMM numbers.
# Table 1's EXAMM row was produced by examm_mpi on Anvil; this runs examm_mt on a Mac.
# Comparing an arm measured here against a row measured there confounds METHOD with
# BINARY AND PARALLELISM -- and this repo has already been burned once by exactly that
# (the 2023 cell / commit 57b375b0 saga documented in anvil_mse_combined.sb). Running
# EXAMM here too costs ~4 h and makes the EXA-GP comparison a controlled one. Treat the
# paper's Anvil row as the external reference, and the local EXAMM arm as the control.
#
# CONTRACT. Writes test_output/<arm>_<cohort>/run_<R>/ with the same .config and .done
# markers the Anvil scripts write, so the existing evaluation pipeline consumes the
# output with no changes:
#     bash scripts/stock_run/eval_ic_run.sh test_output/exagpmin_cohort_2021_aligned test
#
# BASH, NOT SH/ZSH -- deliberately. The 50 training filenames are passed as one
# word-split variable. zsh does not word-split, so under zsh EXAMM receives a single
# 50-path-long "filename", fails to open it, and dies with the misleading
# "Could not get headers from the CSV file. File potentially empty!". Same trap as
# results/results_tables.md warns about for the table regeneration scripts.
#
# Usage:
#   bash scripts/stock_run/run_exagp_local.sh <arm> [cohort] [seeds] [max_genomes]
#     arm    : exagpmin | exagp | examm
#     cohort : cohort_2021_aligned (default) | cohort_2020_aligned
#     seeds  : how many runs, seeded 1..N (default 10 -- the paper's ensemble size)
#
#   bash scripts/stock_run/run_exagp_local.sh exagpmin cohort_2021_aligned 10 10000
#   PANEL=set1 bash scripts/stock_run/run_exagp_local.sh exagpmin cohort_2020_aligned 10 500
set -u

ARM=${1:?usage: run_exagp_local.sh <exagpmin|exagp|examm> [cohort] [seeds] [max_genomes]}
COHORT=${2:-cohort_2021_aligned}
SEEDS=${3:-10}
MAXG=${4:-10000}
THREADS=${THREADS:-8}
BIN=${BIN:-build/multithreaded/examm_mt}
# PANEL selects which of the paper's two dataset families the cohort lives in. Empty (the
# default) is Dataset A, whose cohorts sit directly under walkforward/. set1..set4 are the
# twelve mid_highmid_price panels that make up the other twelve of the paper's fourteen
# cells; score_forecasting.data_dir() resolves them the same way.
PANEL=${PANEL:-}
if [ -n "$PANEL" ]; then
    DATA_DIR=datasets/walkforward/mid_highmid_price/$PANEL/$COHORT
else
    DATA_DIR=datasets/walkforward/$COHORT
fi

# ---- the arm definition: the ONLY thing that differs between arms --------------------
# EXAMM defaults are left implicit (no --possible_node_types) so the control is
# byte-identical to anvil_mse_pooled.sb rather than a hand-retyped copy of its defaults.
case "$ARM" in
    examm)
        ARM_ARGS=()
        ARM_DESC="EXAMM (memory cells, default node types, xavier init)" ;;
    exagp)
        ARM_ARGS=(--possible_node_types sigmoid tanh sum multiply inverse sin cos)
        ARM_DESC="EXA-GP (GP operators, xavier init)" ;;
    exagpmin)
        # EXA-GP-MIN needs BOTH flags. --weight_initialize gp is not cosmetic: it is what
        # makes generate_nn.cxx emit INPUT_NODE_GP/OUTPUT_NODE_GP instead of SIMPLE_NODE,
        # and what makes initialize_gp() seed the network as pure persistence -- weight 1.0
        # on the input whose parameter name equals the output's (here RET->RET), 0.0 on
        # every other input and every recurrent edge. On a returns panel that seed is a
        # random walk, which is the right null to evolve away from. Dropping the flag
        # silently gives you GP-shaped nodes with an ordinary dense xavier seed, i.e. a
        # different algorithm that still runs and still produces plausible numbers.
        ARM_ARGS=(--weight_initialize gp
                  --possible_node_types sigmoid_gp tanh_gp sum_gp multiply_gp inverse_gp sin_gp cos_gp)
        ARM_DESC="EXA-GP-MIN (GP operators, gp persistence init)" ;;
    *)
        echo "ERROR: unknown arm '$ARM' (want exagpmin, exagp or examm)" >&2; exit 1 ;;
esac

if [ ! -x "$BIN" ]; then echo "ERROR: $BIN not found -- build examm_mt first" >&2; exit 1; fi
if [ ! -d "$DATA_DIR" ]; then echo "ERROR: $DATA_DIR missing" >&2; exit 1; fi

# ---- the pooled panel: 50 files x 6 columns, one genome shared across all of them -----
TRAIN_FILES=$(ls "$DATA_DIR"/*_train.csv 2>/dev/null | tr '\n' ' ')
VAL_FILES=$(ls "$DATA_DIR"/*_val.csv   2>/dev/null | tr '\n' ' ')
N_TRAIN=$(echo $TRAIN_FILES | wc -w | tr -d ' ')
N_VAL=$(echo $VAL_FILES   | wc -w | tr -d ' ')
if [ "$N_TRAIN" -lt 50 ] || [ "$N_TRAIN" -ne "$N_VAL" ]; then
    echo "ERROR: $DATA_DIR has $N_TRAIN train / $N_VAL val files (expected equal, >=50)" >&2
    exit 1
fi
# 'aligned' means every stock shares one calendar. eval_ensemble_ic.py aligns stocks by ROW
# INDEX, so a cohort with ragged row counts would silently mis-date the cross-section.
NUNIQ=$(for f in $TRAIN_FILES; do wc -l < "$f"; done | sort -u | wc -l | tr -d ' ')
if [ "$NUNIQ" -ne 1 ]; then
    echo "ERROR: $DATA_DIR train files have $NUNIQ distinct row counts -- not aligned" >&2
    exit 1
fi

OUT_ROOT=test_output/${ARM}_${PANEL:+${PANEL}_}${COHORT}${REPLICATE_TAG:+_$REPLICATE_TAG}
mkdir -p "$OUT_ROOT"

BIN_SHA=$(shasum -a 256 "$BIN" 2>/dev/null | awk '{print $1}')
BIN_MTIME=$(date -r "$BIN" '+%Y-%m-%dT%H:%M:%S' 2>/dev/null)
GIT_HEAD=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)

echo "### $ARM_DESC"
echo "### cohort $COHORT | $N_TRAIN stocks pooled | seeds 1..$SEEDS | max_genomes $MAXG | threads $THREADS"
echo "### binary $BIN sha256 ${BIN_SHA:0:12} mtime $BIN_MTIME git $GIT_HEAD"
echo "### -> $OUT_ROOT"

CAMPAIGN_START=$(date +%s)
for RUN in $(seq 1 "$SEEDS"); do
    OUT=$OUT_ROOT/run_$RUN

    # Re-running the script must be a safe resume, not a silent no-op AND not a silent
    # overwrite of a finished ensemble member.
    if [ -f "$OUT/.done" ]; then
        echo "[run $RUN] already done, skipping"; continue
    fi

    case "$OUT" in */run_[0-9]*) ;; *) echo "ERROR: refusing rm -rf '$OUT'" >&2; exit 1;; esac
    rm -rf "$OUT"; mkdir -p "$OUT"

    # Provenance, in the key=value form eval_ic_run.sh parses. data_dir and target are
    # READ BACK by the evaluator; the rest is for attribution after the fact.
    printf 'data_dir=%s\nloss=mse\ntarget=RET\narm=%s\nconstruction=pooled\nrun=%s\nmax_genomes=%s\nseed=%s\nstocks=%s\nnode_types=%s\nbin=%s\nbin_sha256=%s\nbin_mtime=%s\ngit_head=%s\nhost=%s\n' \
        "$DATA_DIR" "$ARM" "$RUN" "$MAXG" "$RUN" "$N_TRAIN" "${ARM_ARGS[*]+${ARM_ARGS[*]}}" \
        "$BIN" "$BIN_SHA" "$BIN_MTIME" "$GIT_HEAD" "$(hostname -s)" > "$OUT/.config"

    # ${A[@]+"${A[@]}"}, not "${A[@]}": macOS ships bash 3.2, where expanding an EMPTY
    # array under `set -u` is an "unbound variable" fatal. The examm arm's ARM_ARGS is
    # empty by design, so the plain form would kill the control arm and only the control
    # arm -- the failure the whole comparison rests on.
    RUN_START=$(date +%s)
    "$BIN" --number_threads "$THREADS" \
        --training_filenames $TRAIN_FILES \
        --validation_filenames $VAL_FILES \
        --seed "$RUN" \
        --time_offset 1 \
        --input_parameter_names RET VOL_CHANGE BA_SPREAD ILLIQUIDITY sprtrn TURNOVER \
        --output_parameter_names RET \
        --loss mse \
        --number_islands 10 \
        --island_size 10 \
        --max_genomes "$MAXG" \
        --bp_iterations 10 \
        --num_mutations 1 \
        --normalize avg_std_dev \
        --extinction_event_generation_number 500 \
        --islands_to_exterminate 1 \
        --repopulation_method bestGenome \
        ${ARM_ARGS[@]+"${ARM_ARGS[@]}"} \
        --output_directory "$OUT" \
        --save_genome_option none \
        --std_message_level INFO \
        --file_message_level NONE > "$OUT/train.log" 2>&1
    RC=$?
    ELAPSED=$(( $(date +%s) - RUN_START ))

    GB=$(ls "$OUT"/global_best_genome_*.bin 2>/dev/null \
         | sed 's/.*_\([0-9]*\)\.bin/\1 &/' | sort -rn | head -1 | cut -d' ' -f2-)

    # COUNT GENOMES EVALUATED, NOT LINES IN THE LOG. EXAMM writes a fitness_log line per NEW
    # BEST, and how often a search improves varies by arm -- an EXA-GP-MIN run that converges
    # early logs far fewer lines than a still-improving EXAMM run at the same genome count.
    # A line-count guard would therefore fail exactly the arms that converge best. Column 1
    # of fitness_log.csv IS the inserted-genome counter, so read the last row instead: that
    # measures work done, and still catches a truncated run.
    LAST_G=$(awk -F, '$1+0>0 {n=$1+0} END{print n+0}' "$OUT/fitness_log.csv" 2>/dev/null || echo 0)
    MIN_G=$(( MAXG * 9 / 10 ))

    if [ $RC -eq 0 ] && [ -n "$GB" ] && [ -s "$GB" ] && [ "$LAST_G" -ge "$MIN_G" ]; then
        touch "$OUT/.done"
        echo "[run $RUN] FINISHED in ${ELAPSED}s  genomes=$LAST_G  $(basename "$GB")"
    else
        echo "[run $RUN] FAILED rc=$RC genome='${GB:-none}' genomes=$LAST_G (need >=$MIN_G) after ${ELAPSED}s" >&2
        echo "          see $OUT/train.log" >&2
        exit 1
    fi
done

echo "### campaign complete in $(( ( $(date +%s) - CAMPAIGN_START ) / 60 )) min -> $OUT_ROOT"
echo "### next: bash scripts/stock_run/eval_ic_run.sh $OUT_ROOT test"
