#!/bin/bash
# How much do the untuned hyperparameters cost us on PurpleAir?
#
# Three sites deliberately chosen to span the outcome range rather than to flatter a
# setting: az is a clear win for EXAMM (-30%), co is a clear loss (+33%), wi is the
# catastrophic failure (+91%). A hyperparameter that only helps the site that already
# works is not worth adopting.
#
# 5 repeats per configuration because a single run on this dataset spans 2x. That is
# fewer than the 10 used for the headline numbers -- enough to rank configurations,
# not enough to quote as a result.
set -u
cd /Users/jonathanchang/EXAMM-Extended/.claude/worktrees/exagp-stock

REPEATS=${REPEATS:-5}
MAXG=${MAXG:-200}
for CFG in "bp2_i4:2:4:5" "bp10_i4:10:4:5" "bp10_i10:10:10:10"; do
    NAME=${CFG%%:*}; REST=${CFG#*:}
    BP=${REST%%:*}; REST=${REST#*:}
    ISL=${REST%%:*}; ISZ=${REST#*:}
    for SITE in az co wi; do
        for R in $(seq 1 "$REPEATS"); do
            BP_ITER=$BP ISLANDS=$ISL ISLAND_SIZE=$ISZ ARM=examm \
                bash scripts/purpleair/purpleair_run.sh "$SITE" "$MAXG" 2 "hp_${NAME}_examm_r${R}" \
                >/dev/null 2>&1 || echo "FAILED $NAME/$SITE/r$R"
        done
        python3 scripts/purpleair/score_purpleair_ensemble.py \
            --tag "hp_${NAME}" --arm examm --site "$SITE" --repeats "$REPEATS" \
            | sed "s/^/[$NAME] /"
    done
done
