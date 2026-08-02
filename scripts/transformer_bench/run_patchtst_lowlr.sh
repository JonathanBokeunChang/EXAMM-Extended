#!/bin/bash
# PatchTST at a 10x LOWER learning rate, to answer one specific reviewer question.
#
# PatchTST and DeformTime both post NEGATIVE test IC in both cohorts (PatchTST -0.0065
# / -0.0115), and the training logs show why: the best validation epoch is a median of 2
# for PatchTST and 1 in 20/20 runs for DeformTime, versus 3-4 for Crossformer and DLinear.
# Both failing models overfit before they learn the short-horizon reversal structure that
# is the only exploitable effect in this panel, so early stopping returns a barely-trained
# checkpoint whose cross-sectional loading on RET[t] is on the MOMENTUM side (+0.010 for
# PatchTST, +0.084 for DeformTime) while every model that works loads negative (-0.37 to
# -0.57). Being on the wrong side of the one real effect is why they lose 12-25% rather
# than the ~0% random predictions would produce.
#
# That is also the signature of a learning rate too high for the architecture, not proof
# the architecture cannot do the task -- and a reviewer will say so. Every baseline was
# run on ONE shared protocol (lr 5e-4, cosine, 30 epochs, patience 5, batch 256) so that
# nothing was hand-tuned, which is the defensible position; but "we also tried a 10x
# lower LR and it still failed" is a far stronger sentence than "we used the same LR for
# everyone."
#
# SCOPE: PatchTST only, lr 5e-5, cohort_2020_aligned, 10 seeds -- matched seed-for-seed
# against the 10 runs already on disk so the comparison is like-for-like. Everything else
# is byte-identical to the original protocol. DeformTime is deliberately NOT included;
# PatchTST is the milder failure and therefore the easier one to rescue, so if a lower LR
# cannot move PatchTST it will not move DeformTime either.
#
#   bash scripts/transformer_bench/run_patchtst_lowlr.sh
#
# READING THE RESULT: the diagnostic is the BEST VALIDATION EPOCH, not the loss. If it
# stays at 1-2, the model still overfits immediately and the LR was not the problem. If
# it moves out to 5+ and IC turns positive, the original number was an LR artifact and
# the paper must say so.
#
# Finished seeds are skipped via .done, so this is safe to re-run after an interrupt.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TF="$REPO/external/DeformTime"
PY="$REPO/external/tfenv/bin/python"
MODEL=PatchTST
COHORT=${COHORT:-cohort_2020_aligned}
LR=${LR:-0.00005}
SEEDS=${SEEDS:-"1 2 3 4 5 6 7 8 9 10"}

[ -x "$PY" ] || { echo "ERROR: $PY missing" >&2; exit 1; }
[ -f "$TF/run.py" ] || { echo "ERROR: $TF/run.py missing" >&2; exit 1; }
DATA="$REPO/datasets/walkforward/$COHORT"
[ -d "$DATA" ] || { echo "ERROR: $DATA missing" >&2; exit 1; }

# lr goes in the tag as well as the path: run.py encodes lr into its own results dir name,
# so two LRs cannot collide, and the tag keeps that explicit on our side too.
LRTAG=$(printf 'lr%s' "$LR" | tr -d '.')
n_ok=0; n_skip=0; n_fail=0
for SEED in $SEEDS; do
  TAG="seed_${SEED}"
  OUT="$REPO/results/transformer_bench/lowlr/$COHORT/$MODEL/$LRTAG/$TAG"
  if [ -f "$OUT/.done" ]; then n_skip=$((n_skip+1)); continue; fi
  mkdir -p "$OUT"
  MID="${MODEL}_L20_${COHORT}_${LRTAG}_${TAG}"
  cd "$TF"
  # EXAMM_TF_USE_MPS routes to the Apple GPU (run.py:85 -> src/exp/exp_basic.py:31).
  # Pure CPU here is ~10 min/epoch, which puts 10 seeds past 13 hours; MPS makes this
  # an overnight-free job. Set EXAMM_TF_USE_MPS=0 to force CPU.
  RUN_SEED=$SEED STOCK_META_DIR="$OUT" STOCK_NORM_SCOPE=train_val \
  EXAMM_TF_USE_MPS=${EXAMM_TF_USE_MPS:-1} PYTHONUNBUFFERED=1 \
  "$PY" run.py \
    --is_training 1 --model "$MODEL" --model_id "$MID" \
    --data stock_pooled --root_path "$DATA" --data_path unused \
    --features MS --target RET --freq d \
    --seq_len 20 --label_len 0 --pred_len 1 \
    --enc_in 6 --dec_in 6 --c_out 1 \
    --d_model 64 --n_heads 4 --d_ff 128 --dropout 0.1 \
    --e_layers 2 --d_layers 1 --factor 10 \
    --batch_size 256 --learning_rate "$LR" --train_epochs 30 --patience 5 \
    --loss MSE --lradj cosine --itr 1 --des pooled --num_workers 0 \
    --use_gpu True --gpu 0 --checkpoints "$OUT/checkpoints" > "$OUT/train.log" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "FAILED $TAG rc=$rc (see $OUT/train.log)" >&2
    tail -5 "$OUT/train.log" >&2
    n_fail=$((n_fail+1)); continue
  fi
  # Anchor the glob on the full model_id -- an unanchored seed_1 prefix also matches
  # seed_10 (a verified failure mode in anvil_transformer.sb).
  found=0
  for d in results/"${MID}_${MODEL}_stock_pooled_"*; do
    [ -d "$d" ] || continue
    cp -r "$d" "$OUT/" && found=1
  done
  if [ "$found" -ne 1 ]; then
    echo "FAILED $TAG: no results dir matching $MID" >&2; n_fail=$((n_fail+1)); continue
  fi
  "$PY" "$REPO/scripts/transformer_bench/collect_predictions.py" --run-dir "$OUT" \
    >> "$OUT/train.log" 2>&1 || { echo "FAILED collect $TAG" >&2; n_fail=$((n_fail+1)); continue; }
  touch "$OUT/.done"
  n_ok=$((n_ok+1))
  BEST=$(grep -c "Validation loss decreased" "$OUT/train.log")
  echo "OK $TAG  (val improved $BEST times)"
done
echo "### PatchTST $COHORT lr=$LR: $n_ok ok, $n_skip skipped, $n_fail failed"
