#!/bin/bash
# DLinear on the mid_highmid_20yr_portfolios replication universe: 4 sets x 3 cohorts
# x 10 seeds = 120 runs. Local CPU only (DLinear has no attention; ~1.3 s/epoch here),
# so no Anvil/GPU needed -- same as run_dlinear_local.sh, just parameterized by set.
#
#   bash scripts/transformer_bench/run_dlinear_mid_highmid.sh
#
# Protocol is byte-identical to the original-universe DLinear runs (L=20, h=1, pooled
# 50 stocks, MSE, cosine lradj, 30 epochs, patience 5) so the two are comparable.
# Finished seeds are skipped via .done, so this is safe to re-run after an interrupt.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TF="$REPO/external/DeformTime"
PY="$REPO/external/tfenv/bin/python"
MODEL=DLinear

[ -x "$PY" ] || { echo "ERROR: $PY missing" >&2; exit 1; }
[ -f "$TF/run.py" ] || { echo "ERROR: $TF/run.py missing" >&2; exit 1; }

n_ok=0; n_skip=0; n_fail=0
for SET in set1 set2 set3 set4; do
  for COHORT in cohort_2020 cohort_2021 cohort_2022; do
    DATA="$REPO/datasets/walkforward/mid_highmid_price/$SET/${COHORT}_aligned"
    [ -d "$DATA" ] || { echo "SKIP $SET/$COHORT: no data" >&2; continue; }
    for SEED in 1 2 3 4 5 6 7 8 9 10; do
      TAG="seed_${SEED}"
      OUT="$REPO/results/transformer_bench/mid_highmid/$SET/$COHORT/$MODEL/$TAG"
      if [ -f "$OUT/.done" ]; then n_skip=$((n_skip+1)); continue; fi
      mkdir -p "$OUT"
      MID="${MODEL}_L20_${SET}_${COHORT}_${TAG}"
      cd "$TF"
      RUN_SEED=$SEED STOCK_META_DIR="$OUT" STOCK_NORM_SCOPE=train_val \
      "$PY" run.py \
        --is_training 1 --model "$MODEL" --model_id "$MID" \
        --data stock_pooled --root_path "$DATA" --data_path unused \
        --features MS --target RET --freq d \
        --seq_len 20 --label_len 0 --pred_len 1 \
        --enc_in 6 --dec_in 6 --c_out 1 \
        --d_model 64 --n_heads 4 --d_ff 128 --dropout 0.1 \
        --e_layers 2 --d_layers 1 --factor 10 \
        --batch_size 256 --learning_rate 0.0005 --train_epochs 30 --patience 5 \
        --loss MSE --lradj cosine --itr 1 --des pooled --num_workers 0 \
        --use_gpu True --gpu 0 --checkpoints "$OUT/checkpoints" > "$OUT/train.log" 2>&1
      rc=$?
      if [ $rc -ne 0 ]; then
        echo "FAILED $SET/$COHORT/$TAG rc=$rc (see $OUT/train.log)" >&2
        n_fail=$((n_fail+1)); continue
      fi
      # collect this task's results dir only -- anchor the glob so seed_1 does not
      # also match seed_10 (verified failure mode in anvil_transformer.sb)
      found=0
      for d in results/"${MID}_${MODEL}_stock_pooled_"*; do
        [ -d "$d" ] || continue
        cp -r "$d" "$OUT/" && found=1
      done
      if [ "$found" -ne 1 ]; then
        echo "FAILED $SET/$COHORT/$TAG: no results dir matching $MID" >&2
        n_fail=$((n_fail+1)); continue
      fi
      "$PY" "$REPO/scripts/transformer_bench/collect_predictions.py" --run-dir "$OUT" \
        >> "$OUT/train.log" 2>&1 || { echo "FAILED collect $SET/$COHORT/$TAG" >&2; n_fail=$((n_fail+1)); continue; }
      touch "$OUT/.done"
      n_ok=$((n_ok+1))
      echo "OK $SET/$COHORT/$TAG"
    done
  done
done
echo "### DLinear mid_highmid: $n_ok ok, $n_skip skipped, $n_fail failed"
