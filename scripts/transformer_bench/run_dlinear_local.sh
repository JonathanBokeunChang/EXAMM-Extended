#!/bin/bash
# Local (CPU, no Anvil/SLURM) DLinear run -- same protocol as anvil_transformer.sb, minus the
# SLURM/module plumbing DLinear doesn't need (it uses EXTRA=(), no tokenizer patches).
# bash only.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TF_HARNESS=$REPO/external/DeformTime
PY=$REPO/external/tfenv/bin/python
MODEL=DLinear

for COHORT in cohort_2020_aligned cohort_2021_aligned; do
  DATA=$REPO/datasets/walkforward/$COHORT
  for SEED in 1 2 3 4 5 6 7 8 9 10; do
    TAG=seed_${SEED}
    OUT=$REPO/results/transformer_bench/$COHORT/${MODEL}/${TAG}
    if [ -f "$OUT/.done" ]; then echo "SKIP $COHORT seed $SEED (done)"; continue; fi
    mkdir -p "$OUT"
    export RUN_SEED=$SEED
    export STOCK_META_DIR=$OUT
    export STOCK_NORM_SCOPE=train_val
    cd "$TF_HARNESS"
    "$PY" run.py \
      --is_training 1 --model "$MODEL" --model_id "${MODEL}_L20_${COHORT}_${TAG}" \
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
    if [ "$rc" -ne 0 ]; then
      echo "FAILED $COHORT seed $SEED rc=$rc -- see $OUT/train.log" >&2
      tail -20 "$OUT/train.log" >&2
      exit 1
    fi

    found=0
    for d in results/"${MODEL}_L20_${COHORT}_${TAG}_${MODEL}_stock_pooled_"*; do
      [ -d "$d" ] || continue
      cp -r "$d" "$OUT/" && found=1
    done
    if [ "$found" -ne 1 ]; then
      echo "FAILED $COHORT seed $SEED: no results dir matching ${TAG}" >&2; exit 1
    fi

    "$PY" "$REPO/scripts/transformer_bench/collect_predictions.py" --run-dir "$OUT" >> "$OUT/train.log" 2>&1
    touch "$OUT/.done"
    ep=$(grep -c "^Epoch:" "$OUT/train.log")
    echo "DONE  $COHORT seed $SEED  (${ep} epochs)"
  done
done
echo "all runs complete"
