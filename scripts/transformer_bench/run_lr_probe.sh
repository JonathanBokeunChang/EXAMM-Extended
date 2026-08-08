#!/bin/bash
# Learning-rate probe: does ANY lr in a 50x range rescue the models that post negative test IC?
#
#   bash scripts/transformer_bench/run_lr_probe.sh
#   MODELS="PatchTST DeformTime" bash scripts/transformer_bench/run_lr_probe.sh
#
# WHY THIS DRIVES anvil_transformer.sb RATHER THAN run_patchtst_lowlr.sh
# ----------------------------------------------------------------------
# run_patchtst_lowlr.sh is a standalone re-implementation of the same protocol. It hardcodes
# `--train_epochs 30`, which is correct for the shared protocol but WRONG for this probe: at 1e-5
# the model is still descending well past epoch 30, so a 30-epoch cap would confound "a lower
# learning rate does not help" with "we stopped it before it converged" -- which is the entire
# question being asked. anvil_transformer.sb raises the cap to 60 for any non-default LR and routes
# non-default LRs into results/transformer_bench/lowlr/..., keeping the published GPU results at the
# unsuffixed path byte-for-byte. Driving the real job script also guarantees every other knob is the
# one the published runs used, rather than a second copy that can drift.
#
# LR LADDER: 5e-4 (already on disk, the shared protocol) -> 1e-4 -> 1e-5. That is 50x, wide enough
# that "no LR in this range helps" is a real statement. 5e-5 is deliberately EXCLUDED: seed 1 was
# already run there and its best val (0.0010544) is indistinguishable from the original protocol's
# (0.0010556), so it is a known null and spending 10 seeds to re-confirm it buys nothing.
#
# READING THE RESULT: the diagnostic is the BEST VALIDATION EPOCH, not the loss. The failing models
# early-stop on a checkpoint from epoch 1-2, i.e. barely trained, which is why their cross-sectional
# loading sits on the momentum side of a panel whose only exploitable effect is reversal. If the
# best epoch stays at 1-2 across the whole ladder, LR was not the binding constraint and the paper
# can say so. If it moves out to 5+ AND test IC turns positive, the published rows are a protocol
# artifact and both tables need correcting.
#
# Resumable: anvil_transformer.sb skips any run whose OUT/.done exists.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODELS=${MODELS:-"PatchTST"}
LRS=${LRS:-"0.0001 0.00001"}
COHORT=${COHORT:-cohort_2020_aligned}
SEEDS=${SEEDS:-"1 2 3 4 5 6 7 8 9 10"}

fail=0
for MODEL in $MODELS; do
  for LR in $LRS; do
    LRTAG=$(printf 'lr%s' "$LR" | tr -d '.')
    for SEED in $SEEDS; do
      OUT="$REPO/results/transformer_bench/lowlr/$COHORT/$MODEL/$LRTAG/seed_${SEED}"
      if [ -f "$OUT/.done" ]; then
        echo "=== $MODEL lr=$LR seed $SEED -- already done, skipping ==="
        continue
      fi
      echo "=== $MODEL | lr=$LR | $COHORT | seed $SEED ==="
      SLURM_SUBMIT_DIR="$REPO" SLURM_ARRAY_TASK_ID="$SEED" \
        MODEL="$MODEL" COHORT="$COHORT" LR="$LR" \
        bash "$REPO/scripts/transformer_bench/anvil_transformer.sb"
      rc=$?
      if [ "$rc" -ne 0 ]; then
        echo "FAILED $MODEL lr=$LR seed $SEED rc=$rc -- continuing" >&2
        fail=$((fail + 1))
      else
        # The headline diagnostic, printed per run so a tail of the log tells the story.
        imp=$(grep -c "Validation loss decreased" "$OUT/train.log" 2>/dev/null || echo "?")
        echo "    -> val improved $imp time(s)"
      fi
    done
  done
done

echo
echo "### lr probe complete ($fail failed)"
echo "### best-validation-epoch summary (the actual diagnostic):"
for MODEL in $MODELS; do
  for LR in $LRS; do
    LRTAG=$(printf 'lr%s' "$LR" | tr -d '.')
    for SEED in $SEEDS; do
      f="$REPO/results/transformer_bench/lowlr/$COHORT/$MODEL/$LRTAG/seed_${SEED}/train.log"
      [ -f "$f" ] || continue
      printf '%s lr=%s seed=%s improvements=%s best=%s\n' "$MODEL" "$LR" "$SEED" \
        "$(grep -c 'Validation loss decreased' "$f")" \
        "$(grep 'Early stopping on' "$f" | sed 's/.*score //' | head -1)"
    done
  done
done
