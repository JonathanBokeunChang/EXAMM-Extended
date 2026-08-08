#!/bin/bash
# Zero-shot foundation-model grid over all 12 portfolio cells, protocol-matched at L=96.
#
#   bash scripts/transformer_bench/run_zeroshot_grid.sh MODEL [SAMPLES] [DEVICE] [SEED]
#   bash scripts/transformer_bench/run_zeroshot_grid.sh Salesforce/moirai-2.0-R-small 100 cpu
#
# CONTEXT IS 96 BY DELIBERATE CHOICE, NOT BY DEFAULT. The MOIRAI authors ship context_length as a
# required-without-default field (`???`) in their own eval configs because they tune it per
# benchmark; there is no upstream value to inherit. 96 is chosen to match the seq_len every
# transformer in this study is given, so the headline comparison is like-for-like. A longer,
# validation-selected context is a separate experiment and must be reported as one.
#
# RESUMABLE. A cell whose seed dir already holds .done is skipped, so an interrupted grid picks up
# where it stopped -- same discipline as the transformer campaign, where .done is written only
# after the per-ticker count assertion passes.
#
# SAMPLES IS INERT FOR 2.x. Moirai 2.x uses a quantile loss and does not sample; the argument is
# accepted and recorded as 0 so the same driver covers both families.
set -uo pipefail
MODEL=${1:?usage: run_zeroshot_grid.sh MODEL [SAMPLES] [DEVICE] [SEED]}
SAMPLES=${2:-100}
DEVICE=${3:-cpu}
SEED=${4:-0}
CTX=${CTX:-96}
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY=${PY:-$REPO/external/zs_env/bin/python}
ROOT=${ROOT:-$REPO/results/zeroshot/mid_highmid}
LABEL=$(echo "$MODEL" | sed 's|.*/||; s|[.-]||g')
cd "$REPO" || exit 1

echo "=== $MODEL  ctx=$CTX samples=$SAMPLES device=$DEVICE seed=$SEED ==="
done_n=0; ran=0; t0=$(date +%s)
for st in set1 set2 set3 set4; do
  for coh in cohort_2020_aligned cohort_2021_aligned cohort_2022_aligned; do
    d="$ROOT/$st/author/$coh/$LABEL/L$CTX/seed_$SEED"
    if [ -f "$d/.done" ]; then
      echo "  skip $st/$coh (done)"; done_n=$((done_n+1)); continue
    fi
    [ -d "$REPO/datasets/walkforward/mid_highmid_price/$st/$coh" ] || { echo "  MISSING DATA $st/$coh"; continue; }
    "$PY" scripts/transformer_bench/zeroshot_probe.py \
      --set "$st" --cohort "$coh" --model "$MODEL" --context "$CTX" \
      --samples "$SAMPLES" --device "$DEVICE" --seed "$SEED" --batch "${BATCH:-64}" \
      --root "$ROOT" 2>&1 | grep -viE "^warning|hf_hub|it/s" | sed 's/^/    /'
    ran=$((ran+1))
  done
done
echo "=== $LABEL: $ran run, $done_n already done, $(( $(date +%s) - t0 ))s ==="
