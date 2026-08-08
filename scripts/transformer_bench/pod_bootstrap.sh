#!/bin/bash
# Bring a freshly rented GPU pod from bare image to "ready to launch the campaign", with a check
# after every step so a failure surfaces here rather than 40 runs into a paid campaign.
#
#   bash scripts/transformer_bench/pod_bootstrap.sh            # full: build, verify, smoke
#   bash scripts/transformer_bench/pod_bootstrap.sh --skip-smoke
#
# Run it ON THE POD, from the repo root, after the transfer step (see the rsync in the header of
# sync_from_pod.sh, or the runbook). It is idempotent: every stage is skipped if already satisfied,
# so re-running after a fixed problem costs seconds.
#
# WHAT IT DOES NOT DO: it never launches the campaign. The last thing it prints is the command to
# do that, so starting 240 paid runs stays an explicit human action.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SKIP_SMOKE=0; [ "${1:-}" = "--skip-smoke" ] && SKIP_SMOKE=1
cd "$REPO"
fail() { echo "FAIL: $*" >&2; exit 1; }
ok()   { echo "  ok  $*"; }

echo "=============================================================="
echo "  1. HARDWARE"
echo "=============================================================="
command -v nvidia-smi >/dev/null || fail "no nvidia-smi -- this is not a GPU pod"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
NGPU=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')
ok "$NGPU GPU(s) visible"
NCPU=$(nproc 2>/dev/null || echo 0)
ok "$NCPU vCPU"
# 9 shards x (NUM_WORKERS + 1 main). At the default 4 workers that is 45 processes; on a narrow pod
# the shards thrash and the parallelism you paid for evaporates. Recommend rather than enforce.
if [ "$NCPU" -lt 40 ]; then
  echo "  NOTE: $NCPU vCPU is tight for 9 shards at NUM_WORKERS=4 (wants ~45 procs)."
  echo "        Export NUM_WORKERS=2 before launching."
fi
AVAIL=$(df -BG --output=avail "$REPO" 2>/dev/null | tail -1 | tr -dc '0-9')
[ -n "$AVAIL" ] && { [ "$AVAIL" -ge 15 ] && ok "${AVAIL}G free" || echo "  NOTE: only ${AVAIL}G free; the CUDA venv alone is ~6-8G"; }

echo
echo "=============================================================="
echo "  2. DATA"
echo "=============================================================="
if [ ! -d "$REPO/datasets/walkforward/mid_highmid_price/set1/cohort_2020_aligned" ]; then
  B=$(ls "$REPO"/portfolio_data.tar.gz 2>/dev/null | head -1)
  [ -n "$B" ] || fail "no portfolio data and no portfolio_data.tar.gz to unpack"
  echo "  unpacking $B"
  # --no-same-owner: a RunPod network volume (MooseFS) refuses chown, so GNU tar reports
  # "Cannot change ownership" for every entry and exits non-zero even though every file extracted
  # correctly. Without this the unpack looks like a hard failure and aborts a perfectly good pod.
  tar --no-same-owner -xzf "$B" -C "$REPO" || fail "tar failed"
  # AppleDouble sidecars from a macOS-built archive. These MUST go: the loader lists files with
  # os.listdir, which (unlike a glob) returns dotfiles, and "._X_train.csv" ends with "_train.csv"
  # -- so leaving them doubles the pooled panel with binary junk instead of raising.
  n=$(find "$REPO/datasets" -name '._*' -print -delete 2>/dev/null | wc -l | tr -d ' ')
  [ "$n" -gt 0 ] && echo "  removed $n AppleDouble sidecar(s) from the archive"
fi
miss=0
for s in set1 set2 set3 set4; do for c in 2020 2021 2022; do
  d="$REPO/datasets/walkforward/mid_highmid_price/$s/cohort_${c}_aligned"
  n=$(ls "$d"/*_train.csv 2>/dev/null | wc -l | tr -d ' ')
  [ "$n" -eq 50 ] || { echo "  MISSING $s/cohort_$c ($n/50 train files)"; miss=$((miss+1)); }
done; done
[ "$miss" -eq 0 ] || fail "$miss incomplete cell(s)"
ok "12 cells x 50 stocks present"

echo
echo "=============================================================="
echo "  3. BUILD (harness + venv + authors' implementations)"
echo "=============================================================="
# setup_anvil.sh is Anvil-NAMED but not Anvil-ONLY: it probes for Lmod and falls back to system
# python3 when there is none, which is the RunPod case. It clones the harness, builds a cu118 venv
# (cu118 covers the 4090's sm_89), installs all four authors' implementations, and applies the
# run.py patches. It is idempotent, so this is safe to re-run.
bash "$REPO/scripts/transformer_bench/setup_anvil.sh" || fail "setup_anvil.sh failed"

echo
echo "=============================================================="
echo "  4. VERIFY THE BUILD"
echo "=============================================================="
TF="${TF_HARNESS:-$REPO/external/DeformTime}"
PY="$(cat "$REPO/external/.tfenv_path" 2>/dev/null || echo "$REPO/external/tfenv")/bin/python"
[ -x "$PY" ] || fail "venv python missing at $PY"
"$PY" -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' \
  || fail "venv torch cannot see CUDA -- wrong wheel for this driver"
"$PY" -c 'import torch; print("  torch", torch.__version__, "| cuda", torch.version.cuda, "|", torch.cuda.get_device_name(0))'

# THE THREE SILENT-CORRUPTION PATCHES. Each fails invisibly if absent: without RUN_SEED every seed
# is bit-identical while the logs look healthy; without the stride fix DeformTime gets
# non-overlapping patches; without use_norm the authors' iTransformer trains with normalization
# hardcoded on. anvil_transformer.sb re-checks these per run, but finding out here costs seconds
# instead of failing 240 times.
grep -q "RUN_SEED" "$TF/run.py"                              || fail "run.py unpatched: RUN_SEED (all seeds would be identical)"
ok "run.py honours RUN_SEED"
grep -q "^\s*#\s*args\.stride = args\.patch_len" "$TF/run.py" || fail "run.py unpatched: --stride still overwritten"
ok "run.py honours --stride"
grep -q -- "--use_norm'" "$TF/run.py"                         || fail "run.py lacks --use_norm (iTransformer)"
ok "run.py accepts --use_norm"
for f in src/models/ITransformerOfficial.py src/models/CrossformerOfficial.py \
         src/models/PatchTSTOfficial.py src/models/_crossformer_official/cross_former.py; do
  [ -f "$TF/$f" ] || fail "missing authors' implementation: $f"
done
ok "all four authors' implementations installed"
grep -q "configs.use_norm" "$TF/src/models/ITransformerOfficial.py" \
  || fail "ITransformerOfficial.py ignores configs.use_norm -- not the authors' file"
ok "iTransformer reads configs.use_norm"
[ -f "$TF/data/data_provider/stock_pooled_loader.py" ] || fail "stock_pooled_loader.py not installed into harness"
ok "stock_pooled loader installed"

echo
echo "=============================================================="
echo "  5. CONFIGURATION (what the campaign will actually run)"
echo "=============================================================="
MODELS="${MODELS:-CrossformerOfficial:traffic ITransformerOfficial}" \
  DRY_RUN=1 bash "$REPO/scripts/transformer_bench/run_portfolio_campaign.sh" 2>&1 | head -6

if [ "$SKIP_SMOKE" = "1" ]; then
  echo; echo "### --skip-smoke: NOT validated end to end. Do not launch on this alone."
else
echo
echo "=============================================================="
echo "  6. SMOKE (2 epochs each -- the only test of the runtime half)"
echo "=============================================================="
# Everything above is static. DRY_RUN exits before the environment preflight, training, the
# collection glob, the test-count guard and the timing record, so this is the first and only thing
# that exercises UNIVERSE=mid_highmid end to end. SMOKE=1 writes to a `smoke` tag that cannot
# collide with a real seed and never touches .done.
sm=0
run_smoke() {  # $1=model $2=cf_config $3=variant
  echo "--- smoke: $1 ${3:+($2)}"
  SMOKE=1 SLURM_SUBMIT_DIR="$REPO" SLURM_ARRAY_TASK_ID=1 \
    MODEL="$1" CF_CONFIG="$2" VARIANT="$3" \
    UNIVERSE=mid_highmid SET=set1 COHORT=cohort_2020_aligned \
    SEQ_LEN=96 PROFILE=author NUM_WORKERS="${NUM_WORKERS:-2}" \
    bash "$REPO/scripts/transformer_bench/anvil_transformer.sb" 2>&1 | tail -5
  local o="$REPO/results/transformer_bench/mid_highmid/set1/author${3:+/var_$3}/cohort_2020_aligned/$1/L96/smoke"
  [ -f "$o/predictions.csv" ] || { echo "  FAIL: no predictions.csv"; sm=$((sm+1)); return; }
  [ -f "$o/timing.json" ]     || { echo "  FAIL: no timing.json"; sm=$((sm+1)); return; }
  "$PY" -c "
import json;d=json.load(open('$o/timing.json'))
print('  ok  %d epochs, %.0fs train, %ss wall on %s'%(d['epochs'],d['train_sec'],d['wall_sec'],d['gpu']))
assert d['gpu']!='unknown', 'GPU NOT RECORDED'
" || sm=$((sm+1))
}
run_smoke CrossformerOfficial traffic cftraffic
run_smoke ITransformerOfficial etth1 ""
[ "$sm" -eq 0 ] || fail "$sm smoke check(s) failed -- do not launch"
echo "  ok  both smokes clean: data path, windowing guard, collection, timing all verified"
fi

echo
echo "=============================================================="
echo "  READY"
echo "=============================================================="
cat <<LAUNCH
Launch the campaign (240 runs, 9 shards over ${NGPU} GPU):

  export MODELS="CrossformerOfficial:traffic ITransformerOfficial"
  export NUM_WORKERS=${NUM_WORKERS:-2}
  for s in 0 1 2 3 4 5 6 7 8; do
    CUDA_VISIBLE_DEVICES=\$((s % ${NGPU})) SHARD=\$s NSHARDS=9 \\
      nohup bash scripts/transformer_bench/run_portfolio_campaign.sh > shard\$s.log 2>&1 &
  done

Then, FROM YOUR LAPTOP, keep results safe against the pod going away:

  bash scripts/transformer_bench/sync_from_pod.sh USER@HOST:PORT $REPO --watch 600

Progress:  grep -c '^### OUT' shard*.log ; find results/transformer_bench/mid_highmid -name .done | wc -l
LAUNCH
