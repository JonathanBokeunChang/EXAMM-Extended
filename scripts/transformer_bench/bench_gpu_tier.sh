#!/bin/bash
# Measure whether a given rented GPU is CHEAPER PER UNIT OF WORK than the baseline, and project the
# full campaign cost on it. Run this on a candidate pod before committing a multi-hundred-dollar
# campaign to a card nobody has timed.
#
#   bash scripts/transformer_bench/bench_gpu_tier.sh [RATE_USD_PER_HR] [STEPS]
#
#   RATE  $/hr for THIS pod   (default 0.51, the RTX 3090 tier already measured)
#   STEPS timed steps/model   (default 200; 30 warmup steps are discarded first)
#
# WHY $/1000 STEPS AND NOT ms/step: a card at 39% of the price wins unless it is more than 2.55x
# slower. Ranking cards by speed picks the wrong one; ranking by cost per unit of work is the whole
# question. This workload is also latency-bound rather than compute-bound (21x the parameters cost
# only 1.26x the time, measured), and VRAM never binds (largest model is 11.2M params at batch 32),
# so cheap cards are genuinely in play here in a way they would not be for a large-model workload.
#
# NO DATASET AND NO VENV REQUIRED. Step time does not depend on the CONTENT of the tensors, only
# their shapes, so this feeds synthetic batches at the exact real shapes. That makes it runnable on a
# bare PyTorch pod in ~2 minutes instead of after a 25-minute environment build -- which matters,
# because the entire point is to spend as little as possible deciding where to spend.
#
# CONFIGS ARE NOT DUPLICATED HERE. Each model's flags are obtained by running anvil_transformer.sb
# in DRY_RUN mode and parsing the command line it would actually issue, so this benchmark cannot
# drift from the pinned author settings. If the .sb changes, this follows automatically.
set -uo pipefail

RATE=${1:-0.51}
STEPS=${2:-200}
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TF_HARNESS=${TF_HARNESS:-$REPO/external/DeformTime}
SEQ_LEN=${SEQ_LEN:-96}
PY=${PYTHON:-python3}

echo "### repo   : $REPO"
echo "### rate   : \$$RATE/hr"
echo "### steps  : $STEPS timed (+30 warmup)"

# ---- minimal deps. A RunPod PyTorch template already has torch; einops is tiny and pure python.
"$PY" -c 'import torch' 2>/dev/null || { echo "ERROR: no torch on PATH. Use a PyTorch pod image." >&2; exit 1; }
"$PY" -c 'import einops' 2>/dev/null || "$PY" -m pip install --quiet einops

# ---- harness + the two vendored official implementations
if [ ! -d "$TF_HARNESS/.git" ]; then
  echo "### cloning harness"
  git clone --depth 1 --quiet https://github.com/ClaudiaShu/DeformTime.git "$TF_HARNESS"
fi
PYTHON="$PY" bash "$REPO/scripts/transformer_bench/install_patchtst_official.sh"     "$TF_HARNESS" >/dev/null
PYTHON="$PY" bash "$REPO/scripts/transformer_bench/install_crossformer_official.sh"  "$TF_HARNESS" >/dev/null
PYTHON="$PY" bash "$REPO/scripts/transformer_bench/install_itransformer_official.sh" "$TF_HARNESS" >/dev/null
echo "### official implementations installed"

# ---- pull the exact flags each model would run with.
#
# SPECS ARE `label:MODULE[:cf_config]`. Two of the four rows here cannot be addressed by module name
# alone:
#   * CrossformerOfficial has TWO published configurations and this study reports the TRAFFIC one
#     (722,032 params, d_model 64 / d_ff 128 / n_heads 2, seg_len 12). The ETTh1 column
#     (11,185,944 params) is the paper's secondary "see text" row. Benchmarking only the default
#     would price a model the campaign does not run.
#   * ITransformerOfficial was absent entirely, so the one model whose lookback is natively 96 --
#     the cleanest transformer comparison in the paper -- had no cost estimate at all.
# Both are also exactly the two models whose original-universe training logs were pruned, so this
# benchmark is now the only way to recover their step time without retraining.
FLAGS_FILE=$(mktemp); trap 'rm -f "$FLAGS_FILE"' EXIT
for SPEC in "PatchTSTOfficial:PatchTSTOfficial" \
            "CrossformerOfficial[traffic]:CrossformerOfficial:traffic" \
            "CrossformerOfficial[etth1]:CrossformerOfficial:etth1" \
            "ITransformerOfficial:ITransformerOfficial" \
            "DeformTime:DeformTime"; do
  LABEL="${SPEC%%:*}"; REST="${SPEC#*:}"; MOD="${REST%%:*}"
  CFG="${REST#*:}"; [ "$CFG" = "$REST" ] && CFG=etth1
  # A non-etth1 CF_CONFIG makes the .sb demand a VARIANT before it will emit anything.
  VAR=""; [ "$CFG" != "etth1" ] && VAR="cf${CFG}"
  line=$(DRY_RUN=1 SLURM_SUBMIT_DIR="$REPO" SLURM_ARRAY_TASK_ID=1 MODEL=$MOD SEQ_LEN=$SEQ_LEN \
         PROFILE=author CF_CONFIG="$CFG" VARIANT="$VAR" \
         bash "$REPO/scripts/transformer_bench/anvil_transformer.sb" 2>/dev/null \
         | grep '^### cmd:')
  [ -n "$line" ] || { echo "ERROR: could not derive flags for $LABEL" >&2; exit 1; }
  echo "$LABEL|$MOD|${line#\#\#\# cmd: run.py }" >> "$FLAGS_FILE"
done

cd "$TF_HARNESS"
"$PY" - "$FLAGS_FILE" "$RATE" "$STEPS" "$REPO" <<'PYEOF'
import sys, time, types, statistics
import torch

flags_file, rate, steps, repo = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), sys.argv[4]

# timm supplies exactly one symbol here (trunc_normal_), which torch provides natively. Stubbing it
# avoids dragging in torchvision purely to time a matmul. Initialisation does not affect step time.
try:
    import timm  # noqa
except Exception:
    for n in ("timm", "timm.models", "timm.models.layers"):
        sys.modules[n] = types.ModuleType(n)
    sys.modules["timm.models.layers"].trunc_normal_ = torch.nn.init.trunc_normal_

from src.exp.exp_basic import Exp_Basic  # noqa: registers the model dict
import src.models as M

def parse(argv):
    """--flag value pairs -> a configs-like namespace, coercing numerics."""
    ns = types.SimpleNamespace()
    toks = argv.split()
    i = 0
    while i < len(toks):
        if toks[i].startswith("--"):
            k = toks[i][2:]
            v = toks[i+1] if i+1 < len(toks) and not toks[i+1].startswith("--") else "1"
            for cast in (int, float):
                try: v = cast(v); break
                except ValueError: pass
            setattr(ns, k, v); i += 2
        else:
            i += 1
    # defaults the harness supplies that are not on the command line
    for k, d in (("output_attention", False), ("activation", "gelu"), ("embed", "timeF"),
                 ("moving_avg", 25), ("layer_dropout", 0.6), ("d_layers", 1)):
        if not hasattr(ns, k): setattr(ns, k, d)
    return ns

dev = ("cuda" if torch.cuda.is_available() else
       "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
name = torch.cuda.get_device_name(0) if dev == "cuda" else dev
print(f"\n### device : {name}")

def sync():
    if dev == "cuda": torch.cuda.synchronize()

rows = []
for line in open(flags_file):
    # label | module | flags -- label and module differ for the two Crossformer configurations,
    # which are the same class at different capacities.
    label, module, argv = line.rstrip("\n").split("|", 2)
    cfg = parse(argv)
    mdl = getattr(M, module).Model(cfg).to(dev).train()
    B, L, C = cfg.batch_size, cfg.seq_len, cfg.enc_in
    x  = torch.randn(B, L, C, device=dev)
    xm = torch.randn(B, L, 4, device=dev)
    y  = torch.randn(B, cfg.pred_len, 1, device=dev)
    opt = torch.optim.Adam(mdl.parameters(), lr=cfg.learning_rate)
    crit = torch.nn.MSELoss()

    def one():
        opt.zero_grad()
        out = mdl(x, xm, None, None)[:, -cfg.pred_len:, -1:]
        crit(out, y).backward(); opt.step()

    for _ in range(30): one()          # warmup: cudnn autotune, allocator, lazy init
    sync(); t0 = time.perf_counter()
    for _ in range(steps): one()
    sync(); ms = (time.perf_counter() - t0) / steps * 1000

    n_par = sum(p.numel() for p in mdl.parameters())
    rows.append((label, B, n_par, ms))
    print(f"  {label:28s} B={B:<4d} params={n_par:>11,}  {ms:7.2f} ms/step")
    del mdl, opt; torch.cuda.empty_cache() if dev == "cuda" else None

# ---- cost model. Windows per cohort come from the real data when present.
import glob, os
def windows(cohort_dir, L):
    fs = glob.glob(os.path.join(cohort_dir, "*_train.csv"))
    if not fs: return None
    rows_ = sum(1 for _ in open(fs[0])) - 1
    return len(fs) * (rows_ - L)

L = rows and parse(open(flags_file).readline().split("|",2)[2]).seq_len
w_orig = windows(f"{repo}/datasets/walkforward/cohort_2020_aligned", L) or 163500
# THE PORTFOLIO PATH MEASURED THE WRONG SPLIT. It read
# datasets/walkforward/mid_highmid/set1/cohort_2020 -- the UNALIGNED cohort, 5,283 train rows. The
# campaign trains on <cohort>_aligned, which is 4,125, because align_cohort.py truncates every
# stock to a common calendar. The glob therefore succeeded and returned a number that looked
# entirely reasonable while overstating windows by 1.28x, which is the failure mode to watch for
# here: a silently plausible cost estimate is worse than a missing one.
#
# (Either tree is fine for the feature data itself -- mid_highmid and mid_highmid_price hold the
# same tickers and rows, the latter adding PRC/TRAN_COST for trading eval, and stock_pooled_loader
# reads only the six model features. It is aligned-vs-unaligned that matters, not which tree.)
#
# Averaged over the 12 real cells rather than taken from set1/2020, since cohorts differ by ~15%.
import statistics as _st
_wp = [w for s in ("set1","set2","set3","set4") for c in ("2020","2021","2022")
       for w in [windows(f"{repo}/datasets/walkforward/mid_highmid_price/{s}/cohort_{c}_aligned", L)]
       if w]
w_port = int(_st.mean(_wp)) if _wp else int(163500*1.30)
print(f"\n### windows/cohort: original {w_orig:,} | portfolio {w_port:,}"
      f" ({'mean of %d real cells' % len(_wp) if _wp else 'ESTIMATE -- portfolio data not present'})")
print(f"\n{'model':28s} {'$/1000 steps':>13s} {'min/epoch (orig)':>18s}")
# BASE is a single-process RTX 3090 reference, measured with each model alone on the card.
#
# DO NOT COMPARE IT AGAINST CONTENDED NUMBERS. The original-universe author campaign (2026-08-04/05)
# ran on an RTX 4090 at up to 9-WAY CONCURRENCY (9 shards over 3 GPUs, 3 processes per card).
# Its per-process step times work out to 1.43x these 3090 figures, which reads as "the 4090 is
# slower" and is simply false -- it is 3 processes sharing one card. Backing the sharing out, the
# per-process inflation is roughly 1.7-2.1x, so 3 procs/GPU delivered about 1.4-1.75x the aggregate
# throughput of 1 proc/GPU. Worth doing, nowhere near the 3x that shard count suggests.
#
# The practical rule: this benchmark measures a card SOLO, so its verdict is only valid against
# other solo measurements. Campaign planning must use observed wall-clock at the concurrency you
# actually intend to run.
BASE = {"CrossformerOfficial[etth1]": 59.36, "DeformTime": 24.35, "PatchTSTOfficial": 17.38}  # 3090, solo
for m, B, n, ms in rows:
    sph = 3600_000 / ms
    print(f"  {m:26s} {rate/sph*1000:>13.5f} {ms*(w_orig//B)/60000:>18.2f}")

OVER = 6.0     # min/run of dataset construction, measured
# EPOCHS ARE MEASURED, NOT THE CAP. These are the mean epochs/run actually reached on the original
# universe, recovered from the surviving train.log files (10 seeds x 2 cohorts each):
#     Crossformer[etth1]  7.0   against a cap of 20, patience 3
#     DeformTime          6.7   against a cap of 100, patience 5
#     PatchTSTOfficial   27.8   against a cap of 100, patience 20
# The caps are never approached -- DeformTime stops at 7% of its budget -- so the old (lo, hi) cap
# ranges here priced a campaign that cannot happen and inflated the top of every estimate roughly
# 2-3x. A range is still printed, but as +/-30% around the measured mean, which reflects the real
# seed-to-seed spread rather than the distance to an unreachable ceiling.
#
# THE TWO ESTIMATED ROWS. Crossformer[traffic] and ITransformerOfficial had their original-universe
# logs pruned before anyone read the epoch counts off them, so their entries are bounded guesses,
# not measurements: traffic shares etth1's cap-20/patience-3 schedule, and iTransformer's cap is
# only 10 with patience 3, so both are tightly bounded even when unknown. Marked (est) in the
# output so a reader is never misled about which is which. Training one seed of each is the only
# way to replace them.
EP = {"CrossformerOfficial[etth1]": (7.0, False), "DeformTime": (6.7, False),
      "PatchTSTOfficial": (27.8, False),
      "CrossformerOfficial[traffic]": (7.0, True), "ITransformerOfficial": (6.0, True)}
print(f"\n{'campaign':34s} {'GPU-hours':>13s} {'cost @ $%.2f/hr' % rate:>18s}")
for clabel, cells, w, nruns in (("2 original cells (%d runs)" % (2*10*len(rows)), 2, w_orig, 0),
                                ("12 portfolio cells (%d runs)" % (12*10*len(rows)), 12, w_port, 0)):
    lo = hi = 0.0
    for m, B, n, ms in rows:
        mpe = ms * (w // B) / 60000
        e, _est = EP.get(m, (10.0, True))
        lo += cells*10 * (mpe*e*0.7 + OVER) / 60
        hi += cells*10 * (mpe*e*1.3 + OVER) / 60
    print(f"  {clabel:32s} {f'{lo:.0f}-{hi:.0f}':>13s} {f'${lo*rate:.0f}-${hi*rate:.0f}':>18s}")
print("  (epochs measured for 3 of %d models; Crossformer[traffic] and iTransformer are bounded"
      " estimates -- their logs were pruned)" % len(rows))

print("\n### VERDICT vs the RTX 3090 baseline ($0.51/hr)")
tot_this = tot_base = 0.0
for m, B, n, ms in rows:
    if m not in BASE: continue      # only the three with a 3090 reference are comparable
    tot_this += ms; tot_base += BASE[m]
slow = tot_this / tot_base
print(f"  this card is {slow:.2f}x the 3090's step time at {rate/0.51:.2f}x the price")
print(f"  cost per unit of work: {slow * rate / 0.51:.2f}x the 3090")
print("  -> " + ("CHEAPER: use this card" if slow*rate/0.51 < 0.95 else
                 "ABOUT THE SAME: stay on the 3090" if slow*rate/0.51 < 1.05 else
                 "MORE EXPENSIVE: stay on the 3090"))
PYEOF
