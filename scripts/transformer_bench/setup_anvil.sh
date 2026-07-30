#!/bin/bash
# One-time setup for the transformer benchmark on Anvil. Run on a LOGIN node before any sbatch.
#
#   bash scripts/transformer_bench/setup_anvil.sh
#
# WHY ONE REPO FOR BOTH MODELS
# ----------------------------
# The DeformTime repo is a TSLib-style framework whose src/exp/exp_basic.py registers DLinear,
# LightTS, PatchTST, Crossformer, iTransformer AND DeformTime behind a single --model flag, sharing
# one data loader, one training loop, one early-stopping rule and one optimizer. Running Crossformer
# from its own repo instead would mean a different trainer, a different scaler and a different
# split convention -- exactly the confound "apply the same settings across multiple transformers" is
# meant to avoid. So we clone DeformTime only, and get Crossformer + PatchTST from inside it.
#
# WHAT THIS PATCHES, AND WHY EACH IS LOAD-BEARING
# -----------------------------------------------
#  1. Registers Dataset_StockPooled as --data stock_pooled. Without it the only option is
#     Dataset_Custom, which imposes its own 70/10/20 ROW split and would train on our test years.
#  2. Makes the RNG seed settable. run.py hardcodes seed_everything(seed=2021), so all 10 "seeds"
#     would otherwise be bit-identical runs -- the single most damaging silent failure here, since
#     the output looks perfectly healthy.
#  3. Makes Crossformer's seg_len configurable. src/models/Crossformer.py line 21 hardcodes
#     self.seg_len = 12. At seq_len=20 that gives ceil(20/12)*12 = 24 -> in_seg_num = 2, i.e.
#     cross-time attention over TWO tokens. Crossformer would not be broken exactly, but it would be
#     benchmarked in a degenerate configuration and would lose for a reason that has nothing to do
#     with its architecture. seg_len=4 gives 5 segments.
#  4. Removes `args.stride = args.patch_len` (run.py:82), which unconditionally overwrites whatever
#     --stride was passed. Without this, DeformTime silently gets non-overlapping patches.
#  5. Makes PatchTST read configs.patch_len/configs.stride. src/models/PatchTST.py:27 declares them
#     as Python defaults (patch_len=16, stride=8) and exp_MTS_forecasting.py:21 constructs the model
#     as Model(self.args) with no extra arguments -- so the CLI flags are inert and PatchTST would
#     always run 16/8 over a 20-step window: 2 patches, the second half replication-padding.
#  6. Removes the per-epoch TEST evaluation (exp_MTS_forecasting.py:169). Selection uses vali_loss so
#     there is no mechanical leak, but this project has already been burned once by a live
#     mid-training test tracker reintroducing best-of-N-on-test bias. It also costs ~24k
#     batch-size-1 forward passes per epoch, since data_factory pins test batch_size to 1.
#
# Every patch is asserted after application; the script aborts if any did not take effect.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# TF_ prefixed, NOT named WORK or ENV_DIR. Anvil defines $WORK as a standard environment variable
# pointing at the shared project space (/anvil/projects/x-<alloc>), so `WORK=${WORK:-...}` silently
# inherited it and this script tried to git clone a third-party repo straight into the shared
# allocation. It failed only because that directory was non-empty.
#
# WHERE TO PUT THE VENV: $HOME on Anvil is quota-limited (~25 GB) and the CUDA torch wheels are
# several GB, so a home-directory venv hits "Disk quota exceeded" mid-install. Point TF_ENV_DIR at
# scratch, which is large and is the right home for a regenerable build artifact:
#
#   TF_ENV_DIR=$SCRATCH/tfenv bash scripts/transformer_bench/setup_anvil.sh
#
# The harness clone is only a few MB and can stay in the repo. Check headroom with `myquota`.
TF_HARNESS=${TF_HARNESS:-$REPO_ROOT/external/DeformTime}
TF_ENV_DIR=${TF_ENV_DIR:-$REPO_ROOT/external/tfenv}

# Slurm opens the --output/--error files BEFORE the job script runs. If slurm_logs/ is missing the
# job dies at launch with no log saying why.
mkdir -p "$REPO_ROOT/slurm_logs"

echo "### repo root : $REPO_ROOT"
echo "### harness   : $TF_HARNESS"
echo "### venv      : $TF_ENV_DIR"

# ---------------------------------------------------------------- 1. modules
# Anvil GPU jobs need the GPU module tree. Module names differ across RCAC clusters, so we probe
# rather than assume, and print `module avail` if nothing matches -- do not guess in a batch job.
module --force purge
module load modtree/gpu 2>/dev/null || echo "WARN: modtree/gpu unavailable (fine on a login node)"
# The venv's interpreter links against whichever python module is loaded here, so the batch job must
# load the SAME one or it may fail to find libpython. Record the name for anvil_transformer.sb.
PYMOD=""
if   module load anaconda    2>/dev/null; then PYMOD=anaconda
elif module load python/3.10 2>/dev/null; then PYMOD=python/3.10
elif module load python      2>/dev/null; then PYMOD=python
fi
if [ -n "$PYMOD" ]; then
  echo "### loaded $PYMOD"
  mkdir -p "$REPO_ROOT/external" && printf '%s' "$PYMOD" > "$REPO_ROOT/external/.pymodule"
else
  echo "ERROR: no python module found. Available:" >&2
  module avail python 2>&1 | head -20 >&2
  exit 1
fi
python3 -c 'import sys; print("### python", sys.version.split()[0])'

# ---------------------------------------------------------------- 2. clone
# Clobber guard. The real hazard is not the LOCATION -- putting the venv on scratch is correct and
# necessary -- it is writing into a pre-existing populated directory that belongs to something else,
# which is exactly what the inherited $WORK would have done to the shared allocation. So: allow any
# path, but refuse one that already exists, is non-empty, and does not already look like ours.
guard() {  # path, marker-that-proves-it-is-ours, description
  local p=$1 marker=$2 what=$3
  [ -e "$p" ] || return 0                      # does not exist -- we will create it
  [ -e "$p/$marker" ] && return 0              # already ours from a previous run
  [ -z "$(ls -A "$p" 2>/dev/null)" ] && return 0   # exists but empty -- fine
  echo "ERROR: refusing to use '$p' for the $what." >&2
  echo "       It already exists, is not empty, and has no $marker -- it belongs to something else." >&2
  echo "       If this came from the cluster environment, set TF_HARNESS / TF_ENV_DIR explicitly." >&2
  exit 1
}
guard "$TF_HARNESS" ".git"       "harness clone"
guard "$TF_ENV_DIR" "bin/python" "python venv"

mkdir -p "$(dirname "$TF_HARNESS")" "$(dirname "$TF_ENV_DIR")"
if [ -d "$TF_HARNESS/.git" ]; then
  echo "### harness already cloned -- skipping"
else
  git clone --depth 1 https://github.com/ClaudiaShu/DeformTime.git "$TF_HARNESS"
fi
cd "$TF_HARNESS"
echo "### harness commit $(git rev-parse --short HEAD)"

# ---------------------------------------------------------------- 3. environment
# CUDA 11.8 wheels bundle their own runtime, so no separate cuda module is loaded (and per RCAC
# guidance the ml-toolkit modules must NOT be mixed with a custom torch install).
if [ ! -x "$TF_ENV_DIR/bin/python" ]; then
  python3 -m venv "$TF_ENV_DIR"
  "$TF_ENV_DIR/bin/pip" install --quiet --upgrade pip
  "$TF_ENV_DIR/bin/pip" install --quiet torch --index-url https://download.pytorch.org/whl/cu118
  "$TF_ENV_DIR/bin/pip" install --quiet numpy pandas scikit-learn einops timm matplotlib
else
  echo "### venv exists -- skipping install"
fi
"$TF_ENV_DIR/bin/python" -c 'import torch; print("### torch", torch.__version__)'
# Record where the venv landed. It may be on scratch rather than in the repo, so the batch job
# cannot assume a fixed path -- anvil_transformer.sb reads this file.
printf '%s' "$TF_ENV_DIR" > "$REPO_ROOT/external/.tfenv_path"
echo "### recorded venv path -> external/.tfenv_path"

# ---------------------------------------------------------------- 4. patches
cp "$REPO_ROOT/scripts/transformer_bench/stock_pooled_loader.py" \
   "$TF_HARNESS/data/data_provider/stock_pooled_loader.py"

"$TF_ENV_DIR/bin/python" - "$TF_HARNESS" <<'PYEOF'
import re, sys, pathlib
work = pathlib.Path(sys.argv[1])

def patch(rel, subs):
    """subs: (regex, replacement, marker, why). `marker` is a literal string that is present iff the
    patch has already been applied -- it CANNOT be derived from `replacement`, because most
    replacements are raw strings containing backrefs like \\1 and literal '\\n', so any
    split/startswith check on them silently never matches and the patch re-applies on every run.
    That produced duplicate `--seg_len` argparse entries and an ArgumentError on the second setup."""
    p = work / rel
    src = orig = p.read_text()
    for old, new, marker, why in subs:
        if marker in src:
            print(f"    [skip] {rel}: {why} (already applied)")
            continue
        src2 = re.sub(old, new, src, count=1)
        if src2 == src:
            sys.exit(f"ERROR: {rel}: pattern for '{why}' did not match -- upstream changed, "
                     f"re-derive the patch before running anything")
        src = src2
        print(f"    [ok]   {rel}: {why}")
    if src != orig:
        p.write_text(src)

# (1) register the pooled loader
patch("data/data_provider/data_factory.py", [
    (r"(from data\.data_provider\.data_loader import [^\n]+\n)",
     r"\1from data.data_provider.stock_pooled_loader import Dataset_StockPooled\n",
     "stock_pooled_loader import", "import Dataset_StockPooled"),
    (r"(\s*)('custom': Dataset_Custom,)",
     r"\1\2\n\1'stock_pooled': Dataset_StockPooled,",
     "'stock_pooled':", "register 'stock_pooled'"),
])

# (2) seed from the environment -- otherwise all 10 array tasks are bit-identical, and the logs
#     look completely healthy while it happens. This is the most dangerous single failure here.
patch("run.py", [
    (r"seed_everything\(seed=2021\)",
     "seed_everything(seed=int(__import__('os').environ.get('RUN_SEED', 2021)))",
     "RUN_SEED", "RUN_SEED env override"),
])

# (3) Crossformer seg_len / win_size configurable (hardcoded 12 / 2 upstream)
patch("src/models/Crossformer.py", [
    (r"self\.seg_len = 12",
     "self.seg_len = getattr(configs, 'seg_len', 12)",
     "getattr(configs, 'seg_len'", "seg_len from configs"),
    (r"self\.win_size = 2",
     "self.win_size = getattr(configs, 'win_size', 2)",
     "getattr(configs, 'win_size'", "win_size from configs"),
])

# (4) expose --seg_len / --win_size on the CLI
patch("run.py", [
    (r"(\n\s*parser\.add_argument\('--patch_len'[^\n]*\n)",
     r"\1    parser.add_argument('--seg_len', type=int, default=4, help='Crossformer segment length')\n"
     r"    parser.add_argument('--win_size', type=int, default=2, help='Crossformer merge window')\n",
     "'--seg_len'", "add --seg_len/--win_size"),
])

# (5) stop run.py from clobbering --stride with --patch_len
patch("run.py", [
    (r"\n(\s*)args\.stride = args\.patch_len",
     r"\n\1# args.stride = args.patch_len  # PATCHED OUT: silently ignored the --stride flag",
     "PATCHED OUT: silently ignored", "keep --stride as passed"),
])

# (6) PatchTST must read the CLI values instead of its 16/8 signature defaults
patch("src/models/PatchTST.py", [
    (r"def __init__\(self, configs, patch_len=16, stride=8\):",
     "def __init__(self, configs, patch_len=None, stride=None):\n"
     "        patch_len = getattr(configs, 'patch_len', 16) if patch_len is None else patch_len\n"
     "        stride = getattr(configs, 'stride', 8) if stride is None else stride",
     "patch_len=None, stride=None", "patch_len/stride from configs"),
])

# (7) drop the per-epoch test evaluation (bias hazard + ~24k extra bs=1 forwards per epoch)
patch("src/exp/exp_MTS_forecasting.py", [
    (r"\n(\s*)test_loss, test_mse = self\.vali\(test_data, test_loader[^\n]*\n",
     r"\n\1test_loss, test_mse = float('nan'), float('nan')  # PATCHED: no per-epoch test eval\n",
     "PATCHED: no per-epoch test eval", "remove per-epoch test eval"),
])
print("### patches applied")
PYEOF

# ---------------------------------------------------------------- 5. verify
cd "$TF_HARNESS"
"$TF_ENV_DIR/bin/python" - <<'PYEOF'
import sys; sys.path.insert(0, ".")
from data.data_provider.data_factory import data_dict
assert "stock_pooled" in data_dict, "loader did not register"
from src.exp.exp_basic import Exp_Basic
print("### verified: stock_pooled registered; models available")
PYEOF

cat <<EOF

### setup complete.
Next, a mini run on the debug queue BEFORE submitting the arrays (gpu-debug caps at 30 min):

  sbatch --export=ALL,MODEL=Crossformer,SMOKE=1 --partition=gpu-debug --time=00:25:00 \\
         --array=1 scripts/transformer_bench/anvil_transformer.sb

Check slurm_logs/ for a completed epoch and a written test_index.csv, then launch the full arrays.
EOF
