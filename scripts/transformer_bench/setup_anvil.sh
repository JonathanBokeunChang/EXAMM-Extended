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
# `module` itself (Lmod) only exists on HPC login/compute nodes -- on a local machine (e.g. running
# this to validate the pipeline or use Apple MPS before GPU hours land) there is no module system at
# all, and that is not an error: system python3 is already on PATH.
if command -v module >/dev/null 2>&1; then
  module --force purge
  module load modtree/gpu 2>/dev/null || echo "WARN: modtree/gpu unavailable (fine on a login node)"
  # The venv's interpreter links against whichever python module is loaded here, so the batch job
  # must load the SAME one or it may fail to find libpython. Record the name for anvil_transformer.sb.
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
else
  echo "### no 'module' command -- assuming local/non-Lmod environment, using system python3"
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
  "$TF_ENV_DIR/bin/pip" install --quiet numpy pandas scikit-learn einops matplotlib
  # timm is needed only for TemporalDeformAttention.py's `trunc_normal_` import, but timm's own
  # __init__ eagerly imports its full layers tree, which pulls in torchvision. Installing timm
  # plain (no --index-url) pulls torchvision's torch dependency from DEFAULT PyPI -- a DIFFERENT
  # index than the cu118 one torch itself came from -- which silently drags in a second,
  # mismatched torch build (observed: nvidia-nccl-cu13, i.e. CUDA 13, alongside our pinned CUDA
  # 11.8 build). Fix: install timm with --no-deps (keep our torch), then install torchvision
  # explicitly from the SAME cu118 index so it resolves against the torch already present instead
  # of fetching its own.
  "$TF_ENV_DIR/bin/pip" install --quiet --no-deps timm
  "$TF_ENV_DIR/bin/pip" install --quiet torchvision --index-url https://download.pytorch.org/whl/cu118
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

# (8) expose --attn_dropout / --head_dropout, defaulting to None ("use --dropout")
patch("run.py", [
    (r"(\n\s*parser\.add_argument\('--layer_dropout'[^\n]*\n)",
     r"\1    parser.add_argument('--attn_dropout', type=float, default=None,\n"
     r"                        help='attention dropout; None => use --dropout (PatchTST paper uses 0)')\n"
     r"    parser.add_argument('--head_dropout', type=float, default=None,\n"
     r"                        help='prediction-head dropout; None => use --dropout (PatchTST paper uses 0)')\n",
     "'--attn_dropout'", "add --attn_dropout/--head_dropout"),
])

# (9) PatchTST: separate attention and head dropout from --dropout.
#     This file is a REIMPLEMENTATION and wired both to configs.dropout. The authors do neither --
#     PatchTST_supervised/models/PatchTST.py hardcodes attn_dropout=0. and never reads it from
#     configs, and the ETTh1/ETTh2 scripts leave head_dropout=0. At --dropout 0.3 that is 0.3 applied
#     at two sites the published model leaves clean. The head site is the serious one: FlattenHead
#     applies its dropout AFTER the linear, so at pred_len=1 it zeroes the model's single prediction
#     on ~p of training samples. Both fall back to configs.dropout when the flags are absent, so the
#     entire pre-existing shared-protocol tree keeps its original semantics.
patch("src/models/PatchTST.py", [
    (r"(\n\s*padding = stride\n)",
     r"\1        attn_dropout = getattr(configs, 'attn_dropout', None)\n"
     r"        attn_dropout = configs.dropout if attn_dropout is None else attn_dropout\n"
     r"        head_dropout = getattr(configs, 'head_dropout', None)\n"
     r"        head_dropout = configs.dropout if head_dropout is None else head_dropout\n",
     "attn_dropout = getattr", "derive attn/head dropout"),
    (r"attention_dropout=configs\.dropout",
     "attention_dropout=attn_dropout",
     "attention_dropout=attn_dropout", "attention dropout from --attn_dropout"),
    (r"head_dropout=configs\.dropout",
     "head_dropout=head_dropout",
     "head_dropout=head_dropout", "head dropout from --head_dropout"),
])

# (7) drop the per-epoch test evaluation (bias hazard + ~24k extra bs=1 forwards per epoch)
patch("src/exp/exp_MTS_forecasting.py", [
    (r"\n(\s*)test_loss, test_mse = self\.vali\(test_data, test_loader[^\n]*\n",
     r"\n\1test_loss, test_mse = float('nan'), float('nan')  # PATCHED: no per-epoch test eval\n",
     "PATCHED: no per-epoch test eval", "remove per-epoch test eval"),
])

# (8) MPS fallback for local (no-CUDA) runs, e.g. Apple Silicon -- OFF BY DEFAULT, opt in with
#     EXAMM_TF_USE_MPS=1. run.py only ever checked torch.cuda.is_available(), so on a Mac
#     args.use_gpu silently went False and every local run used plain CPU even though
#     torch.backends.mps.is_available() is True. Tried enabling it unconditionally first: on this
#     torch 2.8.0 build BOTH Crossformer and PatchTST abort mid-first-epoch with
#     "MPSNDArray ... buffer is not large enough" (a backend bug in their
#     unfold/windowing ops, not a config issue -- reproduced on two unrelated models). Silently
#     mixing a crashing accelerator into a benchmark whose whole point is identical settings across
#     models is worse than plain CPU, so this stays opt-in until verified stable per-model.
#     On Anvil torch.cuda.is_available() is True so this branch is never taken either way.
patch("run.py", [
    (r"args\.use_gpu = True if torch\.cuda\.is_available\(\) and args\.use_gpu else False",
     "args.use_gpu = True if (torch.cuda.is_available() or (__import__('os').environ.get("
     "'EXAMM_TF_USE_MPS') == '1' and hasattr(torch.backends, 'mps') "
     "and torch.backends.mps.is_available())) and args.use_gpu else False",
     "EXAMM_TF_USE_MPS", "allow opt-in MPS in addition to CUDA"),
])
patch("src/exp/exp_basic.py", [
    (r"    def _acquire_device\(self\):\n"
     r"        if self\.args\.use_gpu:\n"
     r"            os\.environ\[\"CUDA_VISIBLE_DEVICES\"\] = str\(\n"
     r"                self\.args\.gpu\) if not self\.args\.use_multi_gpu else self\.args\.devices\n"
     r"            device = torch\.device\('cuda:\{\}'\.format\(self\.args\.gpu\)\)\n"
     r"            print\('Use GPU: cuda:\{\}'\.format\(self\.args\.gpu\)\)\n"
     r"        else:\n"
     r"            device = torch\.device\('cpu'\)\n"
     r"            print\('Use CPU'\)\n"
     r"        return device\n",
     "    def _acquire_device(self):\n"
     "        if self.args.use_gpu and torch.cuda.is_available():\n"
     "            os.environ[\"CUDA_VISIBLE_DEVICES\"] = str(\n"
     "                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices\n"
     "            device = torch.device('cuda:{}'.format(self.args.gpu))\n"
     "            print('Use GPU: cuda:{}'.format(self.args.gpu))\n"
     "        elif self.args.use_gpu and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():\n"
     "            device = torch.device('mps')\n"
     "            print('Use GPU: mps')\n"
     "        else:\n"
     "            device = torch.device('cpu')\n"
     "            print('Use CPU')\n"
     "        return device\n",
     "Use GPU: mps", "acquire mps device when CUDA is unavailable"),
])
print("### patches applied")

# (10) add lradj 'type3', and make an unknown lradj fail loudly.
#      PatchTST's published ETTh1/ETTh2 runs inherit lradj=type3 from run_longExp.py, but this
#      harness implements only type1/type2/cosine AND has no else branch -- so --lradj type3 raised
#      "UnboundLocalError: lr_adjust" several minutes into training instead of saying what was
#      wrong. Formula copied from the PatchTST authors' utils/tools.py: flat for the first two
#      epochs, then x0.9 per epoch. The else branch converts any future typo into a clear message.
patch("src/utils/tools.py", [
    (r"(\n(\s*)elif args\.lradj == \"cosine\":)",
     "\\n\\2elif args.lradj == 'type3':\\n"
     "\\2    lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}\\1",
     "'type3'", "add lradj type3"),
    (r"(\n(\s*)if epoch in lr_adjust\.keys\(\):)",
     "\\n\\2else:\\n"
     "\\2    raise ValueError('unknown --lradj ' + str(args.lradj) + '; supported: type1, type2, type3, cosine')\\1",
     "unknown --lradj", "fail loudly on unknown lradj"),
])

PYEOF

# ---------------------------------------------------------------- 5. verify
cd "$TF_HARNESS"
# ---- vendor the PatchTST authors' own implementation as MODEL=PatchTSTOfficial.
# MUST run after patch (8) above: the installer anchors its argparse insert on the --head_dropout
# line that (8) creates. It is idempotent and checksum-verifies the upstream files.
# Absolute paths: this script has already cd'd into $TF_HARNESS by now, so a path relative to
# ${BASH_SOURCE[0]} resolves against the WRONG directory and the installers are silently not found.
PYTHON="$TF_ENV_DIR/bin/python" \
  bash "$REPO_ROOT/scripts/transformer_bench/install_patchtst_official.sh" "$TF_HARNESS"
PYTHON="$TF_ENV_DIR/bin/python" \
  bash "$REPO_ROOT/scripts/transformer_bench/install_crossformer_official.sh" "$TF_HARNESS"
# ---- vendor the iTransformer authors' own model as MODEL=ITransformerOfficial.
# Unlike the two above this vendors ONLY the model file: every layer class it uses is AST-identical
# between thuml/iTransformer and this harness, and the installer asserts that at install time rather
# than assuming it. It also adds --use_norm and --class_strategy to run.py, which the authors' model
# reads and this harness's argparse does not define.
PYTHON="$TF_ENV_DIR/bin/python" \
  bash "$REPO_ROOT/scripts/transformer_bench/install_itransformer_official.sh" "$TF_HARNESS"
# ---- vendor the DLinear authors' own model as MODEL=DLinearOfficial.
# THIS WAS MISSING and the omission was silent: install_dlinear_official.sh existed and was never
# called, so a pod built by this script had no DLinearOfficial and PROFILE=author would refuse it
# at run time rather than at setup. src/models/DLinear.py is a MODIFIED port -- it trains from the
# authors' constant weight-init lines, which they ship commented out -- so it cannot stand in.
PYTHON="$TF_ENV_DIR/bin/python" \
  bash "$REPO_ROOT/scripts/transformer_bench/install_dlinear_official.sh" "$TF_HARNESS"

"$TF_ENV_DIR/bin/python" - <<'PYEOF'
import sys; sys.path.insert(0, ".")
from data.data_provider.data_factory import data_dict
assert "stock_pooled" in data_dict, "loader did not register"
from src.exp.exp_basic import Exp_Basic            # must import; registry lives here
# This line previously read:
#     assert "PatchTSTOfficial" in Exp_Basic.__init__.__doc__ or True
# which ALWAYS raises TypeError: __init__ has no docstring, so __doc__ is None, and Python
# evaluates the `in` before it can ever reach `or True`. Under `set -e` that aborted setup at the
# very last step -- after every install had in fact succeeded -- so the script exited non-zero and
# never printed "setup complete", making a healthy build look broken. Replaced with a check that
# actually means something: every model this study runs must import.
for _m in ("PatchTSTOfficial", "CrossformerOfficial", "ITransformerOfficial", "DeformTime"):
    __import__("src.models." + _m)
import src.layers.PatchTST_backbone as _b
assert "BatchNorm" in open(_b.__file__).read(), "authors' backbone missing BatchNorm -- wrong file?"
import src.models.ITransformerOfficial as _i
assert "configs.use_norm" in open(_i.__file__).read(), "iTransformer ignores use_norm -- wrong file?"
print("### verified: stock_pooled registered; all four authors' models importable")
PYEOF

cat <<EOF

### setup complete.
Next, a mini run on the debug queue BEFORE submitting the arrays (gpu-debug caps at 30 min):

  sbatch --export=ALL,MODEL=Crossformer,SMOKE=1 --partition=gpu-debug --time=00:25:00 \\
         --array=1 scripts/transformer_bench/anvil_transformer.sb

Check slurm_logs/ for a completed epoch and a written test_index.csv, then launch the full arrays.
EOF
