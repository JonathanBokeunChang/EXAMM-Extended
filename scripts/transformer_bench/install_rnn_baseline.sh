#!/bin/bash
# Install the LSTM / GRU recurrent baselines into the benchmark harness.
#
#   bash scripts/transformer_bench/install_rnn_baseline.sh [TF_HARNESS_DIR]
#
# WHY THIS EXISTS AT ALL. src/models/RNNBaseline.py lived only in one laptop's working copy:
# external/ is gitignored, so the file was untracked, and setup_anvil.sh never created it. A pod
# built from a clean clone therefore had NO LSTM or GRU baseline and would reject MODEL=LSTMBaseline
# at argparse time -- which is how a fresh box silently loses an entire arm of the study. The model
# source now lives at scripts/transformer_bench/models/RNNBaseline.py, which IS tracked, and this
# script copies it in and registers it.
#
# UNLIKE THE OTHER INSTALLERS, this vendors OUR code, not an upstream author's. There is no commit
# to pin and no checksum to verify against a third party; the guarantee here is simply that the
# harness copy matches the tracked copy, which is asserted after copying.
#
# THE CONFIGURATION IS LYU ET AL.'S, not ours -- 2 hidden layers, hidden width = input size, 1000
# epochs, Adam at 1e-4, Xavier init -- so these rows cite prior work on this exact problem. See the
# model file's own docstring, and the LSTMBaseline|GRUBaseline branch in anvil_transformer.sb for
# the three deviations (patience, instance normalisation, and our horizon/field count).
#
# INSTANCE NORMALISATION IS THE POINT OF THIS ARM. The train_rnn baselines run through EXAMM's own
# pipeline and are therefore unnormalised, and their cross-sectional spread collapses to 0.000158
# against a daily common level of 0.001535 -- a ratio of 9.7, which closes the Algorithm 2 gate on
# every day of 9 of 28 cells. These carry per-window standardisation like every transformer in the
# study, and at ratio 0.51-0.60 they gate on ~190 of 250 days. The pair is the comparison.
#
# Idempotent: safe to re-run.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TF_HARNESS="${1:-${TF_HARNESS:-$REPO/external/DeformTime}}"
SRC="$REPO/scripts/transformer_bench/models/RNNBaseline.py"

[ -f "$SRC" ] || { echo "ERROR: $SRC missing -- the tracked model source is gone" >&2; exit 1; }
[ -d "$TF_HARNESS/src/models" ] || { echo "ERROR: $TF_HARNESS is not the harness (no src/models)" >&2; exit 1; }

DEST="$TF_HARNESS/src/models/RNNBaseline.py"
echo "==> installing -> ${DEST#$REPO/}"
cp "$SRC" "$DEST"

sha_of() { shasum -a 256 "$1" 2>/dev/null | awk '{print $1}' || sha256sum "$1" | awk '{print $1}'; }
[ "$(sha_of "$SRC")" = "$(sha_of "$DEST")" ] \
  || { echo "ERROR: copy does not match the tracked source" >&2; exit 1; }
echo "    [ok]   matches the tracked source"

python3 - "$DEST" <<'PY'
import ast, sys
tree = ast.parse(open(sys.argv[1]).read())
names = {n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)}
missing = {"LSTMModel", "GRUModel"} - names
assert not missing, f"model classes missing: {missing}"
print("    [ok]   LSTMModel and GRUModel present")
PY

# ---------------------------------------------------------------------------
# Register BOTH names. exp_basic maps a --model string to a module, and run.py
# instantiates <module>.Model(configs) -- so each name needs its own shim module
# exposing `Model`, rather than pointing both at RNNBaseline.
# ---------------------------------------------------------------------------
for pair in "LSTMBaseline:LSTMModel" "GRUBaseline:GRUModel"; do
  name="${pair%%:*}"; cls="${pair##*:}"
  cat > "$TF_HARNESS/src/models/${name}.py" <<EOF
"""$name -- thin alias so exp_basic can map a --model string to a Model class.

run.py builds the network as <module>.Model(configs), so LSTM and GRU need separate
modules even though they share one implementation. Written by
scripts/transformer_bench/install_rnn_baseline.sh; edit RNNBaseline.py, not this.
"""
from src.models.RNNBaseline import $cls as Model
EOF
  echo "    [ok]   ${name}.py -> $cls"
done

EXP="$TF_HARNESS/src/exp/exp_basic.py"
python3 - "$EXP" <<'PY'
import ast, re, sys
p = sys.argv[1]; s = open(p).read()
for name in ("LSTMBaseline", "GRUBaseline"):
    imported = any(isinstance(n, ast.ImportFrom) and n.module == "src.models"
                   and any(a.name == name for a in n.names) for n in ast.walk(ast.parse(s)))
    if not imported:
        s, k = re.subn(r"(from src\.models import [^\n]+)", r"\1, " + name, s, count=1)
        assert k == 1, "could not find the 'from src.models import ...' line"
    if f"'{name}':" not in s:
        # Optional trailing comma: a harness set up before the RNN arms has DeformTime as the last
        # dict entry with no comma, one set up after has a comma. Re-emitting the captured comma
        # after the new entry keeps both shapes valid Python.
        s, k = re.subn(r"(\n(\s*)'DeformTime': DeformTime)(,?)",
                       r"\1,\n\2'" + name + "': " + name + r"\3", s, count=1)
        assert k == 1, "could not find the 'DeformTime' model_dict entry to anchor to"
ast.parse(s)
open(p, "w").write(s)
for name in ("LSTMBaseline", "GRUBaseline"):
    assert s.count(f"'{name}': {name}") == 1, f"duplicate model_dict entry for {name}"
    assert s.count(f"{name}, {name}") == 0, f"duplicate import for {name}"
print("    [ok]   registered LSTMBaseline and GRUBaseline (exactly once each)")
PY

# The model reads configs.use_norm. install_itransformer_official.sh normally adds it; add it here
# too so this installer does not depend on the order the installers happen to run in.
python3 - "$TF_HARNESS/run.py" <<'PY'
import re, sys
p = sys.argv[1]; s = open(p).read()
if "'--use_norm'" in s:
    print("    [ok]   run.py already defines --use_norm")
else:
    s, k = re.subn(r"(\n\s*parser\.add_argument\('--factor')",
                   "\n    parser.add_argument('--use_norm', type=int, default=1, "
                   "help='instance normalisation on the RNN baselines')" + r"\1", s, count=1)
    assert k == 1, "could not anchor --use_norm on --factor"
    open(p, "w").write(s)
    print("    [ok]   run.py: added --use_norm")
PY

echo
echo "==> LSTM/GRU baselines installed"
echo "    config: 2 layers, hidden = enc_in, Adam 1e-4, Xavier (Lyu et al.)"
echo "    run with: MODEL=LSTMBaseline or MODEL=GRUBaseline, PROFILE=author"
