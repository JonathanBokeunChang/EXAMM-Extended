#!/bin/bash
# Vendor the DLinear AUTHORS' OWN model file into the benchmark harness, as `DLinearOfficial`,
# alongside (never replacing) the harness's existing `DLinear`.
#
#   bash scripts/transformer_bench/install_dlinear_official.sh [TF_HARNESS_DIR]
#
# WHY A SEPARATE MODEL. src/models/DLinear.py is a modified port. Three differences from
# cure-lab/LTSF-Linear, one of which changes what gets trained:
#
#   1. WEIGHT INIT -- the material one. The authors leave the constant-init lines COMMENTED
#      OUT, with the note "Use this two lines if you want to visualize the weights". Constant
#      init is their visualization aid; they train from PyTorch's default random Kaiming init.
#      The harness port has those lines UNCOMMENTED, so it trains every seed from an identical
#      (1/seq_len)*ones start. DLinear is linear in its parameters, so the MSE objective is
#      convex and both inits share one global optimum -- but with train_epochs 10 and
#      patience 3 the run stops well before convergence, so where it starts still shows up in
#      where it stops. It also means seeds do nothing in the port beyond reshuffling batches,
#      which quietly makes a 10-seed "ensemble" far less diverse than it looks.
#   2. kernel_size -- authors hardcode 25; the port reads configs.moving_avg. Identical when
#      --moving_avg 25 is passed, which the author block in anvil_transformer.sb does.
#   3. forward() -- authors take (self, x); the harness calls model(x, x_mark, dec, dec_mark).
#      Bridged by the adapter below, not by editing upstream.
#
# WHAT IS NOT VENDORED, AND WHY THAT IS SAFE. The model needs moving_avg and series_decomp.
# Both are AST-identical between LTSF-Linear and this harness's Autoformer_EncDec.py, so the
# vendored file imports them rather than shipping a second copy. The installer asserts that
# identity at install time and refuses if they ever diverge -- the same guard used for
# install_itransformer_official.sh, and for the same reason: a model that imports code it
# does not ship can silently stop being the authors' model.
#
# Idempotent: safe to re-run.
set -euo pipefail

SHA=0c113668a3b88c4c4ee586b8c5ec3e539c4de5a6   # 2024-01-27, cure-lab/LTSF-Linear
BASE="https://raw.githubusercontent.com/cure-lab/LTSF-Linear/$SHA"
MODEL_SHA=0893b53cb6473d6bdca7aeca514cb3ee12efa6df227c29c4469571c9711451cc

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TF_HARNESS="${1:-${TF_HARNESS:-$REPO/external/DeformTime}}"
[ -d "$TF_HARNESS/src/models" ] || { echo "ERROR: $TF_HARNESS is not the harness" >&2; exit 1; }

sha_of() { shasum -a 256 "$1" 2>/dev/null | awk '{print $1}' || sha256sum "$1" | awk '{print $1}'; }
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT

echo "==> fetching authors' DLinear @ ${SHA:0:8}"
curl -fsSL "$BASE/models/DLinear.py" -o "$TMP/DLinear.py"
got=$(sha_of "$TMP/DLinear.py")
[ "$got" = "$MODEL_SHA" ] || {
  echo "ERROR: checksum mismatch for models/DLinear.py" >&2
  echo "       expected $MODEL_SHA" >&2
  echo "       got      $got" >&2
  exit 1; }
echo "    [ok]   models/DLinear.py  sha256 verified"

echo "==> verifying the harness's decomposition blocks are the authors' own"
python3 - "$TMP/DLinear.py" "$TF_HARNESS/src/layers/Autoformer_EncDec.py" <<'PY'
import ast, hashlib, sys, os
authors, harness = sys.argv[1], sys.argv[2]
def digest(path, names):
    if not os.path.exists(path): return {}
    return {n.name: hashlib.sha256(ast.unparse(n).encode()).hexdigest()
            for n in ast.walk(ast.parse(open(path).read()))
            if isinstance(n, ast.ClassDef) and n.name in names}
NEED = {"moving_avg", "series_decomp"}
a, h = digest(authors, NEED), digest(harness, NEED)
bad = False
for c in sorted(NEED):
    if c not in a:   print(f"    [FAIL] {c}: not in authors' file");  bad = True
    elif c not in h: print(f"    [FAIL] {c}: not in harness layers"); bad = True
    elif a[c] != h[c]:
        print(f"    [FAIL] {c} DIFFERS between LTSF-Linear and this harness"); bad = True
    else:
        print(f"    [ok]   {c:16s} AST-identical to the authors'")
if bad:
    sys.exit("ERROR: decomposition blocks diverge; vendor them explicitly or this is not "
             "the authors' model.")
PY

DEST="$TF_HARNESS/src/models/DLinearOfficial.py"
echo "==> installing -> ${DEST#$REPO/}"
{
  cat <<'HDR'
"""DLinear, the AUTHORS' implementation (github.com/cure-lab/LTSF-Linear @ 0c113668).

Installed by scripts/transformer_bench/install_dlinear_official.sh. The Model class below is
upstream models/DLinear.py VERBATIM -- in particular the constant-weight-init lines stay
COMMENTED OUT, as the authors left them, so training starts from PyTorch's random init.

Two things are added, both additive and neither touching the authors' logic:
  * series_decomp is imported from this harness's layers (AST-identical to upstream; the
    installer asserts that and refuses otherwise) instead of being redefined here.
  * a forward() shim, because the harness calls model(x, x_mark, dec, dec_mark) while the
    authors' forward takes x alone.

Distinct from src/models/DLinear.py, which UNCOMMENTS the constant init and reads
configs.moving_avg. See the installer header for why that difference matters.
"""
import torch
import torch.nn as nn
from src.layers.Autoformer_EncDec import series_decomp


HDR
  # Upstream verbatim, minus the moving_avg/series_decomp definitions (imported above) and
  # its own imports. The Model class -- including the commented-out init -- is untouched.
  python3 - "$TMP/DLinear.py" <<'PY'
import ast, sys
src = open(sys.argv[1]).read()
tree = ast.parse(src)
lines = src.split("\n")
for n in tree.body:
    if isinstance(n, ast.ClassDef) and n.name == "Model":
        print("\n".join(lines[n.lineno - 1: n.end_lineno]))
PY
  cat <<'SHIM'


# Signature bridge: the harness calls model(x, x_mark, dec, dec_mark); the authors' forward
# takes x alone. Done by REPLACING THE METHOD, not by subclassing. A subclass would have to be
# rebound to the name `Model`, and upstream's __init__ calls `super(Model, self).__init__()` --
# after rebinding, that expression resolves to the subclass's base (the original Model) and
# invokes it without `configs`, raising TypeError at construction. Patching the method leaves
# the class identity, and therefore that super() call, untouched.
_authors_forward = Model.forward


def _harness_forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
    return _authors_forward(self, x_enc)


Model.forward = _harness_forward
SHIM
} > "$DEST"

python3 - "$DEST" <<'PY'
import ast, sys, torch
from argparse import Namespace
src = open(sys.argv[1]).read()
tree = ast.parse(src)
assert any(isinstance(n, ast.ClassDef) and n.name == "Model" for n in ast.walk(tree)), "Model missing"
# The whole point of vendoring: the constant init must NOT be active.
active = [l for l in src.split("\n") if "nn.Parameter" in l and not l.strip().startswith("#")]
assert not active, f"constant weight init is ACTIVE -- this is the port, not the authors': {active}"
print("    [ok]   file parses, Model present, constant init correctly INACTIVE")
PY

EXP="$TF_HARNESS/src/exp/exp_basic.py"
if grep -q "DLinearOfficial" "$EXP"; then
  echo "==> exp_basic.py already registers DLinearOfficial"
else
  echo "==> registering DLinearOfficial in ${EXP#$REPO/}"
  python3 - "$EXP" <<'PY'
import ast, re, sys
p = sys.argv[1]; s = open(p).read()
def imported(src):
    for n in ast.walk(ast.parse(src)):
        if isinstance(n, ast.ImportFrom) and n.module == "src.models":
            return any(a.name == "DLinearOfficial" for a in n.names)
    return False
if not imported(s):
    s, k = re.subn(r"(from src\.models import [^\n]+)", r"\1, DLinearOfficial", s, count=1)
    assert k == 1, "could not find the 'from src.models import ...' line"
if "'DLinearOfficial':" not in s:
    s, k = re.subn(r"(\n(\s*)'DLinear': DLinear,)",
                   r"\1\n\2'DLinearOfficial': DLinearOfficial,", s, count=1)
    assert k == 1, "could not find the 'DLinear' model_dict entry to anchor to"
ast.parse(s)
open(p, "w").write(s)
assert s.count("'DLinearOfficial': DLinearOfficial,") == 1, "duplicate model_dict entry"
assert "DLinearOfficial, DLinearOfficial" not in s, "duplicate import"
print("    [ok]   model_dict entry added (import + registry, exactly once)")
PY
fi

echo
echo "==> DLinear (authors') installed as model 'DLinearOfficial'"
echo "    cure-lab/LTSF-Linear @ ${SHA:0:8}"
echo "    scripts/EXP-LookBackWindow/Linear_DiffWindow.sh -- the authors' OWN look-back sweep,"
echo "    which includes seq_len 96 (ETTh1 arm):"
echo "      --seq_len 96 --batch_size 8   (learning_rate unset -> run_longExp.py default 1e-4)"
echo "    everything else from their argparse defaults:"
echo "      train_epochs 10 | patience 3 | moving_avg 25 | individual False | lradj type1"
