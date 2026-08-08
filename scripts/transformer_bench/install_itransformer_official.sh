#!/bin/bash
# Vendor the iTransformer AUTHORS' OWN model file into the benchmark harness, as a model named
# `ITransformerOfficial`, alongside (never replacing) the harness's existing `iTransformer`.
#
#   bash scripts/transformer_bench/install_itransformer_official.sh [TF_HARNESS_DIR]
#
# HOW THIS DIFFERS FROM THE OTHER TWO INSTALLERS. For PatchTST and Crossformer the bundled files
# were third-party reimplementations that differ from the published models in ways no flag reaches,
# so both installers vendor the authors' model AND its private layer stack. iTransformer is not that
# case: thuml wrote both iTransformer and Time-Series-Library, and every layer class the model uses
# -- Encoder, EncoderLayer, FullAttention, AttentionLayer, DataEmbedding_inverted -- is AST-identical
# between thuml/iTransformer and the copy already in this harness (verified below, at install time).
#
# So this installer vendors ONLY model/iTransformer.py and points it at the harness's existing
# layers. That is deliberate: vendoring a second, byte-identical copy of the layer stack would add
# no fidelity and would drag in reformer_pytorch, which the authors' SelfAttention_Family imports
# for a class iTransformer never instantiates.
#
# THE GUARD THAT MAKES THAT SAFE. Because the vendored model imports layers we did NOT vendor, a
# future edit to the harness's layers would silently change what "the authors' iTransformer"
# computes. This installer therefore downloads the authors' layer files too and asserts that every
# class the model actually uses is AST-identical to the harness's. It does not install them -- it
# only compares. If they ever diverge, this fails loudly rather than training a different model.
#
# WHY A NEW NAME: the harness's own iTransformer.py is a faithful port, but it hardcodes the
# Non-stationary normalization ON rather than reading configs.use_norm, and drops class_strategy and
# the output_attention return path. At the published defaults (use_norm=True, output_attention=False)
# the two compute the same function -- but "the same at today's defaults" is not the same as "the
# authors' code", and only one of those is citable. Adding a second model keeps any existing
# iTransformer row reproducible and lets the two be compared on identical data, seeds and protocol.
#
# The upstream file is used VERBATIM apart from its `from layers.` imports, which are repointed at
# this harness's layer package. Its pre-rewrite sha256 is asserted, so an upstream change or a
# truncated download fails here rather than silently training a different model.
#
# Idempotent: safe to re-run.
set -euo pipefail

SHA=c2426e68ca13f74aaec08045c5c724d8ad328124   # 2025-07-17, thuml/iTransformer
BASE="https://raw.githubusercontent.com/thuml/iTransformer/$SHA"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TF_HARNESS="${1:-${TF_HARNESS:-$REPO/external/DeformTime}}"
[ -d "$TF_HARNESS/src/models" ] || { echo "ERROR: $TF_HARNESS is not the harness (no src/models)" >&2; exit 1; }
[ -d "$TF_HARNESS/src/layers" ] || { echo "ERROR: $TF_HARNESS has no src/layers" >&2; exit 1; }

MODEL_SHA=6f34777a5a12c253a293f79e7e1fd5b6f79ac100b443951e051e269e7e2542db

sha_of() { shasum -a 256 "$1" 2>/dev/null | awk '{print $1}' || sha256sum "$1" | awk '{print $1}'; }

TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT

echo "==> fetching authors' iTransformer @ ${SHA:0:8}"
curl -fsSL "$BASE/model/iTransformer.py" -o "$TMP/iTransformer.py"
got=$(sha_of "$TMP/iTransformer.py")
[ "$got" = "$MODEL_SHA" ] || {
  echo "ERROR: checksum mismatch for model/iTransformer.py" >&2
  echo "       expected $MODEL_SHA" >&2
  echo "       got      $got" >&2
  exit 1; }
echo "    [ok]   model/iTransformer.py  sha256 verified"

# ---------------------------------------------------------------------------
# GUARD: the model imports layers we are NOT vendoring, so those layers must be
# the authors' own code. Compare by AST (not bytes) so that comment or blank-line
# drift does not trip it, while any change to executable structure does.
# ---------------------------------------------------------------------------
echo "==> verifying the harness's layers are the authors' own"
for f in Transformer_EncDec SelfAttention_Family Embed; do
  curl -fsSL "$BASE/layers/$f.py" -o "$TMP/$f.py"
done

python3 - "$TMP" "$TF_HARNESS/src/layers" <<'PY'
import ast, hashlib, sys, os
tmp, harness = sys.argv[1], sys.argv[2]
NEEDED = {
    "Transformer_EncDec.py":   {"Encoder", "EncoderLayer"},
    "SelfAttention_Family.py": {"FullAttention", "AttentionLayer"},
    "Embed.py":                {"DataEmbedding_inverted"},
}
def digest(path, names):
    out = {}
    if not os.path.exists(path):
        return out
    for n in ast.walk(ast.parse(open(path).read())):
        if isinstance(n, ast.ClassDef) and n.name in names:
            out[n.name] = hashlib.sha256(ast.unparse(n).encode()).hexdigest()
    return out
bad = False
for fname, names in NEEDED.items():
    a = digest(os.path.join(tmp, fname), names)
    h = digest(os.path.join(harness, fname), names)
    for c in sorted(names):
        if c not in a:
            print(f"    [FAIL] {c}: not found in authors' {fname}"); bad = True
        elif c not in h:
            print(f"    [FAIL] {c}: not found in harness {fname}"); bad = True
        elif a[c] != h[c]:
            print(f"    [FAIL] {c} in {fname} DIFFERS between authors' repo and harness")
            print(f"           authors {a[c][:16]}  harness {h[c][:16]}")
            print(f"           the vendored model would not be the authors' model -- refusing")
            bad = True
        else:
            print(f"    [ok]   {c:24s} AST-identical to authors' {fname}")
if bad:
    sys.exit("ERROR: harness layers diverge from thuml/iTransformer; vendor them explicitly "
             "before installing, or this model is not what it claims to be.")
PY

# ---------------------------------------------------------------------------
# Install: verbatim except the import lines, which are repointed at the harness.
# ---------------------------------------------------------------------------
DEST="$TF_HARNESS/src/models/ITransformerOfficial.py"
echo "==> installing -> ${DEST#$REPO/}"
{
  cat <<'HDR'
"""iTransformer, the AUTHORS' implementation (github.com/thuml/iTransformer @ c2426e68).

Installed by scripts/transformer_bench/install_itransformer_official.sh. This file is the
upstream model/iTransformer.py VERBATIM apart from the three `from layers.` imports below,
which are repointed at this harness's layer package.

The layers are NOT vendored because every class this model uses is AST-identical between
thuml/iTransformer and this harness; the installer asserts that at install time and refuses
to proceed if they ever diverge. See that script's header for why.

Distinct from src/models/iTransformer.py, which is a faithful port that hardcodes the
Non-stationary normalization on instead of reading configs.use_norm. Equivalent at the
published defaults; not the same code.
"""
HDR
  sed -e 's/^from layers\./from src.layers./' "$TMP/iTransformer.py"
} > "$DEST"

python3 - "$DEST" <<'PY'
import ast, sys
src = open(sys.argv[1]).read()
tree = ast.parse(src)                             # must still be valid Python
# Check the IMPORT STATEMENTS, not raw text: this file's own docstring quotes the
# string "from layers." when explaining the rewrite, which a substring test would
# flag as an unrewritten import.
mods = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module]
stale = [m for m in mods if m.startswith("layers.")]
assert not stale, f"unrewritten imports survived: {stale}"
rewritten = [m for m in mods if m.startswith("src.layers.")]
assert len(rewritten) == 3, f"expected 3 rewritten imports, got {len(rewritten)}: {rewritten}"
assert any(isinstance(n, ast.ClassDef) and n.name == "Model" for n in ast.walk(tree)), "Model missing"
print(f"    [ok]   imports rewritten ({len(rewritten)}), file parses, Model present")
PY

# ---------------------------------------------------------------------------
# Register in the harness's model dictionary.
# ---------------------------------------------------------------------------
EXP="$TF_HARNESS/src/exp/exp_basic.py"
if grep -q "ITransformerOfficial" "$EXP"; then
  echo "==> exp_basic.py already registers ITransformerOfficial"
else
  echo "==> registering ITransformerOfficial in ${EXP#$REPO/}"
  python3 - "$EXP" <<'PY'
import ast, re, sys
p = sys.argv[1]; s = open(p).read()

# Append to the `from src.models import ...` list exactly once. Doing this with a
# single anchored substitution rather than two independent edits: an earlier version
# ran both a literal replace and a regex append, and both matched, emitting
# "ITransformerOfficial, ITransformerOfficial" -- legal Python, silently wrong.
def already_imported(src):
    for n in ast.walk(ast.parse(src)):
        if isinstance(n, ast.ImportFrom) and n.module == "src.models":
            return any(a.name == "ITransformerOfficial" for a in n.names)
    return False

if not already_imported(s):
    s, k = re.subn(r"(from src\.models import [^\n]+)", r"\1, ITransformerOfficial", s, count=1)
    assert k == 1, "could not find the 'from src.models import ...' line"

if "'ITransformerOfficial':" not in s:
    s, k = re.subn(r"(\n(\s*)'iTransformer': iTransformer,)",
                   r"\1\n\2'ITransformerOfficial': ITransformerOfficial,", s, count=1)
    assert k == 1, "could not find the 'iTransformer' model_dict entry to anchor to"

ast.parse(s)                                   # must still be valid Python
open(p, "w").write(s)
assert already_imported(s), "import not present after patch"
assert s.count("'ITransformerOfficial': ITransformerOfficial,") == 1, "duplicate model_dict entry"
assert s.count("ITransformerOfficial, ITransformerOfficial") == 0, "duplicate import"
print("    [ok]   model_dict entry added (import + registry, exactly once)")
PY
fi

# ---------------------------------------------------------------------------
# The authors' model reads two configs the harness's argparse does not define.
# Add them with the AUTHORS' OWN defaults (run.py: use_norm=True, class_strategy='projection').
# ---------------------------------------------------------------------------
RUNPY="$TF_HARNESS/run.py"
python3 - "$RUNPY" <<'PY'
import re, sys
p = sys.argv[1]; s = open(p).read(); added = []
if "'--use_norm'" not in s:
    s = re.sub(r"(\n\s*parser\.add_argument\('--factor')",
               "\n    parser.add_argument('--use_norm', type=int, default=1, "
               "help='use Non-stationary normalization; thuml/iTransformer run.py default True')"
               r"\1", s, count=1)
    added.append("--use_norm")
if "'--class_strategy'" not in s:
    s = re.sub(r"(\n\s*parser\.add_argument\('--factor')",
               "\n    parser.add_argument('--class_strategy', type=str, default='projection', "
               "help='thuml/iTransformer run.py default')"
               r"\1", s, count=1)
    added.append("--class_strategy")
if added:
    open(p, "w").write(s)
    print(f"    [ok]   run.py: added {', '.join(added)}")
else:
    print("    [ok]   run.py already defines --use_norm and --class_strategy")
PY

echo
echo "==> iTransformer (authors') installed as model 'ITransformerOfficial'"
echo "    published config, thuml/iTransformer @ ${SHA:0:8}"
echo "    scripts/multivariate_forecasting/ETT/iTransformer_ETTh1.sh (seq_len 96 -- NATIVE):"
echo "      --seq_len 96 --e_layers 2 --d_model 256 --d_ff 256"
echo "    everything else from run.py argparse defaults:"
echo "      --n_heads 8 --d_layers 1 --factor 1 --dropout 0.1 --activation gelu"
echo "      --embed timeF --use_norm 1 --class_strategy projection"
echo "      lr 1e-4 | batch 32 | 10 epochs | patience 3 | lradj type1"
