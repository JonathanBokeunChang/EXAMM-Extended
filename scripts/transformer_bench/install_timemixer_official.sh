#!/bin/bash
# Vendor the TimeMixer AUTHORS' OWN model into the benchmark harness as model `TimeMixer`.
#
#   bash scripts/transformer_bench/install_timemixer_official.sh [TF_HARNESS_DIR]
#
# WHY TIMEMIXER AND NOT TIMEMIXER++. TimeMixer++ has no runnable reference implementation: it is
# absent from thuml/Time-Series-Library (issue #839, March 2026), the authors released only the
# arXiv paper, and the sole third-party port targets the IMPUTATION task, not forecasting. A
# reimplementation could not be validated against anything, so a weak result would be
# unattributable -- architecture or my code, with no way to tell. TimeMixer (ICLR 2024) has official
# code and is what this installs.
#
# WHY IT IS WORTH A SLOT. Every other baseline here is attention-based (PatchTST, iTransformer,
# DeformTime, Crossformer) or linear (DLinear). TimeMixer is MLP-based multiscale decomposition --
# a third architectural family, and the only model in the lineup that explicitly mixes several
# temporal resolutions.
#
# THIS INSTALLER VENDORS THE LAYER STACK, unlike install_itransformer_official.sh which reuses the
# harness's. That is not a style choice; the AST guard below was written to reuse them and FAILED:
#
#   TimeFeatureEmbedding  -- harness freq_map lacks upstream's 'ms': 7 key
#   DataEmbedding_wo_pos  -- harness forward() lacks upstream's `if x is None and x_mark is not
#                            None: return self.temporal_embedding(x_mark)` branch
#
# The second one is reachable code in this model: TimeMixer calls enc_embedding(None, x_mark_dec)
# when use_future_temporal_feature is on. We run with it off, so the harness copy would compute the
# same function today -- but "the same at today's flags" is not "the authors' model", and a future
# flag change would silently alter it. series_decomp and moving_avg ARE AST-identical; they are
# vendored anyway so the whole stack has one provenance.
#
# The four upstream files are used VERBATIM apart from `from layers.` imports, repointed at the
# private package. Their pre-rewrite sha256 values are asserted, so an upstream change or truncated
# download fails here rather than silently training a different model.
#
# Idempotent: safe to re-run.
set -euo pipefail

SHA=e24610583b36fdd8c76cc17a8df4e65759a5f460   # kwuking/TimeMixer, main
BASE="https://raw.githubusercontent.com/kwuking/TimeMixer/$SHA"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TF_HARNESS="${1:-${TF_HARNESS:-$REPO/external/DeformTime}}"
[ -d "$TF_HARNESS/src/models" ] || { echo "ERROR: $TF_HARNESS is not the harness (no src/models)" >&2; exit 1; }
[ -d "$TF_HARNESS/src/layers" ] || { echo "ERROR: $TF_HARNESS has no src/layers" >&2; exit 1; }

# path-in-repo : sha256 of the file as published
FILES=(
  "models/TimeMixer.py:817d62f4aac54c8566e560f6d3785856e31c8ee51460279ee3d0a4823f11d4be"
  "layers/StandardNorm.py:cc1c0bc65b7b094bbe83f988fb05b86272a59638c030c3781aa52ce8880379df"
  "layers/Embed.py:ab492ea2f68459bbcf3cbffdd1beb75b24d0d70248d017a313a3b470316aaa2b"
  "layers/Autoformer_EncDec.py:48745b4bb647355e9845792a855df9c59fd7df7fcc664c765351fec390c4073e"
)

sha_of() { shasum -a 256 "$1" 2>/dev/null | awk '{print $1}' || sha256sum "$1" | awk '{print $1}'; }

TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT

echo "==> fetching authors' TimeMixer @ ${SHA:0:8}"
for spec in "${FILES[@]}"; do
  rel="${spec%%:*}"; want="${spec##*:}"; base=$(basename "$rel")
  curl -fsSL "$BASE/$rel" -o "$TMP/$base"
  got=$(sha_of "$TMP/$base")
  [ "$got" = "$want" ] || {
    echo "ERROR: checksum mismatch for $rel" >&2
    echo "       expected $want" >&2
    echo "       got      $got" >&2
    echo "       upstream moved -- re-derive the pins before running anything" >&2
    exit 1; }
  echo "    [ok]   $rel  sha256 verified"
done

# ---------------------------------------------------------------------------
# Install the private layer package. Verbatim; these three files import nothing
# but torch/math, so no rewriting is needed inside them.
# ---------------------------------------------------------------------------
PKG="$TF_HARNESS/src/models/_timemixer_official"
echo "==> installing layer stack -> ${PKG#$REPO/}/"
mkdir -p "$PKG"
cat > "$PKG/__init__.py" <<'EOF'
"""The TimeMixer authors' own layer stack (github.com/kwuking/TimeMixer @ e2461058), vendored
verbatim by scripts/transformer_bench/install_timemixer_official.sh.

Private to TimeMixer rather than shared with src/layers/: two classes the model uses differ from
the harness copies (TimeFeatureEmbedding's freq_map, DataEmbedding_wo_pos's x-is-None branch), and
the model must be the authors' code regardless of how the harness's own layers drift.
"""
EOF
for f in StandardNorm.py Embed.py Autoformer_EncDec.py; do
  cp "$TMP/$f" "$PKG/$f"
  echo "    [ok]   $f"
done

# ---------------------------------------------------------------------------
# Install the model: verbatim except the three `from layers.` imports.
# ---------------------------------------------------------------------------
DEST="$TF_HARNESS/src/models/TimeMixer.py"
echo "==> installing model -> ${DEST#$REPO/}"
{
  cat <<'HDR'
"""TimeMixer, the AUTHORS' implementation (github.com/kwuking/TimeMixer @ e2461058).

"TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting", ICLR 2024.

Installed by scripts/transformer_bench/install_timemixer_official.sh. This file is the upstream
models/TimeMixer.py VERBATIM apart from the three `from layers.` imports below, repointed at the
authors' own layer stack vendored under _timemixer_official/.

RUN IT WITH channel_independence=0. At 1 the model builds a 1-channel embedding and reshapes the
decoder output to (B, c_out, pred_len); with enc_in=6 and c_out=1 that reshape raises
"shape '[B,1,1]' is invalid for input of size 6*B". 0 is therefore forced, not preferred -- and it
is also the setting this study wants, since it lets the five exogenous fields inform RET instead of
modelling each variate alone.

TASK NAME: this study passes long_term_forecast, matching the authors' ETTh1 script. forward()
treats long_term_forecast and short_term_forecast identically, so the choice is cosmetic; matching
their script keeps the deviation list shorter.

RUN IT WITH freq='h'. TimeFeatureEmbedding sizes its input projection from freq_map, and the pooled
loader always emits FOUR calendar features. freq='d' maps to 3 and fails with a 768x4-by-3x32 matmul
error. 'h' maps to 4. This selects a dimension; it is not a claim that the data is hourly.

OUTPUT IS 6-WIDE, AND THAT IS CORRECT. out_projection adds a [B,pred_len,1] seasonal projection to a
[B,pred_len,enc_in] per-channel trend, which broadcasts. Channel j is therefore "shared seasonal +
channel j's trend". The harness slices f_dim=-1, and the pooled loader puts RET last, so the scored
channel is RET's own trend plus the shared seasonal term, denormalised with RET's own statistics.
The loss is taken on that same channel, so training optimises exactly what is evaluated.
"""
HDR
  sed -e 's/^from layers\./from src.models._timemixer_official./' "$TMP/TimeMixer.py"
} > "$DEST"

python3 - "$DEST" <<'PY'
import ast, sys
src = open(sys.argv[1]).read()
tree = ast.parse(src)                              # must still be valid Python
# Inspect import NODES, not raw text: this file's own docstring would defeat a substring test.
mods = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module]
stale = [m for m in mods if m.startswith("layers.")]
assert not stale, f"unrewritten imports survived: {stale}"
rewritten = [m for m in mods if m.startswith("src.models._timemixer_official.")]
assert len(rewritten) == 3, f"expected 3 rewritten imports, got {len(rewritten)}: {rewritten}"
assert any(isinstance(n, ast.ClassDef) and n.name == "Model" for n in ast.walk(tree)), "Model missing"
print(f"    [ok]   imports rewritten ({len(rewritten)}), file parses, Model present")
PY

# ---------------------------------------------------------------------------
# Register in the harness's model dictionary.
# ---------------------------------------------------------------------------
EXP="$TF_HARNESS/src/exp/exp_basic.py"
if grep -q "'TimeMixer':" "$EXP"; then
  echo "==> exp_basic.py already registers TimeMixer"
else
  echo "==> registering TimeMixer in ${EXP#$REPO/}"
  python3 - "$EXP" <<'PY'
import ast, re, sys
p = sys.argv[1]; s = open(p).read()

def already_imported(src):
    for n in ast.walk(ast.parse(src)):
        if isinstance(n, ast.ImportFrom) and n.module == "src.models":
            return any(a.name == "TimeMixer" for a in n.names)
    return False

if not already_imported(s):
    s, k = re.subn(r"(from src\.models import [^\n]+)", r"\1, TimeMixer", s, count=1)
    assert k == 1, "could not find the 'from src.models import ...' line"

if "'TimeMixer':" not in s:
    s, k = re.subn(r"(\n(\s*)'DeformTime': DeformTime,)",
                   r"\1\n\2'TimeMixer': TimeMixer,", s, count=1)
    assert k == 1, "could not find the 'DeformTime' model_dict entry to anchor to"

ast.parse(s)                                       # must still be valid Python
open(p, "w").write(s)
assert already_imported(s), "import not present after patch"
assert s.count("'TimeMixer': TimeMixer,") == 1, "duplicate model_dict entry"
assert s.count("TimeMixer, TimeMixer") == 0, "duplicate import"
print("    [ok]   model_dict entry added (import + registry, exactly once)")
PY
fi

# ---------------------------------------------------------------------------
# The authors' model reads nine configs this harness's argparse does not define.
# Defaults are the AUTHORS' OWN, from kwuking/TimeMixer run.py.
# ---------------------------------------------------------------------------
RUNPY="$TF_HARNESS/run.py"
python3 - "$RUNPY" <<'PY'
import re, sys
p = sys.argv[1]; s = open(p).read(); added = []
NEW = [
    ("--task_name",       "str",   "'long_term_forecast'", "task name; TimeMixer dispatches on this"),
    ("--down_sampling_layers", "int", "0",  "number of downsampling levels"),
    ("--down_sampling_window", "int", "1",  "downsampling window size"),
    ("--down_sampling_method", "str", "None", "downsampling method: max, avg or conv"),
    ("--channel_independence", "int", "1",  "0 channel-dependent, 1 channel-independent"),
    ("--decomp_method",   "str",   "'moving_avg'", "series decomposition: moving_avg or dft_decomp"),
    ("--top_k",           "int",   "5",     "top-k for DFT decomposition"),
    ("--num_class",       "int",   "0",     "classification head width; unused for forecasting"),
    ("--use_future_temporal_feature", "int", "0", "use future temporal features"),
]
for flag, typ, default, helptext in NEW:
    if f"'{flag}'" in s:
        continue
    # Anchor on --factor, the same anchor install_itransformer_official.sh uses. Insert BEFORE it so
    # repeated inserts stay ordered and none of them depend on each other's presence.
    s, k = re.subn(r"(\n\s*parser\.add_argument\('--factor')",
                   f"\n    parser.add_argument('{flag}', type={typ}, default={default}, "
                   f"help='{helptext}')" + r"\1", s, count=1)
    assert k == 1, f"could not anchor {flag} on --factor"
    added.append(flag)
if added:
    open(p, "w").write(s)
    print(f"    [ok]   run.py: added {len(added)} flags: {', '.join(added)}")
else:
    print("    [ok]   run.py already defines all nine TimeMixer flags")
PY

echo
echo "==> TimeMixer (authors') installed as model 'TimeMixer'"
echo "    kwuking/TimeMixer @ ${SHA:0:8} -- ICLR 2024"
echo "    THIS STUDY MUST PASS: --channel_independence 0 (1 crashes at c_out=1)"
echo "                          --freq h                (loader emits 4 calendar features)"
echo "                          --task_name long_term_forecast (their ETTh1 script)"
