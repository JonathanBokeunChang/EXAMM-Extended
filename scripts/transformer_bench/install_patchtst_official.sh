#!/bin/bash
# Vendor the PatchTST AUTHORS' OWN implementation into the benchmark harness, as a model named
# `PatchTSTOfficial`, alongside (never replacing) the harness's existing `PatchTST`.
#
#   bash scripts/transformer_bench/install_patchtst_official.sh [TF_HARNESS_DIR]
#
# WHY THIS EXISTS
# external/DeformTime is the DeformTime authors' repository (github.com/ClaudiaShu/DeformTime), so
# `DeformTime` there is genuinely their code. `PatchTST` and `Crossformer` in the same tree are NOT:
# they are Time-Series-Library reimplementations that the DeformTime authors bundled as baselines --
# recognisable by the "Paper link: https://arxiv.org/..." docstring, which is thuml's house style.
# The TSLib PatchTST differs from the published model in at least two ways no flag can reach:
#   * LayerNorm where the authors default to norm='BatchNorm'
#   * no residual attention, where the authors default to res_attention=True
# Running the authors' file removes that whole class of doubt for this model.
#
# WHY A NEW NAME AND NOT AN OVERWRITE
# Every PatchTST row currently in the paper was produced by the TSLib implementation. Overwriting it
# would make those rows unreproducible. Adding a second model keeps them intact and makes the two
# implementations directly A/B-able on the same data, same seeds, same protocol.
#
# WHAT IS AND IS NOT MODIFIED
# The four upstream files are used VERBATIM apart from four import lines (`layers.` -> `src.layers.`,
# because this harness nests its packages one level deeper). Their pre-rewrite sha256s are asserted
# below, so an upstream change or a truncated download fails here instead of silently training a
# different model. The harness adapter lives in its own file and does not touch the authors' code.
#
# Idempotent: safe to re-run. Every step is skipped if already applied and asserted afterwards.
set -euo pipefail

# Pinned to a specific commit, not a branch: `main` moving would silently change the model.
SHA=204c21efe0b39603ad6e2ca640ef5896646ab1a9   # 2023-08-11, yuqinie98/PatchTST
BASE="https://raw.githubusercontent.com/yuqinie98/PatchTST/$SHA/PatchTST_supervised"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TF_HARNESS="${1:-${TF_HARNESS:-$REPO/external/DeformTime}}"
[ -d "$TF_HARNESS/src/models" ] || { echo "ERROR: $TF_HARNESS is not the harness (no src/models)" >&2; exit 1; }

# sha256 of each file AS PUBLISHED, before our import rewrite.
read -r -d '' EXPECT <<'EOF' || true
49d8bb865e1226d6338842f4b33c4bc6992cefb048dc2a4a5a8c410ca708586b  models/PatchTST.py
df67173153787c2356bdfb6491159cd754332ef7382986efe879e1fbea8ebf26  layers/PatchTST_backbone.py
21c06c70a90c60ee2a269b5c600c702834dea22cdfd72915e6b0f8b4a28db3f6  layers/PatchTST_layers.py
e64c0ccded9228b347134e7368420d3fb10f70c75145b5ff2d8bdd8c3af59df6  layers/RevIN.py
EOF

sha_of() { shasum -a 256 "$1" 2>/dev/null | awk '{print $1}' || sha256sum "$1" | awk '{print $1}'; }

TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
echo "==> fetching authors' PatchTST @ ${SHA:0:8}"
while read -r want rel; do
  [ -n "${rel:-}" ] || continue
  curl -fsSL "$BASE/$rel" -o "$TMP/$(basename "$rel")"
  got=$(sha_of "$TMP/$(basename "$rel")")
  [ "$got" = "$want" ] || {
    echo "ERROR: checksum mismatch for $rel" >&2
    echo "       expected $want" >&2
    echo "       got      $got" >&2
    echo "       Upstream changed or the download truncated. Re-derive the pin before running." >&2
    exit 1; }
  echo "    [ok]   $rel  sha256 verified"
done <<< "$EXPECT"

# ---- place files. Only the import lines change; everything else is byte-identical to upstream.
install_rewritten() {  # $1=src basename  $2=dest path
  sed -e 's|^from layers\.|from src.layers.|' "$TMP/$1" > "$2"
}
install_rewritten PatchTST_layers.py   "$TF_HARNESS/src/layers/PatchTST_layers.py"
install_rewritten RevIN.py             "$TF_HARNESS/src/layers/RevIN.py"
install_rewritten PatchTST_backbone.py "$TF_HARNESS/src/layers/PatchTST_backbone.py"
install_rewritten PatchTST.py          "$TF_HARNESS/src/models/_patchtst_official.py"
echo "    [ok]   installed 4 files (imports rewritten: layers. -> src.layers.)"

# ---- adapter. Kept separate so the authors' file above stays clean and checksummable.
cat > "$TF_HARNESS/src/models/PatchTSTOfficial.py" <<'PYEOF'
"""PatchTST, the AUTHORS' implementation (github.com/yuqinie98/PatchTST @ 204c21ef).

Installed by scripts/transformer_bench/install_patchtst_official.sh. The authors' model lives in
_patchtst_official.py exactly as published (only its two import lines are rewritten for this
harness's package layout); this file is the harness adapter and nothing else.

Two interface gaps to bridge:
  1. exp_MTS_forecasting calls model(x_enc, x_mark_enc, x_dec, x_mark_dec). The authors' forward
     takes only x -- PatchTST is channel-independent and uses no time-feature embedding, so the
     three extra tensors are genuinely unused rather than merely ignored here.
  2. The authors read several configs that this harness's run.py may not define. Defaults below are
     the authors' own run_longExp.py argparse defaults, so an unset flag reproduces their behaviour
     rather than silently inventing one.

NOT the same model as `PatchTST` in this tree: that is the Time-Series-Library reimplementation,
which uses LayerNorm instead of the authors' BatchNorm and has no residual-attention path. Both are
kept so existing results stay reproducible and the two can be compared directly.
"""
from src.models._patchtst_official import Model as _AuthorModel

# (attr, default) taken from yuqinie98/PatchTST PatchTST_supervised/run_longExp.py.
# attn_dropout is deliberately absent: the authors hardcode it to 0. in the Model signature and
# never read it from configs, so exposing it here would fake a knob the published model lacks.
_AUTHOR_DEFAULTS = (
    ("fc_dropout", 0.05),
    ("head_dropout", 0.0),
    ("individual", 0),
    ("patch_len", 16),
    ("stride", 8),
    ("padding_patch", "end"),
    ("revin", 1),
    ("affine", 0),
    ("subtract_last", 0),
    ("decomposition", 0),
    ("kernel_size", 25),
)


class Model(_AuthorModel):
    def __init__(self, configs, **kwargs):
        for attr, default in _AUTHOR_DEFAULTS:
            if getattr(configs, attr, None) is None:
                setattr(configs, attr, default)
        super().__init__(configs, **kwargs)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        return super().forward(x_enc)
PYEOF
echo "    [ok]   wrote src/models/PatchTSTOfficial.py (harness adapter)"

# ---- register in the model dict and expose the authors' remaining flags on the CLI
"${PYTHON:-python3}" - "$TF_HARNESS" <<'PYEOF'
import re, sys, pathlib
work = pathlib.Path(sys.argv[1])

def patch(rel, subs):
    p = work / rel
    src = orig = p.read_text()
    for old, new, marker, why in subs:
        if marker in src:
            print(f"    [skip] {rel}: {why} (already applied)"); continue
        src2 = re.sub(old, new, src, count=1)
        if src2 == src:
            sys.exit(f"ERROR: {rel}: pattern for '{why}' did not match -- re-derive the patch")
        src = src2
        print(f"    [ok]   {rel}: {why}")
    if src != orig:
        p.write_text(src)

patch("src/exp/exp_basic.py", [
    (r"(from src\.models import [^\n]+)", r"\1, PatchTSTOfficial",
     "PatchTSTOfficial", "import PatchTSTOfficial"),
    (r"(\n(\s*)'PatchTST': PatchTST,)", r"\1\n\2'PatchTSTOfficial': PatchTSTOfficial,",
     "'PatchTSTOfficial':", "register 'PatchTSTOfficial'"),
])

# default=None so the adapter can tell "unset" from "explicitly set", and fall back to the
# authors' own defaults rather than to argparse's.
flags = "".join(
    f"    parser.add_argument('--{n}', type={t}, default=None, help='PatchTST (authors impl): {h}')\n"
    for n, t, h in [
        ("fc_dropout",    "float", "fully-connected dropout"),
        ("individual",    "int",   "individual head per channel"),
        ("padding_patch", "str",   "end pads the final patch"),  # no quotes: help is embedded in a '...' literal
        ("revin",         "int",   "RevIN instance normalisation"),
        ("affine",        "int",   "RevIN affine"),
        ("subtract_last", "int",   "0 subtract mean, 1 subtract last"),
        ("decomposition", "int",   "series decomposition"),
        ("kernel_size",   "int",   "decomposition kernel"),
    ])
patch("run.py", [
    (r"(\n\s*parser\.add_argument\('--head_dropout'[^\n]*\n[^\n]*\n)", r"\1" + flags,
     "'--fc_dropout'", "add authors' PatchTST flags"),
])
PYEOF

echo "==> done. Use MODEL=PatchTSTOfficial (the TSLib port remains available as MODEL=PatchTST)."
