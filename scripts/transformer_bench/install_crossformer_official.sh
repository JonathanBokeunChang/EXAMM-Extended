#!/bin/bash
# Vendor the Crossformer AUTHORS' OWN implementation into the benchmark harness, as a model named
# `CrossformerOfficial`, alongside (never replacing) the harness's existing `Crossformer`.
#
#   bash scripts/transformer_bench/install_crossformer_official.sh [TF_HARNESS_DIR]
#
# WHY: src/models/Crossformer.py in external/DeformTime is the Time-Series-Library REIMPLEMENTATION
# (recognisable by its "Paper link: https://openreview.net/..." docstring, thuml's house style), not
# the code from Thinklab-SJTU/Crossformer. Same situation as PatchTST -- see
# install_patchtst_official.sh, which this mirrors.
#
# WHY A NEW NAME: every Crossformer row currently in the paper came from the TSLib implementation.
# Overwriting it would make those rows unreproducible; adding a second model keeps them intact and
# lets the two be compared directly on identical data, seeds and protocol.
#
# The five upstream files are used VERBATIM apart from their `from cross_models.` imports, which are
# repointed at the package this installs them into. Pre-rewrite sha256s are asserted, so an upstream
# change or truncated download fails here rather than silently training a different model.
#
# TWO INTERFACE GAPS the adapter bridges (both wider than PatchTST's):
#   1. Crossformer's class takes EXPLICIT constructor arguments, not a configs object.
#   2. Its forward takes x_seq only, while the harness calls model(x, x_mark, dec, dec_mark).
# Note also that the class's own signature defaults (d_model=512, d_ff=1024, n_heads=8, win_size=4)
# are NOT what the authors run -- main_crossformer.py's argparse overrides them with 256/512/4/2.
# The adapter therefore takes everything from configs, and the caller supplies the argparse values.
#
# Idempotent: safe to re-run.
set -euo pipefail

SHA=c10c8eadb153d1dd9798250967747ca3ebb81383   # 2023-12-01, Thinklab-SJTU/Crossformer
BASE="https://raw.githubusercontent.com/Thinklab-SJTU/Crossformer/$SHA"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TF_HARNESS="${1:-${TF_HARNESS:-$REPO/external/DeformTime}}"
[ -d "$TF_HARNESS/src/models" ] || { echo "ERROR: $TF_HARNESS is not the harness (no src/models)" >&2; exit 1; }
PKG="$TF_HARNESS/src/models/_crossformer_official"

read -r -d '' EXPECT <<'EOF' || true
deee243b2ae877084320dce49e691c8860c56f0cbda269c6f0cdb50848f6c708  cross_models/cross_former.py
be2fbf13317c27186563b68ff03326719a5a5db900b896889f46d570605587c2  cross_models/cross_encoder.py
3c1d476df29d7a520d2ed3e8a06ec8e7186e96edda0ba74d6b943c8253fbdd40  cross_models/cross_decoder.py
39256d8f37e1acb3e52d645887316af89136d5910d4224c697c0cc38896f429b  cross_models/attn.py
6b78484268eb67f8fcf135537c3d61982c903992cc3f8f44fb47a46ec47c83fa  cross_models/cross_embed.py
EOF

sha_of() { shasum -a 256 "$1" 2>/dev/null | awk '{print $1}' || sha256sum "$1" | awk '{print $1}'; }

TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
echo "==> fetching authors' Crossformer @ ${SHA:0:8}"
while read -r want rel; do
  [ -n "${rel:-}" ] || continue
  curl -fsSL "$BASE/$rel" -o "$TMP/$(basename "$rel")"
  got=$(sha_of "$TMP/$(basename "$rel")")
  [ "$got" = "$want" ] || {
    echo "ERROR: checksum mismatch for $rel" >&2
    echo "       expected $want" >&2
    echo "       got      $got" >&2
    exit 1; }
  echo "    [ok]   $rel  sha256 verified"
done <<< "$EXPECT"

mkdir -p "$PKG"
: > "$PKG/__init__.py"
for f in cross_former cross_encoder cross_decoder attn cross_embed; do
  sed -e 's|^from cross_models\.|from src.models._crossformer_official.|' "$TMP/$f.py" > "$PKG/$f.py"
done
echo "    [ok]   installed 5 files into src/models/_crossformer_official/"

cat > "$TF_HARNESS/src/models/CrossformerOfficial.py" <<'PYEOF'
"""Crossformer, the AUTHORS' implementation (github.com/Thinklab-SJTU/Crossformer @ c10c8ead).

Installed by scripts/transformer_bench/install_crossformer_official.sh. The authors' five source
files live in _crossformer_official/ exactly as published (only their `from cross_models.` imports
are repointed); this file is the harness adapter and nothing else.

NOT the same model as `Crossformer` in this tree, which is the Time-Series-Library
reimplementation bundled with the DeformTime repo. Both are kept so existing results stay
reproducible and the two can be compared directly.

Defaults below are main_crossformer.py's ARGPARSE defaults, which are what the authors actually
run -- deliberately not the Crossformer class's own signature defaults (d_model=512, d_ff=1024,
n_heads=8, win_size=4), which the argparse layer overrides and which never execute.
"""
import torch
import torch.nn as nn

from src.models._crossformer_official.cross_former import Crossformer as _AuthorCrossformer

_ARGPARSE_DEFAULTS = (
    ("seg_len", 6),
    ("win_size", 2),
    ("factor", 10),
    ("d_model", 256),
    ("d_ff", 512),
    ("n_heads", 4),
    ("e_layers", 3),
    ("dropout", 0.2),
)


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        g = lambda a: (getattr(configs, a, None) if getattr(configs, a, None) is not None
                       else dict(_ARGPARSE_DEFAULTS)[a])
        self.model = _AuthorCrossformer(
            data_dim=configs.enc_in,
            in_len=configs.seq_len,
            out_len=configs.pred_len,
            seg_len=g("seg_len"),
            win_size=g("win_size"),
            factor=g("factor"),
            d_model=g("d_model"),
            d_ff=g("d_ff"),
            n_heads=g("n_heads"),
            e_layers=g("e_layers"),
            dropout=g("dropout"),
            baseline=bool(getattr(configs, "baseline", False)),
            # Stored on the module but never read by any of the authors' code, so its value is
            # irrelevant; passing CPU avoids a hard cuda:0 reference on a CPU-only box.
            device=torch.device("cpu"),
        )

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        # Crossformer is a pure sequence-to-sequence model over the raw channels: it takes no
        # time-feature embedding, so the three extra tensors are genuinely unused.
        return self.model(x_enc)
PYEOF
echo "    [ok]   wrote src/models/CrossformerOfficial.py (harness adapter)"

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
    (r"(from src\.models import [^\n]+)", r"\1, CrossformerOfficial",
     "CrossformerOfficial", "import CrossformerOfficial"),
    (r"(\n(\s*)'Crossformer': Crossformer,)", r"\1\n\2'CrossformerOfficial': CrossformerOfficial,",
     "'CrossformerOfficial':", "register 'CrossformerOfficial'"),
])

# --baseline is the only authors' flag this harness's run.py lacks. seg_len/win_size/factor/d_model/
# d_ff/n_heads/e_layers/dropout all already exist.
patch("run.py", [
    (r"(\n\s*parser\.add_argument\('--win_size'[^\n]*\n)",
     r"\1    parser.add_argument('--baseline', type=int, default=0,\n"
     r"                        help='Crossformer (authors impl): add the input mean as a prediction baseline')\n",
     "'--baseline'", "add --baseline"),
])
PYEOF

echo "==> done. Use MODEL=CrossformerOfficial (the TSLib port remains available as MODEL=Crossformer)."
