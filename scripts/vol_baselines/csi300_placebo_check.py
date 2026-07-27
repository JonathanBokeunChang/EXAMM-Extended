#!/usr/bin/env python3
"""Placebo (permutation) test: does the pipeline manufacture signal from nothing?

Shuffles the TRAINING label within each date -- features, dates, splits and the evaluation set are
left completely untouched, so the only thing destroyed is the true feature<->label pairing. A sound
pipeline must then score IC ~ 0. Anything materially non-zero means predictability is leaking in
structurally (index misalignment, a look-ahead feature, an eval-set join bug) and would appear no
matter what the data contained.

Shuffling WITHIN date (not globally) is deliberate: it preserves each date's cross-sectional label
distribution and the panel's calendar structure, so it isolates the feature->label link rather than
also destroying the time-series structure, which would make the test trivially pass.

What this test CANNOT do: validate the input data. A feed that is systematically cleaner or more
autocorrelated than a real trading feed would produce genuine-looking predictability that this test
passes happily. That question needs cross-source replication -- see csi300_replication_report.py.

Usage:
    python3 scripts/vol_baselines/csi300_placebo_check.py --data datasets/csi300_master_replica_invdata
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from csi300_master_replica_gate import fit_ridge, load, restrict_to_eval_index  # noqa: E402
from ic_stats import DEFAULT_HAC_LAG, report  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=f"{REPO}/datasets/csi300_master_replica_invdata")
    ap.add_argument("--target", default="LABEL_CSRANK")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hac-lag", type=int, default=DEFAULT_HAC_LAG)
    ap.add_argument("--tol", type=float, default=0.01,
                     help="|IC| above this fails the test")
    a = ap.parse_args()

    tr, va, te = (load(a.data, s) for s in ("train", "val", "test"))
    te = restrict_to_eval_index(te, a.data)
    drop = {"date", "instrument", "segment_id", "LABEL", "LABEL_CSRANK"}
    feats = [c for c in tr.columns if c not in drop]

    print(f"=== PLACEBO: shuffle {a.target} WITHIN each date on TRAIN, then refit ===")
    print(f"    train {len(tr):,} rows | eval {len(te):,} rows | {len(feats)} features\n")

    rng = np.random.default_rng(a.seed)
    tr_shuf = tr.copy()
    tr_shuf[a.target] = (tr_shuf.groupby("date")[a.target]
                         .transform(lambda s: rng.permutation(s.to_numpy())))
    # sanity: the shuffle must actually have changed the pairing
    same = float((tr_shuf[a.target].to_numpy() == tr[a.target].to_numpy()).mean())
    print(f"    rows whose label happened to stay put: {same:.2%} (must be small)\n")

    alpha, va_ic, _, ics = fit_ridge(tr_shuf, va, te, feats, a.target)
    print(f"   ridge alpha={alpha:.0f} chosen on val (val IC {va_ic:+.4f})")
    s = report("PLACEBO (shuffled train label)", ics, a.hac_lag)

    ok = abs(s["ic"]) < a.tol and s["ci95_lo"] < 0 < s["ci95_hi"]
    print(f"\n   expected: IC ~ 0 with a 95% interval spanning zero")
    print(f"   observed: IC {s['ic']:+.4f}, 95% [{s['ci95_lo']:+.4f}, {s['ci95_hi']:+.4f}]")
    print(f"\n   {'PASS -- no structural leakage detected' if ok else 'FAIL -- pipeline manufactures signal; investigate before trusting ANY result'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
