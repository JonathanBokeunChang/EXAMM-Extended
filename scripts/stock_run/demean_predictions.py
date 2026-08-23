#!/usr/bin/env python3
"""Cross-sectionally demean an ensemble's predictions, per day, for the trading gate.

    python3 scripts/stock_run/demean_predictions.py \
        --in-dir  test_output/<arm>/ensemble_test \
        --out-dir test_output/<arm>/ensemble_test_demeaned

WHY THIS IS NEEDED, AND WHY IT IS NOT CHEATING. Algorithm 2 trades at time t only if ALL
L candidate longs are predicted positive AND ALL S candidate shorts are predicted
negative. A forecaster whose cross-section sits entirely above zero can therefore never
open the gate: measured on set1/cohort_2022, EXA-GP-MIN predicted positive on 100% of
stock-days and the gate opened 0 times in 251 days, reporting exactly +0.00% return.
That reads as flat performance and is actually "never traded" -- a broken evaluation
rather than a weak signal.

The level of a cross-sectional forecast carries no information the strategy uses. It
ranks stocks and takes a dollar-neutral book; a constant added to every stock on a given
day changes no ranking and no position. Subtracting the daily cross-sectional mean is
therefore a NO-OP for rank IC and for any rank-based selection, and it only affects the
sign test that the gate happens to be phrased in terms of. It uses no future information:
the mean is taken across stocks WITHIN a single day, never across time.

What it must not be allowed to become is a silent default. An arm that only trades after
demeaning is an arm whose raw output is unusable by the strategy as published, and both
numbers belong in any writeup.
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--col", default="predicted_RET")
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.in_dir, "*_predictions.csv")))
    if not files:
        raise SystemExit(f"no *_predictions.csv in {a.in_dir}")

    frames = {os.path.basename(f): pd.read_csv(f) for f in files}
    lens = {len(v) for v in frames.values()}
    if len(lens) != 1:
        raise SystemExit(f"ragged prediction lengths {sorted(lens)} -- cohort not aligned, "
                         "so row index does not identify a shared date")

    # Stocks are aligned by ROW INDEX: an aligned cohort shares one calendar per split,
    # which is the same assumption eval_ensemble_ic.py makes.
    M = np.column_stack([v[a.col].to_numpy(float) for v in frames.values()])
    daily_mean = np.nanmean(M, axis=1, keepdims=True)
    M_d = M - daily_mean

    os.makedirs(a.out_dir, exist_ok=True)
    for i, (name, df) in enumerate(frames.items()):
        out = df.copy()
        out[a.col] = M_d[:, i]
        out.to_csv(os.path.join(a.out_dir, name), index=False)

    frac_pos_before = float(np.nanmean(M > 0))
    frac_pos_after = float(np.nanmean(M_d > 0))
    print(f"wrote {len(frames)} files -> {a.out_dir}")
    print(f"  rows/file        : {lens.pop()}")
    print(f"  frac predictions > 0 : {frac_pos_before:.3f} -> {frac_pos_after:.3f}")
    print(f"  daily mean removed   : mean {daily_mean.mean():.3e}, sd {daily_mean.std():.3e}")


if __name__ == "__main__":
    main()
