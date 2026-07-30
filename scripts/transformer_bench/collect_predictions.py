#!/usr/bin/env python3
"""Turn one transformer run's pred.npy into a tidy, raw-unit predictions.csv keyed by (ticker, date).

WHY THIS IS NOT OPTIONAL
------------------------
The harness saves pred.npy/true.npy in NORMALIZED space -- run.py's --inverse is an
action='store_true' defaulting to False and our job does not pass it. EXAMM's cached predictions, by
contrast, are in raw RET units (verified: its `expected_RET` column equals the raw csv RET exactly).
Comparing the two directly would compare different units.

Rank IC happens to be invariant here, because our scaler is one pooled affine map per column and
Spearman only sees order. MSE, RMSE and MAE are NOT invariant, and the paper reports them alongside
the EvoStar tables. So un-scaling is done once, here, rather than trusted to survive downstream.

The un-scaling is EXAMM's own transform inverted: raw = scaled * (max - avg) + avg, with (avg, max)
read from the scaler.json the loader wrote at construction time.

Row i of test_index.csv describes prediction i. That alignment holds only because data_factory.py
sets shuffle_flag=False and batch_size=1 for the test split; if either changes, this join is wrong
and silently so.

Usage:
    python3 collect_predictions.py --run-dir results/transformer_bench/cohort_2020_aligned/DeformTime/seed_3
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    idx_path = os.path.join(a.run_dir, "test_index.csv")
    sc_path = os.path.join(a.run_dir, "scaler.json")
    for p in (idx_path, sc_path):
        if not os.path.exists(p):
            sys.exit(f"ERROR: missing {p}")

    preds = sorted(glob.glob(os.path.join(a.run_dir, "**", "pred.npy"), recursive=True))
    if len(preds) != 1:
        sys.exit(f"ERROR: expected exactly one pred.npy under {a.run_dir}, found {len(preds)}: "
                 f"{preds}\n(more than one means results from another array task leaked in)")
    pred = np.load(preds[0]).reshape(-1)
    true_p = os.path.join(os.path.dirname(preds[0]), "true.npy")
    true = np.load(true_p).reshape(-1) if os.path.exists(true_p) else None

    idx = pd.read_csv(idx_path)
    if len(pred) != len(idx):
        sys.exit(f"ERROR: {len(pred)} predictions vs {len(idx)} index rows -- these do not "
                 f"correspond. Most likely drop_last discarded samples, i.e. the test loader is no "
                 f"longer batch_size=1/shuffle=False.")

    sc = json.load(open(sc_path))
    avg, denom = sc["target_avg"], sc["target_denom"]
    df = pd.DataFrame({"ticker": idx["ticker"], "date": idx["target_date"],
                       "predicted_RET": pred * denom + avg})
    if true is not None:
        df["expected_RET"] = true * denom + avg
        # the scaler is affine, so a mismatch here means a units or ordering bug, not rounding
        r = float(np.corrcoef(df.predicted_RET, df.expected_RET)[0, 1])
        print(f"[check] corr(pred, true) = {r:+.4f}   RMSE = "
              f"{np.sqrt(((df.predicted_RET - df.expected_RET) ** 2).mean()):.6g}")

    n_dates, n_tick = df.date.nunique(), df.ticker.nunique()
    print(f"[shape] {len(df)} rows = {n_tick} tickers x {n_dates} dates "
          f"({'balanced' if n_tick * n_dates == len(df) else 'UNBALANCED -- check'})")
    print(f"[range] predicted_RET [{df.predicted_RET.min():.6f}, {df.predicted_RET.max():.6f}]")

    out = a.out or os.path.join(a.run_dir, "predictions.csv")
    df.to_csv(out, index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
