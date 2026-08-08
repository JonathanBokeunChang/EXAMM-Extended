#!/usr/bin/env python3
"""Pooled OLS baseline on the walk-forward cohorts, matched EXACTLY to EXAMM's own inputs.

WHY THIS EXISTS
----------------
EXAMM's genomes here are tiny (23-131 weights, Table 5 of results_tables.md). The open question
is whether that small size alone explains its behaviour (i.e. any small model would do about as
well on this near-noise venue) or whether the neuroevolution search is finding real structure a
linear model can't. This script isolates that: it fits ONE pooled linear model (RET[t+1] ~ 1 +
RET[t] + VOL_CHANGE[t] + BA_SPREAD[t] + ILLIQUIDITY[t] + sprtrn[t] + TURNOVER[t]) on the SAME
inputs, SAME pooled-across-50-stocks setup, SAME time_offset=1 mechanism, and SAME train/test
split that scripts/stock_run/anvil_ic.sb used for the mse_cohort_2020/2021_aligned runs. 7
parameters (6 weights + intercept) vs EXAMM's 23-131.

This is NOT a DLinear-style windowed model (that needs the external DeformTime harness + GPU,
not set up locally). It is the more direct question: same features, does the search buy anything.

Emits predictions in the exact directory layout compare_to_examm.py expects for a "transformer"
model, so the existing (already-audited) identical-row-gated HAC comparison can be reused as-is:

    results/transformer_bench/<cohort>/Linear/seed_0/predictions.csv

Usage:
    python3 scripts/transformer_bench/linear_baseline.py --cohort cohort_2020_aligned
    python3 scripts/transformer_bench/linear_baseline.py --cohort cohort_2021_aligned
    # then:
    python3 scripts/transformer_bench/compare_to_examm.py --cohort cohort_2020_aligned \\
        --examm-root test_output/mse_cohort_2020_aligned \\
        --tf-root results/transformer_bench/cohort_2020_aligned --year 2022
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FEATURES = ["RET", "VOL_CHANGE", "BA_SPREAD", "ILLIQUIDITY", "sprtrn", "TURNOVER"]
TARGET = "RET"
EXAMM_DROP = 19  # matches compare_to_examm.py's SEQ_LEN-1, so row sets line up exactly


def make_xy(path, feat_mean, feat_std):
    """One stock's file -> (X[t] standardized, y[t+1], date[t+1]), t=0..n-2."""
    df = pd.read_csv(path, usecols=["date"] + FEATURES)
    x = df[FEATURES].to_numpy(float)[:-1]
    y = df[TARGET].to_numpy(float)[1:]
    dates = df["date"].to_numpy()[1:]
    if feat_std is not None:
        x = (x - feat_mean) / feat_std
    return x, y, dates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--model-name", default="Linear")
    a = ap.parse_args()

    data_dir = os.path.join(REPO, "datasets", "walkforward", a.cohort)
    train_files = sorted(glob.glob(f"{data_dir}/*_train.csv"))
    test_files = sorted(glob.glob(f"{data_dir}/*_test.csv"))
    if not train_files:
        raise SystemExit(f"no *_train.csv under {data_dir}")

    # ---- pool train rows across all 50 stocks, standardize features on TRAIN stats only
    raw_x, raw_y = [], []
    for f in train_files:
        df = pd.read_csv(f, usecols=FEATURES)
        raw_x.append(df[FEATURES].to_numpy(float)[:-1])
        raw_y.append(df[TARGET].to_numpy(float)[1:])
    raw_x = np.concatenate(raw_x)
    raw_y = np.concatenate(raw_y)
    feat_mean = raw_x.mean(axis=0)
    feat_std = raw_x.std(axis=0)
    feat_std[feat_std == 0] = 1.0

    xs = (raw_x - feat_mean) / feat_std
    design = np.column_stack([np.ones(len(xs)), xs])  # intercept + 6 features = 7 params
    coef, *_ = np.linalg.lstsq(design, raw_y, rcond=None)
    print(f"[{a.cohort}] fit on {len(raw_y)} pooled train rows x {len(train_files)} stocks, "
          f"{len(coef)} parameters (intercept + {len(FEATURES)} features)")
    print(f"  intercept={coef[0]:+.6f}  " + "  ".join(f"{n}={c:+.5f}" for n, c in zip(FEATURES, coef[1:])))

    # ---- predict on test files, dropping the first EXAMM_DROP rows to match EXAMM's
    #      truncated row set exactly (compare_to_examm.py:EXAMM_DROP)
    out_dir = os.path.join(REPO, "results", "transformer_bench", a.cohort, a.model_name, "seed_0")
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for f in test_files:
        ticker = os.path.basename(f).split("_test.csv")[0]
        x, y, dates = make_xy(f, feat_mean, feat_std)
        pred = np.column_stack([np.ones(len(x)), x]) @ coef
        rows.append(pd.DataFrame({
            "ticker": ticker, "date": dates[EXAMM_DROP:],
            "predicted_RET": pred[EXAMM_DROP:], "expected_RET": y[EXAMM_DROP:],
        }))
    out = pd.concat(rows, ignore_index=True)
    out_path = os.path.join(out_dir, "predictions.csv")
    out.to_csv(out_path, index=False)
    print(f"  wrote {len(out)} predictions ({len(test_files)} stocks) -> {out_path}")

    # ---- FULL (untruncated) per-ticker predictions, in EXAMM's eval_test/*_test_predictions.csv
    #      layout, for trade_portfolio.py (which needs n_price == n_pred + 1, no drop).
    eval_dir = os.path.join(REPO, "results", "transformer_bench", a.cohort, a.model_name, "eval_test")
    os.makedirs(eval_dir, exist_ok=True)
    for f in test_files:
        ticker = os.path.basename(f).split("_test.csv")[0]
        x, y, dates = make_xy(f, feat_mean, feat_std)
        pred = np.column_stack([np.ones(len(x)), x]) @ coef
        pd.DataFrame({"expected_RET": y, "predicted_RET": pred}).to_csv(
            os.path.join(eval_dir, f"{ticker}_test_predictions.csv"), index=False)
    print(f"  wrote {len(test_files)} full per-ticker prediction files -> {eval_dir}")


if __name__ == "__main__":
    main()
