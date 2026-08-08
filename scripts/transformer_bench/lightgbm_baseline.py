#!/usr/bin/env python3
"""Pooled LightGBM baseline, matched EXACTLY to EXAMM's own inputs -- same 6 features
(RET, VOL_CHANGE, BA_SPREAD, ILLIQUIDITY, sprtrn, TURNOVER) -> next-day RET, same pooled-50-stock
setup, same walk-forward split as scripts/stock_run/anvil_ic.sb. A gradient-boosted-tree baseline
is the standard strong tabular reference point in this literature (Qlib/MASTER etc. all compare
against LightGBM) and this project hasn't tried a non-linear, non-neural model yet -- everything
so far has been neuroevolution (EXAMM), linear (OLS/Reversal), or a linear-decomposition net
(DLinear). No windowing needed (single-day lookback, like OLS/Reversal), so -- like DLinear after
the stock_pooled_loader.py fix -- it covers the FULL test year with zero truncation, matching
EXAMM's own row range exactly (examm-drop 0 in compare_to_examm.py).

10-seed ensemble (seed + light feature/bagging subsampling so seeds actually differ, same
ensembling recipe used throughout this project), early-stopped on validation MSE.

Emits:
    results/transformer_bench/<cohort>/LightGBM/seed_<k>/predictions.csv   (per-seed, for IC)
    results/transformer_bench/<cohort>/LightGBM/ensemble_test/<TICKER>_test_predictions.csv
        (10-seed-averaged, for trade_portfolio.py)

Usage:
    python3 scripts/transformer_bench/lightgbm_baseline.py --cohort cohort_2020_aligned
    python3 scripts/transformer_bench/lightgbm_baseline.py --cohort cohort_2021_aligned
"""
from __future__ import annotations

import argparse
import glob
import os

import lightgbm as lgb
import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FEATURES = ["RET", "VOL_CHANGE", "BA_SPREAD", "ILLIQUIDITY", "sprtrn", "TURNOVER"]
TARGET = "RET"
N_SEEDS = 10


def make_xy(path):
    df = pd.read_csv(path, usecols=["date"] + FEATURES)
    x = df[FEATURES].to_numpy(float)[:-1]
    y = df[TARGET].to_numpy(float)[1:]
    dates = df["date"].to_numpy()[1:]
    return x, y, dates


def pool(files):
    xs, ys = [], []
    for f in files:
        x, y, _ = make_xy(f)
        xs.append(x)
        ys.append(y)
    return np.concatenate(xs), np.concatenate(ys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True)
    a = ap.parse_args()

    data_dir = os.path.join(REPO, "datasets", "walkforward", a.cohort)
    train_files = sorted(glob.glob(f"{data_dir}/*_train.csv"))
    val_files = sorted(glob.glob(f"{data_dir}/*_val.csv"))
    test_files = sorted(glob.glob(f"{data_dir}/*_test.csv"))
    tickers = [os.path.basename(f).split("_test.csv")[0] for f in test_files]

    x_tr, y_tr = pool(train_files)
    x_va, y_va = pool(val_files)
    print(f"[{a.cohort}] pooled train {len(y_tr)} rows, val {len(y_va)} rows, "
          f"{len(train_files)} stocks")

    all_ens_preds = {t: [] for t in tickers}
    for seed in range(1, N_SEEDS + 1):
        params = dict(
            objective="regression", metric="l2", verbosity=-1,
            num_leaves=15, max_depth=4, learning_rate=0.05,
            feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
            min_data_in_leaf=200, seed=seed, bagging_seed=seed, feature_fraction_seed=seed,
        )
        booster = lgb.train(
            params, lgb.Dataset(x_tr, label=y_tr), num_boost_round=500,
            valid_sets=[lgb.Dataset(x_va, label=y_va)],
            callbacks=[lgb.early_stopping(30, verbose=False)],
        )

        seed_dir = os.path.join(REPO, "results", "transformer_bench", a.cohort, "LightGBM", f"seed_{seed}")
        os.makedirs(seed_dir, exist_ok=True)
        rows = []
        for f, ticker in zip(test_files, tickers):
            x, y, dates = make_xy(f)
            pred = booster.predict(x, num_iteration=booster.best_iteration)
            all_ens_preds[ticker].append(pred)
            rows.append(pd.DataFrame({"ticker": ticker, "date": dates,
                                       "predicted_RET": pred, "expected_RET": y}))
        pd.concat(rows, ignore_index=True).to_csv(os.path.join(seed_dir, "predictions.csv"), index=False)
        print(f"  seed {seed:2d}: best_iter={booster.best_iteration:4d}  "
              f"val_l2={booster.best_score['valid_0']['l2']:.6f}")

    ens_dir = os.path.join(REPO, "results", "transformer_bench", a.cohort, "LightGBM", "ensemble_test")
    os.makedirs(ens_dir, exist_ok=True)
    for f, ticker in zip(test_files, tickers):
        _, y, _ = make_xy(f)
        ens_pred = np.mean(all_ens_preds[ticker], axis=0)
        pd.DataFrame({"expected_RET": y, "predicted_RET": ens_pred}).to_csv(
            os.path.join(ens_dir, f"{ticker}_test_predictions.csv"), index=False)
    print(f"[{a.cohort}] wrote {N_SEEDS}-seed ensemble -> {ens_dir}")


if __name__ == "__main__":
    main()
