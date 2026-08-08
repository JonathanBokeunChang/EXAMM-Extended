#!/usr/bin/env python3
"""Build a matched trading setup for DLinear vs EXAMM, actual $ returns (not just IC).

THE WRINKLE: DLinear needs a 20-day lookback INSIDE the test file before its first prediction
(no borrowing from val -- see stock_pooled_loader.py, windows never cross a split boundary). So
DLinear has no prediction for the first 19 trading days of each test file, unlike EXAMM (an RNN,
predicts from day 1) or the plain OLS baseline (needs only 1 lookback day). Table 1's headline
EXAMM trading numbers are computed on the FULL test window (test_output/mse_<cohort>/ensemble_test,
day 1 onward) -- comparing DLinear's return on its 19-days-shorter window against that headline
number would not be apples to apples.

This script builds ONE matched window instead: it drops the first 19 rows from EVERY ticker's
test.csv (a "shadow" data-dir), and re-slices EXAMM's own ensembled predictions to the same 19
rows dropped -- this is the EXACT SAME truncation compare_to_examm.py already uses for the IC
comparison (EXAMM_DROP=19), just applied to raw price data too so trade_portfolio.py's Portfolio
class (which requires n_price == n_pred + 1) accepts it. DLinear's 10-seed predictions are
ensembled (mean per ticker/date, same recipe as EXAMM's own eval_ensemble_ic.py) onto that window.

Emits, per cohort:
  results/transformer_bench/<cohort>/_trading_shadow/data/<TICKER>_test.csv        (truncated prices)
  results/transformer_bench/<cohort>/_trading_shadow/EXAMM_ensemble/<TICKER>_test_predictions.csv
  results/transformer_bench/<cohort>/_trading_shadow/DLinear_ensemble/<TICKER>_test_predictions.csv

Then trade both with trade_portfolio.py --data-dir .../data --pred-dir .../{EXAMM,DLinear}_ensemble,
same --window-start/--window-end/--long/--short as the Table 1 headline cells.
"""
from __future__ import annotations

import argparse
import glob
import os

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DROP = 19  # matches compare_to_examm.py:EXAMM_DROP


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--examm-root", required=True)
    a = ap.parse_args()

    data_dir = os.path.join(REPO, "datasets", "walkforward", a.cohort)
    shadow = os.path.join(REPO, "results", "transformer_bench", a.cohort, "_trading_shadow")
    price_dir = os.path.join(shadow, "data")
    examm_dir = os.path.join(shadow, "EXAMM_ensemble")
    dlin_dir = os.path.join(shadow, "DLinear_ensemble")
    for d in (price_dir, examm_dir, dlin_dir):
        os.makedirs(d, exist_ok=True)

    # ---- 1. truncated price files (drop first DROP rows -> n_price = n_pred + 1 for both models)
    test_files = sorted(glob.glob(f"{data_dir}/*_test.csv"))
    tickers = [os.path.basename(f).split("_test.csv")[0] for f in test_files]
    for f, ticker in zip(test_files, tickers):
        df = pd.read_csv(f)
        df.iloc[DROP:].to_csv(os.path.join(price_dir, f"{ticker}_test.csv"), index=False)
    print(f"[{a.cohort}] wrote {len(tickers)} truncated price files -> {price_dir}")

    # ---- 2. EXAMM: slice its own full ensemble_test predictions the same way
    examm_ens = os.path.join(REPO, a.examm_root, "ensemble_test")
    for ticker in tickers:
        f = os.path.join(examm_ens, f"{ticker}_test_predictions.csv")
        df = pd.read_csv(f)
        df.iloc[DROP:][["expected_RET", "predicted_RET"]].to_csv(
            os.path.join(examm_dir, f"{ticker}_test_predictions.csv"), index=False)
    print(f"[{a.cohort}] wrote {len(tickers)} truncated EXAMM ensemble predictions -> {examm_dir}")

    # ---- 3. DLinear: ensemble (mean) the 10 seeds' predictions.csv onto the same window
    dlin_root = os.path.join(REPO, "results", "transformer_bench", a.cohort, "DLinear")
    seed_files = sorted(glob.glob(f"{dlin_root}/seed_*/predictions.csv"))
    if not seed_files:
        raise SystemExit(f"no DLinear seed predictions under {dlin_root}")
    frames = [pd.read_csv(f) for f in seed_files]
    for ticker in tickers:
        # raw RET straight from the source csv (float64) -- DLinear's own "expected_RET" is a
        # float32 inverse-scaling round-trip and differs from this at the ~1e-8 level, which trips
        # trade_portfolio.py's 1e-10 pairing check. The raw csv is the ground truth either way.
        raw = pd.read_csv(os.path.join(data_dir, f"{ticker}_test.csv"), usecols=["date", "RET"])
        raw["date"] = raw["date"].astype(str)
        ret_lookup = dict(zip(raw["date"], raw["RET"]))

        parts = [f[(f.ticker == ticker)].sort_values("date") for f in frames]
        n = {len(p) for p in parts}
        if len(n) != 1:
            raise SystemExit(f"{ticker}: seed row counts differ {n}")
        base = parts[0][["date"]].reset_index(drop=True)
        base["date"] = base["date"].astype(str)
        base["expected_RET"] = base["date"].map(ret_lookup)
        if base["expected_RET"].isna().any():
            raise SystemExit(f"{ticker}: some DLinear dates not found in raw test csv")
        preds = pd.concat([p["predicted_RET"].reset_index(drop=True) for p in parts], axis=1)
        base["predicted_RET"] = preds.mean(axis=1)
        base[["expected_RET", "predicted_RET"]].to_csv(
            os.path.join(dlin_dir, f"{ticker}_test_predictions.csv"), index=False)
    print(f"[{a.cohort}] wrote {len(tickers)} ensembled ({len(seed_files)} seeds) "
          f"DLinear predictions -> {dlin_dir}")


if __name__ == "__main__":
    main()
