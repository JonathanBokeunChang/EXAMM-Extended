#!/usr/bin/env python3
"""Fan EXAMM's ONE wide combined-prediction file into 50 per-ticker files the scorers accept.

    python3 scripts/stock_run/split_combined_predictions.py \
        --pred test_output/mse_combined_cohort_2021_aligned/run_1/eval_test/combined_predictors_test_predictions.csv \
        --data-dir datasets/walkforward/cohort_2021_aligned \
        --out      test_output/mse_combined_cohort_2021_aligned/run_1/eval_test_split

WHY THIS IS NEEDED. RNN::write_predictions (rnn/rnn.cxx) emits one row per timestep with one column
pair per OUTPUT NODE, so combined produces a single ~400-column file: 300 inputs, then 50
expected_<TICKER>_RET and 50 predicted_<TICKER>_RET. Every scorer in this repo wants the opposite
shape -- trade_portfolio.py globs <TICKER>_test_predictions.csv and reads literal `expected_RET` /
`predicted_RET` columns, and eval_ensemble_ic.py does the same. Splitting here means combined reuses
the IDENTICAL scoring path as pooled and individual, which is the only reason the three rows of
tab:pooling-ablation are comparable at all. Nothing in either scorer changes.

THE LEADING '#'. EXAMM writes the first header cell as '#<name>'. It is stripped, matching what
evaluate_combined.sh already does.

THE ALIGNMENT ASSERT IS LOAD-BEARING, NOT DEFENSIVE. trade_portfolio.py requires
expected_RET == raw RET[t+1] to 1e-10 -- a deliberately tight check whose purpose is to catch a
prediction file paired with the wrong test file. Combined's row count is len(test)-1 because
time_offset 1 consumes the first row, so an off-by-one here would still produce plausible-looking
per-ticker files that silently score the wrong day. We verify against the per-ticker test CSV
directly rather than trusting the wide panel's construction.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="wide *_test_predictions.csv from evaluate_rnn")
    ap.add_argument("--data-dir", required=True, help="per-ticker {TICKER}_test.csv dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--tol", type=float, default=1e-10)
    a = ap.parse_args()

    df = pd.read_csv(a.pred)
    df.columns = [c.lstrip("#") for c in df.columns]

    tickers = sorted({c[len("expected_"):-len("_RET")] for c in df.columns
                      if c.startswith("expected_") and c.endswith("_RET")})
    if len(tickers) != 50:
        sys.exit(f"expected 50 tickers in {a.pred}, found {len(tickers)}")

    os.makedirs(a.out, exist_ok=True)
    worst, worst_t = 0.0, None
    for t in tickers:
        e, p = f"expected_{t}_RET", f"predicted_{t}_RET"
        if e not in df.columns or p not in df.columns:
            sys.exit(f"{t}: missing {e} or {p}")

        src = os.path.join(a.data_dir, f"{t}_{a.split}.csv")
        if not os.path.exists(src):
            sys.exit(f"{t}: no {src} to verify against")
        raw = pd.read_csv(src)["RET"].to_numpy(float)

        # time_offset 1: prediction row i is the forecast OF raw[i+1]. The trader wants exactly
        # n_price == n_pred + 1, which is what this produces.
        if len(df) != len(raw) - 1:
            sys.exit(f"{t}: {len(df)} prediction rows against {len(raw)} price rows "
                     f"(need n_price == n_pred + 1)")
        d = float(np.abs(df[e].to_numpy(float) - raw[1:]).max())
        if d > worst:
            worst, worst_t = d, t
        if d > a.tol:
            sys.exit(f"{t}: expected_{t}_RET does not match RET[t+1] (max diff {d:.3e} > {a.tol:.0e}). "
                     f"The wide panel and {src} are not the same calendar.")

        pd.DataFrame({"expected_RET": df[e].to_numpy(float),
                      "predicted_RET": df[p].to_numpy(float)}).to_csv(
            os.path.join(a.out, f"{t}_test_predictions.csv"), index=False)

    print(f"  wrote {len(tickers)} files to {a.out}")
    print(f"  rows/ticker {len(df)}  |  max |expected - RET[t+1]| = {worst:.3e} ({worst_t}) "
          f"<= {a.tol:.0e}")


if __name__ == "__main__":
    main()
