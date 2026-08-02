#!/usr/bin/env python3
"""Average a transformer baseline's per-seed predictions into the per-ticker layout
that trade_portfolio.py and eval_ensemble_ic.py already consume.

The transformer runs write one tidy predictions.csv per seed
(ticker,date,predicted_RET,expected_RET), while every EXAMM-side tool expects one
<TICKER>_test_predictions.csv per stock with an expected_RET/predicted_RET pair aligned
by ROW INDEX. This converts between them, averaging predicted_RET across seeds per
(ticker, date) -- the same ensemble step used on the EXAMM side, applied at the
prediction level rather than the metric level.

    python3 scripts/transformer_bench/build_ensemble_preds.py \
        --model-dir results/transformer_bench/cohort_2020_aligned/DeformTime \
        --out-dir   test_output/deformtime_orig/cohort_2020_aligned/ensemble_test

expected_RET must agree across seeds for a given (ticker, date) -- they are the same
labels read from the same files -- so a disagreement means the seeds are not aligned and
is a hard error rather than something to average away.
"""
import argparse
import glob
import os
import shutil
import sys

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True, help="dir holding seed_*/predictions.csv")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--expect-seeds", type=int, default=0, help="hard-fail unless exactly this many")
    ap.add_argument(
        "--data-dir",
        help="source cohort dir. The labels in the seed CSVs are float32 (~1e-8 relative), which "
        "trips trade_portfolio.py's 1e-8 pairing guard even when the alignment is perfectly "
        "correct. Given this, expected_RET is re-read from <data-dir>/<TICKER>_test.csv at full "
        "double precision instead -- the same fix already applied to DLinear. The float32 copy is "
        "still checked against it, so a REAL misalignment is still caught.",
    )
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.model_dir, "seed_*", "predictions.csv")))
    if not files:
        sys.exit(f"no seed_*/predictions.csv under {a.model_dir}")
    if a.expect_seeds and len(files) != a.expect_seeds:
        sys.exit(f"expected {a.expect_seeds} seeds, found {len(files)}")

    frames = [pd.read_csv(f) for f in files]
    key = ["ticker", "date"]
    base = frames[0].sort_values(key).reset_index(drop=True)
    preds = np.empty((len(frames), len(base)))
    for i, d in enumerate(frames):
        d = d.sort_values(key).reset_index(drop=True)
        if not d[key].equals(base[key]):
            sys.exit(f"seed {i} has a different (ticker,date) index than seed 0")
        # Labels are identical inputs, not estimates -- a mismatch is misalignment, not noise.
        if not np.allclose(d["expected_RET"], base["expected_RET"], rtol=0, atol=1e-12):
            sys.exit(f"seed {i} disagrees with seed 0 on expected_RET -- seeds are misaligned")
        preds[i] = d["predicted_RET"].to_numpy()

    base["predicted_RET"] = preds.mean(axis=0)
    shutil.rmtree(a.out_dir, ignore_errors=True)
    os.makedirs(a.out_dir)
    n = 0
    for tic, g in base.groupby("ticker", sort=True):
        g = g.sort_values("date").reset_index(drop=True)
        if a.data_dir:
            src = pd.read_csv(os.path.join(a.data_dir, f"{tic}_test.csv"))
            # Predictions are one-step-ahead over rows 1.. of the test file, so the label for
            # prediction k is RET[k+1]. Verified against the known-good Crossformer ensemble
            # already on disk: this offset reproduces it exactly (0.0e+00), offset 0 does not.
            exact = src["RET"].to_numpy()[1 : 1 + len(g)]
            if len(exact) != len(g):
                sys.exit(f"{tic}: {len(src)} source rows cannot supply {len(g)} labels")
            drift = np.abs(exact - g["expected_RET"].to_numpy()).max()
            if drift > 1e-5:
                sys.exit(f"{tic}: labels differ from source by {drift:.3e} -- real misalignment, not float32")
            g["expected_RET"] = exact
        g[["expected_RET", "predicted_RET"]].to_csv(
            os.path.join(a.out_dir, f"{tic}_test_predictions.csv"), index=False
        )
        n += 1
    print(f"{a.model_dir}: {len(files)} seeds -> {n} tickers x {len(base)//n} days  ->  {a.out_dir}")


if __name__ == "__main__":
    main()
