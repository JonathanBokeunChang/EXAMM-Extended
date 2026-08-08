#!/usr/bin/env python3
"""Pure short-term reversal baseline: predicted_RET[t+1] = -RET[t]. Zero parameters, zero
training -- just the sign-flipped most-recent return. Motivated by DLinear turning out to be a
TREND-following model (rides each stock's 20-day moving-average trend and flips sides when the
trend reverses -- see walkforward-linear-vs-examm.md memory). Reversal is the opposite mechanism
(bet AGAINST yesterday's move) and is a real, previously-measured effect on this project's other
return venues (see linear-baseline-finding.md, csi300-dataset-and-ceiling.md). This checks how
much of EXAMM's/DLinear's edge could be explained by reversal alone, before trusting a trend
story.

Emits the same two layouts the other baselines use:
    results/transformer_bench/<cohort>/Reversal/seed_0/predictions.csv        (IC, truncated -19)
    results/transformer_bench/<cohort>/Reversal/eval_test/<TICKER>_test_predictions.csv  (trading)

Usage:
    python3 scripts/transformer_bench/reversal_baseline.py --cohort cohort_2020_aligned
    python3 scripts/transformer_bench/reversal_baseline.py --cohort cohort_2021_aligned
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXAMM_DROP = 19  # matches compare_to_examm.py:EXAMM_DROP


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True)
    a = ap.parse_args()

    data_dir = os.path.join(REPO, "datasets", "walkforward", a.cohort)
    test_files = sorted(glob.glob(f"{data_dir}/*_test.csv"))
    if not test_files:
        raise SystemExit(f"no *_test.csv under {data_dir}")

    seed_dir = os.path.join(REPO, "results", "transformer_bench", a.cohort, "Reversal", "seed_0")
    eval_dir = os.path.join(REPO, "results", "transformer_bench", a.cohort, "Reversal", "eval_test")
    os.makedirs(seed_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)

    rows = []
    for f in test_files:
        ticker = os.path.basename(f).split("_test.csv")[0]
        df = pd.read_csv(f, usecols=["date", "RET"])
        ret = df["RET"].to_numpy(float)
        dates = df["date"].to_numpy()
        pred = -ret[:-1]     # predicted_RET[t+1] = -RET[t]
        exp = ret[1:]
        tgt_dates = dates[1:]

        pd.DataFrame({"expected_RET": exp, "predicted_RET": pred}).to_csv(
            os.path.join(eval_dir, f"{ticker}_test_predictions.csv"), index=False)
        rows.append(pd.DataFrame({"ticker": ticker, "date": tgt_dates[EXAMM_DROP:],
                                   "predicted_RET": pred[EXAMM_DROP:], "expected_RET": exp[EXAMM_DROP:]}))

    out = pd.concat(rows, ignore_index=True)
    out.to_csv(os.path.join(seed_dir, "predictions.csv"), index=False)
    print(f"[{a.cohort}] wrote {len(test_files)} full eval_test files -> {eval_dir}")
    print(f"[{a.cohort}] wrote {len(out)} truncated IC-comparison rows -> {seed_dir}/predictions.csv")


if __name__ == "__main__":
    main()
