#!/usr/bin/env python3
"""Compute the cross-sectional (rank) Information Coefficient from a directory of
per-stock EXAMM prediction CSVs (as written by evaluate_rnn / write_predictions).

For each date, ranks the universe by predicted return and by realized return and
takes their Spearman correlation; reports the time-average (the "IC") plus the
IC information ratio (mean/std * sqrt(n_dates)), the standard forecast-skill
summary. This is the metric the IC training objective optimizes and the number a
walk-forward pilot is judged on.

Stocks are aligned by ROW INDEX -- the aligned walk-forward cohorts share one
calendar per split, so row j is the same date for every stock (verified by
align_cohort.py). Prediction CSVs carry no date column, so index alignment is the
correct join.

Usage:
    # after evaluating the chosen genome on every stock's split:
    python3 scripts/stock_run/eval_cross_sectional_ic.py \
        --pred-dir test_output/ic_test_eval --split test
    # optionally check against a genome's recorded fitness (val):
    python3 scripts/stock_run/eval_cross_sectional_ic.py \
        --pred-dir test_output/ic_valeval --split val --expect-ic 0.022067
"""
import argparse
import csv
import glob
import os
import sys
from collections import defaultdict

import numpy as np
from scipy.stats import rankdata


def load_predictions(pred_dir, split):
    suffix = f"_{split}_predictions.csv"
    files = sorted(glob.glob(os.path.join(pred_dir, f"*{suffix}")))
    if not files:
        sys.exit(f"ERROR: no *{suffix} files in {pred_dir}")

    by_row = defaultdict(dict)  # row_index -> {stock: (pred, realized)}
    n_rows_seen = set()
    for f in files:
        stock = os.path.basename(f)[: -len(suffix)]
        with open(f) as fh:
            rows = list(csv.reader(fh))
        header = rows[0]
        try:
            ei = header.index("expected_RET")
            pi = header.index("predicted_RET")
        except ValueError:
            sys.exit(f"ERROR: {f} lacks expected_RET/predicted_RET columns")
        n_rows_seen.add(len(rows) - 1)
        for j, r in enumerate(rows[1:]):
            by_row[j][stock] = (float(r[pi]), float(r[ei]))

    if len(n_rows_seen) != 1:
        sys.exit(f"ERROR: prediction files have differing row counts {sorted(n_rows_seen)} "
                 f"-- splits are not calendar-aligned; use the *_aligned cohort")
    return files, by_row


def cross_sectional_ic(by_row):
    ics = []
    for j in sorted(by_row):
        d = by_row[j]
        preds = np.array([d[s][0] for s in sorted(d)])
        real = np.array([d[s][1] for s in sorted(d)])
        if preds.std() < 1e-12 or real.std() < 1e-12:
            ics.append(0.0)
            continue
        ics.append(float(np.corrcoef(rankdata(preds), rankdata(real))[0, 1]))
    return np.array(ics)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--split", required=True, choices=["train", "val", "test"])
    ap.add_argument("--expect-ic", type=float, default=None,
                    help="optional: assert mean IC matches this (e.g. a genome's -best_validation_mse)")
    ap.add_argument("--tol", type=float, default=1e-3,
                    help="tolerance for --expect-ic (default 1e-3; CSV rounding limits exactness)")
    args = ap.parse_args()

    files, by_row = load_predictions(args.pred_dir, args.split)
    ics = cross_sectional_ic(by_row)

    n_dates = len(ics)
    mean_ic = ics.mean()
    std_ic = ics.std(ddof=1) if n_dates > 1 else float("nan")
    ir = mean_ic / std_ic * np.sqrt(n_dates) if std_ic > 0 else float("nan")

    print(f"universe        : {len(files)} stocks")
    print(f"dates ({args.split:5s})   : {n_dates}")
    print(f"mean IC         : {mean_ic:+.6f}")
    print(f"IC std / date   : {std_ic:.6f}")
    print(f"IC info ratio   : {ir:+.3f}   (mean/std * sqrt(n_dates))")
    print(f"hit rate (IC>0) : {(ics > 0).mean():.1%}")

    if args.expect_ic is not None:
        diff = abs(mean_ic - args.expect_ic)
        status = "PASS" if diff <= args.tol else "FAIL"
        print(f"\nexpected IC     : {args.expect_ic:+.6f}")
        print(f"abs diff        : {diff:.6f}  (tol {args.tol})  -> {status}")
        if diff > args.tol:
            sys.exit(1)


if __name__ == "__main__":
    main()
