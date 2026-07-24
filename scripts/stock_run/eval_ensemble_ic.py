#!/usr/bin/env python3
"""Ensemble per-run prediction CSVs and report the cross-sectional IC.

Given an OUT_ROOT holding run_1/, run_2/, ... (each with an eval_<split>/ dir of
per-stock <ticker>_<split>_predictions.csv written by evaluate_rnn), this:

  1. averages predicted_RET across runs per (stock, date)  -- the ensemble step
     that tames the winner's-curse variance in neuroevolution (see the volatility
     recipe: pooling + ensembling),
  2. computes the daily cross-sectional Spearman IC of the ensemble, plus the IC
     information ratio and hit rate,
  3. optionally writes the ensembled predictions to --emit-dir in the same format
     evaluate_rnn uses, so trade_portfolio.py can trade the ensemble directly.

Stocks are aligned by ROW INDEX (the aligned cohort shares one calendar per split).

Usage:
    python3 scripts/stock_run/eval_ensemble_ic.py \
        --run-root test_output/ic_pearson_cohort_2021_aligned --split test \
        --emit-dir test_output/ic_pearson_cohort_2021_aligned/ensemble_test
"""
import argparse
import csv
import glob
import math
import os
import sys
from collections import defaultdict

# STDLIB ONLY -- no numpy/scipy. This script must run on Anvil compute nodes, where
# the bare `module load gcc/openmpi` python has no scientific stack. The problem is
# tiny (universe x dates ~ 50 x 250), so pure Python costs nothing, and keeping one
# implementation for both machines means local and Anvil numbers are identical by
# construction. Verified to reproduce the numpy/scipy results to <1e-15.


def _rankdata(a):
    """Ranks 1..n with ties averaged -- scipy.stats.rankdata's default 'average'."""
    n = len(a)
    order = sorted(range(n), key=lambda i: a[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and a[order[j + 1]] == a[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0  # 1-based, averaged over the tie block
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def _mean(a):
    return sum(a) / len(a)


def _std(a, ddof=0):
    n = len(a)
    if n - ddof <= 0:
        return 0.0
    m = _mean(a)
    return math.sqrt(sum((x - m) * (x - m) for x in a) / (n - ddof))


def _pearson(x, y):
    """Pearson correlation; 0.0 if either side is constant (degenerate date)."""
    n = len(x)
    mx, my = _mean(x), _mean(y)
    sxy = sxx = syy = 0.0
    for i in range(n):
        dx = x[i] - mx
        dy = y[i] - my
        sxy += dx * dy
        sxx += dx * dx
        syy += dy * dy
    if sxx <= 0.0 or syy <= 0.0:
        return 0.0
    return sxy / math.sqrt(sxx * syy)


def read_pred(path, target_col):
    with open(path) as f:
        rows = list(csv.reader(f))
    h = rows[0]
    ei, pi = h.index(f"expected_{target_col}"), h.index(f"predicted_{target_col}")
    exp = [float(r[ei]) for r in rows[1:]]
    pred = [float(r[pi]) for r in rows[1:]]
    return h, exp, pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", required=True,
                    help="OUT_ROOT containing run_*/eval_<split>/")
    ap.add_argument("--split", required=True, choices=["val", "test"])
    ap.add_argument("--emit-dir", default=None,
                    help="if set, write ensembled <ticker>_<split>_predictions.csv here")
    ap.add_argument("--min-runs", type=int, default=1,
                    help="require at least this many runs contributed per stock")
    ap.add_argument("--target-col", default="RET",
                    help="output parameter name (prediction CSV column suffix); "
                         "RET for the raw/IC arms, RET_CS for the z-score-MSE arm")
    args = ap.parse_args()

    suffix = f"_{args.split}_predictions.csv"
    eval_dirs = sorted(glob.glob(os.path.join(args.run_root, "run_*", f"eval_{args.split}")))
    if not eval_dirs:
        sys.exit(f"ERROR: no run_*/eval_{args.split}/ under {args.run_root} "
                 f"(run eval_ic_run.sh {args.run_root} {args.split} first)")
    print(f"ensembling {len(eval_dirs)} run(s): {[d.split('/')[-2] for d in eval_dirs]}")

    # stock -> list of pred arrays (one per run); expected taken from first run
    preds = defaultdict(list)
    expected = {}
    nrows = {}
    for d in eval_dirs:
        for f in sorted(glob.glob(os.path.join(d, f"*{suffix}"))):
            stock = os.path.basename(f)[: -len(suffix)]
            h, exp, pr = read_pred(f, args.target_col)
            preds[stock].append(pr)
            if stock not in expected:
                expected[stock] = exp
                nrows[stock] = len(exp)
            elif len(exp) != nrows[stock]:
                sys.exit(f"ERROR: {stock} row count differs across runs ({len(exp)} vs {nrows[stock]})")

    stocks = sorted(preds)
    row_counts = set(nrows[s] for s in stocks)
    if len(row_counts) != 1:
        sys.exit(f"ERROR: stocks have differing row counts {sorted(row_counts)} -- not aligned")
    n_dates = row_counts.pop()

    # average across runs per stock (require >= min-runs contributions)
    ens = {}
    for s in stocks:
        if len(preds[s]) < args.min_runs:
            sys.exit(f"ERROR: {s} has only {len(preds[s])} run(s) (< --min-runs {args.min_runs})")
        n_runs_s = len(preds[s])
        ens[s] = [sum(r[j] for r in preds[s]) / n_runs_s for j in range(n_dates)]

    # per-date cross-sectional Spearman IC of the ensemble
    ics = [0.0] * n_dates
    for j in range(n_dates):
        p = [ens[s][j] for s in stocks]
        e = [expected[s][j] for s in stocks]
        if _std(p) < 1e-12 or _std(e) < 1e-12:
            ics[j] = 0.0
        else:
            ics[j] = _pearson(_rankdata(p), _rankdata(e))

    mean_ic = _mean(ics)
    std_ic = _std(ics, ddof=1)
    ir = mean_ic / std_ic * math.sqrt(n_dates) if std_ic > 0 else float("nan")
    print(f"\n=== ENSEMBLE cross-sectional IC ({args.split}) ===")
    print(f"universe      : {len(stocks)} stocks")
    print(f"dates         : {n_dates}")
    print(f"runs ensembled: {len(eval_dirs)}")
    print(f"mean IC       : {mean_ic:+.6f}")
    print(f"IC info ratio : {ir:+.3f}")
    print(f"hit rate      : {sum(1 for x in ics if x > 0) / len(ics):.1%}")

    if args.emit_dir:
        os.makedirs(args.emit_dir, exist_ok=True)
        for s in stocks:
            out = os.path.join(args.emit_dir, f"{s}{suffix}")
            with open(out, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow([f"expected_{args.target_col}", f"predicted_{args.target_col}"])
                for j in range(n_dates):
                    w.writerow([f"{expected[s][j]:.10g}", f"{ens[s][j]:.10g}"])
        print(f"\nwrote ensembled predictions -> {args.emit_dir}  "
              f"(feed to trade_portfolio.py --pred-dir)")


if __name__ == "__main__":
    main()
