#!/usr/bin/env python3
"""Annualized Sharpe of Algorithm 2's daily long/short P&L, from a prediction dir.

trade_portfolio.py returns only a total return -- the toolbox realizes P&L when it
clears holdings, so no daily series comes back out of it and Sharpe cannot be derived
from its output. This recomputes the daily book directly from the predictions using
Algorithm 2's own rule (portfolio.py:daily_hybrid_long_short_return):

    rank by predicted return; trade only if ALL L longs are predicted > 0 AND ALL S
    shorts are predicted < 0; otherwise hold yesterday's book unchanged.

Daily P&L is the dollar-neutral spread mean(RET of longs) - mean(RET of shorts), which
is what the equal-quota long and short legs earn per unit of capital. Sharpe is
mean/sd * sqrt(252) with rf = 0.

VALIDATE BEFORE TRUSTING: --self-check reproduces the paper's Table 4 cells for EXAMM,
DLinear and Reversal. If those do not match, the daily convention here does not match
the toolbox's and nothing below should be reported.
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

SUF = "_test_predictions.csv"
HOLD_THROUGH = True   # toggled by --flat-on-no-trade


def load(pred_dir, cohort_dir, year):
    fs = sorted(glob.glob(os.path.join(pred_dir, "*" + SUF)))
    if not fs:
        raise SystemExit(f"no predictions under {pred_dir}")
    P = np.vstack([pd.read_csv(f)["predicted_RET"].to_numpy() for f in fs])
    A = np.vstack([pd.read_csv(f)["expected_RET"].to_numpy() for f in fs])
    tic = os.path.basename(fs[0])[: -len(SUF)]
    dates = pd.read_csv(os.path.join(cohort_dir, f"{tic}_test.csv"), usecols=["date"])["date"].to_numpy()
    dates = dates[1 : P.shape[1] + 1]
    if year:
        k = np.array([str(d).startswith(str(year)) for d in dates])
        P, A, dates = P[:, k], A[:, k], dates[k]
    return P, A, dates


def daily_pnl(P, A, long_n, short_n):
    """One return per day under Algorithm 2, including hold-through days.

    Positions are held in SHARES between gate openings, not rebalanced daily: the
    toolbox only clears and re-enters on a trade day, so position weights DRIFT across
    the (often ~180 of 250) days the gate stays shut. Tracking dollar values rather than
    equal weights matters -- a daily-rebalanced approximation reproduced the published
    2023 Sharpe closely but missed 2022 by up to 0.55, because 2022 holds far longer.

    At entry the book is dollar-neutral: equity E is split E/long_n across the longs and
    E/short_n of notional is shorted, so gross exposure is 2E and net is 0.
    """
    out = np.zeros(P.shape[1])
    book = None
    for j in range(P.shape[1]):
        order = np.argsort(P[:, j])
        longs, shorts = order[-long_n:], order[:short_n]
        traded = P[longs, j].min() > 0 and P[shorts, j].max() < 0
        if traded:
            book = (longs, shorts)
        if book is None:
            out[j] = 0.0
        elif HOLD_THROUGH or traded:
            l, s = book
            out[j] = A[l, j].mean() - A[s, j].mean()
        else:
            out[j] = 0.0      # gate shut -> treat as flat
    return out


def sharpe(r):
    sd = r.std(ddof=1)
    return float("nan") if sd == 0 else float(r.mean() / sd * np.sqrt(252))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir")
    ap.add_argument("--data-dir")
    ap.add_argument("--year")
    ap.add_argument("--long", type=int, default=10)
    ap.add_argument("--short", type=int, default=10)
    ap.add_argument("--self-check", action="store_true")
    ap.add_argument("--flat-on-no-trade", action="store_true")
    a = ap.parse_args()
    global HOLD_THROUGH
    HOLD_THROUGH = not a.flat_on_no_trade

    if a.self_check:
        # published Table 4 (tab:sharpe) cells, L/S 5/10/15/20
        REF = {
            ("EXAMM", "2022"): [0.90, 0.98, 1.36, 1.49],
            ("DLinear", "2022"): [0.55, 0.41, -0.51, -0.64],
            ("Reversal", "2022"): [-0.62, 0.56, 0.43, -0.12],
            ("EXAMM", "2023"): [2.04, 2.32, 2.39, 2.07],
            ("DLinear", "2023"): [0.99, 0.64, 0.22, -0.41],
            ("Reversal", "2023"): [2.53, 1.04, 1.25, -0.10],
        }
        SRC = {
            "EXAMM": "test_output/mse_{c}/ensemble_test",
            "DLinear": "results/transformer_bench/{c}/DLinear/ensemble_test",
            "Reversal": "results/transformer_bench/{c}/Reversal/eval_test",
        }
        print(f"{'model':<10}{'year':>6}" + "".join(f"{n:>16}" for n in (5, 10, 15, 20)))
        worst = 0.0
        for (m, y), ref in REF.items():
            c = "cohort_2020_aligned" if y == "2022" else "cohort_2021_aligned"
            P, A, _ = load(SRC[m].format(c=c), f"datasets/walkforward/{c}", y)
            got = [sharpe(daily_pnl(P, A, n, n)) for n in (5, 10, 15, 20)]
            worst = max(worst, max(abs(g - r) for g, r in zip(got, ref)))
            print(f"{m:<10}{y:>6}" + "".join(f"{g:>8.2f}/{r:<7.2f}" for g, r in zip(got, ref)))
        print(f"\n  computed/published -- max abs difference {worst:.2f}")
        return

    P, A, _ = load(a.pred_dir, a.data_dir, a.year)
    print(f"{sharpe(daily_pnl(P, A, a.long, a.short)):+.2f}")


if __name__ == "__main__":
    main()
