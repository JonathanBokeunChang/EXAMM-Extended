#!/usr/bin/env python3
"""Pooled PAIRED test: is EXA-GP-MIN's long/short spread reliably above EXAMM's?

Per-cell t-statistics on the daily spread never exceed 1.9 for either arm, so no single
cell shows skill on its own. But the arms are evaluated on the SAME days over the SAME
universe, so the day-by-day DIFFERENCE (spread_gp - spread_ex) cancels the common market
factor and is far better powered than comparing two noisy series side by side.

Three views, deliberately, because they disagree about what n is:
  - pooled daily, iid t: optimistic, treats ~2,250 days as independent.
  - pooled daily, Newey-West at 5 lags: same series, allowing autocorrelation.
  - cell-level t over the 9 panel-years: the honest unit, since days within a
    panel-year share a model, a universe and a market regime.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

REPO = "/Users/jonathanchang/EXAMM-Extended/.claude/worktrees/exagp-stock"
sys.path.insert(0, os.path.join(REPO, "scripts/stock_run"))
from score_forecasting import data_dir, from_per_ticker  # noqa: E402

L = S = 10


def daily_spread(root, panel, cohort, year):
    df = from_per_ticker(os.path.join(root, "ensemble_test"), data_dir(panel, cohort))
    df = df[df["date"].astype(str).str[:4] == str(year)]
    P = df.pivot(index="date", columns="ticker", values="predicted_RET").sort_index()
    R = df.pivot(index="date", columns="ticker", values="expected_RET").reindex_like(P)
    out = {}
    for d in P.index:
        p, r = P.loc[d].to_numpy(float), R.loc[d].to_numpy(float)
        ok = np.isfinite(p) & np.isfinite(r)
        p, r = p[ok], r[ok]
        if len(p) < L + S:
            continue
        o = np.argsort(-p)
        out[d] = r[o[:L]].mean() - r[o[-S:]].mean()
    return pd.Series(out)


def nw_t(x, lags=5):
    x = np.asarray(x, float)
    n, mu = len(x), x.mean()
    e = x - mu
    var = (e @ e) / n
    for k in range(1, lags + 1):
        var += 2 * (1 - k / (lags + 1)) * ((e[k:] @ e[:-k]) / n)
    return mu / np.sqrt(var / n)


cells = json.load(open(os.path.join(REPO, "results/exagp_cells.json")))
diffs, per_cell = [], []
for c in cells:
    panel, cohort, year = c["panel"], c["cohort"], c["trade_year"]
    mid = f"{panel}_" if panel != "dsA" else ""
    a = daily_spread(os.path.join(REPO, "test_output", f"exagpmin_{mid}{cohort}_g500"),
                     panel, cohort, year)
    b = daily_spread(os.path.join(REPO, "test_output", f"examm_{mid}{cohort}_g500"),
                     panel, cohort, year)
    common = a.index.intersection(b.index)
    d = a[common] - b[common]
    diffs.append(d)
    per_cell.append(d.mean() * 1e4)
    print(f"{panel + '/' + cohort[7:11]:<14} n={len(d):>4}  mean diff {d.mean() * 1e4:+7.2f} bps/day")

D = pd.concat(diffs)
n = len(D)
print(f"\npooled daily differences: n={n}  mean {D.mean() * 1e4:+.2f} bps/day")
print(f"  t (iid, optimistic)    {D.mean() / (D.std(ddof=1) / np.sqrt(n)):+.2f}")
print(f"  t (Newey-West, 5 lags) {nw_t(D.to_numpy()):+.2f}")

pc = np.array(per_cell)
print(f"\ncell-level (n={len(pc)} panel-years, the honest unit)")
print(f"  mean {pc.mean():+.2f} bps/day   sd {pc.std(ddof=1):.2f}   "
      f"t {pc.mean() / (pc.std(ddof=1) / np.sqrt(len(pc))):+.2f}")
print(f"  cells favouring EXA-GP-MIN: {(pc > 0).sum()}/{len(pc)}")
