#!/usr/bin/env python3
"""Does EXA-GP-MIN rank the EXTREMES better while ranking the full cross-section no better?

    python3 scripts/stock_run/tail_vs_full_ic.py

WHY ASK. Across nine paired cells EXA-GP-MIN leads EXAMM 6/9 on realised trading return
but only 4/9 on rank IC over all 50 names -- chance. Several return margins are large
(+3.45 vs -16.54 on set3/2020) while the IC on the same cell favours EXAMM. The obvious
candidate explanation is that the two statistics measure different things: the strategy
only ever holds the predicted top ten and bottom ten, whereas rank IC scores the whole
cross-section including the 30 middle names the book never touches.

THREE STATISTICS, computed per day and averaged, for each arm and cell:
  full_ic   -- Spearman over all names. What the paper currently reports.
  tail_ic   -- Spearman restricted to the union of the predicted top-L and bottom-S. Whether
               the ordering is right AMONG the names actually traded.
  spread    -- mean realised return of the predicted top-L minus that of the predicted
               bottom-S, in basis points per day. This is, up to costs and compounding,
               what the long/short book actually earns, so it is the quantity the trading
               result is made of rather than a proxy for it.

A t-statistic on the daily spread series is reported because a spread mean without one
invites reading a 20-point annual difference as skill when the daily series is noise;
Newey-West would be tighter but the plain t already answers the question at hand.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "scripts/stock_run"))
from score_forecasting import data_dir, from_per_ticker  # noqa: E402

L = S = 10
TRADE_YEAR = {"cohort_2020_aligned": 2022, "cohort_2021_aligned": 2023,
              "cohort_2022_aligned": 2024}


def spearman(a, b):
    if len(a) < 4:
        return np.nan
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    ra, rb = ra - ra.mean(), rb - rb.mean()
    den = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / den) if den > 0 else np.nan


def cell_stats(root, panel, cohort, year):
    ens = os.path.join(root, "ensemble_test")
    if not os.path.isdir(ens):
        return None
    df = from_per_ticker(ens, data_dir(panel, cohort))
    df = df[df["date"].astype(str).str[:4] == str(year)]
    P = df.pivot(index="date", columns="ticker", values="predicted_RET").sort_index()
    R = df.pivot(index="date", columns="ticker", values="expected_RET").reindex_like(P)

    full, tail, spread = [], [], []
    for d in P.index:
        p, r = P.loc[d].to_numpy(float), R.loc[d].to_numpy(float)
        ok = np.isfinite(p) & np.isfinite(r)
        p, r = p[ok], r[ok]
        if len(p) < L + S:
            continue
        full.append(spearman(p, r))
        o = np.argsort(-p)
        top, bot = o[:L], o[-S:]
        idx = np.concatenate([top, bot])
        tail.append(spearman(p[idx], r[idx]))
        spread.append(r[top].mean() - r[bot].mean())

    spread = np.array(spread, float)
    t = (spread.mean() / (spread.std(ddof=1) / np.sqrt(len(spread)))) if len(spread) > 1 else np.nan
    return dict(days=len(spread), full_ic=float(np.nanmean(full)),
                tail_ic=float(np.nanmean(tail)),
                spread_bps=float(spread.mean() * 1e4), t=float(t))


def main():
    cells = json.load(open(os.path.join(REPO, "results/exagp_cells.json")))
    print(f"{'cell':<14} {'arm':<11} {'days':>5} {'full_ic':>9} {'tail_ic':>9} "
          f"{'spread_bps':>11} {'t':>7}")
    agg = {"exagpmin": [], "examm": []}
    for c in cells:
        panel, cohort, year = c["panel"], c["cohort"], c["trade_year"]
        for arm in ("exagpmin", "examm"):
            mid = f"{panel}_" if panel != "dsA" else ""
            root = os.path.join(REPO, "test_output", f"{arm}_{mid}{cohort}_g500")
            st = cell_stats(root, panel, cohort, year)
            if st is None:
                continue
            agg[arm].append(st)
            print(f"{panel+'/'+cohort[7:11]:<14} {arm:<11} {st['days']:>5} "
                  f"{st['full_ic']:>+9.4f} {st['tail_ic']:>+9.4f} "
                  f"{st['spread_bps']:>+11.2f} {st['t']:>+7.2f}")

    print()
    for arm in ("exagpmin", "examm"):
        a = agg[arm]
        if not a:
            continue
        print(f"{arm:<11} mean over {len(a)} cells: "
              f"full_ic {np.mean([x['full_ic'] for x in a]):+.4f}  "
              f"tail_ic {np.mean([x['tail_ic'] for x in a]):+.4f}  "
              f"spread {np.mean([x['spread_bps'] for x in a]):+.2f} bps/day")
    wins_full = sum(1 for i in range(len(agg["exagpmin"]))
                    if agg["exagpmin"][i]["full_ic"] > agg["examm"][i]["full_ic"])
    wins_tail = sum(1 for i in range(len(agg["exagpmin"]))
                    if agg["exagpmin"][i]["tail_ic"] > agg["examm"][i]["tail_ic"])
    wins_spr = sum(1 for i in range(len(agg["exagpmin"]))
                   if agg["exagpmin"][i]["spread_bps"] > agg["examm"][i]["spread_bps"])
    n = len(agg["exagpmin"])
    print(f"\nEXA-GP-MIN beats EXAMM:  full_ic {wins_full}/{n}   "
          f"tail_ic {wins_tail}/{n}   spread {wins_spr}/{n}")


if __name__ == "__main__":
    main()
