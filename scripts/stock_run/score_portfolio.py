#!/usr/bin/env python3
"""Portfolio panel for the paper: net return, gross Sharpe, max drawdown and TURNOVER.

    python3 scripts/stock_run/score_portfolio.py          # table to stderr, JSON to --out

Companion to score_forecasting.py. The two scripts import the SAME loader (score_forecasting.load)
so the forecasting and portfolio tables describe one object per cell rather than two; if the
ensembles ever diverge, both panels move together instead of silently disagreeing.

TURNOVER IS A NEW COLUMN AND IT IS LOAD-BEARING. Algorithm 2 re-enters only on days when every
top-10 prediction is >0 and every bottom-10 is <0, and it does NOT flatten otherwise --
Financial_toolbox portfolio.py:451 updates the held book only inside `if do_trade`, so on a
gate-shut day the previous positions are carried. Gate frequency is therefore the strategy's
holding period, and it varies enormously by model (EXAMM ~18% of days, the transformers 52-94%).
Reporting net return without it lets a turnover advantage read as a forecasting advantage, which
is the single most likely misreading of this paper.

WHICH EXAMM TREE: --replicates, defaulting to `published`, exactly as in score_forecasting.py.
Keep the two scripts in the same mode or the forecasting and portfolio tables will describe
different objects.

SHARPE AND MDD ARE GROSS WHILE THE RETURN COLUMN IS NET. Forced, not chosen:
trade_portfolio_daily_curve.py refuses to emit a curve under --use-tc because its self-check
(instrument the real Portfolio method, rerun the unpatched one, require agreement to 1e-6) fails
at +12.7248% vs +12.7257%. A net Sharpe would need a SECOND implementation of the strategy, which
that script exists to prevent.

DATASET A IS DATE-WINDOWED THROUGH THE TRADER, NOT BY SLICING. cohort_2020's split spans 2022 and
2023; returns COMPOUND so they cannot be sliced after the fact, and handing the trader a 250-row
prefix of a 501-row price file fails outright under --align suffix (which trims to the TRAILING
window). --window-start/--window-end restrict trading dates after alignment, which is correct.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "scripts/transformer_bench"))
sys.path.insert(0, os.path.join(REPO, "scripts/stock_run"))
from score_forecasting import MODELS, cells, data_dir, load, reps_for  # noqa: E402
from trade_portfolio_runs import split_pooled  # noqa: E402

TRADER = os.path.join(REPO, "scripts/stock_run/trade_portfolio.py")
CURVE = os.path.join(REPO, "scripts/stock_run/trade_portfolio_daily_curve.py")


def gate_days(df, long=10, short=10):
    n = 0
    for _, g in df.groupby("date"):
        p = g["predicted_RET"].sort_values()
        if len(p) >= long + short and p.iloc[-long:].min() > 0 and p.iloc[:short].max() < 0:
            n += 1
    return n


def _ret(split, dd, window, tc):
    r = subprocess.run(
        [sys.executable, TRADER, "--pred-dir", split, "--data-dir", dd, "--align", "suffix",
         "--expect-n", "50", "--long", "10", "--short", "10",
         "--strategy", "daily_hybrid_long_short_return"]
        + (["--window-start", window[0], "--window-end", window[1]] if window else [])
        + (["--use-tc"] if tc else []), capture_output=True, text=True)
    for line in r.stdout.splitlines():
        if "strategy return" in line:
            for tok in line.replace("%", " ").split():
                try:
                    return float(tok)
                except ValueError:
                    pass
    return float("nan")


def _curve(split, dd, window):
    """One invocation yields Sharpe, MDD and the traded-day count, all gross."""
    r = subprocess.run(
        [sys.executable, CURVE, "--pred-dir", split, "--data-dir", dd, "--long", "10",
         "--short", "10"]
        + (["--window-start", window[0], "--window-end", window[1]] if window else []),
        capture_output=True, text=True)
    sh = mdd = float("nan")
    for line in r.stdout.splitlines():
        if "Sharpe" in line:
            try:
                sh = float(line.split(":")[1].split()[0])
            except (IndexError, ValueError):
                pass
        elif "Max drawdown" in line:
            try:
                mdd = float(line.split(":")[1].strip().rstrip("%"))
            except (IndexError, ValueError):
                pass
    return sh, mdd


def score_one(df, dd, window):
    with tempfile.TemporaryDirectory() as tmp:
        pooled = os.path.join(tmp, "ens.csv")
        df.to_csv(pooled, index=False)
        split = os.path.join(tmp, "split")
        split_pooled(pooled, split, dd)
        net = _ret(split, dd, window, tc=True)
        gross = _ret(split, dd, window, tc=False)
        sh, mdd = _curve(split, dd, window)
    # Costs cannot raise a return; a violation means the short-credit fix is missing.
    if np.isfinite(net) and np.isfinite(gross) and net > gross + 1e-6:
        raise ValueError(f"NET>GROSS ({net:.2f}>{gross:.2f}) -- check Financial_toolbox short fix")
    nd = df["date"].nunique()
    g = gate_days(df)
    return {"net": net, "gross": gross, "sharpe": sh, "mdd": mdd,
            "gate": g, "days": nd, "turnover": 100.0 * g / nd if nd else float("nan")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replicates", choices=["published", "original", "average"],
                    default="published")
    ap.add_argument("--out", default=os.path.join(REPO, "results/portfolio_cells.json"))
    args = ap.parse_args()

    recs = []
    for panel, yr, pk, cohort in cells():
        dd = data_dir(pk, cohort)
        for model, _ in MODELS:
            reps = reps_for(pk, args.replicates) if model == "EXAMM" else [""]
            got, err = [], ""
            for rep in reps:
                try:
                    df = load(model, pk, cohort, rep)
                    # Window only when the split is longer than the single trade year.
                    window = ((f"{yr}-01-01", f"{yr}-12-31")
                              if df["date"].astype(str).str[:4].nunique() > 1 else None)
                    got.append(score_one(df, dd, window))
                except Exception as exc:                                   # noqa: BLE001
                    err = f"{type(exc).__name__}: {exc}"[:90]
            rec = {"panel": panel, "yr": yr, "model": model, "n_rep": len(got)}
            if got:
                for k in got[0]:
                    rec[k] = float(np.mean([g[k] for g in got]))
            else:
                rec["err"] = err
            recs.append(rec)
            print(f"{panel:10s} {yr} {model:10s} rep={rec['n_rep']} "
                  + (f"net={rec.get('net', float('nan')):+7.2f} "
                     f"sharpe={rec.get('sharpe', float('nan')):+6.2f} "
                     f"mdd={rec.get('mdd', float('nan')):+7.2f} "
                     f"turn={rec.get('turnover', float('nan')):5.1f}%" if got else rec.get("err", "")),
                  file=sys.stderr, flush=True)
            json.dump(recs, open(args.out, "w"), indent=1)

    print(f"\nwrote {args.out} ({len(recs)} rows)", file=sys.stderr)


if __name__ == "__main__":
    main()
