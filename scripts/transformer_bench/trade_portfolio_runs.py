#!/usr/bin/env python3
"""Score completed portfolio-campaign runs: rank IC and the paper's long/short trading return.

    python3 scripts/transformer_bench/trade_portfolio_runs.py            # every completed run
    python3 scripts/transformer_bench/trade_portfolio_runs.py --model ITransformerOfficial

WHAT IT DOES. Each run writes ONE pooled predictions.csv (ticker,date,predicted_RET,expected_RET).
trade_portfolio.py instead wants a directory of <TICKER>_test_predictions.csv holding two POSITIONAL
columns (expected_RET, predicted_RET) in test-file row order, so this splits the pooled file into
that shape and then shells out to the real trader. Nothing here re-implements the strategy.

NO TRUNCATION IS APPLIED, unlike dlinear_trading_prep.py. That script exists because DLinear at
L=20 could not predict the first 19 test days, so EXAMM had to be cut to match. These author-profile
runs use the loader's val-tail lookback borrowing, so every L emits a prediction for every test day
except the last: 250 predictions against 251 price rows, which is exactly the n_price == n_pred + 1
that trade_portfolio.py's Portfolio class requires. Asserted below rather than assumed.

--strategy IS PASSED EXPLICITLY AND MUST STAY THAT WAY. trade_portfolio.py defaults to
daily_long_short_return, the plain unconditional variant. The paper (arXiv 2410.17212, Algorithm 2)
uses daily_hybrid_long_short_return: trade only when all top-L predictions are >0 AND all bottom-S
are <0, else hold. Using the default silently produces roughly half-sized, wrongly-ranked returns.

SINGLE-SEED NUMBERS ARE NOT THE DEPLOYABLE ARTIFACT. This project has repeatedly found per-run test
IC to be dominated by seed noise (per-run spread -0.008..+0.024 on one cohort) while the 10-seed
prediction-mean ensemble is several times the mean run. Everything printed here is per-run, so read
it as a progress check, not as the result.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def daily_ic(df):
    """Mean cross-sectional Spearman IC over days, and its t-stat over the daily series."""
    ics = []
    for _, g in df.groupby("date"):
        if len(g) < 3 or g["predicted_RET"].nunique() < 2:
            continue
        ics.append(g["predicted_RET"].corr(g["expected_RET"], method="spearman"))
    ics = np.asarray([x for x in ics if np.isfinite(x)])
    if ics.size == 0:
        return float("nan"), float("nan"), 0
    t = ics.mean() / (ics.std(ddof=1) / np.sqrt(ics.size)) if ics.size > 1 and ics.std(ddof=1) > 0 else float("nan")
    return float(ics.mean()), float(t), int(ics.size)


def split_pooled(pred_csv, outdir, data_dir):
    """pooled -> <TICKER>_test_predictions.csv (expected_RET,predicted_RET), test-file row order.

    expected_RET IS TAKEN FROM THE TEST CSV, NOT FROM predictions.csv. The pooled file's own
    expected_RET has been through the loader's scaler and back, which leaves ~1e-8 of round-trip
    error. trade_portfolio.py asserts expected_RET == raw RET[t+1] to 1e-10 -- deliberately tight,
    because its purpose is to catch a prediction file paired with the wrong test file, and a
    correct pairing agrees to ~1e-20. Loosening that guard to accommodate our rounding would blunt
    a check worth keeping; taking the ground truth from the price file instead satisfies it exactly
    and is the more correct input anyway.

    Rows are matched ON DATE rather than by position, so a silent off-by-one cannot survive here.
    """
    df = pd.read_csv(pred_csv)
    os.makedirs(outdir, exist_ok=True)
    n = 0
    for tic, g in df.groupby("ticker"):
        g = g.sort_values("date").reset_index(drop=True)
        t = pd.read_csv(os.path.join(data_dir, f"{tic}_test.csv"))
        truth = dict(zip(t["date"].astype(str), t["RET"].astype(float)))
        exp = g["date"].astype(str).map(truth)
        if exp.isna().any():
            raise SystemExit(f"{tic}: {int(exp.isna().sum())} predicted dates absent from {tic}_test.csv")
        pd.DataFrame({"expected_RET": exp.to_numpy(),
                      "predicted_RET": g["predicted_RET"].to_numpy()}).to_csv(
            os.path.join(outdir, f"{tic}_test_predictions.csv"), index=False)
        n += 1
    return n, df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(REPO, "results/transformer_bench/mid_highmid"))
    ap.add_argument("--model", default=None, help="substring filter on model name")
    ap.add_argument("--long", type=int, default=10)
    ap.add_argument("--short", type=int, default=10)
    ap.add_argument("--strategy", default="daily_hybrid_long_short_return")
    args = ap.parse_args()

    runs = [d for d in glob.glob(os.path.join(args.root, "**/predictions.csv"), recursive=True)
            if "/author" in d and "/smoke/" not in d and os.path.exists(
                os.path.join(os.path.dirname(d), ".done"))]
    if args.model:
        runs = [r for r in runs if args.model in r]
    if not runs:
        sys.exit("no completed runs found")

    print(f"{'model':22s} {'cell':22s} {'sd':>9s} {'IC':>8s} {'t':>6s} {'trade%':>8s}  note")
    rows = []
    for pred in sorted(runs):
        d = os.path.dirname(pred)
        t = json.load(open(os.path.join(d, "timing.json")))
        data_dir = os.path.join(REPO, "datasets/walkforward/mid_highmid_price",
                                t["set"], t["cohort"])
        with tempfile.TemporaryDirectory() as tmp:
            n_tic, df = split_pooled(pred, tmp, data_dir)
            sd = df["predicted_RET"].std()
            ic, tstat, ndays = daily_ic(df)
            note = ""
            if n_tic != 50:
                note = f"EXPECTED 50 TICKERS, GOT {n_tic}"
            cmd = [sys.executable, os.path.join(REPO, "scripts/stock_run/trade_portfolio.py"),
                   "--pred-dir", tmp, "--data-dir", data_dir, "--align", "suffix",
                   "--expect-n", "50", "--long", str(args.long), "--short", str(args.short),
                   "--strategy", args.strategy]
            r = subprocess.run(cmd, capture_output=True, text=True)
            ret = float("nan")
            for line in r.stdout.splitlines():
                # trade_portfolio prints the strategy's total return; take the last percentage on
                # a line naming the strategy or the portfolio total.
                if "Portfolio" in line or "portfolio" in line or "return" in line.lower():
                    for tok in line.replace("%", " ").split():
                        try:
                            ret = float(tok)
                        except ValueError:
                            pass
            if r.returncode != 0:
                note = (note + " | " if note else "") + f"trade rc={r.returncode}: {r.stderr.strip().splitlines()[-1][:60] if r.stderr.strip() else '?'}"
        cell = f"{t['set']}/{t['cohort'][7:11]}/s{t['seed']}"
        print(f"  {t['model']:20s} {cell:22s} {sd:9.6f} {ic:+8.4f} {tstat:+6.2f} {ret:8.2f}  {note}")
        rows.append(dict(model=t["model"], cell=cell, sd=sd, ic=ic, t=tstat, ret=ret))

    if rows:
        r = pd.DataFrame(rows)
        print("\nper-model means (single-seed runs -- progress check, not a result):")
        print(r.groupby("model")[["sd", "ic", "ret"]].agg(["count", "mean"]).to_string())


if __name__ == "__main__":
    main()
