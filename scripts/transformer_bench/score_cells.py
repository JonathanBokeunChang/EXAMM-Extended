#!/usr/bin/env python3
"""Per-CELL ensemble scores (rank IC + Algorithm 2 trading return) as JSON, with a cache.

    python3 scripts/transformer_bench/score_cells.py [--root DIR] [--json]

CELL, NOT RUN. trade_portfolio_runs.py scores each seed separately, which is the right granularity
for debugging a single run but the wrong one for reading a campaign: per-seed trading return on this
study spans -2.03% to +14.62% for two seeds of the SAME cell, so a per-seed number tells you about
the seed, not the model. Everything here is computed on the seed-mean ensemble, which is what the
paper reports and what the tables are built from.

WHY A CACHE. This backs a live monitor that repolls every ~20 s, and trading shells out to
trade_portfolio.py once per cell over 50 tickers. Recomputing 36 cells per poll costs more than the
refresh interval, so results are keyed on the cell's SORTED SEED LIST: a poll that finds no new seed
is a dict lookup, and the moment a seed lands that one cell -- and only that cell -- recomputes.
The key holds the seed paths themselves rather than a count, so a rerun that replaces a seed in
place still invalidates.

IC AND TRADING ARE NOT REDUNDANT and neither one substitutes for the other. A model can hold a
respectable IC while its trading return sits at zero, because Algorithm 2 only opens a position on
days when all top-L predictions are >0 and all bottom-S are <0; a collapsed model whose predictions
barely move produces a nearly constant series that still ranks fine day to day. That is exactly how
Crossformer failed here, so `sd` and `gate` are reported alongside both -- they are the fields that
tell the two situations apart.
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trade_portfolio_runs import daily_ic, split_pooled  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(REPO, "datasets/walkforward/mid_highmid_price")
TRADER = os.path.join(REPO, "scripts/stock_run/trade_portfolio.py")


def ensemble(files, data_dir):
    """Seed-mean prediction per (ticker, date), with truth taken from the price files.

    Truth comes from <TICKER>_test.csv rather than the predictions' own expected_RET column for the
    reason split_pooled documents: the pooled value has been through the loader's scaler and back
    and carries ~1e-8 of round-trip error, which the trader's 1e-10 pairing check rejects.
    """
    ens = (pd.concat([pd.read_csv(f).set_index(["ticker", "date"]).predicted_RET for f in files],
                     axis=1).mean(axis=1).rename("predicted_RET").reset_index())
    out = []
    for tic, g in ens.groupby("ticker"):
        t = pd.read_csv(os.path.join(data_dir, f"{tic}_test.csv"))
        truth = dict(zip(t["date"].astype(str), t["RET"].astype(float)))
        g = g.sort_values("date").copy()
        g["expected_RET"] = g["date"].astype(str).map(truth)
        if g["expected_RET"].isna().any():
            raise SystemExit(f"{tic}: predicted dates missing from {tic}_test.csv")
        out.append(g)
    return pd.concat(out, ignore_index=True)


def gate_days(df, long=10, short=10):
    """Days Algorithm 2 actually trades: all top-L predictions >0 and all bottom-S <0."""
    n = 0
    for _, g in df.groupby("date"):
        p = g["predicted_RET"].sort_values()
        if len(p) >= long + short and p.iloc[-long:].min() > 0 and p.iloc[:short].max() < 0:
            n += 1
    return n


def sharpe(split_dir, data_dir, long=10, short=10):
    """Annualised Sharpe from the REAL strategy's daily equity curve.

    Reported GROSS only. trade_portfolio_daily_curve.py refuses to emit a curve under --use-tc:
    its self-check (instrument the real method, rerun the unpatched one, require agreement to
    1e-6) fails at +12.7248% vs +12.7257%. A net Sharpe would need a SECOND implementation of the
    strategy, which is exactly what that script exists to avoid. tab:hp-grid-sharpe already
    reports gross for the same reason.
    """
    r = subprocess.run(
        [sys.executable, os.path.join(REPO, "scripts/stock_run/trade_portfolio_daily_curve.py"),
         "--pred-dir", split_dir, "--data-dir", data_dir,
         "--long", str(long), "--short", str(short)], capture_output=True, text=True)
    for line in r.stdout.splitlines():
        if "Sharpe" in line:
            try:
                return float(line.split(":")[1].split()[0])
            except (IndexError, ValueError):
                pass
    return float("nan")


def trade(df, data_dir, long=10, short=10, strategy="daily_hybrid_long_short_return", tc=False):
    """Algorithm 2 return for the ensemble. --strategy is explicit; the trader's default is the
    unconditional variant, which silently yields roughly half-sized, wrongly-ranked returns.

    tc=True charges the per-share CRSP TRAN_COST on every trade. It relies on the short-selling fix
    in the local Financial_toolbox checkout (Stock.short_stock returning the realised credit rather
    than the nominal quota); without that fix enabling costs could raise a return, which is not a
    thing a friction cost can do. Net must always come out <= gross -- asserted by the caller.
    """
    with tempfile.TemporaryDirectory() as tmp:
        pooled = os.path.join(tmp, "ens.csv")
        df.to_csv(pooled, index=False)
        split = os.path.join(tmp, "split")
        n_tic, _ = split_pooled(pooled, split, data_dir)
        r = subprocess.run(
            [sys.executable, TRADER, "--pred-dir", split, "--data-dir", data_dir,
             "--align", "suffix", "--expect-n", "50", "--long", str(long), "--short", str(short),
             "--strategy", strategy] + (["--use-tc"] if tc else []), capture_output=True, text=True)
        ret = float("nan")
        for line in r.stdout.splitlines():
            if "portfolio" in line.lower() or "return" in line.lower():
                for tok in line.replace("%", " ").split():
                    try:
                        ret = float(tok)
                    except ValueError:
                        pass
        err = "" if r.returncode == 0 else f"rc={r.returncode}"
        sh = sharpe(split, data_dir, long, short) if not tc else float("nan")
        return ret, n_tic, err, sh


def score(root, cache_path):
    try:
        cache = json.load(open(cache_path))
    except Exception:
        cache = {}
    cells = {}
    for f in glob.glob(os.path.join(root, "**/predictions.csv"), recursive=True):
        d = os.path.dirname(f)
        if "/author" not in f or "/smoke/" in f or not os.path.exists(os.path.join(d, ".done")):
            continue
        try:
            t = json.load(open(os.path.join(d, "timing.json")))
        except Exception:
            continue
        cells.setdefault((t["model"], t.get("set"), t["cohort"]), []).append(f)

    out, fresh = [], 0
    for (model, st, coh), files in cells.items():
        files = sorted(files)
        key = "|".join(["v4", model, str(st), coh] + [os.path.relpath(x, root) for x in files])
        if key in cache:
            out.append(cache[key])
            continue
        data_dir = os.path.join(DATA, st, coh)
        try:
            df = ensemble(files, data_dir)
            ic, tstat, ndays = daily_ic(df)
            # Point-accuracy metrics on the SAME seed-mean ensemble as everything else, so the
            # MSE/MAE grid and the IC/return grid describe one object rather than two.
            err = df["predicted_RET"].to_numpy(float) - df["expected_RET"].to_numpy(float)
            mse = float(np.mean(err ** 2)); mae = float(np.mean(np.abs(err)))
            ret, n_tic, err, sh = trade(df, data_dir)
            net, _, err2, _ = trade(df, data_dir, tc=True)
            if np.isfinite(ret) and np.isfinite(net) and net > ret + 1e-6:
                err = (err + " | " if err else "") + f"NET>GROSS ({net:.2f}>{ret:.2f})"
            rec = {"model": model, "set": st, "yr": int(coh[7:11]) + 2, "n": len(files),
                   "ic": round(float(ic), 5), "t": round(float(tstat), 2),
                   "mse": mse, "mae": mae,
                   "sharpe": None if not np.isfinite(sh) else round(float(sh), 3),
                   "sd": round(float(df["predicted_RET"].std()), 7),
                   "net": None if not np.isfinite(net) else round(float(net), 2),
                   "ret": None if not np.isfinite(ret) else round(float(ret), 2),
                   "gate": gate_days(df), "days": int(df["date"].nunique()),
                   "tickers": n_tic, "err": err}
        except Exception as e:
            rec = {"model": model, "set": st, "yr": int(coh[7:11]) + 2, "n": len(files),
                   "ic": None, "t": None, "sd": None, "ret": None, "net": None,
                   "mse": None, "mae": None, "sharpe": None, "gate": 0, "days": 0,
                   "tickers": 0, "err": str(e)[:80]}
        cache[key] = rec
        out.append(rec)
        fresh += 1

    if fresh:
        tmp = cache_path + ".tmp"
        json.dump(cache, open(tmp, "w"))
        os.replace(tmp, cache_path)   # atomic: concurrent pollers never read a half-written cache
    return sorted(out, key=lambda x: (x["model"], x["set"] or "", x["yr"])), fresh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(REPO, "results/transformer_bench/mid_highmid"))
    ap.add_argument("--cache", default=None)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--loop", type=int, default=0, metavar="SEC",
                    help="run forever, rescoring every SEC seconds (daemon that keeps the cache "
                         "warm for the monitor)")
    a = ap.parse_args()
    cache = a.cache or os.path.join(os.path.dirname(a.root.rstrip("/")), ".cell_scores.json")
    if a.loop:
        # The monitor reads the cache and never computes, so a cold cell must not stall a poll --
        # this loop absorbs that cost off the polling path. Errors are swallowed deliberately: a
        # half-written predictions.csv from a run finishing mid-scan should cost one cycle, not
        # the daemon.
        import time
        while True:
            try:
                rows, fresh = score(a.root, cache)
                if fresh:
                    print(f"[{time.strftime('%H:%M:%S')}] scored {fresh} new, {len(rows)} total",
                          flush=True)
            except Exception as e:
                print(f"[{time.strftime('%H:%M:%S')}] {type(e).__name__}: {str(e)[:100]}",
                      flush=True)
            time.sleep(a.loop)
    rows, fresh = score(a.root, cache)
    if a.json:
        print(json.dumps(rows))
        return
    print(f"{'model':22s} {'cell':12s} {'n':>2s} {'IC':>8s} {'t':>6s} "
          f"{'sd':>9s} {'ret%':>8s} {'gate':>9s}")
    for r in rows:
        ic = f"{r['ic']:+8.4f}" if r["ic"] is not None else "       -"
        t = f"{r['t']:+6.2f}" if r["t"] is not None else "     -"
        sd = f"{r['sd']:9.6f}" if r["sd"] is not None else "        -"
        ret = f"{r['ret']:8.2f}" if r["ret"] is not None else "       -"
        print(f"  {r['model'][:20]:20s} {r['set']}/{r['yr']:<6d} {r['n']:>2d} {ic} {t} {sd} "
              f"{ret} {r['gate']:>4d}/{r['days']:<4d} {r['err']}")
    ok = [r for r in rows if r["ic"] is not None]
    if ok:
        print(f"\n{len(ok)} cells · {sum(1 for r in ok if r['ic'] > 0)} IC>0 · "
              f"mean IC {np.mean([r['ic'] for r in ok]):+.4f} · "
              f"{fresh} recomputed, {len(rows) - fresh} cached")


if __name__ == "__main__":
    main()
